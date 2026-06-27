from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.make_ct_csd_llada import (
    build_sequence,
    extract_target_layer_tokens,
    filter_response_hidden_states,
    load_harmful_data,
    load_refusals,
    resolve_path,
    set_seed,
)


@dataclass(frozen=True)
class MILBag:
    hidden: torch.Tensor
    label: float


class LinearMILProbe(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(int(input_dim), 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden).squeeze(-1)


def top_q_pool_logits(logits: torch.Tensor, top_q_ratio: float) -> torch.Tensor:
    if logits.numel() == 0:
        raise ValueError("Cannot pool an empty bag.")
    if top_q_ratio <= 0.0 or top_q_ratio > 1.0:
        raise ValueError("top_q_ratio must be in (0, 1].")
    flat = logits.reshape(-1)
    k = max(1, min(int(flat.numel()), math.ceil(float(top_q_ratio) * int(flat.numel()))))
    return torch.topk(flat, k=k).values.mean()


def compute_bag_logit(
    probe: LinearMILProbe,
    hidden: torch.Tensor,
    top_q_ratio: float,
) -> torch.Tensor:
    return top_q_pool_logits(probe(hidden), top_q_ratio)


def compute_bag_loss(
    probe: LinearMILProbe,
    hidden: torch.Tensor,
    label: float,
    top_q_ratio: float,
) -> torch.Tensor:
    bag_logit = compute_bag_logit(probe, hidden, top_q_ratio)
    target = torch.tensor(float(label), device=bag_logit.device, dtype=bag_logit.dtype)
    return F.binary_cross_entropy_with_logits(bag_logit, target)


def build_probe_state(
    probe: LinearMILProbe,
    args: argparse.Namespace,
    metrics: dict[str, float],
    input_dim: int,
) -> dict:
    return {
        "format": "mil_token_probe_v1",
        "model_family": "llada",
        "target_layer": int(args.target_layer),
        "input_dim": int(input_dim),
        "top_q_ratio": float(args.top_q_ratio),
        "state_dict": {key: value.detach().cpu() for key, value in probe.state_dict().items()},
        "config": {
            "model_path": str(args.model_path),
            "harmful_json": str(args.harmful_json),
            "refusals_txt": str(args.refusals_txt),
            "max_response_len": int(args.max_response_len),
            "max_total_len": int(args.max_total_len),
            "seed": int(args.seed),
        },
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def _build_response_hidden(
    model,
    tokenizer,
    prompt: str,
    response: str,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor | None:
    if not prompt or not response:
        return None
    ids, response_start, response_ids = build_sequence(tokenizer, prompt, response, args.max_response_len)
    if len(ids) > args.max_total_len:
        return None
    hidden = extract_target_layer_tokens(model, ids.unsqueeze(0), response_start, args.target_layer, device)
    hidden = filter_response_hidden_states(tokenizer, response_ids, hidden)
    if hidden.numel() == 0:
        return None
    return hidden


def iter_mil_bags(model, tokenizer, harmful: list[dict], refusals: list[str], args, device: torch.device):
    for sample in harmful:
        prompt = str(sample.get("prompt", "")).strip()
        harmful_resp = str(sample.get("response", "")).strip()
        refusal = random.choice(refusals)

        harmful_hidden = _build_response_hidden(model, tokenizer, prompt, harmful_resp, args, device)
        if harmful_hidden is not None:
            yield MILBag(harmful_hidden, 1.0)

        safe_hidden = _build_response_hidden(model, tokenizer, prompt, str(refusal).strip(), args, device)
        if safe_hidden is not None:
            yield MILBag(safe_hidden, 0.0)


def build_mil_bags(model, tokenizer, harmful: list[dict], refusals: list[str], args, device: torch.device):
    bags = []
    attempted = 2 * len(harmful)
    for bag in iter_mil_bags(model, tokenizer, harmful, refusals, args, device):
        bags.append(bag)
    return bags, attempted - len(bags)


def split_bags(bags: list[MILBag], val_ratio: float, seed: int) -> tuple[list[MILBag], list[MILBag]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")
    by_label = {0.0: [], 1.0: []}
    for bag in bags:
        by_label[float(bag.label)].append(bag)

    rng = random.Random(seed)
    train_bags: list[MILBag] = []
    val_bags: list[MILBag] = []
    for label_bags in by_label.values():
        rng.shuffle(label_bags)
        if len(label_bags) <= 1:
            train_bags.extend(label_bags)
            continue
        val_count = max(1, min(len(label_bags) - 1, round(len(label_bags) * val_ratio)))
        val_bags.extend(label_bags[:val_count])
        train_bags.extend(label_bags[val_count:])

    rng.shuffle(train_bags)
    rng.shuffle(val_bags)
    if not train_bags:
        raise RuntimeError("Training split is empty; add samples or lower val_ratio.")
    if not val_bags:
        raise RuntimeError("Validation split is empty; add samples or raise val_ratio.")
    return train_bags, val_bags


def binary_auc(scores: list[float], labels: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(1 for _score, label in pairs if label == 1.0)
    negatives = sum(1 for _score, label in pairs if label == 0.0)
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for item_idx in range(idx, end):
            if pairs[item_idx][1] == 1.0:
                rank_sum += avg_rank
        idx = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def evaluate_probe(
    probe: LinearMILProbe,
    bags: list[MILBag],
    top_q_ratio: float,
    device: torch.device,
) -> dict[str, float]:
    if not bags:
        raise RuntimeError("Cannot evaluate on an empty bag list.")
    probe.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for bag in bags:
            hidden = bag.hidden.to(device=device, dtype=next(probe.parameters()).dtype)
            loss = compute_bag_loss(probe, hidden, bag.label, top_q_ratio)
            logit = compute_bag_logit(probe, hidden, top_q_ratio)
            losses.append(float(loss.detach().cpu().item()))
            scores.append(float(logit.detach().cpu().item()))
            labels.append(float(bag.label))
    predictions = [1.0 if score >= 0.0 else 0.0 for score in scores]
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return {
        "loss": float(sum(losses) / len(losses)),
        "accuracy": float(correct / len(labels)),
        "auc": binary_auc(scores, labels),
    }


def train_probe(
    probe: LinearMILProbe,
    train_bags: list[MILBag],
    val_bags: list[MILBag],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_train_loss = 0.0
    for _epoch in range(int(args.epochs)):
        probe.train()
        losses: list[float] = []
        for bag in train_bags:
            hidden = bag.hidden.to(device=device, dtype=next(probe.parameters()).dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_bag_loss(probe, hidden, bag.label, args.top_q_ratio)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        final_train_loss = float(sum(losses) / len(losses))

    val_metrics = evaluate_probe(probe, val_bags, args.top_q_ratio, device)
    return {
        "train_loss": final_train_loss,
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_auc": float(val_metrics["auc"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/dev/shm/LLaDA-8B-Instruct")
    parser.add_argument("--harmful_json", default="data/harmbench_csd_train.json")
    parser.add_argument("--refusals_txt", default="utils/refusals.txt")
    parser.add_argument("--output_path", default="outputs/mil_token_probe_llada.pt")
    parser.add_argument("--target_layer", type=int, default=31)
    parser.add_argument("--top_q_ratio", type=float, default=0.1)
    parser.add_argument("--max_response_len", type=int, default=128)
    parser.add_argument("--max_total_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    harmful_path = resolve_path(args.harmful_json, Path.cwd())
    refusals_path = resolve_path(args.refusals_txt, Path.cwd())
    output_path = resolve_path(args.output_path, Path.cwd())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    harmful = load_harmful_data(harmful_path)
    if args.max_samples is not None:
        harmful = harmful[: args.max_samples]
    refusals = load_refusals(refusals_path)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    bags, skipped_bags = build_mil_bags(model, tokenizer, harmful, refusals, args, device)
    if not bags:
        raise RuntimeError("No MIL bags were built from the provided data.")
    input_dim = int(bags[0].hidden.shape[-1])
    train_bags, val_bags = split_bags(bags, args.val_ratio, args.seed)

    probe = LinearMILProbe(input_dim=input_dim)
    metrics = train_probe(probe, train_bags, val_bags, args, device)
    metrics.update(
        {
            "skipped_bags": float(skipped_bags),
            "train_bags": float(len(train_bags)),
            "val_bags": float(len(val_bags)),
        }
    )

    state = build_probe_state(probe, args, metrics, input_dim=input_dim)
    torch.save(state, output_path)
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved MIL token probe -> {output_path}")
    print(f"Saved MIL token probe metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
