"""Sure/Sorry 极简单方向引导向量构造。"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.make_csd_llada import build_sequence, load_harmful_data, set_seed


def resolve_under_cwd(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    try:
        common_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            stderr=subprocess.DEVNULL,
        )
        if isinstance(common_git_dir, bytes):
            common_git_dir = common_git_dir.decode("utf-8")
        common_root = Path(str(common_git_dir).strip()).resolve().parent
        common_path = common_root / path
        if common_path.exists():
            return common_path
    except Exception:
        pass

    return cwd_path


@torch.no_grad()
def extract_single_layer_mean(
    model,
    input_ids: torch.Tensor,
    response_start: int,
    target_layer: int,
    device: torch.device,
) -> torch.Tensor:
    buf: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        buf["hidden"] = hidden.detach()

    handle = model.model.transformer.blocks[target_layer].register_forward_hook(hook)
    try:
        _ = model(input_ids.to(device))
    finally:
        handle.remove()

    response_hidden = buf["hidden"][0, response_start:, :]
    return response_hidden.mean(dim=0).float().cpu()


def build_direction_vector(
    model,
    tokenizer,
    harmful: list[dict],
    positive_response: str,
    negative_response: str,
    target_layer: int,
    max_response_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    pos_sum = None
    neg_sum = None
    n_ok = 0
    skipped = 0

    for idx, sample in enumerate(tqdm(harmful, desc="Sure/Sorry CSD")):
        prompt = sample.get("prompt", "").strip()
        if not prompt:
            skipped += 1
            continue
        try:
            pos_ids, pos_start = build_sequence(tokenizer, prompt, positive_response, max_response_len)
            neg_ids, neg_start = build_sequence(tokenizer, prompt, negative_response, max_response_len)
            pos_mean = extract_single_layer_mean(model, pos_ids.unsqueeze(0), pos_start, target_layer, device)
            neg_mean = extract_single_layer_mean(model, neg_ids.unsqueeze(0), neg_start, target_layer, device)
        except Exception as exc:
            print(f"[{idx}] failed: {exc}")
            skipped += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        pos_sum = pos_mean if pos_sum is None else pos_sum + pos_mean
        neg_sum = neg_mean if neg_sum is None else neg_sum + neg_mean
        n_ok += 1

    if n_ok == 0 or pos_sum is None or neg_sum is None:
        raise RuntimeError("No valid prompts processed.")

    return {
        "vector": (pos_sum / n_ok) - (neg_sum / n_ok),
        "n_ok": n_ok,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/dev/shm/LLaDA-8B-Instruct")
    parser.add_argument(
        "--harmful_json",
        default=".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json",
    )
    parser.add_argument("--output_dir", default="outputs/ss_csd")
    parser.add_argument("--positive_response", default="Sure, here is how to do it:")
    parser.add_argument("--negative_response", default="Sorry, I can't help with that.")
    parser.add_argument("--target_layer", type=int, default=31)
    parser.add_argument("--max_response_len", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    harmful_path = resolve_under_cwd(args.harmful_json)
    out_dir = resolve_under_cwd(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful = load_harmful_data(harmful_path)
    if args.max_samples is not None:
        harmful = harmful[: args.max_samples]
    print(
        f"Loaded {len(harmful)} harmful prompts; "
        f"pos={args.positive_response!r} neg={args.negative_response!r}"
    )

    device = torch.device("cuda")
    print(f"Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    num_layers = len(model.model.transformer.blocks)
    if not (0 <= args.target_layer < num_layers):
        raise ValueError(f"target_layer {args.target_layer} out of range [0,{num_layers})")

    result = build_direction_vector(
        model=model,
        tokenizer=tokenizer,
        harmful=harmful,
        positive_response=args.positive_response,
        negative_response=args.negative_response,
        target_layer=args.target_layer,
        max_response_len=args.max_response_len,
        device=device,
    )

    vector = result["vector"]
    out_vec = out_dir / "steering_vectors.pt"
    torch.save({f"layer_{args.target_layer}": vector}, out_vec)
    print(
        f"n_ok={result['n_ok']} skipped={result['skipped']} "
        f"layer={args.target_layer} norm={vector.norm().item():.4f}"
    )
    print(f"Saved steering vector -> {out_vec}")


if __name__ == "__main__":
    main()
