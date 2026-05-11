import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def load_refusals(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_harmful_data(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat: list[dict] = []
    for item in data:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def build_sequence(tokenizer, prompt: str, response: str, max_response_len: int):
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(response_ids) > max_response_len:
        response_ids = response_ids[:max_response_len]
    full = torch.cat([prompt_ids, response_ids], dim=0)
    return full, int(prompt_ids.shape[0])


@torch.no_grad()
def extract_layer_means(model, input_ids: torch.Tensor, response_start: int, num_layers: int,
                        device: torch.device) -> dict[int, torch.Tensor]:
    buf: dict[int, torch.Tensor] = {}
    handles = []
    for i in range(num_layers):
        def make_hook(idx):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                buf[idx] = h.detach()
            return hook
        handles.append(model.model.transformer.blocks[i].register_forward_hook(make_hook(i)))
    try:
        _ = model(input_ids.to(device))
    finally:
        for h in handles:
            h.remove()

    out: dict[int, torch.Tensor] = {}
    for i in range(num_layers):
        h = buf[i]  # [1, seq, d]
        resp = h[0, response_start:, :]  # [resp_len, d]
        out[i] = resp.mean(dim=0).float().cpu()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--harmful_json", default="./data/wild_baseline/wild_baseline_llama_guard_harmful.json")
    p.add_argument("--refusals_txt", default="./utils/refusals.txt")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--max_response_len", type=int, default=128)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    here = Path(__file__).parent
    harmful_path = here / args.harmful_json
    refusals_path = here / args.refusals_txt
    out_dir = here / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    refusals = load_refusals(refusals_path)
    harmful = load_harmful_data(harmful_path)
    if args.max_samples is not None:
        harmful = harmful[: args.max_samples]
    print(f"Loaded {len(harmful)} harmful samples, {len(refusals)} refusal paraphrases")

    device = torch.device("cuda")
    print(f"Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    num_layers = len(model.model.transformer.blocks)
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    print(f"num_layers={num_layers}, d_model={d_model}")

    harmful_means_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    safe_means_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    n_ok = 0
    skipped = 0
    refusal_log: list[dict] = []

    for idx, sample in enumerate(tqdm(harmful, desc="Extracting")):
        prompt = sample.get("prompt", "").strip()
        harmful_resp = sample.get("response", "").strip()
        if not prompt or not harmful_resp:
            skipped += 1
            continue

        refusal_resp = random.choice(refusals)

        try:
            ids_h, rs_h = build_sequence(tokenizer, prompt, harmful_resp, args.max_response_len)
            ids_s, rs_s = build_sequence(tokenizer, prompt, refusal_resp, args.max_response_len)
        except Exception as e:
            print(f"[{idx}] tokenize failed: {e}")
            skipped += 1
            continue

        if len(ids_h) > args.max_total_len or len(ids_s) > args.max_total_len:
            skipped += 1
            continue

        try:
            h_means = extract_layer_means(model, ids_h.unsqueeze(0), rs_h, num_layers, device)
            s_means = extract_layer_means(model, ids_s.unsqueeze(0), rs_s, num_layers, device)
        except RuntimeError as e:
            print(f"[{idx}] forward failed: {e}")
            torch.cuda.empty_cache()
            skipped += 1
            continue

        for i in range(num_layers):
            harmful_means_per_layer[i].append(h_means[i])
            safe_means_per_layer[i].append(s_means[i])
        n_ok += 1
        refusal_log.append({"idx": idx, "refusal": refusal_resp})

    if n_ok == 0:
        raise RuntimeError("No samples processed successfully.")
    print(f"Aggregated {n_ok} samples ({skipped} skipped)")

    steering_vectors: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        h_stack = torch.stack(harmful_means_per_layer[i])  # [n_ok, d]
        s_stack = torch.stack(safe_means_per_layer[i])     # [n_ok, d]
        h_mean = h_stack.mean(dim=0)
        s_mean = s_stack.mean(dim=0)
        v = h_mean - s_mean  # harmful - safe (matches repo convention)
        steering_vectors[f"layer_{i}"] = v

    out_vec = out_dir / "steering_vectors.pt"
    torch.save(steering_vectors, out_vec)
    print(f"Saved steering vectors -> {out_vec}")


if __name__ == "__main__":
    main()
