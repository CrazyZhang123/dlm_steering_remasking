import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk


HARMFUL_DATA_TYPES = {"vanilla_harmful", "adversarial_harmful"}


def parse_non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def extract_prompt(row):
    data_type = str((row.get("data_type") or "")).strip()
    if data_type not in HARMFUL_DATA_TYPES:
        return ""

    adversarial = str((row.get("adversarial") or "")).strip()
    vanilla = str((row.get("vanilla") or "")).strip()
    return adversarial or vanilla


def normalize_prompts(rows, limit, seed):
    if limit < 0:
        raise ValueError("limit must be non-negative")

    prompts = []
    seen = set()

    for row in rows:
        prompt = extract_prompt(row)
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)

    random.Random(seed).shuffle(prompts)
    return prompts[:limit]


def write_prompt_csv(path, prompts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt"])
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({"prompt": prompt})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="allenai/wildjailbreak")
    parser.add_argument("--subset", default="train")
    parser.add_argument("--split", dest="subset", help=argparse.SUPPRESS)
    parser.add_argument("--save_disk", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_smoke_csv", required=True)
    parser.add_argument("--limit", type=parse_non_negative_int, default=200)
    parser.add_argument("--smoke_count", type=parse_non_negative_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_disk = Path(args.save_disk)
    if save_disk.exists():
        dataset = load_from_disk(str(save_disk))
    else:
        dataset = load_dataset(
            args.dataset_id,
            args.subset,
            delimiter="\t",
            keep_default_na=False,
        )
        dataset.save_to_disk(str(save_disk))

    prompts = normalize_prompts(dataset, limit=args.limit, seed=args.seed)
    write_prompt_csv(args.output_csv, prompts)
    write_prompt_csv(args.output_smoke_csv, prompts[: args.smoke_count])

    print(f"saved {len(prompts)} prompts to {args.output_csv}")
    print(f"saved {min(len(prompts), args.smoke_count)} prompts to {args.output_smoke_csv}")


if __name__ == "__main__":
    main()
