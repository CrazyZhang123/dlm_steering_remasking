import argparse
import json
from pathlib import Path


def merge_unsafe_pairs(round_items):
    merged = []
    seen = set()
    for items in round_items:
        for item in items:
            if not item.get("unsafe"):
                continue
            prompt = str(item.get("prompt", "")).strip()
            response = str(item.get("response", "")).strip()
            if not prompt or not response or prompt in seen:
                continue
            seen.add(prompt)
            merged.append({"prompt": prompt, "response": response})
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--round_dir", action="append", required=True)
    parser.add_argument("--judge_json_name", default="llama_guard_local.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    round_items = []
    for round_dir in args.round_dir:
        judge_path = Path(round_dir) / args.judge_json_name
        payload = json.loads(judge_path.read_text(encoding="utf-8"))
        round_items.append(payload["results"])

    merged = merge_unsafe_pairs(round_items)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(merged)} harmful pairs to {output_path}")


if __name__ == "__main__":
    main()
