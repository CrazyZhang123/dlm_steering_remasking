import argparse
import json
import csv
from pathlib import Path


def summarize_collection(root: Path):
    outputs = root / "outputs"
    completed_result_files = 0
    completed_samples = 0
    judged_result_files = 0
    judged_samples = 0
    unsafe_count = 0
    external_judged_result_files = 0
    external_judged_samples = 0
    external_unsafe_count = 0

    relevant_dirs = []
    relevant_dirs.extend(sorted(outputs.glob("wildjailbreak_round_*_gpu*")))
    smoke_dir = outputs / "wildjailbreak_smoke"
    if smoke_dir.exists():
        relevant_dirs.append(smoke_dir)
    relevant_dirs.extend(sorted(outputs.glob("queue_gpu*")))

    seen_dirs = set()
    for round_dir in relevant_dirs:
        if not round_dir.is_dir() or round_dir in seen_dirs:
            continue
        seen_dirs.add(round_dir)
        results_path = round_dir / "results.json"
        judge_path = round_dir / "lumingapi_gpt54_judge.json"
        local_judge_path = round_dir / "llama_guard_local.json"

        if results_path.exists():
            completed_result_files += 1
            completed_samples += len(json.loads(results_path.read_text(encoding="utf-8")))

        if judge_path.exists():
            external_judged_result_files += 1
            metadata = json.loads(judge_path.read_text(encoding="utf-8"))["metadata"]
            external_judged_samples += metadata.get("resolved_samples", 0)
            external_unsafe_count += metadata.get("unsafe_count", 0)

        if local_judge_path.exists():
            judged_result_files += 1
            metadata = json.loads(local_judge_path.read_text(encoding="utf-8"))["metadata"]
            judged_samples += metadata.get("total_samples", 0)
            unsafe_count += metadata.get("unsafe_count", 0)

    harmful_pairs_path = root / "data" / "csd_llada_harmful_pairs_ext_all.json"
    harmful_pairs = 0
    if harmful_pairs_path.exists():
        harmful_pairs = len(json.loads(harmful_pairs_path.read_text(encoding="utf-8")))

    all_prompts_path = root / "data" / "wildjailbreak_train_prompts_all.csv"
    available_prompt_pool = None
    if all_prompts_path.exists():
        with all_prompts_path.open("r", encoding="utf-8", newline="") as handle:
            available_prompt_pool = sum(1 for _ in csv.DictReader(handle))

    if external_unsafe_count > 0 and external_judged_samples > 0:
        estimated_prompts_needed = round(5763 / (external_unsafe_count / external_judged_samples))
        projected_harmful_pairs = (
            round(available_prompt_pool * (external_unsafe_count / external_judged_samples))
            if available_prompt_pool is not None
            else None
        )
    else:
        estimated_prompts_needed = None
        projected_harmful_pairs = None

    return {
        "completed_result_files": completed_result_files,
        "completed_samples": completed_samples,
        "judged_result_files": judged_result_files,
        "judged_samples": judged_samples,
        "unsafe_count": unsafe_count,
        "external_judged_result_files": external_judged_result_files,
        "external_judged_samples": external_judged_samples,
        "external_unsafe_count": external_unsafe_count,
        "harmful_pairs": harmful_pairs,
        "external_unsafe_rate_over_judged": round(
            external_unsafe_count / max(external_judged_samples, 1), 6
        ),
        "available_prompt_pool": available_prompt_pool,
        "projected_harmful_pairs_at_current_external_rate": projected_harmful_pairs,
        "pool_sufficient_at_current_external_rate": (
            projected_harmful_pairs is not None and projected_harmful_pairs >= 5763
        ),
        "estimated_prompts_needed_at_current_external_rate": estimated_prompts_needed,
        "remaining_to_5763": max(5763 - harmful_pairs, 0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    summary = summarize_collection(Path(args.root).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
