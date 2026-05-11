import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset

SUBJECTS = ["logical_fallacies", "moral_scenarios", "philosophy"]


def load_mmlu():
    datasets_list = [
        load_dataset("cais/mmlu", subject, split="test") for subject in SUBJECTS
    ]
    return list(concatenate_datasets(datasets_list))


def parse_response(text, choices=None):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    m = re.search(r"[0-3]", text)
    if m is not None:
        return int(m.group(0))
    if choices is None:
        return None
    s = text.lower().strip("'\".,!? ")
    for j, c in enumerate(choices):
        cl = c.lower()
        if cl in s or s in cl:
            return j
    return None


def evaluate(result_path, dataset):
    with open(result_path) as f:
        results = json.load(f)

    if len(results) != len(dataset):
        print(
            f"[warn] {result_path}: result count ({len(results)}) "
            f"!= dataset size ({len(dataset)}); evaluating min length."
        )
    n = min(len(results), len(dataset))

    total = defaultdict(int)
    correct = defaultdict(int)
    invalid = defaultdict(int)

    for i in range(n):
        gold = dataset[i]["answer"]
        subject = dataset[i]["subject"]
        pred = parse_response(results[i].get("response", ""), dataset[i]["choices"])

        total["all"] += 1
        total[subject] += 1
        if pred is None:
            invalid["all"] += 1
            invalid[subject] += 1
            continue
        if pred == gold:
            correct["all"] += 1
            correct[subject] += 1

    return total, correct, invalid


def print_report(result_path, total, correct, invalid):
    print(f"\n=== {result_path} ===")
    header = f"{'subject':<20} {'correct':>8} {'total':>8} {'invalid':>8} {'acc(%)':>8}"
    print(header)
    print("-" * len(header))
    for subject in SUBJECTS + ["all"]:
        t = total.get(subject, 0)
        c = correct.get(subject, 0)
        inv = invalid.get(subject, 0)
        acc = 100.0 * c / t if t > 0 else 0.0
        print(f"{subject:<20} {c:>8} {t:>8} {inv:>8} {acc:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Result JSON paths")
    parser.add_argument("--glob", help="Glob pattern for result JSONs")
    parser.add_argument("--output", help="Save aggregated report to this JSON file")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    if not paths:
        parser.error("Provide at least one result path or --glob.")

    dataset = load_mmlu()

    report = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}")
            continue
        total, correct, invalid = evaluate(path, dataset)
        print_report(path, total, correct, invalid)
        entry = {"path": path, "per_subject": {}}
        for subject in SUBJECTS + ["all"]:
            t = total.get(subject, 0)
            c = correct.get(subject, 0)
            inv = invalid.get(subject, 0)
            entry["per_subject"][subject] = {
                "correct": c,
                "total": t,
                "invalid": inv,
                "accuracy": 100.0 * c / t if t else 0.0,
            }
        report.append(entry)

    if len(report) > 1:
        print("\n=== Summary ===")
        header = f"{'file':<70} {'correct':>8} {'total':>8} {'invalid':>8} {'acc(%)':>8}"
        print(header)
        print("-" * len(header))
        for entry in report:
            a = entry["per_subject"]["all"]
            short = entry["path"] if len(entry["path"]) <= 70 else "..." + entry["path"][-67:]
            print(f"{short:<70} {a['correct']:>8} {a['total']:>8} {a['invalid']:>8} {a['accuracy']:>8.2f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
