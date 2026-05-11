import json
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer


def load_results(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_gt_map(dataset) -> dict[str, dict]:
    qt = {}
    for row in dataset:
        q = row["Question"].strip()
        best = row.get("Best Answer", "").strip()
        correct_list = [a.strip() for a in row.get("Correct Answers", "").split(";") if a.strip()]
        qt[q] = {"best": best, "correct": correct_list}
    return qt


def max_rouge_over_refs(scorer, hypothesis, references):
    scores = [scorer.score(ref, hypothesis) for ref in references if ref]
    if not scores:
        return {k: 0.0 for k in scorer.rouge_types}
    return {k: max(s[k].fmeasure for s in scores) for k in scorer.rouge_types}


def evaluate(results_path: str, output_path: str):
    results = load_results(results_path)
    print(f"Loaded {len(results)} results from {results_path}")

    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("domenicrosati/TruthfulQA", split="train")

    gt_map = build_gt_map(dataset)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_types = ["rouge1", "rouge2", "rougeL"]

    per_item = []
    missing = 0

    for item in results:
        try:
            question = item["vanilla prompt"].strip()
        except:
            try:
                question = item["prompt"].strip()
            except:
                question = item["vanilla_prompt"].strip()
        try:
            hypothesis = item["response"].strip()
        except:
            try:
                hypothesis = item["result"].strip()
            except:
                hypothesis = item["response"][0].strip()

        gt = gt_map.get(question)
        if gt is None:
            missing += 1
            continue

        scores_best = scorer.score(gt["best"], hypothesis) if gt["best"] else {}
        scores_correct = max_rouge_over_refs(scorer, hypothesis, gt["correct"])

        per_item.append({
            "question": question,
            "response": hypothesis,
            "best_answer": gt["best"],
            "correct_answers": gt["correct"],
            "scores_best": {k: round(v.fmeasure, 4) for k, v in scores_best.items()},
            "scores_correct": {k: round(v, 4) for k, v in scores_correct.items()},
        })

    if missing:
        print(f"Warning: {missing} sample(s) had no matching question in TruthfulQA.")

    def avg(key):
        return {rt: round(np.mean([x[key][rt] for x in per_item if rt in x[key]]), 4) for rt in rouge_types}

    summary = {
        "num_evaluated": len(per_item),
        "avg_scores_best_answer": avg("scores_best"),
        "avg_scores_correct_answers": avg("scores_correct"),
    }

    print("\n========== ROUGE Scores ==========")
    print(f"Evaluated: {summary['num_evaluated']} samples\n")
    print("[ vs. Best Answer ]")
    for rt, val in summary["avg_scores_best_answer"].items():
        print(f"  {rt:8s}: {val:.4f}")
    print("\n[ vs. Correct Answers (max over refs) ]")
    for rt, val in summary["avg_scores_correct_answers"].items():
        print(f"  {rt:8s}: {val:.4f}")
    print("==================================\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "details": per_item}, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to results.json (generated responses)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save per-item scores as JSON (optional)")
    args = parser.parse_args()
    evaluate(args.results_path, args.output_path)
