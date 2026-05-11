#!/bin/bash
set -e

cd "$(dirname "$0")"

ROOT=results/Generalization
OUT_DIR=results/Generalization/mmlu_eval
mkdir -p "$OUT_DIR"

python mmlu_eval.py \
    "./results/Generalization/Dream/Dream_diffuguard_mmlu.json" \
    "./results/Generalization/Dream/Dream_self-reminder_mmlu.json" \
    "./results/Generalization/Dream/Dream_mmlu.json" \
    "./results/Generalization/Ours/Dream/MMLU/results.json" \
    "./results/Generalization/LLaDA/LLaDA_diffuguard_mmlu.json" \
    "./results/Generalization/LLaDA/LLaDA_self-reminder_mmlu.json" \
    "./results/Generalization/LLaDA/LLaDA_mmlu.json" \
    "./results/Generalization/Ours/LLaDA/LLaDA/results.json" \
    --output "$OUT_DIR/mmlu_accuracy.json" \
    | tee "$OUT_DIR/mmlu_accuracy.txt"
