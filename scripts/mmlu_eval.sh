#!/bin/bash
set -e

cd "$(dirname "$0")"

ROOT=outputs/Generalization
OUT_DIR=outputs/Generalization/mmlu_eval
mkdir -p "$OUT_DIR"

python utils/mmlu_eval.py \
    "./outputs/Generalization/Dream/Dream_diffuguard_mmlu.json" \
    "./outputs/Generalization/Dream/Dream_self-reminder_mmlu.json" \
    "./outputs/Generalization/Dream/Dream_mmlu.json" \
    "./outputs/Generalization/Ours/Dream/MMLU/results.json" \
    "./outputs/Generalization/LLaDA/LLaDA_diffuguard_mmlu.json" \
    "./outputs/Generalization/LLaDA/LLaDA_self-reminder_mmlu.json" \
    "./outputs/Generalization/LLaDA/LLaDA_mmlu.json" \
    "./outputs/Generalization/Ours/LLaDA/LLaDA/results.json" \
    --output "$OUT_DIR/mmlu_accuracy.json" \
    | tee "$OUT_DIR/mmlu_accuracy.txt"
