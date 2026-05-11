#!/bin/bash
set -e

cd "$(dirname "$0")"

OUT_DIR=results/Generalization/math500_eval
mkdir -p "$OUT_DIR"

python MATH-500_eval.py \
    "./results/Generalization/Dream/Dream_MATH-500.json" \
    "./results/Generalization/Dream/Dream_diffuguard_MATH-500.json" \
    "./results/Generalization/Dream/Dream_self-reminder_MATH-500.json" \
    "./results/Generalization/Ours/Dream/MATH/results.json" \
    "./results/Generalization/LLaDA/MATH-500.json" \
    "./results/Generalization/LLaDA/LLaDA_diffuguard_MATH-500.json" \
    "./results/Generalization/LLaDA/LLaDA_self-reminder_MATH-500.json" \
    "./results/Generalization/Ours/LLaDA/MATH/results.json" \
    --output "$OUT_DIR/math500_accuracy.json" \
    | tee "$OUT_DIR/math500_accuracy.txt"
