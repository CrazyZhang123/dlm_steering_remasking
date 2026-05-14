#!/bin/bash
set -e

cd "$(dirname "$0")"

OUT_DIR=outputs/Generalization/math500_eval
mkdir -p "$OUT_DIR"

python utils/MATH-500_eval.py \
    "./outputs/Generalization/Dream/Dream_MATH-500.json" \
    "./outputs/Generalization/Dream/Dream_diffuguard_MATH-500.json" \
    "./outputs/Generalization/Dream/Dream_self-reminder_MATH-500.json" \
    "./outputs/Generalization/Ours/Dream/MATH/results.json" \
    "./outputs/Generalization/LLaDA/MATH-500.json" \
    "./outputs/Generalization/LLaDA/LLaDA_diffuguard_MATH-500.json" \
    "./outputs/Generalization/LLaDA/LLaDA_self-reminder_MATH-500.json" \
    "./outputs/Generalization/Ours/LLaDA/MATH/results.json" \
    --output "$OUT_DIR/math500_accuracy.json" \
    | tee "$OUT_DIR/math500_accuracy.txt"
