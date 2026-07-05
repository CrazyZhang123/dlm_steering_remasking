#!/usr/bin/env bash
# 复用已有 bank,只跑 JBB+DIJA 推理 + Llama-Guard 评判(用于 baseline 重跑/稳定性测试)。
# 用法: bash scripts/run_infer_judge.sh <gpu_id> <bank_path> <tag>
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <bank_path> <tag>" >&2
  exit 2
fi

GPU_ID="$1"; BANK_PATH="$2"; TAG="$3"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"

EVAL_DIR="outputs/jbb_dija_${TAG}"
mkdir -p "${EVAL_DIR}"
MASTER_LOG="outputs/infer_judge_${TAG}.log"

{
  date -Is
  echo "[infer_judge] gpu=${GPU_ID} bank=${BANK_PATH} tag=${TAG}"

  echo "[1/2] JBB + DIJA inference (steering)"
  "${PY}" "eval_llada_steering.py" \
    --csv_path "JBB" \
    --model_path "${MODEL}" \
    --generated_samples_path "${EVAL_DIR}" \
    --attack_method "DIJA" \
    --sampler "steering" \
    --steering_vector_path "${BANK_PATH}" \
    --target_layer 31 \
    --alignment_threshold 0.0 \
    --steering_overshoot 1.0 \
    --initial_steering_ratio 0.1 \
    --max_refinement_iters 5 \
    --sampling_steps 128 \
    --mask_length 128 \
    --block_size 128 \
    --dija_mask_counts 128 \
    --device cuda \
    > "${EVAL_DIR}/run.log" 2>&1
  echo "[1/2] inference done -> ${EVAL_DIR}/results.json"

  echo "[2/2] Llama-Guard judge"
  "${PY}" "scripts/eval_llama_guard_local.py" \
    --data_path "${EVAL_DIR}/results.json" \
    --model_path "${GUARD}" \
    --output_path "${EVAL_DIR}/llama_guard_results.json" \
    --device cuda \
    > "${EVAL_DIR}/judge.log" 2>&1
  echo "[2/2] judge done -> ${EVAL_DIR}/llama_guard_results.json"

  date -Is
  echo "[infer_judge] DONE tag=${TAG}"
} > "${MASTER_LOG}" 2>&1
