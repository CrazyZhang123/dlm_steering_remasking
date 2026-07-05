#!/usr/bin/env bash
# 阶段 1/2 簇数细扫（全局 ct_csd, token_selection=all）：
# 每个 M 构造一次 bank（seed=42 确定性，已存在则跳过），再复用 bank 顺序跑 N 轮推理+评判，
# 用于在 run-to-run 噪声（约 ±2.5%）下评估 M 点位的真实 ASR。
# 用法: bash scripts/run_ct_csd_m_scan_repeat.sh <gpu_id> <n_rounds> <m1> [m2 ...]
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <n_rounds> <m1> [m2 ...]" >&2
  exit 2
fi

GPU_ID="$1"; N_ROUNDS="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"
HARMFUL=".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json"

MASTER_LOG="outputs/stage2_ct_csd_m_scan_gpu${GPU_ID}.log"
mkdir -p outputs

{
  date -Is
  echo "[scan] gpu=${GPU_ID} rounds=${N_ROUNDS} m_list=$*"
  for M in "$@"; do
    BANK_DIR="outputs/ct_csd_llada_m${M}"
    if [[ -f "${BANK_DIR}/ct_csd_bank.pt" ]]; then
      date -Is
      echo "[M=${M}] bank exists, skip build -> ${BANK_DIR}/ct_csd_bank.pt"
    else
      mkdir -p "${BANK_DIR}"
      date -Is
      echo "[M=${M}] build ct_csd bank"
      "${PY}" "utils/make_ct_csd_llada.py" \
        --model_path "${MODEL}" \
        --harmful_json "${HARMFUL}" \
        --refusals_txt "utils/refusals.txt" \
        --output_dir "${BANK_DIR}" \
        --target_layer 31 \
        --max_response_len 128 \
        --max_total_len 2048 \
        --method ct_csd \
        --num_total_clusters "${M}" \
        --kmeans_batch_size 4096 \
        --device cuda \
        --seed 42 \
        > "${BANK_DIR}/run.log" 2>&1
      date -Is
      echo "[M=${M}] bank done -> ${BANK_DIR}/ct_csd_bank.pt"
    fi

    for R in $(seq 1 "${N_ROUNDS}"); do
      EVAL_DIR="outputs/jbb_dija_ct_csd_m${M}_r${R}"
      if [[ -f "${EVAL_DIR}/llama_guard_results.json" ]]; then
        date -Is
        echo "[M=${M} r${R}] already judged, skip"
        continue
      fi
      mkdir -p "${EVAL_DIR}"
      date -Is
      echo "[M=${M} r${R}] JBB+DIJA inference"
      "${PY}" "eval_llada_steering.py" \
        --csv_path "JBB" \
        --model_path "${MODEL}" \
        --generated_samples_path "${EVAL_DIR}" \
        --attack_method "DIJA" \
        --sampler "steering" \
        --steering_vector_path "${BANK_DIR}/ct_csd_bank.pt" \
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
      date -Is
      echo "[M=${M} r${R}] inference done -> ${EVAL_DIR}/results.json"

      echo "[M=${M} r${R}] Llama-Guard judge"
      "${PY}" "scripts/eval_llama_guard_local.py" \
        --data_path "${EVAL_DIR}/results.json" \
        --model_path "${GUARD}" \
        --output_path "${EVAL_DIR}/llama_guard_results.json" \
        --device cuda \
        > "${EVAL_DIR}/judge.log" 2>&1
      date -Is
      echo "[M=${M} r${R}] judge done -> ${EVAL_DIR}/llama_guard_results.json"
    done
    date -Is
    echo "[M=${M}] all rounds done"
  done
  date -Is
  echo "[scan] ALL DONE"
} > "${MASTER_LOG}" 2>&1
