#!/usr/bin/env bash
# Steering 超参单变量扫描：复用现成全局 ct_csd M12 l2 bank（不重建 bank），
# 每个配置只改 --steering_overshoot / --initial_steering_ratio 两个推理参数，
# 跑 N 轮 JBB+DIJA 推理 + Llama-Guard 评判。
# 对照基线：M12 l2 baseline overshoot=1.0 / isr=0.1（已有 65/70/68，均值 67.7），不重跑。
# 用法: bash scripts/run_steering_hparam_scan.sh <gpu_id> <n_rounds> <os:isr> [os:isr ...]
#   例: bash scripts/run_steering_hparam_scan.sh 0 3 1.5:0.1 2.0:0.1
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <n_rounds> <os:isr> [os:isr ...]" >&2
  echo "  os:isr 例 1.5:0.1 表示 steering_overshoot=1.5 initial_steering_ratio=0.1" >&2
  exit 2
fi

GPU_ID="$1"; N_ROUNDS="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"
BANK="outputs/ct_csd_llada_m12/ct_csd_bank.pt"

if [[ ! -f "${BANK}" ]]; then
  echo "bank not found: ${BANK}" >&2
  exit 3
fi

MASTER_LOG="outputs/steering_hparam_scan_gpu${GPU_ID}.log"
mkdir -p outputs

{
  date -Is
  echo "[scan] gpu=${GPU_ID} rounds=${N_ROUNDS} configs=$* bank=${BANK}"
  for CFG in "$@"; do
    OS="${CFG%%:*}"
    ISR="${CFG##*:}"
    TAG="os${OS}_isr${ISR}"
    for R in $(seq 1 "${N_ROUNDS}"); do
      EVAL_DIR="outputs/jbb_dija_ct_csd_m12_${TAG}_r${R}"
      if [[ -f "${EVAL_DIR}/llama_guard_results.json" ]]; then
        date -Is
        echo "[${TAG} r${R}] already judged, skip"
        continue
      fi
      mkdir -p "${EVAL_DIR}"
      date -Is
      echo "[${TAG} r${R}] JBB+DIJA inference (overshoot=${OS} initial_steering_ratio=${ISR})"
      "${PY}" "eval_llada_steering.py" \
        --csv_path "JBB" \
        --model_path "${MODEL}" \
        --generated_samples_path "${EVAL_DIR}" \
        --attack_method "DIJA" \
        --sampler "steering" \
        --steering_vector_path "${BANK}" \
        --target_layer 31 \
        --alignment_threshold 0.0 \
        --steering_overshoot "${OS}" \
        --initial_steering_ratio "${ISR}" \
        --max_refinement_iters 5 \
        --sampling_steps 128 \
        --mask_length 128 \
        --block_size 128 \
        --dija_mask_counts 128 \
        --device cuda \
        > "${EVAL_DIR}/run.log" 2>&1
      date -Is
      echo "[${TAG} r${R}] inference done -> ${EVAL_DIR}/results.json"

      echo "[${TAG} r${R}] Llama-Guard judge"
      "${PY}" "scripts/eval_llama_guard_local.py" \
        --data_path "${EVAL_DIR}/results.json" \
        --model_path "${GUARD}" \
        --output_path "${EVAL_DIR}/llama_guard_results.json" \
        --device cuda \
        > "${EVAL_DIR}/judge.log" 2>&1
      date -Is
      echo "[${TAG} r${R}] judge done -> ${EVAL_DIR}/llama_guard_results.json"
    done
    date -Is
    echo "[${TAG}] all rounds done"
  done
  date -Is
  echo "[scan] ALL DONE"
} > "${MASTER_LOG}" 2>&1
