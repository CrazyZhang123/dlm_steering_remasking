#!/usr/bin/env bash
# Stage 5 预处理消融（M12 固定）：依次构建 center_l2 / center_pca256_l2 两个 bank，
# 共享 --preprocess_stats_cache（首个模式拟合并落盘统计量，后续模式命中缓存跳过拟合遍），
# 每个 bank 复用跑 N 轮 JBB+DIJA 推理 + Llama-Guard 评判。
# 对照组：l2_only M12（65/70/68）与 center_pca128_l2 M12（70/70/72），用于拆分
# "去均值"与"PCA 降维"各自对 ASR 的影响。
# 用法: bash scripts/run_ct_csd_m12_preprocess_ablation.sh <gpu_id> <n_rounds> <mode1> [mode2 ...]
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <n_rounds> <mode1> [mode2 ...]" >&2
  exit 2
fi

GPU_ID="$1"; N_ROUNDS="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"
HARMFUL=".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json"
M=12
STATS_CACHE="outputs/ct_csd_llada_m_all_preprocess_stats.pt"

MASTER_LOG="outputs/stage5_m12_preprocess_ablation_gpu${GPU_ID}.log"
mkdir -p outputs

{
  date -Is
  echo "[ablation] gpu=${GPU_ID} rounds=${N_ROUNDS} M=${M} modes=$* stats_cache=${STATS_CACHE}"
  for MODE in "$@"; do
    BANK_DIR="outputs/ct_csd_llada_m${M}_${MODE}"
    if [[ -f "${BANK_DIR}/ct_csd_bank.pt" ]]; then
      date -Is
      echo "[${MODE}] bank exists, skip build -> ${BANK_DIR}/ct_csd_bank.pt"
    else
      mkdir -p "${BANK_DIR}"
      date -Is
      echo "[${MODE}] build ct_csd bank M=${M}"
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
        --feature_preprocess "${MODE}" \
        --preprocess_stats_cache "${STATS_CACHE}" \
        --kmeans_batch_size 4096 \
        --device cuda \
        --seed 42 \
        > "${BANK_DIR}/run.log" 2>&1
      date -Is
      echo "[${MODE}] bank done -> ${BANK_DIR}/ct_csd_bank.pt"
    fi

    for R in $(seq 1 "${N_ROUNDS}"); do
      EVAL_DIR="outputs/jbb_dija_ct_csd_m${M}_${MODE}_r${R}"
      if [[ -f "${EVAL_DIR}/llama_guard_results.json" ]]; then
        date -Is
        echo "[${MODE} r${R}] already judged, skip"
        continue
      fi
      mkdir -p "${EVAL_DIR}"
      date -Is
      echo "[${MODE} r${R}] JBB+DIJA inference"
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
      echo "[${MODE} r${R}] inference done -> ${EVAL_DIR}/results.json"

      echo "[${MODE} r${R}] Llama-Guard judge"
      "${PY}" "scripts/eval_llama_guard_local.py" \
        --data_path "${EVAL_DIR}/results.json" \
        --model_path "${GUARD}" \
        --output_path "${EVAL_DIR}/llama_guard_results.json" \
        --device cuda \
        > "${EVAL_DIR}/judge.log" 2>&1
      date -Is
      echo "[${MODE} r${R}] judge done -> ${EVAL_DIR}/llama_guard_results.json"
    done
    date -Is
    echo "[${MODE}] all rounds done"
  done
  date -Is
  echo "[ablation] ALL DONE"
} > "${MASTER_LOG}" 2>&1
