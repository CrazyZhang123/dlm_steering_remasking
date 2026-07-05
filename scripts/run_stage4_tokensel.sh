#!/usr/bin/env bash
# Stage 4 token-selection 消融通用脚本（NO 降维，feature_preprocess=l2_only）。
# 用法: bash scripts/run_stage4_tokensel.sh <gpu_id> <bank_method> <token_selection> <M> <ratio> <coarse_type> <tag>
#   bank_method     : ct_csd(全局) | category_ct_csd(按 category)
#   token_selection : direction_top_ratio | random_top_ratio | all
#   M               : --num_total_clusters (如 12 / 16)
#   ratio           : --selection_ratio
#   coarse_type     : --coarse_direction_type (random/all 时不生效但需占位, 传 global 即可)
#   tag             : 输出目录后缀
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 <gpu_id> <bank_method> <token_selection> <M> <ratio> <coarse_type> <tag>" >&2
  exit 2
fi

GPU_ID="$1"; BANK_METHOD="$2"; METHOD="$3"; M="$4"; RATIO="$5"; COARSE="$6"; TAG="$7"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"
HARMFUL=".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json"

BANK_DIR="outputs/tsel_${TAG}"
EVAL_DIR="outputs/jbb_dija_tsel_${TAG}"
mkdir -p "${BANK_DIR}" "${EVAL_DIR}"
MASTER_LOG="outputs/stage4_tsel_${TAG}.log"

{
  date -Is
  echo "[tsel] gpu=${GPU_ID} bank_method=${BANK_METHOD} token_selection=${METHOD} M=${M} ratio=${RATIO} coarse=${COARSE} tag=${TAG}"

  echo "[1/3] build bank (bank_method=${BANK_METHOD}, token_selection=${METHOD}, M=${M}, feature_preprocess=l2_only 不降维)"
  "${PY}" "utils/make_ct_csd_llada.py" \
    --model_path "${MODEL}" \
    --harmful_json "${HARMFUL}" \
    --refusals_txt "utils/refusals.txt" \
    --output_dir "${BANK_DIR}" \
    --target_layer 31 \
    --max_response_len 128 \
    --max_total_len 2048 \
    --method "${BANK_METHOD}" \
    --category_key semantic_category \
    --num_total_clusters "${M}" \
    --kmeans_batch_size 4096 \
    --token_selection "${METHOD}" \
    --selection_ratio "${RATIO}" \
    --coarse_direction_type "${COARSE}" \
    --feature_preprocess l2_only \
    --device cuda \
    --seed 42 \
    > "${BANK_DIR}/run.log" 2>&1
  echo "[1/3] bank done -> ${BANK_DIR}/ct_csd_bank.pt"

  echo "[2/3] JBB + DIJA inference (steering)"
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
  echo "[2/3] inference done -> ${EVAL_DIR}/results.json"

  echo "[3/3] Llama-Guard judge"
  "${PY}" "scripts/eval_llama_guard_local.py" \
    --data_path "${EVAL_DIR}/results.json" \
    --model_path "${GUARD}" \
    --output_path "${EVAL_DIR}/llama_guard_results.json" \
    --device cuda \
    > "${EVAL_DIR}/judge.log" 2>&1
  echo "[3/3] judge done -> ${EVAL_DIR}/llama_guard_results.json"

  date -Is
  echo "[tsel] DONE tag=${TAG}"
} > "${MASTER_LOG}" 2>&1
