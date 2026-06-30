#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <m1> [m2 ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MASTER_LOG="outputs/stage3_category_ct_csd_ablation_gpu${GPU_ID}.log"
mkdir -p "outputs"

{
  date -Is
  echo "[queue] gpu=${GPU_ID} m_list=$*"
  for M in "$@"; do
    BANK_DIR="outputs/category_ct_csd_llada_m${M}"
    EVAL_DIR="outputs/jbb_dija_category_ct_csd_m${M}"
    mkdir -p "${BANK_DIR}" "${EVAL_DIR}"

    date -Is
    echo "[M=${M}] build category_ct_csd bank"
    python "utils/make_ct_csd_llada.py" \
      --model_path "/dev/shm/LLaDA-8B-Instruct" \
      --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
      --refusals_txt "utils/refusals.txt" \
      --output_dir "${BANK_DIR}" \
      --target_layer 31 \
      --max_response_len 128 \
      --max_total_len 2048 \
      --method category_ct_csd \
      --category_key semantic_category \
      --num_total_clusters "${M}" \
      --kmeans_batch_size 4096 \
      --device cuda \
      --seed 42 \
      > "${BANK_DIR}/run.log" 2>&1

    date -Is
    echo "[M=${M}] run JBB DIJA inference"
    python "eval_llada_steering.py" \
      --csv_path "JBB" \
      --model_path "/dev/shm/LLaDA-8B-Instruct" \
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
    echo "[M=${M}] run Llama-Guard judge"
    python "scripts/eval_llama_guard_local.py" \
      --data_path "${EVAL_DIR}/results.json" \
      --model_path "/dev/shm/Llama-Guard-4-12B" \
      --output_path "${EVAL_DIR}/llama_guard_results.json" \
      --device cuda \
      > "${EVAL_DIR}/judge.log" 2>&1

    date -Is
    echo "[M=${M}] done"
  done
  date -Is
  echo "[queue] done"
} > "${MASTER_LOG}" 2>&1
