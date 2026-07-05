#!/usr/bin/env bash
# Sure/Sorry 单方向引导向量扫描：
# 1. 先补 1 轮 no-steering 参考（DIJA 正常跑，但不传 steering_vector_path）
# 2. 再为每个配置构造单方向向量（Sure−Sorry = 有害−安全）
# 3. 复用现成推理入口跑 N 轮 JBB+DIJA 推理 + Llama-Guard 评判
# 用法: bash scripts/run_sure_sorry_csd_scan.sh <gpu_id> <n_rounds> <cfg> [cfg ...]
#   cfg 形如 word:512 / phrase:9605
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <n_rounds> <cfg> [cfg ...]" >&2
  echo "  cfg 形如 word:512 phrase:9605（word=单词，phrase=短句）" >&2
  exit 2
fi

GPU_ID="$1"
N_ROUNDS="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

WORKTREE_ROOT="$(pwd -P)"
PY="/root/miniconda3/bin/python"
MODEL="/dev/shm/LLaDA-8B-Instruct"
GUARD="/dev/shm/Llama-Guard-4-12B"
COMMON_GIT_DIR="$(git rev-parse --git-common-dir)"
COMMON_ROOT="$(cd "$(dirname "${COMMON_GIT_DIR}")" && pwd -P)"
HARMFUL="${COMMON_ROOT}/.worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json"
BUILD_ENTRY="${WORKTREE_ROOT}/utils/make_sure_sorry_csd_llada.py"
EVAL_ENTRY="${WORKTREE_ROOT}/eval_llada_steering.py"
JUDGE_ENTRY="${WORKTREE_ROOT}/scripts/eval_llama_guard_local.py"
MASTER_LOG="${WORKTREE_ROOT}/outputs/sure_sorry_csd_scan_gpu${GPU_ID}.log"

mkdir -p "${WORKTREE_ROOT}/outputs"

{
  date -Is
  echo "[ss-scan] gpu=${GPU_ID} rounds=${N_ROUNDS} configs=$*"

  if [[ "${GPU_ID}" == "0" ]]; then
    NO_STEER_DIR="${WORKTREE_ROOT}/outputs/jbb_dija_no_steering_r1"
    if [[ -f "${NO_STEER_DIR}/llama_guard_results.json" ]]; then
      date -Is
      echo "[no-steering r1] already judged, skip"
    else
      mkdir -p "${NO_STEER_DIR}"
      date -Is
      echo "[no-steering r1] JBB+DIJA inference"
      (cd "${COMMON_ROOT}" && "${PY}" "${EVAL_ENTRY}" \
        --csv_path "JBB" \
        --model_path "${MODEL}" \
        --generated_samples_path "${NO_STEER_DIR}" \
        --attack_method "DIJA" \
        --sampling_steps 128 \
        --mask_length 128 \
        --block_size 128 \
        --dija_mask_counts 128 \
        --gen_batch_size 1 \
        --device cuda) \
        > "${NO_STEER_DIR}/run.log" 2>&1
      date -Is
      echo "[no-steering r1] inference done -> ${NO_STEER_DIR}/results.json"

      echo "[no-steering r1] Llama-Guard judge"
      (cd "${COMMON_ROOT}" && "${PY}" "${JUDGE_ENTRY}" \
        --data_path "${NO_STEER_DIR}/results.json" \
        --model_path "${GUARD}" \
        --output_path "${NO_STEER_DIR}/llama_guard_results.json" \
        --device cuda \
        --batch_size 8) \
        > "${NO_STEER_DIR}/judge.log" 2>&1
      date -Is
      echo "[no-steering r1] judge done -> ${NO_STEER_DIR}/llama_guard_results.json"
    fi
  fi

  for CFG in "$@"; do
    STYLE="${CFG%%:*}"
    NSAMP="${CFG##*:}"
    case "${STYLE}" in
      word)
        POS="Sure"
        NEG="Sorry"
        ;;
      phrase)
        POS="Sure, here is how to do it:"
        NEG="Sorry, I can't help with that."
        ;;
      *)
        echo "unknown style: ${STYLE}" >&2
        exit 4
        ;;
    esac

    TAG="ss_${STYLE}_${NSAMP}"
    BANK_DIR="${WORKTREE_ROOT}/outputs/${TAG}"
    VEC="${BANK_DIR}/steering_vectors.pt"

    if [[ -f "${VEC}" ]]; then
      date -Is
      echo "[${TAG}] vector exists, skip build -> ${VEC}"
    else
      mkdir -p "${BANK_DIR}"
      date -Is
      echo "[${TAG}] build direction (pos='${POS}' neg='${NEG}' n=${NSAMP})"
      "${PY}" "${BUILD_ENTRY}" \
        --model_path "${MODEL}" \
        --harmful_json "${HARMFUL}" \
        --output_dir "${BANK_DIR}" \
        --positive_response "${POS}" \
        --negative_response "${NEG}" \
        --target_layer 31 \
        --max_response_len 128 \
        --max_samples "${NSAMP}" \
        --seed 42 \
        > "${BANK_DIR}/run.log" 2>&1
      date -Is
      echo "[${TAG}] build done -> ${VEC}"
    fi

    for R in $(seq 1 "${N_ROUNDS}"); do
      EVAL_DIR="${WORKTREE_ROOT}/outputs/jbb_dija_${TAG}_r${R}"
      if [[ -f "${EVAL_DIR}/llama_guard_results.json" ]]; then
        date -Is
        echo "[${TAG} r${R}] already judged, skip"
        continue
      fi

      mkdir -p "${EVAL_DIR}"
      date -Is
      echo "[${TAG} r${R}] JBB+DIJA inference"
      (cd "${COMMON_ROOT}" && "${PY}" "${EVAL_ENTRY}" \
        --csv_path "JBB" \
        --model_path "${MODEL}" \
        --generated_samples_path "${EVAL_DIR}" \
        --attack_method "DIJA" \
        --sampler "steering" \
        --steering_vector_path "${VEC}" \
        --target_layer 31 \
        --alignment_threshold 0.0 \
        --steering_overshoot 1.0 \
        --initial_steering_ratio 0.1 \
        --max_refinement_iters 5 \
        --sampling_steps 128 \
        --mask_length 128 \
        --block_size 128 \
        --dija_mask_counts 128 \
        --gen_batch_size 1 \
        --device cuda) \
        > "${EVAL_DIR}/run.log" 2>&1
      date -Is
      echo "[${TAG} r${R}] inference done -> ${EVAL_DIR}/results.json"

      echo "[${TAG} r${R}] Llama-Guard judge"
      (cd "${COMMON_ROOT}" && "${PY}" "${JUDGE_ENTRY}" \
        --data_path "${EVAL_DIR}/results.json" \
        --model_path "${GUARD}" \
        --output_path "${EVAL_DIR}/llama_guard_results.json" \
        --device cuda \
        --batch_size 8) \
        > "${EVAL_DIR}/judge.log" 2>&1
      date -Is
      echo "[${TAG} r${R}] judge done -> ${EVAL_DIR}/llama_guard_results.json"
    done

    date -Is
    echo "[${TAG}] all rounds done"
  done

  date -Is
  echo "[ss-scan] ALL DONE"
} > "${MASTER_LOG}" 2>&1
