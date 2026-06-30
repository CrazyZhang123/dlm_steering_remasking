#!/usr/bin/env bash
# DIJA HarmBench × LLaDA-Instruct-8B 单步解码粒度扫描
# 用法: bash scripts/run_dija_harmbench_decode_scan.sh <phase> [gpu] [N_list]
#   phase  : smoke | full
#   gpu    : CUDA 设备号，默认 1
#   N_list : 解码粒度列表，默认 "1 2 3 4 5"
# 单 GPU 内串行执行（LLaDA-8B fp16 ≈ 18GB，单卡只跑一个进程）。
set -euo pipefail

PHASE="${1:-smoke}"
GPU="${2:-1}"
N_LIST="${3:-1 2 3 4 5}"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

PYTHON=/root/miniconda3/envs/diffuguard/bin/python
MODEL_PATH=/dev/shm/LLaDA-8B-Instruct
ENTRY=DIJA/run_harmbench/models/harmbench_llada.py
ORIG_DATA=DIJA/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_Qwen.json

if [ "$PHASE" = "smoke" ]; then
    DATA=/tmp/harmbench_smoke10.json
    "$PYTHON" -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:10], open(sys.argv[2],'w'), ensure_ascii=False, indent=2)" "$ORIG_DATA" "$DATA"
    OUT_ROOT="outputs/dija_harmbench_llada_instruct/smoke10"
else
    DATA="$ORIG_DATA"
    OUT_ROOT="outputs/dija_harmbench_llada_instruct/full400"
fi

mkdir -p "$OUT_ROOT/logs"

for N in $N_LIST; do
    OUT="$OUT_ROOT/k${N}"
    mkdir -p "$OUT"
    echo "[$(date '+%F %T')] start tokens_per_step=$N on GPU $GPU -> $OUT/results.json"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ENTRY" \
        --model_path "$MODEL_PATH" \
        --attack_prompt "$DATA" \
        --output_json "$OUT/results.json" \
        --attack_method DIJA \
        --tokens_per_step "$N" \
        --steps 128 --gen_length 128 --mask_counts 36 \
        > "$OUT_ROOT/logs/k${N}.log" 2>&1
    echo "[$(date '+%F %T')] done tokens_per_step=$N"
done

echo "[$(date '+%F %T')] ALL DONE phase=$PHASE gpu=$GPU N_list=[$N_LIST]"
