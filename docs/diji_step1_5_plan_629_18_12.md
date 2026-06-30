# DIJA HarmBench × LLaDA-Instruct-8B 单步解码粒度扫描实验计划

## Context

**目标**：在 LLaDA-Instruct-8B 上用 DIJA 攻击跑 HarmBench Refined Qwen 数据集，扫描"每步固定解码 N 个 token"五组配置（N=1,2,3,4,5），观察解码粒度对攻击成功率（ASR）的影响。

**两阶段**：
1. **Smoke test**：每组 N 跑前 10 条样本，验证接口与输出格式。
2. **Full run**：每组 N 跑全部 400 条样本。

**关键背景**（来自代码调研）：
- 走 **DIJA 自己的脚本**（用户明确指令，覆盖 AGENTS.md 第 57-60 行默认约定）。
  - 入口：`DIJA/run_harmbench/models/harmbench_llada.py`
  - 解码循环：`DIJA/run_harmbench/utility/generate_function.py::generate_llada`（行 89-145）
- **DIJA 现状**：
  - 行 112：`num_transfer_tokens = 1` 是硬编码常量
  - 循环 `while (x == mask_id).any():`，`--steps / --gen_length` 在函数体内**未被使用**
  - CLI 没有"每步解码 N"参数
- **数据字段**：`harmbench_behaviors_text_all_refined_Qwen.json` 的 `BehaviorID / Behavior / Refined_behavior` 与 `harmbench_llada.py:137-146` 完全对齐，**无需字段映射**
- 模型：`/dev/shm/LLaDA-8B-Instruct`（含 "Instruct" 子串，自动启用 chat template）
- GPU：2 张 V100-SXM2-32GB

---

## Prerequisite 0 — 环境与模型就绪检查（执行前必跑）

```bash
# 1) Python 环境
conda activate dlm_steering

# 2) 模型存在性（按 AGENTS.md 第 17-53 行约定）
[ -d /dev/shm/LLaDA-8B-Instruct ] || python scripts/restore_llada_model.py
[ -d /dev/shm/Llama-Guard-4-12B ] || python scripts/restore_llama_guard_model.py

# 3) 工具检查
which python && python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# 4) DIJA 子项目 .git 状态确认（必读）
cd DIJA && git status && cd -
```

> **DIJA 子项目独立 .git 警告**：本计划会改 `DIJA/run_harmbench/utility/generate_function.py` 和 `DIJA/run_harmbench/models/harmbench_llada.py`，会让 DIJA 子项目变 dirty。**严禁** `cd DIJA && git commit`；改动只在本机使用，不入 DIJA 仓库。

---

## 改动一：让"每步解码 N 个 token"成为可控参数

### 改动 1A — `DIJA/run_harmbench/utility/generate_function.py`

定位 `generate_llada`（行 89-145）：

1. **函数签名**追加形参，默认 1 保持旧行为：
   ```python
   def generate_llada(..., mask_id=126336, tokens_per_step=1):
   ```
2. **行 112** 改为：
   ```python
   num_transfer_tokens = tokens_per_step
   ```
3. **行 137-141**（topk 调用处）加 clamp，避免剩余 mask < N 时 `topk` 抛错：
   ```python
   available = int(mask_index[j].sum().item())
   k = min(num_transfer_tokens, available)
   if k <= 0:
       continue
   select_index = torch.topk(confidence[j], k=k).indices
   ```

### 改动 1B — `DIJA/run_harmbench/models/harmbench_llada.py`

1. **argparse**（行 28-42）追加：
   ```python
   parser.add_argument("--tokens_per_step", type=int, default=1)
   ```
2. **调用 generate_llada 处**（行 87-96）追加 kwarg：
   ```python
   tokens_per_step=args.tokens_per_step,
   ```

### 不做的事（YAGNI）

- ❌ 不改 JailbreakBench / StrongREJECT 同名文件（本次只跑 HarmBench）
- ❌ 不动 `--steps / --gen_length / --mask_counts` 三个 dead arg（保留透传，避免改动外溢）
- ❌ 不改循环为 for 循环（保 KISS）
- ❌ 不复用 `get_num_transfer_tokens`（语义不同）

### 验证逻辑

- N=1 时行为与现状完全等价（默认值兼容）
- N=5 时每步从 mask 位置选置信度 top-5 填充，循环次数 ≈ `ceil(总 mask 数 / 5)`
- 数据集 mask 总数 25-320（mean 93.2，median 90）；N=5 时单条 5-64 步

---

## 改动二：单元测试（放本项目 tests/，不污染 DIJA）

新建 `tests/test_dija_generate_llada_tokens_per_step.py`：
- 路径处理：`sys.path.insert(0, str(ROOT / "DIJA" / "run_harmbench"))` 后 `from utility.generate_function import generate_llada`
- 用例：
  - `test_default_tokens_per_step_equals_one_behavior`：默认调用与旧版常量化路径等价（**N=1 等价性仅靠单测覆盖**，运行时不重复验证）
  - `test_tokens_per_step_5_writes_5_tokens_per_iter`：mock model，断言每次循环 `transfer_index.sum() == min(5, remaining_mask)`
  - `test_tokens_per_step_clamps_when_few_masks_left`：剩 3 个 mask、N=5 时不抛错，本步只写 3 个

运行：
```bash
/root/miniconda3/bin/python -m unittest tests.test_dija_generate_llada_tokens_per_step -v
```

---

## 调度方案：tmux 后台 + 单 GPU 内串行

### 输出目录命名规范

```
outputs/dija_harmbench_llada_instruct/
├── smoke10/
│   ├── k{1..5}/results.json
│   └── logs/k{1..5}.log
└── full400/
    ├── k{1..5}/results.json
    ├── llama_guard/k{1..5}.json
    ├── logs/k{1..5}.log
    └── asr_summary.json
```

### Smoke 切片数据（python，无 jq 依赖）

```bash
ORIG=DIJA/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_Qwen.json
SMOKE=/tmp/harmbench_smoke10.json
python -c "
import json, sys
data = json.load(open(sys.argv[1]))
json.dump(data[:10], open(sys.argv[2], 'w'), ensure_ascii=False, indent=2)
" "$ORIG" "$SMOKE"
```

### Smoke 调度(GPU0 单卡串行,5 组 10 条,预计 10-30 分钟)

> **显存说明**：LLaDA-8B fp16 ≈ 17-18 GB，V100-32GB 同卡并发 2 个进程会 OOM；所有阶段统一**单 GPU 内串行**。

```bash
tmux new-session -d -s dija_llada_smoke "
set -e
for N in 1 2 3 4 5; do
  OUT=outputs/dija_harmbench_llada_instruct/smoke10/k\${N}
  mkdir -p \$OUT outputs/dija_harmbench_llada_instruct/smoke10/logs
  CUDA_VISIBLE_DEVICES=0 python DIJA/run_harmbench/models/harmbench_llada.py \
    --model_path /dev/shm/LLaDA-8B-Instruct \
    --attack_prompt /tmp/harmbench_smoke10.json \
    --output_json \$OUT/results.json \
    --attack_method DIJA \
    --tokens_per_step \$N \
    --steps 128 --gen_length 128 --mask_counts 36 \
    > outputs/dija_harmbench_llada_instruct/smoke10/logs/k\${N}.log 2>&1
done
"
```

### Full 调度(GPU0 跑 N=1,2,3;GPU1 跑 N=4,5;卡内串行)

```bash
# GPU 0 串行：N=1 → N=2 → N=3
tmux new-session -d -s dija_llada_full_gpu0 "
set -e
DATA=DIJA/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_Qwen.json
for N in 1 2 3; do
  OUT=outputs/dija_harmbench_llada_instruct/full400/k\${N}
  mkdir -p \$OUT
  CUDA_VISIBLE_DEVICES=0 python DIJA/run_harmbench/models/harmbench_llada.py \
    --model_path /dev/shm/LLaDA-8B-Instruct \
    --attack_prompt \$DATA \
    --output_json \$OUT/results.json \
    --attack_method DIJA \
    --tokens_per_step \$N \
    --steps 128 --gen_length 128 --mask_counts 36 \
    > outputs/dija_harmbench_llada_instruct/full400/logs/k\${N}.log 2>&1
done
"

# GPU 1 串行：N=4 → N=5
tmux new-session -d -s dija_llada_full_gpu1 "
set -e
DATA=DIJA/run_harmbench/refine_prompt/harmbench_behaviors_text_all_refined_Qwen.json
for N in 4 5; do
  OUT=outputs/dija_harmbench_llada_instruct/full400/k\${N}
  mkdir -p \$OUT
  CUDA_VISIBLE_DEVICES=1 python DIJA/run_harmbench/models/harmbench_llada.py \
    --model_path /dev/shm/LLaDA-8B-Instruct \
    --attack_prompt \$DATA \
    --output_json \$OUT/results.json \
    --attack_method DIJA \
    --tokens_per_step \$N \
    --steps 128 --gen_length 128 --mask_counts 36 \
    > outputs/dija_harmbench_llada_instruct/full400/logs/k\${N}.log 2>&1
done
"
```

### 时间估算(V100 fp16 口径,保守上界)

| N | 单条 forward 次数(mean) | 单条耗时 | 400 条总时长 |
|---|---|---|---|
| 1 | ≈93 | 47-93s | **5-11 h** |
| 2 | ≈47 | 24-47s | 2.5-5 h |
| 3 | ≈31 | 16-31s | 1.7-3.4 h |
| 4 | ≈24 | 12-24s | 1.3-2.7 h |
| 5 | ≈19 | 10-19s | 1.1-2.2 h |

- GPU0 wall clock 上界 ≈ 5+5+3.4 ≈ **13.4 h**
- GPU1 wall clock 上界 ≈ 2.7+2.2 ≈ **4.9 h**
- 监控按 AGENTS.md "默认 30 分钟一次"

---

## 评测方案(本项目 Llama-Guard)

**推荐口径**：`scripts/eval_llama_guard_local.py` + `/dev/shm/Llama-Guard-4-12B`。
**理由**：与其他 Stage 横向对比、本机现成、不依赖 OpenAI key。
**ASR-k(关键词)** 可作 secondary 跟跑(DIJA 自带,无需模型)。

### 字段映射 helper

DIJA `results.json` 字段：`BehaviorID / Behavior / Refined_behavior / response`。
Llama-Guard 默认 schema：`prompt / response`。
评测前用 python 重命名:

```bash
for N in 1 2 3 4 5; do
  SRC=outputs/dija_harmbench_llada_instruct/full400/k${N}/results.json
  DST=outputs/dija_harmbench_llada_instruct/full400/k${N}/results_for_guard.json
  python -c "
import json, sys
src = json.load(open(sys.argv[1]))
out = [
    {'prompt': r['Behavior'], 'response': r['response'],
     'refined': r.get('Refined_behavior', ''), 'id': r['BehaviorID']}
    for r in src
]
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=2)
" "$SRC" "$DST"
done
```

### Llama-Guard 评测命令(tmux,单 GPU 串行)

> 首次跑时先 `python scripts/eval_llama_guard_local.py --help` 确认 flag 名(`--data_path / --model_path / --output_path` 是本项目惯用命名)。

```bash
tmux new-session -d -s dija_llada_guard "
set -e
for N in 1 2 3 4 5; do
  python scripts/eval_llama_guard_local.py \
    --data_path outputs/dija_harmbench_llada_instruct/full400/k\${N}/results_for_guard.json \
    --model_path /dev/shm/Llama-Guard-4-12B \
    --output_path outputs/dija_harmbench_llada_instruct/full400/llama_guard/k\${N}.json \
    > outputs/dija_harmbench_llada_instruct/full400/logs/guard_k\${N}.log 2>&1
done
"
```

### 汇总 ASR

```bash
python -c "
import json, glob
res = {}
for f in sorted(glob.glob('outputs/dija_harmbench_llada_instruct/full400/llama_guard/k*.json')):
    d = json.load(open(f))
    n = int(f.rsplit('k', 1)[1].split('.')[0])
    unsafe = sum(1 for r in d if r.get('is_unsafe'))
    res[f'k{n}'] = {'unsafe': unsafe, 'total': len(d), 'asr': unsafe / len(d)}
print(json.dumps(res, indent=2))
" > outputs/dija_harmbench_llada_instruct/full400/asr_summary.json
```

---

## 验收/退出标准

### Smoke

- ✅ `smoke10/k{1..5}/results.json` 各 10 条
- ✅ 每条 4 字段齐：`BehaviorID / Behavior / Refined_behavior / response`
- ✅ `response` 非空、长度 >20 字符、不全是 `<|mdm_mask|>`
- ✅ logs 无 `topk out of range / OOM / device-side assert`

### Full

- ✅ `full400/k{1..5}/results.json` 各 400 条
- ✅ 5 组 `llama_guard/k{1..5}.json` 完成
- ✅ `asr_summary.json` 含 5 组 unsafe_count + ASR
- ✅ 轻量 markdown 表(自由长度):5 组 ASR 折线 + 推理耗时对比 + 2-3 个 N=1 vs N=5 case study

---

## 风险与注意点

1. **AGENTS.md 第 57-60 行默认禁用 DIJA 自带脚本** —— 本次按用户明确指令覆盖；不入 git。
2. **DIJA 子项目独立 .git** —— 改动后**严禁** `cd DIJA && git commit`。
3. **`num_transfer_tokens=1` 是 DIJA 现状** —— 与 LLaDA 官方"等分"算法不同,本计划保留 DIJA 语义,仅参数化。
4. **`--steps / --gen_length / --mask_counts` 是 dead args** —— 传值不影响结果,仅为日志可追溯。
5. **clamp 必须做** —— 否则剩余 mask < N 时 `topk` 抛 `RuntimeError`。
6. **显存 OOM 风险** —— LLaDA-8B fp16 + 序列 ≈ 18-20 GB；**单 GPU 同时只能跑 1 个进程**。
7. **断点续跑不支持** —— DIJA 一次性写整个 `output_json`,中途 OOM/挂掉需要从 0 重跑该组 N 的全部 400 条。建议每组完成立刻 `cp` 一份到 `*.done.json` 防误删。
8. **不主动 commit** —— 按用户长期约定,本次实验改动与脚本均不进 git,除非明确要求。
9. **进度监控** —— smoke 阶段 ≤30 分钟可一次性等完；full 阶段每 30 分钟 `tmux ls + tail logs`。

---

## 文件清单

| 路径 | 性质 | 行数估计 |
|---|---|---|
| `DIJA/run_harmbench/utility/generate_function.py` | 改 `generate_llada` 签名 + L112 + clamp | +5 / -1 |
| `DIJA/run_harmbench/models/harmbench_llada.py` | argparse + 调用透传 | +2 / 0 |
| `tests/test_dija_generate_llada_tokens_per_step.py` | 新增单测(本项目 tests/,**不**放 DIJA 内) | +60 |
| `scripts/run_dija_harmbench_decode_scan.sh` | 新增调度脚本(封装上文 tmux 命令) | +50 |
| `scripts/postprocess_dija_results_for_guard.py` | 新增字段映射 helper | +20 |
| `outputs/dija_harmbench_llada_instruct/` | 新增输出目录 | 数据 |
| `docs/dija_harmbench_decode_scan_metrics.md` | 轻量 metrics 表(自由长度) | 视实验结果 |

**严格不动**：
- `eval_llada_steering.py`(本次不走本项目入口)
- `utils/` / `tests/`(除新增单测外)
- `DIJA/run_jailbreakbench/` / `DIJA/run_strongreject/`(YAGNI)
