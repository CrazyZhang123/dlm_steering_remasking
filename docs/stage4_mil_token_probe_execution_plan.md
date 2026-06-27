# Stage 4 MIL Token Probe Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变推理公式的前提下，把 MIL token probe 作为离线 harmful token selection 模块接入 CT-CSD 与 Category-aware CT-CSD。

**Architecture:** Stage 4 只新增一个变量：构造 bank 前用 linear MIL probe 过滤 harmful response tokens。推理阶段仍只加载 `ct_csd_bank.pt`，继续使用 `CTCSDBank` 的 route、alignment、steer 逻辑，不在 `eval_llada_steering.py` 中运行 probe。

**Tech Stack:** Python, PyTorch, Transformers, scikit-learn `MiniBatchKMeans`, `unittest`, LLaDA hidden-state hooks.

---

## 1. 边界与进入条件

进入条件：

```text
Stage 2 已确定默认簇数 M*
Stage 3 的 category_ct_csd bank 已能构造、加载、推理
utils/make_ct_csd_llada.py 的 ct_csd 与 category_ct_csd 单测通过
harmful_json 样本至少包含 prompt、response，优先包含 semantic_category
```

本阶段只做：

```text
训练 MIL token probe
用 probe threshold 过滤 harmful tokens
生成 probe_ct_csd 与 probe_category_ct_csd bank
记录 MIL token selection 诊断
完成 no-probe vs probe 对照实验
```

本阶段不做：

```text
不改变推理公式
不在推理时运行 probe
不修改 eval_llada_steering.py 的 bank 加载逻辑
不把 MIL probe 作为唯一主方法
不移除 no-probe 对照
不执行 git commit、git push、git reset
```

## 2. 文件职责

| 文件 | 操作 | 职责 |
|---|---|---|
| `utils/train_mil_token_probe_llada.py` | 新增 | 训练 linear MIL token probe，保存权重、配置和验证指标 |
| `utils/make_ct_csd_llada.py` | 修改 | 增加 `probe_ct_csd`、`probe_category_ct_csd` method，并把 probe 分数作为 harmful token 过滤条件 |
| `utils/ct_csd_bank.py` | 不改或只补兼容测试 | 保持推理时 bank route、alignment、steer 的单一职责 |
| `eval_llada_steering.py` | 不改 | 推理只加载 Stage 4 生成的 bank，不加载 probe |
| `tests/test_train_mil_token_probe_llada.py` | 新增 | 覆盖 top-q pooling、bag loss、probe 保存格式、验证指标 |
| `tests/test_make_ct_csd_llada.py` | 修改 | 覆盖 probe token selection、probe method 分支、bank MIL metadata |
| `tests/test_ct_csd_bank.py` | 可选修改 | 验证带 `mil.enabled=true` 的 bank 仍按普通 CT-CSD bank 推理 |

## 3. 数据契约

MIL probe 保存格式：

```python
mil_probe_state = {
    "format": "mil_token_probe_v1",
    "model_family": "llada",
    "target_layer": 31,
    "input_dim": hidden_size,
    "top_q_ratio": 0.1,
    "state_dict": probe.state_dict(),
    "config": {
        "model_path": "/dev/shm/LLaDA-8B-Instruct",
        "harmful_json": "data/harmbench_csd_train.json",
        "refusals_txt": "utils/refusals.txt",
        "max_response_len": 128,
        "max_total_len": 2048,
        "seed": 42,
    },
    "metrics": {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
        "val_auc": 0.0,
    },
}
```

Stage 4 bank 的 `mil` 字段：

```python
ct_csd_bank_state["mil"] = {
    "enabled": True,
    "probe_path": "outputs/mil_token_probe_llada.pt",
    "probe_threshold": 0.7,
    "top_q_ratio": 0.1,
}
```

Probe 只影响 harmful token selection：

```text
safe_mean = 所有有效 safe response token 的 sample-balanced global mean
cluster_sums = probe 选中的 harmful token hidden state 聚类后累加
cluster_counts = probe 选中的 harmful token 数
```

---

## 4. Task 4.1：训练脚本骨架与纯函数单测

**Files:**
- Create: `tests/test_train_mil_token_probe_llada.py`
- Create: `utils/train_mil_token_probe_llada.py`

- [ ] **Step 1: 写 top-q pooling 单测**

在 `tests/test_train_mil_token_probe_llada.py` 中覆盖：

```python
def test_top_q_pool_logits_uses_highest_scores():
    logits = torch.tensor([0.1, 0.9, 0.2, 0.8])
    pooled = trainer.top_q_pool_logits(logits, top_q_ratio=0.5)
    assert torch.allclose(pooled, torch.tensor(0.85))


def test_top_q_pool_logits_keeps_at_least_one_token():
    logits = torch.tensor([0.4])
    pooled = trainer.top_q_pool_logits(logits, top_q_ratio=0.1)
    assert torch.allclose(pooled, torch.tensor(0.4))
```

- [ ] **Step 2: 写 bag loss 与 state 格式单测**

测试要求：

```text
compute_bag_loss(probe, hidden, label, top_q_ratio) 返回 scalar tensor
label=1.0 与 label=0.0 都可计算 BCEWithLogitsLoss
build_probe_state(...) 包含 format、model_family、target_layer、top_q_ratio、state_dict、config、metrics
import utils.train_mil_token_probe_llada 不触发模型加载
```

- [ ] **Step 3: 实现训练脚本纯函数**

在 `utils/train_mil_token_probe_llada.py` 中实现：

```python
class LinearMILProbe(torch.nn.Module):
    def __init__(self, input_dim: int) -> None: ...
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: ...


def top_q_pool_logits(logits: torch.Tensor, top_q_ratio: float) -> torch.Tensor: ...


def compute_bag_logit(
    probe: LinearMILProbe,
    hidden: torch.Tensor,
    top_q_ratio: float,
) -> torch.Tensor: ...


def compute_bag_loss(
    probe: LinearMILProbe,
    hidden: torch.Tensor,
    label: float,
    top_q_ratio: float,
) -> torch.Tensor: ...


def build_probe_state(
    probe: LinearMILProbe,
    args: argparse.Namespace,
    metrics: dict[str, float],
    input_dim: int,
) -> dict: ...
```

- [ ] **Step 4: 运行纯函数测试**

Run:

```bash
python -m unittest tests.test_train_mil_token_probe_llada
```

Expected:

```text
OK
```

## 5. Task 4.2：MIL 训练数据构造与训练闭环

**Files:**
- Modify: `utils/train_mil_token_probe_llada.py`
- Modify: `tests/test_train_mil_token_probe_llada.py`

- [ ] **Step 1: 复用现有 LLaDA token hidden-state 工具**

从 `utils.make_ct_csd_llada` 复用：

```python
from utils.make_ct_csd_llada import (
    build_sequence,
    extract_target_layer_tokens,
    filter_response_hidden_states,
    load_harmful_data,
    load_refusals,
    resolve_path,
    set_seed,
)
```

- [ ] **Step 2: 实现 bag 构造**

实现 `iter_mil_bags(model, tokenizer, harmful, refusals, args, device)`：

```text
每条 harmful response 生成一个 label=1.0 的 bag
同一 prompt 随机配一个 refusal response，生成一个 label=0.0 的 bag
harmful 与 safe bag 都过滤 special token 与空白 token
超过 max_total_len 的样本跳过
过滤后无 token 的 bag 跳过并计入 skipped_bags
```

- [ ] **Step 3: 实现 train / validation 切分**

规则：

```text
默认 val_ratio=0.1
使用 seed 打乱 bag index
保持 harmful / safe bag 数量接近 1:1
如果 validation split 为空，直接报错提示降低 val_ratio 或增加样本
```

- [ ] **Step 4: 实现训练循环**

训练配置：

```text
optimizer = torch.optim.AdamW
lr = 1e-3
epochs = 5
weight_decay = 0.01
loss = BCEWithLogitsLoss
pooling = top-q pooling
```

每个 epoch 记录：

```text
train_loss
val_loss
val_accuracy
val_auc
```

- [ ] **Step 5: 实现 CLI**

CLI 参数：

```text
--model_path
--harmful_json
--refusals_txt
--output_path
--target_layer
--top_q_ratio
--max_response_len
--max_total_len
--epochs
--lr
--weight_decay
--val_ratio
--max_samples
--device
--seed
```

- [ ] **Step 6: 保存训练产物**

输出：

```text
outputs/mil_token_probe_llada.pt
outputs/mil_token_probe_llada_metrics.json
```

`metrics.json` 至少包含：

```text
train_loss
val_loss
val_accuracy
val_auc
skipped_bags
train_bags
val_bags
```

- [ ] **Step 7: 跑训练命令**

Run:

```bash
python utils/train_mil_token_probe_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_path outputs/mil_token_probe_llada.pt \
  --target_layer 31 \
  --top_q_ratio 0.1 \
  --max_response_len 128 \
  --max_total_len 2048 \
  --epochs 5 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  --val_ratio 0.1 \
  --device cuda \
  --seed 42
```

Expected:

```text
outputs/mil_token_probe_llada.pt 可被 torch.load(..., weights_only=True) 加载
probe state 的 target_layer 与训练参数一致
metrics 中 train_loss、val_loss、val_accuracy、val_auc 都存在
```

## 6. Task 4.3：把 probe 接入 CT-CSD 离线构造

**Files:**
- Modify: `utils/make_ct_csd_llada.py`
- Modify: `tests/test_make_ct_csd_llada.py`
- Optional Modify: `tests/test_ct_csd_bank.py`

- [ ] **Step 1: 先写 probe token selection 测试**

新增测试覆盖：

```text
filter_response_hidden_states 保持旧行为
新 helper 返回过滤后的 hidden 与对应 response token id
probe 分数低于 probe_threshold 的 harmful token 不进入 kmeans
safe response token 不经过 probe，仍用于 sample-balanced global safe mean
```

- [ ] **Step 2: 增加 token filtering helper**

在 `utils/make_ct_csd_llada.py` 中新增：

```python
def filter_response_tokens(
    tokenizer,
    response_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

保留旧接口：

```python
def filter_response_hidden_states(tokenizer, response_ids, hidden):
    filtered_hidden, _filtered_token_ids = filter_response_tokens(tokenizer, response_ids, hidden)
    return filtered_hidden
```

- [ ] **Step 3: 增加 probe 加载与打分 helper**

新增：

```python
def load_mil_probe(path: Path, target_layer: int, device: torch.device):
    ...


def score_tokens_with_probe(probe, hidden: torch.Tensor) -> torch.Tensor:
    ...


def apply_probe_threshold(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

校验规则：

```text
probe format 必须是 mil_token_probe_v1
model_family 必须是 llada
probe target_layer 必须等于 make_ct_csd_llada.py 的 --target_layer
probe_threshold 必须在 [0.0, 1.0]
```

- [ ] **Step 4: 扩展 CLI**

修改 `--method`：

```text
choices=["ct_csd", "category_ct_csd", "probe_ct_csd", "probe_category_ct_csd"]
```

新增参数：

```text
--mil_probe_path
--probe_threshold
```

- [ ] **Step 5: 重构构造分支**

分支行为：

```text
ct_csd：沿用原始 global clustering
probe_ct_csd：沿用 global clustering，但 harmful tokens 先过 probe threshold
category_ct_csd：沿用 Stage 3 category clustering
probe_category_ct_csd：沿用 Stage 3 category clustering，但每个 category 内的 harmful tokens 先过 probe threshold
```

空样本处理：

```text
某条 harmful response 经 probe 后没有 token，跳过该样本并计入 probe_empty_samples
某个 category 经 probe 后 token 数不足以满足 K_c，构造阶段直接报错
最终任何 cluster 为空时沿用 build_bank_state_from_cluster_sums 的报错
```

- [ ] **Step 6: 写入 bank metadata**

写入：

```python
state["config"]["token_selection"] = "mil_probe_threshold"
state["config"]["probe_empty_samples"] = int(probe_empty_samples)
state["mil"]["enabled"] = True
state["mil"]["probe_path"] = str(mil_probe_path)
state["mil"]["probe_threshold"] = float(args.probe_threshold)
state["mil"]["top_q_ratio"] = float(probe_state["top_q_ratio"])
```

- [ ] **Step 7: 运行构造相关测试**

Run:

```bash
python -m unittest tests.test_make_ct_csd_llada tests.test_ct_csd_bank
```

Expected:

```text
OK
```

## 7. Task 4.4：MIL token selection 诊断输出

**Files:**
- Modify: `utils/make_ct_csd_llada.py`
- Modify: `tests/test_make_ct_csd_llada.py`

- [ ] **Step 1: 收集 retention 统计**

probe 分支记录：

```text
total_harmful_tokens_before_probe
total_harmful_tokens_after_probe
global_retention_ratio
per_response_retention_ratio
per_category_tokens_before_probe
per_category_tokens_after_probe
per_category_retention_ratio
probe_empty_samples
```

- [ ] **Step 2: 收集高分 token 样例**

每条样例记录：

```text
sample_index
category
token_text
token_id
probe_score
selected_by_threshold
```

- [ ] **Step 3: 写出诊断文件**

输出：

```text
mil_token_selection_summary.json
mil_high_score_tokens.md
```

summary 内容：

```text
global retention
category retention
response retention 的 min、p25、median、p75、max
probe_threshold
probe_path
top_q_ratio
```

- [ ] **Step 4: 增加 cluster 高频 token 诊断**

所有 `make_ct_csd_llada.py` methods 都写出：

```text
cluster_token_top_terms.md
```

内容：

```text
global_cluster_id
category
local_cluster_id
cluster_size
top 20 decoded token texts
```

该文件只做诊断，不写入 bank，不影响推理。

- [ ] **Step 5: 运行诊断测试**

Run:

```bash
python -m unittest tests.test_make_ct_csd_llada
```

Expected:

```text
OK
```

## 8. Task 4.5：构造 probe variants

**Files:**
- No source edits after Task 4.4
- Outputs: `outputs/probe_ct_csd_llada_m{MSTAR}_tau07/`
- Outputs: `outputs/probe_category_ct_csd_llada_m{MSTAR}_tau07/`

- [ ] **Step 1: 构造 Probe-CT-CSD**

Run:

```bash
MSTAR=8

python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_dir "outputs/probe_ct_csd_llada_m${MSTAR}_tau07" \
  --target_layer 31 \
  --method probe_ct_csd \
  --mil_probe_path outputs/mil_token_probe_llada.pt \
  --probe_threshold 0.7 \
  --num_total_clusters "${MSTAR}" \
  --max_response_len 128 \
  --max_total_len 2048 \
  --device cuda \
  --seed 42
```

- [ ] **Step 2: 构造 Probe-Category-aware CT-CSD**

Run:

```bash
MSTAR=8

python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_dir "outputs/probe_category_ct_csd_llada_m${MSTAR}_tau07" \
  --target_layer 31 \
  --method probe_category_ct_csd \
  --mil_probe_path outputs/mil_token_probe_llada.pt \
  --probe_threshold 0.7 \
  --num_total_clusters "${MSTAR}" \
  --category_key semantic_category \
  --max_response_len 128 \
  --max_total_len 2048 \
  --device cuda \
  --seed 42
```

- [ ] **Step 3: 构造阈值消融**

Run:

```bash
MSTAR=8

for TAU in 0.5 0.7 0.9; do
  case "${TAU}" in
    0.5) TAU_TAG="tau05" ;;
    0.7) TAU_TAG="tau07" ;;
    0.9) TAU_TAG="tau09" ;;
  esac

  python utils/make_ct_csd_llada.py \
    --model_path /dev/shm/LLaDA-8B-Instruct \
    --harmful_json data/harmbench_csd_train.json \
    --refusals_txt utils/refusals.txt \
    --output_dir "outputs/probe_category_ct_csd_llada_m${MSTAR}_${TAU_TAG}" \
    --target_layer 31 \
    --method probe_category_ct_csd \
    --mil_probe_path outputs/mil_token_probe_llada.pt \
    --probe_threshold "${TAU}" \
    --num_total_clusters "${MSTAR}" \
    --category_key semantic_category \
    --max_response_len 128 \
    --max_total_len 2048 \
    --device cuda \
    --seed 42
done
```

- [ ] **Step 4: 验证 bank 输出**

检查：

```text
每个输出目录都有 ct_csd_bank.pt
每个输出目录都有 ct_csd_bank_summary.json
每个 probe 输出目录都有 mil_token_selection_summary.json
每个 probe 输出目录都有 mil_high_score_tokens.md
每个输出目录都有 cluster_token_top_terms.md
probe variants 的 state["mil"]["enabled"] 为 True
probe_category_ct_csd 的 category_cluster_counts 总和等于 M*
```

## 9. Task 4.6：推理评估与结果汇总

**Files:**
- Create: `stage4_mil_probe_metrics.md`
- Create: `stage4_probe_cluster_token_comparison.md`

评估矩阵：

| 方法 | bank | 变量 |
|---|---|---|
| CT-CSD | `outputs/ct_csd_llada_m{MSTAR}/ct_csd_bank.pt` | no-probe |
| Probe-CT-CSD | `outputs/probe_ct_csd_llada_m{MSTAR}_tau07/ct_csd_bank.pt` | probe |
| Category-aware CT-CSD | `outputs/category_ct_csd_llada_m{MSTAR}/ct_csd_bank.pt` | no-probe |
| Probe-Category-aware CT-CSD | `outputs/probe_category_ct_csd_llada_m{MSTAR}_tau07/ct_csd_bank.pt` | probe |

- [ ] **Step 1: 跑 Probe-Category-aware CT-CSD 推理**

Run:

```bash
MSTAR=8

python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --generated_samples_path "outputs/jbb_dija_probe_category_ct_csd_m${MSTAR}_tau07" \
  --batch_size 32 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --remasking low_confidence \
  --sampler steering \
  --remdm_number 4 \
  --cfg 0 \
  --device cuda:0 \
  --self_reminder False \
  --steering_vector_path "outputs/probe_category_ct_csd_llada_m${MSTAR}_tau07/ct_csd_bank.pt" \
  --steering_overshoot 1.0 \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --max_refinement_iters 5 \
  --initial_steering_ratio 0.1
```

- [ ] **Step 2: 汇总指标**

`stage4_mil_probe_metrics.md` 至少包含：

```text
训练指标：train_loss、val_loss、val_accuracy、val_auc
选择指标：global_retention_ratio、per_category_retention_ratio、probe_empty_samples
安全指标：ASR、unsafe_count
质量指标：refusal rate、over-refusal、平均输出长度、空回复比例
干预指标：activation_rate、平均被 steering token 数、平均 remask token 数
效率指标：推理耗时、bank route 耗时、额外显存
```

- [ ] **Step 3: 汇总 cluster token 对比**

`stage4_probe_cluster_token_comparison.md` 对比：

```text
outputs/ct_csd_llada_m{MSTAR}/cluster_token_top_terms.md
outputs/probe_ct_csd_llada_m{MSTAR}_tau07/cluster_token_top_terms.md
outputs/category_ct_csd_llada_m{MSTAR}/cluster_token_top_terms.md
outputs/probe_category_ct_csd_llada_m{MSTAR}_tau07/cluster_token_top_terms.md
```

标出：

```text
probe 后消失或降权的模板 token
probe 后消失或降权的空白 token
probe 后消失或降权的标点 token
probe 后保留或升权的 harmful semantics token
```

- [ ] **Step 4: 确认单变量对照**

检查：

```text
四个主对照使用同一 M*
四个主对照使用同一 target_layer
四个主对照使用同一 alignment_threshold
四个主对照使用同一 steering_overshoot
probe 组与 no-probe 组只差 harmful token selection
eval_llada_steering.py 没有新增 probe 加载逻辑
每个 eval 输出目录都有 ct_csd_diagnostics.json
```

## 10. Task 4.7：Stage 4 决策规则

完成 Stage 4 后按以下规则决定是否进入 Stage 5：

| 判断项 | 通过条件 | 不通过时处理 |
|---|---|---|
| 工程闭环 | probe 训练、probe bank 构造、推理评估都跑通 | 先修训练或构造链路，不进入 Dream |
| 接口兼容 | `ct_csd_bank.pt` 仍是 `ct_csd_v1`，`CTCSDBank` 无需知道 probe 细节 | 回退 bank metadata，不改推理类职责 |
| 单变量对照 | probe 组与 no-probe 组只差 harmful token selection | 重跑混入额外变量的实验 |
| token 诊断 | 高分 token、retention ratio、cluster 高频 token 能解释结果变化 | 补诊断文件，不直接进入 Stage 5 |
| 指标收益 | Probe-Category-aware CT-CSD 相比 Category-aware CT-CSD 至少不显著伤害质量，且 ASR 或干预效率有改善 | 将 MIL 标记为诊断或负结果，不作为主方法前提 |

## 11. 最终验证命令

文档级验证：

```bash
rg -n "Task 4\\.[1-7]" docs/stage4_mil_token_probe_execution_plan.md
awk '/^```/ {count++} END {print count; exit count % 2}' docs/stage4_mil_token_probe_execution_plan.md
```

实现后最小测试：

```bash
python -m unittest tests.test_train_mil_token_probe_llada
python -m unittest tests.test_make_ct_csd_llada tests.test_ct_csd_bank
```

Stage 4 完成定义：

```text
原始 plan 文档未被 Stage 4 具体执行计划污染
独立执行计划文档存在且可按任务执行
probe 训练产物可加载
probe_ct_csd 与 probe_category_ct_csd bank 可加载
推理阶段不运行 probe
no-probe vs probe 对照完成
三类诊断文件齐全：MIL retention、高分 token、cluster 高频 token
```
