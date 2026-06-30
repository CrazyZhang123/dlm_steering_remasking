# Stage 4 MIL Token Probe Progress

更新时间：2026-06-29T00:00:00+00:00

本文档记录 Stage 4 的代码实现、验证状态、实验进度和当前风险判断。原始总计划 `docs/category_aware_ct_csd_mil_plan.md` 未在本阶段进度整理中修改。

## 1. 当前结论

Stage 4 的工程代码已经提交到 `main`：

```text
3e8fdf3 feat: add stage4 mil token probe
```

当前实验状态：

```text
已完成：MIL token probe 训练与产物保存
已完成：Probe-CT-CSD bank 构造
已完成：Probe-Category-aware CT-CSD bank 构造
已完成：Probe-Category-aware CT-CSD M16 tau=0.7 direct 生成与 Llama Guard 评判
已完成：Probe-Category-aware CT-CSD M16 tau=0.7 普通 pipeline 生成与 CT-CSD 诊断
已完成：Probe-Category-aware CT-CSD M16 tau=0.7 普通 pipeline Llama Guard 评判
```

Stage 4 当前沿用 Stage 3 的 `M=16` no-probe 对照，不使用执行计划示例里的 `M=8`。需要注意：阶段 1 簇数消融完成后，当前最佳已验证 CT-CSD 点位是 `M=12`；Stage 4 继续保留 `M=16` 是为了和 Stage 3 Category-aware CT-CSD M16 做单变量对照，只改变 harmful token selection。

## 2. 方法边界

Stage 4 只把 MIL token probe 接入离线 bank 构造阶段：

```text
harmful response tokens -> linear MIL probe scoring -> threshold filtering -> CT-CSD / Category-aware CT-CSD clustering
```

推理阶段仍只加载 `ct_csd_bank.pt`：

```text
eval_llada_steering.py 不加载 probe
CTCSDBank 不知道 probe 细节
steering 公式、route、alignment、remask/refinement 逻辑不变
```

因此 Stage 4 的单变量对照口径是：

```text
no-probe: harmful tokens 全部进入 clustering
probe: 只有 probe_score >= 0.7 的 harmful tokens 进入 clustering
safe response tokens: 不经过 probe，仍用于 sample-balanced global safe mean
```

## 3. 代码进度

### 3.1 `utils/train_mil_token_probe_llada.py`

新增 MIL token probe 训练脚本，主要职责：

```text
LinearMILProbe: 单层 linear probe，输入 token hidden state，输出 token logit
top_q_pool_logits: bag-level top-q pooling，至少保留一个 token
compute_bag_logit / compute_bag_loss: MIL bag-level BCEWithLogits 训练目标
build_probe_state: 保存 probe 格式、配置、metrics 和 state_dict
iter_mil_bags / build_mil_bags: harmful bag 与 safe refusal bag 构造
split_bags: 按 label 分层切分 train / validation
binary_auc / evaluate_probe: validation loss、accuracy、AUC
train_probe: AdamW 训练闭环
CLI: 支持 model_path、harmful_json、refusals_txt、output_path、target_layer、top_q_ratio、max_response_len、max_total_len、epochs、lr、weight_decay、val_ratio、max_samples、device、seed
```

训练数据构造口径：

```text
正 bag: HarmBench harmful response
负 bag: 同一 prompt 配一个 refusal paraphrase
hidden 来源: LLaDA target_layer=31 response tokens
过滤: special token、空白 token 过滤；超过 max_total_len 的样本跳过
```

保存产物格式：

```text
outputs/mil_token_probe_llada.pt
outputs/mil_token_probe_llada_metrics.json
```

### 3.2 `utils/make_ct_csd_llada.py`

在 Stage 3 bank 构造脚本上扩展 probe 分支，主要新增或扩展职责：

```text
filter_response_tokens: 返回过滤后的 response hidden states 与 token ids
filter_response_hidden_states: 保持旧接口，复用 filter_response_tokens
load_mil_probe: 加载并校验 mil_token_probe_v1、model_family=llada、target_layer
score_tokens_with_probe: 对 harmful token hidden states 打 sigmoid 分数
apply_probe_threshold: 只保留 score >= probe_threshold 的 harmful tokens
new_probe_diagnostics / record_probe_selection / write_probe_diagnostics: 输出 MIL retention 与高分 token 诊断
new_cluster_token_terms / record_cluster_token_terms / write_cluster_token_terms: 输出 cluster 高频 token 诊断
iter_valid_sample_tokens: 在样本级统一处理 safe/harmful token 抽取、probe 过滤和诊断记录
fit_minibatch_kmeans: 支持 ct_csd 与 probe_ct_csd
fit_category_minibatch_kmeans: 支持 category_ct_csd 与 probe_category_ct_csd
accumulate_cluster_sums / accumulate_category_cluster_sums: probe 后 token 进入 cluster sum
write_bank_summary: 输出 bank summary
CLI method choices: ct_csd、category_ct_csd、probe_ct_csd、probe_category_ct_csd
CLI probe 参数: --mil_probe_path、--probe_threshold
```

Stage 4 bank metadata 增加：

```python
state["config"]["token_selection"] = "mil_probe_threshold"
state["config"]["probe_empty_samples"] = int(...)
state["mil"]["enabled"] = True
state["mil"]["probe_path"] = "outputs/mil_token_probe_llada.pt"
state["mil"]["probe_threshold"] = 0.7
state["mil"]["top_q_ratio"] = 0.1
```

probe 输出目录应包含：

```text
ct_csd_bank.pt
ct_csd_bank_summary.json
mil_token_selection_summary.json
mil_high_score_tokens.md
cluster_token_top_terms.md
```

### 3.3 测试代码

`tests/test_train_mil_token_probe_llada.py` 覆盖：

```text
top-q pooling 只聚合最高分 token
top-q pooling 至少保留一个 token
compute_bag_loss 对 label=1/0 都返回 scalar tensor
build_probe_state 保存格式字段
split_bags 保持 train/validation 都有正负样本
evaluate_probe 返回 loss、accuracy、AUC
main 在 mock 数据上保存 probe 与 metrics
```

`tests/test_make_ct_csd_llada.py` 覆盖：

```text
special/blank token 过滤
filter_response_tokens 返回 hidden 与 token ids
apply_probe_threshold 只保留高分 token
target layer hook 提取与异常清理
CT-CSD 构造、safe mean、cluster sum
probe_ct_csd bank metadata
category_ct_csd bank metadata
probe_category_ct_csd bank metadata
MIL selection summary 与 high-score token markdown
cluster_token_top_terms 输出
```

最新最小验证：

```bash
python -m unittest tests.test_train_mil_token_probe_llada tests.test_make_ct_csd_llada
```

结果：

```text
Ran 31 tests in 0.059s
OK
```

## 4. 已完成实验

### 4.1 MIL token probe 训练

训练命令在 `tmux` session 中运行：

```text
stage4_mil_probe_train
```

关键参数：

```text
model_path: /dev/shm/LLaDA-8B-Instruct
harmful_json: .worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json
refusals_txt: utils/refusals.txt
target_layer: 31
top_q_ratio: 0.1
max_response_len: 128
max_total_len: 2048
epochs: 5
lr: 1e-3
weight_decay: 0.01
val_ratio: 0.1
device: CUDA_VISIBLE_DEVICES=1, --device cuda
seed: 42
```

训练产物：

```text
outputs/mil_token_probe_llada.pt
outputs/mil_token_probe_llada_metrics.json
outputs/stage4_mil_probe_train.log
```

probe state 摘要：

```json
{
  "format": "mil_token_probe_v1",
  "model_family": "llada",
  "target_layer": 31,
  "input_dim": 4096,
  "top_q_ratio": 0.1
}
```

训练指标：

```json
{
  "train_loss": 8.790893101923604e-31,
  "val_loss": 6.666953987881271e-33,
  "val_accuracy": 1.0,
  "val_auc": 1.0,
  "skipped_bags": 0.0,
  "train_bags": 17290.0,
  "val_bags": 1920.0
}
```

对该训练结果的当前解释：

```text
工程含义：训练链路、hidden 抽取、bag 构造、保存格式都已跑通。
科学含义：指标过于完美，不能直接证明 probe 学到了真正 harmful token localization。
主要风险：负样本来自 refusal paraphrases，probe 可能学到 refusal-vs-compliance 风格差异，而不是细粒度 harmful semantics。
后续判断依据：mil_high_score_tokens.md、mil_token_selection_summary.json、cluster_token_top_terms.md，以及已经完成的 downstream no-probe vs probe 评估结果。
```

## 5. 当前实验进度

长任务 `stage4_after_train_pipeline` 已结束；当前没有继续运行的 Stage 4 tmux 会话。该 pipeline 已完成以下步骤：

```text
1. 验证 outputs/mil_token_probe_llada.pt
2. 构造 outputs/probe_ct_csd_llada_m16_tau07
3. 构造 outputs/probe_category_ct_csd_llada_m16_tau07
4. 运行 outputs/jbb_dija_probe_category_ct_csd_m16_tau07 生成评估
5. 完成 outputs/jbb_dija_probe_category_ct_csd_m16_tau07 Llama Guard 评判
```

### 5.1 已完成 bank

```text
outputs/probe_ct_csd_llada_m16_tau07/ct_csd_bank.pt
outputs/probe_ct_csd_llada_m16_tau07/ct_csd_bank_summary.json
outputs/probe_ct_csd_llada_m16_tau07/mil_token_selection_summary.json
outputs/probe_ct_csd_llada_m16_tau07/mil_high_score_tokens.md
outputs/probe_ct_csd_llada_m16_tau07/cluster_token_top_terms.md

outputs/probe_category_ct_csd_llada_m16_tau07/ct_csd_bank.pt
outputs/probe_category_ct_csd_llada_m16_tau07/ct_csd_bank_summary.json
outputs/probe_category_ct_csd_llada_m16_tau07/mil_token_selection_summary.json
outputs/probe_category_ct_csd_llada_m16_tau07/mil_high_score_tokens.md
outputs/probe_category_ct_csd_llada_m16_tau07/cluster_token_top_terms.md
```

两组 bank 均记录：

```text
method = probe_category_ct_csd 或 probe_ct_csd
target_layer = 31
num_total_clusters = 16
mil.enabled = true
mil.probe_threshold = 0.7
mil.top_q_ratio = 0.1
```

### 5.2 JBB + DIJA 评估状态

| 分支 | 生成结果 | 诊断 | Llama Guard judge | unsafe_count | ASR |
|---|---|---|---|---:|---:|
| direct | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07_direct/results.json` | 已完成 | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07_direct/llama_guard_results.json` | `71/100` | `71.0%` |
| 普通 pipeline | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07/results.json` | 已完成 | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07/llama_guard_results.json` | `71/100` | `71.0%` |

诊断统计：

| 分支 | total_routed | total_active | activation_rate | route_time_sec |
|---|---:|---:|---:|---:|
| direct | `44175` | `6159` | `0.1394227504244482` | `0.14039896093891002` |
| 普通 pipeline | `44705` | `5737` | `0.1283301644111397` | `0.14487624212051742` |

direct 分支和普通 pipeline 分支的 ASR 均与 Stage 3 no-probe 对照持平，都是 `71.0%`。普通 pipeline 分支的 `activation_rate = 0.1283301644111397`，低于 Stage 3 no-probe 对照的 `0.16145821895245416`，说明 MIL token selection 降低了实际触发 steering / remask 统计的 token 比例，但本轮没有进一步降低 ASR。

## 6. 资源与运行环境

最近检查时，Stage 4 长任务已结束，GPU0/GPU1 均无 Stage 4 进程占用。GPU 状态是瞬时信息，后续以 `nvidia-smi` 当前输出为准。

历史问题与处理：

```text
直接运行训练脚本时曾遇到 ModuleNotFoundError: No module named 'utils'
处理：用 PYTHONPATH=. 运行

第一次在 GPU0 训练时 OOM
根因：GPU0 已被另一个 eval_llada_steering.py 进程占用
处理：切换到 CUDA_VISIBLE_DEVICES=1，并按用户要求全部长命令放入 tmux
```

## 7. 已完成项与后续整理

### 7.1 普通 pipeline Llama Guard judge

已完成输出：

```text
outputs/jbb_dija_probe_category_ct_csd_m16_tau07/llama_guard_results.json
outputs/jbb_dija_probe_category_ct_csd_m16_tau07/llama_guard_run.log
```

评判输入文件：

```text
outputs/jbb_dija_probe_category_ct_csd_m16_tau07/results.json
```

评判参数需与 Stage 3 M16 对照保持一致：

```text
judge: /dev/shm/Llama-Guard-4-12B
device: cuda
```

普通 pipeline judge 结果：

```text
total_samples = 100
unsafe_count = 71/100
ASR = 71.0%
```

### 7.2 结果文档

Stage 4 完整实验已经结束。后续若继续整理分析，应新增或更新以下结果文档：

```text
docs/stage4_mil_probe_metrics.md
docs/stage4_probe_cluster_token_comparison.md
```

`docs/stage4_mil_probe_metrics.md` 至少应汇总：

```text
probe 训练指标
MIL retention ratio
per-category retention ratio
probe_empty_samples
JBB DIJA ASR / unsafe_count
refusal rate / over-refusal / 平均输出长度 / 空回复比例
activation_rate / steering token 数 / remask token 数
推理耗时与诊断开销
```

`docs/stage4_probe_cluster_token_comparison.md` 至少应对比：

```text
outputs/ct_csd_llada_m16/cluster_token_top_terms.md
outputs/probe_ct_csd_llada_m16_tau07/cluster_token_top_terms.md
outputs/category_ct_csd_llada_m16/cluster_token_top_terms.md
outputs/probe_category_ct_csd_llada_m16_tau07/cluster_token_top_terms.md
```

重点判断：

```text
probe 是否降低模板 token、空白 token、标点 token 的权重
probe 是否保留或提升 harmful semantics token
probe_category_ct_csd 是否相比 category_ct_csd 改善 ASR 或干预效率
probe 是否显著伤害生成质量或导致过拒答
```

## 9. 当前风险清单

1. MIL probe validation 指标过于完美。
   当前更应视为 in-distribution separability，而不是 harmful-token localization 已被证明。

2. safe negative 来自 refusal templates。
   需要用高分 token 样例和 downstream 评估确认 probe 没有只学到拒答模板差异。

3. Probe-CT-CSD 与 Probe-Category-aware CT-CSD bank 已构造完成。
   后续分析应重点检查 retention ratio、cluster token 变化与下游 ASR 持平之间的关系。

4. Probe-Category-aware JBB DIJA eval 与 Llama Guard judge 已完成。
   Stage 4 普通 pipeline 和 direct 分支均为 `71/100`、`ASR = 71.0%`，没有优于 Stage 3 no-probe 对照。

## 10. 完成定义对照

当前完成情况：

```text
[x] 原始总 plan 未修改
[x] 独立 Stage 4 执行计划存在
[x] Stage 4 代码已提交
[x] MIL probe 训练产物可加载
[x] Stage 4 最小单测通过
[x] probe_ct_csd bank 可加载
[x] probe_category_ct_csd bank 可加载
[x] 推理阶段不运行 probe 的设计边界保持
[x] no-probe vs probe 对照完成
[x] MIL retention、高分 token、cluster 高频 token 三类诊断齐全
```
