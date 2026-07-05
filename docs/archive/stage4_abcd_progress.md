# Stage 4 ABCD / 5 / 6 进度总览

更新时间：2026-06-30

本文件统一记录 Stage 4 系列（4 / 4A / 4C / 4D）与 Stage 5 / 6 的**规格来源、代码状态、实验状态、与论文的关系**，补齐此前各 Stage 进度文档缺失的横向视图。逐 Stage 细节仍以对应专题文档为准。

## 0. 命名约定（来自 `docs/plan/dlm_steering_project_improvement_plan.md` §1.2）

```text
Stage 3  : Category-aware CT-CSD（baseline，已完成）
Stage 4  : MIL token probe（已完成，效果未达预期，作为对照保留）
Stage 4A : Direction-selected token selection（粗方向打分 + 选 top-ratio）
Stage 4C : Random top-ratio token selection（对照，证明 4A 不是靠减少 token 数生效）
Stage 4D : KNN/ENN label-clean token selection
Stage 5  : Feature preprocessing（mean-centering + 可选 PCA 白化，即"降维"）
Stage 6  : Stage 4A + Stage 5 组合
```

## 1. 总览表

| Stage | 方法要点 | 代码状态 | 实验状态 | 专题文档 |
|---|---|---|---|---|
| 4 (MIL) | MIL token probe + 阈值选 token | 已提交 | 已跑，ASR 71%（未优于对照） | `stage4_mil_token_probe_progress.md` |
| **4A** | direction_top_ratio：粗方向投影选 top-ratio | 已提交（HEAD `d4569a2`） | **进行中**（见 §3） | 本文件 |
| 4C | random_top_ratio：随机选 top-ratio | 已提交（`TOKEN_SELECTION_CHOICES`） | 未跑 | 本文件 |
| 4D | knn_label_clean / per-class KNN | 已提交（`d4569a2 --knn_balanced`） | 全局 KNN 已试，**失效** | `docs/plan/stage4d_knn_label_clean_plan.md` |
| 5 | feature_preprocess：center + PCA 白化 | 已提交（`--feature_preprocess`） | 未跑（本轮**不跑降维**） | 同上 plan §5 |
| 6 | 4A + 5 组合 | 部分（参数已具备） | 未跑 | 同上 plan §6 |

> 说明：之前进度文档只覆盖 Stage 4 (MIL) 主线与 4D plan，4A/4C/5/6 缺独立进度记录——本文件补上。

## 2. 代码状态核对（2026-06-30，客观手段）

- 这批 4A/4D/5 代码**已提交进 main**，不再是"工作区裸奔未提交"状态：
  ```text
  d4569a2 feat(ct_csd): per-class KNN (--knn_balanced)
  978c191 feat(steering): LLaDA/Dream CT-CSD bank
  ```
- `utils/make_ct_csd_llada.py` 真实 token-selection 选项：
  ```text
  TOKEN_SELECTION_CHOICES = (all, direction_top_ratio, random_top_ratio,
                             mil_probe_threshold, knn_label_clean)
  FEATURE_PREPROCESS_CHOICES = (l2_only, center_l2, center_pca128_l2, center_pca256_l2)
  ```
- **真实 build 路径**：bank 构造走 `select_harmful_response_tokens` → `top_ratio_select`
  → `_coarse_direction_for_sample`（AST 调用计数客观确认）。
  `select_tokens` / `select_tokens_direction_top_ratio` 在模块内调用次数=0（死代码，
  仅作旧接口残留，不影响实际实验）。
- **测试覆盖（更正此前误判）**：真实路径 `select_harmful_response_tokens` 的
  `direction_top_ratio` 分支**已有单测**
  `test_direction_top_ratio_selects_tokens_by_category_direction_with_global_fallback`，
  且通过；`top_ratio_select` 亦有单测。全模块 `python -m unittest tests.test_make_ct_csd_llada`
  = **47 tests OK**。此前"真实路径无测试覆盖"的说法系排查过程中的误判，已纠正。
- `eval_llada_steering.py` 追加的是 Stage 5 降维推理支持（`--steering_preprocess_path`，
  默认 None → no-op，不传即与原版行为一致）；`eval_dream_steering.py` 追加的是
  Dream 端 steering/remask 集成（论文方法本体）。两者改动均为纯追加（deleted=0）。

## 3. Stage 4A 进行中实验（2026-06-30 启动）

两组单变量对照（均 `method=category_ct_csd`、`M=16`、`target_layer=31`、
`feature_preprocess=l2_only` **不降维**），脚本 `scripts/run_stage4a_direction.sh`，
base python（唯一 sklearn+transformers+torch 齐全的环境），tmux 后台、双卡并行：

| Config | tmux | GPU | selection_ratio | coarse_direction_type | 口径来源 | 输出 |
|---|---|---|---|---|---|---|
| A | `stage4a_A` | 0 | 0.5 | global | plan §12.5 | `outputs/dir_top_ratio_m16_r05_global` |
| B | `stage4a_B` | 1 | 0.3 | category | 代码默认 | `outputs/dir_top_ratio_m16_r03_cat` |

流程：构造 direction bank → JBB+DIJA 推理（steering，overshoot=1.0、
initial_steering_ratio=0.1、max_refinement_iters=5）→ 本地 Llama-Guard 评判。
bank 由全量 9605 条 harmful 样本构造，含两遍遍历（先累计粗方向、再选 token + 聚类），
单 config 预计数小时级。对照基线为 Stage 3 `outputs/category_ct_csd_llada_m16`（ASR 待比）。

结果（2026-06-30 22:3x 跑完，均 100 样本、JBB+DIJA、本地 Llama-Guard）：

| Config | unsafe_count | ASR | activation_rate | vs 基线(71.0%/0.161) |
|---|---|---|---|---|
| A (r05/global) | 72/100 | 72.0% | 0.1801 (active 8244/45775) | +1，**未改善（略差）** |
| B (r03/category) | 74/100 | 74.0% | 0.1852 (active 8479/45775) | +3，**未改善（略差）** |

**结论**：Stage 4A direction-selected token selection **未降低 ASR**，两组均略高于 Stage 3
`category_ct_csd_llada_m16` 基线（71.0%），且 activation_rate 反而升高（0.18+ vs 基线 0.161，
即触发更多 token 参与 steering/remask 却没换来更低 ASR）。差异仅 1–3 个样本，处于噪声范围，
稳妥结论是"与基线持平至略差，方向选 token 无正收益"。这与 Stage 4 (MIL, 71%) 未超基线的
结论一致——目前各类 token-selection 变体均未带来正向增益。降维(Stage 5)本轮未跑。

## 3.5 全景对比（截至 2026-07-01，均 100 样本 / JBB+DIJA / 本地 Llama-Guard）

| 方案 | M | token 选择 | ASR | activation_rate |
|---|---|---|---:|---:|
| Stage3 baseline | 16 | all | 71.0% | 0.161 |
| Stage3 baseline | 12 | all | **69.0%** | — |
| Stage4 MIL | 16 | mil_probe_threshold(τ0.7) | 71.0% | 0.128 |
| 4A direction | 16 | direction r0.5/global | 72.0% | 0.180 |
| 4A direction | 16 | direction r0.3/category | 74.0% | 0.185 |

要点：
- 目前**最低 ASR 是 Stage3 M12 baseline = 69.0%**（实测；CLAUDE.md 记的 65% 与当前 eval 口径有出入，以实测为准）。
- M16 上所有 token-selection 变体（MIL / direction）均未低于 M16 baseline(71%)。
- 因 M12 base 更优，后续在 **M12 上试 direction**；并加 **random top-ratio 对照**验证 direction 是否优于随机。

## 3.6 全局 ct_csd M12 上的 token-selection（2026-07-01）

最优全局 base 是 `ct_csd` M12（65%，非 category）。在该 base 上加 direction 与 random 对照
（均 `--method ct_csd` 全局、M12、ratio=0.5、`l2_only` 不降维）：

| 方案 | ASR | activation_rate | vs base 65% |
|---|---|---:|---|
| 全局 ct M12 baseline | 65.0% | 0.292 | — |
| + direction r0.5 (R3) | 68.0% | 0.195 | +3，更差 |
| + random r0.5 (R4) | 70.0% | 0.238 | +5，更差 |

**结论（2026-07-01 跑完）**：
- 在最优全局 base（ct M12=65%）上，**direction 与 random token selection 都未改善 ASR，反而更差**。
- direction(68%) 比 random(70%) 低 2 分——说明"按方向选"**确实比随机略强**（方向有信号，不是纯噪声）。
- 但两者都劣于用全部 token 的 baseline(65%)——说明"**把 token 减半**"这一动作本身对 ASR 有害，
  direction 只能部分抵消、无法翻正。
- 综合 M16 上的结果（direction 72/74% vs baseline 71%）：**token-selection 这条线（不论 direction/random/MIL）
  在当前 steering 框架下都没有正收益**。若继续追更低 ASR，建议转向 steering 超参（overshoot/
  initial_steering_ratio/alignment_threshold）或 base 结构，而非 token 选择。

## 3.7 Stage 4D KNN label-clean 现状（2026-07-02 核对）

已建 3 个 KNN bank（均 `--method ct_csd`、`token_selection=knn_label_clean`、M=16、knn_k=6、
keep_ratio=0.5），**均只建了 bank + retention 诊断，未跑 JBB+DIJA 推理与 Llama-Guard，故无 ASR**：

| bank | knn_balanced | retention | 删除 token | ASR |
|---|---|---:|---:|---|
| s4d_knn_smoke | None（冒烟） | 99.80% | 12/6074 | 无 |
| s4d_cmp_standard | False（全局） | 99.80% | 12/6074 | 无 |
| s4d_cmp_balanced | True（per-class 平衡） | 99.14% | 52/6074 | 无 |

**判断**：即使 per-class 平衡版也只删掉 0.86% 的 token（52/6074），bank 与全 token baseline 近乎等价，
补跑 ASR 预计 **≈ 全局 ct M16 baseline（74%）**，信息量低。与既有发现一致（有害:安全≈12:1 不平衡下，
全局/平衡 KNN 都几乎不去噪，方法在此设定下失效）。若仍要补完记录，两个 bank 已就绪，只需跑推理+评判。

## 3.8 后续方向：steering 超参扫描（计划，未跑）

token-selection 三条线（MIL / direction / random）均无正收益后，下一步应转向**直接影响干预强度的
steering 超参**，而非 token 选择。建议在最优 base（全局 ct M12=65%）上做单变量扫：

| 超参 | 当前值 | 建议扫 | 直觉 |
|---|---|---|---|
| steering_overshoot | 1.0 | 1.5, 2.0 | >1 推进安全半空间，更强抑制有害方向 |
| initial_steering_ratio | 0.1 | 0.2, 0.3 | 早期 steering 覆盖更多去噪步 |
| alignment_threshold | 0.0 | 先固定 | 避免多变量，后续再动 |

同一 base 复用（无需重建 bank，只改推理参数），与 ct M12 baseline(65%) 比 ASR + 生成质量
（警惕过度 steering 伤流畅性/过拒答）。这是当前最可能出正收益的方向。

## 4. 与论文的关系（重要）

论文《Adaptive Steering and Remasking for Safe Generation in DLMs》本体方法为：
**单一全局 CSD 方向 + 早期自适应 steering + 有害 token 重掩码**，训练-free，
**不含任何聚类**。因此：

```text
CT-CSD 聚类 bank 框架本身、category-aware、MIL probe、KNN label-clean、
direction 离线 top-ratio 选 token、PCA 降维 —— 均为论文之外的扩展，
不是论文复现。
```

Stage 4A 属于"论文之上找更低 ASR"的探索分支：论文用 CSD 方向做的是**推理时逐步**
选有害 token（给 steering/remask 用），而 4A 是**离线**用粗方向选 token 喂聚类，二者不同。
若目标是严格复现论文，应回到 `make_csd_llada.py` 单方向路线。
