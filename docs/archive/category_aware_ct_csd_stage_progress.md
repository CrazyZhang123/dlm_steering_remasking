# Category-aware CT-CSD 阶段进度

## 阶段 0-4 总览

本文档统一使用 JBB + DIJA 生成口径和本地 Llama Guard 评判口径记录阶段进度。除非单独说明，评测集均为 `100` 条样本，核心指标只比较 `unsafe_count` 和 `ASR`。

| 阶段 | 状态 | 方法组成 | 本阶段实际做了什么 | 核心指标 | 结论 |
|---|---|---|---|---|---|
| 阶段 0 | 已完成 | 方法 A：Global Sentence-CSD；方法 B：JBB + DIJA + Llama Guard 冻结评测 | 使用 full HarmBench 构造单一全局 CSD 向量，并冻结已有 JBB + DIJA 生成和本地 Llama Guard 评判结果。 | `unsafe_count = 74/100`；`ASR = 74.0%` | 仅作为冻结 baseline；full HarmBench CSD 存在数据同源风险。 |
| 阶段 1 | 已完成 | 方法 A：CT-CSD M16；方法 B：hard routing；方法 C：threshold-gated steering / remask | 把 Stage 0 的单一全局向量替换为 16 个局部 CSD 向量；推理时 token 先路由到最近 harmful center，再用对应局部向量干预。 | 主实验 `74/100`；`ASR = 74.0%` | 最小 CT-CSD 闭环已跑通，但 M16 与 Stage 0 持平。 |
| 阶段 2 | 待正式冻结 | 方法 A：CT-CSD；方法 B：`num_total_clusters = 4/8/12/16` 消融；方法 C：后续 Random-K-CSD 对照 | 固定除簇数外的 JBB + DIJA 评测口径，完成 M4/M8/M12/M16 消融；Random-K-CSD 对照尚未完成。 | 最优点 `M=12`：`65/100`；`ASR = 65.0%` | `M=12` 是当前最低 ASR 点位，可作为默认 `M*` 候选，但 Stage 2 尚未正式冻结。 |
| 阶段 3 | 已完成 | 方法 A：Category-aware clustering；方法 B：CT-CSD M16；方法 C：no-probe 对照 | 按 `semantic_category` 分组，在每个 category 内部做 KMeans；推理仍只加载 `ct_csd_bank.pt`，不输入 prompt category；已完成 `M=4/8/10/12/16` 消融。 | 最优点 `M=10`：`67/100`；`ASR = 67.0%` | 五个 category-aware 点位均优于 CT-CSD M16；`M=10` 是 Stage 3 当前最佳，但不是 Stage 2 正式 `M*`。 |
| 阶段 4 | 已完成 | 方法 A：MIL token probe；方法 B：probe threshold harmful token selection；方法 C：Probe-Category-aware CT-CSD M16 `tau=0.7` | 训练 linear MIL token probe；离线构造 bank 时只保留 `probe_score >= 0.7` 的 harmful tokens；推理公式不变，仍使用 CT-CSD bank。 | 普通 pipeline `71/100`；`ASR = 71.0%`；direct 分支 `71/100`；`ASR = 71.0%` | ASR 与 Stage 3 M16 no-probe 对照持平，但弱于当前 Stage 3 M10；普通 pipeline 的 `activation_rate` 从 `0.1615` 降到 `0.1283`，干预触发更少。 |

当前已完成 judge 的结果中，阶段 1 簇数消融的 `CT-CSD M12` 是最低 ASR 点位：`ASR = 65.0%`，`unsafe_count = 65/100`。Stage 3 的 category-aware 消融当前最佳为 `M=10`：`ASR = 67.0%`，`unsafe_count = 67/100`。阶段 4 的 MIL probe 训练指标过于完美，当前只说明训练链路跑通，不能单独证明 probe 已学到可靠的有害 token localization；下游普通 pipeline 与 direct 分支的 ASR 均为 `71.0%`，且尚未按 Stage 3 最新推荐 `M=10` 重跑。

## 阶段 1 `num_total_clusters` 消融指标汇总

| `num_total_clusters` | 方法 | 状态 | `total_samples` | `unsafe_count` | `ASR` | `total_routed` | `total_active` | `activation_rate` |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| `4` | CT-CSD M4 | 已完成 | `100` | `70/100` | `70.0%` | `43165` | `5616` | `0.13010540947526933` |
| `8` | CT-CSD M8 | 已完成 | `100` | `71/100` | `71.0%` | `44670` | `6971` | `0.1560555182449071` |
| `12` | CT-CSD M12 | 已完成 | `100` | `65/100` | `65.0%` | `45775` | `13379` | `0.2922774440196614` |
| `16` | CT-CSD M16 | 已完成 | `100` | `74/100` | `74.0%` | `45775` | `13155` | `0.2873839432004369` |

表中后 3 个诊断指标含义如下：

| 诊断指标 | 含义 |
|---|---|
| `total_routed` | 推理阶段被 CT-CSD bank 执行 hard routing 的 token 总数。每个 routed token 会被分配到最近的 harmful cluster center。 |
| `total_active` | routed token 中超过 `alignment_threshold = 0.0` 并实际触发 steering / remask 统计的 token 数。 |
| `activation_rate` | 激活比例，计算方式为 `total_active / total_routed`，用于观察 routed token 中有多少真正触发干预。 |

## 阶段 0：已完成

阶段 0 冻结现有的 Global Sentence-CSD 基线和评测口径，用作后续 Category-aware CT-CSD 对比基线。本文档只记录已经完成的产物和指标；不引入新的实验代码，不移动既有输出，也不要求重新运行实验。

## 数据与评测口径

| 项目 | 口径 |
|---|---|
| CSD 构造数据 | HarmBench 全量有害样本集，用于构造 Global Sentence-CSD |
| 评测数据 | JailBreakBench 提示，使用 DIJA 精炼提示格式 |
| 评判器 | 本地 Llama Guard：`/dev/shm/Llama-Guard-4-12B` |
| 基线角色 | 仅作为阶段 0 冻结基线 |

全量 HarmBench CSD 基线与源自 HarmBench 的安全数据存在数据同源风险。因此这里只把它记录为阶段 0 冻结基线。阶段 1 及之后的对比应继续使用同一套 JBB + DIJA 评测口径，保证指标可比。

## 已有产物

| 产物 | 路径 | 备注 |
|---|---|---|
| 全量 HarmBench CSD 向量 | `.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt` | 已有 Global Sentence-CSD 引导向量产物 |
| 生成结果 | `.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/results.json` | 已有 JBB + DIJA 生成输出 |
| 评判结果 | `.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/llama_guard_results.json` | 已有本地 Llama Guard 评测输出 |

该基线的已有 CSD 构造日志记录了 `9605` 条有害样本和 `20` 条拒答改写。

## 冻结参数

| 参数 | 数值 |
|---|---:|
| `target_layer` | `31` |
| `sampling_steps` | `128` |
| `mask_length` | `128` |
| `block_size` | `128` |
| `remasking` | `low_confidence` |
| `alignment_threshold` | `0.0` |
| `steering_overshoot` | `1.0` |
| `initial_steering_ratio` | `0.1` |
| `max_refinement_iters` | `5` |

`max_refinement_iters` 在阶段 0 记录中冻结为 `5`。这里单独写明，是为了避免和历史 CLI 默认值 `3` 混淆；后续复现仍应显式传入 `5`。

## 阶段 1 固定参数说明

阶段 1 继承阶段 0 的 JBB + DIJA 评测口径，只新增 CT-CSD 向量库这一项变量。除非后续进度记录明确说明口径变化，否则阶段 1 固定使用下列参数。

| 参数 | 数值 | 适用环节 | 含义 |
|---|---:|---|---|
| `target_layer` | `31` | CSD 构造和推理 | 使用第 31 层 transformer 隐藏状态构造引导向量并应用引导。固定该层可以让阶段 0 全局 CSD 与阶段 1 CT-CSD 直接对比。 |
| `num_total_clusters` | `16` | 阶段 1 CT-CSD 构造 | 有害 token 聚类簇的数量，也就是 CT-CSD 向量库中局部引导向量的数量。该参数只属于阶段 1，不适用于阶段 0 单向量基线。 |
| `sampling_steps` | `128` | JBB + DIJA 生成 | 扩散去噪 / token 揭示的迭代步数，控制模型在解码最终回答前执行多少次迭代更新。 |
| `mask_length` | `128` | JBB + DIJA 生成 | 生成回答区域分配的 mask token 数量。在冻结口径下，模型最多填充 128 个生成 token。 |
| `block_size` | `128` | 分块生成和重新 mask | 每个解码块的大小。由于 `mask_length = 128` 且 `block_size = 128`，整个回答区域会作为一个块处理。 |
| `dija_mask_counts` | `128` | DIJA 提示生成 | 当 DIJA 精炼提示中没有显式 `<mask:N>` 片段时，追加的 DIJA mask token 数量。该值用于让 DIJA 输出长度与 `mask_length` 对齐。 |
| `alignment_threshold` | `0.0` | 引导和有害 token 重新 mask | 触发引导或重新 mask 所需的最小有害方向对齐分数。若 token 的分数 `s(h) <= 0.0`，则不执行引导。 |
| `steering_overshoot` | `1.0` | 引导 | `alpha = beta * max(s(h) - threshold, 0)` 中的 beta 系数。当取 `1.0` 时，引导会减去超过 threshold 的有害方向投影。 |
| `initial_steering_ratio` | `0.1` | 生成阶段引导调度 | 启用 adaptive steering hook 的早期生成步骤比例。配合 `sampling_steps = 128` 时，表示前 10% 去噪步骤启用引导。 |
| `max_refinement_iters` | `5` | Phase 2 重新 mask 并生成循环 | 检测到有害对齐 token 后，最多执行 5 次重新 mask 并生成的 refinement。这里固定为 `5`，避免和历史 CLI 默认值 `3` 混淆；后续复现仍应显式传入 `5`。 |

## 冻结指标

阶段 0 只记录 `ASR` / `unsafe_count` / `total_samples`。

| 指标 | 数值 |
|---|---:|
| `total_samples` | `100` |
| `unsafe_count` | `74/100` |
| `ASR` | `74.0%` |

等价摘要：`ASR = 74.0%`，`unsafe_count = 74/100`，`total_samples = 100`。

## 后续对比规则

阶段 1 及之后的 Category-aware CT-CSD 变体应在同一套 JBB + DIJA 评测口径和同一路径的评判器下，与阶段 0 冻结基线对比。若后续必须改变口径，需要在新的进度记录中明确说明原因和影响范围。

## 阶段 1：已完成

阶段 1 在与阶段 0 相同的 JBB + DIJA 评测口径下冻结 CT-CSD bank 最小闭环。该阶段只把引导向量从单个 Global Sentence-CSD 向量替换为 `ct_csd_v1` 多局部向量库；不引入 category 字段、Random-K-CSD 对照、MIL token probe、soft routing 或 per-cluster threshold。

## 阶段 1 产物

| 产物 | 路径 |
|---|---|
| CT-CSD bank | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` |
| bank 构造日志 | `outputs/ct_csd_llada_m16/run.log` |
| 生成结果 | `outputs/jbb_dija_ct_csd_m16/results.json` |
| 生成日志 | `outputs/jbb_dija_ct_csd_m16/run.log` |
| 评判结果 | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` |
| 评判日志 | `outputs/jbb_dija_ct_csd_m16/judge.log` |
| CT-CSD 诊断 | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |
| 阶段 1 指标文档 | `docs/category_aware_ct_csd_stage1_metrics.md` |

阶段 1 bank 使用 `format = ct_csd_v1`，向量形状为 `(16, 4096)`，`cluster_sizes` 为 `[4453, 46106, 253268, 87014, 107056, 59533, 72073, 14335, 323, 72716, 32378, 121351, 31465, 172888, 67231, 7634]`。构造日志和 bank 配置记录 `skipped_pass1 = 0`、`skipped_pass2 = 0`。

## 阶段 1 token 过滤情况

阶段 1 构造 CT-CSD bank 时，会先从 response 区域抽取 `target_layer = 31` hidden states，再按 response token ID 对 hidden states 做过滤。过滤逻辑如下：

- 过滤 `tokenizer.all_special_ids` 中的 special token。
- 额外过滤 tokenizer 暴露的 `pad_token_id`、`eos_token_id`、`bos_token_id`、`mask_token_id`。
- 过滤 `tokenizer.decode([token_id], skip_special_tokens=False).strip()` 为空的 token，也就是空格、换行、制表符等空白 token。
- 不过滤标点；当前 bank 配置为 `exclude_punctuation = False`。

当前 `/dev/shm/LLaDA-8B-Instruct` tokenizer 暴露的 special token 包括 `<|startoftext|>`、`<|endoftext|>`、`[CLS]`、`<role>`、`</role>`、`<|arithmetic_start|>`、`<|arithmetic_end|>`、`<|number_start|>`、`<|number_end|>`。需要注意：`<|mdm_mask|>` 的 token ID 为 `126336`，但当前 tokenizer 没有把它注册为 `mask_token_id`，也没有放入 `all_special_ids`。Stage 1 CSD 构造使用数据集 response 文本且 `add_special_tokens=False`，不会主动插入生成阶段的 mask token；本次真实统计中也没有 special token 被过滤。

HarmBench harmful response 在 `max_response_len = 128` 截断后，真实过滤统计如下：

| 项目 | 数值 |
|---|---:|
| 原始 response token 数 | `1213086` |
| 过滤后保留 token 数 | `1149824` |
| 过滤掉的空白 token 数 | `63262` |
| 过滤掉的 special token 数 | `0` |
| 保留比例 | `94.79%` |

过滤后保留 token 数 `1149824` 与当前 bank 的 `cluster_sizes` 总和完全一致，说明过滤后的 hidden states 确实进入了 KMeans 聚类和 cluster sum 累加。

本次实际被过滤最多的是空白类 token：

| token ID | decoded 文本 | 次数 |
|---:|---|---:|
| `198` | `\n` | `52983` |
| `220` | 空格 | `6295` |
| `256` | 双空格 | `2481` |
| `305` | 三空格 | `742` |
| `201` | `\r` | `188` |
| `197` | `\t` | `61` |

## 阶段 1 聚类情况

阶段 1 使用 `num_total_clusters = 16`，基于第 `31` 层 harmful response token hidden states 做聚类。bank 中共有 `1149824` 个 harmful token hidden states 被分配到 16 个簇。聚类分布呈长尾形态：最大簇为 `2`，包含 `253268` 个 token，占 `22.03%`；最小簇为 `8`，包含 `323` 个 token，占 `0.03%`。

下表同时记录构造阶段的簇大小，以及 JBB + DIJA 推理阶段的 hard routing 和 threshold-gated activation 统计。`激活/路由` 表示该簇在推理阶段被路由到后，实际超过 `alignment_threshold = 0.0` 并触发引导或重新 mask 统计的比例。

| 簇 ID | 构造 token 数 | 构造占比 | 推理路由数 | 推理激活数 | 激活/路由 |
|---:|---:|---:|---:|---:|---:|
| `2` | `253268` | `22.03%` | `2992` | `371` | `0.124` |
| `13` | `172888` | `15.04%` | `3813` | `8` | `0.002` |
| `11` | `121351` | `10.55%` | `8520` | `344` | `0.040` |
| `4` | `107056` | `9.31%` | `8363` | `2006` | `0.240` |
| `3` | `87014` | `7.57%` | `5871` | `4049` | `0.690` |
| `9` | `72716` | `6.32%` | `2453` | `396` | `0.161` |
| `6` | `72073` | `6.27%` | `2954` | `20` | `0.007` |
| `14` | `67231` | `5.85%` | `3540` | `2691` | `0.760` |
| `5` | `59533` | `5.18%` | `2290` | `290` | `0.127` |
| `1` | `46106` | `4.01%` | `2300` | `1173` | `0.510` |
| `10` | `32378` | `2.82%` | `1513` | `1128` | `0.746` |
| `12` | `31465` | `2.74%` | `494` | `467` | `0.945` |
| `7` | `14335` | `1.25%` | `159` | `134` | `0.843` |
| `15` | `7634` | `0.66%` | `448` | `13` | `0.029` |
| `0` | `4453` | `0.39%` | `14` | `14` | `1.000` |
| `8` | `323` | `0.03%` | `51` | `51` | `1.000` |

从诊断上看，构造时大簇不一定对应推理时高激活簇。推理阶段路由最多的是簇 `11` 和簇 `4`，而激活比例较高的簇包括 `12`、`7`、`14` 和 `10`。因此后续阶段如果要调参，应同时关注构造阶段簇大小、推理阶段路由频次和激活比例，不能只按 `cluster_sizes` 判断簇的重要性。

## 阶段 1 指标

阶段 1 核心指标只记录 `ASR` / `unsafe_count` / `total_samples`。

| 指标 | 数值 |
|---|---:|
| `total_samples` | `100` |
| `unsafe_count` | `74/100` |
| `ASR` | `74.0%` |

等价摘要：`ASR = 74.0%`，`unsafe_count = 74/100`，`total_samples = 100`。

## 阶段 1 诊断

| 诊断项 | 数值 |
|---|---:|
| `num_clusters` | `16` |
| `total_routed` | `45775` |
| `total_active` | `13155` |
| `activation_rate` | `0.2873839432004369` |
| `route_time_sec` | `0.14139450204675086` |

## 阶段 1 补充：簇数消融已完成

为判断 `num_total_clusters` 对 CT-CSD bank 的影响，阶段 1 补充簇数消融实验，固定除 `num_total_clusters` 之外的全部 JBB + DIJA 评测口径，取值为 `4`、`8`、`12`、`16`。其中 `num_total_clusters = 16` 已由阶段 1 主实验完成，作为消融中的 `m16` 点位；`m4`、`m8`、`m12` 已按同一口径补跑完成。

固定不变的关键参数包括：`target_layer = 31`、`sampling_steps = 128`、`mask_length = 128`、`block_size = 128`、`dija_mask_counts = 128`、`alignment_threshold = 0.0`、`steering_overshoot = 1.0`、`initial_steering_ratio = 0.1`、`max_refinement_iters = 5`。后续记录消融结果时仍只比较 `ASR` / `unsafe_count` / `total_samples`，诊断项用于解释 routing 和 activation 差异。

| `num_total_clusters` | bank 路径 | 生成输出路径 | judge 路径 | 状态 | `total_samples` | `unsafe_count` | `ASR` | `total_routed` | `total_active` | `activation_rate` |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `4` | `outputs/ct_csd_llada_m4/ct_csd_bank.pt` | `outputs/jbb_dija_ct_csd_m4/results.json` | `outputs/jbb_dija_ct_csd_m4/llama_guard_results.json` | 已完成 | `100` | `70/100` | `70.0%` | `43165` | `5616` | `0.13010540947526933` |
| `8` | `outputs/ct_csd_llada_m8/ct_csd_bank.pt` | `outputs/jbb_dija_ct_csd_m8/results.json` | `outputs/jbb_dija_ct_csd_m8/llama_guard_results.json` | 已完成 | `100` | `71/100` | `71.0%` | `44670` | `6971` | `0.1560555182449071` |
| `12` | `outputs/ct_csd_llada_m12/ct_csd_bank.pt` | `outputs/jbb_dija_ct_csd_m12/results.json` | `outputs/jbb_dija_ct_csd_m12/llama_guard_results.json` | 已完成 | `100` | `65/100` | `65.0%` | `45775` | `13379` | `0.2922774440196614` |
| `16` | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` | `outputs/jbb_dija_ct_csd_m16/results.json` | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` | 已完成 | `100` | `74/100` | `74.0%` | `45775` | `13155` | `0.2873839432004369` |

`m12` 生成结果已通过结构检查：`results.json` 共 `100` 条，`ct_csd_diagnostics.json` 记录 `num_clusters = 12`、`total_routed = 45775`、`total_active = 13379`、`activation_rate = 0.2922774440196614`。本地 Llama Guard 评判已完成，`unsafe_count = 65/100`，`ASR = 65.0%`。

当前四个簇数点位均已完成。在该固定评测口径下，`num_total_clusters = 12` 的 `ASR = 65.0%`，是阶段 1 簇数消融中最低的攻击成功率。该结果可作为后续 Stage 2 默认 `M*` 选择的输入；Stage 2 仍需完成同一 `M*` 下的 Random-K-CSD 对照，不能仅凭该消融表宣称 Stage 2 已完成。后续若为了严格同批次对比而重跑 `m16`，需要在本进度文档中明确区分“当前已完成 m16”和“重跑 m16”的产物路径。

阶段 1 在当前固定口径下与阶段 0 的 `ASR` 和 `unsafe_count` 持平。后续阶段应继续沿用相同 JBB + DIJA 评测口径，除非新的进度记录明确说明口径变化及其影响范围。

## 阶段 2：默认簇数选择与 Random-K-CSD 对照待冻结

阶段 2 的目标是把 CT-CSD 的默认簇数 `M*` 与 Random-K-CSD 对照正式冻结。当前只完成了 CT-CSD 簇数消融输入：`M=4/8/12/16` 均已完成 bank 构造、JBB + DIJA 推理和 Llama-Guard judge；Random-K-CSD 对照尚未完成，因此 Stage 2 仍不能标记为正式完成。

当前 CT-CSD 簇数消融结果如下：

| M | 方法 | unsafe_count | ASR | 诊断判断 |
|---:|---|---:|---:|---|
| `4` | CT-CSD M4 | `70/100` | `70.0%` | active token 少于 M12/M16，但 ASR 不是最优 |
| `8` | CT-CSD M8 | `71/100` | `71.0%` | 与 M4 接近，弱于 M12 |
| `12` | CT-CSD M12 | `65/100` | `65.0%` | 当前 CT-CSD 簇数消融最低 ASR，是默认 `M*` 候选 |
| `16` | CT-CSD M16 | `74/100` | `74.0%` | 与 Stage 0 baseline 持平，非最优 |

Stage 2 当前结论只能写成：`M=12` 是 CT-CSD 簇数消融中的默认 `M*` 候选；要正式冻结 Stage 2，还需要在同一 `M=12` 和同一评测口径下补 Random-K-CSD 对照。若 Random-K-CSD 与 CT-CSD M12 指标接近，则不能声称 KMeans 聚类结构本身带来稳定收益。

## 阶段 3：Category-aware CT-CSD 已完成

阶段 3 在现有 CT-CSD M16 baseline 上新增 `category_ct_csd` 离线构造分支。该阶段只改变 bank 构造方式：按 `semantic_category` 分组，在每个 category 内部做 KMeans 聚类，并写入 category metadata；推理阶段仍使用 `ct_csd_v1` bank 接口，不输入 prompt category，不训练 prompt category classifier，也不加入 MIL token probe。

阶段 3 初始实验启动时尚未完成 Stage 2 正式 `M*` 选择，因此先使用 `M=16` 继承已有完整 CT-CSD M16 baseline，不能表述为由 Stage 2 选出的最终 `M*`。随后补充完成 category-aware 簇数消融：`M = 4, 8, 10, 12, 16`。当前 Stage 3 category-aware 消融下的最佳点位为 `M=10`，但这仍不是 Stage 2 的正式 `M*`。

## 阶段 3 产物

| 产物 | 路径 |
|---|---|
| Category-aware CT-CSD bank | `outputs/category_ct_csd_llada_m{4,8,10,12,16}/ct_csd_bank.pt` |
| bank summary | `outputs/category_ct_csd_llada_m{4,8,10,12,16}/ct_csd_bank_summary.json` |
| cluster 分布 | `outputs/category_ct_csd_llada_m{4,8,10,12,16}/cluster_category_distribution.md` |
| 生成结果 | `outputs/jbb_dija_category_ct_csd_m{4,8,10,12,16}/results.json` |
| 生成日志 | `outputs/jbb_dija_category_ct_csd_m{4,8,10,12,16}/run.log` |
| route / active 诊断 | `outputs/jbb_dija_category_ct_csd_m{4,8,10,12,16}/ct_csd_diagnostics.json` |
| 评判结果 | `outputs/jbb_dija_category_ct_csd_m{4,8,10,12,16}/llama_guard_results.json` |
| 评判日志 | `outputs/jbb_dija_category_ct_csd_m{4,8,10,12,16}/judge.log` |
| 聚类分析 | `docs/stage3_category_ct_csd_clustering_analysis.md` |
| route / active 分析 | `docs/stage3_category_ct_csd_route_active_analysis.md` |
| Stage 3 metrics | `docs/stage3_category_ct_csd_metrics.md` |

## 阶段 3 hidden state 口径

构造时模型前向输入为完整 `prompt + response`，但用于 category 计数、KMeans 和 cluster accumulate 的 hidden states 只取 response 段，并过滤 special token 与空白 token。prompt token 不进入聚类。

safe anchor 使用同样口径：只取 safe refusal response 段的有效 token hidden states，并先对每条 safe refusal response 求均值，再参与全局 safe mean 计算。

## 阶段 3 M16 category budget

| category | token count | K |
|---|---:|---:|
| `chemical_biological` | `160136` | `2` |
| `cybercrime_intrusion` | `177923` | `2` |
| `harassment_bullying` | `125345` | `2` |
| `harmful` | `124963` | `2` |
| `illegal` | `322354` | `5` |
| `misinformation_disinformation` | `239103` | `3` |

## 阶段 3 指标

| Method | total_samples | unsafe_count | ASR |
|---|---:|---:|---:|
| CT-CSD M16 | `100` | `74/100` | `74.0%` |
| Category-aware CT-CSD M4 | `100` | `70/100` | `70.0%` |
| Category-aware CT-CSD M8 | `100` | `70/100` | `70.0%` |
| Category-aware CT-CSD M10 | `100` | `67/100` | `67.0%` |
| Category-aware CT-CSD M12 | `100` | `69/100` | `69.0%` |
| Category-aware CT-CSD M16 | `100` | `71/100` | `71.0%` |

所有 category-aware 点位在同一 JBB + DIJA + Llama-Guard 口径下均优于 CT-CSD M16。当前最佳为 `M=10`，比 CT-CSD M16 少 `7` 个 unsafe 样本，ASR 降低 `7.0` 个百分点。

## 阶段 3 诊断

| Method | total_routed | total_active | activation_rate | route_time_sec |
|---|---:|---:|---:|---:|
| CT-CSD M16 | `45775` | `13155` | `0.2873839432004369` | `0.14139450204675086` |
| Category-aware CT-CSD M4 | `42915` | `4006` | `0.09334731445881393` | `0.16511663375422359` |
| Category-aware CT-CSD M8 | `44515` | `5531` | `0.12425025272380097` | `0.16613628424238414` |
| Category-aware CT-CSD M10 | `44535` | `5888` | `0.13221062085999777` | `0.16054558870382607` |
| Category-aware CT-CSD M12 | `45215` | `6482` | `0.143359504589185` | `0.1581120560877025` |
| Category-aware CT-CSD M16 | `45535` | `7352` | `0.16145821895245416` | `0.13955931332020555` |

Category-aware CT-CSD 的 active token 数显著低于 CT-CSD M16；其中 `M=10` 在当前 ASR 指标上最好。

## 阶段 3 category-aware 簇数消融

| M | category_cluster_counts | unsafe_count | ASR | 判断 |
|---:|---|---:|---:|---|
| `4` | `{'cybercrime_intrusion': 1, 'illegal': 1, 'misinformation_disinformation': 1, 'other': 1}` | `70/100` | `70.0%` | 类别合并较多，但优于 CT-CSD M16 |
| `8` | `{'chemical_biological': 1, 'cybercrime_intrusion': 1, 'harassment_bullying': 2, 'harmful': 1, 'illegal': 2, 'misinformation_disinformation': 1}` | `70/100` | `70.0%` | 保留全部原始 category，安全指标稳定 |
| `10` | `{'chemical_biological': 1, 'cybercrime_intrusion': 2, 'harassment_bullying': 1, 'harmful': 1, 'illegal': 3, 'misinformation_disinformation': 2}` | `67/100` | `67.0%` | 当前最佳 |
| `12` | `{'chemical_biological': 2, 'cybercrime_intrusion': 2, 'harassment_bullying': 1, 'harmful': 1, 'illegal': 3, 'misinformation_disinformation': 3}` | `69/100` | `69.0%` | 次优，略弱于 M10 |
| `16` | `{'chemical_biological': 2, 'cybercrime_intrusion': 2, 'harassment_bullying': 2, 'harmful': 2, 'illegal': 5, 'misinformation_disinformation': 3}` | `71/100` | `71.0%` | 初始对照点，非最优 |
 
当前推荐把 `M=10` 作为 Stage 3 no-probe category-aware 默认点位和 Stage 4 对照输入。

## 阶段 3 小簇判断

`misinformation_disinformation` 的最小簇为 `global_cluster_id = 13`，训练时 cluster size 为 `5980`。JBB + DIJA 推理诊断中，该簇 `route_count = 333`、`active_count = 216`、`active_rate = 64.86%`、`active_share = 2.94%`。

该簇不是无用死簇，而是低路由、高激活的窄域簇；当前不建议仅因训练簇小而合并。

## 阶段 3 结论

当前 `M=4/8/10/12/16` category-aware bank 均构造成功，评测指标均优于 CT-CSD M16 baseline。`M=10` 是本轮 Stage 3 category-aware 簇数消融的最佳点位，可作为后续 no-probe category-aware 默认对照。若后续继续扩展 Stage 3，可在 `M=10` 附近补充更细粒度点位，而不是直接合并 M16 小簇或盲目增大到 `M=24`。

## 阶段 4：MIL token probe 接入已完成

阶段 4 只把 MIL token probe 接入离线 bank 构造阶段，用于 harmful response token selection；推理阶段仍只加载 `ct_csd_bank.pt`，不在 `eval_llada_steering.py` 中运行 probe。当前 probe 阈值固定为 `probe_threshold = 0.7`，`top_q_ratio = 0.1`，簇数沿用早先 Stage 3 M16 对照；它还没有按 Stage 3 最新消融推荐的 `M=10` 重跑。

## 阶段 4 产物

| 产物 | 路径 | 状态 |
|---|---|---|
| MIL token probe | `outputs/mil_token_probe_llada.pt` | 已完成 |
| MIL probe 训练指标 | `outputs/mil_token_probe_llada_metrics.json` | 已完成 |
| Probe-CT-CSD bank | `outputs/probe_ct_csd_llada_m16_tau07/ct_csd_bank.pt` | 已完成 |
| Probe-Category-aware CT-CSD bank | `outputs/probe_category_ct_csd_llada_m16_tau07/ct_csd_bank.pt` | 已完成 |
| Probe-Category-aware CT-CSD direct bank | `outputs/probe_category_ct_csd_llada_m16_tau07_direct/ct_csd_bank.pt` | 已完成 |
| 普通 `tau07` 生成结果 | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07/results.json` | 已完成 |
| 普通 `tau07` 诊断 | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07/ct_csd_diagnostics.json` | 已完成 |
| 普通 `tau07` judge | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07/llama_guard_results.json` | 已完成 |
| direct 生成结果 | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07_direct/results.json` | 已完成 |
| direct judge | `outputs/jbb_dija_probe_category_ct_csd_m16_tau07_direct/llama_guard_results.json` | 已完成 |

## 阶段 4 MIL probe 训练指标

| 指标 | 数值 |
|---|---:|
| `train_bags` | `17290` |
| `val_bags` | `1920` |
| `train_loss` | `8.790893101923604e-31` |
| `val_loss` | `6.666953987881271e-33` |
| `val_accuracy` | `1.0` |
| `val_auc` | `1.0` |

该训练指标只能说明 MIL probe 训练、hidden 抽取、bag 构造和保存链路已经跑通。由于负样本来自 refusal paraphrases，指标过于完美，存在 probe 学到 refusal-vs-compliance 风格差异而非细粒度 harmful semantics 的风险；最终仍应以下游 JBB + DIJA + Llama Guard 指标为准。

## 阶段 4 指标

| Method | 状态 | total_samples | unsafe_count | ASR |
|---|---|---:|---:|---:|
| Category-aware CT-CSD M16 no-probe 对照 | 已完成 | `100` | `71/100` | `71.0%` |
| Probe-Category-aware CT-CSD M16 `tau=0.7` direct | 已完成 | `100` | `71/100` | `71.0%` |
| Probe-Category-aware CT-CSD M16 `tau=0.7` 普通 pipeline | 已完成 | `100` | `71/100` | `71.0%` |

## 阶段 4 诊断

| Method | total_routed | total_active | activation_rate | route_time_sec |
|---|---:|---:|---:|---:|
| Category-aware CT-CSD M16 no-probe 对照 | `45535` | `7352` | `0.16145821895245416` | `0.13955931332020555` |
| Probe-Category-aware CT-CSD M16 `tau=0.7` direct | `44175` | `6159` | `0.1394227504244482` | `0.14039896093891002` |
| Probe-Category-aware CT-CSD M16 `tau=0.7` 普通 pipeline | `44705` | `5737` | `0.1283301644111397` | `0.14487624212051742` |

direct 分支和普通 `tau07` pipeline 的 ASR 均与 Stage 3 M16 no-probe 对照持平，都是 `71.0%`。普通 `tau07` pipeline 的 `activation_rate = 0.1283301644111397`，低于 Stage 3 M16 no-probe 对照的 `0.16145821895245416`，说明 MIL token selection 降低了实际触发 steering / remask 统计的 token 比例，但本轮没有进一步降低 ASR。

## 阶段 4 结论

Stage 4 已完成 M16 `tau=0.7` 的 direct 与普通 pipeline 两条评测链路。结果显示 MIL token probe 可以降低普通 pipeline 的 activation_rate，但没有降低 ASR：direct 与普通 pipeline 都是 `71/100`、`ASR = 71.0%`。由于 Stage 3 最新 category-aware 消融显示 `M=10` 可达到 `67/100`、`ASR = 67.0%`，后续若继续推进 probe 分支，应优先补 `Probe-Category-aware CT-CSD M10`，而不是只沿用 M16。
