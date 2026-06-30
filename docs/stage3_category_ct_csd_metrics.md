# Stage 3 Category-aware CT-CSD 指标记录

## 实验配置

| 字段 | 取值 |
|---|---|
| Method | `category_ct_csd` |
| M sweep | `4`, `8`, `10`, `12`, `16` |
| 当前推荐 M | `10` |
| M 来源 | Stage 3 补充簇数消融；当前项目尚未完成 Stage 2 正式 `M*` 选择 |
| model | `/dev/shm/LLaDA-8B-Instruct` |
| target_layer | `31` |
| category_key | `semantic_category` |
| cluster_feature | `l2_normalized_hidden` |
| max_response_len | `128` |
| sampling_steps | `128` |
| mask_length | `128` |
| block_size | `128` |
| dija_mask_counts | `128` |
| alignment_threshold | `0.0` |
| steering_overshoot | `1.0` |
| initial_steering_ratio | `0.1` |
| max_refinement_iters | `5` |
| judge | `/dev/shm/Llama-Guard-4-12B` |

Stage 3 构造时模型前向输入为完整 `prompt + response`，但用于 category 计数、KMeans 和 cluster accumulate 的 hidden states 只取 response 段，并过滤 special token 与空白 token。prompt token 不进入聚类。

## 产物路径

| 方法 | 产物 | 路径 |
|---|---|---|
| CT-CSD baseline | bank | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` |
| CT-CSD baseline | generation | `outputs/jbb_dija_ct_csd_m16/results.json` |
| CT-CSD baseline | diagnostics | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |
| CT-CSD baseline | judge | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` |
| Category-aware M4 | bank | `outputs/category_ct_csd_llada_m4/ct_csd_bank.pt` |
| Category-aware M4 | generation / diagnostics / judge | `outputs/jbb_dija_category_ct_csd_m4/` |
| Category-aware M8 | bank | `outputs/category_ct_csd_llada_m8/ct_csd_bank.pt` |
| Category-aware M8 | generation / diagnostics / judge | `outputs/jbb_dija_category_ct_csd_m8/` |
| Category-aware M10 | bank | `outputs/category_ct_csd_llada_m10/ct_csd_bank.pt` |
| Category-aware M10 | generation / diagnostics / judge | `outputs/jbb_dija_category_ct_csd_m10/` |
| Category-aware M12 | bank | `outputs/category_ct_csd_llada_m12/ct_csd_bank.pt` |
| Category-aware M12 | generation / diagnostics / judge | `outputs/jbb_dija_category_ct_csd_m12/` |
| Category-aware M16 | bank | `outputs/category_ct_csd_llada_m16/ct_csd_bank.pt` |
| Category-aware M16 | generation / diagnostics / judge | `outputs/jbb_dija_category_ct_csd_m16/` |
| Category-aware CT-CSD | clustering analysis | `docs/stage3_category_ct_csd_clustering_analysis.md` |
| Category-aware CT-CSD | route / active analysis | `docs/stage3_category_ct_csd_route_active_analysis.md` |

## 安全指标

| 方法 | total_samples | unsafe_count ↓ | ASR ↓ | 相对 CT-CSD M16 变化 |
|---|---:|---:|---:|---:|
| CT-CSD M16 | `100` | `74` | `74.0%` | baseline |
| Category-aware CT-CSD M4 | `100` | `70` | `70.0%` | `-4.0 pp` |
| Category-aware CT-CSD M8 | `100` | `70` | `70.0%` | `-4.0 pp` |
| Category-aware CT-CSD M10 | `100` | `67` | `67.0%` | `-7.0 pp` |
| Category-aware CT-CSD M12 | `100` | `69` | `69.0%` | `-5.0 pp` |
| Category-aware CT-CSD M16 | `100` | `71` | `71.0%` | `-3.0 pp` |

本轮 Stage 3 category-aware 簇数消融中，`M=10` 的 ASR 最低，为 `67.0%`。相比 CT-CSD M16 baseline 少 `7` 个 unsafe 样本。

## Steering 诊断

| 方法 | total_routed | total_active | activation_rate | route_time_sec |
|---|---:|---:|---:|---:|
| CT-CSD M16 | `45,775` | `13,155` | `28.74%` | `0.1414` |
| Category-aware CT-CSD M4 | `42,915` | `4,006` | `9.33%` | `0.1651` |
| Category-aware CT-CSD M8 | `44,515` | `5,531` | `12.43%` | `0.1661` |
| Category-aware CT-CSD M10 | `44,535` | `5,888` | `13.22%` | `0.1605` |
| Category-aware CT-CSD M12 | `45,215` | `6,482` | `14.34%` | `0.1581` |
| Category-aware CT-CSD M16 | `45,535` | `7,352` | `16.15%` | `0.1396` |

Category-aware CT-CSD 的 active token 数随 M 增大整体上升，但 ASR 并不单调。`M=10` 在当前 100 条 JBB + DIJA 口径下取得最低 ASR。

## Category 簇预算

| M | category_cluster_counts |
|---:|---|
| `4` | `{'cybercrime_intrusion': 1, 'illegal': 1, 'misinformation_disinformation': 1, 'other': 1}` |
| `8` | `{'chemical_biological': 1, 'cybercrime_intrusion': 1, 'harassment_bullying': 2, 'harmful': 1, 'illegal': 2, 'misinformation_disinformation': 1}` |
| `10` | `{'chemical_biological': 1, 'cybercrime_intrusion': 2, 'harassment_bullying': 1, 'harmful': 1, 'illegal': 3, 'misinformation_disinformation': 2}` |
| `12` | `{'chemical_biological': 2, 'cybercrime_intrusion': 2, 'harassment_bullying': 1, 'harmful': 1, 'illegal': 3, 'misinformation_disinformation': 3}` |
| `16` | `{'chemical_biological': 2, 'cybercrime_intrusion': 2, 'harassment_bullying': 2, 'harmful': 2, 'illegal': 5, 'misinformation_disinformation': 3}` |

`M=4` 因类别数超过 cluster budget，会把尾部 category 合并进 `other`。`M=10` 是当前安全指标最好的折中点：保留全部原始 category，且给 `cybercrime_intrusion`、`illegal`、`misinformation_disinformation` 分配了额外簇。

## M16 小簇说明

M16 的最小簇为 `global_cluster_id = 13`：

| 字段 | 取值 |
|---|---:|
| category | `misinformation_disinformation` |
| train cluster_size | `5,980` |
| route_count | `333` |
| active_count | `216` |
| route_share | `0.73%` |
| active_share | `2.94%` |
| active_rate | `64.86%` |

该簇不是无用死簇，而是低路由、高激活的窄域簇。当前不建议仅因训练簇小而合并。更重要的是，簇数消融显示 `M=10` 的整体 ASR 更低，因此后续 no-probe 对照应优先使用 M10。

## 当前结论

Stage 3 当前结论：

1. `category_ct_csd` 在 `M=4/8/10/12/16` 上均完成 bank 构造、JBB + DIJA 推理和 Llama-Guard judge。
2. 五个 category-aware 点位均优于 CT-CSD M16 baseline 的 `74.0%` ASR。
3. `M=10` 当前最佳，`unsafe_count = 67/100`，`ASR = 67.0%`。
4. 当前推荐把 `M=10` 作为 Stage 3 no-probe category-aware 默认点位和 Stage 4 对照输入。
5. 当前项目仍不应表述为 Stage 2 已正式选出 `M*`；`M=10` 是 Stage 3 category-aware 消融下的当前最佳点。
