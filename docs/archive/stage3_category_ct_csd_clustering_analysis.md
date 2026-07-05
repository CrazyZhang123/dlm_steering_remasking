# Stage 3 Category-aware CT-CSD 聚类分析

## 产物

| 项 | 路径 |
|---|---|
| bank | `outputs/category_ct_csd_llada_m16/ct_csd_bank.pt` |
| summary | `outputs/category_ct_csd_llada_m16/ct_csd_bank_summary.json` |
| cluster 分布 | `outputs/category_ct_csd_llada_m16/cluster_category_distribution.md` |
| 日志 | `outputs/category_ct_csd_llada_m16/run.log` |

## 基本校验

| 项 | 数值 |
|---|---:|
| format | `ct_csd_v1` |
| method | `category_ct_csd` |
| target_layer | `31` |
| centers shape | `(16, 4096)` |
| total clusters | `16` |
| total response tokens | `1,149,824` |
| skipped_pass1 | `0` |
| skipped_pass2 | `0` |

本次构造没有空簇，也没有样本被跳过。`cluster_sizes` 总和与 category token counts 总和一致。

## Category Budget

| category | token count | K |
|---|---:|---:|
| `chemical_biological` | `160,136` | `2` |
| `cybercrime_intrusion` | `177,923` | `2` |
| `harassment_bullying` | `125,345` | `2` |
| `harmful` | `124,963` | `2` |
| `illegal` | `322,354` | `5` |
| `misinformation_disinformation` | `239,103` | `3` |

## Cluster Size 分布

| category | K | cluster sizes | min | max | max/min | max share | 判断 |
|---|---:|---|---:|---:|---:|---:|---|
| `chemical_biological` | `2` | `[91362, 68774]` | `68,774` | `91,362` | `1.33` | `57.1%` | 均衡 |
| `cybercrime_intrusion` | `2` | `[67378, 110545]` | `67,378` | `110,545` | `1.64` | `62.1%` | 均衡 |
| `harassment_bullying` | `2` | `[76117, 49228]` | `49,228` | `76,117` | `1.55` | `60.7%` | 均衡 |
| `harmful` | `2` | `[54659, 70304]` | `54,659` | `70,304` | `1.29` | `56.3%` | 均衡 |
| `illegal` | `5` | `[101094, 17232, 85505, 15182, 103341]` | `15,182` | `103,341` | `6.81` | `32.1%` | 轻度不均衡 |
| `misinformation_disinformation` | `3` | `[5980, 131354, 101769]` | `5,980` | `131,354` | `21.97` | `54.9%` | 长尾小簇明显 |

## 与普通 CT-CSD M=16 对比

| 指标 | Category-aware CT-CSD M=16 | CT-CSD M=16 |
|---|---:|---:|
| total tokens | `1,149,824` | `1,149,824` |
| min cluster size | `5,980` | `323` |
| max cluster size | `131,354` | `253,268` |
| mean cluster size | `71,864` | `71,864` |
| CV | `0.489` | `0.908` |
| max/min | `21.97` | `784.11` |
| max cluster share | `11.4%` | `22.0%` |
| clusters `< 1%` total tokens | `1` | `3` |
| clusters `< 2%` total tokens | `3` | `4` |

Category-aware CT-CSD 的全局簇大小明显比普通 CT-CSD M=16 更均衡，极小簇数量也更少。

## 判断

1. `M=16` 的 category-aware 聚类可以作为第一版 Stage 3 bank 进入后续诊断；没有空簇或构造失败。
2. 四个 `K=2` 类别分布稳定，最大簇占比在 `56.3%` 到 `62.1%`，没有明显坍缩。
3. `illegal` 类有两个小簇，但最大簇只占该类 `32.1%`，整体像是大类内部存在多个子模式，暂不判定为失败。
4. `misinformation_disinformation` 有一个 `5,980` token 的小簇，只占该类 `2.5%`、全局 `0.52%`。这是当前最需要关注的长尾簇。

## 建议

暂时不要因为 `misinformation_disinformation` 的小簇直接废弃 M=16。当前 bank 比普通 CT-CSD M=16 更均衡，且类别预算符合预期。

下一步优先做同配置 route / active 诊断，观察小簇是否在推理中几乎不用，或是否异常激活。如果小簇长期无用或导致不稳定，再考虑：

1. 继续保留总预算 `M=16`，但对长尾小簇做合并或最小簇约束。
2. 试跑 `M=24` 的 category-aware bank，看大簇是否被更自然拆分。
3. 不建议直接降到 `M=12` 作为默认，因为 `harassment_bullying` 和 `harmful` 会各只剩 1 个簇，可能过粗。
