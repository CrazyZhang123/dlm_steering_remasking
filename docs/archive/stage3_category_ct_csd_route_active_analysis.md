# Stage 3 Category-aware CT-CSD Route / Active 诊断

## 诊断产物

| 项 | 路径 |
|---|---|
| Category-aware 生成结果 | `outputs/jbb_dija_category_ct_csd_m16/results.json` |
| Category-aware route / active 诊断 | `outputs/jbb_dija_category_ct_csd_m16/ct_csd_diagnostics.json` |
| Category-aware 运行日志 | `outputs/jbb_dija_category_ct_csd_m16/run.log` |
| CT-CSD M=16 baseline 诊断 | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |
| Category-aware Llama-Guard judge | `outputs/jbb_dija_category_ct_csd_m16/llama_guard_results.json` |

本诊断先运行 route / active，再补跑 Llama-Guard judge。ASR 结果见下文。

## 总体对比

| 指标 | Category-aware CT-CSD M=16 | CT-CSD M=16 |
|---|---:|---:|
| total_routed | `45,535` | `45,775` |
| total_active | `7,352` | `13,155` |
| activation_rate | `16.15%` | `28.74%` |
| route_time_sec | `0.1396` | `0.1414` |

两者 routed token 数接近，但 Category-aware CT-CSD 的 active token 数减少 `5,803`，相对 CT-CSD M=16 降低约 `44.1%`。这说明 category-aware bank 在同一阈值 `theta=0.0` 下整体更少触发 steering。

## Llama-Guard Judge

| Method | total_samples | unsafe_count | ASR |
|---|---:|---:|---:|
| CT-CSD M=16 | `100` | `74` | `74.0%` |
| Category-aware CT-CSD M=16 | `100` | `71` | `71.0%` |

Category-aware CT-CSD M16 的 ASR 比 CT-CSD M16 低 `3.0` 个百分点。结合 active token 数下降 `44.1%`，本轮结果更像是减少过度触发，而不是明显 steering 不足。

## 每类统计

| category | route_count | active_count | route_share | active_share | active_rate |
|---|---:|---:|---:|---:|---:|
| `chemical_biological` | `3,433` | `506` | `7.54%` | `6.88%` | `14.74%` |
| `cybercrime_intrusion` | `10,660` | `1,469` | `23.41%` | `19.98%` | `13.78%` |
| `harassment_bullying` | `3,659` | `388` | `8.04%` | `5.28%` | `10.60%` |
| `harmful` | `1,464` | `195` | `3.22%` | `2.65%` | `13.32%` |
| `illegal` | `19,094` | `3,195` | `41.93%` | `43.46%` | `16.73%` |
| `misinformation_disinformation` | `7,225` | `1,599` | `15.87%` | `21.75%` | `22.13%` |

`misinformation_disinformation` 的整体 active_rate 是所有 category 中最高的，但 active_share 仍低于 `illegal`。

## 每簇统计

| gid | category | local | train_size | route | active | route_share | active_share | active_rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | `chemical_biological` | `0` | `91,362` | `1,379` | `499` | `3.03%` | `6.79%` | `36.19%` |
| `1` | `chemical_biological` | `1` | `68,774` | `2,054` | `7` | `4.51%` | `0.10%` | `0.34%` |
| `2` | `cybercrime_intrusion` | `0` | `67,378` | `7,068` | `187` | `15.52%` | `2.54%` | `2.65%` |
| `3` | `cybercrime_intrusion` | `1` | `110,545` | `3,592` | `1,282` | `7.89%` | `17.44%` | `35.69%` |
| `4` | `harassment_bullying` | `0` | `76,117` | `1,739` | `388` | `3.82%` | `5.28%` | `22.31%` |
| `5` | `harassment_bullying` | `1` | `49,228` | `1,920` | `0` | `4.22%` | `0.00%` | `0.00%` |
| `6` | `harmful` | `0` | `54,659` | `777` | `0` | `1.71%` | `0.00%` | `0.00%` |
| `7` | `harmful` | `1` | `70,304` | `687` | `195` | `1.51%` | `2.65%` | `28.38%` |
| `8` | `illegal` | `0` | `101,094` | `10,840` | `2,342` | `23.81%` | `31.86%` | `21.61%` |
| `9` | `illegal` | `1` | `17,232` | `626` | `444` | `1.37%` | `6.04%` | `70.93%` |
| `10` | `illegal` | `2` | `85,505` | `1,072` | `105` | `2.35%` | `1.43%` | `9.79%` |
| `11` | `illegal` | `3` | `15,182` | `306` | `252` | `0.67%` | `3.43%` | `82.35%` |
| `12` | `illegal` | `4` | `103,341` | `6,250` | `52` | `13.73%` | `0.71%` | `0.83%` |
| `13` | `misinformation_disinformation` | `0` | `5,980` | `333` | `216` | `0.73%` | `2.94%` | `64.86%` |
| `14` | `misinformation_disinformation` | `1` | `131,354` | `3,874` | `1,383` | `8.51%` | `18.81%` | `35.70%` |
| `15` | `misinformation_disinformation` | `2` | `101,769` | `3,018` | `0` | `6.63%` | `0.00%` | `0.00%` |

## 小簇 13 判断

小簇 13 的训练簇大小只有 `5,980`，是全局最小簇；但它不是无用簇：

| 指标 | 数值 |
|---|---:|
| route_count | `333` |
| active_count | `216` |
| route_share | `0.73%` |
| active_share | `2.94%` |
| active_rate | `64.86%` |

解释：

1. `route_share = 0.73%`，说明它在推理中命中的 token 很少。
2. `active_rate = 64.86%`，说明一旦命中它，大概率会超过阈值并触发 steering。
3. `active_share = 2.94%`，说明它对全局 active token 的贡献不大，不是全局激活失控来源。

因此，小簇 13 更像是一个窄域、高激活的专门簇，而不是完全无用簇。不能仅凭训练簇小就直接合并。

## 其他值得注意的簇

`illegal` 的两个小簇也呈现同样模式：

| gid | category | train_size | route | active | active_rate | active_share |
|---:|---|---:|---:|---:|---:|---:|
| `9` | `illegal` | `17,232` | `626` | `444` | `70.93%` | `6.04%` |
| `11` | `illegal` | `15,182` | `306` | `252` | `82.35%` | `3.43%` |
| `13` | `misinformation_disinformation` | `5,980` | `333` | `216` | `64.86%` | `2.94%` |

这些小簇都不是高频路由簇，但 active_rate 很高，可能捕获了更尖锐的局部 harmful direction。

另有三个簇被路由到但没有激活：

| gid | category | route_count |
|---:|---|---:|
| `5` | `harassment_bullying` | `1,920` |
| `6` | `harmful` | `777` |
| `15` | `misinformation_disinformation` | `3,018` |

这类簇可能对应较中性的 response hidden region，或者当前阈值下方向投影不强。它们不构成小簇异常，但后续评估时需要关注是否导致 steering 不足。

## 结论

1. `misinformation_disinformation` 小簇 13 不是“几乎不用”的死簇，而是低路由、高激活的小簇。
2. 小簇 13 的全局 active_share 只有 `2.94%`，没有显示出全局异常激活。
3. Category-aware CT-CSD M=16 相比普通 CT-CSD M=16，总 active token 数降低约 `44.1%`，同时 ASR 从 `74.0%` 降到 `71.0%`。
4. 暂不建议直接合并小簇 13。更合理的下一步是保留 M=16 作为 no-probe category-aware 对照；如果后续更大样本或其他 judge 显示 ASR 变差，再考虑提高 M、调整阈值，或对零激活簇和长尾簇做单独 ablation。
