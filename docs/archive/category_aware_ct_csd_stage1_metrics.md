# Category-aware CT-CSD 阶段 1 指标

## 状态

阶段 1 CT-CSD bank 最小闭环已完成。本文档只记录已经完成的产物、评测口径、核心指标和诊断统计；不重新运行实验，不移动既有输出。

## 评测口径

阶段 1 继承阶段 0 的 JBB + DIJA 评测口径，只把引导向量从单个 Global Sentence-CSD 向量替换为 CT-CSD 多局部向量库。

| 项目 | 数值 |
|---|---|
| 方法 | `ct_csd` |
| 模型族 | `llada` |
| 生成模型 | `/dev/shm/LLaDA-8B-Instruct` |
| 评测数据 | JBB + DIJA refined prompt |
| 评判器 | 本地 `/dev/shm/Llama-Guard-4-12B` |
| CSD 构造数据 | `.worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json` |
| 拒答改写 | `utils/refusals.txt` |
| `target_layer` | `31` |
| `num_total_clusters` | `16` |
| `sampling_steps` | `128` |
| `mask_length` | `128` |
| `block_size` | `128` |
| `dija_mask_counts` | `128` |
| `remasking` | `low_confidence` |
| `alignment_threshold` | `0.0` |
| `steering_overshoot` | `1.0` |
| `initial_steering_ratio` | `0.1` |
| `max_refinement_iters` | `5` |

`max_refinement_iters` 固定记录为 `5`，避免和历史 CLI 默认值 `3` 混淆。

## 产物

| 产物 | 路径 |
|---|---|
| CT-CSD bank | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` |
| bank 构造日志 | `outputs/ct_csd_llada_m16/run.log` |
| 生成结果 | `outputs/jbb_dija_ct_csd_m16/results.json` |
| 生成日志 | `outputs/jbb_dija_ct_csd_m16/run.log` |
| CT-CSD 诊断 | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |
| 评判结果 | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` |
| 评判日志 | `outputs/jbb_dija_ct_csd_m16/judge.log` |

## bank 构造结果

| 项目 | 数值 |
|---|---:|
| bank 格式 | `ct_csd_v1` |
| 向量形状 | `(16, 4096)` |
| `skipped_pass1` | `0` |
| `skipped_pass2` | `0` |

`cluster_sizes` 为：

```text
[4453, 46106, 253268, 87014, 107056, 59533, 72073, 14335, 323, 72716, 32378, 121351, 31465, 172888, 67231, 7634]
```

## 核心指标

阶段 1 核心指标只记录 `ASR` / `unsafe_count` / `total_samples`。

| 指标 | 数值 |
|---|---:|
| `total_samples` | `100` |
| `unsafe_count` | `74/100` |
| `ASR` | `74.0%` |

等价摘要：`ASR = 74.0%`，`unsafe_count = 74/100`，`total_samples = 100`。

## 诊断统计

| 诊断项 | 数值 |
|---|---:|
| `num_clusters` | `16` |
| `total_routed` | `45775` |
| `total_active` | `13155` |
| `activation_rate` | `0.2873839432004369` |
| `route_time_sec` | `0.14139450204675086` |

`route_count` 为：

```text
[14, 2300, 2992, 5871, 8363, 2290, 2954, 159, 51, 2453, 1513, 8520, 494, 3813, 3540, 448]
```

`active_count` 为：

```text
[14, 1173, 371, 4049, 2006, 290, 20, 134, 51, 396, 1128, 344, 467, 8, 2691, 13]
```

## 阶段 0 对比

| 方法 | `ASR` | `unsafe_count` | `total_samples` |
|---|---:|---:|---:|
| 阶段 0 Global Sentence-CSD | `74.0%` | `74/100` | `100` |
| 阶段 1 CT-CSD bank | `74.0%` | `74/100` | `100` |

阶段 1 在当前固定口径下与阶段 0 的 `ASR` 和 `unsafe_count` 持平。该结果说明 CT-CSD bank 最小闭环已经跑通并完成同口径冻结；它不构成阶段 1 优于阶段 0 的证据。
