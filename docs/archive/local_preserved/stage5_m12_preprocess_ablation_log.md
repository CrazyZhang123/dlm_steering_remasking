# Stage 5 补充：M12 预处理消融（center_l2 / center_pca256_l2）× 3 轮实验日志

> 本文档为**运行中日志**，边运行边补充。全部完成后在 §6 汇总结论，
> 并同步回 `docs/stage5_ct_csd_pca128_m_scan_12_9_10_11_log.md` 与 Stage 进度文档。

## 1. 背景与动机

- 已有结果（均全局 `ct_csd` M12、100 样本 JBB+DIJA、本地 Llama-Guard、3 轮）：
  - `l2_only`：65/70/68，均值 **67.7**；
  - `center_pca128_l2`：70/70/72，均值 **70.7**（+3.0，无收益）。
- 但 pca128 相对 l2_only 同时改了**去均值**与**PCA 投影**两件事，+3 分无法归因。
  本消融补两个点拆开变量（用户 2026-07-03 决策）：
  - **`center_l2`**（只去均值、不降维）：若 ≈67.7 → 去均值无害、劣化来自投影；
    若 ≈70.7 → 去均值本身有害；
  - **`center_pca256_l2`**（保留 256 维）：若明显低于 70.7 → 128 维砍掉了有用信息；
    若仍 ≈70+ → 降维路线可盖棺。

## 1.1 前置代码改进（本轮新增，已过 3 轮 review）

`utils/make_ct_csd_llada.py` 的 `fit_route_preprocess` 两项改进：

1. **统计量 GPU 累加**：原实现每个样本把 token 向量搬回 CPU 做 4096×4096 gram 累加
   （拟合遍前段 ~3s/样本的主因）；现改为在 token 所在设备累加、收尾统一回 CPU。
2. **`--preprocess_stats_cache`**：`token_sum`/`gram`/`token_count` + 口径元数据
   （target_layer / token_selection / selection_ratio / max_response_len / num_samples）
   落盘；命中缓存则校验元数据后**跳过整个拟合遍**（约省 3h/模式），同一份统计量可
   派生 `center_l2` 与任意 `center_pca{N}_l2`。mean/gram 与目标维度无关是其数学依据。

Review 与验证记录：

- R1 方案一致性：仅改预处理性能/缓存；`l2_only` 提前返回未动（GPU0 同期 l2 扫描
  调用本文件建 M11 bank 不受影响）；不带新参数时行为等价（float32 精度不变）。
- R2 代码质量：抽出 `_preprocess_from_stats` 消除拟合/缓存两路径重复（DRY）；
  review 中补上元数据缺 `selection_ratio` 的复用隐患；
  `unittest tests.test_make_ct_csd_llada` **49/49 OK**（新增 2 个缓存单测：
  缓存命中禁止再走前向、缓存派生与直接拟合逐项一致、口径不匹配报错）；
  `tests.test_ct_csd_bank` 11/11 OK。
- R3 验收产物：GPU1 真实 CLI 冒烟（2 样本）——`center_l2` 建 bank 落缓存 →
  `center_pca256_l2` 日志出现 `Loaded preprocess stats cache ... (skip fit pass)`；
  两 bank 均可被 `CTCSDBank` 加载，`route_centers` 分别为 `(12,4096)` / `(12,256)`，
  `route`/`steer` 形状正确。

## 2. 评测口径（与 M12 l2/pca128 完全对齐，唯一差异加粗）

| 项目 | 口径 |
|---|---|
| bank 方法 | 全局 `ct_csd`，`num_total_clusters=12` |
| token selection | `all` |
| 特征预处理 | **`center_l2` 与 `center_pca256_l2`**（两个 bank） |
| CSD 构造数据 | HarmBench 9605 有害样本 + 20 拒答改写，`seed = 42` |
| 推理数据 | JBB + DIJA，100 样本，`dija_mask_counts = 128` |
| 关键参数 | `target_layer=31`、`sampling_steps=128`、`mask_length=128`、`block_size=128`、`alignment_threshold=0.0`、`steering_overshoot=1.0`、`initial_steering_ratio=0.1`、`max_refinement_iters=5` |
| 评判器 | 本地 Llama-Guard：`/dev/shm/Llama-Guard-4-12B` |
| 代码版本 | main `d4569a2` + 工作区 `fit_route_preprocess` 改进（见 §1.1，待提交） |

## 3. 执行方式

- 脚本：`scripts/run_ct_csd_m12_preprocess_ablation.sh`（新增）。用法
  `bash scripts/run_ct_csd_m12_preprocess_ablation.sh <gpu_id> <n_rounds> <mode1> [mode2 ...]`。
  两模式共享统计量缓存 `outputs/ct_csd_llada_m_all_preprocess_stats.pt`
  （命名不含 M：统计量与簇数无关，后续任何 M / 任何维度都可复用）；
  bank 级与轮次级断点续跑同 m_scan 系列脚本。
- 启动命令（tmux 后台，GPU1）：

```bash
tmux new-session -d -s m12_prep_abl -c /root/myproject/DLM_Steering_Remasking \
  "bash scripts/run_ct_csd_m12_preprocess_ablation.sh 1 3 center_l2 center_pca256_l2"
```

- 主日志：`outputs/stage5_m12_preprocess_ablation_gpu1.log`
- 耗时预估（按 M9 pca128 实测拟合遍均值 ~1.1s/it ≈3h、聚类两遍 ≈4.8h、推理 1h/轮）：
  `center_l2` ≈ 拟合 ≤3h（GPU 累加应更快）+ 聚类 4.8h + 3 轮 ≈ **10.5h**；
  `center_pca256_l2` ≈ 拟合 0h（命中缓存）+ 聚类 4.8h + 3 轮 ≈ **8h**；合计 ≈ **18.5h**。

## 4. 产物路径

| 产物 | 路径 |
|---|---|
| 统计量缓存 | `outputs/ct_csd_llada_m_all_preprocess_stats.pt` |
| bank | `outputs/ct_csd_llada_m12_{center_l2,center_pca256_l2}/ct_csd_bank.pt` |
| bank 构造日志 | `outputs/ct_csd_llada_m12_{mode}/run.log` |
| 生成/诊断/评判（第 r 轮） | `outputs/jbb_dija_ct_csd_m12_{mode}_r{r}/{results,ct_csd_diagnostics,llama_guard_results}.json` |

## 5. 运行时间线（边运行边补充）

| 时间 (UTC) | 事件 |
|---|---|
| 2026-07-03 04:24 | 停止 GPU1 上 pca128 M9/10/11 队列（M12 已出"无收益"结论），GPU1 释放 |
| 2026-07-03 04:2x | `fit_route_preprocess` 改进完成：49/49 单测 OK + 3 轮 review + GPU1 CLI 冒烟通过 |
| 2026-07-03 04:30 | tmux `m12_prep_abl` 启动，`[center_l2] build ct_csd bank M=12` 开始 |
| 2026-07-03 13:24 | `center_l2` bank 建成并完成 r1 推理+评判（ASR 72.0%） |
| 2026-07-03 14:27 | `center_l2` r2 完成（ASR 71.0%） |
| 2026-07-03 15:29 | `center_l2` r3 完成（ASR 74.0%）；`[center_l2] all rounds done`，随即开始 `[center_pca256_l2] build ct_csd bank M=12` |
| 2026-07-03 16:29 | 快照：`center_pca256_l2` bank `Stage 1 pass 1: fit clusters` 进度 41%（3956/9605，~1.3 it/s），已运行约 1h；pass 1 预计还需 ~72min，之后 pass 2 累加 + 3 轮推理评判，保守估计还要数小时。GPU1 利用率 100%。（GPU0 上 `batch_accept` 会话在跑批量推理等价性验收，与本消融无关） |
| 2026-07-03 16:38 | 快照：`center_pca256_l2` bank pass 1 进度 47%（4490/9605，速度回升至 ~1.77 it/s），已运行 1h09m；按当前速度 pass 1 剩 ~48min，pass 2 累加 + 3 轮推理评判仍需数小时。进程存活、无报错 |

## 6. 结果汇总（完成一轮填一行）

| mode | 轮次 | `unsafe_count` | `ASR` | `activation_rate` | 状态 |
|---|---:|---:|---:|---:|---|
| center_l2 | r1 | 72 | 72.0% | 0.1923 | 完成 |
| center_l2 | r2 | 71 | 71.0% | 0.1872 | 完成 |
| center_l2 | r3 | 74 | 74.0% | 0.1822 | 完成 |
| center_pca256_l2 | r1 | 68 | 68.0% | — | 完成 |
| center_pca256_l2 | r2 | 66 | 66.0% | — | 完成 |
| center_pca256_l2 | r3 | 70 | 70.0% | — | 完成 |

### 汇总对比（M12 固定，全部完成）

| 预处理 | 维度 | ASR 三轮 | 均值 | vs l2_only(67.7) |
|---|---:|---|---:|---|
| l2_only | 4096 | 65/70/68 | **67.7** | — |
| center_l2 | 4096 | 72/71/74 | 72.3 | +4.6（更差） |
| center_pca128_l2 | 128 | 70/70/72 | 70.7 | +3.0（更差） |
| center_pca256_l2 | 256 | 68/66/70 | **68.0** | +0.3（≈持平） |

## 7. 结论（全部完成）

**拆变量归因成功，且 pca256 修正了此前预期**：

- `center_l2`（只去均值、不降维）三轮均值 **72.3**，比 `l2_only` 基线 67.7 高 **+4.6pp**，
  超出单次 ±2.5pp 噪声带 → **"去均值"这一步本身有害**。
- `center_pca256_l2` 均值 **68.0**，与基线 67.7 基本持平（+0.3，落在噪声内）。三点位连起来
  呈现清晰规律：**降维越多，越能抵消"去均值"的损害**——
  center_l2(4096维,+4.6) → pca128(128维,+3.0) → pca256(256维,+0.3)。
  256 维恰好把去均值的损害几乎补回，但**只追平 l2_only，未超过**。
  （此前 §7 预测"pca256 翻盘概率极低"被结果修正：pca256 并非更差，而是持平。）
- 四点位（l2_only 67.7 ≈ pca256 68.0 < pca128 70.7 < center_l2 72.3）一致指向：**在
  route/聚类特征上做 center / PCA 预处理对本任务无正收益，最好也只是打平 `l2_only`。**

**Stage 5 盖棺**：整条 feature preprocessing 线（center / PCA 各维度）已验证**无正收益**，
最优配置仅能追平不做处理的 `l2_only`。后续不再投入该方向。

**activation_rate 观察**：center_l2 三轮 `activation_rate` 均约 0.18–0.19，低于 l2_only 基线
的 ~0.29（M12）。即"去均值"改变了 route 后触发 steering 的 token 比例，但更低的触发率并未
换来更低的 ASR，反而更高——进一步支持"预处理干扰了有效干预"的判断。

**后续（已启动）**：feature preprocessing 与 token-selection 两条 bank 表示线均走到头后，
转向**推理时引导超参扫描**（复用 M12 l2 bank，不重建），见
`docs/plan/steering_hparam_scan_plan.md`，2026-07-04 02:17 双卡启动。

**activation_rate 观察**：center_l2 三轮 `activation_rate` 均约 0.18–0.19，低于 l2_only 基线
的 ~0.29（M12）。即"去均值"改变了 route 后触发 steering 的 token 比例，但更低的触发率并未
换来更低的 ASR，反而更高——进一步支持"预处理干扰了有效干预"的判断。

> 关联：本轮负结果已纳入 `docs/plan/asr_reduction_improvement_plan.md` §3 现状证据表，作为
> "bank 表示这条线走到头"的又一佐证；该计划建议转向 steering 超参与重掩码机制改进。
