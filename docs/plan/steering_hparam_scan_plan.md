# 计划（待运行）：steering 超参单变量扫描

> 状态：**已启动**（2026-07-04 02:17 UTC 双卡）。脚本 `scripts/run_steering_hparam_scan.sh`；
> 主日志 `outputs/steering_hparam_scan_gpu{0,1}.log`；tmux `steer_os`（GPU0：1.5:0.1 / 2.0:0.1）、
> `steer_isr`（GPU1：1.0:0.2 / 1.0:0.3）。前置条件（M12 消融、l2 扫描）已全部完成。
> 依据 `docs/stage4_abcd_progress.md` §3.8 细化为可执行方案。

## 1. 动机

- token-selection 三条线（MIL / direction / random）与 feature preprocessing
  （pca128 已证无收益；center_l2 / pca256 消融进行中）都作用于"怎么分组/表示"，
  实测 ASR 全部挤在 65–74%，多数差异在 run 噪声（±2.5%/次）内。
- `steering_overshoot` 与 `initial_steering_ratio` 直接控制**干预强度与覆盖步数**，
  自项目开始一直用默认值（1.0 / 0.1），从未扫过——是当前最可能出正收益的方向。
- 关键优势：**不重建 bank**，复用现成最优 base（全局 `ct_csd` M12 l2_only bank），
  每个配置只需 ~1h 推理 + ~2min 评判。

## 2. 方案

- base：`outputs/ct_csd_llada_m12/ct_csd_bank.pt`（l2_only，M12，ASR 基线 65/70/68，均值 67.7）。
- 单变量扫描，其余参数与历史口径一致（100 样本 JBB+DIJA、`target_layer=31`、
  `sampling_steps=128`、`alignment_threshold=0.0` 固定、本地 Llama-Guard）：

| 组 | steering_overshoot | initial_steering_ratio | 直觉 |
|---|---:|---:|---|
| baseline | 1.0 | 0.1 | 已有 3 轮（65/70/68），不重跑 |
| O1 | **1.5** | 0.1 | 更强推离有害方向 |
| O2 | **2.0** | 0.1 | 进一步加强，观察是否过冲 |
| R1 | 1.0 | **0.2** | 早期 steering 覆盖前 20% 去噪步 |
| R2 | 1.0 | **0.3** | 覆盖前 30% 去噪步 |

- 每个配置 **3 轮**（对齐 ±2.5% 噪声口径），共 4 配置 × 3 轮 ≈ **12h 推理 + 评判**（单卡）。
- 若 O 组或 R 组出现单调改善，追加组合点（如 1.5/0.2）二阶验证。

## 3. 质量护栏（防过度 steering）

ASR 之外必须同看生成质量，警惕"全拒答/不流畅换来的低 ASR"：

1. 抽查每配置 r1 的 `results.json` 生成文本（≥10 条），确认非乱码、非空、非全模板拒答；
2. 对最优配置补跑 TruthfulQA（`sh scripts/test_rouge_score.sh`）或 MMLU 子集，
   与 baseline 比较，回归幅度 >2 分则判为过度 steering；
3. 记录 `ct_csd_diagnostics.json` 的 `activation_rate` 变化（overshoot 不影响触发率，
   `initial_steering_ratio` 会提高 routed 总量，应如实记录）。

## 4. 执行方式（启动时照此执行）

- 复用/仿写 m_scan 系列脚本结构：每配置目录
  `outputs/jbb_dija_ct_csd_m12_os{O}_isr{R}_r{r}`，tmux 后台 + 日志重定向 + 断点续跑；
  推理命令仅改 `--steering_overshoot` / `--initial_steering_ratio` 两个 flag，
  `--steering_vector_path` 固定指向 M12 l2 bank。
- GPU：任一空闲卡；4 配置可单卡顺序（~12h）或双卡各 2 配置（~6h，推理阶段并发无建
  bank 的 CPU 争抢问题）。
- 监控按 `AGENTS.md`：30 分钟低频。

## 5. 判读标准

- 与 baseline 均值 67.7 比较：3 轮均值差 ≥5 分才视为真实效应（参考
  `asr-run-noise-baseline`：单次极差可达 5 分）；
- 出现"ASR 降 + 质量护栏全过"→ 候选进入下一 Stage（与最优预处理组合验证）；
- 全部配置无改善 → 记录后转向 base 结构（如论文单方向 CSD 路线回归验证）。

## 6. 前置条件

- [x] GPU1 上 M12 预处理消融（`m12_prep_abl`）完成（2026-07-03 23:26 UTC，Stage 5 盖棺无收益）；
- [x] GPU0 上 l2 版 M9/M10/M11 扫描（`m_scan`）完成（M12 仍为最优簇数，均值 67.7）；
- [x] 用户确认启动（2026-07-04）。
