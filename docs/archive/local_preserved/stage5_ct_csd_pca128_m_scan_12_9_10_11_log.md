# Stage 5 降维版：CT-CSD PCA128 簇数细扫（M12 优先 + M9/10/11）× 3 轮重复实验日志

> 本文档为**运行中日志**，边运行边补充。为 `docs/stage2_ct_csd_m_scan_9_10_11_log.md`
> （l2_only 版）的**降维对应版本**，除 `feature_preprocess` 与 M 点位顺序外全部口径对齐。
> 全部完成后在 §6 汇总结论，并同步回 Stage 进度文档。

## 1. 背景与动机

- Stage 5 定义（`docs/plan/dlm_steering_project_improvement_plan.md` §5、§8）：
  `token_selection=all` + `feature_preprocess=center_pca128_l2`，即 clustering / routing
  特征替换为 `Normalize(P_128(h-μ))`（mean-centering + PCA 128 维投影 + L2），
  steering vector 构造空间不变。此前 Stage 5 **从未跑过**（见 `docs/stage4_abcd_progress.md` §1）。
- 本轮任务来自用户 2026-07-02 指令：对 l2 版 M9/10/11 细扫
  （`docs/stage2_ct_csd_m_scan_9_10_11_log.md`）跑降维对应版本；计划评审后用户决定
  **优先跑 M12**，直接与已知最优点 l2 M12 三轮（65/70/68，均值 67.7）配对比较。
- 点位顺序：**12 → 9 → 10 → 11**。M12 三轮完成（预计 ~14h）即可先下"PCA 是否有收益"
  的判断；若无收益可停掉后续 9/10/11。M9/10/11 与 GPU0 同期 l2 版扫描产出逐 M 配对。
- 单次 ASR 噪声约 ±2.5 个百分点（推理非确定），故每点固定 bank 重复 **3 轮**。

### 1.1 与 l2 版并发运行的说明（偏离"单 GPU 顺序跑"决策的理由）

- l2 版 m_scan 正在 GPU0 运行（预计 ~24h）；本实验占用**空闲的 GPU1**，两条实验各占一卡。
- 此前"只用一个 GPU 顺序跑"的决策针对 l2 扫描自身不拆双卡；本次是独立实验，等待 GPU0
  跑完再串行将浪费 ~24h 卡时。两脚本均支持断点续跑，若确认并发拖慢可随时停掉重排。
- 首次试运行（14:05–14:1x，后因计划评审暂停）核查：GPU0 l2 版 pass1 瞬时速率 ~1.8 it/s
  （比其早期均值 0.91 s/it 更快），**未观察到并发变慢**。GPU1 PCA 拟合阶段 ~3.0 s/it
  是该阶段自身的 CPU gram 累计瓶颈（每样本 4096×4096 矩阵累计），非并发所致。

## 1.2 环境

| 项目 | 值 |
|---|---|
| Python | `/root/miniconda3/bin/python`（base env） |
| 推理模型 | `/dev/shm/LLaDA-8B-Instruct` |
| 评判模型 | `/dev/shm/Llama-Guard-4-12B` |
| GPU | GPU1（Tesla V100 32GB × 2 中的 1 号卡；0 号卡同期跑 l2 版 m_scan） |
| 工作目录 | `/root/myproject/DLM_Steering_Remasking` |

## 2. 评测口径（与 l2 版细扫完全对齐，唯一差异加粗）

| 项目 | 口径 |
|---|---|
| bank 方法 | 全局 `ct_csd`（非 category-aware） |
| token selection | `all`（默认，不做 token 筛选） |
| 特征预处理 | **`center_pca128_l2`**（mean-centering + PCA128 + L2，Stage 5 规格） |
| CSD 构造数据 | HarmBench 9605 有害样本 + 20 拒答改写，`seed = 42` |
| 推理数据 | JBB + DIJA，100 样本，`dija_mask_counts = 128` |
| 关键参数 | `target_layer=31`、`sampling_steps=128`、`mask_length=128`、`block_size=128`、`alignment_threshold=0.0`、`steering_overshoot=1.0`、`initial_steering_ratio=0.1`、`max_refinement_iters=5` |
| 评判器 | 本地 Llama-Guard：`/dev/shm/Llama-Guard-4-12B` |
| 代码版本 | main `d4569a2` + 工作区 `utils/make_ct_csd_llada.py` 未提交改动（同 l2 版：纯注释，无行为变更） |

推理端说明：bank 内嵌 `preprocess` 状态（mean + pca_components），
`CTCSDBank.route()` 加载后自动在 PCA 空间路由，**推理命令与 l2 版完全一致**，
无需额外参数（`docs/stage4_abcd_progress.md` 提到的 `--steering_preprocess_path`
在当前代码中不存在，也不需要）。

## 3. 执行方式

- 脚本：`scripts/run_ct_csd_m_scan_repeat_pca.sh`（新增；`run_ct_csd_m_scan_repeat.sh`
  的 PCA128 版，独立成文件是因原脚本正在 tmux 中运行、不修改运行中脚本）。用法
  `bash scripts/run_ct_csd_m_scan_repeat_pca.sh <gpu_id> <n_rounds> <m1> [m2 ...]`。
  逻辑同 l2 版：每 M 建一次 bank（已存在则跳过），复用 bank 顺序跑 N 轮推理+评判，
  轮次级断点续跑。
- 启动前验证（2026-07-02 14:0x）：
  1. `bash -n` 语法通过；
  2. `unittest tests.test_make_ct_csd_llada -k preprocess` 4 项全过；
  3. GPU1 上 2 样本冒烟建 bank（`center_pca128_l2`, M=9）成功，加载校验
     `preprocess mode=center_pca128_l2`、`pca_components=(128,4096)`、
     `route_centers=(9,128)`、`route/steer` 输出形状正确。
- 启动命令（tmux 后台，GPU1 顺序执行，M12 优先）：

```bash
tmux new-session -d -s m_scan_pca -c /root/myproject/DLM_Steering_Remasking \
  "bash scripts/run_ct_csd_m_scan_repeat_pca.sh 1 3 12 9 10 11"
```

- 主日志：`outputs/stage5_ct_csd_pca128_m_scan_gpu1.log`
- 耗时预估：PCA 版建 bank 比 l2 版多一遍完整前向（Stage 5 fit route preprocess，
  实测 ~3.0 s/it ≈ 8h，CPU gram 累计瓶颈）+ 两遍聚类 ≈3h ⇒ **≈11h/bank**；
  推理 ≈1h/轮，评判 ≈2min/轮；每 M ≈14h，四个 M 合计 **≈56h**
  （M12 结果 ~14h 内先出，可据此决定是否继续 9/10/11）。

### 3.1 脚本内部实际执行的命令（以 M=12 为例）

与 l2 版 §3.1 完全一致，仅两处差异：

1. bank 构造追加 `--feature_preprocess "center_pca128_l2"`，输出目录
   `outputs/ct_csd_llada_m12_pca128`；
2. 推理/评判目录为 `outputs/jbb_dija_ct_csd_m12_pca128_r{r}`，
   `--steering_vector_path` 指向 PCA bank，其余参数逐项相同。

### 3.2 监控与断点续跑

```bash
# 总进度（主日志只记录阶段节点，含时间戳）
cat outputs/stage5_ct_csd_pca128_m_scan_gpu1.log

# 当前阶段细节（PCA 拟合 / 聚类 / 推理进度条）
tail -c 800 outputs/ct_csd_llada_m12_pca128/run.log
tail -c 800 outputs/jbb_dija_ct_csd_m12_pca128_r1/run.log

# tmux 会话
tmux ls                          # 应看到 m_scan_pca（及 GPU0 的 m_scan）
tmux attach -t m_scan_pca        # 进入查看（Ctrl-b d 退出）

# 单轮 ASR 快速读取
/root/miniconda3/bin/python -c "import json; print(json.load(open('outputs/jbb_dija_ct_csd_m12_pca128_r1/llama_guard_results.json'))['metadata'])"
```

- 中断恢复：重新执行 §3 的 tmux 启动命令即可（bank 已存在则跳过；已评判轮次跳过；
  推理半途中断的轮次整轮重跑）。
- 若 M12 三轮显示 PCA 无收益、决定不跑 9/10/11：`tmux kill-session -t m_scan_pca`
  即可（当前 M 的半成品 bank/轮次按上述规则续跑或废弃）。

## 4. 产物路径

| 产物 | 路径 |
|---|---|
| bank | `outputs/ct_csd_llada_m{12,9,10,11}_pca128/ct_csd_bank.pt` |
| bank 构造日志 | `outputs/ct_csd_llada_m{12,9,10,11}_pca128/run.log` |
| 生成结果（第 r 轮） | `outputs/jbb_dija_ct_csd_m{M}_pca128_r{r}/results.json` |
| 诊断（第 r 轮） | `outputs/jbb_dija_ct_csd_m{M}_pca128_r{r}/ct_csd_diagnostics.json` |
| 评判结果（第 r 轮） | `outputs/jbb_dija_ct_csd_m{M}_pca128_r{r}/llama_guard_results.json` |

## 5. 运行时间线（边运行边补充）

| 时间 (UTC) | 事件 |
|---|---|
| 2026-07-02 14:03 | 启动前验证完成：单测 4/4 通过、GPU1 冒烟 bank 构建+加载校验通过 |
| 2026-07-02 14:05 | 首次试运行启动（点位 9/10/11），PCA 拟合 ~3.0 s/it；并发核查 GPU0 未受影响 |
| 2026-07-02 14:1x | 用户要求先评审计划，停止试运行（`m_scan_pca` kill，GPU1 显存归零） |
| 2026-07-02 14:27 | 计划评审通过，点位调整为 **12 → 9 → 10 → 11**，重新启动 tmux `m_scan_pca` |
| 2026-07-02 22:18 | `M=12` PCA bank 建成（实耗 7.85h：PCA 拟合后半程提速至 ~1.0 s/it，快于预估） |
| 2026-07-02 23:20 | `M=12 r1` 评判完成：ASR 70.0% |
| 2026-07-03 00:22 | `M=12 r2` 评判完成：ASR 70.0% |
| 2026-07-03 01:25 | `M=12 r3` 评判完成：ASR 72.0%；**M12 三轮全部完成**，随即开始 `M=9` PCA bank 构造 |
| 2026-07-03 03:57 | 快照：`M=9` PCA 拟合 8101/9605（84%，~1.0 s/it），预计 ~04:25 进入聚类 |
| 2026-07-03 04:24 | 用户决策：M12 已证实 PCA128 无收益，**停止 9/10/11 队列**（kill `m_scan_pca`，当时 M9 拟合 9525/9605；GPU1 显存归零）。GPU1 转投 M12 预处理消融（`center_l2` / `center_pca256_l2`，见新日志文档），并先改进 `fit_route_preprocess`（GPU 累加 + 统计量缓存） |

## 6. 结果汇总（完成一轮填一行）

| M | 轮次 | `unsafe_count` | `ASR` | `total_routed` | `total_active` | `activation_rate` | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | r1 | 70 | 70.0% | 45435 | 10372 | 0.2283 | 完成 |
| 12 | r2 | 70 | 70.0% | 45775 | 10154 | 0.2218 | 完成 |
| 12 | r3 | 72 | 72.0% | 45655 | 10256 | 0.2246 | 完成 |
| 9 | r1 | — | — | — | — | — | 未开始 |
| 9 | r2 | — | — | — | — | — | 未开始 |
| 9 | r3 | — | — | — | — | — | 未开始 |
| 10 | r1 | — | — | — | — | — | 未开始 |
| 10 | r2 | — | — | — | — | — | 未开始 |
| 10 | r3 | — | — | — | — | — | 未开始 |
| 11 | r1 | — | — | — | — | — | 未开始 |
| 11 | r2 | — | — | — | — | — | 未开始 |
| 11 | r3 | — | — | — | — | — | 未开始 |

### 每 M 汇总（3 轮完成后填写）

| M | ASR 均值 | ASR 极差 | 对照 l2 版 | PCA − l2 差值 | 判断 |
|---:|---:|---:|---:|---:|---|
| 12 | 70.7 | 2 | 65/70/68，均值 67.7（已有） | **+3.0** | **无收益，持平至略差**（差值 3 分接近 3 轮均值分辨力边缘，但方向明确不是改善；activation_rate 0.22 与 l2 版量级相当） |
| 9 | — | — | （待 GPU0 扫描） | — | — |
| 10 | — | — | （待 GPU0 扫描） | — | — |
| 11 | — | — | （待 GPU0 扫描） | — | — |

## 7. 结论（全部完成后填写）

待补充。
