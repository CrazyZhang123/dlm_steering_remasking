# 会话说明：Stage 4A token-selection 实验与 baseline 稳定性验证（2026-06-30 ~ 07-02）

本文档记录一次跨三天的工作会话：起因、做了什么、改了哪些文件、实验结果、
途中事故与处理、最终结论与建议。供后续会话/协作者快速接上下文。

## 1. 起因与任务演变

1. 用户发现 `docs/` 缺少 Stage 4 ABCD（4A/4C/4D/5/6）的进度文档，想跑 Stage 4A
   （direction-selected token selection），暂不跑降维（Stage 5）。
2. 会话中任务逐步演变为：
   - 核对仓库实际状态（worktree 是否已合并、代码是否入库、测试是否覆盖真实路径）；
   - 补齐 ABCD 进度文档；
   - 跑 Stage 4A 两组参数（plan §12.5 口径 r0.5/global 与代码默认 r0.3/category）；
   - 用户纠正 base：应该用**全局 `ct_csd` M12（ASR=65%，当前最优）**而非
     `category_ct_csd`，据此停掉错误 base 的轮次并重跑；
   - 加随机对照（random_top_ratio）验证 direction 是否优于随机；
   - **重跑 baseline 验证 65% 的稳定性**（run-to-run 方差），再扫 direction 的
     selection_ratio ∈ {0.3, 0.7}。
   - 在确认 token-selection 全线无稳定收益后，用户进一步决定：**不再等待**
     direction `r0.3/r0.7` 两个半途轮次，而是停掉它们，切换到 Stage 2 的
     `ct_csd M9/M10/M11 × 3 轮重复实验`，检验 M12 周边是否存在更优簇数点位。

## 2. 仓库核对结论（客观手段：AST / git / unittest）

- Stage 4A/4C/4D/5 代码**已合并进 main**（`978c191` CT-CSD bank、`d4569a2` per-class KNN），
  原 worktree `.worktrees/stage4a-stage5-stage6/` 已移除。
- bank 构造的真实 token-selection 路径是 `select_harmful_response_tokens` →
  `top_ratio_select` → `_coarse_direction_for_sample`；`select_tokens` /
  `select_tokens_direction_top_ratio` 是调用次数为 0 的死代码（旧接口残留）。
- **真实路径已有单测覆盖**（`test_direction_top_ratio_selects_tokens_by_category_direction_with_global_fallback`
  等），全模块 `python -m unittest tests.test_make_ct_csd_llada` = 47 tests OK。
  会话早期"真实路径无覆盖"的说法是误判，已更正。
- 环境：无 `dlm_steering` conda env；`diffuguard` 缺 sklearn；**唯一全依赖齐全的是
  base python（/root/miniconda3/bin/python）**，无 pytest（用 unittest 跑测试）。
- `eval_llada_steering.py` 中 `set_seed` 函数**定义了但从未被调用**，生成路径含
  `rand_like` 随机——推理天然非确定，同配置重跑结果不同（这是 baseline 稳定性
  实验可行的前提，也意味着历史所有单次 ASR 都带噪声）。

## 3. 实验总账（均 100 样本 / JBB+DIJA / 本地 Llama-Guard / l2_only 不降维 / layer 31）

### 3.1 baseline 稳定性（同一 ct_csd M12 bank，重复推理+评判）

| run | ASR |
|---|---|
| 原始（历史） | 65.0% |
| rerun1 | 70.0% |
| rerun2 | 68.0% |

**极差 5 个百分点（65~70），单次 ASR 噪声约 ±2.5%。**

### 3.2 token-selection 各方案

| 方案 | bank | ASR | activation_rate |
|---|---|---:|---:|
| baseline（全 token） | category M16 | 71.0% | 0.161 |
| MIL probe τ0.7（历史 Stage 4） | probe category M16 | 71.0% | 0.128 |
| direction r0.5/global | category M16 | 72.0% | 0.180 |
| direction r0.3/category | category M16 | 74.0% | 0.185 |
| baseline（全 token） | **ct M12** | **65~70%（3 次）** | 0.292 |
| direction r0.5/global | ct M12 | 68.0% | 0.195 |
| random r0.5 | ct M12 | 70.0% | 0.238 |
| direction r0.3/global | ct M12 | **已停掉，未取结果** | — |
| direction r0.7/global | ct M12 | **已停掉，未取结果** | — |

### 3.3 结论

1. **baseline 单次 ASR 噪声 ±2.5%**，此前所有"±3 以内"的差异（含 direction 的
   68 vs 65）都落在噪声带内。
2. direction(68%) 与 random(70%) 与 baseline 三次重跑（65/70/68）互相重叠——
   **token-selection（MIL / direction / random 三条线）在当前框架下均无可证实的正收益**。
3. Stage 4D KNN：3 个 bank（smoke/standard/balanced）retention 均 99%+，
   与全 token bank 近乎等价，**未跑 ASR**（预计≈baseline，信息量低）。
4. 建议关闭 token-selection 方向；后续更有希望的是 **steering 超参**
   （`steering_overshoot`、`initial_steering_ratio`、`alignment_threshold`）——
   注意任何单点结论都需考虑 ±2.5% 噪声，必要时同配置跑 2~3 次取均值。
5. 用户在本会话末段选择的下一步不是继续补 token-selection 尾单，而是先做
   **Stage 2 `num_total_clusters` 细扫复核**：`M=9/10/11` 各跑 3 轮，和 M12 的
   `65/70/68` 比较均值与极差，减少被单次噪声误导的概率。

## 4. 本会话新增/修改的文件

- `docs/stage4_abcd_progress.md`：新建，ABCD 全景进度 + 各轮结果 + KNN 现状 +
  超参扫描计划（§3.5~3.8）。
- `docs/session_notes_20260630_0702_stage4a_tokensel.md`：本文件。
- `docs/stage2_ct_csd_m_scan_9_10_11_log.md`：新建，记录 `ct_csd M9/M10/M11`
  单卡顺序跑、每点 3 轮重复推理+评判的运行中日志。
- `scripts/run_stage4a_direction.sh`：Stage 4A（category bank）三段式脚本（早期版本）。
- `scripts/run_stage4_tokensel.sh`：泛化版，支持 `bank_method`（ct_csd/category_ct_csd）
  与 `token_selection`（direction/random/all）参数化。
- `scripts/run_infer_judge.sh`：复用已有 bank 只跑推理+评判（baseline 重跑用）。
- `scripts/run_ct_csd_m_scan_repeat.sh`：新增，复用单个 bank 连续跑多轮推理+评判，
  用于 `M=9/10/11` 细扫重复实验。
- `CLAUDE.md`：修正两处过时的 worktree 引用（指向已合并 main 与新进度文档）。
- `utils/make_ct_csd_llada.py`：**最小修复**——L1425 一行中文注释缺 `#` 导致
  SyntaxError（该处属于工作区一批 +140 行的外部未提交改动，非本会话所写），仅补
  `# `，修复后 47 单测通过。修复前备份：`/tmp/make_ct_csd_backup_before_fix.py`。

## 5. 遗留状态与会话衔接（截至 2026-07-02 13:39 UTC）

- 用户已决定**不再等待** direction `r0.3/r0.7` 两轮：
  `tmux kill-session -t s4_gpu0` 与 `tmux kill-session -t s4_gpu1` 已执行，
  GPU 0/1 显存已释放；因此 Stage 4A 的遗留后台任务已经清空。
- 当前唯一长任务是 `tmux m_scan`，执行：

  ```bash
  bash "scripts/run_ct_csd_m_scan_repeat.sh" 0 3 9 10 11
  ```

  即在 **单 GPU 顺序**模式下做 `ct_csd M9/M10/M11 × 3` 轮重复实验。
- 2026-07-02 13:39 UTC 快照：`M=9` 仍在建 bank，`outputs/ct_csd_llada_m9/run.log`
  显示 `Stage 1 pass 1` 进度 `1811/9605`（18.9%），已运行 `27m22s`；尚未进入
  `M=9 r1` 推理，`outputs/jbb_dija_ct_csd_m9_r1/` 还不存在。
- 本会话后续若继续跟进批量实验，**主承接文档**应切到
  `docs/stage2_ct_csd_m_scan_9_10_11_log.md`；本文件到此主要承担
  “Stage 4A token-selection 结论 + 为什么转向 M 扫描”的会话总结角色。
- `utils/make_ct_csd_llada.py` 工作区仍有 +140 行外部未提交改动（含 per-sample
  probe 诊断类注释），语义未经本会话验证，谁在开发需用户确认。
- `assets/paper_methods_en.md` 曾被误提交进 git（本会话临时产物），工作区副本已删，
  git 里显示为待删除状态，尚未提交删除。

## 6. 方法论备注（重要，后续会话适用）

本会话工具回传对**中文密集文件**出现过多次内容污染（"读到"不存在的代码，如
`hidden_mask_index`、return 后死代码、`gate` 死变量、错误的 pytest 计数等）。
处理原则：**凡下结论必须用客观手段复核**——AST 解析/调用计数、`git diff --numstat`、
`python -m unittest` 实跑、真实 torch 前向验证；肉眼逐行读中文源码的结果只作线索、
不作结论。本文所有数字均来自 JSON/unittest/git 的程序化输出。
