# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本目录 `DLM_Steering_Remasking/` 工作时提供指导。

> **强制前置**：本目录下另有 `AGENTS.md`，其中定义了 **Stage 接收约定、实验运行约定、模型恢复约定、DIJA 数据约定、进度监控约定**。开始任何 Stage 开发、跑实验或监控进度前，必须先读 `AGENTS.md` 并遵守，本文件不重复其全部内容，只做要点索引。

## 项目概述

本项目实现论文《扩散语言模型中用于安全生成的自适应引导与重掩码》——一个**无需训练**的 DLM 安全框架。核心思想：

1. **对比安全方向 (CSD)**：构造潜在方向，区分 `有害回复` 与 `安全拒绝回复` 的语义差异。
2. **早期步骤自适应引导**：在去噪早期抑制隐藏表示中的有害语义方向（有害对齐越强、引导越强）。
3. **有害令牌重掩码**：去噪后选择性重掩码可疑令牌并重新生成，保持流畅性。

目标模型：**LLaDA-8B-Instruct** 与 **Dream**。评测基准：`JailBreakBench`、`AdvBench`、`TruthfulQA`、`MATH-500`、`MMLU`。代码基于 LLaDA / Dream / ReMDM-LLaDA。

## 环境配置

```bash
conda create -n dlm_steering python=3.10
conda activate dlm_steering
pip install -r requirements.txt   # torch / transformers==4.49.0 / scikit-learn 等
mkdir -p outputs
```

关键本地模型路径（详见 `AGENTS.md`，缺失时先用恢复脚本下载到 `/dev/shm`）：
- LLaDA：`/dev/shm/LLaDA-8B-Instruct` → `python scripts/restore_llada_model.py`
- Llama-Guard：`/dev/shm/Llama-Guard-4-12B` → `python scripts/restore_llama_guard_model.py`

## 核心工作流

```bash
# 1. 生成对比安全方向 (CSD)
python utils/make_csd_llada.py
python utils/make_csd_dream.py
# Category-aware CT-CSD bank（Stage 1+）
python utils/make_ct_csd_llada.py

# 2. 推理（编辑脚本中的参数后运行；长任务务必用 tmux）
sh scripts/llada_steer.sh        # 入口 eval_llada_steering.py
sh scripts/dream_steer.sh        # 入口 eval_dream_steering.py
# attack_method: zeroshot | PAP | DIJA | prefix

# 3. 评估
sh scripts/llama_guard.sh        # ASR 安全评测（JBB / AdvBench）
sh scripts/test_rouge_score.sh   # TruthfulQA
sh scripts/mmlu_eval.sh          # MMLU 准确率
sh scripts/math-500_eval.sh      # MATH-500 准确率
```

推理核心参数（见 README 与各入口 `argparse` 默认值）：`--target_layer`、`--sampling_steps`、`--steering_overshoot`、`--initial_steering_ratio`、`--max_refinement_iters`。评估参数优先依据论文 / README / 入口默认值，**不要**用 `DIJA/` 子项目脚本参数覆盖本项目设置。

## Stage 体系（Category-aware CT-CSD）

本项目的实验以 Stage 推进，进度与指标统一记录在 `docs/` 下，评测口径默认 `100` 条样本、JBB+DIJA 生成、本地 Llama Guard 评判，核心指标比较 `unsafe_count` / `ASR`。

| Stage | 方法要点 | 主文档 |
|---|---|---|
| 0 | 全局 Sentence-CSD 冻结 baseline | `docs/category_aware_ct_csd_stage_progress.md` |
| 1 | CT-CSD M16 + hard routing + 阈值门控 | 同上 + `stage3_*` |
| 2 | `num_total_clusters` 4/8/12/16 消融（M12 当前最低 ASR=65%） | `docs/category_aware_ct_csd_stage1_metrics.md` |
| 3 | Category-aware clustering（按 `semantic_category` 分组） | `docs/stage3_category_ct_csd_*.md` |
| 4 | MIL token probe + probe 阈值选择有害 token | `docs/stage4_mil_token_probe_*.md` |
| 4A/5/6 | token selection / feature preprocessing / 组合 metadata（开发中） | `.worktrees/stage4a-stage5-stage6/` |

进行中的 Stage 4A/5/6 工作位于 worktree `.worktrees/stage4a-stage5-stage6/`，改动集中在 `utils/make_ct_csd_llada.py`、`utils/ct_csd_bank.py` 及对应 tests。

## 关键目录

- `eval_llada_steering.py` / `eval_dream_steering.py`：两个模型的推理 + 引导/重掩码主入口。
- `utils/`：`make_csd_*.py`（CSD 生成）、`make_ct_csd_llada.py`（CT-CSD bank 构造）、`ct_csd_bank.py`（bank 路由/预处理）、`llama_guard.py`、`train_mil_token_probe_llada.py`、各评测脚本。
- `scripts/`：实验/评测/数据准备 shell 与 py 脚本，模型恢复脚本。
- `tests/`：pytest 单测与回归测试。
- `docs/`：Stage 进度、指标、分析与 `plan/` 计划文档。
- `data/` / `outputs/`：数据输入与生成/评测产物。
- `DIJA/`：仅作攻击数据与 refined prompt 来源，**不**直接运行其实验脚本（详见 `AGENTS.md`）。

## 测试

```bash
pytest tests/                              # 全量
pytest tests/test_ct_csd_bank.py -q        # 单文件
pytest tests/test_make_ct_csd_llada.py -q
```

Stage 代码变更须满足 `AGENTS.md` 的 **3 轮 review**（方案一致性 / 代码质量与测试 / 验收产物），相关单测与必要回归测试通过后方可接收。

## 代码风格

- Python PEP 8，4 空格缩进；公共函数显式类型标注；导入顺序 stdlib → 第三方 → 本地。
- 路径用 `pathlib`/`os`，避免硬编码绝对路径；命令中文件路径用双引号包裹。
- 脚本通过 `argparse` 保持数据集无关；显式设置 `--device` / `CUDA_VISIBLE_DEVICES`。
- 固定随机种子（默认 `42`）保证可复现。
- 内容搜索优先 `rg`；优先使用 Read/Write/Edit 等专用工具。
- 代码注释语言与现有代码库保持一致。

## 重要约定（摘自 AGENTS.md，细节以原文为准）

- **长任务一律 tmux** 后台运行，日志重定向到输出目录，不依赖前台会话。
- **进度监控低频**：默认每 30 分钟一次，未到 30 分钟不重复探测；用户要求“先等着/不要再检测”时停止主动检查。
- **未主动要求时，绝不**主动计划或执行 git 提交、分支等操作；用户明确要求提交时，遵守 **Conventional Commit** 规范，只提交代码相关内容，忽略数据/模型/第三方子项目（详见 `AGENTS.md`「Git 提交约定」）。
- 大产物（checkpoint、CSV、模型、数据集、`*.tar.gz`/`*.zip` 打包、第三方子项目 `DIJA/`）不入库；对第三方子项目的改动以 patch 存 `scripts/patches/`。
