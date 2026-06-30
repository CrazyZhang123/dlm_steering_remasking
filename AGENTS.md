# AGENTS

## Stage 代码接收约定

- 所有 Stage 阶段的代码生成、行为变更和实验入口变更，必须完成 **3 轮 review** 后才可以被接收。
- 第 1 轮 review 检查方案一致性：确认实现只覆盖当前 Stage 的新增变量，没有混入后续 Stage 的功能。
- 第 2 轮 review 检查代码质量与测试：确认实现符合 KISS、YAGNI、DRY，相关单元测试和必要回归测试已经运行。
- 第 3 轮 review 检查验收产物：确认命令、输出路径、诊断文件和 metrics 文档满足当前 Stage 的退出标准。
- 任一轮 review 发现 Critical 或 Important 问题时，必须先修复并重新 review；不得把未闭环问题带入下一 Stage。

## 实验运行约定

- 运行实验命令时，统一使用 `tmux` 窗口或会话启动。
- 不要直接依赖前台终端会话运行长任务，避免因为会话中断导致实验进程失联或退出。
- 对于耗时较长的 baseline、steering、judge、评测任务，默认采用 `tmux` 后台运行，并将日志重定向到对应输出目录。

## LLaDA 模型恢复约定

- 本项目默认使用 `/dev/shm/LLaDA-8B-Instruct` 作为 `LLaDA-8B-Instruct` 本地模型路径。
- 每次进入本项目且发现 `/dev/shm/LLaDA-8B-Instruct` 不存在或不完整时，先用 ModelScope 恢复模型：

```bash
python "scripts/restore_llada_model.py"
```

- 该脚本默认从 ModelScope 模型 `GSAI-ML/LLaDA-8B-Instruct` 下载到 `/dev/shm/LLaDA-8B-Instruct`，并会在目标目录已经完整时自动跳过下载。
- 如果需要手动指定下载并发或路径，使用：

```bash
python "scripts/restore_llada_model.py" \
  --model_id "GSAI-ML/LLaDA-8B-Instruct" \
  --local_dir "/dev/shm/LLaDA-8B-Instruct" \
  --max_workers 4
```

## Llama-Guard 模型恢复约定

- 本项目默认使用 `/dev/shm/Llama-Guard-4-12B` 作为本地 safety judge 模型路径。
- 每次进入本项目且发现 `/dev/shm/Llama-Guard-4-12B` 不存在或不完整时，先用 ModelScope 恢复模型：

```bash
python "scripts/restore_llama_guard_model.py"
```

- 该脚本默认从 ModelScope 模型 `LLM-Research/Llama-Guard-4-12B` 下载到 `/dev/shm/Llama-Guard-4-12B`，并会在目标目录已经完整时自动跳过下载。
- 如果需要手动指定下载并发或路径，使用：

```bash
python "scripts/restore_llama_guard_model.py" \
  --model_id "LLM-Research/Llama-Guard-4-12B" \
  --local_dir "/dev/shm/Llama-Guard-4-12B" \
  --max_workers 4
```

## DIJA 数据使用约定

- `DIJA/` 目录在本项目中主要作为攻击数据和 refined prompt 的来源。
- 复现本项目论文表格时，不要直接运行 `DIJA/` 子项目中的实验脚本；应使用本项目 README 中的评估入口，例如 `eval_llada_steering.py`、`eval_dream_steering.py` 和 `scripts/` 下的评估脚本。
- 评估参数优先依据论文、README 示例和当前入口脚本的 argparse 默认值；不要用 `DIJA/` 子项目脚本的参数覆盖本项目设置。
- 对于 Table 2 的 `JailBreakBench + DIJA`，数据输入使用 `DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json`，但推理和评估仍走本项目入口。

## 进度监控约定

- 对于“盯着跑”“观察进度”类请求，默认采用低频监控，**除非用户明确要求，否则不要做秒级、分钟级的高频轮询**。
- 默认监控频率为**每 30 分钟一次**；如果距离上一次检查未到 30 分钟，则继续等待，不要追加新的探测命令。
- 在同一轮任务里，如果刚做过一次进度检查，则在满 30 分钟前，**禁止**再次执行任何新的状态探测命令（包括读取快照、扫描输出目录、查询 tmux 会话），除非用户明确要求“现在就查”。
- 监控时优先读取已有进度快照、状态文件和后台输出目录；不要为了“确认一下”而反复执行重复检查。
- 如果用户明确要求“先等着”“不要再检测”“不要没事找事”，则停止主动检查，直到用户再次明确要求恢复监控。

## Git 提交约定

- 仓库已初始化（`git init` 完成）；在新机器或新克隆上开工前，先确认仓库存在且 `remote` 指向正确地址，缺失时再初始化。
- 完成阶段性修改后，遵守 **Conventional Commit** 规范提交：`<type>(<scope>): <subject>`，`type` 取 `feat` / `fix` / `docs` / `refactor` / `test` / `chore` 等；按逻辑拆分为多次提交，不要把互不相关的改动堆进同一个 commit。
- **只提交代码相关内容**：源码、脚本、测试、文档、配置；**不提交**数据集、模型权重、生成/评测产物（`models/`、`data/`、`outputs/`、`*.tar.gz`、`*.zip` 等已在 `.gitignore`）。
- **第三方子项目（如 `DIJA/`、`gpt-oss/`）整体忽略**，不纳入本仓库；对其的必要改动以 patch 形式保存到 `scripts/patches/`，复现时 `cd DIJA && git apply ../scripts/patches/<name>.patch` 还原。
- `push` 等外部不可逆操作前，先确认 `remote` 地址正确；除非用户明确要求，否则不主动 commit / push。
