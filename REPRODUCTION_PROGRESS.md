# DLM_Steering_Remasking 进展文档

## 1. 计划是否已落地

当前仓库里已经有一份可执行的分步计划：

- [docs/superpowers/plans/2026-06-15-llada-two-baseline-rounds.md](</root/myproject/DLM_Steering_Remasking/docs/superpowers/plans/2026-06-15-llada-two-baseline-rounds.md>)

这份计划**已经落地为仓库内的正式文档**，并且其核心前置工作已有代码与结果支撑：

- `scripts/prepare_wildjailbreak_prompts.py` 已存在
- `scripts/eval_llama_guard_local.py` 已存在
- `scripts/build_csd_harmful_pairs.py` 已存在
- 对应测试文件已存在

## 2. 当前进度

最新汇总文件：

- [outputs/harmful_collection_progress.json](</root/myproject/DLM_Steering_Remasking/outputs/harmful_collection_progress.json>)

当前状态：

- `harmful_pairs = 147`
- `remaining_to_5763 = 5616`
- `completed_samples = 2610`
- `external_judged_samples = 2600`

## 3. 当前运行中的队列

当前仍在跑的队列目录：

- `outputs/queue_gpu0_chunk_002600_002699`
- `outputs/queue_gpu1_chunk_002700_002799`

它们当前都还只有：

- `results.partial.json`
- `run.log`

尚未看到：

- `results.json`
- `lumingapi_gpt54_judge.json`
- 下一批新目录

## 4. 最近一次已确认的检查结果

自动低频检查会话已经确认到：

- `outputs/oneshot_progress_check_0640.txt`

这些检查显示：

- 进度快照已刷新到 `2026-06-17 06:40:25 UTC`
- 当时的 `harmful_pairs` 为 `147`
- 目前队列已切到 `2600+`

## 5. 说明

这份文档只记录“计划是否落地”和“当前进度”，不包含有害样本内容本身。
