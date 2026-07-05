# Docs 阅读入口

本目录已经按“主结论 / 运行手册 / 归档”三层收口。

## 最终推荐阅读顺序

### 1. 先看结果

优先阅读：

- `docs/experiment_summary.md`

用途：

- 看当前最佳方法
- 看各阶段 ASR
- 看各类消融（簇数、token selection、preprocess、steering、Sure/Sorry）
- 看最终推荐结论

### 2. 再看怎么复现

然后阅读：

- `docs/reproduction_runbook.md`

用途：

- 查实验入口脚本
- 查统一评测口径
- 查模型准备与结果读取方式

### 3. 如果你关心工程性能

单独阅读：

- `docs/batch_inference_refactor.md`
- `docs/batch_inference_session_log.md`

用途：

- 这是批量推理重构与工程验收文档
- 不属于“方法效果主结果”阅读主线

### 4. 如果你要追原始时间线

去看：

- `docs/archive/`

用途：

- 保存旧的 `progress / log / metrics / reproduction` 文档
- 适合追溯逐轮日志、阶段性判断和历史上下文

### 5. 如果你要看实验规划而不是结果

去看：

- `docs/plan/`

用途：

- 保存各阶段计划、方案和待执行设计
- 不作为最终结果来源

## 一句话导航

- 想知道“现在最好方法是什么”：
  - 看 `docs/experiment_summary.md`
- 想知道“怎么复现这个结果”：
  - 看 `docs/reproduction_runbook.md`
- 想知道“以前每一步是怎么跑出来的”：
  - 看 `docs/archive/`
- 想知道“接下来原本打算怎么做”：
  - 看 `docs/plan/`
