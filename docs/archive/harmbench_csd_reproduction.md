# HarmBench CSD 复现说明

## 范围

本文档记录当前仓库里用于 `LLaDA` 的 `HarmBench -> CSD` 复现路径。

覆盖内容包括：

- 有害数据导出
- train/eval 切分资产
- LLaDA steering vector 构建
- `smoke`、`full`、`split` 三套输出位置
- 下游评估入口

## 脚本映射

- HarmBench 有害数据导出：`.worktrees/harmbench-csd-export/scripts/export_harmbench_testcase_harmful.py`
- LLaDA CSD 构建：`.worktrees/harmbench-csd-export/utils/make_csd_llada.py`
- 运行时路径解析：`.worktrees/harmbench-csd-export/utils/runtime_paths.py`
- LLaDA steering 推理：`eval_llada_steering.py`
- Llama-Guard 评估：`utils/llama_guard.py`

## 训练逻辑

### 1. 导出 HarmBench 有害数据

源 parquet 文件：

- `.worktrees/harmbench-csd-export/data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet`

导出脚本行为：

- 使用 `datasets.load_dataset("parquet", ...)` 加载 parquet 行
- 将 `test_case -> prompt`
- 将 `answer -> response`
- 保留以下元数据字段：
  - `behavior`
  - `behavior_id`
  - `functional_category`
  - `semantic_category`
- 默认去重模式为 `prompt_response`

参考命令：

```bash
python ".worktrees/harmbench-csd-export/scripts/export_harmbench_testcase_harmful.py" \
  --input_parquet ".worktrees/harmbench-csd-export/data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet" \
  --output_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
  --output_csv ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.csv"
```

当前已导出的资产：

- `.worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json`
- `.worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.csv`

### 2. train/eval 切分

当前仓库状态：

- 固定切分资产已经存在
- 本次审计中没有发现单独的切分脚本

当前切分资产：

- `.worktrees/harmbench-csd-export/data/harmbench_csd_train.json`，共 `7684` 条
- `.worktrees/harmbench-csd-export/data/harmbench_csd_eval.json`，共 `1921` 条
- `.worktrees/harmbench-csd-export/data/harmbench_csd_train.csv`
- `.worktrees/harmbench-csd-export/data/harmbench_csd_eval.csv`

操作建议：

- `train` 只用于构建 steering vector
- `eval` 只用于推理和 judge 评估
- 正式无泄漏报告不要使用 `full` vector

### 3. LLaDA steering vector 构建

入口脚本：

- `.worktrees/harmbench-csd-export/utils/make_csd_llada.py`

脚本行为：

- 从 `--harmful_json` 读取 harmful pair
- 从 `./utils/refusals.txt` 读取安全拒答
- 对每条 harmful sample 随机采样一条 refusal
- 为每个 prompt 构造两条序列：
  - `prompt + harmful response`
  - `prompt + safe refusal`
- 提取每个 transformer block 上的 response token hidden state
- 将每段 response 平均成逐层向量
- 计算 `harmful_mean - safe_mean`
- 将结果以 `dict[str, Tensor]` 保存到 `steering_vectors.pt`

实现细节：

- 当前脚本内部直接使用 `torch.device("cuda")`
- 当前版本没有 CLI `--device` 参数

用于正式 split vector 的参考命令：

```bash
python ".worktrees/harmbench-csd-export/utils/make_csd_llada.py" \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_csd_train.json" \
  --output_dir ".worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618"
```

## 现有输出集

### Smoke

- vector：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_smoke_llada_20260618/steering_vectors.pt`
- log：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_smoke_llada_20260618/run.log`

用途：

- 路径联通性检查
- 快速验证整条 pipeline

### Full

- vector：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt`
- log：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/run.log`

已观察到的日志事实：

- 读取了 `9605` 条 harmful sample
- 读取了 `20` 条 refusal paraphrase

用途：

- 只适合探索性实验
- 不建议做正式评估，因为混用了构建集和评估集

### Split

- vector：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618/steering_vectors.pt`
- log：`.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618/run.log`

已观察到的日志事实：

- 聚合了 `7684` 条 train sample
- 已保存到上述 split 输出目录

用途：

- 推荐作为 `harmbench_csd_eval.*` 的正式评估向量

## 下游评估

推理入口：

- `eval_llada_steering.py`

重要路由行为：

- `--csv_path JBB` 且 `--attack_method DIJA` 时，会读取 `./DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json`
- 普通带 `prompt` 列的 CSV 会走 fallback 分支
- `DIJA` 会强制将 sampler 切到 `llada_dija`

当前 LLaDA 运行推荐参数族：

- `sampling_steps=128`
- `mask_length=128`
- `block_size=128`
- `remasking=low_confidence`
- `sampler=steering`
- `remdm_number=4`
- `steering_overshoot=1.0`
- `target_layer=31`
- `initial_steering_ratio=0.1`
- `max_refinement_iters=5`
- `alignment_threshold=0.0`

## 实务说明

- 做 HarmBench 无泄漏报告时，优先用 `split` vector 配合 `harmbench_csd_eval.*`
- 做论文式 JailBreakBench 攻击评估时，优先走专门的 `JBB` 分支，不要误用普通 HarmBench CSV fallback
- 如果要完全复刻历史 split，当前仓库只保留了切分产物，没有保留单独的 split 生成脚本
