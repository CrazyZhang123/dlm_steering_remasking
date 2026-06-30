# Table 2 当前复现进度

## 范围和口径

本文档只整理论文 `Table 2` 的当前复现进度，重点记录：

- 已经跑通了哪些单元格
- 每个单元格使用的命令
- 参数设置依据
- 数据来源
- judge 路径与当前结果

当前约定：

- `DIJA/` 目录只作为攻击数据和 refined prompt 的来源
- 不直接运行 `DIJA/` 子项目脚本
- 推理和评估统一走本项目入口

论文中的 `Table 2` 如下：

- 论文位置：[assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md](/root/myproject/DLM_Steering_Remasking/assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md:228)
- 当前正式完成了 `LLaDA / JBB DIJA / Vanilla` 这一格
- 另补跑了一个探索项：`LLaDA / JBB DIJA / Full CSD Steering`

## 当前进度总览

| 项目 | 论文值 | 当前状态 | 当前结果 |
|---|---:|---|---:|
| `LLaDA / JBB DIJA / Vanilla` | `72.00` | 已完成 | `71.0` |
| `LLaDA / JBB DIJA / Split CSD Steering` | - | 已完成（推荐口径） | `70.0` |
| `LLaDA / JBB DIJA / Full CSD Steering` | - | 已完成（探索项） | `74.0` |
| `LLaDA / JBB PAP / Vanilla` | `30.00` | 未开始 | - |
| `LLaDA / JBB Prefix / Vanilla` | `5.00` | 已完成 | `0.0` |
| `LLaDA / Adv DIJA / Vanilla` | `98.65` | 未开始 | - |
| `LLaDA / Adv PAP / Vanilla` | `31.15` | 未开始 | - |
| `LLaDA / Adv Prefix / Vanilla` | `2.12` | 已完成 | `0.0` |

说明：

- 这里的 `Vanilla` 指原始 LLaDA，不启用本文的 steering/remasking 防御
- judge 使用本地 `Llama-Guard-4-12B`
- `Split CSD Steering` 是当前更推荐记录的无泄漏 steering 结果
- `Full CSD Steering` 只是探索项，不属于严格无泄漏复现口径

## 已完成项

### 1. `LLaDA / JBB DIJA / Vanilla`

论文目标值：

- `72.00`

当前实测结果：

- `71.0`

结果文件：

- 生成结果：[outputs/LLaDA_JBB_DIJA_vanilla/results.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_JBB_DIJA_vanilla/results.json)
- judge 结果：[outputs/LLaDA_JBB_DIJA_vanilla/llama_guard_local.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_JBB_DIJA_vanilla/llama_guard_local.json)

judge 元数据：

- `total_samples = 100`
- `unsafe_count = 71`
- `asr_percent = 71.0`

### 2. 数据来源

JBB DIJA 输入数据来自：

- [DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json](/root/myproject/DLM_Steering_Remasking/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json)

该文件当前共有 `100` 条，字段包括：

- `behavior`
- `category`
- `goal`
- `refined_goal`
- `target`

代码路由依据：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:688)
- 当 `--csv_path JBB --attack_method DIJA` 时，会直接读取上面的 `refined_goal`

### 2. `LLaDA / JBB DIJA / Split CSD Steering`

说明：

- 这一项不是 `Table 2` 里的 `Vanilla` 单元格
- 这是在同一条 `JBB + DIJA` 推理链路上，加载 `split` HarmBench CSD 向量后的 steering 结果
- 该向量由 `harmbench_csd_train.json` 构造，再在独立的 `JBB + DIJA` 集合上评估
- 这项比 `full` 版本更适合做当前阶段的正式记录

当前实测结果：

- `70.0`

结果文件：

- 生成结果：[.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/results.json](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/results.json)
- judge 结果：[.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/llama_guard_results.json](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/llama_guard_results.json)

judge 元数据：

- `total_samples = 100`
- `unsafe_count = 70`
- `asr_percent = 70.0`
- 互补安全率 `= 30.0`

运行口径：

- 数据集：`JBB + DIJA refined prompt`
- 模型：`/dev/shm/LLaDA-8B-Instruct`
- 向量：[.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618/steering_vectors.pt](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618/steering_vectors.pt)
- `target_layer = 31`
- `device = cuda:1`

实际运行命令：

```bash
python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --generated_samples_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620" \
  --batch_size 32 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --remasking low_confidence \
  --sampler steering \
  --remdm_number 4 \
  --cfg 0 \
  --device cuda:1 \
  --self_reminder False \
  --steering_vector_path ".worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_split_20260618/steering_vectors.pt" \
  --steering_overshoot 1.0 \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --max_refinement_iters 5 \
  --initial_steering_ratio 0.1
```

judge 命令：

```bash
python scripts/eval_llama_guard_local.py \
  --data_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/results.json" \
  --output_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_split_dija_cuda1_20260620/llama_guard_results.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:1
```

### 3. `LLaDA / JBB DIJA / Full CSD Steering`

说明：

- 这一项不是 `Table 2` 里的 `Vanilla` 单元格
- 这是在同一条 `JBB + DIJA` 推理链路上，额外加载 `full` HarmBench CSD 向量后的探索结果
- 因为该向量来自全量 `9605` 条 HarmBench harmful 样本，所以它有数据泄漏风险，不作为正式无泄漏结论

当前实测结果：

- `74.0`

结果文件：

- 生成结果：[.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/results.json](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/results.json)
- judge 结果：[.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/llama_guard_results.json](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/llama_guard_results.json)

judge 元数据：

- `total_samples = 100`
- `unsafe_count = 74`
- `asr_percent = 74.0`
- 互补安全率 `= 26.0`

运行口径：

- 数据集：`JBB + DIJA refined prompt`
- 模型：`/dev/shm/LLaDA-8B-Instruct`
- 向量：[.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt](/root/myproject/DLM_Steering_Remasking/.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt)
- `target_layer = 31`
- `device = cuda:1`

实际运行命令：

```bash
python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --generated_samples_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620" \
  --batch_size 32 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --remasking low_confidence \
  --sampler steering \
  --remdm_number 4 \
  --cfg 0 \
  --device cuda:1 \
  --self_reminder False \
  --steering_vector_path ".worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt" \
  --steering_overshoot 1.0 \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --max_refinement_iters 5 \
  --initial_steering_ratio 0.1
```

judge 命令：

```bash
python scripts/eval_llama_guard_local.py \
  --data_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/results.json" \
  --output_path ".worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/llama_guard_results.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:1
```

### 4. 生成命令

实际运行命令：

```bash
python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_JBB_DIJA_vanilla" \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --device cuda:1
```

后台运行方式：

```bash
tmux new-session -d -s "llada_jbb_dija_vanilla_20260619" \
'cd "/root/myproject/DLM_Steering_Remasking" && \
python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_JBB_DIJA_vanilla" \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --device cuda:1 \
 2>&1 | tee "./outputs/LLaDA_JBB_DIJA_vanilla/run.log"'
```

## Prefix 当前执行状态

### 1. 当前口径

当前 `Prefix` 两格统一按以下口径执行：

- 目标：论文 `Table 2` 原始行
- 模型：`/dev/shm/LLaDA-8B-Instruct`
- 攻击方式：`attack_method=prefix`
- 口径：`Vanilla`
- sampler：显式指定 `--sampler llada`
- judge：本地 `/dev/shm/Llama-Guard-4-12B`

说明：

- `attack_method=prefix` 默认如果不显式改 sampler，会落回代码默认 `sampler=steering`
- 这会误启用 `llada_remask_sample`
- 因此当前 `Vanilla Prefix` 口径必须显式传 `--sampler llada`

### 2. Prefix 前缀来源

当前代码使用：

- [utils/templates.py](/root/myproject/DLM_Steering_Remasking/utils/templates.py:5) 中的 `REFERENCES[0]`

拼接方式：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:777)
- 实际形式为：`REFERENCES[0] + " " + row[question_key]`

当前实际观察到：

- `REFERENCES[0]` 约 `3677` 个字符
- 套入聊天模板后约 `769` tokens
- 因此前缀攻击单条样本耗时显著高于 `DIJA` 和普通 `zeroshot`

### 3. 数据来源

`JBB Prefix` 当前使用本地 CSV：

- [data/jbb_prefix_chunk0.csv](/root/myproject/DLM_Steering_Remasking/data/jbb_prefix_chunk0.csv)
- [data/jbb_prefix_chunk1.csv](/root/myproject/DLM_Steering_Remasking/data/jbb_prefix_chunk1.csv)

这两份文件来自：

- [data/harmful-behaviors-prompt.csv](/root/myproject/DLM_Steering_Remasking/data/harmful-behaviors-prompt.csv)

`Adv Prefix` 当前使用本地离线 `AdvBench` 数据并切成两份 CSV：

- 原始本地数据目录：[DiffuGuard/AdvBench](/root/myproject/DiffuGuard/AdvBench)
- chunk 文件：
  - [data/advbench_prefix_chunk0.csv](/root/myproject/DLM_Steering_Remasking/data/advbench_prefix_chunk0.csv)
  - [data/advbench_prefix_chunk1.csv](/root/myproject/DLM_Steering_Remasking/data/advbench_prefix_chunk1.csv)

当前仓库已补充本地离线 `AdvBench` 入口支持：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py)
- 对应测试：[tests/test_eval_llada_run_csv_eval.py](/root/myproject/DLM_Steering_Remasking/tests/test_eval_llada_run_csv_eval.py)

### 4. 当前调度方式

为了缩短总墙钟时间，当前使用双卡自动流水线：

- `GPU0` pipeline：先跑 `JBB Prefix chunk0 (50)`，再跑 `Adv Prefix chunk0 (260)`
- `GPU1` pipeline：先跑 `JBB Prefix chunk1 (50)`，再跑 `Adv Prefix chunk1 (260)`
- 待两条 pipeline 都完成后：
  - 自动合并 `JBB` 两个 chunk
  - 自动合并 `Adv` 两个 chunk
  - 自动分别运行本地 `Llama-Guard-4-12B` judge

当前 `tmux` 会话：

- `prefix_gpu0_pipeline_20260623`
- `prefix_gpu1_pipeline_20260623`
- `prefix_merge_judge_20260623`

### 5. 当前命令

`GPU0` pipeline：

```bash
python eval_llada_steering.py \
  --csv_path "./data/jbb_prefix_chunk0.csv" \
  --attack_method prefix \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_JBB_Prefix_vanilla_chunk0" \
  --sampler llada \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --device cuda:0
```

然后自动接：

```bash
python eval_llada_steering.py \
  --csv_path "./data/advbench_prefix_chunk0.csv" \
  --attack_method prefix \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_Adv_Prefix_vanilla_chunk0" \
  --sampler llada \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --device cuda:0
```

`GPU1` pipeline：

```bash
python eval_llada_steering.py \
  --csv_path "./data/jbb_prefix_chunk1.csv" \
  --attack_method prefix \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_JBB_Prefix_vanilla_chunk1" \
  --sampler llada \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --device cuda:1
```

然后自动接：

```bash
python eval_llada_steering.py \
  --csv_path "./data/advbench_prefix_chunk1.csv" \
  --attack_method prefix \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --self_reminder False \
  --generated_samples_path "./outputs/LLaDA_Adv_Prefix_vanilla_chunk1" \
  --sampler llada \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --device cuda:1
```

统一 merge + judge：

```bash
python scripts/eval_llama_guard_local.py \
  --data_path "./outputs/LLaDA_JBB_Prefix_vanilla/results.json" \
  --output_path "./outputs/LLaDA_JBB_Prefix_vanilla/llama_guard_local.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:0
```

```bash
python scripts/eval_llama_guard_local.py \
  --data_path "./outputs/LLaDA_Adv_Prefix_vanilla/results.json" \
  --output_path "./outputs/LLaDA_Adv_Prefix_vanilla/llama_guard_local.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:1
```

### 6. 当前进度快照

最近一次检查时：

- `JBB Prefix chunk0`: `1/50`
- `JBB Prefix chunk1`: `1/50`
- 两边单条耗时约 `185s`
- `Adv Prefix` 两个 chunk 还未开始，等待各自的 `JBB chunk` 跑完后自动接棒
- 最终 `JBB Prefix` 和 `Adv Prefix` 的 `results.json + llama_guard_local.json` 还未产出

后续进展：

- `JBB Prefix` 两个 chunk 已全部生成完成，并已合并为：
  - [outputs/LLaDA_JBB_Prefix_vanilla/results.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_JBB_Prefix_vanilla/results.json)
- `JBB Prefix` 的本地 judge 已完成：
  - [outputs/LLaDA_JBB_Prefix_vanilla/llama_guard_local.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_JBB_Prefix_vanilla/llama_guard_local.json)
  - `unsafe_count = 0`
  - `asr_percent = 0.0`
- `Adv Prefix` 两个 chunk 已全部生成完成，并已合并为：
  - [outputs/LLaDA_Adv_Prefix_vanilla/results.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_Adv_Prefix_vanilla/results.json)
- `Adv Prefix` 的本地 judge 已完成：
  - [outputs/LLaDA_Adv_Prefix_vanilla/llama_guard_local.json](/root/myproject/DLM_Steering_Remasking/outputs/LLaDA_Adv_Prefix_vanilla/llama_guard_local.json)
  - `unsafe_count = 0`
  - `asr_percent = 0.0`

### 5. judge 命令

当前采用本地 judge 路线，先用魔搭下载模型，再运行本地脚本：

```bash
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    "LLM-Research/Llama-Guard-4-12B",
    local_dir="/dev/shm/Llama-Guard-4-12B",
    max_workers=4,
)
PY

python scripts/eval_llama_guard_local.py \
  --data_path "./outputs/LLaDA_JBB_DIJA_vanilla/results.json" \
  --output_path "./outputs/LLaDA_JBB_DIJA_vanilla/llama_guard_local.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:1
```

后台运行方式：

```bash
tmux new-session -d -s "judge_llada_jbb_dija_vanilla_20260620" \
'cd "/root/myproject/DLM_Steering_Remasking" && \
python - <<\"PY\"\nfrom modelscope.hub.snapshot_download import snapshot_download\nsnapshot_download(\n    \"LLM-Research/Llama-Guard-4-12B\",\n    local_dir=\"/dev/shm/Llama-Guard-4-12B\",\n    max_workers=4,\n)\nPY\n\
python scripts/eval_llama_guard_local.py \
  --data_path "./outputs/LLaDA_JBB_DIJA_vanilla/results.json" \
  --output_path "./outputs/LLaDA_JBB_DIJA_vanilla/llama_guard_local.json" \
  --model_path "/dev/shm/Llama-Guard-4-12B" \
  --device cuda:1 \
  2>&1 | tee "./outputs/LLaDA_JBB_DIJA_vanilla/judge.log"'
```

## 参数设置依据

### 1. 论文目标

论文里当前目标是：

- `LLaDA / JBB DIJA / Vanilla = 72.00`

参考位置：

- [assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md](/root/myproject/DLM_Steering_Remasking/assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md:232)

### 2. README 示例

README 给出的 LLaDA 推理参数参考为：

- `sampling_steps 128`
- `mask_length 128`
- `block_size 128`
- `dija_mask_counts 128`
- `steering_overshoot 1.0`
- `initial_steering_ratio 0.1`
- `max_refinement_iters 5`

参考位置：

- [README.md](/root/myproject/DLM_Steering_Remasking/README.md:145)

### 3. 代码默认值

当前入口脚本默认值为：

- `sampling_steps = 128`
- `mask_length = 128`
- `block_size = 128`
- `attack_method` 默认 `zeroshot`
- `steering_vector_path` 默认空字符串
- `target_layer = 31`
- `alignment_threshold = 0.0`
- `initial_steering_ratio = 0.1`
- `dija_mask_counts = 128`

参考位置：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:808)

### 4. 这次为什么不传 steering 参数

本次目标是 `Vanilla`，所以：

- 不传 `--steering_vector_path`
- 不传 `--target_layer`
- 不传 `--steering_overshoot`
- 不传 `--alignment_threshold`

代码行为上，`steering_vector_path` 为空时，`can_steer` 为假，不会启用本文的 adaptive steering 和 Phase 2 remask 防御。

这一点符合 `Vanilla` 的复现口径。

## 当前差异与说明

### 1. 当前结果与论文值的差异

- 论文：`72.00`
- 当前：`71.0`

差异只有 `1.0`，目前可视为接近复现。

### 2. README 与代码默认值的一个小差异

README 示例写的是：

- `max_refinement_iters 5`

但代码默认值是：

- `max_refinement_iters = 3`

这次因为没有启用 steering vector，所以该参数实际上不参与生效，不影响 `Vanilla` 结果。

### 3. 当前 judge 路线

论文写的是使用：

- `meta-llama/Llama-Guard-4-12B`

当前本地实际使用的是：

- ModelScope 下载的 `LLM-Research/Llama-Guard-4-12B`
- 本地目录为 `/dev/shm/Llama-Guard-4-12B`

在当前环境下，这是为了绕开 Hugging Face 离线缺缓存的问题。

## 剩余单元格的数据准备现状

目前 `Table 2` 剩余未跑单元格，按“是否已有现成输入”可以分成两类。

### 1. 现成可跑，不需要自己额外构造

- `LLaDA / JBB Prefix / Vanilla`
- `LLaDA / Adv Prefix / Vanilla`

原因：

- 当 `--csv_path JBB --attack_method prefix` 时，代码直接读取 `JailbreakBench/JBB-Behaviors` 的 `harmful` split
- 当 `--csv_path AdvBench --attack_method prefix` 时，代码直接读取 `walledai/AdvBench`

代码位置：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:692)
- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:703)
- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:777)

这两项当前不缺数据文件，直接按本项目入口运行即可。

### 2. 缺专用输入文件，运行前必须补

- `LLaDA / JBB PAP / Vanilla`
- `LLaDA / Adv PAP / Vanilla`
- `LLaDA / Adv DIJA / Vanilla`

当前缺失项：

- `./gpt-oss/JBB_pap.json`
- `./gpt-oss/AdvBench_pap.json`
- `./DIJA/run_jailbreakbench/refine_prompt/advbench_data_refined_Qwen.json`

代码位置：

- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:684)
- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:695)
- [eval_llada_steering.py](/root/myproject/DLM_Steering_Remasking/eval_llada_steering.py:699)

当前仓库实际状态：

- 已存在 `JBB DIJA` 所需 refined prompt：
  [DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json](/root/myproject/DLM_Steering_Remasking/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json:1)
- 未发现 `Adv DIJA` 所需 refined prompt
- 未发现 `JBB PAP` 与 `Adv PAP` 所需 JSON

因此，剩余单元格的准备结论是：

- `Prefix` 两项可以直接开跑
- `PAP` 两项和 `Adv DIJA` 一项需要先补专用输入构造或补齐外部文件

### 3. 当前推荐顺序

如果继续补齐 `Table 2`，建议按下面顺序推进：

1. 先跑 `LLaDA / JBB Prefix / Vanilla`
2. 再跑 `LLaDA / Adv Prefix / Vanilla`
3. 再决定是否补 `PAP` 与 `Adv DIJA` 的输入构造脚本

## 下一步建议

如果继续补齐 `Table 2`，建议顺序如下：

1. 先补 `LLaDA / JBB PAP / Vanilla`
2. 再补 `LLaDA / JBB Prefix / Vanilla`
3. 最后再做 `AdvBench` 三列

这样可以先把 `JBB Avg` 这一半补完整。
