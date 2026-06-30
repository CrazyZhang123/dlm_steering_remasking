# HarmBench CSD Data Design

## Goal

将 `HarmfulGeneration-HarmBench` 的 `parquet` 测试集接入当前仓库，导出为现有 CSD 构造流程可直接消费的 `json`，并同步产出便于人工检查与统计的 `csv`。

## Confirmed Decisions

### 1. Source Dataset

- 输入文件：
  - `data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet`
- 数据字段：
  - `behavior`
  - `test_case`
  - `answer`
  - `behavior_id`
  - `functional_category`
  - `semantic_category`

### 2. Prompt / Response Mapping

- `prompt = test_case`
- `response = answer`

选择原因：

- `test_case` 是具体的 jailbreak prompt，更贴近论文中“同一条攻击 prompt 下的 harmful / refusal 对比”设定。
- `behavior` 是抽象行为描述，可作为元数据保留，但不作为当前 CSD 的主 prompt。

### 3. Output Formats

产出两个文件：

1. `data/harmbench_testcase_harmful.json`
2. `data/harmbench_testcase_harmful.csv`

#### JSON schema

每条记录至少包含：

```json
{
  "prompt": "<test_case>",
  "response": "<answer>",
  "behavior": "<behavior>",
  "behavior_id": "<behavior_id>",
  "functional_category": "<functional_category>",
  "semantic_category": "<semantic_category>"
}
```

说明：

- 当前 `make_csd_llada.py` / `make_csd_dream.py` 只强依赖 `prompt` 和 `response`。
- 其余字段作为保留元数据，便于后续抽样、统计和错误分析。

#### CSV columns

列名与 JSON 字段保持一致：

- `prompt`
- `response`
- `behavior`
- `behavior_id`
- `functional_category`
- `semantic_category`

## Pair Construction Design

### Harmful Side

- 直接使用导出数据中的 `response = answer`
- 这些样本已经是对有害 `test_case` 的顺从式 harmful 回复

### Safe Side

- 继续复用 `utils/refusals.txt`
- 不在当前阶段额外生成或收集“真实拒答回复”数据

### Pairing Strategy

运行 `make_csd_llada.py` / `make_csd_dream.py` 时动态构造 pair：

- `prompt = test_case`
- `y_harm = answer`
- `y_safe = 从 refusal 池随机采样的一条拒答`

这样保持与现有仓库实现一致，也与论文摘要描述一致：

- harmful 侧来自真实越狱成功回复
- safe 侧来自固定 refusal paraphrases

## Data Cleaning Rules

导出阶段执行轻量清洗：

1. 跳过空 `test_case`
2. 跳过空 `answer`
3. 默认按 `prompt = test_case` 去重，保留首条记录

当前不额外引入更复杂的规则，例如：

- 按 `prompt + response` 双键去重
- 按类别重采样
- 基于长度或分类器过滤

这些都留作后续增强，不纳入当前范围。

## Out of Scope

当前设计不包含以下内容：

- 改造 `make_csd_llada.py` / `make_csd_dream.py` 直接读取 `parquet`
- 增加新的 refusal 生成流程
- 构造显式预配对的 `pair.jsonl`
- 运行完整 steering / judge / eval 实验

## Next Execution Steps

1. 新增 `parquet -> json/csv` 导出脚本
2. 为导出逻辑补单元测试
3. 用新导出的 `json` 作为 `--harmful_json` 跑一次 smoke 级 CSD
4. 若 smoke 正常，再全量生成 steering vector
