# HarmBench CSD Data Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个脚本，将 HarmBench `parquet` 测试集导出为当前 CSD 流程可直接消费的 harmful `json` 与配套审查 `csv`。

**Architecture:** 在 `scripts/` 下新增一个独立导出脚本，负责读取 `parquet`、做轻量清洗、按 `test_case -> prompt` 与 `answer -> response` 映射并写出 `json/csv`。测试放在 `tests/` 下，覆盖字段映射、空值过滤、按 prompt 去重和 CLI 主流程。

**Tech Stack:** Python 3.10, `datasets`, `json`, `csv`, `pathlib`, `unittest`

---

## File Structure

- Create: `scripts/export_harmbench_testcase_harmful.py`
- Create: `tests/test_export_harmbench_testcase_harmful.py`
- Read for pattern reference: `scripts/prepare_wildjailbreak_prompts.py`
- Read for output conventions: `scripts/build_csd_harmful_pairs.py`

### Task 1: Add Focused Failing Tests For Export Logic

**Files:**
- Create: `tests/test_export_harmbench_testcase_harmful.py`
- Read: `tests/test_prepare_wildjailbreak_prompts.py`

- [ ] **Step 1: Write the failing test file**

```python
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.export_harmbench_testcase_harmful as export


class NormalizeRowsTest(unittest.TestCase):
    def test_normalize_rows_maps_test_case_and_answer_and_dedups_by_prompt(self):
        rows = [
            {
                "test_case": " prompt-a ",
                "answer": " answer-a ",
                "behavior": "behavior-a",
                "behavior_id": "id-a",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-a",
                "answer": "answer-a-duplicate",
                "behavior": "behavior-a-dup",
                "behavior_id": "id-a-dup",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-b",
                "answer": "answer-b",
                "behavior": "behavior-b",
                "behavior_id": "id-b",
                "functional_category": "contextual",
                "semantic_category": "fraud",
            },
        ]

        normalized = export.normalize_rows(rows)

        self.assertEqual(
            normalized,
            [
                {
                    "prompt": "prompt-a",
                    "response": "answer-a",
                    "behavior": "behavior-a",
                    "behavior_id": "id-a",
                    "functional_category": "standard",
                    "semantic_category": "illegal",
                },
                {
                    "prompt": "prompt-b",
                    "response": "answer-b",
                    "behavior": "behavior-b",
                    "behavior_id": "id-b",
                    "functional_category": "contextual",
                    "semantic_category": "fraud",
                },
            ],
        )

    def test_normalize_rows_skips_empty_test_case_or_answer(self):
        rows = [
            {
                "test_case": "",
                "answer": "answer-a",
                "behavior": "behavior-a",
                "behavior_id": "id-a",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-b",
                "answer": "   ",
                "behavior": "behavior-b",
                "behavior_id": "id-b",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-c",
                "answer": "answer-c",
                "behavior": "behavior-c",
                "behavior_id": "id-c",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
        ]

        normalized = export.normalize_rows(rows)

        self.assertEqual(
            normalized,
            [
                {
                    "prompt": "prompt-c",
                    "response": "answer-c",
                    "behavior": "behavior-c",
                    "behavior_id": "id-c",
                    "functional_category": "standard",
                    "semantic_category": "illegal",
                }
            ],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest "tests/test_export_harmbench_testcase_harmful.py" -v
```

Expected:

```text
FAIL or ERROR because scripts.export_harmbench_testcase_harmful does not exist yet
```

- [ ] **Step 3: Add CLI-oriented failing tests**

```python
class WriteOutputsTest(unittest.TestCase):
    def test_write_json_and_csv_outputs_expected_fields(self):
        rows = [
            {
                "prompt": "prompt-a",
                "response": "answer-a",
                "behavior": "behavior-a",
                "behavior_id": "id-a",
                "functional_category": "standard",
                "semantic_category": "illegal",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            json_path = tmp / "out.json"
            csv_path = tmp / "out.csv"

            export.write_json(json_path, rows)
            export.write_csv(csv_path, rows)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(payload, rows)
        self.assertEqual(csv_rows, rows)


class MainTest(unittest.TestCase):
    def test_main_reads_parquet_dataset_and_writes_outputs(self):
        fake_dataset = [
            {
                "test_case": "prompt-a",
                "answer": "answer-a",
                "behavior": "behavior-a",
                "behavior_id": "id-a",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-a",
                "answer": "answer-a-dup",
                "behavior": "behavior-a-dup",
                "behavior_id": "id-a-dup",
                "functional_category": "standard",
                "semantic_category": "illegal",
            },
            {
                "test_case": "prompt-b",
                "answer": "answer-b",
                "behavior": "behavior-b",
                "behavior_id": "id-b",
                "functional_category": "contextual",
                "semantic_category": "fraud",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "sample.parquet"
            output_json = tmp / "harmful.json"
            output_csv = tmp / "harmful.csv"

            argv = [
                "export_harmbench_testcase_harmful.py",
                "--input_parquet",
                str(input_path),
                "--output_json",
                str(output_json),
                "--output_csv",
                str(output_csv),
            ]

            with patch.object(export, "load_dataset", return_value={"train": fake_dataset}) as mock_load:
                with patch("sys.argv", argv):
                    export.main()

            mock_load.assert_called_once_with("parquet", data_files=str(input_path))
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(
            payload,
            [
                {
                    "prompt": "prompt-a",
                    "response": "answer-a",
                    "behavior": "behavior-a",
                    "behavior_id": "id-a",
                    "functional_category": "standard",
                    "semantic_category": "illegal",
                },
                {
                    "prompt": "prompt-b",
                    "response": "answer-b",
                    "behavior": "behavior-b",
                    "behavior_id": "id-b",
                    "functional_category": "contextual",
                    "semantic_category": "fraud",
                },
            ],
        )
```

- [ ] **Step 4: Run tests to verify they still fail for the missing implementation**

Run:

```bash
pytest "tests/test_export_harmbench_testcase_harmful.py" -v
```

Expected:

```text
FAIL because normalize_rows / write_json / write_csv / main are not implemented yet
```

### Task 2: Implement The Minimal Export Script

**Files:**
- Create: `scripts/export_harmbench_testcase_harmful.py`
- Test: `tests/test_export_harmbench_testcase_harmful.py`

- [ ] **Step 1: Create the script with row normalization helpers**

```python
import argparse
import csv
import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_FIELDS = [
    "prompt",
    "response",
    "behavior",
    "behavior_id",
    "functional_category",
    "semantic_category",
]


def normalize_rows(rows):
    normalized = []
    seen_prompts = set()

    for row in rows:
        prompt = str((row.get("test_case") or "")).strip()
        response = str((row.get("answer") or "")).strip()
        if not prompt or not response or prompt in seen_prompts:
            continue

        seen_prompts.add(prompt)
        normalized.append(
            {
                "prompt": prompt,
                "response": response,
                "behavior": str((row.get("behavior") or "")).strip(),
                "behavior_id": str((row.get("behavior_id") or "")).strip(),
                "functional_category": str((row.get("functional_category") or "")).strip(),
                "semantic_category": str((row.get("semantic_category") or "")).strip(),
            }
        )

    return normalized
```

- [ ] **Step 2: Add output writers and CLI**

```python
def write_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args(argv)

    dataset = load_dataset("parquet", data_files=str(args.input_parquet))
    rows = normalize_rows(dataset["train"])
    write_json(args.output_json, rows)
    write_csv(args.output_csv, rows)

    print(f"saved {len(rows)} harmful rows to {args.output_json}")
    print(f"saved {len(rows)} harmful rows to {args.output_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the focused test file**

Run:

```bash
pytest "tests/test_export_harmbench_testcase_harmful.py" -v
```

Expected:

```text
PASS
```

### Task 3: Verify The Script Against The Real HarmBench Parquet

**Files:**
- Read: `data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet`
- Use script: `scripts/export_harmbench_testcase_harmful.py`

- [ ] **Step 1: Run the exporter on the real dataset**

Run:

```bash
python "scripts/export_harmbench_testcase_harmful.py" \
  --input_parquet "data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet" \
  --output_json "data/harmbench_testcase_harmful.json" \
  --output_csv "data/harmbench_testcase_harmful.csv"
```

Expected:

```text
saved <N> harmful rows to data/harmbench_testcase_harmful.json
saved <N> harmful rows to data/harmbench_testcase_harmful.csv
```

- [ ] **Step 2: Sanity-check the generated files**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/harmbench_testcase_harmful.json")
rows = json.loads(path.read_text(encoding="utf-8"))
print("count", len(rows))
print("first_keys", list(rows[0].keys()))
print("first_prompt_prefix", rows[0]["prompt"][:120].replace("\n", " "))
print("first_response_prefix", rows[0]["response"][:120].replace("\n", " "))
PY
```

Expected:

```text
count <N>
first_keys ['prompt', 'response', 'behavior', 'behavior_id', 'functional_category', 'semantic_category']
first_prompt_prefix <non-empty text>
first_response_prefix <non-empty text>
```

### Task 4: Document The New Input In Existing Workflow Notes

**Files:**
- Modify: `REPRODUCTION_SUMMARY.md`

- [ ] **Step 1: Add a short note describing the new export command and resulting harmful JSON**

```md
## HarmBench 数据导出

可使用以下命令将 HarmBench `parquet` 测试集转成当前 CSD 流程可直接消费的 harmful `json`：

```bash
python scripts/export_harmbench_testcase_harmful.py \
  --input_parquet data/HarmfulGeneration-HarmBench/data/test-00000-of-00001.parquet \
  --output_json data/harmbench_testcase_harmful.json \
  --output_csv data/harmbench_testcase_harmful.csv
```

其中：

- `prompt = test_case`
- `response = answer`
- safe 侧仍由 `utils/refusals.txt` 在 `make_csd_llada.py` / `make_csd_dream.py` 中动态采样
```

- [ ] **Step 2: Run a narrow markdown grep to confirm the new section exists**

Run:

```bash
rg -n "HarmBench 数据导出|harmbench_testcase_harmful" "REPRODUCTION_SUMMARY.md"
```

Expected:

```text
<line>:## HarmBench 数据导出
<line>:python scripts/export_harmbench_testcase_harmful.py \
```

## Self-Review

- Spec coverage:
  - 已覆盖数据映射、JSON/CSV 双输出、轻量清洗、按 `prompt` 去重、继续复用 refusal 池
  - 未把 refusal 扩展、显式 pair 文件、直接读 `parquet` 的 CSD 改造纳入实现，符合 spec 的 out-of-scope
- Placeholder scan:
  - 无 `TODO` / `TBD` / “later” 类占位
- Type consistency:
  - 全程使用 `prompt` / `response` / `behavior` / `behavior_id` / `functional_category` / `semantic_category`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-harmbench-csd-data-export.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
