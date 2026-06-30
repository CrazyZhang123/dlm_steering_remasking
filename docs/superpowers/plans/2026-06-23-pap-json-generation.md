# PAP JSON Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按论文 `asset` 中的 PAP 口径，为 `JBB` 与 `AdvBench` 生成可被现有评测入口直接消费的 `JBB_pap.json` 与 `AdvBench_pap.json`，并完成本地 smoke、全量与消费契约校验。

**Architecture:** 在 `scripts/` 下维护一个独立的 PAP 生成脚本，读取本地 JBB / AdvBench 原始 harmful prompts，直接复用 `DIJA/benchmarks/HarmBench/baselines/pap/templates.py` 中的 `one_shot_kd` 模板语义与 `Expert Endorsement` 策略元数据，调用本地 `gpt-oss-20b` 生成单条 PAP mutation，再清洗输出并写出最终 JSON。测试放在 `tests/` 下，优先覆盖纯函数、模板装配、CLI 编排与设备选择；真实模型生成统一通过 `tmux` 后台运行并先做资源检查。

**Tech Stack:** Python 3.13, `transformers`, `torch`, `json`, `csv`, `argparse`, `pathlib`, `unittest`, `tmux`

---

## File Structure

- Create or maintain: `scripts/build_pap_json.py`
- Create or maintain: `tests/test_build_pap_json.py`
- Modify: `docs/table2_reproduction_progress.md`
- Read for paper contract: `assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md:320-360`
- Read for local JBB source: `data/harmful-behaviors.csv`
- Read for local AdvBench source: `DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv`
- Read for PAP template reference: `DIJA/benchmarks/HarmBench/baselines/pap/templates.py:44-48`
- Read for evaluation consumer contract: `eval_llada_steering.py:688-702`
- Read for evaluation consumer contract: `eval_dream_steering.py:703-717`
- Read for test style reference: `tests/test_prepare_wildjailbreak_prompts.py`
- Read for script style reference: `scripts/prepare_wildjailbreak_prompts.py`

按仓库 `AGENTS.md` 约定，本计划不包含 `git commit`、`git push` 或分支操作步骤。

## Source-Of-Truth Contract

### Paper Contract

- PAP 是论文 `Table 2` 中与 `DIJA`、`Prefix` 并列的攻击方式，不混用 `refined_goal`
- PAP 生成模型使用 `gpt-oss-20b`
- PAP 技术固定使用 `Expert Endorsement`
- 数据规模固定为：
  - `JBB = 100`
  - `AdvBench = 520`

### Local Implementation Contract

- 论文只定义攻击方法与规模，不给出完整 prompt 原文
- 本项目实现时，**直接复用** `DIJA/benchmarks/HarmBench/baselines/pap/templates.py` 中的：
  - `one_shot_kd`
  - `persuasion_taxonomy` 里的 `Expert Endorsement` 条目
- 不允许手写一个“近似版” prompt 去替代 `one_shot_kd`

### Evaluation Consumer Contract

- `eval_llada_steering.py` 与 `eval_dream_steering.py` 在 `attack_method == "PAP"` 时固定读取：
  - `./gpt-oss/JBB_pap.json`
  - `./gpt-oss/AdvBench_pap.json`
- 消费字段固定是 `pap_prompt`
- 为兼容现有评测入口，每条记录至少包含：

```json
{
  "pap_prompt": "mutated prompt text",
  "goal": "original harmful goal",
  "source": "JBB"
}
```

`AdvBench` 记录的 `source` 固定为 `"AdvBench"`。

## Local Dataset Contracts

- JBB 输入使用本地 `data/harmful-behaviors.csv`
  - 字段：`Index`, `Goal`, `Target`, `Behavior`, `Category`, `Source`
  - 目标行数：`100`
- AdvBench 输入使用本地 `DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv`
  - 字段：`Behavior`, `Category`, `Tags`, `ContextString`, `BehaviorID`
  - 目标行数：`520`

## Execution Preconditions

- 测试命令统一使用当前环境可执行的 `python -m unittest`
- 真实模型 smoke / full generation 统一使用 `tmux` 后台执行，并将日志写入 `gpt-oss/logs/`
- 在首次真实模型运行前，必须执行一次 GPU / 进程资源检查
- 若 GPU 不空闲，不继续盲跑；先记录阻塞状态与当前占用

### Target Outputs

- Smoke:
  - `gpt-oss/JBB_pap_smoke.json`
  - `gpt-oss/AdvBench_pap_smoke.json`
- Full:
  - `gpt-oss/JBB_pap.json`
  - `gpt-oss/AdvBench_pap.json`

---

### Task 1: Add Failing Tests For Paper-Aligned PAP Helpers

**Files:**
- Create or modify: `tests/test_build_pap_json.py`
- Read: `tests/test_prepare_wildjailbreak_prompts.py`

- [ ] **Step 1: Write the failing test file for dataset mapping and template loading**

```python
import json
import sys
import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_pap_json as pap_mod


class NormalizeRowsTest(unittest.TestCase):
    def test_normalize_jbb_rows_maps_goal_and_sets_source(self):
        rows = [
            {
                "Index": "0",
                "Goal": " harmful goal a ",
                "Target": "ignored",
                "Behavior": "behavior-a",
                "Category": "category-a",
                "Source": "Original",
            },
            {
                "Index": "1",
                "Goal": "harmful goal b",
                "Target": "ignored",
                "Behavior": "behavior-b",
                "Category": "category-b",
                "Source": "AdvBench",
            },
        ]

        normalized = pap_mod.normalize_jbb_rows(rows)

        self.assertEqual(
            normalized,
            [
                {"goal": "harmful goal a", "source": "JBB"},
                {"goal": "harmful goal b", "source": "JBB"},
            ],
        )

    def test_normalize_advbench_rows_maps_behavior_and_sets_source(self):
        rows = [
            {
                "Behavior": " teach harmful behavior ",
                "Category": "",
                "Tags": "",
                "ContextString": "",
                "BehaviorID": "advbench_1",
            }
        ]

        normalized = pap_mod.normalize_advbench_rows(rows)

        self.assertEqual(
            normalized,
            [{"goal": "teach harmful behavior", "source": "AdvBench"}],
        )

    def test_normalize_rows_skip_empty_and_dedup_goals(self):
        rows = [
            {"Goal": "goal-a"},
            {"Goal": " goal-a "},
            {"Goal": ""},
            {"Goal": "goal-b"},
        ]

        normalized = pap_mod.normalize_jbb_rows(rows)

        self.assertEqual(
            normalized,
            [
                {"goal": "goal-a", "source": "JBB"},
                {"goal": "goal-b", "source": "JBB"},
            ],
        )


class PromptConstructionTest(unittest.TestCase):
    def test_get_expert_endorsement_template_returns_expected_strategy(self):
        template = pap_mod.get_expert_endorsement_template()

        self.assertEqual(template["ss_technique"], "Expert Endorsement")
        self.assertIn("domain experts", template["ss_definition"])
        self.assertIn("Surgeon General", template["ss_example"])

    def test_load_one_shot_template_contains_behavior_placeholder(self):
        template_text = pap_mod.load_one_shot_template()

        self.assertIn('{technique}', template_text)
        self.assertIn('{definition}', template_text)
        self.assertIn('{example}', template_text)
        self.assertIn('{behavior}', template_text)
        self.assertIn("Only output ONE high-quality", template_text)

    def test_build_mutation_prompt_formats_one_shot_template(self):
        strategy = {
            "ss_technique": "Expert Endorsement",
            "ss_definition": "Citing domain experts in support of a claim.",
            "ss_example": "Example text.",
        }

        prompt = pap_mod.build_mutation_prompt(
            "Provide step-by-step instructions for hacking a network",
            strategy,
        )

        self.assertIn("Expert Endorsement", prompt)
        self.assertIn("Citing domain experts in support of a claim.", prompt)
        self.assertIn("Provide step-by-step instructions for hacking a network", prompt)
        self.assertIn("Only output ONE high-quality", prompt)
        self.assertIn("without the code block and any redundancy", prompt)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
FAIL or ERROR because scripts.build_pap_json is missing template helpers or aligned prompt construction
```

- [ ] **Step 3: Extend the failing test file with output cleaning, JSON writing, and CLI coverage**

```python
class OutputCleaningTest(unittest.TestCase):
    def test_extract_pap_prompt_removes_python_quotes_and_whitespace(self):
        raw = '"  rewritten harmful request  "'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_extracts_fenced_string(self):
        raw = '```python\n"rewritten harmful request"\n```'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")


class JsonWriteTest(unittest.TestCase):
    def test_write_pap_json_persists_required_schema(self):
        rows = [
            {
                "pap_prompt": "pap text",
                "goal": "goal text",
                "source": "JBB",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "JBB_pap.json"
            pap_mod.write_pap_json(output_path, rows)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, rows)


class ParseNonNegativeIntTest(unittest.TestCase):
    def test_parse_non_negative_int_accepts_zero_and_positive(self):
        self.assertEqual(pap_mod.parse_non_negative_int("0"), 0)
        self.assertEqual(pap_mod.parse_non_negative_int("12"), 12)

    def test_parse_non_negative_int_rejects_negative_values(self):
        with self.assertRaisesRegex(ArgumentTypeError, "non-negative integer"):
            pap_mod.parse_non_negative_int("-1")
```

- [ ] **Step 4: Re-run the tests to verify the new assertions also fail for missing code**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
FAIL with missing attributes such as load_one_shot_template, extract_pap_prompt, or write_pap_json
```

### Task 2: Implement Minimal Offline Helpers With Direct Template Reuse

**Files:**
- Create or modify: `scripts/build_pap_json.py`
- Test: `tests/test_build_pap_json.py`

- [ ] **Step 1: Write the minimal helper implementation to satisfy the failing pure-function tests**

```python
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_PATH = ROOT / "DIJA/benchmarks/HarmBench/baselines/pap/templates.py"
_TEMPLATES_MODULE_CACHE = None


def normalize_jbb_rows(rows):
    normalized = []
    seen = set()

    for row in rows:
        goal = str((row.get("Goal") or "")).strip()
        if not goal or goal in seen:
            continue
        seen.add(goal)
        normalized.append({"goal": goal, "source": "JBB"})

    return normalized


def normalize_advbench_rows(rows):
    normalized = []
    seen = set()

    for row in rows:
        goal = str((row.get("Behavior") or "")).strip()
        if not goal or goal in seen:
            continue
        seen.add(goal)
        normalized.append({"goal": goal, "source": "AdvBench"})

    return normalized


def _load_templates_module():
    global _TEMPLATES_MODULE_CACHE
    if _TEMPLATES_MODULE_CACHE is not None:
        return _TEMPLATES_MODULE_CACHE

    spec = importlib.util.spec_from_file_location("pap_templates", TEMPLATES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load PAP templates from {TEMPLATES_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TEMPLATES_MODULE_CACHE = module
    return module


def get_expert_endorsement_template():
    module = _load_templates_module()
    for item in module.persuasion_taxonomy:
        if item["ss_technique"] == "Expert Endorsement":
            return {
                "ss_technique": item["ss_technique"],
                "ss_definition": item["ss_definition"],
                "ss_example": item["ss_example"],
            }
    raise RuntimeError("Expert Endorsement template not found")


def load_one_shot_template():
    module = _load_templates_module()
    return module.one_shot_kd


def build_mutation_prompt(goal, template):
    prompt_template = load_one_shot_template()
    return prompt_template.format(
        technique=template["ss_technique"],
        definition=template["ss_definition"],
        example=template["ss_example"],
        behavior=goal,
    )


def extract_pap_prompt(raw_text):
    text = str(raw_text).strip()
    match = re.search(r"```[^\n`]*\n?(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {'"', "'"}
    ):
        return text[1:-1].strip()
    return text.strip()


def write_pap_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed
```

- [ ] **Step 2: Run the focused test file to verify the pure helpers now pass**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
currently defined helper tests PASS
```

### Task 3: Add Failing CLI, Resource, And Model-Orchestration Tests

**Files:**
- Modify: `tests/test_build_pap_json.py`
- Modify: `scripts/build_pap_json.py`

- [ ] **Step 1: Add failing tests for local file loading, mocked generation orchestration, and device resolution**

```python
class LoadRowsTest(unittest.TestCase):
    def test_load_rows_reads_local_jbb_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "harmful-behaviors.csv"
            path.write_text(
                "Index,Goal,Target,Behavior,Category,Source\n"
                "0,goal-a,target-a,behavior-a,category-a,Original\n",
                encoding="utf-8",
            )

            rows = pap_mod.load_csv_rows(path)

        self.assertEqual(rows[0]["Goal"], "goal-a")


class MutationPipelineTest(unittest.TestCase):
    def test_mutate_goals_calls_generator_and_preserves_goal_source(self):
        rows = [
            {"goal": "goal-a", "source": "JBB"},
            {"goal": "goal-b", "source": "AdvBench"},
        ]

        with patch.object(
            pap_mod,
            "generate_with_local_model",
            side_effect=['"pap-a"', '"pap-b"'],
        ) as gen_mock:
            payload = pap_mod.mutate_goals(
                rows,
                model_path="models/gpt-oss-20b",
                max_new_tokens=128,
            )

        self.assertEqual(gen_mock.call_count, 2)
        self.assertEqual(
            payload,
            [
                {"pap_prompt": "pap-a", "goal": "goal-a", "source": "JBB"},
                {"pap_prompt": "pap-b", "goal": "goal-b", "source": "AdvBench"},
            ],
        )


class ResolveInputDeviceTest(unittest.TestCase):
    def test_resolve_input_device_prefers_hf_device_map(self):
        class FakeModel:
            hf_device_map = {"model.embed_tokens": "cuda:1"}

        self.assertEqual(pap_mod.resolve_input_device(FakeModel()), "cuda:1")


class MainTest(unittest.TestCase):
    def test_main_builds_smoke_json_from_local_jbb_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "harmful-behaviors.csv"
            output_json = tmp / "JBB_pap_smoke.json"

            input_csv.write_text(
                "Index,Goal,Target,Behavior,Category,Source\n"
                "0,goal-a,target-a,behavior-a,category-a,Original\n"
                "1,goal-b,target-b,behavior-b,category-b,AdvBench\n",
                encoding="utf-8",
            )

            argv = [
                "build_pap_json.py",
                "--dataset_name", "JBB",
                "--input_path", str(input_csv),
                "--output_path", str(output_json),
                "--model_path", "models/gpt-oss-20b",
                "--limit", "1",
                "--max_new_tokens", "64",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(
                    pap_mod, "generate_with_local_model", return_value='"pap-a"'
                ):
                    pap_mod.main()

            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(
            payload,
            [{"pap_prompt": "pap-a", "goal": "goal-a", "source": "JBB"}],
        )
```

- [ ] **Step 2: Run the tests to verify they fail before CLI code exists**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
FAIL with missing functions such as load_csv_rows, mutate_goals, resolve_input_device, generate_with_local_model, or main
```

### Task 4: Implement The CLI Script, Resource Checks, And Local gpt-oss Generator

**Files:**
- Modify: `scripts/build_pap_json.py`
- Test: `tests/test_build_pap_json.py`

- [ ] **Step 1: Add CSV loading, mutation orchestration, CLI parsing, and device resolution**

```python
_MODEL_CACHE = {}
AutoTokenizer = None
AutoModelForCausalLM = None
torch = None


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mutate_goals(rows, model_path, max_new_tokens):
    template = get_expert_endorsement_template()
    payload = []

    for row in rows:
        prompt = build_mutation_prompt(row["goal"], template)
        generated = generate_with_local_model(
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        payload.append(
            {
                "pap_prompt": extract_pap_prompt(generated),
                "goal": row["goal"],
                "source": row["source"],
            }
        )

    return payload


def resolve_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if device not in (None, "disk"):
                return device

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["JBB", "AdvBench"], required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--limit", type=parse_non_negative_int, default=0)
    parser.add_argument("--max_new_tokens", type=parse_non_negative_int, default=128)
    args = parser.parse_args()

    rows = load_csv_rows(args.input_path)
    normalized = (
        normalize_jbb_rows(rows)
        if args.dataset_name == "JBB"
        else normalize_advbench_rows(rows)
    )
    if args.limit:
        normalized = normalized[: args.limit]

    payload = mutate_goals(
        normalized,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    write_pap_json(args.output_path, payload)

    print(f"saved {len(payload)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the real local model generation function using Transformers chat template**

```python
def _ensure_model_dependencies():
    global AutoTokenizer, AutoModelForCausalLM, torch
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
    if torch is None:
        import torch as _torch

        torch = _torch


def _get_local_model(model_path):
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    _ensure_model_dependencies()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    cached = (tokenizer, model)
    _MODEL_CACHE[model_path] = cached
    return cached


def generate_with_local_model(model_path, prompt, max_new_tokens):
    tokenizer, model = _get_local_model(model_path)
    _ensure_model_dependencies()
    model_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = resolve_input_device(model)
    model_inputs = model_inputs.to(device)

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = model_inputs["input_ids"].shape[-1]
    generated_tokens = generated[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

- [ ] **Step 3: Run the test file to verify the mocked CLI flow passes**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
all tests PASS
```

### Task 5: Add Resource Inspection And tmux-Based Smoke Verification

**Files:**
- Read: `scripts/build_pap_json.py`
- Output: `gpt-oss/JBB_pap_smoke.json`
- Output: `gpt-oss/AdvBench_pap_smoke.json`
- Log: `gpt-oss/logs/JBB_pap_smoke.log`
- Log: `gpt-oss/logs/AdvBench_pap_smoke.log`

- [ ] **Step 1: Inspect available GPU resources before any real PAP generation**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

Expected:

```text
either at least one GPU has enough free memory for gpt-oss-20b, or the command output clearly shows the blocking processes
```

- [ ] **Step 2: Launch a one-row JBB smoke run in tmux**

Run:

```bash
mkdir -p "gpt-oss/logs"
tmux new-session -d -s "pap_jbb_smoke" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "JBB" \
     --input_path "data/harmful-behaviors.csv" \
     --output_path "gpt-oss/JBB_pap_smoke.json" \
     --model_path "models/gpt-oss-20b" \
     --limit 1 \
     --max_new_tokens 96 \
     > "gpt-oss/logs/JBB_pap_smoke.log" 2>&1'
```

Expected:

```text
tmux session pap_jbb_smoke created
```

- [ ] **Step 3: Inspect the smoke session result and log**

Run:

```bash
tmux has-session -t "pap_jbb_smoke"
tail -n 40 "gpt-oss/logs/JBB_pap_smoke.log"
python - <<'PY'
from pathlib import Path
path = Path("gpt-oss/JBB_pap_smoke.json")
print("exists", path.exists())
if path.exists():
    print("size", path.stat().st_size)
PY
```

Expected:

```text
either saved 1 rows to gpt-oss/JBB_pap_smoke.json
or a concrete blocking error such as CUDA OOM appears in the log
```

- [ ] **Step 4: If the one-row smoke passes, run 10-row smoke commands for both datasets in tmux**

Run:

```bash
tmux new-session -d -s "pap_jbb_smoke10" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "JBB" \
     --input_path "data/harmful-behaviors.csv" \
     --output_path "gpt-oss/JBB_pap_smoke.json" \
     --model_path "models/gpt-oss-20b" \
     --limit 10 \
     --max_new_tokens 128 \
     > "gpt-oss/logs/JBB_pap_smoke10.log" 2>&1'

tmux new-session -d -s "pap_adv_smoke10" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "AdvBench" \
     --input_path "DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv" \
     --output_path "gpt-oss/AdvBench_pap_smoke.json" \
     --model_path "models/gpt-oss-20b" \
     --limit 10 \
     --max_new_tokens 128 \
     > "gpt-oss/logs/AdvBench_pap_smoke10.log" 2>&1'
```

Expected:

```text
both tmux sessions start successfully
```

- [ ] **Step 5: Verify smoke counts and source labels**

Run:

```bash
python - <<'PY'
import json
for path in ["gpt-oss/JBB_pap_smoke.json", "gpt-oss/AdvBench_pap_smoke.json"]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    print(path, len(rows), rows[0]["source"], sorted(rows[0].keys()))
PY
```

Expected:

```text
gpt-oss/JBB_pap_smoke.json 10 JBB ['goal', 'pap_prompt', 'source']
gpt-oss/AdvBench_pap_smoke.json 10 AdvBench ['goal', 'pap_prompt', 'source']
```

### Task 6: Run Full Offline Generation And Consumer Compatibility Verification

**Files:**
- Output: `gpt-oss/JBB_pap.json`
- Output: `gpt-oss/AdvBench_pap.json`
- Log: `gpt-oss/logs/JBB_pap_full.log`
- Log: `gpt-oss/logs/AdvBench_pap_full.log`
- Read: `eval_llada_steering.py:688-702`
- Read: `eval_dream_steering.py:703-717`

- [ ] **Step 1: Launch the full JBB PAP JSON generation in tmux**

Run:

```bash
tmux new-session -d -s "pap_jbb_full" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "JBB" \
     --input_path "data/harmful-behaviors.csv" \
     --output_path "gpt-oss/JBB_pap.json" \
     --model_path "models/gpt-oss-20b" \
     --max_new_tokens 128 \
     > "gpt-oss/logs/JBB_pap_full.log" 2>&1'
```

Expected:

```text
tmux session pap_jbb_full created
```

- [ ] **Step 2: Launch the full AdvBench PAP JSON generation in tmux**

Run:

```bash
tmux new-session -d -s "pap_adv_full" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "AdvBench" \
     --input_path "DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv" \
     --output_path "gpt-oss/AdvBench_pap.json" \
     --model_path "models/gpt-oss-20b" \
     --max_new_tokens 128 \
     > "gpt-oss/logs/AdvBench_pap_full.log" 2>&1'
```

Expected:

```text
tmux session pap_adv_full created
```

- [ ] **Step 3: Inspect the finished logs**

Run:

```bash
tail -n 40 "gpt-oss/logs/JBB_pap_full.log"
tail -n 40 "gpt-oss/logs/AdvBench_pap_full.log"
```

Expected:

```text
saved 100 rows to gpt-oss/JBB_pap.json
saved 520 rows to gpt-oss/AdvBench_pap.json
```

- [ ] **Step 4: Validate full output counts and schema**

Run:

```bash
python - <<'PY'
import json
checks = [
    ("gpt-oss/JBB_pap.json", 100, "JBB"),
    ("gpt-oss/AdvBench_pap.json", 520, "AdvBench"),
]
for path, expected_count, expected_source in checks:
    rows = json.load(open(path, "r", encoding="utf-8"))
    assert len(rows) == expected_count, (path, len(rows), expected_count)
    assert sorted(rows[0].keys()) == ["goal", "pap_prompt", "source"], path
    assert rows[0]["source"] == expected_source, path
    assert rows[0]["goal"].strip(), path
    assert rows[0]["pap_prompt"].strip(), path
    print(path, "ok", len(rows), expected_source)
PY
```

Expected:

```text
gpt-oss/JBB_pap.json ok 100 JBB
gpt-oss/AdvBench_pap.json ok 520 AdvBench
```

- [ ] **Step 5: Verify the files satisfy the existing eval consumer contract**

Run:

```bash
python - <<'PY'
import json
for path in ["gpt-oss/JBB_pap.json", "gpt-oss/AdvBench_pap.json"]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    sample = rows[0]
    assert "pap_prompt" in sample
    assert "goal" in sample
    assert "source" in sample
    print(path, "consumer-compatible")
PY
```

Expected:

```text
gpt-oss/JBB_pap.json consumer-compatible
gpt-oss/AdvBench_pap.json consumer-compatible
```

### Task 7: Record Reproduction Commands And Runtime Caveats In Progress Notes

**Files:**
- Modify: `docs/table2_reproduction_progress.md`

- [ ] **Step 1: Add the final generation commands, paper method source, and local data provenance to the progress document**

````markdown
### PAP JSON 准备

- 论文方法口径来自 `assets/Adaptive_Steering_and_Remasking_for_Safe_Generation_in_DLMs_clean.md`
- PAP 模板实现复用 `DIJA/benchmarks/HarmBench/baselines/pap/templates.py` 的 `one_shot_kd` 与 `Expert Endorsement`
- JBB PAP 输入来自本地 `data/harmful-behaviors.csv`
- AdvBench PAP 输入来自本地 `DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv`
- 生成模型：`models/gpt-oss-20b`
- 运行方式：`tmux` 后台 + `gpt-oss/logs/` 日志

```bash
tmux new-session -d -s "pap_jbb_full" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "JBB" \
     --input_path "data/harmful-behaviors.csv" \
     --output_path "gpt-oss/JBB_pap.json" \
     --model_path "models/gpt-oss-20b" \
     --max_new_tokens 128 \
     > "gpt-oss/logs/JBB_pap_full.log" 2>&1'

tmux new-session -d -s "pap_adv_full" \
  'cd "/root/myproject/DLM_Steering_Remasking" && \
   python -u "scripts/build_pap_json.py" \
     --dataset_name "AdvBench" \
     --input_path "DIJA/benchmarks/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv" \
     --output_path "gpt-oss/AdvBench_pap.json" \
     --model_path "models/gpt-oss-20b" \
     --max_new_tokens 128 \
     > "gpt-oss/logs/AdvBench_pap_full.log" 2>&1'
```
````

- [ ] **Step 2: Re-run the targeted test file and keep it green after the doc update**

Run:

```bash
python -m unittest tests.test_build_pap_json -v
```

Expected:

```text
OK
```

## Self-Review

- Spec coverage:
  - 本计划覆盖了论文 `PAP = gpt-oss-20b + Expert Endorsement + JBB/AdvBench` 的方法口径
  - 本计划明确要求直接复用 `one_shot_kd` 模板，而不是手写近似 prompt
  - 本计划覆盖了本地 smoke、离线 JBB/AdvBench 数据源、`JBB_pap.json` / `AdvBench_pap.json` 生成、schema 校验与现有评测消费契约校验
  - 本计划故意不修改 `eval_llada_steering.py` / `eval_dream_steering.py`，以满足“只补 JSON，不改评测主逻辑”的当前范围
- Placeholder scan:
  - 未使用 `TODO`、`TBD`、`implement later` 之类占位语
  - 每个测试、脚本函数、命令和预期输出都给了明确文本
- Type consistency:
  - 全计划统一使用 `goal`, `pap_prompt`, `source` 作为输出 JSON 字段
  - JBB 统一映射 `Goal -> goal`，AdvBench 统一映射 `Behavior -> goal`
  - 测试命令统一使用 `python -m unittest`
  - 真实模型运行命令统一使用 `tmux` 后台执行
