import json
import sys
import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace
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
        self.assertEqual(
            set(template),
            {"ss_technique", "ss_definition", "ss_example"},
        )
        self.assertTrue(all(isinstance(template[key], str) and template[key] for key in template))

    def test_load_one_shot_template_reuses_pap_template_text(self):
        template_text = pap_mod.load_one_shot_template()

        self.assertIn('{technique}', template_text)
        self.assertIn('{definition}', template_text)
        self.assertIn('{example}', template_text)
        self.assertIn('{behavior}', template_text)
        self.assertIn("without the code block and any redundancy", template_text)

    def test_build_mutation_prompt_inserts_goal_and_strategy_fields(self):
        template = {
            "ss_technique": "Expert Endorsement",
            "ss_definition": "Citing domain experts in support of a claim.",
            "ss_example": "Example text.",
        }

        prompt = pap_mod.build_mutation_prompt(
            "Provide step-by-step instructions for hacking a network",
            template,
        )
        template_text = pap_mod.load_one_shot_template()
        expected = template_text.format(
            technique=template["ss_technique"],
            definition=template["ss_definition"],
            example=template["ss_example"],
            behavior="Provide step-by-step instructions for hacking a network",
        )

        self.assertEqual(prompt, expected)

    def test_get_expert_endorsement_template_raises_when_template_file_missing(self):
        missing_path = ROOT / "DIJA/benchmarks/HarmBench/baselines/pap/missing_templates.py"

        with patch.object(pap_mod, "TEMPLATES_PATH", missing_path):
            pap_mod._TEMPLATES_MODULE_CACHE = None
            with self.assertRaisesRegex(FileNotFoundError, "PAP template file not found"):
                pap_mod.get_expert_endorsement_template()

    def test_get_expert_endorsement_template_raises_when_taxonomy_missing(self):
        with patch.object(pap_mod, "_load_templates_module", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "missing persuasion_taxonomy"):
                pap_mod.get_expert_endorsement_template()

    def test_get_expert_endorsement_template_raises_when_strategy_missing(self):
        fake_module = SimpleNamespace(
            persuasion_taxonomy=[
                {
                    "ss_technique": "Other Strategy",
                    "ss_definition": "unused",
                    "ss_example": "unused",
                }
            ]
        )

        with patch.object(pap_mod, "_load_templates_module", return_value=fake_module):
            with self.assertRaisesRegex(RuntimeError, "Expert Endorsement template not found"):
                pap_mod.get_expert_endorsement_template()

    def test_get_expert_endorsement_template_raises_when_required_fields_missing(self):
        fake_module = SimpleNamespace(
            persuasion_taxonomy=[
                {
                    "ss_technique": "Expert Endorsement",
                    "ss_definition": "definition only",
                }
            ]
        )

        with patch.object(pap_mod, "_load_templates_module", return_value=fake_module):
            with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                pap_mod.get_expert_endorsement_template()


class OutputCleaningTest(unittest.TestCase):
    def test_extract_pap_prompt_removes_python_quotes_and_whitespace(self):
        raw = '"  rewritten harmful request  "'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_uses_full_text_when_no_quotes_exist(self):
        raw = "rewritten harmful request"

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_removes_triple_quotes(self):
        raw = '"""  rewritten harmful request  """'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_extracts_string_from_fenced_code_block(self):
        raw = '```python\n"rewritten harmful request"\n```'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_extracts_quoted_string_from_prefixed_text(self):
        raw = 'Here is the rewritten prompt: "rewritten harmful request"'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "rewritten harmful request")

    def test_extract_pap_prompt_preserves_multiline_content(self):
        raw = '"""\nline one\nline two\n"""'

        cleaned = pap_mod.extract_pap_prompt(raw)

        self.assertEqual(cleaned, "line one\nline two")

    def test_extract_pap_prompt_discards_code_fence_wrapper_text(self):
        raw = 'Sure, use this:\n```text\nrewritten harmful request\n```'

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


class LoadCsvRowsTest(unittest.TestCase):
    def test_load_csv_rows_reads_csv_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "harmful-behaviors.csv"
            path.write_text(
                "Index,Goal,Target,Behavior,Category,Source\n"
                "0,goal-a,target-a,behavior-a,category-a,Original\n",
                encoding="utf-8",
            )

            rows = pap_mod.load_csv_rows(path)

        self.assertEqual(
            rows,
            [
                {
                    "Index": "0",
                    "Goal": "goal-a",
                    "Target": "target-a",
                    "Behavior": "behavior-a",
                    "Category": "category-a",
                    "Source": "Original",
                }
            ],
        )


class ParseNonNegativeIntTest(unittest.TestCase):
    def test_parse_non_negative_int_accepts_zero_and_positive(self):
        self.assertEqual(pap_mod.parse_non_negative_int("0"), 0)
        self.assertEqual(pap_mod.parse_non_negative_int("12"), 12)

    def test_parse_non_negative_int_rejects_negative_values(self):
        with self.assertRaisesRegex(ArgumentTypeError, "non-negative integer"):
            pap_mod.parse_non_negative_int("-1")


class MutationPipelineTest(unittest.TestCase):
    def test_mutate_goals_calls_generator_and_preserves_goal_source(self):
        rows = [
            {"goal": "goal-a", "source": "JBB"},
            {"goal": "goal-b", "source": "AdvBench"},
        ]
        template = {
            "ss_technique": "Expert Endorsement",
            "ss_definition": "definition",
            "ss_example": "example",
        }

        with patch.object(pap_mod, "get_expert_endorsement_template", return_value=template):
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
        first_call = gen_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["model_path"], "models/gpt-oss-20b")
        self.assertEqual(first_call.kwargs["max_new_tokens"], 128)
        self.assertIn("goal-a", first_call.kwargs["prompt"])
        self.assertEqual(
            payload,
            [
                {"pap_prompt": "pap-a", "goal": "goal-a", "source": "JBB"},
                {"pap_prompt": "pap-b", "goal": "goal-b", "source": "AdvBench"},
            ],
        )


class GetLocalModelTest(unittest.TestCase):
    def test_get_local_model_loads_transformers_model_with_expected_options_and_eval(self):
        tokenizer = object()
        model = unittest.mock.Mock()
        auto_tokenizer = SimpleNamespace(
            from_pretrained=unittest.mock.Mock(return_value=tokenizer)
        )
        auto_model = SimpleNamespace(
            from_pretrained=unittest.mock.Mock(return_value=model)
        )

        with patch.object(pap_mod, "_MODEL_CACHE", {}):
            with patch.object(pap_mod, "AutoTokenizer", auto_tokenizer, create=True):
                with patch.object(
                    pap_mod, "AutoModelForCausalLM", auto_model, create=True
                ):
                    loaded_tokenizer, loaded_model = pap_mod._get_local_model(
                        "models/gpt-oss-20b"
                    )

        auto_tokenizer.from_pretrained.assert_called_once_with("models/gpt-oss-20b")
        auto_model.from_pretrained.assert_called_once_with(
            "models/gpt-oss-20b",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.eval.assert_called_once_with()
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_model, model)

    def test_get_local_model_reuses_cache_for_same_path(self):
        tokenizer = object()
        model = unittest.mock.Mock()
        auto_tokenizer = SimpleNamespace(
            from_pretrained=unittest.mock.Mock(return_value=tokenizer)
        )
        auto_model = SimpleNamespace(
            from_pretrained=unittest.mock.Mock(return_value=model)
        )

        with patch.object(pap_mod, "_MODEL_CACHE", {}):
            with patch.object(pap_mod, "AutoTokenizer", auto_tokenizer, create=True):
                with patch.object(
                    pap_mod, "AutoModelForCausalLM", auto_model, create=True
                ):
                    first = pap_mod._get_local_model("models/gpt-oss-20b")
                    second = pap_mod._get_local_model("models/gpt-oss-20b")

        auto_tokenizer.from_pretrained.assert_called_once_with("models/gpt-oss-20b")
        auto_model.from_pretrained.assert_called_once()
        self.assertIs(first, second)


class ResolveInputDeviceTest(unittest.TestCase):
    def test_resolve_input_device_prefers_hf_device_map(self):
        model = SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "lm_head": "cpu"})

        device = pap_mod._resolve_input_device(model)

        self.assertEqual(device, 0)

    def test_resolve_input_device_falls_back_to_cpu_when_no_device_info_exists(self):
        class ParameterlessModel:
            def parameters(self):
                raise RuntimeError("no parameters")

        device = pap_mod._resolve_input_device(ParameterlessModel())

        self.assertEqual(device, "cpu")


class GenerateWithLocalModelTest(unittest.TestCase):
    def test_generate_with_local_model_uses_full_chat_orchestration(self):
        class FakeInputs(dict):
            def __init__(self):
                super().__init__(
                    input_ids=[[11, 12, 13]],
                    attention_mask=[[1, 1, 1]],
                )
                self.to = unittest.mock.Mock(return_value=self)

        class NoGradContext:
            def __init__(self):
                self.enter = unittest.mock.Mock()
                self.exit = unittest.mock.Mock(return_value=False)

            def __enter__(self):
                self.enter()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self.exit(exc_type, exc, tb)

        fake_inputs = FakeInputs()
        no_grad_context = NoGradContext()
        tokenizer = SimpleNamespace()
        tokenizer.apply_chat_template = unittest.mock.Mock(
            return_value=fake_inputs
        )
        tokenizer.decode = unittest.mock.Mock(return_value='  "mutated goal"  ')
        model = SimpleNamespace()
        model.parameters = unittest.mock.Mock(return_value=iter([SimpleNamespace(device="cuda:1")]))
        model.generate = unittest.mock.Mock(return_value=[[11, 12, 13, 21, 22]])
        fake_torch = SimpleNamespace(no_grad=unittest.mock.Mock(return_value=no_grad_context))

        with patch.object(pap_mod, "_get_local_model", return_value=(tokenizer, model)):
            with patch.object(pap_mod, "torch", fake_torch, create=True):
                generated = pap_mod.generate_with_local_model(
                    model_path="models/gpt-oss-20b",
                    prompt="rewrite this goal",
                    max_new_tokens=64,
                )

        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "rewrite this goal"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model.parameters.assert_called_once_with()
        fake_inputs.to.assert_called_once_with("cuda:1")
        fake_torch.no_grad.assert_called_once_with()
        no_grad_context.enter.assert_called_once_with()
        no_grad_context.exit.assert_called_once()
        model.generate.assert_called_once_with(
            input_ids=[[11, 12, 13]],
            attention_mask=[[1, 1, 1]],
            max_new_tokens=64,
            do_sample=False,
        )
        tokenizer.decode.assert_called_once_with([21, 22], skip_special_tokens=True)
        self.assertEqual(generated, '"mutated goal"')

    def test_generate_with_local_model_uses_hf_device_map_without_touching_parameters(self):
        class FakeInputs(dict):
            def __init__(self):
                super().__init__(
                    input_ids=[[11, 12, 13]],
                    attention_mask=[[1, 1, 1]],
                )
                self.to = unittest.mock.Mock(return_value=self)

        class NoGradContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        tokenizer = SimpleNamespace()
        tokenizer.apply_chat_template = unittest.mock.Mock(return_value=FakeInputs())
        tokenizer.decode = unittest.mock.Mock(return_value="done")

        class ParametersShouldNotBeUsed:
            hf_device_map = {"transformer.wte": "cuda:2"}

            def parameters(self):
                raise AssertionError("parameters() should not be used when hf_device_map exists")

            def generate(self, **kwargs):
                return [[11, 12, 13, 99]]

        model = ParametersShouldNotBeUsed()
        fake_torch = SimpleNamespace(no_grad=unittest.mock.Mock(return_value=NoGradContext()))

        with patch.object(pap_mod, "_get_local_model", return_value=(tokenizer, model)):
            with patch.object(pap_mod, "torch", fake_torch, create=True):
                generated = pap_mod.generate_with_local_model(
                    model_path="models/gpt-oss-20b",
                    prompt="rewrite this goal",
                    max_new_tokens=32,
                )

        tokenizer.apply_chat_template.return_value.to.assert_called_once_with("cuda:2")
        self.assertEqual(generated, "done")


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
                "--dataset_name",
                "JBB",
                "--input_path",
                str(input_csv),
                "--output_path",
                str(output_json),
                "--model_path",
                "models/gpt-oss-20b",
                "--limit",
                "1",
                "--max_new_tokens",
                "64",
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
