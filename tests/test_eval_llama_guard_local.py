import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.eval_llama_guard_local import (
    DEFAULT_MODEL_PATH,
    generate_guard_text,
    is_unsafe_text,
    load_guard_model,
    main,
    run_evaluation,
)


class IsUnsafeTextTest(unittest.TestCase):
    def test_returns_true_when_text_contains_unsafe_without_safe_prefix(self):
        self.assertTrue(is_unsafe_text("unsafe\nS7<|eot_id|>"))
        self.assertTrue(is_unsafe_text("category: unsafe"))

    def test_returns_false_when_text_starts_with_safe(self):
        self.assertFalse(is_unsafe_text("safe"))
        self.assertFalse(is_unsafe_text(" safe\nunsafe mentioned later"))
        self.assertFalse(is_unsafe_text(""))


class RunEvaluationTest(unittest.TestCase):
    def test_load_guard_model_sets_sliding_window_defaults(self):
        class FakeConfig:
            sliding_window = None
            attention_chunk_size = None

        class FakeModel:
            def __init__(self):
                self.config = FakeConfig()

            def to(self, _device):
                return self

            def eval(self):
                return self

        fake_model = FakeModel()

        with mock.patch(
            "scripts.eval_llama_guard_local.AutoTokenizer.from_pretrained",
            return_value=object(),
        ), mock.patch(
            "scripts.eval_llama_guard_local.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ):
            load_guard_model("/tmp/local-guard", "cpu")

        self.assertEqual(fake_model.config.sliding_window, 4096)
        self.assertEqual(fake_model.config.attention_chunk_size, 4096)

    def test_generate_guard_text_uses_dynamic_cache(self):
        class FakeTensorDict(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            eos_token_id = 7

            def apply_chat_template(self, *_args, **_kwargs):
                return FakeTensorDict({"input_ids": torch.tensor([[1, 2, 3]])})

            def decode(self, _tokens):
                return "safe"

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[[1, 2, 3, 4]]]

        tokenizer = FakeTokenizer()
        model = FakeModel()

        generate_guard_text(tokenizer, model, "hello")

        self.assertEqual(model.kwargs["cache_implementation"], "dynamic")

    def test_run_evaluation_writes_metadata_and_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "results.json"
            output_path = tmp_path / "judge_output.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"prompt": "p1", "response": "r1"},
                        {"prompt": "p2", "response": "r2"},
                        {"prompt": "p3", "response": "r3"},
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.eval_llama_guard_local.load_guard_model",
                return_value=(object(), object()),
            ) as load_model_mock, mock.patch(
                "scripts.eval_llama_guard_local.generate_guard_text",
                side_effect=["safe", "unsafe\nS7", "unsafe but no safe prefix"],
            ) as generate_mock:
                payload = run_evaluation(
                    data_path=str(data_path),
                    model_path="/tmp/local-guard",
                    output_path=str(output_path),
                    device="cpu",
                )

            self.assertEqual(load_model_mock.call_args.args, ("/tmp/local-guard", "cpu"))
            self.assertEqual(generate_mock.call_count, 3)
            self.assertEqual(payload["metadata"]["total_samples"], 3)
            self.assertEqual(payload["metadata"]["unsafe_count"], 2)
            self.assertAlmostEqual(payload["metadata"]["asr_percent"], 66.67, places=2)
            self.assertEqual(len(payload["results"]), 3)
            self.assertEqual(payload["results"][1]["judge_output"], "unsafe\nS7")
            self.assertTrue(payload["results"][1]["is_unsafe"])

            saved_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload, payload)

    def test_main_uses_default_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "results.json"
            output_path = tmp_path / "judge_output.json"
            data_path.write_text(
                json.dumps([{"prompt": "p1", "response": "r1"}]),
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.eval_llama_guard_local.run_evaluation",
                return_value={"metadata": {}, "results": []},
            ) as run_mock:
                main(
                    [
                        "--data_path",
                        str(data_path),
                        "--output_path",
                        str(output_path),
                        "--device",
                        "cuda:0",
                    ]
                )

            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["data_path"], str(data_path))
            self.assertEqual(kwargs["output_path"], str(output_path))
            self.assertEqual(kwargs["device"], "cuda:0")
            self.assertEqual(kwargs["model_path"], DEFAULT_MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
