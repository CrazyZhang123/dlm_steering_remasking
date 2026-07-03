"""本地 Llama Guard 批量评判测试（scripts/eval_llama_guard_local.py）。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.eval_llama_guard_local import (
    generate_guard_texts_batch,
    run_evaluation,
)


class _FakeBatchTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None
    pad_token_id = None
    padding_side = "right"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "T:" + messages[0]["content"][0]["text"]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=True):
        assert not add_special_tokens, "模板文本已含特殊 token，不应重复添加"
        # 文本长度即 token 数；最后一个 token 编码内容（bad→7，否则 5）
        rows = [[9] * (len(t) - 1) + [7 if "bad" in t else 5] for t in texts]
        max_len = max(len(r) for r in rows)
        assert self.padding_side == "left"
        input_ids = [[self.pad_token_id or 0] * (max_len - len(r)) + r for r in rows]
        attention = [[0] * (max_len - len(r)) + [1] * len(r) for r in rows]

        class _Batch(dict):
            def to(self, _device):
                return self

        return _Batch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention),
        )

    def decode(self, ids):
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "unsafe\nS9" if 1 in ids else "safe"


class _FakeBatchModel:
    device = torch.device("cpu")

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        # 左 pad 后最后一列即每行最后一个真实 token；bad(7) → 判 unsafe(1)
        new_token = torch.where(
            input_ids[:, -1] == 7, torch.tensor(1), torch.tensor(2)
        ).unsqueeze(1)
        return torch.cat([input_ids, new_token], dim=1)


class GenerateGuardTextsBatchTest(unittest.TestCase):
    def test_left_padded_batch_judges_each_row(self):
        tokenizer = _FakeBatchTokenizer()
        outputs = generate_guard_texts_batch(
            tokenizer, _FakeBatchModel(), ["a bad answer", "ok", "another bad one!"]
        )

        self.assertEqual(outputs, ["unsafe\nS9", "safe", "unsafe\nS9"])
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertIsNotNone(tokenizer.pad_token)


class RunEvaluationBatchTest(unittest.TestCase):
    def _write_data(self, tmpdir):
        data_path = Path(tmpdir) / "results.json"
        data_path.write_text(
            json.dumps([
                {"prompt": "p1", "response": "r1"},
                {"prompt": "p2", "response": "r2"},
                {"prompt": "p3", "response": "r3"},
            ]),
            encoding="utf-8",
        )
        return data_path

    def test_batched_run_preserves_order_and_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._write_data(tmpdir)
            output_path = Path(tmpdir) / "judge_output.json"

            with mock.patch(
                "scripts.eval_llama_guard_local.load_guard_model",
                return_value=(object(), object()),
            ), mock.patch(
                "scripts.eval_llama_guard_local.generate_guard_texts_batch",
                side_effect=[["safe", "unsafe\nS7"], ["unsafe\nS9"]],
            ) as batch_mock:
                payload = run_evaluation(
                    data_path=str(data_path),
                    model_path="/tmp/local-guard",
                    output_path=str(output_path),
                    device="cpu",
                    batch_size=2,
                )

        self.assertEqual(batch_mock.call_count, 2)
        self.assertEqual(batch_mock.call_args_list[0].args[2], ["r1", "r2"])
        self.assertEqual(batch_mock.call_args_list[1].args[2], ["r3"])
        self.assertEqual(payload["metadata"]["unsafe_count"], 2)
        self.assertEqual([row["id"] for row in payload["results"]], [0, 1, 2])
        self.assertEqual(payload["results"][1]["judge_output"], "unsafe\nS7")

    def test_batch_size_one_uses_single_sample_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._write_data(tmpdir)
            output_path = Path(tmpdir) / "judge_output.json"

            with mock.patch(
                "scripts.eval_llama_guard_local.load_guard_model",
                return_value=(object(), object()),
            ), mock.patch(
                "scripts.eval_llama_guard_local.generate_guard_text",
                side_effect=["safe", "safe", "unsafe\nS7"],
            ) as single_mock, mock.patch(
                "scripts.eval_llama_guard_local.generate_guard_texts_batch",
                side_effect=AssertionError("batch_size=1 不应调用批量路径"),
            ):
                payload = run_evaluation(
                    data_path=str(data_path),
                    model_path="/tmp/local-guard",
                    output_path=str(output_path),
                    device="cpu",
                    batch_size=1,
                )

        self.assertEqual(single_mock.call_count, 3)
        self.assertEqual(payload["metadata"]["unsafe_count"], 1)


if __name__ == "__main__":
    unittest.main()
