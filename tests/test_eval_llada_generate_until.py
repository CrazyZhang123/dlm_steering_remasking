import sys
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_llada_steering as llada_mod
from eval_llada_steering import LLaDAEvalHarness


class DummyHarness:
    attack_method = "zeroshot"
    accelerator = None


class _Req:
    def __init__(self, prompt):
        self.args = (prompt, {"until": []})


class _Tokenizer:
    eos_token = "<eos>"

    def __call__(self, text):
        return {"input_ids": [1, 2, 3] if isinstance(text, str) else [1]}

    def decode(self, ids, skip_special_tokens=False):
        return "safe response"


class _HarnessWithOneSample:
    attack_method = "zeroshot"
    accelerator = None
    tokenizer = _Tokenizer()
    device = torch.device("cpu")
    sampler = "llada"
    mask_length = 3
    generated_samples_path = "./outputs/partial_test"

    def llada_conf_sample(self, prompt):
        return torch.tensor([[1, 2, 3]])


class _HookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True
        return None


class _HookBlock:
    def __init__(self):
        self.handle = _HookHandle()

    def register_forward_hook(self, _hook):
        return self.handle


class _RefillModel:
    def __init__(self):
        self.model = type("Inner", (), {})()
        self.block = _HookBlock()
        self.model.transformer = type("Transformer", (), {"blocks": [self.block]})()

    def __call__(self, xt):
        vocab_size = 128
        return type("Output", (), {"logits": torch.zeros(*xt.shape, vocab_size)})()


class _RefillHarness:
    mask_id = 99
    target_layer = 0

    def __init__(self, model=None):
        self.model = model or _RefillModel()

    def _build_adaptive_steering_hook(self, mask_index):
        self.last_mask_index = mask_index.clone()

        def _hook(_module, _input, output):
            return output

        return _hook


class _PhaseOneHarness(_RefillHarness):
    inject_prompt = False
    mask_id = 99
    mask_length = 2
    block_size = 2
    sampling_steps = 2
    initial_steering_ratio = 1.0
    target_layer = 0
    tokenizer = type("Tokenizer", (), {"eos_token_id": 2})()

    def _can_steer(self):
        return True


class _DijaRefillHarness:
    def __init__(self):
        self.mask_id = 99
        self.target_layer = 0
        self.device = torch.device("cpu")
        self.sampling_steps = 1
        self.initial_steering_ratio = 0.0
        self.max_refinement_iters = 1
        self.alignment_threshold = 0.0
        self.steering_vector = torch.tensor([1.0])
        self.steering_bank = None
        self.refill_args = None

        class _Tokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 99]])}

        class _Block:
            def register_forward_hook(self, hook):
                hidden = torch.tensor([[[0.0], [1.0]]])
                hook(None, None, hidden)

                class _Handle:
                    def remove(self):
                        return None

                return _Handle()

        class _Model:
            def __init__(self):
                self.model = type("Inner", (), {})()
                self.model.transformer = type("Transformer", (), {"blocks": [_Block()]})()

            def __call__(self, xt):
                return type("Output", (), {"logits": torch.zeros(*xt.shape, 128)})()

        self.tokenizer = _Tokenizer()
        self.model = _Model()

    def _build_dija_prompt_text(self, refined_goal, mask_counts):
        return "prompt"

    def _can_steer(self):
        return True

    def _refill_masks_with_steering(self, xt, block_start, block_end):
        self.refill_args = (block_start, block_end)
        return xt.masked_fill(xt == self.mask_id, 7)


class _RecordingAlignmentBank:
    def __init__(self):
        self.calls = []

    def alignment(self, hidden, theta=None, record=False):
        self.calls.append((tuple(hidden.shape), record))
        if hidden.ndim == 3:
            return torch.tensor([[0.0, 1.0]])
        return torch.ones(hidden.shape[0])


class GenerateUntilTest(unittest.TestCase):
    def test_save_partial_results_writes_prompt_response_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llada_mod.save_partial_results(
                tmpdir,
                [{"prompt": "p1", "response": "r1"}],
            )

            saved = json.loads(Path(tmpdir, "results.partial.json").read_text(encoding="utf-8"))

        self.assertEqual(saved, [{"prompt": "p1", "response": "r1"}])

    def test_generate_until_reads_attack_method_attribute(self):
        dummy = DummyHarness()

        results = LLaDAEvalHarness.generate_until(dummy, [])

        self.assertEqual(results, [])

    def test_generate_until_saves_partial_results_for_nonempty_requests(self):
        dummy = _HarnessWithOneSample()

        with patch.object(llada_mod, "save_partial_results") as save_mock:
            results = LLaDAEvalHarness.generate_until(dummy, [_Req("hello")])

        self.assertEqual(results, ["safe response"])
        save_mock.assert_called_once()
        self.assertEqual(
            save_mock.call_args.args[1],
            [{"prompt": "hello", "response": "safe response"}],
        )

    def test_generate_until_does_not_build_hf_dataset_for_empty_requests(self):
        dummy = DummyHarness()

        with patch.object(
            llada_mod.Dataset,
            "from_list",
            side_effect=AssertionError("Dataset.from_list should not be called"),
        ):
            results = LLaDAEvalHarness.generate_until(dummy, [])

        self.assertEqual(results, [])

    def test_refill_masks_with_steering_preserves_future_block_masks(self):
        dummy = _RefillHarness()
        xt = torch.tensor([[10, 11, 99, 99, 99, 99]])
        sampled = torch.tensor([[10, 11, 20, 21, 30, 31]])

        with patch.object(llada_mod, "_sample_categorical", return_value=sampled):
            result = LLaDAEvalHarness._refill_masks_with_steering(dummy, xt, block_start=2, block_end=4)

        self.assertEqual(result.tolist(), [[10, 11, 20, 21, 99, 99]])
        self.assertEqual(dummy.last_mask_index.tolist(), [[False, False, True, True, False, False]])

    def test_refill_masks_with_steering_preserves_previous_block_masks(self):
        dummy = _RefillHarness()
        xt = torch.tensor([[99, 11, 99, 99, 12]])
        sampled = torch.tensor([[30, 11, 20, 21, 12]])

        with patch.object(llada_mod, "_sample_categorical", return_value=sampled):
            result = LLaDAEvalHarness._refill_masks_with_steering(dummy, xt, block_start=2, block_end=4)

        self.assertEqual(result.tolist(), [[99, 11, 20, 21, 12]])
        self.assertEqual(dummy.last_mask_index.tolist(), [[False, False, True, True, False]])

    def test_refill_masks_with_steering_removes_hook_when_model_raises(self):
        class _FailingModel(_RefillModel):
            def __call__(self, xt):
                raise RuntimeError("forward failed")

        model = _FailingModel()
        dummy = _RefillHarness(model=model)
        xt = torch.tensor([[10, 99, 99]])

        with self.assertRaises(RuntimeError):
            LLaDAEvalHarness._refill_masks_with_steering(dummy, xt, block_start=1, block_end=3)

        self.assertTrue(model.block.handle.removed)

    def test_llada_remask_sample_phase1_removes_hook_when_model_raises(self):
        class _FailingModel(_RefillModel):
            device = torch.device("cpu")

            def __call__(self, xt):
                raise RuntimeError("forward failed")

        model = _FailingModel()
        dummy = _PhaseOneHarness(model=model)

        with self.assertRaises(RuntimeError):
            LLaDAEvalHarness.llada_remask_sample(dummy, torch.tensor([[1, 2]]))

        self.assertTrue(model.block.handle.removed)

    def test_llada_dija_sample_phase1_removes_hook_when_model_raises(self):
        class _FailingModel(_RefillModel):
            def __call__(self, xt):
                raise RuntimeError("forward failed")

        class _DijaTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 99, 99]])}

        model = _FailingModel()
        dummy = _PhaseOneHarness(model=model)
        dummy.device = torch.device("cpu")
        dummy.tokenizer = _DijaTokenizer()
        dummy._build_dija_prompt_text = lambda refined_goal, mask_counts: "prompt"

        with self.assertRaises(RuntimeError):
            LLaDAEvalHarness.llada_dija_sample(dummy, "refined", mask_counts=2)

        self.assertTrue(model.block.handle.removed)

    def test_llada_dija_phase2_refill_uses_full_sequence_boundaries(self):
        dummy = _DijaRefillHarness()

        _xt, _matching_count = LLaDAEvalHarness.llada_dija_sample(dummy, "refined", mask_counts=1)

        self.assertEqual(dummy.refill_args, (0, 2))

    def test_llada_dija_phase2_records_diagnostics_only_for_original_masks(self):
        dummy = _DijaRefillHarness()
        bank = _RecordingAlignmentBank()
        dummy.steering_bank = bank
        dummy.steering_vector = None

        _xt, _matching_count = LLaDAEvalHarness.llada_dija_sample(dummy, "refined", mask_counts=1)

        self.assertEqual(bank.calls, [((1, 2, 1), False), ((1, 1), True)])


if __name__ == "__main__":
    unittest.main()
