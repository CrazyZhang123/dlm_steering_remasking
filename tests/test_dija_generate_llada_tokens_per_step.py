import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DIJA_HARMBENCH = ROOT / "DIJA" / "run_harmbench"
if str(DIJA_HARMBENCH) not in sys.path:
    sys.path.insert(0, str(DIJA_HARMBENCH))

from utility.generate_function import generate_llada


MASK_ID = 126336


class _FakeModel:
    """Minimal model stub that returns uniform logits except for mask positions."""

    def __init__(self, vocab_size=128, device="cpu"):
        self.device = torch.device(device)
        self._vocab_size = vocab_size
        self.call_count = 0

    def __call__(self, x, attention_mask=None):
        self.call_count += 1
        logits = torch.zeros(*x.shape, self._vocab_size, device=self.device)
        mask_positions = (x == MASK_ID)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if mask_positions[i, j]:
                    logits[i, j, j % self._vocab_size] = 10.0
        return type("Output", (), {"logits": logits})()


class TestTokensPerStepDefault(unittest.TestCase):
    def test_default_tokens_per_step_equals_one_behavior(self):
        num_masks = 3
        input_ids = torch.cat([
            torch.tensor([[1, 2]]),
            torch.full((1, num_masks), MASK_ID)
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        model = _FakeModel()
        result = generate_llada(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask,
            model=model,
            mask_id=MASK_ID,
        )

        self.assertFalse((result == MASK_ID).any(), "All masks should be filled")
        self.assertEqual(model.call_count, num_masks,
                         "Default tokens_per_step=1 should need exactly num_masks forward calls")


class TestTokensPerStep5(unittest.TestCase):
    def test_tokens_per_step_5_fills_all_masks(self):
        num_masks = 12
        input_ids = torch.cat([
            torch.tensor([[1, 2]]),
            torch.full((1, num_masks), MASK_ID)
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        model = _FakeModel()
        result = generate_llada(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask,
            model=model,
            mask_id=MASK_ID,
            tokens_per_step=5,
        )

        self.assertFalse((result == MASK_ID).any(), "All masks should be filled")
        expected_calls = (num_masks + 5 - 1) // 5  # ceil(12/5) = 3
        self.assertEqual(model.call_count, expected_calls,
                         f"tokens_per_step=5 with {num_masks} masks should need {expected_calls} calls")


class TestTokensPerStepClamp(unittest.TestCase):
    def test_clamps_when_fewer_masks_than_tokens_per_step(self):
        num_masks = 3
        input_ids = torch.cat([
            torch.tensor([[1, 2]]),
            torch.full((1, num_masks), MASK_ID)
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        model = _FakeModel()
        result = generate_llada(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask,
            model=model,
            mask_id=MASK_ID,
            tokens_per_step=5,
        )

        self.assertFalse((result == MASK_ID).any(), "All masks should be filled")
        self.assertEqual(model.call_count, 1,
                         "3 masks with tokens_per_step=5 should complete in 1 call")


if __name__ == "__main__":
    unittest.main()
