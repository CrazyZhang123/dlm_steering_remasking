import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.batching import (
    build_padded_xt,
    forward_logits,
    pad_token_rows,
    rowwise_topk_transfer,
)


MASK_ID = 90
PAD_ID = 91


class BuildPaddedXtTest(unittest.TestCase):
    def test_layout_without_inject(self):
        prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        xt, valid, prompt_lens, total_lens = build_padded_xt(
            prompts, mask_length=2, mask_id=MASK_ID, pad_id=PAD_ID
        )

        self.assertEqual(xt.tolist(), [
            [1, 2, MASK_ID, MASK_ID, PAD_ID],
            [3, 4, 5, MASK_ID, MASK_ID],
        ])
        self.assertEqual(valid.tolist(), [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        self.assertEqual(prompt_lens.tolist(), [2, 3])
        self.assertEqual(total_lens.tolist(), [4, 5])

    def test_layout_with_inject_prompt_duplicates_prompt(self):
        prompts = [torch.tensor([1, 2]), torch.tensor([3])]
        xt, valid, prompt_lens, total_lens = build_padded_xt(
            prompts, mask_length=1, mask_id=MASK_ID, pad_id=PAD_ID, inject_prompt=True
        )

        self.assertEqual(xt.tolist(), [
            [1, 2, 1, 2, MASK_ID],
            [3, 3, MASK_ID, PAD_ID, PAD_ID],
        ])
        self.assertEqual(prompt_lens.tolist(), [4, 2])
        self.assertEqual(total_lens.tolist(), [5, 3])
        self.assertEqual(valid[1].tolist(), [True, True, True, False, False])

    def test_rejects_pad_id_equal_to_mask_id(self):
        with self.assertRaises(AssertionError):
            build_padded_xt([torch.tensor([1])], mask_length=1, mask_id=MASK_ID, pad_id=MASK_ID)


class PadTokenRowsTest(unittest.TestCase):
    def test_pads_variable_rows(self):
        rows = [torch.tensor([1, 2, 3]), torch.tensor([4])]
        xt, valid, total_lens = pad_token_rows(rows, pad_id=PAD_ID)

        self.assertEqual(xt.tolist(), [[1, 2, 3], [4, PAD_ID, PAD_ID]])
        self.assertEqual(valid.tolist(), [[True, True, True], [True, False, False]])
        self.assertEqual(total_lens.tolist(), [3, 1])

    def test_equal_rows_do_not_require_pad_id(self):
        rows = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        xt, valid, _ = pad_token_rows(rows, pad_id=None)

        self.assertEqual(xt.tolist(), [[1, 2], [3, 4]])
        self.assertTrue(valid.all())

    def test_variable_rows_require_pad_id(self):
        with self.assertRaises(AssertionError):
            pad_token_rows([torch.tensor([1, 2]), torch.tensor([3])], pad_id=None)


class RowwiseTopkTransferTest(unittest.TestCase):
    def test_matches_per_row_topk(self):
        torch.manual_seed(0)
        confidence = torch.rand(4, 9, dtype=torch.float64)
        confidence[1, 3:] = -float("inf")
        k = 3

        transfer = rowwise_topk_transfer(confidence, k)

        expected = torch.zeros_like(confidence, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=k)
            expected[j, select_index] = True
        self.assertTrue(torch.equal(transfer, expected))

    def test_clamps_k_to_row_length(self):
        confidence = torch.tensor([[0.1, 0.2]])
        transfer = rowwise_topk_transfer(confidence, 5)
        self.assertTrue(transfer.all())


class ForwardLogitsTest(unittest.TestCase):
    class _Model:
        def __init__(self):
            self.calls = []

        def __call__(self, xt, attention_mask=None):
            self.calls.append(attention_mask)
            return type("Output", (), {"logits": torch.zeros(*xt.shape, 4)})()

    def test_all_valid_keeps_plain_call(self):
        model = self._Model()
        xt = torch.zeros(2, 3, dtype=torch.long)

        forward_logits(model, xt, torch.ones(2, 3, dtype=torch.bool))
        forward_logits(model, xt, None)

        self.assertEqual(model.calls, [None, None])

    def test_padded_batch_passes_attention_mask(self):
        model = self._Model()
        xt = torch.zeros(2, 3, dtype=torch.long)
        valid = torch.tensor([[True, True, True], [True, False, False]])

        forward_logits(model, xt, valid)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0].tolist(), [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
