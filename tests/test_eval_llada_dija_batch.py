"""DIJA 模式批量采样测试：逐行 k 调度、matching_count、Phase 2 仅重掩码原始 mask 位置。

复用 test_eval_llada_batch_sample 的确定性 FakeModel：mask 位置有害对齐 0.6
（steering 可压到 0.1），HARM token 对齐 0.9；对齐 > 0.2 生成 HARM，否则 SAFE。
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import eval_llada_steering as llada_mod
from eval_llada_steering import LLaDAEvalHarness
from test_eval_llada_batch_sample import (
    EOS,
    HARM,
    MASK,
    SAFE,
    FakeModel,
    _argmax_sample,
)


class _DijaTokenizer:
    eos_token_id = EOS

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([self.mapping[text]])}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "v:" + messages[0]["content"]


def make_dija_harness(model, mapping, **overrides):
    harness = LLaDAEvalHarness.__new__(LLaDAEvalHarness)
    harness.model = model
    harness.tokenizer = _DijaTokenizer(mapping)
    harness.device = torch.device("cpu")
    harness.mask_id = MASK
    harness.sampling_steps = 2
    harness.initial_steering_ratio = 0.0
    harness.max_refinement_iters = 3
    harness.alignment_threshold = 0.5
    harness.steering_overshoot = 5.0
    harness.steering_vector = torch.tensor([1.0, 0.0])
    harness.steering_bank = None
    harness.target_layer = 0
    harness._build_dija_prompt_text = lambda goal, mask_counts: f"p:{goal}"
    for key, value in overrides.items():
        setattr(harness, key, value)
    return harness


class DijaBatchSchedulingTest(unittest.TestCase):
    MAPPING = {
        "p:g0": [0, 1, MASK, MASK, MASK],
        "p:g1": [1, MASK],
        "v:g0v": [0, 3],
    }

    def test_per_row_k_schedule_and_matching_counts(self):
        # 关闭 steering（无向量/bank）→ 只走 Phase 1，逐行 k 调度：
        # 行 0 有 3 个 mask（step0 揭 2 个、step1 揭 1 个），行 1 只有 1 个。
        harness = make_dija_harness(FakeModel(), self.MAPPING, steering_vector=None)
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            results = harness.llada_dija_sample_batch(["g0", "g1"], ["g0v", None], mask_counts=8)

        ids0, matching0 = results[0]
        ids1, matching1 = results[1]
        # refined [0,1,...] vs vanilla [0,3] 在下标 1 处首次分歧
        self.assertEqual(matching0, 1)
        self.assertEqual(matching1, 0)
        # 未 steer 的 mask 位置全部生成 HARM；行按原始长度截去 pad
        self.assertEqual(ids0.tolist(), [0, 1, HARM, HARM, HARM])
        self.assertEqual(ids1.tolist(), [1, HARM])

    def test_variable_length_rows_pass_attention_mask(self):
        model = FakeModel()
        harness = make_dija_harness(model, self.MAPPING, steering_vector=None)
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            harness.llada_dija_sample_batch(["g0", "g1"], [None, None], mask_counts=8)

        self.assertTrue(all(m is not None for m in model.attention_masks))
        self.assertEqual(model.attention_masks[0][1].tolist(), [1.0, 1.0, 0.0, 0.0, 0.0])


class DijaPhase2Test(unittest.TestCase):
    MAPPING = {
        "p:h0": [0, HARM, MASK],
        "p:h1": [0, 1, 1],
    }

    def test_refines_only_original_mask_positions_with_row_compaction(self):
        model = FakeModel()
        harness = make_dija_harness(model, self.MAPPING)
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            results = harness.llada_dija_sample_batch(["h0", "h1"], [None, None], mask_counts=8)

        # prompt 中的 HARM token（对齐 0.9 > 阈值）不在原始 mask 内 → 不允许被重掩码；
        # 原始 mask 位置先生成 HARM，Phase 2 remask 后经 steering refill 为 SAFE。
        self.assertEqual(results[0][0].tolist(), [0, HARM, SAFE])
        # 行 1 无 mask：原样返回，且不进入 Phase 2
        self.assertEqual(results[1][0].tolist(), [0, 1, 1])
        # Phase 1 一次 [2,3]（step1 无剩余 mask 提前 break）；
        # Phase 2 仅行 0 活跃：检测 [1,3] → refill [1,3] → 复检 [1,3]
        self.assertEqual([shape[0] for shape in model.forward_shapes], [2, 1, 1, 1])


class DijaBatchSingleEquivalenceTest(unittest.TestCase):
    MAPPING = {
        "p:g0": [0, 1, MASK, MASK, MASK],
        "p:g1": [1, MASK],
        "v:g0v": [0, 3],
    }

    def test_batch_matches_single_with_steering(self):
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            singles = [
                make_dija_harness(FakeModel(), self.MAPPING).llada_dija_sample(
                    goal, vanilla_goal=vanilla, mask_counts=8
                )
                for goal, vanilla in [("g0", "g0v"), ("g1", None)]
            ]
            batched = make_dija_harness(FakeModel(), self.MAPPING).llada_dija_sample_batch(
                ["g0", "g1"], ["g0v", None], mask_counts=8
            )

        for (single_ids, single_mc), (batch_ids, batch_mc) in zip(singles, batched):
            self.assertTrue(torch.equal(single_ids[0], batch_ids))
            self.assertEqual(single_mc, batch_mc)


if __name__ == "__main__":
    unittest.main()
