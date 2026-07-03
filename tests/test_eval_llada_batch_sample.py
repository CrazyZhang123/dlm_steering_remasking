"""批量采样与单样本实现的等价性测试。

FakeModel 是逐位置独立的确定性模型（logits 只依赖该位置 token 与绝对位置），
因此批量（右 pad + attention_mask）结果必须与逐样本结果逐 token 一致；
任何 pad 泄漏、逐行 block 边界错位、done 冻结或行压缩的错误都会破坏该等价性。
模型的 hidden 第 0 维即"有害对齐"分量，端到端驱动真实的 steering hook 与
Phase 2 检测/重掩码路径。
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_llada_steering as llada_mod
from eval_llada_steering import LLaDAEvalHarness


VOCAB = 8
SAFE = 2
HARM = 3
EOS_TRIGGER = 4   # prompt 中出现该 token 的行，mask 位置直接生成 EOS
HARM_TRIGGER = 5  # trigger_gated 模式下，只有含该 token 的行才会产生 HARM
MASK = 6
EOS = 7


def _argmax_sample(categorical_probs):
    return categorical_probs.argmax(dim=-1)


class _Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        block = self

        class _Handle:
            def remove(_self):
                if hook in block.hooks:
                    block.hooks.remove(hook)

        return _Handle()


class FakeModel:
    """逐位置独立的确定性 DLM 替身。

    hidden[..., 0] 为有害对齐分量：mask 位置 0.6（可被 steering 压到 0.1），
    HARM token 0.9，其余 -1.0。对齐 > 0.2 的位置生成 HARM，否则 SAFE；
    置信度随绝对位置递减，保证揭码顺序确定。
    """

    device = torch.device("cpu")

    def __init__(self, trigger_gated=False, eos_gated=False):
        self.block = _Block()
        self.model = type("Inner", (), {})()
        self.model.transformer = type("Transformer", (), {"blocks": [self.block]})()
        self.trigger_gated = trigger_gated
        self.eos_gated = eos_gated
        self.forward_shapes = []
        self.attention_masks = []

    def __call__(self, xt, attention_mask=None):
        self.forward_shapes.append(tuple(xt.shape))
        self.attention_masks.append(None if attention_mask is None else attention_mask.clone())
        batch, length = xt.shape

        mask_potential = torch.full((batch, 1), 0.6)
        if self.trigger_gated:
            has_trigger = (xt == HARM_TRIGGER).any(dim=1, keepdim=True)
            mask_potential = torch.where(has_trigger, torch.tensor(0.6), torch.tensor(-1.0))

        hidden = torch.zeros(batch, length, 2)
        hidden[..., 0] = torch.where(
            xt == MASK,
            mask_potential.expand(batch, length),
            torch.where(xt == HARM, torch.tensor(0.9), torch.tensor(-1.0)),
        )

        for hook in list(self.block.hooks):
            out = hook(self.block, (xt,), hidden)
            if out is not None:
                hidden = out[0] if isinstance(out, tuple) else out

        token = torch.where(hidden[..., 0] > 0.2, torch.tensor(HARM), torch.tensor(SAFE))
        if self.eos_gated:
            eos_positions = (xt == EOS_TRIGGER).any(dim=1, keepdim=True) & (xt == MASK)
            token = torch.where(eos_positions, torch.tensor(EOS), token)

        logits = torch.full((batch, length, VOCAB), -10.0)
        position_bonus = 5.0 - 0.01 * torch.arange(length, dtype=torch.float32)
        logits.scatter_(2, token.unsqueeze(-1), position_bonus.expand(batch, length).unsqueeze(-1))
        return type("Output", (), {"logits": logits})()


def make_harness(model, **overrides):
    harness = LLaDAEvalHarness.__new__(LLaDAEvalHarness)
    harness.model = model
    harness.tokenizer = type("Tok", (), {"eos_token_id": EOS})()
    harness.device = torch.device("cpu")
    harness.mask_id = MASK
    harness.mask_length = 4
    harness.block_size = 2
    harness.sampling_steps = 4
    harness.initial_steering_ratio = 0.5
    harness.max_refinement_iters = 3
    harness.alignment_threshold = 0.5
    harness.steering_overshoot = 5.0
    harness.steering_vector = torch.tensor([1.0, 0.0])
    harness.steering_bank = None
    harness.target_layer = 0
    harness.inject_prompt = False
    for key, value in overrides.items():
        setattr(harness, key, value)
    return harness


class BatchSingleEquivalenceTest(unittest.TestCase):
    def test_remask_batch_matches_single_for_mixed_lengths(self):
        prompts = [torch.tensor([0, 1]), torch.tensor([1, 0, 1])]
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            singles = [
                make_harness(FakeModel()).llada_remask_sample(p.unsqueeze(0))[0]
                for p in prompts
            ]
            batched = make_harness(FakeModel()).llada_remask_sample_batch(prompts)

        self.assertEqual(len(batched), 2)
        for single, batch_row, prompt in zip(singles, batched, prompts):
            self.assertEqual(batch_row.shape[0], prompt.shape[0] + 4)
            self.assertTrue(torch.equal(single, batch_row))

    def test_generated_region_is_safe_and_contains_no_mask_or_pad(self):
        prompts = [torch.tensor([0, 1]), torch.tensor([1, 0, 1])]
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            batched = make_harness(FakeModel()).llada_remask_sample_batch(prompts)

        for row, prompt in zip(batched, prompts):
            # Phase 1 未 steer 的步骤会先生成 HARM，Phase 2 必须全部修复为 SAFE
            self.assertEqual(row[-4:].tolist(), [SAFE] * 4)
            self.assertTrue(torch.equal(row[: prompt.shape[0]], prompt))
            self.assertFalse((row == MASK).any())

    def test_conf_batch_matches_single_for_mixed_lengths(self):
        prompts = [torch.tensor([0, 1]), torch.tensor([1, 0, 1])]
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            singles = [
                make_harness(FakeModel()).llada_conf_sample(p.unsqueeze(0))[0]
                for p in prompts
            ]
            batched = make_harness(FakeModel()).llada_conf_sample_batch(prompts)

        for single, batch_row in zip(singles, batched):
            self.assertTrue(torch.equal(single, batch_row))

    def test_inject_prompt_batch_matches_single(self):
        prompts = [torch.tensor([0, 1]), torch.tensor([1, 0, 1])]
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            singles = [
                make_harness(FakeModel(), inject_prompt=True).llada_remask_sample(p.unsqueeze(0))[0]
                for p in prompts
            ]
            batched = make_harness(FakeModel(), inject_prompt=True).llada_remask_sample_batch(prompts)

        for single, batch_row, prompt in zip(singles, batched, prompts):
            self.assertTrue(torch.equal(single, batch_row))
            length = prompt.shape[0]
            self.assertTrue(torch.equal(batch_row[length : 2 * length], prompt))


class AttentionMaskUsageTest(unittest.TestCase):
    def test_single_and_equal_length_batch_do_not_pass_attention_mask(self):
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            model = FakeModel()
            make_harness(model).llada_remask_sample(torch.tensor([[0, 1]]))
            self.assertTrue(all(m is None for m in model.attention_masks))

            model = FakeModel()
            make_harness(model).llada_remask_sample_batch(
                [torch.tensor([0, 1]), torch.tensor([1, 0])]
            )
            self.assertTrue(all(m is None for m in model.attention_masks))

    def test_mixed_length_batch_passes_pad_aware_attention_mask(self):
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            model = FakeModel()
            make_harness(model).llada_remask_sample_batch(
                [torch.tensor([0, 1]), torch.tensor([1, 0, 1])]
            )

        self.assertTrue(all(m is not None for m in model.attention_masks))
        # 短样本 total_len=6，长样本 total_len=7 → 短样本行最后一列为 pad
        self.assertEqual(model.attention_masks[0].tolist(), [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])


class Phase2RowCompactionTest(unittest.TestCase):
    def test_converged_rows_leave_refinement_batch(self):
        # 行 0 无 HARM_TRIGGER：全程生成 SAFE，Phase 2 首轮即收敛退出；
        # 行 1 含 HARM_TRIGGER：未 steer 的步骤生成 HARM，需一轮 remask+refill。
        model = FakeModel(trigger_gated=True)
        harness = make_harness(model)
        prompts = [torch.tensor([0, 1]), torch.tensor([0, HARM_TRIGGER])]

        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            batched = harness.llada_remask_sample_batch(prompts)

        # 每个 block：Phase 1 两步 [2,·]；Phase 2 检测 [2,·] → 仅行 1 refill [1,·]
        # → 复检 [1,·] 收敛。两个 block 形状序列相同。
        batch_dims = [shape[0] for shape in model.forward_shapes]
        self.assertEqual(batch_dims, [2, 2, 2, 1, 1] * 2)
        self.assertEqual(batched[0][-4:].tolist(), [SAFE] * 4)
        self.assertEqual(batched[1][-4:].tolist(), [SAFE] * 4)

    def test_compacted_batch_matches_single_samples(self):
        prompts = [torch.tensor([0, 1]), torch.tensor([0, HARM_TRIGGER])]
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            singles = [
                make_harness(FakeModel(trigger_gated=True)).llada_remask_sample(p.unsqueeze(0))[0]
                for p in prompts
            ]
            batched = make_harness(FakeModel(trigger_gated=True)).llada_remask_sample_batch(prompts)

        for single, batch_row in zip(singles, batched):
            self.assertTrue(torch.equal(single, batch_row))


class EosRowFreezeTest(unittest.TestCase):
    def test_eos_row_freezes_and_keeps_later_blocks_masked(self):
        # 行 1 含 EOS_TRIGGER：block 0 生成 EOS 后整行冻结（对应原实现的提前 return），
        # block 1 的 mask 保持不变；行 0 正常完成全部 block。
        model = FakeModel(eos_gated=True)
        harness = make_harness(model)
        prompts = [torch.tensor([0, 1]), torch.tensor([0, EOS_TRIGGER])]

        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            batched = harness.llada_remask_sample_batch(prompts)

        self.assertEqual(batched[0][-4:].tolist(), [SAFE] * 4)
        self.assertEqual(batched[1][-4:].tolist(), [EOS, EOS, MASK, MASK])

    def test_frozen_row_matches_single_sample_early_return(self):
        prompt = torch.tensor([0, EOS_TRIGGER])
        with patch.object(llada_mod, "_sample_categorical", _argmax_sample):
            single = make_harness(FakeModel(eos_gated=True)).llada_remask_sample(prompt.unsqueeze(0))[0]
            batched = make_harness(FakeModel(eos_gated=True)).llada_remask_sample_batch(
                [prompt, torch.tensor([0, 1])]
            )

        self.assertTrue(torch.equal(single, batched[0]))


class _GenTokenizer:
    eos_token = "<eos>"

    def __call__(self, text):
        return {"input_ids": [int(text[1:])]}

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return f"r{ids[-1]}"


class _Req:
    def __init__(self, prompt):
        self.args = (prompt, {"until": []})


class GenerateUntilBatchTest(unittest.TestCase):
    def _make_gen_harness(self, gen_batch_size):
        harness = LLaDAEvalHarness.__new__(LLaDAEvalHarness)
        harness.attack_method = "zeroshot"
        harness.accelerator = None
        harness.tokenizer = _GenTokenizer()
        harness.device = torch.device("cpu")
        harness.sampler = "llada"
        harness.mask_length = 1
        harness.generated_samples_path = "./outputs/partial_test"
        harness.gen_batch_size = gen_batch_size
        # 恒等采样替身：直接把 prompt token 作为生成结果返回
        harness.llada_conf_sample = lambda prompt: prompt
        harness.llada_conf_sample_batch = lambda prompts: list(prompts)
        return harness

    def test_chunked_generation_preserves_request_order(self):
        harness = self._make_gen_harness(gen_batch_size=2)
        requests = [_Req("q0"), _Req("q1"), _Req("q2")]

        with patch.object(llada_mod, "save_partial_results") as save_mock:
            out = LLaDAEvalHarness.generate_until(harness, requests)

        self.assertEqual(out, ["r0", "r1", "r2"])
        # 3 条请求、batch=2 → 两个 chunk，各落盘一次（累计式）
        self.assertEqual(save_mock.call_count, 2)
        self.assertEqual(
            [row["prompt"] for row in save_mock.call_args.args[1]],
            ["q0", "q1", "q2"],
        )

    def test_batch_size_one_uses_single_sample_path(self):
        harness = self._make_gen_harness(gen_batch_size=1)
        harness.llada_conf_sample_batch = None  # bs=1 不应触碰批量入口
        requests = [_Req("q5")]

        with patch.object(llada_mod, "save_partial_results"):
            out = LLaDAEvalHarness.generate_until(harness, requests)

        self.assertEqual(out, ["r5"])


if __name__ == "__main__":
    unittest.main()
