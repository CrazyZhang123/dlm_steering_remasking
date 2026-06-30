import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_llada_steering as llada_mod


class _FakeAccelerator:
    num_processes = 1
    device = "cpu"


class _Block:
    def register_forward_hook(self, _hook):
        class _Handle:
            def remove(self):
                return None

        return _Handle()


class _FakeModel:
    config = type("Config", (), {"hidden_size": 2})()

    def __init__(self):
        self.model = type("Inner", (), {})()
        self.model.transformer = type("Transformer", (), {"blocks": [_Block() for _ in range(32)]})()

    def eval(self):
        return self

    def to(self, _device):
        return self


class _RecordingBank:
    def __init__(self):
        self.record_values = []

    def steer(self, hidden, beta, theta, record=False):
        self.record_values.append(record)
        return hidden

    def diagnostics(self):
        return {"total_routed": 1}


class _NonMainAccelerator:
    is_main_process = False


class EvalBankLoadingTest(unittest.TestCase):
    def _construct(self, steering_vector_path):
        with patch.object(llada_mod.accelerate, "Accelerator", return_value=_FakeAccelerator()):
            with patch.object(llada_mod.AutoModel, "from_pretrained", return_value=_FakeModel()):
                with patch.object(llada_mod.AutoTokenizer, "from_pretrained", return_value=object()):
                    return llada_mod.LLaDAEvalHarness(
                        model_path="dummy-model",
                        generated_samples_path="./outputs/test",
                        batch_size=32,
                        mc_num=32,
                        device="cpu",
                        target_layer=31,
                        steering_vector_path=str(steering_vector_path),
                    )

    def test_loads_ct_csd_bank_without_setting_global_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ct_csd_bank.pt"
            torch.save(
                {
                    "format": "ct_csd_v1",
                    "model_family": "llada",
                    "target_layer": 31,
                    "safe_anchor_type": "sample_balanced_global_safe_mean",
                    "safe_mean": torch.zeros(2),
                    "centers": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "centers_unit": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "vectors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "vectors_unit": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "cluster_ids": torch.tensor([0, 1]),
                    "cluster_sizes": torch.tensor([3, 4]),
                    "config": {"method": "ct_csd", "num_total_clusters": 2},
                    "mil": {
                        "enabled": False,
                        "probe_path": None,
                        "probe_threshold": None,
                        "top_q_ratio": None,
                    },
                },
                path,
            )

            harness = self._construct(path)

        self.assertIsNone(harness.steering_vector)
        self.assertIsNotNone(harness.steering_bank)
        self.assertEqual(harness.max_refinement_iters, 5)

    def test_keeps_old_layer_vector_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "steering_vectors.pt"
            torch.save({"layer_31": torch.tensor([1.0, 0.0])}, path)

            harness = self._construct(path)

        self.assertIsNotNone(harness.steering_vector)
        self.assertIsNone(harness.steering_bank)
        self.assertTrue(torch.equal(harness.steering_vector, torch.tensor([1.0, 0.0])))

    def test_phase1_bank_steering_does_not_record_diagnostics(self):
        bank = _RecordingBank()
        harness = type(
            "Harness",
            (),
            {
                "steering_bank": bank,
                "steering_vector": None,
                "steering_overshoot": 1.0,
                "alignment_threshold": 0.0,
            },
        )()
        mask_index = torch.tensor([[False, True]])
        hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])

        hook = llada_mod.LLaDAEvalHarness._build_adaptive_steering_hook(harness, mask_index)
        _ = hook(None, None, hidden)

        self.assertEqual(bank.record_values, [False])

    def test_non_main_process_does_not_write_ct_csd_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = type(
                "Harness",
                (),
                {
                    "steering_bank": _RecordingBank(),
                    "generated_samples_path": tmpdir,
                    "accelerator": _NonMainAccelerator(),
                },
            )()

            llada_mod.LLaDAEvalHarness.write_steering_diagnostics(harness)

            self.assertFalse(Path(tmpdir, "ct_csd_diagnostics.json").exists())

    def _stage5_bank_state(self):
        # Stage 5/6 新版 bank：含 preprocess(center_pca) 与 route_centers，
        # 路由特征维度(2)与 steering 向量维度(3)不同，验证 routing/steering 解耦。
        return {
            "format": "ct_csd_v1",
            "model_family": "llada",
            "target_layer": 31,
            "safe_anchor_type": "sample_balanced_global_safe_mean",
            "safe_mean": torch.zeros(3),
            "centers": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "raw_centers": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "vectors": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            "cluster_ids": torch.tensor([0, 1]),
            "cluster_sizes": torch.tensor([3, 4]),
            "route_centers": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "preprocess": {
                "mode": "center_pca128_l2",
                "mean": torch.zeros(3),
                "pca_components": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "pca_dim": 2,
            },
            "config": {
                "method": "ct_csd",
                "num_total_clusters": 2,
                "feature_preprocess": "center_pca128_l2",
            },
        }

    def test_loads_stage5_preprocessed_bank_with_mismatched_route_dim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ct_csd_bank.pt"
            torch.save(self._stage5_bank_state(), path)

            harness = self._construct(path)

        self.assertIsNone(harness.steering_vector)
        self.assertIsNotNone(harness.steering_bank)
        self.assertEqual(harness.steering_bank.preprocess["mode"], "center_pca128_l2")

    def test_stage5_bank_alignment_and_steering_on_eval_paths(self):
        # 直接验证 eval 推理路径用到的 _per_token_alignment 与 steering hook
        # 在 PCA 新版 bank（route 维度≠steering 维度）下端到端正确。
        from utils.ct_csd_bank import CTCSDBank

        bank = CTCSDBank.from_state_dict(self._stage5_bank_state(), device=torch.device("cpu"))
        harness = type(
            "Harness",
            (),
            {
                "steering_bank": bank,
                "steering_vector": None,
                "steering_overshoot": 1.0,
                "alignment_threshold": 1.0,
            },
        )()

        # _per_token_alignment：hidden 投到对应局部 steering 向量上的得分
        block_hidden = torch.tensor([[[3.0, 0.0, 5.0], [0.0, 4.0, 6.0]]])
        alignment = llada_mod.LLaDAEvalHarness._per_token_alignment(harness, block_hidden)
        self.assertTrue(torch.allclose(alignment, torch.tensor([[3.0, 4.0]]), atol=1e-6))

        # steering hook：仅对 alignment 超过 theta 的 mask 位置做扰动
        mask_index = torch.tensor([[True, True]])
        hook = llada_mod.LLaDAEvalHarness._build_adaptive_steering_hook(harness, mask_index)
        steered = hook(None, None, block_hidden.clone())
        out = steered[0] if isinstance(steered, tuple) else steered
        # cluster0: score=3>1 → 减 (3-1)*[1,0,0]；cluster1: score=4>1 → 减 (4-1)*[0,1,0]
        expected = torch.tensor([[[1.0, 0.0, 5.0], [0.0, 1.0, 6.0]]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
