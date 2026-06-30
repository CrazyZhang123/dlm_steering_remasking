import tempfile
import unittest
from pathlib import Path

import torch

from utils.ct_csd_bank import CTCSDBank


class CTCSDBankTest(unittest.TestCase):
    def _state(self):
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        vectors = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        return {
            "format": "ct_csd_v1",
            "model_family": "llada",
            "target_layer": 31,
            "safe_anchor_type": "sample_balanced_global_safe_mean",
            "safe_mean": torch.zeros(2),
            "centers": centers,
            "centers_unit": torch.nn.functional.normalize(centers, dim=-1),
            "vectors": vectors,
            "vectors_unit": torch.nn.functional.normalize(vectors, dim=-1),
            "cluster_ids": torch.tensor([0, 1]),
            "cluster_sizes": torch.tensor([10, 20]),
            "config": {"method": "ct_csd", "num_total_clusters": 2},
            "mil": {
                "enabled": False,
                "probe_path": None,
                "probe_threshold": None,
                "top_q_ratio": None,
            },
        }

    def test_route_uses_nearest_center(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0]]])

        route = bank.route(hidden)

        self.assertEqual(route.tolist(), [[0, 1]])

    def test_route_accepts_explicit_l2_route_centers(self):
        state = self._state()
        state["preprocess"] = {"mode": "l2_only"}
        state["route_centers"] = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0]]])

        route = bank.route(hidden)

        self.assertEqual(route.tolist(), [[0, 1]])

    def test_route_uses_center_l2_preprocess_instead_of_raw_hidden(self):
        state = self._state()
        state["preprocess"] = {"mode": "center_l2", "mean": torch.tensor([10.0, 0.0])}
        state["route_centers"] = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))

        route = bank.route(torch.tensor([[9.0, 0.0], [11.0, 0.0]]))

        self.assertEqual(route.tolist(), [1, 0])

    def test_pca_route_centers_can_have_different_dim_from_steering_vectors(self):
        state = {
            "format": "ct_csd_v1",
            "model_family": "llada",
            "target_layer": 31,
            "safe_anchor_type": "sample_balanced_global_safe_mean",
            "safe_mean": torch.zeros(3),
            "centers": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "vectors": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            "cluster_ids": torch.tensor([0, 1]),
            "cluster_sizes": torch.tensor([10, 20]),
            "route_centers": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "preprocess": {
                "mode": "center_pca128_l2",
                "mean": torch.zeros(3),
                "pca_components": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "pca_dim": 2,
            },
            "config": {"method": "ct_csd", "num_total_clusters": 2},
        }
        bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))
        hidden = torch.tensor([[3.0, 0.0, 5.0], [0.0, 4.0, 6.0]])

        route = bank.route(hidden)
        score = bank.alignment(hidden)
        steered = bank.steer(hidden, beta=1.0, theta=1.0)

        self.assertEqual(route.tolist(), [0, 1])
        self.assertTrue(torch.allclose(score, torch.tensor([3.0, 4.0]), atol=1e-6))
        self.assertTrue(torch.allclose(steered, torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 6.0]]), atol=1e-6))

    def test_center_preprocess_requires_mean(self):
        state = self._state()
        state["preprocess"] = {"mode": "center_l2"}
        state["route_centers"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        with self.assertRaises(ValueError):
            CTCSDBank.from_state_dict(state, device=torch.device("cpu"))

    def test_alignment_uses_routed_local_vector(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0]]])

        score = bank.alignment(hidden)

        self.assertTrue(torch.allclose(score, torch.tensor([[3.0, 4.0]]), atol=1e-6))

    def test_steer_only_changes_scores_above_threshold(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[2.0, 0.0], [0.0, 0.25]])

        steered = bank.steer(hidden, beta=1.0, theta=1.0)

        expected = torch.tensor([[1.0, 0.0], [0.0, 0.25]])
        self.assertTrue(torch.allclose(steered, expected, atol=1e-6))

    def test_record_alignment_updates_route_and_active_counts(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0], [1.0, 0.0]]])

        _ = bank.alignment(hidden, theta=2.5, record=True)
        diagnostics = bank.diagnostics()

        self.assertEqual(diagnostics["route_count"], [2, 1])
        self.assertEqual(diagnostics["active_count"], [1, 1])
        self.assertEqual(diagnostics["total_routed"], 3)
        self.assertEqual(diagnostics["total_active"], 2)
        self.assertAlmostEqual(diagnostics["activation_rate"], 2 / 3)

    def test_steer_can_record_route_and_active_counts(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[2.0, 0.0], [0.0, 0.25]])

        _ = bank.steer(hidden, beta=1.0, theta=1.0, record=True)
        diagnostics = bank.diagnostics()

        self.assertEqual(diagnostics["route_count"], [1, 1])
        self.assertEqual(diagnostics["active_count"], [1, 0])
        self.assertEqual(diagnostics["total_routed"], 2)
        self.assertEqual(diagnostics["total_active"], 1)

    def test_diagnostics_aggregates_optional_center_categories(self):
        state = self._state()
        state["categories"] = ["fraud", "illegal"]
        state["center_categories"] = ["fraud", "illegal"]
        bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0], [0.0, 3.0]]])

        _ = bank.alignment(hidden, theta=2.5, record=True)
        diagnostics = bank.diagnostics()

        self.assertEqual(diagnostics["center_categories"], ["fraud", "illegal"])
        self.assertEqual(diagnostics["category_route_count"], {"fraud": 1, "illegal": 2})
        self.assertEqual(diagnostics["category_active_count"], {"fraud": 1, "illegal": 2})

    def test_load_rejects_wrong_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"format": "global_sentence_csd"}, path)

            with self.assertRaises(ValueError):
                CTCSDBank.load(path, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
