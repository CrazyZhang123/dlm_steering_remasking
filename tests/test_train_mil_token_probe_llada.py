import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils import train_mil_token_probe_llada as trainer


class LinearMILProbeHelpersTest(unittest.TestCase):
    def test_top_q_pool_logits_uses_highest_scores(self):
        logits = torch.tensor([0.1, 0.9, 0.2, 0.8])

        pooled = trainer.top_q_pool_logits(logits, top_q_ratio=0.5)

        self.assertTrue(torch.allclose(pooled, torch.tensor(0.85)))

    def test_top_q_pool_logits_keeps_at_least_one_token(self):
        logits = torch.tensor([0.4])

        pooled = trainer.top_q_pool_logits(logits, top_q_ratio=0.1)

        self.assertTrue(torch.allclose(pooled, torch.tensor(0.4)))

    def test_compute_bag_loss_returns_scalar_for_both_labels(self):
        probe = trainer.LinearMILProbe(input_dim=2)
        hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        harmful_loss = trainer.compute_bag_loss(probe, hidden, 1.0, top_q_ratio=0.5)
        safe_loss = trainer.compute_bag_loss(probe, hidden, 0.0, top_q_ratio=0.5)

        self.assertEqual(harmful_loss.shape, torch.Size([]))
        self.assertEqual(safe_loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(harmful_loss))
        self.assertTrue(torch.isfinite(safe_loss))

    def test_build_probe_state_records_contract_fields(self):
        probe = trainer.LinearMILProbe(input_dim=2)
        args = argparse.Namespace(
            model_path="/dev/shm/LLaDA-8B-Instruct",
            harmful_json="data/harmbench_csd_train.json",
            refusals_txt="utils/refusals.txt",
            target_layer=31,
            top_q_ratio=0.1,
            max_response_len=128,
            max_total_len=2048,
            seed=42,
        )
        metrics = {
            "train_loss": 0.3,
            "val_loss": 0.4,
            "val_accuracy": 0.8,
            "val_auc": 0.75,
        }

        state = trainer.build_probe_state(probe, args, metrics, input_dim=2)

        self.assertEqual(state["format"], "mil_token_probe_v1")
        self.assertEqual(state["model_family"], "llada")
        self.assertEqual(state["target_layer"], 31)
        self.assertEqual(state["input_dim"], 2)
        self.assertEqual(state["top_q_ratio"], 0.1)
        self.assertIn("linear.weight", state["state_dict"])
        self.assertEqual(state["config"]["seed"], 42)
        self.assertEqual(state["metrics"]["val_auc"], 0.75)


class MILTrainingPipelineTest(unittest.TestCase):
    def test_split_bags_keeps_validation_and_train_examples(self):
        bags = [
            trainer.MILBag(torch.ones(1, 2), 1.0),
            trainer.MILBag(torch.ones(1, 2), 1.0),
            trainer.MILBag(torch.zeros(1, 2), 0.0),
            trainer.MILBag(torch.zeros(1, 2), 0.0),
        ]

        train_bags, val_bags = trainer.split_bags(bags, val_ratio=0.5, seed=7)

        self.assertEqual(len(train_bags), 2)
        self.assertEqual(len(val_bags), 2)
        self.assertEqual(sorted(bag.label for bag in train_bags), [0.0, 1.0])
        self.assertEqual(sorted(bag.label for bag in val_bags), [0.0, 1.0])

    def test_evaluate_probe_returns_required_metrics(self):
        probe = trainer.LinearMILProbe(input_dim=2)
        bags = [
            trainer.MILBag(torch.tensor([[1.0, 0.0], [2.0, 0.0]]), 1.0),
            trainer.MILBag(torch.tensor([[0.0, 1.0], [0.0, 2.0]]), 0.0),
        ]

        metrics = trainer.evaluate_probe(probe, bags, top_q_ratio=0.5, device=torch.device("cpu"))

        self.assertEqual(
            sorted(metrics),
            ["accuracy", "auc", "loss"],
        )
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)

    def test_main_saves_probe_and_metrics_with_mocked_training_data(self):
        class _LoadableModel:
            config = type("Config", (), {"hidden_size": 2})()

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        bags = [
            trainer.MILBag(torch.tensor([[2.0, 0.0], [1.0, 0.0]]), 1.0),
            trainer.MILBag(torch.tensor([[0.0, 2.0], [0.0, 1.0]]), 0.0),
            trainer.MILBag(torch.tensor([[3.0, 0.0], [1.0, 0.0]]), 1.0),
            trainer.MILBag(torch.tensor([[0.0, 3.0], [0.0, 1.0]]), 0.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mil_token_probe_llada.pt"
            harmful_path = Path(tmpdir) / "harmful.json"
            refusals_path = Path(tmpdir) / "refusals.txt"
            harmful_path.write_text('[{"prompt": "p", "response": "h"}]', encoding="utf-8")
            refusals_path.write_text("safe\n", encoding="utf-8")

            with patch.object(trainer.AutoTokenizer, "from_pretrained", return_value=object()):
                with patch.object(trainer.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(trainer, "load_harmful_data", return_value=[{"prompt": "p", "response": "h"}]):
                        with patch.object(trainer, "load_refusals", return_value=["safe"]):
                            with patch.object(trainer, "build_mil_bags", return_value=(bags, 0)):
                                trainer.main(
                                    [
                                        "--model_path",
                                        "dummy-model",
                                        "--harmful_json",
                                        str(harmful_path),
                                        "--refusals_txt",
                                        str(refusals_path),
                                        "--output_path",
                                        str(output_path),
                                        "--target_layer",
                                        "31",
                                        "--top_q_ratio",
                                        "0.5",
                                        "--max_response_len",
                                        "128",
                                        "--max_total_len",
                                        "2048",
                                        "--epochs",
                                        "1",
                                        "--val_ratio",
                                        "0.5",
                                        "--device",
                                        "cpu",
                                        "--seed",
                                        "42",
                                    ]
                                )

            metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
            state = torch.load(output_path, map_location="cpu", weights_only=True)
            metrics_exists = metrics_path.exists()

        self.assertEqual(state["format"], "mil_token_probe_v1")
        self.assertEqual(state["target_layer"], 31)
        self.assertEqual(state["top_q_ratio"], 0.5)
        self.assertTrue(metrics_exists)


if __name__ == "__main__":
    unittest.main()
