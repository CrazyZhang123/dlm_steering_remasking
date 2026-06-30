import json
import sys
import tempfile
import unittest
import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_llada_steering as llada_mod
from utils import REFERENCES


class _FakeEvalHarness:
    last_requests = None
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def generate_until(self, requests):
        type(self).last_requests = requests
        return [f"response-{i}" for i, _ in enumerate(requests)]


class RunCsvEvalTest(unittest.TestCase):
    def test_advbench_local_directory_uses_load_from_disk_for_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            local_advbench_dir = Path(tmpdir) / "AdvBench"
            local_advbench_dir.mkdir()
            (local_advbench_dir / "dataset_dict.json").write_text('{"splits":["train"]}', encoding="utf-8")

            args = SimpleNamespace(
                csv_path=str(local_advbench_dir),
                attack_method="prefix",
                model_path="/dev/shm/LLaDA-8B-Instruct",
                generated_samples_path=str(output_dir),
                batch_size=32,
                sampling_steps=128,
                mask_length=128,
                block_size=128,
                remasking="low_confidence",
                sampler="steering",
                remdm_number=4,
                cfg=0.0,
                device="cpu",
                steering_vector_path="",
                steering_overshoot=1.0,
                target_layer=31,
                alignment_threshold=0.0,
                max_refinement_iters=3,
                initial_steering_ratio=0.1,
                dija_mask_counts=128,
                inject_prompt=True,
                self_reminder="False",
            )

            local_dataset = {
                "train": [
                    {"prompt": "harmful prompt a", "target": "target a"},
                    {"prompt": "harmful prompt b", "target": "target b"},
                ]
            }

            with patch.object(llada_mod, "LLaDAEvalHarness", _FakeEvalHarness):
                with patch.object(llada_mod, "load_from_disk", return_value=local_dataset, create=True) as load_disk_mock:
                    with patch.object(
                        llada_mod,
                        "load_dataset",
                        side_effect=AssertionError("load_dataset should not be called for local AdvBench directory"),
                    ):
                        llada_mod.run_csv_eval(args)

            load_disk_mock.assert_called_once_with(str(local_advbench_dir))
            self.assertIsNotNone(_FakeEvalHarness.last_requests)
            self.assertEqual(len(_FakeEvalHarness.last_requests), 2)
            self.assertTrue(_FakeEvalHarness.last_requests[0].args[0].startswith(REFERENCES[0] + " "))

            saved = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(
                saved,
                [
                    {"prompt": "harmful prompt a", "response": "response-0"},
                    {"prompt": "harmful prompt b", "response": "response-1"},
                ],
            )

    def test_prefix_harmbench_source_expands_requests_from_selected_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            csv_path = Path(tmpdir) / "jbb.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prompt"])
                writer.writeheader()
                writer.writerow({"prompt": "goal-a"})
                writer.writerow({"prompt": "goal-b"})

            args = SimpleNamespace(
                csv_path=str(csv_path),
                attack_method="prefix",
                prefix_source="harmbench_human_jailbreaks",
                prefix_random_subset=2,
                prefix_seed=1,
                model_path="/dev/shm/LLaDA-8B-Instruct",
                generated_samples_path=str(output_dir),
                batch_size=32,
                sampling_steps=128,
                mask_length=128,
                block_size=128,
                remasking="low_confidence",
                sampler="llada",
                remdm_number=4,
                cfg=0.0,
                device="cpu",
                steering_vector_path="",
                steering_overshoot=1.0,
                target_layer=31,
                alignment_threshold=0.0,
                max_refinement_iters=3,
                initial_steering_ratio=0.1,
                dija_mask_counts=128,
                inject_prompt=True,
                self_reminder="False",
            )

            fake_prefixes = ["prefix-a", "prefix-b", "prefix-c"]

            with patch.object(llada_mod, "LLaDAEvalHarness", _FakeEvalHarness):
                with patch.object(
                    llada_mod,
                    "load_harmbench_prefix_templates",
                    return_value=fake_prefixes,
                    create=True,
                ):
                    llada_mod.run_csv_eval(args)

            self.assertIsNotNone(_FakeEvalHarness.last_requests)
            self.assertEqual(len(_FakeEvalHarness.last_requests), 4)
            prompts = [req.args[0] for req in _FakeEvalHarness.last_requests]
            self.assertEqual(
                prompts,
                [
                    "prefix-b\n\ngoal-a",
                    "prefix-c\n\ngoal-a",
                    "prefix-b\n\ngoal-b",
                    "prefix-c\n\ngoal-b",
                ],
            )

            saved = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(
                saved,
                [
                    {"prompt": "prefix-b\n\ngoal-a", "response": "response-0"},
                    {"prompt": "prefix-c\n\ngoal-a", "response": "response-1"},
                    {"prompt": "prefix-b\n\ngoal-b", "response": "response-2"},
                    {"prompt": "prefix-c\n\ngoal-b", "response": "response-3"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
