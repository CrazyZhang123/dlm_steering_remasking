import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _HookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _HookBlock:
    def __init__(self, hidden):
        self.hidden = hidden
        self.hook = None
        self.handle = _HookHandle()
        self.register_calls = 0

    def register_forward_hook(self, hook):
        self.hook = hook
        self.register_calls += 1
        return self.handle


class _LayeredHookModel:
    def __init__(self, hiddens):
        blocks = [_HookBlock(hidden) for hidden in hiddens]
        self.model = type("Inner", (), {})()
        self.model.transformer = type("Transformer", (), {"blocks": blocks})()

    def __call__(self, input_ids):
        for block in self.model.transformer.blocks:
            if block.hook is not None:
                block.hook(None, None, block.hidden)
        return object()


class _LoadedModel:
    def __init__(self, num_layers):
        self.model = type("Inner", (), {})()
        self.model.transformer = type("Transformer", (), {"blocks": [object() for _ in range(num_layers)]})()

    def to(self, device):
        return self

    def eval(self):
        return self


class SureSorryBuilderTest(unittest.TestCase):
    def import_builder(self):
        try:
            return importlib.import_module("utils.make_sure_sorry_csd_llada")
        except ModuleNotFoundError as exc:
            self.fail(f"未找到 utils.make_sure_sorry_csd_llada 模块：{exc}")

    def test_extract_single_layer_mean_only_reads_requested_block(self):
        builder = self.import_builder()
        hiddens = [
            torch.tensor([[[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]]),
            torch.tensor([[[0.0, 0.0], [1.0, 3.0], [5.0, 7.0]]]),
            torch.tensor([[[200.0, 200.0], [200.0, 200.0], [200.0, 200.0]]]),
        ]
        model = _LayeredHookModel(hiddens)

        mean = builder.extract_single_layer_mean(
            model=model,
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long),
            response_start=1,
            target_layer=1,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(mean, torch.tensor([3.0, 5.0])))
        blocks = model.model.transformer.blocks
        self.assertEqual(blocks[0].register_calls, 0)
        self.assertEqual(blocks[1].register_calls, 1)
        self.assertEqual(blocks[2].register_calls, 0)
        self.assertTrue(blocks[1].handle.removed)

    def test_build_direction_vector_averages_pos_minus_neg_and_tracks_skips(self):
        builder = self.import_builder()
        harmful = [
            {"prompt": "prompt-1"},
            {"prompt": "   "},
            {"prompt": "prompt-2"},
        ]
        positive = "Sure"
        negative = "Sorry"

        ids_by_pair = {
            ("prompt-1", positive): torch.tensor([11], dtype=torch.long),
            ("prompt-1", negative): torch.tensor([12], dtype=torch.long),
            ("prompt-2", positive): torch.tensor([21], dtype=torch.long),
            ("prompt-2", negative): torch.tensor([22], dtype=torch.long),
        }
        means_by_id = {
            11: torch.tensor([4.0, 1.0]),
            12: torch.tensor([1.0, 1.0]),
            21: torch.tensor([6.0, 2.0]),
            22: torch.tensor([2.0, 3.0]),
        }

        def fake_build_sequence(tokenizer, prompt, response, max_response_len):
            return ids_by_pair[(prompt, response)], 0

        def fake_extract_single_layer_mean(model, input_ids, response_start, target_layer, device):
            return means_by_id[int(input_ids[0].item())]

        with patch.object(builder, "build_sequence", side_effect=fake_build_sequence), patch.object(
            builder,
            "extract_single_layer_mean",
            side_effect=fake_extract_single_layer_mean,
        ):
            result = builder.build_direction_vector(
                model=object(),
                tokenizer=object(),
                harmful=harmful,
                positive_response=positive,
                negative_response=negative,
                target_layer=31,
                max_response_len=128,
                device=torch.device("cpu"),
            )

        self.assertTrue(torch.allclose(result["vector"], torch.tensor([3.5, -0.5])))
        self.assertEqual(result["n_ok"], 2)
        self.assertEqual(result["skipped"], 1)

    def test_main_saves_layer_key_and_nonzero_norm(self):
        builder = self.import_builder()
        fake_vector = torch.tensor([3.0, 4.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            argv = [
                "make_sure_sorry_csd_llada.py",
                "--output_dir",
                str(out_dir),
                "--positive_response",
                "Sure",
                "--negative_response",
                "Sorry",
            ]

            with patch.object(builder, "load_harmful_data", return_value=[{"prompt": "prompt-1"}]), patch.object(
                builder,
                "build_direction_vector",
                return_value={"vector": fake_vector, "n_ok": 1, "skipped": 0},
            ), patch.object(builder, "set_seed"), patch.object(
                builder.AutoTokenizer,
                "from_pretrained",
                return_value=object(),
            ), patch.object(
                builder.AutoModel,
                "from_pretrained",
                return_value=_LoadedModel(num_layers=32),
            ), patch.object(sys, "argv", argv):
                builder.main()

            payload = torch.load(out_dir / "steering_vectors.pt", map_location="cpu", weights_only=True)
            self.assertEqual(sorted(payload.keys()), ["layer_31"])
            self.assertAlmostEqual(float(payload["layer_31"].norm().item()), 5.0, places=4)

    def test_resolve_under_cwd_falls_back_to_git_common_root(self):
        builder = self.import_builder()

        with tempfile.TemporaryDirectory() as tmpdir:
            common_root = Path(tmpdir) / "repo"
            worktree_root = common_root / ".worktrees" / "sure-sorry-csd-scan"
            dataset = common_root / ".worktrees" / "harmbench-csd-export" / "data" / "harmbench_testcase_harmful.json"
            dataset.parent.mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)
            dataset.write_text("[]", encoding="utf-8")

            with patch.object(
                builder,
                "subprocess",
                SimpleNamespace(
                    check_output=lambda *args, **kwargs: str(common_root / ".git").encode("utf-8"),
                    DEVNULL=object(),
                ),
                create=True,
            ), patch.object(builder.Path, "cwd", return_value=worktree_root):
                resolved = builder.resolve_under_cwd(".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json")

        self.assertEqual(resolved, dataset)

    def test_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "utils/make_sure_sorry_csd_llada.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--positive_response", result.stdout)


if __name__ == "__main__":
    unittest.main()
