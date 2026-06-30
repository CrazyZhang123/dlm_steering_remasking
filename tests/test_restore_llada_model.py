import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.restore_llada_model as restore


def write_minimal_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "layer_a": "model-00001-of-00002.safetensors",
                    "layer_b": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "model-00001-of-00002.safetensors").write_bytes(b"1")
    (path / "model-00002-of-00002.safetensors").write_bytes(b"2")


class RestoreLLaDAModelTest(unittest.TestCase):
    def test_is_model_complete_requires_indexed_weight_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "LLaDA-8B-Instruct"
            write_minimal_model_dir(model_dir)

            self.assertTrue(restore.is_model_complete(model_dir))

            (model_dir / "model-00002-of-00002.safetensors").unlink()

            self.assertFalse(restore.is_model_complete(model_dir))

    def test_restore_model_skips_download_when_target_is_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "LLaDA-8B-Instruct"
            write_minimal_model_dir(model_dir)

            with patch.object(restore, "snapshot_download") as snapshot_download:
                result = restore.restore_model(local_dir=model_dir)

            snapshot_download.assert_not_called()
            self.assertEqual(result.status, "already_present")
            self.assertEqual(result.path, model_dir)

    def test_restore_model_downloads_with_modelscope_when_target_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "LLaDA-8B-Instruct"
            model_dir.mkdir()

            with patch.object(restore, "snapshot_download", return_value=str(model_dir)) as snapshot_download:
                result = restore.restore_model(
                    model_id="GSAI-ML/LLaDA-8B-Instruct",
                    local_dir=model_dir,
                    max_workers=3,
                )

            snapshot_download.assert_called_once_with(
                "GSAI-ML/LLaDA-8B-Instruct",
                local_dir=str(model_dir),
                max_workers=3,
            )
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.path, model_dir)


if __name__ == "__main__":
    unittest.main()
