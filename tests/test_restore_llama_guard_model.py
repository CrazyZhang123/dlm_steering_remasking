import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.restore_llama_guard_model as restore_guard


class RestoreLlamaGuardModelTest(unittest.TestCase):
    def test_restore_guard_uses_modelscope_llama_guard_defaults(self):
        with patch.object(restore_guard, "restore_model") as restore_model:
            restore_model.return_value.status = "downloaded"
            restore_model.return_value.path = Path("/dev/shm/Llama-Guard-4-12B")

            exit_code = restore_guard.main([])

        restore_model.assert_called_once_with(
            model_id="LLM-Research/Llama-Guard-4-12B",
            local_dir=Path("/dev/shm/Llama-Guard-4-12B"),
            max_workers=4,
            force=False,
        )
        self.assertEqual(exit_code, 0)

    def test_restore_guard_allows_overriding_download_options(self):
        with patch.object(restore_guard, "restore_model") as restore_model:
            restore_model.return_value.status = "already_present"
            restore_model.return_value.path = Path("/tmp/guard")

            exit_code = restore_guard.main(
                [
                    "--model_id",
                    "custom/guard",
                    "--local_dir",
                    "/tmp/guard",
                    "--max_workers",
                    "2",
                    "--force",
                ]
            )

        restore_model.assert_called_once_with(
            model_id="custom/guard",
            local_dir=Path("/tmp/guard"),
            max_workers=2,
            force=True,
        )
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
