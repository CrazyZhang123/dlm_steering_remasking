import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_llada_steering as llada_mod


class _FakeAccelerator:
    num_processes = 1
    device = "cpu"


class _FakeModel:
    def eval(self):
        return self

    def to(self, _device):
        return self


class ModelLoadingTest(unittest.TestCase):
    def test_llada_harness_uses_low_cpu_mem_loading(self):
        fake_model = _FakeModel()

        with patch.object(llada_mod.accelerate, "Accelerator", return_value=_FakeAccelerator()):
            with patch.object(llada_mod.AutoModel, "from_pretrained", return_value=fake_model) as model_loader:
                with patch.object(llada_mod.AutoTokenizer, "from_pretrained", return_value=object()):
                    llada_mod.LLaDAEvalHarness(
                        model_path="dummy-model",
                        generated_samples_path="./outputs/test",
                        batch_size=32,
                        mc_num=32,
                        device="cpu",
                    )

        self.assertTrue(model_loader.called)
        self.assertTrue(model_loader.call_args.kwargs["low_cpu_mem_usage"])


if __name__ == "__main__":
    unittest.main()
