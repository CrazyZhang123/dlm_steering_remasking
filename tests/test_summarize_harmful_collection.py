import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_harmful_collection import summarize_collection


class SummarizeCollectionTest(unittest.TestCase):
    def test_summarize_collection_counts_rounds_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs = root / "outputs"
            outputs.mkdir()
            round1 = outputs / "wildjailbreak_round_01_gpu0"
            round2 = outputs / "wildjailbreak_round_01_gpu1"
            ignored = outputs / "LLaDA_JBB_local_100"
            round1.mkdir()
            round2.mkdir()
            ignored.mkdir()

            (round1 / "results.json").write_text(
                json.dumps([{"prompt": "a", "response": "ra"}, {"prompt": "b", "response": "rb"}]),
                encoding="utf-8",
            )
            (round1 / "llama_guard_local.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_samples": 2, "unsafe_count": 1},
                        "results": [],
                    }
                ),
                encoding="utf-8",
            )
            (round2 / "results.json").write_text(
                json.dumps([{"prompt": "c", "response": "rc"}]),
                encoding="utf-8",
            )
            (ignored / "results.json").write_text(
                json.dumps([{"prompt": "ignore", "response": "ignore"}]),
                encoding="utf-8",
            )

            pairs_path = root / "data" / "csd_llada_harmful_pairs_ext_all.json"
            pairs_path.parent.mkdir()
            pairs_path.write_text(
                json.dumps([{"prompt": "a", "response": "ra"}]),
                encoding="utf-8",
            )
            all_prompts = root / "data" / "wildjailbreak_train_prompts_all.csv"
            all_prompts.write_text("prompt\na\nb\nc\nd\n", encoding="utf-8")

            summary = summarize_collection(root)

        self.assertEqual(summary["completed_result_files"], 2)
        self.assertEqual(summary["completed_samples"], 3)
        self.assertEqual(summary["judged_result_files"], 1)
        self.assertEqual(summary["judged_samples"], 2)
        self.assertEqual(summary["unsafe_count"], 1)
        self.assertEqual(summary["external_judged_result_files"], 0)
        self.assertEqual(summary["external_judged_samples"], 0)
        self.assertEqual(summary["external_unsafe_count"], 0)
        self.assertEqual(summary["harmful_pairs"], 1)
        self.assertEqual(summary["remaining_to_5763"], 5762)
        self.assertEqual(summary["external_unsafe_rate_over_judged"], 0.0)
        self.assertIsNone(summary["estimated_prompts_needed_at_current_external_rate"])
        self.assertEqual(summary["available_prompt_pool"], 4)
        self.assertIsNone(summary["projected_harmful_pairs_at_current_external_rate"])
        self.assertFalse(summary["pool_sufficient_at_current_external_rate"])


if __name__ == "__main__":
    unittest.main()
