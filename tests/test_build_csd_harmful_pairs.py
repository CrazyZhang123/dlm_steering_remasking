import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_csd_harmful_pairs import merge_unsafe_pairs, main


class MergeUnsafePairsTest(unittest.TestCase):
    def test_merge_unsafe_pairs_dedups_by_prompt(self):
        round_items = [
            [
                {"prompt": "a", "response": "harmful-a", "unsafe": True},
                {"prompt": "b", "response": "safe-b", "unsafe": False},
            ],
            [
                {"prompt": "a", "response": "harmful-a-2", "unsafe": True},
                {"prompt": "c", "response": "harmful-c", "unsafe": True},
            ],
        ]

        merged = merge_unsafe_pairs(round_items)

        self.assertEqual(
            merged,
            [
                {"prompt": "a", "response": "harmful-a"},
                {"prompt": "c", "response": "harmful-c"},
            ],
        )

    def test_merge_unsafe_pairs_skips_empty_prompt_or_response(self):
        round_items = [[
            {"prompt": "", "response": "harmful-a", "unsafe": True},
            {"prompt": "b", "response": "", "unsafe": True},
            {"prompt": "c", "response": "harmful-c", "unsafe": True},
        ]]

        merged = merge_unsafe_pairs(round_items)

        self.assertEqual(merged, [{"prompt": "c", "response": "harmful-c"}])


class MainTest(unittest.TestCase):
    def test_main_reads_round_directories_and_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            round1 = tmp / "round1"
            round2 = tmp / "round2"
            round1.mkdir()
            round2.mkdir()
            output = tmp / "harmful.json"

            (round1 / "llama_guard_local.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "results": [
                            {"prompt": "a", "response": "harmful-a", "unsafe": True},
                            {"prompt": "b", "response": "safe-b", "unsafe": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (round2 / "llama_guard_local.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "results": [
                            {"prompt": "c", "response": "harmful-c", "unsafe": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            main(
                [
                    "--round_dir",
                    str(round1),
                    "--round_dir",
                    str(round2),
                    "--output",
                    str(output),
                ]
            )

            merged = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            merged,
            [
                {"prompt": "a", "response": "harmful-a"},
                {"prompt": "c", "response": "harmful-c"},
            ],
        )

    def test_main_supports_custom_judge_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            round1 = tmp / "round1"
            round1.mkdir()
            output = tmp / "harmful.json"

            (round1 / "external_judge.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "results": [
                            {"prompt": "x", "response": "harmful-x", "unsafe": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            main(
                [
                    "--round_dir",
                    str(round1),
                    "--judge_json_name",
                    "external_judge.json",
                    "--output",
                    str(output),
                ]
            )

            merged = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(merged, [{"prompt": "x", "response": "harmful-x"}])


if __name__ == "__main__":
    unittest.main()
