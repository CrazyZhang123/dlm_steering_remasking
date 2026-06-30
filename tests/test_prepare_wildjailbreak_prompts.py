import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.prepare_wildjailbreak_prompts as prepare


class NormalizePromptsTest(unittest.TestCase):
    def test_normalize_prompts_filters_harmful_prefers_adversarial_and_is_seeded(self):
        rows = [
            {
                "vanilla": "  vanilla harmful  ",
                "adversarial": "",
                "completion": "ignored",
                "data_type": "vanilla_harmful",
            },
            {
                "vanilla": "fallback harmful",
                "adversarial": " adversarial harmful ",
                "completion": "ignored",
                "data_type": "adversarial_harmful",
            },
            {
                "vanilla": "benign prompt",
                "adversarial": "benign adv",
                "completion": "ignored",
                "data_type": "adversarial_benign",
            },
            {
                "vanilla": "duplicate harmful",
                "adversarial": "",
                "completion": "ignored",
                "data_type": "vanilla_harmful",
            },
            {
                "vanilla": "duplicate harmful",
                "adversarial": "",
                "completion": "ignored",
                "data_type": "vanilla_harmful",
            },
            {
                "vanilla": "",
                "adversarial": "   ",
                "completion": "ignored",
                "data_type": "vanilla_harmful",
            },
        ]

        prompts = prepare.normalize_prompts(rows, limit=5, seed=42)

        self.assertEqual(
            prompts,
            ["adversarial harmful", "vanilla harmful", "duplicate harmful"],
        )

    def test_normalize_prompts_rejects_negative_limit(self):
        with self.assertRaises(ValueError):
            prepare.normalize_prompts([], limit=-1, seed=42)


class WritePromptCsvTest(unittest.TestCase):
    def test_write_prompt_csv_writes_prompt_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "prompts.csv"

            prepare.write_prompt_csv(output_path, ["first", "second"])

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [{"prompt": "first"}, {"prompt": "second"}])


class MainTest(unittest.TestCase):
    def test_main_prefers_load_from_disk_when_snapshot_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            snapshot_path = tmpdir_path / "snapshot"
            snapshot_path.mkdir()
            output_csv = tmpdir_path / "all.csv"
            smoke_csv = tmpdir_path / "smoke.csv"
            rows = [
                {
                    "vanilla": "one",
                    "adversarial": "",
                    "completion": "",
                    "data_type": "vanilla_harmful",
                },
                {
                    "vanilla": "two fallback",
                    "adversarial": "two attack",
                    "completion": "",
                    "data_type": "adversarial_harmful",
                },
                {
                    "vanilla": "one",
                    "adversarial": "",
                    "completion": "",
                    "data_type": "vanilla_harmful",
                },
                {
                    "vanilla": "three benign",
                    "adversarial": "",
                    "completion": "",
                    "data_type": "vanilla_benign",
                },
                {
                    "vanilla": "three harmful",
                    "adversarial": "",
                    "completion": "",
                    "data_type": "vanilla_harmful",
                },
            ]

            argv = [
                "prepare_wildjailbreak_prompts.py",
                "--save_disk",
                str(snapshot_path),
                "--output_csv",
                str(output_csv),
                "--output_smoke_csv",
                str(smoke_csv),
                "--limit",
                "3",
                "--smoke_count",
                "2",
                "--seed",
                "42",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(prepare, "load_from_disk", return_value=rows) as mock_load_from_disk:
                    with patch.object(prepare, "load_dataset") as mock_load_dataset:
                        prepare.main()

            self.assertEqual(mock_load_from_disk.call_count, 1)
            mock_load_dataset.assert_not_called()

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                output_rows = list(csv.DictReader(handle))
            with smoke_csv.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle))

        self.assertEqual(
            output_rows,
            [{"prompt": "two attack"}, {"prompt": "one"}, {"prompt": "three harmful"}],
        )
        self.assertEqual(smoke_rows, [{"prompt": "two attack"}, {"prompt": "one"}])

    def test_main_downloads_and_saves_snapshot_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            snapshot_path = tmpdir_path / "snapshot"
            output_csv = tmpdir_path / "all.csv"
            smoke_csv = tmpdir_path / "smoke.csv"

            class FakeDataset(list):
                def __init__(self, rows):
                    super().__init__(rows)
                    self.saved_paths = []

                def save_to_disk(self, path):
                    self.saved_paths.append(path)

            dataset = FakeDataset(
                [
                    {
                        "vanilla": "alpha",
                        "adversarial": "",
                        "completion": "",
                        "data_type": "vanilla_harmful",
                    },
                    {
                        "vanilla": "beta fallback",
                        "adversarial": "beta attack",
                        "completion": "",
                        "data_type": "adversarial_harmful",
                    },
                    {
                        "vanilla": "gamma",
                        "adversarial": "",
                        "completion": "",
                        "data_type": "vanilla_benign",
                    },
                ]
            )
            argv = [
                "prepare_wildjailbreak_prompts.py",
                "--dataset_id",
                "allenai/wildjailbreak",
                "--subset",
                "train",
                "--save_disk",
                str(snapshot_path),
                "--output_csv",
                str(output_csv),
                "--output_smoke_csv",
                str(smoke_csv),
                "--limit",
                "2",
                "--smoke_count",
                "1",
                "--seed",
                "42",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(prepare, "load_dataset", return_value=dataset) as mock_load_dataset:
                    with patch.object(prepare, "load_from_disk") as mock_load_from_disk:
                        prepare.main()

            mock_load_from_disk.assert_not_called()
            mock_load_dataset.assert_called_once_with(
                "allenai/wildjailbreak",
                "train",
                delimiter="\t",
                keep_default_na=False,
            )
            self.assertEqual(dataset.saved_paths, [str(snapshot_path)])

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                output_rows = list(csv.DictReader(handle))
            with smoke_csv.open("r", encoding="utf-8", newline="") as handle:
                smoke_rows = list(csv.DictReader(handle))

        self.assertEqual(output_rows, [{"prompt": "beta attack"}, {"prompt": "alpha"}])
        self.assertEqual(smoke_rows, [{"prompt": "beta attack"}])

    def test_main_rejects_negative_limit_and_smoke_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            snapshot_path = tmpdir_path / "snapshot"
            output_csv = tmpdir_path / "all.csv"
            smoke_csv = tmpdir_path / "smoke.csv"

            bad_argv_cases = [
                (
                    [
                        "prepare_wildjailbreak_prompts.py",
                        "--save_disk",
                        str(snapshot_path),
                        "--output_csv",
                        str(output_csv),
                        "--output_smoke_csv",
                        str(smoke_csv),
                        "--limit",
                        "-1",
                    ],
                    "--limit",
                ),
                (
                    [
                        "prepare_wildjailbreak_prompts.py",
                        "--save_disk",
                        str(snapshot_path),
                        "--output_csv",
                        str(output_csv),
                        "--output_smoke_csv",
                        str(smoke_csv),
                        "--smoke_count",
                        "-1",
                    ],
                    "--smoke_count",
                ),
            ]

            for argv, expected_flag in bad_argv_cases:
                with self.subTest(flag=expected_flag):
                    with patch.object(sys, "argv", argv):
                        with patch.object(prepare, "load_dataset", return_value=[]):
                            with patch.object(prepare, "load_from_disk", return_value=[]):
                                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                                    with self.assertRaises(SystemExit) as raised:
                                        prepare.main()

                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(expected_flag, fake_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
