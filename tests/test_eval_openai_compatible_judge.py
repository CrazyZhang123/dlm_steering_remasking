import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.eval_openai_compatible_judge import (
    extract_json_array,
    is_unsafe_label,
    run_evaluation,
)


class ExtractJsonArrayTest(unittest.TestCase):
    def test_extracts_array_from_wrapped_text(self):
        text = "prefix\n[{\"id\": 0, \"label\": \"safe\", \"reason\": \"ok\"}]\nsuffix"
        parsed = extract_json_array(text)
        self.assertEqual(parsed, [{"id": 0, "label": "safe", "reason": "ok"}])


class IsUnsafeLabelTest(unittest.TestCase):
    def test_detects_unsafe(self):
        self.assertTrue(is_unsafe_label("unsafe"))
        self.assertFalse(is_unsafe_label("safe"))


class RunEvaluationTest(unittest.TestCase):
    def test_run_evaluation_writes_expected_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "results.json"
            out_path = tmp / "judge.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"prompt": "p1", "response": "r1"},
                        {"prompt": "p2", "response": "r2"},
                    ]
                ),
                encoding="utf-8",
            )

            fake_response = mock.Mock()
            fake_response.raise_for_status.return_value = None
            fake_response.json.return_value = {
                "usage": {"total_tokens": 123},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {"id": 0, "label": "safe", "reason": "refusal"},
                                    {"id": 1, "label": "unsafe", "reason": "harmful help"},
                                ]
                            )
                        }
                    }
                ],
            }

            with mock.patch("scripts.eval_openai_compatible_judge.requests.post", return_value=fake_response):
                payload = run_evaluation(
                    data_path=str(data_path),
                    output_path=str(out_path),
                    base_url="https://example.com/v1",
                    api_key="dummy",
                    model="gpt-5.4",
                    batch_size=10,
                    sleep_seconds=0,
                )

            self.assertEqual(payload["metadata"]["total_samples"], 2)
            self.assertEqual(payload["metadata"]["resolved_samples"], 2)
            self.assertEqual(payload["metadata"]["unsafe_count"], 1)
            self.assertEqual(payload["metadata"]["asr_percent"], 50.0)
            self.assertEqual(payload["results"][1]["unsafe"], True)
            self.assertEqual(payload["results"][1]["reason"], "harmful help")
            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), payload)

    def test_run_evaluation_resumes_from_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "results.json"
            out_path = tmp / "judge.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"prompt": "p1", "response": "r1"},
                        {"prompt": "p2", "response": "r2"},
                    ]
                ),
                encoding="utf-8",
            )
            out_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "provider": "OpenAI-compatible API",
                            "resolved_samples": 1,
                            "unsafe_count": 0,
                        },
                        "results": [
                            {
                                "id": 0,
                                "prompt": "p1",
                                "response": "r1",
                                "label": "safe",
                                "reason": "done",
                                "unsafe": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            fake_response = mock.Mock()
            fake_response.raise_for_status.return_value = None
            fake_response.json.return_value = {
                "usage": {"total_tokens": 55},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [{"id": 1, "label": "unsafe", "reason": "harmful help"}]
                            )
                        }
                    }
                ],
            }

            with mock.patch("scripts.eval_openai_compatible_judge.requests.post", return_value=fake_response) as post_mock:
                payload = run_evaluation(
                    data_path=str(data_path),
                    output_path=str(out_path),
                    base_url="https://example.com/v1",
                    api_key="dummy",
                    model="gpt-5.4",
                    batch_size=10,
                    sleep_seconds=0,
                )

            self.assertEqual(post_mock.call_count, 1)
            self.assertEqual(payload["metadata"]["resolved_samples"], 2)
            self.assertEqual(payload["metadata"]["unsafe_count"], 1)
            self.assertEqual([row["id"] for row in payload["results"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
