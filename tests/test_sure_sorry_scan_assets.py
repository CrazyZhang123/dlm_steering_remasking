import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SureSorryScanAssetsTest(unittest.TestCase):
    def test_run_script_exists_and_pins_required_flags(self):
        script_path = ROOT / "scripts" / "run_sure_sorry_csd_scan.sh"
        self.assertTrue(script_path.exists(), f"缺少脚本: {script_path}")

        script = script_path.read_text(encoding="utf-8")
        self.assertIn("outputs/jbb_dija_no_steering_r1", script)
        self.assertIn("--gen_batch_size 1", script)
        self.assertIn("--batch_size 8", script)
        self.assertIn("--steering_vector_path \"${VEC}\"", script)
        self.assertIn("git rev-parse --git-common-dir", script)
        self.assertIn("EVAL_ENTRY=", script)
        self.assertIn("JUDGE_ENTRY=", script)
        self.assertIn("cd \"${COMMON_ROOT}\" && \"${PY}\" \"${EVAL_ENTRY}\"", script)

    def test_log_template_exists_and_records_scope_guards(self):
        summary_path = ROOT / "docs" / "experiment_summary.md"
        archived_log_path = ROOT / "docs" / "archive" / "sure_sorry_csd_scan_log.md"

        self.assertTrue(summary_path.exists(), f"缺少总结果文档: {summary_path}")
        self.assertTrue(archived_log_path.exists(), f"缺少归档日志: {archived_log_path}")

        summary_text = summary_path.read_text(encoding="utf-8")
        archived_log_text = archived_log_path.read_text(encoding="utf-8")
        self.assertIn("# DLM Steering 实验总表", summary_text)
        self.assertIn("Sure/Sorry", summary_text)
        self.assertIn("# Sure/Sorry 单方向引导向量扫描运行日志", archived_log_text)


if __name__ == "__main__":
    unittest.main()
