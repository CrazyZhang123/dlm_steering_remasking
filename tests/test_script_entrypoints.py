import shlex
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointsTest(unittest.TestCase):
    def test_llada_steer_script_python_entrypoint_exists(self):
        script_path = ROOT / "scripts" / "llada_steer.sh"
        first_line = script_path.read_text(encoding="utf-8").splitlines()[0].rstrip("\\").strip()
        parts = shlex.split(first_line)

        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual(parts[0], "python")
        entrypoint = ROOT / parts[1]
        self.assertTrue(
            entrypoint.exists(),
            f"script entrypoint does not exist: {entrypoint}",
        )


if __name__ == "__main__":
    unittest.main()
