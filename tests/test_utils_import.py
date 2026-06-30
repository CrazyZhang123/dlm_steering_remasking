import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class UtilsImportTest(unittest.TestCase):
    def test_utils_import_exposes_template_constants(self):
        import utils

        self.assertTrue(hasattr(utils, "REFERENCES"))
        self.assertTrue(hasattr(utils, "SAFE_REMINDER"))


if __name__ == "__main__":
    unittest.main()
