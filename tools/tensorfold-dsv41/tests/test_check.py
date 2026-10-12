"""check.py's import closure flags what it should: an unknown module, a broken file, a missing declared path."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("check", Path(__file__).resolve().parent.parent / "check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class Closure(unittest.TestCase):
    def closure(self, files, extra=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for rel, text in files.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            with patch.object(check, "HERE", root), patch.object(check, "EXTRA", extra or {}):
                return check.closure()

    def test_sibling_parent_root_and_optional_pass(self):
        bad = self.closure({"a.py": "import json\nimport b\nimport torch\n", "b.py": "",
                            "sub/c.py": "import a\nfrom d import x\n", "sub/d.py": "x = 1\n",
                            "sub/deep/e.py": "import d\nimport a\n"})
        self.assertEqual(bad, [])

    def test_unknown_import_and_syntax_error_fail(self):
        bad = self.closure({"a.py": "import not_here\n", "b.py": "def (:\n"})
        self.assertEqual(len(bad), 2)
        self.assertIn("a.py: import not_here", bad[0] + bad[1])
        self.assertIn("b.py: does not compile", bad[0] + bad[1])

    def test_declared_path_must_exist(self):
        files = {"a.py": "import helper\n", "lib/helper.py": ""}
        self.assertEqual(self.closure(files, {"a.py": {"helper": "lib/helper.py"}}), [])
        bad = self.closure(files, {"a.py": {"helper": "lib/missing.py"}})
        self.assertTrue(any("helper expected at lib/missing.py" in b for b in bad), bad)


if __name__ == "__main__":
    unittest.main()
