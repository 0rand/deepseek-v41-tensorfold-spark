"""dsv41_aot_merge.py: adds only the kernels the destination lacks, cubins included, and is idempotent."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "dsv41_aot_merge.py"


def aot(root, hashes):
    (root / "cubins").mkdir(parents=True)
    for h in hashes:
        (root / "cubins" / f"{h}.cubin").write_bytes(h.encode())
    (root / "aot.json").write_text(json.dumps({"generator": "test", "kernels": [{"hash": h} for h in hashes]}))


class Merge(unittest.TestCase):
    def test_merge_adds_missing_once(self):
        with tempfile.TemporaryDirectory() as d:
            dst, src = Path(d, "dst"), Path(d, "src")
            aot(dst, ["a", "b"])
            aot(src, ["b", "c"])
            for added in (1, 0):
                r = subprocess.run([sys.executable, "-I", str(TOOL), str(dst), str(src)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(json.loads(r.stdout), {"added": added, "kernels": 3})
            kernels = json.loads((dst / "aot.json").read_text())["kernels"]
            self.assertEqual([k["hash"] for k in kernels], ["a", "b", "c"])
            self.assertEqual((dst / "cubins" / "c.cubin").read_bytes(), b"c")


if __name__ == "__main__":
    unittest.main()
