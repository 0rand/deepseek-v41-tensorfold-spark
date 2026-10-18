"""The SASS diagnostic's own logic: kernel keys, per-arch parsing and verdicts, its source map, the shell refusals."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

K = Path(__file__).resolve().parent.parent / "kernels"


def load(name):
    spec = importlib.util.spec_from_file_location(name, K / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC, PS = load("sass_check"), load("py_sources")


def sass(arch, funcs):
    """A cuobjdump -sass listing: one arch header, then each function's instruction lines."""
    out = [f"\tarch = sm_{arch}"]
    for name, body in funcs.items():
        out.append(f"\t\tFunction : {name}")
        out += [f"        /*{i * 16:04x}*/  {ins} ;" for i, ins in enumerate(body)]
        out.append("        /* 0x000fc40000000000 */")
    return "\n".join(out) + "\n"


class SassCheck(unittest.TestCase):
    def test_key_drops_the_first_namespace(self):
        self.assertEqual(SC.key("_ZN4exl35kern1Ev"), "_ZN@5kern1Ev")
        self.assertEqual(SC.key("_ZN6dsv41x5kern1Ev"), "_ZN@5kern1Ev")
        self.assertEqual(SC.key("plain_kernel"), "plain_kernel")

    def run_check(self, ref, ours, sm="121"):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.sass"), Path(d, "b.sass")
            a.write_text(ref)
            b.write_text(ours)
            r = subprocess.run([sys.executable, "-I", str(K / "sass_check.py"), str(a), str(b), "--sm", sm],
                               capture_output=True, text=True)
            return r.returncode, r.stdout

    def test_equal_under_renamed_namespace(self):
        rc, out = self.run_check(sass(121, {"_ZN4exl31kEv": ["MOV R1, c[0x0][0x28]", "EXIT"]}),
                                 sass(121, {"_ZN5dsv411kEv": ["MOV R1, c[0x0][0x28]", "EXIT"]}))
        self.assertEqual(rc, 0, out)
        self.assertIn("1 SASS-equal, 0 differ", out)

    def test_differ_and_missing_fail(self):
        rc, out = self.run_check(sass(121, {"k": ["EXIT"]}), sass(121, {"k": ["NOP", "EXIT"]}))
        self.assertEqual(rc, 1)
        self.assertIn("SASS-DIFFER k", out)
        rc, out = self.run_check(sass(121, {"k": ["EXIT"]}), sass(121, {"j": ["EXIT"]}))
        self.assertEqual(rc, 1)
        self.assertIn("ONLY-ZIG j", out)

    def test_other_arch_ignored_and_empty_fails(self):
        rc, out = self.run_check(sass(121, {"k": ["EXIT"]}), sass(90, {"k": ["NOP"]}))
        self.assertEqual(rc, 1, "no kernel of ours for sm_121 is a failure")
        self.assertIn("0 kernels of ours", out)


class PySources(unittest.TestCase):
    def env(self, **kv):
        return {k: v for k, v in os.environ.items() if not k.startswith("TF_DSV41_")} | kv

    def run_sources(self, **kv):
        return subprocess.run([sys.executable, "-I", str(K / "py_sources.py"), "--sources"], env=self.env(**kv),
                              capture_output=True, text=True)

    def test_repo_is_required(self):
        r = self.run_sources()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("TF_DSV41_REF_SRC", r.stderr)

    def test_keys_fall_back_to_base(self):
        r = self.run_sources(TF_DSV41_REF_SRC="/ref", TF_DSV41_REF_SRC_R1="/r1", TF_DSV41_REF_REV_R1="abc")
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = {line.split()[0]: line.split()[1:] for line in r.stdout.splitlines()}
        self.assertEqual(set(rows), set(PS.COPIES))
        self.assertEqual(rows["x3ld.cu"], ["base", "/ref", "HEAD"])
        self.assertEqual(rows["exl3_experts.cu"], ["r1", "/r1", "abc"])
        self.assertEqual(rows["x3gm.cu"], ["x3gm2", "/ref", "HEAD"])

    def test_shell_refuses_without_kernels_or_reference(self):
        sh = str(K / "sass_check.sh")
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["bash", sh], env=self.env(OUT=d), capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("TF_DSV41_KERNELS", r.stderr)
            r = subprocess.run(["bash", sh], env=self.env(OUT=d, TF_DSV41_KERNELS=d), capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("no DeepSeek-V4.1 kernel copies", r.stderr)
            Path(d, "x3ld.cu").write_text("")
            r = subprocess.run(["bash", sh], env=self.env(OUT=d, TF_DSV41_KERNELS=d), capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("TF_DSV41_REF_SRC", r.stderr)


if __name__ == "__main__":
    unittest.main()
