"""scripts/watch-zig.sh: idle-only probes and the bounded startup-503 retry, against fake curl / docker / ssh."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "watch-zig.sh"
FAKE_CURL = """#!/usr/bin/env bash
n=$(cat "$FAKE/n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$FAKE/n"
echo "$*" >> "$FAKE/calls"
f="$FAKE/resp.$n"; [[ -f "$f" ]] || f="$FAKE/resp.last"
cat "$f"
"""
FAKE_DOCKER = """#!/usr/bin/env bash
case "$*" in *State.Running*) echo true ;; *StartedAt*) date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ ;; *) echo 0 ;; esac
"""
FAKE_SSH = "#!/usr/bin/env bash\necho true\n"
STALE = {"ok": False, "errors": 0, "status": "stalled", "stall_reason": "no round for 300 s with work in flight",
         "oldest_s": 1.5, "rounds": {"last_age_s": 300.0}, "inflight": 1}
HEALTHY = {"ok": True, "errors": 0, "requests_running": 0, "streams": {"max": 4}}


class WatchZigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fake = self.root / "fake"
        self.bin = self.root / "bin"
        self.state = self.root / "state"
        for path in (self.fake, self.bin, self.state):
            path.mkdir()
        for name, body in (("curl", FAKE_CURL), ("docker", FAKE_DOCKER), ("ssh", FAKE_SSH)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)

    def respond(self, *replies, last=None):
        for index, (body, code) in enumerate(replies, 1):
            (self.fake / f"resp.{index}").write_text(f"{body}\n{code}")
        body, code = last or replies[-1]
        (self.fake / "resp.last").write_text(f"{body}\n{code}")

    def calls(self):
        path = self.fake / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def bash(self, snippet, **env):
        environment = {
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE": str(self.fake),
            "CONFIG": str(self.root / "missing.env"), "WATCH_STATE_DIR": str(self.state), "WORKER_SSH": "rank1",
            "WATCH_PROBE_EVERY": "0", **env,
        }
        return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; {snippet}'], env=environment,
                              capture_output=True, text=True, timeout=30)

    def stale_check(self, code, value):
        return self.bash(f"startup_stale_idle {code} '{json.dumps(value)}'").returncode

    def test_only_the_exact_young_request_shape_is_a_startup_false_stall(self):
        self.assertEqual(self.stale_check(503, STALE), 0)
        for change in ({"errors": 1}, {"oldest_s": 4}, {"rounds": {"last_age_s": 100}}, {"inflight": 0},
                       {"stall_reason": "slow round"}, {"fatal": "x"}, {"ok": True}):
            with self.subTest(change=change):
                self.assertEqual(self.stale_check(503, {**STALE, **change}), 1)
        self.assertEqual(self.stale_check(200, STALE), 1)

    def test_startup_retry_accepts_a_real_healthy_recheck(self):
        self.respond((json.dumps(HEALTHY), 200))
        result = self.bash(f"startup_retry 503 '{json.dumps(STALE)}'")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_startup_retry_is_bounded_when_the_stall_persists(self):
        self.respond((json.dumps(STALE), 503))
        started = time.monotonic()
        result = self.bash(f"startup_retry 503 '{json.dumps(STALE)}'")
        self.assertLess(time.monotonic() - started, 6)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(result.stdout.startswith("503\n"))

    def test_guard_deferral_is_not_a_probe_failure(self):
        self.respond(('{"code":"watchdog_deferred"}', 429))
        for _ in range(4):
            self.assertEqual(self.bash(f"probe_tick $(date +%s) '{json.dumps(HEALTHY)}'").returncode, 0)
        self.assertFalse((self.state / "probe_fails").exists())
        self.assertTrue(all("X-TF-Probe: watchdog" in call for call in self.calls()))

    def test_busy_slots_skip_the_probe_without_a_request(self):
        busy = {**HEALTHY, "requests_running": 4}
        self.assertEqual(self.bash(f"probe_tick $(date +%s) '{json.dumps(busy)}'").returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_probe_cadence_is_kept(self):
        self.respond(('{"choices":[]}', 200))
        (self.state / "probe_last").write_text(str(int(time.time())))
        self.bash(f"probe_tick $(date +%s) '{json.dumps(HEALTHY)}'", WATCH_PROBE_EVERY="180")
        self.assertEqual(self.calls(), [])

    def test_consecutive_probe_failures_heal_only_at_the_limit(self):
        self.respond(("", "000"))
        codes = [self.bash(f"probe_tick $(date +%s) '{json.dumps(HEALTHY)}'").returncode for _ in range(3)]
        self.assertEqual(codes, [0, 0, 1])

    def test_tick_resolved_false_stall_is_healthy(self):
        self.respond((json.dumps(STALE), 503), (json.dumps(HEALTHY), 200), ('{"choices":[]}', 200))
        result = self.bash("watch_tick")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resolved by actual healthy 200", result.stderr)
        self.assertEqual((self.state / "fails").read_text().strip(), "0")

    def test_tick_unresolved_stall_is_bad_on_the_first_tick_without_healing_by_default(self):
        self.respond((json.dumps(STALE), 503))
        result = self.bash("watch_tick")
        self.assertEqual(result.returncode, 1)
        self.assertIn("bad tick 1 of 1", result.stderr)
        self.assertIn("not restarting", result.stderr)

    def test_stop_marker_stands_down(self):
        (self.state / "stopped").touch()
        result = self.bash("watch_tick")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
