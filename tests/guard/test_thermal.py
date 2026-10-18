"""CPU-only durable-policy and actual loopback HTTP transport contracts."""

import concurrent.futures
import http.client
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import CLOCK_PROGRAM, Backend
from control import wait_control
from daemon import Manager
from guard_fixtures import Native, TransportFixture, backend_config, synthetic_sha
from lifecycle import decision
from policy import Policy

CURRENT, NEXT = synthetic_sha("current"), synthetic_sha("next")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = Policy(self.root / "state", self.root / "events", initial_profile=CURRENT, now=1000)
        self.now = 1000

    def sample(self, a=80, b=80, memory=8, owned=True, missing=False, step=1):
        self.now += step
        rows = {str(i): dict(epoch=self.now, max_zone_c=t, available_gib=memory) for i, t in enumerate((a, b))}
        if missing:
            rows.pop("1")
        self.p.update(rows, owned, "owned" if owned else "operator_lease", self.now)

    def pause(self):
        self.sample(92)
        self.sample(88, 88)

    def sustain(self):
        for _ in range(602):
            self.sample(90)

    def test_pause_and_both_resume(self):
        self.sample(92)
        self.assertTrue(self.p.data["paused"])
        self.sample(88, 88.1)
        self.assertTrue(self.p.data["paused"])
        self.sample(88, 88)
        self.assertTrue(self.p.ready())

    def test_worker_missing_fail_closed(self):
        self.sample(missing=True)
        self.assertFalse(self.p.ready())
        self.sample()
        self.assertTrue(self.p.ready())

    def test_three_pause_sustain_then_fourth_floor_stops(self):
        for _ in range(3):
            self.pause()
        self.sustain()
        self.assertTrue(self.p.data["transitioning"])
        self.p.clock_changed({}, self.now)
        self.sample(88, 88)
        self.sample(92)
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertEqual(len(self.p.data["global_pauses"]), 4)

    def test_storm_consumed_once_fifth_new_pause_stops(self):
        for _ in range(4):
            self.pause()
        self.p.clock_changed({}, self.now)
        self.sample(88, 88)
        self.assertFalse(self.p.data["terminal_stop"])
        self.sample(92)
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertEqual(len(self.p.data["global_pauses"]), 5)

    def test_persistence_never_resets_ring_or_stop(self):
        for _ in range(4):
            self.pause()
        other = Policy(self.root / "state", self.root / "events")
        self.assertEqual(other.data, self.p.data)
        self.p.clock_changed({}, self.now)
        self.sample(88, 88)
        self.sample(92)
        self.assertTrue(Policy(self.root / "state", self.root / "events").data["terminal_stop"])

    def test_sustain_requires_continuous_fresh_600_seconds(self):
        for _ in range(599):
            self.sample(90)
        self.assertFalse(self.p.data["transitioning"])
        self.sample(90, step=7)
        self.assertFalse(self.p.data["transitioning"])
        self.sustain()
        self.assertTrue(self.p.data["transitioning"])

    def test_hard_stop_abort_and_memory_on_single_fresh_node(self):
        sock = SimpleNamespace(shutdown=lambda *_: setattr(self, "aborted", True))
        self.p.active["a"] = sock
        self.sample(94, missing=True)
        self.assertTrue(self.aborted)
        self.assertEqual(self.p.data["pending_action"], "stop_owned")
        self.assertTrue(self.p.data["hard_stop"])

    def test_memory_floor(self):
        self.sample(memory=3.99)
        self.assertTrue(self.p.data["hard_stop"])

    def test_stop_evidence_durable_before_abort_and_immutable_after_reload(self):
        metadata = {"request-a": {"epoch": 1000, "body_sha256": "abc", "prompt_tokens": None,
                                  "resumed_tokens": None, "token_source": None}}
        self.p.request_metadata = metadata
        self.p.inflight = 1
        cancelled = []

        def shutdown(_):
            cancelled.append(json.loads((self.root / "state").read_text()))
            metadata["request-a"]["prompt_tokens"] = 123

        self.p.active["request-a"] = SimpleNamespace(shutdown=shutdown)
        rows = {
            "0": dict(epoch=1001, max_zone_c="94", available_gib="8"),
            "1": dict(epoch=1000, max_zone_c=80, available_gib=9),
        }
        self.p.update(rows, True, "owned", 1001)
        evidence = cancelled[0]["stop_evidence"]
        self.assertTrue(cancelled[0]["hard_stop"])
        self.assertEqual(cancelled[0]["pending_action"], "stop_owned")
        self.assertEqual(evidence["epoch"], 1001)
        self.assertEqual(evidence["inflight"], 1)
        self.assertEqual(evidence["raw_samples"], rows)
        self.assertEqual(evidence["validated_samples"]["0"]["max_zone_c"], 94.0)
        self.assertIsNone(evidence["request_metadata"]["request-a"]["prompt_tokens"])
        events = [json.loads(line) for line in (self.root / "events").read_text().splitlines()]
        self.assertEqual(events[-1]["stop_evidence"], evidence)
        rows["0"]["max_zone_c"] = 100
        self.p.update(rows, True, "owned", 1002)
        reboot = Policy(self.root / "state", self.root / "events")
        self.assertEqual(reboot.request_metadata, {})
        reboot.update({"1": dict(epoch=1003, max_zone_c=99, available_gib=1)}, True, "owned", 1003)
        self.assertEqual(reboot.data["stop_evidence"], evidence)
        self.assertEqual(json.loads((self.root / "state").read_text())["stop_evidence"], evidence)
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(len((self.root / "events").read_text().splitlines()), 1)

    def test_stop_evidence_distinguishes_conditions_and_ranks(self):
        for temperatures, memories, expected in (
            ((94, 80), (4, 8), [("temperature>=94", "0", 94)]),
            ((80, 80), (8, 3.99), [("memory<4GiB", "1", 3.99)]),
            ((95, 94), (3, 2), [("temperature>=94", "0", 95), ("memory<4GiB", "0", 3),
                                ("temperature>=94", "1", 94), ("memory<4GiB", "1", 2)]),
        ):
            with self.subTest(expected=expected):
                root = self.root / str(len(expected)) / str(temperatures)
                policy = Policy(root / "state", root / "events", now=1000)
                rows = {str(i): dict(epoch=1001, max_zone_c=temperatures[i], available_gib=memories[i])
                        for i in range(2)}
                policy.update(rows, True, "owned", 1001)
                evidence = policy.data["stop_evidence"]
                self.assertEqual([(row["condition"], row["rank"], row["value"]) for row in evidence["fired"]],
                                 expected)
                self.assertEqual(evidence["request_metadata"], {})

    def test_stop_evidence_excludes_invalid_samples_from_triggers(self):
        rows = {
            "0": dict(epoch=1001, max_zone_c=94, available_gib=8),
            "1": dict(epoch=990, max_zone_c=float("nan"), available_gib=1),
        }
        self.p.update(rows, True, "owned", 1001)
        evidence = self.p.data["stop_evidence"]
        self.assertEqual(evidence["raw_samples"]["1"]["max_zone_c"], "NaN")
        self.assertEqual(set(evidence["validated_samples"]), {"0"})
        self.assertEqual(evidence["fired"], [{"condition": "temperature>=94", "rank": "0", "value": 94}])

    def test_hard_stop_threshold_boundaries_and_ownership_unchanged(self):
        self.sample(93.99, 93.99, memory=4)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertNotIn("stop_evidence", self.p.data)
        self.sample(100, memory=1, owned=False)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertNotIn("stop_evidence", self.p.data)

    def test_operator_lease_never_stops_and_never_admits(self):
        self.sample(100, memory=1, owned=False)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertEqual(self.p.data["lease_mode"], "operator_lease")
        self.assertFalse(self.p.ready())

    def test_lifecycle_stop_and_explicit_restore(self):
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (CURRENT, {}))
        state = dict(version=1, profile=CURRENT, hard_stop=False, fresh=True, last_update_epoch=1000,
                     terminal_stop=True, paused=True)
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 78)
        state["bootstrap_profile"] = CURRENT
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 0)
        self.assertEqual(decision({}, state, "watch", backend, 1000)[0], 78)
        state["hard_stop"] = True
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 78)

    def test_boot_proof_current_logs_floor_and_pool_failure(self):
        backend = Backend(backend_config())
        backend.identity = lambda *_: {"id": "exact"}
        backend.idle = lambda: {"streams": {"max": 4}}
        backend.run = lambda *_: "boot growth worst serving point 4.00 GiB"
        self.assertEqual(len(backend.boot_proof("sha")["ranks"]), 2)
        for logs in ("boot growth worst serving point 3.99 GiB",
                     "pool does not fit\nboot growth worst serving point 5 GiB"):
            backend.run = lambda *_, text=logs: text
            with self.assertRaises(ValueError):
                backend.boot_proof("sha")

    def test_boot_proof_requires_native_slots_equal_guard_inflight(self):
        backend = Backend(backend_config(limits={"max_inflight": 2}))
        backend.identity = lambda *_: {"id": "exact"}
        backend.run = lambda *_: "boot growth worst serving point 6.00 GiB"
        for streams in ({"max": 4}, {"max": 1}, {}, None):
            backend.idle = lambda value=streams: {} if value is None else {"streams": value}
            with self.subTest(streams=streams), self.assertRaisesRegex(ValueError, "max_inflight"):
                backend.boot_proof("sha")
        backend.idle = lambda: {"streams": {"max": 2}}
        self.assertEqual(len(backend.boot_proof("sha")["ranks"]), 2)

    def test_control_waits_only_known_hold_and_never_retries_stop(self):
        ticks = iter(range(10))
        hold = {"ok": False, "error": "operator transition requires both fresh<=88 and no local work"}
        with patch("control.transaction", side_effect=[hold, {"ok": True}]) as transaction:
            result = wait_control({}, "prepare-profile", NEXT, 10, lambda: next(ticks), lambda _: None)
            self.assertTrue(result["ok"])
            self.assertEqual(transaction.call_count, 2)
        with patch("control.transaction",
                   return_value={"ok": False, "error": "same-profile STOP cannot be reset"}) as call:
            self.assertFalse(wait_control({}, "prepare-profile", CURRENT)["ok"])
            self.assertEqual(call.call_count, 1)

    def test_clock_boot_unavailable_hold_then_verified_without_stop_reset(self):
        backend = SimpleNamespace(actual_clock=lambda: (_ for _ in ()).throw(OSError("booting")))
        manager = object.__new__(Manager)
        manager.backend, manager.policy = backend, self.p
        manager.check_clock()
        self.assertFalse(self.p.data["clock_verified"])
        self.assertFalse(self.p.data["terminal_stop"])
        backend.actual_clock = lambda: 2100
        manager.check_clock()
        self.assertTrue(self.p.data["clock_verified"])
        with self.p.lock:
            self.p.terminal("existing", self.now)
        manager.check_clock()
        self.assertTrue(self.p.data["terminal_stop"])
        backend.actual_clock = lambda: 2000
        manager.check_clock()
        self.assertFalse(self.p.data["clock_verified"])

    def test_clock_mismatch_at_boot_is_a_terminal_stop_not_an_upclock(self):
        manager = object.__new__(Manager)
        manager.backend, manager.policy = SimpleNamespace(actual_clock=lambda: 2000), self.p
        manager.check_clock()
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertFalse(self.p.data["clock_verified"])
        self.assertEqual(self.p.data["clock_mhz"], 2100)

    def test_profile_bootstrap_archives_stop_keeps_global_ring_and_boot_proof(self):
        self.sample(92)
        self.sample(88, 88)
        with self.p.lock:
            self.p.terminal("current final", self.now)
        manager = object.__new__(Manager)
        manager.policy = self.p
        manager.config = {"profiles": {CURRENT: {}, NEXT: {}}, "state_path": str(self.root / "state")}
        manager.backend = SimpleNamespace(profile=lambda: (NEXT, {}), actual_clock=lambda: 2100,
                                          identity=lambda rank, _: {"id": rank}, idle=lambda: {"inflight": 0},
                                          boot_proof=lambda _: {"real_boot": True})
        manager.cool_idle_operator = lambda *_: {"idle": True}
        before = list(self.p.data["global_pauses"])
        manager.control("prepare-profile", NEXT)
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertEqual(self.p.data["bootstrap_profile"], NEXT)
        manager.control("authorize-profile", NEXT)
        self.assertFalse(self.p.data["terminal_stop"])
        self.assertEqual(self.p.data["global_pauses"], before)
        archives = list((self.root / "archives").glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertTrue(json.loads(archives[0].read_text())["terminal_stop"])
        with self.p.lock:
            self.p.terminal("next final", self.now)
        with self.assertRaisesRegex(ValueError, "same-profile STOP"):
            manager.control("prepare-profile", NEXT)
        with self.assertRaisesRegex(ValueError, "unapproved target"):
            manager.control("prepare-profile", synthetic_sha("unregistered"))

    def test_absent_lowercase_docker_objects_allow_bootstrap_but_running_refused(self):
        self.sample()
        manager = object.__new__(Manager)
        manager.policy = self.p
        manager.config = {"containers": {"0": "native-r0", "1": "native-r1"}}
        manager.backend = SimpleNamespace(
            ownership=lambda **_: (True, "owned"),
            idle=lambda: (_ for _ in ()).throw(OSError("backend absent")),
            run=lambda *_: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, "docker inspect", stderr="error: no such object: native-r0\n")),
        )
        with patch("daemon.time.time", return_value=self.now):
            self.assertTrue(manager.cool_idle_operator()["native_absent_and_both_containers_stopped"])
            manager.backend.run = lambda *_: json.dumps([{"State": {"Running": True}}])
            with self.assertRaisesRegex(ValueError, "running native"):
                manager.cool_idle_operator()

    def test_pending_healthy_bootstrap_start_allowed_watch_held(self):
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (CURRENT, {}))
        state = dict(version=1, profile=CURRENT, bootstrap_profile=CURRENT, hard_stop=False, terminal_stop=False,
                     paused=False, fresh=True, last_update_epoch=1000)
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 0)
        self.assertEqual(decision({}, state, "watch", backend, 1000)[0], 75)

    def test_clock_program_compiles(self):
        compile(CLOCK_PROGRAM, "<clock>", "exec")

    def test_docker_logs_includes_stderr_only_for_logs(self):
        backend = Backend(backend_config())
        with patch("backend.subprocess.run", return_value=SimpleNamespace(stdout="out", stderr="err")):
            self.assertEqual(backend.run("0", ["docker", "logs", "x"]), "outerr")
            self.assertEqual(backend.run("0", ["docker", "inspect", "x"]), "out")


class TransportTests(TransportFixture, unittest.TestCase):
    def test_four_parallel_sse_and_unchanged_auth_body(self):
        # Hold native decode until all four arrive; wall-clock sleeps can undercount overlap.
        Native.finish_barrier = threading.Barrier(4)
        self.addCleanup(setattr, Native, "finish_barrier", None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request(), range(4)))
        self.assertEqual(Native.maximum, 4)
        self.assertEqual(results, [(200, Native.payload)] * 4)
        self.assertEqual(Native.received, [("Bearer exact", b'{"image":"raw","tools":[1]}')] * 4)
        self.wait_idle()
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(len(self.p.active), 0)

    def test_health_passes_paused_and_terminal_blocks_generation(self):
        self.p.data["paused"] = True
        self.assertEqual(self.request("GET"), (200, b"{}"))
        self.p.data["terminal_stop"] = True
        self.assertEqual(self.request()[0], 503)
        self.assertFalse(Native.received)

    def test_oversize_not_forwarded(self):
        self.assertEqual(self.request(body=b"x" * 1025)[0], 502)
        self.assertFalse(Native.received)

    def test_status_endpoint_reports_and_no_http_mutation_api(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=2)
        try:
            connection.request("GET", "/__thermal/status")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["clock_ladder_mhz"], [2100, 2000])
            connection.request("POST", "/__thermal/reset", b"{}")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
        finally:
            connection.close()
        self.assertFalse(Native.received)


if __name__ == "__main__":
    unittest.main()
