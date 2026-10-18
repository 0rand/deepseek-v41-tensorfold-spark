"""Staged clock ladder, backend cap steps, prefill admission and generated-token release contracts."""

import concurrent.futures
import copy
import http.client
import json
import threading
import time
import unittest
from pathlib import PosixPath
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import CLOCK_PROGRAM, Backend
from front import GeneratedTokenProbe
from guard_fixtures import LADDER3, LadderFixture, Native, TransportFixture, backend_config
from lifecycle import decision
from policy import Policy


def props(cap):
    return f"ExecStart=/usr/bin/nvidia-smi -lgc 300,{cap} ;\n" + (
        "ActiveState=active\nUnitFileState=enabled\nResult=success\nExecMainStatus=0\n")


class StabilityTests(LadderFixture, unittest.TestCase):
    def hot_pause(self, rows=("0", "1")):
        now = time.time()
        self.p.update({r: dict(epoch=now, max_zone_c=92, available_gib=8) for r in rows}, True, "owned", now)

    def test_step_down_walks_three_cap_ladder_keeps_global_history_then_floor_stops(self):
        for target in (2100, 2000):
            for _ in range(4 if target == 2100 else 1):
                self.hot_pause()
                self.samples()
            self.assertTrue(self.p.data["transitioning"])
            self.assertEqual(self.p.data["pending_action"], "lower_clock")
            ring = copy.deepcopy(self.p.data["global_pauses"])
            self.p.clock_changed([])
            self.assertEqual(self.p.data["clock_mhz"], target)
            self.assertEqual(self.p.data["global_pauses"], ring)
            self.p = self.reload()
            self.samples()
        self.hot_pause(rows=("0",))
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertFalse(self.p.data["hard_stop"])
        self.assertIsNone(self.p.data["pending_action"])
        self.assertEqual(self.p.data["clock_mhz"], 2000)
        with self.assertRaises(ValueError):
            self.p.clock_changed([])
        history = [row["event"] for row in self.p.data["history"]]
        self.assertEqual(history, ["clock_lowered", "clock_lowered"])

    def test_durable_ladder_mismatch_refused_without_reset(self):
        self.hot_pause()
        before = (self.root / "state").read_bytes()
        for ladder in ((2100, 2000), (2200, 2000), (2300, 2200, 2100, 2000), (2200, 2100, 2000, 1900)):
            with self.subTest(ladder=ladder), self.assertRaisesRegex(ValueError, "ladder differs"):
                Policy(self.root / "state", self.root / "events", ladder=ladder)
        self.assertEqual((self.root / "state").read_bytes(), before)
        self.assertEqual(self.reload().data["clock_ladder_mhz"], list(LADDER3))

    def test_new_state_starts_on_a_ladder_cap(self):
        root = self.root / "fresh"
        self.assertEqual(Policy(root / "a", root / "events", ladder=LADDER3).data["clock_mhz"], 2200)
        self.assertEqual(Policy(root / "b", root / "events", ladder=LADDER3, initial_clock=2100).data["clock_mhz"],
                         2100)
        with self.assertRaisesRegex(ValueError, "ladder cap"):
            Policy(root / "c", root / "events", ladder=LADDER3, initial_clock=2150)
        self.assertFalse((root / "c").exists())

    def test_lifecycle_top_cap_never_clears_hold_or_stop(self):
        self.samples()
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (self.sha, {}))
        self.assertEqual(decision({}, self.p.data, "watch", backend)[0], 0)
        self.p.data["transitioning"] = True
        self.assertEqual(decision({}, self.p.data, "watch", backend)[0], 75)
        with self.p.lock:
            self.p.terminal("test", time.time(), hard=True)
        self.assertEqual(decision({}, self.p.data, "start", backend)[0], 78)
        self.assertTrue(self.reload().data["hard_stop"])

    def test_backend_ladder_and_each_node_gate(self):
        config = backend_config(clock_ladder_mhz=list(LADDER3))
        backend = Backend(config)
        for source, target in ((2200, 2100), (2100, 2000)):
            with self.subTest(source=source):
                clocks = {"0": source, "1": source}
                gates = []

                def run(rank, args, **kwargs):
                    if args[0] == "systemctl":
                        self.assertEqual(args[2], config["clock_service"])
                        return props(clocks[rank])
                    self.assertEqual(args[:4], ["sudo", "-n", "python3", "-c"])
                    self.assertEqual(args[5:], [config["clock_service"], config["clock_dropin"], "2200,2100,2000",
                                                str(source), str(target)])
                    self.assertEqual(len(gates), 1 if rank == "0" else 2)
                    clocks[rank] = target
                    return "{}"

                backend.run = run
                backend.identity = Mock(return_value={"id": "owned"})
                backend.idle = Mock(return_value={"idle": True})
                self.assertEqual(backend.actual_clock(), source)
                receipts = backend.lower(self.sha, lambda: (gates.append(True) or {"clock_mhz": source}))
                self.assertEqual(len(receipts), 2)
                self.assertEqual(backend.actual_clock(), target)
                self.assertEqual(backend.identity.call_count, 8)
                self.assertEqual(backend.idle.call_count, 2)
        with self.assertRaisesRegex(ValueError, "below the floor"):
            backend.lower(self.sha, lambda: None)
        backend.run = lambda rank, *_: props(2200 if rank == "0" else 2100)
        with self.assertRaisesRegex(ValueError, "disagree"):
            backend.actual_clock()
        for output in (props(2150), props(2200) + props(2100), props(2200).replace("Result=success", "")):
            backend.run = lambda *_, text=output: text
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "ladder cap"):
                backend.actual_clock()

    def test_backend_step_refuses_durable_cap_mismatch_before_any_node_change(self):
        backend = Backend(backend_config(clock_ladder_mhz=list(LADDER3)))
        calls = []
        backend.run = lambda rank, args, **_: calls.append(args[0]) or props(2200)
        with self.assertRaisesRegex(ValueError, "durable/current cap mismatch"):
            backend.lower(self.sha, lambda: {"clock_mhz": 2100})
        self.assertNotIn("sudo", calls)

    def test_clock_program_exact_files_and_no_upclock(self):
        dropin = self.root / "dropin"
        sensor = self.root / "temp"
        sensor.write_text("80000")
        fake_thermal = SimpleNamespace(glob=lambda _: [sensor])
        ladder = "2200,2100,2000"
        for source, target in ((2200, 2100), (2100, 2000)):
            raw = (f"[Unit]\nDescription=GPU SM clock cap (300-{source} MHz)\n[Service]\nExecStart=\n"
                   f"ExecStart=/usr/bin/nvidia-smi -lgc 300,{source}\n")
            for extra in ("", "# unexpected\n"):
                dropin.write_text(raw + extra)
                shows = iter([props(source), props(target)])

                def run(args, **kwargs):
                    return SimpleNamespace(stdout=next(shows) if args[1] == "show" else "", stderr="")

                def path(value):
                    return fake_thermal if value == "/sys/class/thermal" else PosixPath(value)

                argv = ["-c", "gpu-clock-cap.service", str(dropin), ladder, str(source), str(target)]
                with patch("pathlib.Path", side_effect=path), patch("subprocess.run", side_effect=run), \
                        patch("sys.argv", argv), patch("builtins.print"):
                    if extra:
                        with self.assertRaises(AssertionError):
                            exec(CLOCK_PROGRAM, {})
                        self.assertEqual(dropin.read_text(), raw + extra)
                    else:
                        exec(CLOCK_PROGRAM, {})
                        self.assertIn(f"300,{target}\n", dropin.read_text())
                        self.assertIn(f"(300-{target} MHz)", dropin.read_text())
        for source, target in (("2000", "2100"), ("2200", "2000"), ("2000", "1900")):
            argv = ["-c", "gpu-clock-cap.service", str(dropin), ladder, source, target]
            with self.subTest(source=source, target=target), patch("sys.argv", argv), \
                    patch("subprocess.run") as run, self.assertRaises(AssertionError):
                exec(CLOCK_PROGRAM, {})
            run.assert_not_called()

    def test_memory_floor_stale_and_native_health_close_admission(self):
        server, health = self.server()
        for memory in (4.25, 4.749):
            self.samples(memory)
            self.assertFalse(server.admit("low"))
            self.assertEqual(self.p.inflight, 0)
            self.assertIsNone(self.p.prefill_request)
        self.samples(4.75)
        self.assertTrue(server.admit("boundary"))
        self.p.release_prefill("boundary")
        self.p.inflight -= 1
        self.samples(epoch=time.time() - 7)
        self.assertFalse(server.admit("stale"))
        self.samples()
        for result in ({"ok": True}, {"ok": True, "queued": 1}, {"ok": True, "queued": False}):
            health.return_value = result
            self.assertFalse(server.admit("native"))
            self.assertIsNone(self.p.prefill_request)
        health.side_effect = OSError("unavailable")
        self.assertFalse(server.admit("unavailable"))

    def test_lane_atomic_fresh_after_body_and_decode_overlap(self):
        server, health = self.server()
        server.config["limits"]["queue_seconds"] = 1
        self.samples()
        self.assertTrue(server.admit("first"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a, b = [pool.submit(server.admit, name) for name in ("a", "b")]
            time.sleep(0.05)
            self.assertEqual(self.p.inflight, 1)
            self.p.release_prefill("first")
            time.sleep(0.05)
            self.assertEqual(self.p.inflight, 1, "cached pre-body samples must not permit another prefill")
            self.samples()
            until = time.monotonic() + 0.5
            while self.p.inflight < 2 and time.monotonic() < until:
                time.sleep(0.005)
            self.assertEqual(self.p.inflight, 2, "decode may overlap next prefill")
            self.assertEqual(sum(f.done() for f in (a, b)), 1)
            with self.p.lock:
                self.p.terminal("cancel queued", time.time(), hard=True)
            self.assertEqual(sorted([a.result(), b.result()]), [False, True])

    def test_stop_during_native_probe_never_increments_inflight(self):
        server, health = self.server()
        self.samples()

        def stop():
            with self.p.lock:
                self.p.terminal("stop during probe", time.time(), hard=True)
            return {"ok": True, "queued": 0, "inflight": 0, "requests_running": 0}

        health.side_effect = stop
        self.assertFalse(server.admit("racing-stop"))
        self.assertEqual(self.p.inflight, 0)
        self.assertIsNone(self.p.prefill_request)

    def test_retained_reservations_and_wait_status_timeout(self):
        server, health = self.server()
        self.samples(5.0)
        self.assertTrue(server.admit("decode"))
        self.p.release_prefill("decode")
        self.samples(5.0)
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5})
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server.admit, "waiting")
            until = time.monotonic() + 0.1
            while "waiting" not in self.p.snapshot()["admission_waiters"] and time.monotonic() < until:
                time.sleep(0.005)
            status = self.p.snapshot()
            self.assertEqual(status["admission_waiters"]["waiting"], "memory_margin_and_retained_reservations")
            self.assertEqual(status["memory_reserved_total_gib"], 0.5)
            self.assertFalse(future.result(timeout=0.5))
        self.assertEqual(self.p.inflight, 1)
        self.assertEqual(self.p.admission_waiters, {})
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5})
        self.samples(5.25)
        self.assertTrue(server.admit("second"))
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5, "second": 0.5})

    def test_unknown_native_requests_after_restart_and_failed_completion_hold(self):
        server, health = self.server()
        self.samples()
        health.return_value = {"ok": True, "queued": 0, "inflight": 1, "requests_running": 1}
        self.assertFalse(server.admit("unknown-after-restart"))
        self.assertEqual(self.p.inflight, 0)
        self.p.memory_reservations["failed"] = 0.5
        self.p.unresolved_requests.add("failed")
        self.assertFalse(server.admit("orphan-still-running"))
        self.assertEqual(self.p.memory_reservations, {"failed": 0.5})
        health.return_value = {"ok": True, "queued": 0, "inflight": 0, "requests_running": 0}
        self.assertFalse(server.admit("need-post-idle-sample"))
        self.assertEqual(self.p.unresolved_requests, set())
        self.assertEqual(self.p.memory_reservations, {})
        self.samples()
        self.assertTrue(server.admit("idle-and-fresh"))

    def test_clock_receipt_cannot_clear_concurrent_hard_stop(self):
        with self.p.lock:
            self.p.condition("test", "test", time.time())
            self.p.terminal("hard stop during clock step", time.time(), hard=True)
        with self.assertRaises(ValueError):
            self.p.clock_changed([])
        self.assertEqual(self.p.data["pending_action"], "stop_owned")
        self.assertEqual(self.p.data["clock_mhz"], 2200)

    def test_generated_delta_probe_fragmentation_bounds_and_unknown(self):
        for delta in ({"content": "x"}, {"reasoning_content": "x"},
                      {"tool_calls": [{"function": {"arguments": "{"}}]},
                      {"function_call": {"arguments": "{"}}):
            probe = GeneratedTokenProbe()
            frame = b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\r\n\r\n"
            self.assertFalse(probe.feed(frame[:-1]))
            self.assertTrue(probe.feed(frame[-1:]))
        for frame in (b": ping\n\n", b"data: [DONE]\n\n", b"data: nope\n\n",
                      b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'):
            self.assertFalse(GeneratedTokenProbe().feed(frame))
        probe = GeneratedTokenProbe()
        self.assertFalse(probe.feed(b"data: " + b"x" * 65536))
        self.assertTrue(probe.disabled)
        self.assertFalse(probe.feed(b'\n\ndata: {"choices":[{"delta":{"content":"x"}}]}\n\n'))


class AdmissionTransportTests(TransportFixture, unittest.TestCase):
    def test_headers_role_keepalive_hold_until_generated_token(self):
        headers_sent, allow_role, role_sent = threading.Event(), threading.Event(), threading.Event()
        allow_token, second_started, finish = threading.Event(), threading.Event(), threading.Event()
        calls = []
        lock = threading.Lock()
        # Deliberately nonzero mislabeled prefilling: never use it as workspace proof.
        self.front.manager.backend.health = lambda: {
            "ok": True, "queued": 0, "inflight": 0, "requests_running": 0, "streams": {"prefilling": 99},
        }
        role = b': keepalive\n\ndata: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        token = b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
        tail = b"data: [DONE]\n\n"

        def post(handler):
            body = handler.rfile.read(int(handler.headers["Content-Length"]))
            with lock:
                calls.append((handler.headers.get("Authorization"), body))
                first = len(calls) == 1
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(role + token + tail)))
            handler.end_headers()
            handler.wfile.flush()
            if first:
                headers_sent.set()
                allow_role.wait(1)
            else:
                second_started.set()
            handler.wfile.write(role)
            handler.wfile.flush()
            if first:
                role_sent.set()
                allow_token.wait(1)
            handler.wfile.write(token)
            handler.wfile.flush()
            finish.wait(1)
            handler.wfile.write(tail)

        for event in (finish, allow_token, allow_role):
            self.addCleanup(event.set)
        with patch.object(Native, "do_POST", post), concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.request)
            self.assertTrue(headers_sent.wait(0.5))
            second = pool.submit(self.request)
            time.sleep(0.05)
            self.assertEqual(len(calls), 1, "headers must not release the lane")
            allow_role.set()
            self.assertTrue(role_sent.wait(0.5))
            time.sleep(0.05)
            self.assertEqual(len(calls), 1, "role/empty content/keepalive must not release the lane")
            allow_token.set()
            self.assertTrue(second_started.wait(0.5))
            self.assertEqual(self.p.inflight, 2)
            self.assertEqual(len(self.p.snapshot()["memory_reservations_gib"]), 2)
            finish.set()
            self.assertEqual(first.result(), (200, role + token + tail))
            self.assertEqual(second.result(), (200, role + token + tail))
        self.assertEqual(calls, [("Bearer exact", b'{"image":"raw","tools":[1]}')] * 2)
        self.wait_idle()
        self.assertEqual(self.p.inflight, 0)
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.memory_reservations, {})

    def test_truncated_native_body_retains_reservation_until_idle(self):
        def truncated(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            handler.send_response(200)
            handler.send_header("Content-Length", "20")
            handler.end_headers()
            handler.wfile.write(b"partial")
            handler.wfile.flush()
            handler.close_connection = True

        with patch.object(Native, "do_POST", truncated):
            with self.assertRaises(http.client.IncompleteRead):
                self.request()
        self.wait_idle()
        self.assertEqual(len(self.p.unresolved_requests), 1)
        self.assertEqual(sum(self.p.memory_reservations.values()), 0.5)
        self.assertIsNone(self.p.prefill_request)

    def test_empty_response_and_transport_failure_release_lane(self):
        def empty(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            handler.send_response(204)
            handler.end_headers()

        with patch.object(Native, "do_POST", empty):
            self.assertEqual(self.request(), (204, b""))
        self.wait_idle()
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.inflight, 0)
        # Verification failure exercises cleanup before any native transport is opened.
        self.front.manager.verify_generation = Mock(side_effect=ValueError("identity changed"))
        self.assertEqual(self.request()[0], 502)
        self.wait_idle()
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(self.p.memory_reservations, {})


if __name__ == "__main__":
    unittest.main()
