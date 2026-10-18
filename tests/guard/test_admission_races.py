"""Admission race, client FIN cancellation and preventive thermal pacing regression checks."""

import concurrent.futures
import copy
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from common import POLICY
from guard_fixtures import LadderFixture, Native, TransportFixture


class AdmissionFixTests(LadderFixture, unittest.TestCase):
    def test_cleanup_during_health_io_invalidates_old_idle_proof(self):
        for recovering in (False, True):
            with self.subTest(recovering=recovering):
                self.p.inflight = 0
                self.p.unresolved_requests.clear()
                self.p.memory_reservations.clear()
                self.samples()
                server, health = self.server()
                entered, release = threading.Event(), threading.Event()
                if recovering:
                    self.p.unresolved_requests.add("old-orphan")
                    self.p.memory_reservations["old-orphan"] = 0.5
                else:
                    self.p.inflight = 1
                    self.p.memory_reservations["active"] = 0.5
                calls = []

                def probe():
                    calls.append(True)
                    if len(calls) == 1:
                        entered.set()
                        release.wait(1)
                        return dict(ok=True, queued=0, inflight=0, requests_running=0)
                    return dict(ok=True, queued=0, inflight=1, requests_running=1)

                health.side_effect = probe
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(server.admit, "next")
                    self.assertTrue(entered.wait(0.5))
                    with self.p.lock:
                        # Actual failed transport cleanup while health I/O is blocked.
                        self.p.inflight = 0
                        self.p.unresolved_requests.add("active")
                        self.p.memory_reservations["active"] = 0.5
                    release.set()
                    self.assertFalse(future.result(timeout=1))
                self.assertIn("active", self.p.unresolved_requests)
                self.assertEqual(self.p.memory_reservations["active"], 0.5)
                self.assertEqual(self.p.inflight, 0)
                self.assertIsNone(self.p.prefill_request)
                self.assertGreater(len(calls), 1)

    def test_exact_bounds_restart_freshness_and_hard_policy(self):
        limits = dict(prefill_admission_pause_c=82, prefill_admission_resume_c=75)
        before = copy.deepcopy(POLICY)
        self.p.configure_admission(limits)
        self.temperatures(75, 75)
        self.assertIsNone(self.p.thermal_wait_reason())
        self.temperatures(81.999, 75)
        self.assertFalse(self.p.data["admission_cooldown"])
        self.temperatures(75, 82)
        self.assertTrue(self.p.data["admission_cooldown"])
        self.assertFalse(self.p.data["paused"])
        self.assertTrue(self.p.ready(), "pacing must not hold lifecycle or abort active work")
        self.assertEqual(self.p.snapshot()["thermal_wait_reason"], "admission_thermal_cooldown")
        self.temperatures(75, 75.001)
        self.assertTrue(self.p.data["admission_cooldown"])
        self.p = self.reload()
        self.p.configure_admission(limits)
        self.temperatures(80, 80)
        self.assertTrue(self.p.data["admission_cooldown"], "restart must retain hot history")
        self.temperatures(75, 75, time.time() - 7)
        self.assertTrue(self.p.data["admission_cooldown"])
        self.assertEqual(self.p.thermal_wait_reason(), "admission_telemetry_stale_or_missing")
        self.temperatures(75, 75)
        self.assertFalse(self.p.data["admission_cooldown"])
        events = [json.loads(line)["event"] for line in (self.root / "events").read_text().splitlines()]
        self.assertEqual(events.count("ADMISSION_COOLDOWN_ENTER"), 1)
        self.assertEqual(events.count("ADMISSION_COOLDOWN_EXIT"), 2)
        self.assertEqual(POLICY, before)
        self.temperatures(94, 75)
        self.temperatures(75, 75)
        self.assertTrue(self.p.data["hard_stop"])
        self.assertTrue(self.p.data["terminal_stop"])

    def test_cooldown_queue_drains_after_both_ranks_fresh_cool(self):
        self.p.configure_admission(dict(prefill_admission_pause_c=82, prefill_admission_resume_c=75))
        self.temperatures(75, 75)
        server, _ = self.server()
        server.config["limits"]["queue_seconds"] = 1
        self.assertTrue(server.admit("active"))
        self.p.release_prefill("active")
        self.temperatures(82, 75)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server.admit, "waiting")
            with self.p.lock:
                self.assertTrue(self.p.lock.wait_for(
                    lambda: self.p.admission_waiters.get("waiting") == "admission_thermal_cooldown", 0.5))
            self.assertFalse(future.done())
            self.assertEqual(self.p.memory_reservations, {"active": 0.5})
            self.assertEqual(self.p.inflight, 1)
            self.temperatures(75, 75.001)
            self.assertFalse(future.done())
            self.temperatures(75, 75)
            self.assertTrue(future.result(timeout=0.5))
        self.assertEqual(self.p.inflight, 2)

    def test_missing_rank_cannot_clear_cooldown_and_dispatch_probe_rechecks_heat(self):
        self.p.configure_admission(dict(prefill_admission_pause_c=82, prefill_admission_resume_c=75))
        self.temperatures(82, 75)
        now = time.time()
        self.p.update({"0": dict(epoch=now, max_zone_c=75, available_gib=8)}, True, "owned")
        self.assertTrue(self.p.data["admission_cooldown"])
        self.temperatures(75, 75)
        server, health = self.server()

        def heat_during_probe():
            self.temperatures(82, 75)
            return dict(ok=True, queued=0, inflight=0, requests_running=0)

        health.side_effect = heat_during_probe
        self.assertFalse(server.admit("heated-during-probe"))
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(self.p.memory_reservations, {})

    def test_old_config_disabled_old_state_conservative_when_enabled(self):
        self.temperatures(83, 80)
        self.assertIsNone(self.p.thermal_wait_reason())
        self.p.configure_admission(dict(prefill_admission_pause_c=82, prefill_admission_resume_c=75))
        self.temperatures(80, 80)
        self.assertEqual(self.p.thermal_wait_reason(), "admission_thermal_cooldown")
        self.temperatures(75, 75)
        self.assertIsNone(self.p.thermal_wait_reason())
        self.p.configure_admission({})
        self.temperatures(83, 83)
        self.assertIsNone(self.p.thermal_wait_reason())

    def test_hard_stop_is_durable_before_optional_pacing_io(self):
        self.p.configure_admission(dict(prefill_admission_pause_c=82, prefill_admission_resume_c=75))
        with patch.object(self.p, "update_admission_cooldown", side_effect=OSError("pacing event I/O")):
            with self.assertRaises(OSError):
                self.temperatures(94, 75)
        self.assertTrue(self.reload().data["hard_stop"])


class DisconnectFixTests(TransportFixture, unittest.TestCase):
    def client(self):
        client = socket.create_connection(("127.0.0.1", self.front.server_port), timeout=2)
        self.addCleanup(client.close)
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nContent-Length: 2\r\n\r\n{}")
        return client

    def test_fin_cancels_blocked_headers_and_body_retaining_budget(self):
        for phase in ("headers", "body"):
            with self.subTest(phase=phase):
                entered, native_closed = threading.Event(), threading.Event()

                def stalled(handler):
                    handler.rfile.read(int(handler.headers["Content-Length"]))
                    if phase == "body":
                        handler.send_response(200)
                        handler.send_header("Content-Length", "99")
                        handler.end_headers()
                        handler.wfile.flush()
                    entered.set()
                    handler.connection.settimeout(1.5)
                    if handler.connection.recv(1) == b"":
                        native_closed.set()
                    handler.close_connection = True

                with patch.object(Native, "do_POST", stalled):
                    client = self.client()
                    self.assertTrue(entered.wait(0.5))
                    if phase == "body":
                        self.assertIn(b"200", client.recv(4096))
                    client.shutdown(socket.SHUT_WR)
                    self.assertTrue(native_closed.wait(0.6), "FIN must close native before transport timeout")
                    self.assertTrue(self.wait_idle())
                    self.assertEqual(len(self.p.unresolved_requests), 1)
                    self.assertEqual(sum(self.p.memory_reservations.values()), 0.5)
                    self.assertIsNone(self.p.prefill_request)
                    self.front.manager.backend.health = lambda: dict(ok=True, queued=0, inflight=1, requests_running=1)
                    self.front.config["limits"]["queue_seconds"] = 0.1
                    self.assertFalse(self.front.admit("still-running"))
                    self.assertEqual(len(self.p.unresolved_requests), 1)
                    self.front.manager.backend.health = lambda: dict(ok=True, queued=0, inflight=0, requests_running=0)
                    self.front.config["limits"]["queue_seconds"] = 1
                    self.assertTrue(self.front.admit("recovered"))
                    with self.p.lock:
                        self.p.release_prefill("recovered")
                        self.p.memory_reservations.clear()
                        self.p.inflight = 0
                    client.close()
                until = time.monotonic() + 0.3
                while (any(t.name == "guard-client-disconnect" for t in threading.enumerate())
                       and time.monotonic() < until):
                    time.sleep(0.01)
                self.assertFalse(any(t.name == "guard-client-disconnect" for t in threading.enumerate()))

    def test_fin_removes_queue_waiter_without_dispatch_or_reservation(self):
        with self.p.lock:
            self.p.prefill_request = "occupied"
        client = self.client()
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: bool(self.p.admission_waiters), 0.5))
        client.shutdown(socket.SHUT_WR)
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: not self.p.admission_waiters, 0.5))
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(self.p.memory_reservations, {})
        self.assertEqual(Native.received, [])

    def test_watcher_preserves_pipelined_request(self):
        client = self.client()
        client.sendall(b"GET /health HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n")
        data = b""
        while True:
            block = client.recv(65536)
            if not block:
                break
            data += block
        self.assertEqual(data.count(b"HTTP/1.1 200"), 2)
        self.assertIn(Native.first, data)
        self.assertTrue(data.endswith(b"{}\r\n0\r\n\r\n"))


if __name__ == "__main__":
    unittest.main()
