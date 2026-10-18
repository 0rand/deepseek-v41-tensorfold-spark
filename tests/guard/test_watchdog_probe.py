"""Idle-only watchdog probes defer (429) without queuing, and never count user pressure as engine failure."""

import http.client
import json
import time
import unittest

from guard_fixtures import Native, ProbeFixture, TransportFixture


class ProbeAdmissionTests(ProbeFixture, unittest.TestCase):
    def test_idle_probe_admits_normally_and_reserves_memory(self):
        self.assertTrue(self.server.admit("probe", probe=True))
        self.assertEqual(self.p.inflight, 1)
        self.assertEqual(self.p.memory_reservations, {"probe": 0.5})

    def test_probe_never_waits_behind_active_user_or_queued_request(self):
        for key, value in (("inflight", 1), ("prefill_request", "user"), ("admission_waiters", {"user": "cooling"})):
            with self.subTest(key=key):
                original = getattr(self.p, key)
                setattr(self.p, key, value)
                start = time.monotonic()
                self.assertFalse(self.server.admit("probe", probe=True))
                self.assertLess(time.monotonic() - start, 0.2)
                self.health.assert_not_called()
                setattr(self.p, key, original)

    def test_cooling_or_memory_pressure_defers_without_native_request(self):
        self.p.configure_admission({"prefill_admission_pause_c": 78, "prefill_admission_resume_c": 70})
        self.p.data["admission_cooldown"] = True
        self.assertFalse(self.server.admit("cool", probe=True))
        self.p.data["admission_cooldown"] = False
        for row in self.p.samples.values():
            row["available_gib"] = 4.7
        self.assertFalse(self.server.admit("memory", probe=True))
        self.health.assert_not_called()

    def test_unknown_native_work_does_not_queue_probe(self):
        self.health.return_value["inflight"] = 1
        start = time.monotonic()
        self.assertFalse(self.server.admit("unknown-native", probe=True))
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertEqual(self.p.memory_reservations, {})
        self.assertEqual(self.p.inflight, 0)


class ProbeHTTPTests(TransportFixture, unittest.TestCase):
    def probe(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=1)
        try:
            connection.request("POST", "/v1/chat/completions", b"{}", {"X-TF-Probe": "watchdog"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_active_user_gets_fast429_probe_without_upstream_work(self):
        with self.p.lock:
            self.p.inflight = 1
        status, body = self.probe()
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body)["code"], "watchdog_deferred")
        self.assertFalse(Native.received)
        self.assertEqual(self.p.inflight, 1)
        self.assertEqual(self.p.admission_waiters, {})

    def test_stop_stays503_for_watchdog_probe(self):
        self.p.data["terminal_stop"] = True
        self.assertEqual(self.probe()[0], 503)
        self.assertFalse(Native.received)


if __name__ == "__main__":
    unittest.main()
