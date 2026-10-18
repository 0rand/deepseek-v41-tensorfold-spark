"""FIFO and watchdog priority checks against actual Server.admit."""

import concurrent.futures
import hashlib
import http.client
import json
import threading
import time
import unittest

from guard_fixtures import Native, ProbeFixture, TransportFixture


class AdmissionFIFOTests(ProbeFixture, unittest.TestCase):
    def wait_ticket(self, name):
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: name in self.p.admission_waiters, 1))

    def release_lane(self, name):
        with self.p.lock:
            self.p.release_prefill(name)
            for row in self.p.samples.values():
                row["epoch"] = time.time()
            self.p.lock.notify_all()

    def test_seeded_older_ticket_blocks_newcomer_before_health_io(self):
        cancelled = threading.Event()
        self.p.admission_waiters["older"] = "cooling"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.server.admit, "newcomer", cancelled)
            try:
                self.wait_ticket("newcomer")
                self.assertFalse(future.done())
                self.health.assert_not_called()
                self.assertIsNone(self.p.prefill_request)
                self.assertEqual(list(self.p.admission_waiters), ["older", "newcomer"])
            finally:
                cancelled.set()
                with self.p.lock:
                    self.p.lock.notify_all()
            self.assertFalse(future.result(timeout=1))
        self.assertEqual(self.p.admission_waiters, {"older": "cooling"})
        self.assertEqual(self.p.memory_reservations, {})

    def test_concurrent_actual_admissions_follow_ticket_order(self):
        self.p.prefill_request = "active"
        cancel = threading.Event()
        order = []

        def admit(name):
            result = self.server.admit(name, cancel)
            if result:
                with self.p.lock:
                    order.append(name)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            try:
                for index in range(4):
                    name = f"user-{index}"
                    futures.append(pool.submit(admit, name))
                    self.wait_ticket(name)
                self.health.assert_not_called()
                self.release_lane("active")
                for index, future in enumerate(futures):
                    self.assertTrue(future.result(timeout=1))
                    self.assertEqual(order, [f"user-{i}" for i in range(index + 1)])
                    self.release_lane(f"user-{index}")
            finally:
                cancel.set()
                with self.p.lock:
                    self.p.lock.notify_all()
        self.assertEqual(self.p.inflight, 4)
        self.assertEqual(self.health.call_count, 4)
        self.assertEqual(self.p.admission_waiters, {})
        self.assertIsNone(self.p.prefill_request)

    def test_cancelled_or_timed_out_head_releases_lane_and_next_ticket(self):
        for ending in ("cancel", "timeout"):
            with self.subTest(ending=ending):
                entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
                self.server.config["limits"]["queue_seconds"] = 0.1 if ending == "timeout" else 2

                def health():
                    if not entered.is_set():
                        entered.set()
                        if not release.wait(2):
                            raise AssertionError("head health I/O never released")
                    return dict(ok=True, queued=0, inflight=0, requests_running=0)

                self.health.side_effect = health
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    head = pool.submit(self.server.admit, "head", cancelled)
                    try:
                        self.assertTrue(entered.wait(1))
                        self.server.config["limits"]["queue_seconds"] = 2
                        tail = pool.submit(self.server.admit, "tail")
                        self.wait_ticket("tail")
                        self.assertEqual(list(self.p.admission_waiters), ["head", "tail"])
                        if ending == "cancel":
                            cancelled.set()
                        else:
                            threading.Event().wait(0.15)
                    finally:
                        release.set()
                    self.assertFalse(head.result(timeout=1))
                    self.assertTrue(tail.result(timeout=1))
                self.assertEqual(self.p.admission_waiters, {})
                self.assertEqual(self.p.memory_reservations, {"tail": 0.5})
                self.assertEqual(self.p.prefill_request, "tail")
                self.release_lane("tail")
                self.p.inflight = 0
                self.p.memory_reservations.clear()

    def test_watchdog_recovery_still_defers_on_memory_or_thermal_hold(self):
        self.p.unresolved_requests.add("orphan")
        self.p.memory_reservations["orphan"] = 0.5
        self.p.configure_admission({"prefill_admission_pause_c": 78, "prefill_admission_resume_c": 70})
        for reason in ("cooling", "memory"):
            with self.subTest(reason=reason):
                self.p.data["admission_cooldown"] = reason == "cooling"
                for row in self.p.samples.values():
                    row["available_gib"] = 8 if reason == "cooling" else 4.7
                self.assertFalse(self.server.admit("probe", probe=True))
                self.health.assert_not_called()
                self.assertEqual(self.p.unresolved_requests, {"orphan"})
                self.assertEqual(self.p.memory_reservations, {"orphan": 0.5})
                self.assertEqual(self.p.admission_waiters, {})
                self.assertIsNone(self.p.prefill_request)

    def test_watchdog_own_ticket_yields_to_user_arriving_during_health_io(self):
        entered, release = threading.Event(), threading.Event()

        def health():
            if not entered.is_set():
                entered.set()
                if not release.wait(2):
                    raise AssertionError("watchdog health I/O never released")
            return dict(ok=True, queued=0, inflight=0, requests_running=0)

        self.health.side_effect = health
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            probe = pool.submit(self.server.admit, "probe", probe=True)
            try:
                self.assertTrue(entered.wait(1), "own ticket must not defer an idle probe")
                user = pool.submit(self.server.admit, "user")
                self.wait_ticket("user")
                self.assertEqual(list(self.p.admission_waiters), ["probe", "user"])
            finally:
                release.set()
            self.assertFalse(probe.result(timeout=1))
            self.assertTrue(user.result(timeout=1))
        self.assertEqual(self.p.inflight, 1)
        self.assertEqual(self.p.memory_reservations, {"user": 0.5})
        self.assertEqual(self.p.admission_waiters, {})
        self.assertEqual(self.p.prefill_request, "user")


class MaintenanceProbeHTTPTests(TransportFixture, unittest.TestCase):
    def test_maintenance_defers_watchdog_without_bypassing_credentials(self):
        maintenance = "verification-fixture"
        self.front.config["maintenance_token_sha256"] = hashlib.sha256(maintenance.encode()).hexdigest()

        def send(headers):
            connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=2)
            try:
                connection.request("POST", "/v1/chat/completions", b"{}", headers)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        for headers in ({"X-TF-Probe": "watchdog"},
                        {"X-TF-Probe": "watchdog", "X-TF-Maintenance": "wrong"}):
            with self.subTest(headers=headers):
                status, body = send(headers)
                self.assertEqual(status, 429)
                self.assertEqual(json.loads(body)["code"], "watchdog_deferred")
                self.assertFalse(Native.received)
                self.assertEqual(self.p.inflight, 0)
                self.assertEqual(self.p.admission_waiters, {})
        self.assertEqual(send({})[0], 503)
        self.assertFalse(Native.received)
        self.assertEqual(send({"X-TF-Maintenance": maintenance}), (200, Native.payload))
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: self.p.inflight == 0, 1))
            self.p.samples = {rank: dict(epoch=time.time(), available_gib=8, max_zone_c=70)
                              for rank in ("0", "1")}
        self.assertEqual(send({"X-TF-Probe": "watchdog", "X-TF-Maintenance": maintenance}),
                         (200, Native.payload))
        self.assertEqual(len(Native.received), 2)


if __name__ == "__main__":
    unittest.main()
