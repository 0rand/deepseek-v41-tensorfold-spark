"""Decode-class admission (decode_admission_max_prompt_tokens) and the sampler period (sampler_period_s)."""

import concurrent.futures
import copy
import http.client
import http.server
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sampler
from backend import Backend
from common import load_config
from front import Front, Server
from guard_fixtures import RANKS, Native, ProbeFixture, backend_config, health_idle, production_config, write_config
from policy import Policy

SHORT = b'{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
IMAGE = b'{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:,"}}]}]}'


class AdmitTests(ProbeFixture, unittest.TestCase):
    """Actual Server.admit with a bare Server, a real Policy and frozen fresh telemetry."""

    def wait_reason(self, name, reason):
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: self.p.admission_waiters.get(name) == reason, 1),
                            (name, reason, dict(self.p.admission_waiters)))

    def refresh(self):
        with self.p.lock:
            for row in self.p.samples.values():
                row["epoch"] = time.time()
            self.p.lock.notify_all()

    def finish(self, name):
        """What dispatch's finally does for a completed request."""
        with self.p.lock:
            self.p.release_prefill(name)
            self.p.memory_reservations.pop(name, None)
            self.p.inflight -= 1
            self.p.lock.notify_all()

    def test_off_short_requests_still_wait_for_post_token_sample(self):
        self.server.config["limits"]["queue_seconds"] = 0.2
        self.assertTrue(self.server.admit("a"))
        self.assertEqual(self.p.prefill_request, "a")
        self.p.release_prefill("a")  # First token.
        self.assertFalse(self.server.admit("b"))
        self.assertEqual(self.p.memory_wait_reason(), "waiting_for_post_token_sample")
        self.refresh()
        self.assertTrue(self.server.admit("b"))
        self.assertEqual(self.p.memory_reservations, {"a": 0.5, "b": 0.5})

    def test_on_four_short_admitted_without_new_sample_and_fifth_waits_for_inflight(self):
        self.server.config["limits"]["queue_seconds"] = 0.2
        epochs = {rank: row["epoch"] for rank, row in self.p.samples.items()}
        for index in range(4):
            self.assertTrue(self.server.admit(f"short-{index}", decode=True))
            self.p.release_prefill(f"short-{index}")  # First token: no lane, so no post-token wait.
        self.assertEqual({rank: row["epoch"] for rank, row in self.p.samples.items()}, epochs)
        self.assertEqual(self.p.inflight, 4)
        self.assertIsNone(self.p.prefill_request)
        self.assertIsNone(self.p.admission_probe)
        self.assertEqual(self.p.prefill_completed_epoch, 0)
        self.assertEqual(self.p.memory_reservations, {f"short-{i}": 0.5 for i in range(4)})
        self.assertEqual(self.health.call_count, 4)
        self.assertFalse(self.server.admit("fifth", decode=True))  # max_inflight 4 is unchanged.
        self.assertEqual(self.health.call_count, 4)
        self.finish("short-0")
        self.assertTrue(self.server.admit("fifth", decode=True))

    def test_margin_counts_every_retained_reservation_until_completion(self):
        self.server.config["limits"]["queue_seconds"] = 0.2
        for row in self.p.samples.values():
            row["available_gib"] = 4.25 + 3 * 0.5  # Room for exactly three 0.5 GiB reservations.
        for name in ("a", "b", "c"):
            self.assertTrue(self.server.admit(name, decode=True))
            self.p.release_prefill(name)  # A first token never drops the reservation.
        self.assertEqual(sum(self.p.memory_reservations.values()), 1.5)
        self.assertFalse(self.server.admit("d", decode=True))
        self.assertEqual(self.p.memory_wait_reason(), "memory_margin_and_retained_reservations")
        self.assertNotIn("d", self.p.memory_reservations)
        self.finish("a")
        self.assertTrue(self.server.admit("d", decode=True))
        self.assertEqual(self.p.memory_reservations, {"b": 0.5, "c": 0.5, "d": 0.5})

    def test_long_request_takes_lane_and_forces_post_token_wait_for_short(self):
        self.assertTrue(self.server.admit("long"))
        self.assertEqual(self.p.prefill_request, "long")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            short = pool.submit(self.server.admit, "short", decode=True)
            self.wait_reason("short", "prefill_lane_busy")
            self.p.release_prefill("long")  # Long request's first token.
            self.wait_reason("short", "waiting_for_post_token_sample")
            self.assertFalse(short.done())
            self.refresh()
            self.assertTrue(short.result(timeout=1))
        completed = self.p.prefill_completed_epoch
        self.assertGreater(completed, 0)
        self.p.release_prefill("short")
        self.assertEqual(self.p.prefill_completed_epoch, completed)  # Only normal requests set the wait.
        self.assertEqual(self.p.inflight, 2)

    def test_fifo_short_behind_waiting_long_keeps_its_turn(self):
        self.p.prefill_request = "active"
        order = []

        def admit(name, decode):
            result = self.server.admit(name, decode=decode)
            with self.p.lock:
                order.append(name)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            long = pool.submit(admit, "long", False)
            self.wait_reason("long", "prefill_lane_busy")
            short = pool.submit(admit, "short", True)
            self.wait_reason("short", "admission_fifo")
            self.assertEqual(list(self.p.admission_waiters), ["long", "short"])
            self.p.release_prefill("active")
            self.refresh()
            self.assertTrue(long.result(timeout=1))
            self.wait_reason("short", "prefill_lane_busy")
            self.p.release_prefill("long")
            self.refresh()
            self.assertTrue(short.result(timeout=1))
        self.assertEqual(order, ["long", "short"])
        self.assertEqual(self.p.admission_waiters, {})

    def test_probe_token_serializes_health_io_for_both_classes(self):
        self.server.config["limits"]["queue_seconds"] = 0.3
        self.p.admission_probe = "other"
        for decode in (True, False):
            with self.subTest(decode=decode), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.server.admit, "next", decode=decode)
                self.wait_reason("next", "native_health_probe_busy")
                self.assertFalse(future.result(timeout=1))
                self.health.assert_not_called()
        self.p.admission_probe = None
        self.assertTrue(self.server.admit("next", decode=True))

    def test_concurrent_decode_probes_never_overlap_and_see_prior_increment(self):
        entered, release = threading.Event(), threading.Event()
        seen = []

        def health():
            with self.p.lock:
                seen.append((self.p.admission_probe, self.p.inflight))
            if not entered.is_set():
                entered.set()
                if not release.wait(2):
                    raise AssertionError("first health I/O never released")
            return dict(ok=True, queued=0, inflight=0, requests_running=0)

        self.health.side_effect = health
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.server.admit, "first", decode=True)
            self.assertTrue(entered.wait(1))
            second = pool.submit(self.server.admit, "second", decode=True)
            self.wait_reason("second", "admission_fifo")
            self.assertEqual(self.health.call_count, 1)
            release.set()
            self.assertTrue(first.result(timeout=1))
            self.assertTrue(second.result(timeout=1))
        self.assertEqual(seen, [("first", 0), ("second", 1)])
        self.assertIsNone(self.p.admission_probe)

    def test_hard_stop_refuses_waiting_and_probing_decode_requests(self):
        entered, release = threading.Event(), threading.Event()

        def health():
            entered.set()
            release.wait(2)
            return dict(ok=True, queued=0, inflight=0, requests_running=0)

        self.health.side_effect = health
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            probing = pool.submit(self.server.admit, "probing", decode=True)
            self.assertTrue(entered.wait(1))
            waiting = pool.submit(self.server.admit, "waiting", decode=True)
            self.wait_reason("waiting", "admission_fifo")
            with self.p.lock:
                self.p.terminal("test hard STOP", time.time(), hard=True)
            release.set()
            self.assertFalse(probing.result(timeout=1))
            self.assertFalse(waiting.result(timeout=1))
        self.assertEqual((self.p.inflight, self.p.memory_reservations, self.p.admission_waiters), (0, {}, {}))
        self.assertIsNone(self.p.admission_probe)
        self.assertIsNone(self.p.prefill_request)
        self.assertFalse(self.server.admit("later", decode=True))

    def test_recovery_keeps_lane_idle_proof_and_post_idle_sample(self):
        self.server.config["limits"]["queue_seconds"] = 0.2
        self.p.unresolved_requests.add("orphan")
        self.p.memory_reservations["orphan"] = 0.5
        self.assertFalse(self.server.admit("short", decode=True))
        self.assertEqual(self.p.unresolved_requests, set())
        self.assertEqual(self.p.memory_reservations, {})
        self.assertGreater(self.p.prefill_completed_epoch, 0)  # Idle proof required a new sample.
        self.assertEqual(self.p.memory_wait_reason(), "waiting_for_post_token_sample")
        self.refresh()
        self.assertTrue(self.server.admit("short", decode=True))
        self.assertIsNone(self.p.prefill_request)


class TokenizeNative(Native):
    count = 12

    def do_POST(self):
        if self.path != "/tokenize":
            return super().do_POST()
        self.rfile.read(int(self.headers["Content-Length"]))
        raw = json.dumps({"count": type(self).count}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class DispatchTests(unittest.TestCase):
    """Real front and fake native over loopback; telemetry is frozen so no post-token sample ever arrives."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = Policy(self.root / "state", self.root / "events")
        self.p.owned = self.p.fresh = True
        self.p.samples = {rank: dict(epoch=time.time(), available_gib=8, max_zone_c=70) for rank in RANKS}
        TokenizeNative.received, TokenizeNative.maximum, TokenizeNative.active = [], 0, 0
        TokenizeNative.finish_barrier, TokenizeNative.count = None, 12
        self.native = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TokenizeNative)
        self.front = Server(("127.0.0.1", 0), Front)
        self.limits = dict(max_pending=8, max_inflight=4, body_buffers=4, max_connections=32, max_body_bytes=4096,
                           queue_seconds=1, upstream_seconds=2)
        backend = Backend(backend_config(upstream=f"http://127.0.0.1:{self.native.server_port}"))
        self.count = Mock(side_effect=backend.request_token_count)
        manager = SimpleNamespace(policy=self.p, verify_generation=lambda: None, backend=SimpleNamespace(
            health=health_idle, request_token_count=self.count))
        self.front.configure(dict(upstream=f"http://127.0.0.1:{self.native.server_port}", limits=self.limits),
                             manager)
        for server in (self.native, self.front):
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def request(self, body=SHORT):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=4)
        try:
            connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def admitted_events(self):
        rows = [json.loads(line) for line in (self.root / "events").read_text().splitlines()]
        return [row for row in rows if row["event"] == "REQUEST_ADMITTED"]

    def generations(self):
        return [body for _, body in TokenizeNative.received]

    def wait_idle(self):
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: self.p.inflight == 0, 1))

    def test_off_second_short_request_waits_for_post_token_sample(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.request)
            with self.p.lock:
                self.assertTrue(self.p.lock.wait_for(lambda: self.p.inflight == 1, 1))
            second = pool.submit(self.request)
            results = [first.result(timeout=3), second.result(timeout=3)]
        self.assertEqual(results[0], (200, TokenizeNative.payload))
        self.assertEqual(results[1][0], 503)
        self.assertIn(b"queue deadline", results[1][1])
        self.assertEqual(self.generations(), [SHORT])
        self.assertEqual(self.count.call_count, 1)  # Counted after admission only, as before.
        self.assertTrue(all("class" not in row and "prompt_tokens" not in row for row in self.admitted_events()))
        self.assertGreater(self.p.prefill_completed_epoch, 0)

    def test_on_four_short_requests_overlap_without_new_sample(self):
        self.limits["decode_admission_max_prompt_tokens"] = 2048
        TokenizeNative.finish_barrier = threading.Barrier(4)
        epochs = {rank: row["epoch"] for rank, row in self.p.samples.items()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request(), range(4)))
        self.assertEqual(results, [(200, TokenizeNative.payload)] * 4)
        self.assertEqual(TokenizeNative.maximum, 4)
        self.assertEqual({rank: row["epoch"] for rank, row in self.p.samples.items()}, epochs)
        self.assertEqual(self.count.call_count, 4)  # One count per request, reused for metadata.
        self.assertEqual([(row["class"], row["prompt_tokens"]) for row in self.admitted_events()],
                         [("decode", 12)] * 4)
        self.wait_idle()
        self.assertEqual(self.p.prefill_completed_epoch, 0)
        self.assertEqual((self.p.memory_reservations, self.p.request_metadata), ({}, {}))

    def test_on_over_limit_multimodal_or_tokenize_failure_takes_normal_path(self):
        self.limits["decode_admission_max_prompt_tokens"] = 11
        cases = (("over_limit", SHORT, 12), ("multimodal", IMAGE, 12), ("tokenize_error", SHORT, -1))
        for name, body, count in cases:
            with self.subTest(name):
                with self.p.lock:
                    TokenizeNative.count = count
                    self.p.prefill_completed_epoch = 0
                    self.p.prefill_sample_epochs = {}
                self.assertEqual(self.request(body), (200, TokenizeNative.payload))
                self.wait_idle()
                row = self.admitted_events()[-1]
                self.assertEqual(row["class"], "prefill")
                self.assertEqual(row["prompt_tokens"], 12 if name == "over_limit" else None)
                self.assertGreater(self.p.prefill_completed_epoch, 0)  # Took the lane; set the post-token wait.
                with self.p.lock:
                    self.p.samples = {r: dict(epoch=time.time(), available_gib=8, max_zone_c=70) for r in RANKS}
        self.assertEqual(self.count.call_count, 3)

    def test_on_metadata_reuses_count_and_stop_cleans_up(self):
        self.limits["decode_admission_max_prompt_tokens"] = 64
        hold = TokenizeNative.finish_barrier = threading.Event()  # Native holds the stream until set.
        self.addCleanup(hold.set)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.request)
            deadline = time.monotonic() + 1
            while not self.generations() and time.monotonic() < deadline:
                threading.Event().wait(0.01)  # Until native holds the dispatched stream.
            with self.p.lock:
                self.assertEqual(len(self.p.active), 1)
                self.assertEqual([(row["prompt_tokens"], row["count_source"])
                                  for row in self.p.request_metadata.values()], [(12, "native_tokenize")])
                self.p.update({r: dict(epoch=time.time(), max_zone_c=94, available_gib=8) for r in RANKS},
                              True, "own_window")
            self.assertTrue(self.p.data["hard_stop"])
            try:
                future.result(timeout=3)  # The cut stream may surface as a client-side read error.
            except (OSError, http.client.HTTPException):
                pass
            hold.set()
        self.wait_idle()
        self.assertEqual(self.count.call_count, 1)
        # A cut stream is unverified: its reservation stays until reconciliation, as before.
        self.assertEqual(len(self.p.unresolved_requests), 1)
        self.assertEqual(list(self.p.memory_reservations.values()), [0.5])
        self.assertEqual((self.p.active, self.p.request_metadata, self.p.admission_waiters), ({}, {}, {}))
        self.assertEqual(self.request()[0], 503)
        self.assertEqual(len(self.generations()), 1)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = production_config(self.root)

    def load(self, limit=None, period=None):
        config = copy.deepcopy(self.base)
        if limit is not None:
            config["limits"]["decode_admission_max_prompt_tokens"] = limit
        if period is not None:
            config["sampler_period_s"] = period
        return load_config(write_config(self.root, config))

    def test_decode_limit_bounds(self):
        self.assertNotIn("decode_admission_max_prompt_tokens", self.load()["limits"])
        for value in (0, 1, 2048):
            self.assertEqual(self.load(limit=value)["limits"]["decode_admission_max_prompt_tokens"], value)
        for value in (-1, 2049, 4096, 1.0, True, "512"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(limit=value)

    def test_sampler_period_bounds(self):
        for value in (0.25, 1, 2.0):
            self.assertEqual(self.load(period=value)["sampler_period_s"], value)
        for value in (0.24, 0, -1, 2.01, 3, math.inf, math.nan, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(period=value)


class SamplerPeriodTests(unittest.TestCase):
    def run_once(self, config):
        class Done(Exception):
            pass

        slept = []

        def sleep(seconds):
            slept.append(seconds)
            raise Done

        sample = dict(epoch=time.time(), available_gib=8, max_zone_c=70)
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(config, samples_path=str(Path(tmp) / "samples.json"))
            with patch.object(sys, "argv", ["sampler", "--config", "unused"]), \
                    patch.object(sampler, "load_config", return_value=config), \
                    patch.object(sampler, "local_sample", return_value=sample), \
                    patch.object(sampler, "worker_sample", return_value=sample), \
                    patch.object(sampler, "time", SimpleNamespace(monotonic=time.monotonic, sleep=sleep)), \
                    self.assertRaises(Done):
                sampler.main()
            self.assertEqual(set(json.loads(Path(config["samples_path"]).read_text())["samples"]), {"0", "1"})
        return slept[0]

    def test_sampler_sleeps_for_configured_period(self):
        self.assertTrue(1.5 < self.run_once({}) <= 2.0)
        self.assertTrue(0.15 < self.run_once({"sampler_period_s": 0.25}) <= 0.25)
        self.assertTrue(0.5 < self.run_once({"sampler_period_s": 1.0}) <= 1.0)


if __name__ == "__main__":
    unittest.main()
