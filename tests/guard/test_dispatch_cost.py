"""Per-request dispatch cost: the generation-verify cache, early body-slot release and the admission gate order."""

import concurrent.futures
import http.client
import http.server
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

from common import load_config
from daemon import Manager
from front import Front, Server
from guard_fixtures import (
    RANKS,
    Native,
    health_idle,
    production_config,
    synthetic_sha,
    write_config,
)
from policy import Policy

SHORT = b'{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
SHA = synthetic_sha("base")
INSPECT_SECONDS = (
    0.2  # Per rank: one verification (local + worker inspect) costs 0.4 s, as measured.
)


class FakeHost:
    """Container inspection with a fixed latency and switchable container IDs, ownership and profile."""

    def __init__(self, latency=0.0):
        self.latency, self.lock = latency, threading.Lock()
        self.ids = {rank: "container-" + rank for rank in RANKS}
        self.owned, self.sha = True, SHA
        self.inspections = 0
        self.boot_proof = Mock(return_value={"epoch": 0})

    def identity(self, rank, expected_sha, require_running=True):
        if not self.owned or self.sha != expected_sha:
            raise ValueError("ownership/profile changed")
        time.sleep(self.latency)
        with self.lock:
            self.inspections += 1
        return {
            "id": self.ids[rank],
            "image": "image-" + rank,
            "profile": expected_sha,
            "lease_mode": "owned",
        }

    def ownership(self):
        return (True, "owned") if self.owned else (False, "operator_lease")

    def profile(self):
        return self.sha, {}


def ready_policy(root):
    policy = Policy(Path(root) / "state", Path(root) / "events", initial_profile=SHA)
    policy.owned = policy.fresh = True
    policy.samples = {
        rank: {"epoch": time.time(), "available_gib": 8, "max_zone_c": 70}
        for rank in RANKS
    }
    policy.data["clock_verified"] = True
    policy.data["boot_verified_ids"] = {rank: "container-" + rank for rank in RANKS}
    return policy


def manager(policy, host, limits):
    """The real Manager.verify_generation without the host-touching constructor."""
    value = object.__new__(Manager)
    value.config, value.backend, value.policy = {"limits": limits}, host, policy
    value.boot_lock, value.verify_lock, value.verified = (
        threading.Lock(),
        threading.Lock(),
        None,
    )
    return value


class VerifyCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = ready_policy(self.tmp.name)
        self.host = FakeHost()
        self.limits = {"generation_verify_cache_s": 0.3}
        self.m = manager(self.p, self.host, self.limits)

    def test_off_inspects_both_ranks_on_every_request(self):
        del self.limits["generation_verify_cache_s"]
        for _ in range(3):
            self.m.verify_generation()
        self.assertEqual(self.host.inspections, 6)
        self.assertIsNone(self.m.verified)

    def test_hit_within_ttl_and_miss_after_expiry(self):
        first = self.m.verify_generation()
        self.assertEqual(self.m.verify_generation(), first)
        self.assertEqual(self.host.inspections, 2)
        time.sleep(0.35)
        self.m.verify_generation()
        self.assertEqual(self.host.inspections, 4)

    def test_native_doubt_invalidates_and_restart_reverifies_boot(self):
        self.m.verify_generation()
        self.host.ids["1"] = "container-1-restarted"
        self.p.native_changed()  # What a failed health read or a native error raises.
        identities = self.m.verify_generation()
        self.assertEqual(self.host.inspections, 4)
        self.assertEqual(identities["1"]["id"], "container-1-restarted")
        self.host.boot_proof.assert_called_once_with(SHA)
        self.assertEqual(self.p.data["boot_verified_ids"]["1"], "container-1-restarted")
        self.m.verify_generation()
        self.assertEqual(
            self.host.inspections, 4
        )  # The re-verified identity is cached again.

    def test_failed_health_read_in_admission_invalidates(self):
        self.m.verify_generation()
        server = object.__new__(Server)
        server.config = {"limits": {"max_inflight": 4, "queue_seconds": 0.1}}
        server.manager = SimpleNamespace(
            policy=self.p, backend=SimpleNamespace(health=Mock(side_effect=OSError))
        )
        self.assertFalse(server.admit("a", decode=True))
        self.m.verify_generation()
        self.assertEqual(self.host.inspections, 4)

    def test_doubt_during_inspection_is_not_cached_as_fresh(self):
        identity = self.host.identity

        def doubting(rank, sha, require_running=True):
            if rank == "1":
                self.p.native_changed()
            return identity(rank, sha, require_running)

        self.host.identity = doubting
        self.m.verify_generation()
        self.host.identity = identity
        self.m.verify_generation()
        self.assertEqual(self.host.inspections, 4)

    def test_lease_profile_and_readiness_are_checked_on_every_hit(self):
        self.m.verify_generation()
        self.host.owned = False
        with self.assertRaisesRegex(ValueError, "ownership/profile changed"):
            self.m.verify_generation()
        self.host.owned = True
        self.host.sha = synthetic_sha("next")
        with self.assertRaisesRegex(ValueError, "ownership/profile changed"):
            self.m.verify_generation()
        self.host.sha = SHA
        self.m.verify_generation()
        with self.p.lock:
            self.p.data["paused"] = True
        with self.assertRaisesRegex(ValueError, "guard state changed"):
            self.m.verify_generation()

    def test_terminal_stop_invalidates(self):
        self.m.verify_generation()
        generation = self.p.native_generation
        with self.p.lock:
            self.p.terminal("test", time.time())
        self.assertGreater(self.p.native_generation, generation)

    def test_concurrent_misses_share_one_inspection(self):
        self.host.latency = INSPECT_SECONDS
        self.limits["generation_verify_cache_s"] = (
            2.0  # The TTL runs from the start of the 0.4 s inspection.
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.m.verify_generation(), range(4)))
        self.assertEqual(self.host.inspections, 2)
        self.assertTrue(all(row == results[0] for row in results))

    def test_config_bounds(self):
        root = Path(self.tmp.name)
        for key, value, ok in (
            ("generation_verify_cache_s", 2.0, True),
            ("generation_verify_cache_s", 0, True),
            ("generation_verify_cache_s", 5.5, False),
            ("generation_verify_cache_s", "2", False),
            ("generation_verify_cache_s", float("nan"), False),
            ("small_body_release_bytes", 1024, True),
            ("small_body_release_bytes", 1025, False),
            ("small_body_release_bytes", -1, False),
            ("small_body_release_bytes", 1.0, False),
        ):
            with self.subTest(key=key, value=value):
                config = production_config(root)
                config["limits"][key] = value
                path = write_config(root, config)
                if ok:
                    self.assertEqual(load_config(path)["limits"][key], value)
                else:
                    with self.assertRaises(ValueError):
                        load_config(path)


class TimedNative(Native):
    """Fake native that records when each generation reaches it."""

    arrivals: ClassVar[list] = []

    def do_POST(self):
        with self.lock:
            type(self).arrivals.append(time.monotonic())
        super().do_POST()


class DispatchFixture:
    """A real front with G1 on, the real cached verification over a slow fake host and a counting tokenize."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = ready_policy(self.tmp.name)
        self.host = FakeHost(INSPECT_SECONDS)
        TimedNative.arrivals, TimedNative.received = [], []
        TimedNative.active = TimedNative.maximum = 0
        TimedNative.finish_barrier = None
        self.native = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TimedNative)
        self.front = Server(("127.0.0.1", 0), Front)
        self.limits = {
            "max_pending": 8,
            "max_inflight": 4,
            "body_buffers": 2,
            "max_connections": 32,
            "max_body_bytes": 16 * 1024 * 1024,
            "queue_seconds": 5,
            "upstream_seconds": 5,
            "decode_admission_max_prompt_tokens": 2048,
        }
        self.m = manager(self.p, self.host, self.limits)
        self.health = Mock(side_effect=lambda: health_idle())
        self.count = Mock(
            return_value={"prompt_tokens": 12, "count_source": "native_tokenize"}
        )
        self.m.backend = SimpleNamespace(
            health=self.health,
            request_token_count=self.count,
            identity=self.host.identity,
            ownership=self.host.ownership,
            profile=self.host.profile,
            boot_proof=self.host.boot_proof,
        )
        self.front.configure(
            {
                "upstream": f"http://127.0.0.1:{self.native.server_port}",
                "limits": self.limits,
            },
            self.m,
        )
        for server in (self.native, self.front):
            threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
            ).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        closed = threading.Event()
        self.addCleanup(closed.set)

        def telemetry():
            while not closed.is_set():
                with self.p.lock:
                    self.p.samples = {
                        r: {"epoch": time.time(), "available_gib": 8, "max_zone_c": 70}
                        for r in RANKS
                    }
                    self.p.lock.notify_all()
                closed.wait(0.01)

        threading.Thread(target=telemetry, daemon=True).start()

    def request(self, body=SHORT):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.front.server_port, timeout=8
        )
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def wait_idle(self):
        with self.p.lock:
            self.assertTrue(self.p.lock.wait_for(lambda: self.p.inflight == 0, 2))

    def burst(self):
        """Four simultaneous short requests; returns the native start spread in seconds."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request(), range(4)))
        self.assertEqual(results, [(200, TimedNative.payload)] * 4)
        self.wait_idle()
        return max(TimedNative.arrivals) - min(TimedNative.arrivals)


class StartSpreadTests(DispatchFixture, unittest.TestCase):
    """Simulated W1 stress: 4 concurrent short prompts, 2 body slots, 0.4 s per generation verification."""

    def test_start_spread_before_and_after(self):
        spreads = {}
        for name, cache, small in (
            ("v1", 0, 0),
            ("cache_only", 2.0, 0),
            ("release_only", 0, 262144),
            ("v2", 2.0, 262144),
        ):
            with self.subTest(name):
                self.limits.update(
                    generation_verify_cache_s=cache, small_body_release_bytes=small
                )
                self.m.verified, TimedNative.arrivals = None, []
                spreads[name] = self.burst()
        print("\nstart spread s:", {k: round(v, 3) for k, v in spreads.items()})
        self.assertGreater(
            spreads["v1"], 0.3
        )  # Reproduces the W1 M2 failure mechanism.
        self.assertLess(spreads["v2"], 0.3)
        self.assertLess(spreads["cache_only"], 0.3)
        self.assertLess(spreads["release_only"], 0.3)


class PermitAndOrderTests(DispatchFixture, unittest.TestCase):
    def hold_lane(self):
        """Keep admissions waiting so the request's body-slot use can be observed."""
        with self.p.lock:
            self.p.prefill_request = "other"

    def wait_waiters(self, count):
        with self.p.lock:
            self.assertTrue(
                self.p.lock.wait_for(lambda: len(self.p.admission_waiters) == count, 2)
            )

    def release_lane(self):
        with self.p.lock:
            self.p.prefill_request = None
            self.p.lock.notify_all()

    def slots_free(self):
        taken = []
        while self.front.buffers.acquire(blocking=False):
            taken.append(1)
        for _ in taken:
            self.front.buffers.release()
        return len(taken)

    def test_off_body_slot_is_held_through_admission(self):
        self.hold_lane()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.request)
            self.wait_waiters(1)
            self.assertEqual(self.slots_free(), 1)
            self.release_lane()
            self.assertEqual(future.result(timeout=3)[0], 200)
        self.assertEqual(self.slots_free(), 2)

    def test_small_body_releases_its_slot_before_admission(self):
        self.limits["small_body_release_bytes"] = len(SHORT)
        self.hold_lane()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.request) for _ in range(2)]
            self.wait_waiters(2)
            self.assertEqual(self.slots_free(), 2)
            self.release_lane()
            self.assertEqual([f.result(timeout=3)[0] for f in futures], [200, 200])
        self.assertEqual(self.slots_free(), 2)
        self.assertEqual([body for _, body in TimedNative.received], [SHORT, SHORT])

    def test_large_body_keeps_its_slot_until_dispatch(self):
        self.limits["small_body_release_bytes"] = len(SHORT) - 1
        self.hold_lane()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.request)
            self.wait_waiters(1)
            self.assertEqual(self.slots_free(), 1)
            self.release_lane()
            self.assertEqual(future.result(timeout=3)[0], 200)
        self.assertEqual(self.slots_free(), 2)

    def test_tokenize_and_verify_run_outside_the_serialized_health_gate(self):
        """Gate order: read body -> tokenize -> admit (one health probe at a time) -> verify -> dispatch."""
        self.limits.update(
            generation_verify_cache_s=2.0, small_body_release_bytes=262144
        )
        trace, probe_open = [], threading.Event()
        count, identity = self.count.return_value, self.host.identity

        def health():
            with self.p.lock:
                trace.append(("health", self.p.admission_probe))
            if not probe_open.is_set():
                probe_open.wait(
                    2
                )  # The first health probe stalls; the gate is held meanwhile.
            return health_idle()

        def tokenize(body, path):
            with self.p.lock:
                trace.append(("tokenize", self.p.admission_probe))
            return count

        def inspect(rank, sha, require_running=True):
            with self.p.lock:
                trace.append(("inspect", self.p.admission_probe))
            return identity(rank, sha, require_running)

        self.m.backend.health, self.m.backend.request_token_count = health, tokenize
        self.m.backend.identity = inspect
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.request)
            with self.p.lock:
                self.assertTrue(
                    self.p.lock.wait_for(lambda: self.p.admission_probe is not None, 2)
                )
            second = pool.submit(self.request)
            with self.p.lock:
                self.assertTrue(
                    self.p.lock.wait_for(
                        lambda: len([t for t in trace if t[0] == "tokenize"]) == 2, 2
                    )
                )
            # The second request was counted while the first held the health gate.
            self.assertIsNotNone(trace[-1][1])
            probe_open.set()
            self.assertEqual(
                [first.result(timeout=3)[0], second.result(timeout=3)[0]], [200, 200]
            )
        kinds = [kind for kind, _ in trace]
        self.assertEqual(kinds[0], "tokenize")
        self.assertLess(kinds.index("health"), kinds.index("inspect"))
        first_probe = trace[kinds.index("health")][1]
        # The first request's own health gate was released before its verification began.
        self.assertTrue(
            all(token != first_probe for kind, token in trace if kind == "inspect")
        )
        self.assertEqual(
            self.host.inspections, 2
        )  # The second request reused the first's inspection.

    def test_native_error_after_dispatch_invalidates(self):
        self.limits.update(generation_verify_cache_s=2.0)
        self.native.shutdown()
        self.native.server_close()
        generation = self.p.native_generation
        self.assertEqual(self.request()[0], 502)
        self.wait_idle()
        self.assertGreater(self.p.native_generation, generation)


class DefaultOffTests(DispatchFixture, unittest.TestCase):
    """New keys absent: per-request inspection, slot held to dispatch, same events and status as before."""

    def test_default_off_matches_previous_behaviour(self):
        self.limits["decode_admission_max_prompt_tokens"] = 0
        self.assertEqual(self.request()[0], 200)
        self.wait_idle()
        self.assertEqual(self.request()[0], 200)
        self.wait_idle()
        self.assertEqual(
            self.host.inspections, 4
        )  # Both ranks inspected on every request.
        self.assertIsNone(self.m.verified)
        events = [
            line for line in (Path(self.tmp.name) / "events").read_text().splitlines()
        ]
        self.assertTrue(all("native_generation" not in line for line in events))
        self.assertNotIn("native_generation", self.p.snapshot())


if __name__ == "__main__":
    unittest.main()
