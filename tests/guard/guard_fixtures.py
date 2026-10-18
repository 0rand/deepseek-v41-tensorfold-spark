"""Shared synthetic fixtures: profiles, configs, a fake native SSE server and policy / admission harnesses."""

import copy
import hashlib
import http.client
import http.server
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from common import FLOOR_KEY, PLAN_KEY, POLICY, POOL_KEY, SCHEDULE_KEY
from front import Front, Server
from policy import Policy

RANKS = ("0", "1")
LADDER3 = (2200, 2100, 2000)
CPU_SETS = {"0": "2-5", "1": "2-5"}
FIXTURE_KEY = "TF_DSV41_FIXTURE"


def synthetic_sha(label):
    """A deterministic 64-hex stand-in for a serving env file's SHA-256."""
    return hashlib.sha256(("guard-fixture:" + str(label)).encode()).hexdigest()


def image_id(label):
    return "sha256:" + synthetic_sha("image:" + str(label))


def make_profile(label="base", cpu_sets=None, **env):
    """A valid two-rank profile; keyword arguments override the inference environment on both ranks."""
    base = {POOL_KEY: "2100000", FLOOR_KEY: "4.75", SCHEDULE_KEY: "blocking", PLAN_KEY: "auto", FIXTURE_KEY: "1"}
    base.update(env)
    profile = {
        "image_ids": {rank: image_id(label + rank) for rank in RANKS},
        "mounts": {rank: {"/model": "/srv/model-" + label} for rank in RANKS},
        "inference_environment": {rank: dict(base) for rank in RANKS},
    }
    if cpu_sets is not None:
        profile["cpu_sets"] = copy.deepcopy(cpu_sets)
    return profile


def make_profiles(*labels):
    labels = labels or ("base",)
    return {synthetic_sha(label): make_profile(label) for label in labels}


def backend_config(**extra):
    """Enough configuration for Backend methods that never reach a real host."""
    config = {
        "upstream": "http://127.0.0.1:8001",
        "containers": {"0": "native-r0", "1": "native-r1"},
        "clock_ladder_mhz": [2100, 2000],
        "clock_service": "gpu-clock-cap.service",
        "clock_dropin": "/etc/systemd/system/gpu-clock-cap.service.d/cap.conf",
        "limits": {"max_inflight": 4},
    }
    config.update(extra)
    return config


def production_config(root, **overrides):
    """A complete config dict that load_config accepts, with every path under the temporary root."""
    root = Path(root)
    config = {
        "enabled": True,
        "mode": "production",
        "policy": copy.deepcopy(POLICY),
        "clock_ladder_mhz": list(LADDER3),
        "clock_service": "gpu-clock-cap.service",
        "clock_dropin": str(root / "cap.conf"),
        "watchdog_units": ["native-watch.timer"],
        "containers": {"0": "native-r0", "1": "native-r1"},
        "upstream": "http://127.0.0.1:8001",
        "listen_host": "127.0.0.1",
        "listen_port": 8000,
        "worker_ssh": "worker",
        "profiles": make_profiles("base", "next"),
        "limits": {"max_inflight": 4, "max_pending": 8, "max_connections": 32, "max_body_bytes": 1024,
                   "body_buffers": 2, "prefill_admission_pause_c": 82, "prefill_admission_resume_c": 75},
    }
    for name in ("state_path", "samples_path", "events_path", "lease_path", "production_config", "control_socket"):
        config[name] = str(root / name)
    config.update(overrides)
    return config


def write_config(root, config):
    path = Path(root) / "config.json"
    path.write_text(json.dumps(config))
    return path


def health_idle():
    return {"ok": True, "queued": 0, "inflight": 0, "requests_running": 0}


class Native(http.server.BaseHTTPRequestHandler):
    """Fake native engine: one SSE token, an optional hold, a second token; counts overlap."""

    protocol_version = "HTTP/1.1"
    lock = threading.Lock()
    active = maximum = 0
    received = []
    finish_barrier = None
    first = b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
    last = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
    payload = first + last

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with self.lock:
            type(self).active += 1
            type(self).maximum = max(type(self).maximum, type(self).active)
            type(self).received.append((self.headers.get("Authorization"), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.first)
        self.wfile.flush()
        if self.finish_barrier is None:
            time.sleep(0.05)
        else:
            self.finish_barrier.wait(timeout=2)
        self.wfile.write(self.last)
        with self.lock:
            type(self).active -= 1


class TransportFixture:
    """A real front Server in front of the fake Native server, with fresh cool telemetry every 10 ms."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.p = Policy(root / "state", root / "events")
        self.p.owned = self.p.fresh = True
        self.native = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Native)
        Native.maximum = Native.active = 0
        Native.received = []
        Native.finish_barrier = None
        self.front = Server(("127.0.0.1", 0), Front)
        limits = dict(max_pending=8, max_inflight=4, body_buffers=4, max_connections=32, max_body_bytes=1024,
                      queue_seconds=1, upstream_seconds=2)
        manager = SimpleNamespace(policy=self.p, verify_generation=lambda: None,
                                  backend=SimpleNamespace(health=health_idle))
        telemetry_closed = threading.Event()
        self.addCleanup(telemetry_closed.set)

        def telemetry():
            while not telemetry_closed.is_set():
                with self.p.lock:
                    self.p.samples = {rank: dict(epoch=time.time(), available_gib=8, max_zone_c=80) for rank in RANKS}
                    self.p.lock.notify_all()
                telemetry_closed.wait(0.01)

        threading.Thread(target=telemetry, daemon=True).start()
        self.front.configure(dict(upstream=f"http://127.0.0.1:{self.native.server_port}", limits=limits), manager)
        for server in (self.native, self.front):
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def request(self, method="POST", body=b'{"image":"raw","tools":[1]}', headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=4)
        try:
            connection.request(method, "/v1/chat/completions", body if method == "POST" else None,
                               {"Authorization": "Bearer exact", **(headers or {})})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def wait_idle(self, timeout=0.5):
        with self.p.lock:
            return self.p.lock.wait_for(lambda: self.p.inflight == 0, timeout=timeout)


class LadderFixture:
    """A durable policy on a three-cap ladder started at the top cap, plus a bare Server for admit()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sha = synthetic_sha("base")
        self.p = Policy(self.root / "state", self.root / "events", ladder=LADDER3, initial_profile=self.sha)

    def reload(self):
        return Policy(self.root / "state", self.root / "events", ladder=LADDER3)

    def samples(self, memory=8, epoch=None):
        epoch = time.time() if epoch is None else epoch
        self.p.update({rank: dict(epoch=epoch, max_zone_c=80, available_gib=memory) for rank in RANKS},
                      True, "owned", epoch)

    def temperatures(self, a, b, epoch=None):
        now = time.time() if epoch is None else epoch
        self.p.update({r: dict(epoch=now, max_zone_c=t, available_gib=8) for r, t in (("0", a), ("1", b))},
                      True, "owned")

    def server(self):
        health = Mock(return_value=health_idle())
        server = object.__new__(Server)
        server.config = {"limits": {"max_inflight": 4, "queue_seconds": 0.2}}
        server.manager = SimpleNamespace(policy=self.p, backend=SimpleNamespace(health=health))
        return server, health


class ProbeFixture:
    """An owned, fresh, cool policy and a bare Server whose native health is a Mock."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.p = Policy(root / "state", root / "events")
        self.p.owned = self.p.fresh = True
        self.p.samples = {rank: dict(epoch=time.time(), available_gib=8, max_zone_c=70) for rank in RANKS}
        self.health = Mock(return_value=health_idle())
        self.server = object.__new__(Server)
        self.server.config = {"limits": {"max_inflight": 4, "queue_seconds": 10}}
        self.server.manager = SimpleNamespace(policy=self.p, backend=SimpleNamespace(health=self.health))
