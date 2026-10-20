"""Bounded HTTP front: FIFO memory-reservation admission, idle-only watchdog probes, cancellation, reconciliation."""

import hashlib
import hmac
import http.client
import http.server
import json
import select
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from common import MEMORY_MARGIN_GIB, RANKS

HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


class GeneratedTokenProbe:
    """Bounded SSE observer only; raw bytes are forwarded unchanged. Unknown output holds the lane."""
    def __init__(self):
        self.buffer = b""
        self.data = []
        self.size = 0
        self.disabled = False

    @staticmethod
    def generated(value):
        if not isinstance(value, dict) or not isinstance(value.get("choices"), list):
            return False
        for choice in value["choices"]:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                delta = {}
            candidates = [choice.get("text"), delta.get("content"), delta.get("reasoning_content")]
            tools = delta.get("tool_calls", [])
            if isinstance(tools, list):
                for tool in tools:
                    function = tool.get("function", {}) if isinstance(tool, dict) else {}
                    if isinstance(function, dict):
                        candidates.append(function.get("arguments"))
            function = delta.get("function_call", {})
            if isinstance(function, dict):
                candidates.append(function.get("arguments"))
            if any(isinstance(text, str) and len(text) > 0 for text in candidates):
                return True
        return False

    def feed(self, block):
        if self.disabled:
            return False
        # read1() is bounded to64KiB; incomplete/event buffering is bounded independently.
        self.buffer += block
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            self.size += len(line)
            if self.size > 65536:
                self.disabled = True
                break
            if not line:
                try:
                    if self.generated(json.loads(b"\n".join(self.data))):
                        return True
                except (ValueError, UnicodeError, RecursionError):
                    pass
                self.data, self.size = [], 0
            elif line.startswith(b"data:"):
                self.data.append(line[5:].removeprefix(b" "))
        if self.disabled or self.size + len(self.buffer) > 65536:
            self.disabled = True
            self.buffer, self.data = b"", []
        return False

class Front(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass

    def reply(self, code, value):
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_body(self):
        limit = self.server.config["limits"]["max_body_bytes"]
        if len(self.headers.get_all("Content-Length", [])) > 1 or (
            self.headers.get("Transfer-Encoding") and self.headers.get("Content-Length")
        ):
            raise ValueError("ambiguous request framing")
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            blocks, total = [], 0
            while True:
                line = self.rfile.readline(128)
                if not line.endswith(b"\n"):
                    raise ValueError("invalid chunk line")
                size = int(line.split(b";")[0], 16)
                if size < 0 or total + size > limit:
                    raise ValueError("body exceeds bound")
                if size == 0:
                    trailers = 0
                    while True:
                        trailer = self.rfile.readline(8193)
                        trailers += len(trailer)
                        if trailers > 8192:
                            raise ValueError("trailers exceed bound")
                        if trailer in (b"\r\n", b"\n", b""):
                            return b"".join(blocks)
                blocks.append(self.rfile.read(size))
                total += size
                if len(blocks[-1]) != size or self.rfile.read(2) != b"\r\n":
                    raise ValueError("invalid chunk framing")
        size = int(self.headers.get("Content-Length", 0))
        if not 0 <= size <= limit:
            raise ValueError("body exceeds bound")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError("incomplete body")
        return raw

    def token_count(self, body):
        """Bounded native /tokenize (3 s); any failure is a reason, never an exception."""
        try:
            return self.server.manager.backend.request_token_count(body, self.path), None
        except Exception as exc:
            return {}, type(exc).__name__

    def dispatch(self):
        manager, policy = self.server.manager, self.server.manager.policy
        if self.path == "/__thermal/status" and self.command == "GET":
            return self.reply(200, policy.snapshot())
        if self.path == "/__thermal/identity" and self.command == "GET":
            try:
                state = policy.snapshot()
                identities = {rank: manager.backend.identity(rank, state["profile"]) for rank in RANKS}
                names = (
                    "common.py",
                    "policy.py",
                    "backend.py",
                    "daemon.py",
                    "front.py",
                    "profile.py",
                    "sampler.py",
                    "lifecycle.py",
                    "control.py",
                )
                sources = {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names
                }
                return self.reply(
                    200,
                    {
                        "profile": state["profile"],
                        "identities": identities,
                        "native_health": manager.backend.health(),
                        "clock_mhz": manager.backend.actual_clock(),
                        "source_sha256": sources,
                        "config_sha256": hashlib.sha256(Path(manager.config["_path"]).read_bytes()).hexdigest(),
                        "epoch": time.time(),
                    },
                )
            except Exception as exc:
                return self.reply(503, {"error": str(exc)})
        if self.path.startswith("/__thermal/"):
            return self.reply(404, {"error": "no HTTP mutation API"})
        generation = self.command not in ("GET", "HEAD", "OPTIONS") and urlsplit(self.path).path not in (
            "/tokenize",
            "/v1/tokenize",
        )
        watchdog_probe = generation and self.headers.get("X-TF-Probe") == "watchdog"
        maintenance = self.server.config.get("maintenance_token_sha256")
        if generation and maintenance:
            token = self.headers.get("X-TF-Maintenance", "")
            if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), maintenance):
                self.close_connection = True
                if watchdog_probe:
                    return self.reply(429, {"code": "watchdog_deferred", "error": "operator verification window"})
                return self.reply(503, {"error": "operator verification window"})
        queued = buffered = admitted = complete = False
        upstream = None
        request_id = str(time.time_ns()) + "-" + str(threading.get_ident())
        received = 0
        response_status = None
        started = native_dispatched = False
        cancelled, watcher_done = threading.Event(), threading.Event()
        watcher = None

        def watch_disconnect():
            while not watcher_done.is_set():
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0.05)
                    if not readable:
                        continue
                    if self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                        # Never consume a pipelined request or spin on its readable bytes.
                        watcher_done.wait(0.05)
                        continue
                except BlockingIOError:
                    continue
                except OSError:
                    if watcher_done.is_set():
                        return
                cancelled.set()
                with policy.lock:
                    connection = policy.active.get(request_id)
                    if connection is not None:
                        try:
                            connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    policy.lock.notify_all()
                return
        try:
            if generation:
                queued = self.server.pending.acquire(blocking=False)
                if not queued:
                    self.close_connection = True
                    return self.reply(429 if watchdog_probe else 503,
                                      {"code": "watchdog_deferred", "error": "bounded generation queue full"})
            if (
                self.command in ("GET", "HEAD", "OPTIONS")
                and not self.headers.get("Transfer-Encoding")
                and (self.headers.get("Content-Length", "0") == "0")
            ):
                body = b""
            else:
                buffered = self.server.buffers.acquire(timeout=5)
                if not buffered:
                    self.close_connection = True
                    return self.reply(429 if watchdog_probe else 503,
                                      {"code": "watchdog_deferred", "error": "bounded body buffers busy"})
                body = self.read_body()
            body_sha = hashlib.sha256(body).hexdigest()
            if generation:
                watcher = threading.Thread(target=watch_disconnect, name="guard-client-disconnect", daemon=True)
                watcher.start()
                # Decode-class admission (off unless configured): a natively counted short text prompt skips the
                # prefill lane. Multimodal, unsupported routes, tokenize errors and probes keep the normal path.
                decode_limit = self.server.config["limits"].get("decode_admission_max_prompt_tokens", 0)
                counts = count_error = None
                if decode_limit and not watchdog_probe:
                    counts, count_error = self.token_count(body)
                decode = (counts is not None and counts.get("count_source") == "native_tokenize"
                          and type(counts.get("prompt_tokens")) is int and 0 <= counts["prompt_tokens"] <= decode_limit)
                admitted = self.server.admit(request_id, cancelled, probe=watchdog_probe, decode=decode)
                if not admitted:
                    self.close_connection = True
                    if watchdog_probe and not policy.data["terminal_stop"]:
                        return self.reply(429, {"code": "watchdog_deferred"})
                    return self.reply(503, {"error": "durable thermal STOP" if policy.data["terminal_stop"]
                                           else "thermal/memory queue deadline exceeded"})
                admission = {}
                if decode_limit:
                    admission = {"class": "decode" if decode else "prefill",
                                 "prompt_tokens": (counts or {}).get("prompt_tokens")}
                with policy.lock:
                    policy.event(
                        "REQUEST_ADMITTED",
                        time.time(),
                        request_id=request_id,
                        body_sha256=body_sha,
                        inflight=policy.inflight,
                        admitted_clock_mhz=policy.data["clock_mhz"],
                        **admission,
                    )
                manager.verify_generation()
                metadata = {"epoch": time.time(), "body_sha256": body_sha,
                            "prompt_tokens": None, "resumed_tokens": None, "count_source": "unavailable"}
                with policy.lock:
                    policy.request_metadata[request_id] = metadata
                if counts is None:
                    # Count once: reuse the pre-admission count (or its error) when decode admission is on.
                    counts, count_error = self.token_count(body)
                with policy.lock:
                    metadata.update(counts)
                    if count_error:
                        metadata["count_error"] = count_error
            upstream = http.client.HTTPConnection(
                "127.0.0.1",
                self.server.upstream.port,
                timeout=self.server.config["limits"].get("upstream_seconds", 1800),
            )
            if generation:
                with policy.lock:
                    if policy.data["hard_stop"] or cancelled.is_set():
                        raise OSError("STOP/disconnect before native connect")
                    policy.active[request_id] = None
            upstream.connect()
            if generation:
                with policy.lock:
                    if policy.data["hard_stop"] or cancelled.is_set():
                        raise OSError("STOP/disconnect during native connect")
                    policy.active[request_id] = upstream.sock
            headers = {key: value for key, value in self.headers.items()
                       if key.lower() not in HOP and key.lower() != "x-tf-maintenance"}
            headers["Content-Length"] = str(len(body))
            if generation:
                with policy.lock:
                    if (cancelled.is_set() or policy.thermal_wait_reason() or not policy.ready()
                            or set(policy.samples) != {"0", "1"}
                            or any(not -1 <= time.time() - row["epoch"] <= 6 for row in policy.samples.values())
                            or min(row["available_gib"] for row in policy.samples.values())
                            < MEMORY_MARGIN_GIB + sum(policy.memory_reservations.values())):
                        if watchdog_probe and not policy.data["terminal_stop"]:
                            return self.reply(429, {"code": "watchdog_deferred"})
                        raise ValueError("guard/telemetry changed before native dispatch")
                native_dispatched = True
            upstream.request(self.command, self.path, body, headers)
            body = b""
            if buffered:
                self.server.buffers.release()
                buffered = False
            response = upstream.getresponse()
            response_status = response.status
            token_probe = GeneratedTokenProbe() if (
                generation and 200 <= response.status < 300
                and response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() == "text/event-stream"
            ) else None
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            no_body = self.command == "HEAD" or response.status in (204, 304) or 100 <= response.status < 200
            if not no_body:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            started = True
            if not no_body:
                while True:
                    block = response.read1(65536)
                    if not block:
                        break
                    if token_probe is not None and token_probe.feed(block):
                        policy.release_prefill(request_id)
                        token_probe = None
                    received += len(block)
                    self.wfile.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                    self.wfile.flush()
                if response.length not in (None, 0):
                    raise http.client.IncompleteRead(b"", response.length)
                # A FIN after a fully framed native response cannot undo completion.
                # Identity-delimited EOF after our shutdown still requires idle recovery.
                complete = not cancelled.is_set() or response.length == 0 or response.chunked
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                complete = True
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if not started:
                try:
                    self.reply(
                        502,
                        {
                            "error": "guard/native transport refused",
                            "type": type(exc).__name__,
                        },
                    )
                except OSError:
                    pass
            self.close_connection = True
        finally:
            watcher_done.set()
            if watcher is not None:
                watcher.join(timeout=0.2)
            if upstream is not None:
                upstream.close()
            if buffered:
                self.server.buffers.release()
            if admitted:
                with policy.lock:
                    policy.release_prefill(request_id)
                    policy.active.pop(request_id, None)
                    policy.request_metadata.pop(request_id, None)
                    if native_dispatched and not complete:
                        # A closed socket does not prove the native request finished.
                        policy.unresolved_requests.add(request_id)
                    else:
                        policy.memory_reservations.pop(request_id, None)
                    policy.inflight -= 1
                    policy.event(
                        "REQUEST_FINISHED",
                        time.time(),
                        request_id=request_id,
                        completed=complete,
                        response_status=response_status,
                        streamed_bytes=received,
                    )
                    policy.lock.notify_all()
            if queued:
                self.server.pending.release()

    do_GET = dispatch
    do_HEAD = dispatch
    do_OPTIONS = dispatch
    do_POST = dispatch
    do_PUT = dispatch
    do_PATCH = dispatch
    do_DELETE = dispatch


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16
    allow_reuse_address = True

    def configure(self, config, manager):
        self.config, self.manager = config, manager
        self.upstream = urlsplit(config["upstream"])
        manager.policy.configure_admission(config["limits"])
        manager.policy.transient_reserve_gib = config["limits"].get("memory_transient_reserve_gib", 0.5)
        self.pending = threading.BoundedSemaphore(config["limits"]["max_pending"] + config["limits"]["max_inflight"])
        self.buffers = threading.BoundedSemaphore(config["limits"]["body_buffers"])
        self.connections = threading.BoundedSemaphore(config["limits"]["max_connections"])

    def admit(self, request_id, cancelled=None, probe=False, decode=False):
        """Atomic prefill lane + per-request transient budget, before inflight increment.

        decode=True (a natively counted short text prompt) neither takes the lane nor sets the post-token sample
        wait; it still waits behind a lane held by a normal request and honors that request's post-token wait.
        """
        policy = self.manager.policy
        limits = self.config["limits"]
        reserve = limits.get("memory_transient_reserve_gib", 0.5)
        deadline = time.monotonic() + limits.get("queue_seconds", 1200)
        admitted = False
        try:
            with policy.lock:
                # dispatch calls admit only after the complete body has been buffered.
                # Updating a reason below must never move this FIFO ticket.
                policy.admission_waiters.setdefault(request_id, "admission_fifo")
                policy.lock.notify_all()
            while time.monotonic() < deadline:
                with policy.lock:
                    if policy.data["terminal_stop"] or (cancelled is not None and cancelled.is_set()):
                        return False
                    unresolved = frozenset(policy.unresolved_requests)
                    recovering = bool(unresolved)
                    reason = policy.memory_wait_reason(reserve)
                    if probe and reason:
                        return False
                    if recovering:
                        reason = None if policy.inflight == 0 else "native_completion_unverified"
                    lane = not decode or recovering  # The idle proof of recovery keeps the lane path.
                    if policy.prefill_request is not None:
                        reason = "prefill_lane_busy"
                    elif policy.admission_probe is not None:
                        reason = "native_health_probe_busy"
                    elif policy.inflight >= limits["max_inflight"]:
                        reason = "inflight_limit"
                    other_waiters = any(key != request_id for key in policy.admission_waiters)
                    if probe and (reason or policy.inflight or other_waiters):
                        return False
                    if next(iter(policy.admission_waiters)) != request_id:
                        reason = "admission_fifo"
                    if reason:
                        policy.admission_waiters[request_id] = reason
                        policy.lock.wait(min(0.05, max(0, deadline - time.monotonic())))
                        continue
                    # A decode-class request holds only a probe token across the unlocked health I/O below, so
                    # one health observation is outstanding at a time, as with the lane. The token is cleared in
                    # the same lock hold that commits the decision (reservation + inflight increment, or refusal),
                    # so the next admission's health read and memory recheck already include this one; two
                    # admissions can never both act on one observation taken before either increment. The FIFO
                    # head rule orders this today; the token keeps it explicit and is never held past admission.
                    if lane:
                        policy.prefill_request = request_id
                    else:
                        policy.admission_probe = request_id
                    policy.admission_waiters[request_id] = "native_health_probe"
                native_ready = False
                health = {}
                try:
                    health = self.manager.backend.health()
                    # streams.prefilling is mislabeled queued; never use it as workspace evidence.
                    native_ready = (health.get("ok") is True and type(health.get("queued")) is int
                                    and health["queued"] == 0
                                    and all(type(health.get(key)) is int and health[key] >= 0
                                            for key in ("inflight", "requests_running")))
                except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
                    pass
                with policy.lock:
                    if policy.admission_probe == request_id:
                        policy.admission_probe = None
                    if policy.data["terminal_stop"] or (cancelled is not None and cancelled.is_set()):
                        return False
                    # A user may arrive while the low-priority health I/O is unlocked.
                    if probe and (policy.inflight or any(key != request_id for key in policy.admission_waiters)):
                        return False
                    if frozenset(policy.unresolved_requests) != unresolved:
                        # Cleanup during health I/O invalidates that observation, even if idle.
                        policy.admission_waiters[request_id] = "native_completion_changed_during_probe"
                    elif recovering:
                        idle = native_ready and all(type(health.get(key)) is int and health[key] == 0
                                                    for key in ("inflight", "requests_running"))
                        if idle and policy.inflight == 0:
                            for orphan in policy.unresolved_requests:
                                policy.memory_reservations.pop(orphan, None)
                            policy.unresolved_requests.clear()
                            policy.release_prefill(request_id)  # Require new telemetry after idle proof.
                        policy.admission_waiters[request_id] = "native_completion_unverified"
                    elif (native_ready and all(health[key] <= policy.inflight
                                               for key in ("inflight", "requests_running"))
                          and policy.memory_ready(reserve) and time.monotonic() < deadline):
                        policy.memory_reservations[request_id] = reserve
                        policy.inflight += 1
                        admitted = True
                        return True
                    if policy.prefill_request == request_id:
                        policy.prefill_request = None
                    if not recovering:
                        policy.admission_waiters[request_id] = "native_health_or_recheck_hold"
                    policy.lock.notify_all()
                    if probe:
                        return False
                    policy.lock.wait(min(0.05, max(0, deadline - time.monotonic())))
            return False
        finally:
            with policy.lock:
                policy.admission_waiters.pop(request_id, None)
                if not admitted and policy.prefill_request == request_id:
                    policy.prefill_request = None
                if policy.admission_probe == request_id:
                    policy.admission_probe = None
                policy.lock.notify_all()

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


