"""Bounded durable storage and strict parameterized deployment configuration."""

import fcntl
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

# The fixed STOP / pause / step-down thresholds; a config must restate them exactly (no silent threshold change).
POLICY = {
    "step_c": 90,
    "step_seconds": 600,
    "pause_c": 92,
    "resume_c": 88,
    "pause_window_seconds": 1800,
    "pause_limit": 3,
    "hard_c": 94,
    "memory_floor_gib": 4,
}
# Admission keeps this much above the 4 GiB hard floor before any per-request reservation.
MEMORY_MARGIN_GIB = 4.25
POOL_KEY = "TF_DSV41_POOL_TOKENS"
FLOOR_KEY = "TF_DSV41_PF_WS_FLOOR_GIB"
SCHEDULE_KEY = "TF_CUDA_SCHED"
PLAN_KEY = "TF_DSV41_PLAN_PIN"
SCHEDULES = (None, "auto", "spin", "yield", "blocking")
CPU_SET = re.compile(r"(0|[1-9][0-9]{0,3})(-(0|[1-9][0-9]{0,3}))?(,(0|[1-9][0-9]{0,3})(-(0|[1-9][0-9]{0,3}))?)*")
RANKS = ("0", "1")


def cpu_set_members(text):
    members = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        low, high = int(low), int(high or low)
        if low > high or high > 4095:
            raise ValueError("invalid host CPU set range")
        members.update(range(low, high + 1))
    return members


def validate_ladder(ladder):
    """Explicit operator clock ladder: strictly descending caps; the last one is the floor."""
    if (not isinstance(ladder, list) or not 2 <= len(ladder) <= 4
            or any(type(cap) is not int or not 1000 <= cap <= 3000 for cap in ladder)
            or any(a <= b for a, b in zip(ladder, ladder[1:]))):
        raise ValueError("clock ladder must be 2..4 strictly descending integer MHz caps within 1000..3000")
    return tuple(ladder)


def validate_profiles(profiles):
    """Each snapshotted profile: exact per-rank identity, rank-equal qualified knobs, floor not under the hard floor."""
    if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 32:
        raise ValueError("1..32 exact profiles required")
    for sha, profile in profiles.items():
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("exact profile config SHA required")
        knobs = []
        sets = profile.get("cpu_sets", {})
        if not isinstance(sets, dict) or not set(sets) <= set(RANKS) or any(type(v) is not str for v in sets.values()):
            raise ValueError("host CPU sets must map rank 0/1 to cpuset strings")
        for rank in RANKS:
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", profile["image_ids"][rank]):
                raise ValueError("exact per-node image ID required")
            if not profile["mounts"][rank].get("/model") or not profile["inference_environment"][rank]:
                raise ValueError("exact mounts and full inference environment required")
            env = profile["inference_environment"][rank]
            floor, schedule, plan = env.get(FLOOR_KEY), env.get(SCHEDULE_KEY), env.get(PLAN_KEY)
            if floor is not None:
                try:
                    value = float(floor)
                except ValueError:
                    raise ValueError("workspace floor must be a number") from None
                if not math.isfinite(value) or value < POLICY["memory_floor_gib"]:
                    raise ValueError("workspace floor below the 4 GiB hard floor can plan a STOP")
            if schedule not in SCHEDULES:
                raise ValueError("TF_CUDA_SCHED must be auto, spin, yield or blocking")
            cpus = sets.get(rank, "")
            if cpus and not CPU_SET.fullmatch(cpus):
                raise ValueError("host CPU set must be canonical cpuset syntax")
            if cpus:
                cpu_set_members(cpus)  # Reversed or out-of-range ranges are not canonical either.
            if plan not in (None, "auto"):
                if not isinstance(plan, str) or not re.fullmatch(r"0|[1-9][0-9]{0,3}", plan) or int(plan) > 4095:
                    raise ValueError("plan CPU must be auto or a canonical local CPU index 0..4095")
                if cpus and int(plan) not in cpu_set_members(cpus):
                    raise ValueError("plan CPU must lie inside the rank's host CPU set")
            knobs.append((env.get(POOL_KEY), floor, schedule))
        if knobs[0] != knobs[1]:
            raise ValueError("pool, workspace floor and host scheduling must match on both ranks")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, separators=(",", ":"), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    config = json.loads(Path(path).read_bytes())
    if config.get("enabled") is not True or config.get("mode") != "production" or config.get("policy") != POLICY:
        raise ValueError("explicitly enabled exact production policy required")
    config["clock_ladder_mhz"] = list(validate_ladder(config.get("clock_ladder_mhz")))
    for name in ("state_path", "samples_path", "events_path", "lease_path", "production_config", "control_socket",
                 "clock_dropin"):
        value = Path(config[name])
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("absolute canonical path required: " + name)
    if not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", config["clock_service"]):
        raise ValueError("invalid clock cap service name")
    units = config.get("watchdog_units", [])
    if not isinstance(units, list) or any(not re.fullmatch(r"[A-Za-z0-9@_.-]+\.timer", u) for u in units):
        raise ValueError("watchdog units must be systemd timer names")
    upstream = urlsplit(config["upstream"])
    if upstream.scheme != "http" or upstream.hostname != "127.0.0.1" or not upstream.port:
        raise ValueError("native upstream must be loopback http with an explicit port")
    if config.get("listen_host") != "127.0.0.1" or type(config.get("listen_port")) is not int:
        raise ValueError("front must listen on loopback with an explicit port")
    if config["listen_port"] == upstream.port:
        raise ValueError("front and native ports must differ")
    for rank in RANKS:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["containers"][rank]):
            raise ValueError("invalid container name")
    validate_profiles(config["profiles"])
    limits = config["limits"]
    for name, lower, upper in [
        ("max_inflight", 1, 4),
        ("max_pending", 1, 16),
        ("max_connections", 8, 64),
        ("max_body_bytes", 1, 64 * 1024 * 1024),
        ("body_buffers", 1, 4),
    ]:
        if type(limits[name]) is not int or not lower <= limits[name] <= upper:
            raise ValueError("invalid bounded limit: " + name)
    reserve = limits.get("memory_transient_reserve_gib", 0.5)
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or not 0.5 <= reserve <= 4:
        raise ValueError("transient reserve must be finite 0.5..4 GiB above 4.25 GiB")
    pause = limits.get("prefill_admission_pause_c")
    resume = limits.get("prefill_admission_resume_c")
    if "prefill_admission_pause_c" in limits or "prefill_admission_resume_c" in limits:
        if type(pause) is not int or type(resume) is not int or not 1 <= resume < pause <= 88:
            raise ValueError("admission temperatures require integer 1<=resume<pause<=88; both required")
    seconds = limits.get("queue_seconds", 1200)
    if type(seconds) is not int or not 1 <= seconds <= 1200:
        raise ValueError("queue timeout must be 1..1200 s")
    decode = limits.get("decode_admission_max_prompt_tokens", 0)
    if type(decode) is not int or not 0 <= decode <= 2048:
        raise ValueError("decode admission prompt limit must be an integer 1..2048 tokens (absent or 0 = off)")
    cache = limits.get("generation_verify_cache_s", 0)
    if type(cache) not in (int, float) or not math.isfinite(cache) or not 0 <= cache <= 5:
        raise ValueError("generation verify cache must be finite 0..5 s (absent or 0 = off)")
    small = limits.get("small_body_release_bytes", 0)
    if type(small) is not int or not 0 <= small <= min(1024 * 1024, limits["max_body_bytes"]):
        raise ValueError("small body release must be an integer 0..min(1 MiB, max_body_bytes) (absent or 0 = off)")
    period = config.get("sampler_period_s", 2.0)
    if type(period) not in (int, float) or not math.isfinite(period) or not 0.25 <= period <= 2.0:
        raise ValueError("sampler period must be finite 0.25..2.0 s")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", config["worker_ssh"]) or config["worker_ssh"].startswith("-"):
        raise ValueError("invalid worker SSH target")
    return config


def daemon_lock(state_path):
    path = Path(str(state_path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream
