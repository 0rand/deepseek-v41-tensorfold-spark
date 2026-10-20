"""Durable thermal state machine; global pauses survive clock/profile/process changes."""

import json
import math
import threading
import time
import uuid
from pathlib import Path

from common import MEMORY_MARGIN_GIB, POLICY, atomic


class Policy:
    def __init__(self, path, events_path, ladder=(2100, 2000), initial_clock=None, initial_profile=None, now=None):
        self.path, self.events_path = Path(path), Path(events_path)
        self.ladder = tuple(ladder)
        self.lock = threading.Condition()
        self.inflight = 0
        self.active = {}
        self.request_metadata = {}
        self.prefill_request = None
        self.admission_probe = None  # Decode-class native health probe token; never held past admission.
        self.prefill_completed_epoch = 0
        self.prefill_sample_epochs = {}
        self.memory_reservations = {}
        self.admission_pause_c = self.admission_resume_c = None
        self.transient_reserve_gib = 0.5
        self.unresolved_requests = set()
        self.admission_waiters = {}
        self.native_generation = 0  # Bumped on any native doubt; keys the optional generation-verify cache.
        self.fresh = self.owned = False
        self.samples = {}
        now = time.time() if now is None else now
        if self.path.exists():
            self.data = json.loads(self.path.read_bytes())
            if self.data.get("version") != 1 or self.data.get("policy") != POLICY:
                raise ValueError("invalid durable policy; no automatic reset")
            if self.data.get("clock_ladder_mhz") != list(self.ladder):
                raise ValueError("durable clock ladder differs from the config; no automatic reset")
        else:
            initial_clock = self.ladder[0] if initial_clock is None else initial_clock
            if initial_clock not in self.ladder:
                raise ValueError("new state requires an idle start on a configured ladder cap")
            self.data = {
                "version": 1,
                "policy": POLICY,
                "clock_ladder_mhz": list(self.ladder),
                "clock_mhz": initial_clock,
                "profile": initial_profile,
                "paused": False,
                "transitioning": False,
                "terminal_stop": False,
                "hard_stop": False,
                "pending_action": None,
                "global_pauses": [],
                "consumed": [],
                "episodes": {"0": None, "1": None},
                "history": [],
                "bootstrap_profile": None,
                "last_update_epoch": now,
                "lease_mode": "unknown",
            }
            self.save()

    def configure_admission(self, limits):
        with self.lock:
            self.admission_pause_c = limits.get("prefill_admission_pause_c")
            self.admission_resume_c = limits.get("prefill_admission_resume_c")
            if self.admission_pause_c is not None:
                # A pre-feature/restarted state without a latch must first prove cool.
                self.data.setdefault("admission_cooldown", True)
                self.update_admission_cooldown(time.time())
                self.save()

    def update_admission_cooldown(self, now):
        if self.admission_pause_c is None:
            return
        before = self.data.get("admission_cooldown", True)
        after = before
        if any(row["max_zone_c"] >= self.admission_pause_c for row in self.samples.values()):
            after = True
        elif (self.fresh and set(self.samples) == {"0", "1"}
              and all(-1 <= now - row["epoch"] <= 6
                      and row["max_zone_c"] <= self.admission_resume_c for row in self.samples.values())):
            after = False
        self.data["admission_cooldown"] = after
        if before != after:
            self.event("ADMISSION_COOLDOWN_ENTER" if after else "ADMISSION_COOLDOWN_EXIT", now,
                       samples=self.samples, pause_c=self.admission_pause_c, resume_c=self.admission_resume_c)

    def thermal_wait_reason(self, now=None):
        if self.admission_pause_c is None:
            return None
        now = time.time() if now is None else now
        if not self.fresh or set(self.samples) != {"0", "1"} or any(
            not -1 <= now - row["epoch"] <= 6 for row in self.samples.values()
        ):
            return "admission_telemetry_stale_or_missing"
        if self.data.get("admission_cooldown", True):
            return "admission_thermal_cooldown"
        return None

    def save(self):
        atomic(self.path, self.data)

    def event(self, kind, now, **fields):
        row = {
            "utc_epoch": now,
            "event": kind,
            "clock_mhz": self.data["clock_mhz"],
            "profile": self.data["profile"],
            **fields,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path.exists() and self.events_path.stat().st_size >= 4 * 1024 * 1024:
            for index in range(3, 0, -1):
                older = Path(str(self.events_path) + "." + str(index))
                newer = Path(str(self.events_path) + "." + str(index + 1))
                if older.exists():
                    older.replace(newer)
            self.events_path.replace(Path(str(self.events_path) + ".1"))
        with self.events_path.open("a") as output:
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
        return row

    def abort(self):
        for connection in list(self.active.values()):
            try:
                if connection is not None:
                    connection.shutdown(2)
            except OSError:
                pass

    def terminal(self, reason, now, hard=False, stop_evidence=None):
        if self.data["terminal_stop"] and (not hard or self.data["hard_stop"]):
            return
        self.data.update(terminal_stop=True, paused=True, hard_stop=hard or self.data["hard_stop"])
        self.native_changed()
        self.data["stop_reason"] = reason
        if hard:
            self.data["pending_action"] = "stop_owned"
        if stop_evidence is not None and "stop_evidence" not in self.data:
            self.data["stop_evidence"] = stop_evidence
        self.save()  # Durable latch precedes cancellation; never authorize watchdog restart after reboot.
        if hard:
            self.abort()
        evidence = {"stop_evidence": self.data["stop_evidence"]} if "stop_evidence" in self.data else {}
        self.event("HARD_STOP" if hard else "FLOOR_STOP", now, reason=reason, inflight=self.inflight, **evidence)
        self.lock.notify_all()

    def native_changed(self):
        """A failed health read, a native error or an unverified completion: a cached identity is no longer trusted."""
        with self.lock:
            self.native_generation += 1

    def condition(self, token, reason, now):
        if token in self.data["consumed"] or self.data["terminal_stop"]:
            return
        self.data["consumed"].append(token)
        if self.data["transitioning"]:
            self.event("condition_consumed_during_handoff", now, condition=token, reason=reason)
            return
        if self.data["clock_mhz"] <= self.ladder[-1]:
            self.terminal(reason, now)
            return
        self.data.update(paused=True, transitioning=True, pending_action="lower_clock", condition=token)
        self.event("STEPDOWN_REQUESTED", now, reason=reason, condition=token, inflight=self.inflight)

    def update(self, samples, owned, lease_mode, now=None):
        now = time.time() if now is None else now
        with self.lock:
            old_update = self.data["last_update_epoch"]
            self.data.update(last_update_epoch=now, lease_mode=lease_mode)
            self.owned = owned
            valid = {}
            for rank, sample in samples.items():
                try:
                    age = now - float(sample["epoch"])
                    temp, memory = float(sample["max_zone_c"]), float(sample["available_gib"])
                    if (
                        rank in ("0", "1")
                        and -1 <= age <= 6
                        and all(map(math.isfinite, (temp, memory)))
                        and 0 < temp < 150
                    ):
                        valid[rank] = {**sample, "max_zone_c": temp, "available_gib": memory}
                except (ValueError, TypeError, KeyError):
                    continue
            self.samples = valid
            self.fresh = set(valid) == {"0", "1"}
            self.data["global_pauses"] = [row for row in self.data["global_pauses"] if row["epoch"] >= now - 1800]
            if not owned:
                self.data["episodes"] = {"0": None, "1": None}
                self.data["fresh"] = self.fresh
                self.update_admission_cooldown(now)
                self.save()
                self.lock.notify_all()
                return
            fired = []
            for rank, row in sorted(valid.items()):
                if row["max_zone_c"] >= 94:
                    fired.append({"condition": "temperature>=94", "rank": rank, "value": row["max_zone_c"]})
                if row["available_gib"] < 4:
                    fired.append({"condition": "memory<4GiB", "rank": rank, "value": row["available_gib"]})
            if fired:
                # Telemetry can include rejected NaN/Infinity values. Retain those raw
                # values as explicit strings so evidence cannot prevent the durable STOP.
                evidence = json.loads(
                    json.dumps({
                        "epoch": now,
                        "raw_samples": samples,
                        "validated_samples": valid,
                        "fired": fired,
                        "inflight": self.inflight,
                        "request_metadata": self.request_metadata,
                    }),
                    parse_constant=str,
                )
                self.terminal("fresh temperature>=94 or memory<4GiB", now, hard=True, stop_evidence=evidence)
            if now - old_update > 6 or now < old_update:
                self.data["episodes"] = {"0": None, "1": None}
            if not self.data["terminal_stop"]:
                if valid and max(row["max_zone_c"] for row in valid.values()) >= 92 and not self.data["paused"]:
                    pause = {"id": uuid.uuid4().hex, "epoch": now}
                    self.data["global_pauses"].append(pause)
                    self.data["paused"] = True
                    self.event("PAUSE", now, samples=valid, global_count=len(self.data["global_pauses"]))
                    if len(self.data["global_pauses"]) > 3:
                        self.condition("storm:" + pause["id"], "new fourth-or-later GLOBAL pause within 1800 s", now)
                for rank in ("0", "1"):
                    row = valid.get(rank)
                    if row is None or row["max_zone_c"] < 90:
                        self.data["episodes"][rank] = None
                        continue
                    if self.data["episodes"][rank] is None:
                        self.data["episodes"][rank] = {"id": uuid.uuid4().hex, "start": now}
                    episode = self.data["episodes"][rank]
                    if now - episode["start"] >= 600:
                        reason = "temperature>=90 continuously 600 s"
                        self.condition("sustain:" + rank + ":" + episode["id"], reason, now)
                if (
                    self.fresh
                    and self.data["paused"]
                    and not self.data["transitioning"]
                    and max(row["max_zone_c"] for row in valid.values()) <= 88
                ):
                    self.data["paused"] = False
                    self.event("RESUME", now, samples=valid)
            referenced = {"storm:" + row["id"] for row in self.data["global_pauses"]}
            referenced.update("sustain:" + rank + ":" + row["id"] for rank, row in self.data["episodes"].items() if row)
            self.data["consumed"] = [token for token in self.data["consumed"] if token in referenced]
            self.data["fresh"] = self.fresh
            # Hard STOP processing above precedes optional pacing/event I/O.
            self.update_admission_cooldown(now)
            self.save()
            self.lock.notify_all()

    def ready(self):
        return (
            self.owned
            and self.data.get("clock_verified", True)
            and self.fresh
            and not self.data["paused"]
            and not self.data["transitioning"]
            and not self.data["terminal_stop"]
            and self.data["bootstrap_profile"] is None
            and not self.data.get("operator_hold", False)
        )

    def memory_wait_reason(self, reserve=0.5, now=None):
        """Conservative per-request budget, retained until completion; not native workspace proof."""
        now = time.time() if now is None else now
        if not self.fresh or set(self.samples) != {"0", "1"} or any(
            not -1 <= now - row["epoch"] <= 6 for row in self.samples.values()
        ):
            return "telemetry_stale_or_missing"
        if self.thermal_wait_reason(now):
            return self.thermal_wait_reason(now)
        if not self.ready():
            return "thermal_or_ownership_hold"
        if any(row["epoch"] <= max(self.prefill_completed_epoch, self.prefill_sample_epochs.get(rank, 0))
               for rank, row in self.samples.items()):
            return "waiting_for_post_token_sample"
        if min(row["available_gib"] for row in self.samples.values()) < (
            MEMORY_MARGIN_GIB + sum(self.memory_reservations.values()) + reserve
        ):
            return "memory_margin_and_retained_reservations"
        return None

    def memory_ready(self, reserve=0.5, now=None):
        return self.memory_wait_reason(reserve, now) is None

    def release_prefill(self, request_id):
        # Positive generated delta, or transport cleanup; reservations remain until completion.
        with self.lock:
            if self.prefill_request == request_id:
                self.prefill_completed_epoch = time.time()
                self.prefill_sample_epochs = {rank: row["epoch"] for rank, row in self.samples.items()}
                self.prefill_request = None
                self.lock.notify_all()

    def clock_changed(self, receipt, now=None):
        now = time.time() if now is None else now
        with self.lock:
            current = self.data["clock_mhz"]
            if (current not in self.ladder[:-1] or not self.data["transitioning"] or self.inflight
                    or self.data["terminal_stop"] or self.data["pending_action"] != "lower_clock"):
                raise ValueError("clock receipt requires pending idle downward transition")
            target = self.ladder[self.ladder.index(current) + 1]
            self.data["clock_mhz"] = target
            self.data["pending_action"] = None
            self.data["transitioning"] = False
            # GLOBAL ring and consumed IDs deliberately remain untouched.
            self.data["history"].append({"event": "clock_lowered", "epoch": now, "condition": self.data["condition"]})
            self.data["history"] = self.data["history"][-128:]
            self.save()
            self.event("CLOCK_LOWERED", now, receipt=receipt, global_pauses=self.data["global_pauses"])
            self.lock.notify_all()

    def snapshot(self):
        with self.lock:
            return {
                **json.loads(json.dumps(self.data)),
                "thermal_wait_reason": self.thermal_wait_reason(),
                "admission_cooldown": (self.admission_pause_c is not None
                                       and self.data.get("admission_cooldown", True)),
                "prefill_admission_pause_c": self.admission_pause_c,
                "prefill_admission_resume_c": self.admission_resume_c,
                "inflight": self.inflight,
                "prefill_request": self.prefill_request,
                "admission_waiters": dict(self.admission_waiters),
                "native_completion_unverified": sorted(self.unresolved_requests),
                "memory_reservations_gib": dict(self.memory_reservations),
                "memory_reserved_total_gib": sum(self.memory_reservations.values()),
                "memory_transient_reserve_gib": self.transient_reserve_gib,
                "memory_margin_gib": MEMORY_MARGIN_GIB,
                "memory_wait_reason": self.memory_wait_reason(self.transient_reserve_gib),
                "prefill_completed_epoch": self.prefill_completed_epoch,
                "owned": self.owned,
                "fresh": self.fresh,
                "samples": dict(self.samples),
                "ready": self.ready(),
                "active_upstreams": len(self.active),
            }
