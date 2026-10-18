"""Operator maintenance lease ownership, watchdog holds and running-pair profile snapshots."""

import hashlib
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import Backend
from common import FLOOR_KEY
from daemon import Manager
from guard_fixtures import CPU_SETS, backend_config, make_profile, production_config, synthetic_sha
from lifecycle import decision
from policy import Policy

GUARD = Path(__file__).resolve().parents[2] / "scripts" / "guard"


def guard_profile_module():
    # Loaded by path: the guard's profile.py would otherwise collide with the standard library's profile module.
    spec = importlib.util.spec_from_file_location("guard_profile", GUARD / "profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lease = self.root / "maintenance.lease"
        self.backend = Backend(backend_config(lease_path=str(self.lease)))

    def age(self, minutes):
        stamp = time.time() - minutes * 60
        os.utime(self.lease, (stamp, stamp))

    def test_no_lease_is_owned(self):
        self.assertEqual(self.backend.ownership(), (True, "owned"))

    def test_fresh_lease_stands_the_guard_aside_until_it_ages_out(self):
        self.lease.write_text("")
        self.assertEqual(self.backend.ownership(), (False, "operator_lease"))
        self.age(19.9)
        self.assertEqual(self.backend.ownership(), (False, "operator_lease"))
        self.age(20.1)
        self.assertEqual(self.backend.ownership(), (True, "owned"))
        self.backend.config["lease_minutes"] = 30
        self.assertEqual(self.backend.ownership(), (False, "operator_lease"))
        self.lease.touch()
        self.backend.config["lease_minutes"] = 20
        self.assertEqual(self.backend.ownership(), (False, "operator_lease"))

    def test_nonregular_lease_is_refused_not_trusted(self):
        target = self.root / "elsewhere"
        target.write_text("")
        self.lease.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "nonregular"):
            self.backend.ownership()
        self.lease.unlink()
        self.lease.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "nonregular"):
            self.backend.ownership()
        self.lease.unlink()
        self.lease.mkdir()
        with self.assertRaisesRegex(ValueError, "nonregular"):
            self.backend.ownership()

    def test_lease_blocks_identity_and_watchdog_hold(self):
        self.backend.profile = lambda: ("p", make_profile())
        self.lease.write_text("")
        with self.assertRaisesRegex(ValueError, "operator_lease"):
            self.backend.identity("0", "p")
        with patch("backend.subprocess.run") as run, self.assertRaisesRegex(ValueError, "ownership changed"):
            self.backend.hold_heals("p")
        run.assert_not_called()

    def test_hold_heals_stops_configured_timers_or_nothing(self):
        self.backend.profile = lambda: ("p", make_profile())
        with patch("backend.subprocess.run") as run:
            self.backend.hold_heals("p")
            run.assert_not_called()
            self.backend.config["watchdog_units"] = ["a.timer", "b.timer"]
            self.backend.hold_heals("p")
            self.assertEqual(run.call_args.args[0], ["systemctl", "--user", "stop", "a.timer", "b.timer"])
            with self.assertRaisesRegex(ValueError, "ownership changed"):
                self.backend.hold_heals("other")

    def test_lifecycle_holds_under_lease_even_with_a_durable_stop(self):
        backend = SimpleNamespace(ownership=lambda: (False, "operator_lease"), profile=Mock())
        for state in ({"version": 1, "hard_stop": True}, {"version": 1, "terminal_stop": True}, {}):
            code, reason = decision({}, state, "start", backend, 1000)
            self.assertEqual((code, reason), (75, "foreign/unowned lifecycle: operator_lease"))
        backend.profile.assert_not_called()


class ManagerOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = production_config(self.root)
        self.sha = next(iter(self.config["profiles"]))
        self.leased = False
        self.backend = SimpleNamespace(
            profile=lambda: (self.sha, {}), actual_clock=lambda: 2100,
            ownership=lambda: (False, "operator_lease") if self.leased else (True, "owned"),
        )

    def test_new_durable_state_only_while_owned(self):
        self.leased = True
        with self.assertRaisesRegex(ValueError, "operator_lease"):
            Manager(self.config, backend=self.backend)
        self.assertFalse(Path(self.config["state_path"]).exists())
        self.leased = False
        manager = Manager(self.config, backend=self.backend)
        self.assertEqual(manager.policy.data["clock_mhz"], 2100)
        self.assertEqual(manager.policy.data["clock_ladder_mhz"], [2200, 2100, 2000])
        self.assertTrue(manager.policy.data["clock_verified"])
        # An existing state is reloaded under a lease; the lease only withholds ownership.
        self.leased = True
        self.assertEqual(Manager(self.config, backend=self.backend).policy.data["profile"], self.sha)

    def test_new_state_refuses_an_actual_cap_off_the_ladder(self):
        self.backend.actual_clock = lambda: 2150
        with self.assertRaisesRegex(ValueError, "ladder cap"):
            Manager(self.config, backend=self.backend)

    def test_tick_under_lease_or_changed_profile_is_unowned(self):
        manager = Manager(self.config, backend=self.backend)
        now = time.time()
        Path(self.config["samples_path"]).write_text(json.dumps({"samples": {
            rank: {"epoch": now, "max_zone_c": 99, "available_gib": 1} for rank in ("0", "1")}}))
        self.leased = True
        manager.tick()
        self.assertFalse(manager.policy.owned)
        self.assertEqual(manager.policy.data["lease_mode"], "operator_lease")
        self.assertFalse(manager.policy.data["hard_stop"])
        self.leased = False
        self.backend.profile = lambda: (synthetic_sha("changed"), {})
        manager.tick()
        self.assertEqual(manager.policy.data["lease_mode"], "profile_mismatch")
        self.assertFalse(manager.policy.data["hard_stop"])
        self.backend.profile = lambda: (self.sha, {})
        manager.tick()
        self.assertTrue(manager.policy.data["hard_stop"])

    def test_actions_skip_everything_while_unowned(self):
        manager = Manager(self.config, backend=self.backend)
        with manager.policy.lock:
            manager.policy.terminal("stop", time.time(), hard=True)
        manager.backend.stop_owned = Mock()
        manager.policy.owned = False
        manager.act()
        manager.backend.stop_owned.assert_not_called()


class SnapshotTests(unittest.TestCase):
    def inspect(self, profile, rank, running=True):
        env = [f"{k}={v}" for k, v in profile["inference_environment"][rank].items()]
        return [{"Image": profile["image_ids"][rank], "State": {"Running": running},
                 "HostConfig": {"CpusetCpus": profile.get("cpu_sets", {}).get(rank, "")},
                 "Mounts": [{"Destination": d, "Source": s} for d, s in profile["mounts"][rank].items()],
                 "Config": {"Env": env + ["PATH=/usr/bin", "TF_DSV41_LAUNCH_T0=1", "OTHER=1"]}}]

    def test_snapshot_keys_profile_by_env_file_sha_and_validates_it(self):
        profile_module = guard_profile_module()
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "serving.env"
            env_file.write_text("TF_DSV41_POOL_TOKENS=2100000\n")
            config = {"production_config": str(env_file), "containers": {"0": "r0", "1": "r1"}}
            for expected in (make_profile(), make_profile(cpu_sets=CPU_SETS)):
                backend = SimpleNamespace(run=lambda rank, _, value=expected: json.dumps(self.inspect(value, rank)))
                sha, profile = profile_module.snapshot(config, backend)
                self.assertEqual(sha, hashlib.sha256(env_file.read_bytes()).hexdigest())
                self.assertEqual(profile, expected)
            low = make_profile(**{FLOOR_KEY: "3.5"})
            backend = SimpleNamespace(run=lambda rank, _: json.dumps(self.inspect(low, rank)))
            with self.assertRaisesRegex(ValueError, "floor"):
                profile_module.snapshot(config, backend)
            stopped = make_profile()
            backend = SimpleNamespace(run=lambda rank, _: json.dumps(self.inspect(stopped, rank, rank == "0")))
            with self.assertRaisesRegex(ValueError, "not running"):
                profile_module.snapshot(config, backend)


class PolicyLeaseTests(unittest.TestCase):
    def test_lease_resets_sustain_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = Policy(root / "state", root / "events", now=1000)

            def rows(now):
                return {r: {"epoch": now, "max_zone_c": 90, "available_gib": 8} for r in ("0", "1")}

            for now in range(1001, 1400):
                policy.update(rows(now), True, "owned", now)
            policy.update(rows(1400), False, "operator_lease", 1400)
            for now in range(1401, 1700):
                policy.update(rows(now), True, "owned", now)
            self.assertFalse(policy.data["transitioning"], "a lease interrupts the 600 s continuous episode")


if __name__ == "__main__":
    unittest.main()
