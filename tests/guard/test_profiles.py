"""General clock-ladder, snapshotted-profile and production-config validation rules."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from backend import Backend
from common import FLOOR_KEY, PLAN_KEY, POLICY, POOL_KEY, SCHEDULE_KEY, load_config, validate_ladder, validate_profiles
from guard_fixtures import CPU_SETS, make_profile, make_profiles, production_config, synthetic_sha, write_config


class LadderTests(unittest.TestCase):
    def test_valid_ladders(self):
        for ladder in ([2100, 2000], [2200, 2100, 2000], [3000, 2500, 2000, 1000]):
            with self.subTest(ladder=ladder):
                self.assertEqual(validate_ladder(ladder), tuple(ladder))

    def test_invalid_ladders(self):
        for ladder in (None, (2100, 2000), [], [2100], [2400, 2300, 2200, 2100, 2000], [2000, 2100],
                       [2100, 2100], [2200, 2000, 2100], [3001, 2000], [2000, 999], [2100.0, 2000],
                       [True, 1000], ["2100", "2000"]):
            with self.subTest(ladder=ladder), self.assertRaisesRegex(ValueError, "clock ladder"):
                validate_ladder(ladder)


class ProfileTests(unittest.TestCase):
    def check(self, profile):
        validate_profiles({synthetic_sha("checked"): profile})

    def rejected(self, change, pattern=None):
        profile = make_profile(cpu_sets=CPU_SETS)
        change(profile)
        with self.assertRaisesRegex(ValueError, pattern or ""):
            self.check(profile)

    def test_valid_profile_accepted_with_and_without_cpu_sets(self):
        self.check(make_profile())
        self.check(make_profile(cpu_sets=CPU_SETS))
        self.check(make_profile(**{FLOOR_KEY: "4"}))
        validate_profiles(make_profiles(*map(str, range(32))))

    def test_profile_count_and_key_shape(self):
        for profiles in ({}, make_profiles(*map(str, range(33))), [], None,
                         {"A" * 64: make_profile()}, {"a" * 63: make_profile()}):
            with self.subTest(count=len(profiles or ())), self.assertRaises(ValueError):
                validate_profiles(profiles)

    def test_per_rank_identity_required(self):
        self.rejected(lambda p: p["image_ids"].update({"1": "sha256:" + "A" * 64}), "image ID")
        self.rejected(lambda p: p["image_ids"].update({"0": synthetic_sha("bare")}), "image ID")
        self.rejected(lambda p: p["mounts"]["0"].pop("/model"), "mounts")
        self.rejected(lambda p: p["inference_environment"].update({"1": {}}), "environment")
        # Per-rank images and mounts may legitimately differ (one host per rank).
        profile = make_profile()
        profile["image_ids"]["1"] = "sha256:" + synthetic_sha("other-image")
        profile["mounts"]["1"]["/model"] = "/srv/elsewhere"
        self.check(profile)

    def test_workspace_floor_below_hard_floor_rejected(self):
        for floor in ("3.99", "3.5", "0", "-4", "nan", "inf", "-inf", "four"):
            with self.subTest(floor=floor):
                self.rejected(lambda p, value=floor: [env.update({FLOOR_KEY: value})
                                                      for env in p["inference_environment"].values()], "floor")

    def test_rank_mismatched_knobs_rejected(self):
        for key, value in ((POOL_KEY, "1500000"), (FLOOR_KEY, "5.0"), (SCHEDULE_KEY, "spin")):
            with self.subTest(key=key):
                self.rejected(lambda p, k=key, v=value: p["inference_environment"]["1"].update({k: v}),
                              "both ranks")
                self.rejected(lambda p, k=key: p["inference_environment"]["1"].pop(k), "both ranks")

    def test_bad_cuda_schedule_rejected(self):
        for value in ("blocking ", "BLOCKING", "busy", ""):
            with self.subTest(value=value):
                self.rejected(lambda p, v=value: [env.update({SCHEDULE_KEY: v})
                                                  for env in p["inference_environment"].values()], "TF_CUDA_SCHED")
        for value in ("auto", "spin", "yield", "blocking"):
            with self.subTest(value=value):
                self.check(make_profile(**{SCHEDULE_KEY: value}))

    def test_plan_pin_outside_cpu_set_rejected(self):
        for value in ("0", "1", "6", "4095"):
            with self.subTest(value=value):
                self.rejected(lambda p, v=value: p["inference_environment"]["1"].update({PLAN_KEY: v}),
                              "inside the rank")
        for value in ("2", "5"):
            self.check(make_profile(cpu_sets=CPU_SETS, **{PLAN_KEY: value}))
        # Without a host set the pin only has to be a canonical local index.
        self.check(make_profile(**{PLAN_KEY: "4095"}))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = production_config(self.root)

    def load(self, config):
        return load_config(write_config(self.root, config))

    def refused(self, change):
        config = copy.deepcopy(self.config)
        change(config)
        with self.assertRaises((ValueError, KeyError)):
            self.load(config)

    def test_valid_config_loads_and_registers_exact_profiles(self):
        loaded = self.load(self.config)
        self.assertEqual(loaded["profiles"], self.config["profiles"])
        self.assertEqual(loaded["clock_ladder_mhz"], [2200, 2100, 2000])
        for units in ([], None):
            config = copy.deepcopy(self.config)
            if units is None:
                config.pop("watchdog_units")
            else:
                config["watchdog_units"] = units
            self.load(config)
        backend = Backend(loaded)
        Path(loaded["production_config"]).write_text("unregistered bytes")
        with self.assertRaisesRegex(ValueError, "unapproved"):
            backend.profile()

    def test_policy_must_be_enabled_and_exact(self):
        self.refused(lambda c: c.update(enabled=False))
        self.refused(lambda c: c.update(mode="shadow"))
        self.refused(lambda c: c["policy"].update(hard_c=95))
        self.refused(lambda c: c["policy"].update(floor_mhz=2000))
        self.assertNotIn("floor_mhz", POLICY)

    def test_clock_settings(self):
        self.refused(lambda c: c.pop("clock_ladder_mhz"))
        self.refused(lambda c: c.update(clock_ladder_mhz=[2000, 2100]))
        self.refused(lambda c: c.update(clock_service="gpu-clock-cap"))
        self.refused(lambda c: c.update(clock_service="gpu clock.service"))
        self.refused(lambda c: c.update(clock_dropin="relative/cap.conf"))
        self.refused(lambda c: c.update(clock_dropin=str(self.root / ".." / "cap.conf")))
        self.refused(lambda c: c.update(watchdog_units=["native-watch.service"]))
        self.refused(lambda c: c.update(watchdog_units="native-watch.timer"))

    def test_endpoints_and_names(self):
        self.refused(lambda c: c.update(upstream="http://127.0.0.1"))
        self.refused(lambda c: c.update(upstream="https://127.0.0.1:8001"))
        self.refused(lambda c: c.update(upstream="http://localhost:8001"))
        self.refused(lambda c: c.update(listen_host="0.0.0.0"))
        self.refused(lambda c: c.update(listen_port="8000"))
        self.refused(lambda c: c.update(listen_port=8001))
        self.refused(lambda c: c["containers"].update({"1": "-flag"}))
        self.refused(lambda c: c["containers"].pop("1"))
        self.refused(lambda c: c.update(worker_ssh="-oProxyCommand=x"))
        self.refused(lambda c: c.update(state_path="relative/state"))

    def test_bounded_limits(self):
        for key, value in (("max_inflight", 0), ("max_inflight", 5), ("max_inflight", True), ("max_pending", 17),
                           ("max_connections", 7), ("body_buffers", 5), ("memory_transient_reserve_gib", 0.49),
                           ("memory_transient_reserve_gib", 4.01), ("memory_transient_reserve_gib", True),
                           ("queue_seconds", 1201), ("queue_seconds", 0)):
            with self.subTest(key=key, value=value):
                self.refused(lambda c, k=key, v=value: c["limits"].update({k: v}))
        for inflight in (1, 2, 4):
            config = copy.deepcopy(self.config)
            config["limits"]["max_inflight"] = inflight
            self.assertEqual(self.load(config)["limits"]["max_inflight"], inflight)

    def test_invalid_profile_fails_the_whole_config(self):
        self.refused(lambda c: next(iter(c["profiles"].values()))["inference_environment"]["0"]
                     .update({FLOOR_KEY: "3.5"}))
        self.refused(lambda c: c.update(profiles={}))

    def test_example_config_needs_only_a_snapshotted_profile(self):
        example = Path(__file__).resolve().parents[2] / "config" / "guard.example.json"
        if not example.exists():
            self.skipTest("no config/guard.example.json")
        config = json.loads(example.read_text())
        with self.assertRaisesRegex(ValueError, "profiles"):
            self.load(config)
        config["profiles"] = make_profiles("snapshot")
        self.assertEqual(self.load(config)["clock_ladder_mhz"], config["clock_ladder_mhz"])

    def test_config_requires_integer_safe_admission_pair(self):
        self.assertEqual(self.load(self.config)["limits"]["prefill_admission_pause_c"], 82)
        for pause, resume in ((82.0, 75), (82, 75.0), (True, 1), (89, 75), (82, 82), (82, 0),
                              (None, 75), (82, None), (None, None)):
            config = copy.deepcopy(self.config)
            config["limits"].update(prefill_admission_pause_c=pause, prefill_admission_resume_c=resume)
            with self.subTest(pause=pause, resume=resume), self.assertRaises(ValueError):
                self.load(config)
        config = copy.deepcopy(self.config)
        for key in ("prefill_admission_pause_c", "prefill_admission_resume_c"):
            config["limits"].pop(key)
        self.assertNotIn("prefill_admission_pause_c", self.load(config)["limits"])
        self.assertEqual(json.loads((self.root / "config.json").read_text())["limits"], config["limits"])


if __name__ == "__main__":
    unittest.main()
