"""Optional per-rank host CPU sets: canonical syntax in profiles and exact Docker HostConfig identity."""

import copy
import json
import unittest

from backend import Backend
from common import PLAN_KEY, validate_profiles
from guard_fixtures import CPU_SETS, backend_config, make_profile, synthetic_sha

SHA = synthetic_sha("pinned")


class CpuSetTests(unittest.TestCase):
    def validate(self, profile):
        validate_profiles({SHA: profile})

    def test_optional_sets_and_plan_pin_inside(self):
        profile = make_profile(cpu_sets=CPU_SETS, **{PLAN_KEY: "3"})
        self.validate(profile)
        profile.pop("cpu_sets")
        self.validate(profile)
        # Sets are per host: they may differ between ranks, or be given for one rank only.
        for sets in ({"0": "2-5", "1": "0,1,6-9"}, {"0": "2-5"}, {"0": "", "1": ""}, {},
                     {"0": "0,1,2,3,4", "1": "2-5"}):
            with self.subTest(sets=sets):
                self.validate(make_profile(cpu_sets=sets))

    def test_malformed_sets_rejected(self):
        for sets in (None, [], "2-5", {"2": "2-5"}, {0: "2-5", 1: "2-5"}, {"0": 2, "1": "2-5"},
                     {"0": "2-5 ", "1": "2-5"}, {"0": "2-5;echo", "1": "2-5"}, {"0": "02-5", "1": "2-5"},
                     {"0": "2-5,", "1": "2-5"}, {"0": "5-2", "1": "2-5"}, {"0": "2-4096", "1": "2-5"}):
            with self.subTest(sets=sets), self.assertRaisesRegex(ValueError, "CPU set"):
                profile = make_profile()
                profile["cpu_sets"] = copy.deepcopy(sets)
                self.validate(profile)

    def test_actual_docker_host_config_is_required_and_exact_on_each_rank(self):
        for pinned in (False, True):
            profile = make_profile(cpu_sets=CPU_SETS if pinned else None)
            for rank in ("0", "1"):
                backend = Backend(backend_config(containers={rank: "r" + rank}))
                backend.ownership = lambda: (True, "owned")
                backend.profile = lambda value=profile: ("p", value)
                item = {"Id": "native", "Name": "/r" + rank, "Image": profile["image_ids"][rank],
                        "State": {"Running": True},
                        "Mounts": [{"Destination": "/model", "Source": profile["mounts"][rank]["/model"],
                                    "RW": False}],
                        "Config": {"Env": [f"{k}={v}" for k, v in profile["inference_environment"][rank].items()]}}
                backend.run = lambda *_, value=item: json.dumps([value])
                expected = CPU_SETS[rank] if pinned else ""
                for actual in (expected, "", "2-5", "0-7", "2-5 ", None, 0):
                    with self.subTest(pinned=pinned, rank=rank, actual=actual):
                        item["HostConfig"] = {"CpusetCpus": actual}
                        if actual == expected:
                            self.assertEqual(backend.identity(rank, "p")["id"], "native")
                        else:
                            with self.assertRaisesRegex(ValueError, "host CPU set"):
                                backend.identity(rank, "p")
                for host in ({}, None, [], "missing"):
                    with self.subTest(host=host):
                        item["HostConfig"] = host
                        if host == "missing":
                            item.pop("HostConfig")
                        with self.assertRaisesRegex(ValueError, "host CPU set"):
                            backend.identity(rank, "p", require_running=False)

    def test_identity_refuses_writable_model_and_changed_mounts(self):
        profile = make_profile()
        backend = Backend(backend_config())
        backend.ownership = lambda: (True, "owned")
        backend.profile = lambda: ("p", profile)
        item = {"Id": "native", "Name": "/native-r0", "Image": profile["image_ids"]["0"], "State": {"Running": True},
                "HostConfig": {"CpusetCpus": ""},
                "Mounts": [{"Destination": "/model", "Source": profile["mounts"]["0"]["/model"], "RW": True}],
                "Config": {"Env": [f"{k}={v}" for k, v in profile["inference_environment"]["0"].items()]}}
        backend.run = lambda *_: json.dumps([item])
        with self.assertRaisesRegex(ValueError, "read only"):
            backend.identity("0", "p")
        item["Mounts"][0].update(RW=False, Source="/srv/other")
        with self.assertRaisesRegex(ValueError, "mounts changed"):
            backend.identity("0", "p")
        item["Mounts"][0]["Source"] = profile["mounts"]["0"]["/model"]
        item["Image"] = profile["image_ids"]["1"]
        with self.assertRaisesRegex(ValueError, "image/name"):
            backend.identity("0", "p")
        item["Image"] = profile["image_ids"]["0"]
        item["State"]["Running"] = False
        with self.assertRaisesRegex(ValueError, "not running"):
            backend.identity("0", "p")
        self.assertEqual(backend.identity("0", "p", require_running=False)["id"], "native")
        with self.assertRaisesRegex(ValueError, "ownership/profile changed"):
            backend.identity("0", "other")


if __name__ == "__main__":
    unittest.main()
