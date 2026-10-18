"""Host CUDA scheduling must agree on both ranks, and every rank's runtime environment is exact identity."""

import copy
import json
import unittest

from backend import Backend
from common import SCHEDULE_KEY, validate_profiles
from guard_fixtures import backend_config, make_profile, synthetic_sha

SHA = synthetic_sha("host-wait")


class HostWaitTests(unittest.TestCase):
    def test_schedule_is_exact_and_agreed_on_both_ranks(self):
        profile = make_profile(**{SCHEDULE_KEY: "blocking"})
        validate_profiles({SHA: profile})
        for value in ("spin", "yield", "blocking ", "auto"):
            changed = copy.deepcopy(profile)
            changed["inference_environment"]["1"][SCHEDULE_KEY] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_profiles({SHA: changed})
        changed = copy.deepcopy(profile)
        changed["inference_environment"]["1"].pop(SCHEDULE_KEY)
        with self.assertRaises(ValueError):
            validate_profiles({SHA: changed})
        for env in changed["inference_environment"].values():
            env.pop(SCHEDULE_KEY, None)
        validate_profiles({SHA: changed})

    def test_identity_covers_global_cuda_schedule_and_forwarded_roce_knob(self):
        backend = Backend(backend_config(containers={"0": "r0"}))
        env = {"TF_DSV41_POOL_TOKENS": "2100000", SCHEDULE_KEY: "blocking"}
        profile = {"image_ids": {"0": "image"}, "mounts": {"0": {"/model": "/weights"}},
                   "inference_environment": {"0": env}}
        backend.ownership = lambda: (True, "owned")
        backend.profile = lambda: ("p", profile)
        item = {"Id": "native", "Name": "/r0", "Image": "image", "State": {"Running": True},
                "HostConfig": {"CpusetCpus": ""},
                "Mounts": [{"Destination": "/model", "Source": "/weights", "RW": False}],
                "Config": {"Env": [f"{key}={value}" for key, value in env.items()]
                           + ["PATH=/usr/bin", "TF_DSV41_LAUNCH_T0=123"]}}
        backend.run = lambda *_: json.dumps([item])
        self.assertEqual(backend.identity("0", "p")["id"], "native")
        item["Config"]["Env"][1] = SCHEDULE_KEY + "=spin"
        with self.assertRaisesRegex(ValueError, "inference environment"):
            backend.identity("0", "p")
        item["Config"]["Env"][1] = SCHEDULE_KEY + "=blocking"
        item["Config"]["Env"].append("TF_TP_ROCE_VAR_KB=1")
        with self.assertRaisesRegex(ValueError, "inference environment"):
            backend.identity("0", "p")


if __name__ == "__main__":
    unittest.main()
