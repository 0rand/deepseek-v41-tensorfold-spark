"""Host plan affinity is a local per-rank value; pool, floor and scheduling stay rank-equal."""

import unittest

from common import PLAN_KEY, validate_profiles
from guard_fixtures import make_profile, synthetic_sha

SHA = synthetic_sha("host-pin")


class HostPinTests(unittest.TestCase):
    def profile(self):
        return make_profile(**{PLAN_KEY: "1"})

    def test_explicit_cpu_ids_are_local_not_rank_agreement_fields(self):
        profile = self.profile()
        validate_profiles({SHA: profile})
        for value in ("10", "auto", "0", "4095"):
            profile["inference_environment"]["1"][PLAN_KEY] = value
            validate_profiles({SHA: profile})
        profile["inference_environment"]["1"].pop(PLAN_KEY)
        validate_profiles({SHA: profile})

    def test_invalid_cpu_never_weakens_identity(self):
        for value in ("-1", "4096", "01", "1 ", "anything", "", "1-2"):
            with self.subTest(value=value):
                profile = self.profile()
                profile["inference_environment"]["0"][PLAN_KEY] = value
                with self.assertRaisesRegex(ValueError, "plan CPU"):
                    validate_profiles({SHA: profile})


if __name__ == "__main__":
    unittest.main()
