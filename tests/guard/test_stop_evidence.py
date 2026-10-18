"""Native STOP request attribution, STOP evidence archiving and maintenance admission contracts."""

import hashlib
import http.client
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import Backend
from daemon import Manager
from guard_fixtures import Native, TransportFixture, backend_config, synthetic_sha
from policy import Policy


class StopAttributionBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend(backend_config(containers={"0": "owned-native"}))

    def test_native_token_count_preserves_request_and_closes_connection(self):
        body = b'{"messages":[{"role":"user","content":"hello"}]}'
        for route in ("/v1/chat/completions?trace=1", "/v1/completions"):
            with self.subTest(route=route), patch("backend.http.client.HTTPConnection") as connect:
                connection = connect.return_value
                connection.getresponse.return_value.status = 200
                connection.getresponse.return_value.read.return_value = b'{"count":123}'
                self.assertEqual(self.backend.request_token_count(body, route),
                                 {"prompt_tokens": 123, "count_source": "native_tokenize"})
                connect.assert_called_once_with("127.0.0.1", 8001, timeout=3)
                connection.request.assert_called_once_with("POST", "/tokenize", body,
                                                           {"Content-Type": "application/json"})
                connection.close.assert_called_once_with()

    def test_unsupported_route_never_tokenizes(self):
        with patch("backend.http.client.HTTPConnection") as connect:
            self.assertEqual(self.backend.request_token_count(b"{}", "/v1/embeddings"),
                             {"count_source": "unsupported_route"})
            connect.assert_not_called()

    def test_multimodal_request_never_renders_or_tokenizes_twice(self):
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]}]}).encode()
        with patch("backend.http.client.HTTPConnection") as connect:
            self.assertEqual(self.backend.request_token_count(body, "/v1/chat/completions"),
                             {"count_source": "multimodal_not_retokenized"})
            connect.assert_not_called()

    def test_invalid_native_token_counts_never_become_attribution(self):
        for count in (True, -1, 1.5, "12", None):
            with self.subTest(count=count), patch("backend.http.client.HTTPConnection") as connect:
                connection = connect.return_value
                connection.getresponse.return_value.status = 200
                connection.getresponse.return_value.read.return_value = json.dumps({"count": count}).encode()
                with self.assertRaisesRegex(ValueError, "invalid native token count"):
                    self.backend.request_token_count(b'{"prompt":"hello"}', "/v1/completions")
                connection.close.assert_called_once_with()

    @staticmethod
    def request(count):
        return {"epoch": 1000, "prompt_tokens": count, "resumed_tokens": None}

    def context(self, requests, lines):
        evidence = {"request_metadata": requests}
        before = json.loads(json.dumps(evidence))
        self.backend.run = Mock(return_value="\n".join(lines))
        context = self.backend.stop_request_context(evidence)
        self.assertEqual(evidence, before, "enrichment must preserve original STOP evidence")
        return context

    def test_exact_unique_prompt_count_attributed_once(self):
        context = self.context(
            {"a": self.request(123), "unknown": self.request(None)},
            ["sessions: slot 2 resumed 100 of 123 prompt tokens from RAM",
             "sessions: slot 3 resumed 50 of 456 prompt tokens from RAM"],
        )
        self.assertEqual(context["requests"]["a"]["resumed_tokens"], 100)
        self.assertEqual(context["requests"]["a"]["resumed_source"], "native_session_log")
        self.assertIsNone(context["requests"]["unknown"]["resumed_tokens"])
        self.assertEqual(len(context["resume_observations"]), 2)
        self.backend.run.assert_called_once_with(
            "0", ["docker", "logs", "--timestamps", "--since", "1000", "--tail", "400", "owned-native"], timeout=4)

    def test_same_prompt_count_on_concurrent_requests_stays_unknown(self):
        context = self.context({"a": self.request(123), "b": self.request(123)},
                               ["sessions: slot 2 resumed 100 of 123 prompt tokens from RAM"])
        for request in context["requests"].values():
            self.assertIsNone(request["resumed_tokens"])
            self.assertNotIn("resumed_source", request)
        self.assertEqual(len(context["resume_observations"]), 1)

    def test_multiple_matching_observations_stay_unknown(self):
        context = self.context({"a": self.request(123)},
                               ["sessions: slot 2 resumed 100 of 123 prompt tokens from RAM",
                                "sessions: slot 3 resumed 90 of 123 prompt tokens from RAM"])
        self.assertIsNone(context["requests"]["a"]["resumed_tokens"])
        self.assertNotIn("resumed_source", context["requests"]["a"])
        self.assertEqual(len(context["resume_observations"]), 2)

    def test_no_active_requests_skips_log_lookup(self):
        self.backend.run = Mock()
        self.assertEqual(self.backend.stop_request_context({}), {"requests": {}, "resume_observations": []})
        self.backend.run.assert_not_called()


class StopLifecycleTests(unittest.TestCase):
    def test_changed_profile_archives_evidence_before_new_stop(self):
        old, new = synthetic_sha("old"), synthetic_sha("new")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = Policy(root / "state", root / "events", initial_profile=old, now=1000)
            policy.update({"0": {"epoch": 1001, "max_zone_c": 94, "available_gib": 8}}, True, "owned", 1001)
            original = json.loads(json.dumps(policy.data["stop_evidence"]))
            manager = object.__new__(Manager)
            manager.policy = policy
            manager.config = {"profiles": {old: {}, new: {}}, "state_path": str(root / "state")}
            manager.cool_idle_operator = Mock(return_value={"idle": True})
            with self.assertRaisesRegex(ValueError, "same-profile STOP"):
                manager.control("prepare-profile", old)
            self.assertEqual(policy.data["stop_evidence"], original)
            manager.cool_idle_operator.assert_not_called()
            manager.control("prepare-profile", new)
            archives = list((root / "archives").glob("*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text())["stop_evidence"], original)
            self.assertNotIn("stop_evidence", policy.data)
            policy.update({"1": {"epoch": 1002, "max_zone_c": 80, "available_gib": 3}}, True, "owned", 1002)
            self.assertEqual(policy.data["stop_evidence"]["epoch"], 1002)
            manager.backend = SimpleNamespace()
            with self.assertRaisesRegex(ValueError, "new hard STOP"):
                manager.control("authorize-profile", new)
            self.assertTrue(Policy(root / "state", root / "events").data["hard_stop"])

    def test_hard_stop_action_records_context_and_clears_only_when_both_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha = synthetic_sha("current")
            policy = Policy(root / "state", root / "events", initial_profile=sha, now=1000)
            policy.owned = True
            policy.update({"0": {"epoch": 1001, "max_zone_c": 95, "available_gib": 8}}, True, "owned", 1001)
            manager = object.__new__(Manager)
            manager.policy, manager.last_action = policy, 0
            records = [{"rank": "0", "stopped": True}, {"rank": "1", "stopped": False, "error": "unreachable"}]
            manager.backend = SimpleNamespace(stop_owned=Mock(return_value=records),
                                              stop_request_context=Mock(side_effect=OSError("logs")))
            manager.act()
            manager.backend.stop_owned.assert_called_once_with(sha)
            self.assertEqual(policy.data["pending_action"], "stop_owned")
            self.assertEqual(policy.data["stop_evidence"]["native_request_context"], {"error": "OSError"})
            manager.last_action = 0
            manager.backend.stop_owned.return_value = [dict(records[0]), {"rank": "1", "stopped": True}]
            manager.act()
            self.assertIsNone(Policy(root / "state", root / "events").data["pending_action"])
            self.assertTrue(policy.data["hard_stop"])


class MaintenanceTests(TransportFixture, unittest.TestCase):
    def send(self, value):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=4)
        try:
            connection.request("POST", "/v1/chat/completions", b'{"prompt":"hello"}', {"X-TF-Maintenance": value})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_maintenance_token_required_and_not_forwarded(self):
        maintenance = "local-maintenance-fixture"
        self.front.config["maintenance_token_sha256"] = hashlib.sha256(maintenance.encode()).hexdigest()
        self.assertEqual(self.request()[0], 503)
        self.assertEqual(self.request("GET"), (200, b"{}"))
        self.assertEqual(self.send("wrong")[0], 503)
        self.assertFalse(Native.received)
        self.assertEqual(self.p.inflight, 0)
        forwarded = []
        native_post = Native.do_POST

        def capture(handler):
            forwarded.append(handler.headers.get("X-TF-Maintenance"))
            return native_post(handler)

        with patch.object(Native, "do_POST", capture):
            self.assertEqual(self.send(maintenance), (200, Native.payload))
        self.wait_idle()
        self.assertEqual(forwarded, [None])
        self.assertEqual(len(Native.received), 1)
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(self.p.request_metadata, {})


if __name__ == "__main__":
    unittest.main()
