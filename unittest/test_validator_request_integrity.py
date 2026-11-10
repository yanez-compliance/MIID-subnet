import os
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from bittensor_wallet import Keypair
from flask import Flask

from MIID.datasets.reputation import process_reward_allocation, validate_reward_allocations
from MIID.datasets.signature_auth import MAX_PROOF_AGE, verify_request_signature
from MIID.utils import validator_requests
from MIID.utils.misc import upload_data


class ValidatorRequestTest(unittest.TestCase):
    def setUp(self):
        self.key = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        nonce_directory = patch("MIID.datasets.signature_auth.DATA_DIR", temp.name)
        endpoint = patch.dict(os.environ, {"MIID_VALIDATOR_API_URL": "https://validator.example"})
        for setting in (nonce_directory, endpoint):
            setting.start()
            self.addCleanup(setting.stop)
        app = Flask(__name__)

        @app.post("/<resource>/<hotkey>")
        def check(resource, hotkey):
            response, status = verify_request_signature(hotkey)
            return ("", 204) if response is None else (response, status)

        self.client = app.test_client()
        self.path = f"/upload_data/{self.key.ss58_address}"

    def signed_request(self, path=None):
        with patch.object(validator_requests.requests, "post") as post:
            validator_requests.post_validator_request(path or self.path, self.key, {"x": 1})
            url = post.call_args.args[0]
            arguments = post.call_args.kwargs
        self.assertFalse(arguments["allow_redirects"])
        return urlsplit(url).path, arguments["data"], arguments["headers"]

    def test_valid_request_cannot_be_replayed(self):
        path, body, headers = self.signed_request()
        self.assertEqual(self.client.post(path, data=body, headers=headers).status_code, 204)
        self.assertEqual(self.client.post(path, data=body, headers=headers).status_code, 409)

    def test_body_route_and_signer_are_bound(self):
        path, body, headers = self.signed_request()
        self.assertEqual(self.client.post(path, data=b'{"x":2}', headers=headers).status_code, 403)
        other_path = path.replace("/upload_data/", "/voice/")
        self.assertEqual(self.client.post(other_path, data=body, headers=headers).status_code, 403)
        outsider = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        with patch.object(validator_requests.requests, "post") as post:
            validator_requests.post_validator_request(path, outsider, {"x": 1})
        self.assertEqual(
            self.client.post(path, data=body, headers=post.call_args.kwargs["headers"]).status_code,
            403,
        )

    def test_old_and_expired_proofs_are_rejected(self):
        self.assertEqual(self.client.post(self.path, json={"signature": "legacy"}).status_code, 401)
        now = time.time()
        with patch.object(validator_requests.time, "time", return_value=now - MAX_PROOF_AGE - 10):
            path, body, headers = self.signed_request()
        self.assertEqual(self.client.post(path, data=body, headers=headers).status_code, 401)

    def test_plain_http_is_rejected(self):
        with patch.dict(os.environ, {"MIID_VALIDATOR_API_URL": "http://validator.example"}):
            with self.assertRaises(ValueError):
                validator_requests.post_validator_request(self.path, self.key, {})

    def test_upload_requires_positive_acknowledgement(self):
        with patch.object(validator_requests, "post_validator_request") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {}
            self.assertIsNone(upload_data(self.key, {}))
            post.return_value.json.return_value = {"allocation_committed": True}
            self.assertEqual(upload_data(self.key, {}), {"allocation_committed": True})


class RewardAllocationTest(unittest.TestCase):
    def test_empty_events_and_duplicate_timestamps_are_rejected(self):
        event = {"timestamp": "2026-11-06_11-30-00", "rep_snapshot_version": None, "miners": []}
        for events in ([{}], [event, event], [event] * (validator_requests.MAX_ALLOCATION_BATCH + 1)):
            with self.subTest(count=len(events)), self.assertRaises(ValueError):
                validate_reward_allocations({"reward_allocation": {"allocations": events}})
        self.assertEqual(validate_reward_allocations({"reward_allocation": {"allocations": [event]}}), [event])

    def test_database_error_response_is_not_a_commit(self):
        def snapshot():
            return {"version": "test", "miners": [{"hotkey": "miner", "rep_score": 1.0}]}

        with self.assertRaises(RuntimeError):
            process_reward_allocation([{}], snapshot, lambda _: {"detail": "failed"})
        result = process_reward_allocation([{}], snapshot, lambda _: {"message": "Reward allocation successful"})
        self.assertAlmostEqual(result["rep_cache"]["miner"]["rep_score"], 0.95)
