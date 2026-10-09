#!/usr/bin/env python3
"""Regression tests for secret-free LLM campaign identity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_lsgemu_serial_eval import (
    extra_argument_file_identities,
    llm_runtime_identity,
    lsgemu_environment_identity,
    normalized_extra_args,
)


class SerialLLMIdentityTests(unittest.TestCase):
    def test_credential_is_hashed_and_never_persisted(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_llm_identity_") as tmpdir:
            config_path = Path(tmpdir) / "llm.yaml"
            config_path.write_text(
                "llm:\n"
                "  model: test-model\n"
                "  api_base: https://example.invalid/v1\n"
                "  api_key_env: LSGEMU_TEST_CAMPAIGN_KEY\n",
                encoding="utf-8",
            )
            environment = {
                "LSGEMU_LLM_CONFIG": str(config_path),
                "LSGEMU_TEST_CAMPAIGN_KEY": "high-entropy-test-key",
            }
            with patch.dict(os.environ, environment, clear=True):
                identity = llm_runtime_identity()
            expected_config_sha256 = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()

        self.assertTrue(identity["credential_available"])
        self.assertEqual(
            "environment:LSGEMU_TEST_CAMPAIGN_KEY",
            identity["credential_source"],
        )
        self.assertEqual(
            hashlib.sha256(b"high-entropy-test-key").hexdigest(),
            identity["credential_sha256"],
        )
        self.assertEqual(
            expected_config_sha256,
            identity["config_file_sha256"],
        )
        self.assertNotIn("high-entropy-test-key", repr(identity))

    def test_campaign_environment_redacts_credential_like_values(self):
        environment = {
            "LSGEMU_TEST_CAMPAIGN_KEY": "high-entropy-test-key",
            "LSGEMU_STAGE_SECONDS": "30",
        }
        with patch.dict(os.environ, environment, clear=True):
            identity = lsgemu_environment_identity()

        self.assertEqual("30", identity["LSGEMU_STAGE_SECONDS"])
        secret_identity = identity["LSGEMU_TEST_CAMPAIGN_KEY"]
        self.assertTrue(secret_identity["redacted"])
        self.assertEqual(
            hashlib.sha256(b"high-entropy-test-key").hexdigest(),
            secret_identity["sha256"],
        )
        self.assertNotIn("high-entropy-test-key", repr(identity))

    def test_serial_runner_rejects_campaign_identity_overrides(self):
        for arguments in (
            ["--output-dir", "/tmp/other"],
            ["--mode=simple"],
            ["--disable-scoped-replay-stages"],
            ["--", "--total-time-minutes", "1"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    normalized_extra_args(arguments)

        self.assertEqual(
            ["--baseline-instructions", "1000"],
            normalized_extra_args(["--baseline-instructions", "1000"]),
        )

    def test_external_run_profile_content_is_fingerprinted(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_profile_identity_") as tmpdir:
            profile = Path(tmpdir) / "profile.yaml"
            profile.write_text("environment: {}\n", encoding="utf-8")
            first = extra_argument_file_identities([
                "--run-profile",
                str(profile),
            ])
            profile.write_text("environment:\n  LSGEMU_TEST: '1'\n", encoding="utf-8")
            second = extra_argument_file_identities([
                f"--run-profile={profile}",
            ])

        self.assertEqual("--run-profile", first[0]["option"])
        self.assertTrue(first[0]["exists"])
        self.assertNotEqual(first[0]["sha256"], second[0]["sha256"])

    def test_missing_credential_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_llm_identity_") as tmpdir:
            config_path = Path(tmpdir) / "llm.yaml"
            config_path.write_text(
                "llm:\n  model: test-model\n  api_key_env: MISSING_KEY\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"LSGEMU_LLM_CONFIG": str(config_path)},
                clear=True,
            ):
                identity = llm_runtime_identity()

        self.assertFalse(identity["credential_available"])
        self.assertEqual("missing", identity["credential_source"])
        self.assertEqual("", identity["credential_sha256"])


if __name__ == "__main__":
    unittest.main()
