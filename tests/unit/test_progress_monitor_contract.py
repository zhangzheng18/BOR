#!/usr/bin/env python3
"""Contracts for isolated, identity-bound progress telemetry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from lsgemu.cached_interleaved_runner import (
    CoverageProgressMonitor,
    env_int,
    progress_artifact_path,
)


class _Prepared:
    firmware_sha256 = "firmware-sha"
    toolchain_fingerprint = {"runtime_fingerprint": "runtime-sha"}
    static_cache_identity = {"identity_hash": "cache-sha"}
    valid_bb_set = {0x08000000, 0x08000004}


class _Runner:
    prepared = _Prepared()


class _StuckThread:
    def __init__(self) -> None:
        self.join_timeout = None

    def join(self, timeout=None) -> None:
        self.join_timeout = timeout

    def is_alive(self) -> bool:
        return True


class ProgressMonitorContractTests(unittest.TestCase):
    def test_env_int_has_one_float_tolerant_definition(self):
        with patch.dict(os.environ, {"LSGEMU_TEST_INTEGER": "12.9"}, clear=False):
            self.assertEqual(12, env_int("LSGEMU_TEST_INTEGER", 7))
        with patch.dict(os.environ, {"LSGEMU_TEST_INTEGER": "invalid"}, clear=False):
            self.assertEqual(7, env_int("LSGEMU_TEST_INTEGER", 7))

    def test_progress_path_honors_explicit_path_and_isolates_direct_rerun(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_progress_") as tmpdir:
            output_dir = Path(tmpdir)
            firmware = output_dir / "firmware.elf"
            explicit = output_dir / "explicit.jsonl"
            with patch.dict(
                os.environ,
                {"LSGEMU_PROGRESS_JSONL": str(explicit)},
                clear=False,
            ):
                self.assertEqual(
                    explicit.resolve(),
                    progress_artifact_path(output_dir, firmware),
                )

            canonical = output_dir / "firmware_coverage_progress.jsonl"
            canonical.touch()
            cleaned_env = {
                "LSGEMU_PROGRESS_JSONL": "",
                "LSGEMU_PROGRESS_REUSE_EXISTING": "",
                "LSGEMU_ATTEMPT_ID": "",
            }
            with patch.dict(os.environ, cleaned_env, clear=False):
                isolated = progress_artifact_path(output_dir, firmware)
            self.assertNotEqual(canonical.resolve(), isolated)
            self.assertEqual(output_dir.resolve(), isolated.parent)
            self.assertTrue(isolated.name.startswith("firmware_"))
            self.assertTrue(isolated.name.endswith("_coverage_progress.jsonl"))

            with patch.dict(
                os.environ,
                {**cleaned_env, "LSGEMU_ATTEMPT_ID": "attempt-1"},
                clear=False,
            ):
                self.assertEqual(
                    canonical.resolve(),
                    progress_artifact_path(output_dir, firmware),
                )

    def test_status_carries_run_identity_and_failure_is_explicit(self):
        expected_denominator = hashlib.sha256(
            b"".join(
                address.to_bytes(4, "little")
                for address in sorted(_Prepared.valid_bb_set)
            )
        ).hexdigest()
        with tempfile.TemporaryDirectory(prefix="lsgemu_monitor_") as tmpdir:
            env = {
                "LSGEMU_ATTEMPT_ID": "attempt-1",
                "LSGEMU_CAMPAIGN_FINGERPRINT": "campaign-sha",
                "LSGEMU_SOURCE_TREE_SHA256": "source-sha",
                "LSGEMU_TOOLCHAIN_IDENTITY_HASH": "toolchain-sha",
            }
            with patch.dict(os.environ, env, clear=False):
                monitor = CoverageProgressMonitor(
                    runner=_Runner(),
                    firmware=Path(tmpdir) / "firmware.elf",
                    output_path=Path(tmpdir) / "progress.jsonl",
                    interval_seconds=0,
                    started_at=time.time(),
                )
            status = monitor.status()
            self.assertTrue(status["healthy"])
            self.assertTrue(status["telemetry_complete"])
            self.assertEqual("attempt-1", status["attempt_id"])
            self.assertEqual("firmware-sha", status["firmware_sha256"])
            self.assertEqual("source-sha", status["source_tree_sha256"])
            self.assertEqual(
                "runtime-sha", status["toolchain_runtime_fingerprint"]
            )
            self.assertEqual("toolchain-sha", status["toolchain_identity_hash"])
            self.assertEqual("cache-sha", status["static_cache_identity_hash"])
            self.assertEqual(
                expected_denominator,
                status["valid_bb_denominator_hash"],
            )

            monitor._record_failure("test", RuntimeError("telemetry failed"))
            failed = monitor.status()
            self.assertFalse(failed["healthy"])
            self.assertFalse(failed["telemetry_complete"])
            self.assertEqual(1, failed["failure_count"])
            self.assertEqual("test", failed["last_failure"]["operation"])

    def test_join_timeout_marks_telemetry_incomplete(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_monitor_") as tmpdir:
            monitor = CoverageProgressMonitor(
                runner=_Runner(),
                firmware=Path(tmpdir) / "firmware.elf",
                output_path=Path(tmpdir) / "progress.jsonl",
                interval_seconds=1,
                started_at=time.time(),
            )
            stuck = _StuckThread()
            monitor._thread = stuck
            captured = {}

            def capture_snapshot(event, *, stage=None, extra=None):
                captured.update(extra or {})
                return {"event": event, "stage": stage}

            monitor.snapshot = capture_snapshot
            monitor.stop("run_complete")

            status = monitor.status()
            self.assertEqual(2.0, stuck.join_timeout)
            self.assertEqual("join_timeout", status["worker_exit_reason"])
            self.assertFalse(status["telemetry_complete"])
            self.assertEqual(1, status["failure_count"])
            self.assertEqual("worker_join", status["last_failure"]["operation"])
            self.assertFalse(captured["progress_monitor"]["telemetry_complete"])


if __name__ == "__main__":
    unittest.main()
