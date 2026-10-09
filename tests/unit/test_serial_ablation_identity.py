#!/usr/bin/env python3
"""Run-identity contracts for serial ablation aggregation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.analyze_serial_ablation_results import (
    Observation,
    observation_from_record,
    summarize_pairs,
    write_csv,
)
from experiments.run_lsgemu_serial_eval import (
    report_identity_fields,
    report_matches_run_identity,
)


def _report(attempt_id: str = "attempt-current") -> dict[str, object]:
    return {
        "coverage_schema": "lsgemu.dynamic_coverage.v2",
        "evaluation_metrics": {"schema": "lsgemu.evaluation_metrics.v1"},
        "campaign_fingerprint": "campaign-sha",
        "attempt_id": attempt_id,
        "firmware_sha256": "firmware-sha",
        "source_tree_sha256": "source-sha",
        "valid_bb_denominator_hash": "denominator-sha",
        "toolchain_runtime_fingerprint": "runtime-sha",
        "toolchain_identity_hash": "toolchain-sha",
        "static_cache_identity_hash": "cache-sha",
        "static_cache_identity": {"identity_hash": "cache-sha"},
        "valid_covered_bbs": 40,
        "valid_total_bbs": 100,
        "execution_time_seconds": 50.0,
    }


def _record(output_dir: Path, report_path: Path, **updates) -> dict[str, object]:
    record: dict[str, object] = {
        "mode": "full",
        "relative_path": "P2IM/Test/Test.elf",
        "status": "success",
        "returncode": 0,
        "attempt_id": "attempt-current",
        "attempt_output_dir": str(output_dir),
        "attempt_report_path": str(report_path),
        "report_path": str(report_path),
        "report_written_by_this_run": True,
        "report_published": True,
        "report_complete": True,
        "campaign_fingerprint": "campaign-sha",
        "firmware_sha256": "firmware-sha",
        "source_tree_sha256": "source-sha",
        "valid_bb_denominator_hash": "denominator-sha",
        "toolchain_fingerprint": "runtime-sha",
        "toolchain_identity_hash": "toolchain-sha",
        "static_cache_identity_hash": "cache-sha",
        "elapsed_seconds": 50.0,
    }
    record.update(updates)
    return record


def _observation(mode: str, **updates) -> Observation:
    values = {
        "mode": mode,
        "relative_path": "P2IM/Test/Test.elf",
        "status": "success",
        "returncode": 0,
        "report_complete": True,
        "source": "final_report",
        "valid_covered_bbs": 40 if mode == "full" else 30,
        "valid_total_bbs": 100,
        "elapsed_seconds": 50.0,
        "identity_verified": True,
        "firmware_sha256": "firmware-sha",
        "valid_bb_denominator_hash": "denominator-sha",
        "source_tree_sha256": "source-sha",
        "toolchain_fingerprint": "runtime-sha",
        "toolchain_identity_hash": "toolchain-sha",
        "static_cache_identity_hash": "cache-sha",
        "timeline": {"final": 40 if mode == "full" else 30},
    }
    values.update(updates)
    return Observation(**values)


class SerialAblationIdentityTests(unittest.TestCase):
    def test_successful_zero_returncode_report_is_complete(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            output_dir = Path(tmpdir) / "attempt"
            output_dir.mkdir()
            report_path = output_dir / "Test_interleaved_report.json"
            report_path.write_text(json.dumps(_report()), encoding="utf-8")

            observation = observation_from_record(
                "full",
                _record(output_dir, report_path),
            )

            self.assertTrue(observation.report_complete)
            self.assertTrue(observation.identity_verified)
            self.assertEqual("final_report", observation.source)
            self.assertEqual(40, observation.valid_covered_bbs)

    def test_failed_attempt_ignores_stale_report_and_filters_progress_attempt(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            output_dir = Path(tmpdir) / "attempt"
            output_dir.mkdir()
            report_path = output_dir / "Test_interleaved_report.json"
            stale_report = _report(attempt_id="attempt-old")
            stale_report["firmware_sha256"] = "stale-firmware-sha"
            stale_report["valid_bb_denominator_hash"] = "stale-denominator-sha"
            report_path.write_text(
                json.dumps(stale_report),
                encoding="utf-8",
            )
            progress_path = output_dir / "Test_coverage_progress.jsonl"
            progress_path.write_text(
                "\n".join([
                    json.dumps({
                        "attempt_id": "attempt-old",
                        "valid_covered_bbs": 99,
                        "valid_total_bbs": 100,
                        "elapsed_seconds": 40.0,
                    }),
                    json.dumps({
                        "attempt_id": "attempt-current",
                        "valid_covered_bbs": 12,
                        "valid_total_bbs": 100,
                        "elapsed_seconds": 20.0,
                        "stage": "baseline",
                        "firmware_sha256": "firmware-sha",
                        "valid_bb_denominator_hash": "denominator-sha",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            record = _record(
                output_dir,
                report_path,
                status="failed",
                returncode=139,
                report_written_by_this_run=False,
                report_published=False,
                report_complete=False,
            )

            observation = observation_from_record("full", record)

            self.assertFalse(observation.report_complete)
            self.assertFalse(observation.identity_verified)
            self.assertTrue(observation.progress_attempt_match)
            self.assertEqual(
                "last_durable_checkpoint_lower_bound",
                observation.source,
            )
            self.assertEqual(12, observation.valid_covered_bbs)
            self.assertEqual("firmware-sha", observation.firmware_sha256)
            self.assertEqual(
                "denominator-sha",
                observation.valid_bb_denominator_hash,
            )

    def test_failed_attempt_without_progress_is_a_missing_observation(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            output_dir = Path(tmpdir) / "attempt"
            output_dir.mkdir()
            missing_report = output_dir / "Test_interleaved_report.json"
            record = _record(
                output_dir,
                missing_report,
                status="failed",
                returncode=1,
                report_written_by_this_run=False,
                report_published=False,
                report_complete=False,
            )

            observation = observation_from_record("full", record)

            self.assertEqual("missing_observation", observation.source)
            self.assertEqual(0, observation.valid_covered_bbs)
            self.assertFalse(observation.progress_attempt_match)

    def test_skipped_report_falls_back_to_published_copy_if_attempt_is_archived(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "attempt-output"
            output_dir.mkdir()
            stable_report = root / "published_report.json"
            stable_report.write_text(json.dumps(_report()), encoding="utf-8")
            missing_attempt_report = output_dir / "Test_interleaved_report.json"
            record = _record(
                output_dir,
                stable_report,
                status="skipped_existing_report",
                attempt_report_path=str(missing_attempt_report),
                report_written_by_this_run=False,
                report_published=True,
            )

            observation = observation_from_record("full", record)

            self.assertTrue(observation.report_complete)
            self.assertEqual(str(stable_report), observation.report_path)
            self.assertEqual("final_report", observation.source)

    def test_causal_pair_requires_all_run_identities(self):
        full = _observation("full")
        ablated = _observation("no_semantic_obligation_v2")
        accepted = summarize_pairs(
            {full.relative_path: full},
            {ablated.relative_path: ablated},
            complete_only=True,
            require_causal_identity=True,
        )
        self.assertEqual(1, accepted["pairs"])

        mutations = {
            "firmware_sha256": ("other-firmware", "firmware_identity_mismatch"),
            "valid_bb_denominator_hash": (
                "other-denominator",
                "denominator_identity_mismatch",
            ),
            "source_tree_sha256": (
                "other-source",
                "source_tree_identity_mismatch",
            ),
            "toolchain_fingerprint": (
                "other-runtime",
                "runtime_toolchain_identity_mismatch",
            ),
            "toolchain_identity_hash": (
                "other-toolchain",
                "full_toolchain_identity_mismatch",
            ),
            "static_cache_identity_hash": (
                "other-cache",
                "static_cache_identity_mismatch",
            ),
        }
        for field_name, (value, expected_reason) in mutations.items():
            with self.subTest(field=field_name):
                changed = replace(ablated, **{field_name: value})
                rejected = summarize_pairs(
                    {full.relative_path: full},
                    {changed.relative_path: changed},
                    complete_only=True,
                    require_causal_identity=True,
                )
                self.assertEqual(0, rejected["pairs"])
                self.assertEqual(
                    1,
                    rejected["excluded_identity_reasons"].get(
                        expected_reason,
                        0,
                    ),
                )

    def test_csv_buffer_is_written_after_row_generation(self):
        observation = _observation("full")
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            output_path = Path(tmpdir) / "analysis.csv"
            write_csv(
                output_path,
                {"full": {observation.relative_path: observation}},
            )
            rows = output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(2, len(rows))
        self.assertIn("Full LSGEmu", rows[1])

    def test_campaign_and_prepared_toolchain_identities_remain_separate(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_identity_") as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report = _report()
            report["prepared_toolchain_identity_hash"] = "prepared-firmware-sha"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            identities = report_identity_fields(report_path)
            self.assertEqual("toolchain-sha", identities["toolchain_identity_hash"])
            self.assertTrue(
                report_matches_run_identity(
                    report_path,
                    campaign_fingerprint="campaign-sha",
                    firmware_sha256="firmware-sha",
                    attempt_id="attempt-current",
                    source_tree_sha256="source-sha",
                    toolchain_runtime_fingerprint="runtime-sha",
                    toolchain_identity_hash="toolchain-sha",
                )
            )


if __name__ == "__main__":
    unittest.main()
