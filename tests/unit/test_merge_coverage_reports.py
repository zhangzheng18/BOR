#!/usr/bin/env python3
"""Contracts for identity-safe coverage report merging."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lsgemu.merge_coverage_reports import CoverageIdentityError, _check_compatible_reports, load_report


def _report(path: Path, *, firmware_sha256: str = "fw", denominator: str = "den", schema: str = "lsgemu.dynamic_coverage.v2") -> None:
    path.write_text(json.dumps({
        "firmware": "/tmp/firmware.elf",
        "firmware_sha256": firmware_sha256,
        "valid_bb_denominator_hash": denominator,
        "coverage_schema": schema,
        "coverage_model": "refined_static_bbs",
        "coverage_source": "dynamic_unicorn_execution",
        "total_bbs": 10,
        "valid_total_bbs": 8,
        "covered_bb_list": [0x1000, 0x1002],
        "covered_valid_bb_list": [0x1000],
    }), encoding="utf-8")


class MergeCoverageReportTests(unittest.TestCase):
    def test_matching_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.json", root / "b.json"
            _report(first)
            _report(second)
            reports = [load_report(first), load_report(second)]
            summary = _check_compatible_reports(reports, allow_legacy=False)
            self.assertTrue(summary["verified"])
            self.assertFalse(summary["legacy"])

    def test_mismatched_firmware_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.json", root / "b.json"
            _report(first)
            _report(second, firmware_sha256="other")
            reports = [load_report(first), load_report(second)]
            with self.assertRaises(CoverageIdentityError):
                _check_compatible_reports(reports, allow_legacy=False)

    def test_legacy_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps({"firmware": "old.elf", "covered_bb_list": [0x1000]}), encoding="utf-8")
            with self.assertRaises(CoverageIdentityError):
                load_report(path)
            report = load_report(path, allow_legacy=True)
            self.assertFalse(report["identity_verified"])
            self.assertTrue(report["legacy_reasons"])


if __name__ == "__main__":
    unittest.main()
