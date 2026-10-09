#!/usr/bin/env python3
"""Merge LSGEmu coverage reports produced by dynamic runs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Set

try:
    from .artifact_io import atomic_json_dump
except ImportError:  # Allow ``python3 lsgemu/merge_coverage_reports.py``.
    from artifact_io import atomic_json_dump


STRICT_COVERAGE_SCHEMA = "lsgemu.dynamic_coverage.v2"


class CoverageIdentityError(ValueError):
    """Raised when reports cannot be compared under one coverage contract."""


def expand_inputs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path in seen or not path.exists():
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _identity_fields(data: Dict[str, object]) -> Dict[str, object]:
    static_identity = data.get("static_cache_identity")
    static_identity_hash = ""
    if isinstance(static_identity, dict):
        static_identity_hash = str(static_identity.get("identity_hash") or "")
    return {
        "firmware_sha256": str(data.get("firmware_sha256") or ""),
        "valid_bb_denominator_hash": str(data.get("valid_bb_denominator_hash") or ""),
        "coverage_schema": str(data.get("coverage_schema") or ""),
        "coverage_model": str(data.get("coverage_model") or ""),
        "coverage_source": str(data.get("coverage_source") or ""),
        "static_identity_hash": static_identity_hash,
        "total_bbs": int(data.get("total_bbs") or 0),
        "valid_total_bbs": int(data.get("valid_total_bbs") or 0),
        "firmware": str(data.get("firmware") or ""),
    }


def _validate_identity(
    path: Path,
    data: Dict[str, object],
    *,
    allow_legacy: bool,
) -> tuple[Dict[str, object], bool, List[str]]:
    identity = _identity_fields(data)
    missing = [
        field
        for field in ("firmware_sha256", "valid_bb_denominator_hash", "coverage_schema")
        if not identity[field]
    ]
    reasons: List[str] = []
    if missing:
        reasons.extend(f"missing_{field}" for field in missing)
    if identity["coverage_schema"] and identity["coverage_schema"] != STRICT_COVERAGE_SCHEMA:
        reasons.append(f"unsupported_coverage_schema:{identity['coverage_schema']}")
    if identity["valid_total_bbs"] < 0 or identity["total_bbs"] < 0:
        reasons.append("negative_denominator")
    if not identity["firmware"] and not identity["firmware_sha256"]:
        reasons.append("missing_firmware_identity")
    if reasons and not allow_legacy:
        raise CoverageIdentityError(f"{path}: incompatible coverage identity ({', '.join(reasons)})")
    return identity, not reasons, reasons


def load_report(path: Path, *, allow_legacy: bool = False) -> Dict[str, object]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise CoverageIdentityError(f"{path}: report must be a JSON object")
    identity, identity_verified, legacy_reasons = _validate_identity(
        path,
        data,
        allow_legacy=allow_legacy,
    )
    covered = data.get("covered_bb_list", [])
    covered_valid = data.get("covered_valid_bb_list", [])
    if not isinstance(covered, list):
        covered = []
    if not isinstance(covered_valid, list):
        covered_valid = []
    return {
        "path": str(path),
        "runner": data.get("runner"),
        "firmware": data.get("firmware"),
        "strategy": data.get("strategy") or path.parent.name,
        "covered_bbs": len(set(int(bb) for bb in covered)),
        "total_bbs": int(data.get("total_bbs") or 0),
        "coverage_rate": float(data.get("coverage_rate") or 0.0),
        "valid_total_bbs": int(data.get("valid_total_bbs") or 0),
        "valid_covered_bbs": len(set(int(bb) for bb in covered_valid)),
        "valid_coverage_rate": float(data.get("valid_coverage_rate") or 0.0),
        "execution_time_seconds": data.get("execution_time_seconds"),
        "covered": set(int(bb) for bb in covered),
        "covered_valid": set(int(bb) for bb in covered_valid),
        "identity": identity,
        "identity_verified": identity_verified,
        "legacy_reasons": legacy_reasons,
    }


def _check_compatible_reports(reports: List[Dict[str, object]], *, allow_legacy: bool) -> Dict[str, object]:
    if not reports:
        return {"verified": False, "legacy": False, "identity": {}}
    verified = [report for report in reports if report.get("identity_verified")]
    anchors = verified or reports
    anchor = dict(anchors[0].get("identity") or {})
    legacy = any(not report.get("identity_verified") for report in reports)
    for report in reports[1:]:
        identity = dict(report.get("identity") or {})
        fields = (
            "firmware_sha256",
            "valid_bb_denominator_hash",
            "coverage_schema",
            "coverage_model",
            "coverage_source",
            "total_bbs",
            "valid_total_bbs",
        )
        for field in fields:
            left = anchor.get(field)
            right = identity.get(field)
            if left and right and left != right:
                raise CoverageIdentityError(
                    f"incompatible reports: {field} differs ({left!r} != {right!r}); "
                    "coverage union would be meaningless"
                )
            if not allow_legacy and (not left or not right):
                raise CoverageIdentityError(
                    f"incompatible reports: {field} is missing; use --allow-legacy "
                    "only for explicitly marked legacy inputs"
                )
        left_static = anchor.get("static_identity_hash")
        right_static = identity.get("static_identity_hash")
        if left_static and right_static and left_static != right_static:
            raise CoverageIdentityError(
                f"incompatible reports: static_identity_hash differs "
                f"({left_static!r} != {right_static!r})"
            )
    return {
        "verified": bool(verified) and not legacy,
        "legacy": legacy,
        "identity": anchor,
        "legacy_reasons": sorted({reason for report in reports for reason in report.get("legacy_reasons", [])}),
    }


def greedy_order(reports: List[Dict[str, object]]) -> List[Dict[str, object]]:
    covered: Set[int] = set()
    order: List[Dict[str, object]] = []
    remaining = list(reports)
    while True:
        best = None
        for report in remaining:
            gain = len(report["covered"] - covered)
            if gain and (best is None or gain > best[0]):
                best = (gain, report)
        if best is None:
            break
        gain, report = best
        covered.update(report["covered"])
        order.append({
            "strategy": report["strategy"],
            "path": report["path"],
            "gain": gain,
            "strategy_bbs": report["covered_bbs"],
            "union_bbs": len(covered),
        })
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="Report JSON paths or glob patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Allow reports without the v2 identity fields; output is marked legacy and is not fully verified.",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.reports)
    reports = []
    for path in paths:
        report = load_report(path, allow_legacy=args.allow_legacy)
        if report["covered_bbs"]:
            reports.append(report)

    identity_summary = _check_compatible_reports(reports, allow_legacy=args.allow_legacy)

    union: Set[int] = set()
    union_valid: Set[int] = set()
    total_bbs = 0
    valid_total_bbs = 0
    firmware = identity_summary.get("identity", {}).get("firmware") or None
    occurrence: Dict[int, int] = {}
    for report in reports:
        covered = report["covered"]
        union.update(covered)
        union_valid.update(report["covered_valid"])
        total_bbs = max(total_bbs, int(report["total_bbs"]))
        valid_total_bbs = max(valid_total_bbs, int(report["valid_total_bbs"]))
        if firmware is None and report.get("firmware"):
            firmware = report["firmware"]
        for bb in covered:
            occurrence[bb] = occurrence.get(bb, 0) + 1

    compact_reports = []
    for report in reports:
        covered = report["covered"]
        compact = {
            "strategy": report["strategy"],
            "path": report["path"],
            "runner": report["runner"],
            "covered_bbs": report["covered_bbs"],
            "coverage_rate": report["coverage_rate"],
            "valid_total_bbs": report["valid_total_bbs"],
            "valid_covered_bbs": report["valid_covered_bbs"],
            "valid_coverage_rate": report["valid_coverage_rate"],
            "execution_time_seconds": report["execution_time_seconds"],
            "absolute_unique_bbs": sum(1 for bb in covered if occurrence.get(bb) == 1),
            "identity_verified": bool(report.get("identity_verified")),
            "legacy_reasons": list(report.get("legacy_reasons") or []),
        }
        compact_reports.append(compact)

    result = {
        "runner": "merge_coverage_reports",
        "coverage_schema": STRICT_COVERAGE_SCHEMA,
        "identity_verified": bool(identity_summary.get("verified")),
        "legacy_input": bool(identity_summary.get("legacy")),
        "identity": identity_summary.get("identity", {}),
        "legacy_reasons": identity_summary.get("legacy_reasons", []),
        "firmware": firmware,
        "input_reports": len(reports),
        "total_bbs": total_bbs,
        "covered_bbs": len(union),
        "coverage_rate": (len(union) / total_bbs * 100) if total_bbs else 0.0,
        "covered_bb_list": sorted(union),
        "valid_total_bbs": valid_total_bbs,
        "valid_covered_bbs": len(union_valid),
        "valid_coverage_rate": (len(union_valid) / valid_total_bbs * 100) if valid_total_bbs else 0.0,
        "covered_valid_bb_list": sorted(union_valid),
        "reports": compact_reports,
        "greedy_order": greedy_order(reports),
    }
    output = Path(args.output)
    atomic_json_dump(result, output, indent=2)

    print(f"input_reports={len(reports)}")
    print(f"covered_bbs={result['covered_bbs']}")
    print(f"total_bbs={result['total_bbs']}")
    print(f"coverage_rate={result['coverage_rate']:.2f}")
    if result["valid_total_bbs"]:
        print(f"valid_covered_bbs={result['valid_covered_bbs']}")
        print(f"valid_total_bbs={result['valid_total_bbs']}")
        print(f"valid_coverage_rate={result['valid_coverage_rate']:.2f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
