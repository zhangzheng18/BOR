#!/usr/bin/env python3
"""Run and summarize a conservative post-simulation MCU security campaign.

The command composes the existing LSGEmu profile and crash-discovery campaign;
it is intentionally a separate entry point so a security campaign cannot alter
the coverage-oriented emulator stages or their oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

try:
    from .campaign import FuzzCampaign, FuzzCampaignConfig
    from .profile import build_profile_from_report, load_profile, write_profile
    from .security_analysis import (
        build_security_report,
        load_report,
        security_report_markdown,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from campaign import FuzzCampaign, FuzzCampaignConfig
    from profile import build_profile_from_report, load_profile, write_profile
    from security_analysis import build_security_report, load_report, security_report_markdown

try:
    from lsgemu.artifact_io import atomic_json_dump
except ImportError:  # pragma: no cover
    atomic_json_dump = None


def _file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _target_identity(
    firmware: Path,
    simulation_report: Dict[str, object],
) -> Dict[str, object]:
    requested_hash = _file_sha256(firmware)
    report_hashes = {
        str(value).lower()
        for value in (
            simulation_report.get("firmware_sha256"),
            simulation_report.get("analysis_firmware_sha256"),
            simulation_report.get("source_firmware_sha256"),
        )
        if value
    }
    report_paths = {
        str(value)
        for value in (
            simulation_report.get("firmware"),
            simulation_report.get("analysis_firmware"),
            simulation_report.get("source_firmware"),
        )
        if value
    }
    requested_path = str(firmware.resolve())
    path_match = requested_path in {str(Path(path).resolve()) for path in report_paths if path}
    hash_match = bool(requested_hash and requested_hash.lower() in report_hashes)
    warnings = []
    if report_paths and not path_match and not hash_match:
        warnings.append("simulation report path and requested firmware do not match")
    if report_hashes and not hash_match:
        warnings.append("requested firmware SHA-256 differs from simulation report")
    if not report_hashes:
        warnings.append("simulation report has no firmware SHA-256; identity is path-only")
    return {
        "requested_firmware": requested_path,
        "requested_sha256": requested_hash,
        "report_firmware_paths": sorted(report_paths),
        "report_sha256_candidates": sorted(report_hashes),
        "path_match": path_match,
        "hash_match": hash_match,
        "identity_ok": bool(hash_match or (path_match and not report_hashes)),
        "warnings": warnings,
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic_json_dump is not None:
        atomic_json_dump(payload, path, indent=2, ensure_ascii=False)
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_security_campaign(args: argparse.Namespace) -> Dict[str, object]:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    simulation_report_path = Path(args.simulation_report).resolve()
    simulation_report = load_report(simulation_report_path)

    firmware_text = args.firmware or simulation_report.get("firmware") or simulation_report.get("analysis_firmware")
    if not firmware_text:
        raise ValueError("firmware is missing from both --firmware and the simulation report")
    firmware = Path(str(firmware_text)).resolve()
    if not firmware.is_file():
        raise FileNotFoundError(firmware)

    identity = _target_identity(firmware, simulation_report)
    if not identity["identity_ok"] and not args.allow_identity_mismatch:
        raise ValueError(
            "simulation report does not identify the requested firmware; "
            "pass --allow-identity-mismatch only for an explicitly reviewed case"
        )

    profile_path: Optional[Path] = None
    if args.profile:
        profile_path = Path(args.profile).resolve()
        # Parse now so a malformed/stale profile fails before a campaign starts.
        load_profile(profile_path)
    elif not args.analyze_only:
        profile_path = work_dir / "security_profile.json"
        profile = build_profile_from_report(simulation_report_path)
        write_profile(profile, profile_path)

    campaign_summary: Optional[Dict[str, object]] = None
    if not args.analyze_only:
        config = FuzzCampaignConfig(
            work_dir=str(work_dir),
            firmware=str(firmware),
            seed_paths=tuple(args.seed),
            iterations=max(0, int(args.iterations)),
            deterministic_cases_per_seed=max(0, int(args.deterministic_cases_per_seed)),
            random_seed=int(args.random_seed),
            time_minutes=float(args.time_minutes),
            timeout_seconds=float(args.timeout_seconds) if args.timeout_seconds else None,
            lsgemu_path=str(args.lsgemu_path),
            python=str(args.python),
            mode=str(args.mode),
            profile=str(profile_path) if profile_path else None,
            run_profile=args.run_profile,
            crash_config=args.crash_config,
            extra_args=tuple(args.extra_arg),
            dictionary=tuple(args.dictionary),
            replay_crashes=max(0, int(args.replay_crashes)),
        )
        campaign_summary = FuzzCampaign(config).run()

    report = build_security_report(
        simulation_report,
        simulation_report_path=simulation_report_path,
        work_dir=work_dir,
        campaign_summary=campaign_summary,
        campaign_ran=not args.analyze_only,
    )
    report["target_identity"] = identity
    report["security_profile"] = str(profile_path) if profile_path else None
    report["command"] = {
        "firmware": str(firmware),
        "simulation_report": str(simulation_report_path),
        "work_dir": str(work_dir),
        "analyze_only": bool(args.analyze_only),
    }
    report_path = work_dir / "security_report.json"
    markdown_path = work_dir / "SECURITY_REPORT.md"
    report["output_json"] = str(report_path)
    report["output_markdown"] = str(markdown_path)
    _write_json(report_path, report)
    markdown_path.write_text(security_report_markdown(report) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a conservative security test campaign from an LSGEmu simulation report"
    )
    parser.add_argument("--simulation-report", required=True, help="completed LSGEmu *_interleaved_report.json")
    parser.add_argument("--firmware", help="firmware ELF; defaults to the report identity")
    parser.add_argument("--work-dir", required=True, help="isolated security campaign directory")
    parser.add_argument("--profile", help="optional existing fuzzengine profile; otherwise build one from the report")
    parser.add_argument("--seed", action="append", default=[], help="seed file or directory; repeatable")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--deterministic-cases-per-seed", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--time-minutes", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--lsgemu-path", default="lsgemu/lsgemu.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mode", choices=("interleaved", "simple"), default="interleaved")
    parser.add_argument("--run-profile")
    parser.add_argument("--crash-config")
    parser.add_argument("--dictionary", action="append", default=[], help="token or hex:...; repeatable")
    parser.add_argument("--extra-arg", action="append", default=[], help="raw argument forwarded to LSGEmu")
    parser.add_argument("--replay-crashes", type=int, default=3)
    parser.add_argument("--analyze-only", action="store_true", help="only build an assessment from existing artifacts")
    parser.add_argument(
        "--allow-identity-mismatch",
        action="store_true",
        help="allow a reviewed report/ELF mismatch; the mismatch remains in the report",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_security_campaign(args)
    except Exception as exc:
        print(f"security campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": (report.get("summary") or {}).get("status"),
        "finding_count": (report.get("summary") or {}).get("finding_count", 0),
        "stable_replay_count": (report.get("summary") or {}).get("stable_replay_count", 0),
        "classification_counts": (report.get("summary") or {}).get("classification_counts", {}),
        "output_json": report.get("output_json"),
        "output_markdown": report.get("output_markdown"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
