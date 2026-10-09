#!/usr/bin/env python3
"""Run LSGEmu 24h P2IM experiments and compare against prior fuzzing reports.

The script targets the same P2IM samples used by the MultiFuzz/Fuzzware
24-hour BB coverage analyses: CNC, Gateway, and PLC.  It keeps the denominator
compatible with Fuzzware by setting ``LSGEMU_VALID_BB_ROOT`` to the Fuzzware
example tree, where each target has ``valid_basic_blocks.txt``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.deployment_config import apply_config_from_argv, configured_path
from lsgemu.artifact_io import append_jsonl as atomic_append_jsonl, atomic_json_dump, atomic_write_text

apply_config_from_argv()

try:
    from fuzzengine.crash_detector import CrashDetector, CrashDetectorConfig
except Exception:  # pragma: no cover - keep summary usable if fuzzengine import breaks.
    CrashDetector = None  # type: ignore[assignment]
    CrashDetectorConfig = None  # type: ignore[assignment]


DEFAULT_FIRMWARE_ROOT = configured_path("P2IM_REAL_TESTS_ROOT", PROJECT_ROOT / "datasets" / "real_tests")
FUZZWARE_ROOT = configured_path("FUZZWARE_SOAT_ROOT", PROJECT_ROOT / "datasets" / "fuzzware")
MULTIFUZZ_ROOT = configured_path("MULTIFUZZ_SOAT_ROOT", PROJECT_ROOT / "datasets" / "MultiFuzz")
DEFAULT_VALID_ROOT = FUZZWARE_ROOT / "examples"
DEFAULT_OUTPUT_ROOT = configured_path("LSGEMU_RUN_OUTPUT_DIR", PROJECT_ROOT / ".lsgemu_runs")
LSGEMU_ENTRYPOINT = PROJECT_ROOT / "lsgemu" / "lsgemu.py"

TARGETS: Dict[str, Dict[str, str]] = {
    "CNC": {
        "firmware": "P2IM.CNC.elf",
        "valid_rel": "P2IM/CNC/valid_basic_blocks.txt",
    },
    "Gateway": {
        "firmware": "P2IM.Gateway.elf",
        "valid_rel": "P2IM/Gateway/valid_basic_blocks.txt",
    },
    "PLC": {
        "firmware": "P2IM.PLC.elf",
        "valid_rel": "P2IM/PLC/valid_basic_blocks.txt",
    },
}

# Values are copied from the two 24h documents named in the user request.  They
# are tool-reported results, not independently validated vulnerability counts.
REFERENCE_RESULTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "CNC": {
        "MultiFuzz": {
            "valid_total_bbs": 3614,
            "valid_covered_bbs": 2738,
            "valid_coverage_rate": 75.76,
            "tool_crashes": 5,
            "executions": 78935610,
        },
        "Fuzzware": {
            "valid_total_bbs": 3614,
            "valid_covered_bbs": 2685,
            "valid_coverage_rate": 74.29,
            "tool_crashes": 121,
            "note": "Fuzzware recomputed 24h analysis reports 121 crashes; another comparison table used a different crash-count口径.",
        },
    },
    "Gateway": {
        "MultiFuzz": {
            "valid_total_bbs": 4921,
            "valid_covered_bbs": 3044,
            "valid_coverage_rate": 61.86,
            "tool_crashes": 1565,
            "executions": 53276876,
        },
        "Fuzzware": {
            "valid_total_bbs": 4921,
            "valid_covered_bbs": 2617,
            "valid_coverage_rate": 53.18,
            "tool_crashes": 0,
            "note": "Gateway uses the documented rerun coverage; crash count is reported as 0 in the comparison table.",
        },
    },
    "PLC": {
        "MultiFuzz": {
            "valid_total_bbs": 2303,
            "valid_covered_bbs": 655,
            "valid_coverage_rate": 28.44,
            "tool_crashes": 525,
            "executions": 86456028,
        },
        "Fuzzware": {
            "valid_total_bbs": 2303,
            "valid_covered_bbs": 625,
            "valid_coverage_rate": 27.14,
            "tool_crashes": 454,
            "note": "Fuzzware recomputed 24h analysis reports 454 PLC crashes.",
        },
    },
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    atomic_json_dump(payload, path, indent=2, ensure_ascii=False, durable=False)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    atomic_append_jsonl(path, payload, sort_keys=True, ensure_ascii=False)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def percent(covered: int, total: int) -> float:
    return (covered / total * 100.0) if total else 0.0


def parse_key_value(items: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"environment override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty environment key in {item!r}")
        parsed[key] = value
    return parsed


def selected_target_names(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["CNC", "Gateway", "PLC"]
    lookup = {name.lower(): name for name in TARGETS}
    selected: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        resolved = lookup.get(token.lower())
        if resolved is None:
            raise ValueError(f"unknown target {token!r}; choose from {', '.join(TARGETS)}")
        selected.append(resolved)
    return selected


def target_paths(target: str, firmware_root: Path, valid_root: Path) -> Dict[str, Path]:
    config = TARGETS[target]
    return {
        "firmware": firmware_root / config["firmware"],
        "valid_bb": valid_root / config["valid_rel"],
    }


def load_valid_total(valid_bb_path: Path) -> int:
    if not valid_bb_path.exists():
        return 0
    count = 0
    with valid_bb_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def find_case_files(case_dir: Path, firmware: Path) -> Dict[str, Path]:
    stem = firmware.stem
    return {
        "report": case_dir / f"{stem}_interleaved_report.json",
        "simple_report": case_dir / f"{stem}_lsgemu_result.json",
        "progress": case_dir / f"{stem}_coverage_progress.jsonl",
        "constraints": case_dir / f"{stem}_lsgemu_constraints.json",
        "branch_catalog": case_dir / f"{stem}_branch_catalog.json",
        "llm_history": case_dir / f"{stem}_llm_history.json",
        "log": case_dir / "lsgemu.stdout.log",
        "process_report": case_dir / "process_crash_report.json",
    }


def build_command(
    *,
    python: str,
    firmware: Path,
    output_dir: Path,
    duration_minutes: float,
    run_profile: Optional[Path],
    baseline_timeout_seconds: float,
    lsgemu_args: List[str],
) -> List[str]:
    command = [
        python,
        str(LSGEMU_ENTRYPOINT),
        str(firmware),
        "--time",
        str(float(duration_minutes)),
        "--output-dir",
        str(output_dir),
    ]
    if run_profile is not None:
        command.extend(["--run-profile", str(run_profile)])
    if baseline_timeout_seconds > 0:
        command.extend(["--baseline-timeout-seconds", str(float(baseline_timeout_seconds))])
    command.extend(lsgemu_args)
    return command


def crash_report_for_process(returncode: int, firmware: Path, target: str) -> Dict[str, Any]:
    if CrashDetector is None:
        return {
            "category": "unknown",
            "classification": "crash_detector_unavailable",
            "returncode": returncode,
        }
    detector = CrashDetector(CrashDetectorConfig() if CrashDetectorConfig else None)
    report = detector.detect(
        {
            "source_layer": "process",
            "stop_reason": "completed" if returncode == 0 else "process_exit",
            "process_returncode": int(returncode),
        },
        returncode=int(returncode),
        firmware=str(firmware),
        input_id=f"p2im_24h_{target}",
        metadata={"target": target, "source_layer": "process"},
    )
    return report.to_dict()


def latest_progress_snapshot(progress_path: Path) -> Dict[str, Any]:
    records = load_jsonl(progress_path)
    return records[-1] if records else {}


def run_one_case(args: argparse.Namespace, target: str, campaign_root: Path) -> Dict[str, Any]:
    paths = target_paths(target, Path(args.firmware_root), Path(args.valid_root))
    firmware = paths["firmware"].resolve()
    valid_bb = paths["valid_bb"].resolve()
    case_dir = campaign_root / target
    case_dir.mkdir(parents=True, exist_ok=True)
    files = find_case_files(case_dir, firmware)

    if not firmware.exists():
        result = {
            "target": target,
            "status": "missing_firmware",
            "firmware": str(firmware),
            "valid_bb_file": str(valid_bb),
        }
        write_json(case_dir / "case_summary.json", result)
        return result

    if args.resume and (files["report"].exists() or files["simple_report"].exists()):
        summary = summarize_one_case(target, campaign_root, args)
        summary["status"] = "completed_cached"
        write_json(case_dir / "case_summary.json", summary)
        return summary

    command = build_command(
        python=args.python,
        firmware=firmware,
        output_dir=case_dir,
        duration_minutes=float(args.duration_minutes),
        run_profile=Path(args.run_profile).resolve() if args.run_profile else None,
        baseline_timeout_seconds=float(args.baseline_timeout_seconds),
        lsgemu_args=list(args.lsgemu_arg or []),
    )

    env = os.environ.copy()
    env["LSGEMU_VALID_BB_ROOT"] = str(Path(args.valid_root).resolve())
    env["LSGEMU_PROGRESS_INTERVAL_SECONDS"] = str(int(args.interval_minutes * 60))
    env["LSGEMU_COVERAGE_CHECKPOINT_INTERVAL"] = str(int(args.interval_minutes * 60))
    env["LSGEMU_PROGRESS_JSONL"] = str(files["progress"])
    env.update(parse_key_value(args.env or []))

    manifest = {
        "target": target,
        "command": command,
        "firmware": str(firmware),
        "valid_bb_file": str(valid_bb),
        "valid_total_bbs_from_file": load_valid_total(valid_bb),
        "output_dir": str(case_dir),
            "duration_minutes": float(args.duration_minutes),
            "interval_minutes": int(args.interval_minutes),
            "baseline_timeout_seconds": float(args.baseline_timeout_seconds),
            "environment": {
                key: env.get(key)
                for key in sorted(env)
            if key.startswith("LSGEMU_")
        },
    }
    write_json(case_dir / "case_manifest.json", manifest)
    if args.dry_run:
        reference = REFERENCE_RESULTS.get(target, {})
        multifuzz = reference.get("MultiFuzz", {})
        fuzzware = reference.get("Fuzzware", {})
        result = {
            **manifest,
            "status": "dry_run",
            "valid_total_bbs": manifest["valid_total_bbs_from_file"],
            "valid_covered_bbs": None,
            "valid_coverage_rate": None,
            "multifuzz_valid_covered_bbs": multifuzz.get("valid_covered_bbs"),
            "multifuzz_valid_coverage_rate": multifuzz.get("valid_coverage_rate"),
            "multifuzz_tool_crashes": multifuzz.get("tool_crashes"),
            "fuzzware_valid_covered_bbs": fuzzware.get("valid_covered_bbs"),
            "fuzzware_valid_coverage_rate": fuzzware.get("valid_coverage_rate"),
            "fuzzware_tool_crashes": fuzzware.get("tool_crashes"),
        }
        write_json(case_dir / "case_summary.json", result)
        return result

    timeout_seconds = None
    if args.supervisor_timeout_hours > 0:
        timeout_seconds = float(args.supervisor_timeout_hours * 3600)

    started = time.time()
    append_jsonl(
        campaign_root / "campaign_progress.jsonl",
        {"event": "case_start", "target": target, "time": started, "command": command},
    )
    returncode: Optional[int] = None
    timed_out = False
    with files["log"].open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = started + timeout_seconds if timeout_seconds else None
        last_status = 0.0
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            now = time.time()
            if deadline and now >= deadline:
                timed_out = True
                process.terminate()
                try:
                    returncode = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=60)
                break
            if now - last_status >= max(30, int(args.status_interval_seconds)):
                last_status = now
                write_live_status(campaign_root, target, started, files["progress"])
            time.sleep(10)

    assert returncode is not None
    process_crash = crash_report_for_process(returncode, firmware, target)
    if timed_out:
        process_crash["timed_out_by_supervisor"] = True
        process_crash["category"] = "timeout"
        process_crash["classification"] = "supervisor_timeout"
        process_crash["source_layer"] = "campaign_supervisor"
    write_json(files["process_report"], process_crash)

    summary = summarize_one_case(target, campaign_root, args)
    summary.update(
        {
            "status": "completed" if returncode == 0 and not timed_out else ("timeout" if timed_out else "failed"),
            "exit_code": int(returncode),
            "supervisor_timed_out": bool(timed_out),
            "wallclock_seconds": time.time() - started,
            "process_crash_report": process_crash,
        }
    )
    write_json(case_dir / "case_summary.json", summary)
    append_jsonl(campaign_root / "campaign_progress.jsonl", {"event": "case_end", **summary})
    return summary


def bucket_progress(
    records: List[Dict[str, Any]],
    *,
    duration_minutes: float,
    interval_minutes: int,
) -> List[Dict[str, Any]]:
    sorted_records = sorted(records, key=lambda item: safe_float(item.get("elapsed_seconds"), 0.0))
    interval_seconds = max(1, int(interval_minutes * 60))
    total_buckets = int(duration_minutes * 60 // interval_seconds)
    trend: List[Dict[str, Any]] = []
    cursor = 0
    last_seen: Optional[Dict[str, Any]] = None
    for bucket_index in range(total_buckets + 1):
        threshold = bucket_index * interval_seconds
        while cursor < len(sorted_records):
            elapsed = safe_float(sorted_records[cursor].get("elapsed_seconds"), 0.0)
            if elapsed > threshold + 1e-6:
                break
            last_seen = sorted_records[cursor]
            cursor += 1
        record = last_seen or (sorted_records[0] if sorted_records else {})
        trend.append(
            {
                "minute": int(threshold // 60),
                "elapsed_seconds": safe_float(record.get("elapsed_seconds"), 0.0) if record else None,
                "stage": record.get("stage") if record else None,
                "event": record.get("event") if record else None,
                "covered_bbs": record.get("covered_bbs") if record else None,
                "coverage_rate": record.get("coverage_rate") if record else None,
                "valid_covered_bbs": record.get("valid_covered_bbs") if record else None,
                "valid_total_bbs": record.get("valid_total_bbs") if record else None,
                "valid_coverage_rate": record.get("valid_coverage_rate") if record else None,
                "strict_real_entry_replayable": record.get("strict_real_entry_replayable") if record else None,
                "coverage_entry_derived": record.get("coverage_entry_derived") if record else None,
            }
        )
    return trend


def extract_report_metrics(report: Dict[str, Any], valid_total_hint: int = 0) -> Dict[str, Any]:
    valid_total = safe_int(report.get("valid_total_bbs"), valid_total_hint)
    valid_covered = safe_int(report.get("valid_covered_bbs"), 0)
    covered = safe_int(report.get("covered_bbs"), 0)
    total = safe_int(report.get("total_bbs"), 0)
    crash_summary = report.get("crash_triage_summary") or {}
    if not isinstance(crash_summary, dict):
        crash_summary = {}
    return {
        "runner": report.get("runner"),
        "coverage_source": report.get("coverage_source"),
        "coverage_entry_derived": report.get("coverage_entry_derived"),
        "strict_real_entry_replayable": report.get("strict_real_entry_replayable"),
        "coverage_counting_rule": report.get("coverage_counting_rule"),
        "covered_bbs": covered,
        "total_bbs": total,
        "coverage_rate": safe_float(report.get("coverage_rate"), percent(covered, total)),
        "valid_covered_bbs": valid_covered,
        "valid_total_bbs": valid_total,
        "valid_coverage_rate": safe_float(report.get("valid_coverage_rate"), percent(valid_covered, valid_total)),
        "execution_time_seconds": safe_float(report.get("execution_time_seconds"), 0.0),
        "valid_entry_static_reachable_bbs": safe_int(report.get("valid_entry_static_reachable_bbs"), 0),
        "valid_entry_static_reachable_covered_bbs": safe_int(
            report.get("valid_entry_static_reachable_covered_bbs"),
            0,
        ),
        "valid_entry_plus_vector_static_reachable_bbs": safe_int(
            report.get("valid_entry_plus_vector_static_reachable_bbs"),
            0,
        ),
        "valid_entry_plus_vector_static_reachable_covered_bbs": safe_int(
            report.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
            0,
        ),
        "valid_extended_static_reachable_bbs": safe_int(report.get("valid_extended_static_reachable_bbs"), 0),
        "valid_extended_static_reachable_covered_bbs": safe_int(
            report.get("valid_extended_static_reachable_covered_bbs"),
            0,
        ),
        "valid_static_unreachable_from_entry_or_vectors": safe_int(
            report.get("valid_static_unreachable_from_entry_or_vectors"),
            0,
        ),
        "valid_static_unreachable_from_extended_roots": safe_int(
            report.get("valid_static_unreachable_from_extended_roots"),
            0,
        ),
        "crash_triage_total_reports": safe_int(crash_summary.get("total_reports"), 0),
        "firmware_crashes": safe_int(crash_summary.get("firmware_crashes"), 0),
        "hangs": safe_int(crash_summary.get("hangs"), 0),
        "model_artifacts": safe_int(crash_summary.get("model_artifacts"), 0),
        "tooling_failures": safe_int(crash_summary.get("tooling_failures"), 0),
        "requires_replay_validation": safe_int(crash_summary.get("requires_replay_validation"), 0),
        "crash_triage_categories": crash_summary.get("categories") or {},
        "crash_triage_classifications": crash_summary.get("classifications") or {},
    }


def summarize_one_case(target: str, campaign_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    paths = target_paths(target, Path(args.firmware_root), Path(args.valid_root))
    firmware = paths["firmware"].resolve()
    valid_bb = paths["valid_bb"].resolve()
    case_dir = campaign_root / target
    files = find_case_files(case_dir, firmware)
    report_path = files["report"] if files["report"].exists() else files["simple_report"]
    report = read_json(report_path)
    progress_records = load_jsonl(files["progress"])
    valid_total_hint = load_valid_total(valid_bb)
    metrics = extract_report_metrics(report, valid_total_hint=valid_total_hint)
    trend = bucket_progress(
        progress_records,
        duration_minutes=float(args.duration_minutes),
        interval_minutes=int(args.interval_minutes),
    )
    reference = REFERENCE_RESULTS.get(target, {})
    multifuzz = reference.get("MultiFuzz", {})
    fuzzware = reference.get("Fuzzware", {})
    lsgemu_valid = safe_int(metrics.get("valid_covered_bbs"), 0)
    return {
        "target": target,
        "status": "completed" if report_path.exists() else "missing_report",
        "firmware": str(firmware),
        "valid_bb_file": str(valid_bb),
        "valid_total_bbs_from_file": valid_total_hint,
        "output_dir": str(case_dir),
        "report_file": str(report_path) if report_path.exists() else "",
        "progress_file": str(files["progress"]),
        "log_file": str(files["log"]),
        "process_report_file": str(files["process_report"]),
        **metrics,
        "progress_points": len(progress_records),
        "progress_trend": trend,
        "last_progress": progress_records[-1] if progress_records else None,
        "multifuzz_valid_covered_bbs": multifuzz.get("valid_covered_bbs"),
        "multifuzz_valid_coverage_rate": multifuzz.get("valid_coverage_rate"),
        "multifuzz_tool_crashes": multifuzz.get("tool_crashes"),
        "fuzzware_valid_covered_bbs": fuzzware.get("valid_covered_bbs"),
        "fuzzware_valid_coverage_rate": fuzzware.get("valid_coverage_rate"),
        "fuzzware_tool_crashes": fuzzware.get("tool_crashes"),
        "delta_vs_multifuzz_bbs": (
            lsgemu_valid - safe_int(multifuzz.get("valid_covered_bbs"), 0)
            if multifuzz.get("valid_covered_bbs") is not None
            else None
        ),
        "delta_vs_fuzzware_bbs": (
            lsgemu_valid - safe_int(fuzzware.get("valid_covered_bbs"), 0)
            if fuzzware.get("valid_covered_bbs") is not None
            else None
        ),
    }


def write_live_status(campaign_root: Path, current_target: str, started: float, progress_path: Path) -> None:
    snapshot = latest_progress_snapshot(progress_path)
    write_json(
        campaign_root / "campaign_live_status.json",
        {
            "current_target": current_target,
            "elapsed_seconds": time.time() - started,
            "last_progress": snapshot,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def write_outputs(campaign_root: Path, summaries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    write_json(
        campaign_root / "campaign_summary.json",
        {
            "runner": "lsgemu_p2im_24h_comparison",
            "duration_minutes": float(args.duration_minutes),
            "interval_minutes": int(args.interval_minutes),
            "baseline_timeout_seconds": float(args.baseline_timeout_seconds),
            "valid_root": str(Path(args.valid_root).resolve()),
            "firmware_root": str(Path(args.firmware_root).resolve()),
            "reference_results": REFERENCE_RESULTS,
            "results": summaries,
        },
    )

    summary_fields = [
        "target",
        "status",
        "valid_covered_bbs",
        "valid_total_bbs",
        "valid_coverage_rate",
        "covered_bbs",
        "total_bbs",
        "coverage_rate",
        "strict_real_entry_replayable",
        "coverage_entry_derived",
        "firmware_crashes",
        "hangs",
        "model_artifacts",
        "tooling_failures",
        "requires_replay_validation",
        "multifuzz_valid_covered_bbs",
        "multifuzz_valid_coverage_rate",
        "multifuzz_tool_crashes",
        "delta_vs_multifuzz_bbs",
        "fuzzware_valid_covered_bbs",
        "fuzzware_valid_coverage_rate",
        "fuzzware_tool_crashes",
        "delta_vs_fuzzware_bbs",
        "execution_time_seconds",
        "report_file",
        "progress_file",
    ]
    summary_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(summary_buffer, fieldnames=summary_fields)
    writer.writeheader()
    for item in summaries:
        writer.writerow({key: item.get(key) for key in summary_fields})
    atomic_write_text(campaign_root / "campaign_summary.csv", summary_buffer.getvalue(), durable=False)

    timeline_fields = [
        "target",
        "minute",
        "elapsed_seconds",
        "stage",
        "event",
        "valid_covered_bbs",
        "valid_total_bbs",
        "valid_coverage_rate",
        "covered_bbs",
        "coverage_rate",
        "strict_real_entry_replayable",
        "coverage_entry_derived",
    ]
    timeline_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(timeline_buffer, fieldnames=timeline_fields)
    writer.writeheader()
    for item in summaries:
        for point in item.get("progress_trend") or []:
            writer.writerow({"target": item.get("target"), **{key: point.get(key) for key in timeline_fields if key != "target"}})
    atomic_write_text(campaign_root / "coverage_timeline.csv", timeline_buffer.getvalue(), durable=False)

    write_json(
        campaign_root / "coverage_timeline.json",
        {
            item.get("target"): item.get("progress_trend") or []
            for item in summaries
        },
    )
    write_markdown_report(campaign_root, summaries, args)


def write_markdown_report(campaign_root: Path, summaries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# LSGEmu P2IM 24h Coverage Comparison",
        "",
        f"- Duration: {float(args.duration_minutes):.1f} minutes per firmware",
        f"- Checkpoint interval: {int(args.interval_minutes)} minutes",
        f"- Baseline timeout: {float(args.baseline_timeout_seconds):.1f} seconds",
        f"- Valid BB root: `{Path(args.valid_root).resolve()}`",
        f"- Output root: `{campaign_root}`",
        "",
        "Crash columns are triage categories, not confirmed vulnerabilities.",
        "",
        "## End-to-End Summary",
        "",
        "| Firmware | LSGEmu valid BB | LSGEmu valid % | MultiFuzz valid BB | Δ vs MultiFuzz | Fuzzware valid BB | Δ vs Fuzzware | LSGEmu firmware crash candidates | Model artifacts | Tool failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {target} | {lv}/{lt} | {lr:.2f}% | {mv} | {dm} | {fv} | {df} | {fc} | {ma} | {tf} |".format(
                target=item.get("target"),
                lv=safe_int(item.get("valid_covered_bbs"), 0),
                lt=safe_int(item.get("valid_total_bbs"), 0),
                lr=safe_float(item.get("valid_coverage_rate"), 0.0),
                mv=item.get("multifuzz_valid_covered_bbs"),
                dm=item.get("delta_vs_multifuzz_bbs"),
                fv=item.get("fuzzware_valid_covered_bbs"),
                df=item.get("delta_vs_fuzzware_bbs"),
                fc=safe_int(item.get("firmware_crashes"), 0),
                ma=safe_int(item.get("model_artifacts"), 0),
                tf=safe_int(item.get("tooling_failures"), 0),
            )
        )
    lines.extend(
        [
            "",
            "## Timeline Files",
            "",
            "- `coverage_timeline.csv`: one row per target per checkpoint bucket.",
            "- `coverage_timeline.json`: same data as structured JSON.",
            "- Per-target `*_coverage_progress.jsonl`: raw LSGEmu progress events.",
            "",
            "## Strictness Fields",
            "",
            "- `coverage_entry_derived=True` means counted BBs came from entry-derived execution/replay rather than arbitrary jump-to-BB probes.",
            "- `strict_real_entry_replayable=True` means the final report considers the counted coverage replayable under the strict entry-derived contract.",
        ]
    )
    atomic_write_text(campaign_root / "comparison_report.md", "\n".join(lines) + "\n", durable=False)


def write_campaign_manifest(campaign_root: Path, targets: List[str], args: argparse.Namespace) -> None:
    cases = []
    for target in targets:
        paths = target_paths(target, Path(args.firmware_root), Path(args.valid_root))
        cases.append(
            {
                "target": target,
                "firmware": str(paths["firmware"].resolve()),
                "valid_bb_file": str(paths["valid_bb"].resolve()),
                "valid_total_bbs_from_file": load_valid_total(paths["valid_bb"]),
            }
        )
    write_json(
        campaign_root / "campaign_manifest.json",
        {
            "runner": "lsgemu_p2im_24h_comparison",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duration_minutes": float(args.duration_minutes),
            "interval_minutes": int(args.interval_minutes),
            "baseline_timeout_seconds": float(args.baseline_timeout_seconds),
            "targets": cases,
            "reference_documents": [
                str(MULTIFUZZ_ROOT / "docs/24h_bb_coverage_analysis.md"),
                str(FUZZWARE_ROOT / "docs/24h_bb_coverage_analysis.md"),
            ],
            "reference_results": REFERENCE_RESULTS,
        },
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="CNC,Gateway,PLC", help="Comma-separated targets: CNC,Gateway,PLC")
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--duration-minutes", type=float, default=None)
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument(
        "--baseline-timeout-seconds",
        type=float,
        default=600.0,
        help=(
            "Forward a wallclock cap to the initial reset-entry baseline stage. "
            "Set <=0 to disable. This prevents polling loops from consuming a "
            "whole long campaign before interleaved replay starts."
        ),
    )
    parser.add_argument("--firmware-root", default=str(DEFAULT_FIRMWARE_ROOT))
    parser.add_argument("--valid-root", default=str(DEFAULT_VALID_ROOT))
    parser.add_argument(
        "--output-root",
        default="",
        help="Campaign output directory. Default creates .lsgemu_runs/p2im_24h_lsgemu_comparison_<timestamp>.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-profile", default="")
    parser.add_argument("--resume", action="store_true", help="Skip targets with an existing LSGEmu report.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize an existing output root.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests and commands but do not execute.")
    parser.add_argument(
        "--supervisor-timeout-hours",
        type=float,
        default=30.0,
        help="Outer watchdog per firmware. Set <=0 to disable. Default allows 24h plus finalization.",
    )
    parser.add_argument("--status-interval-seconds", type=int, default=60)
    parser.add_argument(
        "--lsgemu-arg",
        action="append",
        default=[],
        help="Append one raw argument to lsgemu.py/interleaved scheduler. Repeat for option/value pairs.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Add an environment override for child runs, e.g. LSGEMU_FOO=1.",
    )
    args = parser.parse_args(argv)
    if args.duration_minutes is None:
        args.duration_minutes = float(args.duration_hours) * 60.0
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    targets = selected_target_names(args.targets)
    campaign_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (DEFAULT_OUTPUT_ROOT / f"p2im_24h_lsgemu_comparison_{now_stamp()}").resolve()
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    write_campaign_manifest(campaign_root, targets, args)

    summaries: List[Dict[str, Any]] = []
    if args.summarize_only:
        for target in targets:
            summaries.append(summarize_one_case(target, campaign_root, args))
        write_outputs(campaign_root, summaries, args)
        print(f"summary written to {campaign_root}")
        return 0

    for target in targets:
        print(f"[LSGEmu-24h] {target}: start")
        summary = run_one_case(args, target, campaign_root)
        summaries.append(summary)
        write_outputs(campaign_root, summaries, args)
        print(
            "[LSGEmu-24h] {target}: {status} valid={covered}/{total} ({rate:.2f}%)".format(
                target=target,
                status=summary.get("status"),
                covered=safe_int(summary.get("valid_covered_bbs"), 0),
                total=safe_int(summary.get("valid_total_bbs"), 0),
                rate=safe_float(summary.get("valid_coverage_rate"), 0.0),
            )
        )

    write_outputs(campaign_root, summaries, args)
    print(f"campaign output: {campaign_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
