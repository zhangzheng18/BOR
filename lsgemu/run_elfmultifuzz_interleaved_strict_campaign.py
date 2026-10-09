#!/usr/bin/env python3
"""Run the strict interleaved coverage workflow across all elfmultifuzz ELFs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import DEFAULT_VALID_BB_ROOT, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from lsgemu.stat_elfmultifuzz_valid_coverage import discover_cases
else:
    from .deployment_config import apply_config_from_argv, configured_path
    from .historical_runner import DEFAULT_VALID_BB_ROOT, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from .stat_elfmultifuzz_valid_coverage import discover_cases


apply_config_from_argv()

SOURCE_ROOT = configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4")
GLOBAL_RUNS_ROOT = configured_path("LSGEMU_RUN_OUTPUT_DIR", SOURCE_ROOT / ".lsgemu_runs")
INTERLEAVED_SCRIPT = SOURCE_ROOT / "lsgemu" / "cached_interleaved_runner.py"
INTERLEAVED_MODULE = "lsgemu.cached_interleaved_runner"
LOCAL_UNICORN_BUILD = configured_path("LIBUNICORN_PATH", RUNNER_PROJECT_ROOT / "unicorn" / "build")
LOCAL_UNICORN_SHARED_LIB = configured_path("LSGEMU_UNICORN_SHARED_LIB", LOCAL_UNICORN_BUILD / "libunicorn.so.2")
# Extra wall-clock grace beyond the per-case --total-time-minutes budget
# before the serial-campaign watchdog kills a stuck child process group.
WATCHDOG_GRACE_SECONDS = 600
COMPLETED_STATUSES = {
    "completed",
    "completed_cached",
    "completed_best_progress_after_retry",
    "completed_strict_checkpoint_union",
    "completed_partial_strict_resource_union",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_csv_filter(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def case_matches(case, families: list[str], names: list[str]) -> bool:
    if families and case.family.lower() not in families:
        return False
    if names:
        haystacks = {
            case.name.lower(),
            case.rel_path.lower(),
            case.elf_path.name.lower(),
            case.elf_path.stem.lower(),
        }
        if not any(any(token in hay for hay in haystacks) for token in names):
            return False
    return True


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    try:
        with path.open() as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return records
    return records


def load_last_jsonl_record(path: Path) -> dict[str, object] | None:
    records = load_jsonl_records(path)
    return records[-1] if records else None


def bucket_progress(
    records: Iterable[dict[str, object]],
    *,
    interval_seconds: int,
    duration_minutes: int,
) -> list[dict[str, object] | None]:
    sorted_records = sorted(
        (record for record in records if isinstance(record, dict)),
        key=lambda item: float(item.get("elapsed_seconds") or 0.0),
    )
    buckets: list[dict[str, object] | None] = []
    cursor = 0
    last_seen: dict[str, object] | None = None
    bucket_seconds = max(1, int(interval_seconds))
    total_buckets = max(0, int(duration_minutes * 60 // bucket_seconds))
    for bucket_index in range(total_buckets + 1):
        threshold = bucket_index * bucket_seconds
        while cursor < len(sorted_records):
            elapsed = float(sorted_records[cursor].get("elapsed_seconds") or 0.0)
            if elapsed > threshold + 1e-6:
                break
            last_seen = sorted_records[cursor]
            cursor += 1
        if bucket_index == 0 and last_seen is None and sorted_records:
            last_seen = sorted_records[0]
        buckets.append(last_seen)
    return buckets


def format_bucket_label(index: int, interval_seconds: int, *, width: int = 2) -> str:
    elapsed_seconds = int(index * max(1, int(interval_seconds)))
    if elapsed_seconds % 60 == 0:
        return f"{elapsed_seconds // 60:0{width}d}m"
    minutes, seconds = divmod(elapsed_seconds, 60)
    return f"{minutes:0{width}d}m{seconds:02d}s"


def load_report_summary(path: Path) -> dict[str, object]:
    payload = load_json(path) or {}
    static_reachability = payload.get("static_reachability") or {}
    uncovered_summary = payload.get("uncovered_summary") or {}
    phase_metadata = payload.get("phase_metadata") or {}
    extended_static_roots_available_value = payload.get("extended_static_roots_available")
    if not isinstance(extended_static_roots_available_value, bool):
        extended_static_roots_available_value = static_reachability.get("extended_static_roots_available")
    extended_static_roots_available = (
        bool(extended_static_roots_available_value)
        if isinstance(extended_static_roots_available_value, bool)
        else bool(
            payload.get("extended_static_root_bbs")
            or static_reachability.get("extended_static_root_bbs")
            or payload.get("extended_static_root_categories")
            or static_reachability.get("extended_static_root_categories")
        )
    )
    frontier_phases = [
        phase
        for name, phase in phase_metadata.items()
        if isinstance(phase, dict) and "frontier_successor_replay" in str(name)
    ]
    frontier_tasks = sum(int(phase.get("tasks_run") or 0) for phase in frontier_phases)
    frontier_replay_targets = sum(int(phase.get("target_bbs_discovered") or 0) for phase in frontier_phases)
    frontier_forced_verified = sum(int(phase.get("forced_choices_verified") or 0) for phase in frontier_phases)
    top_frontiers = uncovered_summary.get("top_frontier_predecessors") or []
    if not isinstance(top_frontiers, list):
        top_frontiers = []
    top_frontier_text = "; ".join(
        f"{item.get('bb')}:{item.get('mnemonic')}->{','.join(item.get('sample_uncovered_successors') or [])}"
        for item in top_frontiers[:3]
        if isinstance(item, dict)
    )
    switch_frontiers = uncovered_summary.get("top_switch_frontiers") or []
    if not isinstance(switch_frontiers, list):
        switch_frontiers = []
    call_frontiers = uncovered_summary.get("top_call_frontiers") or []
    if not isinstance(call_frontiers, list):
        call_frontiers = []
    frontier_ready = 0
    frontier_no_snapshot = 0
    frontier_choice_unmapped = 0
    frontier_already_covered = 0
    frontier_snapshot_attempts = 0
    direct_tasks = 0
    direct_targets = 0
    direct_low_yield_stops = 0
    summary_tasks = 0
    summary_targets = 0
    summary_low_yield_stops = 0
    rtos_tasks = 0
    rtos_targets = 0
    for name, phase in phase_metadata.items():
        if not isinstance(phase, dict):
            continue
        diag = phase.get("frontier_successor_candidate_diagnostics") or {}
        if isinstance(diag, dict):
            reasons = diag.get("reason_counts") or {}
            if isinstance(reasons, dict):
                frontier_ready += int(reasons.get("candidate_ready") or 0)
                frontier_no_snapshot += int(reasons.get("no_root_snapshot") or 0)
                frontier_choice_unmapped += int(reasons.get("choice_unmapped") or 0)
                frontier_already_covered += int(reasons.get("successor_already_covered") or 0)
        frontier_snapshot_attempts += int(phase.get("frontier_successor_snapshot_attempts") or 0)
        if "direct_call_continuation" in str(name) or str(name).endswith("_direct_call"):
            direct_tasks += int(phase.get("tasks_run") or 0)
            direct_targets += int(phase.get("target_bbs_discovered") or 0)
            direct_low_yield_stops += 1 if phase.get("low_yield_stop_triggered") else 0
        if "direct_call_summary_return" in str(name):
            summary_tasks += int(phase.get("tasks_run") or 0)
            summary_targets += int(phase.get("target_bbs_discovered") or 0)
            summary_low_yield_stops += 1 if phase.get("low_yield_stop_triggered") else 0
        if "rtos_thread_entry" in str(name):
            rtos_tasks += int(phase.get("tasks_run") or 0)
            rtos_targets += int(phase.get("target_bbs_discovered") or 0)

    valid_total = int(payload.get("valid_total_bbs") or 0)
    valid_static_unreachable = int(
        payload.get("valid_static_unreachable_from_entry_or_vectors")
        or static_reachability.get("valid_static_unreachable_from_entry_or_vectors")
        or 0
    )
    valid_extended_static_unreachable = int(
        payload.get("valid_static_unreachable_from_extended_roots")
        or static_reachability.get("valid_static_unreachable_from_extended_roots")
        or 0
    )
    entry_plus_total = int(
        payload.get("valid_entry_plus_vector_static_reachable_bbs")
        or static_reachability.get("valid_entry_plus_vector_static_reachable_bbs")
        or 0
    )
    extended_total = int(
        payload.get("valid_extended_static_reachable_bbs")
        or static_reachability.get("valid_extended_static_reachable_bbs")
        or 0
    )
    extended_covered = int(
        payload.get("valid_extended_static_reachable_covered_bbs")
        or static_reachability.get("valid_extended_static_reachable_covered_bbs")
        or 0
    )
    entry_plus_uncovered = int(
        static_reachability.get("valid_entry_plus_vector_static_reachable_uncovered_bbs")
        or 0
    )
    extended_uncovered = int(
        payload.get("valid_extended_static_reachable_uncovered_bbs")
        or static_reachability.get("valid_extended_static_reachable_uncovered_bbs")
        or max(0, extended_total - extended_covered)
    )
    frontier_targets = int(uncovered_summary.get("frontier_target_bbs") or 0)
    covered_pred_gaps = int(uncovered_summary.get("uncovered_with_covered_predecessor_bbs") or 0)
    switch_gap_count = len(switch_frontiers)
    call_gap_count = len(call_frontiers)
    static_unreachable_ratio = (valid_static_unreachable / valid_total * 100.0) if valid_total else 0.0
    extended_static_unreachable_ratio = (valid_extended_static_unreachable / valid_total * 100.0) if valid_total else 0.0
    replay_conversion_rate = (frontier_replay_targets / frontier_tasks * 100.0) if frontier_tasks else 0.0

    root_cause_parts: list[str] = []
    if valid_total and static_unreachable_ratio >= 25.0:
        if (
            extended_static_roots_available
            and extended_static_unreachable_ratio < static_unreachable_ratio * 0.75
        ):
            root_cause_parts.append("entry_vector_root_under_modeling")
        else:
            root_cause_parts.append("static_unreachable_denominator")
    if entry_plus_total and entry_plus_uncovered / entry_plus_total >= 0.25:
        root_cause_parts.append("entry_or_irq_state_prefix")
    if (
        extended_static_roots_available
        and extended_total
        and extended_uncovered / extended_total >= 0.25
    ):
        root_cause_parts.append("extended_root_state_prefix")
    if frontier_targets > 0 or covered_pred_gaps > 0:
        root_cause_parts.append("frontier_branch_not_converted")
    if switch_gap_count > 0:
        root_cause_parts.append("dynamic_dispatch_switch")
    if call_gap_count > 0:
        root_cause_parts.append("call_like_return_path")
    if frontier_ready > 0 and frontier_tasks == 0:
        root_cause_parts.append("frontier_replay_not_scheduled")
    elif frontier_tasks > 0 and replay_conversion_rate < 25.0 and frontier_targets > 0:
        root_cause_parts.append("frontier_replay_low_conversion")
    if frontier_no_snapshot > frontier_ready and frontier_no_snapshot > 0:
        root_cause_parts.append("missing_entry_derived_snapshots")
    if not root_cause_parts:
        root_cause_parts.append("mostly_covered_or_needs_manual_triage")

    return {
        "runner": payload.get("runner"),
        "stall_watchdog_stop_reason": payload.get("stall_watchdog_stop_reason"),
        "stall_watchdog": payload.get("stall_watchdog"),
        "stall_watchdog_stage_truncations": payload.get(
            "stall_watchdog_stage_truncations"
        ),
        "terminated_early": bool(payload.get("terminated_early", False)),
        "terminated_early_reason": payload.get("terminated_early_reason"),
        "wallclock_used_seconds": payload.get("wallclock_used_seconds"),
        "configured_budget_seconds": payload.get("configured_budget_seconds"),
        "stall_watchdog_budget_remaining_seconds": payload.get(
            "stall_watchdog_budget_remaining_seconds"
        ),
        "strict_real_entry_replayable": bool(payload.get("strict_real_entry_replayable", False)),
        "coverage_entry_derived": bool(payload.get("coverage_entry_derived", False)),
        "coverage_source": payload.get("coverage_source"),
        "loaded_unicorn_library": payload.get("loaded_unicorn_library"),
        "execution_time_seconds": float(payload.get("execution_time_seconds") or 0.0),
        "covered_bbs": int(payload.get("covered_bbs") or 0),
        "total_bbs": int(payload.get("total_bbs") or 0),
        "coverage_rate": float(payload.get("coverage_rate") or 0.0),
        "valid_covered_bbs": int(payload.get("valid_covered_bbs") or 0),
        "valid_total_bbs": int(payload.get("valid_total_bbs") or 0),
        "valid_coverage_rate": float(payload.get("valid_coverage_rate") or 0.0),
        "valid_entry_plus_vector_static_reachable_bbs": int(
            payload.get("valid_entry_plus_vector_static_reachable_bbs")
            or static_reachability.get("valid_entry_plus_vector_static_reachable_bbs")
            or 0
        ),
        "valid_entry_plus_vector_static_reachable_covered_bbs": int(
            payload.get("valid_entry_plus_vector_static_reachable_covered_bbs")
            or static_reachability.get("valid_entry_plus_vector_static_reachable_covered_bbs")
            or 0
        ),
        "valid_entry_static_reachable_bbs": int(
            payload.get("valid_entry_static_reachable_bbs")
            or static_reachability.get("valid_entry_static_reachable_bbs")
            or 0
        ),
        "valid_entry_static_reachable_covered_bbs": int(
            payload.get("valid_entry_static_reachable_covered_bbs")
            or static_reachability.get("valid_entry_static_reachable_covered_bbs")
            or 0
        ),
        "valid_entry_static_reachable_coverage_rate": float(
            payload.get("valid_entry_static_reachable_coverage_rate")
            or static_reachability.get("valid_entry_static_reachable_coverage_rate")
            or 0.0
        ),
        "valid_entry_plus_vector_reachable_coverage_rate": float(
            payload.get("valid_entry_plus_vector_reachable_coverage_rate")
            or static_reachability.get("valid_entry_plus_vector_reachable_coverage_rate")
            or 0.0
        ),
        "valid_static_unreachable_from_entry_or_vectors": valid_static_unreachable,
        "extended_static_roots_available": bool(extended_static_roots_available),
        "extended_static_root_bbs": int(
            payload.get("extended_static_root_bbs")
            or static_reachability.get("extended_static_root_bbs")
            or 0
        ),
        "extended_static_root_categories": (
            payload.get("extended_static_root_categories")
            or static_reachability.get("extended_static_root_categories")
            or {}
        ),
        "valid_extended_static_reachable_bbs": int(extended_total),
        "valid_extended_static_reachable_covered_bbs": int(extended_covered),
        "valid_extended_static_reachable_uncovered_bbs": int(extended_uncovered),
        "valid_extended_static_reachable_coverage_rate": float(
            payload.get("valid_extended_static_reachable_coverage_rate")
            or static_reachability.get("valid_extended_static_reachable_coverage_rate")
            or ((extended_covered / extended_total * 100.0) if extended_total else 0.0)
        ),
        "valid_static_unreachable_from_extended_roots": valid_extended_static_unreachable,
        "entry_reachable_uncovered_bbs": int(uncovered_summary.get("entry_reachable_uncovered_bbs") or 0),
        "vector_only_uncovered_bbs": int(uncovered_summary.get("vector_only_uncovered_bbs") or 0),
        "entry_plus_vector_uncovered_bbs": entry_plus_uncovered,
        "extended_static_reachable_uncovered_bbs": extended_uncovered,
        "frontier_target_bbs": frontier_targets,
        "uncovered_with_covered_predecessor_bbs": covered_pred_gaps,
        "frontier_successor_tasks": int(frontier_tasks),
        "frontier_successor_targets": int(frontier_replay_targets),
        "frontier_successor_forced_verified": int(frontier_forced_verified),
        "frontier_successor_snapshot_attempts": int(frontier_snapshot_attempts),
        "frontier_candidate_ready": int(frontier_ready),
        "frontier_no_root_snapshot": int(frontier_no_snapshot),
        "frontier_choice_unmapped": int(frontier_choice_unmapped),
        "frontier_successor_already_covered": int(frontier_already_covered),
        "switch_frontier_gap_count": int(switch_gap_count),
        "call_frontier_gap_count": int(call_gap_count),
        "direct_call_tasks": int(direct_tasks),
        "direct_call_targets": int(direct_targets),
        "direct_call_low_yield_stops": int(direct_low_yield_stops),
        "summary_return_tasks": int(summary_tasks),
        "summary_return_targets": int(summary_targets),
        "summary_return_low_yield_stops": int(summary_low_yield_stops),
        "rtos_thread_tasks": int(rtos_tasks),
        "rtos_thread_targets": int(rtos_targets),
        "static_unreachable_valid_ratio": float(static_unreachable_ratio),
        "extended_static_unreachable_valid_ratio": float(extended_static_unreachable_ratio),
        "frontier_replay_conversion_rate": float(replay_conversion_rate),
        "low_coverage_root_cause": ",".join(root_cause_parts),
        "top_frontier_predecessors": top_frontier_text,
    }


def summarize_case_result(
    *,
    case,
    case_output_dir: Path,
    report_path: Path,
    progress_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    exit_code: int | None,
    status: str,
    duration_minutes: int,
    interval_seconds: int,
) -> dict[str, object]:
    report_summary = load_report_summary(report_path) if report_path.exists() else {}
    progress_records = load_jsonl_records(progress_path)
    progress_buckets = bucket_progress(
        progress_records,
        interval_seconds=interval_seconds,
        duration_minutes=duration_minutes,
    )
    trend = []
    bucket_seconds = max(1, int(interval_seconds))
    for index, record in enumerate(progress_buckets):
        minute = int(index * bucket_seconds / 60)
        if record is None:
            trend.append({
                "minute": minute,
                "valid_covered_bbs": None,
                "valid_coverage_rate": None,
                "valid_entry_plus_vector_static_reachable_bbs": None,
                "valid_entry_plus_vector_static_reachable_covered_bbs": None,
                "valid_entry_plus_vector_reachable_coverage_rate": None,
                "extended_static_roots_available": None,
                "valid_extended_static_reachable_bbs": None,
                "valid_extended_static_reachable_covered_bbs": None,
                "valid_extended_static_reachable_coverage_rate": None,
                "covered_bbs": None,
                "coverage_rate": None,
            })
            continue
        trend.append({
            "minute": minute,
            "valid_covered_bbs": record.get("valid_covered_bbs"),
            "valid_coverage_rate": record.get("valid_coverage_rate"),
            "valid_entry_plus_vector_static_reachable_bbs": record.get("valid_entry_plus_vector_static_reachable_bbs"),
            "valid_entry_plus_vector_static_reachable_covered_bbs": record.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
            "valid_entry_plus_vector_reachable_coverage_rate": record.get("valid_entry_plus_vector_reachable_coverage_rate"),
            "extended_static_roots_available": record.get("extended_static_roots_available"),
            "valid_extended_static_reachable_bbs": record.get("valid_extended_static_reachable_bbs"),
            "valid_extended_static_reachable_covered_bbs": record.get("valid_extended_static_reachable_covered_bbs"),
            "valid_extended_static_reachable_coverage_rate": record.get("valid_extended_static_reachable_coverage_rate"),
            "covered_bbs": record.get("covered_bbs"),
            "coverage_rate": record.get("coverage_rate"),
        })

    return {
        "status": status,
        "exit_code": exit_code,
        "family": case.family,
        "name": case.name,
        "rel_path": case.rel_path,
        "firmware": str(case.elf_path.resolve()),
        "output_dir": str(case_output_dir),
        "report_file": str(report_path),
        "progress_jsonl_file": str(progress_path),
        "checkpoint_file": str(checkpoint_path),
        "log_file": str(log_path),
        "runner": report_summary.get("runner"),
        "stall_watchdog_stop_reason": report_summary.get("stall_watchdog_stop_reason"),
        "stall_watchdog": report_summary.get("stall_watchdog"),
        "stall_watchdog_stage_truncations": report_summary.get(
            "stall_watchdog_stage_truncations"
        ),
        "terminated_early": report_summary.get("terminated_early", False),
        "terminated_early_reason": report_summary.get("terminated_early_reason"),
        "wallclock_used_seconds": report_summary.get("wallclock_used_seconds"),
        "configured_budget_seconds": report_summary.get("configured_budget_seconds"),
        "stall_watchdog_budget_remaining_seconds": report_summary.get(
            "stall_watchdog_budget_remaining_seconds"
        ),
        "strict_real_entry_replayable": report_summary.get("strict_real_entry_replayable"),
        "coverage_entry_derived": report_summary.get("coverage_entry_derived"),
        "coverage_source": report_summary.get("coverage_source"),
        "loaded_unicorn_library": report_summary.get("loaded_unicorn_library"),
        "execution_time_seconds": report_summary.get("execution_time_seconds"),
        "covered_bbs": report_summary.get("covered_bbs"),
        "total_bbs": report_summary.get("total_bbs"),
        "coverage_rate": report_summary.get("coverage_rate"),
        "valid_covered_bbs": report_summary.get("valid_covered_bbs"),
        "valid_total_bbs": report_summary.get("valid_total_bbs"),
        "valid_coverage_rate": report_summary.get("valid_coverage_rate"),
        "valid_entry_plus_vector_static_reachable_bbs": report_summary.get("valid_entry_plus_vector_static_reachable_bbs"),
        "valid_entry_plus_vector_static_reachable_covered_bbs": report_summary.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
        "valid_entry_plus_vector_reachable_coverage_rate": report_summary.get("valid_entry_plus_vector_reachable_coverage_rate"),
        "valid_entry_static_reachable_bbs": report_summary.get("valid_entry_static_reachable_bbs"),
        "valid_entry_static_reachable_covered_bbs": report_summary.get("valid_entry_static_reachable_covered_bbs"),
        "valid_entry_static_reachable_coverage_rate": report_summary.get("valid_entry_static_reachable_coverage_rate"),
        "valid_static_unreachable_from_entry_or_vectors": report_summary.get("valid_static_unreachable_from_entry_or_vectors"),
        "extended_static_roots_available": report_summary.get("extended_static_roots_available"),
        "extended_static_root_bbs": report_summary.get("extended_static_root_bbs"),
        "extended_static_root_categories": report_summary.get("extended_static_root_categories"),
        "valid_extended_static_reachable_bbs": report_summary.get("valid_extended_static_reachable_bbs"),
        "valid_extended_static_reachable_covered_bbs": report_summary.get("valid_extended_static_reachable_covered_bbs"),
        "valid_extended_static_reachable_uncovered_bbs": report_summary.get("valid_extended_static_reachable_uncovered_bbs"),
        "valid_extended_static_reachable_coverage_rate": report_summary.get("valid_extended_static_reachable_coverage_rate"),
        "valid_static_unreachable_from_extended_roots": report_summary.get("valid_static_unreachable_from_extended_roots"),
        "extended_static_unreachable_valid_ratio": report_summary.get("extended_static_unreachable_valid_ratio"),
        "entry_reachable_uncovered_bbs": report_summary.get("entry_reachable_uncovered_bbs"),
        "vector_only_uncovered_bbs": report_summary.get("vector_only_uncovered_bbs"),
        "entry_plus_vector_uncovered_bbs": report_summary.get("entry_plus_vector_uncovered_bbs"),
        "extended_static_reachable_uncovered_bbs": report_summary.get("extended_static_reachable_uncovered_bbs"),
        "frontier_target_bbs": report_summary.get("frontier_target_bbs"),
        "uncovered_with_covered_predecessor_bbs": report_summary.get("uncovered_with_covered_predecessor_bbs"),
        "frontier_successor_tasks": report_summary.get("frontier_successor_tasks"),
        "frontier_successor_targets": report_summary.get("frontier_successor_targets"),
        "frontier_successor_forced_verified": report_summary.get("frontier_successor_forced_verified"),
        "frontier_successor_snapshot_attempts": report_summary.get("frontier_successor_snapshot_attempts"),
        "frontier_candidate_ready": report_summary.get("frontier_candidate_ready"),
        "frontier_no_root_snapshot": report_summary.get("frontier_no_root_snapshot"),
        "frontier_choice_unmapped": report_summary.get("frontier_choice_unmapped"),
        "frontier_successor_already_covered": report_summary.get("frontier_successor_already_covered"),
        "switch_frontier_gap_count": report_summary.get("switch_frontier_gap_count"),
        "call_frontier_gap_count": report_summary.get("call_frontier_gap_count"),
        "direct_call_tasks": report_summary.get("direct_call_tasks"),
        "direct_call_targets": report_summary.get("direct_call_targets"),
        "direct_call_low_yield_stops": report_summary.get("direct_call_low_yield_stops"),
        "summary_return_tasks": report_summary.get("summary_return_tasks"),
        "summary_return_targets": report_summary.get("summary_return_targets"),
        "summary_return_low_yield_stops": report_summary.get("summary_return_low_yield_stops"),
        "rtos_thread_tasks": report_summary.get("rtos_thread_tasks"),
        "rtos_thread_targets": report_summary.get("rtos_thread_targets"),
        "static_unreachable_valid_ratio": report_summary.get("static_unreachable_valid_ratio"),
        "frontier_replay_conversion_rate": report_summary.get("frontier_replay_conversion_rate"),
        "low_coverage_root_cause": report_summary.get("low_coverage_root_cause"),
        "top_frontier_predecessors": report_summary.get("top_frontier_predecessors"),
        "progress_points": len(progress_records),
        "progress_trend": trend,
        "last_progress": progress_records[-1] if progress_records else None,
    }


def write_campaign_outputs(
    output_root: Path,
    *,
    started_at: float,
    selected_cases: list,
    results: list[dict[str, object]],
    firmware_minutes: int,
    interval_seconds: int,
    completed: bool,
    current_case: str | None,
) -> None:
    summary_json = output_root / "campaign_summary.json"
    summary_csv = output_root / "campaign_summary.csv"
    summary_md = output_root / "campaign_summary.md"
    live_status_json = output_root / "campaign_live_status.json"

    payload = {
        "runner": "elfmultifuzz_interleaved_strict_campaign",
        "script": str(INTERLEAVED_SCRIPT),
        "module": INTERLEAVED_MODULE,
        "local_unicorn_shared_lib": str(LOCAL_UNICORN_SHARED_LIB),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(started_at)),
        "elapsed_seconds": time.time() - started_at,
        "completed": completed,
        "current_case": current_case,
        "firmware_minutes": firmware_minutes,
        "progress_interval_seconds": interval_seconds,
        "cases_total": len(selected_cases),
        "cases_completed": len([item for item in results if item.get("status") in COMPLETED_STATUSES]),
        "cases_failed": len([item for item in results if item.get("status") == "failed"]),
        # Watchdog early-termination bookkeeping: completed cases that were
        # stopped by the run-level stall watchdog before spending their
        # configured wallclock budget (see per-result terminated_early fields).
        "cases_terminated_early": len(
            [item for item in results if item.get("terminated_early")]
        ),
        "results": results,
    }
    write_json(summary_json, payload)
    write_json(live_status_json, {
        "completed": completed,
        "current_case": current_case,
        "cases_total": len(selected_cases),
        "cases_finished": len(results),
        "cases_completed": payload["cases_completed"],
        "cases_failed": payload["cases_failed"],
        "elapsed_seconds": payload["elapsed_seconds"],
    })

    fieldnames = [
        "status",
        "family",
        "name",
        "rel_path",
        "firmware",
        "valid_covered_bbs",
        "valid_total_bbs",
        "valid_coverage_rate",
        "valid_entry_plus_vector_static_reachable_covered_bbs",
        "valid_entry_plus_vector_static_reachable_bbs",
        "valid_entry_plus_vector_reachable_coverage_rate",
        "valid_entry_static_reachable_covered_bbs",
        "valid_entry_static_reachable_bbs",
        "valid_entry_static_reachable_coverage_rate",
        "valid_static_unreachable_from_entry_or_vectors",
        "extended_static_roots_available",
        "extended_static_root_bbs",
        "valid_extended_static_reachable_covered_bbs",
        "valid_extended_static_reachable_bbs",
        "valid_extended_static_reachable_uncovered_bbs",
        "valid_extended_static_reachable_coverage_rate",
        "valid_static_unreachable_from_extended_roots",
        "extended_static_unreachable_valid_ratio",
        "entry_reachable_uncovered_bbs",
        "vector_only_uncovered_bbs",
        "entry_plus_vector_uncovered_bbs",
        "extended_static_reachable_uncovered_bbs",
        "frontier_target_bbs",
        "uncovered_with_covered_predecessor_bbs",
        "frontier_successor_tasks",
        "frontier_successor_targets",
        "frontier_successor_forced_verified",
        "frontier_successor_snapshot_attempts",
        "frontier_candidate_ready",
        "frontier_no_root_snapshot",
        "frontier_choice_unmapped",
        "frontier_successor_already_covered",
        "switch_frontier_gap_count",
        "call_frontier_gap_count",
        "direct_call_tasks",
        "direct_call_targets",
        "direct_call_low_yield_stops",
        "summary_return_tasks",
        "summary_return_targets",
        "summary_return_low_yield_stops",
        "rtos_thread_tasks",
        "rtos_thread_targets",
        "static_unreachable_valid_ratio",
        "frontier_replay_conversion_rate",
        "low_coverage_root_cause",
        "covered_bbs",
        "total_bbs",
        "coverage_rate",
        "strict_real_entry_replayable",
        "coverage_entry_derived",
        "execution_time_seconds",
        "exit_code",
        "terminated_early",
        "terminated_early_reason",
        "wallclock_used_seconds",
        "configured_budget_seconds",
        "stall_watchdog_stop_reason",
        "stall_watchdog_budget_remaining_seconds",
        "report_file",
        "progress_jsonl_file",
        "checkpoint_file",
        "log_file",
        "resource_failure",
        "segmented_union",
        "case_segment_minutes",
        "case_segments_completed",
        "coverage_list_complete",
        "coverage_list_recoverable_valid_bbs",
        "progress_high_water_valid_covered_bbs",
        "report_file_caveat",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})

    family_groups: dict[str, list[dict[str, object]]] = {}
    for item in results:
        family_groups.setdefault(str(item.get("family") or "unknown"), []).append(item)

    lines: list[str] = []
    lines.append("# elfmultifuzz strict interleaved campaign")
    lines.append("")
    lines.append(f"- Started: {payload['started_at']}")
    lines.append(f"- Completed: {'yes' if completed else 'no'}")
    lines.append(f"- Current case: {current_case or 'none'}")
    lines.append(f"- Firmware duration: {firmware_minutes} min")
    lines.append(f"- Progress interval: {interval_seconds} s")
    lines.append(f"- Local Unicorn: `{LOCAL_UNICORN_SHARED_LIB}`")
    lines.append(f"- Cases total: {len(selected_cases)}")
    lines.append(f"- Cases finished: {len(results)}")
    lines.append(f"- Cases completed: {payload['cases_completed']}")
    lines.append(f"- Cases failed: {payload['cases_failed']}")
    lines.append(
        f"- Cases terminated early (stall watchdog): {payload['cases_terminated_early']}"
    )
    lines.append("")
    lines.append("## Family Summary")
    lines.append("")
    lines.append("| Family | Cases | Avg valid rate | Avg entry+vector rate | Avg extended-root rate | Max valid rate | Min valid rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for family, items in sorted(family_groups.items()):
        rates = [
            float(item.get("valid_coverage_rate") or 0.0)
            for item in items
            if item.get("status") in COMPLETED_STATUSES
        ]
        reachable_rates = [
            float(item.get("valid_entry_plus_vector_reachable_coverage_rate") or 0.0)
            for item in items
            if item.get("status") in COMPLETED_STATUSES
        ]
        extended_rates = [
            float(item.get("valid_extended_static_reachable_coverage_rate") or 0.0)
            for item in items
            if item.get("status") in COMPLETED_STATUSES
        ]
        if rates:
            avg_rate = sum(rates) / len(rates)
            max_rate = max(rates)
            min_rate = min(rates)
        else:
            avg_rate = max_rate = min_rate = 0.0
        avg_reachable_rate = sum(reachable_rates) / len(reachable_rates) if reachable_rates else 0.0
        avg_extended_rate = sum(extended_rates) / len(extended_rates) if extended_rates else 0.0
        lines.append(f"| {family} | {len(items)} | {avg_rate:.2f}% | {avg_reachable_rate:.2f}% | {avg_extended_rate:.2f}% | {max_rate:.2f}% | {min_rate:.2f}% |")
    lines.append("")
    lines.append("## Firmware Summary")
    lines.append("")
    lines.append("| Case | Status | Final valid BB | Final valid rate | Entry+vector BB | Entry+vector rate | Extended-root BB | Extended-root rate | Ext roots | Strict | Resource | List | Time (s) | Early stop |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |")
    for item in results:
        strict_text = "yes" if item.get("strict_real_entry_replayable") else "no"
        resource_text = "yes" if item.get("resource_failure") else "no"
        early_stop_text = (
            f"yes@{int(float(item.get('wallclock_used_seconds') or 0.0) / 60.0)}m"
            if item.get("terminated_early")
            else "no"
        )
        ext_roots_text = (
            "yes"
            if item.get("extended_static_roots_available") is True
            else str(item.get("extended_static_root_fallback") or "no")
        )
        list_text = "yes"
        if item.get("coverage_list_complete") is False:
            list_text = f"partial:{int(item.get('coverage_list_recoverable_valid_bbs') or 0)}"
        final_valid = f"{int(item.get('valid_covered_bbs') or 0)}/{int(item.get('valid_total_bbs') or 0)}"
        reachable_valid = (
            f"{int(item.get('valid_entry_plus_vector_static_reachable_covered_bbs') or 0)}/"
            f"{int(item.get('valid_entry_plus_vector_static_reachable_bbs') or 0)}"
        )
        extended_valid = (
            f"{int(item.get('valid_extended_static_reachable_covered_bbs') or 0)}/"
            f"{int(item.get('valid_extended_static_reachable_bbs') or 0)}"
        )
        lines.append(
            f"| {item.get('rel_path')} | {item.get('status')} | {final_valid} | {float(item.get('valid_coverage_rate') or 0.0):.2f}% | {reachable_valid} | {float(item.get('valid_entry_plus_vector_reachable_coverage_rate') or 0.0):.2f}% | {extended_valid} | {float(item.get('valid_extended_static_reachable_coverage_rate') or 0.0):.2f}% | {ext_roots_text} | {strict_text} | {resource_text} | {list_text} | {float(item.get('execution_time_seconds') or 0.0):.1f} | {early_stop_text} |"
        )
    lines.append("")
    lines.append("## Low Coverage Diagnostics")
    lines.append("")
    lines.append("| Case | Root cause | Ext roots | Entry+vector uncovered | Old static-unreachable | Extended static-unreachable | Frontier targets | Switch/Call gaps | Frontier replay | Direct/Summary/RTOS | Top frontier sample |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in sorted(results, key=lambda row: float(row.get("valid_coverage_rate") or 0.0)):
        ext_roots_text = (
            "yes"
            if item.get("extended_static_roots_available") is True
            else str(item.get("extended_static_root_fallback") or "no")
        )
        replay = (
            f"{int(item.get('frontier_successor_targets') or 0)}/"
            f"{int(item.get('frontier_successor_tasks') or 0)}"
            f"/{int(item.get('frontier_successor_snapshot_attempts') or 0)}"
        )
        switch_call = (
            f"{int(item.get('switch_frontier_gap_count') or 0)}/"
            f"{int(item.get('call_frontier_gap_count') or 0)}"
        )
        direct_summary_rtos = (
            f"{int(item.get('direct_call_targets') or 0)}/{int(item.get('direct_call_tasks') or 0)},"
            f"{int(item.get('summary_return_targets') or 0)}/{int(item.get('summary_return_tasks') or 0)},"
            f"{int(item.get('rtos_thread_targets') or 0)}/{int(item.get('rtos_thread_tasks') or 0)}"
        )
        lines.append(
            f"| {item.get('rel_path')} | {item.get('low_coverage_root_cause') or ''} | {ext_roots_text} | {int(item.get('entry_plus_vector_uncovered_bbs') or 0)} | {int(item.get('valid_static_unreachable_from_entry_or_vectors') or 0)} | {int(item.get('valid_static_unreachable_from_extended_roots') or 0)} | {int(item.get('frontier_target_bbs') or 0)} | {switch_call} | {replay} | {direct_summary_rtos} | {item.get('top_frontier_predecessors') or ''} |"
        )
    lines.append("")
    interval_minutes = max(1, int(round(max(1, interval_seconds) / 60.0)))
    lines.append(f"## {interval_minutes}-minute valid BB trend")
    lines.append("")
    bucket_headers = [
        format_bucket_label(index, interval_seconds)
        for index in range(len(next((item.get("progress_trend", []) for item in results if item.get("progress_trend")), [])))
    ]
    lines.append("| Case | " + " | ".join(bucket_headers) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(bucket_headers)) + " |")
    for item in results:
        cells = []
        for point in item.get("progress_trend", []) or []:
            value = point.get("valid_covered_bbs")
            cells.append("-" if value is None else str(int(value)))
        lines.append(f"| {item.get('rel_path')} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Case | Report | Progress | Log |")
    lines.append("| --- | --- | --- | --- |")
    for item in results:
        lines.append(
            f"| {item.get('rel_path')} | `{item.get('report_file')}` | `{item.get('progress_jsonl_file')}` | `{item.get('log_file')}` |"
        )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("LSGEMU_CONFIG_FILE"), help="Deployment YAML/JSON config.")
    parser.add_argument(
        "--valid-root",
        default=str(DEFAULT_VALID_BB_ROOT),
    )
    parser.add_argument(
        "--output-root",
        default=str(GLOBAL_RUNS_ROOT / f"elfmultifuzz_interleaved_strict_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--firmware-minutes", type=int, default=60)
    parser.add_argument("--progress-interval-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--family-filter", default=None)
    parser.add_argument("--name-filter", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-log-level", default="WARNING")
    parser.add_argument("--python-executable", default=sys.executable or "python3")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    valid_root = Path(args.valid_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases, _alias_map = discover_cases(valid_root)
    families = parse_csv_filter(args.family_filter)
    names = parse_csv_filter(args.name_filter)
    selected_cases = [case for case in cases if case_matches(case, families, names)]
    if args.limit > 0:
        selected_cases = selected_cases[:args.limit]

    manifest = {
        "runner": "elfmultifuzz_interleaved_strict_campaign",
        "script": str(INTERLEAVED_SCRIPT),
        "module": INTERLEAVED_MODULE,
        "local_unicorn_shared_lib": str(LOCAL_UNICORN_SHARED_LIB),
        "cases_total": len(selected_cases),
        "firmware_minutes": args.firmware_minutes,
        "progress_interval_seconds": args.progress_interval_seconds,
        "cases": [
            {
                "family": case.family,
                "name": case.name,
                "rel_path": case.rel_path,
                "firmware": str(case.elf_path.resolve()),
                "valid_basic_blocks": str(case.valid_bb_path.resolve()),
            }
            for case in selected_cases
        ],
    }
    write_json(output_root / "campaign_manifest.json", manifest)

    started_at = time.time()
    results: list[dict[str, object]] = []
    current_case: str | None = None
    campaign_progress_path = output_root / "campaign_progress.jsonl"
    append_jsonl(campaign_progress_path, {
        "event": "campaign_start",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "cases_total": len(selected_cases),
    })
    write_campaign_outputs(
        output_root,
        started_at=started_at,
        selected_cases=selected_cases,
        results=results,
        firmware_minutes=args.firmware_minutes,
        interval_seconds=args.progress_interval_seconds,
        completed=False,
        current_case=None,
    )

    for index, case in enumerate(selected_cases, start=1):
        current_case = case.rel_path
        case_output_dir = (output_root / case.rel_path).resolve()
        case_output_dir.mkdir(parents=True, exist_ok=True)
        firmware = case.elf_path.resolve()
        report_path = case_output_dir / f"{firmware.stem}_interleaved_report.json"
        progress_path = case_output_dir / f"{firmware.stem}_coverage_progress.jsonl"
        checkpoint_path = case_output_dir / f"{firmware.stem}_coverage_checkpoints.jsonl"
        log_path = case_output_dir / "run.log"

        if args.resume and report_path.exists():
            result = summarize_case_result(
                case=case,
                case_output_dir=case_output_dir,
                report_path=report_path,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                exit_code=0,
                status="completed_cached",
                duration_minutes=args.firmware_minutes,
                interval_seconds=args.progress_interval_seconds,
            )
            results.append(result)
            append_jsonl(campaign_progress_path, {
                "event": "case_resume_skip",
                "case_index": index,
                "case": case.rel_path,
                "report_file": str(report_path),
            })
            write_campaign_outputs(
                output_root,
                started_at=started_at,
                selected_cases=selected_cases,
                results=results,
                firmware_minutes=args.firmware_minutes,
                interval_seconds=args.progress_interval_seconds,
                completed=False,
                current_case=current_case,
            )
            continue

        cmd = [
            args.python_executable,
            "-m",
            INTERLEAVED_MODULE,
            "--firmware", str(firmware),
            "--output-dir", str(case_output_dir),
            "--total-time-minutes", str(args.firmware_minutes),
            "--log-level", args.runner_log_level,
        ]
        if args.config:
            cmd.extend(["--config", str(args.config)])
        cmd.extend(args.extra_arg)

        env = os.environ.copy()
        srcv4_path = str(SOURCE_ROOT)
        env["PYTHONPATH"] = (
            f"{srcv4_path}:{env.get('PYTHONPATH', '')}"
            if env.get("PYTHONPATH")
            else srcv4_path
        )
        env["LIBUNICORN_PATH"] = str(LOCAL_UNICORN_BUILD)
        env["LD_LIBRARY_PATH"] = (
            f"{LOCAL_UNICORN_BUILD}:{env.get('LD_LIBRARY_PATH', '')}"
            if env.get("LD_LIBRARY_PATH")
            else str(LOCAL_UNICORN_BUILD)
        )
        env["PYTHONUNBUFFERED"] = "1"
        env["LSGEMU_PROGRESS_JSONL"] = str(progress_path)
        env["LSGEMU_PROGRESS_INTERVAL_SECONDS"] = str(max(1, args.progress_interval_seconds))
        env["LSGEMU_COVERAGE_CHECKPOINT"] = str(checkpoint_path)
        env["LSGEMU_COVERAGE_CHECKPOINT_INTERVAL"] = str(max(1, args.progress_interval_seconds))
        # Branch-snapshot memory guardrail: keep the child's retained
        # current-snapshot cache bounded (unset/0 means unbounded growth over
        # a multi-hour case).  An operator-provided limit wins.
        env.setdefault("LSGEMU_BRANCH_SNAPSHOT_CURRENT_LIMIT", "4096")

        append_jsonl(campaign_progress_path, {
            "event": "case_start",
            "case_index": index,
            "case": case.rel_path,
            "firmware": str(firmware),
            "command": cmd,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        })
        write_campaign_outputs(
            output_root,
            started_at=started_at,
            selected_cases=selected_cases,
            results=results,
            firmware_minutes=args.firmware_minutes,
            interval_seconds=args.progress_interval_seconds,
            completed=False,
            current_case=current_case,
        )

        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(RUNNER_PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            # Watchdog: the child is expected to honor --total-time-minutes
            # itself, but a child stuck in native code (Unicorn deadlock, GIL
            # held forever) would otherwise hang the whole serial campaign.
            # Kill the child's process group after the budget plus a grace
            # period, and always clean it up on driver exceptions/Ctrl-C.
            watchdog_deadline = time.monotonic() + (
                args.firmware_minutes * 60 + WATCHDOG_GRACE_SECONDS
            )
            last_progress_signature: tuple[object, ...] | None = None
            try:
                while True:
                    rc = process.poll()
                    latest_progress = load_last_jsonl_record(progress_path)
                    if latest_progress is not None:
                        signature = (
                            latest_progress.get("event"),
                            latest_progress.get("elapsed_seconds"),
                            latest_progress.get("valid_covered_bbs"),
                        )
                        if signature != last_progress_signature:
                            last_progress_signature = signature
                            append_jsonl(campaign_progress_path, {
                                "event": "case_progress",
                                "case_index": index,
                                "case": case.rel_path,
                                "progress": latest_progress,
                            })
                    if rc is not None:
                        exit_code = int(rc)
                        break
                    if time.monotonic() > watchdog_deadline:
                        append_jsonl(campaign_progress_path, {
                            "event": "case_watchdog_kill",
                            "case_index": index,
                            "case": case.rel_path,
                            "budget_minutes": args.firmware_minutes,
                        })
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            process.kill()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                        exit_code = -9
                        break
                    time.sleep(max(1, args.poll_seconds))
            except BaseException:
                # Driver crash or Ctrl-C: do not orphan the child process.
                if process.poll() is None:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=10)
                    except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                raise

        status = "completed" if exit_code == 0 and report_path.exists() else "failed"
        result = summarize_case_result(
            case=case,
            case_output_dir=case_output_dir,
            report_path=report_path,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            exit_code=exit_code,
            status=status,
            duration_minutes=args.firmware_minutes,
            interval_seconds=args.progress_interval_seconds,
        )
        results.append(result)
        append_jsonl(campaign_progress_path, {
            "event": "case_complete",
            "case_index": index,
            "case": case.rel_path,
            "exit_code": exit_code,
            "status": status,
            "valid_covered_bbs": result.get("valid_covered_bbs"),
            "valid_coverage_rate": result.get("valid_coverage_rate"),
            "strict_real_entry_replayable": result.get("strict_real_entry_replayable"),
        })
        write_campaign_outputs(
            output_root,
            started_at=started_at,
            selected_cases=selected_cases,
            results=results,
            firmware_minutes=args.firmware_minutes,
            interval_seconds=args.progress_interval_seconds,
            completed=False,
            current_case=current_case,
        )

    append_jsonl(campaign_progress_path, {
        "event": "campaign_complete",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "cases_finished": len(results),
    })
    write_campaign_outputs(
        output_root,
        started_at=started_at,
        selected_cases=selected_cases,
        results=results,
        firmware_minutes=args.firmware_minutes,
        interval_seconds=args.progress_interval_seconds,
        completed=True,
        current_case=None,
    )


if __name__ == "__main__":
    main()
