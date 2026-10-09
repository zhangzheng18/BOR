#!/usr/bin/env python3
"""Parallel strict interleaved campaign for elfmultifuzz ELFs."""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.deployment_config import apply_config_from_argv, configured_path

apply_config_from_argv()

from lsgemu.historical_runner import DEFAULT_VALID_BB_ROOT, PROJECT_ROOT as RUNNER_PROJECT_ROOT
from lsgemu.run_elfmultifuzz_interleaved_strict_campaign import (
    COMPLETED_STATUSES,
    INTERLEAVED_MODULE,
    LOCAL_UNICORN_BUILD,
    LOCAL_UNICORN_SHARED_LIB,
    append_jsonl,
    case_matches,
    load_jsonl_records,
    load_last_jsonl_record,
    load_json,
    parse_csv_filter,
    summarize_case_result,
    write_campaign_outputs,
    write_json,
)
from lsgemu.stat_elfmultifuzz_valid_coverage import discover_cases


GLOBAL_RUNS_ROOT = configured_path(
    "LSGEMU_RUN_OUTPUT_DIR",
    configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4") / ".lsgemu_runs",
)


STRICT_4H_TUNING_ARGS: list[str] = [
    "--skip-cold-isr",
    "--post-interleaved-reserve-ratio", "0.80",
    "--interleaved-adaptive-plateau-rounds", "2",
    "--interleaved-adaptive-plateau-min-rounds", "4",
    "--direct-call-continuation-seconds", "60",
    "--direct-call-continuation-max-tasks", "256",
    "--direct-call-continuation-max-targets", "256",
    "--direct-call-continuation-variants-per-call", "4",
    "--direct-call-summary-return-seconds", "60",
    "--direct-call-summary-return-max-tasks", "256",
    "--direct-call-summary-return-max-targets", "256",
    "--direct-call-summary-return-variants-per-call", "4",
    "--direct-call-continuation-no-new-bbs", "32768",
    "--direct-call-summary-return-no-new-bbs", "32768",
    "--direct-call-continuation-low-yield-stop-tasks", "24",
    "--direct-call-continuation-low-yield-min-tasks", "8",
    "--direct-call-summary-return-low-yield-stop-tasks", "24",
    "--direct-call-summary-return-low-yield-min-tasks", "8",
    "--rtos-thread-entry-seconds", "120",
    "--rtos-thread-entry-max-tasks", "256",
    "--rtos-thread-entry-max-targets", "256",
    "--rtos-thread-entry-variants-per-call", "4",
    "--frontier-successor-replay-seconds", "360",
    "--frontier-successor-replay-tail-seconds", "600",
    "--frontier-successor-replay-max-tasks", "2048",
    "--frontier-successor-replay-max-targets", "1024",
    "--frontier-successor-replay-variants-per-branch", "4",
    "--frontier-successor-flush-seconds", "30",
    "--frontier-successor-flush-min-snapshots", "64",
    "--frontier-successor-flush-min-dynamic-edges", "2",
    "--frontier-successor-flush-max-new-bbs", "16",
    "--frontier-cycle-tail-reserve-seconds", "60",
    "--frontier-round-seconds", "90",
    "--frontier-max-rounds", "6",
    "--frontier-max-targets", "256",
    "--frontier-candidate-pool-multiplier", "6",
    "--switch-frontier-round-seconds", "60",
    "--switch-frontier-max-rounds", "4",
    "--switch-frontier-max-targets", "192",
    "--frontier-cycle-max-cycles", "10",
    "--hotspot-frontier-predecessors", "8",
    "--targeted-max-tasks-per-root", "16",
    "--targeted-prefix-snapshot-seed-limit", "128",
    "--contextual-isr-reservoir-seconds", "90",
    "--contextual-isr-frontier-seconds", "60",
    "--vector-cleanup-seconds", "120",
    "--deadline-drain-round-seconds", "300",
    "--deadline-drain-stale-rounds", "5",
    "--deadline-drain-targeted-share", "0.45",
    "--deadline-drain-targeted-first-threshold", "64",
    "--deadline-drain-switch-share", "0.40",
    "--continue-targeted-state",
]


def build_env(
    case_output_dir: Path,
    progress_path: Path,
    checkpoint_path: Path,
    llm_journal_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    srcv4_path = str(configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4"))
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
    env.setdefault("PYTHONHASHSEED", "0")
    env["LSGEMU_PROGRESS_JSONL"] = str(progress_path)
    env["LSGEMU_COVERAGE_CHECKPOINT"] = str(checkpoint_path)
    # 推断历史增量落盘路径：runner 每完成一条推断立即追加写出，
    # 这样 firmware-minutes 超时 SIGTERM 后历史也不会随进程丢失。
    if llm_journal_path is not None:
        env["LSGEMU_LLM_INFERENCE_JOURNAL"] = str(llm_journal_path)
    # Strict replay should not cut finite data loops (memmove/malloc/list scans)
    # after 100 iterations. Only genuine MMIO polling/deadlock loops should be
    # intervened early; finite RAM loops must be allowed to return naturally so
    # call continuations remain entry-derived.
    env.setdefault("LSGEMU_UNKNOWN_LOOP_THRESHOLD", "1000")
    env.setdefault("LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "100000")
    env.setdefault("LSGEMU_DELAY_LOOP_THRESHOLD", "100000")
    env.setdefault("LSGEMU_ENABLE_MEMORY_WAIT_CLASSIFIER", "1")
    env.setdefault("LSGEMU_MEMORY_WAIT_LOOP_THRESHOLD", "160")
    env.setdefault("LSGEMU_DYNAMIC_MEMORY_CONSTRAINTS", "1")
    env.setdefault("LSGEMU_LOOP_EXIT_HINTS", "1")
    env.setdefault("LSGEMU_RECORD_MAPPED_WRITES", "1")
    env.setdefault("LSGEMU_RISKY_SNAPSHOT_MAX_CANDIDATES", "3")
    env.setdefault("LSGEMU_RISKY_SNAPSHOT_MAX_SUFFIX_DEPTH", "10")
    env.setdefault("LSGEMU_RISKY_SKIP_CALLLIKE_LONG_SUFFIX", "0")
    env.setdefault("LSGEMU_FAST_LOOP_ANALYSIS", "1")
    env.setdefault("LSGEMU_MAX_LLM_CALLS_PER_LOOP", "1")
    env.setdefault("LSGEMU_MAX_DEADLOCK_LLM_CALLS_PER_LOOP", "1")
    return env


CRASH_RETRY_PLANS: list[tuple[str, list[str]]] = [
    (
        "skip_cold_and_contextual_isr",
        [
            "--skip-cold-isr",
            "--contextual-isr-contexts", "0",
            "--contextual-isr-reservoir-seconds", "0",
            "--contextual-isr-frontier-seconds", "0",
            "--vector-cleanup-seconds", "0",
        ],
    ),
    (
        "skip_isr_keep_entry_replay",
        [
            "--skip-cold-isr",
            "--contextual-isr-contexts", "0",
            "--contextual-isr-time-seconds", "0",
            "--contextual-isr-reservoir-seconds", "0",
            "--contextual-isr-frontier-seconds", "0",
            "--vector-cleanup-seconds", "0",
            "--rtos-thread-entry-seconds", "0",
        ],
    ),
    (
        "strict_entry_replay_low_pressure",
        [
            "--skip-cold-isr",
            "--contextual-isr-contexts", "0",
            "--contextual-isr-time-seconds", "0",
            "--contextual-isr-reservoir-seconds", "0",
            "--contextual-isr-frontier-seconds", "0",
            "--vector-cleanup-seconds", "0",
            "--rtos-thread-entry-seconds", "0",
            "--direct-call-continuation-seconds", "30",
            "--direct-call-continuation-max-tasks", "128",
            "--direct-call-continuation-max-targets", "128",
            "--direct-call-summary-return-seconds", "30",
            "--direct-call-summary-return-max-tasks", "128",
            "--direct-call-summary-return-max-targets", "128",
            "--frontier-successor-replay-seconds", "240",
            "--frontier-successor-replay-tail-seconds", "480",
            "--frontier-successor-replay-max-tasks", "1024",
            "--frontier-successor-replay-max-targets", "512",
            "--frontier-round-seconds", "60",
            "--frontier-max-rounds", "4",
            "--switch-frontier-round-seconds", "45",
            "--switch-frontier-max-rounds", "3",
            "--frontier-cycle-max-cycles", "8",
            "--deadline-drain-round-seconds", "180",
            "--deadline-drain-stale-rounds", "4",
        ],
    ),
    (
        "strict_successor_drain_only",
        [
            "--skip-cold-isr",
            "--contextual-isr-contexts", "0",
            "--contextual-isr-time-seconds", "0",
            "--contextual-isr-reservoir-seconds", "0",
            "--contextual-isr-frontier-seconds", "0",
            "--vector-cleanup-seconds", "0",
            "--rtos-thread-entry-seconds", "0",
            "--direct-call-continuation-seconds", "15",
            "--direct-call-continuation-max-tasks", "64",
            "--direct-call-continuation-max-targets", "64",
            "--direct-call-summary-return-seconds", "15",
            "--direct-call-summary-return-max-tasks", "64",
            "--direct-call-summary-return-max-targets", "64",
            "--frontier-successor-replay-seconds", "180",
            "--frontier-successor-replay-tail-seconds", "600",
            "--interleaved-frontier-round-seconds", "0",
            "--switch-frontier-round-seconds", "0",
            "--frontier-round-seconds", "45",
            "--frontier-max-rounds", "3",
            "--frontier-cycle-max-cycles", "6",
            "--deadline-drain-round-seconds", "180",
            "--deadline-drain-stale-rounds", "4",
        ],
    ),
    (
        "minimal_entry_frontier_replay",
        [
            "--skip-cold-isr",
            "--contextual-isr-contexts", "0",
            "--contextual-isr-time-seconds", "0",
            "--contextual-isr-reservoir-seconds", "0",
            "--contextual-isr-frontier-seconds", "0",
            "--vector-cleanup-seconds", "0",
            "--rtos-thread-entry-seconds", "0",
            "--direct-call-continuation-seconds", "0",
            "--direct-call-summary-return-seconds", "0",
            "--frontier-successor-replay-seconds", "120",
            "--frontier-successor-replay-tail-seconds", "240",
            "--frontier-successor-replay-max-tasks", "512",
            "--frontier-successor-replay-max-targets", "256",
            "--frontier-round-seconds", "30",
            "--frontier-max-rounds", "2",
            "--switch-frontier-round-seconds", "0",
            "--frontier-cycle-max-cycles", "4",
            "--deadline-drain-round-seconds", "120",
            "--deadline-drain-stale-rounds", "3",
        ],
    ),
]


LOW_COVERAGE_RETRY_PLANS: list[tuple[str, list[str]]] = [
    (
        "low_coverage_legacy_time_compensation",
        [
            "--skip-cold-isr",
            "--replay-time-skip-mode", "legacy",
            "--post-interleaved-reserve-ratio", "0.88",
            "--direct-call-continuation-seconds", "90",
            "--direct-call-continuation-max-tasks", "384",
            "--direct-call-continuation-max-targets", "384",
            "--direct-call-continuation-variants-per-call", "6",
            "--direct-call-summary-return-seconds", "120",
            "--direct-call-summary-return-max-tasks", "512",
            "--direct-call-summary-return-max-targets", "512",
            "--direct-call-summary-return-variants-per-call", "8",
            "--direct-call-summary-return-no-new-bbs", "65536",
            "--direct-call-summary-return-low-yield-stop-tasks", "64",
            "--direct-call-summary-return-low-yield-min-tasks", "16",
            "--frontier-successor-replay-seconds", "600",
            "--frontier-successor-replay-tail-seconds", "900",
            "--frontier-successor-replay-max-tasks", "4096",
            "--frontier-successor-replay-max-targets", "2048",
            "--frontier-successor-replay-variants-per-branch", "8",
            "--frontier-successor-replay-no-new-bbs", "65536",
            "--frontier-round-seconds", "90",
            "--frontier-max-rounds", "6",
            "--frontier-max-targets", "512",
            "--frontier-candidate-pool-multiplier", "8",
            "--switch-frontier-round-seconds", "60",
            "--switch-frontier-max-rounds", "6",
            "--switch-frontier-max-targets", "512",
            "--switch-frontier-candidate-pool-multiplier", "4",
            "--frontier-cycle-max-cycles", "10",
            "--frontier-cycle-stale-cycles", "2",
            "--deadline-drain-round-seconds", "420",
            "--deadline-drain-stale-rounds", "6",
            "--continue-targeted-state",
        ],
    ),
    (
        "low_coverage_deep_entry_replay",
        [
            "--skip-cold-isr",
            "--post-interleaved-reserve-ratio", "0.90",
            "--interleaved-adaptive-plateau-rounds", "3",
            "--interleaved-adaptive-plateau-min-rounds", "6",
            "--direct-call-continuation-seconds", "120",
            "--direct-call-continuation-max-tasks", "512",
            "--direct-call-continuation-max-targets", "512",
            "--direct-call-continuation-variants-per-call", "8",
            "--direct-call-continuation-no-new-bbs", "65536",
            "--direct-call-continuation-low-yield-stop-tasks", "64",
            "--direct-call-continuation-low-yield-min-tasks", "16",
            "--direct-call-summary-return-seconds", "90",
            "--direct-call-summary-return-max-tasks", "512",
            "--direct-call-summary-return-max-targets", "512",
            "--direct-call-summary-return-variants-per-call", "8",
            "--direct-call-summary-return-no-new-bbs", "65536",
            "--direct-call-summary-return-low-yield-stop-tasks", "64",
            "--direct-call-summary-return-low-yield-min-tasks", "16",
            "--rtos-thread-entry-seconds", "180",
            "--rtos-thread-entry-max-tasks", "512",
            "--rtos-thread-entry-max-targets", "512",
            "--rtos-thread-entry-variants-per-call", "6",
            "--frontier-successor-replay-seconds", "720",
            "--frontier-successor-replay-tail-seconds", "1200",
            "--frontier-successor-replay-max-tasks", "4096",
            "--frontier-successor-replay-max-targets", "2048",
            "--frontier-successor-replay-variants-per-branch", "8",
            "--frontier-successor-replay-no-new-bbs", "65536",
            "--frontier-successor-flush-seconds", "45",
            "--frontier-successor-flush-min-snapshots", "32",
            "--frontier-successor-flush-min-dynamic-edges", "1",
            "--frontier-successor-flush-max-new-bbs", "32",
            "--frontier-round-seconds", "120",
            "--frontier-max-rounds", "8",
            "--frontier-max-targets", "512",
            "--frontier-candidate-pool-multiplier", "8",
            "--switch-frontier-round-seconds", "90",
            "--switch-frontier-max-rounds", "8",
            "--switch-frontier-max-targets", "512",
            "--switch-frontier-candidate-pool-multiplier", "4",
            "--frontier-cycle-max-cycles", "12",
            "--frontier-cycle-stale-cycles", "2",
            "--hotspot-frontier-predecessors", "16",
            "--targeted-max-tasks-per-root", "32",
            "--targeted-prefix-snapshot-seed-limit", "256",
            "--deadline-drain-round-seconds", "420",
            "--deadline-drain-stale-rounds", "6",
            "--continue-targeted-state",
        ],
    ),
    (
        "low_coverage_successor_drain_focus",
        [
            "--skip-cold-isr",
            "--post-interleaved-reserve-ratio", "0.92",
            "--direct-call-continuation-seconds", "90",
            "--direct-call-continuation-max-tasks", "384",
            "--direct-call-continuation-max-targets", "384",
            "--direct-call-continuation-variants-per-call", "8",
            "--direct-call-summary-return-seconds", "60",
            "--direct-call-summary-return-max-tasks", "384",
            "--direct-call-summary-return-max-targets", "384",
            "--direct-call-summary-return-variants-per-call", "8",
            "--rtos-thread-entry-seconds", "120",
            "--rtos-thread-entry-max-tasks", "384",
            "--rtos-thread-entry-max-targets", "384",
            "--rtos-thread-entry-variants-per-call", "6",
            "--frontier-successor-replay-seconds", "900",
            "--frontier-successor-replay-tail-seconds", "1800",
            "--frontier-successor-replay-max-tasks", "6144",
            "--frontier-successor-replay-max-targets", "2048",
            "--frontier-successor-replay-variants-per-branch", "10",
            "--frontier-successor-replay-no-new-bbs", "65536",
            "--frontier-round-seconds", "90",
            "--frontier-max-rounds", "6",
            "--frontier-max-targets", "512",
            "--frontier-candidate-pool-multiplier", "8",
            "--switch-frontier-round-seconds", "60",
            "--switch-frontier-max-rounds", "6",
            "--switch-frontier-max-targets", "512",
            "--switch-frontier-candidate-pool-multiplier", "4",
            "--frontier-cycle-max-cycles", "10",
            "--frontier-cycle-stale-cycles", "2",
            "--deadline-drain-round-seconds", "480",
            "--deadline-drain-stale-rounds", "8",
            "--continue-targeted-state",
        ],
    ),
]


def attempt_suffix(retry_count: int, segment_index: int = 0) -> str:
    parts: list[str] = []
    if retry_count > 0:
        parts.append(f"attempt{retry_count}")
    if segment_index > 0:
        parts.append(f"segment{segment_index + 1}")
    return "" if not parts else "_" + "_".join(parts)


def retry_mode_name(retry_count: int) -> str:
    if retry_count <= 0:
        return "normal"
    crash_index = retry_count - 1
    if 0 <= crash_index < len(CRASH_RETRY_PLANS):
        return CRASH_RETRY_PLANS[crash_index][0]
    low_index = retry_count - len(CRASH_RETRY_PLANS) - 1
    if 0 <= low_index < len(LOW_COVERAGE_RETRY_PLANS):
        return LOW_COVERAGE_RETRY_PLANS[low_index][0]
    return f"retry_{retry_count}"


def apply_child_limits(memory_limit_gb: int) -> None:
    if memory_limit_gb <= 0:
        return
    memory_bytes = int(memory_limit_gb) * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def segment_count_for_case(total_minutes: int, segment_minutes: int) -> int:
    if segment_minutes <= 0 or segment_minutes >= total_minutes:
        return 1
    return max(1, (int(total_minutes) + int(segment_minutes) - 1) // int(segment_minutes))


def segment_duration_minutes(total_minutes: int, segment_minutes: int, segment_index: int) -> int:
    if segment_minutes <= 0 or segment_minutes >= total_minutes:
        return int(total_minutes)
    consumed = int(segment_index) * int(segment_minutes)
    remaining = max(1, int(total_minutes) - consumed)
    return max(1, min(int(segment_minutes), remaining))


def log_indicates_resource_failure(log_path: Path) -> bool:
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    markers = (
        "memoryerror",
        "cannot allocate memory",
        "could not allocate dynamic translator buffer",
        "dynamic translator buffer",
        "std::bad_alloc",
        "resource temporarily unavailable",
        "killed",
    )
    return any(marker in tail for marker in markers)


def progress_score(record: dict[str, object] | None) -> tuple[int, float, float]:
    if not isinstance(record, dict):
        return (0, 0.0, 0.0)
    try:
        covered = int(record.get("valid_covered_bbs") or 0)
    except (TypeError, ValueError):
        covered = 0
    try:
        rate = float(record.get("valid_coverage_rate") or 0.0)
    except (TypeError, ValueError):
        rate = 0.0
    try:
        elapsed = float(record.get("elapsed_seconds") or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    return (covered, rate, elapsed)


def low_coverage_retry_minutes_for(args, retry_count: int) -> int:
    try:
        retry_minutes = int(args.low_coverage_retry_minutes)
    except (TypeError, ValueError):
        retry_minutes = 0
    if retry_minutes > 0 and retry_count > len(CRASH_RETRY_PLANS):
        return retry_minutes
    return int(args.firmware_minutes)


def auto_retry_full_duration_low_coverage(args) -> bool:
    """
    Short coverage sweeps are frequently schedule probes, not final runs.

    A case can consume the full requested wallclock and still be low-covered
    because the first schedule spent budget in the wrong entry-derived replay
    family.  Enable low-coverage compensation automatically for short/medium
    runs while keeping long campaigns explicitly controlled.
    """
    raw = os.environ.get("LSGEMU_AUTO_FULL_DURATION_LOW_COVERAGE_RETRY", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        primary_minutes = int(args.firmware_minutes)
    except (TypeError, ValueError):
        primary_minutes = 0
    try:
        retry_minutes = int(args.low_coverage_retry_minutes)
    except (TypeError, ValueError):
        retry_minutes = 0
    effective_minutes = retry_minutes if retry_minutes > 0 else primary_minutes
    return effective_minutes > 0 and effective_minutes <= 30


def effective_low_coverage_retry_plan_limit(args) -> int:
    configured = int(args.low_coverage_retry_max_attempts)
    total_plans = len(LOW_COVERAGE_RETRY_PLANS)
    if configured == 0:
        return total_plans
    if configured > 0:
        return min(total_plans, configured)

    raw = os.environ.get("LSGEMU_AUTO_LOW_COVERAGE_RETRY_ATTEMPTS", "").strip()
    if raw:
        try:
            return min(total_plans, max(0, int(raw)))
        except ValueError:
            pass
    try:
        primary_minutes = int(args.firmware_minutes)
    except (TypeError, ValueError):
        primary_minutes = 0
    try:
        retry_minutes = int(args.low_coverage_retry_minutes)
    except (TypeError, ValueError):
        retry_minutes = 0
    effective_minutes = retry_minutes if retry_minutes > 0 else primary_minutes
    if effective_minutes > 0 and effective_minutes <= 30:
        return min(total_plans, 2)
    return min(total_plans, 1)


def result_stall_watchdog_stopped(result: dict[str, object]) -> bool:
    """True when the attempt ended early through the run-level stall watchdog.

    ``summarize_case_result`` copies ``stall_watchdog_stop_reason`` from the
    runner report; union paths rebuild the dict from the report payload, so the
    embedded watchdog status is checked as a fallback.
    """
    if result.get("stall_watchdog_stop_reason"):
        return True
    watchdog_status = result.get("stall_watchdog")
    return isinstance(watchdog_status, dict) and bool(
        watchdog_status.get("stop_requested")
    )


def should_low_coverage_retry(
    *,
    args,
    result: dict[str, object],
    retry_count: int,
    resource_failure: bool,
    segment_index: int,
    total_segments: int,
) -> tuple[bool, dict[str, object]]:
    """Decide the post-case low-coverage retry (short run + low coverage).

    A stall-watchdog early stop must never feed this loop: the stop already
    proves the run spent the whole stall window without new coverage, so both
    the short-elapsed gate and the low-reachable-rate gate are trivially true
    (``reachable_rate`` is 0.0 whenever the valid-BB denominator is not wired
    up, e.g. the MCUdatabase chain), and re-running would burn exactly the
    wallclock the watchdog just saved.  The short-circuit also covers the
    full-duration bypasses (``--retry-full-duration-low-coverage`` and
    ``auto_retry_full_duration_low_coverage``), which only widen the elapsed
    gate inside the same conjunction.

    Returns ``(retry, diagnostics)``; diagnostics carries
    ``watchdog_stopped`` and ``would_retry_without_watchdog`` so the driver can
    log suppressed retries.
    """
    if result.get("status") not in COMPLETED_STATUSES:
        return False, {"watchdog_stopped": result_stall_watchdog_stopped(result)}
    watchdog_stopped = result_stall_watchdog_stopped(result)
    reachable_rate = float(
        result.get("valid_entry_plus_vector_reachable_coverage_rate")
        or result.get("valid_coverage_rate")
        or 0.0
    )
    elapsed_seconds = float(result.get("execution_time_seconds") or 0.0)
    requested_minutes = low_coverage_retry_minutes_for(args, retry_count)
    requested_seconds = max(0, int(requested_minutes) * 60)
    elapsed_ratio = (
        elapsed_seconds / requested_seconds
        if requested_seconds > 0
        else 1.0
    )
    low_retry_count = max(0, retry_count - len(CRASH_RETRY_PLANS))
    low_retry_plan_limit = effective_low_coverage_retry_plan_limit(args)
    elapsed_gate = (
        elapsed_ratio < max(0.0, float(args.low_coverage_retry_min_elapsed_ratio))
        or bool(args.retry_full_duration_low_coverage)
        or auto_retry_full_duration_low_coverage(args)
    )
    common_gates = (
        result.get("status") in COMPLETED_STATUSES
        and not args.disable_low_coverage_retry
        and not resource_failure
        and segment_index + 1 >= total_segments
        and float(args.low_coverage_retry_rate) > 0.0
        and reachable_rate < float(args.low_coverage_retry_rate)
        and elapsed_gate
        and low_retry_count < low_retry_plan_limit
    )
    return (
        not watchdog_stopped and common_gates,
        {
            "watchdog_stopped": watchdog_stopped,
            "would_retry_without_watchdog": common_gates,
            "reachable_valid_coverage_rate": reachable_rate,
            "elapsed_seconds": elapsed_seconds,
            "requested_seconds": requested_seconds,
            "elapsed_ratio": elapsed_ratio,
        },
    )


def remember_best_attempt(
    best_attempts: dict[int, dict[str, object]],
    index: int,
    result: dict[str, object],
) -> None:
    """Keep the best strict high-water result across compensation attempts."""
    if progress_score(result) >= progress_score(best_attempts.get(index)):
        best_attempts[index] = result


def parse_int_set(values: object) -> set[int]:
    if not isinstance(values, list):
        return set()
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(value, 0) & ~1 if isinstance(value, str) else int(value) & ~1)
        except (TypeError, ValueError):
            continue
    return parsed


def load_checkpoint_union(case_output_dir: Path) -> tuple[set[int], dict[str, object] | None, list[str]]:
    covered: set[int] = set()
    best_checkpoint: dict[str, object] | None = None
    checkpoint_files: list[str] = []
    for checkpoint_path in sorted(case_output_dir.glob("*_coverage_checkpoints*.jsonl")):
        checkpoint_files.append(str(checkpoint_path))
        for record in load_jsonl_records(checkpoint_path):
            record_covered = parse_int_set(record.get("global_covered_bb_list"))
            if not record_covered:
                continue
            covered.update(record_covered)
            if best_checkpoint is None:
                best_checkpoint = record
                continue
            current_count = int(record.get("global_covered_bbs") or len(record_covered) or 0)
            best_count = int(best_checkpoint.get("global_covered_bbs") or 0)
            current_elapsed = float(record.get("elapsed_seconds") or 0.0)
            best_elapsed = float(best_checkpoint.get("elapsed_seconds") or 0.0)
            if (current_count, current_elapsed) > (best_count, best_elapsed):
                best_checkpoint = record
    return covered, best_checkpoint, checkpoint_files


def load_progress_union(case_output_dir: Path) -> tuple[set[int], dict[str, object] | None, list[str]]:
    covered: set[int] = set()
    best_progress_with_list: dict[str, object] | None = None
    progress_files: list[str] = []
    for progress_path in sorted(case_output_dir.glob("*_coverage_progress*.jsonl")):
        progress_files.append(str(progress_path))
        for record in load_jsonl_records(progress_path):
            record_covered = parse_int_set(record.get("covered_bb_list"))
            record_covered.update(parse_int_set(record.get("covered_valid_bb_list")))
            if not record_covered:
                continue
            covered.update(record_covered)
            if best_progress_with_list is None or progress_score(record) > progress_score(best_progress_with_list):
                best_progress_with_list = record
    return covered, best_progress_with_list, progress_files


def load_progress_high_water(case_output_dir: Path) -> dict[str, object] | None:
    best_record: dict[str, object] | None = None
    for progress_path in sorted(case_output_dir.glob("*_coverage_progress*.jsonl")):
        record = best_progress_record(progress_path)
        if progress_score(record) > progress_score(best_record):
            best_record = record
    return best_record


def build_case_union_result(
    *,
    case,
    case_output_dir: Path,
    result: dict[str, object],
    status: str,
    resource_failure: bool,
    duration_minutes: int,
    interval_seconds: int,
) -> dict[str, object]:
    return apply_strict_checkpoint_union(
        case=case,
        result=result,
        case_output_dir=case_output_dir,
        status=status,
        resource_failure=resource_failure,
        duration_minutes=duration_minutes,
        interval_seconds=interval_seconds,
    )


def merged_progress_trend(
    case_output_dir: Path,
    *,
    interval_seconds: int,
    duration_minutes: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for progress_path in sorted(case_output_dir.glob("*_coverage_progress*.jsonl")):
        records.extend(load_jsonl_records(progress_path))
    records = sorted(records, key=lambda item: float(item.get("elapsed_seconds") or 0.0))
    trend: list[dict[str, object]] = []
    cursor = 0
    high_water: dict[str, object] | None = None
    bucket_seconds = max(1, int(interval_seconds))
    total_buckets = max(0, int(duration_minutes * 60 // bucket_seconds))
    for bucket_index in range(total_buckets + 1):
        threshold = bucket_index * bucket_seconds
        while cursor < len(records):
            elapsed = float(records[cursor].get("elapsed_seconds") or 0.0)
            if elapsed > threshold + 1e-6:
                break
            if progress_score(records[cursor]) > progress_score(high_water):
                high_water = records[cursor]
            cursor += 1
        minute = bucket_index * (bucket_seconds // 60)
        if high_water is None:
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
        else:
            trend.append({
                "minute": minute,
                "valid_covered_bbs": high_water.get("valid_covered_bbs"),
                "valid_coverage_rate": high_water.get("valid_coverage_rate"),
                "valid_entry_plus_vector_static_reachable_bbs": high_water.get("valid_entry_plus_vector_static_reachable_bbs"),
                "valid_entry_plus_vector_static_reachable_covered_bbs": high_water.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
                "valid_entry_plus_vector_reachable_coverage_rate": high_water.get("valid_entry_plus_vector_reachable_coverage_rate"),
                "extended_static_roots_available": high_water.get("extended_static_roots_available"),
                "valid_extended_static_reachable_bbs": high_water.get("valid_extended_static_reachable_bbs"),
                "valid_extended_static_reachable_covered_bbs": high_water.get("valid_extended_static_reachable_covered_bbs"),
                "valid_extended_static_reachable_coverage_rate": high_water.get("valid_extended_static_reachable_coverage_rate"),
                "covered_bbs": high_water.get("covered_bbs"),
                "coverage_rate": high_water.get("coverage_rate"),
            })
    return trend


def apply_strict_checkpoint_union(
    *,
    case,
    result: dict[str, object],
    case_output_dir: Path,
    status: str,
    resource_failure: bool,
    duration_minutes: int,
    interval_seconds: int,
) -> dict[str, object]:
    covered, best_checkpoint, checkpoint_files = load_checkpoint_union(case_output_dir)
    progress_covered, best_progress_with_list, progress_files = load_progress_union(case_output_dir)
    covered.update(progress_covered)
    report_file = Path(str(result.get("report_file") or ""))
    report_payload = load_json(report_file) if report_file.exists() else None
    if isinstance(report_payload, dict):
        covered.update(parse_int_set(report_payload.get("covered_bb_list")))
        covered.update(parse_int_set(report_payload.get("covered_valid_bb_list")))
    valid_bbs = set(getattr(case, "valid_bbs", set()) or set())
    best_progress = load_progress_high_water(case_output_dir) or {}
    best_progress_valid_count = 0
    try:
        best_progress_valid_count = int(best_progress.get("valid_covered_bbs") or 0)
    except (TypeError, ValueError):
        best_progress_valid_count = 0
    covered_valid = covered & valid_bbs if valid_bbs else covered
    if not covered_valid and best_progress_valid_count <= 0:
        return result
    current_valid = 0
    try:
        current_valid = int(result.get("valid_covered_bbs") or 0)
    except (TypeError, ValueError):
        current_valid = 0
    if status == "completed" and not resource_failure and len(covered_valid) <= current_valid:
        return result

    merged = dict(result)
    valid_total = len(valid_bbs) or int(merged.get("valid_total_bbs") or 0)
    list_valid_count = len(covered_valid)
    final_valid_count = max(list_valid_count, best_progress_valid_count, current_valid)
    covered_total = int(merged.get("total_bbs") or best_progress.get("total_bbs") or 0)
    if not covered_total:
        covered_total = max(
            int(merged.get("covered_bbs") or 0),
            int(best_progress.get("covered_bbs") or 0),
            int((best_checkpoint or {}).get("global_covered_bbs") or 0),
            len(covered),
        )
    best_progress_covered_count = 0
    try:
        best_progress_covered_count = int(best_progress.get("covered_bbs") or 0)
    except (TypeError, ValueError):
        best_progress_covered_count = 0
    final_covered_count = max(len(covered), best_progress_covered_count, int(merged.get("covered_bbs") or 0))
    elapsed = float(
        merged.get("execution_time_seconds")
        or best_progress.get("elapsed_seconds")
        or (best_checkpoint or {}).get("elapsed_seconds")
        or 0.0
    )
    union_status = (
        "completed_partial_strict_resource_union"
        if resource_failure or status == "failed"
        else "completed_strict_checkpoint_union"
    )
    union_report_file = case_output_dir / f"{Path(str(case.elf_path)).stem}_strict_checkpoint_union_report.json"
    list_complete = final_valid_count == list_valid_count and final_covered_count == len(covered)
    merged.update({
        "status": union_status,
        "strict_real_entry_replayable": True,
        "coverage_entry_derived": True,
        "coverage_source": "dynamic_unicorn_execution",
        "coverage_counting_rule": (
            "Strict union recovered from coverage checkpoints emitted by "
            "entry-derived Unicorn execution/replay; no jump-to-BB coverage is counted."
        ),
        "resource_failure": bool(resource_failure),
        "stall_watchdog_stop_reason": (
            report_payload.get("stall_watchdog_stop_reason")
            if isinstance(report_payload, dict)
            else None
        ),
        "stall_watchdog": (
            report_payload.get("stall_watchdog")
            if isinstance(report_payload, dict)
            else None
        ),
        "stall_watchdog_stage_truncations": (
            report_payload.get("stall_watchdog_stage_truncations")
            if isinstance(report_payload, dict)
            else None
        ),
        "terminated_early": (
            bool(report_payload.get("terminated_early"))
            if isinstance(report_payload, dict)
            else False
        ),
        "terminated_early_reason": (
            report_payload.get("terminated_early_reason")
            if isinstance(report_payload, dict)
            else None
        ),
        "wallclock_used_seconds": (
            report_payload.get("wallclock_used_seconds")
            if isinstance(report_payload, dict)
            else None
        ),
        "configured_budget_seconds": (
            report_payload.get("configured_budget_seconds")
            if isinstance(report_payload, dict)
            else None
        ),
        "stall_watchdog_budget_remaining_seconds": (
            report_payload.get("stall_watchdog_budget_remaining_seconds")
            if isinstance(report_payload, dict)
            else None
        ),
        "checkpoint_union": True,
        "checkpoint_union_files": checkpoint_files,
        "checkpoint_union_source_count": len(checkpoint_files),
        "progress_union_files": progress_files,
        "progress_union_source_count": len(progress_files),
        "checkpoint_union_covered_bbs": len(covered),
        "progress_high_water_covered_bbs": best_progress_covered_count,
        "progress_high_water_valid_covered_bbs": best_progress_valid_count,
        "coverage_list_recoverable_bbs": len(covered),
        "coverage_list_recoverable_valid_bbs": list_valid_count,
        "coverage_list_complete": list_complete,
        "covered_bbs": final_covered_count,
        "total_bbs": covered_total,
        "coverage_rate": (
            final_covered_count / covered_total * 100.0
            if covered_total else merged.get("coverage_rate")
        ),
        "valid_total_bbs": valid_total,
        "valid_covered_bbs": final_valid_count,
        "valid_coverage_rate": (
            final_valid_count / valid_total * 100.0
            if valid_total else 0.0
        ),
        "execution_time_seconds": elapsed,
        "progress_trend": merged_progress_trend(
            case_output_dir,
            interval_seconds=interval_seconds,
            duration_minutes=duration_minutes,
        ),
        "original_report_file": merged.get("report_file"),
        "report_file": str(union_report_file),
        "checkpoint_union_report_file": str(union_report_file),
    })
    for key in (
        "valid_entry_plus_vector_static_reachable_bbs",
        "valid_entry_plus_vector_static_reachable_covered_bbs",
        "valid_entry_plus_vector_reachable_coverage_rate",
        "valid_entry_static_reachable_bbs",
        "valid_entry_static_reachable_covered_bbs",
        "valid_entry_static_reachable_coverage_rate",
        "extended_static_roots_available",
        "extended_static_root_bbs",
        "valid_extended_static_reachable_bbs",
        "valid_extended_static_reachable_covered_bbs",
        "valid_extended_static_reachable_uncovered_bbs",
        "valid_extended_static_reachable_coverage_rate",
        "valid_static_unreachable_from_extended_roots",
        "extended_static_unreachable_valid_ratio",
    ):
        if best_progress.get(key) is not None:
            merged[key] = best_progress.get(key)
    report_payload = {
        "firmware": str(case.elf_path.resolve()),
        "rel_path": case.rel_path,
        "runner": "gateway_interleaved_cached",
        "coverage_source": "dynamic_unicorn_execution",
        "coverage_entry_derived": True,
        "strict_real_entry_replayable": True,
        "strict_real_entry_replay_contract": {
            "counted_phases": ["checkpoint_union"],
            "invalid_counted_phases": [],
            "partial_checkpoint_recovery": bool(resource_failure or status == "failed"),
        },
        "valid_total_bbs": valid_total,
        "valid_covered_bbs": final_valid_count,
        "valid_coverage_rate": (
            final_valid_count / valid_total * 100.0
            if valid_total else 0.0
        ),
        "covered_bbs": final_covered_count,
        "coverage_list_complete": list_complete,
        "coverage_list_recoverable_bbs": len(covered),
        "coverage_list_recoverable_valid_bbs": list_valid_count,
        "progress_high_water_covered_bbs": best_progress_covered_count,
        "progress_high_water_valid_covered_bbs": best_progress_valid_count,
        "covered_valid_bb_list": sorted(covered_valid),
        "covered_bb_list": sorted(covered),
        "checkpoint_files": checkpoint_files,
        "progress_files": progress_files,
        "resource_failure": bool(resource_failure),
        "source_status": status,
        "stall_watchdog_stop_reason": merged.get("stall_watchdog_stop_reason"),
        "stall_watchdog": merged.get("stall_watchdog"),
        "stall_watchdog_stage_truncations": merged.get(
            "stall_watchdog_stage_truncations"
        ),
        "terminated_early": bool(merged.get("terminated_early")),
        "terminated_early_reason": merged.get("terminated_early_reason"),
        "wallclock_used_seconds": merged.get("wallclock_used_seconds"),
        "configured_budget_seconds": merged.get("configured_budget_seconds"),
        "stall_watchdog_budget_remaining_seconds": merged.get(
            "stall_watchdog_budget_remaining_seconds"
        ),
    }
    for key in (
        "valid_entry_plus_vector_static_reachable_bbs",
        "valid_entry_plus_vector_static_reachable_covered_bbs",
        "valid_entry_plus_vector_reachable_coverage_rate",
        "valid_entry_static_reachable_bbs",
        "valid_entry_static_reachable_covered_bbs",
        "valid_entry_static_reachable_coverage_rate",
        "valid_static_unreachable_from_entry_or_vectors",
        "extended_static_roots_available",
        "extended_static_root_bbs",
        "valid_extended_static_reachable_bbs",
        "valid_extended_static_reachable_covered_bbs",
        "valid_extended_static_reachable_uncovered_bbs",
        "valid_extended_static_reachable_coverage_rate",
        "valid_static_unreachable_from_extended_roots",
        "extended_static_unreachable_valid_ratio",
    ):
        if merged.get(key) is not None:
            report_payload[key] = merged.get(key)
    if best_progress_with_list:
        report_payload["best_progress_with_coverage_list"] = {
            key: best_progress_with_list.get(key)
            for key in (
                "event",
                "stage",
                "last_phase",
                "elapsed_seconds",
                "covered_bbs",
                "valid_covered_bbs",
                "valid_coverage_rate",
            )
        }
    if not list_complete:
        report_payload["coverage_list_caveat"] = (
            "Final BB counts include strict entry-derived progress high-water counts, "
            "but the full BB address list was not recoverable for every high-water "
            "point in this older run. Listed BBs are the recoverable strict subset."
        )
    write_json(Path(str(merged["checkpoint_union_report_file"])), report_payload)
    return merged


def write_driver_heartbeat(
    output_root: Path,
    *,
    started_at: float,
    pending: list[tuple[int, object]],
    running: dict[int, dict[str, object]],
    results: list[dict[str, object]],
    event: str,
) -> None:
    running_cases: list[dict[str, object]] = []
    for index, state in sorted(running.items()):
        process = state.get("process")
        case = state.get("case")
        running_cases.append({
            "case_index": index,
            "case": getattr(case, "rel_path", ""),
            "retry_count": state.get("retry_count"),
            "retry_mode": state.get("retry_mode"),
            "pid": getattr(process, "pid", None),
            "returncode": process.poll() if hasattr(process, "poll") else None,
            "progress_path": str(state.get("progress_path") or ""),
            "log_path": str(state.get("log_path") or ""),
            "launched_at": state.get("launched_at"),
        })
    write_json(output_root / "campaign_driver_heartbeat.json", {
        "event": event,
        "pid": os.getpid(),
        "timestamp": time.time(),
        "wallclock_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "elapsed_seconds": time.time() - started_at,
        "pending_cases": len(pending),
        "running_cases": running_cases,
        "results_count": len(results),
    })


def best_progress_record(path: Path) -> dict[str, object] | None:
    best_record = None
    for record in load_jsonl_records(path):
        if progress_score(record) > progress_score(best_record):
            best_record = record
    return best_record


def summarize_progress_only_result(
    *,
    case,
    case_output_dir: Path,
    progress_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    exit_code: int | None,
    status: str,
    duration_minutes: int,
    interval_seconds: int,
    retry_mode: str,
    resource_failure: bool,
) -> dict[str, object]:
    result = summarize_case_result(
        case=case,
        case_output_dir=case_output_dir,
        report_path=Path("__missing_report__.json"),
        progress_path=progress_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        exit_code=exit_code,
        status=status,
        duration_minutes=duration_minutes,
        interval_seconds=interval_seconds,
    )
    best_record = best_progress_record(progress_path) or {}
    result.update({
        "status": status,
        "retry_mode": retry_mode,
        "resource_failure": bool(resource_failure),
        "report_file": None,
        "valid_covered_bbs": best_record.get("valid_covered_bbs"),
        "valid_total_bbs": best_record.get("valid_total_bbs"),
        "valid_coverage_rate": best_record.get("valid_coverage_rate"),
        "covered_bbs": best_record.get("covered_bbs"),
        "total_bbs": best_record.get("total_bbs"),
        "coverage_rate": best_record.get("coverage_rate"),
        "strict_real_entry_replayable": best_record.get("strict_real_entry_replayable"),
        "execution_time_seconds": best_record.get("elapsed_seconds"),
        "valid_entry_plus_vector_static_reachable_bbs": best_record.get("valid_entry_plus_vector_static_reachable_bbs"),
        "valid_entry_plus_vector_static_reachable_covered_bbs": best_record.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
        "valid_entry_plus_vector_reachable_coverage_rate": best_record.get("valid_entry_plus_vector_reachable_coverage_rate"),
        "valid_entry_static_reachable_bbs": best_record.get("valid_entry_static_reachable_bbs"),
        "valid_entry_static_reachable_covered_bbs": best_record.get("valid_entry_static_reachable_covered_bbs"),
        "valid_entry_static_reachable_coverage_rate": best_record.get("valid_entry_static_reachable_coverage_rate"),
        "extended_static_roots_available": best_record.get("extended_static_roots_available"),
        "valid_extended_static_reachable_bbs": best_record.get("valid_extended_static_reachable_bbs"),
        "valid_extended_static_reachable_covered_bbs": best_record.get("valid_extended_static_reachable_covered_bbs"),
        "valid_extended_static_reachable_uncovered_bbs": best_record.get("valid_extended_static_reachable_uncovered_bbs"),
        "valid_extended_static_reachable_coverage_rate": best_record.get("valid_extended_static_reachable_coverage_rate"),
        "valid_static_unreachable_from_extended_roots": best_record.get("valid_static_unreachable_from_extended_roots"),
        "extended_static_unreachable_valid_ratio": best_record.get("extended_static_unreachable_valid_ratio"),
        "best_progress_event": best_record.get("event"),
        "best_progress_stage": best_record.get("stage"),
    })
    return result


def merge_best_progress_with_report_context(
    best_progress: dict[str, object],
    final_result: dict[str, object],
) -> dict[str, object]:
    """Preserve high-water strict coverage while keeping final report diagnostics."""
    merged = dict(best_progress)

    def missing(value: object) -> bool:
        return value is None or value == ""

    diagnostic_keys = (
        "report_file",
        "valid_entry_plus_vector_static_reachable_bbs",
        "valid_entry_plus_vector_static_reachable_covered_bbs",
        "valid_entry_plus_vector_reachable_coverage_rate",
        "valid_entry_static_reachable_bbs",
        "valid_entry_static_reachable_covered_bbs",
        "valid_entry_static_reachable_coverage_rate",
        "valid_static_unreachable_from_entry_or_vectors",
        "extended_static_roots_available",
        "extended_static_root_bbs",
        "valid_extended_static_reachable_bbs",
        "valid_extended_static_reachable_covered_bbs",
        "valid_extended_static_reachable_uncovered_bbs",
        "valid_extended_static_reachable_coverage_rate",
        "valid_static_unreachable_from_extended_roots",
        "extended_static_unreachable_valid_ratio",
        "entry_reachable_uncovered_bbs",
        "vector_only_uncovered_bbs",
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
        "top_frontier_predecessors",
        "runner",
        "coverage_entry_derived",
        "coverage_source",
        "loaded_unicorn_library",
    )
    for key in diagnostic_keys:
        if missing(merged.get(key)) and not missing(final_result.get(key)):
            merged[key] = final_result.get(key)
    merged["final_retry_status"] = final_result.get("status")
    merged["final_retry_valid_covered_bbs"] = final_result.get("valid_covered_bbs")
    merged["final_retry_valid_coverage_rate"] = final_result.get("valid_coverage_rate")
    merged["final_retry_report_file"] = final_result.get("report_file")
    merged["final_retry_progress_jsonl_file"] = final_result.get("progress_jsonl_file")
    if progress_score(merged) > progress_score(final_result):
        merged["report_file"] = None
        merged["report_file_caveat"] = (
            "Best strict progress count exceeded the final report count; no report "
            "file is attached for this count-only high-water result."
        )
    return merged


def should_resume_skip_report(
    *,
    resume: bool,
    retry_count: int,
    segment_index: int,
    report_path: Path,
) -> bool:
    return (
        bool(resume)
        and int(retry_count) == 0
        and int(segment_index) == 0
        and report_path.exists()
    )


def select_resume_report(
    *,
    resume: bool,
    retry_count: int,
    segment_index: int,
    case_output_dir: Path,
    firmware: Path,
) -> Path | None:
    """Return the best existing strict report for --resume.

    Older resume logic only looked for the final interleaved report.  Long
    campaigns can also finish a case through strict checkpoint-union recovery,
    especially after timeouts or native resource failures.  Reusing that report
    keeps the campaign from rerunning already-recovered cases.
    """
    if not (
        bool(resume)
        and int(retry_count) == 0
        and int(segment_index) == 0
    ):
        return None

    candidates = [
        case_output_dir / f"{firmware.stem}_strict_checkpoint_union_report.json",
        case_output_dir / f"{firmware.stem}_interleaved_report.json",
    ]
    best_path: Path | None = None
    best_score: tuple[int, float, float, int] = (0, 0.0, 0.0, -1)
    for priority, path in enumerate(candidates):
        if not path.exists():
            continue
        payload = load_json(path) or {}
        score = progress_score(payload) + (len(candidates) - priority,)
        if score > best_score:
            best_path = path
            best_score = score
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("LSGEMU_CONFIG_FILE"), help="Deployment YAML/JSON config.")
    parser.add_argument("--valid-root", default=str(DEFAULT_VALID_BB_ROOT))
    parser.add_argument(
        "--output-root",
        default=str(GLOBAL_RUNS_ROOT / f"elfmultifuzz_interleaved_strict_parallel_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--firmware-minutes", type=int, default=240)
    parser.add_argument(
        "--case-segment-minutes",
        type=int,
        default=0,
        help=(
            "Run each firmware as multiple independent entry-started child "
            "segments and union strict coverage. 0 keeps the legacy single "
            "process per firmware behavior."
        ),
    )
    parser.add_argument("--progress-interval-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--case-timeout-grace-seconds",
        type=int,
        default=120,
        help=(
            "Kill a child firmware process after requested segment duration "
            "plus this grace period. The campaign preserves strict progress/"
            "checkpoint coverage from the killed process."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of firmware child processes to run concurrently. The "
            "coverage-first default is 1 because short wallclock-bounded "
            "Unicorn replay stages are CPU-sensitive; use >1 only for "
            "throughput-oriented sweeps."
        ),
    )
    parser.add_argument("--family-filter", default=None)
    parser.add_argument("--name-filter", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-log-level", default="WARNING")
    parser.add_argument("--python-executable", default=sys.executable or "python3")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--disable-default-strict-tuning",
        action="store_true",
        help="Do not inject the 4h strict replay tuning defaults before --extra-arg overrides.",
    )
    parser.add_argument(
        "--case-memory-limit-gb",
        type=int,
        default=64,
        help="Per-firmware child process address-space limit in GiB; <=0 disables the limit.",
    )
    parser.add_argument(
        "--disable-crash-retry",
        action="store_true",
        help="Do not retry native-crashed cases with progressively safer strict entry-derived modes.",
    )
    parser.add_argument(
        "--retry-resource-failures",
        action="store_true",
        help=(
            "Retry cases that fail with native/resource exhaustion. By default the "
            "campaign preserves strict checkpoint-union coverage and moves on."
        ),
    )
    parser.add_argument(
        "--low-coverage-retry-rate",
        type=float,
        default=80.0,
        help=(
            "Retry a short completed case when reachable-valid coverage is below "
            "this percentage; <=0 disables coverage-based retry."
        ),
    )
    parser.add_argument(
        "--low-coverage-retry-min-elapsed-ratio",
        type=float,
        default=0.85,
        help=(
            "Only coverage-retry cases that consumed less than this fraction of "
            "the requested per-firmware wallclock budget."
        ),
    )
    parser.add_argument(
        "--retry-full-duration-low-coverage",
        action="store_true",
        help=(
            "Also apply low-coverage retry plans to cases that consumed the full "
            "requested duration.  Use this for adaptive deep-rescue campaigns; "
            "the default only retries suspiciously short low-coverage runs."
        ),
    )
    parser.add_argument(
        "--disable-low-coverage-retry",
        action="store_true",
        help=(
            "Disable coverage-based compensation retries. This is independent "
            "from --disable-crash-retry, which only controls native/runtime "
            "failure retries."
        ),
    )
    parser.add_argument(
        "--low-coverage-retry-minutes",
        type=int,
        default=0,
        help=(
            "Wallclock minutes for coverage-based compensation retries. 0 uses "
            "--firmware-minutes. This is useful for short legacy/stateful "
            "ensemble probes after a full primary run."
        ),
    )
    parser.add_argument(
        "--low-coverage-retry-max-attempts",
        type=int,
        default=-1,
        help=(
            "Maximum number of coverage-based compensation retry plans per "
            "firmware. -1 auto-selects a short-run coverage-first default; "
            "0 allows all configured low-coverage plans."
        ),
    )
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
    best_attempts: dict[int, dict[str, object]] = {}

    started_at = time.time()
    write_json(output_root / "campaign_manifest.json", {
        "runner": "elfmultifuzz_interleaved_strict_parallel_campaign",
        "module": INTERLEAVED_MODULE,
        "local_unicorn_shared_lib": str(LOCAL_UNICORN_SHARED_LIB),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "jobs": max(1, int(args.jobs)),
        "jobs_policy": (
            "coverage_first_serial_default"
            if int(args.jobs) <= 1
            else "throughput_parallel_user_requested"
        ),
        "case_memory_limit_gb": int(args.case_memory_limit_gb),
        "default_strict_tuning_args": [] if args.disable_default_strict_tuning else STRICT_4H_TUNING_ARGS,
        "crash_retry_plans": [name for name, _args in CRASH_RETRY_PLANS],
        "retry_resource_failures": bool(args.retry_resource_failures),
        "low_coverage_retry_plans": [name for name, _args in LOW_COVERAGE_RETRY_PLANS],
        "low_coverage_retry_rate": float(args.low_coverage_retry_rate),
        "low_coverage_retry_min_elapsed_ratio": float(args.low_coverage_retry_min_elapsed_ratio),
        "retry_full_duration_low_coverage": bool(args.retry_full_duration_low_coverage),
        "disable_low_coverage_retry": bool(args.disable_low_coverage_retry),
        "low_coverage_retry_minutes": int(args.low_coverage_retry_minutes),
        "low_coverage_retry_max_attempts": int(args.low_coverage_retry_max_attempts),
        "effective_low_coverage_retry_plan_limit": effective_low_coverage_retry_plan_limit(args),
        "extra_args": list(args.extra_arg or []),
        "cases_total": len(selected_cases),
        "firmware_minutes": args.firmware_minutes,
        "case_segment_minutes": int(args.case_segment_minutes),
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
    })

    results: list[dict[str, object]] = []
    pending = list(enumerate(selected_cases, start=1))
    running: dict[int, dict[str, object]] = {}
    campaign_progress_path = output_root / "campaign_progress.jsonl"
    max_jobs = max(1, int(args.jobs))

    append_jsonl(campaign_progress_path, {
        "event": "campaign_start",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "cases_total": len(selected_cases),
        "jobs": max_jobs,
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
    write_driver_heartbeat(
        output_root,
        started_at=started_at,
        pending=pending,
        running=running,
        results=results,
        event="campaign_start",
    )

    def launch(
        index: int,
        case,
        *,
        retry_count: int = 0,
        retry_args: list[str] | None = None,
        segment_index: int = 0,
    ) -> None:
        case_output_dir = (output_root / case.rel_path).resolve()
        case_output_dir.mkdir(parents=True, exist_ok=True)
        firmware = case.elf_path.resolve()
        suffix = attempt_suffix(retry_count, segment_index)
        report_path = case_output_dir / f"{firmware.stem}_interleaved_report.json"
        progress_path = case_output_dir / f"{firmware.stem}_coverage_progress{suffix}.jsonl"
        checkpoint_path = case_output_dir / f"{firmware.stem}_coverage_checkpoints{suffix}.jsonl"
        llm_journal_path = case_output_dir / f"{firmware.stem}_llm_history{suffix}.jsonl"
        log_path = case_output_dir / f"run{suffix}.log"
        total_segments = segment_count_for_case(
            int(low_coverage_retry_minutes_for(args, retry_count)),
            int(args.case_segment_minutes),
        )
        segment_minutes = segment_duration_minutes(
            int(low_coverage_retry_minutes_for(args, retry_count)),
            int(args.case_segment_minutes),
            int(segment_index),
        )
        elapsed_offset_seconds = int(segment_index) * max(0, int(args.case_segment_minutes)) * 60

        resume_report_path = select_resume_report(
            resume=args.resume,
            retry_count=retry_count,
            segment_index=segment_index,
            case_output_dir=case_output_dir,
            firmware=firmware,
        )
        if resume_report_path is not None:
            result = summarize_case_result(
                case=case,
                case_output_dir=case_output_dir,
                report_path=resume_report_path,
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
                "report_file": str(resume_report_path),
            })
            return

        cmd = [
            args.python_executable,
            "-m",
            INTERLEAVED_MODULE,
            "--firmware", str(firmware),
            "--output-dir", str(case_output_dir),
            "--total-time-minutes", str(segment_minutes),
            "--log-level", args.runner_log_level,
        ]
        if args.config:
            cmd.extend(["--config", str(args.config)])
        if not args.disable_default_strict_tuning:
            cmd.extend(STRICT_4H_TUNING_ARGS)
        cmd.extend(args.extra_arg)
        if retry_args:
            cmd.extend(retry_args)

        env = build_env(case_output_dir, progress_path, checkpoint_path, llm_journal_path)
        env["LSGEMU_PROGRESS_INTERVAL_SECONDS"] = str(max(1, args.progress_interval_seconds))
        env["LSGEMU_COVERAGE_CHECKPOINT_INTERVAL"] = str(max(1, args.progress_interval_seconds))
        if total_segments > 1:
            env["LSGEMU_PROGRESS_ELAPSED_OFFSET_SECONDS"] = str(elapsed_offset_seconds)
            env["LSGEMU_CASE_SEGMENT_INDEX"] = str(int(segment_index))
            env["LSGEMU_CASE_SEGMENT_SECONDS"] = str(max(1, int(args.case_segment_minutes) * 60))
            env["LSGEMU_SEGMENT_DIVERSIFY"] = "1"

        log_file = log_path.open("ab")
        launched_at = time.time()
        preexec_fn = None
        if int(args.case_memory_limit_gb) > 0:
            preexec_fn = lambda: apply_child_limits(int(args.case_memory_limit_gb))

        process = subprocess.Popen(
            cmd,
            cwd=str(configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4")),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec_fn,
        )
        running[index] = {
            "case": case,
            "process": process,
            "log_file": log_file,
            "case_output_dir": case_output_dir,
            "report_path": report_path,
            "progress_path": progress_path,
            "checkpoint_path": checkpoint_path,
            "llm_journal_path": llm_journal_path,
            "log_path": log_path,
            "last_progress_signature": None,
            "retry_count": retry_count,
            "retry_args": list(retry_args or []),
            "retry_mode": retry_mode_name(retry_count),
            "segment_index": int(segment_index),
            "total_segments": int(total_segments),
            "segment_minutes": int(segment_minutes),
            "elapsed_offset_seconds": int(elapsed_offset_seconds),
            "launched_at": launched_at,
        }
        append_jsonl(campaign_progress_path, {
            "event": "case_start",
            "case_index": index,
            "case": case.rel_path,
            "firmware": str(firmware),
            "command": cmd,
            "retry_count": retry_count,
            "retry_mode": running[index]["retry_mode"],
            "segment_index": int(segment_index),
            "total_segments": int(total_segments),
            "segment_minutes": int(segment_minutes),
            "elapsed_offset_seconds": int(elapsed_offset_seconds),
            "case_memory_limit_gb": int(args.case_memory_limit_gb),
            "llm_history_journal": str(llm_journal_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        })

    while pending or running:
        write_driver_heartbeat(
            output_root,
            started_at=started_at,
            pending=pending,
            running=running,
            results=results,
            event="poll_start",
        )
        while pending and len(running) < max_jobs:
            index, case = pending.pop(0)
            launch(index, case)

        finished: list[int] = []
        for index, state in list(running.items()):
            case = state["case"]
            progress_path = state["progress_path"]
            latest_progress = load_last_jsonl_record(progress_path)
            if latest_progress is not None:
                signature = (
                    latest_progress.get("event"),
                    latest_progress.get("elapsed_seconds"),
                    latest_progress.get("valid_covered_bbs"),
                )
                if signature != state.get("last_progress_signature"):
                    state["last_progress_signature"] = signature
                    append_jsonl(campaign_progress_path, {
                        "event": "case_progress",
                        "case_index": index,
                        "case": case.rel_path,
                        "progress": latest_progress,
                    })

            process = state["process"]
            rc = process.poll()
            if rc is None:
                segment_seconds = max(1, int(state.get("segment_minutes") or args.firmware_minutes) * 60)
                grace_seconds = max(0, int(args.case_timeout_grace_seconds))
                launched_at = float(state.get("launched_at") or time.time())
                elapsed_child = time.time() - launched_at
                if elapsed_child > segment_seconds + grace_seconds:
                    state["timeout_killed"] = True
                    append_jsonl(campaign_progress_path, {
                        "event": "case_timeout_kill",
                        "case_index": index,
                        "case": case.rel_path,
                        "pid": getattr(process, "pid", None),
                        "elapsed_seconds": elapsed_child,
                        "segment_seconds": segment_seconds,
                        "grace_seconds": grace_seconds,
                        "retry_count": int(state.get("retry_count") or 0),
                        "retry_mode": state.get("retry_mode"),
                    })
                    try:
                        process.terminate()
                        try:
                            rc = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            rc = process.wait(timeout=10)
                    except Exception:
                        rc = process.poll()
                    if rc is None:
                        continue
            if rc is None:
                continue
            state["log_file"].close()
            report_fresh = False
            if state["report_path"].exists():
                try:
                    report_fresh = state["report_path"].stat().st_mtime >= float(state.get("launched_at") or 0.0)
                except OSError:
                    report_fresh = False
            status = "completed" if int(rc) == 0 and report_fresh else "failed"
            timed_out = bool(state.get("timeout_killed"))
            retry_count = int(state.get("retry_count") or 0)
            resource_failure = log_indicates_resource_failure(state["log_path"])
            if timed_out:
                resource_failure = True
            if status == "failed":
                best_progress = summarize_progress_only_result(
                    case=case,
                    case_output_dir=state["case_output_dir"],
                    progress_path=progress_path,
                    checkpoint_path=state["checkpoint_path"],
                    log_path=state["log_path"],
                    exit_code=int(rc),
                    status="failed_best_progress",
                    duration_minutes=args.firmware_minutes,
                    interval_seconds=args.progress_interval_seconds,
                    retry_mode=str(state.get("retry_mode") or "normal"),
                    resource_failure=bool(resource_failure),
                )
                if timed_out:
                    best_progress["timeout_killed"] = True
                if (
                    progress_score(best_progress)
                    > progress_score(best_attempts.get(index))
                ):
                    remember_best_attempt(best_attempts, index, best_progress)
            if (
                status == "failed"
                and not args.disable_crash_retry
                and retry_count < len(CRASH_RETRY_PLANS)
                and int(state.get("segment_index") or 0) == 0
                and (int(rc) < 0 or (resource_failure and args.retry_resource_failures))
            ):
                retry_mode, retry_args = CRASH_RETRY_PLANS[retry_count]
                append_jsonl(campaign_progress_path, {
                    "event": "case_retry_after_runtime_failure",
                    "case_index": index,
                    "case": case.rel_path,
                    "exit_code": int(rc),
                    "resource_failure": bool(resource_failure),
                    "failed_retry_mode": state.get("retry_mode"),
                    "retry_mode": retry_mode,
                })
                launch(
                    index,
                    case,
                    retry_count=retry_count + 1,
                    retry_args=retry_args,
                )
                write_driver_heartbeat(
                    output_root,
                    started_at=started_at,
                    pending=pending,
                    running=running,
                    results=results,
                    event="retry_after_runtime_failure",
                )
                continue
            result = summarize_case_result(
                case=case,
                case_output_dir=state["case_output_dir"],
                report_path=state["report_path"],
                progress_path=progress_path,
                checkpoint_path=state["checkpoint_path"],
                log_path=state["log_path"],
                exit_code=int(rc),
                status=status,
                duration_minutes=args.firmware_minutes,
                interval_seconds=args.progress_interval_seconds,
            )
            result["llm_history_journal_file"] = str(state.get("llm_journal_path") or "")
            result = apply_strict_checkpoint_union(
                case=case,
                result=result,
                case_output_dir=state["case_output_dir"],
                status=status,
                resource_failure=bool(resource_failure),
                duration_minutes=args.firmware_minutes,
                interval_seconds=args.progress_interval_seconds,
            )
            segment_index = int(state.get("segment_index") or 0)
            total_segments = int(state.get("total_segments") or 1)
            if (
                total_segments > 1
                and segment_index + 1 < total_segments
                and not resource_failure
            ):
                append_jsonl(campaign_progress_path, {
                    "event": "case_segment_complete",
                    "case_index": index,
                    "case": case.rel_path,
                    "exit_code": int(rc),
                    "status": result.get("status", status),
                    "segment_index": segment_index,
                    "total_segments": total_segments,
                    "valid_covered_bbs": result.get("valid_covered_bbs"),
                    "valid_coverage_rate": result.get("valid_coverage_rate"),
                    "strict_real_entry_replayable": result.get("strict_real_entry_replayable"),
                })
                running.pop(index, None)
                launch(
                    index,
                    case,
                    retry_count=retry_count,
                    retry_args=list(state.get("retry_args") or []),
                    segment_index=segment_index + 1,
                )
                write_driver_heartbeat(
                    output_root,
                    started_at=started_at,
                    pending=pending,
                    running=running,
                    results=results,
                    event="case_segment_next",
                )
                continue
            if total_segments > 1:
                segment_elapsed = float(result.get("execution_time_seconds") or 0.0)
                total_elapsed = float(state.get("elapsed_offset_seconds") or 0.0) + segment_elapsed
                result["execution_time_seconds"] = max(segment_elapsed, total_elapsed)
                result["segmented_union"] = True
                result["case_segment_minutes"] = int(args.case_segment_minutes)
                result["case_segments_completed"] = segment_index + 1
                result = build_case_union_result(
                    case=case,
                    case_output_dir=state["case_output_dir"],
                    result=result,
                    status=status,
                    resource_failure=bool(resource_failure),
                    duration_minutes=args.firmware_minutes,
                    interval_seconds=args.progress_interval_seconds,
                )
                result["segmented_union"] = True
                result["case_segment_minutes"] = int(args.case_segment_minutes)
                result["case_segments_completed"] = segment_index + 1
            if result.get("status") in COMPLETED_STATUSES:
                low_coverage_retry, low_retry_diagnostics = should_low_coverage_retry(
                    args=args,
                    result=result,
                    retry_count=retry_count,
                    resource_failure=bool(resource_failure),
                    segment_index=segment_index,
                    total_segments=total_segments,
                )
                if low_retry_diagnostics.get("watchdog_stopped") and low_retry_diagnostics.get(
                    "would_retry_without_watchdog"
                ):
                    append_jsonl(campaign_progress_path, {
                        "event": "case_skip_low_coverage_retry_after_stall_watchdog",
                        "case_index": index,
                        "case": case.rel_path,
                        "reachable_valid_coverage_rate": low_retry_diagnostics.get(
                            "reachable_valid_coverage_rate"
                        ),
                        "elapsed_seconds": low_retry_diagnostics.get("elapsed_seconds"),
                        "requested_seconds": low_retry_diagnostics.get("requested_seconds"),
                        "stall_watchdog_stop_reason": result.get("stall_watchdog_stop_reason"),
                    })
                if low_coverage_retry:
                    remember_best_attempt(best_attempts, index, result)
                    low_retry_count = max(0, retry_count - len(CRASH_RETRY_PLANS))
                    retry_mode, retry_args = LOW_COVERAGE_RETRY_PLANS[low_retry_count]
                    append_jsonl(campaign_progress_path, {
                        "event": "case_retry_after_low_coverage_short_run",
                        "case_index": index,
                        "case": case.rel_path,
                        "reachable_valid_coverage_rate": low_retry_diagnostics.get(
                            "reachable_valid_coverage_rate"
                        ),
                        "elapsed_seconds": low_retry_diagnostics.get("elapsed_seconds"),
                        "requested_seconds": low_retry_diagnostics.get("requested_seconds"),
                        "failed_retry_mode": state.get("retry_mode"),
                        "retry_mode": retry_mode,
                    })
                    launch(
                        index,
                        case,
                        retry_count=len(CRASH_RETRY_PLANS) + low_retry_count + 1,
                        retry_args=retry_args,
                    )
                    write_driver_heartbeat(
                        output_root,
                        started_at=started_at,
                        pending=pending,
                        running=running,
                        results=results,
                        event="retry_after_low_coverage",
                    )
                    continue
            best_progress = best_attempts.get(index)
            if (
                result.get("status") in COMPLETED_STATUSES
                and best_progress is not None
                and progress_score(best_progress) > progress_score(result)
            ):
                best_progress = merge_best_progress_with_report_context(best_progress, result)
                best_progress["status"] = "completed_best_progress_after_retry"
                result = best_progress
            if progress_score(result) >= progress_score(best_attempts.get(index)):
                best_attempts.pop(index, None)
            results.append(result)
            append_jsonl(campaign_progress_path, {
                "event": "case_complete",
                "case_index": index,
                "case": case.rel_path,
                "exit_code": int(rc),
                "status": result.get("status", status),
                "valid_covered_bbs": result.get("valid_covered_bbs"),
                "valid_coverage_rate": result.get("valid_coverage_rate"),
                "strict_real_entry_replayable": result.get("strict_real_entry_replayable"),
            })
            finished.append(index)

        for index in finished:
            running.pop(index, None)

        current_cases = ", ".join(
            f"{index}:{state['case'].rel_path}"
            for index, state in sorted(running.items())
        )
        write_campaign_outputs(
            output_root,
            started_at=started_at,
            selected_cases=selected_cases,
            results=results,
            firmware_minutes=args.firmware_minutes,
            interval_seconds=args.progress_interval_seconds,
            completed=False,
            current_case=current_cases or None,
        )
        write_driver_heartbeat(
            output_root,
            started_at=started_at,
            pending=pending,
            running=running,
            results=results,
            event="poll_end",
        )
        if pending or running:
            time.sleep(max(1, int(args.poll_seconds)))

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
    write_driver_heartbeat(
        output_root,
        started_at=started_at,
        pending=pending,
        running=running,
        results=results,
        event="campaign_complete",
    )


def _argv_output_root() -> Path | None:
    for index, item in enumerate(sys.argv):
        if item == "--output-root" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
        if item.startswith("--output-root="):
            return Path(item.split("=", 1)[1]).resolve()
    return None


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        output_root = _argv_output_root()
        if output_root is not None:
            try:
                output_root.mkdir(parents=True, exist_ok=True)
                with (output_root / "campaign_uncaught_exception.log").open("a") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()))
                    f.write("\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                live_status = output_root / "campaign_live_status.json"
                previous_status = {}
                if live_status.exists():
                    try:
                        previous_status = load_json(live_status) or {}
                    except Exception:
                        previous_status = {}
                previous_status.update({
                    "completed": False,
                    "driver_failed": True,
                    "driver_failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    "driver_failure_log": str(output_root / "campaign_uncaught_exception.log"),
                })
                write_json(live_status, previous_status)
            except Exception:
                pass
        raise
