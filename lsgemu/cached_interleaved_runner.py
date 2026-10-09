#!/usr/bin/env python3
"""
Cached Gateway interleaved test.

Runs:
1. baseline
2. ISR exploration
3. interleaved reservoir + branch_mmio

It reuses HistoricalRunner's static cache so iteration focuses on runtime
behavior instead of repeating Ghidra analysis.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import yaml

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from lsgemu.constraint_utils import merge_learned_constraints_into, seed_constraint_file
    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import HistoricalRunner, DEFAULT_REAL_GATEWAY, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from lsgemu.run_profile import apply_run_profile_environment, load_run_profile
    from lsgemu.analysis.snapshot_memory import configure_snapshot_storage_for_run
else:
    from .constraint_utils import merge_learned_constraints_into, seed_constraint_file
    from .deployment_config import apply_config_from_argv, configured_path
    from .historical_runner import HistoricalRunner, DEFAULT_REAL_GATEWAY, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from .run_profile import apply_run_profile_environment, load_run_profile
    from .analysis.snapshot_memory import configure_snapshot_storage_for_run

apply_config_from_argv()

from lsgemu.runtime_bootstrap import bootstrap_runtime_dependencies

from lsgemu.artifact_io import append_jsonl as atomic_append_jsonl, atomic_json_dump
from lsgemu.scheduler.contracts import SchedulerRuntime
from lsgemu.scheduler.obligation_discovery import (
    build_ablation_fallback_policy as _build_ablation_fallback_policy,
)
from lsgemu.scheduler.queue_policy import (
    QueuePolicy,
    auto_round_limit as _auto_round_limit,
    dedupe_target_order as _dedupe_target_order,
    estimate_targeted_frontier_reserve_seconds as _estimate_targeted_frontier_reserve_seconds,
    rotate_target_order as _rotate_target_order,
    rotate_targets as _rotate_targets,
    target_head_tail_quota as _target_head_tail_quota,
)
from lsgemu.scheduler.stall_watchdog import (
    DEFAULT_WINDDOWN_MAX_SECONDS,
    STAGE_TRUNCATION_REASON,
    STOP_REASON as STALL_WATCHDOG_STOP_REASON,
    StallWatchdog,
    WinddownBudgetTracker,
)


def parse_int(value: str) -> int:
    return int(str(value), 0)


def load_llm_config(project_root: Path):
    path = configured_path("LSGEMU_LLM_CONFIG", project_root / "LLM.yaml")
    if not path.exists():
        return None, None
    with path.open() as f:
        return yaml.safe_load(f), path


def llm_disabled_by_env() -> bool:
    """r42 full_llm_off 臂：环境级 LLM 总闸（与未部署 LLM.yaml 同态）。

    LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS=0 在 llm_guide 语义里是"无上限"
    而非关闭，所以关 LLM 臂需要这把真实闸门：配置按未加载处理，
    use_llm=False，下游 LLM 客户端一律不构造。
    """
    return str(os.environ.get("LSGEMU_DISABLE_LLM") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def workspace_artifact_path(output_dir: Path, firmware: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{firmware.stem}{suffix}"


def progress_artifact_path(output_dir: Path, firmware: Path) -> Path:
    """Choose a progress path without mixing repeated direct invocations."""
    explicit = str(os.environ.get("LSGEMU_PROGRESS_JSONL") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    canonical = workspace_artifact_path(
        output_dir,
        firmware,
        "_coverage_progress.jsonl",
    ).resolve()
    reuse_existing = (
        str(os.environ.get("LSGEMU_PROGRESS_REUSE_EXISTING") or "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    if (
        not canonical.exists()
        or reuse_existing
        or bool(str(os.environ.get("LSGEMU_ATTEMPT_ID") or "").strip())
    ):
        return canonical
    invocation = (
        time.strftime("%Y%m%dT%H%M%S", time.localtime())
        + f"_{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"
    )
    return (
        output_dir
        / f"{firmware.stem}_{invocation}_coverage_progress.jsonl"
    ).resolve()


def auto_round_limit(requested_rounds: int, auto_max_rounds: int) -> int:
    return _auto_round_limit(requested_rounds, auto_max_rounds)


def rotate_targets(targets: list[int], round_index: int, limit: int) -> list[int]:
    return _rotate_targets(targets, round_index, limit)


def rotate_target_order(targets: list[int], round_index: int, step: int) -> list[int]:
    return _rotate_target_order(targets, round_index, step)


def segment_schedule_round_offset() -> int:
    try:
        segment_index = int(os.environ.get("LSGEMU_CASE_SEGMENT_INDEX") or 0)
    except ValueError:
        segment_index = 0
    if segment_index <= 0:
        try:
            elapsed_offset = int(float(os.environ.get("LSGEMU_PROGRESS_ELAPSED_OFFSET_SECONDS") or 0))
        except ValueError:
            elapsed_offset = 0
        try:
            segment_seconds = max(1, int(float(os.environ.get("LSGEMU_CASE_SEGMENT_SECONDS") or 1200)))
        except ValueError:
            segment_seconds = 1200
        segment_index = max(0, elapsed_offset // segment_seconds)
    return max(0, segment_index)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default) or default))
    except (TypeError, ValueError):
        return int(default)


def dedupe_target_order(*target_lists: list[int]) -> list[int]:
    return _dedupe_target_order(*target_lists)


def previewed_candidate_supplier(preview_targets: list[int], supplier):
    """
    Reuse the already-built candidate preview for the first formal round.

    Some firmwares have thousands of valid BBs and expensive uncovered/frontier
    summaries. In short wallclock probes the preview computation can consume
    enough time that recomputing the same candidate set immediately afterward
    prevents any targeted round from starting.
    """
    preview = list(preview_targets or [])
    preview_pending = True

    def _supplier():
        nonlocal preview_pending
        if preview_pending:
            preview_pending = False
            return list(preview)
        return list(supplier() or [])

    return _supplier


def target_head_tail_quota(total: int, head_ratio: float) -> tuple[int, int]:
    return _target_head_tail_quota(total, head_ratio)


def estimate_targeted_frontier_reserve_seconds(
    *,
    total_wallclock_budget_seconds: int,
    switch_frontier_enabled: bool,
    frontier_targeted_enabled: bool,
    frontier_cycle_auto: bool,
    switch_round_seconds: int,
    switch_rounds: int,
    switch_max_rounds: int,
    frontier_round_seconds: int,
    frontier_rounds: int,
    frontier_max_rounds: int,
    frontier_cycle_max_cycles: int,
    frontier_cycle_tail_reserve_seconds: int,
) -> int:
    return _estimate_targeted_frontier_reserve_seconds(
        total_wallclock_budget_seconds=total_wallclock_budget_seconds,
        switch_frontier_enabled=switch_frontier_enabled,
        frontier_targeted_enabled=frontier_targeted_enabled,
        frontier_cycle_auto=frontier_cycle_auto,
        switch_round_seconds=switch_round_seconds,
        switch_rounds=switch_rounds,
        switch_max_rounds=switch_max_rounds,
        frontier_round_seconds=frontier_round_seconds,
        frontier_rounds=frontier_rounds,
        frontier_max_rounds=frontier_max_rounds,
        frontier_cycle_max_cycles=frontier_cycle_max_cycles,
        frontier_cycle_tail_reserve_seconds=frontier_cycle_tail_reserve_seconds,
    )


def prioritized_target_fallback(
    runner: HistoricalRunner,
    target_bbs: set[int],
    *,
    switch_only: bool,
    max_targets: int,
    min_uncovered_successors: int,
    nearby_limit: int,
    exclude_targets: set[int] | None = None,
) -> list[int]:
    policy = getattr(runner, "queue_policy", None) or QueuePolicy(runner)
    return policy.prioritized_targets(
        target_bbs,
        switch_only=switch_only,
        max_targets=max_targets,
        min_uncovered_successors=min_uncovered_successors,
        nearby_limit=nearby_limit,
        exclude_targets=exclude_targets,
    )


def build_ablation_fallback_policy(args: argparse.Namespace) -> dict[str, object]:
    return _build_ablation_fallback_policy(args)


@contextmanager
def temporary_env(overrides: dict[str, str | None]):
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    atomic_append_jsonl(path, payload)


class CoverageProgressMonitor:
    def __init__(
        self,
        *,
        runner: HistoricalRunner,
        firmware: Path,
        output_path: Path,
        interval_seconds: int,
        started_at: float,
        stall_watchdog: StallWatchdog | None = None,
        wallclock_budget_seconds: int | None = None,
    ):
        self.runner = runner
        self.stall_watchdog = stall_watchdog
        self.wallclock_budget_seconds = (
            None if wallclock_budget_seconds is None else max(0, int(wallclock_budget_seconds))
        )
        self.wallclock_deadline = (
            started_at + self.wallclock_budget_seconds
            if self.wallclock_budget_seconds
            else None
        )
        # Frozen at the first telemetry point that sees the watchdog trigger:
        # how much of the configured wallclock budget was still unspent when
        # the run was stopped early (None while no deadline was configured).
        self.stall_watchdog_budget_snapshot: dict[str, object] | None = None
        self.firmware = firmware
        self.output_path = output_path
        self.interval_seconds = max(0, int(interval_seconds))
        self.started_at = float(started_at)
        self.attempt_id = str(os.environ.get("LSGEMU_ATTEMPT_ID") or "")
        self.campaign_fingerprint = str(
            os.environ.get("LSGEMU_CAMPAIGN_FINGERPRINT") or ""
        )
        self.source_tree_sha256 = str(
            os.environ.get("LSGEMU_SOURCE_TREE_SHA256") or ""
        )
        prepared = getattr(self.runner, "prepared", None)
        self.firmware_sha256 = str(
            getattr(prepared, "firmware_sha256", "") or ""
        )
        toolchain = dict(
            getattr(prepared, "toolchain_fingerprint", {}) or {}
        )
        self.toolchain_runtime_fingerprint = str(
            toolchain.get("runtime_fingerprint")
            or os.environ.get("LSGEMU_TOOLCHAIN_RUNTIME_FINGERPRINT")
            or ""
        )
        self.toolchain_identity_hash = str(
            os.environ.get("LSGEMU_TOOLCHAIN_IDENTITY_HASH")
            or toolchain.get("fingerprint")
            or ""
        )
        static_cache = dict(
            getattr(prepared, "static_cache_identity", {}) or {}
        )
        self.static_cache_identity_hash = str(
            static_cache.get("identity_hash") or ""
        )
        valid_bb_set = set(getattr(prepared, "valid_bb_set", set()) or set())
        denominator_bytes = b"".join(
            int(address).to_bytes(4, "little", signed=False)
            for address in sorted(valid_bb_set)
        )
        self.valid_bb_denominator_hash = hashlib.sha256(
            denominator_bytes
        ).hexdigest()
        try:
            self.elapsed_offset_seconds = float(os.environ.get("LSGEMU_PROGRESS_ELAPSED_OFFSET_SECONDS") or 0.0)
        except ValueError:
            self.elapsed_offset_seconds = 0.0
        self._stage = "setup"
        self._lock = threading.Lock()
        self._snapshot_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_thread_id: int | None = None
        self._failure_count = 0
        self._failure_samples: list[dict[str, object]] = []
        self._last_failure: dict[str, object] | None = None
        self._last_successful_snapshot: dict[str, object] | None = None
        self._worker_exit_reason = "not_started"
        self._last_list_valid_covered_bbs = -1
        self._last_console_valid_covered_bbs = -1
        self.console_progress = (
            os.environ.get("LSGEMU_CONSOLE_PROGRESS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.progress_list_mode = os.environ.get("LSGEMU_PROGRESS_COVERAGE_LIST_MODE", "none").strip().lower()
        self.progress_list_interval = max(1, env_int("LSGEMU_PROGRESS_COVERAGE_LIST_INTERVAL", 10))
        self._snapshot_index = 0
        self._last_payload: dict[str, object] = {}
        self._stall_watchdog_event_emitted = False

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = str(stage or "unknown")

    def current_stage(self) -> str:
        with self._lock:
            return self._stage

    def _capture_stall_watchdog_budget_snapshot(self) -> None:
        """Freeze budget bookkeeping at the first sight of the watchdog trigger.

        The capture happens on the telemetry call that observes the trigger
        (monitor interval or per-attempt kernel observe), so it can trail the
        true trigger instant by at most one progress interval.
        """
        watchdog = self.stall_watchdog
        if watchdog is None or self.stall_watchdog_budget_snapshot is not None:
            return
        now = time.time()
        self.stall_watchdog_budget_snapshot = {
            "configured_budget_seconds": self.wallclock_budget_seconds,
            "budget_remaining_seconds": (
                None
                if self.wallclock_deadline is None
                else max(0, int(self.wallclock_deadline - now))
            ),
            "elapsed_seconds_at_capture": round(now - self.started_at, 3),
        }

    def _observe_stall_watchdog(
        self,
        event: str,
        payload: dict[str, object],
        *,
        zero_yield_unit: bool,
    ) -> None:
        """Feed one telemetry point to the stage-level stall watchdog.

        Interval/stage snapshots carry the monitor-computed
        ``evidence_status`` ledger counts (natural / counterfactual_only /
        diagnostic / canonical_unclassified / validated / unclassified), so
        those are passed straight through here; per-attempt observations from
        the exploration kernels only carry the covered count.  Any increase
        on any ledger resets the watchdog's stall clocks.
        """
        watchdog = self.stall_watchdog
        if watchdog is None:
            return
        try:
            covered = payload.get("covered_bbs")
            evidence_status = payload.get("evidence_status")
            watchdog.observe(
                stage=str(payload.get("stage") or self.current_stage()),
                covered_bbs=int(covered) if covered is not None else None,
                zero_yield_unit=zero_yield_unit,
                evidence_ledgers=(
                    evidence_status
                    if isinstance(evidence_status, dict)
                    else None
                ),
            )
            if watchdog.should_stop():
                self._capture_stall_watchdog_budget_snapshot()
            status = watchdog.status()
            payload["stall_watchdog_stall_seconds"] = status["stall_seconds_accrued"]
            payload["stall_watchdog_stop_requested"] = status["stop_requested"]
            truncating = status.get("truncate_stage")
            if truncating:
                payload["stall_watchdog_truncate_stage"] = truncating
        except Exception as exc:
            self._record_failure("stall_watchdog", exc)

    def _maybe_emit_stall_watchdog_stage_events(self) -> None:
        """Write one progress event per stage-level truncation."""
        watchdog = self.stall_watchdog
        if watchdog is None:
            return
        take = getattr(watchdog, "take_unreported_stage_truncations", None)
        if not callable(take):
            return
        try:
            pending = take()
        except Exception as exc:
            self._record_failure("stall_watchdog_stage_events", exc)
            return
        for record in pending:
            self.snapshot_unchanged(
                "stall_watchdog_stage_truncated",
                stage=str(record.get("stage") or self.current_stage()),
                extra={"stall_watchdog_stage_truncation": dict(record)},
            )

    def _maybe_emit_stall_watchdog_event(self) -> None:
        """Write one dedicated progress event when the watchdog has fired."""
        watchdog = self.stall_watchdog
        if watchdog is None or not watchdog.should_stop():
            return
        with self._snapshot_lock:
            if self._stall_watchdog_event_emitted:
                return
            self._stall_watchdog_event_emitted = True
        self._capture_stall_watchdog_budget_snapshot()
        report = watchdog.stop_report() or watchdog.status()
        extra: dict[str, object] = {
            "stop_reason": STALL_WATCHDOG_STOP_REASON,
            "stall_watchdog": report,
        }
        if self.stall_watchdog_budget_snapshot is not None:
            extra.update(self.stall_watchdog_budget_snapshot)
        self.snapshot_unchanged(
            "stall_watchdog_triggered",
            stage=str((report or {}).get("stage") or self.current_stage()),
            extra=extra,
        )

    def _strict_status(
        self,
        *,
        phase_metadata: dict[str, dict[str, object]],
        global_coverage: set[int],
        coverage_by_evidence: dict[str, set[int]],
    ) -> tuple[bool, list[dict[str, str]], dict[str, int]]:
        invalid_counted_phases: list[dict[str, str]] = []
        for phase_name, phase in phase_metadata.items():
            if not isinstance(phase, dict):
                continue
            if phase.get("coverage_counted", True) is False:
                continue
            entry_derivation = str(phase.get("entry_derivation") or "")
            if str(phase_name) == "isr":
                invalid_counted_phases.append({
                    "phase": str(phase_name),
                    "reason": entry_derivation or "cold_vector_handler_direct_entry",
                })
                continue
            if "isr" in str(phase_name) and phase.get("context_snapshot_mode") is False:
                invalid_counted_phases.append({
                    "phase": str(phase_name),
                    "reason": "isr_without_entry_derived_context_snapshot",
                })
        try:
            covered = self.runner.validate_coverage(set(global_coverage))
            evidence_sets = {
                evidence: self.runner.validate_coverage(
                    set(coverage_by_evidence.get(evidence, set()))
                )
                for evidence in ("E0", "E1", "E2", "E3")
            }
            natural = evidence_sets["E0"] | evidence_sets["E1"]
            counterfactual_only = (
                evidence_sets["E2"] | evidence_sets["E3"]
            ) - natural
            classified = set().union(*evidence_sets.values())
            unclassified = covered - classified
            evidence_status = {
                "natural_supported_bbs": len(natural),
                "counterfactual_only_bbs": len(counterfactual_only),
                "unclassified_evidence_bbs": len(unclassified),
            }
        except Exception:
            evidence_status = {
                "natural_supported_bbs": 0,
                "counterfactual_only_bbs": 0,
                "unclassified_evidence_bbs": len(global_coverage),
            }
        # Prefer the runner's canonical binary evidence contract when it is
        # available.  The legacy E0--E3 partition is retained as a fallback
        # for older runner doubles and old progress snapshots.
        canonical = getattr(self.runner, "_canonical_evidence_partitions", None)
        if callable(canonical):
            try:
                canonical_status = canonical()
                validated = set(canonical_status.get("validated", set()) or set())
                diagnostic = set(
                    canonical_status.get(
                        "diagnostic_only",
                        canonical_status.get("diagnostic", set()),
                    )
                    or set()
                )
                unknown = set(canonical_status.get("unclassified", set()) or set())
                evidence_status.update({
                    "validated_replay_bbs": len(validated),
                    "diagnostic_replay_bbs": len(diagnostic),
                    "canonical_unclassified_bbs": len(unknown),
                })
                strict = bool(
                    not invalid_counted_phases
                    and not diagnostic
                    and not unknown
                )
            except Exception:
                strict = bool(
                    not invalid_counted_phases
                    and evidence_status["counterfactual_only_bbs"] == 0
                    and evidence_status["unclassified_evidence_bbs"] == 0
                )
        else:
            strict = bool(
                not invalid_counted_phases
                and evidence_status["counterfactual_only_bbs"] == 0
                and evidence_status["unclassified_evidence_bbs"] == 0
            )
        return strict, invalid_counted_phases, evidence_status

    def _record_failure(self, operation: str, exc: BaseException) -> None:
        with self._snapshot_lock:
            self._failure_count += 1
            failure = {
                "operation": str(operation),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wallclock_time": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(),
                ),
            }
            self._last_failure = dict(failure)
            if len(self._failure_samples) < 16:
                self._failure_samples.append(failure)

    def status(self) -> dict[str, object]:
        with self._snapshot_lock:
            healthy = bool(
                self._failure_count == 0
                and self._worker_exit_reason not in {
                    "join_timeout",
                    "unhandled_exception",
                }
            )
            return {
                "schema": "lsgemu.progress_monitor.v1",
                "healthy": healthy,
                "failure_count": int(self._failure_count),
                "failure_samples": [dict(item) for item in self._failure_samples],
                "last_failure": dict(self._last_failure) if self._last_failure else None,
                "last_successful_snapshot": (
                    dict(self._last_successful_snapshot)
                    if self._last_successful_snapshot
                    else None
                ),
                "worker_alive": bool(self._thread and self._thread.is_alive()),
                "worker_exit_reason": str(self._worker_exit_reason),
                "telemetry_complete": healthy,
                "failure_policy": "continue_execution_and_mark_report",
                "attempt_id": self.attempt_id,
                "campaign_fingerprint": self.campaign_fingerprint,
                "firmware_sha256": self.firmware_sha256,
                "source_tree_sha256": self.source_tree_sha256,
                "toolchain_runtime_fingerprint": (
                    self.toolchain_runtime_fingerprint
                ),
                "toolchain_identity_hash": self.toolchain_identity_hash,
                "static_cache_identity_hash": (
                    self.static_cache_identity_hash
                ),
                "valid_bb_denominator_hash": (
                    self.valid_bb_denominator_hash
                ),
                "output_path": str(self.output_path),
                "native_engine_reads_from_worker": False,
            }

    def snapshot(
        self,
        event: str,
        *,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self._snapshot_lock:
            try:
                return self._snapshot_impl(event, stage=stage, extra=extra)
            except Exception as exc:
                self._record_failure("snapshot", exc)
                return {
                    "event": str(event),
                    "stage": str(stage or self.current_stage()),
                    "telemetry_error": str(exc),
                }

    def _snapshot_impl(
        self,
        event: str,
        *,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        now = time.time()
        runner_state = self.runner.telemetry_state_snapshot()
        global_coverage = set(runner_state.get("global_coverage", set()) or set())
        coverage_by_evidence = {
            str(evidence): set(covered)
            for evidence, covered in (
                runner_state.get("coverage_by_evidence", {}) or {}
            ).items()
        }
        phase_metadata = dict(runner_state.get("phase_metadata", {}) or {})
        covered_bbs = len(global_coverage)
        valid_total_bbs = int(self.runner.prepared.valid_total_bbs)
        valid_covered_bbs = (
            len(self.runner.prepared.valid_coverage(global_coverage))
            if valid_total_bbs > 0
            else 0
        )
        active_observed_bbs = 0
        active_observed_valid_bbs = 0
        try:
            active_set = set(
                getattr(getattr(self.runner, "emulator", None), "bb_addr_set", set())
                or set()
            )
            active_coverage = self.runner.validate_coverage(active_set)
            active_observed_bbs = len(active_coverage)
            active_observed_valid_bbs = (
                len(self.runner.prepared.valid_coverage(active_coverage))
                if valid_total_bbs > 0
                else 0
            )
        except Exception:
            active_observed_bbs = 0
            active_observed_valid_bbs = 0
        try:
            reachability_summary = self.runner.static_reachability_summary(
                coverage=global_coverage
            )
        except Exception as exc:
            reachability_summary = {}
            self._record_failure("static_reachability_summary", exc)
        valid_entry_plus_vector_reachable_bbs = int(
            reachability_summary.get("valid_entry_plus_vector_static_reachable_bbs", 0) or 0
        )
        valid_entry_plus_vector_reachable_covered_bbs = int(
            reachability_summary.get("valid_entry_plus_vector_static_reachable_covered_bbs", 0) or 0
        )
        valid_entry_static_reachable_bbs = int(
            reachability_summary.get("valid_entry_static_reachable_bbs", 0) or 0
        )
        valid_entry_static_reachable_covered_bbs = int(
            reachability_summary.get("valid_entry_static_reachable_covered_bbs", 0) or 0
        )
        valid_extended_static_reachable_bbs = int(
            reachability_summary.get("valid_extended_static_reachable_bbs", 0) or 0
        )
        valid_extended_static_reachable_covered_bbs = int(
            reachability_summary.get("valid_extended_static_reachable_covered_bbs", 0) or 0
        )
        valid_static_unreachable_from_extended_roots = int(
            reachability_summary.get("valid_static_unreachable_from_extended_roots", 0) or 0
        )
        extended_static_roots_available = bool(
            reachability_summary.get("extended_static_roots_available", False)
        )
        strict_real_entry_replayable, invalid_counted_phases, evidence_status = self._strict_status(
            phase_metadata=phase_metadata,
            global_coverage=global_coverage,
            coverage_by_evidence=coverage_by_evidence,
        )
        phase_names = list(phase_metadata.keys())
        counted_phase_names = [
            str(name)
            for name, meta in phase_metadata.items()
            if isinstance(meta, dict) and meta.get("coverage_counted", True) is not False
        ]
        payload: dict[str, object] = {
            "event": str(event),
            "stage": str(stage or self.current_stage()),
            "firmware": str(self.firmware),
            "attempt_id": self.attempt_id,
            "campaign_fingerprint": self.campaign_fingerprint,
            "firmware_sha256": self.firmware_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "toolchain_runtime_fingerprint": (
                self.toolchain_runtime_fingerprint
            ),
            "toolchain_identity_hash": self.toolchain_identity_hash,
            "static_cache_identity_hash": (
                self.static_cache_identity_hash
            ),
            "valid_bb_denominator_hash": (
                self.valid_bb_denominator_hash
            ),
            "wallclock_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "elapsed_seconds": round(now - self.started_at + self.elapsed_offset_seconds, 3),
            "segment_elapsed_seconds": round(now - self.started_at, 3),
            "elapsed_offset_seconds": self.elapsed_offset_seconds,
            "covered_bbs": covered_bbs,
            "total_bbs": int(self.runner.prepared.total_bbs),
            "coverage_rate": (
                covered_bbs / self.runner.prepared.total_bbs * 100.0
                if self.runner.prepared.total_bbs
                else 0.0
            ),
            "valid_covered_bbs": valid_covered_bbs,
            "valid_total_bbs": valid_total_bbs,
            "valid_coverage_rate": (
                valid_covered_bbs / valid_total_bbs * 100.0
                if valid_total_bbs
                else 0.0
            ),
            "active_emulator_observed_bbs": active_observed_bbs,
            "active_emulator_observed_valid_bbs": active_observed_valid_bbs,
            "active_emulator_note": (
                "diagnostic_only_not_counted_until_phase_end"
                if active_observed_bbs
                else "no_live_observed_bbs"
            ),
            "valid_entry_plus_vector_static_reachable_bbs": valid_entry_plus_vector_reachable_bbs,
            "valid_entry_plus_vector_static_reachable_covered_bbs": valid_entry_plus_vector_reachable_covered_bbs,
            "valid_entry_plus_vector_reachable_coverage_rate": (
                valid_entry_plus_vector_reachable_covered_bbs
                / valid_entry_plus_vector_reachable_bbs
                * 100.0
                if valid_entry_plus_vector_reachable_bbs
                else 0.0
            ),
            "valid_entry_static_reachable_bbs": valid_entry_static_reachable_bbs,
            "valid_entry_static_reachable_covered_bbs": valid_entry_static_reachable_covered_bbs,
            "valid_entry_static_reachable_coverage_rate": (
                valid_entry_static_reachable_covered_bbs
                / valid_entry_static_reachable_bbs
                * 100.0
                if valid_entry_static_reachable_bbs
                else 0.0
            ),
            "extended_static_roots_available": extended_static_roots_available,
            "valid_extended_static_reachable_bbs": valid_extended_static_reachable_bbs,
            "valid_extended_static_reachable_covered_bbs": valid_extended_static_reachable_covered_bbs,
            "valid_extended_static_reachable_uncovered_bbs": max(
                0,
                valid_extended_static_reachable_bbs - valid_extended_static_reachable_covered_bbs,
            ),
            "valid_extended_static_reachable_coverage_rate": (
                valid_extended_static_reachable_covered_bbs
                / valid_extended_static_reachable_bbs
                * 100.0
                if valid_extended_static_reachable_bbs
                else 0.0
            ),
            "valid_static_unreachable_from_extended_roots": valid_static_unreachable_from_extended_roots,
            "extended_static_unreachable_valid_ratio": (
                valid_static_unreachable_from_extended_roots
                / valid_total_bbs
                * 100.0
                if valid_total_bbs
                else 0.0
            ),
            "phase_count": len(phase_names),
            "counted_phase_count": len(counted_phase_names),
            "last_phase": phase_names[-1] if phase_names else None,
            "strict_real_entry_replayable": strict_real_entry_replayable,
            "invalid_counted_phase_count": len(invalid_counted_phases),
            "evidence_status": evidence_status,
            "run_profile_hash": str((runner_state.get("run_profile", {}) or {}).get("profile_hash") or ""),
        }
        if runner_state.get("phase_metadata_copy_error"):
            payload["phase_metadata_copy_error"] = runner_state["phase_metadata_copy_error"]
        try:
            payload["scheduler_feedback"] = self.runner.scheduler_feedback_summary_for_progress()
        except Exception as exc:
            payload["scheduler_feedback_error"] = str(exc)
        if threading.get_ident() == self._worker_thread_id:
            payload["current_replay_state_fingerprint"] = {
                "available": False,
                "reason": "background_native_engine_access_disabled",
            }
        else:
            try:
                payload["current_replay_state_fingerprint"] = self.runner.current_replay_state_fingerprint_summary()
            except Exception as exc:
                payload["current_replay_state_fingerprint_error"] = str(exc)
        if extra:
            payload.update(extra)
        self._snapshot_index += 1
        include_coverage_list = self._should_include_coverage_list(
            event=str(event),
            valid_covered_bbs=valid_covered_bbs,
            extra=extra or {},
        )
        if include_coverage_list:
            try:
                covered_list = sorted(self.runner.validate_coverage(global_coverage))
                valid_covered_list = sorted(self.runner.prepared.valid_coverage(set(covered_list)))
                payload.update({
                    "covered_bb_list": covered_list,
                    "covered_valid_bb_list": valid_covered_list,
                    "covered_bb_list_complete": True,
                    "covered_valid_bb_list_complete": True,
                })
                self._last_list_valid_covered_bbs = max(
                    self._last_list_valid_covered_bbs,
                    valid_covered_bbs,
                )
            except Exception as exc:
                payload.update({
                    "covered_bb_list_complete": False,
                    "covered_valid_bb_list_complete": False,
                    "coverage_list_error": str(exc),
                })
        self._observe_stall_watchdog(
            str(event),
            payload,
            zero_yield_unit=str(event) == "interval",
        )
        append_jsonl(self.output_path, payload)
        self._last_successful_snapshot = {
            "event": str(payload.get("event") or event),
            "stage": str(payload.get("stage") or stage or self.current_stage()),
            "elapsed_seconds": float(payload.get("elapsed_seconds") or 0.0),
            "snapshot_index": int(self._snapshot_index),
        }
        self._maybe_print_console_progress(payload)
        self._last_payload = dict(payload)
        return payload

    def snapshot_unchanged(
        self,
        event: str,
        *,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self._snapshot_lock:
            try:
                return self._snapshot_unchanged_impl(event, stage=stage, extra=extra)
            except Exception as exc:
                self._record_failure("snapshot_unchanged", exc)
                return {
                    "event": str(event),
                    "stage": str(stage or self.current_stage()),
                    "telemetry_error": str(exc),
                }

    def _snapshot_unchanged_impl(
        self,
        event: str,
        *,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Record metadata-only transitions without recomputing diagnostics."""
        if not self._last_payload:
            return self.snapshot(event, stage=stage, extra=extra)

        now = time.time()
        stable_keys = {
            "firmware",
            "attempt_id",
            "campaign_fingerprint",
            "firmware_sha256",
            "source_tree_sha256",
            "toolchain_runtime_fingerprint",
            "toolchain_identity_hash",
            "static_cache_identity_hash",
            "valid_bb_denominator_hash",
            "elapsed_offset_seconds",
            "covered_bbs",
            "total_bbs",
            "coverage_rate",
            "valid_covered_bbs",
            "valid_total_bbs",
            "valid_coverage_rate",
            "active_emulator_observed_bbs",
            "active_emulator_observed_valid_bbs",
            "active_emulator_note",
            "valid_entry_plus_vector_static_reachable_bbs",
            "valid_entry_plus_vector_static_reachable_covered_bbs",
            "valid_entry_plus_vector_reachable_coverage_rate",
            "valid_entry_static_reachable_bbs",
            "valid_entry_static_reachable_covered_bbs",
            "valid_entry_static_reachable_coverage_rate",
            "extended_static_roots_available",
            "valid_extended_static_reachable_bbs",
            "valid_extended_static_reachable_covered_bbs",
            "valid_extended_static_reachable_uncovered_bbs",
            "valid_extended_static_reachable_coverage_rate",
            "valid_static_unreachable_from_extended_roots",
            "extended_static_unreachable_valid_ratio",
            "strict_real_entry_replayable",
            "invalid_counted_phase_count",
            "evidence_status",
            "run_profile_hash",
            "scheduler_feedback",
            "scheduler_feedback_error",
            "current_replay_state_fingerprint",
            "current_replay_state_fingerprint_error",
        }
        payload = {
            key: value
            for key, value in self._last_payload.items()
            if key in stable_keys
        }
        runner_state = self.runner.telemetry_state_snapshot()
        phase_metadata = dict(runner_state.get("phase_metadata", {}) or {})
        phase_names = list(phase_metadata.keys())
        counted_phase_names = [
            str(name)
            for name, meta in phase_metadata.items()
            if isinstance(meta, dict)
            and meta.get("coverage_counted", True) is not False
        ]
        payload.update({
            "event": str(event),
            "stage": str(stage or self.current_stage()),
            "wallclock_time": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(now),
            ),
            "elapsed_seconds": round(
                now - self.started_at + self.elapsed_offset_seconds,
                3,
            ),
            "segment_elapsed_seconds": round(now - self.started_at, 3),
            "phase_count": len(phase_names),
            "counted_phase_count": len(counted_phase_names),
            "last_phase": phase_names[-1] if phase_names else None,
        })
        if extra:
            payload.update(extra)
        self._snapshot_index += 1
        self._observe_stall_watchdog(
            str(event),
            payload,
            zero_yield_unit=False,
        )
        append_jsonl(self.output_path, payload)
        self._last_successful_snapshot = {
            "event": str(payload.get("event") or event),
            "stage": str(payload.get("stage") or stage or self.current_stage()),
            "elapsed_seconds": float(payload.get("elapsed_seconds") or 0.0),
            "snapshot_index": int(self._snapshot_index),
        }
        self._maybe_print_console_progress(payload)
        self._last_payload = dict(payload)
        return payload

    def _should_include_coverage_list(
        self,
        *,
        event: str,
        valid_covered_bbs: int,
        extra: dict[str, object],
    ) -> bool:
        if bool(extra.get("force_coverage_list")):
            return True
        mode = self.progress_list_mode
        if mode in {"0", "false", "no", "off", "none", "count", "counts", "counts_only"}:
            return False
        if mode in {"all", "always", "full"}:
            return True
        if mode in {"final", "final_only"}:
            return event in {"run_complete", "run_failed", "finalizing_report"}
        if mode in {"on_change", "change", "changed"}:
            return valid_covered_bbs > self._last_list_valid_covered_bbs
        if mode in {"sample", "sampled", "periodic"}:
            return self._snapshot_index % self.progress_list_interval == 0
        return False

    def _maybe_print_console_progress(self, payload: dict[str, object]) -> None:
        if not self.console_progress:
            return
        event = str(payload.get("event") or "")
        valid_covered = int(payload.get("valid_covered_bbs") or 0)
        important_event = event in {
            "run_start",
            "interval",
            "stage_end",
            "finalizing_report",
            "run_complete",
            "run_failed",
        }
        if not important_event and valid_covered <= self._last_console_valid_covered_bbs:
            return
        self._last_console_valid_covered_bbs = max(
            self._last_console_valid_covered_bbs,
            valid_covered,
        )
        elapsed = float(payload.get("elapsed_seconds") or 0.0)
        valid_total = int(payload.get("valid_total_bbs") or 0)
        valid_rate = float(payload.get("valid_coverage_rate") or 0.0)
        covered = int(payload.get("covered_bbs") or 0)
        total = int(payload.get("total_bbs") or 0)
        stage = str(payload.get("stage") or "")
        strict = "strict" if payload.get("strict_real_entry_replayable") else "mixed"
        active_valid = int(payload.get("active_emulator_observed_valid_bbs") or 0)
        active_text = ""
        if active_valid > valid_covered:
            active_text = f" active={active_valid}/{valid_total}"
        print(
            "[LSGEmu] "
            f"t={elapsed:7.1f}s event={event} stage={stage} "
            f"valid={valid_covered}/{valid_total} ({valid_rate:.2f}%) "
            f"all={covered}/{total}{active_text} mode={strict}",
            flush=True,
        )

    def start(self) -> None:
        self._worker_exit_reason = "running" if self.interval_seconds > 0 else "disabled"
        self.snapshot("run_start", stage=self.current_stage())
        if self.interval_seconds <= 0:
            return
        self._thread = threading.Thread(
            target=self._worker,
            name="lsgemu-progress-monitor",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        self._worker_thread_id = threading.get_ident()
        next_emit = self.started_at + self.interval_seconds
        try:
            while not self._stop_event.is_set():
                wait_seconds = max(0.0, next_emit - time.time())
                if self._stop_event.wait(wait_seconds):
                    self._worker_exit_reason = "stop_requested"
                    break
                try:
                    self.snapshot("interval", stage=self.current_stage())
                except Exception as exc:
                    with self._snapshot_lock:
                        self._record_failure("worker", exc)
                self._maybe_emit_stall_watchdog_event()
                self._maybe_emit_stall_watchdog_stage_events()
                next_emit += self.interval_seconds
        except BaseException as exc:
            with self._snapshot_lock:
                self._record_failure("worker_unhandled", exc)
                self._worker_exit_reason = "unhandled_exception"
        else:
            if self._stop_event.is_set() and self._worker_exit_reason == "running":
                self._worker_exit_reason = "stop_requested"

    def stop(self, final_event: str, *, extra: dict[str, object] | None = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._worker_exit_reason = "join_timeout"
                self._record_failure(
                    "worker_join",
                    TimeoutError("progress monitor did not stop within 2 seconds"),
                )
        final_extra = dict(extra or {})
        final_extra["progress_monitor"] = self.status()
        self._maybe_emit_stall_watchdog_event()
        self._maybe_emit_stall_watchdog_stage_events()
        if self.stall_watchdog is not None:
            watchdog_report = self.stall_watchdog.stop_report()
            if watchdog_report is not None:
                final_extra["stop_reason"] = STALL_WATCHDOG_STOP_REASON
                final_extra["stall_watchdog"] = watchdog_report
        self.snapshot(final_event, stage=self.current_stage(), extra=final_extra)


def parse_target_bbs_discovered(phase_meta: dict[str, object]) -> set[int]:
    parsed: set[int] = set()
    for item in phase_meta.get("target_bbs_discovered_list", []) or []:
        try:
            if isinstance(item, str):
                parsed.add(int(item, 16) & ~1)
            else:
                parsed.add(int(item) & ~1)
        except (TypeError, ValueError):
            continue
    return parsed


def combined_phase_metadata(runner: HistoricalRunner, phase_name: str) -> dict[str, object]:
    combined = dict(runner.phase_metadata.get(phase_name, {}) or {})
    direct = runner.phase_metadata.get(f"{phase_name}_direct_call", {}) or {}
    if not direct:
        return combined

    for key in (
        "new_bbs",
        "target_bbs_discovered",
        "remembered_branch_events",
        "remembered_branch_root_snapshots",
        "dynamic_successor_edges_added",
        "priority_root_tasks_seeded",
    ):
        combined[key] = int(combined.get(key, 0) or 0) + int(direct.get(key, 0) or 0)
    combined["target_hit_tasks"] = (
        int(combined.get("target_hit_tasks", 0) or 0)
        + int(direct.get("tasks_with_target_bbs", 0) or 0)
    )

    discovered = parse_target_bbs_discovered(combined) | parse_target_bbs_discovered(direct)
    combined["target_bbs_discovered_list"] = [
        f"0x{bb:08x}" for bb in sorted(discovered)[:256]
    ]
    combined["direct_call_subphase"] = {
        "phase": f"{phase_name}_direct_call",
        "new_bbs": int(direct.get("new_bbs", 0) or 0),
        "target_bbs_discovered": int(direct.get("target_bbs_discovered", 0) or 0),
        "tasks_run": int(direct.get("tasks_run", 0) or 0),
        "tasks_with_target_bbs": int(direct.get("tasks_with_target_bbs", 0) or 0),
    }
    return combined


def parse_unstarted_frontier_targets(
    uncovered_summary: dict[str, object],
    limit: int,
    *,
    switch_only: bool = False,
) -> list[int]:
    predecessor_targets: list[list[int]] = []
    for item in uncovered_summary.get("top_frontier_predecessors", []) or []:
        if item.get("root_started", False):
            continue
        if not item.get("is_branch_frontier", True):
            continue
        if switch_only and not item.get("is_switch_frontier", False):
            continue
        parsed_row: list[int] = []
        seen_row: set[int] = set()
        for raw in item.get("sample_uncovered_successors", []) or []:
            try:
                bb = int(raw, 16) & ~1
            except (TypeError, ValueError):
                continue
            if bb in seen_row:
                continue
            seen_row.add(bb)
            parsed_row.append(bb)
        if parsed_row:
            predecessor_targets.append(parsed_row)

    targets: list[int] = []
    seen: set[int] = set()
    row_index = 0
    while predecessor_targets:
        row = predecessor_targets[row_index]
        bb = row.pop(0)
        if bb not in seen:
            seen.add(bb)
            targets.append(bb)
            if limit > 0 and len(targets) >= limit:
                return targets
        if not row:
            predecessor_targets.pop(row_index)
            if not predecessor_targets:
                break
            row_index %= len(predecessor_targets)
        else:
            row_index = (row_index + 1) % len(predecessor_targets)
    return targets


def select_round_targets(
    candidate_targets: list[int],
    round_index: int,
    limit: int,
    attempted_rounds: dict[int, int],
    resolved_targets: set[int],
    cooldown_rounds: int,
) -> tuple[list[int], dict[str, int]]:
    pending_targets = [bb for bb in candidate_targets if bb not in resolved_targets]
    effective_round_index = round_index + segment_schedule_round_offset()
    ordered_targets = rotate_target_order(
        pending_targets,
        effective_round_index,
        limit if limit > 0 else max(1, len(pending_targets)),
    )
    if cooldown_rounds <= 0:
        selected_targets = ordered_targets[:limit] if limit > 0 else list(ordered_targets)
        return selected_targets, {
            "raw_candidates": len(candidate_targets),
            "pending_candidates": len(pending_targets),
            "preferred_candidates": len(ordered_targets),
            "cooled_candidates": 0,
        }

    preferred_targets = []
    cooled_targets = []
    effective_cooldown = max(1, cooldown_rounds)
    for bb in ordered_targets:
        last_attempt_round = attempted_rounds.get(bb)
        if last_attempt_round is None or round_index - last_attempt_round >= effective_cooldown:
            preferred_targets.append(bb)
        else:
            cooled_targets.append(bb)

    if limit > 0:
        selected_targets = list(preferred_targets[:limit])
        if len(selected_targets) < limit:
            selected_targets.extend(cooled_targets[: limit - len(selected_targets)])
    else:
        selected_targets = list(preferred_targets) + list(cooled_targets)

    return selected_targets, {
        "raw_candidates": len(candidate_targets),
        "pending_candidates": len(pending_targets),
        "preferred_candidates": len(preferred_targets),
        "cooled_candidates": len(cooled_targets),
    }


def run_targeted_stage(
    runner: HistoricalRunner,
    stage_name: str,
    requested_rounds: int,
    auto_max_rounds: int,
    stale_round_limit: int,
    min_progress_bbs: int,
    max_targets: int,
    cooldown_rounds: int,
    candidate_supplier,
    executor,
    should_continue=None,
) -> tuple[set[int], list[dict[str, object]]]:
    stage_coverage: set[int] = set()
    round_records: list[dict[str, object]] = []
    attempted_rounds: dict[int, int] = {}
    resolved_targets: set[int] = set()
    stale_rounds = 0

    total_rounds = auto_round_limit(requested_rounds, auto_max_rounds)
    for round_index in range(total_rounds):
        if should_continue is not None and not should_continue():
            break
        candidate_targets = list(candidate_supplier() or [])
        selected_targets, selection_stats = select_round_targets(
            candidate_targets=candidate_targets,
            round_index=round_index,
            limit=max_targets,
            attempted_rounds=attempted_rounds,
            resolved_targets=resolved_targets,
            cooldown_rounds=cooldown_rounds,
        )
        if not selected_targets:
            break

        phase_name = stage_name if round_index == 0 else f"{stage_name}_round_{round_index + 1}"
        round_target_set = set(selected_targets)
        round_coverage = executor(phase_name, round_target_set)
        stage_coverage.update(round_coverage)
        phase_meta = combined_phase_metadata(runner, phase_name)
        discovered_targets = parse_target_bbs_discovered(phase_meta)
        if discovered_targets:
            resolved_targets.update(discovered_targets)
        for bb in round_target_set:
            attempted_rounds[bb] = round_index

        new_bbs = int(phase_meta.get("new_bbs", 0) or 0)
        target_hit_tasks = int(phase_meta.get("target_hit_tasks", 0) or 0)
        target_bbs_discovered = int(phase_meta.get("target_bbs_discovered", 0) or 0)
        remembered_branch_events = int(phase_meta.get("remembered_branch_events", 0) or 0)
        remembered_snapshot_variants = int(phase_meta.get("remembered_branch_root_snapshots", 0) or 0)
        dynamic_successor_edges = int(phase_meta.get("dynamic_successor_edges_added", 0) or 0)

        round_records.append({
            "round": round_index + 1,
            "phase": phase_name,
            "candidate_targets": len(candidate_targets),
            "pending_targets": selection_stats["pending_candidates"],
            "preferred_targets": selection_stats["preferred_candidates"],
            "cooled_targets": selection_stats["cooled_candidates"],
            "selected_targets": len(round_target_set),
            "covered_bbs": len(round_coverage),
            "new_bbs": new_bbs,
            "target_hit_tasks": target_hit_tasks,
            "target_bbs_discovered": target_bbs_discovered,
            "remembered_branch_events": remembered_branch_events,
            "remembered_branch_root_snapshots": remembered_snapshot_variants,
            "dynamic_successor_edges_added": dynamic_successor_edges,
            "resolved_targets": len(discovered_targets),
            "resolved_total": len(resolved_targets),
            "selected_target_sample": [f"0x{bb:08x}" for bb in sorted(round_target_set)[:16]],
            "resolved_target_sample": [f"0x{bb:08x}" for bb in sorted(discovered_targets)[:16]],
        })

        progress_threshold = max(1, min_progress_bbs)
        if (
            new_bbs < progress_threshold
            and target_bbs_discovered < progress_threshold
            and remembered_branch_events <= 0
            and remembered_snapshot_variants <= 0
            and dynamic_successor_edges <= 0
        ):
            stale_rounds += 1
        else:
            stale_rounds = 0
        if stale_rounds >= max(1, stale_round_limit):
            break

    return stage_coverage, round_records


def empty_stage_summary() -> dict[str, object]:
    return {
        "coverage": set(),
        "initial_targets": set(),
        "remaining_targets": set(),
        "rounds_executed": 0,
        "new_bbs": 0,
        "target_hit_tasks": 0,
        "target_bbs_discovered": 0,
        "new_candidate_targets": 0,
        "remembered_branch_events": 0,
        "remembered_branch_root_snapshots": 0,
        "priority_root_tasks_seeded": 0,
        "dynamic_successor_edges_added": 0,
        "last_round_new_bbs": 0,
        "last_round_target_bbs_discovered": 0,
        "last_round_new_candidate_targets": 0,
        "last_round_branch_events": 0,
        "last_round_snapshot_variants": 0,
        "last_round_priority_root_tasks_seeded": 0,
        "last_round_dynamic_successor_edges_added": 0,
    }


def parse_stage_strategy(name: str, text: str) -> dict[str, bool]:
    strategy: dict[str, bool] = {}
    normalized = str(text or "").strip().lower()
    if not normalized:
        raise ValueError(f"{name} strategy must not be empty")
    for item in normalized.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"{name} strategy token '{token}' must use key=value format"
            )
        key, raw_value = token.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip().lower()
        if key not in {"targeted_direct_root_focus", "prefer_direct_root_snapshot"}:
            raise ValueError(f"{name} strategy key '{key}' is not supported")
        if raw_value in {"1", "true", "yes", "on"}:
            strategy[key] = True
        elif raw_value in {"0", "false", "no", "off"}:
            strategy[key] = False
        else:
            raise ValueError(
                f"{name} strategy value '{raw_value}' for key '{key}' is not a boolean"
            )
    for key in ("targeted_direct_root_focus", "prefer_direct_root_snapshot"):
        if key not in strategy:
            raise ValueError(f"{name} strategy missing required key '{key}'")
    return strategy


def targeted_phase_name(stage_name: str, cycle_index: int, round_index: int) -> str:
    if cycle_index == 0:
        if round_index == 0:
            return stage_name
        return f"{stage_name}_round_{round_index + 1}"
    cycle_name = f"{stage_name}_cycle_{cycle_index + 1}"
    if round_index == 0:
        return cycle_name
    return f"{cycle_name}_round_{round_index + 1}"


def run_frontier_reservoir_stage(
    *,
    runner: HistoricalRunner,
    stage_name: str,
    cycle_index: int,
    round_seconds: int,
    requested_rounds: int,
    auto_max_rounds: int,
    stale_round_limit: int,
    min_progress_bbs: int,
    max_targets: int,
    cooldown_rounds: int,
    candidate_supplier,
    executor,
    round_records: list[dict[str, object]],
    coverage_accumulator: set[int],
    should_continue=None,
    refresh_remaining_targets: bool = True,
) -> dict[str, object]:
    summary = empty_stage_summary()
    attempted_rounds: dict[int, int] = {}
    resolved_targets: set[int] = set()
    seen_targets: set[int] = set()
    stale_rounds = 0

    def should_run_round(round_index: int) -> bool:
        if should_continue is None:
            return True
        try:
            return bool(should_continue(round_index))
        except TypeError:
            return bool(should_continue())

    total_rounds = auto_round_limit(requested_rounds, auto_max_rounds)
    for round_index in range(total_rounds):
        if not should_run_round(round_index):
            break
        candidate_targets = list(candidate_supplier() or [])
        if round_index == 0:
            summary["initial_targets"] = set(candidate_targets)
        seen_targets.update(candidate_targets)
        selected_targets, selection_stats = select_round_targets(
            candidate_targets=candidate_targets,
            round_index=round_index,
            limit=max_targets,
            attempted_rounds=attempted_rounds,
            resolved_targets=resolved_targets,
            cooldown_rounds=cooldown_rounds,
        )
        if not selected_targets:
            break

        phase_name = targeted_phase_name(stage_name, cycle_index, round_index)
        round_target_set = set(selected_targets)
        round_coverage = executor(phase_name, round_target_set)
        coverage_accumulator.update(round_coverage)
        summary["coverage"].update(round_coverage)
        phase_meta = combined_phase_metadata(runner, phase_name)
        discovered_targets = parse_target_bbs_discovered(phase_meta)
        if discovered_targets:
            resolved_targets.update(discovered_targets)
        for bb in round_target_set:
            attempted_rounds[bb] = round_index

        next_targets = set()
        if round_index + 1 < total_rounds and should_run_round(round_index + 1):
            next_targets = set(candidate_supplier() or [])
        new_candidate_targets = len(next_targets - seen_targets)
        seen_targets.update(next_targets)

        new_bbs = int(phase_meta.get("new_bbs", 0) or 0)
        target_hit_tasks = int(phase_meta.get("target_hit_tasks", 0) or 0)
        target_bbs_discovered = int(phase_meta.get("target_bbs_discovered", 0) or 0)
        remembered_branch_events = int(phase_meta.get("remembered_branch_events", 0) or 0)
        remembered_snapshot_variants = int(phase_meta.get("remembered_branch_root_snapshots", 0) or 0)
        priority_root_tasks_seeded = int(phase_meta.get("priority_root_tasks_seeded", 0) or 0)
        dynamic_successor_edges = int(phase_meta.get("dynamic_successor_edges_added", 0) or 0)

        summary["rounds_executed"] += 1
        summary["new_bbs"] += new_bbs
        summary["target_hit_tasks"] += target_hit_tasks
        summary["target_bbs_discovered"] += target_bbs_discovered
        summary["new_candidate_targets"] += new_candidate_targets
        summary["remembered_branch_events"] += remembered_branch_events
        summary["remembered_branch_root_snapshots"] += remembered_snapshot_variants
        summary["priority_root_tasks_seeded"] += priority_root_tasks_seeded
        summary["dynamic_successor_edges_added"] += dynamic_successor_edges
        summary["last_round_new_bbs"] = new_bbs
        summary["last_round_target_bbs_discovered"] = target_bbs_discovered
        summary["last_round_new_candidate_targets"] = new_candidate_targets
        summary["last_round_branch_events"] = remembered_branch_events
        summary["last_round_snapshot_variants"] = remembered_snapshot_variants
        summary["last_round_priority_root_tasks_seeded"] = priority_root_tasks_seeded
        summary["last_round_dynamic_successor_edges_added"] = dynamic_successor_edges

        round_records.append({
            "cycle": cycle_index + 1,
            "round": round_index + 1,
            "phase": phase_name,
            "candidate_targets": len(candidate_targets),
            "pending_targets": selection_stats["pending_candidates"],
            "preferred_targets": selection_stats["preferred_candidates"],
            "cooled_targets": selection_stats["cooled_candidates"],
            "selected_targets": len(round_target_set),
            "covered_bbs": len(round_coverage),
            "new_bbs": new_bbs,
            "target_hit_tasks": target_hit_tasks,
            "target_bbs_discovered": target_bbs_discovered,
            "new_candidate_targets": new_candidate_targets,
            "remembered_branch_events": remembered_branch_events,
            "remembered_branch_root_snapshots": remembered_snapshot_variants,
            "priority_root_tasks_seeded": priority_root_tasks_seeded,
            "dynamic_successor_edges_added": dynamic_successor_edges,
            "resolved_targets": len(discovered_targets),
            "resolved_total": len(resolved_targets),
            "selected_target_sample": [f"0x{bb:08x}" for bb in sorted(round_target_set)[:16]],
            "resolved_target_sample": [f"0x{bb:08x}" for bb in sorted(discovered_targets)[:16]],
        })

        progress_threshold = max(1, min_progress_bbs)
        stage_is_dispatch_heavy = (
            stage_name.startswith("switch_frontier_targeted")
            or stage_name.startswith("frontier_targeted")
        )
        no_counted_progress = (
            new_bbs < progress_threshold
            and target_bbs_discovered < progress_threshold
            and new_candidate_targets < progress_threshold
        )
        has_structural_metadata = (
            remembered_branch_events > 0
            or remembered_snapshot_variants > 0
            or priority_root_tasks_seeded > 0
        )
        has_convertible_dynamic_edges = dynamic_successor_edges > 0
        metadata_only_progress = (
            no_counted_progress
            and has_structural_metadata
            and not has_convertible_dynamic_edges
        )
        no_metadata_progress = (
            remembered_branch_events <= 0
            and remembered_snapshot_variants <= 0
            and priority_root_tasks_seeded <= 0
            and dynamic_successor_edges <= 0
        )
        stale_round = no_counted_progress and (
            no_metadata_progress
            or (stage_is_dispatch_heavy and metadata_only_progress)
        )
        if stale_round:
            stale_rounds += 1
        else:
            stale_rounds = 0
        if stale_rounds >= max(1, stale_round_limit):
            break

    if refresh_remaining_targets:
        summary["remaining_targets"] = set(candidate_supplier() or [])
    else:
        summary["remaining_targets"] = set(seen_targets) - set(resolved_targets)
    return summary


def main():
    profile_preparser = argparse.ArgumentParser(add_help=False)
    profile_preparser.add_argument("--run-profile", default=None)
    profile_preparser.add_argument("--config", default=None)
    profile_args, _ = profile_preparser.parse_known_args()
    loaded_run_profile = None
    applied_run_profile_env = {}
    if profile_args.run_profile:
        loaded_run_profile = load_run_profile(profile_args.run_profile)
        applied_run_profile_env = apply_run_profile_environment(loaded_run_profile)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-profile",
        default=profile_args.run_profile,
        help="YAML/JSON profile containing LSGEMU_* environment switches for reproducible runs.",
    )
    parser.add_argument("--config", default=profile_args.config, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument("--firmware", default=str(DEFAULT_REAL_GATEWAY))
    parser.add_argument(
        "--firmware-input-provenance-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execution-mode-override",
        choices=["arm", "thumb"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            configured_path(
                "LSGEMU_RUN_OUTPUT_DIR",
                configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4") / ".lsgemu_runs",
            )
            / "gateway_interleaved_cached"
        ),
    )
    parser.add_argument(
        "--entry-point",
        type=parse_int,
        default=None,
        help="Override the analyzed runtime entry point, e.g. 0x01021060 for a raw BIN vendor container.",
    )
    parser.add_argument(
        "--load-base",
        type=parse_int,
        default=None,
        help="Map raw BIN bytes at this runtime base in addition to 0x0, e.g. 0x01021000.",
    )
    parser.add_argument("--constraint-file", default=None)
    parser.add_argument("--clean-constraint-file", action="store_true")
    parser.add_argument("--constraint-seed-mode", choices=["none", "best", "union"], default="none")
    parser.add_argument("--constraint-seed-activation", choices=["initial", "after-main"], default="after-main")
    parser.add_argument("--constraint-seed-root", action="append", default=[])
    parser.add_argument("--baseline-instructions", type=int, default=500000)
    parser.add_argument(
        "--baseline-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional wallclock cap for the initial entry run. A non-positive "
            "value means no explicit cap; short campaigns set an adaptive cap "
            "so entry loops cannot consume the entire exploration budget."
        ),
    )
    parser.add_argument("--baseline-quiescence-bbs", type=int, default=8192)
    parser.add_argument(
        "--enable-baseline-function-summaries",
        action="store_true",
        help="Install safe function summaries during the initial entry run. Disabled by default to keep baseline cheap.",
    )
    parser.add_argument("--semantic-entry-prefix-seconds", type=int, default=20)
    parser.add_argument("--semantic-entry-prefix-instructions", type=int, default=180000)
    parser.add_argument("--semantic-entry-prefix-quiescence-bbs", type=int, default=8192)
    parser.add_argument("--isr-instructions", type=int, default=10000)
    parser.add_argument(
        "--skip-cold-isr",
        action="store_true",
        help="Skip cold vector-handler probing. Strict contextual ISR/replay stages can still be enabled separately.",
    )
    parser.add_argument(
        "--count-cold-isr-coverage",
        action="store_true",
        help=(
            "Count cold vector-handler ISR probing as coverage. Disabled by "
            "default for strict real-entry replayability; contextual ISR "
            "coverage is still counted because it starts from entry-derived "
            "dynamic states."
        ),
    )
    parser.add_argument("--contextual-isr-contexts", type=int, default=8)
    parser.add_argument("--contextual-isr-max-isrs", type=int, default=16)
    parser.add_argument("--contextual-isr-instructions", type=int, default=5000)
    parser.add_argument("--contextual-isr-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--contextual-isr-time-seconds", type=int, default=30)
    parser.add_argument("--contextual-isr-reservoir-seconds", type=int, default=45)
    parser.add_argument("--contextual-isr-reservoir-instructions", type=int, default=20000)
    parser.add_argument("--contextual-isr-reservoir-max-tasks", type=int, default=96)
    parser.add_argument("--contextual-isr-reservoir-max-tasks-per-isr", type=int, default=24)
    parser.add_argument("--contextual-isr-reservoir-contexts", type=int, default=16)
    parser.add_argument("--stream-input-seconds", type=int, default=45)
    parser.add_argument("--stream-input-instructions", type=int, default=120000)
    parser.add_argument("--stream-input-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--stream-input-max-profiles", type=int, default=8)
    parser.add_argument("--stream-input-max-contexts", type=int, default=16)
    parser.add_argument("--stream-input-max-seeds", type=int, default=64)
    parser.add_argument("--stream-input-max-tasks", type=int, default=256)
    parser.add_argument("--stream-input-no-new-bbs", type=int, default=8192)
    parser.add_argument("--semantic-frontier-flush-seconds", type=int, default=20)
    parser.add_argument("--semantic-frontier-flush-max-targets", type=int, default=96)
    parser.add_argument("--semantic-frontier-flush-max-tasks", type=int, default=96)
    parser.add_argument("--semantic-frontier-flush-switch-only", action="store_true")
    parser.add_argument("--direct-call-continuation-seconds", type=int, default=20)
    parser.add_argument("--direct-call-continuation-instructions", type=int, default=300000)
    parser.add_argument("--direct-call-continuation-replay-timeout-us", type=int, default=2000000)
    parser.add_argument("--direct-call-continuation-max-tasks", type=int, default=64)
    parser.add_argument("--direct-call-continuation-max-targets", type=int, default=64)
    parser.add_argument("--direct-call-continuation-variants-per-call", type=int, default=2)
    parser.add_argument("--direct-call-continuation-no-new-bbs", type=int, default=16384)
    parser.add_argument("--direct-call-continuation-low-yield-stop-tasks", type=int, default=24)
    parser.add_argument("--direct-call-continuation-low-yield-min-tasks", type=int, default=8)
    parser.add_argument("--direct-call-summary-return-seconds", type=int, default=20)
    parser.add_argument("--direct-call-summary-return-instructions", type=int, default=300000)
    parser.add_argument("--direct-call-summary-return-replay-timeout-us", type=int, default=2000000)
    parser.add_argument("--direct-call-summary-return-max-tasks", type=int, default=64)
    parser.add_argument("--direct-call-summary-return-max-targets", type=int, default=64)
    parser.add_argument("--direct-call-summary-return-variants-per-call", type=int, default=2)
    parser.add_argument("--direct-call-summary-return-no-new-bbs", type=int, default=16384)
    parser.add_argument("--direct-call-summary-return-low-yield-stop-tasks", type=int, default=24)
    parser.add_argument("--direct-call-summary-return-low-yield-min-tasks", type=int, default=8)
    parser.add_argument("--rtos-thread-entry-seconds", type=int, default=30)
    parser.add_argument("--rtos-thread-entry-instructions", type=int, default=200000)
    parser.add_argument("--rtos-thread-entry-replay-timeout-us", type=int, default=2000000)
    parser.add_argument("--rtos-thread-entry-max-tasks", type=int, default=64)
    parser.add_argument("--rtos-thread-entry-max-targets", type=int, default=64)
    parser.add_argument("--rtos-thread-entry-variants-per-call", type=int, default=2)
    parser.add_argument("--rtos-thread-entry-no-new-bbs", type=int, default=16384)
    parser.add_argument("--frontier-successor-replay-seconds", type=int, default=240)
    parser.add_argument("--frontier-successor-replay-tail-seconds", type=int, default=120)
    parser.add_argument("--frontier-successor-replay-instructions", type=int, default=80000)
    parser.add_argument("--frontier-successor-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--frontier-successor-replay-max-tasks", type=int, default=1024)
    parser.add_argument("--frontier-successor-replay-max-targets", type=int, default=512)
    parser.add_argument("--frontier-successor-replay-variants-per-branch", type=int, default=2)
    parser.add_argument("--frontier-successor-replay-no-new-bbs", type=int, default=4096)
    parser.add_argument(
        "--stall-watchdog-threshold-seconds",
        type=int,
        default=None,
        help=(
            "Stage-level watchdog: end the current stage through its normal "
            "exit path (phase/evidence merge included, pipeline continues) "
            "when every ledger -- covered BBs and the evidence_status counts "
            "natural/counterfactual_only/diagnostic/canonical_unclassified/"
            "validated -- has been frozen for this many seconds and the "
            "zero-yield secondary signal holds. Non-positive or "
            "--disable-stall-watchdog turns it off; "
            "LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS overrides the default "
            "(90 minutes) when the flag is omitted."
        ),
    )
    parser.add_argument(
        "--stall-watchdog-min-zero-yield-units",
        type=int,
        default=None,
        help=(
            "Secondary signal: consecutive zero-yield attempts/interval samples "
            "required before the watchdog may stop the run. "
            "LSGEMU_STALL_WATCHDOG_MIN_ZERO_YIELD_UNITS overrides the default (8)."
        ),
    )
    parser.add_argument(
        "--stall-watchdog-exempt-stages",
        type=str,
        default="",
        help=(
            "Comma-separated stage-name tokens whose wall time never counts as "
            "stall (appended to the built-ins path_naturalization/vector_only_"
            "cleanup/deadline_drain/...). LSGEMU_STALL_WATCHDOG_EXEMPT_STAGES "
            "is the environment equivalent."
        ),
    )
    parser.add_argument(
        "--disable-stall-watchdog",
        action="store_true",
        help="Disable the run-level no-new-BB stall watchdog (default: enabled).",
    )
    parser.add_argument(
        "--stall-watchdog-max-consecutive-zero-stages",
        type=int,
        default=None,
        help=(
            "Run-level fallback: stop the whole case after this many "
            "consecutive depth-1 stages that each ran at least "
            "--stall-watchdog-min-zero-productivity-stage-seconds without "
            "moving any ledger (covered or evidence).  0 disables the "
            "streak rule; LSGEMU_STALL_WATCHDOG_MAX_CONSECUTIVE_ZERO_STAGES "
            "is the environment equivalent (default 3)."
        ),
    )
    parser.add_argument(
        "--stall-watchdog-min-zero-productivity-stage-seconds",
        type=int,
        default=None,
        help=(
            "A stage must run at least this long with zero ledger movement "
            "before it counts toward the consecutive-zero-stage streak "
            "(default 1800). LSGEMU_STALL_WATCHDOG_MIN_ZERO_PRODUCTIVITY_"
            "STAGE_SECONDS is the environment equivalent."
        ),
    )
    parser.add_argument(
        "--stall-watchdog-run-stall-seconds",
        type=int,
        default=None,
        help=(
            "Run-level safety floor: stop the case when the non-exempt "
            "pipeline has been frozen on every ledger for this many "
            "continuous seconds.  Must stay well above the largest "
            "legitimate dead window ever measured (7.85h); default 28800 "
            "(8h), 0 disables the floor. LSGEMU_STALL_WATCHDOG_RUN_STALL_"
            "SECONDS is the environment equivalent."
        ),
    )
    parser.add_argument("--path-naturalization-seconds", type=int, default=120)
    parser.add_argument("--path-naturalization-max-paths", type=int, default=64)
    parser.add_argument("--path-naturalization-max-attempts-per-path", type=int, default=8)
    parser.add_argument(
        "--path-naturalization-max-sources-per-edge",
        "--path-naturalization-max-facts-per-edge",
        dest="path_naturalization_max_sources_per_edge",
        type=int,
        default=4,
    )
    parser.add_argument("--path-naturalization-values-per-source", type=int, default=4)
    parser.add_argument("--path-naturalization-max-compound-candidates", type=int, default=4)
    parser.add_argument("--path-naturalization-instructions", type=int, default=120000)
    parser.add_argument("--path-naturalization-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--frontier-successor-flush-seconds", type=int, default=20)
    parser.add_argument("--frontier-successor-flush-min-snapshots", type=int, default=64)
    parser.add_argument("--frontier-successor-flush-min-dynamic-edges", type=int, default=4)
    parser.add_argument("--frontier-successor-flush-max-new-bbs", type=int, default=8)
    parser.add_argument(
        "--frontier-cycle-tail-reserve-seconds",
        type=int,
        default=-1,
        help=(
            "Reserve wallclock budget after targeted frontier cycles for strict "
            "successor/tail/deadline replay. Negative selects an adaptive reserve."
        ),
    )
    parser.add_argument(
        "--targeted-frontier-reserve-seconds",
        type=int,
        default=-1,
        help=(
            "Wallclock seconds to reserve before targeted frontier cycles. "
            "Negative selects an adaptive value."
        ),
    )
    parser.add_argument("--contextual-isr-frontier-seconds", type=int, default=20)
    parser.add_argument("--contextual-isr-frontier-rounds", type=int, default=0)
    parser.add_argument("--contextual-isr-frontier-max-rounds", type=int, default=2)
    parser.add_argument("--contextual-isr-frontier-stale-rounds", type=int, default=1)
    parser.add_argument("--contextual-isr-frontier-min-progress-bbs", type=int, default=1)
    parser.add_argument("--contextual-isr-frontier-max-targets", type=int, default=64)
    parser.add_argument("--contextual-isr-frontier-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--contextual-isr-frontier-switch-only", action="store_true")
    parser.add_argument("--contextual-isr-frontier-nearby-tail-ratio", type=float, default=0.5)
    parser.add_argument("--contextual-isr-frontier-candidate-pool-multiplier", type=int, default=4)
    parser.add_argument("--semantic-vector-cleanup-seconds", type=int, default=30)
    parser.add_argument("--vector-cleanup-seconds", type=int, default=60)
    parser.add_argument("--vector-cleanup-rounds", type=int, default=0)
    parser.add_argument("--vector-cleanup-max-rounds", type=int, default=2)
    parser.add_argument("--vector-cleanup-stale-rounds", type=int, default=1)
    parser.add_argument("--vector-cleanup-min-progress-bbs", type=int, default=1)
    parser.add_argument("--vector-cleanup-max-targets", type=int, default=128)
    parser.add_argument("--vector-cleanup-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--vector-cleanup-nearby-tail-ratio", type=float, default=0.75)
    parser.add_argument("--total-time-minutes", type=float, default=4.0)
    parser.add_argument("--post-interleaved-reserve-seconds", type=int, default=-1)
    parser.add_argument("--post-interleaved-reserve-ratio", type=float, default=0.40)
    parser.add_argument("--reservoir-round-seconds", type=int, default=180)
    parser.add_argument("--mmio-round-seconds", type=int, default=60)
    parser.add_argument("--interleaved-frontier-round-seconds", type=int, default=0)
    parser.add_argument("--interleaved-frontier-max-targets", type=int, default=64)
    parser.add_argument("--interleaved-frontier-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--interleaved-frontier-switch-only", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--stale-round-limit", type=int, default=2)
    parser.add_argument("--interleaved-adaptive-plateau-rounds", type=int, default=2)
    parser.add_argument("--interleaved-adaptive-plateau-min-rounds", type=int, default=4)
    parser.add_argument("--min-progress-bbs", type=int, default=1)
    parser.add_argument("--reservoir-max-tasks", type=int, default=400)
    parser.add_argument("--targeted-max-tasks-per-root", type=int, default=8)
    parser.add_argument("--reservoir-replay-instructions", type=int, default=50000)
    parser.add_argument("--reservoir-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--mmio-max-branches", type=int, default=8)
    parser.add_argument("--mmio-replay-instructions", type=int, default=50000)
    parser.add_argument("--mmio-replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--mmio-snapshot-variants-per-branch", type=int, default=4)
    parser.add_argument("--frontier-round-seconds", type=int, default=45)
    parser.add_argument("--frontier-rounds", type=int, default=0)
    parser.add_argument("--frontier-max-rounds", type=int, default=3)
    parser.add_argument("--frontier-stale-rounds", type=int, default=1)
    parser.add_argument("--frontier-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--frontier-min-progress-bbs", type=int, default=1)
    parser.add_argument("--frontier-max-targets", type=int, default=128)
    parser.add_argument("--frontier-switch-only", action="store_true")
    parser.add_argument("--frontier-nearby-tail-ratio", type=float, default=0.5)
    parser.add_argument("--frontier-candidate-pool-multiplier", type=int, default=4)
    parser.add_argument("--switch-frontier-round-seconds", type=int, default=30)
    parser.add_argument("--switch-frontier-rounds", type=int, default=0)
    parser.add_argument("--switch-frontier-max-rounds", type=int, default=2)
    parser.add_argument("--switch-frontier-stale-rounds", type=int, default=1)
    parser.add_argument("--switch-frontier-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--switch-frontier-min-progress-bbs", type=int, default=1)
    parser.add_argument("--switch-frontier-max-targets", type=int, default=96)
    parser.add_argument("--switch-frontier-candidate-pool-multiplier", type=int, default=1)
    parser.add_argument("--frontier-cycle-max-cycles", type=int, default=6)
    parser.add_argument("--frontier-cycle-stale-cycles", type=int, default=1)
    parser.add_argument("--hotspot-frontier-predecessors", type=int, default=4)
    parser.add_argument("--switch-frontier-low-yield-new-bbs", type=int, default=2)
    parser.add_argument("--switch-frontier-low-yield-targets", type=int, default=1)
    parser.add_argument("--switch-frontier-disable-low-yield-cycles", type=int, default=2)
    parser.add_argument(
        "--disable-targeted-direct-root-focus",
        action="store_true",
        help="Disable direct frontier-root filtering in targeted stages and use the broader static-potential root queue.",
    )
    parser.add_argument("--targeted-direct-root-fallback-limit", type=int, default=96)
    parser.add_argument("--target-cooldown-rounds", type=int, default=0)
    parser.add_argument(
        "--continue-targeted-state",
        action="store_true",
        help="Keep the existing reservoir state when running targeted frontier stages instead of resetting to the baseline targeted view.",
    )
    parser.add_argument(
        "--switch-frontier-strategy",
        default="targeted_direct_root_focus=1,prefer_direct_root_snapshot=1",
        help="Comma-separated boolean strategy for switch frontier targeted rounds.",
    )
    parser.add_argument(
        "--frontier-targeted-strategy",
        default="targeted_direct_root_focus=0,prefer_direct_root_snapshot=1",
        help="Comma-separated boolean strategy for general frontier targeted rounds.",
    )
    parser.add_argument(
        "--targeted-prefix-snapshot-seed-limit",
        type=int,
        default=64,
        help="Seed cap for targeted prefix snapshots used by targeted reservoir phases.",
    )
    parser.add_argument(
        "--interleaved-targeted-prefix-snapshot-seed-limit",
        type=int,
        default=-1,
        help="Seed cap for targeted prefix snapshots during the interleaved reservoir/mmio stage; negative means reuse --targeted-prefix-snapshot-seed-limit.",
    )
    parser.add_argument(
        "--disable-frontier-successor-replay-tail-use-remaining",
        action="store_true",
        help="Do not let the final successor-replay tail expand to consume leftover wallclock budget.",
    )
    parser.add_argument(
        "--disable-deadline-drain",
        action="store_true",
        help="Disable the final strict replay drain that reuses remaining wallclock budget after fixed stages.",
    )
    parser.add_argument(
        "--disable-semantic-obligation-stages",
        action="store_true",
        help=(
            "Contribution ablation v2: disable stagnation-to-obligation conversion, "
            "semantic types/scoring/feedback, and semantic-driven candidate generation "
            "while preserving a structural CFG/branch/MMIO fallback scheduler."
        ),
    )
    parser.add_argument(
        "--disable-scoped-replay-stages",
        action="store_true",
        help=(
            "Contribution ablation: disable branch-occurrence and target-scoped "
            "replay, leaving global repair fallbacks available."
        ),
    )
    parser.add_argument(
        "--disable-context-event-replay-stages",
        action="store_true",
        help=(
            "Contribution ablation: disable context-aware ISR, thread, and "
            "external-event replay while preserving non-event control-flow replay."
        ),
    )
    parser.add_argument(
        "--disable-constraint-guided-candidates",
        action="store_true",
        help=(
            "Candidate-generation ablation (C2 paired control): replace the "
            "predicate-rule + Z3 candidate generation with a fixed generic "
            "strategy (width boundary values, seed mutation, seeded random "
            "sampling). Target selection, input localization, occurrence "
            "identity, recovered width, and replay validation are unchanged; "
            "equivalent to LSGEMU_DISABLE_CONSTRAINT_GUIDED_CANDIDATES=1."
        ),
    )
    parser.add_argument("--deadline-drain-round-seconds", type=int, default=180)
    parser.add_argument("--deadline-drain-min-seconds", type=int, default=20)
    parser.add_argument("--deadline-drain-stale-rounds", type=int, default=3)
    parser.add_argument(
        "--deadline-drain-pressure-stale-rounds",
        type=int,
        default=12,
        help=(
            "Allow this many zero-growth deadline-drain rounds while replayable "
            "frontier/switch/direct-call pressure remains. This prevents long "
            "wallclock runs from stopping after only the generic stale limit."
        ),
    )
    parser.add_argument(
        "--deadline-drain-targeted-share",
        type=float,
        default=0.40,
        help="Fraction of each deadline-drain round reserved for switch/general frontier targeted replay when frontier pressure remains high.",
    )
    parser.add_argument(
        "--deadline-drain-targeted-first-threshold",
        type=int,
        default=96,
        help="Run targeted frontier before generic successor replay when at least this many replayable frontier BBs remain.",
    )
    parser.add_argument(
        "--deadline-drain-switch-share",
        type=float,
        default=0.35,
        help="Fraction of deadline targeted budget to spend on switch/TBB/TBH-like frontiers when switch targets remain.",
    )
    parser.add_argument(
        "--replay-time-skip-mode",
        choices=["stateful", "legacy", "off"],
        default=os.environ.get("LSGEMU_REPLAY_TIME_SKIP_MODE", "stateful"),
        help=(
            "How replay-created emulators summarize time functions. stateful "
            "returns monotonic time and breaks HAL timeout loops; legacy/off "
            "preserves the older behavior for firmwares where single-call "
            "summary-return exploration is more productive."
        ),
    )
    parser.add_argument(
        "--replay-time-increment",
        type=int,
        default=int(os.environ.get("LSGEMU_REPLAY_TIME_INCREMENT", "1000")),
        help="Increment used by stateful replay time summaries.",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    # r40 P4/C1（k.4 P0-3a）：campaign 默认启用 angr 可达基线口径
    # （LSGEMU_ANGR_REACHABILITY=1）。显式 =0 的 OFF 对照臂不受影响；
    # 文件缺失时产线给 WARNING 降级日志，reachable_* 字段保持 null。
    os.environ.setdefault("LSGEMU_ANGR_REACHABILITY", "1")
    # Derived internal switches used by the contribution-level ablations below.
    # They are not public ablation entry points.
    args.disable_branch_reservoir_stages = False
    args.disable_frontier_stages = False
    args.disable_semantic_frontier_drain_stage = False
    args.disable_contextual_isr_stages = False
    args.disable_direct_stream_thread_stages = False
    if args.run_profile and loaded_run_profile is None:
        loaded_run_profile = load_run_profile(args.run_profile)
        applied_run_profile_env = apply_run_profile_environment(loaded_run_profile)

    ablation_overrides: dict[str, object] = {}

    def record_ablation_override(name: str, value: object) -> None:
        previous = getattr(args, name, None)
        if previous != value:
            ablation_overrides[name] = {"from": previous, "to": value}
            setattr(args, name, value)

    def apply_ablation_stage_disables() -> None:
        if args.disable_semantic_obligation_stages:
            for option_name in (
                "semantic_entry_prefix_seconds",
                "semantic_frontier_flush_seconds",
                "semantic_vector_cleanup_seconds",
            ):
                record_ablation_override(option_name, 0)
            record_ablation_override("disable_semantic_frontier_drain_stage", True)

        if args.disable_scoped_replay_stages:
            record_ablation_override("disable_branch_reservoir_stages", True)
            record_ablation_override("disable_frontier_stages", True)
            for option_name in (
                "contextual_isr_frontier_seconds",
                "semantic_vector_cleanup_seconds",
                "vector_cleanup_seconds",
                "targeted_prefix_snapshot_seed_limit",
                "interleaved_targeted_prefix_snapshot_seed_limit",
                "path_naturalization_seconds",
            ):
                record_ablation_override(option_name, 0)

        if args.disable_context_event_replay_stages:
            record_ablation_override("disable_contextual_isr_stages", True)
            for option_name in (
                "stream_input_seconds",
                "rtos_thread_entry_seconds",
            ):
                record_ablation_override(option_name, 0)

        if args.disable_branch_reservoir_stages:
            record_ablation_override("reservoir_round_seconds", 0)

        if args.disable_frontier_stages:
            for option_name in (
                "interleaved_frontier_round_seconds",
                "semantic_frontier_flush_seconds",
                "frontier_successor_replay_seconds",
                "frontier_successor_replay_tail_seconds",
                "frontier_successor_flush_seconds",
                "frontier_round_seconds",
                "switch_frontier_round_seconds",
            ):
                record_ablation_override(option_name, 0)
            record_ablation_override("frontier_max_rounds", 0)
            record_ablation_override("switch_frontier_max_rounds", 0)
            record_ablation_override("frontier_cycle_max_cycles", 0)

        if args.disable_contextual_isr_stages:
            for option_name in (
                "contextual_isr_contexts",
                "contextual_isr_time_seconds",
                "contextual_isr_reservoir_seconds",
                "contextual_isr_frontier_seconds",
                "semantic_vector_cleanup_seconds",
                "vector_cleanup_seconds",
            ):
                record_ablation_override(option_name, 0)

        if args.disable_direct_stream_thread_stages:
            for option_name in (
                "stream_input_seconds",
                "direct_call_continuation_seconds",
                "direct_call_summary_return_seconds",
                "rtos_thread_entry_seconds",
            ):
                record_ablation_override(option_name, 0)

    apply_ablation_stage_disables()

    adaptive_stage_caps: dict[str, object] = {}

    def apply_wallclock_adaptive_stage_caps() -> None:
        """
        Keep fixed replay stages proportional to the requested wallclock.

        Long campaigns intentionally use deep 4h defaults. Short 5-20 minute
        probes need a complete dynamic loop instead: entry/interleaving,
        direct-call replay, stream replay, successor replay, and a small tail
        drain.  Without these caps one large configured stage can consume the
        entire process budget and make later real-entry stages look empty.
        """
        try:
            total_seconds = max(0, int(float(args.total_time_minutes) * 60))
        except (TypeError, ValueError):
            total_seconds = 0
        if total_seconds <= 0:
            return
        if str(os.environ.get("LSGEMU_DISABLE_WALLCLOCK_STAGE_CAPS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            adaptive_stage_caps["disabled_by_env"] = True
            return
        adaptive_stage_caps.update({"total_seconds": total_seconds, "changes": {}})

        def ensure_baseline_timeout(fraction: float, minimum: float, maximum: float, reason: str) -> None:
            current_baseline_timeout = float(getattr(args, "baseline_timeout_seconds", 0.0) or 0.0)
            if current_baseline_timeout > 0:
                return
            capped_timeout = max(float(minimum), min(float(maximum), total_seconds * float(fraction)))
            args.baseline_timeout_seconds = capped_timeout
            adaptive_stage_caps["changes"]["baseline_timeout_seconds"] = {
                "from": current_baseline_timeout,
                "to": capped_timeout,
                "reason": reason,
            }

        if total_seconds > 7200:
            adaptive_stage_caps["mode"] = "long_bounded_baseline"
            ensure_baseline_timeout(0.02, 90.0, 300.0, "long_budget_entry_cap")
            return

        if total_seconds <= 600:
            mode = "short"
        elif total_seconds <= 1800:
            mode = "medium"
        else:
            mode = "extended"
        adaptive_stage_caps["mode"] = mode

        def cap_seconds(name: str, fraction: float, minimum: int, maximum: int, *, zero_short: bool = False) -> None:
            current = int(getattr(args, name, 0) or 0)
            if current <= 0:
                return
            if zero_short and mode == "short":
                capped = 0
            else:
                capped = min(current, max(int(minimum), min(int(maximum), int(total_seconds * fraction))))
            if capped != current:
                setattr(args, name, capped)
                adaptive_stage_caps["changes"][name] = {"from": current, "to": capped}

        def cap_count(name: str, value: int) -> None:
            current = int(getattr(args, name, 0) or 0)
            if current <= 0:
                return
            capped = min(current, max(1, int(value)))
            if capped != current:
                setattr(args, name, capped)
                adaptive_stage_caps["changes"][name] = {"from": current, "to": capped}

        if mode == "short":
            ensure_baseline_timeout(0.15, 20.0, 45.0, "short_budget_entry_cap")
            if str(os.environ.get("LSGEMU_ENABLE_SHORT_SEMANTIC_ENTRY_PREFIX", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                cap_seconds("semantic_entry_prefix_seconds", 0.06, 8, 16)
            elif int(getattr(args, "semantic_entry_prefix_seconds", 0) or 0) > 0:
                previous = int(args.semantic_entry_prefix_seconds)
                args.semantic_entry_prefix_seconds = 0
                adaptive_stage_caps["changes"]["semantic_entry_prefix_seconds"] = {
                    "from": previous,
                    "to": 0,
                    "reason": "short_budget_reservoir_first",
                }
            cap_seconds("contextual_isr_reservoir_seconds", 0.00, 0, 0, zero_short=True)
            cap_seconds("contextual_isr_frontier_seconds", 0.04, 8, 12)
            cap_seconds("semantic_vector_cleanup_seconds", 0.00, 0, 0, zero_short=True)
            cap_seconds("vector_cleanup_seconds", 0.04, 8, 12)
            cap_seconds("stream_input_seconds", 0.04, 8, 14)
            cap_seconds("semantic_frontier_flush_seconds", 0.02, 6, 8)
            cap_seconds("direct_call_continuation_seconds", 0.08, 16, 28)
            cap_seconds("direct_call_summary_return_seconds", 0.08, 16, 28)
            cap_seconds("rtos_thread_entry_seconds", 0.04, 8, 14)
            cap_seconds("frontier_successor_replay_seconds", 0.12, 24, 40)
            cap_seconds("frontier_successor_replay_tail_seconds", 0.10, 20, 36)
            cap_seconds("path_naturalization_seconds", 0.03, 6, 10)
            cap_seconds("frontier_successor_flush_seconds", 0.04, 8, 12)
            cap_seconds("frontier_round_seconds", 0.06, 12, 20)
            cap_seconds("switch_frontier_round_seconds", 0.05, 10, 16)
            cap_seconds("deadline_drain_round_seconds", 0.10, 20, 36)
            cap_seconds("deadline_drain_min_seconds", 0.03, 8, 10)
            cap_count("frontier_cycle_max_cycles", 2)
            cap_count("frontier_max_rounds", 2)
            cap_count("switch_frontier_max_rounds", 2)
            cap_count("direct_call_continuation_max_tasks", 96)
            cap_count("direct_call_summary_return_max_tasks", 96)
            cap_count("rtos_thread_entry_max_tasks", 64)
            cap_count("frontier_successor_replay_max_tasks", 384)
            cap_count("stream_input_max_tasks", 128)
            cap_count("direct_call_continuation_instructions", 80000)
            cap_count("direct_call_continuation_replay_timeout_us", 600000)
            cap_count("direct_call_continuation_no_new_bbs", 4096)
            cap_count("direct_call_summary_return_instructions", 120000)
            cap_count("direct_call_summary_return_replay_timeout_us", 800000)
            cap_count("direct_call_summary_return_no_new_bbs", 8192)
            cap_count("frontier_successor_replay_instructions", 40000)
            cap_count("frontier_successor_replay_timeout_us", 400000)
            cap_count("frontier_successor_replay_no_new_bbs", 2048)
            short_tail_reserve = max(
                18,
                min(
                    36,
                    int(total_seconds * 0.12),
                    max(0, int(args.direct_call_continuation_seconds))
                    + max(0, int(args.direct_call_summary_return_seconds))
                    + max(0, int(args.frontier_successor_replay_tail_seconds)) // 3,
                ),
            )
            if int(args.frontier_cycle_tail_reserve_seconds) < 0:
                args.frontier_cycle_tail_reserve_seconds = short_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": -1,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                }
            elif int(args.frontier_cycle_tail_reserve_seconds) > short_tail_reserve:
                previous = int(args.frontier_cycle_tail_reserve_seconds)
                args.frontier_cycle_tail_reserve_seconds = short_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": previous,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                    "reason": "short_budget_do_not_starve_frontier_replay",
                }
        elif mode == "medium":
            ensure_baseline_timeout(0.10, 45.0, 120.0, "medium_budget_entry_cap")
            cap_seconds("semantic_entry_prefix_seconds", 0.04, 20, 45)
            cap_seconds("contextual_isr_reservoir_seconds", 0.04, 24, 45)
            cap_seconds("contextual_isr_frontier_seconds", 0.03, 18, 36)
            cap_seconds("semantic_vector_cleanup_seconds", 0.02, 12, 24)
            cap_seconds("vector_cleanup_seconds", 0.04, 24, 45)
            cap_seconds("stream_input_seconds", 0.06, 35, 60)
            cap_seconds("semantic_frontier_flush_seconds", 0.03, 18, 36)
            cap_seconds("direct_call_continuation_seconds", 0.07, 45, 90)
            cap_seconds("direct_call_summary_return_seconds", 0.07, 45, 90)
            cap_seconds("rtos_thread_entry_seconds", 0.06, 35, 75)
            cap_seconds("frontier_successor_replay_seconds", 0.12, 80, 150)
            cap_seconds("frontier_successor_replay_tail_seconds", 0.14, 90, 210)
            cap_seconds("path_naturalization_seconds", 0.03, 24, 45)
            cap_seconds("frontier_successor_flush_seconds", 0.03, 18, 36)
            cap_seconds("frontier_round_seconds", 0.05, 35, 75)
            cap_seconds("switch_frontier_round_seconds", 0.04, 24, 60)
            cap_seconds("deadline_drain_round_seconds", 0.12, 90, 180)
            cap_seconds("deadline_drain_min_seconds", 0.02, 12, 20)
            cap_count("frontier_cycle_max_cycles", 5)
            cap_count("frontier_max_rounds", 4)
            cap_count("switch_frontier_max_rounds", 3)
            cap_count("direct_call_continuation_max_tasks", 256)
            cap_count("direct_call_summary_return_max_tasks", 256)
            cap_count("rtos_thread_entry_max_tasks", 192)
            cap_count("frontier_successor_replay_max_tasks", 1024)
            cap_count("stream_input_max_tasks", 256)
            cap_count("direct_call_continuation_instructions", 120000)
            cap_count("direct_call_continuation_replay_timeout_us", 1000000)
            cap_count("direct_call_continuation_no_new_bbs", 8192)
            cap_count("direct_call_summary_return_instructions", 180000)
            cap_count("direct_call_summary_return_replay_timeout_us", 1200000)
            cap_count("direct_call_summary_return_no_new_bbs", 16384)
            cap_count("frontier_successor_replay_instructions", 60000)
            cap_count("frontier_successor_replay_timeout_us", 700000)
            cap_count("frontier_successor_replay_no_new_bbs", 4096)
            medium_tail_reserve = max(
                60,
                min(
                    150,
                    int(total_seconds * 0.12),
                    max(0, int(args.direct_call_continuation_seconds))
                    + max(0, int(args.direct_call_summary_return_seconds))
                    + max(0, int(args.frontier_successor_replay_tail_seconds)) // 2,
                ),
            )
            if int(args.frontier_cycle_tail_reserve_seconds) < 0:
                args.frontier_cycle_tail_reserve_seconds = medium_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": -1,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                }
            elif int(args.frontier_cycle_tail_reserve_seconds) > medium_tail_reserve:
                previous = int(args.frontier_cycle_tail_reserve_seconds)
                args.frontier_cycle_tail_reserve_seconds = medium_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": previous,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                    "reason": "medium_budget_do_not_starve_frontier_replay",
                }
        else:
            ensure_baseline_timeout(0.05, 90.0, 300.0, "extended_budget_entry_cap")
            cap_seconds("semantic_entry_prefix_seconds", 0.03, 45, 120)
            cap_seconds("contextual_isr_reservoir_seconds", 0.03, 60, 150)
            cap_seconds("contextual_isr_frontier_seconds", 0.025, 45, 120)
            cap_seconds("semantic_vector_cleanup_seconds", 0.015, 20, 60)
            cap_seconds("vector_cleanup_seconds", 0.035, 75, 180)
            cap_seconds("stream_input_seconds", 0.04, 60, 240)
            cap_seconds("semantic_frontier_flush_seconds", 0.02, 30, 90)
            cap_seconds("direct_call_continuation_seconds", 0.045, 90, 240)
            cap_seconds("direct_call_summary_return_seconds", 0.04, 75, 210)
            cap_seconds("rtos_thread_entry_seconds", 0.035, 75, 180)
            cap_seconds("frontier_successor_replay_seconds", 0.09, 180, 600)
            cap_seconds("frontier_successor_replay_tail_seconds", 0.12, 240, 900)
            cap_seconds("path_naturalization_seconds", 0.02, 45, 90)
            cap_seconds("frontier_successor_flush_seconds", 0.015, 20, 60)
            cap_seconds("frontier_round_seconds", 0.025, 60, 150)
            cap_seconds("switch_frontier_round_seconds", 0.02, 45, 120)
            cap_seconds("deadline_drain_round_seconds", 0.07, 180, 420)
            cap_seconds("deadline_drain_min_seconds", 0.01, 20, 45)
            cap_count("frontier_cycle_max_cycles", 10)
            cap_count("frontier_max_rounds", 6)
            cap_count("switch_frontier_max_rounds", 4)
            cap_count("direct_call_continuation_max_tasks", 384)
            cap_count("direct_call_summary_return_max_tasks", 384)
            cap_count("rtos_thread_entry_max_tasks", 384)
            cap_count("frontier_successor_replay_max_tasks", 2048)
            cap_count("stream_input_max_tasks", 512)
            cap_count("direct_call_continuation_no_new_bbs", 32768)
            cap_count("direct_call_summary_return_no_new_bbs", 32768)
            cap_count("frontier_successor_replay_no_new_bbs", 32768)
            extended_tail_reserve = max(30, min(90, int(total_seconds * 0.015)))
            if int(args.frontier_cycle_tail_reserve_seconds) < 0:
                args.frontier_cycle_tail_reserve_seconds = extended_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": -1,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                }
            elif int(args.frontier_cycle_tail_reserve_seconds) > extended_tail_reserve:
                previous = int(args.frontier_cycle_tail_reserve_seconds)
                args.frontier_cycle_tail_reserve_seconds = extended_tail_reserve
                adaptive_stage_caps["changes"]["frontier_cycle_tail_reserve_seconds"] = {
                    "from": previous,
                    "to": int(args.frontier_cycle_tail_reserve_seconds),
                    "reason": "extended_budget_keep_targeted_cycles_active",
                }

    apply_wallclock_adaptive_stage_caps()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))
    os.environ["LSGEMU_REPLAY_TIME_SKIP_MODE"] = str(args.replay_time_skip_mode)
    os.environ["LSGEMU_REPLAY_TIME_INCREMENT"] = str(max(1, int(args.replay_time_increment)))
    if args.disable_targeted_direct_root_focus:
        os.environ["LSGEMU_TARGETED_DIRECT_ROOT_FOCUS"] = "0"
    else:
        os.environ.setdefault("LSGEMU_TARGETED_DIRECT_ROOT_FOCUS", "1")
    os.environ.setdefault(
        "LSGEMU_TARGETED_DIRECT_ROOT_FALLBACK_LIMIT",
        str(max(0, args.targeted_direct_root_fallback_limit)),
    )
    os.environ["LSGEMU_TARGETED_PREFIX_SNAPSHOT_SEED_LIMIT"] = str(
        max(0, args.targeted_prefix_snapshot_seed_limit)
    )
    # The strongest validated Gateway runs relied on targeted root-snapshot
    # replay remaining enabled during both interleaved and targeted frontier
    # stages. Keep that legacy behavior unless the caller overrides it via the
    # environment.
    os.environ.setdefault("LSGEMU_TARGETED_SWITCH_ROOT_SNAPSHOT_REPLAY", "1")
    os.environ.setdefault("LSGEMU_TARGETED_SWITCH_ROOT_BUDGET", "64")
    runtime_bootstrap = bootstrap_runtime_dependencies()
    switch_stage_strategy = parse_stage_strategy(
        "switch_frontier",
        args.switch_frontier_strategy,
    )
    frontier_stage_strategy = parse_stage_strategy(
        "frontier_targeted",
        args.frontier_targeted_strategy,
    )

    firmware = Path(args.firmware).resolve()
    firmware_input_provenance = None
    if args.firmware_input_provenance_json:
        try:
            firmware_input_provenance = json.loads(
                args.firmware_input_provenance_json
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid firmware input provenance JSON: {exc}")
        if not isinstance(firmware_input_provenance, dict):
            parser.error("firmware input provenance JSON must contain an object")
    execution_thumb_override = (
        args.execution_mode_override == "thumb"
        if args.execution_mode_override is not None
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    configure_snapshot_storage_for_run(
        output_dir,
        run_id=os.environ.get("LSGEMU_ATTEMPT_ID") or None,
    )
    llm_config, llm_config_path = load_llm_config(RUNNER_PROJECT_ROOT)
    if llm_disabled_by_env():
        # r42 full_llm_off：按"未部署 LLM 配置"处理，全程零 LLM 客户端。
        llm_config, llm_config_path = None, None
    constraint_file = (
        Path(args.constraint_file).resolve()
        if args.constraint_file
        else workspace_artifact_path(output_dir, firmware, "_lsgemu_constraints.json")
    )
    if args.clean_constraint_file and constraint_file.exists():
        constraint_file.unlink()

    learned_roots = [Path(item).resolve() for item in args.constraint_seed_root]
    if args.constraint_seed_mode != "none":
        learned_roots.append(output_dir.parent)
    initial_seed_mode = args.constraint_seed_mode if args.constraint_seed_activation == "initial" else "none"
    seeded_constraint_count = seed_constraint_file(
        firmware,
        constraint_file,
        learned_mode=initial_seed_mode,
        learned_roots=learned_roots,
    )

    report_file = workspace_artifact_path(output_dir, firmware, "_interleaved_report.json")
    branch_catalog_file = workspace_artifact_path(output_dir, firmware, "_branch_catalog.json")
    llm_history_file = workspace_artifact_path(output_dir, firmware, "_llm_history.json")
    # 增量推断 journal：每条推断完成即落盘。finalize 的 _llm_history.json
    # 只在正常收尾时写出；campaign 超时 SIGTERM 会绕过 finalize，因此
    # journal 是推断历史的崩溃/超时安全载体（campaign driver 会按 attempt
    # 后缀通过 LSGEMU_LLM_INFERENCE_JOURNAL 传入独立路径）。
    llm_journal_env = str(os.environ.get("LSGEMU_LLM_INFERENCE_JOURNAL") or "").strip()
    llm_history_journal_file = (
        Path(llm_journal_env).expanduser().resolve()
        if llm_journal_env
        else workspace_artifact_path(output_dir, firmware, "_llm_history.jsonl")
    )
    progress_jsonl_file = progress_artifact_path(output_dir, firmware)
    try:
        progress_interval_seconds = int(
            os.environ.get("LSGEMU_PROGRESS_INTERVAL_SECONDS")
            or os.environ.get("LSGEMU_COVERAGE_CHECKPOINT_INTERVAL")
            or "300"
        )
    except ValueError:
        progress_interval_seconds = 300

    start = time.time()
    total_wallclock_budget_seconds = max(0, int(args.total_time_minutes * 60))
    wallclock_deadline = (
        start + total_wallclock_budget_seconds
        if total_wallclock_budget_seconds > 0
        else None
    )
    # Stage-level multi-ledger stall watchdog (round 3).  Default action is
    # stage truncation: a stage frozen on every ledger (covered BBs plus the
    # evidence_status counts) for threshold_seconds ends through its ordinary
    # exit path -- exploration kernels break between attempts and still run
    # their _record_phase merge, and the pipeline continues with the next
    # stage under the REAL remaining budget (no wind-down allowance is
    # minted).  A run-level stop only happens as a fallback: K consecutive
    # zero-productivity stages, or the 8h all-ledger safety floor.  Once the
    # run-level stop fires, the budget checks below report an exhausted
    # budget so the scheduler skips the remaining budget-consuming stages
    # through their ordinary no-budget paths while the run still finalizes
    # its report/checkpoints; the bounded evidence wind-down stages
    # (path_naturalization, vector_only_cleanup, path_naturalization_final)
    # draw from a separate capped allowance so a stopped run keeps producing
    # the same obligation/cleanup artifacts as a full-budget run.
    stall_watchdog = StallWatchdog.from_env(args, started_at=start)
    stall_watchdog_winddown = WinddownBudgetTracker(
        max_seconds=env_int(
            "LSGEMU_STALL_WATCHDOG_WINDDOWN_MAX_SECONDS",
            DEFAULT_WINDDOWN_MAX_SECONDS,
        ),
    )

    def remaining_wallclock_seconds() -> int | None:
        if stall_watchdog.should_stop():
            return 0
        if wallclock_deadline is None:
            return None
        return max(0, int(wallclock_deadline - time.time()))

    def true_remaining_wallclock_seconds() -> int | None:
        # Watchdog-independent view: the wallclock that would be left without
        # an early stop.  Used for the budget-saving bookkeeping in reports.
        if wallclock_deadline is None:
            return None
        return max(0, int(wallclock_deadline - time.time()))

    def remaining_winddown_seconds() -> int | None:
        true_remaining = true_remaining_wallclock_seconds()
        return stall_watchdog_winddown.remaining(
            watchdog_stopped=stall_watchdog.should_stop(),
            true_remaining=true_remaining,
            # r37：墙钟耗尽也是触发条件，否则长跑把预算全花在探索上、证据
            # 收尾阶段（path_naturalization 等）恒被 zero-budget 跳过。
            wallclock_exhausted=(
                true_remaining is not None and int(true_remaining) <= 0
            ),
        )

    def has_winddown_budget(min_seconds: int = 1) -> bool:
        remaining = remaining_winddown_seconds()
        return remaining is None or remaining >= max(0, min_seconds)

    def has_winddown_stage_budget(
        configured_seconds: int,
        min_fraction: float = 0.50,
        reserve_after_seconds: int = 0,
    ) -> bool:
        if configured_seconds <= 0:
            return False
        remaining = remaining_winddown_seconds()
        if remaining is None:
            return True
        reserve_after = max(0, int(reserve_after_seconds))
        min_required = max(1, int(configured_seconds * max(0.0, min(1.0, min_fraction))))
        return remaining >= reserve_after + min_required

    def clamp_winddown_stage_seconds(
        configured_seconds: int,
        reserve_after_seconds: int = 0,
    ) -> int:
        if configured_seconds <= 0:
            return 0
        remaining = remaining_winddown_seconds()
        if remaining is None:
            return configured_seconds
        reserve_after = max(0, int(reserve_after_seconds))
        available = remaining - reserve_after
        if available <= 0:
            return 0
        return min(configured_seconds, available)

    def has_wallclock_budget(min_seconds: int = 1) -> bool:
        remaining = remaining_wallclock_seconds()
        return remaining is None or remaining >= max(0, min_seconds)

    def has_stage_wallclock_budget(
        configured_seconds: int,
        min_fraction: float = 0.50,
        reserve_after_seconds: int = 0,
    ) -> bool:
        if configured_seconds <= 0:
            return False
        remaining = remaining_wallclock_seconds()
        if remaining is None:
            return True
        reserve_after = max(0, int(reserve_after_seconds))
        min_required = max(1, int(configured_seconds * max(0.0, min(1.0, min_fraction))))
        return remaining >= reserve_after + min_required

    def clamp_enabled_stage_seconds(
        configured_seconds: int,
        reserve_after_seconds: int = 0,
    ) -> int:
        if configured_seconds <= 0:
            return 0
        remaining = remaining_wallclock_seconds()
        if remaining is None:
            return configured_seconds
        reserve_after = max(0, int(reserve_after_seconds))
        available = remaining - reserve_after
        if available <= 0:
            return 0
        return min(configured_seconds, available)

    def clamp_optional_stage_limit(configured_seconds: int) -> int | None:
        remaining = remaining_wallclock_seconds()
        if remaining is None:
            return configured_seconds if configured_seconds > 0 else None
        if remaining <= 0:
            return 0
        if configured_seconds <= 0:
            return remaining
        return min(configured_seconds, remaining)

    def extend_final_stage_seconds(configured_seconds: int, protected_after_seconds: int = 0) -> int:
        protected = max(0, int(protected_after_seconds))
        base = clamp_enabled_stage_seconds(
            configured_seconds,
            reserve_after_seconds=protected,
        )
        if configured_seconds <= 0 or args.disable_frontier_successor_replay_tail_use_remaining:
            return base
        remaining = remaining_wallclock_seconds()
        if remaining is None:
            return base
        return max(base, max(0, remaining - protected))

    switch_frontier_enabled = (
        args.switch_frontier_round_seconds > 0
        and args.switch_frontier_max_rounds != 0
        and not args.frontier_switch_only
    )
    frontier_targeted_enabled = args.frontier_round_seconds > 0 and args.frontier_max_rounds != 0
    frontier_cycle_auto = (
        (switch_frontier_enabled and args.switch_frontier_rounds <= 0)
        or (frontier_targeted_enabled and args.frontier_rounds <= 0)
    )

    def short_minimum_interleaved_seconds() -> int:
        if not (total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600):
            return 0
        positive_buckets = [
            max(0, int(args.reservoir_round_seconds)),
            max(0, int(args.mmio_round_seconds)),
            max(0, int(args.interleaved_frontier_round_seconds)),
        ]
        positive_buckets = [value for value in positive_buckets if value > 0]
        if not positive_buckets:
            return 0
        # The first reservoir round is where entry-derived branch/root states
        # are harvested.  Do not let later semantic/targeted stages reserve
        # the entire short-run budget before that state exists; otherwise the
        # run degenerates into low-yield replay.
        return max(
            20,
            min(45, max(positive_buckets), int(total_wallclock_budget_seconds * 0.08)),
        )

    def estimate_post_interleaved_reserve_seconds() -> int:
        minimum_interleaved_seconds = short_minimum_interleaved_seconds()
        explicit_reserve = int(args.post_interleaved_reserve_seconds)
        if explicit_reserve >= 0:
            reserve_seconds = max(0, explicit_reserve)
        else:
            reserve_seconds = 0
            planned_frontier_cycles = 1
            if frontier_cycle_auto and (switch_frontier_enabled or frontier_targeted_enabled):
                planned_frontier_cycles = max(1, args.frontier_cycle_max_cycles)
            if args.contextual_isr_contexts != 0:
                reserve_seconds += max(0, args.contextual_isr_reservoir_seconds)
                reserve_seconds += max(0, args.contextual_isr_frontier_seconds)
                reserve_seconds += max(0, args.semantic_vector_cleanup_seconds)
                reserve_seconds += max(0, args.vector_cleanup_seconds)
            reserve_seconds += max(0, args.stream_input_seconds)
            reserve_seconds += max(0, args.semantic_frontier_flush_seconds)
            reserve_seconds += max(0, args.direct_call_continuation_seconds)
            reserve_seconds += max(0, args.rtos_thread_entry_seconds)
            reserve_seconds += max(0, args.direct_call_summary_return_seconds)
            reserve_seconds += max(0, args.frontier_successor_replay_seconds)
            reserve_seconds += max(0, args.frontier_successor_replay_tail_seconds)
            if switch_frontier_enabled:
                reserve_seconds += (
                    max(0, args.switch_frontier_round_seconds)
                    * max(1, auto_round_limit(args.switch_frontier_rounds, args.switch_frontier_max_rounds))
                    * planned_frontier_cycles
                )
            if frontier_targeted_enabled:
                reserve_seconds += (
                    max(0, args.frontier_round_seconds)
                    * max(1, auto_round_limit(args.frontier_rounds, args.frontier_max_rounds))
                    * planned_frontier_cycles
                )

        if total_wallclock_budget_seconds <= 0 or reserve_seconds <= 0:
            return max(0, reserve_seconds)

        clamped_ratio = min(0.9, max(0.0, float(args.post_interleaved_reserve_ratio)))
        if total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600:
            # Short probes must still exercise strict thread/direct-call/tail
            # stages, but over-reserving here leaves too little time to harvest
            # branch/root state during the interleaved stage.
            clamped_ratio = max(clamped_ratio, 0.60)
        reserve_cap = int(total_wallclock_budget_seconds * clamped_ratio)
        if reserve_cap <= 0:
            return 0
        if minimum_interleaved_seconds > 0:
            reserve_cap = min(
                reserve_cap,
                max(0, total_wallclock_budget_seconds - minimum_interleaved_seconds),
            )
        return min(reserve_seconds, reserve_cap)

    reserved_post_interleaved_seconds = estimate_post_interleaved_reserve_seconds()

    def adaptive_frontier_cycle_tail_reserve_seconds() -> int:
        explicit_reserve = int(args.frontier_cycle_tail_reserve_seconds)
        if explicit_reserve >= 0:
            return max(0, explicit_reserve)

        reserve_seconds = 0
        if args.frontier_successor_flush_seconds > 0:
            reserve_seconds = max(reserve_seconds, min(60, max(1, args.frontier_successor_flush_seconds)))
        if args.frontier_successor_replay_tail_seconds > 0:
            tail_probe = min(
                60,
                max(15, int(max(1, args.frontier_successor_replay_tail_seconds) * 0.10)),
            )
            reserve_seconds = max(reserve_seconds, tail_probe)
        if not args.disable_deadline_drain:
            reserve_seconds = max(reserve_seconds, max(0, int(args.deadline_drain_min_seconds)))

        if total_wallclock_budget_seconds > 0:
            reserve_cap = max(15, int(total_wallclock_budget_seconds * 0.10))
            reserve_seconds = min(reserve_seconds, reserve_cap)
        return max(0, reserve_seconds)

    frontier_cycle_tail_reserve_seconds = adaptive_frontier_cycle_tail_reserve_seconds()
    auto_targeted_frontier_reserve_seconds = estimate_targeted_frontier_reserve_seconds(
        total_wallclock_budget_seconds=total_wallclock_budget_seconds,
        switch_frontier_enabled=switch_frontier_enabled,
        frontier_targeted_enabled=frontier_targeted_enabled,
        frontier_cycle_auto=frontier_cycle_auto,
        switch_round_seconds=args.switch_frontier_round_seconds,
        switch_rounds=args.switch_frontier_rounds,
        switch_max_rounds=args.switch_frontier_max_rounds,
        frontier_round_seconds=args.frontier_round_seconds,
        frontier_rounds=args.frontier_rounds,
        frontier_max_rounds=args.frontier_max_rounds,
        frontier_cycle_max_cycles=args.frontier_cycle_max_cycles,
        frontier_cycle_tail_reserve_seconds=frontier_cycle_tail_reserve_seconds,
    )
    targeted_frontier_reserve_seconds = (
        max(0, int(args.targeted_frontier_reserve_seconds))
        if int(args.targeted_frontier_reserve_seconds) >= 0
        else auto_targeted_frontier_reserve_seconds
    )
    runner = HistoricalRunner(
        firmware,
        max_snapshots=5,
        use_llm=llm_config is not None,
        llm_config=llm_config,
        llm_config_path=llm_config_path,
        constraint_file=str(constraint_file),
        entry_point_override=args.entry_point,
        load_base_override=args.load_base,
        replay_time_skip_mode=str(args.replay_time_skip_mode),
        replay_time_increment=max(1, int(args.replay_time_increment)),
        semantic_obligation_enabled=not args.disable_semantic_obligation_stages,
        scoped_replay_enabled=not args.disable_scoped_replay_stages,
        constraint_guided_candidates_enabled=(
            not args.disable_constraint_guided_candidates
        ),
        # Cross-BB causal tracing belongs to replay/hypothesis realization,
        # not to semantic obligation scheduling, so A1 must retain it.
        causal_constraint_recovery_enabled=True,
        firmware_input_provenance=firmware_input_provenance,
        execution_thumb_override=execution_thumb_override,
    )
    # Shared with the exploration kernels through the runner so per-attempt
    # observations from HistoricalRunner and interval samples from the monitor
    # feed the same stall clock.
    runner.stall_watchdog = stall_watchdog
    if runner.llm_guide is not None:
        runner.llm_guide.enable_inference_journal(
            llm_history_journal_file,
            session_id=os.environ.get("LSGEMU_ATTEMPT_ID") or None,
        )
    if (
        args.disable_semantic_obligation_stages
        or args.disable_scoped_replay_stages
        or args.disable_context_event_replay_stages
        or args.disable_constraint_guided_candidates
        or args.disable_branch_reservoir_stages
        or args.disable_frontier_stages
        or args.disable_contextual_isr_stages
        or args.disable_direct_stream_thread_stages
    ):
        runner.phase_metadata.setdefault("ablation_config", {}).update({
            "coverage_counted": False,
            "ablation_schema_version": (
                2
                if (
                    args.disable_semantic_obligation_stages
                    or args.disable_scoped_replay_stages
                )
                else 1
            ),
            "disable_semantic_obligation_stages": bool(args.disable_semantic_obligation_stages),
            "disable_scoped_replay_stages": bool(args.disable_scoped_replay_stages),
            "disable_context_event_replay_stages": bool(args.disable_context_event_replay_stages),
            "disable_constraint_guided_candidates": bool(args.disable_constraint_guided_candidates),
            "effective_internal_switches": {
                "generic_branch_replay_disabled": bool(args.disable_branch_reservoir_stages),
                "targeted_frontier_recovery_disabled": bool(args.disable_frontier_stages),
                "path_naturalization_disabled": bool(
                    args.disable_semantic_obligation_stages
                    or args.disable_scoped_replay_stages
                ),
                "semantic_frontier_drain_disabled": bool(args.disable_semantic_frontier_drain_stage),
                "semantic_stagnation_conversion_disabled": bool(args.disable_semantic_obligation_stages),
                "semantic_obligation_type_state_disabled": bool(args.disable_semantic_obligation_stages),
                "semantic_target_scoring_disabled": bool(args.disable_semantic_obligation_stages),
                "semantic_scheduler_feedback_disabled": bool(args.disable_semantic_obligation_stages),
                "obligation_driven_candidate_generation_disabled": bool(args.disable_semantic_obligation_stages),
                "generic_fallback_scheduler_enabled": bool(args.disable_semantic_obligation_stages),
                "causal_cross_bb_constraint_recovery_enabled": True,
                "learned_pc_occurrence_scoped_constraints_disabled": bool(args.disable_scoped_replay_stages),
                "branch_snapshot_provenance_replay_disabled": bool(args.disable_scoped_replay_stages),
                "global_mmio_fallback_enabled": bool(args.disable_scoped_replay_stages),
                "contextual_isr_replay_disabled": bool(args.disable_contextual_isr_stages),
                "direct_stream_thread_replay_disabled": bool(args.disable_direct_stream_thread_stages),
                "stream_input_event_replay_disabled": bool(
                    args.disable_context_event_replay_stages
                    or args.disable_direct_stream_thread_stages
                ),
                "rtos_thread_entry_replay_disabled": bool(
                    args.disable_context_event_replay_stages
                    or args.disable_direct_stream_thread_stages
                ),
                "direct_call_continuation_disabled": bool(
                    args.disable_direct_stream_thread_stages
                ),
            },
            "argument_overrides": ablation_overrides,
            "fallback_policy": build_ablation_fallback_policy(args),
        })
    if loaded_run_profile is not None:
        runner.phase_metadata.setdefault("run_profile_input", {}).update({
            "profile_file": str(loaded_run_profile.get("profile_file") or args.run_profile),
            "profile_file_sha256": str(loaded_run_profile.get("profile_file_sha256") or ""),
            "applied_environment_entries": len(applied_run_profile_env),
            "applied_environment_keys": sorted(applied_run_profile_env)[:64],
        })
    progress_monitor = CoverageProgressMonitor(
        runner=runner,
        firmware=firmware,
        output_path=progress_jsonl_file,
        interval_seconds=progress_interval_seconds,
        started_at=start,
        stall_watchdog=stall_watchdog,
        wallclock_budget_seconds=total_wallclock_budget_seconds,
    )
    progress_monitor.start()

    scheduler_runtime = SchedulerRuntime(
        runner=runner,
        args=args,
        progress_monitor=progress_monitor,
        remaining_wallclock_seconds=remaining_wallclock_seconds,
        has_wallclock_budget=has_wallclock_budget,
        has_stage_wallclock_budget=has_stage_wallclock_budget,
        clamp_enabled_stage_seconds=clamp_enabled_stage_seconds,
        short_probe_mode=lambda: bool(
            total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600
        ),
        total_wallclock_budget_seconds=total_wallclock_budget_seconds,
        targeted_frontier_reserve_seconds=targeted_frontier_reserve_seconds,
    )
    runner.replay_executor.bind_runtime(scheduler_runtime)

    def run_stage(stage_name: str, func):
        return runner.replay_executor.run_stage(stage_name, func)

    def run_frontier_successor_replay_stage(
        stage_name: str,
        seconds: int,
        *,
        target_bbs: set[int] | None = None,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        reserve_after_seconds: int = 0,
    ) -> set[int]:
        return runner.replay_executor.run_frontier_successor(
            stage_name,
            seconds,
            target_bbs=target_bbs,
            max_tasks=max_tasks,
            max_targets=max_targets,
            reserve_after_seconds=reserve_after_seconds,
        )

    def semantic_entry_stage_reserve_seconds() -> int:
        """Reserve time for stages that mutate real state instead of CFG edges."""
        reserve = 0
        contextual_isr_reservoir_configured = (
            0
            if total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600
            else args.contextual_isr_reservoir_seconds
        )
        if args.contextual_isr_contexts != 0:
            reserve += max(0, int(contextual_isr_reservoir_configured))
            if not (total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600):
                reserve += max(0, int(args.semantic_vector_cleanup_seconds))
        reserve += max(0, int(args.stream_input_seconds))
        reserve += max(0, int(args.semantic_frontier_flush_seconds))
        reserve += max(0, int(args.direct_call_continuation_seconds))
        reserve += max(0, int(args.rtos_thread_entry_seconds))
        reserve += max(0, int(args.direct_call_summary_return_seconds))
        return reserve

    def clamp_before_semantic_entry_stages(configured_seconds: int) -> int:
        if configured_seconds <= 0:
            return 0
        reserve_after = targeted_frontier_reserve_seconds + semantic_entry_stage_reserve_seconds()
        return clamp_enabled_stage_seconds(configured_seconds, reserve_after_seconds=reserve_after)

    def run_direct_call_continuation_stage(
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        return runner.replay_executor.run_direct_call(
            stage_name,
            seconds,
            max_tasks=max_tasks,
            max_targets=max_targets,
            target_bbs=target_bbs,
        )

    def run_direct_call_summary_return_stage(
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        return runner.replay_executor.run_summary_return(
            stage_name,
            seconds,
            max_tasks=max_tasks,
            max_targets=max_targets,
            target_bbs=target_bbs,
        )

    def run_rtos_thread_entry_stage(
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        return runner.replay_executor.run_rtos_thread(
            stage_name,
            seconds,
            max_tasks=max_tasks,
            max_targets=max_targets,
            target_bbs=target_bbs,
        )

    def direct_call_budget_seconds(
        *,
        phase_name: str,
        phase_seconds: int,
        current_targets: set[int] | None = None,
    ) -> int:
        if phase_seconds <= 0 or args.direct_call_continuation_seconds <= 0:
            return 0
        target_count = len(current_targets or set())
        base_seconds = min(
            max(1, phase_seconds // 5),
            max(1, args.direct_call_continuation_seconds),
        )
        if target_count <= 0:
            return 0
        if target_count <= 8:
            base_seconds = min(base_seconds, max(1, phase_seconds // 8))
        elif target_count <= 32:
            base_seconds = min(base_seconds, max(1, phase_seconds // 6))

        previous_meta = runner.phase_metadata.get(f"{phase_name}_direct_call", {}) or {}
        if previous_meta:
            previous_new_bbs = int(previous_meta.get("new_bbs", 0) or 0)
            previous_target_hits = int(previous_meta.get("tasks_with_target_bbs", 0) or 0)
            previous_discovered = int(previous_meta.get("target_bbs_discovered", 0) or 0)
            previous_snapshots = int(previous_meta.get("remembered_branch_root_snapshots", 0) or 0)
            previous_dynamic_edges = int(previous_meta.get("dynamic_successor_edges_added", 0) or 0)
            if (
                previous_new_bbs <= 0
                and previous_target_hits <= 0
                and previous_discovered <= 0
                and previous_snapshots <= 0
                and previous_dynamic_edges <= 0
            ):
                return 0
            if previous_new_bbs <= 0 and previous_target_hits <= 0 and previous_discovered <= 0:
                base_seconds = min(base_seconds, max(1, phase_seconds // 10))
        return clamp_enabled_stage_seconds(base_seconds)

    def clamp_pre_targeted_stage_seconds(configured_seconds: int) -> int:
        return clamp_enabled_stage_seconds(
            configured_seconds,
            reserve_after_seconds=targeted_frontier_reserve_seconds,
        )

    def short_probe_mode() -> bool:
        return bool(total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600)

    def skip_stage(stage_name: str, reason: str) -> None:
        runner.replay_executor.skip_stage(stage_name, reason)

    def run_semantic_frontier_flush(
        stage_name: str,
        anchor_bbs: set[int],
        *,
        reserve_after_seconds: int = 0,
        include_targeted_reserve: bool = True,
    ) -> set[int]:
        if args.disable_semantic_obligation_stages:
            skip_stage(stage_name, "semantic_obligation_ablation_disabled")
            runner.phase_metadata.setdefault(stage_name, {}).update({
                "semantic_anchor_bbs": len(anchor_bbs or set()),
                "semantic_frontier_targets": 0,
            })
            return set()
        if args.disable_frontier_stages:
            skip_stage(stage_name, "frontier_ablation_disabled")
            return set()
        anchors = set(anchor_bbs or set())
        if args.semantic_frontier_flush_seconds <= 0:
            skip_stage(stage_name, "disabled")
            return set()
        if not anchors:
            skip_stage(stage_name, "no_semantic_anchor_bbs")
            runner.phase_metadata[stage_name].update({
                "semantic_anchor_bbs": 0,
                "semantic_frontier_targets": 0,
            })
            return set()
        targets = set(
            runner.frontier_target_bb_list_from_anchors(
                anchors,
                switch_only=args.semantic_frontier_flush_switch_only,
                max_targets=args.semantic_frontier_flush_max_targets,
            )
        )
        if not targets:
            skip_stage(stage_name, "no_anchor_frontier_targets")
            runner.phase_metadata[stage_name].update({
                "semantic_anchor_bbs": len(anchors),
                "semantic_frontier_targets": 0,
            })
            return set()
        seconds = clamp_enabled_stage_seconds(
            args.semantic_frontier_flush_seconds,
            reserve_after_seconds=(
                (targeted_frontier_reserve_seconds if include_targeted_reserve else 0)
                + max(0, int(reserve_after_seconds))
            ),
        )
        if seconds <= 0:
            skip_stage(stage_name, "no_budget_after_semantic_reserve")
            runner.phase_metadata[stage_name].update({
                "semantic_anchor_bbs": len(anchors),
                "semantic_frontier_targets": len(targets),
            })
            return set()
        covered = run_frontier_successor_replay_stage(
            stage_name,
            seconds,
            target_bbs=targets,
            max_tasks=args.semantic_frontier_flush_max_tasks,
            max_targets=args.semantic_frontier_flush_max_targets,
        )
        runner.phase_metadata.setdefault(stage_name, {}).update({
            "semantic_anchor_bbs": len(anchors),
            "semantic_frontier_targets": len(targets),
            "semantic_frontier_target_sample": [f"0x{bb:08x}" for bb in sorted(targets)[:32]],
        })
        return covered

    def run_vector_cleanup_targeted_stage(
        stage_name: str,
        seconds: int,
    ) -> tuple[set[int], list[dict[str, object]]]:
        if seconds <= 0 or args.contextual_isr_contexts == 0 or not has_wallclock_budget():
            skip_stage(stage_name, "disabled_or_no_budget")
            return set(), []
        stage_execution = runner.replay_executor.begin_stage(stage_name)
        vector_cleanup_tail_ratio = min(1.0, max(0.0, args.vector_cleanup_nearby_tail_ratio))
        vector_cleanup_head, vector_cleanup_tail = target_head_tail_quota(
            args.vector_cleanup_max_targets,
            1.0 - vector_cleanup_tail_ratio,
        )

        def vector_cleanup_candidates():
            vector_only_targets = runner.uncovered_bb_sets().get("vector_only", set())
            head_targets = runner.frontier_target_bb_list(
                min_uncovered_successors=max(1, args.vector_cleanup_min_uncovered_successors),
                switch_only=False,
                max_targets=vector_cleanup_head,
                target_filter=vector_only_targets,
            )
            tail_targets = []
            if vector_cleanup_tail > 0:
                tail_targets = runner.prioritized_target_bb_list(
                    vector_only_targets,
                    min_uncovered_successors=max(1, args.vector_cleanup_min_uncovered_successors),
                    switch_only=False,
                    include_nearby_uncovered=True,
                    nearby_limit=vector_cleanup_tail,
                    max_targets=vector_cleanup_tail,
                    exclude_targets=set(head_targets),
                    prefer_expandable=True,
                )
            priority_targets = prioritized_target_fallback(
                runner,
                vector_only_targets,
                switch_only=False,
                max_targets=vector_cleanup_head + vector_cleanup_tail,
                min_uncovered_successors=args.vector_cleanup_min_uncovered_successors,
                nearby_limit=max(8, vector_cleanup_tail or (vector_cleanup_head // 2)),
                exclude_targets=set(head_targets) | set(tail_targets),
            )
            ordered_targets = dedupe_target_order(
                head_targets,
                tail_targets,
                priority_targets or sorted(vector_only_targets),
            )
            expandable_targets = runner.expandable_target_bb_list(
                ordered_targets,
                require_uncovered_non_self_successor=False,
            )
            if expandable_targets:
                return expandable_targets
            return ordered_targets

        covered, rounds = run_targeted_stage(
            runner=runner,
            stage_name=stage_name,
            requested_rounds=args.vector_cleanup_rounds,
            auto_max_rounds=args.vector_cleanup_max_rounds,
            stale_round_limit=args.vector_cleanup_stale_rounds,
            min_progress_bbs=args.vector_cleanup_min_progress_bbs,
            max_targets=args.vector_cleanup_max_targets,
            cooldown_rounds=args.target_cooldown_rounds,
            candidate_supplier=vector_cleanup_candidates,
            executor=lambda phase_name, current_targets: runner.context_recovery.isr_reservoir(
                time_limit_seconds=clamp_enabled_stage_seconds(seconds),
                max_instructions=args.contextual_isr_reservoir_instructions,
                replay_timeout=args.contextual_isr_replay_timeout_us,
                max_tasks=args.contextual_isr_reservoir_max_tasks,
                max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                dynamic_frontier_seeding=True,
                context_snapshots=True,
                max_contexts=args.contextual_isr_contexts,
                include_reservoir_contexts=True,
                max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                max_isrs=args.contextual_isr_max_isrs,
                target_bbs=current_targets,
                prioritize_target_bbs=True,
                phase_name=phase_name,
            ),
            should_continue=lambda: has_wallclock_budget() and clamp_enabled_stage_seconds(seconds) > 0,
        )
        runner.replay_executor.finish_stage(
            stage_execution,
            covered,
            coverage_counted=False,
            extra={
                "stage_covered_bbs": len(covered),
                "rounds_executed": len(rounds),
                "global_covered_bbs": len(runner.global_coverage),
            },
        )
        return covered, rounds

    report = None
    try:
        baseline = run_stage(
            "baseline",
            lambda: runner.run_baseline(
                max_instructions=args.baseline_instructions,
                quiescence_bb_limit=args.baseline_quiescence_bbs,
                enable_function_summaries=args.enable_baseline_function_summaries,
                timeout_seconds=(
                    float(args.baseline_timeout_seconds)
                    if float(args.baseline_timeout_seconds or 0.0) > 0.0
                    else None
                ),
            ),
        )
        # r40 P2/C8：cycle2 k.5 的「基线后装到常驻 self.emulator」安装点已
        # 删除——其后相位走 fresh_emulator=True，钩子挂在不执行的实例上
        # （smokeZ 实测 instructions=0 / delivery_count=0）。主流程投递改由
        # LSGEMU_IRQ_DELIVERY_PHASES 相位白名单装进各相位自建 temp emulator
        # 的重放原语（historical_runner._execute_replay_impl，样板
        # _dfs_flip_replay_once）；缺省空白名单 = 与 k.5 行为逐位一致（即恒
        # 零安装，而非死安装）。
        semantic_entry_prefix = set()
        semantic_entry_prefix_seconds = clamp_pre_targeted_stage_seconds(
            args.semantic_entry_prefix_seconds
        )
        if semantic_entry_prefix_seconds > 0 and has_wallclock_budget():
            semantic_entry_prefix = run_stage(
                "semantic_entry_prefix",
                lambda: runner.run_baseline(
                    max_instructions=args.semantic_entry_prefix_instructions,
                    quiescence_bb_limit=args.semantic_entry_prefix_quiescence_bbs,
                    # r34 Q2/S2：canonical 恒为 True；置 0 得到「同 argv、只关摘要」
                    # 的对照臂，用来判定 1395 的深度是否依赖跳过函数体。
                    enable_function_summaries=(
                        str(
                            os.environ.get(
                                "LSGEMU_SEMANTIC_PREFIX_FUNCTION_SUMMARIES", "1"
                            )
                        )
                        .strip()
                        .lower()
                        not in {"0", "false", "no", "off"}
                    ),
                    phase_name="semantic_entry_prefix",
                    fresh_emulator=True,
                    timeout_seconds=semantic_entry_prefix_seconds,
                ),
            )
        else:
            skip_stage("semantic_entry_prefix", "disabled_or_no_budget")
        if not args.skip_cold_isr:
            isr = run_stage(
                "isr",
                lambda: runner.context_recovery.cold_isr(
                    max_instructions=args.isr_instructions,
                    count_coverage=args.count_cold_isr_coverage,
                ),
            )
        else:
            isr = set()
            skip_stage("isr", "skip_cold_isr")
        contextual_isr = set()
        if args.contextual_isr_contexts != 0 and has_wallclock_budget():
            contextual_isr = run_stage(
                "contextual_isr",
                lambda: runner.context_recovery.contextual_isr(
                    max_contexts=args.contextual_isr_contexts,
                    max_isrs=args.contextual_isr_max_isrs,
                    max_instructions=args.contextual_isr_instructions,
                    replay_timeout=args.contextual_isr_replay_timeout_us,
                    time_limit_seconds=clamp_optional_stage_limit(args.contextual_isr_time_seconds),
                    stage_name="contextual_isr",
                ),
            )
        else:
            skip_stage("contextual_isr", "disabled_or_no_budget")
        if args.constraint_seed_mode != "none" and args.constraint_seed_activation == "after-main":
            seeded_constraint_count = merge_learned_constraints_into(
                firmware,
                constraint_file,
                learned_mode=args.constraint_seed_mode,
                learned_roots=learned_roots,
            )
            progress_monitor.snapshot(
                "constraint_seed_merged",
                stage="constraint_seed_merge",
                extra={"seeded_constraint_count": seeded_constraint_count},
            )

        interleaved_total_seconds = remaining_wallclock_seconds()
        if interleaved_total_seconds is not None:
            dynamic_post_reserve_seconds = reserved_post_interleaved_seconds
            minimum_interleaved_seconds = short_minimum_interleaved_seconds()
            if minimum_interleaved_seconds > 0:
                dynamic_post_reserve_seconds = min(
                    max(0, dynamic_post_reserve_seconds),
                    max(0, int(interleaved_total_seconds) - minimum_interleaved_seconds),
                )
            interleaved_total_seconds = max(0, interleaved_total_seconds - dynamic_post_reserve_seconds)
        else:
            dynamic_post_reserve_seconds = reserved_post_interleaved_seconds
        runner.phase_metadata.setdefault("branch_interleaved_budget", {}).update({
            "coverage_counted": False,
            "reserved_post_interleaved_seconds_start": int(reserved_post_interleaved_seconds),
            "reserved_post_interleaved_seconds_dynamic": int(dynamic_post_reserve_seconds),
            "minimum_interleaved_seconds": int(short_minimum_interleaved_seconds()),
            "remaining_before_interleaved": remaining_wallclock_seconds(),
            "interleaved_total_seconds": interleaved_total_seconds,
        })
        interleaved_targeted_prefix_snapshot_seed_limit = (
            args.targeted_prefix_snapshot_seed_limit
            if args.interleaved_targeted_prefix_snapshot_seed_limit < 0
            else args.interleaved_targeted_prefix_snapshot_seed_limit
        )
        interleaved_stage = runner.replay_executor.begin_stage(
            "interleaved",
            extra={"interleaved_total_time_budget_seconds": interleaved_total_seconds},
        )
        with temporary_env({
            "LSGEMU_TARGETED_PREFIX_SNAPSHOT_SEED_LIMIT": str(
                max(0, interleaved_targeted_prefix_snapshot_seed_limit)
            ),
        }):
            interleaved = runner.run_interleaved_reservoir_mmio_exploration(
                total_time_seconds=interleaved_total_seconds,
                reservoir_round_seconds=args.reservoir_round_seconds,
                mmio_round_seconds=args.mmio_round_seconds,
                frontier_round_seconds=args.interleaved_frontier_round_seconds,
                frontier_max_targets=args.interleaved_frontier_max_targets,
                frontier_min_uncovered_successors=args.interleaved_frontier_min_uncovered_successors,
                frontier_switch_only=args.interleaved_frontier_switch_only,
                max_rounds=max(0, args.max_rounds),
                stale_round_limit=args.stale_round_limit,
                min_progress_bbs=args.min_progress_bbs,
                reservoir_max_tasks=args.reservoir_max_tasks,
                reservoir_replay_instructions=args.reservoir_replay_instructions,
                reservoir_replay_timeout=args.reservoir_replay_timeout_us,
                mmio_max_branches=args.mmio_max_branches,
                mmio_replay_instructions=args.mmio_replay_instructions,
                mmio_replay_timeout=args.mmio_replay_timeout_us,
                mmio_snapshot_variants_per_branch=args.mmio_snapshot_variants_per_branch,
                adaptive_plateau_round_limit=args.interleaved_adaptive_plateau_rounds,
                adaptive_plateau_min_rounds=args.interleaved_adaptive_plateau_min_rounds,
            )
        runner.replay_executor.finish_stage(
            interleaved_stage,
            interleaved,
            coverage_counted=False,
            extra={
                "stage_covered_bbs": len(interleaved),
                "global_covered_bbs": len(runner.global_coverage),
            },
        )
        frontier_successor_replay_early = run_frontier_successor_replay_stage(
            "frontier_successor_replay_early",
            clamp_before_semantic_entry_stages(args.frontier_successor_replay_seconds),
        )
        early_switch_frontier_reservoir = set()
        if switch_frontier_enabled and has_wallclock_budget():
            early_switch_candidate_limit = max(
                args.switch_frontier_max_targets,
                args.switch_frontier_max_targets
                * max(1, args.switch_frontier_candidate_pool_multiplier),
            )
            early_switch_summary = runner.uncovered_coverage_summary()
            early_switch_cold_targets = parse_unstarted_frontier_targets(
                early_switch_summary,
                early_switch_candidate_limit,
                switch_only=True,
            )
            if not early_switch_cold_targets:
                # uncovered_coverage_summary() may synthesize newly replayable
                # dynamic dispatch roots as a side effect. Re-read once so the
                # just-materialized TBB/TBH/LDRPC frontiers are schedulable in
                # the same stage instead of being delayed to the next campaign.
                early_switch_summary = runner.uncovered_coverage_summary()
                early_switch_cold_targets = parse_unstarted_frontier_targets(
                    early_switch_summary,
                    early_switch_candidate_limit,
                    switch_only=True,
                )
            early_switch_targets = dedupe_target_order(
                early_switch_cold_targets,
                runner.hotspot_frontier_target_bb_list(
                    switch_only=True,
                    max_predecessors=max(1, args.hotspot_frontier_predecessors),
                    max_targets=early_switch_candidate_limit,
                    min_uncovered_successors=max(
                        1,
                        args.switch_frontier_min_uncovered_successors,
                    ),
                ),
                runner.frontier_target_bb_list(
                    min_uncovered_successors=max(
                        1,
                        args.switch_frontier_min_uncovered_successors,
                    ),
                    switch_only=True,
                    max_targets=early_switch_candidate_limit,
                ),
            )
            early_switch_seconds = clamp_pre_targeted_stage_seconds(
                min(
                    max(0, args.switch_frontier_round_seconds),
                    max(
                        4,
                        max(0, args.semantic_frontier_flush_seconds),
                    ),
                )
            )
            if early_switch_targets and early_switch_seconds > 0:
                with temporary_env({
                    "LSGEMU_FRONTIER_ROOT_SEED": "1",
                    "LSGEMU_RUNTIME_FRONTIER_ROOT_SEED_MIN_TASKS": "0",
                    "LSGEMU_TARGETED_SNAPSHOT_FRONTIER_BOOTSTRAP": "1",
                    "LSGEMU_TARGETED_SWITCH_ROOT_SNAPSHOT_REPLAY": "1",
                }):
                    early_switch_frontier_reservoir = run_stage(
                        "early_switch_frontier_reservoir",
                        lambda: runner.run_reservoir_branch_exploration(
                            time_limit_seconds=early_switch_seconds,
                            replay_instructions=args.reservoir_replay_instructions,
                            replay_timeout=args.reservoir_replay_timeout_us,
                            max_tasks=args.reservoir_max_tasks,
                            max_tasks_per_root=max(args.targeted_max_tasks_per_root, 16),
                            target_bbs=set(early_switch_targets[:args.switch_frontier_max_targets]),
                            prioritize_target_bbs=True,
                            continue_target_merges=True,
                            target_probe_bbs=min(
                                512,
                                max(64, args.reservoir_replay_instructions // 100),
                            ),
                            phase_name="early_switch_frontier_reservoir",
                            reset_state=True,
                        ),
                    )
                runner.phase_metadata.setdefault("early_switch_frontier_reservoir", {}).update({
                    "early_switch_frontier_targets": len(early_switch_targets),
                    "early_switch_frontier_target_sample": [
                        f"0x{bb:08x}"
                        for bb in sorted(early_switch_targets)[:32]
                    ],
                })
            else:
                skip_stage(
                    "early_switch_frontier_reservoir",
                    "no_switch_targets_or_no_budget",
                )
                runner.phase_metadata.setdefault("early_switch_frontier_reservoir", {}).update({
                    "early_switch_frontier_targets": len(early_switch_targets),
                })
        else:
            skip_stage(
                "early_switch_frontier_reservoir",
                "disabled_or_no_budget",
            )
        early_frontier_reservoir = set()
        if frontier_targeted_enabled and has_wallclock_budget():
            early_frontier_candidate_limit = max(
                args.frontier_max_targets,
                args.frontier_max_targets
                * max(1, args.frontier_candidate_pool_multiplier),
            )
            early_frontier_summary = runner.uncovered_coverage_summary()
            early_frontier_cold_targets = parse_unstarted_frontier_targets(
                early_frontier_summary,
                early_frontier_candidate_limit,
                switch_only=False,
            )
            if not early_frontier_cold_targets:
                early_frontier_summary = runner.uncovered_coverage_summary()
                early_frontier_cold_targets = parse_unstarted_frontier_targets(
                    early_frontier_summary,
                    early_frontier_candidate_limit,
                    switch_only=False,
                )
            early_frontier_uncovered = runner.uncovered_bb_sets().get("uncovered", set())
            early_frontier_targets = dedupe_target_order(
                early_frontier_cold_targets,
                runner.hotspot_frontier_target_bb_list(
                    switch_only=False,
                    max_predecessors=max(1, args.hotspot_frontier_predecessors),
                    max_targets=early_frontier_candidate_limit,
                    min_uncovered_successors=max(
                        1,
                        args.frontier_min_uncovered_successors,
                    ),
                ),
                runner.frontier_target_bb_list(
                    min_uncovered_successors=max(
                        1,
                        args.frontier_min_uncovered_successors,
                    ),
                    switch_only=False,
                    max_targets=early_frontier_candidate_limit,
                ),
                prioritized_target_fallback(
                    runner,
                    early_frontier_uncovered,
                    switch_only=False,
                    max_targets=early_frontier_candidate_limit,
                    min_uncovered_successors=args.frontier_min_uncovered_successors,
                    nearby_limit=max(16, early_frontier_candidate_limit // 4),
                    exclude_targets=set(early_frontier_cold_targets),
                ),
            )
            early_frontier_seconds = clamp_pre_targeted_stage_seconds(
                min(
                    max(0, args.frontier_round_seconds),
                    max(
                        4,
                        max(0, args.semantic_frontier_flush_seconds),
                    ),
                )
            )
            if early_frontier_targets and early_frontier_seconds > 0:
                with temporary_env({
                    "LSGEMU_FRONTIER_ROOT_SEED": "1",
                    "LSGEMU_RUNTIME_FRONTIER_ROOT_SEED_MIN_TASKS": "0",
                    "LSGEMU_TARGETED_SNAPSHOT_FRONTIER_BOOTSTRAP": "1",
                    "LSGEMU_TARGETED_SWITCH_ROOT_SNAPSHOT_REPLAY": "1",
                    "LSGEMU_TARGETED_DIRECT_ROOT_FOCUS": "0",
                }):
                    early_frontier_reservoir = run_stage(
                        "early_frontier_reservoir",
                        lambda: runner.run_reservoir_branch_exploration(
                            time_limit_seconds=early_frontier_seconds,
                            replay_instructions=args.reservoir_replay_instructions,
                            replay_timeout=args.reservoir_replay_timeout_us,
                            max_tasks=args.reservoir_max_tasks,
                            max_tasks_per_root=max(args.targeted_max_tasks_per_root, 12),
                            target_bbs=set(early_frontier_targets[:args.frontier_max_targets]),
                            prioritize_target_bbs=True,
                            continue_target_merges=True,
                            target_probe_bbs=min(
                                512,
                                max(64, args.reservoir_replay_instructions // 100),
                            ),
                            phase_name="early_frontier_reservoir",
                            reset_state=True,
                        ),
                    )
                runner.phase_metadata.setdefault("early_frontier_reservoir", {}).update({
                    "early_frontier_targets": len(early_frontier_targets),
                    "early_frontier_target_sample": [
                        f"0x{bb:08x}"
                        for bb in sorted(early_frontier_targets)[:32]
                    ],
                })
            else:
                skip_stage(
                    "early_frontier_reservoir",
                    "no_frontier_targets_or_no_budget",
                )
                runner.phase_metadata.setdefault("early_frontier_reservoir", {}).update({
                    "early_frontier_targets": len(early_frontier_targets),
                })
        else:
            skip_stage(
                "early_frontier_reservoir",
                "disabled_or_no_budget",
            )
        semantic_frontier_flush = set()
        early_summary_return = set()
        early_summary_return_seconds = clamp_pre_targeted_stage_seconds(
            max(0, args.direct_call_summary_return_seconds // 2)
        )
        if early_summary_return_seconds > 0:
            early_summary_targets = (
                set()
                if args.disable_frontier_stages
                else set(
                    runner.frontier_target_bb_list(
                        max_targets=max(args.direct_call_summary_return_max_targets, args.semantic_frontier_flush_max_targets),
                        include_nearby_uncovered=True,
                        nearby_limit=max(32, args.direct_call_summary_return_max_targets // 2),
                    )
                )
            )
            early_summary_return = run_direct_call_summary_return_stage(
                "direct_call_summary_return_early",
                early_summary_return_seconds,
                max_tasks=max(args.direct_call_summary_return_max_tasks, 128),
                max_targets=max(args.direct_call_summary_return_max_targets, 128),
                target_bbs=early_summary_targets or None,
            )
            runner.phase_metadata.setdefault("direct_call_summary_return_early", {}).update({
                "semantic_frontier_target_suggestions": bool(early_summary_targets),
                "semantic_frontier_target_suggestion_count": len(early_summary_targets),
                "semantic_frontier_target_suggestion_sample": [
                    f"0x{bb:08x}" for bb in sorted(early_summary_targets)[:32]
                ],
            })
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "direct_call_summary_return_early_frontier_flush",
                    early_summary_return,
                    reserve_after_seconds=(
                        max(0, args.stream_input_seconds)
                        + max(0, args.direct_call_continuation_seconds)
                        + max(0, args.rtos_thread_entry_seconds)
                        + max(0, args.direct_call_summary_return_seconds)
                    ),
                    include_targeted_reserve=False,
                )
            )
        else:
            skip_stage("direct_call_summary_return_early", "disabled_or_no_budget")
        contextual_isr_reservoir = set()
        contextual_isr_reservoir_configured_seconds = (
            0
            if total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600
            else args.contextual_isr_reservoir_seconds
        )
        contextual_isr_reservoir_seconds = clamp_pre_targeted_stage_seconds(
            contextual_isr_reservoir_configured_seconds
        )
        if args.contextual_isr_contexts != 0 and contextual_isr_reservoir_seconds > 0 and has_wallclock_budget():
            contextual_isr_reservoir = run_stage(
                "contextual_isr_reservoir",
                lambda: runner.context_recovery.isr_reservoir(
                    time_limit_seconds=contextual_isr_reservoir_seconds,
                    max_instructions=args.contextual_isr_reservoir_instructions,
                    replay_timeout=args.contextual_isr_replay_timeout_us,
                    max_tasks=args.contextual_isr_reservoir_max_tasks,
                    max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                    dynamic_frontier_seeding=True,
                    context_snapshots=True,
                    max_contexts=args.contextual_isr_contexts,
                    include_reservoir_contexts=True,
                    max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                    max_isrs=args.contextual_isr_max_isrs,
                    phase_name="contextual_isr_reservoir",
                ),
            )
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "contextual_isr_reservoir_frontier_flush",
                    contextual_isr_reservoir,
                    reserve_after_seconds=(
                        max(0, args.stream_input_seconds)
                        + max(0, args.direct_call_continuation_seconds)
                        + max(0, args.rtos_thread_entry_seconds)
                        + max(0, args.direct_call_summary_return_seconds)
                    ),
                    include_targeted_reserve=True,
                )
            )
        else:
            skip_stage("contextual_isr_reservoir", "disabled_or_no_budget")
            skip_stage("contextual_isr_reservoir_frontier_flush", "contextual_isr_reservoir_skipped")
        semantic_vector_cleanup = set()
        semantic_vector_cleanup_rounds = []
        semantic_vector_cleanup_configured_seconds = (
            0
            if total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 600
            else args.semantic_vector_cleanup_seconds
        )
        semantic_vector_cleanup_seconds = clamp_enabled_stage_seconds(
            semantic_vector_cleanup_configured_seconds,
            reserve_after_seconds=(
                targeted_frontier_reserve_seconds
                + max(0, args.stream_input_seconds)
                + max(0, args.semantic_frontier_flush_seconds)
                + max(0, args.direct_call_continuation_seconds)
                + max(0, args.rtos_thread_entry_seconds)
                + max(0, args.direct_call_summary_return_seconds)
            ),
        )
        if semantic_vector_cleanup_seconds > 0:
            semantic_vector_cleanup, semantic_vector_cleanup_rounds = run_vector_cleanup_targeted_stage(
                "semantic_vector_cleanup",
                semantic_vector_cleanup_seconds,
            )
        else:
            skip_stage("semantic_vector_cleanup", "disabled_or_no_budget")
        stream_input_replay = set()
        stream_input_seconds = clamp_pre_targeted_stage_seconds(args.stream_input_seconds)
        if stream_input_seconds > 0 and has_wallclock_budget():
            stream_input_replay = run_stage(
                "stream_input_replay",
                lambda: runner.context_recovery.stream_input(
                    time_limit_seconds=stream_input_seconds,
                    max_profiles=args.stream_input_max_profiles,
                    max_contexts=args.stream_input_max_contexts,
                    max_seeds=args.stream_input_max_seeds,
                    max_tasks=args.stream_input_max_tasks,
                    max_instructions=args.stream_input_instructions,
                    replay_timeout=args.stream_input_replay_timeout_us,
                    no_new_bb_limit=args.stream_input_no_new_bbs,
                    phase_name="stream_input_replay",
                ),
            )
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "stream_input_frontier_flush",
                    stream_input_replay,
                    reserve_after_seconds=(
                        max(0, args.direct_call_continuation_seconds)
                        + max(0, args.rtos_thread_entry_seconds)
                        + max(0, args.direct_call_summary_return_seconds)
                    ),
                    include_targeted_reserve=True,
                )
            )
        else:
            skip_stage("stream_input_replay", "disabled_or_no_budget")
            skip_stage("stream_input_frontier_flush", "stream_input_replay_skipped")

        def run_late_stream_input_stage(
            stage_name: str,
            configured_seconds: int,
            *,
            reserve_after_seconds: int | None = None,
            prefer_stream_summary_contexts: bool = False,
        ) -> set[int]:
            if reserve_after_seconds is None:
                reserve_after_seconds = (
                    max(0, args.direct_call_continuation_seconds)
                    + max(0, args.direct_call_summary_return_seconds)
                )
            seconds = clamp_enabled_stage_seconds(
                max(0, configured_seconds),
                reserve_after_seconds=max(0, int(reserve_after_seconds)),
            )
            if seconds <= 0 or not has_wallclock_budget():
                skip_stage(stage_name, "disabled_or_no_budget")
                return set()
            return run_stage(
                stage_name,
                lambda: runner.context_recovery.stream_input(
                    time_limit_seconds=seconds,
                    max_profiles=max(args.stream_input_max_profiles, 8),
                    max_contexts=max(args.stream_input_max_contexts, 32),
                    max_seeds=args.stream_input_max_seeds,
                    max_tasks=max(args.stream_input_max_tasks, 256),
                    max_instructions=args.stream_input_instructions,
                    replay_timeout=args.stream_input_replay_timeout_us,
                    no_new_bb_limit=args.stream_input_no_new_bbs,
                    phase_name=stage_name,
                    prefer_stream_summary_contexts=prefer_stream_summary_contexts,
                ),
            )

        direct_call_continuation = set()
        direct_call_continuation_seconds = clamp_pre_targeted_stage_seconds(
            args.direct_call_continuation_seconds
        )
        if direct_call_continuation_seconds > 0:
            direct_call_frontier_targets = (
                set()
                if args.disable_frontier_stages
                else set(
                    runner.frontier_target_bb_list(
                        max_targets=max(args.direct_call_continuation_max_targets, args.semantic_frontier_flush_max_targets),
                        include_nearby_uncovered=True,
                        nearby_limit=max(16, args.direct_call_continuation_max_targets // 2),
                    )
                )
            )
            direct_call_continuation = run_direct_call_continuation_stage(
                "direct_call_continuation",
                direct_call_continuation_seconds,
            )
            runner.phase_metadata.setdefault("direct_call_continuation", {}).update({
                "semantic_frontier_target_suggestions": bool(direct_call_frontier_targets),
                "semantic_frontier_target_suggestion_count": len(direct_call_frontier_targets),
                "semantic_frontier_target_suggestion_sample": [
                    f"0x{bb:08x}" for bb in sorted(direct_call_frontier_targets)[:32]
                ],
            })
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "direct_call_continuation_frontier_flush",
                    direct_call_continuation,
                    reserve_after_seconds=(
                        max(0, args.rtos_thread_entry_seconds)
                        + max(0, args.direct_call_summary_return_seconds)
                    ),
                    include_targeted_reserve=True,
                )
            )
        else:
            skip_stage("direct_call_continuation", "disabled_or_no_budget")
            skip_stage("direct_call_continuation_frontier_flush", "direct_call_continuation_skipped")
        rtos_thread_entry = set()
        rtos_thread_entry_seconds = clamp_pre_targeted_stage_seconds(args.rtos_thread_entry_seconds)
        if rtos_thread_entry_seconds > 0:
            rtos_thread_entry = run_rtos_thread_entry_stage(
                "rtos_thread_entry",
                rtos_thread_entry_seconds,
            )
        else:
            skip_stage("rtos_thread_entry", "disabled_or_no_budget")
        direct_call_summary_return = set()
        direct_call_summary_return_seconds = clamp_pre_targeted_stage_seconds(
            args.direct_call_summary_return_seconds
        )
        if direct_call_summary_return_seconds > 0:
            direct_call_summary_targets = (
                set()
                if args.disable_frontier_stages
                else set(
                    runner.frontier_target_bb_list(
                        max_targets=max(args.direct_call_summary_return_max_targets, args.semantic_frontier_flush_max_targets),
                        include_nearby_uncovered=True,
                        nearby_limit=max(16, args.direct_call_summary_return_max_targets // 2),
                    )
                )
            )
            direct_call_summary_return = run_direct_call_summary_return_stage(
                "direct_call_summary_return",
                direct_call_summary_return_seconds,
            )
            runner.phase_metadata.setdefault("direct_call_summary_return", {}).update({
                "semantic_frontier_target_suggestions": bool(direct_call_summary_targets),
                "semantic_frontier_target_suggestion_count": len(direct_call_summary_targets),
                "semantic_frontier_target_suggestion_sample": [
                    f"0x{bb:08x}" for bb in sorted(direct_call_summary_targets)[:32]
                ],
            })
        else:
            skip_stage("direct_call_summary_return", "disabled_or_no_budget")
        stream_input_after_threads = set()
        after_thread_stream_protected_seconds = (
            max(0, min(8, frontier_cycle_tail_reserve_seconds // 4))
            if short_probe_mode()
            else frontier_cycle_tail_reserve_seconds
        )
        after_thread_stream_seconds = clamp_enabled_stage_seconds(
            max(0, min(args.stream_input_seconds, max(6, args.stream_input_seconds // 2))),
            reserve_after_seconds=(
                0 if short_probe_mode() else targeted_frontier_reserve_seconds
            )
            + after_thread_stream_protected_seconds,
        )
        if (
            short_probe_mode()
            and after_thread_stream_seconds <= 0
            and args.stream_input_seconds > 0
            and has_wallclock_budget(min_seconds=2)
        ):
            after_thread_stream_seconds = min(
                max(2, int(args.stream_input_seconds // 2) or 2),
                max(2, remaining_wallclock_seconds() or 2),
            )
        if after_thread_stream_seconds > 0 and has_wallclock_budget():
            stream_input_after_threads = run_late_stream_input_stage(
                "stream_input_replay_after_threads",
                after_thread_stream_seconds,
                reserve_after_seconds=after_thread_stream_protected_seconds,
                prefer_stream_summary_contexts=True,
            )
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "stream_input_after_threads_frontier_flush",
                    stream_input_after_threads,
                    reserve_after_seconds=after_thread_stream_protected_seconds,
                    include_targeted_reserve=not short_probe_mode(),
                )
            )
            runner.phase_metadata.setdefault("stream_input_replay_after_threads", {}).update({
                "protected_after_seconds": int(after_thread_stream_protected_seconds),
                "scheduled_after": "rtos_thread_entry_and_direct_call_summary_return",
            })
        else:
            skip_stage("stream_input_replay_after_threads", "disabled_or_no_budget")
            skip_stage("stream_input_after_threads_frontier_flush", "stream_input_replay_after_threads_skipped")
            runner.phase_metadata.setdefault("stream_input_replay_after_threads", {}).update({
                "protected_after_seconds": int(after_thread_stream_protected_seconds),
                "scheduled_after": "rtos_thread_entry_and_direct_call_summary_return",
            })
        direct_call_post_thread_seconds = min(
            direct_call_continuation_seconds,
            max(0, direct_call_summary_return_seconds // 2),
        )
        direct_call_post_thread_stage_seconds = (
            clamp_enabled_stage_seconds(
                direct_call_post_thread_seconds,
                reserve_after_seconds=(
                    max(0, min(6, frontier_cycle_tail_reserve_seconds // 6))
                    if short_probe_mode()
                    else targeted_frontier_reserve_seconds
                ),
            )
            if direct_call_post_thread_seconds > 0
            else 0
        )
        if direct_call_post_thread_stage_seconds > 0:
            direct_call_post_thread_targets = (
                set()
                if args.disable_frontier_stages
                else set(
                    runner.frontier_target_bb_list(
                        max_targets=max(args.direct_call_continuation_max_targets, 128),
                        include_nearby_uncovered=True,
                        nearby_limit=max(32, args.direct_call_continuation_max_targets // 2),
                    )
                )
            )
            direct_call_continuation.update(
                run_direct_call_continuation_stage(
                    "direct_call_continuation_post_thread",
                    direct_call_post_thread_stage_seconds,
                    max_tasks=max(args.direct_call_continuation_max_tasks, 128),
                    max_targets=max(args.direct_call_continuation_max_targets, 128),
                )
            )
            runner.phase_metadata.setdefault("direct_call_continuation_post_thread", {}).update({
                "semantic_frontier_target_suggestions": bool(direct_call_post_thread_targets),
                "semantic_frontier_target_suggestion_count": len(direct_call_post_thread_targets),
                "semantic_frontier_target_suggestion_sample": [
                    f"0x{bb:08x}" for bb in sorted(direct_call_post_thread_targets)[:32]
                ],
            })
        else:
            skip_stage("direct_call_continuation_post_thread", "disabled_or_no_budget")

        semantic_frontier_drain = set()
        semantic_frontier_drain_targets = (
            set()
            if args.disable_frontier_stages
            else set(
                runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                    switch_only=False,
                    max_targets=max(args.semantic_frontier_flush_max_targets, args.frontier_max_targets),
                )
            )
        )
        semantic_frontier_drain_protected_seconds = (
            max(0, min(6, frontier_cycle_tail_reserve_seconds // 6))
            if short_probe_mode()
            else targeted_frontier_reserve_seconds + frontier_cycle_tail_reserve_seconds
        )
        semantic_frontier_drain_configured_seconds = max(
            max(0, args.semantic_frontier_flush_seconds),
            max(0, args.frontier_successor_replay_tail_seconds),
        )
        if short_probe_mode():
            semantic_frontier_drain_configured_seconds = min(
                semantic_frontier_drain_configured_seconds,
                max(6, args.semantic_frontier_flush_seconds),
            )
        semantic_frontier_drain_seconds = clamp_enabled_stage_seconds(
            semantic_frontier_drain_configured_seconds,
            reserve_after_seconds=semantic_frontier_drain_protected_seconds,
        )
        if args.disable_semantic_frontier_drain_stage:
            skip_stage(
                "semantic_frontier_drain",
                "semantic_obligation_ablation_disabled",
            )
            runner.phase_metadata.setdefault("semantic_frontier_drain", {}).update({
                "semantic_frontier_target_count": len(semantic_frontier_drain_targets),
                "configured_seconds": int(semantic_frontier_drain_configured_seconds),
                "protected_after_seconds": int(semantic_frontier_drain_protected_seconds),
            })
        elif semantic_frontier_drain_targets and semantic_frontier_drain_seconds > 0:
            semantic_frontier_drain = run_frontier_successor_replay_stage(
                "semantic_frontier_drain",
                semantic_frontier_drain_seconds,
                target_bbs=semantic_frontier_drain_targets,
                max_tasks=max(args.semantic_frontier_flush_max_tasks, args.frontier_successor_replay_max_tasks // 4),
                max_targets=max(args.semantic_frontier_flush_max_targets, args.frontier_max_targets),
            )
            runner.phase_metadata.setdefault("semantic_frontier_drain", {}).update({
                "semantic_frontier_target_count": len(semantic_frontier_drain_targets),
                "semantic_frontier_target_sample": [
                    f"0x{bb:08x}" for bb in sorted(semantic_frontier_drain_targets)[:32]
                ],
                "configured_seconds": int(semantic_frontier_drain_configured_seconds),
                "protected_after_seconds": int(semantic_frontier_drain_protected_seconds),
            })
        else:
            skip_stage(
                "semantic_frontier_drain",
                "no_targets_or_no_budget",
            )
            runner.phase_metadata.setdefault("semantic_frontier_drain", {}).update({
                "semantic_frontier_target_count": len(semantic_frontier_drain_targets),
                "configured_seconds": int(semantic_frontier_drain_configured_seconds),
                "protected_after_seconds": int(semantic_frontier_drain_protected_seconds),
            })
        late_stream_after_semantic = set()
        late_stream_configured_seconds = max(0, min(args.stream_input_seconds, max(10, args.stream_input_seconds // 2)))
        if late_stream_configured_seconds > 0:
            late_stream_after_semantic = run_late_stream_input_stage(
                "stream_input_replay_late",
                late_stream_configured_seconds,
                reserve_after_seconds=(
                    targeted_frontier_reserve_seconds
                    + max(0, args.direct_call_continuation_seconds)
                    + max(0, args.direct_call_summary_return_seconds)
                ),
            )
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "stream_input_late_frontier_flush",
                    late_stream_after_semantic,
                    reserve_after_seconds=(
                        max(0, args.direct_call_continuation_seconds)
                        + max(0, args.direct_call_summary_return_seconds)
                    ),
                    include_targeted_reserve=True,
                )
            )
        else:
            skip_stage("stream_input_replay_late", "disabled_or_no_budget")
            skip_stage("stream_input_late_frontier_flush", "stream_input_replay_late_skipped")
        frontier_successor_replay = run_frontier_successor_replay_stage(
            "frontier_successor_replay",
            clamp_pre_targeted_stage_seconds(args.frontier_successor_replay_seconds),
            reserve_after_seconds=(
                max(0, min(args.stream_input_seconds, max(10, args.stream_input_seconds // 2)))
                + (0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds))
            ),
        )
        post_frontier_stream_input = set()
        post_frontier_stream_protected_seconds = (
            0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds)
        )
        post_frontier_stream_seconds = clamp_enabled_stage_seconds(
            max(0, min(args.stream_input_seconds, max(10, args.stream_input_seconds // 2))),
            reserve_after_seconds=(
                targeted_frontier_reserve_seconds
                + post_frontier_stream_protected_seconds
            ),
        )
        if post_frontier_stream_seconds > 0 and has_wallclock_budget():
            post_frontier_stream_input = run_late_stream_input_stage(
                "stream_input_replay_post_frontier",
                post_frontier_stream_seconds,
                reserve_after_seconds=post_frontier_stream_protected_seconds,
            )
            semantic_frontier_flush.update(
                run_semantic_frontier_flush(
                    "stream_input_post_frontier_flush",
                    post_frontier_stream_input,
                    reserve_after_seconds=post_frontier_stream_protected_seconds,
                    include_targeted_reserve=True,
                )
            )
            runner.phase_metadata.setdefault("stream_input_replay_post_frontier", {}).update({
                "protected_after_seconds": int(post_frontier_stream_protected_seconds),
            })
        else:
            skip_stage("stream_input_replay_post_frontier", "disabled_or_no_budget")
            skip_stage("stream_input_post_frontier_flush", "stream_input_replay_post_frontier_skipped")
            runner.phase_metadata.setdefault("stream_input_replay_post_frontier", {}).update({
                "protected_after_seconds": int(post_frontier_stream_protected_seconds),
            })
        contextual_isr_frontier_targeted = set()
        contextual_isr_frontier_rounds = []
        contextual_isr_frontier_seconds = clamp_pre_targeted_stage_seconds(
            args.contextual_isr_frontier_seconds
        )
        if args.contextual_isr_contexts != 0 and contextual_isr_frontier_seconds > 0 and has_wallclock_budget():
            contextual_frontier_stage = runner.replay_executor.begin_stage(
                "contextual_isr_frontier_targeted"
            )
            contextual_isr_tail_ratio = min(1.0, max(0.0, args.contextual_isr_frontier_nearby_tail_ratio))
            contextual_isr_candidate_pool_limit = max(
                args.contextual_isr_frontier_max_targets,
                args.contextual_isr_frontier_max_targets * max(1, args.contextual_isr_frontier_candidate_pool_multiplier),
            )
            isr_frontier_head, isr_frontier_tail = target_head_tail_quota(
                args.contextual_isr_frontier_max_targets,
                1.0 - contextual_isr_tail_ratio,
            )

            def contextual_isr_frontier_candidates():
                vector_only_targets = runner.uncovered_bb_sets().get("vector_only", set())
                head_targets = runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.contextual_isr_frontier_min_uncovered_successors),
                    switch_only=args.contextual_isr_frontier_switch_only,
                    max_targets=contextual_isr_candidate_pool_limit,
                    target_filter=vector_only_targets,
                )
                tail_targets = []
                if isr_frontier_tail > 0:
                    tail_targets = runner.prioritized_target_bb_list(
                        vector_only_targets,
                        min_uncovered_successors=max(1, args.contextual_isr_frontier_min_uncovered_successors),
                        switch_only=args.contextual_isr_frontier_switch_only,
                        include_nearby_uncovered=True,
                        nearby_limit=isr_frontier_tail,
                        max_targets=isr_frontier_tail,
                        exclude_targets=set(head_targets),
                    )
                priority_targets = prioritized_target_fallback(
                    runner,
                    vector_only_targets,
                    switch_only=args.contextual_isr_frontier_switch_only,
                    max_targets=contextual_isr_candidate_pool_limit,
                    min_uncovered_successors=args.contextual_isr_frontier_min_uncovered_successors,
                    nearby_limit=max(8, isr_frontier_tail or (isr_frontier_head // 2)),
                    exclude_targets=set(head_targets) | set(tail_targets),
                )
                return dedupe_target_order(
                    head_targets,
                    tail_targets,
                    priority_targets or sorted(vector_only_targets),
                )

            contextual_isr_frontier_targeted, contextual_isr_frontier_rounds = run_targeted_stage(
                runner=runner,
                stage_name="contextual_isr_frontier_targeted",
                requested_rounds=args.contextual_isr_frontier_rounds,
                auto_max_rounds=args.contextual_isr_frontier_max_rounds,
                stale_round_limit=args.contextual_isr_frontier_stale_rounds,
                min_progress_bbs=args.contextual_isr_frontier_min_progress_bbs,
                max_targets=args.contextual_isr_frontier_max_targets,
                cooldown_rounds=args.target_cooldown_rounds,
                candidate_supplier=contextual_isr_frontier_candidates,
                executor=lambda phase_name, current_targets: runner.context_recovery.isr_reservoir(
                    time_limit_seconds=clamp_enabled_stage_seconds(args.contextual_isr_frontier_seconds),
                    max_instructions=args.contextual_isr_reservoir_instructions,
                    replay_timeout=args.contextual_isr_replay_timeout_us,
                    max_tasks=args.contextual_isr_reservoir_max_tasks,
                    max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                    dynamic_frontier_seeding=True,
                    context_snapshots=True,
                    max_contexts=args.contextual_isr_contexts,
                    include_reservoir_contexts=True,
                    max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                    max_isrs=args.contextual_isr_max_isrs,
                    target_bbs=current_targets,
                    prioritize_target_bbs=True,
                    phase_name=phase_name,
                ),
                should_continue=lambda: has_wallclock_budget(),
            )
            runner.replay_executor.finish_stage(
                contextual_frontier_stage,
                contextual_isr_frontier_targeted,
                coverage_counted=False,
                extra={
                    "stage_covered_bbs": len(contextual_isr_frontier_targeted),
                    "rounds_executed": len(contextual_isr_frontier_rounds),
                    "global_covered_bbs": len(runner.global_coverage),
                },
            )
        else:
            skip_stage("contextual_isr_frontier_targeted", "disabled_or_no_budget")

        switch_frontier_targeted = set()
        switch_frontier_rounds = []
        frontier_targeted = set()
        frontier_rounds = []
        frontier_cycles = []
        vector_cleanup_targeted = set()
        vector_cleanup_rounds = []
        frontier_cycle_limit = 1
        if frontier_cycle_auto and (switch_frontier_enabled or frontier_targeted_enabled):
            frontier_cycle_limit = max(1, args.frontier_cycle_max_cycles)
        frontier_stale_cycles = 0
        switch_frontier_runtime_enabled = switch_frontier_enabled
        switch_frontier_low_yield_cycles = 0

        def run_targeted_reservoir_phase(
            *,
            phase_name: str,
            current_targets: set[int],
            phase_seconds: int,
            max_tasks_per_root: int,
            strategy: dict[str, bool],
            reset_state: bool,
            conditional_root_quota: int,
        ) -> set[int]:
            if phase_seconds <= 0:
                return set()
            switch_risky_backoff = False
            if phase_name.startswith("switch_frontier_targeted"):
                previous_switch_meta = None
                for previous_phase_name, previous_meta in runner.phase_metadata.items():
                    if previous_phase_name.startswith("switch_frontier_targeted"):
                        previous_switch_meta = dict(previous_meta or {})
                if previous_switch_meta:
                    previous_new_bbs = int(previous_switch_meta.get("new_bbs", 0) or 0)
                    previous_discovered = int(previous_switch_meta.get("target_bbs_discovered", 0) or 0)
                    previous_deep_replays = int(previous_switch_meta.get("deep_branch_snapshot_replay_tasks", 0) or 0)
                    previous_dispatch_replays = int(previous_switch_meta.get("dynamic_dispatch_snapshot_replay_tasks", 0) or 0)
                    previous_risky_tasks = int(previous_switch_meta.get("direct_snapshot_risky_tasks", 0) or 0)
                    previous_bootstrap_tasks = int(previous_switch_meta.get("snapshot_frontier_bootstrap_tasks_executed", 0) or 0)
                    switch_risky_backoff = (
                        previous_new_bbs <= 0
                        and previous_discovered <= 0
                        and (
                            previous_deep_replays >= 16
                            or previous_dispatch_replays >= 8
                            or previous_risky_tasks >= 16
                            or previous_bootstrap_tasks >= 8
                        )
                    )
            frontier_conditional_bootstrap_env = {
                "LSGEMU_STRICT_DYNAMIC_FRONTIER_EXHAUSTIVE": "0",
                "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_BOOTSTRAP_LIMIT": "4",
                "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_VARIANTS_PER_ADDRESS": "1",
                "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_MAX_UNCOVERED_SUCCESSORS": "2",
            }
            if phase_name.startswith("frontier_targeted"):
                frontier_conditional_bootstrap_env.update({
                    "LSGEMU_STRICT_DYNAMIC_FRONTIER_EXHAUSTIVE": "0",
                    "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_BOOTSTRAP_LIMIT": "6",
                    "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_VARIANTS_PER_ADDRESS": "2",
                    "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_MAX_UNCOVERED_SUCCESSORS": "3",
                })
            phase_env = {
                "LSGEMU_TARGETED_DIRECT_ROOT_FOCUS": "1" if strategy["targeted_direct_root_focus"] else "0",
                "LSGEMU_PREFER_DIRECT_ROOT_SNAPSHOT": "1" if strategy["prefer_direct_root_snapshot"] else "0",
                "LSGEMU_TARGETED_PREFIX_SNAPSHOT_SEED_LIMIT": str(
                    max(0, args.targeted_prefix_snapshot_seed_limit)
                ),
                "LSGEMU_TARGETED_DIRECT_ROOT_CONDITIONAL_QUOTA": str(
                    max(0, conditional_root_quota)
                ),
            }
            if (
                phase_name.startswith("switch_frontier_targeted")
                and os.environ.get("LSGEMU_ENABLE_SWITCH_RISKY_BRANCH_SNAPSHOT_FALLBACK", "0").lower()
                in {"0", "false", "no"}
            ):
                phase_env["LSGEMU_RISKY_BRANCH_SNAPSHOT_FALLBACK"] = "0"
            if switch_risky_backoff:
                phase_env.update({
                    "LSGEMU_TARGETED_SNAPSHOT_FRONTIER_BOOTSTRAP": "0",
                    "LSGEMU_TARGETED_CONDITIONAL_SNAPSHOT_FRONTIER_BOOTSTRAP": "0",
                    "LSGEMU_RISKY_BRANCH_SNAPSHOT_FALLBACK": "0",
                    "LSGEMU_AUTO_TARGETED_ROOT_SNAPSHOT_RETRY": "0",
                })
            phase_env.update(frontier_conditional_bootstrap_env)
            with temporary_env(phase_env):
                return runner.run_reservoir_branch_exploration(
                    time_limit_seconds=phase_seconds,
                    replay_instructions=args.reservoir_replay_instructions,
                    replay_timeout=args.reservoir_replay_timeout_us,
                    max_tasks=args.reservoir_max_tasks,
                    max_tasks_per_root=max_tasks_per_root,
                    target_bbs=current_targets,
                    prioritize_target_bbs=True,
                    continue_target_merges=True,
                    target_probe_bbs=min(512, max(64, args.reservoir_replay_instructions // 100)),
                    phase_name=phase_name,
                    reset_state=reset_state,
                )

        def run_targeted_frontier_phase(
            *,
            phase_name: str,
            current_targets: set[int],
            phase_seconds: int,
            max_tasks_per_root: int,
            strategy: dict[str, bool],
            reset_state: bool,
            conditional_root_quota: int,
        ) -> set[int]:
            if phase_seconds <= 0:
                return set()
            direct_seconds = direct_call_budget_seconds(
                phase_name=phase_name,
                phase_seconds=phase_seconds,
                current_targets=current_targets,
            )
            if (
                phase_name.startswith("frontier_targeted")
                and os.environ.get("LSGEMU_FRONTIER_TARGETED_DIRECT_SUBPHASE", "0").lower()
                in {"0", "false", "no"}
            ):
                direct_seconds = 0
            summary_seconds = 0
            if current_targets and direct_seconds > 0:
                summary_seconds = min(
                    max(0, args.direct_call_summary_return_seconds),
                    max(0, direct_seconds // 2),
                )
            thread_seconds = min(
                max(0, args.rtos_thread_entry_seconds),
                max(0, phase_seconds // 8),
            )
            if (
                short_probe_mode()
                and current_targets
                and max(0, args.rtos_thread_entry_seconds) > 0
                and phase_name.startswith("frontier_targeted")
            ):
                thread_seconds = min(
                    max(0, args.rtos_thread_entry_seconds),
                    max(
                        thread_seconds,
                        min(
                            max(3, phase_seconds // 3),
                            max(0, phase_seconds - max(1, direct_seconds)),
                        ),
                    ),
                )
            strict_subphase_budget = (
                os.environ.get("LSGEMU_STRICT_TARGETED_SUBPHASE_BUDGET", "0").lower()
                in {"1", "true", "yes", "on"}
            )
            if strict_subphase_budget or short_probe_mode():
                auxiliary_seconds = direct_seconds + summary_seconds + thread_seconds
                if auxiliary_seconds >= phase_seconds:
                    overflow = auxiliary_seconds - max(0, phase_seconds - 1)
                    summary_reduction = min(summary_seconds, overflow)
                    summary_seconds -= summary_reduction
                    overflow -= summary_reduction
                    thread_reduction = min(thread_seconds, overflow)
                    thread_seconds -= thread_reduction
                    overflow -= thread_reduction
                    if overflow > 0:
                        direct_seconds = max(0, direct_seconds - overflow)
                reservoir_seconds = max(1, phase_seconds - direct_seconds - summary_seconds - thread_seconds)
            else:
                reservoir_seconds = max(1, phase_seconds - direct_seconds)
            combined: set[int] = set()
            if current_targets and direct_seconds > 0:
                combined.update(
                    run_direct_call_continuation_stage(
                        f"{phase_name}_direct_call",
                        direct_seconds,
                        max_tasks=max(args.direct_call_continuation_max_tasks, len(current_targets)),
                        max_targets=max(args.direct_call_continuation_max_targets, len(current_targets)),
                        target_bbs=current_targets,
                    )
                )
                if summary_seconds > 0:
                    combined.update(
                        run_direct_call_summary_return_stage(
                            f"{phase_name}_direct_call_summary_return",
                            clamp_enabled_stage_seconds(summary_seconds),
                            max_tasks=max(args.direct_call_summary_return_max_tasks, len(current_targets)),
                            max_targets=max(args.direct_call_summary_return_max_targets, len(current_targets)),
                            target_bbs=current_targets,
                        )
                    )
            if current_targets and thread_seconds > 0:
                combined.update(
                    run_rtos_thread_entry_stage(
                        f"{phase_name}_rtos_thread_entry",
                        clamp_enabled_stage_seconds(thread_seconds),
                        max_tasks=max(args.rtos_thread_entry_max_tasks, len(current_targets)),
                        max_targets=max(args.rtos_thread_entry_max_targets, len(current_targets)),
                        target_bbs=current_targets,
                    )
                )
            combined.update(
                run_targeted_reservoir_phase(
                    phase_name=phase_name,
                    current_targets=current_targets,
                    phase_seconds=clamp_enabled_stage_seconds(reservoir_seconds),
                    max_tasks_per_root=max_tasks_per_root,
                    strategy=strategy,
                    reset_state=reset_state,
                    conditional_root_quota=conditional_root_quota,
                )
            )
            return combined

        targeted_frontier_cycles_stage = runner.replay_executor.begin_stage(
            "targeted_frontier_cycles"
        )
        for cycle_index in range(frontier_cycle_limit):
            if short_probe_mode():
                cycle_tail_reserve_seconds = max(
                    8,
                    min(
                        frontier_cycle_tail_reserve_seconds,
                        max(0, int(args.direct_call_continuation_seconds) // 2)
                        + max(0, int(args.direct_call_summary_return_seconds) // 2),
                    ),
                )
            else:
                cycle_tail_reserve_seconds = frontier_cycle_tail_reserve_seconds
            if not has_wallclock_budget(max(1, cycle_tail_reserve_seconds)):
                break
            cycle_switch_summary = empty_stage_summary()
            cycle_frontier_summary = empty_stage_summary()
            cycle_global_before = len(runner.global_coverage)
            cycle_switch_round_seconds = clamp_enabled_stage_seconds(
                args.switch_frontier_round_seconds,
                reserve_after_seconds=cycle_tail_reserve_seconds,
            )
            cycle_frontier_round_seconds = clamp_enabled_stage_seconds(
                args.frontier_round_seconds,
                reserve_after_seconds=cycle_tail_reserve_seconds,
            )
            switch_candidate_pool_limit = max(
                args.switch_frontier_max_targets,
                args.switch_frontier_max_targets * max(1, args.switch_frontier_candidate_pool_multiplier),
            )
            frontier_candidate_pool_limit = max(
                args.frontier_max_targets,
                args.frontier_max_targets * max(1, args.frontier_candidate_pool_multiplier),
            )
            _switch_frontier_head, _switch_frontier_tail = target_head_tail_quota(
                args.switch_frontier_max_targets,
                1.0,
            )
            frontier_tail_ratio = min(1.0, max(0.0, args.frontier_nearby_tail_ratio))
            frontier_head, frontier_tail = target_head_tail_quota(
                args.frontier_max_targets,
                1.0 - frontier_tail_ratio,
            )

            def switch_frontier_candidates():
                uncovered_targets = runner.uncovered_bb_sets().get("uncovered", set())
                summary = runner.uncovered_coverage_summary()
                cold_targets = parse_unstarted_frontier_targets(
                    summary,
                    switch_candidate_pool_limit,
                    switch_only=True,
                )
                if not cold_targets:
                    summary = runner.uncovered_coverage_summary()
                    cold_targets = parse_unstarted_frontier_targets(
                        summary,
                        switch_candidate_pool_limit,
                        switch_only=True,
                    )
                hotspot_targets = []
                if args.hotspot_frontier_predecessors > 0:
                    hotspot_targets = runner.hotspot_frontier_target_bb_list(
                        switch_only=True,
                        max_predecessors=args.hotspot_frontier_predecessors,
                        max_targets=switch_candidate_pool_limit,
                        min_uncovered_successors=max(1, args.switch_frontier_min_uncovered_successors),
                    )
                fallback_targets = runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.switch_frontier_min_uncovered_successors),
                    switch_only=True,
                    max_targets=switch_candidate_pool_limit,
                )
                priority_targets = prioritized_target_fallback(
                    runner,
                    uncovered_targets,
                    switch_only=True,
                    max_targets=switch_candidate_pool_limit,
                    min_uncovered_successors=args.switch_frontier_min_uncovered_successors,
                    nearby_limit=max(8, switch_candidate_pool_limit // 4),
                    exclude_targets=set(hotspot_targets) | set(cold_targets) | set(fallback_targets),
                )
                return dedupe_target_order(
                    hotspot_targets,
                    cold_targets,
                    fallback_targets,
                    priority_targets,
                )

            def frontier_candidates():
                uncovered_targets = runner.uncovered_bb_sets().get("uncovered", set())
                summary = runner.uncovered_coverage_summary()
                cold_targets = parse_unstarted_frontier_targets(summary, frontier_candidate_pool_limit)
                if not cold_targets:
                    summary = runner.uncovered_coverage_summary()
                    cold_targets = parse_unstarted_frontier_targets(
                        summary,
                        frontier_candidate_pool_limit,
                    )
                hotspot_targets = []
                if args.hotspot_frontier_predecessors > 0:
                    hotspot_targets = runner.hotspot_frontier_target_bb_list(
                        switch_only=args.frontier_switch_only,
                        max_predecessors=args.hotspot_frontier_predecessors,
                        max_targets=frontier_candidate_pool_limit,
                        min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                    )
                head_targets = runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                    switch_only=args.frontier_switch_only,
                    max_targets=frontier_candidate_pool_limit,
                )
                tail_targets = []
                if frontier_tail > 0:
                    tail_targets = runner.prioritized_target_bb_list(
                        uncovered_targets,
                        min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                        switch_only=args.frontier_switch_only,
                        include_nearby_uncovered=True,
                        nearby_limit=frontier_tail,
                        max_targets=frontier_tail,
                        exclude_targets=set(head_targets),
                    )
                priority_targets = prioritized_target_fallback(
                    runner,
                    uncovered_targets,
                    switch_only=args.frontier_switch_only,
                    max_targets=frontier_candidate_pool_limit,
                    min_uncovered_successors=args.frontier_min_uncovered_successors,
                    nearby_limit=max(8, frontier_tail or (frontier_head // 2)),
                    exclude_targets=set(hotspot_targets) | set(cold_targets) | set(head_targets) | set(tail_targets),
                )
                return dedupe_target_order(
                    hotspot_targets,
                    cold_targets,
                    head_targets,
                    tail_targets,
                    priority_targets,
                )

            switch_preview_targets = []
            frontier_preview_targets = []
            switch_preview_skipped_reason = ""
            frontier_preview_first = False
            short_frontier_skip_switch_preview = (
                os.environ.get("LSGEMU_SHORT_FRONTIER_SKIP_SWITCH_PREVIEW", "1").lower()
                in {"1", "true", "yes", "on"}
            )
            if short_probe_mode() and frontier_targeted_enabled and cycle_frontier_round_seconds > 0:
                # Build the high-yield general frontier first.  In short runs a
                # large general frontier will defer switch replay anyway; avoid
                # spending tens of seconds materializing switch candidates that
                # will not execute in this cycle.
                frontier_preview_first = True
                frontier_preview_targets = list(frontier_candidates() or [])
                frontier_pressure_threshold = max(
                    64,
                    min(frontier_candidate_pool_limit, max(1, int(args.frontier_max_targets))),
                )
                if (
                    short_frontier_skip_switch_preview
                    and switch_frontier_runtime_enabled
                    and cycle_switch_round_seconds > 0
                    and len(frontier_preview_targets) >= frontier_pressure_threshold
                ):
                    switch_preview_skipped_reason = "deferred_for_general_frontier_pressure_before_preview"
                    cycle_switch_round_seconds = 0
                elif switch_frontier_runtime_enabled and cycle_switch_round_seconds > 0:
                    switch_preview_targets = list(switch_frontier_candidates() or [])
            else:
                if switch_frontier_runtime_enabled and cycle_switch_round_seconds > 0:
                    switch_preview_targets = list(switch_frontier_candidates() or [])
                if frontier_targeted_enabled and cycle_frontier_round_seconds > 0:
                    frontier_preview_targets = list(frontier_candidates() or [])

            targeted_cycle_budget_split = {
                "switch_preview_targets": len(switch_preview_targets),
                "frontier_preview_targets": len(frontier_preview_targets),
                "switch_seconds_before_split": int(cycle_switch_round_seconds),
                "frontier_seconds_before_split": int(cycle_frontier_round_seconds),
                "budget_split_applied": False,
                "frontier_preview_first": bool(frontier_preview_first),
                "switch_preview_skipped_reason": switch_preview_skipped_reason,
            }
            finite_remaining = remaining_wallclock_seconds()
            if finite_remaining is not None:
                regular_available_for_targeted = max(0, int(finite_remaining) - int(cycle_tail_reserve_seconds))
                available_for_targeted = regular_available_for_targeted
                switch_wants_budget = bool(switch_preview_targets) and cycle_switch_round_seconds > 0
                frontier_wants_budget = bool(frontier_preview_targets) and cycle_frontier_round_seconds > 0
                if short_probe_mode() and (switch_wants_budget or frontier_wants_budget):
                    short_finalization_reserve = max(3, min(8, int(cycle_tail_reserve_seconds) // 2 or 3))
                    short_first_round_available = max(0, int(finite_remaining) - short_finalization_reserve)
                    if short_first_round_available > available_for_targeted:
                        available_for_targeted = short_first_round_available
                        targeted_cycle_budget_split.update({
                            "short_first_round_available": int(short_first_round_available),
                            "short_first_round_finalization_reserve": int(short_finalization_reserve),
                            "regular_available_for_targeted": int(regular_available_for_targeted),
                        })
                if switch_wants_budget and frontier_wants_budget and available_for_targeted > 0:
                    desired_switch = min(
                        int(cycle_switch_round_seconds),
                        max(0, int(args.switch_frontier_round_seconds)),
                        available_for_targeted,
                    )
                    desired_frontier = min(
                        int(cycle_frontier_round_seconds),
                        max(0, int(args.frontier_round_seconds)),
                        available_for_targeted,
                    )
                    if desired_switch + desired_frontier > available_for_targeted:
                        if short_probe_mode():
                            minimum_slice = max(2, min(8, available_for_targeted // 3 or 1))
                        else:
                            minimum_slice = max(5, min(30, available_for_targeted // 4 or 1))
                        frontier_bias = 0.60 if len(frontier_preview_targets) >= len(switch_preview_targets) else 0.50
                        frontier_alloc = int(available_for_targeted * frontier_bias)
                        frontier_alloc = max(minimum_slice, frontier_alloc)
                        frontier_alloc = min(desired_frontier, frontier_alloc)
                        switch_alloc = min(desired_switch, max(0, available_for_targeted - frontier_alloc))
                        if (
                            switch_alloc < min(minimum_slice, desired_switch)
                            and desired_switch > 0
                            and available_for_targeted > frontier_alloc
                        ):
                            switch_alloc = min(desired_switch, min(minimum_slice, available_for_targeted))
                            frontier_alloc = min(desired_frontier, max(0, available_for_targeted - switch_alloc))
                        if (
                            frontier_alloc < min(minimum_slice, desired_frontier)
                            and desired_frontier > 0
                            and available_for_targeted > switch_alloc
                        ):
                            frontier_alloc = min(desired_frontier, min(minimum_slice, available_for_targeted))
                            switch_alloc = min(desired_switch, max(0, available_for_targeted - frontier_alloc))
                        cycle_switch_round_seconds = max(0, int(switch_alloc))
                        cycle_frontier_round_seconds = max(0, int(frontier_alloc))
                        targeted_cycle_budget_split["budget_split_applied"] = True
                    else:
                        cycle_switch_round_seconds = desired_switch
                        cycle_frontier_round_seconds = desired_frontier
                elif switch_wants_budget and not frontier_wants_budget:
                    cycle_frontier_round_seconds = 0
                    cycle_switch_round_seconds = min(
                        int(cycle_switch_round_seconds),
                        available_for_targeted,
                    )
                elif frontier_wants_budget and not switch_wants_budget:
                    cycle_switch_round_seconds = 0
                    cycle_frontier_round_seconds = min(
                        int(cycle_frontier_round_seconds),
                        available_for_targeted,
                    )
            targeted_cycle_budget_split.update({
                "switch_seconds_after_split": int(cycle_switch_round_seconds),
                "frontier_seconds_after_split": int(cycle_frontier_round_seconds),
            })
            frontier_pressure_deferred_switch = False
            if (
                frontier_preview_targets
                and cycle_frontier_round_seconds > 0
                and cycle_switch_round_seconds > 0
                and len(frontier_preview_targets) >= max(64, len(switch_preview_targets) * 2)
                and (
                    short_probe_mode()
                    or cycle_frontier_round_seconds < max(0, int(args.frontier_round_seconds))
                )
            ):
                cycle_frontier_round_seconds += cycle_switch_round_seconds
                cycle_switch_round_seconds = 0
                frontier_pressure_deferred_switch = True
                targeted_cycle_budget_split.update({
                    "switch_deferred_for_frontier_pressure": True,
                    "switch_seconds_after_pressure": int(cycle_switch_round_seconds),
                "frontier_seconds_after_pressure": int(cycle_frontier_round_seconds),
                })
            else:
                targeted_cycle_budget_split["switch_deferred_for_frontier_pressure"] = False

            def has_short_first_round_budget(
                round_index: int,
                preview_targets: list[int],
                phase_seconds: int,
            ) -> bool:
                if (
                    not short_probe_mode()
                    or round_index != 0
                    or not preview_targets
                    or phase_seconds <= 0
                ):
                    return False
                remaining = remaining_wallclock_seconds()
                if remaining is None:
                    return True
                min_required = max(3, min(8, int(phase_seconds)))
                return remaining >= min_required

            if switch_frontier_runtime_enabled and cycle_switch_round_seconds > 0:
                cycle_switch_summary = run_frontier_reservoir_stage(
                    runner=runner,
                    stage_name="switch_frontier_targeted",
                    cycle_index=cycle_index,
                    round_seconds=cycle_switch_round_seconds,
                    requested_rounds=args.switch_frontier_rounds,
                    auto_max_rounds=args.switch_frontier_max_rounds,
                    stale_round_limit=args.switch_frontier_stale_rounds,
                    min_progress_bbs=args.switch_frontier_min_progress_bbs,
                    max_targets=args.switch_frontier_max_targets,
                    cooldown_rounds=args.target_cooldown_rounds,
                    candidate_supplier=previewed_candidate_supplier(
                        switch_preview_targets,
                        switch_frontier_candidates,
                    ),
                    executor=lambda phase_name, current_targets: run_targeted_reservoir_phase(
                        phase_name=phase_name,
                        current_targets=current_targets,
                        phase_seconds=cycle_switch_round_seconds,
                        max_tasks_per_root=max(args.targeted_max_tasks_per_root, 32),
                        strategy=switch_stage_strategy,
                        reset_state=not args.continue_targeted_state,
                        conditional_root_quota=0,
                    ),
                    round_records=switch_frontier_rounds,
                    coverage_accumulator=switch_frontier_targeted,
                    should_continue=lambda round_index=0: has_stage_wallclock_budget(
                        cycle_switch_round_seconds,
                        reserve_after_seconds=cycle_tail_reserve_seconds,
                    ) or has_short_first_round_budget(
                        round_index,
                        switch_preview_targets,
                        cycle_switch_round_seconds,
                    ),
                    refresh_remaining_targets=not short_probe_mode(),
                )
                if args.switch_frontier_disable_low_yield_cycles > 0:
                    switch_cycle_new_bbs = int(cycle_switch_summary["new_bbs"])
                    switch_cycle_targets = int(cycle_switch_summary["target_bbs_discovered"])
                    switch_cycle_rounds = int(cycle_switch_summary["rounds_executed"])
                    low_yield = (
                        switch_cycle_rounds > 0
                        and switch_cycle_new_bbs <= max(0, args.switch_frontier_low_yield_new_bbs)
                        and switch_cycle_targets <= max(0, args.switch_frontier_low_yield_targets)
                    )
                    if low_yield:
                        switch_frontier_low_yield_cycles += 1
                    else:
                        switch_frontier_low_yield_cycles = 0
                    if switch_frontier_low_yield_cycles >= max(1, args.switch_frontier_disable_low_yield_cycles):
                        switch_frontier_runtime_enabled = False

            if frontier_targeted_enabled and cycle_frontier_round_seconds > 0:
                cycle_frontier_summary = run_frontier_reservoir_stage(
                    runner=runner,
                    stage_name="frontier_targeted",
                    cycle_index=cycle_index,
                    round_seconds=cycle_frontier_round_seconds,
                    requested_rounds=args.frontier_rounds,
                    auto_max_rounds=args.frontier_max_rounds,
                    stale_round_limit=args.frontier_stale_rounds,
                    min_progress_bbs=args.frontier_min_progress_bbs,
                    max_targets=args.frontier_max_targets,
                    cooldown_rounds=args.target_cooldown_rounds,
                    candidate_supplier=previewed_candidate_supplier(
                        frontier_preview_targets,
                        frontier_candidates,
                    ),
                    executor=lambda phase_name, current_targets: run_targeted_reservoir_phase(
                        phase_name=phase_name,
                        current_targets=current_targets,
                        phase_seconds=cycle_frontier_round_seconds,
                        max_tasks_per_root=args.targeted_max_tasks_per_root,
                        strategy=frontier_stage_strategy,
                        reset_state=not args.continue_targeted_state,
                        conditional_root_quota=0,
                    ) if args.frontier_switch_only else run_targeted_frontier_phase(
                        phase_name=phase_name,
                        current_targets=current_targets,
                        phase_seconds=cycle_frontier_round_seconds,
                        max_tasks_per_root=args.targeted_max_tasks_per_root,
                        strategy=frontier_stage_strategy,
                        reset_state=not args.continue_targeted_state,
                        conditional_root_quota=0,
                    ),
                    round_records=frontier_rounds,
                    coverage_accumulator=frontier_targeted,
                    should_continue=lambda round_index=0: has_stage_wallclock_budget(
                        cycle_frontier_round_seconds,
                        reserve_after_seconds=cycle_tail_reserve_seconds,
                    ) or has_short_first_round_budget(
                        round_index,
                        frontier_preview_targets,
                        cycle_frontier_round_seconds,
                    ),
                    refresh_remaining_targets=not short_probe_mode(),
                )

            cycle_global_after = len(runner.global_coverage)
            cycle_new_bbs = cycle_global_after - cycle_global_before
            cycle_targets_discovered = (
                int(cycle_switch_summary["target_bbs_discovered"])
                + int(cycle_frontier_summary["target_bbs_discovered"])
            )
            cycle_new_candidate_targets = (
                int(cycle_switch_summary["new_candidate_targets"])
                + int(cycle_frontier_summary["new_candidate_targets"])
            )
            cycle_new_branch_events = (
                int(cycle_switch_summary["remembered_branch_events"])
                + int(cycle_frontier_summary["remembered_branch_events"])
            )
            cycle_new_snapshot_variants = (
                int(cycle_switch_summary["remembered_branch_root_snapshots"])
                + int(cycle_frontier_summary["remembered_branch_root_snapshots"])
            )
            cycle_priority_root_tasks_seeded = (
                int(cycle_switch_summary["priority_root_tasks_seeded"])
                + int(cycle_frontier_summary["priority_root_tasks_seeded"])
            )
            cycle_dynamic_successor_edges = (
                int(cycle_switch_summary["dynamic_successor_edges_added"])
                + int(cycle_frontier_summary["dynamic_successor_edges_added"])
            )
            cycle_snapshot_variants = (
                int(cycle_switch_summary["remembered_branch_root_snapshots"])
                + int(cycle_frontier_summary["remembered_branch_root_snapshots"])
            )
            cycle_frontier_flush_covered = set()
            cycle_frontier_flush_seconds = 0
            cycle_frontier_flush_trigger_reasons: list[str] = []
            cycle_frontier_flush_skip_reason = ""
            min_flush_edges = max(1, args.frontier_successor_flush_min_dynamic_edges)
            min_flush_snapshots = max(1, args.frontier_successor_flush_min_snapshots)
            max_flush_new_bbs = max(0, args.frontier_successor_flush_max_new_bbs)
            if cycle_dynamic_successor_edges >= min_flush_edges:
                cycle_frontier_flush_trigger_reasons.append("dynamic_successor_edges")
            if cycle_new_bbs <= max_flush_new_bbs and cycle_snapshot_variants >= min_flush_snapshots:
                cycle_frontier_flush_trigger_reasons.append("snapshot_variants_plateau")
            if (
                args.frontier_successor_flush_seconds > 0
                and cycle_frontier_flush_trigger_reasons
            ):
                if has_wallclock_budget():
                    cycle_frontier_flush_seconds = clamp_enabled_stage_seconds(
                        min(
                            int(args.frontier_successor_flush_seconds),
                            max(
                                1,
                                cycle_switch_round_seconds // 2
                                if cycle_switch_round_seconds > 0
                                else args.frontier_successor_flush_seconds,
                            ),
                        )
                    )
                    if cycle_frontier_flush_seconds > 0:
                        cycle_frontier_flush_covered = run_frontier_successor_replay_stage(
                            f"cycle_frontier_successor_flush_{cycle_index + 1}",
                            cycle_frontier_flush_seconds,
                        )
                    else:
                        cycle_frontier_flush_skip_reason = "no_remaining_after_clamp"
                else:
                    cycle_frontier_flush_skip_reason = "no_wallclock_budget"
            cycle_frontier_flush_new_bbs = max(0, len(runner.global_coverage) - cycle_global_after)
            frontier_cycles.append({
                "cycle": cycle_index + 1,
                "switch_candidate_targets": len(cycle_switch_summary["initial_targets"]),
                "switch_remaining_targets": len(cycle_switch_summary["remaining_targets"]),
                "switch_rounds_executed": int(cycle_switch_summary["rounds_executed"]),
                "switch_new_bbs": int(cycle_switch_summary["new_bbs"]),
                "switch_target_bbs_discovered": int(cycle_switch_summary["target_bbs_discovered"]),
                "switch_new_candidate_targets": int(cycle_switch_summary["new_candidate_targets"]),
                "switch_branch_events": int(cycle_switch_summary["remembered_branch_events"]),
                "switch_snapshot_variants": int(cycle_switch_summary["remembered_branch_root_snapshots"]),
                "switch_priority_root_tasks_seeded": int(cycle_switch_summary["priority_root_tasks_seeded"]),
                "frontier_candidate_targets": len(cycle_frontier_summary["initial_targets"]),
                "frontier_remaining_targets": len(cycle_frontier_summary["remaining_targets"]),
                "frontier_rounds_executed": int(cycle_frontier_summary["rounds_executed"]),
                "frontier_new_bbs": int(cycle_frontier_summary["new_bbs"]),
                "frontier_target_bbs_discovered": int(cycle_frontier_summary["target_bbs_discovered"]),
                "frontier_new_candidate_targets": int(cycle_frontier_summary["new_candidate_targets"]),
                "frontier_branch_events": int(cycle_frontier_summary["remembered_branch_events"]),
                "frontier_snapshot_variants": int(cycle_frontier_summary["remembered_branch_root_snapshots"]),
                "frontier_priority_root_tasks_seeded": int(cycle_frontier_summary["priority_root_tasks_seeded"]),
                "cycle_new_global_bbs": cycle_new_bbs,
                "cycle_target_bbs_discovered": cycle_targets_discovered,
                "cycle_new_candidate_targets": cycle_new_candidate_targets,
                "cycle_branch_events": cycle_new_branch_events,
                "cycle_snapshot_variants": cycle_snapshot_variants,
                "cycle_priority_root_tasks_seeded": cycle_priority_root_tasks_seeded,
                "cycle_dynamic_successor_edges_added": cycle_dynamic_successor_edges,
                "cycle_frontier_tail_reserve_seconds": int(cycle_tail_reserve_seconds),
                "targeted_cycle_budget_split": dict(targeted_cycle_budget_split),
                "cycle_frontier_successor_flush_trigger_reasons": list(cycle_frontier_flush_trigger_reasons),
                "cycle_frontier_successor_flush_skip_reason": cycle_frontier_flush_skip_reason,
                "cycle_frontier_successor_flush_seconds": cycle_frontier_flush_seconds,
                "cycle_frontier_successor_flush_bbs": len(cycle_frontier_flush_covered),
                "cycle_frontier_successor_flush_new_bbs": cycle_frontier_flush_new_bbs,
                "switch_tail_new_bbs": int(cycle_switch_summary["last_round_new_bbs"]),
                "switch_tail_target_bbs_discovered": int(cycle_switch_summary["last_round_target_bbs_discovered"]),
                "switch_tail_new_candidate_targets": int(cycle_switch_summary["last_round_new_candidate_targets"]),
                "switch_tail_branch_events": int(cycle_switch_summary["last_round_branch_events"]),
                "switch_tail_snapshot_variants": int(cycle_switch_summary["last_round_snapshot_variants"]),
                "switch_frontier_runtime_enabled_next_cycle": bool(switch_frontier_runtime_enabled),
                "switch_frontier_low_yield_cycles": int(switch_frontier_low_yield_cycles),
                "frontier_tail_new_bbs": int(cycle_frontier_summary["last_round_new_bbs"]),
                "frontier_tail_target_bbs_discovered": int(cycle_frontier_summary["last_round_target_bbs_discovered"]),
                "frontier_tail_new_candidate_targets": int(cycle_frontier_summary["last_round_new_candidate_targets"]),
                "frontier_tail_branch_events": int(cycle_frontier_summary["last_round_branch_events"]),
                "frontier_tail_snapshot_variants": int(cycle_frontier_summary["last_round_snapshot_variants"]),
                "global_bbs_after_cycle": cycle_global_after,
            })
            if not frontier_cycle_auto or (not switch_frontier_enabled and not frontier_targeted_enabled):
                break
            if (
                not cycle_switch_summary["initial_targets"]
                and not cycle_frontier_summary["initial_targets"]
            ):
                break
            cycle_counted_progress = any((
                cycle_new_bbs > 0,
                cycle_targets_discovered > 0,
                cycle_new_candidate_targets > 0,
                cycle_frontier_flush_new_bbs > 0,
            ))
            cycle_metadata_only_progress = (
                not cycle_counted_progress
                and (
                    cycle_new_branch_events > 0
                    or cycle_snapshot_variants > 0
                    or cycle_priority_root_tasks_seeded > 0
                )
            )
            cycle_progress = cycle_counted_progress or (
                cycle_dynamic_successor_edges > 0
                and cycle_frontier_flush_seconds <= 0
                and not cycle_frontier_flush_skip_reason
            )
            if not cycle_progress:
                break
            if not cycle_counted_progress and (
                cycle_metadata_only_progress
                or (
                    cycle_dynamic_successor_edges > 0
                    and cycle_frontier_flush_seconds > 0
                    and cycle_frontier_flush_new_bbs <= 0
                )
            ):
                frontier_stale_cycles += 1
            else:
                frontier_stale_cycles = 0
            effective_stale_limit = max(1, args.frontier_cycle_stale_cycles)
            if total_wallclock_budget_seconds and total_wallclock_budget_seconds <= 1200:
                effective_stale_limit = max(effective_stale_limit, 2)
            if cycle_frontier_flush_new_bbs > max(0, args.frontier_successor_flush_max_new_bbs):
                effective_stale_limit = max(effective_stale_limit, 2)
            if frontier_stale_cycles >= effective_stale_limit:
                break

        runner.replay_executor.finish_stage(
            targeted_frontier_cycles_stage,
            switch_frontier_targeted | frontier_targeted,
            coverage_counted=False,
            extra={
                "switch_frontier_covered_bbs": len(switch_frontier_targeted),
                "frontier_targeted_covered_bbs": len(frontier_targeted),
                "frontier_cycles_executed": len(frontier_cycles),
                "global_covered_bbs": len(runner.global_coverage),
            },
        )

        path_naturalization = set()
        naturalization_ledger_before = runner.path_naturalization_ledger.summary()
        naturalization_reserve_seconds = (
            max(0, int(frontier_cycle_tail_reserve_seconds))
            + (0 if args.disable_deadline_drain else max(0, int(args.deadline_drain_min_seconds)))
        )
        # Wind-down stage: keeps its bounded budget even after the run-level
        # stall watchdog fires (constraint learning produces obligations and
        # facts, not coverage; skipping it would make stalled runs' artifacts
        # incomparable with full-budget runs of the same firmware).
        naturalization_seconds = clamp_winddown_stage_seconds(
            max(0, int(args.path_naturalization_seconds)),
            reserve_after_seconds=naturalization_reserve_seconds,
        )
        if args.disable_semantic_obligation_stages:
            skip_stage("path_naturalization", "semantic_obligation_ablation_disabled")
        elif args.disable_scoped_replay_stages:
            skip_stage("path_naturalization", "scoped_replay_ablation_disabled")
        elif int(naturalization_ledger_before.get("obligations", 0) or 0) <= 0:
            skip_stage("path_naturalization", "no_forced_path_obligations")
        elif naturalization_seconds <= 0 or not has_winddown_stage_budget(
            naturalization_seconds,
            min_fraction=0.10,
            reserve_after_seconds=naturalization_reserve_seconds,
        ):
            skip_stage("path_naturalization", "disabled_or_no_budget")
        else:
            path_naturalization = run_stage(
                "path_naturalization",
                lambda: runner.run_path_naturalization(
                    time_limit_seconds=naturalization_seconds,
                    max_paths=max(1, int(args.path_naturalization_max_paths)),
                    max_attempts_per_path=max(
                        1,
                        int(args.path_naturalization_max_attempts_per_path),
                    ),
                    max_fact_candidates_per_edge=max(
                        1,
                        int(args.path_naturalization_max_sources_per_edge),
                    ),
                    max_values_per_source=max(
                        1,
                        int(args.path_naturalization_values_per_source),
                    ),
                    max_compound_candidates=max(
                        0,
                        int(args.path_naturalization_max_compound_candidates),
                    ),
                    replay_instructions=max(1, int(args.path_naturalization_instructions)),
                    replay_timeout=max(
                        1,
                        int(args.path_naturalization_replay_timeout_us),
                    ),
                    phase_name="path_naturalization",
                ),
            )
        runner.phase_metadata.setdefault("path_naturalization", {}).update({
            "configured_seconds": int(args.path_naturalization_seconds),
            "effective_seconds": int(naturalization_seconds),
            "protected_after_seconds": int(naturalization_reserve_seconds),
            "obligations_before": int(naturalization_ledger_before.get("obligations", 0) or 0),
            "externalized_facts_before": int(
                naturalization_ledger_before.get("externally_replayable_facts", 0) or 0
            ),
        })

        late_direct_call_frontier_drain = set()
        late_direct_call_summary_return_drain = set()
        late_direct_call_frontier_targets = (
            set()
            if args.disable_frontier_stages
            else set(
                runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                    switch_only=False,
                    max_targets=max(args.direct_call_continuation_max_targets, args.frontier_max_targets, 256),
                    include_nearby_uncovered=True,
                    nearby_limit=max(args.direct_call_continuation_max_targets, args.frontier_max_targets, 256),
                )
            )
        )
        # Late direct-call frontiers are exact entry-derived callsite states.
        # Prefer spending tail budget on them before lower-yield vector cleanup;
        # short successor-tail preservation is opt-in: forcing it by default
        # can starve higher-yield semantic/thread stages on small budgets.
        try:
            configured_short_successor_tail_floor = int(
                os.environ.get("LSGEMU_SHORT_SUCCESSOR_TAIL_FLOOR_SECONDS", "0") or "0"
            )
        except ValueError:
            configured_short_successor_tail_floor = 0
        short_successor_tail_floor = (
            max(0, min(
                configured_short_successor_tail_floor,
                max(0, int(args.frontier_successor_replay_tail_seconds)),
            ))
            if short_probe_mode()
            else 0
        )
        late_direct_call_protected_seconds = (
            0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds)
        ) + short_successor_tail_floor
        late_direct_call_seconds = clamp_enabled_stage_seconds(
            min(
                max(0, args.direct_call_continuation_seconds),
                max(10, max(0, args.frontier_successor_replay_tail_seconds) // 6),
            ),
            reserve_after_seconds=late_direct_call_protected_seconds,
        )
        if late_direct_call_frontier_targets and late_direct_call_seconds > 0:
            late_direct_call_frontier_drain = run_direct_call_continuation_stage(
                "late_direct_call_frontier_drain",
                late_direct_call_seconds,
                max_tasks=max(args.direct_call_continuation_max_tasks, 256),
                max_targets=max(args.direct_call_continuation_max_targets, 256),
                target_bbs=late_direct_call_frontier_targets,
            )
            runner.phase_metadata.setdefault("late_direct_call_frontier_drain", {}).update({
                "frontier_target_count": len(late_direct_call_frontier_targets),
                "frontier_target_sample": [
                    f"0x{bb:08x}" for bb in sorted(late_direct_call_frontier_targets)[:32]
                ],
                "protected_after_seconds": int(late_direct_call_protected_seconds),
            })
        else:
            skip_stage(
                "late_direct_call_frontier_drain",
                "no_targets_or_no_budget",
            )
            runner.phase_metadata.setdefault("late_direct_call_frontier_drain", {}).update({
                "frontier_target_count": len(late_direct_call_frontier_targets),
                "protected_after_seconds": int(late_direct_call_protected_seconds),
            })

        late_summary_protected_seconds = (
            (
                max(0, args.vector_cleanup_seconds)
                if args.contextual_isr_contexts != 0 and not short_probe_mode()
                else 0
            )
            + (0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds))
            + max(short_successor_tail_floor, max(0, int(args.frontier_successor_replay_tail_seconds) // 4))
        )
        late_summary_seconds = clamp_enabled_stage_seconds(
            min(
                max(0, args.direct_call_summary_return_seconds),
                max(8, max(0, args.frontier_successor_replay_tail_seconds) // 8),
            ),
            reserve_after_seconds=late_summary_protected_seconds,
        )
        if late_direct_call_frontier_targets and late_summary_seconds > 0:
            late_direct_call_summary_return_drain = run_direct_call_summary_return_stage(
                "late_direct_call_summary_return_drain",
                late_summary_seconds,
                max_tasks=max(args.direct_call_summary_return_max_tasks, 256),
                max_targets=max(args.direct_call_summary_return_max_targets, 256),
                target_bbs=late_direct_call_frontier_targets,
            )
            runner.phase_metadata.setdefault("late_direct_call_summary_return_drain", {}).update({
                "frontier_target_count": len(late_direct_call_frontier_targets),
                "frontier_target_sample": [
                    f"0x{bb:08x}" for bb in sorted(late_direct_call_frontier_targets)[:32]
                ],
                "protected_after_seconds": int(late_summary_protected_seconds),
            })
        else:
            skip_stage(
                "late_direct_call_summary_return_drain",
                "no_targets_or_no_budget",
            )
            runner.phase_metadata.setdefault("late_direct_call_summary_return_drain", {}).update({
                "frontier_target_count": len(late_direct_call_frontier_targets),
                "protected_after_seconds": int(late_summary_protected_seconds),
            })

        frontier_successor_replay_tail = run_frontier_successor_replay_stage(
            "frontier_successor_replay_tail",
            extend_final_stage_seconds(
                args.frontier_successor_replay_tail_seconds,
                protected_after_seconds=(
                    max(0, args.vector_cleanup_seconds)
                    if args.contextual_isr_contexts != 0 and not short_probe_mode()
                    else 0
                ) + (0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds)),
            ),
        )

        # Wind-down stage: bounded per-round seconds with its own stale-round
        # exit; keeps running after the stall watchdog fires so stalled runs
        # still drain their vector-only cleanup obligations.
        vector_cleanup_seconds = clamp_winddown_stage_seconds(
            args.vector_cleanup_seconds,
            reserve_after_seconds=(
                0 if args.disable_deadline_drain else max(0, args.deadline_drain_min_seconds)
            ),
        )
        if vector_cleanup_seconds > 0 and args.contextual_isr_contexts != 0 and has_winddown_budget():
            vector_cleanup_stage = runner.replay_executor.begin_stage(
                "vector_only_cleanup"
            )
            vector_cleanup_tail_ratio = min(1.0, max(0.0, args.vector_cleanup_nearby_tail_ratio))
            vector_cleanup_head, vector_cleanup_tail = target_head_tail_quota(
                args.vector_cleanup_max_targets,
                1.0 - vector_cleanup_tail_ratio,
            )

            def vector_cleanup_candidates():
                vector_only_targets = runner.uncovered_bb_sets().get("vector_only", set())
                head_targets = runner.frontier_target_bb_list(
                    min_uncovered_successors=max(1, args.vector_cleanup_min_uncovered_successors),
                    switch_only=False,
                    max_targets=vector_cleanup_head,
                    target_filter=vector_only_targets,
                )
                tail_targets = []
                if vector_cleanup_tail > 0:
                    tail_targets = runner.prioritized_target_bb_list(
                        vector_only_targets,
                        min_uncovered_successors=max(1, args.vector_cleanup_min_uncovered_successors),
                        switch_only=False,
                        include_nearby_uncovered=True,
                        nearby_limit=vector_cleanup_tail,
                        max_targets=vector_cleanup_tail,
                        exclude_targets=set(head_targets),
                        prefer_expandable=True,
                    )
                priority_targets = prioritized_target_fallback(
                    runner,
                    vector_only_targets,
                    switch_only=False,
                    max_targets=vector_cleanup_head + vector_cleanup_tail,
                    min_uncovered_successors=args.vector_cleanup_min_uncovered_successors,
                    nearby_limit=max(8, vector_cleanup_tail or (vector_cleanup_head // 2)),
                    exclude_targets=set(head_targets) | set(tail_targets),
                )
                ordered_targets = dedupe_target_order(
                    head_targets,
                    tail_targets,
                    priority_targets or sorted(vector_only_targets),
                )
                expandable_targets = runner.expandable_target_bb_list(
                    ordered_targets,
                    require_uncovered_non_self_successor=False,
                )
                if expandable_targets:
                    return expandable_targets
                return ordered_targets

            vector_cleanup_targeted, vector_cleanup_rounds = run_targeted_stage(
                runner=runner,
                stage_name="vector_only_cleanup",
                requested_rounds=args.vector_cleanup_rounds,
                auto_max_rounds=args.vector_cleanup_max_rounds,
                stale_round_limit=args.vector_cleanup_stale_rounds,
                min_progress_bbs=args.vector_cleanup_min_progress_bbs,
                max_targets=args.vector_cleanup_max_targets,
                cooldown_rounds=args.target_cooldown_rounds,
                candidate_supplier=vector_cleanup_candidates,
                executor=lambda phase_name, current_targets: runner.context_recovery.isr_reservoir(
                    time_limit_seconds=clamp_winddown_stage_seconds(args.vector_cleanup_seconds),
                    max_instructions=args.contextual_isr_reservoir_instructions,
                    replay_timeout=args.contextual_isr_replay_timeout_us,
                    max_tasks=args.contextual_isr_reservoir_max_tasks,
                    max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                    dynamic_frontier_seeding=True,
                    context_snapshots=True,
                    max_contexts=args.contextual_isr_contexts,
                    include_reservoir_contexts=True,
                    max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                    max_isrs=args.contextual_isr_max_isrs,
                    target_bbs=current_targets,
                    prioritize_target_bbs=True,
                    phase_name=phase_name,
                ),
                should_continue=lambda: has_winddown_budget() and clamp_winddown_stage_seconds(args.vector_cleanup_seconds) > 0,
            )
            runner.replay_executor.finish_stage(
                vector_cleanup_stage,
                vector_cleanup_targeted,
                coverage_counted=False,
                extra={
                    "stage_covered_bbs": len(vector_cleanup_targeted),
                    "rounds_executed": len(vector_cleanup_rounds),
                    "global_covered_bbs": len(runner.global_coverage),
                },
            )
        else:
            skip_stage("vector_only_cleanup", "disabled_or_no_budget")

        deadline_drain_covered = set()
        deadline_drain_rounds = []
        deadline_drain_stop_reason = "not_started"
        if not args.disable_deadline_drain and has_wallclock_budget(args.deadline_drain_min_seconds):
            deadline_drain_stage = runner.replay_executor.begin_stage(
                "deadline_drain"
            )
            drain_stale_rounds = 0
            pressure_stale_rounds = 0
            drain_round_index = 0
            while has_wallclock_budget(args.deadline_drain_min_seconds):
                if stall_watchdog.should_truncate_stage("deadline_drain"):
                    # Stage-level truncation: the whole drain composite is a
                    # single budget-owner stage, so its all-ledger stall
                    # accrues across the rotating sub-stages.  Exit through
                    # the normal finish_stage path below; the pipeline then
                    # continues with path_naturalization_final.
                    deadline_drain_stop_reason = STAGE_TRUNCATION_REASON
                    break
                drain_round_index += 1
                round_remaining = remaining_wallclock_seconds()
                round_budget = max(1, int(args.deadline_drain_round_seconds))
                if round_remaining is not None:
                    round_budget = min(round_budget, max(0, int(round_remaining)))
                if round_budget <= 0:
                    break

                global_before = len(runner.global_coverage)
                round_record = {
                    "round": drain_round_index,
                    "budget_seconds": round_budget,
                    "global_bbs_before": global_before,
                }
                drain_frontier_targets = (
                    set()
                    if args.disable_frontier_stages
                    else set(
                        dedupe_target_order(
                            runner.frontier_target_bb_list(
                                min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                                switch_only=False,
                                max_targets=max(args.frontier_max_targets, 256),
                                include_nearby_uncovered=True,
                                nearby_limit=max(args.frontier_max_targets, 256),
                            ),
                            runner.prioritized_target_bb_list(
                                runner.uncovered_bb_sets().get("uncovered", set()),
                                min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                                switch_only=False,
                                include_nearby_uncovered=True,
                                nearby_limit=max(args.frontier_max_targets, 256),
                                max_targets=max(args.frontier_max_targets, 256),
                            ),
                        )
                    )
                )
                direct_target_scope = (
                    runner.uncovered_bb_sets().get("entry_reachable", set())
                    if args.disable_frontier_stages
                    else (
                        drain_frontier_targets
                        or runner.uncovered_bb_sets().get("entry_reachable", set())
                    )
                )
                round_record["frontier_target_count"] = len(drain_frontier_targets)
                round_record["frontier_targeting_disabled_by_ablation"] = bool(args.disable_frontier_stages)
                round_record["frontier_target_sample"] = [
                    f"0x{bb:08x}" for bb in sorted(drain_frontier_targets)[:16]
                ]

                drain_summary = runner.uncovered_coverage_summary(
                    max_frontier_predecessors=max(32, args.hotspot_frontier_predecessors * 4)
                )
                frontier_category_counts = dict(drain_summary.get("frontier_category_counts") or {})
                switch_pressure = int(frontier_category_counts.get("switch_case_frontier", 0) or 0)
                conditional_pressure = int(frontier_category_counts.get("conditional_branch_frontier", 0) or 0)
                indirect_pressure = int(frontier_category_counts.get("indirect_call_or_return_frontier", 0) or 0)
                replayable_frontier_pressure = (
                    0
                    if args.disable_frontier_stages
                    else (
                        len(drain_frontier_targets)
                        or switch_pressure
                        + conditional_pressure
                        + indirect_pressure
                    )
                )
                targeted_first = (
                    frontier_targeted_enabled
                    and replayable_frontier_pressure >= max(1, int(args.deadline_drain_targeted_first_threshold))
                )
                round_record["frontier_category_counts"] = frontier_category_counts
                round_record["frontier_pressure_disabled_by_ablation"] = bool(args.disable_frontier_stages)
                round_record["deadline_targeted_first"] = bool(targeted_first)
                round_record["replayable_frontier_pressure"] = int(replayable_frontier_pressure)
                round_record["remaining_wallclock_seconds_before"] = round_remaining

                def deadline_target_candidates(*, switch_only: bool, limit: int) -> list[int]:
                    uncovered_targets = runner.uncovered_bb_sets().get("uncovered", set())
                    return dedupe_target_order(
                        parse_unstarted_frontier_targets(
                            drain_summary,
                            limit,
                            switch_only=switch_only,
                        ),
                        runner.hotspot_frontier_target_bb_list(
                            switch_only=switch_only,
                            max_predecessors=max(1, args.hotspot_frontier_predecessors),
                            max_targets=limit,
                            min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                        ),
                        runner.frontier_target_bb_list(
                            min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                            switch_only=switch_only,
                            max_targets=limit,
                            include_nearby_uncovered=True,
                            nearby_limit=limit,
                        ),
                        runner.prioritized_target_bb_list(
                            uncovered_targets,
                            min_uncovered_successors=max(1, args.frontier_min_uncovered_successors),
                            switch_only=switch_only,
                            include_nearby_uncovered=True,
                            nearby_limit=limit,
                            max_targets=limit,
                            prefer_expandable=True,
                        ),
                    )

                def run_deadline_targeted_subphase(
                    stage_name: str,
                    targets: list[int],
                    seconds: int,
                    *,
                    switch_only: bool = False,
                ) -> set[int]:
                    selected = list(targets[:max(args.frontier_max_targets, 256)])
                    seconds = clamp_enabled_stage_seconds(max(0, int(seconds)))
                    if not selected or seconds <= 0:
                        skip_stage(stage_name, "no_targets_or_no_budget")
                        runner.phase_metadata.setdefault(stage_name, {}).update({
                            "target_count": len(selected),
                            "switch_only": bool(switch_only),
                        })
                        return set()
                    if switch_only:
                        return run_stage(
                            stage_name,
                            lambda: run_targeted_reservoir_phase(
                                phase_name=stage_name,
                                current_targets=set(selected),
                                phase_seconds=seconds,
                                max_tasks_per_root=max(args.targeted_max_tasks_per_root, 32),
                                strategy=switch_stage_strategy,
                                reset_state=not args.continue_targeted_state,
                                conditional_root_quota=0,
                            ),
                        )
                    return run_stage(
                        stage_name,
                        lambda: run_targeted_frontier_phase(
                            phase_name=stage_name,
                            current_targets=set(selected),
                            phase_seconds=seconds,
                            max_tasks_per_root=args.targeted_max_tasks_per_root,
                            strategy=frontier_stage_strategy,
                            reset_state=not args.continue_targeted_state,
                            conditional_root_quota=max(0, args.targeted_max_tasks_per_root),
                        ),
                    )

                targeted_covered = set()
                switch_targeted_covered = set()
                targeted_targets: list[int] = []
                switch_targeted_targets: list[int] = []
                targeted_seconds_budget = 0
                switch_targeted_seconds = 0
                general_targeted_seconds = 0
                if (
                    has_wallclock_budget(args.deadline_drain_min_seconds)
                    and frontier_targeted_enabled
                    and not args.disable_frontier_stages
                ):
                    targeted_target_limit = max(args.frontier_max_targets, 256)
                    targeted_targets = deadline_target_candidates(
                        switch_only=args.frontier_switch_only,
                        limit=targeted_target_limit,
                    )
                    switch_targeted_targets = (
                        deadline_target_candidates(
                            switch_only=True,
                            limit=max(args.switch_frontier_max_targets, 192),
                        )
                        if switch_pressure > 0 and not args.frontier_switch_only
                        else []
                    )
                    if targeted_targets or switch_targeted_targets:
                        targeted_share = min(0.90, max(0.0, float(args.deadline_drain_targeted_share)))
                        if targeted_first:
                            targeted_seconds_budget = max(10, int(round_budget * targeted_share))
                        else:
                            targeted_seconds_budget = min(
                                max(10, round_budget // 4),
                                int(round_budget * max(0.20, targeted_share * 0.60)),
                            )
                        targeted_seconds_budget = min(
                            targeted_seconds_budget,
                            clamp_enabled_stage_seconds(max(10, targeted_seconds_budget)),
                        )
                        switch_share = min(0.80, max(0.0, float(args.deadline_drain_switch_share)))
                        if switch_targeted_targets and targeted_seconds_budget > 0:
                            switch_targeted_seconds = max(
                                5,
                                int(targeted_seconds_budget * switch_share),
                            )
                            switch_targeted_seconds = min(
                                switch_targeted_seconds,
                                targeted_seconds_budget,
                            )
                        general_targeted_seconds = max(
                            0,
                            targeted_seconds_budget - switch_targeted_seconds,
                        )

                if targeted_first:
                    if switch_targeted_seconds > 0:
                        switch_targeted_covered = run_deadline_targeted_subphase(
                            f"deadline_drain_switch_frontier_targeted_{drain_round_index}",
                            switch_targeted_targets,
                            switch_targeted_seconds,
                            switch_only=True,
                        )
                        targeted_covered.update(switch_targeted_covered)
                    if general_targeted_seconds > 0:
                        targeted_covered.update(
                            run_deadline_targeted_subphase(
                                f"deadline_drain_frontier_targeted_{drain_round_index}",
                                targeted_targets,
                                general_targeted_seconds,
                            )
                        )
                    deadline_drain_covered.update(targeted_covered)

                direct_stage = f"deadline_drain_direct_call_{drain_round_index}"
                direct_seconds = direct_call_budget_seconds(
                    phase_name=direct_stage,
                    phase_seconds=round_budget,
                    current_targets=direct_target_scope,
                )
                if targeted_first and direct_seconds > 0:
                    direct_seconds = min(direct_seconds, max(1, round_budget // 6))
                if direct_seconds > 0:
                    direct_covered = run_direct_call_continuation_stage(
                        direct_stage,
                        direct_seconds,
                        max_tasks=max(args.direct_call_continuation_max_tasks, 256),
                        max_targets=max(args.direct_call_continuation_max_targets, 256),
                        target_bbs=direct_target_scope,
                    )
                    summary_seconds = min(
                        max(0, args.direct_call_summary_return_seconds),
                        max(0, direct_seconds // 2),
                    )
                    if summary_seconds > 0:
                        direct_covered.update(
                            run_direct_call_summary_return_stage(
                                f"{direct_stage}_summary_return",
                                clamp_enabled_stage_seconds(summary_seconds),
                                max_tasks=max(args.direct_call_summary_return_max_tasks, 256),
                                max_targets=max(args.direct_call_summary_return_max_targets, 256),
                                target_bbs=direct_target_scope,
                            )
                        )
                else:
                    direct_covered = set()
                    skip_stage(direct_stage, "adaptive_skip_zero_yield")
                deadline_drain_covered.update(direct_covered)
                round_record["direct_call_bbs"] = len(direct_covered)
                round_record["direct_call_seconds"] = int(direct_seconds)

                if not targeted_first:
                    if switch_targeted_seconds > 0:
                        switch_targeted_covered = run_deadline_targeted_subphase(
                            f"deadline_drain_switch_frontier_targeted_{drain_round_index}",
                            switch_targeted_targets,
                            switch_targeted_seconds,
                            switch_only=True,
                        )
                        targeted_covered.update(switch_targeted_covered)
                    if general_targeted_seconds > 0:
                        targeted_covered.update(
                            run_deadline_targeted_subphase(
                                f"deadline_drain_frontier_targeted_{drain_round_index}",
                                targeted_targets,
                                general_targeted_seconds,
                            )
                        )
                    deadline_drain_covered.update(targeted_covered)

                consumed_before_successor = direct_seconds + targeted_seconds_budget
                frontier_stage = f"deadline_drain_frontier_successor_{drain_round_index}"
                if args.disable_frontier_stages:
                    frontier_seconds = 0
                    frontier_covered = set()
                    skip_stage(frontier_stage, "frontier_ablation_disabled")
                else:
                    frontier_seconds = min(
                        max(1, round_budget - consumed_before_successor),
                        max(args.frontier_successor_replay_tail_seconds, round_budget // 2, 30),
                    )
                    frontier_covered = run_frontier_successor_replay_stage(
                        frontier_stage,
                        clamp_enabled_stage_seconds(frontier_seconds),
                        target_bbs=drain_frontier_targets or None,
                    )
                deadline_drain_covered.update(frontier_covered)
                round_record["frontier_successor_bbs"] = len(frontier_covered)
                round_record["targeted_frontier_targets"] = len(targeted_targets)
                round_record["targeted_frontier_bbs"] = len(targeted_covered)
                round_record["switch_targeted_frontier_targets"] = len(switch_targeted_targets)
                round_record["switch_targeted_frontier_bbs"] = len(switch_targeted_covered)
                round_record["targeted_seconds_budget"] = int(targeted_seconds_budget)
                round_record["switch_targeted_seconds"] = int(switch_targeted_seconds)
                round_record["general_targeted_seconds"] = int(general_targeted_seconds)
                round_record["frontier_successor_seconds"] = int(frontier_seconds)

                global_after = len(runner.global_coverage)
                round_record["global_bbs_after"] = global_after
                round_record["new_global_bbs"] = global_after - global_before
                actionable_pressure = bool(
                    replayable_frontier_pressure > 0
                    or targeted_targets
                    or switch_targeted_targets
                    or direct_target_scope
                )
                round_record["actionable_pressure"] = actionable_pressure
                round_record["remaining_wallclock_seconds_after"] = remaining_wallclock_seconds()
                deadline_drain_rounds.append(round_record)

                if global_after <= global_before:
                    drain_stale_rounds += 1
                    if actionable_pressure:
                        pressure_stale_rounds += 1
                else:
                    drain_stale_rounds = 0
                    pressure_stale_rounds = 0
                round_record["stale_rounds"] = int(drain_stale_rounds)
                round_record["pressure_stale_rounds"] = int(pressure_stale_rounds)
                if drain_stale_rounds >= max(1, args.deadline_drain_stale_rounds):
                    pressure_limit = max(
                        max(1, args.deadline_drain_stale_rounds),
                        int(args.deadline_drain_pressure_stale_rounds),
                    )
                    if actionable_pressure and pressure_stale_rounds < pressure_limit:
                        round_record["stale_break_deferred"] = True
                        continue
                    deadline_drain_stop_reason = (
                        "pressure_stale_round_limit_reached"
                        if actionable_pressure
                        else "stale_round_limit_reached"
                    )
                    break
            else:
                deadline_drain_stop_reason = "wallclock_exhausted"
            if deadline_drain_stop_reason == "not_started":
                deadline_drain_stop_reason = (
                    "wallclock_or_min_budget_exhausted"
                    if not has_wallclock_budget(args.deadline_drain_min_seconds)
                    else "completed_without_break"
                )

            runner.replay_executor.finish_stage(
                deadline_drain_stage,
                deadline_drain_covered,
                coverage_counted=False,
                extra={
                    "stage_covered_bbs": len(deadline_drain_covered),
                    "rounds_executed": len(deadline_drain_rounds),
                    "stop_reason": deadline_drain_stop_reason,
                    "global_covered_bbs": len(runner.global_coverage),
                    "remaining_wallclock_seconds": remaining_wallclock_seconds(),
                },
            )
            runner.phase_metadata.setdefault("deadline_drain", {}).update({
                "coverage_counted": False,
                "stage_covered_bbs": len(deadline_drain_covered),
                "rounds_executed": len(deadline_drain_rounds),
                "stop_reason": deadline_drain_stop_reason,
                "remaining_wallclock_seconds": remaining_wallclock_seconds(),
                "stale_rounds": int(drain_stale_rounds),
                "pressure_stale_rounds": int(pressure_stale_rounds),
                "pressure_stale_round_limit": int(args.deadline_drain_pressure_stale_rounds),
            })
        else:
            deadline_drain_stop_reason = "disabled_or_no_budget"
            skip_stage("deadline_drain", "disabled_or_no_budget")

        path_naturalization_final = set()
        final_naturalization_before = runner.path_naturalization_ledger.summary()
        final_status_counts = dict(
            final_naturalization_before.get("status_counts", {}) or {}
        )
        final_pending_obligations = sum(
            int(count or 0)
            for status, count in final_status_counts.items()
            if str(status) not in {"promoted", "already_natural", "exhausted"}
        )
        final_naturalization_configured_seconds = (
            max(1, min(
                int(args.path_naturalization_seconds),
                max(2, int(args.path_naturalization_seconds) // 3),
            ))
            if int(args.path_naturalization_seconds) > 0
            else 0
        )
        # Wind-down stage (same rationale as path_naturalization above): the
        # final obligation drain is bounded by a fraction of
        # path_naturalization_seconds and must survive a watchdog early stop.
        final_naturalization_seconds = clamp_winddown_stage_seconds(
            final_naturalization_configured_seconds
        )
        if args.disable_semantic_obligation_stages:
            skip_stage(
                "path_naturalization_final",
                "semantic_obligation_ablation_disabled",
            )
        elif args.disable_scoped_replay_stages:
            skip_stage("path_naturalization_final", "scoped_replay_ablation_disabled")
        elif final_pending_obligations <= 0:
            skip_stage("path_naturalization_final", "no_pending_forced_path_obligations")
        elif final_naturalization_seconds <= 0 or not has_winddown_budget(min_seconds=1):
            skip_stage("path_naturalization_final", "no_remaining_wallclock_budget")
        else:
            path_naturalization_final = run_stage(
                "path_naturalization_final",
                lambda: runner.run_path_naturalization(
                    time_limit_seconds=final_naturalization_seconds,
                    max_paths=max(1, int(args.path_naturalization_max_paths)),
                    max_attempts_per_path=max(
                        1,
                        int(args.path_naturalization_max_attempts_per_path),
                    ),
                    max_fact_candidates_per_edge=max(
                        1,
                        int(args.path_naturalization_max_sources_per_edge),
                    ),
                    max_values_per_source=max(
                        1,
                        int(args.path_naturalization_values_per_source),
                    ),
                    max_compound_candidates=max(
                        0,
                        int(args.path_naturalization_max_compound_candidates),
                    ),
                    replay_instructions=max(1, int(args.path_naturalization_instructions)),
                    replay_timeout=max(
                        1,
                        int(args.path_naturalization_replay_timeout_us),
                    ),
                    phase_name="path_naturalization_final",
                ),
            )
        runner.phase_metadata.setdefault("path_naturalization_final", {}).update({
            "configured_seconds": int(final_naturalization_configured_seconds),
            "effective_seconds": int(final_naturalization_seconds),
            "pending_obligations_before": int(final_pending_obligations),
            "obligations_before": int(
                final_naturalization_before.get("obligations", 0) or 0
            ),
            "externalized_facts_before": int(
                final_naturalization_before.get("externally_replayable_facts", 0) or 0
            ),
        })

        progress_monitor.set_stage("finalize")
        progress_monitor.snapshot("finalizing_report", stage="finalize")
        # r40 P2/C8：基线后投递的死安装点已删（见 baseline 相位后注释）；
        # 相位级投递审计在 phase_metadata[phase].irq_delivery（随相位落盘）。
        post_baseline_irq_delivery_audit = None
        elapsed = time.time() - start
        stall_watchdog_stop_reason = (
            stall_watchdog.stop_report()["reason"]
            if stall_watchdog.stop_report() is not None
            else None
        )
        stall_watchdog_budget_snapshot = (
            progress_monitor.stall_watchdog_budget_snapshot or {}
        )
        branch_catalog = runner.save_branch_catalog(branch_catalog_file)
        runner.llm_guide.save_inference_history(str(llm_history_file))
        runner.llm_guide.close_inference_journal()
        # Build first so the final progress-monitor state can be included before
        # the report is atomically persisted.  The previous flow wrote the same
        # large report once here and then rewrote it after ``stop()``.
        report = runner.build_report(
            execution_time_seconds=elapsed,
            extra={
                "runner": "gateway_interleaved_cached",
                "constraint_file": str(constraint_file),
                "seeded_constraint_count": seeded_constraint_count,
                "constraint_seed_mode": args.constraint_seed_mode,
                "constraint_seed_activation": args.constraint_seed_activation,
                "constraint_seed_roots": [str(path) for path in learned_roots],
                "post_interleaved_reserve_seconds": reserved_post_interleaved_seconds,
                "post_interleaved_reserve_ratio": args.post_interleaved_reserve_ratio,
                "interleaved_total_time_budget_seconds": interleaved_total_seconds,
                "requested_total_wallclock_budget_seconds": total_wallclock_budget_seconds,
                "remaining_wallclock_seconds_at_finalize": remaining_wallclock_seconds(),
                "deadline_drain_stop_reason": deadline_drain_stop_reason,
                # r40 P2/C8：k.5 的基线后投递死安装点已删（钩子挂在不执行的
                # 常驻实例上）。字段保留为恒 null 以维持报告 schema 兼容；
                # 相位级投递审计看 phase_metadata[phase].irq_delivery。
                "post_baseline_irq_delivery_audit": post_baseline_irq_delivery_audit,
                # Early-termination bookkeeping: ``terminated_early`` is true
                # exactly when the run-level stall watchdog stopped the run
                # before the configured wallclock budget was spent, so
                # "1440min-budget" comparison tables can separate runs that
                # filled their budget from runs that were stopped early.
                # ``configured_budget_seconds == 0`` means no deadline was set.
                "terminated_early": stall_watchdog_stop_reason is not None,
                "terminated_early_reason": stall_watchdog_stop_reason,
                "wallclock_used_seconds": round(elapsed, 3),
                "configured_budget_seconds": total_wallclock_budget_seconds,
                "stall_watchdog_budget_remaining_seconds": (
                    stall_watchdog_budget_snapshot.get("budget_remaining_seconds")
                ),
                "stall_watchdog_elapsed_seconds_at_trigger": (
                    stall_watchdog_budget_snapshot.get("elapsed_seconds_at_capture")
                ),
                "stall_watchdog_winddown": stall_watchdog_winddown.status(),
                "stall_watchdog_stop_reason": stall_watchdog_stop_reason,
                # Stage-level truncations did NOT stop the run: the pipeline
                # continued under the real remaining budget.  Each record
                # names the stage, the frozen-ledger stall window, and the
                # ledger watermarks at the trigger point.
                "stall_watchdog_stage_truncations": stall_watchdog.stage_truncations(),
                # cycle4 k.5 C2：水位口径兄弟键（与 ledger_watermarks int
                # 字典同级；scheduler/stall_watchdog.py 的 status() 本体
                # 零改动，此处只在装配点合并只读注记，消解 236≠161 / 268≠107
                # 的跨时点误读——水位=观察时点运行最大值，非报告时分区）。
                "stall_watchdog": {
                    **stall_watchdog.status(),
                    "ledger_watermarks_caliber": {
                        "_semantics": "running_max_observed_at_observation_time_not_report_time",
                        "natural_supported_bbs": "E0_union_E1",
                        "counterfactual_only_bbs": "E2_union_E3_minus_natural",
                        "validated_replay_bbs": "canonical_validated",
                        "diagnostic_replay_bbs": "canonical_diagnostic_only",
                    },
                },
                "branch_catalog_file": str(branch_catalog_file),
                "llm_history_file": str(llm_history_file),
                "llm_history_journal_file": str(llm_history_journal_file),
                "runtime_bootstrap": runtime_bootstrap,
                "stage_strategies": {
                    "adaptive_stage_caps": adaptive_stage_caps,
                    "switch_frontier_targeted": switch_stage_strategy,
                    "frontier_targeted": frontier_stage_strategy,
                    "targeted_prefix_snapshot_seed_limit": int(os.environ.get("LSGEMU_TARGETED_PREFIX_SNAPSHOT_SEED_LIMIT", "64")),
                    "frontier_cycle_tail_reserve_seconds": int(frontier_cycle_tail_reserve_seconds),
                    "targeted_frontier_reserve_seconds": int(targeted_frontier_reserve_seconds),
                    "auto_targeted_frontier_reserve_seconds": int(auto_targeted_frontier_reserve_seconds),
                    "semantic_frontier_flush_seconds": int(args.semantic_frontier_flush_seconds),
                    "semantic_frontier_flush_max_targets": int(args.semantic_frontier_flush_max_targets),
                    "semantic_frontier_flush_max_tasks": int(args.semantic_frontier_flush_max_tasks),
                    "semantic_frontier_flush_switch_only": bool(args.semantic_frontier_flush_switch_only),
                    "semantic_vector_cleanup_seconds": int(args.semantic_vector_cleanup_seconds),
                    "path_naturalization_seconds": int(args.path_naturalization_seconds),
                    "path_naturalization_max_paths": int(args.path_naturalization_max_paths),
                    "path_naturalization_max_attempts_per_path": int(
                        args.path_naturalization_max_attempts_per_path
                    ),
                    "path_naturalization_max_sources_per_edge": int(
                        args.path_naturalization_max_sources_per_edge
                    ),
                    "path_naturalization_values_per_source": int(
                        args.path_naturalization_values_per_source
                    ),
                    "path_naturalization_max_compound_candidates": int(
                        args.path_naturalization_max_compound_candidates
                    ),
                    "replay_time_skip_mode": str(args.replay_time_skip_mode),
                    "replay_time_increment": int(args.replay_time_increment),
                    "deadline_drain_targeted_share": float(args.deadline_drain_targeted_share),
                    "deadline_drain_targeted_first_threshold": int(args.deadline_drain_targeted_first_threshold),
                    "deadline_drain_switch_share": float(args.deadline_drain_switch_share),
                },
                "component_bbs": {
                    "baseline": len(baseline),
                    "isr": len(isr),
                    "contextual_isr": len(contextual_isr),
                    "interleaved": len(interleaved),
                    "frontier_successor_replay_early": len(frontier_successor_replay_early),
                    "early_switch_frontier_reservoir": len(early_switch_frontier_reservoir),
                    "early_frontier_reservoir": len(early_frontier_reservoir),
                    "contextual_isr_reservoir": len(contextual_isr_reservoir),
                    "semantic_vector_cleanup": len(semantic_vector_cleanup),
                    "stream_input_replay": len(stream_input_replay),
                    "stream_input_replay_after_threads": len(stream_input_after_threads),
                    "semantic_frontier_flush": len(semantic_frontier_flush),
                    "semantic_frontier_drain": len(semantic_frontier_drain),
                    "stream_input_replay_post_frontier": len(post_frontier_stream_input),
                    "direct_call_continuation": len(direct_call_continuation),
                    "late_direct_call_frontier_drain": len(late_direct_call_frontier_drain),
                    "late_direct_call_summary_return_drain": len(late_direct_call_summary_return_drain),
                    "rtos_thread_entry": len(rtos_thread_entry),
                    "direct_call_summary_return": len(direct_call_summary_return),
                    "frontier_successor_replay": len(frontier_successor_replay),
                    "frontier_successor_replay_tail": len(frontier_successor_replay_tail),
                    "contextual_isr_frontier_targeted": len(contextual_isr_frontier_targeted),
                    "vector_only_cleanup": len(vector_cleanup_targeted),
                    "deadline_drain": len(deadline_drain_covered),
                    "switch_frontier_targeted": len(switch_frontier_targeted),
                    "frontier_targeted": len(frontier_targeted),
                    "path_naturalization": len(path_naturalization),
                    "path_naturalization_final": len(path_naturalization_final),
                },
                "component_bbs_counted": {
                    "baseline": len(baseline),
                    "isr": len(isr) if args.count_cold_isr_coverage else 0,
                    "contextual_isr": len(contextual_isr),
                    "interleaved": len(interleaved),
                    "frontier_successor_replay_early": len(frontier_successor_replay_early),
                    "early_switch_frontier_reservoir": len(early_switch_frontier_reservoir),
                    "early_frontier_reservoir": len(early_frontier_reservoir),
                    "contextual_isr_reservoir": len(contextual_isr_reservoir),
                    "semantic_vector_cleanup": len(semantic_vector_cleanup),
                    "stream_input_replay": len(stream_input_replay),
                    "stream_input_replay_after_threads": len(stream_input_after_threads),
                    "semantic_frontier_flush": len(semantic_frontier_flush),
                    "semantic_frontier_drain": len(semantic_frontier_drain),
                    "stream_input_replay_post_frontier": len(post_frontier_stream_input),
                    "direct_call_continuation": len(direct_call_continuation),
                    "late_direct_call_frontier_drain": len(late_direct_call_frontier_drain),
                    "late_direct_call_summary_return_drain": len(late_direct_call_summary_return_drain),
                    "rtos_thread_entry": len(rtos_thread_entry),
                    "direct_call_summary_return": len(direct_call_summary_return),
                    "frontier_successor_replay": len(frontier_successor_replay),
                    "frontier_successor_replay_tail": len(frontier_successor_replay_tail),
                    "contextual_isr_frontier_targeted": len(contextual_isr_frontier_targeted),
                    "vector_only_cleanup": len(vector_cleanup_targeted),
                    "deadline_drain": len(deadline_drain_covered),
                    "switch_frontier_targeted": len(switch_frontier_targeted),
                    "frontier_targeted": len(frontier_targeted),
                    "path_naturalization": len(path_naturalization),
                    "path_naturalization_final": len(path_naturalization_final),
                },
                "dynamic_successor_summary": {
                    "sources": len(runner.dynamic_successors),
                    "edges": sum(len(successors) for successors in runner.dynamic_successors.values()),
                },
                "static_reachability": runner.static_reachability_summary(),
                "uncovered_summary": runner.uncovered_coverage_summary(),
                "contextual_isr_frontier_rounds": contextual_isr_frontier_rounds,
                "semantic_vector_cleanup_rounds": semantic_vector_cleanup_rounds,
                "vector_cleanup_rounds": vector_cleanup_rounds,
                "deadline_drain_rounds": deadline_drain_rounds,
                "switch_frontier_rounds": switch_frontier_rounds,
                "frontier_rounds": frontier_rounds,
                "frontier_cycles": frontier_cycles,
                "progress_jsonl_file": str(progress_jsonl_file),
                "progress_interval_seconds": progress_interval_seconds,
                "progress_monitor": progress_monitor.status(),
                "branch_catalog_summary": {
                    "static_conditional_branch_points": branch_catalog["static_conditional_branch_points"],
                    "main_path_branch_events": branch_catalog["main_path_branch_events"],
                    "main_path_branch_points": branch_catalog["main_path_branch_points"],
                    "reservoir_discovered_branch_events": branch_catalog["reservoir_discovered_branch_events"],
                    "reservoir_new_branch_points": branch_catalog["reservoir_new_branch_points"],
                },
            },
        )
    except BaseException as exc:
        try:
            runner.replay_executor.fail_active_stages(exc)
        except Exception as lifecycle_exc:
            logging.getLogger(__name__).warning(
                "failed to finalize active scheduler stages: %s",
                lifecycle_exc,
            )
        progress_monitor.set_stage("error")
        try:
            progress_monitor.stop(
                "run_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception as monitor_exc:
            logging.getLogger(__name__).warning(
                "failed to write final progress event: %s",
                monitor_exc,
            )
        finally:
            runner.replay_executor.unbind_runtime(scheduler_runtime)
        raise
    try:
        try:
            progress_monitor.stop(
                "run_complete",
                extra={
                    "report_file": str(report_file),
                    "progress_file": str(progress_jsonl_file),
                    "strict_real_entry_replayable": bool(report.get("strict_real_entry_replayable")) if isinstance(report, dict) else None,
                },
            )
        finally:
            # Persist once on both the normal and monitor-failure paths.  This
            # keeps the completed emulation result recoverable without doubling
            # report encoding, write bandwidth, and fsync traffic.
            if isinstance(report, dict):
                report["progress_monitor"] = progress_monitor.status()
                atomic_json_dump(report, report_file, indent=None)
    finally:
        runner.replay_executor.unbind_runtime(scheduler_runtime)


if __name__ == "__main__":
    main()
