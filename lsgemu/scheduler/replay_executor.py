"""Replay execution boundary for the interleaved scheduler.

The executor owns stage lifecycle, timing, cleanup and invocation contracts.
It delegates the actual Unicorn exploration kernels to ``HistoricalRunner``;
therefore extracting this layer cannot alter the coverage oracle or candidate
semantics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from .contracts import SchedulerRuntime, result_size
from .stall_watchdog import (
    STAGE_TRUNCATION_REASON,
    STOP_REASON as STALL_WATCHDOG_STOP_REASON,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default) or default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(eq=False)
class _StageExecution:
    name: str
    wall_start: float
    cpu_start: float
    lifecycle_error: Optional[dict[str, str]] = None


class ReplayExecutor:
    component_name = "replay_executor"

    def __init__(self, runner: Any, runtime: Optional[SchedulerRuntime] = None):
        self.runner = runner
        self.runtime = runtime
        self._active_stages: list[_StageExecution] = []

    @property
    def active_stage_depth(self) -> int:
        return len(self._active_stages)

    def active_stage_names(self) -> list[str]:
        return [stage.name for stage in self._active_stages]

    def bind_runtime(self, runtime: SchedulerRuntime) -> None:
        if runtime.runner is not self.runner:
            raise ValueError("scheduler runtime belongs to a different runner")
        if self._active_stages:
            raise RuntimeError("cannot replace scheduler runtime while a stage is active")
        self.runtime = runtime

    def unbind_runtime(self, runtime: Optional[SchedulerRuntime] = None) -> bool:
        """Release callbacks captured by a completed interleaved run."""

        if runtime is not None and self.runtime is not runtime:
            return False
        self.runtime = None
        self._active_stages.clear()
        return True

    def _runtime(self) -> SchedulerRuntime:
        if self.runtime is None:
            raise RuntimeError("ReplayExecutor has not been bound to a stage runtime")
        return self.runtime

    def _stall_watchdog_should_stop(self) -> bool:
        """True once the run-level no-new-BB watchdog has fired."""
        watchdog = getattr(self.runner, "stall_watchdog", None)
        return watchdog is not None and watchdog.should_stop()

    def _stall_watchdog_truncates(self, stage_name: str) -> bool:
        """True while the stage-level watchdog has flagged this stage (or an
        enclosing budget-owner stage) for truncation."""
        watchdog = getattr(self.runner, "stall_watchdog", None)
        should_truncate = getattr(watchdog, "should_truncate_stage", None)
        return watchdog is not None and callable(should_truncate) and bool(
            should_truncate(stage_name)
        )

    def _notify_stage_begin(self, stage_name: str) -> None:
        watchdog = getattr(self.runner, "stall_watchdog", None)
        notify = getattr(watchdog, "notify_stage_begin", None)
        if callable(notify):
            try:
                notify(stage_name)
            except Exception:
                pass

    def _notify_stage_end(
        self,
        stage_name: str,
        *,
        wall_seconds: float,
        skipped: bool = False,
        failed: bool = False,
    ) -> None:
        watchdog = getattr(self.runner, "stall_watchdog", None)
        notify = getattr(watchdog, "notify_stage_end", None)
        if callable(notify):
            try:
                notify(
                    stage_name,
                    wall_seconds=wall_seconds,
                    skipped=skipped,
                    failed=failed,
                )
            except Exception:
                pass

    @staticmethod
    def _error_details(exc: BaseException) -> dict[str, str]:
        return {
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }

    def begin_stage(
        self,
        stage_name: str,
        *,
        extra: Optional[dict[str, object]] = None,
    ) -> _StageExecution:
        """Begin a simple or composite stage under one lifecycle contract."""

        runtime = self._runtime()
        monitor = runtime.progress_monitor
        runner = self.runner
        monitor.set_stage(stage_name)
        lifecycle_error = None
        try:
            runner.set_lifecycle_stage(stage_name)
        except Exception as exc:
            lifecycle_error = self._error_details(exc)
        monitor.snapshot(
            "stage_start",
            stage=stage_name,
            extra=dict(extra or {}),
        )
        self._notify_stage_begin(stage_name)
        stage = _StageExecution(
            name=str(stage_name),
            wall_start=time.time(),
            cpu_start=time.process_time(),
            lifecycle_error=lifecycle_error,
        )
        self._active_stages.append(stage)
        return stage

    def _cleanup_stage(self, stage_name: str) -> tuple[int, Optional[dict[str, str]]]:
        try:
            cleaned = self.runner._dispose_active_temp_emulators(
                stage_name=stage_name,
                reason="stage_end",
            )
            return int(cleaned or 0), None
        except Exception as exc:
            return 0, self._error_details(exc)

    @staticmethod
    def _timing(stage: _StageExecution) -> dict[str, object]:
        stage_wall_seconds = max(0.0, time.time() - stage.wall_start)
        stage_cpu_seconds = max(0.0, time.process_time() - stage.cpu_start)
        stage_cpu_wall_ratio = (
            stage_cpu_seconds / stage_wall_seconds
            if stage_wall_seconds > 0.0
            else 0.0
        )
        return {
            "stage_wall_seconds": round(stage_wall_seconds, 3),
            "stage_process_cpu_seconds": round(stage_cpu_seconds, 3),
            "stage_cpu_wall_ratio": round(stage_cpu_wall_ratio, 3),
            "stage_cpu_starved": bool(stage_wall_seconds >= 1.0 and stage_cpu_wall_ratio < 0.55),
        }

    def _pop_stage(self, stage: _StageExecution) -> None:
        if self._active_stages and self._active_stages[-1] is stage:
            self._active_stages.pop()
        elif stage in self._active_stages:
            self._active_stages.remove(stage)
        self._restore_active_stage_context()

    def _restore_active_stage_context(self) -> None:
        if not self._active_stages or self.runtime is None:
            return
        parent_name = self._active_stages[-1].name
        monitor = self.runtime.progress_monitor
        try:
            monitor.set_stage(parent_name)
        except Exception:
            pass
        try:
            self.runner.set_lifecycle_stage(parent_name)
        except Exception:
            pass

    def finish_stage(
        self,
        stage: _StageExecution,
        result: Any,
        *,
        coverage_counted: Optional[bool] = None,
        extra: Optional[dict[str, object]] = None,
    ) -> Any:
        """Finish a stage and restore any enclosing composite-stage context."""

        if stage not in self._active_stages:
            raise RuntimeError(f"stage is not active: {stage.name}")
        runtime = self._runtime()
        monitor = runtime.progress_monitor
        runner = self.runner
        cleaned, cleanup_error = self._cleanup_stage(stage.name)
        timing = self._timing(stage)
        metadata: dict[str, object] = {
            **timing,
            "active_temp_emulators_cleaned": cleaned,
        }
        if coverage_counted is not None:
            metadata["coverage_counted"] = bool(coverage_counted)
        if stage.lifecycle_error:
            metadata["lifecycle_stage_error"] = dict(stage.lifecycle_error)
        if cleanup_error:
            metadata["active_temp_emulator_cleanup_error"] = dict(cleanup_error)
        if extra:
            metadata.update(extra)
        runner.phase_metadata.setdefault(stage.name, {}).update(metadata)
        event_extra: dict[str, object] = {
            "stage_covered_bbs": result_size(result),
            "global_covered_bbs": len(runner.global_coverage),
            **timing,
        }
        if cleanup_error:
            event_extra["active_temp_emulator_cleanup_error"] = dict(cleanup_error)
        if extra:
            event_extra.update(extra)
        try:
            monitor.snapshot("stage_end", stage=stage.name, extra=event_extra)
        finally:
            self._notify_stage_end(
                stage.name,
                wall_seconds=float(timing.get("stage_wall_seconds") or 0.0),
            )
            self._pop_stage(stage)
        return result

    def fail_stage(self, stage: _StageExecution, exc: BaseException) -> None:
        """Record a failed stage without replacing the original exception."""

        if stage not in self._active_stages:
            return
        cleaned, cleanup_error = self._cleanup_stage(stage.name)
        timing = self._timing(stage)
        error_details = self._error_details(exc)
        metadata: dict[str, object] = {
            **timing,
            "stage_failed": True,
            "stage_error": error_details,
            "active_temp_emulators_cleaned": cleaned,
        }
        if stage.lifecycle_error:
            metadata["lifecycle_stage_error"] = dict(stage.lifecycle_error)
        if cleanup_error:
            metadata["active_temp_emulator_cleanup_error"] = dict(cleanup_error)
        try:
            self.runner.phase_metadata.setdefault(stage.name, {}).update(metadata)
        except Exception:
            pass
        try:
            self._runtime().progress_monitor.snapshot_unchanged(
                "stage_failed",
                stage=stage.name,
                extra={
                    **timing,
                    "stage_error": error_details,
                    "active_temp_emulators_cleaned": cleaned,
                    "active_temp_emulator_cleanup_error": cleanup_error,
                    "global_covered_bbs": len(getattr(self.runner, "global_coverage", set())),
                },
            )
        except Exception:
            pass
        finally:
            self._notify_stage_end(
                stage.name,
                wall_seconds=float(timing.get("stage_wall_seconds") or 0.0),
                failed=True,
            )
            self._pop_stage(stage)

    def fail_active_stages(self, exc: BaseException) -> int:
        """Close all enclosing composite stages after an escaping failure."""

        failed = 0
        while self._active_stages:
            self.fail_stage(self._active_stages[-1], exc)
            failed += 1
        return failed

    def run_stage(self, stage_name: str, func):
        """Run one stage with the historical invocation semantics unchanged."""

        stage = self.begin_stage(stage_name)
        try:
            result = func()
        except BaseException as exc:
            self.fail_stage(stage, exc)
            raise
        return self.finish_stage(stage, result)

    def skip_stage(self, stage_name: str, reason: str) -> None:
        runtime = self._runtime()
        monitor = runtime.progress_monitor
        runner = self.runner
        monitor.set_stage(stage_name)
        runner.phase_coverage.setdefault(stage_name, set())
        runner.phase_metadata.setdefault(
            stage_name,
            {
                "covered_bbs": 0,
                "new_bbs": 0,
                "coverage_counted": False,
                "skipped": True,
                "skip_reason": str(reason),
                "remaining_wallclock_seconds": runtime.remaining_wallclock_seconds(),
            },
        )
        monitor.snapshot_unchanged(
            "stage_skipped",
            stage=stage_name,
            extra={"skip_reason": str(reason)},
        )
        self._notify_stage_end(stage_name, wall_seconds=0.0, skipped=True)
        self._restore_active_stage_context()

    def run_frontier_successor(
        self,
        stage_name: str,
        seconds: int,
        *,
        target_bbs: set[int] | None = None,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        reserve_after_seconds: int = 0,
    ) -> set[int]:
        """Invoke successor replay with the existing stage-level caps."""

        runtime = self._runtime()
        args = runtime.args
        if getattr(args, "disable_frontier_stages", False):
            self.skip_stage(stage_name, "frontier_ablation_disabled")
            return set()
        if self._stall_watchdog_should_stop():
            self.skip_stage(stage_name, STALL_WATCHDOG_STOP_REASON)
            return set()
        if self._stall_watchdog_truncates(stage_name):
            # A stage (or an enclosing budget-owner stage) already flagged by
            # the stage-level watchdog: do not start new sub-work inside it.
            self.skip_stage(stage_name, STAGE_TRUNCATION_REASON)
            return set()
        seconds = runtime.clamp_enabled_stage_seconds(
            max(0, int(seconds)),
            reserve_after_seconds=max(0, int(reserve_after_seconds)),
        )
        if seconds <= 0 or not runtime.has_stage_wallclock_budget(
            seconds,
            min_fraction=0.10,
            reserve_after_seconds=max(0, int(reserve_after_seconds)),
        ):
            self.skip_stage(stage_name, "disabled_or_no_budget")
            return set()

        is_flush = (
            stage_name != "frontier_successor_replay_early"
            and "successor_flush" in stage_name
            and not target_bbs
        )
        is_late_probe = (
            stage_name != "frontier_successor_replay_early"
            and stage_name.endswith("_tail")
            and not target_bbs
            and _env_int("LSGEMU_FRONTIER_SUCCESSOR_TAIL_PROBE", 0) > 0
        )
        is_probe = is_flush or is_late_probe
        target_probe_after_hit_bbs = 0
        if is_probe:
            target_probe_after_hit_bbs = max(
                0,
                _env_int(
                    "LSGEMU_FRONTIER_SUCCESSOR_PROBE_TAIL_BBS",
                    16 if runtime.short_probe_mode() else 32,
                ),
            )
        short_sampler = (
            runtime.short_probe_mode()
            and _env_int("LSGEMU_SHORT_FRONTIER_SAMPLER", 0) > 0
            and stage_name != "frontier_successor_replay_early"
            and ("flush" in stage_name or "drain" in stage_name or stage_name.endswith("_tail"))
        )
        if short_sampler:
            replay_instructions = max(
                1000,
                min(
                    args.frontier_successor_replay_instructions,
                    _env_int("LSGEMU_SHORT_FRONTIER_SAMPLER_INSTRUCTIONS", 12000),
                ),
            )
            replay_timeout = max(
                50000,
                min(
                    args.frontier_successor_replay_timeout_us,
                    _env_int("LSGEMU_SHORT_FRONTIER_SAMPLER_TIMEOUT_US", 200000),
                ),
            )
            no_new_bb_limit = max(
                64,
                min(
                    args.frontier_successor_replay_no_new_bbs,
                    _env_int("LSGEMU_SHORT_FRONTIER_SAMPLER_NO_NEW_BBS", 512),
                ),
            )
        else:
            replay_instructions = (
                max(args.frontier_successor_replay_instructions, 80000)
                if runtime.short_probe_mode() and stage_name == "frontier_successor_replay_early"
                else args.frontier_successor_replay_instructions
            )
            replay_timeout = (
                max(args.frontier_successor_replay_timeout_us, 1000000)
                if runtime.short_probe_mode() and stage_name == "frontier_successor_replay_early"
                else args.frontier_successor_replay_timeout_us
            )
            no_new_bb_limit = (
                max(args.frontier_successor_replay_no_new_bbs, 4096)
                if runtime.short_probe_mode() and stage_name == "frontier_successor_replay_early"
                else args.frontier_successor_replay_no_new_bbs
            )
            if is_probe and runtime.short_probe_mode():
                replay_instructions = max(
                    1000,
                    min(
                        replay_instructions,
                        _env_int("LSGEMU_SHORT_SUCCESSOR_PROBE_INSTRUCTIONS", 20000),
                    ),
                )
                replay_timeout = max(
                    50000,
                    min(
                        replay_timeout,
                        _env_int("LSGEMU_SHORT_SUCCESSOR_PROBE_TIMEOUT_US", 250000),
                    ),
                )
                no_new_bb_limit = max(
                    64,
                    min(
                        no_new_bb_limit,
                        _env_int("LSGEMU_SHORT_SUCCESSOR_PROBE_NO_NEW_BBS", 512),
                    ),
                )

        covered = self.run_stage(
            stage_name,
            lambda: self.runner.run_frontier_successor_replay_exploration(
                time_limit_seconds=seconds,
                max_tasks=(
                    max_tasks
                    if max_tasks is not None
                    else args.frontier_successor_replay_max_tasks
                ),
                max_instructions=replay_instructions,
                replay_timeout=replay_timeout,
                no_new_bb_limit=no_new_bb_limit,
                max_targets=(
                    max_targets
                    if max_targets is not None
                    else args.frontier_successor_replay_max_targets
                ),
                variants_per_branch=args.frontier_successor_replay_variants_per_branch,
                target_bbs=target_bbs,
                target_probe_after_hit_bbs=target_probe_after_hit_bbs,
                refresh_after_progress_tasks=(
                    max(1, _env_int("LSGEMU_FRONTIER_SUCCESSOR_PROBE_REFRESH_TASKS", 8))
                    if is_probe
                    else 1
                ),
                max_successor_hits_per_stage=(
                    max(1, _env_int("LSGEMU_FRONTIER_SUCCESSOR_PROBE_MAX_HITS_PER_SUCCESSOR", 1))
                    if is_probe
                    else 0
                ),
                max_successor_zero_new_per_stage=(
                    max(1, _env_int("LSGEMU_FRONTIER_SUCCESSOR_PROBE_MAX_ZERO_NEW_PER_SUCCESSOR", 1))
                    if is_probe
                    else 0
                ),
                phase_name=stage_name,
            ),
        )
        self.runner.phase_metadata.setdefault(stage_name, {}).update({
            "short_frontier_sampler": bool(short_sampler),
            "successor_probe_stage": bool(is_probe),
            "configured_replay_instructions": int(replay_instructions),
            "configured_replay_timeout_us": int(replay_timeout),
            "configured_no_new_bb_limit": int(no_new_bb_limit),
            "configured_target_probe_after_hit_bbs": int(target_probe_after_hit_bbs),
        })
        return covered

    def run_direct_call(
        self,
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        runtime = self._runtime()
        args = runtime.args
        if getattr(args, "disable_direct_stream_thread_stages", False):
            self.skip_stage(stage_name, "direct_stream_thread_ablation_disabled")
            return set()
        if self._stall_watchdog_should_stop():
            self.skip_stage(stage_name, STALL_WATCHDOG_STOP_REASON)
            return set()
        if self._stall_watchdog_truncates(stage_name):
            self.skip_stage(stage_name, STAGE_TRUNCATION_REASON)
            return set()
        if seconds <= 0 or not runtime.has_wallclock_budget(1):
            self.skip_stage(stage_name, "disabled_or_no_budget")
            return set()
        return self.run_stage(
            stage_name,
            lambda: self.runner.run_direct_call_continuation_exploration(
                time_limit_seconds=seconds,
                max_tasks=(max_tasks if max_tasks is not None else args.direct_call_continuation_max_tasks),
                max_instructions=args.direct_call_continuation_instructions,
                replay_timeout=args.direct_call_continuation_replay_timeout_us,
                no_new_bb_limit=args.direct_call_continuation_no_new_bbs,
                max_targets=(max_targets if max_targets is not None else args.direct_call_continuation_max_targets),
                variants_per_call=args.direct_call_continuation_variants_per_call,
                low_yield_stop_tasks=args.direct_call_continuation_low_yield_stop_tasks,
                low_yield_min_tasks=args.direct_call_continuation_low_yield_min_tasks,
                target_bbs=target_bbs,
                phase_name=stage_name,
            ),
        )

    def run_summary_return(
        self,
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        runtime = self._runtime()
        args = runtime.args
        if getattr(args, "disable_direct_stream_thread_stages", False):
            self.skip_stage(stage_name, "direct_stream_thread_ablation_disabled")
            return set()
        if self._stall_watchdog_should_stop():
            self.skip_stage(stage_name, STALL_WATCHDOG_STOP_REASON)
            return set()
        if self._stall_watchdog_truncates(stage_name):
            self.skip_stage(stage_name, STAGE_TRUNCATION_REASON)
            return set()
        if seconds <= 0 or not runtime.has_wallclock_budget(1):
            self.skip_stage(stage_name, "disabled_or_no_budget")
            return set()
        return self.run_stage(
            stage_name,
            lambda: self.runner.run_direct_call_summary_return_exploration(
                time_limit_seconds=seconds,
                max_tasks=(max_tasks if max_tasks is not None else args.direct_call_summary_return_max_tasks),
                max_instructions=args.direct_call_summary_return_instructions,
                replay_timeout=args.direct_call_summary_return_replay_timeout_us,
                no_new_bb_limit=args.direct_call_summary_return_no_new_bbs,
                max_targets=(max_targets if max_targets is not None else args.direct_call_summary_return_max_targets),
                variants_per_call=args.direct_call_summary_return_variants_per_call,
                low_yield_stop_tasks=args.direct_call_summary_return_low_yield_stop_tasks,
                low_yield_min_tasks=args.direct_call_summary_return_low_yield_min_tasks,
                target_bbs=target_bbs,
                phase_name=stage_name,
            ),
        )

    def run_rtos_thread(
        self,
        stage_name: str,
        seconds: int,
        *,
        max_tasks: int | None = None,
        max_targets: int | None = None,
        target_bbs: set[int] | None = None,
    ) -> set[int]:
        runtime = self._runtime()
        args = runtime.args
        if getattr(args, "disable_direct_stream_thread_stages", False):
            self.skip_stage(stage_name, "direct_stream_thread_ablation_disabled")
            return set()
        if self._stall_watchdog_should_stop():
            self.skip_stage(stage_name, STALL_WATCHDOG_STOP_REASON)
            return set()
        if self._stall_watchdog_truncates(stage_name):
            self.skip_stage(stage_name, STAGE_TRUNCATION_REASON)
            return set()
        if seconds <= 0 or not runtime.has_wallclock_budget(1):
            self.skip_stage(stage_name, "disabled_or_no_budget")
            return set()
        return self.run_stage(
            stage_name,
            lambda: self.runner.context_recovery.rtos_thread(
                time_limit_seconds=seconds,
                max_tasks=(max_tasks if max_tasks is not None else args.rtos_thread_entry_max_tasks),
                max_instructions=args.rtos_thread_entry_instructions,
                replay_timeout=args.rtos_thread_entry_replay_timeout_us,
                no_new_bb_limit=args.rtos_thread_entry_no_new_bbs,
                max_targets=(max_targets if max_targets is not None else args.rtos_thread_entry_max_targets),
                variants_per_call=args.rtos_thread_entry_variants_per_call,
                target_bbs=target_bbs,
                phase_name=stage_name,
            ),
        )
