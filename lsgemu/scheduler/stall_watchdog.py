"""Stage-level multi-ledger stall watchdog for long campaign budgets.

Round-3 redesign (2026-09-19), driven by measured counterexamples from the
24h ardupilot/Pixhawk1 control arm
(``campaign_24h_ardupilot_pixhawk1_NOLLM_control_20260918``):

* ``frontier_successor_replay_tail`` held ``covered_bbs`` flat at 1336 for
  9.68 h while ``counterfactual_only_bbs`` kept growing (470 -> 816, +346)
  until t=5.33h; the +346 only entered ``covered_bbs`` at the stage boundary
  (t=13.39h) when the phase set was merged into ``global_coverage``.  A
  run-level wallclock watchdog with any threshold <= 9.68h would have killed
  that run around t~5h and thrown away 20.6% of the final coverage.
* ``deadline_drain`` afterwards burned 4.31 h with every ledger frozen.

Consequences implemented here:

1. **Multi-ledger productivity.**  A production unit counts as progress when
   ANY ledger advances: the covered ledger (per-attempt
   ``len(global_coverage | phase_covered)`` from the exploration kernels and
   interval ``covered_bbs`` from the monitor) or any ``evidence_status``
   ledger (``natural_supported_bbs`` / ``counterfactual_only_bbs`` /
   ``unclassified_evidence_bbs`` / ``validated_replay_bbs`` /
   ``diagnostic_replay_bbs`` / ``canonical_unclassified_bbs``).  The evidence
   ledgers are computed by the progress monitor per interval (600s in the
   campaign, 300s default) and reused verbatim; the covered ledger updates
   per attempt.  A stall clock can therefore only reach its threshold when
   the fine-grained covered signal AND the interval-granularity evidence
   signal are both frozen ("缺一不可"): whichever observes progress first
   resets the clock.  If the monitor thread dies and no evidence ledger is
   ever seen, the watchdog degrades to the covered-only signal rather than
   going blind.
2. **Stage-level truncation (the default action).**  A stage whose ledgers
   are frozen for ``threshold_seconds`` (default 5400s = 90 min) is *ended*
   through its ordinary exit path: the exploration kernels check
   ``should_truncate_stage`` in their budget-exhausted predicate, break
   between attempts, and still run their ``_record_phase`` merge, so
   phase/evidence state committed at the truncation point exactly as at a
   natural stage end.  The pipeline then continues with the next stage.
3. **Run-level stop is only a fallback** (reason stays
   ``stall_no_new_bb_watchdog``; the report records ``level``):
   * ``level="stage"``: ``max_consecutive_zero_stages`` (default 3)
     consecutive depth-1 stages that each ran at least
     ``min_zero_productivity_stage_seconds`` (default 1800s) without moving
     any ledger.  Skipped/exempt/failed stages and shorter stages are
     neutral -- they neither count nor break the streak.
   * ``level="run"``: a safety floor of ``run_stall_threshold_seconds``
     (default 28800s = 8h, above the 7.85h largest legitimate dead window
     ever measured; 0 disables) of continuously frozen ledgers across
     non-exempt stages.  The round-1 run-level 90-min wallclock stop is gone.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Mapping, Optional

STOP_REASON = "stall_no_new_bb_watchdog"
STAGE_TRUNCATION_REASON = "stall_watchdog_stage_truncated"
STATUS_SCHEMA = "lsgemu.stall_watchdog.v2"

DEFAULT_STALL_THRESHOLD_SECONDS = 5400
DEFAULT_MIN_ZERO_YIELD_UNITS = 8
DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES = 3
DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS = 1800
# 8h floor: the largest measured *legitimate* all-ledger freeze (7.85h,
# t=5.33h->13.18h in the control arm) ended with a productive stage boundary
# merge, so the floor must sit above it.  0 restores a pure stage-level
# watchdog with no run-level wallclock backstop.
DEFAULT_RUN_STALL_THRESHOLD_SECONDS = 28800
DEFAULT_WINDDOWN_MAX_SECONDS = 900

# Ledger keys from the monitor's evidence_status whose growth counts as
# production.  Any single increase resets every stall clock.
EVIDENCE_LEDGER_KEYS = (
    "natural_supported_bbs",
    "counterfactual_only_bbs",
    "unclassified_evidence_bbs",
    "validated_replay_bbs",
    "diagnostic_replay_bbs",
    "canonical_unclassified_bbs",
)

# Stages that legitimately spend long stretches without adding covered BBs:
#   path_naturalization(_final) -- replays learned constraints, produces
#     obligations/facts, not coverage, and is SIGTERM-hot near the deadline;
#   vector_only_cleanup -- bounded targeted rounds over vector-only targets
#     (its own stale-round exit applies);
#   setup/finalize/error -- orchestration bookkeeping, never produces coverage.
# ``deadline_drain`` was removed in round 3: the control arm measured 4.31h
# of frozen ledgers inside the drain loop, so it is now monitored and
# truncatable like any coverage-expected stage.
# Substring match keeps suffixed variants (``path_naturalization_final``)
# inside the exemption.
DEFAULT_EXEMPT_STAGE_TOKENS = (
    "path_naturalization",
    "vector_only_cleanup",
    "setup",
    "finalize",
    "finalizing_report",
    "error",
    "teardown",
    "checkpoint",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_tokens(raw: str) -> tuple[str, ...]:
    return tuple(
        token.strip()
        for token in str(raw or "").split(",")
        if token.strip()
    )


class _StageFrame:
    """One ``ReplayExecutor`` stage bracket (nesting = sub-stages)."""

    __slots__ = ("name", "depth", "started_at", "epoch_at_begin", "exempt")

    def __init__(self, name: str, depth: int, started_at: float, epoch: int, exempt: bool):
        self.name = name
        self.depth = depth
        self.started_at = started_at
        self.epoch_at_begin = epoch
        self.exempt = exempt


class StallWatchdog:
    """Track stage-level multi-ledger stall; truncate stages, stop runs rarely.

    ``observe`` is called from the progress monitor (interval samples and
    stage events, carrying ``evidence_status`` ledger counts) and from long
    exploration kernels (per completed attempt, carrying the phase-union
    covered count).  ``ReplayExecutor`` brackets every stage with
    ``notify_stage_begin``/``notify_stage_end`` so stall time is attributed
    to the depth-1 budget-owner stage even while monitor stage names rotate
    through drain sub-stages.

    Actions, in order of preference:

    * stage truncation -- current depth-1 stage frozen on every ledger for
      ``threshold_seconds`` plus ``min_zero_yield_units`` consecutive
      zero-yield units (attempts/interval samples): flag it via
      ``should_truncate_stage``; the stage exits through its normal path
      (merge included) and the pipeline continues;
    * run stop, ``level="stage"`` -- ``max_consecutive_zero_stages``
      consecutive eligible depth-1 stages with no ledger movement at all;
    * run stop, ``level="run"`` -- ``run_stall_threshold_seconds`` of
      continuous non-exempt all-ledger freeze (safety floor, default 8h).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold_seconds: int = DEFAULT_STALL_THRESHOLD_SECONDS,
        min_zero_yield_units: int = DEFAULT_MIN_ZERO_YIELD_UNITS,
        exempt_stage_tokens: tuple[str, ...] = DEFAULT_EXEMPT_STAGE_TOKENS,
        max_consecutive_zero_stages: int = DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES,
        min_zero_productivity_stage_seconds: int = DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS,
        run_stall_threshold_seconds: int = DEFAULT_RUN_STALL_THRESHOLD_SECONDS,
        started_at: Optional[float] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        self.enabled = bool(enabled)
        self.threshold_seconds = max(0, int(threshold_seconds))
        self.min_zero_yield_units = max(0, int(min_zero_yield_units))
        self.exempt_stage_tokens = tuple(exempt_stage_tokens) or (
            DEFAULT_EXEMPT_STAGE_TOKENS
        )
        self.max_consecutive_zero_stages = max(0, int(max_consecutive_zero_stages))
        self.min_zero_productivity_stage_seconds = max(
            0, int(min_zero_productivity_stage_seconds)
        )
        self.run_stall_threshold_seconds = max(
            0, int(run_stall_threshold_seconds)
        )
        self._now = now or time.time
        self._lock = threading.RLock()
        self._started_at = float(started_at if started_at is not None else self._now())
        self._last_observe_wall = self._started_at
        self._last_progress_wall = self._started_at
        # Exemption of the stage covering the gap that the next observe will
        # accrue; the run starts inside monitor bookkeeping ("setup").
        self._gap_stage_exempt = True
        # Multi-ledger watermarks: "covered_bbs" plus the evidence ledgers.
        self._ledger_max: dict[str, int] = {}
        self._max_covered_bbs: Optional[int] = None
        self._evidence_ledgers_seen = False
        self._progress_epoch = 0
        # Stage bookkeeping (depth-1 = budget owner).
        self._stage_stack: list[_StageFrame] = []
        self._fallback_stage_name = ""
        self._stage_stall_seconds = 0.0
        self._run_stall_seconds = 0.0
        self._zero_yield_units = 0
        self._last_stage = ""
        self._observations = 0
        # Truncation / stop state.
        self._truncate_stage_name: Optional[str] = None
        self._stage_truncations: list[dict[str, object]] = []
        self._unreported_truncations = 0
        self._consecutive_zero_stages: list[dict[str, object]] = []
        self._stop_requested = False
        self._triggered: Optional[dict[str, object]] = None

    # ------------------------------------------------------------------ #
    # Construction / classification
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, args: object = None, *, started_at: Optional[float] = None) -> "StallWatchdog":
        """Resolve CLI > env > default precedence for the gateway CLI runner."""
        cli_threshold = int(getattr(args, "stall_watchdog_threshold_seconds", 0) or 0)
        threshold = (
            cli_threshold
            if cli_threshold > 0
            else _env_int("LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS", DEFAULT_STALL_THRESHOLD_SECONDS)
        )
        cli_units = int(getattr(args, "stall_watchdog_min_zero_yield_units", 0) or 0)
        min_units = (
            cli_units
            if cli_units > 0
            else _env_int("LSGEMU_STALL_WATCHDOG_MIN_ZERO_YIELD_UNITS", DEFAULT_MIN_ZERO_YIELD_UNITS)
        )
        cli_exempt = str(getattr(args, "stall_watchdog_exempt_stages", "") or "")
        exempt_raw = (
            cli_exempt
            if cli_exempt.strip()
            else os.environ.get("LSGEMU_STALL_WATCHDOG_EXEMPT_STAGES", "")
        )
        exempt_tokens = DEFAULT_EXEMPT_STAGE_TOKENS + _parse_tokens(exempt_raw)

        def _resolve_int(cli_name: str, env_name: str, default: int) -> int:
            # ``None`` (argparse default) means "not given"; an explicit CLI 0
            # is meaningful ("off"/"count nothing") and must win over env.
            cli_value = getattr(args, cli_name, None)
            if cli_value is not None:
                try:
                    return int(cli_value)
                except (TypeError, ValueError):
                    pass
            return _env_int(env_name, default)

        max_zero_stages = _resolve_int(
            "stall_watchdog_max_consecutive_zero_stages",
            "LSGEMU_STALL_WATCHDOG_MAX_CONSECUTIVE_ZERO_STAGES",
            DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES,
        )
        min_stage_seconds = _resolve_int(
            "stall_watchdog_min_zero_productivity_stage_seconds",
            "LSGEMU_STALL_WATCHDOG_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS",
            DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS,
        )
        run_stall_seconds = _resolve_int(
            "stall_watchdog_run_stall_seconds",
            "LSGEMU_STALL_WATCHDOG_RUN_STALL_SECONDS",
            DEFAULT_RUN_STALL_THRESHOLD_SECONDS,
        )
        cli_disabled = bool(getattr(args, "disable_stall_watchdog", False))
        enabled = (
            False
            if cli_disabled
            else _env_flag("LSGEMU_STALL_WATCHDOG_ENABLED", True)
        )
        # A non-positive threshold is an explicit "off" for the watchdog.
        if threshold <= 0:
            enabled = False
        return cls(
            enabled=enabled,
            threshold_seconds=threshold,
            min_zero_yield_units=min_units,
            exempt_stage_tokens=exempt_tokens,
            max_consecutive_zero_stages=max_zero_stages,
            min_zero_productivity_stage_seconds=min_stage_seconds,
            run_stall_threshold_seconds=run_stall_seconds,
            started_at=started_at,
        )

    def stage_is_exempt(self, stage: str) -> bool:
        name = str(stage or "")
        return any(token and token in name for token in self.exempt_stage_tokens)

    # ------------------------------------------------------------------ #
    # Stage brackets (ReplayExecutor begin/finish/skip/fail)
    # ------------------------------------------------------------------ #

    def notify_stage_begin(self, stage_name: str) -> None:
        frame = _StageFrame(
            name=str(stage_name or "unknown"),
            depth=len(self._stage_stack),
            started_at=self._now(),
            epoch=self._progress_epoch,
            exempt=self.stage_is_exempt(stage_name),
        )
        self._stage_stack.append(frame)
        if frame.depth == 0:
            # A fresh budget-owner stage gets a fresh stage stall clock and
            # zero-yield unit counter; the run-level clock is only reset by
            # ledger progress.
            self._stage_stall_seconds = 0.0
            self._zero_yield_units = 0
            self._gap_stage_exempt = frame.exempt

    def notify_stage_end(
        self,
        stage_name: str,
        *,
        wall_seconds: float = 0.0,
        skipped: bool = False,
        failed: bool = False,
    ) -> None:
        name = str(stage_name or "unknown")
        index = next(
            (i for i in range(len(self._stage_stack) - 1, -1, -1)
             if self._stage_stack[i].name == name),
            None,
        )
        if index is None:
            return
        frame = self._stage_stack[index]
        # Proper nesting pops only the top frame; tolerate defensive pops of
        # stale names by dropping everything above the match as well.
        del self._stage_stack[index:]
        if self._truncate_stage_name == name:
            # The truncated stage has ended through its ordinary path.
            self._truncate_stage_name = None
        if frame.depth == 0:
            if (
                self.enabled
                and not skipped
                and not failed
                and not frame.exempt
                and self._progress_epoch == frame.epoch_at_begin
                and wall_seconds >= self.min_zero_productivity_stage_seconds
            ):
                self._consecutive_zero_stages.append({
                    "stage": name,
                    "wall_seconds": round(float(wall_seconds), 3),
                    "min_zero_productivity_stage_seconds": (
                        self.min_zero_productivity_stage_seconds
                    ),
                })
                self._evaluate_run_stop(name)
            self._stage_stall_seconds = 0.0
            self._zero_yield_units = 0
            self._gap_stage_exempt = (
                self._stage_stack[-1].exempt if self._stage_stack else True
            )

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #

    def _depth1_stage_name(self) -> str:
        if self._stage_stack:
            return self._stage_stack[0].name
        return self._fallback_stage_name

    def _current_stage_exempt(self) -> bool:
        if self._stage_stack:
            return self._stage_stack[0].exempt
        return self.stage_is_exempt(self._fallback_stage_name)

    def _ledger_snapshot(self) -> dict[str, int]:
        snapshot = dict(self._ledger_max)
        if self._max_covered_bbs is not None:
            snapshot["covered_bbs"] = int(self._max_covered_bbs)
        return snapshot

    def observe(
        self,
        *,
        stage: str,
        covered_bbs: Optional[int] = None,
        zero_yield_unit: bool = False,
        evidence_ledgers: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Record one telemetry point; accrue stall time and evaluate.

        ``evidence_ledgers`` is the monitor interval's ``evidence_status``
        mapping (interval granularity); ``covered_bbs`` is the per-attempt
        phase-union count (attempt granularity).  Any increase on any ledger
        resets every stall clock and the zero-productivity stage streak.
        """
        with self._lock:
            now = self._now()
            gap = max(0.0, now - self._last_observe_wall)
            if not self._gap_stage_exempt:
                self._stage_stall_seconds += gap
                self._run_stall_seconds += gap
            self._last_observe_wall = now
            stage_name = str(stage or "")
            if not self._stage_stack and self._fallback_stage_name:
                # No executor brackets (monitor-only contexts): an observed
                # stage-name change is an implicit stage boundary, so the
                # previous stage's stall clock and truncation flag must not
                # leak into the new stage.
                if stage_name != self._fallback_stage_name:
                    if (
                        self._truncate_stage_name is not None
                        and self._truncate_stage_name != stage_name
                    ):
                        self._truncate_stage_name = None
                    self._stage_stall_seconds = 0.0
                    self._zero_yield_units = 0
            self._fallback_stage_name = stage_name
            self._last_stage = stage_name
            self._observations += 1

            progressed = False
            if covered_bbs is not None:
                covered_value = int(covered_bbs)
                if (
                    self._max_covered_bbs is None
                    or covered_value > self._max_covered_bbs
                ):
                    self._max_covered_bbs = covered_value
                    progressed = True
                previous = self._ledger_max.get("covered_bbs")
                if previous is None or covered_value > previous:
                    self._ledger_max["covered_bbs"] = covered_value
            if evidence_ledgers:
                self._evidence_ledgers_seen = True
                for key in EVIDENCE_LEDGER_KEYS:
                    raw = evidence_ledgers.get(key)
                    if raw is None:
                        continue
                    try:
                        value = int(raw)
                    except (TypeError, ValueError):
                        continue
                    previous = self._ledger_max.get(key)
                    if previous is None or value > previous:
                        self._ledger_max[key] = value
                        progressed = True

            self._gap_stage_exempt = self._current_stage_exempt()

            if progressed:
                self._progress_epoch += 1
                self._last_progress_wall = now
                self._stage_stall_seconds = 0.0
                self._run_stall_seconds = 0.0
                self._zero_yield_units = 0
                self._consecutive_zero_stages = []
            elif zero_yield_unit:
                self._zero_yield_units += 1

            self._evaluate(now)

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #

    def _evaluate(self, now: float) -> None:
        if not self.enabled or self._stop_requested:
            return
        if self.run_stall_threshold_seconds > 0 and (
            self._run_stall_seconds >= self.run_stall_threshold_seconds
        ):
            self._request_stop(
                level="run",
                cause="run_stall_floor",
                stage=self._depth1_stage_name(),
                now=now,
            )
            return
        if self._truncate_stage_name is not None:
            return
        stage_name = self._depth1_stage_name()
        if not stage_name or self._current_stage_exempt():
            return
        if self._stage_stall_seconds < self.threshold_seconds:
            return
        if self._zero_yield_units < self.min_zero_yield_units:
            return
        self._truncate_stage_name = stage_name
        record: dict[str, object] = {
            "stage": stage_name,
            "level": "stage",
            "threshold_seconds": int(self.threshold_seconds),
            "stage_stall_seconds": round(self._stage_stall_seconds, 3),
            "run_stall_seconds": round(self._run_stall_seconds, 3),
            "zero_yield_units": int(self._zero_yield_units),
            "evidence_ledgers_observed": bool(self._evidence_ledgers_seen),
            "ledgers": self._ledger_snapshot(),
            "wallclock_time": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(now),
            ),
        }
        self._stage_truncations.append(record)
        self._unreported_truncations += 1

    def _evaluate_run_stop(self, stage_name: str) -> None:
        if not self.enabled or self._stop_requested:
            return
        if self.max_consecutive_zero_stages <= 0:
            return
        if len(self._consecutive_zero_stages) < self.max_consecutive_zero_stages:
            return
        self._request_stop(
            level="stage",
            cause="consecutive_zero_productivity_stages",
            stage=stage_name,
            now=self._now(),
        )

    def _request_stop(
        self,
        *,
        level: str,
        cause: str,
        stage: str,
        now: float,
    ) -> None:
        self._stop_requested = True
        # A run-level stop supersedes any pending stage truncation.
        self._truncate_stage_name = None
        # ``covered_bbs_at_trigger is None`` means the run NEVER observed a
        # single covered BB before the stop (the watchdog is armed from t=0
        # by design: a run with zero coverage for the whole stall window is
        # the most degenerate stall and must also be stoppable).
        # ``had_coverage_before_trigger`` makes that distinction explicit so
        # report consumers cannot confuse it with the per-attempt
        # ``stuck_zero_coverage`` triage class, which classifies a single
        # replay's zero coverage, not a whole run's.
        self._triggered = {
            "reason": STOP_REASON,
            "level": str(level),
            "stop_cause": str(cause),
            "stage": str(stage),
            "threshold_seconds": int(self.threshold_seconds),
            "stall_seconds": round(self._stage_stall_seconds, 3),
            "run_stall_seconds": round(self._run_stall_seconds, 3),
            "seconds_since_last_progress": round(now - self._last_progress_wall, 3),
            "zero_yield_units": int(self._zero_yield_units),
            "consecutive_zero_productivity_stages": (
                [dict(item) for item in self._consecutive_zero_stages]
                if level == "stage"
                else []
            ),
            "stage_truncation_count": len(self._stage_truncations),
            "covered_bbs_at_trigger": (
                None if self._max_covered_bbs is None else int(self._max_covered_bbs)
            ),
            "had_coverage_before_trigger": self._max_covered_bbs is not None,
            "evidence_ledgers_observed": bool(self._evidence_ledgers_seen),
            "ledgers": self._ledger_snapshot(),
            "wallclock_time": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(now),
            ),
        }

    def should_stop(self) -> bool:
        """Run-level stop request (stage truncations never set this)."""
        with self._lock:
            return bool(self._stop_requested)

    def should_truncate_stage(self, stage_name: str) -> bool:
        """True while ``stage_name`` (or an ancestor of it) is truncated."""
        with self._lock:
            if self._truncate_stage_name is None:
                return False
            if stage_name == self._truncate_stage_name:
                return True
            return any(
                frame.name == self._truncate_stage_name
                for frame in self._stage_stack
            )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def stop_report(self) -> Optional[dict[str, object]]:
        with self._lock:
            return dict(self._triggered) if self._triggered else None

    def stage_truncations(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(item) for item in self._stage_truncations]

    def take_unreported_stage_truncations(self) -> list[dict[str, object]]:
        """Drain truncation records not yet written as progress events."""
        with self._lock:
            if not self._unreported_truncations:
                return []
            pending = self._stage_truncations[-self._unreported_truncations:]
            self._unreported_truncations = 0
            return [dict(item) for item in pending]

    def status(self) -> dict[str, object]:
        with self._lock:
            now = self._now()
            return {
                "schema": STATUS_SCHEMA,
                "enabled": bool(self.enabled),
                "threshold_seconds": int(self.threshold_seconds),
                "min_zero_yield_units": int(self.min_zero_yield_units),
                "exempt_stage_tokens": list(self.exempt_stage_tokens),
                "max_consecutive_zero_stages": int(self.max_consecutive_zero_stages),
                "min_zero_productivity_stage_seconds": int(
                    self.min_zero_productivity_stage_seconds
                ),
                "run_stall_threshold_seconds": int(self.run_stall_threshold_seconds),
                "stall_seconds_accrued": round(self._stage_stall_seconds, 3),
                "stage_stall_seconds": round(self._stage_stall_seconds, 3),
                "run_stall_seconds": round(self._run_stall_seconds, 3),
                "seconds_since_last_progress": round(now - self._last_progress_wall, 3),
                "covered_bbs_max": (
                    None if self._max_covered_bbs is None else int(self._max_covered_bbs)
                ),
                "ledger_watermarks": dict(self._ledger_max),
                "evidence_ledgers_observed": bool(self._evidence_ledgers_seen),
                "zero_yield_units": int(self._zero_yield_units),
                "current_stage": self._depth1_stage_name(),
                "last_stage": str(self._last_stage),
                "observations": int(self._observations),
                "stage_truncation_count": len(self._stage_truncations),
                "consecutive_zero_productivity_stage_count": len(
                    self._consecutive_zero_stages
                ),
                "stop_requested": bool(self._stop_requested),
                "stop_reason": STOP_REASON if self._stop_requested else None,
                "truncate_stage": self._truncate_stage_name,
                "triggered": dict(self._triggered) if self._triggered else None,
            }


class WinddownBudgetTracker:
    """Shared, strictly capped budget for post-watchdog wind-down stages.

    Once the run-level watchdog fires, exploration and budget-consuming stages
    must see an exhausted wallclock (that is what ends the stalled stage and
    skips ``deadline_drain``).  The bounded evidence/obligation stages
    (``path_naturalization``, ``vector_only_cleanup``,
    ``path_naturalization_final``) instead draw from this allowance so a
    stalled run still emits the same obligation and vector-cleanup artifacts
    as a full-budget run.

    Stage-level truncations never mint an allowance: the run keeps its real
    remaining budget and the pipeline simply moves on to the next stage.

    The allowance is minted once, at the first query after the trigger, and is
    shared by every wind-down stage, so the total post-trigger wallclock is
    bounded by ``max_seconds`` regardless of how many stages draw from it (and
    by the true remaining budget when that is smaller).  ``max_seconds == 0``
    restores the pure hard stop.

    A run can also end by exhausting its wallclock deadline rather than by the
    watchdog firing.  That case is a trigger too (``wallclock_exhausted``):
    otherwise the exploration stages consume the whole budget and the evidence
    wind-down (``path_naturalization`` and friends) is skipped with a zero
    remaining budget, so a full-length run produces strictly fewer validation
    artifacts than a run that stopped early.  When the trigger is the expired
    deadline the allowance is deliberately *not* capped by the (zero) true
    remaining wallclock: the bounded overrun is the point, and it is still
    bounded by ``max_seconds``.
    """

    def __init__(
        self,
        *,
        max_seconds: int = DEFAULT_WINDDOWN_MAX_SECONDS,
        now: Optional[Callable[[], float]] = None,
    ):
        self.max_seconds = max(0, int(max_seconds))
        self._now = now or time.time
        self._deadline: Optional[float] = None

    def remaining(
        self,
        *,
        watchdog_stopped: bool,
        true_remaining: Optional[int],
        wallclock_exhausted: bool = False,
    ) -> Optional[int]:
        """Wind-down remaining seconds; passthrough while no trigger fired.

        ``true_remaining`` follows the campaign convention: ``None`` means no
        wallclock deadline was configured (unbounded run).  ``wallclock_exhausted``
        is the second trigger: the run's own deadline has passed, so the bounded
        allowance is granted instead of the (now zero) remaining budget.
        """
        if not (watchdog_stopped or wallclock_exhausted):
            return true_remaining
        if self._deadline is None:
            self._deadline = self._now() + self.max_seconds
        now = self._now()
        capped = max(0, int(self._deadline - now))
        # An expired deadline must not cap the allowance away: the allowance is
        # exactly the bounded overrun the wind-down stages are meant to spend.
        if true_remaining is not None and not wallclock_exhausted:
            capped = min(capped, max(0, int(true_remaining)))
        return capped

    def status(self) -> dict[str, object]:
        now = self._now()
        return {
            "max_seconds": int(self.max_seconds),
            "minted": self._deadline is not None,
            "remaining_seconds": (
                None
                if self._deadline is None
                else max(0, int(self._deadline - now))
            ),
        }
