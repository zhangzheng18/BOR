#!/usr/bin/env python3
"""Contract tests for the stage-level multi-ledger stall watchdog (round 3).

The round-1/2 run-level wallclock stop was overturned by measured data from
``campaign_24h_ardupilot_pixhawk1_NOLLM_control_20260918``: the
``frontier_successor_replay_tail`` stage held ``covered_bbs`` flat at 1336 for
9.68h while ``counterfactual_only_bbs`` grew +346, and the +346 only entered
``covered_bbs`` at the stage-boundary merge.  These tests pin the new
contract: multi-ledger productivity, stage-level truncation that exits
through the ordinary merge path, and run-level stops only as a fallback.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lsgemu.path_naturalization import EVIDENCE_E2
from lsgemu.scheduler.contracts import SchedulerRuntime
from lsgemu.scheduler.outcome_feedback import OutcomeFeedback
from lsgemu.scheduler.replay_executor import ReplayExecutor
from lsgemu.scheduler.stall_watchdog import (
    DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES,
    DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS,
    DEFAULT_MIN_ZERO_YIELD_UNITS,
    DEFAULT_RUN_STALL_THRESHOLD_SECONDS,
    DEFAULT_STALL_THRESHOLD_SECONDS,
    DEFAULT_WINDDOWN_MAX_SECONDS,
    STAGE_TRUNCATION_REASON,
    STOP_REASON,
    StallWatchdog,
    WinddownBudgetTracker,
)
from lsgemu.cached_interleaved_runner import CoverageProgressMonitor

TAIL = "frontier_successor_replay_tail"


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _stage_stopped_watchdog() -> StallWatchdog:
    """A watchdog stopped at level="stage" via the K-streak fallback (K=1)."""
    clock = _FakeClock()
    watchdog = StallWatchdog(
        threshold_seconds=10,
        min_zero_yield_units=2,
        max_consecutive_zero_stages=1,
        min_zero_productivity_stage_seconds=5,
        run_stall_threshold_seconds=0,
        started_at=clock(),
        now=clock,
    )
    # Baseline telemetry (run_start/stage events) initializes the ledger
    # watermarks before any stage bracket opens, exactly like the monitor's
    # setup snapshots in a real run.
    watchdog.observe(stage="setup", covered_bbs=100)
    watchdog.notify_stage_begin(TAIL)
    for _ in range(3):
        clock.advance(10)
        watchdog.observe(stage=TAIL, covered_bbs=100, zero_yield_unit=True)
    assert not watchdog.should_stop()
    watchdog.notify_stage_end(TAIL, wall_seconds=60.0)
    assert watchdog.should_stop()
    return watchdog


def _run_floor_stopped_watchdog() -> StallWatchdog:
    """A watchdog stopped at level="run" via the 8h-style safety floor."""
    clock = _FakeClock()
    watchdog = StallWatchdog(
        threshold_seconds=10,
        min_zero_yield_units=2,
        run_stall_threshold_seconds=30,
        started_at=clock(),
        now=clock,
    )
    watchdog.observe(stage="setup", covered_bbs=100)
    watchdog.notify_stage_begin(TAIL)
    for _ in range(3):
        clock.advance(15)
        watchdog.observe(stage=TAIL, covered_bbs=100, zero_yield_unit=True)
    assert watchdog.should_stop()
    return watchdog


class StallWatchdogTests(unittest.TestCase):
    def test_stage_truncation_after_threshold_with_dual_signal(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        # The run reaches the plateau inside exempt bookkeeping, then the
        # campaign shape: flat ledgers in the tail stage, one interval
        # sample every 600 s.
        watchdog.observe(stage="setup", covered_bbs=1336)
        frozen = {
            "natural_supported_bbs": 866,
            "counterfactual_only_bbs": 816,
            "unclassified_evidence_bbs": 0,
            "validated_replay_bbs": 0,
            "diagnostic_replay_bbs": 0,
            "canonical_unclassified_bbs": 0,
        }
        for _ in range(10):
            clock.advance(600)
            watchdog.observe(
                stage=TAIL,
                covered_bbs=1336,
                zero_yield_unit=True,
                evidence_ledgers=frozen,
            )

        # Stage-level truncation, not a run-level stop.
        self.assertFalse(watchdog.should_stop())
        self.assertTrue(watchdog.should_truncate_stage(TAIL))
        self.assertIsNone(watchdog.stop_report())
        status = watchdog.status()
        self.assertEqual(1, status["stage_truncation_count"])
        self.assertEqual(TAIL, status["truncate_stage"])
        record = watchdog.stage_truncations()[0]
        self.assertEqual(TAIL, record["stage"])
        self.assertEqual("stage", record["level"])
        self.assertEqual(5400, record["threshold_seconds"])
        self.assertGreaterEqual(record["stage_stall_seconds"], 5400)
        self.assertGreaterEqual(record["zero_yield_units"], 8)
        self.assertTrue(record["evidence_ledgers_observed"])
        self.assertEqual(1336, record["ledgers"]["covered_bbs"])
        self.assertEqual(816, record["ledgers"]["counterfactual_only_bbs"])

    def test_50_minute_counterexample_does_not_trigger(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin("frontier_successor_replay")
        watchdog.observe(stage="frontier_successor_replay", covered_bbs=1002)
        # 50.3 minutes of zero growth observed at 600 s intervals...
        for _ in range(5):
            clock.advance(600)
            watchdog.observe(
                stage="frontier_successor_replay",
                covered_bbs=1002,
                zero_yield_unit=True,
            )
        # ...then the run finds +58 new BBs and keeps working afterwards.
        clock.advance(18)
        watchdog.observe(
            stage="frontier_successor_replay",
            covered_bbs=1060,
            zero_yield_unit=True,
        )
        for _ in range(3):
            clock.advance(600)
            watchdog.observe(
                stage="frontier_successor_replay",
                covered_bbs=1060,
                zero_yield_unit=True,
            )

        self.assertFalse(watchdog.should_stop())
        self.assertFalse(watchdog.should_truncate_stage("frontier_successor_replay"))
        status = watchdog.status()
        # Progress reset the stall clock; only the post-progress flat stretch
        # remains and it is far below the 90-minute threshold.
        self.assertLess(status["stall_seconds_accrued"], 5400)
        self.assertEqual(1060, status["covered_bbs_max"])

    # ------------------------------------------------------------------ #
    # Requirement round-3 test 1: the measured counterexample curve.
    # covered_bbs frozen at 1336 for hours while counterfactual_only_bbs
    # grows must NOT trigger -- the stage was still earning cf evidence.
    # ------------------------------------------------------------------ #

    def test_real_counterexample_curve_cf_growth_prevents_truncation(self):
        # (elapsed_hours, counterfactual_only_bbs) straight from
        # coverage_progress.jsonl of the 24h no-LLM control arm; the covered
        # ledger (interval view) stays 1336 for the whole window and the
        # per-attempt phase-union view grows with the earned cf evidence.
        curve = [
            (3.50, 470),
            (3.67, 563),
            (3.83, 646),
            (4.00, 698),
            (4.17, 753),
            (5.33, 816),
        ]
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        # The first replayed sample is t0 for the fake clock (coverage had
        # been growing from 0 to 1336 for the preceding 3.5h of the run).
        # Between the measured stamps the hidden 600s samples grow cf
        # linearly, so the last real progress lands exactly at t=5.33h like
        # in the measurement.
        truncated_at = None
        first_hours, first_cf = curve[0]
        clock.advance(0.0)
        self._observe_control_arm_sample(
            watchdog,
            covered_interval=1336,
            covered_union=1336 + (first_cf - 470),
            cf_only=first_cf,
        )
        for (t0, cf0), (t1, cf1) in zip(curve, curve[1:]):
            segment_steps = list(self._interval_steps(t0, t1))
            total_hours = t1 - t0
            elapsed_in_segment = 0.0
            for step_hours in segment_steps:
                elapsed_in_segment += step_hours
                cf_only = cf0 + (cf1 - cf0) * (elapsed_in_segment / total_hours)
                clock.advance(step_hours * 3600.0)
                self._observe_control_arm_sample(
                    watchdog,
                    covered_interval=1336,
                    covered_union=1336 + (cf_only - 470),
                    cf_only=cf_only,
                )
                if watchdog.should_truncate_stage(TAIL):
                    truncated_at = (t1, cf1)
                    break
            if truncated_at:
                break

        # Through the whole productive window (cf evidence still growing,
        # 9.68h of frozen interval covered) there is no truncation at all.
        self.assertIsNone(truncated_at)
        self.assertFalse(watchdog.should_stop())
        self.assertEqual(0, watchdog.status()["stage_truncation_count"])

        # After the last real progress (t=5.33h) every ledger freezes; the
        # stage truncates only once the full 90-minute window has elapsed.
        frozen_at_hours = 5.33
        checked_hours = frozen_at_hours
        fired_hours = None
        while checked_hours < 13.18:
            checked_hours += 600.0 / 3600.0
            clock.advance(600.0)
            self._observe_control_arm_sample(
                watchdog,
                covered_interval=1336,
                covered_union=1336 + (816 - 470),
                cf_only=816,
            )
            if watchdog.should_truncate_stage(TAIL):
                fired_hours = checked_hours
                break
        self.assertIsNotNone(fired_hours)
        self.assertGreaterEqual(fired_hours, frozen_at_hours + 5400.0 / 3600.0 - 1e-6)
        # It fired long before the observed natural stage end (13.18h) and it
        # did not stop the run.
        self.assertLess(fired_hours, 13.18)
        self.assertFalse(watchdog.should_stop())
        record = watchdog.stage_truncations()[0]
        self.assertEqual(TAIL, record["stage"])
        self.assertGreaterEqual(record["stage_stall_seconds"], 5400.0)

    @staticmethod
    def _interval_steps(from_hours: float, to_hours: float):
        """600s interval steps between two elapsed-hour stamps."""
        remaining = (to_hours - from_hours) * 3600.0
        while remaining > 1e-9:
            step = min(600.0, remaining)
            yield step / 3600.0
            remaining -= step

    @staticmethod
    def _observe_control_arm_sample(
        watchdog: StallWatchdog,
        *,
        covered_interval: int,
        covered_union: int,
        cf_only: int,
    ) -> None:
        # One monitor interval sample (interval covered view + evidence
        # ledgers) followed by a burst of per-attempt kernel observations
        # (phase-union covered view), mirroring the real feed mix.
        watchdog.observe(
            stage=TAIL,
            covered_bbs=covered_interval,
            zero_yield_unit=True,
            evidence_ledgers={
                "natural_supported_bbs": 866,
                "counterfactual_only_bbs": cf_only,
                "unclassified_evidence_bbs": 0,
                "validated_replay_bbs": 0,
                "diagnostic_replay_bbs": 0,
                "canonical_unclassified_bbs": 0,
            },
        )
        for _ in range(3):
            watchdog.observe(
                stage=TAIL,
                covered_bbs=covered_union,
                zero_yield_unit=True,
            )

    def test_any_evidence_ledger_growth_resets_clocks(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=1800,
            min_zero_yield_units=2,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        ledgers = {
            "natural_supported_bbs": 866,
            "counterfactual_only_bbs": 470,
            "validated_replay_bbs": 0,
            "diagnostic_replay_bbs": 0,
            "canonical_unclassified_bbs": 0,
        }
        watchdog.observe(stage=TAIL, covered_bbs=1336)
        # 80 minutes of frozen covered ledgers...
        for _ in range(8):
            clock.advance(600)
            watchdog.observe(stage=TAIL, covered_bbs=1336, zero_yield_unit=True)
        self.assertGreaterEqual(watchdog.status()["stall_seconds_accrued"], 4800)
        # ...then a diagnostic ledger moves: full reset even though the
        # covered ledger never moved.
        clock.advance(600)
        ledgers["diagnostic_replay_bbs"] = 12
        watchdog.observe(
            stage=TAIL,
            covered_bbs=1336,
            zero_yield_unit=True,
            evidence_ledgers=ledgers,
        )
        status = watchdog.status()
        self.assertEqual(0.0, status["stall_seconds_accrued"])
        self.assertEqual(0, status["zero_yield_units"])
        self.assertEqual(12, status["ledger_watermarks"]["diagnostic_replay_bbs"])

    # ------------------------------------------------------------------ #
    # Requirement round-3 test 2: all-ledger freeze truncates the stage and
    # the run continues.
    # ------------------------------------------------------------------ #

    def test_all_ledger_freeze_truncates_stage_and_run_continues(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=1800,
            min_zero_yield_units=2,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        frozen = {
            "natural_supported_bbs": 866,
            "counterfactual_only_bbs": 816,
        }
        watchdog.notify_stage_begin(TAIL)
        watchdog.observe(stage=TAIL, covered_bbs=1682, evidence_ledgers=frozen)
        for _ in range(4):
            clock.advance(600)
            watchdog.observe(
                stage=TAIL,
                covered_bbs=1682,
                zero_yield_unit=True,
                evidence_ledgers=frozen,
            )
        self.assertTrue(watchdog.should_truncate_stage(TAIL))
        self.assertFalse(watchdog.should_stop())

        # The stage ends (ordinary exit); the pipeline moves on.  The exempt
        # wind-down stage never accrues, and the next coverage-expected stage
        # starts a fresh stage clock -- the run is still alive.
        watchdog.notify_stage_end(TAIL, wall_seconds=7300.0)
        self.assertFalse(watchdog.should_truncate_stage(TAIL))
        self.assertFalse(watchdog.should_stop())

        watchdog.notify_stage_begin("vector_only_cleanup")
        for _ in range(3):
            clock.advance(600)
            watchdog.observe(stage="vector_only_cleanup", covered_bbs=1682, zero_yield_unit=True)
        self.assertEqual(0.0, watchdog.status()["stall_seconds_accrued"])
        watchdog.notify_stage_end("vector_only_cleanup", wall_seconds=240.0)

        watchdog.notify_stage_begin("deadline_drain")
        watchdog.observe(stage="deadline_drain_frontier_targeted_2", covered_bbs=1682)
        clock.advance(600)
        watchdog.observe(
            stage="deadline_drain_frontier_targeted_2",
            covered_bbs=1682,
            zero_yield_unit=True,
        )
        status = watchdog.status()
        self.assertEqual("deadline_drain", status["current_stage"])
        self.assertEqual(600.0, status["stall_seconds_accrued"])
        self.assertFalse(watchdog.should_stop())

    def test_truncation_flag_matches_enclosing_budget_owner_stage(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=1682)
        watchdog.notify_stage_begin("deadline_drain")
        watchdog.notify_stage_begin("deadline_drain_frontier_targeted_2")
        clock.advance(700)
        watchdog.observe(
            stage="deadline_drain_frontier_targeted_2",
            covered_bbs=1682,
            zero_yield_unit=True,
        )
        self.assertTrue(watchdog.should_truncate_stage("deadline_drain"))
        # Sub-stage names rotate every drain round; the ancestor match lets
        # both the drain loop and the sub-stage kernels see the flag.
        self.assertTrue(
            watchdog.should_truncate_stage("deadline_drain_frontier_successor_3")
        )
        # Ending the sub-stage does not clear the drain-wide flag...
        watchdog.notify_stage_end("deadline_drain_frontier_targeted_2", wall_seconds=700.0)
        self.assertTrue(watchdog.should_truncate_stage("deadline_drain"))
        # ...only ending the budget owner does.
        watchdog.notify_stage_end("deadline_drain", wall_seconds=7300.0)
        self.assertFalse(watchdog.should_truncate_stage("deadline_drain"))

    def test_deadline_drain_is_no_longer_exempt(self):
        # Round 2 classified deadline_drain as exempt "expected to spend the
        # remaining budget"; the control arm measured 4.31h of frozen ledgers
        # there, so round 3 monitors and truncates it.
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=1682)
        watchdog.notify_stage_begin("deadline_drain")
        clock.advance(700)
        watchdog.observe(
            stage="deadline_drain",
            covered_bbs=1682,
            zero_yield_unit=True,
        )
        self.assertTrue(watchdog.should_truncate_stage("deadline_drain"))
        self.assertFalse(watchdog.stage_is_exempt("deadline_drain_targeted"))
        self.assertFalse(watchdog.stage_is_exempt("deadline_drain_switch_frontier_targeted_3"))

    def test_exempt_stages_do_not_accrue_stall(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        for stage in (
            "vector_only_cleanup",
            "path_naturalization",
            "path_naturalization_final",
            "checkpoint",
        ):
            watchdog.notify_stage_begin(stage)
            watchdog.observe(stage=stage, covered_bbs=None)
            for _ in range(4):
                clock.advance(900)
                watchdog.observe(stage=stage, covered_bbs=42, zero_yield_unit=True)
            # The real executor emits its stage_end snapshot before the
            # bracket closes, so the inter-stage gap is consumed by exempt
            # telemetry rather than the next stage.
            clock.advance(600)
            watchdog.observe(stage=stage, covered_bbs=42)
            watchdog.notify_stage_end(stage, wall_seconds=3600.0)

        # 4 x 3600 s of legitimate non-coverage work, still no trigger and
        # no run stop (exempt stages never join the zero-productivity streak
        # either, even at 3600s >= the 1800s eligibility floor).
        self.assertFalse(watchdog.should_stop())
        self.assertFalse(watchdog.should_truncate_stage(TAIL))
        self.assertEqual(0.0, watchdog.status()["stall_seconds_accrued"])
        self.assertEqual(0, watchdog.status()["consecutive_zero_productivity_stage_count"])
        # Time spent inside exempt stages never counts: entering a
        # coverage-expected stage afterwards still starts from zero stall.
        watchdog.notify_stage_begin(TAIL)
        watchdog.observe(
            stage=TAIL,
            covered_bbs=42,
            zero_yield_unit=True,
        )
        self.assertEqual(0.0, watchdog.status()["stall_seconds_accrued"])

    def test_secondary_signal_is_required(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        watchdog.observe(stage=TAIL, covered_bbs=10)
        # Wall-clock-only observations: no completed production units at all
        # (e.g. a single hung attempt), so the secondary signal never holds.
        for _ in range(12):
            clock.advance(600)
            watchdog.observe(stage=TAIL, covered_bbs=10)

        self.assertGreaterEqual(watchdog.status()["stall_seconds_accrued"], 5400)
        self.assertFalse(watchdog.should_stop())
        self.assertFalse(watchdog.should_truncate_stage(TAIL))

    def test_disabled_watchdog_never_stops(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            enabled=False,
            threshold_seconds=10,
            min_zero_yield_units=1,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        for _ in range(4):
            clock.advance(30)
            watchdog.observe(
                stage=TAIL,
                covered_bbs=1,
                zero_yield_unit=True,
            )
        watchdog.notify_stage_end(TAIL, wall_seconds=120.0)
        self.assertFalse(watchdog.should_stop())
        self.assertFalse(watchdog.should_truncate_stage(TAIL))
        self.assertIsNone(watchdog.stop_report())

    # ------------------------------------------------------------------ #
    # Requirement round-3 test 3: K consecutive zero-productivity stages
    # stop the whole run.
    # ------------------------------------------------------------------ #

    def test_consecutive_zero_productivity_stages_stop_run(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            max_consecutive_zero_stages=3,
            min_zero_productivity_stage_seconds=100,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=1682)
        for index, stage in enumerate((
            "late_direct_call_frontier_drain",
            TAIL,
            "deadline_drain",
        )):
            self.assertFalse(watchdog.should_stop())
            watchdog.notify_stage_begin(stage)
            clock.advance(120 + index)
            watchdog.observe(stage=stage, covered_bbs=1682, zero_yield_unit=True)
            watchdog.notify_stage_end(stage, wall_seconds=120.0 + index)

        self.assertTrue(watchdog.should_stop())
        report = watchdog.stop_report()
        self.assertEqual(STOP_REASON, report["reason"])
        self.assertEqual("stage", report["level"])
        self.assertEqual("consecutive_zero_productivity_stages", report["stop_cause"])
        self.assertEqual(3, len(report["consecutive_zero_productivity_stages"]))
        self.assertEqual(
            ["late_direct_call_frontier_drain", TAIL, "deadline_drain"],
            [item["stage"] for item in report["consecutive_zero_productivity_stages"]],
        )
        # A run-level stop supersedes any pending stage truncation flag.
        self.assertFalse(watchdog.should_truncate_stage(TAIL))

    def test_productive_stage_resets_zero_streak(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            max_consecutive_zero_stages=3,
            min_zero_productivity_stage_seconds=100,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=10)
        # Two zero stages...
        for stage in ("zero_stage_a", "zero_stage_b"):
            watchdog.notify_stage_begin(stage)
            clock.advance(150)
            watchdog.observe(stage=stage, covered_bbs=10, zero_yield_unit=True)
            watchdog.notify_stage_end(stage, wall_seconds=150.0)
        self.assertEqual(2, watchdog.status()["consecutive_zero_productivity_stage_count"])
        # ...then a productive one (any ledger move), then another zero: the
        # streak restarted at 1, so no run stop.
        watchdog.notify_stage_begin("productive_stage")
        clock.advance(150)
        watchdog.observe(stage="productive_stage", covered_bbs=58)
        watchdog.notify_stage_end("productive_stage", wall_seconds=150.0)
        watchdog.notify_stage_begin("zero_stage_c")
        clock.advance(150)
        watchdog.observe(stage="zero_stage_c", covered_bbs=58, zero_yield_unit=True)
        watchdog.notify_stage_end("zero_stage_c", wall_seconds=150.0)
        self.assertFalse(watchdog.should_stop())
        self.assertEqual(1, watchdog.status()["consecutive_zero_productivity_stage_count"])

    def test_short_skipped_and_failed_stages_are_streak_neutral(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            max_consecutive_zero_stages=3,
            min_zero_productivity_stage_seconds=1800,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=10)
        # A real zero stage long enough to count.
        watchdog.notify_stage_begin("long_zero")
        clock.advance(2000)
        watchdog.observe(stage="long_zero", covered_bbs=10, zero_yield_unit=True)
        watchdog.notify_stage_end("long_zero", wall_seconds=2000.0)
        self.assertEqual(1, watchdog.status()["consecutive_zero_productivity_stage_count"])

        # Skipped stages (no budget) and short zero stages never join the
        # streak -- otherwise a post-stop no-budget slide would masquerade as
        # zero productivity and mislabel a budget-exhausted run as watchdog
        # stopped.
        watchdog.notify_stage_begin("skipped_stage")
        watchdog.notify_stage_end("skipped_stage", wall_seconds=0.0, skipped=True)
        watchdog.notify_stage_begin("short_zero")
        clock.advance(60)
        watchdog.observe(stage="short_zero", covered_bbs=10, zero_yield_unit=True)
        watchdog.notify_stage_end("short_zero", wall_seconds=60.0)
        watchdog.notify_stage_begin("failed_stage")
        clock.advance(2000)
        watchdog.observe(stage="failed_stage", covered_bbs=10, zero_yield_unit=True)
        watchdog.notify_stage_end("failed_stage", wall_seconds=2000.0, failed=True)
        self.assertEqual(1, watchdog.status()["consecutive_zero_productivity_stage_count"])
        self.assertFalse(watchdog.should_stop())

    def test_run_stall_floor_stops_at_level_run(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            max_consecutive_zero_stages=99,
            run_stall_threshold_seconds=30,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        for _ in range(3):
            clock.advance(15)
            watchdog.observe(stage=TAIL, covered_bbs=1336, zero_yield_unit=True)
        self.assertTrue(watchdog.should_stop())
        report = watchdog.stop_report()
        self.assertEqual(STOP_REASON, report["reason"])
        self.assertEqual("run", report["level"])
        self.assertEqual("run_stall_floor", report["stop_cause"])
        self.assertGreaterEqual(report["run_stall_seconds"], 30)

    def test_run_floor_default_is_eight_hours(self):
        # The floor must sit above the largest measured legitimate dead
        # window (7.85h, control arm t=5.33h -> 13.18h).
        self.assertEqual(28800, DEFAULT_RUN_STALL_THRESHOLD_SECONDS)
        self.assertEqual(3, DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES)
        self.assertEqual(1800, DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS)
        self.assertEqual(5400, DEFAULT_STALL_THRESHOLD_SECONDS)

    def test_zero_coverage_truncation_marks_never_covered_semantics(self):
        # Armed from t=0: a run that produced no covered BB at all still
        # truncates the stage, and a run-level stop report must distinguish
        # "never had coverage" from "had coverage, then stalled".
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=10,
            min_zero_yield_units=2,
            max_consecutive_zero_stages=1,
            min_zero_productivity_stage_seconds=5,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.notify_stage_begin(TAIL)
        for _ in range(3):
            clock.advance(10)
            watchdog.observe(
                stage=TAIL,
                covered_bbs=None,
                zero_yield_unit=True,
            )
        self.assertTrue(watchdog.should_truncate_stage(TAIL))
        watchdog.notify_stage_end(TAIL, wall_seconds=40.0)
        self.assertTrue(watchdog.should_stop())
        report = watchdog.stop_report()
        self.assertIsNone(report["covered_bbs_at_trigger"])
        self.assertFalse(report["had_coverage_before_trigger"])
        self.assertIsNone(watchdog.status()["covered_bbs_max"])

    def test_trigger_after_coverage_marks_had_coverage(self):
        report = _stage_stopped_watchdog().stop_report()
        self.assertEqual(100, report["covered_bbs_at_trigger"])
        self.assertTrue(report["had_coverage_before_trigger"])

    def test_from_env_cli_args_win_over_env(self):
        args = SimpleNamespace(
            stall_watchdog_threshold_seconds=1800,
            stall_watchdog_min_zero_yield_units=3,
            stall_watchdog_exempt_stages="custom_stage",
            stall_watchdog_max_consecutive_zero_stages=5,
            stall_watchdog_min_zero_productivity_stage_seconds=900,
            stall_watchdog_run_stall_seconds=36000,
            disable_stall_watchdog=False,
        )
        env = {
            "LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS": "60",
            "LSGEMU_STALL_WATCHDOG_MIN_ZERO_YIELD_UNITS": "99",
            "LSGEMU_STALL_WATCHDOG_EXEMPT_STAGES": "env_only_stage",
            "LSGEMU_STALL_WATCHDOG_MAX_CONSECUTIVE_ZERO_STAGES": "7",
            "LSGEMU_STALL_WATCHDOG_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS": "60",
            "LSGEMU_STALL_WATCHDOG_RUN_STALL_SECONDS": "72000",
        }
        with patch.dict(os.environ, env, clear=False):
            watchdog = StallWatchdog.from_env(args)
        self.assertEqual(1800, watchdog.threshold_seconds)
        self.assertEqual(3, watchdog.min_zero_yield_units)
        self.assertIn("custom_stage", watchdog.exempt_stage_tokens)
        # A non-empty CLI exempt list wins for that key, so the env value is
        # not merged in; the built-in exemptions always remain.
        self.assertNotIn("env_only_stage", watchdog.exempt_stage_tokens)
        self.assertIn("path_naturalization", watchdog.exempt_stage_tokens)
        self.assertTrue(watchdog.stage_is_exempt("custom_stage_round_2"))
        self.assertEqual(5, watchdog.max_consecutive_zero_stages)
        self.assertEqual(900, watchdog.min_zero_productivity_stage_seconds)
        self.assertEqual(36000, watchdog.run_stall_threshold_seconds)

    def test_from_env_defaults_and_switches(self):
        clean_env = {
            key: ""
            for key in (
                "LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS",
                "LSGEMU_STALL_WATCHDOG_MIN_ZERO_YIELD_UNITS",
                "LSGEMU_STALL_WATCHDOG_ENABLED",
                "LSGEMU_STALL_WATCHDOG_EXEMPT_STAGES",
                "LSGEMU_STALL_WATCHDOG_MAX_CONSECUTIVE_ZERO_STAGES",
                "LSGEMU_STALL_WATCHDOG_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS",
                "LSGEMU_STALL_WATCHDOG_RUN_STALL_SECONDS",
            )
        }
        args = SimpleNamespace(
            stall_watchdog_threshold_seconds=None,
            stall_watchdog_min_zero_yield_units=None,
            stall_watchdog_exempt_stages="",
            stall_watchdog_max_consecutive_zero_stages=None,
            stall_watchdog_min_zero_productivity_stage_seconds=None,
            stall_watchdog_run_stall_seconds=None,
            disable_stall_watchdog=False,
        )
        with patch.dict(os.environ, clean_env, clear=False):
            watchdog = StallWatchdog.from_env(args)
            self.assertEqual(DEFAULT_STALL_THRESHOLD_SECONDS, watchdog.threshold_seconds)
            self.assertEqual(DEFAULT_MIN_ZERO_YIELD_UNITS, watchdog.min_zero_yield_units)
            self.assertEqual(
                DEFAULT_MAX_CONSECUTIVE_ZERO_STAGES,
                watchdog.max_consecutive_zero_stages,
            )
            self.assertEqual(
                DEFAULT_MIN_ZERO_PRODUCTIVITY_STAGE_SECONDS,
                watchdog.min_zero_productivity_stage_seconds,
            )
            self.assertEqual(
                DEFAULT_RUN_STALL_THRESHOLD_SECONDS,
                watchdog.run_stall_threshold_seconds,
            )
            self.assertTrue(watchdog.enabled)

            disabled = StallWatchdog.from_env(
                SimpleNamespace(
                    stall_watchdog_threshold_seconds=None,
                    stall_watchdog_min_zero_yield_units=None,
                    stall_watchdog_exempt_stages="",
                    stall_watchdog_max_consecutive_zero_stages=None,
                    stall_watchdog_min_zero_productivity_stage_seconds=None,
                    stall_watchdog_run_stall_seconds=None,
                    disable_stall_watchdog=True,
                )
            )
            self.assertFalse(disabled.enabled)

        with patch.dict(
            os.environ,
            {**clean_env, "LSGEMU_STALL_WATCHDOG_ENABLED": "0"},
            clear=False,
        ):
            self.assertFalse(StallWatchdog.from_env(args).enabled)
        with patch.dict(
            os.environ,
            {**clean_env, "LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS": "0"},
            clear=False,
        ):
            # A non-positive threshold is an explicit "off".
            self.assertFalse(StallWatchdog.from_env(args).enabled)
        with patch.dict(
            os.environ,
            {**clean_env, "LSGEMU_STALL_WATCHDOG_THRESHOLD_SECONDS": "120"},
            clear=False,
        ):
            self.assertEqual(120, StallWatchdog.from_env(args).threshold_seconds)
        with patch.dict(
            os.environ,
            {**clean_env, "LSGEMU_STALL_WATCHDOG_RUN_STALL_SECONDS": "0"},
            clear=False,
        ):
            # An explicit zero floor (CLI or env) disables the run-level
            # backstop and keeps the pure stage-level watchdog.
            self.assertEqual(0, StallWatchdog.from_env(args).run_stall_threshold_seconds)
        floor_off = StallWatchdog.from_env(
            SimpleNamespace(
                stall_watchdog_threshold_seconds=None,
                stall_watchdog_min_zero_yield_units=None,
                stall_watchdog_exempt_stages="",
                stall_watchdog_max_consecutive_zero_stages=None,
                stall_watchdog_min_zero_productivity_stage_seconds=None,
                stall_watchdog_run_stall_seconds=0,
                disable_stall_watchdog=False,
            )
        )
        self.assertEqual(0, floor_off.run_stall_threshold_seconds)
        self.assertTrue(floor_off.enabled)


class _MonitorRunner:
    class _Prepared:
        firmware_sha256 = "firmware-sha"
        toolchain_fingerprint = {"runtime_fingerprint": "runtime-sha"}
        static_cache_identity = {"identity_hash": "cache-sha"}
        valid_bb_set = {0x08000000, 0x08000004}

    prepared = _Prepared()


class StallWatchdogMonitorIntegrationTests(unittest.TestCase):
    def _monitor(self, watchdog):
        return CoverageProgressMonitor(
            runner=_MonitorRunner(),
            firmware=Path("firmware.elf"),
            output_path=Path("progress.jsonl"),
            interval_seconds=0,
            started_at=time.time(),
            stall_watchdog=watchdog,
        )

    def test_interval_payload_carries_watchdog_state(self):
        watchdog = StallWatchdog(threshold_seconds=5400, min_zero_yield_units=8)
        monitor = self._monitor(watchdog)
        baseline = {
            "stage": TAIL,
            "covered_bbs": 1336,
            "evidence_status": {
                "natural_supported_bbs": 866,
                "counterfactual_only_bbs": 816,
            },
        }
        monitor._observe_stall_watchdog(
            "interval",
            baseline,
            zero_yield_unit=False,
        )
        payload = {
            "stage": TAIL,
            "covered_bbs": 1336,
            "evidence_status": {
                "natural_supported_bbs": 866,
                "counterfactual_only_bbs": 816,
            },
        }
        monitor._observe_stall_watchdog(
            "interval",
            payload,
            zero_yield_unit=True,
        )
        self.assertIn("stall_watchdog_stall_seconds", payload)
        self.assertFalse(payload["stall_watchdog_stop_requested"])
        self.assertEqual(1, watchdog.status()["zero_yield_units"])
        # The monitor-computed evidence_status feeds the ledger watermarks.
        self.assertEqual(
            816,
            watchdog.status()["ledger_watermarks"]["counterfactual_only_bbs"],
        )
        self.assertEqual(
            866,
            watchdog.status()["ledger_watermarks"]["natural_supported_bbs"],
        )

    def test_dedicated_event_emitted_once_and_final_stop_reason(self):
        watchdog = _stage_stopped_watchdog()
        monitor = self._monitor(watchdog)
        captured = []

        def capture_snapshot(event, *, stage=None, extra=None):
            captured.append(("snapshot", event, dict(extra or {})))
            return {"event": event, "stage": stage}

        def capture_unchanged(event, *, stage=None, extra=None):
            captured.append(("unchanged", event, dict(extra or {})))
            return {"event": event, "stage": stage}

        monitor.snapshot = capture_snapshot
        monitor.snapshot_unchanged = capture_unchanged
        monitor._maybe_emit_stall_watchdog_event()
        monitor._maybe_emit_stall_watchdog_event()
        monitor.stop("run_complete")

        dedicated = [item for item in captured if item[1] == "stall_watchdog_triggered"]
        self.assertEqual(1, len(dedicated))
        self.assertEqual(STOP_REASON, dedicated[0][2]["stop_reason"])
        self.assertEqual("stage", dedicated[0][2]["stall_watchdog"]["level"])
        final = captured[-1]
        self.assertEqual("run_complete", final[1])
        self.assertEqual(STOP_REASON, final[2]["stop_reason"])
        self.assertEqual(STOP_REASON, final[2]["stall_watchdog"]["reason"])

    def test_stage_truncation_events_emitted_once_per_record(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=1336)
        watchdog.notify_stage_begin(TAIL)
        clock.advance(700)
        watchdog.observe(stage=TAIL, covered_bbs=1336, zero_yield_unit=True)
        self.assertTrue(watchdog.should_truncate_stage(TAIL))
        monitor = self._monitor(watchdog)
        captured = []

        def capture_unchanged(event, *, stage=None, extra=None):
            captured.append((event, stage, dict(extra or {})))
            return {"event": event, "stage": stage}

        monitor.snapshot_unchanged = capture_unchanged
        monitor._maybe_emit_stall_watchdog_stage_events()
        monitor._maybe_emit_stall_watchdog_stage_events()
        self.assertEqual(1, len(captured))
        event, stage, extra = captured[0]
        self.assertEqual("stall_watchdog_stage_truncated", event)
        self.assertEqual(TAIL, stage)
        self.assertEqual(TAIL, extra["stall_watchdog_stage_truncation"]["stage"])
        self.assertEqual("stage", extra["stall_watchdog_stage_truncation"]["level"])


class WinddownBudgetTrackerTests(unittest.TestCase):
    def test_passthrough_while_watchdog_quiet(self):
        tracker = WinddownBudgetTracker(max_seconds=900)
        self.assertEqual(4321, tracker.remaining(watchdog_stopped=False, true_remaining=4321))
        self.assertIsNone(tracker.remaining(watchdog_stopped=False, true_remaining=None))
        self.assertFalse(tracker.status()["minted"])

    def test_allowance_capped_and_shared_after_trigger(self):
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=900, now=clock)
        first = tracker.remaining(watchdog_stopped=True, true_remaining=61_200)
        self.assertEqual(900, first)
        self.assertTrue(tracker.status()["minted"])
        # The allowance is minted once and shared: later queries see only what
        # is left of the same 900 s, not a fresh grant per stage.
        clock.advance(120)
        self.assertEqual(780, tracker.remaining(watchdog_stopped=True, true_remaining=61_080))
        clock.advance(780)
        self.assertEqual(0, tracker.remaining(watchdog_stopped=True, true_remaining=60_300))

    def test_true_remaining_dominates_when_smaller(self):
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=900, now=clock)
        self.assertEqual(
            42,
            tracker.remaining(watchdog_stopped=True, true_remaining=42),
        )
        # No deadline configured: the cap alone bounds the wind-down.
        clock2 = _FakeClock()
        unbounded = WinddownBudgetTracker(max_seconds=300, now=clock2)
        self.assertEqual(
            300,
            unbounded.remaining(watchdog_stopped=True, true_remaining=None),
        )

    def test_expired_deadline_mints_the_bounded_allowance(self):
        # r37：墙钟耗尽（true_remaining == 0）也必须铸出收尾额度，否则长跑把
        # 预算全花在探索上、path_naturalization 等阶段恒被 zero-budget 跳过。
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=900, now=clock)
        self.assertEqual(
            900,
            tracker.remaining(
                watchdog_stopped=False,
                true_remaining=0,
                wallclock_exhausted=True,
            ),
        )
        self.assertTrue(tracker.status()["minted"])
        # Still shared and bounded: the allowance is not re-minted per query.
        clock.advance(850)
        self.assertEqual(
            50,
            tracker.remaining(
                watchdog_stopped=False,
                true_remaining=0,
                wallclock_exhausted=True,
            ),
        )

    def test_no_trigger_keeps_passthrough_below_deadline(self):
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=900, now=clock)
        self.assertEqual(
            120,
            tracker.remaining(
                watchdog_stopped=False,
                true_remaining=120,
                wallclock_exhausted=False,
            ),
        )
        self.assertFalse(tracker.status()["minted"])

    def test_watchdog_trigger_still_capped_by_true_remaining(self):
        # A watchdog stop that leaves real budget must not overrun the deadline.
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=900, now=clock)
        self.assertEqual(
            42,
            tracker.remaining(
                watchdog_stopped=True,
                true_remaining=42,
                wallclock_exhausted=False,
            ),
        )

    def test_zero_cap_restores_hard_stop(self):
        clock = _FakeClock()
        tracker = WinddownBudgetTracker(max_seconds=0, now=clock)
        self.assertEqual(
            0,
            tracker.remaining(watchdog_stopped=True, true_remaining=61_200),
        )
        self.assertEqual(DEFAULT_WINDDOWN_MAX_SECONDS, 900)

    def test_triggered_event_carries_budget_bookkeeping(self):
        watchdog = _run_floor_stopped_watchdog()
        started_at = time.time() - 3_600.0
        monitor = CoverageProgressMonitor(
            runner=_MonitorRunner(),
            firmware=Path("firmware.elf"),
            output_path=Path("progress.jsonl"),
            interval_seconds=0,
            started_at=started_at,
            stall_watchdog=watchdog,
            wallclock_budget_seconds=86_400,
        )
        captured = []

        def capture_unchanged(event, *, stage=None, extra=None):
            captured.append((event, dict(extra or {})))
            return {"event": event, "stage": stage}

        monitor.snapshot_unchanged = capture_unchanged
        monitor._maybe_emit_stall_watchdog_event()

        event, extra = captured[0]
        self.assertEqual("stall_watchdog_triggered", event)
        self.assertEqual(86_400, extra["configured_budget_seconds"])
        # 24h budget with one hour already spent (small slack for the test's
        # own wallclock): the saving is directly readable from the event.
        self.assertGreaterEqual(extra["budget_remaining_seconds"], 82_000)
        self.assertLessEqual(extra["budget_remaining_seconds"], 82_800)
        self.assertGreaterEqual(extra["elapsed_seconds_at_capture"], 3_600.0)

    def test_budget_snapshot_frozen_at_first_capture(self):
        watchdog = _run_floor_stopped_watchdog()
        monitor = CoverageProgressMonitor(
            runner=_MonitorRunner(),
            firmware=Path("firmware.elf"),
            output_path=Path("progress.jsonl"),
            interval_seconds=0,
            started_at=time.time() - 1_800.0,
            stall_watchdog=watchdog,
            wallclock_budget_seconds=10_800,
        )
        first = {}
        monitor._observe_stall_watchdog(
            "interval",
            first,
            zero_yield_unit=True,
        )
        frozen = monitor.stall_watchdog_budget_snapshot
        self.assertIsNotNone(frozen)
        self.assertEqual(10_800, frozen["configured_budget_seconds"])
        second = {}
        monitor._observe_stall_watchdog(
            "interval",
            second,
            zero_yield_unit=True,
        )
        # Later observations must not move the captured saving figure.
        self.assertEqual(frozen, monitor.stall_watchdog_budget_snapshot)


class StallWatchdogExecutorTests(unittest.TestCase):
    def _executor(self, watchdog, runner=None):
        class _Monitor:
            def __init__(self):
                self.events = []

            def set_stage(self, stage):
                self.events.append(("set_stage", stage))

            def snapshot(self, event, *, stage=None, extra=None):
                self.events.append((event, stage, dict(extra or {})))

            def snapshot_unchanged(self, event, *, stage=None, extra=None):
                self.events.append((event, stage, dict(extra or {})))

        monitor = _Monitor()
        runner = runner or SimpleNamespace(
            phase_metadata={},
            phase_coverage={},
            global_coverage=set(),
            stall_watchdog=watchdog,
        )
        executor = ReplayExecutor(runner)
        runtime = SchedulerRuntime(
            runner=runner,
            args=SimpleNamespace(disable_frontier_stages=False),
            progress_monitor=monitor,
            remaining_wallclock_seconds=lambda: 10_000,
            has_wallclock_budget=lambda _minimum=1: True,
            has_stage_wallclock_budget=lambda *_args, **_kwargs: True,
            clamp_enabled_stage_seconds=lambda value, **_kwargs: value,
            short_probe_mode=lambda: False,
        )
        executor.bind_runtime(runtime)
        return executor, runner, monitor

    def test_executor_skips_frontier_stage_once_run_stopped(self):
        executor, runner, _monitor = self._executor(_stage_stopped_watchdog())
        covered = executor.run_frontier_successor(
            TAIL,
            600,
        )
        self.assertEqual(set(), covered)
        metadata = runner.phase_metadata[TAIL]
        self.assertTrue(metadata["skipped"])
        self.assertEqual(STOP_REASON, metadata["skip_reason"])

    def test_executor_skips_frontier_stage_once_stage_truncated(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        watchdog.observe(stage="setup", covered_bbs=1336)
        watchdog.notify_stage_begin(TAIL)
        clock.advance(700)
        watchdog.observe(stage=TAIL, covered_bbs=1336, zero_yield_unit=True)
        self.assertTrue(watchdog.should_truncate_stage(TAIL))
        executor, runner, _monitor = self._executor(watchdog)
        covered = executor.run_frontier_successor(TAIL, 600)
        self.assertEqual(set(), covered)
        metadata = runner.phase_metadata[TAIL]
        self.assertTrue(metadata["skipped"])
        self.assertEqual(STAGE_TRUNCATION_REASON, metadata["skip_reason"])
        # A truncation flag must not stop the run.
        self.assertFalse(watchdog.should_stop())

    def test_executor_without_triggered_watchdog_unchanged(self):
        executor, _, _ = self._executor(None)
        self.assertFalse(executor._stall_watchdog_should_stop())
        idle_executor, _, _ = self._executor(
            StallWatchdog(threshold_seconds=5400, min_zero_yield_units=8)
        )
        self.assertFalse(idle_executor._stall_watchdog_should_stop())

    def test_executor_brackets_feed_stage_lifecycle(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=600,
            min_zero_yield_units=1,
            max_consecutive_zero_stages=2,
            min_zero_productivity_stage_seconds=10,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        executor, runner, _monitor = self._executor(watchdog)
        watchdog.observe(stage="setup", covered_bbs=10)
        # The bracket-fed stage ran in microseconds of real wall time, so the
        # eligibility floor must be 0 for the lifecycle bookkeeping itself to
        # be exercised here.
        watchdog.min_zero_productivity_stage_seconds = 0

        def kernel():
            clock.advance(60)
            watchdog.observe(stage=TAIL, covered_bbs=10, zero_yield_unit=True)
            return {"covered": 0}

        executor.run_stage(TAIL, kernel)
        self.assertEqual(1, watchdog.status()["consecutive_zero_productivity_stage_count"])
        # Stage clock restarted for the next stage.
        self.assertEqual(0.0, watchdog.status()["stall_seconds_accrued"])


class _MergeRunner:
    """Minimal runner double exposing exactly what OutcomeFeedback needs.

    ``OutcomeFeedback.record_phase`` is the real stage-boundary merge site
    (scheduler/outcome_feedback.py: ``global_coverage.update(valid)`` plus
    ``_record_coverage_evidence``), so driving it directly proves where the
    truncated stage's earned BBs land.
    """

    def __init__(self):
        self.global_coverage: set[int] = set()
        self.coverage_by_evidence: dict[str, set[int]] = {}
        self.phase_coverage: dict[str, set[int]] = {}
        self.phase_metadata: dict[str, dict[str, object]] = {}
        self.phase_metadata_history: dict[str, list[dict[str, object]]] = {}
        self.scheduler_feedback: dict[str, object] = {}
        self.stall_watchdog: object = None

    def validate_coverage(self, covered):
        return set(covered)

    def _record_coverage_evidence(self, evidence_class, covered):
        self.coverage_by_evidence.setdefault(str(evidence_class), set()).update(set(covered))


class StallWatchdogStageTruncationMergeTests(unittest.TestCase):
    """Round-3 requirement 4: a truncated stage still merges its evidence.

    Observable assertions only -- the phase set must land in
    ``global_coverage`` through the real ``OutcomeFeedback.record_phase``
    (the exact call every exploration kernel makes at stage exit), the E2
    evidence set must contain the earned BBs, and the stage_end snapshot
    must report the post-merge global count.
    """

    BASE_GLOBAL = 1336
    EARNED = 346

    def _scenario(self, watchdog):
        runner = _MergeRunner()
        runner.global_coverage = {0x1000 + i for i in range(self.BASE_GLOBAL)}
        runner.stall_watchdog = watchdog
        feedback = OutcomeFeedback(runner=runner)
        return runner, feedback

    def test_record_phase_merges_phase_set_into_global_and_evidence(self):
        runner = _MergeRunner()
        runner.global_coverage = {0x1000 + i for i in range(self.BASE_GLOBAL)}
        feedback = OutcomeFeedback(runner=runner)
        phase_covered = {0x9000 + i for i in range(self.EARNED)}

        merged = feedback.record_phase(
            TAIL,
            set(phase_covered),
            evidence_class=EVIDENCE_E2,
        )

        self.assertEqual(self.EARNED, len(merged))
        self.assertEqual(
            self.BASE_GLOBAL + self.EARNED,
            len(runner.global_coverage),
        )
        self.assertTrue(phase_covered <= runner.global_coverage)
        self.assertTrue(phase_covered <= runner.coverage_by_evidence[EVIDENCE_E2])
        metadata = runner.phase_metadata[TAIL]
        self.assertEqual(self.EARNED, metadata["new_bbs"])
        self.assertTrue(metadata["coverage_counted"])

    def test_truncated_kernel_exit_merges_through_record_phase(self):
        clock = _FakeClock()
        watchdog = StallWatchdog(
            threshold_seconds=5400,
            min_zero_yield_units=8,
            run_stall_threshold_seconds=0,
            started_at=clock(),
            now=clock,
        )
        runner, feedback = self._scenario(watchdog)
        phase_covered: set[int] = set()
        earned_all = {0x9000 + i for i in range(self.EARNED)}
        flag_inside_kernel = {}

        executor = ReplayExecutor(runner)

        class _Monitor:
            def __init__(self):
                self.events = []

            def set_stage(self, stage):
                pass

            def snapshot(self, event, *, stage=None, extra=None):
                self.events.append((event, stage, dict(extra or {})))

            def snapshot_unchanged(self, event, *, stage=None, extra=None):
                self.events.append((event, stage, dict(extra or {})))

        monitor = _Monitor()
        executor.bind_runtime(
            SchedulerRuntime(
                runner=runner,
                args=SimpleNamespace(disable_frontier_stages=False),
                progress_monitor=monitor,
                remaining_wallclock_seconds=lambda: 10_000,
                has_wallclock_budget=lambda _minimum=1: True,
                has_stage_wallclock_budget=lambda *_args, **_kwargs: True,
                clamp_enabled_stage_seconds=lambda value, **_kwargs: value,
                short_probe_mode=lambda: False,
            )
        )

        def kernel():
            # Productive stretch: cf evidence grows every interval (covered
            # ledger views frozen/growing exactly like the control arm).
            cf_earned = 0
            earned_order = sorted(earned_all)
            for _ in range(6):
                clock.advance(600)
                cf_earned = min(self.EARNED, cf_earned + 58)
                phase_covered.clear()
                phase_covered.update(earned_order[:cf_earned])
                watchdog.observe(
                    stage=TAIL,
                    covered_bbs=len(runner.global_coverage | phase_covered),
                    zero_yield_unit=True,
                    evidence_ledgers={
                        "natural_supported_bbs": 866,
                        "counterfactual_only_bbs": 470 + cf_earned,
                    },
                )
                self.assertFalse(watchdog.should_truncate_stage(TAIL))
            # Dead stretch: every ledger frozen at the final values.
            while not watchdog.should_truncate_stage(TAIL):
                clock.advance(600)
                watchdog.observe(
                    stage=TAIL,
                    covered_bbs=len(runner.global_coverage | phase_covered),
                    zero_yield_unit=True,
                    evidence_ledgers={
                        "natural_supported_bbs": 866,
                        "counterfactual_only_bbs": 470 + self.EARNED,
                    },
                )
            flag_inside_kernel["set"] = True
            # The kernel's ordinary exit: _record_phase == the merge.
            return feedback.record_phase(
                TAIL,
                set(phase_covered),
                evidence_class=EVIDENCE_E2,
            )

        executor.run_stage(TAIL, kernel)

        # The truncation was observed inside the kernel (before finish_stage
        # cleared the flag) and exactly one truncation was recorded.
        self.assertTrue(flag_inside_kernel.get("set"))
        self.assertEqual(1, watchdog.status()["stage_truncation_count"])
        # ...but the run was never stopped.
        self.assertFalse(watchdog.should_stop())

        # The merge is observable: global coverage grew by the full earned
        # set and the evidence ledger holds the same BBs.
        self.assertEqual(
            self.BASE_GLOBAL + self.EARNED,
            len(runner.global_coverage),
        )
        self.assertTrue(earned_all <= runner.global_coverage)
        self.assertTrue(earned_all <= runner.coverage_by_evidence[EVIDENCE_E2])
        self.assertEqual(self.EARNED, runner.phase_metadata[TAIL]["new_bbs"])
        # The stage_end snapshot saw the post-merge global count.
        stage_end = [item for item in monitor.events if item[0] == "stage_end"]
        self.assertEqual(1, len(stage_end))
        self.assertEqual(
            self.BASE_GLOBAL + self.EARNED,
            stage_end[0][2]["global_covered_bbs"],
        )
        # The stage-level truncation did not mint a wind-down allowance and
        # the flag is gone once the stage finished.
        self.assertFalse(watchdog.should_truncate_stage(TAIL))


class StallWatchdogCampaignSummaryTests(unittest.TestCase):
    def test_load_report_summary_passes_watchdog_reason_through(self):
        from lsgemu.run_elfmultifuzz_interleaved_strict_campaign import load_report_summary

        with tempfile.TemporaryDirectory(prefix="lsgemu_watchdog_") as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(
                json.dumps({
                    "runner": "cached_interleaved_runner",
                    "stall_watchdog_stop_reason": STOP_REASON,
                    "stall_watchdog": {
                        "reason": STOP_REASON,
                        "level": "stage",
                        "stage": TAIL,
                        "threshold_seconds": 5400,
                        "stall_seconds": 5723.5,
                        "zero_yield_units": 9,
                    },
                    "stall_watchdog_stage_truncations": [
                        {"stage": TAIL, "level": "stage", "stage_stall_seconds": 5400.0}
                    ],
                }),
                encoding="utf-8",
            )
            summary = load_report_summary(report_path)
        self.assertEqual(STOP_REASON, summary["stall_watchdog_stop_reason"])
        self.assertEqual(
            TAIL,
            summary["stall_watchdog_stage_truncations"][0]["stage"],
        )

    def test_terminated_early_fields_pass_through_summarize_case_result(self):
        from lsgemu.run_elfmultifuzz_interleaved_strict_campaign import (
            summarize_case_result,
        )

        case = SimpleNamespace(
            family="MCUdatabase",
            name="demo",
            rel_path="MCUdatabase/demo.elf",
            elf_path=Path("demo.elf"),
            valid_bbs=set(),
        )
        with tempfile.TemporaryDirectory(prefix="lsgemu_watchdog_") as tmpdir:
            root = Path(tmpdir)
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps({
                    "runner": "cached_interleaved_runner",
                    "status": "completed",
                    "stall_watchdog_stop_reason": STOP_REASON,
                    "terminated_early": True,
                    "terminated_early_reason": STOP_REASON,
                    "wallclock_used_seconds": 16200.5,
                    "configured_budget_seconds": 86400,
                    "stall_watchdog_budget_remaining_seconds": 70199.5,
                    "stall_watchdog_stage_truncations": [
                        {"stage": TAIL, "level": "stage"}
                    ],
                    "execution_time_seconds": 16200.5,
                    "covered_bbs": 10,
                    "total_bbs": 20,
                    "valid_covered_bbs": 5,
                    "valid_total_bbs": 10,
                }),
                encoding="utf-8",
            )
            progress_path = root / "progress.jsonl"
            progress_path.write_text("", encoding="utf-8")
            result = summarize_case_result(
                case=case,
                case_output_dir=root,
                report_path=report_path,
                progress_path=progress_path,
                checkpoint_path=root / "checkpoint.jsonl",
                log_path=root / "run.log",
                exit_code=0,
                status="completed",
                duration_minutes=1440,
                interval_seconds=300,
            )
        self.assertTrue(result["terminated_early"])
        self.assertEqual(STOP_REASON, result["terminated_early_reason"])
        self.assertEqual(16200.5, result["wallclock_used_seconds"])
        self.assertEqual(86400, result["configured_budget_seconds"])
        self.assertEqual(70199.5, result["stall_watchdog_budget_remaining_seconds"])
        self.assertEqual(STOP_REASON, result["stall_watchdog_stop_reason"])
        self.assertEqual(
            TAIL,
            result["stall_watchdog_stage_truncations"][0]["stage"],
        )

    def test_union_report_carries_terminated_early_fields(self):
        from lsgemu.run_elfmultifuzz_interleaved_strict_parallel_campaign import (
            apply_strict_checkpoint_union,
        )

        case = SimpleNamespace(
            family="MCUdatabase",
            name="demo",
            rel_path="MCUdatabase/demo.elf",
            elf_path=Path("demo.elf"),
            valid_bbs={0x08000000, 0x08000004, 0x08000008},
        )
        with tempfile.TemporaryDirectory(prefix="lsgemu_watchdog_") as tmpdir:
            root = Path(tmpdir)
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps({
                    "runner": "cached_interleaved_runner",
                    "stall_watchdog_stop_reason": STOP_REASON,
                    "terminated_early": True,
                    "terminated_early_reason": STOP_REASON,
                    "wallclock_used_seconds": 16200.0,
                    "configured_budget_seconds": 86400,
                    "stall_watchdog_budget_remaining_seconds": 70200,
                    "stall_watchdog_stage_truncations": [
                        {"stage": TAIL, "level": "stage"}
                    ],
                    "covered_bb_list": ["0x08000000", "0x08000004"],
                    "covered_valid_bb_list": ["0x08000000", "0x08000004"],
                }),
                encoding="utf-8",
            )
            merged = apply_strict_checkpoint_union(
                case=case,
                result={
                    "status": "completed",
                    "report_file": str(report_path),
                    "covered_bbs": 0,
                    "valid_covered_bbs": 0,
                },
                case_output_dir=root,
                status="completed",
                resource_failure=False,
                duration_minutes=1440,
                interval_seconds=300,
            )
            union_report = json.loads(
                Path(str(merged["checkpoint_union_report_file"])).read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(merged["terminated_early"])
        self.assertEqual(STOP_REASON, merged["terminated_early_reason"])
        self.assertEqual(86400, merged["configured_budget_seconds"])
        self.assertEqual(70200, merged["stall_watchdog_budget_remaining_seconds"])
        self.assertEqual(
            TAIL,
            merged["stall_watchdog_stage_truncations"][0]["stage"],
        )
        self.assertTrue(union_report["terminated_early"])
        self.assertEqual(16200.0, union_report["wallclock_used_seconds"])
        self.assertEqual(86400, union_report["configured_budget_seconds"])
        self.assertEqual(70200, union_report["stall_watchdog_budget_remaining_seconds"])
        self.assertEqual(
            TAIL,
            union_report["stall_watchdog_stage_truncations"][0]["stage"],
        )


if __name__ == "__main__":
    unittest.main()
