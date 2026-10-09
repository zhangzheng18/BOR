#!/usr/bin/env python3
"""Contract tests for the five scheduler component boundaries."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import threading
import unittest

from lsgemu.historical_runner import HistoricalRunner
from lsgemu.path_naturalization import EVIDENCE_E1
from lsgemu.scheduler.context_recovery import ContextRecovery
from lsgemu.scheduler.contracts import SchedulerRuntime
from lsgemu.scheduler.obligation_discovery import (
    ObligationDiscovery,
    semantic_entry_categories_for_name,
)
from lsgemu.scheduler.outcome_feedback import OutcomeFeedback
from lsgemu.scheduler.queue_policy import (
    QueuePolicy,
    dedupe_target_order,
    estimate_targeted_frontier_reserve_seconds,
    rotate_targets,
)
from lsgemu.scheduler.replay_executor import ReplayExecutor


class FakeMonitor:
    def __init__(self):
        self.events = []

    def set_stage(self, stage):
        self.events.append(("set_stage", stage))

    def snapshot(self, event, *, stage=None, extra=None):
        self.events.append((event, stage, dict(extra or {})))

    def snapshot_unchanged(self, event, *, stage=None, extra=None):
        self.events.append((event, stage, dict(extra or {})))


class SchedulerComponentsTest(unittest.TestCase):
    def test_runner_constructs_exactly_five_scheduler_boundaries(self):
        runner = object.__new__(HistoricalRunner)
        runner.semantic_obligation_enabled = True
        runner._ensure_scheduler_components()

        summary = runner.scheduler_component_summary()
        self.assertEqual(
            [
                "obligation_discovery",
                "queue_policy",
                "replay_executor",
                "outcome_feedback",
                "context_recovery",
            ],
            [item["name"] for item in summary["components"]],
        )
        self.assertTrue(summary["invariants"]["coverage_oracle_preserved"])

    def test_queue_policy_preserves_stable_order(self):
        self.assertEqual([3, 1, 2, 4], dedupe_target_order([3, 1, 2], [1, 4]))
        self.assertEqual([3, 4], rotate_targets([1, 2, 3, 4], 1, 2))

        selected, kept = QueuePolicy.pop_best(
            deque([("first", 1), ("best", 3), ("tie", 3), ("last", 0)]),
            lambda item: item[1],
        )
        self.assertEqual(("best", 3), selected)
        self.assertEqual(
            [("first", 1), ("tie", 3), ("last", 0)],
            list(kept),
        )

    def test_queue_budget_formula_matches_historical_values(self):
        short = estimate_targeted_frontier_reserve_seconds(
            total_wallclock_budget_seconds=600,
            switch_frontier_enabled=True,
            frontier_targeted_enabled=True,
            frontier_cycle_auto=True,
            switch_round_seconds=20,
            switch_rounds=0,
            switch_max_rounds=2,
            frontier_round_seconds=30,
            frontier_rounds=0,
            frontier_max_rounds=2,
            frontier_cycle_max_cycles=4,
            frontier_cycle_tail_reserve_seconds=10,
        )
        self.assertEqual(96, short)

    def test_outcome_feedback_can_only_count_valid_oracle_bbs(self):
        runner = SimpleNamespace(
            global_coverage={0x1000},
            phase_coverage={},
            phase_metadata={},
            coverage_by_evidence={},
            scheduler_feedback={"stale": True},
            validate_coverage=lambda covered: set(covered) & {0x1000, 0x2000},
            _runner_state_lock=lambda: threading.RLock(),
            _temp_lifecycle_phase_summary=lambda _name: {"created": 0},
        )

        def record_evidence(evidence, covered):
            runner.coverage_by_evidence.setdefault(evidence, set()).update(covered)

        runner._record_coverage_evidence = record_evidence
        feedback = OutcomeFeedback(runner)
        result = feedback.record_phase(
            "probe",
            {0x2000, 0xDEADBEEF},
            evidence_class=EVIDENCE_E1,
        )

        self.assertEqual({0x2000}, result)
        self.assertEqual({0x1000, 0x2000}, runner.global_coverage)
        self.assertNotIn(0xDEADBEEF, runner.global_coverage)
        self.assertEqual(1, runner.phase_metadata["probe"]["new_bbs"])
        self.assertEqual({0x2000}, runner.coverage_by_evidence[EVIDENCE_E1])
        self.assertEqual({}, runner.scheduler_feedback)

    def test_outcome_feedback_records_lifecycle_summary_failure(self):
        runner = SimpleNamespace(
            global_coverage=set(),
            phase_coverage={},
            phase_metadata={},
            scheduler_feedback={},
            validate_coverage=lambda covered: set(covered),
            _runner_state_lock=lambda: threading.RLock(),
            _record_coverage_evidence=lambda _evidence, _covered: None,
            _temp_lifecycle_phase_summary=lambda _name: (_ for _ in ()).throw(
                RuntimeError("lifecycle unavailable")
            ),
        )

        OutcomeFeedback(runner).record_phase("probe", {0x1000})

        error = runner.phase_metadata["probe"]["temp_emulator_lifecycle_error"]
        self.assertEqual("RuntimeError", error["error_type"])
        self.assertIn("lifecycle unavailable", error["error"])

    def test_replay_executor_keeps_stage_lifecycle_order(self):
        monitor = FakeMonitor()
        runner = SimpleNamespace(
            phase_metadata={},
            phase_coverage={},
            global_coverage={0x1000},
            lifecycle=[],
            cleanups=[],
        )
        runner.set_lifecycle_stage = lambda stage: runner.lifecycle.append(stage)
        runner._dispose_active_temp_emulators = (
            lambda **kwargs: runner.cleanups.append(dict(kwargs)) or 2
        )
        executor = ReplayExecutor(runner)
        runtime = SchedulerRuntime(
            runner=runner,
            args=SimpleNamespace(),
            progress_monitor=monitor,
            remaining_wallclock_seconds=lambda: 100,
            has_wallclock_budget=lambda _minimum=1: True,
            has_stage_wallclock_budget=lambda *_args, **_kwargs: True,
            clamp_enabled_stage_seconds=lambda value, **_kwargs: value,
            short_probe_mode=lambda: False,
        )
        executor.bind_runtime(runtime)

        result = executor.run_stage("stage_a", lambda: {0x1000, 0x2000})

        self.assertEqual({0x1000, 0x2000}, result)
        self.assertEqual(["stage_a"], runner.lifecycle)
        self.assertEqual(1, len(runner.cleanups))
        self.assertEqual("stage_end", runner.cleanups[0]["reason"])
        self.assertEqual("stage_start", monitor.events[1][0])
        self.assertEqual("stage_end", monitor.events[-1][0])
        self.assertEqual(2, monitor.events[-1][2]["stage_covered_bbs"])

    def test_replay_executor_records_failure_and_unbinds_runtime(self):
        monitor = FakeMonitor()
        runner = SimpleNamespace(
            phase_metadata={},
            phase_coverage={},
            global_coverage={0x1000},
            lifecycle=[],
            cleanups=[],
        )
        runner.set_lifecycle_stage = lambda stage: runner.lifecycle.append(stage)
        runner._dispose_active_temp_emulators = (
            lambda **kwargs: runner.cleanups.append(dict(kwargs)) or 1
        )
        executor = ReplayExecutor(runner)
        runtime = SchedulerRuntime(
            runner=runner,
            args=SimpleNamespace(),
            progress_monitor=monitor,
            remaining_wallclock_seconds=lambda: 100,
            has_wallclock_budget=lambda _minimum=1: True,
            has_stage_wallclock_budget=lambda *_args, **_kwargs: True,
            clamp_enabled_stage_seconds=lambda value, **_kwargs: value,
            short_probe_mode=lambda: False,
        )
        executor.bind_runtime(runtime)

        with self.assertRaisesRegex(RuntimeError, "stage exploded"):
            executor.run_stage(
                "stage_failure",
                lambda: (_ for _ in ()).throw(RuntimeError("stage exploded")),
            )

        metadata = runner.phase_metadata["stage_failure"]
        self.assertTrue(metadata["stage_failed"])
        self.assertEqual("RuntimeError", metadata["stage_error"]["error_type"])
        self.assertEqual("stage_failed", monitor.events[-1][0])
        self.assertEqual(0, executor.active_stage_depth)
        self.assertTrue(executor.unbind_runtime(runtime))
        self.assertIsNone(executor.runtime)

    def test_replay_executor_restores_enclosing_stage_after_nested_replay(self):
        monitor = FakeMonitor()
        runner = SimpleNamespace(
            phase_metadata={},
            phase_coverage={},
            global_coverage=set(),
            lifecycle=[],
        )
        runner.set_lifecycle_stage = lambda stage: runner.lifecycle.append(stage)
        runner._dispose_active_temp_emulators = lambda **_kwargs: 0
        executor = ReplayExecutor(runner)
        executor.bind_runtime(SchedulerRuntime(
            runner=runner,
            args=SimpleNamespace(),
            progress_monitor=monitor,
            remaining_wallclock_seconds=lambda: 100,
            has_wallclock_budget=lambda _minimum=1: True,
            has_stage_wallclock_budget=lambda *_args, **_kwargs: True,
            clamp_enabled_stage_seconds=lambda value, **_kwargs: value,
            short_probe_mode=lambda: False,
        ))

        parent = executor.begin_stage("parent")
        executor.run_stage("child", lambda: set())
        self.assertEqual(["parent"], executor.active_stage_names())
        self.assertEqual("parent", monitor.events[-1][1])
        executor.finish_stage(parent, set(), coverage_counted=False)
        self.assertEqual(0, executor.active_stage_depth)

    def test_context_recovery_forwards_context_without_policy_changes(self):
        calls = []
        runner = SimpleNamespace()
        runner.run_contextual_isr_exploration = lambda **kwargs: calls.append(
            ("isr", dict(kwargs))
        ) or {1}
        runner.run_stream_input_exploration = lambda **kwargs: calls.append(
            ("stream", dict(kwargs))
        ) or {2}
        runner.run_rtos_thread_entry_exploration = lambda **kwargs: calls.append(
            ("thread", dict(kwargs))
        ) or {3}
        recovery = ContextRecovery(runner)

        self.assertEqual(
            {1},
            recovery.contextual_isr(
                max_contexts=4,
                max_isrs=2,
                max_instructions=100,
                replay_timeout=200,
                time_limit_seconds=3,
                stage_name="contextual_cycle",
            ),
        )
        self.assertEqual(4, calls[0][1]["max_contexts"])
        self.assertEqual(200, calls[0][1]["replay_timeout"])
        self.assertEqual("contextual_cycle", calls[0][1]["phase_name"])

    def test_obligation_discovery_classifies_semantic_targets(self):
        runner = SimpleNamespace(
            semantic_obligation_enabled=True,
            _function_name_for_addr=lambda _address: "UART_RxCallback",
        )
        discovery = ObligationDiscovery(runner)
        categories = discovery.categories_for_bb(0x1000)
        self.assertIn("callback_or_hook", categories)
        self.assertIn("stream_or_protocol_input", categories)
        self.assertGreater(discovery.priority_for_bb(0x1000), 0)
        self.assertIn(
            "interrupt_or_vector_handler",
            semantic_entry_categories_for_name("UART_IRQHandler"),
        )


if __name__ == "__main__":
    unittest.main()
