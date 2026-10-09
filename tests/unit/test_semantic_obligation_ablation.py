#!/usr/bin/env python3
"""Contract tests for the true semantic-obligation ablation."""

from types import SimpleNamespace
import unittest

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.path_naturalization import PathNaturalizationLedger


class SemanticObligationAblationTest(unittest.TestCase):
    def make_runner(self) -> HistoricalRunner:
        runner = object.__new__(HistoricalRunner)
        runner.semantic_obligation_enabled = False
        runner.scheduler_feedback = {"actions": [{"action": "must_not_escape"}]}
        runner.branch_snapshot_hotset = {0x1000}
        runner.deadlock_failed_directions = {0x2000: {0x2004: {True}}}
        runner.loop_exit_iteration_hints = {}
        runner.unproven_dynamic_code_targets = {}
        runner.emulator = SimpleNamespace(runtime_written_pages=set())
        runner.prepared = SimpleNamespace(loop_exit_iteration_hints={})
        return runner

    def test_semantic_priority_and_feedback_are_neutralized(self):
        runner = self.make_runner()
        self.assertEqual(0, runner._semantic_category_priority({"protocol_callback"}))
        feedback = runner._scheduler_feedback_summary()
        self.assertTrue(feedback["disabled"])
        self.assertFalse(feedback["has_actionable_feedback"])
        self.assertEqual([], feedback["actions"])
        self.assertEqual(
            [],
            runner.scheduler_feedback_target_bb_list(actions=["target_branch_frontier"]),
        )

    def test_emulator_receives_no_semantic_intervention_state(self):
        runner = self.make_runner()
        emulator = SimpleNamespace(
            runtime_written_pages=set(),
            loop_intervention_threshold_cache={},
        )
        runner._configure_emulator_feedback(emulator)
        self.assertFalse(emulator.semantic_obligation_enabled)
        self.assertFalse(emulator.enable_loop_intervention)
        self.assertEqual(set(), emulator.branch_snapshot_hotset)
        self.assertEqual({}, emulator.deadlock_failed_directions)
        self.assertEqual({}, emulator.loop_exit_iteration_hints)
        self.assertEqual(set(), runner.branch_snapshot_hotset)
        self.assertEqual({}, runner.deadlock_failed_directions)

    def test_semantic_category_state_is_absent_but_full_mode_is_unchanged(self):
        runner = self.make_runner()
        self.assertEqual(set(), runner._semantic_categories_for_bb(0x1000))

        runner.semantic_obligation_enabled = True
        runner._function_name_for_addr = lambda _address: "protocol_callback"
        self.assertIn("callback_or_hook", runner._semantic_categories_for_bb(0x1000))

    def test_generic_hot_loop_fallback_does_not_consult_semantic_analysis(self):
        emulator = object.__new__(IntelligentEmulator)
        emulator.semantic_obligation_enabled = False
        emulator.enable_loop_intervention = False
        emulator.hot_loop_halt_threshold = 100
        emulator.hot_loop_min_no_new_bbs = 50
        emulator._no_new_bb_run = 50

        def fail_if_called(_loop_head):
            raise AssertionError("semantic loop analysis escaped the ablation")

        emulator._get_loop_intervention_soft_floor = fail_if_called
        self.assertTrue(emulator._should_halt_hot_loop_without_intervention(0x1000, 100))

    def test_stagnation_analysis_cannot_create_hotset_obligations(self):
        emulator = object.__new__(IntelligentEmulator)
        emulator.semantic_obligation_enabled = False
        emulator.last_deadlock_attempt = {0x1000: (0x1010, True)}
        emulator.deadlock_failed_directions = {}
        emulator.branch_snapshot_hotset = set()
        emulator._mark_deadlock_attempt_failed(0x1000)
        emulator._register_branch_snapshot_interest_from_analysis(
            SimpleNamespace(suggested_constraints=[], critical_pc=0x1010)
        )
        self.assertEqual({}, emulator.deadlock_failed_directions)
        self.assertEqual(set(), emulator.branch_snapshot_hotset)

    def test_forced_replay_cannot_recreate_semantic_path_obligations(self):
        runner = self.make_runner()
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        created = runner._register_forced_path_obligation(
            (((0x1000, 1), True),),
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            source_phase="generic_reservoir_fallback",
        )
        self.assertFalse(created)
        self.assertEqual(0, runner.path_naturalization_ledger.summary()["obligations"])


if __name__ == "__main__":
    unittest.main()
