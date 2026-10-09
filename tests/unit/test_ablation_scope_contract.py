#!/usr/bin/env python3
"""Contract tests for contribution-level ablation boundaries."""

from types import SimpleNamespace
import unittest

from lsgemu.historical_runner import HistoricalRunner
from lsgemu.ablation_entrypoint import ABLATION_ARGS
from lsgemu.cached_interleaved_runner import build_ablation_fallback_policy
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from lsgemu.path_naturalization import (
    EVIDENCE_E0,
    EVIDENCE_E1,
    EVIDENCE_E2,
    EVIDENCE_E3,
)
from lsgemu.runner_models import BranchConstraintCandidate


class ConstraintProbe:
    def __init__(self):
        self.calls = []

    def add_constraint(self, address, value, *, added_by):
        self.calls.append(("global", address, value, added_by))

    def add_pc_constraint(self, read_pc, address, value, *, added_by):
        self.calls.append(("pc", read_pc, address, value, added_by))

    def add_occurrence_constraint(
        self,
        read_pc,
        address,
        occurrence,
        value,
        *,
        added_by,
    ):
        self.calls.append(
            ("occurrence", read_pc, address, occurrence, value, added_by)
        )


class AblationScopeContractTests(unittest.TestCase):
    def make_runner(self) -> HistoricalRunner:
        runner = object.__new__(HistoricalRunner)
        runner.semantic_obligation_enabled = True
        runner.scoped_replay_enabled = True
        runner.causal_constraint_recovery_enabled = True
        runner.scoped_branch_constraints = {}
        runner.scoped_branch_constraint_match_cache = {}
        runner.scoped_branch_constraint_presence_cache = {}
        return runner

    def test_a1_retains_cross_bb_causal_constraint_recovery(self):
        runner = self.make_runner()
        runner.semantic_obligation_enabled = False

        self.assertTrue(runner._causal_constraint_recovery_available())

    def test_a2_entrypoint_uses_versioned_output_identity(self):
        self.assertIn("no_scoped_replay_v2", ABLATION_ARGS)
        self.assertNotIn("no_scoped_replay", ABLATION_ARGS)

    def test_a3_report_distinguishes_event_tasks_from_preserved_direct_calls(self):
        args = SimpleNamespace(
            disable_semantic_obligation_stages=False,
            disable_scoped_replay_stages=False,
            disable_context_event_replay_stages=True,
            disable_branch_reservoir_stages=False,
            disable_frontier_stages=False,
            disable_semantic_frontier_drain_stage=False,
            disable_contextual_isr_stages=True,
            disable_direct_stream_thread_stages=False,
        )

        switches = build_ablation_fallback_policy(args)["effective_internal_switches"]

        self.assertTrue(switches["contextual_isr_replay_disabled"])
        self.assertTrue(switches["stream_input_event_replay_disabled"])
        self.assertTrue(switches["rtos_thread_entry_replay_disabled"])
        self.assertFalse(switches["direct_call_continuation_disabled"])

    def test_a2_globalizes_mmio_and_drops_memory_scope(self):
        runner = self.make_runner()
        runner.scoped_replay_enabled = False
        probe = ConstraintProbe()
        mmio = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40002000,
            value=0x20,
            read_pc=0x08001000,
            read_occurrence=7,
            input_kind="mmio",
            externally_controllable=True,
        )
        memory = BranchConstraintCandidate(
            constraint_type="memory",
            address=0x20000100,
            value=0x41,
            read_pc=0x08001100,
            read_occurrence=2,
            input_kind="external_memory",
            externally_controllable=True,
        )

        runner._apply_temp_mmio_constraints(
            probe,
            [mmio, memory],
            emulator=SimpleNamespace(),
        )

        self.assertEqual(1, len(probe.calls))
        self.assertEqual("global", probe.calls[0][0])
        self.assertEqual(0x40002000, probe.calls[0][1])
        self.assertIn("global_ablation", probe.calls[0][3])

    def test_a2_cannot_store_or_match_scoped_constraints(self):
        runner = self.make_runner()
        runner.scoped_replay_enabled = False
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40002000,
            value=1,
            read_pc=0x08001000,
            input_kind="mmio",
            externally_controllable=True,
        )
        scope = (((0x08002000, 1), True),)

        self.assertFalse(
            runner._remember_scoped_branch_constraint(scope, candidate)
        )
        self.assertEqual([], runner._matching_scoped_branch_constraints(scope))
        self.assertEqual({}, runner.scoped_branch_constraints)

    def test_a2_does_not_reload_learned_scope_from_constraint_file(self):
        learned = {
            "type": "mmio",
            "address": "0x40002000",
            "value": "0x20",
            "read_pc": "0x08001000",
            "read_occurrence": 7,
            "added_by": "intelligent_emulator",
        }
        static_base = {
            "type": "mmio",
            "address": "0x40021004",
            "value": "0x2",
            "read_pc": "0x08002000",
            "added_by": "static_constraint_analyzer",
            "speculation_level": "conservative_loop_exit",
        }
        emulator = object.__new__(IntelligentEmulator)
        emulator.branch_mmio_file_mode = "unscoped-global"

        for predicate in (
            lambda item: StatefulMMIOHandler._should_load_file_constraint(
                item,
                "unscoped-global",
            ),
            lambda item: EnhancedMMIOHandler._should_load_file_constraint(
                item,
                "unscoped-global",
            ),
            emulator._should_load_file_constraint,
        ):
            self.assertFalse(predicate(learned))
            self.assertTrue(predicate(static_base))

    def test_e2_e3_report_partitions_are_mutually_exclusive(self):
        runner = self.make_runner()
        runner.prepared = SimpleNamespace(static_bb_set={1, 2, 3, 4, 5})
        runner.coverage_by_evidence = {
            EVIDENCE_E0: {1},
            EVIDENCE_E1: {2},
            EVIDENCE_E2: {2, 3, 4},
            EVIDENCE_E3: {4, 5},
        }

        partitions = runner._coverage_evidence_partitions()

        self.assertEqual({1, 2}, partitions["witness"])
        self.assertEqual({3}, partitions["e2_only"])
        self.assertEqual({5}, partitions["e3_only"])
        self.assertEqual({4}, partitions["e2_e3_shared"])
        self.assertEqual({3, 4, 5}, partitions["counterfactual_only"])


if __name__ == "__main__":
    unittest.main()
