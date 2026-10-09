#!/usr/bin/env python3
"""Evidence contracts for inferred dynamic-dispatch scheduler roots."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.path_naturalization import EVIDENCE_E0


class SyntheticDispatchEvidenceTests(unittest.TestCase):
    def test_real_branch_event_is_an_observed_direction(self):
        event = BranchSnapshotManager().record_event(
            0x08001000,
            0x08001002,
            0x08002000,
            0x08001004,
            "BNE",
            True,
            3,
        )
        self.assertTrue(event.original_direction_known)
        self.assertEqual("unicorn_execution", event.direction_provenance)
        self.assertFalse(event.synthetic)

    def test_successor_set_root_keeps_candidates_without_natural_direction(self):
        dispatch = 0x08003764
        first_target = 0x080055DA
        second_target = 0x080055EA
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.known_main_branch_events = {}
        runner.known_main_branch_event_order = 0
        runner.known_branch_root_snapshots = {}
        runner.known_branch_root_provenance_signatures = {}
        runner.dynamic_successors = {dispatch: {first_target, second_target}}
        runner.dynamic_successor_version = 0
        runner.effective_successor_cache = None
        runner.static_reachable_cache_by_successor_version = {}
        runner.observed_branch_directions = {}
        runner.branch_direction_evidence = {}
        runner.prepared = SimpleNamespace(
            static_bb_set={dispatch, first_target, second_target},
            static_successors={dispatch: set()},
            branch_instruction_by_bb={
                dispatch: {
                    "address": dispatch + 4,
                    "mnemonic": "BX",
                    "operands": "r3",
                }
            },
            resolve_basic_block=lambda value: (int(value) & ~1) if value else None,
        )
        runner.emulator = SimpleNamespace(
            branch_snapshot_manager=SimpleNamespace(
                get_snapshot=lambda address: SimpleNamespace(
                    address=address,
                    order=5,
                    capture_order=5,
                    depth=4,
                )
            )
        )
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._interrupt_context_mmio_state = lambda snapshot: {}
        runner._trim_root_snapshot_variants = lambda key: None

        added = runner._ensure_dynamic_dispatch_root_events(
            {dispatch},
            evidence_class=EVIDENCE_E0,
        )
        event = runner.known_main_branch_events[(dispatch, 1)]

        self.assertEqual(1, added)
        self.assertFalse(event.original_direction_known)
        self.assertEqual("dynamic_successor_set", event.direction_provenance)
        self.assertTrue(event.synthetic)
        self.assertEqual(
            [first_target, second_target],
            runner._branch_event_choice_values(event),
        )
        self.assertEqual(0, runner._remember_observed_branch_directions([event]))
        self.assertEqual(
            0,
            runner._remember_branch_direction_evidence([event], EVIDENCE_E0),
        )
        self.assertEqual({}, runner.observed_branch_directions)
        self.assertEqual({}, runner.branch_direction_evidence)
        serialized = runner._serialize_branch_event(event)
        self.assertFalse(serialized["original_direction_known"])
        self.assertTrue(serialized["synthetic"])


if __name__ == "__main__":
    unittest.main()
