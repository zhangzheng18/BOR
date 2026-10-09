#!/usr/bin/env python3
"""Regression tests for force-free path naturalization and evidence isolation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import HistoricalRunner
from lsgemu.causal_context import CausalExecutionContext
from lsgemu.analysis.branch_snapshot_manager import BranchEvent, BranchSnapshotManager
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from lsgemu.path_naturalization import (
    EVIDENCE_E0,
    EVIDENCE_E1,
    EVIDENCE_E2,
    EVIDENCE_E3,
    NaturalizationFact,
    PathNaturalizationLedger,
    choice_identity,
    derived_replay_evidence,
    evaluate_force_free_replay,
    match_path_signature,
    merge_constraint_sets,
    refine_constraint_sets,
    reason_category,
)
from lsgemu.runner_models import BranchConstraintCandidate, PrefixReplaySnapshot


def event(
    address: int,
    occurrence: int,
    *,
    taken: bool = False,
    condition: str = "NE",
    original_index=None,
    target: int = 0,
    context_signature=None,
):
    return SimpleNamespace(
        address=address,
        occurrence_index=occurrence,
        original_taken=taken,
        original_index=original_index,
        condition=condition,
        target=target,
        fallthrough=address + 2,
        alternatives=[],
        depth=occurrence,
        order=occurrence,
        first_depth=occurrence,
        first_order=occurrence,
        first_occurrence_index=occurrence,
        context_signature=dict(context_signature or {}),
    )


def mmio_candidate(value: int = 1) -> BranchConstraintCandidate:
    return BranchConstraintCandidate(
        constraint_type="mmio",
        address=0x40001000,
        value=value,
        read_pc=0x08000020,
        constraint_pc=0x08000024,
        source="test",
    )


def completed_run(stop_reason: str, **extra):
    """Return a complete force-free run result for replay test doubles."""
    return {
        "stop_reason": str(stop_reason),
        "instruction_count": 1,
        "execution_attempted": True,
        "execution_started": True,
        "execution_completed_normally": True,
        "execution_failed": False,
        "preflight_failed": False,
        "execution_telemetry_complete": True,
        "execution_intervention_reasons_authoritative": True,
        "execution_intervention_reasons": [],
        "execution_intervention_counts": {},
        "initial_state_fingerprint_summary": {
            "fingerprint_sha256": "test-initial-state",
        },
        **extra,
    }


class PathNaturalizationPureTests(unittest.TestCase):
    def test_persisted_mmio_occurrence_scope_survives_reload(self):
        payload = {
            "constraints": [
                {
                    "type": "mmio",
                    "read_pc": "0x08000020",
                    "address": "0x40001000",
                    "read_occurrence": 5,
                    "value": "0x00000001",
                    "added_by": "branch_mmio",
                }
            ]
        }
        with tempfile.TemporaryDirectory(prefix="lsgemu_occurrence_") as tmp:
            path = Path(tmp) / "constraints.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            handler = EnhancedMMIOHandler(
                None,
                str(path),
                branch_mmio_file_mode="all",
            )
            key = (0x08000020, 0x40001000, 5)
            self.assertEqual(1, handler.occurrence_constraints[key])
            self.assertNotIn(key[:2], handler.pc_constraints)
            handler.read_occurrence_counts[key[:2]] = 4
            handled, value = handler.resolve_bridge_read(key[0], key[1], 4)
            self.assertTrue(handled)
            self.assertEqual(1, value)
            self.assertIn(key, handler.occurrence_constraint_hit_keys)
            primary = StatefulMMIOHandler(
                constraint_json_path=str(path),
                branch_mmio_file_mode="all",
            )
            self.assertIsNone(primary._try_static_constraint(key[1], key[0]))

    def test_snapshot_manager_preserves_lossless_occurrence_stream(self):
        manager = BranchSnapshotManager()
        observed = [
            manager.record_event(
                0x1000,
                0x1002,
                0x1100,
                0x1004,
                "NE",
                taken,
                depth=index,
            )
            for index, taken in enumerate((False, False, True), start=1)
        ]
        self.assertEqual([1, 2, 3], [item.occurrence_index for item in observed])
        raw_events = manager.get_ordered_occurrence_events()
        self.assertEqual(3, len(raw_events))
        matched = match_path_signature(
            (
                ((0x1000, 1), False),
                ((0x1000, 2), False),
                ((0x1000, 3), True),
            ),
            raw_events,
        )
        self.assertTrue(matched.complete)

    def test_derived_replay_never_erases_forced_or_context_provenance(self):
        self.assertEqual(
            EVIDENCE_E1,
            derived_replay_evidence(
                EVIDENCE_E0,
                consumed_environment_fact=True,
            ),
        )
        self.assertEqual(
            EVIDENCE_E2,
            derived_replay_evidence(
                EVIDENCE_E0,
                forced_control=True,
                consumed_environment_fact=True,
            ),
        )
        self.assertEqual(
            EVIDENCE_E2,
            derived_replay_evidence(
                EVIDENCE_E2,
                consumed_environment_fact=True,
            ),
        )
        self.assertEqual(
            EVIDENCE_E3,
            derived_replay_evidence(
                EVIDENCE_E3,
                consumed_environment_fact=True,
            ),
        )

    def test_failure_reasons_map_to_modeling_actions(self):
        self.assertEqual(
            "missing_peripheral_or_input_fact",
            reason_category("missing_causal_fact:wrong_natural_choice"),
        )
        self.assertEqual(
            "transaction_or_temporal_state_conflict",
            reason_category("constraint_conflict"),
        )
        self.assertEqual(
            "missing_prefix_state",
            reason_category("missing_causal_fact:branch_occurrence_not_reached"),
        )

    def test_dynamic_occurrences_match_in_exact_order(self):
        signature = (
            ((0x1000, 1), False),
            ((0x1000, 2), True),
        )
        matched = match_path_signature(
            signature,
            [event(0x1000, 1, taken=False), event(0x1000, 2, taken=True)],
        )
        self.assertTrue(matched.complete)
        self.assertEqual(2, matched.matched_prefix_len)

        wrong = match_path_signature(
            signature,
            [event(0x1000, 1, taken=False), event(0x1000, 2, taken=False)],
        )
        self.assertFalse(wrong.complete)
        self.assertEqual("wrong_natural_choice", wrong.mismatch_reason)
        self.assertEqual((0x1000, 2), wrong.mismatch_key)

    def test_bool_switch_and_call_choices_remain_distinct(self):
        self.assertNotEqual(choice_identity(True), choice_identity(1))
        binary_match = match_path_signature(
            (((0x1000, 1), True),),
            [
                event(
                    0x1000,
                    1,
                    taken=True,
                    condition="NE",
                    original_index=0,
                )
            ],
        )
        self.assertTrue(binary_match.complete)

        switch_match = match_path_signature(
            (((0x2000, 1), 2),),
            [event(0x2000, 1, condition="TBB", original_index=2)],
        )
        self.assertTrue(switch_match.complete)

        call_match = match_path_signature(
            (((0x3000, 1), 0x08004000),),
            [
                event(
                    0x3000,
                    1,
                    condition="BX",
                    original_index=0,
                    target=0x08004001,
                )
            ],
        )
        self.assertTrue(call_match.complete)

    def test_constraint_merge_rejects_conflicts(self):
        merged, reason = merge_constraint_sets([mmio_candidate(1)], [mmio_candidate(2)])
        self.assertIsNone(merged)
        self.assertEqual("conflicting_constraint_values", reason)

    def test_constraint_refinement_replaces_one_runtime_input_site(self):
        previous = mmio_candidate(1)
        replacement = BranchConstraintCandidate(
            constraint_type="mmio",
            address=previous.address,
            value=2,
            read_pc=previous.read_pc,
            constraint_pc=0x08000080,
            source="deeper_dynamic_model",
            dependency_group="different_provenance_group",
            width=8,
        )
        refined, reason, changes = refine_constraint_sets(
            (previous,),
            (replacement,),
        )
        self.assertIsNone(reason)
        self.assertEqual(1, len(refined or ()))
        self.assertEqual(2, int((refined or ())[0].value))
        self.assertEqual(1, len(changes))
        self.assertEqual(1, int(changes[0].previous.value))
        self.assertEqual(2, int(changes[0].replacement.value))

    def test_memory_fact_is_not_external_environment_evidence(self):
        fact = NaturalizationFact(
            branch_key=(0x1000, 1),
            choice=True,
            constraints=(
                BranchConstraintCandidate(
                    constraint_type="memory",
                    address=0x20000000,
                    value=1,
                    read_pc=0x08000010,
                ),
            ),
            paired_control=True,
            candidate_consumed=True,
            local_success=True,
        )
        self.assertFalse(fact.externally_replayable)

        ledger = PathNaturalizationLedger()
        obligation = ledger.register_obligation(
            (((0x1000, 1), True),),
            target_bbs={0x1100},
        )
        self.assertIsNotNone(obligation)
        ledger.register_fact(fact)
        extensions, reason = ledger.candidate_extensions(obligation, (0x1000, 1), True)
        self.assertEqual([], extensions)
        self.assertEqual("fact_not_externalized", reason)

        external_memory_fact = NaturalizationFact(
            branch_key=(0x1000, 1),
            choice=True,
            constraints=(
                BranchConstraintCandidate(
                    constraint_type="memory",
                    address=0x20000100,
                    value=1,
                    read_pc=0x08000010,
                    read_occurrence=2,
                    input_kind="external_memory",
                    externally_controllable=True,
                ),
            ),
            paired_control=True,
            candidate_consumed=True,
            local_success=True,
        )
        self.assertTrue(external_memory_fact.externally_replayable)

    def test_new_target_or_fact_reopens_an_obligation(self):
        ledger = PathNaturalizationLedger()
        signature = (((0x1000, 1), True),)
        obligation = ledger.register_obligation(signature, target_bbs={0x1100})
        self.assertIsNotNone(obligation)
        ledger.promote(
            obligation,
            promoted_bbs={0x1100},
            evidence_class=EVIDENCE_E1,
            source_phase="test",
        )
        obligation.attempts = 7
        ledger.register_obligation(
            signature,
            target_bbs={0x1100, 0x1200},
            discovered_bbs={0x1200},
        )
        self.assertEqual("pending", obligation.status)
        self.assertEqual(0, obligation.attempts)

        obligation.status = "exhausted"
        obligation.attempts = 8
        upstream_obligation = ledger.register_obligation(
            (((0x3000, 1), True),),
            target_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), True),
                ((0x3000, 1), True),
            ),
        )
        self.assertIsNotNone(upstream_obligation)
        upstream_obligation.status = "blocked_no_fact"
        upstream_obligation.attempts = 8
        ledger.register_fact(
            NaturalizationFact(
                branch_key=(0x1000, 1),
                choice=True,
                constraints=(mmio_candidate(1),),
                paired_control=True,
                candidate_consumed=True,
                local_success=True,
            )
        )
        self.assertEqual("pending", obligation.status)
        self.assertEqual(0, obligation.attempts)
        self.assertEqual("pending", upstream_obligation.status)
        self.assertEqual(0, upstream_obligation.attempts)

    def test_same_path_keeps_distinct_witnessed_discovery_contexts(self):
        ledger = PathNaturalizationLedger()
        signature = (((0x3000, 1), True),)
        first = ledger.register_obligation(
            signature,
            target_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), False),
                ((0x3000, 1), True),
            ),
        )
        second = ledger.register_obligation(
            signature,
            target_bbs={0x3100},
            discovery_trace_signature=(
                ((0x2000, 1), True),
                ((0x3000, 1), True),
            ),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertEqual(2, len(ledger.obligations))

        updated = ledger.register_obligation(
            signature,
            target_bbs={0x3200},
        )
        self.assertIsNot(updated, first)
        self.assertIsNot(updated, second)
        self.assertEqual(3, len(ledger.obligations))
        summary = ledger.summary()
        self.assertEqual(1, summary["unique_path_signatures"])
        self.assertEqual(
            1,
            summary["signatures_with_multiple_discovery_contexts"],
        )

    def test_candidate_must_be_consumed_and_force_free(self):
        signature = (((0x1000, 1), True),)
        control_events = [event(0x1000, 1, taken=False)]
        result_events = [event(0x1000, 1, taken=True)]
        common = dict(
            signature=signature,
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            control_events=control_events,
            control_coverage={0x1000},
            result_events=result_events,
            result_coverage={0x1000, 0x1100},
            constraints=[mmio_candidate(1)],
            forced_trace_count=0,
        )
        unconsumed = evaluate_force_free_replay(
            **common,
            constraint_feedback={"all_constraint_reads_matched": False},
        )
        self.assertFalse(unconsumed["success"])
        self.assertEqual("candidate_not_consumed", unconsumed["reason"])

        configured_but_unhit = evaluate_force_free_replay(
            **common,
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_choices_configured=1,
        )
        self.assertTrue(configured_but_unhit["success"])
        self.assertEqual("promoted", configured_but_unhit["reason"])

        forced = evaluate_force_free_replay(
            **{**common, "forced_trace_count": 1},
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_choices_configured=1,
        )
        self.assertFalse(forced["success"])
        self.assertEqual("forced_control_observed_in_validation", forced["reason"])

    def test_force_free_target_delta_is_promotable(self):
        result = evaluate_force_free_replay(
            signature=(((0x1000, 1), True),),
            target_bbs={0x1100},
            discovered_bbs={0x1100, 0x1110},
            control_events=[event(0x1000, 1, taken=False)],
            control_coverage={0x1000},
            result_events=[event(0x1000, 1, taken=True)],
            result_coverage={0x1000, 0x1100, 0x1110},
            constraints=[mmio_candidate(1)],
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_trace_count=0,
        )
        self.assertTrue(result["success"])
        self.assertFalse(result["already_natural"])
        self.assertEqual("promoted", result["reason"])
        self.assertEqual({0x1100, 0x1110}, set(result["promoted_bbs"]))

    def test_force_free_promotion_rejects_different_irq_lineage(self):
        control_context = CausalExecutionContext()
        result_context = CausalExecutionContext()
        result_context.record_irq_deliver(7, pc=0x08000010)
        result = evaluate_force_free_replay(
            signature=(((0x1000, 1), True),),
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            control_events=[event(
                0x1000,
                1,
                taken=False,
                context_signature=control_context.branch_signature(),
            )],
            control_coverage={0x1000},
            result_events=[event(
                0x1000,
                1,
                taken=True,
                context_signature=result_context.branch_signature(),
            )],
            result_coverage={0x1000, 0x1100},
            constraints=[mmio_candidate(1)],
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_trace_count=0,
        )
        self.assertFalse(result["success"])
        self.assertEqual("replay_context_incompatible", result["reason"])
        self.assertIn(
            "irq_stack",
            result["context_compatibility"]["hard_mismatches"],
        )


class PathNaturalizationRunnerTests(unittest.TestCase):
    def test_snapshot_replay_uses_dynamic_input_watermark(self):
        runner = object.__new__(HistoricalRunner)
        read_site = (0x08000020, 0x40001000)
        snapshot = SimpleNamespace(
            address=0x08001000,
            input_occurrence_counts={read_site: 4},
        )
        snapshot_entry = PrefixReplaySnapshot(
            prefix_signature=tuple(),
            next_branch_key=(0x08001000, 1),
            snapshot=snapshot,
        )
        future_candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=read_site[1],
            value=1,
            read_pc=read_site[0],
            read_occurrence=5,
        )
        consumed_candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=read_site[1],
            value=1,
            read_pc=read_site[0],
            read_occurrence=4,
        )
        self.assertTrue(
            runner._snapshot_replay_can_revisit_constraints(
                snapshot_entry,
                [future_candidate],
            )
        )
        self.assertFalse(
            runner._snapshot_replay_can_revisit_constraints(
                snapshot_entry,
                [consumed_candidate],
            )
        )

        emulator = SimpleNamespace(input_occurrence_counts={read_site: 4})
        temp_mmio = SimpleNamespace(read_occurrence_counts={read_site: 99})
        runner._restore_replay_mmio_state(emulator, temp_mmio, {})
        self.assertEqual({read_site: 4}, temp_mmio.read_occurrence_counts)

    def test_snapshot_evidence_is_explicit_and_legacy_safe(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.known_main_branch_events = {}

        explicit_context = PrefixReplaySnapshot(
            prefix_signature=tuple(),
            next_branch_key=(0x1000, 1),
            snapshot=object(),
            evidence_class=EVIDENCE_E3,
        )
        self.assertEqual(
            EVIDENCE_E3,
            runner._snapshot_entry_evidence_class(explicit_context),
        )

        legacy_entry = PrefixReplaySnapshot(
            prefix_signature=tuple(),
            next_branch_key=(0x1000, 1),
            snapshot=object(),
            source_path_signature=(((0x1000, 1), True),),
        )
        self.assertEqual(
            EVIDENCE_E2,
            runner._snapshot_entry_evidence_class(legacy_entry),
        )
        runner.path_naturalization_ledger.register_path_evidence(
            legacy_entry.source_path_signature,
            EVIDENCE_E1,
            source_phase="later_force_free_validation",
        )
        self.assertEqual(
            EVIDENCE_E2,
            runner._snapshot_entry_evidence_class(legacy_entry),
        )

        legacy_reset_entry = PrefixReplaySnapshot(
            prefix_signature=tuple(),
            next_branch_key=(0x1000, 1),
            snapshot=object(),
        )
        self.assertEqual(
            EVIDENCE_E0,
            runner._snapshot_entry_evidence_class(legacy_reset_entry),
        )

        runner.known_branch_root_snapshots = {
            (0x1000, 1): {tuple(): explicit_context},
        }
        runner.direct_call_snapshots = {}
        runner.stream_summary_entry_snapshots = {}
        runner.reservoir_prefix_snapshots = {}
        provenance = runner._snapshot_provenance_summary()
        self.assertEqual(1, provenance["entries"])
        self.assertEqual(
            1,
            provenance["effective_evidence_counts"][EVIDENCE_E3],
        )

    def test_replay_path_composes_snapshot_source_and_verified_suffix(self):
        runner = object.__new__(HistoricalRunner)
        self.assertIs(
            True,
            runner._branch_event_evidence_choice(
                event(
                    0x0800,
                    1,
                    taken=True,
                    condition="NE",
                    original_index=0,
                )
            ),
        )
        self.assertEqual(
            3,
            runner._branch_event_evidence_choice(
                event(
                    0x0900,
                    1,
                    condition="TBB",
                    original_index=3,
                )
            ),
        )
        source = (
            ((0x1000, 1), False),
            ((0x2000, 1), True),
        )
        suffix = (
            ((0x3000, 2), 4),
            ((0x4000, 1), True),
        )
        self.assertEqual(
            source + suffix,
            runner._compose_replay_path_signature(source, suffix),
        )

        shared_trace = source + (((0x3000, 2), 0),)
        snapshot_entry = PrefixReplaySnapshot(
            prefix_signature=tuple(),
            next_branch_key=(0x3000, 2),
            snapshot=object(),
            source_path_signature=source,
            evidence_class=EVIDENCE_E2,
            source_occurrence_trace=shared_trace,
            source_occurrence_prefix_len=2,
        )
        self.assertEqual(
            source,
            runner._snapshot_entry_occurrence_prefix(snapshot_entry),
        )

    def test_root_snapshot_keeps_lossless_entry_occurrence_prefix(self):
        runner = object.__new__(HistoricalRunner)
        first = event(0x1000, 1, taken=False)
        second = event(0x2000, 1, taken=True)
        runner.prepared = SimpleNamespace(branch_instruction_by_bb={})
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.known_main_branch_events = {
            (0x1000, 1): first,
            (0x2000, 1): second,
        }
        runner.known_branch_root_snapshots = {}
        runner.direct_call_snapshots = {}
        runner.stream_summary_entry_snapshots = {}
        snapshot = SimpleNamespace(
            address=0x2000,
            condition="NE",
            occurrence_index=1,
            order=2,
            capture_order=2,
            depth=2,
            mmio_state={},
        )

        added = runner._remember_branch_root_snapshots(
            [second],
            [snapshot],
            evidence_class=EVIDENCE_E0,
            occurrence_events=[first, second],
        )
        self.assertEqual(1, added)
        entry = next(iter(
            runner.known_branch_root_snapshots[(0x2000, 1)].values()
        ))
        self.assertEqual(
            (((0x1000, 1), False),),
            runner._snapshot_entry_occurrence_prefix(entry),
        )
        self.assertEqual(
            (((0x1000, 1), False),),
            runner._scope_signature_from_snapshot_entry(entry),
        )

    def test_root_snapshot_accepts_slotted_branch_event_occurrence_remap(self):
        runner = object.__new__(HistoricalRunner)
        root_event = BranchEvent(
            address=0x2000,
            branch_pc=0x2000,
            target=0x2100,
            fallthrough=0x2002,
            condition="NE",
            original_taken=True,
            order=1,
            depth=1,
            occurrence_index=1,
        )
        runner.prepared = SimpleNamespace(branch_instruction_by_bb={})
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.known_main_branch_events = {}
        runner.known_branch_root_snapshots = {}
        runner.direct_call_snapshots = {}
        runner.stream_summary_entry_snapshots = {}
        runner.snapshot_resource_stats = Counter()
        snapshot = SimpleNamespace(
            address=0x2000,
            condition="NE",
            occurrence_index=1,
            order=1,
            capture_order=1,
            depth=1,
            mmio_state={},
        )

        added = runner._remember_branch_root_snapshots(
            [root_event],
            [snapshot],
            evidence_class=EVIDENCE_E0,
            occurrence_events=[root_event],
            source_occurrence_signature=(((0x2000, 1), True),),
        )
        self.assertGreaterEqual(added, 1)
        self.assertIn((0x2000, 1), runner.known_branch_root_snapshots)

    def test_root_snapshot_variants_do_not_collapse_dynamic_prefixes(self):
        runner = object.__new__(HistoricalRunner)
        source_event = event(0x1000, 1, taken=False)
        root_event = event(0x2000, 1, taken=True)
        runner.prepared = SimpleNamespace(branch_instruction_by_bb={})
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.known_main_branch_events = {
            (0x1000, 1): source_event,
            (0x2000, 1): root_event,
        }
        runner.known_branch_root_snapshots = {}
        runner.direct_call_snapshots = {}
        runner.stream_summary_entry_snapshots = {}
        runner.snapshot_resource_stats = Counter()
        snapshot = SimpleNamespace(
            address=0x2000,
            condition="NE",
            occurrence_index=1,
            order=3,
            capture_order=3,
            depth=2,
            mmio_state={},
        )
        sparse_source = (((0x1000, 1), False),)
        occurrence_prefixes = (
            (((0x1000, 1), False),),
            (
                ((0x1000, 1), False),
                ((0x1000, 2), True),
            ),
        )
        for occurrence_prefix in occurrence_prefixes:
            runner._remember_branch_root_snapshots(
                [root_event],
                [snapshot],
                source_path_signature=sparse_source,
                evidence_class=EVIDENCE_E1,
                occurrence_events=[root_event],
                source_occurrence_signature=occurrence_prefix,
            )
        entries = runner._root_snapshot_entries((0x2000, 1))
        self.assertEqual(2, len(entries))
        self.assertEqual(
            set(occurrence_prefixes),
            {
                runner._scope_signature_from_snapshot_entry(entry)
                for entry in entries
            },
        )

    def test_snapshot_identity_distinguishes_lossless_occurrence_prefix(self):
        runner = object.__new__(HistoricalRunner)
        snapshot = SimpleNamespace(address=0x2000, order=2)
        common = {
            "prefix_signature": tuple(),
            "next_branch_key": (0x2000, 1),
            "snapshot": snapshot,
            "source_path_signature": (((0x1000, 1), False),),
            "evidence_class": EVIDENCE_E1,
        }
        first = PrefixReplaySnapshot(
            **common,
            source_occurrence_trace=(((0x1000, 1), False),),
            source_occurrence_prefix_len=1,
        )
        second = PrefixReplaySnapshot(
            **common,
            source_occurrence_trace=(
                ((0x1000, 1), False),
                ((0x1000, 2), True),
            ),
            source_occurrence_prefix_len=2,
        )
        self.assertNotEqual(
            runner._snapshot_entry_identity(first),
            runner._snapshot_entry_identity(second),
        )

    def test_direct_call_contexts_keep_distinct_occurrence_prefixes(self):
        runner = object.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(branch_instruction_by_bb={})
        runner.known_main_branch_events = {}
        runner.direct_call_snapshots = {}
        snapshot = SimpleNamespace(
            address=0x3000,
            condition="BL",
            capture_order=3,
            order=3,
            depth=2,
            mmio_state={},
        )
        sparse_source = (((0x1000, 1), False),)
        first_occurrences = (((0x1000, 1), False),)
        second_occurrences = (
            ((0x1000, 1), False),
            ((0x1000, 2), True),
        )
        self.assertTrue(runner._remember_direct_call_snapshot(
            snapshot,
            sparse_source,
            evidence_class=EVIDENCE_E1,
            source_occurrence_signature=first_occurrences,
        ))
        self.assertTrue(runner._remember_direct_call_snapshot(
            snapshot,
            sparse_source,
            evidence_class=EVIDENCE_E1,
            source_occurrence_signature=second_occurrences,
        ))
        entries = list(runner.direct_call_snapshots[0x3000].values())
        self.assertEqual(2, len(entries))
        self.assertEqual(
            {first_occurrences, second_occurrences},
            {
                runner._scope_signature_from_snapshot_entry(entry)
                for entry in entries
            },
        )

    def test_occurrence_prefix_is_truncated_at_snapshot_capture(self):
        runner = object.__new__(HistoricalRunner)
        events = [
            event(0x1000, 1, taken=False),
            event(0x2000, 1, taken=True),
        ]
        events[0].order = 1
        events[1].order = 2
        snapshot = SimpleNamespace(
            address=0x2000,
            occurrence_index=1,
            capture_order=2,
            order=2,
        )
        self.assertEqual(
            (((0x1000, 1), False),),
            runner._occurrence_prefix_before_snapshot(
                tuple(),
                events,
                snapshot,
            ),
        )

    def test_interrupt_context_selection_preserves_distinct_input_timelines(self):
        runner = object.__new__(HistoricalRunner)
        runner.emulator = SimpleNamespace()
        runner._ordered_replay_snapshots = lambda emulator: []
        snapshot = SimpleNamespace(
            address=0x2000,
            registers={"sp": 1, "lr": 2, "pc": 3},
        )
        first = {
            "snapshot": snapshot,
            "mmio_state": {0x40000000: 1},
            "source_occurrence_signature": (((0x1000, 1), False),),
        }
        second = {
            "snapshot": snapshot,
            "mmio_state": {0x40000000: 1},
            "source_occurrence_signature": (
                ((0x1000, 1), False),
                ((0x1000, 2), True),
            ),
        }
        runner.reservoir_interrupt_contexts = [first, second]
        selected = runner._select_interrupt_contexts(
            max_contexts=0,
            include_reservoir_contexts=True,
            max_reservoir_contexts=2,
        )
        self.assertEqual(2, len(selected))

    def test_paired_replay_fact_uses_changed_occurrence(self):
        runner = object.__new__(HistoricalRunner)
        changed_key = runner._paired_replay_changed_branch_key(
            branch_bb_start=0x1000,
            target_direction=True,
            control_events=[
                event(0x1000, 1, taken=False),
                event(0x1000, 2, taken=True),
            ],
            result_events=[
                event(0x1000, 1, taken=True),
                event(0x1000, 2, taken=True),
            ],
            fallback_key=(0x1000, 2),
        )
        self.assertEqual((0x1000, 1), changed_key)

    def test_constraint_feedback_compares_effective_read_width(self):
        runner = object.__new__(HistoricalRunner)
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=0x100,
            read_pc=0x08000020,
            read_occurrence=1,
            input_kind="mmio",
            externally_controllable=True,
            width=8,
        )
        mmio = SimpleNamespace(
            access_occurrence_history=[
                (0x08000020, 0x40001000, 0, True, 1),
            ],
            access_history=[],
        )
        feedback = runner._collect_constraint_runtime_feedback(
            mmio,
            (candidate,),
        )
        self.assertTrue(feedback["all_constraint_reads_matched"])
        self.assertEqual(1, feedback["matched_requested_occurrences"])
        self.assertEqual(0, feedback["value_delivery_mismatches"])

    def test_replay_cache_signature_ignores_non_runtime_provenance(self):
        runner = object.__new__(HistoricalRunner)
        first = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=1,
            read_pc=0x08000020,
            read_occurrence=2,
            constraint_pc=0x08000024,
            source="static_dependency",
            dependency_group="g1",
            width=8,
        )
        second = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=1,
            read_pc=0x08000020,
            read_occurrence=2,
            constraint_pc=0x08000080,
            source="dynamic_ssa",
            dependency_group="g2",
            width=16,
        )
        self.assertEqual(
            runner._constraint_items_signature((first,)),
            runner._constraint_items_signature((second,)),
        )

    def test_e2_event_does_not_become_natural_direction(self):
        runner = object.__new__(HistoricalRunner)
        runner.known_main_branch_events = {}
        runner.known_main_branch_event_order = 0
        runner.observed_branch_directions = {}
        runner.branch_direction_evidence = {}
        runner._refresh_loop_exit_iteration_hints = lambda: None

        forced_event = event(0x1000, 1, taken=True)
        runner._remember_main_branch_events([forced_event], evidence_class=EVIDENCE_E2)
        self.assertEqual(set(), runner._observed_branch_directions(0x1000))

        natural_event = event(0x1000, 2, taken=False)
        runner._remember_main_branch_events([natural_event], evidence_class=EVIDENCE_E1)
        self.assertEqual({False}, runner._observed_branch_directions(0x1000))

    def test_runner_promotes_only_through_force_free_entry_replay(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._remember_branch_root_snapshots = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0

        def record_phase(name, covered, **metadata):
            valid = set(covered or set())
            runner.global_coverage.update(valid)
            runner.phase_coverage[name] = valid
            runner.phase_metadata[name] = dict(metadata)
            return valid

        runner._record_phase = record_phase
        signature = (((0x1000, 1), True),)
        obligation = runner.path_naturalization_ledger.register_obligation(
            signature,
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            source_phase="forced_test",
        )
        self.assertIsNotNone(obligation)
        runner.path_naturalization_ledger.register_fact(
            NaturalizationFact(
                branch_key=(0x1000, 1),
                choice=True,
                constraints=(mmio_candidate(1),),
                source_phase="branch_mmio",
                strategy="entry_fallback",
                paired_control=True,
                candidate_consumed=True,
                local_success=True,
            )
        )

        replay_calls = []

        def replay(constraints, replay_instructions, replay_timeout):
            items = list(constraints or [])
            replay_calls.append(items)
            if items:
                return (
                    {0x1000, 0x1100},
                    completed_run(
                        "target",
                        constraint_feedback={"all_constraint_reads_matched": True},
                        forced_branch_trace_count=0,
                        forced_branch_choices_configured=0,
                    ),
                    [event(0x1000, 1, taken=True)],
                    [],
                )
            return (
                {0x1000},
                completed_run(
                    "control",
                    constraint_feedback={"all_constraint_reads_matched": False},
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=False)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        covered = runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=2,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual(2, len(replay_calls))
        self.assertEqual({0x1000, 0x1100}, covered)
        self.assertEqual("promoted", obligation.status)
        self.assertEqual({0x1100}, runner.naturalized_coverage)
        self.assertIn(0x1100, runner.coverage_by_evidence[EVIDENCE_E1])

    def test_runner_bootstraps_first_fact_from_force_free_entry_replay(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._remember_branch_root_snapshots = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0

        def record_phase(name, covered, **metadata):
            valid = set(covered or set())
            runner.global_coverage.update(valid)
            runner.phase_coverage[name] = valid
            runner.phase_metadata[name] = dict(metadata)
            return valid

        runner._record_phase = record_phase
        signature = (((0x1000, 1), True),)
        obligation = runner.path_naturalization_ledger.register_obligation(
            signature,
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            source_phase="forced_test",
        )
        self.assertIsNotNone(obligation)
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=1,
            read_pc=0x08000020,
            read_occurrence=1,
            input_kind="mmio",
            externally_controllable=True,
        )
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: [candidate]
        )

        def replay(constraints, replay_instructions, replay_timeout):
            items = list(constraints or [])
            if items:
                return (
                    {0x1000, 0x1100},
                    completed_run(
                        "target",
                        constraint_feedback={"all_constraint_reads_matched": True},
                        forced_branch_trace_count=0,
                        forced_branch_choices_configured=0,
                    ),
                    [event(0x1000, 1, taken=True)],
                    [],
                )
            return (
                {0x1000},
                completed_run(
                    "control",
                    constraint_feedback={"all_constraint_reads_matched": False},
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=False)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        covered = runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=2,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual({0x1000, 0x1100}, covered)
        self.assertEqual("promoted", obligation.status)
        summary = runner.path_naturalization_ledger.summary()
        self.assertEqual("path_naturalization.v5", summary["schema"])
        self.assertEqual(1, summary["externally_replayable_facts"])
        self.assertEqual(1, runner.phase_metadata["path_naturalization"]["bootstrap_facts_registered"])

    def test_runner_retries_another_value_for_the_same_consumed_input(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._remember_branch_root_snapshots = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0

        def record_phase(name, covered, **metadata):
            valid = set(covered or set())
            runner.global_coverage.update(valid)
            runner.phase_coverage[name] = valid
            runner.phase_metadata[name] = dict(metadata)
            return valid

        runner._record_phase = record_phase
        obligation = runner.path_naturalization_ledger.register_obligation(
            (((0x1000, 1), True),),
            target_bbs={0x1100},
            discovered_bbs={0x1100},
            source_phase="forced_test",
        )
        first = mmio_candidate(1)
        second = mmio_candidate(0x20)
        runner.path_naturalization_ledger.register_fact(
            NaturalizationFact(
                branch_key=(0x1000, 1),
                choice=True,
                constraints=(first,),
                source_phase="earlier_context",
                strategy="paired_existing_fact",
                paired_control=True,
                candidate_consumed=True,
                local_success=True,
            )
        )
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: [first, second]
        )
        replay_values = []

        def replay(constraints, replay_instructions, replay_timeout):
            items = list(constraints or [])
            value = int(items[-1].value) if items else None
            replay_values.append(value)
            if value == 0x20:
                return (
                    {0x1000, 0x1100},
                    completed_run(
                        "target",
                        constraint_feedback={
                            "all_constraint_reads_matched": True,
                            "matched_constraint_reads": 1,
                            "matched_address_reads": 1,
                        },
                        forced_branch_trace_count=0,
                        forced_branch_choices_configured=0,
                    ),
                    [event(0x1000, 1, taken=True)],
                    [],
                )
            return (
                {0x1000},
                completed_run(
                    "wrong_direction",
                    constraint_feedback={
                        "all_constraint_reads_matched": bool(items),
                        "matched_constraint_reads": int(bool(items)),
                        "matched_address_reads": int(bool(items)),
                    },
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=False)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        covered = runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=3,
            max_fact_candidates_per_edge=1,
            max_values_per_source=2,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual([None, 1, 0x20], replay_values)
        self.assertEqual("promoted", obligation.status)
        self.assertEqual({0x1000, 0x1100}, covered)
        self.assertEqual(
            1,
            runner.phase_metadata["path_naturalization"]
            ["failure_category_counts"]
            ["consumed_but_predicate_unchanged"],
        )

    def test_runner_refines_earlier_input_while_preserving_cross_bb_prefix(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._remember_branch_root_snapshots = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0

        def record_phase(name, covered, **metadata):
            valid = set(covered or set())
            runner.global_coverage.update(valid)
            runner.phase_coverage[name] = valid
            runner.phase_metadata[name] = dict(metadata)
            return valid

        runner._record_phase = record_phase
        signature = (
            ((0x1000, 1), True),
            ((0x2000, 1), True),
        )
        obligation = runner.path_naturalization_ledger.register_obligation(
            signature,
            target_bbs={0x2100},
            discovered_bbs={0x2100},
            discovery_trace_signature=signature,
            source_phase="forced_cross_bb_test",
        )
        self.assertIsNotNone(obligation)

        def input_value(value):
            return BranchConstraintCandidate(
                constraint_type="mmio",
                address=0x40001000,
                value=value,
                read_pc=0x08000020,
                read_occurrence=1,
                input_kind="mmio",
                externally_controllable=True,
                dependency_group=(
                    "first_predicate" if value == 1 else "deeper_predicate"
                ),
                width=8,
            )

        def bootstrap(branch_key, expected_choice, **kwargs):
            self.assertTrue(expected_choice)
            if branch_key == (0x1000, 1):
                return [input_value(1)]
            if branch_key == (0x2000, 1):
                return [input_value(2)]
            return []

        runner._naturalization_bootstrap_candidates = bootstrap
        replay_values = []

        def replay(constraints, replay_instructions, replay_timeout):
            items = list(constraints or [])
            value = int(items[0].value) if items else None
            replay_values.append(value)
            feedback = {
                "all_constraint_reads_matched": bool(items),
                "matched_constraint_reads": int(bool(items)),
                "matched_address_reads": int(bool(items)),
                "matched_requested_occurrences": int(bool(items)),
            }
            common = {
                "constraint_feedback": feedback,
                "forced_branch_trace_count": 0,
                "forced_branch_choices_configured": 0,
            }
            if value == 2:
                return (
                    {0x1000, 0x2000, 0x2100},
                    completed_run("target", **common),
                    [
                        event(0x1000, 1, taken=True),
                        event(0x2000, 1, taken=True),
                    ],
                    [],
                )
            if value == 1:
                return (
                    {0x1000, 0x2000},
                    completed_run("second_predicate", **common),
                    [
                        event(0x1000, 1, taken=True),
                        event(0x2000, 1, taken=False),
                    ],
                    [],
                )
            return (
                {0x1000},
                completed_run("first_predicate", **common),
                [event(0x1000, 1, taken=False)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        covered = runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=2,
            max_fact_candidates_per_edge=1,
            max_values_per_source=1,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual([None, 1, 2], replay_values)
        self.assertEqual("promoted", obligation.status)
        self.assertEqual(2, int(obligation.accepted_constraints[0].value))
        self.assertEqual(1, obligation.accepted_constraint_refinements)
        self.assertEqual({0x1000, 0x2000, 0x2100}, covered)
        metadata = runner.phase_metadata["path_naturalization"]
        self.assertEqual(1, metadata["constraint_refinement_replays"])
        self.assertEqual(1, metadata["accepted_constraint_refinements"])

    def test_compound_candidates_require_explicit_composite_provenance(self):
        runner = object.__new__(HistoricalRunner)
        first = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=1,
            read_pc=0x2000,
            dependency_group="r0|ORR|pair",
            composite_dependency=True,
        )
        second = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40002000,
            value=2,
            read_pc=0x2002,
            dependency_group="r0|ORR|pair",
            composite_dependency=True,
        )
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: [first, second]
        )
        candidate_sets = runner._naturalization_bootstrap_candidate_sets(
            (0x1000, 1),
            True,
            max_sources=2,
            max_values_per_source=1,
            max_compound_candidates=1,
        )
        self.assertEqual([1, 1, 2], [len(items) for items in candidate_sets])

        non_composite = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40002000,
            value=2,
            read_pc=0x2002,
        )
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: [first, non_composite]
        )
        candidate_sets = runner._naturalization_bootstrap_candidate_sets(
            (0x1000, 1),
            True,
            max_sources=2,
            max_values_per_source=1,
            max_compound_candidates=1,
        )
        self.assertEqual([1, 1], [len(items) for items in candidate_sets])

    def test_occurrence_retarget_uses_observed_read_timeline(self):
        runner = object.__new__(HistoricalRunner)
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=0x20,
            read_pc=0x2000,
            read_occurrence=5,
        )
        variants = runner._naturalization_occurrence_retargets(
            (candidate,),
            {
                "items": [{
                    "constraint": {
                        "type": "mmio",
                        "address": "0x40001000",
                        "read_pc": "0x00002000",
                        "read_occurrence": 5,
                    },
                    "address_read_seen": True,
                    "exact_value_seen": False,
                    "observed_occurrence": 4,
                }],
            },
        )
        self.assertEqual(1, len(variants))
        self.assertEqual(4, variants[0][0].read_occurrence)
        self.assertIn("observed_occurrence_retarget", variants[0][0].source)

    def test_missing_deep_branch_repairs_first_upstream_trace_divergence(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._remember_branch_root_snapshots = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0

        def record_phase(name, covered, **metadata):
            valid = set(covered or set())
            runner.global_coverage.update(valid)
            runner.phase_coverage[name] = valid
            runner.phase_metadata[name] = dict(metadata)
            return valid

        runner._record_phase = record_phase
        obligation = runner.path_naturalization_ledger.register_obligation(
            (((0x3000, 1), True),),
            target_bbs={0x3100},
            discovered_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), False),
                ((0x2000, 1), True),
                ((0x3000, 1), True),
            ),
            source_phase="forced_deep_test",
        )
        self.assertIsNotNone(obligation)

        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=0,
            read_pc=0x08000020,
            read_occurrence=1,
            input_kind="mmio",
            externally_controllable=True,
        )
        selected_edges = []

        def bootstrap(branch_key, expected_choice, **kwargs):
            selected_edges.append((branch_key, expected_choice))
            return [candidate]

        runner._naturalization_bootstrap_candidates = bootstrap

        def replay(constraints, replay_instructions, replay_timeout):
            if constraints:
                return (
                    {0x1000, 0x2000, 0x3000, 0x3100},
                    completed_run(
                        "target",
                        constraint_feedback={
                            "all_constraint_reads_matched": True,
                        },
                        forced_branch_trace_count=0,
                        forced_branch_choices_configured=0,
                    ),
                    [
                        event(0x1000, 1, taken=False),
                        event(0x2000, 1, taken=True),
                        event(0x3000, 1, taken=True),
                    ],
                    [],
                )
            return (
                {0x1000},
                completed_run(
                    "control_diverged",
                    constraint_feedback={
                        "all_constraint_reads_matched": False,
                    },
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=True)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        covered = runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=2,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual([((0x1000, 1), False)], selected_edges)
        self.assertEqual("promoted", obligation.status)
        self.assertEqual({0x1000, 0x2000, 0x3000, 0x3100}, covered)
        self.assertEqual(
            1,
            runner.phase_metadata["path_naturalization"]
            ["upstream_discovery_mismatches_selected"],
        )

    def test_discovery_only_prefix_harvest_keeps_concrete_provenance(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())
        runner._remember_main_branch_events = lambda *args, **kwargs: 0
        runner._record_dynamic_successors = lambda *args, **kwargs: 0
        harvested_sources = []

        def remember_snapshots(*args, **kwargs):
            harvested_sources.append(tuple(kwargs.get("source_path_signature") or ()))
            return 0

        runner._remember_branch_root_snapshots = remember_snapshots

        def record_phase(name, covered, **metadata):
            runner.phase_coverage[name] = set(covered or set())
            runner.phase_metadata[name] = dict(metadata)
            return runner.phase_coverage[name]

        runner._record_phase = record_phase
        obligation = runner.path_naturalization_ledger.register_obligation(
            (((0x3000, 1), True),),
            target_bbs={0x3100},
            discovered_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), False),
                ((0x3000, 1), True),
            ),
        )
        self.assertIsNotNone(obligation)
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40001000,
            value=1,
            read_pc=0x08000020,
            read_occurrence=1,
            input_kind="mmio",
            externally_controllable=True,
        )
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: [candidate]
        )

        def replay(constraints, replay_instructions, replay_timeout):
            items = list(constraints or ())
            if items:
                return (
                    {0x1000},
                    completed_run(
                        "advanced_to_deeper_prefix",
                        constraint_feedback={
                            "all_constraint_reads_matched": True,
                            "matched_constraint_reads": 1,
                            "matched_address_reads": 1,
                            "matched_requested_occurrences": 1,
                        },
                        forced_branch_trace_count=0,
                        forced_branch_choices_configured=0,
                    ),
                    [event(0x1000, 1, taken=False)],
                    [],
                )
            return (
                {0x1000},
                completed_run(
                    "upstream_divergence",
                    constraint_feedback={
                        "all_constraint_reads_matched": False,
                    },
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=True)],
                [],
            )

        runner._replay_mmio_from_entry = replay
        runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=1,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual([(((0x1000, 1), False),)], harvested_sources)
        self.assertEqual(1, obligation.matched_discovery_prefix_len)
        self.assertEqual(
            1,
            runner.phase_metadata["path_naturalization"]
            ["discovery_prefix_provenance_harvests"],
        )

    def test_missing_deep_branch_without_observed_divergence_is_not_guessed(self):
        runner = object.__new__(HistoricalRunner)
        runner.path_naturalization_ledger = PathNaturalizationLedger()
        runner.naturalized_coverage = set()
        runner.coverage_by_evidence = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.validate_coverage = lambda covered: set(covered or set())

        def record_phase(name, covered, **metadata):
            runner.phase_coverage[name] = set(covered or set())
            runner.phase_metadata[name] = dict(metadata)
            return runner.phase_coverage[name]

        runner._record_phase = record_phase
        obligation = runner.path_naturalization_ledger.register_obligation(
            (((0x3000, 1), True),),
            target_bbs={0x3100},
            discovered_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), False),
                ((0x3000, 1), True),
            ),
        )
        self.assertIsNotNone(obligation)
        runner._naturalization_bootstrap_candidates = (
            lambda *args, **kwargs: self.fail(
                "an unexecuted deep predicate must not be guessed"
            )
        )
        runner._replay_mmio_from_entry = (
            lambda constraints, replay_instructions, replay_timeout: (
                {0x1000},
                completed_run(
                    "control_ended_before_deep_branch",
                    constraint_feedback={
                        "all_constraint_reads_matched": False,
                    },
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                ),
                [event(0x1000, 1, taken=False)],
                [],
            )
        )
        runner.run_path_naturalization(
            time_limit_seconds=5,
            max_paths=1,
            max_attempts_per_path=2,
            replay_instructions=100,
            replay_timeout=1000,
        )
        self.assertEqual("blocked_no_fact", obligation.status)
        self.assertTrue(
            obligation.last_reason.startswith(
                "missing_observed_upstream_divergence:"
            )
        )


if __name__ == "__main__":
    unittest.main()
