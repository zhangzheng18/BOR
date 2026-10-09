"""Regression tests for wrapper-aware snapshot provenance."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager
from lsgemu.evidence_contract import snapshot_provenance_record
from lsgemu.evidence_contract import (
    DIAGNOSTIC_REPLAY,
    VALIDATED_REPLAY,
    build_execution_evidence_record,
    compact_record_is_verified_validated,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.runner_models import PrefixReplaySnapshot


def _wrapper(source: object, **kwargs: object) -> PrefixReplaySnapshot:
    return PrefixReplaySnapshot(
        prefix_signature=tuple(),
        next_branch_key=(0x08001000, 1),
        snapshot=source,
        **kwargs,
    )


class SnapshotProvenanceContractTests(unittest.TestCase):
    @staticmethod
    def _snapshot(execution_id="execution-1"):
        return SimpleNamespace(
            address=0x1000,
            source_execution_id=execution_id,
            provenance_status="validated",
            provenance_reasons=tuple(),
            prefix_intervention_reasons=tuple(),
            prefix_telemetry_complete=True,
            provenance_finalized=True,
            provenance_invalidated=False,
            provenance_invalidation_reasons=tuple(),
        )

    def test_wrapper_reads_finalized_state_from_mutated_underlying_snapshot(self):
        source = SimpleNamespace(
            provenance_status="pending",
            provenance_reasons=tuple(),
            source_execution_id="execution-1",
            prefix_telemetry_complete=False,
            provenance_finalized=False,
        )
        wrapper = _wrapper(source)

        before = HistoricalRunner._snapshot_provenance_record(wrapper)
        self.assertEqual("pending", before["status"])
        self.assertFalse(before["snapshot_ancestor_validated"])

        # BranchSnapshotManager.finalize_snapshot_provenance mutates the
        # underlying object after the wrapper was created.
        source.provenance_status = "validated"
        source.prefix_telemetry_complete = True
        source.provenance_finalized = True

        after = snapshot_provenance_record(wrapper)
        self.assertEqual("validated", after["status"])
        self.assertEqual("execution-1", after["execution_id"])
        self.assertTrue(after["provenance_finalized"])
        self.assertTrue(after["snapshot_ancestor_validated"])
        self.assertEqual([], after["reasons"])

    def test_wrapper_keeps_real_intervention_from_any_lineage_layer(self):
        source = SimpleNamespace(
            provenance_status="validated",
            provenance_reasons=tuple(),
            source_execution_id="execution-2",
            prefix_telemetry_complete=True,
            provenance_finalized=True,
        )
        wrapper = _wrapper(
            source,
            provenance_status="diagnostic",
            provenance_reasons=("forced_branch",),
            prefix_intervention_reasons=("forced_branch",),
            source_execution_id="execution-2",
            prefix_telemetry_complete=True,
            provenance_finalized=True,
        )

        result = snapshot_provenance_record(wrapper)
        self.assertEqual("diagnostic", result["status"])
        self.assertIn("forced_branch", result["reasons"])
        self.assertFalse(result["snapshot_ancestor_validated"])

    def test_missing_finalization_is_not_promoted(self):
        source = SimpleNamespace(
            provenance_status="validated",
            source_execution_id="legacy-execution",
            prefix_telemetry_complete=True,
        )
        result = snapshot_provenance_record(_wrapper(source))
        self.assertNotEqual("validated", result["status"])
        self.assertFalse(result["provenance_finalized"])
        self.assertFalse(result["snapshot_ancestor_validated"])
        self.assertTrue(
            any("finalization" in str(reason) for reason in result["reasons"])
        )

    def test_mismatched_wrapper_id_cannot_inherit_finalized_source(self):
        source = SimpleNamespace(
            provenance_status="validated",
            source_execution_id="source-execution",
            prefix_telemetry_complete=True,
            provenance_finalized=True,
        )
        wrapper = _wrapper(
            source,
            provenance_status="pending",
            source_execution_id="different-execution",
            prefix_telemetry_complete=False,
            provenance_finalized=False,
        )
        # PrefixReplaySnapshot.__post_init__ mirrors legacy source fields for
        # compatibility.  Model a later wrapper update explicitly so the
        # outer execution id is genuinely different and still unfinalized.
        wrapper.provenance_finalized = False
        wrapper.prefix_telemetry_complete = False

        result = snapshot_provenance_record(wrapper)

        self.assertEqual("unverified", result["status"])
        self.assertFalse(result["provenance_finalized"])
        self.assertFalse(result["snapshot_ancestor_validated"])
        self.assertIn(
            "snapshot_provenance_execution_id_mismatch",
            result["reasons"],
        )

    def test_environment_only_diagnostic_without_finalization_is_unverified(self):
        source = SimpleNamespace(
            provenance_status="diagnostic",
            provenance_reasons=("interrupt_delivery",),
            source_execution_id="environment-execution",
            prefix_telemetry_complete=True,
            provenance_finalized=False,
        )

        result = snapshot_provenance_record(_wrapper(source))

        self.assertEqual("unverified", result["status"])
        self.assertFalse(result["snapshot_ancestor_validated"])
        self.assertIn("snapshot_provenance_not_finalized", result["reasons"])

    def test_emulator_restore_uses_same_effective_provenance(self):
        source = SimpleNamespace(
            provenance_status="pending",
            source_execution_id="execution-3",
            prefix_telemetry_complete=False,
            provenance_finalized=False,
        )
        wrapper = _wrapper(source)
        source.provenance_status = "validated"
        source.prefix_telemetry_complete = True
        source.provenance_finalized = True

        emulator = object.__new__(IntelligentEmulator)
        emulator._set_active_prefix_provenance(wrapper)
        active = emulator._active_prefix_provenance
        self.assertEqual("validated", active["status"])
        self.assertTrue(active["telemetry_complete"])
        self.assertTrue(active["provenance_finalized"])
        self.assertEqual("execution-3", active["execution_id"])
        self.assertEqual((), active["reasons"])

    def test_continuation_keeps_intervention_from_previous_run(self):
        emulator = object.__new__(IntelligentEmulator)
        emulator._active_prefix_provenance = {}

        first = {
            "execution_id": "run-1",
            "execution_failed": False,
            "preflight_failed": False,
            "execution_telemetry_complete": True,
            "execution_intervention_reasons": ["function_summary_or_skip"],
            "execution_provenance": {
                "status": "diagnostic",
                "reasons": ["function_summary_or_skip"],
            },
        }
        emulator._advance_active_prefix_provenance_from_run(first)
        self.assertEqual("diagnostic", emulator._active_prefix_provenance["status"])
        self.assertIn(
            "function_summary_or_skip",
            emulator._active_prefix_provenance["reasons"],
        )

        second = {
            "execution_id": "run-2",
            "execution_failed": False,
            "preflight_failed": False,
            "execution_telemetry_complete": True,
            "execution_intervention_reasons": [],
            "execution_provenance": {
                "status": "validated",
                "reasons": [],
            },
        }
        emulator._advance_active_prefix_provenance_from_run(second)
        self.assertEqual("diagnostic", emulator._active_prefix_provenance["status"])
        self.assertIn(
            "function_summary_or_skip",
            emulator._active_prefix_provenance["prefix_intervention_reasons"],
        )

    def test_cleanup_invalidation_is_idempotent_and_preserves_observation(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("cleanup-execution")
        manager.snapshots[source.address] = source

        first = manager.invalidate_snapshot_provenance(
            "cleanup-execution",
            reason="replay_cleanup_exception",
        )
        second = manager.invalidate_snapshot_provenance(
            "cleanup-execution",
            reason="replay_cleanup_exception",
        )

        self.assertEqual(1, first)
        self.assertEqual(0, second)
        self.assertEqual(
            1,
            manager.capture_stats["provenance_invalidated"],
        )
        self.assertTrue(source.provenance_invalidated)
        self.assertEqual("diagnostic", source.provenance_status)
        self.assertTrue(source.provenance_finalized)
        self.assertFalse(source.prefix_telemetry_complete)

    def test_cleanup_helper_does_not_use_an_unrelated_old_execution_id(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("other-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="",
        )

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            reason="replay_cleanup_exception",
            error="native close failed",
        )

        self.assertEqual("", result["execution_id"])
        self.assertEqual(0, result["invalidated_snapshots"])
        self.assertFalse(source.provenance_invalidated)
        self.assertEqual(
            "snapshot_provenance_execution_id_missing",
            result["error"],
        )

    def test_cleanup_does_not_use_stale_id_after_unstarted_attempt(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("old-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        runner._dispose_temp_emulator = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("native close failed")
        )
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
            _lsgemu_cleanup_errors=[],
        )
        run_result = {
            "execution_attempted": True,
            "execution_started": False,
            "execution_failed": True,
            "execution_error": "preflight failed",
        }
        record = {
            "execution_id": "logical-unstarted-attempt",
        }

        result = runner._safe_dispose_temp_emulator(
            emulator,
            evidence_record=record,
            run_result=run_result,
        )

        self.assertIn("native close failed", result)
        self.assertFalse(source.provenance_invalidated)
        self.assertEqual(
            "snapshot_provenance_execution_id_missing",
            run_result.get("provenance_invalidation_error"),
        )

    def test_cleanup_prefers_runtime_id_over_logical_child_id(self):
        manager = BranchSnapshotManager()
        runtime_id = "emu-current-7"
        source = self._snapshot(runtime_id)
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        runner._dispose_temp_emulator = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("native close failed")
        )
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id=runtime_id,
            _execution_active=False,
            _lsgemu_cleanup_errors=[],
        )
        run_result = {
            "execution_id": runtime_id,
            "stop_reason": "bounded_completion",
            "instruction_count": 1,
            "execution_attempted": True,
            "execution_started": True,
            "execution_completed_normally": True,
            "execution_failed": False,
            "execution_telemetry_complete": True,
            "execution_intervention_reasons_authoritative": True,
            "execution_intervention_reasons": [],
            "execution_intervention_counts": {},
        }
        record = build_execution_evidence_record(
            phase_name="logical-child",
            run_result=run_result,
            covered_bbs={source.address},
            execution_id="logical-child-attempt-7",
        )

        runner._safe_dispose_temp_emulator(
            emulator,
            evidence_record=record,
            run_result=run_result,
        )

        self.assertTrue(source.provenance_invalidated)
        self.assertEqual(
            runtime_id,
            runner._invalidate_temp_emulator_snapshot_provenance(
                emulator,
                evidence_record=record,
                run_result=run_result,
            )["execution_id"],
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])

    def test_empty_baseline_does_not_use_stale_emulator_id(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("old-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
        )

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            execution_baseline={},
            run_result={
                "execution_attempted": True,
                "execution_started": False,
            },
            reason="replay_cleanup_exception",
            error="setup failed",
        )

        self.assertEqual("", result["execution_id"])
        self.assertEqual(
            "snapshot_provenance_execution_id_missing",
            result["error"],
        )
        self.assertFalse(source.provenance_invalidated)

    def test_current_run_result_id_wins_over_stale_emulator_id(self):
        manager = BranchSnapshotManager()
        old_source = self._snapshot("old-execution")
        current_source = self._snapshot("current-execution")
        manager.snapshots[old_source.address] = old_source
        manager.snapshot_history[current_source.address] = [current_source]
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
        )

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            run_result={
                "execution_id": "current-execution",
                "execution_attempted": True,
                "execution_started": True,
            },
            reason="replay_cleanup_exception",
            error="cleanup failed",
        )

        self.assertEqual("current-execution", result["execution_id"])
        self.assertEqual("run_result", result["execution_id_source"])
        self.assertFalse(old_source.provenance_invalidated)
        self.assertTrue(current_source.provenance_invalidated)

    def test_unmatched_current_id_never_falls_back_to_old_catalog_id(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("old-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
        )

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            run_result={
                "execution_id": "current-execution",
                "execution_attempted": True,
                "execution_started": True,
            },
            reason="replay_cleanup_exception",
            error="cleanup failed",
        )

        self.assertEqual("", result["execution_id"])
        self.assertEqual(0, result["invalidated_snapshots"])
        self.assertEqual(
            "snapshot_provenance_execution_id_unmatched",
            result["error"],
        )
        self.assertFalse(source.provenance_invalidated)

    def test_started_evidence_id_is_used_only_when_it_matches_catalog(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("current-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
        )
        record = {
            "execution_id": "current-execution",
            "execution_started": True,
            "execution_attempted": True,
        }

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            evidence_record=record,
            reason="replay_cleanup_exception",
        )

        self.assertEqual("current-execution", result["execution_id"])
        self.assertEqual("evidence_record", result["execution_id_source"])
        self.assertTrue(source.provenance_invalidated)

    def test_conflicting_started_evidence_does_not_fall_back_to_stale_direct_id(self):
        manager = BranchSnapshotManager()
        source = self._snapshot("old-execution")
        manager.snapshots[source.address] = source
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            branch_snapshot_manager=manager,
            execution_id="old-execution",
            _execution_active=False,
            _direct_execution_baseline={"execution_id": "old-execution"},
        )
        record = {
            "execution_id": "logical-current-attempt",
            "execution_started": True,
            "execution_attempted": True,
        }

        result = runner._invalidate_temp_emulator_snapshot_provenance(
            emulator,
            evidence_record=record,
            reason="replay_cleanup_exception",
        )

        self.assertEqual("", result["execution_id"])
        self.assertEqual(0, result["invalidated_snapshots"])
        self.assertEqual(
            "snapshot_provenance_execution_id_unmatched",
            result["error"],
        )
        self.assertFalse(source.provenance_invalidated)

    def test_invalidated_observation_remains_diagnostic_coverage(self):
        run = {
            "stop_reason": "bounded_completion",
            "instruction_count": 4,
            "execution_started": True,
            "execution_completed_normally": True,
            "execution_failed": False,
            "execution_telemetry_complete": True,
            "execution_provenance": {
                "status": "validated",
                "telemetry_complete": True,
                "provenance_finalized": True,
                "provenance_invalidated": True,
                "provenance_invalidation_reasons": [
                    "replay_cleanup_exception"
                ],
            },
        }

        record = build_execution_evidence_record(
            phase_name="cleanup",
            run_result=run,
            covered_bbs={0x1000},
        )

        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])
        self.assertEqual([0x1000], record["covered_bbs"])
        self.assertTrue(record["provenance_invalidated"])
        self.assertFalse(compact_record_is_verified_validated(record))

    def test_direct_finalization_failure_is_in_same_record(self):
        class FakeEmulator:
            instruction_count = 3
            bb_addr_set = {0x1000}
            forced_branch_trace = []
            intervention_count = 0
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

            @staticmethod
            def _execution_counter_snapshot():
                return {}

        emulator = FakeEmulator()
        record = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct-finalization",
            emulator=emulator,
            covered_bbs={0x1000},
            execution_started=True,
            execution_error="native execution failed",
            finalization_result={
                "execution_id": "direct-finalization-1",
                "finalized_snapshots": 0,
                "finalization_error": "snapshot manager failed",
                "finalization_errors": ["snapshot manager failed"],
            },
        )

        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])
        self.assertEqual(
            "snapshot manager failed",
            record["direct_finalization_error"],
        )
        self.assertIn("native execution failed", record["execution_error"])
        self.assertIn("snapshot manager failed", record["execution_error"])
        self.assertIn(
            "direct_execution_finalization_failed",
            record["execution_failure_reasons"],
        )

    def test_finalization_invalidation_fields_survive_compaction(self):
        class FakeEmulator:
            instruction_count = 1
            bb_addr_set = {0x1000}
            forced_branch_trace = []
            intervention_count = 0
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

            @staticmethod
            def _execution_counter_snapshot():
                return {}

        record = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct-finalization-compaction",
            emulator=FakeEmulator(),
            covered_bbs={0x1000},
            finalization_result={
                "execution_id": "direct-compaction-1",
                "finalized_snapshots": 0,
                "finalization_error": "snapshot manager failed",
                "finalization_errors": ["snapshot manager failed"],
                "provenance_invalidated": True,
                "provenance_invalidation_reasons": [
                    "direct_execution_finalization_exception"
                ],
                "invalidated_snapshots": 3,
            },
        )
        compact = build_execution_evidence_record(
            phase_name="direct-finalization-compaction",
            run_result=record,
            covered_bbs=record["covered_bbs"],
            execution_id=record["execution_id"],
        )

        self.assertTrue(compact["provenance_invalidated"])
        self.assertEqual(
            ["direct_execution_finalization_exception"],
            compact["provenance_invalidation_reasons"],
        )
        self.assertEqual(
            3,
            compact["snapshot_provenance_invalidated_snapshots"],
        )
        self.assertEqual(
            3,
            compact["direct_execution_finalization"]["invalidated_snapshots"],
        )


if __name__ == "__main__":
    unittest.main()
