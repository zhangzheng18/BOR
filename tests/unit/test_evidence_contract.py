from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lsgemu.evidence_contract import (
    DIAGNOSTIC_REPLAY,
    EVIDENCE_SCHEMA,
    UNCLASSIFIED_REPLAY,
    VALIDATED_REPLAY,
    build_execution_evidence_record,
    classify_phase,
    compact_record_is_verified_validated,
    environment_input_facts,
    environment_model_diagnostics,
    execution_intervention_counts_from_emulator,
    execution_intervention_reasons,
    is_environment_fact_reason,
    summarize_child_execution,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.scheduler.outcome_feedback import OutcomeFeedback
from lsgemu.path_naturalization import (
    EVIDENCE_E0,
    EVIDENCE_E1,
    EVIDENCE_E2,
    EVIDENCE_E3,
    evaluate_force_free_replay,
)
from lsgemu.runner_models import BranchConstraintCandidate


def completed_run(stop_reason="completed", *, interventions=None, **extra):
    """Return the minimum complete execution record used by test adapters."""
    intervention_counts = dict(interventions or {})
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
        "execution_intervention_reasons": list(intervention_counts),
        "execution_intervention_counts": intervention_counts,
        "initial_state_fingerprint_summary": {
            "fingerprint_sha256": "test-initial-state",
        },
        **extra,
    }


class EvidenceContractTests(unittest.TestCase):
    @staticmethod
    def _evidence_runner(*bbs):
        runner = object.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set=set(bbs))
        runner.global_coverage = set(bbs)
        runner.coverage_by_evidence = {}
        runner.phase_execution_records = []
        runner.execution_evidence_validated_bbs = set()
        runner.execution_evidence_diagnostic_bbs = set()
        runner.execution_evidence_unclassified_bbs = set()
        runner.execution_evidence_validated_refcounts = {}
        runner.execution_evidence_diagnostic_refcounts = {}
        runner.execution_evidence_unclassified_refcounts = {}
        runner.execution_evidence_legacy_bbs = {}
        runner.execution_evidence_ledger_initialized = False
        runner.execution_evidence_records_dropped = 0
        return runner

    def test_replay_postprocess_failure_downgrades_existing_witness(self):
        runner = object.__new__(HistoricalRunner)
        run_result = completed_run("completed")
        record = build_execution_evidence_record(
            phase_name="postprocess",
            run_result=run_result,
            covered_bbs={0x1000},
            execution_id="postprocess-1",
        )
        self.assertEqual(VALIDATED_REPLAY, record["status"])

        value = runner._safe_replay_postprocess(
            "branch_catalog",
            lambda: (_ for _ in ()).throw(RuntimeError("catalog failed")),
            evidence_record=record,
            run_result=run_result,
            default="continue-child",
        )

        self.assertEqual("continue-child", value)
        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])
        self.assertIn(
            "replay_postprocess_exception:branch_catalog",
            record["reasons"],
        )
        self.assertTrue(run_result["execution_failed"])
        self.assertFalse(run_result["execution_telemetry_complete"])

    def test_direct_counter_adapter_uses_per_attempt_delta(self):
        class FakeEmulator:
            def __init__(self):
                self.intervention_count = 2
                self.forced_branch_trace = [{"pc": 1}]
                self.runtime_loop_branch_force_stats = {"applied": 1}
                self.skip_function_stats = {"applied": 0}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}

            def _execution_counter_snapshot(self):
                return {
                    "intervention_count": self.intervention_count,
                    "forced_branch_trace_count": len(self.forced_branch_trace),
                    "runtime_loop_branch_force_stats": dict(
                        self.runtime_loop_branch_force_stats
                    ),
                    "skip_function_stats": dict(self.skip_function_stats),
                    "lzo_decompress_summary_stats": {},
                    "external_rom_call_stats": {},
                    "svc_stats": {},
                    "thumb_state_guard_stats": {},
                    "invalid_thumb_recovery_stats": {},
                    "thumb_indirect_branch_repair_stats": {},
                    "execution_preflight_stats": {},
                }

            @staticmethod
            def _execution_counter_delta(current, baseline):
                result = {}
                for key, value in current.items():
                    old = baseline.get(key, 0)
                    if isinstance(value, dict):
                        old_map = old if isinstance(old, dict) else {}
                        result[key] = {
                            nested_key: int(nested_value or 0)
                            - int(old_map.get(nested_key, 0) or 0)
                            for nested_key, nested_value in value.items()
                            if int(nested_value or 0)
                            > int(old_map.get(nested_key, 0) or 0)
                        }
                    else:
                        result[key] = max(0, int(value or 0) - int(old or 0))
                return result

        emulator = FakeEmulator()
        baseline = {
            "counter_snapshot": emulator._execution_counter_snapshot(),
            "forced_trace_len": len(emulator.forced_branch_trace),
        }
        emulator.intervention_count += 1
        emulator.runtime_loop_branch_force_stats["applied"] += 2
        emulator.forced_branch_trace.append({"pc": 2})
        self.assertEqual(
            {
                "loop_intervention": 1,
                "runtime_loop_branch_force": 2,
                "forced_branch": 1,
            },
            execution_intervention_counts_from_emulator(emulator, baseline),
        )

    def test_direct_evidence_adapter_without_run_result_is_materialized(self):
        class FakeEmulator:
            intervention_count = 0
            forced_branch_trace = []
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}
            instruction_count = 7
            bb_addr_set = {0x1000}

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        emulator = FakeEmulator()
        record = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct_test",
            emulator=emulator,
            covered_bbs={0x1000},
            execution_started=True,
        )
        self.assertEqual(VALIDATED_REPLAY, record["status"])
        self.assertTrue(record["execution_facts_available"])
        self.assertTrue(record["telemetry_complete"])
        self.assertEqual([0x1000], record["covered_bbs"])

    def test_direct_evidence_reports_peripheral_stats_as_per_attempt_delta(self):
        class FakeMMIO:
            def __init__(self):
                self.peripheral_input_stats = {
                    "rx_ready_status_reads": 0,
                    "rx_data_reads": 0,
                    "rx_bytes_consumed": 0,
                }

        class FakeEmulator:
            def __init__(self):
                self.mmio_handler = FakeMMIO()
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.instruction_count = 0
                self.bb_addr_set = set()

            def _execution_counter_snapshot(self):
                return {
                    "intervention_count": self.intervention_count,
                    "forced_branch_trace_count": len(self.forced_branch_trace),
                    "runtime_loop_branch_force_stats": dict(
                        self.runtime_loop_branch_force_stats
                    ),
                    "skip_function_stats": dict(self.skip_function_stats),
                    "lzo_decompress_summary_stats": {},
                    "external_rom_call_stats": {},
                    "svc_stats": {},
                    "thumb_state_guard_stats": {},
                    "invalid_thumb_recovery_stats": {},
                    "thumb_indirect_branch_repair_stats": {},
                    "execution_preflight_stats": {},
                    "environment_input_delivery_stats": {},
                    "peripheral_input_stats": dict(
                        self.mmio_handler.peripheral_input_stats
                    ),
                }

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        emulator = FakeEmulator()
        baseline = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_rx_delta",
        )
        emulator.mmio_handler.peripheral_input_stats["rx_ready_status_reads"] = 1
        emulator.mmio_handler.peripheral_input_stats["rx_data_reads"] = 1
        emulator.mmio_handler.peripheral_input_stats["rx_bytes_consumed"] = 1
        first = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct_rx_delta",
            emulator=emulator,
            covered_bbs={0x1000},
            execution_baseline=baseline,
        )
        self.assertEqual(1, first["peripheral_input_stats"]["rx_data_reads"])

        second_baseline = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_rx_delta",
        )
        emulator.mmio_handler.peripheral_input_stats["rx_ready_status_reads"] += 2
        emulator.mmio_handler.peripheral_input_stats["rx_data_reads"] += 2
        emulator.mmio_handler.peripheral_input_stats["rx_bytes_consumed"] += 2
        second = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct_rx_delta",
            emulator=emulator,
            covered_bbs={0x2000},
            execution_baseline=second_baseline,
        )
        self.assertEqual(2, second["peripheral_input_stats"]["rx_data_reads"])
        self.assertEqual(2, second["peripheral_input_stats"]["rx_bytes_consumed"])

    def test_direct_baseline_assigns_a_fresh_execution_id(self):
        class FakeEmulator:
            uc = object()
            instruction_count = 0
            bb_addr_set = set()
            forced_branch_trace = []

            def _execution_counter_snapshot(self):
                return {}

        emulator = FakeEmulator()
        first = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_id_test",
        )
        second = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_id_test",
        )

        self.assertTrue(first["execution_id"])
        self.assertTrue(second["execution_id"])
        self.assertNotEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(second["execution_id"], emulator.execution_id)
        self.assertEqual(
            second["counter_snapshot"],
            emulator._execution_counter_baseline,
        )
        self.assertEqual(0, emulator._forced_trace_baseline_len)
        self.assertEqual(0, emulator._stream_summary_event_baseline_len)
        self.assertEqual(0, emulator._stream_payload_write_baseline_len)

    def test_direct_stats_prefer_registered_primary_handler(self):
        class Handler:
            def __init__(self, reads):
                self.peripheral_input_stats = {
                    "rx_data_reads": reads,
                    "rx_bytes_consumed": reads,
                }

        class FakeEmulator:
            def __init__(self, primary):
                self.uc = object()
                self.primary = primary
                self.mmio_handler = Handler(100)
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.instruction_count = 0
                self.bb_addr_set = set()

            def _execution_counter_snapshot(self):
                return {
                    "peripheral_input_stats": dict(
                        self.primary.peripheral_input_stats
                    ),
                    "environment_input_delivery_stats": {},
                }

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        primary = Handler(0)
        explicit = Handler(200)
        emulator = FakeEmulator(primary)
        with patch(
            "lsgemu.historical_runner.get_primary_mmio_handler",
            return_value=primary,
        ):
            baseline = HistoricalRunner._capture_direct_execution_baseline(
                emulator,
                "direct_primary_stats",
                mmio_handler=explicit,
            )
            primary.peripheral_input_stats["rx_data_reads"] = 2
            primary.peripheral_input_stats["rx_bytes_consumed"] = 2
            explicit.peripheral_input_stats["rx_data_reads"] = 202
            explicit.peripheral_input_stats["rx_bytes_consumed"] = 202
            record = HistoricalRunner._direct_execution_evidence_record(
                phase_name="direct_primary_stats",
                emulator=emulator,
                covered_bbs=(),
                execution_baseline=baseline,
                mmio_handler=explicit,
            )

        self.assertEqual("primary_mmio_handler", record["peripheral_input_stats_source"])
        self.assertEqual(2, record["peripheral_input_stats"]["rx_data_reads"])
        self.assertEqual(2, record["peripheral_input_stats"]["rx_bytes_consumed"])

    def test_direct_stats_do_not_fallback_to_overlay_when_primary_has_no_delta(self):
        class Handler:
            def __init__(self, reads):
                self.peripheral_input_stats = {
                    "rx_data_reads": reads,
                    "rx_bytes_consumed": reads,
                }

        class FakeEmulator:
            def __init__(self, primary):
                self.uc = object()
                self.primary = primary
                self.mmio_handler = Handler(100)
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.instruction_count = 0
                self.bb_addr_set = set()

            def _execution_counter_snapshot(self):
                return {
                    "peripheral_input_stats": dict(
                        self.primary.peripheral_input_stats
                    ),
                    "environment_input_delivery_stats": {},
                }

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        primary = Handler(7)
        explicit = Handler(200)
        emulator = FakeEmulator(primary)
        with patch(
            "lsgemu.historical_runner.get_primary_mmio_handler",
            return_value=primary,
        ):
            baseline = HistoricalRunner._capture_direct_execution_baseline(
                emulator,
                "direct_primary_no_delta",
                mmio_handler=explicit,
            )
            # The explicit overlay observes a value, but the primary handler
            # is the owner that actually executed the read and has no delta.
            explicit.peripheral_input_stats["rx_data_reads"] = 201
            explicit.peripheral_input_stats["rx_bytes_consumed"] = 201
            record = HistoricalRunner._direct_execution_evidence_record(
                phase_name="direct_primary_no_delta",
                emulator=emulator,
                covered_bbs=(),
                execution_baseline=baseline,
                mmio_handler=explicit,
            )

        self.assertEqual("primary_mmio_handler", record["peripheral_input_stats_source"])
        self.assertEqual({}, record["peripheral_input_stats"])
        self.assertNotIn("direct_peripheral_input_stats", record)

    def test_direct_stats_ignore_stale_counter_owner_when_primary_exists(self):
        class Handler:
            def __init__(self, reads):
                self.peripheral_input_stats = {
                    "rx_data_reads": reads,
                    "rx_bytes_consumed": reads,
                }

        class FakeEmulator:
            def __init__(self, primary, secondary):
                self.uc = object()
                self.primary = primary
                self.mmio_handler = secondary
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.instruction_count = 0
                self.bb_addr_set = set()

            def _execution_counter_snapshot(self):
                # Simulate an older adapter whose compact snapshot still reads
                # a stale secondary owner.  The registry-selected primary is
                # the only authoritative source for this direct attempt.
                return {
                    "peripheral_input_stats": dict(
                        self.mmio_handler.peripheral_input_stats
                    ),
                    "environment_input_delivery_stats": {},
                }

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        primary = Handler(5)
        secondary = Handler(100)
        emulator = FakeEmulator(primary, secondary)
        with patch(
            "lsgemu.historical_runner.get_primary_mmio_handler",
            return_value=primary,
        ):
            baseline = HistoricalRunner._capture_direct_execution_baseline(
                emulator,
                "direct_primary_owner",
                mmio_handler=secondary,
            )
            # Only the non-owning handler changes after the baseline.
            secondary.peripheral_input_stats["rx_data_reads"] = 200
            secondary.peripheral_input_stats["rx_bytes_consumed"] = 200
            record = HistoricalRunner._direct_execution_evidence_record(
                phase_name="direct_primary_owner",
                emulator=emulator,
                covered_bbs=(),
                execution_baseline=baseline,
                mmio_handler=secondary,
            )

        self.assertEqual("primary_mmio_handler", record["peripheral_input_stats_source"])
        self.assertEqual({}, record["peripheral_input_stats"])

    def test_direct_adapter_preserves_per_run_stats_without_baseline(self):
        class Handler:
            peripheral_input_stats = {
                "rx_data_reads": 99,
                "rx_bytes_consumed": 99,
            }

        class FakeEmulator:
            uc = object()
            mmio_handler = Handler()
            intervention_count = 0
            forced_branch_trace = []
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}
            instruction_count = 1
            bb_addr_set = set()

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        emulator = FakeEmulator()
        record = HistoricalRunner._direct_execution_evidence_record(
            phase_name="legacy_direct_adapter",
            emulator=emulator,
            covered_bbs=(),
            execution_started=True,
            run_result={
                "stop_reason": "completed",
                "execution_started": True,
                "execution_completed_normally": True,
                "execution_failed": False,
                "execution_telemetry_complete": True,
                "execution_intervention_reasons_authoritative": True,
                "peripheral_input_stats": {"rx_data_reads": 2},
                "peripheral_input_stats_source": "run_result_delta",
            },
        )
        self.assertEqual("run_result_delta", record["peripheral_input_stats_source"])
        self.assertEqual(2, record["peripheral_input_stats"]["rx_data_reads"])

    def test_direct_execution_finalizes_pending_provenance_on_success(self):
        class SnapshotManager:
            def __init__(self, emulator):
                self.emulator = emulator
                self.calls = []

            def finalize_snapshot_provenance(self, *args, **kwargs):
                self.calls.append((args, kwargs, self.emulator._execution_active))
                return 3

        class FakeEmulator:
            def __init__(self):
                self.uc = object()
                self.execution_id = "old"
                self._execution_active = True
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.branch_snapshot_manager = SnapshotManager(self)

            def _execution_counter_snapshot(self):
                return {
                    "intervention_count": self.intervention_count,
                    "forced_branch_trace_count": len(self.forced_branch_trace),
                }

        emulator = FakeEmulator()
        baseline = {
            "execution_id": "direct-success",
            "counter_snapshot": emulator._execution_counter_snapshot(),
            "forced_trace_len": 0,
        }
        result = HistoricalRunner._finalize_direct_execution(
            emulator,
            baseline,
            execution_started=True,
        )

        self.assertFalse(emulator._execution_active)
        self.assertEqual(3, result["finalized_snapshots"])
        self.assertEqual(1, len(emulator.branch_snapshot_manager.calls))
        args, kwargs, active_at_finalize = emulator.branch_snapshot_manager.calls[0]
        self.assertEqual(("direct-success",), args)
        self.assertTrue(kwargs["successful"])
        self.assertTrue(kwargs["telemetry_complete"])
        self.assertFalse(active_at_finalize)

    def test_direct_execution_finalizes_as_failed_after_native_exception(self):
        class SnapshotManager:
            def __init__(self):
                self.kwargs = None

            def finalize_snapshot_provenance(self, _execution_id, **kwargs):
                self.kwargs = kwargs
                return 1

        class FakeEmulator:
            execution_id = "direct-failure"
            _execution_active = True
            intervention_count = 0
            forced_branch_trace = []
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}

            def __init__(self):
                self.uc = object()
                self.branch_snapshot_manager = SnapshotManager()

            def _execution_counter_snapshot(self):
                return {"intervention_count": 0, "forced_branch_trace_count": 0}

        emulator = FakeEmulator()
        baseline = {
            "execution_id": "direct-failure",
            "counter_snapshot": emulator._execution_counter_snapshot(),
            "forced_trace_len": 0,
        }
        HistoricalRunner._finalize_direct_execution(
            emulator,
            baseline,
            execution_started=True,
            execution_error="native engine error",
        )

        self.assertFalse(emulator._execution_active)
        self.assertFalse(emulator.branch_snapshot_manager.kwargs["successful"])
        self.assertFalse(
            emulator.branch_snapshot_manager.kwargs["telemetry_complete"]
        )

    def test_direct_baseline_recapture_excludes_snapshot_prefix_telemetry(self):
        class FakeMMIO:
            def __init__(self):
                self.peripheral_input_stats = {
                    "rx_ready_status_reads": 0,
                    "rx_data_reads": 0,
                    "rx_bytes_consumed": 0,
                }

        class FakeEmulator:
            def __init__(self):
                self.uc = object()
                self.mmio_handler = FakeMMIO()
                self.intervention_count = 0
                self.forced_branch_trace = []
                self.runtime_loop_branch_force_stats = {}
                self.skip_function_stats = {}
                self.lzo_decompress_summary_stats = {}
                self.external_rom_call_stats = {}
                self.svc_stats = {}
                self.thumb_state_guard_stats = {}
                self.invalid_thumb_recovery_stats = {}
                self.thumb_indirect_branch_repair_stats = {}
                self.execution_preflight_stats = {}
                self.instruction_count = 0
                self.bb_addr_set = set()

            def _execution_counter_snapshot(self):
                return {
                    "peripheral_input_stats": dict(
                        self.mmio_handler.peripheral_input_stats
                    ),
                    "environment_input_delivery_stats": {},
                }

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

        emulator = FakeEmulator()
        pre_restore = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_snapshot_prefix",
        )
        # This is the state restored from the saved execution prefix.
        emulator.mmio_handler.peripheral_input_stats["rx_ready_status_reads"] = 11
        emulator.mmio_handler.peripheral_input_stats["rx_data_reads"] = 9
        emulator.mmio_handler.peripheral_input_stats["rx_bytes_consumed"] = 9

        # The production callers recapture here, after restore and before the
        # native attempt.  The old pre-restore baseline must not be reused.
        post_restore = HistoricalRunner._capture_direct_execution_baseline(
            emulator,
            "direct_snapshot_prefix",
        )
        self.assertEqual(
            9,
            post_restore["direct_peripheral_input_stats"]["rx_data_reads"],
        )
        emulator.mmio_handler.peripheral_input_stats["rx_ready_status_reads"] += 1
        emulator.mmio_handler.peripheral_input_stats["rx_data_reads"] += 1
        emulator.mmio_handler.peripheral_input_stats["rx_bytes_consumed"] += 1
        record = HistoricalRunner._direct_execution_evidence_record(
            phase_name="direct_snapshot_prefix",
            emulator=emulator,
            covered_bbs=(),
            execution_baseline=post_restore,
            run_result={
                "execution_started": True,
                "execution_completed_normally": True,
                "execution_failed": False,
                "execution_telemetry_complete": True,
                "execution_intervention_reasons_authoritative": True,
                # The adapter must replace this with the exact delta from
                # the restored-state baseline, rather than trusting a
                # cumulative value supplied by an older runner adapter.
                "peripheral_input_stats": {"rx_data_reads": 99},
            },
        )

        self.assertEqual(1, record["peripheral_input_stats"]["rx_data_reads"])
        self.assertEqual(1, record["peripheral_input_stats"]["rx_bytes_consumed"])
        self.assertNotEqual(
            9,
            record["peripheral_input_stats"]["rx_data_reads"],
        )
        self.assertNotEqual(
            pre_restore["execution_id"],
            post_restore["execution_id"],
        )

    def _rtos_probe_runner(self, emulator):
        runner = object.__new__(HistoricalRunner)
        runner.rtos_thread_call_arg_cache = {}
        runner.probe_diagnostics = []
        runner.prepared = SimpleNamespace(
            branch_instruction_by_bb={0x1000: {"address": 0x1004}},
            static_bbs={0x1000: [0x1000, 0x1002]},
        )
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = None
        runner._effective_llm_config_path = lambda: None
        runner._temp_stage_for = lambda purpose: purpose
        runner._snapshot_entry_identity = lambda _entry: ("snapshot", 1)
        runner._scope_signature_from_snapshot_entry = lambda _entry: ()
        runner._configure_emulator_feedback = lambda _emulator: None
        runner._restore_branch_snapshot = lambda *_args, **_kwargs: True
        runner._restore_replay_mmio_state = lambda *_args, **_kwargs: True
        runner._apply_scoped_branch_constraints = lambda *_args, **_kwargs: None
        runner._new_temp_emulator = lambda **_kwargs: emulator
        runner._new_temp_register_tracer = lambda _emulator: None
        runner._safe_dispose_temp_emulator = lambda *_args, **_kwargs: None

        @contextmanager
        def no_op_scope(*_args, **_kwargs):
            yield

        runner.guided_modules = no_op_scope
        return runner

    def test_rtos_probe_compatibility_retry_failure_is_diagnostic(self):
        class FakeEmulator:
            uc = object()

            def run(self, **kwargs):
                if "stop_before_pc" in kwargs:
                    raise TypeError("legacy adapter")
                raise RuntimeError("fallback execution failed")

        runner = self._rtos_probe_runner(FakeEmulator())
        entry = SimpleNamespace(snapshot=object(), mmio_state={})
        with patch(
            "lsgemu.historical_runner.EnhancedMMIOHandler",
            return_value=SimpleNamespace(),
        ):
            result = runner._thread_create_call_args_from_live_pre_call(
                entry,
                0x1000,
            )

        self.assertIsNone(result)
        self.assertEqual({}, runner.rtos_thread_call_arg_cache)
        self.assertTrue(
            any(
                item.get("compatibility_retry")
                and item.get("reason") == "probe_run_exception"
                for item in runner.probe_diagnostics
            )
        )

    def test_rtos_probe_failure_and_poisoned_state_are_not_cached(self):
        for poisoned, run_result in (
            (
                False,
                {
                    "execution_failed": True,
                    "execution_error": "native execution failed",
                    "stop_reason": "execution_exception",
                },
            ),
            (True, completed_run("bounded_completion")),
        ):
            class FakeEmulator:
                uc = object()
                replay_state_poisoned = poisoned

                def run(self, **_kwargs):
                    return run_result

            runner = self._rtos_probe_runner(FakeEmulator())
            entry = SimpleNamespace(snapshot=object(), mmio_state={})
            with patch(
                "lsgemu.historical_runner.EnhancedMMIOHandler",
                return_value=SimpleNamespace(),
            ):
                result = runner._thread_create_call_args_from_live_pre_call(
                    entry,
                    0x1000,
                )

            self.assertIsNone(result)
            self.assertEqual({}, runner.rtos_thread_call_arg_cache)
            expected_reason = (
                "probe_replay_state_poisoned"
                if poisoned
                else "probe_execution_failed"
            )
            self.assertTrue(
                any(item.get("reason") == expected_reason for item in runner.probe_diagnostics)
            )

    def test_contextual_isr_probe_failure_is_retained_as_bounded_diagnostic(self):
        runner = object.__new__(HistoricalRunner)
        runner.global_coverage = set()
        runner.dynamic_successors = {}
        runner.known_branch_root_snapshots = {}
        runner.reservoir_interrupt_contexts = []
        runner.probe_diagnostics = []
        runner._select_interrupt_contexts = lambda *_args, **_kwargs: []
        runner._temp_stage_for = lambda purpose: purpose

        def create_failure(**_kwargs):
            raise RuntimeError("context probe construction failed")

        runner._new_temp_emulator = create_failure
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner._record_phase = lambda _name, _covered, **metadata: metadata

        metadata = runner.run_contextual_isr_exploration(
            max_contexts=1,
            max_instructions=8,
            phase_name="contextual_isr_test",
        )

        self.assertIn("context probe construction failed", metadata["isr_probe_error"])
        self.assertEqual(1, len(metadata["probe_diagnostics"]))
        self.assertEqual(
            "contextual_isr_probe",
            metadata["probe_diagnostics"][0]["probe_kind"],
        )
        self.assertEqual(1, len(runner.probe_diagnostics))

    def test_compact_validated_record_rejects_late_failure_fields(self):
        record = build_execution_evidence_record(
            phase_name="test",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
        )
        self.assertTrue(compact_record_is_verified_validated(record))
        record["execution_error"] = "native failure after serialization"
        self.assertFalse(compact_record_is_verified_validated(record))

    def test_mmio_rx_facts_are_reported_separately_from_model_warnings(self):
        run = completed_run(
            "bounded_completion",
            peripheral_input_stats={
                "rx_ready_status_reads": 2,
                "rx_bytes_consumed": 2,
                "rx_data_reads_without_ready": 1,
                "rx_overlay_mismatches": 1,
            },
            stream_input_summary_events=[
                {"kind": "mmio_stream_status", "return_value": "ready"},
                {"kind": "mmio_stream_byte", "byte": 0x41},
            ],
        )
        facts = environment_input_facts(run)
        self.assertEqual(2, facts["input_ready"])
        self.assertEqual(2, facts["stream_input_delivery"])
        diagnostics = environment_model_diagnostics(run)
        self.assertEqual(1, diagnostics["rx_data_reads_without_ready"])
        self.assertEqual(1, diagnostics["rx_overlay_mismatches"])
        self.assertNotIn("rx_overlay_mismatches", facts)

    def test_context_replay_requires_verified_snapshot_lineage(self):
        run = completed_run("bounded_completion")
        metadata = {
            "context_snapshot_mode": True,
            "entry_derivation": "entry_derived_context_snapshot_event_replay",
        }
        record = build_execution_evidence_record(
            phase_name="contextual_isr",
            run_result=run,
            covered_bbs={0x2000},
            metadata=metadata,
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])
        self.assertIn("context_snapshot_ancestor_not_validated", record["reasons"])

    def test_context_replay_accepts_finalized_observed_lineage(self):
        run = completed_run(
            "bounded_completion",
            execution_provenance={
                "status": "validated",
                "reasons": [],
                "telemetry_complete": True,
                "provenance_finalized": True,
                "snapshot_ancestor_validated": True,
            },
        )
        record = build_execution_evidence_record(
            phase_name="contextual_isr",
            run_result=run,
            covered_bbs={0x2000},
            metadata={
                "context_snapshot_mode": True,
                "entry_derivation": "entry_derived_context_snapshot_event_replay",
            },
        )
        self.assertEqual(VALIDATED_REPLAY, record["status"])
        self.assertTrue(compact_record_is_verified_validated(record))

    def test_compact_record_rejects_prefix_intervention_after_status_copy(self):
        record = build_execution_evidence_record(
            phase_name="test",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
        )
        record["prefix_intervention_reasons"] = ["forced_branch"]
        self.assertFalse(compact_record_is_verified_validated(record))

    def test_cleanup_failure_cannot_leave_a_strict_replay_witness(self):
        runner = object.__new__(HistoricalRunner)

        def fail_dispose(*_args, **_kwargs):
            raise RuntimeError("native close failed")

        runner._dispose_temp_emulator = fail_dispose
        emulator = SimpleNamespace(_lsgemu_cleanup_errors=[])
        run_result = completed_run("completed")

        cleanup_error = runner._safe_dispose_temp_emulator(
            emulator,
            run_result=run_result,
            stage_name="reservoir_replay",
        )

        self.assertIn("dispose:RuntimeError:native close failed", cleanup_error)
        self.assertTrue(run_result["execution_failed"])
        self.assertFalse(run_result["execution_telemetry_complete"])
        self.assertTrue(HistoricalRunner._replay_result_has_failure(run_result))
        record = build_execution_evidence_record(
            phase_name="reservoir_replay",
            run_result=run_result,
            covered_bbs={0x1000, 0x2000},
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, record["status"])

    def test_healthy_replay_result_remains_eligible(self):
        self.assertFalse(
            HistoricalRunner._replay_result_has_failure(
                completed_run("timeout_or_max_instructions_reached")
            )
        )

    def test_child_summary_zero_expected_with_failure_is_incomplete(self):
        summary = summarize_child_execution(
            [
                {
                    "execution_attempted": True,
                    "execution_failed": True,
                    "execution_error": "constructor failed",
                    "stop_reason": "emulator_create_exception",
                }
            ],
            expected_count=0,
        )
        self.assertEqual(1, summary["child_completed_with_result_count"])
        self.assertFalse(summary["execution_telemetry_complete"])

    def test_runner_child_summary_rejects_failure_when_expected_is_zero(self):
        failure = HistoricalRunner._execution_failure_evidence_record(
            phase_name="child",
            execution_id="failed-1",
            reason="emulator_create_exception",
            execution_error="constructor failed",
        )
        summary = HistoricalRunner._child_execution_summary(
            0,
            {},
            expected_count=0,
            execution_records=[failure],
        )
        self.assertEqual(1, summary["child_materialized_record_count"])
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertGreaterEqual(summary["child_failure_count"], 1)

    def test_runner_child_summary_checks_extra_failure_records(self):
        success = build_execution_evidence_record(
            phase_name="child",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
            execution_id="success-1",
        )
        failure = HistoricalRunner._execution_failure_evidence_record(
            phase_name="child",
            execution_id="failed-1",
            reason="execution_exception",
            execution_error="runtime failed",
        )
        summary = HistoricalRunner._child_execution_summary(
            1,
            {},
            expected_count=1,
            execution_records=[success, failure],
        )
        self.assertEqual(2, summary["child_materialized_record_count"])
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertGreaterEqual(summary["child_failure_count"], 1)

    @staticmethod
    def _snapshot_branch_runner(new_emulator, *, dispose=None):
        """Build the smallest runner surface needed by snapshot replay tests."""
        runner = object.__new__(HistoricalRunner)
        snapshot = SimpleNamespace(
            address=0x1000,
            target=0x2000,
            fallthrough=0x3000,
        )
        branch = SimpleNamespace(
            start_addr=0x1000,
            successors={0x2000, 0x3000},
        )
        runner.emulator = SimpleNamespace(
            branch_snapshot_manager=SimpleNamespace(
                has_snapshot=lambda _address: True,
                get_snapshot=lambda _address: snapshot,
            )
        )
        runner.global_coverage = set()
        runner.max_snapshots = 3
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = None
        runner.conditional_branches = lambda: [branch]
        runner._effective_llm_config_path = lambda: None
        runner._temp_stage_for = lambda purpose: purpose
        runner.validate_coverage = lambda values: set(values)
        runner._new_temp_emulator = new_emulator
        runner._dispose_temp_emulator = dispose or (lambda *args, **kwargs: None)
        runner._record_phase = lambda _name, covered, **metadata: (
            setattr(runner, "_test_phase_metadata", metadata) or set(covered)
        )
        return runner

    def test_snapshot_branch_creation_failure_is_materialized(self):
        def create_failure(**_kwargs):
            raise RuntimeError("injected constructor failure")

        runner = self._snapshot_branch_runner(create_failure)
        covered = HistoricalRunner.run_snapshot_branch_exploration(runner)

        self.assertEqual(set(), covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertEqual(2, summary["child_failure_count"])
        self.assertEqual(2, summary["child_materialized_record_count"])
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertEqual(
            {"emulator_create_exception"},
            {
                reason
                for record in summary["child_execution_records"]
                for reason in record.get("reasons", [])
                if reason == "emulator_create_exception"
            },
        )

    def test_snapshot_branch_mmio_setup_failure_continues_to_next_direction(self):
        class FakeEmulator:
            uc = object()

            def set_forced_branch_choices(self, _choices):
                return None

        created = []

        def create_emulator(**_kwargs):
            emulator = FakeEmulator()
            created.append(emulator)
            return emulator

        disposed = []

        def dispose(emulator, **_kwargs):
            disposed.append(emulator)

        runner = self._snapshot_branch_runner(create_emulator, dispose=dispose)
        with patch(
            "lsgemu.historical_runner.EnhancedMMIOHandler",
            side_effect=RuntimeError("injected MMIO setup failure"),
        ):
            covered = HistoricalRunner.run_snapshot_branch_exploration(runner)

        self.assertEqual(set(), covered)
        self.assertEqual(2, len(created))
        self.assertEqual(created, disposed)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertEqual(2, summary["child_failure_count"])
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertTrue(
            any(
                "replay_setup_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    def test_snapshot_branch_cleanup_failure_is_diagnostic_and_nonfatal(self):
        class FakeEmulator:
            uc = object()

            def set_forced_branch_choices(self, _choices):
                return None

            def run(self, **_kwargs):
                return completed_run("bounded_completion")

        def create_emulator(**_kwargs):
            return FakeEmulator()

        runner = self._snapshot_branch_runner(
            create_emulator,
            dispose=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected cleanup failure")
            ),
        )

        @contextmanager
        def no_op_scope(*_args, **_kwargs):
            yield

        @contextmanager
        def trace_scope(_emulator):
            yield SimpleNamespace(covered_bbs={0x2000})

        runner.guided_modules = no_op_scope
        runner.capture_coverage = trace_scope
        runner.prepared = SimpleNamespace(entry_point=0x08000001)

        covered = HistoricalRunner.run_snapshot_branch_exploration(runner)

        self.assertEqual({0x2000}, covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertGreaterEqual(summary["child_failure_count"], 2)
        self.assertEqual([], summary["validated_bbs"])
        self.assertIn(0x2000, summary["diagnostic_bbs"])
        self.assertTrue(
            any(
                "replay_cleanup_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    def test_isr_reservoir_materializes_replay_creation_failure(self):
        class FakeMMIO:
            def __init__(self, *_args, **_kwargs):
                self.mmio_state = {}

            def mark_mmio_state_explicit(self):
                return None

            def start_hooking(self):
                return None

            def stop_hooking(self):
                return None

        class FakeExplorer:
            def __init__(self, *_args, **_kwargs):
                self.isr_addresses = {1: 0x1000}

            def learned_irq_candidates(self, limit=128):
                del limit
                return []

            def _setup_isr_context(self, *_args, **_kwargs):
                return None

            def start_isr_execution(self, *_args, **_kwargs):
                return True

            def get_statistics(self):
                return {}

        class FakeEmulator:
            uc = SimpleNamespace(reg_read=lambda _reg: 0)
            causal_context = None
            branch_snapshot_manager = SimpleNamespace(
                get_ordered_events=lambda: []
            )
            forced_branch_trace = []
            runtime_successor_edges = set()
            bb_addr_set = set()
            instruction_count = 0
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
                return {"pc": 0x1000}

        class FakeLease:
            def __init__(self):
                self.emulator = FakeEmulator()
                self.mmio_handler = None

            def bind_mmio(self, handler):
                self.mmio_handler = handler
                return handler

        runner = object.__new__(HistoricalRunner)
        runner.global_coverage = set()
        runner.max_snapshots = 3
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = None
        runner.reservoir_interrupt_contexts = []
        runner.branch_snapshot_hotset = set()
        runner.known_branch_root_snapshots = {}
        runner.dynamic_successors = {}
        runner.mmio_handler = SimpleNamespace(mmio_state={})
        runner.prepared = SimpleNamespace(
            static_bbs={},
            static_bb_set=set(),
            instruction_to_bb={},
        )
        runner._effective_successor_map = lambda: {}
        runner._vector_table_base = lambda: 0
        runner._code_address_ranges = lambda: []
        runner._effective_llm_config_path = lambda: None
        runner._temp_stage_for = lambda purpose: purpose
        runner._prioritized_isr_items = lambda items, **_kwargs: list(items)
        runner.validate_coverage = lambda values: set(values)
        runner._record_emulator_runtime_successors = lambda _emulator: 0
        runner._ordered_replay_snapshots = lambda _emulator: []
        runner._record_dynamic_successors = lambda _events: 0
        runner._remember_main_branch_events = lambda *_args, **_kwargs: 0
        runner._remember_branch_root_snapshots = lambda *_args, **_kwargs: 0
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner._record_coverage_evidence = lambda *_args, **_kwargs: None
        runner._record_phase = lambda _name, covered, **metadata: (
            setattr(runner, "_test_phase_metadata", metadata) or set(covered)
        )

        managed_calls = []

        @contextmanager
        def managed_temp(**_kwargs):
            managed_calls.append(True)
            if len(managed_calls) > 1:
                raise RuntimeError("injected ISR replay constructor failure")
            yield FakeLease()

        @contextmanager
        def no_op_scope(*_args, **_kwargs):
            yield

        @contextmanager
        def trace_scope(_emulator):
            yield set()

        runner._managed_temp_emulator = managed_temp
        runner.guided_modules = no_op_scope
        runner.capture_coverage = trace_scope

        with patch("lsgemu.historical_runner.EnhancedMMIOHandler", FakeMMIO), patch(
            "lsgemu.historical_runner.ISRExplorer", FakeExplorer
        ):
            covered = HistoricalRunner.run_isr_reservoir_exploration(
                runner,
                max_tasks=1,
                max_tasks_per_isr=1,
            )

        self.assertEqual(set(), covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertEqual(1, summary["child_failure_count"])
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertTrue(
            any(
                "emulator_create_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    def test_isr_reservoir_cleanup_failure_downgrades_child(self):
        class FakeMMIO:
            def __init__(self, *_args, **_kwargs):
                self.mmio_state = {}

            def mark_mmio_state_explicit(self):
                return None

            def start_hooking(self):
                return None

            def stop_hooking(self):
                return None

        class FakeExplorer:
            def __init__(self, *_args, **_kwargs):
                self.isr_addresses = {1: 0x1000}

            def learned_irq_candidates(self, limit=128):
                del limit
                return []

            def _setup_isr_context(self, *_args, **_kwargs):
                return None

            def start_isr_execution(self, *_args, **_kwargs):
                return True

            def get_statistics(self):
                return {}

        class FakeEmulator:
            causal_context = None
            forced_branch_trace = []
            runtime_successor_edges = set()
            bb_addr_set = set()
            instruction_count = 0
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

            def __init__(self):
                self.uc = SimpleNamespace(reg_read=lambda _reg: 0)
                self.branch_snapshot_manager = SimpleNamespace(
                    get_ordered_events=lambda: [],
                    get_ordered_occurrence_events=lambda: [],
                )

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

            @staticmethod
            def _execution_preflight(_pc):
                return True

        class FakeLease:
            def __init__(self):
                self.emulator = FakeEmulator()
                self.mmio_handler = None

            def bind_mmio(self, handler):
                self.mmio_handler = handler
                return handler

        runner = object.__new__(HistoricalRunner)
        runner.global_coverage = set()
        runner.max_snapshots = 3
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = None
        runner.reservoir_interrupt_contexts = []
        runner.known_branch_root_snapshots = {}
        runner.dynamic_successors = {}
        runner.mmio_handler = SimpleNamespace(mmio_state={})
        runner.prepared = SimpleNamespace(
            static_bbs={},
            static_bb_set=set(),
            instruction_to_bb={},
        )
        runner._effective_successor_map = lambda: {}
        runner._vector_table_base = lambda: 0
        runner._code_address_ranges = lambda: []
        runner._effective_llm_config_path = lambda: None
        runner._temp_stage_for = lambda purpose: purpose
        runner._prioritized_isr_items = lambda items, **_kwargs: list(items)
        runner.validate_coverage = lambda values: set(values)
        runner._record_emulator_runtime_successors = lambda _emulator: 0
        runner._ordered_replay_snapshots = lambda _emulator: []
        runner._record_dynamic_successors = lambda _events: 0
        runner._remember_main_branch_events = lambda *_args, **_kwargs: 0
        runner._remember_branch_root_snapshots = lambda *_args, **_kwargs: 0
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner._record_coverage_evidence = lambda *_args, **_kwargs: None
        runner._record_phase = lambda _name, covered, **metadata: (
            setattr(runner, "_test_phase_metadata", metadata) or set(covered)
        )

        manager_calls = []

        @contextmanager
        def managed_temp(**_kwargs):
            manager_calls.append(True)
            lease = FakeLease()
            yield lease
            if len(manager_calls) >= 2:
                # This is the failure mode of the real managed lease: its
                # finally block records cleanup errors on the owned emulator
                # instead of raising over the completed replay body.
                lease.emulator._lsgemu_cleanup_errors = [
                    "managed_cleanup:RuntimeError:injected ISR reservoir cleanup failure"
                ]

        @contextmanager
        def no_op_scope(*_args, **_kwargs):
            yield

        @contextmanager
        def trace_scope(_emulator):
            yield set()

        runner._managed_temp_emulator = managed_temp
        runner.guided_modules = no_op_scope
        runner.capture_coverage = trace_scope

        with patch("lsgemu.historical_runner.EnhancedMMIOHandler", FakeMMIO), patch(
            "lsgemu.historical_runner.ISRExplorer", FakeExplorer
        ):
            covered = HistoricalRunner.run_isr_reservoir_exploration(
                runner,
                max_tasks=1,
                max_tasks_per_isr=1,
            )

        self.assertEqual(set(), covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertEqual([], summary["validated_bbs"])
        self.assertTrue(
            any(
                "replay_cleanup_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    @staticmethod
    def _contextual_isr_failure_fixture(
        *,
        stats_failure=False,
        cleanup_failure=False,
    ):
        class FakeMMIO:
            def __init__(self, *_args, **_kwargs):
                self.mmio_state = {}

            def mark_mmio_state_explicit(self):
                return None

            def start_hooking(self):
                return None

            def stop_hooking(self):
                return None

        class FakeExplorer:
            def __init__(self, *_args, **_kwargs):
                self.isr_addresses = {1: 0x1000}

            def learned_irq_candidates(self, limit=128):
                del limit
                return []

            def _setup_isr_context(self, *_args, **_kwargs):
                return None

            def start_isr_execution(self, *_args, **_kwargs):
                return True

            def get_statistics(self):
                if stats_failure:
                    raise RuntimeError("injected contextual statistics failure")
                return {}

        class FakeEmulator:
            causal_context = None
            forced_branch_trace = []
            forced_branch_choices = {}
            forced_branch_sequence = []
            runtime_loop_branch_force_stats = {}
            skip_function_stats = {}
            lzo_decompress_summary_stats = {}
            external_rom_call_stats = {}
            svc_stats = {}
            thumb_state_guard_stats = {}
            invalid_thumb_recovery_stats = {}
            thumb_indirect_branch_repair_stats = {}
            execution_preflight_stats = {}
            runtime_successor_edges = set()
            bb_addr_set = set()
            instruction_count = 0
            intervention_count = 0

            def __init__(self):
                self.uc = SimpleNamespace(reg_read=lambda _reg: 0)
                self.branch_snapshot_manager = SimpleNamespace(
                    get_ordered_events=lambda: [],
                    get_ordered_occurrence_events=lambda: [],
                )

            def _execution_counter_snapshot(self):
                return {
                    "intervention_count": self.intervention_count,
                    "forced_branch_trace_count": len(self.forced_branch_trace),
                    "runtime_loop_branch_force_stats": {},
                    "skip_function_stats": {},
                    "lzo_decompress_summary_stats": {},
                    "external_rom_call_stats": {},
                    "svc_stats": {},
                    "thumb_state_guard_stats": {},
                    "invalid_thumb_recovery_stats": {},
                    "thumb_indirect_branch_repair_stats": {},
                    "execution_preflight_stats": {},
                }

            @staticmethod
            def _execution_counter_delta(current, baseline):
                del baseline
                return dict(current)

            @staticmethod
            def _get_registers():
                return {"pc": 0x1001, "sp": 0x2000}

            @staticmethod
            def _execution_preflight(_start_pc):
                return True

        runner = object.__new__(HistoricalRunner)
        runner.global_coverage = set()
        runner.max_snapshots = 3
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = None
        runner.reservoir_interrupt_contexts = []
        runner.known_branch_root_snapshots = {}
        runner.dynamic_successors = {}
        runner.mmio_handler = SimpleNamespace(mmio_state={})
        runner.prepared = SimpleNamespace(
            static_bbs={0x1000: object(), 0x2000: object()},
            static_bb_set={0x1000, 0x2000},
            instruction_to_bb={},
        )
        snapshot = SimpleNamespace(address=0x1000)
        runner._select_interrupt_contexts = lambda *_args, **_kwargs: [object()]
        runner._interrupt_context_snapshot = lambda _context: snapshot
        runner._interrupt_context_mmio_state = lambda _context: {}
        runner._interrupt_context_occurrence_signature = lambda _context: tuple()
        runner._effective_llm_config_path = lambda: None
        runner._temp_stage_for = lambda purpose: purpose
        runner._vector_table_base = lambda: 0
        runner._code_address_ranges = lambda: []
        runner._prioritized_isr_items = lambda _items, **_kwargs: [(1, 0x1000)]
        runner.validate_coverage = lambda values: set(values)
        runner._restore_branch_snapshot = lambda *_args, **_kwargs: True
        runner._restore_replay_mmio_state = lambda *_args, **_kwargs: True
        runner._ordered_replay_snapshots = lambda _emulator: []
        runner._record_dynamic_successors = lambda _events: 0
        runner._record_emulator_runtime_successors = lambda _emulator: 0
        runner._remember_main_branch_events = lambda *_args, **_kwargs: 0
        runner._remember_branch_root_snapshots = lambda *_args, **_kwargs: 0
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner._record_coverage_evidence = lambda *_args, **_kwargs: None
        runner._record_phase = lambda _name, covered, **metadata: (
            setattr(runner, "_test_phase_metadata", metadata) or set(covered)
        )

        created = []

        def new_emulator(**_kwargs):
            emulator = FakeEmulator()
            created.append(emulator)
            return emulator

        runner._new_temp_emulator = new_emulator
        dispose_calls = []

        def dispose(emulator, **_kwargs):
            dispose_calls.append(emulator)
            if cleanup_failure and len(dispose_calls) >= 2:
                raise RuntimeError("injected contextual cleanup failure")

        runner._dispose_temp_emulator = dispose

        @contextmanager
        def no_op_scope(*_args, **_kwargs):
            yield

        @contextmanager
        def trace_scope(_emulator):
            yield {0x2000}

        runner.guided_modules = no_op_scope
        runner.capture_coverage = trace_scope
        return runner, FakeMMIO, FakeExplorer

    def test_contextual_isr_result_collection_failure_is_diagnostic(self):
        runner, fake_mmio, fake_explorer = self._contextual_isr_failure_fixture(
            stats_failure=True
        )
        with patch("lsgemu.historical_runner.EnhancedMMIOHandler", fake_mmio), patch(
            "lsgemu.historical_runner.ISRExplorer", fake_explorer
        ):
            covered = HistoricalRunner.run_contextual_isr_exploration(
                runner,
                max_contexts=1,
                max_isrs=1,
                max_instructions=10,
            )

        self.assertEqual({0x2000}, covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertEqual([], summary["validated_bbs"])
        self.assertEqual([0x2000], summary["diagnostic_bbs"])
        self.assertTrue(
            any(
                "replay_result_collection_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    def test_contextual_isr_cleanup_failure_is_diagnostic_and_nonfatal(self):
        runner, fake_mmio, fake_explorer = self._contextual_isr_failure_fixture(
            cleanup_failure=True
        )
        with patch("lsgemu.historical_runner.EnhancedMMIOHandler", fake_mmio), patch(
            "lsgemu.historical_runner.ISRExplorer", fake_explorer
        ):
            covered = HistoricalRunner.run_contextual_isr_exploration(
                runner,
                max_contexts=1,
                max_isrs=1,
                max_instructions=10,
            )

        self.assertEqual({0x2000}, covered)
        summary = runner._test_phase_metadata["child_execution_summary"]
        self.assertFalse(summary["execution_telemetry_complete"])
        self.assertEqual([], summary["validated_bbs"])
        self.assertEqual([0x2000], summary["diagnostic_bbs"])
        self.assertTrue(
            any(
                "replay_cleanup_exception" in record.get("reasons", [])
                for record in summary["child_execution_records"]
            )
        )

    def test_environment_input_is_validated(self):
        status, reasons = classify_phase(
            "branch_mmio_replay",
            {
                "run_result": completed_run(
                    "bounded_completion",
                    intervention_count=0,
                    forced_branch_trace_count=0,
                    forced_branch_choices_configured=0,
                    skip_function_stats={"installed": 0, "applied": 0},
                ),
            },
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertEqual([], reasons)

    def test_successful_stream_summary_is_environment_input(self):
        run = completed_run(
            "bounded_completion",
            interventions={"function_summary_or_skip": 1},
            skip_function_stats={"applied": 1},
            environment_input_delivery_stats={
                "observed": 1,
                "accepted": 1,
                "rejected": 0,
            },
            stream_input_summary_events=[
                {
                    "kind": "stream_byte",
                    "return_value": "0x41",
                    "symbol": "serial_getc",
                }
            ],
        )
        status, reasons = classify_phase(
            "stream_input_replay",
            {"run_result": run, "environment_input_consumed": True},
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertNotIn("function_summary_or_skip", reasons)
        record = build_execution_evidence_record(
            phase_name="stream_input_replay",
            run_result=run,
            covered_bbs={0x1000},
            metadata={"environment_input_consumed": True},
        )
        self.assertEqual(1, record["environment_facts"]["stream_input_delivery"])

    def test_failed_stream_summary_does_not_hide_a_generic_skip(self):
        status, reasons = classify_phase(
            "stream_input_replay",
            {
                "run_result": completed_run(
                    "bounded_completion",
                    interventions={"function_summary_or_skip": 1},
                    skip_function_stats={"applied": 1},
                    stream_input_summary_events=[
                        {
                            "kind": "stream_read",
                            "reason": "implausible_buffer_address",
                            "return_value": "0x0",
                        }
                    ],
                )
            },
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, status)
        self.assertIn("function_summary_or_skip", reasons)

    def test_context_event_delivery_is_reported_without_being_control_intervention(self):
        status, reasons = classify_phase(
            "contextual_isr",
            {
                "entry_derivation": "entry_derived_context_snapshot_event_replay",
                "context_snapshot_mode": True,
                "child_execution_summary": summarize_child_execution(
                    [completed_run("bounded_completion")], expected_count=1
                ),
            },
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertEqual([], reasons)
        record = build_execution_evidence_record(
            phase_name="contextual_isr",
            run_result=completed_run("bounded_completion"),
            covered_bbs={0x2000},
            metadata={
                "entry_derivation": "entry_derived_context_snapshot_event_replay",
                "context_snapshot_mode": True,
                "irq": 3,
            },
        )
        self.assertEqual(1, record["environment_facts"]["interrupt_delivery"])

    def test_environment_reason_in_legacy_actual_facts_is_not_blocking(self):
        status, reasons = classify_phase(
            "environment_replay",
            {
                "run_result": completed_run(
                    "bounded_completion",
                    interventions={"stream_input_delivery": 1},
                )
            },
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertEqual([], reasons)

    def test_environment_only_diagnostic_provenance_is_promotable(self):
        run = completed_run(
            "bounded_completion",
            execution_provenance={
                "status": "diagnostic",
                "reasons": ["input_ready"],
                "telemetry_complete": True,
                "provenance_finalized": True,
            },
        )
        status, reasons = classify_phase(
            "environment_replay",
            {"run_result": run},
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertEqual([], reasons)

    def test_bare_or_unknown_diagnostic_provenance_remains_blocking(self):
        for provenance in (
            {
                "status": "diagnostic",
                "telemetry_complete": True,
                "provenance_finalized": True,
            },
            {
                "status": "diagnostic",
                "reasons": ["new_control_intervention"],
                "telemetry_complete": True,
                "provenance_finalized": True,
            },
        ):
            run = completed_run(
                "bounded_completion",
                execution_provenance=provenance,
            )
            status, reasons = classify_phase(
                "environment_replay",
                {"run_result": run},
            )
            self.assertEqual(DIAGNOSTIC_REPLAY, status)
            self.assertTrue(reasons)

    def test_loop_intervention_is_diagnostic(self):
        self.assertEqual(
            ["loop_intervention"],
            execution_intervention_reasons({"intervention_count": 1}),
        )

    def test_summary_return_is_diagnostic(self):
        status, reasons = classify_phase(
            "direct_call_summary_return",
            {"run_result": {"skip_function_stats": {"applied": 1}}},
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, status)
        self.assertIn("function_summary_or_skip", reasons)

    def test_forced_branch_is_diagnostic(self):
        status, reasons = classify_phase(
            "branch_reservoir",
            {
                "run_result": {"forced_branch_trace_count": 1},
                "evidence_class": "E2",
            },
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, status)
        self.assertIn("forced_branch", reasons)

    def test_context_stage_can_aggregate_child_replays(self):
        status, reasons = classify_phase(
            "contextual_isr",
            {
                "context_snapshot_mode": True,
                "child_execution_summary": summarize_child_execution(
                    [completed_run("bounded_completion")], expected_count=1
                ),
            },
        )
        self.assertEqual(VALIDATED_REPLAY, status)
        self.assertEqual([], reasons)

    def test_context_stage_without_child_facts_is_conservative(self):
        status, reasons = classify_phase(
            "contextual_isr",
            {"context_snapshot_mode": True},
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, status)
        self.assertIn("missing_phase_execution_summary", reasons)

    def test_environment_models_are_not_execution_interventions(self):
        self.assertEqual(
            [],
            execution_intervention_reasons(
                {
                    "modeled_async_memory_stats": {"installed": 2, "applied_reads": 8},
                    "mapped_mmio_preload_stats": {"applied": 4},
                    "status_loop_solver_stats": {"attempted": 3, "applied": 2},
                    "execution_preflight_stats": {"passed": 5},
                }
            ),
        )

    def test_architecture_repair_is_diagnostic(self):
        # r9 用户裁定：execution_preflight_repair 判环境事实（豁免），
        # thumb_indirect_branch_repair 仍属干预（diagnostic）。
        reasons = execution_intervention_reasons(
            {
                "execution_preflight_stats": {"passed": 1, "thumb_state_corrected": 1},
                "thumb_indirect_branch_repair_stats": {"repaired": 1},
            }
        )
        self.assertNotIn("execution_preflight_repair", reasons)
        self.assertIn("thumb_indirect_branch_repair", reasons)
        self.assertTrue(is_environment_fact_reason("execution_preflight_repair"))

    def test_cold_vector_probe_is_diagnostic_even_without_counting(self):
        status, reasons = classify_phase(
            "isr",
            {"coverage_counted": False},
        )
        self.assertEqual(DIAGNOSTIC_REPLAY, status)
        self.assertIn("coverage_not_counted", reasons)

    def test_paired_replay_rejects_execution_shortcut(self):
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40000000,
            value=1,
            read_pc=0x1000,
            read_occurrence=1,
        )
        result = evaluate_force_free_replay(
            signature=(((0x2000, 1), True),),
            target_bbs={0x3000},
            discovered_bbs=set(),
            control_events=[{"address": 0x2000, "choice": False}],
            control_coverage=set(),
            result_events=[{"address": 0x2000, "choice": True}],
            result_coverage={0x3000},
            constraints=[candidate],
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_trace_count=0,
            control_run_result=completed_run("control"),
            result_run_result=completed_run(
                "candidate",
                interventions={"loop_intervention": 1},
            ),
        )
        self.assertFalse(result["success"])
        self.assertEqual("execution_intervention_observed_in_validation", result["reason"])
        self.assertIn("loop_intervention", result["result_execution_interventions"])

    def test_paired_replay_requires_equivalent_initial_state(self):
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40000000,
            value=1,
            read_pc=0x1000,
            read_occurrence=1,
        )
        result = evaluate_force_free_replay(
            signature=(((0x2000, 1), True),),
            target_bbs={0x3000},
            discovered_bbs=set(),
            control_events=[{"address": 0x2000, "choice": False}],
            control_coverage=set(),
            result_events=[{"address": 0x2000, "choice": True}],
            result_coverage={0x3000},
            constraints=[candidate],
            constraint_feedback={"all_constraint_reads_matched": True},
            forced_trace_count=0,
            control_run_result=completed_run(
                "control",
                initial_state_fingerprint_summary={"fingerprint_sha256": "a"},
            ),
            result_run_result=completed_run(
                "candidate",
                initial_state_fingerprint_summary={"fingerprint_sha256": "b"},
            ),
        )
        self.assertFalse(result["success"])
        self.assertEqual("initial_state_fingerprint_diverged", result["reason"])
        self.assertFalse(result["same_initial_fingerprint"])

    def test_child_summary_separates_outcome_dimensions(self):
        summary = summarize_child_execution(
            [
                completed_run("validated"),
                completed_run(
                    "diagnostic",
                    interventions={"function_summary_or_skip": 1},
                ),
                {
                    "stop_reason": "snapshot_restore_failed",
                    "execution_attempted": True,
                    "execution_started": False,
                    "execution_failed": True,
                    "restore_failed": True,
                    "execution_telemetry_complete": False,
                },
                None,
                {"forced_branch_choices_configured": 1},
            ],
            expected_count=5,
        )
        self.assertEqual(3, summary["child_completed_with_result_count"])
        self.assertEqual(2, summary["child_successful_execution_count"])
        self.assertEqual(1, summary["child_failure_count"])
        self.assertEqual(1, summary["child_validated_execution_count"])
        self.assertEqual(2, summary["child_diagnostic_execution_count"])
        self.assertEqual(2, summary["child_unclassified_execution_count"])
        self.assertEqual(2, summary["child_execution_facts_count"])
        self.assertEqual(2, summary["child_missing_result_count"])
        self.assertFalse(summary["execution_telemetry_complete"])

    def test_canonical_partition_uses_witness_facts_not_legacy_labels(self):
        runner = object.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set={1, 2, 3})
        runner.global_coverage = {1, 2, 3}
        runner.coverage_by_evidence = {
            # Deliberately contradict the canonical records below.  E0--E3
            # must not decide the authoritative result.
            EVIDENCE_E0: {2},
            EVIDENCE_E1: set(),
            EVIDENCE_E2: {1},
            EVIDENCE_E3: {3},
        }
        runner.execution_evidence_validated_bbs = set()
        runner.execution_evidence_diagnostic_bbs = set()
        runner.execution_evidence_unclassified_bbs = set()
        runner.phase_execution_records = [
            build_execution_evidence_record(
                phase_name="environment_input_replay",
                run_result=completed_run("candidate"),
                covered_bbs={1},
                metadata={"environment_input_consumed": True},
                execution_id="validated-1",
            ),
            build_execution_evidence_record(
                phase_name="forced_probe",
                run_result=completed_run(
                    "forced",
                    interventions={"forced_branch": 1},
                ),
                covered_bbs={1, 2},
                execution_id="diagnostic-1",
            ),
            build_execution_evidence_record(
                phase_name="legacy_gap",
                run_result={"forced_branch_choices_configured": 1},
                covered_bbs={3},
                execution_id="unknown-1",
            ),
        ]

        partitions = runner._canonical_evidence_partitions()
        self.assertEqual({1}, partitions["validated"])
        self.assertEqual({1, 2}, partitions["diagnostic"])
        self.assertEqual({2}, partitions["diagnostic_only"])
        self.assertEqual({3}, partitions["unclassified"])

    def test_canonical_partition_attributes_unclassified_bbs_by_reason(self):
        """r36 K4: the gray account carries its per-reason BB attribution.

        Purely additive bookkeeping: a synthetic unclassified record (the
        ``missing_phase_execution_summary`` shape produced by reservoir/branch
        phases) attributes its BBs to that reason, and an observed BB with no
        typed witness at all lands in ``no_typed_witness``.  No status changes.
        """
        runner = object.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set={1, 2, 3, 4})
        runner.global_coverage = {1, 2, 3, 4}
        runner.coverage_by_evidence = {EVIDENCE_E0: set(), EVIDENCE_E1: set(), EVIDENCE_E2: set(), EVIDENCE_E3: set()}
        runner.execution_evidence_validated_bbs = set()
        runner.execution_evidence_diagnostic_bbs = set()
        runner.execution_evidence_unclassified_bbs = set()
        runner.phase_execution_records = [
            # The exact record shape _execution_records_for_phase materializes
            # for a branch phase with no child execution telemetry.
            {
                "schema": EVIDENCE_SCHEMA,
                "phase": "branch_reservoir_round_1",
                "execution_id": "branch_reservoir_round_1#1",
                "status": UNCLASSIFIED_REPLAY,
                "reasons": ["missing_phase_execution_summary"],
                "covered_bbs": [2, 3],
                "telemetry_complete": False,
                "execution_facts_available": False,
            },
        ]

        partitions = runner._canonical_evidence_partitions()
        # Every observed BB without a typed witness is unclassified: the
        # record's {2, 3} plus the unattributed observations {1, 4}.
        self.assertEqual({1, 2, 3, 4}, partitions["unclassified"])
        counts = partitions["unclassified_reason_counts"]
        self.assertEqual(2, counts.get("missing_phase_execution_summary"))
        self.assertEqual(2, counts.get("no_typed_witness"))
        # The classification itself did not move.
        self.assertEqual(set(), partitions["validated"])
        self.assertEqual(set(), partitions["diagnostic"])

    def test_canonical_phase_unclassified_discloses_report_level_supersession(self):
        """r41 D4: phase-level gray vs report-level exclusive partition, annotated.

        The r40 short arms show ``canonical_evidence_by_phase['branch_interleaved']``
        with ``status=unclassified`` (e.g. 50/55 BBs) while the report-level
        ``canonical_unclassified_bbs`` is 0.  That is the documented supersession
        (validated > diagnostic > unclassified across phases), not a hole: the
        same BBs carry better witnesses elsewhere.  The annotation key
        ``unclassified_superseded_bbs`` makes the relation machine-checkable:
        report-level contribution = phase ``unclassified_bbs`` minus the
        superseded count.  Clean phases keep their shape (no new key).
        """
        runner = object.__new__(HistoricalRunner)
        # 第一幕只有 {1,2,3}：全部有 typed 见证，报告级 unclassified 恒空。
        runner.prepared = SimpleNamespace(static_bb_set={1, 2, 3})
        runner.global_coverage = {1, 2, 3}
        runner.coverage_by_evidence = {EVIDENCE_E0: set(), EVIDENCE_E1: set(), EVIDENCE_E2: set(), EVIDENCE_E3: set()}
        runner.execution_evidence_validated_bbs = set()
        runner.execution_evidence_diagnostic_bbs = set()
        runner.execution_evidence_unclassified_bbs = set()
        gray_record = {
            "schema": EVIDENCE_SCHEMA,
            "phase": "branch_interleaved",
            "execution_id": "branch_interleaved#1",
            "status": UNCLASSIFIED_REPLAY,
            "reasons": ["missing_phase_execution_summary"],
            "covered_bbs": [1, 2, 3],
            "telemetry_complete": False,
            "execution_facts_available": False,
        }
        diagnostic_record = {
            "schema": EVIDENCE_SCHEMA,
            "phase": "branch_reservoir_round_1",
            "execution_id": "branch_reservoir_round_1#1",
            "status": DIAGNOSTIC_REPLAY,
            "reasons": ["partial_hit_tasks"],
            "covered_bbs": [1, 2, 3],
            "telemetry_complete": True,
            "execution_facts_available": True,
        }
        runner.phase_execution_records = [gray_record, diagnostic_record]

        partitions = runner._canonical_evidence_partitions()
        # 相位级：灰账原样保留（branch_interleaved 三枚 BB 全灰）。
        by_phase = partitions["status_by_phase"]
        self.assertEqual(3, by_phase["branch_interleaved"]["unclassified_bbs"])
        # 报告级：同一批 BB 拿到 diagnostic 见证 ⇒ 互斥分区里不灰。
        self.assertEqual(set(), partitions["unclassified"])
        self.assertEqual({1, 2, 3}, partitions["diagnostic_only"])
        # r41 D4 标注：被压倒数量可复算两级读数之差。
        self.assertEqual(
            3,
            by_phase["branch_interleaved"].get("unclassified_superseded_bbs"),
        )
        # 干净相位（无 unclassified BB）不出现新键——形状最小扰动。
        self.assertNotIn(
            "unclassified_superseded_bbs",
            by_phase["branch_reservoir_round_1"],
        )
        # 真未被压倒的灰 BB 仍如实留在报告级 unclassified。
        gray_record["covered_bbs"] = [1, 4]
        runner.global_coverage = {1, 2, 3, 4}
        runner.prepared = SimpleNamespace(static_bb_set={1, 2, 3, 4})
        partitions = runner._canonical_evidence_partitions()
        by_phase = partitions["status_by_phase"]
        self.assertEqual(2, by_phase["branch_interleaved"]["unclassified_bbs"])
        self.assertEqual(
            1,
            by_phase["branch_interleaved"].get("unclassified_superseded_bbs"),
        )
        self.assertEqual({4}, partitions["unclassified"])

    def test_late_downgrade_withdraws_unique_validated_witness(self):
        runner = self._evidence_runner(0x1000)
        record = build_execution_evidence_record(
            phase_name="environment_replay",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
            execution_id="late-downgrade-1",
        )
        feedback = OutcomeFeedback(runner)
        feedback._append_execution_records(
            "environment_replay",
            {0x1000},
            {"execution_records": [record]},
            "environment_replay#1",
        )
        self.assertEqual(
            {0x1000},
            runner._canonical_evidence_partitions()["validated"],
        )

        runner._downgrade_execution_evidence_record(
            record,
            reason="replay_cleanup_exception",
        )

        partitions = runner._canonical_evidence_partitions()
        self.assertEqual(set(), partitions["validated"])
        self.assertEqual({0x1000}, partitions["diagnostic_only"])
        self.assertEqual(
            DIAGNOSTIC_REPLAY,
            runner.phase_execution_records[0]["status"],
        )

    def test_late_downgrade_preserves_independent_validated_witness(self):
        runner = self._evidence_runner(0x1000)
        first = build_execution_evidence_record(
            phase_name="environment_replay",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
            execution_id="independent-1",
        )
        second = build_execution_evidence_record(
            phase_name="environment_replay",
            run_result=completed_run("completed"),
            covered_bbs={0x1000},
            execution_id="independent-2",
        )
        feedback = OutcomeFeedback(runner)
        feedback._append_execution_records(
            "environment_replay",
            {0x1000},
            {"execution_records": [first, second]},
            "environment_replay#1",
        )

        runner._downgrade_execution_evidence_record(
            first,
            reason="replay_cleanup_exception",
        )

        partitions = runner._canonical_evidence_partitions()
        self.assertEqual({0x1000}, partitions["validated"])
        self.assertEqual({0x1000}, partitions["diagnostic"])
        self.assertEqual(set(), partitions["diagnostic_only"])
        self.assertEqual(
            [DIAGNOSTIC_REPLAY, VALIDATED_REPLAY],
            [record["status"] for record in runner.phase_execution_records],
        )

    def test_dropped_record_keeps_validated_witness_in_long_lived_ledger(self):
        runner = self._evidence_runner(0x1000, 0x2000)
        diagnostic = build_execution_evidence_record(
            phase_name="forced_probe",
            run_result=completed_run(
                "completed",
                interventions={"forced_branch": 1},
            ),
            covered_bbs={0x1000},
            execution_id="retained-diagnostic",
        )
        validated = build_execution_evidence_record(
            phase_name="environment_replay",
            run_result=completed_run("completed"),
            covered_bbs={0x2000},
            execution_id="dropped-validated",
        )
        feedback = OutcomeFeedback(runner)
        with patch.dict(
            "os.environ",
            {"LSGEMU_EXECUTION_EVIDENCE_RECORD_LIMIT": "1"},
        ):
            feedback._append_execution_records(
                "mixed_replay",
                {0x1000, 0x2000},
                {"execution_records": [diagnostic, validated]},
                "mixed_replay#1",
            )

        self.assertEqual(1, len(runner.phase_execution_records))
        self.assertEqual(1, runner.execution_evidence_records_dropped)
        partitions = runner._canonical_evidence_partitions()
        self.assertEqual({0x2000}, partitions["validated"])
        self.assertEqual({0x1000}, partitions["diagnostic_only"])

    def test_late_downgrade_withdraws_archived_validated_witness_once(self):
        runner = self._evidence_runner(0x1000, 0x2000)
        retained = build_execution_evidence_record(
            phase_name="forced_probe",
            run_result=completed_run(
                "completed",
                interventions={"forced_branch": 1},
            ),
            covered_bbs={0x1000},
            execution_id="retained-diagnostic",
        )
        archived = build_execution_evidence_record(
            phase_name="environment_replay",
            run_result=completed_run("completed"),
            covered_bbs={0x2000},
            execution_id="archived-validated",
        )
        feedback = OutcomeFeedback(runner)
        with patch.dict(
            "os.environ",
            {"LSGEMU_EXECUTION_EVIDENCE_RECORD_LIMIT": "1"},
        ):
            feedback._append_execution_records(
                "mixed_replay",
                {0x1000, 0x2000},
                {"execution_records": [retained, archived]},
                "mixed_replay#1",
            )

        runner._downgrade_execution_evidence_record(
            archived,
            reason="replay_cleanup_exception",
        )
        runner._downgrade_execution_evidence_record(
            archived,
            reason="replay_cleanup_exception",
        )

        partitions = runner._canonical_evidence_partitions()
        self.assertEqual(set(), partitions["validated"])
        self.assertEqual({0x1000, 0x2000}, partitions["diagnostic_only"])
        self.assertEqual(
            0,
            runner.execution_evidence_validated_refcounts[0x2000],
        )
        self.assertEqual(
            1,
            runner.execution_evidence_diagnostic_refcounts[0x2000],
        )


if __name__ == "__main__":
    unittest.main()
