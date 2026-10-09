#!/usr/bin/env python3
"""Contracts for causal event identity and bounded LLM hypotheses."""

from types import SimpleNamespace
from unittest.mock import patch
import unittest

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager
from lsgemu.dynamic_constraint_recovery import DynamicExpressionGraph
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.llm_guide.llm_guide import LLMGuide
from lsgemu.runner_models import BranchConstraintCandidate, constraint_delivery_site_identity
from lsgemu.register_tracer.register_tracer import RegisterTracer


class _FakeGuide(LLMGuide):
    def __init__(self, payload):
        super().__init__({}, use_llm=True)
        self.payload = payload

    def _call_llm_json(self, prompt, repair_prompt=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))]
        )


class CausalRecoveryContractTests(unittest.TestCase):
    def test_occurrence_cursor_survives_retention_limit(self):
        manager = BranchSnapshotManager()
        manager.max_occurrence_events = 2
        for index in range(3):
            manager.record_event(
                0x1000,
                0x1002,
                0x1100,
                0x1004,
                "NE",
                False,
                depth=index,
            )
        self.assertEqual(2, len(manager.get_ordered_occurrence_events()))
        self.assertEqual(3, manager.get_occurrence_event_cursor())
        self.assertEqual(3, manager.get_statistics()["occurrence_event_sequence"])

    def test_dynamic_result_reports_sat_and_unsat_separately(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x2000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction({
            "address": 0x2002,
            "mnemonic": "CMP",
            "operands": "r0, #1",
        })
        graph.observe_instruction({
            "address": 0x2004,
            "mnemonic": "BEQ",
            "operands": "0x2100",
        }, cpsr=0)
        sat = graph.recover_branch_inputs(
            branch_pc=0x2004,
            occurrence=1,
            target_taken=True,
        )
        self.assertTrue(sat.models)
        self.assertEqual("sat", sat.solver_status)
        self.assertFalse(sat.fallback_eligible)

        site = graph._input_site_identity(graph.inputs[next(iter(graph.inputs))])
        unsat = graph.recover_branch_inputs(
            branch_pc=0x2004,
            occurrence=1,
            target_taken=True,
            avoid_values_by_site={site: {1}},
        )
        self.assertFalse(unsat.models)
        self.assertEqual("unsat", unsat.solver_status)

    def test_external_memory_trace_id_is_provenance_only(self):
        graph = DynamicExpressionGraph()
        graph.observe_memory_load(
            read_pc=0x2100,
            address=0x20001000,
            size=1,
            observed_value=0x12,
            destination_register="r0",
            external_input=True,
            occurrence=3,
            trace_event_id=77,
        )
        item = next(iter(graph.inputs.values()))
        self.assertEqual(77, item.trace_event_id)
        self.assertEqual(
            ("memory", 0x20001000, 0x2100, 3),
            graph._input_site_identity(item),
        )

    def test_opaque_instruction_retains_leaves_for_checksum_fallback(self):
        graph = DynamicExpressionGraph()
        for index, value in enumerate((0x10, 0x11, 0xFF)):
            graph.observe_external_read(
                kind="external_memory",
                read_pc=0x2200 + index * 2,
                address=0x20002000 + index,
                occurrence=1,
                size=1,
                observed_value=value,
                destination_register=f"r{index}",
                trace_event_id=index + 1,
            )
        graph.observe_instruction({
            "address": 0x2210,
            "mnemonic": "UADD8",
            "operands": "r3, r0, r1",
        })
        graph.observe_instruction({
            "address": 0x2212,
            "mnemonic": "EOR",
            "operands": "r3, r3, r2",
        })
        graph.observe_instruction({
            "address": 0x2214,
            "mnemonic": "CMP",
            "operands": "r3, #0",
        })
        graph.observe_instruction({
            "address": 0x2216,
            "mnemonic": "BEQ",
            "operands": "0x2300",
        }, cpsr=0)

        unsupported = graph.recover_branch_inputs(
            branch_pc=0x2216,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual("unsupported", unsupported.solver_status)
        fallback = graph.recover_checksum_candidates(
            branch_pc=0x2216,
            occurrence=1,
            max_models=4,
        )
        self.assertTrue(fallback.models)
        self.assertEqual("checksum_enumerator", fallback.solver_backend)
        self.assertTrue(all(not model.causal_complete for model in fallback.models))

    def test_flag_setting_arithmetic_drives_exact_branch_recovery(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio", read_pc=0x2300, address=0x40000000,
            occurrence=1, size=1, observed_value=1,
            destination_register="r0",
        )
        graph.observe_instruction({
            "address": 0x2302, "mnemonic": "SUBS", "operands": "r1, r0, #1",
        })
        graph.observe_instruction({
            "address": 0x2304, "mnemonic": "BNE", "operands": "0x2400",
        }, cpsr=(1 << 30))

        result = graph.recover_branch_inputs(
            branch_pc=0x2304,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual("sat", result.solver_status)
        self.assertTrue(result.models)
        self.assertNotEqual(1, result.models[0].assignments[0].value)

    def test_logical_flag_writer_refuses_unknown_carry_semantics(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio", read_pc=0x2320, address=0x40000000,
            occurrence=1, size=1, observed_value=1,
            destination_register="r0",
        )
        graph.observe_instruction({
            "address": 0x2322, "mnemonic": "ANDS", "operands": "r1, r0, #1",
        })
        graph.observe_instruction({
            "address": 0x2324, "mnemonic": "BCS", "operands": "0x2400",
        }, cpsr=(1 << 29))

        result = graph.recover_branch_inputs(
            branch_pc=0x2324,
            occurrence=1,
            target_taken=False,
        )
        self.assertFalse(result.models)
        self.assertEqual("unsupported", result.solver_status)

    def test_opaque_call_return_keeps_callee_internal_input_leaves(self):
        graph = DynamicExpressionGraph()
        argument_leaf = graph.observe_external_read(
            kind="mmio", read_pc=0x2340, address=0x40000000,
            occurrence=1, size=1, observed_value=1,
            destination_register="r0",
        )
        internal_leaf = graph.observe_external_read(
            kind="external_memory", read_pc=0x2342, address=0x20001000,
            occurrence=1, size=1, observed_value=2,
            destination_register="r4",
        )
        graph.observe_instruction({
            "address": 0x2344, "mnemonic": "UADD8", "operands": "r0, r0, r4",
        })
        returned = graph.observe_call_return(
            call_pc=0x2330,
            return_pc=0x2350,
            argument_nodes={"r0": int(argument_leaf)},
            concrete_result=3,
            result_register="r0",
            callee_target=0x2340,
            argument_value_before=1,
        )
        self.assertIsNotNone(returned)
        self.assertEqual(
            graph._leaf_nodes(argument_leaf) | graph._leaf_nodes(internal_leaf),
            graph._leaf_nodes(returned),
        )

    def test_budget_result_gets_one_bounded_solver_escalation(self):
        tracer = RegisterTracer(None, {}, dynamic_graph_enabled=True)
        graph = tracer.dynamic_graph
        graph.observe_external_read(
            kind="mmio", read_pc=0x2400, address=0x40000000,
            occurrence=1, size=1, observed_value=0,
            destination_register="r0",
        )
        graph.observe_external_read(
            kind="mmio", read_pc=0x2402, address=0x40000004,
            occurrence=1, size=1, observed_value=0,
            destination_register="r1",
        )
        graph.observe_instruction({
            "address": 0x2404, "mnemonic": "ORR", "operands": "r2, r0, r1",
        })
        graph.observe_instruction({
            "address": 0x2406, "mnemonic": "CMP", "operands": "r2, #1",
        })
        graph.observe_instruction({
            "address": 0x2408, "mnemonic": "BEQ", "operands": "0x2500",
        }, cpsr=0)

        with patch.dict("os.environ", {
            "LSGEMU_DYNAMIC_SLICE_MAX_INPUTS": "1",
            "LSGEMU_DYNAMIC_ESCALATION_MAX_INPUTS": "4",
            "LSGEMU_DYNAMIC_SLICE_MAX_NODES": "20000",
            "LSGEMU_DYNAMIC_ESCALATION_MAX_NODES": "1000",
        }):
            result = tracer.recover_dynamic_branch_inputs(
                branch_pc=0x2408,
                occurrence=1,
                target_taken=True,
            )
        self.assertTrue(result.models)
        self.assertTrue(result.escalated)
        self.assertEqual("budget", result.initial_solver_status)
        self.assertEqual(2, result.solver_attempts)
        initial_limits = dict(result.initial_limits)
        final_limits = dict(result.final_limits)
        self.assertTrue(all(
            final_limits[name] >= value
            for name, value in initial_limits.items()
        ))

    def test_compound_llm_can_only_select_observed_sites(self):
        guide = _FakeGuide(
            '{"assignments":[{"site_index":0,"value":"0x41"},'
            '{"site_index":1,"value":"0x42"}],"confidence":0.7}'
        )
        result = guide.infer_compound_alternative_constraints(
            0x3000,
            "BEQ",
            True,
            [
                {"address": 0x40000000, "read_pc": 0x2000, "occurrence": 1,
                 "width": 8, "current_value": 0, "failed_values": [0]},
                {"address": 0x40000004, "read_pc": 0x2002, "occurrence": 1,
                 "width": 8, "current_value": 0, "failed_values": [0]},
            ],
        )
        self.assertEqual(
            [{"site_index": 0, "value": 0x41}, {"site_index": 1, "value": 0x42}],
            result,
        )

        invalid = _FakeGuide(
            '{"assignments":[{"site_index":99,"value":"0x41"}]}'
        )
        self.assertEqual([], invalid.infer_compound_alternative_constraints(
            0x3000,
            "BEQ",
            True,
            [
                {"address": 0x40000000, "width": 8, "current_value": 0},
                {"address": 0x40000004, "width": 8, "current_value": 0},
            ],
        ))

    def test_historical_runner_builds_compound_candidate_set(self):
        runner = object.__new__(HistoricalRunner)
        runner.known_main_branch_events = {}
        runner.llm_guide = _FakeGuide(
            '{"assignments":[{"site_index":0,"value":"0x11"},'
            '{"site_index":1,"value":"0x22"}]}'
        )
        runner._branch_instruction = lambda _bb: {
            "address": 0x3002,
            "mnemonic": "BEQ",
        }
        runner._dependency_constraint_candidates = lambda _pc: []
        first = BranchConstraintCandidate(
            constraint_type="mmio", address=0x40000000, value=0,
            read_pc=0x2000, read_occurrence=1, width=8,
        )
        second = BranchConstraintCandidate(
            constraint_type="mmio", address=0x40000004, value=0,
            read_pc=0x2002, read_occurrence=1, width=8,
        )
        variants = runner._naturalization_llm_alternatives(
            (0x3000, 1),
            True,
            (first, second),
            failure_context={"failure_reason": "wrong_natural_choice"},
            avoid_values_by_site={
                constraint_delivery_site_identity(first): {0},
                constraint_delivery_site_identity(second): {0},
            },
        )
        self.assertTrue(variants)
        values = {
            candidate.address: candidate.value
            for candidate in variants[0]
        }
        self.assertEqual({0x40000000: 0x11, 0x40000004: 0x22}, values)


if __name__ == "__main__":
    unittest.main()
