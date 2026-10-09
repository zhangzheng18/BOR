#!/usr/bin/env python3
"""Escalation chain: symbolic solve failure -> LLM slice solve.

The LLM tier is injectable, so these tests use a stub fallback.  They assert
the chain triggers on eligible failures (not on authoritative UNSAT), that
returned assignments re-enter the same result channel as z3 models (hypothesis
status, replay-gated upstream), and that every outcome is observable in stats.
"""

import unittest

from unicorn.arm_const import UC_ARM_REG_CPSR, UC_ARM_REG_PC

from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment
from lsgemu.register_tracer.register_tracer import RegisterTracer
from lsgemu.runner_models import external_input_site_identity


class DummyUC:
    def __init__(self):
        self._registers = {UC_ARM_REG_CPSR: 1 << 30, UC_ARM_REG_PC: 0}

    def hook_add(self, *args, **kwargs):
        return 1

    def hook_del(self, handle):
        return None

    def reg_read(self, register):
        return int(self._registers.get(register, 0))


class StubFallback:
    def __init__(self, assignments=None, error: Exception = None):
        self.requests = []
        self.assignments = list(assignments or [])
        self.error = error

    def __call__(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return list(self.assignments)


def insn(address, mnemonic, operands):
    return {
        "address": int(address),
        "mnemonic": str(mnemonic),
        "operands": str(operands),
        "size": 2,
    }


def make_unsupported_tracer(fallback=None):
    """One input site whose predicate hits an opaque node (unsupported).

    ``ADD r2, r0, r3, LSL #2`` carries shifted-register semantics the DAG does
    not model, so the compare feeds an opaque node: z3 returns ``unsupported``,
    and the deterministic checksum enumerator needs >= 2 sites, so the LLM tier
    is the only remaining solver.
    """
    static_bbs = {
        0x1000: [
            insn(0x1000, "LDRB", "r0, [r1]"),
            insn(0x1002, "ADD", "r2, r0, r3, LSL #2"),
            insn(0x1004, "CMP", "r2, #0x41"),
            insn(0x1006, "BEQ", "0x1010"),
        ]
    }
    tracer = RegisterTracer(
        DummyUC(),
        static_bbs,
        dynamic_graph_enabled=True,
        dynamic_llm_fallback=fallback,
    )
    graph = tracer.dynamic_graph
    graph.observe_external_read(
        kind="mmio",
        read_pc=0x1000,
        address=0x40000000,
        occurrence=1,
        size=1,
        observed_value=0,
        destination_register="r0",
    )
    for item in static_bbs[0x1000][1:]:
        graph.observe_instruction(item, cpsr=0 if item["mnemonic"] == "BEQ" else None)
    return tracer


def make_authoritative_unsat_tracer(fallback=None):
    """Prior same-variable constraint contradicts the desired direction.

    The earlier ``BNE`` (taken) pins ``r0 != 0x20``; asking the later ``BEQ``
    to be taken requires ``r0 == 0x20``: exact-record UNSAT, which the chain
    must treat as terminal instead of escalating.
    """
    static_bbs = {
        0x2000: [
            insn(0x2000, "LDRB", "r0, [r1]"),
            insn(0x2002, "CMP", "r0, #0x20"),
            insn(0x2004, "BNE", "0x2100"),
            insn(0x2006, "CMP", "r0, #0x20"),
            insn(0x2008, "BEQ", "0x2200"),
        ]
    }
    tracer = RegisterTracer(
        DummyUC(),
        static_bbs,
        dynamic_graph_enabled=True,
        dynamic_llm_fallback=fallback,
    )
    graph = tracer.dynamic_graph
    graph.observe_external_read(
        kind="mmio",
        read_pc=0x2000,
        address=0x40000000,
        occurrence=1,
        size=1,
        observed_value=0,
        destination_register="r0",
    )
    for item in static_bbs[0x2000][1:]:
        graph.observe_instruction(item, cpsr=0)
    return tracer


def site_identity(address=0x40000000, read_pc=0x1000, occurrence=1):
    return external_input_site_identity(
        constraint_type="mmio",
        address=address,
        read_pc=read_pc,
        read_occurrence=occurrence,
        input_kind="mmio",
    )


class DynamicLLMSliceFallbackTests(unittest.TestCase):
    def test_eligible_failure_escalates_and_returns_hypothesis_model(self):
        stub = StubFallback(assignments=[DynamicInputAssignment(
            kind="mmio",
            address=0x40000000,
            read_pc=0x1000,
            occurrence=1,
            width=8,
            value=0x141,
            observed_value=0,
            trace_event_id=1,
        )])
        tracer = make_unsupported_tracer(fallback=stub)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual(1, len(stub.requests))
        request = stub.requests[0]
        self.assertEqual("BEQ", request.condition)
        self.assertTrue(request.target_taken)
        self.assertEqual(1, request.coupled_input_count)
        self.assertEqual(1, len(request.input_sites))
        self.assertEqual(0x40000000, request.input_sites[0]["address"])
        self.assertEqual("unsupported", request.prior_failure["dynamic_solver_status"])
        joined = "\n".join(request.instruction_lines)
        self.assertIn("LDRB", joined)
        self.assertIn("ADD r2, r0, r3, LSL #2", joined)
        self.assertIn("CMP r2, #0x41", joined)

        # The escalation re-enters the exact same channel as z3 models: one
        # hypothesis model whose assignments replay-validate upstream.
        self.assertEqual("llm_slice_models", result.reason)
        self.assertEqual("llm", result.solver_backend)
        self.assertEqual("hypothesis", result.solver_status)
        self.assertEqual(1, len(result.models))
        model = result.models[0]
        self.assertEqual("llm_slice_solver", model.strategy)
        self.assertFalse(model.causal_complete)
        # Width masking keeps the assignment inside the observed read width.
        self.assertEqual(0x41, model.assignments[0].value)

        stats = tracer.dynamic_llm_fallback_stats
        self.assertEqual(1, stats["calls"])
        self.assertEqual(1, stats["attempted"])
        self.assertEqual(1, stats["success"])
        summary = tracer.get_statistics()
        self.assertEqual(1, summary["dynamic_llm_fallback"]["calls"])
        self.assertTrue(summary["dynamic_llm_fallback_configured"])

    def test_avoid_values_reach_the_request_as_failed_values(self):
        stub = StubFallback(assignments=[DynamicInputAssignment(
            kind="mmio",
            address=0x40000000,
            read_pc=0x1000,
            occurrence=1,
            width=8,
            value=0x41,
            observed_value=0,
            trace_event_id=1,
        )])
        tracer = make_unsupported_tracer(fallback=stub)

        tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
            avoid_values_by_site={site_identity(): {0x00, 0x33}},
        )

        self.assertEqual(1, len(stub.requests))
        self.assertEqual([0x00, 0x33], stub.requests[0].input_sites[0]["failed_values"])

    def test_authoritative_unsat_does_not_escalate(self):
        stub = StubFallback(assignments=[DynamicInputAssignment(
            kind="mmio",
            address=0x40000000,
            read_pc=0x2000,
            occurrence=1,
            width=8,
            value=0x20,
            observed_value=0,
            trace_event_id=1,
        )])
        tracer = make_authoritative_unsat_tracer(fallback=stub)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x2008,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual("unsat", result.solver_status)
        self.assertEqual(0, len(result.models))
        self.assertEqual(0, len(stub.requests))
        self.assertEqual(0, tracer.dynamic_llm_fallback_stats["calls"])

    def test_no_fallback_injected_keeps_previous_behavior(self):
        tracer = make_unsupported_tracer(fallback=None)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual("unsupported_dynamic_predicate", result.reason)
        self.assertEqual(0, len(result.models))
        self.assertFalse(tracer.get_statistics()["dynamic_llm_fallback_configured"])

    def test_hallucinated_site_is_rejected(self):
        stub = StubFallback(assignments=[
            DynamicInputAssignment(
                kind="mmio",
                address=0x40009999,
                read_pc=0x1000,
                occurrence=1,
                width=8,
                value=0x41,
                observed_value=0,
                trace_event_id=1,
            ),
        ])
        tracer = make_unsupported_tracer(fallback=stub)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual(0, len(result.models))
        stats = tracer.dynamic_llm_fallback_stats
        self.assertEqual(1, stats["rejected_unknown_sites"])
        self.assertEqual(1, stats["no_valid_assignments"])
        self.assertEqual(0, stats["success"])

    def test_fallback_exception_is_observable_and_non_fatal(self):
        stub = StubFallback(error=RuntimeError("boom"))
        tracer = make_unsupported_tracer(fallback=stub)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual("unsupported_dynamic_predicate", result.reason)
        self.assertEqual(1, tracer.dynamic_llm_fallback_stats["errors"])
        self.assertEqual(0, tracer.dynamic_llm_fallback_stats["success"])

    def test_setter_attaches_fallback_after_construction(self):
        stub = StubFallback(assignments=[DynamicInputAssignment(
            kind="mmio",
            address=0x40000000,
            read_pc=0x1000,
            occurrence=1,
            width=8,
            value=0x41,
            observed_value=0,
            trace_event_id=1,
        )])
        tracer = make_unsupported_tracer(fallback=None)
        self.assertFalse(tracer.get_statistics()["dynamic_llm_fallback_configured"])
        tracer.set_dynamic_llm_fallback(stub)

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1006,
            occurrence=1,
            target_taken=True,
        )

        self.assertEqual("llm_slice_models", result.reason)


if __name__ == "__main__":
    unittest.main()
