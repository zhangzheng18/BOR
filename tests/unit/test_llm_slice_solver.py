#!/usr/bin/env python3
"""LLM slice-solver tier: guide prompt, adapter mapping, runner wiring.

Task-1 acceptance: an eligible symbolic failure escalates, the LLM result
re-enters the same candidate-set channel as z3 models (replay-gated), and
failures are observable.  Task-3: the coupled-input count selects the prompt's
solving mode.  Task-4: retries carry the previous values and the real replay
outcome.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from unicorn.arm_const import UC_ARM_REG_CPSR, UC_ARM_REG_PC

from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.llm_guide.llm_guide import LLMGuide
from lsgemu.llm_slice_solver import LLMGuideSliceSolver
from lsgemu.register_tracer.register_tracer import (
    DynamicLLMSliceRequest,
    RegisterTracer,
)


class DummyUC:
    def __init__(self):
        self._registers = {UC_ARM_REG_CPSR: 1 << 30, UC_ARM_REG_PC: 0}

    def hook_add(self, *args, **kwargs):
        return 1

    def hook_del(self, handle):
        return None

    def reg_read(self, register):
        return int(self._registers.get(register, 0))


def insn(address, mnemonic, operands):
    return {
        "address": int(address),
        "mnemonic": str(mnemonic),
        "operands": str(operands),
        "size": 2,
    }


def make_unsupported_tracer(fallback=None):
    """Single-site predicate behind an opaque node: z3 says ``unsupported``."""
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
    return tracer, static_bbs


def sample_sites(count=2):
    sites = []
    for index in range(count):
        sites.append({
            "site_index": index,
            "kind": "mmio",
            "address": 0x40000000 + 4 * index,
            "read_pc": 0x1000 + 2 * index,
            "occurrence": index + 1,
            "width": 8,
            "current_value": 0x00,
            "trace_event_id": index + 1,
            "failed_values": [0x11] if index == 0 else [],
        })
    return sites


def sample_request(**overrides):
    fields = dict(
        branch_pc=0x1006,
        branch_occurrence=1,
        condition="BEQ",
        compare_op="CMP",
        compare_pc=0x1004,
        target_taken=True,
        relation_kind="opaque_input_relation",
        instruction_lines=(
            "0x00001000: LDRB r0, [r1]  ; input[0] mmio[0x40000000] occ=1 width=8 current=0x00",
            "0x00001002: ADD r2, r0, r3, LSL #2",
            "0x00001004: CMP r2, #0x41  ; failing compare (CMP); observed not-taken",
            "0x00001006: BEQ 0x1010  ; failing branch (BEQ); observed not-taken",
        ),
        constraint_note="",
        input_sites=tuple(sample_sites()),
        coupled_input_count=2,
        same_variable_constraints=1,
        slice_instruction_count=4,
        truncated_instructions=0,
        prior_failure={"dynamic_solver_status": "unsupported"},
    )
    fields.update(overrides)
    return DynamicLLMSliceRequest(**fields)


class _Response:
    def __init__(self, text):
        self._text = text
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

    def __str__(self):
        return self._text


class LLMGuideSliceAssignmentTests(unittest.TestCase):
    def make_guide(self):
        return LLMGuide({}, use_llm=True)

    def test_returns_assignments_and_records_history(self):
        guide = self.make_guide()
        payload = json.dumps({
            "assignments": [{"site_index": 1, "value": "0x42"}],
            "analysis": "ok",
            "confidence": 0.9,
            "why_values_satisfies_branch": "ok",
        })
        with patch.object(guide, "_call_llm_json", return_value=_Response(payload)):
            result = guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=list(sample_request().instruction_lines),
                input_sites=sample_sites(),
                coupled_input_count=2,
            )
        self.assertEqual([{"site_index": 1, "value": 0x42}], result)
        record = guide.get_inference_history()[-1]
        self.assertEqual("llm_slice", record["method"])
        self.assertTrue(record["requires_force_free_replay_validation"])
        self.assertEqual("llm_slice", guide.last_inference_metadata["inference_method"])

    def test_prompt_carries_slice_coupling_and_prior_failure(self):
        guide = self.make_guide()
        prompts = []

        def fake_call(prompt, repair_prompt=None):
            prompts.append(prompt)
            return _Response(json.dumps({
                "assignments": [{"site_index": 0, "value": "0x41"}],
                "analysis": "ok",
                "confidence": 0.5,
                "why_values_satisfies_branch": "ok",
            }))

        with patch.object(guide, "_call_llm_json", side_effect=fake_call):
            guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=list(sample_request().instruction_lines),
                input_sites=sample_sites(),
                coupled_input_count=2,
                relation_kind="opaque_input_relation",
                same_variable_constraints=1,
                prior_failure={"dynamic_solver_status": "unsupported"},
            )
        prompt = prompts[0]
        self.assertIn("ADD r2, r0, r3, LSL #2", prompt)
        self.assertIn("LDRB r0, [r1]", prompt)
        self.assertIn("2 个输入来源需要同时联合确定", prompt)
        self.assertIn("dynamic_solver_status=unsupported", prompt)
        self.assertIn("先前约束条数：1", prompt)
        self.assertIn("0x40000000", prompt)

    def test_single_source_prompt_declares_single_variable_mode(self):
        guide = self.make_guide()
        prompts = []

        def fake_call(prompt, repair_prompt=None):
            prompts.append(prompt)
            return _Response(json.dumps({
                "assignments": [{"site_index": 0, "value": "0x41"}],
                "analysis": "ok",
                "confidence": 0.5,
                "why_values_satisfies_branch": "ok",
            }))

        with patch.object(guide, "_call_llm_json", side_effect=fake_call):
            guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=["0x00001000: LDRB r0, [r1]"],
                input_sites=sample_sites(1),
                coupled_input_count=1,
            )
        self.assertIn("单变量求解", prompts[0])

    def test_retry_prompt_carries_previous_values_and_replay_outcome(self):
        guide = self.make_guide()
        prompts = []

        def fake_call(prompt, repair_prompt=None):
            prompts.append(prompt)
            return _Response(json.dumps({
                "assignments": [{"site_index": 0, "value": "0x42"}],
                "analysis": "retry",
                "confidence": 0.5,
                "why_values_satisfies_branch": "retry",
            }))

        previous = {
            "assignments": [{"site_index": 0, "value": "0x41"}],
            "replay_outcome": {
                "outcome": "no_effect_same_outcome",
                "failure_reason": "branch_direction_unchanged",
            },
        }
        with patch.object(guide, "_call_llm_json", side_effect=fake_call):
            guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=["0x00001000: LDRB r0, [r1]"],
                input_sites=sample_sites(1),
                coupled_input_count=1,
                previous_attempt=previous,
            )
        prompt = prompts[0]
        self.assertIn("上次给出的值: site_index=0 value=0x00000041", prompt)
        self.assertIn("no_effect_same_outcome", prompt)
        self.assertIn("替代值", prompt)
        record = guide.get_inference_history()[-1]
        self.assertEqual("llm_slice", record["method"])
        self.assertEqual(
            "slice_retry_after_replay_failure", record["prompt_purpose"]
        )

    def test_same_or_failed_value_rejected(self):
        guide = self.make_guide()
        payload = json.dumps({
            # Both equal their site's current value: no effective assignment.
            "assignments": [
                {"site_index": 0, "value": "0x00"},
                {"site_index": 1, "value": "0x00"},
            ],
            "analysis": "bad",
            "confidence": 0.1,
            "why_values_satisfies_branch": "bad",
        })
        with patch.object(guide, "_call_llm_json", return_value=_Response(payload)):
            result = guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=["0x00001000: LDRB r0, [r1]"],
                input_sites=sample_sites(2),
                coupled_input_count=2,
            )
        self.assertIsNone(result)
        record = guide.get_inference_history()[-1]
        self.assertEqual("llm_slice_error", record["method"])

    def test_dedicated_budget_bounds_slice_calls(self):
        guide = self.make_guide()
        guide.inference_history.append({"method": "llm_slice", "dummy": True})
        with patch.dict(
            "os.environ", {"LSGEMU_MAX_LLM_SLICE_CALLS": "1"}
        ), patch.object(guide, "_call_llm_json") as fake_call:
            result = guide.infer_slice_assignments(
                branch_pc=0x1006,
                condition="BEQ",
                target_taken=True,
                instruction_lines=["0x00001000: LDRB r0, [r1]"],
                input_sites=sample_sites(1),
                coupled_input_count=1,
            )
        self.assertIsNone(result)
        fake_call.assert_not_called()
        record = guide.get_inference_history()[-1]
        self.assertEqual("llm_slice_skipped", record["method"])


class FakeGuide:
    def __init__(self, use_llm=True, raw=None):
        self.use_llm = use_llm
        self.raw = raw or [{"site_index": 0, "value": 0x41}]
        self.calls = []

    def infer_slice_assignments(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.raw)


class LLMGuideSliceSolverAdapterTests(unittest.TestCase):
    def test_maps_request_to_assignments_and_feeds_retry_memory(self):
        guide = FakeGuide()
        adapter = LLMGuideSliceSolver(guide)
        request = sample_request(input_sites=tuple(sample_sites(1)), coupled_input_count=1)

        assignments = adapter(request)
        self.assertEqual(1, len(assignments))
        assignment = assignments[0]
        self.assertEqual("mmio", assignment.kind)
        self.assertEqual(0x40000000, assignment.address)
        self.assertEqual(0x1000, assignment.read_pc)
        self.assertEqual(1, assignment.occurrence)
        self.assertEqual(8, assignment.width)
        self.assertEqual(0x41, assignment.value)
        self.assertEqual(0, assignment.observed_value)
        # First call is not a retry.
        self.assertIsNone(guide.calls[0]["previous_attempt"])

        # Replay failure recorded by the naturalization layer turns the next
        # escalation for the same branch into a retry with real feedback.
        self.assertTrue(adapter.note_replay_failure(
            0x1006, 1, {"outcome": "no_effect_same_outcome"}
        ))
        adapter(request)
        previous = guide.calls[1]["previous_attempt"]
        self.assertEqual([{"site_index": 0, "value": 0x41}], previous["assignments"])
        self.assertEqual(
            {"outcome": "no_effect_same_outcome"}, previous["replay_outcome"]
        )
        self.assertEqual(1, adapter.get_statistics()["retries"])

    def test_skips_when_guide_disabled(self):
        adapter = LLMGuideSliceSolver(FakeGuide(use_llm=False))
        self.assertIsNone(adapter(sample_request()))
        self.assertEqual(1, adapter.get_statistics()["skipped"])


class PreparedShim:
    def __init__(self, static_bbs, lookup):
        self.static_bbs = static_bbs
        self.instruction_lookup = lookup
        self.compare_lookup = {0x1006: lookup[0x1004]}

    def get_branch_instruction(self, bb_start):
        instructions = self.static_bbs.get(int(bb_start)) or []
        return instructions[-1] if instructions else None


class RunnerCandidateChannelTests(unittest.TestCase):
    """The escalated LLM model must enter the same channel as z3 models."""

    def test_llm_slice_result_becomes_replay_validated_candidate_set(self):
        stub_assignments = [DynamicInputAssignment(
            kind="mmio",
            address=0x40000000,
            read_pc=0x1000,
            occurrence=1,
            width=8,
            value=0x41,
            observed_value=0,
            trace_event_id=1,
        )]

        class StubFallback:
            def __init__(self):
                self.requests = []

            def __call__(self, request):
                self.requests.append(request)
                return list(stub_assignments)

        stub = StubFallback()
        tracer, static_bbs = make_unsupported_tracer(fallback=stub)
        lookup = {item["address"]: item for item in static_bbs[0x1000]}

        runner = object.__new__(HistoricalRunner)
        runner.prepared = PreparedShim(static_bbs, lookup)
        runner.register_tracer = tracer
        runner.known_main_branch_events = {}
        runner.causal_constraint_recovery_enabled = True
        runner.last_dynamic_recovery_result = None
        runner.last_dynamic_recovery_key = None

        candidate_sets = runner._naturalization_dynamic_candidate_sets(
            (0x1006, 1),
            True,
            max_models=2,
        )

        self.assertEqual(1, len(candidate_sets))
        candidate = candidate_sets[0][0]
        self.assertEqual("mmio", candidate.constraint_type)
        self.assertEqual(0x40000000, candidate.address)
        self.assertEqual(0x41, candidate.value)
        self.assertEqual(0x1000, candidate.read_pc)
        self.assertIn("llm_slice_solver", candidate.source)
        self.assertTrue(candidate.externally_controllable)
        # Same observability as the z3 path: backend/reason recorded in stats.
        stats = runner.dynamic_constraint_recovery_stats
        self.assertEqual(1, stats["calls"])
        self.assertEqual(1, stats["solver_backend_llm"])
        self.assertEqual(1, stats["reason_llm_slice_models"])


if __name__ == "__main__":
    unittest.main()
