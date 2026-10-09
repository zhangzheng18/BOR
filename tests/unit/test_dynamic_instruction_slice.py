#!/usr/bin/env python3
"""Instruction-level slice export for LLM-assisted constraint solving.

Design §5.1 trimming rules under test:
1. data-dependency closure back to the source instructions;
2. *all* constraints on the same variable are retained;
3. control dependencies are not retained (only same-leaf constraints are).
"""

import unittest

from lsgemu.dynamic_constraint_recovery import DynamicExpressionGraph


def instruction(address, mnemonic, operands):
    return {
        "address": int(address),
        "mnemonic": str(mnemonic),
        "operands": str(operands),
        "size": 2,
    }


def build_lookup(*instructions):
    return {int(item["address"]): item for item in instructions}


class InstructionSliceExportTests(unittest.TestCase):
    def test_multibyte_slice_contains_dependency_chain_and_inputs(self):
        lookup = build_lookup(
            instruction(0x1000, "LDRB", "r0, [r1, #0]"),
            instruction(0x1100, "LSL", "r2, r0, #8"),
            instruction(0x1102, "ORR", "r2, r2, r0"),
            instruction(0x1200, "CMP", "r2, #0x4142"),
            instruction(0x1202, "BEQ", "0x1300"),
        )
        graph = DynamicExpressionGraph(
            instruction_to_bb={
                0x1000: 0x1000,
                0x1100: 0x1100,
                0x1102: 0x1100,
                0x1200: 0x1200,
                0x1202: 0x1200,
            }
        )
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(lookup[0x1100])
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=2,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(lookup[0x1102])
        graph.observe_instruction(lookup[0x1200])
        graph.observe_instruction(lookup[0x1202], cpsr=0)

        slice_result = graph.export_instruction_slice(
            branch_pc=0x1202,
            occurrence=1,
            instruction_lookup=lookup,
        )
        self.assertIsNotNone(slice_result)
        addresses = [line.address for line in slice_result.lines]
        # Full def-use chain: both input reads, both data instructions, and the
        # compare/branch pair must be present.
        for expected in (0x1000, 0x1100, 0x1102, 0x1200, 0x1202):
            self.assertIn(expected, addresses)
        kinds = {line.address: line.kind for line in slice_result.lines}
        self.assertEqual("target_compare", kinds[0x1200])
        self.assertEqual("target_branch", kinds[0x1202])
        input_lines = [line for line in slice_result.lines if line.kind == "input"]
        self.assertEqual(1, len(input_lines))
        self.assertIn("occ=1", input_lines[0].detail)
        self.assertIn("occ=2", input_lines[0].detail)
        # Execution order: the second read (occ=2) must come after the LSL.
        self.assertLess(addresses.index(0x1100), addresses.index(0x1102))
        self.assertEqual(2, len(slice_result.input_sites))
        self.assertEqual(0, slice_result.truncated_instructions)
        self.assertEqual(2, graph.count_branch_input_sources(0x1202, 1))
        # Formatted prompt lines carry addresses, mnemonics and operands.
        formatted = slice_result.format_instruction_lines()
        self.assertTrue(any("LSL r2, r0, #8" in line for line in formatted))
        self.assertTrue(any("input[" in line for line in formatted))

    def test_slice_excludes_unrelated_instructions_and_control_dependencies(self):
        lookup = build_lookup(
            instruction(0x1000, "LDRB", "r0, [r1, #0]"),
            instruction(0x1100, "CMP", "r0, #0x20"),
            instruction(0x1102, "BNE", "0x1400"),
            instruction(0x2000, "LDRB", "r7, [r6, #0]"),
            instruction(0x2100, "CMP", "r7, #5"),
            instruction(0x2102, "BNE", "0x1500"),
            instruction(0x2200, "CMP", "r0, #0x10"),
            instruction(0x2202, "BEQ", "0x1600"),
        )
        graph = DynamicExpressionGraph(
            instruction_to_bb={
                0x1000: 0x1000,
                0x1100: 0x1100,
                0x1102: 0x1100,
                0x2000: 0x2000,
                0x2100: 0x2100,
                0x2102: 0x2100,
                0x2200: 0x2200,
                0x2202: 0x2200,
            }
        )
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(lookup[0x1100])
        graph.observe_instruction(lookup[0x1102], cpsr=0)
        # Unrelated chain on a disjoint input: neither its data nor its branch
        # (a pure control dependency for the target) may enter the slice.
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x2000,
            address=0x40000010,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r7",
        )
        graph.observe_instruction(lookup[0x2100])
        graph.observe_instruction(lookup[0x2102], cpsr=0)
        graph.observe_instruction(lookup[0x2200])
        graph.observe_instruction(lookup[0x2202], cpsr=0)

        slice_result = graph.export_instruction_slice(
            branch_pc=0x2202,
            occurrence=1,
            instruction_lookup=lookup,
        )
        addresses = {line.address for line in slice_result.lines}
        for absent in (0x2000, 0x2100, 0x2102):
            self.assertNotIn(absent, addresses)
        for present in (0x1000, 0x1100, 0x1102, 0x2200, 0x2202):
            self.assertIn(present, addresses)

    def test_slice_keeps_all_same_variable_constraints(self):
        # `cmp r0,#0x20` then `cmp r0,#0x10`: both constrain the same input,
        # so both compares (and both branches) must survive the trim.
        lookup = build_lookup(
            instruction(0x1000, "LDRB", "r0, [r1, #0]"),
            instruction(0x1100, "CMP", "r0, #0x20"),
            instruction(0x1102, "BNE", "0x1400"),
            instruction(0x1200, "CMP", "r0, #0x10"),
            instruction(0x1202, "BEQ", "0x1600"),
        )
        graph = DynamicExpressionGraph(
            instruction_to_bb={
                0x1000: 0x1000,
                0x1100: 0x1100,
                0x1102: 0x1100,
                0x1200: 0x1200,
                0x1202: 0x1200,
            }
        )
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(lookup[0x1100])
        graph.observe_instruction(lookup[0x1102], cpsr=0)
        graph.observe_instruction(lookup[0x1200])
        graph.observe_instruction(lookup[0x1202], cpsr=0)

        slice_result = graph.export_instruction_slice(
            branch_pc=0x1202,
            occurrence=1,
            instruction_lookup=lookup,
        )
        addresses = {line.address for line in slice_result.lines}
        self.assertIn(0x1100, addresses)
        self.assertIn(0x1102, addresses)
        self.assertEqual(1, slice_result.same_variable_constraints)
        constraint_lines = [
            line for line in slice_result.lines if line.kind == "constraint"
        ]
        self.assertTrue(constraint_lines)
        self.assertIn("must keep holding", constraint_lines[0].detail)
        # The prior constraint appears in execution order, before the target.
        ordered = [line.address for line in slice_result.lines]
        self.assertLess(ordered.index(0x1100), ordered.index(0x1200))
        self.assertEqual(1, slice_result.input_sites and len(slice_result.input_sites))

    def test_slice_truncation_is_explicit_not_silent(self):
        lookup = build_lookup(
            instruction(0x1000, "LDRB", "r0, [r1, #0]"),
            instruction(0x1100, "CMP", "r0, #0x20"),
            instruction(0x1102, "BNE", "0x1400"),
            instruction(0x1200, "CMP", "r0, #0x10"),
            instruction(0x1202, "BEQ", "0x1600"),
        )
        graph = DynamicExpressionGraph(instruction_to_bb={})
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(lookup[0x1100])
        graph.observe_instruction(lookup[0x1102], cpsr=0)
        graph.observe_instruction(lookup[0x1200])
        graph.observe_instruction(lookup[0x1202], cpsr=0)

        slice_result = graph.export_instruction_slice(
            branch_pc=0x1202,
            occurrence=1,
            instruction_lookup=lookup,
            max_instructions=3,
        )
        self.assertEqual(3, len(slice_result.lines))
        self.assertGreater(slice_result.truncated_instructions, 0)
        note = slice_result.format_constraint_note()
        self.assertIn("truncated", note)
        # The tail (closest to the failing branch) is what survives.
        kept = [line.address for line in slice_result.lines]
        self.assertIn(0x1200, kept)
        self.assertIn(0x1202, kept)

    def test_missing_record_or_inputless_predicate_returns_none(self):
        graph = DynamicExpressionGraph()
        self.assertIsNone(
            graph.export_instruction_slice(branch_pc=0x9999, occurrence=1)
        )
        self.assertIsNone(graph.count_branch_input_sources(0x9999, 1))

    def test_count_branch_input_sources_single_vs_coupled(self):
        single = DynamicExpressionGraph()
        single.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        single.observe_instruction(instruction(0x1100, "CMP", "r0, #1"))
        single.observe_instruction(instruction(0x1102, "BEQ", "0x1300"), cpsr=0)
        self.assertEqual(1, single.count_branch_input_sources(0x1102, 1))

        coupled = DynamicExpressionGraph()
        coupled.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        coupled.observe_instruction(instruction(0x1100, "MOV", "r2, r0"))
        coupled.observe_external_read(
            kind="mmio",
            read_pc=0x1004,
            address=0x40000004,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r3",
        )
        coupled.observe_instruction(instruction(0x1104, "AND", "r2, r2, r3"))
        coupled.observe_instruction(instruction(0x1200, "CMP", "r2, #3"))
        coupled.observe_instruction(instruction(0x1202, "BEQ", "0x1300"), cpsr=0)
        self.assertEqual(2, coupled.count_branch_input_sources(0x1202, 1))


if __name__ == "__main__":
    unittest.main()
