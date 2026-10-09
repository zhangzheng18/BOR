#!/usr/bin/env python3
"""Regression tests for cross-BB dynamic input constraint recovery."""

import unittest

from lsgemu.dynamic_constraint_recovery import DynamicExpressionGraph
from lsgemu.runner_models import external_input_site_identity


def instruction(address, mnemonic, operands):
    return {
        "address": int(address),
        "mnemonic": str(mnemonic),
        "operands": str(operands),
        "size": 2,
    }


class DynamicConstraintRecoveryTests(unittest.TestCase):
    def test_solves_two_occurrences_across_basic_blocks(self):
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
        graph.observe_instruction(instruction(0x1100, "LSL", "r2, r0, #8"))
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x1000,
            address=0x40000000,
            occurrence=2,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(instruction(0x1102, "ORR", "r2, r2, r0"))
        graph.observe_instruction(instruction(0x1200, "CMP", "r2, #0x4142"))
        graph.observe_instruction(instruction(0x1202, "BEQ", "0x1300"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x1202,
            occurrence=1,
            target_taken=True,
            max_models=2,
        )
        self.assertEqual("dynamic_models", result.reason)
        self.assertEqual("multibyte_assembly", result.relation_kind)
        values = {
            item.occurrence: item.value
            for item in result.models[0].assignments
        }
        self.assertEqual({1: 0x41, 2: 0x42}, values)

    def test_symbolic_bytes_survive_ram_store_and_load(self):
        graph = DynamicExpressionGraph(
            instruction_to_bb={0x2000: 0x2000, 0x2100: 0x2100, 0x2102: 0x2100}
        )
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x2000,
            address=0x40001000,
            occurrence=3,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_memory_store(
            write_pc=0x2002,
            address=0x20000120,
            size=1,
            source_register="r0",
        )
        graph.register_nodes.clear()
        graph.observe_memory_load(
            read_pc=0x2100,
            address=0x20000120,
            size=1,
            observed_value=0,
            destination_register="r3",
        )
        graph.observe_instruction(instruction(0x2100, "CMP", "r3, #0x7e"))
        graph.observe_instruction(instruction(0x2102, "BEQ", "0x2200"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x2102,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(1, len(result.models))
        assignment = result.models[0].assignments[0]
        self.assertEqual(3, assignment.occurrence)
        self.assertEqual(0x7E, assignment.value)
        self.assertEqual("cross_basic_block", result.relation_kind)

    def test_input_dependent_pointer_is_preserved_as_opaque_dependency(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x2300,
            address=0x40003000,
            occurrence=1,
            size=4,
            observed_value=0x20001000,
            destination_register="r1",
        )
        graph.observe_memory_load(
            read_pc=0x2302,
            address=0x20001000,
            size=1,
            observed_value=0x10,
            destination_register="r0",
            address_registers=("r1",),
        )
        graph.observe_instruction(instruction(0x2304, "CMP", "r0, #0x20"))
        graph.observe_instruction(instruction(0x2306, "BEQ", "0x2400"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x2306,
            occurrence=1,
            target_taken=True,
        )
        self.assertFalse(result.models)
        self.assertEqual("unsupported", result.solver_status)
        self.assertEqual(1, result.target_inputs)
        self.assertEqual(1, graph.stats["opaque_indirect_memory_loads"])

    def test_shared_input_prefix_predicate_is_preserved(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x3000,
            address=0x40002000,
            occurrence=1,
            size=1,
            observed_value=1,
            destination_register="r0",
        )
        graph.observe_instruction(instruction(0x3010, "CMP", "r0, #1"))
        graph.observe_instruction(
            instruction(0x3012, "BEQ", "0x3020"),
            cpsr=(1 << 30),
        )
        graph.observe_instruction(instruction(0x3020, "CMP", "r0, #2"))
        graph.observe_instruction(instruction(0x3022, "BEQ", "0x3030"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x3022,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual("dynamic_predicate_unsat_or_timeout", result.reason)
        self.assertFalse(result.models)
        self.assertEqual(1, result.path_predicates)

    def test_dynamic_model_repairs_current_prefix_baseline(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x3000,
            address=0x40002000,
            occurrence=1,
            size=1,
            observed_value=1,
            destination_register="r0",
        )
        graph.observe_instruction(instruction(0x3010, "CMP", "r0, #1"))
        graph.observe_instruction(
            instruction(0x3012, "BEQ", "0x3020"),
            cpsr=(1 << 30),
        )
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x3002,
            address=0x40002004,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r1",
        )
        graph.observe_instruction(instruction(0x3020, "ADD", "r2, r0, r1"))
        graph.observe_instruction(instruction(0x3022, "CMP", "r2, #3"))
        graph.observe_instruction(instruction(0x3024, "BEQ", "0x3030"), cpsr=0)

        first_site = external_input_site_identity(
            constraint_type="mmio",
            address=0x40002000,
            read_pc=0x3000,
            read_occurrence=1,
            input_kind="mmio",
        )
        result = graph.recover_branch_inputs(
            branch_pc=0x3024,
            occurrence=1,
            target_taken=True,
            baseline_values_by_site={first_site: 9},
        )
        self.assertEqual("dynamic_models", result.reason)
        assignments = {
            item.address: item.value
            for item in result.models[0].assignments
        }
        self.assertEqual(1, assignments[0x40002000])
        self.assertEqual(2, assignments[0x40002004])
        self.assertEqual(1, result.models[0].baseline_corrections)

    def test_dynamic_failed_value_avoidance_uses_runtime_site_identity(self):
        graph = DynamicExpressionGraph()
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x6000,
            address=0x40005000,
            occurrence=1,
            size=1,
            observed_value=0,
            destination_register="r0",
        )
        graph.observe_instruction(instruction(0x6010, "CMP", "r0, #2"))
        graph.observe_instruction(instruction(0x6012, "BEQ", "0x6020"), cpsr=0)
        site = external_input_site_identity(
            constraint_type="mmio",
            address=0x40005000,
            read_pc=0x6000,
            read_occurrence=1,
            input_kind="mmio",
        )
        result = graph.recover_branch_inputs(
            branch_pc=0x6012,
            occurrence=1,
            target_taken=True,
            avoid_values_by_site={site: {2}},
        )
        self.assertFalse(result.models)
        self.assertEqual("dynamic_predicate_unsat_or_timeout", result.reason)

    def test_repairs_xor_checksum_field_without_changing_payload(self):
        graph = DynamicExpressionGraph()
        payload = (1, 2, 4)
        for occurrence, value in enumerate(payload, start=1):
            graph.observe_external_read(
                kind="mmio",
                read_pc=0x4000,
                address=0x40003000,
                occurrence=occurrence,
                size=1,
                observed_value=value,
                destination_register="r0",
            )
            if occurrence == 1:
                graph.observe_instruction(instruction(0x4100, "MOV", "r2, r0"))
            else:
                graph.observe_instruction(instruction(0x4102, "EOR", "r2, r2, r0"))
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x4000,
            address=0x40003000,
            occurrence=4,
            size=1,
            observed_value=0,
            destination_register="r1",
        )
        graph.observe_instruction(instruction(0x4200, "CMP", "r2, r1"))
        graph.observe_instruction(instruction(0x4202, "BEQ", "0x4300"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x4202,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual("xor_checksum", result.relation_kind)
        assignments = {
            item.occurrence: item
            for item in result.models[0].assignments
        }
        self.assertEqual({4}, set(assignments))
        self.assertEqual(1 ^ 2 ^ 4, assignments[4].value)
        self.assertEqual("dynamic_ssa_checksum_repair", result.models[0].strategy)

    def test_long_checksum_slice_emits_only_checksum_field(self):
        graph = DynamicExpressionGraph()
        checksum = 0
        for occurrence in range(1, 81):
            value = (occurrence * 7) & 0xFF
            checksum ^= value
            graph.observe_external_read(
                kind="mmio",
                read_pc=0x5000,
                address=0x40004000,
                occurrence=occurrence,
                size=1,
                observed_value=value,
                destination_register="r0",
            )
            if occurrence == 1:
                graph.observe_instruction(instruction(0x5100, "MOV", "r2, r0"))
            else:
                graph.observe_instruction(instruction(0x5102, "EOR", "r2, r2, r0"))
        graph.observe_external_read(
            kind="mmio",
            read_pc=0x5000,
            address=0x40004000,
            occurrence=81,
            size=1,
            observed_value=0,
            destination_register="r1",
        )
        graph.observe_instruction(instruction(0x5200, "CMP", "r2, r1"))
        graph.observe_instruction(instruction(0x5202, "BEQ", "0x5300"), cpsr=0)

        result = graph.recover_branch_inputs(
            branch_pc=0x5202,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(81, result.target_inputs)
        self.assertEqual(1, len(result.models[0].assignments))
        assignment = result.models[0].assignments[0]
        self.assertEqual(81, assignment.occurrence)
        self.assertEqual(checksum, assignment.value)


if __name__ == "__main__":
    unittest.main()
