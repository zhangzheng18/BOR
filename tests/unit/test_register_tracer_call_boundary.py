#!/usr/bin/env python3
"""Contracts for concrete-value-aware provenance across function calls."""

from __future__ import annotations

import unittest

from unicorn.arm_const import (
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_R3,
)

from lsgemu.register_tracer.register_tracer import RegisterSource, RegisterTracer


class _RegisterUC:
    def __init__(self) -> None:
        self.values = {
            UC_ARM_REG_R0: 1,
            UC_ARM_REG_R1: 2,
            UC_ARM_REG_R2: 3,
            UC_ARM_REG_R3: 4,
        }

    def reg_read(self, register):
        return self.values.get(register, 0)


def _source(register: str, address: int) -> RegisterSource:
    return RegisterSource(
        register=register,
        source_type="memory",
        source_pc=0x08000100,
        memory_address=address,
        operation="load",
        chain=(f"RAM[0x{address:08x}]",),
    )


class RegisterTracerCallBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uc = _RegisterUC()
        self.call = {
            "address": 0x08001000,
            "mnemonic": "BL",
            "operands": "0x08002001",
            "size": 4,
        }
        self.tracer = RegisterTracer(
            self.uc,
            {0x08001000: [self.call]},
            dynamic_graph_enabled=False,
        )

    def test_changed_return_preserves_only_opaque_argument_dependencies(self):
        r0_source = _source("r0", 0x20000000)
        r1_source = _source("r1", 0x20000004)
        self.tracer.register_sources.update({"r0": r0_source, "r1": r1_source})
        self.tracer._record_call_frame(0x08001000, 4, self.call)

        self.uc.values[UC_ARM_REG_R0] = 9
        self.tracer._retire_call_frames(0x08001004)

        returned = self.tracer.register_sources["r0"]
        self.assertTrue(returned.opaque_dependency)
        self.assertFalse(returned.causal_complete)
        self.assertEqual("opaque_call_return", returned.operation)
        self.assertIs(r1_source, self.tracer.register_sources["r1"])
        self.assertEqual(
            1,
            self.tracer.provenance_kill_stats["opaque_call_dependency_preserved"],
        )
        frame = self.tracer.completed_call_frames[-1]
        self.assertEqual(["r0"], frame["concrete_clobbered_registers"])
        self.assertEqual([], frame["stale_provenance_cleared"])
        self.assertIsNone(frame["return_propagated_to"])
        self.assertEqual(0x08002000, frame["callee_target"])
        self.assertTrue(frame["callee_target_statically_resolved"])

    def test_callee_generated_source_survives_concrete_clobber(self):
        old_source = _source("r0", 0x20000000)
        new_source = _source("r0", 0x20000100)
        self.tracer.register_sources["r0"] = old_source
        self.tracer._record_call_frame(0x08001000, 4, self.call)

        self.uc.values[UC_ARM_REG_R0] = 7
        self.tracer.register_sources["r0"] = new_source
        self.tracer._retire_call_frames(0x08001004)

        self.assertIs(new_source, self.tracer.register_sources["r0"])
        self.assertEqual([], self.tracer.completed_call_frames[-1]["stale_provenance_cleared"])
        self.assertEqual(0, self.tracer.provenance_kill_stats["callee_concrete_clobber"])

    def test_unchanged_r0_preserves_legacy_identity_return_fallback(self):
        source = _source("r0", 0x20000000)
        self.tracer.register_sources["r0"] = source
        self.tracer._record_call_frame(0x08001000, 4, self.call)
        self.tracer.register_sources.pop("r0")

        self.tracer._retire_call_frames(0x08001004)

        returned = self.tracer.register_sources["r0"]
        self.assertEqual("call_return", returned.operation)
        self.assertEqual(0x20000000, returned.memory_address)
        self.assertEqual("r0", self.tracer.completed_call_frames[-1]["return_propagated_to"])

    def test_blx_register_target_is_concrete_but_not_statically_resolved(self):
        self.uc.values[UC_ARM_REG_R3] = 0x08003001
        instruction = {
            "address": 0x08001100,
            "mnemonic": "BLX",
            "operands": "r3",
            "size": 2,
        }

        self.tracer._record_call_frame(0x08001100, 2, instruction)
        frame = self.tracer.call_stack[-1]

        self.assertEqual(0x08003000, frame["callee_target"])
        self.assertEqual("register_concrete", frame["callee_target_kind"])
        self.assertEqual("r3", frame["callee_target_register"])
        self.assertFalse(frame["callee_target_statically_resolved"])


if __name__ == "__main__":
    unittest.main()
