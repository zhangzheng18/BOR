#!/usr/bin/env python3
"""Smoke-test replay state fingerprints."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.replay_fingerprint import (
    _ARM_REG_NAME_TO_CONST,
    _jsonable,
    build_replay_state_fingerprint,
)


class DummyUC:
    def __init__(self, memory, registers=None):
        self.memory = dict(memory)
        self.registers = dict(registers or {})

    def mem_read(self, address, size):
        return self.memory.get((int(address), int(size)), b"\x00" * int(size))

    def reg_read(self, reg_id):
        return self.registers.get(reg_id, 0)


class DummyMMIO:
    def __init__(self, state):
        self.mmio_state = dict(state)


class DummyState:
    def __init__(self, value):
        self.current_value = value


class DummyStatefulMMIO:
    def __init__(self, state):
        self.mmio_states = {
            int(address): DummyState(value)
            for address, value in state.items()
        }


class ReplayFingerprintContractTests(unittest.TestCase):
    def test_paired_replays_share_pure_initial_fingerprint(self):
        pc_reg = _ARM_REG_NAME_TO_CONST.get("pc")
        uc = DummyUC(
            {(0x20001000, 16): b"abcdefghijklmnop"},
            {pc_reg: 0x08001010},
        )
        control_mmio = DummyMMIO({0x40021004: 0x8})
        candidate_mmio = DummyMMIO({0x40021004: 0x8})
        candidate_mmio.runtime_constraints = {("pc", 0x40021004): 0x2}
        candidate_mmio.runtime_occurrence_constraints = {
            (0x08001010, 0x40021004, 2): 0x2
        }
        control = type("E", (), {"runtime_written_pages": {0x20001000}})()
        candidate = type(
            "E",
            (),
            {
                "runtime_written_pages": {0x20001000},
                "memory_occurrence_constraints": {(0x08001010, 0x20001000, 1): 0x41},
                "forced_branch_choices": {0x08001020: True},
                "forced_branch_sequence": [(0x08001024, False)],
            },
        )()

        control_initial = build_replay_state_fingerprint(
            uc=uc,
            bb_addr=0x08001000,
            mmio_handler=control_mmio,
            emulator=control,
            constraint_hit_keys={("pc", 0x08001000)},
            include_candidate_configuration=False,
            include_observed_constraint_hits=False,
        )
        candidate_initial = build_replay_state_fingerprint(
            uc=uc,
            bb_addr=0x08001000,
            mmio_handler=candidate_mmio,
            emulator=candidate,
            constraint_hit_keys={("pc", 0x08001000), ("pc", 0x08001001)},
            include_candidate_configuration=False,
            include_observed_constraint_hits=False,
        )
        self.assertEqual(
            control_initial["fingerprint_sha256"],
            candidate_initial["fingerprint_sha256"],
        )

        control_audit = build_replay_state_fingerprint(
            uc=uc,
            bb_addr=0x08001000,
            mmio_handler=control_mmio,
            emulator=control,
            constraint_hit_keys={("pc", 0x08001000)},
            include_candidate_configuration=True,
            include_observed_constraint_hits=True,
        )
        candidate_audit = build_replay_state_fingerprint(
            uc=uc,
            bb_addr=0x08001000,
            mmio_handler=candidate_mmio,
            emulator=candidate,
            constraint_hit_keys={("pc", 0x08001000), ("pc", 0x08001001)},
            include_candidate_configuration=True,
            include_observed_constraint_hits=True,
        )
        self.assertNotEqual(
            control_audit["candidate_configuration"],
            candidate_audit["candidate_configuration"],
        )
        self.assertNotEqual(
            control_audit["constraint_hits"],
            candidate_audit["constraint_hits"],
        )

    def test_set_normalization_is_stable(self):
        self.assertEqual(_jsonable({3, 1, 2}), [1, 2, 3])


def main() -> int:
    pc_reg = _ARM_REG_NAME_TO_CONST.get("pc")
    r0_reg = _ARM_REG_NAME_TO_CONST.get("r0")
    uc = DummyUC(
        {
            (0x20001000, 16): b"abcdefghijklmnop",
        },
        {
            pc_reg: 0x08001010,
            r0_reg: 0x12345678,
        },
    )
    mmio = DummyMMIO({
        0x40021004: 0x8,
    })
    emulator = type("E", (), {"runtime_written_pages": {0x20001000}, "stream_input_state": {"len": 4}})()
    fp1 = build_replay_state_fingerprint(
        uc=uc,
        bb_addr=0x08001000,
        mmio_handler=mmio,
        emulator=emulator,
        reg_values={"r0": 1, "r1": 2, "pc": 0x08001010},
        constraint_hit_keys=[("pc", 0x08001000)],
        call_stack=[{"call_pc": 0x08000010, "return_pc": 0x08000020, "depth": 1}],
        max_dirty_pages=4,
        ram_sample_bytes=16,
    )
    fp2 = build_replay_state_fingerprint(
        uc=uc,
        bb_addr=0x08001000,
        mmio_handler=mmio,
        emulator=emulator,
        reg_values={"r0": 1, "r1": 2, "pc": 0x08001010},
        constraint_hit_keys=[("pc", 0x08001000)],
        call_stack=[{"call_pc": 0x08000010, "return_pc": 0x08000020, "depth": 1}],
        max_dirty_pages=4,
        ram_sample_bytes=16,
    )
    mmio_changed = DummyMMIO({
        0x40021004: 0x4,
    })
    fp3 = build_replay_state_fingerprint(
        uc=uc,
        bb_addr=0x08001000,
        mmio_handler=mmio_changed,
        emulator=emulator,
        reg_values={"r0": 1, "r1": 2, "pc": 0x08001010},
        constraint_hit_keys=[("pc", 0x08001000)],
        call_stack=[{"call_pc": 0x08000010, "return_pc": 0x08000020, "depth": 1}],
        max_dirty_pages=4,
        ram_sample_bytes=16,
    )
    fp4 = build_replay_state_fingerprint(
        uc=uc,
        bb_addr=0x08001000,
        mmio_handler=mmio,
        emulator=emulator,
        reg_ids=("r0", "pc"),
        max_dirty_pages=4,
        ram_sample_bytes=16,
    )
    stateful_fp = build_replay_state_fingerprint(
        uc=uc,
        bb_addr=0x08001000,
        mmio_handler=DummyStatefulMMIO({0x40021004: 0x8}),
        emulator=emulator,
        reg_values={"r0": 1, "r1": 2, "pc": 0x08001010},
        max_dirty_pages=4,
        ram_sample_bytes=16,
    )
    checks = {
        "stable_ok": fp1["fingerprint_sha256"] == fp2["fingerprint_sha256"],
        "sensitive_ok": fp1["fingerprint_sha256"] != fp3["fingerprint_sha256"],
        "dirty_pages_ok": len(fp1["dirty_pages"]) == 1,
        "mmio_state_ok": fp1["mmio_state"]["0x40021004"] == "0x00000008",
        "string_register_read_ok": fp4["regs"]["r0"] == 0x12345678 and fp4["regs"]["pc"] == 0x08001010,
        "stateful_mmio_read_ok": stateful_fp["mmio_state"]["0x40021004"] == "0x00000008",
    }
    print(json.dumps({"fp1": fp1, "fp3": fp3, "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    # Keep the historical standalone diagnostic output while allowing the
    # repository unittest discovery to execute the contract tests above.
    diagnostic_status = main()
    test_status = unittest.main(exit=False).result.wasSuccessful()
    raise SystemExit(0 if diagnostic_status == 0 and test_status else 1)
