#!/usr/bin/env python3
"""
Smoke-test local branch semantic validation for analysis-produced constraints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from unicorn.arm_const import UC_ARM_REG_R2, UC_ARM_REG_R4


GATEWAY_ELF = os.environ.get(
    "LSGEMU_TEST_GATEWAY_ELF",
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Gateway/Gateway.elf",
)


def build_emulator(static_bbs):
    emulator = IntelligentEmulator(
        firmware_path=GATEWAY_ELF,
        mmio_constraints={},
        static_bbs=static_bbs,
        constraint_json_path=None,
        max_snapshots=5,
        llm_config_path=None,
    )
    return emulator


def main() -> int:
    cases = []

    cmp_static = {
        0x08001000: [
            {"address": 0x08001000, "mnemonic": "MOVS", "operands": "R1, #0x20", "size": 2},
            {"address": 0x08001002, "mnemonic": "LDR", "operands": "R0, [R4, #0x04]", "size": 2},
            {"address": 0x08001004, "mnemonic": "CMP", "operands": "R0, #0x20", "size": 2},
            {"address": 0x08001006, "mnemonic": "BEQ", "operands": "0x08001020", "size": 2},
        ]
    }
    emulator = build_emulator(cmp_static)
    emulator.uc.reg_write(UC_ARM_REG_R4, 0x20000000)
    emulator.branch_snapshot_manager.save_snapshot(
        emulator.uc,
        0x08001000,
        0x08001020,
        0x08001008,
        "EQ",
        original_taken=False,
        depth=0,
        mmio_state={},
    )
    cmp_ok = {
        "type": "memory",
        "read_pc": 0x08001002,
        "address": 0x20000004,
        "value": 0x20,
        "constraint_pc": 0x08001004,
    }
    cmp_bad = dict(cmp_ok, value=0x21)
    accepted_ok, reason_ok = emulator._validate_analysis_constraint(cmp_ok, 0x08001006, True)
    accepted_bad, reason_bad = emulator._validate_analysis_constraint(cmp_bad, 0x08001006, True)
    cases.append({
        "name": "cmp_beq",
        "accepted_ok": accepted_ok,
        "accepted_bad": accepted_bad,
        "reason_ok": reason_ok,
        "reason_bad": reason_bad,
        "ok": accepted_ok is True and accepted_bad is False,
    })

    tst_static = {
        0x08002000: [
            {"address": 0x08002000, "mnemonic": "LDR", "operands": "R3, [R2, #0x10]", "size": 2},
            {"address": 0x08002002, "mnemonic": "TST", "operands": "R3, #0x2000000", "size": 2},
            {"address": 0x08002004, "mnemonic": "BNE", "operands": "0x08002020", "size": 2},
        ]
    }
    emulator = build_emulator(tst_static)
    emulator.uc.reg_write(UC_ARM_REG_R2, 0x40021000)
    emulator.branch_snapshot_manager.save_snapshot(
        emulator.uc,
        0x08002000,
        0x08002020,
        0x08002006,
        "NE",
        original_taken=False,
        depth=0,
        mmio_state={0x40021010: 0},
    )
    tst_ok = {
        "type": "mmio",
        "read_pc": 0x08002000,
        "address": 0x40021010,
        "value": 0x02000000,
        "constraint_pc": 0x08002002,
    }
    tst_bad = dict(tst_ok, value=0x0)
    accepted_ok, reason_ok = emulator._validate_analysis_constraint(tst_ok, 0x08002004, True)
    accepted_bad, reason_bad = emulator._validate_analysis_constraint(tst_bad, 0x08002004, True)
    cases.append({
        "name": "tst_bne",
        "accepted_ok": accepted_ok,
        "accepted_bad": accepted_bad,
        "reason_ok": reason_ok,
        "reason_bad": reason_bad,
        "ok": accepted_ok is True and accepted_bad is False,
    })

    print(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))
    return 0 if all(case["ok"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
