#!/usr/bin/env python3
"""Regression test: mapped-MMIO preload must still be visible to loop analysis."""

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
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R3


CONSOLE_ELF = os.environ.get(
    "LSGEMU_TEST_CONSOLE_ELF",
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Console/Console.elf",
)


def main() -> int:
    static_bbs = {
        0x1C36: [
            {"address": 0x1C36, "mnemonic": "LDRB", "operands": "r3, [r3, #6]", "size": 2},
            {"address": 0x1C38, "mnemonic": "TST", "operands": "r3, #0x10", "size": 2},
            {"address": 0x1C3A, "mnemonic": "BEQ", "operands": "0x00001c36", "size": 2},
        ],
    }
    emulator = IntelligentEmulator(
        firmware_path=CONSOLE_ELF,
        mmio_constraints={},
        static_bbs=static_bbs,
        constraint_json_path=None,
        max_snapshots=5,
        llm_config_path=None,
    )
    emulator.uc.reg_write(UC_ARM_REG_R3, 0x40064000)
    emulator.uc.reg_write(UC_ARM_REG_PC, 0x1C36 | 1)
    emulator._mapped_mmio_preload_hook(emulator.uc, 0x1C36, 2, None)

    mmio_reads = [
        item for item in emulator.mmio_access_history
        if item[2] is True and int(item[1]) == 0x40064006
    ]
    classifier_reads = [
        item for item in emulator.loop_classifier.mmio_accesses.get(0x1C36, [])
        if item[1] is True and int(item[0]) == 0x40064006
    ]
    checks = {
        "preload_applied": emulator.mapped_mmio_preload_stats["applied"] == 1,
        "history_observed": bool(mmio_reads),
        "classifier_observed": bool(classifier_reads),
    }
    print(json.dumps({
        "mmio_access_history": emulator.mmio_access_history,
        "classifier": emulator.loop_classifier.mmio_accesses,
        "checks": checks,
    }, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
