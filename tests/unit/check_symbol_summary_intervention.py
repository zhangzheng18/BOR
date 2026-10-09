#!/usr/bin/env python3
"""Regression test for summary-returning config/API functions during intervention."""

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
from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0


CONSOLE_ELF = os.environ.get(
    "LSGEMU_TEST_CONSOLE_ELF",
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Console/Console.elf",
)


def main() -> int:
    static_bbs = {
        0x24DC: [
            {"address": 0x24DC, "mnemonic": "PUSH", "operands": "{r7}", "size": 2},
            {"address": 0x24DE, "mnemonic": "MOV", "operands": "r7, sp", "size": 2},
        ],
        0x2562: [
            {"address": 0x2562, "mnemonic": "ADDS", "operands": "r3, #1", "size": 2},
            {"address": 0x2564, "mnemonic": "CMP", "operands": "r3, #101", "size": 2},
            {"address": 0x2566, "mnemonic": "BLS", "operands": "0x2556", "size": 2},
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
    emulator.symbols_by_addr = {0x24DC: "NVIC_SetPriority"}
    emulator.uc.reg_write(UC_ARM_REG_LR, 0x2563)
    emulator.uc.reg_write(UC_ARM_REG_R0, 0xFFFFFFFF)

    calls = {"mmio_inferencer": 0}

    def fail_if_called(*_args, **_kwargs):
        calls["mmio_inferencer"] += 1
        raise AssertionError("MMIO inferencer must not run for summaryable config function")

    emulator.mmio_inferencer.analyze_polling_loop = fail_if_called
    emulator._handle_intervention(0x24DC)

    pc = int(emulator.uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
    r0 = int(emulator.uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
    checks = {
        "returned_to_lr": (pc & ~1) == 0x2562,
        "thumb_mode_inferred": emulator.execution_thumb is True,
        "r0_zero": r0 == 0,
        "skip_stat_applied": int(emulator.skip_function_stats.get("applied", 0) or 0) == 1,
        "mmio_inferencer_not_called": calls["mmio_inferencer"] == 0,
    }
    print(json.dumps({"pc": f"0x{pc:08x}", "r0": f"0x{r0:08x}", "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
