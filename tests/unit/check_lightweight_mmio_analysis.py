#!/usr/bin/env python3
"""Regression tests for lightweight MMIO static analysis and indexed preload."""

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
from lsgemu.analysis.lightweight_mmio_analysis import LightweightMMIOAnalyzer
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R2


CONSOLE_ELF = os.environ.get(
    "LSGEMU_TEST_CONSOLE_ELF",
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Console/Console.elf",
)


def main() -> int:
    static_bbs = {
        0x1000: [
            {"address": 0x1000, "mnemonic": "MOVW", "operands": "r0, #0x4000", "size": 4},
            {"address": 0x1004, "mnemonic": "MOVT", "operands": "r0, #0x4006", "size": 4},
            {"address": 0x1008, "mnemonic": "LDR", "operands": "r1, [r0, #0x10]", "size": 2},
        ],
        0x2000: [
            {"address": 0x2000, "mnemonic": "LDR", "operands": "r0, =0x40064000", "size": 4},
            {"address": 0x2004, "mnemonic": "MOVS", "operands": "r2, #3", "size": 2},
            {"address": 0x2006, "mnemonic": "LDRB", "operands": "r3, [r0, r2, LSL #1]", "size": 2},
        ],
        0x3000: [
            {"address": 0x3000, "mnemonic": "MOVW", "operands": "r0, #0x2000", "size": 4},
            {"address": 0x3004, "mnemonic": "MOVT", "operands": "r0, #0x2000", "size": 4},
            {"address": 0x3008, "mnemonic": "LDR", "operands": "r1, [r0]", "size": 2},
        ],
        0x4000: [
            {"address": 0x4000, "mnemonic": "LDR", "operands": "r0, =0x40065000", "size": 4},
            {"address": 0x4004, "mnemonic": "LDRH", "operands": "r4, [r0, r2, LSL #2]", "size": 2},
        ],
        0x6000: [
            {"address": 0x6000, "mnemonic": "LDR", "operands": "r3, [0x00007000]", "size": 2},
            {"address": 0x6002, "mnemonic": "LDRB", "operands": "r3, [r3, #6]", "size": 2},
        ],
    }
    class LiteralAnalyzer(LightweightMMIOAnalyzer):
        def _read_image_bytes(self, address, size):
            if int(address) == 0x7000 and int(size) == 4:
                return (0x40064000).to_bytes(4, "little")
            return super()._read_image_bytes(address, size)

    analyzer = LiteralAnalyzer(static_bbs, symbols_by_addr={0x1000: "init_periph"})
    accesses = analyzer.analyze()
    exact_addresses = {access.address for access in accesses if access.address is not None}
    unresolved = [access for access in accesses if access.address is None]
    summaries = analyzer.build_function_summaries(accesses)

    emulator = IntelligentEmulator(
        firmware_path=CONSOLE_ELF,
        mmio_constraints={},
        static_bbs={
            0x5000: [
                {"address": 0x5000, "mnemonic": "LDRB", "operands": "r3, [r0, r2, LSL #1]", "size": 2},
            ]
        },
        static_mmio_accesses=[access.to_dict() for access in accesses],
        function_mmio_summaries={entry: summary.to_dict() for entry, summary in summaries.items()},
        constraint_json_path=None,
        max_snapshots=5,
        llm_config_path=None,
    )
    emulator.uc.reg_write(UC_ARM_REG_R0, 0x40064000)
    emulator.uc.reg_write(UC_ARM_REG_R2, 3)
    emulator.uc.reg_write(UC_ARM_REG_PC, 0x5000 | 1)
    emulator._mapped_mmio_preload_hook(emulator.uc, 0x5000, 2, None)

    checks = {
        "movw_movt_exact": 0x40064010 in exact_addresses,
        "pseudo_ldr_index_exact": 0x40064006 in exact_addresses,
        "absolute_literal_pool_exact": any(access.pc == 0x6002 and access.address == 0x40064006 for access in accesses),
        "non_mmio_rejected": 0x20002000 not in exact_addresses,
        "unknown_index_recorded": bool(unresolved) and unresolved[0].kind == "unknown_index_mmio_base",
        "summary_effectful": any(summary.reads or summary.unresolved_reads for summary in summaries.values()),
        "indexed_preload_applied": emulator.mapped_mmio_preload_stats["applied"] == 1,
        "indexed_preload_counted": emulator.mapped_mmio_preload_stats["indexed_checked"] == 1,
        "indexed_history": any(item[2] is True and int(item[1]) == 0x40064006 for item in emulator.mmio_access_history),
        "prediction_stats": emulator.static_mmio_prediction_stats["total_accesses"] >= 3,
    }
    print(json.dumps({
        "accesses": [access.to_dict() for access in accesses],
        "summaries": {hex(entry): summary.to_dict() for entry, summary in summaries.items()},
        "preload_stats": emulator.mapped_mmio_preload_stats,
        "checks": checks,
    }, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
