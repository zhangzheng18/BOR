#!/usr/bin/env python3
"""Regression test for contextual ISR injection with an unmapped SP."""

from __future__ import annotations

from pathlib import Path
import json
import sys

from unicorn import Uc, UC_ARCH_ARM, UC_MODE_MCLASS, UC_MODE_THUMB, UC_PROT_ALL
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_SP

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.isr_explorer.isr_explorer import ISRExplorer


def main() -> int:
    uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
    uc.mem_map(0x08000000, 0x1000, UC_PROT_ALL)
    uc.mem_write(0x08000000, (0).to_bytes(4, "little"))
    uc.mem_write(0x08000004, (0x08000101).to_bytes(4, "little"))
    uc.mem_write(0x08000000 + (16 + 3) * 4, (0x08000201).to_bytes(4, "little"))
    uc.reg_write(UC_ARM_REG_SP, 0)
    uc.reg_write(UC_ARM_REG_PC, 0x08000101)

    static_bbs = {0x08000200: [{"address": 0x08000200, "mnemonic": "BX", "operands": "LR"}]}
    explorer = ISRExplorer(
        uc,
        static_bbs,
        vtor=0x08000000,
        instruction_to_bb={0x08000200: 0x08000200},
        code_ranges=[(0x08000000, 0x08001000)],
    )
    explorer._setup_isr_context(0x08000200, 3, preserve_registers=True, return_pc=0x08000100)

    sp = uc.reg_read(UC_ARM_REG_SP)
    frame = bytes(uc.mem_read(sp, 32))
    stats = explorer.get_statistics()["isr_stack"]
    checks = {
        "sp_repaired": 0x20000000 <= sp < 0x20010000,
        "frame_written": len(frame) == 32 and int.from_bytes(frame[24:28], "little") == 0x08000100,
        "stats_repaired": stats.get("stack_frame_repaired", 0) == 1,
        "no_push_failure": stats.get("stack_push_failures", 0) == 0,
    }
    print(json.dumps({"sp": f"0x{sp:08x}", "stats": stats, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
