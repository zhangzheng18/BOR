#!/usr/bin/env python3
"""Smoke-test generic Cortex-M IRQ event inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.isr_explorer.isr_explorer import ISRExplorer


class DummyUC:
    def __init__(self):
        self.memory = {}

    def mem_read(self, address, size):
        return self.memory.get(int(address), b"\x00" * int(size))


class DummyMMIO:
    def __init__(self):
        self.mmio_state = {
            0xE000E100: 1 << 5,
            0xE000E200: 1 << 5,
            0x40000010: 0x1,
        }


def main() -> int:
    uc = DummyUC()
    # Vector index 16+5 points to IRQ5 handler at 0x08001001.
    uc.memory[0x08000000 + (16 + 5) * 4] = (0x08001001).to_bytes(4, "little")
    static_bbs = {0x08001000: [{"address": 0x08001000, "mnemonic": "BX", "operands": "LR"}]}
    explorer = ISRExplorer(
        uc,
        static_bbs,
        vtor=0x08000000,
        mmio_handler=DummyMMIO(),
        instruction_to_bb={0x08001000: 0x08001000},
        broad_interrupt_flags=False,
        code_ranges=[(0x08000000, 0x08100000)],
    )
    candidates = explorer.learned_irq_candidates()
    stats = explorer.get_statistics()
    checks = {
        "irq5_present_ok": bool(candidates) and candidates[0]["irq"] == 5,
        "null_vectors_ignored_ok": all(address != 0 for address in explorer.isr_addresses.values()),
        "enabled_ok": 5 in stats["nvic_enabled_irqs"],
        "pending_ok": 5 in stats["nvic_pending_irqs"],
        "candidate_reason_ok": "nvic_pending" in candidates[0]["reasons"] and "nvic_enabled" in candidates[0]["reasons"],
    }
    print(json.dumps({"candidates": candidates, "stats": stats, "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
