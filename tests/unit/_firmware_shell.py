"""Minimal runner shell shared by unit tests (extracted from an internal test module)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import HistoricalRunner

FLASH_BASE = 0x08000000


def _firmware_shell(static_bbs, compare_lookup=None, successors=None, literals: dict = None):
    """最小 runner shell：只装精确分析所需的静态视图与固件字读取。

    ``instruction_to_bb`` 从 static_bbs 自动推导（每条指令的 pc 都要能定位 BB）。
    """
    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.precise_branch_mmio_cache = {}
    instruction_to_bb = {
        int(insn["address"]): bb for bb, insns in static_bbs.items() for insn in insns
    }
    runner.prepared = SimpleNamespace(
        instruction_to_bb=instruction_to_bb,
        static_bbs=dict(static_bbs),
        compare_lookup=dict(compare_lookup or {}),
        static_successors=dict(successors or {}),
    )
    literals = dict(literals or {})
    if literals:

        def _segments():
            high = max(literals) + 4
            data = bytearray(max(0x100, high - FLASH_BASE + 0x100))
            for addr, value in literals.items():
                data[addr - FLASH_BASE: addr - FLASH_BASE + 4] = int(value).to_bytes(
                    4, "little"
                )
            return [(FLASH_BASE, bytes(data))]

        runner._load_firmware_segments = _segments
    else:
        runner._load_firmware_segments = lambda: []
    return runner
