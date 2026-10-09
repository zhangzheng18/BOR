"""Reusable fixtures for DFS-flip unit tests (image builder + emulator builder)."""


from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import tempfile

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.dfs_anchor_pool import DFSAnchorEntry, DFSAnchorPool
from lsgemu.dfs_flip import (
    DFSFlipGuardrails,
    DFSFlipLedger,
    DFSFlipPlanner,
    DFSFlipTask,
    flip_edge_evidence,
    verify_flip_by_real_execution,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.runner_models import BranchConstraintCandidate
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MMIO_ADDR = 0x40000000

LOAD_BASE = 0x08004000

ENTRY_PC = 0x08004100

FLIP_FRAGMENT = bytes.fromhex(
    # 注意：Thumb 指令按小端字节序书写（指令 0x4806 = 字节 06 48）。
    "0648"      # ldr r0, [pc, #0x18]
    "0068"      # ldr r0, [r0]
    "2028"      # cmp r0, #0x20
    "05d0"      # beq 0x08004114
    "0121"      # movs r1, #1     (B)
    "04e0"      # b 0x08004116
    "fee7" "fee7" "fee7" "fee7"  # pads (b .)
    "0221"      # movs r1, #2     (A)
    "fee7"      # b .
    "0000" "0000"
)

FLIP_BRANCH_BB = ENTRY_PC + 4

FLIP_BRANCH_PC = ENTRY_PC + 6

FLIP_STATIC_BBS = {
    ENTRY_PC: [
        {"address": ENTRY_PC, "mnemonic": "LDR", "operands": "r0, [pc, #0x18]", "size": 2},
    ],
    ENTRY_PC + 2: [
        {"address": ENTRY_PC + 2, "mnemonic": "LDR", "operands": "r0, [r0]", "size": 2},
    ],
    FLIP_BRANCH_BB: [
        {"address": ENTRY_PC + 4, "mnemonic": "CMP", "operands": "r0, #0x20", "size": 2},
        {"address": ENTRY_PC + 6, "mnemonic": "BEQ", "operands": f"0x{ENTRY_PC + 0x14:x}", "size": 2},
    ],
}

FLIP_COMPARE_LOOKUP = {FLIP_BRANCH_BB: FLIP_STATIC_BBS[FLIP_BRANCH_BB][0]}

def build_flip_image(path: str) -> None:
    image = bytearray(0x2000)
    struct.pack_into("<I", image, 0x0, 0x20000600)      # MSP
    struct.pack_into("<I", image, 0x4, ENTRY_PC | 1)    # reset -> entry
    image[0x100:0x100 + len(FLIP_FRAGMENT)] = FLIP_FRAGMENT
    struct.pack_into("<I", image, 0x11C, MMIO_ADDR)     # 0x0800411C literal
    Path(path).write_bytes(bytes(image))

def build_flip_emulator(firmware_path: str) -> IntelligentEmulator:
    emulator = IntelligentEmulator(
        firmware_path=firmware_path,
        mmio_constraints={},
        static_bbs=dict(FLIP_STATIC_BBS),
        constraint_json_path=None,
        max_snapshots=5,
        llm_config_path=None,
        execution_thumb_override=True,
        raw_load_base=LOAD_BASE,
    )
    emulator.setup_memory()
    emulator.load_firmware()
    emulator.register_hooks()
    return emulator
