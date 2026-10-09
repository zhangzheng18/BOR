#!/usr/bin/env python3
"""
Regression test for ARM PC-relative literal loads feeding async-state waits.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.llm_code_analyzer import LLMCodeAnalyzer


def main() -> int:
    instructions = [
        {"address": 0x000A5858, "mnemonic": "ADD", "operands": "R0, R0, R4", "size": 4},
        {"address": 0x000A585C, "mnemonic": "LDR", "operands": "R3, [PC, #0xc]", "size": 4},
        {"address": 0x000A5860, "mnemonic": "LDR", "operands": "R3, [R3, #4]", "size": 4},
        {"address": 0x000A5864, "mnemonic": "CMP", "operands": "R0, R3", "size": 4},
        {"address": 0x000A5868, "mnemonic": "BHI", "operands": "#0xa585c", "size": 4},
    ]
    code_page = bytearray(0x1000)
    code_page[0x870:0x874] = (0x00153830).to_bytes(4, "little")
    state_page = bytearray(0x1000)
    state_page[0x834:0x838] = (0).to_bytes(4, "little")
    static_bbs = {0x000A5858: instructions}
    analyzer = LLMCodeAnalyzer(config_path=None, static_bbs=static_bbs, thumb_mode=False)
    snapshot = SimpleNamespace(
        bb_address=0x000A5858,
        cpu_state={"r0": 0x20, "r4": 0x10, "pc": 0x000A5858},
        memory_regions={
            (0x000A5000, 0x1000): bytes(code_page),
            (0x00153000, 0x1000): bytes(state_page),
        },
        mmio_values={},
        bb_instructions=instructions,
    )

    analysis = analyzer._analyze_self_loop_constraint(snapshot)
    constraint = (analysis.suggested_constraints or [{}])[0]
    source_right_taken = analyzer._infer_branch_condition_value(
        "BHI",
        "CMP",
        "r3",
        ["r0", "r3"],
        {"r0": 0x30},
        4,
        True,
    )
    source_right_not_taken = analyzer._infer_branch_condition_value(
        "BHI",
        "CMP",
        "r3",
        ["r0", "r3"],
        {"r0": 0x30},
        4,
        False,
    )
    result = {
        "confidence": analysis.confidence,
        "address": constraint.get("address"),
        "read_pc": constraint.get("read_pc"),
        "value": constraint.get("value"),
        "source_right_taken": source_right_taken,
        "source_right_not_taken": source_right_not_taken,
        "reason": analysis.reason,
        "checks": {
            "resolved_state_address": constraint.get("address") == 0x00153834,
            "resolved_second_load_pc": constraint.get("read_pc") == 0x000A5860,
            "exit_value_tracks_deadline": constraint.get("value") == 0x30,
            "not_literal_or_branch_address": constraint.get("address") not in {0x000A5870, 0x000A5868},
            "source_right_bhi_taken": source_right_taken == 0x2F,
            "source_right_bhi_not_taken": source_right_not_taken == 0x30,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
