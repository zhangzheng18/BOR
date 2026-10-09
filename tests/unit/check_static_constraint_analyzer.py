#!/usr/bin/env python3
"""Regression tests for conservative static MMIO constraint extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.static_constraint_analyzer import StaticConstraintAnalyzer


def main() -> int:
    static_bbs = {
        0x08001000: [
            {"address": 0x08001000, "mnemonic": "MOVW", "operands": "R0, #0x1000", "size": 4},
            {"address": 0x08001004, "mnemonic": "MOVT", "operands": "R0, #0x4002", "size": 4},
            {"address": 0x08001008, "mnemonic": "LDR", "operands": "R1, [R0, #0x4]", "size": 2},
        ],
        0x0800100A: [
            {"address": 0x0800100A, "mnemonic": "TST", "operands": "R1, #0x2", "size": 2},
            {"address": 0x0800100C, "mnemonic": "BEQ", "operands": "0x08001000", "size": 2},
        ],
        0x08002000: [
            {"address": 0x08002000, "mnemonic": "MOVW", "operands": "R0, #0xe010", "size": 4},
            {"address": 0x08002004, "mnemonic": "MOVT", "operands": "R0, #0xe000", "size": 4},
            {"address": 0x08002008, "mnemonic": "LDR", "operands": "R2, [R0, #0x0]", "size": 2},
            {"address": 0x0800200A, "mnemonic": "CMP", "operands": "R2, #0", "size": 2},
            {"address": 0x0800200C, "mnemonic": "BEQ", "operands": "0x08002008", "size": 2},
        ],
        0x08003000: [
            {"address": 0x08003000, "mnemonic": "MOVW", "operands": "R0, #0x2000", "size": 4},
            {"address": 0x08003004, "mnemonic": "MOVT", "operands": "R0, #0x4000", "size": 4},
            {"address": 0x08003008, "mnemonic": "LDR", "operands": "R3, [R0, #0x0]", "size": 2},
            {"address": 0x0800300A, "mnemonic": "CBZ", "operands": "R3, 0x08003008", "size": 2},
        ],
        0x08004000: [
            {"address": 0x08004000, "mnemonic": "MOVW", "operands": "R0, #0x3000", "size": 4},
            {"address": 0x08004004, "mnemonic": "MOVT", "operands": "R0, #0x4002", "size": 4},
            {"address": 0x08004008, "mnemonic": "LDR", "operands": "R4, [R0, #0x8]", "size": 2},
            {"address": 0x0800400A, "mnemonic": "TST", "operands": "R4, #0x4", "size": 2},
            {"address": 0x0800400C, "mnemonic": "BNE", "operands": "0x08004020", "size": 2},
        ],
    }
    old_forward = os.environ.pop("LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES", None)
    constraints = StaticConstraintAnalyzer(static_bbs).infer_constraints()
    if old_forward is not None:
        os.environ["LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES"] = old_forward
    by_branch = {int(item["branch_pc"]): item for item in constraints}
    multi_bb = by_branch.get(0x0800100C)
    cortexm = by_branch.get(0x0800200C)
    cbz = by_branch.get(0x0800300A)
    forward_default = by_branch.get(0x0800400C)
    os.environ["LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES"] = "0"
    try:
        forward_disabled_constraints = StaticConstraintAnalyzer(static_bbs).infer_constraints()
    finally:
        if old_forward is None:
            os.environ.pop("LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES", None)
        else:
            os.environ["LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES"] = old_forward
    forward_disabled_by_branch = {int(item["branch_pc"]): item for item in forward_disabled_constraints}
    forward_disabled = forward_disabled_by_branch.get(0x0800400C)
    checks = {
        "multi_bb_found": multi_bb is not None,
        "multi_bb_read_pc_ok": multi_bb is not None and int(multi_bb["read_pc"]) == 0x08001008,
        "multi_bb_address_ok": multi_bb is not None and int(multi_bb["address"]) == 0x40021004,
        "multi_bb_value_ok": multi_bb is not None and int(multi_bb["value"]) == 0x2,
        "cortexm_system_mmio_ok": cortexm is not None and int(cortexm["address"]) == 0xE000E010,
        "cbz_exit_value_ok": cbz is not None and int(cbz["value"]) == 0x1,
        "forward_branch_default_found": forward_default is not None,
        "forward_branch_value_ok": forward_default is not None and int(forward_default["value"]) == 0x4,
        "forward_branch_dynamic_replay_marked": (
            forward_default is not None
            and forward_default.get("speculation_level") == "dynamic_replay_required"
            and forward_default.get("coverage_credit_policy") == "unicorn_execution_only"
        ),
        "forward_branch_can_be_disabled": forward_disabled is None,
    }
    print(json.dumps({"constraints": constraints, "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
