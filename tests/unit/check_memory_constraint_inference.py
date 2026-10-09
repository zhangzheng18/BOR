#!/usr/bin/env python3
"""
Smoke-test memory-backed branch constraint inference and candidate priority.
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

from lsgemu.historical_runner import BranchConstraintCandidate, HistoricalRunner
from lsgemu.llm_guide.llm_guide import LLMGuide


def build_runner(static_bbs):
    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.prepared = SimpleNamespace(
        compare_lookup={
            instructions[-1]["address"]: instructions[-2]
            for instructions in static_bbs.values()
            if len(instructions) >= 2
        },
        static_bbs=static_bbs,
        instruction_lookup={
            insn["address"]: insn
            for instructions in static_bbs.values()
            for insn in instructions
        },
    )
    runner.llm_guide = LLMGuide(
        static_bbs,
        use_llm=False,
        instruction_lookup=runner.prepared.instruction_lookup,
        compare_lookup=runner.prepared.compare_lookup,
    )
    return runner


def main() -> int:
    cmp_static = {
        0x08001000: [
            {"address": 0x08001000, "mnemonic": "LDR", "operands": "R3, [R4, #0x20]", "size": 2},
            {"address": 0x08001002, "mnemonic": "CMP", "operands": "R3, #0x20", "size": 2},
            {"address": 0x08001004, "mnemonic": "BEQ", "operands": "0x08001010", "size": 2},
        ]
    }
    tst_static = {
        0x08002000: [
            {"address": 0x08002000, "mnemonic": "LDR", "operands": "R3, [R4, #0x24]", "size": 2},
            {"address": 0x08002002, "mnemonic": "TST", "operands": "R3, #0x2000000", "size": 2},
            {"address": 0x08002004, "mnemonic": "BNE", "operands": "0x08002010", "size": 2},
        ]
    }
    teq_static = {
        0x08003000: [
            {"address": 0x08003000, "mnemonic": "LDR", "operands": "R3, [R4, #0x28]", "size": 2},
            {"address": 0x08003002, "mnemonic": "TEQ", "operands": "R3, #0x55", "size": 2},
            {"address": 0x08003004, "mnemonic": "BEQ", "operands": "0x08003010", "size": 2},
        ]
    }
    masked_static = {
        0x08004000: [
            {"address": 0x08004000, "mnemonic": "LDR", "operands": "R2, [R4, #0x2c]", "size": 2},
            {"address": 0x08004002, "mnemonic": "AND.W", "operands": "R2, R2, #0xc", "size": 4},
            {"address": 0x08004006, "mnemonic": "CMP", "operands": "R2, #0x8", "size": 2},
            {"address": 0x08004008, "mnemonic": "BEQ", "operands": "0x08004020", "size": 2},
        ]
    }

    cmp_runner = build_runner(cmp_static)
    tst_runner = build_runner(tst_static)
    teq_runner = build_runner(teq_static)
    masked_runner = build_runner(masked_static)

    cmp_candidate = BranchConstraintCandidate(
        constraint_type="memory",
        address=0x20000020,
        value=0,
        read_pc=0x08001000,
        constraint_pc=0x08001002,
        source="dependency",
    )
    tst_candidate = BranchConstraintCandidate(
        constraint_type="memory",
        address=0x20000024,
        value=0,
        read_pc=0x08002000,
        constraint_pc=0x08002002,
        source="dependency",
    )
    teq_candidate = BranchConstraintCandidate(
        constraint_type="memory",
        address=0x20000028,
        value=0,
        read_pc=0x08003000,
        constraint_pc=0x08003002,
        source="dependency",
    )
    masked_candidate = BranchConstraintCandidate(
        constraint_type="memory",
        address=0x2000002C,
        value=0,
        read_pc=0x08004000,
        constraint_pc=0x08004006,
        source="dependency",
    )
    mmio_candidate = BranchConstraintCandidate(
        constraint_type="mmio",
        address=0x40000000,
        value=0,
        read_pc=0x08001000,
        constraint_pc=0x08001002,
        source="dependency_mmio",
    )

    cmp_taken = cmp_runner._infer_memory_constraint_value(0x08001004, "BEQ", True, cmp_candidate)
    cmp_not_taken = cmp_runner._infer_memory_constraint_value(0x08001004, "BEQ", False, cmp_candidate)
    tst_taken = tst_runner._infer_memory_constraint_value(0x08002004, "BNE", True, tst_candidate)
    tst_not_taken = tst_runner._infer_memory_constraint_value(0x08002004, "BNE", False, tst_candidate)
    teq_taken = teq_runner._infer_memory_constraint_value(0x08003004, "BEQ", True, teq_candidate)
    teq_not_taken = teq_runner._infer_memory_constraint_value(0x08003004, "BEQ", False, teq_candidate)
    masked_taken = masked_runner._infer_memory_constraint_value(0x08004008, "BEQ", True, masked_candidate)
    masked_not_taken = masked_runner._infer_memory_constraint_value(0x08004008, "BEQ", False, masked_candidate)
    masked_mmio_taken = masked_runner.llm_guide.infer_constraint(0x08004008, "BEQ", True, 0x40021004)
    masked_mmio_not_taken = masked_runner.llm_guide.infer_constraint(0x08004008, "BEQ", False, 0x40021004)

    priority_cmp = cmp_runner._candidate_dependency_priority(cmp_candidate)
    priority_mmio = cmp_runner._candidate_dependency_priority(mmio_candidate)

    result = {
        "cmp_taken": cmp_taken,
        "cmp_not_taken": cmp_not_taken,
        "tst_taken": tst_taken,
        "tst_not_taken": tst_not_taken,
        "teq_taken": teq_taken,
        "teq_not_taken": teq_not_taken,
        "masked_taken": masked_taken,
        "masked_not_taken": masked_not_taken,
        "masked_mmio_taken": masked_mmio_taken,
        "masked_mmio_not_taken": masked_mmio_not_taken,
        "priority_cmp": priority_cmp,
        "priority_mmio": priority_mmio,
        "checks": {
            "cmp_taken_ok": cmp_taken == 0x20,
            "cmp_not_taken_ok": cmp_not_taken == 0x21,
            "tst_taken_ok": tst_taken == 0x02000000,
            "tst_not_taken_ok": tst_not_taken == 0x0,
            "teq_taken_ok": teq_taken == 0x55,
            "teq_not_taken_ok": teq_not_taken == 0x54,
            "masked_taken_ok": masked_taken == 0x8,
            "masked_not_taken_ok": masked_not_taken == 0x0,
            "masked_mmio_taken_ok": masked_mmio_taken == 0x8,
            "masked_mmio_not_taken_ok": masked_mmio_not_taken == 0x0,
            "memory_priority_ok": priority_cmp > priority_mmio,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
