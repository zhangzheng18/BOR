#!/usr/bin/env python3
"""
Conservative static MMIO constraint extraction.

The analyzer only emits constraints for local polling-like patterns:

    LDR   Rt, [Rbase, #off]   ; read MMIO
    TST/CMP Rt, #imm          ; check status
    Bxx   back_edge           ; loop while condition holds

The stored read_pc is the PC of the MMIO load, while constraint_pc is the PC of
the compare/test instruction that creates the branch constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..artifact_io import atomic_json_dump

logger = logging.getLogger(__name__)


CONDITIONAL_BRANCHES = {
    "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
    "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
    "CBZ", "CBNZ",
}


@dataclass(frozen=True)
class MMIOReadSource:
    read_pc: int
    mmio_addr: int


class StaticConstraintAnalyzer:
    """Extract PC-specific MMIO constraints from static basic blocks."""

    def __init__(self, static_bbs: Dict[int, List[Dict[str, object]]]):
        self.static_bbs = static_bbs or {}
        self.max_loop_window_bbs = self._env_int("LSGEMU_STATIC_CONSTRAINT_MAX_LOOP_BBS", 4)
        self.max_loop_window_instructions = self._env_int("LSGEMU_STATIC_CONSTRAINT_MAX_LOOP_INSNS", 32)
        self.enable_forward_branches = self._env_bool("LSGEMU_STATIC_CONSTRAINT_FORWARD_BRANCHES", True)
        # 预排序的 (first_pc, start, last_pc, instructions) 索引。
        # 大固件（10 万 BB、数万回边）上 _linearized_loop_window 若对每个
        # 候选分支全量扫描 static_bbs，复杂度是 O(branches × bbs) ≈ 20 亿次
        # 迭代，仅这一步就吃掉 20+ 分钟（F427 实测）。预排序后每个窗口用
        # bisect 定位，均摊 O(log n + window)。
        self._sorted_blocks: Optional[List[Tuple[int, int, int, List[Dict[str, object]]]]] = None

    def _ensure_sorted_blocks(self) -> List[Tuple[int, int, int, List[Dict[str, object]]]]:
        if self._sorted_blocks is None:
            items: List[Tuple[int, int, int, List[Dict[str, object]]]] = []
            for start, instructions in self.static_bbs.items():
                if not instructions:
                    continue
                start_i = self._as_int(start)
                if start_i is None:
                    continue
                first_pc = self._as_int(instructions[0].get("address"))
                last_pc = self._as_int(instructions[-1].get("address"))
                if first_pc is None or last_pc is None:
                    continue
                items.append((first_pc, start_i, last_pc, instructions))
            items.sort(key=lambda item: item[0])
            self._sorted_blocks = items
        return self._sorted_blocks

    def infer_constraints(self) -> List[Dict[str, object]]:
        constraints: List[Dict[str, object]] = []

        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue

            branch = instructions[-1]
            condition = self._normalize_mnemonic(str(branch.get("mnemonic", "")))
            if condition not in CONDITIONAL_BRANCHES:
                continue

            branch_pc = self._as_int(branch.get("address"))
            target = self._parse_branch_target(str(branch.get("operands", "")))
            if branch_pc is None or target is None:
                continue

            is_forward_branch = target > branch_pc
            # Only solve obvious loop-back branches by default. Forward branches
            # are normal control flow in many parsers; enable them explicitly for
            # targeted experiments where the report can label them as speculative.
            if is_forward_branch and not self.enable_forward_branches:
                continue

            target_direction = bool(is_forward_branch)
            inferred = self._analyze_basic_block(
                instructions,
                condition,
                target_direction=target_direction,
            )
            if inferred is None and not is_forward_branch:
                inferred = self._analyze_loop_window(
                    bb_addr=bb_addr,
                    loop_target=target,
                    branch_pc=branch_pc,
                    branch_condition=condition,
                )
            if inferred is None:
                continue

            read_source, value, constraint_pc, reason = inferred
            if is_forward_branch:
                reason = (
                    f"{reason}; conservative forward-branch constraint "
                    f"0x{branch_pc:08x}->0x{target:08x}"
                )
            constraints.append({
                "type": "mmio",
                "read_pc": read_source.read_pc,
                "address": read_source.mmio_addr,
                "value": value,
                "constraint_pc": constraint_pc,
                "branch_pc": branch_pc,
                "description": reason,
                "added_by": "static_constraint_analyzer",
                "constraint_scope": "forward_branch_dynamic_replay" if is_forward_branch else "polling_loop_exit",
                "environment_assumption": "external_mmio_value",
                "speculation_level": "dynamic_replay_required" if is_forward_branch else "conservative_loop_exit",
                "coverage_credit_policy": "unicorn_execution_only",
            })

        return constraints

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(default)), 0))
        except ValueError:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _analyze_loop_window(
        self,
        *,
        bb_addr: int,
        loop_target: int,
        branch_pc: int,
        branch_condition: str,
    ) -> Optional[Tuple[MMIOReadSource, int, int, str]]:
        window = self._linearized_loop_window(bb_addr, loop_target, branch_pc)
        if not window:
            return None
        inferred = self._analyze_basic_block(window, branch_condition)
        if inferred is None:
            return None
        read_source, value, constraint_pc, reason = inferred
        return (
            read_source,
            value,
            constraint_pc,
            f"{reason}; conservative multi-BB loop window 0x{loop_target:08x}-0x{branch_pc:08x}",
        )

    def _linearized_loop_window(
        self,
        bb_addr: int,
        loop_target: int,
        branch_pc: int,
    ) -> Optional[List[Dict[str, object]]]:
        if self.max_loop_window_bbs <= 1:
            return None
        block_items: List[Tuple[int, List[Dict[str, object]]]] = []
        # bisect 定位 first_pc ∈ [loop_target, branch_pc] 的块，避免全量扫描。
        sorted_blocks = self._ensure_sorted_blocks()
        first_pcs = [item[0] for item in sorted_blocks]
        lo = bisect.bisect_left(first_pcs, loop_target)
        hi = bisect.bisect_right(first_pcs, branch_pc)
        for first_pc, start_i, last_pc, instructions in sorted_blocks[lo:hi]:
            if last_pc > branch_pc:
                continue
            block_items.append((start_i, instructions))

        if len(block_items) <= 1 or len(block_items) > self.max_loop_window_bbs:
            return None
        if not any(start == bb_addr for start, _instructions in block_items):
            return None

        linearized: List[Dict[str, object]] = []
        seen_pcs: Set[int] = set()
        for _start, instructions in block_items:
            for insn in instructions:
                pc = self._as_int(insn.get("address"))
                if pc is None or pc in seen_pcs or pc > branch_pc:
                    continue
                seen_pcs.add(pc)
                linearized.append(insn)

        linearized.sort(key=lambda item: self._as_int(item.get("address")) or 0)
        if not linearized:
            return None
        if len(linearized) > self.max_loop_window_instructions:
            return None
        final_pc = self._as_int(linearized[-1].get("address"))
        if final_pc != branch_pc:
            return None

        for insn in linearized[:-1]:
            mnemonic = self._normalize_mnemonic(str(insn.get("mnemonic", "")))
            if mnemonic in CONDITIONAL_BRANCHES or mnemonic in {"B", "BL", "BLX", "BX"}:
                return None
        return linearized

    def write_constraints(self, output_path: str | Path) -> int:
        constraints = self.infer_constraints()
        if not constraints:
            return 0

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            try:
                with output_path.open("r") as f:
                    data = json.load(f)
            except Exception:
                data = {"constraints": []}
        else:
            data = {"constraints": []}

        if not isinstance(data, dict) or "constraints" not in data:
            data = {"constraints": []}

        updated = 0
        for constraint in constraints:
            item = self._serialize_constraint(constraint)
            if self._upsert(data["constraints"], item):
                updated += 1

        if updated:
            atomic_json_dump(data, output_path, indent=2)
            logger.info("静态预求解约束: %d 条 -> %s", updated, output_path)

        return updated

    def _analyze_basic_block(
        self,
        instructions: List[Dict[str, object]],
        branch_condition: str,
        *,
        target_direction: bool = False,
    ) -> Optional[Tuple[MMIOReadSource, int, int, str]]:
        reg_values: Dict[str, int] = {}
        reg_sources: Dict[str, MMIOReadSource] = {}
        last_compare: Optional[Dict[str, object]] = None
        last_compare_source: Optional[MMIOReadSource] = None

        for insn in instructions[:-1]:
            mnemonic = self._normalize_mnemonic(str(insn.get("mnemonic", "")))
            operands = str(insn.get("operands", ""))
            parts = self._split_operands(operands)

            if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
                dest = self._reg(parts[0])
                imm = self._parse_immediate(parts[1])
                if dest and imm is not None:
                    reg_values[dest] = imm & 0xFFFFFFFF
                    reg_sources.pop(dest, None)
                continue

            if mnemonic == "MOVT" and len(parts) >= 2:
                dest = self._reg(parts[0])
                imm = self._parse_immediate(parts[1])
                if dest and imm is not None:
                    low = reg_values.get(dest, 0) & 0xFFFF
                    reg_values[dest] = ((imm & 0xFFFF) << 16) | low
                    reg_sources.pop(dest, None)
                continue

            if mnemonic in {"ADD", "ADDS", "ADD.W"} and len(parts) >= 3:
                dest = self._reg(parts[0])
                base = self._reg(parts[1])
                imm = self._parse_immediate(parts[2])
                if dest and base and imm is not None and base in reg_values:
                    reg_values[dest] = (reg_values[base] + imm) & 0xFFFFFFFF
                    reg_sources.pop(dest, None)
                continue

            if mnemonic in {"SUB", "SUBS", "SUB.W"} and len(parts) >= 3:
                dest = self._reg(parts[0])
                base = self._reg(parts[1])
                imm = self._parse_immediate(parts[2])
                if dest and base and imm is not None and base in reg_values:
                    reg_values[dest] = (reg_values[base] - imm) & 0xFFFFFFFF
                    reg_sources.pop(dest, None)
                continue

            if mnemonic in {"LDR", "LDR.W", "LDRB", "LDRH"} and len(parts) >= 2:
                dest = self._reg(parts[0])
                addr = self._resolve_memory_operand(parts[1], reg_values)
                if dest and addr is not None and self._is_mmio_address(addr):
                    read_pc = self._as_int(insn.get("address"))
                    if read_pc is not None:
                        reg_sources[dest] = MMIOReadSource(read_pc=read_pc, mmio_addr=addr)
                    reg_values.pop(dest, None)
                elif dest:
                    reg_sources.pop(dest, None)
                    reg_values.pop(dest, None)
                continue

            if mnemonic in {"AND", "ANDS", "AND.W", "ANDS.W", "ORR", "ORRS"} and len(parts) >= 2:
                dest = self._reg(parts[0])
                source = self._reg(parts[1]) if len(parts) >= 3 else dest
                if dest and source and source in reg_sources:
                    reg_sources[dest] = reg_sources[source]
                continue

            if mnemonic in {"TST", "TST.W", "TEQ", "CMP", "CMN"} and len(parts) >= 2:
                checked_reg = self._reg(parts[0])
                if checked_reg and checked_reg in reg_sources:
                    last_compare = insn
                    last_compare_source = reg_sources[checked_reg]

        if (not last_compare or not last_compare_source) and branch_condition in {"CBZ", "CBNZ"}:
            branch = instructions[-1] if instructions else {}
            operands = self._split_operands(str(branch.get("operands", "")))
            checked_reg = self._reg(operands[0]) if operands else None
            if checked_reg and checked_reg in reg_sources:
                last_compare = branch
                last_compare_source = reg_sources[checked_reg]

        if not last_compare or not last_compare_source:
            return None

        value = self._infer_exit_value(
            last_compare,
            branch_condition,
            target_direction=target_direction,
        )
        if value is None:
            return None

        constraint_pc = self._as_int(last_compare.get("address"))
        if constraint_pc is None:
            return None

        mnemonic = str(last_compare.get("mnemonic", "")).upper()
        direction_text = "taken" if target_direction else "not-taken"
        reason = (
            f"Static polling exit constraint from {mnemonic} "
            f"@ 0x{constraint_pc:08x}, branch {branch_condition} {direction_text}"
        )
        return last_compare_source, value, constraint_pc, reason

    def _infer_exit_value(
        self,
        compare_insn: Dict[str, object],
        branch_condition: str,
        *,
        target_direction: bool = False,
    ) -> Optional[int]:
        mnemonic = self._normalize_mnemonic(str(compare_insn.get("mnemonic", "")))
        operands = self._split_operands(str(compare_insn.get("operands", "")))
        condition = self._normalize_mnemonic(branch_condition)

        if condition == "CBZ":
            return 0 if target_direction else 1
        if condition == "CBNZ":
            return 1 if target_direction else 0

        if len(operands) < 2:
            return None

        imm = self._parse_immediate(operands[1])
        if imm is None:
            return None

        if mnemonic in {"TST", "TST.W", "TEQ"}:
            if condition == "BNE":
                return imm if target_direction else 0
            if condition == "BEQ":
                return 0 if target_direction else imm
            return imm if not target_direction else 0

        compare_value = imm
        if condition == "BEQ":
            return compare_value if target_direction else (compare_value + 1)
        if condition == "BNE":
            return (compare_value + 1) if target_direction else compare_value
        if condition in {"BGT", "BHI"}:
            return (compare_value + 1) if target_direction else compare_value
        if condition in {"BLT", "BLO"}:
            return (compare_value - 1) if target_direction else compare_value
        if condition in {"BGE", "BHS"}:
            return compare_value if target_direction else (compare_value - 1)
        if condition in {"BLE", "BLS"}:
            return compare_value if target_direction else (compare_value + 1)
        return None

    def _resolve_memory_operand(self, operand: str, reg_values: Dict[str, int]) -> Optional[int]:
        match = re.search(r"\[([^,\]]+)(?:,\s*([^\]]+))?\]", operand)
        if not match:
            return None

        base = self._reg(match.group(1))
        if not base or base not in reg_values:
            return None

        offset = self._parse_immediate(match.group(2) or "#0")
        if offset is None:
            return None

        return (reg_values[base] + offset) & 0xFFFFFFFF

    @staticmethod
    def _is_mmio_address(address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        return (
            0x40000000 <= address < 0x60000000
            or 0xE0000000 <= address < 0xE0100000
        )

    def _normalize_mnemonic(self, mnemonic: str) -> str:
        raw = str(mnemonic or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B":
            condition = parts[1]
            if condition not in {"N", "W", "NW"}:
                return f"B{condition}"
            return "B"
        return parts[0]

    def _split_operands(self, operands: str) -> List[str]:
        parts: List[str] = []
        current = []
        bracket_depth = 0
        for char in operands:
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            if char == "," and bracket_depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return parts

    def _reg(self, value: str) -> Optional[str]:
        text = value.strip().lower()
        match = re.match(r"^(r(?:1[0-5]|[0-9])|sp|lr|pc)\b", text)
        return match.group(1) if match else None

    def _parse_immediate(self, value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        text = value.strip().lstrip("#").lower()
        if not text:
            return None
        sign = -1 if text.startswith("-") else 1
        text = text[1:] if text.startswith(("-", "+")) else text
        try:
            parsed = int(text, 16) if text.startswith("0x") else int(text, 10)
            return sign * parsed
        except ValueError:
            return None

    def _parse_branch_target(self, operands: str) -> Optional[int]:
        parts = self._split_operands(operands)
        text = parts[-1].strip().split()[0] if parts else ""
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
        except ValueError:
            return None

    def _as_int(self, value) -> Optional[int]:
        if isinstance(value, int):
            return value
        if value is None:
            return None
        try:
            text = str(value)
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            return None

    def _serialize_constraint(self, constraint: Dict[str, object]) -> Dict[str, object]:
        serialized = dict(constraint)
        for key in ("read_pc", "address", "value", "constraint_pc", "branch_pc"):
            if key in serialized and isinstance(serialized[key], int):
                serialized[key] = f"0x{serialized[key]:08x}"
        return serialized

    def _upsert(self, items: List[Dict[str, object]], new_item: Dict[str, object]) -> bool:
        identity = (
            new_item.get("type"),
            new_item.get("read_pc") or new_item.get("pc"),
            new_item.get("address"),
            new_item.get("constraint_pc"),
        )
        for item in items:
            current = (
                item.get("type"),
                item.get("read_pc") or item.get("pc"),
                item.get("address"),
                item.get("constraint_pc"),
            )
            if current == identity:
                if item != new_item:
                    item.update(new_item)
                    return True
                return False
        items.append(new_item)
        return True
