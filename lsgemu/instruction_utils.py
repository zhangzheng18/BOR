#!/usr/bin/env python3
"""Thumb/ARM instruction classification helpers for LSGEmu scheduling."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


COMPARE_MNEMONICS = {"CMP", "CMN", "TST", "TEQ"}
THUMB_CONDITION_CODES = {
    "EQ", "NE", "CS", "HS", "CC", "LO", "MI", "PL",
    "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE",
}
CONDITIONAL_BRANCH_MNEMONICS = {
    "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
    "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
    "CBZ", "CBNZ", "TBB", "TBH",
}
SWITCH_DISPATCH_MNEMONICS = {"TBB", "TBH", "LDRPC"}
CALL_SPLIT_MNEMONICS = {"BL", "BLX"}


def _normalize_mnemonic(value: object) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    parts = [part for part in raw.split(".") if part]
    if len(parts) >= 2 and parts[0] == "B":
        condition = parts[1]
        if condition not in {"N", "W", "NW"}:
            return f"B{condition}"
        return "B"
    return parts[0]


def _predicated_condition_for_mnemonic(value: object) -> Optional[str]:
    raw = str(value or "").strip().upper()
    if not raw or "." not in raw:
        return None
    parts = [part for part in raw.split(".") if part]
    if len(parts) < 2:
        return None
    base = parts[0]
    if base in {"B", "BL", "BLX", "BX", "BXJ"} or base.startswith("IT"):
        return None
    for part in parts[1:]:
        if part in THUMB_CONDITION_CODES:
            return part
    return None


def _it_predicate_token_for_instruction(insn: Dict[str, object]) -> Optional[str]:
    condition = _predicated_condition_for_mnemonic(insn.get("mnemonic"))
    return f"IT{condition}" if condition else None


def _is_frontier_branch_mnemonic(value: object) -> bool:
    mnemonic = _normalize_mnemonic(value)
    if mnemonic.startswith("IT") and mnemonic[2:] in THUMB_CONDITION_CODES:
        return True
    if mnemonic in SWITCH_DISPATCH_MNEMONICS or mnemonic in {"CBZ", "CBNZ"}:
        return True
    if mnemonic.startswith("B") and mnemonic not in {"B", "BAL", "BL", "BLX", "BX", "BXJ"}:
        return True
    return False


def _is_call_frontier_mnemonic(value: object) -> bool:
    return _is_indirect_dispatch_mnemonic(value)


def _is_direct_call_mnemonic(value: object) -> bool:
    return _normalize_mnemonic(value) == "BL"


def _is_indirect_dispatch_mnemonic(value: object) -> bool:
    mnemonic = _normalize_mnemonic(value)
    return mnemonic in {"BX", "BXJ", "BLX"}


def _is_return_like_instruction(insn: Dict[str, object]) -> bool:
    mnemonic = _normalize_mnemonic(insn.get("mnemonic"))
    operands = str(insn.get("operands", "")).lower()
    if mnemonic in {"POP", "LDM", "LDMIA", "LDMFD"} and "pc" in operands:
        return True
    if mnemonic in {"MOV", "MOVS"} and re.search(r"\bpc\s*,", operands):
        return True
    if mnemonic == "BX" and re.search(r"\blr\b", operands):
        return True
    if mnemonic.startswith("LDR"):
        parts = [part.strip().lower() for part in operands.split(",") if part.strip()]
        if parts and parts[0] == "pc" and _parse_indexed_pc_load_dispatch_operands(operands) is None:
            return True
    return False


def _is_dispatch_frontier_mnemonic(value: object) -> bool:
    return _is_frontier_branch_mnemonic(value) or _is_call_frontier_mnemonic(value)


def _is_replayable_frontier_mnemonic(value: object) -> bool:
    return _is_dispatch_frontier_mnemonic(value) or _is_direct_call_mnemonic(value)


def _parse_indexed_pc_load_dispatch_operands(operands: object) -> Optional[Tuple[str, str, int]]:
    """Parse real indexed `ldr pc, [base, index, lsl #shift]` jump-table forms."""
    text = str(operands or "").lower()
    match = re.search(
        r'^\s*pc\s*,\s*\[\s*(r(?:[0-9]|1[0-2])|pc)\s*,\s*(r(?:[0-9]|1[0-2]))'
        r'(?:\s*,\s*lsl\s*#?(\d+))?\s*\]',
        text,
    )
    if match:
        shift = int(match.group(3) or 0)
        if 0 <= shift <= 5:
            return match.group(1), match.group(2), shift
        return None

    parts = [part.strip().lower() for part in str(operands or "").split(",") if part.strip()]
    if len(parts) < 3 or parts[0] != "pc":
        return None
    base_reg = parts[1]
    index_reg = parts[2]
    if not re.fullmatch(r"r(?:[0-9]|1[0-2])|pc", base_reg):
        return None
    if not re.fullmatch(r"r(?:[0-9]|1[0-2])", index_reg):
        return None
    shift = 0
    if len(parts) >= 4:
        try:
            shift = int(parts[3].replace("lsl", "").replace("#", "").strip() or "0", 0)
        except ValueError:
            return None
    if shift < 0 or shift > 5:
        return None
    return base_reg, index_reg, shift


def _is_pc_load_dispatch_instruction(insn: Dict[str, object]) -> bool:
    mnemonic = _normalize_mnemonic(insn.get("mnemonic"))
    if not mnemonic.startswith("LDR"):
        return False
    operands = [part.strip().lower() for part in str(insn.get("operands", "")).split(",") if part.strip()]
    if not operands or operands[0] != "pc":
        return False
    return _parse_indexed_pc_load_dispatch_operands(insn.get("operands", "")) is not None


def _dispatch_mnemonic_for_instruction(insn: Dict[str, object]) -> str:
    if _is_pc_load_dispatch_instruction(insn):
        return "LDRPC"
    it_predicate = _it_predicate_token_for_instruction(insn)
    if it_predicate is not None:
        return it_predicate
    return _normalize_mnemonic(insn.get("mnemonic"))


def _is_switch_frontier_mnemonic(value: object) -> bool:
    return _normalize_mnemonic(value) in SWITCH_DISPATCH_MNEMONICS


def _is_switch_frontier_instruction(insn: Dict[str, object]) -> bool:
    return _is_switch_frontier_mnemonic(_dispatch_mnemonic_for_instruction(insn))


def _is_dispatch_frontier_instruction(insn: Dict[str, object]) -> bool:
    return _is_dispatch_frontier_mnemonic(_dispatch_mnemonic_for_instruction(insn))


def _should_split_after_mnemonic(value: object) -> bool:
    mnemonic = _normalize_mnemonic(value)
    if mnemonic in CALL_SPLIT_MNEMONICS or mnemonic in CONDITIONAL_BRANCH_MNEMONICS:
        return True
    return mnemonic in {"B", "BX", "BXJ"}


def _parse_direct_target(operands: object) -> Optional[int]:
    matches = re.findall(r"0x[0-9a-fA-F]+", str(operands or ""))
    if not matches:
        return None
    try:
        return int(matches[-1], 16) & ~1
    except ValueError:
        return None


def _parse_int_token(value: object) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _parse_mem_offset(operands: object, base_reg: str = "r0") -> Optional[int]:
    pattern = r"\[\s*" + re.escape(base_reg.lower()) + r"\s*,\s*#\s*(0x[0-9a-f]+|[0-9]+)\s*\]"
    match = re.search(pattern, str(operands or "").lower())
    if not match:
        return None
    return _parse_int_token(match.group(1))


def _parse_dest_register(operands: object) -> Optional[str]:
    parts = [part.strip().lower() for part in str(operands or "").split(",", 1)]
    if not parts:
        return None
    if re.fullmatch(r"r(?:[0-9]|1[0-2])|lr|pc", parts[0]):
        return parts[0]
    return None


def _parse_add_base_immediate(operands: object, base_reg: str = "r0") -> Optional[Tuple[str, int]]:
    parts = [part.strip().lower() for part in str(operands or "").split(",") if part.strip()]
    if len(parts) < 3 or parts[1] != base_reg.lower():
        return None
    dest = parts[0]
    if re.fullmatch(r"r(?:[0-9]|1[0-2])", dest) is None:
        return None
    immediate = _parse_int_token(parts[2].lstrip("#"))
    if immediate is None:
        return None
    return dest, int(immediate)


def _parse_dispatch_target_register(operands: object) -> Optional[str]:
    parts = [part.strip().lower() for part in str(operands or "").split(",") if part.strip()]
    if not parts:
        return None
    reg = parts[0]
    if re.fullmatch(r"r(?:[0-9]|1[0-2])|lr|pc", reg):
        return reg
    return None
