#!/usr/bin/env python3
"""Lightweight static MMIO address analysis for Cortex-M firmware.

The analyzer is intentionally conservative: it only emits concrete MMIO
addresses when a small abstract state can prove the address. Unknown indexed
loads are reported as unresolved candidates instead of guessed values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


MMIO_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x40000000, 0x60000000),
    (0xE0000000, 0xE0100000),
)
LIGHTWEIGHT_MMIO_ANALYSIS_VERSION = 3


@dataclass(frozen=True)
class StaticMMIOAccess:
    pc: int
    bb_addr: int
    address: Optional[int]
    access_type: str
    width: int
    confidence: str
    kind: str
    base_reg: Optional[str] = None
    index_reg: Optional[str] = None
    offset: int = 0
    mnemonic: str = ""
    operands: str = ""

    def to_dict(self) -> Dict[str, object]:
        item = asdict(self)
        if self.address is not None:
            item["address"] = int(self.address) & 0xFFFFFFFF
        return item

    @classmethod
    def from_dict(cls, item: Dict[str, object]) -> "StaticMMIOAccess":
        return cls(
            pc=int(item.get("pc", 0) or 0),
            bb_addr=int(item.get("bb_addr", 0) or 0),
            address=(
                int(item["address"]) & 0xFFFFFFFF
                if item.get("address") is not None
                else None
            ),
            access_type=str(item.get("access_type", "") or ""),
            width=int(item.get("width", 4) or 4),
            confidence=str(item.get("confidence", "") or ""),
            kind=str(item.get("kind", "") or ""),
            base_reg=str(item.get("base_reg")) if item.get("base_reg") else None,
            index_reg=str(item.get("index_reg")) if item.get("index_reg") else None,
            offset=int(item.get("offset", 0) or 0),
            mnemonic=str(item.get("mnemonic", "") or ""),
            operands=str(item.get("operands", "") or ""),
        )


@dataclass
class FunctionMMIOSummary:
    entry_addr: int
    name: str = ""
    reads: Set[int] = field(default_factory=set)
    writes: Set[int] = field(default_factory=set)
    unresolved_reads: int = 0
    unresolved_writes: int = 0
    callsites: List[int] = field(default_factory=list)
    callees: Set[int] = field(default_factory=set)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_addr": int(self.entry_addr) & 0xFFFFFFFF,
            "name": self.name,
            "reads": sorted(int(value) & 0xFFFFFFFF for value in self.reads),
            "writes": sorted(int(value) & 0xFFFFFFFF for value in self.writes),
            "unresolved_reads": int(self.unresolved_reads),
            "unresolved_writes": int(self.unresolved_writes),
            "callsites": sorted(int(value) & 0xFFFFFFFF for value in self.callsites),
            "callees": sorted(int(value) & 0xFFFFFFFF for value in self.callees),
        }


class LightweightMMIOAnalyzer:
    """Small abstract interpreter focused on MMIO address calculation."""

    def __init__(
        self,
        static_bbs: Dict[int, List[Dict[str, object]]],
        instruction_lookup: Optional[Dict[int, Dict[str, object]]] = None,
        symbols_by_addr: Optional[Dict[int, str]] = None,
        thumb_mode: bool = True,
        firmware_path: Optional[str | Path] = None,
        raw_load_base: Optional[int] = None,
    ):
        self.static_bbs = static_bbs or {}
        self.instruction_lookup = instruction_lookup or self._build_instruction_lookup(self.static_bbs)
        self.symbols_by_addr = {
            int(address) & ~1: str(name)
            for address, name in (symbols_by_addr or {}).items()
        }
        self.thumb_mode = bool(thumb_mode)
        self.firmware_path = Path(firmware_path).resolve() if firmware_path else None
        self.raw_load_base = int(raw_load_base) & 0xFFFFFFFF if raw_load_base is not None else None
        self._segment_map = self._build_segment_map()

    @staticmethod
    def _build_instruction_lookup(
        static_bbs: Dict[int, List[Dict[str, object]]]
    ) -> Dict[int, Dict[str, object]]:
        lookup: Dict[int, Dict[str, object]] = {}
        for instructions in static_bbs.values():
            for insn in instructions or []:
                try:
                    lookup[int(insn.get("address", 0) or 0)] = insn
                except Exception:
                    continue
        return lookup

    @staticmethod
    def is_mmio_address(address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        return any(start <= address < end for start, end in MMIO_RANGES)

    @staticmethod
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

    @staticmethod
    def _split_operands(operands: object) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        bracket_depth = 0
        brace_depth = 0
        for ch in str(operands or ""):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
            if ch == "," and bracket_depth == 0 and brace_depth == 0:
                item = "".join(current).strip()
                if item:
                    parts.append(item)
                current = []
                continue
            current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _parse_register(value: object) -> Optional[str]:
        text = str(value or "").strip().lower()
        if re.fullmatch(r"r(?:[0-9]|1[0-2])|sp|lr|pc", text):
            return text
        return None

    @staticmethod
    def _parse_int(value: object) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        text = text.replace("#", "").strip()
        text = text.strip("{}")
        try:
            return int(text, 0)
        except Exception:
            return None

    @staticmethod
    def _memory_width(mnemonic: str) -> int:
        if mnemonic in {"LDRB", "STRB", "LDRSB"}:
            return 1
        if mnemonic in {"LDRH", "STRH", "LDRSH"}:
            return 2
        if mnemonic in {"LDRD", "STRD"}:
            return 8
        return 4

    @staticmethod
    def _access_type(mnemonic: str) -> Optional[str]:
        if mnemonic.startswith("LDR"):
            return "read"
        if mnemonic.startswith("STR"):
            return "write"
        return None

    @staticmethod
    def _pc_value(insn_address: int, thumb_mode: bool) -> int:
        if thumb_mode:
            return (int(insn_address) + 4) & ~3
        return (int(insn_address) + 8) & 0xFFFFFFFF

    def _read_literal_word(self, address: int) -> Optional[int]:
        address = int(address) & 0xFFFFFFFF
        image_bytes = self._read_image_bytes(address, 4)
        if image_bytes is not None and len(image_bytes) >= 4:
            return int.from_bytes(image_bytes[:4], "little") & 0xFFFFFFFF

        insn = self.instruction_lookup.get(address & ~1)
        if insn is not None:
            raw = insn.get("bytes")
            if isinstance(raw, (bytes, bytearray)) and len(raw) >= 4:
                return int.from_bytes(bytes(raw[:4]), "little") & 0xFFFFFFFF
            # Disassemblers often format literal pool data as a numeric operand.
            text = str(insn.get("operands", "") or "")
            match = re.search(r"0x[0-9a-fA-F]{1,8}", text)
            if match:
                return int(match.group(0), 16) & 0xFFFFFFFF
        return None

    def _build_segment_map(self) -> List[Tuple[int, int, int]]:
        if self.firmware_path is None:
            return []
        try:
            from elftools.elf.elffile import ELFFile
        except Exception:
            return []
        try:
            with self.firmware_path.open("rb") as f:
                elf = ELFFile(f)
                segments: List[Tuple[int, int, int]] = []
                for segment in elf.iter_segments():
                    if segment["p_type"] != "PT_LOAD":
                        continue
                    start = int(segment["p_vaddr"]) & 0xFFFFFFFF
                    filesz = int(segment["p_filesz"]) & 0xFFFFFFFF
                    memsz = int(segment["p_memsz"]) & 0xFFFFFFFF
                    if filesz <= 0 and memsz <= 0:
                        continue
                    end = start + max(filesz, memsz)
                    segments.append((start, end, int(segment["p_offset"]) & 0xFFFFFFFF))
                return segments
        except Exception:
            return []

    def _read_image_bytes(self, address: int, size: int) -> Optional[bytes]:
        if self.firmware_path is None or size <= 0:
            return None
        try:
            data = self.firmware_path.read_bytes()
        except Exception:
            return None

        address = int(address) & 0xFFFFFFFF
        if self.raw_load_base is not None:
            base = int(self.raw_load_base) & 0xFFFFFFFF
            if base <= address and address + size <= base + len(data):
                offset = address - base
                return bytes(data[offset:offset + size])

        for start, end, file_offset in self._segment_map:
            if start <= address and address + size <= end:
                offset = file_offset + (address - start)
                if 0 <= offset and offset + size <= len(data):
                    return bytes(data[offset:offset + size])

        if 0 <= address and address + size <= len(data):
            return bytes(data[address:address + size])
        return None

    def _resolve_memory_operand(
        self,
        operand: str,
        state: Dict[str, Optional[int]],
        insn_address: int,
    ) -> Tuple[Optional[int], str, Optional[str], Optional[str], int]:
        match = re.search(r"\[([^\]]+)\]", str(operand or ""))
        if not match:
            return None, "no_memory_operand", None, None, 0
        inner = match.group(1)
        parts = [part.strip() for part in self._split_operands(inner)]
        if not parts:
            return None, "empty_memory_operand", None, None, 0

        base_reg = self._parse_register(parts[0])
        if base_reg is None:
            absolute = self._parse_int(parts[0])
            if absolute is not None:
                return absolute & 0xFFFFFFFF, "absolute", None, None, 0
            return None, "unknown_base", None, None, 0
        if base_reg == "pc":
            base_value: Optional[int] = self._pc_value(insn_address, self.thumb_mode)
        else:
            base_value = state.get(base_reg)

        if base_value is None:
            return None, "unknown_base_value", base_reg, None, 0

        offset = 0
        index_reg = None
        if len(parts) >= 2:
            raw_offset = parts[1].strip()
            immediate = self._parse_int(raw_offset)
            if immediate is not None:
                offset += immediate
            else:
                index_reg = self._parse_register(raw_offset)
                if index_reg is None:
                    return None, "unknown_offset", base_reg, None, 0
                index_value = state.get(index_reg)
                if index_value is None:
                    kind = "unknown_index_mmio_base" if self.is_mmio_address(base_value) else "unknown_index"
                    return None, kind, base_reg, index_reg, 0
                shift = 0
                if len(parts) >= 3:
                    shift_match = re.match(r"(?i)(?:lsl|lsr)\s*#?(\d+)", parts[2].strip())
                    if shift_match:
                        shift = int(shift_match.group(1))
                    else:
                        return None, "unknown_shift", base_reg, index_reg, 0
                offset += (index_value << shift) & 0xFFFFFFFF

        return (int(base_value) + int(offset)) & 0xFFFFFFFF, "resolved", base_reg, index_reg, int(offset)

    def _assign_register(
        self,
        state: Dict[str, Optional[int]],
        dest: str,
        value: Optional[int],
    ) -> None:
        state[dest] = int(value) & 0xFFFFFFFF if value is not None else None

    def _apply_data_processing(
        self,
        mnemonic: str,
        operands: Sequence[str],
        state: Dict[str, Optional[int]],
        insn_address: int,
    ) -> bool:
        if not operands:
            return False
        dest = self._parse_register(operands[0])
        if dest is None or dest == "pc":
            return False

        if mnemonic in {"MOV", "MOVS", "MOVW"} and len(operands) >= 2:
            src_reg = self._parse_register(operands[1])
            if src_reg is not None:
                self._assign_register(state, dest, state.get(src_reg))
                return True
            immediate = self._parse_int(operands[1])
            self._assign_register(state, dest, immediate)
            return True

        if mnemonic == "MOVT" and len(operands) >= 2:
            immediate = self._parse_int(operands[1])
            if immediate is None:
                self._assign_register(state, dest, None)
                return True
            previous = state.get(dest) or 0
            self._assign_register(state, dest, (previous & 0xFFFF) | ((immediate & 0xFFFF) << 16))
            return True

        if mnemonic in {"ADD", "ADDS", "ADDW", "SUB", "SUBS", "SUBW", "ORR", "ORRS", "BIC", "BICS"} and len(operands) >= 2:
            if len(operands) == 2:
                lhs_reg = dest
                rhs_text = operands[1]
            else:
                lhs_reg = self._parse_register(operands[1])
                rhs_text = operands[2]
            lhs = state.get(lhs_reg) if lhs_reg is not None else None
            rhs_reg = self._parse_register(rhs_text)
            rhs = state.get(rhs_reg) if rhs_reg is not None else self._parse_int(rhs_text)
            if lhs is None or rhs is None:
                self._assign_register(state, dest, None)
                return True
            if mnemonic.startswith("ADD"):
                value = lhs + rhs
            elif mnemonic.startswith("SUB"):
                value = lhs - rhs
            elif mnemonic.startswith("ORR"):
                value = lhs | rhs
            else:
                value = lhs & (~rhs)
            self._assign_register(state, dest, value)
            return True

        if mnemonic in {"LSL", "LSLS", "LSR", "LSRS"} and len(operands) >= 2:
            source_reg = dest if len(operands) == 2 else self._parse_register(operands[1])
            amount_text = operands[1] if len(operands) == 2 else operands[2]
            source = state.get(source_reg) if source_reg is not None else None
            amount = self._parse_int(amount_text)
            if source is None or amount is None:
                self._assign_register(state, dest, None)
                return True
            if mnemonic.startswith("LSL"):
                self._assign_register(state, dest, source << amount)
            else:
                self._assign_register(state, dest, (source & 0xFFFFFFFF) >> amount)
            return True

        if mnemonic in {"ADR", "ADRL"} and len(operands) >= 2:
            immediate = self._parse_int(operands[1])
            self._assign_register(
                state,
                dest,
                None if immediate is None else self._pc_value(insn_address, self.thumb_mode) + immediate,
            )
            return True

        return False

    def _handle_pseudo_literal_load(
        self,
        mnemonic: str,
        operands: Sequence[str],
        state: Dict[str, Optional[int]],
    ) -> bool:
        if not mnemonic.startswith("LDR") or len(operands) < 2:
            return False
        dest = self._parse_register(operands[0])
        if dest is None or dest == "pc":
            return False
        source = str(operands[1] or "").strip()
        if "[" in source:
            return False
        if source.startswith("="):
            value = self._parse_int(source[1:])
            self._assign_register(state, dest, value)
            return True
        value = self._parse_int(source)
        if value is not None and self.is_mmio_address(value):
            self._assign_register(state, dest, value)
            return True
        return False

    def analyze(self) -> List[StaticMMIOAccess]:
        accesses: List[StaticMMIOAccess] = []
        for bb_addr in sorted(self.static_bbs):
            state: Dict[str, Optional[int]] = {}
            for insn in self.static_bbs.get(bb_addr, []) or []:
                pc = int(insn.get("address", 0) or 0)
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
                operands = self._split_operands(insn.get("operands", ""))
                access_type = self._access_type(mnemonic)

                if access_type and len(operands) >= 2:
                    address, kind, base_reg, index_reg, offset = self._resolve_memory_operand(
                        operands[1],
                        state,
                        pc,
                    )
                    if address is not None and self.is_mmio_address(address):
                        accesses.append(StaticMMIOAccess(
                            pc=pc,
                            bb_addr=int(bb_addr),
                            address=address,
                            access_type=access_type,
                            width=self._memory_width(mnemonic),
                            confidence="exact",
                            kind="abstract_interpretation",
                            base_reg=base_reg,
                            index_reg=index_reg,
                            offset=offset,
                            mnemonic=mnemonic,
                            operands=str(insn.get("operands", "") or ""),
                        ))
                    elif kind == "unknown_index_mmio_base":
                        accesses.append(StaticMMIOAccess(
                            pc=pc,
                            bb_addr=int(bb_addr),
                            address=None,
                            access_type=access_type,
                            width=self._memory_width(mnemonic),
                            confidence="base_only",
                            kind=kind,
                            base_reg=base_reg,
                            index_reg=index_reg,
                            offset=offset,
                            mnemonic=mnemonic,
                            operands=str(insn.get("operands", "") or ""),
                        ))

                # Update state after observing the memory access.
                if self._handle_pseudo_literal_load(mnemonic, operands, state):
                    continue

                if mnemonic.startswith("LDR") and len(operands) >= 2:
                    dest = self._parse_register(operands[0])
                    if dest is not None and dest != "pc":
                        address, kind, _base_reg, _index_reg, _offset = self._resolve_memory_operand(
                            operands[1],
                            state,
                            pc,
                        )
                        value = None
                        if kind in {"resolved", "absolute"} and address is not None:
                            value = self._read_literal_word(address)
                            if value is None and self.is_mmio_address(address):
                                value = None
                        self._assign_register(state, dest, value)
                    continue

                if self._apply_data_processing(mnemonic, operands, state, pc):
                    continue

                # Unknown writes to registers make later addresses unreliable.
                if operands:
                    dest = self._parse_register(operands[0])
                    if dest is not None and dest != "pc" and mnemonic not in {
                        "CMP", "CMN", "TST", "TEQ", "STR", "STRB", "STRH", "STRD",
                    }:
                        self._assign_register(state, dest, None)

        return self._dedupe_accesses(accesses)

    @staticmethod
    def _dedupe_accesses(accesses: Iterable[StaticMMIOAccess]) -> List[StaticMMIOAccess]:
        result: List[StaticMMIOAccess] = []
        seen = set()
        for access in accesses:
            key = (
                int(access.pc),
                int(access.address) if access.address is not None else None,
                access.access_type,
                access.width,
                access.kind,
                access.base_reg,
                access.index_reg,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(access)
        return result

    def build_function_summaries(self, accesses: Sequence[StaticMMIOAccess]) -> Dict[int, FunctionMMIOSummary]:
        """Build coarse function summaries from symbols and static MMIO accesses."""
        if not self.static_bbs:
            return {}
        function_entries = sorted(
            address for address in self.symbols_by_addr
            if address in self.static_bbs
        )
        if not function_entries:
            function_entries = sorted(self.static_bbs)

        def owner_function(address: int) -> int:
            address = int(address) & ~1
            candidate = function_entries[0]
            for entry in function_entries:
                if entry > address:
                    break
                candidate = entry
            return candidate

        summaries: Dict[int, FunctionMMIOSummary] = {
            entry: FunctionMMIOSummary(entry_addr=entry, name=self.symbols_by_addr.get(entry, ""))
            for entry in function_entries
        }
        for access in accesses:
            entry = owner_function(access.pc)
            summary = summaries.setdefault(
                entry,
                FunctionMMIOSummary(entry_addr=entry, name=self.symbols_by_addr.get(entry, "")),
            )
            if access.address is None:
                if access.access_type == "read":
                    summary.unresolved_reads += 1
                elif access.access_type == "write":
                    summary.unresolved_writes += 1
            elif access.access_type == "read":
                summary.reads.add(int(access.address) & 0xFFFFFFFF)
            elif access.access_type == "write":
                summary.writes.add(int(access.address) & 0xFFFFFFFF)

        for instructions in self.static_bbs.values():
            for insn in instructions or []:
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
                if mnemonic not in {"BL", "BLX"}:
                    continue
                pc = int(insn.get("address", 0) or 0)
                target = self._parse_direct_target(insn.get("operands", ""))
                caller = owner_function(pc)
                summary = summaries.setdefault(
                    caller,
                    FunctionMMIOSummary(entry_addr=caller, name=self.symbols_by_addr.get(caller, "")),
                )
                summary.callsites.append(pc)
                if target is not None:
                    summary.callees.add(owner_function(target))

        return summaries

    @staticmethod
    def _parse_direct_target(operands: object) -> Optional[int]:
        matches = re.findall(r"0x[0-9a-fA-F]+", str(operands or ""))
        if not matches:
            return None
        try:
            return int(matches[-1], 16) & ~1
        except Exception:
            return None


def analyze_static_mmio(
    static_bbs: Dict[int, List[Dict[str, object]]],
    instruction_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    symbols_by_addr: Optional[Dict[int, str]] = None,
    thumb_mode: bool = True,
    firmware_path: Optional[str | Path] = None,
    raw_load_base: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, object]]]:
    analyzer = LightweightMMIOAnalyzer(
        static_bbs=static_bbs,
        instruction_lookup=instruction_lookup,
        symbols_by_addr=symbols_by_addr,
        thumb_mode=thumb_mode,
        firmware_path=firmware_path,
        raw_load_base=raw_load_base,
    )
    accesses = analyzer.analyze()
    summaries = analyzer.build_function_summaries(accesses)
    return (
        [access.to_dict() for access in accesses],
        {int(entry): summary.to_dict() for entry, summary in summaries.items()},
    )


# mnemonics whose immediate operand can serve as an initial MMIO value clue
# (the value the firmware compares the loaded register against)
_IMMEDIATE_CLUE_MNEMONICS = {"CMP", "CMN", "SUBS"}


def _immediate_clue_after_read(
    static_bbs: Dict[int, List[Dict[str, object]]],
    bb_addr: int,
    read_pc: int,
    dest_reg: Optional[str],
) -> Optional[int]:
    """Find the first compare-immediate applied to ``dest_reg`` after a MMIO load.

    Scans forward inside the same basic block.  The scan stops when the loaded
    register is redefined (the MMIO value no longer flows to the compare) or at
    the end of the block.  A register-vs-register compare yields no numeric
    clue; later compares on the same register are still considered (a variable
    may carry several constraints).
    """
    if dest_reg is None:
        return None
    instructions = static_bbs.get(int(bb_addr)) or []
    started = False
    for insn in instructions or []:
        pc = int(insn.get("address", 0) or 0)
        if not started:
            if pc == int(read_pc):
                started = True
            continue
        mnemonic = LightweightMMIOAnalyzer._normalize_mnemonic(insn.get("mnemonic", ""))
        operands = LightweightMMIOAnalyzer._split_operands(insn.get("operands", ""))
        if not operands:
            continue
        first = LightweightMMIOAnalyzer._parse_register(operands[0])
        if first == dest_reg and mnemonic in _IMMEDIATE_CLUE_MNEMONICS and len(operands) >= 2:
            immediate = LightweightMMIOAnalyzer._parse_int(operands[1])
            if immediate is not None:
                return immediate & 0xFFFFFFFF
            # register-vs-register compare: no numeric clue from this instruction
            continue
        if (
            first == dest_reg
            and first is not None
            and mnemonic not in {"CMP", "CMN", "TST", "TEQ"}
            and not mnemonic.startswith(("B", "STR"))
        ):
            # dest register redefined before a usable compare: the loaded
            # MMIO value no longer reaches a numeric comparison
            return None
    return None


def derive_mmio_seed_table(
    static_bbs: Dict[int, List[Dict[str, object]]],
    static_mmio_accesses: List[Dict[str, object]],
    *,
    max_seeds: int = 4096,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    """Derive the initial MMIO input table {(address, occurrence=1): value}.

    Seed values follow the design's §2.3 policy: use the immediate the firmware
    compares the loaded value against when one is provable in the same basic
    block, otherwise 0.  The derivation is a pure function of the static
    analysis output, so the same firmware and configuration always produce the
    same table (determinism is what makes baseline comparisons reproducible).
    """
    seed_table: Dict[int, int] = {}
    meta = {
        "read_records": 0,
        "from_immediate": 0,
        "default_zero": 0,
    }
    records: List[Dict[str, object]] = []
    for item in static_mmio_accesses or []:
        if not isinstance(item, dict):
            continue
        address = item.get("address")
        if address is None:
            continue
        if str(item.get("access_type", "") or "") != "read":
            continue
        try:
            address_int = int(address) & 0xFFFFFFFF
        except (TypeError, ValueError):
            continue
        try:
            pc = int(item.get("pc", 0) or 0)
        except (TypeError, ValueError):
            pc = 0
        records.append((pc, address_int, item))
    records.sort(key=lambda entry: (entry[0], entry[1]))
    for _pc, address_int, item in records:
        if len(seed_table) >= max(1, int(max_seeds)):
            break
        if address_int in seed_table:
            continue
        meta["read_records"] += 1
        try:
            width = max(1, int(item.get("width", 4) or 4))
        except (TypeError, ValueError):
            width = 4
        load_operands = LightweightMMIOAnalyzer._split_operands(item.get("operands", ""))
        dest_reg = (
            LightweightMMIOAnalyzer._parse_register(load_operands[0])
            if load_operands
            else None
        )
        clue = _immediate_clue_after_read(
            static_bbs,
            int(item.get("bb_addr", 0) or 0),
            int(item.get("pc", 0) or 0),
            dest_reg,
        )
        if clue is not None:
            seed_table[address_int] = clue & ((1 << (8 * width)) - 1)
            meta["from_immediate"] += 1
        else:
            seed_table[address_int] = 0
            meta["default_zero"] += 1
    return seed_table, meta
