#!/usr/bin/env python3
"""Bounded dynamic expression recovery for external-input branch predicates.

The graph is built only from instructions observed in a concrete execution.  It
does not invent CFG edges or force control flow.  A solver model is therefore a
candidate environment witness and still requires the caller's force-free replay
validation before it can contribute coverage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .checksum_recovery import ChecksumInputSite, generate_checksum_hypotheses
from .runner_models import external_input_site_identity

try:
    import z3  # type: ignore
except Exception:  # pragma: no cover - minimal deployments use legacy fallback
    z3 = None


U32_MASK = 0xFFFFFFFF
InputKey = Tuple[str, int, int, int]


@dataclass(frozen=True, slots=True)
class ExpressionNode:
    op: str
    args: Tuple[int, ...] = field(default_factory=tuple)
    width: int = 32
    value: Optional[int] = None
    params: Tuple[int, ...] = field(default_factory=tuple)
    pc: int = 0
    label: str = ""


@dataclass(frozen=True, slots=True)
class DynamicInput:
    key: InputKey
    node_id: int
    kind: str
    address: int
    read_pc: int
    occurrence: int
    width: int
    observed_value: int
    event_index: int
    trace_event_id: int = 0


@dataclass(frozen=True, slots=True)
class DynamicBranchPredicate:
    branch_pc: int
    branch_bb: int
    occurrence: int
    order: int
    condition: str
    compare_op: str
    left_node: int
    right_node: int
    actual_taken: bool
    compare_pc: int = 0


@dataclass(frozen=True, slots=True)
class DynamicInputAssignment:
    kind: str
    address: int
    read_pc: int
    # None = pc 级 / 全出现语义（差分扰动见证，复审 round2 步骤 4 +
    # 第 3 轮收口）：探测在「全出现强制」下取证，见证验证同样覆盖该
    # 读点全部出现（add_pc_constraint），不再退化为「仅第 1 次出现」。
    # z3/LLM 来源恒为 int。
    occurrence: Optional[int]
    width: int
    value: int
    observed_value: int
    trace_event_id: int = 0


@dataclass(frozen=True, slots=True)
class DynamicConstraintModel:
    assignments: Tuple[DynamicInputAssignment, ...]
    strategy: str
    relation_kind: str
    branch_pc: int
    branch_occurrence: int
    path_predicates: int
    slice_nodes: int
    baseline_corrections: int = 0
    solver_status: str = "sat"
    causal_complete: bool = True
    omitted_path_predicates: int = 0
    branch_record_match: str = "exact"


@dataclass(frozen=True, slots=True)
class InstructionSliceLine:
    address: int
    mnemonic: str
    operands: str
    kind: str = "data"  # data | input | constraint | target_compare | target_branch
    detail: str = ""
    order: float = 0.0


@dataclass(frozen=True, slots=True)
class DynamicInstructionSlice:
    """Instruction-level view of one predicate's data-dependency slice.

    Equivalent to replaying from the located snapshot to the failing branch
    and trimming that trace: the expression graph is the recording of exactly
    that deterministic execution, so node creation order reproduces the
    dynamic instruction order without a second replay.
    """

    branch_pc: int
    branch_occurrence: int
    condition: str
    compare_op: str
    compare_pc: int
    lines: Tuple[InstructionSliceLine, ...] = field(default_factory=tuple)
    input_sites: Tuple[DynamicInput, ...] = field(default_factory=tuple)
    same_variable_constraints: int = 0
    truncated_instructions: int = 0
    truncated_nodes: bool = False
    unmapped_nodes: int = 0
    relation_kind: str = "unknown"

    def format_instruction_lines(self) -> List[str]:
        formatted: List[str] = []
        for line in self.lines:
            text = f"0x{int(line.address):08x}: {line.mnemonic} {line.operands}".strip()
            if line.detail:
                text = f"{text}  ; {line.detail}"
            formatted.append(text)
        return formatted

    def format_constraint_note(self) -> str:
        if self.truncated_instructions <= 0 and not self.truncated_nodes:
            return ""
        parts: List[str] = []
        if self.truncated_instructions > 0:
            parts.append(
                f"earliest {int(self.truncated_instructions)} instructions "
                "were truncated to fit the budget"
            )
        if self.truncated_nodes:
            parts.append(
                "the expression slice exceeded the node budget; only the "
                "collected portion is shown"
            )
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class DynamicRecoveryResult:
    models: Tuple[DynamicConstraintModel, ...]
    reason: str
    solver_backend: str
    target_inputs: int = 0
    path_predicates: int = 0
    slice_nodes: int = 0
    relation_kind: str = "unknown"
    solver_status: str = "not_run"
    fallback_eligible: bool = False
    omitted_path_predicates: int = 0
    branch_record_match: str = "none"
    solver_attempts: int = 1
    escalated: bool = False
    initial_solver_status: str = "not_run"
    initial_limits: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    final_limits: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)


class DynamicExpressionGraph:
    """Concrete-trace expression DAG with occurrence-scoped external leaves."""

    _COMPARE_OPS = {"CMP", "CMN", "TST", "TEQ"}
    _LOAD_OPS = {"LDR", "LDRB", "LDRH", "LDRSB", "LDRSH", "LDRD"}
    _STORE_OPS = {"STR", "STRB", "STRH", "STRD"}
    _CONDITIONS = {
        "EQ", "NE", "CS", "HS", "CC", "LO", "MI", "PL",
        "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE",
    }
    _FLAG_WRITERS = {
        "MOVS", "MVNS",
        "ADDS", "ADCS", "SUBS", "SBCS", "RSBS",
        "ANDS", "ORRS", "ORNS", "EORS", "BICS",
        "LSLS", "LSRS", "ASRS", "RORS", "RRXS", "MULS",
    }
    _NZ_ONLY_FLAG_RESULTS = {
        "MOVS", "MVNS", "ANDS", "ORRS", "ORNS", "EORS", "BICS",
        "LSLS", "LSRS", "ASRS", "RORS", "MULS",
    }

    def __init__(
        self,
        *,
        instruction_to_bb: Optional[Dict[int, int]] = None,
        max_nodes: Optional[int] = None,
        max_branch_records: Optional[int] = None,
    ) -> None:
        self.instruction_to_bb = instruction_to_bb or {}
        self.max_nodes = max(1024, int(max_nodes or self._env_int(
            "LSGEMU_DYNAMIC_EXPR_MAX_NODES", 120000
        )))
        self.max_branch_records = max(64, int(max_branch_records or self._env_int(
            "LSGEMU_DYNAMIC_EXPR_MAX_BRANCHES", 20000
        )))
        self.nodes: Dict[int, ExpressionNode] = {}
        self._next_node_id = 1
        self._constant_cache: Dict[Tuple[int, int], int] = {}
        self.register_nodes: Dict[str, int] = {}
        self.memory_nodes: Dict[int, int] = {}
        self.inputs: Dict[int, DynamicInput] = {}
        self.input_nodes_by_key: Dict[InputKey, int] = {}
        self.input_event_index = 0
        self._next_trace_event_id = 0
        self.last_compare: Optional[Tuple[str, int, int, int]] = None
        self.branch_occurrence_counts: Counter[int] = Counter()
        self.branch_records: Dict[Tuple[int, int], DynamicBranchPredicate] = {}
        self.branch_order: List[DynamicBranchPredicate] = []
        self._next_branch_order = 0
        self.checksum_selection_offsets: Counter[Tuple[int, int]] = Counter()
        self.stats: Counter[str] = Counter()

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default)), 0))
        except ValueError:
            return default

    @property
    def has_execution_data(self) -> bool:
        return bool(self.inputs or self.branch_order or self.nodes)

    @staticmethod
    def _normalize_mnemonic(value: object) -> str:
        raw = str(value or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B" and parts[1] in DynamicExpressionGraph._CONDITIONS:
            return f"B{parts[1]}"
        return parts[0]

    @staticmethod
    def _split_operands(value: object) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        square_depth = 0
        brace_depth = 0
        for char in str(value or ""):
            if char == "[":
                square_depth += 1
            elif char == "]":
                square_depth = max(0, square_depth - 1)
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(0, brace_depth - 1)
            if char == "," and square_depth == 0 and brace_depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    @staticmethod
    def _register(value: object) -> Optional[str]:
        text = str(value or "").strip().lower()
        match = re.fullmatch(r"(?:r(?:1[0-5]|[0-9])|sp|lr|pc)", text)
        return match.group(0) if match else None

    @staticmethod
    def _immediate(value: object) -> Optional[int]:
        text = str(value or "").strip().lower()
        if text.startswith("#"):
            text = text[1:]
        if not re.fullmatch(r"[-+]?(?:0x[0-9a-f]+|\d+)", text):
            return None
        try:
            return int(text, 0) & U32_MASK
        except ValueError:
            return None

    def _new_node(
        self,
        op: str,
        args: Iterable[int] = (),
        *,
        width: int = 32,
        value: Optional[int] = None,
        params: Iterable[int] = (),
        pc: int = 0,
        label: str = "",
    ) -> Optional[int]:
        if len(self.nodes) >= self.max_nodes:
            self.stats["nodes_dropped_capacity"] += 1
            return None
        effective_width = max(1, min(32, int(width or 32)))
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[node_id] = ExpressionNode(
            op=str(op),
            args=tuple(int(item) for item in args),
            width=effective_width,
            value=(int(value) if value is not None else None),
            params=tuple(int(item) for item in params),
            pc=int(pc) & U32_MASK,
            label=str(label or ""),
        )
        self.stats["nodes_created"] += 1
        return node_id

    def _constant(self, value: int, width: int = 32) -> Optional[int]:
        effective_width = max(1, min(32, int(width or 32)))
        mask = (1 << effective_width) - 1 if effective_width < 32 else U32_MASK
        key = (int(value) & mask, effective_width)
        cached = self._constant_cache.get(key)
        if cached is not None:
            return cached
        node_id = self._new_node("const", width=effective_width, value=key[0])
        if node_id is not None:
            self._constant_cache[key] = node_id
        return node_id

    def _unary(
        self,
        op: str,
        source: Optional[int],
        *,
        width: int = 32,
        params: Iterable[int] = (),
        pc: int = 0,
    ) -> Optional[int]:
        if source is None:
            return None
        return self._new_node(op, (source,), width=width, params=params, pc=pc)

    def _binary(
        self,
        op: str,
        left: Optional[int],
        right: Optional[int],
        *,
        width: int = 32,
        pc: int = 0,
    ) -> Optional[int]:
        if left is None or right is None:
            return None
        return self._new_node(op, (left, right), width=width, pc=pc)

    def _opaque(
        self,
        sources: Iterable[Optional[int]],
        *,
        pc: int,
        label: str,
        width: int = 32,
        concrete_value: Optional[int] = None,
    ) -> Optional[int]:
        retained: List[int] = []
        for source in sources:
            if source is None or int(source) not in self.nodes:
                continue
            if not self._leaf_nodes(int(source)):
                continue
            if int(source) not in retained:
                retained.append(int(source))
        if not retained:
            return None
        node = self._new_node(
            "opaque",
            retained,
            width=width,
            value=concrete_value,
            pc=pc,
            label=label,
        )
        if node is not None:
            self.stats["opaque_nodes"] += 1
            self.stats[f"opaque_{str(label or 'unknown').lower()}"] += 1
        return node

    @staticmethod
    def _operand_registers(parts: Sequence[str]) -> List[str]:
        registers: List[str] = []
        for part in parts:
            for token in re.findall(
                r"\b(?:r(?:1[0-5]|[0-9])|sp|lr|pc)\b",
                str(part or "").lower(),
            ):
                if token not in registers:
                    registers.append(token)
        return registers

    def _operand_node(
        self,
        operand: object,
        concrete_register: Optional[Callable[[str], Optional[int]]],
    ) -> Optional[int]:
        immediate = self._immediate(operand)
        if immediate is not None:
            return self._constant(immediate)
        register = self._register(operand)
        if register is None:
            return None
        existing = self.register_nodes.get(register)
        if existing is not None:
            return existing
        if concrete_register is None:
            return None
        concrete = concrete_register(register)
        if concrete is None:
            return None
        return self._constant(concrete)

    def _cast_input(self, node_id: Optional[int], width: int, signed: bool, pc: int) -> Optional[int]:
        if node_id is None:
            return None
        if width >= 32:
            return node_id
        return self._unary(
            "sext" if signed else "zext",
            node_id,
            width=32,
            params=(width,),
            pc=pc,
        )

    def observe_external_read(
        self,
        *,
        kind: str,
        read_pc: int,
        address: int,
        occurrence: int,
        size: int,
        observed_value: int,
        destination_register: Optional[str],
        signed: bool = False,
        trace_event_id: Optional[int] = None,
    ) -> Optional[int]:
        destination = self._register(destination_register)
        if destination is None:
            return None
        width = max(1, min(32, int(size or 1) * 8))
        normalized_kind = str(kind or "input").lower()
        key: InputKey = (
            normalized_kind,
            int(read_pc) & U32_MASK,
            int(address) & U32_MASK,
            max(1, int(occurrence or 1)),
        )
        leaf = self.input_nodes_by_key.get(key)
        if leaf is None:
            leaf = self._new_node("input", width=width, pc=read_pc)
            if leaf is None:
                self.register_nodes.pop(destination, None)
                return None
            self.input_event_index += 1
            if trace_event_id is None or int(trace_event_id or 0) <= 0:
                self._next_trace_event_id += 1
                normalized_trace_event_id = self._next_trace_event_id
            else:
                normalized_trace_event_id = int(trace_event_id)
                self._next_trace_event_id = max(
                    self._next_trace_event_id,
                    normalized_trace_event_id,
                )
            mask = (1 << width) - 1 if width < 32 else U32_MASK
            input_item = DynamicInput(
                key=key,
                node_id=leaf,
                kind=normalized_kind,
                address=int(address) & U32_MASK,
                read_pc=int(read_pc) & U32_MASK,
                occurrence=max(1, int(occurrence or 1)),
                width=width,
                observed_value=int(observed_value) & mask,
                event_index=self.input_event_index,
                trace_event_id=normalized_trace_event_id,
            )
            self.inputs[leaf] = input_item
            self.input_nodes_by_key[key] = leaf
            self.stats["input_leaves"] += 1
        register_node = self._cast_input(leaf, width, bool(signed), int(read_pc))
        if register_node is None:
            self.register_nodes.pop(destination, None)
            return None
        self.register_nodes[destination] = register_node
        return register_node

    def observe_memory_load(
        self,
        *,
        read_pc: int,
        address: int,
        size: int,
        observed_value: int,
        destination_register: Optional[str],
        external_input: bool = False,
        occurrence: int = 1,
        signed: bool = False,
        trace_event_id: Optional[int] = None,
        address_registers: Sequence[str] = (),
    ) -> Optional[int]:
        destination = self._register(destination_register)
        if destination is None:
            return None
        effective_size = max(1, min(4, int(size or 1)))
        address_nodes = [
            self.register_nodes[register]
            for register in dict.fromkeys(
                str(item).lower() for item in address_registers or ()
            )
            if register in self.register_nodes
            and self._leaf_nodes(self.register_nodes[register])
        ]
        if external_input:
            if address_nodes:
                self.stats["external_loads_with_symbolic_address"] += 1
            return self.observe_external_read(
                kind="external_memory",
                read_pc=read_pc,
                address=address,
                occurrence=occurrence,
                size=effective_size,
                observed_value=observed_value,
                destination_register=destination,
                signed=signed,
                trace_event_id=trace_event_id,
            )

        if not any(
            ((int(address) + offset) & U32_MASK) in self.memory_nodes
            for offset in range(effective_size)
        ):
            opaque = self._opaque(
                address_nodes,
                pc=read_pc,
                label="indirect_memory_load_address",
                width=effective_size * 8,
                concrete_value=int(observed_value),
            )
            register_node = self._cast_input(
                opaque,
                effective_size * 8,
                bool(signed),
                int(read_pc),
            )
            if register_node is None:
                self.register_nodes.pop(destination, None)
                return None
            self.register_nodes[destination] = register_node
            self.stats["opaque_indirect_memory_loads"] += 1
            return register_node

        byte_nodes: List[int] = []
        any_symbolic = False
        for offset in range(effective_size):
            byte_node = self.memory_nodes.get((int(address) + offset) & U32_MASK)
            if byte_node is None:
                byte_node = self._constant((int(observed_value) >> (8 * offset)) & 0xFF, 8)
            else:
                any_symbolic = any_symbolic or bool(self._leaf_nodes(byte_node))
            if byte_node is None:
                self.register_nodes.pop(destination, None)
                return None
            byte_nodes.append(byte_node)
        if effective_size == 1:
            assembled = byte_nodes[0]
        else:
            assembled = self._new_node(
                "concat_le",
                tuple(byte_nodes),
                width=effective_size * 8,
                pc=read_pc,
            )
        if address_nodes:
            assembled = self._opaque(
                [assembled, *address_nodes],
                pc=read_pc,
                label="indirect_memory_load",
                width=effective_size * 8,
                concrete_value=int(observed_value),
            )
        register_node = self._cast_input(
            assembled,
            effective_size * 8,
            bool(signed),
            int(read_pc),
        )
        if register_node is None:
            self.register_nodes.pop(destination, None)
            return None
        self.register_nodes[destination] = register_node
        if any_symbolic:
            self.stats["symbolic_memory_loads"] += 1
        return register_node

    def observe_memory_store(
        self,
        *,
        write_pc: int,
        address: int,
        size: int,
        source_register: Optional[str],
        concrete_value: int = 0,
        address_registers: Sequence[str] = (),
    ) -> None:
        source = self._register(source_register)
        effective_size = max(1, min(4, int(size or 1)))
        source_node = self.register_nodes.get(source) if source is not None else None
        address_nodes = [
            self.register_nodes[register]
            for register in dict.fromkeys(
                str(item).lower() for item in address_registers or ()
            )
            if register in self.register_nodes
            and self._leaf_nodes(self.register_nodes[register])
        ]
        if address_nodes:
            source_node = self._opaque(
                [source_node, *address_nodes],
                pc=write_pc,
                label="indirect_memory_store",
                width=32,
                concrete_value=int(concrete_value),
            )
            if source_node is not None:
                self.stats["opaque_indirect_memory_stores"] += 1
        for offset in range(effective_size):
            target = (int(address) + offset) & U32_MASK
            if source_node is None:
                self.memory_nodes.pop(target, None)
                continue
            byte_node = self._unary(
                "extract",
                source_node,
                width=8,
                params=(8 * offset, 8),
                pc=write_pc,
            )
            if byte_node is None:
                self.memory_nodes.pop(target, None)
            else:
                self.memory_nodes[target] = byte_node
        if source_node is not None and self._leaf_nodes(source_node):
            self.stats["symbolic_memory_stores"] += 1

    def observe_instruction(
        self,
        instruction: Dict[str, object],
        *,
        concrete_register: Optional[Callable[[str], Optional[int]]] = None,
        cpsr: Optional[int] = None,
    ) -> None:
        mnemonic = self._normalize_mnemonic(instruction.get("mnemonic"))
        parts = self._split_operands(instruction.get("operands"))
        pc = int(instruction.get("address", 0) or 0) & U32_MASK
        if not mnemonic:
            return

        condition = self._branch_condition(mnemonic)
        if condition is not None:
            self._observe_branch(
                pc,
                condition,
                parts,
                concrete_register=concrete_register,
                cpsr=cpsr,
            )
            return

        if mnemonic in self._COMPARE_OPS:
            if not self.inputs and not self.register_nodes:
                self.last_compare = None
                return
            if len(parts) < 2:
                self.last_compare = None
                return
            left = self._operand_node(parts[0], concrete_register)
            right = self._operand_node(parts[1], concrete_register)
            if left is None or right is None:
                self.last_compare = None
            else:
                if self._leaf_nodes(left) or self._leaf_nodes(right):
                    self.last_compare = (mnemonic, left, right, pc)
                else:
                    self.last_compare = None
            return

        destination = self._register(parts[0]) if parts else None
        if mnemonic in self._LOAD_OPS:
            if destination is not None:
                self.register_nodes.pop(destination, None)
            return
        if mnemonic in self._STORE_OPS or mnemonic in {"PUSH", "POP"}:
            return
        if destination is None:
            if mnemonic in {"BL", "BLX", "SVC"}:
                self.last_compare = None
            return

        if not self.inputs and not self.register_nodes:
            return

        result: Optional[int] = None
        if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
            result = self._operand_node(parts[1], concrete_register)
        elif mnemonic == "MOVT" and len(parts) >= 2:
            high = self._immediate(parts[1])
            low_source = self.register_nodes.get(destination)
            if high is not None and low_source is not None:
                low = self._unary("extract", low_source, width=16, params=(0, 16), pc=pc)
                high_node = self._constant(high & 0xFFFF, 16)
                if low is not None and high_node is not None:
                    result = self._new_node("concat_be", (high_node, low), width=32, pc=pc)
        elif mnemonic in {"MVN", "MVNS"} and len(parts) >= 2:
            result = self._unary("not", self._operand_node(parts[1], concrete_register), pc=pc)
        elif mnemonic in {"UXTB", "UXTH", "SXTB", "SXTH"} and len(parts) >= 2:
            width = 8 if mnemonic.endswith("B") else 16
            source = self._operand_node(parts[1], concrete_register)
            low = self._unary("extract", source, width=width, params=(0, width), pc=pc)
            result = self._unary(
                "sext" if mnemonic.startswith("SX") else "zext",
                low,
                width=32,
                params=(width,),
                pc=pc,
            )
        elif mnemonic in {"REV", "REV16", "REVSH"} and len(parts) >= 2:
            source = self._operand_node(parts[1], concrete_register)
            result = self._unary(mnemonic.lower(), source, width=32, pc=pc)
        elif mnemonic == "UBFX" and len(parts) >= 4:
            source = self._operand_node(parts[1], concrete_register)
            lsb = self._immediate(parts[2])
            width = self._immediate(parts[3])
            if source is not None and lsb is not None and width is not None and 0 < width <= 32 - lsb:
                field_node = self._unary(
                    "extract", source, width=width, params=(lsb, width), pc=pc
                )
                result = self._unary("zext", field_node, width=32, params=(width,), pc=pc)
        else:
            result = self._observe_arithmetic(
                mnemonic,
                parts,
                destination,
                pc,
                concrete_register,
            )

        flag_expression = self._flag_expression(
            mnemonic,
            parts,
            destination,
            result,
            pc,
            concrete_register,
        )

        if result is None:
            source_parts = list(parts[1:])
            if len(parts) == 2 and mnemonic in {
                "MOVT", "ADC", "ADCS", "SBC", "SBCS", "RRX", "RRXS",
            }:
                source_parts.insert(0, destination)
            source_nodes = [
                self.register_nodes.get(register)
                for register in self._operand_registers(source_parts)
            ]
            result = self._opaque(
                source_nodes,
                pc=pc,
                label=f"instruction:{mnemonic}",
            )

        if result is None or not self._leaf_nodes(result):
            self.register_nodes.pop(destination, None)
        else:
            self.register_nodes[destination] = result

        if self._sets_flags(mnemonic):
            self.last_compare = flag_expression

    def _observe_arithmetic(
        self,
        mnemonic: str,
        parts: Sequence[str],
        destination: str,
        pc: int,
        concrete_register: Optional[Callable[[str], Optional[int]]],
    ) -> Optional[int]:
        operation_map = {
            "ADD": "add", "ADDS": "add", "SUB": "sub", "SUBS": "sub",
            "RSB": "rsb", "RSBS": "rsb", "AND": "and", "ANDS": "and",
            "ORR": "or", "ORRS": "or", "ORN": "orn", "ORNS": "orn", "EOR": "xor",
            "EORS": "xor", "BIC": "bic", "BICS": "bic", "LSL": "shl",
            "LSLS": "shl", "LSR": "lshr", "LSRS": "lshr", "ASR": "ashr",
            "ASRS": "ashr", "ROR": "ror", "RORS": "ror", "MUL": "mul",
            "MULS": "mul",
        }
        op = operation_map.get(mnemonic)
        if op is None:
            return None
        if len(parts) > 3:
            # Shifted-register operands carry additional semantics that are not
            # represented by the simple binary node below.
            return None
        if len(parts) == 2:
            left = self.register_nodes.get(destination)
            if left is None:
                left = self._operand_node(destination, concrete_register)
            right = self._operand_node(parts[1], concrete_register)
        elif len(parts) >= 3:
            left = self._operand_node(parts[1], concrete_register)
            right = self._operand_node(parts[2], concrete_register)
        else:
            return None
        if not self._leaf_nodes(left) and not self._leaf_nodes(right):
            return None
        return self._binary(op, left, right, pc=pc)

    def _flag_expression(
        self,
        mnemonic: str,
        parts: Sequence[str],
        destination: str,
        result: Optional[int],
        pc: int,
        concrete_register: Optional[Callable[[str], Optional[int]]],
    ) -> Optional[Tuple[str, int, int, int]]:
        """Return an exact flags definition or None for an opaque writer."""
        if mnemonic not in self._FLAG_WRITERS or result is None:
            return None
        if not self._leaf_nodes(result):
            return None
        if mnemonic in self._NZ_ONLY_FLAG_RESULTS:
            zero = self._constant(0)
            if zero is None:
                return None
            return (mnemonic, int(result), int(zero), int(pc))
        if mnemonic not in {"ADDS", "SUBS", "RSBS"} or len(parts) > 3:
            return None
        if len(parts) == 2:
            left = self.register_nodes.get(destination)
            if left is None:
                left = self._operand_node(destination, concrete_register)
            right = self._operand_node(parts[1], concrete_register)
        elif len(parts) == 3:
            left = self._operand_node(parts[1], concrete_register)
            right = self._operand_node(parts[2], concrete_register)
        else:
            return None
        if left is None or right is None:
            return None
        if not self._leaf_nodes(left) and not self._leaf_nodes(right):
            return None
        return (mnemonic, int(left), int(right), int(pc))

    def capture_call_arguments(
        self,
        registers: Sequence[str] = ("r0", "r1", "r2", "r3"),
    ) -> Dict[str, int]:
        """Capture symbolic call arguments without assuming callee semantics."""
        return {
            str(register): int(node_id)
            for register in registers
            for node_id in (self.register_nodes.get(str(register)),)
            if node_id is not None and self._leaf_nodes(node_id)
        }

    def observe_call_return(
        self,
        *,
        call_pc: int,
        return_pc: int,
        argument_nodes: Dict[str, int],
        concrete_result: Optional[int],
        result_register: str = "r0",
        callee_target: Optional[int] = None,
        argument_value_before: Optional[int] = None,
    ) -> Optional[int]:
        """Preserve causal leaves across an unmodeled or partially modeled call."""
        current = self.register_nodes.get(str(result_register))
        symbolic_arguments = [
            int(node_id)
            for node_id in dict(argument_nodes or {}).values()
            if int(node_id) in self.nodes and self._leaf_nodes(int(node_id))
        ]
        if not symbolic_arguments:
            return current

        current_is_modeled_result = bool(
            current is not None
            and current not in set(symbolic_arguments)
            and self._leaf_nodes(current)
        )
        current_node = self.nodes.get(int(current)) if current is not None else None
        if current_is_modeled_result and current_node is not None and current_node.op != "opaque":
            self.stats["modeled_call_returns_preserved"] += 1
            return current
        opaque_sources: List[int] = []
        if current is not None and self._leaf_nodes(current):
            opaque_sources.append(int(current))
        for node_id in symbolic_arguments:
            if int(node_id) not in opaque_sources:
                opaque_sources.append(int(node_id))
        label = (
            f"call:0x{int(callee_target) & U32_MASK:08x}"
            if callee_target is not None
            else f"call_pc:0x{int(call_pc) & U32_MASK:08x}"
        )
        opaque = self._opaque(
            opaque_sources,
            pc=return_pc,
            label=label,
            concrete_value=concrete_result,
        )
        if opaque is None:
            self.register_nodes.pop(str(result_register), None)
            return None
        self.register_nodes[str(result_register)] = opaque
        self.stats["opaque_call_returns"] += 1
        return opaque

    @classmethod
    def _branch_condition(cls, mnemonic: str) -> Optional[str]:
        if mnemonic in {"CBZ", "CBNZ"}:
            return mnemonic
        if not mnemonic.startswith("B") or mnemonic in {"B", "BL", "BLX", "BX", "BXJ"}:
            return None
        suffix = mnemonic[1:]
        return suffix if suffix in cls._CONDITIONS else None

    @staticmethod
    def _sets_flags(mnemonic: str) -> bool:
        return mnemonic in DynamicExpressionGraph._FLAG_WRITERS

    @staticmethod
    def _condition_from_cpsr(condition: str, cpsr: Optional[int]) -> Optional[bool]:
        if cpsr is None:
            return None
        n = bool((int(cpsr) >> 31) & 1)
        z = bool((int(cpsr) >> 30) & 1)
        c = bool((int(cpsr) >> 29) & 1)
        v = bool((int(cpsr) >> 28) & 1)
        rules = {
            "EQ": z,
            "NE": not z,
            "CS": c,
            "HS": c,
            "CC": not c,
            "LO": not c,
            "MI": n,
            "PL": not n,
            "VS": v,
            "VC": not v,
            "HI": c and not z,
            "LS": (not c) or z,
            "GE": n == v,
            "LT": n != v,
            "GT": (not z) and (n == v),
            "LE": z or (n != v),
        }
        return rules.get(condition)

    def _observe_branch(
        self,
        branch_pc: int,
        condition: str,
        parts: Sequence[str],
        *,
        concrete_register: Optional[Callable[[str], Optional[int]]],
        cpsr: Optional[int],
    ) -> None:
        self.branch_occurrence_counts[int(branch_pc)] += 1
        occurrence = int(self.branch_occurrence_counts[int(branch_pc)])
        compare_op = ""
        compare_pc = 0
        left: Optional[int] = None
        right: Optional[int] = None
        actual_taken: Optional[bool]
        if condition in {"CBZ", "CBNZ"}:
            if not parts:
                return
            left = self._operand_node(parts[0], concrete_register)
            right = self._constant(0)
            compare_op = "CMP"
            concrete = None
            register = self._register(parts[0])
            if register is not None and concrete_register is not None:
                concrete = concrete_register(register)
            actual_taken = (
                (int(concrete or 0) == 0)
                if concrete is not None and condition == "CBZ"
                else (int(concrete or 0) != 0)
                if concrete is not None
                else None
            )
        else:
            if self.last_compare is None:
                self.stats["branches_without_expression"] += 1
                return
            compare_op, left, right, compare_pc = self.last_compare
            actual_taken = self._condition_from_cpsr(condition, cpsr)
        if left is None or right is None or actual_taken is None:
            self.stats["branches_without_expression"] += 1
            return
        leaves = self._leaf_nodes(left) | self._leaf_nodes(right)
        if not leaves:
            return
        record = DynamicBranchPredicate(
            branch_pc=int(branch_pc) & U32_MASK,
            branch_bb=int(self.instruction_to_bb.get(int(branch_pc), int(branch_pc))) & U32_MASK,
            occurrence=occurrence,
            order=self._next_branch_order,
            condition=str(condition),
            compare_op=str(compare_op),
            left_node=int(left),
            right_node=int(right),
            actual_taken=bool(actual_taken),
            compare_pc=int(compare_pc) & U32_MASK,
        )
        self._next_branch_order += 1
        self.branch_records[(record.branch_pc, record.occurrence)] = record
        self.branch_order.append(record)
        if len(self.branch_order) > self.max_branch_records:
            removed = self.branch_order.pop(0)
            self.branch_records.pop((removed.branch_pc, removed.occurrence), None)
            self.stats["branch_records_dropped"] += 1
        self.stats["input_dependent_branches"] += 1

    def _leaf_nodes(self, root: Optional[int]) -> Set[int]:
        if root is None or root not in self.nodes:
            return set()
        result: Set[int] = set()
        stack = [int(root)]
        visited: Set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node.op == "input":
                result.add(node_id)
            else:
                stack.extend(node.args)
        return result

    def _slice_nodes(self, roots: Iterable[int], limit: int) -> Optional[Set[int]]:
        result: Set[int] = set()
        stack = [int(root) for root in roots]
        while stack:
            node_id = stack.pop()
            if node_id in result:
                continue
            if len(result) >= max(1, int(limit)):
                return None
            node = self.nodes.get(node_id)
            if node is None:
                return None
            result.add(node_id)
            stack.extend(node.args)
        return result

    def find_branch_record(
        self,
        branch_pc: int,
        occurrence: int,
    ) -> Optional[DynamicBranchPredicate]:
        exact = self.branch_records.get((int(branch_pc) & U32_MASK, max(1, int(occurrence))))
        if exact is not None:
            return exact
        candidates = [
            item for item in self.branch_order
            if int(item.branch_pc) == (int(branch_pc) & U32_MASK)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(int(item.occurrence) - int(occurrence)))

    def count_branch_input_sources(
        self,
        branch_pc: int,
        occurrence: int,
    ) -> Optional[int]:
        """Pre-hoc coupled-input count: distinct input leaves of one predicate.

        Determined from the recorded expression graph, not from a failed solve.
        ``None`` means no retained branch predicate for this key.
        """
        record = self.branch_records.get(
            (int(branch_pc) & U32_MASK, max(1, int(occurrence)))
        ) or self.find_branch_record(branch_pc, occurrence)
        if record is None:
            return None
        return len(self._leaf_nodes(record.left_node) | self._leaf_nodes(record.right_node))

    def _slice_nodes_soft(
        self,
        roots: Iterable[int],
        hard_cap: int,
    ) -> Tuple[Set[int], bool]:
        """Best-effort transitive slice that degrades instead of failing.

        Unlike :meth:`_slice_nodes`, exceeding ``hard_cap`` keeps the collected
        prefix and reports the overflow, which is exactly the degradation an
        LLM-context export needs when the strict solver path already gave up.
        """
        result: Set[int] = set()
        stack = [int(root) for root in roots]
        exceeded = False
        while stack:
            node_id = stack.pop()
            if node_id in result:
                continue
            if len(result) >= max(1, int(hard_cap)):
                exceeded = True
                break
            node = self.nodes.get(node_id)
            if node is None:
                continue
            result.add(node_id)
            stack.extend(node.args)
        return result, exceeded

    def _same_variable_constraint_records(
        self,
        record: DynamicBranchPredicate,
        *,
        limit: int,
    ) -> List[DynamicBranchPredicate]:
        """Earlier input-dependent branches constraining the same leaves.

        Keeps *all* constraints on the shared variable (e.g. ``cmp r1,#0x20``
        then ``cmp r1,#0x10``) so a proposed witness cannot satisfy only the
        final comparison.
        """
        try:
            target_order_index = self.branch_order.index(record)
        except ValueError:
            return []
        selected: List[DynamicBranchPredicate] = []
        shared: Set[int] = self._leaf_nodes(record.left_node) | self._leaf_nodes(
            record.right_node
        )
        for previous in reversed(self.branch_order[:target_order_index]):
            if len(selected) >= max(0, int(limit)):
                break
            previous_leaves = self._leaf_nodes(previous.left_node) | self._leaf_nodes(
                previous.right_node
            )
            if not previous_leaves or not (previous_leaves & shared):
                continue
            shared |= previous_leaves
            selected.append(previous)
        selected.reverse()
        return selected

    def export_instruction_slice(
        self,
        *,
        branch_pc: int,
        occurrence: int,
        instruction_lookup: Optional[Dict[int, Dict]] = None,
        max_instructions: Optional[int] = None,
    ) -> Optional[DynamicInstructionSlice]:
        """Export the predicate slice as an ordered instruction sequence.

        Trimming rules (design §5.1): data-dependency closure only; all
        same-variable constraints retained; control dependencies excluded —
        only branches whose predicate shares input leaves with the target are
        listed, and only as constraints that must keep holding.
        """
        record = self.branch_records.get(
            (int(branch_pc) & U32_MASK, max(1, int(occurrence)))
        ) or self.find_branch_record(branch_pc, occurrence)
        if record is None:
            return None
        target_leaves = self._leaf_nodes(record.left_node) | self._leaf_nodes(
            record.right_node
        )
        if not target_leaves:
            return None

        node_budget = max(32, self._env_int("LSGEMU_LLM_SLICE_MAX_NODES", 16384))
        slice_nodes, truncated_nodes = self._slice_nodes_soft(
            (record.left_node, record.right_node), node_budget
        )
        constraint_records = self._same_variable_constraint_records(
            record,
            limit=self._env_int("LSGEMU_LLM_SLICE_MAX_PATH_CONSTRAINTS", 32),
        )
        for item in constraint_records:
            remaining = node_budget - len(slice_nodes)
            if remaining <= 0:
                truncated_nodes = True
                break
            slice_nodes.update(
                self._slice_nodes_soft((item.left_node, item.right_node), remaining)[0]
            )

        lookup = instruction_lookup or {}
        ordered_leaves = sorted(
            (leaf for leaf in target_leaves if leaf in self.inputs),
            key=lambda leaf: (
                int(self.inputs[leaf].trace_event_id or 0)
                or int(self.inputs[leaf].event_index),
                int(self.inputs[leaf].event_index),
            ),
        )
        ordered_inputs = [self.inputs[leaf] for leaf in ordered_leaves]
        input_index_by_leaf = {leaf: index for index, leaf in enumerate(ordered_leaves)}

        candidates: List[Tuple[float, InstructionSliceLine]] = []
        seen_nodes: Set[int] = set()
        unmapped = 0
        for node_id in sorted(slice_nodes):
            node = self.nodes.get(node_id)
            if node is None or node.op == "const":
                continue
            seen_nodes.add(node_id)
            address = int(node.pc) & U32_MASK
            instruction = lookup.get(address)
            if address == 0 or instruction is None:
                unmapped += 1
                continue
            detail = ""
            kind = "data"
            if node.op == "input" and node_id in self.inputs:
                kind = "input"
                input_item = self.inputs[node_id]
                index = input_index_by_leaf.get(node_id, 0)
                hex_width = max(2, (int(input_item.width) + 3) // 4)
                detail = (
                    f"input[{index}] {input_item.kind}[0x{input_item.address:08x}] "
                    f"occ={input_item.occurrence} width={input_item.width} "
                    f"current=0x{input_item.observed_value:0{hex_width}x}"
                )
            candidates.append((
                float(node_id),
                InstructionSliceLine(
                    address=address,
                    mnemonic=str(instruction.get("mnemonic") or ""),
                    operands=str(instruction.get("operands") or ""),
                    kind=kind,
                    detail=detail,
                ),
            ))

        # Same-variable constraints from earlier branch records, interleaved by
        # their operand recency (creation order) so they appear in execution
        # order relative to the data instructions.
        for item in constraint_records:
            anchor = float(max(int(item.left_node), int(item.right_node)))
            for offset, (address, kind, detail) in enumerate((
                (
                    int(item.compare_pc) & U32_MASK,
                    "constraint",
                    f"prior constraint, must keep holding: {item.condition} "
                    f"{'taken' if item.actual_taken else 'not-taken'}",
                ),
                (
                    int(item.branch_pc) & U32_MASK,
                    "constraint",
                    f"constraint branch ({item.condition})",
                ),
            )):
                instruction = lookup.get(address)
                if address == 0 or instruction is None:
                    unmapped += 1
                    continue
                candidates.append((
                    anchor + 0.25 + 0.25 * offset,
                    InstructionSliceLine(
                        address=address,
                        mnemonic=str(instruction.get("mnemonic") or ""),
                        operands=str(instruction.get("operands") or ""),
                        kind=kind,
                        detail=detail,
                    ),
                ))

        target_compare = lookup.get(int(record.compare_pc) & U32_MASK)
        if target_compare is not None and int(record.compare_pc) & U32_MASK:
            candidates.append((
                float("inf"),
                InstructionSliceLine(
                    address=int(record.compare_pc) & U32_MASK,
                    mnemonic=str(target_compare.get("mnemonic") or ""),
                    operands=str(target_compare.get("operands") or ""),
                    kind="target_compare",
                    detail=(
                        f"failing compare ({record.compare_op}); observed "
                        f"{'taken' if record.actual_taken else 'not-taken'}"
                    ),
                ),
            ))
        else:
            unmapped += 1
        target_branch = lookup.get(int(record.branch_pc) & U32_MASK)
        if target_branch is not None:
            candidates.append((
                float("inf") + 0.5,
                InstructionSliceLine(
                    address=int(record.branch_pc) & U32_MASK,
                    mnemonic=str(target_branch.get("mnemonic") or record.condition),
                    operands=str(target_branch.get("operands") or ""),
                    kind="target_branch",
                    detail=(
                        f"failing branch ({record.condition}); observed "
                        f"{'taken' if record.actual_taken else 'not-taken'}"
                    ),
                ),
            ))
        else:
            unmapped += 1

        candidates.sort(key=lambda item: item[0])
        deduped_by_address: Dict[int, InstructionSliceLine] = {}
        input_details: Dict[int, List[str]] = {}
        for _, line in candidates:
            address = int(line.address)
            if line.kind == "input" and line.detail:
                # One instruction may serve several occurrences of the same
                # read; merge their annotations onto the surviving line.
                input_details.setdefault(address, []).append(line.detail)
            if address in deduped_by_address:
                continue
            deduped_by_address[address] = line
        for address, details in input_details.items():
            line = deduped_by_address.get(address)
            if line is not None and details:
                deduped_by_address[address] = replace(line, detail="; ".join(details))
        deduped: List[InstructionSliceLine] = list(deduped_by_address.values())

        instruction_budget = max(1, int(max_instructions or self._env_int(
            "LSGEMU_LLM_SLICE_MAX_INSTRUCTIONS", 160
        )))
        truncated_instructions = 0
        if len(deduped) > instruction_budget:
            truncated_instructions = len(deduped) - instruction_budget
            deduped = deduped[truncated_instructions:]

        # Prefer the full branch mnemonic (BEQ/BNE...) over the bare condition
        # suffix stored in the record when the instruction index knows it.
        branch_mnemonic = (
            self._normalize_mnemonic(target_branch.get("mnemonic"))
            if target_branch is not None
            else ""
        ) or str(record.condition)

        return DynamicInstructionSlice(
            branch_pc=int(record.branch_pc) & U32_MASK,
            branch_occurrence=int(record.occurrence),
            condition=branch_mnemonic,
            compare_op=str(record.compare_op),
            compare_pc=int(record.compare_pc) & U32_MASK,
            lines=tuple(deduped),
            input_sites=tuple(ordered_inputs),
            same_variable_constraints=len(constraint_records),
            truncated_instructions=truncated_instructions,
            truncated_nodes=truncated_nodes,
            unmapped_nodes=unmapped,
            relation_kind=self._classify_relation(record, target_leaves, slice_nodes),
        )

    def recover_branch_inputs(
        self,
        *,
        branch_pc: int,
        occurrence: int,
        target_taken: bool,
        max_models: int = 4,
        max_inputs: Optional[int] = None,
        max_slice_nodes: Optional[int] = None,
        max_path_predicates: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        avoid_values_by_site: Optional[Dict[Tuple[object, ...], Set[int]]] = None,
        baseline_values_by_site: Optional[Dict[Tuple[object, ...], int]] = None,
    ) -> DynamicRecoveryResult:
        if z3 is None:
            return DynamicRecoveryResult(
                tuple(),
                "z3_unavailable",
                "unavailable",
                solver_status="unavailable",
                fallback_eligible=True,
            )
        exact_record = self.branch_records.get(
            (int(branch_pc) & U32_MASK, max(1, int(occurrence)))
        )
        record = exact_record or self.find_branch_record(branch_pc, occurrence)
        if record is None:
            return DynamicRecoveryResult(
                tuple(),
                "dynamic_branch_predicate_missing",
                "z3",
                solver_status="missing",
                fallback_eligible=True,
            )
        branch_record_match = "exact" if exact_record is not None else "nearest_retained"
        input_limit = max(1, int(max_inputs or self._env_int(
            "LSGEMU_DYNAMIC_SLICE_MAX_INPUTS", 256
        )))
        assignment_limit = max(1, self._env_int(
            "LSGEMU_DYNAMIC_MODEL_MAX_ASSIGNMENTS", 64
        ))
        node_limit = max(32, int(max_slice_nodes or self._env_int(
            "LSGEMU_DYNAMIC_SLICE_MAX_NODES", 8192
        )))
        path_limit = max(0, int(max_path_predicates if max_path_predicates is not None else self._env_int(
            "LSGEMU_DYNAMIC_PATH_PREDICATE_LIMIT", 128
        )))
        solver_timeout = max(1, int(timeout_ms or self._env_int(
            "LSGEMU_DYNAMIC_SOLVER_TIMEOUT_MS", 150
        )))

        target_leaves = self._leaf_nodes(record.left_node) | self._leaf_nodes(record.right_node)
        if not target_leaves:
            return DynamicRecoveryResult(
                tuple(),
                "target_predicate_has_no_input",
                "z3",
                solver_status="not_applicable",
            )
        if len(target_leaves) > input_limit:
            return DynamicRecoveryResult(
                tuple(),
                "dynamic_input_limit_exceeded",
                "z3",
                target_inputs=len(target_leaves),
                solver_status="budget",
                fallback_eligible=True,
            )

        selected_leaves = set(target_leaves)
        path_records: List[DynamicBranchPredicate] = []
        try:
            target_order_index = self.branch_order.index(record)
        except ValueError:
            target_order_index = len(self.branch_order)
        for previous in reversed(self.branch_order[:target_order_index]):
            if len(path_records) >= path_limit:
                break
            previous_leaves = self._leaf_nodes(previous.left_node) | self._leaf_nodes(previous.right_node)
            if not previous_leaves or not (previous_leaves & selected_leaves):
                continue
            if len(selected_leaves | previous_leaves) > input_limit:
                continue
            selected_leaves.update(previous_leaves)
            path_records.append(previous)
        path_records.reverse()

        roots = [record.left_node, record.right_node]
        for item in path_records:
            roots.extend((item.left_node, item.right_node))
        slice_nodes = self._slice_nodes(roots, node_limit)
        if slice_nodes is None:
            return DynamicRecoveryResult(
                tuple(),
                "dynamic_slice_node_limit_exceeded",
                "z3",
                target_inputs=len(target_leaves),
                path_predicates=len(path_records),
                solver_status="budget",
                fallback_eligible=True,
            )

        relation_kind = self._classify_relation(record, target_leaves, slice_nodes)
        cache: Dict[int, object] = {}
        variables: Dict[int, object] = {}
        target_formula = self._predicate_formula(record, cache, variables)
        if target_formula is None:
            return DynamicRecoveryResult(
                tuple(),
                "unsupported_dynamic_predicate",
                "z3",
                target_inputs=len(target_leaves),
                path_predicates=len(path_records),
                slice_nodes=len(slice_nodes),
                relation_kind=relation_kind,
                solver_status="unsupported",
                fallback_eligible=True,
            )
        desired_formula = target_formula if bool(target_taken) else z3.Not(target_formula)
        path_formulas: List[object] = []
        omitted_path_predicates = 0
        for item in path_records:
            formula = self._predicate_formula(item, cache, variables)
            if formula is None:
                omitted_path_predicates += 1
                continue
            path_formulas.append(formula if item.actual_taken else z3.Not(formula))

        left_leaves = self._leaf_nodes(record.left_node)
        right_leaves = self._leaf_nodes(record.right_node)
        repair_leaves = self._select_repair_leaves(left_leaves, right_leaves, target_leaves)
        models: List[DynamicConstraintModel] = []
        attempts: List[Set[int]] = []
        singleton_limit = max(1, self._env_int(
            "LSGEMU_DYNAMIC_SINGLETON_REPAIR_LIMIT", 8
        ))
        # Try minimum-cardinality repairs first for every relation.  Equality
        # over assembled fields will reject singleton attempts and naturally
        # fall through to the complete causal leaf set.
        if 1 < len(repair_leaves) <= singleton_limit:
            attempts.extend({leaf} for leaf in sorted(repair_leaves))
        attempts.append(set(repair_leaves))
        if repair_leaves != target_leaves:
            attempts.append(set(target_leaves))
        deduped_attempts: List[Set[int]] = []
        seen_attempts: Set[frozenset[int]] = set()
        for attempt in attempts:
            identity = frozenset(attempt)
            if not identity or identity in seen_attempts:
                continue
            seen_attempts.add(identity)
            deduped_attempts.append(attempt)
        attempts = deduped_attempts
        seen_model_signatures: Set[Tuple[Tuple[InputKey, int], ...]] = set()
        baseline_values = {
            tuple(site): int(value) & U32_MASK
            for site, value in (baseline_values_by_site or {}).items()
            if isinstance(site, (tuple, list))
        }

        solver = z3.Solver()
        solver.set("timeout", solver_timeout)
        solver.set("random_seed", 0)
        solver.add(desired_formula)
        solver.add(*path_formulas)
        for leaf in selected_leaves:
            variable = variables.get(leaf)
            input_item = self.inputs.get(leaf)
            if variable is None or input_item is None:
                continue
            site = self._input_site_identity(input_item)
            for avoided in (avoid_values_by_site or {}).get(site, set()):
                solver.add(variable != z3.BitVecVal(int(avoided), input_item.width))

        saw_unknown = False
        saw_unsat = False
        unknown_reasons: Counter[str] = Counter()
        for free_leaves in attempts:
            if len(models) >= max(1, int(max_models)):
                break
            solver.push()
            for leaf in selected_leaves:
                variable = variables.get(leaf)
                input_item = self.inputs.get(leaf)
                if variable is None or input_item is None:
                    continue
                if leaf not in free_leaves:
                    solver.add(variable == z3.BitVecVal(input_item.observed_value, input_item.width))

            check_result = solver.check()
            while len(models) < max(1, int(max_models)) and check_result == z3.sat:
                model = solver.model()
                assignments: List[DynamicInputAssignment] = []
                signature_items: List[Tuple[InputKey, int]] = []
                block_terms: List[object] = []
                for leaf in sorted(selected_leaves, key=lambda item: self.inputs[item].event_index):
                    input_item = self.inputs.get(leaf)
                    variable = variables.get(leaf)
                    if input_item is None or variable is None:
                        continue
                    value_expr = model.eval(variable, model_completion=True)
                    value = int(value_expr.as_long()) & (
                        (1 << input_item.width) - 1 if input_item.width < 32 else U32_MASK
                    )
                    signature_items.append((input_item.key, value))
                    site = self._input_site_identity(input_item)
                    baseline_value = baseline_values.get(site)
                    # A fixed leaf normally remains implicit because the
                    # concrete trace already supplied its value.  If the
                    # accepted prefix has since refined that same site, emit
                    # an explicit correction even when the solver kept the
                    # leaf fixed while preserving earlier predicates.
                    needs_baseline_correction = (
                        baseline_value is not None
                        and int(baseline_value) != int(value)
                    )
                    if leaf in free_leaves or needs_baseline_correction:
                        assignments.append(DynamicInputAssignment(
                            kind=input_item.kind,
                            address=input_item.address,
                            read_pc=input_item.read_pc,
                            occurrence=input_item.occurrence,
                            width=input_item.width,
                            value=value,
                            observed_value=input_item.observed_value,
                            trace_event_id=input_item.trace_event_id,
                        ))
                    if leaf in free_leaves:
                        block_terms.append(variable != z3.BitVecVal(value, input_item.width))
                signature = tuple(signature_items)
                changed_assignments = [
                    item for item in assignments
                    if (
                        int(item.value) != int(item.observed_value)
                        or baseline_values.get(
                            self._input_site_identity_from_assignment(item)
                        )
                        not in (None, int(item.value))
                    )
                ]
                if changed_assignments:
                    assignments = changed_assignments
                if len(assignments) > assignment_limit:
                    self.stats["models_dropped_assignment_limit"] += 1
                    break
                if signature not in seen_model_signatures and assignments:
                    seen_model_signatures.add(signature)
                    strategy = (
                        "dynamic_ssa_checksum_repair"
                        if "checksum" in relation_kind or "crc" in relation_kind
                        else "dynamic_ssa_multibyte"
                        if len(assignments) > 1
                        else "dynamic_ssa_cross_bb"
                    )
                    models.append(DynamicConstraintModel(
                        assignments=tuple(assignments),
                        strategy=strategy,
                        relation_kind=relation_kind,
                        branch_pc=record.branch_pc,
                        branch_occurrence=record.occurrence,
                        path_predicates=len(path_formulas),
                        slice_nodes=len(slice_nodes),
                        baseline_corrections=sum(
                            int(
                                baseline_values.get(
                                    self._input_site_identity_from_assignment(item)
                                )
                                is not None
                                and int(
                                    baseline_values.get(
                                        self._input_site_identity_from_assignment(item)
                                    )
                                )
                                != int(item.value)
                            )
                            for item in assignments
                        ),
                        solver_status="sat",
                        causal_complete=(
                            branch_record_match == "exact"
                            and not bool(omitted_path_predicates)
                        ),
                        omitted_path_predicates=omitted_path_predicates,
                        branch_record_match=branch_record_match,
                    ))
                if not block_terms:
                    break
                solver.add(z3.Or(*block_terms))
                check_result = solver.check()
            if check_result == z3.unknown:
                saw_unknown = True
                unknown_reasons[str(solver.reason_unknown() or "unknown")] += 1
            elif check_result == z3.unsat:
                saw_unsat = True
            solver.pop()
            if models:
                break

        reason = "dynamic_models" if models else "dynamic_predicate_unsat_or_timeout"
        solver_status = (
            "sat"
            if models
            else "unknown"
            if saw_unknown
            else "unsat"
            if saw_unsat
            else "not_run"
        )
        self.stats["solver_calls"] += 1
        self.stats["solver_models"] += len(models)
        self.stats[f"solver_status_{solver_status}"] += 1
        self.stats["omitted_path_predicates"] += omitted_path_predicates
        for unknown_reason, count in unknown_reasons.items():
            self.stats[f"solver_unknown_{unknown_reason}"] += count
        self.stats[f"relation_{relation_kind}"] += int(bool(models))
        return DynamicRecoveryResult(
            models=tuple(models),
            reason=reason,
            solver_backend="z3",
            target_inputs=len(target_leaves),
            path_predicates=len(path_formulas),
            slice_nodes=len(slice_nodes),
            relation_kind=relation_kind,
            solver_status=solver_status,
            fallback_eligible=bool(
                not models
                and (
                    solver_status in {"unknown", "not_run"}
                    or branch_record_match != "exact"
                )
            ),
            omitted_path_predicates=omitted_path_predicates,
            branch_record_match=branch_record_match,
        )

    def recover_checksum_candidates(
        self,
        *,
        branch_pc: int,
        occurrence: int,
        max_models: int = 4,
        trigger_status: str = "unsupported",
    ) -> DynamicRecoveryResult:
        """Generate deterministic checksum hypotheses from one observed slice."""
        exact_record = self.branch_records.get(
            (int(branch_pc) & U32_MASK, max(1, int(occurrence)))
        )
        record = exact_record or self.find_branch_record(branch_pc, occurrence)
        if record is None:
            return DynamicRecoveryResult(
                tuple(),
                "checksum_branch_predicate_missing",
                "checksum_enumerator",
                solver_status="missing",
                fallback_eligible=True,
            )
        branch_record_match = "exact" if exact_record is not None else "nearest_retained"
        target_leaves = self._leaf_nodes(record.left_node) | self._leaf_nodes(record.right_node)
        ordered_inputs = sorted(
            (
                self.inputs[leaf]
                for leaf in target_leaves
                if leaf in self.inputs
            ),
            key=lambda item: (
                int(item.trace_event_id or 0) or int(item.event_index),
                int(item.event_index),
            ),
        )
        max_sites = self._env_int("LSGEMU_CHECKSUM_MAX_INPUT_SITES", 256)
        if len(ordered_inputs) < 2 or len(ordered_inputs) > max_sites:
            return DynamicRecoveryResult(
                tuple(),
                "checksum_input_shape_not_applicable",
                "checksum_enumerator",
                target_inputs=len(ordered_inputs),
                relation_kind="unknown",
                solver_status="not_applicable",
                fallback_eligible=True,
                branch_record_match=branch_record_match,
            )
        sites = [
            ChecksumInputSite(
                site_index=index,
                kind=input_item.kind,
                address=input_item.address,
                read_pc=input_item.read_pc,
                occurrence=input_item.occurrence,
                width=input_item.width,
                observed_value=input_item.observed_value,
                event_index=input_item.event_index,
                trace_event_id=input_item.trace_event_id,
            )
            for index, input_item in enumerate(ordered_inputs)
        ]
        selection_key = (int(record.branch_pc), int(record.occurrence))
        selection_offset = int(self.checksum_selection_offsets[selection_key])
        hypotheses = generate_checksum_hypotheses(
            sites,
            max_hypotheses=max(1, int(max_models)),
            max_sites=max_sites,
            max_payload_bytes=self._env_int(
                "LSGEMU_CHECKSUM_MAX_PAYLOAD_BYTES", 1024
            ),
            selection_offset=selection_offset,
        )
        if hypotheses:
            self.checksum_selection_offsets[selection_key] += len(hypotheses)
        models: List[DynamicConstraintModel] = []
        for hypothesis in hypotheses:
            assignments: List[DynamicInputAssignment] = []
            for hypothesis_assignment in hypothesis.assignments:
                if not (0 <= int(hypothesis_assignment.site_index) < len(ordered_inputs)):
                    continue
                input_item = ordered_inputs[int(hypothesis_assignment.site_index)]
                assignments.append(DynamicInputAssignment(
                    kind=input_item.kind,
                    address=input_item.address,
                    read_pc=input_item.read_pc,
                    occurrence=input_item.occurrence,
                    width=input_item.width,
                    value=int(hypothesis_assignment.value),
                    observed_value=input_item.observed_value,
                    trace_event_id=input_item.trace_event_id,
                ))
            if not assignments:
                continue
            models.append(DynamicConstraintModel(
                assignments=tuple(assignments),
                strategy=(
                    f"deterministic_{hypothesis.algorithm}:"
                    f"{hypothesis.strategy}"
                ),
                relation_kind=f"checksum_hypothesis:{hypothesis.algorithm}",
                branch_pc=record.branch_pc,
                branch_occurrence=record.occurrence,
                path_predicates=0,
                slice_nodes=0,
                solver_status="hypothesis",
                causal_complete=False,
                omitted_path_predicates=0,
                branch_record_match=branch_record_match,
            ))
        self.stats["checksum_fallback_calls"] += 1
        self.stats["checksum_hypotheses"] += len(models)
        self.stats["checksum_selection_offset"] += selection_offset
        self.stats[f"checksum_trigger_{str(trigger_status or 'unknown')}"] += 1
        return DynamicRecoveryResult(
            models=tuple(models),
            reason=("checksum_hypotheses" if models else "checksum_no_hypothesis"),
            solver_backend="checksum_enumerator",
            target_inputs=len(ordered_inputs),
            path_predicates=0,
            slice_nodes=0,
            relation_kind=(
                models[0].relation_kind if models else "checksum_candidate"
            ),
            solver_status=("hypothesis" if models else "not_applicable"),
            fallback_eligible=not bool(models),
            branch_record_match=branch_record_match,
        )

    @staticmethod
    def _select_repair_leaves(
        left: Set[int],
        right: Set[int],
        target: Set[int],
    ) -> Set[int]:
        if left and not right:
            return set(left)
        if right and not left:
            return set(right)
        if left and right:
            smaller = left if len(left) <= len(right) else right
            if len(smaller) <= 4:
                return set(smaller)
        return set(target)

    @staticmethod
    def _input_site_identity(input_item: DynamicInput) -> Tuple[object, ...]:
        constraint_type = "mmio" if input_item.kind == "mmio" else "memory"
        input_kind = "mmio" if constraint_type == "mmio" else "external_memory"
        return external_input_site_identity(
            constraint_type=constraint_type,
            address=input_item.address,
            read_pc=input_item.read_pc,
            read_occurrence=input_item.occurrence,
            input_kind=input_kind,
        )

    @staticmethod
    def _input_site_identity_from_assignment(
        assignment: DynamicInputAssignment,
    ) -> Tuple[object, ...]:
        constraint_type = "mmio" if assignment.kind == "mmio" else "memory"
        input_kind = "mmio" if constraint_type == "mmio" else "external_memory"
        return external_input_site_identity(
            constraint_type=constraint_type,
            address=assignment.address,
            read_pc=assignment.read_pc,
            read_occurrence=assignment.occurrence,
            input_kind=input_kind,
        )

    def _classify_relation(
        self,
        record: DynamicBranchPredicate,
        target_leaves: Set[int],
        slice_nodes: Set[int],
    ) -> str:
        counts = Counter(self.nodes[node_id].op for node_id in slice_nodes if node_id in self.nodes)
        if len(target_leaves) > 2 and counts["xor"] >= 2 and (
            counts["shl"] + counts["lshr"] + counts["and"]
        ) >= 2:
            return "crc_or_xor_checksum"
        if len(target_leaves) > 2 and counts["add"] >= 2:
            return "additive_checksum"
        if len(target_leaves) > 2 and counts["xor"] >= 2:
            return "xor_checksum"
        if counts["opaque"]:
            return "opaque_input_relation"
        if len(target_leaves) > 1 and counts["or"] and counts["shl"]:
            return "multibyte_assembly"
        if len(target_leaves) > 1:
            return "multi_input_relation"
        compare_bb = self.instruction_to_bb.get(record.compare_pc, record.compare_pc)
        if int(compare_bb or 0) != int(record.branch_bb):
            return "cross_basic_block"
        input_item = self.inputs.get(next(iter(target_leaves)))
        if input_item is not None:
            read_bb = self.instruction_to_bb.get(input_item.read_pc, input_item.read_pc)
            if int(read_bb or 0) != int(record.branch_bb):
                return "cross_basic_block"
        return "local_dynamic"

    def _expr_to_z3(
        self,
        node_id: int,
        cache: Dict[int, object],
        variables: Dict[int, object],
    ):
        cached = cache.get(node_id)
        if cached is not None:
            return cached
        node = self.nodes.get(int(node_id))
        if node is None:
            return None
        if node.op == "const":
            result = z3.BitVecVal(int(node.value or 0), node.width)
        elif node.op == "input":
            input_item = self.inputs.get(int(node_id))
            if input_item is None:
                return None
            name = (
                f"input_{input_item.kind}_{input_item.read_pc:08x}_"
                f"{input_item.address:08x}_{input_item.occurrence}"
            )
            result = z3.BitVec(name, input_item.width)
            variables[int(node_id)] = result
        else:
            args = [self._expr_to_z3(item, cache, variables) for item in node.args]
            if any(item is None for item in args):
                return None
            result = self._apply_z3_node(node, args)
            if result is None:
                return None
        cache[int(node_id)] = result
        return result

    @staticmethod
    def _apply_z3_node(node: ExpressionNode, args: Sequence[object]):
        if node.op == "zext":
            return z3.ZeroExt(node.width - args[0].size(), args[0])
        if node.op == "sext":
            return z3.SignExt(node.width - args[0].size(), args[0])
        if node.op == "extract":
            lsb, width = node.params
            return z3.Extract(lsb + width - 1, lsb, args[0])
        if node.op == "concat_le":
            return z3.Concat(*reversed(args))
        if node.op == "concat_be":
            return z3.Concat(*args)
        if node.op == "not":
            return ~args[0]
        if node.op == "add":
            return args[0] + args[1]
        if node.op == "sub":
            return args[0] - args[1]
        if node.op == "rsb":
            return args[1] - args[0]
        if node.op == "and":
            return args[0] & args[1]
        if node.op == "or":
            return args[0] | args[1]
        if node.op == "orn":
            return args[0] | ~args[1]
        if node.op == "xor":
            return args[0] ^ args[1]
        if node.op == "bic":
            return args[0] & ~args[1]
        if node.op == "mul":
            return args[0] * args[1]
        if node.op == "shl":
            return args[0] << args[1]
        if node.op == "lshr":
            return z3.LShR(args[0], args[1])
        if node.op == "ashr":
            return args[0] >> args[1]
        if node.op == "ror":
            simplified = z3.simplify(args[1])
            if not z3.is_bv_value(simplified):
                return None
            return z3.RotateRight(args[0], simplified.as_long() & 0x1F)
        if node.op in {"rev", "rev16", "revsh"}:
            value = args[0]
            b0 = z3.Extract(7, 0, value)
            b1 = z3.Extract(15, 8, value)
            b2 = z3.Extract(23, 16, value)
            b3 = z3.Extract(31, 24, value)
            if node.op == "rev":
                return z3.Concat(b0, b1, b2, b3)
            if node.op == "rev16":
                return z3.Concat(b2, b3, b0, b1)
            low = z3.Concat(b0, b1)
            return z3.SignExt(16, low)
        return None

    def _predicate_formula(
        self,
        record: DynamicBranchPredicate,
        cache: Dict[int, object],
        variables: Dict[int, object],
    ):
        left = self._expr_to_z3(record.left_node, cache, variables)
        right = self._expr_to_z3(record.right_node, cache, variables)
        if left is None or right is None:
            return None
        width = max(int(left.size()), int(right.size()))
        if left.size() < width:
            left = z3.ZeroExt(width - left.size(), left)
        if right.size() < width:
            right = z3.ZeroExt(width - right.size(), right)

        compare_op = str(record.compare_op or "CMP").upper()
        nz_only = False
        if compare_op in {"CMP", "SUBS"}:
            result = left - right
            carry = z3.UGE(left, right)
            overflow = z3.And(
                z3.Extract(width - 1, width - 1, left)
                != z3.Extract(width - 1, width - 1, right),
                z3.Extract(width - 1, width - 1, result)
                != z3.Extract(width - 1, width - 1, left),
            )
        elif compare_op in {"CMN", "ADDS"}:
            extended = z3.ZeroExt(1, left) + z3.ZeroExt(1, right)
            result = z3.Extract(width - 1, 0, extended)
            carry = z3.Extract(width, width, extended) == z3.BitVecVal(1, 1)
            overflow = z3.And(
                z3.Extract(width - 1, width - 1, left)
                == z3.Extract(width - 1, width - 1, right),
                z3.Extract(width - 1, width - 1, result)
                != z3.Extract(width - 1, width - 1, left),
            )
        elif compare_op == "RSBS":
            result = right - left
            carry = z3.UGE(right, left)
            overflow = z3.And(
                z3.Extract(width - 1, width - 1, right)
                != z3.Extract(width - 1, width - 1, left),
                z3.Extract(width - 1, width - 1, result)
                != z3.Extract(width - 1, width - 1, right),
            )
        elif compare_op == "TST":
            result = left & right
            carry = z3.BoolVal(False)
            overflow = z3.BoolVal(False)
            nz_only = True
        elif compare_op == "TEQ":
            result = left ^ right
            carry = z3.BoolVal(False)
            overflow = z3.BoolVal(False)
            nz_only = True
        elif compare_op in self._NZ_ONLY_FLAG_RESULTS:
            result = left
            carry = z3.BoolVal(False)
            overflow = z3.BoolVal(False)
            nz_only = True
        else:
            return None

        zero = result == z3.BitVecVal(0, width)
        negative = z3.Extract(width - 1, width - 1, result) == z3.BitVecVal(1, 1)
        condition = str(record.condition or "").upper()
        if condition == "CBZ":
            return left == right
        if condition == "CBNZ":
            return left != right
        if nz_only and condition not in {"EQ", "NE", "MI", "PL"}:
            # Logical/result-only flag writers do not provide enough evidence
            # for preserved V or shifter-derived C in the concrete trace DAG.
            return None
        rules = {
            "EQ": zero,
            "NE": z3.Not(zero),
            "CS": carry,
            "HS": carry,
            "CC": z3.Not(carry),
            "LO": z3.Not(carry),
            "MI": negative,
            "PL": z3.Not(negative),
            "VS": overflow,
            "VC": z3.Not(overflow),
            "HI": z3.And(carry, z3.Not(zero)),
            "LS": z3.Or(z3.Not(carry), zero),
            "GE": negative == overflow,
            "LT": negative != overflow,
            "GT": z3.And(z3.Not(zero), negative == overflow),
            "LE": z3.Or(zero, negative != overflow),
        }
        return rules.get(condition)

    def summary(self) -> Dict[str, object]:
        relation_counts = {
            key[len("relation_"):]: int(value)
            for key, value in self.stats.items()
            if str(key).startswith("relation_")
        }
        return {
            "nodes": len(self.nodes),
            "inputs": len(self.inputs),
            "input_dependent_branches": len(self.branch_order),
            "branch_sequence": int(self._next_branch_order),
            "memory_shadow_bytes": len(self.memory_nodes),
            "solver_calls": int(self.stats.get("solver_calls", 0)),
            "solver_models": int(self.stats.get("solver_models", 0)),
            "relation_counts": relation_counts,
            "stats": dict(self.stats),
        }
