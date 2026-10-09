#!/usr/bin/env python3
"""Data models shared by the LSGEmu historical runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .evidence_contract import coerce_bool


@dataclass(slots=True)
class ExecutionTrace:
    covered_bbs: Set[int] = field(default_factory=set)
    # Recent sequence retained for diagnostics and local path tails. Exact
    # counts and successor edges are accumulated separately, so evicting an old
    # sequence entry cannot remove coverage or control-flow evidence.
    executed_bbs: List[int] = field(default_factory=list)
    total_executed_bbs: int = 0
    bb_counts: Dict[int, int] = field(default_factory=dict)
    successor_edges: Set[Tuple[int, int]] = field(default_factory=set)
    history_limit: int = 8192
    history_entries_discarded: int = 0
    _last_bb: Optional[int] = field(default=None, repr=False)

    def record_basic_block(self, bb_addr: int) -> bool:
        """Record one normalized BB transition and retain a bounded tail."""
        current = int(bb_addr)
        if self._last_bb == current:
            return False
        if self._last_bb is not None:
            self.successor_edges.add((int(self._last_bb), current))
        self._last_bb = current
        self.total_executed_bbs += 1
        self.bb_counts[current] = int(self.bb_counts.get(current, 0) or 0) + 1
        self.executed_bbs.append(current)

        limit = max(512, int(self.history_limit or 8192))
        if len(self.executed_bbs) > limit * 2:
            discarded = len(self.executed_bbs) - limit
            del self.executed_bbs[:discarded]
            self.history_entries_discarded += discarded
        return True

    def merge_successor_edges(self, edges: Set[Tuple[int, int]]) -> None:
        for source, target in edges or set():
            source_i = int(source)
            target_i = int(target)
            if source_i != target_i:
                self.successor_edges.add((source_i, target_i))


@dataclass(slots=True)
class PrefixReplaySnapshot:
    prefix_signature: Tuple[Tuple[Tuple[int, int], object], ...]
    next_branch_key: Tuple[int, int]
    snapshot: object
    mmio_state: Dict[int, int] = field(default_factory=dict)
    source_path_signature: Tuple[Tuple[Tuple[int, int], object], ...] = field(default_factory=tuple)
    branch_depth: int = 0
    origin: str = "unknown"
    evidence_class: str = "unknown"
    # The lossless dynamic branch prefix that reached this snapshot.  All
    # entries captured during one replay may share the same tuple and use the
    # prefix length as a cheap immutable view into it.
    source_occurrence_trace: Tuple[Tuple[Tuple[int, int], object], ...] = field(
        default_factory=tuple
    )
    source_occurrence_prefix_len: int = 0
    # Binary evidence lineage.  ``unspecified`` is retained for snapshots
    # serialized by older versions; it is not silently treated as validated.
    provenance_status: str = "unspecified"
    provenance_reasons: Tuple[str, ...] = field(default_factory=tuple)
    source_execution_id: str = ""
    prefix_telemetry_complete: bool = False
    prefix_intervention_reasons: Tuple[str, ...] = field(default_factory=tuple)
    # A wrapper may outlive the run that produced its underlying snapshot.
    # Keep finalization explicit so a pending/legacy ancestor cannot silently
    # become a validated replay when the wrapper is copied.
    provenance_finalized: bool = False
    provenance_invalidated: bool = False
    provenance_invalidation_reasons: Tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def _source_value(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, Mapping) and key in source:
            return source.get(key)
        return getattr(source, key, default)

    def __post_init__(self) -> None:
        """Propagate capture lineage when wrapping a BranchSnapshot.

        Many historical constructors only supplied the wrapped snapshot.  A
        post-init adapter keeps those call sites compatible while ensuring a
        diagnostic ancestor cannot disappear when the wrapper is copied.
        """
        source = self.snapshot
        if source is None:
            return
        nested = self._source_value(source, "execution_provenance", {})
        if not isinstance(nested, Mapping):
            nested = self._source_value(source, "snapshot_provenance", {})
        if not isinstance(nested, Mapping):
            nested = {}
        if self.provenance_status == "unspecified":
            inherited = self._source_value(source, "provenance_status", None)
            inherited = inherited or nested.get("status") or nested.get(
                "provenance_status"
            ) or "unspecified"
            if inherited:
                self.provenance_status = str(inherited)
        if not self.provenance_reasons:
            self.provenance_reasons = tuple(
                str(item)
                for item in (
                    self._source_value(source, "provenance_reasons", None)
                    or nested.get("reasons")
                    or nested.get("intervention_reasons")
                    or ()
                )
                if str(item)
            )
        if not self.source_execution_id:
            self.source_execution_id = str(
                self._source_value(source, "source_execution_id", None)
                or nested.get("execution_id")
                or ""
            )
        source_telemetry = self._source_value(source, "prefix_telemetry_complete", None)
        if source_telemetry is None:
            source_telemetry = nested.get("telemetry_complete", None)
        if source_telemetry is not None:
            self.prefix_telemetry_complete = coerce_bool(
                source_telemetry,
                self.prefix_telemetry_complete,
            )
        if not self.prefix_intervention_reasons:
            self.prefix_intervention_reasons = tuple(
                str(item)
                for item in (
                    self._source_value(source, "prefix_intervention_reasons", None)
                    or nested.get("prefix_intervention_reasons")
                    or nested.get("reasons")
                    or self.provenance_reasons
                    or ()
                )
                if str(item)
            )
        source_finalized = self._source_value(source, "provenance_finalized", None)
        if source_finalized is None:
            source_finalized = nested.get("provenance_finalized", None)
        if source_finalized is not None:
            self.provenance_finalized = coerce_bool(
                source_finalized,
                self.provenance_finalized,
            )
        source_invalidated = self._source_value(source, "provenance_invalidated", None)
        if source_invalidated is None:
            source_invalidated = nested.get("provenance_invalidated", None)
        if source_invalidated is not None:
            self.provenance_invalidated = coerce_bool(
                source_invalidated,
                self.provenance_invalidated,
            )
        source_invalidation_reasons = self._source_value(
            source,
            "provenance_invalidation_reasons",
            None,
        )
        if source_invalidation_reasons is None:
            source_invalidation_reasons = nested.get(
                "provenance_invalidation_reasons",
                None,
            )
        if source_invalidation_reasons:
            if isinstance(source_invalidation_reasons, str):
                source_invalidation_reasons = (source_invalidation_reasons,)
            self.provenance_invalidation_reasons = tuple(
                dict.fromkeys(
                    tuple(self.provenance_invalidation_reasons or ())
                    + tuple(
                        str(item)
                        for item in source_invalidation_reasons
                        if str(item)
                    )
                )
            )
        if self.provenance_invalidated:
            self.provenance_status = "diagnostic"
            self.prefix_telemetry_complete = False
            self.provenance_finalized = True


@dataclass
class ReservoirTaskContext:
    task_kind: str = "default"
    source_reason: str = ""
    budget_multiplier: float = 1.0
    stop_after_no_new_bbs: Optional[int] = None
    prefer_loop_intervention: bool = False
    hot_loop_halt_threshold: Optional[int] = None
    hot_loop_min_no_new_bbs: Optional[int] = None
    continuation_generation: int = 0
    continuation_origin_depth: int = 0
    continuation_origin_root: Optional[Tuple[int, int]] = None
    continuation_origin_path: Tuple[Tuple[Tuple[int, int], object], ...] = field(default_factory=tuple)
    source_stop_reason: str = ""
    source_merge_reachable_bbs: int = 0


@dataclass(frozen=True)
class BranchConstraintCandidate:
    constraint_type: str
    address: int
    value: int
    read_pc: Optional[int] = None
    constraint_pc: Optional[int] = None
    source: str = ""
    read_occurrence: Optional[int] = None
    input_kind: str = ""
    externally_controllable: Optional[bool] = None
    dependency_register: Optional[str] = None
    dependency_expression: str = ""
    dependency_group: str = ""
    composite_dependency: bool = False
    width: Optional[int] = None

    def normalized(self) -> "BranchConstraintCandidate":
        constraint_type = str(self.constraint_type or "").lower()
        normalized_width = (
            min(32, max(1, int(self.width)))
            if self.width is not None
            else None
        )
        value_mask = (
            (1 << normalized_width) - 1
            if normalized_width is not None and normalized_width < 32
            else 0xFFFFFFFF
        )
        externally_controllable = self.externally_controllable
        if externally_controllable is None:
            externally_controllable = constraint_type == "mmio"
        return BranchConstraintCandidate(
            constraint_type=constraint_type,
            address=int(self.address) & 0xFFFFFFFF,
            value=int(self.value) & value_mask,
            read_pc=int(self.read_pc) & 0xFFFFFFFF if self.read_pc is not None else None,
            constraint_pc=int(self.constraint_pc) & 0xFFFFFFFF if self.constraint_pc is not None else None,
            source=str(self.source or ""),
            read_occurrence=(
                max(1, int(self.read_occurrence))
                if self.read_occurrence is not None
                else None
            ),
            input_kind=str(self.input_kind or constraint_type),
            externally_controllable=bool(externally_controllable),
            dependency_register=(
                str(self.dependency_register).lower()
                if self.dependency_register is not None
                else None
            ),
            dependency_expression=str(self.dependency_expression or ""),
            dependency_group=str(self.dependency_group or ""),
            composite_dependency=bool(self.composite_dependency),
            width=normalized_width,
        )

    @property
    def is_environment_fact(self) -> bool:
        normalized = self.normalized()
        return bool(
            normalized.externally_controllable
            and (
                normalized.constraint_type in {"mmio", "stream", "event", "external_memory"}
                or (
                    normalized.constraint_type == "memory"
                    and normalized.input_kind == "external_memory"
                )
            )
        )


def external_input_site_identity(
    *,
    constraint_type: object,
    address: object,
    read_pc: object = None,
    read_occurrence: object = None,
    input_kind: object = "",
) -> Tuple[object, ...]:
    """Identify one runtime input delivery site, excluding provenance labels.

    A concrete replay handler can select a value by input type, address,
    consumer PC, and dynamic occurrence.  Fields such as dependency group,
    compare PC, source, and inferred width explain why a candidate exists, but
    they cannot make the same runtime read consume two different values.
    """

    normalized_type = str(constraint_type or "").lower()
    # ``input_kind`` describes provenance (for example, external SRAM versus
    # ordinary memory).  Delivery is still performed by the same runtime hook
    # selected by ``constraint_type`` and therefore cannot distinguish it.
    _ = input_kind
    normalized_read_pc = int(read_pc or 0) & 0xFFFFFFFF
    normalized_occurrence = (
        max(0, int(read_occurrence or 0))
        if normalized_read_pc
        else 0
    )
    return (
        normalized_type,
        int(address or 0) & 0xFFFFFFFF,
        normalized_read_pc,
        normalized_occurrence,
    )


def constraint_delivery_site_identity(
    candidate: BranchConstraintCandidate,
) -> Tuple[object, ...]:
    """Return the runtime identity at which a candidate value is delivered."""

    normalized = candidate.normalized()
    return external_input_site_identity(
        constraint_type=normalized.constraint_type,
        address=normalized.address,
        read_pc=normalized.read_pc,
        read_occurrence=normalized.read_occurrence,
        input_kind=normalized.input_kind,
    )


def serialize_constraint_candidate(candidate) -> Dict[str, object]:
    if isinstance(candidate, BranchConstraintCandidate):
        normalized = candidate.normalized()
        return {
            "type": normalized.constraint_type,
            "address": f"0x{normalized.address:08x}",
            "value": f"0x{normalized.value:08x}",
            "read_pc": f"0x{normalized.read_pc:08x}" if normalized.read_pc is not None else None,
            "constraint_pc": f"0x{normalized.constraint_pc:08x}" if normalized.constraint_pc is not None else None,
            "source": normalized.source or None,
            "read_occurrence": normalized.read_occurrence,
            "input_kind": normalized.input_kind or None,
            "externally_controllable": bool(normalized.externally_controllable),
            "dependency_register": normalized.dependency_register,
            "dependency_expression": normalized.dependency_expression or None,
            "dependency_group": normalized.dependency_group or None,
            "composite_dependency": bool(normalized.composite_dependency),
            "width": normalized.width,
        }
    if isinstance(candidate, tuple) and len(candidate) >= 3:
        address, value, read_pc = candidate[:3]
        return {
            "type": "mmio",
            "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
            "value": f"0x{int(value) & 0xFFFFFFFF:08x}",
            "read_pc": f"0x{int(read_pc) & 0xFFFFFFFF:08x}" if read_pc is not None else None,
            "constraint_pc": None,
            "source": None,
            "read_occurrence": None,
            "input_kind": "mmio",
            "externally_controllable": True,
            "dependency_register": None,
            "dependency_expression": None,
            "dependency_group": None,
            "composite_dependency": False,
            "width": None,
        }
    return {
        "type": "unknown",
        "address": None,
        "value": None,
        "read_pc": None,
        "constraint_pc": None,
        "source": None,
        "read_occurrence": None,
        "input_kind": None,
        "externally_controllable": False,
        "dependency_register": None,
        "dependency_expression": None,
        "dependency_group": None,
        "composite_dependency": False,
        "width": None,
    }


def normalize_constraint_candidate(candidate) -> Optional[BranchConstraintCandidate]:
    if isinstance(candidate, BranchConstraintCandidate):
        return candidate.normalized()
    if isinstance(candidate, tuple) and len(candidate) >= 3:
        address, value, read_pc = candidate[:3]
        return BranchConstraintCandidate(
            constraint_type="mmio",
            address=int(address),
            value=int(value),
            read_pc=int(read_pc) if read_pc is not None else None,
        ).normalized()
    return None


@dataclass(frozen=True)
class StreamInputProfile:
    """A replay-time input channel whose backing memory can be safely mutated."""

    profile_kind: str
    object_symbol: str
    object_addr: int
    buffer_pointer_offset: int
    head_offset: int
    tail_offset: int
    capacity: int
    inline_buffer_offset: Optional[int] = None
    absolute_buffer_addr: Optional[int] = None
    absolute_head_addr: Optional[int] = None
    absolute_tail_addr: Optional[int] = None
    absolute_length_addr: Optional[int] = None
    auxiliary_addr: Optional[int] = None
    auxiliary_buffer_addr: Optional[int] = None


STREAM_INPUT_CONSUMER_TERMS: Tuple[str, ...] = (
    "loop",
    "advance",
    "read",
    "available",
    "getc",
    "receive",
    "recv",
    "rx",
    "parse",
    "parser",
    "processinput",
    "processnextcommand",
    "getserialcommands",
    "getavailablecommands",
    "nextcommand",
    "queue",
    "serial",
    "uart",
    "usart",
    "xml",
    "packet",
    "pkt",
)


STREAM_RUNTIME_NONCONSUMER_CONTEXT_TERMS: Tuple[str, ...] = (
    "ticker",
    "systick",
    "timer",
    "rtt",
    "getcounter",
    "ztimer",
    "xtimer",
    "clock",
    "clk",
    "time",
    "timeout",
    "delay",
    "millis",
    "micros",
    "thread",
    "threadcreate",
    "threadmeasure",
    "threadisr",
    "isrstack",
    "stackusage",
    "stackpointer",
    "stackstart",
    "sched",
    "mutex",
    "lock",
    "malloc",
    "calloc",
    "free",
    "hwrng",
    "rng",
    "random",
    "idle",
    "pendsv",
    "isrpendsv",
    "isrsvc",
    "svc",
    "svcall",
    "hardfault",
    "memmanage",
    "busfault",
    "usagefault",
    "defaultisr",
    "defaulthandler",
)


STREAM_OUTPUT_OR_CONFIG_CONTEXT_TERMS: Tuple[str, ...] = (
    "write",
    "writable",
    "print",
    "println",
    "puts",
    "putc",
    "putchar",
    "printf",
    "flush",
    "echo",
    "report",
    "serialprint",
    "serialecho",
    "serialputc",
    "serialformat",
    "serialbaud",
    "format",
    "baud",
    "tx",
    "isatty",
    "size",
    "config",
    "regread",
    "readreg",
    "trxregread",
    "irq",
    "callbackif",
    "callbackassign",
    "callbackd2",
    "callbackc2",
    "workq",
    "workqueue",
    "zworkq",
    "stdouthook",
    "hookinstall",
    "alreadyattached",
    "send",
    "transmit",
    "begin",
    "setup",
    "foreach",
    "enable",
    "disable",
    "deinit",
    "init",
    "usartput",
    "serialput",
    "putudec",
)
