#!/usr/bin/env python3
"""Bounded causal execution context for concrete MCU replay.

The context records only transitions observed or explicitly modeled by the
emulator.  It is metadata for snapshot identity, replay validation, and audit;
it never forces a branch, interrupt, task switch, or peripheral value.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import os
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple


U32_MASK = 0xFFFFFFFF


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)), 0))
    except (TypeError, ValueError):
        return max(minimum, int(default))


@dataclass
class ContextTransition:
    sequence: int
    kind: str
    pc: int = 0
    address: Optional[int] = None
    value: Optional[int] = None
    size: int = 0
    task: str = "reset"
    irq_stack: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)

    def serializable(self) -> Dict[str, object]:
        item = asdict(self)
        item["irq_stack"] = list(self.irq_stack)
        return item


@dataclass
class DMAChannelState:
    channel: str
    phase: str = "idle"
    source: Optional[int] = None
    destination: Optional[int] = None
    count: Optional[int] = None
    started_sequence: int = 0
    completed_sequence: int = 0
    evidence: str = "observed"

    def serializable(self) -> Dict[str, object]:
        return asdict(self)


class CausalExecutionContext:
    """Snapshot-aware event state shared by MMIO, ISR, and replay layers."""

    SCHEMA = "lsgemu.causal_context.v1"

    def __init__(
        self,
        *,
        max_transitions: Optional[int] = None,
        max_pending_events: Optional[int] = None,
    ) -> None:
        self.max_transitions = max(
            16,
            int(max_transitions or _env_int(
                "LSGEMU_CAUSAL_CONTEXT_TRANSITION_LIMIT", 2048, minimum=16
            )),
        )
        self.max_pending_events = max(
            8,
            int(max_pending_events or _env_int(
                "LSGEMU_CAUSAL_CONTEXT_PENDING_EVENT_LIMIT", 128, minimum=8
            )),
        )
        self.active_task = "reset"
        self.irq_stack: List[int] = []
        self.pending_irqs: Set[int] = set()
        self.enabled_irqs: Set[int] = set()
        self.peripheral_phases: Dict[str, str] = {}
        self.stream_cursors: Dict[str, int] = {}
        self.dma_channels: Dict[str, DMAChannelState] = {}
        self.pending_events: Deque[Dict[str, object]] = deque(
            maxlen=self.max_pending_events
        )
        self.transitions: Deque[ContextTransition] = deque(
            maxlen=self.max_transitions
        )
        self.sequence = 0
        self.input_event_count = 0
        self.input_lineage_chain_sha256 = _stable_digest([])
        self.transition_chain_sha256 = _stable_digest([])
        self.dropped_transitions = 0
        self.dropped_pending_events = 0
        self.event_counts: Counter[str] = Counter()

    def _append_pending_event(self, event: Dict[str, object]) -> None:
        normalized = copy.deepcopy(dict(event))
        if any(
            item.get("kind") == normalized.get("kind")
            and item.get("key") == normalized.get("key")
            for item in self.pending_events
        ):
            return
        if len(self.pending_events) >= self.max_pending_events:
            self.dropped_pending_events += 1
        self.pending_events.append(normalized)

    def _remove_pending_event(self, kind: str, key: object) -> None:
        retained = deque(maxlen=self.max_pending_events)
        removed = False
        for item in self.pending_events:
            if (
                not removed
                and str(item.get("kind") or "") == str(kind)
                and item.get("key") == key
            ):
                removed = True
                continue
            retained.append(item)
        self.pending_events = retained

    def record_transition(
        self,
        kind: str,
        *,
        pc: int = 0,
        address: Optional[int] = None,
        value: Optional[int] = None,
        size: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ContextTransition:
        self.sequence += 1
        if len(self.transitions) >= self.max_transitions:
            self.dropped_transitions += 1
        transition = ContextTransition(
            sequence=int(self.sequence),
            kind=str(kind or "unknown"),
            pc=int(pc) & U32_MASK,
            address=(None if address is None else int(address) & U32_MASK),
            value=(None if value is None else int(value) & U32_MASK),
            size=max(0, int(size or 0)),
            task=str(self.active_task or "unknown"),
            irq_stack=tuple(int(irq) for irq in self.irq_stack),
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        self.transitions.append(transition)
        self.transition_chain_sha256 = _stable_digest({
            "previous": self.transition_chain_sha256,
            "transition": transition.serializable(),
        })
        self.event_counts[transition.kind] += 1
        return transition

    def record_mmio_read(
        self, pc: int, address: int, value: int, size: int, *, source: str
    ) -> None:
        self.record_transition(
            "mmio_read",
            pc=pc,
            address=address,
            value=value,
            size=size,
            metadata={"source": str(source or "modeled")},
        )

    def record_mmio_write(self, pc: int, address: int, value: int, size: int) -> None:
        self._apply_nvic_write(address, value)
        self.record_transition(
            "mmio_write", pc=pc, address=address, value=value, size=size
        )

    @staticmethod
    def _irq_bits(address: int, value: int, base: int) -> Iterable[int]:
        word_index = (int(address) - int(base)) // 4
        if word_index < 0:
            return ()
        irq_base = word_index * 32
        return tuple(
            irq_base + bit for bit in range(32) if int(value) & (1 << bit)
        )

    def _apply_nvic_write(self, address: int, value: int) -> None:
        address = int(address) & U32_MASK
        value = int(value) & U32_MASK
        if 0xE000E100 <= address < 0xE000E180:
            self.enabled_irqs.update(self._irq_bits(address, value, 0xE000E100))
        elif 0xE000E180 <= address < 0xE000E200:
            self.enabled_irqs.difference_update(
                self._irq_bits(address, value, 0xE000E180)
            )
        elif 0xE000E200 <= address < 0xE000E280:
            irqs = set(self._irq_bits(address, value, 0xE000E200))
            self.pending_irqs.update(irqs)
            for irq in sorted(irqs):
                self._append_pending_event({"kind": "irq", "key": int(irq)})
        elif 0xE000E280 <= address < 0xE000E300:
            irqs = set(self._irq_bits(address, value, 0xE000E280))
            self.pending_irqs.difference_update(irqs)
            for irq in sorted(irqs):
                self._remove_pending_event("irq", int(irq))

    def set_active_task(
        self,
        task: object,
        *,
        pc: int = 0,
        source: str = "observed",
    ) -> None:
        normalized = str(task if task is not None else "unknown")
        self._remove_pending_event("task", normalized)
        if normalized == self.active_task:
            return
        previous = self.active_task
        self.active_task = normalized
        self.record_transition(
            "task_switch",
            pc=pc,
            metadata={"previous": previous, "source": str(source or "observed")},
        )

    def record_task_wakeup(self, task: object, *, pc: int = 0, source: str = "observed") -> None:
        key = str(task if task is not None else "unknown")
        self._append_pending_event({"kind": "task", "key": key})
        self.record_transition(
            "task_wakeup", pc=pc, metadata={"task": key, "source": source}
        )

    def record_irq_pending(self, irq: int, *, pc: int = 0, source: str = "observed") -> None:
        irq = int(irq)
        self.pending_irqs.add(irq)
        self._append_pending_event({"kind": "irq", "key": irq})
        self.record_transition(
            "irq_pending", pc=pc, metadata={"irq": irq, "source": source}
        )

    def record_irq_deliver(self, irq: int, *, pc: int = 0, source: str = "observed") -> None:
        irq = int(irq)
        self.pending_irqs.discard(irq)
        self._remove_pending_event("irq", irq)
        self.irq_stack.append(irq)
        self.record_transition(
            "irq_deliver", pc=pc, metadata={"irq": irq, "source": source}
        )

    def record_irq_return(self, irq: Optional[int] = None, *, pc: int = 0) -> None:
        expected = None if irq is None else int(irq)
        returned = None
        if self.irq_stack:
            if expected is None or self.irq_stack[-1] == expected:
                returned = self.irq_stack.pop()
            elif expected in self.irq_stack:
                index = len(self.irq_stack) - 1 - self.irq_stack[::-1].index(expected)
                returned = self.irq_stack.pop(index)
        self.record_transition(
            "irq_return",
            pc=pc,
            metadata={"irq": returned, "requested_irq": expected},
        )

    def mark_peripheral_phase(
        self,
        peripheral: object,
        phase: str,
        *,
        pc: int = 0,
        address: Optional[int] = None,
        source: str = "observed",
    ) -> None:
        key = str(peripheral or "unknown")
        normalized_phase = str(phase or "unknown")
        previous = self.peripheral_phases.get(key)
        self.peripheral_phases[key] = normalized_phase
        if previous == normalized_phase:
            return
        self.record_transition(
            "peripheral_phase",
            pc=pc,
            address=address,
            metadata={
                "peripheral": key,
                "previous": previous,
                "phase": normalized_phase,
                "source": str(source or "observed"),
            },
        )

    def record_input_ready(
        self,
        stream_key: object,
        *,
        pc: int = 0,
        address: Optional[int] = None,
        source: str = "modeled",
    ) -> None:
        key = str(stream_key or "default")
        already_pending = any(
            item.get("kind") == "stream" and item.get("key") == key
            for item in self.pending_events
        )
        if already_pending:
            self.event_counts["input_ready_reobserved"] += 1
            return
        self._append_pending_event({"kind": "stream", "key": key})
        self.record_transition(
            "input_ready",
            pc=pc,
            address=address,
            metadata={"stream": key, "source": source},
        )

    def record_input_consume(
        self,
        stream_key: object,
        *,
        pc: int = 0,
        address: Optional[int] = None,
        value: Optional[int] = None,
        source: str = "modeled",
    ) -> None:
        key = str(stream_key or "default")
        self._remove_pending_event("stream", key)
        self.stream_cursors[key] = int(self.stream_cursors.get(key, 0) or 0) + 1
        self._record_input_lineage({
            "kind": "stream",
            "stream": key,
            "cursor": int(self.stream_cursors[key]),
            "source": str(source or "modeled"),
        })
        self.record_transition(
            "input_consume",
            pc=pc,
            address=address,
            value=value,
            metadata={
                "stream": key,
                "cursor": int(self.stream_cursors[key]),
                "source": source,
            },
        )

    def record_external_input(
        self,
        *,
        kind: str,
        pc: int,
        address: int,
        occurrence: int,
        trace_event_id: int,
        value: int,
        size: int,
    ) -> None:
        self.input_event_count += 1
        self._record_input_lineage({
            "kind": str(kind or "external"),
            "pc": int(pc) & U32_MASK,
            "address": int(address) & U32_MASK,
            "occurrence": max(1, int(occurrence or 1)),
            "trace_event_id": max(0, int(trace_event_id or 0)),
            "size": max(1, int(size or 1)),
        })
        self.record_transition(
            "external_input",
            pc=pc,
            address=address,
            value=value,
            size=size,
            metadata={
                "kind": str(kind or "external"),
                "occurrence": max(1, int(occurrence or 1)),
                "trace_event_id": max(0, int(trace_event_id or 0)),
            },
        )

    def _record_input_lineage(self, identity: Dict[str, object]) -> None:
        self.input_lineage_chain_sha256 = _stable_digest({
            "previous": self.input_lineage_chain_sha256,
            "input": dict(identity or {}),
        })

    def record_dma_start(
        self,
        channel: object,
        *,
        pc: int = 0,
        source: Optional[int] = None,
        destination: Optional[int] = None,
        count: Optional[int] = None,
        evidence: str = "observed",
    ) -> None:
        key = str(channel or "dma")
        state = self.dma_channels.get(key) or DMAChannelState(channel=key)
        state.phase = "active"
        state.source = None if source is None else int(source) & U32_MASK
        state.destination = (
            None if destination is None else int(destination) & U32_MASK
        )
        state.count = None if count is None else max(0, int(count))
        state.started_sequence = self.sequence + 1
        state.evidence = str(evidence or "observed")
        self.dma_channels[key] = state
        self._append_pending_event({"kind": "dma", "key": key})
        self.record_transition(
            "dma_start",
            pc=pc,
            metadata={
                "channel": key,
                "source": state.source,
                "destination": state.destination,
                "count": state.count,
                "evidence": state.evidence,
            },
        )

    def record_dma_complete(
        self, channel: object, *, pc: int = 0, evidence: str = "observed"
    ) -> None:
        key = str(channel or "dma")
        state = self.dma_channels.get(key) or DMAChannelState(channel=key)
        state.phase = "complete"
        state.completed_sequence = self.sequence + 1
        state.evidence = str(evidence or state.evidence)
        self.dma_channels[key] = state
        self._remove_pending_event("dma", key)
        self.record_transition(
            "dma_complete",
            pc=pc,
            metadata={"channel": key, "evidence": state.evidence},
        )

    def transition_digest(self) -> str:
        return str(self.transition_chain_sha256)

    def branch_signature(self) -> Dict[str, object]:
        """Return the context dimensions that must agree at one branch event."""
        pending_irqs = sorted(self.pending_irqs)
        enabled_irqs = sorted(self.enabled_irqs)
        stream_cursors = dict(sorted(self.stream_cursors.items()))
        dma_phases = {
            key: state.phase for key, state in sorted(self.dma_channels.items())
        }
        peripheral_phases = dict(sorted(self.peripheral_phases.items()))
        pending_event_lineage = [
            {
                "kind": str(item.get("kind") or "unknown"),
                "key": item.get("key"),
            }
            for item in self.pending_events
        ]
        hard_components = {
            "active_task": str(self.active_task),
            "irq_stack": list(self.irq_stack),
            "pending_irqs_sha256": _stable_digest(pending_irqs),
            "enabled_irqs_sha256": _stable_digest(enabled_irqs),
            "stream_cursors_sha256": _stable_digest(stream_cursors),
            "input_event_count": int(self.input_event_count),
            "input_lineage_sha256": str(self.input_lineage_chain_sha256),
            "pending_events_sha256": _stable_digest(pending_event_lineage),
            "dma_phases_sha256": _stable_digest(dma_phases),
        }
        return {
            "schema": self.SCHEMA,
            **hard_components,
            "peripheral_phases_sha256": _stable_digest(peripheral_phases),
            "hard_compatibility_sha256": _stable_digest(hard_components),
            "transition_sequence": int(self.sequence),
            "transition_digest": self.transition_digest(),
        }

    def fingerprint_components(self) -> Dict[str, object]:
        signature = self.branch_signature()
        return {
            **signature,
            "pending_irqs": sorted(self.pending_irqs),
            "enabled_irqs": sorted(self.enabled_irqs),
            "stream_cursors": dict(sorted(self.stream_cursors.items())),
            "peripheral_phases": dict(sorted(self.peripheral_phases.items())),
            "pending_events": [copy.deepcopy(item) for item in self.pending_events],
            "dma_channels": {
                key: state.serializable()
                for key, state in sorted(self.dma_channels.items())
            },
            "event_counts": dict(sorted(self.event_counts.items())),
            "retained_transitions": len(self.transitions),
            "dropped_transitions": int(self.dropped_transitions),
            "dropped_pending_events": int(self.dropped_pending_events),
        }

    def snapshot_runtime_state(
        self,
        *,
        include_transitions: bool = False,
        copy_values: bool = True,
    ) -> Dict[str, object]:
        """Return the replay state in the historical mapping schema.

        ``copy_values=False`` is an internal capture mode used only when the
        caller serializes the returned mapping immediately.  Mutable
        containers are then borrowed until that serialization completes,
        avoiding a second deep object graph at every branch/input snapshot.
        The default remains a detached snapshot for existing callers.
        """
        if not copy_values:
            state = {
                "schema": self.SCHEMA,
                "max_transitions": int(self.max_transitions),
                "max_pending_events": int(self.max_pending_events),
                "active_task": str(self.active_task),
                "irq_stack": self.irq_stack,
                "pending_irqs": self.pending_irqs,
                "enabled_irqs": self.enabled_irqs,
                "peripheral_phases": self.peripheral_phases,
                "stream_cursors": self.stream_cursors,
                "dma_channels": {
                    key: channel.serializable()
                    for key, channel in self.dma_channels.items()
                },
                "pending_events": self.pending_events,
                "sequence": int(self.sequence),
                "transition_chain_sha256": str(self.transition_chain_sha256),
                "input_event_count": int(self.input_event_count),
                "input_lineage_chain_sha256": str(
                    self.input_lineage_chain_sha256
                ),
                "dropped_transitions": int(self.dropped_transitions),
                "dropped_pending_events": int(self.dropped_pending_events),
                "event_counts": self.event_counts,
            }
            if include_transitions:
                # The restore contract expects serializable mappings rather
                # than ContextTransition instances.  This optional history is
                # intentionally bounded by max_transitions.
                state["transitions"] = [
                    transition.serializable() for transition in self.transitions
                ]
            return state

        state = {
            "schema": self.SCHEMA,
            "max_transitions": int(self.max_transitions),
            "max_pending_events": int(self.max_pending_events),
            "active_task": str(self.active_task),
            "irq_stack": list(self.irq_stack),
            "pending_irqs": sorted(self.pending_irqs),
            "enabled_irqs": sorted(self.enabled_irqs),
            "peripheral_phases": dict(self.peripheral_phases),
            "stream_cursors": dict(self.stream_cursors),
            "dma_channels": {
                key: state.serializable() for key, state in self.dma_channels.items()
            },
            "pending_events": [copy.deepcopy(item) for item in self.pending_events],
            "sequence": int(self.sequence),
            "transition_chain_sha256": str(self.transition_chain_sha256),
            "input_event_count": int(self.input_event_count),
            "input_lineage_chain_sha256": str(
                self.input_lineage_chain_sha256
            ),
            "dropped_transitions": int(self.dropped_transitions),
            "dropped_pending_events": int(self.dropped_pending_events),
            "event_counts": dict(self.event_counts),
        }
        if include_transitions:
            state["transitions"] = [
                transition.serializable() for transition in self.transitions
            ]
        return state

    def restore_runtime_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self.active_task = str(state.get("active_task") or "reset")
        self.irq_stack = [int(item) for item in state.get("irq_stack", []) or []]
        self.pending_irqs = {
            int(item) for item in state.get("pending_irqs", []) or []
        }
        self.enabled_irqs = {
            int(item) for item in state.get("enabled_irqs", []) or []
        }
        self.peripheral_phases = {
            str(key): str(value)
            for key, value in dict(state.get("peripheral_phases", {}) or {}).items()
        }
        self.stream_cursors = {
            str(key): max(0, int(value))
            for key, value in dict(state.get("stream_cursors", {}) or {}).items()
        }
        self.dma_channels = {}
        for key, raw in dict(state.get("dma_channels", {}) or {}).items():
            if not isinstance(raw, dict):
                continue
            self.dma_channels[str(key)] = DMAChannelState(
                channel=str(raw.get("channel") or key),
                phase=str(raw.get("phase") or "idle"),
                source=(
                    None if raw.get("source") is None else int(raw["source"]) & U32_MASK
                ),
                destination=(
                    None
                    if raw.get("destination") is None
                    else int(raw["destination"]) & U32_MASK
                ),
                count=(None if raw.get("count") is None else max(0, int(raw["count"]))),
                started_sequence=max(0, int(raw.get("started_sequence", 0) or 0)),
                completed_sequence=max(0, int(raw.get("completed_sequence", 0) or 0)),
                evidence=str(raw.get("evidence") or "observed"),
            )
        self.pending_events = deque(
            (copy.deepcopy(dict(item)) for item in state.get("pending_events", []) or []),
            maxlen=self.max_pending_events,
        )
        restored_transitions: List[ContextTransition] = []
        for raw in state.get("transitions", []) or []:
            if not isinstance(raw, dict):
                continue
            restored_transitions.append(ContextTransition(
                sequence=max(0, int(raw.get("sequence", 0) or 0)),
                kind=str(raw.get("kind") or "unknown"),
                pc=int(raw.get("pc", 0) or 0) & U32_MASK,
                address=(
                    None if raw.get("address") is None else int(raw["address"]) & U32_MASK
                ),
                value=(None if raw.get("value") is None else int(raw["value"]) & U32_MASK),
                size=max(0, int(raw.get("size", 0) or 0)),
                task=str(raw.get("task") or "reset"),
                irq_stack=tuple(int(item) for item in raw.get("irq_stack", []) or []),
                metadata=copy.deepcopy(dict(raw.get("metadata", {}) or {})),
            ))
        self.transitions = deque(
            restored_transitions[-self.max_transitions :],
            maxlen=self.max_transitions,
        )
        self.sequence = max(0, int(state.get("sequence", 0) or 0))
        self.transition_chain_sha256 = str(
            state.get("transition_chain_sha256")
            or _stable_digest([
                transition.serializable() for transition in restored_transitions
            ])
        )
        self.input_event_count = max(
            0, int(state.get("input_event_count", 0) or 0)
        )
        self.input_lineage_chain_sha256 = str(
            state.get("input_lineage_chain_sha256") or _stable_digest([])
        )
        self.dropped_transitions = max(
            0, int(state.get("dropped_transitions", 0) or 0)
        )
        self.dropped_pending_events = max(
            0, int(state.get("dropped_pending_events", 0) or 0)
        )
        self.event_counts = Counter({
            str(key): int(value or 0)
            for key, value in dict(state.get("event_counts", {}) or {}).items()
        })


def compare_branch_contexts(
    left: Optional[Dict[str, object]],
    right: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Compare hard execution lineage while reporting softer phase differences."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {
            "known": False,
            "compatible": True,
            "reason": "context_signature_missing",
            "hard_mismatches": [],
            "phase_mismatches": [],
        }
    hard_fields = (
        "active_task",
        "irq_stack",
        "pending_irqs_sha256",
        "enabled_irqs_sha256",
        "stream_cursors_sha256",
        "input_event_count",
        "input_lineage_sha256",
        "pending_events_sha256",
        "dma_phases_sha256",
    )
    incomplete_fields = sorted({
        field_name
        for field_name in hard_fields
        if field_name not in left or field_name not in right
    })
    if (
        left.get("schema") != CausalExecutionContext.SCHEMA
        or right.get("schema") != CausalExecutionContext.SCHEMA
        or incomplete_fields
    ):
        return {
            "known": False,
            "compatible": True,
            "reason": "context_signature_incomplete",
            "hard_mismatches": [],
            "phase_mismatches": [],
            "incomplete_fields": incomplete_fields,
        }
    hard_mismatches = [
        field_name
        for field_name in hard_fields
        if left.get(field_name) != right.get(field_name)
    ]
    phase_mismatches = []
    if left.get("peripheral_phases_sha256") != right.get("peripheral_phases_sha256"):
        phase_mismatches.append("peripheral_phases")
    return {
        "known": True,
        "compatible": not bool(hard_mismatches),
        "reason": (
            "context_compatible" if not hard_mismatches else "context_lineage_mismatch"
        ),
        "hard_mismatches": hard_mismatches,
        "phase_mismatches": phase_mismatches,
        "left_hard_sha256": left.get("hard_compatibility_sha256"),
        "right_hard_sha256": right.get("hard_compatibility_sha256"),
    }
