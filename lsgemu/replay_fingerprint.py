#!/usr/bin/env python3
"""Stable replay-state fingerprints for scheduler pruning.

The goal is to detect "same BB, different state" and "same state, different
snapshot origin" cases without forcing the scheduler to keep raw emulator
state objects in memory.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from unicorn.arm_const import (
        UC_ARM_REG_CPSR,
        UC_ARM_REG_LR,
        UC_ARM_REG_PC,
        UC_ARM_REG_R0,
        UC_ARM_REG_R1,
        UC_ARM_REG_R2,
        UC_ARM_REG_R3,
        UC_ARM_REG_R4,
        UC_ARM_REG_R5,
        UC_ARM_REG_R6,
        UC_ARM_REG_R7,
        UC_ARM_REG_R8,
        UC_ARM_REG_R9,
        UC_ARM_REG_R10,
        UC_ARM_REG_R11,
        UC_ARM_REG_R12,
        UC_ARM_REG_SP,
    )
except Exception:  # pragma: no cover - imported in non-Unicorn unit tests too
    UC_ARM_REG_CPSR = UC_ARM_REG_LR = UC_ARM_REG_PC = UC_ARM_REG_SP = None
    UC_ARM_REG_R0 = UC_ARM_REG_R1 = UC_ARM_REG_R2 = UC_ARM_REG_R3 = None
    UC_ARM_REG_R4 = UC_ARM_REG_R5 = UC_ARM_REG_R6 = UC_ARM_REG_R7 = None
    UC_ARM_REG_R8 = UC_ARM_REG_R9 = UC_ARM_REG_R10 = UC_ARM_REG_R11 = None
    UC_ARM_REG_R12 = None


DEFAULT_FP_REG_IDS = (
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "r7",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "sp",
    "lr",
    "pc",
    "cpsr",
)

_ARM_REG_NAME_TO_CONST = {
    "r0": UC_ARM_REG_R0,
    "r1": UC_ARM_REG_R1,
    "r2": UC_ARM_REG_R2,
    "r3": UC_ARM_REG_R3,
    "r4": UC_ARM_REG_R4,
    "r5": UC_ARM_REG_R5,
    "r6": UC_ARM_REG_R6,
    "r7": UC_ARM_REG_R7,
    "r8": UC_ARM_REG_R8,
    "r9": UC_ARM_REG_R9,
    "r10": UC_ARM_REG_R10,
    "r11": UC_ARM_REG_R11,
    "r12": UC_ARM_REG_R12,
    "sp": UC_ARM_REG_SP,
    "lr": UC_ARM_REG_LR,
    "pc": UC_ARM_REG_PC,
    "cpsr": UC_ARM_REG_CPSR,
}


def _hex(value: Any) -> str:
    if isinstance(value, int):
        return f"0x{value & 0xFFFFFFFF:08x}"
    if value is None:
        return "null"
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _collect_reg_state(uc, reg_ids: Sequence[Any]) -> Dict[str, int]:
    regs: Dict[str, int] = {}
    for reg_id in reg_ids:
        try:
            if isinstance(reg_id, str):
                reg_name = reg_id.lower()
                resolved = _ARM_REG_NAME_TO_CONST.get(reg_name)
                if resolved is None or not hasattr(uc, "reg_read"):
                    regs[reg_name] = 0
                else:
                    regs[reg_name] = int(uc.reg_read(resolved)) & 0xFFFFFFFF
            else:
                regs[str(reg_id)] = int(uc.reg_read(reg_id)) & 0xFFFFFFFF
        except Exception:
            regs[str(reg_id)] = 0
    return regs


def _coerce_mmio_current_value(state: Any) -> Optional[int]:
    """Read one compact current value from either MMIO state representation."""
    if isinstance(state, dict):
        value = state.get("current_value", state.get("value"))
    else:
        value = getattr(state, "current_value", None)
        if value is None:
            value = getattr(state, "value", None)
    if value is None:
        return None
    try:
        return int(value) & 0xFFFFFFFF
    except (TypeError, ValueError, OverflowError):
        return None


def _collect_mmio_state(mmio_handler: Any) -> Dict[str, str]:
    """Normalize EnhancedMMIOHandler and StatefulMMIOHandler state maps.

    The two handlers deliberately remain separate at runtime.  Fingerprinting
    must nevertheless observe the same logical register state; reading only
    ``mmio_state`` silently produced an empty summary for the stateful handler.
    """
    values: Dict[int, int] = {}
    try:
        for address, value in dict(getattr(mmio_handler, "mmio_state", {}) or {}).items():
            try:
                values[int(address) & 0xFFFFFFFF] = int(value) & 0xFFFFFFFF
            except (TypeError, ValueError, OverflowError):
                continue
    except Exception:
        pass
    try:
        for address, state in dict(getattr(mmio_handler, "mmio_states", {}) or {}).items():
            current = _coerce_mmio_current_value(state)
            if current is None:
                continue
            values[int(address) & 0xFFFFFFFF] = current
    except Exception:
        pass
    return {
        f"0x{address:08x}": f"0x{value:08x}"
        for address, value in sorted(values.items())
    }


def _collect_peripheral_state(mmio_handler: Any) -> Dict[str, Any]:
    """Collect bounded external-input and transaction state for replay checks."""
    state: Dict[str, Any] = {}
    try:
        pending = getattr(mmio_handler, "_peripheral_input_pending", {})
        if isinstance(pending, dict):
            state["peripheral_input_pending"] = {
                str(key): int(value) & 0xFF
                for key, value in sorted(pending.items(), key=lambda item: str(item[0]))
            }
    except Exception:
        pass
    try:
        phase = getattr(mmio_handler, "current_phase", None)
        phase_value = getattr(phase, "value", phase)
        if phase_value is not None:
            state["current_phase"] = str(phase_value)
    except Exception:
        pass
    try:
        registry = getattr(mmio_handler, "semantic_profiles", None)
        transactions = getattr(registry, "transactions", None)
        if isinstance(transactions, dict):
            compact_transactions = {}
            for key, transaction in sorted(transactions.items(), key=lambda item: str(item[0])):
                summary = getattr(transaction, "summary", None)
                if callable(summary):
                    try:
                        compact_transactions[str(key)] = dict(summary() or {})
                        continue
                    except Exception:
                        pass
                if isinstance(transaction, dict):
                    compact_transactions[str(key)] = {
                        str(field): transaction[field]
                        for field in ("phase", "sequence", "completion_count")
                        if field in transaction
                    }
            if compact_transactions:
                state["transaction_phases"] = compact_transactions
    except Exception:
        pass
    return state


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Convert bounded replay metadata to deterministic JSON-compatible data."""
    if depth > 3:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, set):
        # Sets occur in candidate metadata and constraint-hit collections.
        # Sorting their normalized representation keeps fingerprints stable
        # across equivalent runs with different hash iteration order.
        normalized = [_jsonable(item, depth + 1) for item in value]
        return sorted(normalized, key=_stable_json)
    return str(value)


def _bounded_mapping_summary(mapping: Any, *, limit: int = 32) -> Dict[str, Any]:
    """Summarize a candidate map without retaining an unbounded replay trace."""
    if not isinstance(mapping, dict):
        return {"count": 0, "sha256": None, "sample": []}
    entries = [
        [str(key), _jsonable(value)]
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    ]
    encoded = _stable_json(entries).encode("utf-8")
    return {
        "count": len(entries),
        "sha256": hashlib.sha256(encoded).hexdigest() if entries else None,
        "sample": entries[: max(0, int(limit))],
    }


def _bounded_sequence_summary(sequence: Any, *, limit: int = 32) -> Dict[str, Any]:
    if not isinstance(sequence, (list, tuple)):
        return {"count": 0, "sha256": None, "sample": []}
    values = [_jsonable(item) for item in sequence]
    encoded = _stable_json(values).encode("utf-8")
    return {
        "count": len(values),
        "sha256": hashlib.sha256(encoded).hexdigest() if values else None,
        "sample": values[: max(0, int(limit))],
    }


def _collect_candidate_configuration(
    mmio_handler: Any,
    emulator: Any,
) -> Dict[str, Any]:
    """Capture replay controls separately from restored device state.

    A candidate value, forced branch, or memory overlay is an input to a replay,
    not part of the state from which a paired control and candidate execution
    started.  Keeping this metadata in a separate bounded section lets reports
    audit the hypothesis without making the initial-state equivalence test
    reject the expected candidate difference.
    """
    configuration: Dict[str, Any] = {}
    if mmio_handler is not None:
        for field_name in (
            "runtime_constraints",
            "runtime_pc_constraints",
            "runtime_occurrence_constraints",
        ):
            try:
                summary = _bounded_mapping_summary(
                    getattr(mmio_handler, field_name, {}) or {}
                )
            except Exception:
                summary = {"count": 0, "sha256": None, "sample": []}
            if summary["count"]:
                configuration[field_name] = summary

    if emulator is not None:
        for field_name in (
            "memory_occurrence_constraints",
            "memory_pc_constraints",
        ):
            try:
                summary = _bounded_mapping_summary(
                    getattr(emulator, field_name, {}) or {}
                )
            except Exception:
                summary = {"count": 0, "sha256": None, "sample": []}
            if summary["count"]:
                configuration[field_name] = summary
        try:
            addresses = sorted(
                int(address) & 0xFFFFFFFF
                for address in (
                    getattr(emulator, "external_memory_input_addresses", set())
                    or set()
                )
            )
            if addresses:
                configuration["external_memory_input_addresses"] = {
                    "count": len(addresses),
                    "sha256": hashlib.sha256(
                        _stable_json(addresses).encode("utf-8")
                    ).hexdigest(),
                    "sample": [f"0x{address:08x}" for address in addresses[:32]],
                }
        except Exception:
            pass
        try:
            choices = dict(getattr(emulator, "forced_branch_choices", {}) or {})
            summary = _bounded_mapping_summary(choices)
            if summary["count"]:
                configuration["forced_branch_choices"] = summary
        except Exception:
            pass
        try:
            summary = _bounded_sequence_summary(
                getattr(emulator, "forced_branch_sequence", []) or []
            )
            if summary["count"]:
                configuration["forced_branch_sequence"] = summary
        except Exception:
            pass
    return configuration


def build_replay_state_fingerprint(
    *,
    uc,
    bb_addr: int,
    mmio_handler,
    emulator=None,
    snapshot_entry=None,
    reg_values: Optional[Dict[str, int]] = None,
    reg_ids: Sequence[Any] = DEFAULT_FP_REG_IDS,
    constraint_hit_keys: Optional[Iterable[Any]] = None,
    call_stack: Optional[Iterable[Any]] = None,
    max_dirty_pages: int = 8,
    ram_sample_bytes: int = 256,
    include_ram: bool = True,
    include_candidate_configuration: bool = False,
    include_observed_constraint_hits: bool = True,
) -> Dict[str, Any]:
    """Build a compact replay fingerprint.

    The fingerprint intentionally summarizes only a few dirty SRAM pages and
    already-observed state to keep replay overhead bounded.
    """
    if reg_values is None:
        reg_values = _collect_reg_state(uc, reg_ids)
    else:
        reg_values = {str(key): int(value) & 0xFFFFFFFF for key, value in reg_values.items()}

    mmio_state = _collect_mmio_state(mmio_handler)
    peripheral_state = _collect_peripheral_state(mmio_handler)

    runtime_written_pages = set()
    try:
        runtime_written_pages = {
            int(page) & ~0xFFF
            for page in (getattr(emulator, "runtime_written_pages", set()) or set())
        }
    except Exception:
        runtime_written_pages = set()

    sram_pages: List[Dict[str, Any]] = []
    dirty_hash_parts: List[bytes] = []
    if include_ram and runtime_written_pages:
        for page in sorted(runtime_written_pages)[: max(0, int(max_dirty_pages))]:
            try:
                data = uc.mem_read(int(page), max(4, int(ram_sample_bytes)))
            except Exception:
                data = b""
            sram_pages.append({
                "page": f"0x{int(page) & 0xFFFFFFFF:08x}",
                "bytes": len(data),
                "sha256": hashlib.sha256(bytes(data)).hexdigest() if data else None,
            })
            dirty_hash_parts.append(f"page:{int(page):08x}:{len(data)}".encode("ascii"))
            if data:
                dirty_hash_parts.append(bytes(data))

    stream_state = {}
    try:
        stream_state = dict(getattr(emulator, "stream_input_state", {}) or {})
    except Exception:
        stream_state = {}

    causal_context = {}
    try:
        context = getattr(emulator, "causal_context", None)
        collector = getattr(context, "fingerprint_components", None)
        if callable(collector):
            causal_context = dict(collector() or {})
    except Exception:
        causal_context = {}

    call_stack_state = []
    for frame in list(call_stack or getattr(getattr(emulator, "register_tracer", None), "call_stack", []) or []):
        if isinstance(frame, dict):
            call_stack_state.append({
                "call_pc": _hex(frame.get("call_pc")),
                "return_pc": _hex(frame.get("return_pc")),
                "depth": int(frame.get("depth", 0) or 0),
            })
        else:
            call_stack_state.append(str(frame))

    constraint_keys = []
    if include_observed_constraint_hits:
        for item in constraint_hit_keys or []:
            if isinstance(item, (tuple, list)):
                constraint_keys.append(tuple(_hex(part) for part in item))
            else:
                constraint_keys.append(_hex(item))

    snapshot_identity = None
    if snapshot_entry is not None:
        snapshot_identity = {
            "origin": str(getattr(snapshot_entry, "origin", "unknown") or "unknown"),
            "prefix_signature_len": len(tuple(getattr(snapshot_entry, "prefix_signature", tuple()) or tuple())),
            "source_path_signature_len": len(tuple(getattr(snapshot_entry, "source_path_signature", tuple()) or tuple())),
            "branch_depth": int(getattr(snapshot_entry, "branch_depth", 0) or 0),
            "next_branch_key": [
                int(getattr(snapshot_entry, "next_branch_key", (0, 0))[0]) if getattr(snapshot_entry, "next_branch_key", None) else 0,
                int(getattr(snapshot_entry, "next_branch_key", (0, 0))[1]) if getattr(snapshot_entry, "next_branch_key", None) else 0,
            ],
        }

    candidate_configuration = _collect_candidate_configuration(
        mmio_handler,
        emulator,
    ) if include_candidate_configuration else {}
    components = {
        "bb_addr": f"0x{int(bb_addr) & 0xFFFFFFFF:08x}",
        "regs": reg_values,
        "mmio_state": mmio_state,
        "peripheral_state": peripheral_state,
        "dirty_pages": sram_pages,
        "stream_state": stream_state,
        "causal_context": causal_context,
        "call_stack": call_stack_state,
        "constraint_hits": constraint_keys,
        "snapshot": snapshot_identity,
        "candidate_configuration": candidate_configuration,
        "fingerprint_scope": (
            "state_and_replay_configuration"
            if include_candidate_configuration
            else "restored_state"
        ),
    }
    digest = hashlib.sha256(_stable_json(components).encode("utf-8")).hexdigest()
    if dirty_hash_parts:
        components["ram_dirty_sha256"] = _hash_bytes(dirty_hash_parts)
    else:
        components["ram_dirty_sha256"] = None
    components["fingerprint_sha256"] = digest
    return components
