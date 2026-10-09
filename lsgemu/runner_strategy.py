#!/usr/bin/env python3
"""Stateless replay strategy helpers for the LSGEmu runner."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

from .instruction_utils import SWITCH_DISPATCH_MNEMONICS


def ordered_replay_exact_occurrence_enabled() -> bool:
    return (
        os.environ.get("LSGEMU_ORDERED_REPLAY_EXACT_OCCURRENCE", "0").strip().lower()
        not in {"", "0", "false", "no"}
    )


def root_provenance_replay_order_enabled() -> bool:
    return (
        os.environ.get("LSGEMU_ROOT_PROVENANCE_REPLAY_ORDER", "0").strip().lower()
        not in {"", "0", "false", "no"}
    )


def scoped_subsequence_match_enabled() -> bool:
    return (
        os.environ.get("LSGEMU_SCOPED_SUBSEQUENCE_MATCH", "0").strip().lower()
        not in {"", "0", "false", "no"}
    )


def root_snapshot_variant_limit() -> int:
    try:
        return int(os.environ.get("LSGEMU_ROOT_SNAPSHOT_VARIANT_LIMIT", "0"))
    except ValueError:
        return 0


def root_provenance_signature_limit() -> int:
    try:
        return int(os.environ.get("LSGEMU_ROOT_PROVENANCE_SIGNATURE_LIMIT", "0"))
    except ValueError:
        return 0


def branch_trace_matches_token(trace_item: Dict[str, object], token: object) -> bool:
    if isinstance(token, tuple):
        return (
            int(trace_item.get("branch")) == int(token[0])
            and int(trace_item.get("occurrence_index", 1)) == int(token[1])
        )
    return int(trace_item.get("branch")) == int(token)


def queue_path_signature(
    choices: Dict[Tuple[int, int], object],
) -> Tuple[Tuple[Tuple[int, int], object], ...]:
    return tuple(sorted(choices.items(), key=lambda item: item[0]))


def is_switch_branch_event(event: object) -> bool:
    return (
        str(getattr(event, "condition", "") or "").upper() in SWITCH_DISPATCH_MNEMONICS
        and bool(getattr(event, "alternatives", []) or [])
    )


def state_equivalence_record(
    control_run_result: Optional[Dict[str, object]],
    result_run_result: Optional[Dict[str, object]],
) -> Dict[str, object]:
    control_fp = dict((control_run_result or {}).get("state_fingerprint_summary", {}) or {})
    result_fp = dict((result_run_result or {}).get("state_fingerprint_summary", {}) or {})
    control_initial_fp = dict(
        (control_run_result or {}).get("initial_state_fingerprint_summary", {}) or {}
    )
    result_initial_fp = dict(
        (result_run_result or {}).get("initial_state_fingerprint_summary", {}) or {}
    )
    if not control_fp and not result_fp and not control_initial_fp and not result_initial_fp:
        return {}
    control_hash = control_fp.get("fingerprint_sha256")
    result_hash = result_fp.get("fingerprint_sha256")
    control_initial_hash = control_initial_fp.get("fingerprint_sha256")
    result_initial_hash = result_initial_fp.get("fingerprint_sha256")
    initial_available = bool(control_initial_hash and result_initial_hash)
    return {
        "paired_replay": bool(control_run_result is not None and result_run_result is not None),
        "control_fingerprint": control_fp,
        "result_fingerprint": result_fp,
        "control_initial_fingerprint": control_initial_fp,
        "result_initial_fingerprint": result_initial_fp,
        "initial_state_fingerprint_available": initial_available,
        "same_initial_fingerprint": bool(
            initial_available and control_initial_hash == result_initial_hash
        ),
        "initial_state_fingerprint_missing": bool(
            (control_initial_fp or result_initial_fp) and not initial_available
        ),
        "same_fingerprint": bool(control_hash and result_hash and control_hash == result_hash),
        "same_ram_dirty_hash": bool(
            control_fp.get("ram_dirty_sha256")
            and result_fp.get("ram_dirty_sha256")
            and control_fp.get("ram_dirty_sha256") == result_fp.get("ram_dirty_sha256")
        ),
        "same_bb": bool(
            control_fp.get("bb_addr")
            and result_fp.get("bb_addr")
            and control_fp.get("bb_addr") == result_fp.get("bb_addr")
        ),
        "mmio_state_delta": int(result_fp.get("mmio_state_entries", 0) or 0) - int(control_fp.get("mmio_state_entries", 0) or 0),
        "constraint_hit_delta": int(result_fp.get("constraint_hit_keys", 0) or 0) - int(control_fp.get("constraint_hit_keys", 0) or 0),
    }


def environment_feasibility_record(
    *,
    strategy: str,
    outcome: str,
    constraint_items: Iterable[Dict[str, object]],
    constraint_feedback: Optional[Dict[str, object]],
    result_eval: Dict[str, object],
    state_equivalence: Optional[Dict[str, object]],
    coverage: Optional[Dict[str, object]],
) -> Dict[str, object]:
    items = [item for item in list(constraint_items or []) if isinstance(item, dict)]
    mmio_items = [item for item in items if str(item.get("type") or "").lower() == "mmio"]
    memory_items = [item for item in items if str(item.get("type") or "").lower() == "memory"]
    feedback = dict(constraint_feedback or {})
    state = dict(state_equivalence or {})
    coverage_dict = dict(coverage or {})
    strategy_text = str(strategy or "")
    entry_derived = strategy_text in {
        "snapshot",
        "entry_fallback",
        "entry_provenance",
        "entry_ordered_provenance",
    } or "provenance" in strategy_text or "snapshot" in strategy_text
    mmio_satisfied = not mmio_items or bool(feedback.get("all_constraint_reads_matched"))
    target_hit = str(outcome or "") == "success" or bool(result_eval.get("success"))
    valid_dynamic_execution = int(coverage_dict.get("result_valid_bbs", 0) or 0) > 0
    paired_replay = bool(state.get("paired_replay"))
    if paired_replay:
        state_available = bool(state.get("initial_state_fingerprint_available"))
        state_compatible = bool(state.get("same_initial_fingerprint"))
    else:
        state_available = bool(state.get("control_fingerprint") or state.get("result_fingerprint"))
        # A non-paired attempt has no control/result starting-state contract.
        # Preserve the old diagnostic fields, but do not use its ending state
        # as proof that a paired replay started from an equivalent state.
        state_compatible = True
    accepted = bool(
        entry_derived
        and mmio_satisfied
        and target_hit
        and valid_dynamic_execution
        and (not paired_replay or (state_available and state_compatible))
    )
    reasons: List[str] = []
    if not entry_derived:
        reasons.append("not_entry_derived_strategy")
    if not mmio_satisfied:
        reasons.append("mmio_constraint_not_observed")
    if not target_hit:
        reasons.append("target_obligation_not_hit")
    if not valid_dynamic_execution:
        reasons.append("no_valid_dynamic_bb_observed")
    if paired_replay and not state_available:
        reasons.append("initial_state_fingerprint_missing")
    elif paired_replay and not state_compatible:
        reasons.append("initial_state_fingerprint_diverged")
    return {
        "schema": "environment_feasibility.v1",
        "accepted": accepted,
        "entry_derived_state": entry_derived,
        "external_nondeterminism": {
            "mmio_constraints": len(mmio_items),
            "memory_constraints": len(memory_items),
            "irq_timing_allowed": "isr" in strategy_text.lower(),
            "mmio_values_are_external_inputs": bool(mmio_items),
        },
        "mmio_constraints_satisfied": mmio_satisfied,
        "target_obligation_hit": target_hit,
        "valid_dynamic_execution": valid_dynamic_execution,
        "state_fingerprint_available": state_available,
        "state_compatible": state_compatible,
        "initial_state_fingerprint_available": bool(
            state.get("initial_state_fingerprint_available")
        ),
        "same_initial_fingerprint": bool(state.get("same_initial_fingerprint")),
        "coverage_credit_policy": "unicorn_executed_valid_bbs_only",
        "reject_reasons": reasons,
    }
