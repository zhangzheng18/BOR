#!/usr/bin/env python3
"""Replay diagnostics that can be consumed by schedulers.

The historical runner already records rich failure details.  This module keeps
the classification logic small, deterministic, and JSON-safe so reports and
runtime schedulers can share the same root-cause/action vocabulary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_ACTIONS: Dict[str, Tuple[str, int, str]] = {
    "success": ("none", 0, "already_resolved"),
    "mmio_read_not_reached_before_branch": (
        "prefer_root_provenance_replay",
        95,
        "constraint read PC was not reached before the target branch",
    ),
    "mmio_read_not_reached": (
        "retry_closer_prefix_snapshot",
        90,
        "constraint read PC was not reached in replay",
    ),
    "branch_not_reached": (
        "prefer_root_provenance_replay",
        88,
        "target branch was not reached",
    ),
    "mmio_address_reached_but_exact_value_unseen_before_branch": (
        "retry_same_read_pc_with_alt_value",
        84,
        "MMIO address was read before branch but not with the requested value",
    ),
    "mmio_address_reached_but_exact_value_unseen": (
        "retry_same_read_pc_with_alt_value",
        80,
        "MMIO address was read but exact value did not appear",
    ),
    "exact_value_seen_but_branch_outcome_unchanged": (
        "request_llm_alternative",
        78,
        "exact value was supplied but branch outcome did not change",
    ),
    "no_effect_same_outcome": (
        "request_llm_alternative",
        76,
        "replay showed no behavioral delta",
    ),
    "exact_value_seen_but_target_branch_not_reached": (
        "retry_root_provenance_with_state_check",
        74,
        "value was seen but target branch disappeared",
    ),
    "exact_value_seen_but_replay_state_mismatch": (
        "retry_root_provenance_with_state_check",
        82,
        "exact value was supplied but replay state fingerprint differed from control",
    ),
    "exact_value_seen_but_branch_lost_after_constraint": (
        "retry_root_provenance_with_state_check",
        72,
        "constraint changed the prefix enough to lose the branch",
    ),
    "branch_seen_but_desired_successor_not_seen_after_exact_value": (
        "inspect_non_mmio_dependency",
        70,
        "branch was reached but the desired successor was not observed",
    ),
    "exact_value_seen_but_replay_failed_other": (
        "inspect_state_or_transaction_model",
        66,
        "value was seen but replay still failed",
    ),
    "no_constraint_feedback": (
        "collect_runtime_feedback",
        45,
        "attempt did not include MMIO feedback",
    ),
}

LLM_RECOMMENDED_ACTIONS: Dict[str, Tuple[str, int, str]] = {
    "retry_closer_snapshot": (
        "retry_closer_prefix_snapshot",
        91,
        "LLM judged the value plausible but the replay prefix/snapshot was likely too far from the branch",
    ),
    "retry_root_provenance": (
        "prefer_root_provenance_replay",
        90,
        "LLM requested replay from the recorded branch-root provenance",
    ),
    "infer_transaction_sequence": (
        "learn_peripheral_transaction_model",
        86,
        "LLM identified a likely peripheral transaction/status-sequence dependency",
    ),
    "inspect_non_mmio_dependency": (
        "inspect_non_mmio_dependency",
        82,
        "LLM identified a non-MMIO or RAM-backed dependency",
    ),
    "skip_value_change": (
        "inspect_state_or_transaction_model",
        68,
        "LLM judged the candidate value unlikely to be the root cause",
    ),
    "use_value": (
        "request_llm_alternative",
        60,
        "LLM accepted the value; replay failure still needs alternate validation",
    ),
}

LLM_ROOT_CAUSE_ACTIONS: Dict[str, Tuple[str, int, str]] = {
    "wrong_read_pc": (
        "retry_closer_prefix_snapshot",
        92,
        "LLM indicated the constraint was probably bound to the wrong read PC",
    ),
    "state_prefix_mismatch": (
        "retry_root_provenance_with_state_check",
        90,
        "LLM indicated a replay state/prefix mismatch",
    ),
    "needs_transaction_model": (
        "learn_peripheral_transaction_model",
        88,
        "LLM indicated a peripheral transaction model is needed",
    ),
    "not_mmio_dependency": (
        "inspect_non_mmio_dependency",
        84,
        "LLM indicated the branch is not controlled by the candidate MMIO value",
    ),
    "try_alternative_value": (
        "request_llm_alternative",
        78,
        "LLM requested an alternate semantic value",
    ),
    "unique_semantic_value": (
        "retry_root_provenance_with_state_check",
        74,
        "LLM judged the local value unique; replay state should be checked",
    ),
}

FRONTIER_ACTIONS: Dict[str, Tuple[str, int, str]] = {
    "switch_case_frontier": (
        "target_switch_case_expansion",
        92,
        "expand TBB/TBH/LDRPC cases from entry-derived snapshots",
    ),
    "indirect_call_or_return_frontier": (
        "target_dynamic_call_successor",
        88,
        "complete BLX/BX dynamic successor replay from provenance snapshots",
    ),
    "direct_call_continuation_frontier": (
        "target_direct_call_continuation",
        84,
        "execute direct callee or safe continuation from reached callsites",
    ),
    "conditional_branch_frontier": (
        "target_branch_frontier",
        76,
        "replay alternate branch outcomes from reached branch snapshots",
    ),
    "non_replayable_successor_frontier": (
        "inspect_cfg_or_state_model",
        40,
        "frontier successor exists but current replay engine cannot map a choice",
    ),
}

SEMANTIC_ACTIONS: Dict[str, Tuple[str, int, str]] = {
    "interrupt_or_vector_handler": (
        "target_contextual_isr",
        86,
        "inject observed pending/enabled IRQs on reservoir snapshots",
    ),
    "stream_or_protocol_input": (
        "target_stream_input_replay",
        82,
        "mutate discovered stream/ring-buffer state from initialized snapshots",
    ),
    "protocol_callback": (
        "target_stream_input_replay",
        82,
        "drive parser/callback paths through stream-shaped input",
    ),
    "hal_or_peripheral_state": (
        "learn_peripheral_transaction_model",
        74,
        "learn write-control/read-status completion or ack semantics",
    ),
    "rtos_or_task_entry": (
        "target_rtos_thread_entry_replay",
        70,
        "replay task entry with captured create-call arguments and state",
    ),
}


def _hex_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value & 0xFFFFFFFF
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) & 0xFFFFFFFF if text.lower().startswith("0x") else int(text, 0) & 0xFFFFFFFF
    except ValueError:
        return None


def _first_feedback_read_pc(items: Iterable[Dict[str, Any]]) -> Optional[int]:
    for item in items or []:
        if not isinstance(item, dict):
            continue
        constraint = item.get("constraint") or {}
        if isinstance(constraint, dict):
            read_pc = _hex_int(constraint.get("read_pc") or constraint.get("pc"))
            if read_pc is not None:
                return read_pc
        observed_pc = _hex_int(item.get("observed_pc"))
        if observed_pc is not None:
            return observed_pc
    return None


def _compact_dependency_context(record: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """Extract RAM/MMIO dependency context without assuming one runner schema."""
    candidates = []
    for key in (
        "branch_dependencies",
        "dependency_context",
        "provenance_dependencies",
        "non_mmio_dependencies",
    ):
        value = record.get(key)
        if isinstance(value, dict):
            value = value.get("dependencies")
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))

    result: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        dep_type = str(item.get("type") or item.get("source_type") or "")
        address = _hex_int(item.get("address") or item.get("memory_address") or item.get("mmio_address"))
        register = str(item.get("register") or "")
        if address is None and not register:
            continue
        key = (dep_type, address, register)
        if key in seen:
            continue
        seen.add(key)
        compact = {
            "register": register or None,
            "type": dep_type or None,
            "address": f"0x{address:08x}" if address is not None else None,
            "source_pc": (
                f"0x{_hex_int(item.get('source_pc')):08x}"
                if _hex_int(item.get("source_pc")) is not None
                else None
            ),
            "memory_role": item.get("memory_role"),
            "memory_base": (
                f"0x{_hex_int(item.get('memory_base')):08x}"
                if _hex_int(item.get("memory_base")) is not None
                else None
            ),
            "memory_offset": item.get("memory_offset"),
            "composite": bool(item.get("composite", False)),
        }
        result.append({k: v for k, v in compact.items() if v is not None})
        if len(result) >= max(0, limit):
            break
    return result


def classify_branch_mmio_attempt(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one branch-MMIO attempt and attach a scheduler action."""
    result_eval = dict(record.get("result_eval", {}) or {})
    feedback = dict(record.get("constraint_feedback", {}) or {})
    items = [item for item in (feedback.get("items", []) or []) if isinstance(item, dict)]
    outcome = str(record.get("outcome", "") or "")
    failure_reason = str(result_eval.get("failure_reason", "") or "")
    any_address_seen = any(bool(item.get("address_read_seen", False)) for item in items)
    any_exact_value_seen = any(bool(item.get("exact_value_seen", False)) for item in items)
    branch_seen = bool(result_eval.get("branch_seen", False))
    desired_successor_seen = bool(result_eval.get("desired_successor_seen", False))
    state_equivalence = dict(record.get("state_equivalence", {}) or {})
    state_fingerprint_available = bool(
        state_equivalence.get("control_fingerprint") or state_equivalence.get("result_fingerprint")
    )
    same_fingerprint = bool(state_equivalence.get("same_fingerprint", False))
    dependency_context = _compact_dependency_context(record)
    llm_meta_items = [
        item
        for item in (record.get("constraint_inference_metadata", []) or [])
        if isinstance(item, dict)
    ]
    llm_root_causes = []
    llm_recommended_actions = []
    for meta in llm_meta_items:
        latest_root = meta.get("latest_root_cause")
        if latest_root:
            llm_root_causes.append(str(latest_root))
        latest_action = meta.get("latest_recommended_action")
        if latest_action:
            llm_recommended_actions.append(str(latest_action))
        for root_cause, count in (meta.get("root_cause_counts", {}) or {}).items():
            if int(count or 0) > 0:
                llm_root_causes.append(str(root_cause))
        for action_name, count in (meta.get("recommended_action_counts", {}) or {}).items():
            if int(count or 0) > 0:
                llm_recommended_actions.append(str(action_name))

    if outcome == "success":
        root_case = "success"
    elif failure_reason == "branch_not_reached":
        if not any_address_seen:
            root_case = "mmio_read_not_reached_before_branch"
        elif not any_exact_value_seen:
            root_case = "mmio_address_reached_but_exact_value_unseen_before_branch"
        else:
            root_case = "exact_value_seen_but_target_branch_not_reached"
    elif not items:
        root_case = "no_constraint_feedback"
    elif not any_address_seen:
        root_case = "mmio_read_not_reached"
    elif not any_exact_value_seen:
        root_case = "mmio_address_reached_but_exact_value_unseen"
    elif (
        state_fingerprint_available
        and not same_fingerprint
        and any_exact_value_seen
        and outcome in {
            "no_effect_same_outcome",
            "branch_lost_after_constraint",
            "branch_not_reached",
        }
    ):
        root_case = "exact_value_seen_but_replay_state_mismatch"
    elif outcome == "no_effect_same_outcome":
        root_case = "exact_value_seen_but_branch_outcome_unchanged"
    elif outcome == "branch_lost_after_constraint":
        root_case = "exact_value_seen_but_branch_lost_after_constraint"
    elif branch_seen and not desired_successor_seen:
        root_case = "branch_seen_but_desired_successor_not_seen_after_exact_value"
    else:
        root_case = "exact_value_seen_but_replay_failed_other"

    action, priority, reason = ROOT_ACTIONS.get(
        root_case,
        ("inspect_replay_failure", 50, "unclassified branch-MMIO replay failure"),
    )
    llm_action_source = None
    for llm_action in llm_recommended_actions:
        mapped = LLM_RECOMMENDED_ACTIONS.get(str(llm_action))
        if mapped is None:
            continue
        mapped_action, mapped_priority, mapped_reason = mapped
        if int(mapped_priority) >= int(priority):
            action, priority, reason = mapped_action, mapped_priority, mapped_reason
            llm_action_source = f"recommended_action:{llm_action}"
            break
    if llm_action_source is None:
        for llm_root in llm_root_causes:
            mapped = LLM_ROOT_CAUSE_ACTIONS.get(str(llm_root))
            if mapped is None:
                continue
            mapped_action, mapped_priority, mapped_reason = mapped
            if int(mapped_priority) >= int(priority):
                action, priority, reason = mapped_action, mapped_priority, mapped_reason
                llm_action_source = f"root_cause:{llm_root}"
                break
    return {
        "root_case": root_case,
        "action": action,
        "priority": int(priority),
        "reason": reason,
        "llm_action_source": llm_action_source,
        "llm_root_causes": sorted(set(llm_root_causes)),
        "llm_recommended_actions": sorted(set(llm_recommended_actions)),
        "branch_pc": str(record.get("branch_pc") or "") or None,
        "read_pc": (
            f"0x{_first_feedback_read_pc(items):08x}"
            if _first_feedback_read_pc(items) is not None
            else None
        ),
        "strategy": str(record.get("strategy", "") or "") or None,
        "outcome": outcome or None,
        "failure_reason": failure_reason or None,
        "any_address_seen": bool(any_address_seen),
        "any_exact_value_seen": bool(any_exact_value_seen),
        "branch_seen": bool(branch_seen),
        "desired_successor_seen": bool(desired_successor_seen),
        "matched_address_reads": int(feedback.get("matched_address_reads", 0) or 0),
        "matched_constraint_reads": int(feedback.get("matched_constraint_reads", 0) or 0),
        "state_fingerprint_available": state_fingerprint_available,
        "same_state_fingerprint": same_fingerprint if state_fingerprint_available else None,
        "state_equivalence": {
            key: value
            for key, value in state_equivalence.items()
            if key in {
                "same_fingerprint",
                "same_ram_dirty_hash",
                "same_bb",
                "mmio_state_delta",
                "constraint_hit_delta",
            }
        },
        "dependency_context": dependency_context,
    }


def summarize_branch_mmio_root_causes(
    phase_metadata: Dict[str, Any],
    *,
    max_samples: int = 16,
    per_branch_limit: int = 16,
) -> Dict[str, Any]:
    root_case_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    failure_reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    per_branch_cases: Dict[str, Counter[str]] = defaultdict(Counter)
    sample_failures: List[Dict[str, Any]] = []

    for phase_name, phase in (phase_metadata or {}).items():
        if not str(phase_name).startswith("branch_mmio"):
            continue
        if not isinstance(phase, dict):
            continue
        for record in phase.get("attempt_records", []) or []:
            if not isinstance(record, dict):
                continue
            classified = classify_branch_mmio_attempt(record)
            root_case = str(classified["root_case"])
            action = str(classified["action"])
            branch_pc = str(classified.get("branch_pc") or "")
            outcome = str(classified.get("outcome") or "unknown")
            failure_reason = str(classified.get("failure_reason") or "none")
            root_case_counts[root_case] += 1
            outcome_counts[outcome] += 1
            failure_reason_counts[failure_reason] += 1
            action_counts[action] += 1
            if branch_pc:
                per_branch_cases[branch_pc][root_case] += 1
            if root_case != "success" and len(sample_failures) < max(0, max_samples):
                sample = dict(classified)
                sample["phase"] = str(phase_name)
                sample_failures.append(sample)

    if not root_case_counts:
        return {}
    return {
        "root_cases": dict(root_case_counts),
        "attempt_outcomes": dict(outcome_counts),
        "failure_reasons": dict(failure_reason_counts),
        "scheduler_actions": dict(action_counts),
        "per_branch_cases": {
            branch_pc: dict(counter)
            for branch_pc, counter in sorted(
                per_branch_cases.items(),
                key=lambda item: (sum(item[1].values()), item[0]),
                reverse=True,
            )[: max(0, per_branch_limit)]
        },
        "sample_failures": sample_failures,
    }


def _parse_bb_list(items: Iterable[Any], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items or []:
        value = _hex_int(item)
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(f"0x{value:08x}")
        if limit > 0 and len(result) >= limit:
            break
    return result


def build_scheduler_feedback(
    *,
    branch_mmio_root_causes: Optional[Dict[str, Any]] = None,
    uncovered_summary: Optional[Dict[str, Any]] = None,
    frontier_diagnostics: Optional[Dict[str, Any]] = None,
    direct_call_diagnostics: Optional[Dict[str, Any]] = None,
    dispatch_case_status: Optional[Dict[str, Any]] = None,
    max_samples: int = 32,
) -> Dict[str, Any]:
    """Convert diagnostics into compact scheduler recommendations."""
    actions: Dict[str, Dict[str, Any]] = {}

    def add_action(
        action: str,
        *,
        priority: int,
        reason: str,
        count: int = 1,
        source: str,
        targets: Optional[Iterable[Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not action or action == "none":
            return
        record = actions.setdefault(
            action,
            {
                "action": action,
                "priority": int(priority),
                "count": 0,
                "sources": [],
                "reasons": [],
                "target_bbs": [],
                "details": {},
            },
        )
        record["priority"] = max(int(record.get("priority", 0) or 0), int(priority))
        record["count"] = int(record.get("count", 0) or 0) + int(count)
        if source not in record["sources"]:
            record["sources"].append(source)
        if reason and reason not in record["reasons"]:
            record["reasons"].append(reason)
        existing_targets = set(record.get("target_bbs", []) or [])
        for target in _parse_bb_list(targets or [], max_samples):
            if target in existing_targets:
                continue
            record["target_bbs"].append(target)
            existing_targets.add(target)
            if len(record["target_bbs"]) >= max(0, max_samples):
                break
        if details:
            record["details"].update(details)

    root_causes = branch_mmio_root_causes or {}
    sample_failures = [
        sample
        for sample in (root_causes.get("sample_failures", []) or [])
        if isinstance(sample, dict)
    ]
    sample_keys = set()
    for sample in sample_failures:
        sample_action = str(sample.get("action") or "")
        action_tuple = None
        if sample_action:
            action_tuple = (
                sample_action,
                int(sample.get("priority", 50) or 50),
                str(sample.get("reason") or "classified branch-MMIO replay failure"),
            )
        root_case = str(sample.get("root_case") or "")
        if action_tuple is None:
            action_tuple = ROOT_ACTIONS.get(
                root_case,
                ("inspect_replay_failure", 50, "unclassified branch-MMIO root cause"),
            )
        action, priority, reason = action_tuple
        target = sample.get("branch_pc")
        sample_keys.add((root_case, target))
        add_action(
            action,
            priority=priority,
            reason=reason,
            count=1,
            source="branch_mmio_root_causes",
            targets=[target] if target else [],
            details={
                "sample_root_case": root_case,
                "llm_action_source": sample.get("llm_action_source"),
            },
        )

    for root_case, count in (root_causes.get("root_cases", {}) or {}).items():
        sample_count = sum(
            1
            for sample_root_case, _target in sample_keys
            if sample_root_case == str(root_case)
        )
        residual_count = max(0, int(count or 0) - sample_count)
        if residual_count <= 0:
            continue
        action, priority, reason = ROOT_ACTIONS.get(
            str(root_case),
            ("inspect_replay_failure", 50, "unclassified branch-MMIO root cause"),
        )
        add_action(
            action,
            priority=priority,
            reason=reason,
            count=residual_count,
            source="branch_mmio_root_causes",
            targets=[
                sample.get("branch_pc")
                for sample in sample_failures
                if isinstance(sample, dict) and sample.get("root_case") == root_case
            ],
        )

    summary = uncovered_summary or {}
    for item in summary.get("top_frontier_predecessors", []) or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("frontier_category") or "")
        action_tuple = FRONTIER_ACTIONS.get(category)
        if not action_tuple:
            continue
        action, priority, reason = action_tuple
        add_action(
            action,
            priority=priority,
            reason=reason,
            count=int(item.get("uncovered_successors", 0) or 0),
            source="uncovered_frontier",
            targets=item.get("sample_uncovered_successors", []) or [],
            details={
                "frontier_predecessor_sample": item.get("bb"),
                "frontier_mnemonic_sample": item.get("mnemonic"),
            },
        )

    for category, count in (summary.get("uncovered_function_category_counts", {}) or {}).items():
        action_tuple = SEMANTIC_ACTIONS.get(str(category))
        if not action_tuple:
            continue
        action, priority, reason = action_tuple
        examples = [
            example.get("bb")
            for example in (summary.get("uncovered_function_category_examples", {}) or {}).get(category, [])
            if isinstance(example, dict)
        ]
        add_action(
            action,
            priority=priority,
            reason=reason,
            count=int(count or 0),
            source="uncovered_semantic_category",
            targets=examples,
        )

    for diag, source_name in (
        (frontier_diagnostics or {}, "frontier_diagnostics"),
        (direct_call_diagnostics or {}, "direct_call_diagnostics"),
    ):
        for reason, count in (diag.get("reason_counts", {}) or {}).items():
            if not str(reason).startswith("candidate_ready"):
                continue
            add_action(
                "drain_ready_frontier_candidates",
                priority=68,
                reason=str(reason),
                count=int(count or 0),
                source=source_name,
            )

    dispatch_summary = dispatch_case_status or {}
    for root in dispatch_summary.get("roots", []) or []:
        if not isinstance(root, dict):
            continue
        mnemonic = str(root.get("mnemonic") or "").upper()
        pred_bb = root.get("pred_bb")
        successor_rows = [
            row
            for row in (root.get("successors", []) or [])
            if isinstance(row, dict)
        ]
        if not successor_rows:
            continue

        ready_targets = [
            row.get("successor_bb")
            for row in successor_rows
            if row.get("status") == "ready_unattempted"
        ]
        if ready_targets:
            if mnemonic in {"TBB", "TBH"} or "LDR" in mnemonic:
                action, priority, reason = (
                    "target_switch_case_expansion",
                    93,
                    "dispatch case has mapped choice and entry-derived snapshot but has not been attempted",
                )
            else:
                action, priority, reason = (
                    "target_dynamic_call_successor",
                    89,
                    "dynamic call/branch successor has mapped choice and entry-derived snapshot but has not been attempted",
                )
            add_action(
                action,
                priority=priority,
                reason=reason,
                count=len(ready_targets),
                source="dispatch_case_status",
                targets=ready_targets,
                details={"dispatch_root_sample": pred_bb, "dispatch_mnemonic_sample": mnemonic},
            )

        no_snapshot_targets = [
            row.get("successor_bb")
            for row in successor_rows
            if row.get("status") == "no_snapshot"
        ]
        if no_snapshot_targets:
            add_action(
                "prefer_root_provenance_replay",
                priority=87,
                reason="dispatch target is known but no root snapshot variant is available",
                count=len(no_snapshot_targets),
                source="dispatch_case_status",
                targets=[pred_bb] if pred_bb else no_snapshot_targets,
                details={"dispatch_root_sample": pred_bb, "dispatch_mnemonic_sample": mnemonic},
            )

        unmapped_targets = [
            row.get("successor_bb")
            for row in successor_rows
            if row.get("status") == "choice_unmapped"
        ]
        if unmapped_targets:
            add_action(
                "inspect_cfg_or_state_model",
                priority=62,
                reason="dispatch target exists but current replay engine cannot map the case choice",
                count=len(unmapped_targets),
                source="dispatch_case_status",
                targets=unmapped_targets,
                details={"dispatch_root_sample": pred_bb, "dispatch_mnemonic_sample": mnemonic},
            )

        zero_new_targets = [
            row.get("successor_bb")
            for row in successor_rows
            if row.get("status") == "attempted_zero_new"
        ]
        if zero_new_targets:
            add_action(
                "retry_root_provenance_with_state_check",
                priority=64,
                reason="dispatch target was attempted but produced no new coverage; replay state equivalence should be checked",
                count=len(zero_new_targets),
                source="dispatch_case_status",
                targets=zero_new_targets,
                details={"dispatch_root_sample": pred_bb, "dispatch_mnemonic_sample": mnemonic},
            )

    ordered = sorted(
        actions.values(),
        key=lambda item: (
            int(item.get("priority", 0) or 0),
            int(item.get("count", 0) or 0),
            str(item.get("action", "")),
        ),
        reverse=True,
    )
    return {
        "actions": ordered,
        "recommended_next_stages": [item["action"] for item in ordered],
        "reason_counts": {
            item["action"]: int(item.get("count", 0) or 0)
            for item in ordered
        },
        "target_bbs_by_action": {
            item["action"]: item.get("target_bbs", [])
            for item in ordered
            if item.get("target_bbs")
        },
        "has_actionable_feedback": bool(ordered),
    }
