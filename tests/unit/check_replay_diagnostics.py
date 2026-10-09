#!/usr/bin/env python3
"""Smoke-test replay diagnostics scheduler feedback."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.replay_diagnostics import (
    build_scheduler_feedback,
    classify_branch_mmio_attempt,
    summarize_branch_mmio_root_causes,
)


class FakePrepared:
    def __init__(self):
        self.valid_bb_set = {0x08002020, 0x08002040, 0x08003000}
        self.static_bb_set = set(self.valid_bb_set)


class FakeRunner:
    def __init__(self, feedback):
        self.scheduler_feedback = feedback
        self.global_coverage = {0x08002040}
        self.prepared = FakePrepared()

    def _scheduler_feedback_summary(self):
        return self.scheduler_feedback

    @staticmethod
    def validate_coverage(values):
        result = set()
        for item in values or []:
            if isinstance(item, str):
                result.add(int(item, 16) & 0xFFFFFFFF)
            else:
                result.add(int(item) & 0xFFFFFFFF)
        return result

    def frontier_target_bb_list(self, **kwargs):
        target_filter = kwargs.get("target_filter") or set()
        return sorted(target_filter)


def main() -> int:
    attempt = {
        "branch_pc": "0x08001234",
        "outcome": "no_effect_same_outcome",
        "result_eval": {
            "branch_seen": True,
            "desired_successor_seen": False,
            "failure_reason": "wrong_direction",
        },
        "constraint_feedback": {
            "matched_address_reads": 1,
            "matched_constraint_reads": 1,
            "items": [
                {
                    "address_read_seen": True,
                    "exact_value_seen": True,
                    "constraint": {
                        "read_pc": "0x08001200",
                        "address": "0x40021004",
                        "value": "0x00000008",
                    },
                }
            ],
        },
    }
    transaction_attempt = {
        "branch_pc": "0x08005678",
        "outcome": "no_effect_same_outcome",
        "result_eval": {
            "branch_seen": True,
            "desired_successor_seen": False,
            "failure_reason": "wrong_direction",
        },
        "constraint_feedback": {
            "matched_address_reads": 1,
            "matched_constraint_reads": 1,
            "items": [
                {
                    "address_read_seen": True,
                    "exact_value_seen": True,
                    "constraint": {
                        "read_pc": "0x08005660",
                        "address": "0x40002004",
                        "value": "0x00000002",
                    },
                }
            ],
        },
        "constraint_inference_metadata": [
            {
                "inference_method": "llm",
                "latest_root_cause": "needs_transaction_model",
                "latest_recommended_action": "infer_transaction_sequence",
                "root_cause_counts": {"needs_transaction_model": 1},
                "recommended_action_counts": {"infer_transaction_sequence": 1},
            }
        ],
        "branch_dependencies": {
            "dependencies": [
                {
                    "register": "r4",
                    "type": "memory",
                    "address": 0x20001018,
                    "source_pc": 0x08005500,
                    "memory_role": "stack_frame",
                    "memory_base": 0x20001000,
                    "memory_offset": 0x18,
                    "composite": True,
                }
            ]
        },
    }
    state_mismatch_attempt = {
        "branch_pc": "0x08008888",
        "outcome": "no_effect_same_outcome",
        "result_eval": {
            "branch_seen": True,
            "desired_successor_seen": False,
            "failure_reason": "wrong_direction",
        },
        "constraint_feedback": {
            "matched_address_reads": 1,
            "matched_constraint_reads": 1,
            "items": [
                {
                    "address_read_seen": True,
                    "exact_value_seen": True,
                    "constraint": {
                        "read_pc": "0x08008870",
                        "address": "0x40012000",
                        "value": "0x00000001",
                    },
                }
            ],
        },
        "state_equivalence": {
            "control_fingerprint": {"fingerprint_sha256": "aaa", "bb_addr": "0x08008880"},
            "result_fingerprint": {"fingerprint_sha256": "bbb", "bb_addr": "0x08008880"},
            "same_fingerprint": False,
            "same_ram_dirty_hash": False,
            "same_bb": True,
            "mmio_state_delta": 1,
            "constraint_hit_delta": 1,
        },
    }
    classified = classify_branch_mmio_attempt(attempt)
    transaction_classified = classify_branch_mmio_attempt(transaction_attempt)
    state_mismatch_classified = classify_branch_mmio_attempt(state_mismatch_attempt)
    phase_metadata = {"branch_mmio_round_1": {"attempt_records": [attempt, transaction_attempt, state_mismatch_attempt]}}
    root_summary = summarize_branch_mmio_root_causes(phase_metadata)
    uncovered_summary = {
        "top_frontier_predecessors": [
            {
                "frontier_category": "switch_case_frontier",
                "uncovered_successors": 3,
                "bb": "0x08002000",
                "mnemonic": "TBB",
                "sample_uncovered_successors": [
                    "0x08002020",
                    "0x08002040",
                ],
            }
        ],
        "uncovered_function_category_counts": {
            "interrupt_or_vector_handler": 2,
            "hal_or_peripheral_state": 1,
        },
        "uncovered_function_category_examples": {
            "interrupt_or_vector_handler": [{"bb": "0x08003000"}],
            "hal_or_peripheral_state": [{"bb": "0x08004000"}],
        },
    }
    feedback = build_scheduler_feedback(
        branch_mmio_root_causes=root_summary,
        uncovered_summary=uncovered_summary,
        dispatch_case_status={
            "root_count": 2,
            "status_counts": {
                "ready_unattempted": 2,
                "choice_unmapped": 1,
                "no_snapshot": 1,
                "attempted_zero_new": 1,
            },
            "roots": [
                {
                    "pred_bb": "0x08005000",
                    "mnemonic": "TBB",
                    "known_snapshot_variants": 1,
                    "successors": [
                        {"successor_bb": "0x08005020", "status": "ready_unattempted"},
                        {"successor_bb": "0x08005040", "status": "choice_unmapped"},
                        {"successor_bb": "0x08005060", "status": "attempted_zero_new"},
                    ],
                },
                {
                    "pred_bb": "0x08006000",
                    "mnemonic": "BLX",
                    "known_snapshot_variants": 0,
                    "successors": [
                        {"successor_bb": "0x08006020", "status": "ready_unattempted"},
                        {"successor_bb": "0x08006040", "status": "no_snapshot"},
                    ],
                },
            ],
        },
    )
    from lsgemu.historical_runner import HistoricalRunner

    fake_runner = FakeRunner(feedback)
    feedback_targets = HistoricalRunner.scheduler_feedback_target_bb_list(
        fake_runner,
        actions=["target_switch_case_expansion"],
        max_targets=8,
    )
    checks = {
        "classified_action_ok": classified["action"] == "request_llm_alternative",
        "root_summary_action_ok": root_summary["scheduler_actions"]["request_llm_alternative"] == 1,
        "feedback_has_switch_ok": "target_switch_case_expansion" in feedback["recommended_next_stages"],
        "feedback_has_isr_ok": "target_contextual_isr" in feedback["recommended_next_stages"],
        "feedback_targets_ok": feedback["target_bbs_by_action"]["target_switch_case_expansion"][0] == "0x08002020",
        "dispatch_ready_switch_target_ok": "0x08005020" in feedback["target_bbs_by_action"]["target_switch_case_expansion"],
        "dispatch_ready_call_target_ok": "0x08006020" in feedback["target_bbs_by_action"]["target_dynamic_call_successor"],
        "dispatch_no_snapshot_root_ok": "0x08006000" in feedback["target_bbs_by_action"]["prefer_root_provenance_replay"],
        "dispatch_unmapped_action_ok": "inspect_cfg_or_state_model" in feedback["recommended_next_stages"],
        "dispatch_zero_new_action_ok": "retry_root_provenance_with_state_check" in feedback["recommended_next_stages"],
        "runner_feedback_targets_ok": feedback_targets == [0x08002020],
        "llm_transaction_action_ok": transaction_classified["action"] == "learn_peripheral_transaction_model",
        "llm_action_source_ok": transaction_classified["llm_action_source"] == "recommended_action:infer_transaction_sequence",
        "dependency_context_ok": transaction_classified["dependency_context"][0]["memory_role"] == "stack_frame",
        "state_mismatch_root_case_ok": state_mismatch_classified["root_case"] == "exact_value_seen_but_replay_state_mismatch",
        "state_mismatch_action_ok": state_mismatch_classified["action"] == "retry_root_provenance_with_state_check",
    }
    print(json.dumps({
        "classified": classified,
        "transaction_classified": transaction_classified,
        "state_mismatch_classified": state_mismatch_classified,
        "root_summary": root_summary,
        "feedback": feedback,
        "checks": checks,
    }, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
