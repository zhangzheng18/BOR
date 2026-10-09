#!/usr/bin/env python3
"""
Targeted regression checks for switch-case expansion, call-like frontier
classification, and target-aware prefix replay selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import (
    HistoricalRunner,
    PrefixReplaySnapshot,
    _is_call_frontier_mnemonic,
    _is_dispatch_frontier_mnemonic,
)
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.cached_interleaved_runner import estimate_targeted_frontier_reserve_seconds


class FakeUC:
    def __init__(self):
        self.writes = {}

    def reg_write(self, reg_id, value):
        self.writes[int(reg_id)] = int(value) & 0xFFFFFFFF


def compute_switch_choices(event):
    values = []
    original_index = getattr(event, "original_index", None)
    for index, target in enumerate(getattr(event, "alternatives", [])):
        if target == 0 or index == original_index:
            continue
        values.append(index)
    return values


def choose_best_prefix(entries, ordered_items, target_successor, successor_lookup):
    best = None
    best_suffix = None
    best_score = None
    for index in range(len(ordered_items) - 1, -1, -1):
        prefix_signature = tuple(ordered_items[:index])
        next_key = ordered_items[index][0]
        entry = entries.get((prefix_signature, next_key))
        if entry is None:
            continue
        suffix_items = ordered_items[index:]
        suffix_successor = successor_lookup.get((suffix_items[0][0], suffix_items[0][1]))
        score = (
            1 if suffix_successor == target_successor else 0,
            len(prefix_signature),
            -len(suffix_items),
            int(getattr(entry, "branch_depth", 0)),
        )
        if best_score is None or score > best_score:
            best = entry
            best_suffix = suffix_items
            best_score = score
    return best, best_suffix, best_score


def main() -> int:
    switch_event = SimpleNamespace(
        address=0x08006EBC,
        condition="TBH",
        alternatives=[
            0x08007000,
            0x08007020,
            0x08007020,
            0x08007040,
        ],
        original_index=0,
    )
    switch_choices = compute_switch_choices(switch_event)

    call_checks = {
        "bl_call_frontier": _is_call_frontier_mnemonic("BL"),
        "bl_dispatch_frontier": _is_dispatch_frontier_mnemonic("BL"),
        "bx_dispatch_frontier": _is_dispatch_frontier_mnemonic("BX"),
        "bne_dispatch_frontier": _is_dispatch_frontier_mnemonic("BNE"),
        "mov_dispatch_frontier": _is_dispatch_frontier_mnemonic("MOV"),
    }

    ordered_items = [
        ((0x08001000, 1), False),
        ((0x08002000, 1), 3),
        ((0x08003000, 1), True),
    ]
    shallow_entry = PrefixReplaySnapshot(
        prefix_signature=tuple(ordered_items[:1]),
        next_branch_key=(0x08002000, 1),
        snapshot=SimpleNamespace(address=0x08002000, order=10),
        mmio_state={},
        source_path_signature=tuple(ordered_items[:1]),
        branch_depth=2,
        origin="prefix",
    )
    deep_entry = PrefixReplaySnapshot(
        prefix_signature=tuple(ordered_items[:2]),
        next_branch_key=(0x08003000, 1),
        snapshot=SimpleNamespace(address=0x08003000, order=11),
        mmio_state={},
        source_path_signature=tuple(ordered_items[:2]),
        branch_depth=3,
        origin="prefix",
    )
    entries = {
        (tuple(ordered_items[:1]), (0x08002000, 1)): shallow_entry,
        (tuple(ordered_items[:2]), (0x08003000, 1)): deep_entry,
    }
    successor_lookup = {
        ((0x08002000, 1), 3): 0x08009000,
        ((0x08003000, 1), True): 0x0800A000,
    }
    best_entry, best_suffix, best_score = choose_best_prefix(
        entries,
        ordered_items,
        0x0800A000,
        successor_lookup,
    )

    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.known_main_branch_events = {}
    runner.known_main_branch_event_order = 0
    runner.known_branch_root_snapshots = {}
    runner.dynamic_successors = {0x08003764: {0x080055DA, 0x080055EA}}
    runner.phase_metadata = {
        "frontier_successor_replay": {
            "candidate_debug_records": [
                {
                    "pred_bb": "0x08003764",
                    "mnemonic": "BX",
                    "successor_bb": "0x080055da",
                    "choice_value": 0x080055DA,
                    "new_bbs": 0,
                    "target_bbs": 0,
                    "stop_reason": "no_new_bb_limit",
                }
            ]
        }
    }
    runner.prepared = SimpleNamespace(
        static_bb_set={0x08003764, 0x080055DA, 0x080055EA},
        valid_bb_set={0x08003764, 0x080055DA, 0x080055EA},
        static_successors={0x08003764: set()},
        branch_instruction_by_bb={
            0x08003764: {"address": 0x08003768, "mnemonic": "BX", "operands": "lr"}
        },
        resolve_basic_block=lambda value: (int(value) & ~1) if value else None,
    )
    runner.emulator = SimpleNamespace(
        branch_snapshot_manager=SimpleNamespace(
            get_snapshot=lambda addr: SimpleNamespace(address=addr, order=5, depth=4)
        )
    )
    runner.validate_coverage = lambda covered: set(covered or set())
    runner._interrupt_context_mmio_state = lambda snapshot: {}
    runner._event_source_signature = lambda event, signature, position_lookup=None: tuple(signature or tuple())
    runner._trim_root_snapshot_variants = lambda key: None
    synthetic_added = runner._ensure_dynamic_dispatch_root_events({0x08003764})
    synthetic_event = runner.known_main_branch_events.get((0x08003764, 1))
    synthetic_snapshot_entry = (runner.known_branch_root_snapshots.get((0x08003764, 1)) or {}).get(tuple())
    synthetic_choices = runner._branch_event_choice_values(synthetic_event)
    synthetic_successors = [
        runner._branch_event_choice_successor_bb(synthetic_event, choice)
        for choice in synthetic_choices
    ]
    dispatch_status = runner.dispatch_case_status_summary(coverage={0x08003764}, max_roots=4)

    emulator = IntelligentEmulator.__new__(IntelligentEmulator)
    fake_uc = FakeUC()
    bx_forced = emulator._force_call_dispatch_target(
        fake_uc,
        {"mnemonic": "BX", "operands": "lr", "address": 0x08003768, "size": 2},
        0x080055DA,
    )
    blx_forced = emulator._force_call_dispatch_target(
        fake_uc,
        {"mnemonic": "BLX", "operands": "r3", "address": 0x08003768, "size": 2},
        0x080055EA,
    )
    dispatch_tokens = {
        "bx": emulator._dispatch_condition_for_mnemonic("BX"),
        "blx": emulator._dispatch_condition_for_mnemonic("BLX"),
        "beq": emulator._dispatch_condition_for_mnemonic("BEQ"),
    }
    short_probe_targeted_reserve = estimate_targeted_frontier_reserve_seconds(
        total_wallclock_budget_seconds=480,
        switch_frontier_enabled=True,
        frontier_targeted_enabled=True,
        frontier_cycle_auto=True,
        switch_round_seconds=30,
        switch_rounds=0,
        switch_max_rounds=2,
        frontier_round_seconds=45,
        frontier_rounds=0,
        frontier_max_rounds=3,
        frontier_cycle_max_cycles=6,
        frontier_cycle_tail_reserve_seconds=48,
    )
    gateway_20m_targeted_reserve = estimate_targeted_frontier_reserve_seconds(
        total_wallclock_budget_seconds=1200,
        switch_frontier_enabled=True,
        frontier_targeted_enabled=True,
        frontier_cycle_auto=True,
        switch_round_seconds=30,
        switch_rounds=0,
        switch_max_rounds=2,
        frontier_round_seconds=45,
        frontier_rounds=0,
        frontier_max_rounds=3,
        frontier_cycle_max_cycles=6,
        frontier_cycle_tail_reserve_seconds=20,
    )

    result = {
        "switch_choices": switch_choices,
        "call_checks": call_checks,
        "best_prefix_next_branch": f"0x{best_entry.next_branch_key[0]:08x}" if best_entry else None,
        "best_prefix_suffix_len": len(best_suffix or []),
        "best_prefix_score": list(best_score) if best_score else None,
        "synthetic_dispatch_root_added": synthetic_added,
        "synthetic_dispatch_condition": getattr(synthetic_event, "condition", None),
        "synthetic_dispatch_alternatives": list(getattr(synthetic_event, "alternatives", []) or []),
        "synthetic_dispatch_choices": synthetic_choices,
        "synthetic_dispatch_successors": synthetic_successors,
        "synthetic_dispatch_snapshot": bool(synthetic_snapshot_entry),
        "dispatch_status": dispatch_status,
        "call_dispatch_force": {
            "bx_forced": bx_forced,
            "blx_forced": blx_forced,
            "register_writes": fake_uc.writes,
            "dispatch_tokens": dispatch_tokens,
        },
        "short_probe_targeted_reserve": short_probe_targeted_reserve,
        "gateway_20m_targeted_reserve": gateway_20m_targeted_reserve,
        "checks": {
            "switch_duplicate_target_preserved": switch_choices == [1, 2, 3],
            "direct_bl_not_forced_frontier": not call_checks["bl_call_frontier"] and not call_checks["bl_dispatch_frontier"],
            "bx_dispatch_detected": call_checks["bx_dispatch_frontier"],
            "conditional_still_dispatch": call_checks["bne_dispatch_frontier"],
            "non_branch_not_dispatch": not call_checks["mov_dispatch_frontier"],
            "deeper_target_prefix_preferred": best_entry is deep_entry and len(best_suffix or []) == 1,
            "synthetic_dispatch_root_created": synthetic_added == 1,
            "synthetic_dispatch_event_shape_ok": getattr(synthetic_event, "condition", None) == "BX" and list(getattr(synthetic_event, "alternatives", []) or []) == [0x080055DA, 0x080055EA],
            "synthetic_dispatch_direction_is_unknown": getattr(synthetic_event, "original_direction_known", True) is False and getattr(synthetic_event, "direction_provenance", "") == "dynamic_successor_set",
            "synthetic_dispatch_choices_are_targets": synthetic_choices == [0x080055DA, 0x080055EA],
            "synthetic_dispatch_successors_ok": synthetic_successors == [0x080055DA, 0x080055EA],
            "synthetic_dispatch_snapshot_ok": bool(synthetic_snapshot_entry),
            "dispatch_status_tracks_cases": dispatch_status["root_count"] == 1 and dispatch_status["status_counts"].get("attempted_zero_new") == 1 and dispatch_status["status_counts"].get("ready_unattempted") == 1,
            "call_dispatch_force_ok": bx_forced and blx_forced,
            "call_dispatch_tokens_ok": dispatch_tokens == {"bx": "BX", "blx": "BLX", "beq": "EQ"},
            "short_probe_keeps_targeted_smoke_budget": short_probe_targeted_reserve == 76,
            "gateway_20m_balances_successor_replay_and_targeted": gateway_20m_targeted_reserve == 144,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
