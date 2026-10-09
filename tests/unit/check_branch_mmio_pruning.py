#!/usr/bin/env python3
"""
Smoke-test branch-MMIO direction bookkeeping and replay-pruning caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import BranchConstraintCandidate, HistoricalRunner, PrefixReplaySnapshot


def build_runner() -> HistoricalRunner:
    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.known_main_branch_events = {}
    runner.observed_branch_directions = {}
    runner.branch_mmio_already_target_replays = set()
    runner.branch_mmio_unreachable_replays = set()
    return runner


def make_event(address: int, taken: bool, occurrence: int = 1):
    return SimpleNamespace(
        address=address,
        branch_pc=address + 2,
        target=address + 0x10,
        fallthrough=address + 4,
        condition="BNE",
        original_taken=taken,
        order=0,
        depth=0,
        occurrence_index=occurrence,
        alternatives=[],
        original_index=None,
        first_order=0,
        first_depth=0,
        first_occurrence_index=occurrence,
    )


def make_snapshot_entry(signature):
    return PrefixReplaySnapshot(
        prefix_signature=tuple(),
        next_branch_key=(0x08001000, 1),
        snapshot=SimpleNamespace(address=0x08001000, order=7),
        mmio_state={0x40021004: 0x8},
        source_path_signature=tuple(signature),
        branch_depth=2,
        origin="prefix",
    )


def main() -> int:
    runner = build_runner()

    runner.known_main_branch_events[(0x08001000, 1)] = make_event(0x08001000, False, 1)
    runner._remember_single_observed_branch_direction(0x08001000, True)
    merged_observed = runner._observed_branch_directions(0x08001000)

    constraint = BranchConstraintCandidate(
        constraint_type="mmio",
        address=0x40021004,
        value=0x8,
        read_pc=0x08005774,
        constraint_pc=0x0800577A,
        source="llm_mmio",
    ).normalized()
    snapshot_entry = make_snapshot_entry(((((0x08000010, 1)), False),))

    initial_skip = runner._branch_mmio_cached_skip_reason(
        0x0800577C,
        True,
        "snapshot",
        [constraint],
        snapshot_entry=snapshot_entry,
    )
    runner._remember_branch_mmio_already_target(
        0x0800577C,
        True,
        "snapshot",
        snapshot_entry=snapshot_entry,
    )
    already_skip = runner._branch_mmio_cached_skip_reason(
        0x0800577C,
        True,
        "snapshot",
        [constraint],
        snapshot_entry=snapshot_entry,
    )

    runner2 = build_runner()
    runner2._remember_branch_mmio_unreachable_replay(
        0x0800561A,
        False,
        "entry_provenance",
        [constraint],
        snapshot_entry=snapshot_entry,
    )
    unreachable_skip = runner2._branch_mmio_cached_skip_reason(
        0x0800561A,
        False,
        "entry_provenance",
        [constraint],
        snapshot_entry=snapshot_entry,
    )
    different_constraint_skip = runner2._branch_mmio_cached_skip_reason(
        0x0800561A,
        False,
        "entry_provenance",
        [BranchConstraintCandidate(
            constraint_type="mmio",
            address=0x40021004,
            value=0x4,
            read_pc=0x08005774,
            constraint_pc=0x0800577A,
            source="llm_mmio",
        ).normalized()],
        snapshot_entry=snapshot_entry,
    )

    result = {
        "merged_observed": sorted(list(merged_observed)),
        "initial_skip": initial_skip,
        "already_skip": already_skip,
        "unreachable_skip": unreachable_skip,
        "different_constraint_skip": different_constraint_skip,
        "checks": {
            "merged_observed_ok": merged_observed == {False, True},
            "initial_skip_ok": initial_skip is None,
            "already_skip_ok": already_skip == "cached_already_target_without_constraint",
            "unreachable_skip_ok": unreachable_skip == "cached_unreachable_replay",
            "different_constraint_skip_ok": different_constraint_skip is None,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
