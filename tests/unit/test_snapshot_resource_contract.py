#!/usr/bin/env python3
"""Contracts for snapshot ownership accounting and optional retention caps."""

from __future__ import annotations

from collections import Counter
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lsgemu.analysis.snapshot_memory import SnapshotPageStore, SnapshotStateBlob
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.runner_models import PrefixReplaySnapshot


class _BranchManager:
    def __init__(self, first, second, page_store) -> None:
        self.snapshots = {0x1000: first, 0x2000: second}
        self.snapshot_history = {0x1000: [first]}
        self.max_current_snapshots = 0
        self.max_history_per_address = 0
        self.max_history_total = 0
        self.max_occurrence_events = 200000
        self.page_store = page_store

    def get_statistics(self):
        return {
            "capture": {"current_snapshot_evictions": 0},
            "occurrence_events_truncated": 0,
        }


class _UniqueManager:
    def __init__(self, second) -> None:
        self.snapshots = [second]
        self.max_snapshots = 5
        self.max_lightweight_anchors = 512

    def get_statistics(self):
        return {
            "capture": {"snapshot_fifo_evictions": 0},
            "lightweight_anchors": 1,
        }


def _entry(snapshot, address: int) -> PrefixReplaySnapshot:
    return PrefixReplaySnapshot(
        prefix_signature=tuple(),
        next_branch_key=(address, 1),
        snapshot=snapshot,
    )


class SnapshotResourceContractTests(unittest.TestCase):
    def test_summary_distinguishes_references_from_snapshot_objects(self):
        page_store = SnapshotPageStore(page_size=256)
        shared_memory = page_store.intern_region(b"A" * 512)
        first = SimpleNamespace(
            address=0x1000,
            memory_regions={(0x20000000, 512): shared_memory},
            external_model_state=SnapshotStateBlob.from_mapping(
                {"events": [{"pc": 0x1000}] * 512},
                min_raw_bytes=0,
            ),
        )
        second = SimpleNamespace(
            address=0x2000,
            memory_regions={(0x20000000, 512): shared_memory},
            external_model_state={},
        )
        branch_manager = _BranchManager(first, second, page_store)
        unique_manager = _UniqueManager(second)
        emulator = SimpleNamespace(
            branch_snapshot_manager=branch_manager,
            snapshot_manager=unique_manager,
            snapshot_page_store=page_store,
            causal_input_snapshots=[{"snapshot": first}],
            causal_input_retention_stats=Counter(events_retained=1),
            max_causal_input_events=8192,
            max_causal_input_snapshots=64,
            max_causal_input_snapshots_per_site=2,
        )
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.emulator = emulator
        runner.known_branch_root_snapshots = {
            (0x1000, 1): {tuple(): _entry(first, 0x1000)}
        }
        runner.direct_call_snapshots = {
            0x1000: {tuple(): _entry(first, 0x1000)}
        }
        runner.stream_summary_entry_snapshots = {
            0x2000: {tuple(): _entry(second, 0x2000)}
        }
        runner.reservoir_prefix_snapshots = {
            (tuple(), (0x2000, 1)): _entry(second, 0x2000)
        }
        runner.reservoir_interrupt_contexts = [{"snapshot": second}]
        runner.snapshot_resource_stats = Counter(reservoir_prefix_evictions=2)

        summary = runner._snapshot_resource_summary()

        self.assertEqual("lsgemu.snapshot_resources.v1", summary["schema"])
        self.assertEqual(10, summary["total_snapshot_references"])
        self.assertEqual(2, summary["unique_snapshot_objects"])
        self.assertEqual(8, summary["duplicate_snapshot_references"])
        self.assertEqual(2, summary["objects_referenced_by_multiple_stores"])
        self.assertEqual(
            1,
            summary["references_by_store"]["causal"]["references"],
        )
        self.assertEqual(
            1,
            summary["retained_memory"]["interned_page_storage_objects"],
        )
        self.assertEqual(
            256,
            summary["retained_memory"]["unique_storage_bytes"],
        )
        external_state = summary["retained_memory"]["external_state"]
        self.assertEqual(1, external_state["blob_objects"])
        self.assertEqual(1, external_state["fallback_dict_objects"])
        self.assertGreater(external_state["raw_bytes"], 0)
        self.assertGreater(external_state["stored_bytes"], 0)
        self.assertEqual(0, external_state["materialized_blob_objects"])
        self.assertEqual(
            2,
            summary["retention_events"]["runner"]["reservoir_prefix_evictions"],
        )

    def test_direct_call_variant_cap_is_explicit_and_zero_is_unbounded(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.known_main_branch_events = {}
        runner.snapshot_resource_stats = Counter()
        snapshots = [
            SimpleNamespace(order=index, capture_order=index, depth=index)
            for index in range(3)
        ]
        signatures = (
            tuple(),
            (((0x1000, 1), True),),
            (((0x2000, 1), False),),
        )
        runner.direct_call_snapshots = {
            0x3000: {
                signature: _entry(snapshot, 0x3000)
                for signature, snapshot in zip(signatures, snapshots)
            }
        }

        with patch.dict(
            os.environ,
            {"LSGEMU_DIRECT_CALL_SNAPSHOT_VARIANTS_PER_ADDRESS": "0"},
        ):
            runner._trim_direct_call_snapshot_variants(0x3000)
        self.assertEqual(3, len(runner.direct_call_snapshots[0x3000]))

        with patch.dict(
            os.environ,
            {"LSGEMU_DIRECT_CALL_SNAPSHOT_VARIANTS_PER_ADDRESS": "2"},
        ):
            runner._trim_direct_call_snapshot_variants(0x3000)
        self.assertEqual(2, len(runner.direct_call_snapshots[0x3000]))
        self.assertEqual(
            1,
            runner.snapshot_resource_stats["direct_call_variant_evictions"],
        )


if __name__ == "__main__":
    unittest.main()
