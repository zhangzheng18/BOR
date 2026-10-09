#!/usr/bin/env python3
"""Contracts for bounded runtime histories and shared snapshot metadata."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.snapshot_memory import SnapshotMetadataStore
from lsgemu.runner_models import ExecutionTrace
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.reachability_view import PackedReachabilitySet
from types import SimpleNamespace


class ResourceHistoryRetentionTests(unittest.TestCase):
    def test_emulator_access_tails_keep_exact_counts_and_latest_mmio_read(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.mmio_access_history = []
        emulator.mmio_access_history_limit = 1024
        emulator.mmio_access_history_total = 0
        emulator.mmio_access_history_entries_discarded = 0
        emulator.latest_mmio_read_by_address = {}
        emulator.memory_access_history = []
        emulator.memory_access_history_limit = 1024
        emulator.memory_access_history_total = 0
        emulator.memory_access_history_entries_discarded = 0

        for index in range(2050):
            emulator._record_mmio_access_history(
                0x08001000 + index * 2,
                0x40000000,
                True,
                index,
            )
            emulator._record_memory_access_history(
                0x08002000 + index * 2,
                0x20000000 + index,
                True,
                index,
            )

        self.assertEqual(2050, emulator.mmio_access_history_total)
        self.assertEqual(2050, emulator.memory_access_history_total)
        self.assertLessEqual(len(emulator.mmio_access_history), 1025)
        self.assertLessEqual(len(emulator.memory_access_history), 1025)
        self.assertGreater(emulator.mmio_access_history_entries_discarded, 0)
        self.assertGreater(emulator.memory_access_history_entries_discarded, 0)
        self.assertEqual(
            (0x08001000 + 2049 * 2, 2049),
            emulator.latest_mmio_read_by_address[0x40000000],
        )

    def test_execution_trace_retains_exact_edges_after_tail_eviction(self):
        trace = ExecutionTrace(history_limit=512)
        for index in range(1100):
            trace.record_basic_block(0x08000000 + index * 2)

        self.assertEqual(1100, trace.total_executed_bbs)
        self.assertEqual(1100, len(trace.bb_counts))
        self.assertEqual(1099, len(trace.successor_edges))
        self.assertLessEqual(len(trace.executed_bbs), 1024)
        self.assertGreater(trace.history_entries_discarded, 0)

    def test_snapshot_metadata_store_weakly_interns_equal_sets(self):
        store = SnapshotMetadataStore()
        first = store.intern_set({0x20000000, 0x20001000})
        second = store.intern_set({0x20001000, 0x20000000})

        self.assertIs(first, second)
        self.assertEqual({0x20000000, 0x20001000}, first)
        stats = store.get_statistics()
        self.assertEqual(1, stats["live_unique_sets"])
        self.assertEqual(1, stats["set_reuses"])

    def test_legacy_injected_bb_history_seeds_exact_depth(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.bb_history = [0x08001000, 0x08001010, 0x08001020]
        emulator.bb_history_total = 0

        self.assertEqual(3, emulator._bb_history_depth())
        self.assertEqual(3, emulator.bb_history_total)

    def test_reachability_cache_is_bounded_without_changing_results(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set=set(range(8)))
        runner.effective_successor_cache = (0, {
            0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {6}, 6: {7}, 7: set(),
        })
        runner.dynamic_successor_version = 0
        runner.static_reachable_cache_by_successor_version = {}
        runner.static_reachable_cache_node_count = 0
        runner.static_reachable_cache_hits = 0
        runner.static_reachable_cache_misses = 0
        runner.static_reachable_cache_evictions = 0
        runner.static_reachable_cache_oversized_skips = 0

        with patch.dict("os.environ", {"LSGEMU_STATIC_REACHABILITY_CACHE_MAX_NODES": "5"}):
            first = runner._static_reachable_from_cached(
                0,
                runner.effective_successor_cache[1],
                runner.static_reachable_cache_by_successor_version,
            )
            second = runner._static_reachable_from_cached(
                4,
                runner.effective_successor_cache[1],
                runner.static_reachable_cache_by_successor_version,
            )

        self.assertEqual(set(range(8)), first)
        self.assertEqual({4, 5, 6, 7}, second)
        self.assertLessEqual(runner.static_reachable_cache_node_count, 5)
        self.assertLessEqual(
            sum(len(value) for value in runner.static_reachable_cache_by_successor_version.values()),
            5,
        )
        self.assertGreaterEqual(runner.static_reachable_cache_oversized_skips, 1)

    def test_packed_cache_byte_guard_only_skips_recomputation_cache(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set=set(range(4)))
        successors = {0: {1}, 1: {2}, 2: {3}, 3: set()}
        runner.dynamic_successor_version = 0
        runner.effective_successor_cache = (0, successors)
        runner.static_reachable_cache_by_successor_version = {}
        runner.static_reachable_cache_node_count = 0
        runner.static_reachable_cache_bytes = 0
        runner.static_reachable_cache_hits = 0
        runner.static_reachable_cache_misses = 0
        runner.static_reachable_cache_evictions = 0
        runner.static_reachable_cache_oversized_skips = 0

        with patch.dict(
            "os.environ",
            {
                "LSGEMU_STATIC_REACHABILITY_CACHE_MAX_BYTES": "1",
                "LSGEMU_STATIC_REACHABILITY_CACHE_MAX_NODES": "0",
            },
        ):
            result = runner._static_reachable_from_cached(0, successors, {})

        self.assertEqual({0, 1, 2, 3}, set(result))
        self.assertEqual({}, runner.static_reachable_cache_by_successor_version)
        self.assertEqual(1, runner.static_reachable_cache_oversized_skips)

    def test_packed_reachability_cache_matches_transitive_set_operations(self):
        nodes = set(range(12)) | {100}
        successors = {
            0: {1, 2},
            1: {3, 4},
            2: {4, 5},
            3: {1, 6},
            4: {7},
            5: {7, 100},
            6: set(),
            7: {8},
            8: {9},
            9: {7},
            10: {11},
            11: set(),
            100: set(),
        }
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(static_bb_set=nodes)
        runner.dynamic_successor_version = 0
        runner.effective_successor_cache = (0, successors)
        runner.static_reachable_cache_by_successor_version = {}
        runner.static_reachable_cache_node_count = 0
        runner.static_reachable_cache_bytes = 0
        runner.static_reachable_cache_hits = 0
        runner.static_reachable_cache_misses = 0
        runner.static_reachable_cache_evictions = 0
        runner.static_reachable_cache_oversized_skips = 0

        def expected(start, graph=successors):
            pending = [start]
            seen = {start}
            while pending:
                current = pending.pop()
                for successor in graph.get(current, set()):
                    if successor not in seen:
                        seen.add(successor)
                        pending.append(successor)
            return seen

        for start in sorted(nodes):
            reachable = runner._static_reachable_from_cached(
                start,
                successors,
                {},
            )
            self.assertIsInstance(
                runner.static_reachable_cache_by_successor_version[start],
                PackedReachabilitySet,
            )
            expected_set = expected(start)
            self.assertEqual(expected_set, set(reachable))
            self.assertEqual(expected_set & {1, 4, 7, 100}, set(reachable) & {1, 4, 7, 100})
            self.assertEqual(expected_set - {1, 4}, set(reachable) - {1, 4})
            self.assertEqual(len(expected_set), len(reachable))

        self.assertLess(
            runner.static_reachable_cache_bytes,
            sum(len(expected(start)) * 32 for start in nodes),
        )
        self.assertGreaterEqual(runner.static_reachable_cache_misses, len(nodes))

        # A dynamic successor version must invalidate the packed payload and
        # produce the new exact set, even when the address universe is equal.
        runner._invalidate_dynamic_successor_caches()
        new_successors = {key: set(value) for key, value in successors.items()}
        new_successors[10] = {0}
        runner.effective_successor_cache = (1, new_successors)
        updated = runner._static_reachable_from_cached(10, new_successors, {})
        self.assertEqual(expected(10, new_successors), set(updated))

    def test_emulator_bb_tail_keeps_exact_depth_across_eviction(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.bb_history = []
        emulator.bb_history_limit = 1024
        emulator.bb_history_total = 0
        emulator.bb_history_entries_discarded = 0
        emulator.runtime_successor_edges = set()
        emulator.bb_visit_counts = None

        for index in range(2050):
            emulator._append_bb_history(0x08000000 + index * 2)

        self.assertEqual(2050, emulator._bb_history_depth())
        self.assertEqual(2050, sum(emulator.bb_visit_counts.values()))
        self.assertEqual(2049, len(emulator.runtime_successor_edges))
        self.assertLessEqual(len(emulator.bb_history), 1025)
        self.assertGreater(emulator.bb_history_entries_discarded, 0)


if __name__ == "__main__":
    unittest.main()
