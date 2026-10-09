#!/usr/bin/env python3
"""Tests for bounded long-sequence input checkpointing and range tracking."""

from collections import Counter
from types import SimpleNamespace
import unittest

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator


class SnapshotManagerStub:
    def __init__(self):
        self.capture_order = 0

    def save_snapshot(self, *args, **kwargs):
        snapshot = SimpleNamespace(capture_order=self.capture_order)
        self.capture_order += 1
        return snapshot

    def get_ordered_occurrence_events(self):
        return []


class CausalInputSequenceTests(unittest.TestCase):
    def test_repeated_input_site_keeps_sparse_sequence_checkpoints(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.enable_branch_snapshot = True
        emulator.max_causal_input_snapshots = 64
        emulator.max_causal_input_snapshots_per_site = 2
        emulator.causal_input_sequence_checkpoints = True
        emulator.causal_input_snapshots = []
        emulator.causal_input_snapshot_site_counts = Counter()
        emulator.causal_input_event_count = 0
        emulator.input_occurrence_counts = {}
        emulator.instruction_to_bb = {0x1000: 0x1000}
        emulator.branch_snapshot_manager = SnapshotManagerStub()
        emulator.uc = object()
        emulator.bb_history = []
        emulator._current_mmio_state = lambda: {}

        for _ in range(8):
            emulator._begin_causal_input_read(
                kind="mmio",
                pc=0x1000,
                address=0x40000000,
                size=1,
            )

        self.assertEqual(
            [1, 2, 4, 8],
            [item["occurrence"] for item in emulator.causal_input_snapshots],
        )
        self.assertEqual(
            ["initial_site", "initial_site", "sparse_sequence", "sparse_sequence"],
            [item["checkpoint_kind"] for item in emulator.causal_input_snapshots],
        )

    def test_stream_payload_marks_every_byte_in_recorded_range(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.external_memory_input_addresses = set()
        emulator.stream_input_payload_writes = [
            {"address": "0x20001000", "size": 16}
        ]
        emulator.stream_input_injected_ranges = []

        self.assertTrue(emulator._is_external_input_memory_range(0x20001000, 4))
        self.assertTrue(emulator._is_external_input_memory_range(0x2000100F, 1))
        self.assertFalse(emulator._is_external_input_memory_range(0x20001010, 1))

    def test_completed_input_event_notifies_transient_observer(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.causal_input_events = []
        emulator.max_causal_input_events = 8
        emulator.branch_snapshot_manager = SnapshotManagerStub()
        emulator.external_input_observers = []
        emulator.external_input_observer_stats = {
            "registered": 0,
            "notifications": 0,
            "failures": 0,
        }
        observed = []

        def observer(event):
            observed.append(dict(event))

        self.assertTrue(emulator.add_external_input_observer(observer))
        emulator._finish_causal_input_read(
            {
                "event_id": 1,
                "kind": "mmio",
                "pc": 0x1000,
                "address": 0x40000000,
                "size": 1,
                "occurrence": 3,
                "delivery": "mapped_mmio_preload",
            },
            value=0x7E,
            source="mapped_mmio_preload",
        )
        self.assertEqual(1, len(observed))
        self.assertEqual(3, observed[0]["occurrence"])
        self.assertEqual(0x7E, observed[0]["value"])
        self.assertTrue(emulator.remove_external_input_observer(observer))
        self.assertFalse(emulator.external_input_observers)


if __name__ == "__main__":
    unittest.main()
