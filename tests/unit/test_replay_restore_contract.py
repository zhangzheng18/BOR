#!/usr/bin/env python3
"""Regression tests for snapshot and MMIO replay restore boundaries."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from lsgemu.mmio_handler.mmio_hook_registry import (
    register_primary_mmio_handler,
    unregister_primary_mmio_handler,
)


class _FakeUC:
    def __init__(self, *, fail_writes: bool = False):
        self.memory = {}
        self.fail_writes = fail_writes
        self.writes = []

    def mem_read(self, address, size):
        address = int(address)
        size = int(size)
        return bytes(self.memory.get(address + offset, 0) for offset in range(size))

    def mem_write(self, address, data):
        if self.fail_writes:
            raise RuntimeError("injected mem_write failure")
        address = int(address)
        payload = bytes(data)
        self.writes.append((address, payload))
        for offset, value in enumerate(payload):
            self.memory[address + offset] = value

    def mem_unmap(self, _address, _size):
        return None


def _replay_fixture(*, fail_writes: bool = False):
    address = 0x40001000
    uc = _FakeUC(fail_writes=fail_writes)
    for offset, value in enumerate(b"\xaa\xbb\xcc\xdd"):
        uc.memory[address + offset] = value
    primary = StatefulMMIOHandler()
    primary.get_or_create_state(address).current_value = 7
    temp = EnhancedMMIOHandler(None)
    temp.mmio_state.update({address: 9, address + 4: 0x55})
    temp.mark_mmio_state_explicit()
    temp.read_occurrence_counts[(0x08000010, address)] = 99
    emulator = SimpleNamespace(
        uc=uc,
        mmio_handler=primary,
        input_occurrence_counts={(0x08000010, address): 4},
        _ensure_memory_mapped=lambda _address, _size: True,
        _is_memory_mapped=lambda _address, _size: True,
    )
    return object.__new__(HistoricalRunner), emulator, temp, primary, address


class ReplayRestoreContractTests(unittest.TestCase):
    def test_poisoned_engine_run_is_hard_gated_before_unicorn(self):
        emulator = object.__new__(IntelligentEmulator)
        emulator.replay_state_poisoned = True
        native_calls = []
        emulator._managed_emu_start = lambda *args, **kwargs: native_calls.append(
            (args, kwargs)
        )

        result = emulator.run(
            entry_point=0x08001001,
            max_instructions=16,
            preserve_cpu_state=True,
        )

        self.assertTrue(result["execution_attempted"])
        self.assertFalse(result["execution_started"])
        self.assertTrue(result["execution_failed"])
        self.assertTrue(result["preflight_failed"])
        self.assertFalse(result["execution_telemetry_complete"])
        self.assertEqual("replay_state_poisoned", result["stop_reason"])
        self.assertEqual([], native_calls)

    def test_restore_replaces_stale_overlay_and_aligns_primary_state(self):
        runner, emulator, temp, primary, address = _replay_fixture()

        self.assertTrue(
            runner._restore_replay_mmio_state(
                emulator,
                temp,
                {address: 1},
            )
        )
        self.assertEqual({address: 1}, temp.mmio_state)
        self.assertEqual({(0x08000010, address): 4}, temp.read_occurrence_counts)
        self.assertEqual(1, primary.mmio_states[address].current_value)
        self.assertEqual(b"\x01\x00\x00\x00", bytes(
            emulator.uc.memory[address + offset] for offset in range(4)
        ))
        self.assertFalse(getattr(emulator, "replay_state_poisoned", False))
        self.assertTrue(emulator.last_replay_mmio_state_restore["success"])

    def test_memory_write_failure_rolls_back_and_poisoned_engine_cannot_continue(self):
        runner, emulator, temp, primary, address = _replay_fixture(fail_writes=True)
        old_overlay = dict(temp.mmio_state)
        old_occurrences = dict(temp.read_occurrence_counts)

        self.assertFalse(
            runner._restore_replay_mmio_state(
                emulator,
                temp,
                {address: 1},
            )
        )
        self.assertEqual(old_overlay, temp.mmio_state)
        self.assertEqual(old_occurrences, temp.read_occurrence_counts)
        self.assertEqual(7, primary.mmio_states[address].current_value)
        self.assertTrue(getattr(emulator, "replay_state_poisoned", False))
        self.assertTrue(emulator.last_replay_mmio_state_restore["rolled_back"])
        self.assertFalse(
            runner._restore_replay_mmio_state(emulator, temp, {address: 2})
        )
        self.assertIn(
            "emulator_replay_state_poisoned",
            emulator.last_replay_mmio_state_restore["errors"],
        )

    def test_unmapped_address_requires_registered_primary_hook(self):
        runner, emulator, temp, _primary, address = _replay_fixture()
        emulator._ensure_memory_mapped = lambda _address, _size: False
        emulator._is_memory_mapped = lambda _address, _size: False

        self.assertFalse(
            runner._restore_replay_mmio_state(emulator, temp, {address: 1})
        )
        self.assertTrue(
            any(
                "mmio_memory_unmapped_without_primary_hook" in item
                for item in emulator.last_replay_mmio_state_restore["errors"]
            )
        )

    def test_registered_primary_hook_may_own_unmapped_address(self):
        runner, emulator, temp, primary, address = _replay_fixture()
        emulator._ensure_memory_mapped = lambda _address, _size: False
        emulator._is_memory_mapped = lambda _address, _size: False
        register_primary_mmio_handler(emulator.uc, primary)
        try:
            self.assertTrue(
                runner._restore_replay_mmio_state(emulator, temp, {address: 1})
            )
            self.assertEqual(
                [f"0x{address:08x}"],
                emulator.last_replay_mmio_state_restore["mapping_skips"],
            )
            self.assertEqual(1, primary.mmio_states[address].current_value)
        finally:
            unregister_primary_mmio_handler(emulator.uc, primary)

    def test_snapshot_manager_exception_is_explicit_and_poisoned(self):
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            uc=object(),
            branch_snapshot_manager=SimpleNamespace(
                restore_snapshot=lambda _uc, _snapshot: (_ for _ in ()).throw(
                    RuntimeError("bad snapshot")
                )
            ),
        )
        snapshot = SimpleNamespace(address=0x08001000, dirty_pages=set())

        self.assertFalse(runner._restore_branch_snapshot(emulator, snapshot))
        self.assertIn(
            "snapshot_restore_exception:RuntimeError",
            emulator.last_snapshot_restore_errors,
        )
        self.assertTrue(emulator.replay_state_poisoned)

    def test_poisoned_snapshot_engine_cannot_attempt_a_second_restore(self):
        runner = object.__new__(HistoricalRunner)
        calls = []
        emulator = SimpleNamespace(
            replay_state_poisoned=True,
            branch_snapshot_manager=SimpleNamespace(
                restore_snapshot=lambda _uc, _snapshot: calls.append(True)
                or True
            ),
            uc=object(),
        )
        snapshot = SimpleNamespace(address=0x08001000, dirty_pages=set())

        self.assertFalse(runner._restore_branch_snapshot(emulator, snapshot))
        self.assertEqual([], calls)
        self.assertIn(
            "emulator_replay_state_poisoned",
            emulator.last_snapshot_restore_errors,
        )

    def test_external_restore_false_is_not_usable(self):
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            uc=object(),
            branch_snapshot_manager=SimpleNamespace(
                restore_snapshot=lambda _uc, _snapshot: True
            ),
            restore_snapshot_external_state=lambda _snapshot: False,
            runtime_written_pages=set(),
        )
        snapshot = SimpleNamespace(address=0x08001000, dirty_pages=set())

        self.assertFalse(runner._restore_branch_snapshot(emulator, snapshot))
        self.assertIn(
            "external_state_restore_returned_false",
            emulator.last_snapshot_restore_errors,
        )

    def test_dirty_page_without_runtime_provenance_is_rejected(self):
        runner = object.__new__(HistoricalRunner)
        emulator = SimpleNamespace(
            uc=object(),
            branch_snapshot_manager=SimpleNamespace(
                restore_snapshot=lambda _uc, _snapshot: True
            ),
        )
        snapshot = SimpleNamespace(
            address=0x08001000,
            dirty_pages={0x20001000},
        )

        self.assertFalse(runner._restore_branch_snapshot(emulator, snapshot))
        self.assertIn(
            "runtime_written_page_provenance_unavailable",
            emulator.last_snapshot_restore_errors,
        )


if __name__ == "__main__":
    unittest.main()
