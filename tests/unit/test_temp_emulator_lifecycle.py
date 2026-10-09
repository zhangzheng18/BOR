#!/usr/bin/env python3
"""Contracts for temporary emulator ownership and cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from collections import Counter
import unittest
from unittest.mock import MagicMock, patch

from lsgemu.temp_emulator_lifecycle import TempEmulatorLifecycleMixin


class _FakeEmulator:
    def __init__(self) -> None:
        self.uc = object()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        self.uc = None


class _FakeRunner(TempEmulatorLifecycleMixin):
    def __init__(self) -> None:
        self.max_snapshots = 3
        self.current_stage_name = "unknown"
        self.temp_emulator_cleanup_stats = Counter()
        self.temp_emulator_lifecycle_by_stage = {}
        self.temp_emulator_stage_start_counts = {}
        self.temp_emulator_stage_high_frequency = set()
        self.temp_emulator_stage_first_created_at = {}
        self.temp_emulator_stage_last_created_at = {}
        self.temp_emulator_quarantine = []
        self.active_temp_emulators = {}
        self.prepared = SimpleNamespace(new_emulator=lambda **_: _FakeEmulator())


class _BookkeepingFailureRunner(_FakeRunner):
    def _track_temp_emulator(self, emulator, **kwargs):
        raise RuntimeError("injected tracking failure")


class _CleanupFailureEmulator(_FakeEmulator):
    def close(self) -> None:
        self.closed += 1
        raise RuntimeError("injected native close failure")


class TempEmulatorLifecycleTests(unittest.TestCase):
    def test_managed_lease_cleans_up_on_exception_and_tracks_bindings(self):
        runner = _FakeRunner()
        mmio = object()
        tracer = object()
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with runner._managed_temp_emulator(stage_name="stage") as lease:
                lease.bind_mmio(mmio)
                lease.bind_tracer(tracer)
                self.assertEqual(1, len(runner.active_temp_emulators))
                raise RuntimeError("stop")

        self.assertEqual({}, runner.active_temp_emulators)
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["created"])
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["disposed"])
        self.assertEqual(1, runner.temp_emulator_lifecycle_by_stage["stage"]["managed_exceptions"])

    def test_stage_change_disposes_unpaired_instance(self):
        runner = _FakeRunner()
        emulator = runner._new_temp_emulator(stage_name="old")
        runner.set_lifecycle_stage("new")

        self.assertEqual(1, emulator.closed)
        self.assertEqual({}, runner.active_temp_emulators)
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["disposed"])
        self.assertEqual(
            1,
            runner.temp_emulator_lifecycle_by_stage["old"]["active_cleanup_stage_change"],
        )

    def test_explicit_dispose_is_not_repeated_by_backstop(self):
        runner = _FakeRunner()
        emulator = runner._new_temp_emulator(stage_name="stage")
        runner._dispose_temp_emulator(emulator, quarantine=False)
        runner._dispose_active_temp_emulators(stage_name="stage", reason="stage_end")

        self.assertEqual(1, emulator.closed)
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["created"])
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["disposed"])
        self.assertNotIn("active_cleanup_stage_end", runner.temp_emulator_lifecycle_by_stage["stage"])

    def test_periodic_collection_runs_after_disposal(self):
        runner = _FakeRunner()
        emulator = runner._new_temp_emulator(stage_name="stage")
        environment = {
            "LSGEMU_TEMP_EMULATOR_GC_INTERVAL": "1",
            "LSGEMU_TEMP_EMULATOR_MALLOC_TRIM_INTERVAL": "0",
        }
        with patch.dict("os.environ", environment), patch(
            "lsgemu.temp_emulator_lifecycle.gc.collect",
            return_value=3,
        ) as collect:
            runner._dispose_temp_emulator(emulator, quarantine=False)

        collect.assert_called_once_with()
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["periodic_gc_runs"])
        self.assertEqual(3, runner.temp_emulator_cleanup_stats["periodic_gc_collected"])

    def test_creation_bookkeeping_failure_does_not_lose_engine(self):
        runner = _BookkeepingFailureRunner()

        with self.assertRaisesRegex(RuntimeError, "injected tracking failure"):
            runner._new_temp_emulator(stage_name="stage")

        self.assertEqual(1, runner.temp_emulator_cleanup_stats["created"])
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["disposed"])
        self.assertEqual({}, runner.active_temp_emulators)

    def test_managed_cleanup_errors_remain_observable_after_context_exit(self):
        runner = _FakeRunner()
        emulator = _CleanupFailureEmulator()
        runner.prepared = SimpleNamespace(new_emulator=lambda **_: emulator)

        with runner._managed_temp_emulator(stage_name="stage"):
            pass

        errors = runner._collect_temp_emulator_cleanup_errors(emulator)
        self.assertTrue(any("emulator_close:RuntimeError" in item for item in errors))
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["disposed"])

    def test_pressure_trim_collects_closed_emulator_cycles_first(self):
        runner = _FakeRunner()
        emulator = runner._new_temp_emulator(stage_name="stage")
        runner._lsgemu_last_trim_rss_bytes = 1
        fake_trim = MagicMock(return_value=1)
        fake_libc = SimpleNamespace(malloc_trim=fake_trim)
        environment = {
            "LSGEMU_TEMP_EMULATOR_GC_INTERVAL": "0",
            "LSGEMU_TEMP_EMULATOR_MALLOC_TRIM_INTERVAL": "0",
            "LSGEMU_TEMP_EMULATOR_RSS_TRIM_THRESHOLD_MIB": "1",
            "LSGEMU_TEMP_EMULATOR_GC_ON_PRESSURE": "1",
        }
        with patch.dict("os.environ", environment), patch(
            "lsgemu.temp_emulator_lifecycle._current_process_rss_bytes",
            side_effect=(2 * 1024 * 1024, 1024 * 1024, 512 * 1024),
        ), patch(
            "lsgemu.temp_emulator_lifecycle.gc.collect",
            return_value=11,
        ) as collect, patch(
            "lsgemu.temp_emulator_lifecycle.ctypes.CDLL",
            return_value=fake_libc,
        ):
            runner._dispose_temp_emulator(emulator, quarantine=False)

        collect.assert_called_once_with()
        fake_trim.assert_called_once_with(0)
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["pressure_gc_runs"])
        self.assertEqual(11, runner.temp_emulator_cleanup_stats["pressure_gc_collected"])
        self.assertEqual(1, runner.temp_emulator_cleanup_stats["malloc_trim_runs"])


if __name__ == "__main__":
    unittest.main()
