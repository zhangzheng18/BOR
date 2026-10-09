#!/usr/bin/env python3
"""K4: stop-conditioning + no-progress watchdog, and the delivery boundary.

The predicates under test only touch a handful of attributes, so they are
exercised on a lightweight stub rather than a fully built emulator.
"""
from __future__ import annotations

from collections import Counter
import unittest
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

IDLE_THREAD = 0x081360C4
UNHANDLED_EXCEPTION = 0x08004FBA
NMI_HANDLER = 0x080DB04C
PORT_EXIT_FROM_ISR = 0x08005108


class _FakeController:
    def __init__(self, pending: bool):
        self.pending = bool(pending)

    def has_pending_event(self) -> bool:
        return self.pending


class _Predicates:
    """Borrow the emulator predicates so a stub can carry them unbound."""

    IRQ_WAIT_SINK_SYMBOL_TOKENS = IntelligentEmulator.IRQ_WAIT_SINK_SYMBOL_TOKENS
    _symbol_name_for_bb = IntelligentEmulator._symbol_name_for_bb
    _ensure_irq_symbol_table = IntelligentEmulator._ensure_irq_symbol_table
    _is_interruptible_wait_sink = IntelligentEmulator._is_interruptible_wait_sink
    _irq_delivery_has_pending_event = IntelligentEmulator._irq_delivery_has_pending_event
    _irq_delivery_defers_terminal_stop = (
        IntelligentEmulator._irq_delivery_defers_terminal_stop
    )
    _irq_delivery_defers_hot_loop_halt = (
        IntelligentEmulator._irq_delivery_defers_hot_loop_halt
    )
    _irq_boundary_crossed = IntelligentEmulator._irq_boundary_crossed
    _irq_delivery_watchdog_tick = IntelligentEmulator._irq_delivery_watchdog_tick


class _Stub(_Predicates):
    """Minimal host for the IntelligentEmulator predicates."""

    def __init__(self, *, pending: bool, enabled: bool = True):
        self.irq_delivery_enabled = enabled
        self.irq_delivery_controller = _FakeController(pending)
        self.irq_delivery_deferral_stats = Counter()
        self.irq_delivery_boundary = 0
        self._irq_delivery_boundary_seen = 0
        self.symbols_by_addr = {
            IDLE_THREAD: "__idle_thread",
            UNHANDLED_EXCEPTION: "_unhandled_exception",
            NMI_HANDLER: "NMI_Handler",
            PORT_EXIT_FROM_ISR: "__port_exit_from_isr",
        }
        self._irq_symbol_by_addr = None
        self._irq_symbol_starts_cache = None
        self._irq_sink_symbol_cache = {}

    def _load_elf_all_symbols_by_name(self):
        return {}


def _defer(stub, bb):
    return IntelligentEmulator._irq_delivery_defers_terminal_stop(stub, bb)


class StopConditioningTests(unittest.TestCase):
    def test_real_error_handler_still_stops_even_with_a_pending_event(self):
        """K4 anti-example arm: the fault terminal must not be talked out of it."""
        stub = _Stub(pending=True)
        self.assertFalse(_defer(stub, UNHANDLED_EXCEPTION))
        self.assertFalse(_defer(stub, NMI_HANDLER))
        self.assertEqual(stub.irq_delivery_deferral_stats["fault_terminal_sink"], 2)

    def test_idle_sink_is_not_terminal_when_an_event_is_pending(self):
        stub = _Stub(pending=True)
        self.assertTrue(_defer(stub, IDLE_THREAD))
        # The ChibiOS port hand-off zombies are wait sinks too.
        self.assertTrue(_defer(stub, PORT_EXIT_FROM_ISR))

    def test_idle_sink_is_still_terminal_without_an_event(self):
        stub = _Stub(pending=False)
        self.assertFalse(_defer(stub, IDLE_THREAD))
        self.assertEqual(stub.irq_delivery_deferral_stats["wait_sink_no_event"], 1)

    def test_unknown_symbol_sink_stays_terminal(self):
        stub = _Stub(pending=True)
        self.assertFalse(_defer(stub, 0x0800FF00))

    def test_feature_off_keeps_historical_behaviour(self):
        stub = _Stub(pending=True, enabled=False)
        self.assertFalse(_defer(stub, IDLE_THREAD))

    def test_hot_loop_halt_defers_only_while_an_event_is_outstanding(self):
        pending = _Stub(pending=True)
        self.assertTrue(
            IntelligentEmulator._irq_delivery_defers_hot_loop_halt(pending)
        )
        idle = _Stub(pending=False)
        self.assertFalse(
            IntelligentEmulator._irq_delivery_defers_hot_loop_halt(idle)
        )
        off = _Stub(pending=True, enabled=False)
        self.assertFalse(IntelligentEmulator._irq_delivery_defers_hot_loop_halt(off))


class _WatchdogStub(_Stub):
    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.irq_delivery_watchdog_limit = limit
        self.irq_delivery_watchdog_spins = 0
        self.stop_requested_reason = None
        self.uc = self

    def emu_stop(self):
        self.stopped = True


class NoProgressWatchdogTests(unittest.TestCase):
    def test_watchdog_trips_with_an_independent_stop_reason(self):
        stub = _WatchdogStub(limit=3, pending=True)
        stub.stopped = False
        for _ in range(2):
            self.assertFalse(IntelligentEmulator._irq_delivery_watchdog_tick(stub, True))
        self.assertTrue(IntelligentEmulator._irq_delivery_watchdog_tick(stub, True))
        self.assertEqual(stub.stop_requested_reason, "no_progress_watchdog")
        self.assertNotEqual(stub.stop_requested_reason, "fatal_sink_terminal")
        self.assertTrue(stub.stopped)

    def test_progress_resets_the_watchdog(self):
        stub = _WatchdogStub(limit=3, pending=True)
        stub.stopped = False
        IntelligentEmulator._irq_delivery_watchdog_tick(stub, True)
        IntelligentEmulator._irq_delivery_watchdog_tick(stub, False)
        self.assertEqual(stub.irq_delivery_watchdog_spins, 0)
        self.assertIsNone(stub.stop_requested_reason)

    def test_watchdog_is_disabled_when_the_feature_is_off(self):
        stub = _WatchdogStub(limit=1, pending=True, enabled=False)
        stub.stopped = False
        self.assertFalse(IntelligentEmulator._irq_delivery_watchdog_tick(stub, True))
        self.assertIsNone(stub.stop_requested_reason)


class DeliveryBoundaryTests(unittest.TestCase):
    def test_boundary_marker_is_consumed_once(self):
        stub = _Stub(pending=True)
        self.assertFalse(IntelligentEmulator._irq_boundary_crossed(stub))
        stub.irq_delivery_boundary = 1  # interrupt entry
        self.assertTrue(IntelligentEmulator._irq_boundary_crossed(stub))
        self.assertFalse(IntelligentEmulator._irq_boundary_crossed(stub))
        stub.irq_delivery_boundary = 2  # interrupt return
        self.assertTrue(IntelligentEmulator._irq_boundary_crossed(stub))

    def test_no_boundary_when_the_feature_is_off(self):
        stub = _Stub(pending=True, enabled=False)
        stub.irq_delivery_boundary = 7
        self.assertFalse(IntelligentEmulator._irq_boundary_crossed(stub))


if __name__ == "__main__":
    unittest.main()
