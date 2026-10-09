#!/usr/bin/env python3
"""r36 K1 → r40 P2/C8: main-flow IRQ/SVC delivery install gate.

Properties under test, all on stubs (no Unicorn engine needed):

1. **Default off is a no-op.**  ``baseline_irq_delivery_enabled`` defaults to
   False, and nothing installs unless *both* env flags are on.
2. **Baseline phase stays clean.**  ``run_baseline`` installs nothing even
   when both flags are on — the baseline phase must run without delivery
   (Y-type natural stop, no ``svc_intervention`` diagnostics).
3. **r40 P2/C8: the k.5 post-baseline install point is gone.**  Installing on
   the persistent ``self.emulator`` after ``run_baseline`` was dead wiring
   (subsequent phases run ``fresh_emulator=True``; smokeZ measured
   instructions=0 / delivery_count=0).  The method was deleted; production
   delivery now goes through the ``LSGEMU_IRQ_DELIVERY_PHASES`` phase
   whitelist installed inside each phase's own temp emulator (see
   ``test_irq_delivery_phase_wiring.py``).
4. **The counterexample invariant survives the main flow.**  A ``b .`` sink
   whose symbol is a fault terminal (``_unhandled_exception`` …) still stops
   the replay even with a pending modelled event — the r31 P4 tokens are the
   only deferral window, and no new token was added for the port sentinel's
   neighbours.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest import mock

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401
from lsgemu import historical_runner as hr
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator


class BaselineFlagDefaults(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LSGEMU_IRQ_DELIVERY_BASELINE", None)
            os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
            self.assertFalse(hr.baseline_irq_delivery_enabled())
            self.assertFalse(hr.irq_delivery_enabled())

    def test_gate_requires_both_flags(self):
        with mock.patch.dict(os.environ, {"LSGEMU_IRQ_DELIVERY": "1"}, clear=False):
            os.environ.pop("LSGEMU_IRQ_DELIVERY_BASELINE", None)
            self.assertFalse(hr.baseline_irq_delivery_enabled())
        with mock.patch.dict(
            os.environ,
            {"LSGEMU_IRQ_DELIVERY_BASELINE": "1"},
            clear=False,
        ):
            os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
            self.assertFalse(hr.irq_delivery_enabled())


class _Zero:
    """Falsy, empty, callable-zero stand-in for unstubbed diagnostics."""

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __call__(self, *args, **kwargs):
        return 0

    def __getitem__(self, key):
        return _Zero()

    def get(self, key, default=None):
        return default


class _RecordingRunner:
    """Host for ``run_baseline`` with every heavy collaborator stubbed."""

    def __init__(self):
        self.installed = []
        self.uninstalled = []
        self.phase_metadata = {}
        # Read-only bookkeeping attributes the real dataclass always carries.
        self.known_main_branch_events = {}
        self.known_branch_root_snapshots = {}
        self.dynamic_successors = {}
        self.irq_delivery_controller = None

    def __getattr__(self, name):
        # Anything not explicitly stubbed reads as a "zero-shaped" object so
        # the phase-record tail cannot fail on diagnostics we do not model:
        # falsy, len()==0, iterates empty, and calls return 0.
        if name.startswith("__"):
            raise AttributeError(name)
        return _Zero()

    # -- collaborators stubbed by name ---------------------------------
    def _install_irq_delivery(self, emulator):
        controller = mock.Mock()
        controller.audit_payload.return_value = {"enabled": True, "irq": 50}
        controller.uninstall.side_effect = lambda: self.uninstalled.append(True)
        self.installed.append(emulator)
        self.irq_delivery_controller = controller
        return controller

    _record_snapshot_resource_event = lambda self, name: None  # noqa: E731

    def _record_phase(self, name, covered, **metadata):
        self.phase_metadata = {"name": name, **metadata}
        return set(covered)


def _run_baseline_with(runner, **env):
    """Drive the install/uninstall slice of run_baseline without a real replay."""
    emulator = mock.Mock()
    emulator.stop_after_no_new_bbs = None
    emulator.run.return_value = {"stop_reason": "completed"}
    emulator.branch_snapshot_manager.get_ordered_events.return_value = []
    emulator.branch_snapshot_manager.get_statistics.return_value = {}
    emulator.bb_addr_set = set()
    runner.emulator = emulator
    runner.mmio_handler = mock.Mock()
    runner.register_tracer = mock.Mock()
    runner.prepared = mock.Mock()
    runner._configure_emulator_feedback = lambda emu: None
    runner._apply_recorded_input_trace = lambda mmio, phase: {}
    runner.guided_modules = lambda mmio, tracer=None, **kw: mock.MagicMock()
    runner.capture_execution_trace = lambda emu: mock.MagicMock()
    runner._materialize_replay_failure_result = (
        lambda reason, **kw: {"stop_reason": reason}
    )
    runner._normalize_replay_run_result = lambda result, **kw: result
    runner._safe_replay_postprocess = (
        lambda operation, callback, run_result=None, default=None: (
            callback() if callable(callback) else default
        )
    )
    runner._remember_main_branch_events = lambda events, **kw: 0
    runner._remember_branch_root_snapshots = lambda events, snaps, **kw: 0
    runner._record_dynamic_successors = lambda events: 0
    runner._record_emulator_runtime_successors = lambda emu: 0
    runner._ensure_dynamic_dispatch_root_events = lambda covered, **kw: 0
    runner._harvest_emulator_feedback = lambda emu: None
    runner._ordered_replay_snapshots = lambda emu: []
    runner._summary_skip_coverage_accounting = lambda emu, covered: {}
    runner._dump_input_trace = lambda mmio, phase, result: {}
    runner._wall_forensics = lambda emu, result: {}
    runner._temp_stage_for = lambda name: name
    with mock.patch.dict(os.environ, env, clear=False):
        os.environ.pop("LSGEMU_IRQ_DELIVERY_BASELINE", None)
        os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
        for key, value in env.items():
            os.environ[key] = value
        try:
            hr.HistoricalRunner.run_baseline(
                runner,
                max_instructions=10,
                enable_branch_snapshots=False,
            )
        finally:
            for key in env:
                os.environ.pop(key, None)
    return runner


class InstallGate(unittest.TestCase):
    def test_flags_off_installs_nothing(self):
        runner = _RecordingRunner()
        _run_baseline_with(runner)
        self.assertEqual(runner.installed, [])
        self.assertIsNone(runner.phase_metadata.get("irq_delivery_audit"))

    def test_flags_on_baseline_phase_installs_nothing(self):
        """P2 核心性质：双开时 baseline 相位本身零投递（Y 型自然停点），
        相位 metadata 不再带 irq_delivery_audit（r36 曾在重放前装、finally
        摘并把审计记进相位——cycle2 证实这让 baseline 整相位 diagnostic 化）。"""
        runner = _RecordingRunner()
        _run_baseline_with(
            runner,
            LSGEMU_IRQ_DELIVERY="1",
            LSGEMU_IRQ_DELIVERY_BASELINE="1",
        )
        self.assertEqual(runner.installed, [])
        self.assertIsNone(runner.phase_metadata.get("irq_delivery_audit"))

    def test_r40_dead_post_baseline_helper_removed(self):
        """r40 P2/C8：k.5 的基线后安装点（装在常驻 self.emulator，其后相位
        不执行该实例）已删除；产线投递走 LSGEMU_IRQ_DELIVERY_PHASES 相位
        白名单（test_irq_delivery_phase_wiring.py 钉住）。"""
        self.assertFalse(
            hasattr(hr.HistoricalRunner, "install_post_baseline_irq_delivery")
        )


class FaultSinkCounterexample(unittest.TestCase):
    """The deferral window did not widen: fault symbols still stop."""

    class _Predicates:
        IRQ_WAIT_SINK_SYMBOL_TOKENS = IntelligentEmulator.IRQ_WAIT_SINK_SYMBOL_TOKENS
        _symbol_name_for_bb = IntelligentEmulator._symbol_name_for_bb
        _ensure_irq_symbol_table = IntelligentEmulator._ensure_irq_symbol_table
        _is_interruptible_wait_sink = IntelligentEmulator._is_interruptible_wait_sink
        _irq_delivery_has_pending_event = (
            IntelligentEmulator._irq_delivery_has_pending_event
        )
        _irq_delivery_defers_terminal_stop = (
            IntelligentEmulator._irq_delivery_defers_terminal_stop
        )

    class _Controller:
        def has_pending_event(self):
            return True

    def _stub(self):
        stub = self._Predicates()
        stub.irq_delivery_enabled = True
        stub.irq_delivery_controller = self._Controller()
        stub.irq_delivery_deferral_stats = Counter()
        stub.symbols_by_addr = {}
        stub._irq_symbol_by_addr = None
        stub._irq_symbol_starts_cache = None
        stub._irq_sink_symbol_cache = {}

        def _load_symbols():
            return {}

        stub._load_elf_all_symbols_by_name = _load_symbols
        return stub

    def test_real_fault_sink_stops_even_with_pending_event(self):
        stub = self._stub()
        stub.symbols_by_addr = {
            0x08004FBA: "_unhandled_exception",
            0x081360C4: "__idle_thread",
        }
        self.assertFalse(stub._irq_delivery_defers_terminal_stop(0x08004FBA))
        self.assertTrue(stub._irq_delivery_defers_terminal_stop(0x081360C4))
        self.assertEqual(
            stub.irq_delivery_deferral_stats["fault_terminal_sink"], 1
        )

    def test_port_sentinel_still_defers_but_zombies_and_errors_do_not_widen(self):
        stub = self._stub()
        stub.symbols_by_addr = {
            0x08005100: ".zombies",
            0x08005108: "__port_exit_from_isr",
            0x080DB04C: "NMI_Handler",
        }
        # Port-exit wait defers (it is the SVC trampoline), NMI does not.
        self.assertTrue(stub._irq_delivery_defers_terminal_stop(0x08005108))
        self.assertFalse(stub._irq_delivery_defers_terminal_stop(0x080DB04C))


if __name__ == "__main__":
    unittest.main()
