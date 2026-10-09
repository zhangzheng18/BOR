#!/usr/bin/env python3
"""r40 P2/C8 单测：IRQ 投递相位白名单——装到**真执行的**实例上。

k.4 P0-1 的接线修复：k.5 P2 曾把主流程投递装到常驻 ``self.emulator``，
而其后相位走 ``fresh_emulator=True`` → ``_new_temp_emulator`` ⇒ 钩子挂在
不执行的实例上（smokeZ 实测 instructions=0 / delivery_count=0）。本文件
钉住三件事：

1. **白名单语义**：``LSGEMU_IRQ_DELIVERY_PHASES`` 缺省空 = 现状逐位不变；
   精确 / ``prefix*`` 匹配；master 开关 ``LSGEMU_IRQ_DELIVERY`` 仍然必开；
   ``semantic_entry_prefix`` 在拒绝清单上（写了也不装并告警一次）。
2. **install_irq_delivery_for 门控**：不命中不构造；命中装到给定实例且
   **不写**翻转 pass 专属的 ``self.irq_delivery_controller`` 槽位。
3. **带牙（teeth）**：驱动真实 reservoir 重放任务，断言安装目标实例在该
   任务内**确有指令执行**（instruction_count 在安装→卸载之间前进 >0）、
   每次安装都有配对卸载、审计字段随 run_result 聚合进相位 metadata。
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  (unicorn 绑定补丁：先 import lsgemu 再建 Uc)
from lsgemu import historical_runner as hr
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.scheduler.queue_policy import QueuePolicy
from lsgemu.dfs_flip_kit import (
    ENTRY_PC,
    FLIP_BRANCH_BB,
    build_flip_emulator,
    build_flip_image,
)

try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0


class IrqDeliveryPhaseWhitelistTests(unittest.TestCase):
    """LSGEMU_IRQ_DELIVERY_PHASES 解析语义（纯 env，无引擎）。"""

    def test_empty_whitelist_matches_nothing(self):
        # 缺省空 = 现状逐位不变（k.4 P0-1 的硬条款）。
        for raw in ("", None, "  ", ","):
            with patch.dict(os.environ, {}, clear=False):
                if raw is None:
                    os.environ.pop("LSGEMU_IRQ_DELIVERY_PHASES", None)
                else:
                    os.environ["LSGEMU_IRQ_DELIVERY_PHASES"] = raw
                self.assertFalse(
                    hr.irq_delivery_phase_enabled("branch_reservoir_round_1")
                )

    def test_exact_and_prefix_match(self):
        env = {
            "LSGEMU_IRQ_DELIVERY": "1",
            "LSGEMU_IRQ_DELIVERY_PHASES": (
                "branch_reservoir_round_*, isr_probe_exact"
            ),
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertTrue(
                hr.irq_delivery_phase_enabled("branch_reservoir_round_7")
            )
            self.assertTrue(hr.irq_delivery_phase_enabled("isr_probe_exact"))
            # 前缀不完整 / 其它相位不匹配。
            self.assertFalse(hr.irq_delivery_phase_enabled("branch_reservoir"))
            self.assertFalse(hr.irq_delivery_phase_enabled("baseline"))
            self.assertFalse(hr.irq_delivery_phase_enabled(None))
            self.assertFalse(hr.irq_delivery_phase_enabled(""))

    def test_master_switch_still_required(self):
        # 白名单不越权：master 关 ⇒ 全部 False。
        with patch.dict(
            os.environ,
            {
                "LSGEMU_IRQ_DELIVERY_PHASES": "branch_reservoir_round_*",
            },
            clear=False,
        ):
            os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
            self.assertFalse(
                hr.irq_delivery_phase_enabled("branch_reservoir_round_1")
            )

    def test_semantic_entry_prefix_refused_even_if_listed(self):
        # 拒绝清单相位：写进白名单也不装，且告警只发一次（不刷屏）。
        hr._irq_delivery_refused_warned.clear()
        with patch.dict(
            os.environ,
            {
                "LSGEMU_IRQ_DELIVERY": "1",
                "LSGEMU_IRQ_DELIVERY_PHASES": "semantic_entry_prefix",
            },
            clear=False,
        ):
            with self.assertLogs("lsgemu.historical_runner", level="WARNING"):
                self.assertFalse(
                    hr.irq_delivery_phase_enabled("semantic_entry_prefix")
                )
            # 第二次判定不再告警。
            with self.assertRaises(AssertionError):
                with self.assertLogs(
                    "lsgemu.historical_runner", level="WARNING"
                ):
                    self.assertFalse(
                        hr.irq_delivery_phase_enabled("semantic_entry_prefix")
                    )
        hr._irq_delivery_refused_warned.clear()


class InstallForGateTests(unittest.TestCase):
    """install_irq_delivery_for 的门控与槽位纪律（stub 构造器）。"""

    def _shell(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.irq_delivery_controller = None
        return runner

    def test_whitelisted_phase_builds_on_given_emulator_without_slot_write(self):
        runner = self._shell()
        sentinel = object()
        built = []

        def fake_build(runner_self, emulator):
            built.append(emulator)
            return SimpleNamespace(uninstall=lambda: None, run_result_fields=lambda: {})

        with patch.object(
            HistoricalRunner, "_build_irq_delivery_controller", fake_build
        ), patch.dict(
            os.environ,
            {
                "LSGEMU_IRQ_DELIVERY": "1",
                "LSGEMU_IRQ_DELIVERY_PHASES": "branch_reservoir_round_*",
            },
            clear=False,
        ):
            controller = runner.install_irq_delivery_for(
                sentinel, "branch_reservoir_round_3"
            )
        self.assertIsNotNone(controller)
        self.assertEqual(built, [sentinel])
        # 相位白名单路径不写翻转 pass 专属槽位（调用方持有并 finally 卸载）。
        self.assertIsNone(runner.irq_delivery_controller)

    def test_non_whitelisted_phase_builds_nothing(self):
        runner = self._shell()
        sentinel = object()
        built = []

        def fake_build(runner_self, emulator):
            built.append(emulator)
            return None

        with patch.object(
            HistoricalRunner, "_build_irq_delivery_controller", fake_build
        ), patch.dict(
            os.environ,
            {"LSGEMU_IRQ_DELIVERY": "1"},
            clear=False,
        ):
            os.environ.pop("LSGEMU_IRQ_DELIVERY_PHASES", None)
            controller = runner.install_irq_delivery_for(sentinel, "baseline")
        self.assertIsNone(controller)
        self.assertEqual(built, [])


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class PhaseWiringTeethTests(unittest.TestCase):
    """带牙回归：安装目标实例在相位内确有指令执行，否则 FAIL。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_irq_phase_")
        self.firmware_path = str(Path(self._temporary.name) / "flip.bin")
        build_flip_image(self.firmware_path)
        # 安装/卸载记录：emulator 对象 + 当时的指令计数（牙）。
        self.installs = []
        self.uninstalls = []
        self.temp_emulators = []

    def tearDown(self):
        for emulator in self.temp_emulators:
            with contextlib.suppress(Exception):
                emulator.uc.close()
        self._temporary.cleanup()

    def _recording_build_factory(self):
        """构造类属性可用的桩：闭包（而非绑定方法）才能收到 runner 实参。"""
        recording = self

        def recording_build(runner_self, emulator):
            recording.installs.append(
                (emulator, int(emulator.instruction_count))
            )

            class _RecordingController:
                def uninstall(self_inner):
                    recording.uninstalls.append(
                        (emulator, int(emulator.instruction_count))
                    )

                def run_result_fields(self_inner):
                    return {
                        "irq_event_delivered": "interrupt_delivery",
                        "irq": 50,
                        "irq_delivery": {
                            "source": "wiring_test_stub",
                            "delivery_count": 1,
                        },
                    }

            return _RecordingController()

        return recording_build

    def _wiring_shell(self, emulator: IntelligentEmulator) -> HistoricalRunner:
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.queue_policy = QueuePolicy(runner)
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = "none"
        runner.max_snapshots = 5
        runner.current_stage_name = "branch_reservoir_round_1"
        runner.snapshot_resource_stats = {}
        runner.semantic_obligation_enabled = False
        runner.prepared = SimpleNamespace(
            entry_point=ENTRY_PC,
            static_successors={},
            reachable_from=None,
            static_bb_set=set(),
            branch_instruction_by_bb={},
            resolve_basic_block=lambda address: None,
        )
        runner.dfs_anchor_pool = None
        runner.dfs_flip_ledger = None
        runner.emulator = emulator
        runner.register_tracer = SimpleNamespace(call_stack=[])
        runner.validate_coverage = lambda covered: set(covered or set())
        runner.reservoir_task_queue = deque(
            [((FLIP_BRANCH_BB, 1), {(FLIP_BRANCH_BB, 1): True}, None)]
        )
        runner.reservoir_deferred_tasks = deque()
        runner.reservoir_scoped_probe_tasks = deque()
        runner.reservoir_task_contexts = {}
        runner.reservoir_initialized = True
        runner.known_main_branch_events = {}
        runner.known_branch_root_snapshots = {}
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner.reservoir_explored_edges = set()
        runner.reservoir_explored_paths = set()
        runner.reservoir_incomplete_paths = set()
        runner.reservoir_pending_paths = set()
        runner.reservoir_discovered_branch_keys = set()
        runner.reservoir_new_branch_points = set()
        runner.reservoir_interrupt_contexts = []
        runner.interrupt_contexts = []
        runner.reservoir_root_budget_used = {}
        runner.reservoir_roots_started = set()
        runner.reservoir_roots_started_cumulative = set()
        runner.reservoir_saturated_edges = {}
        runner.reservoir_state_signatures = set()
        runner.reservoir_prefix_snapshots = {}
        runner.reservoir_state_file = None
        runner.remembered_root_snapshots = {}
        runner.reservoir_explored_edge_stats = {}
        runner.dynamic_successors = {}
        runner.branch_mmio_pending_root_keys = set()
        runner.branch_mmio_harvested_root_keys = set()
        runner.branch_snapshot_hotset = set()
        runner.scoped_branch_constraints = {}
        runner.scoped_branch_constraint_match_cache = {}
        runner.scoped_branch_constraint_presence_cache = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.coverage_by_evidence = {}
        runner._dfs_anchor_pool_or_none = lambda: None
        # 重放原语的重协作方：真 temp emulator + 轻量反馈桩（与空队列回归
        # 测试同款边界——不改变重放算法本身）。
        runner._new_temp_emulator = self._new_temp_emulator
        runner._configure_emulator_feedback = lambda emu: None
        runner._new_temp_register_tracer = lambda emu: SimpleNamespace(
            call_stack=[], get_statistics=lambda: {}
        )
        runner.guided_modules = (
            lambda mmio, tracer=None, **kw: contextlib.nullcontext()
        )
        runner.capture_coverage = lambda emu: contextlib.nullcontext(set())
        runner._harvest_emulator_feedback = lambda emu: None
        runner._materialize_replay_failure_result = (
            lambda reason, **kw: {"stop_reason": reason}
        )
        runner._safe_dispose_temp_emulator = (
            lambda emu, **kw: None
        )
        runner._replay_result_has_failure = lambda result: False
        runner._record_emulator_runtime_successors = lambda emu: 0
        return runner

    def _new_temp_emulator(self, **kwargs):
        emulator = build_flip_emulator(self.firmware_path)
        self.temp_emulators.append(emulator)
        return emulator

    def test_whitelisted_phase_installs_on_executing_instance(self):
        emulator = build_flip_emulator(self.firmware_path)
        self.temp_emulators.append(emulator)
        runner = self._wiring_shell(emulator)
        env = {
            "LSGEMU_IRQ_DELIVERY": "1",
            "LSGEMU_IRQ_DELIVERY_PHASES": "branch_reservoir_round_*",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            HistoricalRunner,
            "_build_irq_delivery_controller",
            self._recording_build_factory(),
        ):
            runner.run_reservoir_branch_exploration(
                time_limit_seconds=10,
                replay_instructions=200,
                replay_timeout=200_000,
                max_tasks=2,
                phase_name="branch_reservoir_round_1",
            )
        # 牙 1：白名单命中且至少装到 1 个 temp emulator。
        self.assertGreaterEqual(
            len(self.installs), 1, "白名单相位必须至少安装一次投递控制器"
        )
        # 牙 2：每个安装目标实例在安装→卸载之间确有指令执行。
        for installed_emulator, count_at_install in self.installs:
            matches = [
                count_at_uninstall
                for uninstalled_emulator, count_at_uninstall in self.uninstalls
                if uninstalled_emulator is installed_emulator
            ]
            self.assertEqual(
                len(matches),
                1,
                "每次安装都必须恰好配对一次卸载",
            )
            self.assertGreater(
                matches[0],
                count_at_install,
                "安装目标实例必须在任务内真实执行（instruction_count 前进）",
            )
        # 牙 3：投递事实随 run_result 聚合进相位 metadata。
        phase_meta = runner.phase_metadata.get("branch_reservoir_round_1", {})
        delivery_stats = phase_meta.get("irq_delivery") or {}
        self.assertTrue(delivery_stats.get("enabled"))
        self.assertGreaterEqual(delivery_stats.get("installed", 0), 1)
        self.assertGreaterEqual(
            delivery_stats.get("tasks_with_delivery", 0), 1
        )
        self.assertGreaterEqual(delivery_stats.get("deliveries", 0), 1)

    def test_default_empty_whitelist_leaves_reservoir_untouched(self):
        # 缺省（白名单空）= 现状逐位不变：不安装、相位审计恒 off/0。
        emulator = build_flip_emulator(self.firmware_path)
        self.temp_emulators.append(emulator)
        runner = self._wiring_shell(emulator)
        with patch.dict(os.environ, {}, clear=False), patch.object(
            HistoricalRunner,
            "_build_irq_delivery_controller",
            self._recording_build_factory(),
        ):
            os.environ.pop("LSGEMU_IRQ_DELIVERY_PHASES", None)
            os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
            runner.run_reservoir_branch_exploration(
                time_limit_seconds=10,
                replay_instructions=200,
                replay_timeout=200_000,
                max_tasks=2,
                phase_name="branch_reservoir_round_1",
            )
        self.assertEqual(self.installs, [])
        self.assertEqual(self.uninstalls, [])
        phase_meta = runner.phase_metadata.get("branch_reservoir_round_1", {})
        delivery_stats = phase_meta.get("irq_delivery") or {}
        self.assertFalse(delivery_stats.get("enabled"))
        self.assertEqual(delivery_stats.get("installed", None), 0)


if __name__ == "__main__":
    unittest.main()
