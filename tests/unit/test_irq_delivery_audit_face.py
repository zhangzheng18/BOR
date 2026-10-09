#!/usr/bin/env python3
"""r41 D1 单测：投递审计面**无条件**产出「装到没装、跑没跑到」。

k.7 回炉的阻断项：k.4/k.5 之后 ON 臂相位聚合 ``irq_delivery`` 只有 5 个
计数器（enabled/install_attempts/installed/tasks_with_delivery/deliveries），
没有 ``instructions``——``FINAL_PLAN_CONSOLIDATION.md`` §2-3 的机器判据
``audit.instructions > 0 ∧ delivery_count > 0`` 在字段层面不可判；且零投递时
``IrqDeliveryController.run_result_fields`` 整体 fail-silent。本文件钉住：

1. **零投递也有审计面**（控制器级）：真执行 N 条指令后 ``run_result_fields``
   携带 ``irq_delivery.audit``，其中 ``instructions`` 等于**实测**执行指令数
   （独立 UC_HOOK_CODE 计数器对账，不是 0 常量）、``errors`` 键在（可为空
   dict）、``tim5_summary``/``svc_stats`` 键在（器件未注册 = None/空）。
2. **证据语义不被稀释**：零投递时 ``irq_event_delivered`` / ``svc_dispatch``
   字段**缺席**（环境输入事实只在真投递时成立）；一次投递发生后二者齐全
   且 ``instructions > 0``。
3. **相位聚合面**（接线级）：白名单命中相位零投递也聚合
   instructions/errors/tim5_summary/svc_stats + 每任务 ``stop_reason``；
   缺省空白名单（OFF）相位 dict 保持 r40 六键形态逐位不变。
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
import unicorn
from lsgemu import historical_runner as hr
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.isr_explorer.isr_explorer import IrqDeliveryController
from lsgemu.scheduler.queue_policy import QueuePolicy
from lsgemu.dfs_flip_kit import (
    ENTRY_PC,
    FLIP_BRANCH_BB,
    LOAD_BASE,
    build_flip_emulator,
    build_flip_image,
)
from unicorn import (
    UC_ARCH_ARM,
    UC_MODE_THUMB,
    UC_PROT_ALL,
    UC_PROT_EXEC,
)
from unicorn.arm_const import UC_ARM_REG_SP
from unicorn.arm_const import UC_CPU_ARM_CORTEX_M4

try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0


STACK_BASE = 0x20000000
STACK_SIZE = 0x10000
CODE_BASE = 0x08000000
CODE_SIZE = 0x1000
ISR_ADDR = 0x08000400
LOOP_ADDR = 0x08000200

# Thumb: ``b .``（0xE7FE，小端 FE E7）——自旋，emu_start 的 count=N 即恰好
# 执行 N 条指令，给「instructions == 实测执行数」一个可独立对账的量。
SELF_LOOP = bytes.fromhex("fee7")


def _new_cpu():
    uc = unicorn.Uc(UC_ARCH_ARM, UC_MODE_THUMB)
    uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M4)
    uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_ALL)
    uc.mem_map(CODE_BASE, CODE_SIZE, UC_PROT_ALL | UC_PROT_EXEC)
    uc.mem_write(LOOP_ADDR, SELF_LOOP)
    return uc


def _new_controller(uc):
    controller = IrqDeliveryController(
        SimpleNamespace(uc=uc, mmio_handler=None)
    )
    controller.vector = ISR_ADDR
    controller.vector_table_base = CODE_BASE
    controller.emu.irq_delivery_watchdog_spins = 0
    controller.emu.causal_context = None
    return controller


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class ZeroDeliveryAuditFaceTests(unittest.TestCase):
    """牙：零投递审计面必须反映真实执行，而不是 0 常量。"""

    def test_zero_delivery_audit_tracks_real_instruction_count(self):
        uc = _new_cpu()
        controller = _new_controller(uc)
        self.assertTrue(controller.install())
        independent = {"count": 0}
        uc.hook_add(
            unicorn.UC_HOOK_CODE,
            lambda _uc, _address, _size, _user: independent.__setitem__(
                "count", independent["count"] + 1
            ),
        )
        expected = 37
        uc.emu_start(LOOP_ADDR | 1, 0, count=expected)
        # 独立计数器对账：emu_start(count=N) 真的执行了 N 条。
        self.assertEqual(independent["count"], expected)
        self.assertEqual(controller.instructions, expected)
        self.assertEqual(controller.deliveries, [])
        self.assertEqual(controller.svcs, [])

        fields = controller.run_result_fields()
        # 零投递也带 irq_delivery 审计 blob……
        delivery = fields.get("irq_delivery")
        self.assertIsInstance(delivery, dict)
        self.assertEqual(delivery.get("delivery_count"), 0)
        audit = delivery.get("audit")
        self.assertIsInstance(audit, dict)
        # ……且 instructions 是实测执行数，不是 0 常量。
        self.assertEqual(audit.get("instructions"), expected)
        # errors 可为空 dict，但键必须在；器件面键无条件在。
        self.assertIn("errors", audit)
        self.assertEqual(audit.get("errors"), {})
        self.assertIn("tim5_summary", audit)
        self.assertIsNone(audit.get("tim5_summary"))
        self.assertIn("svc_stats", audit)
        self.assertEqual(audit.get("svc_stats"), {})
        # 证据语义不被稀释：环境输入事实缺席（= 未发生）。
        self.assertNotIn("irq_event_delivered", fields)
        self.assertNotIn("svc_dispatch", fields)

    def test_zero_delivery_evidence_compaction_stays_absent(self):
        # r41 D1 配套：翻转 pass 的 attempt 证据压实只收事实——零投递审计
        # blob 不得变成 attempt 记录里的 irq_delivery 证据。
        uc = _new_cpu()
        controller = _new_controller(uc)
        self.assertTrue(controller.install())
        uc.emu_start(LOOP_ADDR | 1, 0, count=5)
        fields = controller.run_result_fields()
        evidence = HistoricalRunner._flip_attempt_mechanism_evidence(fields)
        self.assertNotIn("irq_delivery", evidence)
        self.assertEqual(evidence, {})


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class DeliveredAuditFaceTests(unittest.TestCase):
    """牙：一次投递发生后 deliveries>0 且 instructions>0。"""

    def test_delivery_makes_audit_and_evidence(self):
        uc = _new_cpu()
        controller = _new_controller(uc)
        self.assertTrue(controller.install())
        executed = 11
        uc.emu_start(LOOP_ADDR | 1, 0, count=executed)
        uc.reg_write(UC_ARM_REG_SP, STACK_BASE + STACK_SIZE - 0x40)
        record = controller._deliver(at_pc=LOOP_ADDR)
        self.assertIsNotNone(record)
        self.assertEqual(len(controller.deliveries), 1)

        fields = controller.run_result_fields()
        delivery = fields.get("irq_delivery") or {}
        audit = delivery.get("audit") or {}
        self.assertEqual(delivery.get("delivery_count"), 1)
        self.assertGreater(audit.get("instructions", 0), 0)
        self.assertEqual(audit.get("instructions"), executed)
        self.assertEqual(
            fields.get("irq_event_delivered"), "interrupt_delivery"
        )
        evidence = HistoricalRunner._flip_attempt_mechanism_evidence(fields)
        self.assertEqual(
            (evidence.get("irq_delivery") or {}).get("delivery_count"), 1
        )


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class PhaseAuditAggregationTests(unittest.TestCase):
    """牙：相位级 irq_delivery 零投递也含审计四件 + 每任务 stop_reason。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_audit_face_")
        self.firmware_path = str(Path(self._temporary.name) / "flip.bin")
        build_flip_image(self.firmware_path)
        # 向量表下标 16+50（TIM5）落一个可信 handler，让**真**控制器能装上
        # （install 的向量可信性检查：code 区内即过）。
        image = bytearray(Path(self.firmware_path).read_bytes())
        import struct

        struct.pack_into("<I", image, (16 + 50) * 4, (ENTRY_PC + 0x14) | 1)
        Path(self.firmware_path).write_bytes(bytes(image))
        self.temp_emulators = []

    def tearDown(self):
        for emulator in self.temp_emulators:
            with contextlib.suppress(Exception):
                emulator.uc.close()
        self._temporary.cleanup()

    def _new_temp_emulator(self, **kwargs):
        emulator = build_flip_emulator(self.firmware_path)
        # 向量表基址 = 镜像装载基址（真安装路径读 vtor+264 命中上面的补丁）。
        emulator.vector_table_base = LOAD_BASE
        self.temp_emulators.append(emulator)
        return emulator

    def _audit_shell(self, emulator: IntelligentEmulator) -> HistoricalRunner:
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
        runner._safe_dispose_temp_emulator = lambda emu, **kw: None
        runner._replay_result_has_failure = lambda result: False
        runner._record_emulator_runtime_successors = lambda emu: 0
        return runner

    def test_whitelisted_phase_aggregates_zero_delivery_audit(self):
        emulator = build_flip_emulator(self.firmware_path)
        emulator.vector_table_base = LOAD_BASE
        self.temp_emulators.append(emulator)
        runner = self._audit_shell(emulator)
        env = {
            "LSGEMU_IRQ_DELIVERY": "1",
            "LSGEMU_IRQ_DELIVERY_PHASES": "branch_reservoir_round_*",
        }
        with patch.dict(os.environ, env, clear=False):
            runner.run_reservoir_branch_exploration(
                time_limit_seconds=10,
                replay_instructions=200,
                replay_timeout=200_000,
                max_tasks=2,
                phase_name="branch_reservoir_round_1",
            )
        phase_meta = runner.phase_metadata.get("branch_reservoir_round_1", {})
        stats = phase_meta.get("irq_delivery") or {}
        # 基础五计数器语义不变。
        self.assertTrue(stats.get("enabled"))
        self.assertGreaterEqual(stats.get("installed", 0), 1)
        self.assertEqual(stats.get("tasks_with_delivery", 0), 0)
        self.assertEqual(stats.get("deliveries", 0), 0)
        # r41 D1 审计四件：零投递时也在，且 instructions 是真执行读数。
        self.assertIn("instructions", stats)
        self.assertGreater(stats.get("instructions", 0), 0)
        self.assertIn("errors", stats)
        self.assertIn("tim5_summary_last", stats)
        self.assertIn("svc_stats", stats)
        # 每任务 stop_reason：非空字符串（指令帽/墙钟/合并停……）。
        task_audits = stats.get("task_audits") or []
        self.assertGreaterEqual(len(task_audits), 1)
        for entry in task_audits:
            self.assertIn("stop_reason", entry)
            self.assertIsInstance(entry["stop_reason"], str)
            self.assertTrue(entry["stop_reason"].strip())
            self.assertGreater(entry.get("instructions", 0), 0)
            self.assertIn("errors", entry)
            self.assertIn("tim5_summary", entry)
            self.assertIn("svc_stats", entry)
        # 与真控制器对账：相位 instructions = 各任务审计 instructions 之和。
        self.assertEqual(
            stats.get("instructions"),
            sum(entry.get("instructions", 0) for entry in task_audits),
        )

    def test_default_empty_whitelist_keeps_r40_six_key_shape(self):
        # OFF 臂逐位不变：缺省空白名单 ⇒ 相位 dict 保持 r40 的六键形态，
        # 不出现 r41 新审计键（新键只在控制器真装上时由聚合块写入）。
        emulator = build_flip_emulator(self.firmware_path)
        emulator.vector_table_base = LOAD_BASE
        self.temp_emulators.append(emulator)
        runner = self._audit_shell(emulator)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LSGEMU_IRQ_DELIVERY_PHASES", None)
            os.environ.pop("LSGEMU_IRQ_DELIVERY", None)
            runner.run_reservoir_branch_exploration(
                time_limit_seconds=10,
                replay_instructions=200,
                replay_timeout=200_000,
                max_tasks=2,
                phase_name="branch_reservoir_round_1",
            )
        phase_meta = runner.phase_metadata.get("branch_reservoir_round_1", {})
        stats = phase_meta.get("irq_delivery") or {}
        self.assertEqual(
            sorted(stats.keys()),
            [
                "deliveries",
                "enabled",
                "install_attempts",
                "installed",
                "phase_name",
                "tasks_with_delivery",
            ],
        )
        self.assertFalse(stats.get("enabled"))
        self.assertEqual(stats.get("install_attempts", None), 0)


if __name__ == "__main__":
    unittest.main()
