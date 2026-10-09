#!/usr/bin/env python3
"""r40 P3/C7 单测：异常入口/解栈**一份实现，两侧共用**。

k.4 P1-1/P1-2 合并后必须成立的不变量：

1. explorer 冷启动注入（``ISRExplorer.build_exception_entry`` 包装）与
   投递/重放（``IrqDeliveryController._deliver`` / ``_dispatch_svc``）都
   路由到**同一个**模块级 ``build_exception_entry``；
2. stack_fixer 合约按实写帧长开窗（104B 扩展帧不再写出修复窗 72B）；
3. 投递入口带上栈窗修复（此前 ``_deliver`` 不带 stack_fixer，SP 失效即
   入口失败）；修复级联与 explorer 同源（``ensure_writable_exception_frame``）；
4. ``LSGEMU_ISR_LEGACY_FRAME`` 缺省 = 旧语义（旧体逐位保留）；显式 =0 才
   走共享构造器（EXC_RETURN=0xFFFFFFF9 / 32B 基本帧 / PC=ISR|1 不变）。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  (unicorn 绑定补丁)
import unicorn
from lsgemu.isr_explorer import isr_explorer as ie
from lsgemu.isr_explorer.isr_explorer import (
    FRAME_SIZE_BASIC,
    FRAME_SIZE_EXTENDED,
    ISRExplorer,
    IrqDeliveryController,
    ensure_writable_exception_frame,
)
from unicorn import (
    UC_ARCH_ARM,
    UC_MODE_THUMB,
    UC_PROT_ALL,
    UC_PROT_EXEC,
)
from unicorn.arm_const import (
    UC_ARM_REG_LR,
    UC_ARM_REG_PC,
    UC_ARM_REG_PSP,
    UC_ARM_REG_SP,
    UC_CPU_ARM_CORTEX_M4,
)

try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0

STACK_BASE = 0x20000000
STACK_SIZE = 0x10000
CODE_BASE = 0x08000000
CODE_SIZE = 0x1000
ISR_ADDR = 0x08000400


def _new_cpu():
    uc = unicorn.Uc(UC_ARCH_ARM, UC_MODE_THUMB)
    uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M4)
    uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_ALL)
    uc.mem_map(CODE_BASE, CODE_SIZE, UC_PROT_ALL | UC_PROT_EXEC)
    return uc


def _new_explorer(uc):
    return ISRExplorer(uc, static_bbs={}, vtor=CODE_BASE)


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class SharedEntryRoutingTests(unittest.TestCase):
    """牙：两侧必须命中同一个模块级构造器。"""

    def test_explorer_wrapper_and_delivery_use_same_module_entry(self):
        uc = _new_cpu()
        explorer = _new_explorer(uc)
        controller = IrqDeliveryController(
            SimpleNamespace(uc=uc, mmio_handler=None)
        )
        controller.vector = ISR_ADDR
        controller.vector_table_base = CODE_BASE
        controller.emu.irq_delivery_watchdog_spins = 0
        controller.emu.causal_context = None
        uc.reg_write(UC_ARM_REG_SP, STACK_BASE + STACK_SIZE - 0x40)

        with patch.object(
            ie,
            "build_exception_entry",
            wraps=ie.build_exception_entry,
        ) as spy:
            explorer.build_exception_entry(
                irq=50, vector=ISR_ADDR, frame_format="basic32"
            )
            self.assertEqual(spy.call_count, 1)
            record = controller._deliver(at_pc=0x08000200)
            self.assertEqual(spy.call_count, 2)
        self.assertIsNotNone(record)
        self.assertEqual(len(controller.deliveries), 1)

    def test_dispatch_svc_uses_same_module_entry(self):
        uc = _new_cpu()
        controller = IrqDeliveryController(
            SimpleNamespace(uc=uc, mmio_handler=None)
        )
        controller.svc_vector = ISR_ADDR
        controller.vector_table_base = CODE_BASE
        controller.emu.irq_delivery_watchdog_spins = 0
        controller.emu.causal_context = None
        uc.reg_write(UC_ARM_REG_SP, STACK_BASE + STACK_SIZE - 0x40)
        with patch.object(
            ie, "build_exception_entry", wraps=ie.build_exception_entry
        ) as spy:
            dispatched = controller._dispatch_svc(0x08000200, 0, 2, 0)
        self.assertTrue(dispatched)
        self.assertEqual(spy.call_count, 1)


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class StackFixerContractTests(unittest.TestCase):
    """Side B 缺陷二：修复窗按实写帧长开（104B 不再写出窗口 72B）。"""

    def test_fixer_receives_written_frame_size(self):
        seen = []

        def fixer(sp, frame_size):
            seen.append(int(frame_size))
            return (int(sp) & ~7) - int(frame_size), "test"

        uc = _new_cpu()
        # ext104 ⇒ EXC_RETURN Thread/PSP ⇒ 栈银行读 PSP。
        uc.reg_write(UC_ARM_REG_PSP, STACK_BASE + STACK_SIZE - 0x40)
        uc.reg_write(UC_ARM_REG_SP, STACK_BASE + STACK_SIZE - 0x40)
        entry = ie.build_exception_entry(
            uc,
            irq=50,
            vector=ISR_ADDR,
            frame_format="ext104",
            stack_fixer=fixer,
        )
        self.assertEqual(seen, [FRAME_SIZE_EXTENDED])
        self.assertEqual(entry["frame_size"], FRAME_SIZE_EXTENDED)
        frame_base = int(entry["frame_base"], 16)
        # 帧基址 == fixer 返回的窗口基址（旧行为会低 72B）。
        self.assertEqual(frame_base, ((STACK_BASE + STACK_SIZE - 0x40) & ~7) - FRAME_SIZE_EXTENDED)
        # 104B 全窗可写（帧内容真实落盘）。
        window = bytes(uc.mem_read(frame_base, FRAME_SIZE_EXTENDED))
        self.assertEqual(len(window), FRAME_SIZE_EXTENDED)

    def test_module_fixer_opens_window_by_frame_size(self):
        uc = _new_cpu()
        # SP 落在未映射区（0x30000000 在 SRAM 启发式范围内 ⇒ current_sp_mapped
        # 臂会把窗口页映射出来）。
        frame_start, source = ensure_writable_exception_frame(
            uc, 0x30000000, FRAME_SIZE_EXTENDED
        )
        self.assertEqual(source, "current_sp_mapped")
        self.assertEqual(frame_start, 0x30000000 - FRAME_SIZE_EXTENDED)
        # 整个 104B 窗口可写。
        uc.mem_write(frame_start, b"\x5a" * FRAME_SIZE_EXTENDED)
        self.assertEqual(
            bytes(uc.mem_read(frame_start, FRAME_SIZE_EXTENDED)),
            b"\x5a" * FRAME_SIZE_EXTENDED,
        )


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class DeliveryStackRepairTests(unittest.TestCase):
    """Side B 缺陷一：投递入口带上与 explorer 同源的栈窗修复。"""

    def test_delivery_with_invalid_sp_repairs_instead_of_failing(self):
        uc = _new_cpu()
        controller = IrqDeliveryController(
            SimpleNamespace(uc=uc, mmio_handler=None)
        )
        controller.vector = ISR_ADDR
        controller.vector_table_base = CODE_BASE
        controller.emu.irq_delivery_watchdog_spins = 0
        controller.emu.causal_context = None
        # SP 失效（未映射）——旧行为：入口抛异常进 errors；新行为：修复。
        uc.reg_write(UC_ARM_REG_SP, 0x30000000)
        record = controller._deliver(at_pc=0x08000200)
        self.assertIsNotNone(record)
        self.assertEqual(len(controller.deliveries), 1)
        entry_errors = {
            key for key in controller.errors if key.startswith("entry:")
        }
        self.assertEqual(entry_errors, set())
        self.assertGreater(
            sum(
                controller.stack_repair_stats.get(key, 0)
                for key in ("stack_frame_native", "stack_frame_repaired")
            ),
            0,
        )
        self.assertIn("stack_repair_stats", controller.audit_payload())


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class LegacyFrameGateTests(unittest.TestCase):
    """LSGEMU_ISR_LEGACY_FRAME 缺省 = 旧语义；=0 走共享构造器。"""

    def _fresh(self):
        uc = _new_cpu()
        return uc, _new_explorer(uc)

    def test_default_env_keeps_legacy_body(self):
        uc, explorer = self._fresh()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LSGEMU_ISR_LEGACY_FRAME", None)
            with patch.object(
                ie,
                "build_exception_entry",
                wraps=ie.build_exception_entry,
            ) as spy:
                explorer._setup_isr_context(ISR_ADDR, 50)
        # 缺省 legacy：不得路由到共享构造器。
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(
            int(uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF, 0xFFFFFFF9
        )
        self.assertEqual(int(uc.reg_read(UC_ARM_REG_PC)) & ~1, ISR_ADDR)

    def test_legacy_off_routes_through_shared_constructor(self):
        uc, explorer = self._fresh()
        with patch.dict(
            os.environ, {"LSGEMU_ISR_LEGACY_FRAME": "0"}, clear=False
        ):
            with patch.object(
                ie,
                "build_exception_entry",
                wraps=ie.build_exception_entry,
            ) as spy:
                explorer._setup_isr_context(ISR_ADDR, 50)
        self.assertEqual(spy.call_count, 1)
        # 共享臂的关键语义锚点与旧体一致：EXC_RETURN / 帧长 / PC。
        self.assertEqual(
            int(uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF, 0xFFFFFFF9
        )
        self.assertEqual(int(uc.reg_read(UC_ARM_REG_PC)) & ~1, ISR_ADDR)
        sp = int(uc.reg_read(UC_ARM_REG_SP)) & 0xFFFFFFFF
        frame = bytes(uc.mem_read(sp, FRAME_SIZE_BASIC))
        self.assertEqual(len(frame), FRAME_SIZE_BASIC)
        # 32B 基本帧（EXC_RETURN.bit4=1）。
        stacked_xpsr = int.from_bytes(frame[28:32], "little")
        self.assertTrue(stacked_xpsr & 0x01000000)  # Thumb 位

    def test_explicit_legacy_on_keeps_legacy_body(self):
        uc, explorer = self._fresh()
        with patch.dict(
            os.environ, {"LSGEMU_ISR_LEGACY_FRAME": "1"}, clear=False
        ):
            with patch.object(
                ie,
                "build_exception_entry",
                wraps=ie.build_exception_entry,
            ) as spy:
                explorer._setup_isr_context(ISR_ADDR, 50)
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(
            int(uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF, 0xFFFFFFF9
        )


if __name__ == "__main__":
    unittest.main()
