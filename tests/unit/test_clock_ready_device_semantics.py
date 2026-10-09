#!/usr/bin/env python3
"""r33 单测：时钟/电源就绪位 = 器件语义（不是常量注入）。

本轮把 `stm32_clock_init` 现场（`0x08135764`，8 个硬件忙等循环）的 5 个预装
常量换成"写使能 → 器件自己置就绪"的器件模型。这个文件把 r27 §2.1 那张
"固件动作 ↔ 就绪位"真值表变成 CI 断言，让"永远返回就绪"的假就绪通路在
CI 层面不可能复发（r27 §3.2-5 的建议）。

被钉住的三条性质：
  1. 固件不使能 ⇒ 器件不置就绪位（真值表）；
  2. 固件写使能 0 ⇒ 就绪位必须转 0（r33 P3-2 反例臂）；
  3. 器件状态寄存器不被固件写镜像遮蔽、也不接受外部注入值——
     读来源必须是 `device_profile`，绝不是 `_try_simple_pattern` 的假 ready。
"""

from __future__ import annotations

import os
import unittest

from lsgemu.analysis.peripheral_semantic_profiles import (
    CLOCK_READY_DISABLE_ENV,
    READY_LATENCY_ENV,
    STM32PWRProfile,
    STM32RCCProfile,
)
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler

RCC_CR = 0x40023800
RCC_CFGR = 0x40023808
RCC_CSR = 0x40023874
PWR_CR = 0x40007000
PWR_CSR = 0x40007004
RCC_APB2ENR = 0x40023840


class _EnvScope:
    """临时设置/还原环境变量（profile 在构造时读取，所以要先设后建）。"""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


class ClockReadyDeviceSemanticsTests(unittest.TestCase):
    """RCC_CR / RCC_CSR / RCC_CFGR / PWR_CR / PWR_CSR 的器件语义。"""

    def setUp(self):
        os.environ.pop(CLOCK_READY_DISABLE_ENV, None)
        os.environ.pop(READY_LATENCY_ENV, None)
        os.environ["LSGEMU_MMIO_SOURCE_AUDIT"] = "1"

    def tearDown(self):
        os.environ.pop("LSGEMU_MMIO_SOURCE_AUDIT", None)

    @staticmethod
    def handler():
        return StatefulMMIOHandler()

    @staticmethod
    def read_source(handler, address):
        return handler.mmio_read_source_counts.get(int(address), {})

    # -- 真值表：不使能 ⇒ 不置位 -----------------------------------------
    def test_ready_bit_requires_the_firmware_enable_write(self):
        handler = self.handler()
        value = handler.handle_read(RCC_CR, 0x081357A6, 4)
        self.assertEqual(0, value & 0x00020000, "没人写 HSEON，HSERDY 不许为 1")

    def test_hse_enable_write_sets_hserdy(self):
        handler = self.handler()
        # 固件现场：ldr r2,[r3] / orr #0x10000 / str r2,[r3]
        current = handler.handle_read(RCC_CR, 0x0813579E, 4)
        handler.handle_write(RCC_CR, 0x081357A4, current | 0x00010000, 4)
        value = handler.handle_read(RCC_CR, 0x081357A6, 4)
        self.assertTrue(value & 0x00010000, "固件写的 HSEON 必须留在寄存器里")
        self.assertTrue(value & 0x00020000, "使能后器件必须回置 HSERDY")

    def test_hsi_and_pll_pairs(self):
        handler = self.handler()
        handler.handle_write(RCC_CR, 0x0813577A, 0x00000001, 4)  # HSION
        self.assertTrue(handler.handle_read(RCC_CR, 0x0813577C, 4) & 0x2)
        current = handler.handle_read(RCC_CR, 0x081357C0, 4)
        handler.handle_write(RCC_CR, 0x081357C6, current | 0x01000000, 4)  # PLLON
        value = handler.handle_read(RCC_CR, 0x081357D2, 4)
        self.assertTrue(value & 0x01000000)
        self.assertTrue(value & 0x02000000, "PLLON 置位后 PLLRDY 必须跟随")

    def test_hsical_bit_is_not_an_enable_bit(self):
        """bit8 是只读校准值 HSICAL[0]，不是使能位（历史假就绪对已删）。"""
        handler = self.handler()
        handler.handle_write(RCC_CR, 0x08000010, 0x00000100, 4)
        value = handler.handle_read(RCC_CR, 0x08000014, 4)
        self.assertEqual(0, value & 0x00000400, "HSICAL[0]=1 不许造出 bit10 假就绪")

    # -- r33 P3-2 反例臂：写 0 ⇒ 就绪位必须转 0 ---------------------------
    def test_clearing_the_enable_bit_clears_the_ready_bit(self):
        handler = self.handler()
        handler.handle_write(RCC_CR, 0x081357A4, 0x00010001, 4)
        self.assertTrue(handler.handle_read(RCC_CR, 0x081357A6, 4) & 0x00020000)
        handler.handle_write(RCC_CR, 0x081357AC, 0x00000001, 4)  # HSEON=0
        value = handler.handle_read(RCC_CR, 0x081357AE, 4)
        self.assertEqual(0, value & 0x00020000, "HSEON 写 0 ⇒ HSERDY 必须转 0")
        self.assertEqual(0, value & 0x00010000)

    def test_pwr_vos_zero_clears_vosrdy(self):
        handler = self.handler()
        handler.handle_write(PWR_CR, 0x08135772, 0x0000C000, 4)
        self.assertTrue(handler.handle_read(PWR_CSR, 0x081357CA, 4) & 0x4000)
        handler.handle_write(PWR_CR, 0x08135774, 0x00000000, 4)  # VOS=0
        value = handler.handle_read(PWR_CSR, 0x081357CC, 4)
        self.assertEqual(0, value & 0x4000, "VOS=0 ⇒ VOSRDY 必须转 0")

    def test_pwr_cr_bit14_is_vos_not_vosrdy(self):
        """PWR_CSR.VOSRDY(bit14) 绝不能写回 PWR_CR —— 那里的 bit15:14 是 VOS。"""
        handler = self.handler()
        handler.handle_write(PWR_CR, 0x08135772, 0x0000C000, 4)
        handler.handle_read(PWR_CSR, 0x081357CA, 4)
        cr = handler.handle_read(PWR_CR, 0x081357CC, 4)
        self.assertEqual(0x0000C000, cr & 0x0000C000)

    # -- RCC_CSR ---------------------------------------------------------
    def test_lsion_drives_lsirdy(self):
        handler = self.handler()
        handler.handle_write(RCC_CSR, 0x081357B2, 0x00000001, 4)  # LSION
        self.assertTrue(handler.handle_read(RCC_CSR, 0x081357B6, 4) & 0x2)
        handler.handle_write(RCC_CSR, 0x081357B4, 0x00000000, 4)
        self.assertEqual(0, handler.handle_read(RCC_CSR, 0x081357B8, 4) & 0x2)

    def test_csr_rmvf_is_write_one_to_clear_and_reads_zero(self):
        handler = self.handler()
        handler.handle_write(RCC_CSR, 0x081350EE, 0x01000000, 4)  # RMVF
        value = handler.handle_read(RCC_CSR, 0x081350D8, 4)
        self.assertEqual(0, value & 0x01000000, "RMVF 读回必须是 0")
        self.assertEqual(0, value & 0xFF000000, "写 1 清 ⇒ 复位标志位被清")

    # -- RCC_CFGR --------------------------------------------------------
    def test_sw_write_is_followed_by_sws(self):
        handler = self.handler()
        handler.handle_write(RCC_CFGR, 0x08135812, 0x00000002, 4)  # SW=PLL
        value = handler.handle_read(RCC_CFGR, 0x08135814, 4)
        self.assertEqual(0x2, (value >> 2) & 0x3, "SW=0b10 ⇒ SWS 必须跟到 0b10")

    def test_sw_back_to_hsi_is_followed_by_sws(self):
        """现场 0x0813578C：固件写 SW=0 后等 SWS==0。"""
        handler = self.handler()
        handler.handle_write(RCC_CFGR, 0x08135812, 0x00000002, 4)
        handler.handle_read(RCC_CFGR, 0x08135814, 4)
        handler.handle_write(RCC_CFGR, 0x08135788, 0x00000000, 4)  # SW=HSI
        value = handler.handle_read(RCC_CFGR, 0x0813578C, 4)
        self.assertEqual(0x0, (value >> 2) & 0x3)

    # -- 读路径优先级（r27 根因 / r33 P2）--------------------------------
    def test_device_profile_beats_the_firmware_write_mirror(self):
        """优先级钉：固件写镜像持有 raw 值，读必须仍是器件派生值。

        用**真实路径**（primary.handle_write → _notify_overlay_write）造镜像：
        overlay.mmio_state 里只有固件写的 0x00010001，**没有** HSERDY。
        """
        primary = StatefulMMIOHandler()
        overlay = EnhancedMMIOHandler(None, None)
        primary.push_mmio_overlay(overlay)
        primary.handle_write(RCC_CR, 0x081357A4, 0x00010001, 4)
        self.assertEqual(0x00010001, overlay.mmio_state[RCC_CR], "镜像里没有就绪位")

        value = primary.handle_read(RCC_CR, 0x081357A6, 4)
        self.assertTrue(value & 0x00020000, "写镜像不得遮蔽器件就绪位")
        self.assertEqual(
            {"device_profile": 1}, dict(self.read_source(primary, RCC_CR))
        )

    def test_device_profile_is_not_a_value_injection_channel(self):
        """显式外部输入约束不能给器件寄存器供值——值必须由固件挣来。"""
        primary = StatefulMMIOHandler()
        overlay = EnhancedMMIOHandler(None, None)
        primary.push_mmio_overlay(overlay)
        primary.handle_write(RCC_CR, 0x081357A4, 0x00010001, 4)
        primary.apply_external_loop_exit_input(0x081357A6, RCC_CR, 0x00030001)

        value = primary.handle_read(RCC_CR, 0x081357A6, 4)
        # 注入值是 0x00030001（bit1=0）；器件语义值 = 固件写的 HSION|HSEON
        # + 器件回的 HSIRDY|HSERDY = 0x00030003。bit1 是判别位。
        self.assertEqual(
            0x00030003, value,
            "读值必须是固件写的使能位 + 器件就绪位，不是注入的 0x00030001",
        )
        self.assertEqual(
            {"device_profile": 1}, dict(self.read_source(primary, RCC_CR))
        )

    def test_pwr_csr_is_not_served_by_simple_pattern(self):
        """P3-2：历史上的假 ready 通路（_try_simple_pattern）不许碰 PWR_CSR。"""
        primary = StatefulMMIOHandler()
        overlay = EnhancedMMIOHandler(None, None)
        primary.push_mmio_overlay(overlay)
        primary.handle_write(PWR_CR, 0x08135772, 0x0000C000, 4)
        self.assertEqual(0x0000C000, overlay.mmio_state[PWR_CR])

        value = primary.handle_read(PWR_CSR, 0x081357CA, 4)
        self.assertTrue(value & 0x4000)
        sources = dict(self.read_source(primary, PWR_CSR))
        self.assertNotIn("pattern", sources)
        self.assertEqual({"device_profile": 1}, sources)

    def test_plain_rw_rcc_register_still_reads_back_the_mirror(self):
        """对照：非器件状态寄存器（APB2ENR）保持写镜像回读语义。"""
        primary = StatefulMMIOHandler()
        overlay = EnhancedMMIOHandler(None, None)
        primary.push_mmio_overlay(overlay)
        overlay.record_bridge_write(0x0813576A, RCC_APB2ENR, 4, 0x10000000)
        self.assertEqual(
            0x10000000, primary.handle_read(RCC_APB2ENR, 0x0813576C, 4)
        )

    # -- 确定性延迟与快照 -------------------------------------------------
    def test_latency_is_measured_in_simulated_accesses(self):
        with _EnvScope(**{READY_LATENCY_ENV: 3}):
            handler = self.handler()
            handler.handle_write(RCC_CR, 0x081357A4, 0x00010001, 4)
            seen = []
            for _ in range(5):
                value = handler.handle_read(RCC_CR, 0x081357A6, 4)
                seen.append(1 if value & 0x00020000 else 0)
            # 写（计数 1）登记 ready_at=1+3=4 ⇒ 访问计数 >= 4 的读才可见。
            self.assertEqual([0, 0, 1, 1, 1], seen)

    def test_ready_state_survives_snapshot_restore(self):
        handler = self.handler()
        handler.handle_write(RCC_CR, 0x081357A4, 0x00010001, 4)
        handler.handle_read(RCC_CR, 0x081357A6, 4)
        snapshot = handler.snapshot_runtime_state()

        restored = self.handler()
        restored.restore_runtime_state(snapshot)
        value = restored.handle_read(RCC_CR, 0x081357A6, 4)
        self.assertTrue(value & 0x00020000, "快照恢复后就绪状态必须一致")

    # -- P3-1 模型关臂（单元级）------------------------------------------
    def test_disable_env_restores_pre_r33_behaviour(self):
        with _EnvScope(**{CLOCK_READY_DISABLE_ENV: 1}):
            handler = self.handler()
            names = [p.name for p in handler.semantic_profiles.profiles]
            self.assertNotIn(STM32PWRProfile.name, names)
            self.assertFalse(
                handler.semantic_profiles.profile_by_name(
                    STM32RCCProfile.name
                ).owns_read(RCC_CR)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
