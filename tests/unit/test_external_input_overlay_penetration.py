#!/usr/bin/env python3
"""r9 B2 单测：外部输入（轮询/等待环退出值）穿透 overlay 生效。

A 件定位的覆盖崩塌根因：EnhancedMMIOHandler 桥接为主 handler 的 overlay 后，
record_bridge_write 把固件每次 MMIO 写镜像进 mmio_state 并按「scoped replay
evidence」在优先级 1 供值，遮蔽主表 static_constraints 与直写 RAM——合法的
轮询环退出 MMIO 写（外部输入）永远读不回来。B2 给主 handler 增加
apply_external_loop_exit_input：写主表的同时把值登记进活跃 overlay 的显式
PC/地址约束，使其先于固件写镜像被消费。
"""

from __future__ import annotations

import unittest

from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler


class ExternalInputOverlayPenetrationTests(unittest.TestCase):
    # r33：本文件验证的是 r9 的 **overlay 穿透机制**，与具体寄存器无关。
    # 地址必须是"纯 RW 外设寄存器"，不能是器件语义 profile 认领的设备状态
    # 寄存器（RCC_CR/CFGR/CSR、PWR_CR/CSR）——那些在 r31 P2 裁定下走读路径
    # 优先级 0，由器件语义供值（见 test_clock_ready_device_semantics.py 的
    # 优先级钉）。RCC_APB2ENR(0x40023840) 正是固件在 stm32_clock_init 里
    # 先写的那类纯配置寄存器，语义上就该是写镜像回读。
    RCC = 0x40023840  # RCC_APB2ENR（纯 RW）
    WRITE_PC = 0x081357A4  # stm32_clock_init 里 orr #0x10000; str r2,[r3]
    POLL_PC = 0x081357A6  # ldr r2,[r3]; lsls #14; bpl（等 HSERDY bit17）

    def _primary_with_bridge_overlay(self):
        primary = StatefulMMIOHandler({}, None)
        overlay = EnhancedMMIOHandler(None, None)
        primary.push_mmio_overlay(overlay)
        return primary, overlay

    def test_firmware_write_mirror_shadows_primary_table(self):
        # 回归钉：overlay 的固件写镜像确实会遮蔽主表 static_constraints
        # （A 件崩塌根因的机制复现——修复前合法外部输入写不进去）。
        primary, overlay = self._primary_with_bridge_overlay()
        overlay.record_bridge_write(self.WRITE_PC, self.RCC, 4, 0x00010001)
        primary.static_constraints[(self.POLL_PC, self.RCC)] = 0x00020000
        self.assertEqual(
            0x00010001,
            primary.handle_read(self.RCC, self.POLL_PC, 4),
        )

    def test_external_loop_exit_input_penetrates_overlay(self):
        primary, overlay = self._primary_with_bridge_overlay()
        overlay.record_bridge_write(self.WRITE_PC, self.RCC, 4, 0x00010001)
        primary.apply_external_loop_exit_input(
            self.POLL_PC, self.RCC, 0x00030001
        )
        self.assertEqual(
            0x00030001,
            primary.handle_read(self.RCC, self.POLL_PC, 4),
        )

    def test_override_lands_in_explicit_pc_constraint(self):
        primary, overlay = self._primary_with_bridge_overlay()
        primary.apply_external_loop_exit_input(
            self.POLL_PC, self.RCC, 0x00020000
        )
        self.assertEqual(
            0x00020000,
            overlay.runtime_pc_constraints[(self.POLL_PC, self.RCC)],
        )
        self.assertEqual(
            0x00020000,
            primary.static_constraints[(self.POLL_PC, self.RCC)],
        )

    def test_pc_zero_uses_address_level_global(self):
        # read_pc 未知时按地址级全局约束登记（与主表 (0, addr) 回退语义对齐）。
        primary, overlay = self._primary_with_bridge_overlay()
        overlay.record_bridge_write(self.WRITE_PC, self.RCC, 4, 0x00010001)
        primary.apply_external_loop_exit_input(0, self.RCC, 0x00020000)
        self.assertEqual(
            0x00020000,
            primary.handle_read(self.RCC, self.POLL_PC, 4),
        )

    def test_other_read_pcs_keep_mirror_semantics(self):
        # PC 约束按读取点生效：未被覆盖的读取点仍按 overlay 原语义供值。
        primary, overlay = self._primary_with_bridge_overlay()
        overlay.record_bridge_write(self.WRITE_PC, self.RCC, 4, 0x00010001)
        primary.apply_external_loop_exit_input(
            self.POLL_PC, self.RCC, 0x00030001
        )
        handled, value = overlay.resolve_bridge_read(0x081357B6, self.RCC, 4)
        self.assertTrue(handled)
        self.assertEqual(0x00010001, value)


if __name__ == "__main__":
    unittest.main()
