#!/usr/bin/env python3
"""cycle3 k.5 C2（P0-A）：VFP 寄存器捕获/恢复保真——roundtrip 回归。

k.4 证据（FINAL_PLAN_CYCLE3 v6-1）：历史 armP3 blob 的捕获面只有 r0–pc，
D0–D31/FPSCR 全空，E4 红面 3/82 的根因正是丢失的 VFP 上下文。本测试钉住：

1. 捕获面（branch_snapshot_manager.py reg_list）带 ``d0..d31`` + ``fpscr``；
2. 恢复面（reg_map）同步，捕获→恢复后逐寄存器相等（含 ``FPSCR≠0`` 样例
   与 D0..D31 非零 64 位样例）；FPSCR 经循环写回、先于 CPSR 写（次序由
   reg_map 循环天然保证）；
3. 旧形态快照（无 VFP 键，integrity hash 按旧键重算自洽）仍可恢复——
   恢复循环 ``if name in snapshot_registers`` 自然跳过缺失键。
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  (unicorn 绑定补丁：先 import lsgemu 再建 Uc)

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager

try:
    import unicorn
    from unicorn import arm_const
    from unicorn import UC_MODE_MCLASS
except ImportError:  # pragma: no cover
    unicorn = None
    arm_const = None
    UC_MODE_MCLASS = 0

RAM_BASE = 0x20000000
RAM_SIZE = 0x1000
BRANCH_PC = 0x08000100
TARGET_PC = 0x08000200
FALLTHROUGH_PC = 0x08000104
FPSCR_SAMPLE = 0xF0000000  # N|Z|C|V 全 1，非零样例


def _close_quietly(uc):
    with contextlib.suppress(Exception):
        uc.close()


def _build_uc():
    uc = unicorn.Uc(
        unicorn.UC_ARCH_ARM, UC_MODE_MCLASS | unicorn.UC_MODE_THUMB
    )
    uc.mem_map(RAM_BASE, RAM_SIZE)
    uc.mem_write(RAM_BASE, b"\x5a" * 64)
    return uc


def _d_reg(i: int) -> int:
    return getattr(arm_const, f"UC_ARM_REG_D{i}")


@unittest.skipIf(
    unicorn is None or UC_MODE_MCLASS == 0,
    "本机 unicorn 缺少 UC_MODE_MCLASS",
)
class VfpSnapshotRoundtripTests(unittest.TestCase):
    def _manager(self) -> BranchSnapshotManager:
        manager = BranchSnapshotManager()
        manager.set_memory_regions([(RAM_BASE, RAM_SIZE)])
        return manager

    def test_roundtrip_preserves_vfp_and_gprs(self):
        uc = _build_uc()
        self.addCleanup(_close_quietly, uc)
        manager = self._manager()

        expected_vfp = {}
        for i in range(32):
            value = (0xDEAD0000 + i) | ((0x51 + i) << 32)  # 非零 64 位样例
            uc.reg_write(_d_reg(i), value)
            expected_vfp[f"d{i}"] = value
        uc.reg_write(arm_const.UC_ARM_REG_FPSCR, FPSCR_SAMPLE)
        uc.reg_write(arm_const.UC_ARM_REG_R0, 0x13579BDF)
        uc.reg_write(arm_const.UC_ARM_REG_SP, RAM_BASE + 0x800)
        # 本机 unicorn 对 CPSR 强制 EPSR 位（T 等），断言必须对**生效值**做。
        uc.reg_write(arm_const.UC_ARM_REG_CPSR, 0x61000000)
        cpsr_sample = uc.reg_read(arm_const.UC_ARM_REG_CPSR)

        snapshot = manager.save_snapshot(
            uc, BRANCH_PC, TARGET_PC, FALLTHROUGH_PC, "EQ"
        )

        # 1) 捕获面带全部 VFP 键（blob 面随之自然带出）。
        self.assertIn("fpscr", snapshot.registers)
        self.assertEqual(snapshot.registers["fpscr"], FPSCR_SAMPLE)
        for i in range(32):
            self.assertIn(f"d{i}", snapshot.registers)
            self.assertEqual(snapshot.registers[f"d{i}"], expected_vfp[f"d{i}"])

        # 2) 摧毁全部相关寄存器后恢复，逐寄存器必须回到捕获值。
        for i in range(32):
            uc.reg_write(_d_reg(i), 0)
        uc.reg_write(arm_const.UC_ARM_REG_FPSCR, 0)
        uc.reg_write(arm_const.UC_ARM_REG_R0, 0)
        uc.reg_write(arm_const.UC_ARM_REG_SP, 0)
        uc.reg_write(arm_const.UC_ARM_REG_CPSR, 0)

        self.assertTrue(manager.restore_snapshot(uc, snapshot))

        for i in range(32):
            self.assertEqual(
                uc.reg_read(_d_reg(i)),
                expected_vfp[f"d{i}"],
                f"D{i} 恢复后不相等",
            )
        self.assertEqual(
            uc.reg_read(arm_const.UC_ARM_REG_FPSCR), FPSCR_SAMPLE
        )
        self.assertEqual(uc.reg_read(arm_const.UC_ARM_REG_R0), 0x13579BDF)
        self.assertEqual(
            uc.reg_read(arm_const.UC_ARM_REG_SP), RAM_BASE + 0x800
        )
        self.assertEqual(uc.reg_read(arm_const.UC_ARM_REG_CPSR), cpsr_sample)

    def test_legacy_snapshot_without_vfp_keys_still_restores(self):
        # 3) 旧形态（无 VFP 键）快照：integrity hash 按旧键集重算保持自洽，
        #    恢复循环必须跳过缺失键并成功恢复 r0–pc。
        uc = _build_uc()
        self.addCleanup(_close_quietly, uc)
        manager = self._manager()

        uc.reg_write(arm_const.UC_ARM_REG_R3, 0x0BADF00D)
        snapshot = manager.save_snapshot(
            uc, BRANCH_PC, TARGET_PC, FALLTHROUGH_PC, "NE"
        )

        legacy_registers = {
            name: value
            for name, value in snapshot.registers.items()
            if not (name.startswith("d") or name == "fpscr")
        }
        legacy_regions = {
            key: value for key, value in snapshot.memory_regions.items()
        }
        legacy = dataclasses.replace(
            snapshot,
            registers=legacy_registers,
            region_hashes={
                key: manager._region_digest(value)
                for key, value in legacy_regions.items()
            },
            integrity_hash=manager._integrity_digest(
                legacy_registers, snapshot.cpsr, legacy_regions
            ),
        )
        self.assertNotIn("fpscr", legacy.registers)

        uc.reg_write(arm_const.UC_ARM_REG_R3, 0)
        self.assertTrue(manager.restore_snapshot(uc, legacy))
        self.assertEqual(uc.reg_read(arm_const.UC_ARM_REG_R3), 0x0BADF00D)


if __name__ == "__main__":
    unittest.main()
