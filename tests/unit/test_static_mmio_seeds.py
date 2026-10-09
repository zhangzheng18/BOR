#!/usr/bin/env python3
"""静态 MMIO 初始种子（设计 §2.3）的回归测试。

覆盖三层：
1. 纯逻辑层：derive_mmio_seed_table 的立即数线索提取、无线索填 0、
   目的寄存器重定义截断、确定性（与输入顺序无关）；
2. handler 层：种子仅在首读未建模路径生效，静态约束遮蔽种子，
   未注册地址读取返回 0 且不抛错（地址空间鲁棒性，§2.2）；
3. 仿真器层：种子表由 static_mmio_accesses 实际导出并注入读取路径，
   开关默认开启、LSGEMU_STATIC_MMIO_SEEDS=0 关闭。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  # 触发包级 bootstrap（unicorn 2.1.4 绑定优先）
from lsgemu.analysis.lightweight_mmio_analysis import derive_mmio_seed_table
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler

CONSOLE_ELF = os.environ.get(
    "LSGEMU_TEST_CONSOLE_ELF",
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Console/Console.elf",
)


def _read_record(pc, bb_addr, address, operands, width=1):
    return {
        "pc": pc,
        "bb_addr": bb_addr,
        "address": address,
        "access_type": "read",
        "width": width,
        "confidence": "exact",
        "kind": "abstract_interpretation",
        "mnemonic": "LDRB" if width == 1 else "LDR",
        "operands": operands,
    }


class DeriveSeedTableTests(unittest.TestCase):
    def test_immediate_clue_used_as_seed(self):
        static_bbs = {
            0x1000: [
                {"address": 0x1000, "mnemonic": "LDRB", "operands": "r3, [r0, #6]", "size": 2},
                {"address": 0x1002, "mnemonic": "CMP", "operands": "r3, #0x10", "size": 2},
                {"address": 0x1004, "mnemonic": "BNE", "operands": "0x00001000", "size": 2},
            ],
        }
        accesses = [_read_record(0x1000, 0x1000, 0x40064006, "r3, [r0, #6]")]
        table, meta = derive_mmio_seed_table(static_bbs, accesses)
        self.assertEqual(table, {0x40064006: 0x10})
        self.assertEqual(meta["from_immediate"], 1)
        self.assertEqual(meta["default_zero"], 0)

    def test_no_clue_falls_back_to_zero(self):
        static_bbs = {
            0x1000: [
                {"address": 0x1000, "mnemonic": "LDRB", "operands": "r3, [r0, #6]", "size": 2},
                {"address": 0x1002, "mnemonic": "TST", "operands": "r3, #0x10", "size": 2},
            ],
        }
        accesses = [_read_record(0x1000, 0x1000, 0x40064006, "r3, [r0, #6]")]
        table, meta = derive_mmio_seed_table(static_bbs, accesses)
        self.assertEqual(table, {0x40064006: 0})
        self.assertEqual(meta["default_zero"], 1)

    def test_clue_rejected_after_dest_register_redefinition(self):
        static_bbs = {
            0x1000: [
                {"address": 0x1000, "mnemonic": "LDR", "operands": "r1, [r0, #0x10]", "size": 2},
                {"address": 0x1002, "mnemonic": "MOVS", "operands": "r1, #0x55", "size": 2},
                {"address": 0x1004, "mnemonic": "CMP", "operands": "r1, #0x77", "size": 2},
            ],
        }
        accesses = [_read_record(0x1000, 0x1000, 0x40064010, "r1, [r0, #0x10]", width=4)]
        table, meta = derive_mmio_seed_table(static_bbs, accesses)
        # 0x77 比较的是 MOVS 的常量而非 MMIO 值，不得作为种子
        self.assertEqual(table, {0x40064010: 0})
        self.assertEqual(meta["from_immediate"], 0)

    def test_register_compare_is_not_a_numeric_clue(self):
        static_bbs = {
            0x1000: [
                {"address": 0x1000, "mnemonic": "LDR", "operands": "r1, [r0]", "size": 2},
                {"address": 0x1002, "mnemonic": "CMP", "operands": "r1, r2", "size": 2},
                {"address": 0x1004, "mnemonic": "CMP", "operands": "r1, #0x20", "size": 2},
            ],
        }
        accesses = [_read_record(0x1000, 0x1000, 0x40053000, "r1, [r0]", width=4)]
        table, _meta = derive_mmio_seed_table(static_bbs, accesses)
        # 寄存器比较无线索，但同变量后续数值约束仍可作种子
        self.assertEqual(table, {0x40053000: 0x20})

    def test_deterministic_and_first_read_pc_wins(self):
        static_bbs = {
            0x1000: [
                {"address": 0x1000, "mnemonic": "LDRB", "operands": "r3, [r0, #6]", "size": 2},
                {"address": 0x1002, "mnemonic": "CMP", "operands": "r3, #0x10", "size": 2},
            ],
            0x2000: [
                {"address": 0x2000, "mnemonic": "LDRB", "operands": "r4, [r1, #6]", "size": 2},
                {"address": 0x2002, "mnemonic": "CMP", "operands": "r4, #0x30", "size": 2},
            ],
        }
        first = _read_record(0x1000, 0x1000, 0x40064006, "r3, [r0, #6]")
        second = _read_record(0x2000, 0x2000, 0x40064006, "r4, [r1, #6]")
        table_a, _ = derive_mmio_seed_table(static_bbs, [first, second])
        table_b, _ = derive_mmio_seed_table(static_bbs, [second, first])
        # 同一地址多个读点：结果与输入顺序无关，由最低 pc 的记录获胜
        self.assertEqual(table_a, table_b)
        self.assertEqual(table_a, {0x40064006: 0x10})

    def test_write_records_and_unresolved_addresses_skipped(self):
        accesses = [
            _read_record(0x1000, 0x1000, 0x40064006, "r3, [r0, #6]"),
            {
                "pc": 0x3000,
                "bb_addr": 0x3000,
                "address": 0x40064008,
                "access_type": "write",
                "width": 4,
                "operands": "r1, [r0, #8]",
            },
            {
                "pc": 0x4000,
                "bb_addr": 0x4000,
                "address": None,
                "access_type": "read",
                "width": 4,
                "operands": "r2, [r5]",
            },
        ]
        table, meta = derive_mmio_seed_table({}, accesses)
        self.assertEqual(table, {0x40064006: 0})
        self.assertEqual(meta["read_records"], 1)


class HandlerSeedTierTests(unittest.TestCase):
    def test_seed_applies_only_to_first_unmodeled_read(self):
        handler = StatefulMMIOHandler(mmio_seed_values={0x40053006: 0x10})
        self.assertEqual(handler.handle_read(0x40053006, 0x08000100, 1), 0x10)
        self.assertEqual(handler.mmio_seed_stats["applied"], 1)
        # 第二次读取由运行时路径接管（fallback：返回最后值）
        self.assertEqual(handler.handle_read(0x40053006, 0x08000100, 1), 0x10)
        self.assertEqual(handler.mmio_seed_stats["applied"], 1)

    def test_seed_width_masked_to_access_size(self):
        handler = StatefulMMIOHandler(mmio_seed_values={0x40053010: 0x1FF})
        self.assertEqual(handler.handle_read(0x40053010, 0x08000100, 1), 0xFF)

    def test_seed_not_used_after_write(self):
        handler = StatefulMMIOHandler(mmio_seed_values={0x40053014: 0x10})
        handler.handle_write(0x40053014, 0x08000100, 0xAB, 1)
        # 写后读回不得返回静态种子；ready-after-write 等模式优先
        value = handler.handle_read(0x40053014, 0x08000100, 1)
        self.assertNotEqual(value, 0x10)
        self.assertEqual(handler.mmio_seed_stats["applied"], 0)

    def test_static_constraint_shadows_seed(self):
        handler = StatefulMMIOHandler(
            {(0, 0x40053006): 0x55},
            mmio_seed_values={0x40053006: 0x10},
        )
        self.assertEqual(handler.handle_read(0x40053006, 0x08000100, 1), 0x55)
        self.assertEqual(handler.mmio_seed_stats["applied"], 0)

    def test_unregistered_address_returns_zero_without_error(self):
        handler = StatefulMMIOHandler(mmio_seed_values={0x40053006: 0x10})
        self.assertEqual(handler.handle_read(0x40099998, 0x08000100, 4), 0)
        self.assertEqual(handler.handle_read(0x40099999, 0x08000200, 1), 0)
        self.assertEqual(handler.mmio_seed_stats["applied"], 0)


@unittest.skipUnless(Path(CONSOLE_ELF).exists(), "回归固件样本不存在")
class EmulatorSeedInjectionTests(unittest.TestCase):
    @staticmethod
    def _build_emulator(seed_bbs, accesses, **kwargs):
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

        return IntelligentEmulator(
            firmware_path=CONSOLE_ELF,
            mmio_constraints={},
            static_bbs=seed_bbs,
            static_mmio_accesses=accesses,
            constraint_json_path=None,
            max_snapshots=5,
            llm_config_path=None,
            **kwargs,
        )

    def test_seed_exported_and_injected_into_first_read(self):
        from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R3

        static_bbs = {
            0x5000: [
                {"address": 0x5000, "mnemonic": "LDRB", "operands": "r3, [r0, #6]", "size": 2},
                {"address": 0x5002, "mnemonic": "CMP", "operands": "r3, #0x10", "size": 2},
            ],
            0x6000: [
                {"address": 0x6000, "mnemonic": "LDRB", "operands": "r4, [r0, #8]", "size": 2},
            ],
        }
        accesses = [
            _read_record(0x5000, 0x5000, 0x40053006, "r3, [r0, #6]"),
            _read_record(0x6000, 0x6000, 0x40053008, "r4, [r0, #8]"),
        ]
        emulator = self._build_emulator(static_bbs, accesses)
        self.assertEqual(emulator.mmio_seed_values, {0x40053006: 0x10, 0x40053008: 0})
        self.assertEqual(emulator.mmio_handler.mmio_seed_values, emulator.mmio_seed_values)

        emulator.uc.reg_write(UC_ARM_REG_R0, 0x40053000)
        emulator.uc.reg_write(UC_ARM_REG_R3, 0)
        emulator.uc.reg_write(UC_ARM_REG_PC, 0x5000 | 1)
        emulator._mapped_mmio_preload_hook(emulator.uc, 0x5000, 2, None)
        self.assertEqual(emulator.uc.reg_read(UC_ARM_REG_R3) & 0xFF, 0x10)
        self.assertEqual(emulator.mmio_handler.mmio_seed_stats["applied"], 1)

        # 同地址第二次读取不再消耗种子（运行时路径接管）
        emulator.uc.reg_write(UC_ARM_REG_PC, 0x5000 | 1)
        emulator._mapped_mmio_preload_hook(emulator.uc, 0x5000, 2, None)
        self.assertEqual(emulator.mmio_handler.mmio_seed_stats["applied"], 1)

        stats = emulator._mmio_seed_report_stats()
        self.assertEqual(stats["table_size"], 2)
        self.assertEqual(stats["from_immediate"], 1)
        self.assertEqual(stats["applied_first_reads"], 1)

    def test_unregistered_address_read_returns_zero_at_emulator_level(self):
        from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R4

        static_bbs = {
            0x6000: [
                {"address": 0x6000, "mnemonic": "LDRB", "operands": "r4, [r0, #8]", "size": 2},
            ],
        }
        emulator = self._build_emulator(static_bbs, [])
        emulator.uc.reg_write(UC_ARM_REG_R0, 0x40053000)
        emulator.uc.reg_write(UC_ARM_REG_R4, 0x5A)
        emulator.uc.reg_write(UC_ARM_REG_PC, 0x6000 | 1)
        # 0x40053008 未登记：读值必须为 0 且不抛错
        emulator._mapped_mmio_preload_hook(emulator.uc, 0x6000, 2, None)
        self.assertEqual(emulator.uc.reg_read(UC_ARM_REG_R4) & 0xFF, 0)

    def test_env_switch_disables_seeds(self):
        static_bbs = {
            0x5000: [
                {"address": 0x5000, "mnemonic": "LDRB", "operands": "r3, [r0, #6]", "size": 2},
                {"address": 0x5002, "mnemonic": "CMP", "operands": "r3, #0x10", "size": 2},
            ],
        }
        accesses = [_read_record(0x5000, 0x5000, 0x40053006, "r3, [r0, #6]")]
        with patch.dict(os.environ, {"LSGEMU_STATIC_MMIO_SEEDS": "0"}):
            emulator = self._build_emulator(static_bbs, accesses)
        self.assertFalse(emulator.enable_static_mmio_seeds)
        self.assertEqual(emulator.mmio_seed_values, {})
        self.assertEqual(emulator.mmio_handler.mmio_seed_values, {})
        self.assertEqual(
            emulator.mmio_handler.handle_read(0x40053006, 0x5000, 1), 0
        )


if __name__ == "__main__":
    unittest.main()
