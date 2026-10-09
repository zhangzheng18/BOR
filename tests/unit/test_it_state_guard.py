#!/usr/bin/env python3
"""r29: 脏 ITSTATE 守卫（站点无关）——分类器单测 + 反例臂行为单测。

背景见 ``docs/GDB_DEBUG_R28_20260926.md`` 与 ``lsgemu/analysis/thumb_it_state.py``。

被测生产代码
------------
* ``thumb_it_unpredictable_kind`` —— 纯分类器（只吃两个半字，无地址白名单）；
* ``IntelligentEmulator._it_state_guard_check`` / ``_repair_stale_itstate``
  以及 ``_managed_emu_start`` 里的"清 IT + 原地重入"通路。

关于反例臂的可复现边界（重要，勿当成测试偷懒）
----------------------------------------------
把 ``xPSR`` 的 IT 位直接写成非 0，能确定性复现 r28 的**单条指令被跳过**：
unconditional ``b`` / ``pop {pc}`` 会被当作条件指令、条件为假时整条跳过。
夹具用的 IT 字段 ``0x4800`` 正是 r28 在 ``0x0806D3B0`` 实测到的出错调用
``xpsr=0x81004800``；``0x0400`` 是实测能让 ``pop {pc}`` 被跳过的取值。

但 r28 那道墙的**成对**签名（``b`` 先被执行、紧随其后的 ``pop`` 才被跳过）**无法**
由静态寄存器注入复现：本文件外的穷举扫描（IT 字段 256 种 × NZCV 16 种 = 4096 个
状态）里，"``b`` 执行 ∧ ``pop`` 被跳过" 出现 **0** 次。这与 r28 §1.1/§1.3 的结论
一致——**该缺陷是 TB 翻译上下文（跨 TB 的 condexec 簿记）的产物，不是静态状态**。
因此本文件不伪造那条路径，而是：

* 用单条指令的被跳过签名证明"旧行为确实错误"（``反例臂``）；
* 用生产方法 ``_repair_stale_itstate`` + 守卫计数器证明"修复通路确实生效、
  且只动 IT 位"（``守卫契约``）；
* 端到端修复效果由 r29 的重放实验（covered 877 → 1447）承担，不在此单测里伪造。
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  （绑定补丁必须先于 Uc 构造）
from unicorn import UC_HOOK_CODE
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_SP, UC_ARM_REG_XPSR

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.thumb_it_state import (
    IT_STATE_MASK,
    thumb_it_unpredictable_kind,
)

LOAD_BASE = 0x08004000
B_ENTRY_PC = 0x08004100        # 受检对象：T1 无条件 b
POP_ENTRY_PC = 0x08004140
POP_INSN_PC = 0x08004142       # 受检对象：pop {pc}
POP_RETURN_PC = 0x08004160     # pop 真正执行时返回到这里
STACK_TOP = 0x20000800

# r28 在 0x0806D3B0 实测到的出错调用 xpsr=0x81004800 ⇒ IT 字段 0x4800。
DIRTY_XPSR_B_SKIPPED = 0x01004800
# 实测能让独立 pop{pc} 被 IT 跳过的取值（IT 字段 0x0400）。
DIRTY_XPSR_POP_SKIPPED = 0x01000400
# IT 位非 0，且残留条件**为真** ⇒ 指令照常执行、code hook 会触发，守卫才有机会动手。
# 0x81004800 就是 r28 在 0x0806D3B0 抓到的那次出错调用的原值：
# xPSR[15:12]=4 (MI)、N=1 ⇒ MI 为真 ⇒ b 被执行；紧随其后的 pop 才被残留条件吃掉。
DIRTY_XPSR_B_FIRES = 0x81004800
# pop 臂同样需要一个"条件为真、hook 能触发"的脏取值。
DIRTY_XPSR_POP_FIRES = 0x01001000
# 干净状态：IT 位全 0。
CLEAN_XPSR = 0x01000000

NOP = 0xBF00
# 地址 -> 半字；未列出的位置填 NOP。
_HALFWORDS = {
    # --- b 臂 ---
    0x08004100: 0xE006,   # b 0x08004110    <- 受检对象（T1 无条件 b）
    0x08004102: 0x2011,   # movs r0, #0x11  标记 A：只有 b 被跳过才会执行
    0x08004104: 0xE00C,   # b 0x08004120
    0x08004110: 0x2022,   # movs r0, #0x22  标记 B：b 真的执行才会到达
    0x08004112: 0xE005,   # b 0x08004120
    0x08004120: 0xE7FE,   # b .             停
    0x08004130: 0xD000,   # beq 0x08004134  条件分支：IT 块内合法，守卫不得动它
    # --- pop{pc} 臂 ---
    0x08004140: 0x2033,   # movs r0, #0x33
    0x08004142: 0xBD00,   # pop {pc}        <- 受检对象
    0x08004144: 0x2044,   # movs r0, #0x44  标记 A'：pop 被跳过才会执行
    0x08004146: 0xE003,   # b 0x08004150
    0x08004150: 0xE7FE,   # b .             停 A'
    0x08004160: 0x2055,   # movs r0, #0x55  标记 B'：pop 真执行才会到达
    0x08004162: 0xE7FE,   # b .             停 B'
}
_FRAGMENT_SPAN = 0x68


def _assemble_fragment() -> bytes:
    blob = bytearray()
    for offset in range(0, _FRAGMENT_SPAN, 2):
        blob += struct.pack("<H", _HALFWORDS.get(B_ENTRY_PC + offset, NOP))
    return bytes(blob)


FRAGMENT = _assemble_fragment()


def build_counterexample_image(path: str) -> None:
    """最小裸镜像：向量表在 LOAD_BASE，代码片段在偏移 0x100。"""
    image = bytearray(0x400)
    struct.pack_into("<I", image, 0x0, STACK_TOP)            # MSP
    struct.pack_into("<I", image, 0x4, B_ENTRY_PC | 1)       # reset -> entry
    at = B_ENTRY_PC - LOAD_BASE
    image[at:at + len(FRAGMENT)] = FRAGMENT
    Path(path).write_bytes(bytes(image))


class ThumbItUnpredictableKindTests(unittest.TestCase):
    """纯分类器：只按编码判定，与地址无关。"""

    def test_unconditional_branch_family_matches(self):
        cases = [
            (0xE006, None, "B"),        # T1 b
            (0xE7FE, None, "B"),        # T1 b .
            (0xF000, 0x9000, "B.W"),    # B.W T4
            (0xF3FF, 0x9FFF, "B.W"),    # B.W T4 边界立即数
            (0x4700, None, "BX"),       # bx r0
            (0x4750, None, "BX"),       # bx r10（Rm 在 [6:3]）
            (0x4780, None, "BLX"),      # blx r0
            (0xBD00, None, "POP{pc}"),  # pop {pc}
            (0xBD10, None, "POP{pc}"),  # pop {r4, pc}
            (0xE8BD, 0x8000, "POP.W{pc}"),
            (0xF000, 0xC000, "BLX.W"),  # blx <imm>
        ]
        for hw1, hw2, expected in cases:
            with self.subTest(hw1=hex(hw1), hw2=hw2):
                self.assertEqual(expected, thumb_it_unpredictable_kind(hw1, hw2))

    def test_conditional_and_other_instructions_are_left_alone(self):
        cases = [
            (0xD001, None, "T2 b<cond> 在 IT 块内是合法条件分支"),
            (0xBC10, None, "pop {r4} 不含 pc，IT 块内合法"),
            (0xF000, 0x8000, "B.W T3 条件分支，IT 块内合法"),
            (0xF000, 0xD000, "BL 是调用，明确排除在守卫集合外"),
            (0xE8BD, 0x0000, "pop.w 不含 pc"),
            (0x4751, None, "bit[2:0] 非 0 ⇒ 不是合法 BX 编码"),
            (0xBF00, None, "nop"),
            (0x2011, None, "movs r0, #0x11"),
        ]
        for hw1, hw2, why in cases:
            with self.subTest(hw1=hex(hw1), hw2=hw2):
                self.assertIsNone(thumb_it_unpredictable_kind(hw1, hw2), why)

    def test_32bit_candidates_need_second_halfword(self):
        # 只给第一个半字时不能猜 32 位编码，必须报"不匹配"。
        self.assertIsNone(thumb_it_unpredictable_kind(0xF000, None))
        self.assertIsNone(thumb_it_unpredictable_kind(0xE8BD, None))

    def test_it_state_mask_isolates_it_bits_only(self):
        # 掩码必须正好覆盖 xPSR[26:25] 与 xPSR[15:10]（= condexec_bits 的全部 8 位）：
        # 不得碰 NZCV[31:28]、不得碰 Thumb 位[5]、不得碰 ISR 号[8:0]。
        # 依据 qemu/target/arm/cpu.h: xpsr_read 把 condexec_bits[1:0] 放到 xPSR[26:25]、
        # condexec_bits[7:2] 放到 xPSR[15:10]；r28 的 0x0400FC00 漏了 bit 25。
        self.assertEqual(0x0600FC00, IT_STATE_MASK)
        self.assertEqual(0x06000000, IT_STATE_MASK & 0x06000000)  # [26:25] 全要
        self.assertEqual(0, IT_STATE_MASK & 0xF0000000)   # NZCV
        self.assertEqual(0, IT_STATE_MASK & 0x00000020)   # Thumb 位
        self.assertEqual(0, IT_STATE_MASK & 0x000001FF)   # IPSR/异常号
        for dirty in (DIRTY_XPSR_B_SKIPPED, DIRTY_XPSR_B_FIRES, DIRTY_XPSR_POP_FIRES):
            self.assertNotEqual(0, dirty & IT_STATE_MASK)
        self.assertEqual(0, CLEAN_XPSR & IT_STATE_MASK)
        # condexec_bits 的 bit0（xPSR bit 25）单独置位也必须算"ITSTATE 非 0"。
        self.assertNotEqual(0, (1 << 25) & IT_STATE_MASK)


class _ItGuardFixture(unittest.TestCase):
    """公共装置：最小裸镜像 + 直驱 ``_managed_emu_start``。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.firmware_path = str(
            Path(self._temporary.name) / "it_guard_counterexample.bin"
        )
        build_counterexample_image(self.firmware_path)

    def tearDown(self):
        self._temporary.cleanup()

    def _new_emulator(self, guard_enabled: bool) -> IntelligentEmulator:
        emulator = IntelligentEmulator(
            firmware_path=self.firmware_path,
            mmio_constraints={},
            static_bbs={},
            constraint_json_path=None,
            max_snapshots=1,
            llm_config_path=None,
            execution_thumb_override=True,
            raw_load_base=LOAD_BASE,
        )
        emulator.setup_memory()
        emulator.load_firmware()
        emulator.register_hooks()
        emulator.it_state_guard_enabled = guard_enabled
        return emulator

    def _run_from(self, emulator, entry: int, xpsr: int) -> int:
        uc = emulator.uc
        uc.reg_write(UC_ARM_REG_SP, STACK_TOP)
        uc.reg_write(UC_ARM_REG_PC, entry | 1)
        uc.reg_write(UC_ARM_REG_XPSR, xpsr)
        emulator._managed_emu_start(entry | 1, 0, 0, 400)
        return int(uc.reg_read(UC_ARM_REG_R0))

    def _run_pop_arm(self, emulator, xpsr: int) -> int:
        # pop{pc} 的落点由栈上返回地址决定。
        emulator.uc.mem_write(STACK_TOP, struct.pack("<I", POP_RETURN_PC | 1))
        return self._run_from(emulator, POP_ENTRY_PC, xpsr)


class StaleItStateCounterexampleTests(_ItGuardFixture):
    """反例臂：证明脏 ITSTATE 下旧行为确实跳过指令（守卫关闭）。"""

    def test_dirty_itstate_skips_unconditional_b(self):
        """r28 那道墙的指令级签名：无条件 b 被 IT 条件吃掉、整条跳过。"""
        emulator = self._new_emulator(guard_enabled=False)
        r0 = self._run_from(emulator, B_ENTRY_PC, DIRTY_XPSR_B_SKIPPED)
        self.assertEqual(
            0x11,
            r0,
            "脏 ITSTATE(0x4800) 下 T1 无条件 b 应被跳过（r28 的签名）",
        )
        self.assertEqual(0, emulator.it_state_guard_stats["trips"])

    def test_dirty_itstate_skips_pop_pc(self):
        """同样机制作用在 pop {pc} 上：整条跳过 ⇒ SP 不回收（r28 的墙）。"""
        emulator = self._new_emulator(guard_enabled=False)
        r0 = self._run_pop_arm(emulator, DIRTY_XPSR_POP_SKIPPED)
        self.assertEqual(
            0x44, r0, "脏 ITSTATE(0x0400) 下 pop {pc} 应被跳过"
        )

    def test_clean_itstate_executes_normally(self):
        """对照：IT 位为 0 时两条指令都按自身编码执行。"""
        emulator = self._new_emulator(guard_enabled=False)
        self.assertEqual(0x22, self._run_from(emulator, B_ENTRY_PC, CLEAN_XPSR))
        self.assertEqual(0x55, self._run_pop_arm(emulator, CLEAN_XPSR))


class ItStateGuardContractTests(_ItGuardFixture):
    """守卫契约：只在「ITSTATE 非 0 ∧ 无条件族指令」触发，且只清 IT 位。

    这里直接调用生产方法 ``_it_state_guard_check(uc, pc)``，而不是靠 hook 触发：
    QEMU 在 **TB 翻译时**就把 ``env->condexec_bits`` 清零
    （``translate.c`` 的 "Reset the conditional execution bits immediately"），
    所以在 hook 里读 ``xPSR`` 看不到本 TB 自己消费掉的 IT 状态。TB 之外直接调用
    才能确定性地把"脏 ITSTATE 落在某条指令上"这个前提摆好。
    """

    def _plant_dirty_itstate(self, emulator, xpsr: int):
        uc = emulator.uc
        uc.reg_write(UC_ARM_REG_XPSR, xpsr)
        return uc

    def test_guard_trips_on_unconditional_b_and_requests_stop(self):
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, DIRTY_XPSR_B_FIRES)
        stops = []
        original_stop = uc.emu_stop
        uc.emu_stop = lambda: (stops.append(1), original_stop())[1]

        emulator._it_state_guard_check(uc, B_ENTRY_PC)

        self.assertEqual(1, emulator.it_state_guard_stats["trips"])
        self.assertEqual(1, emulator.it_state_guard_stats["checked"])
        self.assertEqual(1, len(stops), "触发后必须打断 TB（emu_stop）")
        self.assertTrue(emulator._it_state_guard_pending)
        self.assertEqual(
            1, emulator._it_state_guard_sites[B_ENTRY_PC]
        )
        self.assertEqual(1, emulator._it_state_guard_kinds["B"])

    def test_guard_is_site_independent(self):
        """同一守卫在另一个地址上的 pop{pc} 也必须触发（没有地址白名单）。"""
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, DIRTY_XPSR_POP_FIRES)
        emulator._it_state_guard_check(uc, POP_INSN_PC)
        self.assertEqual(1, emulator.it_state_guard_stats["trips"])
        self.assertEqual(1, emulator._it_state_guard_kinds["POP{pc}"])
        self.assertIn(POP_INSN_PC, emulator._it_state_guard_sites)

    def test_guard_ignores_conditional_branch(self):
        """条件分支在 IT 块内合法 ⇒ 即使 ITSTATE 非 0 也不得触发。"""
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, DIRTY_XPSR_B_FIRES)
        emulator._it_state_guard_check(uc, 0x08004130)   # beq 0x08004134
        self.assertEqual(1, emulator.it_state_guard_stats["checked"])
        self.assertEqual(0, emulator.it_state_guard_stats["trips"])
        # 对照：同一个 ITSTATE 下，0x08004100 的无条件 b 必须触发。
        emulator._it_state_guard_check(uc, B_ENTRY_PC)
        self.assertEqual(1, emulator.it_state_guard_stats["trips"])

    def test_guard_ignores_non_family_instruction(self):
        """movs 在 ITSTATE 非 0 时也不得触发（不越界干预）。"""
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, DIRTY_XPSR_B_FIRES)
        emulator._it_state_guard_check(uc, B_ENTRY_PC + 2)   # movs r0, #0x11
        # checked 计的是"ITSTATE 非 0 的取样次数"，movs 会被取样但不该触发。
        self.assertEqual(1, emulator.it_state_guard_stats["checked"])
        self.assertEqual(0, emulator.it_state_guard_stats["trips"])
        self.assertFalse(emulator._it_state_guard_pending)

    def test_guard_ignores_clean_itstate(self):
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, CLEAN_XPSR)
        emulator._it_state_guard_check(uc, B_ENTRY_PC)
        self.assertEqual(0, emulator.it_state_guard_stats["trips"])
        self.assertEqual(0, emulator.it_state_guard_stats["checked"])

    def test_guard_disabled_never_breaks_translation_block(self):
        emulator = self._new_emulator(guard_enabled=False)
        uc = self._plant_dirty_itstate(emulator, DIRTY_XPSR_B_FIRES)
        emulator._it_state_guard_check(uc, B_ENTRY_PC)
        self.assertEqual(0, emulator.it_state_guard_stats["trips"])
        self.assertFalse(emulator._it_state_guard_pending)

    def test_repair_touches_only_it_bits(self):
        """``_repair_stale_itstate`` 只能清 IT 位，其余 xPSR 位原样保留。"""
        emulator = self._new_emulator(guard_enabled=True)
        uc = self._plant_dirty_itstate(emulator, 0xF1004800)  # NZCV=0xF 且 IT 非 0
        emulator._it_state_guard_pending = True
        self.assertTrue(emulator._repair_stale_itstate())
        after = int(uc.reg_read(UC_ARM_REG_XPSR)) & 0xFFFFFFFF
        self.assertEqual(0, after & IT_STATE_MASK, "IT 位必须被清空")
        self.assertEqual(
            0xF0000000, after & 0xF0000000, "NZCV 不得被守卫改动"
        )
        self.assertEqual(0, after & 0x000001FF, "IPSR/异常号不得被改动")
        self.assertEqual(1, emulator.it_state_guard_stats["repaired"])
        # 旗标是一次性的，不能被复用（否则会重复重入）。
        self.assertFalse(emulator._repair_stale_itstate())
        self.assertEqual(1, emulator.it_state_guard_stats["repaired"])

    def test_translator_zeroes_itstate_at_tb_entry(self):
        """记录机制：TB 翻译时 env->condexec_bits 被清零，hook 里读不到它。

        这是 QEMU ``translate.c`` 的 "Reset the conditional execution bits
        immediately" 契约，也是"脏 ITSTATE 只在 TB 缓存边界上出现"的直接原因。
        断言它是为了把这个前提钉死：谁要改守卫的取样位置，先看这个测试。
        """
        emulator = self._new_emulator(guard_enabled=False)
        uc = emulator.uc
        uc.reg_write(UC_ARM_REG_XPSR, DIRTY_XPSR_B_FIRES)
        self.assertNotEqual(0, int(uc.reg_read(UC_ARM_REG_XPSR)) & IT_STATE_MASK)
        seen = []

        def probe(u, address, size, user_data):
            seen.append(int(u.reg_read(UC_ARM_REG_XPSR)) & IT_STATE_MASK)

        handle = uc.hook_add(UC_HOOK_CODE, probe)
        try:
            self._run_from(emulator, B_ENTRY_PC, DIRTY_XPSR_B_FIRES)
        finally:
            uc.hook_del(handle)
        self.assertTrue(seen, "片段必须执行到至少一条指令")
        self.assertEqual(
            [0] * len(seen),
            seen,
            "TB 内 hook 看到的 IT 位必须恒为 0（翻译期已消费）",
        )


if __name__ == "__main__":
    unittest.main()
