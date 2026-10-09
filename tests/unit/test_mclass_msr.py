#!/usr/bin/env python3
"""UC_MODE_MCLASS（Cortex-M profile）系统指令原生执行的回归测试。

背景（根因，详见 intelligent_emulator 构造函数注释）：
- 不加 UC_MODE_MCLASS 时 unicorn 默认选 cortex-a15（无 ARM_FEATURE_M），
  v7-M 系统指令（msr msp/psp、mrs、cpsid 等）翻译失败 -> UC_ERR_INSN_INVALID。
- ardupilot_Pixhawk1(STM32F427) 的 reset 片段第 3 条指令就是 ``msr msp, r0``，
  旧配置下 baseline 在此死亡（instruction_count=2）。
- 加 UC_MODE_MCLASS 后 CPU 切成 cortex-m33，这些指令原生翻译执行。

本文件三层验证：
1. 纯 unicorn 层：MCLASS 下 msr/mrs/cpsid 执行且寄存器值正确；无 MCLASS 时
   msr 报 INSN_INVALID（把根因固化成回归断言）。
2. IntelligentEmulator 全 hook 环境：F427 真实 reset 片段（0x08004fb4 起 10 条）
   原生跑通 >=10 条指令。
3. 回退开关：LSGEMU_DISABLE_MCLASS=1 时回到 A15+软件 hook 模式（行为与旧版
   一致，软件模型仍工作）。
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from unicorn import (
    Uc,
    UcError,
    UC_ARCH_ARM,
    UC_HOOK_CODE,
    UC_MODE_ARM,
    UC_MODE_THUMB,
)
from unicorn.arm_const import (
    UC_ARM_REG_CPSR,
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_SP,
)

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0

# 与真实 ardupilot_Pixhawk1_STM32F427.bin 相同的布局：
# load base 0x08004000（reset 指针 0x08004fb5 按扇区对齐推出），向量表在文件
# 偏移 0，reset 片段在文件偏移 0xfb4，literal pool 在文件偏移 0x109c。
LOAD_BASE = 0x08004000
RESET_PC = 0x08004FB4
NEW_MSP = 0x20000600
NEW_PSP = 0x20002200
VTOR_VALUE = 0x08004000
VTOR_ADDR = 0xE000ED08

# 0x08004fb4 起的 10 条 reset 指令（字节逐条取自真实固件）：
#   b 0x08004fbc /（占位 bl、自旋 b .）/ cpsid i / ldr / msr msp / ldr /
#   msr psp / ldr / ldr / str r0,[r1]（写 SCB->VTOR）/ movw r0,#0
RESET_FRAGMENT = bytes.fromhex(
    "02e0"          # b 0x08004fbc
    "00f0" "00f8"   # bl 0x08004fba（占位，不会被走到）
    "e7fe"          # b .（自旋占位）
    "72b6"          # cpsid i
    "3748"          # ldr r0, [pc, #0xdc]  -> 0x0800509c = 0x20000600（新 MSP）
    "80f3" "0888"   # msr msp, r0
    "3648"          # ldr r0, [pc, #0xd8]  -> 0x080050a0 = 0x20002200（新 PSP）
    "80f3" "0988"   # msr psp, r0
    "3648"          # ldr r0, [pc, #0xd8]  -> 0x080050a4 = 0x08004000
    "3649"          # ldr r1, [pc, #0xd8]  -> 0x080050a8 = 0xE000ED08（VTOR）
    "0860"          # str r0, [r1]
    "40f2" "0000"   # movw r0, #0
)
FRAGMENT_INSTRUCTION_COUNT = 10


def build_f427_style_image(path: str) -> None:
    """按真实 F427 镜像布局合成一个最小 BIN。"""
    image = bytearray(0x2000)
    struct.pack_into("<I", image, 0x0, NEW_MSP)
    struct.pack_into("<I", image, 0x4, RESET_PC | 1)
    image[0xFB4:0xFB4 + len(RESET_FRAGMENT)] = RESET_FRAGMENT
    struct.pack_into("<I", image, 0x109C, NEW_MSP)     # 0x0800509c
    struct.pack_into("<I", image, 0x10A0, NEW_PSP)     # 0x080050a0
    struct.pack_into("<I", image, 0x10A4, VTOR_VALUE)  # 0x080050a4
    struct.pack_into("<I", image, 0x10A8, VTOR_ADDR)   # 0x080050a8
    Path(path).write_bytes(bytes(image))


def reset_fragment_static_bbs() -> dict:
    """软件 hook 回退模式所需的静态 BB（含 MSR/CPSID 条目才会被 map 收集）。"""
    instructions = [
        {"address": 0x08004FB4, "mnemonic": "B", "operands": "#0x8004fbc", "size": 2},
        {"address": 0x08004FBC, "mnemonic": "CPSID", "operands": "i", "size": 2},
        {"address": 0x08004FBE, "mnemonic": "LDR", "operands": "r0, [pc, #0xdc]", "size": 2},
        {"address": 0x08004FC0, "mnemonic": "MSR", "operands": "msp, r0", "size": 4},
        {"address": 0x08004FC4, "mnemonic": "LDR", "operands": "r0, [pc, #0xd8]", "size": 2},
        {"address": 0x08004FC6, "mnemonic": "MSR", "operands": "psp, r0", "size": 4},
        {"address": 0x08004FCA, "mnemonic": "LDR", "operands": "r0, [pc, #0xd8]", "size": 2},
        {"address": 0x08004FCC, "mnemonic": "LDR", "operands": "r1, [pc, #0xd8]", "size": 2},
        {"address": 0x08004FCE, "mnemonic": "STR", "operands": "r0, [r1]", "size": 2},
        {"address": 0x08004FD0, "mnemonic": "MOVW", "operands": "r0, #0", "size": 4},
    ]
    return {0x08004FB4: instructions}


@unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
class MclassNativeSystemInstructionTests(unittest.TestCase):
    """纯 unicorn 层：MCLASS 原生执行 v7-M 系统指令。"""

    def _make_uc(self) -> Uc:
        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        uc.mem_map(0x08000000, 0x1000)
        uc.mem_map(0x20000000, 0x1000)
        return uc

    def test_msr_msp_mrs_roundtrip(self):
        # msr msp, r0; mrs r1, msp; nop —— 编码取自真实 F427 固件（SYSm=0x88 变体）
        code = bytes.fromhex("80f3" "0888" "eff3" "0881" "00bf")
        uc = self._make_uc()
        uc.mem_write(0x08000100, code)
        uc.reg_write(UC_ARM_REG_R0, NEW_MSP)
        uc.emu_start(0x08000101, 0x08000100 + len(code), timeout=5_000_000)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R1), NEW_MSP)
        # 当前 CONTROL=0（handler 模式用 MSP），主栈指针应即时生效
        self.assertEqual(uc.reg_read(UC_ARM_REG_SP) & 0xFFFFFFFF, NEW_MSP)

    def test_msr_psp_banked_write(self):
        # msr psp, r0; mrs r1, psp —— PSP 是 banked 寄存器，不改变当前 SP
        code = bytes.fromhex("80f3" "0988" "eff3" "0981" "00bf")
        uc = self._make_uc()
        uc.mem_write(0x08000100, code)
        uc.reg_write(UC_ARM_REG_R0, NEW_PSP)
        uc.reg_write(UC_ARM_REG_SP, NEW_MSP)
        uc.emu_start(0x08000101, 0x08000100 + len(code), timeout=5_000_000)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R1), NEW_PSP)
        self.assertEqual(uc.reg_read(UC_ARM_REG_SP) & 0xFFFFFFFF, NEW_MSP)

    def test_cpsid_sets_primask(self):
        code = bytes.fromhex("72b6" "00bf")
        uc = self._make_uc()
        uc.mem_write(0x08000100, code)
        uc.emu_start(0x08000101, 0x08000100 + len(code), timeout=5_000_000)
        cpsr = uc.reg_read(UC_ARM_REG_CPSR) & 0xFFFFFFFF
        self.assertEqual(cpsr & 0x1, 0x1, f"PRIMASK 未置位: cpsr=0x{cpsr:08x}")

    def test_without_mclass_msr_is_insn_invalid(self):
        # 根因固化：cortex-a15（默认 CPU）上 MSR(M-profile) 翻译失败。
        # RAM 已映射，排除 WRITE_UNMAPPED 干扰，失败必须是 INSN_INVALID。
        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        uc.mem_map(0x08000000, 0x1000)
        uc.mem_map(0x20000000, 0x1000)
        code = bytes.fromhex("80f3" "0888" "00bf")
        uc.mem_write(0x08000100, code)
        uc.reg_write(UC_ARM_REG_R0, NEW_MSP)
        with self.assertRaises(UcError) as ctx:
            uc.emu_start(0x08000101, 0x08000100 + len(code), timeout=5_000_000)
        self.assertIn("UC_ERR_INSN_INVALID", str(ctx.exception))

    def test_arm_mode_not_gated_by_mclass(self):
        # cortex-m33 只支持 Thumb：A32（UC_MODE_ARM）不能加 MCLASS 位，
        # 否则普通 ARM 指令也 INSN_INVALID。IntelligentEmulator 只对 Thumb
        # 固件启用 MCLASS，这里直接固化 unicorn 层的原因。
        uc = Uc(UC_ARCH_ARM, UC_MODE_ARM | UC_MODE_MCLASS)
        uc.mem_map(0x08000000, 0x1000)
        # mov r0, #1 (e3a00001)
        uc.mem_write(0x08000100, bytes.fromhex("e3a00001"))
        with self.assertRaises(UcError) as ctx:
            uc.emu_start(0x08000100, 0x08000104, timeout=5_000_000)
        self.assertIn("UC_ERR_INSN_INVALID", str(ctx.exception))


class MclassEmulatorResetFragmentTests(unittest.TestCase):
    """IntelligentEmulator 全 hook 环境下执行 F427 reset 片段。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_mclass_test_")
        self.firmware_path = str(Path(self._temporary.name) / "f427_reset.bin")
        build_f427_style_image(self.firmware_path)

    def tearDown(self):
        self._temporary.cleanup()

    def _run_fragment(self, emulator) -> list:
        executed: list = []

        def count_hook(uc, address, size, user_data):
            executed.append(int(address))
            # movw（0x08004fd0）执行完之后的下一条地址处停止：
            # UC_HOOK_CODE 在指令执行前触发，阈值必须严格大于 movw 本身，
            # 否则 movw 不会真正执行；同时避免落入 0x08004fba 的占位自旋。
            if int(address) > 0x08004FD0:
                uc.emu_stop()

        hook = emulator.uc.hook_add(UC_HOOK_CODE, count_hook)
        try:
            emulator.run(entry_point=RESET_PC, max_instructions=1000)
        finally:
            emulator.uc.hook_del(hook)
        return executed

    def _new_emulator(self, **kwargs) -> IntelligentEmulator:
        emulator = IntelligentEmulator(
            firmware_path=self.firmware_path,
            execution_thumb_override=True,
            raw_load_base=LOAD_BASE,
            **kwargs,
        )
        emulator.setup_memory()
        emulator.load_firmware()
        emulator.register_hooks()
        return emulator

    @unittest.skipIf(UC_MODE_MCLASS == 0, "本机 unicorn 缺少 UC_MODE_MCLASS")
    def test_native_mclass_runs_reset_fragment(self):
        emulator = self._new_emulator()
        try:
            self.assertTrue(emulator.cortex_m_native_mclass_enabled)
            executed = self._run_fragment(emulator)
            # 核心断言：reset 片段 10 条指令全部原生执行（旧配置在第 3 条
            # msr msp 处 INSN_INVALID，只能执行 2 条）
            self.assertGreaterEqual(
                len(executed),
                FRAGMENT_INSTRUCTION_COUNT,
                f"执行指令数不足: {[hex(a) for a in executed]}",
            )
            self.assertIn(0x08004FC0, executed, "msr msp 未被执行")
            self.assertIn(0x08004FC6, executed, "msr psp 未被执行")
            # msr msp 真正写入了主栈指针
            self.assertEqual(
                emulator.uc.reg_read(UC_ARM_REG_SP) & 0xFFFFFFFF, NEW_MSP
            )
            # ldr/str 链条工作：r1 = VTOR 地址，movw r0,#0 已执行
            self.assertEqual(emulator.uc.reg_read(UC_ARM_REG_R1), VTOR_ADDR)
            self.assertEqual(emulator.uc.reg_read(UC_ARM_REG_R0) & 0xFFFFFFFF, 0)
            # 原生模式下软件 hook 必须未注册：cortex_m_system_registers 保持空，
            # 否则就是"原生执行 + hook 再改一遍"的双重写。
            self.assertEqual(emulator.cortex_m_system_registers, {})
        finally:
            emulator.close()

    def test_disable_mclass_env_falls_back_to_software_hook(self):
        emulator = None
        try:
            with patch.dict(os.environ, {"LSGEMU_DISABLE_MCLASS": "1"}):
                emulator = self._new_emulator(
                    static_bbs=reset_fragment_static_bbs()
                )
                self.assertFalse(emulator.cortex_m_native_mclass_enabled)
                executed = self._run_fragment(emulator)
                self.assertGreaterEqual(
                    len(executed),
                    FRAGMENT_INSTRUCTION_COUNT,
                    f"回退模式下执行指令数不足: {[hex(a) for a in executed]}",
                )
                # 软件 hook 模式：MSR/CPSID 的值必须记录在软件模型里
                self.assertEqual(
                    emulator.cortex_m_system_registers.get("msp"), NEW_MSP
                )
                self.assertEqual(
                    emulator.cortex_m_system_registers.get("psp"), NEW_PSP
                )
                self.assertEqual(
                    emulator.cortex_m_system_registers.get("primask"), 1
                )
        finally:
            if emulator is not None:
                emulator.close()

    def test_arm_state_firmware_never_enables_mclass(self):
        # raw ARM BIN（execution_thumb=False）不能启用 MCLASS：
        # cortex-m33 无法执行 A32 指令。
        emulator = None
        try:
            emulator = IntelligentEmulator(
                firmware_path=self.firmware_path,
                execution_thumb_override=False,
                raw_load_base=LOAD_BASE,
            )
            self.assertFalse(emulator.execution_thumb)
            self.assertFalse(emulator.cortex_m_native_mclass_enabled)
        finally:
            if emulator is not None:
                emulator.close()


if __name__ == "__main__":
    unittest.main()
