#!/usr/bin/env python3
"""Cortex-M CPU 型号自动匹配（cpu_profile）契约测试。

三层验证：
1. 纯逻辑层：MCU_NAME_TO_CPU 表命中、ELF e_flags 手工头解析、
   LSGEMU_FORCE_CPU_MODEL 覆盖、Thumb-2 前缀启发式计数；
2. unicorn 层：MCLASS + ctl_set_cpu_model 五档 Cortex-M 型号可切换并执行
   Thumb-2 片段（本机 unicorn 2.1.4 常量缺失/无 ctl API 时跳过）；
3. 集成层：IntelligentEmulator 构造时按固件名自动切到正确型号
   （F427→M4、H743→M7），LSGEMU_DISABLE_MCLASS=1 时 profile 为 None。

cpu_profile 本体不依赖 unicorn；本文件先 import lsgemu（包 __init__ 会
bootstrap unicorn 2.1.4 绑定）再取常量，保证与真实运行环境一致。
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

import lsgemu  # noqa: F401  # 触发包级 bootstrap（unicorn 2.1.4 绑定优先）
from lsgemu.cpu_profile import (
    CORTEX_M_FALLBACK_CONSTANT,
    MCU_NAME_TO_CPU,
    count_thumb2_halfword_prefixes,
    cpu_profile_to_dict,
    infer_cortex_m_cpu,
    is_armv6m_profile,
    parse_elf_arm_flags,
    pretty_cpu_name,
    resolve_cortex_m_cpu,
    unicorn_cpu_model_id,
)

# 真实固件回归样本（构建产物缺失时跳过对应用例）。
MCU_DATABASE = Path("/opt/artifact/benchmarks/drone_firmware")
F427_ELF = MCU_DATABASE / "ardupilot_Pixhawk1_STM32F427.elf"
H743_ELF = MCU_DATABASE / "ardupilot_MatekH743_STM32H743.elf"


def build_elf32(path: Path, e_flags: int, e_entry: int = 0x08000001) -> Path:
    """写一个最小 ELF32 小端文件头（e_flags/e_entry 可控）。

    cpu_profile 只读 52 字节文件头，这里不构造段表——够用即可。
    """
    header = bytearray(52)
    header[0:4] = b"\x7fELF"
    header[4] = 1  # ELFCLASS32
    header[5] = 1  # ELFDATA2LSB
    header[6] = 1  # EV_CURRENT
    struct.pack_into("<HH", header, 0x10, 2, 0x28)  # e_type=ET_EXEC, e_machine=EM_ARM
    struct.pack_into("<I", header, 0x18, e_entry)
    struct.pack_into("<I", header, 0x24, e_flags)
    path.write_bytes(bytes(header))
    return path


def build_elf32_with_segments(
    path: Path,
    segments: list[tuple[int, bytes, bool]],
    e_flags: int = 0x05000000,
) -> Path:
    """构造带 PT_LOAD 段表的最小 ELF32（供 Thumb-2 计数的可执行段过滤用例）。

    segments: (p_flags, 数据, 是否可执行)；段数据在文件中连续排布，
    p_offset 按序回填。
    """
    e_phoff = 52
    e_phentsize = 32
    e_phnum = len(segments)
    data_offset = e_phoff + e_phentsize * e_phnum
    headers = bytearray()
    body = b""
    offset = data_offset
    for p_flags, payload, executable in segments:
        flags = p_flags | (0x1 if executable else 0)  # PF_X
        entry = struct.pack(
            "<8I",
            1,  # PT_LOAD
            offset,  # p_offset
            0x08000000 + len(body),  # p_vaddr
            0x08000000 + len(body),  # p_paddr
            len(payload),  # p_filesz
            len(payload),  # p_memsz
            flags,
            0x1000,
        )
        headers += entry
        body += payload
        offset += len(payload)
    header = bytearray(52)
    header[0:4] = b"\x7fELF"
    header[4] = 1
    header[5] = 1
    header[6] = 1
    struct.pack_into("<HH", header, 0x10, 2, 0x28)  # e_type=ET_EXEC, e_machine=EM_ARM
    struct.pack_into("<I", header, 0x1C, e_phoff)
    struct.pack_into("<H", header, 0x2A, e_phentsize)
    struct.pack_into("<H", header, 0x2C, e_phnum)
    struct.pack_into("<I", header, 0x24, e_flags)
    path.write_bytes(bytes(header) + bytes(headers) + body)
    return path


class McuNameInferenceTests(unittest.TestCase):
    """MCU 表命中：固件名唯一确定型号（source=mcu_name）。"""

    def test_table_covers_required_mcu_families(self):
        # 任务约定的最小集合必须存在。
        for key, suffix in [
            ("STM32F4", "UC_CPU_ARM_CORTEX_M4"),
            ("STM32F7", "UC_CPU_ARM_CORTEX_M7"),
            ("STM32H7", "UC_CPU_ARM_CORTEX_M7"),
            ("STM32F1", "UC_CPU_ARM_CORTEX_M3"),
            ("STM32L0", "UC_CPU_ARM_CORTEX_M0"),
            ("STM32G0", "UC_CPU_ARM_CORTEX_M0"),
            ("RP2350", "UC_CPU_ARM_CORTEX_M33"),
            ("AT32F4", "UC_CPU_ARM_CORTEX_M4"),
        ]:
            self.assertEqual(MCU_NAME_TO_CPU.get(key), suffix, key)

    def test_infer_from_firmware_names(self):
        cases = [
            ("ardupilot_Pixhawk1_STM32F427.bin", "UC_CPU_ARM_CORTEX_M4"),
            ("ardupilot_MatekH743_STM32H743.elf", "UC_CPU_ARM_CORTEX_M7"),
            ("ardupilot_KakuteF7_STM32F745.elf", "UC_CPU_ARM_CORTEX_M7"),
            ("bluepill_STM32F103C8.bin", "UC_CPU_ARM_CORTEX_M3"),
            ("sensor_STM32L073RZ.bin", "UC_CPU_ARM_CORTEX_M0"),
            ("hub_STM32G0B1.bin", "UC_CPU_ARM_CORTEX_M0"),
            ("rp_rp2350_firmware.elf", "UC_CPU_ARM_CORTEX_M33"),
            ("vyton_AT32F435.bin", "UC_CPU_ARM_CORTEX_M4"),
            ("mesh_nRF52840.elf", "UC_CPU_ARM_CORTEX_M4"),
        ]
        for name, expected in cases:
            with self.subTest(name=name):
                profile = infer_cortex_m_cpu(f"/nonexistent/{name}")
                self.assertEqual(profile.unicorn_constant_name, expected)
                self.assertEqual(profile.source, "mcu_name")

    def test_longest_key_wins(self):
        # 长键优先：同前缀不同家族时不被短键抢先。
        profile = infer_cortex_m_cpu("/nonexistent/STM32F413.bin")
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M4")

    def test_firmware_name_argument_beats_path_name(self):
        profile = infer_cortex_m_cpu(
            "/nonexistent/generic.bin", firmware_name="STM32H743_custom"
        )
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M7")


class ElfFlagsInferenceTests(unittest.TestCase):
    """e_flags 族约束：无名称命中时取代表型号（source=elf_flags）。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_cpu_flags_")
        self.dir = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_parse_elf_arm_flags_reads_header(self):
        elf = build_elf32(self.dir / "a.elf", e_flags=0x05000400)
        info = parse_elf_arm_flags(elf)
        self.assertIsNotNone(info)
        self.assertEqual(info["e_flags"], 0x05000400)
        self.assertEqual(info["eabi_version"], 5)
        self.assertEqual(info["float_abi"], "hard")
        self.assertEqual(info["elf_class"], 32)

    def test_eabi5_hard_float_maps_to_m4_family_representative(self):
        elf = build_elf32(self.dir / "b.elf", e_flags=0x05000400)
        profile = infer_cortex_m_cpu(elf, firmware_name="mystery_image")
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M4")
        self.assertEqual(profile.source, "elf_flags")

    def test_eabi5_soft_or_no_float_maps_to_m3_family_representative(self):
        for e_flags in (0x05000200, 0x05000002, 0x05000000):
            with self.subTest(e_flags=hex(e_flags)):
                elf = build_elf32(self.dir / f"f{e_flags:x}.elf", e_flags=e_flags)
                profile = infer_cortex_m_cpu(elf, firmware_name="mystery_image")
                self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M3")
                self.assertEqual(profile.source, "elf_flags")

    def test_non_elf_or_legacy_flags_fall_back_to_m33(self):
        garbage = self.dir / "raw.bin"
        garbage.write_bytes(b"\x00" * 64)
        profile = infer_cortex_m_cpu(garbage)
        self.assertEqual(profile.unicorn_constant_name, CORTEX_M_FALLBACK_CONSTANT)
        self.assertEqual(profile.source, "fallback")

        legacy = build_elf32(self.dir / "legacy.elf", e_flags=0x00000000)
        profile = infer_cortex_m_cpu(legacy, firmware_name="unknown_vendor")
        self.assertEqual(profile.source, "fallback")

    def test_name_conflict_with_hard_float_keeps_name(self):
        # 名称说 M0（无 FPU）但 e_flags 标 hard-float：名称是更强信号，保留名称。
        elf = build_elf32(self.dir / "STM32G071.elf", e_flags=0x05000400)
        profile = infer_cortex_m_cpu(elf)
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M0")
        self.assertEqual(profile.source, "mcu_name")


class ForcedModelTests(unittest.TestCase):
    """LSGEMU_FORCE_CPU_MODEL 覆盖。"""

    def _resolve(self, value: str, elf="/nonexistent/STM32F407.bin"):
        with patch.dict(
            "os.environ", {"LSGEMU_FORCE_CPU_MODEL": value}, clear=False
        ):
            return resolve_cortex_m_cpu(elf)

    def test_force_accepts_common_spellings(self):
        for value, expected in [
            ("cortex-m7", "UC_CPU_ARM_CORTEX_M7"),
            ("M4", "UC_CPU_ARM_CORTEX_M4"),
            ("m33", "UC_CPU_ARM_CORTEX_M33"),
            ("UC_CPU_ARM_CORTEX_M3", "UC_CPU_ARM_CORTEX_M3"),
        ]:
            with self.subTest(value=value):
                profile = self._resolve(value)
                self.assertEqual(profile.unicorn_constant_name, expected)
                self.assertEqual(profile.source, "env_force")

    def test_force_garbage_is_ignored_with_name_fallback(self):
        profile = self._resolve("not-a-cpu", elf="/nonexistent/STM32F407.bin")
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M4")
        self.assertEqual(profile.source, "mcu_name")

    def test_no_env_uses_inference(self):
        environ = dict(os.environ)
        environ.pop("LSGEMU_FORCE_CPU_MODEL", None)
        with patch.dict("os.environ", environ, clear=True):
            profile = resolve_cortex_m_cpu("/nonexistent/STM32H743.bin")
        self.assertEqual(profile.source, "mcu_name")


class Thumb2PrefixCountTests(unittest.TestCase):
    """M0 语义提示的启发式计数。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_cpu_t2_")
        self.dir = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_counts_e8_plus_prefixes_in_raw_bin(self):
        raw = self.dir / "raw.bin"
        # e800/f000/f83x 前缀各一 + 普通半字，共 3 个疑似 32 位 Thumb 首半字。
        raw.write_bytes(
            b"\x00\xe8\x00\x00" + b"\x00\xf0\x00\x00" + b"\x3a\xf8\x00\x00" + b"\x01\x20\x00\x47"
        )
        self.assertEqual(count_thumb2_halfword_prefixes(raw), 3)

    def test_zero_prefixes_for_plain_thumb(self):
        raw = self.dir / "plain.bin"
        raw.write_bytes(b"\x01\x20" * 8)  # movs r0,#1 ...
        self.assertEqual(count_thumb2_halfword_prefixes(raw), 0)

    def test_elf_only_counts_executable_segments(self):
        # 可执行段 1 个前缀；只读数据段 4 个前缀不应计入。
        elf = build_elf32_with_segments(
            self.dir / "seg.elf",
            [
                (0x4, b"\x00\xe8\x00\x00", True),   # R+X
                (0x4, b"\x00\xe8" * 4, False),      # R 只读
            ],
        )
        self.assertEqual(count_thumb2_halfword_prefixes(elf), 1)


class ReportViewTests(unittest.TestCase):
    """cpu_profile_to_dict 报告视图。"""

    def test_none_profile_stays_null(self):
        self.assertIsNone(cpu_profile_to_dict(None))

    def test_dict_fields(self):
        profile = infer_cortex_m_cpu("/nonexistent/STM32F427.bin")
        payload = cpu_profile_to_dict(profile)
        self.assertEqual(payload["unicorn_constant_name"], "UC_CPU_ARM_CORTEX_M4")
        self.assertEqual(payload["cpu_model"], "cortex-m4")
        self.assertEqual(payload["source"], "mcu_name")
        self.assertIn("STM32F4", payload["reason"])
        # M0 档也应有干净的 pretty 名与 v6-M 判定。
        m0 = infer_cortex_m_cpu("/nonexistent/STM32G071.bin")
        self.assertEqual(pretty_cpu_name(m0.unicorn_constant_name), "cortex-m0")
        self.assertTrue(is_armv6m_profile(m0))
        self.assertFalse(is_armv6m_profile(profile))


class UnicornCpuModelSwitchTests(unittest.TestCase):
    """unicorn 层：MCLASS + ctl_set_cpu_model 切换与执行。

    本机 unicorn 2.1.4 提供 UC_CPU_ARM_CORTEX_* 常量与 ctl API；
    旧绑定（1.0.2 派生）缺失时整组跳过——IntelligentEmulator 会走
    fallback M33 路径，不影响正确性。
    """

    MODELS = [
        "UC_CPU_ARM_CORTEX_M0",
        "UC_CPU_ARM_CORTEX_M3",
        "UC_CPU_ARM_CORTEX_M4",
        "UC_CPU_ARM_CORTEX_M7",
        "UC_CPU_ARM_CORTEX_M33",
    ]

    def _make_uc(self, constant_name: str):
        from unicorn import UC_ARCH_ARM, UC_MODE_MCLASS, UC_MODE_THUMB, Uc

        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        uc.ctl_set_cpu_model(int(unicorn_cpu_model_id(constant_name)))
        uc.mem_map(0x08000000, 0x1000)
        uc.mem_map(0x20000000, 0x1000)
        return uc

    @unittest.skipIf(
        any(unicorn_cpu_model_id(n) is None for n in MODELS),
        "本机 unicorn 缺少 UC_CPU_ARM_CORTEX_* 常量",
    )
    def test_ctl_switch_all_models_and_execute_thumb2(self):
        from unicorn.arm_const import UC_ARM_REG_R2

        # movw r2, #0x134（Thumb-2）+ nop：五档 Cortex-M 均切换成功且执行到位。
        # 注：本机 qemu cortex-m0 对 Thumb-2 宽松（详见 cpu_profile 模块注释），
        # 因此不区分 v6-M 语义，只验证"可切换且可执行"。
        code = bytes.fromhex("40f23412") + b"\x00\xbf"
        for name in self.MODELS:
            with self.subTest(model=name):
                uc = self._make_uc(name)
                uc.mem_write(0x08000100, code)
                uc.emu_start(0x08000101, 0x08000100 + len(code), timeout=2_000_000)
                self.assertEqual(uc.reg_read(UC_ARM_REG_R2), 0x134)

    @unittest.skipIf(
        unicorn_cpu_model_id("UC_CPU_ARM_CORTEX_M4") is None,
        "本机 unicorn 缺少 UC_CPU_ARM_CORTEX_M4",
    )
    def test_emulator_applies_inferred_model(self):
        # 正常路径：STM32F427 命名文件 -> M4 (mcu_name)。
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

        with tempfile.TemporaryDirectory(prefix="lsgemu_cpu_emu_") as tmp:
            elf = build_elf32(Path(tmp) / "STM32F427_mini.elf", e_flags=0x05000400)
            emulator = IntelligentEmulator(firmware_path=str(elf))
            try:
                self.assertEqual(
                    emulator.cortex_m_cpu_profile.unicorn_constant_name,
                    "UC_CPU_ARM_CORTEX_M4",
                )
                self.assertEqual(emulator.cortex_m_cpu_profile.source, "mcu_name")
            finally:
                close = getattr(emulator, "close", None)
                if callable(close):
                    close()

    def test_emulator_falls_back_when_model_id_unresolvable(self):
        # 回退路径：当前绑定解析不出常量数值（模拟旧 unicorn 绑定）时，
        # _apply_cortex_m_cpu_profile 必须返回 fallback M33 而不是抛异常。
        from lsgemu.analysis import intelligent_emulator
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
        from lsgemu.cpu_profile import CpuProfile

        with tempfile.TemporaryDirectory(prefix="lsgemu_cpu_emu_") as tmp:
            elf = build_elf32(Path(tmp) / "STM32F427_mini.elf", e_flags=0x05000400)
            emulator = IntelligentEmulator(firmware_path=str(elf))
            try:
                stub = CpuProfile(
                    "UC_CPU_ARM_CORTEX_M4",
                    None,  # 模拟旧绑定：常量缺失
                    "mcu_name",
                    "测试桩：常量不可解析",
                )
                with patch.object(
                    intelligent_emulator, "resolve_cortex_m_cpu", return_value=stub
                ):
                    profile = emulator._apply_cortex_m_cpu_profile()
                self.assertEqual(
                    profile.unicorn_constant_name, CORTEX_M_FALLBACK_CONSTANT
                )
                self.assertEqual(profile.source, "fallback")
                self.assertIn("原推断 UC_CPU_ARM_CORTEX_M4", profile.reason)
            finally:
                close = getattr(emulator, "close", None)
                if callable(close):
                    close()

    def test_emulator_falls_back_when_ctl_raises(self):
        # 回退路径：ctl_set_cpu_model 抛异常（实例属性影子方法模拟）时回退 M33。
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

        with tempfile.TemporaryDirectory(prefix="lsgemu_cpu_emu_") as tmp:
            elf = build_elf32(Path(tmp) / "STM32F427_mini.elf", e_flags=0x05000400)
            emulator = IntelligentEmulator(firmware_path=str(elf))
            try:
                original = getattr(emulator.uc, "ctl_set_cpu_model", None)
                if original is None:
                    self.skipTest("本机 unicorn 缺少 ctl_set_cpu_model")

                def _boom(model_id):
                    raise RuntimeError("simulated ctl failure")

                emulator.uc.ctl_set_cpu_model = _boom
                try:
                    profile = emulator._apply_cortex_m_cpu_profile()
                finally:
                    emulator.uc.ctl_set_cpu_model = original
                self.assertEqual(
                    profile.unicorn_constant_name, CORTEX_M_FALLBACK_CONSTANT
                )
                self.assertEqual(profile.source, "fallback")
            finally:
                close = getattr(emulator, "close", None)
                if callable(close):
                    close()


class RealFirmwareEmulatorProfileTests(unittest.TestCase):
    """真实固件集成：F427→M4、H743→M7（IntelligentEmulator 构造期生效）。"""

    def _profile_for(self, elf_path: Path):
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

        emulator = IntelligentEmulator(firmware_path=str(elf_path))
        try:
            self.assertTrue(emulator.cortex_m_native_mclass_enabled)
            return emulator.cortex_m_cpu_profile
        finally:
            close = getattr(emulator, "close", None)
            if callable(close):
                close()

    @unittest.skipIf(not F427_ELF.is_file(), f"缺少回归固件 {F427_ELF}")
    def test_f427_infers_cortex_m4(self):
        profile = self._profile_for(F427_ELF)
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M4")
        self.assertEqual(profile.source, "mcu_name")

    @unittest.skipIf(not H743_ELF.is_file(), f"缺少回归固件 {H743_ELF}")
    def test_h743_infers_cortex_m7(self):
        profile = self._profile_for(H743_ELF)
        self.assertEqual(profile.unicorn_constant_name, "UC_CPU_ARM_CORTEX_M7")
        self.assertEqual(profile.source, "mcu_name")

    @unittest.skipIf(not F427_ELF.is_file(), f"缺少回归固件 {F427_ELF}")
    def test_disable_mclass_leaves_profile_null(self):
        from lsgemu.analysis.intelligent_emulator import IntelligentEmulator

        with patch.dict("os.environ", {"LSGEMU_DISABLE_MCLASS": "1"}, clear=False):
            emulator = IntelligentEmulator(firmware_path=str(F427_ELF))
        try:
            self.assertFalse(emulator.cortex_m_native_mclass_enabled)
            self.assertIsNone(emulator.cortex_m_cpu_profile)
        finally:
            close = getattr(emulator, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    unittest.main()
