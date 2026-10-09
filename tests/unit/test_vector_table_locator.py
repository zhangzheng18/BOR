#!/usr/bin/env python3
"""向量表自动定位（vector_table_locator）与 BIN 自动布局契约。"""

from __future__ import annotations

import random
import struct
import tempfile
import unittest
from pathlib import Path

from lsgemu.vector_table_locator import (
    CONFIDENT_MIN_SCORE,
    RAM_WINDOWS,
    VectorTableCandidate,
    scan_vector_table,
)

# 真实固件回归样本（构建产物缺失时跳过对应用例）。
FIRMWARE_BUILT = Path("/opt/artifact/benchmarks/drone_firmware")
F427_BIN = FIRMWARE_BUILT / "ardupilot_Pixhawk1_STM32F427.bin"
F427_ELF = FIRMWARE_BUILT / "ardupilot_Pixhawk1_STM32F427.elf"


def synthetic_image(
    vector_offset: int,
    *,
    load_base: int = 0x08000000,
    initial_sp: int = 0x20002000,
    tail_size: int = 0x2000,
    reserved_zero_slots: range | None = None,
) -> bytes:
    """构造一张带合法 Cortex-M 向量表的合成镜像。

    向量表放在 ``vector_offset``，Reset/异常向量指向表后紧跟的 Thumb
    代码区（与真实链接布局一致），前导区域填随机非零字节干扰。
    """
    rng = random.Random(0xC0FFEE + vector_offset)
    image = bytearray(vector_offset + tail_size)
    for index in range(len(image) & ~3):
        image[index] = rng.randrange(1, 256)

    reset_pc = (load_base + vector_offset + 0x100) | 1
    words = [initial_sp, reset_pc]
    for slot in range(14):
        if reserved_zero_slots is not None and slot in reserved_zero_slots:
            words.append(0)
        else:
            words.append((load_base + vector_offset + 0x200 + slot * 8) | 1)
    image[vector_offset:vector_offset + 64] = struct.pack("<16I", *words)
    return bytes(image)


def elf_header_bytes() -> bytes:
    """52 字节 ELF 头（魔数 + EI_CLASS/EI_DATA/EI_VERSION 特征）。"""
    header = bytearray(52)
    header[0:4] = b"\x7fELF"
    header[4] = 1  # EI_CLASS = ELFCLASS32
    header[5] = 1  # EI_DATA = little
    header[6] = 1  # EI_VERSION
    header[16:20] = struct.pack("<H H", 3, 0x28)  # e_type=EXEC, e_machine=ARM
    return bytes(header)


class VectorTableLocatorTests(unittest.TestCase):
    def test_synthetic_vector_table_at_common_offsets(self):
        # 已知基址（load_base_hint）时逐一验证：常见偏移的表都能定位。
        for offset in (0, 0x4000, 0x10000):
            with self.subTest(offset=hex(offset)):
                image = synthetic_image(offset)
                candidates = scan_vector_table(image, load_base_hint=0x08000000)
                self.assertTrue(candidates)
                top = candidates[0]
                self.assertEqual(offset, top.offset)
                self.assertEqual(0x20002000, top.initial_sp)
                self.assertEqual(0x08000000, top.load_base)
                self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)
                self.assertTrue(top.confident)
                self.assertEqual(offset + 0x08000000, top.vector_table_address)
                self.assertIn("sp_ram_window", top.signals)
                self.assertIn("reset_thumb_in_image", top.signals)

    def test_application_image_prefers_sector_aligned_base(self):
        # 未知基址时，Reset 指针的扇区对齐基址（紧凑解释）参与排序并胜出：
        # 真实场景如 bootloader 之后的应用镜像（ardupilot 表@0x08004000）。
        image = synthetic_image(0, load_base=0x08004000)
        candidates = scan_vector_table(image)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(0, top.offset)
        self.assertEqual(0x08004000, top.load_base)
        self.assertEqual(0x08004101, top.reset_pc)
        self.assertEqual(0x20002000, top.initial_sp)
        self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)

    def test_load_base_hint_pins_reported_base(self):
        # hint 语义是"只验证该基址"：候选的 load_base 全部等于 hint。
        image = synthetic_image(0x4000)
        candidates = scan_vector_table(image, load_base_hint=0x08000000)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(0x08000000, candidate.load_base)
        top = candidates[0]
        # 表位置本身仍正确定位，只是地址解释随 hint 平移。
        self.assertEqual(0x4000, top.offset)
        self.assertEqual(0x08004000, top.vector_table_address)
        self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)

    def test_elf_header_pollution_is_rejected(self):
        # 事故重演：PT_LOAD 把 ELF 头映射进 flash，真实向量表在 0x4000。
        image = elf_header_bytes() + bytearray(b"\x5a" * (0x4000 - 52))
        image = bytes(image) + synthetic_image(0, tail_size=0x800)
        candidates = scan_vector_table(image, load_base_hint=0x08000000)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(0x4000, top.offset)
        self.assertEqual(0x08004000, top.vector_table_address)
        self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)
        self.assertNotIn(0, [candidate.offset for candidate in candidates[:1]])

    def test_random_images_never_produce_confident_candidate(self):
        rng = random.Random(20260907)
        worst = 0.0
        produced = 0
        for _ in range(1000):
            image = bytes(rng.getrandbits(8) for _ in range(4096))
            candidates = scan_vector_table(image)
            if candidates:
                produced += 1
                worst = max(worst, candidates[0].score)
                # 随机镜像中不应出现任何置信候选（最高分候选不是随机偏移）。
                for candidate in candidates:
                    self.assertLess(
                        candidate.score,
                        CONFIDENT_MIN_SCORE,
                        f"随机数据不应达到置信阈值: {candidate}",
                    )
        # SP+Reset 双巧合封顶 0.60，必须与置信阈值保持明显距离。
        self.assertLess(worst, 0.61)
        self.assertLessEqual(produced, 1000)

    def test_all_zero_vector_table_is_weak_signal(self):
        # word[2..15] 全 0：合法但弱，低于置信阈值（数据区巧合的主要来源）。
        image = struct.pack("<16I", 0x20002000, 0x08000101, *([0] * 14)) + b"\x00" * 0x100
        candidates = scan_vector_table(image)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertLess(top.score, CONFIDENT_MIN_SCORE)
        self.assertFalse(top.confident)
        self.assertIn("vectors_all_zero_weak", top.signals)

    def test_cortex_m3_reserved_slots_stay_confident(self):
        # Cortex-M3 的 word[7..10]（UsageFault 保留区）为 0 属正常布局。
        image = synthetic_image(0, reserved_zero_slots=range(5, 9))
        candidates = scan_vector_table(image)
        self.assertTrue(candidates)
        self.assertGreaterEqual(candidates[0].score, CONFIDENT_MIN_SCORE)

    def test_large_image_uses_windowed_scan(self):
        # >4MB 镜像走窗口扫描：64KB 边界上的向量表仍可定位。
        image = bytearray(5 * 1024 * 1024)
        table = synthetic_image(0, tail_size=0x800)
        image[0x10000:0x10000 + len(table)] = table
        candidates = scan_vector_table(bytes(image))
        self.assertTrue(candidates)
        self.assertEqual(0x10000, candidates[0].offset)
        self.assertGreaterEqual(candidates[0].score, CONFIDENT_MIN_SCORE)

    def test_argument_validation_and_short_images(self):
        with self.assertRaisesRegex(ValueError, "arm"):
            scan_vector_table(b"\x00" * 64, arch="mips")
        with self.assertRaisesRegex(ValueError, "endian"):
            scan_vector_table(b"\x00" * 64, endian="middle")
        self.assertEqual([], scan_vector_table(b"\x00\x01"))
        # 纯 0 镜像：SP=0 不在 RAM 窗口内，无候选。
        self.assertEqual([], scan_vector_table(b"\x00" * 4096))

    def test_ram_windows_cover_documented_ranges(self):
        for value in (0x20000600, 0x10000000, 0x1FFF8000, 0x24001000, 0x30002000):
            self.assertTrue(any(start <= value < end for start, end in RAM_WINDOWS))
        for value in (0x08000000, 0x464C457F, 0x00010101, 0x40000000):
            self.assertFalse(any(start <= value < end for start, end in RAM_WINDOWS))

    def test_candidate_dataclass_fields(self):
        candidate = VectorTableCandidate(
            offset=0x4000,
            load_base=0x08000000,
            initial_sp=0x20000600,
            reset_pc=0x08004FB5,
            score=0.95,
            signals=("sp_ram_window",),
        )
        self.assertEqual(0x08004000, candidate.vector_table_address)
        self.assertTrue(candidate.confident)


class RealFirmwareRegressionTests(unittest.TestCase):
    """真实固件回归：ardupilot_Pixhawk1_STM32F427。"""

    def test_bin_locates_vector_table_at_offset_zero(self):
        if not F427_BIN.is_file():
            self.skipTest(f"firmware sample unavailable: {F427_BIN}")
        candidates = scan_vector_table(F427_BIN.read_bytes())
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(0, top.offset)
        self.assertEqual(0x20000600, top.initial_sp)
        self.assertEqual(0x08004FB5, top.reset_pc)
        self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)
        # 应用镜像（bootloader 之后）：Reset 指针扇区对齐推导出 0x08004000，
        # 与 ELF 的 .data LMA（0x08181d38..0x08182ab4）覆盖范围一致。
        self.assertEqual(0x08004000, top.load_base)

    def test_elf_memory_image_locates_vector_table_at_0x4000(self):
        if not F427_ELF.is_file():
            self.skipTest(f"firmware sample unavailable: {F427_ELF}")
        try:
            from elftools.elf.elffile import ELFFile
        except ImportError:
            self.skipTest("pyelftools unavailable")

        with F427_ELF.open("rb") as handle:
            elf = ELFFile(handle)
            segment = next(
                segment
                for segment in elf.iter_segments()
                if segment["p_type"] == "PT_LOAD" and int(segment["p_offset"]) == 0
            )
            memory_image = segment.data()
            vaddr = int(segment["p_vaddr"])

        candidates = scan_vector_table(memory_image, load_base_hint=vaddr)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(0x4000, top.offset)
        self.assertEqual(vaddr + 0x4000, top.vector_table_address)
        self.assertEqual(0x08004000, top.vector_table_address)
        self.assertEqual(0x20000600, top.initial_sp)
        self.assertEqual(0x08004FB5, top.reset_pc)
        self.assertGreaterEqual(top.score, CONFIDENT_MIN_SCORE)


class FirmwareInputAutoLayoutTests(unittest.TestCase):
    """BIN 自动布局（BintoElf 之前的向量表扫描）契约。"""

    @classmethod
    def setUpClass(cls):
        from lsgemu.firmware_input import DEFAULT_BINTOELF_ROOT

        if not (Path(DEFAULT_BINTOELF_ROOT) / "bin2elf" / "__init__.py").is_file():
            raise unittest.SkipTest("BintoElf test dependency is unavailable")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lsgemu_vt_test_")
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.bin_path = self.root / "cortex_app.bin"
        # 向量表在偏移 0 的应用镜像（load_base=0x08000000，reset Thumb）。
        self.bin_path.write_bytes(synthetic_image(0, tail_size=0x1000))

    def tearDown(self):
        self.temporary.cleanup()

    def resolve(self, **kwargs):
        from lsgemu.firmware_input import resolve_firmware_input

        return resolve_firmware_input(self.bin_path, cache_root=self.cache, **kwargs)

    def test_zero_parameter_bin_is_wrapped_with_scan_layout(self):
        resolved = self.resolve()
        self.assertTrue(resolved.converted)
        detection = resolved.provenance["conversion"]["vector_table_detection"]
        self.assertEqual("full_scan", detection["method"])
        self.assertEqual(0, detection["offset"])
        self.assertGreaterEqual(detection["score"], CONFIDENT_MIN_SCORE)
        self.assertEqual("vector_table_scan", resolved.provenance["layout_authority"])
        configuration = resolved.provenance["conversion"]["configuration"]
        self.assertEqual(0x08000000, configuration["base_address"])
        self.assertEqual(0x08000101, configuration["entry"])
        self.assertEqual("thumb", configuration["execution_mode"])
        self.assertIs(resolved.execution_thumb_override, True)
        # 布局决策参与确定性 identity，重跑命中缓存。
        again = self.resolve()
        self.assertTrue(again.provenance["conversion"]["cache_hit"])
        self.assertEqual(resolved.analysis_path, again.analysis_path)

    def test_garbage_bin_keeps_legacy_raw_behavior(self):
        self.bin_path.write_bytes(bytes.fromhex("00bf00bf704700bf"))
        resolved = self.resolve()
        self.assertFalse(resolved.converted)
        self.assertEqual("legacy_raw_bin", resolved.provenance["source_kind"])

    def test_raw_format_forces_legacy_loader(self):
        resolved = self.resolve(input_format="raw")
        self.assertFalse(resolved.converted)
        self.assertEqual("legacy_raw_bin", resolved.provenance["source_kind"])

    def test_bin_format_without_confident_table_still_errors(self):
        self.bin_path.write_bytes(bytes.fromhex("00bf00bf704700bf"))
        from lsgemu.firmware_input import FirmwareInputError

        with self.assertRaisesRegex(FirmwareInputError, "vector-table"):
            self.resolve(input_format="bin")

    def test_explicit_bin_spec_skips_scan(self):
        from lsgemu.firmware_input import BinConversionSpec

        resolved = self.resolve(
            bin_spec=BinConversionSpec(
                arch="arm",
                bits=32,
                endian="little",
                base_address=0x08000000,
                entry=0x08000101,
                thumb=True,
            )
        )
        self.assertTrue(resolved.converted)
        self.assertEqual("explicit_bin_spec", resolved.provenance["layout_authority"])
        self.assertNotIn(
            "vector_table_detection", resolved.provenance["conversion"]
        )


if __name__ == "__main__":
    unittest.main()
