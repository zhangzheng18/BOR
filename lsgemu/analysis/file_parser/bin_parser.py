#!/usr/bin/env python3
"""
BIN文件解析器
"""

import logging
from pathlib import Path
from typing import Optional

from .architecture import Architecture, Endianness, ArchInfo

logger = logging.getLogger(__name__)


class BINParser:
    """BIN文件解析器"""

    def __init__(self, file_path: str):
        """
        初始化BIN解析器

        Args:
            file_path: BIN文件路径
        """
        self.file_path = Path(file_path)
        self.arch_info: Optional[ArchInfo] = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    def parse_with_hint(self,
                       architecture: Architecture,
                       endianness: Endianness = Endianness.LITTLE,
                       bits: int = 32,
                       base_addr: int = 0x08000000) -> ArchInfo:
        """
        使用提示信息解析BIN文件（逻辑保持不变）
        """
        file_size = self.file_path.stat().st_size

        self.arch_info = ArchInfo(
            architecture=architecture,
            endianness=endianness,
            bits=bits,
            machine_type='Binary (user specified)',
            base_addr=base_addr,
            entry_point=base_addr,  # BIN文件通常从基址开始执行
            code_size=file_size
        )

        logger.info(f"BIN文件解析（用户指定）: {self.file_path.name}")
        logger.info(f"  架构: {architecture.value}")
        logger.info(f"  字节序: {endianness.value}")
        logger.info(f"  位宽: {bits}位")

        return self.arch_info

    def parse_heuristic(self) -> ArchInfo:
        """
        增强版启发式识别：支持32/64位全架构 + 精准字节序检测
        """
        with open(self.file_path, 'rb') as f:
            data = f.read(4096)  # 扩大采样范围到4KB，提升识别准确率

        # 初始化默认值
        architecture = Architecture.UNKNOWN
        endianness = Endianness.UNKNOWN
        bits = 32

        # 优先级：64位架构 → 32位架构（避免32位特征误判64位）
        # 1. 检测64位架构
        if self._is_arm64(data):
            architecture = Architecture.ARM64
            endianness = Endianness.LITTLE
            bits = 64
            logger.info("启发式识别: ARM64 (AArch64)")
        elif self._is_x86_64(data):
            architecture = Architecture.X86_64
            endianness = Endianness.LITTLE
            bits = 64
            logger.info("启发式识别: x86-64")
        elif self._is_mips64(data):
            architecture = Architecture.MIPS64
            endianness = self._detect_mips_endianness(data)
            bits = 64
            logger.info(f"启发式识别: MIPS64 ({endianness.value})")
        elif self._is_ppc64(data):
            architecture = Architecture.PPC64
            endianness = self._detect_ppc_endianness(data)
            bits = 64
            logger.info(f"启发式识别: PowerPC64 ({endianness.value})")
        
        # 2. 检测32位架构
        elif self._is_arm_thumb(data):
            architecture = Architecture.ARM
            endianness = Endianness.LITTLE
            bits = 32
            logger.info("启发式识别: ARM Thumb")
        elif self._is_arm(data):
            architecture = Architecture.ARM
            endianness = Endianness.LITTLE
            bits = 32
            logger.info("启发式识别: ARM")
        elif self._is_mips(data):
            architecture = Architecture.MIPS
            endianness = self._detect_mips_endianness(data)
            bits = 32
            logger.info(f"启发式识别: MIPS ({endianness.value})")
        elif self._is_x86(data):
            architecture = Architecture.X86
            endianness = Endianness.LITTLE
            bits = 32
            logger.info("启发式识别: x86 (32位)")
        elif self._is_ppc(data):
            architecture = Architecture.PPC
            endianness = self._detect_ppc_endianness(data)
            bits = 32
            logger.info(f"启发式识别: PowerPC ({endianness.value})")
        else:
            logger.warning("无法通过启发式方法识别架构，默认返回Unknown")
            endianness = Endianness.UNKNOWN

        self.arch_info = ArchInfo(
            architecture=architecture,
            endianness=endianness,
            bits=bits,
            machine_type='Binary (heuristic)'
        )

        return self.arch_info

    # -------------------------- ARM 系列检测 --------------------------
    def _is_arm_thumb(self, data: bytes) -> bool:
        """增强版ARM Thumb（16位）指令检测"""
        if len(data) < 4:
            return False

        # 新增更多Thumb指令特征
        thumb_patterns = [
            b'\xf0\xb5',  # push {r4-r7, lr}
            b'\xf0\xbd',  # pop {r4-r7, pc}
            b'\x00\xb5',  # push {lr}
            b'\x00\xbd',  # pop {pc}
            b'\x20\xb9',  # push {r0}
            b'\x20\xbc',  # pop {r0}
            b'\x07\x46',  # mov r7, r0
            b'\x00\x20',  # movs r0, #0
        ]

        # 统计匹配数量，提升鲁棒性
        match_count = sum(1 for p in thumb_patterns if p in data[:200])
        return match_count >= 2

    def _is_arm(self, data: bytes) -> bool:
        """增强版ARM（32位）指令检测"""
        if len(data) < 4:
            return False

        # 统计有效ARM条件码指令数量
        cond_count = 0
        valid_conds = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xE, 0xF}  # 常见条件码
        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'little')
            cond = (word >> 28) & 0xF
            if cond in valid_conds:
                cond_count += 1

        # 新增ARM指令特征匹配（双重验证）
        arm_patterns = [
            b'\x00\x00\x80\xe2',  # add r0, r0, r0
            b'\x01\x10\xa0\xe3',  # mov r1, #1
            b'\xfe\xff\xff\xea',  # b #-2 (无限循环)
        ]
        pattern_match = any(p in data[:200] for p in arm_patterns)

        return cond_count > 12 or (cond_count > 8 and pattern_match)

    def _is_arm64(self, data: bytes) -> bool:
        if len(data) < 4:
            return False
        aarch64_opcodes = []
        match_count = 0

        # 遍历前200字节，按32位解析
        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'little')
            opc = (word >> 24) & 0xFF

            # 匹配常见ARM64操作码
            if opc in {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27}:  # movz/movk
                match_count += 1
            elif opc in {0x8B, 0x89, 0x8D}:  # add/sub
                match_count += 1
            elif opc in {0xD6, 0xD4}:  # ret/br
                match_count += 1

        return match_count >= 8

    # -------------------------- MIPS 系列检测 --------------------------
    def _is_mips(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # 常见MIPS操作码（大端解析）
        mips_opcodes = {0x00, 0x08, 0x09, 0x20, 0x24, 0x25, 0x8C, 0xAC, 0x0C}
        opcode_count = 0

        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            opcode = (word >> 26) & 0x3F
            if opcode in mips_opcodes:
                opcode_count += 1

        return opcode_count >= 6

    def _is_mips64(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # MIPS64特征：包含64位专用指令（如dadd、daddu、ld、sd）
        mips64_opcodes = {0x2D, 0x31, 0x37, 0x39, 0x3B}  # ld, sd, dadd, daddu, etc.
        opcode_count = 0

        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            opcode = (word >> 26) & 0x3F
            if opcode in mips64_opcodes:
                opcode_count += 1

        # MIPS64需满足：64位指令特征 + 基础MIPS特征
        return opcode_count >= 4 and self._is_mips(data)

    def _detect_mips_endianness(self, data: bytes) -> Endianness:
        if len(data) < 8:
            return Endianness.LITTLE

        # 分别按大端/小端解析，统计有效指令数
        big_endian_count = 0
        little_endian_count = 0
        valid_opcodes = {0x20, 0x24, 0x8C, 0xAC, 0x08, 0x0C}

        for i in range(0, min(len(data) - 3, 100), 4):
            # 大端解析
            big_word = int.from_bytes(data[i:i+4], 'big')
            big_op = (big_word >> 26) & 0x3F
            if big_op in valid_opcodes:
                big_endian_count += 1

            # 小端解析
            little_word = int.from_bytes(data[i:i+4], 'little')
            little_op = (little_word >> 26) & 0x3F
            if little_op in valid_opcodes:
                little_endian_count += 1

        # 选择有效指令更多的字节序
        if big_endian_count > little_endian_count and big_endian_count >= 3:
            return Endianness.BIG
        else:
            return Endianness.LITTLE

    # -------------------------- X86 系列检测 --------------------------
    def _is_x86(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # 新增更多x86特征
        x86_patterns = [
            b'\x55',        # push ebp
            b'\x89\xe5',    # mov ebp, esp
            b'\xc3',        # ret
            b'\x90',        # nop
            b'\x83\xec\x10',# sub esp, 0x10
            b'\x5d',        # pop ebp
            b'\xb8\x00\x00\x00\x00',  # mov eax, 0
            b'\x8b\x45\x08',# mov eax, [ebp+0x8]
        ]

        match_count = sum(1 for p in x86_patterns if p in data[:200])
        return match_count >= 2

    def _is_x86_64(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # x86-64专属特征（AMD64）
        x86_64_patterns = [
            b'\x53',        # push rbx (64位特有寄存器)
            b'\x48\x89\xe5',# mov rbp, rsp (64位前缀+指令)
            b'\x48\x83\xec',# sub rsp, xx (64位栈操作)
            b'\x48\x8b\x45',# mov rax, [rbp+xx]
            b'\xc3',        # ret (通用，但结合64位特征)
            b'\x49\x89\xca',# mov r10, rcx
        ]

        match_count = sum(1 for p in x86_64_patterns if p in data[:200])
        return match_count >= 2

    # -------------------------- PowerPC 系列检测 --------------------------
    def _is_ppc(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # PowerPC指令特征（大端为主，固定32位，操作码在高6位）
        ppc_opcodes = {0x38, 0x80, 0x90, 0x48, 0x7c, 0x88, 0xa0, 0x4c}  # li, lwz, stw, b, etc.
        opcode_count = 0

        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            opcode = (word >> 26) & 0x3F
            if opcode in ppc_opcodes:
                opcode_count += 1

        # 补充PowerPC特有指令模式
        ppc_patterns = [
            b'\x7c\x08\x02\xa6',  # mflr r0
            b'\x7c\x7e\x1b\x78',  # mr r31, r31
            b'\x4e\x80\x00\x20',  # blr
        ]
        pattern_match = any(p in data[:200] for p in ppc_patterns)

        return opcode_count >= 5 or (opcode_count >= 3 and pattern_match)

    def _is_ppc64(self, data: bytes) -> bool:
        if len(data) < 4:
            return False

        # PPC64特有指令（64位寄存器操作）
        ppc64_opcodes = {0x3d, 0x60, 0x61, 0x7d}  # lis, ld, std, etc.
        opcode_count = 0

        for i in range(0, min(len(data) - 3, 200), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            opcode = (word >> 26) & 0x3F
            if opcode in ppc64_opcodes:
                opcode_count += 1

        # PPC64需满足：64位指令特征 + 基础PPC特征
        return opcode_count >= 4 and self._is_ppc(data)

    def _detect_ppc_endianness(self, data: bytes) -> bool:
        if len(data) < 4:
            return Endianness.BIG

        # 统计大端/小端下有效PPC指令数
        big_count = 0
        little_count = 0
        ppc_ops = {0x38, 0x80, 0x90, 0x48, 0x7c}

        for i in range(0, min(len(data) - 3, 100), 4):
            # 大端解析
            big_word = int.from_bytes(data[i:i+4], 'big')
            big_op = (big_word >> 26) & 0x3F
            if big_op in ppc_ops:
                big_count += 1

            # 小端解析
            little_word = int.from_bytes(data[i:i+4], 'little')
            little_op = (little_word >> 26) & 0x3F
            if little_op in ppc_ops:
                little_count += 1

        if big_count > little_count and big_count >= 3:
            return Endianness.BIG
        elif little_count >= 3:
            return Endianness.LITTLE
        else:
            return Endianness.BIG  # PPC默认大端