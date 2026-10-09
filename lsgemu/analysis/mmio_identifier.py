#!/usr/bin/env python3
"""
MMIO 访问识别器（Python 实现）

基于反汇编指令识别 MMIO 访问，支持：
1. 直接访问
2. 间接访问
3. 多架构支持
4. 泛化设计
"""

import logging
from typing import List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MMIOAccess:
    """MMIO 访问信息"""
    address: int              # 指令地址
    instruction: str          # 指令文本
    mnemonic: str             # 助记符
    operands: str             # 操作数
    access_type: str          # 'read' or 'write'
    target_addr: Optional[int] = None  # 目标地址（如果可确定）
    base_reg: Optional[str] = None     # 基址寄存器
    offset: Optional[int] = None       # 偏移量
    is_indirect: bool = False          # 是否间接访问

    def __str__(self):
        access = 'R' if self.access_type == 'read' else 'W'
        target = f" -> 0x{self.target_addr:08x}" if self.target_addr else ""
        indirect = " (indirect)" if self.is_indirect else ""
        return f"0x{self.address:08x}: [{access}] {self.mnemonic:8s} {self.operands:20s}{target}{indirect}"


class MMIOIdentifier:
    """MMIO 访问识别器"""

    # MMIO 地址范围（可配置）
    MMIO_RANGES = [
        (0x40000000, 0x60000000),  # ARM Cortex-M 外设
        (0x42000000, 0x44000000),  # ARM Cortex-M 位带区域
        (0x1F000000, 0x20000000),  # 其他 MCU 外设
        (0xE0000000, 0xE0100000),  # ARM Cortex-M 系统外设
        (0xA0000000, 0xC0000000),  # 其他外设区域
    ]

    def __init__(self, code_start: int, code_end: int):
        """
        初始化识别器

        Args:
            code_start: 代码段起始地址
            code_end: 代码段结束地址
        """
        self.code_start = code_start
        self.code_end = code_end
        logger.info(f"MMIO 识别器初始化: 代码段 0x{code_start:08x} - 0x{code_end:08x}")

    def is_mmio_address(self, addr: int) -> bool:
        """检查地址是否在 MMIO 范围内"""
        # 排除代码段
        if self.code_start <= addr <= self.code_end:
            return False

        # 检查 MMIO 范围
        for start, end in self.MMIO_RANGES:
            if start <= addr < end:
                return True

        return False

    def identify_from_instructions(self, instructions) -> List[MMIOAccess]:
        """
        从指令列表中识别 MMIO 访问

        Args:
            instructions: 指令列表

        Returns:
            MMIO 访问列表
        """
        mmio_accesses = []

        for insn in instructions:
            access = self._analyze_instruction(insn)
            if access:
                mmio_accesses.append(access)

        logger.info(f"识别了 {len(mmio_accesses)} 个 MMIO 访问")
        return mmio_accesses

    def _analyze_instruction(self, insn) -> Optional[MMIOAccess]:
        """分析单条指令"""
        mnemonic = insn.mnemonic.lower()
        operands = insn.op_str

        # 判断是否是内存访问指令
        is_read = self._is_memory_read(mnemonic)
        is_write = self._is_memory_write(mnemonic)

        if not (is_read or is_write):
            return None

        # 分析操作数
        access_info = self._parse_operands(operands, insn.address)

        if not access_info:
            return None

        base_reg, offset, target_addr, is_indirect = access_info

        # 检查是否是 MMIO 访问
        if target_addr and self.is_mmio_address(target_addr):
            return MMIOAccess(
                address=insn.address,
                instruction=f"{insn.mnemonic} {insn.op_str}",
                mnemonic=insn.mnemonic,
                operands=insn.op_str,
                access_type='read' if is_read else 'write',
                target_addr=target_addr,
                base_reg=base_reg,
                offset=offset,
                is_indirect=is_indirect
            )

        # 启发式判断：无法确定地址时
        if self._is_likely_mmio(mnemonic, operands, base_reg, offset):
            return MMIOAccess(
                address=insn.address,
                instruction=f"{insn.mnemonic} {insn.op_str}",
                mnemonic=insn.mnemonic,
                operands=insn.op_str,
                access_type='read' if is_read else 'write',
                target_addr=None,
                base_reg=base_reg,
                offset=offset,
                is_indirect=is_indirect
            )

        return None

    def _is_memory_read(self, mnemonic: str) -> bool:
        """判断是否是内存读指令"""
        return mnemonic.startswith('ldr') or \
               mnemonic in ['lw', 'lh', 'lb', 'lwu', 'lhu', 'lbu'] or \
               mnemonic in ['mov', 'movzx', 'movsx']

    def _is_memory_write(self, mnemonic: str) -> bool:
        """判断是否是内存写指令"""
        return mnemonic.startswith('str') or \
               mnemonic in ['sw', 'sh', 'sb'] or \
               mnemonic == 'mov'

    def _parse_operands(self, operands: str, pc: int) -> Optional[tuple]:
        """
        解析操作数

        Returns:
            (base_reg, offset, target_addr, is_indirect) 或 None
        """
        # 检查是否包含内存引用
        if '[' not in operands:
            return None

        # 提取内存引用部分
        start = operands.find('[')
        end = operands.find(']')
        if start == -1 or end == -1:
            return None

        mem_ref = operands[start+1:end].strip()

        # 解析不同的寻址模式
        base_reg = None
        offset = 0
        target_addr = None
        is_indirect = True

        # 模式 1: [reg]
        if ',' not in mem_ref and '+' not in mem_ref and '-' not in mem_ref:
            base_reg = mem_ref.strip()

            # PC 相对寻址
            if base_reg.lower() == 'pc':
                # ARM Thumb: PC = 当前地址 + 4 (对齐)
                pc_value = (pc + 4) & ~3
                target_addr = pc_value
                is_indirect = False

        # 模式 2: [reg, #offset] 或 [reg, offset]
        elif ',' in mem_ref:
            parts = mem_ref.split(',')
            base_reg = parts[0].strip()

            # 提取偏移量
            offset_str = parts[1].strip()
            if offset_str.startswith('#'):
                offset_str = offset_str[1:]

            try:
                # 支持十六进制和十进制
                if offset_str.startswith('0x'):
                    offset = int(offset_str, 16)
                else:
                    offset = int(offset_str)

                # PC 相对寻址可以计算目标地址
                if base_reg.lower() == 'pc':
                    pc_value = (pc + 4) & ~3
                    target_addr = pc_value + offset
                    is_indirect = False

            except ValueError:
                # 无法解析偏移量（可能是寄存器）
                pass

        # 模式 3: [reg + offset] 或 [reg - offset]
        elif '+' in mem_ref or '-' in mem_ref:
            if '+' in mem_ref:
                parts = mem_ref.split('+')
                sign = 1
            else:
                parts = mem_ref.split('-')
                sign = -1

            base_reg = parts[0].strip()
            offset_str = parts[1].strip()

            try:
                if offset_str.startswith('0x'):
                    offset = sign * int(offset_str, 16)
                else:
                    offset = sign * int(offset_str)

                if base_reg.lower() == 'pc':
                    pc_value = (pc + 4) & ~3
                    target_addr = pc_value + offset
                    is_indirect = False

            except ValueError:
                pass

        return (base_reg, offset, target_addr, is_indirect)

    def _is_likely_mmio(self, mnemonic: str, operands: str,
                       base_reg: Optional[str], offset: Optional[int]) -> bool:
        """
        启发式判断：无法静态确定地址时，判断是否可能是 MMIO

        规则：
        1. 排除栈访问（sp）
        2. 排除 PC 相对访问
        3. 排除大偏移（>0x1000）
        4. 保留特定寄存器 + 小偏移
        """
        if not base_reg:
            return False

        # 排除栈访问
        if base_reg.lower() in ['sp', 'esp', 'rsp']:
            return False

        # 排除 PC 相对（已经处理过）
        if base_reg.lower() in ['pc', 'rip']:
            return False

        # 排除大偏移
        if offset and abs(offset) > 0x1000:
            return False

        # 保留 r4-r11（ARM 中通常用于保存 MMIO 基址）
        if base_reg.lower() in ['r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11']:
            if offset is None or abs(offset) < 0x100:
                return True

        return False

    def get_statistics(self, mmio_accesses: List[MMIOAccess]) -> dict:
        """获取统计信息"""
        read_count = sum(1 for m in mmio_accesses if m.access_type == 'read')
        write_count = sum(1 for m in mmio_accesses if m.access_type == 'write')

        # 统计唯一的 MMIO 地址
        unique_addrs = set(m.target_addr for m in mmio_accesses if m.target_addr)

        # 统计直接和间接访问
        direct_count = sum(1 for m in mmio_accesses if not m.is_indirect)
        indirect_count = sum(1 for m in mmio_accesses if m.is_indirect)

        return {
            'total': len(mmio_accesses),
            'read_count': read_count,
            'write_count': write_count,
            'unique_addresses': len(unique_addrs),
            'direct_count': direct_count,
            'indirect_count': indirect_count,
        }
