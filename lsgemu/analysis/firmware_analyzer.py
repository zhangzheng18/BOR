#!/usr/bin/env python3
"""
统一的固件静态分析系统

整合 file_parser 和 ghidra_disassembler，提供完整的固件分析能力：
1. 文件解析（ELF/BIN）
2. 架构识别
3. 反汇编（Ghidra）
4. 指令分析
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .file_parser import ELFParser, BINParser, Architecture, Endianness, ArchInfo
from .disasm.ghidra_disassembler_with_bb import GhidraDisassemblerWithBB, Instruction, BasicBlock
from .mmio_identifier import MMIOIdentifier, MMIOAccess

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析结果"""
    # 文件信息
    file_path: str
    file_size: int
    file_type: str  # 'ELF' or 'BIN'

    # 架构信息
    arch_info: ArchInfo

    # 反汇编结果
    instructions: List[Instruction]
    total_instructions: int
    total_bytes: int

    # 基本块信息（新增）
    basic_blocks: List[BasicBlock]
    total_basic_blocks: int

    # MMIO 访问信息（新增）
    mmio_accesses: List[MMIOAccess]
    total_mmio_accesses: int
    mmio_read_count: int
    mmio_write_count: int

    # 指令统计
    branch_count: int
    call_count: int
    return_count: int
    memory_access_count: int

    # 地址范围
    min_address: int
    max_address: int

    # 分析时间
    parse_time: float
    disasm_time: float
    total_time: float


class FirmwareAnalyzer:
    """统一的固件分析器"""

    def __init__(self, use_ghidra: bool = True):
        """
        初始化分析器

        Args:
            use_ghidra: 是否使用 Ghidra（默认 True）
        """
        self.use_ghidra = use_ghidra
        self.arch_info: Optional[ArchInfo] = None
        self.instructions: List[Instruction] = []

        logger.info(f"固件分析器初始化: 使用 {'Ghidra' if use_ghidra else 'Capstone'}")

    def analyze(self, file_path: str) -> AnalysisResult:
        """
        分析固件文件

        Args:
            file_path: 固件文件路径

        Returns:
            AnalysisResult 对象
        """
        import time

        start_time = time.time()

        logger.info(f"开始分析固件: {file_path}")

        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_size = Path(file_path).stat().st_size

        # 步骤 1: 解析文件
        parse_start = time.time()
        arch_info, file_type = self._parse_file(file_path)
        parse_time = time.time() - parse_start

        logger.info(f"文件解析完成: {file_type}, {arch_info.architecture.value}, {parse_time:.2f}s")

        # 步骤 2: 反汇编（包含基本块和 MMIO 识别）
        disasm_start = time.time()
        instructions, basic_blocks, mmio_accesses = self._disassemble(file_path, arch_info)
        disasm_time = time.time() - disasm_start

        logger.info(f"反汇编完成: {len(instructions)} 条指令, {len(basic_blocks)} 个基本块, {len(mmio_accesses)} 个 MMIO 访问, {disasm_time:.2f}s")

        # 步骤 3: 统计分析
        stats = self._analyze_instructions(instructions)
        mmio_stats = self._analyze_mmio(mmio_accesses)

        total_time = time.time() - start_time

        # 构建结果
        result = AnalysisResult(
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            arch_info=arch_info,
            instructions=instructions,
            total_instructions=len(instructions),
            total_bytes=stats['total_bytes'],
            basic_blocks=basic_blocks,
            total_basic_blocks=len(basic_blocks),
            mmio_accesses=mmio_accesses,
            total_mmio_accesses=len(mmio_accesses),
            mmio_read_count=mmio_stats['read_count'],
            mmio_write_count=mmio_stats['write_count'],
            branch_count=stats['branch_count'],
            call_count=stats['call_count'],
            return_count=stats['return_count'],
            memory_access_count=stats['memory_access_count'],
            min_address=stats['min_address'],
            max_address=stats['max_address'],
            parse_time=parse_time,
            disasm_time=disasm_time,
            total_time=total_time
        )

        logger.info(f"分析完成: 总时间 {total_time:.2f}s")

        return result

    def _parse_file(self, file_path: str) -> tuple:
        """
        解析文件

        Returns:
            (ArchInfo, file_type)
        """
        # 检查文件类型
        with open(file_path, 'rb') as f:
            magic = f.read(4)

        is_elf = (magic == b'\x7fELF')

        if is_elf:
            # ELF 文件
            parser = ELFParser(file_path)
            arch_info = parser.parse()
            file_type = 'ELF'
        else:
            # BIN 文件 - 尝试启发式识别
            parser = BINParser(file_path)
            arch_info = parser.parse_heuristic()

            # 如果启发式失败，使用默认配置（ARM Cortex-M）
            if arch_info.architecture == Architecture.UNKNOWN:
                logger.warning("启发式识别失败，使用默认配置（ARM Cortex-M）")
                arch_info = parser.parse_with_hint(
                    architecture=Architecture.ARM,
                    endianness=Endianness.LITTLE,
                    bits=32,
                    base_addr=0x08000000
                )

            file_type = 'BIN'

        self.arch_info = arch_info
        return arch_info, file_type

    def _disassemble(self, file_path: str, arch_info: ArchInfo) -> Tuple[List[Instruction], List[BasicBlock], List[MMIOAccess]]:
        """
        反汇编文件（包含基本块和 MMIO 识别）

        Returns:
            (指令列表, 基本块列表, MMIO访问列表)
        """
        if self.use_ghidra:
            # 使用 Ghidra 获取指令和基本块
            disasm = GhidraDisassemblerWithBB(arch_info)
            instructions, basic_blocks = disasm.disassemble_file(file_path)

            # 在 Python 层面识别 MMIO
            if instructions:
                # 获取代码段范围
                code_start = min(insn.address for insn in instructions)
                code_end = max(insn.address for insn in instructions)

                # 创建 MMIO 识别器
                mmio_identifier = MMIOIdentifier(code_start, code_end)
                mmio_accesses = mmio_identifier.identify_from_instructions(instructions)
            else:
                mmio_accesses = []

        else:
            # 使用 Capstone（不支持基本块和 MMIO）
            from .disasm import UniversalDisassembler

            disasm = UniversalDisassembler(arch_info)

            # 读取文件
            with open(file_path, 'rb') as f:
                code = f.read()

            instructions = list(disasm.disassemble(code, arch_info.base_addr))
            basic_blocks = []
            mmio_accesses = []

        self.instructions = instructions
        self.basic_blocks = basic_blocks
        self.mmio_accesses = mmio_accesses

        return instructions, basic_blocks, mmio_accesses

    def _analyze_mmio(self, mmio_accesses: List[MMIOAccess]) -> Dict[str, Any]:
        """
        分析 MMIO 访问统计信息

        Returns:
            统计字典
        """
        if not mmio_accesses:
            return {
                'read_count': 0,
                'write_count': 0,
            }

        read_count = sum(1 for m in mmio_accesses if m.access_type == 'read')
        write_count = sum(1 for m in mmio_accesses if m.access_type == 'write')

        return {
            'read_count': read_count,
            'write_count': write_count,
        }

    def _analyze_instructions(self, instructions: List[Instruction]) -> Dict[str, Any]:
        """
        分析指令统计信息

        Returns:
            统计字典
        """
        if not instructions:
            return {
                'total_bytes': 0,
                'branch_count': 0,
                'call_count': 0,
                'return_count': 0,
                'memory_access_count': 0,
                'min_address': 0,
                'max_address': 0,
            }

        total_bytes = sum(insn.size for insn in instructions)
        branch_count = sum(1 for insn in instructions if insn.is_branch())
        call_count = sum(1 for insn in instructions if insn.is_call())
        return_count = sum(1 for insn in instructions if insn.is_return())
        memory_access_count = sum(1 for insn in instructions if hasattr(insn, 'is_memory_access') and insn.is_memory_access())

        min_address = min(insn.address for insn in instructions)
        max_address = max(insn.address for insn in instructions)

        return {
            'total_bytes': total_bytes,
            'branch_count': branch_count,
            'call_count': call_count,
            'return_count': return_count,
            'memory_access_count': memory_access_count,
            'min_address': min_address,
            'max_address': max_address,
        }

    def print_summary(self, result: AnalysisResult):
        """打印分析摘要"""
        print("\n" + "=" * 80)
        print("固件分析摘要")
        print("=" * 80)

        print(f"\n[文件信息]")
        print(f"  文件路径: {result.file_path}")
        print(f"  文件大小: {result.file_size} bytes ({result.file_size / 1024:.2f} KB)")
        print(f"  文件类型: {result.file_type}")

        print(f"\n[架构信息]")
        print(f"  架构: {result.arch_info.architecture.value}")
        print(f"  字节序: {result.arch_info.endianness.value}")
        print(f"  位宽: {result.arch_info.bits} 位")
        print(f"  入口点: 0x{result.arch_info.entry_point:08x}")
        print(f"  基址: 0x{result.arch_info.base_addr:08x}")

        print(f"\n[反汇编结果]")
        print(f"  总指令数: {result.total_instructions}")
        print(f"  反汇编字节数: {result.total_bytes} bytes")
        print(f"  地址范围: 0x{result.min_address:08x} - 0x{result.max_address:08x}")

        print(f"\n[基本块信息]")
        print(f"  总基本块数: {result.total_basic_blocks}")
        if result.total_basic_blocks > 0:
            avg_bb_size = result.total_instructions / result.total_basic_blocks
            print(f"  平均基本块大小: {avg_bb_size:.2f} 条指令")

        print(f"\n[MMIO 访问]")
        print(f"  总 MMIO 访问: {result.total_mmio_accesses}")
        print(f"  读操作: {result.mmio_read_count}")
        print(f"  写操作: {result.mmio_write_count}")

        print(f"\n[指令统计]")
        print(f"  分支指令: {result.branch_count} ({result.branch_count / result.total_instructions * 100:.2f}%)")
        print(f"  调用指令: {result.call_count} ({result.call_count / result.total_instructions * 100:.2f}%)")
        print(f"  返回指令: {result.return_count} ({result.return_count / result.total_instructions * 100:.2f}%)")
        if result.memory_access_count > 0:
            print(f"  内存访问: {result.memory_access_count} ({result.memory_access_count / result.total_instructions * 100:.2f}%)")

        print(f"\n[性能]")
        print(f"  文件解析: {result.parse_time:.2f}s")
        print(f"  反汇编: {result.disasm_time:.2f}s")
        print(f"  总时间: {result.total_time:.2f}s")

        print("=" * 80)

    def print_basic_blocks(self, result: AnalysisResult, count: int = 10):
        """打印基本块列表"""
        if not result.basic_blocks:
            print(f"\n[基本块] 无基本块信息")
            return

        print(f"\n[前 {count} 个基本块]")
        for i, bb in enumerate(result.basic_blocks[:count]):
            print(f"  {i+1:3d}. {bb}")
            if bb.successors:
                succ_str = ', '.join(f"0x{s:08x}" for s in bb.successors)
                print(f"       后继: {succ_str}")

    def print_mmio_accesses(self, result: AnalysisResult, count: int = 20):
        """打印 MMIO 访问列表"""
        if not result.mmio_accesses:
            print(f"\n[MMIO 访问] 无 MMIO 访问")
            return

        print(f"\n[前 {count} 个 MMIO 访问]")
        for i, mmio in enumerate(result.mmio_accesses[:count]):
            print(f"  {i+1:3d}. {mmio}")

    def print_instructions(self, result: AnalysisResult, count: int = 20):
        """打印指令列表"""
        print(f"\n[前 {count} 条指令]")
        for i, insn in enumerate(result.instructions[:count]):
            print(f"  {insn}")

        if len(result.instructions) > count:
            print(f"\n[最后 10 条指令]")
            for insn in result.instructions[-10:]:
                print(f"  {insn}")

    def export_to_file(self, result: AnalysisResult, output_path: str):
        """导出分析结果到文件"""
        with open(output_path, 'w') as f:
            f.write("# 固件分析结果\n\n")

            f.write("## 文件信息\n")
            f.write(f"- 文件路径: {result.file_path}\n")
            f.write(f"- 文件大小: {result.file_size} bytes\n")
            f.write(f"- 文件类型: {result.file_type}\n\n")

            f.write("## 架构信息\n")
            f.write(f"- 架构: {result.arch_info.architecture.value}\n")
            f.write(f"- 字节序: {result.arch_info.endianness.value}\n")
            f.write(f"- 位宽: {result.arch_info.bits} 位\n")
            f.write(f"- 入口点: 0x{result.arch_info.entry_point:08x}\n")
            f.write(f"- 基址: 0x{result.arch_info.base_addr:08x}\n\n")

            f.write("## 反汇编结果\n")
            f.write(f"- 总指令数: {result.total_instructions}\n")
            f.write(f"- 反汇编字节数: {result.total_bytes} bytes\n")
            f.write(f"- 地址范围: 0x{result.min_address:08x} - 0x{result.max_address:08x}\n\n")

            f.write("## 指令统计\n")
            f.write(f"- 分支指令: {result.branch_count}\n")
            f.write(f"- 调用指令: {result.call_count}\n")
            f.write(f"- 返回指令: {result.return_count}\n\n")

            f.write("## 指令列表\n\n")
            f.write("```\n")
            for insn in result.instructions:
                f.write(f"{insn}\n")
            f.write("```\n")

        logger.info(f"分析结果已导出到: {output_path}")


def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='固件静态分析系统')
    parser.add_argument('firmware', help='固件文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--capstone', action='store_true', help='使用 Capstone 而非 Ghidra')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('-n', '--num-instructions', type=int, default=20, help='显示的指令数量')

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(name)s - %(message)s'
    )

    # 创建分析器
    analyzer = FirmwareAnalyzer(use_ghidra=not args.capstone)

    try:
        # 分析固件
        result = analyzer.analyze(args.firmware)

        # 打印摘要
        analyzer.print_summary(result)

        # 打印指令
        analyzer.print_instructions(result, count=args.num_instructions)

        # 导出到文件
        if args.output:
            analyzer.export_to_file(result, args.output)

        return 0

    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
