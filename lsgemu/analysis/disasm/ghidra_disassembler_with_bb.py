#!/usr/bin/env python3
"""
Ghidra 反汇编器 - 支持基本块识别

功能：
1. 反汇编指令
2. 识别基本块（使用 Ghidra）
3. MMIO 识别（Python 层面）
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..file_parser.architecture import Architecture, Endianness, ArchInfo

logger = logging.getLogger(__name__)


@dataclass
class Instruction:
    """指令信息"""
    address: int
    mnemonic: str
    op_str: str
    size: int
    bytes: bytes

    def __str__(self):
        return f"0x{self.address:08x}: {self.mnemonic:8s} {self.op_str}"

    def is_branch(self) -> bool:
        mnem = self.mnemonic.lower()
        return mnem in ['b', 'beq', 'bne', 'blt', 'ble', 'bgt', 'bge',
                       'bl', 'blx', 'bx', 'cbz', 'cbnz', 'tbb', 'tbh',
                       'j', 'jal', 'jr', 'jalr', 'beqz', 'bnez',
                       'jmp', 'je', 'jne', 'jz', 'jnz', 'jl', 'jle', 'jg', 'jge']

    def is_call(self) -> bool:
        mnem = self.mnemonic.lower()
        return mnem in ['bl', 'blx', 'jal', 'jalr', 'call']

    def is_return(self) -> bool:
        mnem = self.mnemonic.lower()
        return (mnem in ['bx', 'ret', 'retn'] and 'lr' in self.op_str.lower()) or \
               (mnem == 'pop' and 'pc' in self.op_str.lower())


@dataclass
class BasicBlock:
    """基本块信息"""
    start_addr: int
    end_addr: int
    size: int
    instruction_count: int
    successors: List[int]
    predecessors: List[int]

    def __str__(self):
        return f"BB[0x{self.start_addr:08x} - 0x{self.end_addr:08x}] ({self.instruction_count} insns)"


class GhidraDisassemblerWithBB:
    """Ghidra 反汇编器（支持基本块）"""

    def __init__(self, arch_info: ArchInfo):
        """初始化反汇编器"""
        self.arch_info = arch_info
        self.analyze_headless = self._resolve_analyze_headless()
        self.ghidra_path = str(Path(self.analyze_headless).parents[1])

        logger.info(f"Ghidra 初始化（支持基本块）: {arch_info.architecture.value}, {arch_info.endianness.value}, {arch_info.bits}位")

    def _resolve_analyze_headless(self) -> str:
        candidates = []
        if os.environ.get("GHIDRA_ANALYZE_HEADLESS"):
            candidates.append(os.environ["GHIDRA_ANALYZE_HEADLESS"])
        if os.environ.get("GHIDRA_INSTALL_DIR"):
            candidates.append(os.path.join(os.environ["GHIDRA_INSTALL_DIR"], "support", "analyzeHeadless"))
        tried = []
        for candidate in candidates:
            if not candidate:
                continue
            tried.append(candidate)
            if os.path.exists(candidate):
                return candidate

        raise RuntimeError(
            "未找到可用的 Ghidra analyzeHeadless。请在 LSGEMU_CONFIG_FILE 的 ghidra.install_dir/"
            "ghidra.analyze_headless 中配置，或设置 GHIDRA_INSTALL_DIR/GHIDRA_ANALYZE_HEADLESS。"
            + (" 已尝试: " + ", ".join(tried) if tried else "")
        )

    def _get_ghidra_processor(self) -> str:
        """获取 Ghidra 处理器 ID"""
        arch = self.arch_info.architecture.value
        endian = self.arch_info.endianness

        if arch == 'ARM':
            return 'ARM:LE:32:Cortex' if endian == Endianness.LITTLE else 'ARM:BE:32:v7'
        elif arch == 'ARM64':
            return 'AARCH64:LE:64:v8A' if endian == Endianness.LITTLE else 'AARCH64:BE:64:v8A'
        elif arch == 'MIPS':
            return 'MIPS:LE:32:default' if endian == Endianness.LITTLE else 'MIPS:BE:32:default'
        elif arch == 'x86':
            return 'x86:LE:32:default'
        elif arch == 'x86-64':
            return 'x86:LE:64:default'
        else:
            raise ValueError(f"不支持的架构: {arch}")

    def disassemble_file(self, file_path: str) -> Tuple[List[Instruction], List[BasicBlock]]:
        """
        使用 Ghidra 反汇编文件并提取基本块

        Returns:
            (指令列表, 基本块列表)
        """
        logger.info(f"使用 Ghidra 反汇编: {file_path}")

        temp_dir = tempfile.mkdtemp(prefix='ghidra_bb_')
        logger.info(f"临时目录: {temp_dir}")

        try:
            project_name = "temp_project"
            insn_output = os.path.join(temp_dir, "instructions.txt")
            bb_output = os.path.join(temp_dir, "basic_blocks.txt")

            # 创建 Ghidra 脚本
            script_path = self._create_script(temp_dir, insn_output, bb_output)

            # 构建命令
            processor = self._get_ghidra_processor()

            cmd = [
                self.analyze_headless,
                temp_dir,
                project_name,
                '-import', file_path,
                '-processor', processor,
                '-scriptPath', temp_dir,
                '-postScript', 'DisassembleWithBB.java',
                '-scriptlog', os.path.join(temp_dir, 'script.log'),
                '-log', os.path.join(temp_dir, 'ghidra.log'),
                '-overwrite',
                '-deleteProject',
            ]

            logger.info(f"执行 Ghidra...")

            # 执行（超时时间增加到600秒）
            env = os.environ.copy()
            env.pop("DISPLAY", None)
            headless_option = "-Djava.awt.headless=true"
            java_tool_options = str(env.get("JAVA_TOOL_OPTIONS") or "").strip()
            if headless_option not in java_tool_options.split():
                env["JAVA_TOOL_OPTIONS"] = " ".join(
                    item for item in (java_tool_options, headless_option) if item
                )
            ghidra_config_home = os.path.join(temp_dir, "xdg_config")
            os.makedirs(ghidra_config_home, exist_ok=True)
            env.setdefault("XDG_CONFIG_HOME", ghidra_config_home)
            # Some local workstations have an older /usr/local/lib/libfreetype
            # ahead of the distro library cache.  Ghidra's Java process then
            # loads libfontconfig from /usr/lib but resolves freetype symbols
            # from /usr/local/lib, failing with missing FT_Done_MM_Var.  Keep
            # this scoped to Ghidra so the Python/Unicorn process can retain
            # its own library setup.
            system_lib_dirs = [
                "/usr/lib/x86_64-linux-gnu",
                "/lib/x86_64-linux-gnu",
            ]
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                system_lib_dirs + ([existing_ld_path] if existing_ld_path else [])
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

            logger.info(f"Ghidra 返回码: {result.returncode}")
            if result.returncode != 0:
                if result.stdout:
                    logger.warning("Ghidra stdout:\n%s", result.stdout[-4000:])
                if result.stderr:
                    logger.warning("Ghidra stderr:\n%s", result.stderr[-4000:])

            # 解析结果
            instructions = self._parse_instructions(insn_output) if os.path.exists(insn_output) else []
            basic_blocks = self._parse_basic_blocks(bb_output) if os.path.exists(bb_output) else []

            logger.info(f"解析完成: {len(instructions)} 条指令, {len(basic_blocks)} 个基本块")

            return instructions, basic_blocks

        except Exception as e:
            logger.error(f"执行失败: {e}")
            return [], []

    def _create_script(self, temp_dir: str, insn_output: str, bb_output: str) -> str:
        """创建 Ghidra 脚本"""
        insn_escaped = insn_output.replace('\\', '\\\\')
        bb_escaped = bb_output.replace('\\', '\\\\')

        # 使用简单的脚本名称
        script_content = f'''//Disassemble with basic blocks
//@category Analysis
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;
import ghidra.program.model.block.*;
import java.io.*;

public class DisassembleWithBB extends GhidraScript {{
    public void run() throws Exception {{
        println("Starting analysis...");

        Program program = currentProgram;
        Listing listing = program.getListing();
        AddressSetView execSet = program.getMemory().getExecuteSet();

        // 1. Disassemble instructions
        File insnFile = new File("{insn_escaped}");
        PrintWriter insnWriter = new PrintWriter(insnFile);

        int count = 0;
        InstructionIterator iter = listing.getInstructions(execSet, true);
        while (iter.hasNext()) {{
            Instruction insn = iter.next();
            String addr = insn.getAddress().toString();
            String mnem = insn.getMnemonicString();
            String ops = "";
            for (int i = 0; i < insn.getNumOperands(); i++) {{
                if (i > 0) ops += ", ";
                ops += insn.getDefaultOperandRepresentation(i);
            }}
            byte[] bytes = insn.getBytes();
            String hex = "";
            for (byte b : bytes) hex += String.format("%02x", b & 0xFF);

            insnWriter.println(addr + "|" + mnem + "|" + ops + "|" + insn.getLength() + "|" + hex);
            count++;
        }}
        insnWriter.close();
        println("Instructions: " + count);

        // 2. Extract basic blocks
        File bbFile = new File("{bb_escaped}");
        PrintWriter bbWriter = new PrintWriter(bbFile);

        BasicBlockModel bbModel = new BasicBlockModel(program);
        CodeBlockIterator bbIter = bbModel.getCodeBlocks(monitor);

        int bbCount = 0;
        while (bbIter.hasNext()) {{
            CodeBlock block = bbIter.next();
            String start = block.getFirstStartAddress().toString();
            String end = block.getMaxAddress().toString();

            int insnCount = 0;
            InstructionIterator blockIter = listing.getInstructions(block, true);
            while (blockIter.hasNext()) {{
                blockIter.next();
                insnCount++;
            }}

            String succs = "";
            CodeBlockReferenceIterator destIter = block.getDestinations(monitor);
            while (destIter.hasNext()) {{
                if (succs.length() > 0) succs += ",";
                succs += destIter.next().getDestinationAddress().toString();
            }}

            String preds = "";
            CodeBlockReferenceIterator srcIter = block.getSources(monitor);
            while (srcIter.hasNext()) {{
                if (preds.length() > 0) preds += ",";
                preds += srcIter.next().getSourceAddress().toString();
            }}

            long size = block.getMaxAddress().subtract(block.getFirstStartAddress()) + 1;
            bbWriter.println(start + "|" + end + "|" + size + "|" + insnCount + "|" + succs + "|" + preds);
            bbCount++;
        }}
        bbWriter.close();
        println("Basic blocks: " + bbCount);
        println("Done!");
    }}
}}
'''

        script_path = os.path.join(temp_dir, 'DisassembleWithBB.java')
        with open(script_path, 'w') as f:
            f.write(script_content)

        logger.info(f"创建脚本: {script_path}")
        return script_path

    def _parse_instructions(self, output_file: str) -> List[Instruction]:
        """解析指令"""
        instructions = []
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) != 5:
                        continue

                    addr = int(parts[0], 16)
                    mnem = parts[1]
                    ops = parts[2]
                    size = int(parts[3])
                    bytes_data = bytes.fromhex(parts[4])

                    instructions.append(Instruction(addr, mnem, ops, size, bytes_data))
        except Exception as e:
            logger.error(f"解析指令失败: {e}")

        return instructions

    def _parse_basic_blocks(self, output_file: str) -> List[BasicBlock]:
        """解析基本块"""
        basic_blocks = []
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) != 6:
                        continue

                    start = int(parts[0], 16)
                    end = int(parts[1], 16)
                    size = int(parts[2])
                    count = int(parts[3])
                    succs = [int(s, 16) for s in parts[4].split(',') if s]
                    preds = [int(p, 16) for p in parts[5].split(',') if p]

                    basic_blocks.append(BasicBlock(start, end, size, count, succs, preds))
        except Exception as e:
            logger.error(f"解析基本块失败: {e}")

        return basic_blocks


# 兼容性
GhidraDisassembler = GhidraDisassemblerWithBB
