#!/usr/bin/env python3
"""
智能MMIO值推断

当检测到轮询循环时，使用可审计的局部指令规则推断候选 MMIO 值。
未解决的循环由 ``IntelligentEmulator`` 统一交给带候选地址校验、缓存
和调用预算的 ``DeadlockLLMSolver``，避免两套 LLM 路径生成不同语义
的约束。
"""

import logging
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class IntelligentMMIOInferencer:
    """
    智能MMIO值推断器

    核心思路：
    1. 检测到轮询循环
    2. 提取循环中的MMIO访问指令
    3. 根据比较、位测试和分支方向推断候选值
    4. 将未解决项留给调用方的受控 fallback
    """

    def __init__(self, llm_analyzer=None):
        """
        初始化

        Args:
            llm_analyzer: 旧调用方兼容参数；LLM fallback 由调用方统一管理
        """
        self.llm_analyzer = llm_analyzer
        self.inferred_values = {}  # {mmio_addr: value}

    @staticmethod
    def _is_mmio_address(address: int) -> bool:
        """Generic MCU MMIO range, including Cortex-M system control space."""
        address = int(address) & 0xFFFFFFFF
        return (
            0x40000000 <= address < 0x60000000
            or 0xE0000000 <= address < 0xE0100000
        )

    def analyze_polling_loop(self,
                            loop_bbs: List[int],
                            static_bbs: Dict[int, List[Dict]],
                            mmio_accesses: List[Tuple[int, int, bool, int]]) -> Dict[int, int]:
        """
        分析轮询循环，推断MMIO值

        Args:
            loop_bbs: 循环体的BB列表
            static_bbs: 静态BB信息 {address: [instructions]}
            mmio_accesses: MMIO访问历史 [(pc, addr, is_read, value), ...]

        Returns:
            {mmio_addr: inferred_value}
        """
        logger.info(f"\n分析轮询循环...")
        logger.info(f"循环体BB: {[f'0x{bb:08x}' for bb in loop_bbs]}")
        logger.info(f"MMIO访问历史: {len(mmio_accesses)} 条记录")

        # 1. 提取循环中的MMIO读取
        mmio_reads = self._extract_mmio_reads(loop_bbs, static_bbs, mmio_accesses)

        if not mmio_reads:
            logger.warning("未找到MMIO读取")
            return {}

        logger.info(f"找到 {len(mmio_reads)} 个MMIO读取")

        # 2. 分析每个MMIO读取
        inferred = {}

        for mmio_addr, read_pc, test_insn, branch_condition, target_direction, constraint_pc in mmio_reads:
            logger.info(f"\n分析MMIO @ 0x{mmio_addr:08x}")
            logger.info(f"  读取PC: 0x{read_pc:08x}")
            logger.info(f"  测试指令: {test_insn}")
            if branch_condition:
                logger.info(f"  约束分支: {branch_condition}, 目标方向: {'taken' if target_direction else 'not-taken'}")

            # 3. 推断值
            value = self._infer_value(
                mmio_addr,
                read_pc,
                test_insn,
                loop_bbs,
                static_bbs,
                branch_condition=branch_condition,
                target_direction=target_direction,
            )

            if value is not None:
                inferred[mmio_addr] = value
                logger.info(f"  ✓ 推断值: 0x{value:08x}")

        return inferred

    @staticmethod
    def _bb_contains_pc(bb_addr: int, instructions: List[Dict], pc: int) -> bool:
        if not instructions:
            return int(bb_addr) <= int(pc) < int(bb_addr) + 4
        first = int(instructions[0].get("address", bb_addr) or bb_addr)
        last = instructions[-1]
        last_end = int(last.get("address", first) or first) + int(last.get("size", 4) or 4)
        return first <= int(pc) < last_end

    def _extract_mmio_reads(self,
                           loop_bbs: List[int],
                           static_bbs: Dict[int, List[Dict]],
                           mmio_accesses: List[Tuple[int, int, bool, int]]) -> List[Tuple[int, int, str, Optional[str], bool, Optional[int]]]:
        """
        提取循环中的MMIO读取

        改进：直接从mmio_accesses中提取，然后查找对应的测试指令
        关键：收集所有相关的测试指令（AND, CMP等）

        Returns:
            [(mmio_addr, read_pc, test_instruction, branch_condition, target_direction, constraint_pc), ...]
        """
        mmio_reads = []
        seen = set()

        # 从MMIO访问历史中提取循环体内的MMIO读取
        for pc, addr, is_read, value in mmio_accesses:
            if not is_read or not self._is_mmio_address(addr):
                continue

            # 检查PC是否在循环体内
            in_loop = False
            for bb_addr in loop_bbs:
                instructions = static_bbs.get(bb_addr, [])
                if self._bb_contains_pc(bb_addr, instructions, pc):
                    in_loop = True
                    break

            if not in_loop:
                continue

            # 查找这个PC所在的BB和后续的测试指令
            for bb_addr in loop_bbs:
                instructions = static_bbs.get(bb_addr, [])

                if self._bb_contains_pc(bb_addr, instructions, pc):
                    # 找到了包含这个PC的BB
                    for i, insn in enumerate(instructions):
                        mnemonic = insn.get('mnemonic', '').upper()
                        operands = insn.get('operands', '')

                        # 查找测试指令
                        if mnemonic in ['TST', 'TST.W', 'CMP', 'AND', 'ANDS', 'TEQ', 'LSL', 'LSLS', 'LSR', 'LSRS']:
                            branch_condition, target_direction, constraint_pc = self._find_following_branch(
                                instructions, i, loop_bbs
                            )
                            test_str = f"{mnemonic} {operands}"
                            key = (addr, pc, test_str, branch_condition, target_direction, constraint_pc)
                            if key not in seen:
                                seen.add(key)
                                mmio_reads.append((addr, pc, test_str, branch_condition, target_direction, constraint_pc))
                                logger.debug(
                                    f"找到MMIO读取: PC=0x{pc:08x}, Addr=0x{addr:08x}, Test={test_str}"
                                )
                            break
                    break

        return mmio_reads

    def _find_following_branch(self, instructions: List[Dict], test_index: int,
                               loop_bbs: List[int]) -> Tuple[Optional[str], bool, Optional[int]]:
        branch_mnemonics = {
            'BEQ', 'BNE', 'BCS', 'BCC', 'BHS', 'BLO', 'BMI', 'BPL',
            'BVS', 'BVC', 'BHI', 'BLS', 'BGE', 'BLT', 'BGT', 'BLE',
        }
        loop_set = set(loop_bbs)
        for insn in instructions[test_index + 1:test_index + 5]:
            mnemonic = insn.get('mnemonic', '').upper()
            if mnemonic not in branch_mnemonics:
                continue
            target = self._parse_branch_target(insn.get('operands', ''))
            target_direction = True
            if target is not None:
                target_direction = target not in loop_set
            return mnemonic, target_direction, insn.get('address')
        return None, True, None

    def _infer_value(self,
                    mmio_addr: int,
                    read_pc: int,
                    test_insn: str,
                    loop_bbs: List[int],
                    static_bbs: Dict[int, List[Dict]],
                    branch_condition: Optional[str] = None,
                    target_direction: bool = True) -> Optional[int]:
        """
        推断MMIO值

        Args:
            mmio_addr: MMIO地址
            read_pc: 读取指令的PC
            test_insn: 测试指令（如 "TST.W R3, #0x2000000"）
            loop_bbs: 循环体BB列表
            static_bbs: 静态BB信息

        Returns:
            推断的值，如果无法推断则返回None
        """
        # 启发式分析
        value = self._heuristic_infer(test_insn, branch_condition, target_direction)

        if value is not None:
            return value

        # The caller owns the validated LLM fallback. Returning None here
        # prevents a second unscoped path from bypassing its candidate-address
        # checks and per-loop call budget.
        return None

    def _heuristic_infer(self, test_insn: str, branch_condition: Optional[str] = None,
                         target_direction: bool = True) -> Optional[int]:
        """
        启发式推断

        规则：
        1. TST.W R3, #0x2000000 → 返回 0x2000000（设置该位）
        2. TST R3, #0x1 → 返回 0x1
        3. CMP R3, #0 → 返回 非0值（如 0x1）
        4. AND R3, #0xc; CMP R3, #0x8 → 返回 0x8（满足CMP条件）
        """
        test_insn_upper = test_insn.upper()

        # 特殊情况：AND + CMP 组合
        if 'AND' in test_insn_upper and 'CMP' in test_insn_upper:
            # 提取CMP的比较值（这是最终需要的值）
            cmp_match = re.search(r'CMP[^#]*#(0X[0-9A-F]+|[0-9]+)', test_insn_upper)
            if cmp_match:
                imm_str = cmp_match.group(1)
                if imm_str.startswith('0X'):
                    value = int(imm_str, 16)
                else:
                    value = int(imm_str)

                logger.info(f"  启发式: AND+CMP 组合，返回CMP值 0x{value:08x}")
                return value

        # 规则1: TST 指令
        if 'TST' in test_insn_upper:
            # 提取立即数
            match = re.search(r'#(0x[0-9A-Fa-f]+|[0-9]+)', test_insn)
            if match:
                imm_str = match.group(1)
                if imm_str.startswith('0x'):
                    value = int(imm_str, 16)
                else:
                    value = int(imm_str)

                if branch_condition == 'BEQ':
                    inferred = 0 if target_direction else value
                elif branch_condition == 'BNE':
                    inferred = value if target_direction else 0
                else:
                    inferred = value
                logger.info(f"  启发式: TST 指令，返回 0x{inferred:08x}")
                return inferred

        # 规则2: CMP 指令
        if 'CMP' in test_insn_upper:
            # 提取立即数
            match = re.search(r'#(0x[0-9A-Fa-f]+|[0-9]+)', test_insn)
            if match:
                imm_str = match.group(1)
                if imm_str.startswith('0x'):
                    cmp_value = int(imm_str, 16)
                else:
                    cmp_value = int(imm_str)

                inferred = self._infer_cmp_value(cmp_value, branch_condition, target_direction)
                logger.info(f"  启发式: CMP with 0x{cmp_value:x}，返回 0x{inferred:08x}")
                return inferred

        # 规则3: AND/ANDS 指令（单独出现）
        if 'AND' in test_insn_upper:
            match = re.search(r'#(0x[0-9A-Fa-f]+|[0-9]+)', test_insn)
            if match:
                imm_str = match.group(1)
                if imm_str.startswith('0x'):
                    value = int(imm_str, 16)
                else:
                    value = int(imm_str)

                logger.info(f"  启发式: AND 指令，设置位 0x{value:08x}")
                return value

        # 规则4: shift + sign/zero branch.  Example:
        # `lsls r3, r3, #0x15; bpl loop` exits when source bit 10 is set.
        if 'LSL' in test_insn_upper:
            match = re.search(r'#(0x[0-9A-Fa-f]+|[0-9]+)', test_insn)
            if match and branch_condition in {'BPL', 'BMI'}:
                shift_text = match.group(1)
                shift = int(shift_text, 16) if shift_text.lower().startswith('0x') else int(shift_text)
                if 0 <= shift <= 31:
                    source_bit = 31 - shift
                    # target_direction means the branch target exits the loop.
                    # For BPL, branch is taken when N=0; for BMI, when N=1.
                    if branch_condition == 'BPL':
                        need_negative = not target_direction
                    else:
                        need_negative = target_direction
                    value = (1 << source_bit) if need_negative else 0
                    logger.info(f"  启发式: LSL/{branch_condition}，返回 0x{value:08x}")
                    return value

        return None

    def _infer_cmp_value(self, compare_value: int, condition: Optional[str], target_direction: bool) -> int:
        if condition == 'BEQ':
            return compare_value if target_direction else compare_value + 1
        if condition == 'BNE':
            return compare_value + 1 if target_direction else compare_value
        if condition in ['BGT', 'BHI']:
            return compare_value + 1 if target_direction else compare_value
        if condition in ['BLT', 'BLO']:
            return compare_value - 1 if target_direction else compare_value
        if condition in ['BGE', 'BHS']:
            return compare_value if target_direction else compare_value - 1
        if condition in ['BLE', 'BLS']:
            return compare_value if target_direction else compare_value + 1
        return 0x1 if compare_value == 0 else compare_value

    def _parse_branch_target(self, operands: str) -> Optional[int]:
        text = str(operands).strip().split()[0] if str(operands).strip() else ""
        try:
            return int(text, 16) if text.lower().startswith('0x') else int(text)
        except ValueError:
            return None
