#!/usr/bin/env python3
"""
智能循环分类器

核心思路：
不是简单地检测"PC重复次数"，而是分析循环的语义特征，
区分不同类型的循环，并采取相应的策略。

循环类型：
1. 初始化循环 - 有明确的结束条件，会自然退出
2. 轮询循环 - 等待MMIO状态变化
3. 延迟循环 - 纯计数延迟
4. 死循环 - 无条件跳转到自己（Error_Handler）

这是真正智能的方法，而不是暴力检测。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import os
import re


def _env_flag(name: str, default: str) -> bool:
    """r36 Q5：与 historical_runner._env_flag 同款的三态环境开关。"""
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

logger = logging.getLogger(__name__)


class LoopType(Enum):
    """循环类型"""
    INITIALIZATION = "initialization"  # 初始化循环（复制数据、清零BSS）
    POLLING = "polling"                # 轮询循环（等待MMIO状态）
    MEMORY_WAIT = "memory_wait"        # 等待普通内存/队列状态变化
    DELAY = "delay"                    # 延迟循环（纯计数）
    DEADLOCK = "deadlock"              # 死循环（Error_Handler）
    UNKNOWN = "unknown"                # 未知类型


@dataclass
class LoopCharacteristics:
    """
    循环特征

    通过分析循环的指令模式和行为，识别循环类型
    """
    # 基本信息
    loop_head: int                     # 循环头地址
    loop_body: List[int]               # 循环体BB列表
    iteration_count: int               # 已执行次数

    # 指令特征
    has_memory_write: bool = False     # 是否有内存写入
    has_memory_read: bool = False      # 是否有内存读取
    has_mmio_access: bool = False      # 是否有MMIO访问
    has_counter_increment: bool = False # 是否有计数器递增
    has_comparison: bool = False       # 是否有比较指令
    has_conditional_branch: bool = False # 是否有条件分支
    has_unconditional_branch: bool = False # 是否有无条件分支

    # 行为特征
    memory_addresses_accessed: Set[int] = None  # 访问的内存地址集合
    mmio_addresses_accessed: Set[int] = None    # 访问的MMIO地址集合
    register_changes: Dict[str, List[int]] = None  # 寄存器值变化历史

    # 退出条件
    has_exit_condition: bool = False   # 是否有退出条件
    exit_condition_type: str = ""      # 退出条件类型

    def __post_init__(self):
        if self.memory_addresses_accessed is None:
            self.memory_addresses_accessed = set()
        if self.mmio_addresses_accessed is None:
            self.mmio_addresses_accessed = set()
        if self.register_changes is None:
            self.register_changes = {}


class IntelligentLoopClassifier:
    """
    智能循环分类器

    核心创新：
    不是简单地"PC重复>N次就是死循环"，
    而是分析循环的语义特征，理解它在做什么。
    """

    def __init__(self, static_bbs: Dict[int, List[Dict]] = None):
        """
        初始化

        Args:
            static_bbs: 静态BB信息 {address: [instructions]}
        """
        self.static_bbs = static_bbs or {}
        self.enable_memory_wait = (
            os.environ.get("LSGEMU_ENABLE_MEMORY_WAIT_CLASSIFIER", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # 循环检测
        self.loop_heads: Dict[int, LoopCharacteristics] = {}
        self.bb_history: List[int] = []
        try:
            self.max_bb_history = max(
                256,
                int(os.environ.get("LSGEMU_LOOP_BB_HISTORY_LIMIT", "512")),
            )
        except ValueError:
            self.max_bb_history = 512
        self.total_bb_observations = 0
        self.bb_history_entries_discarded = 0
        self.instruction_to_bb = self._build_instruction_to_bb_lookup()
        self.static_bb_set = set(self.static_bbs.keys())
        self.active_loop_head: Optional[int] = None
        self.active_loop_body: List[int] = []
        self.active_loop_iterations: Dict[int, int] = {}
        self.sequence_cycle_heads: Set[int] = set()
        self.sequence_cycle_count_phases: Dict[int, int] = {}

        # 执行追踪
        self.memory_writes: Dict[int, List[Tuple[int, int]]] = {}  # pc -> [(addr, value), ...]
        self.memory_reads: Dict[int, List[int]] = {}  # pc -> [addr, ...]
        self.mmio_accesses: Dict[int, List[Tuple[int, bool, int]]] = {}  # pc -> [(addr, is_read, value), ...]
        self.mmio_addresses_by_pc: Dict[int, Set[int]] = {}
        try:
            self.max_mmio_accesses_per_pc = max(
                16,
                int(os.environ.get("LSGEMU_LOOP_MMIO_ACCESS_PC_LIMIT", "256")),
            )
        except ValueError:
            self.max_mmio_accesses_per_pc = 256
        self.total_mmio_observations = 0
        self.mmio_observations_discarded = 0
        try:
            self.max_memory_accesses_per_pc = max(
                16,
                int(os.environ.get("LSGEMU_LOOP_MEMORY_ACCESS_PC_LIMIT", "256")),
            )
        except ValueError:
            self.max_memory_accesses_per_pc = 256
        self.static_predecessors: Dict[int, Set[int]] = self._build_static_predecessors()
        self.register_snapshots: List[Dict[str, int]] = []
        try:
            self.max_register_snapshots = max(
                16,
                int(os.environ.get("LSGEMU_LOOP_REGISTER_HISTORY_LIMIT", "32")),
            )
        except ValueError:
            self.max_register_snapshots = 32
        self.total_register_snapshots = 0
        self.register_snapshots_discarded = 0
        self._last_logged_loop_state: Dict[int, Tuple[str, int]] = {}
        self.log_first_loop_classification = (
            os.environ.get("LSGEMU_LOG_FIRST_LOOP_CLASSIFICATION", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def record_execution(self, bb_addr: int, registers: Dict[str, int] = None):
        """
        记录执行

        Args:
            bb_addr: 当前BB地址
            registers: 寄存器状态
        """
        self.bb_history.append(bb_addr)
        self.total_bb_observations += 1
        if len(self.bb_history) > self.max_bb_history * 2:
            discarded = len(self.bb_history) - self.max_bb_history
            del self.bb_history[:discarded]
            self.bb_history_entries_discarded += discarded

        if registers:
            self.register_snapshots.append(registers.copy())
            self.total_register_snapshots += 1
            if len(self.register_snapshots) > self.max_register_snapshots * 2:
                discarded = len(self.register_snapshots) - self.max_register_snapshots
                del self.register_snapshots[:discarded]
                self.register_snapshots_discarded += discarded

        # 检测循环。这里统计的是“当前活跃循环”的连续迭代次数，
        # 不把同一库函数在不同调用点的历史命中累加到一起。
        self._detect_loop(bb_addr)

    def record_memory_access(self, pc: int, address: int, is_write: bool, value: int = 0):
        """记录内存访问"""
        if is_write:
            if pc not in self.memory_writes:
                self.memory_writes[pc] = []
            if len(self.memory_writes[pc]) >= self.max_memory_accesses_per_pc:
                return
            self.memory_writes[pc].append((address, value))
        else:
            if pc not in self.memory_reads:
                self.memory_reads[pc] = []
            if len(self.memory_reads[pc]) >= self.max_memory_accesses_per_pc:
                return
            self.memory_reads[pc].append(address)

    def wants_memory_access_sample(self, pc: int, is_write: bool) -> bool:
        records = self.memory_writes if is_write else self.memory_reads
        return len(records.get(int(pc), [])) < self.max_memory_accesses_per_pc

    def record_mmio_access(self, pc: int, address: int, is_read: bool, value: int):
        """记录MMIO访问"""
        pc = int(pc)
        address = int(address) & 0xFFFFFFFF
        if pc not in self.mmio_accesses:
            self.mmio_accesses[pc] = []
        self.total_mmio_observations += 1
        self.mmio_addresses_by_pc.setdefault(pc, set()).add(address)
        records = self.mmio_accesses[pc]
        if len(records) < self.max_mmio_accesses_per_pc:
            records.append((address, bool(is_read), int(value or 0) & 0xFFFFFFFF))
        else:
            self.mmio_observations_discarded += 1

    def _detect_loop(self, bb_addr: int):
        """
        检测循环

        只有真实回边才算一次循环迭代。旧逻辑只要当前BB出现在最近
        100个BB中就累加，会把短小可退出的memchr/查表循环在多次调用
        之间累加，最终误判成死循环。
        """
        if len(self.bb_history) < 2:
            return

        prev_bb = self.bb_history[-2]

        if self._is_loop_back_edge(prev_bb, bb_addr):
            loop_body = self._extract_current_loop_body(bb_addr)
            if bb_addr not in self.loop_heads:
                self.loop_heads[bb_addr] = LoopCharacteristics(
                    loop_head=bb_addr,
                    loop_body=loop_body,
                    iteration_count=0
                )

            if self.active_loop_head == bb_addr:
                self.active_loop_iterations[bb_addr] = self.active_loop_iterations.get(bb_addr, 0) + 1
            else:
                self.active_loop_head = bb_addr
                if self._find_recent_repeated_cycle():
                    self.active_loop_iterations[bb_addr] = self.active_loop_iterations.get(bb_addr, 0) + 1
                else:
                    self.active_loop_iterations[bb_addr] = 1

            self.active_loop_body = loop_body
            characteristics = self.loop_heads[bb_addr]
            characteristics.loop_body = loop_body
            characteristics.iteration_count = self.active_loop_iterations[bb_addr]
            return

        # If a syntactic back-edge loop is already active, do not also count the
        # same periodic body through suffix repetition at every phase.
        if (
            self.active_loop_head is not None
            and self.active_loop_head not in self.sequence_cycle_heads
            and bb_addr in self.active_loop_body
        ):
            return

        repeated_cycle = self._find_recent_repeated_cycle()
        if repeated_cycle:
            # Indirect/call-return cycles may have no syntactic backward edge.
            # Count them under one canonical head instead of the current phase
            # BB; otherwise a loop like A -> callee -> B -> A never reaches the
            # intervention threshold because each detected phase stays at 1.
            cycle_head = min(int(bb) for bb in repeated_cycle)
            loop_body = self._expand_loop_body_static(repeated_cycle, cycle_head)
            if cycle_head not in self.loop_heads:
                self.loop_heads[cycle_head] = LoopCharacteristics(
                    loop_head=cycle_head,
                    loop_body=loop_body,
                    iteration_count=0
                )

            count_phase = self.sequence_cycle_count_phases.setdefault(cycle_head, int(bb_addr))
            should_count_iteration = int(bb_addr) == int(count_phase)

            if self.active_loop_head == cycle_head:
                if should_count_iteration:
                    self.active_loop_iterations[cycle_head] = self.active_loop_iterations.get(cycle_head, 0) + 1
            else:
                self.active_loop_head = cycle_head
                self.active_loop_iterations[cycle_head] = 1 if should_count_iteration else 0
                self.sequence_cycle_heads.add(cycle_head)

            self.active_loop_body = loop_body
            characteristics = self.loop_heads[cycle_head]
            characteristics.loop_body = loop_body
            characteristics.iteration_count = self.active_loop_iterations[cycle_head]
            return

        # 当前BB已经离开活跃循环体，说明下一次再进入同一地址是新调用/新循环，
        # 迭代计数必须重新开始。
        if self.active_loop_head is not None and bb_addr not in self.active_loop_body:
            self.active_loop_head = None
            self.active_loop_body = []

    def _find_recent_repeated_cycle(self) -> List[int]:
        """Detect tight multi-BB cycles from dynamic history, including indirect edges."""
        history = self.bb_history
        max_cycle_len = min(16, len(history) // 2)
        # Prefer the longest stable suffix so bodies with repeated adjacent BBs
        # are not collapsed to a one-BB pseudo-loop.
        for cycle_len in range(max_cycle_len, 0, -1):
            current = history[-cycle_len:]
            previous = history[-2 * cycle_len:-cycle_len]
            if current and current == previous:
                if cycle_len > 1 and len(set(current)) <= 1:
                    continue
                seen = set()
                unique_body = []
                for bb in current:
                    if bb not in seen:
                        seen.add(bb)
                        unique_body.append(bb)
                return unique_body
        return []

    def _build_instruction_to_bb_lookup(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                address = insn.get("address")
                if address is not None:
                    lookup[address] = bb_addr
        return lookup

    def _build_static_predecessors(self) -> Dict[int, Set[int]]:
        predecessors: Dict[int, Set[int]] = {int(bb): set() for bb in self.static_bbs}
        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            last_insn = instructions[-1]
            mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic", ""))
            if not self._is_loop_branch_mnemonic(mnemonic):
                continue
            target = self._parse_branch_target(str(last_insn.get("operands", "")))
            target_bb = self._resolve_bb_start(target)
            if target_bb is not None and target_bb in predecessors:
                predecessors[target_bb].add(int(bb_addr))
        return predecessors

    def add_dynamic_basic_block(self, bb_addr: int, instructions: List[Dict[str, object]]) -> bool:
        """Register a runtime-discovered BB so loop classification can use it."""
        bb_addr = int(bb_addr) & ~1
        if not instructions or bb_addr in self.static_bbs:
            return False

        normalized_instructions = []
        for insn in instructions:
            try:
                normalized = dict(insn)
                normalized["address"] = int(normalized.get("address", 0) or 0) & ~1
                normalized["size"] = int(normalized.get("size", 2) or 2)
            except Exception:
                continue
            if normalized["address"]:
                normalized_instructions.append(normalized)
        if not normalized_instructions:
            return False

        self.static_bbs[bb_addr] = normalized_instructions
        self.static_bb_set.add(bb_addr)
        self.static_predecessors.setdefault(bb_addr, set())
        for insn in normalized_instructions:
            self.instruction_to_bb[int(insn["address"])] = bb_addr

        last_insn = normalized_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic", ""))
        if self._is_loop_branch_mnemonic(mnemonic):
            target = self._parse_branch_target(str(last_insn.get("operands", "")))
            target_bb = self._resolve_bb_start(target)
            if target_bb is not None:
                self.static_predecessors.setdefault(int(target_bb), set()).add(bb_addr)
        return True

    def _resolve_bb_start(self, address: Optional[int]) -> Optional[int]:
        if address is None:
            return None
        if address in self.static_bb_set:
            return address
        return self.instruction_to_bb.get(address)

    def _parse_branch_target(self, operands: str) -> Optional[int]:
        matches = re.findall(r"0x[0-9a-fA-F]+", str(operands or ""))
        if not matches:
            return None
        try:
            return int(matches[-1], 16)
        except ValueError:
            return None

    @staticmethod
    def _normalize_mnemonic(mnemonic: object) -> str:
        raw = str(mnemonic or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B":
            condition = parts[1]
            if condition not in {"N", "W", "NW"}:
                return f"B{condition}"
            return "B"
        return parts[0]

    @staticmethod
    def _is_loop_branch_mnemonic(mnemonic: str) -> bool:
        """True only for branch instructions that can form loop back-edges."""
        return mnemonic in {
            "B", "BAL",
            "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
            "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
            "CBZ", "CBNZ", "TBB", "TBH",
        }

    def _is_loop_back_edge(self, prev_bb: int, curr_bb: int) -> bool:
        instructions = self.static_bbs.get(prev_bb, [])
        if not instructions:
            return curr_bb <= prev_bb

        last_insn = instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic", ""))
        operands = str(last_insn.get("operands", ""))
        branch_like = self._is_loop_branch_mnemonic(mnemonic)
        if not branch_like:
            return False

        target = self._parse_branch_target(operands)
        target_bb = self._resolve_bb_start(target)
        if target_bb == curr_bb and curr_bb <= prev_bb:
            return True

        # Capstone/Ghidra对少数间接或表跳转无法直接给出目标时，保守地把
        # 地址回退的分支边视为回边。
        if curr_bb <= prev_bb:
            if target is None:
                return True
            try:
                fallthrough = int(last_insn.get("address", prev_bb) or prev_bb) + int(last_insn.get("size", 2) or 2)
            except Exception:
                fallthrough = int(prev_bb)
            if int(curr_bb) != int(fallthrough):
                return True
        return False

    def _extract_current_loop_body(self, loop_head: int) -> List[int]:
        recent_history = self.bb_history[-200:]
        positions = [i for i, bb in enumerate(recent_history[:-1]) if bb == loop_head]
        if not positions:
            return self._expand_loop_body_static([loop_head], loop_head)

        start = positions[-1]
        loop_body = recent_history[start:-1]
        seen = set()
        unique_body = []
        for bb in loop_body:
            if bb not in seen:
                seen.add(bb)
                unique_body.append(bb)
        return self._expand_loop_body_static(unique_body if unique_body else [loop_head], loop_head)

    def _expand_loop_body_static(self, loop_body: List[int], loop_head: int) -> List[int]:
        """
        Complete tiny runtime loop bodies using static back-edge predecessors.

        Runtime history can report only the loop head when the back edge is
        head <- body, so the classifier misses load/store/update instructions
        in the body BB and classifies finite memmove/memchr/malloc loops as
        UNKNOWN. Add direct static predecessors that branch back to the head.
        """
        ordered: List[int] = []
        seen: Set[int] = set()
        for bb in loop_body:
            if bb not in seen:
                seen.add(bb)
                ordered.append(bb)

        queue = list(self.static_predecessors.get(loop_head, set()))
        while queue and len(ordered) < 16:
            pred = queue.pop(0)
            if pred in seen:
                continue
            # Only fold local backward edges into the loop body; this avoids
            # pulling in distant callers that merely branch to a common block.
            if abs(int(pred) - int(loop_head)) > 0x100:
                continue
            seen.add(pred)
            ordered.append(pred)
            for parent in sorted(self.static_predecessors.get(pred, set())):
                if parent not in seen and abs(int(parent) - int(loop_head)) <= 0x100:
                    queue.append(parent)
        return sorted(ordered)

    def classify_loop(self, loop_head: int) -> LoopType:
        """
        分类循环

        这是核心方法：通过分析循环的语义特征，判断循环类型

        Args:
            loop_head: 循环头地址

        Returns:
            循环类型
        """
        if loop_head not in self.loop_heads:
            return LoopType.UNKNOWN

        characteristics = self.loop_heads[loop_head]

        # 分析循环体的指令
        self._analyze_loop_instructions(characteristics)

        # 分析循环的行为
        self._analyze_loop_behavior(characteristics)

        # 根据特征分类
        loop_type = self._determine_loop_type(characteristics)

        if self._should_log_loop(loop_head, loop_type, characteristics.iteration_count):
            logger.info(f"\n循环分类: 0x{loop_head:08x}")
            logger.info(f"  类型: {loop_type.value}")
            logger.info(f"  迭代次数: {characteristics.iteration_count}")
            logger.info(f"  特征:")
            logger.info(f"    - 内存写入: {characteristics.has_memory_write}")
            logger.info(f"    - 内存读取: {characteristics.has_memory_read}")
            logger.info(f"    - MMIO访问: {characteristics.has_mmio_access}")
            logger.info(f"    - 计数器递增: {characteristics.has_counter_increment}")
            logger.info(f"    - 条件分支: {characteristics.has_conditional_branch}")
            logger.info(f"    - 退出条件: {characteristics.has_exit_condition}")

        return loop_type

    def _should_log_loop(self, loop_head: int, loop_type: LoopType, iteration: int) -> bool:
        """Throttle loop classification logs; classification runs on every BB."""
        previous = self._last_logged_loop_state.get(loop_head)
        if previous is None or previous[0] != loop_type.value:
            self._last_logged_loop_state[loop_head] = (loop_type.value, iteration)
            if self.log_first_loop_classification:
                return True
            return iteration in {20, 100} or (iteration > 0 and iteration % 1000 == 0)
        if iteration in {20, 100} or iteration % 1000 == 0:
            if previous[1] != iteration:
                self._last_logged_loop_state[loop_head] = (loop_type.value, iteration)
                return True
        return False

    def _analyze_loop_instructions(self, characteristics: LoopCharacteristics):
        """
        分析循环体的指令

        提取循环的静态特征

        改进：分析完整的循环体，而不仅仅是循环头
        """
        loop_head = characteristics.loop_head

        # 获取循环体的所有BB
        loop_body_bbs = self._get_loop_body_bbs(loop_head)

        # 分析所有BB的指令
        for bb_addr in loop_body_bbs:
            instructions = self.static_bbs.get(bb_addr, [])

            for insn in instructions:
                mnemonic = self._normalize_mnemonic(insn.get('mnemonic', ''))
                operands = insn.get('operands', '')

                # 检测内存访问。Capstone/Ghidra 可能把 Thumb 宽指令写成
                # ldrb.w/strb.w，_normalize_mnemonic() 会去掉 .w 后缀。
                if mnemonic in [
                    'LDR', 'LDRB', 'LDRH', 'LDRSB', 'LDRSH', 'LDRD',
                    'LDM', 'LDMIA', 'POP',
                ]:
                    characteristics.has_memory_read = True

                if mnemonic in ['STR', 'STRB', 'STRH', 'STRD', 'STM', 'STMIA', 'PUSH']:
                    characteristics.has_memory_write = True

                # 检测计数器/指针更新。很多 libc/flash 扫描循环不是 add #1，
                # 而是 pre/post-indexed load/store，例如 [r3, #-1]! 或 [r1], #1。
                if mnemonic in ['ADDS', 'ADD', 'SUBS', 'SUB', 'RSB', 'RSBS']:
                    if (
                        '#' in str(operands)
                        or re.search(r'\br\d+\s*,\s*r\d+\s*,\s*r\d+', str(operands or ''), re.I)
                    ):
                        characteristics.has_counter_increment = True

                # 检测 write-back / post-index 模式。
                operand_text = str(operands or '')
                if (
                    ('[' in operand_text and ']' in operand_text and '!' in operand_text)
                    or re.search(r'\]\s*,\s*#?-?0x?[0-9a-fA-F]+', operand_text)
                ):
                    characteristics.has_counter_increment = True

                # 检测比较
                if mnemonic in ['CMP', 'TST', 'TEQ', 'CMN']:
                    characteristics.has_comparison = True

                # 检测分支（包含所有ARM条件分支指令）
                if mnemonic in ['BEQ', 'BNE', 'BCC', 'BCS', 'BGT', 'BLT', 'BGE', 'BLE',
                               'CBZ', 'CBNZ', 'BLO', 'BLS', 'BHI', 'BHS', 'BMI', 'BPL',
                               'BVS', 'BVC', 'BAL']:
                    characteristics.has_conditional_branch = True
                    characteristics.has_exit_condition = True

                if mnemonic == 'B' and len(instructions) <= 2:
                    characteristics.has_unconditional_branch = True

    def _get_loop_body_bbs(self, loop_head: int) -> List[int]:
        """
        获取循环体的所有BB

        通过分析最近的执行历史，找到循环体包含的所有BB
        """
        if loop_head not in self.loop_heads:
            return [loop_head]

        characteristics = self.loop_heads[loop_head]
        if characteristics.loop_body:
            return self._expand_loop_body_static(characteristics.loop_body, loop_head)

        # 在最近的历史中找到循环模式
        recent_history = self.bb_history[-100:]

        # 找到循环头的所有出现位置
        loop_head_positions = [i for i, bb in enumerate(recent_history) if bb == loop_head]

        if len(loop_head_positions) < 2:
            return self._expand_loop_body_static([loop_head], loop_head)

        # 提取两次循环头之间的BB序列
        start = loop_head_positions[-2]
        end = loop_head_positions[-1]

        loop_body = recent_history[start:end]

        # 去重并保持顺序
        seen = set()
        unique_body = []
        for bb in loop_body:
            if bb not in seen:
                seen.add(bb)
                unique_body.append(bb)

        return self._expand_loop_body_static(unique_body if unique_body else [loop_head], loop_head)

    def _analyze_loop_behavior(self, characteristics: LoopCharacteristics):
        """
        分析循环的运行时行为

        提取循环的动态特征

        改进：检查整个循环体的MMIO访问，而不仅仅是循环头
        关键修复：MMIO访问的PC可能是BB内的任意指令，不一定是BB头
        """
        loop_head = characteristics.loop_head

        # r36 Q5（默认关）：`LSGEMU_LOOP_MMIO_STRICT_WINDOW=1` 时，
        # ①每次分析先把 has_mmio_access 归零重算——旧行为只在命中时
        # 赋 True、从不回落，标志在多次分析间粘滞（r35 T2 实测 memset
        # 环 0x0813a9ec 标志 True 而当前窗口 mmio_accesses 0 条）；
        # ②BB 结束地址改用最后一条指令的真实 end——旧的
        # `bb_addr + len(insns)*4` 按每条 4 字节高估，会把紧随其后的
        # 外函数 MMIO PC 算进本环体。两者都只影响「本环是否有 MMIO」
        # 的惰性判定，不改任何分类阈值。
        strict_window = _env_flag("LSGEMU_LOOP_MMIO_STRICT_WINDOW", "0")
        if strict_window:
            characteristics.has_mmio_access = False
            characteristics.mmio_addresses_accessed = set()

        # 获取循环体的所有BB
        loop_body_bbs = self._get_loop_body_bbs(loop_head)

        # 检查循环体中所有BB的MMIO访问
        # 关键：需要检查BB内的所有指令地址，不仅仅是BB头
        for bb_addr in loop_body_bbs:
            # 获取BB的指令
            bb_instructions = self.static_bbs.get(bb_addr, [])

            # 计算BB的结束地址（粗略估计）
            # 假设每条指令2-4字节（Thumb模式）
            if strict_window and bb_instructions:
                # r36 Q5：真实 end = max(addr+size)。指令缺 size 时退回
                # 旧的高估口径（宁可误报也不缩小窗口漏报）。
                try:
                    bb_end = max(
                        int(insn.get("address", bb_addr) or bb_addr)
                        + int(insn.get("size", 4) or 4)
                        for insn in bb_instructions
                    )
                except (TypeError, ValueError):
                    bb_end = bb_addr + len(bb_instructions) * 4
            else:
                bb_end = bb_addr + len(bb_instructions) * 4

            # 检查所有MMIO访问记录
            for pc, accesses in self.mmio_accesses.items():
                # 如果PC在这个BB的范围内
                if bb_addr <= pc < bb_end:
                    addresses = self.mmio_addresses_by_pc.get(int(pc), set())
                    if not addresses:
                        # Compatibility with classifiers restored from older
                        # serialized state that predates the address index.
                        addresses = {
                            int(item[0])
                            for item in accesses
                            if isinstance(item, (tuple, list)) and item
                        }
                    characteristics.has_mmio_access = bool(addresses)
                    for addr in addresses:
                        characteristics.mmio_addresses_accessed.add(int(addr))

        # 检查内存访问模式
        for bb_addr in loop_body_bbs:
            bb_instructions = self.static_bbs.get(bb_addr, [])
            bb_end = bb_addr + len(bb_instructions) * 4

            for pc, writes in self.memory_writes.items():
                if bb_addr <= pc < bb_end:
                    for addr, value in writes:
                        characteristics.memory_addresses_accessed.add(addr)

            for pc, reads in self.memory_reads.items():
                if bb_addr <= pc < bb_end:
                    for addr in reads:
                        characteristics.memory_addresses_accessed.add(addr)

        # 分析寄存器变化
        if len(self.register_snapshots) >= 2:
            # 检查寄存器是否在变化
            recent_snapshots = self.register_snapshots[-10:]
            for reg in ['r0', 'r1', 'r2', 'r3']:
                values = [s.get(reg, 0) for s in recent_snapshots if reg in s]
                if len(set(values)) > 1:  # 寄存器值在变化
                    characteristics.register_changes[reg] = values

    def _determine_loop_type(self, characteristics: LoopCharacteristics) -> LoopType:
        """
        根据特征判断循环类型

        这是智能分类的核心逻辑
        """
        # 规则1: 死循环（Error_Handler）
        # 特征：无条件分支到自己，没有退出条件
        if characteristics.has_unconditional_branch and not characteristics.has_exit_condition:
            return LoopType.DEADLOCK

        has_progress = (
            characteristics.has_counter_increment
            or bool(characteristics.register_changes)
        )

        # 轮询循环优先于初始化循环识别。很多驱动等待循环会在读取
        # status MMIO 的同时维护少量RAM状态/计数器；如果先按
        # "memory_write + counter" 归类为 initialization，就永远不会
        # 触发 MMIO 求解。
        if (
            characteristics.has_mmio_access
            and characteristics.has_conditional_branch
            and (
                not characteristics.has_memory_write
                or len(characteristics.memory_addresses_accessed) <= 8
            )
        ):
            return LoopType.POLLING

        # 只读、小地址集合的软件状态等待不能当作“有限初始化扫描”
        # 放任到千万次。典型形态是等待队列/flag/函数返回状态变化，
        # 需要交给 memory_wait 的短循环约束求解。
        if (
            self.enable_memory_wait
            and characteristics.has_memory_read
            and characteristics.has_exit_condition
            and not characteristics.has_memory_write
            and not characteristics.has_mmio_access
            and 0 < len(characteristics.memory_addresses_accessed) <= 4
        ):
            return LoopType.MEMORY_WAIT

        # Some ARM7/ARM9-style MCU/SoC images map peripheral registers outside
        # the usual Cortex-M 0x40000000 MMIO window. They look like ordinary
        # memory because the rehoster maps the page on first access:
        # write control bits, read the same small status register set, then
        # branch on flags. Do not classify these as finite initialization just
        # because a scratch register changes across iterations.
        if (
            self.enable_memory_wait
            and characteristics.has_memory_read
            and characteristics.has_memory_write
            and characteristics.has_exit_condition
            and not characteristics.has_mmio_access
            and not characteristics.has_counter_increment
            and 0 < len(characteristics.memory_addresses_accessed) <= 8
        ):
            return LoopType.MEMORY_WAIT

        # 规则2: 初始化/有限数据循环
        # 特征：有内存写入 + 有计数器递增 + 有条件分支（退出条件）
        if (characteristics.has_memory_write and
            has_progress and
            characteristics.has_exit_condition):
            # 进一步检查：访问的内存地址是否在增长
            # 如果没有运行时数据，也认为是初始化循环（基于静态特征）
            if len(characteristics.memory_addresses_accessed) > 1 or len(characteristics.memory_addresses_accessed) == 0:
                return LoopType.INITIALIZATION

        # 规则2.1: 只读扫描循环，例如 memchr/strstr/flash-record scan。
        # 这类循环有明确退出条件和指针/索引进展，但没有写入；按 UNKNOWN@100
        # 截断会导致 setup() 或 parser continuation 无法返回。
        if (
            characteristics.has_memory_read
            and characteristics.has_exit_condition
            and has_progress
            and not characteristics.has_mmio_access
        ):
            return LoopType.INITIALIZATION

        # 规则2.5: 栈变量计数延迟循环。
        # 编译器常把 for/while 延迟计数器spill到栈上，表现为同一栈地址
        # 反复读写 + counter increment + 条件分支。它不是外设等待，不应
        # 100次后回滚；让它按入口路径自然结束即可。
        if (
            characteristics.has_memory_write
            and characteristics.has_memory_read
            and characteristics.has_counter_increment
            and characteristics.has_exit_condition
            and not characteristics.has_mmio_access
            and 0 < len(characteristics.memory_addresses_accessed) <= 4
        ):
            return LoopType.DELAY

        # 规则3: 轮询循环
        # 特征：有MMIO访问 + 有条件分支 + 没有内存写入。
        if (characteristics.has_mmio_access and
            characteristics.has_conditional_branch and
            not characteristics.has_memory_write):
            return LoopType.POLLING

        # 规则3.5: 普通内存等待循环
        # 特征：有内存读取 + 有条件分支 + 没有内存写入 + 没有MMIO访问。
        # 这类循环常见于环形队列、任务标志、软件缓冲区等待，很多会自然退出，
        # 不应按 UNKNOWN@100 过早干预。
        if (self.enable_memory_wait and
            characteristics.has_memory_read and
            characteristics.has_conditional_branch and
            not characteristics.has_memory_write and
            not characteristics.has_mmio_access):
            return LoopType.MEMORY_WAIT

        # 规则4: 延迟循环
        # 特征：有计数器递增 + 有条件分支 + 没有内存访问 + 没有MMIO访问
        if (characteristics.has_counter_increment and
            characteristics.has_conditional_branch and
            not characteristics.has_memory_write and
            not characteristics.has_memory_read and
            not characteristics.has_mmio_access):
            return LoopType.DELAY

        return LoopType.UNKNOWN

    def should_intervene(self, loop_head: int) -> Tuple[bool, str]:
        """
        判断是否需要干预

        新策略：对所有超过阈值的循环都干预，不只是polling

        Args:
            loop_head: 循环头地址

        Returns:
            (是否需要干预, 原因)
        """
        if loop_head not in self.loop_heads:
            return False, ""

        characteristics = self.loop_heads[loop_head]
        loop_type = self.classify_loop(loop_head)
        iteration = characteristics.iteration_count

        try:
            deadlock_threshold = max(1, int(os.environ.get("LSGEMU_DEADLOCK_LOOP_THRESHOLD", "20")))
        except ValueError:
            deadlock_threshold = 20
        try:
            memory_wait_threshold = max(1, int(os.environ.get("LSGEMU_MEMORY_WAIT_LOOP_THRESHOLD", "100")))
        except ValueError:
            memory_wait_threshold = 100
        try:
            delay_threshold = max(1, int(os.environ.get("LSGEMU_DELAY_LOOP_THRESHOLD", "100000")))
        except ValueError:
            delay_threshold = 100000
        try:
            initialization_threshold = max(1, int(os.environ.get("LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "10000000")))
        except ValueError:
            initialization_threshold = 10000000
        try:
            unknown_threshold = max(1, int(os.environ.get("LSGEMU_UNKNOWN_LOOP_THRESHOLD", "1000")))
        except ValueError:
            unknown_threshold = 1000

        thresholds = {
            LoopType.POLLING: 100,
            LoopType.MEMORY_WAIT: memory_wait_threshold,
            LoopType.DEADLOCK: deadlock_threshold,
            LoopType.UNKNOWN: unknown_threshold,
            LoopType.DELAY: delay_threshold,
            LoopType.INITIALIZATION: initialization_threshold,
        }
        threshold = thresholds.get(loop_type, 100)

        if iteration >= threshold:
            return True, f"{loop_type.value}循环迭代{iteration}次 @ 0x{loop_head:08x}"

        return False, ""

    def get_intervention_strategy(self, loop_head: int) -> Dict:
        """
        获取干预策略

        新策略（更简单）：
        所有超过100次的循环都使用LLM推断

        Returns:
            {
                "action": "llm_inference",
                "use_llm": True,
                "reason": str
            }
        """
        if loop_head not in self.loop_heads:
            return {"action": "skip", "reason": "Loop not found"}

        loop_type = self.classify_loop(loop_head)
        characteristics = self.loop_heads[loop_head]

        if loop_type == LoopType.INITIALIZATION:
            return {
                "action": "check_config",
                "reason": f"Initialization loop exceeded {characteristics.iteration_count} iterations"
            }

        if loop_type == LoopType.DELAY:
            return {
                "action": "skip",
                "reason": f"Delay loop exceeded {characteristics.iteration_count} iterations"
            }

        # 特殊处理：polling循环优先尝试MMIO推断
        if loop_type == LoopType.POLLING and characteristics.mmio_addresses_accessed:
            return {
                "action": "adjust_mmio",
                "mmio_addresses": list(characteristics.mmio_addresses_accessed),
                "suggested_values": [0x1, 0x80, 0x100],
                "reason": "Polling loop - try MMIO inference first"
            }

        if loop_type == LoopType.MEMORY_WAIT:
            return {
                "action": "rollback",
                "use_llm": False,
                "rollback_levels": 1,
                "reason": (
                    f"Memory-wait loop exceeded {characteristics.iteration_count} iterations; "
                    "prefer fast rollback over expensive LLM analysis"
                )
            }

        if (
            loop_type == LoopType.UNKNOWN
            and characteristics.has_memory_read
            and not characteristics.has_memory_write
            and not characteristics.mmio_addresses_accessed
        ):
            return {
                "action": "rollback",
                "use_llm": False,
                "rollback_levels": 1,
                "reason": (
                    f"Unknown RAM wait loop exceeded {characteristics.iteration_count} iterations; "
                    "prefer fast rollback over expensive LLM analysis"
                )
            }

        # 其他所有循环：使用LLM推断
        return {
            "action": "llm_inference",
            "use_llm": True,
            "rollback_levels": 2,
            "reason": f"Loop exceeded 100 iterations ({characteristics.iteration_count}), use LLM inference"
        }

    def get_statistics(self) -> Dict[str, object]:
        """Return compact loop diagnostics for reports and low-coverage triage."""
        by_type = {loop_type.value: 0 for loop_type in LoopType}
        hot_loops = []
        for loop_head, characteristics in self.loop_heads.items():
            loop_type = self.classify_loop(loop_head)
            by_type[loop_type.value] = by_type.get(loop_type.value, 0) + 1
            hot_loops.append({
                "loop_head": f"0x{int(loop_head) & 0xFFFFFFFF:08x}",
                "type": loop_type.value,
                "iterations": int(characteristics.iteration_count),
                "has_memory_read": bool(characteristics.has_memory_read),
                "has_memory_write": bool(characteristics.has_memory_write),
                "has_mmio_access": bool(characteristics.has_mmio_access),
                "has_counter_increment": bool(characteristics.has_counter_increment),
                "has_conditional_branch": bool(characteristics.has_conditional_branch),
                "mmio_addresses": [
                    f"0x{int(addr) & 0xFFFFFFFF:08x}"
                    for addr in sorted(characteristics.mmio_addresses_accessed)[:8]
                ],
                "memory_address_count": len(characteristics.memory_addresses_accessed),
            })
        hot_loops.sort(key=lambda item: int(item["iterations"]), reverse=True)
        return {
            "total_loops": len(self.loop_heads),
            "by_type": by_type,
            "memory_wait_classifier_enabled": bool(self.enable_memory_wait),
            "history_retention": {
                "bb_limit": int(self.max_bb_history),
                "bb_retained": len(self.bb_history),
                "bb_total": int(self.total_bb_observations),
                "bb_discarded": int(self.bb_history_entries_discarded),
                "register_limit": int(self.max_register_snapshots),
                "register_retained": len(self.register_snapshots),
                "register_total": int(self.total_register_snapshots),
                "register_discarded": int(self.register_snapshots_discarded),
                "mmio_limit_per_pc": int(self.max_mmio_accesses_per_pc),
                "mmio_total": int(self.total_mmio_observations),
                "mmio_retained": sum(len(items) for items in self.mmio_accesses.values()),
                "mmio_discarded": int(self.mmio_observations_discarded),
            },
            "top_hot_loops": hot_loops[:16],
        }
