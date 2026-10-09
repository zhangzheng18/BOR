"""
路径探索器 - 分支变异功能

策略：
1. 记录执行路径中的所有分支点
2. 从最深的分支点开始，翻转分支条件
3. 探索另一个分支
4. 回溯到前一个分支，继续探索
5. 最终探索完所有可达路径
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BranchType(Enum):
    """分支类型"""
    CONDITIONAL = "conditional"  # 条件分支 (BEQ, BNE, etc.)
    UNCONDITIONAL = "unconditional"  # 无条件分支 (B)
    CALL = "call"  # 函数调用 (BL, BLX)
    RETURN = "return"  # 返回 (BX LR)


@dataclass
class BranchPoint:
    """分支点信息"""
    address: int  # 分支指令地址
    branch_type: BranchType  # 分支类型
    condition: str  # 分支条件 (EQ, NE, LT, etc.)
    taken: bool  # 是否跳转
    target_address: int  # 目标地址
    fallthrough_address: int  # 不跳转时的地址

    # 执行状态
    cpsr: int  # CPSR寄存器值
    registers: Dict[int, int]  # 相关寄存器值

    # 快照信息
    snapshot_id: int  # 快照ID
    path_id: str  # 路径ID
    depth: int  # 深度


@dataclass
class ExplorationPath:
    """探索路径"""
    path_id: str  # 路径ID
    branch_sequence: List[Tuple[int, bool]]  # 分支序列 [(address, taken), ...]
    coverage: int  # 覆盖的BB数量
    explored: bool  # 是否已探索


class PathExplorer:
    """
    路径探索器

    实现分支变异和路径探索功能
    """

    def __init__(self, emulator):
        """
        初始化

        Args:
            emulator: 仿真器实例
        """
        self.emulator = emulator

        # 分支点记录
        self.branch_points: List[BranchPoint] = []
        self.current_path_branches: List[Tuple[int, bool]] = []

        # 路径管理
        self.explored_paths: Dict[str, ExplorationPath] = {}
        self.pending_paths: List[ExplorationPath] = []

        # 统计
        self.total_branches = 0
        self.flipped_branches = 0
        self.new_coverage = 0

        # 配置
        self.max_depth = 100  # 最大分支深度
        self.max_paths = 1000  # 最大路径数

    def record_branch(self, address: int, insn_mnemonic: str,
                     taken: bool, target: int, fallthrough: int) -> None:
        """
        记录分支点

        Args:
            address: 分支指令地址
            insn_mnemonic: 指令助记符
            taken: 是否跳转
            target: 目标地址
            fallthrough: 不跳转时的地址
        """
        # 判断分支类型
        branch_type = self._get_branch_type(insn_mnemonic)

        # 只记录条件分支
        if branch_type != BranchType.CONDITIONAL:
            return

        # 提取条件
        condition = self._extract_condition(insn_mnemonic)

        # 读取CPSR和相关寄存器
        try:
            from unicorn.arm_const import UC_ARM_REG_CPSR
            cpsr = self.emulator.uc.reg_read(UC_ARM_REG_CPSR)

            # 读取R0-R3（常用的比较寄存器）
            from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3
            registers = {
                0: self.emulator.uc.reg_read(UC_ARM_REG_R0),
                1: self.emulator.uc.reg_read(UC_ARM_REG_R1),
                2: self.emulator.uc.reg_read(UC_ARM_REG_R2),
                3: self.emulator.uc.reg_read(UC_ARM_REG_R3),
            }
        except Exception:
            cpsr = 0
            registers = {}

        # 创建分支点
        branch = BranchPoint(
            address=address,
            branch_type=branch_type,
            condition=condition,
            taken=taken,
            target_address=target,
            fallthrough_address=fallthrough,
            cpsr=cpsr,
            registers=registers,
            snapshot_id=len(self.emulator.snapshot_manager.snapshots),
            path_id=self._get_current_path_id(),
            depth=len(self.current_path_branches)
        )

        self.branch_points.append(branch)
        self.current_path_branches.append((address, taken))
        self.total_branches += 1

        logger.debug(f"记录分支: 0x{address:08x} {condition} -> {'taken' if taken else 'not taken'}")

    def find_next_branch_to_flip(self) -> Optional[BranchPoint]:
        """
        找到下一个要翻转的分支

        策略：从最深的分支开始，找到第一个未探索的分支

        Returns:
            要翻转的分支点，如果没有则返回None
        """
        # 从后往前遍历（最深的分支优先）
        for branch in reversed(self.branch_points):
            # 检查这个分支的另一个方向是否已探索
            flipped_path_id = self._get_flipped_path_id(branch)

            if flipped_path_id not in self.explored_paths:
                logger.info(f"找到未探索分支: 0x{branch.address:08x} {branch.condition}")
                return branch

        return None

    def flip_branch(self, branch: BranchPoint) -> bool:
        """
        翻转分支条件

        Args:
            branch: 要翻转的分支点

        Returns:
            是否成功翻转
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"翻转分支")
        logger.info(f"{'='*80}")
        logger.info(f"地址: 0x{branch.address:08x}")
        logger.info(f"条件: {branch.condition}")
        logger.info(f"原方向: {'taken' if branch.taken else 'not taken'}")
        logger.info(f"新方向: {'taken' if not branch.taken else 'not taken'}")

        # 1. 恢复到分支点的快照
        if not self._restore_to_branch(branch):
            logger.error("恢复快照失败")
            return False

        # 2. 修改CPSR标志位，强制走另一个分支
        if not self._force_branch_direction(branch, not branch.taken):
            logger.error("修改分支方向失败")
            return False

        self.flipped_branches += 1
        logger.info(f"✓ 分支翻转成功")

        return True

    def _get_branch_type(self, mnemonic: str) -> BranchType:
        """获取分支类型"""
        mnemonic_upper = mnemonic.upper()

        if mnemonic_upper in ['BL', 'BLX']:
            return BranchType.CALL
        elif mnemonic_upper == 'BX' or 'BX LR' in mnemonic_upper:
            return BranchType.RETURN
        elif mnemonic_upper == 'B':
            return BranchType.UNCONDITIONAL
        elif mnemonic_upper.startswith('B'):
            return BranchType.CONDITIONAL
        else:
            return BranchType.UNCONDITIONAL

    def _extract_condition(self, mnemonic: str) -> str:
        """提取分支条件"""
        mnemonic_upper = mnemonic.upper()

        # 条件分支的条件码
        conditions = ['EQ', 'NE', 'CS', 'CC', 'MI', 'PL', 'VS', 'VC',
                     'HI', 'LS', 'GE', 'LT', 'GT', 'LE', 'AL']

        for cond in conditions:
            if mnemonic_upper.startswith('B' + cond):
                return cond

        return 'AL'  # 无条件

    def _get_current_path_id(self) -> str:
        """获取当前路径ID"""
        # 路径ID = 分支序列的字符串表示
        return '_'.join([f"{addr:08x}_{int(taken)}"
                        for addr, taken in self.current_path_branches])

    def _get_flipped_path_id(self, branch: BranchPoint) -> str:
        """获取翻转后的路径ID"""
        # 复制当前路径，翻转指定分支
        flipped_branches = self.current_path_branches[:branch.depth]
        flipped_branches.append((branch.address, not branch.taken))

        return '_'.join([f"{addr:08x}_{int(taken)}"
                        for addr, taken in flipped_branches])

    def _restore_to_branch(self, branch: BranchPoint) -> bool:
        """恢复到分支点"""
        try:
            # 找到对应的快照
            if branch.snapshot_id < len(self.emulator.snapshot_manager.snapshots):
                snapshot = self.emulator.snapshot_manager.snapshots[branch.snapshot_id]
                return bool(
                    self.emulator.snapshot_manager.restore_snapshot(
                        self.emulator.uc,
                        snapshot,
                    )
                )
            else:
                logger.error(f"快照不存在: {branch.snapshot_id}")
                return False
        except Exception as e:
            logger.error(f"恢复快照失败: {e}")
            return False

    def _force_branch_direction(self, branch: BranchPoint, take: bool) -> bool:
        """
        强制分支方向

        通过修改CPSR标志位来改变分支结果

        Args:
            branch: 分支点
            take: True=跳转, False=不跳转

        Returns:
            是否成功
        """
        try:
            from unicorn.arm_const import UC_ARM_REG_CPSR

            # 读取当前CPSR
            cpsr = self.emulator.uc.reg_read(UC_ARM_REG_CPSR)

            # CPSR标志位
            # N (bit 31): Negative
            # Z (bit 30): Zero
            # C (bit 29): Carry
            # V (bit 28): Overflow

            # 根据条件修改标志位
            new_cpsr = self._modify_cpsr_for_condition(cpsr, branch.condition, take)

            # 写入新的CPSR
            self.emulator.uc.reg_write(UC_ARM_REG_CPSR, new_cpsr)

            logger.debug(f"CPSR: 0x{cpsr:08x} -> 0x{new_cpsr:08x}")

            return True

        except Exception as e:
            logger.error(f"修改CPSR失败: {e}")
            return False

    def _modify_cpsr_for_condition(self, cpsr: int, condition: str, take: bool) -> int:
        """
        修改CPSR以满足条件

        Args:
            cpsr: 当前CPSR值
            condition: 条件码
            take: 是否要跳转

        Returns:
            修改后的CPSR
        """
        # 标志位掩码
        N_FLAG = 1 << 31
        Z_FLAG = 1 << 30
        C_FLAG = 1 << 29
        V_FLAG = 1 << 28

        new_cpsr = cpsr

        if condition == 'EQ':  # Equal (Z=1)
            if take:
                new_cpsr |= Z_FLAG  # 设置Z
            else:
                new_cpsr &= ~Z_FLAG  # 清除Z

        elif condition == 'NE':  # Not Equal (Z=0)
            if take:
                new_cpsr &= ~Z_FLAG  # 清除Z
            else:
                new_cpsr |= Z_FLAG  # 设置Z

        elif condition == 'GT':  # Greater Than (Z=0 and N=V)
            if take:
                new_cpsr &= ~Z_FLAG  # 清除Z
                # 确保N=V
                n = (new_cpsr & N_FLAG) != 0
                v = (new_cpsr & V_FLAG) != 0
                if n != v:
                    new_cpsr ^= V_FLAG  # 翻转V使N=V
            else:
                new_cpsr |= Z_FLAG  # 设置Z

        elif condition == 'LE':  # Less or Equal (Z=1 or N!=V)
            if take:
                new_cpsr |= Z_FLAG  # 设置Z
            else:
                new_cpsr &= ~Z_FLAG  # 清除Z
                # 确保N=V
                n = (new_cpsr & N_FLAG) != 0
                v = (new_cpsr & V_FLAG) != 0
                if n != v:
                    new_cpsr ^= V_FLAG  # 翻转V使N=V

        elif condition == 'LT':  # Less Than (N!=V)
            n = (new_cpsr & N_FLAG) != 0
            v = (new_cpsr & V_FLAG) != 0
            if take:
                if n == v:
                    new_cpsr ^= V_FLAG  # 翻转V使N!=V
            else:
                if n != v:
                    new_cpsr ^= V_FLAG  # 翻转V使N=V

        elif condition == 'GE':  # Greater or Equal (N=V)
            n = (new_cpsr & N_FLAG) != 0
            v = (new_cpsr & V_FLAG) != 0
            if take:
                if n != v:
                    new_cpsr ^= V_FLAG  # 翻转V使N=V
            else:
                if n == v:
                    new_cpsr ^= V_FLAG  # 翻转V使N!=V

        elif condition == 'HI':  # Higher (C=1 and Z=0)
            if take:
                new_cpsr |= C_FLAG  # 设置C
                new_cpsr &= ~Z_FLAG  # 清除Z
            else:
                new_cpsr &= ~C_FLAG  # 清除C

        elif condition == 'LS':  # Lower or Same (C=0 or Z=1)
            if take:
                new_cpsr &= ~C_FLAG  # 清除C
            else:
                new_cpsr |= C_FLAG  # 设置C
                new_cpsr &= ~Z_FLAG  # 清除Z

        return new_cpsr

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_branches': self.total_branches,
            'flipped_branches': self.flipped_branches,
            'explored_paths': len(self.explored_paths),
            'pending_paths': len(self.pending_paths),
            'new_coverage': self.new_coverage
        }
