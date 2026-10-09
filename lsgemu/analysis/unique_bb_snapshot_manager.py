#!/usr/bin/env python3
"""
唯一BB快照管理器

核心策略：
1. 每次遇到新的唯一BB时，保存快照
2. 保留最近N个唯一BB的快照（FIFO队列）
3. 检测到死循环时，逐级回退
4. 结合LLM分析最近的BB指令序列

这样可以精确定位到导致错误的分支点
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Deque
from collections import Counter, deque
import hashlib
import logging
import os
import time

from .snapshot_memory import (
    PagedMemory,
    SnapshotCaptureError,
    SnapshotIntegrityError,
    SnapshotPageStore,
)

logger = logging.getLogger(__name__)


@dataclass
class UniqueBBSnapshot:
    """
    唯一BB快照

    每次遇到新的唯一BB时保存
    """
    snapshot_id: int
    bb_address: int                 # BB地址
    timestamp: float                # 创建时间
    instruction_count: int          # 指令计数

    # CPU state
    cpu_state: Dict[str, int] = field(default_factory=dict)
    pc: int = 0
    flags: int = 0

    # Memory (RAM)
    memory_regions: Dict[Tuple[int, int], bytes | PagedMemory] = field(default_factory=dict)

    # MMIO state
    mmio_values: Dict[int, int] = field(default_factory=dict)
    mmio_access_history: List[Tuple[int, int, bool, int]] = field(default_factory=list)

    # Execution metadata
    execution_path: List[int] = field(default_factory=list)
    loop_counters: Dict[int, int] = field(default_factory=dict)
    dirty_pages: Set[int] = field(default_factory=set)

    # BB指令序列（用于LLM分析）
    bb_instructions: List[Dict] = field(default_factory=list)  # [{"mnemonic": "LDR", "operands": "R3, [R0]"}, ...]
    capture_schema: str = "lsgemu.unique_bb_snapshot.v2"
    capture_complete: bool = True
    region_hashes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    integrity_hash: str = ""

    def get_memory_size(self) -> int:
        """获取内存快照大小"""
        return sum(len(data) for data in self.memory_regions.values())


@dataclass
class LightweightSnapshotAnchor:
    """Long-lived lightweight state marker for scheduling and diagnostics."""

    snapshot_id: int
    bb_address: int
    pc: int
    timestamp: float
    instruction_count: int
    cpu_state: Dict[str, int] = field(default_factory=dict)
    flags: int = 0
    mmio_values: Dict[int, int] = field(default_factory=dict)
    execution_tail: List[int] = field(default_factory=list)
    dirty_page_hashes: Dict[int, str] = field(default_factory=dict)

    def estimate_size_bytes(self) -> int:
        return 256 + 8 * len(self.cpu_state) + 16 * len(self.mmio_values) + 48 * len(self.dirty_page_hashes)


class UniqueBBSnapshotManager:
    """
    唯一BB快照管理器

    核心策略：
    1. 每次遇到新的唯一BB时，保存快照
    2. 保留最近N个唯一BB的快照
    3. 支持逐级回退
    """

    def __init__(self,
                 max_snapshots: int = 5,
                 memory_regions: Optional[List[Tuple[int, int]]] = None,
                 page_store: Optional[SnapshotPageStore] = None):
        """
        初始化

        Args:
            max_snapshots: 最大快照数（保留最近N个唯一BB）
            memory_regions: 需要快照的内存区域 [(start, size), ...]
        """
        self.max_snapshots = max_snapshots
        self.memory_regions = memory_regions or [
            (0x20000000, 0x40000),  # RAM: 256KB
        ]

        # 快照队列（FIFO）
        self.snapshots: Deque[UniqueBBSnapshot] = deque(maxlen=max_snapshots)
        self.snapshot_counter: int = 0

        # 已见过的唯一BB集合
        self.seen_bbs: set = set()

        # 统计
        self.total_snapshots_created: int = 0
        self.total_rollbacks: int = 0
        self.page_store = page_store or SnapshotPageStore()
        self.capture_stats: Counter[str] = Counter()
        try:
            self.max_lightweight_anchors = max(0, int(os.environ.get("LSGEMU_LIGHTWEIGHT_SNAPSHOT_ANCHORS", "512")))
        except Exception:
            self.max_lightweight_anchors = 512
        self.lightweight_anchors: Deque[LightweightSnapshotAnchor] = deque(maxlen=self.max_lightweight_anchors)

    @staticmethod
    def _region_digest(data: bytes | PagedMemory) -> str:
        digest = getattr(data, "sha256", None)
        return str(digest) if digest else hashlib.sha256(bytes(data)).hexdigest()

    @classmethod
    def _integrity_digest(
        cls,
        cpu_state: Dict[str, int],
        flags: int,
        memory_regions: Dict[Tuple[int, int], bytes | PagedMemory],
    ) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(cpu_state.items()):
            digest.update(str(name).encode("ascii", errors="replace"))
            digest.update((int(value) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
        digest.update((int(flags) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
        for (base, size), data in sorted(memory_regions.items()):
            digest.update(int(base).to_bytes(8, "little", signed=False))
            digest.update(int(size).to_bytes(8, "little", signed=False))
            digest.update(cls._region_digest(data).encode("ascii"))
        return digest.hexdigest()

    def _capture_error(self, component: str, exc: object, address: int) -> SnapshotCaptureError:
        self.capture_stats["capture_failures"] += 1
        self.capture_stats[f"capture_failure_{component}"] += 1
        return SnapshotCaptureError(component, str(exc), address=address)

    def should_create_snapshot(self, bb_address: int) -> bool:
        """
        判断是否应该创建快照

        策略：只在遇到新的唯一BB时创建

        Args:
            bb_address: 当前BB地址

        Returns:
            是否应该创建快照
        """
        if bb_address not in self.seen_bbs:
            self.seen_bbs.add(bb_address)
            return True
        return False

    def create_snapshot(self,
                       uc,  # Unicorn实例
                       bb_address: int,
                       mmio_values: Optional[Dict[int, int]] = None,
                       mmio_history: Optional[List] = None,
                       execution_path: Optional[List[int]] = None,
                       loop_counters: Optional[Dict[int, int]] = None,
                       instruction_count: int = 0,
                       bb_instructions: Optional[List[Dict]] = None,
                       dirty_pages: Optional[set] = None) -> UniqueBBSnapshot:
        """
        创建唯一BB快照

        Args:
            uc: Unicorn实例
            bb_address: 当前BB地址
            mmio_values: MMIO值
            mmio_history: MMIO访问历史
            execution_path: 执行路径
            loop_counters: 循环计数器
            instruction_count: 指令计数
            bb_instructions: BB指令序列

        Returns:
            创建的快照
        """
        from unicorn.arm_const import (
            UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
            UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
            UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
            UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
            UC_ARM_REG_CPSR
        )

        snapshot = UniqueBBSnapshot(
            snapshot_id=self.snapshot_counter,
            bb_address=bb_address,
            timestamp=time.time(),
            instruction_count=instruction_count
        )

        self.snapshot_counter += 1

        # (1) 保存CPU状态
        reg_map = {
            'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1, 'r2': UC_ARM_REG_R2, 'r3': UC_ARM_REG_R3,
            'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5, 'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7,
            'r8': UC_ARM_REG_R8, 'r9': UC_ARM_REG_R9, 'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
            'r12': UC_ARM_REG_R12, 'sp': UC_ARM_REG_SP, 'lr': UC_ARM_REG_LR, 'pc': UC_ARM_REG_PC
        }

        for name, reg_id in reg_map.items():
            try:
                snapshot.cpu_state[name] = int(uc.reg_read(reg_id)) & 0xFFFFFFFFFFFFFFFF
            except Exception as exc:
                raise self._capture_error(f"register_{name}", exc, bb_address) from exc

        try:
            snapshot.pc = int(snapshot.cpu_state["pc"]) & 0xFFFFFFFF
            snapshot.flags = int(uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFFFFFFFFFF
        except Exception as exc:
            raise self._capture_error("register_cpsr", exc, bb_address) from exc

        # (2) 保存Memory
        for start, size in self.memory_regions:
            try:
                data = bytes(uc.mem_read(start, size))
            except Exception as exc:
                raise self._capture_error(f"memory_0x{int(start):08x}", exc, bb_address) from exc
            if len(data) != int(size):
                raise self._capture_error(
                    f"memory_0x{int(start):08x}",
                    f"short read: expected {int(size)}, got {len(data)}",
                    bb_address,
                )
            snapshot.memory_regions[(start, size)] = self.page_store.intern_region(data)
        snapshot.region_hashes = {
            key: self._region_digest(data)
            for key, data in snapshot.memory_regions.items()
        }
        snapshot.integrity_hash = self._integrity_digest(
            snapshot.cpu_state,
            snapshot.flags,
            snapshot.memory_regions,
        )

        # (3) 保存MMIO state
        if mmio_values:
            snapshot.mmio_values = mmio_values.copy()
        if mmio_history:
            snapshot.mmio_access_history = mmio_history.copy()

        # (4) 保存Execution metadata
        if execution_path:
            snapshot.execution_path = execution_path.copy()
        if loop_counters:
            snapshot.loop_counters = loop_counters.copy()
        snapshot.dirty_pages = {int(page) & ~0xFFF for page in (dirty_pages or set())}

        # (5) 保存BB指令序列（用于LLM分析）
        if bb_instructions:
            snapshot.bb_instructions = bb_instructions.copy()

        self._create_lightweight_anchor(
            uc=uc,
            snapshot=snapshot,
            dirty_pages=dirty_pages,
        )

        # 添加到队列（自动FIFO）
        if self.snapshots.maxlen and len(self.snapshots) >= self.snapshots.maxlen:
            self.capture_stats["snapshot_fifo_evictions"] += 1
        self.snapshots.append(snapshot)
        self.total_snapshots_created += 1
        self.capture_stats["captures_succeeded"] += 1

        logger.debug(f"✓ 创建唯一BB快照 #{snapshot.snapshot_id} @ 0x{bb_address:08x} "
                    f"(总快照数: {len(self.snapshots)}/{self.max_snapshots}, "
                    f"mem={snapshot.get_memory_size()} bytes)")

        return snapshot

    def _create_lightweight_anchor(self, uc, snapshot: UniqueBBSnapshot, dirty_pages: Optional[set] = None) -> None:
        if self.max_lightweight_anchors <= 0:
            return
        dirty_page_hashes: Dict[int, str] = {}
        dirty_pages = set(dirty_pages or set())
        for page in sorted(dirty_pages)[-64:]:
            try:
                page_i = int(page) & ~0xFFF
                data = bytes(uc.mem_read(page_i, 0x1000))
                dirty_page_hashes[page_i] = hashlib.sha1(data).hexdigest()[:16]
            except Exception:
                continue
        anchor = LightweightSnapshotAnchor(
            snapshot_id=snapshot.snapshot_id,
            bb_address=snapshot.bb_address,
            pc=snapshot.pc,
            timestamp=snapshot.timestamp,
            instruction_count=snapshot.instruction_count,
            cpu_state=dict(snapshot.cpu_state),
            flags=snapshot.flags,
            mmio_values=dict(snapshot.mmio_values),
            execution_tail=list(snapshot.execution_path[-16:]),
            dirty_page_hashes=dirty_page_hashes,
        )
        self.lightweight_anchors.append(anchor)

    def get_recent_snapshots(self, n: int = 3) -> List[UniqueBBSnapshot]:
        """
        获取最近N个快照

        Args:
            n: 快照数量

        Returns:
            最近N个快照（从新到旧）
        """
        return list(reversed(list(self.snapshots)))[:n]

    def rollback_one_level(self) -> Optional[UniqueBBSnapshot]:
        """
        回退一级（回退到上一个唯一BB）

        Returns:
            回退到的快照，如果没有快照则返回None
        """
        if len(self.snapshots) < 2:
            logger.warning("快照数量不足，无法回退")
            return None

        # 移除最后一个快照（当前卡死的BB）
        current = self.snapshots.pop()
        logger.info(f"⏪ 移除当前快照 #{current.snapshot_id} @ 0x{current.bb_address:08x}")

        # 返回新的最后一个快照（上一个唯一BB）
        target = self.snapshots[-1]
        self.total_rollbacks += 1

        logger.info(f"⏪ 回退到快照 #{target.snapshot_id} @ 0x{target.bb_address:08x}")
        return target

    def rearm_from_retained_snapshots(self):
        """
        恢复后重新开启快照唯一性窗口。

        只保留当前快照队列里的BB为“已见”，这样从恢复点重新执行时，
        后续路径上的旧BB仍然可以再次生成新快照。
        """
        self.seen_bbs = {snapshot.bb_address for snapshot in self.snapshots}

    def reset_history(self):
        """清空快照历史，让一次新的重放重新积累上下文。"""
        self.snapshots.clear()
        self.seen_bbs.clear()
        self.lightweight_anchors.clear()

    def restore_snapshot(self, uc, snapshot: UniqueBBSnapshot) -> bool:
        """
        恢复快照

        Args:
            uc: Unicorn实例
            snapshot: 要恢复的快照
        """
        from unicorn.arm_const import (
            UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
            UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
            UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
            UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
            UC_ARM_REG_CPSR
        )

        logger.info(f"恢复快照 #{snapshot.snapshot_id} @ 0x{snapshot.bb_address:08x}")

        try:
            if getattr(snapshot, "capture_complete", True) is not True:
                raise SnapshotIntegrityError("snapshot was not captured completely")
            for (start, size), data in snapshot.memory_regions.items():
                if int(size) <= 0 or len(data) != int(size):
                    raise SnapshotIntegrityError(
                        f"region 0x{int(start):08x} length {len(data)} != {int(size)}"
                    )
                expected_hash = dict(getattr(snapshot, "region_hashes", {}) or {}).get(
                    (start, size)
                )
                if expected_hash and self._region_digest(data) != str(expected_hash):
                    raise SnapshotIntegrityError(
                        f"region 0x{int(start):08x} digest mismatch"
                    )
            expected_integrity = str(getattr(snapshot, "integrity_hash", "") or "")
            if expected_integrity and self._integrity_digest(
                dict(snapshot.cpu_state), int(snapshot.flags), snapshot.memory_regions
            ) != expected_integrity:
                raise SnapshotIntegrityError("snapshot integrity hash mismatch")

            # (1) 恢复CPU状态
            reg_map = {
                'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1, 'r2': UC_ARM_REG_R2, 'r3': UC_ARM_REG_R3,
                'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5, 'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7,
                'r8': UC_ARM_REG_R8, 'r9': UC_ARM_REG_R9, 'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
                'r12': UC_ARM_REG_R12, 'sp': UC_ARM_REG_SP, 'lr': UC_ARM_REG_LR, 'pc': UC_ARM_REG_PC
            }

            for name, reg_id in reg_map.items():
                if name in snapshot.cpu_state:
                    uc.reg_write(reg_id, snapshot.cpu_state[name])

            uc.reg_write(UC_ARM_REG_PC, snapshot.pc)
            uc.reg_write(UC_ARM_REG_CPSR, snapshot.flags)

            # (2) 恢复Memory
            for (start, _size), data in snapshot.memory_regions.items():
                uc.mem_write(start, bytes(data))
        except Exception as exc:
            self.capture_stats["restore_failures"] += 1
            if isinstance(exc, SnapshotIntegrityError):
                self.capture_stats["restore_integrity_failures"] += 1
            logger.error("恢复快照失败: %s", exc)
            return False

        logger.info(f"✓ 快照恢复完成")
        return True

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "current_snapshots": len(self.snapshots),
            "max_snapshots": self.max_snapshots,
            "total_created": self.total_snapshots_created,
            "total_rollbacks": self.total_rollbacks,
            "unique_bbs_seen": len(self.seen_bbs),
            "snapshot_addresses": [f"0x{s.bb_address:08x}" for s in self.snapshots],
            "lightweight_anchors": len(self.lightweight_anchors),
            "max_lightweight_anchors": self.max_lightweight_anchors,
            "lightweight_estimated_bytes": sum(anchor.estimate_size_bytes() for anchor in self.lightweight_anchors),
            "lightweight_anchor_addresses": [f"0x{anchor.bb_address:08x}" for anchor in list(self.lightweight_anchors)[-8:]],
            "capture": dict(self.capture_stats),
            "page_store": self.page_store.get_statistics(),
        }
