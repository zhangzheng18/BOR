#!/usr/bin/env python3
"""
Enhanced MMIO Handler - 完善的 MMIO 处理器

功能:
1. 从 JSON 约束文件加载约束
2. Hook 所有 MMIO 访问（读/写）
3. 最新值覆盖旧值
4. 支持读取和写入
"""

import json
import logging
import os
from typing import Dict, Optional, List, Set, Tuple
from unicorn import *
from unicorn.arm_const import *

from ..artifact_io import atomic_json_dump

try:
    from .mmio_hook_registry import get_primary_mmio_handler
except ImportError:
    def get_primary_mmio_handler(uc):
        return None

from ..hook_lifecycle import managed_hook_add, managed_hook_del, managed_mem_map

logger = logging.getLogger(__name__)


class EnhancedMMIOHandler:
    """增强的 MMIO 处理器"""

    def __init__(
        self,
        uc,
        constraint_file: Optional[str] = None,
        *,
        branch_mmio_file_mode: Optional[str] = None,
    ):
        """
        初始化

        Args:
            uc: Unicorn 实例
            constraint_file: 约束 JSON 文件路径
        """
        self.uc = uc
        self.constraint_file = constraint_file
        self.branch_mmio_file_mode = self._normalize_branch_mmio_file_mode(branch_mmio_file_mode)

        # 文件层约束
        self.file_constraints: Dict[int, int] = {}
        self.file_pc_constraints: Dict[Tuple[int, int], int] = {}
        self.file_occurrence_constraints: Dict[Tuple[int, int, int], int] = {}
        self.file_constraint_meta: Dict[int, Dict] = {}
        self.file_pc_constraint_meta: Dict[Tuple[int, int], Dict] = {}
        self.file_occurrence_constraint_meta: Dict[Tuple[int, int, int], Dict] = {}

        # 运行时层约束
        self.runtime_constraints: Dict[int, int] = {}
        self.runtime_pc_constraints: Dict[Tuple[int, int], int] = {}
        self.runtime_occurrence_constraints: Dict[Tuple[int, int, int], int] = {}
        self.runtime_constraint_meta: Dict[int, Dict] = {}
        self.runtime_pc_constraint_meta: Dict[Tuple[int, int], Dict] = {}
        self.runtime_occurrence_constraint_meta: Dict[Tuple[int, int, int], Dict] = {}

        # 合并视图
        self.constraints: Dict[int, int] = {}
        self.pc_constraints: Dict[Tuple[int, int], int] = {}
        self.occurrence_constraints: Dict[Tuple[int, int, int], int] = {}
        self.constraint_meta: Dict[int, Dict] = {}
        self.pc_constraint_meta: Dict[Tuple[int, int], Dict] = {}
        self.occurrence_constraint_meta: Dict[Tuple[int, int, int], Dict] = {}
        self._constraint_mtime: Optional[float] = None

        # MMIO 状态 {address: current_value}
        self.mmio_state: Dict[int, int] = {}

        # 访问历史 [(pc, address, value, is_read)]
        self.access_history: List = []
        self.access_occurrence_history: List = []
        try:
            self.access_history_limit = max(
                1024,
                int(os.environ.get("LSGEMU_ENHANCED_MMIO_HISTORY_LIMIT", "16384")),
            )
        except ValueError:
            self.access_history_limit = 16384
        self.access_history_total = 0
        self.access_history_entries_discarded = 0
        self.read_occurrence_counts: Dict[Tuple[int, int], int] = {}

        # 统计
        self.read_count = 0
        self.write_count = 0
        self.constraint_hits = 0
        self.pc_constraint_hits = 0
        self.occurrence_constraint_hits = 0
        self.global_constraint_hits = 0
        self.branch_mmio_pc_constraint_hits = 0
        self.branch_mmio_global_constraint_hits = 0
        self.branch_mmio_scoped_pc_constraint_hits = 0
        self.branch_mmio_scoped_global_constraint_hits = 0
        self.pc_constraint_hit_keys: Set[Tuple[int, int]] = set()
        self.occurrence_constraint_hit_keys: Set[Tuple[int, int, int]] = set()
        self.branch_mmio_pc_constraint_hit_keys: Set[Tuple[int, int]] = set()
        self.branch_mmio_scoped_pc_constraint_hit_keys: Set[Tuple[int, int]] = set()

        # Hook 句柄
        self.read_hooks: List[int] = []
        self.write_hooks: List[int] = []
        self.read_hook = None
        self.write_hook = None
        self._bridged_to_primary = False
        self._bridged_primary_handler = None
        self._explicit_mmio_state_addresses: Set[int] = set()
        self._bridge_initialized_explicit_state = False

        # 加载约束
        if constraint_file:
            self.load_constraints(constraint_file)

        self.uart_tx_ready_default = str(
            os.environ.get("LSGEMU_UART_TX_READY_DEFAULT", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}

    def _append_access_history(self, record) -> None:
        """Retain a bounded diagnostic tail without changing MMIO state."""
        self.access_history.append(record)
        self.access_history_total += 1
        if len(self.access_history) > self.access_history_limit * 2:
            discarded = len(self.access_history) - self.access_history_limit
            del self.access_history[:discarded]
            self.access_history_entries_discarded += discarded

    @staticmethod
    def _mmio_ranges() -> List[Tuple[int, int]]:
        return [
            (0x40000000, 0x60000000),
            (0xE0000000, 0xE0100000),
        ]

    @staticmethod
    def _is_mmio_address(address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        return any(start <= address < end for start, end in EnhancedMMIOHandler._mmio_ranges())

    @staticmethod
    def _is_common_uart_status_register(address: int) -> bool:
        """Common STM32/libmaple USART SR addresses. TXE/TC ready unblocks print paths."""
        addr = int(address) & 0xFFFFFFFF
        common_bases = {
            0x40004400,  # USART2
            0x40004800,  # USART3
            0x40004C00,  # UART4
            0x40005000,  # UART5
            0x40007800,  # UART7 on some STM32 parts
            0x40007C00,  # UART8 on some STM32 parts
            0x40011000,  # USART1 on STM32F1
            0x40011400,  # USART6
            0x40013800,  # USART1 on STM32F2/F4/libmaple targets
        }
        return addr in common_bases

    def _default_mmio_read_value(self, pc: int, address: int, size: int) -> int:
        if self.uart_tx_ready_default and self._is_common_uart_status_register(address):
            # STM32 USART_SR: TXE=bit7, TC=bit6. Do not set RXNE, so receive
            # paths still depend on explicit input/modeling.
            return 0x000000C0
        return 0

    def _rebuild_constraints(self):
        """重建文件层 + 运行时层的合并视图。"""
        self.constraints = dict(self.file_constraints)
        self.constraints.update(self.runtime_constraints)

        self.pc_constraints = dict(self.file_pc_constraints)
        self.pc_constraints.update(self.runtime_pc_constraints)

        self.occurrence_constraints = dict(self.file_occurrence_constraints)
        self.occurrence_constraints.update(self.runtime_occurrence_constraints)

        self.constraint_meta = dict(self.file_constraint_meta)
        self.constraint_meta.update(self.runtime_constraint_meta)

        self.pc_constraint_meta = dict(self.file_pc_constraint_meta)
        self.pc_constraint_meta.update(self.runtime_pc_constraint_meta)

        self.occurrence_constraint_meta = dict(self.file_occurrence_constraint_meta)
        self.occurrence_constraint_meta.update(self.runtime_occurrence_constraint_meta)

    def load_constraints(self, constraint_file: str):
        """
        从 JSON 文件加载约束

        JSON 格式:
        {
            "0x40021000": 0x02000081,
            "0x40013800": 0x00000020,
            ...
        }
        """
        try:
            if not os.path.exists(constraint_file):
                logger.info(f"[MMIOHandler] 约束文件不存在，使用空约束: {constraint_file}")
                self.file_constraints.clear()
                self.file_pc_constraints.clear()
                self.file_occurrence_constraints.clear()
                self.file_constraint_meta.clear()
                self.file_pc_constraint_meta.clear()
                self.file_occurrence_constraint_meta.clear()
                self._rebuild_constraints()
                return
            self._constraint_mtime = os.path.getmtime(constraint_file)

            with open(constraint_file, 'r') as f:
                data = json.load(f)

            file_constraints: Dict[int, int] = {}
            file_pc_constraints: Dict[Tuple[int, int], int] = {}
            file_occurrence_constraints: Dict[Tuple[int, int, int], int] = {}
            file_constraint_meta: Dict[int, Dict] = {}
            file_pc_constraint_meta: Dict[Tuple[int, int], Dict] = {}
            file_occurrence_constraint_meta: Dict[Tuple[int, int, int], Dict] = {}
            if isinstance(data, dict) and "constraints" in data:
                for item in data.get("constraints", []):
                    if item.get("type") != "mmio":
                        continue
                    if not self._should_load_file_constraint(item, self.branch_mmio_file_mode):
                        continue
                    addr = self._parse_int(item.get("address"))
                    value = self._parse_int(item.get("value"))
                    if addr is not None and value is not None:
                        if "read_pc" in item and item.get("read_pc") is None and item.get("pc") is None:
                            continue
                        read_pc = self._parse_int(item.get("read_pc"))
                        if read_pc is None:
                            read_pc = self._parse_int(item.get("pc"))
                        read_occurrence = self._parse_int(item.get("read_occurrence"))
                        if read_pc is not None and read_occurrence is not None and read_occurrence > 0:
                            occurrence_key = (read_pc, addr, int(read_occurrence))
                            file_occurrence_constraints[occurrence_key] = value
                            file_occurrence_constraint_meta[occurrence_key] = dict(item)
                        elif read_pc is not None:
                            file_pc_constraints[(read_pc, addr)] = value
                            file_pc_constraint_meta[(read_pc, addr)] = dict(item)
                        else:
                            file_constraints[addr] = value
                            file_constraint_meta[addr] = dict(item)
            elif isinstance(data, dict):
                # 兼容旧格式: {"0x40021000": 1, ...}
                for addr_str, value in data.items():
                    addr = self._parse_int(addr_str)
                    parsed_value = self._parse_int(value)
                    if addr is not None and parsed_value is not None:
                        file_constraints[addr] = parsed_value
                        file_constraint_meta[addr] = {
                            "type": "mmio",
                            "address": addr_str,
                            "value": value,
                        }

            self.file_constraints = file_constraints
            self.file_pc_constraints = file_pc_constraints
            self.file_occurrence_constraints = file_occurrence_constraints
            self.file_constraint_meta = file_constraint_meta
            self.file_pc_constraint_meta = file_pc_constraint_meta
            self.file_occurrence_constraint_meta = file_occurrence_constraint_meta
            self._rebuild_constraints()

            logger.info(
                "[MMIOHandler] 加载了 %d 个约束",
                len(self.constraints)
                + len(self.pc_constraints)
                + len(self.occurrence_constraints),
            )

        except Exception as e:
            logger.error(f"[MMIOHandler] 加载约束失败: {e}")

    @staticmethod
    def _normalize_branch_mmio_file_mode(mode: Optional[str]) -> Optional[str]:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        return normalized or None

    @classmethod
    def _resolve_branch_mmio_file_mode(cls, explicit_mode: Optional[str] = None) -> str:
        normalized = cls._normalize_branch_mmio_file_mode(explicit_mode)
        if normalized is not None:
            return normalized
        env_mode = os.environ.get("LSGEMU_LOAD_BRANCH_MMIO_FILE_CONSTRAINTS", "")
        normalized = cls._normalize_branch_mmio_file_mode(env_mode)
        return normalized or ""

    @classmethod
    def _should_load_file_constraint(cls, item: Dict, explicit_mode: Optional[str] = None) -> bool:
        added_by = str(item.get("added_by", ""))
        speculation_level = str(item.get("speculation_level", "") or "")
        mode = cls._resolve_branch_mmio_file_mode(explicit_mode)
        if mode in {"unscoped-global", "global-ablation"}:
            return bool(
                speculation_level != "dynamic_replay_required"
                and added_by not in {"branch_mmio", "intelligent_emulator"}
                and item.get("read_occurrence") in (None, "", 0, "0")
            )
        if added_by != "branch_mmio":
            return True
        if mode in {"1", "true", "yes", "on", "all"}:
            return True
        if mode in {"0", "false", "no", "off", "none"}:
            return False
        if mode in {"scoped", "pc", "read-pc"}:
            return item.get("read_pc") is not None or item.get("pc") is not None
        # 默认仅加载按 read_pc 作用域命中的 branch_mmio 约束，避免地址级全局污染。
        return item.get("read_pc") is not None or item.get("pc") is not None

    def _parse_int(self, value) -> Optional[int]:
        """解析十六进制或十进制整数"""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        try:
            return int(text, 16) if text.lower().startswith('0x') else int(text)
        except ValueError:
            return None

    def add_constraint(self, address: int, value: int, *, added_by: str = "runtime"):
        """
        添加约束（最新值覆盖旧值）

        Args:
            address: MMIO 地址
            value: 约束值
        """
        address = int(address)
        value = int(value)
        self.runtime_constraints[address] = value
        self.runtime_constraint_meta[address] = {
            "type": "mmio",
            "address": f"0x{address & 0xFFFFFFFF:08x}",
            "value": f"0x{value & 0xFFFFFFFF:08x}",
            "added_by": added_by,
        }
        self._rebuild_constraints()
        logger.debug(f"[MMIOHandler] 添加约束: {hex(address)} = {hex(value)}")

    def add_pc_constraint(self, read_pc: int, address: int, value: int, *, added_by: str = "runtime"):
        """
        添加按读取点生效的约束。

        Args:
            read_pc: 读取该 MMIO 的指令地址
            address: MMIO 地址
            value: 约束值
            added_by: 元信息来源
        """
        key = (int(read_pc), int(address))
        self.runtime_pc_constraints[key] = int(value)
        self.runtime_pc_constraint_meta[key] = {
            "type": "mmio",
            "read_pc": f"0x{int(read_pc) & 0xFFFFFFFF:08x}",
            "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
            "value": f"0x{int(value) & 0xFFFFFFFF:08x}",
            "added_by": added_by,
        }
        self._rebuild_constraints()
        logger.debug(
            f"[MMIOHandler] 添加PC约束: PC={hex(int(read_pc))} "
            f"MMIO[{hex(int(address))}] = {hex(int(value))}"
        )

    def add_occurrence_constraint(
        self,
        read_pc: int,
        address: int,
        occurrence: int,
        value: int,
        *,
        added_by: str = "runtime",
    ) -> None:
        """Add a value that applies only to one dynamic read occurrence."""
        key = (int(read_pc), int(address), max(1, int(occurrence)))
        self.runtime_occurrence_constraints[key] = int(value)
        self.runtime_occurrence_constraint_meta[key] = {
            "type": "mmio",
            "read_pc": f"0x{key[0] & 0xFFFFFFFF:08x}",
            "address": f"0x{key[1] & 0xFFFFFFFF:08x}",
            "read_occurrence": key[2],
            "value": f"0x{int(value) & 0xFFFFFFFF:08x}",
            "added_by": added_by,
        }
        self._rebuild_constraints()
        logger.debug(
            "[MMIOHandler] 添加occurrence约束: PC=%s MMIO[%s]#%d = %s",
            hex(key[0]),
            hex(key[1]),
            key[2],
            hex(int(value)),
        )

    def _advance_read_occurrence(self, pc: int, address: int) -> int:
        site = (int(pc), int(address) & 0xFFFFFFFF)
        occurrence = int(self.read_occurrence_counts.get(site, 0) or 0) + 1
        self.read_occurrence_counts[site] = occurrence
        return occurrence

    def _record_occurrence_constraint_hit(self, key: Tuple[int, int, int]) -> None:
        self.constraint_hits += 1
        self.pc_constraint_hits += 1
        self.occurrence_constraint_hits += 1
        self.pc_constraint_hit_keys.add((int(key[0]), int(key[1])))
        self.occurrence_constraint_hit_keys.add(key)
        meta = self.occurrence_constraint_meta.get(key, {})
        if meta.get("added_by") == "branch_mmio":
            self.branch_mmio_pc_constraint_hits += 1
            self.branch_mmio_pc_constraint_hit_keys.add((int(key[0]), int(key[1])))
        elif meta.get("added_by") == "branch_mmio_scoped_runtime":
            self.branch_mmio_scoped_pc_constraint_hits += 1
            self.branch_mmio_scoped_pc_constraint_hit_keys.add((int(key[0]), int(key[1])))

    def refresh_constraints(self):
        """约束文件变化时重新加载，支持运行时补全约束。"""
        if not self.constraint_file:
            return
        try:
            if not os.path.exists(self.constraint_file):
                return
            mtime = os.path.getmtime(self.constraint_file)
            if self._constraint_mtime == mtime:
                return
            self.load_constraints(self.constraint_file)
        except Exception as e:
            logger.debug(f"[MMIOHandler] 刷新约束失败: {e}")

    def update_constraint(self, address: int, value: int):
        """
        更新约束（别名，与 add_constraint 相同）

        Args:
            address: MMIO 地址
            value: 新值
        """
        self.add_constraint(address, value)

    def mark_mmio_state_explicit(self, addresses=None):
        """Mark seeded MMIO state as replay evidence rather than passive history."""
        if addresses is None:
            addresses = self.mmio_state.keys()
        for address in list(addresses or []):
            try:
                self._explicit_mmio_state_addresses.add(int(address) & 0xFFFFFFFF)
            except Exception:
                continue

    def _record_constraint_hit(self, pc: int, address: int, *, pc_scoped: bool) -> None:
        self.constraint_hits += 1
        if pc_scoped:
            key = (int(pc), int(address))
            self.pc_constraint_hits += 1
            self.pc_constraint_hit_keys.add(key)
            constraint_meta = self.pc_constraint_meta.get(key, {})
            if constraint_meta.get("added_by") == "branch_mmio":
                self.branch_mmio_pc_constraint_hits += 1
                self.branch_mmio_pc_constraint_hit_keys.add(key)
            elif constraint_meta.get("added_by") == "branch_mmio_scoped_runtime":
                self.branch_mmio_scoped_pc_constraint_hits += 1
                self.branch_mmio_scoped_pc_constraint_hit_keys.add(key)
        else:
            self.global_constraint_hits += 1
            constraint_meta = self.constraint_meta.get(int(address), {})
            if constraint_meta.get("added_by") == "branch_mmio":
                self.branch_mmio_global_constraint_hits += 1
            elif constraint_meta.get("added_by") == "branch_mmio_scoped_runtime":
                self.branch_mmio_scoped_global_constraint_hits += 1

    def apply_external_input_override(self, pc: int, address: int, value: int) -> None:
        """r9 裁定：外部输入（轮询/等待环退出值）登记为显式约束。

        ready 位等外设状态只能由外设侧置位，不可能来自固件写；把该值登记为
        按读取点生效的显式约束后，resolve_bridge_read 的 PC 约束分支先于
        「bridge 快照状态」（固件写镜像）命中，读路径即可消费到该输入。
        pc=0 时登记为地址级全局约束（与主表 static_constraints 的
        (0, addr) 回退语义对齐）。
        """
        address = int(address) & 0xFFFFFFFF
        value = int(value) & 0xFFFFFFFF
        if int(pc or 0) == 0:
            self.add_constraint(address, value, added_by="external_loop_exit_input")
        else:
            self.add_pc_constraint(
                int(pc), address, value, added_by="external_loop_exit_input"
            )

    def resolve_bridge_read(self, pc: int, address: int, size: int) -> Tuple[bool, int]:
        """Return explicit scoped replay evidence for the primary MMIO handler."""
        pc = int(pc)
        address = int(address) & 0xFFFFFFFF
        mask = (1 << (max(1, int(size or 1)) * 8)) - 1
        self.refresh_constraints()
        occurrence = self._advance_read_occurrence(pc, address)

        occurrence_key = (pc, address, occurrence)
        if occurrence_key in self.occurrence_constraints:
            return_value = int(self.occurrence_constraints[occurrence_key]) & mask
            self._record_occurrence_constraint_hit(occurrence_key)
            logger.debug(
                "[MMIOHandler] bridge occurrence约束 @ %s = %s (PC=%s #%d)",
                hex(address),
                hex(return_value),
                hex(pc),
                occurrence,
            )
            return True, return_value

        key = (pc, address)
        if key in self.pc_constraints:
            return_value = int(self.pc_constraints[key]) & mask
            self._record_constraint_hit(pc, address, pc_scoped=True)
            logger.debug(f"[MMIOHandler] bridge PC约束 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")
            return True, return_value

        if address in self.constraints:
            return_value = int(self.constraints[address]) & mask
            self._record_constraint_hit(pc, address, pc_scoped=False)
            logger.debug(f"[MMIOHandler] bridge 地址约束 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")
            return True, return_value

        if address in self._explicit_mmio_state_addresses and address in self.mmio_state:
            return_value = int(self.mmio_state[address]) & mask
            logger.debug(f"[MMIOHandler] bridge 快照状态 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")
            return True, return_value

        return False, 0

    def record_bridge_read(self, pc: int, address: int, size: int, value: int) -> None:
        """Mirror a primary-handler read into this handler's state/statistics."""
        pc = int(pc)
        address = int(address) & 0xFFFFFFFF
        mask = (1 << (max(1, int(size or 1)) * 8)) - 1
        value = int(value) & mask
        self.read_count += 1
        self.mmio_state[address] = value
        self._append_access_history((pc, address, value, True))
        occurrence = int(self.read_occurrence_counts.get((pc, address), 0) or 0)
        self.access_occurrence_history.append((pc, address, value, True, occurrence))

    def record_bridge_write(self, pc: int, address: int, size: int, value: int) -> None:
        """Mirror a primary-handler write into this handler's state/statistics."""
        pc = int(pc)
        address = int(address) & 0xFFFFFFFF
        mask = (1 << (max(1, int(size or 1)) * 8)) - 1
        value = int(value) & mask
        self.write_count += 1
        self.mmio_state[address] = value
        self._explicit_mmio_state_addresses.add(address)
        self._append_access_history((pc, address, value, False))

    def _sync_bridge_state_from_primary(self):
        primary = self._bridged_primary_handler
        if primary is None:
            return
        peek = getattr(primary, "peek_register_value", None)
        if not callable(peek):
            return
        for address in list(self.mmio_state.keys()):
            try:
                value = peek(int(address), 4)
            except Exception:
                continue
            if value is not None:
                self.mmio_state[int(address) & 0xFFFFFFFF] = int(value) & 0xFFFFFFFF

    def start_hooking(self):
        """开始 Hook MMIO 访问"""
        if self._bridged_to_primary or self.read_hooks or self.write_hooks:
            self.stop_hooking()
        self.read_hooks = []
        self.write_hooks = []
        primary = get_primary_mmio_handler(self.uc)
        if primary is not None and hasattr(primary, "push_mmio_overlay"):
            self.refresh_constraints()
            if not self._bridge_initialized_explicit_state:
                self.mark_mmio_state_explicit()
                self._bridge_initialized_explicit_state = True
            primary.push_mmio_overlay(self)
            self._bridged_to_primary = True
            self._bridged_primary_handler = primary
            self.read_hook = None
            self.write_hook = None
            logger.debug("[MMIOHandler] 桥接到主MMIO handler，未安装第二套Unicorn hook")
            return

        for start, end in self._mmio_ranges():
            self.read_hooks.append(managed_hook_add(self.uc,
                UC_HOOK_MEM_READ,
                self._handle_mmio_read,
                None,
                start,
                end,
            ))
            self.write_hooks.append(managed_hook_add(self.uc,
                UC_HOOK_MEM_WRITE,
                self._handle_mmio_write,
                None,
                start,
                end,
            ))
        self.read_hook = self.read_hooks[0] if self.read_hooks else None
        self.write_hook = self.write_hooks[0] if self.write_hooks else None

        logger.info("[MMIOHandler] 开始 Hook MMIO 访问")

    def stop_hooking(self):
        """停止 Hook"""
        if self._bridged_to_primary:
            try:
                primary = self._bridged_primary_handler
                if primary is not None and hasattr(primary, "pop_mmio_overlay"):
                    primary.pop_mmio_overlay(self)
                self._sync_bridge_state_from_primary()
            finally:
                self._bridged_to_primary = False
                self._bridged_primary_handler = None
                self.read_hooks = []
                self.write_hooks = []
                self.read_hook = None
                self.write_hook = None
            logger.debug("[MMIOHandler] 解除主MMIO handler桥接")
            return

        for hook in list(self.read_hooks or []):
            try:
                managed_hook_del(self.uc, hook)
            except Exception:
                pass
        for hook in list(self.write_hooks or []):
            try:
                managed_hook_del(self.uc, hook)
            except Exception:
                pass
        self.read_hooks = []
        self.write_hooks = []
        self.read_hook = None
        self.write_hook = None
        logger.info("[MMIOHandler] 停止 Hook")

    def _handle_mmio_read(self, uc, access, address, size, value, user_data):
        """处理 MMIO 读取"""
        try:
            pc = uc.reg_read(UC_ARM_REG_PC)
            self.read_count += 1
            self.refresh_constraints()
            occurrence = self._advance_read_occurrence(pc, address)

            # 1. 优先使用约束值
            occurrence_key = (int(pc), int(address), int(occurrence))
            if occurrence_key in self.occurrence_constraints:
                return_value = self.occurrence_constraints[occurrence_key]
                self._record_occurrence_constraint_hit(occurrence_key)
                logger.debug(
                    "[MMIOHandler] 读取occurrence约束 @ %s = %s (PC=%s #%d)",
                    hex(address),
                    hex(return_value),
                    hex(pc),
                    occurrence,
                )

            elif (pc, address) in self.pc_constraints:
                return_value = self.pc_constraints[(pc, address)]
                self.constraint_hits += 1
                self.pc_constraint_hits += 1
                self.pc_constraint_hit_keys.add((pc, address))
                constraint_meta = self.pc_constraint_meta.get((pc, address), {})
                if constraint_meta.get("added_by") == "branch_mmio":
                    self.branch_mmio_pc_constraint_hits += 1
                    self.branch_mmio_pc_constraint_hit_keys.add((pc, address))
                elif constraint_meta.get("added_by") == "branch_mmio_scoped_runtime":
                    self.branch_mmio_scoped_pc_constraint_hits += 1
                    self.branch_mmio_scoped_pc_constraint_hit_keys.add((pc, address))
                logger.debug(f"[MMIOHandler] 读取PC约束 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")

            elif address in self.constraints:
                return_value = self.constraints[address]
                self.constraint_hits += 1
                self.global_constraint_hits += 1
                constraint_meta = self.constraint_meta.get(address, {})
                if constraint_meta.get("added_by") == "branch_mmio":
                    self.branch_mmio_global_constraint_hits += 1
                elif constraint_meta.get("added_by") == "branch_mmio_scoped_runtime":
                    self.branch_mmio_scoped_global_constraint_hits += 1
                logger.debug(f"[MMIOHandler] 读取约束 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")

            # 2. 其次使用当前状态
            elif address in self.mmio_state:
                return_value = self.mmio_state[address]
                logger.debug(f"[MMIOHandler] 读取状态 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")

            # 3. 默认返回 0
            else:
                return_value = self._default_mmio_read_value(pc, address, size)
                logger.debug(f"[MMIOHandler] 读取默认 @ {hex(address)} = {hex(return_value)} (PC={hex(pc)})")

            # 更新状态
            self.mmio_state[address] = return_value

            # 记录历史
            self._append_access_history((pc, address, return_value, True))
            self.access_occurrence_history.append(
                (pc, address, return_value, True, occurrence)
            )

            # 写入内存（让 Unicorn 读取到这个值）
            try:
                mask = (1 << (size * 8)) - 1
                uc.mem_write(address, (return_value & mask).to_bytes(size, 'little'))
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"[MMIOHandler] 读取处理错误: {e}")

    def _handle_mmio_write(self, uc, access, address, size, value, user_data):
        """处理 MMIO 写入"""
        try:
            pc = uc.reg_read(UC_ARM_REG_PC)
            self.write_count += 1

            # 更新状态（最新值覆盖旧值）
            address = int(address) & 0xFFFFFFFF
            self.mmio_state[address] = value
            self._explicit_mmio_state_addresses.add(address)

            # 记录历史
            self._append_access_history((pc, address, value, False))

            logger.debug(f"[MMIOHandler] 写入 @ {hex(address)} = {hex(value)} (PC={hex(pc)})")

        except Exception as e:
            logger.debug(f"[MMIOHandler] 写入处理错误: {e}")

    def get_mmio_value(self, address: int) -> Optional[int]:
        """
        获取 MMIO 当前值

        Args:
            address: MMIO 地址

        Returns:
            当前值或 None
        """
        # 优先返回约束值
        if address in self.constraints:
            return self.constraints[address]

        # 其次返回状态值
        if address in self.mmio_state:
            return self.mmio_state[address]

        return None

    def set_mmio_value(self, address: int, value: int):
        """
        设置 MMIO 值

        Args:
            address: MMIO 地址
            value: 值
        """
        address = int(address) & 0xFFFFFFFF
        value = int(value) & 0xFFFFFFFF

        # Establish the page before committing model state.  When this method
        # is reached from a Unicorn callback, managed_mem_map raises the
        # owner's deferred-mapping signal and the whole callback is retried
        # after the native execution has unwound.
        if self.uc is not None:
            # The owner's private deferred-mapping signal derives from
            # BaseException and propagates to managed_emu_start; ordinary map
            # failures retain the historical best-effort model-only behavior.
            try:
                managed_mem_map(self.uc, address & ~0xFFF, 0x1000)
            except Exception:
                pass

        # 同时更新约束和状态
        self.constraints[address] = value
        self.mmio_state[address] = value
        self._explicit_mmio_state_addresses.add(address)

        # 写入内存
        try:
            if self.uc is not None:
                self.uc.mem_write(address, value.to_bytes(4, 'little'))
        except Exception:
            pass

    def get_recent_accesses(self, count: int = 10) -> List:
        """获取最近的访问记录"""
        return self.access_history[-count:]

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        branch_mmio_constraint_count = sum(
            1 for item in self.constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio'
        ) + sum(
            1 for item in self.pc_constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio'
        ) + sum(
            1 for item in self.occurrence_constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio'
        )
        branch_mmio_scoped_constraint_count = sum(
            1 for item in self.constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio_scoped_runtime'
        ) + sum(
            1 for item in self.pc_constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio_scoped_runtime'
        ) + sum(
            1 for item in self.occurrence_constraint_meta.values()
            if isinstance(item, dict) and item.get('added_by') == 'branch_mmio_scoped_runtime'
        )
        return {
            'total_reads': self.read_count,
            'total_writes': self.write_count,
            'constraint_hits': self.constraint_hits,
            'pc_constraint_hits': self.pc_constraint_hits,
            'occurrence_constraint_hits': self.occurrence_constraint_hits,
            'global_constraint_hits': self.global_constraint_hits,
            'branch_mmio_constraint_hits': self.branch_mmio_pc_constraint_hits + self.branch_mmio_global_constraint_hits,
            'branch_mmio_pc_constraint_hits': self.branch_mmio_pc_constraint_hits,
            'branch_mmio_global_constraint_hits': self.branch_mmio_global_constraint_hits,
            'branch_mmio_scoped_constraint_hits': self.branch_mmio_scoped_pc_constraint_hits + self.branch_mmio_scoped_global_constraint_hits,
            'branch_mmio_scoped_pc_constraint_hits': self.branch_mmio_scoped_pc_constraint_hits,
            'branch_mmio_scoped_global_constraint_hits': self.branch_mmio_scoped_global_constraint_hits,
            'pc_constraint_count': len(self.pc_constraints),
            'occurrence_constraint_count': len(self.occurrence_constraints),
            'global_constraint_count': len(self.constraints),
            'branch_mmio_constraint_count': branch_mmio_constraint_count,
            'branch_mmio_scoped_constraint_count': branch_mmio_scoped_constraint_count,
            'unique_pc_constraint_hits': len(self.pc_constraint_hit_keys),
            'unique_occurrence_constraint_hits': len(self.occurrence_constraint_hit_keys),
            'unique_branch_mmio_pc_constraint_hits': len(self.branch_mmio_pc_constraint_hit_keys),
            'unique_branch_mmio_scoped_pc_constraint_hits': len(self.branch_mmio_scoped_pc_constraint_hit_keys),
            'total_constraints': (
                len(self.constraints)
                + len(self.pc_constraints)
                + len(self.occurrence_constraints)
            ),
            'mmio_state_size': len(self.mmio_state),
            'access_history_size': len(self.access_history),
            'access_history_retention': {
                'limit': int(getattr(self, 'access_history_limit', 0) or 0),
                'retained': len(self.access_history),
                'total': int(
                    getattr(self, 'access_history_total', len(self.access_history)) or 0
                ),
                'discarded': int(
                    getattr(self, 'access_history_entries_discarded', 0) or 0
                ),
            },
        }

    def save_constraints(self, output_file: str):
        """
        保存当前约束到 JSON 文件

        Args:
            output_file: 输出文件路径
        """
        try:
            # 转换为可序列化格式
            data = {hex(addr): value for addr, value in self.constraints.items()}

            atomic_json_dump(data, output_file, indent=2)

            logger.info(f"[MMIOHandler] 保存了 {len(data)} 个约束到 {output_file}")

        except Exception as e:
            logger.error(f"[MMIOHandler] 保存约束失败: {e}")

    def clear(self):
        """清空所有数据"""
        self.file_constraints.clear()
        self.file_pc_constraints.clear()
        self.file_occurrence_constraints.clear()
        self.file_constraint_meta.clear()
        self.file_pc_constraint_meta.clear()
        self.file_occurrence_constraint_meta.clear()
        self.runtime_constraints.clear()
        self.runtime_pc_constraints.clear()
        self.runtime_occurrence_constraints.clear()
        self.runtime_constraint_meta.clear()
        self.runtime_pc_constraint_meta.clear()
        self.runtime_occurrence_constraint_meta.clear()
        self.constraints.clear()
        self.pc_constraints.clear()
        self.occurrence_constraints.clear()
        self.constraint_meta.clear()
        self.pc_constraint_meta.clear()
        self.occurrence_constraint_meta.clear()
        self.mmio_state.clear()
        self._explicit_mmio_state_addresses.clear()
        self._bridge_initialized_explicit_state = False
        self.access_history.clear()
        self.access_occurrence_history.clear()
        self.access_history_total = 0
        self.access_history_entries_discarded = 0
        self.read_occurrence_counts.clear()
        self.read_count = 0
        self.write_count = 0
        self.constraint_hits = 0
        self.pc_constraint_hits = 0
        self.occurrence_constraint_hits = 0
        self.global_constraint_hits = 0
        self.branch_mmio_pc_constraint_hits = 0
        self.branch_mmio_global_constraint_hits = 0
        self.branch_mmio_scoped_pc_constraint_hits = 0
        self.branch_mmio_scoped_global_constraint_hits = 0
        self.pc_constraint_hit_keys.clear()
        self.occurrence_constraint_hit_keys.clear()
        self.branch_mmio_pc_constraint_hit_keys.clear()
        self.branch_mmio_scoped_pc_constraint_hit_keys.clear()
