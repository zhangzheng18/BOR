#!/usr/bin/env python3
"""
状态化MMIO处理器

按照idea.md的要求，从stateless response改为stateful MMIO Handler：
- 维护访问历史
- 推断执行阶段
- 循环计数器
- 智能fallback策略（替换危险的return 1）

核心理念：
MMIO Handler = 状态机
value = f(mmio_addr, pc, state)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Deque
from collections import Counter, deque
from enum import Enum
import copy
import json
import logging
import os

from ..causal_context import CausalExecutionContext
from .peripheral_semantic_profiles import SemanticProfileRegistry

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """执行阶段"""
    INIT = "init"               # 初始化阶段
    POLLING = "polling"         # 轮询阶段
    DATA_TRANSFER = "data_transfer"  # 数据传输阶段
    IDLE = "idle"               # 空闲阶段
    ERROR = "error"             # 错误阶段
    UNKNOWN = "unknown"         # 未知阶段


@dataclass
class MMIOAccessRecord:
    """MMIO访问记录"""
    pc: int                     # 访问的PC
    mmio_addr: int              # MMIO地址
    is_read: bool               # 是否是读操作
    value: int                  # 读取或写入的值
    timestamp: int              # 时间戳（指令计数）


class MMIOState:
    """
    MMIO状态

    维护单个MMIO地址的状态信息
    """

    def __init__(self, mmio_addr: int, history_size: int = 10):
        self.mmio_addr = mmio_addr
        self.history_size = history_size

        # 访问历史（最近N次访问）
        self.history: Deque[MMIOAccessRecord] = deque(maxlen=history_size)

        # 当前值
        self.current_value: int = 0

        # 统计信息
        self.read_count: int = 0
        self.write_count: int = 0
        self.last_write_value: Optional[int] = None

        # 模式识别
        self.is_status_register: bool = False    # 是否是状态寄存器
        self.is_data_register: bool = False      # 是否是数据寄存器
        self.is_control_register: bool = False   # 是否是控制寄存器

        # 位模式
        self.ready_bit: Optional[int] = None     # ready位的位置
        self.error_bit: Optional[int] = None     # error位的位置

    def record_access(self, pc: int, is_read: bool, value: int, timestamp: int):
        """记录一次访问"""
        record = MMIOAccessRecord(pc, self.mmio_addr, is_read, value, timestamp)
        self.history.append(record)

        if is_read:
            self.read_count += 1
        else:
            self.write_count += 1
            self.last_write_value = value
            self.current_value = value

    def infer_register_type(self):
        """推断寄存器类型"""
        if self.read_count > self.write_count * 3:
            # 读多写少 -> 可能是状态寄存器
            self.is_status_register = True
        elif self.write_count > self.read_count * 3:
            # 写多读少 -> 可能是控制寄存器
            self.is_control_register = True
        elif self.read_count > 0 and self.write_count > 0:
            # 读写都有 -> 可能是数据寄存器
            self.is_data_register = True

    def detect_ready_bit(self) -> Optional[int]:
        """
        检测ready位

        策略：如果某一位频繁翻转，可能是ready位
        """
        if len(self.history) < 5:
            return None

        # 统计每一位的翻转次数
        bit_flips = [0] * 32
        last_value = None

        for record in self.history:
            if not record.is_read:
                continue

            if last_value is not None:
                for bit in range(32):
                    if ((last_value >> bit) & 1) != ((record.value >> bit) & 1):
                        bit_flips[bit] += 1

            last_value = record.value

        # 找到翻转最频繁的位
        max_flips = max(bit_flips)
        if max_flips >= 3:  # 至少翻转3次
            ready_bit = bit_flips.index(max_flips)
            self.ready_bit = ready_bit
            return ready_bit

        return None

    def get_last_n_values(self, n: int = 5) -> List[int]:
        """获取最近N次读取的值"""
        values = []
        for record in reversed(self.history):
            if record.is_read:
                values.append(record.value)
                if len(values) >= n:
                    break
        return list(reversed(values))


class StatefulMMIOHandler:
    """
    状态化MMIO处理器

    核心改进：
    1. 维护每个MMIO地址的状态
    2. 推断执行阶段
    3. 智能fallback策略
    """

    def __init__(
        self,
        static_constraints: Optional[Dict] = None,
        constraint_json_path: Optional[str] = None,
        *,
        branch_mmio_file_mode: Optional[str] = None,
        causal_context: Optional[CausalExecutionContext] = None,
        mmio_seed_values: Optional[Dict[int, int]] = None,
    ):
        """
        初始化

        Args:
            static_constraints: 静态约束（来自静态分析）
            mmio_seed_values: 静态初始种子 {地址: 值}，仅在该地址首次
                读取落到未建模路径时作为初始输入值（见 handle_read 优先级5.5）
        """
        self.static_constraints = static_constraints or {}
        self.constraint_json_path = constraint_json_path
        self.branch_mmio_file_mode = self._normalize_branch_mmio_file_mode(branch_mmio_file_mode)
        self.mmio_seed_values: Dict[int, int] = {
            int(address) & 0xFFFFFFFF: int(value) & 0xFFFFFFFF
            for address, value in dict(mmio_seed_values or {}).items()
        }
        self.mmio_seed_stats: Dict[str, int] = {
            "configured": len(self.mmio_seed_values),
            "applied": 0,
        }
        self._constraint_mtime: Optional[float] = None
        if constraint_json_path:
            self.refresh_constraints(force=True)

        # MMIO状态表
        self.mmio_states: Dict[int, MMIOState] = {}
        self.causal_context = causal_context or CausalExecutionContext()

        # 全局状态
        self.instruction_count: int = 0
        self.current_phase: ExecutionPhase = ExecutionPhase.INIT

        # 循环计数器
        self.loop_counters: Dict[int, int] = {}  # pc -> count

        # 执行历史（用于推断阶段）
        self.pc_history: Deque[int] = deque(maxlen=100)

        # 通用外设语义 profile。它只复用经过本地验证或 SVD 支持的
        # polling/status 规则，避免继续依赖单一厂商的硬编码地址。
        self.semantic_profiles = SemanticProfileRegistry(
            causal_context=self.causal_context
        )
        self.uart_tx_ready_default = str(
            os.environ.get("LSGEMU_UART_TX_READY_DEFAULT", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        # Opt-in read-path attribution: which priority level served each address.
        # Used to prove that a registered semantic profile is not masked by a
        # firmware write mirror (overlay) or a file constraint.
        self.mmio_source_audit_enabled = str(
            os.environ.get("LSGEMU_MMIO_SOURCE_AUDIT", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.mmio_read_source_counts: Dict[int, Counter] = {}
        self.legacy_uart_tx_ready_hits: int = 0
        self._mmio_overlays: List[Any] = []
        self.peripheral_input_ready_default = str(
            os.environ.get("LSGEMU_UART_RX_READY_DEFAULT", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._stream_input_byte_provider: Optional[Callable[[str], int]] = None
        self._stream_input_event_recorder: Optional[Callable[[Dict[str, object]], None]] = None
        self._local_input_default_bytes = self._load_local_input_seed()
        self._local_input_cursors: Dict[str, int] = {}
        self._peripheral_input_pending: Dict[str, int] = {}
        self.peripheral_input_stats: Dict[str, int] = {
            "rx_ready_status_reads": 0,
            "rx_data_reads": 0,
            "rx_bytes_prepared": 0,
            "rx_bytes_consumed": 0,
            "rx_events_recorded": 0,
            "rx_data_reads_without_ready": 0,
            "rx_overlay_mismatches": 0,
        }

    def get_or_create_state(self, mmio_addr: int) -> MMIOState:
        """获取或创建MMIO状态"""
        if mmio_addr not in self.mmio_states:
            self.mmio_states[mmio_addr] = MMIOState(mmio_addr)
        return self.mmio_states[mmio_addr]

    def snapshot_runtime_state(
        self,
        *,
        copy_values: bool = True,
        include_causal_context: bool = True,
    ) -> Dict[str, object]:
        """Capture all mutable peripheral state needed by a causal replay.

        ``copy_values=False`` is used only when the caller immediately
        serializes the returned view before guest execution resumes.  Existing
        callers retain deep-copy behavior by default.  The causal context can
        be omitted when the owning emulator stores the same context as a
        separate top-level component; this removes a duplicate representation
        without changing the restore contract.
        """
        if not copy_values:
            state = {
                "mmio_states": self.mmio_states,
                "instruction_count": int(self.instruction_count),
                "current_phase": self.current_phase.value,
                "loop_counters": self.loop_counters,
                "pc_history": self.pc_history,
                "pc_history_maxlen": self.pc_history.maxlen,
                "semantic_profiles": self.semantic_profiles.snapshot_runtime_state(
                    copy_values=False
                ),
                "local_input_cursors": self._local_input_cursors,
                "peripheral_input_pending": self._peripheral_input_pending,
                "peripheral_input_stats": self.peripheral_input_stats,
            }
            if include_causal_context:
                state["causal_context"] = self.causal_context.snapshot_runtime_state(
                    copy_values=False
                )
            return state
        state = {
            "mmio_states": copy.deepcopy(self.mmio_states),
            "instruction_count": int(self.instruction_count),
            "current_phase": self.current_phase.value,
            "loop_counters": dict(self.loop_counters),
            "pc_history": list(self.pc_history),
            "pc_history_maxlen": self.pc_history.maxlen,
            "semantic_profiles": self.semantic_profiles.snapshot_runtime_state(),
            "local_input_cursors": dict(self._local_input_cursors),
            "peripheral_input_pending": dict(self._peripheral_input_pending),
            "peripheral_input_stats": dict(self.peripheral_input_stats),
        }
        if include_causal_context:
            state["causal_context"] = self.causal_context.snapshot_runtime_state()
        return state

    def restore_runtime_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore a causal replay state without replacing configured callbacks."""
        if not isinstance(state, dict):
            return
        restored_mmio = state.get("mmio_states")
        if isinstance(restored_mmio, dict):
            self.mmio_states = copy.deepcopy(restored_mmio)
        self.instruction_count = max(0, int(state.get("instruction_count", 0) or 0))
        phase_value = str(state.get("current_phase") or ExecutionPhase.INIT.value)
        try:
            self.current_phase = ExecutionPhase(phase_value)
        except ValueError:
            self.current_phase = ExecutionPhase.UNKNOWN
        self.loop_counters = {
            int(pc): int(count)
            for pc, count in dict(state.get("loop_counters", {}) or {}).items()
        }
        try:
            history_maxlen = max(1, int(state.get("pc_history_maxlen", 100) or 100))
        except (TypeError, ValueError):
            history_maxlen = 100
        self.pc_history = deque(
            (int(pc) for pc in list(state.get("pc_history", []) or [])),
            maxlen=history_maxlen,
        )
        self.semantic_profiles.restore_runtime_state(state.get("semantic_profiles"))
        self._local_input_cursors = {
            str(key): int(value)
            for key, value in dict(state.get("local_input_cursors", {}) or {}).items()
        }
        self._peripheral_input_pending = {
            str(key): int(value)
            for key, value in dict(state.get("peripheral_input_pending", {}) or {}).items()
        }
        restored_stats = state.get("peripheral_input_stats")
        if isinstance(restored_stats, dict):
            self.peripheral_input_stats = {
                str(key): int(value or 0)
                for key, value in restored_stats.items()
            }
        self.causal_context.restore_runtime_state(state.get("causal_context"))

    def push_mmio_overlay(self, overlay: Any) -> None:
        """Install a scoped replay overlay above the persistent MMIO model."""
        if overlay is None:
            return
        self._mmio_overlays = [item for item in self._mmio_overlays if item is not overlay]
        self._mmio_overlays.append(overlay)

    def pop_mmio_overlay(self, overlay: Any) -> None:
        """Remove a scoped replay overlay without touching persistent state."""
        if overlay is None:
            return
        self._mmio_overlays = [item for item in self._mmio_overlays if item is not overlay]

    def apply_external_loop_exit_input(self, pc: int, address: int, value: int) -> None:
        """r9 裁定：外部输入（轮询/等待环退出值）写入并穿透 overlay 生效。

        写 MMIO 值让轮询环退出 = 外设状态变化（用户原则 1：外部输入可达）。
        只写主表 static_constraints 不够：活跃 overlay 的固件写镜像
        （EnhancedMMIOHandler.record_bridge_write → mmio_state）在
        handle_read 优先级 1 被当作 scoped replay evidence 供值，会遮蔽
        该输入——ready 位本就只能由外设侧置位，不可能来自固件写。因此
        同步把值登记进所有活跃 overlay 的显式 PC/地址约束。
        """
        address = int(address) & 0xFFFFFFFF
        value = int(value) & 0xFFFFFFFF
        self.static_constraints[(int(pc or 0), address)] = value
        for overlay in list(self._mmio_overlays):
            applier = getattr(overlay, "apply_external_input_override", None)
            if not callable(applier):
                continue
            try:
                applier(pc, address, value)
            except Exception as exc:
                logger.debug(
                    "外部输入 overlay 穿透失败 @ 0x%08x: %s", address, exc
                )

    def set_stream_input_provider(
        self,
        byte_provider: Optional[Callable[[str], int]],
        *,
        event_recorder: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        """Attach the emulator's replayable stream input source to MMIO RX."""
        self._stream_input_byte_provider = byte_provider
        self._stream_input_event_recorder = event_recorder

    @staticmethod
    def _load_local_input_seed() -> bytes:
        configured = (
            os.environ.get("LSGEMU_UART_INPUT_BYTES", "").strip()
            or os.environ.get("LSGEMU_STREAM_INPUT_BYTES", "").strip()
        )
        if configured:
            try:
                cleaned = "".join(ch for ch in configured if ch in "0123456789abcdefABCDEF")
                if cleaned and len(cleaned) % 2 == 0:
                    data = bytes.fromhex(cleaned)
                    if data:
                        return data
            except Exception:
                pass
        return bytes([
            0xDD, 0xFF, 0xBB, 0x31, 0x32, 0x33, 0x00,
            0xFF, 0x41, 0x42, 0x43, 0x0A,
            ord("G"), ord("1"), ord(" "), ord("X"), ord("1"), ord("\n"),
            0x00, 0x01, 0x02, 0x08, 0x10, 0x48, 0x68, 0x7F, 0x80, 0xFF,
        ])

    def peripheral_input_ready_enabled(self) -> bool:
        return bool(self.peripheral_input_ready_default)

    def _next_local_input_byte(self, stream_key: str) -> int:
        data = self._local_input_default_bytes or b"\x00"
        key = str(stream_key or "mmio")
        cursor = int(self._local_input_cursors.get(key, 0) or 0)
        self._local_input_cursors[key] = cursor + 1
        return int(data[cursor % len(data)]) & 0xFF

    def _next_external_input_byte(self, stream_key: str) -> int:
        provider = self._stream_input_byte_provider
        if callable(provider):
            try:
                return int(provider(str(stream_key or "mmio"))) & 0xFF
            except Exception as exc:
                logger.debug(f"MMIO input byte provider failed: {exc}")
        return self._next_local_input_byte(stream_key)

    def _record_peripheral_input_event(self, event: Dict[str, object]) -> None:
        recorder = self._stream_input_event_recorder
        if callable(recorder):
            try:
                recorder(event)
                self.peripheral_input_stats["rx_events_recorded"] += 1
                return
            except Exception as exc:
                logger.debug(f"MMIO input event recorder failed: {exc}")

    def prepare_peripheral_input_byte(
        self,
        stream_key: str,
        *,
        status_addr: int,
        data_addr: int,
        pc: int,
        source: str,
    ) -> int:
        """Prepare one replayable byte for an input-ready status bit."""
        key = str(stream_key or f"mmio:0x{int(data_addr) & 0xFFFFFFFF:08x}")
        if key not in self._peripheral_input_pending:
            byte = self._next_external_input_byte(key)
            self._peripheral_input_pending[key] = byte
            self.peripheral_input_stats["rx_bytes_prepared"] += 1
            self._record_peripheral_input_event({
                "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
                "symbol": key,
                "kind": "mmio_stream_status",
                "style": "uart_rx_ready",
                "status_addr": f"0x{int(status_addr) & 0xFFFFFFFF:08x}",
                "data_addr": f"0x{int(data_addr) & 0xFFFFFFFF:08x}",
                "byte": int(byte) & 0xFF,
                "return_value": "ready",
                "source": str(source or "peripheral_profile"),
            })
        self.causal_context.record_input_ready(
            key,
            pc=pc,
            address=status_addr,
            source=source or "peripheral_profile",
        )
        self.peripheral_input_stats["rx_ready_status_reads"] += 1
        return int(self._peripheral_input_pending[key]) & 0xFF

    def consume_peripheral_input_byte(
        self,
        stream_key: str,
        *,
        status_addr: int,
        data_addr: int,
        pc: int,
        source: str,
        supplied_byte: Optional[int] = None,
    ) -> int:
        """Consume the byte that justified a previous input-ready status."""
        key = str(stream_key or f"mmio:0x{int(data_addr) & 0xFFFFFFFF:08x}")
        pending_byte = None
        if key in self._peripheral_input_pending:
            pending_byte = int(self._peripheral_input_pending.pop(key)) & 0xFF
        else:
            self.peripheral_input_stats["rx_data_reads_without_ready"] = (
                int(self.peripheral_input_stats.get("rx_data_reads_without_ready", 0) or 0)
                + 1
            )
        if supplied_byte is not None:
            byte = int(supplied_byte) & 0xFF
            if pending_byte is not None and pending_byte != byte:
                self.peripheral_input_stats["rx_overlay_mismatches"] = (
                    int(self.peripheral_input_stats.get("rx_overlay_mismatches", 0) or 0)
                    + 1
                )
        elif pending_byte is not None:
            byte = pending_byte
        else:
            byte = self._next_external_input_byte(key)
            self.peripheral_input_stats["rx_bytes_prepared"] += 1
        self.peripheral_input_stats["rx_bytes_consumed"] += 1
        self.peripheral_input_stats["rx_data_reads"] += 1
        self._record_peripheral_input_event({
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
            "symbol": key,
            "kind": "mmio_stream_byte",
            "style": "uart_data_register",
            "status_addr": f"0x{int(status_addr) & 0xFFFFFFFF:08x}",
            "data_addr": f"0x{int(data_addr) & 0xFFFFFFFF:08x}",
            "return_value": f"0x{byte:02x}",
            "byte": byte,
            "source": str(source or "peripheral_profile"),
        })
        self.causal_context.record_input_consume(
            key,
            pc=pc,
            address=data_addr,
            value=byte,
            source=source or "peripheral_profile",
        )
        return byte

    def _try_overlay_read(self, mmio_addr: int, pc: int, size: int) -> Optional[int]:
        """Return a scoped overlay value when replay supplied explicit evidence."""
        for overlay in reversed(list(self._mmio_overlays)):
            resolver = getattr(overlay, "resolve_bridge_read", None)
            if not callable(resolver):
                continue
            try:
                handled, value = resolver(pc, mmio_addr, size)
            except Exception as exc:
                logger.debug(f"MMIO overlay read resolver failed: {exc}")
                continue
            if handled:
                return int(value) & self._size_mask(size)
        return None

    def _notify_overlay_read(self, pc: int, mmio_addr: int, size: int, value: int) -> None:
        for overlay in list(self._mmio_overlays):
            recorder = getattr(overlay, "record_bridge_read", None)
            if not callable(recorder):
                continue
            try:
                recorder(pc, mmio_addr, size, value)
            except Exception as exc:
                logger.debug(f"MMIO overlay read recorder failed: {exc}")

    def _notify_overlay_write(self, pc: int, mmio_addr: int, size: int, value: int) -> None:
        for overlay in list(self._mmio_overlays):
            recorder = getattr(overlay, "record_bridge_write", None)
            if not callable(recorder):
                continue
            try:
                recorder(pc, mmio_addr, size, value)
            except Exception as exc:
                logger.debug(f"MMIO overlay write recorder failed: {exc}")

    def handle_read(self, mmio_addr: int, pc: int, size: int) -> int:
        """
        处理MMIO读取

        优先级策略（按照idea.md）：
        1. static constraint match
        2. simple pattern (bit flip / ready-after-write)
        3. fallback = 0 or last value

        Args:
            mmio_addr: MMIO地址
            pc: 当前PC
            size: 读取大小

        Returns:
            读取的值
        """
        self.instruction_count += 1
        self.refresh_constraints()
        state = self.get_or_create_state(mmio_addr)

        # 优先级0: 器件自有寄存器（r31 裁定）。中断/节拍寄存器的值由外设状态
        # 决定，固件写镜像不是证据而是副作用——r27 已在轮询退出值上裁定过同一
        # 问题（见 apply_external_loop_exit_input 的注释）：ready/事件位只可能
        # 由外设侧置位。只有显式声明 owns_read 的 profile 参与，故既有 profile
        # 与未注册器件的行为一字不变。
        value = self._try_device_owned_profile_model(mmio_addr, pc, size)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用器件自有profile = 0x{value:08x}")
            self._record_read(state, pc, mmio_addr, value, size, source="device_profile")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级1: replay overlay。临时约束/快照状态只在当前 replay
        # 作用域内生效，避免把 scoped repair 泄漏成全局 MMIO 模型。
        value = self._try_overlay_read(mmio_addr, pc, size)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用scoped replay overlay = 0x{value:08x}")
            self.semantic_profiles.apply_external_read_side_effects(
                mmio_addr,
                pc,
                size,
                value,
                backend=self,
            )
            self._record_read(state, pc, mmio_addr, value, size, source="overlay")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级2: 静态约束匹配
        value = self._try_static_constraint(mmio_addr, pc)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用静态约束 = 0x{value:08x}")
            self.semantic_profiles.apply_external_read_side_effects(
                mmio_addr,
                pc,
                size,
                value,
                backend=self,
            )
            self._record_read(state, pc, mmio_addr, value, size, source="static")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级3: 通用语义 profile。包含经过验证/提升的学习规则，
        # 以及可注册的 Cortex-M/STM32 等硬件完成语义 profile。
        value = self._try_semantic_profile_model(mmio_addr, pc, size)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用通用语义profile = 0x{value:08x}")
            self._record_read(state, pc, mmio_addr, value, size, source="semantic")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级4: legacy fast-status profile。该路径迁移旧 EnhancedMMIOHandler
        # 中合理的 UART TX ready 行为，但只设置 TXE/TC，不默认设置 RXNE。
        value = self._try_legacy_fast_status_model(mmio_addr, pc, size)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用legacy fast-status profile = 0x{value:08x}")
            self._record_read(state, pc, mmio_addr, value, size, source="legacy")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级5: 简单模式
        value = self._try_simple_pattern(mmio_addr, pc, state)
        if value is not None:
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用模式推断 = 0x{value:08x}")
            self._record_read(state, pc, mmio_addr, value, size, source="pattern")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级5.5: 静态初始种子。只在该地址于本仿真器实例中的第一次访问
        # （读、写计数均为 0）落到未建模路径（overlay/静态约束/语义profile/
        # 模式均未命中）时提供初始输入值；此后读取完全由运行时生成/修复
        # 路径接管，种子不再参与（写后读回同样不走种子）。
        if (
            state.read_count == 0
            and state.write_count == 0
            and mmio_addr in self.mmio_seed_values
        ):
            value = int(self.mmio_seed_values[mmio_addr]) & self._size_mask(size)
            self.mmio_seed_stats["applied"] += 1
            logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                        f"使用静态初始种子 = 0x{value:08x}")
            self._record_read(state, pc, mmio_addr, value, size, source="static_seed")
            self._notify_overlay_read(pc, mmio_addr, size, value)
            return value

        # 优先级6: 智能fallback
        value = self._smart_fallback(mmio_addr, pc, state)
        logger.debug(f"MMIO读取 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                    f"使用fallback = 0x{value:08x}")
        self._record_read(state, pc, mmio_addr, value, size, source="fallback")
        self._notify_overlay_read(pc, mmio_addr, size, value)
        return value

    def handle_write(self, mmio_addr: int, pc: int, value: int, size: int):
        """
        处理MMIO写入

        Args:
            mmio_addr: MMIO地址
            pc: 当前PC
            value: 写入的值
            size: 写入大小
        """
        self.instruction_count += 1
        state = self.get_or_create_state(mmio_addr)

        logger.debug(f"MMIO写入 0x{mmio_addr:08x} @ PC 0x{pc:08x}: "
                    f"value = 0x{value:08x}")

        value &= self._size_mask(size)
        previous_value = self._read_stored_register_value(
            self._word_addr(mmio_addr),
            4,
        )
        state.record_access(pc, False, value, self.instruction_count)
        self._store_register_value(mmio_addr, value, size)
        self.semantic_profiles.observe_write(pc, mmio_addr, value, size, self.instruction_count)
        self.semantic_profiles.apply_write(
            mmio_addr,
            size,
            self,
            previous_value=previous_value,
            written_value=value,
            pc=pc,
        )
        self.causal_context.record_mmio_write(pc, mmio_addr, value, size)
        self._notify_overlay_write(pc, mmio_addr, size, value)

        # 推断寄存器类型
        state.infer_register_type()

    def _record_read(
        self,
        state: MMIOState,
        pc: int,
        mmio_addr: int,
        value: int,
        size: int,
        *,
        source: str = "modeled",
    ):
        """Record a read in both the local state and semantic profile registry."""
        value &= self._size_mask(size)
        if self.mmio_source_audit_enabled:
            address = int(mmio_addr) & 0xFFFFFFFF
            bucket = self.mmio_read_source_counts.get(address)
            if bucket is None:
                bucket = Counter()
                self.mmio_read_source_counts[address] = bucket
            bucket[str(source or "modeled")] += 1
        self._remember_read_value(mmio_addr, value, size)
        state.record_access(pc, True, value, self.instruction_count)
        state.infer_register_type()
        self.semantic_profiles.observe_read(pc, mmio_addr, value, size, self.instruction_count)
        self.semantic_profiles.finalize_read_side_effects(
            mmio_addr,
            size,
            self,
        )
        self.causal_context.record_mmio_read(
            pc,
            mmio_addr,
            value,
            size,
            source=source or "modeled",
        )

    @staticmethod
    def _size_mask(size: int) -> int:
        bits = max(1, min(4, int(size or 4))) * 8
        return (1 << bits) - 1

    @staticmethod
    def _word_addr(address: int) -> int:
        return int(address) & ~0x3

    def peek_register_value(self, mmio_addr: int, size: int = 4) -> Optional[int]:
        """Return the modeled register value without recording a read."""
        return self._read_stored_register_value(mmio_addr, size)

    def _read_stored_register_value(self, mmio_addr: int, size: int = 4) -> Optional[int]:
        address = int(mmio_addr) & 0xFFFFFFFF
        size = max(1, min(4, int(size or 4)))
        mask = self._size_mask(size)

        exact = self.mmio_states.get(address)
        if exact is not None and (
            exact.write_count > 0
            or exact.last_write_value is not None
            or exact.current_value != 0
        ):
            return int(exact.current_value) & mask

        word_addr = self._word_addr(address)
        word_state = self.mmio_states.get(word_addr)
        if word_state is None:
            return None
        if not (word_state.write_count > 0 or word_state.last_write_value is not None or word_state.current_value != 0):
            return None

        shift = (address - word_addr) * 8
        return (int(word_state.current_value) >> shift) & mask

    def _set_word_register_value(self, word_addr: int, value: int):
        word_addr = self._word_addr(word_addr) & 0xFFFFFFFF
        word_state = self.get_or_create_state(word_addr)
        word_state.current_value = int(value) & 0xFFFFFFFF
        if word_state.last_write_value is None:
            word_state.last_write_value = word_state.current_value

    def _remember_read_value(self, mmio_addr: int, value: int, size: int):
        """Remember a modeled read value without marking it as a guest write."""
        address = int(mmio_addr) & 0xFFFFFFFF
        size = max(1, min(4, int(size or 4)))
        value = int(value) & self._size_mask(size)
        state = self.get_or_create_state(address)
        state.current_value = value

        word_addr = self._word_addr(address)
        word_state = self.get_or_create_state(word_addr)
        if size >= 4 and address == word_addr:
            word_state.current_value = value
            return

        old_word = int(word_state.current_value) & 0xFFFFFFFF
        shift = (address - word_addr) * 8
        field_mask = self._size_mask(size) << shift
        word_state.current_value = (old_word & ~field_mask) | ((value << shift) & field_mask)

    def _store_register_value(self, mmio_addr: int, value: int, size: int):
        address = int(mmio_addr) & 0xFFFFFFFF
        size = max(1, min(4, int(size or 4)))
        value = int(value) & self._size_mask(size)
        state = self.get_or_create_state(address)
        state.current_value = value

        word_addr = self._word_addr(address)
        if size >= 4 and address == word_addr:
            self._set_word_register_value(word_addr, value)
            return

        old_word = self._read_stored_register_value(word_addr, 4)
        if old_word is None:
            old_word = 0
        shift = (address - word_addr) * 8
        field_mask = self._size_mask(size) << shift
        new_word = (int(old_word) & ~field_mask) | ((value << shift) & field_mask)
        self._set_word_register_value(word_addr, new_word)

    def _extract_register_value(self, word_addr: int, read_addr: int, size: int) -> int:
        word = int(self._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
        shift = (int(read_addr) - self._word_addr(word_addr)) * 8
        return (word >> shift) & self._size_mask(size)

    def _try_device_owned_profile_model(self, mmio_addr: int, pc: int, size: int) -> Optional[int]:
        """Serve a device-owned register straight from its registered profile.

        This deliberately sits *above* the overlay/static-constraint steps: for
        these addresses the profile is the register file, and a firmware write
        mirror would otherwise hand back a stale copy (r31 P2 measured exactly
        that for TIM5 SR/DIER, which silenced the tick handler).
        """
        profiles = getattr(self.semantic_profiles, "profiles", None)
        if not profiles:
            return None
        for profile in profiles:
            owns = getattr(profile, "owns_read", None)
            if not callable(owns):
                continue
            try:
                if not owns(mmio_addr):
                    continue
            except Exception:
                continue
            try:
                value = profile.model_read(mmio_addr, pc, size, self)
            except Exception:
                continue
            if value is None:
                continue
            return int(value) & self._size_mask(size)
        return None

    def _try_semantic_profile_model(self, mmio_addr: int, pc: int, size: int) -> Optional[int]:
        current = self._read_stored_register_value(self._word_addr(mmio_addr), 4)
        return self.semantic_profiles.model_read(mmio_addr, pc, size, current, backend=self)

    @staticmethod
    def _is_legacy_uart_tx_status_register(address: int) -> bool:
        """Common UART/USART TX status registers where TXE/TC-ready is safe."""
        addr = int(address) & 0xFFFFFFFF
        return addr in {
            0x40004400,  # USART2 SR on STM32F1-like layouts
            0x40004800,  # USART3 SR
            0x40004C00,  # UART4 SR
            0x40005000,  # UART5 SR
            0x40007800,  # UART7 SR on some STM32 parts
            0x40007C00,  # UART8 SR on some STM32 parts
            0x40011000,  # USART1 SR on STM32F1
            0x40011400,  # USART6 SR / compatible status register
            0x40013800,  # USART1 SR on STM32F2/F4/libmaple targets
        }

    def _try_legacy_fast_status_model(self, mmio_addr: int, pc: int, size: int) -> Optional[int]:
        if not self.uart_tx_ready_default:
            return None
        if not self._is_legacy_uart_tx_status_register(mmio_addr):
            return None
        # STM32 USART_SR: TXE=bit7, TC=bit6.  RXNE remains clear so input
        # paths still require stream/event modeling rather than unconditional
        # receive readiness.
        self.legacy_uart_tx_ready_hits += 1
        return 0x000000C0 & self._size_mask(size)

    def learn_validated_polling_rule(
        self,
        *,
        mmio_addr: int,
        value: int,
        read_pc: Optional[int],
        constraint_pc: Optional[int],
        loop_head: Optional[int],
        mask: Optional[int],
        source: str,
        confidence: float = 1.0,
    ):
        """Promote a locally validated loop-exit MMIO value into a semantic rule."""
        if mask is None:
            mask = self.semantic_profiles.infer_status_mask_from_svd(mmio_addr)
        return self.semantic_profiles.learn_polling_exit_rule(
            status_address=mmio_addr,
            value=value,
            read_pc=read_pc,
            constraint_pc=constraint_pc,
            loop_head=loop_head,
            mask=mask,
            source=source,
            confidence=confidence,
        )

    def refresh_constraints(self, force: bool = False):
        """Reload persisted constraints when the JSON file changes."""
        if not self.constraint_json_path:
            return
        try:
            if not os.path.exists(self.constraint_json_path):
                return
            mtime = os.path.getmtime(self.constraint_json_path)
            if not force and self._constraint_mtime == mtime:
                return
            loaded = self._load_constraints_from_json(self.constraint_json_path)
            if loaded:
                self.static_constraints.update(loaded)
            self._constraint_mtime = mtime
        except Exception as e:
            logger.debug(f"刷新MMIO约束失败: {e}")

    def _load_constraints_from_json(self, path: str) -> Dict[Tuple[int, int], int]:
        with open(path, "r") as f:
            data = json.load(f)

        constraints: Dict[Tuple[int, int], int] = {}
        if isinstance(data, dict) and "constraints" in data:
            for item in data.get("constraints", []):
                if item.get("type") != "mmio":
                    continue
                # Occurrence-scoped constraints are resolved by the replay
                # overlay, which owns the dynamic read counter.  Loading one
                # here would silently broaden it to every read at this PC.
                if self._parse_int(item.get("read_occurrence")) is not None:
                    continue
                if not self._should_load_file_constraint(item, self.branch_mmio_file_mode):
                    continue
                address = self._parse_int(item.get("address"))
                value = self._parse_int(item.get("value"))
                if "read_pc" in item and item.get("read_pc") is None and item.get("pc") is None:
                    continue
                read_pc = self._parse_int(item.get("read_pc"))
                if read_pc is None:
                    read_pc = self._parse_int(item.get("pc")) or 0
                if address is not None and value is not None:
                    constraints[(read_pc, address)] = value & 0xFFFFFFFF
        elif isinstance(data, dict):
            for address_text, value_text in data.items():
                address = self._parse_int(address_text)
                value = self._parse_int(value_text)
                if address is not None and value is not None:
                    constraints[(0, address)] = value & 0xFFFFFFFF
        return constraints

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
        if speculation_level == "dynamic_replay_required":
            if mode in {"1", "true", "yes", "on", "all", "scoped", "pc", "read-pc"}:
                return item.get("read_pc") is not None or item.get("pc") is not None
            return False
        if added_by != "branch_mmio":
            return True
        if mode in {"1", "true", "yes", "on", "all"}:
            return True
        if mode in {"0", "false", "no", "off", "none"}:
            return False
        if mode in {"scoped", "pc", "read-pc"}:
            return item.get("read_pc") is not None or item.get("pc") is not None
        return item.get("read_pc") is not None or item.get("pc") is not None

    def _parse_int(self, value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            return None

    def _try_static_constraint(self, mmio_addr: int, pc: int) -> Optional[int]:
        """
        尝试使用静态约束

        Returns:
            约束值，如果没有匹配则返回None
        """
        # 查找匹配的约束
        key = (pc, mmio_addr)
        if key in self.static_constraints:
            return self.static_constraints[key]

        # 只允许显式的全局约束 (pc == 0) 作为地址级回退，避免把别的 read_pc 约束泄漏过来
        for (constraint_pc, constraint_mmio), value in self.static_constraints.items():
            if constraint_pc == 0 and constraint_mmio == mmio_addr:
                return value

        return None

    def _try_simple_pattern(self, mmio_addr: int, pc: int, state: MMIOState) -> Optional[int]:
        """
        尝试简单模式

        模式包括：
        1. ready-after-write: 写入后返回ready
        2. bit flip: 某位翻转
        3. 状态寄存器: 返回ready状态

        Returns:
            推断的值，如果无法推断则返回None
        """
        # 模式1: ready-after-write
        if state.last_write_value is not None and state.is_status_register:
            # 如果刚写入过，状态寄存器应该返回ready
            # 假设bit 0是ready位
            return 0x1  # ready

        # 模式2: 检测ready位并翻转
        ready_bit = state.detect_ready_bit()
        if ready_bit is not None:
            # 翻转ready位
            last_values = state.get_last_n_values(1)
            if last_values:
                last_value = last_values[0]
                # 翻转ready位
                return last_value ^ (1 << ready_bit)

        # 模式3: 状态寄存器默认返回ready
        if state.is_status_register and state.read_count > 5:
            # 经过多次轮询后，返回ready
            return 0x1

        return None

    def _smart_fallback(self, mmio_addr: int, pc: int, state: MMIOState) -> int:
        """
        智能fallback策略

        不使用危险的return 1，而是：
        1. 如果有历史值，返回最后一个值
        2. 如果是状态寄存器，返回0（not ready）
        3. 否则返回0

        Args:
            mmio_addr: MMIO地址
            pc: 当前PC
            state: MMIO状态

        Returns:
            fallback值
        """
        # 策略1: 返回最后一个值
        last_values = state.get_last_n_values(1)
        if last_values:
            return last_values[0]

        # 策略2: 如果是状态寄存器，返回0（not ready）
        if state.is_status_register:
            return 0x0

        # 策略3: 默认返回0（而不是1）
        return 0x0

    def update_phase(self, pc: int):
        """
        更新执行阶段

        根据PC历史推断当前执行阶段
        """
        self.pc_history.append(pc)

        # 检测循环
        if pc in self.loop_counters:
            self.loop_counters[pc] += 1
        else:
            self.loop_counters[pc] = 1

        # 推断阶段
        if self.instruction_count < 1000:
            self.current_phase = ExecutionPhase.INIT
        elif any(count > 100 for count in self.loop_counters.values()):
            self.current_phase = ExecutionPhase.POLLING
        else:
            self.current_phase = ExecutionPhase.UNKNOWN

    # Ranges owned by the interrupt/tick profiles (see P2/P5 of r31).
    INTERRUPT_PROFILE_RANGES = (
        ("tim5", 0x40000C00, 0x40000FFF),
        ("scs_icsr", 0xE000ED04, 0xE000ED04),
        ("otgfs", 0x50000000, 0x5003FFFF),
    )

    def interrupt_register_read_sources(self) -> Dict[str, Dict[str, object]]:
        """Attribute reads of the interrupt/tick registers to a read-path level.

        Proves whether a registered profile (priority 3) actually served the
        register, or whether a firmware write mirror (priority 1) / file
        constraint (priority 2) masked it.
        """
        payload: Dict[str, Dict[str, object]] = {}
        for name, start, end in self.INTERRUPT_PROFILE_RANGES:
            entries: Dict[str, object] = {}
            for address, bucket in sorted(self.mmio_read_source_counts.items()):
                if start <= address <= end:
                    entries[f"0x{address:08x}"] = dict(bucket)
            payload[name] = entries
        return payload

    def get_state_summary(self) -> Dict:
        """获取状态摘要"""
        return {
            "instruction_count": self.instruction_count,
            "current_phase": self.current_phase.value,
            "mmio_count": len(self.mmio_states),
            "loop_counters": dict(list(self.loop_counters.items())[:10]),  # 前10个
            "semantic_profiles": self.semantic_profiles.summary(),
        }

    def get_mmio_statistics(self) -> Dict:
        """获取MMIO统计信息"""
        stats = {
            "total_mmio_addresses": len(self.mmio_states),
            "status_registers": 0,
            "control_registers": 0,
            "data_registers": 0,
            "total_reads": 0,
            "total_writes": 0,
            "legacy_uart_tx_ready_hits": int(self.legacy_uart_tx_ready_hits),
            "peripheral_input": dict(self.peripheral_input_stats),
        }

        for state in self.mmio_states.values():
            if state.is_status_register:
                stats["status_registers"] += 1
            if state.is_control_register:
                stats["control_registers"] += 1
            if state.is_data_register:
                stats["data_registers"] += 1

            stats["total_reads"] += state.read_count
            stats["total_writes"] += state.write_count

        return stats
