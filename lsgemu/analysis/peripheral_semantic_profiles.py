#!/usr/bin/env python3
"""Generic MMIO semantic profiles.

This module is intentionally conservative.  It does not guess peripheral
values from vendor base addresses.  Instead it records access sequences and
only replays a semantic status value when the value was validated by a local
load/test/branch solver or supported by SVD metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import logging
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import Counter, deque

from ..causal_context import CausalExecutionContext

logger = logging.getLogger(__name__)


def _size_mask(size: int) -> int:
    bits = max(1, min(4, int(size or 4))) * 8
    return (1 << bits) - 1


# ── r33：时钟/电源就绪位的器件语义开关与确定性延迟 ─────────────────────────
# 就绪位（HSIRDY/HSERDY/PLLRDY/LSIRDY/VOSRDY/SWS）不是固件能写出来的东西：
# 固件只能写"使能"，器件在自身时序完成后回置"就绪"。这条语义必须能整体关掉，
# 否则无法证明"是模型在起作用"（r33 P3 反例臂 1）。
CLOCK_READY_DISABLE_ENV = "LSGEMU_DISABLE_CLOCK_READY_SEMANTICS"
READY_LATENCY_ENV = "LSGEMU_READY_LATENCY_ACCESSES"
# 就绪延迟 = 若干次"被模拟的 MMIO 访问"（StatefulMMIOHandler.instruction_count）。
# 1 表示：置使能的那次访问看不到就绪，下一次访问才看到——器件语义上"就绪绝不
# 与置位同拍"。用计数器而非墙钟，是为了快照/重放可复现（C5）。
DEFAULT_READY_LATENCY_ACCESSES = 1


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def clock_ready_semantics_enabled() -> bool:
    """r33 P3-1：``LSGEMU_DISABLE_CLOCK_READY_SEMANTICS=1`` 关掉本轮新语义。"""
    return not _env_truthy(CLOCK_READY_DISABLE_ENV)


def ready_latency_accesses() -> int:
    """确定性就绪延迟（单位：模拟 MMIO 访问计数）。"""
    try:
        return max(0, int(os.environ.get(READY_LATENCY_ENV, str(DEFAULT_READY_LATENCY_ACCESSES)), 0))
    except (TypeError, ValueError):
        return DEFAULT_READY_LATENCY_ACCESSES


STATUS_FIELD_TOKENS = (
    "ready",
    "rdy",
    "done",
    "complete",
    "comp",
    "ack",
    "txe",
    "tc",
    "rxne",
    "busy",
    "flag",
    "if",
    "irq",
    "pending",
)


@dataclass
class MMIOAccessEvent:
    pc: int
    address: int
    is_read: bool
    value: int
    size: int
    timestamp: int
    transaction_sequence: int = 0


@dataclass
class PeripheralTransactionState:
    peripheral_key: str
    sequence: int = 0
    phase: str = "idle"
    last_write_by_address: Dict[int, MMIOAccessEvent] = field(default_factory=dict)
    last_read: Optional[MMIOAccessEvent] = None
    completion_count: int = 0
    conflicting_write_count: int = 0

    def summary(self) -> Dict[str, object]:
        return {
            "sequence": int(self.sequence),
            "phase": str(self.phase),
            "last_write_addresses": [
                f"0x{int(address) & 0xFFFFFFFF:08x}"
                for address in sorted(self.last_write_by_address)
            ][-16:],
            "completion_count": int(self.completion_count),
            "conflicting_write_count": int(self.conflicting_write_count),
        }


@dataclass
class SVDFieldInfo:
    name: str
    bit_offset: int
    bit_width: int
    description: str = ""
    read_action: Optional[str] = None
    modified_write_values: Optional[str] = None

    @property
    def mask(self) -> int:
        width = max(1, min(32, int(self.bit_width or 1)))
        offset = max(0, min(31, int(self.bit_offset or 0)))
        return ((1 << width) - 1) << offset if width < 32 else 0xFFFFFFFF

    def has_status_name(self) -> bool:
        text = f"{self.name} {self.description}".lower()
        return any(token in text for token in STATUS_FIELD_TOKENS)

    def normalized_read_action(self) -> str:
        return str(self.read_action or "").strip().lower()

    def normalized_write_action(self) -> str:
        return str(self.modified_write_values or "").strip().lower()


@dataclass
class SVDRegisterInfo:
    address: int
    peripheral_name: str
    register_name: str
    size: int = 32
    fields: List[SVDFieldInfo] = field(default_factory=list)
    read_action: Optional[str] = None
    modified_write_values: Optional[str] = None

    def status_field_mask(self) -> Optional[int]:
        mask = 0
        for field_info in self.fields:
            if field_info.has_status_name():
                mask |= field_info.mask
        return mask & 0xFFFFFFFF if mask else None

    def is_status_like(self) -> bool:
        text = f"{self.peripheral_name} {self.register_name}".lower()
        if any(token in text for token in STATUS_FIELD_TOKENS):
            return True
        return self.status_field_mask() is not None

    def read_clear_mask(self) -> int:
        mask = 0
        register_action = str(self.read_action or "").strip().lower()
        if register_action in {"clear", "clearonread", "clear-on-read"}:
            mask = 0xFFFFFFFF
        for field_info in self.fields:
            if field_info.normalized_read_action() in {
                "clear", "clearonread", "clear-on-read"
            }:
                mask |= field_info.mask
        return mask & 0xFFFFFFFF

    def write_action_masks(self) -> Dict[str, int]:
        masks: Dict[str, int] = {}
        register_action = str(self.modified_write_values or "").strip().lower()
        if register_action:
            masks[register_action] = 0xFFFFFFFF
        for field_info in self.fields:
            action = field_info.normalized_write_action()
            if action:
                masks[action] = int(masks.get(action, 0)) | int(field_info.mask)
        return {str(key): int(value) & 0xFFFFFFFF for key, value in masks.items()}


class SVDRegisterIndex:
    """Small CMSIS-SVD register lookup used as optional semantic evidence."""

    def __init__(self, paths: Optional[Sequence[str]] = None):
        self.registers: Dict[int, SVDRegisterInfo] = {}
        self.peripheral_ranges: List[Tuple[int, int, str]] = []
        for path_text in paths or []:
            path = Path(path_text).expanduser()
            if not path.exists():
                continue
            try:
                self._load(path)
            except Exception as exc:
                logger.debug("SVD加载失败 %s: %s", path, exc)

    @classmethod
    def from_environment(cls) -> "SVDRegisterIndex":
        raw = os.environ.get("LSGEMU_SVD_PATHS") or os.environ.get("LSGEMU_SVD_PATH") or ""
        paths = [item for item in raw.split(os.pathsep) if item.strip()]
        return cls(paths)

    def _load(self, path: Path) -> None:
        tree = ET.parse(str(path))
        root = tree.getroot()
        peripherals = self._children(root, "peripherals")
        for peripherals_node in peripherals:
            for peripheral in self._children(peripherals_node, "peripheral"):
                self._load_peripheral(peripheral)

    def _load_peripheral(self, node: ET.Element) -> None:
        name = self._text(node, "name") or "PERIPHERAL"
        base = self._parse_int(self._text(node, "baseAddress"))
        if base is None:
            return
        max_offset = 0
        registers_nodes = self._children(node, "registers")
        for registers_node in registers_nodes:
            for register in self._children(registers_node, "register"):
                offset = self._parse_int(self._text(register, "addressOffset"))
                if offset is None:
                    continue
                reg_name = self._text(register, "name") or f"REG_{offset:x}"
                size_bits = self._parse_int(self._text(register, "size")) or 32
                register_read_action = self._text(register, "readAction")
                register_write_action = self._text(register, "modifiedWriteValues")
                fields = self._load_fields(
                    register,
                    inherited_read_action=register_read_action,
                    inherited_write_action=register_write_action,
                )
                address = (base + offset) & 0xFFFFFFFF
                self.registers[address] = SVDRegisterInfo(
                    address=address,
                    peripheral_name=name,
                    register_name=reg_name,
                    size=int(size_bits),
                    fields=fields,
                    read_action=register_read_action,
                    modified_write_values=register_write_action,
                )
                max_offset = max(max_offset, int(offset) + max(4, int(size_bits + 7) // 8))
        if max_offset:
            self.peripheral_ranges.append((base & 0xFFFFFFFF, (base + max_offset) & 0xFFFFFFFF, name))

    def _load_fields(
        self,
        register: ET.Element,
        *,
        inherited_read_action: Optional[str] = None,
        inherited_write_action: Optional[str] = None,
    ) -> List[SVDFieldInfo]:
        fields: List[SVDFieldInfo] = []
        for fields_node in self._children(register, "fields"):
            for field_node in self._children(fields_node, "field"):
                name = self._text(field_node, "name") or "FIELD"
                description = self._text(field_node, "description") or ""
                bit_offset = self._parse_int(self._text(field_node, "bitOffset"))
                bit_width = self._parse_int(self._text(field_node, "bitWidth"))
                if bit_offset is None:
                    bit_range = self._text(field_node, "bitRange")
                    if bit_range and ":" in bit_range:
                        left, right = bit_range.strip("[]").split(":", 1)
                        high = self._parse_int(left)
                        low = self._parse_int(right)
                        if high is not None and low is not None:
                            bit_offset = min(high, low)
                            bit_width = abs(high - low) + 1
                if bit_offset is None:
                    lsb = self._parse_int(self._text(field_node, "lsb"))
                    msb = self._parse_int(self._text(field_node, "msb"))
                    if lsb is not None and msb is not None:
                        bit_offset = min(lsb, msb)
                        bit_width = abs(msb - lsb) + 1
                if bit_offset is None:
                    continue
                fields.append(
                    SVDFieldInfo(
                        name=name,
                        bit_offset=int(bit_offset),
                        bit_width=int(bit_width or 1),
                        description=description,
                        read_action=(
                            self._text(field_node, "readAction")
                            or inherited_read_action
                        ),
                        modified_write_values=(
                            self._text(field_node, "modifiedWriteValues")
                            or inherited_write_action
                        ),
                    )
                )
        return fields

    def register_for(self, address: int) -> Optional[SVDRegisterInfo]:
        return self.registers.get(int(address) & 0xFFFFFFFF)

    def peripheral_key_for(self, address: int) -> str:
        addr = int(address) & 0xFFFFFFFF
        for start, end, name in self.peripheral_ranges:
            if start <= addr < end:
                return f"svd:{name}:0x{start:08x}"
        return f"page:0x{addr & ~0x3ff:08x}"

    @staticmethod
    def _children(node: ET.Element, tag: str) -> List[ET.Element]:
        return [child for child in list(node) if child.tag.split("}")[-1] == tag]

    @classmethod
    def _text(cls, node: ET.Element, tag: str) -> Optional[str]:
        for child in cls._children(node, tag):
            if child.text is not None:
                return child.text.strip()
        return None

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text, 0)
        except ValueError:
            return None


@dataclass
class SemanticMMIORule:
    status_address: int
    value: int
    mask: Optional[int]
    read_pc: Optional[int]
    constraint_pc: Optional[int]
    loop_head: Optional[int]
    source: str
    confidence: float = 1.0
    related_write_address: Optional[int] = None
    related_write_pc: Optional[int] = None
    related_write_value: Optional[int] = None
    related_write_timestamp: Optional[int] = None
    evidence_read_pcs: Set[int] = field(default_factory=set)
    promoted: bool = False
    kind: str = "polling_exit"
    transaction_signature: Optional[str] = None

    def applies_to(self, address: int, pc: int, registry: Optional["SemanticProfileRegistry"] = None) -> bool:
        if (int(address) & 0xFFFFFFFF) != (int(self.status_address) & 0xFFFFFFFF):
            return False
        if self.related_write_address is not None and registry is not None:
            if not registry.recent_transaction_matches(self):
                return False
        if self.promoted:
            return True
        return self.read_pc is not None and (int(pc) & 0xFFFFFFFF) == (int(self.read_pc) & 0xFFFFFFFF)

    def modeled_value(self, current: int) -> int:
        current &= 0xFFFFFFFF
        value = int(self.value) & 0xFFFFFFFF
        if self.mask is None:
            return value
        mask = int(self.mask) & 0xFFFFFFFF
        return ((current & ~mask) | (value & mask)) & 0xFFFFFFFF


class PeripheralSemanticProfile:
    """Base interface for deterministic peripheral completion semantics."""

    name = "generic"

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        return None

    def owns_read(self, address: int) -> bool:
        """True when this profile is the *authority* for a read of ``address``.

        Device-owned interrupt/tick registers must not be served by the firmware
        write mirror (an overlay's ``mmio_state`` branch): the value there is a
        side effect of the guest's own store, not replay evidence, and a stale
        mirror silently defeats an event latch.  Only profiles that model
        device state (not polling-exit heuristics) opt in — the default keeps
        every pre-existing profile at its historical priority.
        """
        return False

    def apply_external_read_side_effects(
        self,
        address: int,
        pc: int,
        size: int,
        value: int,
        backend,
    ) -> bool:
        """Apply state transitions for a value supplied by a replay overlay.

        This is intentionally separate from ``model_read``: invoking the model
        would generate a second value and could consume a FIFO byte twice.
        """
        return False

    def apply_write(self, address: int, size: int, backend) -> bool:
        return False

    @staticmethod
    def _word_addr(address: int) -> int:
        return int(address) & ~0x3

    @staticmethod
    def _extract_register_value(backend, word_addr: int, read_addr: int, size: int) -> int:
        word = int(backend._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
        shift = (int(read_addr) - (int(word_addr) & ~0x3)) * 8
        return (word >> shift) & _size_mask(size)


class CortexMSysTickProfile(PeripheralSemanticProfile):
    """Cortex-M system timer model, independent of MCU vendor."""

    name = "cortexm_systick"

    def __init__(self, step: Optional[int] = None):
        self.base = 0xE000E010
        self.current = 0
        self.countflag = False
        if step is None:
            try:
                step = max(1, int(os.environ.get("LSGEMU_SYSTICK_STEP", "0x400"), 0))
            except ValueError:
                step = 0x400
        self.step = int(step)

    def _matches(self, address: int) -> bool:
        return self.base <= (int(address) & 0xFFFFFFFF) < self.base + 0x10

    def _reload_value(self, backend) -> int:
        reload_value = int(backend._read_stored_register_value(self.base + 0x04, 4) or 0) & 0x00FFFFFF
        return reload_value if reload_value else 0x00FFFFFF

    def _advance(self, backend) -> int:
        csr = int(backend._read_stored_register_value(self.base + 0x00, 4) or 0) & 0xFFFFFFFF
        if not (csr & 0x1):
            return int(self.current) & 0x00FFFFFF
        reload_value = self._reload_value(backend)
        if self.current <= 0:
            self.current = reload_value
        if self.current <= self.step:
            self.current = reload_value
            self.countflag = True
        else:
            self.current -= self.step
        self.current &= 0x00FFFFFF
        backend._set_word_register_value(self.base + 0x08, self.current)
        return self.current

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        address = int(address) & 0xFFFFFFFF
        if not self._matches(address):
            return None
        offset = address - self.base
        word_addr = self._word_addr(address)
        if offset == 0x00:
            csr = int(backend._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
            if self.countflag:
                csr |= 1 << 16
            else:
                csr &= ~(1 << 16)
            self.countflag = False
            backend._set_word_register_value(word_addr, csr & ~(1 << 16))
            return self._extract_register_value(backend, word_addr, address, size) | (csr & _size_mask(size))
        if offset == 0x04:
            if backend._read_stored_register_value(word_addr, 4) is None:
                backend._set_word_register_value(word_addr, 0x00FFFFFF)
            return self._extract_register_value(backend, word_addr, address, size)
        if offset == 0x08:
            self._advance(backend)
            return self._extract_register_value(backend, word_addr, address, size)
        if offset == 0x0C:
            backend._set_word_register_value(word_addr, 0x0000270F)
            return self._extract_register_value(backend, word_addr, address, size)
        return None

    def apply_write(self, address: int, size: int, backend) -> bool:
        address = self._word_addr(int(address) & 0xFFFFFFFF)
        if not self._matches(address):
            return False
        offset = address - self.base
        if offset == 0x00:
            csr = int(backend._read_stored_register_value(self.base, 4) or 0) & 0xFFFFFFFF
            backend._set_word_register_value(self.base, csr & ~(1 << 16))
            self.countflag = False
            return True
        if offset == 0x04:
            self.current = self._reload_value(backend)
            backend._set_word_register_value(self.base + 0x08, self.current)
            return True
        if offset == 0x08:
            self.current = self._reload_value(backend)
            self.countflag = False
            backend._set_word_register_value(self.base + 0x08, self.current)
            return True
        return False


class STM32RCCProfile(PeripheralSemanticProfile):
    """STM32 RCC clock-ready and clock-switch completion semantics.

    r33 P1：就绪位是**器件**对"使能写"的响应，不是固件写镜像的一部分。
    因此本 profile 对本组寄存器声明 ``owns_read``（读路径优先级 0），并把
    使能位→就绪位做成带**确定性延迟**的状态机：

      * ``CR``：HSION→HSIRDY / HSEON→HSERDY / PLLON→PLLRDY / PLLI2SON→PLLI2SRDY；
      * ``CSR``：LSION→LSIRDY（并使 RMVF 表现为写 1 清、读回 0）；
      * ``CFGR``：SW→SWS 在切换完成后跟随。

    使能写 0 ⇒ 就绪位必须转 0（r33 P3-2 反例臂钉死）。
    """

    name = "stm32_rcc"

    # (使能位, 就绪位) —— 仅收录 RCC_CR 里成对的 oscillator/PLL 位。
    # 注意：bit8 是 HSICAL[0]（只读校准值），不是使能位；历史实现把它当作
    # "使能→bit10" 的假就绪对，r33 删除（那是"永远返回就绪"的一类缺陷）。
    CR_READY_PAIRS = (
        (0x00000001, 0x00000002),  # HSION  -> HSIRDY
        (0x00010000, 0x00020000),  # HSEON  -> HSERDY
        (0x01000000, 0x02000000),  # PLLON  -> PLLRDY
        (0x04000000, 0x08000000),  # PLLI2SON -> PLLI2SRDY
    )
    CSR_READY_PAIRS = (
        (0x00000001, 0x00000002),  # LSION -> LSIRDY
    )
    CSR_RESET_FLAG_MASK = 0xFF000000  # LPWRRSTF..RMVF（bit31:24）
    CSR_RMVF = 0x01000000            # bit24：写 1 清复位标志，读回 0

    def __init__(self):
        # per-base 寄存器布局：F1 (0x40021000) 与 F4 (0x40023800)。
        self.bases = {
            0x40021000: {"cfgr_offsets": {0x04, 0x08}, "csr_offset": 0x24},
            0x40023800: {"cfgr_offsets": {0x08}, "csr_offset": 0x74},
        }
        self.enabled = clock_ready_semantics_enabled()
        self.latency = ready_latency_accesses()
        # (寄存器字地址, 就绪位) -> 就绪可见的那次访问计数
        self._ready_at: Dict[Tuple[int, int], int] = {}
        # 系统时钟切换：字地址 -> (请求的 SW, 完成时刻)；字地址 -> 已完成 SWS
        self._sw_pending: Dict[int, Tuple[int, int]] = {}
        self._sws: Dict[int, int] = {}

    # -- 工具 -------------------------------------------------------------
    def _base_for(self, address: int) -> Optional[int]:
        address = int(address) & 0xFFFFFFFF
        for base in self.bases:
            if base <= address < base + 0x400:
                return base
        return None

    def _layout(self, base: int) -> Dict[str, object]:
        return dict(self.bases.get(int(base), {}) or {})

    @staticmethod
    def _now(backend) -> Optional[int]:
        """模拟访问计数（快照内可复现）；不可得时退回"零延迟"。"""
        value = getattr(backend, "instruction_count", None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _default_word_value(offset: int) -> int:
        # RCC_CR 复位值含 HSION|HSIRDY（HSI 是复位后的系统时钟）。
        return 0x00000003 if offset == 0x00 else 0

    def _owned_word_addrs(self, base: int) -> Set[int]:
        layout = self._layout(base)
        offsets = {0x00}
        offsets |= set(layout.get("cfgr_offsets", set()) or set())
        offsets.add(int(layout.get("csr_offset", -1)))
        return {int(base) + int(offset) for offset in offsets if int(offset) >= 0}

    def _owned_offsets(self, base: int) -> Set[int]:
        return {int(addr) - int(base) for addr in self._owned_word_addrs(base)}

    def owns_read(self, address: int) -> bool:
        """r33 P2：就绪位只能由器件置位，固件写镜像不是证据。

        只有显式声明 owns_read 的 profile 走读路径优先级 0（r31 P2 裁定），
        所以这一声明**只**改变 RCC 自己认领的 5 个字，其它 profile 一字不动。
        """
        if not self.enabled:
            return False
        base = self._base_for(address)
        if base is None:
            return False
        return (int(self._word_addr(address)) - int(base)) in self._owned_offsets(base)

    # -- 就绪位状态机 -----------------------------------------------------
    def _apply_ready_pairs(
        self,
        register_key: int,
        value: int,
        pairs: Sequence[Tuple[int, int]],
        now: Optional[int],
    ) -> int:
        value = int(value) & 0xFFFFFFFF
        for enable_bit, ready_bit in pairs:
            key = (int(register_key), int(ready_bit))
            if value & enable_bit:
                ready_at = self._ready_at.get(key)
                if ready_at is None:
                    ready_at = (now or 0) + int(self.latency)
                    self._ready_at[key] = ready_at
                if now is None or now >= ready_at:
                    value |= ready_bit
                else:
                    value &= ~ready_bit
            else:
                # 使能写 0 ⇒ 就绪位必须转 0（真实器件行为：振荡器停振）。
                self._ready_at.pop(key, None)
                value &= ~ready_bit
        return value & 0xFFFFFFFF

    def _apply_switch_status(self, word_addr: int, value: int, now: Optional[int]) -> int:
        value = int(value) & 0xFFFFFFFF
        requested = value & 0x3
        pending = self._sw_pending.get(int(word_addr))
        if pending is None or int(pending[0]) != requested:
            pending = (requested, (now or 0) + int(self.latency))
            self._sw_pending[int(word_addr)] = pending
        req, ready_at = int(pending[0]), int(pending[1])
        completed = now is None or now >= ready_at
        if completed:
            self._sws[int(word_addr)] = req
            self._sw_pending.pop(int(word_addr), None)
            sws = req
        else:
            # 切换未完成前，SWS 仍反映**旧**时钟源（复位后是 HSI=0）。
            sws = int(self._sws.get(int(word_addr), 0)) & 0x3
        return ((value & ~0xC) | (sws << 2)) & 0xFFFFFFFF

    def _apply_csr_bits(self, word_addr: int, value: int, now: Optional[int]) -> int:
        value = self._apply_ready_pairs(
            int(word_addr), value, self.CSR_READY_PAIRS, now
        )
        # RMVF 是"写 1 清"触发器，且读回永远为 0。
        return value & ~self.CSR_RMVF & 0xFFFFFFFF

    @staticmethod
    def _cr_with_ready_bits(value: int) -> int:
        """r33 之前的组合逻辑（无延迟、无 owns_read）——只用于"关语义"反例臂。"""
        value = int(value) & 0xFFFFFFFF
        for enable_bit, ready_bit in STM32RCCProfile.CR_READY_PAIRS:
            if value & enable_bit:
                value |= ready_bit
            else:
                value &= ~ready_bit
        return value & 0xFFFFFFFF

    @staticmethod
    def _cfgr_with_switch_status(value: int) -> int:
        """r33 之前的组合逻辑（SWS 立即等于 SW）——只用于"关语义"反例臂。"""
        value = int(value) & 0xFFFFFFFF
        requested = value & 0x3
        return ((value & ~0xC) | (requested << 2)) & 0xFFFFFFFF

    def _apply_word(self, base: int, word_addr: int, backend) -> Optional[int]:
        offset = int(word_addr) - int(base)
        layout = self._layout(base)
        if not self.enabled:
            # 关语义臂（LSGEMU_DISABLE_CLOCK_READY_SEMANTICS=1）：逐字复现
            # r33 之前的组合逻辑，让 A/B 成为严格的本轮前后对照。
            if offset == 0x00:
                current = backend._read_stored_register_value(word_addr, 4)
                if current is None:
                    current = self._default_word_value(offset)
                modeled = self._cr_with_ready_bits(current)
                backend._set_word_register_value(word_addr, modeled)
                return modeled
            if offset in set(layout.get("cfgr_offsets", set()) or set()):
                current = backend._read_stored_register_value(word_addr, 4)
                if current is None:
                    current = self._default_word_value(offset)
                modeled = self._cfgr_with_switch_status(current)
                backend._set_word_register_value(word_addr, modeled)
                return modeled
            return None

        cfgr_offsets = set(layout.get("cfgr_offsets", set()) or set())
        csr_offset = layout.get("csr_offset")
        now = self._now(backend)

        if offset == 0x00:
            current = backend._read_stored_register_value(word_addr, 4)
            if current is None:
                current = self._default_word_value(offset)
            modeled = self._apply_ready_pairs(
                word_addr, current, self.CR_READY_PAIRS, now
            )
            backend._set_word_register_value(word_addr, modeled)
            return modeled

        if csr_offset is not None and offset == int(csr_offset):
            current = backend._read_stored_register_value(word_addr, 4)
            if current is None:
                current = 0
            modeled = self._apply_csr_bits(word_addr, current, now)
            backend._set_word_register_value(word_addr, modeled)
            return modeled

        if offset in cfgr_offsets:
            current = backend._read_stored_register_value(word_addr, 4)
            if current is None:
                current = self._default_word_value(offset)
            modeled = self._apply_switch_status(word_addr, current, now)
            backend._set_word_register_value(word_addr, modeled)
            return modeled
        return None

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        base = self._base_for(address)
        if base is None:
            return None
        word_addr = self._word_addr(address)
        if self._apply_word(base, word_addr, backend) is None:
            return None
        return self._extract_register_value(backend, word_addr, address, size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        base = self._base_for(address)
        if base is None:
            return False
        word_addr = self._word_addr(address)
        offset = int(word_addr) - int(base)
        # 写侧只登记"使能时刻"，值本身由 handle_write 的写镜像保存。
        layout = self._layout(base)
        if self.enabled and offset in self._owned_offsets(base):
            current = backend._read_stored_register_value(word_addr, 4)
            if current is None:
                current = self._default_word_value(offset)
            if offset == 0x00:
                self._apply_ready_pairs(word_addr, current, self.CR_READY_PAIRS, self._now(backend))
            elif layout.get("csr_offset") is not None and offset == int(layout["csr_offset"]):
                self._apply_ready_pairs(word_addr, current, self.CSR_READY_PAIRS, self._now(backend))
                self._clear_csr_reset_flags(word_addr, current, backend)
            elif offset in set(layout.get("cfgr_offsets", set()) or set()):
                self._apply_switch_status(word_addr, current, self._now(backend))
            return True
        return self._apply_word(base, word_addr, backend) is not None

    def _clear_csr_reset_flags(self, word_addr: int, value: int, backend) -> None:
        """RCC_CSR.RMVF 写 1 清复位标志（RM0090，RMVF 读回 0）。"""
        if int(value) & self.CSR_RMVF:
            backend._set_word_register_value(
                word_addr, int(value) & ~self.CSR_RESET_FLAG_MASK & 0xFFFFFFFF
            )

    # -- 快照 -------------------------------------------------------------
    def snapshot_state(self) -> Dict[str, object]:
        return {
            "ready_at": {f"{a}:{b}": int(t) for (a, b), t in self._ready_at.items()},
            "sw_pending": {
                str(addr): [int(req), int(at)]
                for addr, (req, at) in self._sw_pending.items()
            },
            "sws": {str(addr): int(v) for addr, v in self._sws.items()},
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self._ready_at = {}
        for key, value in dict(state.get("ready_at", {}) or {}).items():
            try:
                addr_text, bit_text = str(key).split(":")
                self._ready_at[(int(addr_text), int(bit_text))] = int(value)
            except (TypeError, ValueError):
                continue
        self._sw_pending = {}
        for addr, pair in dict(state.get("sw_pending", {}) or {}).items():
            try:
                self._sw_pending[int(addr)] = (int(pair[0]), int(pair[1]))
            except (TypeError, ValueError, IndexError):
                continue
        self._sws = {}
        for addr, value in dict(state.get("sws", {}) or {}).items():
            try:
                self._sws[int(addr)] = int(value)
            except (TypeError, ValueError):
                continue

    def summary(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "latency_accesses": int(self.latency),
            "pending_ready": len(self._ready_at),
            "pending_switch": len(self._sw_pending),
            "switch_status": {
                f"0x{addr:08x}": int(value) for addr, value in sorted(self._sws.items())
            },
        }


class STM32PWRProfile(PeripheralSemanticProfile):
    """STM32 PWR：VOS 电压档 → PWR_CSR.VOSRDY 器件语义。

    r33 P1：固件写 ``PWR_CR.VOS``（本固件写 0xC000 ⇒ VOS=0b11），器件在自身
    时序完成后回置 ``PWR_CSR.VOSRDY``(bit14)。就绪位不可由固件写镜像供给。
    """

    name = "stm32_pwr"
    BASE = 0x40007000
    CR_OFFSET = 0x00
    CSR_OFFSET = 0x04
    VOS_MASK = 0x0000C000
    VOSRDY = 0x00004000
    PVDO = 0x00000004
    # RM0090（F42x/F43x）：PWR_CR 复位值 0x0000C000 ⇒ VOS=0b11。
    CR_RESET_VALUE = 0x0000C000

    def __init__(self):
        self.enabled = clock_ready_semantics_enabled()
        self.latency = ready_latency_accesses()
        self._vosrdy_at: Optional[int] = None

    def _in_range(self, address: int) -> bool:
        addr = int(address) & 0xFFFFFFFF
        return self.BASE <= addr < self.BASE + 0x400

    def _word_offset(self, address: int) -> int:
        return int(self._word_addr(address)) - self.BASE

    def _cr_word(self, backend) -> int:
        value = backend._read_stored_register_value(self.BASE + self.CR_OFFSET, 4)
        if value is None:
            return int(self.CR_RESET_VALUE) & 0xFFFFFFFF
        return int(value) & 0xFFFFFFFF

    def owns_read(self, address: int) -> bool:
        if not self.enabled or not self._in_range(address):
            return False
        return self._word_offset(address) in (self.CR_OFFSET, self.CSR_OFFSET)

    @staticmethod
    def _now(backend) -> Optional[int]:
        value = getattr(backend, "instruction_count", None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _with_ready(self, value: int, vos: int, now: Optional[int]) -> int:
        value = int(value) & 0xFFFFFFFF
        if vos & 0x3:
            if self._vosrdy_at is None:
                self._vosrdy_at = (now or 0) + int(self.latency)
            if now is None or now >= int(self._vosrdy_at):
                value |= self.VOSRDY
            else:
                value &= ~self.VOSRDY
        else:
            # VOS 写 0 ⇒ 稳压器档位无效 ⇒ 就绪位必须转 0。
            self._vosrdy_at = None
            value &= ~self.VOSRDY
        return value & 0xFFFFFFFF

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        if not self._in_range(address):
            return None
        offset = self._word_offset(address)
        if offset not in (self.CR_OFFSET, self.CSR_OFFSET):
            return None
        word_addr = self.BASE + offset
        current = backend._read_stored_register_value(word_addr, 4)
        if current is None:
            current = self.CR_RESET_VALUE if offset == self.CR_OFFSET else 0
        if offset == self.CR_OFFSET:
            # PWR_CR 自身：VOS 就在 bit15:14，原样回读固件写的档位。
            # （VOSRDY 只在 PWR_CSR 里，绝不写回 CR——bit14 是 VOS[0]。）
            return self._extract_register_value(backend, word_addr, address, size)
        vos = (self._cr_word(backend) & self.VOS_MASK) >> 14
        modeled = self._with_ready(current, vos, self._now(backend))
        backend._set_word_register_value(word_addr, modeled)
        return self._extract_register_value(backend, word_addr, address, size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        if not self.enabled or not self._in_range(address):
            return False
        offset = self._word_offset(address)
        if offset not in (self.CR_OFFSET, self.CSR_OFFSET):
            return False
        if offset == self.CR_OFFSET:
            # 登记就绪时刻（写镜像里已有固件写的 VOS）。
            vos = (self._cr_word(backend) & self.VOS_MASK) >> 14
            self._with_ready(0, vos, self._now(backend))
        return True

    def snapshot_state(self) -> Dict[str, object]:
        return {"vosrdy_at": None if self._vosrdy_at is None else int(self._vosrdy_at)}

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        raw = state.get("vosrdy_at")
        try:
            self._vosrdy_at = None if raw is None else int(raw)
        except (TypeError, ValueError):
            self._vosrdy_at = None

    def summary(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "latency_accesses": int(self.latency),
            "vosrdy_at": None if self._vosrdy_at is None else int(self._vosrdy_at),
        }


class STM32USARTProfile(PeripheralSemanticProfile):
    """STM32 USART ready/status completion profile."""

    name = "stm32_usart"
    LEGACY_RXNE = 1 << 5
    LEGACY_TC = 1 << 6
    LEGACY_TXE = 1 << 7
    V2_RXNE = 1 << 5
    V2_TC = 1 << 6
    V2_TXE = 1 << 7

    def __init__(self):
        self.bases = {
            0x40004400,
            0x40004800,
            0x40004C00,
            0x40005000,
            0x40008000,
            0x40011000,
            0x40011400,
            0x40013800,
        }
        self.stats: Dict[str, int] = {
            "legacy_status_reads": 0,
            "legacy_data_reads": 0,
            "v2_status_reads": 0,
            "v2_data_reads": 0,
            "rx_ready_injected": 0,
            "rx_bytes_returned": 0,
        }

    def _base_for(self, address: int) -> Optional[int]:
        address = int(address) & 0xFFFFFFFF
        for base in self.bases:
            if base <= address < base + 0x400:
                return base
        return None

    @staticmethod
    def _status_from_control(base: int, current: int, backend) -> int:
        current = int(current) & 0xFFFFFFFF
        cr1 = int(backend._read_stored_register_value(base + 0x00, 4) or 0) & 0xFFFFFFFF
        current |= STM32USARTProfile.V2_TXE | STM32USARTProfile.V2_TC
        if cr1 & 0x1:
            if cr1 & (1 << 3):
                current |= 1 << 21
            if cr1 & (1 << 2):
                current |= 1 << 22
        else:
            current &= ~((1 << 21) | (1 << 22))
        return current & 0xFFFFFFFF

    @staticmethod
    def _input_ready_enabled(backend) -> bool:
        checker = getattr(backend, "peripheral_input_ready_enabled", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    @staticmethod
    def _stream_key(base: int, layout: str) -> str:
        return f"uart:{layout}:0x{int(base) & 0xFFFFFFFF:08x}"

    @staticmethod
    def _legacy_layout_active(base: int, backend) -> bool:
        state = getattr(backend, "mmio_states", {}).get(int(base) & 0xFFFFFFFF)
        return not (state is not None and int(getattr(state, "write_count", 0) or 0) > 0)

    def _prepare_rx_byte(
        self,
        *,
        base: int,
        layout: str,
        status_addr: int,
        data_addr: int,
        pc: int,
        backend,
    ) -> Optional[int]:
        if not self._input_ready_enabled(backend):
            return None
        preparer = getattr(backend, "prepare_peripheral_input_byte", None)
        if not callable(preparer):
            return None
        byte = int(
            preparer(
                self._stream_key(base, layout),
                status_addr=status_addr,
                data_addr=data_addr,
                pc=pc,
                source=self.name,
            )
        ) & 0xFF
        backend._set_word_register_value(data_addr, byte)
        self.stats["rx_ready_injected"] += 1
        return byte

    def _consume_rx_byte(
        self,
        *,
        base: int,
        layout: str,
        status_addr: int,
        data_addr: int,
        pc: int,
        backend,
        supplied_byte: Optional[int] = None,
    ) -> int:
        consumer = getattr(backend, "consume_peripheral_input_byte", None)
        if callable(consumer):
            byte = int(
                consumer(
                    self._stream_key(base, layout),
                    status_addr=status_addr,
                    data_addr=data_addr,
                    pc=pc,
                    source=self.name,
                    supplied_byte=supplied_byte,
                )
            ) & 0xFF
        else:
            byte = (
                int(supplied_byte) & 0xFF
                if supplied_byte is not None
                else int(backend._read_stored_register_value(data_addr, 4) or 0) & 0xFF
            )
        backend._set_word_register_value(data_addr, byte)
        status = int(backend._read_stored_register_value(status_addr, 4) or 0) & 0xFFFFFFFF
        status &= ~self.LEGACY_RXNE
        backend._set_word_register_value(status_addr, status)
        self.stats["rx_bytes_returned"] += 1
        return byte

    def _apply_word(self, base: int, word_addr: int, backend, pc: int = 0) -> Optional[int]:
        offset = int(word_addr) - int(base)
        if offset == 0x00:
            if not self._legacy_layout_active(base, backend):
                return None
            current = backend._read_stored_register_value(word_addr, 4)
            modeled = (int(current or 0) | self.LEGACY_TXE | self.LEGACY_TC) & 0xFFFFFFFF
            if self._prepare_rx_byte(
                base=base,
                layout="legacy",
                status_addr=base,
                data_addr=base + 0x04,
                pc=pc,
                backend=backend,
            ) is not None:
                modeled |= self.LEGACY_RXNE
            backend._set_word_register_value(word_addr, modeled)
            self.stats["legacy_status_reads"] += 1
            return modeled
        if offset == 0x04:
            if not self._legacy_layout_active(base, backend):
                return None
            byte = self._consume_rx_byte(
                base=base,
                layout="legacy",
                status_addr=base,
                data_addr=base + 0x04,
                pc=pc,
                backend=backend,
            )
            self.stats["legacy_data_reads"] += 1
            return byte
        if offset == 0x1C:
            current = backend._read_stored_register_value(word_addr, 4)
            modeled = self._status_from_control(base, int(current or 0), backend)
            if self._prepare_rx_byte(
                base=base,
                layout="v2",
                status_addr=base + 0x1C,
                data_addr=base + 0x24,
                pc=pc,
                backend=backend,
            ) is not None:
                modeled |= self.V2_RXNE
            backend._set_word_register_value(word_addr, modeled)
            self.stats["v2_status_reads"] += 1
            return modeled
        if offset == 0x20:
            icr = int(backend._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
            isr_addr = base + 0x1C
            isr = int(backend._read_stored_register_value(isr_addr, 4) or 0) & 0xFFFFFFFF
            isr &= ~icr
            backend._set_word_register_value(isr_addr, self._status_from_control(base, isr, backend))
            return icr
        if offset == 0x24:
            byte = self._consume_rx_byte(
                base=base,
                layout="v2",
                status_addr=base + 0x1C,
                data_addr=base + 0x24,
                pc=pc,
                backend=backend,
            )
            self.stats["v2_data_reads"] += 1
            return byte
        if offset == 0x28:
            isr_addr = base + 0x1C
            isr = int(backend._read_stored_register_value(isr_addr, 4) or 0) & 0xFFFFFFFF
            isr |= self.V2_TXE | self.V2_TC
            backend._set_word_register_value(isr_addr, self._status_from_control(base, isr, backend))
            return backend._read_stored_register_value(word_addr, 4) or 0
        return None

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        base = self._base_for(address)
        if base is None:
            return None
        word_addr = self._word_addr(address)
        if self._apply_word(base, word_addr, backend, pc=pc) is None:
            return None
        return self._extract_register_value(backend, word_addr, address, size)

    def apply_external_read_side_effects(
        self,
        address: int,
        pc: int,
        size: int,
        value: int,
        backend,
    ) -> bool:
        """Keep UART RX state coherent when a scoped value wins over a model."""
        base = self._base_for(address)
        if base is None:
            return False
        word_addr = self._word_addr(address)
        offset = int(word_addr) - int(base)
        value = int(value) & _size_mask(size)
        if offset == 0x00 and self._legacy_layout_active(base, backend):
            if value & self.LEGACY_RXNE:
                self._prepare_rx_byte(
                    base=base,
                    layout="legacy",
                    status_addr=base,
                    data_addr=base + 0x04,
                    pc=pc,
                    backend=backend,
                )
            else:
                pending = getattr(backend, "_peripheral_input_pending", None)
                if isinstance(pending, dict):
                    pending.pop(self._stream_key(base, "legacy"), None)
            return True
        if offset == 0x1C:
            if value & self.V2_RXNE:
                self._prepare_rx_byte(
                    base=base,
                    layout="v2",
                    status_addr=base + 0x1C,
                    data_addr=base + 0x24,
                    pc=pc,
                    backend=backend,
                )
            else:
                pending = getattr(backend, "_peripheral_input_pending", None)
                if isinstance(pending, dict):
                    pending.pop(self._stream_key(base, "v2"), None)
            return True
        if offset == 0x04 and self._legacy_layout_active(base, backend):
            self._consume_rx_byte(
                base=base,
                layout="legacy",
                status_addr=base,
                data_addr=base + 0x04,
                pc=pc,
                backend=backend,
                supplied_byte=value & 0xFF,
            )
            return True
        if offset == 0x24:
            self._consume_rx_byte(
                base=base,
                layout="v2",
                status_addr=base + 0x1C,
                data_addr=base + 0x24,
                pc=pc,
                backend=backend,
                supplied_byte=value & 0xFF,
            )
            return True
        return False

    def apply_write(self, address: int, size: int, backend) -> bool:
        base = self._base_for(address)
        if base is None:
            return False
        word_addr = self._word_addr(address)
        changed = self._apply_word(base, word_addr, backend) is not None
        if word_addr == base + 0x00:
            changed = self._apply_word(base, base + 0x1C, backend) is not None or changed
        return changed

    def summary(self) -> Dict[str, object]:
        return dict(self.stats)


class STM32BxCANProfile(PeripheralSemanticProfile):
    """STM32 bxCAN mode-ack and empty-mailbox semantics."""

    name = "stm32_bxcan"

    def __init__(self):
        self.bases = {0x40006400, 0x40006800}

    def _base_for(self, address: int) -> Optional[int]:
        address = int(address) & 0xFFFFFFFF
        for base in self.bases:
            if base <= address < base + 0x400:
                return base
        return None

    @staticmethod
    def _status_from_control(base: int, current: int, backend) -> int:
        current = int(current) & 0xFFFFFFFF
        mcr = int(backend._read_stored_register_value(base + 0x00, 4) or 0) & 0xFFFFFFFF
        if mcr & 0x1:
            current |= 0x1
        else:
            current &= ~0x1
        if mcr & 0x2:
            current |= 0x2
        else:
            current &= ~0x2
        return current & 0xFFFFFFFF

    def _apply_word(self, base: int, word_addr: int, backend) -> Optional[int]:
        offset = int(word_addr) - int(base)
        if offset == 0x04:
            current = backend._read_stored_register_value(word_addr, 4)
            modeled = self._status_from_control(base, int(current or 0), backend)
            backend._set_word_register_value(word_addr, modeled)
            return modeled
        if offset == 0x0C:
            current = int(backend._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
            modeled = current | (1 << 26) | (1 << 27) | (1 << 28)
            backend._set_word_register_value(word_addr, modeled)
            return modeled
        if offset == 0x1B4:
            current = int(backend._read_stored_register_value(word_addr, 4) or 0) & 0xFFFFFFFF
            modeled = current & ~0x3
            backend._set_word_register_value(word_addr, modeled)
            return modeled
        return None

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        base = self._base_for(address)
        if base is None:
            return None
        word_addr = self._word_addr(address)
        if self._apply_word(base, word_addr, backend) is None:
            return None
        return self._extract_register_value(backend, word_addr, address, size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        base = self._base_for(address)
        if base is None:
            return False
        word_addr = self._word_addr(address)
        changed = self._apply_word(base, word_addr, backend) is not None
        if word_addr == base + 0x00:
            changed = self._apply_word(base, base + 0x04, backend) is not None or changed
        return changed


class TIM5SemanticProfile(PeripheralSemanticProfile):
    """STM32 TIM5 tick source as an **event-latched** device (IRQ50).

    This is deliberately *not* a free-running periodic driver: the model latches
    ``SR.CC1IF`` when ``CNT`` crosses ``CCR1`` with ``CR1.CEN`` and ``DIER.CC1IE``
    set, and the flag stays latched until the firmware clears it.  That is the
    contract the firmware actually uses:

    ``st_lld_serve_interrupt`` @ ``0x08136618``::

        8136618  ldr  r2, [pc, #36]     ; r2 = 0x40000C00 (TIM5)
        813661c  ldr  r1, [r2, #16]     ; r1 = SR
        813661e  ldr  r3, [r2, #12]     ; r3 = DIER
        8136620  ands r3, r1            ; r3 = SR & DIER
        8136622  uxtb r1, r3
        8136624  mvns r1, r1
        8136628  str  r1, [r2, #16]     ; SR = ~(SR & DIER)   -> rc_w0 clear
        8136626  lsls r3, r3, #30       ; test CC1IF
        813662a  bpl  ...

    so a write to ``SR`` behaves as ``SR &= written`` (write-0-to-clear), and
    ``~(sr & dier)`` has a 0 exactly at the bits that were both set and enabled.

    Virtual time is an explicit function of the executed instruction count.  The
    rate is an auditable knob (``LSGEMU_IRQ_US_PER_INSN``, default 0.05 us/insn)
    and the fractional remainder **lives in this profile's state** so that a
    snapshot/restore round trip cannot lose sub-tick time.
    """

    name = "tim5"

    BASE = 0x40000C00
    SIZE = 0x400
    CR1 = 0x00
    CR2 = 0x04
    SMCR = 0x08
    DIER = 0x0C
    SR = 0x10
    EGR = 0x14
    CCMR1 = 0x18
    CCMR2 = 0x1C
    CCER = 0x20
    CNT = 0x24
    PSC = 0x28
    ARR = 0x2C
    RCR = 0x30
    CCR1 = 0x34

    CR1_CEN = 1 << 0
    DIER_CC1IE = 1 << 1
    SR_CC1IF = 1 << 1
    EGR_UG = 1 << 0

    KNOWN_OFFSETS = (
        CR1, CR2, SMCR, DIER, SR, EGR, CCMR1, CCMR2, CCER, CNT, PSC, ARR, RCR, CCR1,
    )

    DEFAULT_US_PER_INSN = 0.05

    def __init__(self, us_per_insn: Optional[float] = None):
        if us_per_insn is None:
            try:
                us_per_insn = float(
                    os.environ.get("LSGEMU_IRQ_US_PER_INSN", str(self.DEFAULT_US_PER_INSN))
                )
            except (TypeError, ValueError):
                us_per_insn = self.DEFAULT_US_PER_INSN
        self.us_per_insn = float(us_per_insn)
        self.regs: Dict[int, int] = {}
        self.cnt = 0
        self.last_cnt = 0
        self.cc1if = False
        self.time_acc = 0.0
        self.delivered_pending = False
        self.delivered_instructions = 0
        # Audit counters (mirrored into summary()/evidence).
        self.stats: Dict[str, object] = {
            "advance_calls": 0,
            "advanced_us": 0,
            "latch_events": 0,
            "deliveries": 0,
            "deadline_skips": 0,
            "sr_reads": 0,
            "cnt_reads": 0,
            "dier_reads": 0,
            "writes": 0,
        }
        self.last_delivery: Optional[Dict[str, object]] = None

    # -- addressing ------------------------------------------------------
    def in_range(self, address: int) -> bool:
        addr = int(address) & 0xFFFFFFFF
        return self.BASE <= addr < self.BASE + self.SIZE

    def _offset(self, address: int) -> int:
        return (int(address) & 0xFFFFFFFF) - self.BASE

    def _word_offset(self, address: int) -> int:
        return self._offset(address) & ~0x3

    # -- timer semantics -------------------------------------------------
    def armed(self) -> bool:
        return bool(self.regs.get(self.CR1, 0) & self.CR1_CEN) and bool(
            self.regs.get(self.DIER, 0) & self.DIER_CC1IE
        )

    def deadline(self) -> int:
        return int(self.regs.get(self.CCR1, 0)) & 0xFFFFFFFF

    def _latch(self) -> None:
        """Set CC1IF only on a real *crossing* of the compare value."""
        if self.cc1if or not self.armed():
            return
        deadline = self.deadline()
        if deadline == 0:
            return
        crossed = self.last_cnt < deadline <= self.cnt
        if self.cnt < self.last_cnt:  # counter wrapped
            crossed = crossed or deadline > self.last_cnt or deadline <= self.cnt
        if crossed:
            self.cc1if = True
            self.stats["latch_events"] = int(self.stats["latch_events"]) + 1

    def advance_by_instructions(self, instructions: int) -> None:
        """Advance virtual time; the fractional remainder stays in profile state."""
        instructions = int(instructions)
        if instructions <= 0:
            return
        self.stats["advance_calls"] = int(self.stats["advance_calls"]) + 1
        self.time_acc += instructions * self.us_per_insn
        step = int(self.time_acc)
        if step <= 0:
            return
        self.time_acc -= step
        self.stats["advanced_us"] = int(self.stats["advanced_us"]) + step
        self.last_cnt = int(self.cnt) & 0xFFFFFFFF
        self.cnt = (self.cnt + step) & 0xFFFFFFFF
        self._latch()

    def fast_forward_to_deadline(self) -> bool:
        """Skip virtual time to the next compare event (used while idling)."""
        if not self.armed():
            return False
        deadline = self.deadline()
        if deadline == 0 or self.cc1if:
            return False
        if self.cnt < deadline:
            delta = deadline - self.cnt
        elif self.cnt == deadline:
            return False
        else:
            delta = (0x100000000 - self.cnt) + deadline
        self.last_cnt = int(self.cnt) & 0xFFFFFFFF
        self.cnt = (self.cnt + delta) & 0xFFFFFFFF
        self.time_acc = 0.0
        self.stats["deadline_skips"] = int(self.stats["deadline_skips"]) + 1
        self._latch()
        return True

    def due(self) -> bool:
        return bool(self.cc1if) and self.armed() and not self.delivered_pending

    def note_delivered(self, *, pc: int = 0, instructions: int = 0) -> None:
        """Hardware does not clear CC1IF on delivery; wait for the firmware ack."""
        self.delivered_pending = True
        self.delivered_instructions = int(instructions)
        self.stats["deliveries"] = int(self.stats["deliveries"]) + 1
        self.last_delivery = {
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
            "cnt": f"0x{int(self.cnt) & 0xFFFFFFFF:08x}",
            "ccr1": f"0x{self.deadline():08x}",
            "dier": f"0x{int(self.regs.get(self.DIER, 0)) & 0xFFFFFFFF:08x}",
            "cr1": f"0x{int(self.regs.get(self.CR1, 0)) & 0xFFFFFFFF:08x}",
        }

    # -- register file ---------------------------------------------------
    def _word_value(self, offset: int) -> int:
        if offset == self.CNT:
            return int(self.cnt) & 0xFFFFFFFF
        if offset == self.SR:
            base = int(self.regs.get(self.SR, 0)) & 0xFFFFFFFF
            base &= ~self.SR_CC1IF
            if self.cc1if:
                base |= self.SR_CC1IF
            return base
        return int(self.regs.get(offset, 0)) & 0xFFFFFFFF

    def owns_read(self, address: int) -> bool:
        return self.in_range(address) and self._word_offset(address) in self.KNOWN_OFFSETS

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        if not self.in_range(address):
            return None
        offset = self._word_offset(address)
        if offset not in self.KNOWN_OFFSETS:
            return None
        if offset == self.SR:
            self.stats["sr_reads"] = int(self.stats["sr_reads"]) + 1
        elif offset == self.CNT:
            self.stats["cnt_reads"] = int(self.stats["cnt_reads"]) + 1
        elif offset == self.DIER:
            self.stats["dier_reads"] = int(self.stats["dier_reads"]) + 1
        word_addr = int(address) & ~0x3
        word_value = self._word_value(offset)
        if backend is not None:
            backend._set_word_register_value(word_addr, word_value)
        shift = (int(address) - word_addr) * 8
        return (word_value >> shift) & _size_mask(size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        if not self.in_range(address):
            return False
        offset = self._word_offset(address)
        if offset not in self.KNOWN_OFFSETS:
            return False
        value = int(backend._read_stored_register_value(offset + self.BASE, 4) or 0) & 0xFFFFFFFF
        self.stats["writes"] = int(self.stats["writes"]) + 1
        if offset == self.SR:
            # rc_w0: only bits written as 0 are cleared.
            self.regs[self.SR] = int(self.regs.get(self.SR, 0)) & value
            if not (value & self.SR_CC1IF):
                self.cc1if = False
                self.delivered_pending = False
        elif offset == self.CNT:
            self.last_cnt = int(self.cnt) & 0xFFFFFFFF
            self.cnt = value & 0xFFFFFFFF
        elif offset == self.EGR:
            if value & self.EGR_UG:
                self.last_cnt = int(self.cnt) & 0xFFFFFFFF
                self.cnt = 0
                self.cc1if = False
                self.delivered_pending = False
                self.regs[self.SR] = 0
        else:
            self.regs[offset] = value
        return True

    # -- snapshot ---------------------------------------------------------
    def snapshot_state(self) -> Dict[str, object]:
        return {
            "regs": dict(self.regs),
            "cnt": int(self.cnt),
            "last_cnt": int(self.last_cnt),
            "cc1if": bool(self.cc1if),
            "time_acc": float(self.time_acc),
            "delivered_pending": bool(self.delivered_pending),
            "delivered_instructions": int(self.delivered_instructions),
            "stats": dict(self.stats),
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self.regs = {
            int(offset): int(value) & 0xFFFFFFFF
            for offset, value in dict(state.get("regs", {}) or {}).items()
        }
        self.cnt = int(state.get("cnt", 0) or 0) & 0xFFFFFFFF
        self.last_cnt = int(state.get("last_cnt", 0) or 0) & 0xFFFFFFFF
        self.cc1if = bool(state.get("cc1if", False))
        self.time_acc = float(state.get("time_acc", 0.0) or 0.0)
        self.delivered_pending = bool(state.get("delivered_pending", False))
        self.delivered_instructions = int(state.get("delivered_instructions", 0) or 0)
        restored_stats = state.get("stats")
        if isinstance(restored_stats, dict):
            self.stats = {str(key): value for key, value in restored_stats.items()}

    def summary(self) -> Dict[str, object]:
        payload = dict(self.stats)
        payload.update(
            {
                "us_per_insn": float(self.us_per_insn),
                "cnt": f"0x{int(self.cnt) & 0xFFFFFFFF:08x}",
                "ccr1": f"0x{self.deadline():08x}",
                "cr1": f"0x{int(self.regs.get(self.CR1, 0)) & 0xFFFFFFFF:08x}",
                "dier": f"0x{int(self.regs.get(self.DIER, 0)) & 0xFFFFFFFF:08x}",
                "armed": self.armed(),
                "cc1if": bool(self.cc1if),
                "time_acc": float(self.time_acc),
                "last_delivery": self.last_delivery,
            }
        )
        return payload


class CortexMSCSInterruptProfile(PeripheralSemanticProfile):
    """Minimal Cortex-M SCS model: only ``ICSR`` (0xE000ED04) is claimed.

    ``ICSR.RETTOBASE`` (bit11) is ``SCB_ICSR_RETTOBASE_Msk`` — a *defined,
    read-only, hardware-maintained* status bit meaning "no preempted active
    exceptions" ("there are no active exceptions, or the currently-executing
    exception is the only active exception").  It is therefore **derived** from
    the modelled NVIC active-exception stack, never written as a constant:

        active_count <= 1  ->  RETTOBASE == 1

    ``__port_irq_epilogue`` (0x0812EDA8 ``ands r3, r3, #0x800``) gates the
    preemption branch on this bit, so a wrong value changes which port function
    the exception return lands in.
    """

    name = "cortexm_scs_irq"

    BASE = 0xE000E000
    SIZE = 0x1000
    ICSR = 0xE000ED04
    ICSR_RETTOBASE = 1 << 11

    def __init__(self):
        self.regs: Dict[int, int] = {}
        self.active_exceptions: List[int] = []
        self.stats: Dict[str, object] = {
            "icsr_reads": 0,
            "icsr_writes": 0,
            "rettobase_true": 0,
            "rettobase_false": 0,
        }
        self.last_derivation: Optional[Dict[str, object]] = None

    def in_range(self, address: int) -> bool:
        addr = int(address) & 0xFFFFFFFF
        return self.BASE <= addr < self.BASE + self.SIZE

    def push_active(self, irq: int) -> None:
        self.active_exceptions.append(int(irq))

    def pop_active(self, irq: Optional[int] = None) -> Optional[int]:
        if not self.active_exceptions:
            return None
        if irq is None or self.active_exceptions[-1] == int(irq):
            return self.active_exceptions.pop()
        if int(irq) in self.active_exceptions:
            index = len(self.active_exceptions) - 1 - self.active_exceptions[::-1].index(int(irq))
            return self.active_exceptions.pop(index)
        return None

    def rettobase(self) -> bool:
        return len(self.active_exceptions) <= 1

    def owns_read(self, address: int) -> bool:
        return (int(address) & 0xFFFFFFFF) & ~0x3 == self.ICSR

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        addr = int(address) & 0xFFFFFFFF
        if addr & ~0x3 != self.ICSR:
            return None
        self.stats["icsr_reads"] = int(self.stats["icsr_reads"]) + 1
        rettobase = self.rettobase()
        word = int(self.regs.get(self.ICSR, 0)) & 0xFFFFFFFF
        if rettobase:
            word |= self.ICSR_RETTOBASE
            self.stats["rettobase_true"] = int(self.stats["rettobase_true"]) + 1
        else:
            word &= ~self.ICSR_RETTOBASE
            self.stats["rettobase_false"] = int(self.stats["rettobase_false"]) + 1
        self.last_derivation = {
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
            "active_exceptions": list(self.active_exceptions),
            "active_count": len(self.active_exceptions),
            "rettobase": rettobase,
            "value": f"0x{word:08x}",
        }
        if backend is not None:
            backend._set_word_register_value(self.ICSR, word)
        shift = (int(address) - self.ICSR) * 8
        return (word >> shift) & _size_mask(size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        if int(address) & ~0x3 != self.ICSR:
            return False
        value = int(backend._read_stored_register_value(self.ICSR, 4) or 0) & 0xFFFFFFFF
        self.stats["icsr_writes"] = int(self.stats["icsr_writes"]) + 1
        # Only the software-writable bits are kept; RETTOBASE is derived on read.
        self.regs[self.ICSR] = value & ~self.ICSR_RETTOBASE
        return True

    def snapshot_state(self) -> Dict[str, object]:
        return {
            "regs": dict(self.regs),
            "active_exceptions": list(self.active_exceptions),
            "stats": dict(self.stats),
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self.regs = {
            int(addr): int(value) & 0xFFFFFFFF
            for addr, value in dict(state.get("regs", {}) or {}).items()
        }
        self.active_exceptions = [int(irq) for irq in list(state.get("active_exceptions", []) or [])]
        restored_stats = state.get("stats")
        if isinstance(restored_stats, dict):
            self.stats = {str(key): value for key, value in restored_stats.items()}

    def summary(self) -> Dict[str, object]:
        payload = dict(self.stats)
        payload.update(
            {
                "active_exceptions": list(self.active_exceptions),
                "rettobase_now": self.rettobase(),
                "last_derivation": self.last_derivation,
            }
        )
        return payload


class CortexMDWTProfile(PeripheralSemanticProfile):
    """DWT cycle counter (``DWT_CYCCNT`` @ 0xE0001004) as a free-running clock.

    ``chSysPolledDelayX`` @ 0x08133CDC is ChibiOS's polled delay::

        8133cdc  ldr  r2, [pc, #12]   ; r2 = 0xE0001000 (DWT)
        8133cde  ldr  r1, [r2, #4]    ; start = CYCCNT
        8133ce0  ldr  r3, [r2, #4]    ; CYCCNT
        8133ce2  subs r3, r3, r1
        8133ce4  cmp  r0, r3
        8133ce6  bhi  0x8133ce0      ; while (cycles > CYCCNT - start)

    With a constant read-back the loop can never exit, which stalls
    ``usb_lld_start`` between the CSRST write (0x081303D8) and the completion
    wait (0x081303E0).  A cycle counter *is* a function of executed
    instructions, so modelling it here is an external clock input, not a skip.
    """

    name = "cortexm_dwt"

    BASE = 0xE0001000
    SIZE = 0x10
    DWT_CTRL = 0xE0001000
    DWT_CYCCNT = 0xE0001004
    CTRL_CYCCNTENA = 1 << 0

    DEFAULT_CYCLES_PER_INSN = 1.0

    def __init__(self, cycles_per_insn: Optional[float] = None):
        if cycles_per_insn is None:
            try:
                cycles_per_insn = float(
                    os.environ.get(
                        "LSGEMU_DWT_CYCLES_PER_INSN", str(self.DEFAULT_CYCLES_PER_INSN)
                    )
                )
            except (TypeError, ValueError):
                cycles_per_insn = self.DEFAULT_CYCLES_PER_INSN
        self.cycles_per_insn = float(cycles_per_insn)
        self.cycles = 0
        self.cycle_acc = 0.0
        self.regs: Dict[int, int] = {}
        self.stats: Dict[str, object] = {
            "cyccnt_reads": 0,
            "advanced_cycles": 0,
        }
        self.last_read: Optional[Dict[str, object]] = None

    def in_range(self, address: int) -> bool:
        addr = int(address) & 0xFFFFFFFF
        return self.BASE <= addr < self.BASE + self.SIZE

    def owns_read(self, address: int) -> bool:
        addr = int(address) & ~0x3
        return addr in (self.DWT_CTRL, self.DWT_CYCCNT)

    def advance_by_instructions(self, instructions: int) -> None:
        instructions = int(instructions)
        if instructions <= 0:
            return
        self.cycle_acc += instructions * self.cycles_per_insn
        step = int(self.cycle_acc)
        if step <= 0:
            return
        self.cycle_acc -= step
        self.cycles = (self.cycles + step) & 0xFFFFFFFF
        self.stats["advanced_cycles"] = int(self.stats["advanced_cycles"]) + step

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        addr = int(address) & ~0x3
        if not self.in_range(addr):
            return None
        if addr == self.DWT_CYCCNT:
            self.stats["cyccnt_reads"] = int(self.stats["cyccnt_reads"]) + 1
            word = int(self.cycles) & 0xFFFFFFFF
            self.last_read = {
                "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
                "cyccnt": f"0x{word:08x}",
            }
        elif addr == self.DWT_CTRL:
            # The counter is modelled as armed: the firmware's polled delay
            # depends on CYCCNT advancing (it would hang on real hardware too).
            word = (int(self.regs.get(self.DWT_CTRL, 0)) | self.CTRL_CYCCNTENA) & 0xFFFFFFFF
        else:
            return None
        if backend is not None:
            backend._set_word_register_value(addr, word)
        shift = (int(address) - addr) * 8
        return (word >> shift) & _size_mask(size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        addr = int(address) & ~0x3
        if addr not in (self.DWT_CTRL, self.DWT_CYCCNT):
            return False
        value = int(backend._read_stored_register_value(addr, 4) or 0) & 0xFFFFFFFF
        if addr == self.DWT_CYCCNT:
            self.cycles = value
            self.cycle_acc = 0.0
        else:
            self.regs[self.DWT_CTRL] = value
        return True

    def snapshot_state(self) -> Dict[str, object]:
        return {
            "cycles": int(self.cycles),
            "cycle_acc": float(self.cycle_acc),
            "regs": dict(self.regs),
            "stats": dict(self.stats),
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self.cycles = int(state.get("cycles", 0) or 0) & 0xFFFFFFFF
        self.cycle_acc = float(state.get("cycle_acc", 0.0) or 0.0)
        self.regs = {
            int(addr): int(value) & 0xFFFFFFFF
            for addr, value in dict(state.get("regs", {}) or {}).items()
        }
        restored_stats = state.get("stats")
        if isinstance(restored_stats, dict):
            self.stats = {str(key): value for key, value in restored_stats.items()}

    def summary(self) -> Dict[str, object]:
        payload = dict(self.stats)
        payload.update(
            {
                "cycles_per_insn": float(self.cycles_per_insn),
                "cyccnt": f"0x{int(self.cycles) & 0xFFFFFFFF:08x}",
                "last_read": self.last_read,
            }
        )
        return payload


class OTGFSResetProfile(PeripheralSemanticProfile):
    """OTG_FS core soft-reset handshake (``GRSTCTL`` @ 0x50000010).

    ``AHBIDLE`` (bit31) reads 1 when no AHB transfer is outstanding, so the core
    is idle and can be reset.  ``usb_lld_start`` performs::

        81303d0  ldr r3, [r6, #16]      ; GRSTCTL
        81303d2  cmp r3, #0
        81303d4  bge 0x81303d0          ; wait until AHBIDLE (bit31) reads 1
        81303d8  str r3(=1), [r6, #16]  ; CSRST = 1
        81303da  movs r0, #12
        81303dc  bl  chSysPolledDelayX  ; 12-cycle polled delay
        81303e0  ldr r3, [r6, #16]      ; lsls #31 puts bit0 into N, so this
        81303e4  bmi 0x81303e0          ; loop waits until CSRST self-clears
        81303ec  ldr r3, [r6, #16]
        81303ee  cmp r3, #0
        81303f0  bge 0x81303ec          ; AHBIDLE is 1 again

    So the model is: AHBIDLE is 1 at rest; a ``CSRST=1`` write starts the reset;
    the first ``GRSTCTL`` read afterwards reports ``CSRST=1`` / ``AHBIDLE=0``
    (reset in flight); the reset then completes, so later reads report
    ``CSRST=0`` / ``AHBIDLE=1``.  This is an external readiness input, not a
    PC/register override.
    """

    name = "otgfs_reset"

    BASE = 0x50000000
    SIZE = 0x40000
    GRSTCTL = 0x10
    AHBIDLE = 1 << 31
    CSRST = 1 << 0

    def __init__(self):
        self.regs: Dict[int, int] = {}
        self.reset_pending = False
        self.stats: Dict[str, object] = {
            "grstctl_reads": 0,
            "csrst_writes": 0,
            "ahbidle_low_reads": 0,
            "resets_completed": 0,
        }
        self.last_read: Optional[Dict[str, object]] = None

    def in_range(self, address: int) -> bool:
        addr = int(address) & 0xFFFFFFFF
        return self.BASE <= addr < self.BASE + self.SIZE

    def _grstctl_value(self) -> int:
        value = int(self.regs.get(self.GRSTCTL, 0)) & ~(self.AHBIDLE | self.CSRST)
        if self.reset_pending:
            # Reset in flight: CSRST reads back 1, AHBIDLE is low.
            value |= self.CSRST
        else:
            value |= self.AHBIDLE
        return value & 0xFFFFFFFF

    def owns_read(self, address: int) -> bool:
        return int(address) & ~0x3 == self.BASE + self.GRSTCTL

    def model_read(self, address: int, pc: int, size: int, backend) -> Optional[int]:
        if int(address) & ~0x3 != self.BASE + self.GRSTCTL:
            return None
        self.stats["grstctl_reads"] = int(self.stats["grstctl_reads"]) + 1
        # A read while the reset is in flight observes CSRST=1/AHBIDLE=0; the
        # reset then completes, so subsequent reads see CSRST=0/AHBIDLE=1.
        in_reset = bool(self.reset_pending)
        word = self._grstctl_value()
        if in_reset:
            self.stats["ahbidle_low_reads"] = int(self.stats["ahbidle_low_reads"]) + 1
            self.stats["resets_completed"] = int(self.stats["resets_completed"]) + 1
            self.reset_pending = False
        self.last_read = {
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
            "value": f"0x{word:08x}",
            "ahbidle": bool(word & self.AHBIDLE),
            "csrst": bool(word & self.CSRST),
        }
        if backend is not None:
            backend._set_word_register_value(self.BASE + self.GRSTCTL, word)
        return (word >> ((int(address) - (self.BASE + self.GRSTCTL)) * 8)) & _size_mask(size)

    def apply_write(self, address: int, size: int, backend) -> bool:
        if int(address) & ~0x3 != self.BASE + self.GRSTCTL:
            return False
        value = int(backend._read_stored_register_value(self.BASE + self.GRSTCTL, 4) or 0) & 0xFFFFFFFF
        self.regs[self.GRSTCTL] = value
        if value & self.CSRST:
            self.reset_pending = True
            self.stats["csrst_writes"] = int(self.stats["csrst_writes"]) + 1
        return True

    def snapshot_state(self) -> Dict[str, object]:
        return {
            "regs": dict(self.regs),
            "reset_pending": bool(self.reset_pending),
            "stats": dict(self.stats),
        }

    def restore_state(self, state: Optional[Dict[str, object]]) -> None:
        if not isinstance(state, dict):
            return
        self.regs = {
            int(offset): int(value) & 0xFFFFFFFF
            for offset, value in dict(state.get("regs", {}) or {}).items()
        }
        self.reset_pending = bool(state.get("reset_pending", False))
        restored_stats = state.get("stats")
        if isinstance(restored_stats, dict):
            self.stats = {str(key): value for key, value in restored_stats.items()}

    def summary(self) -> Dict[str, object]:
        payload = dict(self.stats)
        payload.update(
            {
                "grstctl": f"0x{self._grstctl_value():08x}",
                "reset_pending": bool(self.reset_pending),
                "last_read": self.last_read,
            }
        )
        return payload


class SemanticProfileRegistry:
    """Access-sequence based semantic model registry."""

    def __init__(
        self,
        svd_index: Optional[SVDRegisterIndex] = None,
        profiles: Optional[Sequence[PeripheralSemanticProfile]] = None,
        causal_context: Optional[CausalExecutionContext] = None,
    ):
        self.svd_index = svd_index or SVDRegisterIndex.from_environment()
        self.profiles: List[PeripheralSemanticProfile] = list(profiles) if profiles is not None else self._default_profiles()
        self.causal_context = causal_context
        self.events: Deque[MMIOAccessEvent] = deque(maxlen=self._env_int("LSGEMU_MMIO_SEMANTIC_EVENT_LIMIT", 512))
        self.transactions: Dict[str, PeripheralTransactionState] = {}
        self.rules: List[SemanticMMIORule] = []
        self.learned_rule_keys: Set[Tuple[int, Optional[int], Optional[int], int]] = set()
        self.stats: Dict[str, int] = {
            "observed_reads": 0,
            "observed_writes": 0,
            "rules_learned": 0,
            "rules_promoted": 0,
            "rule_hits": 0,
            "profile_hits": 0,
            "profile_write_updates": 0,
            "transaction_rules_learned": 0,
            "transaction_rule_promoted": 0,
            "svd_registers": len(self.svd_index.registers),
            "registered_profiles": len(self.profiles),
            "svd_read_clear_hits": 0,
            "svd_write_side_effect_hits": 0,
            "transaction_conflicting_writes": 0,
            "dma_start_observations": 0,
            "dma_completion_observations": 0,
        }
        self.promote_evidence = self._env_int("LSGEMU_GENERIC_MMIO_PROMOTE_EVIDENCE", 2)
        self.enabled = os.environ.get("LSGEMU_ENABLE_GENERIC_MMIO_SEMANTICS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _default_profiles() -> List[PeripheralSemanticProfile]:
        if os.environ.get("LSGEMU_DISABLE_BUILTIN_PERIPHERAL_PROFILES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return []
        profiles: List[PeripheralSemanticProfile] = [
            CortexMSysTickProfile(),
            STM32RCCProfile(),
            STM32USARTProfile(),
            STM32BxCANProfile(),
        ]
        # r33 P1：PWR 只在时钟/电源就绪语义开启时注册（关语义臂必须等于
        # 本轮之前的 profile 集合，否则 A/B 不干净）。
        if clock_ready_semantics_enabled():
            profiles.append(STM32PWRProfile())
        return profiles

    def register_profile(self, profile: PeripheralSemanticProfile) -> PeripheralSemanticProfile:
        """Install a profile at read-path priority 3 (idempotent by ``name``)."""
        name = str(getattr(profile, "name", "") or "")
        for existing in self.profiles:
            if name and str(getattr(existing, "name", "") or "") == name:
                return existing
        self.profiles.append(profile)
        self.stats["registered_profiles"] = len(self.profiles)
        return profile

    def profile_by_name(self, name: str) -> Optional[PeripheralSemanticProfile]:
        wanted = str(name or "")
        for profile in self.profiles:
            if str(getattr(profile, "name", "") or "") == wanted:
                return profile
        return None

    def _snapshot_profile_states(self) -> Dict[str, object]:
        """Capture opt-in profile state **by value** at capture time.

        ``snapshot_runtime_state(copy_values=False)`` hands out live references
        for the generic container fields; the timer/NVIC state must not alias, or
        "capture, mutate, restore" would silently restore the mutated state.
        """
        states: Dict[str, object] = {}
        for profile in self.profiles:
            grab = getattr(profile, "snapshot_state", None)
            if not callable(grab):
                continue
            try:
                states[str(getattr(profile, "name", profile.__class__.__name__))] = grab()
            except Exception:
                continue
        return states

    def _restore_profile_states(self, states: Optional[Dict[str, object]]) -> None:
        if not isinstance(states, dict):
            return
        for profile in self.profiles:
            name = str(getattr(profile, "name", profile.__class__.__name__))
            if name not in states:
                continue
            applier = getattr(profile, "restore_state", None)
            if not callable(applier):
                continue
            try:
                applier(copy.deepcopy(states[name]))
            except Exception:
                continue

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(default)), 0))
        except ValueError:
            return default

    def _transaction_for(self, address: int) -> PeripheralTransactionState:
        key = self.svd_index.peripheral_key_for(address)
        transaction = self.transactions.get(key)
        if transaction is None:
            transaction = PeripheralTransactionState(peripheral_key=key)
            self.transactions[key] = transaction
        return transaction

    def _mark_context_phase(
        self,
        transaction: PeripheralTransactionState,
        event: MMIOAccessEvent,
        phase: str,
    ) -> None:
        context = self.causal_context
        if context is None:
            return
        context.mark_peripheral_phase(
            transaction.peripheral_key,
            phase,
            pc=event.pc,
            address=event.address,
            source="svd_or_observed_transaction",
        )

    @staticmethod
    def _dma_channel_key(info: SVDRegisterInfo) -> str:
        register_name = str(info.register_name or "REG")
        channel_match = re.search(r"(?:CH|CHANNEL|C)(\d+)", register_name, re.IGNORECASE)
        channel = channel_match.group(1) if channel_match else "shared"
        return f"svd:{info.peripheral_name}:channel:{channel}"

    @staticmethod
    def _field_tokens(field_info: SVDFieldInfo) -> Set[str]:
        text = f"{field_info.name} {field_info.description}".lower()
        return set(re.findall(r"[a-z0-9]+", text))

    @staticmethod
    def _event_register_value(event: MMIOAccessEvent) -> int:
        shift = (int(event.address) & 0x3) * 8
        return (
            (int(event.value) & _size_mask(event.size)) << shift
        ) & 0xFFFFFFFF

    def _observe_svd_dma_write(
        self,
        info: Optional[SVDRegisterInfo],
        event: MMIOAccessEvent,
    ) -> None:
        if info is None or self.causal_context is None:
            return
        identity = f"{info.peripheral_name} {info.register_name}".lower()
        if "dma" not in identity:
            return
        start_masks = 0
        for field_info in info.fields:
            tokens = self._field_tokens(field_info)
            if tokens & {"en", "enable", "start", "swstart"}:
                start_masks |= field_info.mask
        if not start_masks:
            return
        channel = self._dma_channel_key(info)
        transaction = self._transaction_for(event.address)
        if self._event_register_value(event) & int(start_masks):
            transaction.phase = "dma_active"
            self.stats["dma_start_observations"] += 1
            self.causal_context.record_dma_start(
                channel,
                pc=event.pc,
                evidence="svd_enable_field_write",
            )
        else:
            transaction.phase = "dma_idle"
            self._mark_context_phase(transaction, event, "dma_idle")

    def _observe_svd_dma_read(
        self,
        info: Optional[SVDRegisterInfo],
        event: MMIOAccessEvent,
    ) -> None:
        if info is None or self.causal_context is None:
            return
        identity = f"{info.peripheral_name} {info.register_name}".lower()
        if "dma" not in identity:
            return
        completion_mask = 0
        for field_info in info.fields:
            tokens = self._field_tokens(field_info)
            compact_name = str(field_info.name or "").lower().replace("_", "")
            if (
                tokens & {"done", "complete", "completed", "tcif"}
                or compact_name in {"tc", "tcif", "transfercomplete"}
            ):
                completion_mask |= field_info.mask
        if completion_mask and self._event_register_value(event) & int(completion_mask):
            transaction = self._transaction_for(event.address)
            transaction.phase = "dma_complete"
            transaction.completion_count += 1
            self.stats["dma_completion_observations"] += 1
            self.causal_context.record_dma_complete(
                self._dma_channel_key(info),
                pc=event.pc,
                evidence="svd_completion_field_read",
            )

    def _apply_svd_read_semantics(
        self,
        address: int,
        size: int,
        backend,
        returned_value: int,
    ) -> int:
        if backend is None:
            return int(returned_value) & _size_mask(size)
        word_address = int(address) & ~0x3
        info = self.svd_index.register_for(word_address)
        if info is None:
            return int(returned_value) & _size_mask(size)
        clear_mask = int(info.read_clear_mask()) & 0xFFFFFFFF
        if not clear_mask:
            return int(returned_value) & _size_mask(size)
        current_word = int(
            backend._read_stored_register_value(word_address, 4) or 0
        ) & 0xFFFFFFFF
        backend._set_word_register_value(word_address, current_word & ~clear_mask)
        self.stats["svd_read_clear_hits"] += 1
        return int(returned_value) & _size_mask(size)

    def finalize_read_side_effects(self, address: int, size: int, backend=None) -> None:
        """Re-apply documented read effects after the handler records the read."""
        if backend is None:
            return
        word_address = int(address) & ~0x3
        info = self.svd_index.register_for(word_address)
        if info is None:
            return
        clear_mask = int(info.read_clear_mask()) & 0xFFFFFFFF
        if not clear_mask:
            return
        current_word = int(
            backend._read_stored_register_value(word_address, 4) or 0
        ) & 0xFFFFFFFF
        backend._set_word_register_value(word_address, current_word & ~clear_mask)

    def _apply_svd_write_semantics(
        self,
        address: int,
        size: int,
        backend,
        *,
        previous_value: Optional[int],
        written_value: Optional[int],
    ) -> bool:
        word_address = int(address) & ~0x3
        info = self.svd_index.register_for(word_address)
        if info is None:
            return False
        action_masks = info.write_action_masks()
        if not action_masks:
            return False
        current_word = int(
            backend._read_stored_register_value(word_address, 4) or 0
        ) & 0xFFFFFFFF
        old_word = (
            current_word
            if previous_value is None
            else int(previous_value) & 0xFFFFFFFF
        )
        write_word = current_word
        if written_value is not None:
            write_size = max(1, min(4, int(size or 4)))
            shift = (int(address) - word_address) * 8
            write_mask = _size_mask(write_size) << shift
            write_word = (
                (current_word & ~write_mask)
                | ((int(written_value) << shift) & write_mask)
            ) & 0xFFFFFFFF
        else:
            write_mask = (_size_mask(size) << ((int(address) - word_address) * 8)) & 0xFFFFFFFF

        result = current_word
        handled_mask = 0
        for raw_action, raw_mask in action_masks.items():
            action = str(raw_action or "").replace("_", "").replace("-", "").lower()
            mask = int(raw_mask) & int(write_mask) & 0xFFFFFFFF
            if not mask:
                continue
            if action == "onetoclear":
                field_value = (old_word & mask) & ~(write_word & mask)
            elif action == "onetoset":
                field_value = (old_word & mask) | (write_word & mask)
            elif action == "onetotoggle":
                field_value = (old_word & mask) ^ (write_word & mask)
            elif action == "zerotoclear":
                field_value = (old_word & mask) & (write_word & mask)
            elif action == "zerotoset":
                field_value = (old_word & mask) | ((~write_word) & mask)
            elif action == "zerototoggle":
                field_value = (old_word & mask) ^ ((~write_word) & mask)
            else:
                continue
            result = (result & ~mask) | (field_value & mask)
            handled_mask |= mask
        if not handled_mask:
            return False
        backend._set_word_register_value(word_address, result & 0xFFFFFFFF)
        self.stats["svd_write_side_effect_hits"] += 1
        return True

    def observe_read(self, pc: int, address: int, value: int, size: int, timestamp: int) -> None:
        if not self.enabled:
            return
        self.stats["observed_reads"] += 1
        event = MMIOAccessEvent(
            pc=int(pc) & 0xFFFFFFFF,
            address=int(address) & 0xFFFFFFFF,
            is_read=True,
            value=int(value) & 0xFFFFFFFF,
            size=max(1, min(4, int(size or 4))),
            timestamp=int(timestamp),
        )
        transaction = self._transaction_for(event.address)
        event.transaction_sequence = int(transaction.sequence)
        transaction.last_read = event
        info = self.svd_index.register_for(event.address & ~0x3)
        if info is not None and info.is_status_like():
            transaction.phase = "status_observed"
            self._mark_context_phase(transaction, event, "status_observed")
        self._observe_svd_dma_read(info, event)
        self.events.append(event)

    def observe_write(self, pc: int, address: int, value: int, size: int, timestamp: int) -> None:
        if not self.enabled:
            return
        self.stats["observed_writes"] += 1
        event = MMIOAccessEvent(
            pc=int(pc) & 0xFFFFFFFF,
            address=int(address) & 0xFFFFFFFF,
            is_read=False,
            value=int(value) & 0xFFFFFFFF,
            size=max(1, min(4, int(size or 4))),
            timestamp=int(timestamp),
        )
        transaction = self._transaction_for(event.address)
        previous = transaction.last_write_by_address.get(event.address)
        if previous is not None and int(previous.value) != int(event.value):
            transaction.conflicting_write_count += 1
            self.stats["transaction_conflicting_writes"] += 1
        transaction.sequence += 1
        event.transaction_sequence = int(transaction.sequence)
        transaction.last_write_by_address[event.address] = event
        transaction.phase = "configured"
        self._mark_context_phase(transaction, event, "configured")
        info = self.svd_index.register_for(event.address & ~0x3)
        self._observe_svd_dma_write(info, event)
        self.events.append(event)

    def model_read(
        self,
        address: int,
        pc: int,
        size: int,
        current_value: Optional[int] = None,
        backend=None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        addr = int(address) & 0xFFFFFFFF
        current = int(current_value or 0) & 0xFFFFFFFF
        for rule in reversed(self.rules):
            if not rule.applies_to(addr, pc, registry=self):
                continue
            value = rule.modeled_value(current)
            self.stats["rule_hits"] += 1
            transaction = self._transaction_for(addr)
            transaction.phase = "validated_completion"
            transaction.completion_count += 1
            event = MMIOAccessEvent(
                pc=int(pc) & 0xFFFFFFFF,
                address=addr,
                is_read=True,
                value=int(value) & 0xFFFFFFFF,
                size=max(1, min(4, int(size or 4))),
                timestamp=max(
                    (int(item.timestamp) for item in self.events),
                    default=0,
                ),
                transaction_sequence=int(transaction.sequence),
            )
            self._mark_context_phase(transaction, event, "validated_completion")
            return self._apply_svd_read_semantics(
                addr,
                size,
                backend,
                self._extract_value(value, addr, size),
            )
        if backend is not None:
            for profile in self.profiles:
                value = profile.model_read(addr, pc, size, backend)
                if value is not None:
                    self.stats["profile_hits"] += 1
                    return self._apply_svd_read_semantics(
                        addr,
                        size,
                        backend,
                        int(value) & _size_mask(size),
                    )
            info = self.svd_index.register_for(addr & ~0x3)
            if info is not None and info.read_clear_mask():
                current_word = int(
                    backend._read_stored_register_value(addr & ~0x3, 4) or 0
                ) & 0xFFFFFFFF
                return self._apply_svd_read_semantics(
                    addr,
                    size,
                    backend,
                    self._extract_value(current_word, addr, size),
                )
        return None

    def apply_external_read_side_effects(
        self,
        address: int,
        pc: int,
        size: int,
        value: int,
        backend=None,
    ) -> bool:
        """Apply profile transitions without generating another read value."""
        if not self.enabled or backend is None:
            return False
        changed = False
        for profile in self.profiles:
            try:
                if profile.apply_external_read_side_effects(
                    address,
                    pc,
                    size,
                    value,
                    backend,
                ):
                    changed = True
            except Exception:
                continue
        return changed

    def apply_write(
        self,
        address: int,
        size: int,
        backend=None,
        *,
        previous_value: Optional[int] = None,
        written_value: Optional[int] = None,
        pc: int = 0,
    ) -> bool:
        if not self.enabled or backend is None:
            return False
        changed = self._apply_svd_write_semantics(
            address,
            size,
            backend,
            previous_value=previous_value,
            written_value=written_value,
        )
        for profile in self.profiles:
            if profile.apply_write(address, size, backend):
                changed = True
        if changed:
            self.stats["profile_write_updates"] += 1
        return changed

    def learn_polling_exit_rule(
        self,
        *,
        status_address: int,
        value: int,
        read_pc: Optional[int],
        constraint_pc: Optional[int],
        loop_head: Optional[int],
        mask: Optional[int],
        source: str,
        confidence: float = 1.0,
    ) -> Optional[SemanticMMIORule]:
        """Learn a rule only after the caller validated the loop-exit value."""
        if not self.enabled:
            return None
        status_address = int(status_address) & 0xFFFFFFFF
        value = int(value) & 0xFFFFFFFF
        read_pc_i = None if read_pc is None else int(read_pc) & 0xFFFFFFFF
        constraint_pc_i = None if constraint_pc is None else int(constraint_pc) & 0xFFFFFFFF
        loop_head_i = None if loop_head is None else int(loop_head) & 0xFFFFFFFF
        mask_i = None if mask is None else int(mask) & 0xFFFFFFFF
        if mask_i == 0:
            mask_i = None

        key = (status_address, read_pc_i, constraint_pc_i, value if mask_i is None else value & mask_i)
        svd_info = self.svd_index.register_for(status_address)
        related_write = self._find_recent_related_write(status_address)
        kind = "polling_exit"
        if related_write is not None:
            kind = "write_then_status"
        elif svd_info and svd_info.is_status_like():
            kind = "status_like_polling_exit"
        for rule in self.rules:
            if (
                int(rule.status_address) == status_address
                and rule.read_pc == read_pc_i
                and rule.constraint_pc == constraint_pc_i
                and (rule.mask or 0) == (mask_i or 0)
                and (rule.value & (mask_i or 0xFFFFFFFF)) == (value & (mask_i or 0xFFFFFFFF))
            ):
                if read_pc_i is not None:
                    rule.evidence_read_pcs.add(read_pc_i)
                self._maybe_promote(rule)
                return rule

        rule = SemanticMMIORule(
            status_address=status_address,
            value=value,
            mask=mask_i,
            read_pc=read_pc_i,
            constraint_pc=constraint_pc_i,
            loop_head=loop_head_i,
            source=source,
            confidence=max(0.0, min(1.0, float(confidence))),
            related_write_address=related_write.address if related_write else None,
            related_write_pc=related_write.pc if related_write else None,
            related_write_value=related_write.value if related_write else None,
            related_write_timestamp=related_write.timestamp if related_write else None,
            evidence_read_pcs={read_pc_i} if read_pc_i is not None else set(),
            kind=kind,
            transaction_signature=(
                f"0x{int(status_address) & 0xFFFFFFFF:08x}"
                f"<=0x{int(related_write.address) & 0xFFFFFFFF:08x}"
                f":0x{int(related_write.value) & 0xFFFFFFFF:08x}"
                if related_write is not None
                else None
            ),
        )
        self.rules.append(rule)
        self.learned_rule_keys.add(key)
        self.stats["rules_learned"] += 1
        if related_write is not None:
            self.stats["transaction_rules_learned"] += 1
        self._maybe_promote(rule)
        return rule

    def _maybe_promote(self, rule: SemanticMMIORule) -> None:
        if rule.promoted:
            return
        if rule.mask is None:
            return
        svd_info = self.svd_index.register_for(rule.status_address)
        svd_supports_status = bool(svd_info and svd_info.is_status_like())
        related_rules = [
            item
            for item in self.rules
            if int(item.status_address) == int(rule.status_address)
            and (item.mask or 0) == (rule.mask or 0)
            and (item.value & int(rule.mask)) == (rule.value & int(rule.mask))
        ]
        evidence_read_pcs: Set[int] = set()
        for item in related_rules:
            evidence_read_pcs.update(item.evidence_read_pcs)
        enough_evidence = len(evidence_read_pcs) >= max(1, self.promote_evidence)
        if svd_supports_status or enough_evidence:
            for item in related_rules:
                if item.promoted:
                    continue
                item.promoted = True
                self.stats["rules_promoted"] += 1
                if item.related_write_address is not None:
                    self.stats["transaction_rule_promoted"] += 1

    def _find_recent_related_write(self, status_address: int) -> Optional[MMIOAccessEvent]:
        status_key = self.svd_index.peripheral_key_for(status_address)
        transaction = self.transactions.get(status_key)
        if transaction is not None and transaction.last_write_by_address:
            return max(
                transaction.last_write_by_address.values(),
                key=lambda event: (
                    int(event.transaction_sequence),
                    int(event.timestamp),
                ),
            )
        for event in reversed(self.events):
            if event.is_read:
                continue
            if self.svd_index.peripheral_key_for(event.address) == status_key:
                return event
        return None

    def recent_transaction_matches(self, rule: SemanticMMIORule) -> bool:
        """Return True only when the validated write->status context is present."""
        if rule.related_write_address is None:
            return True
        related_addr = int(rule.related_write_address) & 0xFFFFFFFF
        related_value = None if rule.related_write_value is None else int(rule.related_write_value) & 0xFFFFFFFF
        transaction = self.transactions.get(
            self.svd_index.peripheral_key_for(rule.status_address)
        )
        if transaction is not None:
            latest = transaction.last_write_by_address.get(related_addr)
            if latest is None:
                return False
            max_distance = self._env_int(
                "LSGEMU_GENERIC_MMIO_TRANSACTION_WINDOW", 128
            )
            event_distance = next(
                (
                    distance
                    for distance, event in enumerate(reversed(self.events))
                    if event == latest
                ),
                None,
            )
            if event_distance is None or int(event_distance) > max_distance:
                return False
            if related_value is not None and (
                int(latest.value) & _size_mask(latest.size)
            ) != (related_value & _size_mask(latest.size)):
                return False
            return True
        max_distance = self._env_int("LSGEMU_GENERIC_MMIO_TRANSACTION_WINDOW", 128)
        for distance, event in enumerate(reversed(self.events)):
            if distance > max_distance:
                break
            if event.is_read:
                continue
            if (int(event.address) & 0xFFFFFFFF) != related_addr:
                continue
            if related_value is not None and (int(event.value) & _size_mask(event.size)) != (related_value & _size_mask(event.size)):
                continue
            return True
        return False

    @staticmethod
    def _extract_value(word_value: int, read_addr: int, size: int) -> int:
        size = max(1, min(4, int(size or 4)))
        mask = (1 << (size * 8)) - 1
        shift = (int(read_addr) & 0x3) * 8
        return (int(word_value) >> shift) & mask

    def infer_status_mask_from_svd(self, address: int) -> Optional[int]:
        info = self.svd_index.register_for(address)
        if not info:
            return None
        return info.status_field_mask()

    def snapshot_runtime_state(
        self,
        *,
        copy_values: bool = True,
    ) -> Dict[str, object]:
        """Capture mutable profile state without duplicating the immutable SVD index."""
        if not copy_values:
            return {
                "events": self.events,
                "events_maxlen": self.events.maxlen,
                "rules": self.rules,
                "learned_rule_keys": self.learned_rule_keys,
                "stats": self.stats,
                "profiles": self.profiles,
                "transactions": self.transactions,
                # Timer/NVIC state is small and must be value-captured even on the
                # compressed path (see _snapshot_profile_states).
                "profile_states": copy.deepcopy(self._snapshot_profile_states()),
            }
        return {
            "events": copy.deepcopy(list(self.events)),
            "events_maxlen": self.events.maxlen,
            "rules": copy.deepcopy(self.rules),
            "learned_rule_keys": set(self.learned_rule_keys),
            "stats": dict(self.stats),
            "profiles": copy.deepcopy(self.profiles),
            "transactions": copy.deepcopy(self.transactions),
            "profile_states": copy.deepcopy(self._snapshot_profile_states()),
        }

    def restore_runtime_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore mutable profile state captured by :meth:`snapshot_runtime_state`."""
        if not isinstance(state, dict):
            return
        maxlen = state.get("events_maxlen", self.events.maxlen)
        try:
            maxlen = max(1, int(maxlen or self.events.maxlen or 512))
        except (TypeError, ValueError):
            maxlen = self.events.maxlen or 512
        self.events = deque(copy.deepcopy(list(state.get("events", []) or [])), maxlen=maxlen)
        self.rules = copy.deepcopy(list(state.get("rules", []) or []))
        self.learned_rule_keys = set(state.get("learned_rule_keys", set()) or set())
        restored_stats = state.get("stats")
        if isinstance(restored_stats, dict):
            self.stats = {str(key): int(value or 0) for key, value in restored_stats.items()}
        restored_profiles = state.get("profiles")
        if isinstance(restored_profiles, list):
            self.profiles = copy.deepcopy(restored_profiles)
        # Applied last so a capture-point timer/NVIC state wins over the profile
        # objects that the (possibly alias-returned) "profiles" list carries.
        self._restore_profile_states(state.get("profile_states"))
        restored_transactions = state.get("transactions")
        if isinstance(restored_transactions, dict):
            self.transactions = copy.deepcopy(restored_transactions)

    def summary(self) -> Dict[str, object]:
        summary = dict(self.stats)
        summary["rule_kind_counts"] = dict(Counter(rule.kind for rule in self.rules))
        summary["transaction_signatures"] = [
            rule.transaction_signature
            for rule in self.rules
            if rule.transaction_signature
        ][-16:]
        summary["transaction_states"] = {
            key: transaction.summary()
            for key, transaction in sorted(self.transactions.items())
        }
        profile_summaries = {}
        for profile in self.profiles:
            profile_summary = getattr(profile, "summary", None)
            if callable(profile_summary):
                try:
                    profile_summaries[getattr(profile, "name", profile.__class__.__name__)] = profile_summary()
                except Exception:
                    continue
        if profile_summaries:
            summary["profile_summaries"] = profile_summaries
        return summary
