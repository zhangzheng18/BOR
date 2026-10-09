#!/usr/bin/env python3
"""Runtime crash monitor for Unicorn-based MCU rehosting.

The monitor is deliberately small: it installs Unicorn hooks, records the guest
state at the first crash-like event, and converts that state into the same
plain run-result dictionary consumed by :mod:`fuzzengine.crash_detector`.
Native Unicorn/Python crashes still need process-level supervision; in-process
hooks can only see guest-level events before the host process dies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from unicorn import (
        UC_HOOK_CODE,
        UC_HOOK_MEM_INVALID,
        UC_MEM_FETCH_PROT,
        UC_MEM_FETCH_UNMAPPED,
        UC_MEM_READ_PROT,
        UC_MEM_READ_UNMAPPED,
        UC_MEM_WRITE_PROT,
        UC_MEM_WRITE_UNMAPPED,
    )
    from unicorn.arm_const import (
        UC_ARM_REG_LR,
        UC_ARM_REG_PC,
        UC_ARM_REG_R0,
        UC_ARM_REG_R1,
        UC_ARM_REG_R10,
        UC_ARM_REG_R11,
        UC_ARM_REG_R12,
        UC_ARM_REG_R2,
        UC_ARM_REG_R3,
        UC_ARM_REG_R4,
        UC_ARM_REG_R5,
        UC_ARM_REG_R6,
        UC_ARM_REG_R7,
        UC_ARM_REG_R8,
        UC_ARM_REG_R9,
        UC_ARM_REG_SP,
    )
    # Unicorn 1.x does not expose XPSR on every ARM binding.  It is useful
    # evidence, but not required for invalid-memory or fault monitoring.
    try:
        from unicorn.arm_const import UC_ARM_REG_XPSR
    except (ImportError, AttributeError):
        UC_ARM_REG_XPSR = None
except Exception:  # pragma: no cover - import failure is reported at install time.
    UC_HOOK_CODE = None
    UC_HOOK_MEM_INVALID = None
    UC_MEM_FETCH_PROT = UC_MEM_FETCH_UNMAPPED = None
    UC_MEM_READ_PROT = UC_MEM_READ_UNMAPPED = None
    UC_MEM_WRITE_PROT = UC_MEM_WRITE_UNMAPPED = None
    UC_ARM_REG_LR = UC_ARM_REG_PC = UC_ARM_REG_SP = UC_ARM_REG_XPSR = None
    UC_ARM_REG_R0 = UC_ARM_REG_R1 = UC_ARM_REG_R2 = UC_ARM_REG_R3 = None
    UC_ARM_REG_R4 = UC_ARM_REG_R5 = UC_ARM_REG_R6 = UC_ARM_REG_R7 = None
    UC_ARM_REG_R8 = UC_ARM_REG_R9 = UC_ARM_REG_R10 = UC_ARM_REG_R11 = UC_ARM_REG_R12 = None

try:
    from .crash_detector import CrashDetectorConfig
except ImportError:  # Allow running tests as ``python3 fuzzengine/test_*.py``.
    from crash_detector import CrashDetectorConfig


def _hex(value: Optional[int]) -> Optional[str]:
    return f"0x{int(value) & 0xFFFFFFFF:08x}" if value is not None else None


def _in_ranges(address: Optional[int], ranges: Sequence[Tuple[int, int]]) -> bool:
    if address is None:
        return False
    normalized = int(address) & ~1
    return any(int(start) <= normalized < int(end) for start, end in ranges)


def _event_access_name(access: int) -> str:
    names = {
        UC_MEM_READ_UNMAPPED: "read_unmapped",
        UC_MEM_WRITE_UNMAPPED: "write_unmapped",
        UC_MEM_FETCH_UNMAPPED: "fetch_unmapped",
        UC_MEM_READ_PROT: "read_prot",
        UC_MEM_WRITE_PROT: "write_prot",
        UC_MEM_FETCH_PROT: "fetch_prot",
    }
    return names.get(access, f"mem_access_{access}")


def _event_stop_reason(access_name: str) -> str:
    if "prot" in access_name:
        return "Invalid memory protection (UC_ERR_PROT)"
    if "write" in access_name:
        return "Invalid memory write (UC_ERR_WRITE_UNMAPPED)"
    if "fetch" in access_name:
        return "Invalid memory fetch (UC_ERR_FETCH_UNMAPPED)"
    return "Invalid memory read (UC_ERR_READ_UNMAPPED)"


@dataclass
class RuntimeCrashEvent:
    """Guest-level event captured while Unicorn is executing firmware code."""

    kind: str
    stop_reason: str
    pc: Optional[str] = None
    lr: Optional[str] = None
    sp: Optional[str] = None
    access: Optional[str] = None
    access_address: Optional[str] = None
    size: Optional[int] = None
    value: Optional[int] = None
    handler: Optional[str] = None
    trace_tail: List[str] = field(default_factory=list)
    registers: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RuntimeCrashMonitor:
    """Install Unicorn hooks that discover crash-like guest events online."""

    def __init__(
        self,
        uc,
        config: Optional[CrashDetectorConfig] = None,
        *,
        trace_tail_limit: int = 64,
        stop_on_pc_outside_code: bool = False,
        monitor_invalid_memory: bool = True,
        hook_add: Optional[Callable[..., object]] = None,
        hook_del: Optional[Callable[[object], None]] = None,
    ):
        self.uc = uc
        self.config = config or CrashDetectorConfig()
        self.trace_tail_limit = max(1, int(trace_tail_limit or 64))
        self.stop_on_pc_outside_code = bool(stop_on_pc_outside_code)
        self.monitor_invalid_memory = bool(monitor_invalid_memory)
        self.trace_tail: List[int] = []
        self.event: Optional[RuntimeCrashEvent] = None
        self._hooks: List[object] = []
        self._installed = False
        # The owning emulator can provide lifecycle-aware operations.  Tests
        # and standalone users continue to use the Unicorn methods directly.
        self._hook_add = hook_add or self.uc.hook_add
        self._hook_del = hook_del or self.uc.hook_del

    def install(self) -> None:
        if self._installed:
            return
        if UC_HOOK_CODE is None or UC_HOOK_MEM_INVALID is None:
            raise RuntimeError("unicorn Python bindings are not available")
        if self.monitor_invalid_memory:
            self._hooks.append(self._hook_add(UC_HOOK_MEM_INVALID, self._invalid_mem_hook))
        self._hooks.append(self._hook_add(UC_HOOK_CODE, self._code_hook))
        self._installed = True

    def uninstall(self) -> None:
        for hook in reversed(self._hooks):
            try:
                self._hook_del(hook)
            except Exception:
                pass
        self._hooks.clear()
        self._installed = False

    def reset(self) -> None:
        self.trace_tail.clear()
        self.event = None

    def _invalid_mem_hook(self, uc, access, address, size, value, user_data=None):
        access_name = _event_access_name(int(access))
        registers = self._read_registers(uc)
        pc = registers.get("pc")
        event = RuntimeCrashEvent(
            kind="invalid_memory_access",
            stop_reason=_event_stop_reason(access_name),
            pc=_hex(pc),
            lr=_hex(registers.get("lr")),
            sp=_hex(registers.get("sp")),
            access=access_name,
            access_address=_hex(address),
            size=int(size),
            value=int(value) & 0xFFFFFFFF if value is not None else None,
            trace_tail=[_hex(item) or "0x00000000" for item in self.trace_tail[-self.trace_tail_limit :]],
            registers={key: _hex(val) or "0x00000000" for key, val in registers.items()},
            evidence={
                "address_in_mmio_range": _in_ranges(address, self.config.mmio_ranges),
                "address_in_ram_range": _in_ranges(address, self.config.ram_ranges),
                "address_in_code_range": _in_ranges(address, self.config.code_ranges),
            },
        )
        self._record_event(event)
        return False

    def _code_hook(self, uc, address, size, user_data=None) -> None:
        normalized = int(address) & ~1
        self.trace_tail.append(normalized)
        if len(self.trace_tail) > self.trace_tail_limit * 2:
            del self.trace_tail[: len(self.trace_tail) - self.trace_tail_limit]

        if self.event is not None:
            return
        registers = None
        if normalized in self.config.crash_points:
            registers = self._read_registers(uc)
            self._record_event(RuntimeCrashEvent(
                kind="configured_crash_point",
                stop_reason="configured_crash_point",
                pc=_hex(normalized),
                lr=_hex(registers.get("lr")),
                sp=_hex(registers.get("sp")),
                trace_tail=[_hex(item) or "0x00000000" for item in self.trace_tail[-self.trace_tail_limit :]],
                registers={key: _hex(val) or "0x00000000" for key, val in registers.items()},
                evidence={"crash_point": _hex(normalized)},
            ))
            return

        for name, handler_addr in self.config.fault_handler_addrs.items():
            if normalized == (int(handler_addr) & ~1):
                registers = registers or self._read_registers(uc)
                self._record_event(RuntimeCrashEvent(
                    kind="fault_handler_pc",
                    stop_reason=f"fault_handler_pc:{name}",
                    pc=_hex(normalized),
                    lr=_hex(registers.get("lr")),
                    sp=_hex(registers.get("sp")),
                    handler=str(name),
                    trace_tail=[_hex(item) or "0x00000000" for item in self.trace_tail[-self.trace_tail_limit :]],
                    registers={key: _hex(val) or "0x00000000" for key, val in registers.items()},
                    evidence={"handler": str(name), "handler_address": _hex(handler_addr)},
                ))
                return

        if (
            self.stop_on_pc_outside_code
            and self.config.code_ranges
            and not _in_ranges(normalized, self.config.code_ranges)
        ):
            registers = registers or self._read_registers(uc)
            self._record_event(RuntimeCrashEvent(
                kind="pc_outside_executable_ranges",
                stop_reason="pc_outside_executable_ranges",
                pc=_hex(normalized),
                lr=_hex(registers.get("lr")),
                sp=_hex(registers.get("sp")),
                trace_tail=[_hex(item) or "0x00000000" for item in self.trace_tail[-self.trace_tail_limit :]],
                registers={key: _hex(val) or "0x00000000" for key, val in registers.items()},
                evidence={"code_ranges": [[_hex(start), _hex(end)] for start, end in self.config.code_ranges[:16]]},
            ))

    def _record_event(self, event: RuntimeCrashEvent) -> None:
        if self.event is not None:
            return
        self.event = event
        try:
            self.uc.emu_stop()
        except Exception:
            pass

    def _read_registers(self, uc) -> Dict[str, int]:
        regs = {
            "pc": UC_ARM_REG_PC,
            "lr": UC_ARM_REG_LR,
            "sp": UC_ARM_REG_SP,
            "xpsr": UC_ARM_REG_XPSR,
            "r0": UC_ARM_REG_R0,
            "r1": UC_ARM_REG_R1,
            "r2": UC_ARM_REG_R2,
            "r3": UC_ARM_REG_R3,
            "r4": UC_ARM_REG_R4,
            "r5": UC_ARM_REG_R5,
            "r6": UC_ARM_REG_R6,
            "r7": UC_ARM_REG_R7,
            "r8": UC_ARM_REG_R8,
            "r9": UC_ARM_REG_R9,
            "r10": UC_ARM_REG_R10,
            "r11": UC_ARM_REG_R11,
            "r12": UC_ARM_REG_R12,
        }
        out: Dict[str, int] = {}
        for name, reg_id in regs.items():
            if reg_id is None:
                continue
            try:
                out[name] = int(uc.reg_read(reg_id)) & 0xFFFFFFFF
            except Exception:
                continue
        return out

    def as_record(self, *, exception: Optional[BaseException] = None, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Return a crash-detector compatible run-result dictionary."""
        event = self.event
        registers = event.registers if event else {
            key: _hex(value) or "0x00000000" for key, value in self._read_registers(self.uc).items()
        }
        stop_reason = event.stop_reason if event else "completed"
        if exception is not None and event is None:
            stop_reason = f"{type(exception).__name__}: {exception}"
        record: Dict[str, object] = {
            "source_layer": "runtime_monitor",
            "stop_reason": stop_reason,
            "registers": registers,
            "trace_tail": event.trace_tail if event else [_hex(item) for item in self.trace_tail[-self.trace_tail_limit :]],
        }
        if event is not None:
            record["runtime_crash_event"] = event.to_dict()
            if event.access_address:
                record["last_unmapped_access"] = {
                    "access": event.access,
                    "address": event.access_address,
                    "size": event.size,
                    "value": event.value,
                    "pc": event.pc,
                    "registers": event.registers,
                }
        if extra:
            record.update(extra)
        return record
