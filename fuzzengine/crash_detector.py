#!/usr/bin/env python3
"""Crash triage utilities for LSGEmu fuzzing runs.

The detector is intentionally independent from the emulator loop.  It consumes
plain execution dictionaries such as ``IntelligentEmulator.run()`` results,
stream/direct-call debug records, or process return-code records and produces
stable crash buckets.  This mirrors the practical pieces used by Fuzzware,
MultiFuzz, and uEmu: classify exit reasons, distinguish hangs, honor explicit
crash points, and bucket by PC/LR/access context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from .utils import parse_int, safe_fragment, safe_int, short_hash
except ImportError:
    from utils import parse_int, safe_fragment, safe_int, short_hash

try:
    from lsgemu.artifact_io import atomic_json_dump, atomic_write_bytes
except ImportError:  # pragma: no cover - supports running fuzzengine standalone.
    atomic_json_dump = atomic_write_bytes = None


FIRMWARE_CRASH = "firmware_crash"
HANG = "hang"
MODEL_ARTIFACT = "model_or_harness_artifact"
TOOLING_FAILURE = "tooling_or_runtime_failure"
NORMAL = "normal"

TRIGGER_EXTERNAL_INPUT = "external_input"
TRIGGER_MMIO_ENVIRONMENT = "mmio_peripheral_environment"
TRIGGER_IRQ_ENVIRONMENT = "irq_or_interrupt_environment"
TRIGGER_EXTERNAL_PLUS_MMIO = "external_input_plus_peripheral_state"
TRIGGER_MODEL_ARTIFACT = "model_or_harness_artifact"
TRIGGER_TOOLING_RUNTIME = "tooling_or_runtime_failure"
TRIGGER_UNKNOWN = "unknown"

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

FAULT_HANDLER_NAMES = {
    "hardfault",
    "hard_fault",
    "memmanage",
    "memorymanagement",
    "busfault",
    "bus_fault",
    "usagefault",
    "usage_fault",
    "securefault",
}

NATIVE_CRASH_SIGNALS = {
    getattr(signal, "SIGSEGV", None): "SIGSEGV",
    getattr(signal, "SIGABRT", None): "SIGABRT",
    getattr(signal, "SIGBUS", None): "SIGBUS",
    getattr(signal, "SIGILL", None): "SIGILL",
    getattr(signal, "SIGFPE", None): "SIGFPE",
}
NATIVE_CRASH_SIGNALS = {key: value for key, value in NATIVE_CRASH_SIGNALS.items() if key is not None}

def _hex(value: Optional[int]) -> Optional[str]:
    return f"0x{int(value) & 0xFFFFFFFF:08x}" if value is not None else None


def _normalize_addr_set(values: Optional[Iterable[object]]) -> Set[int]:
    out: Set[int] = set()
    for value in values or []:
        parsed = parse_int(value)
        if parsed is not None:
            out.add(parsed & ~1)
    return out


def _normalize_ranges(values: Optional[Iterable[Tuple[object, object]]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for start, end in values or []:
        parsed_start = parse_int(start)
        parsed_end = parse_int(end)
        if parsed_start is None or parsed_end is None:
            continue
        if parsed_end < parsed_start:
            parsed_start, parsed_end = parsed_end, parsed_start
        ranges.append((parsed_start, parsed_end))
    return ranges


def _in_ranges(address: Optional[int], ranges: Sequence[Tuple[int, int]]) -> bool:
    if address is None:
        return False
    address = int(address) & ~1
    return any(start <= address < end for start, end in ranges)


def _looks_like_poison_or_controlled_pointer(address: Optional[int]) -> bool:
    """Heuristic for values that should not be hidden as missing MMIO models."""
    if address is None:
        return False
    value = int(address) & 0xFFFFFFFF
    if value in {
        0x00000000,
        0xFFFFFFFF,
        0xDEADBEEF,
        0xBAADF00D,
        0xBAD0C0DE,
        0xCAFEBABE,
        0xCCCCCCCC,
        0xFEEEFEEE,
        0xAAAAAAAA,
        0x55555555,
    }:
        return True
    bytes_le = [(value >> (8 * index)) & 0xFF for index in range(4)]
    if len(set(bytes_le)) == 1 and bytes_le[0] not in {0x00, 0xFF}:
        return True
    printable = sum(1 for byte in bytes_le if 0x20 <= byte <= 0x7E)
    return printable >= 3

def _seed_sha256(seed_bytes: Optional[bytes], seed_path: Optional[object]) -> Optional[str]:
    if seed_bytes is not None:
        return hashlib.sha256(seed_bytes).hexdigest()
    if seed_path:
        path = Path(seed_path)
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return None
    return None


@dataclass
class CrashSignal:
    """One normalized crash/hang/tooling signal extracted from an execution."""

    kind: str
    category: str
    confidence: str
    reason: str
    severity: str = LOW
    source_layer: str = ""
    pc: Optional[str] = None
    lr: Optional[str] = None
    sp: Optional[str] = None
    access: Optional[str] = None
    access_address: Optional[str] = None
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CrashReport:
    """Crash triage result for one execution or seed replay."""

    is_crash: bool
    is_hang: bool
    category: str
    classification: str
    confidence: str
    severity: str
    bucket_key: str
    stop_reason: str = ""
    source_layer: str = ""
    requires_replay_validation: bool = False
    validation_priority: str = LOW
    evidence_level: str = "not_firmware_crash"
    firmware: Optional[str] = None
    input_id: Optional[str] = None
    seed_sha256: Optional[str] = None
    pc: Optional[str] = None
    lr: Optional[str] = None
    sp: Optional[str] = None
    access_address: Optional[str] = None
    phase: Optional[str] = None
    trigger_source: str = TRIGGER_UNKNOWN
    trigger_confidence: str = LOW
    trigger_evidence: Dict[str, object] = field(default_factory=dict)
    trace_tail: List[str] = field(default_factory=list)
    signals: List[CrashSignal] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["signals"] = [signal.to_dict() for signal in self.signals]
        return data


@dataclass
class CrashDetectorConfig:
    """Configuration for firmware crash triage."""

    crash_points: Set[int] = field(default_factory=set)
    fault_handler_addrs: Dict[str, int] = field(default_factory=dict)
    code_ranges: List[Tuple[int, int]] = field(default_factory=list)
    ram_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [(0x20000000, 0x20100000)])
    mmio_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0x40000000, 0x60000000), (0xE0000000, 0xE0100000)]
    )
    classify_timeouts_as_hangs: bool = True
    classify_max_instruction_as_hang: bool = False
    require_code_range_for_pc_crash: bool = False
    treat_unmapped_mmio_as_artifact: bool = True
    treat_unmapped_configured_memory_as_artifact: bool = True
    bucket_trace_tail: int = 8

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "CrashDetectorConfig":
        payload = dict(payload or {})
        fault_handler_addrs: Dict[str, int] = {}
        for name, addr in (payload.get("fault_handler_addrs") or {}).items():
            parsed = parse_int(addr)
            if parsed is not None:
                fault_handler_addrs[str(name).lower()] = parsed & ~1
        return cls(
            crash_points=_normalize_addr_set(payload.get("crash_points")),
            fault_handler_addrs=fault_handler_addrs,
            code_ranges=_normalize_ranges(payload.get("code_ranges")),
            ram_ranges=_normalize_ranges(payload.get("ram_ranges")) or [(0x20000000, 0x20100000)],
            mmio_ranges=_normalize_ranges(payload.get("mmio_ranges"))
            or [(0x40000000, 0x60000000), (0xE0000000, 0xE0100000)],
            classify_timeouts_as_hangs=bool(payload.get("classify_timeouts_as_hangs", True)),
            classify_max_instruction_as_hang=bool(payload.get("classify_max_instruction_as_hang", False)),
            require_code_range_for_pc_crash=bool(payload.get("require_code_range_for_pc_crash", False)),
            treat_unmapped_mmio_as_artifact=bool(payload.get("treat_unmapped_mmio_as_artifact", True)),
            treat_unmapped_configured_memory_as_artifact=bool(
                payload.get("treat_unmapped_configured_memory_as_artifact", True)
            ),
            bucket_trace_tail=int(payload.get("bucket_trace_tail", 8) or 8),
        )


class CrashDetector:
    """Classify and bucket crash-like outcomes from emulation/fuzzing runs."""

    _invalid_memory_patterns = (
        (re.compile(r"UC_ERR_READ_UNMAPPED|Invalid memory read|read_unmapped", re.I), "invalid_memory_read"),
        (re.compile(r"UC_ERR_WRITE_UNMAPPED|Invalid memory write|write_unmapped", re.I), "invalid_memory_write"),
        (re.compile(r"UC_ERR_FETCH_UNMAPPED|Invalid memory fetch|fetch_unmapped", re.I), "invalid_memory_fetch"),
        (re.compile(r"UC_ERR_READ_PROT|UC_ERR_WRITE_PROT|UC_ERR_FETCH_PROT|protection", re.I), "memory_protection"),
    )
    _invalid_instruction_pattern = re.compile(r"UC_ERR_INSN_INVALID|Invalid instruction|illegal instruction", re.I)
    _model_artifact_pattern = re.compile(
        r"unproven_dynamic_code|停止未证明动态代码执行|UC_ERR_MAP|Invalid memory mapping|restore_failed|"
        r"state_prefix_mismatch|ISR.*压栈失败|压栈失败",
        re.I,
    )
    _benign_terminal_pattern = re.compile(
        r"^$|^completed$|^returned$|^stop_before_pc$|^target_probe_tail_complete$|^fatal_sink_terminal$|^exit_called$"
        r"|^quiescence_no_new_bbs$|^timeout_or_max_instructions_reached$",
        re.I,
    )
    _timeout_pattern = re.compile(r"timeout|hang|quiescence|hot_loop|InstructionLimit", re.I)
    _max_instruction_pattern = re.compile(r"max_instructions|max_instruction|instruction_limit", re.I)
    _explicit_fault_pattern = re.compile(
        r"hardfault|hard_fault|busfault|bus_fault|usagefault|usage_fault|memmanage|securefault|fault_handler",
        re.I,
    )
    _abort_pattern = re.compile(r"abort_called|stack_chk_fail|assert|panic|fatal", re.I)
    _normal_pattern = re.compile(
        r"^$|completed|returned|stopped|stop_before_pc|target_probe_tail_complete|fatal_sink_terminal|exit_called",
        re.I,
    )

    def __init__(self, config: Optional[CrashDetectorConfig] = None):
        self.config = config or CrashDetectorConfig()

    def detect(
        self,
        run_result: Optional[Dict[str, object]] = None,
        *,
        exception: Optional[BaseException] = None,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        firmware: Optional[str] = None,
        input_id: Optional[str] = None,
        seed_bytes: Optional[bytes] = None,
        seed_path: Optional[object] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> CrashReport:
        record = dict(run_result or {})
        if exception is not None:
            record.setdefault("stop_reason", f"{type(exception).__name__}: {exception}")
        if returncode is not None:
            record["process_returncode"] = int(returncode)
        if stdout:
            record.setdefault("stdout_tail", str(stdout)[-4096:])
        if stderr:
            record.setdefault("stderr_tail", str(stderr)[-4096:])
        metadata = dict(metadata or {})
        source_layer = str(
            record.get("source_layer")
            or metadata.get("source_layer")
            or ("process" if returncode is not None else "run_result")
        )

        stop_reason = str(record.get("stop_reason") or record.get("exit") or "")
        registers = self._extract_registers(record)
        pc = registers.get("pc")
        lr = registers.get("lr")
        sp = registers.get("sp")
        trace_tail = self._extract_trace_tail(record)
        access = self._extract_access(record)

        signals: List[CrashSignal] = []
        signals.extend(self._process_signals(record, pc, lr, sp))
        signals.extend(self._stop_reason_signals(stop_reason, record, pc, lr, sp, access))
        signals.extend(self._pc_signals(record, pc, lr, sp))
        signals.extend(self._sp_signals(record, pc, lr, sp))
        for signal in signals:
            if not signal.source_layer:
                signal.source_layer = source_layer

        category, classification, confidence, severity, is_crash, is_hang = self._collapse_signals(signals)
        requires_replay_validation, validation_priority, evidence_level = self._validation_fields(
            category,
            confidence,
            severity,
            source_layer,
        )
        signal_access_address = next((signal.access_address for signal in signals if signal.access_address), None)
        access_address = signal_access_address
        bucket_key = self._bucket_key(
            category=category,
            classification=classification,
            pc=pc,
            lr=lr,
            access_address=parse_int(access_address),
            stop_reason=stop_reason,
            trace_tail=trace_tail,
        )
        if not signals and self._normal_pattern.search(stop_reason or "completed"):
            classification = "normal_exit"
            category = NORMAL
        trigger_source, trigger_confidence, trigger_evidence = self._classify_trigger_source(
            record,
            category=category,
            source_layer=source_layer,
            stop_reason=stop_reason,
            access=access,
        )

        return CrashReport(
            is_crash=is_crash,
            is_hang=is_hang,
            category=category,
            classification=classification,
            confidence=confidence,
            severity=severity,
            bucket_key=bucket_key,
            stop_reason=stop_reason,
            source_layer=source_layer,
            requires_replay_validation=requires_replay_validation,
            validation_priority=validation_priority,
            evidence_level=evidence_level,
            firmware=firmware,
            input_id=input_id,
            seed_sha256=_seed_sha256(seed_bytes, seed_path),
            pc=_hex(pc),
            lr=_hex(lr),
            sp=_hex(sp),
            access_address=_hex(parse_int(access_address)),
            phase=str(record.get("phase") or metadata.get("phase") or "") or None,
            trigger_source=trigger_source,
            trigger_confidence=trigger_confidence,
            trigger_evidence=trigger_evidence,
            trace_tail=trace_tail,
            signals=signals,
            metadata={
                **metadata,
                "record_summary": self._record_summary(record),
            },
        )

    def _extract_registers(self, record: Dict[str, object]) -> Dict[str, int]:
        registers = record.get("registers") or {}
        if not registers and isinstance(record.get("run_result"), dict):
            registers = record.get("run_result", {}).get("registers") or {}
        out: Dict[str, int] = {}
        if isinstance(registers, dict):
            for name, value in registers.items():
                parsed = parse_int(value)
                if parsed is not None:
                    out[str(name).lower()] = parsed
        for name in ("pc", "lr", "sp"):
            parsed = parse_int(record.get(name))
            if parsed is not None:
                out.setdefault(name, parsed)
        unmapped = record.get("last_unmapped_access")
        if isinstance(unmapped, dict):
            unmapped_registers = unmapped.get("registers") or {}
            if isinstance(unmapped_registers, dict):
                for name, value in unmapped_registers.items():
                    parsed = parse_int(value)
                    if parsed is not None:
                        out.setdefault(str(name).lower(), parsed)
            parsed_pc = parse_int(unmapped.get("pc"))
            if parsed_pc is not None:
                out.setdefault("pc", parsed_pc)
        return out

    def _extract_trace_tail(self, record: Dict[str, object]) -> List[str]:
        raw = (
            record.get("trace_tail")
            or record.get("last_10_pcs")
            or record.get("callstack")
            or []
        )
        if not raw and isinstance(record.get("run_result"), dict):
            raw = record.get("run_result", {}).get("last_10_pcs") or []
        out: List[str] = []
        for value in raw or []:
            parsed = parse_int(value)
            out.append(_hex(parsed) if parsed is not None else str(value))
        return out[-64:]

    def _extract_access(self, record: Dict[str, object]) -> Dict[str, object]:
        access = record.get("last_unmapped_access")
        if isinstance(access, dict) and access:
            return dict(access)
        invalid_memory = record.get("invalid_memory")
        if isinstance(invalid_memory, list) and invalid_memory:
            first = invalid_memory[0]
            if isinstance(first, dict):
                return dict(first)
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                return {"access": first[0], "address": first[1], "size": first[2]}
        memory_tail = record.get("memory_access_tail") or []
        if not memory_tail and isinstance(record.get("run_result"), dict):
            memory_tail = record.get("run_result", {}).get("memory_access_tail") or []
        if memory_tail and isinstance(memory_tail[-1], dict):
            return dict(memory_tail[-1])
        return {}

    def _process_signals(
        self,
        record: Dict[str, object],
        pc: Optional[int],
        lr: Optional[int],
        sp: Optional[int],
    ) -> List[CrashSignal]:
        signals: List[CrashSignal] = []
        # Timeout is an intentional campaign outcome, not a native signal.
        # ``subprocess_runner`` may use a negative sentinel when the child was
        # terminated after the deadline; treating it as SIG999 would hide the
        # actual hang classification.
        if bool(record.get("timed_out")):
            return signals
        returncode = record.get("process_returncode")
        if returncode is None:
            return signals
        try:
            code = int(returncode)
        except (TypeError, ValueError):
            return signals
        if code == -999:
            return signals
        if code < 0:
            signal_name = NATIVE_CRASH_SIGNALS.get(abs(code), f"signal_{abs(code)}")
            signals.append(CrashSignal(
                kind="native_process_crash",
                category=TOOLING_FAILURE,
                confidence=HIGH,
                severity=MEDIUM,
                reason=f"emulator process terminated by {signal_name}",
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"returncode": code, "signal": signal_name},
            ))
        return signals

    def _stop_reason_signals(
        self,
        stop_reason: str,
        record: Dict[str, object],
        pc: Optional[int],
        lr: Optional[int],
        sp: Optional[int],
        access: Dict[str, object],
    ) -> List[CrashSignal]:
        text = stop_reason or ""
        signals: List[CrashSignal] = []
        if not text:
            return signals
        if self._benign_terminal_pattern.fullmatch(text.strip()):
            return signals

        if self._model_artifact_pattern.search(text):
            signals.append(CrashSignal(
                kind="model_artifact_stop",
                category=MODEL_ARTIFACT,
                confidence=HIGH,
                severity=LOW,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"stop_reason": text},
            ))
            return signals

        for pattern, kind in self._invalid_memory_patterns:
            if pattern.search(text):
                access_addr = parse_int(access.get("address")) if access else None
                category, confidence, severity, normalized_kind, artifact_reason = self._classify_invalid_memory(
                    kind,
                    access_addr,
                )
                signals.append(CrashSignal(
                    kind=normalized_kind,
                    category=category,
                    confidence=confidence,
                    severity=severity,
                    reason=text,
                    pc=_hex(pc),
                    lr=_hex(lr),
                    sp=_hex(sp),
                    access=str(access.get("access") or kind) if access else kind,
                    access_address=_hex(access_addr),
                    evidence={
                        "stop_reason": text,
                        "last_unmapped_access": access,
                        "artifact_reason": artifact_reason,
                    },
                ))
                return signals

        if self._invalid_instruction_pattern.search(text):
            pc_in_code = _in_ranges(pc & ~1 if pc is not None else None, self.config.code_ranges)
            instruction_category = FIRMWARE_CRASH if pc_in_code else MODEL_ARTIFACT
            signals.append(CrashSignal(
                kind="invalid_instruction",
                category=instruction_category,
                confidence=MEDIUM if pc_in_code else HIGH,
                severity=MEDIUM if pc_in_code else LOW,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={
                    "stop_reason": text,
                    "pc_in_configured_code_range": bool(pc_in_code),
                },
            ))
        if self._explicit_fault_pattern.search(text):
            signals.append(CrashSignal(
                kind="fault_handler_stop",
                category=FIRMWARE_CRASH,
                confidence=HIGH,
                severity=HIGH,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"stop_reason": text},
            ))
        if self._abort_pattern.search(text):
            signals.append(CrashSignal(
                kind="abort_or_stack_guard",
                category=FIRMWARE_CRASH,
                confidence=HIGH,
                severity=HIGH if "stack_chk" in text else MEDIUM,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"stop_reason": text},
            ))
        if self.config.classify_timeouts_as_hangs and self._timeout_pattern.search(text):
            signals.append(CrashSignal(
                kind="hang_or_timeout",
                category=HANG,
                confidence=MEDIUM,
                severity=MEDIUM,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"stop_reason": text},
            ))
        elif self.config.classify_max_instruction_as_hang and self._max_instruction_pattern.search(text):
            signals.append(CrashSignal(
                kind="instruction_limit_hang",
                category=HANG,
                confidence=LOW,
                severity=LOW,
                reason=text,
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"stop_reason": text},
            ))
        return signals

    def _classify_invalid_memory(
        self,
        kind: str,
        access_addr: Optional[int],
    ) -> Tuple[str, str, str, str, Optional[str]]:
        """Separate guest crashes from missing rehosting mappings.

        In MCU rehosting, an unmapped access to a configured MMIO/code/RAM range
        usually means the harness did not map or model a legal firmware address.
        An access outside all configured address classes is a stronger firmware
        crash candidate.
        """
        if access_addr is not None:
            suspicious_pointer = _looks_like_poison_or_controlled_pointer(access_addr)
            if (
                self.config.treat_unmapped_mmio_as_artifact
                and _in_ranges(access_addr, self.config.mmio_ranges)
                and not suspicious_pointer
            ):
                suffix = kind.replace("invalid_memory_", "")
                return MODEL_ARTIFACT, HIGH, LOW, f"unmapped_mmio_{suffix}", "missing_peripheral_model_or_mmio_map"
            configured_ranges = list(self.config.code_ranges) + list(self.config.ram_ranges)
            if (
                self.config.treat_unmapped_configured_memory_as_artifact
                and configured_ranges
                and _in_ranges(access_addr, configured_ranges)
                and not suspicious_pointer
            ):
                suffix = kind.replace("invalid_memory_", "")
                return MODEL_ARTIFACT, MEDIUM, LOW, f"unmapped_configured_memory_{suffix}", "missing_memory_map"

        severity = HIGH if kind in {"invalid_memory_write", "invalid_memory_fetch"} else MEDIUM
        confidence = HIGH if access_addr is not None else MEDIUM
        return FIRMWARE_CRASH, confidence, severity, kind, None

    def _pc_signals(
        self,
        record: Dict[str, object],
        pc: Optional[int],
        lr: Optional[int],
        sp: Optional[int],
    ) -> List[CrashSignal]:
        signals: List[CrashSignal] = []
        normalized_pc = pc & ~1 if pc is not None else None
        if normalized_pc is None:
            return signals

        if normalized_pc in self.config.crash_points:
            signals.append(CrashSignal(
                kind="configured_crash_point",
                category=FIRMWARE_CRASH,
                confidence=HIGH,
                severity=HIGH,
                reason="PC reached configured crash point",
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"crash_point": _hex(normalized_pc)},
            ))
        for name, address in self.config.fault_handler_addrs.items():
            if normalized_pc == (int(address) & ~1):
                category = FIRMWARE_CRASH if name.lower() in FAULT_HANDLER_NAMES else MODEL_ARTIFACT
                signals.append(CrashSignal(
                    kind="fault_handler_pc" if category == FIRMWARE_CRASH else "configured_stop_point",
                    category=category,
                    confidence=HIGH,
                    severity=HIGH if category == FIRMWARE_CRASH else LOW,
                    reason=f"PC reached {name}",
                    pc=_hex(pc),
                    lr=_hex(lr),
                    sp=_hex(sp),
                    evidence={"handler": name, "handler_address": _hex(address)},
                ))

        if self.config.code_ranges and not _in_ranges(normalized_pc, self.config.code_ranges):
            category = FIRMWARE_CRASH
            confidence = MEDIUM
            if self.config.require_code_range_for_pc_crash:
                confidence = HIGH
            signals.append(CrashSignal(
                kind="pc_outside_executable_ranges",
                category=category,
                confidence=confidence,
                severity=HIGH,
                reason="PC is outside configured executable ranges",
                pc=_hex(pc),
                lr=_hex(lr),
                sp=_hex(sp),
                evidence={"code_ranges": [[_hex(start), _hex(end)] for start, end in self.config.code_ranges[:16]]},
            ))
        return signals

    def _sp_signals(
        self,
        record: Dict[str, object],
        pc: Optional[int],
        lr: Optional[int],
        sp: Optional[int],
    ) -> List[CrashSignal]:
        if sp is None or not self.config.ram_ranges:
            return []
        if _in_ranges(sp, self.config.ram_ranges):
            return []
        return [CrashSignal(
            kind="stack_pointer_out_of_ram",
            category=FIRMWARE_CRASH,
            confidence=MEDIUM,
            severity=MEDIUM,
            reason="SP is outside configured RAM ranges",
            pc=_hex(pc),
            lr=_hex(lr),
            sp=_hex(sp),
            evidence={"ram_ranges": [[_hex(start), _hex(end)] for start, end in self.config.ram_ranges[:16]]},
        )]

    def _collapse_signals(self, signals: Sequence[CrashSignal]) -> Tuple[str, str, str, str, bool, bool]:
        if not signals:
            return NORMAL, "normal_exit", LOW, LOW, False, False

        priority = {
            TOOLING_FAILURE: 7,
            MODEL_ARTIFACT: 6,
            FIRMWARE_CRASH: 5,
            HANG: 4,
            NORMAL: 1,
        }
        severity_score = {HIGH: 3, MEDIUM: 2, LOW: 1}
        confidence_score = {HIGH: 3, MEDIUM: 2, LOW: 1}
        best = max(
            signals,
            key=lambda item: (
                priority.get(item.category, 0),
                severity_score.get(item.severity, 0),
                confidence_score.get(item.confidence, 0),
            ),
        )
        category = best.category
        return (
            category,
            best.kind,
            best.confidence,
            best.severity,
            category == FIRMWARE_CRASH,
            category == HANG,
        )

    @staticmethod
    def _validation_fields(
        category: str,
        confidence: str,
        severity: str,
        source_layer: str,
    ) -> Tuple[bool, str, str]:
        if category == FIRMWARE_CRASH:
            if confidence == HIGH and severity == HIGH and source_layer == "runtime_monitor":
                return True, HIGH, "strong_runtime_candidate"
            if confidence == HIGH:
                return True, HIGH, "strong_candidate"
            return True, MEDIUM, "candidate"
        if category == HANG:
            return True, MEDIUM, "hang_candidate"
        if category == TOOLING_FAILURE:
            return True, MEDIUM, "tooling_failure_repro_required"
        if category == MODEL_ARTIFACT:
            return False, LOW, "model_artifact_not_firmware_crash"
        return False, LOW, "not_firmware_crash"

    def _bucket_key(
        self,
        *,
        category: str,
        classification: str,
        pc: Optional[int],
        lr: Optional[int],
        access_address: Optional[int],
        stop_reason: str,
        trace_tail: List[str],
    ) -> str:
        stop_token = self._stop_reason_token(stop_reason)
        tail = trace_tail[-max(0, int(self.config.bucket_trace_tail)):] if trace_tail else []
        bucket_material = {
            "category": category,
            "classification": classification,
            "pc": _hex(pc & ~1 if pc is not None else None),
            "lr": _hex(lr & ~1 if lr is not None else None),
            "access_address": _hex(access_address),
            "stop": stop_token,
            "tail": tail,
        }
        return f"{classification}_{short_hash(bucket_material)}"

    @staticmethod
    def _stop_reason_token(stop_reason: str) -> str:
        text = str(stop_reason or "").strip()
        if not text:
            return ""
        text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        text = re.sub(r"\d+", "N", text)
        return text[:160]

    @staticmethod
    def _record_summary(record: Dict[str, object]) -> Dict[str, object]:
        keys = [
            "stop_reason",
            "unique_bbs",
            "instruction_count",
            "elapsed_time",
            "intervention_count",
            "phase",
            "profile_kind",
            "object_symbol",
            "input_read_hit_count",
            "sink_call_event_count",
            "stream_input_summary_event_count",
            "stream_input_payload_write_count",
        ]
        out = {key: record.get(key) for key in keys if key in record}
        for key in (
            "last_unmapped_access",
            "mmio_access_tail",
            "memory_access_tail",
            "stream_input_summary_events",
            "watch_memory_events",
            "watch_pc_events",
        ):
            if key in record:
                value = record.get(key)
                out[key] = value[-8:] if isinstance(value, list) else value
        return out

    def _classify_trigger_source(
        self,
        record: Dict[str, object],
        *,
        category: str,
        source_layer: str,
        stop_reason: str,
        access: Dict[str, object],
    ) -> Tuple[str, str, Dict[str, object]]:
        """Classify the likely trigger surface for crash-discovery reports.

        This is intentionally evidence-based. A seed being present is not enough
        to call a crash externally triggered; the firmware must actually consume
        stream/API input or write seed material into a modeled input buffer.
        """
        process_returncode = record.get("process_returncode")
        try:
            native_process_failure = (
                source_layer == "process"
                and not bool(record.get("timed_out"))
                and int(process_returncode) < 0
                and int(process_returncode) != -999
            )
        except (TypeError, ValueError):
            native_process_failure = False
        if category == TOOLING_FAILURE or native_process_failure:
            return TRIGGER_TOOLING_RUNTIME, HIGH, {
                "reason": "native emulator/subprocess failure",
                "process_returncode": record.get("process_returncode"),
            }
        if category == MODEL_ARTIFACT:
            return TRIGGER_MODEL_ARTIFACT, HIGH, {
                "reason": "classified as model/harness artifact before trigger attribution",
                "stop_reason": stop_reason,
            }

        input_evidence = self._external_input_evidence(record)
        mmio_evidence = self._mmio_environment_evidence(record, access, stop_reason)
        irq_evidence = self._irq_environment_evidence(record, stop_reason)

        input_score = int(input_evidence.get("score", 0) or 0)
        mmio_score = int(mmio_evidence.get("score", 0) or 0)
        irq_score = int(irq_evidence.get("score", 0) or 0)
        peripheral_score = mmio_score + irq_score
        evidence = {
            "external_input": input_evidence,
            "mmio": mmio_evidence,
            "irq": irq_evidence,
        }

        if input_score > 0 and peripheral_score > 0:
            return TRIGGER_EXTERNAL_PLUS_MMIO, HIGH if input_score >= 2 else MEDIUM, evidence
        if input_score > 0:
            return TRIGGER_EXTERNAL_INPUT, HIGH if input_score >= 2 else MEDIUM, evidence
        if irq_score > 0:
            return TRIGGER_IRQ_ENVIRONMENT, HIGH if irq_score >= 2 else MEDIUM, evidence
        if mmio_score > 0:
            return TRIGGER_MMIO_ENVIRONMENT, HIGH if mmio_score >= 2 else MEDIUM, evidence
        return TRIGGER_UNKNOWN, LOW, evidence

    @staticmethod
    def _external_input_evidence(record: Dict[str, object]) -> Dict[str, object]:
        input_reads = safe_int(record.get("input_read_hit_count"))
        payload_writes = safe_int(record.get("stream_input_payload_write_count"))
        summary_count = safe_int(record.get("stream_input_summary_event_count"))
        seed_observed_events = 0
        stream_events = record.get("stream_input_summary_events") or []
        if isinstance(stream_events, list):
            for item in stream_events:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "")
                has_seed = bool(item.get("seed_hex") or item.get("seed_offset") is not None)
                if has_seed or kind.startswith("stream_"):
                    seed_observed_events += 1
        seed_present_only = bool(record.get("seed_hex") or record.get("seed_sha256"))
        score = 0
        if input_reads > 0:
            score += 3
        if payload_writes > 0:
            score += 3
        if summary_count > 0 or seed_observed_events > 0:
            score += 2
        return {
            "score": score,
            "input_read_hit_count": input_reads,
            "stream_input_payload_write_count": payload_writes,
            "stream_input_summary_event_count": summary_count,
            "seed_observed_event_count": seed_observed_events,
            "seed_present_without_consumption": seed_present_only and score == 0,
        }

    def _mmio_environment_evidence(
        self,
        record: Dict[str, object],
        access: Dict[str, object],
        stop_reason: str,
    ) -> Dict[str, object]:
        mmio_tail = record.get("mmio_access_tail") or []
        mmio_tail_count = len(mmio_tail) if isinstance(mmio_tail, list) else 0
        mmio_unmapped = False
        access_addr = parse_int(access.get("address")) if isinstance(access, dict) else None
        suspicious_pointer = _looks_like_poison_or_controlled_pointer(access_addr)
        if (
            access_addr is not None
            and _in_ranges(access_addr, self.config.mmio_ranges)
            and not suspicious_pointer
        ):
            mmio_unmapped = True
        text_hint = "mmio" in str(stop_reason or "").lower()
        score = 0
        if mmio_tail_count > 0:
            score += 2
        if mmio_unmapped:
            score += 2
        if text_hint:
            score += 1
        return {
            "score": score,
            "mmio_access_tail_count": mmio_tail_count,
            "unmapped_access_in_mmio_range": mmio_unmapped,
            "access_address_looks_like_controlled_pointer": bool(suspicious_pointer),
            "access_address": _hex(access_addr),
            "stop_reason_mentions_mmio": text_hint,
        }

    @staticmethod
    def _irq_environment_evidence(record: Dict[str, object], stop_reason: str) -> Dict[str, object]:
        phase = str(record.get("phase") or "").lower()
        text = f"{phase} {stop_reason}".lower()
        explicit_irq = any(token in text for token in ("irq", "isr", "interrupt"))
        fields = {
            key: record.get(key)
            for key in (
                "irq",
                "irq_num",
                "isr_addr",
                "learned_irq_candidate_count",
                "reservoir_interrupt_contexts",
                "reservoir_interrupt_contexts_added",
            )
            if key in record
        }
        score = 0
        if explicit_irq:
            score += 1
        if fields:
            score += 2
        return {
            "score": score,
            "phase_or_reason_mentions_irq": explicit_irq,
            "fields": fields,
        }

class CrashStore:
    """Persist crash reports using MultiFuzz/Fuzzware-like bucket directories."""

    def __init__(self, root: object):
        self.root = Path(root)
        self.crash_dir = self.root / "crashes"
        self.hang_dir = self.root / "hangs"
        self.artifact_dir = self.root / "model_artifacts"
        self.tooling_dir = self.root / "tooling_failures"
        for directory in (self.crash_dir, self.hang_dir, self.artifact_dir, self.tooling_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        report: CrashReport,
        *,
        seed_bytes: Optional[bytes] = None,
        seed_path: Optional[object] = None,
    ) -> Path:
        if report.is_crash:
            base = self.crash_dir
        elif report.is_hang:
            base = self.hang_dir
        elif report.category == MODEL_ARTIFACT:
            base = self.artifact_dir
        elif report.category == TOOLING_FAILURE:
            base = self.tooling_dir
        else:
            base = self.root / "non_crashes"
            base.mkdir(parents=True, exist_ok=True)

        bucket = base / safe_fragment(report.bucket_key, max_len=180)
        bucket.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stem = f"{timestamp}_{time.time_ns()}_{short_hash(report.to_dict(), 10)}"
        report_path = bucket / f"{stem}.json"
        if atomic_json_dump is not None:
            atomic_json_dump(report.to_dict(), report_path, indent=2, ensure_ascii=False)
        else:
            report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        if seed_bytes is not None:
            seed_destination = bucket / f"{stem}.seed"
            if atomic_write_bytes is not None:
                atomic_write_bytes(seed_destination, seed_bytes, durable=False)
            else:
                seed_destination.write_bytes(seed_bytes)
        elif seed_path:
            try:
                data = Path(seed_path).read_bytes()
                seed_destination = bucket / f"{stem}.seed"
                if atomic_write_bytes is not None:
                    atomic_write_bytes(seed_destination, data, durable=False)
                else:
                    seed_destination.write_bytes(data)
            except Exception:
                pass
        return report_path

def _phase_records(phase_name: str, phase: Dict[str, object]) -> Iterator[Dict[str, object]]:
    if isinstance(phase.get("run_result"), dict):
        record = dict(phase["run_result"])
        record.setdefault("phase", phase_name)
        yield record
    if "stop_reason" in phase:
        record = {
            "phase": phase_name,
            "stop_reason": phase.get("stop_reason"),
            "stop_reasons": phase.get("stop_reasons"),
        }
        yield record
    for list_key in ("debug_records", "candidate_debug_records", "materialized_records"):
        for item in phase.get(list_key) or []:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record.setdefault("phase", phase_name)
            if isinstance(record.get("run_result"), dict):
                merged = dict(record.get("run_result") or {})
                merged.update({key: value for key, value in record.items() if key != "run_result"})
                record = merged
            yield record


def detect_many_from_lsgemu_report(
    report_payload: Dict[str, object],
    detector: Optional[CrashDetector] = None,
    *,
    firmware: Optional[str] = None,
) -> List[CrashReport]:
    """Extract crash candidates from an LSGEmu JSON report."""
    detector = detector or CrashDetector()
    firmware = firmware or str(report_payload.get("firmware") or "")
    reports: List[CrashReport] = []
    # Historical reports used ``phases`` while the current interleaved runner
    # stores the same phase records under ``phase_metadata``.  Prefer the
    # explicit ``phases`` payload when both exist, and use metadata for phases
    # absent from it so an already-written guest crash is not silently missed.
    phases: Dict[str, Dict[str, object]] = {}
    for container_name in ("phase_metadata", "phases"):
        container = report_payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for name, phase in container.items():
            if not isinstance(phase, dict):
                continue
            merged = dict(phases.get(str(name), {}))
            merged.update(phase)
            phases[str(name)] = merged
    if not phases:
        return reports
    for phase_name, phase in phases.items():
        if not isinstance(phase, dict):
            continue
        for record in _phase_records(str(phase_name), phase):
            crash_report = detector.detect(record, firmware=firmware, metadata={"phase": str(phase_name)})
            if crash_report.category != NORMAL:
                reports.append(crash_report)
    return reports


def summarize_reports(reports: Sequence[CrashReport]) -> Dict[str, object]:
    buckets: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    classifications: Dict[str, int] = {}
    for report in reports:
        buckets[report.bucket_key] = buckets.get(report.bucket_key, 0) + 1
        categories[report.category] = categories.get(report.category, 0) + 1
        classifications[report.classification] = classifications.get(report.classification, 0) + 1
    return {
        "total": len(reports),
        "firmware_crashes": sum(1 for item in reports if item.is_crash),
        "hangs": sum(1 for item in reports if item.is_hang),
        "model_artifacts": sum(1 for item in reports if item.category == MODEL_ARTIFACT),
        "tooling_failures": sum(1 for item in reports if item.category == TOOLING_FAILURE),
        "unique_buckets": len(buckets),
        "categories": categories,
        "classifications": classifications,
        "buckets": buckets,
    }


def _load_config(path: Optional[str]) -> CrashDetectorConfig:
    if not path:
        return CrashDetectorConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CrashDetectorConfig.from_dict(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classify LSGEmu/fuzzing crash records")
    parser.add_argument("--run-result", help="single run_result JSON file")
    parser.add_argument("--lsgemu-report", help="LSGEmu interleaved report JSON")
    parser.add_argument("--config", help="crash detector config JSON")
    parser.add_argument("--out-dir", help="optional directory for crash bucket artifacts")
    args = parser.parse_args(argv)

    detector = CrashDetector(_load_config(args.config))
    reports: List[CrashReport] = []
    input_provided = bool(args.run_result or args.lsgemu_report)

    if args.run_result:
        payload = json.loads(Path(args.run_result).read_text(encoding="utf-8"))
        reports.append(detector.detect(payload))
    if args.lsgemu_report:
        payload = json.loads(Path(args.lsgemu_report).read_text(encoding="utf-8"))
        reports.extend(detect_many_from_lsgemu_report(payload, detector))
    if not input_provided:
        parser.error("provide --run-result or --lsgemu-report")

    if args.out_dir:
        store = CrashStore(args.out_dir)
        for report in reports:
            store.record(report)
    print(json.dumps({
        "summary": summarize_reports(reports),
        "reports": [report.to_dict() for report in reports],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
