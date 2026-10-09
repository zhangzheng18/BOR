#!/usr/bin/env python3
"""Minimal regression tests for crash_detector."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from crash_detector import (
    CrashDetector,
    CrashDetectorConfig,
    CrashStore,
    FIRMWARE_CRASH,
    HANG,
    MODEL_ARTIFACT,
    TOOLING_FAILURE,
    detect_many_from_lsgemu_report,
)
from runtime_crash_monitor import RuntimeCrashMonitor
from subprocess_runner import run_supervised

try:
    from unicorn import Uc, UC_ARCH_ARM, UC_MODE_MCLASS, UC_MODE_THUMB
    from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_SP
except Exception:
    Uc = None
    UC_ARCH_ARM = UC_MODE_MCLASS = UC_MODE_THUMB = None
    UC_ARM_REG_PC = UC_ARM_REG_SP = None


def main() -> int:
    detector = CrashDetector(CrashDetectorConfig(
        crash_points={0x08001234},
        fault_handler_addrs={"HardFault": 0x08000100},
        code_ranges=[(0x08000000, 0x08020000)],
        ram_ranges=[(0x20000000, 0x20010000)],
    ))

    invalid_write = detector.detect({
        "stop_reason": "Invalid memory write (UC_ERR_WRITE_UNMAPPED)",
        "registers": {"pc": "0x08001000", "lr": "0x08002000", "sp": "0x20001000"},
        "last_unmapped_access": {
            "access": "write_unmapped",
            "pc": "0x08001000",
            "address": "0x41414140",
            "size": 4,
        },
    })
    assert invalid_write.is_crash
    assert invalid_write.category == FIRMWARE_CRASH
    assert invalid_write.classification == "invalid_memory_write"
    assert invalid_write.trigger_source == "unknown"

    input_bound_crash = detector.detect({
        "stop_reason": "Invalid memory write (UC_ERR_WRITE_UNMAPPED)",
        "registers": {"pc": "0x08001000", "lr": "0x08002000", "sp": "0x20001000"},
        "input_read_hit_count": 1,
        "stream_input_payload_write_count": 1,
        "last_unmapped_access": {
            "access": "write_unmapped",
            "pc": "0x08001000",
            "address": "0x41414140",
            "size": 4,
        },
    })
    assert input_bound_crash.is_crash
    assert input_bound_crash.trigger_source == "external_input"

    hardfault = detector.detect({
        "stop_reason": "completed",
        "registers": {"pc": "0x08000101", "lr": "0xfffffff9", "sp": "0x20001000"},
    })
    assert hardfault.is_crash
    assert hardfault.classification == "fault_handler_pc"

    timeout = detector.detect({
        "stop_reason": "timeout_reached",
        "registers": {"pc": "0x08002000", "lr": "0x08001000", "sp": "0x20001000"},
    })
    assert timeout.is_hang
    assert timeout.category == HANG

    terminal_sink = detector.detect({
        "stop_reason": "fatal_sink_terminal",
        "registers": {"pc": "0x0800742c", "lr": "0x08003649", "sp": "0x20001000"},
        "memory_access_tail": [{"address": "0x42420060"}],
    })
    assert not terminal_sink.is_crash
    assert terminal_sink.category == "normal"
    assert terminal_sink.access_address is None

    model_artifact = detector.detect({
        "stop_reason": "unproven_dynamic_code_target",
        "registers": {"pc": "0x20001000", "lr": "0x08001000", "sp": "0x20001000"},
    })
    assert not model_artifact.is_crash
    assert model_artifact.category == MODEL_ARTIFACT

    unmapped_mmio = detector.detect({
        "stop_reason": "Invalid memory read (UC_ERR_READ_UNMAPPED)",
        "registers": {"pc": "0x08001000", "lr": "0x08002000", "sp": "0x20001000"},
        "last_unmapped_access": {
            "access": "read_unmapped",
            "pc": "0x08001000",
            "address": "0x40001000",
            "size": 4,
        },
    })
    assert not unmapped_mmio.is_crash
    assert unmapped_mmio.category == MODEL_ARTIFACT
    assert unmapped_mmio.classification == "unmapped_mmio_read"
    assert unmapped_mmio.trigger_source == "model_or_harness_artifact"

    mixed_crash = detector.detect({
        "stop_reason": "Invalid memory write (UC_ERR_WRITE_UNMAPPED)",
        "registers": {"pc": "0x08001000", "lr": "0x08002000", "sp": "0x20001000"},
        "input_read_hit_count": 1,
        "mmio_access_tail": [{"pc": "0x08000010", "addr": "0x40001000", "value": "0x1", "is_read": True}],
        "last_unmapped_access": {
            "access": "write_unmapped",
            "pc": "0x08001000",
            "address": "0x41414140",
            "size": 4,
        },
    })
    assert mixed_crash.trigger_source == "external_input_plus_peripheral_state"

    native = detector.detect({
        "registers": {"pc": "0x08001000", "lr": "0x08001000", "sp": "0x20001000"},
    }, returncode=-11)
    assert native.category == TOOLING_FAILURE
    assert not native.is_crash
    assert native.requires_replay_validation

    config_from_hex = CrashDetectorConfig.from_dict({
        "fault_handler_addrs": {"HardFault": "0x08000100"},
        "crash_points": ["0x08001234"],
    })
    assert config_from_hex.fault_handler_addrs["hardfault"] == 0x08000100
    assert 0x08001234 in config_from_hex.crash_points

    baseline_record = dict(invalid_write.metadata["record_summary"])
    baseline_record.update({
        "stop_reason": invalid_write.stop_reason,
        "registers": {"pc": invalid_write.pc, "lr": invalid_write.lr, "sp": invalid_write.sp},
        "last_unmapped_access": {"access": "write_unmapped", "address": invalid_write.access_address},
    })
    report = {
        "firmware": "dummy.elf",
        "phases": {
            "baseline": {"run_result": baseline_record},
            "stream": {"debug_records": [{
                "stop_reason": "timeout_reached",
                "registers": {"pc": "0x08002000", "lr": "0x08001000", "sp": "0x20001000"},
            }]},
        },
    }
    extracted = detect_many_from_lsgemu_report(report, detector)
    assert len(extracted) == 2

    with tempfile.TemporaryDirectory() as tmp:
        path = CrashStore(tmp).record(invalid_write, seed_bytes=b"AAAA")
        assert path.exists()
        assert list((Path(tmp) / "crashes").glob("**/*.seed"))

    _run_runtime_monitor_smoke()

    supervised_native = run_supervised(
        ["python3", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"],
        detector=detector,
        input_id="synthetic_sigsegv",
    )
    assert supervised_native.returncode == -11
    assert supervised_native.crash_report.category == TOOLING_FAILURE

    supervised_timeout = run_supervised(
        ["python3", "-c", "import time; time.sleep(5)"],
        timeout=0.05,
        detector=detector,
        input_id="synthetic_timeout",
    )
    assert supervised_timeout.timed_out
    assert supervised_timeout.crash_report.is_hang
    assert supervised_timeout.crash_report.category == HANG
    assert supervised_timeout.crash_report.category != TOOLING_FAILURE

    print(json.dumps({
        "invalid_write": invalid_write.to_dict(),
        "hardfault": hardfault.to_dict(),
        "timeout": timeout.to_dict(),
        "terminal_sink": terminal_sink.to_dict(),
        "model_artifact": model_artifact.to_dict(),
        "unmapped_mmio": unmapped_mmio.to_dict(),
        "input_bound_crash": input_bound_crash.to_dict(),
        "mixed_crash": mixed_crash.to_dict(),
        "native": native.to_dict(),
        "supervised_native": supervised_native.to_dict(),
        "extracted": [item.to_dict() for item in extracted],
    }, indent=2, ensure_ascii=False))
    return 0


def _run_runtime_monitor_smoke() -> None:
    if Uc is None:
        return
    config = CrashDetectorConfig(
        code_ranges=[(0x08000000, 0x08001000)],
        ram_ranges=[(0x20000000, 0x20001000)],
    )
    uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
    uc.mem_map(0x08000000, 0x1000)
    uc.mem_map(0x20000000, 0x1000)
    # Thumb:
    #   movs r0, #1
    #   lsls r0, r0, #30  ; r0 = 0x40000000
    #   str r0, [r0]      ; unmapped MMIO write, classified as model artifact
    code = bytes.fromhex("012080070060")
    uc.mem_write(0x08000000, code)
    uc.reg_write(UC_ARM_REG_SP, 0x20000800)
    uc.reg_write(UC_ARM_REG_PC, 0x08000001)
    monitor = RuntimeCrashMonitor(uc, config)
    monitor.install()
    try:
        try:
            uc.emu_start(0x08000001, 0x08000000 + len(code), count=8)
        except Exception:
            pass
        record = monitor.as_record()
    finally:
        monitor.uninstall()

    assert record["source_layer"] == "runtime_monitor"
    assert record["last_unmapped_access"]["address"] == "0x40000000"
    report = CrashDetector(config).detect(record)
    assert report.category == MODEL_ARTIFACT
    assert report.source_layer == "runtime_monitor"


if __name__ == "__main__":
    raise SystemExit(main())
