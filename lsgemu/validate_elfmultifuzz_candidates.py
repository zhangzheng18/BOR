#!/usr/bin/env python3
"""Strict validation rollup for elfmultifuzz manual-audit candidates.

This script is intentionally conservative.  It validates the manual reports
against firmware identity, static evidence, existing entry-derived LSGEmu
coverage, and local PoC seed availability.  It does not turn strings or high
coverage alone into CVE confirmations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lsgemu.deployment_config import configured_path


_MANUAL_SELECTED_ROOT = configured_path(
    "MANUAL_MCU_SELECTED_ROOT",
    Path("datasets") / "mcu_type_firmware_selected",
)
DEFAULT_AUDIT_DIR = _MANUAL_SELECTED_ROOT / "reports" / "manual_mcu_audit_elfmultifuzz"
DEFAULT_FIRMWARE_ROOT = configured_path("ELFMULTIFUZZ_ROOT", Path("datasets") / "elfmultifuzz")
DEFAULT_OUTPUT_DIR = Path("reports/elfmultifuzz_candidate_validation")
DEFAULT_LATEST_CAMPAIGN_POINTER = Path(".lsgemu_runs/latest_strict_4h_loopfix_dir.txt")

HELPER_REPORTS = {
    "README.md",
    "candidate_matrix.md",
    "validation_playbook.md",
    "breakpoint_watchpoint_plan.md",
    "high_priority_static_evidence.md",
    "cve_recheck_existing_cves.md",
    "finding_record_template.md",
}

STRICT_CONFIRMING_OBSERVATIONS = (
    "hardfault",
    "usagefault",
    "busfault",
    "memmanage",
    "invalid_write",
    "out_of_bounds",
    "firmware_region_write",
    "flash_write",
    "flash_erase",
    "unauthorized_state_change",
    "input_materialized_effect",
)

SINK_SYMBOLS = (
    "memcpy",
    "memmove",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "snprintf",
    "vsnprintf",
    "sscanf",
)

FLASH_SINK_TERMS = (
    "eraseflash",
    "writebytesflash",
    "flashfirmware",
)

SEED_DIR_RULES: Sequence[Tuple[Sequence[str], Sequence[str]]] = (
    (("xml", "expat"), ("xml_parser",)),
    (("gateway", "firmata"), ("gateway_firmata",)),
    (("drone", "msp"), ("drone_msp",)),
    (("cnc", "3dprinter", "g-code", "gcode"), ("gcode",)),
    (("heat_press", "plc", "modbus"), ("modbus_rtu",)),
    (("utasker_modbus",), ("modbus_rtu", "utasker_commands")),
    (("utasker_usb", "usb"), ("utasker_commands",)),
    (("gpstracker", "gsm", "sms"), ("gpstracker_sms_gsm",)),
    (("socketcan", "can"), ("zephyr_socketcan",)),
    (("6lowpan", "sixlowpan"), ("sixlowpan_fragments",)),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text(argv: Sequence[str], timeout: int = 20) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"<command failed: {type(exc).__name__}: {exc}>"


def normalize_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return value or "CASE"


def parse_intish(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0)
        except ValueError:
            try:
                return int(text, 16)
            except ValueError:
                return None
    return None


def read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def report_files(audit_dir: Path) -> List[Path]:
    return sorted(
        path for path in audit_dir.glob("*.md")
        if path.name not in HELPER_REPORTS
    )


def first_regex(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def parse_report_identity(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    firmware_path = first_regex(r"路径[：:]\s*`([^`]+)`", text)
    sha = first_regex(r"SHA-?256[：:]\s*`?([0-9a-fA-F]{64})`?", text)
    entry = first_regex(r"入口[：:]\s*`?(0x[0-9a-fA-F]+)`?", text)
    return firmware_path, sha.lower() if sha else None, entry


def resolve_firmware(path_text: Optional[str], firmware_root: Path) -> Optional[Path]:
    if not path_text:
        return None
    raw = Path(path_text)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(firmware_root / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def split_candidate_sections(text: str) -> List[Tuple[str, str]]:
    matches = [
        match for match in re.finditer(r"^###\s+(.+)$", text, re.MULTILINE)
        if match.group(1).strip().startswith("候选")
    ]
    sections: List[Tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end].strip()))
    return sections


def heading_title(heading: str) -> str:
    if "：" in heading:
        return heading.split("：", 1)[1].strip()
    if ":" in heading:
        return heading.split(":", 1)[1].strip()
    return heading.strip()


def code_addresses(text: str) -> List[int]:
    out: List[int] = []
    code_terms = (
        "函数",
        "入口",
        "调用",
        "断点",
        "handler",
        "callback",
        "parse",
        "parser",
        "process",
        "handler",
        "sink",
        "PC",
        "pc",
    )
    non_code_terms = ("offset", "偏移", "CVE-", "SHA", "大小")
    for line in text.splitlines():
        if any(term in line for term in non_code_terms):
            continue
        if not any(term in line for term in code_terms):
            continue
        for token in re.findall(r"0x[0-9a-fA-F]{5,10}", line):
            value = int(token, 16) & ~1
            if value >= 0x1000:
                out.append(value)
    return sorted(set(out))


def is_useful_marker(value: str) -> bool:
    marker = value.strip()
    if len(marker) < 2 or len(marker) > 96:
        return False
    if marker.lower().startswith("0x"):
        return False
    if marker.startswith("/home/") or marker.startswith("firmware/"):
        return False
    printable = sum(1 for ch in marker if ord(ch) < 128 and ch.isprintable())
    return printable >= max(2, len(marker) // 2)


def derive_markers(report_text: str, body: str) -> List[str]:
    markers: List[str] = []
    for source in (body, report_text):
        for value in re.findall(r"`([^`]+)`", source):
            if is_useful_marker(value):
                markers.append(value.strip())
    body_upper = body.upper()
    extra_profiles = {
        "FIRMATA": ["FirmataParser", "processSysexMessage", "parserBuffer", "Servo", "TwoWire"],
        "MODBUS": ["MODBUS", "fnHandleMODBUS_input", "ucModbusSerialInputBuffer", "BAD LENGTH"],
        "USB": ["USB-SW download", "usb-load", "fnUSB_handle_frame", "fnWriteBytesFlash"],
        "XML": ["XML_Parse", "XML_GetBuffer", "XmlParseXmlDecl", "Parse error"],
        "G-CODE": ["GCode", "Line overflow", "M112", "M997", "flashFirmware"],
        "GCODE": ["GCode", "Line overflow", "M112", "M997", "flashFirmware"],
        "MSP": ["MSP", "HAL_UART_RxCpltCallback", "PID"],
        "CAN": ["CAN", "socketcan", "filter", "msgq"],
        "6LOWPAN": ["6LoWPAN", "fragment", "RPL", "UDP"],
        "RIOT": ["RIOT", "GNRC", "shell", "msg_receive"],
        "CCN": ["ccnl", "CCN-lite", "prefix", "interest"],
    }
    for key, values in extra_profiles.items():
        if key in body_upper:
            markers.extend(values)
    seen = set()
    out = []
    for marker in markers:
        key = marker.lower()
        if key not in seen:
            seen.add(key)
            out.append(marker)
    return out[:80]


def scan_markers(path: Path, markers: Sequence[str]) -> Dict[str, Dict[str, object]]:
    data = path.read_bytes()
    lower = data.lower()
    hits: Dict[str, Dict[str, object]] = {}
    for marker in markers:
        try:
            needle = marker.encode("latin1")
        except UnicodeEncodeError:
            continue
        offsets: List[int] = []
        for haystack, actual_needle in ((data, needle), (lower, needle.lower())):
            start = 0
            while len(offsets) < 16:
                offset = haystack.find(actual_needle, start)
                if offset < 0:
                    break
                if offset not in offsets:
                    offsets.append(offset)
                start = offset + 1
            if offsets:
                break
        if offsets:
            first = offsets[0]
            chunk = data[max(0, first - 48): first + 96]
            context = re.sub(r"[^ -~]", ".", chunk.decode("latin1", errors="replace"))
            hits[marker] = {
                "count": len(offsets),
                "offsets": [f"0x{offset:x}" for offset in offsets],
                "first_context": context,
            }
    return hits


def collect_symbols(path: Path) -> Dict[str, int]:
    out = command_text(["arm-none-eabi-nm", "-n", str(path)], timeout=60)
    symbols: Dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
            kind = parts[1]
            if kind.lower() in {"t", "w"}:
                symbols[parts[2]] = int(parts[0], 16) & ~1
    return symbols


def scan_sink_references(path: Path) -> Dict[str, object]:
    text = command_text(["arm-none-eabi-objdump", "-dr", str(path)], timeout=90)
    sink_counts: Dict[str, int] = {}
    callsites: List[Dict[str, str]] = []
    previous_addr = ""
    previous_line = ""
    for line in text.splitlines():
        code_match = re.match(r"\s*([0-9a-fA-F]{8}):\s+[0-9a-fA-F]+\s+(.+)$", line)
        if code_match:
            previous_addr = f"0x{code_match.group(1).lower()}"
            previous_line = line.strip()
        reloc_match = re.search(r"R_ARM_(?:CALL|JUMP24|THM_CALL|THM_JUMP24)\s+([A-Za-z_][A-Za-z0-9_@.]*)", line)
        if not reloc_match:
            continue
        symbol = reloc_match.group(1)
        if symbol in SINK_SYMBOLS:
            sink_counts[symbol] = sink_counts.get(symbol, 0) + 1
            if len(callsites) < 64:
                callsites.append({
                    "address": previous_addr,
                    "symbol": symbol,
                    "instruction": previous_line,
                    "relocation": line.strip(),
                })
    return {"sink_counts": sink_counts, "callsites": callsites}


def load_campaign_summary(campaign_root: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not campaign_root:
        return {}
    csv_path = campaign_root / "campaign_summary.csv"
    if not csv_path.exists():
        return {}
    rows: Dict[str, Dict[str, object]] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            firmware = str(row.get("firmware") or "")
            if firmware:
                rows[firmware] = dict(row)
    return rows


def load_covered_bbs(report_file: Optional[str]) -> Tuple[set[int], Optional[Dict[str, object]]]:
    if not report_file:
        return set(), None
    path = Path(report_file)
    payload = read_json(path)
    if not payload:
        return set(), None
    covered = set()
    for item in payload.get("covered_bb_list") or []:
        value = parse_intish(item)
        if value is not None:
            covered.add(value & ~1)
    return covered, payload


def seed_files_for_report(report_stem: str, title: str, seed_root: Path) -> List[Path]:
    haystack = f"{report_stem} {title}".lower()
    dirs: List[str] = []
    for keys, values in SEED_DIR_RULES:
        if any(key.lower() in haystack for key in keys):
            dirs.extend(values)
    result: List[Path] = []
    for dirname in dict.fromkeys(dirs):
        directory = seed_root / dirname
        if directory.is_dir():
            result.extend(
                path for path in sorted(directory.iterdir())
                if path.is_file() and path.name != "README.md"
            )
    return result


def surface_kind(report_stem: str, title: str, body: str) -> str:
    text = f"{report_stem} {title} {body}".lower()
    if "xml" in text or "expat" in text:
        return "xml_serial_parser"
    if "firmata" in text or "sysex" in text:
        return "firmata_serial"
    if "modbus" in text:
        return "modbus"
    if "usb" in text:
        return "usb_command_or_update"
    if "g-code" in text or "gcode" in text or "3dprinter" in text or "cnc" in text:
        return "gcode"
    if "msp" in text or "drone" in text:
        return "msp_uart"
    if "can" in text or "socketcan" in text:
        return "can"
    if "6lowpan" in text or "gnrc" in text or "ccn" in text or "riot" in text:
        return "network_stack"
    if "gsm" in text or "sms" in text or "gps" in text:
        return "gsm_sms"
    if "flash" in text or "update" in text or "download" in text:
        return "update_or_flash"
    if "i2c" in text or "adc" in text or "servo" in text or "pwm" in text:
        return "peripheral_control"
    return "generic_mcu_input"


def strict_observations_from_report(payload: Optional[Dict[str, object]]) -> List[str]:
    if not payload:
        return []
    text = json.dumps(payload, ensure_ascii=False).lower()
    return [term for term in STRICT_CONFIRMING_OBSERVATIONS if term in text]


def load_seed_replay_results(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not path or not path.exists():
        return {}
    payload = read_json(path)
    if not payload:
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for item in payload.get("runs") or []:
        if not isinstance(item, dict):
            continue
        report = item.get("report")
        if not report:
            continue
        result_path = item.get("result_path")
        if result_path:
            raw_report = read_json(Path(str(result_path)))
            if raw_report:
                item = dict(item)
                item["summary_raw"] = item.get("summary")
                item["summary"] = summarize_seed_replay_report(raw_report)
        out[Path(str(report)).stem] = item
    return out


def _record_has_strong_input(record: Dict[str, object]) -> bool:
    return (
        int(record.get("input_read_hit_count") or 0) > 0
        or int(record.get("stream_input_payload_write_count") or 0) > 0
    )


def _record_has_path_progress(record: Dict[str, object]) -> bool:
    return int(record.get("new_bbs") or 0) > 0


def _record_is_seed_bound(record: Dict[str, object]) -> bool:
    return _record_has_strong_input(record) or _record_has_path_progress(record)


def _count_sink_event(symbols: Dict[str, int], event: Dict[str, object]) -> None:
    symbol = str(event.get("symbol") or "")
    if symbol:
        symbols[symbol] = int(symbols.get(symbol, 0) or 0) + 1


def _is_flash_sink_symbol(symbol: str) -> bool:
    lowered = symbol.lower()
    return any(term in lowered for term in FLASH_SINK_TERMS)


def summarize_seed_replay_report(report: Dict[str, object]) -> Dict[str, object]:
    """Recompute seed replay evidence with strict input-to-sink binding.

    Older summaries treated stream status events and new-BB-only progress as
    input materialization. Those are useful exploration signals, but strict
    materialization requires a real input read or payload write.
    """
    stream_phases = {
        name: phase
        for name, phase in (report.get("phase_metadata") or {}).items()
        if isinstance(phase, dict) and "stream_input" in name
    }
    summary: Dict[str, object] = {
        "covered_bbs": report.get("covered_bbs"),
        "total_bbs": report.get("total_bbs"),
        "coverage_rate": report.get("coverage_rate"),
        "valid_covered_bbs": report.get("valid_covered_bbs"),
        "valid_total_bbs": report.get("valid_total_bbs"),
        "valid_coverage_rate": report.get("valid_coverage_rate"),
        "strict_real_entry_replayable": report.get("strict_real_entry_replayable"),
        "stream_phase_count": len(stream_phases),
        "stream_profiles_discovered": 0,
        "stream_tasks_run": 0,
        "stream_tasks_injected": 0,
        "stream_tasks_with_new_bbs": 0,
        "stream_input_read_hit_count": 0,
        "stream_summary_event_count": 0,
        "stream_payload_write_count": 0,
        "stream_strong_input_record_count": 0,
        "stream_path_progress_record_count": 0,
        "stream_seed_bound_record_count": 0,
        "stream_debug_records_with_bb_lists": 0,
        "stream_materialized_records_with_bb_lists": 0,
        "stream_seed_new_bbs": {},
        "stream_stop_reasons": {},
        "stream_sink_call_event_count": 0,
        "stream_sink_call_symbols": {},
        "stream_flash_sink_call_symbols": {},
        "stream_seed_bound_sink_call_event_count": 0,
        "stream_seed_bound_sink_call_symbols": {},
        "stream_seed_bound_flash_sink_call_symbols": {},
        "stream_path_progress_sink_call_event_count": 0,
        "stream_path_progress_sink_call_symbols": {},
        "stream_strong_input_sink_call_event_count": 0,
        "stream_strong_input_sink_call_symbols": {},
        "stream_strong_input_flash_sink_call_symbols": {},
        "seed_bound_flash_examples": [],
        "strict_evidence_policy": (
            "input_materialized requires a real input read or payload write; "
            "seed-specific new BBs are recorded as path progress only."
        ),
    }
    seed_new_bbs: Dict[str, int] = {}
    stop_reasons: Dict[str, int] = {}
    stream_symbols = summary["stream_sink_call_symbols"]
    stream_flash_symbols = summary["stream_flash_sink_call_symbols"]
    seed_bound_symbols = summary["stream_seed_bound_sink_call_symbols"]
    seed_bound_flash_symbols = summary["stream_seed_bound_flash_sink_call_symbols"]
    path_progress_symbols = summary["stream_path_progress_sink_call_symbols"]
    strong_symbols = summary["stream_strong_input_sink_call_symbols"]
    strong_flash_symbols = summary["stream_strong_input_flash_sink_call_symbols"]
    examples = summary["seed_bound_flash_examples"]

    for phase in stream_phases.values():
        summary["stream_profiles_discovered"] = int(summary["stream_profiles_discovered"]) + int(phase.get("profiles_discovered") or 0)
        summary["stream_tasks_run"] = int(summary["stream_tasks_run"]) + int(phase.get("tasks_run") or 0)
        summary["stream_tasks_injected"] = int(summary["stream_tasks_injected"]) + int(phase.get("tasks_injected") or 0)
        summary["stream_tasks_with_new_bbs"] = int(summary["stream_tasks_with_new_bbs"]) + int(phase.get("tasks_with_new_bbs") or 0)
        for record in phase.get("debug_records") or []:
            if not isinstance(record, dict):
                continue
            summary["stream_input_read_hit_count"] = int(summary["stream_input_read_hit_count"]) + int(record.get("input_read_hit_count") or 0)
            summary["stream_summary_event_count"] = int(summary["stream_summary_event_count"]) + int(record.get("stream_input_summary_event_count") or 0)
            summary["stream_payload_write_count"] = int(summary["stream_payload_write_count"]) + int(record.get("stream_input_payload_write_count") or 0)
            if isinstance(record.get("covered_bb_list"), list) or isinstance(record.get("new_bb_list"), list):
                summary["stream_debug_records_with_bb_lists"] = int(summary["stream_debug_records_with_bb_lists"]) + 1
            strong_input_record = _record_has_strong_input(record)
            path_progress_record = _record_has_path_progress(record)
            seed_bound_record = _record_is_seed_bound(record)
            if strong_input_record:
                summary["stream_strong_input_record_count"] = int(summary["stream_strong_input_record_count"]) + 1
            if path_progress_record:
                summary["stream_path_progress_record_count"] = int(summary["stream_path_progress_record_count"]) + 1
            if seed_bound_record:
                summary["stream_seed_bound_record_count"] = int(summary["stream_seed_bound_record_count"]) + 1
                if isinstance(record.get("covered_bb_list"), list) or isinstance(record.get("new_bb_list"), list):
                    summary["stream_materialized_records_with_bb_lists"] = int(summary["stream_materialized_records_with_bb_lists"]) + 1
            seed_hex = str(record.get("seed_hex") or "")
            if seed_hex and int(record.get("new_bbs") or 0) > 0:
                seed_new_bbs[seed_hex] = max(seed_new_bbs.get(seed_hex, 0), int(record.get("new_bbs") or 0))
            reason = str(record.get("stop_reason") or "unknown")
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
            sink_events = [event for event in (record.get("sink_call_events") or []) if isinstance(event, dict)]
            summary["stream_sink_call_event_count"] = int(summary["stream_sink_call_event_count"]) + len(sink_events)
            if seed_bound_record:
                summary["stream_seed_bound_sink_call_event_count"] = int(summary["stream_seed_bound_sink_call_event_count"]) + len(sink_events)
            if path_progress_record:
                summary["stream_path_progress_sink_call_event_count"] = int(summary["stream_path_progress_sink_call_event_count"]) + len(sink_events)
            if strong_input_record:
                summary["stream_strong_input_sink_call_event_count"] = int(summary["stream_strong_input_sink_call_event_count"]) + len(sink_events)
            for event in sink_events:
                symbol = str(event.get("symbol") or "")
                if not symbol:
                    continue
                _count_sink_event(stream_symbols, event)
                if _is_flash_sink_symbol(symbol):
                    _count_sink_event(stream_flash_symbols, event)
                if seed_bound_record:
                    _count_sink_event(seed_bound_symbols, event)
                    if _is_flash_sink_symbol(symbol):
                        _count_sink_event(seed_bound_flash_symbols, event)
                        if isinstance(examples, list) and len(examples) < 16:
                            seed_preview = ""
                            if seed_hex:
                                try:
                                    seed_preview = bytes.fromhex(seed_hex[:160]).decode("ascii", errors="replace")[:120]
                                except Exception:
                                    seed_preview = seed_hex[:80]
                            examples.append(
                                {
                                    "symbol": symbol,
                                    "pc": event.get("pc"),
                                    "context_function": record.get("context_function"),
                                    "context_bb": record.get("context_bb"),
                                    "seed_preview": seed_preview,
                                    "r0": event.get("r0"),
                                    "r1": event.get("r1"),
                                    "r2": event.get("r2"),
                                    "r3": event.get("r3"),
                                }
                            )
                if path_progress_record:
                    _count_sink_event(path_progress_symbols, event)
                if strong_input_record:
                    _count_sink_event(strong_symbols, event)
                    if _is_flash_sink_symbol(symbol):
                        _count_sink_event(strong_flash_symbols, event)
        for seed_hex, value in (phase.get("seed_hits") or {}).items():
            try:
                seed_new_bbs[str(seed_hex)] = max(seed_new_bbs.get(str(seed_hex), 0), int(value or 0))
            except Exception:
                pass

    summary["stream_seed_new_bbs"] = seed_new_bbs
    summary["stream_stop_reasons"] = stop_reasons
    summary["input_materialized_weak"] = (
        int(summary["stream_summary_event_count"]) > 0
        or int(summary["stream_input_read_hit_count"]) > 0
        or int(summary["stream_payload_write_count"]) > 0
    )
    summary["input_path_progress"] = int(summary["stream_path_progress_record_count"]) > 0
    summary["input_path_progress_only"] = (
        bool(summary["input_path_progress"])
        and int(summary["stream_strong_input_record_count"]) <= 0
    )
    summary["input_materialized"] = int(summary["stream_strong_input_record_count"]) > 0
    return summary


def candidate_verdict(
    *,
    identity_ok: bool,
    marker_hits: Dict[str, object],
    candidate_targets: Sequence[int],
    report_targets: Sequence[int],
    covered_bbs: set[int],
    campaign_row: Optional[Dict[str, object]],
    strict_observations: Sequence[str],
    seed_count: int,
    seed_replay: Optional[Dict[str, object]] = None,
) -> Tuple[str, bool, str]:
    if not identity_ok:
        return "tooling_or_runtime_failure", False, "固件身份未解析或 SHA 不匹配，不能进行严格确认。"
    if strict_observations:
        return (
            "reachable_bug_needs_impact",
            False,
            "仿真报告中出现疑似故障/写入关键词，但缺少候选输入到根因的逐字节控制证明，暂不计严格 CVE。",
        )
    seed_summary = (seed_replay or {}).get("summary") or {}
    if bool(seed_summary.get("input_materialized")):
        new_tasks = int(seed_summary.get("stream_tasks_with_new_bbs") or 0)
        return (
            "reachable_bug_needs_impact",
            False,
            f"PoC seed 已被真实读取或写入业务输入路径，新增路径任务数为 {new_tasks}；但未观察到 crash、越界、未授权 Flash 写或安全状态改变，暂不计严格 CVE。",
        )
    if bool(seed_summary.get("input_path_progress_only")):
        new_tasks = int(seed_summary.get("stream_tasks_with_new_bbs") or 0)
        return (
            "reachable_bug_needs_impact",
            False,
            f"PoC replay 产生路径进展或 sink 命中，新增路径任务数为 {new_tasks}；但没有同任务真实输入读取/payload 写入，不能作为严格输入到 sink 证据。",
        )
    if bool(seed_summary.get("input_materialized_weak")):
        return (
            "reachable_bug_needs_impact",
            False,
            "PoC replay 阶段运行并出现弱 stream 摘要事件，但没有真实输入读取、payload 写入或 seed 特异性新增路径；不能把 sink 命中归因到该输入。",
        )
    target_hit = any((addr & ~1) in covered_bbs for addr in candidate_targets)
    report_hit = any((addr & ~1) in covered_bbs for addr in report_targets)
    valid_cov = float((campaign_row or {}).get("valid_coverage_rate") or 0.0)
    if target_hit:
        return "reachable_bug_needs_impact", False, "候选段落中的目标地址已被入口派生仿真覆盖，但未观察到可控输入导致 crash/越界/状态改变。"
    if report_hit:
        return "reachable_bug_needs_impact", False, "同报告的协议/解析器地址已被覆盖；该具体候选仍缺少输入物化和影响证据。"
    if marker_hits and valid_cov >= 50.0:
        return "reachable_bug_needs_impact", False, "候选静态证据存在且固件入口覆盖较高，但目标路径/影响未被严格触发。"
    if marker_hits:
        return "unreachable_under_current_model", False, "候选静态证据存在，但当前入口派生覆盖未证明候选路径语义可达。"
    if seed_count:
        return "unreachable_under_current_model", False, "有 PoC seed，但静态候选标记未确认，需先定位真实 parser/handler。"
    return "unreachable_under_current_model", False, "当前没有足够静态或动态证据确认该候选。"


def seed_sink_symbols(seed_replay: Optional[Dict[str, object]]) -> Dict[str, int]:
    summary = (seed_replay or {}).get("summary") or {}
    raw = summary.get("stream_sink_call_symbols")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value or 0)
        except Exception:
            continue
    return out


def seed_bound_sink_symbols(seed_replay: Optional[Dict[str, object]]) -> Dict[str, int]:
    summary = (seed_replay or {}).get("summary") or {}
    raw = summary.get("stream_seed_bound_sink_call_symbols")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value or 0)
        except Exception:
            continue
    return out


def seed_bound_flash_sink_symbols(seed_replay: Optional[Dict[str, object]]) -> Dict[str, int]:
    summary = (seed_replay or {}).get("summary") or {}
    raw = summary.get("stream_seed_bound_flash_sink_call_symbols")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value or 0)
        except Exception:
            continue
    return out


def strong_input_sink_symbols(seed_replay: Optional[Dict[str, object]]) -> Dict[str, int]:
    summary = (seed_replay or {}).get("summary") or {}
    raw = summary.get("stream_strong_input_sink_call_symbols")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value or 0)
        except Exception:
            continue
    return out


def strong_input_flash_sink_symbols(seed_replay: Optional[Dict[str, object]]) -> Dict[str, int]:
    summary = (seed_replay or {}).get("summary") or {}
    raw = summary.get("stream_strong_input_flash_sink_call_symbols")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value or 0)
        except Exception:
            continue
    return out


def has_flash_effect(seed_replay: Optional[Dict[str, object]]) -> bool:
    symbols = strong_input_flash_sink_symbols(seed_replay)
    for symbol, count in symbols.items():
        if count > 0 and _is_flash_sink_symbol(symbol):
            return True
    return False


def has_stage_flash_effect(seed_replay: Optional[Dict[str, object]]) -> bool:
    symbols = seed_sink_symbols(seed_replay)
    for symbol, count in symbols.items():
        if count > 0 and _is_flash_sink_symbol(symbol):
            return True
    return False


def flash_candidate_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "flash",
            "storage erase",
            "update",
            "download",
            "固件",
            "更新",
            "软件下载",
            "未认证",
            "未授权",
        )
    )


def build_results(args: argparse.Namespace) -> Dict[str, object]:
    audit_dir = Path(args.audit_dir).resolve()
    firmware_root = Path(args.firmware_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    seed_root = audit_dir / "poc_seeds"
    campaign_root = Path(args.campaign_root).resolve() if args.campaign_root else None
    campaign_rows = load_campaign_summary(campaign_root)
    seed_replay_path = Path(args.seed_replay_results).resolve() if args.seed_replay_results else None
    seed_replay_by_report = load_seed_replay_results(seed_replay_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    firmware_cache: Dict[Path, Dict[str, object]] = {}
    case_results: List[Dict[str, object]] = []
    candidate_results: List[Dict[str, object]] = []

    for report in report_files(audit_dir):
        text = report.read_text(encoding="utf-8", errors="replace")
        title = first_regex(r"^#\s+(.+)$", text, re.MULTILINE) or report.stem
        firmware_text, report_sha, report_entry = parse_report_identity(text)
        firmware = resolve_firmware(firmware_text, firmware_root)
        firmware_info: Optional[Dict[str, object]] = None
        identity_status = "unresolved"
        covered_bbs: set[int] = set()
        runtime_payload = None
        campaign_row = None
        report_targets = code_addresses(text)
        if firmware:
            if firmware not in firmware_cache:
                digest = sha256_file(firmware)
                symbols = collect_symbols(firmware)
                sinks = scan_sink_references(firmware)
                row = campaign_rows.get(str(firmware))
                covered, payload = load_covered_bbs(row.get("report_file") if row else None)
                firmware_cache[firmware] = {
                    "path": str(firmware),
                    "relpath": relative_or_absolute(firmware, firmware_root),
                    "sha256": digest,
                    "file_type": command_text(["file", "-b", str(firmware)], timeout=5),
                    "symbols": symbols,
                    "sink_references": sinks,
                    "campaign_row": row,
                    "covered_bbs": covered,
                    "runtime_payload": payload,
                }
            cached = firmware_cache[firmware]
            digest = str(cached["sha256"])
            if report_sha:
                identity_status = "sha256_match" if digest == report_sha else "sha256_mismatch"
            else:
                identity_status = "path_resolved_no_report_hash"
            campaign_row = cached.get("campaign_row")
            covered_bbs = set(cached.get("covered_bbs") or set())
            runtime_payload = cached.get("runtime_payload")
            firmware_info = {
                "path": str(firmware),
                "relpath": cached["relpath"],
                "sha256": digest,
                "report_sha256": report_sha,
                "identity_status": identity_status,
                "file_type": cached["file_type"],
                "entry": report_entry,
            }
        seed_files = seed_files_for_report(report.stem, title, seed_root)
        seed_records = [
            {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
            }
            for path in seed_files
        ]
        strict_observations = strict_observations_from_report(runtime_payload)
        seed_replay = seed_replay_by_report.get(report.stem)
        case_item = {
            "report": str(report),
            "report_stem": report.stem,
            "title": title,
            "firmware": firmware_info,
            "candidate_sections": len(split_candidate_sections(text)),
            "report_target_addresses": [f"0x{addr:08x}" for addr in report_targets],
            "report_target_covered": [f"0x{addr:08x}" for addr in report_targets if addr in covered_bbs],
            "seed_files": seed_records,
            "campaign": {
                "root": str(campaign_root) if campaign_root else None,
                "status": (campaign_row or {}).get("status"),
                "report_file": (campaign_row or {}).get("report_file"),
                "valid_covered_bbs": parse_intish((campaign_row or {}).get("valid_covered_bbs")),
                "valid_total_bbs": parse_intish((campaign_row or {}).get("valid_total_bbs")),
                "valid_coverage_rate": float((campaign_row or {}).get("valid_coverage_rate") or 0.0),
                "entry_plus_vector_reachable_coverage_rate": float((campaign_row or {}).get("valid_entry_plus_vector_reachable_coverage_rate") or 0.0),
                "strict_real_entry_replayable": str((campaign_row or {}).get("strict_real_entry_replayable")).lower() == "true",
            },
            "strict_observations": strict_observations,
            "seed_replay": seed_replay,
        }
        case_results.append(case_item)

        for index, (heading, body) in enumerate(split_candidate_sections(text), start=1):
            candidate_id = f"ELFMF-{normalize_id(report.stem)}-C{index:02d}"
            markers = derive_markers(text, body)
            marker_hits = scan_markers(firmware, markers) if firmware else {}
            candidate_targets = code_addresses(body)
            classification, strict, interpretation = candidate_verdict(
                identity_ok=identity_status in {"sha256_match", "path_resolved_no_report_hash"},
                marker_hits=marker_hits,
                candidate_targets=candidate_targets,
                report_targets=report_targets,
                covered_bbs=covered_bbs,
                campaign_row=campaign_row,
                strict_observations=strict_observations,
                seed_count=len(seed_records),
                seed_replay=seed_replay,
            )
            seed_summary = (seed_replay or {}).get("summary") or {}
            sink_symbols = seed_sink_symbols(seed_replay)
            bound_sink_symbols = seed_bound_sink_symbols(seed_replay)
            bound_flash_symbols = seed_bound_flash_sink_symbols(seed_replay)
            strong_sink_symbols = strong_input_sink_symbols(seed_replay)
            strong_flash_symbols = strong_input_flash_sink_symbols(seed_replay)
            stage_flash_effect = has_stage_flash_effect(seed_replay)
            flash_effect = bool(seed_summary.get("input_materialized")) and has_flash_effect(seed_replay)
            if flash_effect and flash_candidate_text(heading_title(heading)):
                classification = "confirmed_security_weakness"
                strict = False
                interpretation = (
                    "PoC seed 与同一入口派生 replay task 中的 Flash 擦写函数命中存在严格绑定 "
                    f"{strong_flash_symbols}；这是强安全影响证据，但仍需补齐认证边界和真实 Flash 副作用证据后再计严格 CVE。"
                )
            elif stage_flash_effect and flash_candidate_text(heading_title(heading)):
                interpretation = (
                    f"{interpretation} replay 阶段曾命中 Flash sink {sink_symbols}，"
                    "但没有同一 task 的强输入绑定证据，因此仅作为背景现象记录，不能升级为确认弱点。"
                )
            candidate_results.append({
                "id": candidate_id,
                "heading": heading,
                "title": heading_title(heading),
                "report": str(report),
                "report_stem": report.stem,
                "surface_kind": surface_kind(report.stem, title, body),
                "firmware": firmware_info,
                "identity_status": identity_status,
                "classification": classification,
                "strict_cve_confirmed": strict,
                "candidate_target_addresses": [f"0x{addr:08x}" for addr in candidate_targets],
                "candidate_target_covered": [f"0x{addr:08x}" for addr in candidate_targets if addr in covered_bbs],
                "report_target_covered": case_item["report_target_covered"],
                "markers_requested": markers,
                "marker_hits": marker_hits,
                "seed_files": seed_records,
                "campaign": case_item["campaign"],
                "strict_observations": strict_observations,
                "seed_replay": seed_replay,
                "dynamic_validated": bool(seed_summary.get("input_materialized")),
                "dynamic_sink_call_symbols": sink_symbols,
                "dynamic_seed_bound_sink_call_symbols": bound_sink_symbols,
                "dynamic_seed_bound_flash_sink_call_symbols": bound_flash_symbols,
                "dynamic_strong_input_sink_call_symbols": strong_sink_symbols,
                "dynamic_strong_input_flash_sink_call_symbols": strong_flash_symbols,
                "dynamic_flash_operation_observed": flash_effect,
                "dynamic_stage_flash_operation_observed": stage_flash_effect,
                "dynamic_weak_input_observed": bool(seed_summary.get("input_materialized_weak")),
                "dynamic_path_progress_observed": bool(seed_summary.get("input_path_progress")),
                "dynamic_path_progress_only": bool(seed_summary.get("input_path_progress_only")),
                "interpretation": interpretation,
                "body_excerpt": re.sub(r"\s+", " ", body)[:600],
            })

    results = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "audit_dir": str(audit_dir),
        "firmware_root": str(firmware_root),
        "campaign_root": str(campaign_root) if campaign_root else None,
        "seed_replay_results": str(seed_replay_path) if seed_replay_path else None,
        "report_count": len(case_results),
        "firmware_count": len({(item.get("firmware") or {}).get("path") for item in case_results if item.get("firmware")}),
        "candidate_count": len(candidate_results),
        "strict_cve_confirmed_count": sum(1 for item in candidate_results if item["strict_cve_confirmed"]),
        "classification_counts": count_by(candidate_results, "classification"),
        "surface_counts": count_by(candidate_results, "surface_kind"),
        "cases": case_results,
        "candidates": candidate_results,
    }
    return results


def count_by(items: Iterable[Dict[str, object]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_markdown(results: Dict[str, object]) -> str:
    lines = [
        "# elfmultifuzz Candidate Strict Validation",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Audit dir: `{results['audit_dir']}`",
        f"- Firmware root: `{results['firmware_root']}`",
        f"- Campaign root: `{results['campaign_root']}`",
        f"- Seed replay results: `{results.get('seed_replay_results')}`",
        f"- Reports: `{results['report_count']}`",
        f"- Firmwares: `{results['firmware_count']}`",
        f"- Candidates: `{results['candidate_count']}`",
        f"- Strict confirmed CVEs: `{results['strict_cve_confirmed_count']}`",
        "",
        "## Policy",
        "",
        "- Static strings, symbols, known attack surfaces, and coverage are evidence, but not strict CVE confirmation by themselves.",
        "- A strict confirmation requires reachable firmware execution plus attacker-controlled input materialization plus a concrete impact such as crash, out-of-bounds access, unauthorized flash write, or unsafe state change.",
        "- Existing campaign coverage is entry-derived LSGEmu evidence; PoC seed presence only means a candidate has local test inputs, not that the seed was consumed.",
        "",
        "## Summary",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for key, value in results["classification_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "| Surface | Count |", "| --- | ---: |"])
    for key, value in results["surface_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "## Firmware Cases",
        "",
        "| Report | Firmware | Valid BB Coverage | Entry+Vector Reachable Coverage | Seeds | Seed Replay | Target Hits |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ])
    for item in results["cases"]:
        fw = item.get("firmware") or {}
        campaign = item.get("campaign") or {}
        seed_replay = item.get("seed_replay") or {}
        seed_summary = seed_replay.get("summary") or {}
        seed_replay_text = (
            f"{int(seed_summary.get('stream_tasks_run') or 0)} tasks, "
            f"weak={bool(seed_summary.get('input_materialized_weak'))}, "
            f"path={bool(seed_summary.get('input_path_progress'))}, "
            f"strict={bool(seed_summary.get('input_materialized'))}"
            if seed_replay else "<none>"
        )
        lines.append(
            f"| `{item['report_stem']}` | `{fw.get('relpath', '<unresolved>')}` | "
            f"{int(campaign.get('valid_covered_bbs') or 0)} / {int(campaign.get('valid_total_bbs') or 0)} "
            f"({float(campaign.get('valid_coverage_rate') or 0.0):.2f}%) | "
            f"{float(campaign.get('entry_plus_vector_reachable_coverage_rate') or 0.0):.2f}% | "
            f"{len(item.get('seed_files') or [])} | {seed_replay_text} | {len(item.get('report_target_covered') or [])} |"
        )

    lines.extend([
        "",
        "## Candidate Results",
        "",
        "| ID | Candidate | Surface | Classification | Strict | Weak Stream | Strict Input | Stage Flash | Bound Flash | Marker Hits | Target Hits | Interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ])
    for item in results["candidates"]:
        marker_hits = len(item.get("marker_hits") or {})
        target_hits = len(item.get("candidate_target_covered") or []) or len(item.get("report_target_covered") or [])
        interp = str(item.get("interpretation") or "").replace("|", "\\|")
        title = str(item.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| `{item['id']}` | {title} | `{item['surface_kind']}` | `{item['classification']}` | "
            f"`{item['strict_cve_confirmed']}` | `{item.get('dynamic_weak_input_observed')}` | "
            f"`{item.get('dynamic_validated')}` | `{item.get('dynamic_stage_flash_operation_observed')}` | "
            f"`{bool(item.get('dynamic_seed_bound_flash_sink_call_symbols'))}` | "
            f"{marker_hits} | {target_hits} | {interp} |"
        )

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /opt/artifact",
        "source /opt/artifact/.virtualenvs/fuzzware/bin/activate",
        "PYTHONPATH=/opt/artifact:/opt/artifact/artifact/unicorn/bindings/python \\",
        "LIBUNICORN_PATH=/opt/artifact/artifact/unicorn/build \\",
        "LD_LIBRARY_PATH=/opt/artifact/artifact/unicorn/build \\",
        "python3 lsgemu/validate_elfmultifuzz_candidates.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def default_campaign_root() -> Optional[str]:
    pointer = DEFAULT_LATEST_CAMPAIGN_POINTER
    if pointer.exists():
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
        if text and Path(text).exists():
            return text
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--firmware-root", default=str(DEFAULT_FIRMWARE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--campaign-root", default=default_campaign_root())
    parser.add_argument(
        "--seed-replay-results",
        default=str(DEFAULT_OUTPUT_DIR / "seed_replay_results.json"),
        help="Optional focused seed replay result JSON to merge into candidate verdicts.",
    )
    args = parser.parse_args()

    results = build_results(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "elfmultifuzz_candidate_validation_results.json"
    md_path = output_dir / "ELFMULTIFUZZ_CANDIDATE_VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(results) + "\n", encoding="utf-8")
    print(json.dumps({
        "report_count": results["report_count"],
        "firmware_count": results["firmware_count"],
        "candidate_count": results["candidate_count"],
        "strict_cve_confirmed_count": results["strict_cve_confirmed_count"],
        "classification_counts": results["classification_counts"],
        "output_json": str(json_path),
        "output_md": str(md_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
