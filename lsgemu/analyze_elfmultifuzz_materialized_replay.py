#!/usr/bin/env python3
"""Deep-dive materialized seed replay evidence for elfmultifuzz candidates."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_VALIDATION_JSON = Path("reports/elfmultifuzz_candidate_validation/elfmultifuzz_candidate_validation_results.json")
DEFAULT_SEED_REPLAY_JSON = Path("reports/elfmultifuzz_candidate_validation/seed_replay_results.json")
DEFAULT_OUTPUT_DIR = Path("reports/elfmultifuzz_candidate_validation")


CRITICAL_TERMS: Dict[str, Sequence[str]] = {
    "p2im_heat_press": (
        "Modbus::poll",
        "Modbus::getRxBuffer",
        "Modbus::validateAnswer",
        "Modbus::get_FC3",
        "digitalWrite",
        "strcpy",
        "memcpy",
    ),
    "p2im_plc": (
        "Modbus::validateRequest",
        "Modbus::process_FC1",
        "Modbus::process_FC3",
        "Modbus::process_FC5",
        "Modbus::process_FC6",
        "Modbus::process_FC15",
        "Modbus::process_FC16",
        "Modbus::poll",
        "HardwareSerial::read",
        "digitalWrite",
        "HAL_UART_RxCpltCallback",
    ),
    "uemu_3dprinter": (
        "GCodeQueue::get_serial_commands",
        "GCodeQueue::_commit_command",
        "GCodeQueue::_enqueue",
        "GCodeQueue::advance",
        "GCodeParser::parse",
        "flashFirmware",
        "strcpy",
        "sprintf",
        "memcpy",
        "USBSerial::read",
        "HardwareSerial::read",
    ),
    "uemu_utasker_modbus": (
        "fnHandleMODBUS_input",
        "fnMODBUS",
        "fnMODBUSuserFunction",
        "fnCommandInput",
        "fnDoCommand",
        "tADMINCommand",
        "fnWriteBytesFlash",
        "fnEraseFlashSector",
        "uMemcpy",
        "uStrcpy",
    ),
    "uemu_utasker_usb": (
        "fnUSB_handle_frame",
        "fnHandleUSB_command_input",
        "fnHandleMODBUS_input",
        "fnMODBUS",
        "fnCommandInput",
        "fnDoCommand",
        "tADMINCommand",
        "tUSBCommand",
        "fnWriteBytesFlash",
        "fnEraseFlashSector",
        "fnTaskUSB",
    ),
}


IMPACT_TERMS = (
    "hardfault",
    "usagefault",
    "busfault",
    "memmanage",
    "out_of_bounds",
    "invalid_write",
    "firmware_region_write",
    "flash_write",
    "flash_erase",
    "unauthorized_state_change",
)

FLASH_SINK_TERMS = (
    "eraseflash",
    "writebytesflash",
    "flashfirmware",
)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_text(argv: Sequence[str], timeout: int = 60) -> str:
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
        return completed.stdout
    except Exception as exc:
        return f"<command failed: {type(exc).__name__}: {exc}>"


def parse_intish(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except Exception:
            return None
    return None


def load_covered(report_path: Path) -> Tuple[set[int], Dict[str, object]]:
    payload = read_json(report_path)
    covered: set[int] = set()
    for item in payload.get("covered_bb_list") or []:
        value = parse_intish(item)
        if value is not None:
            covered.add(value & ~1)
    return covered, payload


def phase_map(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    raw = payload.get("phase_metadata")
    if not isinstance(raw, dict):
        raw = payload.get("phases")
    if not isinstance(raw, dict):
        return {}
    return {str(name): phase for name, phase in raw.items() if isinstance(phase, dict)}


def bb_set_from_items(items: object) -> set[int]:
    out: set[int] = set()
    if not isinstance(items, list):
        return out
    for item in items:
        value = parse_intish(item)
        if value is not None:
            out.add(value & ~1)
    return out


def bb_set_from_container(container: Dict[str, object], prefix: str) -> set[int]:
    # Historical reports may contain only a sample when the list was truncated.
    # The replay validation scripts now request untruncated lists; if only a
    # sample exists, use it as partial evidence rather than silently falling
    # back to global coverage.
    return bb_set_from_items(container.get(f"{prefix}_list")) or bb_set_from_items(
        container.get(f"{prefix}_list_sample")
    )


def is_materialized_record(record: Dict[str, object]) -> bool:
    return is_strong_input_record(record)


def is_strong_input_record(record: Dict[str, object]) -> bool:
    return (
        int(record.get("input_read_hit_count") or 0) > 0
        or int(record.get("stream_input_payload_write_count") or 0) > 0
    )


def is_seed_bound_record(record: Dict[str, object]) -> bool:
    return is_strong_input_record(record) or int(record.get("new_bbs") or 0) > 0


def is_path_progress_record(record: Dict[str, object]) -> bool:
    return int(record.get("new_bbs") or 0) > 0


def is_flash_sink_event(event: Dict[str, object]) -> bool:
    symbol = str(event.get("symbol") or "").lower()
    return any(term in symbol for term in FLASH_SINK_TERMS)


def collect_stream_evidence(payload: Dict[str, object]) -> Dict[str, object]:
    stream_phase_covered: set[int] = set()
    stream_phase_new: set[int] = set()
    stream_task_covered: set[int] = set()
    stream_task_new: set[int] = set()
    materialized_task_covered: set[int] = set()
    materialized_task_new: set[int] = set()
    stream_phase_count = 0
    stream_phases_with_lists = 0
    stream_records_with_lists = 0
    weak_input_record_count = 0
    materialized_record_count = 0
    strong_input_record_count = 0
    path_progress_record_count = 0
    materialized_records_with_lists = 0
    path_progress_records_with_lists = 0
    stream_sink_call_events: List[Dict[str, object]] = []
    materialized_sink_call_events: List[Dict[str, object]] = []
    path_progress_sink_call_events: List[Dict[str, object]] = []

    for phase_name, phase in phase_map(payload).items():
        if "stream_input" not in str(phase_name):
            continue
        stream_phase_count += 1
        phase_covered = bb_set_from_container(phase, "covered_bb")
        phase_new = bb_set_from_container(phase, "new_bb")
        if phase_covered or phase_new:
            stream_phases_with_lists += 1
        stream_phase_covered.update(phase_covered)
        stream_phase_new.update(phase_new)
        for record in phase.get("debug_records") or []:
            if not isinstance(record, dict):
                continue
            record_covered = bb_set_from_container(record, "covered_bb")
            record_new = bb_set_from_container(record, "new_bb")
            if record_covered or record_new:
                stream_records_with_lists += 1
            if int(record.get("stream_input_summary_event_count") or 0) > 0:
                weak_input_record_count += 1
            for event in record.get("sink_call_events") or []:
                if isinstance(event, dict):
                    copied = dict(event)
                    copied.setdefault("phase", phase_name)
                    copied.setdefault("context_function", record.get("context_function"))
                    seed_hex = str(record.get("seed_hex") or "")
                    copied.setdefault("seed_hex_prefix", seed_hex[:160])
                    if seed_hex:
                        try:
                            copied.setdefault(
                                "seed_ascii_preview",
                                bytes.fromhex(seed_hex[:160]).decode("ascii", errors="replace")[:120],
                            )
                        except Exception:
                            copied.setdefault("seed_ascii_preview", seed_hex[:80])
                    stream_sink_call_events.append(copied)
            stream_task_covered.update(record_covered)
            stream_task_new.update(record_new)
            if is_strong_input_record(record):
                strong_input_record_count += 1
            if is_path_progress_record(record):
                path_progress_record_count += 1
                if record_covered or record_new:
                    path_progress_records_with_lists += 1
                for event in record.get("sink_call_events") or []:
                    if isinstance(event, dict):
                        copied = dict(event)
                        copied.setdefault("phase", phase_name)
                        copied.setdefault("context_function", record.get("context_function"))
                        seed_hex = str(record.get("seed_hex") or "")
                        copied.setdefault("seed_hex_prefix", seed_hex[:160])
                        if seed_hex:
                            try:
                                copied.setdefault(
                                    "seed_ascii_preview",
                                    bytes.fromhex(seed_hex[:160]).decode("ascii", errors="replace")[:120],
                                )
                            except Exception:
                                copied.setdefault("seed_ascii_preview", seed_hex[:80])
                        path_progress_sink_call_events.append(copied)
            if is_materialized_record(record):
                materialized_record_count += 1
                if record_covered or record_new:
                    materialized_records_with_lists += 1
                for event in record.get("sink_call_events") or []:
                    if isinstance(event, dict):
                        copied = dict(event)
                        copied.setdefault("phase", phase_name)
                        copied.setdefault("context_function", record.get("context_function"))
                        seed_hex = str(record.get("seed_hex") or "")
                        copied.setdefault("seed_hex_prefix", seed_hex[:160])
                        if seed_hex:
                            try:
                                copied.setdefault(
                                    "seed_ascii_preview",
                                    bytes.fromhex(seed_hex[:160]).decode("ascii", errors="replace")[:120],
                                )
                            except Exception:
                                copied.setdefault("seed_ascii_preview", seed_hex[:80])
                        materialized_sink_call_events.append(copied)
                materialized_task_covered.update(record_covered)
                materialized_task_new.update(record_new)

    return {
        "stream_phase_count": stream_phase_count,
        "stream_phases_with_bb_lists": stream_phases_with_lists,
        "stream_records_with_bb_lists": stream_records_with_lists,
        "weak_input_record_count": weak_input_record_count,
        "materialized_record_count": materialized_record_count,
        "strong_input_record_count": strong_input_record_count,
        "path_progress_record_count": path_progress_record_count,
        "materialized_records_with_bb_lists": materialized_records_with_lists,
        "path_progress_records_with_bb_lists": path_progress_records_with_lists,
        "stream_phase_covered": stream_phase_covered,
        "stream_phase_new": stream_phase_new,
        "stream_task_covered": stream_task_covered,
        "stream_task_new": stream_task_new,
        "materialized_task_covered": materialized_task_covered,
        "materialized_task_new": materialized_task_new,
        "stream_sink_call_events": stream_sink_call_events,
        "materialized_sink_call_events": materialized_sink_call_events,
        "path_progress_sink_call_events": path_progress_sink_call_events,
    }


def load_functions(elf: Path) -> List[Dict[str, object]]:
    output = command_text(["arm-none-eabi-nm", "-n", "-C", str(elf)], timeout=90)
    funcs: List[Dict[str, object]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        addr_text, kind, name = parts
        if kind.lower() not in {"t", "w"}:
            continue
        if not re.fullmatch(r"[0-9a-fA-F]+", addr_text):
            continue
        funcs.append({"start": int(addr_text, 16) & ~1, "name": name})
    funcs.sort(key=lambda item: int(item["start"]))
    for index, item in enumerate(funcs):
        next_start = int(funcs[index + 1]["start"]) if index + 1 < len(funcs) else int(item["start"]) + 4
        item["end"] = max(int(item["start"]) + 2, next_start)
    return funcs


def load_sized_symbols(elf: Path) -> List[Dict[str, object]]:
    output = command_text(["arm-none-eabi-nm", "-n", "-S", "-C", str(elf)], timeout=90)
    symbols: List[Dict[str, object]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        addr_text, size_text, kind, name = parts
        if not re.fullmatch(r"[0-9a-fA-F]+", addr_text) or not re.fullmatch(r"[0-9a-fA-F]+", size_text):
            continue
        size = int(size_text, 16)
        if size <= 0:
            continue
        start = int(addr_text, 16) & ~1
        symbols.append(
            {
                "start": start,
                "end": start + size,
                "size": size,
                "kind": kind,
                "name": name,
            }
        )
    symbols.sort(key=lambda item: (int(item["start"]), int(item["size"])))
    return symbols


def load_alloc_sections(elf: Path) -> List[Dict[str, object]]:
    output = command_text(["arm-none-eabi-readelf", "-SW", str(elf)], timeout=90)
    sections: List[Dict[str, object]] = []
    section_type_re = (
        r"NULL|PROGBITS|SYMTAB|STRTAB|RELA?|NOBITS|NOTE|INIT_ARRAY|FINI_ARRAY|"
        r"PREINIT_ARRAY|GROUP|SYMTAB_SHNDX|ARM_ATTRIBUTES"
    )
    pattern = re.compile(
        rf"^\s*\[\s*\d+\]\s+(.+?)\s+({section_type_re})\s+"
        r"([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+"
        r"[0-9a-fA-F]+\s+([A-Z]*)"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, section_type, addr_text, size_text, flags = match.groups()
        if "A" not in flags:
            continue
        size = int(size_text, 16)
        if size <= 0:
            continue
        start = int(addr_text, 16) & ~1
        sections.append(
            {
                "name": name.strip(),
                "type": section_type,
                "start": start,
                "end": start + size,
                "size": size,
                "flags": flags,
                "writable": "W" in flags,
                "executable": "X" in flags,
            }
        )
    sections.sort(key=lambda item: (int(item["start"]), int(item["size"])))
    return sections


def symbol_containing(symbols: Sequence[Dict[str, object]], address: int) -> Optional[Dict[str, object]]:
    matches = [
        symbol
        for symbol in symbols
        if int(symbol["start"]) <= address < int(symbol["end"])
    ]
    if not matches:
        return None
    # Prefer the smallest enclosing object, which handles nested/local symbols.
    return min(matches, key=lambda symbol: int(symbol["size"]))


def section_containing(sections: Sequence[Dict[str, object]], address: int) -> Optional[Dict[str, object]]:
    matches = [
        section
        for section in sections
        if int(section["start"]) <= address < int(section["end"])
    ]
    if not matches:
        return None
    return min(matches, key=lambda section: int(section["size"]))


def common_mcu_region(address: int) -> str:
    address &= 0xFFFFFFFF
    if 0x00000000 <= address < 0x00200000:
        return "low_alias_or_rom"
    if 0x08000000 <= address < 0x10000000:
        return "flash_or_rom"
    if 0x10000000 <= address < 0x20000000:
        return "external_or_code_region"
    if 0x20000000 <= address < 0x40000000:
        return "sram_dynamic_or_stack"
    if 0x40000000 <= address < 0x60000000:
        return "peripheral_mmio"
    if 0x60000000 <= address < 0xA0000000:
        return "external_memory"
    if 0xE0000000 <= address <= 0xE00FFFFF:
        return "cortex_m_system_control"
    return "outside_common_mcu_memory"


def _event_int(event: Dict[str, object], *keys: str) -> Optional[int]:
    for key in keys:
        value = parse_intish(event.get(key))
        if value is not None:
            return value
    return None


def copy_length_for_event(event: Dict[str, object]) -> Optional[int]:
    symbol = str(event.get("symbol") or "").lower()
    if "strcpy" in symbol:
        observed = parse_intish(event.get("src_observed_length"))
        if observed is not None:
            return observed + 1 if event.get("src_nul_seen") else observed
    if "strcat" in symbol:
        observed = parse_intish(event.get("src_observed_length"))
        if observed is not None:
            # strcat writes at the current destination terminator. Without the
            # pre-call destination length this is only a lower bound.
            return observed + 1 if event.get("src_nul_seen") else observed
    length = parse_intish(event.get("length"))
    if length is not None:
        return length
    if "memcpy" in symbol or "memmove" in symbol or "strncpy" in symbol:
        return _event_int(event, "r2")
    if "snprintf" in symbol or "vsnprintf" in symbol:
        return _event_int(event, "r1")
    return None


def _safe_seed_ascii(event: Dict[str, object]) -> str:
    seed_hex = str(event.get("seed_hex_prefix") or "")
    if not seed_hex:
        return str(event.get("seed_ascii_preview") or "")
    try:
        return bytes.fromhex(seed_hex).decode("ascii", errors="replace")
    except Exception:
        return str(event.get("seed_ascii_preview") or "")


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def enrich_seed_payload_provenance(event: Dict[str, object]) -> Dict[str, object]:
    out = dict(event)
    seed_ascii = _safe_seed_ascii(out)
    if not seed_ascii:
        return out
    best_kind = None
    best_match = 0
    best_preview = ""
    for key in ("src_preview", "format_preview"):
        preview = str(out.get(key) or "")
        if not preview:
            continue
        prefix_len = _common_prefix_len(preview, seed_ascii)
        contains_len = len(preview) if len(preview) >= 8 and preview in seed_ascii else 0
        match_len = max(prefix_len, contains_len)
        if match_len > best_match:
            best_kind = key
            best_match = match_len
            best_preview = preview
    if best_kind:
        out["seed_payload_match_field"] = best_kind
        out["seed_payload_match_length"] = best_match
        out["seed_payload_match_preview"] = best_preview[:96]
        out["source_matches_seed_payload"] = bool(best_match >= 8)
    return out


def enrich_sink_event_boundaries(
    event: Dict[str, object],
    symbols: Sequence[Dict[str, object]],
    sections: Sequence[Dict[str, object]] = (),
) -> Dict[str, object]:
    out = enrich_seed_payload_provenance(event)
    symbol_name = str(out.get("symbol") or "").lower()
    memory_terms = (
        "strcpy",
        "strncpy",
        "strcat",
        "memcpy",
        "memmove",
        "sprintf",
        "snprintf",
        "vsprintf",
        "vsnprintf",
    )
    if not any(term in symbol_name for term in memory_terms):
        return out

    dest_addr = _event_int(out, "dest", "r0")
    src_addr = _event_int(out, "src", "r1")
    copy_len = copy_length_for_event(out)
    out["copy_length_bytes"] = copy_len

    if dest_addr is not None:
        out["dest_region"] = common_mcu_region(dest_addr)
        dest_symbol = symbol_containing(symbols, dest_addr)
        if dest_symbol:
            offset = dest_addr - int(dest_symbol["start"])
            remaining = int(dest_symbol["end"]) - dest_addr
            out["dest_object"] = str(dest_symbol["name"])
            out["dest_object_start"] = f"0x{int(dest_symbol['start']):08x}"
            out["dest_object_size"] = int(dest_symbol["size"])
            out["dest_object_offset"] = offset
            out["dest_remaining_bytes"] = remaining
            if copy_len is not None:
                out["copy_fits_dest_object"] = copy_len <= remaining
                out["copy_overflow_bytes"] = max(0, copy_len - remaining)
        dest_section = section_containing(sections, dest_addr)
        if dest_section:
            section_remaining = int(dest_section["end"]) - dest_addr
            out["dest_section"] = str(dest_section["name"])
            out["dest_section_start"] = f"0x{int(dest_section['start']):08x}"
            out["dest_section_size"] = int(dest_section["size"])
            out["dest_section_flags"] = str(dest_section["flags"])
            out["dest_section_writable"] = bool(dest_section["writable"])
            out["dest_section_remaining_bytes"] = section_remaining
            if copy_len is not None:
                out["copy_fits_dest_section"] = copy_len <= section_remaining

    if src_addr is not None:
        out["src_region"] = common_mcu_region(src_addr)
        src_symbol = symbol_containing(symbols, src_addr)
        if src_symbol:
            offset = src_addr - int(src_symbol["start"])
            remaining = int(src_symbol["end"]) - src_addr
            out["src_object"] = str(src_symbol["name"])
            out["src_object_start"] = f"0x{int(src_symbol['start']):08x}"
            out["src_object_size"] = int(src_symbol["size"])
            out["src_object_offset"] = offset
            out["src_remaining_bytes"] = remaining
        src_section = section_containing(sections, src_addr)
        if src_section:
            out["src_section"] = str(src_section["name"])
            out["src_section_start"] = f"0x{int(src_section['start']):08x}"
            out["src_section_size"] = int(src_section["size"])
            out["src_section_flags"] = str(src_section["flags"])

    if out.get("dest_region") in {"outside_common_mcu_memory", "peripheral_mmio", "cortex_m_system_control"}:
        out["boundary_verdict"] = "invalid_or_model_artifact_destination"
    elif out.get("dest_section_writable") is False:
        out["boundary_verdict"] = "write_to_nonwritable_alloc_section"
    elif out.get("copy_fits_dest_section") is False and not out.get("dest_object"):
        out["boundary_verdict"] = "copy_exceeds_alloc_section"
    elif "sprintf" in symbol_name and "snprintf" not in symbol_name and copy_len is None:
        out["boundary_verdict"] = "formatted_write_length_unknown"
    elif "strcat" in symbol_name and copy_len is not None and out.get("copy_fits_dest_object") is True:
        out["boundary_verdict"] = "append_copy_lower_bound_fits_destination"
    elif "strcat" in symbol_name and copy_len is not None and out.get("copy_fits_dest_object") is False:
        out["boundary_verdict"] = "append_copy_lower_bound_exceeds_destination"
    elif out.get("copy_fits_dest_object") is True:
        out["boundary_verdict"] = "bounded_copy_fits_destination"
    elif out.get("copy_fits_dest_object") is False:
        out["boundary_verdict"] = "copy_exceeds_destination"
    elif copy_len is not None:
        out["boundary_verdict"] = "copy_length_known_destination_unknown"
    else:
        out["boundary_verdict"] = "copy_length_unknown"
    return out


def summarize_boundary_events(events: Sequence[Dict[str, object]]) -> Dict[str, object]:
    verdict_counts: Dict[str, int] = {}
    copy_events = []
    for event in events:
        verdict = event.get("boundary_verdict")
        if not verdict:
            continue
        copy_events.append(event)
        verdict_counts[str(verdict)] = int(verdict_counts.get(str(verdict), 0)) + 1

    overflow_verdicts = {
        "copy_exceeds_destination",
        "append_copy_lower_bound_exceeds_destination",
        "copy_exceeds_alloc_section",
        "write_to_nonwritable_alloc_section",
    }
    bounded_verdicts = {
        "bounded_copy_fits_destination",
    }
    lower_bound_verdicts = {
        "append_copy_lower_bound_fits_destination",
    }
    unknown_verdicts = {
        "copy_length_known_destination_unknown",
        "copy_length_unknown",
        "formatted_write_length_unknown",
    }
    artifact_verdicts = {
        "invalid_or_model_artifact_destination",
    }
    overflow_events = [
        event for event in copy_events
        if str(event.get("boundary_verdict")) in overflow_verdicts
    ]
    bounded_events = [
        event for event in copy_events
        if str(event.get("boundary_verdict")) in bounded_verdicts
    ]
    lower_bound_events = [
        event for event in copy_events
        if str(event.get("boundary_verdict")) in lower_bound_verdicts
    ]
    unknown_events = [
        event for event in copy_events
        if str(event.get("boundary_verdict")) in unknown_verdicts
    ]
    artifact_events = [
        event for event in copy_events
        if str(event.get("boundary_verdict")) in artifact_verdicts
    ]
    return {
        "copy_event_count": len(copy_events),
        "verdict_counts": verdict_counts,
        "overflow_event_count": len(overflow_events),
        "bounded_event_count": len(bounded_events),
        "append_lower_bound_fit_event_count": len(lower_bound_events),
        "unknown_event_count": len(unknown_events),
        "model_artifact_event_count": len(artifact_events),
        "all_copy_events_bounded": bool(copy_events) and len(bounded_events) == len(copy_events),
        "has_overflow_evidence": bool(overflow_events),
        "has_unknown_boundary": bool(unknown_events or lower_bound_events),
        "has_model_artifact_pointer": bool(artifact_events),
        "overflow_events": overflow_events[:8],
        "unknown_events": unknown_events[:8],
        "model_artifact_events": artifact_events[:8],
    }


def function_covered(func: Dict[str, object], covered: set[int]) -> bool:
    start = int(func["start"])
    end = int(func["end"])
    return any(start <= bb < end for bb in covered)


def matching_functions(functions: Sequence[Dict[str, object]], terms: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for term in terms:
        needle = term.lower()
        for func in functions:
            name = str(func["name"])
            if needle in name.lower() and int(func["start"]) not in seen:
                seen.add(int(func["start"]))
                out.append(dict(func))
    return out


def materialized_records(report_payload: Dict[str, object]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for phase_name, phase in phase_map(report_payload).items():
        if not isinstance(phase, dict) or "stream_input" not in str(phase_name):
            continue
        for record in phase.get("debug_records") or []:
            if not isinstance(record, dict):
                continue
            materialized = is_materialized_record(record)
            if not materialized and int(record.get("new_bbs") or 0) <= 0:
                continue
            seed_hex = str(record.get("seed_hex") or "")
            seed_preview = ""
            if seed_hex:
                try:
                    seed_preview = bytes.fromhex(seed_hex[:160]).decode("ascii", errors="replace")
                except Exception:
                    seed_preview = seed_hex[:80]
            covered_bb_list = sorted(bb_set_from_container(record, "covered_bb"))
            new_bb_list = sorted(bb_set_from_container(record, "new_bb"))
            records.append({
                "phase": phase_name,
                "profile_kind": record.get("profile_kind"),
                "object_symbol": record.get("object_symbol"),
                "context_function": record.get("context_function"),
                "seed_hex_prefix": seed_hex[:160],
                "seed_ascii_preview": seed_preview[:120],
                "new_bbs": int(record.get("new_bbs") or 0),
                "stop_reason": record.get("stop_reason"),
                "input_read_hit_count": int(record.get("input_read_hit_count") or 0),
                "stream_input_summary_event_count": int(record.get("stream_input_summary_event_count") or 0),
                "stream_input_payload_write_count": int(record.get("stream_input_payload_write_count") or 0),
                "strong_input_bound": is_strong_input_record(record),
                "seed_bound": is_seed_bound_record(record),
                "first_input_read": (record.get("input_read_hits") or [None])[0],
                "sink_call_event_count": int(record.get("sink_call_event_count") or 0),
                "sink_call_events": (record.get("sink_call_events") or [])[:8],
                "covered_bb_list_present": bool(covered_bb_list),
                "covered_bb_list_count": len(covered_bb_list),
                "new_bb_list_present": bool(new_bb_list),
                "new_bb_list_count": len(new_bb_list),
                "new_bb_list_sample": [f"0x{bb:08x}" for bb in new_bb_list[:16]],
            })
    return records


def impact_terms_present(payload: Dict[str, object]) -> List[str]:
    runtime_fragments: List[object] = []
    for _, phase in phase_map(payload).items():
        for key in (
            "run_result",
            "stop_reason",
            "stop_reasons",
            "terminal_reason",
            "crash",
            "fault",
            "invalid_access",
            "memory_access",
        ):
            if key in phase:
                runtime_fragments.append(phase.get(key))
        for record in phase.get("debug_records") or []:
            if not isinstance(record, dict):
                continue
            runtime_fragments.append({
                "stop_reason": record.get("stop_reason"),
                "input_read_hits": record.get("input_read_hits"),
                "stream_input_payload_writes": record.get("stream_input_payload_writes"),
            })
    text = json.dumps(runtime_fragments, ensure_ascii=False).lower()
    return [term for term in IMPACT_TERMS if term in text]


def sink_event_priority(event: Dict[str, object]) -> Tuple[int, str]:
    symbol = str(event.get("symbol") or "").lower()
    if "erase" in symbol or "flash" in symbol or "writebytes" in symbol:
        return (0, symbol)
    if "strcpy" in symbol or "sprintf" in symbol or "memcpy" in symbol:
        return (1, symbol)
    if "digitalwrite" in symbol:
        return (2, symbol)
    return (3, symbol)


def verdict_for_case(
    *,
    materialized: bool,
    materialized_critical: Sequence[Dict[str, object]],
    materialized_impact: Sequence[Dict[str, object]],
    stream_critical: Sequence[Dict[str, object]],
    stream_impact: Sequence[Dict[str, object]],
    impact_terms: Sequence[str],
    model_artifact_stop_count: int,
) -> Tuple[str, str]:
    if impact_terms and materialized:
        return (
            "needs_manual_fault_triage",
            "输入已物化且报告中出现故障/写入关键词，需要人工确认是否为真实候选影响而非 harness 工件。",
        )
    if materialized and materialized_impact:
        return (
            "input_materialized_impact_function_reached_no_effect",
            "输入已物化，且同一 materialized replay task 执行到潜在影响函数；但未记录越界、Flash 写入成功、认证绕过或状态改变，暂不严格确认。",
        )
    if materialized and materialized_critical:
        return (
            "input_materialized_parser_reached_no_effect",
            "输入已物化，且同一 materialized replay task 执行到 parser/协议函数，但未触发具体安全影响。",
        )
    if materialized and (stream_impact or stream_critical):
        return (
            "input_materialized_stream_stage_reached_candidate_not_same_task",
            "stream replay 阶段覆盖到候选函数且存在输入物化，但当前证据不能证明二者发生在同一个 task，不能作为严格验证。",
        )
    if materialized:
        return (
            "input_materialized_no_candidate_sink",
            "输入已物化，但当前 trace 未覆盖候选关键函数或影响函数。",
        )
    if model_artifact_stop_count:
        return (
            "model_or_harness_artifact",
            "存在 replay 映射/模型停止现象，但缺少输入物化或安全影响，不能作为漏洞证据。",
        )
    return (
        "not_materialized",
        "PoC seed 未被语义读取或写入业务输入路径。",
    )


def build_results(validation_json: Path, seed_replay_json: Path) -> Dict[str, object]:
    validation = read_json(validation_json)
    replay = read_json(seed_replay_json)
    candidates_by_stem: Dict[str, List[Dict[str, object]]] = {}
    for candidate in validation.get("candidates") or []:
        if isinstance(candidate, dict):
            candidates_by_stem.setdefault(str(candidate.get("report_stem")), []).append(candidate)

    cases: List[Dict[str, object]] = []
    for run in replay.get("runs") or []:
        if not isinstance(run, dict):
            continue
        stem = Path(str(run.get("report") or "")).stem
        result_path = Path(str(run.get("result_path") or ""))
        firmware = Path(str(run.get("firmware") or ""))
        if not result_path.exists() or not firmware.exists():
            continue
        covered, payload = load_covered(result_path)
        stream_evidence = collect_stream_evidence(payload)
        summary = dict(run.get("summary") or {})
        summary["input_materialized_raw"] = bool(summary.get("input_materialized"))
        summary["input_materialized_weak"] = bool(
            stream_evidence["weak_input_record_count"]
            or int(summary.get("stream_summary_event_count") or 0) > 0
        )
        summary["input_materialized"] = bool(stream_evidence["materialized_record_count"])
        summary["input_path_progress_record_count"] = stream_evidence["path_progress_record_count"]
        summary["input_path_progress_only"] = bool(
            stream_evidence["path_progress_record_count"]
            and not stream_evidence["materialized_record_count"]
        )
        summary["stream_seed_bound_record_count"] = (
            stream_evidence["materialized_record_count"]
            + stream_evidence["path_progress_record_count"]
        )
        summary["stream_strong_input_record_count"] = stream_evidence["strong_input_record_count"]
        if (
            not summary.get("input_materialized")
            and not summary.get("input_materialized_weak")
            and not summary.get("input_path_progress_only")
        ):
            continue
        funcs = load_functions(firmware)
        sized_symbols = load_sized_symbols(firmware)
        alloc_sections = load_alloc_sections(firmware)
        terms = CRITICAL_TERMS.get(stem, ())
        matched = matching_functions(funcs, terms)
        impact_terms = ("flash", "erase", "writebytesflash", "strcpy", "sprintf", "memcpy", "digitalwrite")
        impact_funcs = [
            func for func in funcs
            if any(term in str(func["name"]).lower() for term in impact_terms)
        ]
        globally_covered_matched = [func for func in matched if function_covered(func, covered)]
        globally_covered_impact = [func for func in impact_funcs if function_covered(func, covered)]
        stream_phase_covered_matched = [
            func for func in matched
            if function_covered(func, stream_evidence["stream_phase_covered"])
        ]
        stream_phase_new_matched = [
            func for func in matched
            if function_covered(func, stream_evidence["stream_phase_new"])
        ]
        stream_phase_covered_impact = [
            func for func in impact_funcs
            if function_covered(func, stream_evidence["stream_phase_covered"])
        ]
        stream_phase_new_impact = [
            func for func in impact_funcs
            if function_covered(func, stream_evidence["stream_phase_new"])
        ]
        materialized_task_covered_matched = [
            func for func in matched
            if function_covered(func, stream_evidence["materialized_task_covered"])
        ]
        materialized_task_new_matched = [
            func for func in matched
            if function_covered(func, stream_evidence["materialized_task_new"])
        ]
        materialized_task_covered_impact = [
            func for func in impact_funcs
            if function_covered(func, stream_evidence["materialized_task_covered"])
        ]
        materialized_task_new_impact = [
            func for func in impact_funcs
            if function_covered(func, stream_evidence["materialized_task_new"])
        ]
        records = materialized_records(payload)
        impact_hits = impact_terms_present(payload)
        model_artifacts = sum(
            1
            for record in records
            if "UC_ERR_MAP" in str(record.get("stop_reason"))
            or "Invalid memory mapping" in str(record.get("stop_reason"))
        )
        prioritized_materialized_sink_events = sorted(
            [
                enrich_sink_event_boundaries(event, sized_symbols, alloc_sections)
                for event in stream_evidence["materialized_sink_call_events"]
            ],
            key=sink_event_priority,
        )
        prioritized_path_progress_sink_events = sorted(
            [
                enrich_sink_event_boundaries(event, sized_symbols, alloc_sections)
                for event in stream_evidence["path_progress_sink_call_events"]
            ],
            key=sink_event_priority,
        )
        seed_content_bound_path_events = [
            event for event in prioritized_path_progress_sink_events
            if event.get("source_matches_seed_payload") is True
        ]
        prioritized_input_bound_sink_events = sorted(
            prioritized_materialized_sink_events + seed_content_bound_path_events,
            key=sink_event_priority,
        )
        boundary_evidence = summarize_boundary_events(prioritized_materialized_sink_events)
        path_progress_boundary_evidence = summarize_boundary_events(prioritized_path_progress_sink_events)
        input_bound_boundary_evidence = summarize_boundary_events(prioritized_input_bound_sink_events)
        classification, interpretation = verdict_for_case(
            materialized=bool(summary.get("input_materialized")),
            materialized_critical=materialized_task_covered_matched,
            materialized_impact=materialized_task_covered_impact,
            stream_critical=stream_phase_covered_matched,
            stream_impact=stream_phase_covered_impact,
            impact_terms=impact_hits,
            model_artifact_stop_count=model_artifacts,
        )
        input_bound_flash_events = [
            event for event in prioritized_input_bound_sink_events
            if sink_event_priority(event)[0] == 0
        ]
        if bool(summary.get("input_materialized")) and input_bound_boundary_evidence["has_overflow_evidence"]:
            classification = "input_materialized_memory_boundary_exceeded"
            interpretation = (
                "PoC seed 已物化，且 input-bound sink 事件的拷贝/格式化写入超过目标对象边界；"
                "这是高置信内存破坏证据，仍需结合攻击面和崩溃/状态改变整理为严格 CVE。"
            )
        elif bool(summary.get("input_materialized")) and input_bound_flash_events:
            classification = "input_materialized_flash_operation_observed"
            interpretation = (
                "PoC seed 已物化，且 input-bound sink 事件进入 Flash 擦写函数；"
                "这是强安全影响证据，但仍需把认证边界和真实硬件 Flash 副作用整理成单独 CVE 证据包。"
            )
        elif (
            bool(summary.get("input_materialized"))
            and input_bound_boundary_evidence["all_copy_events_bounded"]
            and not input_bound_flash_events
            and not any("digitalwrite" in str(event.get("symbol") or "").lower() for event in prioritized_input_bound_sink_events)
        ):
            classification = "input_materialized_bounded_copy_no_overflow"
            interpretation = (
                "PoC seed 已物化并触达拷贝类 sink；对象大小检查显示当前观测到的写入均未越过目标对象边界，"
                "因此该证据不足以确认内存破坏漏洞。"
            )
        elif summary.get("input_path_progress_only"):
            classification = "path_progress_without_strong_input_binding"
            interpretation = (
                "replay 任务产生新增 BB 或 sink 命中，但没有观察到真实输入读取或 payload 写入；"
                "该结果可用于继续探索路径，不能作为严格输入到 sink 证据。"
            )
        elif summary.get("input_materialized_weak") and not summary.get("input_materialized"):
            classification = "weak_stream_event_not_input_bound"
            interpretation = (
                "replay 阶段存在 stream 摘要事件，但没有真实输入读取、payload 写入或 seed 特异性新增路径；"
                "该阶段 sink 命中不能归因到 PoC 输入。"
            )
        cases.append({
            "report_stem": stem,
            "report": run.get("report"),
            "firmware": str(firmware),
            "result_path": str(result_path),
            "summary": summary,
            "classification": classification,
            "strict_cve_confirmed": False,
            "interpretation": interpretation,
            "coverage_evidence": {
                "stream_phase_count": stream_evidence["stream_phase_count"],
                "stream_phases_with_bb_lists": stream_evidence["stream_phases_with_bb_lists"],
                "stream_records_with_bb_lists": stream_evidence["stream_records_with_bb_lists"],
                "materialized_records_with_bb_lists": stream_evidence["materialized_records_with_bb_lists"],
                "path_progress_records_with_bb_lists": stream_evidence["path_progress_records_with_bb_lists"],
                "stream_phase_covered_bbs": len(stream_evidence["stream_phase_covered"]),
                "stream_phase_new_bbs": len(stream_evidence["stream_phase_new"]),
                "stream_task_covered_bbs": len(stream_evidence["stream_task_covered"]),
                "stream_task_new_bbs": len(stream_evidence["stream_task_new"]),
                "materialized_task_covered_bbs": len(stream_evidence["materialized_task_covered"]),
                "materialized_task_new_bbs": len(stream_evidence["materialized_task_new"]),
                "stream_sink_call_events": len(stream_evidence["stream_sink_call_events"]),
                "materialized_sink_call_events": len(stream_evidence["materialized_sink_call_events"]),
                "path_progress_sink_call_events": len(stream_evidence["path_progress_sink_call_events"]),
            },
            "boundary_evidence": boundary_evidence,
            "path_progress_boundary_evidence": path_progress_boundary_evidence,
            "input_bound_boundary_evidence": input_bound_boundary_evidence,
            "materialized_sink_call_events": prioritized_materialized_sink_events[:64],
            "path_progress_sink_call_events": prioritized_path_progress_sink_events[:64],
            "seed_content_bound_path_sink_events": seed_content_bound_path_events[:64],
            "input_bound_sink_call_events": prioritized_input_bound_sink_events[:64],
            "critical_terms": list(terms),
            "critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                    "global_covered": function_covered(func, covered),
                    "stream_phase_covered": function_covered(func, stream_evidence["stream_phase_covered"]),
                    "stream_phase_new": function_covered(func, stream_evidence["stream_phase_new"]),
                    "materialized_task_covered": function_covered(func, stream_evidence["materialized_task_covered"]),
                    "materialized_task_new": function_covered(func, stream_evidence["materialized_task_new"]),
                }
                for func in matched
            ],
            "globally_covered_critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in globally_covered_matched
            ],
            "stream_phase_covered_critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in stream_phase_covered_matched
            ],
            "stream_phase_new_critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in stream_phase_new_matched
            ],
            "materialized_task_covered_critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in materialized_task_covered_matched
            ],
            "materialized_task_new_critical_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in materialized_task_new_matched
            ],
            "globally_covered_impact_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in globally_covered_impact[:32]
            ],
            "stream_phase_covered_impact_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in stream_phase_covered_impact[:32]
            ],
            "stream_phase_new_impact_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in stream_phase_new_impact[:32]
            ],
            "materialized_task_covered_impact_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in materialized_task_covered_impact[:32]
            ],
            "materialized_task_new_impact_functions": [
                {
                    "name": str(func["name"]),
                    "start": f"0x{int(func['start']):08x}",
                    "end": f"0x{int(func['end']):08x}",
                }
                for func in materialized_task_new_impact[:32]
            ],
            "impact_terms_present": impact_hits,
            "model_artifact_stop_records": model_artifacts,
            "materialized_records": records[:32],
            "candidates": [
                {
                    "id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "classification": candidate.get("classification"),
                    "dynamic_validated": candidate.get("dynamic_validated"),
                    "strict_cve_confirmed": candidate.get("strict_cve_confirmed"),
                }
                for candidate in candidates_by_stem.get(stem, [])
            ],
        })
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "validation_json": str(validation_json),
        "seed_replay_json": str(seed_replay_json),
        "case_count": len(cases),
        "strict_cve_confirmed_count": 0,
        "cases": cases,
    }


def write_markdown(results: Dict[str, object], path: Path) -> None:
    lines = [
        "# elfmultifuzz Materialized Replay Deep Dive",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Cases: `{results['case_count']}`",
        f"- Strict confirmed CVEs: `{results['strict_cve_confirmed_count']}`",
        "",
        "## Summary",
        "",
        "| Report | Classification | Strong Input | Path Progress | Mat Critical | Mat Impact | Mat Sink Calls | Path Sink Calls | Boundary | Stream New BBs | Model Artifacts | Interpretation |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        summary = case.get("summary") or {}
        evidence = case.get("coverage_evidence") or {}
        boundary = case.get("input_bound_boundary_evidence") or case.get("boundary_evidence") or {}
        boundary_text = (
            f"copy={int(boundary.get('copy_event_count') or 0)}, "
            f"overflow={int(boundary.get('overflow_event_count') or 0)}, "
            f"unknown={int(boundary.get('unknown_event_count') or 0)}, "
            f"artifact={int(boundary.get('model_artifact_event_count') or 0)}"
        )
        interp = str(case.get("interpretation") or "").replace("|", "\\|")
        lines.append(
            f"| `{case['report_stem']}` | `{case['classification']}` | `{summary.get('input_materialized')}` | "
            f"`{summary.get('input_path_progress_record_count', 0)}` | "
            f"{len(case.get('materialized_task_covered_critical_functions') or [])} | "
            f"{len(case.get('materialized_task_covered_impact_functions') or [])} | "
            f"{int(evidence.get('materialized_sink_call_events') or 0)} | "
            f"{int(evidence.get('path_progress_sink_call_events') or 0)} | "
            f"`{boundary_text}` | "
            f"{int(evidence.get('stream_phase_new_bbs') or 0)} | "
            f"{case.get('model_artifact_stop_records', 0)} | {interp} |"
        )
    for case in results["cases"]:
        evidence = case.get("coverage_evidence") or {}
        boundary = case.get("boundary_evidence") or {}
        path_boundary = case.get("path_progress_boundary_evidence") or {}
        input_boundary = case.get("input_bound_boundary_evidence") or {}
        lines.extend([
            "",
            f"## {case['report_stem']}",
            "",
            f"- Classification: `{case['classification']}`",
            f"- Strict CVE confirmed: `{case['strict_cve_confirmed']}`",
            f"- Result: `{case['result_path']}`",
            f"- Interpretation: {case['interpretation']}",
            f"- Stream evidence: phases `{evidence.get('stream_phase_count')}`, phases with BB lists `{evidence.get('stream_phases_with_bb_lists')}`, materialized records with BB lists `{evidence.get('materialized_records_with_bb_lists')}`",
            f"- Stream BBs: phase covered `{evidence.get('stream_phase_covered_bbs')}`, phase new `{evidence.get('stream_phase_new_bbs')}`, materialized task covered `{evidence.get('materialized_task_covered_bbs')}`, materialized task new `{evidence.get('materialized_task_new_bbs')}`",
            f"- Sink call events: stream `{evidence.get('stream_sink_call_events')}`, materialized `{evidence.get('materialized_sink_call_events')}`, path-progress-only `{evidence.get('path_progress_sink_call_events')}`",
            f"- Boundary evidence: copy events `{boundary.get('copy_event_count', 0)}`, bounded `{boundary.get('bounded_event_count', 0)}`, lower-bound-fit `{boundary.get('append_lower_bound_fit_event_count', 0)}`, overflow `{boundary.get('overflow_event_count', 0)}`, unknown `{boundary.get('unknown_event_count', 0)}`, model-artifact pointers `{boundary.get('model_artifact_event_count', 0)}`",
            f"- Path-progress boundary evidence: copy events `{path_boundary.get('copy_event_count', 0)}`, overflow `{path_boundary.get('overflow_event_count', 0)}`, unknown `{path_boundary.get('unknown_event_count', 0)}`, model-artifact pointers `{path_boundary.get('model_artifact_event_count', 0)}`",
            f"- Input-bound boundary evidence: copy events `{input_boundary.get('copy_event_count', 0)}`, bounded `{input_boundary.get('bounded_event_count', 0)}`, overflow `{input_boundary.get('overflow_event_count', 0)}`, unknown `{input_boundary.get('unknown_event_count', 0)}`",
            "",
            "Materialized-task critical functions:",
        ])
        covered = case.get("materialized_task_covered_critical_functions") or []
        if covered:
            for func in covered[:24]:
                lines.append(f"- `{func['start']}` `{func['name']}`")
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Materialized-task impact/sink functions:")
        impact = case.get("materialized_task_covered_impact_functions") or []
        if impact:
            for func in impact[:24]:
                lines.append(f"- `{func['start']}` `{func['name']}`")
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Stream-phase new critical functions:")
        stream_new = case.get("stream_phase_new_critical_functions") or []
        if stream_new:
            for func in stream_new[:16]:
                lines.append(f"- `{func['start']}` `{func['name']}`")
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Materialized sink call events:")
        sink_events = case.get("materialized_sink_call_events") or []
        if sink_events:
            for event in sink_events[:12]:
                details = []
                for key in (
                    "source_matches_seed_payload",
                    "seed_payload_match_field",
                    "seed_payload_match_length",
                    "dest_symbol",
                    "src_symbol",
                    "src_observed_length",
                    "src_nul_seen",
                    "src_preview",
                    "format_preview",
                    "length",
                    "copy_length_bytes",
                    "dest_object",
                    "dest_object_size",
                    "dest_remaining_bytes",
                    "dest_region",
                    "dest_section",
                    "dest_section_remaining_bytes",
                    "copy_fits_dest_object",
                    "copy_fits_dest_section",
                    "copy_overflow_bytes",
                    "boundary_verdict",
                    "pin",
                    "value",
                ):
                    if event.get(key) not in (None, ""):
                        details.append(f"{key}={event.get(key)!r}")
                lines.append(
                    f"- `{event.get('pc')}` `{event.get('symbol')}` "
                    f"seed={event.get('seed_ascii_preview')!r} "
                    f"r0={event.get('r0')} r1={event.get('r1')} r2={event.get('r2')} "
                    + " ".join(details)
                )
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Input-bound sink call events:")
        input_bound_events = case.get("input_bound_sink_call_events") or []
        if input_bound_events:
            for event in input_bound_events[:8]:
                details = []
                for key in (
                    "source_matches_seed_payload",
                    "seed_payload_match_field",
                    "seed_payload_match_length",
                    "dest_symbol",
                    "src_symbol",
                    "length",
                    "copy_length_bytes",
                    "dest_object",
                    "dest_object_size",
                    "dest_remaining_bytes",
                    "boundary_verdict",
                ):
                    if event.get(key) not in (None, ""):
                        details.append(f"{key}={event.get(key)!r}")
                lines.append(
                    f"- `{event.get('pc')}` `{event.get('symbol')}` "
                    f"seed={event.get('seed_ascii_preview')!r} "
                    f"r0={event.get('r0')} r1={event.get('r1')} r2={event.get('r2')} "
                    + " ".join(details)
                )
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Path-progress-only sink call events:")
        path_sink_events = case.get("path_progress_sink_call_events") or []
        if path_sink_events:
            for event in path_sink_events[:8]:
                details = []
                for key in (
                    "dest_symbol",
                    "src_symbol",
                    "length",
                    "copy_length_bytes",
                    "dest_region",
                    "dest_section",
                    "dest_object",
                    "dest_object_size",
                    "boundary_verdict",
                    "pin",
                    "value",
                ):
                    if event.get(key) not in (None, ""):
                        details.append(f"{key}={event.get(key)!r}")
                lines.append(
                    f"- `{event.get('pc')}` `{event.get('symbol')}` "
                    f"seed={event.get('seed_ascii_preview')!r} "
                    f"r0={event.get('r0')} r1={event.get('r1')} r2={event.get('r2')} "
                    + " ".join(details)
                )
        else:
            lines.append("- `<none>`")
        lines.append("")
        lines.append("Strong-input/path-progress seed records:")
        records = case.get("materialized_records") or []
        if records:
            for record in records[:8]:
                lines.append(
                    f"- `{record.get('context_function')}` seed `{record.get('seed_ascii_preview')}` "
                    f"new_bbs={record.get('new_bbs')} stop=`{record.get('stop_reason')}` "
                    f"reads={record.get('input_read_hit_count')} summary={record.get('stream_input_summary_event_count')} "
                    f"writes={record.get('stream_input_payload_write_count')}"
                )
        else:
            lines.append("- `<none>`")
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /opt/artifact",
        "python3 lsgemu/analyze_elfmultifuzz_materialized_replay.py",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-json", default=str(DEFAULT_VALIDATION_JSON))
    parser.add_argument("--seed-replay-json", default=str(DEFAULT_SEED_REPLAY_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    results = build_results(Path(args.validation_json), Path(args.seed_replay_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "materialized_replay_deep_dive.json"
    md_path = output_dir / "MATERIALIZED_REPLAY_DEEP_DIVE.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, md_path)
    print(json.dumps({
        "case_count": results["case_count"],
        "strict_cve_confirmed_count": results["strict_cve_confirmed_count"],
        "output_json": str(json_path),
        "output_md": str(md_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
