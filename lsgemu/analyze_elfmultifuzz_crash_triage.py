#!/usr/bin/env python3
"""Triage abnormal stop reasons in elfmultifuzz seed replay reports.

This script is intentionally conservative: an emulator stop is not a firmware
crash unless the report binds it to attacker-controlled input and provides
enough replay state for root-cause analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SEED_REPLAY_JSON = Path("reports/elfmultifuzz_candidate_validation/seed_replay_results.json")
DEFAULT_OUTPUT_DIR = Path("reports/elfmultifuzz_candidate_validation")
DEFAULT_SEED_REPLAY_DIR = DEFAULT_OUTPUT_DIR / "seed_replay"


NON_CRASH_STOP_REASONS = (
    "timeout_or_max_instructions_reached",
    "max_instructions_reached",
    "quiescence_no_new_bbs",
    "target_probe_tail_complete",
    "stall_no_new_bb_watchdog",
)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def parse_intish(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except Exception:
            return None
    return None


def seed_preview(seed_hex: object, limit: int = 96) -> str:
    text = str(seed_hex or "")
    if not text:
        return ""
    try:
        return bytes.fromhex(text[: limit * 2]).decode("ascii", errors="replace")
    except Exception:
        return text[:limit]


def is_non_crash_stop(reason: str) -> bool:
    reason_lower = reason.lower()
    return any(token.lower() == reason_lower for token in NON_CRASH_STOP_REASONS)


def is_abnormal_stop(reason: str) -> bool:
    if not reason or is_non_crash_stop(reason):
        return False
    lowered = reason.lower()
    return any(
        token in lowered
        for token in (
            "uc_err",
            "invalid",
            "fatal",
            "exception",
            "unproven_dynamic_code_target",
            "hardfault",
            "busfault",
            "usagefault",
            "memmanage",
        )
    )


def stream_summary_has_seed_bytes(record: Dict[str, object]) -> bool:
    for event in record.get("stream_input_summary_events") or []:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "")
        if event.get("seed_hex") and kind in {
            "stream_byte",
            "stream_status_read",
            "stream_payload_write",
            "stream_buffer",
        }:
            return True
    return False


def binding_for_record(record: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    input_reads = int(record.get("input_read_hit_count") or 0)
    payload_writes = int(record.get("stream_input_payload_write_count") or 0)
    summary_events = int(record.get("stream_input_summary_event_count") or 0)
    new_bbs = int(record.get("new_bbs") or 0)
    sink_events = int(record.get("sink_call_event_count") or 0)
    api_stream = summary_events > 0 and stream_summary_has_seed_bytes(record)

    if input_reads > 0 or payload_writes > 0:
        binding = "strong_input_read_or_payload_write"
    elif api_stream:
        binding = "api_stream_input_observed"
    elif new_bbs > 0:
        binding = "path_progress_only"
    else:
        binding = "not_input_bound"

    return binding, {
        "input_read_hit_count": input_reads,
        "stream_input_payload_write_count": payload_writes,
        "stream_input_summary_event_count": summary_events,
        "api_stream_seed_bytes_observed": api_stream,
        "new_bbs": new_bbs,
        "sink_call_event_count": sink_events,
    }


def classify_stop(reason: str, binding: str) -> Tuple[str, str, int]:
    lowered = reason.lower()
    input_observed = binding in {
        "strong_input_read_or_payload_write",
        "api_stream_input_observed",
    }
    path_only = binding == "path_progress_only"

    if "uc_err_map" in lowered or "invalid memory mapping" in lowered:
        if input_observed:
            return (
                "input_observed_invalid_memory_needs_replay",
                "输入已被 API/内存模型消费，随后出现 Unicorn invalid memory；需要带 PC/LR/SP/访存地址重放确认是真实 firmware fault 还是映射模型缺失。",
                90,
            )
        return (
            "model_or_harness_invalid_memory",
            "没有输入绑定证据的 invalid memory，更可能是 replay 映射、外设、栈/堆模型或错误入口造成。",
            35 if path_only else 25,
        )
    if "uc_err_insn_invalid" in lowered or "invalid instruction" in lowered:
        if input_observed:
            return (
                "input_observed_invalid_instruction_needs_replay",
                "输入已被消费后出现 invalid instruction；需要检查 PC Thumb 位、间接跳转目标和 trace tail。",
                85,
            )
        return (
            "model_or_harness_invalid_instruction",
            "没有输入绑定证据的 invalid instruction，优先怀疑错误入口、Thumb 状态、未建模代码区或动态代码识别问题。",
            30 if path_only else 20,
        )
    if "fatal_sink_terminal" in lowered or "hardfault" in lowered or "busfault" in lowered or "usagefault" in lowered or "memmanage" in lowered:
        if input_observed:
            return (
                "input_observed_terminal_or_fault_needs_root_cause",
                "输入已被消费后到达 fatal/fault 终止点；这是高优先级候选，但仍需确认 root cause、攻击面和可复现影响。",
                88,
            )
        return (
            "path_or_model_terminal_without_input_binding",
            "到达 fatal/fault 类终止点，但缺少输入绑定，不能作为 crash 漏洞证据。",
            40 if path_only else 25,
        )
    if "unproven_dynamic_code_target" in lowered:
        if input_observed:
            return (
                "input_observed_unproven_dynamic_target_needs_indirect_cf_triage",
                "输入已被消费后遇到未证明动态代码目标；需确认是否为输入影响的间接跳转，或只是 jump-table/dynamic-code 模型不足。",
                80,
            )
        return (
            "model_or_harness_unproven_dynamic_target",
            "未证明动态代码目标缺少输入绑定，当前更像 CFG/动态代码模型工件。",
            30 if path_only else 20,
        )
    if "exception" in lowered:
        return (
            "emulator_exception_needs_log_triage",
            "runner 捕获到异常但缺少具体 Unicorn 停止类型；需要查看 stdout/stderr 或重跑补寄存器。",
            60 if input_observed else 25,
        )
    return (
        "abnormal_stop_unclassified",
        "异常停止类型未归类，需要人工查看原始 record。",
        50 if input_observed else 20,
    )


def sink_symbols(record: Dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in record.get("sink_call_events") or []:
        if not isinstance(event, dict):
            continue
        symbol = str(event.get("symbol") or "")
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def first_stream_event(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    for event in record.get("stream_input_summary_events") or []:
        if isinstance(event, dict):
            return {
                "pc": event.get("pc"),
                "symbol": event.get("symbol"),
                "kind": event.get("kind"),
                "style": event.get("style"),
                "return_value": event.get("return_value"),
                "seed_hex": event.get("seed_hex"),
            }
    return None


def collect_report_inputs(
    seed_replay_json: Path,
    seed_replay_dir: Optional[Path],
    include_existing_reports: bool = True,
) -> List[Dict[str, object]]:
    reports: Dict[str, Dict[str, object]] = {}

    def add_report(path: Path, meta: Dict[str, object]) -> None:
        if not path.exists():
            return
        key = str(path.resolve())
        existing = reports.get(key, {})
        merged = dict(existing)
        for meta_key, meta_value in meta.items():
            if meta_value is not None and meta_value != "":
                merged[meta_key] = meta_value
        merged["result_path"] = str(path)
        reports[key] = merged

    if seed_replay_json.exists():
        seed_replay = read_json(seed_replay_json)
        for run in seed_replay.get("runs") or []:
            if not isinstance(run, dict):
                continue
            result_path = Path(str(run.get("result_path") or ""))
            if not result_path.is_absolute():
                result_path = seed_replay_json.parent / result_path
            report_stem = Path(str(run.get("report") or "")).stem or result_path.parent.name
            add_report(result_path, {
                "report_stem": report_stem,
                "firmware": run.get("firmware"),
                "report_source": "seed_replay_json",
            })

    if include_existing_reports and seed_replay_dir and seed_replay_dir.exists():
        for result_path in sorted(seed_replay_dir.glob("*/*_interleaved_report.json")):
            add_report(result_path, {
                "report_stem": result_path.parent.name,
                "firmware": None,
                "report_source": "seed_replay_dir",
            })

    return list(reports.values())


def collect_records(
    seed_replay_json: Path,
    seed_replay_dir: Optional[Path] = DEFAULT_SEED_REPLAY_DIR,
    include_existing_reports: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    report_inputs = collect_report_inputs(
        seed_replay_json,
        seed_replay_dir,
        include_existing_reports=include_existing_reports,
    )
    records: List[Dict[str, object]] = []
    for report_input in report_inputs:
        result_path = Path(str(report_input.get("result_path") or ""))
        report_stem = str(report_input.get("report_stem") or result_path.parent.name)
        if not result_path.exists():
            continue
        payload = read_json(result_path)
        firmware = report_input.get("firmware") or payload.get("firmware")
        for phase_name, phase in (payload.get("phase_metadata") or {}).items():
            if not isinstance(phase, dict):
                continue
            for index, record in enumerate(phase.get("debug_records") or []):
                if not isinstance(record, dict):
                    continue
                reason = str(record.get("stop_reason") or "")
                if not is_abnormal_stop(reason):
                    continue
                binding, binding_details = binding_for_record(record)
                classification, interpretation, priority = classify_stop(reason, binding)
                registers = record.get("registers") if isinstance(record.get("registers"), dict) else {}
                run_result = record.get("run_result") if isinstance(record.get("run_result"), dict) else {}
                item = {
                    "report_stem": report_stem,
                    "firmware": firmware,
                    "result_path": str(result_path),
                    "report_source": report_input.get("report_source"),
                    "phase": phase_name,
                    "record_index": index,
                    "context_bb": record.get("context_bb"),
                    "context_function": record.get("context_function"),
                    "profile_kind": record.get("profile_kind"),
                    "object_symbol": record.get("object_symbol"),
                    "stop_reason": reason,
                    "classification": classification,
                    "interpretation": interpretation,
                    "priority": priority,
                    "binding": binding,
                    "binding_details": binding_details,
                    "seed_hex_prefix": str(record.get("seed_hex") or "")[:160],
                    "seed_ascii_preview": seed_preview(record.get("seed_hex")),
                    "pc": record.get("pc") or registers.get("pc"),
                    "lr": record.get("lr") or registers.get("lr"),
                    "sp": record.get("sp") or registers.get("sp"),
                    "trace_tail": record.get("trace_tail") or run_result.get("last_10_pcs"),
                    "mmio_access_tail": record.get("mmio_access_tail") or run_result.get("mmio_access_tail"),
                    "memory_access_tail": record.get("memory_access_tail") or run_result.get("memory_access_tail"),
                    "watch_memory_events": record.get("watch_memory_events") or run_result.get("watch_memory_events"),
                    "watch_pc_events": record.get("watch_pc_events") or run_result.get("watch_pc_events"),
                    "last_unmapped_access": record.get("last_unmapped_access") or run_result.get("last_unmapped_access"),
                    "memory_constraint_guard_stats": record.get("memory_constraint_guard_stats") or run_result.get("memory_constraint_guard_stats"),
                    "memory_constraint_guard_history": record.get("memory_constraint_guard_history") or run_result.get("memory_constraint_guard_history"),
                    "first_input_read": (record.get("input_read_hits") or [None])[0],
                    "first_stream_event": first_stream_event(record),
                    "sink_symbols": sink_symbols(record),
                    "sink_call_events_sample": (record.get("sink_call_events") or [])[:4],
                }
                if not item["trace_tail"]:
                    covered = record.get("covered_bb_list") or record.get("covered_bb_list_sample") or []
                    item["trace_tail"] = [
                        f"0x{int(bb) & 0xFFFFFFFF:08x}"
                        for bb in covered[-10:]
                        if parse_intish(bb) is not None
                    ]
                records.append(item)
    records.sort(key=lambda item: (-int(item["priority"]), str(item["report_stem"]), str(item["stop_reason"])))
    return records, report_inputs


def count_by(items: Iterable[Dict[str, object]], key: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts[str(item.get(key))] += 1
    return dict(sorted(counts.items()))


def build_results(
    seed_replay_json: Path,
    seed_replay_dir: Path,
    include_existing_reports: bool = True,
) -> Dict[str, object]:
    records, report_inputs = collect_records(
        seed_replay_json,
        seed_replay_dir=seed_replay_dir,
        include_existing_reports=include_existing_reports,
    )
    by_case: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        by_case[str(item["report_stem"])][str(item["classification"])] += 1
    high_priority = [item for item in records if int(item.get("priority") or 0) >= 80]
    input_observed = [
        item for item in records
        if item.get("binding") in {"strong_input_read_or_payload_write", "api_stream_input_observed"}
    ]
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed_replay_json": str(seed_replay_json),
        "seed_replay_dir": str(seed_replay_dir),
        "include_existing_reports": include_existing_reports,
        "reports_scanned": len(report_inputs),
        "report_inputs": report_inputs,
        "abnormal_record_count": len(records),
        "input_observed_abnormal_count": len(input_observed),
        "high_priority_record_count": len(high_priority),
        "classification_counts": count_by(records, "classification"),
        "binding_counts": count_by(records, "binding"),
        "stop_reason_counts": count_by(records, "stop_reason"),
        "case_classification_counts": {
            case: dict(counter)
            for case, counter in sorted(by_case.items())
        },
        "high_priority_records": high_priority[:64],
        "records": records,
    }


def write_markdown(results: Dict[str, object], path: Path) -> None:
    lines = [
        "# elfmultifuzz Crash / Fault Triage",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Seed replay JSON: `{results['seed_replay_json']}`",
        f"- Seed replay dir: `{results['seed_replay_dir']}`",
        f"- Include existing per-case reports: `{results['include_existing_reports']}`",
        f"- Reports scanned: `{results['reports_scanned']}`",
        f"- Abnormal stop records: `{results['abnormal_record_count']}`",
        f"- Input-observed abnormal records: `{results['input_observed_abnormal_count']}`",
        f"- High-priority records: `{results['high_priority_record_count']}`",
        "",
        "## Interpretation",
        "",
        "- `input_observed_*` means the seed was consumed by a stream API or written/read through the input model before the abnormal stop. It is a high-priority candidate, not a confirmed crash.",
        "- `path_progress_only` means the task reached new code but the seed was not observed as consumed. Treat this as exploration evidence, not crash proof.",
        "- `model_or_harness_*` means the current evidence is more consistent with replay/model/entry issues than a firmware vulnerability.",
        "",
        "## Summary",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for key, value in results["classification_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "| Binding | Count |", "| --- | ---: |"])
    for key, value in results["binding_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "| Stop Reason | Count |", "| --- | ---: |"])
    for key, value in results["stop_reason_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "## High Priority",
        "",
        "| Report | Stop | Binding | Context | PC | LR | Seed | Interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    if results["high_priority_records"]:
        for item in results["high_priority_records"]:
            interp = str(item.get("interpretation") or "").replace("|", "\\|")
            seed = repr(str(item.get("seed_ascii_preview") or "")[:60]).replace("|", "\\|")
            lines.append(
                f"| `{item['report_stem']}` | `{item['stop_reason']}` | `{item['binding']}` | "
                f"`{item.get('context_function')}` | `{item.get('pc')}` | `{item.get('lr')}` | "
                f"{seed} | {interp} |"
            )
    else:
        lines.append("| `<none>` | | | | | | | |")

    lines.extend(["", "## Case Details", ""])
    for case, counts in results["case_classification_counts"].items():
        lines.append(f"### {case}")
        for key, value in counts.items():
            lines.append(f"- `{key}`: `{value}`")
        for item in [record for record in results["records"] if record["report_stem"] == case][:12]:
            lines.append(
                f"- `{item['stop_reason']}` `{item['classification']}` binding=`{item['binding']}` "
                f"context=`{item.get('context_function')}` pc=`{item.get('pc')}` lr=`{item.get('lr')}` "
                f"new_bbs=`{item['binding_details'].get('new_bbs')}` sinks=`{item.get('sink_symbols')}`"
            )
            if item.get("first_stream_event"):
                lines.append(f"  stream_event: `{item['first_stream_event']}`")
            if item.get("trace_tail"):
                lines.append(f"  trace_tail: `{item['trace_tail']}`")
            if item.get("watch_memory_events"):
                lines.append(f"  watch_memory_events: `{item['watch_memory_events']}`")
            if item.get("watch_pc_events"):
                lines.append(f"  watch_pc_events: `{item['watch_pc_events']}`")
            if item.get("last_unmapped_access"):
                lines.append(f"  last_unmapped_access: `{item['last_unmapped_access']}`")
            if item.get("memory_constraint_guard_stats"):
                lines.append(f"  memory_constraint_guard_stats: `{item['memory_constraint_guard_stats']}`")
            if item.get("memory_constraint_guard_history"):
                lines.append(f"  memory_constraint_guard_history: `{item['memory_constraint_guard_history']}`")
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /opt/artifact",
        "python3 lsgemu/analyze_elfmultifuzz_crash_triage.py",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-replay-json", default=str(DEFAULT_SEED_REPLAY_JSON))
    parser.add_argument("--seed-replay-dir", default=str(DEFAULT_SEED_REPLAY_DIR))
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Only scan reports referenced by --seed-replay-json; do not scan existing per-case reports.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = build_results(
        Path(args.seed_replay_json),
        Path(args.seed_replay_dir),
        include_existing_reports=not args.current_only,
    )
    json_path = output_dir / "crash_triage_results.json"
    md_path = output_dir / "CRASH_TRIAGE_REPORT.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, md_path)
    print(json.dumps({
        "reports_scanned": results["reports_scanned"],
        "abnormal_record_count": results["abnormal_record_count"],
        "input_observed_abnormal_count": results["input_observed_abnormal_count"],
        "high_priority_record_count": results["high_priority_record_count"],
        "output_json": str(json_path),
        "output_md": str(md_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
