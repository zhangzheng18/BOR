#!/usr/bin/env python3
"""Run focused stream-input seed replay for elfmultifuzz manual candidates."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    script_dir = str(Path(__file__).resolve().parent)
    project_root_text = str(PROJECT_ROOT)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    if project_root_text in sys.path:
        sys.path.remove(project_root_text)
    sys.path.insert(0, project_root_text)

from lsgemu.validate_elfmultifuzz_candidates import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_FIRMWARE_ROOT,
    DEFAULT_OUTPUT_DIR,
    parse_report_identity,
    report_files,
    resolve_firmware,
    seed_files_for_report,
)
from lsgemu.deployment_config import configured_path


SRCV4_ROOT = configured_path("LSGEMU_SOURCE_ROOT", Path(__file__).resolve().parents[1])
PROJECT_ROOT = configured_path("LSGEMU_PROJECT_ROOT", SRCV4_ROOT.parent)
LOCAL_UNICORN_BUILD = configured_path("LIBUNICORN_PATH", PROJECT_ROOT / "unicorn" / "build")
LOCAL_UNICORN_BINDINGS = configured_path("LSGEMU_UNICORN_PYTHON_BINDINGS", PROJECT_ROOT / "unicorn" / "bindings" / "python")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "case"


def read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def build_env(seed_files: Sequence[Path]) -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(SRCV4_ROOT), str(LOCAL_UNICORN_BINDINGS)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["LIBUNICORN_PATH"] = str(LOCAL_UNICORN_BUILD)
    env["LD_LIBRARY_PATH"] = (
        f"{LOCAL_UNICORN_BUILD}{os.pathsep}{env['LD_LIBRARY_PATH']}"
        if env.get("LD_LIBRARY_PATH")
        else str(LOCAL_UNICORN_BUILD)
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LSGEMU_UNKNOWN_LOOP_THRESHOLD", "1000")
    env.setdefault("LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "100000")
    env.setdefault("LSGEMU_DELAY_LOOP_THRESHOLD", "100000")
    env.setdefault("LSGEMU_FAST_LOOP_ANALYSIS", "1")
    env.setdefault("LSGEMU_MAX_LLM_CALLS_PER_LOOP", "1")
    env.setdefault("LSGEMU_MAX_DEADLOCK_LLM_CALLS_PER_LOOP", "1")
    env.setdefault("LSGEMU_STREAM_ALLOW_FALLBACK_CONTEXTS", "1")
    env.setdefault("LSGEMU_STREAM_CONTEXTS_PER_PROFILE_SEED", "2")
    env.setdefault("LSGEMU_STREAM_ZERO_NEW_CONTEXT_LIMIT", "4")
    env.setdefault("LSGEMU_RECORD_PHASE_BB_LISTS", "1")
    env.setdefault("LSGEMU_PHASE_BB_LIST_LIMIT", "0")
    env.setdefault("LSGEMU_STREAM_DEBUG_RECORD_LIMIT", "128")
    env.setdefault("LSGEMU_STREAM_TASK_BB_LIST_LIMIT", "0")
    env.setdefault("LSGEMU_STREAM_EFFECT_TRACE", "1")
    env.setdefault("LSGEMU_STREAM_EFFECT_EVENT_LIMIT", "64")
    env["LSGEMU_STREAM_FIRST_PASS_SEEDS"] = str(max(1, len(seed_files)))
    seed_hex = []
    for path in seed_files:
        data = path.read_bytes()
        if not data:
            continue
        seed_hex.append(data[:4096].hex())
    env["LSGEMU_STREAM_EXTRA_SEEDS_HEX"] = ";".join(seed_hex)
    return env


def expected_report_path(output_dir: Path, firmware: Path) -> Path:
    return output_dir / f"{firmware.stem}_interleaved_report.json"


def summarize_report(report: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not report:
        return {}
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
        "stream_seed_new_bbs": {},
        "stream_stop_reasons": {},
        "stream_phase_new_bb_list_count": 0,
        "stream_debug_records_with_bb_lists": 0,
        "stream_materialized_records_with_bb_lists": 0,
        "stream_sink_call_event_count": 0,
        "stream_sink_call_symbols": {},
        "stream_seed_bound_sink_call_event_count": 0,
        "stream_seed_bound_sink_call_symbols": {},
        "stream_path_progress_sink_call_event_count": 0,
        "stream_path_progress_sink_call_symbols": {},
        "stream_strong_input_sink_call_event_count": 0,
        "stream_strong_input_sink_call_symbols": {},
    }
    seed_new_bbs: Dict[str, int] = {}
    stop_reasons: Dict[str, int] = {}
    for phase in stream_phases.values():
        summary["stream_profiles_discovered"] = int(summary["stream_profiles_discovered"]) + int(phase.get("profiles_discovered") or 0)
        summary["stream_tasks_run"] = int(summary["stream_tasks_run"]) + int(phase.get("tasks_run") or 0)
        summary["stream_tasks_injected"] = int(summary["stream_tasks_injected"]) + int(phase.get("tasks_injected") or 0)
        summary["stream_tasks_with_new_bbs"] = int(summary["stream_tasks_with_new_bbs"]) + int(phase.get("tasks_with_new_bbs") or 0)
        if isinstance(phase.get("new_bb_list"), list):
            summary["stream_phase_new_bb_list_count"] = int(summary["stream_phase_new_bb_list_count"]) + len(phase.get("new_bb_list") or [])
        for record in phase.get("debug_records") or []:
            if not isinstance(record, dict):
                continue
            summary["stream_input_read_hit_count"] = int(summary["stream_input_read_hit_count"]) + int(record.get("input_read_hit_count") or 0)
            summary["stream_summary_event_count"] = int(summary["stream_summary_event_count"]) + int(record.get("stream_input_summary_event_count") or 0)
            summary["stream_payload_write_count"] = int(summary["stream_payload_write_count"]) + int(record.get("stream_input_payload_write_count") or 0)
            strong_input_record = (
                int(record.get("input_read_hit_count") or 0) > 0
                or int(record.get("stream_input_payload_write_count") or 0) > 0
            )
            path_progress_record = int(record.get("new_bbs") or 0) > 0
            seed_bound_record = strong_input_record or path_progress_record
            if strong_input_record:
                summary["stream_strong_input_record_count"] = int(summary["stream_strong_input_record_count"]) + 1
            if path_progress_record:
                summary["stream_path_progress_record_count"] = int(summary["stream_path_progress_record_count"]) + 1
            if seed_bound_record:
                summary["stream_seed_bound_record_count"] = int(summary["stream_seed_bound_record_count"]) + 1
            if isinstance(record.get("covered_bb_list"), list) or isinstance(record.get("new_bb_list"), list):
                summary["stream_debug_records_with_bb_lists"] = int(summary["stream_debug_records_with_bb_lists"]) + 1
                if seed_bound_record:
                    summary["stream_materialized_records_with_bb_lists"] = int(summary["stream_materialized_records_with_bb_lists"]) + 1
            summary["stream_sink_call_event_count"] = int(summary["stream_sink_call_event_count"]) + int(record.get("sink_call_event_count") or 0)
            sink_symbols = summary["stream_sink_call_symbols"]
            seed_bound_sink_symbols = summary["stream_seed_bound_sink_call_symbols"]
            path_progress_sink_symbols = summary["stream_path_progress_sink_call_symbols"]
            strong_input_sink_symbols = summary["stream_strong_input_sink_call_symbols"]
            if isinstance(sink_symbols, dict):
                for event in record.get("sink_call_events") or []:
                    if not isinstance(event, dict):
                        continue
                    symbol = str(event.get("symbol") or "")
                    if symbol:
                        sink_symbols[symbol] = int(sink_symbols.get(symbol, 0) or 0) + 1
                        if seed_bound_record and isinstance(seed_bound_sink_symbols, dict):
                            seed_bound_sink_symbols[symbol] = int(seed_bound_sink_symbols.get(symbol, 0) or 0) + 1
                        if path_progress_record and isinstance(path_progress_sink_symbols, dict):
                            path_progress_sink_symbols[symbol] = int(path_progress_sink_symbols.get(symbol, 0) or 0) + 1
                        if strong_input_record and isinstance(strong_input_sink_symbols, dict):
                            strong_input_sink_symbols[symbol] = int(strong_input_sink_symbols.get(symbol, 0) or 0) + 1
            if seed_bound_record:
                summary["stream_seed_bound_sink_call_event_count"] = (
                    int(summary["stream_seed_bound_sink_call_event_count"])
                    + int(record.get("sink_call_event_count") or 0)
                )
            if path_progress_record:
                summary["stream_path_progress_sink_call_event_count"] = (
                    int(summary["stream_path_progress_sink_call_event_count"])
                    + int(record.get("sink_call_event_count") or 0)
                )
            if strong_input_record:
                summary["stream_strong_input_sink_call_event_count"] = (
                    int(summary["stream_strong_input_sink_call_event_count"])
                    + int(record.get("sink_call_event_count") or 0)
                )
            seed_hex = str(record.get("seed_hex") or "")
            if seed_hex and int(record.get("new_bbs") or 0) > 0:
                seed_new_bbs[seed_hex] = max(seed_new_bbs.get(seed_hex, 0), int(record.get("new_bbs") or 0))
            reason = str(record.get("stop_reason") or "unknown")
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
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


def run_case(
    *,
    report_path: Path,
    firmware: Path,
    seed_files: Sequence[Path],
    output_root: Path,
    seconds: int,
    max_tasks: int,
    max_seeds: int,
    force: bool,
    exploration_mode: str,
) -> Dict[str, object]:
    case_dir = output_root / ("full_exploration" if exploration_mode == "full" else "seed_replay") / safe_name(report_path.stem)
    case_dir.mkdir(parents=True, exist_ok=True)
    report_file = expected_report_path(case_dir, firmware)
    stdout_file = case_dir / "stdout.log"
    stderr_file = case_dir / "stderr.log"
    command_file = case_dir / "command.json"
    selected_seeds = list(seed_files)[:max(1, int(max_seeds))]

    if report_file.exists() and not force:
        payload = read_json(report_file)
        return {
            "report": str(report_path),
            "firmware": str(firmware),
            "seed_files": [str(path) for path in selected_seeds],
            "case_dir": str(case_dir),
            "result_path": str(report_file),
            "status": "reused_existing_result",
            "summary": summarize_report(payload),
        }

    total_minutes = max(2, int((seconds + 119) // 60) + 1)
    cmd = [
        sys.executable,
        "-m",
        "lsgemu.cached_interleaved_runner",
        "--firmware",
        str(firmware),
        "--output-dir",
        str(case_dir),
        "--clean-constraint-file",
        "--skip-cold-isr",
        "--baseline-instructions",
        "500000",
        "--baseline-timeout-seconds",
        "90",
        "--stream-input-seconds",
        str(max(1, int(seconds))),
        "--stream-input-instructions",
        "120000",
        "--stream-input-replay-timeout-us",
        "1000000",
        "--stream-input-max-profiles",
        "16",
        "--stream-input-max-contexts",
        "48",
        "--stream-input-max-seeds",
        str(max(1, int(max_seeds))),
        "--stream-input-max-tasks",
        str(max(1, int(max_tasks))),
        "--stream-input-no-new-bbs",
        "8192",
        "--total-time-minutes",
        str(total_minutes),
    ]

    if exploration_mode == "stream-only":
        cmd.extend([
        "--semantic-entry-prefix-seconds",
        "0",
        "--contextual-isr-contexts",
        "0",
        "--contextual-isr-time-seconds",
        "0",
        "--contextual-isr-reservoir-seconds",
        "0",
        "--contextual-isr-frontier-seconds",
        "0",
        "--vector-cleanup-seconds",
        "0",
        "--semantic-frontier-flush-seconds",
        "0",
        "--direct-call-continuation-seconds",
        "0",
        "--direct-call-summary-return-seconds",
        "0",
        "--rtos-thread-entry-seconds",
        "0",
        "--frontier-successor-replay-seconds",
        "0",
        "--frontier-successor-replay-tail-seconds",
        "0",
        "--frontier-successor-flush-seconds",
        "0",
        "--frontier-cycle-tail-reserve-seconds",
        "0",
        "--targeted-frontier-reserve-seconds",
        "0",
        "--reservoir-round-seconds",
        "0",
        "--mmio-round-seconds",
        "0",
        "--interleaved-frontier-round-seconds",
        "0",
        "--max-rounds",
        "0",
        "--frontier-round-seconds",
        "0",
        "--switch-frontier-round-seconds",
        "0",
        "--frontier-cycle-max-cycles",
        "0",
        "--post-interleaved-reserve-seconds",
        "0",
        "--post-interleaved-reserve-ratio",
        "0",
        "--disable-deadline-drain",
        ])
    else:
        # Full exploration keeps the complete lsgemu backtracking/reservoir
        # pipeline active while still replaying report seeds.  Keep explicit
        # caps modest; cached_interleaved_runner will scale them with the
        # wallclock budget unless LSGEMU_DISABLE_WALLCLOCK_STAGE_CAPS is set.
        cmd.extend([
            "--contextual-isr-contexts",
            "8",
            "--contextual-isr-time-seconds",
            "30",
            "--contextual-isr-reservoir-seconds",
            "45",
            "--contextual-isr-frontier-seconds",
            "20",
            "--semantic-frontier-flush-seconds",
            "20",
            "--direct-call-continuation-seconds",
            "20",
            "--direct-call-summary-return-seconds",
            "20",
            "--rtos-thread-entry-seconds",
            "30",
            "--frontier-successor-replay-seconds",
            "240",
            "--frontier-successor-replay-tail-seconds",
            "120",
            "--frontier-successor-flush-seconds",
            "20",
            "--reservoir-round-seconds",
            "180",
            "--mmio-round-seconds",
            "60",
            "--max-rounds",
            "4",
            "--frontier-round-seconds",
            "45",
            "--switch-frontier-round-seconds",
            "30",
            "--frontier-cycle-max-cycles",
            "6",
        ])
    command_file.write_text(
        json.dumps(
            {
                "command": cmd,
                "seed_files": [str(path) for path in selected_seeds],
                "seconds": seconds,
                "max_tasks": max_tasks,
                "max_seeds": max_seeds,
                "exploration_mode": exploration_mode,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(SRCV4_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=max(240, int(seconds) + 600),
            check=False,
            env=build_env(selected_seeds),
        )
        stdout_file.write_text(completed.stdout, encoding="utf-8", errors="replace")
        stderr_file.write_text(completed.stderr, encoding="utf-8", errors="replace")
        payload = read_json(report_file)
        status = "completed" if completed.returncode == 0 and payload else "failed"
        return {
            "report": str(report_path),
            "firmware": str(firmware),
            "seed_files": [str(path) for path in selected_seeds],
            "case_dir": str(case_dir),
            "result_path": str(report_file),
            "stdout_path": str(stdout_file),
            "stderr_path": str(stderr_file),
            "command_path": str(command_file),
            "status": status,
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "summary": summarize_report(payload),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        stdout_file.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_file.write_text(stderr, encoding="utf-8", errors="replace")
        return {
            "report": str(report_path),
            "firmware": str(firmware),
            "seed_files": [str(path) for path in selected_seeds],
            "case_dir": str(case_dir),
            "stdout_path": str(stdout_file),
            "stderr_path": str(stderr_file),
            "command_path": str(command_file),
            "status": "timeout",
            "elapsed_seconds": time.time() - started,
            "summary": summarize_report(read_json(report_file)),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }


def discover_seed_cases(audit_dir: Path, firmware_root: Path) -> List[Dict[str, object]]:
    seed_root = audit_dir / "poc_seeds"
    cases: List[Dict[str, object]] = []
    for report_path in report_files(audit_dir):
        text = report_path.read_text(encoding="utf-8", errors="replace")
        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        firmware_text, _, _ = parse_report_identity(text)
        firmware = resolve_firmware(firmware_text, firmware_root)
        if not firmware:
            continue
        seeds = seed_files_for_report(report_path.stem, title, seed_root)
        if not seeds:
            continue
        cases.append({
            "report_path": report_path,
            "firmware": firmware,
            "seed_files": seeds,
        })
    return cases


def write_markdown(results: Dict[str, object], path: Path) -> None:
    lines = [
        "# elfmultifuzz Seed Replay Validation",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Cases selected: `{results['case_count']}`",
        f"- Cases run: `{len(results['runs'])}`",
        f"- Exploration mode: `{results.get('exploration_mode', 'stream-only')}`",
        "",
        "| Report | Status | Seeds | Tasks | Injected | New-BB Tasks | Weak Stream | Path Progress | Strict Input | Seed-Bound Records | Seed-Bound Sinks | Valid Coverage | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in results["runs"]:
        summary = item.get("summary") or {}
        valid_cov = float(summary.get("valid_coverage_rate") or 0.0)
        lines.append(
            f"| `{Path(item['report']).stem}` | `{item.get('status')}` | {len(item.get('seed_files') or [])} | "
            f"{int(summary.get('stream_tasks_run') or 0)} | {int(summary.get('stream_tasks_injected') or 0)} | "
            f"{int(summary.get('stream_tasks_with_new_bbs') or 0)} | `{summary.get('input_materialized_weak')}` | "
            f"`{summary.get('input_path_progress')}` | "
            f"`{summary.get('input_materialized')}` | {int(summary.get('stream_seed_bound_record_count') or 0)} | "
            f"{int(summary.get('stream_seed_bound_sink_call_event_count') or 0)} | "
            f"{valid_cov:.2f}% | `{item.get('result_path', item.get('case_dir'))}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--firmware-root", default=str(DEFAULT_FIRMWARE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seconds-per-case", type=int, default=90)
    parser.add_argument("--max-tasks", type=int, default=96)
    parser.add_argument("--max-seeds", type=int, default=16)
    parser.add_argument(
        "--exploration-mode",
        choices=["stream-only", "full"],
        default="stream-only",
        help="stream-only validates candidate inputs; full enables lsgemu reservoir/frontier/backtracking stages for coverage.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name-filter", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir).resolve()
    firmware_root = Path(args.firmware_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_seed_cases(audit_dir, firmware_root)
    if args.name_filter:
        token = args.name_filter.lower()
        cases = [
            case for case in cases
            if token in str(case["report_path"]).lower() or token in str(case["firmware"]).lower()
        ]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    runs = []
    for case in cases:
        runs.append(
            run_case(
                report_path=case["report_path"],
                firmware=case["firmware"],
                seed_files=case["seed_files"],
                output_root=output_dir,
                seconds=args.seconds_per_case,
                max_tasks=args.max_tasks,
                max_seeds=args.max_seeds,
                force=args.force,
                exploration_mode=args.exploration_mode,
            )
        )
        summary_path = output_dir / "seed_replay_results.json"
        summary_path.write_text(
            json.dumps(
                {
                    "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "case_count": len(cases),
                    "exploration_mode": args.exploration_mode,
                    "runs": runs,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    results = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "case_count": len(cases),
        "exploration_mode": args.exploration_mode,
        "runs": runs,
    }
    json_path = output_dir / "seed_replay_results.json"
    md_path = output_dir / "SEED_REPLAY_VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, md_path)
    print(json.dumps({
        "case_count": len(cases),
        "completed": sum(1 for item in runs if item.get("status") in {"completed", "reused_existing_result"}),
        "weak_stream_input": sum(1 for item in runs if (item.get("summary") or {}).get("input_materialized_weak")),
        "path_progress": sum(1 for item in runs if (item.get("summary") or {}).get("input_path_progress")),
        "strict_input_materialized": sum(1 for item in runs if (item.get("summary") or {}).get("input_materialized")),
        "output_json": str(json_path),
        "output_md": str(md_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
