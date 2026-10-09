#!/usr/bin/env python3
"""One-command runner for the LSGEmu RQ1/RQ2/RQ3 evaluation.

The script intentionally keeps orchestration outside the emulator.  LSGEmu
produces the authoritative execution reports; this file runs the selected
experiments, records commands/logs, and extracts paper-ready summaries.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, IO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.deployment_config import apply_config_from_argv, configured_path
from lsgemu.artifact_io import atomic_json_dump
from experiments.run_lsgemu_serial_eval import llm_runtime_identity

DEPLOYMENT_CONFIG = apply_config_from_argv()

LSGEMU = PROJECT_ROOT / "lsgemu" / "lsgemu.py"
RESULTS_ROOT = PROJECT_ROOT / "evaluation" / "results"
ELFMULTIFUZZ_ROOT = configured_path("ELFMULTIFUZZ_ROOT", PROJECT_ROOT / "datasets" / "elfmultifuzz")


P2IM_TARGETS: dict[str, Path] = {
    "CNC": ELFMULTIFUZZ_ROOT / "P2IM" / "CNC" / "CNC.elf",
    "Console": ELFMULTIFUZZ_ROOT / "P2IM" / "Console" / "Console.elf",
    "Drone": ELFMULTIFUZZ_ROOT / "P2IM" / "Drone" / "Drone.elf",
    "Gateway": ELFMULTIFUZZ_ROOT / "P2IM" / "Gateway" / "Gateway.elf",
    "Heat_Press": ELFMULTIFUZZ_ROOT / "P2IM" / "Heat_Press" / "Heat_Press.elf",
    "PLC": ELFMULTIFUZZ_ROOT / "P2IM" / "PLC" / "PLC.elf",
    "Reflow_Oven": ELFMULTIFUZZ_ROOT / "P2IM" / "Reflow_Oven" / "Reflow_Oven.elf",
    "Robot": ELFMULTIFUZZ_ROOT / "P2IM" / "Robot" / "Robot.elf",
    "Soldering_Iron": ELFMULTIFUZZ_ROOT / "P2IM" / "Soldering_Iron" / "Soldering_Iron.elf",
    "Steering_Control": ELFMULTIFUZZ_ROOT / "P2IM" / "Steering_Control" / "Steering_Control.elf",
}

P2IM_THREE = ("CNC", "Gateway", "PLC")

ABLATIONS: dict[str, list[str]] = {
    "full": [],
    "no_semantic_obligation_v2": ["--disable-semantic-obligation-stages"],
    "no_scoped_replay_v2": ["--disable-scoped-replay-stages"],
    "no_context_event_replay": ["--disable-context-event-replay-stages"],
    # r42：C2 配对对照——只替换候选生成步骤（固定通用策略，无符号问题）；
    # 目标选择/输入定位/出现身份/范围恢复/重放验证全部保留。
    "no_constraint_guided_candidates": ["--disable-constraint-guided-candidates"],
    # r42：与 full 成对的关 LLM 配套臂（历史 full 在 LLM.yaml 存在时开 LLM；
    # 差异必须能单独归因到候选生成，而不是混入 LLM 影响）。
    "full_llm_off": [],
    # Legacy implementation-level cuts remain available only for diagnostics.
    "no_diagnostic_replay": ["--disable-diagnostic-replay-stages"],
    "no_branch_reservoir": ["--disable-branch-reservoir-stages"],
    "no_isr": ["--disable-contextual-isr-stages"],
    "no_frontier": ["--disable-frontier-stages"],
    "no_contextual_isr": ["--disable-contextual-isr-stages"],
    "no_direct_stream_thread": ["--disable-direct-stream-thread-stages"],
}

# Per-variant environment overrides.  full_llm_off must actually switch the
# LLM off: LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS=0 alone means "no cap" in
# llm_guide, so the real teeth are in LSGEMU_DISABLE_LLM=1 (config load gate).
ABLATION_ENV_OVERRIDES: dict[str, dict[str, str]] = {
    "full_llm_off": {
        "LSGEMU_DISABLE_LLM": "1",
        "LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS": "0",
    },
}

DEFAULT_RQ2_ABLATIONS = (
    "full",
    "full_llm_off",
    "no_constraint_guided_candidates",
    "no_semantic_obligation_v2",
    "no_scoped_replay_v2",
    "no_context_event_replay",
)

TIME_THRESHOLDS = (50, 60, 70, 80, 90, 95)
BYTES_PER_GIB = 1024 ** 3


def _config_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


EVALUATION_DEFAULTS = _config_mapping(DEPLOYMENT_CONFIG.get("evaluation"))
SCHEDULER_DEFAULTS = _config_mapping(EVALUATION_DEFAULTS.get("scheduler"))


def _configured_int(mapping: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(mapping.get(key, fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _configured_float(mapping: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(mapping.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


@dataclass(frozen=True)
class Target:
    name: str
    firmware: Path
    entry_point: int | None = None
    load_base: int | None = None


@dataclass(frozen=True)
class RunSpec:
    rq: str
    target: Target
    variant: str
    output_dir: Path
    minutes: float
    extra_args: tuple[str, ...]
    env_overrides: dict[str, str]
    component_summary: bool = False


@dataclass
class RunningSpecProcess:
    spec: RunSpec
    cmd: list[str]
    proc: subprocess.Popen
    log_handle: IO[str]
    command_record: dict[str, Any]
    started_monotonic: float
    timeout_seconds: int
    peak_tree_rss_bytes: int = 0
    terminated_for_memory: bool = False
    timed_out: bool = False
    termination_reason: str = ""
    terminate_requested_at: float | None = None
    killed_after_grace: bool = False


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    atomic_json_dump(payload, path, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open() as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        info[key] = int(parts[1]) * 1024
                    except ValueError:
                        continue
    except OSError:
        return {}
    return info


def available_memory_bytes() -> int:
    info = read_meminfo()
    if not info:
        return 0
    return int(info.get("MemAvailable") or info.get("MemFree") or 0)


def swap_used_bytes() -> int:
    info = read_meminfo()
    if not info:
        return 0
    return max(0, int(info.get("SwapTotal", 0)) - int(info.get("SwapFree", 0)))


def format_gib(value: int | float) -> str:
    return f"{float(value) / BYTES_PER_GIB:.1f}GiB"


def current_lsgemu_tree_rss_bytes(proc: subprocess.Popen) -> int:
    root_pid = int(proc.pid)
    ppid_by_pid: dict[int, int] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            pid = int(item.name)
            ppid = 0
            with (item / "status").open() as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            ppid_by_pid[pid] = ppid
        except Exception:
            continue
    children: dict[int, list[int]] = {}
    for pid, ppid in ppid_by_pid.items():
        children.setdefault(ppid, []).append(pid)
    pids = [root_pid]
    index = 0
    while index < len(pids):
        pids.extend(children.get(pids[index], []))
        index += 1
    total = 0
    for pid in pids:
        try:
            with Path(f"/proc/{pid}/status").open() as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
                        break
        except Exception:
            continue
    return total


def terminate_process_tree(proc: subprocess.Popen, *, timeout: float = 8.0) -> bool:
    if proc.poll() is not None:
        return False
    children: list[int] = []
    try:
        out = subprocess.check_output(["pgrep", "-P", str(proc.pid)], text=True, timeout=2)
        children = [int(item) for item in out.split() if item.strip().isdigit()]
    except Exception:
        children = []
    for pid in children:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    try:
        proc.terminate()
    except OSError:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        time.sleep(0.2)
    for pid in children:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    return True


def process_group_kwargs() -> dict[str, Any]:
    if hasattr(os, "setsid"):
        return {"preexec_fn": os.setsid}
    return {}


def terminate_process_group(proc: subprocess.Popen, *, timeout: float = 8.0) -> bool:
    if proc.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        return terminate_process_tree(proc, timeout=timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        terminate_process_tree(proc, timeout=1.0)
    return True


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "--"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def load_manifest(path: Path) -> list[Target]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    targets: list[Target] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item #{index + 1} must be an object")
        name = str(item.get("name") or Path(str(item.get("firmware") or "")).stem)
        firmware = Path(str(item.get("firmware") or "")).expanduser()
        if not firmware:
            raise ValueError(f"manifest item #{index + 1} lacks firmware")
        entry_point = item.get("entry_point")
        load_base = item.get("load_base")
        targets.append(
            Target(
                name=sanitize_name(name),
                firmware=firmware,
                entry_point=parse_optional_int(entry_point),
                load_base=parse_optional_int(load_base),
            )
        )
    return targets


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned or "target"


def discover_elfmultifuzz_targets(root: Path = ELFMULTIFUZZ_ROOT) -> list[Target]:
    targets: list[Target] = []
    if not root.exists():
        return targets
    for firmware in sorted(root.rglob("*.elf")):
        rel = firmware.relative_to(root)
        if len(rel.parts) >= 2:
            name = sanitize_name("__".join(rel.parts[:-1]))
        else:
            name = sanitize_name(firmware.stem)
        targets.append(Target(name=name, firmware=firmware))
    return targets


def built_in_targets_for_selection() -> dict[str, Path]:
    targets = dict(P2IM_TARGETS)
    for target in discover_elfmultifuzz_targets():
        targets.setdefault(target.name, target.firmware)
        targets.setdefault(target.firmware.stem, target.firmware)
    return targets


def select_targets(args: argparse.Namespace) -> list[Target]:
    if args.manifest:
        targets = load_manifest(Path(args.manifest))
        if args.target:
            wanted = {sanitize_name(item) for item in args.target}
            targets = [target for target in targets if target.name in wanted or target.firmware.stem in wanted]
        if not targets:
            raise SystemExit("no targets selected")
        missing = [target for target in targets if not target.firmware.exists()]
        if missing and not args.allow_missing_targets:
            details = "\n".join(f"- {target.name}: {target.firmware}" for target in missing)
            raise SystemExit(f"missing firmware targets:\n{details}")
        return targets

    if args.target:
        targets = []
        builtins = built_in_targets_for_selection()
        for raw_name in args.target:
            name = sanitize_name(raw_name)
            if name not in builtins:
                known = ", ".join(sorted(builtins)[:64])
                raise SystemExit(f"unknown built-in target '{raw_name}'. Known targets: {known}")
            targets.append(Target(name, builtins[name]))
    elif args.target_suite == "elfmultifuzz-23":
        targets = discover_elfmultifuzz_targets()
        if len(targets) != 23 and not args.allow_non23_elfmultifuzz:
            raise SystemExit(
                f"expected 23 elfmultifuzz ELF targets under {ELFMULTIFUZZ_ROOT}, found {len(targets)}. "
                "Use --allow-non23-elfmultifuzz to continue."
            )
    elif args.target_suite == "p2im-all":
        targets = [Target(name, path) for name, path in P2IM_TARGETS.items()]
    else:
        targets = [Target(name, P2IM_TARGETS[name]) for name in P2IM_THREE]

    if not targets:
        raise SystemExit("no targets selected")

    missing = [target for target in targets if not target.firmware.exists()]
    if missing and not args.allow_missing_targets:
        details = "\n".join(f"- {target.name}: {target.firmware}" for target in missing)
        raise SystemExit(f"missing firmware targets:\n{details}")
    return targets


def latest_report(output_dir: Path, firmware: Path) -> Path | None:
    patterns = [
        f"{firmware.stem}*_interleaved_report.json",
        f"{firmware.stem}*_lsgemu_result.json",
        "*_interleaved_report.json",
        "*_lsgemu_result.json",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(output_dir.glob(pattern))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    return candidates[-1] if candidates else None


def report_progress_path(report: dict[str, Any], output_dir: Path, firmware: Path) -> Path | None:
    raw = report.get("progress_jsonl_file")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    return latest_progress_path(output_dir, firmware)


def latest_progress_path(output_dir: Path, firmware: Path) -> Path | None:
    candidates = sorted(output_dir.glob(f"{firmware.stem}*_coverage_progress.jsonl"))
    if candidates:
        return candidates[-1]
    candidates = sorted(output_dir.glob("*_coverage_progress.jsonl"))
    return candidates[-1] if candidates else None


def read_last_jsonl_record(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buffer = bytearray()
        while pos > 0:
            step = min(8192, pos)
            pos -= step
            f.seek(pos)
            chunk = f.read(step)
            buffer[:0] = chunk
            lines = buffer.splitlines()
            if len(lines) > 1 or pos == 0:
                for line in reversed(lines):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    return item if isinstance(item, dict) else {}
        return {}


def report_llm_history_path(report: dict[str, Any], output_dir: Path, firmware: Path) -> Path | None:
    raw = report.get("llm_history_file")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    candidates = sorted(output_dir.glob(f"{firmware.stem}*_llm_history.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted(output_dir.glob("*_llm_history.json"))
    return candidates[-1] if candidates else None


def build_lsgemu_command(
    target: Target,
    output_dir: Path,
    minutes: float,
    *,
    extra_args: Iterable[str] = (),
    run_profile: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(LSGEMU),
        str(target.firmware),
        "--time",
        str(float(minutes)),
        "--max-instructions",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    if run_profile:
        cmd.extend(["--run-profile", run_profile])
    if target.entry_point is not None:
        cmd.extend(["--entry-point", hex(target.entry_point & 0xFFFFFFFF)])
    if target.load_base is not None:
        cmd.extend(["--load-base", hex(target.load_base & 0xFFFFFFFF)])
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def write_command_record(cmd: list[str], output_dir: Path, env_overrides: dict[str, str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_record = {
        "command": cmd,
        "command_text": " ".join(shlex.quote(part) for part in cmd),
        "cwd": str(PROJECT_ROOT),
        "env_overrides": dict(env_overrides),
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "command.json", command_record)
    return command_record


def completed_command_result(
    command_record: dict[str, Any],
    *,
    returncode: int,
    elapsed_seconds: float,
    skipped_existing: bool,
    dry_run: bool,
    report: Path | None,
    timed_out: bool = False,
    memory_guard_terminated: bool = False,
    termination_reason: str = "",
    peak_tree_rss_bytes: int = 0,
) -> dict[str, Any]:
    result = {
        **command_record,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timed_out": timed_out,
        "skipped_existing": skipped_existing,
        "dry_run": dry_run,
        "report": str(report) if report else "",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "memory_guard_terminated": memory_guard_terminated,
        "termination_reason": termination_reason,
        "peak_tree_rss_bytes": int(peak_tree_rss_bytes),
        "peak_tree_rss_gib": round(float(peak_tree_rss_bytes) / BYTES_PER_GIB, 3) if peak_tree_rss_bytes else 0.0,
    }
    write_json(Path(str(command_record["output_dir"])) / "run_result.json", result)
    return result


def run_command(
    cmd: list[str],
    *,
    output_dir: Path,
    env_overrides: dict[str, str],
    dry_run: bool,
    skip_existing_report: bool,
    expected_firmware: Path,
    external_timeout_seconds: int,
) -> dict[str, Any]:
    existing_report = latest_report(output_dir, expected_firmware)
    command_record = write_command_record(cmd, output_dir, env_overrides)

    if skip_existing_report and existing_report and not dry_run:
        return completed_command_result(
            command_record,
            returncode=0,
            elapsed_seconds=0.0,
            skipped_existing=True,
            dry_run=False,
            report=existing_report,
        )

    if dry_run:
        return completed_command_result(
            command_record,
            returncode=0,
            elapsed_seconds=0.0,
            skipped_existing=False,
            dry_run=True,
            report=existing_report,
        )

    env = os.environ.copy()
    env.update(env_overrides)
    start = time.time()
    returncode = 0
    timed_out = False
    with (output_dir / "run.log").open("w") as log:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=external_timeout_seconds if external_timeout_seconds > 0 else None,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log.write(f"\n[run_rq_experiments] external timeout after {external_timeout_seconds}s\n")
    elapsed = time.time() - start
    report = latest_report(output_dir, expected_firmware)
    return completed_command_result(
        command_record,
        returncode=returncode,
        elapsed_seconds=elapsed,
        skipped_existing=False,
        dry_run=False,
        report=report,
        timed_out=timed_out,
    )


def execute_spec(args: argparse.Namespace, spec: RunSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cmd = build_lsgemu_command(
        spec.target,
        spec.output_dir,
        spec.minutes,
        extra_args=spec.extra_args,
        run_profile=args.run_profile,
    )
    print(
        f"[{spec.rq}] {spec.variant}/{spec.target.name}: "
        f"{' '.join(shlex.quote(part) for part in cmd)}",
        flush=True,
    )
    run_result = run_command(
        cmd,
        output_dir=spec.output_dir,
        env_overrides=spec.env_overrides,
        dry_run=args.dry_run,
        skip_existing_report=not args.rerun,
        expected_firmware=spec.target.firmware,
        external_timeout_seconds=external_timeout(spec.minutes, args.wallclock_grace_minutes),
    )
    row = extract_report_summary(
        rq=spec.rq,
        target=spec.target,
        variant=spec.variant,
        output_dir=spec.output_dir,
        run_result=run_result,
        requested_minutes=spec.minutes,
    )
    if spec.component_summary:
        add_component_summary(row)
    return row, progress_rows_for_report(row)


def prepare_spec_process(args: argparse.Namespace, spec: RunSpec) -> RunningSpecProcess | tuple[dict[str, Any], list[dict[str, Any]]] | None:
    cmd = build_lsgemu_command(
        spec.target,
        spec.output_dir,
        spec.minutes,
        extra_args=spec.extra_args,
        run_profile=args.run_profile,
    )
    print(
        f"[{spec.rq}] {spec.variant}/{spec.target.name}: "
        f"{' '.join(shlex.quote(part) for part in cmd)}",
        flush=True,
    )
    existing_report = latest_report(spec.output_dir, spec.target.firmware)
    command_record = write_command_record(cmd, spec.output_dir, spec.env_overrides)
    if not args.rerun and existing_report and not args.dry_run:
        run_result = completed_command_result(
            command_record,
            returncode=0,
            elapsed_seconds=0.0,
            skipped_existing=True,
            dry_run=False,
            report=existing_report,
        )
        row = extract_report_summary(
            rq=spec.rq,
            target=spec.target,
            variant=spec.variant,
            output_dir=spec.output_dir,
            run_result=run_result,
            requested_minutes=spec.minutes,
        )
        if spec.component_summary:
            add_component_summary(row)
        return row, progress_rows_for_report(row)
    if args.dry_run:
        run_result = completed_command_result(
            command_record,
            returncode=0,
            elapsed_seconds=0.0,
            skipped_existing=False,
            dry_run=True,
            report=existing_report,
        )
        row = extract_report_summary(
            rq=spec.rq,
            target=spec.target,
            variant=spec.variant,
            output_dir=spec.output_dir,
            run_result=run_result,
            requested_minutes=spec.minutes,
        )
        if spec.component_summary:
            add_component_summary(row)
        return row, progress_rows_for_report(row)

    env = os.environ.copy()
    env.update(spec.env_overrides)
    log_handle = (spec.output_dir / "run.log").open("w")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **process_group_kwargs(),
        )
    except Exception:
        log_handle.close()
        raise
    return RunningSpecProcess(
        spec=spec,
        cmd=cmd,
        proc=proc,
        log_handle=log_handle,
        command_record=command_record,
        started_monotonic=time.time(),
        timeout_seconds=external_timeout(spec.minutes, args.wallclock_grace_minutes),
    )


def finish_spec_process(running: RunningSpecProcess) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proc = running.proc
    if proc.poll() is None:
        proc.wait()
    elapsed = time.time() - running.started_monotonic
    try:
        running.log_handle.flush()
    except Exception:
        pass
    try:
        running.log_handle.close()
    except Exception:
        pass
    report = latest_report(running.spec.output_dir, running.spec.target.firmware)
    run_result = completed_command_result(
        running.command_record,
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        elapsed_seconds=elapsed,
        skipped_existing=False,
        dry_run=False,
        report=report,
        timed_out=running.timed_out,
        memory_guard_terminated=running.terminated_for_memory,
        termination_reason=running.termination_reason,
        peak_tree_rss_bytes=running.peak_tree_rss_bytes,
    )
    row = extract_report_summary(
        rq=running.spec.rq,
        target=running.spec.target,
        variant=running.spec.variant,
        output_dir=running.spec.output_dir,
        run_result=run_result,
        requested_minutes=running.spec.minutes,
    )
    if running.terminated_for_memory:
        row["status"] = "memory_guard_terminated"
        row["termination_reason"] = running.termination_reason
    elif running.timed_out:
        row["timed_out"] = True
    row["peak_tree_rss_bytes"] = running.peak_tree_rss_bytes
    row["peak_tree_rss_gib"] = round(float(running.peak_tree_rss_bytes) / BYTES_PER_GIB, 3) if running.peak_tree_rss_bytes else 0.0
    if running.spec.component_summary:
        add_component_summary(row)
    return row, progress_rows_for_report(row)


def compute_observed_per_job_bytes(args: argparse.Namespace, running: list[RunningSpecProcess]) -> int:
    configured = max(1.0, float(args.scheduler_per_job_memory_gib)) * BYTES_PER_GIB
    peaks = [item.peak_tree_rss_bytes for item in running if item.peak_tree_rss_bytes > 0]
    retained_peak = max(
        0,
        int(getattr(args, "_scheduler_observed_peak_bytes", 0) or 0),
    )
    if retained_peak > 0:
        peaks.append(retained_peak)
    if not peaks:
        return int(configured)
    return int(max(configured, max(peaks) * float(args.scheduler_peak_multiplier)))


def can_start_more(args: argparse.Namespace, running: list[RunningSpecProcess], last_start_time: float) -> tuple[bool, str]:
    max_running = max(1, int(args.jobs))
    if len(running) >= max_running:
        return False, "jobs_limit"
    ramp_limit = min(max_running, max(1, int(args.scheduler_initial_jobs)) + int((time.time() - last_start_time) // max(1, int(args.scheduler_ramp_seconds))))
    if len(running) >= ramp_limit:
        return False, "ramp_limit"
    available = available_memory_bytes()
    if available <= 0:
        return True, "meminfo_unavailable"
    min_free = max(0.0, float(args.scheduler_min_free_memory_gib)) * BYTES_PER_GIB
    projected = compute_observed_per_job_bytes(args, running)
    if available - projected < min_free:
        return False, f"memory_hold available={format_gib(available)} projected_job={format_gib(projected)} min_free={format_gib(min_free)}"
    swap_used = swap_used_bytes()
    max_swap = max(0.0, float(args.scheduler_max_swap_used_gib)) * BYTES_PER_GIB
    if max_swap > 0 and swap_used > max_swap:
        return False, f"swap_hold swap_used={format_gib(swap_used)} max_swap={format_gib(max_swap)}"
    return True, "ok"


def update_running_memory(args: argparse.Namespace, running: list[RunningSpecProcess]) -> None:
    for item in running:
        rss = current_lsgemu_tree_rss_bytes(item.proc)
        if rss > item.peak_tree_rss_bytes:
            item.peak_tree_rss_bytes = rss
        if item.peak_tree_rss_bytes > int(
            getattr(args, "_scheduler_observed_peak_bytes", 0) or 0
        ):
            args._scheduler_observed_peak_bytes = int(item.peak_tree_rss_bytes)


def maybe_apply_memory_guard(args: argparse.Namespace, running: list[RunningSpecProcess]) -> None:
    if not running:
        return
    available = available_memory_bytes()
    kill_below = float(args.scheduler_kill_below_free_memory_gib) * BYTES_PER_GIB
    swap_used = swap_used_bytes()
    kill_swap = float(args.scheduler_kill_above_swap_used_gib) * BYTES_PER_GIB
    memory_low = kill_below > 0 and available > 0 and available < kill_below
    swap_high = kill_swap > 0 and swap_used > kill_swap
    if not memory_low and not swap_high:
        return
    victim = max(running, key=lambda item: (item.peak_tree_rss_bytes, time.time() - item.started_monotonic))
    if victim.terminate_requested_at is not None:
        return
    if memory_low:
        reason = f"available_memory_below_guard available={format_gib(available)} guard={format_gib(kill_below)}"
    else:
        reason = f"swap_used_above_guard swap_used={format_gib(swap_used)} guard={format_gib(kill_swap)}"
    victim.terminated_for_memory = True
    victim.termination_reason = reason
    victim.terminate_requested_at = time.time()
    try:
        victim.log_handle.write(f"\n[run_rq_experiments] memory guard terminating run: {reason}\n")
        victim.log_handle.flush()
    except Exception:
        pass
    victim.killed_after_grace = terminate_process_group(victim.proc)
    print(f"[scheduler] memory guard terminated {victim.spec.variant}/{victim.spec.target.name}: {reason}", flush=True)


def maybe_apply_timeouts(running: list[RunningSpecProcess]) -> None:
    now = time.time()
    for item in running:
        if item.timeout_seconds <= 0 or item.proc.poll() is not None:
            continue
        if now - item.started_monotonic <= item.timeout_seconds:
            continue
        if item.terminate_requested_at is not None:
            continue
        item.timed_out = True
        item.termination_reason = f"external timeout after {item.timeout_seconds}s"
        item.terminate_requested_at = now
        try:
            item.log_handle.write(f"\n[run_rq_experiments] {item.termination_reason}\n")
            item.log_handle.flush()
        except Exception:
            pass
        item.killed_after_grace = terminate_process_group(item.proc)


def execute_specs_guarded(
    args: argparse.Namespace,
    specs: list[RunSpec],
    *,
    on_result,
) -> None:
    pending: deque[RunSpec] = deque(specs)
    running: list[RunningSpecProcess] = []
    completed = 0
    total = len(specs)
    scheduler_started_at = time.time()
    last_status_at = 0.0

    print(
        "[scheduler] guarded subprocess mode "
        f"tasks={total} jobs={args.jobs} initial_jobs={args.scheduler_initial_jobs} "
        f"min_free={args.scheduler_min_free_memory_gib}GiB per_job={args.scheduler_per_job_memory_gib}GiB "
        f"kill_below={args.scheduler_kill_below_free_memory_gib}GiB",
        flush=True,
    )

    try:
        while pending or running:
            update_running_memory(args, running)
            maybe_apply_timeouts(running)
            maybe_apply_memory_guard(args, running)

            still_running: list[RunningSpecProcess] = []
            for item in running:
                if item.proc.poll() is None:
                    still_running.append(item)
                    continue
                row, progress = finish_spec_process(item)
                completed += 1
                on_result(row, progress)
                print(
                    f"[scheduler] completed {completed}/{total} {item.spec.variant}/{item.spec.target.name} "
                    f"rc={row.get('returncode')} status={row.get('status')} peak_rss={row.get('peak_tree_rss_gib')}GiB",
                    flush=True,
                )
            running = still_running

            while pending:
                can_start, reason = can_start_more(args, running, scheduler_started_at)
                if not can_start:
                    if time.time() - last_status_at >= max(5, int(args.scheduler_status_interval_seconds)):
                        print(
                            f"[scheduler] hold start reason={reason} running={len(running)} pending={len(pending)} "
                            f"available={format_gib(available_memory_bytes())} swap_used={format_gib(swap_used_bytes())}",
                            flush=True,
                        )
                        last_status_at = time.time()
                    break
                spec = pending.popleft()
                try:
                    prepared = prepare_spec_process(args, spec)
                except Exception as exc:
                    row = {
                        "rq": spec.rq,
                        "target": spec.target.name,
                        "variant": spec.variant,
                        "firmware": str(spec.target.firmware),
                        "status": "runner_exception",
                        "error": str(exc),
                        "output_dir": str(spec.output_dir),
                        "requested_minutes": spec.minutes,
                    }
                    on_result(row, [])
                    completed += 1
                    continue
                if isinstance(prepared, RunningSpecProcess):
                    running.append(prepared)
                    print(
                        f"[scheduler] started {prepared.spec.variant}/{prepared.spec.target.name} "
                        f"pid={prepared.proc.pid} running={len(running)} pending={len(pending)} "
                        f"available={format_gib(available_memory_bytes())}",
                        flush=True,
                    )
                elif prepared is not None:
                    row, progress = prepared
                    on_result(row, progress)
                    completed += 1

            if running or pending:
                time.sleep(max(1.0, float(args.scheduler_poll_seconds)))
    except KeyboardInterrupt:
        print("[scheduler] interrupted; terminating running LSGEmu subprocesses", flush=True)
        for item in running:
            item.terminated_for_memory = True
            item.termination_reason = "scheduler interrupted"
            terminate_process_group(item.proc)
        raise


def execute_specs(
    args: argparse.Namespace,
    specs: list[RunSpec],
    *,
    on_result,
) -> None:
    if args.dry_run:
        print(f"[scheduler] dry-run tasks={len(specs)}; resource guards bypassed", flush=True)
        for spec in specs:
            prepared = prepare_spec_process(args, spec)
            if isinstance(prepared, RunningSpecProcess):
                raise RuntimeError("dry-run unexpectedly started a subprocess")
            if prepared is not None:
                row, progress = prepared
                on_result(row, progress)
        return
    execute_specs_guarded(args, specs, on_result=on_result)


def extract_report_summary(
    *,
    rq: str,
    target: Target,
    variant: str,
    output_dir: Path,
    run_result: dict[str, Any],
    requested_minutes: float,
) -> dict[str, Any]:
    report_text = str(run_result.get("report") or "").strip()
    report_path = Path(report_text) if report_text else None
    row: dict[str, Any] = {
        "rq": rq,
        "target": target.name,
        "variant": variant,
        "firmware": str(target.firmware),
        "returncode": run_result.get("returncode"),
        "elapsed_seconds": run_result.get("elapsed_seconds"),
        "requested_minutes": requested_minutes,
        "skipped_existing": bool(run_result.get("skipped_existing")),
        "dry_run": bool(run_result.get("dry_run")),
        "report": str(report_path) if report_path is not None else "",
        "output_dir": str(output_dir),
        "memory_guard_terminated": bool(run_result.get("memory_guard_terminated")),
        "termination_reason": run_result.get("termination_reason") or "",
        "timed_out": bool(run_result.get("timed_out")),
        "peak_tree_rss_bytes": safe_int(run_result.get("peak_tree_rss_bytes")),
        "peak_tree_rss_gib": safe_float(run_result.get("peak_tree_rss_gib")),
    }
    progress_path = latest_progress_path(output_dir, target.firmware)
    if progress_path:
        row["progress_jsonl_file"] = str(progress_path)
        add_last_observed_progress(row, progress_path)
    if report_path is None or not report_path.is_file():
        row["status"] = "missing_report"
        apply_outcome_class(row)
        return row

    report = load_json(report_path)
    progress_path = report_progress_path(report, output_dir, target.firmware)
    llm_path = report_llm_history_path(report, output_dir, target.firmware)
    row.update(
        {
            "status": "ok" if safe_int(run_result.get("returncode")) == 0 else "nonzero_returncode_with_report",
            "covered_bbs": safe_int(report.get("covered_bbs")),
            "total_bbs": safe_int(report.get("total_bbs")),
            "coverage_rate": safe_float(report.get("coverage_rate")),
            "valid_covered_bbs": safe_int(report.get("valid_covered_bbs")),
            "valid_total_bbs": safe_int(report.get("valid_total_bbs")),
            "valid_coverage_rate": safe_float(report.get("valid_coverage_rate")),
            "execution_time_seconds": safe_float(report.get("execution_time_seconds")),
            "coverage_source": report.get("coverage_source"),
            "coverage_entry_derived": report.get("coverage_entry_derived"),
            "strict_real_entry_replayable": report.get("strict_real_entry_replayable"),
            "run_profile_hash": report.get("run_profile_hash"),
            "progress_jsonl_file": str(progress_path) if progress_path else "",
            "llm_history_file": str(llm_path) if llm_path else "",
            "branch_attempts": safe_int(
                ((report.get("environment_feasibility_summary") or {}).get("counts") or {}).get("attempts")
            ),
            "branch_attempts_accepted": safe_int(
                ((report.get("environment_feasibility_summary") or {}).get("counts") or {}).get("accepted")
            ),
            "branch_attempts_rejected": safe_int(
                ((report.get("environment_feasibility_summary") or {}).get("counts") or {}).get("rejected")
            ),
        }
    )
    if progress_path:
        add_last_observed_progress(row, progress_path)
    apply_outcome_class(row)
    return row


def add_last_observed_progress(row: dict[str, Any], progress_path: Path) -> None:
    item = read_last_jsonl_record(progress_path)
    if not item:
        return
    row.update(
        {
            "last_observed_elapsed_seconds": safe_float(item.get("elapsed_seconds")),
            "last_observed_stage": item.get("stage"),
            "last_observed_event": item.get("event"),
            "last_observed_valid_covered_bbs": safe_int(item.get("valid_covered_bbs")),
            "last_observed_valid_total_bbs": safe_int(item.get("valid_total_bbs")),
            "last_observed_valid_coverage_rate": safe_float(item.get("valid_coverage_rate")),
            "last_observed_covered_bbs": safe_int(item.get("covered_bbs")),
            "last_observed_total_bbs": safe_int(item.get("total_bbs")),
            "last_observed_coverage_rate": safe_float(item.get("coverage_rate")),
            "last_observed_wallclock_time": item.get("wallclock_time"),
            "last_observed_strict_real_entry_replayable": item.get("strict_real_entry_replayable"),
        }
    )


def apply_outcome_class(row: dict[str, Any]) -> None:
    returncode = safe_int(row.get("returncode"), default=0)
    status = str(row.get("status") or "")
    has_report = bool(str(row.get("report") or ""))
    if bool(row.get("dry_run")):
        outcome = "dry_run"
    elif bool(row.get("memory_guard_terminated")) or status == "memory_guard_terminated":
        outcome = "resource_censored"
    elif bool(row.get("timed_out")) or returncode == 124:
        outcome = "timeout_censored"
    elif returncode == -11 or returncode == 139:
        outcome = "segmentation_fault"
    elif returncode != 0:
        outcome = "nonzero_exit"
    elif status == "ok" and has_report:
        outcome = "completed"
    elif status == "missing_report" and row.get("progress_jsonl_file"):
        outcome = "incomplete_no_report"
    else:
        outcome = "unknown"
    row["outcome_class"] = outcome
    row["is_final_coverage"] = outcome == "completed"
    row["is_censored"] = outcome in {"resource_censored", "timeout_censored", "incomplete_no_report"}


def progress_rows_for_report(row: dict[str, Any]) -> list[dict[str, Any]]:
    path_text = str(row.get("progress_jsonl_file") or "")
    if not path_text:
        return []
    path = Path(path_text)
    rows = []
    for item in read_jsonl(path):
        rows.append(
            {
                "rq": row.get("rq"),
                "target": row.get("target"),
                "variant": row.get("variant"),
                "elapsed_seconds": safe_float(item.get("elapsed_seconds")),
                "event": item.get("event"),
                "stage": item.get("stage"),
                "valid_covered_bbs": safe_int(item.get("valid_covered_bbs")),
                "valid_total_bbs": safe_int(item.get("valid_total_bbs")),
                "valid_coverage_rate": safe_float(item.get("valid_coverage_rate")),
                "covered_bbs": safe_int(item.get("covered_bbs")),
                "total_bbs": safe_int(item.get("total_bbs")),
                "coverage_rate": safe_float(item.get("coverage_rate")),
                "strict_real_entry_replayable": item.get("strict_real_entry_replayable"),
            }
        )
    return rows


def time_to_thresholds(progress_rows: list[dict[str, Any]], final_row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ordered = sorted(progress_rows, key=lambda item: safe_float(item.get("elapsed_seconds")))
    for threshold in TIME_THRESHOLDS:
        reached = None
        for item in ordered:
            if safe_float(item.get("valid_coverage_rate")) >= threshold:
                reached = safe_float(item.get("elapsed_seconds"))
                break
        result[f"time_to_{threshold}_pct_seconds"] = reached
    final_rate = safe_float(final_row.get("valid_coverage_rate"))
    result["final_valid_coverage_rate"] = final_rate
    result["completed_before_requested_budget"] = (
        safe_float(final_row.get("execution_time_seconds") or final_row.get("elapsed_seconds"))
        < safe_float(final_row.get("requested_minutes")) * 60.0
    )
    return result


def build_rq1_outcome_audit(
    targets: list[Target],
    rq_root: Path,
    requested_minutes: float,
    summary_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summary_by_target = {str(row.get("target")): row for row in (summary_rows or [])}
    for target in targets:
        output_dir = rq_root / "runs" / target.name
        run_result_path = output_dir / "run_result.json"
        run_result: dict[str, Any] = {}
        if run_result_path.exists():
            try:
                payload = load_json(run_result_path)
                if isinstance(payload, dict):
                    run_result = payload
            except Exception as exc:
                run_result = {"returncode": "", "run_result_load_error": str(exc)}
        if not run_result:
            latest = latest_report(output_dir, target.firmware)
            run_result = {
                "returncode": 0 if latest else "",
                "elapsed_seconds": "",
                "dry_run": False,
                "skipped_existing": False,
                "report": str(latest) if latest else "",
            }
        row = extract_report_summary(
            rq="RQ1",
            target=target,
            variant="full",
            output_dir=output_dir,
            run_result=run_result,
            requested_minutes=requested_minutes,
        )
        previous = summary_by_target.get(target.name)
        if previous:
            for key in ("peak_tree_rss_bytes", "peak_tree_rss_gib", "termination_reason"):
                if key not in row or row.get(key) in {"", 0, 0.0, None}:
                    row[key] = previous.get(key)
        row["run_result_exists"] = run_result_path.exists()
        row["report_exists"] = bool(str(row.get("report") or "")) and Path(str(row.get("report"))).exists()
        row["progress_exists"] = bool(str(row.get("progress_jsonl_file") or "")) and Path(str(row.get("progress_jsonl_file"))).exists()
        refine_rq1_outcome(row)
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        outcome = str(row.get("outcome_class") or "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
    completed_rows = [row for row in rows if row.get("outcome_class") == "completed"]
    rerun_rows = [row for row in rows if should_rerun_rq1(row)]
    live_rows = [row for row in rows if row.get("outcome_class") == "still_running_or_orphaned"]
    return {
        "rows": rows,
        "counts": counts,
        "completed_rows": completed_rows,
        "rerun_rows": rerun_rows,
        "live_rows": live_rows,
        "target_count": len(targets),
        "completed_count": len(completed_rows),
        "rerun_count": len(rerun_rows),
        "live_count": len(live_rows),
    }


def refine_rq1_outcome(row: dict[str, Any]) -> None:
    if not row.get("run_result_exists") and row.get("progress_exists") and not row.get("report_exists"):
        row["outcome_class"] = "still_running_or_orphaned"
        row["is_final_coverage"] = False
        row["is_censored"] = True
    if is_stuck_zero_coverage(row):
        row["outcome_class"] = "stuck_zero_coverage"
        row["is_final_coverage"] = False
        row["is_censored"] = True
        row["stuck_reason"] = "baseline_zero_coverage_after_long_runtime"
    if row.get("outcome_class") == "completed":
        row["paper_valid_covered_bbs"] = row.get("valid_covered_bbs")
        row["paper_valid_total_bbs"] = row.get("valid_total_bbs")
        row["paper_valid_coverage_rate"] = row.get("valid_coverage_rate")
    else:
        row["paper_valid_covered_bbs"] = ""
        row["paper_valid_total_bbs"] = row.get("last_observed_valid_total_bbs") or row.get("valid_total_bbs") or ""
        row["paper_valid_coverage_rate"] = ""
    row["rerun_reason"] = rq1_rerun_reason(row)


def is_stuck_zero_coverage(row: dict[str, Any]) -> bool:
    elapsed = safe_float(row.get("last_observed_elapsed_seconds") or row.get("elapsed_seconds"))
    valid_covered = safe_int(row.get("last_observed_valid_covered_bbs") or row.get("valid_covered_bbs"))
    stage = str(row.get("last_observed_stage") or "").lower()
    return elapsed >= 3600.0 and valid_covered == 0 and ("baseline" in stage or not stage)


def rq1_rerun_reason(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome_class") or "")
    if outcome == "resource_censored":
        return "resource_censored_memory_guard"
    if outcome == "segmentation_fault":
        return "segmentation_fault_needs_debug"
    if outcome == "stuck_zero_coverage":
        return "stuck_zero_coverage_needs_bounded_baseline_rerun"
    if outcome == "timeout_censored":
        return "timeout_censored"
    if outcome == "nonzero_exit":
        return "nonzero_exit"
    if outcome == "incomplete_no_report":
        return "incomplete_no_final_report"
    if outcome == "still_running_or_orphaned":
        return "still_running_or_orphaned_needs_completion_or_finalization"
    return ""


def should_rerun_rq1(row: dict[str, Any]) -> bool:
    return bool(row.get("rerun_reason"))


def write_rq1_outcome_artifacts(rq_root: Path, audit: dict[str, Any]) -> None:
    rows = list(audit.get("rows") or [])
    completed_rows = list(audit.get("completed_rows") or [])
    rerun_rows = list(audit.get("rerun_rows") or [])
    write_json(rq_root / "rq1_outcome_audit.json", audit)
    write_csv(rq_root / "rq1_outcome_audit.csv", rows)
    write_csv(rq_root / "rq1_paper_completed_only.csv", completed_rows)
    rerun_manifest = [
        {
            "name": row.get("target"),
            "firmware": row.get("firmware"),
            "reason": row.get("rerun_reason"),
            "last_observed_valid_coverage_rate": row.get("last_observed_valid_coverage_rate"),
            "last_observed_valid_covered_bbs": row.get("last_observed_valid_covered_bbs"),
            "last_observed_valid_total_bbs": row.get("last_observed_valid_total_bbs"),
            "peak_tree_rss_gib": row.get("peak_tree_rss_gib"),
        }
        for row in rerun_rows
    ]
    write_json(rq_root / "rq1_rerun_manifest.json", rerun_manifest)
    for reason_prefix, filename in [
        ("resource_censored", "rq1_rerun_resource_censored_manifest.json"),
        ("segmentation_fault", "rq1_debug_segmentation_fault_manifest.json"),
        ("stuck_zero_coverage", "rq1_debug_stuck_zero_coverage_manifest.json"),
        ("timeout_censored", "rq1_rerun_timeout_manifest.json"),
        ("nonzero_exit", "rq1_debug_nonzero_exit_manifest.json"),
        ("still_running_or_orphaned", "rq1_live_or_orphaned_manifest.json"),
    ]:
        subset = [
            item for item in rerun_manifest
            if str(item.get("reason") or "").startswith(reason_prefix)
        ]
        write_json(rq_root / filename, subset)
    (rq_root / "rq1_outcome_audit.md").write_text(build_rq1_outcome_markdown(audit))


def build_rq1_outcome_markdown(audit: dict[str, Any]) -> str:
    rows = list(audit.get("rows") or [])
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    lines = [
        "# RQ1 Outcome Audit",
        "",
        "Only `completed` rows are valid final RQ1 coverage. Resource-censored, crashed, stuck, and live rows are last-observed evidence, not final coverage.",
        "",
        "## Outcome Counts",
        "",
        markdown_table(["Outcome", "Count"], [[key, counts[key]] for key in sorted(counts)]),
        "",
        "## Per-Target Status",
        "",
        markdown_table(
            [
                "Target",
                "Outcome",
                "Paper cov.",
                "Last observed cov.",
                "Last stage",
                "Elapsed min",
                "Peak RSS",
                "Rerun reason",
            ],
            [
                [
                    row.get("target"),
                    row.get("outcome_class"),
                    f"{safe_float(row.get('paper_valid_coverage_rate')):.2f}%" if row.get("paper_valid_coverage_rate") not in {"", None} else "--",
                    f"{safe_float(row.get('last_observed_valid_coverage_rate')):.2f}%" if row.get("last_observed_valid_coverage_rate") not in {"", None} else "--",
                    row.get("last_observed_stage") or "--",
                    f"{safe_float(row.get('last_observed_elapsed_seconds') or row.get('elapsed_seconds')) / 60.0:.1f}",
                    f"{safe_float(row.get('peak_tree_rss_gib')):.1f}GiB" if row.get("peak_tree_rss_gib") not in {"", None} else "--",
                    row.get("rerun_reason") or "--",
                ]
                for row in rows
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def collect_reports_from_roots(roots: Iterable[Path]) -> list[Path]:
    reports: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        reports.extend(root.rglob("*_interleaved_report.json"))
        reports.extend(root.rglob("*_lsgemu_result.json"))
    return sorted(set(reports))


def iter_phase_dicts(report: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    phases = report.get("phase_metadata")
    if not isinstance(phases, dict):
        phases = report.get("phases")
    if not isinstance(phases, dict):
        return []
    return [(str(name), phase) for name, phase in phases.items() if isinstance(phase, dict)]


def source_count(counter: dict[str, Any], needle: str) -> int:
    return sum(safe_int(value) for key, value in counter.items() if needle in str(key).lower())


def audit_hypothesis_quality(report_paths: list[Path]) -> dict[str, Any]:
    totals: dict[str, int] = {
        "reports_scanned": 0,
        "reports_with_hypothesis_audit": 0,
        "hypothesis_records": 0,
        "hypotheses_generated": 0,
        "hypotheses_replayed": 0,
        "accepted_by_replay": 0,
        "persisted": 0,
        "scoped": 0,
        "rejected": 0,
        "new_bbs_vs_control": 0,
        "new_bbs_vs_phase_before": 0,
        "new_bbs_vs_global": 0,
        "llm_history_files": 0,
        "llm_history_entries": 0,
        "llm_method_entries": 0,
        "llm_actual_calls": 0,
        "llm_successful_values": 0,
        "llm_rejected_values": 0,
        "llm_errors": 0,
        "llm_derived_candidates_built": 0,
        "llm_derived_candidates_replayed": 0,
        "llm_derived_attempts": 0,
        "llm_derived_replayed_attempts": 0,
        "llm_derived_success_attempts": 0,
        "llm_derived_new_bbs_vs_control": 0,
        "llm_derived_constraints_persisted": 0,
        "llm_derived_constraints_scoped": 0,
        "environment_attempts": 0,
        "environment_accepted": 0,
        "environment_rejected": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    reject_reasons: dict[str, int] = {}
    case_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    def source_row(source: str) -> dict[str, int]:
        return by_source.setdefault(
            source,
            {
                "generated": 0,
                "replayed": 0,
                "accepted_by_replay": 0,
                "persisted": 0,
                "scoped": 0,
                "rejected": 0,
                "new_bbs_vs_control": 0,
                "new_bbs_vs_phase_before": 0,
                "new_bbs_vs_global": 0,
            },
        )

    for report_path in report_paths:
        try:
            report = load_json(report_path)
        except Exception:
            continue
        totals["reports_scanned"] += 1
        case = sanitize_name(Path(str(report.get("firmware") or report_path.stem)).stem)
        case_counter = {
            "case": case,
            "report": str(report_path),
            "hypothesis_records": 0,
            "accepted_by_replay": 0,
            "llm_actual_calls": 0,
            "llm_derived_success_attempts": 0,
            "environment_attempts": 0,
            "environment_accepted": 0,
            "environment_rejected": 0,
        }
        report_has_hypothesis = False

        llm_path = report_llm_history_path(report, report_path.parent, Path(str(report.get("firmware") or report_path.stem)))
        if llm_path and llm_path.exists():
            try:
                history = load_json(llm_path)
            except Exception:
                history = []
            entries = history if isinstance(history, list) else history.get("history", []) if isinstance(history, dict) else []
            if isinstance(entries, list):
                totals["llm_history_files"] += 1
                totals["llm_history_entries"] += len(entries)
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    method = str(item.get("method") or item.get("inference_method") or "").lower()
                    if "llm" in method:
                        totals["llm_method_entries"] += 1

        feasibility = report.get("environment_feasibility_summary")
        if isinstance(feasibility, dict):
            counts = feasibility.get("counts") if isinstance(feasibility.get("counts"), dict) else {}
            for key, total_key in [
                ("attempts", "environment_attempts"),
                ("accepted", "environment_accepted"),
                ("rejected", "environment_rejected"),
            ]:
                value = safe_int(counts.get(key))
                totals[total_key] += value
                case_counter[total_key] += value
            reasons = feasibility.get("reject_reasons") if isinstance(feasibility.get("reject_reasons"), dict) else {}
            for reason, count in reasons.items():
                reject_reasons[str(reason)] = reject_reasons.get(str(reason), 0) + safe_int(count)

        for phase_name, phase in iter_phase_dicts(report):
            for key in [
                "llm_actual_calls",
                "llm_successful_values",
                "llm_rejected",
                "llm_errors",
                "llm_derived_candidates_built",
                "llm_derived_candidates_replayed",
                "llm_derived_attempts",
                "llm_derived_replayed_attempts",
                "llm_derived_success_attempts",
                "llm_derived_new_bbs_vs_control",
                "llm_derived_constraints_persisted",
                "llm_derived_constraints_scoped",
            ]:
                mapped = {
                    "llm_rejected": "llm_rejected_values",
                }.get(key, key)
                value = safe_int(phase.get(key))
                totals[mapped] = totals.get(mapped, 0) + value
                if key in {"llm_actual_calls", "llm_derived_success_attempts"}:
                    case_counter[key] += value

            for source, metric_key in [
                ("llm", "llm_derived"),
                ("rules", "rule_derived"),
                ("dependency", "dependency_derived"),
            ]:
                row = source_row(source)
                row["generated"] += source_count(phase.get("built_constraint_source_stats") or {}, source)
                row["replayed"] += source_count(phase.get("replayed_attempt_source_stats") or {}, source)
                row["accepted_by_replay"] += source_count(phase.get("successful_attempt_source_stats") or {}, source)
                row["persisted"] += source_count(phase.get("persisted_constraint_source_stats") or {}, source)
                row["scoped"] += source_count(phase.get("scoped_constraint_source_stats") or {}, source)
                row["new_bbs_vs_control"] += source_count(phase.get("attempt_new_bbs_vs_control_by_source") or {}, source)
                row["new_bbs_vs_phase_before"] += source_count(
                    phase.get("attempt_new_bbs_vs_phase_before_by_source") or {}, source
                )
                row["new_bbs_vs_global"] += source_count(phase.get("attempt_new_bbs_vs_global_by_source") or {}, source)

            records = phase.get("hypothesis_audit_records")
            if not isinstance(records, list):
                continue
            if records:
                report_has_hypothesis = True
            for record in records:
                if not isinstance(record, dict):
                    continue
                source = str(record.get("source") or "unknown")
                stage = str(record.get("stage") or "unknown")
                row = source_row(source)
                totals["hypothesis_records"] += 1
                case_counter["hypothesis_records"] += 1
                if stage == "generated":
                    totals["hypotheses_generated"] += 1
                    row["generated"] += 1
                if bool(record.get("replayed")):
                    totals["hypotheses_replayed"] += 1
                    row["replayed"] += 1
                if bool(record.get("accepted_by_replay")):
                    totals["accepted_by_replay"] += 1
                    case_counter["accepted_by_replay"] += 1
                    row["accepted_by_replay"] += 1
                if bool(record.get("persisted")):
                    totals["persisted"] += 1
                    row["persisted"] += 1
                if bool(record.get("scoped")):
                    totals["scoped"] += 1
                    row["scoped"] += 1
                if record.get("reject_reason"):
                    totals["rejected"] += 1
                    row["rejected"] += 1
                    reason = str(record.get("reject_reason"))
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                delta = record.get("new_bb_delta") if isinstance(record.get("new_bb_delta"), dict) else {}
                for key in ("new_bbs_vs_control", "new_bbs_vs_phase_before", "new_bbs_vs_global"):
                    value = safe_int(delta.get(key))
                    totals[key] += value
                    row[key] += value
                if len(samples) < 32:
                    samples.append(
                        {
                            "case": case,
                            "phase": phase_name,
                            "source": source,
                            "stage": stage,
                            "replayed": bool(record.get("replayed")),
                            "accepted_by_replay": bool(record.get("accepted_by_replay")),
                            "persisted": bool(record.get("persisted")),
                            "scoped": bool(record.get("scoped")),
                            "reject_reason": record.get("reject_reason"),
                            "target_obligation": record.get("target_obligation"),
                            "candidate": record.get("candidate"),
                        }
                    )
        if report_has_hypothesis:
            totals["reports_with_hypothesis_audit"] += 1
        case_rows.append(case_counter)

    source_rows = []
    for source, row in sorted(by_source.items()):
        replayed = row["replayed"]
        accepted = row["accepted_by_replay"]
        source_rows.append(
            {
                "source": source,
                **row,
                "accept_rate": (accepted / replayed * 100.0) if replayed else None,
            }
        )

    return {
        "summary": totals,
        "by_source": source_rows,
        "reject_reasons": reject_reasons,
        "case_rows": case_rows,
        "samples": samples,
    }


def run_rq1(args: argparse.Namespace, targets: list[Target], root: Path) -> dict[str, Any]:
    rq_root = root / "RQ1_coverage_reachability"
    rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    specs = [
        RunSpec(
            rq="RQ1",
            target=target,
            variant="full",
            output_dir=rq_root / "runs" / target.name,
            minutes=args.rq1_time,
            extra_args=tuple(args.lsgemu_arg),
            env_overrides={
                "LSGEMU_PROGRESS_INTERVAL_SECONDS": str(args.progress_interval_seconds),
                "LSGEMU_CONSOLE_PROGRESS": "1",
                "LSGEMU_PROGRESS_COVERAGE_LIST_MODE": args.progress_coverage_list_mode,
            },
        )
        for target in targets
    ]

    def on_result(row: dict[str, Any], target_progress: list[dict[str, Any]]) -> None:
        rows.append(row)
        progress_rows.extend(target_progress)
        threshold_rows.append({"target": row.get("target"), **time_to_thresholds(target_progress, row)})
        write_json(rq_root / "rq1_summary.json", rows)
        write_csv(rq_root / "rq1_summary.csv", rows)
        write_csv(rq_root / "rq1_progress.csv", progress_rows)
        write_csv(rq_root / "rq1_time_to_thresholds.csv", threshold_rows)

    execute_specs(args, specs, on_result=on_result)
    outcome_audit = build_rq1_outcome_audit(targets, rq_root, args.rq1_time, rows)
    write_rq1_outcome_artifacts(rq_root, outcome_audit)

    payload = {
        "rq": "RQ1",
        "claim": "Coverage and reachability under a wallclock upper bound; runs may terminate before the 24h budget.",
        "summary_rows": rows,
        "time_to_thresholds": threshold_rows,
        "outcome_audit": outcome_audit,
    }
    write_json(rq_root / "rq1_payload.json", payload)
    (rq_root / "rq1_summary.md").write_text(build_rq1_markdown(payload))
    return payload


def run_rq2(args: argparse.Namespace, targets: list[Target], root: Path) -> dict[str, Any]:
    rq_root = root / "RQ2_mechanism_diagnostics"
    variants = list(DEFAULT_RQ2_ABLATIONS)
    if args.include_fine_ablations:
        for variant in ("no_frontier", "no_contextual_isr", "no_direct_stream_thread"):
            if variant not in variants:
                variants.append(variant)
    if args.ablation:
        selected = set(args.ablation)
        variants = ["full"] + [variant for variant in sorted(selected) if variant != "full"]

    rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    specs: list[RunSpec] = []
    for variant in variants:
        extra = [*ABLATIONS[variant], *args.lsgemu_arg]
        variant_env = ABLATION_ENV_OVERRIDES.get(variant, {})
        for target in targets:
            specs.append(RunSpec(
                rq="RQ2",
                target=target,
                variant=variant,
                output_dir=rq_root / "runs" / variant / target.name,
                minutes=args.rq2_time,
                extra_args=tuple(extra),
                env_overrides={
                    "LSGEMU_PROGRESS_INTERVAL_SECONDS": str(args.progress_interval_seconds),
                    "LSGEMU_CONSOLE_PROGRESS": "1",
                    "LSGEMU_PROGRESS_COVERAGE_LIST_MODE": args.progress_coverage_list_mode,
                    **variant_env,
                },
                component_summary=True,
            ))

    def on_result(row: dict[str, Any], target_progress: list[dict[str, Any]]) -> None:
        rows.append(row)
        progress_rows.extend(target_progress)
        write_json(rq_root / "rq2_raw_rows.json", rows)
        write_csv(rq_root / "rq2_raw_rows.csv", rows)
        write_csv(rq_root / "rq2_progress.csv", progress_rows)

    execute_specs(args, specs, on_result=on_result)

    delta_rows = compute_ablation_deltas(rows)
    payload = {
        "rq": "RQ2",
        "claim": (
            "Contribution of semantic obligation discovery, evidence-scoped "
            "replay, and context-aware event recovery under equal budgets."
        ),
        "raw_rows": rows,
        "delta_rows": delta_rows,
    }
    write_json(rq_root / "rq2_payload.json", payload)
    write_csv(rq_root / "rq2_ablation_deltas.csv", delta_rows)
    (rq_root / "rq2_summary.md").write_text(build_rq2_markdown(payload))
    return payload


def add_component_summary(row: dict[str, Any]) -> None:
    report_path = Path(str(row.get("report") or ""))
    if not report_path.exists():
        return
    try:
        report = load_json(report_path)
    except Exception:
        return
    components = report.get("component_bbs_counted") if isinstance(report.get("component_bbs_counted"), dict) else {}
    for key in [
        "baseline",
        "contextual_isr",
        "interleaved",
        "frontier_successor_replay",
        "frontier_successor_replay_tail",
        "direct_call_continuation",
        "direct_call_summary_return",
        "rtos_thread_entry",
        "stream_input_replay",
        "deadline_drain",
        "switch_frontier_targeted",
        "frontier_targeted",
    ]:
        row[f"component_{key}_bbs"] = safe_int(components.get(key))


def compute_ablation_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_target = {
        str(row.get("target")): row
        for row in rows
        if row.get("variant") == "full" and row.get("status") in {"ok", "nonzero_returncode_with_report"}
    }
    delta_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("variant") == "full":
            continue
        full = full_by_target.get(str(row.get("target")))
        delta_rows.append(
            {
                "target": row.get("target"),
                "ablation": row.get("variant"),
                "full_valid_covered_bbs": safe_int(full.get("valid_covered_bbs")) if full else None,
                "ablation_valid_covered_bbs": safe_int(row.get("valid_covered_bbs")),
                "delta_valid_bbs": safe_int(row.get("valid_covered_bbs")) - safe_int(full.get("valid_covered_bbs")) if full else None,
                "full_valid_coverage_rate": safe_float(full.get("valid_coverage_rate")) if full else None,
                "ablation_valid_coverage_rate": safe_float(row.get("valid_coverage_rate")),
                "delta_valid_coverage_rate": safe_float(row.get("valid_coverage_rate")) - safe_float(full.get("valid_coverage_rate")) if full else None,
                "returncode": row.get("returncode"),
                "status": row.get("status"),
                "report": row.get("report"),
            }
        )
    return delta_rows


def run_rq3(args: argparse.Namespace, targets: list[Target], root: Path, prior_roots: list[Path]) -> dict[str, Any]:
    rq_root = root / "RQ3_hypothesis_quality_false_progress"
    llm_run_rows: list[dict[str, Any]] = []
    llm_mode = args.rq3_llm_mode
    llm_identity = llm_runtime_identity()
    should_run_llm = False
    llm_skip_reason = ""
    if llm_mode == "force":
        if not llm_identity.get("credential_available"):
            raise SystemExit(
                "--rq3-llm-mode force requires the credential named by the "
                "LLM config (currently missing)"
            )
        should_run_llm = True
    elif llm_mode == "auto":
        should_run_llm = bool(
            llm_identity.get("config_file_exists")
            and llm_identity.get("credential_available")
        )
        if not llm_identity.get("config_file_exists"):
            llm_skip_reason = "LLM config not found; running audit-only RQ3."
        elif not llm_identity.get("credential_available"):
            llm_skip_reason = (
                "LLM credential unavailable; running audit-only RQ3 with no "
                "LLM attribution claim."
            )
    else:
        llm_skip_reason = "LLM attribution run disabled by --rq3-llm-mode=off."

    if should_run_llm:
        wanted_targets = set(args.rq3_target or ["Gateway"])
        rq3_targets = [
            target for target in targets
            if (
                target.name in wanted_targets
                or target.firmware.stem in wanted_targets
                or any(target.name.endswith(f"__{wanted}") for wanted in wanted_targets)
            )
        ]
        if not rq3_targets:
            rq3_targets = [targets[0]]
        for target in rq3_targets:
            output_dir = rq_root / "llm_attribution_runs" / target.name
            env = {
                "LSGEMU_PROGRESS_INTERVAL_SECONDS": str(args.progress_interval_seconds),
                "LSGEMU_CONSOLE_PROGRESS": "1",
                "LSGEMU_PROGRESS_COVERAGE_LIST_MODE": args.progress_coverage_list_mode,
                "LSGEMU_FORCE_LLM_BRANCH_INFERENCE": "1",
                "LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS": str(args.rq3_llm_call_budget),
                "LSGEMU_BRANCH_MMIO_LLM_SECOND_OPINION": "1",
            }
            cmd = build_lsgemu_command(
                target,
                output_dir,
                args.rq3_time,
                extra_args=args.lsgemu_arg,
                run_profile=args.run_profile,
            )
            print(f"[RQ3] llm_attribution/{target.name}: {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
            run_result = run_command(
                cmd,
                output_dir=output_dir,
                env_overrides=env,
                dry_run=args.dry_run,
                skip_existing_report=not args.rerun,
                expected_firmware=target.firmware,
                external_timeout_seconds=external_timeout(args.rq3_time, args.wallclock_grace_minutes),
            )
            row = extract_report_summary(
                rq="RQ3",
                target=target,
                variant="llm_attribution",
                output_dir=output_dir,
                run_result=run_result,
                requested_minutes=args.rq3_time,
            )
            llm_run_rows.append(row)
            write_json(rq_root / "rq3_llm_run_rows.json", llm_run_rows)
            write_csv(rq_root / "rq3_llm_run_rows.csv", llm_run_rows)

    audit_roots = [root_item for root_item in prior_roots if root_item.exists()]
    audit_roots.append(rq_root)
    extra_roots = [Path(item) for item in args.rq3_audit_root]
    audit_roots.extend(extra_roots)
    report_paths = collect_reports_from_roots(audit_roots)
    audit = audit_hypothesis_quality(report_paths)
    payload = {
        "rq": "RQ3",
        "claim": "Hypotheses, LLM outputs, and false-progress candidates are accepted only through replay evidence.",
        "llm_mode": llm_mode,
        "llm_runtime_identity": llm_identity,
        "llm_skip_reason": llm_skip_reason,
        "llm_run_rows": llm_run_rows,
        "audit_roots": [str(path) for path in audit_roots],
        "report_count": len(report_paths),
        "audit": audit,
    }
    write_json(rq_root / "rq3_payload.json", payload)
    write_csv(rq_root / "rq3_case_rows.csv", audit["case_rows"])
    write_csv(rq_root / "rq3_by_source.csv", audit["by_source"])
    (rq_root / "rq3_summary.md").write_text(build_rq3_markdown(payload))
    return payload


def build_rq1_markdown(payload: dict[str, Any]) -> str:
    rows = payload["summary_rows"]
    threshold_rows = payload["time_to_thresholds"]
    audit = payload.get("outcome_audit") if isinstance(payload.get("outcome_audit"), dict) else {}
    completed_rows = audit.get("completed_rows") if isinstance(audit.get("completed_rows"), list) else [
        row for row in rows if row.get("outcome_class", "completed") == "completed"
    ]
    audit_rows = audit.get("rows") if isinstance(audit.get("rows"), list) else rows
    lines = [
        "# RQ1 Coverage and Reachability",
        "",
        payload["claim"],
        "",
        "## Final Coverage (Completed Runs Only)",
        "",
        markdown_table(
            ["Target", "Valid BBs", "Valid total", "Valid cov.", "Exec min", "Entry-derived", "Strict replay"],
            [
                [
                    row["target"],
                    row.get("valid_covered_bbs"),
                    row.get("valid_total_bbs"),
                    f"{safe_float(row.get('valid_coverage_rate')):.2f}%",
                    f"{safe_float(row.get('execution_time_seconds')) / 60.0:.1f}",
                    row.get("coverage_entry_derived"),
                    row.get("strict_real_entry_replayable"),
                ]
                for row in completed_rows
            ],
        ),
        "",
        "Rows terminated by memory guard, timeout, segmentation fault, or missing final report are not final coverage; their last-observed progress is tracked separately.",
        "",
        "## Outcome Audit",
        "",
        markdown_table(
            ["Target", "Outcome", "Last observed cov.", "Last stage", "Peak RSS", "Rerun reason"],
            [
                [
                    row.get("target"),
                    row.get("outcome_class") or row.get("status"),
                    f"{safe_float(row.get('last_observed_valid_coverage_rate')):.2f}%" if row.get("last_observed_valid_coverage_rate") not in {"", None} else "--",
                    row.get("last_observed_stage") or "--",
                    f"{safe_float(row.get('peak_tree_rss_gib')):.1f}GiB" if row.get("peak_tree_rss_gib") not in {"", None} else "--",
                    row.get("rerun_reason") or "--",
                ]
                for row in audit_rows
            ],
        ),
        "",
        "## Time To Coverage Thresholds",
        "",
        markdown_table(
            ["Target", "50%", "60%", "70%", "80%", "90%", "95%"],
            [
                [
                    row["target"],
                    seconds_text(row.get("time_to_50_pct_seconds")),
                    seconds_text(row.get("time_to_60_pct_seconds")),
                    seconds_text(row.get("time_to_70_pct_seconds")),
                    seconds_text(row.get("time_to_80_pct_seconds")),
                    seconds_text(row.get("time_to_90_pct_seconds")),
                    seconds_text(row.get("time_to_95_pct_seconds")),
                ]
                for row in threshold_rows
            ],
        ),
        "",
        "Coverage is counted from LSGEmu reports with `coverage_source=dynamic_unicorn_execution`; static CFG data is not credited as covered BBs.",
        "",
    ]
    return "\n".join(lines)


def build_rq2_markdown(payload: dict[str, Any]) -> str:
    rows = payload["delta_rows"]
    lines = [
        "# RQ2 Mechanism Diagnostics",
        "",
        payload["claim"],
        "",
        markdown_table(
            ["Target", "Ablation", "Full valid BBs", "Ablated valid BBs", "Delta BBs", "Delta cov.", "Status"],
            [
                [
                    row.get("target"),
                    row.get("ablation"),
                    row.get("full_valid_covered_bbs"),
                    row.get("ablation_valid_covered_bbs"),
                    row.get("delta_valid_bbs"),
                    f"{safe_float(row.get('delta_valid_coverage_rate')):.2f}%" if row.get("delta_valid_coverage_rate") is not None else "--",
                    row.get("status"),
                ]
                for row in rows
            ],
        ),
        "",
        "Interpret deltas as equal-budget ablations. Per-stage BB counters remain diagnostics and are not by themselves causal evidence.",
        "",
    ]
    return "\n".join(lines)


def build_rq3_markdown(payload: dict[str, Any]) -> str:
    audit = payload["audit"]
    summary = audit["summary"]
    lines = [
        "# RQ3 Hypothesis Quality and False Progress",
        "",
        payload["claim"],
        "",
        f"LLM mode: `{payload['llm_mode']}`.",
    ]
    if payload.get("llm_skip_reason"):
        lines.extend(["", f"LLM attribution run note: {payload['llm_skip_reason']}"])
    lines.extend(
        [
            "",
            "## Attribution Chain",
            "",
            markdown_table(
                ["Stage", "Metric", "Value"],
                [
                    ["Reports", "Reports scanned", summary.get("reports_scanned")],
                    ["Hypothesis", "Audit records", summary.get("hypothesis_records")],
                    ["Hypothesis", "Generated", summary.get("hypotheses_generated")],
                    ["Replay", "Replayed", summary.get("hypotheses_replayed")],
                    ["Replay", "Accepted by replay", summary.get("accepted_by_replay")],
                    ["Replay", "Rejected", summary.get("rejected")],
                    ["Coverage", "+BB vs control", summary.get("new_bbs_vs_control")],
                    ["LLM", "History entries", summary.get("llm_history_entries")],
                    ["LLM", "LLM-method entries", summary.get("llm_method_entries")],
                    ["LLM", "Actual calls", summary.get("llm_actual_calls")],
                    ["LLM", "Derived attempts", summary.get("llm_derived_attempts")],
                    ["LLM", "Successful attempts", summary.get("llm_derived_success_attempts")],
                    ["LLM", "+BB vs control", summary.get("llm_derived_new_bbs_vs_control")],
                    ["Env", "Feasibility attempts", summary.get("environment_attempts")],
                    ["Env", "Accepted", summary.get("environment_accepted")],
                    ["Env", "Rejected", summary.get("environment_rejected")],
                ],
            ),
            "",
            "## By Source",
            "",
            markdown_table(
                ["Source", "Generated", "Replayed", "Accepted", "Persisted", "Scoped", "+BB vs control", "Accept rate"],
                [
                    [
                        row.get("source"),
                        row.get("generated"),
                        row.get("replayed"),
                        row.get("accepted_by_replay"),
                        row.get("persisted"),
                        row.get("scoped"),
                        row.get("new_bbs_vs_control"),
                        f"{safe_float(row.get('accept_rate')):.2f}%" if row.get("accept_rate") is not None else "--",
                    ]
                    for row in audit["by_source"]
                ],
            ),
            "",
            "This table separates proposed facts from replay-accepted facts. LLM history alone is not counted as coverage impact.",
            "",
        ]
    )
    return "\n".join(lines)


def seconds_text(value: Any) -> str:
    if value is None:
        return "--"
    seconds = safe_float(value)
    if seconds <= 0:
        return "0.0m"
    return f"{seconds / 60.0:.1f}m"


def external_timeout(minutes: float, grace_minutes: float) -> int:
    if minutes <= 0:
        return 0
    return int((float(minutes) + float(grace_minutes)) * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LSGEmu RQ1/RQ2/RQ3 experiments and summarize paper-ready data.")
    parser.add_argument("--config", default=None, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument("--rq", choices=["all", "rq1", "rq2", "rq3"], default="all")
    parser.add_argument(
        "--target-suite",
        choices=["elfmultifuzz-23", "p2im-three", "p2im-all"],
        default="elfmultifuzz-23",
    )
    parser.add_argument("--target", action="append", default=[], help="Target name to include. May be repeated.")
    parser.add_argument("--manifest", default=None, help="Optional JSON target manifest.")
    parser.add_argument("--allow-missing-targets", action="store_true")
    parser.add_argument("--allow-non23-elfmultifuzz", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--audit-rq1-root",
        default=None,
        help="Audit an existing RQ1_coverage_reachability directory or experiment root without launching runs.",
    )
    parser.add_argument("--run-profile", default=None)
    parser.add_argument("--progress-interval-seconds", type=int, default=600)
    parser.add_argument(
        "--progress-coverage-list-mode",
        choices=["none", "final", "on_change", "sample", "all"],
        default="none",
        help="Whether progress JSONL records include full covered BB lists. Final reports still keep authoritative lists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_configured_int(EVALUATION_DEFAULTS, "safe_jobs_default", 1),
        help="Number of LSGEmu subprocesses to run concurrently. Use 0 for auto.",
    )
    parser.add_argument(
        "--max-safe-jobs",
        type=int,
        default=_configured_int(SCHEDULER_DEFAULTS, "max_safe_jobs", 8),
        help="Safety cap applied unless --allow-high-jobs is set. For long LSGEmu runs this is an upper bound, not a target.",
    )
    parser.add_argument(
        "--allow-high-jobs",
        action="store_true",
        help="Allow --jobs above --max-safe-jobs. Use only after monitoring memory, swap, and disk wait.",
    )
    parser.add_argument("--wallclock-grace-minutes", type=float, default=30.0)
    parser.add_argument(
        "--scheduler-initial-jobs",
        type=int,
        default=_configured_int(SCHEDULER_DEFAULTS, "initial_jobs", 2),
        help="Guarded mode starts at this many concurrent LSGEmu subprocesses, then ramps up.",
    )
    parser.add_argument(
        "--scheduler-ramp-seconds",
        type=int,
        default=_configured_int(SCHEDULER_DEFAULTS, "ramp_seconds", 300),
        help="Minimum seconds before guarded mode may add one more concurrent subprocess.",
    )
    parser.add_argument(
        "--scheduler-poll-seconds",
        type=float,
        default=5.0,
        help="Guarded scheduler polling interval.",
    )
    parser.add_argument(
        "--scheduler-status-interval-seconds",
        type=int,
        default=30,
        help="How often guarded scheduler prints hold/status messages.",
    )
    parser.add_argument(
        "--scheduler-min-free-memory-gib",
        type=float,
        default=_configured_float(SCHEDULER_DEFAULTS, "min_free_memory_gib", 16.0),
        help="Do not start a new LSGEmu subprocess unless this much MemAvailable remains after projected per-job RSS.",
    )
    parser.add_argument(
        "--scheduler-per-job-memory-gib",
        type=float,
        default=_configured_float(SCHEDULER_DEFAULTS, "per_job_memory_gib", 8.0),
        help="Initial per-job memory projection before observed RSS peaks are available.",
    )
    parser.add_argument(
        "--scheduler-peak-multiplier",
        type=float,
        default=_configured_float(SCHEDULER_DEFAULTS, "peak_multiplier", 1.25),
        help="Multiply observed per-job peak RSS when projecting whether a new job is safe.",
    )
    parser.add_argument(
        "--scheduler-max-swap-used-gib",
        type=float,
        default=_configured_float(SCHEDULER_DEFAULTS, "max_swap_used_gib", 4.0),
        help="Hold new starts when swap usage exceeds this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--scheduler-kill-below-free-memory-gib",
        type=float,
        default=_configured_float(
            SCHEDULER_DEFAULTS,
            "kill_below_free_memory_gib",
            16.0,
        ),
        help="Terminate the largest running LSGEmu subprocess if MemAvailable falls below this guard. Use 0 to disable.",
    )
    parser.add_argument(
        "--scheduler-kill-above-swap-used-gib",
        type=float,
        default=_configured_float(
            SCHEDULER_DEFAULTS,
            "kill_above_swap_used_gib",
            8.0,
        ),
        help="Terminate the largest running LSGEmu subprocess if swap usage exceeds this guard. Use 0 to disable.",
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun even when an output report already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands without executing LSGEmu.")
    parser.add_argument("--quick", action="store_true", help="Use short debug budgets instead of paper budgets.")
    parser.add_argument("--lsgemu-arg", action="append", default=[], help="Raw extra argument forwarded to lsgemu.py/interleaved runner.")

    parser.add_argument(
        "--rq1-time",
        type=float,
        default=_configured_float(EVALUATION_DEFAULTS, "default_rq1_minutes", 1440.0),
        help="RQ1 wallclock upper bound in minutes.",
    )
    parser.add_argument(
        "--rq2-time",
        type=float,
        default=_configured_float(EVALUATION_DEFAULTS, "default_rq2_minutes", 60.0),
        help="RQ2 equal-budget ablation upper bound in minutes.",
    )
    parser.add_argument(
        "--rq3-time",
        type=float,
        default=_configured_float(EVALUATION_DEFAULTS, "default_rq3_minutes", 30.0),
        help="RQ3 optional LLM attribution upper bound in minutes.",
    )
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), action="append", default=[])
    parser.add_argument(
        "--include-fine-ablations",
        action="store_true",
        help=(
            "Add legacy implementation-level no_frontier/no_contextual_isr/"
            "no_direct_stream_thread diagnostics to the contribution-level "
            "RQ2 ablations."
        ),
    )
    parser.add_argument("--rq3-llm-mode", choices=["auto", "force", "off"], default="auto")
    parser.add_argument("--rq3-llm-call-budget", type=int, default=8)
    parser.add_argument("--rq3-target", action="append", default=[], help="Target for the focused RQ3 LLM attribution run.")
    parser.add_argument("--rq3-audit-root", action="append", default=[], help="Existing artifact root to include in RQ3 audit.")
    return parser


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.rq1_time = min(args.rq1_time, 5.0)
    args.rq2_time = min(args.rq2_time, 3.0)
    args.rq3_time = min(args.rq3_time, 3.0)
    args.progress_interval_seconds = min(args.progress_interval_seconds, 60)
    args.wallclock_grace_minutes = min(args.wallclock_grace_minutes, 5.0)


def normalize_jobs(args: argparse.Namespace, target_count: int) -> None:
    requested = int(args.jobs)
    safe_cap = max(1, int(args.max_safe_jobs))
    if requested <= 0:
        auto_jobs = min(target_count, safe_cap)
        args.jobs = max(1, auto_jobs)
        print(f"[scheduler] auto jobs={args.jobs} target_count={target_count} safe_cap={safe_cap}", flush=True)
        return
    if requested > safe_cap and not args.allow_high_jobs:
        print(
            f"[scheduler] requested jobs={requested} exceeds safe cap={safe_cap}; "
            f"clamping to {safe_cap}. Use --allow-high-jobs to override.",
            flush=True,
        )
        args.jobs = safe_cap


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_quick_defaults(args)

    targets = select_targets(args)
    if args.audit_rq1_root:
        raw_root = Path(args.audit_rq1_root).resolve()
        rq_root = raw_root if raw_root.name == "RQ1_coverage_reachability" else raw_root / "RQ1_coverage_reachability"
        if not rq_root.exists():
            raise SystemExit(f"RQ1 root not found: {rq_root}")
        summary_path = rq_root / "rq1_summary.csv"
        summary_rows: list[dict[str, Any]] = []
        if summary_path.exists():
            with summary_path.open(newline="") as f:
                summary_rows = list(csv.DictReader(f))
        audit = build_rq1_outcome_audit(targets, rq_root, args.rq1_time, summary_rows)
        write_rq1_outcome_artifacts(rq_root, audit)
        print(f"rq1_audit_root={rq_root}", flush=True)
        print(f"outcome_counts={json.dumps(audit['counts'], sort_keys=True)}", flush=True)
        print(f"rerun_manifest={rq_root / 'rq1_rerun_manifest.json'}", flush=True)
        return 0
    normalize_jobs(args, len(targets))
    output_root = Path(args.output_root) if args.output_root else RESULTS_ROOT / f"rq1_rq2_rq3_{now_tag()}"
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit(),
        "rq": args.rq,
        "target_suite": args.target_suite,
        "target_count": len(targets),
        "jobs": max(1, int(args.jobs)),
        "process_model": "one scheduler parent plus up to jobs LSGEmu subprocesses",
        "guarded_scheduler": {
            "initial_jobs": args.scheduler_initial_jobs,
            "ramp_seconds": args.scheduler_ramp_seconds,
            "poll_seconds": args.scheduler_poll_seconds,
            "min_free_memory_gib": args.scheduler_min_free_memory_gib,
            "per_job_memory_gib": args.scheduler_per_job_memory_gib,
            "peak_multiplier": args.scheduler_peak_multiplier,
            "max_swap_used_gib": args.scheduler_max_swap_used_gib,
            "kill_below_free_memory_gib": args.scheduler_kill_below_free_memory_gib,
            "kill_above_swap_used_gib": args.scheduler_kill_above_swap_used_gib,
        },
        "targets": [
            {
                "name": target.name,
                "firmware": str(target.firmware),
                "entry_point": target.entry_point,
                "load_base": target.load_base,
                "exists": target.firmware.exists(),
            }
            for target in targets
        ],
        "budgets_minutes": {
            "rq1": args.rq1_time,
            "rq2": args.rq2_time,
            "rq3": args.rq3_time,
        },
        "progress_interval_seconds": args.progress_interval_seconds,
        "progress_coverage_list_mode": args.progress_coverage_list_mode,
        "run_profile": args.run_profile,
        "dry_run": args.dry_run,
    }
    write_json(output_root / "experiment_manifest.json", manifest)

    payloads: dict[str, Any] = {}
    prior_roots: list[Path] = []
    if args.rq in {"all", "rq1"}:
        payloads["rq1"] = run_rq1(args, targets, output_root)
        prior_roots.append(output_root / "RQ1_coverage_reachability")
    if args.rq in {"all", "rq2"}:
        payloads["rq2"] = run_rq2(args, targets, output_root)
        prior_roots.append(output_root / "RQ2_mechanism_diagnostics")
    if args.rq in {"all", "rq3"}:
        payloads["rq3"] = run_rq3(args, targets, output_root, prior_roots)

    write_json(output_root / "combined_payload.json", payloads)
    print(f"output_root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
