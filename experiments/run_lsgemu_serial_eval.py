#!/usr/bin/env python3
"""Serial LSGEmu evaluation runner.

Usage:
    python experiments/run_lsgemu_serial_eval.py 1
    python experiments/run_lsgemu_serial_eval.py 2 --time 1440

Modes:
    1  Full LSGEmu
    2  Without Semantic Obligation
    3  Without Scoped Replay
    4  Without Context-aware Event Recovery
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path("/opt/artifact/benchmarks/elfmultifuzz")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / ".lsgemu_runs" / "evaluation_serial"

FINGERPRINT_ROOTS = (
    PROJECT_ROOT / "lsgemu",
    PROJECT_ROOT / "fuzzengine",
    PROJECT_ROOT / "configs",
)
FINGERPRINT_FILES = (
    PROJECT_ROOT / "experiments" / "lsgemu_no_semantic_obligation.py",
    PROJECT_ROOT / "experiments" / "lsgemu_no_scoped_replay.py",
    PROJECT_ROOT / "experiments" / "lsgemu_no_context_event_replay.py",
    PROJECT_ROOT / "experiments" / "run_lsgemu_serial_eval.py",
    PROJECT_ROOT / "Requirements.txt",
    PROJECT_ROOT / "LLM.yaml",
)
FINGERPRINT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt"}

# These values define the serial campaign itself. Allowing them through the
# child argument passthrough would make the recorded campaign identity differ
# from the command that actually executes.
MANAGED_CHILD_OPTIONS = frozenset({
    "--firmware",
    "--mode",
    "--output-dir",
    "--time",
    "--total-time-minutes",
    "--disable-semantic-obligation-stages",
    "--disable-scoped-replay-stages",
    "--disable-context-event-replay-stages",
})
IDENTITY_FILE_OPTIONS = frozenset({"--config", "--run-profile"})
SENSITIVE_ENVIRONMENT_TOKENS = frozenset({
    "APIKEY",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
})

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.artifact_io import (
    append_jsonl as atomic_append_jsonl,
    atomic_json_dump,
    atomic_publish_file,
)
from lsgemu.toolchain_fingerprint import (
    collect_toolchain_fingerprint,
    file_identity,
    sha256_file,
)


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    entrypoint: Path
    paper_label: str


MODES: dict[str, ModeConfig] = {
    "1": ModeConfig(
        key="full",
        label="full",
        entrypoint=PROJECT_ROOT / "lsgemu" / "lsgemu.py",
        paper_label="Full LSGEmu",
    ),
    "2": ModeConfig(
        key="no_semantic_obligation_v2",
        label="no_semantic_obligation_v2",
        entrypoint=PROJECT_ROOT / "experiments" / "lsgemu_no_semantic_obligation.py",
        paper_label="Without Semantic Obligation (v2)",
    ),
    "3": ModeConfig(
        key="no_scoped_replay_v2",
        label="no_scoped_replay_v2",
        entrypoint=PROJECT_ROOT / "experiments" / "lsgemu_no_scoped_replay.py",
        paper_label="Without Scoped Replay (v2)",
    ),
    "4": ModeConfig(
        key="no_context_event_replay",
        label="no_context_event_replay",
        entrypoint=PROJECT_ROOT / "experiments" / "lsgemu_no_context_event_replay.py",
        paper_label="Without Context-aware Event Recovery",
    ),
}


@dataclass(frozen=True)
class FirmwareJob:
    elf_path: Path
    relative_path: Path
    family: str
    firmware_dir: str
    elf_stem: str


def _sensitive_environment_name(name: str) -> bool:
    normalized = str(name or "").upper()
    tokens = set(normalized.replace("-", "_").split("_"))
    return bool(tokens & SENSITIVE_ENVIRONMENT_TOKENS)


def lsgemu_environment_identity() -> dict[str, object]:
    """Record LSGEMU configuration without persisting credential values."""

    identity: dict[str, object] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("LSGEMU_"):
            continue
        if _sensitive_environment_name(key):
            identity[key] = {
                "redacted": True,
                "present": bool(value),
                "sha256": (
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    if value
                    else ""
                ),
            }
        else:
            identity[key] = value
    return identity


def extra_argument_file_identities(arguments: list[str]) -> list[dict[str, object]]:
    """Fingerprint external configuration files accepted by the child CLI."""

    identities: list[dict[str, object]] = []
    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        option, separator, inline_value = token.partition("=")
        if option not in IDENTITY_FILE_OPTIONS:
            index += 1
            continue
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(arguments):
                identities.append({
                    "option": option,
                    "path": None,
                    "exists": False,
                    "error": "missing_option_value",
                })
                index += 1
                continue
            value = str(arguments[index + 1])
            index += 1
        identities.append({"option": option, **file_identity(value)})
        index += 1
    return identities


def llm_runtime_identity() -> dict[str, object]:
    """Describe LLM availability without persisting a credential value."""

    config_path = Path(
        os.environ.get("LSGEMU_LLM_CONFIG")
        or PROJECT_ROOT / "LLM.yaml"
    ).expanduser().resolve()
    config: dict[str, object] = {}
    config_error = ""
    config_file_sha256 = ""
    if config_path.is_file():
        try:
            config_file_sha256 = sha256_file(config_path)
            with config_path.open() as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                nested = loaded.get("llm")
                config = dict(nested) if isinstance(nested, dict) else dict(loaded)
        except Exception as exc:
            config_error = f"{type(exc).__name__}: {exc}"

    configured_env = str(config.get("api_key_env") or "").strip()
    candidate_env_names = tuple(dict.fromkeys(
        name
        for name in (
            configured_env,
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "OPENAI_API_KEY",
        )
        if name
    ))
    inline_credential = str(config.get("api_key") or "").strip()
    credential = inline_credential
    credential_source = "inline_config" if inline_credential else ""
    for env_name in candidate_env_names:
        env_value = str(os.environ.get(env_name) or "").strip()
        if not credential and env_value:
            credential = env_value
            credential_source = f"environment:{env_name}"
            break

    api_base = str(
        config.get("api_base")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("DASHSCOPE_API_BASE")
        or ""
    ).strip()
    model = str(
        config.get("model")
        or os.environ.get("LSGEMU_LLM_MODEL")
        or ""
    ).strip()
    return {
        "schema": "lsgemu.llm_runtime_identity.v1",
        "config_file": str(config_path),
        "config_file_exists": config_path.is_file(),
        "config_file_sha256": config_file_sha256,
        "config_error": config_error,
        "model": model,
        "api_base": api_base,
        "credential_available": bool(credential),
        "credential_source": credential_source or "missing",
        "credential_sha256": (
            hashlib.sha256(credential.encode("utf-8")).hexdigest()
            if credential
            else ""
        ),
        "inline_credential_configured": bool(inline_credential),
    }


def source_tree_fingerprint() -> dict[str, object]:
    paths: set[Path] = set()
    for root in FINGERPRINT_ROOTS:
        if not root.exists():
            continue
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in FINGERPRINT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    paths.update(path for path in FINGERPRINT_FILES if path.is_file())

    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(PROJECT_ROOT))):
        relative = str(path.relative_to(PROJECT_ROOT))
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "schema": "lsgemu.source_tree_fingerprint.v1",
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "included_paths": relative_paths,
    }


def build_campaign_metadata(
    *,
    args: argparse.Namespace,
    mode: ModeConfig,
    benchmark_root: Path,
    output_root: Path,
    extra_args: list[str],
) -> dict[str, object]:
    source = source_tree_fingerprint()
    toolchain = collect_toolchain_fingerprint()
    llm_identity = llm_runtime_identity()
    configuration = {
        "mode": mode.key,
        "entrypoint": str(mode.entrypoint.resolve()),
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "time_minutes": float(args.time),
        "python": str(args.python),
        "extra_lsgemu_args": list(extra_args),
        "extra_argument_files": extra_argument_file_identities(extra_args),
        "lsgemu_environment": lsgemu_environment_identity(),
        "source_tree_sha256": str(source["sha256"]),
        "toolchain_fingerprint": str(toolchain.get("runtime_fingerprint") or ""),
        "toolchain_identity_hash": str(toolchain.get("fingerprint") or ""),
        "llm_runtime_identity": llm_identity,
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "lsgemu.serial_campaign_metadata.v1",
        "campaign_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "source_tree": source,
        "toolchain": toolchain,
        "configuration": configuration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSGEmu or a contribution-level ablation over all benchmark ELF files serially.",
    )
    parser.add_argument(
        "mode",
        choices=sorted(MODES),
        help="1=full, 2=without semantic obligation, 3=without scoped replay, 4=without context/event replay.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=f"Benchmark root containing family/firmware/*.elf directories. Default: {DEFAULT_BENCHMARK_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root for campaign outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1440.0,
        help="Per-firmware LSGEmu time budget in minutes. Default: 1440.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for child LSGEmu runs.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Run only this benchmark family. May be repeated, e.g. --family P2IM --family uEmu.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only jobs whose relative path contains this substring. May be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip jobs whose relative path contains this substring. May be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many jobs after filtering. Default: 0 means no limit.",
    )
    parser.add_argument(
        "--start-after",
        default="",
        help="Skip jobs through and including this relative path substring, then run the rest.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run even if an output report already exists. By default completed reports are skipped.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the serial campaign when one firmware exits non-zero.",
    )
    parser.add_argument(
        "--process-timeout-minutes",
        type=float,
        default=0.0,
        help="Optional outer subprocess timeout per firmware. Default: 0 disables it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned serial jobs without executing them.",
    )
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    args.extra_lsgemu_args = extra_args
    return args


def discover_firmware_jobs(benchmark_root: Path) -> list[FirmwareJob]:
    benchmark_root = benchmark_root.resolve()
    if not benchmark_root.exists():
        raise SystemExit(f"benchmark root does not exist: {benchmark_root}")
    elf_paths = sorted(path.resolve() for path in benchmark_root.rglob("*.elf") if path.is_file())
    jobs: list[FirmwareJob] = []
    for elf_path in elf_paths:
        relative_path = elf_path.relative_to(benchmark_root)
        parts = relative_path.parts
        family = parts[0] if len(parts) >= 1 else "unknown"
        firmware_dir = parts[1] if len(parts) >= 2 else elf_path.parent.name
        jobs.append(
            FirmwareJob(
                elf_path=elf_path,
                relative_path=relative_path,
                family=family,
                firmware_dir=firmware_dir,
                elf_stem=elf_path.stem,
            )
        )
    return jobs


def filter_jobs(jobs: Iterable[FirmwareJob], args: argparse.Namespace) -> list[FirmwareJob]:
    selected = list(jobs)
    families = {item.strip() for item in args.family if item.strip()}
    if families:
        selected = [job for job in selected if job.family in families]

    only_patterns = [item for item in args.only if item]
    if only_patterns:
        selected = [
            job
            for job in selected
            if any(pattern in str(job.relative_path) for pattern in only_patterns)
        ]

    exclude_patterns = [item for item in args.exclude if item]
    if exclude_patterns:
        selected = [
            job
            for job in selected
            if not any(pattern in str(job.relative_path) for pattern in exclude_patterns)
        ]

    if args.start_after:
        remaining: list[FirmwareJob] = []
        past_marker = False
        for job in selected:
            if not past_marker:
                if args.start_after in str(job.relative_path):
                    past_marker = True
                continue
            remaining.append(job)
        selected = remaining

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    return selected


def job_output_dir(output_root: Path, mode: ModeConfig, job: FirmwareJob) -> Path:
    relative_parent = job.relative_path.parent
    return (output_root / mode.label / relative_parent).resolve()


def expected_report_path(output_dir: Path, job: FirmwareJob) -> Path:
    return output_dir / f"{job.elf_stem}_interleaved_report.json"


def report_is_complete(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    try:
        with report_path.open() as f:
            report = json.load(f)
    except Exception:
        return False
    metrics = report.get("evaluation_metrics")
    if not isinstance(metrics, dict):
        return False
    return str(metrics.get("schema") or "") == "lsgemu.evaluation_metrics.v1"


def load_report(report_path: Path) -> dict[str, object] | None:
    try:
        with report_path.open() as handle:
            report = json.load(handle)
    except Exception:
        return None
    return report if isinstance(report, dict) else None


def report_identity_fields(report_path: Path) -> dict[str, object]:
    report = load_report(report_path) or {}
    toolchain = report.get("toolchain_fingerprint")
    static_cache = report.get("static_cache_identity")
    return {
        "valid_bb_denominator_hash": str(
            report.get("valid_bb_denominator_hash") or ""
        ),
        "toolchain_runtime_fingerprint": str(
            report.get("toolchain_runtime_fingerprint")
            or (
                toolchain.get("runtime_fingerprint")
                if isinstance(toolchain, dict)
                else ""
            )
            or ""
        ),
        "toolchain_identity_hash": str(
            report.get("toolchain_identity_hash")
            or (
                toolchain.get("fingerprint")
                if isinstance(toolchain, dict)
                else ""
            )
            or ""
        ),
        "static_cache_identity_hash": str(
            report.get("static_cache_identity_hash")
            or (
                static_cache.get("identity_hash")
                if isinstance(static_cache, dict)
                else ""
            )
            or ""
        ),
    }


def report_matches_run_identity(
    report_path: Path,
    *,
    campaign_fingerprint: str,
    firmware_sha256: str,
    attempt_id: str | None = None,
    source_tree_sha256: str | None = None,
    toolchain_runtime_fingerprint: str | None = None,
    toolchain_identity_hash: str | None = None,
) -> bool:
    report = load_report(report_path)
    if report is None or not report_is_complete(report_path):
        return False
    if str(report.get("campaign_fingerprint") or "") != str(campaign_fingerprint):
        return False
    if str(report.get("firmware_sha256") or "") != str(firmware_sha256):
        return False
    if attempt_id is not None and str(report.get("attempt_id") or "") != str(attempt_id):
        return False
    if (
        source_tree_sha256
        and str(report.get("source_tree_sha256") or "") != str(source_tree_sha256)
    ):
        return False
    report_identity = report_identity_fields(report_path)
    if (
        toolchain_runtime_fingerprint
        and str(report_identity.get("toolchain_runtime_fingerprint") or "")
        != str(toolchain_runtime_fingerprint)
    ):
        return False
    if (
        toolchain_identity_hash
        and str(report_identity.get("toolchain_identity_hash") or "")
        != str(toolchain_identity_hash)
    ):
        return False
    return bool(str(report.get("valid_bb_denominator_hash") or ""))


def report_is_reusable(
    report_path: Path,
    status_path: Path,
    *,
    campaign_fingerprint: str,
    firmware_sha256: str,
    source_tree_sha256: str,
    toolchain_runtime_fingerprint: str,
    toolchain_identity_hash: str,
) -> bool:
    if not report_path.exists() or not status_path.exists():
        return False
    try:
        with status_path.open() as f:
            status = json.load(f)
    except Exception:
        return False
    if not isinstance(status, dict):
        return False
    attempt_id = str(status.get("attempt_id") or "")
    return bool(
        str(status.get("campaign_fingerprint") or "") == str(campaign_fingerprint)
        and str(status.get("firmware_sha256") or "") == str(firmware_sha256)
        and str(status.get("status") or "") in {"success", "skipped_existing_report"}
        and attempt_id
        and report_matches_run_identity(
            report_path,
            campaign_fingerprint=campaign_fingerprint,
            firmware_sha256=firmware_sha256,
            attempt_id=attempt_id,
            source_tree_sha256=source_tree_sha256,
            toolchain_runtime_fingerprint=toolchain_runtime_fingerprint,
            toolchain_identity_hash=toolchain_identity_hash,
        )
    )


def normalized_extra_args(raw: list[str]) -> list[str]:
    normalized = list(raw[1:] if raw and raw[0] == "--" else raw)
    conflicts = sorted({
        token.split("=", 1)[0]
        for token in normalized
        if token.split("=", 1)[0] in MANAGED_CHILD_OPTIONS
    })
    if conflicts:
        raise SystemExit(
            "serial runner owns these child options; configure them through "
            "the serial runner mode/options instead: " + ", ".join(conflicts)
        )
    return normalized


def build_command(
    python_exe: str,
    mode: ModeConfig,
    job: FirmwareJob,
    output_dir: Path,
    time_minutes: float,
    extra_args: list[str],
) -> list[str]:
    command = [
        python_exe,
        str(mode.entrypoint),
        str(job.elf_path),
        "--time",
        str(float(time_minutes)),
        "--output-dir",
        str(output_dir),
    ]
    if mode.key == "full":
        command.extend(["--mode", "interleaved"])
    command.extend(extra_args)
    return command


def write_jsonl(path: Path, record: dict[str, object]) -> None:
    atomic_append_jsonl(path, record)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    atomic_json_dump(summary, path, indent=2, sort_keys=True)


def proc_status_kib(pid: int, field_name: str) -> int:
    try:
        with Path(f"/proc/{int(pid)}/status").open() as f:
            for line in f:
                if not line.startswith(f"{field_name}:"):
                    continue
                fields = line.split()
                return int(fields[1]) if len(fields) >= 2 else 0
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return 0
    return 0


def process_tree_pids(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = [int(item) for item in children_path.read_text().split()]
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            children = []
        pending.extend(child for child in children if child not in seen)
    return seen


def host_memory_kib() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open() as f:
            for line in f:
                key, _, remainder = line.partition(":")
                if key not in {"MemAvailable", "SwapFree"}:
                    continue
                fields = remainder.split()
                values[key] = int(fields[0]) if fields else 0
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return values.get("MemAvailable", 0), values.get("SwapFree", 0)


PROC_IO_FIELDS = ("rchar", "wchar", "read_bytes", "write_bytes")


def parse_proc_io(text: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line in str(text or "").splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in PROC_IO_FIELDS:
            continue
        try:
            counters[key] = max(0, int(value.strip()))
        except ValueError:
            continue
    return counters


def proc_io_counters(pid: int) -> dict[str, int]:
    try:
        return parse_proc_io(Path(f"/proc/{int(pid)}/io").read_text())
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return {}


def public_resource_usage(state: dict[str, object]) -> dict[str, object]:
    """Remove sampler-only state before persisting the resource record."""
    return {
        str(key): value
        for key, value in state.items()
        if not str(key).startswith("_")
    }


def sample_resources(pid: int, state: dict[str, object]) -> None:
    pids = process_tree_pids(pid)
    tree_rss_kib = sum(proc_status_kib(child_pid, "VmRSS") for child_pid in pids)
    root_hwm_kib = proc_status_kib(pid, "VmHWM")
    mem_available_kib, swap_free_kib = host_memory_kib()
    state["samples"] = int(state.get("samples", 0) or 0) + 1
    state["peak_process_tree_rss_kib"] = max(
        int(state.get("peak_process_tree_rss_kib", 0) or 0), tree_rss_kib
    )
    state["peak_root_hwm_kib"] = max(
        int(state.get("peak_root_hwm_kib", 0) or 0), root_hwm_kib
    )
    state["max_process_tree_members"] = max(
        int(state.get("max_process_tree_members", 0) or 0), len(pids)
    )
    if mem_available_kib > 0:
        previous = int(state.get("min_host_mem_available_kib", 0) or 0)
        state["min_host_mem_available_kib"] = (
            mem_available_kib if previous <= 0 else min(previous, mem_available_kib)
        )
    if swap_free_kib > 0:
        previous = int(state.get("min_host_swap_free_kib", 0) or 0)
        state["min_host_swap_free_kib"] = (
            swap_free_kib if previous <= 0 else min(previous, swap_free_kib)
        )

    last_by_pid = state.setdefault("_io_last_by_pid", {})
    if not isinstance(last_by_pid, dict):
        last_by_pid = {}
        state["_io_last_by_pid"] = last_by_pid
    for child_pid in pids:
        current = proc_io_counters(child_pid)
        if not current:
            continue
        previous = last_by_pid.get(child_pid)
        if not isinstance(previous, dict):
            previous = {}
        for field in PROC_IO_FIELDS:
            value = int(current.get(field, 0) or 0)
            before = int(previous.get(field, 0) or 0)
            delta = value if field not in previous else max(0, value - before)
            output_key = f"process_tree_{field}_lower_bound"
            state[output_key] = int(state.get(output_key, 0) or 0) + delta
        last_by_pid[child_pid] = current


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate one serial job and every subprocess it started."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            return
    process.wait()


def run_one_job(
    *,
    args: argparse.Namespace,
    mode: ModeConfig,
    job: FirmwareJob,
    index: int,
    total: int,
    campaign_dir: Path,
    manifest_path: Path,
    extra_args: list[str],
    campaign_metadata: dict[str, object],
) -> dict[str, object]:
    output_dir = job_output_dir(campaign_dir, mode, job)
    report_path = expected_report_path(output_dir, job)
    stable_log_path = output_dir / f"{job.elf_stem}_runner.log"
    status_path = output_dir / f"{job.elf_stem}_run_status.json"
    campaign_fingerprint = str(campaign_metadata.get("campaign_fingerprint") or "")
    source_tree = campaign_metadata.get("source_tree") or {}
    source_tree_sha256 = (
        str(source_tree.get("sha256") or "") if isinstance(source_tree, dict) else ""
    )
    expected_configuration = campaign_metadata.get("configuration") or {}
    if not isinstance(expected_configuration, dict):
        raise RuntimeError("campaign metadata has no valid configuration identity")
    expected_llm_identity = expected_configuration.get("llm_runtime_identity") or {}
    current_llm_identity = llm_runtime_identity()
    if current_llm_identity != expected_llm_identity:
        raise RuntimeError(
            "LLM configuration or credential identity changed after the campaign "
            "fingerprint was created"
        )
    expected_environment = expected_configuration.get("lsgemu_environment") or {}
    current_environment = lsgemu_environment_identity()
    if current_environment != expected_environment:
        raise RuntimeError(
            "LSGEMU environment changed after the campaign fingerprint was created"
        )
    expected_argument_files = expected_configuration.get("extra_argument_files") or []
    current_argument_files = extra_argument_file_identities(extra_args)
    if current_argument_files != expected_argument_files:
        raise RuntimeError(
            "external config or run-profile content changed after the campaign "
            "fingerprint was created"
        )
    current_source_sha256 = str(source_tree_fingerprint().get("sha256") or "")
    if current_source_sha256 != source_tree_sha256:
        raise RuntimeError(
            "execution/configuration sources changed after the campaign fingerprint "
            f"was created: expected {source_tree_sha256}, got {current_source_sha256}"
        )
    expected_toolchain_fingerprint = str(
        ((campaign_metadata.get("toolchain") or {}).get("runtime_fingerprint") or "")
        if isinstance(campaign_metadata.get("toolchain"), dict)
        else ""
    )
    expected_toolchain_identity_hash = str(
        ((campaign_metadata.get("toolchain") or {}).get("fingerprint") or "")
        if isinstance(campaign_metadata.get("toolchain"), dict)
        else ""
    )
    current_toolchain = collect_toolchain_fingerprint()
    current_toolchain_fingerprint = str(
        current_toolchain.get("runtime_fingerprint") or ""
    )
    if current_toolchain_fingerprint != expected_toolchain_fingerprint:
        raise RuntimeError(
            "runtime toolchain changed after the campaign fingerprint was created: "
            f"expected {expected_toolchain_fingerprint}, got {current_toolchain_fingerprint}"
        )
    current_toolchain_identity_hash = str(
        current_toolchain.get("fingerprint") or ""
    )
    if current_toolchain_identity_hash != expected_toolchain_identity_hash:
        raise RuntimeError(
            "toolchain identity changed after the campaign fingerprint was created: "
            f"expected {expected_toolchain_identity_hash}, "
            f"got {current_toolchain_identity_hash}"
        )
    firmware_sha256 = sha256_file(job.elf_path)

    base_record: dict[str, object] = {
        "index": index,
        "total": total,
        "mode": mode.key,
        "paper_label": mode.paper_label,
        "firmware": str(job.elf_path),
        "relative_path": str(job.relative_path),
        "family": job.family,
        "firmware_dir": job.firmware_dir,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "latest_log_path": str(stable_log_path),
        "campaign_fingerprint": campaign_fingerprint,
        "source_tree_sha256": source_tree_sha256,
        "toolchain_fingerprint": expected_toolchain_fingerprint,
        "toolchain_identity_hash": expected_toolchain_identity_hash,
        "firmware_sha256": firmware_sha256,
    }

    if not args.rerun and report_is_reusable(
        report_path,
        status_path,
        campaign_fingerprint=campaign_fingerprint,
        firmware_sha256=firmware_sha256,
        source_tree_sha256=source_tree_sha256,
        toolchain_runtime_fingerprint=expected_toolchain_fingerprint,
        toolchain_identity_hash=expected_toolchain_identity_hash,
    ):
        existing_status: dict[str, object] = {}
        try:
            with status_path.open() as handle:
                loaded_status = json.load(handle)
            if isinstance(loaded_status, dict):
                existing_status = loaded_status
        except Exception:
            pass
        record = {
            **base_record,
            **report_identity_fields(report_path),
            "attempt_id": str(existing_status.get("attempt_id") or ""),
            "attempt_output_dir": existing_status.get("attempt_output_dir"),
            "attempt_report_path": existing_status.get("attempt_report_path"),
            "log_path": existing_status.get("log_path") or str(stable_log_path),
            "status": "skipped_existing_report",
            "returncode": 0,
            "started_at": None,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": 0.0,
        }
        write_jsonl(manifest_path, record)
        write_summary(status_path, record)
        return record

    attempt_id = (
        datetime.now().strftime("%Y%m%dT%H%M%S")
        + f"-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    attempt_output_dir = output_dir / "attempts" / attempt_id
    attempt_report_path = expected_report_path(attempt_output_dir, job)
    log_path = attempt_output_dir / f"{job.elf_stem}_runner.log"
    attempt_status_path = attempt_output_dir / f"{job.elf_stem}_run_status.json"
    attempt_output_dir.mkdir(parents=True, exist_ok=False)
    base_record.update({
        "attempt_id": attempt_id,
        "attempt_output_dir": str(attempt_output_dir),
        "attempt_report_path": str(attempt_report_path),
        "attempt_status_path": str(attempt_status_path),
        "log_path": str(log_path),
    })
    command = build_command(
        python_exe=str(args.python),
        mode=mode,
        job=job,
        output_dir=attempt_output_dir,
        time_minutes=float(args.time),
        extra_args=extra_args,
    )
    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    record = {
        **base_record,
        "status": "running",
        "command": command,
        "started_at": started_at,
    }
    write_summary(status_path, record)
    write_summary(attempt_status_path, record)
    print(f"[{index}/{total}] START {mode.label}: {job.relative_path}", flush=True)
    print(f"    attempt: {attempt_output_dir}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["LSGEMU_ATTEMPT_ID"] = attempt_id
    env["LSGEMU_ATTEMPT_DIR"] = str(attempt_output_dir)
    env["LSGEMU_CAMPAIGN_FINGERPRINT"] = campaign_fingerprint
    env["LSGEMU_FIRMWARE_SHA256"] = firmware_sha256
    env["LSGEMU_SOURCE_TREE_SHA256"] = source_tree_sha256
    env["LSGEMU_TOOLCHAIN_RUNTIME_FINGERPRINT"] = (
        expected_toolchain_fingerprint
    )
    env["LSGEMU_TOOLCHAIN_IDENTITY_HASH"] = expected_toolchain_identity_hash
    timeout_seconds = (
        float(args.process_timeout_minutes) * 60.0
        if float(args.process_timeout_minutes or 0.0) > 0.0
        else None
    )

    timed_out = False
    resource_usage: dict[str, object] = {
        "schema": "lsgemu.serial_resource_usage.v1",
        "sampling_interval_seconds": 1.0,
        "io_counter_semantics": "sampled_process_tree_lower_bound",
        "samples": 0,
        "peak_process_tree_rss_kib": 0,
        "peak_root_hwm_kib": 0,
        "max_process_tree_members": 0,
        "min_host_mem_available_kib": 0,
        "min_host_swap_free_kib": 0,
        "process_tree_rchar_lower_bound": 0,
        "process_tree_wchar_lower_bound": 0,
        "process_tree_read_bytes_lower_bound": 0,
        "process_tree_write_bytes_lower_bound": 0,
    }
    with log_path.open("w") as log_file:
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            while True:
                sample_resources(process.pid, resource_usage)
                polled = process.poll()
                if polled is not None:
                    returncode = int(polled)
                    break
                if timeout_seconds is not None and (time.time() - started) >= timeout_seconds:
                    timed_out = True
                    log_file.write("\n# outer process timeout expired\n")
                    log_file.flush()
                    terminate_process_group(process)
                    returncode = 124
                    sample_resources(process.pid, resource_usage)
                    break
                time.sleep(1.0)
        except BaseException:
            terminate_process_group(process)
            aborted = {
                **base_record,
                "status": "aborted",
                "returncode": process.returncode,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(max(0.0, time.time() - started), 3),
                "command": command,
                "resource_usage": public_resource_usage(resource_usage),
            }
            write_jsonl(manifest_path, aborted)
            write_summary(status_path, aborted)
            write_summary(attempt_status_path, aborted)
            raise

    elapsed = max(0.0, time.time() - started)
    report_written_by_this_run = attempt_report_path.is_file()
    complete_report = bool(
        report_written_by_this_run
        and report_matches_run_identity(
            attempt_report_path,
            campaign_fingerprint=campaign_fingerprint,
            firmware_sha256=firmware_sha256,
            attempt_id=attempt_id,
            source_tree_sha256=source_tree_sha256,
            toolchain_runtime_fingerprint=expected_toolchain_fingerprint,
            toolchain_identity_hash=expected_toolchain_identity_hash,
        )
    )
    report_published = False
    report_hardlinked = False
    log_hardlinked = False
    publish_error = None
    try:
        if complete_report:
            atomic_publish_file(attempt_report_path, report_path)
            report_published = True
            report_hardlinked = os.path.samefile(attempt_report_path, report_path)
        atomic_publish_file(log_path, stable_log_path)
        log_hardlinked = os.path.samefile(log_path, stable_log_path)
    except Exception as exc:
        publish_error = str(exc)
    if timed_out:
        status = "timeout"
    elif returncode == 0 and complete_report and report_published:
        status = "success"
    elif returncode == 0:
        status = "missing_or_incomplete_report"
    else:
        status = "failed"

    record = {
        **base_record,
        **(
            report_identity_fields(attempt_report_path)
            if report_written_by_this_run
            else {}
        ),
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "report_complete": complete_report,
        "report_written_by_this_run": report_written_by_this_run,
        "report_published": report_published,
        "report_hardlinked": report_hardlinked,
        "log_hardlinked": log_hardlinked,
        "publish_error": publish_error,
        "command": command,
        "resource_usage": public_resource_usage(resource_usage),
    }
    write_jsonl(manifest_path, record)
    write_summary(status_path, record)
    write_summary(attempt_status_path, record)
    print(
        f"[{index}/{total}] END {mode.label}: {job.relative_path} "
        f"status={status} returncode={returncode} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return record


def main() -> int:
    args = parse_args()
    mode = MODES[args.mode]
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    campaign_dir = output_root
    manifest_path = campaign_dir / mode.label / f"{mode.label}_manifest.jsonl"
    summary_path = campaign_dir / mode.label / f"{mode.label}_summary.json"
    extra_args = normalized_extra_args(list(args.extra_lsgemu_args or []))
    campaign_metadata = build_campaign_metadata(
        args=args,
        mode=mode,
        benchmark_root=benchmark_root,
        output_root=output_root,
        extra_args=extra_args,
    )

    jobs = filter_jobs(discover_firmware_jobs(benchmark_root), args)
    if not jobs:
        raise SystemExit("no ELF jobs selected")

    print(f"Mode {args.mode}: {mode.paper_label} ({mode.label})")
    print(f"Benchmark root: {benchmark_root}")
    print(f"Output root: {output_root}")
    print(f"Campaign fingerprint: {campaign_metadata['campaign_fingerprint']}")
    print(f"Source-tree SHA-256: {campaign_metadata['source_tree']['sha256']}")
    llm_identity = campaign_metadata["configuration"]["llm_runtime_identity"]
    print(
        "LLM runtime: "
        f"model={llm_identity.get('model') or 'unset'} "
        f"credential={llm_identity.get('credential_source')}"
    )
    if (
        llm_identity.get("config_file_exists")
        and not llm_identity.get("credential_available")
    ):
        print(
            "WARNING: LLM config exists but no credential is available; "
            "runs will use deterministic/heuristic fallbacks only.",
            file=sys.stderr,
        )
    print(f"Selected ELF jobs: {len(jobs)}")
    for index, job in enumerate(jobs, 1):
        print(f"  {index:02d}. {job.relative_path}")

    if args.dry_run:
        return 0

    records: list[dict[str, object]] = []
    campaign_started = time.time()
    for index, job in enumerate(jobs, 1):
        record = run_one_job(
            args=args,
            mode=mode,
            job=job,
            index=index,
            total=len(jobs),
            campaign_dir=campaign_dir,
            manifest_path=manifest_path,
            extra_args=extra_args,
            campaign_metadata=campaign_metadata,
        )
        records.append(record)
        write_summary(
            summary_path,
            {
                "mode": mode.key,
                "paper_label": mode.paper_label,
                "benchmark_root": str(benchmark_root),
                "output_root": str(output_root),
                "total_jobs": len(jobs),
                "completed_records": len(records),
                "elapsed_seconds": round(max(0.0, time.time() - campaign_started), 3),
                "status_counts": {
                    status: sum(1 for item in records if item.get("status") == status)
                    for status in sorted({str(item.get("status")) for item in records})
                },
                "manifest_path": str(manifest_path),
                "campaign_metadata": campaign_metadata,
                "records": records,
            },
        )
        if args.stop_on_failure and int(record.get("returncode", 0) or 0) != 0:
            return int(record.get("returncode", 1) or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
