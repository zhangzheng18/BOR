#!/usr/bin/env python3
"""Run LSGEmu once with a concrete stream-input seed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

try:
    from .crash_detector import CrashDetector, CrashDetectorConfig, CrashReport, NORMAL, detect_many_from_lsgemu_report
    from .subprocess_runner import run_supervised
    from .utils import parse_int_set, read_json_object, safe_int
except ImportError:
    from crash_detector import CrashDetector, CrashDetectorConfig, CrashReport, NORMAL, detect_many_from_lsgemu_report
    from subprocess_runner import run_supervised
    from utils import parse_int_set, read_json_object, safe_int


def latest_progress(progress_path: Path) -> Optional[Dict[str, object]]:
    if not progress_path.exists():
        return None
    last: Optional[Dict[str, object]] = None
    try:
        with progress_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    last = payload
    except Exception:
        return last
    return last


@dataclass
class LSGEmuRunConfig:
    firmware: str
    lsgemu_path: str = "lsgemu/lsgemu.py"
    python: str = sys.executable
    time_minutes: float = 2.0
    mode: str = "interleaved"
    max_instructions: int = 0
    output_root: str = ".lsgemu_fuzz_runs"
    run_profile: Optional[str] = None
    crash_config: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    timeout_seconds: Optional[float] = None
    profile_path: Optional[str] = None
    profile_sha256: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LSGEmuRunConfig":
        return cls(
            firmware=str(payload["firmware"]),
            lsgemu_path=str(payload.get("lsgemu_path") or "lsgemu/lsgemu.py"),
            python=str(payload.get("python") or sys.executable),
            time_minutes=float(payload.get("time_minutes") or 2.0),
            mode=str(payload.get("mode") or "interleaved"),
            max_instructions=int(payload.get("max_instructions") or 0),
            output_root=str(payload.get("output_root") or ".lsgemu_fuzz_runs"),
            run_profile=str(payload["run_profile"]) if payload.get("run_profile") else None,
            crash_config=str(payload["crash_config"]) if payload.get("crash_config") else None,
            extra_args=list(payload.get("extra_args") or []),
            env={str(k): str(v) for k, v in dict(payload.get("env") or {}).items()},
            cwd=str(payload["cwd"]) if payload.get("cwd") else None,
            timeout_seconds=float(payload["timeout_seconds"]) if payload.get("timeout_seconds") else None,
            profile_path=str(payload["profile_path"]) if payload.get("profile_path") else None,
            profile_sha256=str(payload["profile_sha256"]) if payload.get("profile_sha256") else None,
        )


@dataclass
class LSGEmuSeedRunResult:
    seed_id: str
    seed_sha256: str
    output_dir: str
    report_path: Optional[str]
    progress_path: Optional[str]
    returncode: int
    timed_out: bool
    elapsed_seconds: float
    covered_bbs: int = 0
    valid_covered_bbs: int = 0
    valid_total_bbs: int = 0
    coverage_rate: Optional[float] = None
    valid_coverage_rate: Optional[float] = None
    covered_bb_list: List[int] = field(default_factory=list)
    new_bbs: int = 0
    crash_reports: List[Dict[str, object]] = field(default_factory=list)
    process_crash_report: Optional[Dict[str, object]] = None
    command: List[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LSGEmuSeedRunner:
    """Execute LSGEmu in a supervised subprocess for one concrete seed."""

    def __init__(self, config: LSGEmuRunConfig, detector_config: Optional[CrashDetectorConfig] = None):
        self.config = config
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        if detector_config is None and config.crash_config:
            detector_config = CrashDetectorConfig.from_dict(
                json.loads(Path(config.crash_config).read_text(encoding="utf-8"))
            )
        self.detector = CrashDetector(detector_config or CrashDetectorConfig())

    def run_seed(self, seed_id: str, seed_bytes: bytes, *, seed_sha256: str) -> LSGEmuSeedRunResult:
        started = time.time()
        output_dir = self.output_root / f"{seed_id}_{seed_sha256[:10]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        firmware = Path(self.config.firmware)
        report_path = output_dir / f"{firmware.stem}_interleaved_report.json"
        progress_path = output_dir / f"{firmware.stem}_coverage_progress.jsonl"
        command = self._command(output_dir)
        env = self._env(seed_bytes)
        timeout = self.config.timeout_seconds
        if timeout is None:
            timeout = max(60.0, float(self.config.time_minutes) * 60.0 + 300.0)

        supervised = run_supervised(
            command,
            timeout=timeout,
            detector=self.detector,
            firmware=str(firmware),
            input_id=seed_id,
            seed_path=None,
            env=env,
            cwd=self.config.cwd,
        )
        process_report = supervised.crash_report
        report_payload = read_json_object(report_path)
        progress_payload = latest_progress(progress_path)
        summary_source, summary_source_name = best_summary_source(report_payload, progress_payload)

        covered_list = parse_int_set(summary_source.get("covered_bb_list"))
        covered_list.update(parse_int_set(summary_source.get("global_covered_bb_list")))
        if not covered_list and report_payload:
            for phase in (report_payload.get("phase_metadata") or {}).values():
                if isinstance(phase, dict):
                    covered_list.update(parse_int_set(phase.get("covered_bb_list")))

        crash_reports: List[CrashReport] = []
        if report_payload:
            crash_reports.extend(detect_many_from_lsgemu_report(report_payload, self.detector, firmware=str(firmware)))
        if process_report.category != NORMAL:
            crash_reports.append(process_report)

        result = LSGEmuSeedRunResult(
            seed_id=seed_id,
            seed_sha256=seed_sha256,
            output_dir=str(output_dir),
            report_path=str(report_path) if report_path.exists() else None,
            progress_path=str(progress_path) if progress_path.exists() else None,
            returncode=int(supervised.returncode),
            timed_out=bool(supervised.timed_out),
            elapsed_seconds=time.time() - started,
            covered_bbs=safe_int(summary_source.get("covered_bbs"), len(covered_list) or 0),
            valid_covered_bbs=safe_int(summary_source.get("valid_covered_bbs"), 0),
            valid_total_bbs=safe_int(summary_source.get("valid_total_bbs"), 0),
            coverage_rate=_float_or_none(summary_source.get("coverage_rate")),
            valid_coverage_rate=_float_or_none(summary_source.get("valid_coverage_rate")),
            covered_bb_list=sorted(covered_list),
            crash_reports=[item.to_dict() for item in crash_reports],
            process_crash_report=process_report.to_dict(),
            command=list(command),
            stdout_tail=supervised.stdout_tail,
            stderr_tail=supervised.stderr_tail,
            metadata={
                "summary_source": summary_source_name,
                "report_exists": report_path.exists(),
                "progress_exists": progress_path.exists(),
                "profile_path": self.config.profile_path,
                "profile_sha256": self.config.profile_sha256,
                "profile_reuse_enabled": bool(self.config.profile_path),
            },
        )
        return result

    def _command(self, output_dir: Path) -> List[str]:
        command = [
            self.config.python,
            self.config.lsgemu_path,
            self.config.firmware,
            "--time",
            str(float(self.config.time_minutes)),
            "--mode",
            self.config.mode,
            "--max-instructions",
            str(int(self.config.max_instructions)),
            "--output-dir",
            str(output_dir),
        ]
        if self.config.run_profile:
            command.extend(["--run-profile", self.config.run_profile])
        command.extend(_normalize_interleaved_args(self.config.extra_args))
        return command

    def _env(self, seed_bytes: bytes) -> Dict[str, str]:
        env = dict(self.config.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("LSGEMU_CRASH_TRIAGE", "1")
        env.setdefault("LSGEMU_RUNTIME_CRASH_MONITOR", "1")
        env.setdefault("LSGEMU_RUNTIME_CRASH_MONITOR_INVALID_MEMORY", "1")
        env.setdefault("LSGEMU_RECORD_PHASE_BB_LISTS", "1")
        env.setdefault("LSGEMU_PHASE_BB_LIST_LIMIT", "0")
        env.setdefault("LSGEMU_STREAM_FIRST_PASS_SEEDS", "1")
        if self.config.profile_path:
            env.setdefault("LSGEMU_FUZZ_PROFILE_REPORT", self.config.profile_path)
        if self.config.profile_sha256:
            env.setdefault("LSGEMU_FUZZ_PROFILE_SHA256", self.config.profile_sha256)
        env["LSGEMU_STREAM_EXTRA_SEEDS_HEX"] = seed_bytes[:4096].hex()
        return env


def _normalize_interleaved_args(values: Sequence[str]) -> List[str]:
    """Keep option-like values attached to the outer forwarding option.

    The top-level ``lsgemu.py`` parser owns ``--interleaved-arg``.  A value
    such as ``--baseline-timeout-seconds`` would otherwise be parsed as a new
    top-level option instead of the value of ``--interleaved-arg``.  The
    ``--name=value`` spelling is accepted by argparse and preserves the raw
    argument passed to the interleaved scheduler.
    """
    normalized: List[str] = []
    items = [str(item) for item in values]
    index = 0
    while index < len(items):
        item = items[index]
        if item == "--interleaved-arg" and index + 1 < len(items):
            value = items[index + 1]
            if value.startswith("-"):
                normalized.append(f"--interleaved-arg={value}")
            else:
                normalized.extend([item, value])
            index += 2
            continue
        normalized.append(item)
        index += 1
    return normalized


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _summary_score(payload: Optional[Dict[str, object]]) -> tuple[int, int, int]:
    if not payload:
        return (0, 0, 0)
    valid = safe_int(payload.get("valid_covered_bbs"), 0)
    covered = safe_int(payload.get("covered_bbs"), 0)
    if covered <= 0:
        covered = len(parse_int_set(payload.get("covered_bb_list"))) + len(parse_int_set(payload.get("global_covered_bb_list")))
    elapsed = _float_or_none(payload.get("elapsed_seconds") or payload.get("time_seconds") or payload.get("timestamp")) or 0.0
    return (valid, covered, int(elapsed))


def best_summary_source(
    report_payload: Optional[Dict[str, object]],
    progress_payload: Optional[Dict[str, object]],
) -> tuple[Dict[str, object], str]:
    """Choose the most informative coverage summary.

    A long campaign may be killed after progress JSONL is written but before the
    final report catches up. Prefer the source with the higher valid/all coverage
    score instead of blindly trusting a stale final report.
    """
    if report_payload and progress_payload and _summary_score(progress_payload) > _summary_score(report_payload):
        return progress_payload, "progress"
    if report_payload:
        return report_payload, "report"
    if progress_payload:
        return progress_payload, "progress"
    return {}, "process"
