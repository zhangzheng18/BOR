#!/usr/bin/env python3
"""Build crash-fuzzing profiles from long LSGEmu simulation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

try:
    from .utils import file_sha256, safe_int
except ImportError:
    from utils import file_sha256, safe_int

try:
    from lsgemu.artifact_io import atomic_json_dump
except ImportError:  # pragma: no cover - supports running fuzzengine standalone.
    atomic_json_dump = None


def _phase_map(report: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    phases: Dict[str, Dict[str, object]] = {}
    for container_key in ("phase_metadata", "phases"):
        container = report.get(container_key)
        if not isinstance(container, dict):
            continue
        for name, phase in container.items():
            if isinstance(phase, dict):
                current = phases.setdefault(str(name), {})
                current.update(phase)
    return phases


def _count_debug_records(phase: Dict[str, object]) -> int:
    total = 0
    for key in ("debug_records", "candidate_debug_records", "materialized_records"):
        value = phase.get(key)
        if isinstance(value, list):
            total += len(value)
    return total


def _stream_input_score(phase_name: str, phase: Dict[str, object]) -> int:
    score = 0
    if "stream_input" in phase_name:
        score += 4
    score += min(safe_int(phase.get("profiles_discovered")), 16)
    score += min(safe_int(phase.get("tasks_with_new_bbs")), 16)
    score += min(safe_int(phase.get("stream_input_summary_event_count")), 16)
    for record in phase.get("debug_records") or []:
        if not isinstance(record, dict):
            continue
        if safe_int(record.get("input_read_hit_count")) > 0:
            score += 4
        if safe_int(record.get("stream_input_payload_write_count")) > 0:
            score += 4
        if safe_int(record.get("new_bbs")) > 0:
            score += 1
    return score


def _mmio_score(phase_name: str, phase: Dict[str, object]) -> int:
    score = 0
    text = f"{phase_name} {json.dumps(phase, ensure_ascii=False)[:4096]}".lower()
    if "mmio" in text:
        score += 2
    if phase.get("mmio_access_tail"):
        score += 4
    score += min(safe_int(phase.get("constraints_applied")), 32)
    score += min(safe_int(phase.get("branch_mmio_successes")), 32)
    return score


def _irq_score(phase_name: str, phase: Dict[str, object]) -> int:
    score = 0
    text = phase_name.lower()
    if any(token in text for token in ("isr", "irq", "interrupt")):
        score += 4
    score += min(safe_int(phase.get("learned_irq_candidate_count")), 16)
    score += min(safe_int(phase.get("reservoir_interrupt_contexts")), 16)
    score += min(safe_int(phase.get("reservoir_interrupt_contexts_added")), 16)
    return score


@dataclass
class FuzzProfile:
    """Reusable profile extracted from a long LSGEmu simulation."""

    schema: str
    report_path: str
    report_sha256: str
    created_at: str
    firmware: str
    profile_mode: str = "guidance_only"
    source_run_profile_hash: str = ""
    strict_real_entry_replayable: Optional[bool] = None
    valid_covered_bbs: int = 0
    valid_total_bbs: int = 0
    coverage_rate: Optional[float] = None
    stream_profile_score: int = 0
    mmio_environment_score: int = 0
    irq_environment_score: int = 0
    recommended_time_minutes: float = 0.25
    recommended_timeout_seconds: float = 120.0
    recommended_extra_args: List[str] = field(default_factory=list)
    recommended_env: Dict[str, str] = field(default_factory=dict)
    stream_phases: List[Dict[str, object]] = field(default_factory=list)
    mmio_phases: List[Dict[str, object]] = field(default_factory=list)
    irq_phases: List[Dict[str, object]] = field(default_factory=list)
    crash_triage_summary: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FuzzProfile":
        return cls(
            schema=str(payload.get("schema") or "lsgemu.fuzzengine.profile.v1"),
            report_path=str(payload.get("report_path") or ""),
            report_sha256=str(payload.get("report_sha256") or ""),
            created_at=str(payload.get("created_at") or ""),
            firmware=str(payload.get("firmware") or ""),
            profile_mode=str(payload.get("profile_mode") or "guidance_only"),
            source_run_profile_hash=str(payload.get("source_run_profile_hash") or ""),
            strict_real_entry_replayable=payload.get("strict_real_entry_replayable"),
            valid_covered_bbs=safe_int(payload.get("valid_covered_bbs")),
            valid_total_bbs=safe_int(payload.get("valid_total_bbs")),
            coverage_rate=float(payload["coverage_rate"]) if payload.get("coverage_rate") is not None else None,
            stream_profile_score=safe_int(payload.get("stream_profile_score")),
            mmio_environment_score=safe_int(payload.get("mmio_environment_score")),
            irq_environment_score=safe_int(payload.get("irq_environment_score")),
            recommended_time_minutes=float(payload.get("recommended_time_minutes") or 0.25),
            recommended_timeout_seconds=float(payload.get("recommended_timeout_seconds") or 120.0),
            recommended_extra_args=[str(item) for item in payload.get("recommended_extra_args") or []],
            recommended_env={str(k): str(v) for k, v in dict(payload.get("recommended_env") or {}).items()},
            stream_phases=list(payload.get("stream_phases") or []),
            mmio_phases=list(payload.get("mmio_phases") or []),
            irq_phases=list(payload.get("irq_phases") or []),
            crash_triage_summary=dict(payload.get("crash_triage_summary") or {}),
            notes=[str(item) for item in payload.get("notes") or []],
        )


def build_profile_from_report(report_path: object) -> FuzzProfile:
    path = Path(report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    report_digest = file_sha256(path)

    phases = _phase_map(report)
    stream_phases: List[Dict[str, object]] = []
    mmio_phases: List[Dict[str, object]] = []
    irq_phases: List[Dict[str, object]] = []
    stream_score = mmio_score_total = irq_score_total = 0

    for name, phase in phases.items():
        stream_score_phase = _stream_input_score(name, phase)
        mmio_score_phase = _mmio_score(name, phase)
        irq_score_phase = _irq_score(name, phase)
        stream_score += stream_score_phase
        mmio_score_total += mmio_score_phase
        irq_score_total += irq_score_phase
        common = {
            "name": name,
            "debug_records": _count_debug_records(phase),
            "new_bbs": phase.get("new_bbs"),
            "tasks_run": phase.get("tasks_run"),
            "tasks_with_new_bbs": phase.get("tasks_with_new_bbs"),
        }
        if stream_score_phase > 0:
            stream_phases.append({**common, "score": stream_score_phase})
        if mmio_score_phase > 0:
            mmio_phases.append({**common, "score": mmio_score_phase})
        if irq_score_phase > 0:
            irq_phases.append({**common, "score": irq_score_phase})

    stream_phases.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
    mmio_phases.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
    irq_phases.sort(key=lambda item: int(item.get("score") or 0), reverse=True)

    recommended_extra_args = _recommended_replay_args(stream_phases)
    notes = [
        "profile extracted from a long LSGEmu run; campaign uses short crash-focused replay",
        "profile_mode=guidance_only: the report guides scheduler settings, it does not restore snapshots",
        "coverage fields are context only and are not the fuzzing objective",
    ]
    if stream_score <= 0:
        notes.append("no strong stream-input evidence found; fuzzing may fall back to full LSGEmu replay semantics")
    if mmio_score_total > 0 or irq_score_total > 0:
        notes.append("peripheral/IRQ evidence exists; classify crash trigger source before claiming device-reproducibility")

    return FuzzProfile(
        schema="lsgemu.fuzzengine.profile.v1",
        report_path=str(path),
        report_sha256=report_digest,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        firmware=str(report.get("firmware") or ""),
        profile_mode="guidance_only",
        source_run_profile_hash=str(report.get("run_profile_hash") or ""),
        strict_real_entry_replayable=report.get("strict_real_entry_replayable"),
        valid_covered_bbs=safe_int(report.get("valid_covered_bbs")),
        valid_total_bbs=safe_int(report.get("valid_total_bbs")),
        coverage_rate=float(report["valid_coverage_rate"]) if report.get("valid_coverage_rate") is not None else None,
        stream_profile_score=stream_score,
        mmio_environment_score=mmio_score_total,
        irq_environment_score=irq_score_total,
        recommended_time_minutes=0.25 if stream_score > 0 else 1.0,
        recommended_timeout_seconds=120.0 if stream_score > 0 else 240.0,
        recommended_extra_args=recommended_extra_args,
        recommended_env={
            "LSGEMU_CRASH_TRIAGE": "1",
            "LSGEMU_RUNTIME_CRASH_MONITOR": "1",
            "LSGEMU_RUNTIME_CRASH_MONITOR_INVALID_MEMORY": "1",
            "LSGEMU_RECORD_PHASE_BB_LISTS": "1",
            "LSGEMU_PHASE_BB_LIST_LIMIT": "0",
            "LSGEMU_STREAM_FIRST_PASS_SEEDS": "1",
            "LSGEMU_FUZZ_PROFILE_REPORT": str(path),
            "LSGEMU_FUZZ_PROFILE_SHA256": report_digest,
        },
        stream_phases=stream_phases[:16],
        mmio_phases=mmio_phases[:16],
        irq_phases=irq_phases[:16],
        crash_triage_summary=dict(report.get("crash_triage_summary") or {}),
        notes=notes,
    )


def _recommended_replay_args(stream_phases: Sequence[Dict[str, object]]) -> List[str]:
    """Build short crash-replay scheduler args from profile evidence."""
    return [
        "--interleaved-arg", "--baseline-timeout-seconds",
        "--interleaved-arg", "30",
        "--interleaved-arg", "--stream-input-max-profiles",
        "--interleaved-arg", str(max(4, min(16, len(stream_phases) or 4))),
        "--interleaved-arg", "--stream-input-max-seeds",
        "--interleaved-arg", "1",
        "--interleaved-arg", "--stream-input-replay-timeout-us",
        "--interleaved-arg", "300000",
        "--interleaved-arg", "--disable-frontier-successor-replay-tail-use-remaining",
        "--interleaved-arg", "--disable-deadline-drain",
    ]


def load_profile(path: object) -> FuzzProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile must be a JSON object: {path}")
    if str(payload.get("schema") or "").startswith("lsgemu.fuzzengine.profile."):
        return FuzzProfile.from_dict(payload)
    return build_profile_from_report(path)


def write_profile(profile: FuzzProfile, path: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.to_dict()
    if atomic_json_dump is not None:
        atomic_json_dump(payload, output, indent=2, ensure_ascii=False)
    else:
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fuzzengine profile from an LSGEmu report")
    parser.add_argument("--report", required=True, help="LSGEmu *_interleaved_report.json from a long run")
    parser.add_argument("--out", required=True, help="output profile JSON")
    args = parser.parse_args(argv)
    profile = build_profile_from_report(args.report)
    output = write_profile(profile, args.out)
    print(json.dumps({"profile": str(output), **profile.to_dict()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
