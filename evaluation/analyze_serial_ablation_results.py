#!/usr/bin/env python3
"""Aggregate serial full/ablation campaigns without dropping failed runs.

Final reports and durable JSONL checkpoints are deliberately kept as separate
observation types. Complete-pair statistics support direct ablation claims;
checkpoint observations are lower bounds used for failure-inclusive reporting.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.artifact_io import atomic_json_dump, atomic_write_text


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / ".lsgemu_runs" / "evaluation_serial"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "serial_ablation_first_round.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "serial_ablation_first_round.csv"

MODE_LABELS = {
    "full": "Full LSGEmu",
    "no_semantic_obligation": "Legacy stage-only semantic cut",
    "no_semantic_obligation_v2": "Without Semantic Obligation (v2)",
    "no_scoped_replay": "Legacy scoped-stage cut",
    "no_scoped_replay_v2": "Without Scoped Replay (v2)",
    "no_context_event_replay": "Without Context-aware Event Recovery",
}

CHECKPOINTS = {
    "10m": 600.0,
    "30m": 1800.0,
    "1h": 3600.0,
    "2h": 7200.0,
    "4h": 14400.0,
}

LEGACY_TREATMENT_WARNINGS = {
    "no_semantic_obligation": (
        "This stage-only cut retained semantic scores, hotsets, feedback, and "
        "downstream consumers; it is not an estimate of semantic-obligation discovery."
    ),
    "no_scoped_replay": (
        "This composite stage cut retained branch-specific snapshots, provenance, "
        "and learned PC/occurrence constraints while removing neighboring targeted "
        "phase families; it is not an estimate of evidence-scoped replay."
    ),
}

CONTRIBUTION_TREATMENTS = {
    "no_semantic_obligation_v2",
    "no_scoped_replay_v2",
    "no_context_event_replay",
}


@dataclass
class Observation:
    mode: str
    relative_path: str
    status: str
    returncode: int
    report_complete: bool
    source: str
    valid_covered_bbs: int
    valid_total_bbs: int
    elapsed_seconds: float
    report_path: Optional[str] = None
    progress_path: Optional[str] = None
    termination_reason: Optional[str] = None
    timeline: Dict[str, int] = field(default_factory=dict)
    attempt_id: Optional[str] = None
    identity_verified: bool = False
    progress_attempt_match: bool = False
    firmware_sha256: Optional[str] = None
    valid_bb_denominator_hash: Optional[str] = None
    source_tree_sha256: Optional[str] = None
    toolchain_fingerprint: Optional[str] = None
    toolchain_identity_hash: Optional[str] = None
    static_cache_identity_hash: Optional[str] = None
    identity_reasons: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        if self.valid_total_bbs <= 0:
            return 0.0
        return self.valid_covered_bbs / self.valid_total_bbs

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "relative_path": self.relative_path,
            "status": self.status,
            "returncode": self.returncode,
            "report_complete": self.report_complete,
            "source": self.source,
            "valid_covered_bbs": self.valid_covered_bbs,
            "valid_total_bbs": self.valid_total_bbs,
            "valid_coverage_rate": self.rate * 100.0,
            "elapsed_seconds": self.elapsed_seconds,
            "report_path": self.report_path,
            "progress_path": self.progress_path,
            "termination_reason": self.termination_reason,
            "timeline": dict(self.timeline),
            "attempt_id": self.attempt_id,
            "identity_verified": self.identity_verified,
            "progress_attempt_match": self.progress_attempt_match,
            "firmware_sha256": self.firmware_sha256,
            "valid_bb_denominator_hash": self.valid_bb_denominator_hash,
            "source_tree_sha256": self.source_tree_sha256,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "toolchain_identity_hash": self.toolchain_identity_hash,
            "static_cache_identity_hash": self.static_cache_identity_hash,
            "identity_reasons": list(self.identity_reasons),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def runtime_toolchain_fingerprint(value: object) -> str:
    """Return the runtime-only toolchain identity from report or record data."""
    if isinstance(value, dict):
        return str(value.get("runtime_fingerprint") or "")
    return str(value or "")


def full_toolchain_identity(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("fingerprint") or "")
    return ""


def static_cache_identity_hash(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("identity_hash") or "")
    return str(value or "")


def progress_records(
    path: Path,
    *,
    attempt_id: Optional[str] = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    with path.open(errors="replace") as f:
        for line in f:
            try:
                item = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            if attempt_id:
                if str(item.get("attempt_id") or "") != str(attempt_id):
                    continue
            try:
                valid_total = int(item.get("valid_total_bbs", 0) or 0)
                valid_covered = int(item.get("valid_covered_bbs", 0) or 0)
                elapsed = float(item.get("elapsed_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if valid_total <= 0 or valid_covered < 0 or elapsed < 0:
                continue
            records.append(item)
    return records


def latest_at_or_before(
    records: Iterable[dict[str, object]],
    seconds: float,
) -> Optional[dict[str, object]]:
    eligible = [
        item
        for item in records
        if float(item.get("elapsed_seconds", 0.0) or 0.0) <= seconds
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item.get("elapsed_seconds", 0.0) or 0.0),
            int(item.get("valid_covered_bbs", 0) or 0),
        ),
    )


def high_water_record(records: Iterable[dict[str, object]]) -> Optional[dict[str, object]]:
    records = list(records)
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            int(item.get("valid_covered_bbs", 0) or 0),
            float(item.get("elapsed_seconds", 0.0) or 0.0),
        ),
    )


def report_termination_reason(report: dict[str, object]) -> Optional[str]:
    reason = report.get("deadline_drain_stop_reason")
    if reason:
        return str(reason)
    phases = report.get("phase_metadata") or report.get("phases") or {}
    if isinstance(phases, dict):
        for phase_name in reversed(list(phases)):
            phase = phases.get(phase_name)
            if not isinstance(phase, dict):
                continue
            for key in ("stop_reason", "termination_reason", "deadline_drain_stop_reason"):
                if phase.get(key):
                    return str(phase[key])
            run_result = phase.get("run_result")
            if isinstance(run_result, dict) and run_result.get("stop_reason"):
                return str(run_result["stop_reason"])
    return None


def _path_from_record(record: dict[str, object], key: str) -> Optional[Path]:
    value = str(record.get(key) or "").strip()
    if not value:
        return None
    return Path(value)


def _report_identity_mismatches(
    report: Optional[dict[str, object]],
    record: dict[str, object],
) -> list[str]:
    mismatches: list[str] = []
    if not isinstance(report, dict):
        return ["missing_report"]
    metrics = report.get("evaluation_metrics")
    if not isinstance(metrics, dict):
        mismatches.append("missing_evaluation_metrics")
        return mismatches
    if str(metrics.get("schema") or "") != "lsgemu.evaluation_metrics.v1":
        mismatches.append("evaluation_metrics_schema_mismatch")
    expected_campaign = str(record.get("campaign_fingerprint") or "")
    expected_firmware = str(record.get("firmware_sha256") or "")
    expected_attempt = str(record.get("attempt_id") or "")
    expected_denominator = str(record.get("valid_bb_denominator_hash") or "")
    if not expected_campaign:
        mismatches.append("missing_expected_campaign_fingerprint")
    elif str(report.get("campaign_fingerprint") or "") != expected_campaign:
        mismatches.append("campaign_fingerprint_mismatch")
    if not expected_firmware:
        mismatches.append("missing_expected_firmware_sha256")
    elif str(report.get("firmware_sha256") or "") != expected_firmware:
        mismatches.append("firmware_sha256_mismatch")
    if not expected_attempt:
        mismatches.append("missing_expected_attempt_id")
    elif str(report.get("attempt_id") or "") != expected_attempt:
        mismatches.append("attempt_id_mismatch")
    if str(report.get("coverage_schema") or "") != "lsgemu.dynamic_coverage.v2":
        mismatches.append("coverage_schema_mismatch")
    report_denominator = str(report.get("valid_bb_denominator_hash") or "")
    if expected_denominator and report_denominator != expected_denominator:
        mismatches.append("valid_bb_denominator_hash_mismatch")
    if not report_denominator:
        mismatches.append("missing_valid_bb_denominator_hash")

    expected_source = str(record.get("source_tree_sha256") or "")
    report_source = str(report.get("source_tree_sha256") or "")
    if expected_source and report_source != expected_source:
        mismatches.append("source_tree_sha256_mismatch")
    expected_runtime_toolchain = str(record.get("toolchain_fingerprint") or "")
    report_runtime_toolchain = runtime_toolchain_fingerprint(
        report.get("toolchain_runtime_fingerprint")
        or report.get("toolchain_fingerprint")
    )
    if expected_runtime_toolchain and report_runtime_toolchain != expected_runtime_toolchain:
        mismatches.append("toolchain_runtime_fingerprint_mismatch")
    expected_toolchain_identity = str(record.get("toolchain_identity_hash") or "")
    report_toolchain_identity = str(report.get("toolchain_identity_hash") or "")
    if expected_toolchain_identity and report_toolchain_identity != expected_toolchain_identity:
        mismatches.append("toolchain_identity_hash_mismatch")
    expected_static_cache = str(record.get("static_cache_identity_hash") or "")
    report_static_cache = str(
        report.get("static_cache_identity_hash")
        or static_cache_identity_hash(report.get("static_cache_identity"))
        or ""
    )
    if expected_static_cache and report_static_cache != expected_static_cache:
        mismatches.append("static_cache_identity_hash_mismatch")
    return mismatches


def _report_matches_record(
    report: Optional[dict[str, object]],
    record: dict[str, object],
) -> bool:
    return not _report_identity_mismatches(report, record)


def _report_eligible_for_record(
    record: dict[str, object],
    report: Optional[dict[str, object]],
) -> bool:
    status = str(record.get("status") or "")
    try:
        returncode = int(record.get("returncode", 1))
    except (TypeError, ValueError):
        returncode = 1
    if status == "success":
        return bool(
            returncode == 0
            and bool(record.get("report_written_by_this_run"))
            and bool(record.get("report_published"))
            and bool(record.get("report_complete"))
            and _report_matches_record(report, record)
        )
    if status == "skipped_existing_report":
        return bool(
            returncode == 0
            and _report_matches_record(report, record)
        )
    return False


def _pair_identity_mismatches(
    full: Observation,
    ablated: Observation,
) -> list[str]:
    """Return reasons why a complete pair cannot support a causal estimate."""
    mismatches: list[str] = []
    if not full.report_complete or not ablated.report_complete:
        mismatches.append("incomplete_final_report")
    if not full.identity_verified or not ablated.identity_verified:
        mismatches.append("run_identity_unverified")
    if not full.firmware_sha256 or not ablated.firmware_sha256:
        mismatches.append("missing_firmware_identity")
    elif full.firmware_sha256 != ablated.firmware_sha256:
        mismatches.append("firmware_identity_mismatch")
    if not full.valid_bb_denominator_hash or not ablated.valid_bb_denominator_hash:
        mismatches.append("missing_denominator_identity")
    elif full.valid_bb_denominator_hash != ablated.valid_bb_denominator_hash:
        mismatches.append("denominator_identity_mismatch")
    if not full.source_tree_sha256 or not ablated.source_tree_sha256:
        mismatches.append("missing_source_tree_identity")
    elif full.source_tree_sha256 != ablated.source_tree_sha256:
        mismatches.append("source_tree_identity_mismatch")
    if not full.toolchain_fingerprint or not ablated.toolchain_fingerprint:
        mismatches.append("missing_runtime_toolchain_identity")
    elif full.toolchain_fingerprint != ablated.toolchain_fingerprint:
        mismatches.append("runtime_toolchain_identity_mismatch")
    if not full.toolchain_identity_hash or not ablated.toolchain_identity_hash:
        mismatches.append("missing_full_toolchain_identity")
    elif full.toolchain_identity_hash != ablated.toolchain_identity_hash:
        mismatches.append("full_toolchain_identity_mismatch")
    if (
        full.static_cache_identity_hash
        and ablated.static_cache_identity_hash
        and full.static_cache_identity_hash != ablated.static_cache_identity_hash
    ):
        mismatches.append("static_cache_identity_mismatch")
    return list(dict.fromkeys(mismatches))


def observation_from_record(mode: str, record: dict[str, object]) -> Observation:
    relative_path = str(record.get("relative_path") or "")
    stable_report_path = _path_from_record(record, "report_path")
    attempt_report_path = _path_from_record(record, "attempt_report_path")
    report_path = attempt_report_path or stable_report_path
    if (
        attempt_report_path is not None
        and not attempt_report_path.exists()
        and str(record.get("status") or "")
        in {"success", "skipped_existing_report"}
    ):
        report_path = stable_report_path
    output_dir = _path_from_record(record, "attempt_output_dir")
    if output_dir is None:
        output_dir = _path_from_record(record, "output_dir")
    if output_dir is None:
        output_dir = stable_report_path.parent if stable_report_path else Path(".")
    stem = Path(relative_path).stem
    attempt_id = str(record.get("attempt_id") or "") or None
    progress_path = output_dir / f"{stem}_coverage_progress.jsonl"
    progress = progress_records(progress_path, attempt_id=attempt_id)
    progress_high_water = high_water_record(progress)

    report: Optional[dict[str, object]] = None
    if report_path is not None and report_path.exists():
        try:
            report = load_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            report = None

    identity_mismatches = _report_identity_mismatches(report, record)
    identity_verified = _report_eligible_for_record(record, report)
    complete = bool(identity_verified)
    if complete:
        covered = int(report.get("valid_covered_bbs", 0) or 0)
        total = int(report.get("valid_total_bbs", 0) or 0)
        elapsed = float(
            report.get("execution_time_seconds", record.get("elapsed_seconds", 0.0))
            or 0.0
        )
        source = "final_report"
        termination_reason = report_termination_reason(report)
    else:
        if progress_high_water is None:
            covered = 0
            total = 0
            elapsed = float(record.get("elapsed_seconds", 0.0) or 0.0)
            source = "missing_observation"
            termination_reason = "no_durable_checkpoint"
        else:
            covered = int(
                progress_high_water.get("valid_covered_bbs", 0) or 0
            )
            total = int(progress_high_water.get("valid_total_bbs", 0) or 0)
            elapsed = float(
                progress_high_water.get("elapsed_seconds", 0.0) or 0.0
            )
            source = (
                "last_durable_checkpoint_lower_bound"
                if attempt_id
                else "legacy_unscoped_checkpoint_lower_bound"
            )
            termination_reason = str(
                progress_high_water.get("stage")
                or progress_high_water.get("event")
                or "checkpoint"
            )

    timeline: dict[str, int] = {}
    for checkpoint_name, checkpoint_seconds in CHECKPOINTS.items():
        checkpoint = latest_at_or_before(progress, checkpoint_seconds)
        if checkpoint is not None:
            timeline[checkpoint_name] = int(checkpoint.get("valid_covered_bbs", 0) or 0)
        elif complete and elapsed <= checkpoint_seconds:
            timeline[checkpoint_name] = covered
    timeline["final"] = covered

    return Observation(
        mode=mode,
        relative_path=relative_path,
        status=str(record.get("status") or "unknown"),
        returncode=int(record.get("returncode", 0) or 0),
        report_complete=complete,
        source=source,
        valid_covered_bbs=covered,
        valid_total_bbs=total,
        elapsed_seconds=elapsed,
        report_path=str(report_path) if report_path else None,
        progress_path=str(progress_path) if progress_path.exists() else None,
        termination_reason=termination_reason,
        timeline=timeline,
        attempt_id=attempt_id,
        identity_verified=identity_verified,
        progress_attempt_match=bool(progress and attempt_id),
        firmware_sha256=(
            str(
                record.get("firmware_sha256")
                or (report or {}).get("firmware_sha256")
                or (progress_high_water or {}).get("firmware_sha256")
                or ""
            )
            or None
        ),
        valid_bb_denominator_hash=(
            str(
                record.get("valid_bb_denominator_hash")
                or (report or {}).get("valid_bb_denominator_hash")
                or (progress_high_water or {}).get("valid_bb_denominator_hash")
                or ""
            )
            or None
        ),
        source_tree_sha256=(
            str(
                record.get("source_tree_sha256")
                or (report or {}).get("source_tree_sha256")
                or (progress_high_water or {}).get("source_tree_sha256")
                or ""
            )
            or None
        ),
        toolchain_fingerprint=(
            str(
                record.get("toolchain_fingerprint")
                or runtime_toolchain_fingerprint(
                    (report or {}).get("toolchain_runtime_fingerprint")
                    or (report or {}).get("toolchain_fingerprint")
                )
                or (progress_high_water or {}).get(
                    "toolchain_runtime_fingerprint"
                )
                or ""
            )
            or None
        ),
        toolchain_identity_hash=(
            str(
                record.get("toolchain_identity_hash")
                or (report or {}).get("toolchain_identity_hash")
                or full_toolchain_identity((report or {}).get("toolchain_fingerprint"))
                or (progress_high_water or {}).get("toolchain_identity_hash")
                or ""
            )
            or None
        ),
        static_cache_identity_hash=(
            str(
                record.get("static_cache_identity_hash")
                or (report or {}).get("static_cache_identity_hash")
                or static_cache_identity_hash((report or {}).get("static_cache_identity"))
                or (progress_high_water or {}).get("static_cache_identity_hash")
                or ""
            )
            or None
        ),
        identity_reasons=([] if identity_verified else identity_mismatches),
    )


def discover_mode_summaries(results_root: Path) -> dict[str, Path]:
    summaries: dict[str, Path] = {}
    for summary_path in sorted(results_root.glob("*/*_summary.json")):
        data = load_json(summary_path)
        mode = str(data.get("mode") or summary_path.parent.name)
        summaries[mode] = summary_path
    return summaries


def load_campaigns(results_root: Path) -> dict[str, dict[str, Observation]]:
    campaigns: dict[str, dict[str, Observation]] = {}
    for mode, summary_path in discover_mode_summaries(results_root).items():
        summary = load_json(summary_path)
        observations: dict[str, Observation] = {}
        for record in summary.get("records", []) or []:
            if not isinstance(record, dict):
                continue
            observation = observation_from_record(mode, record)
            observations[observation.relative_path] = observation
        campaigns[mode] = observations
    return campaigns


def exact_two_sided_sign_p(full_wins: int, ablated_wins: int) -> Optional[float]:
    n = full_wins + ablated_wins
    if n <= 0:
        return None
    tail = min(full_wins, ablated_wins)
    probability = 2.0 * sum(math.comb(n, index) for index in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def summarize_pairs(
    full: dict[str, Observation],
    ablated: dict[str, Observation],
    *,
    complete_only: bool,
    require_causal_identity: bool = False,
) -> dict[str, object]:
    common = sorted(set(full) & set(ablated))
    pairs = []
    excluded_identity_reasons: Counter[str] = Counter()
    for target in common:
        full_obs = full[target]
        ablated_obs = ablated[target]
        if complete_only and not (full_obs.report_complete and ablated_obs.report_complete):
            continue
        if require_causal_identity:
            identity_mismatches = _pair_identity_mismatches(full_obs, ablated_obs)
            if identity_mismatches:
                excluded_identity_reasons.update(identity_mismatches)
                continue
        if full_obs.valid_total_bbs <= 0 or ablated_obs.valid_total_bbs <= 0:
            continue
        if full_obs.valid_total_bbs != ablated_obs.valid_total_bbs:
            continue
        pairs.append((target, full_obs, ablated_obs))

    full_total = sum(item[1].valid_total_bbs for item in pairs)
    ablated_total = sum(item[2].valid_total_bbs for item in pairs)
    full_covered = sum(item[1].valid_covered_bbs for item in pairs)
    ablated_covered = sum(item[2].valid_covered_bbs for item in pairs)
    full_rates = [item[1].rate for item in pairs]
    ablated_rates = [item[2].rate for item in pairs]
    delta_rates = [(item[2].rate - item[1].rate) * 100.0 for item in pairs]
    full_wins = sum(item[1].valid_covered_bbs > item[2].valid_covered_bbs for item in pairs)
    ties = sum(item[1].valid_covered_bbs == item[2].valid_covered_bbs for item in pairs)
    ablated_wins = len(pairs) - full_wins - ties

    timeline: dict[str, object] = {}
    for checkpoint_name in [*CHECKPOINTS, "final"]:
        checkpoint_pairs = [
            item
            for item in pairs
            if checkpoint_name in item[1].timeline and checkpoint_name in item[2].timeline
        ]
        denominator = sum(item[1].valid_total_bbs for item in checkpoint_pairs)
        full_checkpoint = sum(item[1].timeline[checkpoint_name] for item in checkpoint_pairs)
        ablated_checkpoint = sum(item[2].timeline[checkpoint_name] for item in checkpoint_pairs)
        timeline[checkpoint_name] = {
            "pairs": len(checkpoint_pairs),
            "valid_total_bbs": denominator,
            "full_valid_covered_bbs": full_checkpoint,
            "ablated_valid_covered_bbs": ablated_checkpoint,
            "full_weighted_rate": (full_checkpoint / denominator * 100.0) if denominator else None,
            "ablated_weighted_rate": (ablated_checkpoint / denominator * 100.0) if denominator else None,
            "delta_pp_ablated_minus_full": (
                (ablated_checkpoint - full_checkpoint) / denominator * 100.0
                if denominator
                else None
            ),
        }

    rows = []
    for target, full_obs, ablated_obs in pairs:
        rows.append({
            "relative_path": target,
            "valid_total_bbs": full_obs.valid_total_bbs,
            "full_valid_covered_bbs": full_obs.valid_covered_bbs,
            "ablated_valid_covered_bbs": ablated_obs.valid_covered_bbs,
            "full_rate": full_obs.rate * 100.0,
            "ablated_rate": ablated_obs.rate * 100.0,
            "delta_bbs_ablated_minus_full": (
                ablated_obs.valid_covered_bbs - full_obs.valid_covered_bbs
            ),
            "delta_pp_ablated_minus_full": (ablated_obs.rate - full_obs.rate) * 100.0,
            "full_status": full_obs.status,
            "ablated_status": ablated_obs.status,
            "full_source": full_obs.source,
            "ablated_source": ablated_obs.source,
            "causal_identity_eligible": not bool(
                _pair_identity_mismatches(full_obs, ablated_obs)
            ),
            "full_firmware_sha256": full_obs.firmware_sha256,
            "ablated_firmware_sha256": ablated_obs.firmware_sha256,
            "full_valid_bb_denominator_hash": full_obs.valid_bb_denominator_hash,
            "ablated_valid_bb_denominator_hash": ablated_obs.valid_bb_denominator_hash,
        })

    return {
        "sample_policy": (
            "causal_complete_report_pairs"
            if complete_only and require_causal_identity
            else "complete_report_pairs"
            if complete_only
            else "failure_inclusive_checkpoint_lower_bounds"
        ),
        "pair_identity_policy": (
            "same_verified_firmware_denominator_source_tree_full_toolchain_and_matching_static_cache_when_available"
            if require_causal_identity
            else "report_identity_and_equal_valid_bb_count"
        ),
        "pairs": len(pairs),
        "excluded_identity_reasons": dict(excluded_identity_reasons),
        "valid_total_bbs": full_total if full_total == ablated_total else None,
        "full_valid_covered_bbs": full_covered,
        "ablated_valid_covered_bbs": ablated_covered,
        "full_weighted_rate": (full_covered / full_total * 100.0) if full_total else None,
        "ablated_weighted_rate": (ablated_covered / ablated_total * 100.0) if ablated_total else None,
        "delta_bbs_ablated_minus_full": ablated_covered - full_covered,
        "delta_pp_ablated_minus_full": (
            (ablated_covered / ablated_total - full_covered / full_total) * 100.0
            if full_total and ablated_total
            else None
        ),
        "full_mean_rate": statistics.mean(full_rates) * 100.0 if full_rates else None,
        "ablated_mean_rate": statistics.mean(ablated_rates) * 100.0 if ablated_rates else None,
        "median_delta_pp_ablated_minus_full": statistics.median(delta_rates) if delta_rates else None,
        "wins_full_tie_ablated": [full_wins, ties, ablated_wins],
        "exact_two_sided_sign_p": exact_two_sided_sign_p(full_wins, ablated_wins),
        "full_mean_elapsed_seconds": (
            statistics.mean(item[1].elapsed_seconds for item in pairs) if pairs else None
        ),
        "ablated_mean_elapsed_seconds": (
            statistics.mean(item[2].elapsed_seconds for item in pairs) if pairs else None
        ),
        "timeline": timeline,
        "rows": rows,
    }


def mode_summary(observations: dict[str, Observation]) -> dict[str, object]:
    statuses: dict[str, int] = {}
    returncodes: dict[str, int] = {}
    sources: dict[str, int] = {}
    for observation in observations.values():
        statuses[observation.status] = statuses.get(observation.status, 0) + 1
        returncode = str(observation.returncode)
        returncodes[returncode] = returncodes.get(returncode, 0) + 1
        sources[observation.source] = sources.get(observation.source, 0) + 1
    known = [item for item in observations.values() if item.valid_total_bbs > 0]
    denominator = sum(item.valid_total_bbs for item in known)
    covered = sum(item.valid_covered_bbs for item in known)
    return {
        "targets": len(observations),
        "complete_reports": sum(item.report_complete for item in observations.values()),
        "durable_observations": len(known),
        "status_counts": statuses,
        "returncode_counts": returncodes,
        "observation_source_counts": sources,
        "lower_bound_valid_covered_bbs": covered,
        "valid_total_bbs_sum": denominator,
        "lower_bound_weighted_rate": (covered / denominator * 100.0) if denominator else None,
    }


def nearest_rank(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(float(quantile) * len(ordered)) - 1))
    return ordered[index]


def aggregate_full_report_metrics(
    observations: dict[str, Observation],
) -> dict[str, object]:
    """Aggregate the report-native RQ3/RQ4 counters used by the paper."""
    reports: list[dict[str, object]] = []
    for observation in observations.values():
        if not observation.report_complete or not observation.report_path:
            continue
        report_path = Path(observation.report_path)
        if not report_path.exists():
            continue
        try:
            report = load_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        metrics = report.get("evaluation_metrics")
        if not isinstance(metrics, dict):
            continue
        if str(metrics.get("schema") or "") != "lsgemu.evaluation_metrics.v1":
            continue
        reports.append(report)

    category_fields = (
        "phase_count",
        "new_bbs_sum",
        "candidate_or_obligation_count",
        "replay_attempt_count",
        "successful_replay_count",
        "target_hit_count",
        "persisted_constraints",
        "scoped_constraints_added",
        "constraint_hits",
        "llm_calls",
    )
    categories: dict[str, dict[str, int]] = {}
    hypothesis_funnel: dict[str, int] = {}
    hypothesis_reject_reasons: dict[str, int] = {}

    for report in reports:
        metrics = report.get("evaluation_metrics") or {}
        repair_distribution = metrics.get("repair_distribution") or {}
        report_categories = repair_distribution.get("categories") or {}
        if isinstance(report_categories, dict):
            for category, raw_values in report_categories.items():
                if not isinstance(raw_values, dict):
                    continue
                totals = categories.setdefault(str(category), {field: 0 for field in category_fields})
                for field_name in category_fields:
                    try:
                        totals[field_name] += int(raw_values.get(field_name, 0) or 0)
                    except (TypeError, ValueError):
                        continue

        replay_validation = metrics.get("replay_validation") or {}
        if not isinstance(replay_validation, dict):
            continue
        for destination, source_name in (
            (hypothesis_funnel, "hypothesis_audit_candidates"),
            (hypothesis_reject_reasons, "hypothesis_audit_reject_reasons"),
        ):
            source = replay_validation.get(source_name) or {}
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                try:
                    destination[str(key)] = destination.get(str(key), 0) + int(value or 0)
                except (TypeError, ValueError):
                    continue

    valid_covered_union_sum = sum(
        int(report.get("valid_covered_bbs", 0) or 0) for report in reports
    )
    phase_yield_sum = sum(
        int(category.get("new_bbs_sum", 0) or 0) for category in categories.values()
    )
    elapsed = [float(report.get("execution_time_seconds", 0.0) or 0.0) for report in reports]
    termination_reasons: dict[str, int] = {}
    for report in reports:
        reason = report_termination_reason(report) or "unknown"
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    return {
        "sample_policy": "complete_full_reports",
        "reports": len(reports),
        "valid_covered_bbs_sum": valid_covered_union_sum,
        "phase_new_bb_yield_sum": phase_yield_sum,
        "phase_yield_minus_valid_union": phase_yield_sum - valid_covered_union_sum,
        "repair_distribution_categories": categories,
        "hypothesis_audit_funnel": hypothesis_funnel,
        "hypothesis_audit_reject_reasons_nonexclusive": hypothesis_reject_reasons,
        "runtime_seconds": {
            "mean": statistics.mean(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "p95_nearest_rank": nearest_rank(elapsed, 0.95),
        },
        "termination_reason_counts": termination_reasons,
    }


def write_csv(path: Path, campaigns: dict[str, dict[str, Observation]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "paper_label",
        "relative_path",
        "status",
        "returncode",
        "report_complete",
        "source",
        "valid_covered_bbs",
        "valid_total_bbs",
        "valid_coverage_rate",
        "elapsed_seconds",
        "termination_reason",
        "report_path",
        "progress_path",
        "attempt_id",
        "identity_verified",
        "progress_attempt_match",
        "firmware_sha256",
        "valid_bb_denominator_hash",
        "source_tree_sha256",
        "toolchain_fingerprint",
        "toolchain_identity_hash",
        "static_cache_identity_hash",
        "identity_reasons",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for mode in sorted(campaigns):
        for target in sorted(campaigns[mode]):
            observation = campaigns[mode][target]
            row = observation.to_dict()
            row["paper_label"] = MODE_LABELS.get(mode, mode)
            row["identity_reasons"] = ";".join(
                observation.identity_reasons
            )
            writer.writerow({key: row.get(key) for key in fields})
    atomic_write_text(path, buffer.getvalue(), durable=True)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    summary_paths = discover_mode_summaries(results_root)
    campaigns = load_campaigns(results_root)
    if "full" not in campaigns:
        raise SystemExit("missing full campaign")

    campaign_metadata: dict[str, object] = {}
    source_hashes: dict[str, str] = {}
    toolchain_runtime_hashes: dict[str, str] = {}
    toolchain_identity_hashes: dict[str, str] = {}
    for mode, summary_path in sorted(summary_paths.items()):
        summary = load_json(summary_path)
        metadata = summary.get("campaign_metadata")
        if not isinstance(metadata, dict):
            continue
        campaign_metadata[mode] = metadata
        source_tree = metadata.get("source_tree")
        if isinstance(source_tree, dict) and source_tree.get("sha256"):
            source_hashes[mode] = str(source_tree["sha256"])
        toolchain = metadata.get("toolchain")
        if isinstance(toolchain, dict):
            runtime_hash = runtime_toolchain_fingerprint(toolchain)
            identity_hash = full_toolchain_identity(toolchain)
            if runtime_hash:
                toolchain_runtime_hashes[mode] = runtime_hash
            if identity_hash:
                toolchain_identity_hashes[mode] = identity_hash

    full_source_hash = source_hashes.get("full", "")
    full_toolchain_runtime_hash = toolchain_runtime_hashes.get("full", "")

    def comparison_provenance(
        mode: str,
        observations: dict[str, Observation],
    ) -> dict[str, object]:
        mode_source_hash = source_hashes.get(mode, "")
        mode_toolchain_hash = toolchain_runtime_hashes.get(mode, "")
        reasons: list[str] = []
        if not full_source_hash or not mode_source_hash:
            reasons.append("missing_source_tree_fingerprint")
        elif full_source_hash != mode_source_hash:
            reasons.append("source_tree_fingerprint_mismatch")
        if not full_toolchain_runtime_hash or not mode_toolchain_hash:
            reasons.append("missing_runtime_toolchain_fingerprint")
        elif full_toolchain_runtime_hash != mode_toolchain_hash:
            reasons.append("runtime_toolchain_fingerprint_mismatch")
        causal_pairs = summarize_pairs(
            campaigns["full"],
            observations,
            complete_only=True,
            require_causal_identity=True,
        )
        if int(causal_pairs.get("pairs", 0) or 0) <= 0:
            reasons.append("no_verified_firmware_denominator_pair")
        return {
            "full_source_tree_sha256": full_source_hash or None,
            "mode_source_tree_sha256": mode_source_hash or None,
            "source_tree_match": bool(
                full_source_hash
                and mode_source_hash
                and full_source_hash == mode_source_hash
            ),
            "full_toolchain_runtime_fingerprint": (
                full_toolchain_runtime_hash or None
            ),
            "mode_toolchain_runtime_fingerprint": mode_toolchain_hash or None,
            "runtime_toolchain_match": bool(
                full_toolchain_runtime_hash
                and mode_toolchain_hash
                and full_toolchain_runtime_hash == mode_toolchain_hash
            ),
            "causal_pair_count": int(causal_pairs.get("pairs", 0) or 0),
            "causal_effect_eligible": not reasons,
            "ineligibility_reasons": reasons,
        }

    comparisons: dict[str, object] = {}
    for mode, observations in sorted(campaigns.items()):
        if mode == "full":
            continue
        provenance = comparison_provenance(mode, observations)
        treatment_is_v2 = mode in CONTRIBUTION_TREATMENTS
        treatment_reasons = list(
            provenance.get("ineligibility_reasons", []) or []
        )
        if not treatment_is_v2:
            treatment_reasons.insert(0, "legacy_non_identifying_treatment")
        comparisons[mode] = {
            "paper_label": MODE_LABELS.get(mode, mode),
            "treatment_definition": {
                "status": (
                    "legacy_non_identifying"
                    if mode in LEGACY_TREATMENT_WARNINGS
                    else "contribution_level"
                ),
                "causal_effect_eligible": bool(
                    treatment_is_v2
                    and provenance.get("causal_effect_eligible")
                ),
                "warning": LEGACY_TREATMENT_WARNINGS.get(mode),
                "eligibility_reasons": list(dict.fromkeys(treatment_reasons)),
            },
            "pair_provenance": provenance,
            "complete_pairs": summarize_pairs(
                campaigns["full"],
                observations,
                complete_only=True,
            ),
            "causal_effect_pairs": summarize_pairs(
                campaigns["full"],
                observations,
                complete_only=True,
                require_causal_identity=True,
            ),
            "failure_inclusive_lower_bounds": summarize_pairs(
                campaigns["full"],
                observations,
                complete_only=False,
            ),
        }

    common_complete = sorted(
        target
        for target in set.intersection(*(set(items) for items in campaigns.values()))
        if all(campaigns[mode][target].report_complete for mode in campaigns)
    )
    common_causal_complete = sorted(
        target
        for target in set.intersection(*(set(items) for items in campaigns.values()))
        if all(
            not _pair_identity_mismatches(
                campaigns["full"][target],
                campaigns[mode][target],
            )
            for mode in campaigns
            if mode != "full"
        )
    )
    output = {
        "schema": "lsgemu.serial_ablation_analysis.v1",
        "results_root": str(results_root),
        "campaign_provenance": {
            "metadata_by_mode": campaign_metadata,
            "source_tree_sha256_by_mode": source_hashes,
            "toolchain_runtime_fingerprint_by_mode": toolchain_runtime_hashes,
            "toolchain_identity_hash_by_mode": toolchain_identity_hashes,
            "all_modes_fingerprinted": len(campaign_metadata) == len(campaigns),
            "source_tree_consistent": bool(
                len(source_hashes) == len(campaigns)
                and len(set(source_hashes.values())) == 1
            ),
            "runtime_toolchain_consistent": bool(
                len(toolchain_runtime_hashes) == len(campaigns)
                and len(set(toolchain_runtime_hashes.values())) == 1
            ),
            "campaign_fingerprint_comparison_policy": (
                "mode-specific campaign fingerprints are intentionally not required "
                "to match; source tree, runtime toolchain, firmware, and denominator "
                "identities are compared instead"
            ),
        },
        "counting_policy": {
            "complete_pairs": "Both modes emitted final evaluation reports with identical valid-BB denominators.",
            "failure_inclusive": "Final reports are used when present; otherwise the maximum durable progress checkpoint is a lower bound. Censored comparisons are descriptive, not final treatment effects.",
            "legacy_semantic_warning": LEGACY_TREATMENT_WARNINGS["no_semantic_obligation"],
            "legacy_scoped_warning": LEGACY_TREATMENT_WARNINGS["no_scoped_replay"],
            "causal_estimate_policy": "A complete pair is necessary but not sufficient: legacy_non_identifying treatments remain descriptive, and final v2 estimates require matching source/configuration provenance.",
            "causal_pair_identity": (
                "Both final reports must be verified for this attempt and match on "
                "firmware SHA-256, valid-BB denominator hash, source-tree SHA-256, "
                "and runtime toolchain fingerprint. Campaign fingerprints may differ "
                "because the treatment configuration differs."
            ),
        },
        "modes": {
            mode: {
                "paper_label": MODE_LABELS.get(mode, mode),
                **mode_summary(observations),
            }
            for mode, observations in sorted(campaigns.items())
        },
        "comparisons": comparisons,
        "common_complete_all_modes": {
            "targets": len(common_complete),
            "relative_paths": common_complete,
        },
        "common_causal_complete_all_modes": {
            "targets": len(common_causal_complete),
            "relative_paths": common_causal_complete,
        },
        "full_complete_report_metrics": aggregate_full_report_metrics(campaigns["full"]),
        "observations": {
            mode: [
                observations[target].to_dict()
                for target in sorted(observations)
            ]
            for mode, observations in sorted(campaigns.items())
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output, args.output_json, indent=2, sort_keys=True)
    write_csv(args.output_csv, campaigns)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
