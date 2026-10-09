#!/usr/bin/env python3
"""Monitor and summarize a strict elfmultifuzz campaign output directory."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COMPLETED_STATUSES = {
    "completed",
    "completed_cached",
    "completed_best_progress_after_retry",
    "completed_strict_checkpoint_union",
    "completed_partial_strict_resource_union",
}


STATIC_REACHABILITY_KEYS = (
    "extended_static_roots_available",
    "extended_static_root_fallback",
    "valid_entry_plus_vector_static_reachable_bbs",
    "valid_entry_plus_vector_static_reachable_covered_bbs",
    "valid_entry_plus_vector_static_reachable_uncovered_bbs",
    "valid_entry_plus_vector_reachable_coverage_rate",
    "valid_entry_static_reachable_bbs",
    "valid_entry_static_reachable_covered_bbs",
    "valid_entry_static_reachable_coverage_rate",
    "valid_static_unreachable_from_entry_or_vectors",
    "static_unreachable_valid_ratio",
    "extended_static_root_bbs",
    "extended_static_root_categories",
    "extended_static_root_examples",
    "valid_extended_static_reachable_bbs",
    "valid_extended_static_reachable_covered_bbs",
    "valid_extended_static_reachable_uncovered_bbs",
    "valid_extended_static_reachable_coverage_rate",
    "valid_static_unreachable_from_extended_roots",
    "extended_static_unreachable_valid_ratio",
    "extended_static_reachable_uncovered_bbs",
)


def has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def copy_missing_static_fields(target: Dict[str, Any], *sources: Dict[str, Any]) -> None:
    for key in STATIC_REACHABILITY_KEYS:
        if has_value(target.get(key)):
            continue
        for source in sources:
            value = source.get(key) if isinstance(source, dict) else None
            if has_value(value):
                target[key] = value
                break


def merge_static_high_water(target: Dict[str, Any], *sources: Dict[str, Any]) -> None:
    for key in (
        "valid_entry_plus_vector_static_reachable_covered_bbs",
        "valid_entry_static_reachable_covered_bbs",
        "valid_extended_static_reachable_covered_bbs",
    ):
        best_value = intval(target.get(key))
        for source in sources:
            if isinstance(source, dict):
                best_value = max(best_value, intval(source.get(key)))
        if best_value:
            target[key] = best_value


def refresh_static_rates(item: Dict[str, Any]) -> None:
    valid_total = intval(item.get("valid_total_bbs"))
    for total_key, covered_key, uncovered_key, rate_key in (
        (
            "valid_entry_plus_vector_static_reachable_bbs",
            "valid_entry_plus_vector_static_reachable_covered_bbs",
            "valid_entry_plus_vector_static_reachable_uncovered_bbs",
            "valid_entry_plus_vector_reachable_coverage_rate",
        ),
        (
            "valid_extended_static_reachable_bbs",
            "valid_extended_static_reachable_covered_bbs",
            "valid_extended_static_reachable_uncovered_bbs",
            "valid_extended_static_reachable_coverage_rate",
        ),
    ):
        total = intval(item.get(total_key))
        covered = intval(item.get(covered_key))
        if total:
            item[rate_key] = (covered / total * 100.0) if covered else pct(item.get(rate_key))
            item[uncovered_key] = intval(item.get(uncovered_key)) or max(0, total - covered)
    static_unreachable = intval(item.get("valid_static_unreachable_from_entry_or_vectors"))
    extended_unreachable = intval(item.get("valid_static_unreachable_from_extended_roots"))
    if valid_total:
        if static_unreachable and not has_value(item.get("static_unreachable_valid_ratio")):
            item["static_unreachable_valid_ratio"] = static_unreachable / valid_total * 100.0
        if extended_unreachable and not has_value(item.get("extended_static_unreachable_valid_ratio")):
            item["extended_static_unreachable_valid_ratio"] = extended_unreachable / valid_total * 100.0


def has_extended_root_evidence(item: Dict[str, Any]) -> bool:
    return bool(
        intval(item.get("extended_static_root_bbs"))
        or has_value(item.get("extended_static_root_categories"))
        or has_value(item.get("extended_static_root_examples"))
    )


def ensure_extended_static_fallback(item: Dict[str, Any]) -> None:
    """Keep old campaign reports readable when extended-root fields are absent."""
    if item.get("extended_static_roots_available") is True:
        return

    if item.get("extended_static_roots_available") is False:
        if not has_extended_root_evidence(item):
            item.setdefault("extended_static_root_fallback", "entry_plus_vector")
        if intval(item.get("valid_extended_static_reachable_bbs")):
            return

    if has_extended_root_evidence(item) and intval(item.get("valid_extended_static_reachable_bbs")):
        item["extended_static_roots_available"] = True
        return

    if intval(item.get("valid_extended_static_reachable_bbs")) and not has_extended_root_evidence(item):
        item["extended_static_roots_available"] = False
        item.setdefault("extended_static_root_fallback", "entry_plus_vector")
        return

    entry_total = intval(item.get("valid_entry_plus_vector_static_reachable_bbs"))
    if not entry_total:
        item.setdefault("extended_static_roots_available", False)
        return

    entry_covered = intval(item.get("valid_entry_plus_vector_static_reachable_covered_bbs"))
    entry_uncovered = intval(item.get("valid_entry_plus_vector_static_reachable_uncovered_bbs"))
    if not entry_uncovered:
        entry_uncovered = max(0, entry_total - entry_covered)

    item["valid_extended_static_reachable_bbs"] = entry_total
    item["valid_extended_static_reachable_covered_bbs"] = entry_covered
    item["valid_extended_static_reachable_uncovered_bbs"] = entry_uncovered
    item["valid_extended_static_reachable_coverage_rate"] = (
        entry_covered / entry_total * 100.0
    ) if entry_total else 0.0
    item["valid_static_unreachable_from_extended_roots"] = intval(
        item.get("valid_static_unreachable_from_entry_or_vectors")
    )
    valid_total = intval(item.get("valid_total_bbs"))
    if valid_total:
        item["extended_static_unreachable_valid_ratio"] = (
            intval(item.get("valid_static_unreachable_from_extended_roots")) / valid_total * 100.0
        )
    item["extended_static_roots_available"] = False
    item["extended_static_root_fallback"] = "entry_plus_vector"


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        pass
    return records


def latest_progress(progress_file: Path) -> Optional[Dict[str, Any]]:
    records = load_jsonl(progress_file)
    return records[-1] if records else None


def progress_score(record: Optional[Dict[str, Any]]) -> Tuple[int, float, float]:
    if not record:
        return (0, 0.0, 0.0)
    return (
        intval(record.get("valid_covered_bbs")),
        pct(record.get("valid_coverage_rate")),
        pct(record.get("elapsed_seconds")),
    )


def parse_int_set(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(value, 0) & ~1 if isinstance(value, str) else int(value) & ~1)
        except Exception:
            continue
    return parsed


def log_indicates_resource_failure(log_path: Path) -> bool:
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    markers = (
        "memoryerror",
        "cannot allocate memory",
        "could not allocate dynamic translator buffer",
        "dynamic translator buffer",
        "std::bad_alloc",
        "resource temporarily unavailable",
        "killed",
    )
    return any(marker in tail for marker in markers)


def manifest_case_info(output_root: Path, rel_path: str) -> Dict[str, Any]:
    manifest = load_json(output_root / "campaign_manifest.json")
    for item in manifest.get("cases") or []:
        if isinstance(item, dict) and str(item.get("rel_path") or "") == rel_path:
            return item
    return {}


def load_valid_bbs_for_case(output_root: Path, rel_path: str) -> set[int]:
    info = manifest_case_info(output_root, rel_path)
    valid_path = Path(str(info.get("valid_basic_blocks") or ""))
    valid_bbs: set[int] = set()
    try:
        with valid_path.open() as f:
            for line in f:
                text = line.strip()
                if text:
                    valid_bbs.add(int(text, 16) & ~1)
    except Exception:
        return set()
    return valid_bbs


def best_progress(progress_file: Path) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for record in load_jsonl(progress_file):
        if progress_score(record) > progress_score(best):
            best = record
    return best


def best_progress_for_case(output_root: Path, rel_path: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    case_dir = output_root / rel_path
    for progress_file in sorted(case_dir.glob("*_coverage_progress*.jsonl")):
        record = best_progress(progress_file)
        if progress_score(record) > progress_score(best):
            best = record
    return best


def artifact_union_counts(output_root: Path, rel_path: str) -> Optional[Dict[str, Any]]:
    case_dir = output_root / rel_path
    if not case_dir.exists():
        return None
    covered: set[int] = set()
    files = 0
    latest_record: Dict[str, Any] = {}
    for artifact_path in sorted(
        list(case_dir.glob("*_coverage_checkpoints*.jsonl"))
        + list(case_dir.glob("*_coverage_progress*.jsonl"))
    ):
        files += 1
        for record in load_jsonl(artifact_path):
            record_covered = parse_int_set(record.get("global_covered_bb_list"))
            record_covered.update(parse_int_set(record.get("covered_bb_list")))
            record_covered.update(parse_int_set(record.get("covered_valid_bb_list")))
            if not record_covered:
                continue
            covered.update(record_covered)
            if progress_score(record) >= progress_score(latest_record):
                latest_record = record
    if not covered:
        return None
    valid_bbs = load_valid_bbs_for_case(output_root, rel_path)
    valid_total = len(valid_bbs) or intval(latest_record.get("valid_total_bbs"))
    valid_covered = covered & valid_bbs if valid_bbs else covered
    return {
        "files": files,
        "covered_bbs": len(covered),
        "valid_covered_bbs": len(valid_covered),
        "valid_total_bbs": valid_total,
        "valid_coverage_rate": (len(valid_covered) / valid_total * 100.0) if valid_total else 0.0,
        "elapsed_seconds": latest_record.get("elapsed_seconds"),
        "stage": latest_record.get("stage") or latest_record.get("status") or "artifact_union",
    }


def normalize_result_from_artifacts(output_root: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    rel_path = str(item.get("rel_path") or "")
    if not rel_path:
        return item
    normalized = dict(item)
    best = best_progress_for_case(output_root, rel_path) or {}
    union_counts = artifact_union_counts(output_root, rel_path) or {}
    report_path = Path(str(normalized.get("report_file") or ""))
    report = load_json(report_path) if report_path.exists() else {}

    candidate_valid = [
        intval(normalized.get("valid_covered_bbs")),
        intval(best.get("valid_covered_bbs")),
        intval(union_counts.get("valid_covered_bbs")),
        intval(report.get("valid_covered_bbs")),
    ]
    final_valid = max(candidate_valid)
    valid_total = (
        intval(normalized.get("valid_total_bbs"))
        or intval(best.get("valid_total_bbs"))
        or intval(union_counts.get("valid_total_bbs"))
        or intval(report.get("valid_total_bbs"))
    )
    final_covered = max(
        intval(normalized.get("covered_bbs")),
        intval(best.get("covered_bbs")),
        intval(union_counts.get("covered_bbs")),
        intval(report.get("covered_bbs")),
    )
    total_bbs = (
        intval(normalized.get("total_bbs"))
        or intval(best.get("total_bbs"))
        or intval(report.get("total_bbs"))
        or final_covered
    )
    if final_valid > intval(normalized.get("valid_covered_bbs")):
        normalized["valid_covered_bbs"] = final_valid
        normalized["valid_total_bbs"] = valid_total
        normalized["valid_coverage_rate"] = (final_valid / valid_total * 100.0) if valid_total else 0.0
    if final_covered > intval(normalized.get("covered_bbs")):
        normalized["covered_bbs"] = final_covered
        normalized["total_bbs"] = total_bbs
        normalized["coverage_rate"] = (final_covered / total_bbs * 100.0) if total_bbs else 0.0

    copy_missing_static_fields(normalized, report, best)
    merge_static_high_water(normalized, best, report)
    refresh_static_rates(normalized)
    ensure_extended_static_fallback(normalized)
    refresh_static_rates(normalized)
    if best:
        normalized.setdefault("progress_high_water_valid_covered_bbs", best.get("valid_covered_bbs"))
        normalized.setdefault("progress_high_water_covered_bbs", best.get("covered_bbs"))
    if union_counts:
        list_valid = intval(union_counts.get("valid_covered_bbs"))
        normalized["coverage_list_recoverable_valid_bbs"] = list_valid
        normalized["coverage_list_complete"] = list_valid >= final_valid
        if list_valid < final_valid:
            normalized["report_file_caveat"] = (
                "Final strict count is recovered from progress high-water counters; "
                "full BB address list is only partially recoverable from artifacts."
            )
    if normalized.get("log_file"):
        normalized["resource_failure"] = bool(normalized.get("resource_failure")) or log_indicates_resource_failure(
            Path(str(normalized.get("log_file")))
        )
    if report_path.exists() and intval(report.get("valid_covered_bbs")) < final_valid:
        normalized["report_file_caveat"] = (
            normalized.get("report_file_caveat")
            or "Report file has lower coverage than normalized strict progress high-water."
        )
    return normalized


def normalize_results_from_artifacts(output_root: Path, results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_result_from_artifacts(output_root, item) for item in results if isinstance(item, dict)]


def latest_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    latest: Optional[Dict[str, Any]] = None
    try:
        with path.open() as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    latest = payload
    except Exception:
        return latest
    return latest


def pct(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def intval(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def index_results(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in summary.get("results") or []:
        if isinstance(item, dict):
            rel = str(item.get("rel_path") or "")
            if rel:
                result[rel] = item
    return result


def collect_live_progress(output_root: Path) -> List[Dict[str, Any]]:
    best_by_case: Dict[str, Tuple[Path, Dict[str, Any], float]] = {}
    for progress_path in sorted(output_root.glob("**/*_coverage_progress*.jsonl")):
        try:
            rel_path = str(progress_path.parent.relative_to(output_root))
        except Exception:
            rel_path = str(progress_path.parent)
        best_record = best_progress(progress_path)
        if not best_record:
            continue
        try:
            mtime = progress_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        previous = best_by_case.get(rel_path)
        previous_score = progress_score(previous[1]) if previous else (0, 0.0, 0.0)
        score = progress_score(best_record)
        if previous is None or score > previous_score or (score == previous_score and mtime >= previous[2]):
            best_by_case[rel_path] = (progress_path, best_record, mtime)

    rows: List[Dict[str, Any]] = []
    for rel_path, (progress_path, latest, _mtime) in sorted(best_by_case.items()):
        row = {
            "case": rel_path,
            "progress_file": str(progress_path),
            "elapsed_seconds": latest.get("elapsed_seconds"),
            "event": latest.get("event"),
            "stage": latest.get("stage"),
            "last_phase": latest.get("last_phase"),
            "valid_covered_bbs": latest.get("valid_covered_bbs"),
            "valid_total_bbs": latest.get("valid_total_bbs"),
            "valid_coverage_rate": latest.get("valid_coverage_rate"),
            "valid_entry_plus_vector_static_reachable_bbs": latest.get("valid_entry_plus_vector_static_reachable_bbs"),
            "valid_entry_plus_vector_static_reachable_covered_bbs": latest.get("valid_entry_plus_vector_static_reachable_covered_bbs"),
            "valid_entry_plus_vector_reachable_coverage_rate": latest.get("valid_entry_plus_vector_reachable_coverage_rate"),
            "strict_real_entry_replayable": latest.get("strict_real_entry_replayable"),
        }
        copy_missing_static_fields(row, latest)
        refresh_static_rates(row)
        ensure_extended_static_fallback(row)
        refresh_static_rates(row)
        rows.append(row)
    return rows


def checkpoint_union_counts(output_root: Path, rel_path: str) -> Optional[Dict[str, Any]]:
    return artifact_union_counts(output_root, rel_path)


def classify_low_coverage(item: Dict[str, Any]) -> Tuple[str, str]:
    status = str(item.get("status") or "")
    valid_rate = pct(item.get("valid_coverage_rate"))
    reachable_rate = pct(item.get("valid_entry_plus_vector_reachable_coverage_rate"))
    extended_total = intval(item.get("valid_extended_static_reachable_bbs"))
    extended_rate = pct(item.get("valid_extended_static_reachable_coverage_rate"))
    entry_uncovered = intval(item.get("entry_reachable_uncovered_bbs"))
    vector_uncovered = intval(item.get("vector_only_uncovered_bbs"))
    static_unreachable = intval(item.get("valid_static_unreachable_from_entry_or_vectors"))
    extended_static_unreachable = intval(item.get("valid_static_unreachable_from_extended_roots"))
    static_ratio = pct(item.get("static_unreachable_valid_ratio"))
    extended_static_ratio = pct(item.get("extended_static_unreachable_valid_ratio"))
    extended_uncovered = intval(item.get("valid_extended_static_reachable_uncovered_bbs"))
    pred_gaps = intval(item.get("uncovered_with_covered_predecessor_bbs"))
    frontier_tasks = intval(item.get("frontier_successor_tasks"))
    frontier_targets = intval(item.get("frontier_successor_targets"))
    strict = bool(item.get("strict_real_entry_replayable"))
    extended_available = item.get("extended_static_roots_available") is True

    if status not in COMPLETED_STATUSES:
        return "tooling_or_runtime_failure", "case did not complete with a strict report"
    if not strict:
        return "strictness_failure", "report is not marked strict-real-entry-replayable"
    if extended_available and extended_total and extended_rate < 80.0 and extended_uncovered >= 100:
        return (
            "extended-root dynamic state/path gap",
            (
                f"{extended_uncovered} extended-root reachable valid BB remain uncovered; "
                "likely state prefix, IRQ/timer phase, dynamic dispatch, or persistent-state precondition"
            ),
        )
    if (
        extended_available
        and extended_total
        and static_ratio >= 20.0
        and extended_static_ratio < static_ratio * 0.75
    ):
        return (
            "entry/vector static-root under-modeling",
            (
                f"entry+vector static-unreachable is {static_unreachable}, but extended-root "
                f"static-unreachable drops to {extended_static_unreachable}; inspect task/callback/protocol roots"
            ),
        )
    if extended_available and extended_total and valid_rate < 80.0 and extended_rate >= 90.0:
        return (
            "valid-denominator/extended-static-unreachable dominated",
            (
                f"extended-root reachable-valid is {extended_rate:.2f}% but "
                f"{extended_static_unreachable} valid BB remain unreachable from modeled roots"
            ),
        )
    if valid_rate < 80.0 and reachable_rate >= 90.0:
        return (
            "valid-denominator/static-unreachable dominated",
            f"reachable-valid is {reachable_rate:.2f}% but {static_unreachable} valid BB are static-unreachable from entry/vectors",
        )
    if entry_uncovered >= 1000:
        return (
            "large entry-reachable state/path gap",
            f"{entry_uncovered} entry-reachable valid BB remain uncovered; likely software state, task scheduling, IRQ phase, or long parser state",
        )
    if vector_uncovered >= 100:
        return (
            "vector/contextual-ISR scheduling gap",
            f"{vector_uncovered} vector-only valid BB remain uncovered",
        )
    if frontier_tasks >= 100 and frontier_targets * 3 < frontier_tasks:
        return (
            "dynamic dispatch/call-continuation replay gap",
            f"frontier replay discovered {frontier_targets}/{frontier_tasks} targets",
        )
    if pred_gaps >= 25:
        return (
            "covered-predecessor branch frontier gap",
            f"{pred_gaps} uncovered BB have covered predecessors",
        )
    return "mixed or low-priority residual gap", "no single dominant diagnostic counter"


def format_elapsed(seconds: Any) -> str:
    value = int(pct(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_bucket_label(index: int, interval_seconds: int, *, width: int = 2) -> str:
    elapsed_seconds = int(index * max(1, int(interval_seconds)))
    if elapsed_seconds % 60 == 0:
        return f"{elapsed_seconds // 60:0{width}d}m"
    minutes, secs = divmod(elapsed_seconds, 60)
    return f"{minutes:0{width}d}m{secs:02d}s"


def extended_roots_label(item: Dict[str, Any]) -> str:
    if item.get("extended_static_roots_available") is False:
        return str(item.get("extended_static_root_fallback") or "fallback")
    if item.get("extended_static_roots_available") is True or intval(item.get("extended_static_root_bbs")):
        return "yes"
    if intval(item.get("valid_extended_static_reachable_bbs")):
        return "unknown"
    return "no"


def load_manifest_cases(output_root: Path) -> List[str]:
    manifest = load_json(output_root / "campaign_manifest.json")
    cases: List[str] = []
    for item in manifest.get("cases") or []:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("rel_path") or "")
        if rel_path:
            cases.append(rel_path)
    return cases


def driver_state(output_root: Path, stale_after_seconds: int = 900) -> Dict[str, Any]:
    heartbeat = load_json(output_root / "campaign_driver_heartbeat.json")
    if not heartbeat:
        return {
            "state": "unknown",
            "message": "driver heartbeat is missing",
        }
    timestamp = pct(heartbeat.get("timestamp"))
    age = max(0.0, time.time() - timestamp) if timestamp > 0 else 0.0
    pid = intval(heartbeat.get("pid"))
    heartbeat_current = age <= stale_after_seconds
    alive = False
    if pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    if heartbeat_current:
        state = "alive"
    elif alive:
        state = "stale"
    else:
        state = "dead"
    return {
        "state": state,
        "pid": pid,
        "age_seconds": age,
        "event": heartbeat.get("event"),
        "running_cases": heartbeat.get("running_cases") or [],
        "message": (
            "driver heartbeat is current"
            if state == "alive"
            else "driver heartbeat is stale or process is not alive"
        ),
        "pid_alive": alive,
        "heartbeat_current": heartbeat_current,
    }


def generate_live_markdown(output_root: Path, payload: Dict[str, Any]) -> Path:
    summary = load_json(output_root / "campaign_summary.json")
    normalized_summary = dict(summary)
    normalized_summary["results"] = normalize_results_from_artifacts(
        output_root,
        summary.get("results") or [],
    )
    results_by_case = index_results(normalized_summary)
    live_rows = collect_live_progress(output_root)
    live_by_case = {str(row.get("case") or ""): row for row in live_rows if row.get("case")}
    manifest_cases = load_manifest_cases(output_root)
    cases = manifest_cases or sorted(set(results_by_case) | set(live_by_case))
    driver = driver_state(output_root)

    duration_minutes = intval(summary.get("firmware_minutes")) or 240
    interval_seconds = intval(summary.get("progress_interval_seconds")) or 300
    total_buckets = max(0, int(duration_minutes * 60 // max(1, interval_seconds)))
    max_elapsed = 0.0
    for row in live_by_case.values():
        max_elapsed = max(max_elapsed, pct(row.get("elapsed_seconds")))
    for item in results_by_case.values():
        max_elapsed = max(max_elapsed, pct(item.get("execution_time_seconds")))
    observed_buckets = min(total_buckets, max(0, int(max_elapsed // max(1, interval_seconds)) + 1))

    running_diagnostics: Dict[str, Dict[str, Any]] = {}
    current_rows: List[List[Any]] = []
    for rel_path in cases:
        result = results_by_case.get(rel_path)
        live = live_by_case.get(rel_path)
        if result:
            status = str(result.get("status") or "finished")
            valid_covered = intval(result.get("valid_covered_bbs"))
            valid_total = intval(result.get("valid_total_bbs"))
            valid_rate = pct(result.get("valid_coverage_rate"))
            reachable_covered = intval(result.get("valid_entry_plus_vector_static_reachable_covered_bbs"))
            reachable_total = intval(result.get("valid_entry_plus_vector_static_reachable_bbs"))
            reachable_rate = pct(result.get("valid_entry_plus_vector_reachable_coverage_rate"))
            extended_covered = intval(result.get("valid_extended_static_reachable_covered_bbs"))
            extended_total = intval(result.get("valid_extended_static_reachable_bbs"))
            extended_rate = pct(result.get("valid_extended_static_reachable_coverage_rate"))
            elapsed = result.get("execution_time_seconds")
            stage = "final"
            strict = bool(result.get("strict_real_entry_replayable"))
        elif live:
            status = "running"
            valid_covered = intval(live.get("valid_covered_bbs"))
            valid_total = intval(live.get("valid_total_bbs"))
            valid_rate = pct(live.get("valid_coverage_rate"))
            reachable_covered = intval(live.get("valid_entry_plus_vector_static_reachable_covered_bbs"))
            reachable_total = intval(live.get("valid_entry_plus_vector_static_reachable_bbs"))
            reachable_rate = pct(live.get("valid_entry_plus_vector_reachable_coverage_rate"))
            extended_covered = intval(live.get("valid_extended_static_reachable_covered_bbs"))
            extended_total = intval(live.get("valid_extended_static_reachable_bbs"))
            extended_rate = pct(live.get("valid_extended_static_reachable_coverage_rate"))
            elapsed = live.get("elapsed_seconds")
            stage = str(live.get("stage") or live.get("last_phase") or "")
            strict = bool(live.get("strict_real_entry_replayable"))
            union_counts = checkpoint_union_counts(output_root, rel_path)
            if union_counts and intval(union_counts.get("valid_covered_bbs")) > valid_covered:
                valid_covered = intval(union_counts.get("valid_covered_bbs"))
                valid_total = intval(union_counts.get("valid_total_bbs"))
                valid_rate = pct(union_counts.get("valid_coverage_rate"))
                stage = f"{stage} + checkpoint_union"
                strict = True
            progress_file = Path(str(live.get("progress_file") or ""))
            stage_diag = running_stage_summary(progress_file) if progress_file.exists() else {}
            checkpoint_diag = running_checkpoint_summary(progress_file) if progress_file.exists() else {}
            running_diagnostics[rel_path] = {
                "stage": stage_diag,
                "checkpoint": checkpoint_diag,
                "classification": classify_running_gap(stage_diag, checkpoint_diag),
            }
        else:
            status = "pending"
            valid_covered = valid_total = 0
            valid_rate = 0.0
            reachable_covered = reachable_total = 0
            reachable_rate = 0.0
            extended_covered = extended_total = 0
            extended_rate = 0.0
            elapsed = 0
            stage = ""
            strict = False
        current_rows.append([
            rel_path,
            status,
            format_elapsed(elapsed),
            f"{valid_covered}/{valid_total}",
            f"{valid_rate:.2f}%",
            f"{reachable_covered}/{reachable_total}",
            f"{reachable_rate:.2f}%",
            f"{extended_covered}/{extended_total}",
            f"{extended_rate:.2f}%",
            extended_roots_label(result or live or {}),
            stage,
            "yes" if strict else "no",
        ])

    trend_headers = [
        format_bucket_label(bucket, interval_seconds, width=3)
        for bucket in range(observed_buckets + 1)
    ]
    trend_rows: List[List[Any]] = []
    for rel_path in cases:
        result = results_by_case.get(rel_path)
        progress_file = Path(str(result.get("progress_jsonl_file") or "")) if result else None
        if not progress_file or not progress_file.exists():
            live = live_by_case.get(rel_path) or {}
            progress_file = Path(str(live.get("progress_file") or ""))
        merged_cells = merged_case_trend_cells(output_root, rel_path, interval_seconds, duration_minutes)
        if merged_cells:
            cells = merged_cells[: observed_buckets + 1]
        elif progress_file.exists():
            cells = trend_cells(progress_file, interval_seconds, duration_minutes)[: observed_buckets + 1]
        else:
            cells = ["-"] * (observed_buckets + 1)
        trend_rows.append([rel_path] + cells)

    lines: List[str] = []
    lines.append("# Strict Campaign Live Progress")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z', time.localtime())}")
    lines.append(f"- Completed: {'yes' if payload.get('completed') else 'no'}")
    lines.append(f"- Current case: {payload.get('current_case') or 'none'}")
    lines.append(f"- Cases: finished {intval(payload.get('cases_finished'))}/{intval(payload.get('cases_total'))}, completed {intval(payload.get('cases_completed'))}, failed {intval(payload.get('cases_failed'))}")
    lines.append(f"- Driver: {driver.get('state')} pid={driver.get('pid', 0)} heartbeat_age={int(pct(driver.get('age_seconds')))}s event={driver.get('event') or ''}")
    lines.append("")
    lines.append("## Current Coverage")
    lines.extend(markdown_table(
        [
            "Case",
            "Status",
            "Elapsed",
            "Valid BB",
            "Valid rate",
            "Reachable-valid BB",
            "Reachable-valid rate",
            "Extended-valid BB",
            "Extended-valid rate",
            "Ext roots",
            "Stage",
            "Strict",
        ],
        current_rows,
    ))
    lines.append("")
    lines.append("## 5-Minute Trend")
    lines.extend(markdown_table(["Case"] + trend_headers, trend_rows))
    lines.append("")
    if running_diagnostics:
        diag_rows: List[List[Any]] = []
        for rel_path, diag in sorted(running_diagnostics.items()):
            stage = diag.get("stage") or {}
            checkpoint = diag.get("checkpoint") or {}
            diag_rows.append([
                rel_path,
                stage.get("latest_stage") or "",
                f"+{intval(stage.get('growth_5m'))}/+{intval(stage.get('growth_15m'))}",
                f"{stage.get('best_stage') or ''} (+{intval(stage.get('best_delta'))})",
                f"{intval(checkpoint.get('tasks_run'))}/{intval(checkpoint.get('queued_tasks'))}/{intval(checkpoint.get('pending_paths'))}",
                f"{intval(checkpoint.get('pc_constraint_hits'))}/{intval(checkpoint.get('branch_mmio_constraint_hits'))}",
                f"{intval(checkpoint.get('state_signature_new'))}/{intval(checkpoint.get('state_signature_repeated'))}",
                diag.get("classification") or "",
            ])
        lines.append("## Running Diagnostics")
        lines.extend(markdown_table(
            [
                "Case",
                "Latest stage",
                "Growth 5m/15m",
                "Best stage",
                "Tasks/queued/pending",
                "PC/MMIO hits",
                "New/repeated states",
                "Running classification",
            ],
            diag_rows,
        ))
        lines.append("")
    lines.append("## Notes")
    lines.append("- Counted BBs are only valid when `Strict` is `yes`; the campaign runner records strict entry-derived execution/replay progress.")
    lines.append("- This live file is generated from existing JSONL artifacts and does not change emulator execution or coverage accounting.")

    live_path = output_root / "LIVE_PROGRESS.md"
    live_path.write_text("\n".join(lines) + "\n")
    return live_path


def trend_cells(progress_file: Path, interval_seconds: int, duration_minutes: int) -> List[str]:
    records = sorted(
        load_jsonl(progress_file),
        key=lambda rec: pct(rec.get("elapsed_seconds")),
    )
    cells: List[str] = []
    cursor = 0
    last: Optional[Dict[str, Any]] = None
    total_buckets = max(0, int(duration_minutes * 60 // max(1, interval_seconds)))
    for bucket in range(total_buckets + 1):
        threshold = bucket * max(1, interval_seconds)
        while cursor < len(records) and pct(records[cursor].get("elapsed_seconds")) <= threshold + 1e-6:
            last = records[cursor]
            cursor += 1
        if bucket == 0 and last is None and records:
            last = records[0]
        value = None if last is None else last.get("valid_covered_bbs")
        cells.append("-" if value is None else str(intval(value)))
    return cells


def merged_case_trend_cells(output_root: Path, rel_path: str, interval_seconds: int, duration_minutes: int) -> List[str]:
    case_dir = output_root / rel_path
    progress_files = sorted(case_dir.glob("*_coverage_progress*.jsonl"))
    if not progress_files:
        return []
    records: List[Dict[str, Any]] = []
    for progress_file in progress_files:
        records.extend(load_jsonl(progress_file))
    if not records:
        return []
    records = sorted(records, key=lambda rec: pct(rec.get("elapsed_seconds")))
    cells: List[str] = []
    cursor = 0
    best_value: Optional[int] = None
    total_buckets = max(0, int(duration_minutes * 60 // max(1, interval_seconds)))
    for bucket in range(total_buckets + 1):
        threshold = bucket * max(1, interval_seconds)
        while cursor < len(records) and pct(records[cursor].get("elapsed_seconds")) <= threshold + 1e-6:
            value = records[cursor].get("valid_covered_bbs")
            if value is not None:
                parsed = intval(value)
                best_value = parsed if best_value is None else max(best_value, parsed)
            cursor += 1
        if bucket == 0 and best_value is None:
            for record in records:
                value = record.get("valid_covered_bbs")
                if value is not None:
                    best_value = intval(value)
                    break
        cells.append("-" if best_value is None else str(best_value))
    return cells


def progress_file_to_checkpoint_file(progress_file: Path) -> Path:
    name = progress_file.name.replace("_coverage_progress", "_coverage_checkpoints")
    return progress_file.with_name(name)


def value_at_or_before(records: List[Dict[str, Any]], threshold: float) -> Optional[int]:
    last_value: Optional[int] = None
    for record in records:
        elapsed = pct(record.get("elapsed_seconds"))
        if elapsed > threshold + 1e-6:
            break
        value = record.get("valid_covered_bbs")
        if value is not None:
            last_value = intval(value)
    return last_value


def running_stage_summary(progress_file: Path) -> Dict[str, Any]:
    records = sorted(load_jsonl(progress_file), key=lambda rec: pct(rec.get("elapsed_seconds")))
    latest = records[-1] if records else {}
    if not records:
        return {}
    latest_valid = intval(latest.get("valid_covered_bbs"))
    latest_elapsed = pct(latest.get("elapsed_seconds"))
    prev_valid: Optional[int] = None
    best_stage = ""
    best_delta = 0
    last_positive_stage = ""
    last_positive_delta = 0
    for record in records:
        value = record.get("valid_covered_bbs")
        if value is None:
            continue
        valid = intval(value)
        delta = 0 if prev_valid is None else valid - prev_valid
        stage = str(record.get("stage") or record.get("last_phase") or record.get("event") or "")
        if delta > best_delta:
            best_delta = delta
            best_stage = stage
        if delta > 0:
            last_positive_stage = stage
            last_positive_delta = delta
        prev_valid = valid
    value_5m = value_at_or_before(records, max(0.0, latest_elapsed - 300.0))
    value_15m = value_at_or_before(records, max(0.0, latest_elapsed - 900.0))
    growth_5m = latest_valid - value_5m if value_5m is not None else latest_valid
    growth_15m = latest_valid - value_15m if value_15m is not None else latest_valid
    return {
        "latest_valid": latest_valid,
        "latest_elapsed": latest_elapsed,
        "latest_stage": latest.get("stage") or latest.get("last_phase") or "",
        "best_stage": best_stage,
        "best_delta": best_delta,
        "last_positive_stage": last_positive_stage,
        "last_positive_delta": last_positive_delta,
        "growth_5m": growth_5m,
        "growth_15m": growth_15m,
    }


def running_checkpoint_summary(progress_file: Path) -> Dict[str, Any]:
    checkpoint_file = progress_file_to_checkpoint_file(progress_file)
    checkpoint = latest_jsonl(checkpoint_file)
    if not checkpoint:
        return {}
    keys = [
        "global_covered_bbs",
        "phase_covered_bbs",
        "tasks_run",
        "queued_tasks",
        "pending_paths",
        "deferred_tasks",
        "stale_deferred_tasks",
        "edge_saturated_edges",
        "constraint_hits",
        "pc_constraint_hits",
        "branch_mmio_constraint_hits",
        "unique_pc_constraint_sites",
        "unique_branch_mmio_pc_constraint_sites",
        "scoped_probe_tasks",
        "state_signature_new",
        "state_signature_repeated",
        "last_completed_stop_reason",
        "last_completed_new_bbs",
    ]
    return {key: checkpoint.get(key) for key in keys if key in checkpoint}


def classify_running_gap(stage: Dict[str, Any], checkpoint: Dict[str, Any]) -> str:
    growth_5m = intval(stage.get("growth_5m"))
    growth_15m = intval(stage.get("growth_15m"))
    best_stage = str(stage.get("best_stage") or "")
    latest_stage = str(stage.get("latest_stage") or "")
    branch_mmio_hits = intval(checkpoint.get("branch_mmio_constraint_hits"))
    pc_hits = intval(checkpoint.get("pc_constraint_hits"))
    queued = intval(checkpoint.get("queued_tasks"))
    repeated = intval(checkpoint.get("state_signature_repeated"))
    new_states = intval(checkpoint.get("state_signature_new"))

    if branch_mmio_hits > 0:
        return "direct MMIO branch constraints still appear in checkpoints"
    if growth_15m == 0 and queued > 0:
        return "frontier tasks remain queued but are currently plateaued; likely state/context or replay scheduling"
    if growth_5m <= 5 and ("direct_call" in latest_stage or "summary_return" in latest_stage):
        return "direct-call/summary-return stage has low marginal yield"
    if growth_5m <= 5 and "rtos_thread" in latest_stage:
        return "RTOS/thread-entry replay has low marginal yield under current entry-derived states"
    if "frontier_successor_replay" in best_stage and growth_15m > 0:
        return "successor replay is the dominant productive phase; keep draining entry-derived frontier variants"
    if pc_hits > 0 and branch_mmio_hits == 0:
        return "path constraints are mostly non-MMIO PC/state constraints"
    if repeated > max(1, new_states) * 20:
        return "state signatures are heavily repeated; missing paths likely need new scheduling/context, not more identical replay"
    return "mixed running gap; wait for final strict report diagnostics"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> List[str]:
    header_list = list(headers)
    lines = ["| " + " | ".join(header_list) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header_list)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def generate_report(output_root: Path, baseline_summary_path: Optional[Path]) -> Path:
    summary = load_json(output_root / "campaign_summary.json")
    baseline = load_json(baseline_summary_path) if baseline_summary_path else {}
    baseline_results = index_results(baseline)
    results = normalize_results_from_artifacts(output_root, summary.get("results") or [])
    duration_minutes = intval(summary.get("firmware_minutes")) or 240
    interval_seconds = intval(summary.get("progress_interval_seconds")) or 300

    completed = [row for row in results if row.get("status") in COMPLETED_STATUSES]
    failed = [row for row in results if row.get("status") == "failed"]
    strict_completed = [row for row in completed if row.get("strict_real_entry_replayable")]
    low_rows = [
        row for row in completed
        if pct(row.get("valid_coverage_rate")) < 80.0
    ]

    lines: List[str] = []
    lines.append("# Strict Campaign Overall Analysis")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Baseline summary: `{baseline_summary_path}`" if baseline_summary_path else "- Baseline summary: none")
    lines.append(f"- Completed: {'yes' if summary.get('completed') else 'no'}")
    lines.append(f"- Cases total: {intval(summary.get('cases_total'))}")
    lines.append(f"- Cases completed: {len(completed)}")
    lines.append(f"- Cases failed: {len(failed)}")
    lines.append(f"- Strict completed: {len(strict_completed)}/{len(completed)}")
    lines.append("- Strict coverage rule: counted BBs come from reset-entry execution or entry-derived replay/snapshot states; no artificial jump-to-BB coverage is counted.")
    lines.append("")

    rows = []
    for item in sorted(results, key=lambda row: str(row.get("rel_path") or "")):
        rel = str(item.get("rel_path") or "")
        base = baseline_results.get(rel, {})
        delta_bb = intval(item.get("valid_covered_bbs")) - intval(base.get("valid_covered_bbs"))
        delta_rate = pct(item.get("valid_coverage_rate")) - pct(base.get("valid_coverage_rate"))
        reachable = (
            f"{intval(item.get('valid_entry_plus_vector_static_reachable_covered_bbs'))}/"
            f"{intval(item.get('valid_entry_plus_vector_static_reachable_bbs'))}"
        )
        extended = (
            f"{intval(item.get('valid_extended_static_reachable_covered_bbs'))}/"
            f"{intval(item.get('valid_extended_static_reachable_bbs'))}"
        )
        rows.append([
            rel,
            item.get("status") or "",
            f"{intval(item.get('valid_covered_bbs'))}/{intval(item.get('valid_total_bbs'))}",
            f"{pct(item.get('valid_coverage_rate')):.2f}%",
            reachable,
            f"{pct(item.get('valid_entry_plus_vector_reachable_coverage_rate')):.2f}%",
            extended,
            f"{pct(item.get('valid_extended_static_reachable_coverage_rate')):.2f}%",
            extended_roots_label(item),
            intval(item.get("valid_static_unreachable_from_entry_or_vectors")),
            intval(item.get("valid_static_unreachable_from_extended_roots")),
            "yes" if item.get("strict_real_entry_replayable") else "no",
            "yes" if item.get("resource_failure") else "no",
            (
                "yes"
                if item.get("coverage_list_complete") is not False
                else f"partial:{intval(item.get('coverage_list_recoverable_valid_bbs'))}"
            ),
            f"{delta_bb:+d}",
            f"{delta_rate:+.2f}pp",
        ])
    lines.append("## Firmware Results")
    lines.extend(markdown_table(
        [
            "Case",
            "Status",
            "Valid BB",
            "Valid rate",
            "Reachable-valid BB",
            "Reachable-valid rate",
            "Extended-valid BB",
            "Extended-valid rate",
            "Ext roots",
            "Static-unreach",
            "Extended static-unreach",
            "Strict",
            "Resource",
            "List",
            "Delta BB",
            "Delta rate",
        ],
        rows,
    ))
    lines.append("")

    low_diag_rows = []
    for item in sorted(low_rows, key=lambda row: pct(row.get("valid_coverage_rate"))):
        category, reason = classify_low_coverage(item)
        low_diag_rows.append([
            item.get("rel_path") or "",
            f"{pct(item.get('valid_coverage_rate')):.2f}%",
            f"{pct(item.get('valid_entry_plus_vector_reachable_coverage_rate')):.2f}%",
            f"{pct(item.get('valid_extended_static_reachable_coverage_rate')):.2f}%",
            extended_roots_label(item),
            intval(item.get("valid_static_unreachable_from_entry_or_vectors")),
            intval(item.get("valid_static_unreachable_from_extended_roots")),
            category,
            reason,
        ])
    lines.append("## Low Coverage Diagnostics")
    lines.extend(markdown_table(
        [
            "Case",
            "Valid rate",
            "Reachable-valid rate",
            "Extended-valid rate",
            "Ext roots",
            "Static-unreach",
            "Extended static-unreach",
            "Classification",
            "Evidence",
        ],
        low_diag_rows,
    ))
    lines.append("")

    trend_rows = []
    for item in sorted(results, key=lambda row: str(row.get("rel_path") or "")):
        progress_file = Path(str(item.get("progress_jsonl_file") or ""))
        rel = str(item.get("rel_path") or "")
        cells = merged_case_trend_cells(output_root, rel, interval_seconds, duration_minutes)
        if not cells and progress_file.exists():
            cells = trend_cells(progress_file, interval_seconds, duration_minutes)
        trend_rows.append([item.get("rel_path") or ""] + cells)
    max_cells = max((len(row) - 1 for row in trend_rows), default=0)
    trend_headers = [
        format_bucket_label(index, interval_seconds)
        for index in range(max_cells)
    ]
    lines.append("## 5-Minute Valid BB Trend")
    lines.extend(markdown_table(["Case"] + trend_headers, trend_rows))
    lines.append("")

    lines.append("## Notes")
    lines.append("- A low final valid rate with high reachable-valid rate usually means the valid-BB denominator includes code not statically reachable from reset entry or vector handlers under the current model.")
    lines.append("- `Extended-valid` adds conservative semantic roots such as observed callbacks, task/protocol roots, and dynamic successor evidence for attribution only; it does not add any BB to real Unicorn-executed coverage.")
    lines.append("- `Ext roots=entry_plus_vector` means the campaign artifacts predate extended-root metadata and the report is showing a compatibility fallback, not a new extended-root analysis result.")
    lines.append("- A low reachable-valid rate with many entry-reachable uncovered BBs is a real execution gap, usually caused by software state, interrupt phase, dynamic dispatch, persistent state, or long protocol/task state.")
    lines.append("- MMIO-only failures should show up as polling-loop or branch-MMIO constraints; the current dominant residual pattern is expected to be dynamic dispatch/call continuation and state scheduling rather than simple MMIO comparison values.")
    lines.append("- `List=partial:N` means strict entry-derived counters reached the displayed BB count, but only N valid BB addresses were recoverable from checkpoint/progress artifacts; do not use the partial list as the full coverage set.")

    report_path = output_root / "campaign_overall_analysis.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def monitor(output_root: Path, baseline_summary_path: Optional[Path], interval_seconds: int, until_complete: bool) -> None:
    monitor_path = output_root / "monitor_status_5min.jsonl"
    while True:
        summary = load_json(output_root / "campaign_summary.json")
        live = load_json(output_root / "campaign_live_status.json")
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "completed": bool(summary.get("completed")),
            "cases_total": live.get("cases_total", summary.get("cases_total")),
            "cases_finished": live.get("cases_finished", summary.get("cases_finished")),
            "cases_completed": live.get("cases_completed", summary.get("cases_completed")),
            "cases_failed": live.get("cases_failed", summary.get("cases_failed")),
            "current_case": live.get("current_case", summary.get("current_case")),
            "elapsed_seconds": live.get("elapsed_seconds", summary.get("elapsed_seconds")),
            "progress": collect_live_progress(output_root),
        }
        append_jsonl(monitor_path, payload)
        generate_live_markdown(output_root, payload)
        if payload["completed"]:
            report_path = generate_report(output_root, baseline_summary_path)
            append_jsonl(monitor_path, {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                "event": "final_report_generated",
                "report": str(report_path),
            })
            return
        if not until_complete:
            return
        time.sleep(max(1, int(interval_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-summary", default="")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--until-complete", action="store_true")
    parser.add_argument("--generate-report-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    baseline_summary_path = Path(args.baseline_summary).resolve() if args.baseline_summary else None
    if args.generate_report_only:
        print(generate_report(output_root, baseline_summary_path))
        return
    monitor(output_root, baseline_summary_path, args.interval_seconds, args.until_complete)


if __name__ == "__main__":
    main()
