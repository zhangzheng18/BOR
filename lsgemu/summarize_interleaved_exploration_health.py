#!/usr/bin/env python3
"""Summarize whether an interleaved lsgemu report actually ran exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


EXPLORATION_COMPONENTS = (
    "interleaved",
    "frontier_successor_replay_early",
    "early_switch_frontier_reservoir",
    "early_frontier_reservoir",
    "contextual_isr_reservoir",
    "semantic_frontier_flush",
    "semantic_frontier_drain",
    "direct_call_continuation",
    "late_direct_call_frontier_drain",
    "rtos_thread_entry",
    "direct_call_summary_return",
    "frontier_successor_replay",
    "frontier_successor_replay_tail",
    "contextual_isr_frontier_targeted",
    "deadline_drain",
    "switch_frontier_targeted",
    "frontier_targeted",
)


def load_json(path: Path) -> Dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def summarize(path: Path) -> Dict[str, object]:
    report = load_json(path)
    components = report.get("component_bbs_counted") or {}
    branch_catalog = report.get("branch_catalog_summary") or {}
    phase_metadata = report.get("phase_metadata") or {}
    exploration_sum = sum(int(components.get(name) or 0) for name in EXPLORATION_COMPONENTS)
    stream_sum = sum(
        int(value or 0)
        for name, value in components.items()
        if isinstance(name, str) and "stream_input" in name
    )
    skip_reasons: Dict[str, int] = {}
    for phase in phase_metadata.values():
        if not isinstance(phase, dict):
            continue
        reason = phase.get("skip") or phase.get("stop") or phase.get("status")
        if not reason:
            continue
        skip_reasons[str(reason)] = skip_reasons.get(str(reason), 0) + 1
    main_points = int(branch_catalog.get("main_path_branch_points") or 0)
    reservoir_points = int(branch_catalog.get("reservoir_new_branch_points") or 0)
    likely_stream_only = main_points > 0 and reservoir_points == 0 and exploration_sum == 0
    return {
        "path": str(path),
        "valid_covered_bbs": report.get("valid_covered_bbs"),
        "valid_total_bbs": report.get("valid_total_bbs"),
        "valid_coverage_rate": report.get("valid_coverage_rate"),
        "baseline_bbs": int(components.get("baseline") or 0),
        "stream_bbs": stream_sum,
        "exploration_bbs": exploration_sum,
        "main_path_branch_points": main_points,
        "reservoir_new_branch_points": reservoir_points,
        "likely_stream_only": likely_stream_only,
        "skip_reasons": skip_reasons,
    }


def iter_reports(paths: Iterable[str]) -> List[Path]:
    reports: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            reports.extend(sorted(path.rglob("*_interleaved_report.json")))
        elif path.exists():
            reports.append(path)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Report JSON files or directories")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    rows = [summarize(path) for path in iter_reports(args.paths)]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    print("| Report | Valid Coverage | Baseline BB | Stream BB | Exploration BB | Main Branch Points | Reservoir New Points | Diagnosis |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        diagnosis = "stream-only/budget-starved" if row["likely_stream_only"] else "exploration-active"
        print(
            f"| `{Path(str(row['path'])).name}` | "
            f"{float(row.get('valid_coverage_rate') or 0.0):.2f}% | "
            f"{row['baseline_bbs']} | {row['stream_bbs']} | {row['exploration_bbs']} | "
            f"{row['main_path_branch_points']} | {row['reservoir_new_branch_points']} | "
            f"`{diagnosis}` |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
