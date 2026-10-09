#!/usr/bin/env python3
"""Analyze an A/B watchdog run produced by ab_watchdog_stage_truncation_*.sh.

For each arm directory, reads the case's ``coverage_progress.jsonl`` and the
final report, and prints the round-3 acceptance comparison:

* final/max covered BBs and the elapsed time each watermark was first seen
  (arrival time of the +346-style stage-boundary merge);
* ``stall_watchdog_stage_truncated`` events with stage and stall window;
* tail-stage ``stage_end`` merge delta;
* terminated_early / stop reason (run-level) vs stage truncations.

Optionally overlays the 24h no-LLM control arm timeline for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CONTROL_PROGRESS = Path(
    ".lsgemu_runs/campaign_24h_ardupilot_pixhawk1_NOLLM_control_20260918/"
    "ardupilot/ardupilot_Pixhawk1_STM32F427/"
    "ardupilot_Pixhawk1_STM32F427_coverage_progress.jsonl"
)


def _iter_events(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                continue


def _find_progress(arm_dir: Path) -> Path | None:
    candidates = sorted(arm_dir.rglob("*_coverage_progress.jsonl"))
    return candidates[-1] if candidates else None


def _find_report(arm_dir: Path) -> Path | None:
    candidates = sorted(
        p for p in arm_dir.rglob("*.json")
        if "report" in p.name and "progress" not in p.name
    )
    return candidates[-1] if candidates else None


def summarize_arm(name: str, arm_dir: Path) -> dict | None:
    progress = _find_progress(arm_dir)
    if progress is None:
        print(f"[{name}] no coverage_progress.jsonl under {arm_dir}")
        return None
    covered_max = 0
    covered_first_seen: list[tuple[int, float]] = []
    truncation_events: list[dict] = []
    stage_ends: list[dict] = []
    for event in _iter_events(progress):
        elapsed = float(event.get("elapsed_seconds") or 0.0)
        covered = int(event.get("covered_bbs") or 0)
        if covered > covered_max:
            covered_max = covered
            # arrival of a merge: a new covered maximum
            covered_first_seen.append((covered, elapsed))
        if event.get("event") == "stall_watchdog_stage_truncated":
            truncation_events.append(event)
        if event.get("event") == "stage_end":
            stage_ends.append(event)
    print(f"[{name}] progress={progress}")
    print(f"[{name}] covered_max={covered_max}")
    for value, elapsed in covered_first_seen[-6:]:
        print(f"[{name}]   covered {value:5d} first seen at t={elapsed / 3600.0:.2f}h")
    for event in truncation_events:
        record = event.get("stall_watchdog_stage_truncation") or {}
        print(
            f"[{name}] STAGE TRUNCATED t={float(event.get('elapsed_seconds') or 0) / 3600.0:.2f}h "
            f"stage={record.get('stage')} stall={record.get('stage_stall_seconds')}s"
        )
    if not truncation_events:
        print(f"[{name}] stage truncations: none")
    for event in stage_ends:
        stage = str(event.get("stage") or "")
        if "frontier_successor_replay_tail" in stage:
            print(
                f"[{name}] tail stage_end t={float(event.get('elapsed_seconds') or 0) / 3600.0:.2f}h "
                f"global_covered={event.get('global_covered_bbs')}"
            )
    report_path = _find_report(arm_dir)
    if report_path is not None:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            report = {}
        print(
            f"[{name}] report: covered_bbs={report.get('covered_bbs')} "
            f"terminated_early={report.get('terminated_early')} "
            f"stop_reason={report.get('stall_watchdog_stop_reason')} "
            f"stage_truncations={len(report.get('stall_watchdog_stage_truncations') or [])} "
            f"wallclock_used={report.get('wallclock_used_seconds')}"
        )
    return {"name": name, "covered_max": covered_max, "merges": covered_first_seen}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ab_root", type=Path, help="A/B output root directory")
    parser.add_argument(
        "--control",
        type=Path,
        default=_CONTROL_PROGRESS,
        help="optional control-arm progress jsonl for reference overlay",
    )
    args = parser.parse_args()

    arm_b = args.ab_root / "armB_watchdog_on"
    arm_a = args.ab_root / "armA_watchdog_off"
    summarize_arm("B watchdog-on ", arm_b)
    summarize_arm("A watchdog-off", arm_a)
    if args.control.exists():
        # Reference: the 24h control reached 1682 only at t=13.39h.
        covered_max = 0
        arrival = None
        for event in _iter_events(args.control):
            covered = int(event.get("covered_bbs") or 0)
            if covered > covered_max:
                covered_max = covered
                arrival = float(event.get("elapsed_seconds") or 0.0)
        print(
            f"[control 24h ] covered_max={covered_max} "
            f"(last increase at t={(arrival or 0) / 3600.0:.2f}h)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
