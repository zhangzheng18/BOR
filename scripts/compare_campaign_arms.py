#!/usr/bin/env python3
"""对比两条 campaign 臂（LLM / 无 LLM）的覆盖-时间轨迹，用于量化大模型增益。

用法:
    python3.10 scripts/compare_campaign_arms.py <run_dir_A> <run_dir_B> [--step 600]
默认从 coverage_progress.jsonl 取 event=interval 采样，按 wall-clock 秒对齐。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple


def load_run(run_dir: str) -> Tuple[List[dict], List[dict]]:
    """返回 (interval 采样列表, stage 记录列表)，按 elapsed_seconds 排序。"""
    files = glob.glob(os.path.join(run_dir, "**", "*_coverage_progress.jsonl"), recursive=True)
    intervals: List[dict] = []
    stages: List[dict] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "interval":
                        intervals.append(rec)
                    elif rec.get("event", "").startswith("stage_"):
                        stages.append(rec)
        except OSError:
            continue
    intervals.sort(key=lambda r: float(r.get("elapsed_seconds") or 0.0))
    stages.sort(key=lambda r: float(r.get("elapsed_seconds") or 0.0))
    return intervals, stages


def label(run_dir: str) -> str:
    name = os.path.basename(run_dir.rstrip("/"))
    return name[:46]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--step", type=int, default=600, help="对齐步长（秒），默认 600")
    args = ap.parse_args()

    data = {}
    for tag, run_dir in (("A", args.run_a), ("B", args.run_b)):
        intervals, stages = load_run(run_dir)
        data[tag] = {"intervals": intervals, "stages": stages, "dir": run_dir}
        print(f"[{tag}] {label(run_dir)}: interval 采样 {len(intervals)} 条, 阶段事件 {len(stages)} 条")

    def at(tag: str, t: float) -> int | None:
        """取 <=t 的最后一条采样的 covered_bbs；该臂在 t 之前没有采样则返回 None。"""
        best = None
        for rec in data[tag]["intervals"]:
            if float(rec.get("elapsed_seconds") or 0.0) <= t:
                best = int(rec.get("covered_bbs") or 0)
            else:
                break
        return best

    marks = []
    t = args.step
    last_t = max(
        [float(r.get("elapsed_seconds") or 0.0) for r in data["A"]["intervals"]]
        + [float(r.get("elapsed_seconds") or 0.0) for r in data["B"]["intervals"]]
        + [0.0]
    )
    while t <= last_t + args.step:
        marks.append(t)
        t += args.step

    print()
    print(f"{'wall(s)':>8} | {'A covered':>10} | {'B covered':>10} | {'Δ(A-B)':>8}")
    print("-" * 46)
    for mark in marks:
        a = at("A", mark)
        b = at("B", mark)
        delta = f"{a - b:>+8d}" if (a is not None and b is not None) else "       -"
        print(
            f"{mark:>8.0f} | {(a if a is not None else '-'):>10} | "
            f"{(b if b is not None else '-'):>10} | {delta}"
        )

    print()
    for tag in ("A", "B"):
        print(f"=== [{tag}] 阶段时间线（{label(data[tag]['dir'])}）===")
        for rec in data[tag]["stages"]:
            print(
                f"  {float(rec.get('elapsed_seconds') or 0.0):>9.1f}s "
                f"{rec.get('event'):<12} {str(rec.get('stage') or ''):<38} "
                f"covered={rec.get('covered_bbs')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
