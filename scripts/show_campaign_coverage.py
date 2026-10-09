#!/usr/bin/env python3
"""打印 24h 实验的覆盖采样序列（每 10 分钟一条）。

用法：
    /usr/local/bin/python3.10 scripts/show_campaign_coverage.py [--csv out.csv] [--tail N]

数据来源：<run>/ardupilot/ardupilot_Pixhawk1_STM32F427/ardupilot_Pixhawk1_STM32F427_coverage_progress.jsonl
只取 event=interval 的记录（stage_start/stage_end 是阶段边界事件，不是定时采样）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

DEFAULT_RUN = (
    ".lsgemu_runs/campaign_24h_ardupilot_pixhawk1_LLM_v41flash_probefix_20260918"
)
PROGRESS_NAME = "ardupilot_Pixhawk1_STM32F427_coverage_progress.jsonl"
RUN_GLOB = ".lsgemu_runs/campaign_*"


def newest_run() -> str:
    """默认取最新的 campaign 目录（实验目录常被改名保全，不要写死）。"""
    import glob

    if os.path.isdir(DEFAULT_RUN):
        return DEFAULT_RUN
    cands = [p for p in glob.glob(RUN_GLOB) if os.path.isdir(p)]
    if not cands:
        return DEFAULT_RUN
    return max(cands, key=os.path.getmtime)


def progress_path(run_root: str) -> str:
    return os.path.join(
        run_root, "ardupilot", "ardupilot_Pixhawk1_STM32F427", PROGRESS_NAME
    )


def load_samples(path: str) -> list[dict]:
    samples = []
    if not os.path.exists(path):
        return samples
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "interval":
                samples.append(rec)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="campaign 输出根目录（默认取最新一个）")
    ap.add_argument("--tail", type=int, default=0, help="只显示最后 N 条（0=全部）")
    ap.add_argument("--csv", default=None, help="同时导出 CSV")
    args = ap.parse_args()
    if not args.run:
        args.run = newest_run()

    path = progress_path(args.run)
    samples = load_samples(path)
    if args.tail:
        samples = samples[-args.tail :]

    print(f"进度文件: {path}")
    if not samples:
        print("（暂无 10 分钟采样记录）")
        return 0

    print(f"{'采样时刻':<20}{'运行时长':>12}{'BB 覆盖':>14}{'覆盖率':>10}  阶段")
    for rec in samples:
        wall = (rec.get("wallclock_time") or "?")[:19]
        elapsed = rec.get("elapsed_seconds") or 0.0
        covered = rec.get("covered_bbs")
        total = rec.get("total_bbs") or 0
        rate = rec.get("coverage_rate") or 0.0  # 字段本身就是百分比（0.1997 = 0.1997%）
        print(f"{wall:<20}{elapsed / 60:>10.1f}min{covered:>8}/{total:<6}{rate:>10.4f}%  {rec.get('stage')}")

    last = samples[-1]
    print(
        f"\n最新：t={ (last.get('elapsed_seconds') or 0) / 3600:.2f}h  "
        f"covered={last.get('covered_bbs')}/{last.get('total_bbs')}  "
        f"stage={last.get('stage')}  采样数={len(samples)}"
    )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["wallclock_time", "elapsed_seconds", "covered_bbs", "total_bbs", "coverage_rate", "stage"])
            for rec in samples:
                writer.writerow([
                    rec.get("wallclock_time"),
                    rec.get("elapsed_seconds"),
                    rec.get("covered_bbs"),
                    rec.get("total_bbs"),
                    rec.get("coverage_rate"),
                    rec.get("stage"),
                ])
        print(f"CSV 已写出: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
