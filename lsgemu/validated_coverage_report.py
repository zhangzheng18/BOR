#!/usr/bin/env python3
"""validated-only 覆盖报告：只看 validated BB，diagnostic 一律不计。

用法: validated_coverage_report.py <run_dir 或 report.json> [更多...]
分母：<firmware>_angr_cfgfast.txt（angr CFGFast 静态枚举），按固件名匹配。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Denominator directory: the angr CFGFast block lists shipped with the
# artifact. Override with LSGEMU_REACHABILITY_DIR, otherwise the default
# is <artifact root>/benchmarks/reachability/angr_reachable.
ANG_DIR = Path(
    os.environ.get("LSGEMU_REACHABILITY_DIR")
    or (Path(__file__).resolve().parents[1] / "benchmarks" / "reachability" / "angr_reachable")
)


def load_denominator(firmware_stem: str) -> int | None:
    for cand in sorted(ANG_DIR.glob("*_angr_cfgfast.txt")):
        if firmware_stem.split(".")[0] in cand.name or cand.name.split("_angr")[0] in firmware_stem:
            return sum(1 for _ in cand.open(encoding="utf-8", errors="ignore"))
    return None


def analyze(path: Path) -> dict | None:
    if path.is_dir():
        cands = sorted(path.glob("*_interleaved_report.json"))
        if not cands:
            return None
        path = cands[0]
    if not path.is_file():
        return None
    try:
        r = json.load(open(path, encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    if "validated_replay_bbs" not in json.dumps(r)[:4000] and "path_naturalization_summary" not in r:
        return None
    valid = set(int(x) & ~1 for x in ((r.get("path_naturalization_summary") or {})
                                      .get("canonical_validated_replay_bb_list") or []))
    contract = r.get("strict_real_entry_replay_contract", {}) or {}
    if not valid and contract.get("validated_replay_bbs"):
        return {"path": str(path), "firmware": path.name, "validated": None,
                "note": "报告只给计数未给清单", "count": contract["validated_replay_bbs"]}
    stem = path.name.replace("_interleaved_report.json", "")
    denom = load_denominator(stem)
    return {
        "path": str(path),
        "firmware": stem,
        "validated": len(valid),
        "denominator": denom,
        "rate": (len(valid) / denom * 100.0) if denom else None,
        "diagnostic_ignored": contract.get("diagnostic_replay_bbs"),
        "diagnostic_ignored_observed": contract.get("diagnostic_replay_observed_bbs"),
    }


def main(argv: list[str]) -> int:
    rows = [x for x in (analyze(Path(a)) for a in argv) if x]
    if not rows:
        print("没有可解析的报告")
        return 1
    print(f"{'固件':44} {'validated':>10} {'分母(CFGFast)':>14} {'valid%':>8}  诊断(已忽略)")
    print("-" * 92)
    for x in rows:
        if x.get("validated") is None:
            print(f"{x['firmware']:44} {'计数: ' + str(x['count']):>10} {'—':>14} {'—':>8}")
            continue
        d = f"{x['denominator']:,}" if x["denominator"] else "—"
        rt = f"{x['rate']:.2f}%" if x.get("rate") else "—"
        print(f"{x['firmware']:44} {x['validated']:>10,} {d:>14} {rt:>8}  {x['diagnostic_ignored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
