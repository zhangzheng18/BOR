#!/usr/bin/env python3
"""
Summarize valid-basic-block coverage for elfmultifuzz firmware runs.

The coverage counting logic follows Fuzzware's convention:
`valid_basic_blocks.txt` contains one hexadecimal basic-block address per line,
and coverage is the size of the intersection between that set and the executed
basic block set reported by an emulator result.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .deployment_config import configured_path


SOURCE_ROOT = configured_path("LSGEMU_SOURCE_ROOT", Path(__file__).resolve().parents[1])
RUN_OUTPUT_ROOT = configured_path("LSGEMU_RUN_OUTPUT_DIR", SOURCE_ROOT / ".lsgemu_runs")
DEFAULT_VALID_ROOT = configured_path("ELFMULTIFUZZ_ROOT", SOURCE_ROOT / "datasets" / "elfmultifuzz")
DEFAULT_RESULT_ROOTS = [
    RUN_OUTPUT_ROOT,
    SOURCE_ROOT / "results",
]
DEFAULT_OUTPUT_JSON = RUN_OUTPUT_ROOT / "elfmultifuzz_valid_bb_coverage_summary.json"
DEFAULT_OUTPUT_CSV = RUN_OUTPUT_ROOT / "elfmultifuzz_valid_bb_coverage_summary.csv"
KNOWN_RESULT_SUFFIXES = (
    "_merged_result.json",
    "_ensemble_result.json",
    "_reservoir_test_result.json",
)
RESULT_PRIORITY = {
    "merged": 3,
    "ensemble": 2,
    "reservoir": 1,
}


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def parse_intish(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0)
        except ValueError:
            try:
                return int(text, 16)
            except ValueError:
                return None
    return None


@dataclass
class FirmwareCase:
    rel_path: str
    family: str
    name: str
    elf_path: Path
    valid_bb_path: Path
    valid_bbs: Set[int]
    aliases: Set[str] = field(default_factory=set)


@dataclass
class CoverageCandidate:
    case_rel_path: str
    report_path: Path
    report_type: str
    firmware_hint: str
    covered_bbs: Set[int]
    valid_covered_bbs: int
    valid_total_bbs: int
    coverage_pct: float
    absolute_covered_bbs: int
    runner: Optional[str]


def load_valid_bbs(path: Path) -> Set[int]:
    with path.open() as f:
        return {int(line.strip(), 16) & ~1 for line in f if line.strip()}


def discover_cases(valid_root: Path) -> Tuple[List[FirmwareCase], Dict[str, List[FirmwareCase]]]:
    cases: List[FirmwareCase] = []
    alias_map: Dict[str, List[FirmwareCase]] = {}

    for valid_bb_path in sorted(valid_root.glob("*/*/valid_basic_blocks.txt")):
        case_dir = valid_bb_path.parent
        elf_paths = sorted(case_dir.glob("*.elf"))
        if not elf_paths:
            continue
        elf_path = elf_paths[0]
        rel = case_dir.relative_to(valid_root)
        valid_bbs = load_valid_bbs(valid_bb_path)
        case = FirmwareCase(
            rel_path=str(rel),
            family=rel.parts[0],
            name=rel.parts[1],
            elf_path=elf_path,
            valid_bb_path=valid_bb_path,
            valid_bbs=valid_bbs,
        )
        alias_inputs = {
            str(rel),
            ".".join(rel.parts),
            rel.parts[-1],
            elf_path.name,
            elf_path.stem,
        }
        for item in alias_inputs:
            alias = normalize_name(item)
            if alias:
                case.aliases.add(alias)
        cases.append(case)

    for case in cases:
        for alias in case.aliases:
            alias_map.setdefault(alias, []).append(case)
    return cases, alias_map


def iter_candidate_report_paths(result_roots: Iterable[Path]) -> Iterable[Path]:
    for root in result_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name
            if name == "final_report.json" or name.endswith(KNOWN_RESULT_SUFFIXES):
                yield path


def report_type_for_path(path: Path) -> str:
    name = path.name
    if name.endswith("_merged_result.json"):
        return "merged"
    if name.endswith("_ensemble_result.json"):
        return "ensemble"
    if name.endswith("_reservoir_test_result.json"):
        return "reservoir"
    if name == "final_report.json":
        return "legacy_final_report"
    return "unknown"


def extract_firmware_hints(report_path: Path, data: object) -> List[str]:
    hints: List[str] = []

    if isinstance(data, dict) and data.get("firmware"):
        hints.append(str(data["firmware"]))

    name = report_path.name
    for suffix in KNOWN_RESULT_SUFFIXES:
        if name.endswith(suffix):
            hints.append(name[: -len(suffix)])
            break

    if name == "final_report.json":
        hints.append(report_path.parent.name)

    expanded: List[str] = []
    for hint in hints:
        expanded.append(hint)
        expanded.append(Path(hint).name)
        expanded.append(Path(hint).stem)
    return [item for item in expanded if item]


def match_case(report_path: Path, data: object, alias_map: Dict[str, List[FirmwareCase]]) -> Tuple[Optional[FirmwareCase], str]:
    hints = extract_firmware_hints(report_path, data)
    for hint in hints:
        alias = normalize_name(hint)
        if not alias:
            continue
        matches = alias_map.get(alias, [])
        if len(matches) == 1:
            return matches[0], hint
    return None, hints[0] if hints else ""


def extract_covered_bbs(data: object) -> Optional[Set[int]]:
    if not isinstance(data, dict):
        return None
    covered_list = data.get("covered_bb_list")
    if not isinstance(covered_list, list):
        return None

    covered: Set[int] = set()
    for item in covered_list:
        value = parse_intish(item)
        if value is not None:
            covered.add(value & ~1)
    return covered


def load_candidate(report_path: Path, alias_map: Dict[str, List[FirmwareCase]]) -> Tuple[Optional[CoverageCandidate], Optional[Dict[str, object]]]:
    try:
        with report_path.open() as f:
            data = json.load(f)
    except Exception as exc:
        return None, {"path": str(report_path), "status": "json_error", "error": str(exc)}

    case, firmware_hint = match_case(report_path, data, alias_map)
    if case is None:
        return None, {"path": str(report_path), "status": "unmatched_case", "firmware_hint": firmware_hint}

    covered_bbs = extract_covered_bbs(data)
    if covered_bbs is None:
        return None, {
            "path": str(report_path),
            "status": "unsupported_format",
            "case": case.rel_path,
            "firmware_hint": firmware_hint,
            "report_type": report_type_for_path(report_path),
        }

    valid_covered = len(covered_bbs & case.valid_bbs)
    candidate = CoverageCandidate(
        case_rel_path=case.rel_path,
        report_path=report_path,
        report_type=report_type_for_path(report_path),
        firmware_hint=firmware_hint,
        covered_bbs=covered_bbs,
        valid_covered_bbs=valid_covered,
        valid_total_bbs=len(case.valid_bbs),
        coverage_pct=(valid_covered / len(case.valid_bbs) * 100.0) if case.valid_bbs else 0.0,
        absolute_covered_bbs=len(covered_bbs),
        runner=data.get("runner") if isinstance(data, dict) else None,
    )
    return candidate, None


def choose_best_candidate(candidates: List[CoverageCandidate]) -> CoverageCandidate:
    return max(
        candidates,
        key=lambda item: (
            item.valid_covered_bbs,
            RESULT_PRIORITY.get(item.report_type, 0),
            item.absolute_covered_bbs,
            str(item.report_path),
        ),
    )


def summarize_cases(
    cases: List[FirmwareCase],
    usable_candidates: Dict[str, List[CoverageCandidate]],
    skipped_reports: List[Dict[str, object]],
) -> Dict[str, object]:
    skipped_by_case: Dict[str, List[Dict[str, object]]] = {}
    for item in skipped_reports:
        case_rel_path = item.get("case")
        if case_rel_path:
            skipped_by_case.setdefault(str(case_rel_path), []).append(item)

    summary_rows = []
    total_valid_all = 0
    total_best_valid_covered = 0
    usable_cases = 0
    unsupported_cases = 0
    missing_cases = 0

    for case in sorted(cases, key=lambda item: item.rel_path):
        total_valid_all += len(case.valid_bbs)
        candidates = usable_candidates.get(case.rel_path, [])
        if candidates:
            best = choose_best_candidate(candidates)
            total_best_valid_covered += best.valid_covered_bbs
            usable_cases += 1
            summary_rows.append({
                "family": case.family,
                "firmware": case.name,
                "case_rel_path": case.rel_path,
                "status": "ok",
                "valid_total_bbs": len(case.valid_bbs),
                "covered_valid_bbs": best.valid_covered_bbs,
                "coverage_pct": round(best.coverage_pct, 4),
                "absolute_covered_bbs": best.absolute_covered_bbs,
                "report_type": best.report_type,
                "report_path": str(best.report_path),
                "runner": best.runner,
                "candidate_reports": len(candidates),
            })
            continue

        skipped = skipped_by_case.get(case.rel_path, [])
        if skipped:
            unsupported_cases += 1
            summary_rows.append({
                "family": case.family,
                "firmware": case.name,
                "case_rel_path": case.rel_path,
                "status": "unsupported_report_format",
                "valid_total_bbs": len(case.valid_bbs),
                "covered_valid_bbs": 0,
                "coverage_pct": 0.0,
                "absolute_covered_bbs": 0,
                "report_type": skipped[0].get("report_type"),
                "report_path": skipped[0].get("path"),
                "runner": None,
                "candidate_reports": 0,
            })
            continue

        missing_cases += 1
        summary_rows.append({
            "family": case.family,
            "firmware": case.name,
            "case_rel_path": case.rel_path,
            "status": "missing_result",
            "valid_total_bbs": len(case.valid_bbs),
            "covered_valid_bbs": 0,
            "coverage_pct": 0.0,
            "absolute_covered_bbs": 0,
            "report_type": None,
            "report_path": None,
            "runner": None,
            "candidate_reports": 0,
        })

    return {
        "totals": {
            "firmwares": len(cases),
            "usable_results": usable_cases,
            "unsupported_only": unsupported_cases,
            "missing_results": missing_cases,
            "valid_bbs_all": total_valid_all,
            "best_covered_valid_bbs_sum": total_best_valid_covered,
            "best_coverage_pct_over_all_valid": round(
                total_best_valid_covered / total_valid_all * 100.0, 4
            ) if total_valid_all else 0.0,
        },
        "rows": summary_rows,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "firmware",
        "case_rel_path",
        "status",
        "valid_total_bbs",
        "covered_valid_bbs",
        "coverage_pct",
        "absolute_covered_bbs",
        "report_type",
        "report_path",
        "runner",
        "candidate_reports",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize elfmultifuzz valid basic block coverage from emulator result files.")
    parser.add_argument("--valid-root", default=str(DEFAULT_VALID_ROOT))
    parser.add_argument(
        "--result-root",
        action="append",
        dest="result_roots",
        help="Result root to scan. Can be specified multiple times.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    valid_root = Path(args.valid_root).resolve()
    result_roots = [Path(item).resolve() for item in (args.result_roots or [str(path) for path in DEFAULT_RESULT_ROOTS])]
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()

    cases, alias_map = discover_cases(valid_root)
    usable_candidates: Dict[str, List[CoverageCandidate]] = {}
    skipped_reports: List[Dict[str, object]] = []
    scanned_reports = 0

    for report_path in iter_candidate_report_paths(result_roots):
        scanned_reports += 1
        candidate, skipped = load_candidate(report_path, alias_map)
        if candidate is not None:
            usable_candidates.setdefault(candidate.case_rel_path, []).append(candidate)
        elif skipped is not None:
            skipped_reports.append(skipped)

    summary = summarize_cases(cases, usable_candidates, skipped_reports)
    payload = {
        "valid_root": str(valid_root),
        "result_roots": [str(path) for path in result_roots],
        "scanned_reports": scanned_reports,
        "skipped_reports": skipped_reports,
        **summary,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    write_csv(output_csv, summary["rows"])

    print(f"scanned_reports={scanned_reports}")
    print(f"usable_results={summary['totals']['usable_results']}")
    print(f"unsupported_only={summary['totals']['unsupported_only']}")
    print(f"missing_results={summary['totals']['missing_results']}")
    print(f"sum_best_valid_bb_coverage={summary['totals']['best_covered_valid_bbs_sum']}")
    print(f"sum_valid_bbs={summary['totals']['valid_bbs_all']}")
    print(f"coverage_pct_over_all_valid={summary['totals']['best_coverage_pct_over_all_valid']}")
    print(f"json={output_json}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
