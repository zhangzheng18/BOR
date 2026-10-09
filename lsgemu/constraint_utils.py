"""Utilities for seeding and merging runtime-learned constraints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .artifact_io import atomic_json_dump
from .runner_models import external_input_site_identity


def parse_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return None


def normalize_constraint_item(item: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not isinstance(item, dict):
        return None
    address = parse_int(item.get("address"))
    value = parse_int(item.get("value"))
    if address is None or value is None:
        return None

    normalized = dict(item)
    normalized["address"] = f"0x{address & 0xFFFFFFFF:08x}"
    normalized["value"] = f"0x{value & 0xFFFFFFFF:08x}"
    for key in ("read_pc", "pc", "constraint_pc", "branch_pc"):
        parsed = parse_int(normalized.get(key))
        if parsed is not None:
            normalized[key] = f"0x{parsed & 0xFFFFFFFF:08x}"
        elif key in normalized:
            normalized[key] = None
    occurrence = parse_int(normalized.get("read_occurrence"))
    if occurrence is not None:
        normalized["read_occurrence"] = max(1, int(occurrence))
    width = parse_int(normalized.get("width") or normalized.get("value_width"))
    if width is not None:
        normalized["width"] = min(32, max(1, int(width)))
    return normalized


def constraint_identity(item: Dict[str, object]) -> Tuple[object, ...]:
    return external_input_site_identity(
        constraint_type=item.get("type"),
        address=parse_int(item.get("address")) or 0,
        read_pc=parse_int(item.get("read_pc") or item.get("pc")),
        read_occurrence=parse_int(item.get("read_occurrence")),
        input_kind=item.get("input_kind") or item.get("type") or "",
    )


def load_constraints(path: Path) -> List[Dict[str, object]]:
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return []
    items = data.get("constraints", []) if isinstance(data, dict) else []
    constraints = []
    for item in items:
        normalized = normalize_constraint_item(item)
        if normalized:
            constraints.append(normalized)
    return constraints


def merge_constraint_files(paths: Iterable[Path], output_path: Path) -> int:
    constraints: List[Dict[str, object]] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for item in load_constraints(path):
            key = constraint_identity(item)
            if key in seen:
                continue
            seen.add(key)
            constraints.append(item)

    atomic_json_dump({"constraints": constraints}, output_path, indent=2)
    return len(constraints)


def discover_learned_constraint_files(
    firmware: Path,
    search_roots: Sequence[Path],
    exclude: Optional[Path] = None,
) -> List[Path]:
    filename = f"{firmware.stem}_lsgemu_constraints.json"
    excluded = exclude.resolve() if exclude is not None else None
    candidates = []
    seen = set()
    for root in search_roots:
        if not root or not root.exists():
            continue
        for path in root.rglob(filename):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if excluded is not None and resolved == excluded:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return sorted(candidates)


def discover_best_learned_constraint_file(
    firmware: Path,
    search_roots: Sequence[Path],
    exclude: Optional[Path] = None,
) -> Optional[Path]:
    report_names = (
        f"{firmware.stem}_interleaved_report.json",
        f"{firmware.stem}_reservoir_test_result.json",
    )
    excluded = exclude.resolve() if exclude is not None else None
    best_score = -1
    best_path: Optional[Path] = None
    seen_reports = set()
    for root in search_roots:
        if not root or not root.exists():
            continue
        for report_name in report_names:
            for report_path in root.rglob(report_name):
                try:
                    resolved_report = report_path.resolve()
                except OSError:
                    continue
                if resolved_report in seen_reports:
                    continue
                seen_reports.add(resolved_report)
                try:
                    with resolved_report.open() as f:
                        report = json.load(f)
                except Exception:
                    continue
                if not isinstance(report, dict):
                    continue
                score = parse_int(report.get("valid_covered_bbs"))
                if score is None:
                    score = parse_int(report.get("covered_bbs")) or 0
                constraint_path = report.get("constraint_file")
                if constraint_path:
                    candidate = Path(str(constraint_path))
                    if not candidate.is_absolute():
                        candidate = resolved_report.parent / candidate
                else:
                    candidate = resolved_report.with_name(f"{firmware.stem}_lsgemu_constraints.json")
                try:
                    resolved_candidate = candidate.resolve()
                except OSError:
                    continue
                if excluded is not None and resolved_candidate == excluded:
                    continue
                if not resolved_candidate.exists():
                    continue
                if score > best_score:
                    best_score = score
                    best_path = resolved_candidate
    return best_path


def seed_constraint_file(
    firmware: Path,
    destination: Path,
    *,
    learned_mode: str = "none",
    learned_roots: Sequence[Path] = (),
) -> int:
    """Create a constraint file from static and optional learned constraints.

    Existing destination files are left untouched because they may already
    contain constraints learned by an interrupted or resumable run.
    """
    if destination.exists():
        return len(load_constraints(destination))

    source = firmware.with_name(f"{firmware.stem}_lsgemu_constraints.json")
    if learned_mode == "none":
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return len(load_constraints(destination))
        return 0

    paths: List[Path] = []
    if source.exists():
        paths.append(source)
    if learned_mode == "best":
        best = discover_best_learned_constraint_file(firmware, learned_roots, exclude=destination)
        if best is not None:
            paths.append(best)
    elif learned_mode == "union":
        paths.extend(discover_learned_constraint_files(firmware, learned_roots, exclude=destination))
    if not paths:
        return 0
    return merge_constraint_files(paths, destination)


def merge_learned_constraints_into(
    firmware: Path,
    destination: Path,
    *,
    learned_mode: str,
    learned_roots: Sequence[Path] = (),
) -> int:
    paths: List[Path] = []
    if destination.exists():
        paths.append(destination)
    if learned_mode == "best":
        best = discover_best_learned_constraint_file(firmware, learned_roots, exclude=destination)
        if best is not None:
            paths.append(best)
    elif learned_mode == "union":
        paths.extend(discover_learned_constraint_files(firmware, learned_roots, exclude=destination))
    else:
        return len(load_constraints(destination)) if destination.exists() else 0
    if not paths:
        return 0
    return merge_constraint_files(paths, destination)
