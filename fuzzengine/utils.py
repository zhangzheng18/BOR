#!/usr/bin/env python3
"""Shared utility helpers for the deployable fuzzengine package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Set


def parse_int(value: object, *, mask_u32: bool = True) -> Optional[int]:
    """Parse decimal/hex integer-like values used in reports and configs."""
    if value is None:
        return None
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = int(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = int(text, 0)
        except (TypeError, ValueError):
            return None
    return parsed & 0xFFFFFFFF if mask_u32 else parsed


def parse_int_set(values: object, *, normalize_thumb: bool = True) -> Set[int]:
    out: Set[int] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        parsed = parse_int(item)
        if parsed is not None:
            out.add(parsed & ~1 if normalize_thumb else parsed)
    return out


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def read_json_object(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def file_sha256(path: object) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def short_hash(data: object, size: int = 16) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8", errors="replace")).hexdigest()[: int(size)]


def safe_fragment(value: str, *, default: str = "unknown", max_len: int = 120) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or default))
    return text[: int(max_len)].strip("._-") or default


def load_seed_files(paths: Iterable[object]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        elif path.is_file():
            files.append(path)
    return files
