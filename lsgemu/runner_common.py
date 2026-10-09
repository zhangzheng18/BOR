#!/usr/bin/env python3
"""Common utilities shared by the deployable LSGEmu runner."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .deployment_config import configured_path, source_root
from .artifact_io import atomic_json_dump

logger = logging.getLogger(__name__)


SRCV4_ROOT = configured_path("LSGEMU_SOURCE_ROOT", source_root())
PROJECT_ROOT = configured_path("LSGEMU_PROJECT_ROOT", SRCV4_ROOT.parent)
DEFAULT_STATIC_CACHE_DIR = configured_path("LSGEMU_STATIC_CACHE_DIR", SRCV4_ROOT / ".lsgemu_cache")
DEFAULT_VALID_BB_ROOT = configured_path(
    "LSGEMU_VALID_BB_ROOT",
    configured_path("ELFMULTIFUZZ_ROOT", SRCV4_ROOT / "datasets" / "elfmultifuzz"),
)

_VALID_BB_RESOLUTION_CACHE: Dict[str, Tuple[Optional[Path], Set[int]]] = {}
_FILE_SHA1_CACHE: Dict[str, str] = {}


def _loaded_unicorn_library_path() -> Optional[str]:
    try:
        from unicorn.unicorn_py3 import unicorn as unicorn_py3
    except Exception:
        return None

    lib = getattr(unicorn_py3, "uclib", None)
    lib_name = getattr(lib, "_name", None)
    if not lib_name:
        return None
    return str(lib_name)


def _static_cache_path(firmware_path: Path) -> Path:
    cache_dir = Path(os.environ.get("LSGEMU_STATIC_CACHE_DIR", DEFAULT_STATIC_CACHE_DIR))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Shared /tmp fallback: pin the directory to this user and tighten
        # permissions. The directory name is predictable, so without 0700 any
        # local user could pre-plant or replace cache pickles here.
        cache_dir = Path("/tmp/lsgemu_static_cache")
        try:
            cache_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError:
            # Pre-existing directory (possibly created by an older run with
            # loose permissions, or hostilely pre-created by another user).
            stat_mode = cache_dir.stat().st_mode & 0o777
            if stat_mode != 0o700 or not os.access(cache_dir, os.W_OK | os.X_OK):
                # Fall back to a user-private path instead of a shared one.
                cache_dir = Path(
                    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
                ) / "lsgemu_static_cache"
                cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            else:
                cache_dir.chmod(0o700)
        except OSError:
            cache_dir = Path(
                os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            ) / "lsgemu_static_cache"
            cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    digest = hashlib.sha1(str(firmware_path).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{firmware_path.stem}_{digest}_static_cache.pkl"


def _loop_exit_hints_enabled() -> bool:
    return os.environ.get("LSGEMU_LOOP_EXIT_HINTS", "1").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _hex_to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value) & 0xFFFFFFFF
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) & 0xFFFFFFFF if text.lower().startswith("0x") else int(text, 0) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None


def _instruction_dict(insn) -> Dict[str, object]:
    return {
        "address": insn.address,
        "mnemonic": insn.mnemonic,
        "operands": insn.op_str,
        "size": insn.size,
    }


def _dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(float(os.environ.get(name, default) or default))
    except (TypeError, ValueError):
        return int(default)


def _segment_schedule_identity() -> Tuple[int, int]:
    """Return (segment_index, elapsed_offset_seconds) for deterministic task rotation."""
    elapsed_offset = _env_int("LSGEMU_PROGRESS_ELAPSED_OFFSET_SECONDS", 0)
    explicit_index = os.environ.get("LSGEMU_CASE_SEGMENT_INDEX")
    if explicit_index is not None:
        segment_index = _env_int("LSGEMU_CASE_SEGMENT_INDEX", 0)
    else:
        segment_seconds = max(1, _env_int("LSGEMU_CASE_SEGMENT_SECONDS", 1200))
        segment_index = max(0, elapsed_offset // segment_seconds)
    return max(0, segment_index), max(0, elapsed_offset)


def _segment_rotation_offset(length: int, salt: str) -> int:
    if length <= 1:
        return 0
    segment_index, elapsed_offset = _segment_schedule_identity()
    diversify = os.environ.get("LSGEMU_SEGMENT_DIVERSIFY", "").strip().lower()
    if segment_index <= 0 and elapsed_offset <= 0 and diversify not in {"1", "true", "yes", "on"}:
        return 0
    digest = hashlib.sha1(
        f"{salt}:{segment_index}:{elapsed_offset}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % length


def _segment_rotate_sequence(values: Iterable, salt: str) -> List:
    items = list(values or [])
    offset = _segment_rotation_offset(len(items), salt)
    if offset <= 0:
        return items
    return items[offset:] + items[:offset]


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _sanitize_u32(value: int) -> int:
    return max(0, min(0xFFFFFFFF, int(value)))


def _signed32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def _file_sha1(path: Path) -> str:
    resolved = str(path.resolve())
    cached = _FILE_SHA1_CACHE.get(resolved)
    if cached is not None:
        return cached

    digest = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_SHA1_CACHE[resolved] = value
    return value


def _load_valid_bbs(path: Path) -> Set[int]:
    with path.open() as f:
        return {int(line.strip(), 16) & ~1 for line in f if line.strip()}


def _resolve_valid_bb_metadata(firmware_path: Path) -> Tuple[Optional[Path], Set[int]]:
    firmware_path = firmware_path.resolve()
    cache_key = str(firmware_path)
    cached = _VALID_BB_RESOLUTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    local_path = firmware_path.parent / "valid_basic_blocks.txt"
    if local_path.exists():
        result = (local_path, _load_valid_bbs(local_path))
        _VALID_BB_RESOLUTION_CACHE[cache_key] = result
        return result

    valid_root = Path(os.environ.get("LSGEMU_VALID_BB_ROOT", DEFAULT_VALID_BB_ROOT))
    if not valid_root.exists():
        # r40 P4/P0-3b（主代理裁决：fail-closed）：设备侧 valid_* 分母未接线
        # 之前，缺分母必须显式降级（denominator=unavailable），报告字段保持
        # 空集读数 + 状态位，禁止静默空集冒充数字。
        logger.warning(
            "r40 P4 降级：valid 分母未提供——denominator=unavailable"
            "（LSGEMU_VALID_BB_ROOT=%s 不存在，且固件同目录无 valid_basic_blocks.txt）",
            valid_root,
        )
        result = (None, set())
        _VALID_BB_RESOLUTION_CACHE[cache_key] = result
        return result

    aliases = {
        _normalize_name(firmware_path.name),
        _normalize_name(firmware_path.stem),
        _normalize_name(firmware_path.parent.name),
    }
    aliases.discard("")

    candidate_cases: List[Tuple[Path, Path]] = []
    all_cases: List[Tuple[Path, Path]] = []
    for valid_bb_path in sorted(valid_root.glob("*/*/valid_basic_blocks.txt")):
        case_dir = valid_bb_path.parent
        elf_paths = sorted(case_dir.glob("*.elf"))
        if not elf_paths:
            continue
        elf_path = elf_paths[0]
        all_cases.append((valid_bb_path, elf_path))
        case_aliases = {
            _normalize_name(case_dir.name),
            _normalize_name(elf_path.name),
            _normalize_name(elf_path.stem),
            _normalize_name(".".join(case_dir.relative_to(valid_root).parts)),
        }
        if aliases & case_aliases:
            candidate_cases.append((valid_bb_path, elf_path))

    if len(candidate_cases) == 1:
        valid_bb_path, _elf_path = candidate_cases[0]
        result = (valid_bb_path, _load_valid_bbs(valid_bb_path))
        _VALID_BB_RESOLUTION_CACHE[cache_key] = result
        return result

    try:
        firmware_sha1 = _file_sha1(firmware_path)
    except Exception:
        firmware_sha1 = ""

    if firmware_sha1:
        search_cases = candidate_cases or all_cases
        for valid_bb_path, elf_path in search_cases:
            try:
                if _file_sha1(elf_path) == firmware_sha1:
                    result = (valid_bb_path, _load_valid_bbs(valid_bb_path))
                    _VALID_BB_RESOLUTION_CACHE[cache_key] = result
                    return result
            except Exception:
                continue

    # r40 P4/P0-3b：root 存在但无唯一匹配 ⇒ 同样显式降级，不静默。
    logger.warning(
        "r40 P4 降级：valid 分母未提供——denominator=unavailable"
        "（LSGEMU_VALID_BB_ROOT=%s 下无 %s 的唯一匹配 valid_basic_blocks.txt）",
        valid_root,
        firmware_path.name,
    )
    result = (None, set())
    _VALID_BB_RESOLUTION_CACHE[cache_key] = result
    return result


def _write_json_debug_record(path: Path, payload: Dict[str, object]) -> None:
    try:
        atomic_json_dump(payload, path, indent=2)
    except Exception:
        pass
