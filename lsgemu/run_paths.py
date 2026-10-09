#!/usr/bin/env python3
"""Unique, content-identified workspaces for direct firmware runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile


def resolve_output_path(
    root: str | Path,
    *components: str | Path,
    label: str = "output path",
) -> Path:
    """Resolve a generated path and require it to remain below *root*.

    Resolving both paths makes pre-existing parent and leaf symlinks part of
    the containment decision.  Callers should perform this check immediately
    before creating or opening the returned path.
    """
    resolved_root = Path(root).expanduser().resolve()
    requested = Path(root).expanduser().joinpath(*components)
    resolved = requested.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes output root via traversal or symlink: "
            f"root={resolved_root}, requested={requested}, resolved={resolved}"
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_firmware_run_directory(
    root: str | Path,
    firmware: str | Path,
    *,
    namespace: str,
) -> Path:
    """Create a private run directory carrying a stable firmware identity prefix."""
    firmware_path = Path(firmware).expanduser().resolve()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", firmware_path.stem).strip("._")
    safe_stem = (safe_stem or "firmware")[:48]
    safe_namespace = re.sub(r"[^A-Za-z0-9._-]+", "_", str(namespace)).strip("._")
    safe_namespace = (safe_namespace or "run")[:48]
    workspace = resolve_output_path(
        root,
        safe_namespace,
        label="firmware run workspace",
    )
    workspace.mkdir(parents=True, exist_ok=True)
    identity = _file_sha256(firmware_path)[:12]
    return Path(tempfile.mkdtemp(
        prefix=f"{safe_stem}_{identity}_",
        dir=str(workspace),
    )).resolve()
