#!/usr/bin/env python3
"""Deterministic identity records for the native/static analysis toolchain."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {"path": None, "exists": False}
    try:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "exists": True,
            "size": int(stat.st_size),
            "sha256": sha256_file(resolved),
        }
    except (OSError, ValueError):
        return {"path": str(path), "exists": False}


def module_identity(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"module": module_name, "available": False, "error": str(exc)}
    path = getattr(module, "__file__", None)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as package_version

            version = package_version(module_name)
        except Exception:
            version = None
    record: dict[str, object] = {
        "module": module_name,
        "available": True,
        "version": str(version) if version is not None else None,
    }
    if path:
        record["file"] = file_identity(path)
    return record


def executable_identity(
    command: str | Path | None,
    args: Iterable[str] = (),
    *,
    success_output_markers: Iterable[str] = (),
) -> dict[str, object]:
    if not command:
        return {"command": None, "available": False}
    command_text = str(command)
    resolved = command_text
    if not os.path.isabs(command_text):
        resolved = shutil.which(command_text) or str(Path(command_text).expanduser())
    try:
        completed = subprocess.run(
            [command_text, *[str(arg) for arg in args]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        output = (completed.stdout or "").strip()
        probe_succeeded = completed.returncode == 0
        marker_matched = any(
            str(marker) in output
            for marker in success_output_markers
            if str(marker)
        )
        identity = file_identity(resolved if Path(resolved).exists() else command_text)
        identity.update({
            "command": command_text,
            "available": bool(probe_succeeded or marker_matched),
            "probe_succeeded": bool(probe_succeeded),
            "success_output_marker_matched": bool(marker_matched),
            "returncode": int(completed.returncode),
            "version_output": output[-1000:],
        })
        return identity
    except Exception as exc:
        return {"command": command_text, "available": False, "error": str(exc)}


def collect_toolchain_fingerprint(*, valid_bb_path: str | Path | None = None) -> dict[str, object]:
    """Collect stable dependency identities without changing runtime behavior."""
    ghidra_launcher = os.environ.get("GHIDRA_ANALYZE_HEADLESS")
    if not ghidra_launcher and os.environ.get("GHIDRA_INSTALL_DIR"):
        ghidra_launcher = str(
            Path(os.environ["GHIDRA_INSTALL_DIR"]) / "support" / "analyzeHeadless"
        )
    unicorn_library = os.environ.get("LSGEMU_UNICORN_SHARED_LIB") or os.environ.get("LIBUNICORN_PATH")
    try:
        from unicorn.unicorn_py3 import unicorn as unicorn_py3

        loaded_library = getattr(getattr(unicorn_py3, "uclib", None), "_name", None)
        if loaded_library:
            unicorn_library = str(loaded_library)
    except Exception:
        pass
    if unicorn_library and Path(unicorn_library).is_dir():
        candidates = sorted(Path(unicorn_library).glob("libunicorn.so*"))
        unicorn_library = str(candidates[0]) if candidates else unicorn_library

    valid_identity = file_identity(valid_bb_path)
    runtime_payload: dict[str, object] = {
        "schema": "lsgemu.toolchain_fingerprint.v1",
        "python": {
            "version": sys.version,
            "executable": file_identity(sys.executable),
        },
        "platform": platform.platform(),
        "unicorn_python": module_identity("unicorn"),
        "capstone": module_identity("capstone"),
        "z3": module_identity("z3"),
        "unicorn_native": file_identity(unicorn_library),
        "ghidra_launcher": executable_identity(
            ghidra_launcher,
            ("-h",),
            success_output_markers=("[-postscript",),
        ),
        "java": executable_identity(os.environ.get("JAVA", "java"), ("-version",)),
    }
    runtime_encoded = json.dumps(
        runtime_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    runtime_payload["runtime_fingerprint"] = hashlib.sha256(runtime_encoded).hexdigest()
    payload = {
        **runtime_payload,
        "valid_basic_blocks": valid_identity,
    }
    full_identity = {
        key: value
        for key, value in payload.items()
        if key != "runtime_fingerprint"
    }
    encoded = json.dumps(
        full_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload
