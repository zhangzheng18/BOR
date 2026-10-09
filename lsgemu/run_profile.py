#!/usr/bin/env python3
"""Run profile capture and hashing.

LSGEMU has many environment switches.  A stable profile hash makes coverage
changes explainable without immediately rewriting every call site to YAML.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .artifact_io import atomic_json_dump, atomic_write_text


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_VOLATILE_TOP_LEVEL_PROFILE_FIELDS = {
    "profile_hash",
    "created_at",
    "argv",
    # Commit identity is provenance, not execution semantics: the same argv on
    # two commits keeps one profile hash so coverage deltas stay attributable
    # to the change under test rather than to the commit that carries it.
    "vcs",
}
_VOLATILE_ENVIRONMENT_KEYS = {
    "LSGEMU_ATTEMPT_ID",
    "LSGEMU_ATTEMPT_DIR",
    "LSGEMU_ATTEMPT_OUTPUT_DIR",
    "LSGEMU_ATTEMPT_REPORT_PATH",
    "LSGEMU_ATTEMPT_STATUS_PATH",
    "LSGEMU_CAMPAIGN_FINGERPRINT",
    "LSGEMU_PROGRESS_JSONL",
    "LSGEMU_RUN_OUTPUT_DIR",
    "LSGEMU_RUN_PROFILE_FILE",
    # Generated per-attempt storage paths do not change replay semantics. The
    # disk-backing switch itself remains in the hash because it affects timing.
    "LSGEMU_SNAPSHOT_STORAGE_DIR",
    "LSGEMU_SNAPSHOT_STORAGE_AUTO",
    "LSGEMU_SNAPSHOT_STORAGE_RUN_KEY",
}


def profile_hash_payload(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable, semantic portion of a collected run profile.

    Attempt identifiers, output locations, timestamps, and the process argv
    identify an invocation rather than its execution semantics.  Keeping them
    out of the hash lets two equivalent runs be compared across serial
    attempts while preserving explicit CLI/environment/toolchain settings.
    """
    payload = {
        key: value
        for key, value in dict(profile or {}).items()
        if key not in _VOLATILE_TOP_LEVEL_PROFILE_FIELDS
    }
    environment = payload.get("environment")
    if isinstance(environment, dict):
        payload["environment"] = {
            key: value
            for key, value in environment.items()
            if str(key) not in _VOLATILE_ENVIRONMENT_KEYS
        }
    # The path of an input profile is not semantic; its content digest is.
    payload.pop("input_profile_file", None)
    payload.pop("profile_file", None)
    return payload


def profile_hash(profile: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(profile_hash_payload(profile)).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# Profile environment keys that must never be overridden by a profile file:
# they identify the profile itself or pin the config-selection machinery.
_PROTECTED_PROFILE_ENVIRONMENT_KEYS = frozenset({
    "LSGEMU_CALLER_CWD",
    "LSGEMU_CONFIG_DIR",
    "LSGEMU_CONFIG_FILE",
    "LSGEMU_RUN_PROFILE_FILE",
    "LSGEMU_RUN_PROFILE_FILE_SHA256",
})


def load_run_profile(path: str | Path) -> Dict[str, Any]:
    """Load a YAML/JSON run profile."""
    profile_path = Path(path).expanduser().resolve()
    with profile_path.open() as f:
        if profile_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml

                payload = yaml.safe_load(f)
            except Exception as exc:
                raise ValueError(f"failed to load YAML run profile {profile_path}: {exc}") from exc
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"run profile must be a mapping: {profile_path}")
    payload = dict(payload)
    payload.setdefault("profile_file", str(profile_path))
    payload.setdefault("profile_file_sha256", file_sha256(profile_path))
    return payload


def apply_run_profile_environment(profile: Dict[str, Any]) -> Dict[str, str]:
    """Apply only LSGEMU_* environment entries from a loaded run profile."""
    applied: Dict[str, str] = {}
    env = profile.get("environment", {})
    if not isinstance(env, dict):
        return applied
    for key, value in sorted(env.items()):
        key_text = str(key)
        if (
            not key_text.startswith("LSGEMU_")
            or key_text in _PROTECTED_PROFILE_ENVIRONMENT_KEYS
        ):
            continue
        if value is None:
            os.environ.pop(key_text, None)
            applied[key_text] = ""
            continue
        value_text = str(value)
        os.environ[key_text] = value_text
        applied[key_text] = value_text
    profile_file = profile.get("profile_file")
    if profile_file:
        os.environ["LSGEMU_RUN_PROFILE_FILE"] = str(profile_file)
        applied["LSGEMU_RUN_PROFILE_FILE"] = str(profile_file)
    profile_file_sha256 = profile.get("profile_file_sha256")
    if profile_file_sha256:
        os.environ["LSGEMU_RUN_PROFILE_FILE_SHA256"] = str(profile_file_sha256)
        applied["LSGEMU_RUN_PROFILE_FILE_SHA256"] = str(profile_file_sha256)
    return applied


def _collect_vcs_state() -> Optional[Dict[str, Any]]:
    """Best-effort git provenance for the lsgemu source tree.

    Returns ``{"revision": <short sha>, "dirty": <bool>}``, or ``None`` when
    git or the repository is unavailable.  Never raises: a missing VCS must
    not break profile collection.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rev.returncode != 0 or not rev.stdout.strip():
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "revision": rev.stdout.strip(),
            "dirty": status.returncode == 0 and bool(status.stdout.strip()),
        }
    except Exception:
        return None


def collect_run_profile(
    *,
    firmware_path: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    include_unset_defaults: bool = False,
) -> Dict[str, Any]:
    env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("LSGEMU_")
    }
    profile = {
        "schema": "lsgemu.run_profile.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "firmware_path": str(firmware_path) if firmware_path is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vcs": _collect_vcs_state(),
        "argv": list(sys.argv),
        "cli_args": dict(cli_args or {}),
        "environment": env,
        "include_unset_defaults": bool(include_unset_defaults),
    }
    if os.environ.get("LSGEMU_RUN_PROFILE_FILE"):
        profile["input_profile_file"] = os.environ.get("LSGEMU_RUN_PROFILE_FILE")
    if os.environ.get("LSGEMU_RUN_PROFILE_FILE_SHA256"):
        profile["input_profile_file_sha256"] = os.environ.get("LSGEMU_RUN_PROFILE_FILE_SHA256")
    if extra:
        profile["extra"] = dict(extra)
    profile["profile_hash"] = profile_hash(profile)
    return profile


def write_run_profile(path: str | Path, profile: Dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        serialized = yaml.safe_dump(profile, sort_keys=True, allow_unicode=False)
        atomic_write_text(output, serialized)
    except Exception:
        atomic_json_dump(profile, output, indent=2, sort_keys=True)
    return output


def run_profile_arg_from_argv(argv: Optional[list[str]] = None) -> Optional[str]:
    """Return the last explicit ``--run-profile`` value, if present.

    Deliberately minimal (no argparse) so it can run before heavy modules are
    imported and freeze path-dependent globals.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    value: Optional[str] = None
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--run-profile" and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        if token.startswith("--run-profile="):
            value = token.split("=", 1)[1]
        index += 1
    return value


def apply_run_profile_from_argv(
    argv: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Apply a CLI run profile before modules freeze path-dependent globals.

    ``runner_common`` and friends read ``LSGEMU_*`` environment variables at
    import time; applying the profile only in ``main()`` would silently void
    every path-type entry in it.
    """
    profile_path = run_profile_arg_from_argv(argv)
    if not profile_path:
        return None
    profile = load_run_profile(profile_path)
    apply_run_profile_environment(profile)
    return profile
