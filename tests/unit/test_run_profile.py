#!/usr/bin/env python3
"""Smoke-test run profile loading, environment application, and hashing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.run_profile import (
    apply_run_profile_environment,
    collect_run_profile,
    load_run_profile,
    profile_hash_payload,
    profile_hash,
)


def test_profile_hash_payload_ignores_vcs():
    """The ``vcs`` provenance field must not rotate the profile hash."""
    base = {
        "schema": "lsgemu.run_profile.v1",
        "firmware_path": "/tmp/fw.elf",
        "python": "3.10.14",
        "environment": {"LSGEMU_IRQ_DELIVERY": "1"},
    }
    with_vcs = dict(base, vcs={"revision": "85445ed", "dirty": True})
    other_vcs = dict(base, vcs={"revision": "deadbee", "dirty": False})
    assert profile_hash_payload(with_vcs) == profile_hash_payload(base)
    assert "vcs" not in profile_hash_payload(with_vcs)
    assert profile_hash(with_vcs) == profile_hash(base)
    assert profile_hash(other_vcs) == profile_hash(base)


def test_collect_run_profile_vcs_field(monkeypatch):
    """``vcs`` is None-or-dict and never raises, even without git."""
    profile = collect_run_profile(firmware_path="/tmp/fw.elf", cli_args={"time": 1})
    vcs = profile.get("vcs")
    assert vcs is None or isinstance(vcs, dict)
    if isinstance(vcs, dict):
        assert isinstance(vcs.get("revision"), str) and vcs["revision"]
        assert isinstance(vcs.get("dirty"), bool)
    monkeypatch.setenv("PATH", "/nonexistent-lsgemu-vcs-test")
    assert collect_run_profile(firmware_path="/tmp/fw.elf").get("vcs") is None


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lsgemu_run_profile_") as tmpdir:
        profile_path = Path(tmpdir) / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "environment": {
                        "LSGEMU_REPLAY_TIME_SKIP_MODE": "stateful",
                        "LSGEMU_TARGETED_DIRECT_ROOT_FOCUS": 1,
                        "IGNORED_NON_LSGEMU": "no",
                    }
                },
                sort_keys=True,
            )
        )
        old_env = {
            key: os.environ.get(key)
            for key in (
                "LSGEMU_REPLAY_TIME_SKIP_MODE",
                "LSGEMU_TARGETED_DIRECT_ROOT_FOCUS",
                "IGNORED_NON_LSGEMU",
                "LSGEMU_RUN_PROFILE_FILE",
                "LSGEMU_RUN_PROFILE_FILE_SHA256",
            )
        }
        try:
            loaded = load_run_profile(profile_path)
            applied = apply_run_profile_environment(loaded)
            collected = collect_run_profile(firmware_path="/tmp/fw.elf", cli_args={"time": 1})
            equivalent = dict(collected)
            equivalent["created_at"] = "2099-01-01T00:00:00+0000"
            equivalent["argv"] = ["different", "invocation"]
            equivalent["environment"] = dict(collected.get("environment") or {})
            equivalent["environment"]["LSGEMU_ATTEMPT_ID"] = "different-attempt"
            equivalent["environment"]["LSGEMU_ATTEMPT_DIR"] = "/different/output"
            equivalent["environment"]["LSGEMU_SNAPSHOT_STORAGE_DIR"] = (
                "/different/snapshot/output"
            )
            storage_mode_changed = dict(collected)
            storage_mode_changed["environment"] = dict(
                collected.get("environment") or {}
            )
            storage_mode_changed["environment"]["LSGEMU_SNAPSHOT_DISK_BACKING"] = (
                "0"
                if storage_mode_changed["environment"].get(
                    "LSGEMU_SNAPSHOT_DISK_BACKING"
                )
                == "1"
                else "1"
            )
            checks = {
                "loaded_path_ok": Path(loaded["profile_file"]) == profile_path.resolve(),
                "hash_stable_ok": profile_hash(collected) == collected["profile_hash"],
                "hash_ignores_invocation_ok": profile_hash(equivalent) == collected["profile_hash"],
                "hash_keeps_storage_mode_ok": profile_hash(storage_mode_changed) != collected["profile_hash"],
                "volatile_fields_removed_ok": "created_at" not in profile_hash_payload(collected) and "argv" not in profile_hash_payload(collected),
                "lsgemu_env_applied_ok": os.environ.get("LSGEMU_REPLAY_TIME_SKIP_MODE") == "stateful",
                "non_lsgemu_ignored_ok": os.environ.get("IGNORED_NON_LSGEMU") != "no",
                "profile_file_recorded_ok": collected.get("input_profile_file") == str(profile_path.resolve()),
                "applied_keys_ok": "LSGEMU_TARGETED_DIRECT_ROOT_FOCUS" in applied,
            }
            print(json.dumps({"loaded": loaded, "applied": applied, "checks": checks}, indent=2, ensure_ascii=False))
            return 0 if all(checks.values()) else 1
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
