#!/usr/bin/env python3
"""Shared launcher for LSGEmu ablation entrypoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from lsgemu import lsgemu as base_lsgemu
else:
    from . import lsgemu as base_lsgemu


ABLATION_ARGS = {
    "no_semantic_obligation_v2": ["--disable-semantic-obligation-stages"],
    "no_scoped_replay_v2": ["--disable-scoped-replay-stages"],
    "no_context_event_replay": ["--disable-context-event-replay-stages"],
}


def _infer_firmware_arg(argv: list[str]) -> str | None:
    """Best-effort firmware path discovery for default output naming only."""
    for token in argv:
        if token.startswith("-"):
            continue
        path = Path(token)
        if path.exists() or path.suffix.lower() in {".elf", ".bin", ".hex"}:
            return token
    return None


def run_ablation(ablation_name: str) -> None:
    if ablation_name not in ABLATION_ARGS:
        raise SystemExit(f"unknown ablation: {ablation_name}")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["interleaved", "simple"], default=None)
    known, unknown = parser.parse_known_args(sys.argv[1:])
    if known.mode == "simple":
        raise SystemExit("ablation entrypoints require --mode interleaved; --mode simple ignores interleaved ablation flags")

    forwarded = list(sys.argv[1:])
    firmware_arg = _infer_firmware_arg(unknown)
    if known.output_dir is None and firmware_arg:
        firmware_stem = Path(firmware_arg).stem
        output_root = Path(os.environ.get(
            "LSGEMU_RUN_OUTPUT_DIR",
            str(Path(base_lsgemu.RUNNER_PROJECT_ROOT) / "srcv4" / ".lsgemu_runs"),
        ))
        default_output = output_root / "ablation" / ablation_name / firmware_stem
        forwarded.extend(["--output-dir", str(default_output)])

    forwarded.extend(ABLATION_ARGS[ablation_name])
    sys.argv = [sys.argv[0], *forwarded]
    base_lsgemu.main()
