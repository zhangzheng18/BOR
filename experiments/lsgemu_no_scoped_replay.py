#!/usr/bin/env python3
"""Run LSGEmu without evidence-scoped replay.

Usage:
    python experiments/lsgemu_no_scoped_replay.py firmware.elf --time 1440

This contribution-level ablation answers whether scoped replay is necessary.
It keeps a reset-entry, task-local address-global MMIO fallback available, but
disables branch snapshots/provenance, learned PC/occurrence-scoped constraints,
branch occurrence replay, and target-scoped frontier/dispatch replay.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.ablation_entrypoint import run_ablation


def main() -> None:
    run_ablation("no_scoped_replay_v2")


if __name__ == "__main__":
    main()
