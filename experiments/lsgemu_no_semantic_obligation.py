#!/usr/bin/env python3
"""Run LSGEmu without semantic-obligation discovery (ablation v2).

Usage:
    python experiments/lsgemu_no_semantic_obligation.py firmware.elf --time 1440

This contribution-level ablation disables stagnation-to-obligation conversion,
semantic types, target scoring, scheduler feedback, and obligation-driven
candidate generation. It retains a structural CFG/branch/MMIO fallback
scheduler so the emulator remains executable.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.ablation_entrypoint import run_ablation


def main() -> None:
    run_ablation("no_semantic_obligation_v2")


if __name__ == "__main__":
    main()
