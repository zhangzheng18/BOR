#!/usr/bin/env python3
"""Run LSGEmu without context-aware event replay.

Usage:
    python experiments/lsgemu_no_context_event_replay.py firmware.elf --time 1440

This contribution-level ablation answers whether event context is necessary.
It disables contextual ISR, stream-input event, and RTOS thread-entry replay
while preserving non-event control-flow replay.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.ablation_entrypoint import run_ablation


def main() -> None:
    run_ablation("no_context_event_replay")


if __name__ == "__main__":
    main()
