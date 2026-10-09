#!/usr/bin/env python3
"""Regression tests for Cortex-M system-control MMIO handling."""

from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.intelligent_mmio_inferencer import IntelligentMMIOInferencer
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from lsgemu.register_tracer.register_tracer import RegisterTracer


def main() -> int:
    static_bbs = {
        0x24DC: [
            {"address": 0x24DC, "mnemonic": "LDRB", "operands": "r3, [r2]", "size": 2},
            {"address": 0x24DE, "mnemonic": "TST", "operands": "r3, #0x10", "size": 2},
            {"address": 0x24E0, "mnemonic": "BEQ", "operands": "0x000024dc", "size": 2},
        ],
    }
    inferencer = IntelligentMMIOInferencer()
    inferred_reads = inferencer._extract_mmio_reads(
        [0x24DC],
        static_bbs,
        [(0x24DC, 0xE000E400, True, 0)],
    )
    checks = {
        "inferencer_accepts_system_mmio": bool(inferred_reads) and inferred_reads[0][0] == 0xE000E400,
        "handler_range_accepts_system_mmio": EnhancedMMIOHandler._is_mmio_address(0xE000E400),
        "tracer_range_accepts_system_mmio": RegisterTracer._is_mmio_address(0xE000E400),
        "non_mmio_rejected": not IntelligentMMIOInferencer._is_mmio_address(0x20000000),
    }
    print(json.dumps({"reads": inferred_reads, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
