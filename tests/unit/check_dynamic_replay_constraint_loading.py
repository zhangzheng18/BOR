#!/usr/bin/env python3
"""Regression test for dynamic-replay-only static constraint loading."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler


def main() -> int:
    payload = {
        "constraints": [
            {
                "type": "mmio",
                "address": "0x40023008",
                "value": "0x00000004",
                "read_pc": "0x08004008",
                "constraint_pc": "0x0800400a",
                "added_by": "static_constraint_analyzer",
                "speculation_level": "dynamic_replay_required",
                "coverage_credit_policy": "unicorn_execution_only",
            },
            {
                "type": "mmio",
                "address": "0x40021004",
                "value": "0x00000002",
                "read_pc": "0x08001008",
                "constraint_pc": "0x0800100a",
                "added_by": "static_constraint_analyzer",
                "speculation_level": "conservative_loop_exit",
            },
        ]
    }
    with tempfile.TemporaryDirectory(prefix="lsgemu_constraints_") as tmp:
        path = Path(tmp) / "constraints.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        baseline = StatefulMMIOHandler(
            constraint_json_path=str(path),
            branch_mmio_file_mode="none",
        )
        replay = StatefulMMIOHandler(
            constraint_json_path=str(path),
            branch_mmio_file_mode="scoped",
        )
        checks = {
            "baseline_does_not_load_dynamic_replay_required": (
                baseline._try_static_constraint(0x40023008, 0x08004008) is None
            ),
            "baseline_loads_conservative_loop_exit": (
                baseline._try_static_constraint(0x40021004, 0x08001008) == 0x2
            ),
            "replay_loads_dynamic_replay_required": (
                replay._try_static_constraint(0x40023008, 0x08004008) == 0x4
            ),
        }
    print(json.dumps({"checks": checks}, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
