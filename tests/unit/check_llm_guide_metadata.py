#!/usr/bin/env python3
"""Smoke-test LLMGuide metadata summarization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.llm_guide.llm_guide import LLMGuide


def main() -> int:
    guide = LLMGuide({}, use_llm=False)
    records = [
        {
            "method": "llm",
            "root_cause": "wrong_read_pc",
            "recommended_action": "retry_closer_snapshot",
        },
        {
            "method": "llm_rejected",
            "llm_root_cause": "wrong_read_pc",
            "llm_recommended_action": "retry_closer_snapshot",
        },
        {
            "method": "rules",
        },
    ]
    summary = guide._summarize_inference_records(
        records,
        branch_pc=0x08001000,
        condition="BNE",
        target_direction=True,
        mmio_addr=0x40021004,
        value=0x8,
    )
    checks = {
        "method_ok": summary["inference_method"] == "llm",
        "root_cause_ok": summary["root_cause_counts"]["wrong_read_pc"] == 2,
        "action_ok": summary["recommended_action_counts"]["retry_closer_snapshot"] == 2,
        "latest_root_cause_ok": summary["latest_root_cause"] == "wrong_read_pc",
    }
    print(json.dumps({"summary": summary, "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
