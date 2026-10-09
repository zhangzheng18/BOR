#!/usr/bin/env python3
"""
Minimal live test for qwen-plus integration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.runtime_bootstrap import bootstrap_runtime_dependencies
from lsgemu.llm_json_utils import call_llm_json, create_openai_compatible_client
from lsgemu.llm_guide.llm_guide import LLMGuide


def load_llm_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "LLM.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    report = bootstrap_runtime_dependencies()
    config = load_llm_config()

    llm_cfg = config["llm"]
    client, model, backend = create_openai_compatible_client(
        llm_cfg,
        default_model=llm_cfg.get("model", "qwen-plus"),
    )
    if client is None:
        raise RuntimeError(f"无法初始化qwen客户端: backend={backend}")

    direct = call_llm_json(
        client=client,
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "只返回 JSON，不要其他内容："
                    '{"status":"ok","provider":"qwen-plus","reason":"connectivity-test"}'
                ),
            }
        ],
        max_tokens=120,
        temperature=0,
    )
    direct_text = direct.choices[0].message.content.strip()

    cases = [
        {
            "name": "tst_bne_taken",
            "static_bbs": {
                0x08005818: [
                    {"address": 0x08005818, "mnemonic": "LDR", "operands": "R3, [R2, #0x10]"},
                    {"address": 0x0800581C, "mnemonic": "TST", "operands": "R3, #0x2000000"},
                    {"address": 0x08005820, "mnemonic": "BNE", "operands": "0x08005824"},
                ]
            },
            "branch_pc": 0x08005820,
            "branch_condition": "BNE",
            "target_direction": True,
            "mmio_addr": 0x40021000,
            "expected": 0x02000000,
        },
        {
            "name": "cmp_beq_taken",
            "static_bbs": {
                0x08001230: [
                    {"address": 0x08001230, "mnemonic": "LDR", "operands": "R0, [R1, #0x04]"},
                    {"address": 0x08001234, "mnemonic": "CMP", "operands": "R0, #0x20"},
                    {"address": 0x08001238, "mnemonic": "BEQ", "operands": "0x08001260"},
                ]
            },
            "branch_pc": 0x08001238,
            "branch_condition": "BEQ",
            "target_direction": True,
            "mmio_addr": 0x40010004,
            "expected": 0x00000020,
        },
    ]

    case_results = []
    all_ok = True
    for case in cases:
        guide = LLMGuide(
            static_bbs=case["static_bbs"],
            use_llm=True,
            llm_config=config,
        )
        inferred = guide.infer_constraint(
            branch_pc=case["branch_pc"],
            branch_condition=case["branch_condition"],
            target_direction=case["target_direction"],
            mmio_addr=case["mmio_addr"],
        )
        last_history = guide.get_inference_history()[-1] if guide.get_inference_history() else {}
        case_ok = (
            inferred is not None
            and inferred == case["expected"]
            and last_history.get("method") in {"rules", "llm"}
        )
        all_ok = all_ok and case_ok
        case_results.append({
            "name": case["name"],
            "value": f"0x{inferred:08x}" if inferred is not None else None,
            "expected": f"0x{case['expected']:08x}",
            "method": last_history.get("method"),
            "validation": last_history.get("local_validation"),
            "analysis": last_history.get("llm_analysis"),
            "ok": case_ok,
        })

    llm_forced_cases = []
    for case in cases:
        guide = LLMGuide(
            static_bbs=case["static_bbs"],
            use_llm=True,
            llm_config=config,
        )
        inferred = guide._infer_with_llm(
            branch_pc=case["branch_pc"],
            branch_condition=case["branch_condition"],
            target_direction=case["target_direction"],
            mmio_addr=case["mmio_addr"],
        )
        last_history = guide.get_inference_history()[-1] if guide.get_inference_history() else {}
        case_ok = (
            inferred is not None
            and inferred == case["expected"]
            and last_history.get("method") == "llm"
            and last_history.get("local_validation") is True
        )
        all_ok = all_ok and case_ok
        llm_forced_cases.append({
            "name": case["name"],
            "value": f"0x{inferred:08x}" if inferred is not None else None,
            "expected": f"0x{case['expected']:08x}",
            "method": last_history.get("method"),
            "validation": last_history.get("local_validation"),
            "analysis": last_history.get("llm_analysis"),
            "why": last_history.get("llm_why_value_satisfies_branch"),
            "ok": case_ok,
        })

    parser_samples = [
        {
            "name": "strict_json",
            "text": '{"analysis":"ok","mmio_value":"0x20","confidence":0.9,"why_value_satisfies_branch":"eq"}',
            "expected": 0x20,
        },
        {
            "name": "yaml_like_hex",
            "text": '{analysis: "ok", mmio_value: 0x20, confidence: 0.9, why_value_satisfies_branch: "eq"}',
            "expected": 0x20,
        },
        {
            "name": "markdown_fenced",
            "text": '```json\n{"analysis":"ok","mmio_value":"0x2000000","confidence":0.8,"why_value_satisfies_branch":"bit set"}\n```',
            "expected": 0x02000000,
        },
        {
            "name": "assignment_style",
            "text": 'mmio_value=0x2000000, analysis=bit set, why_value_satisfies_branch=branch non-zero',
            "expected": 0x02000000,
        },
    ]
    parser_results = []
    parser_guide = LLMGuide(static_bbs={}, use_llm=False)
    for sample in parser_samples:
        parsed = parser_guide._parse_json_response(sample["text"])
        parsed_value = parser_guide._parse_mmio_value(parsed.get("mmio_value"))
        ok = parsed_value == sample["expected"]
        all_ok = all_ok and ok
        parser_results.append({
            "name": sample["name"],
            "parsed": parsed,
            "parsed_value": f"0x{parsed_value:08x}" if parsed_value is not None else None,
            "expected": f"0x{sample['expected']:08x}",
            "ok": ok,
        })

    output = {
        "runtime": report,
        "llm_backend": backend,
        "direct_response": direct_text,
        "guide_cases": case_results,
        "forced_llm_cases": llm_forced_cases,
        "parser_cases": parser_results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
