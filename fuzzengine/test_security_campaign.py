#!/usr/bin/env python3
"""Regression tests for the post-simulation security evidence layer."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import stat
import tempfile
import textwrap
from argparse import Namespace

from security_analysis import (
    MODEL_ARTIFACT,
    REACHABLE_BUG_NEEDS_IMPACT,
    TOOLING_FAILURE,
    build_security_report,
    classify_security_observation,
    collect_campaign_findings,
    extract_attack_surface,
    security_report_markdown,
)
from security_campaign import run_security_campaign


def _simulation_report() -> dict[str, object]:
    return {
        "firmware": "/tmp/demo.elf",
        "firmware_sha256": "fw-hash",
        "valid_covered_bbs": 12,
        "valid_total_bbs": 20,
        "valid_coverage_rate": 60.0,
        "strict_real_entry_replayable": True,
        "coverage_evidence_summary": {"e0": 12, "e1": 4, "e2": 2, "e3": 1},
        "phase_metadata": {
            "stream_input_replay": {
                "debug_records": [{
                    "input_read_hit_count": 2,
                    "stream_input_payload_write_count": 1,
                    "stream_input_summary_event_count": 1,
                    "input_read_hits": [{
                        "pc": "0x08001001",
                        "address": "0x20000020",
                        "size": 1,
                        "payload_offset": 3,
                        "source": "uart",
                    }],
                    "sink_call_events": [{
                        "symbol": "memcpy",
                        "pc": "0x08001020",
                        "dest": "0x20000040",
                        "src": "0x20000020",
                        "length": 64,
                        "boundary_verdict": "copy_exceeds_destination",
                    }],
                }],
            },
            "contextual_isr": {"contexts_selected": 2},
        },
        "mmio_semantic_summary": {
            "observed_reads": 4,
            "observed_writes": 2,
            "transaction_signatures": 1,
        },
        "causal_context_summary": {
            "enabled_irqs": [5],
            "pending_irqs": [5],
            "event_counts": {"irq_delivery": 1},
        },
        "counterfactual_only_coverage_bbs": 2,
        "exploratory_only_covered_bbs": 2,
        "context_recovery_only_covered_bbs": 1,
    }


def _crash_report(**updates: object) -> dict[str, object]:
    result = {
        "category": "firmware_crash",
        "classification": "invalid_memory_write",
        "source_layer": "runtime_monitor",
        "trigger_source": "external_input",
        "trigger_confidence": "high",
        "evidence_level": "strong_runtime_candidate",
        "metadata": {
            "record_summary": {
                "input_read_hit_count": 1,
                "stream_input_payload_write_count": 1,
            }
        },
    }
    result.update(updates)
    return result


def main() -> int:
    report = _simulation_report()
    surface = extract_attack_surface(report)
    kinds = set(surface["surface_kinds"])
    assert "external_input" in kinds
    assert "memory_or_effect_sink" in kinds
    assert "peripheral_environment" in kinds
    assert "interrupt_or_task_event" in kinds
    assert surface["input_sites"][0]["payload_offset"] == 3

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        finding_dir = root / "findings" / "firmware_crash" / "bucket_a"
        finding_dir.mkdir(parents=True)
        payload = {
            "seed": {"seed_id": "id_000001", "sha256": "seed-hash"},
            "crash_report": _crash_report(bucket_key="bucket_a"),
        }
        (finding_dir / "finding.json").write_text(json.dumps(payload), encoding="utf-8")
        replay_dir = root / "replays" / "bucket_a"
        replay_dir.mkdir(parents=True)
        (replay_dir / "id_000001_replay.json").write_text(json.dumps({
            "bucket": "bucket_a",
            "stable_bucket": True,
            "replays": [{"crash_reports": [{"bucket_key": "bucket_a"}]}] * 3,
        }), encoding="utf-8")
        findings = collect_campaign_findings(root)
        assert len(findings) == 1
        assessment = findings[0]["assessment"]
        assert assessment["classification"] == REACHABLE_BUG_NEEDS_IMPACT
        assert assessment["strict_confirmation"] is False
        assert assessment["stable_replay"] is True

        security = build_security_report(
            report,
            simulation_report_path="/tmp/simulation.json",
            work_dir=root,
            campaign_summary={"crash_buckets": {"bucket_a": 1}},
            campaign_ran=True,
        )
        markdown = security_report_markdown(security)
        assert "reachable_bug_needs_impact" in markdown
        assert security["summary"]["strict_confirmed_count"] == 0

        # Exercise the complete wrapper with a fake emulator. This checks that
        # the report identity gate, profile creation, campaign, and security
        # rollup use the same artifact contract without starting real firmware.
        firmware = root / "demo.elf"
        firmware.write_bytes(b"synthetic firmware")
        firmware_hash = hashlib.sha256(firmware.read_bytes()).hexdigest()
        simulation_path = root / "simulation.json"
        simulation_payload = _simulation_report()
        simulation_payload["firmware"] = str(firmware)
        simulation_payload["firmware_sha256"] = firmware_hash
        simulation_path.write_text(json.dumps(simulation_payload), encoding="utf-8")
        fake = root / "fake_lsgemu.py"
        fake.write_text(textwrap.dedent(
            """
            import argparse, json, os
            from pathlib import Path
            parser = argparse.ArgumentParser()
            parser.add_argument("firmware")
            parser.add_argument("--output-dir", required=True)
            args, _ = parser.parse_known_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            seed = os.environ.get("LSGEMU_STREAM_EXTRA_SEEDS_HEX", "")
            record = {
                "stop_reason": "Invalid memory write (UC_ERR_WRITE_UNMAPPED)",
                "registers": {"pc": "0x08001000", "lr": "0x08002000", "sp": "0x20001000"},
                "input_read_hit_count": 1,
                "stream_input_payload_write_count": 1,
                "last_unmapped_access": {"access": "write_unmapped", "address": "0x41414140", "size": 4},
            }
            report = {
                "firmware": args.firmware,
                "firmware_sha256": %s,
                "covered_bbs": 1,
                "valid_covered_bbs": 1,
                "valid_total_bbs": 4,
                "coverage_rate": 25.0,
                "valid_coverage_rate": 25.0,
                "covered_bb_list": ["0x08000000"],
                "phase_metadata": {"stream_input_replay": {"debug_records": [record]}},
            }
            path = out / (Path(args.firmware).stem + "_interleaved_report.json")
            path.write_text(json.dumps(report), encoding="utf-8")
            (out / (Path(args.firmware).stem + "_coverage_progress.jsonl")).write_text(json.dumps(report) + "\\n", encoding="utf-8")
            """ % json.dumps(firmware_hash)
        ), encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        args = Namespace(
            simulation_report=str(simulation_path),
            firmware=str(firmware),
            work_dir=str(root / "wrapped_campaign"),
            profile=None,
            seed=[],
            iterations=1,
            deterministic_cases_per_seed=0,
            random_seed=0,
            time_minutes=0.01,
            timeout_seconds=10.0,
            lsgemu_path=str(fake),
            python="python3",
            mode="interleaved",
            run_profile=None,
            crash_config=None,
            dictionary=[],
            extra_arg=[],
            replay_crashes=1,
            analyze_only=False,
            allow_identity_mismatch=False,
        )
        wrapped = run_security_campaign(args)
        assert wrapped["summary"]["reachable_bug_candidate_count"] == 1
        assert Path(wrapped["output_json"]).exists()

    model = classify_security_observation({
        "category": "model_or_harness_artifact",
        "classification": "unmapped_mmio_read",
        "source_layer": "runtime_monitor",
    })
    assert model["classification"] == MODEL_ARTIFACT

    tooling = classify_security_observation({
        "category": "tooling_or_runtime_failure",
        "classification": "native_process_crash",
        "source_layer": "process",
    })
    assert tooling["classification"] == TOOLING_FAILURE

    # A supervised child is not automatically a native failure.  Timeouts and
    # guest observations may have process provenance while retaining their
    # original security evidence class.
    process_hang = classify_security_observation({
        "category": "hang",
        "classification": "hang_or_timeout",
        "source_layer": "process",
        "trigger_source": "external_input",
        "metadata": {"record_summary": {"input_read_hit_count": 1}},
    })
    assert process_hang["classification"] == REACHABLE_BUG_NEEDS_IMPACT
    assert process_hang["classification"] != TOOLING_FAILURE

    process_guest_crash = classify_security_observation({
        "category": "firmware_crash",
        "classification": "invalid_memory_write",
        "source_layer": "process",
        "trigger_source": "external_input",
        "metadata": {"record_summary": {"input_read_hit_count": 1}},
    })
    assert process_guest_crash["classification"] == REACHABLE_BUG_NEEDS_IMPACT

    process_native_signal = classify_security_observation({
        "category": "unknown",
        "classification": "native_process_crash",
        "source_layer": "process",
    })
    assert process_native_signal["classification"] == TOOLING_FAILURE
    print("security evidence layer regression: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
