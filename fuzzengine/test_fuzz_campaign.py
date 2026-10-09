#!/usr/bin/env python3
"""Regression test for the lightweight fuzz campaign loop."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import textwrap

from campaign import FuzzCampaign, FuzzCampaignConfig
from lsgemu_runner import LSGEmuRunConfig, LSGEmuSeedRunner, best_summary_source
from profile import build_profile_from_report, write_profile


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fake = root / "fake_lsgemu.py"
        fake.write_text(textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            import argparse, json, os, signal
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("firmware")
            parser.add_argument("--time")
            parser.add_argument("--mode")
            parser.add_argument("--max-instructions")
            parser.add_argument("--output-dir", required=True)
            args, _ = parser.parse_known_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            seed_hex = os.environ.get("LSGEMU_STREAM_EXTRA_SEEDS_HEX", "")
            seed = bytes.fromhex(seed_hex) if seed_hex else b""
            covered = [0x08000000, 0x08000010]
            if b"A" in seed:
                covered.append(0x08000020)
            if b"B" in seed:
                covered.append(0x08000030)
            report = {
                "firmware": args.firmware,
                "covered_bbs": len(covered),
                "valid_covered_bbs": len(covered),
                "valid_total_bbs": 16,
                "coverage_rate": len(covered) / 16 * 100,
                "valid_coverage_rate": len(covered) / 16 * 100,
                "covered_bb_list": covered,
                "phases": {},
                "phase_metadata": {},
            }
            if b"!" in seed:
                report["phases"]["stream"] = {
                    "debug_records": [{
                        "stop_reason": "Invalid memory write (UC_ERR_WRITE_UNMAPPED)",
                        "registers": {"pc": "0x08000022", "lr": "0x08000010", "sp": "0x20000040"},
                        "input_read_hit_count": 1,
                        "stream_input_payload_write_count": 1,
                        "last_unmapped_access": {"access": "write_unmapped", "address": "0x41414140", "size": 4},
                    }]
                }
            if b"NATIVE" in seed:
                os.kill(os.getpid(), signal.SIGSEGV)
            stem = Path(args.firmware).stem
            (out / f"{stem}_interleaved_report.json").write_text(json.dumps(report), encoding="utf-8")
            (out / f"{stem}_coverage_progress.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")
            '''
        ), encoding="utf-8")
        fake.chmod(0o755)

        seed_dir = root / "seeds"
        seed_dir.mkdir()
        (seed_dir / "a.seed").write_bytes(b"A")
        (seed_dir / "crash.seed").write_bytes(b"!")

        long_report = root / "long_interleaved_report.json"
        long_report.write_text(json.dumps({
            "firmware": str(root / "dummy.elf"),
            "run_profile_hash": "profilehash",
            "strict_real_entry_replayable": True,
            "valid_covered_bbs": 12,
            "valid_total_bbs": 16,
            "valid_coverage_rate": 75.0,
            "phase_metadata": {
                "stream_input_replay": {
                    "profiles_discovered": 2,
                    "tasks_with_new_bbs": 1,
                    "debug_records": [{
                        "input_read_hit_count": 1,
                        "stream_input_payload_write_count": 1,
                        "new_bbs": 1,
                    }],
                },
                "contextual_isr": {
                    "learned_irq_candidate_count": 2,
                    "reservoir_interrupt_contexts": 3,
                },
            },
        }), encoding="utf-8")
        profile_path = root / "profile.json"
        profile = build_profile_from_report(long_report)
        write_profile(profile, profile_path)
        assert profile.stream_profile_score > 0

        # Profile options beginning with '--' must remain values of the outer
        # forwarding option when the real lsgemu.py parser sees the command.
        command = LSGEmuSeedRunner(LSGEmuRunConfig(
            firmware=str(root / "dummy.elf"),
            extra_args=tuple(profile.recommended_extra_args),
        ))._command(root / "command_output")
        assert "--interleaved-arg=--baseline-timeout-seconds" in command
        assert "--interleaved-arg=--disable-deadline-drain" in command
        assert any(
            left == "--interleaved-arg" and right == "30"
            for left, right in zip(command, command[1:])
        )

        summary_source, summary_source_name = best_summary_source(
            {"valid_covered_bbs": 1, "covered_bbs": 2},
            {"valid_covered_bbs": 3, "covered_bbs": 4},
        )
        assert summary_source_name == "progress"
        assert summary_source["valid_covered_bbs"] == 3

        work = root / "work"
        config = FuzzCampaignConfig(
            work_dir=str(work),
            firmware=str(root / "dummy.elf"),
            profile=str(profile_path),
            seed_paths=[str(seed_dir)],
            iterations=4,
            deterministic_cases_per_seed=0,
            lsgemu_path=str(fake),
            python="python3",
            timeout_seconds=10,
            replay_crashes=1,
        )
        summary = FuzzCampaign(config).run()
        assert summary["iterations"] == 4
        assert summary["corpus_size"] >= 2
        assert summary["coverage_context"]["global_covered_bbs"] >= 2
        assert summary["firmware_crash_candidates"] >= 1
        assert summary["unique_crash_buckets"] >= 1
        assert summary["trigger_sources"]["external_input"] >= 1
        assert summary["purpose"] == "crash_discovery_and_triage"
        assert summary["profile_context"]["enabled"]
        assert summary["profile_context"]["stream_profile_score"] > 0
        assert "coverage_context" in summary
        assert (work / "summary.json").exists()
        assert list((work / "findings").rglob("*.seed"))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
