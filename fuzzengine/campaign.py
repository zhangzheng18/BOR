#!/usr/bin/env python3
"""Crash-discovery LSGEmu fuzzing campaign loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

try:
    from .corpus import CorpusStore, SeedRecord
    from .crash_detector import FIRMWARE_CRASH, HANG, MODEL_ARTIFACT, NORMAL, TOOLING_FAILURE
    from .lsgemu_runner import LSGEmuRunConfig, LSGEmuSeedRunResult, LSGEmuSeedRunner
    from .mutator import ByteMutator, MutationConfig, parse_dictionary
    from .profile import load_profile
    from .utils import safe_fragment
except ImportError:
    from corpus import CorpusStore, SeedRecord
    from crash_detector import FIRMWARE_CRASH, HANG, MODEL_ARTIFACT, NORMAL, TOOLING_FAILURE
    from lsgemu_runner import LSGEmuRunConfig, LSGEmuSeedRunResult, LSGEmuSeedRunner
    from mutator import ByteMutator, MutationConfig, parse_dictionary
    from profile import load_profile
    from utils import safe_fragment

try:
    from lsgemu.artifact_io import append_jsonl, atomic_json_dump, atomic_write_bytes
except ImportError:  # pragma: no cover - supports running fuzzengine standalone.
    append_jsonl = atomic_json_dump = atomic_write_bytes = None


@dataclass
class FuzzCampaignConfig:
    work_dir: str
    firmware: str = ""
    seed_paths: Sequence[str] = field(default_factory=tuple)
    iterations: int = 100
    deterministic_cases_per_seed: int = 16
    random_seed: int = 0
    max_seed_size: int = 4096
    time_minutes: Optional[float] = None
    timeout_seconds: Optional[float] = None
    lsgemu_path: str = "lsgemu/lsgemu.py"
    python: Optional[str] = None
    mode: str = "interleaved"
    profile: Optional[str] = None
    run_profile: Optional[str] = None
    crash_config: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)
    dictionary: Sequence[str] = field(default_factory=tuple)
    replay_crashes: int = 1

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FuzzCampaignConfig":
        return cls(
            firmware=str(payload.get("firmware") or ""),
            work_dir=str(payload.get("work_dir") or "fuzzengine_runs/default"),
            seed_paths=[str(item) for item in payload.get("seed_paths") or []],
            iterations=int(payload.get("iterations") or 100),
            deterministic_cases_per_seed=int(payload.get("deterministic_cases_per_seed") or 16),
            random_seed=int(payload.get("random_seed") or 0),
            max_seed_size=int(payload.get("max_seed_size") or 4096),
            time_minutes=float(payload["time_minutes"]) if payload.get("time_minutes") is not None else None,
            timeout_seconds=float(payload["timeout_seconds"]) if payload.get("timeout_seconds") else None,
            lsgemu_path=str(payload.get("lsgemu_path") or "lsgemu/lsgemu.py"),
            python=str(payload["python"]) if payload.get("python") else None,
            mode=str(payload.get("mode") or "interleaved"),
            profile=str(payload["profile"]) if payload.get("profile") else None,
            run_profile=str(payload["run_profile"]) if payload.get("run_profile") else None,
            crash_config=str(payload["crash_config"]) if payload.get("crash_config") else None,
            extra_args=[str(item) for item in payload.get("extra_args") or []],
            env={str(key): str(value) for key, value in dict(payload.get("env") or {}).items()},
            dictionary=[str(item) for item in payload.get("dictionary") or []],
            replay_crashes=int(payload.get("replay_crashes") or 1),
        )


@dataclass
class CampaignIteration:
    iteration: int
    seed_id: str
    parent_id: Optional[str]
    generation: int
    seed_sha256: str
    seed_size: int
    new_bbs: int
    covered_bbs: int
    valid_covered_bbs: int
    valid_total_bbs: int
    crash_categories: Dict[str, int]
    crash_buckets: List[str]
    trigger_sources: Dict[str, int]
    elapsed_seconds: float
    result_path: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class FuzzCampaign:
    """Crash-discovery campaign around LSGEmu seed replay.

        Coverage is retained as replay context and to keep state-diverse seeds,
        but campaign success is measured by stable crash/hang/artifact buckets
        and trigger-source attribution rather than by coverage growth.
    """

    def __init__(self, config: FuzzCampaignConfig):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.work_dir / "results"
        self.findings_dir = self.work_dir / "findings"
        self.replays_dir = self.work_dir / "replays"
        for directory in (self.results_dir, self.findings_dir, self.replays_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.corpus = CorpusStore(self.work_dir / "corpus")
        self.mutator = ByteMutator(
            seed=config.random_seed,
            config=MutationConfig(
                max_size=config.max_seed_size,
                dictionary=parse_dictionary(config.dictionary),
            ),
        )
        self.profile = load_profile(config.profile) if config.profile else None
        self.firmware = config.firmware or (self.profile.firmware if self.profile else "")
        runner_extra_args = list(config.extra_args)
        runner_env = dict(config.env)
        runner_time_minutes = config.time_minutes
        runner_timeout_seconds = config.timeout_seconds
        profile_path = None
        profile_sha256 = None
        if self.profile:
            runner_extra_args = list(self.profile.recommended_extra_args) + runner_extra_args
            runner_env.update(self.profile.recommended_env)
            runner_time_minutes = (
                float(config.time_minutes)
                if config.time_minutes is not None
                else float(self.profile.recommended_time_minutes)
            )
            runner_timeout_seconds = (
                float(config.timeout_seconds)
                if config.timeout_seconds is not None
                else float(self.profile.recommended_timeout_seconds)
            )
            profile_path = self.profile.report_path
            profile_sha256 = self.profile.report_sha256
        self.runner = LSGEmuSeedRunner(LSGEmuRunConfig(
            firmware=self.firmware,
            lsgemu_path=config.lsgemu_path,
            python=config.python or "python3",
            time_minutes=float(runner_time_minutes if runner_time_minutes is not None else 2.0),
            mode=config.mode,
            output_root=str(self.work_dir / "lsgemu_runs"),
            run_profile=config.run_profile,
            crash_config=config.crash_config,
            extra_args=tuple(runner_extra_args),
            env=runner_env,
            timeout_seconds=runner_timeout_seconds,
            profile_path=profile_path,
            profile_sha256=profile_sha256,
        ))
        self.iterations: List[CampaignIteration] = []
        self.seen_crash_buckets: set[str] = set()
        self.summary_path = self.work_dir / "summary.json"
        self.progress_path = self.work_dir / "progress.jsonl"

    def initialize(self) -> None:
        imported = self.corpus.import_paths(self.config.seed_paths)
        if not imported and not self.corpus.records:
            imported = [self.corpus.add_seed(b"", origin="empty_seed")]
        if self.config.deterministic_cases_per_seed > 0:
            for record in list(imported):
                data = Path(record.path).read_bytes()
                for case in self.mutator.deterministic_cases(
                    data,
                    limit=self.config.deterministic_cases_per_seed,
                ):
                    self.corpus.add_seed(
                        case,
                        parent_id=record.seed_id,
                        generation=record.generation + 1,
                        origin="deterministic",
                    )
        self.corpus.save()

    def run(self) -> Dict[str, object]:
        self.initialize()
        started = time.time()
        for index in range(max(0, int(self.config.iterations))):
            record = self._next_record(index)
            result = self._run_record(record)
            iteration = self._record_result(index, record, result)
            self.iterations.append(iteration)
            self._write_json(self.results_dir / f"iter_{index:06d}_{record.seed_id}.json", result.to_dict())
            self._append_progress(iteration)
            self.corpus.save()
        summary = self._summary(time.time() - started)
        self._write_json(self.summary_path, summary)
        return summary

    def _next_record(self, index: int) -> SeedRecord:
        base = self.corpus.pick(0)
        if base.runs == 0:
            return base
        data = Path(base.path).read_bytes()
        mutated = self.mutator.mutate(data)
        return self.corpus.add_seed(
            mutated,
            parent_id=base.seed_id,
            generation=base.generation + 1,
            origin="mutated",
        )

    def _run_record(self, record: SeedRecord) -> LSGEmuSeedRunResult:
        data = Path(record.path).read_bytes()
        return self.runner.run_seed(record.seed_id, data, seed_sha256=record.sha256)

    def _record_result(
        self,
        index: int,
        record: SeedRecord,
        result: LSGEmuSeedRunResult,
    ) -> CampaignIteration:
        crash_categories = Counter()
        trigger_sources = Counter()
        crash_buckets: List[str] = []
        for report in result.crash_reports:
            category = str(report.get("category") or NORMAL)
            if category == NORMAL:
                continue
            crash_categories[category] += 1
            trigger_sources[str(report.get("trigger_source") or "unknown")] += 1
            bucket = str(report.get("bucket_key") or "")
            if bucket:
                crash_buckets.append(bucket)
                self._record_finding(record, result, report)
                if bucket not in self.seen_crash_buckets and self.config.replay_crashes > 0:
                    self.seen_crash_buckets.add(bucket)
                    self._replay_candidate(record, bucket)

        new_bbs = self.corpus.mark_result(
            record,
            covered_bbs=result.covered_bb_list,
            covered_bbs_count=result.covered_bbs,
            valid_covered_bbs=result.valid_covered_bbs,
            crash_bucket=crash_buckets[0] if crash_buckets else None,
        )
        result.new_bbs = new_bbs
        return CampaignIteration(
            iteration=index,
            seed_id=record.seed_id,
            parent_id=record.parent_id,
            generation=record.generation,
            seed_sha256=record.sha256,
            seed_size=record.size,
            new_bbs=new_bbs,
            covered_bbs=result.covered_bbs,
            valid_covered_bbs=result.valid_covered_bbs,
            valid_total_bbs=result.valid_total_bbs,
            crash_categories=dict(crash_categories),
            crash_buckets=sorted(set(crash_buckets)),
            trigger_sources=dict(trigger_sources),
            elapsed_seconds=result.elapsed_seconds,
            result_path=str(self.results_dir / f"iter_{index:06d}_{record.seed_id}.json"),
        )

    def _record_finding(self, record: SeedRecord, result: LSGEmuSeedRunResult, report: Dict[str, object]) -> None:
        category = str(report.get("category") or "unknown")
        bucket = str(report.get("bucket_key") or "unknown")
        bucket_dir = self.findings_dir / safe_fragment(category) / safe_fragment(bucket)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        stem = f"{stamp}_{record.seed_id}"
        self._write_json(bucket_dir / f"{stem}.json", {
            "schema": "lsgemu.fuzzengine.finding.v1",
            "seed": record.to_dict(),
            "run": result.to_dict(),
            "crash_report": report,
            "verification_note": (
                "Automated crash-discovery finding. Treat firmware_crash/hang as candidates "
                "until deterministic replay validates PC/LR/SP/access, trigger source, "
                "attacker/input binding, and model fidelity."
            ),
        })
        try:
            seed_destination = bucket_dir / f"{stem}.seed"
            if atomic_write_bytes is not None:
                atomic_write_bytes(seed_destination, Path(record.path).read_bytes(), durable=False)
            else:
                seed_destination.write_bytes(Path(record.path).read_bytes())
        except Exception:
            pass

    def _replay_candidate(self, record: SeedRecord, bucket: str) -> None:
        data = Path(record.path).read_bytes()
        replay_dir = self.replays_dir / safe_fragment(bucket)
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_results = []
        for index in range(max(0, int(self.config.replay_crashes))):
            replay_record = self.runner.run_seed(
                f"{record.seed_id}_replay{index}",
                data,
                seed_sha256=record.sha256,
            )
            replay_results.append(replay_record.to_dict())
        self._write_json(replay_dir / f"{record.seed_id}_replay.json", {
            "seed": record.to_dict(),
            "bucket": bucket,
            "replays": replay_results,
            "stable_bucket": self._stable_bucket(bucket, replay_results),
        })

    @staticmethod
    def _stable_bucket(bucket: str, replay_results: Sequence[Dict[str, object]]) -> bool:
        if not replay_results:
            return False
        for result in replay_results:
            reports = result.get("crash_reports") or []
            if not any(str(report.get("bucket_key") or "") == bucket for report in reports if isinstance(report, dict)):
                return False
        return True

    def _append_progress(self, iteration: CampaignIteration) -> None:
        record = iteration.to_dict()
        if append_jsonl is not None:
            append_jsonl(self.progress_path, record, sort_keys=True, ensure_ascii=False)
        else:
            with self.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _summary(self, elapsed_seconds: float) -> Dict[str, object]:
        category_counts = Counter()
        bucket_counts = Counter()
        trigger_counts = Counter()
        for item in self.iterations:
            for category, count in item.crash_categories.items():
                category_counts[category] += int(count)
            for trigger, count in item.trigger_sources.items():
                trigger_counts[trigger] += int(count)
            for bucket in item.crash_buckets:
                bucket_counts[bucket] += 1
        return {
            "schema": "lsgemu.fuzzengine.campaign.summary.v1",
            "purpose": "crash_discovery_and_triage",
            "firmware": self.firmware,
            "work_dir": str(self.work_dir),
            "profile_context": self._profile_context(),
            "iterations": len(self.iterations),
            "elapsed_seconds": elapsed_seconds,
            "corpus_size": len(self.corpus.records),
            "crash_categories": dict(category_counts),
            "trigger_sources": dict(trigger_counts),
            "unique_crash_buckets": len(bucket_counts),
            "crash_buckets": dict(bucket_counts),
            "firmware_crash_candidates": category_counts.get(FIRMWARE_CRASH, 0),
            "hang_candidates": category_counts.get(HANG, 0),
            "model_artifacts": category_counts.get(MODEL_ARTIFACT, 0),
            "tooling_failures": category_counts.get(TOOLING_FAILURE, 0),
            "coverage_context": {
                "global_covered_bbs": len(self.corpus.global_coverage),
                "max_valid_covered_bbs": max((item.valid_covered_bbs for item in self.iterations), default=0),
                "valid_total_bbs": max((item.valid_total_bbs for item in self.iterations), default=0),
                "note": "coverage is replay context only; crash buckets and trigger attribution are the objective",
            },
            "summary_path": str(self.summary_path),
            "progress_path": str(self.progress_path),
        }

    def _profile_context(self) -> Dict[str, object]:
        if not self.profile:
            return {
                "enabled": False,
                "note": "no long-run profile supplied; each seed uses the configured LSGEmu replay path",
            }
        return {
            "enabled": True,
            "report_path": self.profile.report_path,
            "report_sha256": self.profile.report_sha256,
            "source_run_profile_hash": self.profile.source_run_profile_hash,
            "strict_real_entry_replayable": self.profile.strict_real_entry_replayable,
            "valid_covered_bbs": self.profile.valid_covered_bbs,
            "valid_total_bbs": self.profile.valid_total_bbs,
            "stream_profile_score": self.profile.stream_profile_score,
            "mmio_environment_score": self.profile.mmio_environment_score,
            "irq_environment_score": self.profile.irq_environment_score,
            "recommended_time_minutes": self.profile.recommended_time_minutes,
            "recommended_timeout_seconds": self.profile.recommended_timeout_seconds,
            "notes": list(self.profile.notes),
        }

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if atomic_json_dump is not None:
            atomic_json_dump(payload, path, indent=2, ensure_ascii=False)
        else:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_campaign_config(path: object) -> FuzzCampaignConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"campaign config must be a JSON object: {config_path}")
    return FuzzCampaignConfig.from_dict(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight LSGEmu crash-discovery campaign")
    parser.add_argument("--config", help="JSON campaign config")
    parser.add_argument("--firmware")
    parser.add_argument("--work-dir")
    parser.add_argument("--seed", action="append", default=[], help="seed file or directory; repeatable")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--time-minutes", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--lsgemu-path", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--mode", default=None, choices=["interleaved", "simple"])
    parser.add_argument("--profile", help="fuzzengine profile JSON or LSGEmu *_interleaved_report.json from a long run")
    parser.add_argument("--run-profile")
    parser.add_argument("--crash-config")
    parser.add_argument("--dictionary", action="append", default=[])
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--replay-crashes", type=int)
    args = parser.parse_args(argv)

    if args.config:
        config = load_campaign_config(args.config)
    else:
        if not args.work_dir:
            parser.error("provide --config or --work-dir")
        if not args.firmware and not args.profile:
            parser.error("provide --firmware, or provide --profile with firmware metadata")
        config = FuzzCampaignConfig(firmware=args.firmware or "", work_dir=args.work_dir)

    updates = {
        "seed_paths": args.seed or None,
        "iterations": args.iterations,
        "time_minutes": args.time_minutes,
        "timeout_seconds": args.timeout_seconds,
        "lsgemu_path": args.lsgemu_path,
        "python": args.python,
        "mode": args.mode,
        "profile": args.profile,
        "run_profile": args.run_profile,
        "crash_config": args.crash_config,
        "dictionary": args.dictionary or None,
        "extra_args": args.extra_arg or None,
        "replay_crashes": args.replay_crashes,
    }
    for key, value in updates.items():
        if value is not None:
            setattr(config, key, value)

    summary = FuzzCampaign(config).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
