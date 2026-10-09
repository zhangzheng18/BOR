#!/usr/bin/env python3
"""Evidence-preserving post-processing for defensive MCU security tests.

This module deliberately sits after LSGEmu and the existing crash campaign.  It
does not change emulation semantics, add coverage, or claim that an emulator
observation is a hardware-validated vulnerability.  Its job is to turn the
available report and replay artifacts into an attack-surface inventory and a
conservative finding assessment.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Iterator, List, Optional, Tuple


CONFIRMED_VULNERABILITY = "confirmed_vulnerability"
CONFIRMED_SECURITY_WEAKNESS = "confirmed_security_weakness"
REACHABLE_BUG_NEEDS_IMPACT = "reachable_bug_needs_impact"
REACHABLE_BUT_EXPECTED = "reachable_but_expected_behavior"
UNREACHABLE_CURRENT_MODEL = "unreachable_under_current_model"
MODEL_ARTIFACT = "model_or_harness_artifact"
DUPLICATE_VARIANT = "duplicate_or_variant"
TOOLING_FAILURE = "tooling_or_runtime_failure"

FIRMWARE_CRASH = "firmware_crash"
HANG = "hang"
NORMAL = "normal"

EXTERNAL_TRIGGERS = {
    "external_input",
    "external_input_plus_peripheral_state",
}

SINK_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("memory_copy", ("memcpy", "memmove", "strcpy", "strncpy", "strcat", "strncat")),
    ("formatted_output", ("sprintf", "snprintf", "vsprintf", "vsnprintf", "printf", "scanf", "sscanf")),
    ("flash_update", ("flash", "erase", "writebytes", "boot", "update", "firmware")),
    ("indirect_control", ("callback", "dispatch", "function_pointer", "blx", "bx")),
    ("privileged_or_state_change", ("nvic", "control", "primask", "basepri", "mpu", "security", "auth")),
)


def _int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, str):
            return int(value, 0)
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _phase_map(report: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Merge the two report phase containers without changing their values."""
    phases: Dict[str, Dict[str, object]] = {}
    for container_name in ("phase_metadata", "phases"):
        container = report.get(container_name)
        if not isinstance(container, dict):
            continue
        for name, phase in container.items():
            if isinstance(phase, dict):
                phases.setdefault(str(name), {}).update(phase)
    return phases


def _phase_records(phase_name: str, phase: Dict[str, object]) -> Iterator[Dict[str, object]]:
    """Yield report records that may carry input or sink evidence."""
    run_result = phase.get("run_result")
    if isinstance(run_result, dict):
        record = dict(run_result)
        record.setdefault("phase", phase_name)
        yield record

    for key in (
        "debug_records",
        "candidate_debug_records",
        "materialized_records",
        "records",
        "replay_records",
    ):
        values = phase.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            nested = record.pop("run_result", None)
            if isinstance(nested, dict):
                merged = dict(nested)
                merged.update(record)
                record = merged
            record.setdefault("phase", phase_name)
            yield record


def _iter_report_records(report: Dict[str, object]) -> Iterator[Dict[str, object]]:
    for phase_name, phase in _phase_map(report).items():
        yield from _phase_records(phase_name, phase)


def _sink_family(symbol: object) -> Optional[str]:
    name = str(symbol or "").strip().lower()
    if not name:
        return None
    for family, terms in SINK_FAMILIES:
        if any(term in name for term in terms):
            return family
    return "other_sink"


def _address(value: object) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"0x{parsed & 0xFFFFFFFF:08x}"


def _record_input_sites(record: Dict[str, object], limit: int = 64) -> List[Dict[str, object]]:
    sites: List[Dict[str, object]] = []
    for item in record.get("input_read_hits") or []:
        if not isinstance(item, dict):
            continue
        site = {
            "pc": _address(item.get("pc")),
            "address": _address(item.get("address")),
            "size": _int(item.get("size")),
            "payload_offset": _int(item.get("payload_offset"), -1),
            "source": item.get("source"),
            "profile_kind": item.get("profile_kind"),
        }
        if site not in sites:
            sites.append(site)
        if len(sites) >= limit:
            break
    return sites


def _compact_sink_event(event: Dict[str, object], phase: str) -> Dict[str, object]:
    symbol = str(event.get("symbol") or "")
    compact: Dict[str, object] = {
        "phase": phase,
        "symbol": symbol,
        "family": _sink_family(symbol),
        "pc": _address(event.get("pc")),
        "boundary_verdict": event.get("boundary_verdict"),
    }
    for key in (
        "dest",
        "src",
        "length",
        "r0",
        "r1",
        "r2",
        "r3",
        "copy_length_bytes",
        "copy_overflow_bytes",
        "dest_object",
        "dest_object_size",
        "dest_remaining_bytes",
        "source_matches_seed_payload",
    ):
        if key in event:
            value = event.get(key)
            compact[key] = _address(value) if key in {"dest", "src"} else value
    return compact


def extract_attack_surface(
    report: Dict[str, object],
    *,
    report_path: Optional[object] = None,
) -> Dict[str, object]:
    """Extract observed, model-dependent security surfaces from one report.

    The output is an inventory, not a vulnerability verdict.  It intentionally
    keeps only bounded samples and never copies complete seed payloads into the
    report.
    """
    phases = _phase_map(report)
    phase_names = list(phases)
    input_sites: List[Dict[str, object]] = []
    sink_events: List[Dict[str, object]] = []
    sink_symbols: Counter[str] = Counter()
    sink_families: Counter[str] = Counter()
    input_reads = 0
    payload_writes = 0
    stream_summary_events = 0
    watch_memory_events = 0
    watch_pc_events = 0
    stream_phase_names: List[str] = []
    irq_phase_names: List[str] = []

    for phase_name, phase in phases.items():
        lowered = phase_name.lower()
        if "stream_input" in lowered:
            stream_phase_names.append(phase_name)
        if any(token in lowered for token in ("isr", "irq", "interrupt", "rtos_thread")):
            irq_phase_names.append(phase_name)
        for record in _phase_records(phase_name, phase):
            input_reads += _int(record.get("input_read_hit_count"))
            payload_writes += _int(record.get("stream_input_payload_write_count"))
            stream_summary_events += _int(record.get("stream_input_summary_event_count"))
            watch_memory_events += len(record.get("watch_memory_events") or [])
            watch_pc_events += len(record.get("watch_pc_events") or [])
            for site in _record_input_sites(record):
                if site not in input_sites and len(input_sites) < 64:
                    input_sites.append(site)
            for raw_event in record.get("sink_call_events") or []:
                if not isinstance(raw_event, dict):
                    continue
                event = _compact_sink_event(raw_event, phase_name)
                sink_events.append(event)
                symbol = str(event.get("symbol") or "")
                family = str(event.get("family") or "other_sink")
                if symbol:
                    sink_symbols[symbol] += 1
                sink_families[family] += 1
                if len(sink_events) >= 128:
                    break
            if len(sink_events) >= 128:
                break

    mmio_summary = report.get("mmio_semantic_summary")
    mmio_summary = dict(mmio_summary) if isinstance(mmio_summary, dict) else {}
    causal_summary = report.get("causal_context_summary")
    causal_summary = dict(causal_summary) if isinstance(causal_summary, dict) else {}
    crash_summary = report.get("crash_triage_summary")
    crash_summary = dict(crash_summary) if isinstance(crash_summary, dict) else {}
    indirect_targets = report.get("unproven_dynamic_code_targets")
    if isinstance(indirect_targets, dict):
        indirect_count = len(indirect_targets)
    elif isinstance(indirect_targets, list):
        indirect_count = len(indirect_targets)
    else:
        indirect_count = _int(report.get("unproven_dynamic_code_target_count"))

    surfaces: List[Dict[str, object]] = []
    if input_reads or payload_writes or stream_summary_events or stream_phase_names:
        confidence = "high" if input_reads or payload_writes else "medium"
        surfaces.append({
            "kind": "external_input",
            "confidence": confidence,
            "evidence": {
                "stream_phases": stream_phase_names[:32],
                "input_read_hits": input_reads,
                "payload_writes": payload_writes,
                "summary_events": stream_summary_events,
                "observed_sites": input_sites[:32],
            },
        })
    if sink_events:
        surfaces.append({
            "kind": "memory_or_effect_sink",
            "confidence": "high",
            "evidence": {
                "event_count": len(sink_events),
                "symbols": dict(sink_symbols.most_common(32)),
                "families": dict(sink_families),
                "samples": sink_events[:16],
            },
        })
    mmio_observed = (
        _int(mmio_summary.get("observed_reads"))
        + _int(mmio_summary.get("observed_writes"))
        + _int(mmio_summary.get("profile_hits"))
    )
    if mmio_observed or mmio_summary.get("transaction_signatures"):
        surfaces.append({
            "kind": "peripheral_environment",
            "confidence": "medium",
            "evidence": {
                "observed_operations": mmio_observed,
                "transaction_signatures": _int(mmio_summary.get("transaction_signatures")),
                "rule_hits": mmio_summary.get("rule_hits"),
                "transaction_states": mmio_summary.get("transaction_states"),
            },
        })
    irq_count = (
        len(causal_summary.get("enabled_irqs") or [])
        + len(causal_summary.get("pending_irqs") or [])
        + len(causal_summary.get("irq_stack") or [])
    )
    if irq_phase_names or irq_count or causal_summary.get("event_counts"):
        surfaces.append({
            "kind": "interrupt_or_task_event",
            "confidence": "medium",
            "evidence": {
                "phases": irq_phase_names[:32],
                "irq_state_entries": irq_count,
                "event_counts": causal_summary.get("event_counts"),
                "active_task": causal_summary.get("active_task"),
            },
        })
    if indirect_count or report.get("dynamic_successor_summary"):
        surfaces.append({
            "kind": "indirect_control_flow",
            "confidence": "medium",
            "evidence": {
                "unproven_dynamic_targets": indirect_count,
                "dynamic_successor_summary": report.get("dynamic_successor_summary"),
            },
        })

    coverage_evidence = report.get("coverage_evidence_summary")
    coverage_evidence = dict(coverage_evidence) if isinstance(coverage_evidence, dict) else {}
    model_dependencies = {
        "strict_real_entry_replayable": report.get("strict_real_entry_replayable"),
        "coverage_entry_derived": report.get("coverage_entry_derived"),
        "evidence_summary": coverage_evidence,
        "counterfactual_only_bbs": _int(report.get("counterfactual_only_coverage_bbs")),
        "context_recovery_only_bbs": _int(report.get("context_recovery_only_covered_bbs")),
        "forced_only_bbs": _int(report.get("exploratory_only_covered_bbs")),
        "unclassified_bbs": _int(report.get("unclassified_evidence_coverage_bbs")),
    }

    source_report = str(report_path) if report_path is not None else None
    report_digest = _sha256(Path(report_path)) if report_path is not None else None
    return {
        "schema": "lsgemu.security.attack_surface.v1",
        "source_report": source_report,
        "source_report_sha256": report_digest,
        "firmware": report.get("firmware") or report.get("analysis_firmware"),
        "firmware_sha256": report.get("firmware_sha256") or report.get("analysis_firmware_sha256"),
        "phase_count": len(phase_names),
        "phase_names": phase_names[:128],
        "surfaces": surfaces,
        "surface_kinds": [str(item.get("kind")) for item in surfaces],
        "input_sites": input_sites,
        "sink_symbols": dict(sink_symbols),
        "sink_families": dict(sink_families),
        "sink_event_samples": sink_events[:32],
        "model_dependencies": model_dependencies,
        "baseline_crash_summary": {
            "firmware_crashes": _int(crash_summary.get("firmware_crashes")),
            "hangs": _int(crash_summary.get("hangs")),
            "model_artifacts": _int(crash_summary.get("model_artifacts")),
            "tooling_failures": _int(crash_summary.get("tooling_failures")),
            "unique_buckets": _int(crash_summary.get("unique_buckets")),
        },
        "limits": [
            "The inventory contains observed emulator evidence, not a complete physical attack-surface proof.",
            "Input sites and sink samples are bounded and may omit events beyond report retention limits.",
            "MMIO, IRQ, forced-control, and reconstructed-context assumptions remain model-dependent.",
        ],
    }


def _record_summary(report: Dict[str, object]) -> Dict[str, object]:
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    summary = metadata.get("record_summary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    summary.update({
        key: report.get(key)
        for key in (
            "input_read_hit_count",
            "stream_input_payload_write_count",
            "stream_input_summary_event_count",
            "sink_call_event_count",
        )
        if key in report
    })
    return summary


def _input_bound(report: Dict[str, object]) -> Tuple[bool, Dict[str, object]]:
    trigger = str(report.get("trigger_source") or "")
    summary = _record_summary(report)
    evidence = report.get("trigger_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    input_evidence = evidence.get("external_input")
    input_evidence = input_evidence if isinstance(input_evidence, dict) else {}
    reads = _int(summary.get("input_read_hit_count"))
    writes = _int(summary.get("stream_input_payload_write_count"))
    observed = _int(summary.get("stream_input_summary_event_count"))
    bound = bool(trigger in EXTERNAL_TRIGGERS or reads > 0 or writes > 0)
    return bound, {
        "trigger_source": trigger or "unknown",
        "input_read_hit_count": reads,
        "payload_write_count": writes,
        "summary_event_count": observed,
        "detector_input_evidence": input_evidence,
        "seed_present_without_consumption": bool(input_evidence.get("seed_present_without_consumption")),
    }


def _is_explicit_native_process_failure(
    crash_report: Dict[str, object],
    *,
    category: str,
    classification: str,
    source_layer: str,
) -> bool:
    """Return true only for evidence of a host/native process failure.

    ``LSGEmuSeedRunner`` labels every supervised child result with
    ``source_layer=process``.  That label describes the observation boundary,
    not its cause: a child can time out, emit a guest fault report, or exit
    normally.  Treating the layer itself as a native crash would discard the
    distinction this evidence layer is meant to preserve.
    """
    if category == TOOLING_FAILURE:
        return True
    if source_layer != "process":
        return False
    if classification in {"native_process_crash", "native_process_signal"}:
        return True

    # Keep this compatible with callers that pass a serialized process result
    # directly instead of the normalized ``native_process_crash`` class.
    metadata = crash_report.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    record_summary = metadata.get("record_summary")
    record_summary = record_summary if isinstance(record_summary, dict) else {}
    timed_out = bool(
        crash_report.get("timed_out")
        or metadata.get("timed_out")
        or record_summary.get("timed_out")
    )
    if timed_out:
        return False
    return_code = (
        crash_report.get("process_returncode")
        or metadata.get("process_returncode")
        or record_summary.get("process_returncode")
    )
    try:
        return int(return_code) < 0 and int(return_code) != -999
    except (TypeError, ValueError):
        return False


def classify_security_observation(
    crash_report: Dict[str, object],
    *,
    stable_replay: bool = False,
    replay_attempts: int = 0,
) -> Dict[str, object]:
    """Apply the conservative finding vocabulary to one crash report."""
    category = str(crash_report.get("category") or NORMAL)
    classification = str(crash_report.get("classification") or "unknown")
    source_layer = str(crash_report.get("source_layer") or "")
    bound, input_evidence = _input_bound(crash_report)
    missing: List[str] = []

    if _is_explicit_native_process_failure(
        crash_report,
        category=category,
        classification=classification,
        source_layer=source_layer,
    ):
        result_class = TOOLING_FAILURE
        validation_state = "host_failure"
    elif category == "model_or_harness_artifact":
        result_class = MODEL_ARTIFACT
        validation_state = "model_dependent"
    elif category == NORMAL:
        result_class = REACHABLE_BUT_EXPECTED
        validation_state = "no_security_signal"
    elif not bound:
        result_class = UNREACHABLE_CURRENT_MODEL
        validation_state = "no_controlled_input_binding"
    else:
        # A stable emulator crash is still only a reachable bug candidate until
        # root cause, impact, and physical/model feasibility are reviewed.
        result_class = REACHABLE_BUG_NEEDS_IMPACT
        validation_state = "stable_replay" if stable_replay else "replay_required"

    if not bound and category in {FIRMWARE_CRASH, HANG}:
        missing.append("controlled_input_consumption")
    if category in {FIRMWARE_CRASH, HANG} and not stable_replay:
        missing.append("deterministic_replay")
    if category in {FIRMWARE_CRASH, HANG}:
        missing.extend(["root_cause", "security_impact", "hardware_or_spec_validation"])
    if category == "tooling_or_runtime_failure":
        missing.append("fresh_child_process_reproduction")

    if "write" in classification or "overflow" in classification or "stack" in classification:
        impact_hypothesis = "memory_safety_candidate"
    elif "instruction" in classification or "pc_" in classification or "dynamic" in classification:
        impact_hypothesis = "control_flow_candidate"
    elif category == HANG:
        impact_hypothesis = "availability_candidate"
    elif "fault" in classification or "abort" in classification:
        impact_hypothesis = "fault_or_abort_candidate"
    else:
        impact_hypothesis = "unclassified_runtime_effect"

    return {
        "classification": result_class,
        "strict_confirmation": False,
        "validation_state": validation_state,
        "impact_hypothesis": impact_hypothesis,
        "input_bound": bound,
        "input_evidence": input_evidence,
        "stable_replay": bool(stable_replay),
        "replay_attempts": int(replay_attempts),
        "missing_confirmation": sorted(set(missing)),
        "source_category": category,
        "source_classification": classification,
        "source_layer": source_layer,
        "evidence_level": crash_report.get("evidence_level"),
        "trigger_confidence": crash_report.get("trigger_confidence"),
    }


def _replay_index(work_dir: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    replay_root = work_dir / "replays"
    if not replay_root.exists():
        return index
    for path in replay_root.glob("*/*.json"):
        payload = _read_json(path)
        if not payload:
            continue
        bucket = str(payload.get("bucket") or path.parent.name)
        replays = payload.get("replays")
        replays = replays if isinstance(replays, list) else []
        index[bucket] = {
            "stable_bucket": bool(payload.get("stable_bucket")),
            "replay_attempts": len(replays),
            "path": str(path),
        }
    return index


def collect_campaign_findings(
    work_dir: object,
    *,
    campaign_summary: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Group finding artifacts by crash bucket and attach replay evidence."""
    root = Path(work_dir)
    replay_index = _replay_index(root)
    grouped: Dict[str, Dict[str, object]] = {}
    for path in sorted((root / "findings").glob("**/*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        crash = payload.get("crash_report")
        if not isinstance(crash, dict):
            continue
        bucket = str(crash.get("bucket_key") or path.parent.name)
        item = grouped.setdefault(bucket, {
            "finding_id": f"bucket:{bucket}",
            "bucket_key": bucket,
            "observations": 0,
            "seed_ids": [],
            "seed_sha256": [],
            "categories": Counter(),
            "trigger_sources": Counter(),
            "representative": crash,
            "artifact_paths": [],
        })
        item["observations"] = int(item["observations"]) + 1
        seed = payload.get("seed")
        if isinstance(seed, dict):
            seed_id = str(seed.get("seed_id") or "")
            seed_hash = str(seed.get("sha256") or "")
            if seed_id and seed_id not in item["seed_ids"]:
                item["seed_ids"].append(seed_id)
            if seed_hash and seed_hash not in item["seed_sha256"]:
                item["seed_sha256"].append(seed_hash)
        item["categories"][str(crash.get("category") or NORMAL)] += 1
        item["trigger_sources"][str(crash.get("trigger_source") or "unknown")] += 1
        item["artifact_paths"].append(str(path))

    findings: List[Dict[str, object]] = []
    for bucket, raw in grouped.items():
        replay = replay_index.get(bucket, {})
        assessment = classify_security_observation(
            dict(raw["representative"]),
            stable_replay=bool(replay.get("stable_bucket")),
            replay_attempts=_int(replay.get("replay_attempts")),
        )
        findings.append({
            "finding_id": raw["finding_id"],
            "bucket_key": bucket,
            "observations": int(raw["observations"]),
            "seed_ids": list(raw["seed_ids"]),
            "seed_sha256": list(raw["seed_sha256"]),
            "categories": dict(raw["categories"]),
            "trigger_sources": dict(raw["trigger_sources"]),
            "assessment": assessment,
            "representative": raw["representative"],
            "replay": replay or {
                "stable_bucket": False,
                "replay_attempts": 0,
                "path": None,
            },
            "artifact_paths": list(raw["artifact_paths"]),
        })

    # A process can die before the finding JSON is durable. Preserve that fact
    # in the summary rather than manufacturing a finding without a PC/input.
    if campaign_summary:
        known = {str(item["bucket_key"]) for item in findings}
        for bucket in (campaign_summary.get("crash_buckets") or {}):
            bucket_text = str(bucket)
            if bucket_text in known:
                continue
            findings.append({
                "finding_id": f"unmaterialized:{bucket_text}",
                "bucket_key": bucket_text,
                "observations": _int((campaign_summary.get("crash_buckets") or {}).get(bucket)),
                "seed_ids": [],
                "seed_sha256": [],
                "categories": {},
                "trigger_sources": {},
                "assessment": {
                    "classification": TOOLING_FAILURE,
                    "strict_confirmation": False,
                    "validation_state": "missing_durable_finding_artifact",
                    "missing_confirmation": ["durable_guest_evidence", "deterministic_replay"],
                },
                "representative": None,
                "replay": replay_index.get(bucket_text, {}),
                "artifact_paths": [],
            })
    findings.sort(key=lambda item: str(item.get("finding_id")))
    return findings


def build_security_report(
    simulation_report: Dict[str, object],
    *,
    simulation_report_path: Optional[object] = None,
    work_dir: Optional[object] = None,
    campaign_summary: Optional[Dict[str, object]] = None,
    campaign_ran: bool = False,
) -> Dict[str, object]:
    findings = collect_campaign_findings(work_dir, campaign_summary=campaign_summary) if work_dir else []
    classifications = Counter(
        str((item.get("assessment") or {}).get("classification") or "unknown")
        for item in findings
    )
    stable = sum(
        1 for item in findings
        if bool((item.get("assessment") or {}).get("stable_replay"))
    )
    candidate_count = sum(
        1 for item in findings
        if str((item.get("assessment") or {}).get("classification")) == REACHABLE_BUG_NEEDS_IMPACT
    )
    if candidate_count:
        status = "candidates_require_root_cause_and_impact_validation"
    elif classifications.get(MODEL_ARTIFACT):
        status = "model_artifacts_only"
    elif classifications.get(TOOLING_FAILURE):
        status = "tooling_failures_or_missing_evidence"
    else:
        status = "no_security_candidate"

    return {
        "schema": "lsgemu.security.test.v1",
        "generated_at_epoch": time.time(),
        "simulation": {
            "report_path": str(simulation_report_path) if simulation_report_path is not None else None,
            "report_sha256": _sha256(Path(simulation_report_path)) if simulation_report_path is not None else None,
            "firmware": simulation_report.get("firmware") or simulation_report.get("analysis_firmware"),
            "firmware_sha256": simulation_report.get("firmware_sha256") or simulation_report.get("analysis_firmware_sha256"),
            "valid_covered_bbs": _int(simulation_report.get("valid_covered_bbs")),
            "valid_total_bbs": _int(simulation_report.get("valid_total_bbs")),
            "valid_coverage_rate": simulation_report.get("valid_coverage_rate"),
            "strict_real_entry_replayable": simulation_report.get("strict_real_entry_replayable"),
        },
        "campaign": campaign_summary or {
            "ran": False,
            "purpose": "post_simulation_security_assessment",
        },
        "campaign_ran": bool(campaign_ran),
        "attack_surface": extract_attack_surface(
            simulation_report,
            report_path=simulation_report_path,
        ),
        "findings": findings,
        "summary": {
            "status": status,
            "finding_count": len(findings),
            "stable_replay_count": stable,
            "reachable_bug_candidate_count": candidate_count,
            "classification_counts": dict(classifications),
            "strict_confirmed_count": 0,
        },
        "policy": {
            "coverage_is_not_a_vulnerability_oracle": True,
            "input_presence_is_not_input_consumption": True,
            "native_emulator_signal_is_not_a_firmware_crash": True,
            "forced_or_reconstructed_context_requires_separate_validation": True,
            "strict_confirmation_requires": [
                "stable_target_and_firmware_identity",
                "controlled_input_or_event_consumption",
                "root_cause",
                "security_impact",
                "deterministic_replay",
                "model_or_hardware_feasibility_review",
            ],
        },
        "limitations": [
            "The post-simulation profile guides replay but does not restore the original Unicorn snapshots.",
            "A generic byte mutator does not infer a complete protocol grammar or checksum relation.",
            "Findings remain candidates until the listed confirmation gates are satisfied.",
        ],
    }


def security_report_markdown(report: Dict[str, object]) -> str:
    summary = report.get("summary") or {}
    simulation = report.get("simulation") or {}
    surface = report.get("attack_surface") or {}
    lines = [
        "# LSGEmu Security Test Report",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Firmware: `{simulation.get('firmware')}`",
        f"- Firmware SHA-256: `{simulation.get('firmware_sha256')}`",
        f"- Simulation report: `{simulation.get('report_path')}`",
        f"- Valid BB coverage: `{simulation.get('valid_covered_bbs', 0)}/{simulation.get('valid_total_bbs', 0)}`",
        "",
        "## Evidence Policy",
        "",
        "This report distinguishes emulator observations from firmware and hardware claims. "
        "No finding is automatically marked as a confirmed vulnerability.",
        "",
        "## Observed Surfaces",
        "",
        "| Surface | Evidence | Confidence |",
        "| --- | ---: | --- |",
    ]
    for item in surface.get("surfaces") or []:
        evidence = item.get("evidence") or {}
        count = evidence.get("event_count") or evidence.get("input_read_hits") or evidence.get("observed_operations") or evidence.get("unproven_dynamic_targets") or 0
        lines.append(f"| `{item.get('kind')}` | {count} | `{item.get('confidence')}` |")
    lines.extend([
        "",
        "## Findings",
        "",
        "| Finding | Classification | Replay | Input bound | Missing confirmation |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in report.get("findings") or []:
        assessment = item.get("assessment") or {}
        missing = ", ".join(str(value) for value in assessment.get("missing_confirmation") or [])
        lines.append(
            f"| `{item.get('finding_id')}` | `{assessment.get('classification')}` | "
            f"`{assessment.get('stable_replay', False)}` | `{assessment.get('input_bound', False)}` | {missing} |"
        )
    lines.extend([
        "",
        "## Required Next Validation",
        "",
        "1. Reproduce the candidate from a clean child process with the same firmware hash and input hash.",
        "2. Check the first faulting instruction, registers, access address, and input-consumption trace.",
        "3. Separate missing MMIO/IRQ/model state from firmware logic and assess security impact.",
        "4. Keep emulator-only, forced-context, and reconstructed-event results separate from hardware claims.",
        "",
    ])
    return "\n".join(lines)


def load_report(path: object) -> Dict[str, object]:
    payload = _read_json(Path(path))
    if payload is None:
        raise ValueError(f"expected a JSON object: {path}")
    return payload
