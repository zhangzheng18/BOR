"""Outcome and evidence accounting for scheduler stages.

The component owns bookkeeping only.  In particular, it delegates validity
to the runner's existing static-BB oracle and never inserts a BB into coverage
on its own.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Set

from ..path_naturalization import EVIDENCE_E0, EVIDENCE_E1, EVIDENCE_E2, EVIDENCE_E3
from ..evidence_contract import (
    EVIDENCE_SCHEMA,
    DIAGNOSTIC_REPLAY,
    VALIDATED_REPLAY,
    build_execution_evidence_record,
    classify_phase,
    compact_record_is_verified_validated,
    coerce_bool,
    execution_intervention_reasons,
    has_execution_facts,
    is_compact_evidence_record,
)
from ..replay_diagnostics import (
    build_scheduler_feedback,
    summarize_branch_mmio_root_causes,
)
from ..runner_common import _env_flag, _env_int


_EVIDENCE_CLASSES = {EVIDENCE_E0, EVIDENCE_E1, EVIDENCE_E2, EVIDENCE_E3}

# r37 P3: positive evidence that a phase actually ran child replays, even when
# no per-child execution record survived to the ledger.  These are runtime
# counters/collections produced by the stage itself, never configuration
# labels: `counterfactual_control` describes how a stage was *scheduled*, so a
# stage that was armed but ran zero tasks must stay an evidence gap.
_AGGREGATE_EXECUTION_EVIDENCE_KEYS = (
    "attempt_records",
    "directions_replayed",
    "branches_considered",
    "control_replays",
)
_AGGREGATE_EXECUTION_EVIDENCE_SUFFIXES = (
    "_tasks_run",
    "_hit_tasks",
    "_attempts",
    "_replayed",
    "_replays",
)
_AGGREGATE_EXECUTION_EVIDENCE_LIMIT = 8


def _aggregate_execution_evidence(metadata: Mapping[str, Any]) -> list[str]:
    """Return the runtime counters proving this phase's children executed.

    Deliberately conservative: only collections that are non-empty and numeric
    counters that are strictly positive count.  A missing or zero counter means
    "no evidence", which keeps the phase an evidence gap rather than promoting
    it on configuration alone.
    """
    found: list[str] = []

    def positive(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return int(value) > 0
        return False

    for key in _AGGREGATE_EXECUTION_EVIDENCE_KEYS:
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                found.append(key)
        elif positive(value):
            found.append(key)
    for key, value in metadata.items():
        name = str(key)
        if not any(name.endswith(suffix) for suffix in _AGGREGATE_EXECUTION_EVIDENCE_SUFFIXES):
            continue
        if positive(value) and name not in found:
            found.append(name)
    return found[:_AGGREGATE_EXECUTION_EVIDENCE_LIMIT]


class OutcomeFeedback:
    """Centralize phase results and emulator feedback without changing policy."""

    component_name = "outcome_feedback"

    _UNCLASSIFIED = "unclassified"

    @staticmethod
    def _as_list(value: object) -> list[object]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    @staticmethod
    def _has_execution_fact(item: object) -> bool:
        if not isinstance(item, Mapping):
            return False
        if is_compact_evidence_record(item):
            return coerce_bool(item.get("execution_facts_available"), False)
        nested = item.get("run_result")
        if isinstance(nested, Mapping):
            return has_execution_facts(nested)
        return has_execution_facts(item)

    @staticmethod
    def _normalize_compact_record(
        item: Mapping[str, object],
        *,
        phase: str,
        execution_id: str,
    ) -> dict[str, object]:
        """Copy a canonical child record without trusting unverified fields."""
        record = dict(item)
        record.setdefault("schema", EVIDENCE_SCHEMA)
        record.setdefault("phase", str(phase))
        record.setdefault("execution_id", str(execution_id))
        record.setdefault("covered_bbs", [])
        record.setdefault("reasons", [])
        facts_available = coerce_bool(
            record.get("execution_facts_available"), False
        )
        telemetry_complete = coerce_bool(
            record.get("telemetry_complete"), False
        )
        status = str(record.get("status") or "unclassified")
        if status == VALIDATED_REPLAY and not compact_record_is_verified_validated(
            record
        ):
            status = "unclassified"
            record["status"] = status
            record["reasons"] = list(record.get("reasons", []) or []) + [
                "compact_record_not_verified"
            ]
        if status not in {VALIDATED_REPLAY, DIAGNOSTIC_REPLAY, "unclassified"}:
            record["status"] = "unclassified"
            record["reasons"] = list(record.get("reasons", []) or []) + [
                "unknown_compact_status"
            ]
        return record

    def _execution_records_for_phase(
        self,
        name: str,
        valid: Set[int],
        metadata: Mapping[str, object],
        invocation_id: str,
    ) -> tuple[list[dict[str, object]], bool]:
        """Extract compact per-execution facts without retaining raw payloads."""
        raw_records: list[object] = []
        source = ""
        for key in ("execution_records", "child_execution_records", "replay_records"):
            candidate = self._as_list(metadata.get(key))
            if candidate:
                raw_records = candidate
                source = key
                break
        child_summary = metadata.get("child_execution_summary")
        if not raw_records and isinstance(child_summary, Mapping):
            # Several replay kernels already place the per-child records in
            # the aggregate returned by ``_child_execution_summary``.  Read
            # that nested list before falling back to aggregate counters; the
            # latter cannot attribute a BB to a particular execution.
            for key in ("child_execution_records", "execution_records", "replay_records"):
                candidate = self._as_list(child_summary.get(key))
                if candidate:
                    raw_records = candidate
                    source = f"child_execution_summary.{key}"
                    break
        if not raw_records:
            # Debug lists are bounded in several replay kernels.  Use them only
            # when they actually contain execution facts; scheduler summaries
            # without a run result must not masquerade as a witness.
            for key in ("debug_records", "candidate_debug_records", "materialized_records"):
                candidate = [
                    item
                    for item in self._as_list(metadata.get(key))
                    if self._has_execution_fact(item)
                ]
                if candidate:
                    raw_records = candidate
                    source = key
                    break

        records: list[dict[str, object]] = []
        phase_context = dict(metadata)
        for index, item in enumerate(raw_records):
            item_map = item if isinstance(item, Mapping) else {}
            nested = item_map.get("run_result")
            run_result = nested if isinstance(nested, Mapping) else item
            if is_compact_evidence_record(item_map) and not isinstance(nested, Mapping):
                normalized = self._normalize_compact_record(
                    item_map,
                    phase=name,
                    execution_id=f"{invocation_id}.child-{index}",
                )
                if isinstance(item, dict):
                    item.update(normalized)
                    record = item
                else:
                    record = normalized
            elif isinstance(run_result, Mapping) and has_execution_facts(run_result):
                item_context = dict(phase_context)
                item_context.update(dict(item_map))
                record = build_execution_evidence_record(
                    phase_name=name,
                    run_result=run_result,
                    covered_bbs=(
                        item_map.get("covered_bb_list")
                        if item_map.get("covered_bb_list") is not None
                        else item_map.get("covered_bbs")
                    ),
                    metadata=item_context,
                    execution_id=f"{invocation_id}.child-{index}",
                )
            else:
                record = {
                    "schema": EVIDENCE_SCHEMA,
                    "phase": str(name),
                    "execution_id": f"{invocation_id}.child-{index}",
                    "status": self._UNCLASSIFIED,
                    "reasons": ["missing_run_result"],
                    "covered_bbs": [],
                    "telemetry_complete": False,
                }
            record["source"] = "child"
            record["source_field"] = source
            records.append(record)

        if records:
            expected = 0
            if isinstance(child_summary, Mapping):
                try:
                    expected = max(0, int(child_summary.get("expected_child_replay_count", 0) or 0))
                except (TypeError, ValueError, OverflowError):
                    expected = 0
            summary_truncated = bool(
                isinstance(child_summary, Mapping)
                and coerce_bool(
                    child_summary.get("execution_records_truncated"), False
                )
            )
            if expected and len(records) < expected:
                return records, True
            return records, summary_truncated

        run_result = metadata.get("run_result")
        if isinstance(run_result, Mapping) and has_execution_facts(run_result):
            return [
                {
                    **build_execution_evidence_record(
                        phase_name=name,
                        run_result=run_result,
                        covered_bbs=valid,
                        metadata=metadata,
                        execution_id=invocation_id,
                    ),
                    "source": "phase",
                    "coverage_scope": "phase",
                }
            ], False

        child_summary = metadata.get("child_execution_summary")
        if isinstance(child_summary, Mapping):
            expected = 0
            completed = 0
            try:
                expected = max(
                    0,
                    int(child_summary.get("expected_child_replay_count", 0) or 0),
                )
                completed = max(
                    0,
                    int(
                        child_summary.get(
                            "child_completed_with_result_count", 0
                        )
                        or 0
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                pass
            # Aggregate counters have no BB-to-child attribution.  Preserve
            # them for audit, but never attach the phase coverage to a
            # synthetic validated record.
            aggregate_reasons = ["missing_child_execution_records"]
            if expected == 0:
                aggregate_reasons = ["no_child_execution_attempt"]
            elif completed < expected:
                aggregate_reasons.append("incomplete_child_execution_telemetry")
            return [
                {
                    "schema": EVIDENCE_SCHEMA,
                    "phase": str(name),
                    "execution_id": invocation_id,
                    "status": DIAGNOSTIC_REPLAY if expected else self._UNCLASSIFIED,
                    "reasons": aggregate_reasons,
                    "covered_bbs": [],
                    "claimed_covered_bbs": sorted(valid),
                    "telemetry_complete": False,
                    "execution_facts_available": False,
                    "source": "phase_aggregate",
                    "coverage_scope": "none",
                    "aggregate_child_summary": dict(child_summary),
                }
            ], bool(expected and completed < expected)

        # No per-child records were retained.  When the phase metadata still
        # carries positive runtime evidence that its children executed (task
        # counters, forced-direction counters, attempt records), the phase is a
        # diagnostic aggregate, not an evidence gap: an unclassified record
        # would claim nothing ran, which is false for a reservoir/branch round.
        # It is deliberately *not* promoted to validated — without per-child
        # records there is no BB-to-witness attribution, so the coverage stays
        # a claim, never a validated witness.
        aggregate_evidence = _aggregate_execution_evidence(metadata)
        if aggregate_evidence:
            return [
                {
                    "schema": EVIDENCE_SCHEMA,
                    "phase": str(name),
                    "execution_id": invocation_id,
                    "status": DIAGNOSTIC_REPLAY,
                    "reasons": [
                        "missing_child_execution_records",
                        *aggregate_evidence,
                    ],
                    "covered_bbs": sorted(valid),
                    "telemetry_complete": False,
                    "execution_facts_available": False,
                    "source": "phase_aggregate",
                    "coverage_scope": "phase",
                }
            ], False

        # Nothing at all ran, or the stage left no runtime trace.  Keep the
        # observed BBs explicit but do not call them diagnostic: absence of
        # telemetry is an evidence gap.
        return [
            {
                "schema": EVIDENCE_SCHEMA,
                "phase": str(name),
                "execution_id": invocation_id,
                "status": self._UNCLASSIFIED,
                "reasons": ["missing_phase_execution_summary"],
                "covered_bbs": sorted(valid),
                "telemetry_complete": False,
                "execution_facts_available": False,
                "source": "phase",
                "coverage_scope": "phase",
            }
        ], False

    def _append_execution_records(
        self,
        name: str,
        valid: Set[int],
        metadata: Mapping[str, object],
        invocation_id: str,
    ) -> tuple[list[dict[str, object]], bool]:
        runner = self.runner
        records, truncated = self._execution_records_for_phase(
            name,
            valid,
            metadata,
            invocation_id,
        )
        # A bounded list protects long campaigns.  The per-BB sets below are
        # retained independently, so dropping verbose records cannot turn a
        # previously observed witness into a false validated result.
        try:
            limit = max(1, _env_int("LSGEMU_EXECUTION_EVIDENCE_RECORD_LIMIT", 16384))
        except (TypeError, ValueError, OverflowError):
            limit = 16384
        existing = getattr(runner, "phase_execution_records", None)
        if not isinstance(existing, list):
            existing = []
            runner.phase_execution_records = existing
        validated_bbs = getattr(runner, "execution_evidence_validated_bbs", None)
        diagnostic_bbs = getattr(runner, "execution_evidence_diagnostic_bbs", None)
        unclassified_bbs = getattr(runner, "execution_evidence_unclassified_bbs", None)
        if not isinstance(validated_bbs, set):
            validated_bbs = set()
            runner.execution_evidence_validated_bbs = validated_bbs
        if not isinstance(diagnostic_bbs, set):
            diagnostic_bbs = set()
            runner.execution_evidence_diagnostic_bbs = diagnostic_bbs
        if not isinstance(unclassified_bbs, set):
            unclassified_bbs = set()
            runner.execution_evidence_unclassified_bbs = unclassified_bbs

        represented: Set[int] = set()
        for record in records:
            try:
                bbs = runner.validate_coverage(record.get("covered_bbs", []))
            except Exception:
                bbs = set()
            record["covered_bbs"] = sorted(bbs)
            represented.update(bbs)
            status = str(record.get("status") or self._UNCLASSIFIED)
            # Re-check at the long-lived ledger boundary.  A producer can add
            # pair/provenance fields after initial materialization; conversely,
            # a stale compact record must never regain validated status merely
            # because it survived in a bounded diagnostic list.
            if status == VALIDATED_REPLAY and not compact_record_is_verified_validated(
                record
            ):
                status = self._UNCLASSIFIED
                record["status"] = status
                record["reasons"] = list(record.get("reasons", []) or []) + [
                    "compact_record_not_verified"
                ]
            if len(existing) < limit:
                existing.append(dict(record))
            else:
                account_record = getattr(
                    runner,
                    "_account_execution_evidence_record",
                    None,
                )
                if callable(account_record):
                    status = str(
                        account_record(
                            record,
                            status,
                            bbs,
                            archived=True,
                        )
                        or status
                    )
                    record["status"] = status
                elif status == VALIDATED_REPLAY:
                    validated_bbs.update(bbs)
                elif status == DIAGNOSTIC_REPLAY:
                    diagnostic_bbs.update(bbs)
                else:
                    unclassified_bbs.update(bbs)
                runner.execution_evidence_records_dropped = int(
                    getattr(runner, "execution_evidence_records_dropped", 0) or 0
                ) + 1

        if truncated:
            missing = set(valid) - represented
            if missing:
                account_record = getattr(
                    runner,
                    "_account_execution_evidence_record",
                    None,
                )
                if callable(account_record):
                    account_record(
                        {
                            "execution_id": f"{invocation_id}.unattributed",
                            "covered_bbs": sorted(missing),
                        },
                        self._UNCLASSIFIED,
                        missing,
                        archived=True,
                    )
                else:
                    unclassified_bbs.update(missing)
        return records, truncated

    @staticmethod
    def _compact_history_metadata(
        name: str,
        invocation_id: str,
        recorded_metadata: Mapping[str, object],
        records: list[Mapping[str, object]],
        truncated: bool,
    ) -> dict[str, object]:
        statuses = Counter(str(item.get("status") or "unclassified") for item in records)
        return {
            "phase": str(name),
            "invocation_id": str(invocation_id),
            "covered_bbs": int(recorded_metadata.get("covered_bbs", 0) or 0),
            "new_bbs": int(recorded_metadata.get("new_bbs", 0) or 0),
            "coverage_counted": bool(recorded_metadata.get("coverage_counted", True)),
            "canonical_evidence_status": str(
                recorded_metadata.get("canonical_evidence_status") or ""
            ),
            "canonical_evidence_reasons": list(
                recorded_metadata.get("canonical_evidence_reasons", []) or []
            ),
            "execution_record_count": len(records),
            "execution_record_status_counts": dict(statuses),
            "execution_records_truncated": bool(truncated),
        }

    def __init__(self, runner: Any):
        self.runner = runner

    def record_phase(
        self,
        name: str,
        covered: Iterable[int],
        *,
        count_coverage: bool = True,
        **metadata: object,
    ) -> Set[int]:
        runner = self.runner
        valid = runner.validate_coverage(covered)
        lock_factory = getattr(runner, "_runner_state_lock", None)
        lock = lock_factory() if callable(lock_factory) else _NullLock()
        with lock:
            global_coverage = getattr(runner, "global_coverage", set())
            new_coverage = valid - global_coverage
            if count_coverage:
                global_coverage.update(valid)
                evidence_class = str(metadata.get("evidence_class") or "")
                if evidence_class in _EVIDENCE_CLASSES:
                    runner._record_coverage_evidence(evidence_class, valid)

            phase_coverage = getattr(runner, "phase_coverage", None)
            if phase_coverage is None:
                phase_coverage = {}
                runner.phase_coverage = phase_coverage
            phase_metadata = getattr(runner, "phase_metadata", None)
            if phase_metadata is None:
                phase_metadata = {}
                runner.phase_metadata = phase_metadata
            lifecycle_summary = {}
            lifecycle_summary_error = None
            lifecycle = getattr(runner, "_temp_lifecycle_phase_summary", None)
            if callable(lifecycle):
                try:
                    lifecycle_summary = lifecycle(str(name))
                except Exception as exc:
                    lifecycle_summary = {}
                    lifecycle_summary_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                    }
            phase_coverage[name] = valid
            recorded_metadata = {
                "covered_bbs": len(valid),
                "new_bbs": len(new_coverage),
                "coverage_counted": bool(count_coverage),
                "temp_emulator_lifecycle": lifecycle_summary,
                **metadata,
            }
            canonical_status, canonical_reasons = classify_phase(
                name,
                recorded_metadata,
                legacy_evidence=recorded_metadata.get("evidence_class"),
            )
            recorded_metadata["canonical_evidence_status"] = canonical_status
            recorded_metadata["canonical_evidence_reasons"] = canonical_reasons
            if isinstance(recorded_metadata.get("run_result"), dict):
                recorded_metadata["execution_intervention_reasons"] = (
                    execution_intervention_reasons(recorded_metadata["run_result"])
                )
            # Keep the values importable by report consumers without requiring
            # callers to know the implementation constants.
            recorded_metadata.setdefault(
                "canonical_evidence_contract",
                "validated_replay_or_diagnostic_replay",
            )
            if lifecycle_summary_error:
                recorded_metadata["temp_emulator_lifecycle_error"] = lifecycle_summary_error
            # Keep a unique invocation history while preserving the historical
            # latest-value maps used by schedulers and older report readers.
            history = getattr(runner, "phase_metadata_history", None)
            if not isinstance(history, dict):
                history = {}
                runner.phase_metadata_history = history
            invocation_id = f"{str(name)}#{len(history.get(str(name), []) or []) + 1}"
            execution_records, execution_records_truncated = self._append_execution_records(
                str(name),
                valid,
                recorded_metadata,
                invocation_id,
            )
            recorded_metadata["execution_record_count"] = len(execution_records)
            recorded_metadata["execution_record_ids"] = [
                str(item.get("execution_id") or "")
                for item in execution_records[:64]
            ]
            recorded_metadata["execution_records_truncated"] = bool(
                execution_records_truncated
            )
            phase_metadata[name] = recorded_metadata
            history.setdefault(str(name), []).append(
                self._compact_history_metadata(
                    str(name),
                    invocation_id,
                    recorded_metadata,
                    execution_records,
                    execution_records_truncated,
                )
            )
            if _env_flag("LSGEMU_RECORD_PHASE_BB_LISTS", "0"):
                limit = max(0, _env_int("LSGEMU_PHASE_BB_LIST_LIMIT", 50000))
                sorted_valid = sorted(valid)
                sorted_new = sorted(new_coverage)
                if limit <= 0 or len(sorted_valid) <= limit:
                    phase_metadata[name]["covered_bb_list"] = sorted_valid
                else:
                    phase_metadata[name]["covered_bb_list_sample"] = sorted_valid[:limit]
                    phase_metadata[name]["covered_bb_list_truncated"] = True
                    phase_metadata[name]["covered_bb_list_total"] = len(sorted_valid)
                if limit <= 0 or len(sorted_new) <= limit:
                    phase_metadata[name]["new_bb_list"] = sorted_new
                else:
                    phase_metadata[name]["new_bb_list_sample"] = sorted_new[:limit]
                    phase_metadata[name]["new_bb_list_truncated"] = True
                    phase_metadata[name]["new_bb_list_total"] = len(sorted_new)
            # A phase result invalidates the derived scheduler feedback cache;
            # it does not itself create a new scheduling action.
            runner.scheduler_feedback = {}
            if hasattr(runner, "scheduler_feedback_cached_stage"):
                runner.scheduler_feedback_cached_stage = ""
        return valid

    def harvest_emulator_feedback(self, emulator: Any) -> bool:
        """Merge observations produced by one emulator into runner state."""

        runner = self.runner
        causal_stats = getattr(runner, "causal_input_feedback_stats", None)
        if not isinstance(causal_stats, Counter):
            causal_stats = Counter()
            runner.causal_input_feedback_stats = causal_stats
        if not bool(getattr(emulator, "_lsgemu_causal_feedback_harvested", False)):
            causal_events = list(getattr(emulator, "causal_input_events", []) or [])
            causal_stats["emulators"] += 1
            causal_stats["recorded_events"] += len(causal_events)
            causal_stats["total_event_count"] += int(
                getattr(emulator, "causal_input_event_count", 0) or 0
            )
            causal_stats["pre_read_snapshots"] += len(
                list(getattr(emulator, "causal_input_snapshots", []) or [])
            )
            causal_stats["unretained_event_ids"] += max(
                0,
                int(getattr(emulator, "causal_input_event_count", 0) or 0)
                - len(causal_events),
            )
            for event in causal_events:
                if not isinstance(event, dict):
                    continue
                causal_stats[f"kind.{str(event.get('kind') or 'unknown')}"] += 1
                causal_stats[f"source.{str(event.get('source') or 'unknown')}"] += 1
            try:
                emulator._lsgemu_causal_feedback_harvested = True
            except Exception:
                pass

        try:
            runner.emulator.external_memory_input_addresses.update(
                int(address) & 0xFFFFFFFF
                for address in (
                    getattr(emulator, "external_memory_input_addresses", set()) or set()
                )
            )
        except Exception:
            pass

        for address, count in getattr(emulator, "unproven_dynamic_code_targets", Counter()).items():
            runner.unproven_dynamic_code_targets[int(address) & 0xFFFFFFFF] += int(count or 0)
        discovery = getattr(runner, "obligation_discovery", None)
        if discovery is None:
            ensure_components = getattr(runner, "_ensure_scheduler_components", None)
            if callable(ensure_components):
                ensure_components()
            discovery = getattr(runner, "obligation_discovery", None)
        if discovery is None:
            return False
        return bool(discovery.harvest_emulator_feedback(emulator))

    def scheduler_feedback_summary(self) -> dict[str, object]:
        """Derive the next-action feedback from completed stage outcomes."""

        runner = self.runner
        if not getattr(runner, "semantic_obligation_enabled", True):
            feedback: dict[str, object] = {
                "schema": "lsgemu.scheduler_feedback.v1",
                "disabled": True,
                "disabled_reason": "semantic_obligation_ablation",
                "has_actionable_feedback": False,
                "recommended_next_stages": [],
                "reason_counts": {},
                "actions": [],
                "target_bbs_by_action": {},
            }
            runner.scheduler_feedback = feedback
            return feedback

        branch_root_causes = summarize_branch_mmio_root_causes(runner.phase_metadata)
        uncovered_summary = runner.uncovered_coverage_summary()
        frontier_diag = runner._frontier_successor_candidate_diagnostics()
        direct_call_diag = runner._direct_call_candidate_diagnostics()
        dispatch_case_status = runner.dispatch_case_status_summary()
        feedback = build_scheduler_feedback(
            branch_mmio_root_causes=branch_root_causes,
            uncovered_summary=uncovered_summary,
            frontier_diagnostics=frontier_diag,
            direct_call_diagnostics=direct_call_diag,
            dispatch_case_status=dispatch_case_status,
        )
        feedback["branch_root_causes"] = branch_root_causes
        feedback["uncovered_summary"] = uncovered_summary
        feedback["frontier_diagnostics"] = frontier_diag
        feedback["direct_call_diagnostics"] = direct_call_diag
        feedback["dispatch_case_status"] = dispatch_case_status
        runner.scheduler_feedback = feedback
        return feedback


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
