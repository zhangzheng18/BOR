#!/usr/bin/env python3
"""Evidence ledger and pure helpers for counterfactual path naturalization.

Forced replay is useful for discovering a branch-occurrence sequence, but it
does not provide an environment witness for that sequence.  This module keeps
the discovered sequence as an obligation, associates locally validated MMIO
facts with individual dynamic edges, and evaluates later force-free replays.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .causal_context import compare_branch_contexts
from .evidence_contract import (
    classify_execution,
    coerce_bool,
    execution_intervention_reasons,
    execution_telemetry_complete,
    has_execution_facts,
)
from .runner_strategy import state_equivalence_record
from .runner_models import (
    BranchConstraintCandidate,
    constraint_delivery_site_identity,
)


BranchKey = Tuple[int, int]
PathSignature = Tuple[Tuple[BranchKey, object], ...]
ObligationKey = Tuple[PathSignature, PathSignature]

EVIDENCE_E0 = "E0"
EVIDENCE_E1 = "E1"
EVIDENCE_E2 = "E2"
EVIDENCE_E3 = "E3"
EVIDENCE_UNKNOWN = "unknown"

_EVIDENCE_STRENGTH = {
    EVIDENCE_UNKNOWN: 0,
    EVIDENCE_E3: 1,
    EVIDENCE_E2: 2,
    EVIDENCE_E1: 3,
    EVIDENCE_E0: 4,
}


def normalize_evidence_class(value: object) -> str:
    evidence = str(value or EVIDENCE_UNKNOWN)
    if evidence not in _EVIDENCE_STRENGTH:
        return EVIDENCE_UNKNOWN
    return evidence


def derived_replay_evidence(
    source_evidence: object,
    *,
    forced_control: bool = False,
    execution_context_recovery: bool = False,
    consumed_environment_fact: bool = False,
) -> str:
    """Classify a replay without allowing weaker provenance to disappear."""
    source = normalize_evidence_class(source_evidence)
    if execution_context_recovery or source == EVIDENCE_E3:
        return EVIDENCE_E3
    if forced_control or source == EVIDENCE_E2:
        return EVIDENCE_E2
    if consumed_environment_fact and source in {EVIDENCE_E0, EVIDENCE_E1}:
        return EVIDENCE_E1
    if source in {EVIDENCE_E0, EVIDENCE_E1}:
        return source
    return EVIDENCE_UNKNOWN


def normalize_choice(choice: object) -> object:
    """Keep bool and integer choices distinct while making values hashable."""
    if isinstance(choice, bool):
        return bool(choice)
    if isinstance(choice, int):
        return int(choice)
    if isinstance(choice, (str, bytes, type(None))):
        return choice
    return str(choice)


def normalize_signature(signature: Iterable[Tuple[BranchKey, object]]) -> PathSignature:
    normalized: List[Tuple[BranchKey, object]] = []
    seen: Set[BranchKey] = set()
    for raw_key, raw_choice in signature or ():
        try:
            key = (int(raw_key[0]) & 0xFFFFFFFF, max(1, int(raw_key[1])))
        except (TypeError, ValueError, IndexError):
            continue
        if key in seen:
            # A dynamic occurrence is a unique decision point.  Keep the last
            # value without changing its original position.
            normalized = [(old_key, value) for old_key, value in normalized if old_key != key]
        seen.add(key)
        normalized.append((key, normalize_choice(raw_choice)))
    return tuple(normalized)


def choice_identity(choice: object) -> Tuple[str, object]:
    normalized = normalize_choice(choice)
    if isinstance(normalized, bool):
        return ("bool", normalized)
    if isinstance(normalized, int):
        return ("int", normalized)
    return (type(normalized).__name__, normalized)


def reason_category(reason: object) -> str:
    text = str(reason or "unknown")
    if text in {
        "promoted",
        "already_natural_control",
        "force_free_prefix_advanced",
        "force_free_discovery_prefix_advanced",
    }:
        return "validated"
    if "branch_occurrence_not_reached" in text:
        return "missing_prefix_state"
    if text.startswith("missing_observed_upstream_divergence"):
        return "missing_prefix_state"
    if text.startswith("missing_causal_fact"):
        return "missing_peripheral_or_input_fact"
    if text.startswith("fact_not_externalized"):
        return "non_external_state_candidate"
    if text.startswith("constraint_conflict") or text == "conflicting_constraint_values":
        return "transaction_or_temporal_state_conflict"
    if "wrong_natural_choice" in text:
        return "missing_predicate_state"
    if text == "candidate_not_consumed":
        return "candidate_read_not_observed"
    if text == "scope_mismatch":
        return "fact_scope_not_on_required_prefix"
    if text == "forced_control_observed_in_validation":
        return "force_free_validation_invariant_failed"
    if text == "replay_context_incompatible":
        return "execution_context_mismatch"
    if text in {
        "target_not_reached",
        "target_not_reached_after_complete_signature",
        "candidate_no_target_delta",
    }:
        return "downstream_state_or_budget_insufficient"
    return "other"


def serialize_signature(signature: PathSignature) -> List[Dict[str, object]]:
    return [
        {
            "branch": f"0x{int(key[0]) & 0xFFFFFFFF:08x}",
            "occurrence": int(key[1]),
            "choice_type": choice_identity(choice)[0],
            "choice": choice_identity(choice)[1],
        }
        for key, choice in normalize_signature(signature)
    ]


def _constraint_identity(candidate: BranchConstraintCandidate) -> Tuple[object, ...]:
    return constraint_delivery_site_identity(candidate)


@dataclass(frozen=True)
class ConstraintRefinement:
    """A proposed replacement at one already constrained input occurrence."""

    site_identity: Tuple[object, ...]
    previous: BranchConstraintCandidate
    replacement: BranchConstraintCandidate


def merge_constraint_sets(
    base: Iterable[BranchConstraintCandidate],
    added: Iterable[BranchConstraintCandidate],
) -> Tuple[Optional[Tuple[BranchConstraintCandidate, ...]], Optional[str]]:
    """Merge path-scoped facts, rejecting contradictory values at one read."""
    merged: Dict[Tuple[object, ...], BranchConstraintCandidate] = {}
    order: List[Tuple[object, ...]] = []
    for raw_candidate in list(base or ()) + list(added or ()):
        if not isinstance(raw_candidate, BranchConstraintCandidate):
            return None, "invalid_constraint_candidate"
        candidate = raw_candidate.normalized()
        identity = _constraint_identity(candidate)
        existing = merged.get(identity)
        if existing is not None and int(existing.value) != int(candidate.value):
            return None, "conflicting_constraint_values"
        if existing is None:
            order.append(identity)
        merged[identity] = candidate
    return tuple(merged[key] for key in order), None


def refine_constraint_sets(
    base: Iterable[BranchConstraintCandidate],
    added: Iterable[BranchConstraintCandidate],
) -> Tuple[
    Optional[Tuple[BranchConstraintCandidate, ...]],
    Optional[str],
    Tuple[ConstraintRefinement, ...],
]:
    """Compose a candidate environment while allowing prefix-preserving edits.

    Values already accepted for a prefix are a replay witness, not immutable
    firmware state.  A deeper predicate may require changing the same earlier
    input occurrence to another value that still satisfies that prefix.  This
    helper permits that replacement, while rejecting contradictory values
    inside one newly synthesized candidate set.  The caller must replay from
    reset and prove that the old prefix is preserved before accepting it.
    """

    merged: List[BranchConstraintCandidate] = []
    positions: Dict[Tuple[object, ...], int] = {}
    for raw_candidate in base or ():
        if not isinstance(raw_candidate, BranchConstraintCandidate):
            return None, "invalid_constraint_candidate", tuple()
        candidate = raw_candidate.normalized()
        identity = _constraint_identity(candidate)
        existing_index = positions.get(identity)
        if existing_index is not None:
            if int(merged[existing_index].value) != int(candidate.value):
                return None, "conflicting_base_constraint_values", tuple()
            continue
        positions[identity] = len(merged)
        merged.append(candidate)

    additions: List[BranchConstraintCandidate] = []
    addition_values: Dict[Tuple[object, ...], int] = {}
    for raw_candidate in added or ():
        if not isinstance(raw_candidate, BranchConstraintCandidate):
            return None, "invalid_constraint_candidate", tuple()
        candidate = raw_candidate.normalized()
        identity = _constraint_identity(candidate)
        existing_value = addition_values.get(identity)
        if existing_value is not None and int(existing_value) != int(candidate.value):
            return None, "conflicting_refinement_values", tuple()
        if existing_value is None:
            addition_values[identity] = int(candidate.value)
            additions.append(candidate)

    refinements: List[ConstraintRefinement] = []
    for candidate in additions:
        identity = _constraint_identity(candidate)
        existing_index = positions.get(identity)
        if existing_index is None:
            positions[identity] = len(merged)
            merged.append(candidate)
            continue
        previous = merged[existing_index]
        if int(previous.value) == int(candidate.value):
            continue
        refinements.append(ConstraintRefinement(identity, previous, candidate))
        merged[existing_index] = candidate

    return tuple(merged), None, tuple(refinements)


@dataclass(frozen=True)
class NaturalizationFact:
    """A local paired-replay result for one dynamic branch decision."""

    branch_key: BranchKey
    choice: object
    constraints: Tuple[BranchConstraintCandidate, ...]
    scope_signature: PathSignature = tuple()
    source_phase: str = ""
    strategy: str = ""
    paired_control: bool = False
    candidate_consumed: bool = False
    local_success: bool = False

    def normalized(self) -> "NaturalizationFact":
        return NaturalizationFact(
            branch_key=(int(self.branch_key[0]) & 0xFFFFFFFF, max(1, int(self.branch_key[1]))),
            choice=normalize_choice(self.choice),
            constraints=tuple(candidate.normalized() for candidate in self.constraints),
            scope_signature=normalize_signature(self.scope_signature),
            source_phase=str(self.source_phase or ""),
            strategy=str(self.strategy or ""),
            paired_control=bool(self.paired_control),
            candidate_consumed=bool(self.candidate_consumed),
            local_success=bool(self.local_success),
        )

    @property
    def externally_replayable(self) -> bool:
        normalized = self.normalized()
        return bool(
            normalized.constraints
            and normalized.local_success
            and normalized.paired_control
            and normalized.candidate_consumed
            and all(candidate.is_environment_fact for candidate in normalized.constraints)
        )

    def identity(self) -> Tuple[object, ...]:
        normalized = self.normalized()
        return (
            normalized.branch_key,
            choice_identity(normalized.choice),
            tuple(
                (
                    candidate.constraint_type,
                    int(candidate.address),
                    int(candidate.value),
                    int(candidate.read_pc or 0),
                    int(candidate.read_occurrence or 0),
                    candidate.input_kind,
                    candidate.dependency_group,
                    candidate.width,
                )
                for candidate in normalized.constraints
            ),
            normalized.scope_signature,
        )


@dataclass
class PathNaturalizationObligation:
    signature: PathSignature
    discovery_trace_signature: PathSignature = tuple()
    target_bbs: Set[int] = field(default_factory=set)
    discovered_bbs: Set[int] = field(default_factory=set)
    source_phases: Set[str] = field(default_factory=set)
    accepted_constraints: Tuple[BranchConstraintCandidate, ...] = tuple()
    matched_prefix_len: int = 0
    matched_discovery_prefix_len: int = 0
    attempts: int = 0
    status: str = "pending"
    last_reason: str = ""
    reason_counts: Counter[str] = field(default_factory=Counter)
    promoted_bbs: Set[int] = field(default_factory=set)
    accepted_progress_steps: int = 0
    accepted_constraint_refinements: int = 0
    created_order: int = 0

    def __post_init__(self) -> None:
        self.signature = normalize_signature(self.signature)
        self.discovery_trace_signature = normalize_signature(
            self.discovery_trace_signature
        )
        self.target_bbs = {int(value) & 0xFFFFFFFF for value in self.target_bbs}
        self.discovered_bbs = {int(value) & 0xFFFFFFFF for value in self.discovered_bbs}
        self.accepted_constraints = tuple(
            candidate.normalized() for candidate in self.accepted_constraints
        )

    def record_reason(self, reason: str) -> None:
        normalized = str(reason or "unknown")
        self.last_reason = normalized
        self.reason_counts[normalized] += 1


@dataclass(frozen=True)
class PathMatch:
    matched_prefix_len: int
    expected_count: int
    complete: bool
    mismatch_reason: Optional[str]
    mismatch_key: Optional[BranchKey]
    observed_choice: object = None


def _event_key(event: object) -> BranchKey:
    return (
        int(getattr(event, "address", 0) or 0) & 0xFFFFFFFF,
        max(1, int(getattr(event, "occurrence_index", 1) or 1)),
    )


def _event_choice(event: object, expected_choice: object) -> object:
    condition = str(getattr(event, "condition", "") or "").upper()
    if condition in {"BX", "BXJ", "BLX", "CALL"}:
        try:
            return int(getattr(event, "target", 0) or 0) & ~1
        except (TypeError, ValueError):
            return getattr(event, "target", None)

    if condition in {"TBB", "TBH", "LDRPC"}:
        original_index = getattr(event, "original_index", None)
        if original_index is not None:
            try:
                return int(original_index)
            except (TypeError, ValueError):
                return original_index

    return bool(getattr(event, "original_taken", False))


def _choices_equal(expected: object, observed: object, event: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(observed, bool) and bool(expected) == bool(observed)
    if isinstance(expected, int) and isinstance(observed, int):
        condition = str(getattr(event, "condition", "") or "").upper()
        if condition in {"BX", "BXJ", "BLX", "CALL"}:
            return (int(expected) & ~1) == (int(observed) & ~1)
        return int(expected) == int(observed)
    return normalize_choice(expected) == normalize_choice(observed)


def match_path_signature(signature: PathSignature, events: Sequence[object]) -> PathMatch:
    """Match exact dynamic occurrences in order, stopping at the first gap."""
    expected = normalize_signature(signature)
    if not expected:
        return PathMatch(0, 0, True, None, None)

    cursor = 0
    matched = 0
    event_list = list(events or ())
    for expected_key, expected_choice in expected:
        found_event = None
        found_index = None
        for index in range(cursor, len(event_list)):
            if _event_key(event_list[index]) == expected_key:
                found_event = event_list[index]
                found_index = index
                break
        if found_event is None:
            return PathMatch(
                matched,
                len(expected),
                False,
                "branch_occurrence_not_reached",
                expected_key,
            )
        observed = _event_choice(found_event, expected_choice)
        if not _choices_equal(expected_choice, observed, found_event):
            return PathMatch(
                matched,
                len(expected),
                False,
                "wrong_natural_choice",
                expected_key,
                observed,
            )
        matched += 1
        cursor = int(found_index) + 1

    return PathMatch(matched, len(expected), True, None, None)


def compare_replay_context_prefix(
    signature: PathSignature,
    control_events: Sequence[object],
    result_events: Sequence[object],
) -> Dict[str, object]:
    """Compare concrete execution lineage through the control divergence site."""
    expected = normalize_signature(signature)
    control = list(control_events or ())
    result = list(result_events or ())
    control_cursor = 0
    result_cursor = 0
    checked = 0
    unknown = 0
    phase_mismatches: Set[str] = set()

    for expected_key, expected_choice in expected:
        control_event = None
        result_event = None
        for index in range(control_cursor, len(control)):
            if _event_key(control[index]) == expected_key:
                control_event = control[index]
                control_cursor = index + 1
                break
        for index in range(result_cursor, len(result)):
            if _event_key(result[index]) == expected_key:
                result_event = result[index]
                result_cursor = index + 1
                break
        if control_event is None or result_event is None:
            break

        comparison = compare_branch_contexts(
            getattr(control_event, "context_signature", None),
            getattr(result_event, "context_signature", None),
        )
        phase_mismatches.update(comparison.get("phase_mismatches", []) or [])
        if not comparison.get("known"):
            unknown += 1
        else:
            checked += 1
        if not comparison.get("compatible", True):
            return {
                "compatible": False,
                "known": True,
                "reason": str(comparison.get("reason") or "context_lineage_mismatch"),
                "mismatch_key": [int(expected_key[0]), int(expected_key[1])],
                "hard_mismatches": list(comparison.get("hard_mismatches", []) or []),
                "phase_mismatches": sorted(phase_mismatches),
                "checked_events": checked,
                "unknown_events": unknown,
            }

        control_choice = _event_choice(control_event, expected_choice)
        if not _choices_equal(expected_choice, control_choice, control_event):
            break

    return {
        "compatible": True,
        "known": bool(checked),
        "reason": "context_compatible" if checked else "context_signature_missing",
        "mismatch_key": None,
        "hard_mismatches": [],
        "phase_mismatches": sorted(phase_mismatches),
        "checked_events": checked,
        "unknown_events": unknown,
    }


class PathNaturalizationLedger:
    """In-memory obligation, fact, attempt, and promotion ledger."""

    def __init__(self, *, max_facts_per_edge: int = 8, max_attempt_records: int = 256):
        self.max_facts_per_edge = max(1, int(max_facts_per_edge))
        self.max_attempt_records = max(1, int(max_attempt_records))
        self.obligations: Dict[ObligationKey, PathNaturalizationObligation] = {}
        self.facts: Dict[Tuple[BranchKey, Tuple[str, object]], List[NaturalizationFact]] = defaultdict(list)
        self.path_evidence: Dict[PathSignature, Dict[str, object]] = {}
        self.attempt_records: List[Dict[str, object]] = []
        self._created_order = 0

    @staticmethod
    def fact_key(branch_key: BranchKey, choice: object) -> Tuple[BranchKey, Tuple[str, object]]:
        return (
            (int(branch_key[0]) & 0xFFFFFFFF, max(1, int(branch_key[1]))),
            choice_identity(choice),
        )

    def register_path_evidence(
        self,
        signature: PathSignature,
        evidence_class: str,
        *,
        source_phase: str = "",
        assumptions: Iterable[str] = (),
    ) -> None:
        normalized = normalize_signature(signature)
        evidence = str(evidence_class or EVIDENCE_UNKNOWN)
        current = self.path_evidence.get(normalized)
        if current is None:
            current = {
                "evidence_class": evidence,
                "source_phases": set(),
                "assumptions": set(),
            }
            self.path_evidence[normalized] = current
        elif _EVIDENCE_STRENGTH.get(evidence, 0) > _EVIDENCE_STRENGTH.get(
            str(current.get("evidence_class") or EVIDENCE_UNKNOWN), 0
        ):
            current["evidence_class"] = evidence
        if source_phase:
            current.setdefault("source_phases", set()).add(str(source_phase))
        current.setdefault("assumptions", set()).update(str(item) for item in assumptions if item)

    def evidence_class(self, signature: PathSignature) -> str:
        normalized = normalize_signature(signature)
        record = self.path_evidence.get(normalized)
        if record is None:
            return EVIDENCE_E0 if not normalized else EVIDENCE_UNKNOWN
        return str(record.get("evidence_class") or EVIDENCE_UNKNOWN)

    def register_obligation(
        self,
        signature: PathSignature,
        *,
        target_bbs: Iterable[int],
        discovered_bbs: Iterable[int] = (),
        discovery_trace_signature: PathSignature = tuple(),
        source_phase: str = "",
    ) -> Optional[PathNaturalizationObligation]:
        normalized = normalize_signature(signature)
        normalized_discovery_trace = normalize_signature(
            discovery_trace_signature
        )
        targets = {int(value) & 0xFFFFFFFF for value in target_bbs or ()}
        discovered = {int(value) & 0xFFFFFFFF for value in discovered_bbs or ()}
        if not normalized or not (targets or discovered):
            return None
        obligation_key = (normalized, normalized_discovery_trace)
        obligation = self.obligations.get(obligation_key)
        if obligation is None and not normalized_discovery_trace:
            same_signature = [
                item
                for (stored_signature, _stored_trace), item
                in self.obligations.items()
                if stored_signature == normalized
            ]
            if len(same_signature) == 1:
                # A target-only update should enrich the sole witnessed
                # context instead of creating an unscoped duplicate.
                obligation = same_signature[0]
        if obligation is None:
            obligation = PathNaturalizationObligation(
                signature=normalized,
                discovery_trace_signature=normalized_discovery_trace,
                target_bbs=targets,
                discovered_bbs=discovered,
                source_phases={str(source_phase)} if source_phase else set(),
                created_order=self._created_order,
            )
            self._created_order += 1
            self.obligations[obligation_key] = obligation
        else:
            previous_targets = set(obligation.target_bbs)
            previous_discovered = set(obligation.discovered_bbs)
            obligation.target_bbs.update(targets)
            obligation.discovered_bbs.update(discovered)
            if source_phase:
                obligation.source_phases.add(str(source_phase))
            newly_required = (
                (obligation.target_bbs - previous_targets)
                | (obligation.discovered_bbs - previous_discovered)
            ) - obligation.promoted_bbs
            if newly_required:
                obligation.status = "pending"
                obligation.attempts = 0
            elif obligation.status in {"deferred", "blocked_no_fact", "partial"}:
                obligation.status = "pending"
        self.register_path_evidence(
            normalized,
            EVIDENCE_E2,
            source_phase=source_phase,
            assumptions=("forced_control",),
        )
        return obligation

    def register_fact(self, fact: NaturalizationFact) -> bool:
        normalized = fact.normalized()
        key = self.fact_key(normalized.branch_key, normalized.choice)
        facts = self.facts[key]
        identity = normalized.identity()
        if any(existing.identity() == identity for existing in facts):
            return False
        facts.append(normalized)
        facts.sort(
            key=lambda item: (
                1 if item.externally_replayable else 0,
                1 if item.strategy == "entry_fallback" else 0,
                -len(item.constraints),
            ),
            reverse=True,
        )
        del facts[self.max_facts_per_edge :]
        for obligation in self.obligations.values():
            if any(
                edge_key == normalized.branch_key
                and choice_identity(edge_choice) == choice_identity(normalized.choice)
                for edge_key, edge_choice in (
                    obligation.signature
                    + obligation.discovery_trace_signature
                )
            ) and obligation.status in {"deferred", "blocked_no_fact", "exhausted"}:
                obligation.status = "pending"
                obligation.attempts = 0
        return True

    def facts_for(self, branch_key: BranchKey, choice: object) -> List[NaturalizationFact]:
        return list(self.facts.get(self.fact_key(branch_key, choice), ()))

    def pending_obligations(self, *, max_paths: int = 0, max_attempts_per_path: int = 8) -> List[PathNaturalizationObligation]:
        pending = [
            obligation
            for obligation in self.obligations.values()
            if obligation.status not in {"promoted", "already_natural", "exhausted"}
            and obligation.attempts < max(1, int(max_attempts_per_path))
        ]
        pending.sort(
            key=lambda obligation: (
                obligation.matched_prefix_len,
                obligation.matched_discovery_prefix_len,
                sum(
                    1
                    for key, choice in obligation.signature
                    if any(fact.externally_replayable for fact in self.facts_for(key, choice))
                ),
                len(obligation.target_bbs),
                -len(obligation.signature),
                -obligation.created_order,
            ),
            reverse=True,
        )
        return pending[:max_paths] if max_paths and max_paths > 0 else pending

    def candidate_extensions(
        self,
        obligation: PathNaturalizationObligation,
        branch_key: BranchKey,
        choice: object,
    ) -> Tuple[List[Tuple[Tuple[BranchConstraintCandidate, ...], NaturalizationFact]], str]:
        candidates = []
        rejected_non_external = 0
        rejected_conflict = 0
        rejected_scope = 0
        expected_prefix: PathSignature = tuple()
        for trace in (
            obligation.discovery_trace_signature,
            obligation.signature,
        ):
            for index, (expected_key, _expected_choice) in enumerate(trace):
                if expected_key == branch_key:
                    expected_prefix = trace[:index]
                    break
            if expected_prefix or (trace and trace[0][0] == branch_key):
                break

        facts = self.facts_for(branch_key, choice)
        facts.sort(
            key=lambda fact: (
                bool(
                    not fact.scope_signature
                    or expected_prefix[: len(fact.scope_signature)] == fact.scope_signature
                ),
                fact.externally_replayable,
                -len(fact.constraints),
            ),
            reverse=True,
        )
        for fact in facts:
            if not fact.externally_replayable:
                rejected_non_external += 1
                continue
            fact_scope = normalize_signature(fact.scope_signature)
            if fact_scope and expected_prefix[: len(fact_scope)] != fact_scope:
                rejected_scope += 1
                continue
            merged, reason, _refinements = refine_constraint_sets(
                obligation.accepted_constraints,
                fact.constraints,
            )
            if merged is None:
                rejected_conflict += 1
                continue
            if merged == obligation.accepted_constraints:
                continue
            candidates.append((merged, fact))
        if candidates:
            return candidates, "ready"
        if rejected_conflict:
            return [], "constraint_conflict"
        if rejected_scope:
            return [], "scope_mismatch"
        if rejected_non_external:
            return [], "fact_not_externalized"
        return [], "missing_causal_fact"

    def record_attempt(self, record: Dict[str, object]) -> None:
        if len(self.attempt_records) >= self.max_attempt_records:
            return
        self.attempt_records.append(dict(record))

    def accept_progress(
        self,
        obligation: PathNaturalizationObligation,
        *,
        constraints: Iterable[BranchConstraintCandidate],
        matched_prefix_len: int,
        matched_discovery_prefix_len: int,
        refinements: Iterable[ConstraintRefinement] = (),
    ) -> None:
        """Commit only progress that a force-free reset replay established."""

        obligation.accepted_constraints = tuple(
            candidate.normalized() for candidate in constraints or ()
        )
        obligation.matched_prefix_len = max(
            int(obligation.matched_prefix_len),
            max(0, int(matched_prefix_len)),
        )
        obligation.matched_discovery_prefix_len = max(
            int(obligation.matched_discovery_prefix_len),
            max(0, int(matched_discovery_prefix_len)),
        )
        obligation.accepted_progress_steps += 1
        obligation.accepted_constraint_refinements += len(tuple(refinements or ()))

    def promote(
        self,
        obligation: PathNaturalizationObligation,
        *,
        promoted_bbs: Iterable[int],
        evidence_class: str,
        source_phase: str,
        already_natural: bool = False,
    ) -> None:
        obligation.promoted_bbs.update(int(value) & 0xFFFFFFFF for value in promoted_bbs)
        obligation.status = "already_natural" if already_natural else "promoted"
        obligation.last_reason = obligation.status
        self.register_path_evidence(
            obligation.signature,
            evidence_class,
            source_phase=source_phase,
            assumptions=("force_free_replay",),
        )

    def summary(self) -> Dict[str, object]:
        status_counts = Counter(obligation.status for obligation in self.obligations.values())
        evidence_counts = Counter(
            str(record.get("evidence_class") or EVIDENCE_UNKNOWN)
            for record in self.path_evidence.values()
        )
        promoted_bbs = {
            bb
            for obligation in self.obligations.values()
            for bb in obligation.promoted_bbs
        }
        reason_counts: Counter[str] = Counter()
        for obligation in self.obligations.values():
            reason_counts.update(obligation.reason_counts)
        reason_category_counts: Counter[str] = Counter()
        for reason, count in reason_counts.items():
            reason_category_counts[reason_category(reason)] += int(count)
        fact_strategy_counts: Counter[str] = Counter()
        fact_source_phase_counts: Counter[str] = Counter()
        for facts in self.facts.values():
            for fact in facts:
                fact_strategy_counts[str(fact.strategy or "unknown")] += 1
                fact_source_phase_counts[str(fact.source_phase or "unknown")] += 1
        candidate_source_counts: Counter[str] = Counter()
        candidate_set_width_counts: Counter[str] = Counter()
        failure_classification_counts: Counter[str] = Counter()
        for attempt in self.attempt_records:
            constraints = list(attempt.get("constraints", []) or [])
            candidate_set_width_counts[str(len(constraints))] += 1
            for constraint in constraints:
                if not isinstance(constraint, dict):
                    continue
                candidate_source_counts[str(constraint.get("source") or "unknown")] += 1
            evaluation = dict(attempt.get("evaluation", {}) or {})
            classification = dict(
                evaluation.get("failure_classification", {}) or {}
            )
            category = classification.get("category")
            if category:
                failure_classification_counts[str(category)] += 1
        contexts_per_signature = Counter(
            obligation.signature for obligation in self.obligations.values()
        )
        records = []
        for obligation in sorted(self.obligations.values(), key=lambda item: item.created_order)[:128]:
            discovery_trace = obligation.discovery_trace_signature
            discovery_trace_sample = (
                discovery_trace
                if len(discovery_trace) <= 16
                else discovery_trace[:8] + discovery_trace[-8:]
            )
            records.append({
                "signature": serialize_signature(obligation.signature),
                "signature_length": len(obligation.signature),
                "discovery_trace_length": len(
                    discovery_trace
                ),
                "discovery_trace_sample": serialize_signature(
                    discovery_trace_sample
                ),
                "discovery_trace_sample_truncated": (
                    len(discovery_trace) > len(discovery_trace_sample)
                ),
                "discovery_trace_sample_strategy": "head_tail",
                "status": obligation.status,
                "target_bbs": [f"0x{bb:08x}" for bb in sorted(obligation.target_bbs)[:64]],
                "discovered_bbs": [f"0x{bb:08x}" for bb in sorted(obligation.discovered_bbs)[:64]],
                "promoted_bbs": [f"0x{bb:08x}" for bb in sorted(obligation.promoted_bbs)[:64]],
                "source_phases": sorted(obligation.source_phases),
                "accepted_constraints": len(obligation.accepted_constraints),
                "accepted_constraint_records": [
                    {
                        "type": candidate.constraint_type,
                        "address": f"0x{int(candidate.address) & 0xFFFFFFFF:08x}",
                        "value": f"0x{int(candidate.value) & 0xFFFFFFFF:08x}",
                        "read_pc": (
                            f"0x{int(candidate.read_pc) & 0xFFFFFFFF:08x}"
                            if candidate.read_pc is not None
                            else None
                        ),
                        "read_occurrence": candidate.read_occurrence,
                        "source": candidate.source or None,
                    }
                    for candidate in obligation.accepted_constraints[:16]
                ],
                "matched_prefix_len": int(obligation.matched_prefix_len),
                "matched_discovery_prefix_len": int(
                    obligation.matched_discovery_prefix_len
                ),
                "attempts": int(obligation.attempts),
                "accepted_progress_steps": int(
                    obligation.accepted_progress_steps
                ),
                "accepted_constraint_refinements": int(
                    obligation.accepted_constraint_refinements
                ),
                "last_reason": obligation.last_reason or None,
                "reason_counts": dict(obligation.reason_counts),
                "evidence_class": self.evidence_class(obligation.signature),
            })
        return {
            "schema": "path_naturalization.v5",
            "obligations": len(self.obligations),
            "unique_path_signatures": len(contexts_per_signature),
            "signatures_with_multiple_discovery_contexts": sum(
                int(count > 1) for count in contexts_per_signature.values()
            ),
            "max_discovery_contexts_per_signature": max(
                contexts_per_signature.values(),
                default=0,
            ),
            "status_counts": dict(status_counts),
            "facts": sum(len(items) for items in self.facts.values()),
            "externally_replayable_facts": sum(
                1 for items in self.facts.values() for fact in items if fact.externally_replayable
            ),
            "path_evidence_counts": dict(evidence_counts),
            "promoted_bbs": len(promoted_bbs),
            "accepted_progress_steps": sum(
                int(item.accepted_progress_steps)
                for item in self.obligations.values()
            ),
            "accepted_constraint_refinements": sum(
                int(item.accepted_constraint_refinements)
                for item in self.obligations.values()
            ),
            "promoted_bb_list": [f"0x{bb:08x}" for bb in sorted(promoted_bbs)[:512]],
            "attempts": len(self.attempt_records),
            "reason_counts": dict(reason_counts),
            "reason_category_counts": dict(reason_category_counts),
            "fact_strategy_counts": dict(fact_strategy_counts),
            "fact_source_phase_counts": dict(fact_source_phase_counts),
            "candidate_source_counts": dict(candidate_source_counts),
            "candidate_set_width_counts": dict(candidate_set_width_counts),
            "failure_classification_counts": dict(
                failure_classification_counts
            ),
            "records": records,
            "attempt_records": list(self.attempt_records),
        }


def evaluate_force_free_replay(
    *,
    signature: PathSignature,
    target_bbs: Iterable[int],
    discovered_bbs: Iterable[int],
    control_events: Sequence[object],
    control_coverage: Iterable[int],
    result_events: Sequence[object],
    result_coverage: Iterable[int],
    constraints: Sequence[BranchConstraintCandidate],
    constraint_feedback: Optional[Dict[str, object]],
    forced_trace_count: int,
    control_forced_trace_count: int = 0,
    forced_choices_configured: int = 0,
    control_forced_choices_configured: int = 0,
    result_run_result: Optional[Dict[str, object]] = None,
    control_run_result: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    expected = normalize_signature(signature)
    targets = {int(value) & 0xFFFFFFFF for value in target_bbs or ()}
    discovered = {int(value) & 0xFFFFFFFF for value in discovered_bbs or ()}
    required = targets or discovered
    control_set = {int(value) & 0xFFFFFFFF for value in control_coverage or ()}
    result_set = {int(value) & 0xFFFFFFFF for value in result_coverage or ()}
    control_match = match_path_signature(expected, control_events)
    result_match = match_path_signature(expected, result_events)
    context_evaluation = compare_replay_context_prefix(
        expected,
        control_events,
        result_events,
    )
    control_targets = control_set & required
    result_targets = result_set & required
    normalized_constraints = [candidate.normalized() for candidate in constraints or ()]
    environment_constraints = [
        candidate for candidate in normalized_constraints if candidate.is_environment_fact
    ]
    feedback = dict(constraint_feedback or {})
    constraints_consumed = bool(
        environment_constraints
        and len(environment_constraints) == len(normalized_constraints)
        and feedback.get("all_constraint_reads_matched")
    )
    paired_results_supplied = (
        control_run_result is not None or result_run_result is not None
    )
    paired_results_complete = (
        control_run_result is not None and result_run_result is not None
    )
    paired_state = state_equivalence_record(
        control_run_result,
        result_run_result,
    )
    initial_state_fingerprint_available = bool(
        paired_state.get("initial_state_fingerprint_available")
    )
    same_initial_fingerprint = bool(paired_state.get("same_initial_fingerprint"))

    def inspect_validation_run(
        run_result: Optional[Dict[str, object]],
        explicit_forced_trace_count: int,
    ) -> Dict[str, object]:
        """Inspect actual execution facts; configuration is never evidence."""
        if isinstance(run_result, Mapping):
            interventions = list(execution_intervention_reasons(run_result))
            facts_available = has_execution_facts(run_result)
            telemetry_complete = execution_telemetry_complete(run_result)
            _status, contract_reasons = classify_execution(run_result)
            # The explicit count is retained as a compatibility fallback for
            # adapters that expose a trace count but not the typed reason.
            if int(explicit_forced_trace_count or 0) > 0 and "forced_branch" not in interventions:
                interventions.append("forced_branch")
            return {
                "interventions": list(dict.fromkeys(interventions)),
                "facts_available": bool(facts_available),
                "telemetry_complete": bool(telemetry_complete),
                "contract_reasons": list(dict.fromkeys(contract_reasons)),
            }

        # The pure helper historically accepted only event/coverage arguments.
        # Preserve that API for old callers, but mark it explicitly as a legacy
        # mode.  Production naturalization passes both concrete run results and
        # therefore takes the strict branch below.
        interventions = []
        if int(explicit_forced_trace_count or 0) > 0:
            interventions.append("forced_branch")
        return {
            "interventions": interventions,
            "facts_available": False,
            "telemetry_complete": False,
            "contract_reasons": [],
        }

    control_state = inspect_validation_run(
        control_run_result,
        control_forced_trace_count,
    )
    result_state = inspect_validation_run(
        result_run_result,
        forced_trace_count,
    )
    control_interventions = list(control_state["interventions"])
    result_interventions = list(result_state["interventions"])
    control_force_free = "forced_branch" not in control_interventions
    result_force_free = "forced_branch" not in result_interventions
    no_forced_control = control_force_free and result_force_free

    if paired_results_complete:
        complete_execution_telemetry = True
        execution_telemetry_is_complete = bool(
            control_state["facts_available"]
            and result_state["facts_available"]
            and control_state["telemetry_complete"]
            and result_state["telemetry_complete"]
        )
        no_execution_intervention = not (
            control_interventions or result_interventions
        )
        execution_contract_reasons = list(
            dict.fromkeys(
                list(control_state["contract_reasons"])
                + list(result_state["contract_reasons"])
            )
        )
        if not initial_state_fingerprint_available:
            execution_contract_reasons.append("initial_state_fingerprint_missing")
        elif not same_initial_fingerprint:
            execution_contract_reasons.append("initial_state_fingerprint_diverged")
    elif paired_results_supplied:
        complete_execution_telemetry = False
        execution_telemetry_is_complete = False
        no_execution_intervention = False
        execution_contract_reasons = ["missing_paired_execution_result"]
    else:
        # Compatibility mode for callers that have not been upgraded to pass
        # run results.  The explicit trace arguments still enforce the old
        # contract, but the returned flag makes the evidence gap visible.
        complete_execution_telemetry = False
        execution_telemetry_is_complete = True
        no_execution_intervention = not (
            control_interventions or result_interventions
        )
        execution_contract_reasons = ["legacy_execution_facts_not_supplied"]

    control_evidence_valid = bool(
        (not paired_results_complete)
        or (
            control_state["facts_available"]
            and control_state["telemetry_complete"]
            and not control_state["contract_reasons"]
        )
    )
    result_evidence_valid = bool(
        (not paired_results_complete)
        or (
            result_state["facts_available"]
            and result_state["telemetry_complete"]
            and not result_state["contract_reasons"]
        )
    )
    candidate_changes_target = bool(result_targets and not control_targets)
    complete = bool(
        result_match.complete
        and result_targets
        and no_forced_control
        and no_execution_intervention
        and execution_telemetry_is_complete
        and (
            not paired_results_complete
            or (initial_state_fingerprint_available and same_initial_fingerprint)
        )
        and control_evidence_valid
        and result_evidence_valid
        and bool(context_evaluation.get("compatible", True))
        and (not normalized_constraints or constraints_consumed)
    )
    already_natural = bool(
        control_match.complete
        and control_targets
        and control_force_free
        and not control_interventions
        and execution_telemetry_is_complete
        and control_evidence_valid
        and (
            not paired_results_complete
            or (initial_state_fingerprint_available and same_initial_fingerprint)
        )
    )
    promoted_bbs = (
        control_set if already_natural else result_set
    ) & (targets | discovered)

    if already_natural:
        reason = "already_natural_control"
        success = True
    elif paired_results_supplied and not paired_results_complete:
        reason = "missing_paired_execution_result"
        success = False
    elif not no_forced_control:
        reason = "forced_control_observed_in_validation"
        success = False
    elif not no_execution_intervention:
        reason = "execution_intervention_observed_in_validation"
        success = False
    elif paired_results_complete and not execution_telemetry_is_complete:
        reason = "incomplete_execution_telemetry"
        success = False
    elif paired_results_complete and not initial_state_fingerprint_available:
        reason = "initial_state_fingerprint_missing"
        success = False
    elif paired_results_complete and not same_initial_fingerprint:
        reason = "initial_state_fingerprint_diverged"
        success = False
    elif paired_results_complete and not (control_evidence_valid and result_evidence_valid):
        reason = "execution_evidence_contract_failed"
        success = False
    elif normalized_constraints and not constraints_consumed:
        reason = "candidate_not_consumed"
        success = False
    elif not context_evaluation.get("compatible", True):
        reason = "replay_context_incompatible"
        success = False
    elif not result_match.complete:
        reason = result_match.mismatch_reason or "path_signature_not_matched"
        success = False
    elif not result_targets:
        reason = "target_not_reached"
        success = False
    elif control_targets:
        reason = "target_already_reached_by_control"
        success = False
    elif not candidate_changes_target:
        reason = "candidate_no_target_delta"
        success = False
    else:
        reason = "promoted"
        success = complete

    return {
        "success": bool(success),
        "already_natural": bool(already_natural),
        "reason": reason,
        "context_compatibility": context_evaluation,
        "control_match": {
            "matched_prefix_len": control_match.matched_prefix_len,
            "expected_count": control_match.expected_count,
            "complete": control_match.complete,
            "mismatch_reason": control_match.mismatch_reason,
            "mismatch_key": list(control_match.mismatch_key) if control_match.mismatch_key else None,
        },
        "result_match": {
            "matched_prefix_len": result_match.matched_prefix_len,
            "expected_count": result_match.expected_count,
            "complete": result_match.complete,
            "mismatch_reason": result_match.mismatch_reason,
            "mismatch_key": list(result_match.mismatch_key) if result_match.mismatch_key else None,
            "observed_choice": result_match.observed_choice,
        },
        "control_target_bbs": [f"0x{bb:08x}" for bb in sorted(control_targets)],
        "result_target_bbs": [f"0x{bb:08x}" for bb in sorted(result_targets)],
        "promoted_bbs": sorted(promoted_bbs),
        "constraints_consumed": constraints_consumed,
        "forced_trace_count": int(forced_trace_count or 0),
        "control_forced_trace_count": int(control_forced_trace_count or 0),
        "forced_choices_configured": int(forced_choices_configured or 0),
        "control_forced_choices_configured": int(control_forced_choices_configured or 0),
        "candidate_changes_target": candidate_changes_target,
        "control_execution_interventions": control_interventions,
        "result_execution_interventions": result_interventions,
        "control_execution_facts_available": bool(control_state["facts_available"]),
        "result_execution_facts_available": bool(result_state["facts_available"]),
        "control_telemetry_complete": bool(control_state["telemetry_complete"]),
        "result_telemetry_complete": bool(result_state["telemetry_complete"]),
        "execution_contract_reasons": execution_contract_reasons,
        "evidence_contract_mode": (
            "paired_strict"
            if paired_results_complete
            else "legacy_unpaired"
            if not paired_results_supplied
            else "incomplete_pair"
        ),
        "complete_execution_telemetry": complete_execution_telemetry,
        "execution_telemetry_is_complete": execution_telemetry_is_complete,
        "state_equivalence": paired_state,
        "initial_state_fingerprint_available": initial_state_fingerprint_available,
        "same_initial_fingerprint": same_initial_fingerprint,
    }
