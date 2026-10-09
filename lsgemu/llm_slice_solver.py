#!/usr/bin/env python3
"""Adapter wiring the tracer's LLM slice-escalation protocol to LLMGuide.

The tracer owns the expression graph and materializes a pure-data
``DynamicLLMSliceRequest``; LLMGuide owns prompting, parsing and the
inference journal.  This module is the only place where the two meet, keeping
``dynamic_constraint_recovery`` and ``register_tracer`` free of any LLM
dependency while the upper layer (``historical_runner``) stays in charge of
construction and lifetime.

The adapter also carries a bounded per-branch memory of the previous slice
attempt plus its replay outcome, so a *retry* prompt can state the last
values and the real replay result and ask for a reasoned alternative
(implementation task 4).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .dynamic_constraint_recovery import DynamicInputAssignment
from .register_tracer.register_tracer import DynamicLLMSliceRequest

logger = logging.getLogger(__name__)


class LLMGuideSliceSolver:
    """Implement ``DynamicLLMSliceFallback`` on top of :class:`LLMGuide`."""

    def __init__(self, llm_guide, *, attempt_memory_limit: int = 32):
        self.llm_guide = llm_guide
        self.attempt_memory_limit = max(1, int(attempt_memory_limit or 32))
        # (branch_pc, occurrence) -> {"assignments": [...], "sites": [...],
        #                             "replay_outcome": {...}}
        self._previous_attempts: "OrderedDict[Tuple[int, int], Dict[str, Any]]" = (
            OrderedDict()
        )
        self.stats: Dict[str, int] = {
            "calls": 0,
            "skipped": 0,
            "no_assignments": 0,
            "assignments": 0,
            "retries": 0,
            "replay_outcomes_recorded": 0,
        }

    def __call__(
        self,
        request: DynamicLLMSliceRequest,
    ) -> Optional[List[DynamicInputAssignment]]:
        guide = self.llm_guide
        if guide is None or not bool(getattr(guide, "use_llm", False)):
            self.stats["skipped"] += 1
            return None
        if not request.input_sites:
            self.stats["skipped"] += 1
            return None

        key = (int(request.branch_pc), int(request.branch_occurrence))
        previous_attempt = self._previous_attempts.get(key)
        if previous_attempt is not None:
            self.stats["retries"] += 1

        self.stats["calls"] += 1
        raw_assignments = guide.infer_slice_assignments(
            branch_pc=int(request.branch_pc),
            condition=str(request.condition),
            target_taken=bool(request.target_taken),
            instruction_lines=list(request.instruction_lines),
            input_sites=[dict(site) for site in request.input_sites],
            coupled_input_count=int(request.coupled_input_count),
            relation_kind=str(request.relation_kind),
            compare_op=str(request.compare_op),
            compare_pc=int(request.compare_pc),
            constraint_note=str(request.constraint_note),
            same_variable_constraints=int(request.same_variable_constraints),
            prior_failure=dict(request.prior_failure or {}),
            previous_attempt=previous_attempt,
        )
        if not raw_assignments:
            self.stats["no_assignments"] += 1
            return None

        by_index = {
            int(site.get("site_index", -1)): site
            for site in request.input_sites
        }
        assignments: List[DynamicInputAssignment] = []
        for item in raw_assignments:
            try:
                site_index = int(item.get("site_index"))
                value = int(item.get("value"))
            except (TypeError, ValueError, AttributeError):
                continue
            site = by_index.get(site_index)
            if site is None:
                continue
            try:
                width = max(1, min(32, int(site.get("width") or 32)))
            except (TypeError, ValueError):
                width = 32
            mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
            assignments.append(DynamicInputAssignment(
                kind=str(site.get("kind") or "input"),
                address=int(site.get("address") or 0) & 0xFFFFFFFF,
                read_pc=int(site.get("read_pc") or 0) & 0xFFFFFFFF,
                occurrence=max(1, int(site.get("occurrence") or 1)),
                width=width,
                value=value & mask,
                observed_value=int(site.get("current_value") or 0) & mask,
                trace_event_id=int(site.get("trace_event_id") or 0),
            ))
        if not assignments:
            self.stats["no_assignments"] += 1
            return None

        self._remember_attempt(key, raw_assignments, request)
        self.stats["assignments"] += len(assignments)
        return assignments

    def note_replay_failure(
        self,
        branch_pc: int,
        occurrence: int,
        failure_context: Optional[Dict[str, Any]],
    ) -> bool:
        """Attach the real replay outcome to the remembered slice attempt.

        Called by the naturalization layer after the replay validation of an
        LLM-slice candidate set failed; the next escalation for the same
        branch then becomes a retry prompt carrying the previous values and
        the observed failure.
        """
        key = (int(branch_pc) & 0xFFFFFFFF, max(1, int(occurrence or 1)))
        entry = self._previous_attempts.get(key)
        if entry is None:
            return False
        entry["replay_outcome"] = dict(failure_context or {})
        self._previous_attempts.move_to_end(key)
        self.stats["replay_outcomes_recorded"] += 1
        return True

    def _remember_attempt(
        self,
        key: Tuple[int, int],
        raw_assignments: List[Dict[str, int]],
        request: DynamicLLMSliceRequest,
    ) -> None:
        self._previous_attempts[key] = {
            "assignments": [dict(item) for item in raw_assignments],
            "sites": [
                {
                    "site_index": site.get("site_index"),
                    "address": site.get("address"),
                    "occurrence": site.get("occurrence"),
                }
                for site in request.input_sites
            ],
        }
        self._previous_attempts.move_to_end(key)
        while len(self._previous_attempts) > self.attempt_memory_limit:
            self._previous_attempts.popitem(last=False)

    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)
