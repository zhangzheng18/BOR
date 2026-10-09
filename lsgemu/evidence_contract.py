"""Canonical execution-evidence contract.

The historical runner used E0--E3 labels to mix several independent facts:
where a replay started, whether control flow was forced, and whether an event
context was reconstructed.  Those labels remain in compatibility fields, but
new reports use two outcomes:

``validated_replay``
    Unicorn executed the BB under the declared environment model.  MMIO,
    stream/external-memory values, and an event delivered from an observed
    execution context are environment inputs and are allowed.

``diagnostic_replay``
    The execution included a control or execution intervention, an invalid or
    unknown prefix, a cold entry, or incomplete telemetry.  It is useful for
    exploration and debugging, but does not by itself establish a validated
    witness.

This module is intentionally side-effect free.  It is used by the runner,
progress monitor, replay validation helpers, and tests so that those consumers
apply the same contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


VALIDATED_REPLAY = "validated_replay"
DIAGNOSTIC_REPLAY = "diagnostic_replay"
UNCLASSIFIED_REPLAY = "unclassified"
MIXED_REPLAY = "mixed"
EVIDENCE_SCHEMA = "lsgemu.binary_execution_evidence.v2"

# These facts describe values or events supplied by the declared execution
# environment.  They are deliberately kept separate from control-flow and
# execution substitutions: an input byte, an MMIO observation, or an IRQ
# delivered from an observed context is a modeled environmental condition,
# whereas a forced edge or a skipped callee changes the program execution
# itself.
_ENVIRONMENT_FACT_ALIASES = {
    "mmio": "mmio_value",
    "mmio_value": "mmio_value",
    "mmio_read": "mmio_value",
    "mmio_overlay": "mmio_value",
    "external_memory_input": "external_memory_input",
    "external_memory_delivery": "external_memory_input",
    "stream_input": "stream_input_delivery",
    "stream_input_delivery": "stream_input_delivery",
    "stream_input_consumed": "stream_input_delivery",
    "input_ready": "input_ready",
    "environment_input": "environment_input_delivery",
    "environment_input_delivery": "environment_input_delivery",
    "event_delivery": "event_delivery",
    "interrupt_delivery": "interrupt_delivery",
    "irq_delivery": "interrupt_delivery",
    "task_event_delivery": "task_event_delivery",
    "dma_event_delivery": "dma_event_delivery",
    # r9 用户裁定（docs/DECISIONS_ledger_reclass_20260920.md）：
    # execution_preflight_repair 是快照编码的确定性环境函数（重放前纠正
    # Thumb 状态，不选方向），按环境事实豁免；单次重放上限见
    # dfs_flip.dfs_preflight_repair_limit。
    "execution_preflight_repair": "execution_preflight_repair",
}
_STREAM_SUMMARY_KINDS = frozenset(
    {
        "stream_status",
        "stream_byte",
        "stream_read",
        "stream_status_read",
        "riot_msg_receive",
    }
)
_PROVENANCE_INVALID_STATUSES = frozenset(
    {
        "diagnostic",
        "pending",
        "unverified",
        "unknown",
        "unspecified",
        "incomplete",
        "invalid",
        "",
    }
)

# r9 用户裁定（docs/DECISIONS_ledger_reclass_20260920.md）：以下干预家族是
# 工程优化（循环体快进，纯批量内存写、不改分支方向）或外部输入（写 MMIO
# 让轮询/等待环退出 = 外设状态变化），覆盖照常计入、不作 provenance 降级
# 原因。强制转分支（runtime_loop_branch_force / forced_branch）不在此列，
# 禁令不放宽。
_ENGINEERING_OPTIMIZATION_REASONS = frozenset(
    {
        "loop_fast_forward_emulation",
        "loop_mmio_adjust",
        "loop_wait_handled",
    }
)


def _canonical_environment_fact_reason(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized in _ENVIRONMENT_FACT_ALIASES:
        return _ENVIRONMENT_FACT_ALIASES[normalized]
    return normalized


def is_environment_fact_reason(value: Any) -> bool:
    """Return whether a reason is an allowed environment fact.

    Unknown strings are intentionally *not* treated as environmental.  This
    fail-closed rule prevents a new control-flow intervention from becoming a
    validated witness merely because its name contains ``input`` or ``event``.
    """
    return _canonical_environment_fact_reason(value) in set(
        _ENVIRONMENT_FACT_ALIASES.values()
    )


def _contains_non_environment_reason(value: Any) -> bool:
    """Return whether a reason container contains a blocking intervention."""
    if isinstance(value, Mapping):
        return any(
            _positive_int(count)
            for reason, count in value.items()
            if not is_environment_fact_reason(reason)
        )
    if isinstance(value, (list, tuple, set)):
        return any(
            str(reason) and not is_environment_fact_reason(reason)
            for reason in value
        )
    if isinstance(value, str):
        return bool(value.strip()) and not is_environment_fact_reason(value)
    return False


def _normalized_provenance_status(value: Any) -> str:
    return str(value or "").strip().lower()


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse a telemetry boolean without treating ``"false"`` as true."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y", "t"}:
            return True
        if normalized in {"0", "false", "no", "off", "n", "f", ""}:
            return False
    return bool(default)


def snapshot_provenance_record(snapshot: Any) -> dict[str, object]:
    """Resolve provenance from a snapshot and its wrapped source snapshot.

    ``PrefixReplaySnapshot`` is a lightweight wrapper around a mutable
    ``BranchSnapshot``.  The wrapper is often created while the Unicorn run is
    still active, so its copied fields can legitimately say ``pending`` even
    after the wrapped object is finalized.  Reading only the outer object then
    turns a valid prefix into an evidence gap on the next replay.

    The deepest finalized source is authoritative for the common wrapper case,
    while explicit intervention reasons from any layer are retained.  A
    missing finalization bit is never upgraded implicitly.  The function is
    deliberately independent of runner/emulator classes so serialized and
    in-memory snapshots use the same contract.
    """

    def field(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    def values(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        elif isinstance(value, Mapping):
            value = tuple(
                str(key)
                for key, item in value.items()
                if item and str(key)
            )
        try:
            return tuple(str(item) for item in value if str(item))
        except TypeError:
            return ()

    def normalize_status(value: Any) -> str:
        status = str(value or "unspecified").strip().lower()
        if status in {"", "unspecified", "unknown"}:
            return "unverified"
        if status not in {"validated", "diagnostic", "pending", "unverified"}:
            return "unverified"
        return status

    def layer(source: Any) -> dict[str, object]:
        if isinstance(source, Mapping):
            raw = dict(source)
        else:
            raw = {}
            for key in (
                "status",
                "provenance_status",
                "reasons",
                "provenance_reasons",
                "intervention_reasons",
                "prefix_intervention_reasons",
                "execution_id",
                "source_execution_id",
                "telemetry_complete",
                "prefix_telemetry_complete",
                "execution_provenance",
                "snapshot_provenance",
                "provenance_finalized",
                "provenance_invalidated",
                "provenance_invalidation_reasons",
            ):
                if hasattr(source, key):
                    raw[key] = getattr(source, key)

        nested = raw.get("execution_provenance")
        if not isinstance(nested, Mapping):
            nested = raw.get("snapshot_provenance")
        if not isinstance(nested, Mapping):
            nested = {}

        finalization_value = raw.get("provenance_finalized")
        finalization_present = "provenance_finalized" in raw
        if finalization_value is None and "provenance_finalized" in nested:
            finalization_value = nested.get("provenance_finalized")
            finalization_present = True

        status = normalize_status(
            raw.get("status")
            or raw.get("provenance_status")
            or nested.get("status")
            or nested.get("classification")
            or nested.get("provenance_status")
        )
        raw_reasons = tuple(
            dict.fromkeys(
                item
                for item in (
                    values(raw.get("reasons"))
                    + values(raw.get("provenance_reasons"))
                    + values(raw.get("intervention_reasons"))
                    + values(raw.get("prefix_intervention_reasons"))
                    + values(nested.get("reasons"))
                    + values(nested.get("intervention_reasons"))
                    + values(nested.get("prefix_intervention_reasons"))
                )
                if item
            )
        )
        reasons = tuple(
            item for item in raw_reasons if not is_environment_fact_reason(item)
        )
        execution_id = str(
            raw.get("execution_id")
            or raw.get("source_execution_id")
            or nested.get("execution_id")
            or nested.get("source_execution_id")
            or ""
        )
        telemetry_value = raw.get("telemetry_complete")
        if telemetry_value is None:
            telemetry_value = raw.get("prefix_telemetry_complete")
        if telemetry_value is None:
            telemetry_value = nested.get("telemetry_complete")
        if telemetry_value is None:
            telemetry_value = nested.get("prefix_telemetry_complete")

        return {
            "status": status,
            "raw_reasons": raw_reasons,
            "reasons": reasons,
            "execution_id": execution_id,
            "telemetry_value": telemetry_value,
            "finalized": coerce_bool(finalization_value, False),
            "finalization_present": finalization_present,
            "invalidated": coerce_bool(
                raw.get("provenance_invalidated")
                if "provenance_invalidated" in raw
                else nested.get("provenance_invalidated"),
                False,
            ),
            "invalidation_reasons": values(
                raw.get("provenance_invalidation_reasons")
                or nested.get("provenance_invalidation_reasons")
            ),
            "lineage_available": bool(
                execution_id
                or reasons
                or nested
                or any(
                    key in raw
                    for key in (
                        "status",
                        "provenance_status",
                        "provenance_finalized",
                        "telemetry_complete",
                        "prefix_telemetry_complete",
                    )
                )
            ),
        }

    layers: list[dict[str, object]] = []
    seen: set[int] = set()

    def walk(source: Any, depth: int = 0) -> None:
        if source is None or depth > 4:
            return
        identity = id(source)
        if identity in seen:
            return
        seen.add(identity)
        current = layer(source)
        layers.append(current)
        wrapped = field(source, "snapshot", None)
        if wrapped is not None and wrapped is not source:
            walk(wrapped, depth + 1)

    walk(snapshot)
    if not layers:
        return {
            "status": "unverified",
            "reasons": ["missing_snapshot_object"],
            "prefix_intervention_reasons": [],
            "execution_id": "",
            "telemetry_complete": False,
            "provenance_finalized": False,
            "lineage_available": False,
            "snapshot_ancestor_validated": False,
        }

    # Prefer a finalized deepest source, but only when it is compatible with
    # the wrapper's execution id.  An independently-produced wrapper must not
    # inherit a finalized state from an unrelated snapshot.
    top_id = str(layers[0].get("execution_id") or "")
    finalized_layers = [
        item
        for item in layers
        if bool(item.get("finalization_present")) and bool(item.get("finalized"))
    ]
    compatible_finalized = [
        item
        for item in finalized_layers
        if not top_id
        or not str(item.get("execution_id") or "")
        or str(item.get("execution_id") or "") == top_id
    ]
    incompatible_finalized = bool(
        top_id and finalized_layers and not compatible_finalized
    )
    authoritative = (
        compatible_finalized[-1]
        if compatible_finalized
        else layers[0]
        if incompatible_finalized
        else layers[-1]
    )

    all_reasons = list(
        dict.fromkeys(
            str(reason)
            for item in layers
            for reason in tuple(item.get("reasons", ()) or ())
            if str(reason)
        )
    )
    invalidation_reasons = list(
        dict.fromkeys(
            str(reason)
            for item in layers
            for reason in tuple(item.get("invalidation_reasons", ()) or ())
            if str(reason)
        )
    )
    provenance_invalidated = any(
        bool(item.get("invalidated")) for item in layers
    )
    if provenance_invalidated:
        all_reasons.extend(invalidation_reasons)
        if not invalidation_reasons:
            all_reasons.append("snapshot_provenance_invalidated")
    # Older adapters sometimes used ``diagnostic`` as a catch-all status after
    # delivering an MMIO/input/event fact.  That status is promotable only when
    # the layer explicitly reports at least one reason and every such reason is
    # a known environment fact.  A bare diagnostic status, or one containing an
    # unknown reason, remains fail-closed.
    environment_only_diagnostic = any(
        str(item.get("status") or "") == "diagnostic"
        and bool(item.get("raw_reasons"))
        and all(
            is_environment_fact_reason(reason)
            for reason in tuple(item.get("raw_reasons", ()) or ())
        )
        for item in layers
    )
    explicit_diagnostic = any(
        str(item.get("status") or "") == "diagnostic"
        and not (
            bool(item.get("raw_reasons"))
            and all(
                is_environment_fact_reason(reason)
                for reason in tuple(item.get("raw_reasons", ()) or ())
            )
        )
        for item in layers
    )
    status = str(authoritative.get("status") or "unverified")
    finalized = bool(authoritative.get("finalization_present")) and bool(
        authoritative.get("finalized")
    )
    telemetry_value = authoritative.get("telemetry_value")
    telemetry_complete = coerce_bool(
        telemetry_value,
        status == "validated" and finalized and bool(
            str(authoritative.get("execution_id") or "")
        ),
    )
    execution_id = str(
        authoritative.get("execution_id")
        or next(
            (
                str(item.get("execution_id") or "")
                for item in layers
                if str(item.get("execution_id") or "")
            ),
            "",
        )
    )

    if provenance_invalidated:
        # Invalidation is terminal for strict evidence, even if a stale outer
        # wrapper still carries a copied ``validated`` status.
        status = "diagnostic"
    elif explicit_diagnostic or all_reasons:
        status = "diagnostic"
    elif status == "diagnostic" and environment_only_diagnostic:
        # The value/event is part of the declared environment model; it does
        # not change program control flow in the way a forced branch or skipped
        # callee does.  Finalization and telemetry are still checked below.
        status = "validated"
    elif status == "validated":
        if not finalized:
            status = "unverified"
            all_reasons.append("snapshot_provenance_not_finalized")
        elif not telemetry_complete:
            status = "unverified"
            all_reasons.append("snapshot_provenance_telemetry_incomplete")
    elif status not in {"pending", "unverified", "diagnostic"}:
        status = "unverified"

    if incompatible_finalized:
        # A wrapper carrying a different execution id cannot inherit a
        # finalized state from its wrapped snapshot.  Keep the outer layer as
        # the local view and fail closed at the evidence boundary.
        status = "unverified"
        all_reasons.append("snapshot_provenance_execution_id_mismatch")

    # Environment-only promotion above is still subject to the same
    # finalization gate as an explicitly validated source.  Do this after the
    # status transition so a diagnostic source cannot remain validated merely
    # because its reason was an allowed environmental fact.
    if status == "validated" and not finalized:
        status = "unverified"
        all_reasons.append("snapshot_provenance_not_finalized")

    if not bool(authoritative.get("finalization_present")):
        all_reasons.append("snapshot_provenance_finalization_missing")
    if not finalized and status != "diagnostic":
        all_reasons.append("snapshot_provenance_not_finalized")
    if status not in {"validated", "diagnostic"} and not all_reasons:
        all_reasons.append("snapshot_ancestor_not_validated")
    all_reasons = list(dict.fromkeys(all_reasons))
    prefix_reasons = list(
        dict.fromkeys(
            str(reason)
            for item in layers
            for reason in tuple(item.get("reasons", ()) or ())
            if str(reason)
        )
    )
    return {
        "status": status,
        "reasons": all_reasons,
        "prefix_intervention_reasons": prefix_reasons,
        "execution_id": execution_id,
        "telemetry_complete": telemetry_complete,
        "provenance_finalized": finalized,
        "provenance_invalidated": provenance_invalidated,
        "provenance_invalidation_reasons": invalidation_reasons,
        "lineage_available": any(
            bool(item.get("lineage_available")) for item in layers
        ),
        "snapshot_ancestor_validated": bool(
            status == "validated"
            and telemetry_complete
            and finalized
            and not all_reasons
        ),
    }


_CONCRETE_EXECUTION_FACT_KEYS = {
    # These fields describe an execution that actually reached the emulator
    # boundary.  Counters/configuration flags are intentionally absent: they
    # can be copied into a scheduler record without Unicorn ever running.
    "stop_reason",
    "instruction_count",
    "unique_bbs",
    "registers",
    "covered_bbs",
    "covered_bb_list",
    "trace_tail",
    "last_10_pcs",
    "execution_trace",
    "branch_trace",
}


def is_compact_evidence_record(value: Any) -> bool:
    """Return whether *value* is a canonical record, rather than raw facts."""
    return isinstance(value, Mapping) and str(value.get("schema") or "") == EVIDENCE_SCHEMA


def compact_record_is_verified_validated(value: Any) -> bool:
    """Return whether a compact record can own validated BB evidence.

    A status string is not sufficient.  This predicate is deliberately shared
    by phase aggregation, bounded-record retention, and final reporting so a
    malformed or stale compact record cannot be accepted at one boundary and
    rejected at another.
    """
    if not is_compact_evidence_record(value):
        return False
    if str(value.get("status") or "").strip().lower() != VALIDATED_REPLAY:
        return False
    if not coerce_bool(value.get("execution_facts_available"), False):
        return False
    if not coerce_bool(value.get("telemetry_complete"), False):
        return False
    # A producer must not be able to turn a failed attempt into a strict
    # witness by leaving the status field at ``validated_replay``.  Compact
    # records are consumed after the original run result has been discarded,
    # so check both the normalized failure list and the copied failure fields.
    if _execution_failure_reasons(value):
        return False
    if any(
        coerce_bool(value.get(key), False)
        for key in (
            "execution_failed",
            "runtime_failure",
            "restore_failed",
            "preflight_failed",
        )
    ):
        return False
    if any(
        value.get(key) not in (None, "", False)
        for key in ("execution_error", "failure_reason", "restore_error")
    ):
        return False
    if any(
        not is_environment_fact_reason(reason)
        for reason in list(value.get("reasons", []) or [])
    ):
        return False
    if list(value.get("execution_failure_reasons", []) or []):
        return False
    # Prefix and execution provenance are independent of the current suffix.
    # A compact record may not discard a blocking ancestor reason and later be
    # accepted solely because its top-level status was copied as validated.
    for key in (
        "prefix_intervention_reasons",
        "execution_provenance_reasons",
    ):
        if _contains_non_environment_reason(value.get(key)):
            return False
    if coerce_bool(value.get("provenance_invalidated"), False):
        return False
    if value.get("provenance_invalidation_reasons") not in (None, "", [], (), set()):
        return False
    for key in ("prefix_provenance_status", "execution_provenance_status"):
        if key not in value or value.get(key) in (None, ""):
            continue
        if _normalized_provenance_status(value.get(key)) != "validated":
            return False
    if "prefix_telemetry_complete" in value and not coerce_bool(
        value.get("prefix_telemetry_complete"), False
    ):
        return False
    if "provenance_finalized" in value and not coerce_bool(
        value.get("provenance_finalized"), False
    ):
        return False
    for key in ("snapshot_provenance", "prefix_provenance", "execution_provenance"):
        nested = value.get(key)
        if not isinstance(nested, Mapping):
            continue
        nested_status = _normalized_provenance_status(
            nested.get("status")
            or nested.get("classification")
            or nested.get("provenance_status")
        )
        if nested_status and nested_status != "validated":
            return False
        if _contains_non_environment_reason(
            nested.get("reasons")
            or nested.get("provenance_reasons")
            or nested.get("intervention_reasons")
            or nested.get("prefix_intervention_reasons")
        ):
            return False
        if coerce_bool(nested.get("provenance_invalidated"), False):
            return False
        if nested.get("provenance_invalidation_reasons") not in (
            None,
            "",
            [],
            (),
            set(),
        ):
            return False
        if "telemetry_complete" in nested and not coerce_bool(
            nested.get("telemetry_complete"), False
        ):
            return False
        if "provenance_finalized" in nested and not coerce_bool(
            nested.get("provenance_finalized"), False
        ):
            return False

    # Context/event replay is valid only when the event is delivered from a
    # validated observed prefix.  The event itself remains an environment
    # input; this check only prevents a synthetic context label from replacing
    # the required snapshot lineage.
    if coerce_bool(value.get("context_snapshot_mode"), False):
        ancestor_validated = coerce_bool(
            value.get("snapshot_ancestor_validated"), False
        )
        prefix_status = _normalized_provenance_status(
            value.get("prefix_provenance_status")
        )
        prefix_complete = coerce_bool(
            value.get("prefix_telemetry_complete"), False
        )
        finalized = coerce_bool(value.get("provenance_finalized"), False)
        for key in ("snapshot_provenance", "prefix_provenance", "execution_provenance"):
            nested = value.get(key)
            if not isinstance(nested, Mapping):
                continue
            ancestor_validated = ancestor_validated or coerce_bool(
                nested.get("snapshot_ancestor_validated"), False
            )
            if prefix_status != "validated":
                prefix_status = _normalized_provenance_status(
                    nested.get("prefix_provenance_status")
                    or nested.get("status")
                    or nested.get("provenance_status")
                )
            prefix_complete = prefix_complete or coerce_bool(
                nested.get("prefix_telemetry_complete")
                if "prefix_telemetry_complete" in nested
                else nested.get("telemetry_complete"),
                False,
            )
            finalized = finalized or coerce_bool(
                nested.get("provenance_finalized"), False
            )
        if not ancestor_validated:
            return False
        if prefix_status != "validated":
            return False
        if not prefix_complete or not finalized:
            return False
    actual = value.get("actual_intervention_reasons", {})
    if isinstance(actual, Mapping):
        if any(
            _positive_int(count)
            for reason, count in actual.items()
            if not is_environment_fact_reason(reason)
        ):
            return False
    elif isinstance(actual, (list, tuple, set)) and any(
        not is_environment_fact_reason(reason) for reason in actual
    ):
        return False
    elif (
        isinstance(actual, str)
        and actual.strip()
        and not is_environment_fact_reason(actual)
    ):
        return False
    if coerce_bool(value.get("paired_validation_required"), False):
        if not coerce_bool(value.get("paired_validation_complete"), False):
            return False
        if not coerce_bool(value.get("paired_initial_state_same"), False):
            return False
    return True


def is_materialized_execution_attempt(value: Any) -> bool:
    """Return whether *value* describes an attempted execution or its failure.

    Setup and restore failures often occur before Unicorn emits registers or a
    trace.  They still need one diagnostic record.  A bare configuration map,
    including zero-valued counters, is not an attempted execution.
    """
    if not isinstance(value, Mapping):
        return False
    nested = value.get("run_result")
    if isinstance(nested, Mapping):
        return is_materialized_execution_attempt(nested)
    if has_execution_facts(value):
        return True
    if any(
        key in value
        for key in (
            "execution_attempted",
            "execution_started",
            "execution_completed_normally",
            "execution_failed",
            "preflight_failed",
            "runtime_failure",
            "restore_failed",
            "execution_error",
            "exception",
            "crash_error",
        )
    ):
        return True
    stop_reason = str(value.get("stop_reason") or "").strip()
    if stop_reason:
        return True
    return bool(_intervention_counter_from_mapping(value))


def _contains_concrete_execution_facts(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    # A producer may return a diagnostic setup record with counters and a
    # stop reason even though it never crossed the Unicorn execution boundary.
    # Keep that distinction explicit; such a record is useful for debugging,
    # but it cannot own a BB witness.
    if "execution_started" in value and not coerce_bool(
        value.get("execution_started"), False
    ):
        return False
    nested = value.get("run_result")
    if isinstance(nested, Mapping):
        return _contains_concrete_execution_facts(nested)
    if is_compact_evidence_record(value):
        return False
    if any(key in value for key in _CONCRETE_EXECUTION_FACT_KEYS):
        return True
    telemetry = value.get("execution_telemetry")
    return isinstance(telemetry, Mapping) and bool(
        set(telemetry).intersection(_CONCRETE_EXECUTION_FACT_KEYS)
    )


def has_execution_facts(value: Any) -> bool:
    """Return whether *value* contains a recognizable per-run fact record.

    An arbitrary mapping is not evidence.  This distinction matters for child
    replay summaries: a worker that returned ``{}`` or a scheduler metadata
    dictionary did not necessarily execute Unicorn successfully.
    """
    if not isinstance(value, Mapping):
        return False
    nested = value.get("run_result")
    if isinstance(nested, Mapping):
        return has_execution_facts(nested)
    return _contains_concrete_execution_facts(value)


def _positive_int(value: Any) -> int:
    """Parse a counter without allowing malformed telemetry to abort a run."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _add_environment_fact(
    counter: Counter[str],
    reason: Any,
    value: Any = 1,
) -> None:
    canonical = _canonical_environment_fact_reason(reason)
    if not is_environment_fact_reason(canonical):
        return
    count = _positive_int(value)
    if count:
        counter[canonical] = max(int(counter.get(canonical, 0) or 0), count)


def _successful_stream_summary_count(value: Any) -> int:
    """Count stream summaries that actually delivered/observed input state.

    ``skip_function_stats.applied`` is cumulative and includes unrelated
    summaries such as allocators and time functions.  A stream event is an
    environment fact only when the stream adapter produced a concrete result;
    failed calls with a ``reason`` remain ordinary summary/skip interventions.
    """
    if not isinstance(value, (list, tuple, set)):
        return 0
    count = 0
    for item in value:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in _STREAM_SUMMARY_KINDS:
            continue
        if str(item.get("reason") or "").strip():
            continue
        if "return_value" in item or "payload_size" in item or "byte" in item:
            count += 1
    return count


def _peripheral_input_stats(mapping: Any) -> Mapping[str, Any]:
    """Return RX/input-ready counters from a run or MMIO statistics bundle."""
    if not isinstance(mapping, Mapping):
        return {}
    for key in ("peripheral_input_stats", "peripheral_input"):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    mmio_statistics = mapping.get("mmio_statistics")
    if isinstance(mmio_statistics, Mapping):
        value = mmio_statistics.get("peripheral_input")
        if isinstance(value, Mapping):
            return value
    return {}


def environment_model_diagnostics(mapping: Any) -> Counter[str]:
    """Collect RX model consistency warnings without treating them as inputs.

    A status read followed by a matching data read is an environmental fact.
    A data read without a preceding ready observation, or an overlay byte that
    disagrees with the byte prepared by the status read, is different: it is a
    model-consistency condition that must remain visible to the report.  These
    counters are deliberately not returned by :func:`environment_input_facts`
    and therefore cannot make a control-flow intervention look environmental.
    """
    if not isinstance(mapping, Mapping):
        return Counter()
    diagnostics: Counter[str] = Counter()
    stats = _peripheral_input_stats(mapping)
    for key in ("rx_data_reads_without_ready", "rx_overlay_mismatches"):
        count = _positive_int(stats.get(key))
        if count:
            diagnostics[key] = count
    events = mapping.get("stream_input_summary_events")
    if isinstance(events, (list, tuple, set)):
        for item in events:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind not in {"mmio_stream_status", "mmio_stream_byte"}:
                continue
            reason = str(item.get("model_diagnostic") or "").strip()
            if reason:
                diagnostics[reason] += 1
    return diagnostics


def environment_input_facts(mapping: Any) -> Counter[str]:
    """Extract explicitly observed environmental values/events from a run.

    This function is intentionally independent of the intervention classifier.
    It records facts for audit and allows a typed stream summary to replace its
    corresponding generic skip count.  It never turns a missing execution
    result into a witness and it never accepts an unknown category.
    """
    if not isinstance(mapping, Mapping):
        return Counter()
    facts: Counter[str] = Counter()

    for key in (
        "environment_input_facts",
        "environment_event_facts",
        "environment_delivery_facts",
    ):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            for reason, count in value.items():
                _add_environment_fact(facts, reason, count)
        elif isinstance(value, (list, tuple, set)):
            for reason in value:
                _add_environment_fact(facts, reason)
        elif value not in (None, "", False):
            _add_environment_fact(facts, value)

    # Boolean/string fields are emitted by replay stages that deliver an event
    # after restoring a validated context snapshot.  They are facts, not
    # control-flow choices; cold-vector classification remains a separate
    # boundary check in classify_execution().
    for key, reason in (
        ("environment_input_consumed", "environment_input_delivery"),
        ("environment_input_delivered", "environment_input_delivery"),
        ("environment_event_delivery", "event_delivery"),
        ("interrupt_event_delivered", "interrupt_delivery"),
        ("irq_event_delivered", "interrupt_delivery"),
        ("task_event_delivered", "task_event_delivery"),
        ("dma_event_delivered", "dma_event_delivery"),
    ):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            # A string such as ``interrupt`` carries a more precise category.
            _add_environment_fact(facts, value, 1)
        elif coerce_bool(value, False):
            _add_environment_fact(facts, reason, 1)

    entry_derivation = str(mapping.get("entry_derivation") or "").lower()
    if coerce_bool(mapping.get("context_snapshot_mode"), False) and (
        "event_replay" in entry_derivation
        or "context_snapshot" in entry_derivation
    ):
        if (
            "isr" in entry_derivation
            or "interrupt" in entry_derivation
            or mapping.get("irq") is not None
            or mapping.get("isr_address") is not None
        ):
            _add_environment_fact(facts, "interrupt_delivery")
        elif "thread" in entry_derivation or "rtos" in entry_derivation:
            _add_environment_fact(facts, "task_event_delivery")
        else:
            _add_environment_fact(facts, "event_delivery")

    stream_events = mapping.get("stream_input_summary_events")
    successful_stream_events = _successful_stream_summary_count(stream_events)
    if successful_stream_events:
        facts["stream_input_delivery"] = max(
            int(facts.get("stream_input_delivery", 0) or 0),
            successful_stream_events,
        )

    stats = mapping.get("environment_input_delivery_stats")
    if isinstance(stats, Mapping):
        for key in ("accepted", "delivered", "successful"):
            accepted = _positive_int(stats.get(key))
            if accepted:
                facts["stream_input_delivery"] = max(
                    int(facts.get("stream_input_delivery", 0) or 0),
                    accepted,
                )
                break

    # UART/RX semantic profiles emit typed MMIO events rather than generic
    # function-summary events.  The ready observation and the subsequent data
    # consumption are both environmental facts, while consistency warnings are
    # kept separate by ``environment_model_diagnostics``.
    mmio_status_events = 0
    mmio_byte_events = 0
    if isinstance(stream_events, (list, tuple, set)):
        for item in stream_events:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("reason") or "").strip():
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind == "mmio_stream_status":
                mmio_status_events += 1
            elif kind == "mmio_stream_byte":
                mmio_byte_events += 1
    peripheral_stats = _peripheral_input_stats(mapping)
    ready_reads = max(
        mmio_status_events,
        _positive_int(peripheral_stats.get("rx_ready_status_reads")),
    )
    consumed_reads = max(
        mmio_byte_events,
        _positive_int(peripheral_stats.get("rx_bytes_consumed")),
        _positive_int(peripheral_stats.get("rx_data_reads")),
    )
    if ready_reads:
        facts["input_ready"] = max(
            int(facts.get("input_ready", 0) or 0),
            ready_reads,
        )
    if consumed_reads:
        facts["stream_input_delivery"] = max(
            int(facts.get("stream_input_delivery", 0) or 0),
            consumed_reads,
        )

    payload_writes = mapping.get("stream_input_payload_writes")
    if isinstance(payload_writes, (list, tuple, set)) and payload_writes:
        facts["external_memory_input"] = max(
            int(facts.get("external_memory_input", 0) or 0),
            len(payload_writes),
        )

    return facts


def _blocking_reason_counter(counter: Mapping[str, Any]) -> Counter[str]:
    """Remove only explicitly recognized environment facts from a counter."""
    result: Counter[str] = Counter()
    for reason, value in counter.items():
        if is_environment_fact_reason(reason):
            continue
        if str(reason) in _ENGINEERING_OPTIMIZATION_REASONS:
            # r9 用户裁定：工程优化/外部输入家族不作降级原因（覆盖照常计入）。
            # 计数保留在 run 结果的 intervention_event_labels 里供审计。
            continue
        count = _positive_int(value)
        if count:
            result[str(reason)] = count
    return result


def _subtract_environment_stream_summaries(
    counter: Counter[str],
    mapping: Mapping[str, Any],
) -> None:
    """Remove stream summaries that are backed by concrete input telemetry."""
    allowed = _successful_stream_summary_count(
        mapping.get("stream_input_summary_events")
    )
    stats = mapping.get("environment_input_delivery_stats")
    if isinstance(stats, Mapping):
        allowed = max(
            allowed,
            max(
                (
                    _positive_int(stats.get(key))
                    for key in ("accepted", "delivered", "successful")
                ),
                default=0,
            ),
        )
    allowed = max(
        allowed,
        _positive_int(mapping.get("stream_input_summary_applied_count")),
    )
    if allowed <= 0:
        return
    current = _positive_int(counter.get("function_summary_or_skip"))
    if current <= 0:
        return
    remaining = current - min(current, allowed)
    if remaining:
        counter["function_summary_or_skip"] = remaining
    else:
        counter.pop("function_summary_or_skip", None)


def _count_mapping(mapping: Any, keys: Iterable[str]) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    return max((_positive_int(mapping.get(key)) for key in keys), default=0)


def _add_max(counter: Counter[str], reason: str, value: Any) -> None:
    count = _positive_int(value)
    if count:
        counter[str(reason)] = max(int(counter.get(str(reason), 0) or 0), count)


def _add_explicit_reasons(counter: Counter[str], value: Any) -> None:
    """Merge a typed actual-intervention field, if a caller supplied one."""
    if isinstance(value, Mapping):
        for reason, count in value.items():
            _add_max(counter, str(reason), count)
    elif isinstance(value, (list, tuple, set)):
        for reason in value:
            if str(reason):
                _add_max(counter, str(reason), 1)


def _intervention_counter_from_mapping(mapping: Any) -> Counter[str]:
    """Return execution-changing facts actually observed in one run.

    Configuration and attempts are deliberately excluded.  For example,
    ``forced_branch_choices_configured`` says that a choice was installed, but
    it does not prove that the corresponding branch was encountered or that
    the hook changed the path.  The emulator's ``forced_branch_trace`` and
    applied counters are the facts that can make a replay diagnostic.
    """
    if not isinstance(mapping, Mapping):
        return Counter()

    # IntelligentEmulator.run() publishes a per-run authoritative summary.
    # Prefer it over cumulative compatibility counters below; otherwise a
    # prior replay on a reused emulator would contaminate the current witness.
    if coerce_bool(mapping.get("execution_intervention_reasons_authoritative"), False):
        authoritative: Counter[str] = Counter()
        _add_explicit_reasons(
            authoritative,
            mapping.get("execution_intervention_reasons"),
        )
        # Some adapters emit the authoritative result under the counter name
        # only.  It is still a per-run delta when the authoritative marker is
        # present; do not fall back to cumulative compatibility counters.
        if not authoritative:
            _add_explicit_reasons(
                authoritative,
                mapping.get("execution_intervention_counts"),
            )
        _subtract_environment_stream_summaries(authoritative, mapping)
        return _blocking_reason_counter(authoritative)

    counts: Counter[str] = Counter()
    scalar_rules = (
        ("forced_branch_trace_count", "forced_branch"),
        ("forced_branch_applied_count", "forced_branch"),
        ("forced_choices_applied", "forced_branch"),
        ("forced_control_count", "forced_branch"),
        ("control_forced_trace_count", "forced_branch"),
        ("loop_unresolved_limit_trips", "loop_unresolved_limit"),
        ("summary_return_count", "function_summary_or_skip"),
        ("summary_return_applied_count", "function_summary_or_skip"),
        ("external_rom_summary_count", "external_rom_summary"),
    )
    for key, reason in scalar_rules:
        _add_max(counts, reason, mapping.get(key))

    # r9 裁定：干预事件按家族标签记账时，loop_intervention 只保留未被家族
    # 覆盖的残差（本地约束/LLM/回退/未生效干预保持诊断口径）。无标签的旧
    # 生产者退回 intervention_count 整数（行为不变）。
    event_labels = mapping.get("intervention_event_labels")
    labeled: Counter[str] = Counter()
    if isinstance(event_labels, Mapping):
        for family, value in event_labels.items():
            count = _positive_int(value)
            if count:
                labeled[str(family)] += count
    raw_intervention_count = _positive_int(mapping.get("intervention_count"))
    if labeled:
        for family, count in labeled.items():
            _add_max(counts, family, count)
        residual = raw_intervention_count - sum(labeled.values())
        if residual > 0:
            _add_max(counts, "loop_intervention", residual)
    else:
        _add_max(counts, "loop_intervention", raw_intervention_count)

    trace = mapping.get("forced_branch_trace")
    if isinstance(trace, (list, tuple, set)) and trace:
        _add_max(counts, "forced_branch", len(trace))

    nested_rules = (
        ("runtime_loop_branch_force_stats", ("applied",), "runtime_loop_branch_force"),
        ("skip_function_stats", ("applied", "stateful_applied", "applied_total"), "function_summary_or_skip"),
        ("lzo_decompress_summary_stats", ("applied",), "function_summary_or_skip"),
        ("external_rom_call_stats", ("returned", "applied"), "external_rom_summary"),
        ("svc_stats", ("handled",), "svc_intervention"),
        ("thumb_state_guard_stats", ("repaired",), "thumb_state_repair"),
        ("invalid_thumb_recovery_stats", ("recovered",), "invalid_thumb_recovery"),
        ("thumb_indirect_branch_repair_stats", ("repaired",), "thumb_indirect_branch_repair"),
        ("execution_preflight_stats", ("thumb_state_corrected",), "execution_preflight_repair"),
        # r7 口径 2：循环体快进仿真单列计数（保持 diagnostic 干预口径）。
        ("loop_fast_forward_emulation", ("total",), "loop_fast_forward_emulation"),
    )
    for key, fields, reason in nested_rules:
        _add_max(counts, reason, _count_mapping(mapping.get(key), fields))

    # Newer adapters expose the already-delta-normalized counters separately.
    # Merge them after legacy fields so a typed reason is never lost.  This is
    # deliberately not used when the authoritative marker above is set.
    _add_explicit_reasons(counts, mapping.get("execution_intervention_counts"))

    # This field is reserved for actual facts.  Configuration summaries must
    # use ``configured_intervention_counts`` below instead.
    _add_explicit_reasons(counts, mapping.get("execution_intervention_reasons"))
    _subtract_environment_stream_summaries(counts, mapping)
    return _blocking_reason_counter(counts)


def configured_intervention_counts(mapping: Any) -> Counter[str]:
    """Return configured/attempted choices for diagnostics, never classification.

    Keeping this separate prevents a configured-but-unhit choice from being
    treated as a control-flow intervention.  Reports can still expose these
    counters for auditability.
    """
    if not isinstance(mapping, Mapping):
        return Counter()
    counts: Counter[str] = Counter()
    # New producers may already expose a normalized configuration map.  It is
    # retained for audit output only; this function is never used by the
    # actual-intervention classifier.
    _add_explicit_reasons(counts, mapping.get("configured_intervention_counts"))
    for key in (
        "forced_branch_choices_configured",
        "control_forced_choices_configured",
        "forced_choices_requested",
        "forced_directions",
        "forced_branch_edges",
    ):
        value = mapping.get(key)
        if isinstance(value, (list, tuple, set, Mapping)):
            value = len(value)
        _add_max(counts, "forced_branch", value)
    for key, fields, reason in (
        ("runtime_loop_branch_force_stats", ("installed",), "runtime_loop_branch_force"),
        ("skip_function_stats", ("installed",), "function_summary_or_skip"),
    ):
        _add_max(counts, reason, _count_mapping(mapping.get(key), fields))
    return counts


def execution_intervention_reasons(run_result: Any) -> list[str]:
    """Return actual non-environment intervention categories for one run."""
    if not isinstance(run_result, Mapping):
        return ["missing_run_result"]
    return list(_intervention_counter_from_mapping(run_result))


def execution_telemetry_complete(run_result: Any) -> bool:
    """Whether a run result claims to contain a complete execution fact set."""
    if not has_execution_facts(run_result):
        return False
    if isinstance(run_result, Mapping):
        # Error fields are authoritative even when a legacy producer forgot
        # to set ``execution_failed``.  A stop reason is not sufficient by
        # itself, but an explicit error is enough to reject the witness.
        if _execution_failure_reasons(run_result):
            return False
        if "execution_started" in run_result and not coerce_bool(
            run_result.get("execution_started"), False
        ):
            return False
        if coerce_bool(run_result.get("execution_failed"), False):
            return False
        if coerce_bool(run_result.get("preflight_failed"), False):
            return False
        if (
            "execution_completed_normally" in run_result
            and not coerce_bool(
                run_result.get("execution_completed_normally"), False
            )
        ):
            return False
    explicit = run_result.get("execution_telemetry_complete")
    if explicit is not None:
        return coerce_bool(explicit, False)
    telemetry = run_result.get("execution_telemetry")
    if isinstance(telemetry, Mapping) and telemetry.get("complete") is not None:
        return coerce_bool(telemetry.get("complete"), False)
    # A concrete stop reason without an explicit completeness bit is useful
    # for crash diagnosis, but it is not enough to establish a strict witness.
    # This prevents legacy/debug dictionaries from silently becoming valid
    # coverage merely because they contain ``stop_reason``.
    return False


def _execution_failure_reasons(mapping: Any) -> list[str]:
    """Extract runtime failures that invalidate a replay witness.

    A bounded timeout or an intentional quiescence stop is a complete concrete
    execution and is therefore not a failure here.  Exceptions, failed
    restoration, architecture preflight failures, and runtime crash monitor
    events are different: they mean Unicorn did not provide a valid suffix.
    """
    if not isinstance(mapping, Mapping):
        return []
    reasons: list[str] = []
    nested = mapping.get("run_result")
    if isinstance(nested, Mapping) and nested is not mapping:
        reasons.extend(_execution_failure_reasons(nested))
    for key, reason in (
        ("execution_failed", "execution_failed"),
        ("runtime_failure", "runtime_failure"),
        ("restore_failed", "restore_failed"),
        ("preflight_failed", "preflight_failed"),
    ):
        if key in mapping and coerce_bool(mapping.get(key), False):
            reasons.append(reason)

    stop_reason = str(mapping.get("stop_reason") or "").strip().lower()
    if stop_reason:
        failure_tokens = (
            "exception",
            "restore_failed",
            "preflight_failed",
            "runtime_crash",
            "segmentation fault",
            "invalid memory",
            "unmapped",
            "native_crash",
            "emulator_create_exception",
            "replay_setup_exception",
            "execution_setup_exception",
            "snapshot_restore_failed",
            "isr_start_failed",
            "hook_installation_failed",
        )
        if any(token in stop_reason for token in failure_tokens):
            reasons.append(f"execution_failure:{stop_reason[:160]}")
    for key in (
        "exception",
        "error",
        "crash_error",
        "execution_error",
        "failure_reason",
        "restore_error",
    ):
        value = mapping.get(key)
        if value not in (None, "", False):
            # ``error`` is only a failure marker when the producer did not
            # explicitly mark it as informational.
            if key != "error" or coerce_bool(mapping.get("execution_failed"), True):
                reasons.append(f"execution_failure:{str(value)[:160]}")
    for key in ("replay_setup_errors", "hook_installation_errors"):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            if value:
                reasons.append(f"{key}_observed")
        elif isinstance(value, (list, tuple, set)):
            if value:
                reasons.append(f"{key}_observed")
        elif isinstance(value, str) and value.strip():
            reasons.append(f"{key}_observed")
    return list(dict.fromkeys(reasons))


def _provenance_reasons(mapping: Any) -> list[str]:
    if not isinstance(mapping, Mapping):
        return []
    reasons: list[str] = []

    def reason_values(source: Mapping[str, Any]) -> tuple[str, ...]:
        """Collect explicit provenance reasons without trusting field names.

        A diagnostic provenance status is promotable only when every explicit
        reason is a recognized environment fact.  Collect all reason-bearing
        fields here rather than selecting the first non-empty field: a wrapper
        may put an environmental fact in ``reasons`` and a blocking control
        intervention in ``prefix_intervention_reasons``.
        """
        values: list[str] = []
        for field_name in (
            "reasons",
            "provenance_reasons",
            "intervention_reasons",
            "actual_intervention_reasons",
            "execution_intervention_counts",
            "prefix_intervention_reasons",
        ):
            value = source.get(field_name)
            if isinstance(value, Mapping):
                for item, count in value.items():
                    if _positive_int(count) and str(item):
                        values.append(str(item))
            elif isinstance(value, (list, tuple, set)):
                values.extend(str(item) for item in value if str(item))
            elif isinstance(value, str) and value.strip():
                values.append(value.strip())
        return tuple(dict.fromkeys(values))

    def invalidation_values(source: Mapping[str, Any]) -> tuple[str, ...]:
        values: list[str] = []
        raw = source.get("provenance_invalidation_reasons")
        if isinstance(raw, str):
            values.append(raw.strip())
        elif isinstance(raw, (list, tuple, set)):
            values.extend(str(item) for item in raw if str(item))
        elif isinstance(raw, Mapping):
            values.extend(str(item) for item, count in raw.items() if count and str(item))
        if coerce_bool(source.get("provenance_invalidated"), False):
            values.append("provenance_invalidated")
        return tuple(dict.fromkeys(item for item in values if item))

    def diagnostic_status_is_environment_only(
        source: Mapping[str, Any],
        status: str,
    ) -> bool:
        explicit = reason_values(source)
        return bool(explicit) and all(
            is_environment_fact_reason(item) for item in explicit
        )

    for key in (
        "entry_provenance_status",
        "prefix_provenance_status",
        "snapshot_provenance_status",
        "context_provenance_status",
        "provenance_status",
    ):
        if key not in mapping:
            continue
        value = str(mapping.get(key) or "").strip().lower()
        if value in {
            "diagnostic",
            "pending",
            "unverified",
            "unknown",
            "unspecified",
            "incomplete",
            "invalid",
            "",
        }:
            if value != "diagnostic" or not diagnostic_status_is_environment_only(
                mapping,
                value,
            ):
                reasons.append(f"{key}:{value}")

    for item in invalidation_values(mapping):
        reasons.append(str(item))

    for key in (
        "snapshot_provenance",
        "prefix_provenance",
        "context_provenance",
        "execution_provenance",
        "provenance",
    ):
        value = mapping.get(key)
        if not isinstance(value, Mapping):
            continue
        status = str(
            value.get("status")
            or value.get("classification")
            or value.get("provenance_status")
            or ""
        ).strip().lower()
        if status in {
            "diagnostic",
            "pending",
            "unverified",
            "unknown",
            "unspecified",
            "incomplete",
            "invalid",
            "",
        }:
            if status != "diagnostic" or not diagnostic_status_is_environment_only(
                value,
                status,
            ):
                reasons.append(f"{key}:{status}")
        nested_reasons = reason_values(value)
        if any(not is_environment_fact_reason(item) for item in nested_reasons):
            reasons.append(f"{key}:ancestor_intervention")
        for item in invalidation_values(value):
            reasons.append(f"{key}:{item}")
        if "telemetry_complete" in value and not coerce_bool(
            value.get("telemetry_complete"), False
        ):
            reasons.append(f"{key}:telemetry_incomplete")
        if "provenance_finalized" in value and not coerce_bool(
            value.get("provenance_finalized"), False
        ):
            reasons.append(f"{key}:provenance_pending_finalization")

    if "prefix_telemetry_complete" in mapping and not coerce_bool(
        mapping.get("prefix_telemetry_complete"), False
    ):
        reasons.append("prefix_telemetry_incomplete")
    if "snapshot_provenance_complete" in mapping and not coerce_bool(
        mapping.get("snapshot_provenance_complete"), False
    ):
        reasons.append("snapshot_provenance_incomplete")
    if "snapshot_ancestor_validated" in mapping and not coerce_bool(
        mapping.get("snapshot_ancestor_validated"), False
    ):
        reasons.append("snapshot_ancestor_not_validated")
    if "provenance_finalized" in mapping and not coerce_bool(
        mapping.get("provenance_finalized"), False
    ):
        reasons.append("provenance_pending_finalization")

    prefix_reasons = mapping.get("prefix_intervention_reasons")
    if isinstance(prefix_reasons, Mapping):
        if any(
            _positive_int(item)
            for reason, item in prefix_reasons.items()
            if not is_environment_fact_reason(reason)
        ):
            reasons.append("prefix_ancestor_intervention")
    elif isinstance(prefix_reasons, (list, tuple, set)) and any(
        not is_environment_fact_reason(item) for item in prefix_reasons
    ):
        reasons.append("prefix_ancestor_intervention")
    elif (
        isinstance(prefix_reasons, str)
        and prefix_reasons.strip()
        and not is_environment_fact_reason(prefix_reasons)
    ):
        reasons.append("prefix_ancestor_intervention")
    return list(dict.fromkeys(reasons))


def _context_snapshot_provenance_reasons(mapping: Any) -> list[str]:
    """Require an observed, finalized prefix for context/event replays.

    ``context_snapshot_mode`` is a scheduling annotation, not proof that a
    real snapshot was restored.  This check is intentionally applied only to a
    concrete execution result; phase-level summaries still use their child
    records as the source of truth.
    """
    if not isinstance(mapping, Mapping) or not coerce_bool(
        mapping.get("context_snapshot_mode"), False
    ):
        return []

    nested_values = [
        mapping.get("snapshot_provenance"),
        mapping.get("prefix_provenance"),
        mapping.get("execution_provenance"),
    ]
    nested = [item for item in nested_values if isinstance(item, Mapping)]

    def first_value(*keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping.get(key) not in (None, ""):
                return mapping.get(key)
        for item in nested:
            for key in keys:
                if key in item and item.get(key) not in (None, ""):
                    return item.get(key)
        return None

    reasons: list[str] = []
    ancestor = first_value("snapshot_ancestor_validated")
    if not coerce_bool(ancestor, False):
        reasons.append("context_snapshot_ancestor_not_validated")

    status = _normalized_provenance_status(
        first_value(
            "prefix_provenance_status",
            "snapshot_provenance_status",
            "provenance_status",
            "status",
        )
    )
    if status != "validated":
        reasons.append("context_snapshot_provenance_unvalidated")

    telemetry = first_value("prefix_telemetry_complete", "telemetry_complete")
    if not coerce_bool(telemetry, False):
        reasons.append("context_snapshot_prefix_telemetry_incomplete")

    finalized = first_value("provenance_finalized")
    if not coerce_bool(finalized, False):
        reasons.append("context_snapshot_provenance_not_finalized")
    return list(dict.fromkeys(reasons))


def classify_execution(
    run_result: Any,
    *,
    metadata: Any = None,
) -> tuple[str, list[str]]:
    """Classify one execution witness using actual facts and prefix lineage."""
    if not isinstance(run_result, Mapping):
        return DIAGNOSTIC_REPLAY, ["missing_run_result"]

    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    reasons: list[str] = []
    facts_available = has_execution_facts(run_result)
    # Extract typed execution-changing facts before checking whether the
    # producer supplied the complete fact bundle.  A setup record can lack a
    # register/trace payload and still prove that a forced branch or summary
    # shortcut was applied; that attempt is diagnostic, not silently
    # unclassified.  Conversely, a record containing only configuration
    # counters remains an evidence gap.
    actual_interventions = execution_intervention_reasons(run_result)
    if not facts_available:
        # A constructor/setup failure can create a scheduler record without
        # crossing the native Unicorn boundary.  That is an evidence gap, not
        # a diagnostic execution witness.
        reasons.extend(actual_interventions)
        reasons.extend(_execution_failure_reasons(run_result))
        reasons.extend(_provenance_reasons(run_result))
        reasons.extend(_provenance_reasons(metadata_map))
        reasons.append("missing_execution_facts")
        if coerce_bool(run_result.get("execution_attempted"), False):
            reasons.append("execution_not_started")
        normalized_reasons = list(dict.fromkeys(reasons))
        # Known execution-changing or setup/provenance failures are
        # diagnostic observations.  A bare scheduler/configuration record is
        # still unclassified because no concrete execution occurred.
        known_diagnostic = bool(
            actual_interventions
            or _execution_failure_reasons(run_result)
            or _provenance_reasons(run_result)
            or _provenance_reasons(metadata_map)
        )
        return (
            DIAGNOSTIC_REPLAY if known_diagnostic else UNCLASSIFIED_REPLAY,
            normalized_reasons,
        )
    reasons.extend(actual_interventions)
    reasons.extend(_execution_failure_reasons(run_result))
    reasons.extend(_provenance_reasons(run_result))
    reasons.extend(_provenance_reasons(metadata_map))
    # Context metadata and execution provenance are emitted by different
    # layers: the runner owns ``context_snapshot_mode`` while the emulator
    # owns the finalized execution-provenance payload.  Validate the merged
    # view so a correct child is not rejected merely because either layer does
    # not duplicate the other's fields.
    context_view = dict(run_result)
    context_view.update(metadata_map)
    for key in (
        "execution_provenance",
        "snapshot_provenance",
        "prefix_provenance",
    ):
        if key in run_result and key not in metadata_map:
            context_view[key] = run_result.get(key)
    reasons.extend(_context_snapshot_provenance_reasons(context_view))

    if not execution_telemetry_complete(run_result):
        reasons.append("incomplete_execution_telemetry")
    if coerce_bool(metadata_map.get("paired_validation_required"), False):
        if not coerce_bool(metadata_map.get("paired_validation_complete"), False):
            reasons.append("paired_validation_incomplete")
        if not coerce_bool(metadata_map.get("paired_initial_state_same"), False):
            reasons.append("initial_state_fingerprint_not_equivalent")
    if "coverage_counted" in metadata_map and not coerce_bool(
        metadata_map.get("coverage_counted"), True
    ):
        reasons.append("coverage_not_counted")

    entry_derivation = str(metadata_map.get("entry_derivation") or "").lower()
    if "cold_vector" in entry_derivation:
        reasons.append("cold_vector_entry")
    if (
        "isr" in entry_derivation
        and "context_snapshot_mode" in metadata_map
        and not coerce_bool(metadata_map.get("context_snapshot_mode"), False)
    ):
        reasons.append("isr_without_context_snapshot")

    return (
        (DIAGNOSTIC_REPLAY, list(dict.fromkeys(reasons)))
        if reasons
        else (VALIDATED_REPLAY, [])
    )


def _coerce_address(value: Any) -> int | None:
    """Parse one BB-like value without allowing malformed telemetry to abort."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value) & 0xFFFFFFFF
    if isinstance(value, float):
        if value.is_integer():
            return int(value) & 0xFFFFFFFF
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0) & 0xFFFFFFFF
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def safe_basic_block_list(value: Any) -> list[int]:
    """Return normalized BB addresses from tolerant, JSON-like telemetry.

    Coverage producers use both integer addresses and strings such as
    ``"0x08001234"``.  A single malformed entry must be ignored rather than
    discarding the rest of a run or raising while a long campaign is being
    serialized.  Mapping entries are accepted for common trace formats, but
    arbitrary mappings are never interpreted as an address.
    """
    values: set[int] = set()

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 3 or item is None:
            return
        address = _coerce_address(item)
        if address is not None:
            values.add(address)
            return
        if isinstance(item, Mapping):
            for key in (
                "covered_bb_list",
                "covered_bbs",
                "bb",
                "address",
                "pc",
                "start",
            ):
                if key in item:
                    visit(item.get(key), depth + 1)
                    return
            return
        if isinstance(item, (str, bytes, bytearray)):
            return
        try:
            iterator = iter(item)
        except TypeError:
            return
        for child in iterator:
            visit(child, depth + 1)

    visit(value)
    return sorted(values)


def _covered_bb_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("covered_bb_list", value.get("covered_bbs"))
    return safe_basic_block_list(value)


def build_execution_evidence_record(
    *,
    phase_name: object,
    run_result: Any,
    covered_bbs: Any = None,
    metadata: Any = None,
    execution_id: object = None,
) -> dict[str, object]:
    """Create a compact JSON-safe witness record for report aggregation."""
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    status, reasons = classify_execution(run_result, metadata=metadata_map)
    actual = _intervention_counter_from_mapping(run_result)
    configured = configured_intervention_counts(run_result)
    if not configured:
        configured = configured_intervention_counts(metadata_map)
    environment = environment_input_facts(run_result)
    for reason, count in environment_input_facts(metadata_map).items():
        environment[reason] = max(
            int(environment.get(reason, 0) or 0),
            int(count or 0),
        )
    selected_bbs = _covered_bb_list(covered_bbs)
    if not selected_bbs and isinstance(run_result, Mapping):
        selected_bbs = _covered_bb_list(run_result.get("covered_bb_list"))
    provenance = (
        run_result.get("execution_provenance")
        if isinstance(run_result, Mapping)
        else None
    )
    if not isinstance(provenance, Mapping):
        provenance = None
    provenance_invalidated = bool(
        isinstance(run_result, Mapping)
        and coerce_bool(run_result.get("provenance_invalidated"), False)
    ) or bool(
        provenance is not None
        and coerce_bool(provenance.get("provenance_invalidated"), False)
    )

    def bounded_invalidation_reasons(source: Any) -> list[str]:
        if not isinstance(source, Mapping):
            return []
        value = source.get("provenance_invalidation_reasons", [])
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Mapping):
            values = [key for key, count in value.items() if count]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = []
        return list(dict.fromkeys(str(item) for item in values if str(item)))[:16]

    provenance_invalidation_reasons = bounded_invalidation_reasons(run_result)
    if provenance is not None:
        provenance_invalidation_reasons = list(
            dict.fromkeys(
                provenance_invalidation_reasons
                + bounded_invalidation_reasons(provenance)
            )
        )[:16]
    record = {
        "schema": EVIDENCE_SCHEMA,
        "phase": str(phase_name or "unknown"),
        "execution_id": str(execution_id or ""),
        "status": status,
        "reasons": reasons,
        "covered_bbs": selected_bbs,
        "telemetry_complete": execution_telemetry_complete(run_result),
        "execution_facts_available": has_execution_facts(run_result),
        "execution_failure_reasons": _execution_failure_reasons(run_result),
        "actual_intervention_reasons": dict(actual),
        "environment_facts": dict(environment),
        "environment_input_facts": dict(environment),
        "environment_model_diagnostics": dict(
            environment_model_diagnostics(run_result)
        ),
        "configured_intervention_counts": dict(configured),
        "prefix_provenance_status": (
            run_result.get("prefix_provenance_status")
            if isinstance(run_result, Mapping)
            else None
        ),
        "prefix_intervention_reasons": (
            list(run_result.get("prefix_intervention_reasons", []) or [])
            if isinstance(run_result, Mapping)
            else []
        ),
        "execution_provenance_status": (
            provenance.get("status") if provenance is not None else None
        ),
        "execution_provenance_reasons": (
            list(provenance.get("reasons", []) or [])
            if provenance is not None
            else []
        ),
        "provenance_invalidated": provenance_invalidated,
        "provenance_invalidation_reasons": provenance_invalidation_reasons,
    }
    if provenance is not None:
        # Keep a bounded provenance view in the compact ledger.  The full
        # execution trace remains outside this record, but these fields are
        # required to re-check a context replay after the run result is gone.
        record["execution_provenance"] = {
            key: provenance.get(key)
            for key in (
                "status",
                "reasons",
                "intervention_reasons",
                "prefix_intervention_reasons",
                "telemetry_complete",
                "prefix_telemetry_complete",
                "provenance_finalized",
                "snapshot_ancestor_validated",
                "prefix_provenance_status",
                "snapshot_source_execution_id",
                "execution_id",
                "provenance_invalidated",
                "provenance_invalidation_reasons",
            )
            if key in provenance
        }
        for key in (
            "snapshot_ancestor_validated",
            "prefix_telemetry_complete",
            "provenance_finalized",
        ):
            if key in provenance:
                record[key] = provenance.get(key)
    # Keep the small set of facts that lets a report audit a failed or direct
    # replay without retaining the complete run result.  In particular,
    # ``stop_reason`` alone is never used as proof, but it is indispensable for
    # explaining why a materialized attempt was rejected.
    if isinstance(run_result, Mapping):
        for key in (
            "execution_attempted",
            "execution_started",
            "execution_completed_normally",
            "execution_failed",
            "preflight_failed",
            "runtime_failure",
            "restore_failed",
            "stop_reason",
            "execution_error",
            "execution_failure_reasons",
            "provenance_finalized",
            "provenance_invalidated",
            "provenance_invalidation_reasons",
            "provenance_invalidation_error",
            "snapshot_provenance_invalidated_snapshots",
            "prefix_telemetry_complete",
            "snapshot_ancestor_validated",
            "direct_finalized_snapshots",
            "direct_finalization_error",
            "direct_finalization_errors",
            "direct_execution_finalization",
        ):
            if key in run_result:
                value = run_result.get(key)
                if key == "execution_error" and value not in (None, ""):
                    value = str(value)[:2000]
                elif key == "provenance_invalidation_error" and value not in (None, ""):
                    value = str(value)[:2000]
                elif key == "execution_failure_reasons":
                    value = [
                        str(item)[:512]
                        for item in list(value or [])
                        if str(item)
                    ][:32]
                elif key == "provenance_invalidation_reasons":
                    value = bounded_invalidation_reasons(
                        {key: value}
                    )
                elif key == "direct_execution_finalization":
                    if isinstance(value, Mapping):
                        value = {
                            str(item_key): item_value
                            for item_key, item_value in value.items()
                            if str(item_key)
                            in {
                                "execution_id",
                                "finalized_snapshots",
                                "finalization_error",
                                "finalization_errors",
                                "actual_intervention_reasons",
                                "provenance_invalidated",
                                "provenance_invalidation_reasons",
                                "invalidated_snapshots",
                            }
                        }
                        if "finalization_error" in value:
                            value["finalization_error"] = str(
                                value["finalization_error"] or ""
                            )[:2000]
                        if "finalization_errors" in value:
                            value["finalization_errors"] = [
                                str(item)[:512]
                                for item in list(value.get("finalization_errors", []) or [])
                                if str(item)
                            ][:16]
                        if "provenance_invalidation_reasons" in value:
                            value["provenance_invalidation_reasons"] = (
                                bounded_invalidation_reasons(value)
                            )
                        if "invalidated_snapshots" in value:
                            value["invalidated_snapshots"] = _positive_int(
                                value.get("invalidated_snapshots")
                            )
                    else:
                        value = {}
                record[key] = value
        for source_map in (run_result, metadata_map):
            if not isinstance(source_map, Mapping):
                continue
            for source_key in ("snapshot_provenance", "prefix_provenance"):
                source_value = source_map.get(source_key)
                if not isinstance(source_value, Mapping):
                    continue
                record[source_key] = {
                    key: source_value.get(key)
                    for key in (
                        "status",
                        "reasons",
                        "prefix_intervention_reasons",
                        "telemetry_complete",
                        "prefix_telemetry_complete",
                        "provenance_finalized",
                        "snapshot_ancestor_validated",
                        "execution_id",
                        "provenance_invalidated",
                        "provenance_invalidation_reasons",
                    )
                    if key in source_value
                }
        if isinstance(run_result.get("peripheral_input_stats"), Mapping):
            record["peripheral_input_stats"] = {
                str(key): _positive_int(value)
                for key, value in run_result["peripheral_input_stats"].items()
            }
        elif isinstance(metadata_map.get("peripheral_input_stats"), Mapping):
            record["peripheral_input_stats"] = {
                str(key): _positive_int(value)
                for key, value in metadata_map["peripheral_input_stats"].items()
            }
        for source_key in (
            "peripheral_input_stats_source",
            "direct_peripheral_input_stats_source",
        ):
            source_value = run_result.get(source_key)
            if source_value not in (None, ""):
                record[source_key] = str(source_value)[:128]
        if isinstance(run_result.get("direct_peripheral_input_stats"), Mapping):
            record["direct_peripheral_input_stats"] = {
                str(key): _positive_int(value)
                for key, value in run_result["direct_peripheral_input_stats"].items()
            }
        if isinstance(run_result.get("environment_model_diagnostics"), Mapping):
            record["environment_model_diagnostics"] = {
                str(key): _positive_int(value)
                for key, value in run_result["environment_model_diagnostics"].items()
            }
    # These fields describe the replay boundary, rather than the execution
    # itself.  Copy only scalar/short values so a debug payload cannot turn the
    # canonical ledger into an unbounded second trace.
    for key in (
        "entry_derivation",
        "context_snapshot_mode",
        "counterfactual_control",
        "coverage_counted",
        "candidate_consumed",
        "source",
        "coverage_scope",
        "environment_input_consumed",
        "environment_input_delivered",
        "environment_event_delivery",
        "interrupt_event_delivered",
        "irq_event_delivered",
        "task_event_delivered",
        "dma_event_delivered",
        "environment_input_source",
        "environment_event_source",
        "replay_setup_errors",
        "paired_validation_required",
        "paired_validation_complete",
        "paired_initial_state_same",
        "paired_validation_reason",
        "snapshot_identity",
        "snapshot_origin",
        "snapshot_address",
        "snapshot_order",
        "snapshot_source_execution_id",
        "restore_error",
        "prefix_telemetry_complete",
        "snapshot_ancestor_validated",
        "provenance_finalized",
        "provenance_invalidated",
        "provenance_invalidation_reasons",
        "direct_execution_finalization",
    ):
        if key in metadata_map:
            value = metadata_map.get(key)
            if isinstance(value, (str, bool, int, float)) or value is None:
                record[key] = value
    return record


def execution_intervention_reasons_from_emulator(emulator: Any) -> list[str]:
    """Classify a manually driven emulator execution using actual counters."""
    if emulator is None:
        return ["missing_run_result"]
    result = {
        "intervention_count": getattr(emulator, "intervention_count", 0),
        "forced_branch_trace": getattr(emulator, "forced_branch_trace", ()) or (),
        "forced_branch_trace_count": len(
            getattr(emulator, "forced_branch_trace", ()) or ()
        ),
        "runtime_loop_branch_force_stats": getattr(
            emulator, "runtime_loop_branch_force_stats", {}
        ),
        "skip_function_stats": getattr(emulator, "skip_function_stats", {}),
        "lzo_decompress_summary_stats": getattr(
            emulator, "lzo_decompress_summary_stats", {}
        ),
        "external_rom_call_stats": getattr(emulator, "external_rom_call_stats", {}),
        "svc_stats": getattr(emulator, "svc_stats", {}),
        "thumb_state_guard_stats": getattr(emulator, "thumb_state_guard_stats", {}),
        "invalid_thumb_recovery_stats": getattr(
            emulator, "invalid_thumb_recovery_stats", {}
        ),
        "thumb_indirect_branch_repair_stats": getattr(
            emulator, "thumb_indirect_branch_repair_stats", {}
        ),
        "execution_preflight_stats": getattr(
            emulator, "execution_preflight_stats", {}
        ),
        "environment_input_delivery_stats": getattr(
            emulator, "environment_input_delivery_stats", {}
        ),
    }
    return execution_intervention_reasons(result)


def execution_intervention_counts_from_emulator(
    emulator: Any,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Return typed intervention counts observed on a manually driven run.

    Direct ISR replays use ``emu_start`` rather than ``IntelligentEmulator.run``
    and therefore cannot rely on the latter's per-run delta record.  When a
    caller supplies a pre-execution ``baseline`` this function computes a
    per-attempt delta; without one it retains the legacy cumulative behavior
    for compatibility with external callers.
    """
    if emulator is None:
        return {"missing_run_result": 1}
    result = {
        "intervention_count": getattr(emulator, "intervention_count", 0),
        "forced_branch_trace": getattr(emulator, "forced_branch_trace", ()) or (),
        "forced_branch_trace_count": len(
            getattr(emulator, "forced_branch_trace", ()) or ()
        ),
        "runtime_loop_branch_force_stats": getattr(
            emulator, "runtime_loop_branch_force_stats", {}
        ),
        "skip_function_stats": getattr(emulator, "skip_function_stats", {}),
        "lzo_decompress_summary_stats": getattr(
            emulator, "lzo_decompress_summary_stats", {}
        ),
        "external_rom_call_stats": getattr(emulator, "external_rom_call_stats", {}),
        "svc_stats": getattr(emulator, "svc_stats", {}),
        "thumb_state_guard_stats": getattr(emulator, "thumb_state_guard_stats", {}),
        "invalid_thumb_recovery_stats": getattr(
            emulator, "invalid_thumb_recovery_stats", {}
        ),
        "thumb_indirect_branch_repair_stats": getattr(
            emulator, "thumb_indirect_branch_repair_stats", {}
        ),
        "execution_preflight_stats": getattr(
            emulator, "execution_preflight_stats", {}
        ),
        "environment_input_delivery_stats": getattr(
            emulator, "environment_input_delivery_stats", {}
        ),
    }
    if not isinstance(baseline, Mapping):
        return dict(_intervention_counter_from_mapping(result))

    current_snapshot = None
    snapshot_reader = getattr(emulator, "_execution_counter_snapshot", None)
    if callable(snapshot_reader):
        try:
            current_snapshot = snapshot_reader()
        except Exception:
            current_snapshot = None
    baseline_snapshot = baseline.get("counter_snapshot")
    if not isinstance(baseline_snapshot, Mapping):
        baseline_snapshot = baseline

    if isinstance(current_snapshot, Mapping):
        delta_builder = getattr(emulator, "_execution_counter_delta", None)
        if callable(delta_builder):
            try:
                delta = delta_builder(
                    dict(current_snapshot),
                    dict(baseline_snapshot),
                )
            except Exception:
                delta = {}
        else:
            delta = {}
            for key, value in current_snapshot.items():
                old = baseline_snapshot.get(key, 0)
                if isinstance(value, Mapping):
                    old_map = old if isinstance(old, Mapping) else {}
                    nested = {}
                    for nested_key, nested_value in value.items():
                        try:
                            amount = int(nested_value or 0) - int(
                                old_map.get(nested_key, 0) or 0
                            )
                        except (TypeError, ValueError, OverflowError):
                            amount = 0
                        if amount > 0:
                            nested[str(nested_key)] = amount
                    delta[str(key)] = nested
                else:
                    try:
                        amount = int(value or 0) - int(old or 0)
                    except (TypeError, ValueError, OverflowError):
                        amount = 0
                    delta[str(key)] = max(0, amount)
    else:
        # A test double or older adapter may not expose the compact counter
        # snapshot API.  In that case the explicit baseline fields still let
        # us isolate the forced-branch trace; other counters remain best-effort.
        delta = dict(result)

    try:
        trace_start = max(
            0,
            int(
                baseline.get(
                    "forced_trace_len",
                    baseline_snapshot.get("forced_branch_trace_count", 0),
                )
                or 0
            ),
        )
    except (TypeError, ValueError, OverflowError):
        trace_start = 0
    trace = list(getattr(emulator, "forced_branch_trace", ()) or ())
    if trace[trace_start:]:
        delta["forced_branch_trace"] = trace[trace_start:]
    return dict(_intervention_counter_from_mapping(delta))


def summarize_child_execution(
    run_results: Iterable[Any] | None,
    *,
    expected_count: int | None = None,
) -> dict[str, object]:
    """Build a safe aggregate for a phase composed of child replays.

    Missing child results are explicit.  The old implementation compared a
    counter to itself at the call site, which made every aggregate appear
    complete even when a child failed before returning a result.
    """
    results = list(run_results or ())
    expected = (
        _positive_int(expected_count)
        if expected_count is not None
        else len(results)
    )
    reasons: Counter[str] = Counter()
    configured: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    completed = 0
    incomplete = 0
    successful = 0
    failed = 0
    diagnostic = 0
    unclassified = 0
    validated = 0
    execution_facts = 0
    missing_result_slots = 0
    validated_bbs: set[int] = set()
    diagnostic_bbs: set[int] = set()
    unclassified_bbs: set[int] = set()

    for index, item in enumerate(results):
        item_map = item if isinstance(item, Mapping) else {}

        # A compact record is already classified.  It must not be fed back
        # through ``classify_execution`` because doing so would mistake its
        # bookkeeping fields for a fresh Unicorn run.  An unverified compact
        # record is downgraded rather than trusted.
        if is_compact_evidence_record(item_map):
            record = dict(item_map)
            record.setdefault("phase", "child")
            record.setdefault("execution_id", f"child-{index}")
            record.setdefault("covered_bbs", [])
            record.setdefault("reasons", [])
            facts_available = coerce_bool(
                record.get("execution_facts_available"), False
            )
            telemetry_complete = coerce_bool(
                record.get("telemetry_complete"), False
            )
            status = str(record.get("status") or UNCLASSIFIED_REPLAY)
            if status == VALIDATED_REPLAY and not compact_record_is_verified_validated(
                record
            ):
                status = UNCLASSIFIED_REPLAY
                record["status"] = status
                record["reasons"] = list(record.get("reasons", []) or []) + [
                    "compact_record_not_verified"
                ]
            records.append(record)
        else:
            nested = item_map.get("run_result")
            run_result = nested if isinstance(nested, Mapping) else item
            if is_materialized_execution_attempt(run_result):
                reasons.update(execution_intervention_reasons(run_result))
                configured.update(configured_intervention_counts(run_result))
                record = build_execution_evidence_record(
                    phase_name="child",
                    run_result=run_result,
                    covered_bbs=(
                        item_map.get("covered_bb_list", item_map.get("covered_bbs"))
                        if isinstance(item, Mapping)
                        else None
                    ),
                    metadata=item_map if isinstance(item, Mapping) else None,
                    execution_id=f"child-{index}",
                )
                records.append(record)
            else:
                # Preserve an explicit missing child slot.  This makes a
                # missing worker result visible to the phase classifier while
                # keeping its BB set empty.
                records.append({
                    "schema": EVIDENCE_SCHEMA,
                    "phase": "child",
                    "execution_id": f"child-{index}",
                    "status": UNCLASSIFIED_REPLAY,
                    "reasons": ["missing_run_result"],
                    "covered_bbs": [],
                    "telemetry_complete": False,
                    "execution_facts_available": False,
                })
                missing_result_slots += 1

        record = records[-1]
        record_status = str(record.get("status") or UNCLASSIFIED_REPLAY)
        facts_available = coerce_bool(
            record.get("execution_facts_available"), False
        )
        telemetry_complete = coerce_bool(
            record.get("telemetry_complete"), False
        )
        failure_reasons = list(record.get("execution_failure_reasons", []) or [])
        explicit_failed = any(
            coerce_bool(record.get(key), False)
            for key in (
                "execution_failed",
                "preflight_failed",
                "runtime_failure",
                "restore_failed",
            )
        ) or any(
            record.get(key) not in (None, "", False)
            for key in ("execution_error", "failure_reason", "restore_error")
        )
        if is_materialized_execution_attempt(item_map):
            completed += 1
        elif is_compact_evidence_record(item_map) and (
            facts_available or failure_reasons or explicit_failed
        ):
            completed += 1
        if facts_available:
            execution_facts += 1
        if not (facts_available and telemetry_complete):
            incomplete += 1
        if failure_reasons or explicit_failed:
            failed += 1
        elif facts_available and telemetry_complete:
            successful += 1
        covered = set(_covered_bb_list(record.get("covered_bbs", [])))
        if record_status == VALIDATED_REPLAY:
            validated += 1
            validated_bbs.update(covered)
        elif record_status == DIAGNOSTIC_REPLAY:
            diagnostic += 1
            diagnostic_bbs.update(covered)
        else:
            unclassified += 1
            unclassified_bbs.update(covered)
        for reason in list(record.get("reasons", []) or []):
            if str(reason):
                reasons[str(reason)] += 1
        configured.update(
            configured_intervention_counts(
                item_map if is_compact_evidence_record(item_map) else {}
            )
        )

    # A failed attempt is a materialized child record, not a missing worker
    # result.  Keep the two failure modes separate so phase reports do not
    # claim that an observed setup/runtime failure was silently dropped.
    materialized = len(records)
    unreported_slots = max(0, expected - len(results))
    missing = missing_result_slots + unreported_slots
    incomplete += unreported_slots
    complete = (
        missing == 0
        and len(results) >= expected
        and len(records) >= expected
        and incomplete == 0
        and all(
            coerce_bool(record.get("execution_facts_available"), False)
            and coerce_bool(record.get("telemetry_complete"), False)
            and not any(
                coerce_bool(record.get(key), False)
                for key in (
                    "execution_failed",
                    "runtime_failure",
                    "restore_failed",
                    "preflight_failed",
                )
            )
            and not any(
                record.get(key) not in (None, "", False)
                for key in (
                    "execution_error",
                    "failure_reason",
                    "restore_error",
                )
            )
            for record in records
        )
    )
    if not complete:
        reasons["incomplete_child_execution_telemetry"] += missing or 1
    return {
        "child_replay_count": len(results),
        "child_attempt_count": len(results),
        "child_completed_with_result_count": completed,
        "child_successful_execution_count": successful,
        "child_failure_count": failed,
        "child_validated_execution_count": validated,
        "child_diagnostic_execution_count": diagnostic,
        "child_unclassified_execution_count": unclassified,
        "child_execution_facts_count": execution_facts,
        "child_materialized_record_count": materialized,
        "child_missing_result_count": missing,
        "child_incomplete_result_count": incomplete,
        "expected_child_replay_count": expected,
        "execution_telemetry_complete": bool(complete),
        "execution_records_truncated": bool(len(results) < expected),
        "intervention_reasons": dict(reasons),
        "configured_intervention_counts": dict(configured),
        "validated_bbs": sorted(validated_bbs),
        "diagnostic_bbs": sorted(diagnostic_bbs),
        "unclassified_bbs": sorted(unclassified_bbs),
        "child_execution_records": records,
    }


def _phase_child_records(metadata: Mapping[str, Any]) -> list[Any]:
    child_summary = metadata.get("child_execution_summary")
    if isinstance(child_summary, Mapping):
        value = child_summary.get("child_execution_records")
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    for key in (
        "execution_records",
        "child_execution_records",
        "replay_records",
    ):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def classify_phase(
    phase_name: object,
    phase: Any,
    *,
    legacy_evidence: object = None,
) -> tuple[str, list[str]]:
    """Classify a phase conservatively while ignoring legacy E0--E3 labels."""
    del legacy_evidence
    metadata = phase if isinstance(phase, Mapping) else {}
    reasons: list[str] = []
    statuses: list[str] = []

    if "coverage_counted" in metadata and not coerce_bool(
        metadata.get("coverage_counted"), True
    ):
        reasons.append("coverage_not_counted")

    name = str(phase_name or "").lower()
    entry_derivation = str(metadata.get("entry_derivation") or "").lower()
    # An ISR reached through a captured execution context is an allowed event
    # input.  Only a cold vector/direct handler entry is diagnostic.
    if "cold_vector" in entry_derivation:
        reasons.append("cold_vector_entry")
    if (
        "isr" in name
        and "context_snapshot_mode" in metadata
        and not coerce_bool(metadata.get("context_snapshot_mode"), False)
    ):
        reasons.append("isr_without_context_snapshot")

    run_result = metadata.get("run_result")
    if isinstance(run_result, Mapping):
        status, execution_reasons = classify_execution(
            run_result,
            metadata=metadata,
        )
        statuses.append(status)
        reasons.extend(execution_reasons)

    child_records = _phase_child_records(metadata)
    if child_records:
        for item in child_records:
            item_map = item if isinstance(item, Mapping) else {}
            declared_status = str(item_map.get("status") or "").strip()
            declared_schema = str(item_map.get("schema") or "")
            if declared_status in {
                VALIDATED_REPLAY,
                DIAGNOSTIC_REPLAY,
                UNCLASSIFIED_REPLAY,
                MIXED_REPLAY,
            } and declared_schema == EVIDENCE_SCHEMA:
                # Records produced by build_execution_evidence_record already
                # contain the per-child execution facts.  Preserve their
                # status instead of treating the compact record itself as a
                # fresh run result (which would lose an unclassified marker).
                status = declared_status
                execution_reasons = list(item_map.get("reasons", []) or [])
                facts_available = coerce_bool(
                    item_map.get("execution_facts_available"), False
                )
                telemetry_complete = coerce_bool(
                    item_map.get("telemetry_complete"), False
                )
                if not telemetry_complete:
                    execution_reasons.append("incomplete_execution_telemetry")
                if not facts_available:
                    execution_reasons.append("missing_run_result")
                if status == VALIDATED_REPLAY and not compact_record_is_verified_validated(
                    item_map
                ):
                    status = UNCLASSIFIED_REPLAY
                    execution_reasons.append("compact_record_not_verified")
                statuses.append(status)
                reasons.extend(execution_reasons)
                continue
            nested = item_map.get("run_result")
            candidate = nested if isinstance(nested, Mapping) else item
            if isinstance(candidate, Mapping):
                status, execution_reasons = classify_execution(
                    candidate,
                    metadata={**metadata, **dict(item_map)},
                )
            else:
                status, execution_reasons = (
                    UNCLASSIFIED_REPLAY,
                    ["missing_run_result"],
                )
            statuses.append(status)
            reasons.extend(execution_reasons)
        if coerce_bool(metadata.get("execution_records_truncated"), False):
            reasons.append("incomplete_child_execution_telemetry")
        child_summary = metadata.get("child_execution_summary")
        if isinstance(child_summary, Mapping) and not coerce_bool(
            child_summary.get("execution_telemetry_complete"), False
        ):
            reasons.append("incomplete_child_execution_telemetry")
    else:
        child_summary = metadata.get("child_execution_summary")
        if isinstance(child_summary, Mapping):
            child_reasons = child_summary.get("intervention_reasons", {})
            if isinstance(child_reasons, Mapping):
                for reason, value in child_reasons.items():
                    if _positive_int(value):
                        reasons.append(str(reason))
            expected = _positive_int(
                child_summary.get("expected_child_replay_count")
            )
            completed = _positive_int(
                child_summary.get("child_completed_with_result_count")
            )
            if not coerce_bool(
                child_summary.get("execution_telemetry_complete"), False
            ):
                reasons.append("incomplete_child_execution_telemetry")
            if expected > 0 and completed == 0:
                reasons.append("missing_child_execution_results")
            if expected > 0 and completed > 0:
                # An aggregate count proves that workers reported something,
                # but without their per-run records it cannot prove which BB
                # had a force-free witness.  Keep the phase diagnostic and do
                # not let this status promote the phase coverage.
                reasons.append("missing_child_execution_records")
                statuses.append(DIAGNOSTIC_REPLAY)
            elif expected == 0:
                # A phase that did not execute a child cannot establish a
                # witness for its claimed coverage.
                statuses.append(UNCLASSIFIED_REPLAY)
                reasons.append("no_child_execution_attempt")
        elif not isinstance(run_result, Mapping):
            reasons.append("missing_phase_execution_summary")
            # At phase level this is a known evidence failure, not a clean
            # execution.  The BB-level accumulator still keeps such BBs
            # unclassified because no actual execution record owns them.
            statuses.append(DIAGNOSTIC_REPLAY)

    # Only actual typed counters are considered here.  Configured choices and
    # scheduler requests describe a plan, not what Unicorn changed.
    metadata_interventions = _intervention_counter_from_mapping(metadata)
    reasons.extend(metadata_interventions.keys())

    normalized_statuses = {
        str(status)
        for status in statuses
        if str(status)
    }
    if not normalized_statuses:
        status = DIAGNOSTIC_REPLAY if reasons else UNCLASSIFIED_REPLAY
    elif VALIDATED_REPLAY in normalized_statuses and (
        len(normalized_statuses) > 1
    ):
        # MIXED is an aggregate phase status only.  Individual BB records
        # retain their own validated/diagnostic/unclassified witness status.
        status = MIXED_REPLAY
    elif DIAGNOSTIC_REPLAY in normalized_statuses:
        status = DIAGNOSTIC_REPLAY
    elif UNCLASSIFIED_REPLAY in normalized_statuses:
        status = DIAGNOSTIC_REPLAY if reasons else UNCLASSIFIED_REPLAY
    else:
        status = VALIDATED_REPLAY
    if reasons and status == VALIDATED_REPLAY:
        status = DIAGNOSTIC_REPLAY
    return status, list(dict.fromkeys(str(item) for item in reasons if str(item)))
