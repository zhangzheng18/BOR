#!/usr/bin/env python3
"""Temporary emulator lifecycle management for LSGEmu runner phases."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .analysis.intelligent_emulator import IntelligentEmulator
from .mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from .register_tracer.register_tracer import RegisterTracer
from .runner_common import _env_flag


logger = logging.getLogger(__name__)


def _current_process_rss_bytes() -> int:
    """Read this process RSS without adding a psutil dependency."""
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    return max(0, int(fields[1]) * 1024) if len(fields) >= 2 else 0
    except (OSError, ValueError):
        pass
    return 0


@dataclass
class TempEmulatorLease:
    """Resources owned by one temporary replay emulator.

    Existing callers may keep using ``_new_temp_emulator`` and
    ``_dispose_temp_emulator``. New or high-frequency paths should use the
    managed lease so early returns and exceptions cannot bypass cleanup.
    """

    emulator: IntelligentEmulator
    mmio_handler: Optional[EnhancedMMIOHandler] = None
    register_tracer: Optional[RegisterTracer] = None
    _bind_callback: Optional[object] = None

    def bind_mmio(self, handler: EnhancedMMIOHandler) -> EnhancedMMIOHandler:
        self.mmio_handler = handler
        callback = self._bind_callback
        if callable(callback):
            callback(self.emulator, mmio_handler=handler)
        return handler

    def bind_tracer(self, tracer: RegisterTracer) -> RegisterTracer:
        self.register_tracer = tracer
        callback = self._bind_callback
        if callable(callback):
            callback(self.emulator, register_tracer=tracer)
        return tracer


class TempEmulatorLifecycleMixin:
    @staticmethod
    def _collect_temp_emulator_cleanup_errors(
        emulator: Optional[IntelligentEmulator],
        *,
        mmio_handler: Optional[EnhancedMMIOHandler] = None,
        register_tracer: Optional[RegisterTracer] = None,
    ) -> List[str]:
        """Return bounded teardown errors retained by replay-owned resources.

        Managed leases deliberately do not raise a teardown error over an
        execution error.  Callers that materialize evidence after leaving the
        lease therefore need one common way to recover those retained errors
        and downgrade the exact execution record.
        """
        errors: List[str] = []
        if emulator is not None:
            for attribute in (
                "_lsgemu_trace_cleanup_errors",
                "_lsgemu_cleanup_errors",
            ):
                try:
                    errors.extend(
                        str(item)[:768]
                        for item in list(getattr(emulator, attribute, []) or [])
                        if str(item)
                    )
                except Exception:
                    continue
        for owner in (mmio_handler, register_tracer):
            if owner is None:
                continue
            for attribute in (
                "_lsgemu_guided_cleanup_errors",
                "_lsgemu_trace_cleanup_errors",
            ):
                try:
                    errors.extend(
                        str(item)[:768]
                        for item in list(getattr(owner, attribute, []) or [])
                        if str(item)
                    )
                except Exception:
                    continue
        return list(dict.fromkeys(errors))

    def set_lifecycle_stage(self, stage_name: Optional[str]) -> None:
        next_stage = str(stage_name or "unknown")
        previous_stage = str(getattr(self, "current_stage_name", "") or "unknown")
        if previous_stage != next_stage:
            # Feedback derived for a previous stage must not be reused while
            # that stage is still changing; finalization caching is reset at
            # every lifecycle boundary.
            if hasattr(self, "scheduler_feedback_cached_stage"):
                self.scheduler_feedback_cached_stage = ""
            # A stage normally owns no live temporary emulator after its
            # function returns.  Clean up any unpaired construction before
            # releasing the stage's delayed native references.
            self._dispose_active_temp_emulators(
                stage_name=(None if previous_stage == "unknown" else previous_stage),
                reason="stage_change",
            )
        if (
            previous_stage
            and previous_stage != "unknown"
            and previous_stage != next_stage
            and _env_flag("LSGEMU_TEMP_EMULATOR_FLUSH_ON_STAGE_CHANGE", "1")
        ):
            self._flush_temp_emulator_quarantine(previous_stage, reason="stage_change")
        self.current_stage_name = next_stage

    def _lifecycle_stage_name(self) -> str:
        return str(getattr(self, "current_stage_name", "") or "unknown")

    def _temp_stage_for(self, purpose: str, stage_name: Optional[str] = None) -> str:
        """Return a stable lifecycle stage name for a replay/probe purpose."""
        purpose_name = str(purpose or "temp_emulator")
        parent = str(stage_name or self._lifecycle_stage_name() or "unknown")
        if not parent or parent == "unknown":
            return purpose_name
        if parent == purpose_name or parent.endswith(f":{purpose_name}"):
            return parent
        return f"{parent}:{purpose_name}"

    def _temp_lifecycle_counter(self, stage_name: Optional[str] = None) -> Counter[str]:
        stage = str(stage_name or self._lifecycle_stage_name())
        return self.temp_emulator_lifecycle_by_stage.setdefault(stage, Counter())

    def _active_temp_emulator_registry(
        self,
    ) -> Dict[int, Tuple[IntelligentEmulator, Optional[EnhancedMMIOHandler], Optional[RegisterTracer]]]:
        """Return the live-instance registry, including legacy test doubles."""
        registry = getattr(self, "active_temp_emulators", None)
        if registry is None:
            registry = {}
            self.active_temp_emulators = registry
        return registry

    def _track_temp_emulator(
        self,
        emulator: IntelligentEmulator,
        *,
        mmio_handler: Optional[EnhancedMMIOHandler] = None,
        register_tracer: Optional[RegisterTracer] = None,
    ) -> None:
        self._active_temp_emulator_registry()[id(emulator)] = (
            emulator,
            mmio_handler,
            register_tracer,
        )

    def _update_tracked_temp_emulator(
        self,
        emulator: IntelligentEmulator,
        *,
        mmio_handler: Optional[EnhancedMMIOHandler] = None,
        register_tracer: Optional[RegisterTracer] = None,
    ) -> None:
        registry = self._active_temp_emulator_registry()
        key = id(emulator)
        current = registry.get(key)
        if current is None or current[0] is not emulator:
            self._track_temp_emulator(
                emulator,
                mmio_handler=mmio_handler,
                register_tracer=register_tracer,
            )
            return
        registry[key] = (
            emulator,
            mmio_handler if mmio_handler is not None else current[1],
            register_tracer if register_tracer is not None else current[2],
        )

    def _dispose_active_temp_emulators(
        self,
        *,
        stage_name: Optional[str] = None,
        reason: str = "manual",
    ) -> int:
        """Dispose instances that escaped their local normal cleanup path.

        This is a correctness backstop, not a scheduling policy.  It never
        removes a task or changes coverage; it only releases native resources
        that were created by this runner and are no longer owned by a live
        replay scope.
        """
        registry = self._active_temp_emulator_registry()
        requested_stage = str(stage_name) if stage_name is not None else None
        pending = []
        for key, resource in list(registry.items()):
            emulator, mmio_handler, register_tracer = resource
            actual_stage = str(
                getattr(emulator, "_lsgemu_lifecycle_stage", None)
                or self._lifecycle_stage_name()
            )
            if requested_stage is not None and not (
                actual_stage == requested_stage
                or actual_stage.startswith(f"{requested_stage}:")
            ):
                continue
            pending.append((key, emulator, mmio_handler, register_tracer, actual_stage))
        for key, emulator, mmio_handler, register_tracer, actual_stage in pending:
            registry.pop(key, None)
            self._temp_lifecycle_counter(actual_stage)[f"active_cleanup_{reason}"] += 1
            try:
                self._dispose_temp_emulator(
                    emulator,
                    mmio_handler=mmio_handler,
                    register_tracer=register_tracer,
                    quarantine=False,
                    stage_name=actual_stage,
                )
            except Exception as exc:
                counter = self._temp_lifecycle_counter(actual_stage)
                counter["active_cleanup_failures"] += 1
                self.temp_emulator_cleanup_stats["active_cleanup_failures"] += 1
                text = (
                    f"active_cleanup:{type(exc).__name__}:{str(exc)[:512]}"
                )
                try:
                    existing = list(
                        getattr(emulator, "_lsgemu_cleanup_errors", []) or []
                    )
                    emulator._lsgemu_cleanup_errors = list(
                        dict.fromkeys(existing + [text])
                    )
                except Exception:
                    pass
                logger.debug(
                    "temporary emulator active cleanup failed for %s: %s",
                    actual_stage,
                    exc,
                )
        return len(pending)

    def _temp_lifecycle_phase_summary(self, phase_name: str) -> Dict[str, object]:
        phase = str(phase_name or "unknown")
        prefix = f"{phase}:"
        aggregate: Counter[str] = Counter()
        by_stage: Dict[str, Dict[str, int]] = {}
        for stage, counter in sorted(self.temp_emulator_lifecycle_by_stage.items()):
            if stage == phase or stage.startswith(prefix):
                aggregate.update(counter)
                by_stage[stage] = dict(counter)
        return {
            "aggregate": dict(aggregate),
            "by_stage": by_stage,
        }


    def _new_temp_emulator(
        self,
        *,
        max_snapshots: Optional[int] = None,
        llm_config_path: Optional[str | Path] = None,
        constraint_json_path: Optional[str | Path] = None,
        branch_mmio_file_mode: Optional[str] = None,
        stage_name: Optional[str] = None,
    ) -> IntelligentEmulator:
        """Create a replay-only emulator and record stage-level lifecycle pressure."""
        stage = str(stage_name or self._lifecycle_stage_name())
        emulator = None
        try:
            emulator = self.prepared.new_emulator(
                max_snapshots=max_snapshots if max_snapshots is not None else max(3, self.max_snapshots),
                llm_config_path=llm_config_path,
                constraint_json_path=constraint_json_path,
                branch_mmio_file_mode=branch_mmio_file_mode,
            )
            counter = self._temp_lifecycle_counter(stage)
            counter["created"] += 1
            self.temp_emulator_cleanup_stats["created"] += 1
            now = time.time()
            self.temp_emulator_stage_first_created_at.setdefault(stage, now)
            self.temp_emulator_stage_last_created_at[stage] = now
            if counter["created"] == 1:
                self.temp_emulator_stage_start_counts[stage] = int(self.temp_emulator_cleanup_stats["created"])
            setattr(emulator, "_lsgemu_lifecycle_stage", stage)
            setattr(emulator, "_lsgemu_lifecycle_disposed", False)
            # These fields are intentionally attached to the owned engine. A
            # replay record may be materialized before teardown, so the caller
            # needs a bounded way to inspect cleanup failures without keeping
            # the native engine alive or changing normal control flow.
            setattr(emulator, "_lsgemu_cleanup_errors", [])
            self._track_temp_emulator(emulator)
            return emulator
        except BaseException:
            # ``PreparedFirmware`` normally closes failures during its own
            # initialization.  This second guard covers failures in lifecycle
            # bookkeeping itself after the engine has been returned, so the
            # runner never loses ownership of a live temporary engine.
            if emulator is not None:
                try:
                    self._dispose_temp_emulator(
                        emulator,
                        quarantine=False,
                        stage_name=stage,
                    )
                except Exception as cleanup_error:
                    # Preserve the original construction/bookkeeping error.
                    counter = self._temp_lifecycle_counter(stage)
                    counter["creation_failure_cleanup_errors"] += 1
                    self.temp_emulator_cleanup_stats[
                        "creation_failure_cleanup_errors"
                    ] += 1
                    try:
                        existing = list(
                            getattr(emulator, "_lsgemu_cleanup_errors", []) or []
                        )
                        emulator._lsgemu_cleanup_errors = list(
                            dict.fromkeys(
                                existing
                                + [
                                    "creation_failure_cleanup:"
                                    f"{type(cleanup_error).__name__}:"
                                    f"{str(cleanup_error)[:512]}"
                                ]
                            )
                        )
                    except Exception:
                        pass
                    logger.debug(
                        "temporary emulator cleanup after creation failure failed: %s",
                        cleanup_error,
                    )
            raise

    @contextmanager
    def _managed_temp_emulator(
        self,
        *,
        max_snapshots: Optional[int] = None,
        llm_config_path: Optional[str | Path] = None,
        constraint_json_path: Optional[str | Path] = None,
        branch_mmio_file_mode: Optional[str] = None,
        stage_name: Optional[str] = None,
        quarantine: Optional[bool] = None,
    ) -> Iterator[TempEmulatorLease]:
        """Yield a temporary emulator whose resources are always disposed."""
        emulator = None
        lease = None
        try:
            emulator = self._new_temp_emulator(
                max_snapshots=max_snapshots,
                llm_config_path=llm_config_path,
                constraint_json_path=constraint_json_path,
                branch_mmio_file_mode=branch_mmio_file_mode,
                stage_name=stage_name,
            )
            lease = TempEmulatorLease(
                emulator=emulator,
                _bind_callback=self._update_tracked_temp_emulator,
            )
            yield lease
        except BaseException:
            if emulator is None:
                raise
            stage = str(
                getattr(emulator, "_lsgemu_lifecycle_stage", None)
                or stage_name
                or self._lifecycle_stage_name()
            )
            self._temp_lifecycle_counter(stage)["managed_exceptions"] += 1
            raise
        finally:
            if emulator is not None:
                stage = str(
                    getattr(emulator, "_lsgemu_lifecycle_stage", None)
                    or stage_name
                    or self._lifecycle_stage_name()
                )
                try:
                    self._dispose_temp_emulator(
                        emulator,
                        mmio_handler=lease.mmio_handler if lease is not None else None,
                        register_tracer=lease.register_tracer if lease is not None else None,
                        quarantine=quarantine,
                        stage_name=stage_name,
                    )
                except Exception as cleanup_error:
                    counter = self._temp_lifecycle_counter(stage)
                    counter["managed_cleanup_failures"] += 1
                    self.temp_emulator_cleanup_stats["managed_cleanup_failures"] += 1
                    try:
                        existing = list(
                            getattr(emulator, "_lsgemu_cleanup_errors", []) or []
                        )
                        emulator._lsgemu_cleanup_errors = list(
                            dict.fromkeys(
                                existing
                                + [
                                    "managed_cleanup:"
                                    f"{type(cleanup_error).__name__}:"
                                    f"{str(cleanup_error)[:512]}"
                                ]
                            )
                        )
                    except Exception:
                        pass
                    logger.debug(
                        "managed temporary emulator cleanup failed for %s: %s",
                        stage,
                        cleanup_error,
                    )

    def _stage_needs_safe_quarantine(self, stage_name: str) -> bool:
        # ``IntelligentEmulator.close`` detaches hooks and this module releases
        # the remaining Uc owner references below.  Retaining hundreds of
        # already-closed Python wrappers is therefore unnecessary and can keep
        # replay bookkeeping alive for the whole high-frequency stage.
        # Preserve quarantine as an explicit diagnostic escape hatch only.
        if not _env_flag("LSGEMU_TEMP_EMULATOR_QUARANTINE_AFTER_CLOSE", "0"):
            return False
        counter = self._temp_lifecycle_counter(stage_name)
        try:
            burst_threshold = max(1, int(os.environ.get("LSGEMU_TEMP_EMULATOR_HIGH_FREQ_THRESHOLD", "8")))
        except ValueError:
            burst_threshold = 8
        try:
            total_threshold = max(1, int(os.environ.get("LSGEMU_TEMP_EMULATOR_HIGH_FREQ_TOTAL_THRESHOLD", "64")))
        except ValueError:
            total_threshold = 64
        try:
            burst_window = max(0.1, float(os.environ.get("LSGEMU_TEMP_EMULATOR_HIGH_FREQ_WINDOW_SECONDS", "30")))
        except ValueError:
            burst_window = 30.0
        try:
            rate_threshold = max(0.0, float(os.environ.get("LSGEMU_TEMP_EMULATOR_HIGH_FREQ_RATE", "0.25")))
        except ValueError:
            rate_threshold = 0.25

        created = int(counter.get("created", 0) or 0)
        first_seen = float(self.temp_emulator_stage_first_created_at.get(stage_name, time.time()))
        last_seen = float(self.temp_emulator_stage_last_created_at.get(stage_name, time.time()))
        elapsed = max(0.001, last_seen - first_seen)
        rate = created / elapsed
        burst_detected = created >= burst_threshold and elapsed <= burst_window
        sustained_rate_detected = created >= burst_threshold and rate >= rate_threshold
        total_detected = created >= total_threshold
        if burst_detected or sustained_rate_detected or total_detected:
            self.temp_emulator_stage_high_frequency.add(stage_name)
            counter["high_frequency_detected"] = 1
            counter["high_frequency_created"] = created
            counter["high_frequency_elapsed_ms"] = int(elapsed * 1000)
            counter["high_frequency_rate_per_min_x100"] = int(rate * 60 * 100)
            return True
        if stage_name in self.temp_emulator_stage_high_frequency:
            return True
        return _env_flag("LSGEMU_TEMP_EMULATOR_ALWAYS_QUARANTINE", "0")

    def temp_emulator_lifecycle_summary(self) -> Dict[str, object]:
        return {
            "global": dict(self.temp_emulator_cleanup_stats),
            "high_frequency_stages": sorted(self.temp_emulator_stage_high_frequency),
            "by_stage": {
                stage: dict(counter)
                for stage, counter in sorted(self.temp_emulator_lifecycle_by_stage.items())
            },
        }

    def _dispose_temp_emulator(
        self,
        emulator: Optional[IntelligentEmulator],
        *,
        mmio_handler: Optional[EnhancedMMIOHandler] = None,
        register_tracer: Optional[RegisterTracer] = None,
        quarantine: Optional[bool] = None,
        stage_name: Optional[str] = None,
    ) -> None:
        """Stop hooks and release a replay-only Unicorn instance safely."""
        if emulator is None:
            return
        cleanup_errors = list(
            getattr(emulator, "_lsgemu_cleanup_errors", []) or []
        )

        for owner in (mmio_handler, register_tracer):
            for attribute in (
                "_lsgemu_guided_cleanup_errors",
                "_lsgemu_trace_cleanup_errors",
            ):
                try:
                    for error in list(getattr(owner, attribute, []) or []):
                        text = f"{attribute}:{str(error)[:512]}"
                        if text not in cleanup_errors:
                            cleanup_errors.append(text)
                except Exception:
                    continue

        def remember(operation: str, error: BaseException) -> None:
            text = f"{operation}:{type(error).__name__}:{str(error)[:512]}"
            if text not in cleanup_errors:
                cleanup_errors.append(text)

        registry = self._active_temp_emulator_registry()
        tracked = registry.get(id(emulator))
        if tracked is not None and tracked[0] is emulator:
            if mmio_handler is None:
                mmio_handler = tracked[1]
            if register_tracer is None:
                register_tracer = tracked[2]
            registry.pop(id(emulator), None)
        stage = str(
            stage_name
            or getattr(emulator, "_lsgemu_lifecycle_stage", None)
            or self._lifecycle_stage_name()
        )
        counter = self._temp_lifecycle_counter(stage)
        if bool(getattr(emulator, "_lsgemu_lifecycle_disposed", False)):
            counter["duplicate_dispose_ignored"] += 1
            self.temp_emulator_cleanup_stats["duplicate_dispose_ignored"] += 1
            return
        setattr(emulator, "_lsgemu_lifecycle_disposed", True)
        if register_tracer is not None:
            try:
                register_tracer.stop_tracing()
            except Exception as exc:
                counter["tracer_stop_failures"] += 1
                remember("tracer_stop", exc)
        if mmio_handler is not None:
            try:
                mmio_handler.stop_hooking()
            except Exception as exc:
                counter["mmio_stop_failures"] += 1
                remember("mmio_stop", exc)
        try:
            emulator.close()
        except Exception as exc:
            counter["emulator_close_failures"] += 1
            remember("emulator_close", exc)
        # The close operation deliberately keeps the native Uc object in
        # ``_closed_uc`` until its Python owner disappears.  At this point all
        # emulator-owned hooks have been removed and the transient handlers no
        # longer need to access the engine, so detach those final references.
        # Unicorn's own weakref finalizer then performs the one native close;
        # no private finalizer is invoked here.
        native_uc = getattr(emulator, "_closed_uc", None)
        if native_uc is not None:
            for owner in (mmio_handler, register_tracer):
                if owner is None:
                    continue
                try:
                    if getattr(owner, "uc", None) is native_uc:
                        owner.uc = None
                except Exception as exc:
                    remember("detach_native_owner", exc)
            try:
                emulator._closed_uc = None
                counter["native_owner_references_released"] += 1
                self.temp_emulator_cleanup_stats["native_owner_references_released"] += 1
            except Exception as exc:
                counter["native_owner_reference_release_failures"] += 1
                remember("release_native_owner", exc)
        native_uc = None
        self.temp_emulator_cleanup_stats["disposed"] += 1
        counter["disposed"] += 1
        effective_quarantine = (
            self._stage_needs_safe_quarantine(stage)
            if quarantine is None
            else bool(quarantine)
        )
        if effective_quarantine:
            self.temp_emulator_quarantine.append((stage, emulator, mmio_handler, register_tracer))
            self.temp_emulator_cleanup_stats["quarantined"] += 1
            counter["quarantined"] += 1
            try:
                soft_limit = max(0, int(os.environ.get("LSGEMU_TEMP_EMULATOR_QUARANTINE_LIMIT", "64")))
            except ValueError:
                soft_limit = 64
            try:
                hard_limit = max(soft_limit, int(os.environ.get("LSGEMU_TEMP_EMULATOR_QUARANTINE_HARD_LIMIT", "512")))
            except ValueError:
                hard_limit = max(soft_limit, 512)
            evict_on_dispose = _env_flag("LSGEMU_TEMP_EMULATOR_EVICT_ON_DISPOSE", "0")
            eviction_limit = soft_limit if evict_on_dispose else hard_limit
            while eviction_limit and len(self.temp_emulator_quarantine) > eviction_limit:
                evicted = self.temp_emulator_quarantine.popleft()
                evicted_stage = str(evicted[0] if isinstance(evicted, tuple) and evicted else stage)
                evicted_counter = self._temp_lifecycle_counter(evicted_stage)
                self.temp_emulator_cleanup_stats["quarantine_evicted"] += 1
                evicted_counter["quarantine_evicted"] += 1
        try:
            self._periodic_temp_emulator_reclamation(counter)
        except Exception as exc:
            # Reclamation is an optimization after the engine has been
            # detached. It must never turn a completed replay into a phase
            # exception, but it remains visible to evidence auditing.
            counter["reclamation_failures"] += 1
            self.temp_emulator_cleanup_stats["reclamation_failures"] += 1
            remember("reclamation", exc)
        try:
            emulator._lsgemu_cleanup_errors = cleanup_errors
        except Exception:
            # Test doubles and unusual proxy objects may reject attributes;
            # cleanup has already completed and there is no safe owner field
            # left to update in that case.
            pass

    def _periodic_temp_emulator_reclamation(self, counter: Counter[str]) -> None:
        """Finalize dead callback cycles and periodically return free heap pages.

        Unicorn Python hooks contain callback objects that can participate in
        cycles.  A long replay stage may create thousands of engines before a
        generation-2 collection runs naturally.  Collection happens only
        after hook deletion and native-owner detachment, so this changes
        resource timing rather than emulator semantics.
        """
        disposed = int(self.temp_emulator_cleanup_stats.get("disposed", 0) or 0)
        try:
            gc_interval = max(
                0,
                int(os.environ.get("LSGEMU_TEMP_EMULATOR_GC_INTERVAL", "64")),
            )
        except ValueError:
            gc_interval = 64
        gc_ran = False
        if gc_interval > 0 and disposed > 0 and disposed % gc_interval == 0:
            collected = int(gc.collect())
            gc_ran = True
            counter["periodic_gc_runs"] += 1
            counter["periodic_gc_collected"] += max(0, collected)
            self.temp_emulator_cleanup_stats["periodic_gc_runs"] += 1
            self.temp_emulator_cleanup_stats["periodic_gc_collected"] += max(
                0,
                collected,
            )

        try:
            trim_interval = max(
                0,
                int(
                    os.environ.get(
                        "LSGEMU_TEMP_EMULATOR_MALLOC_TRIM_INTERVAL",
                        "64",
                    )
                ),
            )
        except ValueError:
            trim_interval = 256
        current_rss = _current_process_rss_bytes()
        last_trim_rss = int(
            getattr(self, "_lsgemu_last_trim_rss_bytes", 0) or 0
        )
        if last_trim_rss <= 0 and current_rss > 0:
            self._lsgemu_last_trim_rss_bytes = current_rss
            last_trim_rss = current_rss
        try:
            rss_trim_threshold = max(
                0,
                int(
                    os.environ.get(
                        "LSGEMU_TEMP_EMULATOR_RSS_TRIM_THRESHOLD_MIB",
                        "256",
                    )
                ),
            ) * 1024 * 1024
        except ValueError:
            rss_trim_threshold = 256 * 1024 * 1024
        interval_due = bool(
            trim_interval > 0
            and disposed > 0
            and disposed % trim_interval == 0
        )
        pressure_due = bool(
            rss_trim_threshold > 0
            and current_rss > 0
            and last_trim_rss > 0
            and current_rss - last_trim_rss >= rss_trim_threshold
        )
        if not interval_due and not pressure_due:
            return

        # ``malloc_trim`` only returns already-free pages to the host.  Hook
        # callbacks and their owners can form cycles after ``close()``, so a
        # pressure-triggered trim without a preceding collection may leave
        # the largest part of the replay engine graph unreachable but still
        # retained.  Collect only on the pressure path; the normal interval
        # remains unchanged to avoid adding GC work to every replay.
        if (
            pressure_due
            and not gc_ran
            and _env_flag("LSGEMU_TEMP_EMULATOR_GC_ON_PRESSURE", "1")
        ):
            rss_before_gc = current_rss
            collected = int(gc.collect())
            gc_ran = True
            rss_after_gc = _current_process_rss_bytes()
            gc_rss_reduced = (
                max(0, int(rss_before_gc) - int(rss_after_gc))
                if rss_before_gc > 0 and rss_after_gc > 0
                else 0
            )
            counter["pressure_gc_runs"] += 1
            counter["pressure_gc_collected"] += max(0, collected)
            counter["pressure_gc_rss_reduced_bytes"] += gc_rss_reduced
            self.temp_emulator_cleanup_stats["pressure_gc_runs"] += 1
            self.temp_emulator_cleanup_stats["pressure_gc_collected"] += max(
                0,
                collected,
            )
            self.temp_emulator_cleanup_stats[
                "pressure_gc_rss_reduced_bytes"
            ] += gc_rss_reduced
            current_rss = rss_after_gc or current_rss
        try:
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim")
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            trim_result = int(malloc_trim(0))
            rss_after_trim = _current_process_rss_bytes()
            rss_reduced = (
                max(0, int(current_rss) - int(rss_after_trim))
                if current_rss > 0 and rss_after_trim > 0
                else 0
            )
            counter["malloc_trim_runs"] += 1
            counter["malloc_trim_successes"] += int(trim_result != 0)
            counter["malloc_trim_rss_reduced_bytes"] += rss_reduced
            self.temp_emulator_cleanup_stats["malloc_trim_runs"] += 1
            self.temp_emulator_cleanup_stats["malloc_trim_successes"] += int(
                trim_result != 0
            )
            self.temp_emulator_cleanup_stats[
                "malloc_trim_rss_reduced_bytes"
            ] += rss_reduced
            self._lsgemu_last_trim_rss_bytes = rss_after_trim or current_rss
        except Exception:
            counter["malloc_trim_unavailable"] += 1
            self.temp_emulator_cleanup_stats["malloc_trim_unavailable"] += 1

    def _flush_temp_emulator_quarantine(
        self,
        stage_name: Optional[str] = None,
        *,
        reason: str = "manual",
    ) -> None:
        stage = str(stage_name or self._lifecycle_stage_name())
        counter = self._temp_lifecycle_counter(stage)
        retained = deque()
        flushed_by_stage: Counter[str] = Counter()
        while self.temp_emulator_quarantine:
            item = self.temp_emulator_quarantine.popleft()
            if not isinstance(item, tuple) or len(item) != 4:
                flushed_by_stage[stage] += 1
                continue
            item_stage, emulator, mmio_handler, register_tracer = item
            item_stage = str(item_stage or "unknown")
            if stage_name is None or item_stage == stage or item_stage.startswith(f"{stage}:"):
                flushed_by_stage[item_stage] += 1
            else:
                retained.append((item_stage, emulator, mmio_handler, register_tracer))
        self.temp_emulator_quarantine = retained
        count = sum(flushed_by_stage.values())
        if count:
            self.temp_emulator_cleanup_stats["quarantine_flushed"] += count
            counter["quarantine_flush_events"] += 1
            counter[f"quarantine_flushed_{reason}"] += count
            for item_stage, item_count in flushed_by_stage.items():
                item_counter = self._temp_lifecycle_counter(item_stage)
                item_counter["quarantine_flushed"] += int(item_count)
                item_counter[f"quarantine_flushed_{reason}"] += int(item_count)
        if _env_flag("LSGEMU_TEMP_EMULATOR_GC_AFTER_FLUSH", "0"):
            gc.collect()
