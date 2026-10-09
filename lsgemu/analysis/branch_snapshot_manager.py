"""
分支快照管理器

功能：
1. 在分支点和外部输入读取前保存完整快照
2. 恢复CPU、RAM和外部模型状态
3. 为探索阶段保留显式标注的反事实分支能力
"""

import logging
import os
import hashlib
import copy
from collections import Counter
from typing import Callable, Dict, List, Mapping, Optional, Set, Tuple
from dataclasses import dataclass, field

from .snapshot_memory import (
    SnapshotMetadataStore,
    PagedMemory,
    SnapshotCaptureError,
    SnapshotIntegrityError,
    SnapshotPageStore,
    SnapshotBlobStore,
    SnapshotStateBlob,
)
from ..evidence_contract import coerce_bool, is_environment_fact_reason

logger = logging.getLogger(__name__)


def _snapshot_value(snapshot: object, key: str, default: object = None) -> object:
    """Read a snapshot field from either a dataclass or a serialized mapping."""
    if isinstance(snapshot, Mapping):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


@dataclass(slots=True)
class BranchSnapshot:
    """分支点快照"""
    address: int  # 分支地址
    target: int  # 跳转目标
    fallthrough: int  # 不跳转地址
    condition: str  # 条件码
    original_taken: bool  # 主路径上的实际方向
    order: int  # 主路径发现顺序
    depth: int  # 主路径分支深度

    # 寄存器状态
    registers: Dict[str, int]
    cpsr: int

    # 内存状态（只保存RAM区域）
    memory_data: bytes | PagedMemory
    memory_base: int
    memory_size: int
    memory_regions: Dict[Tuple[int, int], bytes | PagedMemory] = field(default_factory=dict)
    mmio_state: Dict[int, int] = field(default_factory=dict)
    alternatives: List[int] = field(default_factory=list)
    original_index: Optional[int] = None
    occurrence_index: int = 1
    capture_order: int = 0
    dirty_pages: Set[int] = field(default_factory=set)
    input_event_index: int = 0
    input_occurrence_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    external_model_state: Dict[str, object] | SnapshotStateBlob = field(
        default_factory=dict
    )
    capture_schema: str = "lsgemu.branch_snapshot.v2"
    capture_complete: bool = True
    region_hashes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    integrity_hash: str = ""
    # Capture-time execution lineage.  These optional fields preserve
    # compatibility with old pickles/reports while allowing a replay verifier
    # to reject a clean suffix derived from a diagnostic ancestor.
    provenance_status: str = "unspecified"
    provenance_reasons: Tuple[str, ...] = field(default_factory=tuple)
    source_execution_id: str = ""
    prefix_telemetry_complete: bool = False
    prefix_intervention_reasons: Tuple[str, ...] = field(default_factory=tuple)
    # A snapshot is captured while a Unicorn run is still in progress.  The
    # prefix can be observed synchronously, but its lineage must not be
    # presented as final until the capture operation itself has succeeded.
    provenance_finalized: bool = False
    # Teardown/finalization failures can be discovered after this object has
    # been wrapped and handed to another replay queue.  Keep an explicit
    # terminal marker on the source object so every wrapper can fail closed
    # without discarding the BBs observed by the failed execution.
    provenance_invalidated: bool = False
    provenance_invalidation_reasons: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class BranchEvent:
    """主路径上的一次条件分支事件，按出现次数区分循环中的同一分支。"""
    address: int
    branch_pc: int
    target: int
    fallthrough: int
    condition: str
    original_taken: bool
    order: int
    depth: int
    occurrence_index: int
    alternatives: List[int] = field(default_factory=list)
    original_index: Optional[int] = None
    first_order: int = 0
    first_depth: int = 0
    first_occurrence_index: int = 0
    # A dynamic successor set does not establish which edge was taken first.
    # Existing serialized events without these fields remain real observations
    # through the compatibility defaults in the runner.
    original_direction_known: bool = True
    direction_provenance: str = "unicorn_execution"
    synthetic: bool = False
    context_signature: Dict[str, object] = field(default_factory=dict)


class BranchSnapshotManager:
    """
    分支快照管理器

    在基准运行时保存所有分支点的快照
    在路径探索时恢复快照并翻转分支
    """

    def __init__(
        self,
        page_store: Optional[SnapshotPageStore] = None,
        metadata_store: Optional[SnapshotMetadataStore] = None,
        blob_store: Optional[SnapshotBlobStore] = None,
        *,
        max_current_snapshots: Optional[int] = None,
    ):
        self.snapshots: Dict[int, BranchSnapshot] = {}  # {address: snapshot}
        self.snapshot_history: Dict[int, List[BranchSnapshot]] = {}
        self._next_order = 0
        self._next_capture_order = 0
        self._next_event_order = 0
        self._next_access_order = 0
        self._snapshot_last_used: Dict[int, int] = {}
        # `events` is a compact branch-edge catalog used by the scheduler.
        # `occurrence_events` is the exact retained prefix used by witness
        # validation.  `_next_event_order` remains monotonic even when this
        # bounded payload reaches capacity, so causal input records never use
        # the retained-list length as a dynamic event identity.
        self.events: List[BranchEvent] = []
        self.occurrence_events: List[BranchEvent] = []
        self.occurrence_events_truncated = 0
        self.occurrence_counts: Dict[int, int] = {}
        self._event_by_edge: Dict[tuple, BranchEvent] = {}
        # Keep a bounded main-path window; very long tail loops otherwise
        # dominate bottom-up scheduling during short reservoir iterations.
        self.max_events = 10000
        try:
            self.max_occurrence_events = max(
                1,
                int(os.environ.get("LSGEMU_BRANCH_OCCURRENCE_EVENT_LIMIT", "200000")),
            )
        except ValueError:
            self.max_occurrence_events = 200000
        try:
            self.max_history_per_address = int(
                os.environ.get("LSGEMU_BRANCH_SNAPSHOT_HISTORY_PER_ADDRESS", "0")
            )
        except ValueError:
            self.max_history_per_address = 0
        try:
            self.max_history_total = int(
                os.environ.get("LSGEMU_BRANCH_SNAPSHOT_HISTORY_TOTAL", "0")
            )
        except ValueError:
            self.max_history_total = 0
        try:
            # Zero preserves the historical unbounded current-snapshot
            # behavior.  A positive value is an explicit replay-memory cap;
            # branch/event catalogs remain available for scheduling/auditing.
            configured_limit = (
                max_current_snapshots
                if max_current_snapshots is not None
                else os.environ.get("LSGEMU_BRANCH_SNAPSHOT_CURRENT_LIMIT", "0")
            )
            self.max_current_snapshots = max(0, int(configured_limit))
        except (TypeError, ValueError):
            self.max_current_snapshots = 0
        self.memory_regions: List[Tuple[int, int]] = [(0x20000000, 0x100000)]
        self.dirty_page_provider: Optional[Callable[[], Set[int]]] = None
        self.external_state_provider: Optional[Callable[[], Dict[str, object]]] = None
        # P0-D（cycle3 k.5 C3）：save_snapshot 调用 external_state_provider
        # 期间的捕获定址上下文（emulator 侧 identity 遥测读取；其余时间为
        # None）。纯遥测通道，不参与任何执行/恢复语义。
        self._pending_capture_context: Optional[Dict[str, int]] = None
        self.page_store = page_store or SnapshotPageStore()
        self.metadata_store = metadata_store or SnapshotMetadataStore()
        self.blob_store = blob_store or SnapshotBlobStore.from_environment()
        self.capture_stats: Counter[str] = Counter()

    @staticmethod
    def _region_digest(data: bytes | PagedMemory) -> str:
        digest = getattr(data, "sha256", None)
        return str(digest) if digest else hashlib.sha256(bytes(data)).hexdigest()

    @classmethod
    def _integrity_digest(
        cls,
        registers: Dict[str, int],
        cpsr: int,
        memory_regions: Dict[Tuple[int, int], bytes | PagedMemory],
    ) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(registers.items()):
            digest.update(str(name).encode("ascii", errors="replace"))
            digest.update(int(value).to_bytes(8, "little", signed=False))
        digest.update(int(cpsr).to_bytes(8, "little", signed=False))
        for (base, size), data in sorted(memory_regions.items()):
            digest.update(int(base).to_bytes(8, "little", signed=False))
            digest.update(int(size).to_bytes(8, "little", signed=False))
            digest.update(cls._region_digest(data).encode("ascii"))
        return digest.hexdigest()

    def _capture_error(self, component: str, exc: object, address: int) -> SnapshotCaptureError:
        self.capture_stats["capture_failures"] += 1
        self.capture_stats[f"capture_failure_{component}"] += 1
        return SnapshotCaptureError(component, str(exc), address=address)

    def set_memory_regions(self, memory_regions: List[Tuple[int, int]]) -> None:
        normalized: List[Tuple[int, int]] = []
        for start, size in memory_regions or []:
            try:
                start_i = int(start)
                size_i = int(size)
            except Exception:
                continue
            if size_i > 0:
                normalized.append((start_i, size_i))
        self.memory_regions = normalized or [(0x20000000, 0x100000)]

    def set_dirty_page_provider(self, provider: Optional[Callable[[], Set[int]]]) -> None:
        """Attach emulator-owned runtime-written page provenance to snapshots."""
        self.dirty_page_provider = provider

    def set_external_state_provider(
        self,
        provider: Optional[Callable[[], Dict[str, object]]],
    ) -> None:
        """Attach causal-input and peripheral-model state to new snapshots."""
        self.external_state_provider = provider

    def record_event(self, address: int, branch_pc: int, target: int, fallthrough: int,
                     condition: str, original_taken: bool, depth: int,
                     alternatives: Optional[List[int]] = None,
                     original_index: Optional[int] = None,
                     context_signature: Optional[Dict[str, object]] = None) -> BranchEvent:
        """记录一次真实执行路径上的分支事件。"""
        occurrence_index = self.occurrence_counts.get(address, 0) + 1
        self.occurrence_counts[address] = occurrence_index

        if alternatives is None:
            alternatives = [target, fallthrough]
        if original_index is None:
            original_index = 0 if original_taken else 1
        event_order = self._next_event_order
        self._next_event_order += 1
        edge_key = (
            address,
            condition,
            original_index if condition in {"TBB", "TBH", "LDRPC"} else original_taken,
        )

        raw_event = BranchEvent(
            address=address,
            branch_pc=branch_pc,
            target=target,
            fallthrough=fallthrough,
            condition=condition,
            original_taken=original_taken,
            order=event_order,
            depth=depth,
            occurrence_index=occurrence_index,
            alternatives=list(alternatives),
            original_index=original_index,
            first_order=event_order,
            first_depth=depth,
            first_occurrence_index=occurrence_index,
            original_direction_known=True,
            direction_provenance="unicorn_execution",
            synthetic=False,
            context_signature=copy.deepcopy(dict(context_signature or {})),
        )
        self._annotate_matching_snapshots(raw_event)
        if len(self.occurrence_events) < self.max_occurrence_events:
            self.occurrence_events.append(raw_event)
        else:
            self.occurrence_events_truncated += 1

        existing = self._event_by_edge.get(edge_key)
        if existing is not None:
            # For direct loop-back edges, keep the earliest occurrence so a
            # wait loop can be exited early. For ordinary repeated branches,
            # move the occurrence forward with the deepest/latest observation.
            existing.order = event_order
            existing.depth = depth
            existing.target = target
            existing.fallthrough = fallthrough
            existing.alternatives = list(alternatives)
            existing.original_index = original_index
            existing.context_signature = copy.deepcopy(
                dict(context_signature or {})
            )
            if not self._is_direct_loop_edge(branch_pc, target, fallthrough, condition, original_taken):
                existing.occurrence_index = occurrence_index
            return raw_event

        catalog_event = BranchEvent(
            address=address,
            branch_pc=branch_pc,
            target=target,
            fallthrough=fallthrough,
            condition=condition,
            original_taken=original_taken,
            order=event_order,
            depth=depth,
            occurrence_index=occurrence_index,
            alternatives=list(alternatives),
            original_index=original_index,
            first_order=event_order,
            first_depth=depth,
            first_occurrence_index=occurrence_index,
            original_direction_known=True,
            direction_provenance="unicorn_execution",
            synthetic=False,
            context_signature=copy.deepcopy(dict(context_signature or {})),
        )

        if len(self.events) < self.max_events:
            self.events.append(catalog_event)
            self._event_by_edge[edge_key] = catalog_event
        return raw_event

    def _annotate_matching_snapshots(self, event: BranchEvent) -> None:
        """Bind a pre-branch entry snapshot to the direction Unicorn observed."""
        candidates = []
        current = self.snapshots.get(int(event.address))
        if current is not None:
            candidates.append(current)
        candidates.extend(self.snapshot_history.get(int(event.address), []) or [])
        seen = set()
        for snapshot in candidates:
            if id(snapshot) in seen:
                continue
            seen.add(id(snapshot))
            if int(getattr(snapshot, "occurrence_index", 1) or 1) != int(event.occurrence_index):
                continue
            snapshot.target = int(event.target)
            snapshot.fallthrough = int(event.fallthrough)
            snapshot.original_taken = bool(event.original_taken)
            snapshot.alternatives = list(event.alternatives or [])
            snapshot.original_index = event.original_index
            self.capture_stats["snapshots_annotated_from_execution"] += 1

    def _is_direct_loop_edge(self, branch_pc: int, target: int, fallthrough: int,
                             condition: str, original_taken: bool) -> bool:
        if condition in {"TBB", "TBH", "LDRPC"}:
            return False
        chosen = target if original_taken else fallthrough
        if chosen == 0:
            return False
        return chosen <= branch_pc

    def save_snapshot(self, uc, address: int, target: int,
                     fallthrough: int, condition: str,
                     original_taken: bool = False,
                     depth: int = 0,
                     preserve_order: bool = False,
                     mmio_state: Optional[Dict[int, int]] = None,
                     alternatives: Optional[List[int]] = None,
                     original_index: Optional[int] = None,
                     occurrence_index: int = 1,
                     update_current: bool = True,
                     provenance: Optional[Mapping[str, object]] = None) -> BranchSnapshot:
        """
        保存分支点快照

        Args:
            uc: Unicorn实例
            address: 分支地址
            target: 跳转目标
            fallthrough: 不跳转地址
            condition: 条件码

        Returns:
            BranchSnapshot
        """
        from unicorn.arm_const import (
            UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
            UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
            UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
            UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
            UC_ARM_REG_CPSR, UC_ARM_REG_FPSCR
        )
        from unicorn import arm_const as _arm_const

        # 保存寄存器
        registers = {}
        reg_list = [
            ('r0', UC_ARM_REG_R0), ('r1', UC_ARM_REG_R1),
            ('r2', UC_ARM_REG_R2), ('r3', UC_ARM_REG_R3),
            ('r4', UC_ARM_REG_R4), ('r5', UC_ARM_REG_R5),
            ('r6', UC_ARM_REG_R6), ('r7', UC_ARM_REG_R7),
            ('r8', UC_ARM_REG_R8), ('r9', UC_ARM_REG_R9),
            ('r10', UC_ARM_REG_R10), ('r11', UC_ARM_REG_R11),
            ('r12', UC_ARM_REG_R12), ('sp', UC_ARM_REG_SP),
            ('lr', UC_ARM_REG_LR), ('pc', UC_ARM_REG_PC)
        ]
        # P0-A（cycle3 k.5 C2）：VFP/标志位捕获保真——D0–D31 + FPSCR 并列
        # 进捕获面（k.4 证据：历史 armP3 blob 该面全空，E4 红面 3/82 正是
        # 丢失的 VFP 上下文）。CPSR 仍单独捕获（下方，不动）。
        reg_list += [
            (f'd{i}', getattr(_arm_const, f'UC_ARM_REG_D{i}')) for i in range(32)
        ]
        reg_list.append(('fpscr', UC_ARM_REG_FPSCR))

        for name, reg_id in reg_list:
            try:
                registers[name] = int(uc.reg_read(reg_id)) & 0xFFFFFFFFFFFFFFFF
            except Exception as exc:
                raise self._capture_error(f"register_{name}", exc, address) from exc

        try:
            cpsr = int(uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFFFFFFFFFF
        except Exception as exc:
            raise self._capture_error("register_cpsr", exc, address) from exc

        memory_regions: Dict[Tuple[int, int], bytes | PagedMemory] = {}
        for memory_base, memory_size in self.memory_regions:
            try:
                raw_memory = bytes(uc.mem_read(memory_base, memory_size))
            except Exception as exc:
                raise self._capture_error(
                    f"memory_0x{int(memory_base):08x}", exc, address
                ) from exc
            if len(raw_memory) != int(memory_size):
                raise self._capture_error(
                    f"memory_0x{int(memory_base):08x}",
                    f"short read: expected {int(memory_size)}, got {len(raw_memory)}",
                    address,
                )
            memory_regions[(memory_base, memory_size)] = self.page_store.intern_region(raw_memory)
        memory_base, memory_size = self.memory_regions[0]
        memory_data = memory_regions[(memory_base, memory_size)]
        dirty_pages: Set[int] = set()
        if self.dirty_page_provider is not None:
            try:
                dirty_pages = {
                    int(page) & ~0xFFF
                    for page in (self.dirty_page_provider() or set())
                }
            except Exception:
                dirty_pages = set()
        external_state: Dict[str, object] = {}
        external_state_capture_error: Optional[str] = None
        if self.external_state_provider is not None:
            # P0-D（cycle3 k.5 C3）：把本次捕获的定址上下文挂在管理器上，
            # 供 emulator 侧零参 provider 读取（纯遥测通道，零执行语义）。
            self._pending_capture_context = {
                "address": int(address),
                "occurrence_index": int(occurrence_index),
            }
            try:
                provided = self.external_state_provider() or {}
                if isinstance(provided, Mapping):
                    external_state = dict(provided)
                else:
                    external_state_capture_error = "provider_returned_non_mapping"
            except Exception:
                external_state_capture_error = "provider_exception"
            finally:
                self._pending_capture_context = None
        input_occurrence_counts = self.metadata_store.intern_mapping({
            (int(key[0]), int(key[1])): int(value)
            for key, value in dict(external_state.get("input_occurrence_counts", {}) or {}).items()
            if isinstance(key, tuple) and len(key) == 2
        })
        region_hashes = {
            key: self._region_digest(data)
            for key, data in memory_regions.items()
        }
        integrity_hash = self._integrity_digest(registers, cpsr, memory_regions)

        existing = self.snapshots.get(address) if preserve_order else None
        capture_order = self._next_capture_order
        self._next_capture_order += 1
        raw_external_model_state = external_state.get("external_model_state", {})
        external_model_state = (
            dict(raw_external_model_state)
            if isinstance(raw_external_model_state, Mapping)
            else {}
        )
        raw_capture_provenance = external_state.get("execution_provenance", {})
        capture_provenance = (
            dict(raw_capture_provenance)
            if isinstance(raw_capture_provenance, Mapping)
            else {}
        )
        if isinstance(provenance, Mapping):
            capture_provenance.update(dict(provenance))
        if external_state_capture_error:
            capture_provenance.setdefault("status", "unverified")
            existing_reasons = capture_provenance.get("reasons") or ()
            if isinstance(existing_reasons, str):
                existing_reasons = (existing_reasons,)
            capture_provenance["reasons"] = tuple(existing_reasons) + (
                f"external_state_capture_failed:{external_state_capture_error}",
            )
            capture_provenance["telemetry_complete"] = False
        provenance_status = str(
            capture_provenance.get("status")
            or capture_provenance.get("classification")
            or "unspecified"
        )
        provenance_reasons = tuple(
            str(reason)
            for reason in (
                capture_provenance.get("reasons")
                or capture_provenance.get("intervention_reasons")
                or ()
            )
            if str(reason)
        )
        prefix_intervention_reasons = tuple(
            str(reason)
            for reason in (
                capture_provenance.get("prefix_intervention_reasons")
                or provenance_reasons
                or ()
            )
            if str(reason)
        )
        source_execution_id = str(
            capture_provenance.get("execution_id")
            or capture_provenance.get("source_execution_id")
            or ""
        )
        prefix_telemetry_complete = coerce_bool(
            capture_provenance.get("telemetry_complete"),
            True,
        )
        capture_execution_active = coerce_bool(
            capture_provenance.get("execution_active"),
            False,
        )
        if capture_execution_active and not external_state_capture_error:
            # The snapshot is a real prefix observation, but the run has not
            # crossed its reporting boundary yet.  Keep its capture-time
            # reasons and finalize the status after ``run()`` returns.
            provenance_status = "pending"
            prefix_telemetry_complete = False
            provenance_finalized = False
        else:
            provenance_finalized = True
        compact_state_enabled = str(
            os.environ.get("LSGEMU_SNAPSHOT_STATE_COMPRESSION", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        try:
            compact_state_level = min(
                9,
                max(
                    0,
                    int(
                        os.environ.get(
                            "LSGEMU_SNAPSHOT_STATE_COMPRESSION_LEVEL",
                            "1",
                        )
                    ),
                ),
            )
        except ValueError:
            compact_state_level = 1
        compact_external_model_state = (
            SnapshotStateBlob.from_mapping(
                external_model_state,
                compression_level=compact_state_level,
                store=self.blob_store,
            )
            if compact_state_enabled
            else external_model_state
        )
        if isinstance(compact_external_model_state, SnapshotStateBlob):
            self.capture_stats["external_state_raw_bytes"] += (
                compact_external_model_state.raw_size
            )
            self.capture_stats["external_state_stored_bytes"] += (
                compact_external_model_state.stored_size
            )
            if compact_external_model_state.is_compressed:
                self.capture_stats["external_state_compressed"] += 1
            else:
                self.capture_stats["external_state_uncompressed"] += 1
        else:
            self.capture_stats["external_state_fallback_dict"] += 1
            if not compact_state_enabled:
                self.capture_stats["external_state_compression_disabled"] += 1

        snapshot = BranchSnapshot(
            address=address,
            target=target,
            fallthrough=fallthrough,
            condition=condition,
            original_taken=original_taken,
            order=existing.order if existing is not None else self._next_order,
            depth=depth,
            registers=self.metadata_store.intern_mapping(registers),
            cpsr=cpsr,
            memory_data=memory_data,
            memory_base=memory_base,
            memory_size=memory_size
            ,
            memory_regions=memory_regions,
            mmio_state=self.metadata_store.intern_mapping({
                int(mmio_addr): int(value) & 0xFFFFFFFF
                for mmio_addr, value in (mmio_state or {}).items()
            }),
            alternatives=list(alternatives or []),
            original_index=original_index,
            occurrence_index=max(1, int(occurrence_index or 1)),
            capture_order=capture_order,
            dirty_pages=self.metadata_store.intern_set(dirty_pages),
            input_event_index=max(0, int(external_state.get("input_event_index", 0) or 0)),
            input_occurrence_counts=input_occurrence_counts,
            external_model_state=compact_external_model_state,
            region_hashes=self.metadata_store.intern_mapping(region_hashes),
            integrity_hash=integrity_hash,
            provenance_status=provenance_status,
            provenance_reasons=provenance_reasons,
            source_execution_id=source_execution_id,
            prefix_telemetry_complete=prefix_telemetry_complete,
            prefix_intervention_reasons=prefix_intervention_reasons,
            provenance_finalized=provenance_finalized,
        )

        if update_current:
            self._evict_current_snapshot_if_needed(address)
            self.snapshots[address] = snapshot
            self._touch_snapshot(address)
        if update_current and existing is None:
            self._next_order += 1
        self._remember_history_snapshot(snapshot)
        self.capture_stats["captures_succeeded"] += 1

        logger.debug(f"保存分支快照 @ 0x{address:08x}")

        return snapshot

    def _iter_unique_snapshots(self):
        """Yield current and historical snapshots exactly once."""
        seen: Set[int] = set()
        for snapshot in self.snapshots.values():
            identity = id(snapshot)
            if identity in seen:
                continue
            seen.add(identity)
            yield snapshot
        for history in self.snapshot_history.values():
            for snapshot in history:
                identity = id(snapshot)
                if identity in seen:
                    continue
                seen.add(identity)
                yield snapshot

    def finalize_snapshot_provenance(
        self,
        execution_id: object,
        *,
        successful: bool = True,
        telemetry_complete: bool = True,
        actual_intervention_reasons: Optional[List[str] | Tuple[str, ...] | Set[str]] = None,
    ) -> int:
        """Finalize snapshots captured by one execution.

        ``actual_intervention_reasons`` is accepted for audit compatibility,
        but is intentionally not copied into every snapshot: an intervention
        that occurred *after* a snapshot was captured must not invalidate that
        earlier prefix.  Capture-time reasons are authoritative for the
        snapshot's lineage.

        ``successful`` and ``telemetry_complete`` are used only when a legacy
        provider did not mark a prefix as synchronously observed.  A failed
        suffix therefore does not erase a valid prefix witness, while a
        capture/provider failure remains diagnostic.
        """
        del actual_intervention_reasons
        normalized_id = str(execution_id or "")
        if not normalized_id:
            return 0
        finalized = 0
        for snapshot in self._iter_unique_snapshots():
            source_id = str(getattr(snapshot, "source_execution_id", "") or "")
            if source_id != normalized_id:
                continue
            if bool(getattr(snapshot, "provenance_finalized", False)) and str(
                getattr(snapshot, "provenance_status", "") or ""
            ).lower() != "pending":
                continue

            raw_reasons = tuple(
                str(reason)
                for reason in (getattr(snapshot, "provenance_reasons", ()) or ())
                if str(reason)
            )
            reasons = tuple(
                reason
                for reason in raw_reasons
                if not is_environment_fact_reason(reason)
            )
            raw_status = str(
                getattr(snapshot, "provenance_status", "") or ""
            ).strip().lower()
            environment_only_diagnostic = bool(raw_reasons) and not reasons and all(
                is_environment_fact_reason(reason) for reason in raw_reasons
            )
            prefix_observed = bool(
                getattr(snapshot, "capture_complete", True)
                and getattr(snapshot, "memory_regions", None)
            )
            if reasons or (
                raw_status == "diagnostic" and not environment_only_diagnostic
            ):
                status = "diagnostic"
                prefix_complete = bool(telemetry_complete or prefix_observed)
            elif raw_status in {
                "unverified",
                "unknown",
                "unspecified",
                "invalid",
            }:
                # A provider that explicitly reported an unverified prefix
                # cannot be promoted merely because the suffix later
                # completed.  ``pending`` is the one lifecycle status that is
                # intentionally promoted after a synchronous capture.
                status = "unverified"
                prefix_complete = False
                reasons = (f"capture_provenance_status:{raw_status}",)
            elif prefix_observed:
                # Capture is synchronous with the BB/input hook, so a later
                # runtime failure does not retroactively change this prefix.
                status = "validated"
                prefix_complete = True
            elif successful and telemetry_complete:
                status = "validated"
                prefix_complete = True
            else:
                status = "unverified"
                prefix_complete = False
                reasons = ("snapshot_prefix_not_observed",)

            try:
                snapshot.provenance_status = status
                snapshot.provenance_reasons = tuple(dict.fromkeys(reasons))
                snapshot.prefix_intervention_reasons = tuple(
                    dict.fromkeys(
                        str(reason)
                        for reason in (
                            getattr(snapshot, "prefix_intervention_reasons", ()) or ()
                        )
                        if str(reason)
                    )
                )
                snapshot.prefix_telemetry_complete = bool(prefix_complete)
                snapshot.provenance_finalized = True
                finalized += 1
            except Exception:
                self.capture_stats["provenance_finalize_failures"] += 1
                continue
            self.capture_stats[f"provenance_finalized_{status}"] += 1
        if finalized:
            self.capture_stats["provenance_finalized"] += finalized
        return finalized

    def invalidate_snapshot_provenance(
        self,
        execution_id: object,
        *,
        reason: str = "snapshot_provenance_invalidated",
        error: object = None,
    ) -> int:
        """Terminally invalidate snapshots produced by one execution.

        A replay snapshot may already be referenced by a prefix/root/context
        wrapper when a later teardown operation fails.  Removing the wrapper
        would lose useful scheduling and diagnostic state, while leaving it
        validated would allow a suffix to inherit an execution whose evidence
        boundary was not completed.  This method therefore changes only the
        provenance contract: observed memory/CPU state, branch events, and
        coverage remain intact.

        The invalidation is idempotent.  ``provenance_finalized`` stays true so
        a later compatibility finalizer cannot accidentally promote the
        terminal diagnostic state back to validated.
        """
        normalized_id = str(execution_id or "")
        if not normalized_id:
            return 0
        reason_text = str(reason or "snapshot_provenance_invalidated")[:256]
        invalidation_reasons = (reason_text,)
        if error not in (None, ""):
            # Keep detailed cleanup text out of the frequently copied reason
            # tuple; the execution record owns the bounded error payload.
            invalidation_reasons = (reason_text,)

        def read(source: object, key: str, default: object = None) -> object:
            return _snapshot_value(source, key, default)

        def write(source: object, key: str, value: object) -> None:
            if isinstance(source, Mapping):
                source[key] = value
            else:
                setattr(source, key, value)

        invalidated = 0
        for snapshot in self._iter_unique_snapshots():
            source_id = str(read(snapshot, "source_execution_id", "") or "")
            if source_id != normalized_id:
                continue
            already_invalidated = coerce_bool(
                read(snapshot, "provenance_invalidated", False),
                False,
            )
            existing_reasons = tuple(
                str(item)
                for item in (read(snapshot, "provenance_reasons", ()) or ())
                if str(item)
            )
            existing_invalidation_reasons = tuple(
                str(item)
                for item in (
                    read(snapshot, "provenance_invalidation_reasons", ()) or ()
                )
                if str(item)
            )
            if already_invalidated and reason_text in existing_invalidation_reasons:
                # Cleanup can be reported by more than one owner (for example
                # the MMIO hook and the lease).  Repeating the same terminal
                # transition must not inflate invalidation statistics or make
                # a later report look as if multiple snapshots were created.
                continue
            merged_reasons = tuple(
                dict.fromkeys(existing_reasons + invalidation_reasons)
            )
            merged_invalidation_reasons = tuple(
                dict.fromkeys(
                    existing_invalidation_reasons + invalidation_reasons
                )
            )
            try:
                write(snapshot, "provenance_status", "diagnostic")
                write(snapshot, "provenance_reasons", merged_reasons)
                write(
                    snapshot,
                    "prefix_intervention_reasons",
                    tuple(
                        dict.fromkeys(
                            tuple(
                                str(item)
                                for item in (
                                    read(
                                        snapshot,
                                        "prefix_intervention_reasons",
                                        (),
                                    )
                                    or ()
                                )
                                if str(item)
                            )
                            + invalidation_reasons
                        )
                    ),
                )
                write(snapshot, "prefix_telemetry_complete", False)
                write(snapshot, "provenance_finalized", True)
                write(snapshot, "provenance_invalidated", True)
                write(
                    snapshot,
                    "provenance_invalidation_reasons",
                    merged_invalidation_reasons,
                )
            except Exception:
                self.capture_stats["provenance_invalidation_failures"] += 1
                continue
            if not already_invalidated:
                invalidated += 1

        if invalidated:
            self.capture_stats["provenance_invalidated"] += invalidated
            self.capture_stats[
                f"provenance_invalidated_{reason_text}"
            ] += invalidated
        return invalidated

    def _touch_snapshot(self, address: int) -> None:
        self._next_access_order += 1
        self._snapshot_last_used[int(address)] = self._next_access_order

    def _evict_current_snapshot_if_needed(self, address: int) -> None:
        """Apply the optional current-snapshot cap before inserting *address*.

        Eviction releases only a reusable CPU/RAM state.  It never removes
        branch events, occurrence events, or coverage, so the cap is a host
        memory policy and not a hidden coverage policy.
        """
        limit = int(self.max_current_snapshots or 0)
        if limit <= 0 or int(address) in self.snapshots:
            return
        while len(self.snapshots) >= limit:
            candidates = [
                candidate for candidate in self.snapshots if int(candidate) != int(address)
            ]
            if not candidates:
                return
            victim = min(
                candidates,
                key=lambda item: (
                    int(self._snapshot_last_used.get(int(item), 0) or 0),
                    int(getattr(self.snapshots[item], "capture_order", 0) or 0),
                ),
            )
            self.snapshots.pop(int(victim), None)
            self._snapshot_last_used.pop(int(victim), None)
            self.capture_stats["current_snapshot_evictions"] += 1

    def should_save_history_snapshot(self, address: int, occurrence_index: int) -> bool:
        """Whether a non-current history snapshot is still useful for this branch."""
        if self.max_history_per_address <= 0 or self.max_history_total <= 0:
            return False
        history = self.snapshot_history.get(address, [])
        occurrence_index = max(1, int(occurrence_index or 1))
        if not any(int(getattr(snapshot, "occurrence_index", 1)) == occurrence_index for snapshot in history):
            return True
        return len(history) < self.max_history_per_address

    def _remember_history_snapshot(self, snapshot: BranchSnapshot):
        if self.max_history_per_address <= 0 or self.max_history_total <= 0:
            return
        history = self.snapshot_history.setdefault(snapshot.address, [])
        occurrence_index = int(getattr(snapshot, "occurrence_index", 1) or 1)
        for index, existing in enumerate(history):
            if int(getattr(existing, "occurrence_index", 1) or 1) == occurrence_index:
                # Keep the latest concrete state for the same dynamic occurrence.
                history[index] = snapshot
                break
        else:
            history.append(snapshot)
        history.sort(key=lambda item: int(getattr(item, "capture_order", getattr(item, "order", 0)) or 0))
        while len(history) > self.max_history_per_address:
            history.pop(0)
            self.capture_stats["history_per_address_evictions"] += 1
        self._trim_history_total()

    def _trim_history_total(self):
        if self.max_history_total <= 0:
            self.snapshot_history.clear()
            return
        while sum(len(items) for items in self.snapshot_history.values()) > self.max_history_total:
            oldest_addr = None
            oldest_order = None
            for address, items in self.snapshot_history.items():
                if not items:
                    continue
                order = int(getattr(items[0], "capture_order", getattr(items[0], "order", 0)) or 0)
                if oldest_order is None or order < oldest_order:
                    oldest_addr = address
                    oldest_order = order
            if oldest_addr is None:
                return
            items = self.snapshot_history.get(oldest_addr, [])
            if items:
                items.pop(0)
                self.capture_stats["history_total_evictions"] += 1
            if not items:
                self.snapshot_history.pop(oldest_addr, None)

    def restore_snapshot(self, uc, snapshot: BranchSnapshot) -> bool:
        """
        恢复快照

        Args:
            uc: Unicorn实例
            snapshot: 快照

        Returns:
            是否成功
        """
        from unicorn.arm_const import (
            UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
            UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
            UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
            UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
            UC_ARM_REG_CPSR, UC_ARM_REG_FPSCR
        )
        from unicorn import arm_const as _arm_const

        try:
            if not coerce_bool(
                _snapshot_value(snapshot, "capture_complete", True),
                True,
            ):
                raise SnapshotIntegrityError("snapshot was not captured completely")
            memory_regions = _snapshot_value(snapshot, "memory_regions", None) or {}
            if not memory_regions:
                memory_regions = {
                    (
                        int(_snapshot_value(snapshot, "memory_base", 0)),
                        int(_snapshot_value(snapshot, "memory_size", 0)),
                    ): _snapshot_value(snapshot, "memory_data", b"")
                }
            for (memory_base, memory_size), memory_data in memory_regions.items():
                if int(memory_size) <= 0 or len(memory_data) != int(memory_size):
                    raise SnapshotIntegrityError(
                        f"region 0x{int(memory_base):08x} length {len(memory_data)} != {int(memory_size)}"
                    )
                expected_hash = dict(
                    _snapshot_value(snapshot, "region_hashes", {}) or {}
                ).get(
                    (memory_base, memory_size)
                )
                if expected_hash and self._region_digest(memory_data) != str(expected_hash):
                    raise SnapshotIntegrityError(
                        f"region 0x{int(memory_base):08x} digest mismatch"
                    )
            expected_integrity = str(
                _snapshot_value(snapshot, "integrity_hash", "") or ""
            )
            snapshot_registers = dict(
                _snapshot_value(snapshot, "registers", {}) or {}
            )
            snapshot_cpsr = int(_snapshot_value(snapshot, "cpsr", 0) or 0)
            if expected_integrity and self._integrity_digest(
                snapshot_registers, snapshot_cpsr, memory_regions
            ) != expected_integrity:
                raise SnapshotIntegrityError("snapshot integrity hash mismatch")

            # 恢复寄存器
            reg_map = {
                'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1,
                'r2': UC_ARM_REG_R2, 'r3': UC_ARM_REG_R3,
                'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
                'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7,
                'r8': UC_ARM_REG_R8, 'r9': UC_ARM_REG_R9,
                'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
                'r12': UC_ARM_REG_R12, 'sp': UC_ARM_REG_SP,
                'lr': UC_ARM_REG_LR, 'pc': UC_ARM_REG_PC
            }
            # P0-A（cycle3 k.5 C2）：恢复面与捕获面同步补 VFP。硬条款：
            # FPSCR 必须经下方循环写回（即先于 CPSR 写）——次序颠倒会破坏
            # 条件 VFP 指令的标志恢复。旧快照无这些键时循环自然跳过。
            for _i in range(32):
                reg_map[f'd{_i}'] = getattr(_arm_const, f'UC_ARM_REG_D{_i}')
            reg_map['fpscr'] = UC_ARM_REG_FPSCR

            for name, reg_id in reg_map.items():
                if name in snapshot_registers:
                    uc.reg_write(reg_id, snapshot_registers[name])

            # 恢复CPSR
            uc.reg_write(UC_ARM_REG_CPSR, snapshot_cpsr)

            # 恢复内存
            for (memory_base, _memory_size), memory_data in memory_regions.items():
                uc.mem_write(int(memory_base), bytes(memory_data))

            logger.debug(
                "恢复分支快照 @ 0x%08x",
                int(_snapshot_value(snapshot, "address", 0) or 0),
            )

            return True

        except Exception as e:
            self.capture_stats["restore_failures"] += 1
            if isinstance(e, SnapshotIntegrityError):
                self.capture_stats["restore_integrity_failures"] += 1
            logger.error(f"恢复快照失败: {e}")
            return False

    def flip_branch(self, uc, snapshot: BranchSnapshot, take: bool) -> bool:
        """
        翻转分支

        Args:
            uc: Unicorn实例
            snapshot: 快照
            take: True=跳转, False=不跳转

        Returns:
            是否成功
        """
        # 先恢复快照
        if not self.restore_snapshot(uc, snapshot):
            return False

        # 修改CPSR保留给后续代码中读取标志位的场景；实际分支方向直接通过PC强制。
        if not self._modify_cpsr(uc, snapshot.condition, take):
            return False

        # 设置PC到强制方向。不要把PC设为branch+size后再依赖CPSR，否则分支指令已经被跳过。
        from unicorn.arm_const import UC_ARM_REG_PC

        next_pc = snapshot.target if take else snapshot.fallthrough
        uc.reg_write(UC_ARM_REG_PC, next_pc | 1)

        logger.info(f"翻转分支 @ 0x{snapshot.address:08x} -> {'taken' if take else 'not_taken'}")

        return True

    def _modify_cpsr(self, uc, condition: str, take: bool) -> bool:
        """
        修改CPSR标志位

        Args:
            uc: Unicorn实例
            condition: 条件码
            take: True=跳转, False=不跳转

        Returns:
            是否成功
        """
        from unicorn.arm_const import UC_ARM_REG_CPSR

        try:
            if str(condition).upper().startswith("IT"):
                condition = str(condition).upper()[2:]
            cpsr = uc.reg_read(UC_ARM_REG_CPSR)

            # 标志位
            N_FLAG = 1 << 31
            Z_FLAG = 1 << 30
            C_FLAG = 1 << 29
            V_FLAG = 1 << 28

            new_cpsr = cpsr

            if condition == 'EQ':  # Z=1
                new_cpsr = (new_cpsr | Z_FLAG) if take else (new_cpsr & ~Z_FLAG)
            elif condition == 'NE':  # Z=0
                new_cpsr = (new_cpsr & ~Z_FLAG) if take else (new_cpsr | Z_FLAG)
            elif condition == 'CS' or condition == 'HS':  # C=1
                new_cpsr = (new_cpsr | C_FLAG) if take else (new_cpsr & ~C_FLAG)
            elif condition == 'CC' or condition == 'LO':  # C=0
                new_cpsr = (new_cpsr & ~C_FLAG) if take else (new_cpsr | C_FLAG)
            elif condition == 'MI':  # N=1
                new_cpsr = (new_cpsr | N_FLAG) if take else (new_cpsr & ~N_FLAG)
            elif condition == 'PL':  # N=0
                new_cpsr = (new_cpsr & ~N_FLAG) if take else (new_cpsr | N_FLAG)
            elif condition == 'VS':  # V=1
                new_cpsr = (new_cpsr | V_FLAG) if take else (new_cpsr & ~V_FLAG)
            elif condition == 'VC':  # V=0
                new_cpsr = (new_cpsr & ~V_FLAG) if take else (new_cpsr | V_FLAG)
            elif condition == 'HI':  # C=1 and Z=0
                if take:
                    new_cpsr |= C_FLAG
                    new_cpsr &= ~Z_FLAG
                else:
                    new_cpsr &= ~C_FLAG
            elif condition == 'LS':  # C=0 or Z=1
                if take:
                    new_cpsr &= ~C_FLAG
                else:
                    new_cpsr |= C_FLAG
                    new_cpsr &= ~Z_FLAG
            elif condition == 'GE':  # N=V
                n = (new_cpsr & N_FLAG) != 0
                v = (new_cpsr & V_FLAG) != 0
                if take:
                    if n != v:
                        new_cpsr ^= V_FLAG
                else:
                    if n == v:
                        new_cpsr ^= V_FLAG
            elif condition == 'LT':  # N!=V
                n = (new_cpsr & N_FLAG) != 0
                v = (new_cpsr & V_FLAG) != 0
                if take:
                    if n == v:
                        new_cpsr ^= V_FLAG
                else:
                    if n != v:
                        new_cpsr ^= V_FLAG
            elif condition == 'GT':  # Z=0 and N=V
                if take:
                    new_cpsr &= ~Z_FLAG
                    n = (new_cpsr & N_FLAG) != 0
                    v = (new_cpsr & V_FLAG) != 0
                    if n != v:
                        new_cpsr ^= V_FLAG
                else:
                    new_cpsr |= Z_FLAG
            elif condition == 'LE':  # Z=1 or N!=V
                if take:
                    new_cpsr |= Z_FLAG
                else:
                    new_cpsr &= ~Z_FLAG
                    n = (new_cpsr & N_FLAG) != 0
                    v = (new_cpsr & V_FLAG) != 0
                    if n != v:
                        new_cpsr ^= V_FLAG

            uc.reg_write(UC_ARM_REG_CPSR, new_cpsr)

            logger.debug(f"CPSR: 0x{cpsr:08x} -> 0x{new_cpsr:08x} ({condition}, {'take' if take else 'not_take'})")

            return True

        except Exception as e:
            logger.error(f"修改CPSR失败: {e}")
            return False

    def get_snapshot(self, address: int) -> Optional[BranchSnapshot]:
        """获取快照"""
        snapshot = self.snapshots.get(address)
        if snapshot is not None:
            self._touch_snapshot(address)
        return snapshot

    def has_snapshot(self, address: int) -> bool:
        """是否有快照"""
        present = address in self.snapshots
        if present:
            self._touch_snapshot(address)
        return present

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_snapshots': len(self.snapshots),
            'max_current_snapshots': int(self.max_current_snapshots),
            'current_snapshot_limit_enabled': bool(self.max_current_snapshots > 0),
            'current_snapshot_evictions': int(
                self.capture_stats.get('current_snapshot_evictions', 0) or 0
            ),
            'snapshot_history_entries': sum(
                len(items) for items in self.snapshot_history.values()
            ),
            'catalog_events': len(self.events),
            'occurrence_events': len(self.occurrence_events),
            'occurrence_event_sequence': int(self._next_event_order),
            'occurrence_events_truncated': int(self.occurrence_events_truncated),
            'snapshot_addresses': [hex(addr) for addr in sorted(self.snapshots.keys())[:10]],
            'capture': dict(self.capture_stats),
            'provenance_pending_snapshots': sum(
                1
                for snapshot in self._iter_unique_snapshots()
                if str(getattr(snapshot, 'provenance_status', '') or '').lower()
                == 'pending'
            ),
            'provenance_finalized_snapshots': sum(
                1
                for snapshot in self._iter_unique_snapshots()
                if bool(getattr(snapshot, 'provenance_finalized', False))
            ),
            'page_store': self.page_store.get_statistics(),
            'metadata_store': self.metadata_store.get_statistics(),
            'blob_store': (
                self.blob_store.get_statistics()
                if self.blob_store is not None
                else {}
            ),
        }

    def get_ordered_snapshot_history(self):
        """Return bounded per-address history snapshots for precise event matching."""
        seen = set()
        snapshots = []
        for history in self.snapshot_history.values():
            for snapshot in history:
                identity = id(snapshot)
                if identity in seen:
                    continue
                seen.add(identity)
                snapshots.append(snapshot)
        for snapshot in self.snapshots.values():
            identity = id(snapshot)
            if identity in seen:
                continue
            seen.add(identity)
            snapshots.append(snapshot)
        return sorted(
            snapshots,
            key=lambda snapshot: int(getattr(snapshot, "capture_order", getattr(snapshot, "order", 0)) or 0),
        )

    def get_ordered_snapshots(self):
        """按主路径发现顺序返回当前稳定分支快照。"""
        return sorted(self.snapshots.values(), key=lambda snapshot: snapshot.order)

    def get_ordered_events(self):
        """Return the compact edge catalog in observation order."""
        return sorted(self.events, key=lambda event: event.order)

    def get_ordered_occurrence_events(self):
        """Return every retained dynamic branch occurrence in execution order."""
        return list(self.occurrence_events)

    def get_occurrence_event_cursor(self) -> int:
        """Return the monotonic number of observed branch occurrences.

        This cursor is deliberately independent from retained trace capacity.
        It is suitable for correlating external-input events with branch
        intervals, while ``get_ordered_occurrence_events`` remains the bounded
        payload used for concrete witness matching.
        """
        return int(self._next_event_order)
