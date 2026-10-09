#!/usr/bin/env python3
"""
Register Tracer - MMIO 反向追踪模块

功能:
1. 追踪寄存器的数据来源
2. 识别寄存器值来自哪个 MMIO 地址
3. 建立寄存器-MMIO 映射关系
"""

import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Set, Optional, Tuple, List, Iterable, Protocol
from dataclasses import dataclass, field, replace
from unicorn import *
from unicorn.arm_const import *
import unicorn.arm_const as arm_const

from ..analysis.differential_probe import DifferentialProbeRequest
from ..dynamic_constraint_recovery import (
    DynamicConstraintModel,
    DynamicExpressionGraph,
    DynamicInputAssignment,
    DynamicRecoveryResult,
)
from ..hook_lifecycle import managed_hook_add, managed_hook_del
# r19 产生者语义（B1 接线）：单指令 NZCV 写判定与索引同源，避免 tracer 侧
# 复制一份产生者表漂移。
from ..prepared_firmware import _producer_flag_writes

logger = logging.getLogger(__name__)


@dataclass
class RegisterSource:
    """寄存器来源信息"""
    register: str           # 寄存器名称 (r0, r1, ...)
    source_type: str        # 来源类型: 'mmio', 'immediate', 'register', 'memory'
    mmio_address: Optional[int] = None  # MMIO 地址
    mmio_value: Optional[int] = None    # MMIO 读取的值
    source_pc: Optional[int] = None     # 来源指令的 PC
    origin_source_pc: Optional[int] = None  # 原始读取/定义的 PC
    instruction: Optional[str] = None   # 来源指令
    memory_address: Optional[int] = None
    memory_role: Optional[str] = None
    memory_base: Optional[int] = None
    memory_offset: Optional[int] = None
    memory_access_pc: Optional[int] = None
    memory_size: Optional[int] = None
    operation: Optional[str] = None
    expression: Optional[str] = None
    chain: Tuple[str, ...] = field(default_factory=tuple)
    call_depth: int = 0
    source_occurrence: Optional[int] = None
    # A register can be derived from multiple independent MMIO/RAM reads, e.g.
    # `(MMIO_A & mask) | (RAM_B & mask)`.  The legacy fields above keep the
    # primary source for backward compatibility; dependency_sites preserves the
    # full set for branch-level combined-constraint replay.
    dependency_sites: Tuple[Tuple[str, int, Optional[int], Optional[int]], ...] = field(default_factory=tuple)
    opaque_dependency: bool = False
    causal_complete: bool = True
    # B2：影子内存逐字节记录「落盘时的具体字节值」。load 转发前用它做值相等
    # 校验（对齐 :1833-1843 返回值回填启发）；None 表示无值可校（保持旧行为）。
    stored_value: Optional[int] = None


@dataclass(frozen=True)
class DynamicLLMSliceRequest:
    """Pure-data escalation request handed to the upper-layer LLM solver.

    Built by the tracer at the point where the symbolic path has already
    failed, so the receiver never needs the expression graph itself: the
    instruction-level slice, the observed input sites and the failure summary
    fully describe the solving task.
    """

    branch_pc: int
    branch_occurrence: int
    condition: str
    compare_op: str
    compare_pc: int
    target_taken: bool
    relation_kind: str = "unknown"
    instruction_lines: Tuple[str, ...] = ()
    constraint_note: str = ""
    input_sites: Tuple[Dict[str, object], ...] = ()
    coupled_input_count: int = 0
    same_variable_constraints: int = 0
    slice_instruction_count: int = 0
    truncated_instructions: int = 0
    prior_failure: Dict[str, object] = field(default_factory=dict)


class DynamicLLMSliceFallback(Protocol):
    """Escalation hook: solve one slice with an LLM, replay-gated upstream.

    The implementation may only assign values to the request's observed input
    sites; the returned assignments must describe sites that were actually
    observed, and every model still passes the caller's force-free replay
    validation before it can contribute coverage.
    """

    def __call__(
        self,
        request: DynamicLLMSliceRequest,
    ) -> Optional[List[DynamicInputAssignment]]: ...


class DifferentialProbeFallback(Protocol):
    """二级定位钩子：表达式图无该分支记录时的差分扰动探测（设计 §4.2）。

    实现方负责候选来源（运行时 MMIO 访问历史 + 静态地址清单）与重放
    oracle；返回的 assignments 是扰动见证值（hypothesis），与 z3/LLM
    结果走同一候选通道，仍由上层无强制重放验证把关。
    """

    def __call__(
        self,
        request: DifferentialProbeRequest,
    ) -> Optional[List[DynamicInputAssignment]]: ...


class RegisterTracer:
    """寄存器追踪器 - MMIO 反向追踪"""

    def __init__(self, uc, static_bbs, instruction_lookup: Optional[Dict[int, Dict]] = None,
                 compare_lookup: Optional[Dict[int, Dict]] = None,
                 external_memory_input_predicate: Optional[Callable[[int, int], bool]] = None,
                 dynamic_graph_enabled: Optional[bool] = None,
                 external_input_event_source: Optional[object] = None,
                 dynamic_llm_fallback: Optional[DynamicLLMSliceFallback] = None,
                 differential_probe_fallback: Optional[DifferentialProbeFallback] = None,
                 producer_index: Optional[Dict[int, object]] = None):
        """
        初始化

        Args:
            uc: Unicorn 实例
            static_bbs: 静态基本块 {bb_addr: [instructions]}
            instruction_lookup: 指令地址索引（可选）
            compare_lookup: 分支对应比较指令索引（可选）
            producer_index: r19 NZCV 产生者索引（可选）——B1 起 Bcc 站点的
                快照触发改为「最近产生者延迟提交」，见 _record_flag_producer_snapshot
        """
        self.uc = uc
        self.static_bbs = static_bbs
        self.instruction_lookup = instruction_lookup or self._build_instruction_lookup()
        self.compare_lookup = compare_lookup or self._build_compare_lookup()
        self.compare_pc_to_branch_pcs = self._build_compare_pc_to_branch_lookup()
        self.producer_index = dict(producer_index or {})
        self.producer_pc_to_branch_pcs = self._build_producer_pc_to_branch_lookup()
        # B1：producer_index 内的 Bcc 站点改走「最近产生者延迟提交」；
        # 旧 compare 直录路径只为索引外站点保留（CBZ 值通道等行为不变）。
        self._deferred_branch_pcs: Set[int] = set(self.producer_index.keys())
        self._pending_flag_snapshot: Optional[Dict[str, object]] = None
        # B3：VFP 双槽——VCMP/VMSR 写 FPSCR 槽，VMRS APSR 转发进 NZCV 槽。
        self._pending_fpscr_snapshot: Optional[Dict[str, object]] = None
        self._flag_generation = 0
        self._flag_writer_profile_cache: Dict[int, Tuple[bool, List[str]]] = {}
        self._vfp_flag_profile_cache: Dict[int, Tuple[Optional[Tuple[str, List[str]]],]] = {}
        self.instruction_to_bb = self._build_instruction_to_bb()

        # 寄存器来源映射 {register: RegisterSource}
        self.register_sources: Dict[str, RegisterSource] = {}

        # MMIO 访问历史 [(pc, mmio_addr, value, is_read)]
        self.mmio_history: List[Tuple[int, int, int, bool]] = []
        self.mmio_history_limit = max(
            1024,
            self._env_int("LSGEMU_REGISTER_TRACER_MMIO_HISTORY_LIMIT", 16384),
        )
        self.mmio_history_total = 0
        self.mmio_history_entries_discarded = 0
        self.mmio_read_total = 0
        self.mmio_write_total = 0
        self.mmio_read_occurrence_counts: Dict[Tuple[int, int], int] = {}
        self.external_memory_read_occurrence_counts: Dict[Tuple[int, int], int] = {}

        # 寄存器-MMIO 映射 {register: mmio_addr}
        self.reg_mmio_map: Dict[str, int] = {}

        # RAM shadow provenance. Stored per byte so STRB/LDRB and unaligned
        # accesses do not accidentally inherit a whole-word source.
        self.memory_sources: Dict[int, RegisterSource] = {}

        # Lightweight call provenance. We do not force path changes here; this
        # only keeps argument/return context available while execution crosses BL.
        self.call_stack: List[Dict[str, object]] = []
        self.completed_call_frames: List[Dict[str, object]] = []
        self.max_call_depth_seen = 0
        self.branch_dependency_snapshots: Dict[int, List[Dict[str, object]]] = {}
        self.mmio_hooks: List[int] = []
        self.mmio_hook = None
        self.ram_hooks: List[int] = []
        self.ram_hook = None
        self.code_hook = None
        self.min_branch_snapshot_variants = self._env_int("LSGEMU_BRANCH_DEP_SNAPSHOT_MIN", 16)
        self.max_branch_snapshot_variants = self._env_int("LSGEMU_BRANCH_DEP_SNAPSHOT_MAX", 64)
        self.external_memory_input_predicate = external_memory_input_predicate
        # IntelligentEmulator can service mapped MMIO loads directly from a
        # code hook.  In that case Unicorn never reaches UC_HOOK_MEM_READ, so
        # subscribe to the emulator's completed input events when available.
        self.external_input_event_source = external_input_event_source
        self._external_input_event_callback = self._external_input_event_hook
        self._external_input_observer_attached = False
        self._preloaded_external_loads: Set[Tuple[int, str]] = set()
        self.external_input_event_stats: Dict[str, int] = {
            "received": 0,
            "mmio": 0,
            "external_memory": 0,
            "unsupported": 0,
            "target_register_missing": 0,
            "preloaded_loads_preserved": 0,
        }
        self.provenance_kill_stats: Counter[str] = Counter()
        self.dynamic_graph_enabled = (
            bool(dynamic_graph_enabled)
            if dynamic_graph_enabled is not None
            else os.environ.get("LSGEMU_DYNAMIC_CONSTRAINT_RECOVERY", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.dynamic_graph_archive_limit = self._env_int(
            "LSGEMU_DYNAMIC_GRAPH_ARCHIVE_LIMIT", 8
        )
        self.dynamic_graph_archive_node_limit = self._env_int(
            "LSGEMU_DYNAMIC_GRAPH_ARCHIVE_NODE_LIMIT", 300000
        )
        self.dynamic_graph_archives: List[DynamicExpressionGraph] = []
        self.dynamic_graph = self._new_dynamic_graph()
        self.dynamic_solver_escalation_stats: Counter[str] = Counter()
        self.dynamic_checksum_fallback_stats: Counter[str] = Counter()
        # 「符号求解失败 → 大模型求解切片」升级钩子，由上层（historical_runner）
        # 注入实现；为 None 时升级链不触发，行为与原先一致。
        self.dynamic_llm_fallback: Optional[DynamicLLMSliceFallback] = dynamic_llm_fallback
        self.dynamic_llm_fallback_stats: Counter[str] = Counter()
        # 二级定位钩子（设计 §4.2）：表达式图对该分支没有任何记录时，
        # 交给上层做有界差分扰动；为 None 时保持原先的死路返回。
        self.differential_probe_fallback: Optional[DifferentialProbeFallback] = (
            differential_probe_fallback
        )
        self.differential_probe_stats: Counter[str] = Counter()
        # B2 影子内存反饥饿：
        # - tombstone（默认开）：未知来源的写不再级联清空影子，而是落
        #   「该地址被写过但来源未知」的 opaque 条目——只表达未知，不携带任何
        #   MMIO 身份（假来源红线：宁可 opaque，不造过期 MMIO 来源）。
        # - 值相等保留（默认关）：未知来源的写若逐字节等于影子既有条目的
        #   stored_value，保留旧来源。值相等不证明来源相等（可能撞值），
        #   假阳性概率见 /tmp/dfs_r20b.md 审计，故默认关闭、显式开启。
        self._shadow_tombstone_enabled = os.environ.get(
            "LSGEMU_SHADOW_TOMBSTONE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._shadow_value_equal_retain = os.environ.get(
            "LSGEMU_SHADOW_VALUE_EQUAL_RETAIN", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # r23 计量：寄存器写事件日志 + 比较点死操作数回溯（默认关，诊断探针
        # 显式开启）。关时全程零开销（每个 note 入口先查本开关）。
        # 事件 = {pc, mnemonic, kind, sites(写后依赖点), inputs((reg, 事件引用)),
        # depth}；只留每寄存器栈顶，引用链 depth≤16 封顶防自增链无限滞留。
        self._lineage_walk_enabled = os.environ.get(
            "LSGEMU_LINEAGE_WALK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._lineage_last_write: Dict[str, Dict[str, object]] = {}
        self._lineage_walk_cache: Dict[Tuple[str, int, str], Dict[str, object]] = {}
        self._lineage_note_context: Optional[Dict[str, object]] = None
        self.provenance_dead_operand_stats: Counter[str] = Counter()

    def _append_mmio_history(self, record: Tuple[int, int, int, bool]) -> None:
        """Retain recent raw accesses; occurrence counters remain lossless."""
        self.mmio_history.append(record)
        self.mmio_history_total += 1
        if bool(record[3]):
            self.mmio_read_total += 1
        else:
            self.mmio_write_total += 1
        if len(self.mmio_history) > self.mmio_history_limit * 2:
            discarded = len(self.mmio_history) - self.mmio_history_limit
            del self.mmio_history[:discarded]
            self.mmio_history_entries_discarded += discarded

    @staticmethod
    def _mmio_ranges() -> List[Tuple[int, int]]:
        return [
            (0x40000000, 0x60000000),
            (0xE0000000, 0xE0100000),
        ]

    @staticmethod
    def _ram_ranges() -> List[Tuple[int, int]]:
        """B4：RAM 影子 hook 区间——SRAM + CCM（F427 核心 64KB）。"""
        return [
            (0x20000000, 0x20100000),
            (0x10000000, 0x10010000),
        ]

    @staticmethod
    def _is_mmio_address(address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        return any(start <= address < end for start, end in RegisterTracer._mmio_ranges())

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default)), 0))
        except ValueError:
            return default

    def _build_instruction_lookup(self) -> Dict[int, Dict]:
        """构建指令地址索引"""
        lookup: Dict[int, Dict] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                lookup[insn['address']] = insn
        return lookup

    def _build_instruction_to_bb(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for bb_start, instructions in self.static_bbs.items():
            for instruction in instructions or []:
                try:
                    lookup[int(instruction.get("address", 0) or 0)] = int(bb_start)
                except Exception:
                    continue
        return lookup

    def _new_dynamic_graph(self) -> DynamicExpressionGraph:
        return DynamicExpressionGraph(instruction_to_bb=self.instruction_to_bb)

    def set_external_memory_input_predicate(
        self,
        predicate: Optional[Callable[[int, int], bool]],
    ) -> None:
        self.external_memory_input_predicate = predicate

    def set_dynamic_llm_fallback(
        self,
        fallback: Optional[DynamicLLMSliceFallback],
    ) -> None:
        """Attach (or detach) the upper-layer LLM slice solver at runtime."""
        self.dynamic_llm_fallback = fallback

    def set_differential_probe_fallback(
        self,
        fallback: Optional[DifferentialProbeFallback],
    ) -> None:
        """Attach (or detach) the upper-layer differential-probe solver."""
        self.differential_probe_fallback = fallback

    def _dynamic_llm_fallback_eligible(self, result: DynamicRecoveryResult) -> bool:
        """Escalate on every symbolic failure except authoritative UNSAT.

        An exact-record UNSAT with no omitted path predicates is a proof that
        no assignment of the recorded causal inputs can satisfy the target;
        the naturalization layer already treats it as terminal, and an LLM
        cannot produce a model for a proven-unsat formula over the same leaf
        set (replay would reject it anyway).
        """
        if result.models:
            return False
        if result.solver_status in {
            "unsupported", "unknown", "budget", "unavailable", "not_run",
        }:
            return True
        return bool(result.fallback_eligible)

    def _dynamic_llm_slice_solve(
        self,
        *,
        graph: DynamicExpressionGraph,
        branch_pc: int,
        occurrence: int,
        target_taken: bool,
        failure_result: DynamicRecoveryResult,
        avoid_values_by_site: Optional[Dict[Tuple[object, ...], Set[int]]],
    ) -> Optional[DynamicRecoveryResult]:
        """Escalate one failed symbolic solve to the injected LLM solver.

        The returned models are hypotheses (`solver_status="hypothesis"`):
        they re-enter the exact same candidate-set channel as z3 models and
        remain gated by the caller's force-free replay validation.
        """
        fallback = self.dynamic_llm_fallback
        if fallback is None:
            return None
        stats = self.dynamic_llm_fallback_stats
        stats["calls"] += 1
        stats[f"trigger_{failure_result.solver_status}"] += 1
        try:
            slice_export = graph.export_instruction_slice(
                branch_pc=int(branch_pc),
                occurrence=max(1, int(occurrence)),
                instruction_lookup=self.instruction_lookup,
            )
        except Exception as exc:  # pragma: no cover - defensive
            stats["slice_export_errors"] += 1
            logger.debug("LLM切片导出失败 @%s: %s", hex(int(branch_pc)), exc)
            return None
        if slice_export is None:
            stats["slice_export_missing"] += 1
            return None

        input_sites: List[Dict[str, object]] = []
        for index, item in enumerate(slice_export.input_sites):
            site = graph._input_site_identity(item)
            failed_values = sorted(
                {
                    int(value) & 0xFFFFFFFF
                    for value in ((avoid_values_by_site or {}).get(site) or set())
                }
            )
            input_sites.append({
                "site_index": index,
                "kind": str(item.kind),
                "address": int(item.address) & 0xFFFFFFFF,
                "read_pc": int(item.read_pc) & 0xFFFFFFFF,
                "occurrence": int(item.occurrence),
                "width": int(item.width),
                "current_value": int(item.observed_value) & 0xFFFFFFFF,
                "trace_event_id": int(item.trace_event_id or 0),
                "failed_values": failed_values,
            })

        request = DynamicLLMSliceRequest(
            branch_pc=int(slice_export.branch_pc) & 0xFFFFFFFF,
            branch_occurrence=int(slice_export.branch_occurrence),
            condition=str(slice_export.condition),
            compare_op=str(slice_export.compare_op),
            compare_pc=int(slice_export.compare_pc) & 0xFFFFFFFF,
            target_taken=bool(target_taken),
            relation_kind=str(slice_export.relation_kind),
            instruction_lines=tuple(slice_export.format_instruction_lines()),
            constraint_note=str(slice_export.format_constraint_note()),
            input_sites=tuple(input_sites),
            coupled_input_count=len(input_sites),
            same_variable_constraints=int(slice_export.same_variable_constraints),
            slice_instruction_count=len(slice_export.lines),
            truncated_instructions=int(slice_export.truncated_instructions),
            prior_failure={
                "dynamic_solver_status": str(failure_result.solver_status),
                "dynamic_solver_reason": str(failure_result.reason),
                "dynamic_initial_solver_status": str(
                    failure_result.initial_solver_status
                ),
                "dynamic_solver_escalated": bool(failure_result.escalated),
                "dynamic_target_inputs": int(failure_result.target_inputs),
                "dynamic_path_predicates": int(failure_result.path_predicates),
                "dynamic_slice_nodes": int(failure_result.slice_nodes),
                "dynamic_omitted_path_predicates": int(
                    failure_result.omitted_path_predicates
                ),
                "dynamic_branch_record_match": str(failure_result.branch_record_match),
            },
        )
        stats["attempted"] += 1
        try:
            assignments = fallback(request)
        except Exception as exc:
            stats["errors"] += 1
            logger.warning(
                "LLM切片求解升级失败 @%s occ=%s: %s",
                hex(int(branch_pc)), occurrence, exc,
            )
            return None
        if not assignments:
            stats["no_assignments"] += 1
            return None

        valid: List[DynamicInputAssignment] = []
        for assignment in assignments:
            key = (
                str(assignment.kind or "").lower(),
                int(assignment.read_pc) & 0xFFFFFFFF,
                int(assignment.address) & 0xFFFFFFFF,
                max(1, int(assignment.occurrence or 1)),
            )
            if key not in graph.input_nodes_by_key:
                stats["rejected_unknown_sites"] += 1
                continue
            width = max(1, min(32, int(assignment.width or 32)))
            mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
            valid.append(replace(
                assignment,
                kind=key[0],
                read_pc=key[1],
                address=key[2],
                occurrence=key[3],
                width=width,
                value=int(assignment.value) & mask,
            ))
        if not valid:
            stats["no_valid_assignments"] += 1
            return None

        stats["success"] += 1
        stats["assignments"] += len(valid)
        stats["changed_assignments"] += sum(
            1 for item in valid
            if int(item.value) != int(item.observed_value)
        )
        model = DynamicConstraintModel(
            assignments=tuple(valid),
            strategy="llm_slice_solver",
            relation_kind=request.relation_kind,
            branch_pc=request.branch_pc,
            branch_occurrence=request.branch_occurrence,
            path_predicates=int(failure_result.path_predicates),
            slice_nodes=int(failure_result.slice_nodes),
            solver_status="hypothesis",
            causal_complete=False,
            omitted_path_predicates=int(failure_result.omitted_path_predicates),
            branch_record_match=str(failure_result.branch_record_match),
        )
        return DynamicRecoveryResult(
            models=(model,),
            reason="llm_slice_models",
            solver_backend="llm",
            target_inputs=int(failure_result.target_inputs) or len(input_sites),
            path_predicates=int(failure_result.path_predicates),
            slice_nodes=int(failure_result.slice_nodes),
            relation_kind=request.relation_kind,
            solver_status="hypothesis",
            fallback_eligible=False,
            omitted_path_predicates=int(failure_result.omitted_path_predicates),
            branch_record_match=str(failure_result.branch_record_match),
            solver_attempts=int(failure_result.solver_attempts),
            escalated=bool(failure_result.escalated),
            initial_solver_status=str(failure_result.initial_solver_status),
            initial_limits=failure_result.initial_limits,
            final_limits=failure_result.final_limits,
        )

    def _differential_probe_recover(
        self,
        *,
        branch_pc: int,
        occurrence: int,
        target_taken: bool,
    ) -> Optional[DynamicRecoveryResult]:
        """二级定位：表达式图无该分支记录时交给上层差分扰动（设计 §4.2）。

        返回的 models 与 z3/LLM 结果同通道（hypothesis，重放验证把关）；
        上层无实现、无候选或预算耗尽时返回 None，维持死路返回不变。
        """
        fallback = self.differential_probe_fallback
        if fallback is None:
            return None
        stats = self.differential_probe_stats
        stats["calls"] += 1
        request = DifferentialProbeRequest(
            branch_pc=int(branch_pc) & 0xFFFFFFFF,
            branch_occurrence=max(1, int(occurrence)),
            target_taken=bool(target_taken),
            max_candidates=self._env_int("LSGEMU_DIFF_PROBE_MAX_CANDIDATES", 6),
            # 预算自洽（复审 round2 步骤 1）：6 候选 × 4 次/候选 = 24 总预算。
            # 调大总预算或候选数时应同步调大 per-candidate 上限，否则贪心
            # 分配下首个候选吃光总预算、其余候选零探测（旋钮失效）。
            max_probes_per_candidate=self._env_int(
                "LSGEMU_DIFF_PROBE_MAX_PROBES_PER_CANDIDATE", 4
            ),
            max_total_probes=self._env_int("LSGEMU_DIFF_PROBE_TOTAL_PROBES", 24),
            timeout_ms=self._env_int("LSGEMU_DIFF_PROBE_TIMEOUT_MS", 10000),
        )
        stats["attempted"] += 1
        try:
            assignments = fallback(request)
        except Exception as exc:
            stats["errors"] += 1
            logger.warning(
                "差分扰动定位失败 @%s occ=%s: %s",
                hex(int(branch_pc)), occurrence, exc,
            )
            return None
        if not assignments:
            stats["no_assignments"] += 1
            return None

        valid: List[DynamicInputAssignment] = []
        for assignment in assignments:
            width = max(1, min(32, int(assignment.width or 32)))
            mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
            valid.append(replace(
                assignment,
                kind=str(assignment.kind or "mmio").lower(),
                read_pc=int(assignment.read_pc) & 0xFFFFFFFF,
                address=int(assignment.address) & 0xFFFFFFFF,
                # 差分扰动的见证带 occurrence=None（pc 级/全出现语义，
                # 复审 round2 步骤 4 + 第 3 轮收口）：接缝处保留 None 传给
                # 下游，historical_runner 侧据此走 read_occurrence=None →
                # add_pc_constraint（pc 级、覆盖该读点全部出现），
                # 「探测强制范围 = 见证验证范围」达成。z3/LLM 来源的
                # occurrence 恒为 int，不受本行影响。
                occurrence=(
                    int(assignment.occurrence)
                    if assignment.occurrence is not None
                    else None
                ),
                width=width,
                value=int(assignment.value) & mask,
            ))
        if not valid:
            stats["no_valid_assignments"] += 1
            return None
        stats["success"] += 1
        stats["assignments"] += len(valid)
        model = DynamicConstraintModel(
            assignments=tuple(valid),
            strategy="differential_probe",
            relation_kind="differential_probe",
            branch_pc=int(branch_pc) & 0xFFFFFFFF,
            branch_occurrence=max(1, int(occurrence)),
            path_predicates=0,
            slice_nodes=0,
            solver_status="hypothesis",
            causal_complete=False,
            omitted_path_predicates=0,
            branch_record_match="none",
        )
        return DynamicRecoveryResult(
            models=(model,),
            reason="differential_probe_models",
            solver_backend="differential_probe",
            target_inputs=len(valid),
            path_predicates=0,
            slice_nodes=0,
            relation_kind="differential_probe",
            solver_status="hypothesis",
            fallback_eligible=False,
            branch_record_match="none",
        )

    def _archive_dynamic_graph(self, graph: Optional[DynamicExpressionGraph]) -> None:
        if graph is None or not graph.branch_order:
            return
        if any(existing is graph for existing in self.dynamic_graph_archives):
            return
        self.dynamic_graph_archives.append(graph)
        if len(self.dynamic_graph_archives) > self.dynamic_graph_archive_limit:
            del self.dynamic_graph_archives[:-self.dynamic_graph_archive_limit]
        while (
            len(self.dynamic_graph_archives) > 1
            and sum(len(item.nodes) for item in self.dynamic_graph_archives)
            > self.dynamic_graph_archive_node_limit
        ):
            self.dynamic_graph_archives.pop(0)

    def _dynamic_graph_candidates(self) -> List[DynamicExpressionGraph]:
        graphs = list(self.dynamic_graph_archives)
        if self.dynamic_graph is not None:
            graphs.append(self.dynamic_graph)
        return graphs

    def _build_compare_lookup(self) -> Dict[int, Dict]:
        """构建分支到比较指令的索引"""
        lookup: Dict[int, Dict] = {}
        for instructions in self.static_bbs.values():
            if not instructions:
                continue

            branch_insn = instructions[-1]
            for previous in reversed(instructions[max(0, len(instructions) - 6):-1]):
                if previous['mnemonic'].upper().split('.')[0] in ['CMP', 'CMN', 'TST', 'TEQ']:
                    lookup[branch_insn['address']] = previous
                    break

        return lookup

    def _build_compare_pc_to_branch_lookup(self) -> Dict[int, List[int]]:
        lookup: Dict[int, List[int]] = {}
        for branch_pc, compare_insn in self.compare_lookup.items():
            compare_pc = int(compare_insn.get('address', 0) or 0)
            if compare_pc == 0:
                continue
            lookup.setdefault(compare_pc, []).append(int(branch_pc))
        return lookup

    def _build_producer_pc_to_branch_lookup(self) -> Dict[int, List[int]]:
        """产生者 pc → branch_pc 反转（B1；来源 r19 producer_index）。

        与 compare_pc_to_branch_pcs（仅 CMP/CMN/TST/TEQ 同 BB 5 条窗）互补，
        覆盖 SUBS/SBCS/LSLS 等产生者。运行时提交语义见
        _record_flag_producer_snapshot（最近产生者胜 + 分支执行时延迟提交）。
        """
        lookup: Dict[int, List[int]] = {}
        for branch_pc, record in (self.producer_index or {}).items():
            try:
                producers = (record or ([], "", 0))[0] or ()
            except Exception:
                continue
            for producer in producers:
                try:
                    producer_pc = int(producer[0]) & 0xFFFFFFFF
                except Exception:
                    continue
                if producer_pc == 0:
                    continue
                lookup.setdefault(producer_pc, []).append(int(branch_pc) & 0xFFFFFFFF)
        for branch_pcs in lookup.values():
            branch_pcs.sort()
        return lookup

    @staticmethod
    def _select_dependency_site(
        source_type: str,
        mmio_addr: Optional[int],
        memory_addr: Optional[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        normalized_type = str(source_type or "").lower()
        if normalized_type == "memory" and memory_addr is not None:
            return "memory", int(memory_addr)
        if normalized_type == "mmio" and mmio_addr is not None:
            return "mmio", int(mmio_addr)
        if memory_addr is not None and mmio_addr is None:
            return "memory", int(memory_addr)
        if mmio_addr is not None:
            return "mmio", int(mmio_addr)
        if memory_addr is not None:
            return "memory", int(memory_addr)
        return None, None

    @staticmethod
    def _normalize_dependency_site(site) -> Optional[Tuple[str, int, Optional[int], Optional[int]]]:
        if not isinstance(site, (tuple, list)) or len(site) < 2:
            return None
        dep_type = str(site[0] or "").lower()
        if dep_type not in {"mmio", "memory"}:
            return None
        try:
            address = int(site[1]) & 0xFFFFFFFF
        except Exception:
            return None

        def optional_int(index: int) -> Optional[int]:
            if len(site) <= index or site[index] is None:
                return None
            try:
                return int(site[index]) & 0xFFFFFFFF
            except Exception:
                return None

        return dep_type, address, optional_int(2), optional_int(3)

    def _dependency_sites_for_source(self, source: RegisterSource) -> Tuple[Tuple[str, int, Optional[int], Optional[int]], ...]:
        sites: List[Tuple[str, int, Optional[int], Optional[int]]] = []
        for site in source.dependency_sites or ():
            normalized = self._normalize_dependency_site(site)
            if normalized is not None:
                sites.append(normalized)

        dep_type, address = self._select_dependency_site(
            source.source_type,
            source.mmio_address,
            source.memory_address,
        )
        if dep_type is not None and address is not None:
            fallback_site = (
                dep_type,
                int(address) & 0xFFFFFFFF,
                int(source.source_pc) & 0xFFFFFFFF if source.source_pc is not None else None,
                int(source.origin_source_pc) & 0xFFFFFFFF if source.origin_source_pc is not None else None,
            )
            fallback_identity = (fallback_site[0], fallback_site[1])
            if not any((site[0], site[1]) == fallback_identity for site in sites):
                sites.append(fallback_site)

        return self._dedupe_dependency_sites(sites)

    @staticmethod
    def _dedupe_dependency_sites(
        sites: Iterable[Tuple[str, int, Optional[int], Optional[int]]]
    ) -> Tuple[Tuple[str, int, Optional[int], Optional[int]], ...]:
        result: List[Tuple[str, int, Optional[int], Optional[int]]] = []
        seen = set()
        for site in sites or ():
            normalized = RegisterTracer._normalize_dependency_site(site)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return tuple(result)

    @staticmethod
    def _primary_site(
        sites: Iterable[Tuple[str, int, Optional[int], Optional[int]]]
    ) -> Optional[Tuple[str, int, Optional[int], Optional[int]]]:
        normalized = RegisterTracer._dedupe_dependency_sites(sites)
        if not normalized:
            return None
        for site in normalized:
            if site[0] == "mmio":
                return site
        return normalized[0]

    @staticmethod
    def _dependency_identity_count(
        sites: Iterable[Tuple[str, int, Optional[int], Optional[int]]]
    ) -> int:
        return len({
            (dep_type, int(address) & 0xFFFFFFFF)
            for dep_type, address, _source_pc, _origin_source_pc in RegisterTracer._dedupe_dependency_sites(sites)
        })

    @staticmethod
    def _source_to_snapshot_dict(source: RegisterSource) -> Dict[str, object]:
        return {
            'source_type': source.source_type,
            'mmio_address': int(source.mmio_address) if source.mmio_address is not None else None,
            'mmio_value': int(source.mmio_value or 0) & 0xFFFFFFFF,
            'source_pc': int(source.source_pc or 0),
            'origin_source_pc': int(source.origin_source_pc or source.source_pc or 0),
            'memory_address': int(source.memory_address) if source.memory_address is not None else None,
            'memory_role': source.memory_role,
            'memory_base': int(source.memory_base) if source.memory_base is not None else None,
            'memory_offset': int(source.memory_offset) if source.memory_offset is not None else None,
            'memory_access_pc': int(source.memory_access_pc) if source.memory_access_pc is not None else None,
            'memory_size': int(source.memory_size) if source.memory_size is not None else None,
            'operation': source.operation,
            'expression': source.expression,
            'chain': list(source.chain),
            'source_occurrence': (
                int(source.source_occurrence)
                if source.source_occurrence is not None
                else None
            ),
            'dependency_sites': [
                {
                    'type': dep_type,
                    'address': int(address),
                    'source_pc': int(source_pc) if source_pc is not None else None,
                    'origin_source_pc': int(origin_source_pc) if origin_source_pc is not None else None,
                }
                for dep_type, address, source_pc, origin_source_pc in source.dependency_sites
            ],
            'opaque_dependency': bool(source.opaque_dependency),
            'causal_complete': bool(source.causal_complete),
        }

    def _dependency_sites_from_snapshot_source(
        self,
        source: Dict[str, object],
    ) -> Tuple[Tuple[str, int, Optional[int], Optional[int]], ...]:
        sites: List[Tuple[str, int, Optional[int], Optional[int]]] = []
        for item in source.get("dependency_sites", []) or []:
            if isinstance(item, dict):
                normalized = self._normalize_dependency_site((
                    item.get("type"),
                    item.get("address"),
                    item.get("source_pc"),
                    item.get("origin_source_pc"),
                ))
            else:
                normalized = self._normalize_dependency_site(item)
            if normalized is not None:
                sites.append(normalized)

        dep_type, address = self._select_dependency_site(
            str(source.get("source_type", "") or ""),
            source.get("mmio_address"),
            source.get("memory_address"),
        )
        if dep_type is not None and address is not None:
            source_pc = source.get("source_pc")
            origin_source_pc = source.get("origin_source_pc", source_pc)
            fallback_site = (
                dep_type,
                int(address) & 0xFFFFFFFF,
                int(source_pc) & 0xFFFFFFFF if source_pc is not None else None,
                int(origin_source_pc) & 0xFFFFFFFF if origin_source_pc is not None else None,
            )
            fallback_identity = (fallback_site[0], fallback_site[1])
            if not any((site[0], site[1]) == fallback_identity for site in sites):
                sites.append(fallback_site)
        return self._dedupe_dependency_sites(sites)

    def _dependency_records_from_sites(
        self,
        register: str,
        sites: Iterable[Tuple[str, int, Optional[int], Optional[int]]],
        *,
        source_type: str,
        mmio_addr: Optional[int],
        memory_addr: Optional[int],
        call_depth: int,
        expression: Optional[str],
        memory_context: Optional[Dict[str, object]] = None,
        opaque_dependency: bool = False,
        causal_complete: bool = True,
    ) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        normalized_sites = self._dedupe_dependency_sites(sites)
        composite = self._dependency_identity_count(normalized_sites) > 1
        dependency_group = ""
        if composite:
            dependency_group = "|".join([
                str(register or "").lower(),
                str(expression or ""),
                ",".join(
                    f"{dep_type}:0x{int(address) & 0xFFFFFFFF:08x}:"
                    f"0x{int(source_pc or 0) & 0xFFFFFFFF:08x}"
                    for dep_type, address, source_pc, _origin_source_pc
                    in normalized_sites
                ),
            ])
        memory_context = dict(memory_context or {})
        for dep_type, address, source_pc, origin_source_pc in normalized_sites:
            record = {
                'register': register,
                'type': dep_type,
                'address': int(address),
                'source_pc': int(source_pc or 0),
                'origin_source_pc': int(origin_source_pc or source_pc or 0),
                'memory_address': int(address) if dep_type == "memory" else (int(memory_addr) if memory_addr is not None else None),
                'mmio_address': int(address) if dep_type == "mmio" else (int(mmio_addr) if mmio_addr is not None else None),
                'call_depth': int(call_depth or 0),
                'expression': expression,
                'source_type': source_type,
                'composite': composite,
                'dependency_group': dependency_group,
                'opaque_dependency': bool(opaque_dependency),
                'causal_complete': bool(causal_complete),
            }
            if dep_type == "memory":
                for key in (
                    "memory_role",
                    "memory_base",
                    "memory_offset",
                    "memory_access_pc",
                    "memory_size",
                ):
                    if key in memory_context and memory_context.get(key) is not None:
                        record[key] = memory_context.get(key)
            records.append(record)
        return records

    @staticmethod
    def _memory_context_from_source(source: RegisterSource) -> Dict[str, object]:
        return {
            "memory_role": source.memory_role,
            "memory_base": source.memory_base,
            "memory_offset": source.memory_offset,
            "memory_access_pc": source.memory_access_pc,
            "memory_size": source.memory_size,
        }

    @staticmethod
    def _memory_context_from_snapshot_source(source: Dict[str, object]) -> Dict[str, object]:
        return {
            "memory_role": source.get("memory_role"),
            "memory_base": source.get("memory_base"),
            "memory_offset": source.get("memory_offset"),
            "memory_access_pc": source.get("memory_access_pc"),
            "memory_size": source.get("memory_size"),
        }

    @staticmethod
    def _classify_memory_context(
        address: int,
        size: int,
        *,
        access_pc: Optional[int] = None,
        stack_pointer: Optional[int] = None,
    ) -> Dict[str, object]:
        address = int(address) & 0xFFFFFFFF
        size = max(1, int(size or 1))
        role = "unknown"
        base = address & ~0xFFF
        offset = address - base
        if stack_pointer is not None:
            sp = int(stack_pointer) & 0xFFFFFFFF
            delta = int(address) - int(sp)
            if -0x2000 <= delta <= 0x2000:
                role = "stack_frame"
                base = sp
                offset = delta
        if role == "unknown" and 0x20000000 <= address < 0x40000000:
            role = "sram_global_or_struct"
        if role == "unknown" and 0x10000000 <= address < 0x10010000:
            role = "ccm_data"
        return {
            "memory_role": role,
            "memory_base": int(base) & 0xFFFFFFFF,
            "memory_offset": int(offset),
            "memory_access_pc": int(access_pc) & 0xFFFFFFFF if access_pc is not None else None,
            "memory_size": size,
        }

    def _composite_source(
        self,
        dest_reg: str,
        sources: Iterable[RegisterSource],
        instruction_text: str,
        *,
        operation: Optional[str] = None,
        source_pc: Optional[int] = None,
    ) -> Optional[RegisterSource]:
        dependency_sites: List[Tuple[str, int, Optional[int], Optional[int]]] = []
        source_list = [source for source in (sources or []) if isinstance(source, RegisterSource)]
        for source in source_list:
            dependency_sites.extend(self._dependency_sites_for_source(source))
        sites = self._dedupe_dependency_sites(dependency_sites)
        if not sites:
            return None
        primary = self._primary_site(sites)
        mmio_addr = primary[1] if primary and primary[0] == "mmio" else None
        memory_addr = primary[1] if primary and primary[0] == "memory" else None
        primary_memory_context: Dict[str, object] = {}
        if memory_addr is not None:
            for source in source_list:
                if source.memory_address == memory_addr:
                    primary_memory_context = self._memory_context_from_source(source)
                    break
        composite = self._dependency_identity_count(sites) > 1
        origin_pcs = [site[3] for site in sites if site[3] is not None]
        source_pcs = [site[2] for site in sites if site[2] is not None]
        chain_parts: List[str] = []
        for source in source_list:
            chain_parts.extend(str(item) for item in (source.chain or ()))
        chain_parts.append(operation or instruction_text)
        return RegisterSource(
            register=dest_reg,
            source_type="composite" if composite else str(primary[0]),
            mmio_address=mmio_addr,
            mmio_value=None,
            source_pc=source_pc if source_pc is not None else (max(source_pcs) if source_pcs else None),
            origin_source_pc=min(origin_pcs) if origin_pcs else (min(source_pcs) if source_pcs else None),
            instruction=instruction_text,
            memory_address=memory_addr,
            memory_role=primary_memory_context.get("memory_role"),
            memory_base=primary_memory_context.get("memory_base"),
            memory_offset=primary_memory_context.get("memory_offset"),
            memory_access_pc=primary_memory_context.get("memory_access_pc"),
            memory_size=primary_memory_context.get("memory_size"),
            operation=operation or "composite",
            expression=instruction_text,
            chain=tuple(chain_parts[-16:]),
            call_depth=len(self.call_stack),
            source_occurrence=(
                source_list[0].source_occurrence
                if source_list
                and all(
                    item.source_occurrence == source_list[0].source_occurrence
                    for item in source_list
                )
                else None
            ),
            dependency_sites=sites,
            opaque_dependency=any(
                bool(source.opaque_dependency) for source in source_list
            ),
            causal_complete=all(
                bool(source.causal_complete) for source in source_list
            ),
        )

    def start_tracing(self):
        """开始追踪"""
        if self.mmio_hooks or self.ram_hook is not None or self.code_hook is not None:
            self.stop_tracing()
        self._preloaded_external_loads.clear()
        if self.dynamic_graph_enabled:
            if self.dynamic_graph.has_execution_data:
                self._archive_dynamic_graph(self.dynamic_graph)
                self.dynamic_graph = self._new_dynamic_graph()
        event_backed_reads = self._attach_external_input_observer()
        # Hook MMIO writes in every mode.  Reads are observed through the
        # emulator event source when available, because mapped-MMIO preload
        # bypasses Unicorn's normal memory-read callback entirely.
        self.mmio_hooks = []
        for start, end in self._mmio_ranges():
            self.mmio_hooks.append(managed_hook_add(self.uc,
                UC_HOOK_MEM_WRITE if event_backed_reads else UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE,
                self._mmio_access_hook,
                None,
                start,
                end,
            ))
        self.mmio_hook = self.mmio_hooks[0] if self.mmio_hooks else None

        # Hook SRAM reads/writes so provenance can survive STR/LDR, PUSH/POP,
        # and ordinary cross-BB stack/global-memory traffic.
        # B4：补 CCM RAM（0x10000000-0x10010000，Pixhawk1/F427 的 64KB 核心
        # 耦合内存）——此前该区 load/store 全部丢 provenance（r20a §2.2-2）。
        self.ram_hooks = []
        for start, end in self._ram_ranges():
            self.ram_hooks.append(managed_hook_add(self.uc,
                UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE,
                self._ram_access_hook,
                None,
                start,
                end,
            ))
        self.ram_hook = self.ram_hooks[0] if self.ram_hooks else None

        # Hook 代码执行
        self.code_hook = managed_hook_add(self.uc,
            UC_HOOK_CODE,
            self._code_hook
        )

        logger.info("[RegisterTracer] 开始追踪")

    def stop_tracing(self):
        """停止追踪"""
        self._detach_external_input_observer()
        for hook in list(self.mmio_hooks or []):
            try:
                managed_hook_del(self.uc, hook)
            except Exception:
                pass
        self.mmio_hooks = []
        self.mmio_hook = None
        for hook in list(getattr(self, "ram_hooks", []) or []):
            try:
                managed_hook_del(self.uc, hook)
            except Exception:
                pass
        self.ram_hooks = []
        if self.ram_hook is not None:
            try:
                managed_hook_del(self.uc, self.ram_hook)
            except Exception:
                pass
            self.ram_hook = None
        if self.code_hook is not None:
            try:
                managed_hook_del(self.uc, self.code_hook)
            except Exception:
                pass
            self.code_hook = None
        logger.info("[RegisterTracer] 停止追踪")

    def _attach_external_input_observer(self) -> bool:
        source = self.external_input_event_source
        attach = getattr(source, "add_external_input_observer", None)
        if not callable(attach):
            return False
        try:
            attach(self._external_input_event_callback)
            self._external_input_observer_attached = True
        except Exception:
            self._external_input_observer_attached = False
        return self._external_input_observer_attached

    def _detach_external_input_observer(self) -> None:
        if not self._external_input_observer_attached:
            return
        source = self.external_input_event_source
        detach = getattr(source, "remove_external_input_observer", None)
        if callable(detach):
            try:
                detach(self._external_input_event_callback)
            except Exception:
                pass
        self._external_input_observer_attached = False

    def _mmio_access_hook(self, uc, access, address, size, value, user_data):
        """MMIO 访问 Hook"""
        try:
            pc = uc.reg_read(UC_ARM_REG_PC)
            is_read = (access == UC_MEM_READ)

            # 记录 MMIO 访问
            self._append_mmio_history((pc, address, value, is_read))

            # 如果是读取，尝试识别目标寄存器
            if is_read:
                site = (int(pc), int(address) & 0xFFFFFFFF)
                occurrence = int(self.mmio_read_occurrence_counts.get(site, 0) or 0) + 1
                self.mmio_read_occurrence_counts[site] = occurrence
                observed_value = self._read_memory_value(
                    uc,
                    int(address),
                    int(size),
                    fallback=int(value or 0),
                )
                self._identify_target_register(
                    pc,
                    address,
                    observed_value,
                    occurrence,
                    size=int(size),
                )

        except Exception as e:
            logger.debug(f"[RegisterTracer] MMIO Hook 错误: {e}")

    def _external_input_event_hook(self, event: Dict[str, object]) -> None:
        """Record a concrete environment read served by IntelligentEmulator.

        The event path is authoritative whenever it is attached: it covers
        ordinary mapped reads, unmapped reads, and the mapped-MMIO preload
        fast path that advances PC before Unicorn emits UC_HOOK_MEM_READ.
        """
        if not isinstance(event, dict):
            self.external_input_event_stats["unsupported"] += 1
            return
        kind = str(event.get("kind") or "").strip().lower()
        if kind not in {"mmio", "external_memory"}:
            self.external_input_event_stats["unsupported"] += 1
            return
        try:
            pc = int(event.get("pc", 0) or 0) & 0xFFFFFFFF
            address = int(event.get("address", 0) or 0) & 0xFFFFFFFF
            size = max(1, min(4, int(event.get("size", 1) or 1)))
            occurrence = max(1, int(event.get("occurrence", 1) or 1))
            value = int(event.get("value", 0) or 0) & ((1 << (size * 8)) - 1)
            trace_event_id = max(0, int(event.get("trace_event_id", 0) or 0))
        except (TypeError, ValueError):
            self.external_input_event_stats["unsupported"] += 1
            return

        self.external_input_event_stats["received"] += 1
        self.external_input_event_stats[kind] += 1
        if kind == "mmio":
            self._append_mmio_history((pc, address, value, True))
            site = (pc, address)
            self.mmio_read_occurrence_counts[site] = max(
                int(self.mmio_read_occurrence_counts.get(site, 0) or 0),
                occurrence,
            )
        else:
            site = (pc, address)
            self.external_memory_read_occurrence_counts[site] = max(
                int(self.external_memory_read_occurrence_counts.get(site, 0) or 0),
                occurrence,
            )

        recorded = self._record_event_backed_input_read(
            kind=kind,
            pc=pc,
            address=address,
            size=size,
            value=value,
            occurrence=occurrence,
            preloaded=str(event.get("delivery") or "") == "mapped_mmio_preload",
            trace_event_id=trace_event_id,
        )
        if not recorded:
            self.external_input_event_stats["target_register_missing"] += 1

    def _record_event_backed_input_read(
        self,
        *,
        kind: str,
        pc: int,
        address: int,
        size: int,
        value: int,
        occurrence: int,
        preloaded: bool,
        trace_event_id: int = 0,
    ) -> bool:
        instruction = self._find_instruction_at_pc(int(pc))
        if not instruction:
            return False
        target_reg = self._identify_load_target_register_for_access(
            self.uc,
            instruction,
            int(address),
            int(size),
        )
        if target_reg is None:
            return False
        address_registers = self._memory_address_registers(instruction)
        address_sources = [
            self.register_sources[register]
            for register in address_registers
            if register in self.register_sources
            and self._dependency_sites_for_source(self.register_sources[register])
        ]
        normalized_mnemonic = str(instruction.get("mnemonic", "")).upper().split(".")[0]
        instruction_text = (
            f"{str(instruction.get('mnemonic', '')).upper()} "
            f"{instruction.get('operands', '')}"
        ).strip()
        normalized_kind = "mmio" if str(kind).lower() == "mmio" else "external_memory"
        if normalized_kind == "mmio":
            source = RegisterSource(
                register=target_reg,
                source_type="mmio",
                mmio_address=int(address),
                mmio_value=int(value),
                source_pc=int(pc),
                origin_source_pc=int(pc),
                instruction=instruction_text,
                expression=f"MMIO[{hex(int(address))}]",
                chain=(f"MMIO[{hex(int(address))}]@{hex(int(pc))}",),
                call_depth=len(self.call_stack),
                source_occurrence=max(1, int(occurrence)),
                dependency_sites=(("mmio", int(address), int(pc), int(pc)),),
            )
        else:
            source = RegisterSource(
                register=target_reg,
                source_type="memory",
                source_pc=int(pc),
                origin_source_pc=int(pc),
                instruction=instruction_text,
                memory_address=int(address),
                memory_access_pc=int(pc),
                memory_size=max(1, int(size)),
                expression=f"INPUT_MEM[{hex(int(address))}]",
                chain=(f"INPUT_MEM[{hex(int(address))}]@{hex(int(pc))}",),
                call_depth=len(self.call_stack),
                source_occurrence=max(1, int(occurrence)),
                dependency_sites=(("memory", int(address), int(pc), int(pc)),),
            )
        source = self._combine_indirect_memory_source(
            target_reg,
            source,
            address_sources,
            instruction_text,
            access_pc=int(pc),
            address=int(address),
            size=int(size),
            operation="indirect_load",
        ) or source
        self.register_sources[target_reg] = source
        if source.mmio_address is not None:
            self.reg_mmio_map[target_reg] = int(source.mmio_address)
        else:
            self.reg_mmio_map.pop(target_reg, None)
        self._lineage_note_write(
            target_reg,
            pc,
            str(instruction.get("mnemonic", "") or ""),
            "mmio_read" if normalized_kind == "mmio" else "input_mem_read",
            address_registers,
        )

        if self.dynamic_graph_enabled:
            self.dynamic_graph.observe_external_read(
                kind=normalized_kind,
                read_pc=int(pc),
                address=int(address),
                occurrence=max(1, int(occurrence)),
                size=max(1, int(size)),
                observed_value=int(value),
                destination_register=target_reg,
                signed=normalized_mnemonic in {"LDRSB", "LDRSH"},
                trace_event_id=(trace_event_id if trace_event_id > 0 else None),
            )
        if preloaded:
            self._preloaded_external_loads.add((int(pc), str(target_reg).lower()))
        return True

    def _ram_access_hook(self, uc, access, address, size, value, user_data):
        """SRAM 访问 Hook，用 shadow memory 延续 provenance。"""
        try:
            pc = uc.reg_read(UC_ARM_REG_PC)
            instruction = self._find_instruction_at_pc(pc)
            if not instruction:
                return

            if access == UC_MEM_READ:
                target_reg = self._identify_load_target_register_for_access(
                    uc,
                    instruction,
                    int(address),
                    int(size),
                )
                if target_reg is None:
                    return
                address_registers = self._memory_address_registers(instruction)
                address_sources = [
                    self.register_sources[register]
                    for register in address_registers
                    if register in self.register_sources
                    and self._dependency_sites_for_source(
                        self.register_sources[register]
                    )
                ]
                observed_value = self._read_memory_value(
                    uc,
                    int(address),
                    int(size),
                    fallback=int(value or 0),
                )
                external_input = self._is_external_memory_input(int(address), int(size))
                occurrence = 1
                if external_input:
                    site = (int(pc), int(address) & 0xFFFFFFFF)
                    if self._external_input_observer_attached:
                        occurrence = max(
                            1,
                            int(
                                self.external_memory_read_occurrence_counts.get(site, 0)
                                or 0
                            ),
                        )
                    else:
                        occurrence = int(
                            self.external_memory_read_occurrence_counts.get(site, 0) or 0
                        ) + 1
                        self.external_memory_read_occurrence_counts[site] = occurrence
                event_backed_external = bool(
                    external_input and self._external_input_observer_attached
                )
                if event_backed_external and target_reg in self.register_sources:
                    return
                if self.dynamic_graph_enabled and not event_backed_external:
                    mnemonic = str(instruction.get("mnemonic", "")).upper().split(".")[0]
                    self.dynamic_graph.observe_memory_load(
                        read_pc=int(pc),
                        address=int(address),
                        size=int(size),
                        observed_value=observed_value,
                        destination_register=target_reg,
                        external_input=external_input,
                        occurrence=occurrence,
                        signed=mnemonic in {"LDRSB", "LDRSH"},
                        address_registers=address_registers,
                    )
                source = self._get_memory_source(int(address), int(size))
                source = self._verify_shadow_source_value(
                    source,
                    int(address),
                    int(size),
                    int(observed_value),
                )
                source = self._combine_indirect_memory_source(
                    target_reg,
                    source,
                    address_sources,
                    f"{instruction['mnemonic'].upper()} {instruction.get('operands', '')}".strip(),
                    access_pc=int(pc),
                    address=int(address),
                    size=int(size),
                    operation="indirect_load",
                )
                if source is None:
                    if event_backed_external:
                        return
                    self._clear_register_source(target_reg)
                    self._lineage_note_write(
                        target_reg, pc,
                        str(instruction.get('mnemonic', '') or ''),
                        "ram_load", address_registers,
                    )
                    return
                self._set_register_source(
                    target_reg,
                    source,
                    f"{instruction['mnemonic'].upper()} {instruction.get('operands', '')}".strip(),
                    source_type="memory",
                    memory_address=int(address),
                    source_pc=int(pc),
                    operation="load",
                    memory_context=self._classify_memory_context(
                        int(address),
                        int(size),
                        access_pc=int(pc),
                        stack_pointer=self._safe_read_sp(uc),
                    ),
                )
                self._lineage_note_write(
                    target_reg, pc,
                    str(instruction.get('mnemonic', '') or ''),
                    "ram_load", address_registers,
                )
                return

            if access == UC_MEM_WRITE:
                source_reg = self._identify_store_source_register_for_access(
                    uc,
                    instruction,
                    int(address),
                    int(size),
                )
                if source_reg is None:
                    return
                address_registers = self._memory_address_registers(instruction)
                address_sources = [
                    self.register_sources[register]
                    for register in address_registers
                    if register in self.register_sources
                    and self._dependency_sites_for_source(
                        self.register_sources[register]
                    )
                ]
                if self.dynamic_graph_enabled:
                    self.dynamic_graph.observe_memory_store(
                        write_pc=int(pc),
                        address=int(address),
                        size=int(size),
                        source_register=source_reg,
                        concrete_value=int(value or 0),
                        address_registers=address_registers,
                    )
                source = self.register_sources.get(source_reg)
                source = self._combine_indirect_memory_source(
                    source_reg,
                    source,
                    address_sources,
                    f"{instruction['mnemonic'].upper()} {instruction.get('operands', '')}".strip(),
                    access_pc=int(pc),
                    address=int(address),
                    size=int(size),
                    operation="indirect_store",
                )
                if source is None:
                    # B2：无源写不再级联清空影子——落 tombstone（默认）。
                    # 保留旧条目仅当值相等校验通过且开关显式打开。
                    if self._shadow_value_equal_retain and self._retain_shadow_by_value(
                        int(address), int(size), int(value or 0)
                    ):
                        return
                    if self._shadow_tombstone_enabled:
                        self._write_memory_tombstone(
                            int(address),
                            int(size),
                            access_pc=int(pc),
                            store_reg=source_reg,
                            store_value=int(value or 0),
                            stack_pointer=self._safe_read_sp(uc),
                        )
                    else:
                        self._clear_memory_source(int(address), int(size))
                    return
                self._set_memory_source(
                    int(address),
                    int(size),
                    source,
                    access_pc=int(pc),
                    stack_pointer=self._safe_read_sp(uc),
                    store_value=int(value or 0),
                )

        except Exception as e:
            logger.debug(f"[RegisterTracer] RAM Hook 错误: {e}")

    def _code_hook(self, uc, address, size, user_data):
        """代码执行 Hook"""
        try:
            self._retire_call_frames(address)
            instruction = self._find_instruction_at_pc(address)
            if instruction:
                preloaded_external = self._consume_preloaded_external_load(
                    int(address),
                    instruction,
                )
                if self.dynamic_graph_enabled:
                    if preloaded_external:
                        self.dynamic_graph.stats["preloaded_input_loads_preserved"] += 1
                    else:
                        try:
                            self.dynamic_graph.observe_instruction(
                                instruction,
                                concrete_register=self._read_register_value,
                                cpsr=self._read_cpsr(),
                            )
                        except Exception as exc:
                            self.dynamic_graph.stats["instruction_errors"] += 1
                            logger.debug("[RegisterTracer] 动态表达式传播失败 @ 0x%08x: %s", address, exc)
                if not preloaded_external:
                    self._propagate_register_sources(instruction)
                self._record_flag_producer_snapshot(address, instruction)
                self._record_branch_dependency_snapshot(address)
                self._record_call_frame(address, size, instruction)
        except Exception as e:
            logger.debug(f"[RegisterTracer] Code Hook 错误: {e}")

    def _consume_preloaded_external_load(
        self,
        address: int,
        instruction: Dict[str, object],
    ) -> bool:
        mnemonic = str(instruction.get("mnemonic", "")).upper().split(".")[0]
        if mnemonic not in {"LDR", "LDRB", "LDRH", "LDRSB", "LDRSH", "VLDR"}:
            return False
        operands = self._split_operands(str(instruction.get("operands", "")))
        target_reg = self._normalize_register(operands[0]) if operands else None
        if target_reg is None:
            return False
        key = (int(address) & 0xFFFFFFFF, target_reg)
        if key not in self._preloaded_external_loads:
            return False
        self._preloaded_external_loads.discard(key)
        self.external_input_event_stats["preloaded_loads_preserved"] += 1
        return True

    # B3：寄存器名 → unicorn id（含 VFP 域 s0-s31/d0-d15；d 取低 32 位——
    # 值谱系用途够用，双精度值相等校验按低字近似，报告口径注明）。
    # 模块级构建一次，code hook 热路径零构造开销。
    _REGISTER_IDS: Dict[str, Any] = {
        "r0": UC_ARM_REG_R0, "r1": UC_ARM_REG_R1,
        "r2": UC_ARM_REG_R2, "r3": UC_ARM_REG_R3,
        "r4": UC_ARM_REG_R4, "r5": UC_ARM_REG_R5,
        "r6": UC_ARM_REG_R6, "r7": UC_ARM_REG_R7,
        "r8": UC_ARM_REG_R8, "r9": UC_ARM_REG_R9,
        "r10": UC_ARM_REG_R10, "r11": UC_ARM_REG_R11,
        "r12": UC_ARM_REG_R12, "sp": UC_ARM_REG_SP,
        "lr": UC_ARM_REG_LR, "pc": UC_ARM_REG_PC,
        **{
            f"s{index}": getattr(arm_const, f"UC_ARM_REG_S{index}", None)
            for index in range(32)
        },
        **{
            f"d{index}": getattr(arm_const, f"UC_ARM_REG_D{index}", None)
            for index in range(16)
        },
    }

    def _read_register_value(self, register: str, uc=None) -> Optional[int]:
        register_id = self._REGISTER_IDS.get(str(register or "").lower())
        if register_id is None:
            return None
        try:
            source_uc = uc if uc is not None else self.uc
            return int(source_uc.reg_read(register_id)) & 0xFFFFFFFF
        except Exception:
            return None

    def _read_cpsr(self) -> Optional[int]:
        try:
            return int(self.uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFF
        except Exception:
            return None

    @staticmethod
    def _read_memory_value(uc, address: int, size: int, *, fallback: int = 0) -> int:
        effective_size = max(1, min(4, int(size or 1)))
        try:
            return int.from_bytes(
                bytes(uc.mem_read(int(address), effective_size)),
                "little",
            )
        except Exception:
            return int(fallback) & ((1 << (effective_size * 8)) - 1)

    def _is_external_memory_input(self, address: int, size: int) -> bool:
        predicate = self.external_memory_input_predicate
        if predicate is None:
            return False
        try:
            return bool(predicate(int(address) & 0xFFFFFFFF, max(1, int(size or 1))))
        except Exception:
            return False

    def _normalize_register(self, token: str) -> Optional[str]:
        text = str(token or "").strip().lower()
        if re.fullmatch(r"r(?:1[0-2]|\d)", text):
            return text
        if text in {"sp", "lr", "pc"}:
            return text
        # B3：VFP 寄存器域——VMRS 站 85% 不透明的钥匙（r20a §2.2-5）。
        if re.fullmatch(r"s(?:3[01]|[12]?\d)", text):
            return text
        if re.fullmatch(r"d(?:1[0-5]|\d)", text):
            return text
        return None

    def _extract_operand_registers(self, operand_text: str) -> List[str]:
        registers = []
        for token in re.findall(
            r"\b(?:r(?:1[0-2]|\d)|sp|lr|pc|s(?:3[01]|[12]?\d)|d(?:1[0-5]|\d))\b",
            str(operand_text or "").lower(),
        ):
            reg = self._normalize_register(token)
            if reg and reg not in registers:
                registers.append(reg)
        return registers

    @staticmethod
    def _safe_read_sp(uc) -> Optional[int]:
        try:
            return int(uc.reg_read(UC_ARM_REG_SP)) & 0xFFFFFFFF
        except Exception:
            return None

    def _split_operands(self, operands: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        bracket_depth = 0
        brace_depth = 0
        for char in str(operands or ""):
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(0, brace_depth - 1)
            if char == "," and bracket_depth == 0 and brace_depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    def _register_list(self, operands: str) -> List[str]:
        match = re.search(r"\{([^}]+)\}", str(operands or "").lower())
        if not match:
            return []
        regs: List[str] = []
        for item in match.group(1).split(","):
            item = item.strip()
            if "-" in item:
                start, end = [part.strip() for part in item.split("-", 1)]
                start_reg = self._normalize_register(start)
                end_reg = self._normalize_register(end)
                if (
                    start_reg and end_reg
                    and start_reg[0] == end_reg[0]
                    and start_reg[0] in "rsd"
                ):
                    for number in range(int(start_reg[1:]), int(end_reg[1:]) + 1):
                        regs.append(f"{start_reg[0]}{number}")
                continue
            reg = self._normalize_register(item)
            if reg:
                regs.append(reg)
        order = {f"r{i}": i for i in range(13)}
        order.update({"sp": 13, "lr": 14, "pc": 15})
        # B3：VFP 列表（VPUSH/VPOP {s0-s3} 等）
        order.update({f"s{i}": 32 + i for i in range(32)})
        order.update({f"d{i}": 64 + i for i in range(16)})
        return sorted(dict.fromkeys(regs), key=lambda reg: order.get(reg, 99))

    def _clone_source(
        self,
        dest_reg: str,
        source: RegisterSource,
        instruction_text: str,
        *,
        source_type: Optional[str] = None,
        memory_address: Optional[int] = None,
        source_pc: Optional[int] = None,
        operation: Optional[str] = None,
        memory_context: Optional[Dict[str, object]] = None,
    ) -> RegisterSource:
        dependency_sites = self._dependency_sites_for_source(source)
        memory_context = dict(memory_context or {})
        propagated = RegisterSource(
            register=dest_reg,
            source_type=source_type or source.source_type,
            mmio_address=source.mmio_address,
            mmio_value=source.mmio_value,
            source_pc=source_pc if source_pc is not None else source.source_pc,
            origin_source_pc=source.origin_source_pc if source.origin_source_pc is not None else source.source_pc,
            instruction=instruction_text,
            memory_address=memory_address if memory_address is not None else source.memory_address,
            memory_role=memory_context.get("memory_role", source.memory_role),
            memory_base=memory_context.get("memory_base", source.memory_base),
            memory_offset=memory_context.get("memory_offset", source.memory_offset),
            memory_access_pc=memory_context.get("memory_access_pc", source.memory_access_pc),
            memory_size=memory_context.get("memory_size", source.memory_size),
            operation=operation if operation is not None else source.operation,
            expression=instruction_text,
            chain=tuple(source.chain) + ((operation or instruction_text),),
            call_depth=len(self.call_stack),
            source_occurrence=source.source_occurrence,
            dependency_sites=dependency_sites,
            opaque_dependency=bool(source.opaque_dependency),
            causal_complete=bool(source.causal_complete),
        )
        return propagated

    def _set_register_source(
        self,
        dest_reg: str,
        source: RegisterSource,
        instruction_text: str,
        *,
        source_type: Optional[str] = None,
        memory_address: Optional[int] = None,
        source_pc: Optional[int] = None,
        operation: Optional[str] = None,
        memory_context: Optional[Dict[str, object]] = None,
    ):
        propagated = self._clone_source(
            dest_reg,
            source,
            instruction_text,
            source_type=source_type,
            memory_address=memory_address,
            source_pc=source_pc,
            operation=operation,
            memory_context=memory_context,
        )
        self.register_sources[dest_reg] = propagated
        if propagated.mmio_address is not None:
            self.reg_mmio_map[dest_reg] = propagated.mmio_address
        else:
            self.reg_mmio_map.pop(dest_reg, None)

    def _clear_register_source(self, dest_reg: str):
        source = self.register_sources.pop(dest_reg, None)
        if source is not None and (
            source.mmio_address is not None
            or any(site[0] == "mmio" for site in (source.dependency_sites or ()))
        ):
            # B2 计量：mmio 身份的源在哪类清除点死亡（诊断用，不改语义）。
            self.provenance_kill_stats["mmio_source_cleared"] += 1
        self.reg_mmio_map.pop(dest_reg, None)

    # -- r23 计量：寄存器写事件日志 + 比较点死操作数回溯（默认关） ---------

    _LINEAGE_MAX_CHAIN_DEPTH = 16
    # 回溯深度/节点上限可由探针放宽（LSGEMU_LINEAGE_WALK_DEPTH，默认 6）——
    # 深栈溢写链（sinf/cosf 一族）6 层内只见内存往返，放宽用于排除
    # 「mmio 藏在更深处」的假阴性。
    _LINEAGE_WALK_MAX_DEPTH = max(
        1, int(os.environ.get("LSGEMU_LINEAGE_WALK_DEPTH", "6") or 6)
    )
    _LINEAGE_WALK_MAX_NODES = max(
        24, int(os.environ.get("LSGEMU_LINEAGE_WALK_NODES", "64") or 64)
    )

    def _lineage_note_write(
        self,
        reg: Optional[str],
        pc: Optional[int],
        mnemonic: Optional[str],
        kind: str,
        input_regs: Iterable[str] = (),
    ) -> None:
        """记录一次寄存器写事件（每寄存器只留栈顶；引用链 depth 封顶）。

        ``sites`` 取写后该寄存器源上的依赖点（有 mmio/memory 身份 = 谱系活；
        空 = 谱系死或从未有过）。``inputs`` 逐项记录输入寄存器**当时**的栈顶
        事件引用——比较点回溯时沿引用走到的就是写入时刻的真实上游。
        """
        if not self._lineage_walk_enabled or not reg:
            return
        inputs = []
        max_input_depth = 0
        for input_reg in input_regs or ():
            input_reg = str(input_reg or "").lower()
            if not input_reg:
                continue
            event = self._lineage_last_write.get(input_reg)
            if event is not None and int(event.get("depth", 0)) >= (
                self._LINEAGE_MAX_CHAIN_DEPTH
            ):
                event = None
            if event is not None:
                max_input_depth = max(
                    max_input_depth, int(event.get("depth", 0))
                )
            inputs.append((input_reg, event))
        source = self.register_sources.get(reg)
        sites = self._dependency_sites_for_source(source) if source else ()
        self._lineage_last_write[reg] = {
            "pc": int(pc or 0) & 0xFFFFFFFF,
            "mnemonic": str(mnemonic or ""),
            "kind": str(kind or ""),
            "sites": tuple(
                (str(site[0]), int(site[1])) for site in (sites or ())
            ),
            "inputs": tuple(inputs),
            "depth": 1 + max_input_depth,
        }

    def _lineage_reg_dead_at_snapshot(self, reg: str) -> bool:
        source = self.register_sources.get(reg)
        if source is None:
            return True
        return not self._dependency_sites_for_source(source)

    def _lineage_dead_operand_report(
        self,
        operand_texts: Iterable[str],
    ) -> Dict[str, Dict[str, object]]:
        """比较/标志写者视角：无活源操作数寄存器的死亡归因（逐操作数）。

        返回 {reg: {verdict, deaths[], frontier[]}}；verdict:
        - ``no_prior_write``  追踪期内无写事件（回放中途起跑/域外写者）；
        - ``no_external_lineage``  有界回溯内上游全无外部依赖（内部状态真源）；
        - ``death_at``  回溯撞到「输入带外部依赖、结果不带」的边——
          deaths[] 给出死亡指令 pc/助记符/途经寄存器（file:line 由调用方
          符号化；tracer 侧对应清除路径见 deaths[].tracer_site）。
        """
        report: Dict[str, Dict[str, object]] = {}
        if not self._lineage_walk_enabled:
            return report
        for operand in operand_texts or ():
            for reg in self._extract_operand_registers(operand):
                if reg in report or not self._lineage_reg_dead_at_snapshot(reg):
                    continue
                report[reg] = self._lineage_walk(reg)
        return report

    def _lineage_walk(self, reg: str) -> Dict[str, object]:
        top = self._lineage_last_write.get(reg)
        cache_key = None
        if top is not None:
            cache_key = (
                str(reg),
                int(top.get("pc", 0) or 0),
                str(top.get("mnemonic", "") or ""),
            )
            cached = self._lineage_walk_cache.get(cache_key)
            if cached is not None:
                return cached
        if top is None:
            result: Dict[str, object] = {
                "verdict": "no_prior_write",
                "deaths": [],
                "frontier": [],
            }
            self.provenance_dead_operand_stats["no_prior_write"] += 1
            return result
        # BFS：事件节点沿 inputs 引用向写时刻上游走；首次撞到带 sites 的
        # 输入即定位死亡边（该输入的外部谱系在消费点被丢弃）。
        deaths: List[Dict[str, object]] = []
        frontier: List[Dict[str, object]] = []
        visited = 0
        queue = [(reg, top, 0)]
        seen_events = {id(top)}
        found_external = False
        while queue and visited < self._LINEAGE_WALK_MAX_NODES:
            via_reg, event, depth = queue.pop(0)
            visited += 1
            for input_reg, input_event in (event.get("inputs") or ()):
                if input_event is None:
                    frontier.append({
                        "reg": input_reg,
                        "at_pc": int(event.get("pc", 0) or 0),
                        "reason": "no_prior_write",
                    })
                    continue
                if id(input_event) in seen_events:
                    continue
                seen_events.add(id(input_event))
                if input_event.get("sites"):
                    found_external = True
                    if len(deaths) < 3:
                        deaths.append({
                            "at_pc": int(event.get("pc", 0) or 0),
                            "at_mnemonic": str(event.get("mnemonic", "") or ""),
                            "kind": str(event.get("kind", "") or ""),
                            "via_reg": input_reg,
                            "input_sites": [
                                f"{t}:0x{int(a):08x}"
                                for t, a in input_event["sites"]
                            ],
                            "tracer_site": self._lineage_tracer_site(
                                event, input_reg == via_reg
                            ),
                        })
                    continue
                if depth + 1 < self._LINEAGE_WALK_MAX_DEPTH:
                    queue.append((input_reg, input_event, depth + 1))
                else:
                    frontier.append({
                        "reg": input_reg,
                        "at_pc": int(input_event.get("pc", 0) or 0),
                        "reason": "depth_exhausted",
                    })
        if found_external and deaths:
            verdict = "death_at"
        elif found_external:
            verdict = "external_beyond_horizon"
        else:
            verdict = "no_external_lineage"
        result = {
            "verdict": verdict,
            "deaths": deaths,
            "frontier": frontier[:6],
        }
        self.provenance_dead_operand_stats[verdict] += 1
        if cache_key is not None:
            if len(self._lineage_walk_cache) > 16384:
                self._lineage_walk_cache.clear()
            self._lineage_walk_cache[cache_key] = result
        return result

    @staticmethod
    def _lineage_tracer_site(event: Dict[str, object], via_self: bool) -> str:
        """死亡事件 kind → tracer 侧清除路径（python file:line 口径）。"""
        kind = str(event.get("kind", "") or "")
        mnemonic = str(event.get("mnemonic", "") or "").upper().split(".")[0]
        return {
            "propagate": (
                "register_tracer.py:_propagate_register_sources "
                f"({mnemonic} 传播/清源路径)"
            ),
            "ram_load": "register_tracer.py:_ram_access_hook 读路径",
            "mmio_read": "register_tracer.py:_record_event_backed_input_read",
            "input_mem_read": "register_tracer.py:_record_event_backed_input_read",
            "call_retire": "register_tracer.py:_retire_call_frames r0-r3",
            "callee_saved_restore": (
                "register_tracer.py:_restore_callee_saved_sources"
            ),
            "pop": "register_tracer.py:_propagate_register_sources POP/VPOP",
            "mrc": "register_tracer.py:_propagate_register_sources MRC",
            "movt": "register_tracer.py:_propagate_register_sources MOVT",
        }.get(
            kind,
            f"register_tracer.py:_propagate_register_sources ({kind or mnemonic})",
        )

    def _set_memory_source(
        self,
        address: int,
        size: int,
        source: RegisterSource,
        *,
        access_pc: Optional[int] = None,
        stack_pointer: Optional[int] = None,
        store_value: Optional[int] = None,
    ):
        context = self._classify_memory_context(
            int(address),
            int(size),
            access_pc=access_pc,
            stack_pointer=stack_pointer,
        )
        for offset in range(max(1, int(size))):
            clone = self._clone_source(
                source.register,
                source,
                source.instruction or "memory-store",
                source_type="memory",
                memory_address=int(address),
                operation="store",
                memory_context=context,
            )
            if store_value is not None:
                clone.stored_value = (int(store_value) >> (8 * offset)) & 0xFF
            self.memory_sources[int(address) + offset] = clone

    def _write_memory_tombstone(
        self,
        address: int,
        size: int,
        *,
        access_pc: int,
        store_reg: str,
        store_value: int,
        stack_pointer: Optional[int] = None,
    ):
        """B2：未知来源写的影子标记——「该地址被写过但来源未知」。

        只表达未知（opaque_dependency=True、无 mmio 身份、dependency_sites
        仅含 memory 地址本身），后续 load 命中会得到 opaque 源而不是级联清空；
        转发前的 _verify_shadow_source_value 值校验兜底模拟器侧 API 直写 RAM
        （绕过 UC_HOOK_MEM_WRITE）造成的影子陈旧。
        """
        context = self._classify_memory_context(
            int(address),
            int(size),
            access_pc=int(access_pc),
            stack_pointer=stack_pointer,
        )
        for offset in range(max(1, int(size))):
            self.memory_sources[int(address) + offset] = RegisterSource(
                register=str(store_reg or "memory"),
                source_type="memory",
                source_pc=int(access_pc),
                origin_source_pc=int(access_pc),
                instruction="memory-tombstone",
                memory_address=int(address) & 0xFFFFFFFF,
                memory_role=context.get("memory_role"),
                memory_base=context.get("memory_base"),
                memory_offset=context.get("memory_offset"),
                memory_access_pc=int(access_pc) & 0xFFFFFFFF,
                memory_size=max(1, int(size)),
                operation="store",
                expression=f"RAM-tombstone[{hex(int(address) & 0xFFFFFFFF)}]",
                chain=(f"tombstone@{hex(int(access_pc) & 0xFFFFFFFF)}",),
                call_depth=len(self.call_stack),
                dependency_sites=(),
                opaque_dependency=True,
                causal_complete=False,
                stored_value=(int(store_value) >> (8 * offset)) & 0xFF,
            )
        self.provenance_kill_stats["memory_tombstone_written"] += 1

    def _retain_shadow_by_value(self, address: int, size: int, store_value: int) -> bool:
        """值相等保留（默认关）：旧条目逐字节等于本次写入值时保留旧来源。

        值相等不证明来源相等（可能撞值），假阳性概率见 /tmp/dfs_r20b.md 审计；
        策略与开关检查（LSGEMU_SHADOW_VALUE_EQUAL_RETAIN）都在本方法内。
        """
        if not self._shadow_value_equal_retain:
            return False
        for offset in range(max(1, int(size))):
            entry = self.memory_sources.get(int(address) + offset)
            if entry is None or entry.stored_value is None:
                return False
            if int(entry.stored_value) != ((int(store_value) >> (8 * offset)) & 0xFF):
                return False
        self.provenance_kill_stats["shadow_value_retained"] += 1
        return True

    def _verify_shadow_source_value(
        self,
        source: Optional[RegisterSource],
        address: int,
        size: int,
        observed_value: int,
    ) -> Optional[RegisterSource]:
        """load 转发前的值校验（默认开，安全网）。

        影子条目带 stored_value 且与实际读出值逐字节不符 → 来源不再可信
        （模拟器直写、混合尺寸覆写等），降级 opaque（保留 memory 地址、
        不再以可信身份转发）；无值可校（None）保持旧行为。
        """
        if source is None:
            return source
        mismatch = False
        checkable = False
        for offset in range(max(1, int(size))):
            entry = self.memory_sources.get(int(address) + offset)
            if entry is None or entry.stored_value is None:
                continue
            checkable = True
            if int(entry.stored_value) != ((int(observed_value) >> (8 * offset)) & 0xFF):
                mismatch = True
                break
        if not checkable:
            return source
        if not mismatch:
            self.provenance_kill_stats["shadow_forward_value_match"] += 1
            return source
        self.provenance_kill_stats["shadow_forward_value_mismatch"] += 1
        return replace(
            source,
            opaque_dependency=True,
            causal_complete=False,
            expression=f"value-mismatch@{hex(int(address) & 0xFFFFFFFF)}",
        )

    def _clear_memory_source(self, address: int, size: int):
        for offset in range(max(1, int(size))):
            self.memory_sources.pop(int(address) + offset, None)

    def _get_memory_source(self, address: int, size: int) -> Optional[RegisterSource]:
        sources = [
            self.memory_sources.get(int(address) + offset)
            for offset in range(max(1, int(size)))
        ]
        sources = [source for source in sources if source is not None]
        if not sources:
            return None
        source_span_complete = len(sources) == max(1, int(size))
        if not source_span_complete:
            return None
        provenance_keys = {
            (
                str(source.source_type or ""),
                int(source.mmio_address) if source.mmio_address is not None else None,
                int(source.memory_address) if source.memory_address is not None else None,
                int(source.source_pc) if source.source_pc is not None else None,
                tuple(source.chain),
            )
            for source in sources
        }
        if len(provenance_keys) != 1:
            composite = self._composite_source(
                sources[0].register if sources else "memory",
                sources,
                f"memory-load-composite 0x{int(address) & 0xFFFFFFFF:08x}/0x{max(1, int(size)):x}",
                operation="load_composite",
            )
            if composite is not None:
                context = self._classify_memory_context(int(address), int(size))
                composite.memory_address = int(address) & 0xFFFFFFFF
                composite.memory_role = context.get("memory_role")
                composite.memory_base = context.get("memory_base")
                composite.memory_offset = context.get("memory_offset")
                composite.memory_size = max(1, int(size))
            return composite
        return sources[0]

    def _source_dependency_key(self, source: RegisterSource) -> Optional[Tuple[str, int]]:
        if source.mmio_address is not None:
            return ("mmio", int(source.mmio_address))
        if source.memory_address is not None:
            return ("memory", int(source.memory_address))
        return None

    def _call_target_metadata(self, instruction: Dict) -> Dict[str, object]:
        mnemonic = str(instruction.get("mnemonic", "") or "").upper().split(".")[0]
        operands = self._split_operands(str(instruction.get("operands", "") or ""))
        operand = operands[0] if operands else ""
        target_register = self._normalize_register(operand)
        if target_register is not None:
            raw_target = self._read_register_value(target_register)
            return {
                "call_mnemonic": mnemonic,
                "callee_target": (
                    int(raw_target) & ~1 if raw_target is not None else None
                ),
                "callee_target_raw": raw_target,
                "callee_target_kind": (
                    "register_concrete" if raw_target is not None else "register_unresolved"
                ),
                "callee_target_register": target_register,
                "callee_target_statically_resolved": False,
            }

        immediate_match = re.search(
            r"(?<![A-Za-z0-9_])(?:0x[0-9A-Fa-f]+|[0-9]+)",
            operand.replace("#", ""),
        )
        raw_target = None
        if immediate_match is not None:
            try:
                raw_target = int(immediate_match.group(0), 0) & 0xFFFFFFFF
            except (TypeError, ValueError):
                raw_target = None
        return {
            "call_mnemonic": mnemonic,
            "callee_target": (
                int(raw_target) & ~1 if raw_target is not None else None
            ),
            "callee_target_raw": raw_target,
            "callee_target_kind": (
                "direct_immediate" if raw_target is not None else "unresolved"
            ),
            "callee_target_register": None,
            "callee_target_statically_resolved": raw_target is not None,
        }

    def _retire_call_frames(self, address: int):
        while self.call_stack and int(self.call_stack[-1].get("return_pc", 0)) == int(address):
            frame = self.call_stack.pop()
            try:
                self._restore_callee_saved_sources(frame, int(address))
                args = {
                    reg: source
                    for reg, source in (frame.get("args", {}) or {}).items()
                    if isinstance(source, RegisterSource)
                }
                before_values = {
                    str(reg): (
                        int(value) & 0xFFFFFFFF if value is not None else None
                    )
                    for reg, value in dict(
                        frame.get("argument_values_before", {}) or {}
                    ).items()
                }
                after_values = {
                    reg: self._read_register_value(reg)
                    for reg in ("r0", "r1", "r2", "r3")
                }
                if (
                    self.dynamic_graph_enabled
                    and int(frame.get("dynamic_graph_id", 0) or 0)
                    == id(self.dynamic_graph)
                ):
                    try:
                        self.dynamic_graph.observe_call_return(
                            call_pc=int(frame.get("call_pc", 0) or 0),
                            return_pc=int(address),
                            argument_nodes=dict(
                                frame.get("dynamic_argument_nodes", {}) or {}
                            ),
                            concrete_result=after_values.get("r0"),
                            result_register="r0",
                            callee_target=frame.get("callee_target"),
                            argument_value_before=before_values.get("r0"),
                        )
                    except Exception:
                        self.dynamic_graph.stats["opaque_call_return_errors"] += 1
                concrete_clobbered = []
                stale_provenance_cleared = []
                for reg in ("r0", "r1", "r2", "r3"):
                    before = before_values.get(reg)
                    after = after_values.get(reg)
                    if before is None or after is None or int(before) == int(after):
                        continue
                    concrete_clobbered.append(reg)
                    pre_call_source = args.get(reg)
                    if (
                        pre_call_source is not None
                        and self.register_sources.get(reg) is pre_call_source
                    ):
                        opaque = None
                        if reg == "r0" and args:
                            opaque = self._composite_source(
                                "r0",
                                args.values(),
                                f"opaque-call-return 0x{int(frame.get('call_pc', 0) or 0):08x}",
                                operation="opaque_call_return",
                                source_pc=int(address),
                            )
                        if opaque is not None:
                            opaque.opaque_dependency = True
                            opaque.causal_complete = False
                            self.register_sources[reg] = opaque
                            if opaque.mmio_address is not None:
                                self.reg_mmio_map[reg] = opaque.mmio_address
                            else:
                                self.reg_mmio_map.pop(reg, None)
                            self.provenance_kill_stats[
                                "opaque_call_dependency_preserved"
                            ] += 1
                        else:
                            self._clear_register_source(reg)
                            stale_provenance_cleared.append(reg)
                            self.provenance_kill_stats["callee_concrete_clobber"] += 1
                            self.provenance_kill_stats[
                                f"callee_concrete_clobber_{reg}"
                            ] += 1
                    self._lineage_note_write(
                        reg, int(address), "RET", "call_retire", list(args.keys())
                    )
                r0_source = self.register_sources.get("r0")
                r0_concrete_clobbered = "r0" in concrete_clobbered
                if r0_source is None and args and not r0_concrete_clobbered:
                    preferred = args.get("r0") or next(iter(args.values()))
                    self._set_register_source(
                        "r0",
                        preferred,
                        f"RET 0x{int(frame.get('call_pc', 0) or 0):08x}",
                        source_type=preferred.source_type,
                        memory_address=preferred.memory_address,
                        source_pc=int(address),
                        operation="call_return",
                    )
                serialized_args = {
                    reg: self._source_to_snapshot_dict(source)
                    for reg, source in args.items()
                }
                self.completed_call_frames.append({
                    "call_pc": int(frame.get("call_pc", 0) or 0),
                    "return_pc": int(frame.get("return_pc", 0) or 0),
                    "args": serialized_args,
                    "argument_values_before": before_values,
                    "argument_values_after": after_values,
                    "concrete_clobbered_registers": concrete_clobbered,
                    "stale_provenance_cleared": stale_provenance_cleared,
                    "return_propagated_to": (
                        "r0"
                        if r0_source is None and args and not r0_concrete_clobbered
                        else None
                    ),
                    "call_depth": int(len(self.call_stack)),
                    "call_mnemonic": str(frame.get("call_mnemonic", "") or ""),
                    "callee_target": frame.get("callee_target"),
                    "callee_target_raw": frame.get("callee_target_raw"),
                    "callee_target_kind": str(
                        frame.get("callee_target_kind", "unresolved") or "unresolved"
                    ),
                    "callee_target_register": frame.get("callee_target_register"),
                    "callee_target_statically_resolved": bool(
                        frame.get("callee_target_statically_resolved", False)
                    ),
                })
                if len(self.completed_call_frames) > 64:
                    del self.completed_call_frames[:-64]
            except Exception:
                pass

    _CALLEE_SAVED_REGISTERS = tuple(f"r{index}" for index in range(4, 12))

    def _restore_callee_saved_sources(self, frame: Dict[str, object], return_pc: int) -> None:
        """B4：callee-saved r4-r11 源跨调用恢复（值相等才恢复）。

        - 值相等 + 调用前有源 → 恢复（调用未改数据，调用方谱系仍真）；
        - 值不等 → 不恢复：寄存器被 callee 真实写入，调用方侧传播出的
          当前源才是真值（红线：不得拿过期源冒充）；
        - 调用前无源 → 维持现状（callee 内产生的更精确来源不抹掉）。
        """
        callee_saved = frame.get("callee_saved") or {}
        for reg, entry in callee_saved.items():
            try:
                saved_source, saved_value = entry
            except Exception:
                continue
            current_value = self._read_register_value(reg)
            if (
                saved_value is None
                or current_value is None
                or int(current_value) != int(saved_value)
            ):
                self.provenance_kill_stats["callee_saved_value_mismatch"] += 1
                continue
            if isinstance(saved_source, RegisterSource):
                self._set_register_source(
                    reg,
                    saved_source,
                    f"RESTORE {reg} @0x{return_pc:08x}",
                    operation="callee_saved_restore",
                    source_pc=int(return_pc),
                )
                self.provenance_kill_stats["callee_saved_restored_with_source"] += 1
            else:
                self.provenance_kill_stats["callee_saved_restored_no_source"] += 1
            self._lineage_note_write(
                reg, int(return_pc), "RESTORE", "callee_saved_restore", ()
            )

    def _record_call_frame(self, address: int, size: int, instruction: Dict):
        mnemonic = str(instruction.get('mnemonic', '')).upper().split('.')[0]
        if mnemonic not in {'BL', 'BLX'}:
            return
        return_pc = (int(address) + int(size or instruction.get('size', 2) or 2)) & ~1
        args = {
            reg: self.register_sources[reg]
            for reg in ('r0', 'r1', 'r2', 'r3')
            if reg in self.register_sources
        }
        # B4：callee-saved r4-r11 的（源, 值）快照——返回地址处值相等才恢复
        # （对齐 r0 返回值回填启发 :1833-1843；值不等即调用方侧为传播真值）。
        callee_saved = {}
        for reg in self._CALLEE_SAVED_REGISTERS:
            callee_saved[reg] = (
                self.register_sources.get(reg),
                self._read_register_value(reg),
            )
        target_metadata = self._call_target_metadata(instruction)
        self.call_stack.append({
            "call_pc": int(address),
            "return_pc": return_pc,
            "args": args,
            "callee_saved": callee_saved,
            "argument_values_before": {
                reg: self._read_register_value(reg)
                for reg in ("r0", "r1", "r2", "r3")
            },
            "dynamic_graph_id": id(self.dynamic_graph),
            "dynamic_argument_nodes": (
                self.dynamic_graph.capture_call_arguments()
                if self.dynamic_graph_enabled
                else {}
            ),
            **target_metadata,
        })
        self.max_call_depth_seen = max(self.max_call_depth_seen, len(self.call_stack))

    def _memory_operand_base_address(self, uc, operand_text: str) -> Optional[int]:
        """"[rn{, #±imm}]" 的常量解析（B2：LDRD/STRD 双寄存器定位用）。

        寄存器偏移 / 不可解析形态返回 None；后缀 writeback（"!"、
        后置 ", #imm"）不影响基址本身。
        """
        match = re.match(r"\[([^\]]+)\]", str(operand_text or "").strip())
        if not match:
            return None
        inner_parts = [p.strip().lower() for p in match.group(1).split(",") if p.strip()]
        if not inner_parts:
            return None
        base_reg = self._normalize_register(inner_parts[0])
        if base_reg is None:
            return None
        # 用调用方传入的 uc（hook 回调的 uc 才是当前执行现场），self.uc 仅兜底。
        base_value = self._read_register_value(base_reg, uc=uc)
        if base_value is None:
            return None
        total = int(base_value)
        for extra in inner_parts[1:]:
            token = extra.replace("#", "").strip()
            sign = 1
            if token.startswith("-"):
                sign = -1
                token = token[1:]
            if not re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", token):
                return None
            try:
                total += sign * int(token, 0)
            except ValueError:
                return None
        return total & 0xFFFFFFFF

    def _double_operand_register(self, uc, parts: List[str], address: int) -> Optional[str]:
        """LDRD/STRD rd1, rd2, [rn{, #off}]：按访问地址定位 rd1/rd2。

        rd1 ← [base]，rd2 ← [base+4]（T32 双字访问按字对齐成对递增）。
        """
        first = self._normalize_register(parts[0])
        second = self._normalize_register(parts[1])
        if first is None and second is None:
            return None
        base = self._memory_operand_base_address(uc, parts[2])
        if base is None:
            return None
        index = (int(address) - int(base)) // 4
        if index == 0:
            return first
        if index == 1:
            return second
        return None

    def _identify_load_target_register_for_access(self, uc, instruction: Dict, address: int, size: int) -> Optional[str]:
        mnemonic = str(instruction.get('mnemonic', '')).upper().split('.')[0]
        operands = str(instruction.get('operands', ''))
        parts = self._split_operands(operands)
        if mnemonic in {'LDR', 'LDRB', 'LDRH', 'LDRSB', 'LDRSH', 'LDR.W', 'VLDR'} and parts:
            return self._normalize_register(parts[0])
        if mnemonic == 'LDRD' and len(parts) >= 3:
            return self._double_operand_register(uc, parts, int(address))
        if mnemonic in {'POP', 'VPOP'} or mnemonic.startswith('LDM'):
            regs = self._register_list(operands)
            if not regs:
                return None
            # VPOP d 寄存器每项 8 字节，4 字节索引定位不了——拒绝（宁缺勿错）。
            if any(reg.startswith('d') for reg in regs):
                return None
            try:
                sp_value = int(uc.reg_read(UC_ARM_REG_SP))
            except Exception:
                sp_value = address
            index = (int(address) - sp_value) // 4
            if 0 <= index < len(regs):
                return regs[index]
        return None

    def _memory_address_registers(self, instruction: Dict) -> List[str]:
        """Return registers used to select the concrete memory location."""
        mnemonic = str(instruction.get("mnemonic", "") or "").upper().split(".")[0]
        operands = str(instruction.get("operands", "") or "")
        parts = self._split_operands(operands)
        for operand in parts:
            if "[" in operand or "]" in operand:
                return self._extract_operand_registers(operand)
        if mnemonic in {"PUSH", "POP"}:
            return ["sp"]
        if (mnemonic.startswith("LDM") or mnemonic.startswith("STM")) and parts:
            return self._extract_operand_registers(parts[0])
        return []

    def _combine_indirect_memory_source(
        self,
        dest_reg: str,
        data_source: Optional[RegisterSource],
        address_sources: Iterable[RegisterSource],
        instruction_text: str,
        *,
        access_pc: int,
        address: int,
        size: int,
        operation: str,
    ) -> Optional[RegisterSource]:
        pointer_sources = [
            source
            for source in address_sources or ()
            if isinstance(source, RegisterSource)
            and self._dependency_sites_for_source(source)
        ]
        if not pointer_sources:
            return data_source
        sources: List[RegisterSource] = []
        if isinstance(data_source, RegisterSource):
            sources.append(data_source)
        sources.extend(pointer_sources)
        combined = self._composite_source(
            dest_reg,
            sources,
            instruction_text,
            operation=f"opaque:{operation}",
            source_pc=int(access_pc),
        )
        if combined is None:
            return data_source
        context = self._classify_memory_context(
            int(address),
            int(size),
            access_pc=int(access_pc),
            stack_pointer=self._safe_read_sp(self.uc),
        )
        combined.memory_role = context.get("memory_role")
        combined.memory_base = context.get("memory_base")
        combined.memory_offset = context.get("memory_offset")
        combined.memory_access_pc = context.get("memory_access_pc")
        combined.memory_size = context.get("memory_size")
        combined.opaque_dependency = True
        combined.causal_complete = False
        self.provenance_kill_stats["opaque_indirect_address_dependency"] += 1
        self.provenance_kill_stats[f"opaque_{operation}"] += 1
        return combined

    def _identify_store_source_register_for_access(self, uc, instruction: Dict, address: int, size: int) -> Optional[str]:
        mnemonic = str(instruction.get('mnemonic', '')).upper().split('.')[0]
        operands = str(instruction.get('operands', ''))
        parts = self._split_operands(operands)
        if mnemonic in {'STR', 'STRB', 'STRH', 'STR.W', 'VSTR'} and parts:
            return self._normalize_register(parts[0])
        if mnemonic == 'STRD' and len(parts) >= 3:
            return self._double_operand_register(uc, parts, int(address))
        if mnemonic in {'PUSH', 'VPUSH'} or mnemonic.startswith('STM'):
            regs = self._register_list(operands)
            if not regs:
                return None
            # VPUSH d 寄存器每项 8 字节，索引定位不了——拒绝（宁缺勿错）。
            if any(reg.startswith('d') for reg in regs):
                return None
            try:
                sp_value = int(uc.reg_read(UC_ARM_REG_SP))
            except Exception:
                sp_value = address
            candidates = []
            start_old_sp = sp_value - 4 * len(regs)
            candidates.append((int(address) - start_old_sp) // 4)
            candidates.append((int(address) - sp_value) // 4)
            for index in candidates:
                if 0 <= index < len(regs):
                    return regs[index]
        return None

    # -- B1：产生者侧快照（最近产生者胜）+ 分支执行时延迟提交 ---------------

    _CMP_FAMILY = {'CMP', 'CMN', 'TST', 'TEQ'}

    def _flag_writer_profile(self, address: int, instruction: Optional[Dict]) -> Optional[Tuple[bool, List[str]]]:
        """pc → (是否写 NZCV, 参与快照的操作数文本列表)，静态指令级缓存。

        操作数抽取按家族跳过目的寄存器（SUBS r1,r2,r3 只记 r2/r3），
        防止把目的寄存器的陈旧源当作标志来源（假来源红线）：
        - CMP 族：前两个操作数（既有口径，双方都是比较输入）；
        - VMRS：无整数操作数来源（FP 域 B3 前不可见，记空而不是伪造）；
        - MSR：值操作数在第二位起；
        - MOVS/MVNS 两操作数：整体替换，旧值不是输入 → 只看第二操作数；
        - 其余两操作数读-改-写（ADDS r1,r2 / LSLS r1,r2 / LSLS r1,#imm）：
          旧值参与结果 → 两操作数都记。
        """
        if instruction is None:
            return None
        cached = self._flag_writer_profile_cache.get(int(address))
        if cached is not None:
            return cached
        covers, _kind, _c_source, _in_it = _producer_flag_writes(instruction)
        mnemonic = str(instruction.get('mnemonic', '') or '').upper()
        normalized = mnemonic.split('.')[0]
        parts = self._split_operands(str(instruction.get('operands', '')))
        operand_texts: List[str] = []
        if covers:
            if normalized in self._CMP_FAMILY:
                operand_texts = parts[:2]
            elif normalized == 'VMRS':
                operand_texts = []
            elif len(parts) >= 3:
                operand_texts = parts[1:]
            elif len(parts) == 2:
                if normalized in {'MOVS', 'MVNS', 'RRX', 'RRXS'}:
                    # 整体替换（MOVS rd,#imm / MOVS rd,rm）与纯写目的
                    # （RRX rd,rm）：旧值不是标志输入。
                    operand_texts = parts[1:]
                else:
                    operand_texts = parts
        profile = (bool(covers), operand_texts)
        self._flag_writer_profile_cache[int(address)] = profile
        return profile

    def _serialize_call_stack(self, limit: int = 8) -> List[Dict[str, object]]:
        return [
            {
                'call_pc': int(frame.get('call_pc', 0) or 0),
                'return_pc': int(frame.get('return_pc', 0) or 0),
                'call_mnemonic': str(frame.get('call_mnemonic', '') or ''),
                'callee_target': frame.get('callee_target'),
                'callee_target_kind': str(
                    frame.get('callee_target_kind', 'unresolved') or 'unresolved'
                ),
                'callee_target_register': frame.get('callee_target_register'),
                'callee_target_statically_resolved': bool(
                    frame.get('callee_target_statically_resolved', False)
                ),
                'argument_values_before': dict(
                    frame.get('argument_values_before', {}) or {}
                ),
                'args': {
                    reg: {
                        **self._source_to_snapshot_dict(source),
                    }
                    for reg, source in frame.get('args', {}).items()
                    if isinstance(source, RegisterSource)
                },
            }
            for frame in self.call_stack[-limit:]
        ]

    def _vfp_flag_profile(
        self, address: int, instruction: Optional[Dict]
    ) -> Optional[Tuple[str, List[str]]]:
        """VFP 标志谱系分类（B3，逐 pc 缓存）。

        - ('fpscr', texts)：VCMP/VCMPE 写 FPSCR 标志——pending_fpscr 槽；
        - ('vmsr', texts)：VMSR FPSCR, rX 整域搬运——pending_fpscr 槽；
        - ('vmrs_apsr', None)：VMRS APSR 把 FPSCR 转发进 APSR——
          pending_nzcv ← pending_fpscr（不产生新数据）。
        """
        if instruction is None:
            return None
        cached = self._vfp_flag_profile_cache.get(int(address))
        if cached is not None:
            return cached[0]
        mnemonic = str(instruction.get('mnemonic', '') or '').upper().split('.')[0]
        operands = str(instruction.get('operands', '') or '')
        profile = None
        if mnemonic in {'VCMP', 'VCMPE'}:
            parts = self._split_operands(operands)
            profile = ('fpscr', parts[:2])
        elif mnemonic == 'VMSR' and 'fpscr' in operands.lower():
            parts = self._split_operands(operands)
            profile = ('vmsr', parts[1:])
        elif mnemonic == 'VMRS' and 'apsr' in operands.lower():
            profile = ('vmrs_apsr', [])
        self._vfp_flag_profile_cache[int(address)] = (profile,)
        return profile

    def _build_flag_operand_sources(
        self, operand_texts: List[str]
    ) -> Dict[str, Dict[str, object]]:
        register_sources: Dict[str, Dict[str, object]] = {}
        for operand in operand_texts:
            for reg in self._extract_operand_registers(operand):
                source = self.register_sources.get(reg)
                if source is None:
                    continue
                if (
                    source.mmio_address is None
                    and source.memory_address is None
                    and not self._dependency_sites_for_source(source)
                ):
                    continue
                register_sources[reg] = {
                    **self._source_to_snapshot_dict(source),
                    'instruction': source.instruction,
                    'call_depth': int(source.call_depth),
                }
        return register_sources

    def _record_flag_producer_snapshot(self, address: int, instruction: Optional[Dict]) -> None:
        """写 NZCV 的指令执行时刷新 pending 槽（B1；最近产生者胜）。

        pending 只在「索引内 Bcc 执行」时提交（_record_branch_dependency_snapshot
        的延迟提交路径）。中途任何标志写者（含无活源者，register_sources 为空）
        都会覆盖槽——这保证提交时刻 pending 即该分支架构上真实的最近标志写者，
        不会把过期产生者记到分支头上。

        B3：VFP 链路走双槽模型——VCMP/VMSR 写 pending_fpscr；
        VMRS APSR 把 pending_fpscr 转发进 pending_nzcv（VMRS 不产生新数据，
        但覆写 APSR：上一个整数产生者必须失效）。
        """
        vfp_profile = self._vfp_flag_profile(int(address), instruction)
        if vfp_profile is not None:
            kind, operand_texts = vfp_profile
            if kind in {'fpscr', 'vmsr'}:
                register_sources = self._build_flag_operand_sources(operand_texts)
                self._flag_generation += 1
                pending: Dict[str, object] = {
                    'compare_pc': int(address),
                    'register_sources': register_sources,
                    'call_depth': len(self.call_stack),
                    'flag_generation': self._flag_generation,
                    'flag_domain': 'fpscr',
                }
                if register_sources:
                    pending['call_stack'] = self._serialize_call_stack()
                    pending['completed_call_frames'] = list(self.completed_call_frames[-8:])
                elif self._lineage_walk_enabled:
                    pending['dead_operands'] = self._lineage_dead_operand_report(
                        operand_texts
                    )
                self._pending_fpscr_snapshot = pending
            else:  # vmrs_apsr：转发或清空
                self._flag_generation += 1
                fpscr = self._pending_fpscr_snapshot
                if fpscr and fpscr.get('register_sources'):
                    self._pending_flag_snapshot = {
                        **fpscr,
                        'vmrs_forward_pc': int(address),
                        'flag_generation': self._flag_generation,
                    }
                else:
                    self._pending_flag_snapshot = {
                        'compare_pc': int(address),
                        'register_sources': {},
                        'call_depth': len(self.call_stack),
                        'flag_generation': self._flag_generation,
                        'vmrs_forward_pc': int(address),
                        'flag_domain': 'fpscr_empty',
                    }
            return
        profile = self._flag_writer_profile(int(address), instruction)
        if profile is None:
            return
        writes_flags, operand_texts = profile
        if not writes_flags:
            return
        register_sources = self._build_flag_operand_sources(operand_texts)
        self._flag_generation += 1
        pending = {
            'compare_pc': int(address),
            'register_sources': register_sources,
            'call_depth': len(self.call_stack),
            'flag_generation': self._flag_generation,
        }
        if register_sources:
            pending['call_stack'] = self._serialize_call_stack()
            pending['completed_call_frames'] = list(self.completed_call_frames[-8:])
        elif self._lineage_walk_enabled:
            # r23 计量：无活源的产生者把死操作数归因（死亡指令/回溯边界）
            # 随 pending 带走——提交时刻事件表已前进，必须产生者现场算。
            pending['dead_operands'] = self._lineage_dead_operand_report(
                operand_texts
            )
        self._pending_flag_snapshot = pending

    def _commit_pending_flag_snapshot(self, branch_pc: int) -> None:
        pending = self._pending_flag_snapshot
        if not pending:
            return
        if not pending.get('register_sources'):
            # r23 计量：空源 pending 只在带死因负载且显式开启时留诊断快照
            # （无 dependency_sites，不进候选通道，analyze 走 legacy 回退——
            # 生产行为零变化）；每分支至多 2 条防噪声。
            if not (self._lineage_walk_enabled and pending.get('dead_operands')):
                return
            items = self.branch_dependency_snapshots.setdefault(int(branch_pc), [])
            if sum(1 for item in items if item.get('diagnostic_only')) >= 2:
                return
            items.append({
                **pending,
                'register_sources': {},
                'diagnostic_only': True,
            })
            if len(items) > 4:
                del items[: len(items) - 4]
            return
        items = self.branch_dependency_snapshots.setdefault(int(branch_pc), [])
        items.append({
            **pending,
            'register_sources': dict(pending['register_sources']),
        })
        limit = self._branch_snapshot_limit(items)
        if len(items) > limit:
            del items[: len(items) - limit]

    def _record_branch_dependency_snapshot(self, address: int):
        address = int(address)
        if address in self._deferred_branch_pcs:
            # B1：索引内 Bcc——执行时提交最近产生者的 pending 快照。
            self._commit_pending_flag_snapshot(address)
            return
        branch_pcs = [
            int(branch_pc)
            for branch_pc in self.compare_pc_to_branch_pcs.get(address, [])
            if int(branch_pc) not in self._deferred_branch_pcs
        ]
        if not branch_pcs:
            return

        compare_insn = self._find_instruction_at_pc(address)
        if not compare_insn:
            return

        operands = self._split_operands(compare_insn.get('operands', ''))
        if len(operands) < 2:
            return

        register_sources: Dict[str, Dict[str, object]] = {}
        for operand in operands[:2]:
            for reg in self._extract_operand_registers(operand):
                source = self.register_sources.get(reg)
                if source is None:
                    continue
                if (
                    source.mmio_address is None
                    and source.memory_address is None
                    and not self._dependency_sites_for_source(source)
                ):
                    continue
                register_sources[reg] = {
                    **self._source_to_snapshot_dict(source),
                    'instruction': source.instruction,
                    'call_depth': int(source.call_depth),
                }

        if not register_sources:
            if self._lineage_walk_enabled:
                # r23 计量：索引外站（CBZ 值通道等）同样留死因负载。
                snapshot = {
                    'compare_pc': int(address),
                    'register_sources': {},
                    'dead_operands': self._lineage_dead_operand_report(
                        operands[:2]
                    ),
                    'call_depth': len(self.call_stack),
                    'diagnostic_only': True,
                }
                for branch_pc in branch_pcs:
                    items = self.branch_dependency_snapshots.setdefault(
                        int(branch_pc), []
                    )
                    if sum(
                        1 for item in items if item.get('diagnostic_only')
                    ) >= 2:
                        continue
                    items.append(dict(snapshot))
                    if len(items) > 4:
                        del items[: len(items) - 4]
            return

        snapshot = {
            'compare_pc': int(address),
            'register_sources': register_sources,
            'call_depth': len(self.call_stack),
            'call_stack': self._serialize_call_stack(),
            'completed_call_frames': list(self.completed_call_frames[-8:]),
        }
        for branch_pc in branch_pcs:
            items = self.branch_dependency_snapshots.setdefault(int(branch_pc), [])
            items.append(snapshot)
            limit = self._branch_snapshot_limit(items)
            if len(items) > limit:
                del items[: len(items) - limit]

    def _branch_snapshot_limit(self, snapshots: List[Dict[str, object]]) -> int:
        """Keep more variants when a branch has many distinct dependency shapes."""
        min_limit = max(1, int(self.min_branch_snapshot_variants or 16))
        max_limit = max(min_limit, int(self.max_branch_snapshot_variants or min_limit))
        signatures = set()
        for snapshot in snapshots or []:
            reg_sources = snapshot.get("register_sources", {}) if isinstance(snapshot, dict) else {}
            signature_parts = []
            for reg, source in sorted((reg_sources or {}).items()):
                if not isinstance(source, dict):
                    continue
                sites = self._dependency_sites_from_snapshot_source(source)
                signature_parts.append((str(reg), tuple((site[0], site[1]) for site in sites)))
            if signature_parts:
                signatures.add(tuple(signature_parts))
        adaptive = min_limit + max(0, len(signatures) - 1) * 2
        return min(max_limit, max(min_limit, adaptive))

    def _propagate_register_sources(self, instruction: Dict):
        """r23 计量外壳（默认关）：真实传播逻辑在 _impl，写事件留痕给回溯。"""
        if not self._lineage_walk_enabled:
            self._propagate_register_sources_impl(instruction)
            return
        mnemonic = str(instruction.get('mnemonic', '') or '').upper()
        operands = str(instruction.get('operands', ''))
        parts = self._split_operands(operands)
        normalized = mnemonic.split('.')[0]
        dest_regs: List[str] = []
        if normalized in {'POP', 'VPOP'}:
            dest_regs = list(self._register_list(operands))
        elif normalized == 'MRC' and len(parts) >= 3:
            mrc_dest = self._normalize_register(parts[2])
            dest_regs = [mrc_dest] if mrc_dest else []
        elif parts:
            first = self._normalize_register(parts[0])
            if first is not None:
                dest_regs.append(first)
            if normalized == 'LDRD' and len(parts) > 1:
                second = self._normalize_register(parts[1])
                if second is not None and second not in dest_regs:
                    dest_regs.append(second)
        input_regs = [
            reg
            for operand in parts
            for reg in self._extract_operand_registers(operand)
        ]
        try:
            self._lineage_note_context = {
                "pc": int(instruction.get('address', 0) or 0),
                "mnemonic": mnemonic,
                "kind": "propagate",
                "input_regs": input_regs,
            }
            self._propagate_register_sources_impl(instruction)
        finally:
            self._lineage_note_context = None
            pc = int(instruction.get('address', 0) or 0)
            for reg in dest_regs:
                self._lineage_note_write(
                    reg, pc, mnemonic, "propagate", input_regs
                )

    def _propagate_register_sources_impl(self, instruction: Dict):
        mnemonic = instruction['mnemonic'].upper()
        operands = str(instruction.get('operands', ''))
        parts = self._split_operands(operands)
        if not parts:
            return

        dest_reg = self._normalize_register(parts[0])
        normalized_mnemonic = mnemonic.split('.')[0]

        if normalized_mnemonic in {'STR', 'STRB', 'STRH', 'STRD', 'PUSH', 'VSTR', 'VPUSH'} or normalized_mnemonic.startswith('STM'):
            # STRD 是 store（双字），寄存器侧不清源；影子转发由 RAM hook 按
            # 访问地址定位 rd1/rd2（B2 传播族补全）。VSTR/VPUSH 同理（B3）。
            return

        # 以下家族不依赖 parts[0] 作目的寄存器——必须在 dest_reg None 检查前
        # 处理（pop {r4} 的首操作数是花括号列表，p15 也不是寄存器名）。
        if normalized_mnemonic in {'POP', 'VPOP'}:
            # B3 修正：本分支原先排在 dest_reg 检查之后，花括号形态永远走不到
            # （生产靠 RAM hook 兜底）；现提前到此处，弹出寄存器一律清源，
            # 影子命中时由 RAM hook 转发正确的目的。
            for reg in self._register_list(operands):
                self._clear_register_source(reg)
            return

        if normalized_mnemonic == 'MRC' and len(parts) >= 3:
            # 协处理器读：dest 在第三位（mrc p15, #0, r0, ...），值不来自
            # 域内任何寄存器——清除旧源（此前会漏清、留陈旧源）。
            mrc_dest = self._normalize_register(parts[2])
            if mrc_dest:
                self._clear_register_source(mrc_dest)
            return

        if dest_reg is None:
            return

        instruction_text = f"{mnemonic} {operands}".strip()

        if normalized_mnemonic == 'VMOV' and len(parts) == 2:
            # B3 整数桥：vmov s1, r0 / vmov r0, s1 / vmov.f32 s2, s3。
            source_regs = self._extract_operand_registers(parts[1])
            if len(source_regs) != 1:
                self._clear_register_source(dest_reg)
                return
            source = self.register_sources.get(source_regs[0])
            if source is None:
                self._clear_register_source(dest_reg)
                return
            self._set_register_source(dest_reg, source, instruction_text, operation=normalized_mnemonic)
            return

        unary_copy_ops = {
            'MOV', 'MOVS', 'MOVW', 'UXTB', 'UXTH', 'SXTB', 'SXTH',
            'REV', 'REV16', 'REVSH', 'UBFX', 'SBFX',
            # B2：RRX Rd,Rm 是纯写目的（Rd = C:Rm>>1），旧值不是输入；
            # C 输入在标志域（B1 pending / r19 谓词），不占整数寄存器通道。
            'RRX', 'RRXS',
            # B3：VFP 一元（vmov.f32 s0, #imm 走 MOV 同款清源路径）
            'VABS', 'VNEG', 'VCVT', 'VSQRT',
        }
        binary_propagate_ops = {
            'ADD', 'ADDS', 'SUB', 'SUBS',
            'AND', 'ANDS', 'ORR', 'ORRS', 'EOR', 'EORS', 'BIC', 'BICS',
            'LSL', 'LSLS', 'LSR', 'LSRS', 'ASR', 'ASRS', 'RSB', 'RSBS',
            'ORN', 'ORNS', 'ROR', 'RORS', 'MUL', 'MULS',
            # B2：单目的乘法族（无标志写，r18b 裁定）——三操作数形态
            # rd = rn*rm + ra，输入全在 parts[1:]。
            'MLA', 'MLS',
            # B3：VFP 算术（VMOV 三操作数形态 vmov d0, r0, r1 也在此）。
            'VADD', 'VSUB', 'VMUL', 'VDIV', 'VMLA', 'VNMLA', 'VMLS',
            'VNMLS', 'VNMUL', 'VMOV',
        }
        load_ops = {'LDR', 'LDR.W', 'LDRB', 'LDRH', 'LDRSB', 'LDRSH', 'VLDR'}

        if normalized_mnemonic in unary_copy_ops:
            if len(parts) < 2:
                self._clear_register_source(dest_reg)
                return
            source_operand = parts[1]
            source_regs = self._extract_operand_registers(source_operand)
            if len(source_regs) != 1:
                self._clear_register_source(dest_reg)
                return
            source = self.register_sources.get(source_regs[0])
            if source is None:
                self._clear_register_source(dest_reg)
                return
            self._set_register_source(dest_reg, source, instruction_text, operation=normalized_mnemonic)
            return

        if normalized_mnemonic in binary_propagate_ops:
            source_regs = []
            if len(parts) == 2:
                source_regs.extend(self._extract_operand_registers(parts[0]))
            for operand in parts[1:]:
                source_regs.extend(self._extract_operand_registers(operand))
            unique_source_regs = []
            for reg in source_regs:
                if reg not in unique_source_regs:
                    unique_source_regs.append(reg)

            tracked_sources = []
            for reg in unique_source_regs:
                source = self.register_sources.get(reg)
                if source and self._dependency_sites_for_source(source):
                    tracked_sources.append(source)
            dependency_keys = {
                site
                for source in tracked_sources
                for site in self._dependency_sites_for_source(source)
            }
            if not tracked_sources or not dependency_keys:
                if dest_reg in self.register_sources or dest_reg in self.reg_mmio_map:
                    self.provenance_kill_stats["unknown_destination_write"] += 1
                    self.provenance_kill_stats[f"mnemonic_{normalized_mnemonic}"] += 1
                self._clear_register_source(dest_reg)
                return
            if len(dependency_keys) == 1:
                self._set_register_source(dest_reg, tracked_sources[0], instruction_text, operation=normalized_mnemonic)
                return
            composite = self._composite_source(
                dest_reg,
                tracked_sources,
                instruction_text,
                operation=normalized_mnemonic,
            )
            if composite is None:
                if dest_reg in self.register_sources or dest_reg in self.reg_mmio_map:
                    self.provenance_kill_stats["unknown_destination_write"] += 1
                    self.provenance_kill_stats[f"mnemonic_{normalized_mnemonic}"] += 1
                self._clear_register_source(dest_reg)
                return
            self.register_sources[dest_reg] = composite
            if composite.mmio_address is not None:
                self.reg_mmio_map[dest_reg] = composite.mmio_address
            else:
                self.reg_mmio_map.pop(dest_reg, None)
            return

        if normalized_mnemonic in load_ops:
            # Non-MMIO loads should not inherit stale provenance.
            self._clear_register_source(dest_reg)
            return

        if normalized_mnemonic == 'LDRD':
            # 双目的 load：两个目的都不保留旧源（B2 前第二目的会带陈旧源——
            # 假来源缺口）；影子命中时由 RAM hook 转发正确的目的寄存器。
            self._clear_register_source(dest_reg)
            second_reg = self._normalize_register(parts[1]) if len(parts) > 1 else None
            if second_reg and second_reg != dest_reg:
                self._clear_register_source(second_reg)
            return

        if normalized_mnemonic in {'SMULL', 'UMULL', 'SMLAL', 'UMLAL'}:
            # B2：双目的乘法族。rdlo/rdhi 都不保留旧源；累加形（SMLAL/UMLAL）
            # 的 rdlo/rdhi 同时是输入。无活源输入时清两目的（带 kill 计量）。
            second_reg = self._normalize_register(parts[1]) if len(parts) > 1 else None
            source_regs: List[str] = []
            for operand in parts[2:]:
                source_regs.extend(self._extract_operand_registers(operand))
            if normalized_mnemonic in {'SMLAL', 'UMLAL'}:
                for operand in parts[:2]:
                    source_regs.extend(self._extract_operand_registers(operand))
            tracked_sources = []
            for reg in dict.fromkeys(source_regs):
                source = self.register_sources.get(reg)
                if source and self._dependency_sites_for_source(source):
                    tracked_sources.append(source)
            if not tracked_sources:
                for reg in (dest_reg, second_reg):
                    if reg and (reg in self.register_sources or reg in self.reg_mmio_map):
                        self.provenance_kill_stats["unknown_destination_write"] += 1
                    self._clear_register_source(reg)
                return
            composite = self._composite_source(
                dest_reg,
                tracked_sources,
                instruction_text,
                operation=normalized_mnemonic,
            )
            if composite is None:
                for reg in (dest_reg, second_reg):
                    self._clear_register_source(reg)
                return
            for reg in (dest_reg, second_reg):
                if not reg:
                    continue
                self.register_sources[reg] = (
                    composite if reg == dest_reg else self._clone_source(
                        reg, composite, instruction_text, operation=normalized_mnemonic
                    )
                )
                if self.register_sources[reg].mmio_address is not None:
                    self.reg_mmio_map[reg] = self.register_sources[reg].mmio_address
                else:
                    self.reg_mmio_map.pop(reg, None)
            return

        if normalized_mnemonic in {'BL', 'BLX', 'BX', 'BXJ'}:
            # 跳转/调用不写通用寄存器（BL/BLX 写 lr 但 lr 不在 parts[0] 位）。
            # B2 前漏了 BX：BX r3 被当「写 r3」清源，误杀跳转目标寄存器的
            # MMIO 源（实测 10 次 mmio_source_cleared）。
            return

        if normalized_mnemonic == 'MOVT':
            existing = self.register_sources.get(dest_reg)
            if existing is None:
                self._clear_register_source(dest_reg)
            else:
                self._set_register_source(
                    dest_reg,
                    existing,
                    instruction_text,
                    operation='MOVT',
                )
            return

        non_writing_ops = {
            'CMP', 'CMN', 'TST', 'TEQ', 'NOP', 'IT', 'DMB', 'DSB', 'ISB',
            'WFI', 'WFE', 'SEV', 'SVC', 'BKPT', 'CPSID', 'CPSIE',
            # B3：VFP 比较不写通用/VFP 寄存器（写 FPSCR，走 pending_fpscr 槽）
            'VCMP', 'VCMPE',
        }
        if normalized_mnemonic in non_writing_ops:
            return

        source_regs = []
        if len(parts) == 2 and normalized_mnemonic in {
            'ADC', 'ADCS', 'SBC', 'SBCS',
        }:
            # 累加/借位族两操作数形态：旧值参与结果（r1 = r1 ± r2 ± C）。
            # RRX/RRXS 已移入 unary_copy_ops（纯写目的，B2）。
            source_regs.extend(self._extract_operand_registers(parts[0]))
        for operand in parts[1:]:
            source_regs.extend(self._extract_operand_registers(operand))
        tracked_sources = [
            self.register_sources[register]
            for register in dict.fromkeys(source_regs)
            if register in self.register_sources
            and self._dependency_sites_for_source(self.register_sources[register])
        ]
        if tracked_sources:
            opaque = self._composite_source(
                dest_reg,
                tracked_sources,
                instruction_text,
                operation=f"opaque:{normalized_mnemonic}",
            )
            if opaque is not None:
                opaque.opaque_dependency = True
                opaque.causal_complete = False
                self.register_sources[dest_reg] = opaque
                if opaque.mmio_address is not None:
                    self.reg_mmio_map[dest_reg] = opaque.mmio_address
                else:
                    self.reg_mmio_map.pop(dest_reg, None)
                self.provenance_kill_stats["opaque_dependency_preserved"] += 1
                self.provenance_kill_stats[
                    f"opaque_mnemonic_{normalized_mnemonic}"
                ] += 1
                return

        # If no observed dependency reaches the destination, invalidate the old
        # source rather than attributing a later branch to stale provenance.
        if dest_reg in self.register_sources or dest_reg in self.reg_mmio_map:
            self.provenance_kill_stats["unknown_destination_write"] += 1
            self.provenance_kill_stats[f"mnemonic_{normalized_mnemonic}"] += 1
        self._clear_register_source(dest_reg)

    def _identify_target_register(
        self,
        pc: int,
        mmio_addr: int,
        value: int,
        occurrence: Optional[int] = None,
        size: Optional[int] = None,
    ):
        """
        识别 MMIO 读取的目标寄存器

        策略:
        1. 找到 PC 对应的指令
        2. 解析指令，提取目标寄存器
        3. 建立寄存器-MMIO 映射
        """
        # 查找指令
        instruction = self._find_instruction_at_pc(pc)
        if not instruction:
            return

        mnemonic = instruction['mnemonic'].upper()
        normalized_mnemonic = mnemonic.split('.')[0]
        operands = instruction['operands']

        # 解析 LDR 指令
        if normalized_mnemonic in ['LDR', 'LDRB', 'LDRH', 'LDRSB', 'LDRSH']:
            # LDR Rd, [Rn, #offset] 或 LDR Rd, [Rn]
            parts = self._split_operands(operands)
            if len(parts) >= 1:
                target_reg = parts[0].strip().lower()

                # 记录寄存器来源
                source = RegisterSource(
                    register=target_reg,
                    source_type='mmio',
                    mmio_address=mmio_addr,
                    mmio_value=value,
                    source_pc=pc,
                    origin_source_pc=pc,
                    instruction=f"{mnemonic} {operands}",
                    expression=f"MMIO[{hex(int(mmio_addr))}]",
                    chain=(f"MMIO[{hex(int(mmio_addr))}]@{hex(int(pc))}",),
                    call_depth=len(self.call_stack),
                    source_occurrence=(
                        max(1, int(occurrence))
                        if occurrence is not None
                        else None
                    ),
                    dependency_sites=((
                        "mmio",
                        int(mmio_addr) & 0xFFFFFFFF,
                        int(pc) & 0xFFFFFFFF,
                        int(pc) & 0xFFFFFFFF,
                    ),),
                )

                self.register_sources[target_reg] = source
                self.reg_mmio_map[target_reg] = mmio_addr

                if self.dynamic_graph_enabled:
                    if normalized_mnemonic in {"LDRB", "LDRSB"}:
                        read_size = 1
                    elif normalized_mnemonic in {"LDRH", "LDRSH"}:
                        read_size = 2
                    else:
                        read_size = max(1, min(4, int(size or 4)))
                    self.dynamic_graph.observe_external_read(
                        kind="mmio",
                        read_pc=int(pc),
                        address=int(mmio_addr),
                        occurrence=max(1, int(occurrence or 1)),
                        size=read_size,
                        observed_value=int(value),
                        destination_register=target_reg,
                        signed=normalized_mnemonic in {"LDRSB", "LDRSH"},
                    )

                logger.debug(f"[RegisterTracer] {target_reg} <- MMIO[{hex(mmio_addr)}] = {hex(value)} @ PC={hex(pc)}")

    def _find_instruction_at_pc(self, pc: int) -> Optional[Dict]:
        """查找 PC 对应的指令"""
        return self.instruction_lookup.get(pc)

    def trace_register_back(self, register: str, max_depth: int = 10) -> Optional[RegisterSource]:
        """
        反向追踪寄存器来源

        Args:
            register: 寄存器名称 (如 'r0')
            max_depth: 最大追踪深度

        Returns:
            RegisterSource 或 None
        """
        current_reg = register.lower()
        depth = 0
        visited = set()

        while depth < max_depth:
            if current_reg in visited:
                break
            visited.add(current_reg)
            # 查找当前寄存器的来源
            if current_reg in self.register_sources:
                source = self.register_sources[current_reg]

                # 如果来源是 MMIO 或 memory，返回
                if (
                    source.mmio_address is not None
                    or source.memory_address is not None
                    or self._dependency_sites_for_source(source)
                ):
                    return source

                # 如果来源是另一个寄存器，继续追踪
                if source.source_type == 'register':
                    source_regs = self._source_registers_from_instruction(source.instruction or "")
                    next_regs = [reg for reg in source_regs if reg != current_reg]
                    if len(next_regs) != 1:
                        break
                    current_reg = next_regs[0]
                    depth += 1
                    continue

            break

        return None

    def _source_registers_from_instruction(self, instruction_text: str) -> List[str]:
        text = str(instruction_text or "").strip()
        if not text:
            return []
        if " " in text:
            _mnemonic, operands = text.split(None, 1)
        else:
            operands = ""
        parts = self._split_operands(operands)
        if len(parts) < 2:
            return []
        regs: List[str] = []
        for operand in parts[1:]:
            for reg in self._extract_operand_registers(operand):
                if reg not in regs:
                    regs.append(reg)
        return regs

    def get_mmio_for_register(self, register: str) -> Optional[int]:
        """
        获取寄存器对应的 MMIO 地址

        Args:
            register: 寄存器名称

        Returns:
            MMIO 地址或 None
        """
        reg = register.lower()
        return self.reg_mmio_map.get(reg)

    def get_recent_mmio_reads(self, count: int = 10) -> List[Tuple[int, int, int]]:
        """
        获取最近的 MMIO 读取

        Returns:
            [(pc, mmio_addr, value), ...]
        """
        reads = [(pc, addr, val) for pc, addr, val, is_read in self.mmio_history if is_read]
        return reads[-count:]

    def analyze_branch_mmio_dependency(self, branch_pc: int) -> Dict:
        """
        分析分支对 MMIO 的依赖

        Args:
            branch_pc: 分支指令的 PC

        Returns:
            {
                'mmio_addresses': [addr1, addr2, ...],
                'registers': ['r0', 'r1', ...],
                'dependencies': [(reg, mmio_addr), ...]
            }
        """
        result = {
            'mmio_addresses': [],
            'registers': [],
            'dependencies': [],
            'snapshot_count': 0,
            'call_stack_samples': [],
        }

        snapshots = self.branch_dependency_snapshots.get(int(branch_pc), [])
        if snapshots:
            result['snapshot_count'] = len(snapshots)
            for snapshot in reversed(snapshots):
                if len(result['call_stack_samples']) < 8:
                    result['call_stack_samples'].append({
                        'compare_pc': int(snapshot.get('compare_pc', 0) or 0),
                        'call_depth': int(snapshot.get('call_depth', 0) or 0),
                        'call_stack': list(snapshot.get('call_stack', []) or []),
                        'completed_call_frames': list(snapshot.get('completed_call_frames', []) or []),
                    })
                for reg, source in snapshot.get('register_sources', {}).items():
                    if reg not in result['registers']:
                        result['registers'].append(reg)
                    for dep_type, address, _source_pc, _origin_pc in self._dependency_sites_from_snapshot_source(source):
                        if dep_type != "mmio":
                            continue
                        result['mmio_addresses'].append(int(address))
                        result['dependencies'].append((reg, int(address)))
            result['mmio_addresses'] = list(dict.fromkeys(result['mmio_addresses']))
            if result['mmio_addresses']:
                return result

        # 查找分支前的比较指令
        compare_insn = self._find_compare_before_branch(branch_pc)
        if not compare_insn:
            return result

        # 解析比较指令的操作数
        operands = self._split_operands(compare_insn['operands'])
        if len(operands) < 2:
            return result

        for operand in operands[:2]:
            for reg in self._extract_operand_registers(operand):
                if reg not in result['registers']:
                    result['registers'].append(reg)
                source = self.trace_register_back(reg)
                if source:
                    for dep_type, address, _source_pc, _origin_pc in self._dependency_sites_for_source(source):
                        if dep_type != "mmio":
                            continue
                        result['mmio_addresses'].append(address)
                        result['dependencies'].append((reg, address))

        result['mmio_addresses'] = list(dict.fromkeys(result['mmio_addresses']))
        return result

    def analyze_branch_dependencies(self, branch_pc: int) -> Dict:
        """
        分析分支对 memory/MMIO 的统一依赖。

        Returns:
            {
                'registers': [...],
                'dependencies': [
                    {
                        'register': 'r3',
                        'type': 'mmio' | 'memory',
                        'address': 0x...,
                        'source_pc': 0x...,
                        'memory_address': 0x... | None,
                        'mmio_address': 0x... | None,
                        'call_depth': 0,
                    }, ...
                ],
                'mmio_addresses': [...],
                'memory_addresses': [...],
                'snapshot_count': N,
            }
        """
        result = {
            'registers': [],
            'dependencies': [],
            'mmio_addresses': [],
            'memory_addresses': [],
            'snapshot_count': 0,
            'call_stack_samples': [],
        }

        seen_dependencies = set()
        snapshots = self.branch_dependency_snapshots.get(int(branch_pc), [])
        if snapshots:
            result['snapshot_count'] = len(snapshots)
            for snapshot in reversed(snapshots):
                if len(result['call_stack_samples']) < 8:
                    result['call_stack_samples'].append({
                        'compare_pc': int(snapshot.get('compare_pc', 0) or 0),
                        'call_depth': int(snapshot.get('call_depth', 0) or 0),
                        'call_stack': list(snapshot.get('call_stack', []) or []),
                        'completed_call_frames': list(snapshot.get('completed_call_frames', []) or []),
                    })
                for reg, source in snapshot.get('register_sources', {}).items():
                    if reg not in result['registers']:
                        result['registers'].append(reg)
                    source_type = str(source.get('source_type', '') or '').lower()
                    mmio_addr = source.get('mmio_address')
                    memory_addr = source.get('memory_address')
                    sites = self._dependency_sites_from_snapshot_source(source)
                    for record in self._dependency_records_from_sites(
                        reg,
                        sites,
                        source_type=source_type,
                        mmio_addr=mmio_addr,
                        memory_addr=memory_addr,
                        call_depth=int(source.get('call_depth', 0) or 0),
                        expression=source.get('expression'),
                        memory_context=self._memory_context_from_snapshot_source(source),
                        opaque_dependency=bool(source.get('opaque_dependency', False)),
                        causal_complete=bool(source.get('causal_complete', True)),
                    ):
                        if record.get('type') == 'mmio' and source.get('source_occurrence') is not None:
                            record['read_occurrence'] = max(
                                1,
                                int(source.get('source_occurrence')),
                            )
                        dep_key = (
                            reg,
                            record['type'],
                            int(record['address']),
                            int(record.get('source_pc', 0) or 0),
                            int(record.get('read_occurrence', 0) or 0),
                        )
                        if dep_key in seen_dependencies:
                            continue
                        seen_dependencies.add(dep_key)
                        result['dependencies'].append(record)
                        if record['type'] == 'mmio':
                            result['mmio_addresses'].append(int(record['address']))
                        else:
                            result['memory_addresses'].append(int(record['address']))
            result['mmio_addresses'] = list(dict.fromkeys(result['mmio_addresses']))
            result['memory_addresses'] = list(dict.fromkeys(result['memory_addresses']))
            if result['dependencies']:
                return result

        compare_insn = self._find_compare_before_branch(branch_pc)
        if not compare_insn:
            return result

        operands = self._split_operands(compare_insn['operands'])
        if len(operands) < 2:
            return result

        for operand in operands[:2]:
            for reg in self._extract_operand_registers(operand):
                if reg not in result['registers']:
                    result['registers'].append(reg)
                source = self.trace_register_back(reg)
                if source is None:
                    continue
                for record in self._dependency_records_from_sites(
                    reg,
                    self._dependency_sites_for_source(source),
                    source_type=source.source_type,
                    mmio_addr=source.mmio_address,
                    memory_addr=source.memory_address,
                    call_depth=int(source.call_depth or 0),
                    expression=source.expression,
                    memory_context=self._memory_context_from_source(source),
                    opaque_dependency=bool(source.opaque_dependency),
                    causal_complete=bool(source.causal_complete),
                ):
                    if record.get('type') == 'mmio' and source.source_occurrence is not None:
                        record['read_occurrence'] = max(1, int(source.source_occurrence))
                    dep_key = (
                        reg,
                        record['type'],
                        int(record['address']),
                        int(record.get('source_pc', 0) or 0),
                        int(record.get('read_occurrence', 0) or 0),
                    )
                    if dep_key in seen_dependencies:
                        continue
                    seen_dependencies.add(dep_key)
                    result['dependencies'].append(record)
                    if record['type'] == 'mmio':
                        result['mmio_addresses'].append(int(record['address']))
                    else:
                        result['memory_addresses'].append(int(record['address']))

        result['mmio_addresses'] = list(dict.fromkeys(result['mmio_addresses']))
        result['memory_addresses'] = list(dict.fromkeys(result['memory_addresses']))
        return result

    def _find_compare_before_branch(self, branch_pc: int) -> Optional[Dict]:
        """查找分支前的比较指令"""
        if branch_pc in self.compare_lookup:
            return self.compare_lookup[branch_pc]

        # 兼容旧调用方：回退到线性查找
        for bb_addr, instructions in self.static_bbs.items():
            if bb_addr <= branch_pc < bb_addr + 0x100:
                for i, insn in enumerate(instructions):
                    if insn['address'] == branch_pc:
                        # 向前查找比较指令
                        for j in range(i - 1, max(-1, i - 6), -1):
                            prev_insn = instructions[j]
                            if prev_insn['mnemonic'].upper().split('.')[0] in ['CMP', 'CMN', 'TST', 'TEQ']:
                                return prev_insn
        return None

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        dynamic_summaries = [
            graph.summary()
            for graph in self._dynamic_graph_candidates()
            if graph.has_execution_data
        ]
        return {
            'total_mmio_accesses': int(
                getattr(self, 'mmio_history_total', len(self.mmio_history)) or 0
            ),
            'retained_mmio_accesses': len(self.mmio_history),
            'mmio_history_retention': {
                'limit': int(getattr(self, 'mmio_history_limit', 0) or 0),
                'retained': len(self.mmio_history),
                'total': int(
                    getattr(self, 'mmio_history_total', len(self.mmio_history)) or 0
                ),
                'discarded': int(
                    getattr(self, 'mmio_history_entries_discarded', 0) or 0
                ),
            },
            'mmio_reads': int(
                getattr(self, 'mmio_read_total', 0)
                or sum(1 for _, _, _, is_read in self.mmio_history if is_read)
            ),
            'mmio_writes': int(
                getattr(self, 'mmio_write_total', 0)
                or sum(1 for _, _, _, is_read in self.mmio_history if not is_read)
            ),
            'tracked_registers': len(self.register_sources),
            'reg_mmio_mappings': len(self.reg_mmio_map),
            'tracked_memory_bytes': len(self.memory_sources),
            'call_depth': len(self.call_stack),
            'max_call_depth_seen': self.max_call_depth_seen,
            'completed_call_frames': len(self.completed_call_frames),
            'branch_dependency_snapshot_points': len(self.branch_dependency_snapshots),
            'mmio_read_occurrence_sites': len(self.mmio_read_occurrence_counts),
            'external_memory_read_occurrence_sites': len(
                self.external_memory_read_occurrence_counts
            ),
            'external_input_event_source_configured': bool(
                self.external_input_event_source is not None
            ),
            'external_input_event_observer_attached': bool(
                self._external_input_observer_attached
            ),
            'external_input_event_stats': dict(self.external_input_event_stats),
            'provenance_kill_stats': dict(self.provenance_kill_stats),
            'dynamic_constraint_recovery_enabled': bool(self.dynamic_graph_enabled),
            'dynamic_graph_sessions': len(dynamic_summaries),
            'dynamic_expression_nodes': sum(
                int(item.get('nodes', 0) or 0) for item in dynamic_summaries
            ),
            'dynamic_input_leaves': sum(
                int(item.get('inputs', 0) or 0) for item in dynamic_summaries
            ),
            'dynamic_input_dependent_branches': sum(
                int(item.get('input_dependent_branches', 0) or 0)
                for item in dynamic_summaries
            ),
            'dynamic_solver_calls': sum(
                int(item.get('solver_calls', 0) or 0) for item in dynamic_summaries
            ),
            'dynamic_solver_models': sum(
                int(item.get('solver_models', 0) or 0) for item in dynamic_summaries
            ),
            'dynamic_relation_counts': self._dynamic_relation_counts(dynamic_summaries),
            'dynamic_solver_status_counts': self._dynamic_stat_prefix_counts(
                dynamic_summaries,
                "solver_status_",
            ),
            'dynamic_solver_unknown_reasons': self._dynamic_stat_prefix_counts(
                dynamic_summaries,
                "solver_unknown_",
            ),
            'dynamic_omitted_path_predicates': sum(
                int(item.get('stats', {}).get('omitted_path_predicates', 0) or 0)
                for item in dynamic_summaries
            ),
            'dynamic_solver_escalation': dict(
                self.dynamic_solver_escalation_stats
            ),
            'dynamic_checksum_fallback': dict(
                self.dynamic_checksum_fallback_stats
            ),
            'dynamic_llm_fallback': dict(
                self.dynamic_llm_fallback_stats
            ),
            'dynamic_llm_fallback_configured': bool(
                self.dynamic_llm_fallback is not None
            ),
            'differential_probe': dict(
                self.differential_probe_stats
            ),
            'differential_probe_configured': bool(
                self.differential_probe_fallback is not None
            ),
        }

    @staticmethod
    def _dynamic_relation_counts(summaries: Iterable[Dict[str, object]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for summary in summaries or []:
            for key, value in dict(summary.get("relation_counts", {}) or {}).items():
                counts[str(key)] = int(counts.get(str(key), 0) or 0) + int(value or 0)
        return counts

    @staticmethod
    def _dynamic_stat_prefix_counts(
        summaries: Iterable[Dict[str, object]],
        prefix: str,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for summary in summaries or []:
            for key, value in dict(summary.get("stats", {}) or {}).items():
                text = str(key)
                if not text.startswith(prefix):
                    continue
                normalized = text[len(prefix):] or "unknown"
                counts[normalized] = int(counts.get(normalized, 0) or 0) + int(value or 0)
        return counts

    def recover_dynamic_branch_inputs(
        self,
        *,
        branch_pc: int,
        occurrence: int,
        target_taken: bool,
        max_models: int = 4,
        avoid_values_by_site: Optional[Dict[Tuple[object, ...], Set[int]]] = None,
        baseline_values_by_site: Optional[Dict[Tuple[object, ...], int]] = None,
    ) -> DynamicRecoveryResult:
        """Solve the newest exact dynamic predicate, retaining legacy fallback."""
        if not self.dynamic_graph_enabled:
            return DynamicRecoveryResult(tuple(), "dynamic_recovery_disabled", "disabled")
        best_failure = DynamicRecoveryResult(
            tuple(),
            "dynamic_branch_predicate_missing",
            "z3",
        )
        exact_graphs: List[DynamicExpressionGraph] = []
        fallback_graphs: List[DynamicExpressionGraph] = []
        for graph in reversed(self._dynamic_graph_candidates()):
            if (int(branch_pc) & 0xFFFFFFFF, max(1, int(occurrence))) in graph.branch_records:
                exact_graphs.append(graph)
            elif graph.find_branch_record(branch_pc, occurrence) is not None:
                fallback_graphs.append(graph)
        if not exact_graphs and not fallback_graphs:
            # 第一级（表达式图）不可用：没有任何图记录该分支（间接跳转、
            # 动态代码、图溢出等）。交由上层做二级差分扰动定位（设计 §4.2）；
            # 无实现或无结果时维持原先的死路返回。
            probe_result = self._differential_probe_recover(
                branch_pc=int(branch_pc),
                occurrence=max(1, int(occurrence)),
                target_taken=bool(target_taken),
            )
            if probe_result is not None:
                return probe_result
        for graph in exact_graphs + fallback_graphs:
            initial_limits = {
                "max_inputs": self._env_int("LSGEMU_DYNAMIC_SLICE_MAX_INPUTS", 256),
                "max_slice_nodes": self._env_int("LSGEMU_DYNAMIC_SLICE_MAX_NODES", 8192),
                "max_path_predicates": self._env_int("LSGEMU_DYNAMIC_PATH_PREDICATE_LIMIT", 128),
                "timeout_ms": self._env_int("LSGEMU_DYNAMIC_SOLVER_TIMEOUT_MS", 150),
            }
            result = graph.recover_branch_inputs(
                branch_pc=int(branch_pc),
                occurrence=max(1, int(occurrence)),
                target_taken=bool(target_taken),
                max_models=max(1, int(max_models)),
                max_inputs=initial_limits["max_inputs"],
                max_slice_nodes=initial_limits["max_slice_nodes"],
                max_path_predicates=initial_limits["max_path_predicates"],
                timeout_ms=initial_limits["timeout_ms"],
                avoid_values_by_site=avoid_values_by_site,
                baseline_values_by_site=baseline_values_by_site,
            )
            result = replace(
                result,
                initial_solver_status=str(result.solver_status),
                initial_limits=tuple(sorted(initial_limits.items())),
                final_limits=tuple(sorted(initial_limits.items())),
            )
            if not result.models and result.solver_status in {"unknown", "budget"}:
                caps = {
                    "max_inputs": self._env_int("LSGEMU_DYNAMIC_ESCALATION_MAX_INPUTS", 512),
                    "max_slice_nodes": self._env_int("LSGEMU_DYNAMIC_ESCALATION_MAX_NODES", 16384),
                    "max_path_predicates": self._env_int("LSGEMU_DYNAMIC_ESCALATION_MAX_PATH_PREDICATES", 256),
                    "timeout_ms": self._env_int("LSGEMU_DYNAMIC_ESCALATION_MAX_TIMEOUT_MS", 500),
                }
                escalated_limits = {
                    key: max(
                        int(value),
                        min(int(caps[key]), int(value) * 2),
                    )
                    for key, value in initial_limits.items()
                }
                if escalated_limits != initial_limits:
                    self.dynamic_solver_escalation_stats["attempted"] += 1
                    self.dynamic_solver_escalation_stats[
                        f"reason_{result.solver_status}"
                    ] += 1
                    escalated = graph.recover_branch_inputs(
                        branch_pc=int(branch_pc),
                        occurrence=max(1, int(occurrence)),
                        target_taken=bool(target_taken),
                        max_models=max(1, int(max_models)),
                        max_inputs=escalated_limits["max_inputs"],
                        max_slice_nodes=escalated_limits["max_slice_nodes"],
                        max_path_predicates=escalated_limits["max_path_predicates"],
                        timeout_ms=escalated_limits["timeout_ms"],
                        avoid_values_by_site=avoid_values_by_site,
                        baseline_values_by_site=baseline_values_by_site,
                    )
                    self.dynamic_solver_escalation_stats[
                        f"result_{escalated.solver_status}"
                    ] += 1
                    self.dynamic_solver_escalation_stats["models"] += len(
                        escalated.models
                    )
                    result = replace(
                        escalated,
                        solver_attempts=2,
                        escalated=True,
                        initial_solver_status=str(result.solver_status),
                        initial_limits=tuple(sorted(initial_limits.items())),
                        final_limits=tuple(sorted(escalated_limits.items())),
                    )
            if result.models:
                return result
            is_exact = graph in exact_graphs
            if (
                is_exact
                and result.solver_status == "unsat"
                and int(result.omitted_path_predicates or 0) == 0
                and str(result.branch_record_match) == "exact"
            ):
                return result
            if result.solver_status in {
                "unsupported",
                "unknown",
                "budget",
                "unavailable",
            }:
                checksum_result = graph.recover_checksum_candidates(
                    branch_pc=int(branch_pc),
                    occurrence=max(1, int(occurrence)),
                    max_models=max(1, int(max_models)),
                    trigger_status=str(result.solver_status),
                )
                self.dynamic_checksum_fallback_stats["calls"] += 1
                self.dynamic_checksum_fallback_stats[
                    f"trigger_{result.solver_status}"
                ] += 1
                self.dynamic_checksum_fallback_stats["models"] += len(
                    checksum_result.models
                )
                if checksum_result.models:
                    return replace(
                        checksum_result,
                        solver_attempts=result.solver_attempts,
                        escalated=result.escalated,
                        initial_solver_status=result.initial_solver_status,
                        initial_limits=result.initial_limits,
                        final_limits=result.final_limits,
                    )
            # 第三级：符号路径与确定性校验和枚举都失败后，把该次求解使用的
            # 切片升级给大模型求解；结果仍是 hypothesis，走与 z3 完全相同的
            # 候选集与无强制重放验证。
            if self.dynamic_llm_fallback is not None and self._dynamic_llm_fallback_eligible(result):
                llm_result = self._dynamic_llm_slice_solve(
                    graph=graph,
                    branch_pc=int(branch_pc),
                    occurrence=max(1, int(occurrence)),
                    target_taken=bool(target_taken),
                    failure_result=result,
                    avoid_values_by_site=avoid_values_by_site,
                )
                if llm_result is not None and llm_result.models:
                    return llm_result
            if result.reason != "dynamic_branch_predicate_missing":
                best_failure = result
        return best_failure

    def merge_from(self, other: "RegisterTracer") -> None:
        if other is None:
            return

        for key, value in dict(
            getattr(other, "external_input_event_stats", {}) or {}
        ).items():
            try:
                self.external_input_event_stats[str(key)] = int(
                    self.external_input_event_stats.get(str(key), 0) or 0
                ) + int(value or 0)
            except (TypeError, ValueError):
                continue

        self.provenance_kill_stats.update(
            Counter(getattr(other, "provenance_kill_stats", Counter()) or {})
        )
        # r23 计量：死操作数归因计数随合并累加（诊断口径，无语义影响）。
        self.provenance_dead_operand_stats.update(
            Counter(getattr(other, "provenance_dead_operand_stats", Counter()) or {})
        )
        self.dynamic_solver_escalation_stats.update(
            Counter(
                getattr(other, "dynamic_solver_escalation_stats", Counter())
                or {}
            )
        )
        self.dynamic_checksum_fallback_stats.update(
            Counter(
                getattr(other, "dynamic_checksum_fallback_stats", Counter())
                or {}
            )
        )
        self.dynamic_llm_fallback_stats.update(
            Counter(
                getattr(other, "dynamic_llm_fallback_stats", Counter())
                or {}
            )
        )
        self.differential_probe_stats.update(
            Counter(
                getattr(other, "differential_probe_stats", Counter())
                or {}
            )
        )

        for branch_pc, snapshots in getattr(other, "branch_dependency_snapshots", {}).items():
            items = self.branch_dependency_snapshots.setdefault(int(branch_pc), [])
            items.extend(list(snapshots or []))
            if len(items) > 32:
                del items[:-32]

        other_completed = list(getattr(other, "completed_call_frames", []) or [])
        if other_completed:
            self.completed_call_frames.extend(other_completed)
            if len(self.completed_call_frames) > 64:
                del self.completed_call_frames[:-64]

        self.max_call_depth_seen = max(
            int(self.max_call_depth_seen),
            int(getattr(other, "max_call_depth_seen", 0) or 0),
        )
        for graph in list(getattr(other, "dynamic_graph_archives", []) or []):
            self._archive_dynamic_graph(graph)
        self._archive_dynamic_graph(getattr(other, "dynamic_graph", None))

    def clear(self):
        """清空追踪数据"""
        self.register_sources.clear()
        self.mmio_history.clear()
        self.mmio_history_total = 0
        self.mmio_history_entries_discarded = 0
        self.mmio_read_total = 0
        self.mmio_write_total = 0
        self.mmio_read_occurrence_counts.clear()
        self.external_memory_read_occurrence_counts.clear()
        self.reg_mmio_map.clear()
        self.memory_sources.clear()
        self.call_stack.clear()
        self.completed_call_frames.clear()
        self.max_call_depth_seen = 0
        self.branch_dependency_snapshots.clear()
        self._preloaded_external_loads.clear()
        # B1：产生者 pending 槽与逐 pc 缓存随会话重置。
        self._pending_flag_snapshot = None
        # B3：FPSCR 槽同步重置。
        self._pending_fpscr_snapshot = None
        self._flag_generation = 0
        self._flag_writer_profile_cache.clear()
        self._vfp_flag_profile_cache.clear()
        for key in list(self.external_input_event_stats):
            self.external_input_event_stats[key] = 0
        self.provenance_kill_stats.clear()
        # r23 计量状态随会话重置（不跨 merge_from——事件表是单次重放口径）。
        self._lineage_last_write.clear()
        self._lineage_walk_cache.clear()
        self._lineage_note_context = None
        self.provenance_dead_operand_stats.clear()
        self.dynamic_solver_escalation_stats.clear()
        self.dynamic_checksum_fallback_stats.clear()
        self.dynamic_llm_fallback_stats.clear()
        self.differential_probe_stats.clear()
        self.dynamic_graph_archives.clear()
        self.dynamic_graph = self._new_dynamic_graph()
