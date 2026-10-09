#!/usr/bin/env python3
"""
智能仿真器 - 集成智能循环分类器

核心创新：
1. 不是简单的"PC重复>N次"检测
2. 而是理解循环的语义，分类处理
3. 这是真正智能的方法，区别于Fuzzware
"""

from __future__ import annotations

from unicorn import *
from unicorn.arm_const import *

# Keep an explicit module reference.  The star import above does not expose a
# reliable module object on all Unicorn binding versions, while the managed
# execution path needs the native error constants to classify a stopped
# invalid-memory callback.
import unicorn as _unicorn

# Unicorn 2.x 提供 UC_MODE_MCLASS（Cortex-M profile，见构造函数中的根因注释）；
# 旧版本缺失该常量时兜底为 0——按位或后不改变任何行为，安全降级回
# cortex-a15 + Cortex-M 系统指令软件 hook 模式。注意不能用
# getattr(unicorn, ...)：上面的 star import 会把名字 ``unicorn`` 覆盖成
# 子模块 unicorn.unicorn（那里没有该常量），因此用 try/except 导入兜底。
try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0
from collections import Counter
import copy
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from ..artifact_io import atomic_json_dump
from ..causal_context import CausalExecutionContext
from ..cpu_profile import (
    CORTEX_M_FALLBACK_CONSTANT,
    CpuProfile,
    count_thumb2_halfword_prefixes,
    is_armv6m_profile,
    resolve_cortex_m_cpu,
    unicorn_cpu_model_id,
)
from .intelligent_loop_classifier import IntelligentLoopClassifier, LoopType
from .intelligent_mmio_inferencer import IntelligentMMIOInferencer
from .lightweight_mmio_analysis import derive_mmio_seed_table
from .thumb_it_state import (
    IT_STATE_MASK,
    thumb_it_unpredictable_kind,
)
from .unique_bb_snapshot_manager import UniqueBBSnapshotManager
from .snapshot_memory import (
    SnapshotCaptureError,
    SnapshotMetadataStore,
    SnapshotPageStore,
    SnapshotBlobStore,
    SnapshotStateBlob,
)
from .llm_code_analyzer import LLMCodeAnalyzer
from .stateful_mmio_handler import StatefulMMIOHandler
from .simple_time_function_handler import SimpleTimeFunctionHandler
from .simple_wait_loop_handler import SimpleWaitLoopHandler
from .path_explorer import PathExplorer
from .branch_snapshot_manager import BranchSnapshotManager
from .deadlock_llm_solver import DeadlockLLMSolver
from ..hook_lifecycle import register_hook_owner, unregister_hook_owner
from ..evidence_contract import (
    configured_intervention_counts,
    coerce_bool,
    environment_input_facts,
    environment_model_diagnostics,
    execution_intervention_reasons,
    is_environment_fact_reason,
    snapshot_provenance_record,
)
try:
    from ..mmio_handler.mmio_hook_registry import (
        get_primary_mmio_handler,
        register_primary_mmio_handler,
        unregister_primary_mmio_handler,
    )
except ImportError:
    def get_primary_mmio_handler(uc):
        return None

    def register_primary_mmio_handler(uc, handler):
        return None

    def unregister_primary_mmio_handler(uc, handler=None):
        return None

try:
    from fuzzengine.crash_detector import CrashDetector, CrashDetectorConfig
    from fuzzengine.runtime_crash_monitor import RuntimeCrashMonitor
except Exception:
    CrashDetector = None
    CrashDetectorConfig = None
    RuntimeCrashMonitor = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_GLOBAL_UNPROVEN_DYNAMIC_CODE_TARGETS: Counter[int] = Counter()


class _DeferredMemoryMapping(BaseException):
    """Signal that a callback must be retried after a safe-point mapping.

    This deliberately derives from ``BaseException`` rather than ``Exception``.
    A number of legacy summary handlers have broad ``except Exception`` blocks;
    allowing this control-flow signal to be swallowed would make a summary
    appear to succeed while its memory side effect was never performed.
    Unicorn's Python callback wrapper catches ``BaseException``, stops the
    native execution, and re-raises the original object after ``emu_start``
    returns.  ``_managed_emu_start`` consumes it at that safe boundary.
    """

    def __init__(self, pages: Tuple[int, ...]):
        self.pages = tuple(int(page) & 0xFFFFFFFF for page in pages)
        super().__init__(
            "memory mapping deferred until the current Unicorn execution "
            f"returns ({len(self.pages)} page(s))"
        )


def _snapshot_field(snapshot: object, key: str, default: Any = None) -> Any:
    """Read serialized or in-memory snapshot fields uniformly."""
    if isinstance(snapshot, Mapping):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _safe_execution_provenance(owner: object) -> Dict[str, object]:
    """Read execution lineage without requiring every legacy adapter to expose it.

    Snapshot providers are also used by small test doubles and by older
    out-of-tree adapters.  Missing lineage is an evidence gap, not a reason to
    abort snapshot capture; importantly, the fallback is explicitly
    unverified so it cannot promote a replay to validated coverage.
    """
    reader = getattr(owner, "_current_execution_provenance", None)
    if callable(reader):
        try:
            value = reader()
            if isinstance(value, Mapping):
                return dict(value)
            return {
                "status": "unverified",
                "reasons": ["execution_provenance_returned_non_mapping"],
                "telemetry_complete": False,
                "provenance_finalized": False,
            }
        except Exception as exc:
            return {
                "status": "unverified",
                "reasons": [
                    f"execution_provenance_capture_failed:{type(exc).__name__}"
                ],
                "telemetry_complete": False,
                "provenance_finalized": False,
            }
    return {
        "status": "unverified",
        "reasons": ["execution_provenance_unavailable"],
        "telemetry_complete": False,
        "provenance_finalized": False,
    }

THUMB_CONDITION_CODES = {
    "EQ", "NE", "CS", "HS", "CC", "LO", "MI", "PL",
    "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE",
}


class IntelligentEmulator:
    """
    智能仿真器

    核心特性：
    1. 智能循环分类（区分初始化/轮询/延迟/死循环）
    2. 动态阈值（根据循环类型调整）
    3. 精确干预（针对性解决问题）
    4. 唯一BB快照（支持逐级回退）
    5. LLM代码分析（理解代码语义）
    """

    def __init__(self, firmware_path, mmio_constraints=None, static_bbs=None,
                 constraint_json_path=None, max_snapshots=5, iteration=0,
                 llm_config_path=None, branch_mmio_file_mode: Optional[str] = None,
                 raw_load_base: Optional[int] = None,
                 static_mmio_accesses: Optional[List[Dict[str, object]]] = None,
                 function_mmio_summaries: Optional[Dict[int, Dict[str, object]]] = None,
                 execution_thumb_override: Optional[bool] = None,
                 snapshot_page_store: Optional[SnapshotPageStore] = None,
                 snapshot_metadata_store: Optional[SnapshotMetadataStore] = None,
                 snapshot_blob_store: Optional[SnapshotBlobStore] = None):
        """
        初始化

        Args:
            firmware_path: 固件路径
            mmio_constraints: MMIO约束
            static_bbs: 静态BB信息 {address: [instructions]}
            constraint_json_path: 约束JSON文件路径
            max_snapshots: 最大快照数
            iteration: 迭代号
            llm_config_path: LLM配置文件路径
        """
        self.firmware_path = firmware_path
        self.static_bbs = static_bbs or {}
        self.constraint_json_path = constraint_json_path
        self.iteration = iteration
        self.llm_config_path = llm_config_path
        self.branch_mmio_file_mode = self._normalize_branch_mmio_file_mode(branch_mmio_file_mode)
        self.raw_load_base = int(raw_load_base) & 0xFFFFFFFF if raw_load_base is not None else None
        self.execution_thumb_override = (
            bool(execution_thumb_override)
            if execution_thumb_override is not None
            else None
        )
        self.static_mmio_accesses = list(static_mmio_accesses or [])
        self.function_mmio_summaries = dict(function_mmio_summaries or {})
        self.static_mmio_prediction_stats = self._build_static_mmio_prediction_stats()
        # 静态初始种子（设计 §2.3）：静态地址清单 + 立即数线索 → 首读初始值，
        # 无线索填 0。推导是静态输出的纯函数，保证同固件同配置必然一致。
        # 开关默认开启；种子只在该地址首次读取落到未建模路径时生效。
        self.enable_static_mmio_seeds = (
            os.environ.get("LSGEMU_STATIC_MMIO_SEEDS", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        if self.enable_static_mmio_seeds:
            seed_table, seed_meta = derive_mmio_seed_table(
                self.static_bbs,
                self.static_mmio_accesses,
            )
        else:
            seed_table, seed_meta = {}, {"read_records": 0, "from_immediate": 0, "default_zero": 0}
        self.mmio_seed_values = seed_table
        self.mmio_seed_derivation_meta = dict(seed_meta)

        logger.info("="*80)
        logger.info("智能仿真器初始化")
        logger.info("="*80)

        # 解析固件。历史路径默认只支持 ELF；manual MCU audit 中大量样本是
        # raw vendor BIN，必须保留 BINParser 给出的基址/入口/端序，否则完整仿真
        # 会在初始化阶段失败，无法进入候选验证。
        from .file_parser import Architecture, BINParser, ELFParser, Endianness
        self.is_elf_firmware = False
        with open(firmware_path, "rb") as f:
            magic = f.read(4)
        if magic == b"\x7fELF":
            parser = ELFParser(firmware_path)
            self.arch_info = parser.parse()
            self.is_elf_firmware = True
        else:
            parser = BINParser(firmware_path)
            self.arch_info = parser.parse_heuristic()
            if self.arch_info.architecture == Architecture.UNKNOWN:
                logger.warning("BIN启发式识别失败，使用默认 ARM little-endian base=0")
                self.arch_info = parser.parse_with_hint(
                    architecture=Architecture.ARM,
                    endianness=Endianness.LITTLE,
                    bits=32,
                    base_addr=0x00000000,
                )
            if not self.arch_info.code_size:
                self.arch_info.code_size = os.path.getsize(firmware_path)
        if self.arch_info.architecture is not Architecture.ARM:
            raise ValueError(
                "LSGEmu execution currently supports 32-bit ARM/Cortex-M only; "
                f"{firmware_path} reports architecture="
                f"{self.arch_info.architecture.value}, bits={self.arch_info.bits}, "
                f"machine={self.arch_info.machine_type} (0x{int(self.arch_info.machine_code):x}). "
                "Use a matching emulator backend or provide an ARM firmware image."
            )
        self.base_addr = self.arch_info.base_addr
        self.entry_point = self.arch_info.entry_point
        self._load_segment_cache: Optional[List[Dict[str, int]]] = None
        self.mapped_ranges: List[Tuple[int, int]] = []
        self.ram_snapshot_regions: List[Tuple[int, int]] = [(0x20000000, 0x100000)]
        self.vector_table_base = self.base_addr
        # 向量表检测诊断信息；仅在 _full_scan_vector_table_base 触发时填充
        # （fast path 命中时保持 None，报告输出 null，见 build_report）。
        self.vector_table_detection: Optional[Dict[str, object]] = None
        self.cortex_m_system_registers: Dict[str, int] = {}
        self.symbols_by_addr: Dict[int, str] = {}
        self._elf_all_symbols_by_name_cache: Optional[Dict[str, int]] = None
        self._zephyr_device_name_cache: Optional[Dict[str, List[Tuple[int, str]]]] = None

        # Unicorn实例。ELF/Cortex-M 固件通常以 Thumb 入口执行；raw ARM BIN
        # 经常从 ARM reset branch 开始，不能强制 entry|1。
        self.execution_thumb = self._infer_thumb_execution_mode()
        # Cortex-M profile（UC_MODE_MCLASS）：
        # 不加 MCLASS 时 unicorn 默认选 cortex-a15（cpu.c: 无 MCLASS 且无显式
        # cpu_model 时回退 A15），A15 无 ARM_FEATURE_M，qemu translate.c 的
        # trans_MSR_v7m/trans_MRS_v7m 直接 return false，导致 v7-M 系统指令
        # （msr msp/psp、mrs、cpsid/cpsie、MRC/MCR 访问 SCB 等）全部
        # UC_ERR_INSN_INVALID——实测 ardupilot_Pixhawk1(STM32F427) reset 第 3 条
        # 指令 `msr msp, r0` 即失败，baseline instruction_count 只有 2。加上
        # MCLASS 后 unicorn 把 CPU 切成 cortex-m33（cpu.c: mode & UC_MODE_MCLASS
        # -> UC_CPU_ARM_CORTEX_M33），MSR/MRS/CPSID 等由 m_helper.c 原生翻译执行
        # （fuzzware 同样以 THUMB|MCLASS 初始化）。实测该模式下 SP FPU
        # （vmov.f32/vadd.f32，FPv5-SP 兼容 FPv4-SP）可执行，仅 f64 双精度缺失，
        # 而 Cortex-M4F 固件（如 F427 的 fpv4-sp-d16）本身不含 f64 指令。
        # 兜底：旧 unicorn 无该常量时 getattr 取 0（位标志不生效，行为不变）；
        # LSGEMU_DISABLE_MCLASS=1 可显式回退旧的 A15+软件 hook 模式用于对比；
        # 仅 Thumb 固件启用——cortex-m33 只支持 Thumb，A32（execution_thumb=False
        # 的 raw ARM BIN）加 MCLASS 会立刻 INSN_INVALID。
        self.cortex_m_native_mclass_enabled = bool(
            UC_MODE_MCLASS
            and self.execution_thumb
            and os.environ.get("LSGEMU_DISABLE_MCLASS", "").strip().lower()
            not in {"1", "true", "yes", "on"}
        )
        unicorn_mode = UC_MODE_THUMB if self.execution_thumb else UC_MODE_ARM
        if self.cortex_m_native_mclass_enabled:
            unicorn_mode |= UC_MODE_MCLASS
        if getattr(self.arch_info, "endianness", None) == Endianness.BIG:
            unicorn_mode |= UC_MODE_BIG_ENDIAN
        self.uc = Uc(UC_ARCH_ARM, unicorn_mode)
        # Cortex-M CPU 型号自动匹配：MCLASS 模式下 unicorn 默认 cortex-m33，
        # 这里按 ELF e_flags/固件名推断实际型号（F427→M4、H743→M7 等）并
        # ctl_set_cpu_model 精确切换；推断/切换失败一律回退 M33 默认并告警。
        # 非 MCLASS 路径（A15+软件 hook、raw ARM BIN）保持 None，报告输出 null。
        self.cortex_m_cpu_profile: Optional[CpuProfile] = None
        if self.cortex_m_native_mclass_enabled:
            self.cortex_m_cpu_profile = self._apply_cortex_m_cpu_profile()
        self._owned_hooks: List[object] = []
        self._owned_hook_records: Dict[object, Dict[str, int]] = {}
        self._pending_owned_hook_removals: List[object] = []
        self._native_emulation_depth = 0
        self._native_emulation_thread_id: Optional[int] = None
        self._native_emulation_state_lock = threading.RLock()
        self._close_requested = False
        self._close_completed = False
        self.hook_lifecycle_stats: Counter[str] = Counter()
        self._closed_uc = None
        # Unicorn/QEMU owns pointers into its translated address-space
        # topology while ``emu_start`` is active.  Runtime summaries and
        # invalid-memory callbacks may discover a new page, but the actual
        # topology mutation must happen after the native call has unwound.
        self._pending_memory_map_pages: Dict[int, int] = {}
        self._pending_memory_map_origins: Set[str] = set()
        self._pending_memory_map_external_addresses: Set[int] = set()
        self._pending_memory_map_write_replays: List[Dict[str, int]] = []
        self._pending_memory_map_retries_block_callback = False
        self._deferred_retry_skip_block_pc: Optional[int] = None
        self._deferred_retry_reenter_block_pc: Optional[int] = None
        self._deferred_write_retry_hooks: List[object] = []
        self.memory_mapping_stats: Counter[str] = Counter()
        self.last_memory_mapping: Dict[str, object] = {
            "status": "idle",
            "requested_pages": [],
            "mapped_pages": [],
            "error": None,
        }
        self.execution_preflight_stats: Counter[str] = Counter()
        self.last_execution_preflight: Dict[str, object] = {}

        # 1. 智能循环分类器（核心创新）
        self.loop_classifier = IntelligentLoopClassifier(static_bbs=static_bbs)

        # 2. 唯一BB快照管理器
        # PreparedFirmware shares one immutable page store across the primary
        # and all temporary replay emulators.  Snapshot ownership remains per
        # emulator, but identical RAM pages no longer occupy memory once per
        # replay engine.
        self.snapshot_page_store = snapshot_page_store or SnapshotPageStore()
        self.snapshot_metadata_store = (
            snapshot_metadata_store or SnapshotMetadataStore()
        )
        # External replay state is shared by the primary emulator and all
        # temporary replay emulators for this process/run.  The store is
        # append-only and immutable after capture, so sharing it cannot alter
        # snapshot identity or replay semantics.
        self.snapshot_blob_store = (
            snapshot_blob_store or SnapshotBlobStore.from_environment()
        )
        self.snapshot_capture_stats: Counter[str] = Counter()
        self.snapshot_manager = UniqueBBSnapshotManager(
            max_snapshots=max_snapshots,
            memory_regions=[(0x20000000, 0x100000)],  # 1MB RAM
            page_store=self.snapshot_page_store,
        )

        # 3. LLM代码分析器
        self.code_analyzer = LLMCodeAnalyzer(
            # ``None`` means disabled. Callers that want the project default
            # must resolve and pass that path explicitly; otherwise no-LLM
            # runs silently create a real client from the working directory.
            config_path=llm_config_path,
            static_bbs=static_bbs,
            thumb_mode=self.execution_thumb,
        )

        # 4. MMIO值推断器（新增）
        self.mmio_inferencer = IntelligentMMIOInferencer()
        self.instruction_to_bb = self._build_instruction_to_bb_lookup()
        self.branch_pc_to_bb = self._build_branch_pc_lookup()
        self.branch_entry_pc_to_bb = self._build_branch_entry_lookup()

        # 5. 状态化MMIO处理器
        self.memory_read_constraints: Dict[Tuple[int, int], int] = {}
        self.memory_occurrence_constraints: Dict[Tuple[int, int, int], int] = {}
        self.memory_occurrence_constraint_sites: Set[Tuple[int, int]] = set()
        self.memory_read_occurrence_counts: Counter[Tuple[int, int]] = Counter()
        self.memory_constraint_hit_counts: Counter[Tuple[int, int]] = Counter()
        self.memory_constraint_hit_history: List[Dict[str, object]] = []
        self.dynamic_memory_read_constraints: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.pending_persisted_memory_constraints: List[Dict[str, object]] = []
        self.modeled_async_memory_constraints: Set[Tuple[int, int]] = set()
        self.modeled_async_memory_stats = {
            "installed": 0,
            "applied_reads": 0,
        }
        self.status_loop_solver_stats = {
            "attempted": 0,
            "applied": 0,
            "rejected": 0,
        }
        self.memory_constraint_guard_stats = {
            "rejected_call_boundary": 0,
            "rejected_pointer_field": 0,
            "rejected_other": 0,
        }
        self.memory_constraint_guard_history: List[Dict[str, object]] = []
        self.last_unmapped_access: Optional[Dict[str, object]] = None
        self.external_memory_input_addresses: Set[int] = set()
        self.memory_access_history: List[Tuple[int, int, bool, int]] = []
        try:
            self.memory_access_history_limit = max(
                1024,
                int(os.environ.get("LSGEMU_MEMORY_ACCESS_HISTORY_LIMIT", "16384")),
            )
        except ValueError:
            self.memory_access_history_limit = 16384
        self.memory_access_history_total = 0
        self.memory_access_history_entries_discarded = 0
        self.enable_dynamic_memory_constraints = (
            os.environ.get("LSGEMU_DYNAMIC_MEMORY_CONSTRAINTS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.allow_external_memory_constraints = (
            os.environ.get("LSGEMU_EXTERNAL_MEMORY_CONSTRAINTS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.local_self_loop_before_wait = (
            os.environ.get("LSGEMU_LOCAL_SELF_LOOP_BEFORE_WAIT", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        static_constraints_dict = self._convert_to_static_dict(mmio_constraints)
        static_constraints_dict.update(self._load_constraints_from_json(constraint_json_path))
        self.causal_context = CausalExecutionContext()
        self.mmio_handler = StatefulMMIOHandler(
            static_constraints_dict,
            constraint_json_path,
            branch_mmio_file_mode=self.branch_mmio_file_mode,
            causal_context=self.causal_context,
            mmio_seed_values=dict(self.mmio_seed_values),
        )

        # 6. 简单时间函数处理器（新增）
        self.time_handler = SimpleTimeFunctionHandler()

        # 7. 简单等待循环处理器（新增）
        self.wait_loop_handler = SimpleWaitLoopHandler()

        # 8. 死循环LLM求解器（新增）
        if hasattr(self.code_analyzer, 'client') and self.code_analyzer.client:
            self.deadlock_solver = DeadlockLLMSolver(
                self.code_analyzer.client,
                self.code_analyzer.llm_model if hasattr(self.code_analyzer, 'llm_model') else 'gpt-4',
                static_bbs
            )
        else:
            self.deadlock_solver = None

        # 9. 路径探索器（新增 - 分支变异功能）
        self.path_explorer = None  # 延迟初始化，在第一次运行后创建

        # Causal input provenance.  Events are recorded at the external read
        # boundary, while a bounded set of pre-read checkpoints preserves the
        # state from which an alternative environment value can be replayed.
        self.causal_input_events: List[Dict[str, object]] = []
        self.causal_input_event_count = 0
        # Logical event IDs rewind with a restored input prefix.  Trace IDs do
        # not: they distinguish repeated attempts that revisit the same prefix.
        self.causal_input_trace_sequence = 0
        self.input_occurrence_counts: Dict[Tuple[int, int], int] = {}
        self.causal_input_snapshots: List[Dict[str, object]] = []
        self.causal_input_snapshot_site_counts: Counter[Tuple[str, int, int]] = Counter()
        self.causal_input_retention_stats: Counter[str] = Counter()
        try:
            self.max_causal_input_events = max(
                1,
                int(os.environ.get("LSGEMU_CAUSAL_INPUT_EVENT_LIMIT", "8192")),
            )
        except ValueError:
            self.max_causal_input_events = 8192
        try:
            self.max_causal_input_snapshots = max(
                0,
                int(os.environ.get("LSGEMU_CAUSAL_INPUT_SNAPSHOT_LIMIT", "64")),
            )
        except ValueError:
            self.max_causal_input_snapshots = 64
        try:
            self.max_causal_input_snapshots_per_site = max(
                1,
                int(os.environ.get("LSGEMU_CAUSAL_INPUT_SNAPSHOTS_PER_SITE", "2")),
            )
        except ValueError:
            self.max_causal_input_snapshots_per_site = 2
        self.causal_input_sequence_checkpoints = (
            os.environ.get("LSGEMU_CAUSAL_INPUT_SEQUENCE_CHECKPOINTS", "1")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        # Tracers subscribe to completed environment reads instead of relying
        # on a raw Unicorn memory hook.  This is necessary because mapped MMIO
        # loads may be serviced by the preload hook before Unicorn executes an
        # LDR instruction at all.
        self.external_input_observers: List[
            Callable[[Dict[str, object]], None]
        ] = []
        self.external_input_observer_stats = {
            "registered": 0,
            "notifications": 0,
            "failures": 0,
        }

        # 10. 分支快照管理器（新增 - 在分支点保存快照）
        self.branch_snapshot_manager = BranchSnapshotManager(
            page_store=self.snapshot_page_store,
            metadata_store=self.snapshot_metadata_store,
            blob_store=self.snapshot_blob_store,
        )
        self.branch_snapshot_manager.set_dirty_page_provider(
            lambda: set(getattr(self, "runtime_written_pages", set()) or set())
        )
        self.branch_snapshot_manager.set_external_state_provider(
            self._snapshot_external_state
        )
        self.enable_branch_snapshot = False  # 默认关闭，在基准运行时开启
        self.forced_branch_choices: Dict[object, bool] = {}
        self.forced_branch_sequence: List[Tuple[int, object]] = []
        self.forced_branch_sequence_index = 0
        self.forced_branch_hits: Set[object] = set()
        self.forced_branch_trace: List[Dict[str, int]] = []
        self.forced_branch_code_hook = None
        # r38→r40：forced_branch 总开关（缺省 1 = 零 force；=0 显式恢复诊断臂）。
        # 缓存值由 _refresh_forced_branch_disabled_flag 在构造/setter/run() 入口刷新；
        # 钩子内只读缓存，避免每条指令查 os.environ。
        self.forced_branch_disabled_blocks: Dict[str, int] = {
            "setter": 0,
            "hook": 0,
        }
        self._refresh_forced_branch_disabled_flag()
        self.branch_entry_code_hook = None
        self.runtime_loop_branch_force_hook = None
        self.runtime_loop_branch_forces: Dict[int, Dict[str, object]] = {}
        self.runtime_loop_branch_force_stats = {
            "installed": 0,
            "applied": 0,
            "failed": 0,
        }
        # r7 口径 2：循环体快进仿真（跳过循环体指令、直接改内存）按「干预」
        # 处理，单列计数名 loop_fast_forward_emulation（by_handler 细分）。
        self.loop_fast_forward_emulation: Dict[str, object] = {
            "total": 0,
            "by_handler": {},
        }
        # r15 D1：快转逐条事件（(loop_head, handler)，封顶 64 条）——翻转重放
        # 证据负载据此记录快转发生位点，便于审计「哪些环被 O(1) 物化」。
        self.loop_fast_forward_events: List[Tuple[int, str]] = []
        # r9 用户裁定（docs/DECISIONS_ledger_reclass_20260920.md）：干预事件按
        # 家族单独记账——loop_fast_forward_emulation / loop_mmio_adjust /
        # loop_wait_handled 属工程优化或外部输入（不入干预集合），loop_intervention
        # （本地约束/LLM/回退等）与 loop_unresolved_limit 保持 diagnostic。
        # 每次 intervention_count 自增恰对应一个家族标签，残差为 0。
        self.intervention_event_labels: Counter[str] = Counter()
        self.loop_unresolved_limit_trips = 0
        self.cortex_m_system_instruction_map = self._build_cortex_m_system_instruction_map()
        self.svc_instruction_pcs = self._build_svc_instruction_pcs()
        self.thumb_indirect_branch_instruction_map = self._build_thumb_indirect_branch_instruction_map()
        self.mapped_mmio_load_instruction_map = self._build_mapped_mmio_load_instruction_map()
        self.mapped_mmio_preload_stats = {
            "checked": 0,
            "applied": 0,
            "alias_applied": 0,
            "failed": 0,
            "indexed_checked": 0,
        }
        self.thumb_indirect_branch_repair_stats = {
            "checked": 0,
            "repaired": 0,
            "skipped_non_executable": 0,
            "failed": 0,
        }
        self.thumb_state_guard_stats = {
            "checked": 0,
            "repaired": 0,
            "failed": 0,
        }
        # r29: 残留脏 ITSTATE 守卫（站点无关）。详见 analysis/thumb_it_state.py。
        # 判定条件只看「xPSR 的 IT 位非 0 ∧ 当前指令属于 IT 块内为 UNPREDICTABLE
        # 的无条件族（B/B.W/BX/BLX/POP{pc}）」，不依赖任何地址白名单。
        self.it_state_guard_enabled = (
            os.environ.get("LSGEMU_IT_STATE_GUARD", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.it_state_guard_stats = {
            "checked": 0,
            "trips": 0,
            "repaired": 0,
            "failed": 0,
            "reentries": 0,
        }
        self._it_state_guard_pending = False
        self._it_state_guard_sites: Counter[int] = Counter()
        self._it_state_guard_kinds: Counter[str] = Counter()
        # 重入上限：单次 _managed_emu_start 调用内最多打断多少次，防止病态循环。
        try:
            self.it_state_guard_reentry_limit = max(
                1, int(os.environ.get("LSGEMU_IT_STATE_GUARD_LIMIT", "200000"))
            )
        except ValueError:
            self.it_state_guard_reentry_limit = 200000
        self.invalid_thumb_recovery_stats = {
            "attempted": 0,
            "recovered": 0,
            "failed": 0,
        }
        self.handle_svc_as_noop = (
            os.environ.get("LSGEMU_HANDLE_SVC_AS_NOOP", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.svc_stats = {
            "handled": 0,
            "by_number": {},
        }
        self.terminal_self_loop_bbs = self._build_terminal_self_loop_bbs()
        self.forced_branch_encounters: Dict[int, int] = {}
        self.forced_branch_pending: Dict[int, Dict[str, int]] = {}
        self.forced_branch_target_bbs: Set[int] = set()
        self.enable_loop_intervention = True
        self.semantic_obligation_enabled = True

        # 统计
        self.bb_addr_set = set()
        self.instruction_count = 0
        self.mmio_access_history = []
        try:
            self.mmio_access_history_limit = max(
                1024,
                int(os.environ.get("LSGEMU_MMIO_ACCESS_HISTORY_LIMIT", "16384")),
            )
        except ValueError:
            self.mmio_access_history_limit = 16384
        self.mmio_access_history_total = 0
        self.mmio_access_history_entries_discarded = 0
        self.latest_mmio_read_by_address: Dict[int, Tuple[int, int]] = {}
        self.bb_history = []
        try:
            self.bb_history_limit = max(
                1024,
                int(os.environ.get("LSGEMU_EMULATOR_BB_HISTORY_LIMIT", "8192")),
            )
        except ValueError:
            self.bb_history_limit = 8192
        self.bb_history_total = 0
        self.bb_history_entries_discarded = 0
        self.runtime_successor_edges: Set[Tuple[int, int]] = set()
        self.bb_visit_counts: Counter[int] = Counter()
        self.internal_unicorn_bb_fragments = 0
        self.mapped_write_hook_enabled = (
            os.environ.get("LSGEMU_RECORD_MAPPED_WRITES", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.mapped_memory_write_stats = {
            "sram_writes": 0,
            "mmio_writes": 0,
            "bitband_alias_writes": 0,
            "other_writes": 0,
        }
        self.dynamic_static_bb_enabled = (
            os.environ.get("LSGEMU_DYNAMIC_STATIC_BBS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        try:
            self.dynamic_static_bb_max = max(0, int(os.environ.get("LSGEMU_DYNAMIC_STATIC_BB_MAX", "20000")))
        except ValueError:
            self.dynamic_static_bb_max = 20000
        self.dynamic_static_bbs_added = 0
        self.dynamic_static_bb_rejections = 0
        self.dynamic_static_bb_starts: Set[int] = set()
        self.dynamic_static_bb_split_count = 0
        self.runtime_written_pages: Set[int] = set()
        self.unproven_dynamic_code_targets: Counter[int] = Counter()
        self.external_rom_call_targets: Counter[int] = Counter()
        self.external_rom_call_stats = {
            "attempted": 0,
            "returned": 0,
            "failed": 0,
        }
        self._external_rom_summary_target: Optional[int] = None
        self.summarize_external_rom_calls = (
            os.environ.get("LSGEMU_SUMMARIZE_EXTERNAL_ROM_CALLS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.stop_on_unproven_dynamic_code = (
            os.environ.get("LSGEMU_STOP_ON_UNPROVEN_DYNAMIC_CODE", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.polling_value_attempts: Dict[Tuple[int, int], Set[int]] = {}
        self.polling_value_exhausted: Set[Tuple[int, int]] = set()
        self.record_mapped_data_accesses = (
            os.environ.get("LSGEMU_RECORD_MAPPED_DATA_ACCESSES", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.stop_after_no_new_bbs: Optional[int] = None
        self._no_new_bb_run = 0
        # r35 T1：判停误报豁免。当连续无新增覆盖面时，若当前 BB 恰好是某个
        # 已分类 INITIALIZATION/DELAY 循环（有退出条件 + 计数/寄存器变化）的
        # **出口 fallthrough**（环头条件分支的落空后继，且不在环体内），静默
        # 收口会停在「环已跑完、正准备返回」的 BB 上——那是误报。开关给出一次
        # 有界的宽限：再执行 N 个 BB 才允许 quiescence 收口；期间一旦出现新增
        # 覆盖（_no_new_bb_run 归零）宽限立即作废。默认 0 = 关闭 = canonical。
        try:
            self.quiescence_loop_exit_grace_budget = max(
                0,
                int(os.environ.get("LSGEMU_QUIESCENCE_LOOP_EXIT_GRACE_BB", "0")),
            )
        except ValueError:
            self.quiescence_loop_exit_grace_budget = 0
        self._quiescence_exit_grace_remaining = 0
        self.quiescence_loop_exit_grace_stats: Counter = Counter()
        # r35 T2：指针即计数器 store 循环快转族的授予/拒绝分账，用于 A/B 与
        # 负对照取证（granted / non_terminating / mmio_target）。
        self.pointer_limit_store_ff_declines: Counter = Counter()
        self.stop_requested_reason: Optional[str] = None
        self.stop_before_pc: Optional[int] = None
        # r31 interrupt delivery (default off; the controller installs itself).
        # `irq_delivery_boundary` is bumped once on interrupt entry and once on
        # interrupt return so the branch/coverage bookkeeping can tell a real
        # control-flow edge from an asynchronous hijack.
        self.irq_delivery_enabled = False
        self.irq_delivery_boundary = 0
        self._irq_delivery_boundary_seen = 0
        self.irq_delivery_controller = None
        # No-progress watchdog: independent of fatal_sink_terminal, and only
        # armed while terminal/hot-loop stops are being deferred for a pending
        # hardware event.  Bounds the r30 observation of 20,002 idle spins.
        try:
            self.irq_delivery_watchdog_limit = max(
                0,
                int(os.environ.get("LSGEMU_IRQ_DELIVERY_WATCHDOG", "20000")),
            )
        except ValueError:
            self.irq_delivery_watchdog_limit = 20000
        self.irq_delivery_watchdog_spins = 0
        self.irq_delivery_deferral_stats: Counter = Counter()
        self._irq_symbol_by_addr: Optional[Dict[int, str]] = None
        self._irq_symbol_starts_cache: Optional[List[int]] = None
        self._irq_sink_symbol_cache: Dict[int, str] = {}
        self.skip_function_returns: Dict[int, Dict[str, object]] = {}
        self.skip_function_state: Dict[int, Dict[str, int]] = {}
        self.skipped_function_entry_bbs: Set[int] = set()
        self.skip_function_stats = {
            "installed": 0,
            "applied": 0,
            "stateful_applied": 0,
            "by_symbol": {},
            "applied_entry_bbs": [],
        }
        self.enable_lzo_decompress_summary = (
            os.environ.get("LSGEMU_LZO_DECOMPRESS_SUMMARY", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.lzo_decompress_summary_stats = {
            "applied": 0,
            "failed": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "entry_bbs": [],
        }
        try:
            self.model_heap_next = int(os.environ.get("LSGEMU_MODEL_HEAP_BASE", "0x20070000"), 0)
        except ValueError:
            self.model_heap_next = 0x20070000
        try:
            self.model_heap_end = int(os.environ.get("LSGEMU_MODEL_HEAP_END", "0x20100000"), 0)
        except ValueError:
            self.model_heap_end = 0x20100000
        self.model_heap_allocations: Dict[int, int] = {}
        self.stream_input_state: Dict[str, int] = {}
        self.stream_input_default_bytes = self._load_stream_input_seed()
        self.stream_input_summary_events: List[Dict[str, object]] = []
        self.stream_input_payload_writes: List[Dict[str, object]] = []
        # Stream APIs are environment boundaries.  Keep typed, cumulative
        # telemetry so a per-run delta can distinguish a real input delivery
        # from an unrelated function summary without changing execution.
        self.environment_input_delivery_stats = {
            "observed": 0,
            "accepted": 0,
            "rejected": 0,
        }
        self.mmio_handler.set_stream_input_provider(
            self._next_stream_input_byte,
            event_recorder=self._record_stream_input_summary_event,
        )
        self.watch_memory_ranges = self._load_watch_memory_ranges()
        self.watch_memory_events: List[Dict[str, object]] = []
        self.watch_pcs = self._load_watch_pcs()
        self.watch_pc_events: List[Dict[str, object]] = []
        self.last_run_result: Dict[str, object] = {}
        self.enable_crash_triage = (
            os.environ.get("LSGEMU_CRASH_TRIAGE", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.enable_runtime_crash_monitor = (
            os.environ.get("LSGEMU_RUNTIME_CRASH_MONITOR", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.runtime_crash_monitor_invalid_memory = (
            os.environ.get("LSGEMU_RUNTIME_CRASH_MONITOR_INVALID_MEMORY", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.crash_detector = None
        self.crash_detector_config = None

        # 干预统计
        self.intervention_count = 0
        self.intervention_by_type = {
            LoopType.INITIALIZATION: 0,
            LoopType.POLLING: 0,
            LoopType.MEMORY_WAIT: 0,
            LoopType.DELAY: 0,
            LoopType.DEADLOCK: 0,
            LoopType.UNKNOWN: 0
        }
        # Execution-evidence lineage is kept separate from the emulator's
        # cumulative diagnostic counters.  Replay stages often reuse an
        # emulator object, so each run records a delta from this baseline.
        self._execution_sequence = 0
        self.execution_id = ""
        self._execution_counter_baseline: Dict[str, object] = {}
        self._active_prefix_provenance: Dict[str, object] = {}
        # P0-D（cycle3 k.5 C3）：blob identity 遥测。era = 快照恢复代数
        # （每次 ``restore_snapshot_external_state`` 调用 +1）；出现数按
        # ``(era, bb)`` 限定键累计——跨恢复不混计（现库 identity 塌缩最大簇
        # 3070 的机械根因正是计数跨域混并）。纯遥测，零执行语义。
        self._snapshot_identity_era: int = 0
        self._snapshot_identity_seq: int = 0
        self._snapshot_identity_occurrences: Dict[Tuple[int, int], int] = {}
        self._root_occurrence_local: Optional[int] = None
        self._execution_active = False
        self.last_snapshot_restore: Dict[str, object] = {
            "success": True,
            "errors": [],
        }
        self.last_snapshot_restore_errors: List[str] = []
        self.last_replay_mmio_state_restore: Dict[str, object] = {
            "success": True,
            "errors": [],
            "restored_addresses": [],
            "mapping_skips": [],
            "occurrence_sites": 0,
        }
        # A failed replay-state restore may leave native memory or model state
        # only partially restored.  Such an engine must be disposed rather
        # than reused for another Unicorn execution.
        self.replay_state_poisoned = False
        self.last_intervention_iterations = {}
        self.loop_intervention_failures: Dict[int, int] = {}
        self.loop_exit_iteration_hints: Dict[int, int] = {}
        self.loop_intervention_threshold_cache: Dict[int, int] = {}
        try:
            self.max_unresolved_loop_interventions = max(
                1,
                int(os.environ.get("LSGEMU_MAX_UNRESOLVED_LOOP_INTERVENTIONS", "3")),
            )
        except ValueError:
            self.max_unresolved_loop_interventions = 3
        try:
            self.observed_loop_exit_margin = max(
                0,
                int(os.environ.get("LSGEMU_OBSERVED_LOOP_EXIT_MARGIN", "16")),
            )
        except ValueError:
            self.observed_loop_exit_margin = 16
        try:
            self.ram_self_loop_soft_threshold = max(
                1,
                int(os.environ.get("LSGEMU_RAM_SELF_LOOP_SOFT_THRESHOLD", "100")),
            )
        except ValueError:
            self.ram_self_loop_soft_threshold = 100
        try:
            self.simple_wait_retry_gap = max(
                1,
                int(os.environ.get("LSGEMU_SIMPLE_WAIT_RETRY_GAP", "1")),
            )
        except ValueError:
            self.simple_wait_retry_gap = 1
        try:
            self.hot_loop_halt_threshold = max(
                0,
                int(os.environ.get("LSGEMU_HOT_LOOP_HALT_THRESHOLD", "20000")),
            )
        except ValueError:
            self.hot_loop_halt_threshold = 20000
        try:
            self.hot_loop_min_no_new_bbs = max(
                0,
                int(os.environ.get("LSGEMU_HOT_LOOP_MIN_NO_NEW_BBS", "4096")),
            )
        except ValueError:
            self.hot_loop_min_no_new_bbs = 4096
        self.early_byte_copy_fast_forward = (
            os.environ.get("LSGEMU_EARLY_BYTE_COPY_FAST_FORWARD", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # r15 D1：force-free 翻转重放专用——在 enable_loop_intervention=False
        # 时仍允许「确定性循环快转族」（纯内存写 + 确定退出 + 不改控制流方向）。
        # 只由 HistoricalRunner._dfs_flip_replay_once 打开；不改变主跑行为。
        self.dfs_flip_deterministic_fast_forward = False
        self.last_deadlock_attempt: Dict[int, Tuple[int, bool]] = {}
        self.deadlock_failed_directions: Dict[int, Dict[int, Set[bool]]] = {}
        self.branch_snapshot_hotset: Set[int] = set()
        self.constraint_validation_history_limit = 512
        self.constraint_validation_history: List[Dict[str, object]] = []
        self.constraint_validation_stats = {
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
        }
        logger.info(f"✓ 固件: {firmware_path}")
        logger.info(f"✓ 基址: 0x{self.base_addr:08x}")
        logger.info(f"✓ 入口点: 0x{self.entry_point:08x}")
        logger.info(f"✓ 静态BB: {len(self.static_bbs)}")
        logger.info(f"✓ 智能循环分类器: 已启用")
        logger.info("="*80)

    def _crash_code_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        if getattr(self, "is_elf_firmware", False):
            for segment in self._load_segments():
                if not (int(segment.get('flags', 0)) & 0x1):
                    continue
                start = int(segment.get('vaddr', 0) or 0) & 0xFFFFFFFF
                end = start + max(
                    int(segment.get('filesz', 0) or 0),
                    int(segment.get('memsz', 0) or 0),
                    1,
                )
                if end > start:
                    ranges.append((start, end))
        else:
            try:
                file_size = os.path.getsize(self.firmware_path)
            except Exception:
                file_size = int(getattr(self.arch_info, "code_size", 0) or 0)
            if file_size > 0:
                ranges.append((int(self.base_addr) & 0xFFFFFFFF, (int(self.base_addr) + file_size) & 0xFFFFFFFF))
                if self.raw_load_base is not None:
                    ranges.append((int(self.raw_load_base) & 0xFFFFFFFF, (int(self.raw_load_base) + file_size) & 0xFFFFFFFF))
        if not ranges:
            code_size = int(getattr(self.arch_info, "code_size", 0) or 0)
            if code_size > 0:
                ranges.append((int(self.base_addr) & 0xFFFFFFFF, int(self.base_addr) + code_size))
        if not ranges and self.static_bbs:
            addresses = [int(addr) & ~1 for addr in self.static_bbs.keys()]
            if addresses:
                ranges.append((min(addresses), max(addresses) + 4))
        return self._merge_ranges([(start, end) for start, end in ranges if end > start])

    def _crash_ram_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for start, size in getattr(self, "ram_snapshot_regions", []) or []:
            start_i = int(start) & 0xFFFFFFFF
            size_i = max(1, int(size or 1))
            ranges.append((start_i, start_i + size_i))
        if not ranges:
            ranges = [
                (0x1FFF0000, 0x20000000),
                (0x20000000, 0x20100000),
                (0x10000000, 0x10040000),
            ]
        return self._merge_ranges(ranges)

    def _crash_fault_handler_addrs(self) -> Dict[str, int]:
        names = {
            "HardFault": 3,
            "MemManage": 4,
            "BusFault": 5,
            "UsageFault": 6,
            "SecureFault": 7,
        }
        out: Dict[str, int] = {}
        base = int(getattr(self, "vector_table_base", self.base_addr) or self.base_addr)
        for name, index in names.items():
            value = self._read_firmware_word(base + index * 4)
            if value is None:
                continue
            target = int(value) & ~1
            if target and self._is_executable_address(target):
                out[name] = target
        return out

    def _build_crash_detector_config(self):
        if CrashDetectorConfig is None:
            return None
        return CrashDetectorConfig(
            crash_points=set(),
            fault_handler_addrs=self._crash_fault_handler_addrs(),
            code_ranges=self._crash_code_ranges(),
            ram_ranges=self._crash_ram_ranges(),
            mmio_ranges=[(0x40000000, 0x60000000), (0xE0000000, 0xE0100000)],
            classify_timeouts_as_hangs=True,
            classify_max_instruction_as_hang=False,
            require_code_range_for_pc_crash=False,
            treat_unmapped_mmio_as_artifact=True,
            treat_unmapped_configured_memory_as_artifact=True,
        )

    def _get_crash_detector(self):
        if not self.enable_crash_triage or CrashDetector is None:
            return None
        if self.crash_detector is None:
            self.crash_detector_config = self._build_crash_detector_config()
            self.crash_detector = CrashDetector(self.crash_detector_config)
        return self.crash_detector

    def _install_runtime_crash_monitor(self):
        if not self.enable_runtime_crash_monitor or RuntimeCrashMonitor is None:
            return None
        config = self.crash_detector_config or self._build_crash_detector_config()
        if config is None:
            return None
        monitor = RuntimeCrashMonitor(
            self.uc,
            config,
            monitor_invalid_memory=self.runtime_crash_monitor_invalid_memory,
            stop_on_pc_outside_code=False,
            hook_add=self._add_owned_hook,
            hook_del=self._remove_owned_hook,
        )
        monitor.install()
        return monitor

    def _attach_crash_triage(self, run_result: Dict[str, object]) -> None:
        detector = self._get_crash_detector()
        if detector is None:
            return
        try:
            report = detector.detect(
                run_result,
                firmware=str(self.firmware_path),
                metadata={
                    "source_layer": str(run_result.get("source_layer") or "lsgemu_run_result"),
                    "emulator": "IntelligentEmulator",
                },
            )
            run_result["crash_triage"] = report.to_dict()
        except Exception as exc:
            run_result["crash_triage_error"] = str(exc)

    def _add_owned_hook(self, *args, **kwargs):
        """Register and account for a Unicorn hook owned by this emulator."""
        with self._native_state_lock():
            return self._add_owned_hook_locked(*args, **kwargs)

    def _add_owned_hook_locked(self, *args, **kwargs):
        """Add a hook while the lifecycle state lock is held."""
        uc = getattr(self, "uc", None)
        if uc is None or bool(getattr(self, "_close_requested", False)):
            raise RuntimeError("cannot register hook on a closed emulator")
        active_depth = int(getattr(self, "_native_emulation_depth", 0) or 0)
        active_thread = getattr(self, "_native_emulation_thread_id", None)
        if active_depth > 0 and active_thread != threading.get_ident():
            self.hook_lifecycle_stats["cross_thread_hook_add_rejected"] += 1
            raise RuntimeError(
                "cannot change Unicorn hooks from a different thread during emulation"
            )
        register_hook_owner(uc, self)
        hook = uc.hook_add(*args, **kwargs)
        self._owned_hooks.append(hook)
        try:
            hook_type = int(args[0]) if args else int(kwargs.get("htype", 0))
        except (TypeError, ValueError):
            hook_type = 0
        try:
            begin = int(kwargs.get("begin", args[3] if len(args) > 3 else 1))
        except (TypeError, ValueError):
            begin = 1
        try:
            end = int(kwargs.get("end", args[4] if len(args) > 4 else 0))
        except (TypeError, ValueError):
            end = 0
        try:
            self._owned_hook_records[hook] = {
                "type": hook_type,
                "begin": begin,
                "end": end,
            }
        except (TypeError, AttributeError):
            # Real Unicorn handles are integers.  Keep compatibility with
            # unusual test doubles that return an unhashable handle.
            pass
        self.hook_lifecycle_stats["added"] += 1
        return hook

    def _delete_owned_hook_now(self, hook):
        """Delete one hook after native emulation has fully returned."""
        with self._native_state_lock():
            self._delete_owned_hook_now_locked(hook)

    def _delete_owned_hook_now_locked(self, hook):
        """Delete one hook while the lifecycle state lock is held."""
        uc = getattr(self, "uc", None)
        if hook not in getattr(self, "_owned_hooks", []):
            try:
                self._pending_owned_hook_removals.remove(hook)
            except (ValueError, AttributeError):
                pass
            return
        try:
            if uc is not None:
                uc.hook_del(hook)
                self.hook_lifecycle_stats["deleted"] += 1
        except Exception:
            # Preserve the historical best-effort cleanup contract.  The
            # handle is removed from the Python ownership table even when
            # native cleanup reports that it was already gone.
            self.hook_lifecycle_stats["delete_errors"] += 1
        finally:
            try:
                self._owned_hooks.remove(hook)
            except (ValueError, AttributeError):
                pass
            try:
                self._owned_hook_records.pop(hook, None)
            except (KeyError, AttributeError, TypeError):
                pass
            try:
                while hook in self._pending_owned_hook_removals:
                    self._pending_owned_hook_removals.remove(hook)
            except (ValueError, AttributeError):
                pass

    def _drain_pending_owned_hook_removals(self):
        """Apply hook deletions queued by callbacks during emulation."""
        with self._native_state_lock():
            if getattr(self, "_native_emulation_depth", 0) > 0:
                return
            pending = list(getattr(self, "_pending_owned_hook_removals", []) or [])
            self._pending_owned_hook_removals = []
            for hook in pending:
                self._delete_owned_hook_now_locked(hook)

    def _native_state_lock(self):
        """Return the lifecycle lock, including for legacy test doubles."""
        lock = getattr(self, "_native_emulation_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._native_emulation_state_lock = lock
        return lock

    def _complete_deferred_close(self, uc=None) -> bool:
        """Finish a close requested from a native callback after it returns."""
        with self._native_state_lock():
            if bool(getattr(self, "_close_completed", False)):
                return True
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                return False
        current_uc = getattr(self, "uc", None)
        if uc is None:
            uc = current_uc
        if uc is None:
            self._discard_pending_memory_maps("close_without_unicorn")
            self._close_completed = True
            return True

        # ``_clear_owned_hooks`` is idempotent at depth zero.  Keep all native
        # hook deletion and registry removal in this one post-emu_start path.
        self._clear_owned_hooks(uc)
        self._discard_pending_memory_maps("close_completed")
        if getattr(self, "uc", None) is uc:
            self.uc = None
        self._closed_uc = uc
        with self._native_state_lock():
            self._close_completed = True
            self.hook_lifecycle_stats["close_completed"] += 1
        return True

    @contextmanager
    def _native_emulation_scope(self):
        """Keep hook topology stable until a native emulation call returns.

        Unicorn may execute a code or block callback from inside a translated
        block.  Deleting that callback from the same callback can leave the
        native translation loop observing a topology that is being mutated.
        Components therefore queue deletion while this scope is active; the
        queue is drained immediately after ``uc_emu_start`` unwinds.
        """
        current_thread_id = threading.get_ident()
        with self._native_state_lock():
            current_depth = int(getattr(self, "_native_emulation_depth", 0) or 0)
            owner_thread_id = getattr(self, "_native_emulation_thread_id", None)
            if current_depth > 0:
                if owner_thread_id != current_thread_id:
                    self.hook_lifecycle_stats["cross_thread_emu_start_rejected"] += 1
                    raise RuntimeError(
                        "concurrent emulation on one Unicorn engine is unsupported"
                    )
                if os.environ.get("LSGEMU_ALLOW_REENTRANT_EMU_START", "0").strip().lower() not in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    self.hook_lifecycle_stats["reentrant_emu_start_rejected"] += 1
                    raise RuntimeError(
                        "reentrant emulation on one Unicorn engine is unsupported"
                    )
            else:
                self._native_emulation_thread_id = current_thread_id

            self._native_emulation_depth = current_depth + 1
        try:
            yield
        finally:
            with self._native_state_lock():
                self._native_emulation_depth = max(
                    0,
                    int(getattr(self, "_native_emulation_depth", 1) or 1) - 1,
                )
                outermost = self._native_emulation_depth == 0
                if outermost:
                    self._native_emulation_thread_id = None
            if outermost:
                self._drain_pending_owned_hook_removals()
                if bool(getattr(self, "_close_requested", False)):
                    self._complete_deferred_close()

    def _remember_mapped_range(self, start: int, end: int) -> None:
        """Update the Python-side map index after a successful native map."""
        if int(end) <= int(start):
            return
        try:
            current = list(getattr(self, "mapped_ranges", []) or [])
            self.mapped_ranges = self._merge_ranges(
                current + [(int(start), int(end))]
            )
        except Exception:
            # Mapping correctness is still checked against Unicorn.  The
            # index is only a fast path and must not turn a bookkeeping issue
            # into an execution failure.
            pass

    def _pending_memory_map_context_snapshot(
        self,
    ) -> Tuple[Dict[int, int], Set[str], bool]:
        """Read queued pages and callback origins as one lifecycle snapshot."""
        with self._native_state_lock():
            return (
                dict(getattr(self, "_pending_memory_map_pages", {}) or {}),
                set(getattr(self, "_pending_memory_map_origins", set()) or set()),
                bool(
                    getattr(
                        self,
                        "_pending_memory_map_retries_block_callback",
                        False,
                    )
                ),
            )

    def _queue_deferred_memory_mapping(
        self,
        pages: Set[int],
        *,
        permissions: Optional[int] = None,
        origin: str = "unknown",
        retry_block_callback: bool = False,
    ) -> None:
        """Queue pages and abort the current native slice for a safe retry."""
        normalized = tuple(sorted({int(page) & 0xFFFFFFFF for page in pages}))
        if not normalized:
            return

        origin, retry_block_callback = self._infer_mapping_callback_context(
            origin,
            retry_block_callback,
        )

        current_thread_id = threading.get_ident()
        with self._native_state_lock():
            depth = int(getattr(self, "_native_emulation_depth", 0) or 0)
            owner_thread_id = getattr(self, "_native_emulation_thread_id", None)
            if depth <= 0:
                raise RuntimeError(
                    "deferred memory mapping requested outside native execution"
                )
            if owner_thread_id != current_thread_id:
                self.memory_mapping_stats["cross_thread_requests_rejected"] += 1
                raise RuntimeError(
                    "memory mapping requested by a non-owner thread during emulation"
                )

            pending = getattr(self, "_pending_memory_map_pages", None)
            if not isinstance(pending, dict):
                pending = {}
                self._pending_memory_map_pages = pending
            added = 0
            requested_permissions = int(permissions or 0)
            for page in normalized:
                if page not in pending:
                    pending[page] = requested_permissions
                    added += 1
                else:
                    # A page may be requested first by the default mapper and
                    # later by an auxiliary component with explicit writable
                    # permissions. Zero means Unicorn's default UC_PROT_ALL,
                    # so it dominates any explicit subset; otherwise combine
                    # all requested permission bits.
                    existing_permissions = int(pending.get(page, 0) or 0)
                    pending[page] = (
                        0
                        if existing_permissions == 0 or requested_permissions == 0
                        else existing_permissions | requested_permissions
                    )
            self.memory_mapping_stats["deferred_requests"] += 1
            self.memory_mapping_stats["deferred_pages"] += added
            self.memory_mapping_stats["duplicate_page_requests"] += (
                len(normalized) - added
            )
            self.last_memory_mapping = {
                "status": "deferred",
                "requested_pages": self._bounded_page_sample(list(normalized)),
                "mapped_pages": [],
                "error": None,
            }
            origin = str(origin or "unknown")
            origins = getattr(self, "_pending_memory_map_origins", None)
            if not isinstance(origins, set):
                origins = set()
                self._pending_memory_map_origins = origins
            origins.add(str(origin))
            self._pending_memory_map_retries_block_callback = bool(
                getattr(
                    self,
                    "_pending_memory_map_retries_block_callback",
                    False,
                )
                or retry_block_callback
            )
            self.memory_mapping_stats[f"deferred_origin_{origin}"] += 1

        # Stop explicitly before raising.  Recent bindings propagate the
        # BaseException through their callback wrapper; older bindings report
        # ``Exception ignored`` from ctypes and return from emu_start instead.
        # Calling emu_stop covers both behaviours without mutating the native
        # address-space topology from the callback.
        try:
            stop = getattr(getattr(self, "uc", None), "emu_stop", None)
            if callable(stop):
                stop()
                self.memory_mapping_stats["callback_stops"] += 1
        except Exception as exc:
            self.memory_mapping_stats["callback_stop_failures"] += 1
            logger.debug("延迟映射时停止Unicorn失败: %s", exc)
        raise _DeferredMemoryMapping(normalized)

    @staticmethod
    def _infer_mapping_callback_context(
        fallback_origin: str,
        retry_block_callback: bool,
    ) -> Tuple[str, bool]:
        """Classify the active hook only on the rare missing-page path."""
        callback_origins = {
            "bb_hook": ("block", True),
            "_skip_function_return_hook": ("code", False),
            "_mapped_mmio_preload_hook": ("code", False),
            "mmio_read_hook": ("memory", False),
            "mem_write_hook": ("memory", False),
            "mmio_unmapped_hook": ("invalid_memory", False),
        }
        try:
            frame = sys._getframe(2)
        except (AttributeError, ValueError):
            frame = None
        while frame is not None:
            classified = callback_origins.get(frame.f_code.co_name)
            if classified is not None:
                return classified
            frame = frame.f_back
        return str(fallback_origin or "unknown"), bool(retry_block_callback)

    @staticmethod
    def _bounded_page_sample(pages: List[int], limit: int = 256) -> List[str]:
        """Serialize a bounded page sample without retaining huge requests."""
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 256
        ordered = sorted({int(page) & 0xFFFFFFFF for page in pages})
        sample = ordered[:limit]
        result = [f"0x{page:08x}" for page in sample]
        if len(ordered) > limit:
            result.append(f"...(+{len(ordered) - limit} pages)")
        return result

    def _native_range_is_mapped(self, start: int, size: int) -> bool:
        """Query Unicorn directly, bypassing the Python fast-path index."""
        start = int(start)
        size = max(1, int(size))
        end = start + size
        try:
            regions = sorted(
                (int(region_start), int(region_end) + 1)
                for region_start, region_end, _perms in self.uc.mem_regions()
            )
        except Exception:
            return False
        cursor = start
        for region_start, region_end in regions:
            if region_end <= cursor:
                continue
            if region_start > cursor:
                return False
            cursor = max(cursor, region_end)
            if cursor >= end:
                return True
        return False

    def _map_memory_pages_now(self, pages: Mapping[int, int] | Set[int] | List[int]) -> bool:
        """Map queued pages after ``emu_start`` has completely unwound."""
        page_items = pages.items() if isinstance(pages, Mapping) else ((page, 0) for page in pages)
        normalized: Dict[int, int] = {
            int(page) & 0xFFFFFFFF: int(perms or 0)
            for page, perms in page_items
        }
        if not normalized:
            return True

        with self._native_state_lock():
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                raise RuntimeError("cannot drain memory mappings during native execution")

        missing: Dict[int, int] = {}
        already_mapped: List[int] = []
        for page, perms in sorted(normalized.items()):
            if self._native_range_is_mapped(page, 1):
                self._remember_mapped_range(page, page + 0x1000)
                already_mapped.append(page)
                continue
            missing[page] = int(perms or 0)

        # Group adjacent pages with the same permission request.  This keeps
        # the safe-point mechanism cheap for large buffers and avoids making
        # QEMU rebuild its address-space topology once per 4-KiB page.
        runs: List[Tuple[int, int, int]] = []
        for page in sorted(missing):
            perms = int(missing[page] or 0)
            if runs:
                run_start, run_end, run_perms = runs[-1]
                if page == run_end and perms == run_perms:
                    runs[-1] = (run_start, page + 0x1000, run_perms)
                    continue
            runs.append((page, page + 0x1000, perms))

        mapped_pages: List[int] = list(already_mapped)
        report_limit = os.environ.get("LSGEMU_MEMORY_MAP_REPORT_PAGE_LIMIT", "256")
        try:
            report_limit_int = max(1, int(report_limit))
        except (TypeError, ValueError):
            report_limit_int = 256

        for run_start, run_end, perms in runs:
            run_size = run_end - run_start
            try:
                if perms:
                    try:
                        self.uc.mem_map(run_start, run_size, perms)
                    except TypeError:
                        # Preserve compatibility with minimal test doubles and
                        # old bindings that expose only the two-argument API.
                        self.uc.mem_map(run_start, run_size)
                else:
                    self.uc.mem_map(run_start, run_size)
            except Exception as exc:
                # A duplicate/concurrent map can report an error after the
                # requested topology is already present.  The direct native
                # postcondition is authoritative here.
                if not self._native_range_is_mapped(run_start, run_size):
                    self.memory_mapping_stats["map_failures"] += 1
                    self.last_memory_mapping = {
                        "status": "failed",
                        "requested_pages": self._bounded_page_sample(
                            list(normalized), report_limit_int
                        ),
                        "mapped_pages": self._bounded_page_sample(
                            mapped_pages, report_limit_int
                        ),
                        "error": f"{type(exc).__name__}: {str(exc)[:512]}",
                    }
                    return False
                self.memory_mapping_stats["duplicate_map_postcondition_ok"] += 1
            if not self._native_range_is_mapped(run_start, run_size):
                self.memory_mapping_stats["map_postcondition_failures"] += 1
                self.last_memory_mapping = {
                    "status": "failed",
                    "requested_pages": self._bounded_page_sample(
                        list(normalized), report_limit_int
                    ),
                    "mapped_pages": self._bounded_page_sample(
                        mapped_pages, report_limit_int
                    ),
                    "error": "map_postcondition_failed",
                }
                return False
            self._remember_mapped_range(run_start, run_end)
            mapped_pages.extend(range(run_start, run_end, 0x1000))
            self.memory_mapping_stats["map_runs"] += 1

        newly_mapped_count = len(mapped_pages) - len(already_mapped)
        self.memory_mapping_stats["pages_mapped"] += max(0, newly_mapped_count)
        if runs or already_mapped:
            self.memory_mapping_stats["safe_point_drains"] += 1
        self.last_memory_mapping = {
            "status": "mapped",
            "requested_pages": self._bounded_page_sample(
                list(normalized), report_limit_int
            ),
            "mapped_pages": self._bounded_page_sample(
                mapped_pages, report_limit_int
            ),
            "error": None,
        }
        return True

    def _forget_mapped_range(self, start: int, size: int) -> None:
        """Remove an unmapped interval from the Python-side fast-path index."""
        start = self._page_down(int(start))
        end = self._page_up(int(start) + max(1, int(size)))
        if end <= start:
            return
        retained: List[Tuple[int, int]] = []
        for region_start, region_end in list(getattr(self, "mapped_ranges", []) or []):
            region_start = int(region_start)
            region_end = int(region_end)
            if region_end <= start or region_start >= end:
                retained.append((region_start, region_end))
                continue
            if region_start < start:
                retained.append((region_start, start))
            if region_end > end:
                retained.append((end, region_end))
        self.mapped_ranges = self._merge_ranges(retained)
        self.memory_mapping_stats["index_invalidations"] += 1

    def _discard_pending_memory_maps(self, reason: str) -> None:
        """Drop a stale deferred request when the execution cannot be retried."""
        with self._native_state_lock():
            pending = getattr(self, "_pending_memory_map_pages", {}) or {}
            count = len(pending) if isinstance(pending, Mapping) else 0
            self._pending_memory_map_pages = {}
            self._pending_memory_map_origins = set()
            self._pending_memory_map_external_addresses = set()
            self._pending_memory_map_write_replays = []
            self._pending_memory_map_retries_block_callback = False
            self._deferred_retry_skip_block_pc = None
            self._deferred_retry_reenter_block_pc = None
        self._clear_deferred_write_retry_hooks()
        if count:
            self.memory_mapping_stats["stale_requests_discarded"] += count
            self.last_memory_mapping = {
                "status": "discarded",
                "requested_pages": [],
                "mapped_pages": [],
                "error": str(reason),
            }

    def _drain_pending_memory_maps(self) -> bool:
        """Apply all queued maps at a post-native safe point."""
        with self._native_state_lock():
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                return False
            pending = dict(getattr(self, "_pending_memory_map_pages", {}) or {})
            origins = set(
                getattr(self, "_pending_memory_map_origins", set()) or set()
            )
            external_addresses = set(
                getattr(
                    self,
                    "_pending_memory_map_external_addresses",
                    set(),
                )
                or set()
            )
            write_replays = [
                dict(item)
                for item in list(
                    getattr(self, "_pending_memory_map_write_replays", []) or []
                )
                if isinstance(item, Mapping)
            ]
            self._pending_memory_map_pages = {}
            self._pending_memory_map_origins = set()
            self._pending_memory_map_external_addresses = set()
            self._pending_memory_map_write_replays = []
            self._pending_memory_map_retries_block_callback = False
        if not pending:
            return True
        mapped = self._map_memory_pages_now(pending)
        if mapped and write_replays:
            mapped = self._install_deferred_write_retry_hooks(write_replays)
        if mapped and external_addresses:
            self.external_memory_input_addresses.update(external_addresses)
            self.memory_mapping_stats["external_addresses_committed"] += len(
                external_addresses
            )
        if isinstance(getattr(self, "last_memory_mapping", None), dict):
            self.last_memory_mapping["origins"] = sorted(origins)
        return mapped

    def _mark_pending_external_memory_address(self, address: int) -> None:
        """Commit external-memory classification only after its page is mapped."""
        with self._native_state_lock():
            pending = getattr(
                self,
                "_pending_memory_map_external_addresses",
                None,
            )
            if not isinstance(pending, set):
                pending = set()
                self._pending_memory_map_external_addresses = pending
            pending.add(int(address) & 0xFFFFFFFF)

    def _mark_pending_unmapped_write_replay(
        self,
        *,
        pc: int,
        address: int,
        size: int,
        value: int,
    ) -> None:
        """Remember one model-relevant write for a post-map one-shot hook."""
        record = {
            "pc": int(pc) & ~1,
            "address": int(address) & 0xFFFFFFFF,
            "size": max(1, int(size or 1)),
            "value": int(value or 0) & 0xFFFFFFFF,
        }
        with self._native_state_lock():
            pending = getattr(self, "_pending_memory_map_write_replays", None)
            if not isinstance(pending, list):
                pending = []
                self._pending_memory_map_write_replays = pending
            if record not in pending:
                pending.append(record)
                self.memory_mapping_stats["write_replays_queued"] += 1

    def _install_deferred_write_retry_hooks(
        self,
        records: List[Dict[str, int]],
    ) -> bool:
        """Install low-cost one-shot hooks when global write tracing is off."""
        if bool(getattr(self, "mapped_write_hook_enabled", True)):
            return True
        with self._native_state_lock():
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                return False

        installed: List[object] = []
        try:
            for raw_record in records:
                record = {
                    "pc": int(raw_record.get("pc", 0)) & ~1,
                    "address": int(raw_record.get("address", 0)) & 0xFFFFFFFF,
                    "size": max(1, int(raw_record.get("size", 1) or 1)),
                    "value": int(raw_record.get("value", 0)) & 0xFFFFFFFF,
                }
                state: Dict[str, object] = {"done": False, "handle": None}

                def replay_write_hook(
                    uc,
                    access,
                    address,
                    size,
                    value,
                    user_data,
                    *,
                    expected=record,
                    hook_state=state,
                ):
                    if bool(hook_state["done"]):
                        return
                    try:
                        current_pc = int(uc.reg_read(UC_ARM_REG_PC)) & ~1
                    except Exception:
                        current_pc = 0
                    if (
                        current_pc != int(expected["pc"])
                        or (int(address) & 0xFFFFFFFF)
                        != int(expected["address"])
                    ):
                        return
                    hook_state["done"] = True
                    try:
                        self.mem_write_hook(
                            uc,
                            access,
                            address,
                            size,
                            value,
                            user_data,
                        )
                        self.memory_mapping_stats["write_replays_applied"] += 1
                    finally:
                        handle = hook_state.get("handle")
                        if handle is not None:
                            self._remove_owned_hook(handle)

                handle = self._add_owned_hook(
                    UC_HOOK_MEM_WRITE,
                    replay_write_hook,
                    None,
                    int(record["address"]),
                    int(record["address"]) + int(record["size"]) - 1,
                )
                state["handle"] = handle
                installed.append(handle)
            self._deferred_write_retry_hooks.extend(installed)
            self.memory_mapping_stats["write_replay_hooks_installed"] += len(
                installed
            )
            return True
        except Exception as exc:
            self.memory_mapping_stats["write_replay_hook_failures"] += 1
            for handle in reversed(installed):
                self._remove_owned_hook(handle)
            self.last_memory_mapping = {
                "status": "failed",
                "requested_pages": list(
                    (self.last_memory_mapping or {}).get("requested_pages", [])
                ),
                "mapped_pages": list(
                    (self.last_memory_mapping or {}).get("mapped_pages", [])
                ),
                "error": (
                    "write_replay_hook_install_failed:"
                    f"{type(exc).__name__}:{str(exc)[:384]}"
                ),
            }
            return False

    def _clear_deferred_write_retry_hooks(self) -> None:
        """Remove stale one-shot hooks after a managed native call finishes."""
        hooks = list(getattr(self, "_deferred_write_retry_hooks", []) or [])
        self._deferred_write_retry_hooks = []
        for handle in reversed(hooks):
            self._remove_owned_hook(handle)

    def _set_deferred_retry_block_skip(
        self,
        resume_pc: int,
        retry_block_callback: bool,
    ) -> None:
        """Avoid repeating BB bookkeeping when a later callback was retried."""
        with self._native_state_lock():
            if retry_block_callback:
                self._deferred_retry_skip_block_pc = None
                self._deferred_retry_reenter_block_pc = int(resume_pc) & ~1
                self.memory_mapping_stats["retry_block_reentries_armed"] += 1
                return
            self._deferred_retry_reenter_block_pc = None
            self._deferred_retry_skip_block_pc = int(resume_pc) & ~1
            self.memory_mapping_stats["retry_block_skips_armed"] += 1

    def _consume_deferred_retry_block_skip(self, address: int) -> bool:
        """Consume the one-shot duplicate-BB marker for a resumed instruction."""
        current = int(address) & ~1
        with self._native_state_lock():
            expected = getattr(self, "_deferred_retry_skip_block_pc", None)
            if expected is None:
                return False
            self._deferred_retry_skip_block_pc = None
            if int(expected) == current:
                self.memory_mapping_stats["retry_block_skips_consumed"] += 1
                return True
            self.memory_mapping_stats["retry_block_skip_mismatches"] += 1
            return False

    def _consume_deferred_retry_block_reentry(self, address: int) -> bool:
        """Consume a retry that must rerun block actions without BB bookkeeping."""
        current = int(address) & ~1
        with self._native_state_lock():
            expected = getattr(self, "_deferred_retry_reenter_block_pc", None)
            if expected is None:
                return False
            self._deferred_retry_reenter_block_pc = None
            if int(expected) == current:
                self.memory_mapping_stats["retry_block_reentries_consumed"] += 1
                return True
            self.memory_mapping_stats["retry_block_reentry_mismatches"] += 1
            return False

    def _clear_deferred_retry_block_skip(self, reason: str) -> None:
        with self._native_state_lock():
            skip_pending = getattr(self, "_deferred_retry_skip_block_pc", None)
            reentry_pending = getattr(
                self,
                "_deferred_retry_reenter_block_pc",
                None,
            )
            if skip_pending is None and reentry_pending is None:
                return
            self._deferred_retry_skip_block_pc = None
            self._deferred_retry_reenter_block_pc = None
            self.memory_mapping_stats[f"retry_block_skip_cleared_{reason}"] += 1

    def _resume_pc_after_deferred_mapping(self) -> Optional[int]:
        """Return the faulting instruction PC for a transparent retry."""
        try:
            pc = int(self.uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
        except Exception:
            return None
        return (pc | 1) if self.execution_thumb else (pc & ~1)

    @staticmethod
    def _set_native_call_argument(
        call_args: List[object],
        call_kwargs: Dict[str, object],
        index: int,
        name: str,
        value: object,
    ) -> None:
        if name in call_kwargs:
            call_kwargs[name] = value
        elif len(call_args) > index:
            call_args[index] = value
        else:
            call_kwargs[name] = value

    def _managed_emu_start(self, *args, **kwargs):
        """Run native emulation, transparently retrying deferred page maps.

        A retry begins at the PC synchronized by Unicorn for the interrupted
        memory access.  Consequently the guest instruction that requested the
        page is executed exactly once after the page becomes available; the
        caller does not need a special recovery path.
        """
        if bool(getattr(self, "_close_requested", False)) or bool(
            getattr(self, "_close_completed", False)
        ):
            raise RuntimeError("cannot start emulation after close was requested")

        call_args = list(args)
        call_kwargs = dict(kwargs)
        try:
            timeout_us = int(
                call_kwargs.get(
                    "timeout",
                    call_args[2] if len(call_args) > 2 else 0,
                )
                or 0
            )
        except (TypeError, ValueError, OverflowError):
            timeout_us = 0
        deadline = (
            time.monotonic() + (timeout_us / 1_000_000.0)
            if timeout_us > 0
            else None
        )
        try:
            retry_limit = max(
                1,
                int(os.environ.get("LSGEMU_MEMORY_MAP_RETRY_LIMIT", "128")),
            )
        except ValueError:
            retry_limit = 128
        retries = 0
        # ITSTATE 守卫的重入次数按「一次 _managed_emu_start 调用」计，不跨调用累计。
        it_reentries = 0
        self._it_state_guard_pending = False

        # A callback can surface the deferred signal directly, or Unicorn can
        # convert it to UC_ERR_MAP/UC_ERR_*_UNMAPPED after the invalid-memory
        # dispatcher observes that the page is still absent.  Both cases are
        # recoverable only when our queue is non-empty.
        recoverable_native_errors = {
            int(getattr(_unicorn, "UC_ERR_MAP", 11)),
            int(getattr(_unicorn, "UC_ERR_READ_UNMAPPED", 6)),
            int(getattr(_unicorn, "UC_ERR_WRITE_UNMAPPED", 7)),
            int(getattr(_unicorn, "UC_ERR_FETCH_UNMAPPED", 8)),
        }

        while True:
            caught: Optional[BaseException] = None
            deferred_origins: Set[str] = set()
            retry_block_callback = False
            try:
                with self._native_emulation_scope():
                    result = self.uc.emu_start(*call_args, **call_kwargs)
            except BaseException as exc:
                pending, deferred_origins, retry_block_callback = (
                    self._pending_memory_map_context_snapshot()
                )
                errno = getattr(exc, "errno", None)
                if not pending or not (
                    isinstance(exc, _DeferredMemoryMapping)
                    or errno in recoverable_native_errors
                ):
                    if pending:
                        self._discard_pending_memory_maps(
                            f"non_recoverable_native_error:{type(exc).__name__}"
                        )
                    else:
                        self._clear_deferred_retry_block_skip(
                            "non_recoverable_native_error"
                        )
                        self._clear_deferred_write_retry_hooks()
                    raise
                caught = exc
            else:
                pending, deferred_origins, retry_block_callback = (
                    self._pending_memory_map_context_snapshot()
                )
                if not pending:
                    self._clear_deferred_retry_block_skip("native_return")
                    self._clear_deferred_write_retry_hooks()
                    # r29 脏 ITSTATE 守卫：TB 已退出 => condexec 已写回 env，
                    # 此刻清 IT 才不会被覆盖。清理后在原 PC 原地重入 = 打断 TB，
                    # 让该指令按其自身（无条件）编码执行。
                    if self._it_state_guard_pending and self._repair_stale_itstate():
                        it_reentries += 1
                        self.it_state_guard_stats["reentries"] += 1
                        resume_pc = self._resume_pc_after_deferred_mapping()
                        expired = False
                        if deadline is not None:
                            remaining_us = int(
                                max(0.0, deadline - time.monotonic()) * 1_000_000
                            )
                            if remaining_us <= 0:
                                expired = True
                            else:
                                self._set_native_call_argument(
                                    call_args, call_kwargs, 2, "timeout", remaining_us
                                )
                        if (
                            resume_pc is not None
                            and (int(resume_pc) & ~1) != 0
                            and it_reentries <= self.it_state_guard_reentry_limit
                            and not expired
                        ):
                            if call_args:
                                call_args[0] = resume_pc
                            else:
                                call_kwargs["begin"] = resume_pc
                            continue
                        if it_reentries > self.it_state_guard_reentry_limit:
                            logger.warning(
                                "脏ITSTATE守卫重入超限(%d)，放弃本轮重入",
                                self.it_state_guard_reentry_limit,
                            )
                    return result

            if retries >= retry_limit:
                self.memory_mapping_stats["retry_exhausted"] += 1
                self._discard_pending_memory_maps("retry_limit_exhausted")
                error = RuntimeError(
                    "deferred memory mapping retry limit exhausted"
                )
                if caught is not None:
                    raise error from caught
                raise error

            if not self._drain_pending_memory_maps():
                self.memory_mapping_stats["retry_map_failures"] += 1
                self._discard_pending_memory_maps("safe_point_mapping_failed")
                error = RuntimeError("deferred memory mapping failed at safe point")
                if caught is not None:
                    raise error from caught
                raise error

            resume_pc = self._resume_pc_after_deferred_mapping()
            if resume_pc is None:
                self._discard_pending_memory_maps("resume_pc_unreadable")
                error = RuntimeError(
                    "cannot read PC after deferred memory mapping"
                )
                if caught is not None:
                    raise error from caught
                raise error

            retries += 1
            self.memory_mapping_stats["retries"] += 1
            if not call_args and not call_kwargs:
                raise RuntimeError("invalid emulation call without a start address")
            if call_args:
                call_args[0] = resume_pc
            else:
                call_kwargs["begin"] = resume_pc
            self._set_deferred_retry_block_skip(
                resume_pc,
                retry_block_callback,
            )

            if deadline is not None:
                remaining_us = int(
                    max(0.0, deadline - time.monotonic()) * 1_000_000
                )
                if remaining_us <= 0:
                    self._clear_deferred_retry_block_skip("timeout")
                    self._clear_deferred_write_retry_hooks()
                    error = RuntimeError(
                        "deferred memory mapping retry exceeded emulation timeout"
                    )
                    if caught is not None:
                        raise error from caught
                    raise error
                self._set_native_call_argument(
                    call_args,
                    call_kwargs,
                    2,
                    "timeout",
                    remaining_us,
                )
            logger.debug(
                "native执行安全点映射后重试: retry=%d pc=0x%08x",
                retries,
                int(resume_pc) & 0xFFFFFFFF,
            )

    def _remove_owned_hook(self, hook):
        """Remove one owned hook, deferring mutation during native execution."""
        if hook is None:
            return
        with self._native_state_lock():
            if hook in getattr(self, "_owned_hooks", []):
                if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                    if hook not in getattr(self, "_pending_owned_hook_removals", []):
                        self._pending_owned_hook_removals.append(hook)
                        self.hook_lifecycle_stats["deferred_deletes"] += 1
                    return
                self._delete_owned_hook_now_locked(hook)

    def _clear_owned_hooks(self, uc=None):
        """Delete emulator-owned hooks explicitly before releasing Unicorn."""
        with self._native_state_lock():
            uc = uc if uc is not None else getattr(self, "uc", None)
            if uc is None:
                return
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                for hook in list(getattr(self, "_owned_hooks", []) or []):
                    if hook not in self._pending_owned_hook_removals:
                        self._pending_owned_hook_removals.append(hook)
                self.hook_lifecycle_stats["clear_deferred"] += 1
                return
            try:
                unregister_primary_mmio_handler(uc, getattr(self, "mmio_handler", None))
            except Exception:
                pass
            try:
                unregister_hook_owner(uc, self)
            except Exception:
                pass
            self._drain_pending_owned_hook_removals()
            for hook in reversed(list(getattr(self, "_owned_hooks", []) or [])):
                self._delete_owned_hook_now_locked(hook)
            self._owned_hooks = []
            self._owned_hook_records = {}
            self._pending_owned_hook_removals = []
            self.forced_branch_code_hook = None
            self.branch_entry_code_hook = None
            self.runtime_loop_branch_force_hook = None

    def close(self):
        """Release references to the native Unicorn engine safely.

        Unicorn's private finalizer is intentionally not called here. Replay
        phases create and destroy many engines while hook context managers are
        active; invoking the private finalizer manually can double-finalize the
        underlying handle later and crash the Python process.
        """
        current_thread_id = threading.get_ident()
        with self._native_state_lock():
            if bool(getattr(self, "_close_completed", False)):
                self.hook_lifecycle_stats["duplicate_close_ignored"] += 1
                return
            uc = getattr(self, "uc", None)
            if uc is None:
                self._close_requested = True
                self._discard_pending_memory_maps("close_without_unicorn")
                self._close_completed = True
                self.hook_lifecycle_stats["duplicate_close_ignored"] += 1
                return
            if not bool(getattr(self, "_close_requested", False)):
                self._close_requested = True
                self.hook_lifecycle_stats["close_requested"] += 1

            active = int(getattr(self, "_native_emulation_depth", 0) or 0) > 0
            owner_thread_id = getattr(self, "_native_emulation_thread_id", None)
            if (
                active
                and owner_thread_id is not None
                and owner_thread_id != current_thread_id
            ):
                # ``uc_emu_stop`` is a native operation on the same engine as
                # ``uc_emu_start``.  Calling it from a lifecycle/monitor thread
                # races the translation loop on Unicorn versions that do not
                # guarantee cross-thread engine access.  The emulation owner
                # observes ``_close_requested`` when its scope unwinds and
                # performs the actual hook/native cleanup there.
                self.hook_lifecycle_stats["cross_thread_close_deferred"] += 1
                return
        try:
            uc.emu_stop()
        except Exception:
            pass
        with self._native_state_lock():
            active = int(getattr(self, "_native_emulation_depth", 0) or 0) > 0
        if active:
            self.hook_lifecycle_stats["close_deferred"] += 1
            self._clear_owned_hooks(uc)
            return

        # Normal replay cleanup completes synchronously.  A callback-triggered
        # close reaches the same helper from ``_native_emulation_scope`` after
        # ``uc_emu_start`` has unwound.
        self._complete_deferred_close(uc)

    @staticmethod
    def _forced_branch_master_disabled() -> bool:
        """r38→r40：读 LSGEMU_FORCED_BRANCH_DISABLED 总开关（默认 1 = 零 force）。

        r40 P1 起缺省翻转：r38 A/B 实测 force 对覆盖与证据零贡献
        （797 vs 797、validated 797 vs 797、丢失集 ∅），按用户
        「覆盖的基本块都是真实的，不需要 Forced」拍板转正。
        显式 =0 可恢复旧诊断臂。
        """
        return os.environ.get(
            "LSGEMU_FORCED_BRANCH_DISABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _refresh_forced_branch_disabled_flag(self) -> bool:
        """刷新并返回缓存的总开关值；setter/run() 入口调用。"""
        self._forced_branch_disabled = self._forced_branch_master_disabled()
        return bool(self._forced_branch_disabled)

    def _reset_forced_branch_state(self) -> None:
        """清空 forced_branch 全部状态（零 force 语义：无 choices、无目标集）。"""
        self.forced_branch_choices = {}
        self.forced_branch_sequence = []
        self.forced_branch_sequence_index = 0
        self.forced_branch_hits.clear()
        self.forced_branch_trace.clear()
        self.forced_branch_encounters.clear()
        self.forced_branch_pending.clear()
        self.forced_branch_target_bbs = set()

    def _forced_branch_audit_fields(self) -> Dict[str, object]:
        """r38 审计面：总开关是否生效 + 被挡下的 force 请求计数。"""
        return {
            "forced_branch_disabled": bool(
                getattr(self, "_forced_branch_disabled", False)
            ),
            "forced_branch_disabled_blocks": {
                str(key): int(value)
                for key, value in dict(
                    getattr(self, "forced_branch_disabled_blocks", {}) or {}
                ).items()
            },
        }

    def set_forced_branch_choices(self, choices: Dict[object, bool]):
        """
        设置本次从入口重放时必须满足的分支方向。

        choices 的 key 可以是条件分支所在BB地址，或 (BB地址, 第N次出现)。
        value=True 表示 taken，value=False 表示 not-taken。
        """
        if self._refresh_forced_branch_disabled_flag():
            # r38 总开关：零 force 臂。签名与调用点保持不变，只清状态、
            # 不装钩子；若此前已装钩，钩子入口的保险丝会挡下残留 force。
            if choices:
                self.forced_branch_disabled_blocks["setter"] += 1
            self._reset_forced_branch_state()
            return
        self.forced_branch_choices = dict(choices or {})
        self.forced_branch_sequence = []
        self.forced_branch_sequence_index = 0
        self.forced_branch_hits.clear()
        self.forced_branch_trace.clear()
        self.forced_branch_encounters.clear()
        self.forced_branch_pending.clear()
        self.forced_branch_target_bbs = set()
        for key in self.forced_branch_choices:
            if isinstance(key, tuple):
                try:
                    self.forced_branch_target_bbs.add(int(key[0]))
                except Exception:
                    continue
            else:
                try:
                    self.forced_branch_target_bbs.add(int(key))
                except Exception:
                    continue
        if self.forced_branch_choices and self.forced_branch_code_hook is None:
            self.forced_branch_code_hook = self._add_owned_hook(
                UC_HOOK_CODE,
                self._forced_branch_instruction_hook
            )

    def set_ordered_forced_branch_choices(self, choices: List[Tuple[object, object]]):
        """
        设置按路径顺序触发的分支约束。

        普通地址级约束会在第一次遇到该BB时生效；多分支路径中，子分支
        地址可能也出现在前缀路径里。顺序约束只在前一个约束已经触发后
        才等待下一个分支点，从而保留“从入口真实执行到该路径”的语义。
        """
        if self._refresh_forced_branch_disabled_flag():
            # r38 总开关：同 set_forced_branch_choices 的关断语义。
            if choices:
                self.forced_branch_disabled_blocks["setter"] += 1
            self._reset_forced_branch_state()
            return
        self.forced_branch_choices = {}
        normalized_sequence = []
        for address_or_key, choice in (choices or []):
            if isinstance(address_or_key, tuple):
                normalized_sequence.append((
                    (int(address_or_key[0]), int(address_or_key[1])),
                    choice,
                ))
            else:
                normalized_sequence.append((int(address_or_key), choice))
        self.forced_branch_sequence = normalized_sequence
        self.forced_branch_sequence_index = 0
        self.forced_branch_hits.clear()
        self.forced_branch_trace.clear()
        self.forced_branch_encounters.clear()
        self.forced_branch_pending.clear()
        self.forced_branch_target_bbs = set()
        for address_or_key, _choice in self.forced_branch_sequence:
            if isinstance(address_or_key, tuple):
                self.forced_branch_target_bbs.add(int(address_or_key[0]))
            else:
                self.forced_branch_target_bbs.add(int(address_or_key))
        if self.forced_branch_sequence and self.forced_branch_code_hook is None:
            self.forced_branch_code_hook = self._add_owned_hook(
                UC_HOOK_CODE,
                self._forced_branch_instruction_hook
            )

    def _normalize_mnemonic(self, mnemonic: str) -> str:
        """Normalize Thumb mnemonics while preserving B.<cond> as BCOND."""
        raw = str(mnemonic or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B":
            condition = parts[1]
            if condition not in {"N", "W", "NW"}:
                return f"B{condition}"
            return "B"
        return parts[0]

    def _predicated_condition_for_mnemonic(self, mnemonic: str) -> Optional[str]:
        """Return the Thumb IT predicate condition encoded in mnemonics like strb.le.w."""
        raw = str(mnemonic or "").strip().upper()
        if not raw or "." not in raw:
            return None
        parts = [part for part in raw.split(".") if part]
        if len(parts) < 2:
            return None
        base = parts[0]
        if base in {"B", "BL", "BLX", "BX", "BXJ"} or base.startswith("IT"):
            return None
        for part in parts[1:]:
            if part in THUMB_CONDITION_CODES:
                return part
        return None

    def _it_predicate_token_for_instruction(self, insn: Dict[str, object]) -> Optional[str]:
        condition = self._predicated_condition_for_mnemonic(insn.get("mnemonic", ""))
        return f"IT{condition}" if condition else None

    def _condition_from_dispatch_token(self, condition: str) -> str:
        normalized = self._normalize_mnemonic(condition or "")
        if normalized.startswith("IT") and normalized[2:] in THUMB_CONDITION_CODES:
            return normalized[2:]
        return normalized

    def _branch_condition_for_mnemonic(self, mnemonic: str) -> Optional[str]:
        """Return a branch condition token for supported conditional branches."""
        return {
            'BEQ': 'EQ', 'BNE': 'NE', 'BCS': 'CS', 'BCC': 'CC',
            'BMI': 'MI', 'BPL': 'PL', 'BVS': 'VS', 'BVC': 'VC',
            'BHI': 'HI', 'BLS': 'LS', 'BGE': 'GE', 'BLT': 'LT',
            'BGT': 'GT', 'BLE': 'LE', 'BHS': 'HS', 'BLO': 'LO',
            'CBZ': 'CBZ', 'CBNZ': 'CBNZ', 'TBB': 'TBB', 'TBH': 'TBH',
        }.get(self._normalize_mnemonic(mnemonic))

    @staticmethod
    def _condition_result_from_cpsr(condition: str, cpsr: int) -> Optional[bool]:
        condition = str(condition or "").upper()
        cpsr = int(cpsr) & 0xFFFFFFFF
        n = bool(cpsr & (1 << 31))
        z = bool(cpsr & (1 << 30))
        c = bool(cpsr & (1 << 29))
        v = bool(cpsr & (1 << 28))
        if condition == "EQ":
            return z
        if condition == "NE":
            return not z
        if condition in {"CS", "HS"}:
            return c
        if condition in {"CC", "LO"}:
            return not c
        if condition == "MI":
            return n
        if condition == "PL":
            return not n
        if condition == "VS":
            return v
        if condition == "VC":
            return not v
        if condition == "HI":
            return c and not z
        if condition == "LS":
            return (not c) or z
        if condition == "GE":
            return n == v
        if condition == "LT":
            return n != v
        if condition == "GT":
            return (not z) and (n == v)
        if condition == "LE":
            return z or (n != v)
        return None

    def _dispatch_condition_for_mnemonic(self, mnemonic: str) -> Optional[str]:
        """Return a dispatch token for branch-like calls/returns that can be forced only after being reached."""
        normalized = self._normalize_mnemonic(mnemonic)
        if normalized in {'BLX', 'BX', 'BXJ'}:
            return normalized
        return self._branch_condition_for_mnemonic(normalized)

    def _dispatch_condition_for_instruction(self, insn: Dict[str, object]) -> Optional[str]:
        """Return a dispatch token using operands for PC-load switch patterns."""
        if self._is_ldr_pc_dispatch_instruction(insn):
            return "LDRPC"
        it_predicate = self._it_predicate_token_for_instruction(insn)
        if it_predicate is not None:
            return it_predicate
        return self._dispatch_condition_for_mnemonic(insn.get('mnemonic', ''))

    def _is_switch_dispatch_condition(self, condition: Optional[str]) -> bool:
        return self._normalize_mnemonic(condition or "") in {'TBB', 'TBH', 'LDRPC'}

    def _is_direct_call_mnemonic(self, mnemonic: str) -> bool:
        """Direct BL is not a controllable branch, but its reached state is useful for strict continuation replay."""
        return self._normalize_mnemonic(mnemonic) == 'BL'

    def _is_call_dispatch_condition(self, condition: Optional[str]) -> bool:
        return self._normalize_mnemonic(condition or "") in {'BLX', 'BX', 'BXJ'}

    def _is_ldr_pc_dispatch_instruction(self, insn: Dict[str, object]) -> bool:
        """Detect ARM/Thumb table dispatches such as `ldr.w pc, [r2, r3, lsl #2]`."""
        mnemonic = self._normalize_mnemonic(insn.get('mnemonic', ''))
        if not mnemonic.startswith('LDR'):
            return False
        operands = self._split_operands(insn.get('operands', ''))
        if not operands or operands[0].strip().lower() != 'pc':
            return False
        return self._parse_ldr_pc_switch_operands(insn.get('operands', '')) is not None

    def _parse_branch_target(self, operands: str) -> Optional[int]:
        """Parse the destination address from a branch operand string."""
        matches = re.findall(r'0x[0-9a-fA-F]+', str(operands or ""))
        if not matches:
            return None
        return int(matches[-1], 16)

    def _split_operands(self, operands: str) -> List[str]:
        """Split ARM operands without breaking bracketed memory operands."""
        parts: List[str] = []
        current: List[str] = []
        bracket_depth = 0
        brace_depth = 0
        for ch in str(operands or ""):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1

            if ch == "," and bracket_depth == 0 and brace_depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _parse_cbz_register(self, operands: str) -> Optional[str]:
        parts = self._split_operands(operands)
        if not parts:
            return None
        reg = parts[0].lower()
        if re.fullmatch(r'r(?:[0-9]|1[0-2])', reg):
            return reg
        return None

    def _write_register_by_name(self, uc, reg_name: str, value: int) -> bool:
        reg_map = {
            'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1, 'r2': UC_ARM_REG_R2,
            'r3': UC_ARM_REG_R3, 'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
            'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7, 'r8': UC_ARM_REG_R8,
            'r9': UC_ARM_REG_R9, 'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
            'r12': UC_ARM_REG_R12, 'lr': UC_ARM_REG_LR, 'pc': UC_ARM_REG_PC,
        }
        reg_id = reg_map.get(reg_name.lower())
        if reg_id is None:
            return False
        uc.reg_write(reg_id, value & 0xFFFFFFFF)
        return True

    def _parse_dispatch_target_register(self, operands: str) -> Optional[str]:
        """Parse register-indirect BX/BLX targets such as `bx lr` or `blx r3`."""
        parts = self._split_operands(operands)
        if not parts:
            return None
        reg = parts[0].lower()
        if re.fullmatch(r'r(?:[0-9]|1[0-2])|lr|pc', reg):
            return reg
        return None

    def _force_cbz_cbnz_register(self, uc, branch_insn: Dict, take_branch: bool) -> bool:
        """Force CBZ/CBNZ by changing only the tested register value."""
        mnemonic = self._normalize_mnemonic(branch_insn.get('mnemonic', ''))
        reg_name = self._parse_cbz_register(branch_insn.get('operands', ''))
        if reg_name is None:
            return False

        want_zero = (mnemonic == 'CBZ' and take_branch) or (mnemonic == 'CBNZ' and not take_branch)
        if want_zero:
            return self._write_register_by_name(uc, reg_name, 0)

        reg_id = self._arm_reg_id(reg_name)
        if reg_id is not None:
            try:
                current = int(uc.reg_read(reg_id)) & 0xFFFFFFFF
                if current != 0:
                    # Preserve a concrete non-zero value.  Several replay
                    # paths later use the same CBZ-tested register as a
                    # pointer; replacing an existing pointer with integer 1
                    # creates artificial faults while still satisfying CBZ.
                    return True
            except Exception:
                pass
        return self._write_register_by_name(uc, reg_name, 1)

    def _parse_switch_index_register(self, operands: str) -> Optional[str]:
        """Parse TBB/TBH index register from forms like [pc,r5]."""
        match = re.search(r'\[\s*pc\s*,\s*(r(?:[0-9]|1[0-2]))', str(operands or "").lower())
        return match.group(1) if match else None

    def _parse_ldr_pc_switch_operands(self, operands: str) -> Optional[Tuple[str, str, int]]:
        """Parse `ldr pc, [base, index, lsl #shift]` jump table operands."""
        text = str(operands or "").lower()
        match = re.search(
            r'^\s*pc\s*,\s*\[\s*(r(?:[0-9]|1[0-2])|pc)\s*,\s*(r(?:[0-9]|1[0-2]))'
            r'(?:\s*,\s*lsl\s*#?(\d+))?\s*\]',
            text,
        )
        if not match:
            parts = [part.strip().lower() for part in self._split_operands(operands)]
            if len(parts) < 3 or parts[0] != 'pc':
                return None
            base_reg = parts[1]
            index_reg = parts[2]
            if not re.fullmatch(r'r(?:[0-9]|1[0-2])|pc', base_reg):
                return None
            if not re.fullmatch(r'r(?:[0-9]|1[0-2])', index_reg):
                return None
            shift = self._parse_int(parts[3]) if len(parts) >= 4 else 0
            if shift is None:
                shift = self._parse_immediate_operand(parts[3]) if len(parts) >= 4 else 0
            shift = int(shift or 0)
            if shift < 0 or shift > 5:
                return None
            return base_reg, index_reg, shift
        shift = int(match.group(3) or 0)
        if shift < 0 or shift > 5:
            return None
        return match.group(1), match.group(2), shift

    def _switch_entry_size(self, mnemonic: str) -> int:
        return 1 if self._normalize_mnemonic(mnemonic) == 'TBB' else 2

    def _read_switch_targets(self, uc, bb_addr: int, last_insn: Dict) -> List[int]:
        """Decode TBB/TBH table targets in index order."""
        mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
        if mnemonic not in {'TBB', 'TBH'}:
            return []

        branch_pc = last_insn.get('address', bb_addr)
        table_base = branch_pc + last_insn.get('size', 4)
        entry_size = self._switch_entry_size(mnemonic)
        next_code_bbs = [addr for addr in self.static_bbs if addr > table_base]
        if not next_code_bbs:
            return []

        table_end = min(next_code_bbs)
        table_size = max(0, table_end - table_base)
        entry_count = table_size // entry_size
        if entry_count <= 0:
            return []

        # Dense switch tables may be large; cap defensively to avoid decoding code bytes on bad metadata.
        entry_count = min(entry_count, 256)
        targets: List[int] = []
        try:
            raw = bytes(uc.mem_read(table_base, entry_count * entry_size))
            for index in range(entry_count):
                if entry_size == 1:
                    offset = raw[index]
                else:
                    offset = int.from_bytes(raw[index * 2:index * 2 + 2], 'little')
                target = table_base + 2 * offset
                if self._resolve_bb_start(target) in self.static_bbs:
                    targets.append(target)
                else:
                    targets.append(0)
        except Exception as e:
            logger.debug(f"读取switch表失败 @ 0x{branch_pc:08x}: {e}")
        return targets

    def _read_ldr_pc_switch_targets(self, uc, bb_addr: int, last_insn: Dict) -> List[int]:
        """Decode `ldr pc, [base, index, lsl #2]` table targets from the reached state."""
        parsed = self._parse_ldr_pc_switch_operands(last_insn.get('operands', ''))
        if parsed is None:
            return []
        base_reg, _index_reg, shift = parsed
        branch_pc = int(last_insn.get('address', bb_addr) or bb_addr)
        stride = max(1, 1 << shift)
        if stride < 4:
            stride = 4

        try:
            if base_reg == 'pc':
                table_base = (branch_pc + 4) & ~3
            else:
                reg_id = self._arm_reg_id(base_reg)
                if reg_id is None:
                    return []
                table_base = int(uc.reg_read(reg_id)) & 0xFFFFFFFF
        except Exception:
            return []

        next_code_bbs = [addr for addr in self.static_bbs if addr > table_base]
        if next_code_bbs:
            table_end = min(next_code_bbs)
            entry_count = max(0, (table_end - table_base) // stride)
        else:
            entry_count = 0
        if entry_count <= 0:
            entry_count = 64
        entry_count = min(entry_count, 256)

        targets: List[int] = []
        for index in range(entry_count):
            try:
                raw = bytes(uc.mem_read(table_base + index * stride, 4))
                target = int.from_bytes(raw, 'little') & 0xFFFFFFFF
            except Exception:
                break
            target = target & ~1
            target_bb = self._resolve_bb_start(target)
            if target_bb in self.static_bbs:
                targets.append(target)
            elif targets:
                # Stop after the contiguous table ends; leading invalid entries
                # are kept as zero to preserve sparse indices.
                break
            else:
                targets.append(0)
        return targets

    def _force_switch_index(self, uc, branch_insn: Dict, target_index) -> bool:
        """Force a TBB/TBH path by changing only the table index register."""
        try:
            index = int(target_index)
        except (TypeError, ValueError):
            return False
        if index < 0:
            return False
        reg_name = self._parse_switch_index_register(branch_insn.get('operands', ''))
        if reg_name is None:
            return False
        return self._write_register_by_name(uc, reg_name, index)

    def _force_ldr_pc_switch_index(self, uc, branch_insn: Dict, target_index) -> bool:
        """Force an LDR-PC jump table path by changing only its index register."""
        try:
            index = int(target_index)
        except (TypeError, ValueError):
            return False
        if index < 0:
            return False
        parsed = self._parse_ldr_pc_switch_operands(branch_insn.get('operands', ''))
        if parsed is None:
            return False
        _base_reg, index_reg, _shift = parsed
        return self._write_register_by_name(uc, index_reg, index)

    def _force_call_dispatch_target(self, uc, branch_insn: Dict, target_bb) -> bool:
        """Force a call-like dispatch target after the dispatch BB has been reached."""
        try:
            desired = int(target_bb) & ~1
        except (TypeError, ValueError):
            return False
        if desired <= 0:
            return False

        mnemonic = self._normalize_mnemonic(branch_insn.get('mnemonic', ''))
        operands = branch_insn.get('operands', '')
        branch_pc = int(branch_insn.get('address', desired) or desired)
        size = int(branch_insn.get('size', 2) or 2)

        if mnemonic in {'BX', 'BXJ'} or (mnemonic == 'BLX' and self._parse_dispatch_target_register(operands)):
            reg_name = self._parse_dispatch_target_register(operands)
            if reg_name is None:
                return False
            return self._write_register_by_name(uc, reg_name, desired | 1)

        parsed_target = self._parse_branch_target(operands)
        if parsed_target is not None and self._resolve_bb_start(parsed_target) == desired:
            return True

        return False

    def _get_direct_call_info(self, bb_address: int) -> Optional[Dict[str, int]]:
        """Return direct BL target/fallthrough metadata without treating BL as a forced branch."""
        bb_instructions = self.static_bbs.get(bb_address, [])
        if not bb_instructions:
            return None
        last_insn = bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
        if not self._is_direct_call_mnemonic(mnemonic):
            return None
        target = self._parse_branch_target(last_insn.get('operands', ''))
        if target is None:
            return None
        branch_pc = int(last_insn.get('address', bb_address) or bb_address)
        size = int(last_insn.get('size', 2) or 2)
        return {
            "branch_pc": branch_pc,
            "target": self._resolve_bb_start(target),
            "fallthrough": self._resolve_bb_start(branch_pc + size),
        }

    def mmio_read_hook(self, uc, access, address, size, value, user_data):
        """
        普通内存读取hook

        用于捕获已映射的MMIO访问（0x40000000-0x60000000）
        以及基于 read_pc 的持久化内存约束
        """
        pc = int(uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
        address = int(address) & 0xFFFFFFFF
        decoded_alias = self._decode_bitband_alias(address)
        is_bitband_alias = decoded_alias is not None
        is_mmio_read = bool(
            self._is_mmio_address(address)
            or (
                decoded_alias is not None
                and self._is_mmio_address(decoded_alias[0])
            )
        )
        if is_bitband_alias:
            # The alias page can be mapped independently of its backing page.
            # Make the backing dependency explicit before allocating an input
            # occurrence or invoking the stateful MMIO model.
            if not self._ensure_bitband_access_mapped(address, size):
                return
        is_external_memory_read = bool(
            not is_mmio_read
            and not is_bitband_alias
            and self._is_external_input_memory_range(address, size)
        )
        causal_input_token = None
        if is_mmio_read or is_external_memory_read:
            causal_input_token = self._begin_causal_input_read(
                kind="mmio" if is_mmio_read else "external_memory",
                pc=pc,
                address=address,
                size=size,
            )

        memory_site = (pc, address)
        memory_occurrence = 0
        if is_external_memory_read and causal_input_token is not None:
            memory_occurrence = int(causal_input_token.get("occurrence", 0) or 0)
            self.memory_read_occurrence_counts[memory_site] = memory_occurrence
        elif memory_site in self.memory_occurrence_constraint_sites:
            self.memory_read_occurrence_counts[memory_site] += 1
            memory_occurrence = int(self.memory_read_occurrence_counts[memory_site])

        # 调试：记录所有内存读取（在0x080057fc附近）
        if 0x080057f0 <= pc <= 0x08005810:
            logger.debug(f"内存读取: PC=0x{pc:08x}, Addr=0x{address:08x}, Size={size}")
        self._record_watch_memory_event(pc, address, size, is_write=False)

        occurrence_constraint_key = (pc, address, memory_occurrence)
        memory_constraint_source = ""
        if (
            memory_occurrence > 0
            and occurrence_constraint_key in self.memory_occurrence_constraints
        ):
            memory_constraint = self.memory_occurrence_constraints[
                occurrence_constraint_key
            ]
            memory_constraint_source = "occurrence_memory_constraint"
        else:
            dynamic_memory_constraint = (
                self.dynamic_memory_read_constraints.get((pc, address))
                if self.enable_dynamic_memory_constraints
                else None
            )
            memory_constraint = self._evaluate_dynamic_memory_constraint(
                uc,
                dynamic_memory_constraint,
                size,
            )
            if memory_constraint is not None:
                memory_constraint_source = "dynamic_memory_constraint"
        if memory_constraint is None:
            memory_constraint = self.memory_read_constraints.get((pc, address))
            if memory_constraint is not None:
                memory_constraint_source = "memory_constraint"
        if memory_constraint is not None:
            guard_reason = self._memory_constraint_safety_reason({
                "type": "memory",
                "read_pc": pc,
                "address": address,
                "value": int(memory_constraint) & 0xFFFFFFFF,
            })
            if guard_reason is not None:
                self._record_memory_constraint_guard(
                    "read_hook",
                    pc,
                    address,
                    int(memory_constraint) & 0xFFFFFFFF,
                    guard_reason,
                )
                self.memory_read_constraints.pop((pc, address), None)
                self.dynamic_memory_read_constraints.pop((pc, address), None)
                self.memory_occurrence_constraints.pop(
                    occurrence_constraint_key,
                    None,
                )
                memory_constraint = None
        applied_memory_constraint = False
        if memory_constraint is not None:
            try:
                mask = (1 << (size * 8)) - 1
                uc.mem_write(address, (memory_constraint & mask).to_bytes(size, 'little'))
                applied_memory_constraint = True
                logger.debug(
                    "命中内存约束: PC=0x%08x Addr=0x%08x Value=0x%08x",
                    pc,
                    address,
                    memory_constraint & mask,
                )
            except Exception as e:
                logger.debug(f"应用内存约束失败 @ 0x{address:08x}: {e}")
            if applied_memory_constraint and (int(pc), int(address)) in self.modeled_async_memory_constraints:
                self.modeled_async_memory_stats["applied_reads"] += 1
            memory_key = (int(pc), int(address) & 0xFFFFFFFF)
            if applied_memory_constraint:
                self.memory_constraint_hit_counts[memory_key] += 1
            if applied_memory_constraint and len(self.memory_constraint_hit_history) < 4096:
                self.memory_constraint_hit_history.append({
                    "pc": int(pc) & 0xFFFFFFFF,
                    "address": int(address) & 0xFFFFFFFF,
                    "value": int(memory_constraint) & 0xFFFFFFFF,
                    "occurrence": int(
                        memory_occurrence
                        or self.memory_constraint_hit_counts[memory_key]
                    ),
                    "source": memory_constraint_source or "memory_constraint",
                })

            if applied_memory_constraint and is_mmio_read:
                constrained_value = memory_constraint & ((1 << (size * 8)) - 1)
                self.loop_classifier.record_mmio_access(pc, address, True, constrained_value)
                self._record_mmio_access_history(pc, address, True, constrained_value)
                self._finish_causal_input_read(
                    causal_input_token,
                    value=constrained_value,
                    source=memory_constraint_source or "memory_constraint",
                )
                return

        # 普通 RAM 读也记录给循环分析器。只记录 SRAM 和曾经 unmapped
        # 的外部输入页，避免把取指/常量池访问塞爆历史；这能让非 MMIO
        # 的结构体字段/外部表等待循环可被本地求解。
        if (
            (0x20000000 <= address < 0x20100000 or address in self.external_memory_input_addresses)
            and self.loop_classifier.wants_memory_access_sample(pc, False)
        ):
            self.loop_classifier.record_memory_access(pc, address, False, 0)
            self._record_memory_access_history(pc, address, True, 0)
        elif (
            self.record_mapped_data_accesses
            and self.loop_classifier.wants_memory_access_sample(pc, False)
            and self._is_mapped_non_mmio_data_address(address, size)
        ):
            self.loop_classifier.record_memory_access(pc, address, False, 0)
            self._record_memory_access_history(pc, address, True, 0)

        if is_bitband_alias:
            alias_value = self._apply_bitband_alias_read(address, pc)
            if alias_value is None:
                if causal_input_token is not None:
                    self._finish_causal_input_read(
                        causal_input_token,
                        value=0,
                        source="bitband_alias_unavailable",
                    )
                return
            try:
                mask = (1 << (size * 8)) - 1
                uc.mem_write(
                    address,
                    (int(alias_value) & mask).to_bytes(size, 'little'),
                )
            except Exception as exc:
                logger.debug("写入bit-band alias读值失败 @ 0x%08x: %s", address, exc)
            self._finish_causal_input_read(
                causal_input_token,
                value=int(alias_value),
                source="bitband_alias",
            )
            return

        # 只处理MMIO区域
        if not is_mmio_read:
            if causal_input_token is not None:
                if applied_memory_constraint:
                    observed_value = int(memory_constraint) & (
                        (1 << (max(1, int(size or 1)) * 8)) - 1
                    )
                    observed_source = memory_constraint_source or "memory_constraint"
                else:
                    try:
                        observed_value = int.from_bytes(
                            bytes(uc.mem_read(address, max(1, int(size or 1)))),
                            "little",
                        )
                    except Exception:
                        observed_value = 0
                    observed_source = "mapped_external_memory"
                self._finish_causal_input_read(
                    causal_input_token,
                    value=observed_value,
                    source=observed_source,
                )
            return

        mmio_value = self.mmio_handler.handle_read(address, pc, size)
        try:
            mask = (1 << (size * 8)) - 1
            uc.mem_write(address, (mmio_value & mask).to_bytes(size, 'little'))
        except Exception as e:
            logger.debug(f"写入已映射MMIO值失败 @ 0x{address:08x}: {e}")

        # 记录到循环分类器
        self.loop_classifier.record_mmio_access(pc, address, True, mmio_value)

        # 关键：也要记录到mmio_access_history！
        self._record_mmio_access_history(pc, address, True, mmio_value)
        self._finish_causal_input_read(
            causal_input_token,
            value=mmio_value,
            source="stateful_mmio",
        )

        # 调试日志
        logger.debug(f"MMIO读取(已映射): PC=0x{pc:08x}, Addr=0x{address:08x}, Value=0x{mmio_value:08x}")

    def _is_external_input_memory_range(self, address: int, size: int = 1) -> bool:
        """Return whether a concrete RAM read falls inside a modeled input buffer."""
        start = int(address) & 0xFFFFFFFF
        read_size = max(1, int(size or 1))
        end = start + read_size
        if start in set(getattr(self, "external_memory_input_addresses", set()) or set()):
            return True
        range_sources = (
            list(getattr(self, "stream_input_payload_writes", []) or []),
            list(getattr(self, "stream_input_injected_ranges", []) or []),
        )
        for records in range_sources:
            for item in records:
                if not isinstance(item, dict):
                    continue
                range_start = self._parse_int(item.get("address"))
                try:
                    range_size = int(item.get("size", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if range_start is None or range_size <= 0:
                    continue
                if int(range_start) <= start and end <= int(range_start) + range_size:
                    return True
        return False

    def mem_write_hook(self, uc, access, address, size, value, user_data):
        """
        Observe mapped SRAM/MMIO writes for loop classification.

        Unmapped writes are handled by mmio_unmapped_hook. This hook catches
        writes after a peripheral page has been mapped, and ordinary SRAM
        writes that distinguish finite init/scan loops from state waits.
        """
        try:
            pc = int(uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
        except Exception:
            pc = 0

        address = int(address) & 0xFFFFFFFF
        value = int(value or 0) & 0xFFFFFFFF
        is_bitband_alias = self._decode_bitband_alias(address) is not None
        if is_bitband_alias and not self._ensure_bitband_access_mapped(
            address,
            size,
            origin="memory",
        ):
            return
        self._record_watch_memory_event(pc, address, size, is_write=True, value=value)

        # Alias addresses fall inside the broad 0x20000000-0x3fffffff data
        # interval.  Handle them before the ordinary SRAM fast path so a page
        # that was mapped by a deferred retry still preserves bit-band
        # semantics.
        if is_bitband_alias:
            self.mapped_memory_write_stats["bitband_alias_writes"] += 1
            if self._apply_bitband_alias_write(address, pc, value):
                return

        if 0x20000000 <= address < 0x40000000:
            self.mapped_memory_write_stats["sram_writes"] += 1
            self._mark_runtime_written_pages(address, size)
            if self.loop_classifier.wants_memory_access_sample(pc, True):
                self.loop_classifier.record_memory_access(pc, address, True, value)
            return

        if self._is_mmio_address(address):
            self.mapped_memory_write_stats["mmio_writes"] += 1
            if self._apply_bitband_alias_write(address, pc, value):
                return
            self.mmio_handler.handle_write(address, pc, value, size)
            self._record_mmio_access_history(pc, address, False, value)
            self.loop_classifier.record_mmio_access(pc, address, False, value)
            return

        self.mapped_memory_write_stats["other_writes"] += 1
        self._mark_runtime_written_pages(address, size)
        if address in self.external_memory_input_addresses:
            if self.loop_classifier.wants_memory_access_sample(pc, True):
                self.loop_classifier.record_memory_access(pc, address, True, value)
                self._record_memory_access_history(pc, address, False, value)
        elif (
            self.record_mapped_data_accesses
            and self.loop_classifier.wants_memory_access_sample(pc, True)
            and self._is_mapped_non_mmio_data_address(address, size)
        ):
            self.loop_classifier.record_memory_access(pc, address, True, value)

    def _convert_to_static_dict(self, mmio_constraints):
        """转换为StatefulMMIOHandler需要的格式"""
        if not mmio_constraints:
            return {}

        result = {}
        for key, value in mmio_constraints.items():
            if isinstance(key, tuple) and len(key) == 2:
                result[key] = value
        return result

    def _parse_int(self, value):
        """解析十六进制或十进制整数"""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        try:
            return int(text, 16) if text.lower().startswith('0x') else int(text)
        except ValueError:
            return None

    def _normalize_dynamic_memory_rule(self, rule) -> Optional[Dict[str, object]]:
        if not isinstance(rule, dict):
            return None
        kind = str(rule.get("kind", "") or "")
        if kind != "self_loop_exit":
            return None
        branch_mnemonic = self._normalize_mnemonic(rule.get("branch_mnemonic", ""))
        compare_mnemonic = self._normalize_mnemonic(rule.get("compare_mnemonic", ""))
        if branch_mnemonic not in {"BEQ", "BNE", "BGT", "BLT", "BGE", "BLE", "BHI", "BLO", "BHS", "BLS"}:
            return None
        if compare_mnemonic not in {"CMP", "CMN", "TST", "TEQ"}:
            return None

        width = self._parse_int(rule.get("width"))
        width = width if width in {1, 2, 4} else 4
        normalized = {
            "kind": kind,
            "branch_pc": self._parse_int(rule.get("branch_pc")),
            "branch_mnemonic": branch_mnemonic,
            "compare_pc": self._parse_int(rule.get("compare_pc")),
            "compare_mnemonic": compare_mnemonic,
            "rhs_operand": str(rule.get("rhs_operand", "") or ""),
            "source_reg": self._normalize_register_name(rule.get("source_reg")),
            "source_mask": self._parse_int(rule.get("source_mask")),
            "immediate": self._parse_int(rule.get("immediate")),
            "width": width,
        }
        if normalized["immediate"] is None and normalized["source_reg"] is None:
            return None
        return normalized

    def _serialize_dynamic_memory_rule(self, rule) -> Optional[Dict[str, object]]:
        normalized = self._normalize_dynamic_memory_rule(rule)
        if normalized is None:
            return None
        return {
            "kind": normalized["kind"],
            "branch_pc": self._format_optional_hex(normalized.get("branch_pc")),
            "branch_mnemonic": normalized.get("branch_mnemonic"),
            "compare_pc": self._format_optional_hex(normalized.get("compare_pc")),
            "compare_mnemonic": normalized.get("compare_mnemonic"),
            "rhs_operand": normalized.get("rhs_operand", ""),
            "source_reg": normalized.get("source_reg"),
            "source_mask": self._format_optional_hex(normalized.get("source_mask")),
            "immediate": self._format_optional_hex(normalized.get("immediate")),
            "width": int(normalized.get("width") or 4),
        }

    def _normalize_register_name(self, value) -> Optional[str]:
        text = str(value or "").strip().lower()
        if re.fullmatch(r"r(?:[0-9]|1[0-2])", text):
            return text
        if text in {"sp", "lr", "pc"}:
            return text
        return None

    def _read_register_by_name(self, uc, reg_name: str) -> Optional[int]:
        reg_map = {
            'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1, 'r2': UC_ARM_REG_R2,
            'r3': UC_ARM_REG_R3, 'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
            'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7, 'r8': UC_ARM_REG_R8,
            'r9': UC_ARM_REG_R9, 'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
            'r12': UC_ARM_REG_R12, 'sp': UC_ARM_REG_SP, 'lr': UC_ARM_REG_LR,
            'pc': UC_ARM_REG_PC,
        }
        reg_id = reg_map.get(str(reg_name or "").lower())
        if reg_id is None:
            return None
        try:
            return int(uc.reg_read(reg_id)) & 0xFFFFFFFF
        except Exception:
            return None

    def _evaluate_dynamic_memory_constraint(self, uc, rule, read_size: int) -> Optional[int]:
        normalized = self._normalize_dynamic_memory_rule(rule)
        if normalized is None:
            return None

        rhs_value = normalized.get("immediate")
        if rhs_value is None:
            reg_value = self._read_register_by_name(uc, str(normalized.get("source_reg") or ""))
            if reg_value is None:
                return None
            source_mask = normalized.get("source_mask")
            rhs_value = reg_value & int(source_mask if source_mask is not None else 0xFFFFFFFF)

        width = int(normalized.get("width") or read_size or 4)
        width = width if width in {1, 2, 4} else max(1, min(4, int(read_size or 4)))
        mask = (1 << (width * 8)) - 1
        rhs_value = int(rhs_value) & mask
        branch_mnemonic = str(normalized.get("branch_mnemonic") or "")
        compare_mnemonic = str(normalized.get("compare_mnemonic") or "")

        if compare_mnemonic in {"CMP", "CMN"}:
            if branch_mnemonic == "BEQ":
                return (rhs_value + 1) & mask
            if branch_mnemonic == "BNE":
                return rhs_value
            if branch_mnemonic in {"BGT", "BHI"}:
                return rhs_value
            if branch_mnemonic in {"BLT", "BLO"}:
                return rhs_value
            if branch_mnemonic in {"BGE", "BHS"}:
                return (rhs_value - 1) & mask
            if branch_mnemonic in {"BLE", "BLS"}:
                return (rhs_value + 1) & mask
            return None

        if compare_mnemonic == "TST":
            if branch_mnemonic == "BEQ":
                return rhs_value
            if branch_mnemonic == "BNE":
                return 0
        return None

    def _validated_dynamic_memory_rule(
        self,
        rule,
        constraint: Dict[str, object],
        *,
        source: str,
    ) -> Optional[Dict[str, object]]:
        """
        Keep dynamic memory rules only when they agree with the concrete local
        solver at the current state.

        Dynamic rules are useful for waits whose threshold is an independent
        register, but they are unsafe when the RHS register is itself derived
        from the pending memory load.  At UC_HOOK_MEM_READ time that register
        still contains a stale value, which can overwrite a correct concrete
        constraint with zero.  Agreement with the concrete solver is a cheap
        semantic guard and turns unsafe rules back into read-PC scoped fixed
        constraints.
        """
        normalized = self._normalize_dynamic_memory_rule(rule)
        if normalized is None:
            return None

        read_pc = self._parse_int(constraint.get("read_pc"))
        read_size = self._constraint_write_size(read_pc)
        dynamic_value = self._evaluate_dynamic_memory_constraint(self.uc, normalized, read_size)
        if dynamic_value is None:
            logger.debug("%s动态内存规则缺少当前可求值上下文，降级为固定约束: %s", source, normalized)
            return None

        expected_value = self._parse_int(constraint.get("value"))
        if expected_value is None:
            return None
        mask = (1 << (max(1, min(4, read_size)) * 8)) - 1
        if (int(dynamic_value) & mask) != (int(expected_value) & mask):
            logger.warning(
                "%s动态内存规则与本地求解值不一致，降级为固定约束: read_pc=%s addr=%s local=0x%08x dynamic=0x%08x rule=%s",
                source,
                self._format_optional_hex(read_pc),
                self._format_optional_hex(constraint.get("address")),
                int(expected_value) & mask,
                int(dynamic_value) & mask,
                normalized,
            )
            return None

        constraint["value"] = int(dynamic_value) & mask
        return normalized

    def _load_constraints_from_json(self, constraint_json_path):
        """加载持久化约束，格式兼容 constraints 列表和旧式地址映射"""
        if not constraint_json_path:
            return {}

        try:
            import os
            if not os.path.exists(constraint_json_path):
                return {}

            with open(constraint_json_path, 'r') as f:
                data = json.load(f)

            constraints = {}
            if isinstance(data, dict) and "constraints" in data:
                for item in data.get("constraints", []):
                    constraint_type = item.get("type")
                    if constraint_type == "mmio":
                        # The EnhancedMMIOHandler replay overlay enforces
                        # dynamic occurrence scope.  The primary static map is
                        # only (read_pc,address)-scoped and must not broaden it.
                        if self._parse_int(item.get("read_occurrence")) is not None:
                            continue
                        if not self._should_load_file_constraint(item):
                            continue
                        normalized = self.code_analyzer.normalize_constraint({
                            "type": "mmio",
                            "address": item.get("address"),
                            "value": item.get("value"),
                            "read_pc": item.get("read_pc") if item.get("read_pc") is not None else item.get("pc"),
                            "constraint_pc": item.get("constraint_pc"),
                            "description": item.get("description", ""),
                        })
                        if normalized is None:
                            logger.debug(f"忽略无效持久化MMIO约束: {item}")
                            continue
                        if not self._should_load_semantic_mmio_constraint(normalized):
                            continue
                        pc = self._parse_int(normalized.get("read_pc")) or 0
                        constraints[(pc, normalized["address"])] = normalized["value"]
                        continue

                    if constraint_type != "memory":
                        continue
                    if not self._should_load_file_constraint(item):
                        continue

                    raw_memory_constraint = {
                        "type": "memory",
                        "address": item.get("address"),
                        "value": item.get("value"),
                        "read_pc": item.get("read_pc") if item.get("read_pc") is not None else item.get("pc"),
                        "constraint_pc": item.get("constraint_pc"),
                        "description": item.get("description", ""),
                        "dynamic": item.get("dynamic"),
                        "external_memory": item.get("external_memory", False),
                        "modeled_async_state": item.get("modeled_async_state", False),
                        "persisted_modeled_async_state": item.get("modeled_async_state", False),
                    }
                    normalized = self._normalize_runtime_constraint(raw_memory_constraint)
                    if normalized is None:
                        read_pc = self._parse_int(raw_memory_constraint.get("read_pc"))
                        if read_pc is not None and read_pc not in self.code_analyzer.instruction_index:
                            self.pending_persisted_memory_constraints.append(raw_memory_constraint)
                            logger.debug(f"挂起动态read_pc内存约束，等待运行时BB补齐: {item}")
                            continue
                        logger.debug(f"忽略无效持久化memory约束: {item}")
                        continue
                    self._install_persisted_memory_constraint(normalized, raw_memory_constraint)
            elif isinstance(data, dict):
                for address_text, value_text in data.items():
                    address = self._parse_int(address_text)
                    value = self._parse_int(value_text)
                    if address is not None and value is not None:
                        constraints[(0, address)] = value

            if constraints:
                logger.info(f"加载持久化约束: {len(constraints)} 条")
            if self.memory_read_constraints:
                logger.info(f"加载持久化内存约束: {len(self.memory_read_constraints)} 条")
            if self.dynamic_memory_read_constraints:
                logger.info(f"加载动态内存约束: {len(self.dynamic_memory_read_constraints)} 条")
            if self.pending_persisted_memory_constraints:
                logger.info(f"挂起动态BB内存约束: {len(self.pending_persisted_memory_constraints)} 条")
            return constraints
        except Exception as e:
            logger.error(f"加载持久化约束失败: {e}")
            return {}

    def _install_persisted_memory_constraint(self, normalized: Dict, raw_constraint: Dict) -> bool:
        read_pc = self._parse_int(normalized.get("read_pc"))
        if read_pc is None:
            return False
        address = int(normalized["address"]) & 0xFFFFFFFF
        self.memory_read_constraints[(read_pc, address)] = int(normalized["value"]) & 0xFFFFFFFF
        if bool(normalized.get("modeled_async_state")):
            key = (int(read_pc), address)
            if key not in self.modeled_async_memory_constraints:
                self.modeled_async_memory_constraints.add(key)
                self.modeled_async_memory_stats["installed"] += 1
        dynamic_rule = (
            self._normalize_dynamic_memory_rule(raw_constraint.get("dynamic"))
            if self.enable_dynamic_memory_constraints
            else None
        )
        if dynamic_rule is not None:
            self.dynamic_memory_read_constraints[(read_pc, address)] = dynamic_rule
        return True

    def _activate_pending_persisted_memory_constraints(self) -> int:
        if not self.pending_persisted_memory_constraints:
            return 0
        remaining: List[Dict[str, object]] = []
        activated = 0
        for raw_constraint in self.pending_persisted_memory_constraints:
            read_pc = self._parse_int(raw_constraint.get("read_pc"))
            if read_pc is None or read_pc not in self.code_analyzer.instruction_index:
                remaining.append(raw_constraint)
                continue
            normalized = self._normalize_runtime_constraint(raw_constraint)
            if normalized is None:
                logger.debug("丢弃仍无法归一化的挂起memory约束: %s", raw_constraint)
                continue
            if self._install_persisted_memory_constraint(normalized, raw_constraint):
                activated += 1
        self.pending_persisted_memory_constraints = remaining
        if activated:
            logger.info("激活运行时BB持久化内存约束: %d 条", activated)
        return activated

    def _should_load_semantic_mmio_constraint(self, constraint: Dict) -> bool:
        """
        Reject stale persisted MMIO constraints only when local loop semantics
        can prove the stored value cannot exit the polling loop.
        """
        read_pc = self._parse_int(constraint.get("read_pc"))
        address = self._parse_int(constraint.get("address"))
        value = self._parse_int(constraint.get("value"))
        if read_pc is None or address is None or value is None:
            return True
        loop_bb = self.instruction_to_bb.get(read_pc)
        if loop_bb is None:
            return True
        try:
            accepted = self._validate_loop_exit_mmio_value(
                [int(loop_bb)],
                int(loop_bb),
                int(address),
                int(value),
                read_pc=read_pc,
                constraint_pc=constraint.get("constraint_pc"),
                allow_runtime_operands=False,
            )
        except Exception as exc:
            logger.debug("持久化MMIO约束语义校验跳过 @ 0x%08x: %s", read_pc, exc)
            return True
        if not accepted:
            logger.warning(
                "拒绝过期持久化MMIO约束: read_pc=0x%08x address=0x%08x value=0x%08x",
                read_pc,
                int(address) & 0xFFFFFFFF,
                int(value) & 0xFFFFFFFF,
            )
            return False
        return True

    @staticmethod
    def _normalize_branch_mmio_file_mode(mode: Optional[str]) -> Optional[str]:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        return normalized or None

    def _resolve_branch_mmio_file_mode(self, explicit_mode: Optional[str] = None) -> str:
        normalized = self._normalize_branch_mmio_file_mode(explicit_mode)
        if normalized is not None:
            return normalized
        mode = os.environ.get("LSGEMU_LOAD_BRANCH_MMIO_FILE_CONSTRAINTS", "").strip().lower()
        return mode or ""

    def _should_load_file_constraint(self, item: Dict) -> bool:
        added_by = str(item.get("added_by", ""))
        speculation_level = str(item.get("speculation_level", "") or "")
        mode = self._resolve_branch_mmio_file_mode(self.branch_mmio_file_mode)
        if mode in {"unscoped-global", "global-ablation"}:
            return bool(
                speculation_level != "dynamic_replay_required"
                and added_by not in {"branch_mmio", "intelligent_emulator"}
                and item.get("read_occurrence") in (None, "", 0, "0")
            )
        if speculation_level == "dynamic_replay_required":
            if mode in {"1", "true", "yes", "on", "all", "scoped", "pc", "read-pc"}:
                return item.get("read_pc") is not None or item.get("pc") is not None
            return False
        if added_by != "branch_mmio":
            return True
        if mode in {"1", "true", "yes", "on", "all"}:
            return True
        if mode in {"0", "false", "no", "off", "none"}:
            return False
        if mode in {"scoped", "pc", "read-pc"}:
            return item.get("read_pc") is not None or item.get("pc") is not None
        # 默认仅加载按 read_pc 作用域命中的 branch_mmio 约束，避免地址级全局污染。
        return item.get("read_pc") is not None or item.get("pc") is not None

    @staticmethod
    def _page_down(value: int) -> int:
        return int(value) & ~0xFFF

    @staticmethod
    def _page_up(value: int) -> int:
        return (int(value) + 0xFFF) & ~0xFFF

    @staticmethod
    def _is_mmio_address(address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        return (
            0x40000000 <= address < 0x60000000
            or 0xE0000000 <= address < 0xE0100000
        )

    @staticmethod
    def _decode_bitband_alias(address: int) -> Optional[Tuple[int, int]]:
        """Decode Cortex-M bit-band alias addresses to backing byte + bit.

        Peripheral aliases such as 0x42470000 are not independent registers:
        writing 0/1 to the alias word clears/sets one bit in 0x40000000-
        backed MMIO. If the emulator only maps the alias page, clock/status
        initialization observes stale backing registers and can produce false
        terminal paths.
        """
        address = int(address) & 0xFFFFFFFF
        if 0x22000000 <= address < 0x24000000:
            alias_base = 0x22000000
            region_base = 0x20000000
        elif 0x42000000 <= address < 0x44000000:
            alias_base = 0x42000000
            region_base = 0x40000000
        else:
            return None
        offset = address - alias_base
        byte_offset = offset // 32
        bit_number = (offset % 32) // 4
        if bit_number < 0 or bit_number > 7:
            return None
        return (region_base + byte_offset) & 0xFFFFFFFF, int(bit_number)

    def _bitband_mapping_ranges(
        self,
        address: int,
        size: int = 1,
    ) -> List[Tuple[int, int]]:
        """Return the alias and backing ranges required by one bit-band access."""
        decoded = self._decode_bitband_alias(address)
        if decoded is None:
            return []
        target, _bit_number = decoded
        return [
            (int(address) & 0xFFFFFFFF, max(1, int(size or 1))),
            (int(target) & ~0x3, 4),
        ]

    def _ensure_bitband_access_mapped(
        self,
        address: int,
        size: int = 1,
        *,
        origin: str = "unknown",
        retry_block_callback: bool = False,
        verify_native: bool = False,
    ) -> bool:
        """Preflight both sides of a bit-band access before changing state."""
        ranges = self._bitband_mapping_ranges(address, size)
        if not ranges:
            return True
        return self._ensure_memory_ranges_mapped(
            ranges,
            origin=origin,
            retry_block_callback=retry_block_callback,
            verify_native=verify_native,
        )

    def _read_backing_word_for_bitband(self, address: int, pc: int) -> int:
        address = int(address) & 0xFFFFFFFF
        word_addr = address & ~0x3
        if self._is_mmio_address(word_addr):
            try:
                modeled = self.mmio_handler.peek_register_value(word_addr, 4)
                if modeled is not None:
                    return int(modeled) & 0xFFFFFFFF
            except Exception:
                pass
            # A mapped peripheral page is only transport storage for Unicorn;
            # it is not the peripheral model.  Ask the stateful handler for an
            # uninitialized backing register so status profiles and scoped
            # overlays remain effective after a deferred map.
            return int(self.mmio_handler.handle_read(word_addr, pc, 4)) & 0xFFFFFFFF
        try:
            if self._is_memory_mapped(word_addr, 4):
                return int.from_bytes(bytes(self.uc.mem_read(word_addr, 4)), "little") & 0xFFFFFFFF
        except Exception:
            pass
        return 0

    def _apply_bitband_alias_read(self, address: int, pc: int) -> Optional[int]:
        decoded = self._decode_bitband_alias(address)
        if decoded is None:
            return None
        if not self._ensure_bitband_access_mapped(
            address,
            1,
            origin="memory",
        ):
            return None
        target, bit_number = decoded
        word_addr = target & ~0x3
        word_bit = ((target - word_addr) * 8) + bit_number
        backing = self._read_backing_word_for_bitband(target, pc)
        alias_value = (backing >> word_bit) & 1
        if self._is_mmio_address(target):
            self.loop_classifier.record_mmio_access(pc, word_addr, True, backing)
            self._record_mmio_access_history(pc, word_addr, True, backing)
        else:
            self.loop_classifier.record_memory_access(pc, word_addr, False, backing)
            self._record_memory_access_history(pc, word_addr, True, backing)
        return alias_value

    def _apply_bitband_alias_write(self, address: int, pc: int, value: int) -> bool:
        decoded = self._decode_bitband_alias(address)
        if decoded is None:
            return False
        if not self._ensure_bitband_access_mapped(
            address,
            1,
            origin="memory",
        ):
            return False
        target, bit_number = decoded
        word_addr = target & ~0x3
        word_bit = ((target - word_addr) * 8) + bit_number
        backing = self._read_backing_word_for_bitband(target, pc)
        if int(value or 0) & 1:
            backing |= 1 << word_bit
        else:
            backing &= ~(1 << word_bit)
        backing &= 0xFFFFFFFF
        try:
            self.uc.mem_write(word_addr, backing.to_bytes(4, "little"))
        except Exception:
            # Do not publish the modeled side effect when the backing write did
            # not complete.  A deferred map is a BaseException and therefore
            # still propagates to the safe-point retry path.
            return False
        if self._is_mmio_address(target):
            self.mmio_handler.handle_write(word_addr, pc, backing, 4)
            self.loop_classifier.record_mmio_access(pc, word_addr, False, backing)
            self._record_mmio_access_history(pc, word_addr, False, backing)
        else:
            self.loop_classifier.record_memory_access(pc, word_addr, True, backing)
            self._record_memory_access_history(pc, word_addr, False, backing)
        return True

    def _is_mapped_non_mmio_data_address(self, address: int, size: int = 1) -> bool:
        """True for mapped ordinary memory, including raw-loader low RAM aliases."""
        address = int(address) & 0xFFFFFFFF
        if self._is_mmio_address(address):
            return False
        if not self._contains_mapped_range(address, max(1, int(size or 1))):
            return False
        return True

    def _mark_runtime_written_pages(self, address: int, size: int = 1) -> None:
        address = int(address) & 0xFFFFFFFF
        start = self._page_down(address)
        end = self._page_up(address + max(1, int(size or 1)))
        candidate_pages = range(start, max(start + 0x1000, end), 0x1000)
        if all(page in self.runtime_written_pages for page in candidate_pages):
            return
        if not self._is_mapped_non_mmio_data_address(address, size):
            return
        for page in range(start, max(start + 0x1000, end), 0x1000):
            self.runtime_written_pages.add(page)

    def _dynamic_code_provenance(self, address: int) -> Optional[str]:
        """Require executable provenance before treating dynamic bytes as BBs."""
        address = int(address) & ~1
        if not self._contains_mapped_range(address, 1):
            return None
        try:
            file_size = os.path.getsize(self.firmware_path)
        except Exception:
            file_size = int(getattr(self.arch_info, "code_size", 0) or 0)

        if not getattr(self, "is_elf_firmware", False):
            raw_ranges = [(int(self.base_addr), int(self.base_addr) + file_size)]
            if self.raw_load_base is not None:
                raw_ranges.append((int(self.raw_load_base), int(self.raw_load_base) + file_size))
            if any(start <= address < end for start, end in raw_ranges):
                return "raw_image"
        else:
            for segment in self._load_segments():
                if not (int(segment.get("flags", 0) or 0) & 0x1):
                    continue
                start = int(segment.get("vaddr", 0) or 0)
                end = start + max(int(segment.get("filesz", 0) or 0), int(segment.get("memsz", 0) or 0))
                if start <= address < end:
                    return "elf_executable_segment"

        if self._page_down(address) in self.runtime_written_pages:
            return "runtime_written_page"
        return None

    @staticmethod
    def _is_external_rom_address(address: int) -> bool:
        address = int(address) & ~1
        return 0x1FFF0000 <= address < 0x20000000

    def _summarize_external_rom_call(self, address: int) -> bool:
        """Model vendor/system ROM calls as opaque functions returning via LR."""
        if not self.summarize_external_rom_calls:
            return False
        normalized = int(address) & ~1
        if not self._is_external_rom_address(normalized):
            return False
        # Runtime-written SRAM code should still go through dynamic-code
        # provenance, not this opaque ROM model.
        if self._page_down(normalized) in self.runtime_written_pages:
            return False
        self.external_rom_call_stats["attempted"] += 1
        self.external_rom_call_targets[normalized] += 1
        try:
            lr = int(self.uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF
            return_pc = lr & ~1
            if lr <= 1 or not self._is_executable_address(return_pc):
                self.external_rom_call_stats["failed"] += 1
                return False
            self.uc.reg_write(UC_ARM_REG_R0, 0)
            self.uc.reg_write(
                UC_ARM_REG_PC,
                (return_pc | 1) if self.execution_thumb else return_pc,
            )
            self.external_rom_call_stats["returned"] += 1
            self._external_rom_summary_target = normalized
            logger.debug(
                "外部ROM调用摘要: target=0x%08x return=0x%08x",
                normalized,
                return_pc,
            )
            return True
        except Exception as exc:
            self.external_rom_call_stats["failed"] += 1
            logger.debug("外部ROM调用摘要失败 @ 0x%08x: %s", normalized, exc)
            return False

    def _load_segments(self) -> List[Dict[str, int]]:
        if self._load_segment_cache is not None:
            return self._load_segment_cache

        segments: List[Dict[str, int]] = []
        try:
            from elftools.elf.elffile import ELFFile

            with open(self.firmware_path, 'rb') as f:
                elf = ELFFile(f)
                for segment in elf.iter_segments():
                    if segment['p_type'] != 'PT_LOAD':
                        continue
                    vaddr = int(segment['p_vaddr'])
                    filesz = int(segment['p_filesz'])
                    memsz = int(segment['p_memsz'])
                    flags = int(segment['p_flags'])
                    if max(filesz, memsz) <= 0:
                        continue
                    segments.append({
                        'vaddr': vaddr,
                        'paddr': int(segment['p_paddr']),
                        'filesz': filesz,
                        'memsz': memsz,
                        'flags': flags,
                    })
        except Exception as exc:
            logger.debug(f"读取ELF加载段失败: {exc}")
        self._load_segment_cache = segments
        return segments

    @staticmethod
    def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        ordered = sorted((int(start), int(end)) for start, end in ranges if int(end) > int(start))
        merged: List[Tuple[int, int]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _contains_mapped_range(self, start: int, size: int) -> bool:
        end = int(start) + max(1, int(size))
        return any(int(region_start) <= int(start) and end <= int(region_end)
                   for region_start, region_end in self.mapped_ranges)

    def _is_executable_address(self, address: int) -> bool:
        target = int(address) & ~1
        if target in self.instruction_to_bb or target in self.static_bbs:
            return True
        for segment in self._load_segments():
            if not (int(segment.get('flags', 0)) & 0x1):
                continue
            start = int(segment['vaddr'])
            end = start + max(int(segment.get('filesz', 0)), int(segment.get('memsz', 0)))
            if start <= target < end:
                return True
        return False

    def _apply_cortex_m_cpu_profile(self) -> CpuProfile:
        """MCLASS 模式下按推断型号 ctl_set_cpu_model 精确切换 CPU。

        语义提示（详见 lsgemu.cpu_profile 模块注释）：
        - M4/M7/M33 之间普通代码无差；SP 浮点编码互相兼容（M33=FPv5-SP
          超集 M4F=VFPv4-SP），覆盖率场景无影响；
        - M0/M0+ 无 Thumb-2、无硬件除法——若推断为 M0 且固件疑似含 32 位
          Thumb-2 指令，打 warning 建议用 LSGEMU_FORCE_CPU_MODEL=cortex-m3；
        - 任何失败（旧 unicorn 绑定无 ctl API/常量、ctl 调用异常）都回退
          MCLASS 默认 cortex-m33 并告警，保证行为不劣于旧版。
        """
        profile = resolve_cortex_m_cpu(self.firmware_path)
        ctl_set_cpu_model = getattr(self.uc, "ctl_set_cpu_model", None)
        fallback_reason = None
        if profile.cpu_model_id is None:
            fallback_reason = (
                f"当前 unicorn 绑定缺少常量 {profile.unicorn_constant_name}"
            )
        elif not callable(ctl_set_cpu_model):
            fallback_reason = "当前 unicorn 绑定缺少 ctl_set_cpu_model API"
        if fallback_reason is None:
            try:
                ctl_set_cpu_model(int(profile.cpu_model_id))
            except Exception as exc:
                fallback_reason = f"ctl_set_cpu_model({profile.unicorn_constant_name}) 失败: {exc}"
        if fallback_reason is not None:
            logger.warning(
                "Cortex-M CPU 型号切换不可用（%s），回退 MCLASS 默认 cortex-m33；"
                "原推断 %s (source=%s)",
                fallback_reason,
                profile.unicorn_constant_name,
                profile.source,
            )
            return CpuProfile(
                unicorn_constant_name=CORTEX_M_FALLBACK_CONSTANT,
                cpu_model_id=unicorn_cpu_model_id(CORTEX_M_FALLBACK_CONSTANT),
                source="fallback",
                reason=(
                    f"{fallback_reason}，回退 MCLASS 默认 cortex-m33；"
                    f"原推断 {profile.unicorn_constant_name} (source={profile.source}: {profile.reason})"
                ),
            )
        logger.info(
            "Cortex-M CPU 型号: %s (id=%s, source=%s)",
            profile.unicorn_constant_name,
            hex(int(profile.cpu_model_id)),
            profile.source,
        )
        if is_armv6m_profile(profile):
            thumb2_prefixes = count_thumb2_halfword_prefixes(self.firmware_path)
            if thumb2_prefixes:
                logger.warning(
                    "推断型号 %s 为 ARMv6-M（无 Thumb-2/硬件除法），但固件疑似含 "
                    "%d 个 32 位 Thumb-2 前缀半字；如遇 UC_ERR_INSN_INVALID 建议 "
                    "LSGEMU_FORCE_CPU_MODEL=cortex-m3",
                    profile.unicorn_constant_name,
                    thumb2_prefixes,
                )
        return profile

    def _infer_thumb_execution_mode(self) -> bool:
        if self.execution_thumb_override is not None:
            return bool(self.execution_thumb_override)
        entry = int(getattr(self.arch_info, "entry_point", 0) or 0)
        if entry & 1:
            return True
        # Raw BIN files in this corpus often begin with ARM reset branches
        # (for example ea00001e encoded little-endian as 1e0000ea). Running
        # them as Thumb immediately produces invalid-instruction failures.
        if not getattr(self, "is_elf_firmware", False):
            try:
                with open(self.firmware_path, "rb") as f:
                    first_word = f.read(4)
                if len(first_word) == 4:
                    little = int.from_bytes(first_word, "little")
                    big = int.from_bytes(first_word, "big")
                    if (little >> 24) in {0xEA, 0xEB} or (big >> 24) in {0xEA, 0xEB}:
                        return False
            except Exception:
                pass
            # If Ghidra/static analysis found mostly Thumb-looking addresses,
            # callers can still override by using an odd explicit entry point.
            return False
        # ELF firmware: an odd entry (bit0 set) means Thumb; an even entry is
        # an ARM-state entry. Defaulting every ELF to Thumb silently
        # misdecodes valid ARM-state images, so default to ARM state and let
        # callers override via an odd explicit entry point when needed.
        return bool(entry & 1)

    @staticmethod
    def _plausible_stack_pointer(value: int) -> bool:
        value = int(value) & 0xFFFFFFFF
        if value & 0x3:
            return False
        return (
            0x1FFF0000 <= value < 0x30000000
            or 0x10000000 <= value < 0x11000000
        )

    def _read_firmware_word(self, address: int) -> Optional[int]:
        target = int(address) & 0xFFFFFFFF
        try:
            data = self.uc.mem_read(target, 4)
            return int.from_bytes(data, 'little') & 0xFFFFFFFF
        except Exception:
            pass
        return None

    def _detect_vector_table_base(self) -> int:
        candidates: List[int] = []
        for segment in self._load_segments():
            candidates.append(int(segment['vaddr']))
        if self.base_addr not in candidates:
            candidates.append(int(self.base_addr))

        for base in candidates:
            initial_sp = self._read_firmware_word(base)
            reset = self._read_firmware_word(base + 4)
            if initial_sp is None or reset is None:
                continue
            if self._plausible_stack_pointer(initial_sp) and self._is_executable_address(reset):
                return int(base)

        # fast path 全部失败（典型：ELF 的 PT_LOAD p_offset=0，文件头被映射
        # 进 flash，在段 vaddr 读到 ELF 魔数当 SP）。改为对已加载镜像做
        # 全量向量表扫描，取达到置信阈值的最高分候选。
        scanned = self._full_scan_vector_table_base()
        if scanned is not None:
            return scanned
        return int(self.base_addr)

    def _vector_table_scan_regions(self) -> List[Tuple[int, bytes]]:
        """收集向量表全扫描用的 (区域基址, 镜像字节) 列表。

        ELF 固件扫描每个可执行 PT_LOAD 在 uc 中已写入的内容（基址=段
        vaddr）；raw BIN 固件扫描整个文件（基址=base_addr）。
        """
        regions: List[Tuple[int, bytes]] = []
        if getattr(self, "is_elf_firmware", False):
            for segment in self._load_segments():
                if not (int(segment.get('flags', 0)) & 0x1):
                    continue
                vaddr = int(segment['vaddr'])
                filesz = int(segment.get('filesz', 0) or 0)
                if filesz < 8:
                    continue
                try:
                    data = bytes(self.uc.mem_read(vaddr, filesz))
                except Exception as exc:
                    logger.debug(f"向量表扫描读取段失败 @ 0x{vaddr:08x}: {exc}")
                    continue
                regions.append((vaddr, data))
        else:
            try:
                with open(self.firmware_path, "rb") as f:
                    data = f.read()
            except Exception as exc:
                logger.debug(f"向量表扫描读取BIN失败: {exc}")
                return regions
            if len(data) >= 8:
                regions.append((int(self.base_addr), data))
        return regions

    def _full_scan_vector_table_base(self) -> Optional[int]:
        """向量表全镜像扫描回退（fast path 失败时才调用）。

        使用 lsgemu.vector_table_locator 纯函数模块对各扫描区域打分，
        只有达到置信阈值（CONFIDENT_MIN_SCORE）的最高分候选才被接受。
        检测结果记录到 self.vector_table_detection 供报告输出。
        """
        from ..vector_table_locator import CONFIDENT_MIN_SCORE, scan_vector_table

        best = None
        for region_base, data in self._vector_table_scan_regions():
            try:
                region_candidates = scan_vector_table(data, load_base_hint=region_base)
            except Exception as exc:
                logger.debug(f"向量表全扫描异常 @ 0x{region_base:08x}: {exc}")
                continue
            if not region_candidates:
                continue
            top = region_candidates[0]
            if best is None or (-top.score, (top.reset_pc & ~1) - top.load_base) < (
                -best.score,
                (best.reset_pc & ~1) - best.load_base,
            ):
                best = top

        if best is None:
            return None
        detection: Dict[str, object] = {
            "method": "full_scan",
            "accepted": bool(best.confident),
            "score": float(best.score),
            "signals": list(best.signals),
            "offset": int(best.offset),
            "load_base": int(best.load_base),
            "vector_table_base": int(best.vector_table_address),
            "initial_sp": int(best.initial_sp),
            "reset_pc": int(best.reset_pc),
            "confidence_threshold": float(CONFIDENT_MIN_SCORE),
        }
        self.vector_table_detection = detection
        if best.confident:
            logger.info(
                "✓ 向量表全扫描定位: base=0x%08x SP=0x%08x Reset=0x%08x score=%.3f signals=%s",
                best.vector_table_address,
                best.initial_sp,
                best.reset_pc,
                best.score,
                ",".join(best.signals),
            )
            return int(best.vector_table_address)
        logger.info(
            "向量表全扫描最高分 %.3f 低于置信阈值 %.2f（offset=0x%x），维持默认基址",
            best.score,
            CONFIDENT_MIN_SCORE,
            best.offset,
        )
        return None

    def _refresh_snapshot_regions(self, writable_ranges: List[Tuple[int, int]]) -> None:
        ram_ranges = list(writable_ranges)
        ram_ranges.extend([
            (0x1FFF0000, 0x20000000),
            (0x20000000, 0x20100000),
            (0x10000000, 0x10040000),
        ])
        filtered: List[Tuple[int, int]] = []
        for start, end in self._merge_ranges(ram_ranges):
            if end <= start:
                continue
            if start < 0x10000000 or start >= 0x40000000:
                continue
            # Snapshot RAM/state, not large flash aliases.
            if 0x08000000 <= start < 0x10000000:
                continue
            filtered.append((start, min(end - start, 0x200000)))
        self.ram_snapshot_regions = filtered or [(0x20000000, 0x100000)]
        self.snapshot_manager.memory_regions = list(self.ram_snapshot_regions)
        if hasattr(self.branch_snapshot_manager, "set_memory_regions"):
            self.branch_snapshot_manager.set_memory_regions(self.ram_snapshot_regions)

    def setup_memory(self):
        """设置内存映射，按ELF加载段和常见Cortex-M SRAM窗口精确覆盖。"""
        ranges: List[Tuple[int, int]] = []
        writable_ranges: List[Tuple[int, int]] = []
        load_segments = self._load_segments()

        for segment in load_segments:
            start = self._page_down(segment['vaddr'])
            end = self._page_up(segment['vaddr'] + max(segment.get('memsz', 0), segment.get('filesz', 0), 1))
            ranges.append((start, end))
            paddr = int(segment.get('paddr', segment['vaddr']) or 0)
            filesz = int(segment.get('filesz', 0) or 0)
            if filesz > 0 and paddr and paddr != int(segment['vaddr']):
                # Some MCU ELFs keep .data VMA in RAM but startup copies it
                # from flash LMA/p_paddr. Map that load image too; otherwise
                # reset code overwrites the correctly loaded RAM .data with
                # zeros from an unmapped/empty flash hole.
                ranges.append((self._page_down(paddr), self._page_up(paddr + filesz)))
            if int(segment.get('flags', 0)) & 0x2:
                writable_ranges.append((start, end))

        if not ranges:
            if self.base_addr == 0x08000000:
                ranges.extend([(0x08000000, 0x09000000), (0x00000000, 0x00004000)])
            elif self.base_addr == 0x00000000:
                ranges.append((0x00000000, 0x01000000))
            else:
                page_start = self._page_down(self.base_addr)
                ranges.append((page_start, page_start + 0x01000000))

        if not getattr(self, "is_elf_firmware", False) and self.raw_load_base is not None:
            try:
                file_size = os.path.getsize(self.firmware_path)
            except Exception:
                file_size = int(getattr(self.arch_info, "code_size", 0) or 0)
            if file_size > 0:
                ranges.append((self._page_down(self.raw_load_base), self._page_up(self.raw_load_base + file_size)))

        # Common MCU RAM windows. PT_LOAD ranges cover exact data/bss; these
        # windows cover stacks/heaps declared only through vector-table SP.
        ranges.extend([
            (0x1FFF0000, 0x20000000),
            (0x20000000, 0x20100000),
            (0x10000000, 0x10040000),
        ])
        writable_ranges.extend([
            (0x1FFF0000, 0x20000000),
            (0x20000000, 0x20100000),
            (0x10000000, 0x10040000),
        ])

        # Static MMIO recovery gives us a cheap, high-confidence prefetch of
        # the peripheral pages that the image actually references.  Mapping
        # these pages before the first ``emu_start`` keeps the common mapped
        # MMIO path fast and avoids entering the deferred fault path for every
        # first access.  Unresolved/base-only accesses remain lazy and are
        # handled by the same safe-point mechanism at runtime.
        predicted_mmio_pages: Set[int] = set()
        for item in getattr(self, "static_mmio_accesses", []) or []:
            if not isinstance(item, Mapping):
                continue
            predicted = self._parse_int(item.get("address"))
            if predicted is None or not self._is_mmio_address(predicted):
                continue
            predicted_mmio_pages.add(self._page_down(int(predicted)))
        ranges.extend(
            (page, page + 0x1000) for page in sorted(predicted_mmio_pages)
        )

        # Cortex-M usually aliases the boot flash/vector table at 0x00000000.
        if self.base_addr != 0 and not any(start <= 0 < end for start, end in ranges):
            ranges.append((0x00000000, 0x00004000))

        self.mapped_ranges = []
        for start, end in self._merge_ranges(ranges):
            try:
                self.uc.mem_map(start, end - start)
                self.mapped_ranges.append((start, end))
            except Exception as exc:
                logger.debug(f"内存映射跳过 0x{start:08x}-0x{end:08x}: {exc}")

        self._refresh_snapshot_regions(writable_ranges)
        logger.info("✓ 内存映射完成: %d ranges", len(self.mapped_ranges))

    def load_firmware(self):
        """加载固件"""
        if not getattr(self, "is_elf_firmware", False):
            with open(self.firmware_path, "rb") as f:
                data = f.read()
            if data:
                if not self._contains_mapped_range(self.base_addr, len(data)):
                    start = self._page_down(self.base_addr)
                    end = self._page_up(self.base_addr + len(data))
                    self.uc.mem_map(start, end - start)
                    self.mapped_ranges = self._merge_ranges(self.mapped_ranges + [(start, end)])
                self.uc.mem_write(self.base_addr, data)
                if self.raw_load_base is not None:
                    if not self._contains_mapped_range(self.raw_load_base, len(data)):
                        start = self._page_down(self.raw_load_base)
                        end = self._page_up(self.raw_load_base + len(data))
                        self.uc.mem_map(start, end - start)
                        self.mapped_ranges = self._merge_ranges(self.mapped_ranges + [(start, end)])
                    self.uc.mem_write(self.raw_load_base, data)
                    logger.info(f"✓ BIN运行基址别名: 0x{self.raw_load_base:08x}")

            self.vector_table_base = self._detect_vector_table_base()
            if self.vector_table_base != 0 and self._contains_mapped_range(0, 0x400):
                try:
                    vector_data = self.uc.mem_read(self.vector_table_base, 0x400)
                    self.uc.mem_write(0, bytes(vector_data))
                except Exception as exc:
                    logger.debug(f"写入0地址向量别名失败: {exc}")

            initial_sp = self._read_firmware_word(self.vector_table_base)
            if initial_sp is not None and self._plausible_stack_pointer(initial_sp):
                self.uc.reg_write(UC_ARM_REG_SP, initial_sp)
                logger.info(f"✓ 设置初始SP: 0x{initial_sp:08x}")
            else:
                default_sp = 0x20005000
                self.uc.reg_write(UC_ARM_REG_SP, default_sp)
                if initial_sp is None:
                    logger.warning(f"无法读取BIN初始SP，使用默认值 0x{default_sp:08x}")
                else:
                    logger.warning(f"BIN初始SP值异常: 0x{initial_sp:08x}，使用默认值 0x{default_sp:08x}")

            logger.info("✓ BIN固件加载完成")
            return

        from elftools.elf.elffile import ELFFile

        with open(self.firmware_path, 'rb') as f:
            elf = ELFFile(f)
            for segment in elf.iter_segments():
                if segment['p_type'] == 'PT_LOAD':
                    vaddr = segment['p_vaddr']
                    paddr = int(segment['p_paddr'])
                    data = segment.data()
                    if len(data) > 0:
                        if not self._contains_mapped_range(vaddr, len(data)):
                            start = self._page_down(vaddr)
                            end = self._page_up(vaddr + len(data))
                            self.uc.mem_map(start, end - start)
                            self.mapped_ranges = self._merge_ranges(self.mapped_ranges + [(start, end)])
                        self.uc.mem_write(vaddr, data)
                        if paddr and paddr != int(vaddr):
                            if not self._contains_mapped_range(paddr, len(data)):
                                start = self._page_down(paddr)
                                end = self._page_up(paddr + len(data))
                                self.uc.mem_map(start, end - start)
                                self.mapped_ranges = self._merge_ranges(self.mapped_ranges + [(start, end)])
                            self.uc.mem_write(paddr, data)

        self.vector_table_base = self._detect_vector_table_base()
        if self.vector_table_base != 0 and self._contains_mapped_range(0, 0x400):
            try:
                vector_data = self.uc.mem_read(self.vector_table_base, 0x400)
                self.uc.mem_write(0, bytes(vector_data))
            except Exception as exc:
                logger.debug(f"写入0地址向量别名失败: {exc}")

        # 重要：设置初始SP
        # ARM Cortex-M的向量表第一个条目是初始SP值
        try:
            # 尝试从向量表读取初始SP
            initial_sp_bytes = self.uc.mem_read(self.vector_table_base, 4)
            initial_sp = int.from_bytes(initial_sp_bytes, 'little')

            # 检查SP是否在合理SRAM范围内
            if self._plausible_stack_pointer(initial_sp):
                self.uc.reg_write(UC_ARM_REG_SP, initial_sp)
                logger.info(f"✓ 设置初始SP: 0x{initial_sp:08x}")
            else:
                logger.warning(f"初始SP值异常: 0x{initial_sp:08x}，使用默认值")
                self.uc.reg_write(UC_ARM_REG_SP, 0x20005000)
        except Exception as e:
            logger.warning(f"无法读取初始SP: {e}，使用默认值")
            self.uc.reg_write(UC_ARM_REG_SP, 0x20005000)

        logger.info("✓ 固件加载完成")

    def register_hooks(self):
        """注册hooks"""
        register_primary_mmio_handler(self.uc, self.mmio_handler)
        self._add_owned_hook(UC_HOOK_BLOCK, self.bb_hook, None)
        self._add_owned_hook(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED,
                             self.mmio_unmapped_hook, None)

        # 添加普通内存读取hook，用于捕获已映射的MMIO访问
        self._add_owned_hook(UC_HOOK_MEM_READ, self.mmio_read_hook, None)
        if self.mapped_write_hook_enabled:
            self._add_owned_hook(UC_HOOK_MEM_WRITE, self.mem_write_hook, None)
        # MCLASS（cortex-m33）原生执行 MSR/MRS/CPSID/MRC 等 v7-M 系统指令后，
        # 软件 hook 若仍注册会在原生执行之后再改写一遍寄存器和 PC（双重写），
        # 因此原生模式可用时跳过注册；仅当回退到 A15（旧 unicorn 无
        # UC_MODE_MCLASS，或 LSGEMU_DISABLE_MCLASS=1）时才依赖软件模型兜底。
        if self.cortex_m_system_instruction_map and not self.cortex_m_native_mclass_enabled:
            self._add_owned_hook(UC_HOOK_CODE, self._cortex_m_system_instruction_hook, None)
        if self.handle_svc_as_noop:
            self._add_owned_hook(UC_HOOK_CODE, self._svc_instruction_hook, None)
        if self.execution_thumb:
            self._add_owned_hook(UC_HOOK_CODE, self._thumb_state_guard_hook, None)
        if self.thumb_indirect_branch_instruction_map:
            self._add_owned_hook(UC_HOOK_CODE, self._thumb_indirect_branch_target_hook, None)
        if self.mapped_mmio_load_instruction_map:
            self._add_owned_hook(UC_HOOK_CODE, self._mapped_mmio_preload_hook, None)
        if self.branch_entry_pc_to_bb:
            self.branch_entry_code_hook = self._add_owned_hook(
                UC_HOOK_CODE,
                self._branch_entry_instruction_hook,
                None,
            )
        self.runtime_loop_branch_force_hook = self._add_owned_hook(
            UC_HOOK_CODE,
            self._runtime_loop_branch_force_hook,
            None,
        )
        self._add_owned_hook(UC_HOOK_CODE, self._stop_before_pc_hook, None)
        self._add_owned_hook(UC_HOOK_CODE, self._skip_function_return_hook, None)
        if self.watch_pcs:
            self._add_owned_hook(UC_HOOK_CODE, self._watch_pc_hook, None)

        logger.info("✓ Hooks注册完成")

    def _watch_pc_hook(self, uc, address, size, user_data):
        self._record_watch_pc_event(uc, address, size)

    def _stop_before_pc_hook(self, uc, address, size, user_data):
        target = self.stop_before_pc
        if target is None:
            return
        if (int(address) & ~1) == (int(target) & ~1):
            self.stop_requested_reason = "stop_before_pc"
            uc.emu_stop()

    def _skip_function_return_hook(self, uc, address, size, user_data):
        normalized_address = int(address) & ~1
        rule = self.skip_function_returns.get(normalized_address)
        if not rule:
            return
        try:
            rule_kind = str(rule.get("kind") or "")
            if rule_kind == "monotonic_time":
                state = self.skip_function_state.setdefault(
                    normalized_address,
                    {
                        "value": int(rule.get("start", 0) or 0) & 0xFFFFFFFF,
                        "calls": 0,
                    },
                )
                increment = int(rule.get("increment", 1000) or 1000) & 0xFFFFFFFF
                state["calls"] = int(state.get("calls", 0) or 0) + 1
                state["value"] = (int(state.get("value", 0) or 0) + increment) & 0xFFFFFFFF
                return_value = int(state["value"]) & 0xFFFFFFFF
                self.skip_function_stats["stateful_applied"] = (
                    int(self.skip_function_stats.get("stateful_applied", 0) or 0) + 1
                )
            elif rule_kind in {"memcpy", "memmove", "memset"}:
                return_value = self._apply_memory_intrinsic_summary(uc, rule_kind)
            elif rule_kind in {"malloc", "calloc", "realloc", "free", "sbrk"}:
                return_value = self._apply_allocator_summary(uc, rule)
            elif rule_kind in {"stream_status", "stream_byte", "stream_read", "stream_status_read", "riot_msg_receive"}:
                return_value = self._apply_stream_input_summary(uc, rule, normalized_address)
            elif rule_kind in {"zephyr_clock_subsys_rate", "zephyr_device_get_binding"}:
                return_value = self._apply_zephyr_summary(uc, rule)
            else:
                return_value = int(rule.get("return_value", 0) or 0) & 0xFFFFFFFF
            uc.reg_write(UC_ARM_REG_R0, return_value)
            lr = int(uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF
            return_pc = lr & ~1
            if lr <= 1:
                return
            if not self._is_executable_address(return_pc) and self._page_down(return_pc) not in self.runtime_written_pages:
                self.skip_function_stats["invalid_lr"] = int(self.skip_function_stats.get("invalid_lr", 0) or 0) + 1
                logger.debug(
                    "函数快进跳过: entry=0x%08x LR=0x%08x 不可执行且无动态代码证明",
                    normalized_address,
                    lr,
                )
                return
            uc.reg_write(UC_ARM_REG_PC, (return_pc | 1) if self.execution_thumb else return_pc)
            bb_addr = self.instruction_to_bb.get(normalized_address)
            if bb_addr is not None:
                self.skipped_function_entry_bbs.add(int(bb_addr))
            symbol = str(rule.get("symbol") or f"0x{normalized_address:08x}")
            self.skip_function_stats["applied"] = int(self.skip_function_stats.get("applied", 0) or 0) + 1
            by_symbol = self.skip_function_stats.setdefault("by_symbol", {})
            by_symbol[symbol] = int(by_symbol.get(symbol, 0) or 0) + 1
            applied_entry_bbs = self.skip_function_stats.setdefault("applied_entry_bbs", [])
            if isinstance(applied_entry_bbs, list) and len(applied_entry_bbs) < 256:
                applied_entry_bbs.append(f"0x{int(bb_addr if bb_addr is not None else address) & ~1:08x}")
        except Exception as exc:
            logger.debug(f"函数快进失败 @ 0x{int(address):08x}: {exc}")

    def _apply_symbol_summary_return(self, address: int, symbol: str) -> bool:
        """Apply the same safe function-summary semantics during loop intervention.

        Some setup functions are repeatedly entered from an outer finite loop
        (for example CMSIS/NVIC configuration helpers). The loop detector may
        observe the callee entry as a hot loop head before the generic code hook
        gets a chance to summarize it. In that case we should return through LR
        instead of treating the function body as an MMIO polling loop.
        """
        normalized_address = int(address) & ~1
        symbol = str(symbol or "")
        if not symbol:
            return False

        rule = self.skip_function_returns.get(normalized_address)
        if rule is None:
            return_value = self._default_skip_return_for_symbol(symbol)
            if return_value is None:
                return False
            rule = {
                "symbol": symbol,
                "return_value": int(return_value) & 0xFFFFFFFF,
            }

        rule_kind = str(rule.get("kind") or "")
        # Intervention-time summaries must be side-effect conservative. Rich
        # summaries (stream, allocator, memcpy, time) already run at function
        # entry through _skip_function_return_hook and may need ABI-specific
        # state updates, so only scalar return summaries are applied here.
        if rule_kind:
            return False

        try:
            lr = int(self.uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF
            return_pc = lr & ~1
            if lr <= 1 or not self._is_executable_address(return_pc):
                logger.debug(
                    "配置/API函数摘要跳过: %s @ 0x%08x LR=0x%08x 不可执行",
                    symbol,
                    normalized_address,
                    lr,
                )
                return False

            return_value = int(rule.get("return_value", 0) or 0) & 0xFFFFFFFF
            self.uc.reg_write(UC_ARM_REG_R0, return_value)
            self.uc.reg_write(UC_ARM_REG_PC, (return_pc | 1) if self.execution_thumb else return_pc)

            bb_addr = self.instruction_to_bb.get(normalized_address, normalized_address)
            self.skipped_function_entry_bbs.add(int(bb_addr) & ~1)
            self.skip_function_stats["applied"] = int(self.skip_function_stats.get("applied", 0) or 0) + 1
            by_symbol = self.skip_function_stats.setdefault("by_symbol", {})
            by_symbol[symbol] = int(by_symbol.get(symbol, 0) or 0) + 1
            applied_entry_bbs = self.skip_function_stats.setdefault("applied_entry_bbs", [])
            if isinstance(applied_entry_bbs, list) and len(applied_entry_bbs) < 256:
                applied_entry_bbs.append(f"0x{int(bb_addr) & ~1:08x}")

            logger.info(
                "配置/API函数摘要返回: %s @ 0x%08x -> LR 0x%08x, R0=0x%08x",
                symbol,
                normalized_address,
                return_pc,
                return_value,
            )
            return True
        except Exception as exc:
            logger.debug(
                "配置/API函数摘要失败: %s @ 0x%08x: %s",
                symbol,
                normalized_address,
                exc,
            )
            return False

    def _save_summary_entry_snapshot_if_needed(self, address: int, kind: str) -> None:
        """Save entry-derived state at summary-mode input APIs for later input replay."""
        normalized_address = int(address) & ~1
        bb_addr = self.instruction_to_bb.get(normalized_address, normalized_address)
        bb_addr = int(bb_addr) & ~1
        existing_summary_orders = set()
        if self.branch_snapshot_manager.has_snapshot(bb_addr):
            try:
                existing = self.branch_snapshot_manager.get_snapshot(bb_addr)
                existing_condition = str(getattr(existing, "condition", "") or "").upper()
                if existing_condition.startswith("SUMMARY"):
                    existing_summary_orders.add(int(getattr(existing, "capture_order", 0) or 0))
            except Exception:
                pass
        try:
            for existing in self.branch_snapshot_manager.snapshot_history.get(bb_addr, []) or []:
                existing_condition = str(getattr(existing, "condition", "") or "").upper()
                if existing_condition.startswith("SUMMARY"):
                    existing_summary_orders.add(int(getattr(existing, "capture_order", 0) or 0))
        except Exception:
            pass
        try:
            max_summary_variants = max(
                1,
                int(os.environ.get("LSGEMU_SUMMARY_ENTRY_SNAPSHOT_HISTORY_PER_ADDRESS", "8") or 8),
            )
        except ValueError:
            max_summary_variants = 8
        if len(existing_summary_orders) >= max_summary_variants:
            return
        try:
            lr = int(self.uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF
        except Exception:
            lr = 0
        predicted_occurrence = self.branch_snapshot_manager.occurrence_counts.get(bb_addr, 0) + 1
        try:
            snapshot = self.branch_snapshot_manager.save_snapshot(
                self.uc,
                bb_addr,
                lr & ~1,
                lr & ~1,
                f"SUMMARY:{kind}",
                original_taken=True,
                depth=self._bb_history_depth(),
                mmio_state=self._current_mmio_state(),
                alternatives=[lr & ~1] if lr else [],
                original_index=0,
                occurrence_index=predicted_occurrence,
                update_current=not self.branch_snapshot_manager.has_snapshot(bb_addr),
            )
            try:
                history = self.branch_snapshot_manager.snapshot_history.setdefault(bb_addr, [])
                if all(
                    int(getattr(existing, "capture_order", -1) or -1) != int(getattr(snapshot, "capture_order", -2) or -2)
                    for existing in history
                ):
                    history.append(snapshot)
                history.sort(key=lambda item: int(getattr(item, "capture_order", getattr(item, "order", 0)) or 0))
                while len([
                    item for item in history
                    if str(getattr(item, "condition", "") or "").upper().startswith("SUMMARY")
                ]) > max_summary_variants:
                    for index, item in enumerate(history):
                        if str(getattr(item, "condition", "") or "").upper().startswith("SUMMARY"):
                            history.pop(index)
                            break
                    else:
                        break
            except Exception:
                pass
            logger.debug("保存summary入口快照 @ 0x%08x (%s)", bb_addr, kind)
        except Exception as exc:
            logger.debug("保存summary入口快照失败 @ 0x%08x: %s", bb_addr, exc)

    def _apply_memory_intrinsic_summary(self, uc, kind: str) -> int:
        dest = int(uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
        src_or_value = int(uc.reg_read(UC_ARM_REG_R1)) & 0xFFFFFFFF
        size = int(uc.reg_read(UC_ARM_REG_R2)) & 0xFFFFFFFF
        try:
            max_copy = max(0, int(os.environ.get("LSGEMU_MEMORY_INTRINSIC_MAX_BYTES", "65536")))
        except ValueError:
            max_copy = 65536
        bounded_size = min(size, max_copy)
        if bounded_size <= 0:
            return dest

        try:
            if kind == "memset":
                # Preflight the complete destination before committing any
                # bytes.  A deferred mapping must restart the summary as one
                # unit, rather than leaving a partially materialized memset.
                if not self._ensure_memory_ranges_mapped([(dest, bounded_size)]):
                    return dest
                uc.mem_write(dest, bytes([src_or_value & 0xFF]) * bounded_size)
                return dest

            ranges = [(dest, bounded_size)]
            source_is_mapped = self._is_memory_mapped(src_or_value, bounded_size)
            ranges.append((src_or_value, bounded_size))
            try:
                if not self._ensure_memory_ranges_mapped(ranges):
                    return dest
            except _DeferredMemoryMapping:
                if not source_is_mapped:
                    self._mark_pending_external_memory_address(src_or_value)
                raise
            if not source_is_mapped:
                # External source buffers are valid replay inputs even when the
                # ELF memory map does not declare them. Publish that fact only
                # after the mapping precondition has completed.
                self.external_memory_input_addresses.add(src_or_value & 0xFFFFFFFF)
            data = bytes(uc.mem_read(src_or_value, bounded_size))
            if kind == "memmove" and dest > src_or_value and dest < src_or_value + bounded_size:
                data = bytes(data)
            uc.mem_write(dest, data)
        except Exception as exc:
            logger.debug(
                "内存内建函数摘要失败: kind=%s dest=0x%08x src/value=0x%08x size=0x%x err=%s",
                kind,
                dest,
                src_or_value,
                size,
                exc,
            )
        return dest

    def _try_lzo1x_decompress_summary(self, bb_start_addr: int) -> bool:
        """Summarize reached LZO1X-style decompressor functions.

        Honeywell scanner images spend most reset-entry time in a concrete
        decompressor before reaching boot/update command code. This summary is
        deliberately narrow: it only fires for the exact ABI and instruction
        prologue shape observed in the firmware, then materializes the decoded
        bytes in Unicorn memory and returns through LR. It does not credit the
        skipped decompressor body as covered.
        """
        if not self.enable_lzo_decompress_summary:
            return False
        if not self._is_lzo1x_decompress_entry(bb_start_addr):
            return False

        src = int(self.uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
        src_len = int(self.uc.reg_read(UC_ARM_REG_R1)) & 0xFFFFFFFF
        dst = int(self.uc.reg_read(UC_ARM_REG_R2)) & 0xFFFFFFFF
        out_len_ptr = int(self.uc.reg_read(UC_ARM_REG_R3)) & 0xFFFFFFFF
        lr = int(self.uc.reg_read(UC_ARM_REG_LR)) & 0xFFFFFFFF

        try:
            max_input = max(1, int(os.environ.get("LSGEMU_LZO_SUMMARY_MAX_INPUT", "16777216")))
            max_output = max(1, int(os.environ.get("LSGEMU_LZO_SUMMARY_MAX_OUTPUT", "67108864")))
        except ValueError:
            max_input = 16 * 1024 * 1024
            max_output = 64 * 1024 * 1024

        if src_len <= 0 or src_len > max_input:
            return False
        if lr <= 1:
            return False
        if not self._is_memory_mapped(src, src_len):
            return False

        try:
            compressed = bytes(self.uc.mem_read(src, src_len))
            decoded, consumed = self._lzo1x_decompress_safe(compressed, max_output=max_output)
            trailing = compressed[consumed:] if 0 <= consumed <= src_len else b""
            if consumed != src_len and not (
                0 < len(trailing) <= 16 and all(byte == 0 for byte in trailing)
            ):
                self.lzo_decompress_summary_stats["failed"] = (
                    int(self.lzo_decompress_summary_stats.get("failed", 0) or 0) + 1
                )
                logger.debug(
                    "LZO摘要拒绝: entry=0x%08x consumed=0x%x input=0x%x",
                    int(bb_start_addr),
                    int(consumed),
                    int(src_len),
                )
                return False
            output_ranges = []
            if decoded:
                output_ranges.append((dst, len(decoded)))
            if out_len_ptr:
                output_ranges.append((out_len_ptr, 4))
            if output_ranges and not self._ensure_memory_ranges_mapped(output_ranges):
                return False
            if decoded:
                self.uc.mem_write(dst, decoded)
                self._mark_runtime_written_range(dst, len(decoded))
                dump_dir = os.environ.get("LSGEMU_LZO_DUMP_DIR", "").strip()
                if dump_dir:
                    try:
                        os.makedirs(dump_dir, exist_ok=True)
                        stem = os.path.splitext(os.path.basename(self.firmware_path))[0]
                        tag = f"{stem}_0x{int(bb_start_addr) & ~1:08x}_0x{dst:08x}"
                        with open(os.path.join(dump_dir, f"{tag}_compressed.bin"), "wb") as f:
                            f.write(compressed)
                        with open(os.path.join(dump_dir, f"{tag}_decoded.bin"), "wb") as f:
                            f.write(decoded)
                    except Exception as dump_exc:
                        logger.debug("LZO摘要dump失败: %s", dump_exc)
            if out_len_ptr:
                self.uc.mem_write(out_len_ptr, int(len(decoded)).to_bytes(4, "little"))
            # The firmware's decompressor writes output length before returning
            # both success and input-length mismatch errors. Preserve that
            # return-code behavior; many callers immediately overwrite r0, but
            # keeping it faithful avoids hiding real validation paths.
            return_value = 0 if consumed == src_len else 0xFFFFFFF8
            self.uc.reg_write(UC_ARM_REG_R0, return_value)
            next_pc = (lr | 1) if self.execution_thumb else (lr & ~1)
            self.uc.reg_write(UC_ARM_REG_PC, next_pc)

            self.skipped_function_entry_bbs.add(int(bb_start_addr) & ~1)
            stats = self.lzo_decompress_summary_stats
            stats["applied"] = int(stats.get("applied", 0) or 0) + 1
            stats["bytes_in"] = int(stats.get("bytes_in", 0) or 0) + int(src_len)
            stats["bytes_out"] = int(stats.get("bytes_out", 0) or 0) + len(decoded)
            entries = stats.setdefault("entry_bbs", [])
            if isinstance(entries, list) and len(entries) < 128:
                entries.append(f"0x{int(bb_start_addr) & ~1:08x}")
            logger.info(
                "✓ LZO1X解压摘要: entry=0x%08x src=0x%08x in=0x%x dst=0x%08x out=0x%x lr=0x%08x",
                int(bb_start_addr) & ~1,
                src,
                src_len,
                dst,
                len(decoded),
                lr & ~1,
            )
            return True
        except Exception as exc:
            self.lzo_decompress_summary_stats["failed"] = (
                int(self.lzo_decompress_summary_stats.get("failed", 0) or 0) + 1
            )
            logger.debug("LZO1X解压摘要失败 @ 0x%08x: %s", int(bb_start_addr), exc)
            return False

    def _mark_runtime_written_range(self, address: int, size: int) -> None:
        if size <= 0:
            return
        page = self._page_down(int(address))
        end = self._page_down(int(address) + int(size) - 1)
        while page <= end:
            self.runtime_written_pages.add(int(page))
            page += 0x1000

    def _is_lzo1x_decompress_entry(self, bb_start_addr: int) -> bool:
        instructions = [dict(insn) for insn in self.static_bbs.get(int(bb_start_addr), []) or []]
        if len(instructions) < 7:
            return False
        tokens = [
            (
                self._normalize_mnemonic(insn.get("mnemonic", "")),
                str(insn.get("operands", "") or "").replace(" ", "").lower(),
            )
            for insn in instructions[:7]
        ]
        expected = [
            ("PUSH", "{r4,r5,r6,lr}"),
            ("ADD", "r4,r0,r1"),
            ("MOV", "r1,#0"),
            ("STR", "r1,[r3]"),
            ("LDRB", "ip,[r0]"),
            ("MOV", "r1,r2"),
            ("CMP", "ip,#0x11"),
        ]
        if tokens != expected:
            return False

        # Avoid matching the output-bounded variant at 0x300eb4; its ABI reads
        # an initial output limit from [r3] before decoding and needs a
        # different safety contract.
        operands = str(instructions[1].get("operands", "") or "").replace(" ", "").lower()
        return operands == "r4,r0,r1"

    def _lzo1x_decompress_safe(self, data: bytes, *, max_output: int) -> Tuple[bytes, int]:
        """Minimal LZO1X decompressor compatible with the reached boot stub.

        The state machine mirrors the ARM implementation at 0x00300c60:

        * 0x00300ca4 starts/extends literal runs;
        * 0x00300d08 handles the special M1-after-literal case;
        * 0x00300d44 dispatches M1/M2/M3/M4 matches.

        Keeping those states explicit avoids a subtle bug where a token read
        after match-trailing literals was incorrectly treated as a fresh
        literal-run token.
        """
        ip = 0
        out = bytearray()
        n = len(data)

        def need_input(count: int) -> None:
            if ip + count > n:
                raise ValueError("input overrun")

        def append_literal(start: int, count: int) -> None:
            if count < 0 or start < 0 or start + count > n:
                raise ValueError("literal input overrun")
            if len(out) + count > max_output:
                raise ValueError("output limit")
            out.extend(data[start:start + count])

        def append_match(offset: int, count: int) -> None:
            if count < 0:
                raise ValueError("negative match length")
            if offset < 0 or offset >= len(out):
                raise ValueError("lookbehind overrun")
            if len(out) + count > max_output:
                raise ValueError("output limit")
            for _ in range(count):
                out.append(out[offset])
                offset += 1

        def read_byte() -> int:
            nonlocal ip
            need_input(1)
            value = data[ip]
            ip += 1
            return value

        def read_len(base: int) -> int:
            nonlocal ip
            value = 0
            while True:
                b = read_byte()
                if b != 0:
                    return value + base + b
                value += 255

        def read_u16() -> int:
            nonlocal ip
            need_input(2)
            value = data[ip] | (data[ip + 1] << 8)
            ip += 2
            return value

        def after_match() -> Tuple[str, int]:
            literal_count = data[ip - 2] & 3
            if literal_count == 0:
                return "literal_start", read_byte()
            append_literal(ip, literal_count)
            next_ip = ip + literal_count
            if next_ip > n:
                raise ValueError("literal input overrun")
            nonlocal_set_ip(next_ip)
            return "match_dispatch", read_byte()

        def nonlocal_set_ip(value: int) -> None:
            nonlocal ip
            ip = value

        if n == 0:
            raise ValueError("empty input")

        t = read_byte()
        if t > 17:
            t -= 17
            if t < 4:
                append_literal(ip, t)
                ip += t
                state = "match_dispatch"
                t = read_byte()
            else:
                append_literal(ip, t)
                ip += t
                state = "read_m1_or_match"
                t = read_byte()
        else:
            state = "literal_start"

        while True:
            if state == "literal_start":
                if t >= 16:
                    state = "match_dispatch"
                    continue
                if t == 0:
                    t = read_len(15)
                t += 3
                append_literal(ip, t)
                ip += t
                t = read_byte()
                state = "read_m1_or_match"
                continue

            if state == "read_m1_or_match":
                if t >= 16:
                    state = "match_dispatch"
                    continue
                m_pos = len(out) - 1 - 0x0800 - (t >> 2) - (read_byte() << 2)
                append_match(m_pos, 3)
                state, t = after_match()
                continue

            if state == "match_dispatch":
                if t >= 64:
                    m_pos = len(out) - 1 - ((t >> 2) & 7) - (read_byte() << 3)
                    match_len = (t >> 5) + 1
                elif t >= 32:
                    t &= 31
                    if t == 0:
                        t = read_len(31)
                    m_pos = len(out) - 1 - (read_u16() >> 2)
                    match_len = t + 2
                elif t >= 16:
                    m_pos = len(out) - ((t & 8) << 11)
                    t &= 7
                    if t == 0:
                        t = read_len(7)
                    m_pos -= read_u16() >> 2
                    if m_pos == len(out):
                        return bytes(out), ip
                    m_pos -= 0x4000
                    match_len = t + 2
                else:
                    m_pos = len(out) - 1 - (t >> 2) - (read_byte() << 2)
                    match_len = 2

                append_match(m_pos, match_len)
                state, t = after_match()
                continue

            raise ValueError(f"invalid lzo state {state!r}")

    def _normalized_model_heap_allocation(
        self,
        size: int,
        *,
        with_chunk_header: bool = True,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute an allocator request without changing allocator state.

        Summary callbacks use this pure preflight before producing payloads or
        advancing the model cursor.  That makes a deferred page-map retry
        idempotent and keeps the heap model aligned with the concrete write.
        The tuple is ``(chunk, user_ptr, end, aligned_size)``.
        """
        try:
            max_alloc = max(
                1,
                int(os.environ.get("LSGEMU_MODEL_HEAP_MAX_ALLOC", "262144")),
            )
        except ValueError:
            max_alloc = 262144
        try:
            requested = int(size or 1)
        except (TypeError, ValueError, OverflowError):
            requested = 1
        requested = max(1, min(requested, max_alloc))
        aligned = (requested + 7) & ~7
        header_size = 8 if with_chunk_header else 0
        chunk = (int(self.model_heap_next) + 7) & ~7
        ptr = chunk + header_size
        end = ptr + aligned
        if end > int(self.model_heap_end):
            return None
        return chunk, ptr, end, aligned

    def _model_heap_alloc(self, size: int, *, zero: bool = False, with_chunk_header: bool = True) -> int:
        allocation = self._normalized_model_heap_allocation(
            size,
            with_chunk_header=with_chunk_header,
        )
        if allocation is None:
            return 0
        chunk, ptr, end, aligned = allocation
        if not self._ensure_memory_mapped(chunk, end - chunk):
            return 0
        if with_chunk_header:
            try:
                # newlib malloc/free reads the size word at user_ptr - 4.
                # Store a positive aligned size so _free_r can execute the
                # concrete header checks instead of faulting on an artificial
                # pointer or negative previous-chunk marker.
                self.uc.mem_write(chunk, b"\x00" * header_size)
                self.uc.mem_write(ptr - 4, int(aligned | 1).to_bytes(4, "little"))
            except Exception:
                return 0
        if zero:
            try:
                self.uc.mem_write(ptr, b"\x00" * aligned)
            except Exception:
                return 0
        self.model_heap_next = end
        self.model_heap_allocations[ptr] = aligned
        return ptr & 0xFFFFFFFF

    def _model_sbrk(self, increment: int) -> int:
        try:
            max_increment = max(1, int(os.environ.get("LSGEMU_MODEL_SBRK_MAX_INCREMENT", "262144")))
        except ValueError:
            max_increment = 262144
        increment = int(increment or 0)
        if increment <= 0:
            increment = 1
        increment = min(increment, max_increment)
        start = (int(self.model_heap_next) + 7) & ~7
        end = (start + increment + 7) & ~7
        if end > int(self.model_heap_end):
            return 0xFFFFFFFF
        if not self._ensure_memory_mapped(start, max(1, end - start)):
            return 0xFFFFFFFF
        self.model_heap_next = end
        return start & 0xFFFFFFFF

    def _apply_allocator_summary(self, uc, rule: Dict[str, object]) -> int:
        kind = str(rule.get("kind") or "")
        style = str(rule.get("arg_style") or "")
        r0 = int(uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
        r1 = int(uc.reg_read(UC_ARM_REG_R1)) & 0xFFFFFFFF
        r2 = int(uc.reg_read(UC_ARM_REG_R2)) & 0xFFFFFFFF

        if kind == "free":
            return 0
        if kind == "sbrk":
            increment = r1 if style == "reent" else r0
            return self._model_sbrk(increment)
        if kind == "malloc":
            size = r1 if style == "reent" else r0
            return self._model_heap_alloc(size, zero=False)
        if kind == "calloc":
            nmemb, elem_size = (r1, r2) if style == "reent" else (r0, r1)
            total = (int(nmemb or 0) * int(elem_size or 0)) & 0xFFFFFFFF
            return self._model_heap_alloc(total, zero=True)
        if kind == "realloc":
            old_ptr, new_size = (r1, r2) if style == "reent" else (r0, r1)
            new_ptr = self._model_heap_alloc(new_size, zero=False)
            if new_ptr and old_ptr:
                old_size = int(self.model_heap_allocations.get(int(old_ptr), 0) or 0)
                copy_size = min(old_size, int(new_size or 0), int(self.model_heap_allocations.get(new_ptr, 0) or 0))
                if copy_size > 0:
                    try:
                        data = bytes(uc.mem_read(old_ptr, copy_size))
                        uc.mem_write(new_ptr, data)
                    except Exception:
                        pass
            return new_ptr
        return 0

    def _read_c_string(self, address: int, max_len: int = 128) -> str:
        address = int(address) & 0xFFFFFFFF
        if address == 0:
            return ""
        data = bytearray()
        try:
            for offset in range(max(1, int(max_len))):
                byte = bytes(self.uc.mem_read(address + offset, 1))[0]
                if byte == 0:
                    break
                data.append(byte)
        except Exception:
            return ""
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _apply_zephyr_summary(self, uc, rule: Dict[str, object]) -> int:
        kind = str(rule.get("kind") or "")
        if kind == "zephyr_clock_subsys_rate":
            rate_ptr = int(uc.reg_read(UC_ARM_REG_R2)) & 0xFFFFFFFF
            try:
                configured = int(os.environ.get("LSGEMU_ZEPHYR_CLOCK_RATE", "72000000"), 0)
            except ValueError:
                configured = 72000000
            rate = int(configured) & 0xFFFFFFFF
            if rate_ptr:
                try:
                    if self._ensure_memory_mapped(rate_ptr, 4):
                        uc.mem_write(rate_ptr, rate.to_bytes(4, "little"))
                except Exception as exc:
                    logger.debug("Zephyr clock rate写入失败 @ 0x%08x: %s", rate_ptr, exc)
            return 0

        if kind == "zephyr_device_get_binding":
            name_ptr = int(uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
            requested = self._read_c_string(name_ptr)
            device = self._find_zephyr_device_by_name(requested)
            return int(device or 0) & 0xFFFFFFFF

        return 0

    @staticmethod
    def _normalize_device_name(value: str) -> str:
        return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())

    def _load_elf_all_symbols_by_name(self) -> Dict[str, int]:
        """Load all ELF symbols, including data objects such as Zephyr devices."""
        if self._elf_all_symbols_by_name_cache is not None:
            return self._elf_all_symbols_by_name_cache

        symbols: Dict[str, int] = {}
        if not getattr(self, "is_elf_firmware", False):
            self._elf_all_symbols_by_name_cache = symbols
            return symbols
        try:
            from elftools.elf.elffile import ELFFile

            with open(self.firmware_path, "rb") as f:
                elf = ELFFile(f)
                for section in elf.iter_sections():
                    if section.header.get("sh_type") not in {"SHT_SYMTAB", "SHT_DYNSYM"}:
                        continue
                    if not hasattr(section, "iter_symbols"):
                        continue
                    for sym in section.iter_symbols():
                        name = str(sym.name or "")
                        if not name:
                            continue
                        value = int(sym.entry.get("st_value", 0) or 0)
                        if value == 0:
                            continue
                        symbols.setdefault(name, value & 0xFFFFFFFF)
        except Exception as exc:
            logger.debug("读取ELF全量符号失败: %s", exc)
        self._elf_all_symbols_by_name_cache = symbols
        return symbols

    def _iter_zephyr_device_candidates(self) -> List[Tuple[int, str]]:
        candidates_by_addr: Dict[int, str] = {}
        for address, symbol in sorted(getattr(self, "symbols_by_addr", {}).items()):
            name = str(symbol or "")
            normalized = name.lower()
            if normalized.startswith("__device_") or normalized.startswith("device_") or "__device_" in normalized:
                candidates_by_addr.setdefault(int(address) & 0xFFFFFFFF, name)

        for name, address in sorted(self._load_elf_all_symbols_by_name().items()):
            normalized = name.lower()
            if normalized.startswith("__device_") or "__device_" in normalized:
                candidates_by_addr.setdefault(int(address) & 0xFFFFFFFF, name)

        return [(address, symbol) for address, symbol in sorted(candidates_by_addr.items())]

    def _zephyr_device_table_ranges(self) -> List[Tuple[int, int]]:
        symbols = self._load_elf_all_symbols_by_name()
        ranges: List[Tuple[int, int]] = []
        start = symbols.get("__device_start")
        end = symbols.get("__device_end")
        if start is not None and end is not None and int(start) < int(end):
            ranges.append((int(start) & 0xFFFFFFFF, int(end) & 0xFFFFFFFF))

        device_addrs = [
            int(address) & 0xFFFFFFFF
            for address, symbol in self._iter_zephyr_device_candidates()
            if str(symbol or "").lower().startswith("__device_")
            and not any(marker in str(symbol or "") for marker in ("_start", "_end"))
        ]
        if device_addrs:
            lo = min(device_addrs)
            hi = max(device_addrs) + 0x20
            if lo < hi:
                ranges.append((lo, hi))

        unique = []
        seen = set()
        for lo, hi in ranges:
            if hi <= lo or (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            unique.append((lo, hi))
        return unique

    def _plausible_zephyr_device_string(self, value: str) -> bool:
        if not value or len(value) > 80:
            return False
        printable = sum(1 for ch in value if 0x20 <= ord(ch) <= 0x7E)
        if printable != len(value):
            return False
        return any(ch.isalnum() for ch in value)

    def _scan_zephyr_devices_by_name(self) -> Dict[str, List[Tuple[int, str]]]:
        """Discover Zephyr device objects from loaded memory by struct layout.

        Older Zephyr stores `struct device` objects in RAM/data sections.  The
        first word is a pointer to the device name string.  Relying only on text
        symbols makes `device_get_binding()` return NULL in stripped or partially
        loaded runs, so this scan builds a name -> object address map from the
        actual loaded image.
        """
        if self._zephyr_device_name_cache is not None:
            return self._zephyr_device_name_cache

        ranges = self._zephyr_device_table_ranges()
        if not ranges:
            for start, end in self.mapped_ranges:
                # Zephyr device objects in this corpus live in normal SRAM.  Keep
                # the fallback bounded to avoid interpreting large flash regions.
                if start < 0x20100000 and end > 0x20000000:
                    ranges.append((max(start, 0x20000000), min(end, 0x20100000)))
                elif start < 0x10040000 and end > 0x10000000:
                    ranges.append((max(start, 0x10000000), min(end, 0x10040000)))

        discovered: Dict[str, List[Tuple[int, str]]] = {}
        scanned = 0
        max_scan = 2 * 1024 * 1024
        for start, end in ranges:
            start = int(start) & ~0x3
            end = int(end) & ~0x3
            if end <= start:
                continue
            if scanned >= max_scan:
                break
            scan_end = min(end, start + (max_scan - scanned))
            cursor = start
            while cursor + 4 <= scan_end:
                try:
                    name_ptr = int.from_bytes(bytes(self.uc.mem_read(cursor, 4)), "little") & 0xFFFFFFFF
                except Exception:
                    cursor += 4
                    continue
                if not self._contains_mapped_range(name_ptr, 1):
                    cursor += 4
                    continue
                actual = self._read_c_string(name_ptr, max_len=80)
                if not self._plausible_zephyr_device_string(actual):
                    cursor += 4
                    continue
                normalized = self._normalize_device_name(actual)
                if normalized:
                    entries = discovered.setdefault(normalized, [])
                    if not any(address == cursor for address, _name in entries):
                        entries.append((cursor, actual))
                cursor += 4
            scanned += max(0, scan_end - start)

        self._zephyr_device_name_cache = discovered
        return discovered

    def _find_zephyr_device_by_name(self, requested: str) -> Optional[int]:
        requested_norm = self._normalize_device_name(requested)
        if not requested_norm:
            return None

        # First use the real Zephyr device table layout: word[0] points to the
        # device name string. This preserves firmware-defined device ordering.
        for address, _symbol in self._iter_zephyr_device_candidates():
            try:
                name_ptr = int.from_bytes(bytes(self.uc.mem_read(address, 4)), "little") & 0xFFFFFFFF
            except Exception:
                continue
            actual = self._read_c_string(name_ptr)
            if actual and self._normalize_device_name(actual) == requested_norm:
                return address

        scanned = self._scan_zephyr_devices_by_name()
        matches = scanned.get(requested_norm, [])
        if matches:
            return int(matches[0][0]) & 0xFFFFFFFF

        # Fallback for partially initialized tables: use symbol names only.
        for address, symbol in self._iter_zephyr_device_candidates():
            symbol_norm = self._normalize_device_name(symbol)
            if requested_norm in symbol_norm or symbol_norm.endswith(requested_norm):
                return address
        return None

    @staticmethod
    def _load_stream_input_seed() -> bytes:
        configured = os.environ.get("LSGEMU_STREAM_INPUT_BYTES", "").strip()
        if configured:
            try:
                cleaned = re.sub(r"[^0-9a-fA-F]", "", configured)
                if cleaned and len(cleaned) % 2 == 0:
                    data = bytes.fromhex(cleaned)
                    if data:
                        return data
            except Exception:
                pass
        # Command-rich but firmware-independent byte stream: common MCU command
        # bytes first, followed by printable strings and terminators.
        return bytes([
            0xDD, 0xFF, 0xBB, 0x31, 0x32, 0x33, 0x00,
            0xFF, 0x41, 0x42, 0x43, 0x0A,
            0xBB, 0x70, 0x71, 0x73, 0x00,
            ord("G"), ord("1"), ord(" "), ord("X"), ord("1"), ord("\n"),
            ord("<"), ord("a"), ord("/"), ord(">"),
            0x00, 0x01, 0x02, 0x08, 0x10, 0x48, 0x68, 0x7F, 0x80, 0xFF,
        ])

    @staticmethod
    def _normalized_symbol_name(symbol: str) -> str:
        return "".join(ch.lower() for ch in str(symbol or "") if ch.isalnum())

    @classmethod
    def _stream_input_summary_for_symbol(cls, symbol: str) -> Optional[Dict[str, str]]:
        lower = str(symbol or "").strip().lower()
        normalized = cls._normalized_symbol_name(lower)
        if not lower:
            return None

        if lower in {"read", "_read"}:
            return {"kind": "stream_read", "arg_style": "posix"}
        if lower == "_read_r":
            return {"kind": "stream_read", "arg_style": "reent"}
        if lower == "__sread":
            return {"kind": "stream_read", "arg_style": "sread"}
        if (
            "directserial4read" in normalized
            or "mbed6stream4read" in normalized
            or "zsockrecv" in normalized
            or "zsockrecvfrom" in normalized
            or "socketrecv" in normalized
            or "recvfrom" in normalized
            or "mbed3i2c4read" in normalized
            or ("i2c4read" in normalized and "mbed" in normalized)
        ):
            return {"kind": "stream_read", "arg_style": "buffer_r1_r2"}

        if "haluartreceive" in normalized or "halspireceive" in normalized:
            # HAL_UART_Receive[_IT](huart, pData, Size[, Timeout])
            # HAL_SPI_Receive(hspi, pData, Size, Timeout)
            return {"kind": "stream_status_read", "arg_style": "buffer_r1_r2"}
        if "halspimasterreceive" in normalized or "hali2cmasterreceive" in normalized:
            # Master I2C style APIs often include an address before the buffer:
            # HAL_I2C_Master_Receive(hi2c, DevAddress, pData, Size, Timeout)
            return {"kind": "stream_status_read", "arg_style": "buffer_r2_r3"}

        if "netcontextrecv" in normalized:
            return {"kind": "stream_status", "arg_style": "status"}

        if (
            lower in {"serial_readable"}
            or "readable" in normalized
            or "available" in normalized
            or "isready" in normalized
            or "isavailable" in normalized
        ):
            return {"kind": "stream_status", "arg_style": "status"}

        if (
            lower in {"serial_getc", "fgetc", "getchar"}
            or "mbed9mbedgetc" in normalized
            or "stream4getc" in normalized
            or "serial5getc" in normalized
            or "serialbase10basegetc" in normalized
            or "hardwareserial4readev" in normalized
            or "usbserial4readev" in normalized
            or normalized.endswith("4readev")
            or normalized.endswith("4getcev")
            or normalized.endswith("3getev")
        ):
            return {"kind": "stream_byte", "arg_style": "byte"}

        if lower in {"_msg_receive", "msg_receive"}:
            return {"kind": "riot_msg_receive", "arg_style": "msg_ptr_r0"}

        return None

    def _next_stream_input_byte(self, stream_key: str) -> int:
        data = self.stream_input_default_bytes or b"\x00"
        key = str(stream_key or "default")
        cursor = int(self.stream_input_state.get(key, 0) or 0)
        self.stream_input_state[key] = cursor + 1
        return int(data[cursor % len(data)]) & 0xFF

    def _stream_input_payload(self, stream_key: str, size: int) -> bytes:
        size = max(0, min(int(size or 0), 65536))
        return bytes(self._next_stream_input_byte(stream_key) for _ in range(size))

    def _record_stream_input_summary_event(self, event: Dict[str, object]) -> None:
        """Keep bounded provenance for synthetic stream input summaries."""
        try:
            events = self.stream_input_summary_events
        except AttributeError:
            self.stream_input_summary_events = []
            events = self.stream_input_summary_events
        item = dict(event)
        item.setdefault("seed_hex", bytes(self.stream_input_default_bytes or b"").hex())
        kind = str(item.get("kind") or "")
        symbol = str(item.get("symbol") or item.get("style") or "stream")
        pc = self._parse_int(item.get("pc")) or 0
        if kind in {
            "stream_status",
            "stream_byte",
            "stream_read",
            "stream_status_read",
            "riot_msg_receive",
        }:
            stats = getattr(self, "environment_input_delivery_stats", None)
            if not isinstance(stats, dict):
                stats = {"observed": 0, "accepted": 0, "rejected": 0}
                self.environment_input_delivery_stats = stats
            stats["observed"] = int(stats.get("observed", 0) or 0) + 1
            if str(item.get("reason") or "").strip():
                stats["rejected"] = int(stats.get("rejected", 0) or 0) + 1
            else:
                stats["accepted"] = int(stats.get("accepted", 0) or 0) + 1
                item.setdefault("environment_fact", "stream_input_delivery")
        causal_context = getattr(self, "causal_context", None)
        if causal_context is not None and kind not in {"mmio_stream_status", "mmio_stream_byte"}:
            if kind == "stream_status":
                causal_context.record_input_ready(
                    symbol,
                    pc=pc,
                    source="function_summary",
                )
            elif kind in {
                "stream_byte",
                "stream_read",
                "stream_status_read",
                "riot_msg_receive",
            } and not item.get("reason"):
                value = self._parse_int(item.get("return_value"))
                causal_context.record_input_consume(
                    symbol,
                    pc=pc,
                    value=value,
                    source="function_summary",
                )
        if len(events) < 256:
            events.append(item)

    def _record_stream_input_payload_write(self, address: int, payload: bytes, source: str) -> None:
        try:
            writes = self.stream_input_payload_writes
        except AttributeError:
            self.stream_input_payload_writes = []
            writes = self.stream_input_payload_writes
        if len(writes) >= 256:
            return
        writes.append({
            "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
            "size": int(len(payload)),
            "payload_hex_preview": bytes(payload[:64]).hex(),
            "source": str(source or "stream_summary"),
        })

    @staticmethod
    def _load_watch_memory_ranges() -> List[Tuple[int, int, str]]:
        raw = os.environ.get("LSGEMU_WATCH_MEMORY_RANGES", "")
        ranges: List[Tuple[int, int, str]] = []
        for index, item in enumerate(raw.split(";")):
            text = item.strip()
            if not text:
                continue
            label = f"watch_{index}"
            range_text = text
            if "=" in text:
                label, range_text = text.split("=", 1)
                label = label.strip() or label
            try:
                if "+" in range_text:
                    start_text, size_text = range_text.split("+", 1)
                    start = int(start_text, 0) & 0xFFFFFFFF
                    size = int(size_text, 0)
                    end = (start + max(0, size)) & 0xFFFFFFFF
                elif "-" in range_text:
                    start_text, end_text = range_text.split("-", 1)
                    start = int(start_text, 0) & 0xFFFFFFFF
                    end = int(end_text, 0) & 0xFFFFFFFF
                else:
                    start = int(range_text, 0) & 0xFFFFFFFF
                    end = start + 4
            except Exception:
                continue
            if end <= start:
                continue
            ranges.append((start, end, label))
        return ranges

    def _record_watch_memory_event(
        self,
        pc: int,
        address: int,
        size: int,
        *,
        is_write: bool,
        value: Optional[int] = None,
    ) -> None:
        if not self.watch_memory_ranges:
            return
        address = int(address) & 0xFFFFFFFF
        size = max(1, int(size or 1))
        access_end = address + size
        for start, end, label in self.watch_memory_ranges:
            if access_end <= start or address >= end:
                continue
            if len(self.watch_memory_events) >= 1024:
                return
            event = {
                "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
                "address": f"0x{address:08x}",
                "size": size,
                "is_write": bool(is_write),
                "label": label,
            }
            if value is not None:
                event["value"] = f"0x{int(value) & 0xFFFFFFFF:08x}"
            else:
                try:
                    raw = bytes(self.uc.mem_read(address, min(size, 4)))
                    event["value"] = f"0x{int.from_bytes(raw, 'little') & 0xFFFFFFFF:08x}"
                except Exception:
                    pass
            self.watch_memory_events.append(event)

    @staticmethod
    def _load_watch_pcs() -> Set[int]:
        raw = os.environ.get("LSGEMU_WATCH_PCS", "")
        pcs: Set[int] = set()
        for item in raw.replace(",", ";").split(";"):
            text = item.strip()
            if not text:
                continue
            try:
                pcs.add(int(text, 0) & ~1)
            except Exception:
                continue
        return pcs

    def _record_watch_pc_event(self, uc, address: int, size: int) -> None:
        normalized_address = int(address) & ~1
        if normalized_address not in self.watch_pcs:
            return
        if len(self.watch_pc_events) >= 1024:
            return
        regs = self._get_registers()
        event = {
            "pc": f"0x{normalized_address:08x}",
            "size": int(size or 0),
            "registers": {
                name: f"0x{int(value) & 0xFFFFFFFF:08x}"
                for name, value in regs.items()
            },
        }
        # Useful for pointer-load triage. These reads are best-effort and only
        # happen for explicit debug watches.
        for reg_name in ("r0", "r1", "r2", "r3", "r4", "sp", "lr"):
            value = regs.get(reg_name)
            if value is None:
                continue
            value = int(value) & 0xFFFFFFFF
            if not (0x20000000 <= value < 0x40000000 or 0x08000000 <= value < 0x10000000):
                continue
            try:
                raw = bytes(uc.mem_read(value, 4))
                event[f"mem32_at_{reg_name}"] = f"0x{int.from_bytes(raw, 'little') & 0xFFFFFFFF:08x}"
            except Exception:
                pass
        self.watch_pc_events.append(event)

    @staticmethod
    def _is_plausible_stream_buffer_address(address: int) -> bool:
        address &= 0xFFFFFFFF
        if address < 0x1000:
            return False
        if 0x08000000 <= address < 0x10000000:
            return False
        if 0x40000000 <= address < 0x60000000:
            return False
        if 0xE0000000 <= address <= 0xE00FFFFF:
            return False
        return True

    def _apply_stream_input_summary(self, uc, rule: Dict[str, object], address: int) -> int:
        kind = str(rule.get("kind") or "")
        style = str(rule.get("arg_style") or "")
        key = f"{int(address) & ~1:08x}:{style}"
        symbol = str(rule.get("symbol") or f"0x{int(address) & ~1:08x}")
        if kind == "stream_status":
            self._save_summary_entry_snapshot_if_needed(address, kind)
            try:
                value = int(os.environ.get("LSGEMU_STREAM_STATUS_VALUE", "1"), 0) & 0xFFFFFFFF
            except ValueError:
                value = 1
            self._record_stream_input_summary_event({
                "pc": f"0x{int(address) & ~1:08x}",
                "symbol": symbol,
                "kind": kind,
                "style": style,
                "return_value": f"0x{value:08x}",
            })
            return value
        if kind == "stream_byte":
            self._save_summary_entry_snapshot_if_needed(address, kind)
            value = self._next_stream_input_byte(key)
            self._record_stream_input_summary_event({
                "pc": f"0x{int(address) & ~1:08x}",
                "symbol": symbol,
                "kind": kind,
                "style": style,
                "return_value": f"0x{value:02x}",
                "byte": value,
            })
            return value
        if kind == "riot_msg_receive":
            msg_ptr = int(uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
            if msg_ptr == 0:
                self._save_summary_entry_snapshot_if_needed(address, kind)
                self._record_stream_input_summary_event({
                    "pc": f"0x{int(address) & ~1:08x}",
                    "symbol": symbol,
                    "kind": kind,
                    "style": style,
                    "return_value": "0x00000000",
                    "reason": "null_msg_ptr",
                })
                return 0
            try:
                max_payload = max(8, int(os.environ.get("LSGEMU_RIOT_MSG_PAYLOAD_BYTES", "128")))
            except ValueError:
                max_payload = 128
            allocation = self._normalized_model_heap_allocation(
                max_payload,
                with_chunk_header=True,
            )
            preflight_ranges = [(msg_ptr, 8)]
            if allocation is not None:
                chunk, _payload_ptr, end, _aligned = allocation
                preflight_ranges.append((chunk, end - chunk))
            if not self._ensure_memory_ranges_mapped(preflight_ranges):
                return 0
            self._save_summary_entry_snapshot_if_needed(address, kind)
            payload = self._stream_input_payload(key, max_payload)
            try:
                msg_type = (
                    int(os.environ.get("LSGEMU_RIOT_MSG_TYPE", "0"), 0) & 0xFFFF
                    if os.environ.get("LSGEMU_RIOT_MSG_TYPE")
                    else 0
                )
            except ValueError:
                msg_type = 0
            if msg_type == 0 and len(payload) >= 2:
                msg_type = (int(payload[0]) | (int(payload[1]) << 8)) & 0xFFFF
            if msg_type == 0:
                msg_type = 0x100
            try:
                sender = (
                    int(os.environ.get("LSGEMU_RIOT_MSG_SENDER", "1"), 0) & 0xFFFF
                    if os.environ.get("LSGEMU_RIOT_MSG_SENDER")
                    else 1
                )
            except ValueError:
                sender = 1
            payload_ptr = self._model_heap_alloc(len(payload), zero=False)
            try:
                if payload_ptr:
                    uc.mem_write(payload_ptr, payload)
                    self.external_memory_input_addresses.add(payload_ptr & 0xFFFFFFFF)
                    self._record_stream_input_payload_write(payload_ptr, payload, symbol)
                uc.mem_write(
                    msg_ptr,
                    sender.to_bytes(2, "little")
                    + msg_type.to_bytes(2, "little")
                    + int(payload_ptr or 0).to_bytes(4, "little"),
                )
                self.external_memory_input_addresses.add(msg_ptr & 0xFFFFFFFF)
                self._record_stream_input_summary_event({
                    "pc": f"0x{int(address) & ~1:08x}",
                    "symbol": symbol,
                    "kind": kind,
                    "style": style,
                    "return_value": "0x00000001",
                    "msg_ptr": f"0x{msg_ptr:08x}",
                    "payload_ptr": f"0x{int(payload_ptr or 0):08x}",
                    "payload_size": len(payload),
                    "payload_hex_preview": payload[:64].hex(),
                })
                return 1
            except Exception as exc:
                logger.debug(
                    "RIOT msg_receive摘要失败: msg=0x%08x type=0x%04x payload=0x%08x err=%s",
                    msg_ptr,
                    msg_type,
                    int(payload_ptr or 0),
                    exc,
                )
            return 0
        if kind not in {"stream_read", "stream_status_read"}:
            return 0

        r0 = int(uc.reg_read(UC_ARM_REG_R0)) & 0xFFFFFFFF
        r1 = int(uc.reg_read(UC_ARM_REG_R1)) & 0xFFFFFFFF
        r2 = int(uc.reg_read(UC_ARM_REG_R2)) & 0xFFFFFFFF
        r3 = int(uc.reg_read(UC_ARM_REG_R3)) & 0xFFFFFFFF
        if style in {"reent", "sread", "buffer_r2_r3"}:
            buf, size = r2, r3
        elif style == "posix":
            buf, size = r1, r2
        else:
            buf, size = r1, r2
        try:
            max_read = max(1, int(os.environ.get("LSGEMU_STREAM_READ_MAX_BYTES", "64")))
        except ValueError:
            max_read = 64
        size = min(int(size or 0), max_read)
        if buf == 0 or size <= 0 or not self._is_plausible_stream_buffer_address(buf):
            self._save_summary_entry_snapshot_if_needed(address, kind)
            self._record_stream_input_summary_event({
                "pc": f"0x{int(address) & ~1:08x}",
                "symbol": symbol,
                "kind": kind,
                "style": style,
                "return_value": "0x00000000",
                "buf": f"0x{buf:08x}",
                "size": int(size),
                "reason": (
                    "empty_buffer_or_size"
                    if buf == 0 or size <= 0
                    else "implausible_buffer_address"
                ),
                "r0": f"0x{r0:08x}",
                "r1": f"0x{r1:08x}",
                "r2": f"0x{r2:08x}",
                "r3": f"0x{r3:08x}",
            })
            return 0
        # Mapping must succeed before consuming bytes from the stream.  On a
        # safe-point retry this keeps the cursor and the resulting payload
        # identical to a run where the buffer was pre-mapped.
        if not self._ensure_memory_mapped(buf, size):
            return 0
        self._save_summary_entry_snapshot_if_needed(address, kind)
        payload = self._stream_input_payload(key, size)
        try:
            uc.mem_write(buf, payload)
            self.external_memory_input_addresses.add(buf & 0xFFFFFFFF)
            self._record_stream_input_payload_write(buf, payload, symbol)
            return_value = 0 if kind == "stream_status_read" else (len(payload) & 0xFFFFFFFF)
            self._record_stream_input_summary_event({
                "pc": f"0x{int(address) & ~1:08x}",
                "symbol": symbol,
                "kind": kind,
                "style": style,
                "return_value": f"0x{return_value:08x}",
                "buf": f"0x{buf:08x}",
                "size": int(size),
                "payload_size": len(payload),
                "payload_hex_preview": payload[:64].hex(),
                "r0": f"0x{r0:08x}",
                "r1": f"0x{r1:08x}",
                "r2": f"0x{r2:08x}",
                "r3": f"0x{r3:08x}",
            })
            if kind == "stream_status_read":
                return 0
            return len(payload) & 0xFFFFFFFF
        except Exception as exc:
            logger.debug(
                "流输入摘要写缓冲失败: style=%s buf=0x%08x size=%d err=%s",
                style,
                buf,
                size,
                exc,
            )
        return 0

    @staticmethod
    def _is_time_skip_symbol(symbol: str) -> bool:
        normalized = "".join(ch.lower() for ch in str(symbol or "") if ch.isalnum())
        if not normalized:
            return False
        if normalized in {
            "halgettick",
            "millis",
            "micros",
            "getcurrentmilli",
            "getcurrentmicro",
            "rttgetcounter",
            "ztimernow",
            "xtimernow",
        }:
            return True
        return (
            "halgettick" in normalized
            or "getcurrentmilli" in normalized
            or "getcurrentmicro" in normalized
            or "rttgetcounter" in normalized
            or "ztimer" in normalized
            or "xtimer" in normalized
            or normalized.endswith("millis")
            or normalized.endswith("micros")
        )

    @staticmethod
    def _default_skip_return_for_symbol(symbol: str) -> Optional[int]:
        name = str(symbol or "")
        if not name:
            return None
        lower = name.lower()
        normalized = "".join(ch.lower() for ch in name if ch.isalnum())
        if (
            ("i2c" in normalized and "write" in normalized)
            or ("spi" in normalized and ("write" in normalized or "send" in normalized or "transmit" in normalized))
            or ("uart" in normalized and ("write" in normalized or "send" in normalized or "transmit" in normalized))
            or "serialbaud" in normalized
            or "serialformat" in normalized
            or "serialirqhandler" in normalized
            or "pinmappinout" in normalized
            or "gpioinit" in normalized
            or "i2cinit" in normalized
            or "spiinit" in normalized
            or "uartinit" in normalized
            or "nvicsetpriority" in normalized
            or "nvicenableirq" in normalized
            or "nvicdisableirq" in normalized
            or "nvicsetpendingirq" in normalized
            or "nvicclearpendingirq" in normalized
            or "aquire" in normalized
            or "acquire" in normalized
            or "release" in normalized
        ):
            return 0
        if any(
            part in normalized
            for part in (
                "waiton",
                "untiltimeout",
                "waitus",
                "waitms",
                "delayms",
                "delayus",
                "sleepms",
                "writebyte",
                "writebytes",
                "i2cwrite",
                "spisend",
                "spitransmit",
                "uartwrite",
                "uarttransmit",
                "serialwrite",
                "halmemwrite",
                "hali2cmemwrite",
                "hali2cmastertransmit",
                "haluarttransmit",
                "halspitransmit",
            )
        ):
            return 0
        if lower == "wait" or normalized in {"wait", "waitus", "waitms"}:
            return 0
        if lower in {
            "__malloc_lock",
            "__malloc_unlock",
            "__retarget_lock_acquire_recursive",
            "__retarget_lock_release_recursive",
            "__retarget_lock_init_recursive",
            "__sfp_lock_acquire",
            "__sfp_lock_release",
            "_nib_acquire",
            "_nib_release",
            "_gnrc_netreg_acquire_exclusive",
            "_gnrc_netreg_release_exclusive",
            "mutex_lock",
            "mutex_unlock",
            "thread_yield",
            "thread_sleep",
            "sched_arch_idle",
            "thread_create",
            "hwrng_read",
            "__sinit",
        }:
            return 0
        if any(
            part in normalized
            for part in (
                "mutexlock",
                "mutexunlock",
                "lockacquire",
                "lockrelease",
                "schedarchidle",
                "threadyield",
                "threadsleep",
                "hwrngread",
                "randomread",
            )
        ):
            return 0
        if lower in {"puts", "_puts_r", "iprintf", "printf", "putchar", "_vfiprintf_r", "_vfprintf_r"}:
            return 1
        if any(
            part in normalized
            for part in (
                "printf",
                "vfprintf",
                "vfiprintf",
                "fprintf",
                "snprintf",
                "vsnprintf",
                "putchar",
                "puts",
                "putc",
                "flush",
            )
        ):
            return 1
        if lower in {"_write_r", "_read_r", "_close_r", "_fstat_r", "_isatty_r", "_lseek_r"}:
            return 0
        if lower in {"_sbrk_r", "sbrk", "sbrk_aligned"}:
            return 0
        if lower in {"__smakebuf_r"}:
            return 0
        return None

    @staticmethod
    def _zephyr_summary_for_symbol(symbol: str) -> Optional[Dict[str, str]]:
        lower = str(symbol or "").strip().lower()
        if lower == "stm32_clock_control_get_subsys_rate":
            return {"kind": "zephyr_clock_subsys_rate"}
        if lower == "z_impl_device_get_binding":
            return {"kind": "zephyr_device_get_binding"}
        return None

    @staticmethod
    def _allocator_summary_for_symbol(symbol: str) -> Optional[Dict[str, str]]:
        lower = str(symbol or "").strip().lower()
        if lower in {"malloc", "_malloc_r"}:
            return {"kind": "malloc", "arg_style": "reent" if lower.startswith("_") else "plain"}
        if lower in {"calloc", "_calloc_r"}:
            return {"kind": "calloc", "arg_style": "reent" if lower.startswith("_") else "plain"}
        if lower in {"realloc", "_realloc_r"}:
            return {"kind": "realloc", "arg_style": "reent" if lower.startswith("_") else "plain"}
        if lower in {"free", "_free_r"}:
            return {"kind": "free", "arg_style": "reent" if lower.startswith("_") else "plain"}
        if lower in {"sbrk", "_sbrk_r", "sbrk_aligned"}:
            return {"kind": "sbrk", "arg_style": "reent" if lower.startswith("_") else "plain"}
        return None

    @staticmethod
    def _memory_intrinsic_kind_for_symbol(symbol: str) -> Optional[str]:
        lower = str(symbol or "").strip().lower()
        normalized = "".join(ch.lower() for ch in lower if ch.isalnum() or ch == "_")
        aliases = {
            "memcpy": "memcpy",
            "__aeabi_memcpy": "memcpy",
            "__aeabi_memcpy4": "memcpy",
            "__aeabi_memcpy8": "memcpy",
            "memmove": "memmove",
            "__aeabi_memmove": "memmove",
            "__aeabi_memmove4": "memmove",
            "__aeabi_memmove8": "memmove",
            "memset": "memset",
            "__aeabi_memset": "memset",
            "__aeabi_memclr": "memset",
            "__aeabi_memclr4": "memset",
            "__aeabi_memclr8": "memset",
        }
        if lower in aliases:
            return aliases[lower]
        return aliases.get(normalized)

    def configure_skip_function_returns(
        self,
        symbols_by_addr: Dict[int, str],
        *,
        enabled: bool = True,
        include_time: bool = False,
        time_start: int = 0,
        time_increment: int = 1000,
        exclude_addresses: Optional[Set[int]] = None,
    ) -> None:
        self.skip_function_returns.clear()
        self.skip_function_state.clear()
        self.skipped_function_entry_bbs.clear()
        self.skip_function_stats = {
            "installed": 0,
            "applied": 0,
            "stateful_applied": 0,
            "by_symbol": {},
            "applied_entry_bbs": [],
        }
        # r37 P2 → r40 P6：按家族选择性关闭"函数摘要/跳过"的安装。被排除的
        # 家族不安装规则 ⇒ callee 按真实指令执行。家族名：time / allocator /
        # memory_intrinsic / stream / zephyr / default_return。
        # r40 P6 缺省翻转（用户拍板「覆盖的基本块都是真实的」）：默认关闭
        # 「执行替换」三族 memory_intrinsic/allocator/default_return（r37 实测
        # function_summary_or_skip 残差 = memset/calloc/gpio_init/i2cInit/spiInit，
        # 属契约 execution_changes_not_accepted）。time/stream/zephyr 是契约
        # 允许的环境事实族，保持安装。显式设 env（如 =""）可恢复 r37 现状。
        disabled_families = {
            item.strip().lower()
            for item in str(
                os.environ.get(
                    "LSGEMU_REPLAY_SKIP_FUNCTION_FAMILIES",
                    "memory_intrinsic,allocator,default_return",
                )
            ).split(",")
            if item.strip()
        }

        def family_enabled(family: str) -> bool:
            return str(family) not in disabled_families

        if not enabled:
            return
        excluded = {int(address) & ~1 for address in (exclude_addresses or set())}
        self.symbols_by_addr = {
            int(address) & ~1: str(symbol)
            for address, symbol in (symbols_by_addr or {}).items()
        }
        for address, symbol in sorted(self.symbols_by_addr.items()):
            if int(address) not in self.static_bbs:
                continue
            if int(address) in excluded:
                continue
            if (
                include_time
                and family_enabled("time")
                and self._is_time_skip_symbol(symbol)
            ):
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    "kind": "monotonic_time",
                    "start": int(time_start) & 0xFFFFFFFF,
                    "increment": max(1, int(time_increment)) & 0xFFFFFFFF,
                }
                continue
            allocator_summary = (
                self._allocator_summary_for_symbol(symbol)
                if family_enabled("allocator")
                else None
            )
            if allocator_summary is not None:
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    **allocator_summary,
                }
                continue
            intrinsic_kind = (
                self._memory_intrinsic_kind_for_symbol(symbol)
                if family_enabled("memory_intrinsic")
                else None
            )
            if intrinsic_kind is not None:
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    "kind": intrinsic_kind,
                }
                continue
            stream_summary = (
                self._stream_input_summary_for_symbol(symbol)
                if family_enabled("stream")
                else None
            )
            if stream_summary is not None:
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    **stream_summary,
                }
                continue
            zephyr_summary = (
                self._zephyr_summary_for_symbol(symbol)
                if family_enabled("zephyr")
                else None
            )
            if zephyr_summary is not None:
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    **zephyr_summary,
                }
                continue
            return_value = (
                self._default_skip_return_for_symbol(symbol)
                if family_enabled("default_return")
                else None
            )
            if return_value is not None:
                self.skip_function_returns[int(address)] = {
                    "symbol": str(symbol),
                    "return_value": int(return_value) & 0xFFFFFFFF,
                }
        self.skip_function_stats["installed"] = len(self.skip_function_returns)

    def _is_internal_unicorn_bb_fragment(
        self,
        raw_address: int,
        canonical_bb: int,
        previous_bb: Optional[int],
    ) -> bool:
        """Identify a Unicorn translation fragment inside the current static BB."""
        if previous_bb is None:
            return False
        raw_address = int(raw_address) & ~1
        canonical_bb = int(canonical_bb) & ~1
        previous_bb = int(previous_bb) & ~1
        if previous_bb != canonical_bb or raw_address == canonical_bb:
            return False
        if raw_address in self.static_bbs:
            return False
        return int(self.instruction_to_bb.get(raw_address, -1)) == canonical_bb

    def _bb_history_depth(self) -> int:
        """Return the exact path depth, independent of retained-tail eviction."""
        history = getattr(self, "bb_history", None) or []
        retained_length = len(history)
        recorded_total = int(getattr(self, "bb_history_total", 0) or 0)
        # Keep compatibility with callers/tests that populate bb_history directly.
        # A retained tail can never be longer than the exact count, but an
        # externally supplied legacy list may be the only available count.
        if recorded_total < retained_length:
            recorded_total = retained_length
            self.bb_history_total = recorded_total
        return max(0, recorded_total)

    def _append_retained_history(
        self,
        attribute: str,
        record: object,
        *,
        limit_attribute: str,
        total_attribute: str,
        discarded_attribute: str,
    ) -> None:
        """Append an observational record while retaining a bounded tail.

        Coverage, branch occurrence counts, and causal input indices are kept
        in their dedicated ledgers.  These lists are diagnostic/heuristic
        inputs whose consumers already inspect a recent window, so evicting
        only their old payloads does not alter the replay state contract.
        """
        history = getattr(self, attribute, None)
        if history is None:
            history = []
            setattr(self, attribute, history)
        elif not isinstance(history, list):
            history = list(history)
            setattr(self, attribute, history)
        history.append(record)
        setattr(
            self,
            total_attribute,
            int(getattr(self, total_attribute, 0) or 0) + 1,
        )
        limit = max(1024, int(getattr(self, limit_attribute, 16384) or 16384))
        if len(history) > limit * 2:
            discarded = len(history) - limit
            del history[:discarded]
            setattr(
                self,
                discarded_attribute,
                int(getattr(self, discarded_attribute, 0) or 0) + discarded,
            )

    def _record_mmio_access_history(
        self,
        pc: int,
        address: int,
        is_read: bool,
        value: int,
    ) -> None:
        record = (
            int(pc) & 0xFFFFFFFF,
            int(address) & 0xFFFFFFFF,
            bool(is_read),
            int(value or 0) & 0xFFFFFFFF,
        )
        self._append_retained_history(
            "mmio_access_history",
            record,
            limit_attribute="mmio_access_history_limit",
            total_attribute="mmio_access_history_total",
            discarded_attribute="mmio_access_history_entries_discarded",
        )
        if record[2]:
            self.latest_mmio_read_by_address[record[1]] = (record[0], record[3])

    def _record_memory_access_history(
        self,
        pc: int,
        address: int,
        is_read: bool,
        value: int,
    ) -> None:
        self._append_retained_history(
            "memory_access_history",
            (
                int(pc) & 0xFFFFFFFF,
                int(address) & 0xFFFFFFFF,
                bool(is_read),
                int(value or 0) & 0xFFFFFFFF,
            ),
            limit_attribute="memory_access_history_limit",
            total_attribute="memory_access_history_total",
            discarded_attribute="memory_access_history_entries_discarded",
        )

    def _append_bb_history(self, bb_addr: int) -> None:
        """Record a BB, preserving exact successor evidence and a bounded tail."""
        current = int(bb_addr) & ~1
        history = getattr(self, "bb_history", None)
        if history is None:
            history = []
            self.bb_history = history
        previous_depth = self._bb_history_depth()
        previous = history[-1] if history else None
        if previous is not None and int(previous) != current:
            edges = getattr(self, "runtime_successor_edges", None)
            if edges is None:
                edges = set()
                self.runtime_successor_edges = edges
            edges.add((int(previous), current))
        history.append(current)
        self.bb_history_total = previous_depth + 1
        visit_counts = getattr(self, "bb_visit_counts", None)
        if visit_counts is None:
            visit_counts = Counter()
            self.bb_visit_counts = visit_counts
        visit_counts[current] += 1
        limit = max(
            1024,
            int(getattr(self, "bb_history_limit", 8192) or 8192),
        )
        if len(history) > limit * 2:
            discarded = len(history) - limit
            del history[:discarded]
            self.bb_history_entries_discarded = int(
                getattr(self, "bb_history_entries_discarded", 0) or 0
            ) + discarded

    # ------------------------------------------------------------------
    # r31: interrupt-delivery boundaries + stop conditioning
    # ------------------------------------------------------------------
    #
    # Symbol tokens that mark a self-loop as a *wait* sink the RTOS parks on
    # (interruptible, forward progress comes from a hardware event) rather than
    # a fault terminal.  Anything not matching keeps the historical behaviour:
    # stop.  This is deliberately a name test, never an address whitelist.
    IRQ_WAIT_SINK_SYMBOL_TOKENS = (
        "idle",
        "zombie",
        "port_exit_from_isr",
        "port_switch_from_isr",
    )

    def _irq_delivery_defers_hot_loop_halt(self) -> bool:
        """A hot loop is a stable tail only when nothing is outstanding.

        Unlike the sink case this is not gated on the loop's symbol: any hot
        loop with an outstanding modelled event is waiting on that event, not
        proving a quiescent tail.
        """
        if not getattr(self, "irq_delivery_enabled", False):
            return False
        if not self._irq_delivery_has_pending_event():
            self.irq_delivery_deferral_stats["hot_loop_no_event"] += 1
            return False
        self.irq_delivery_deferral_stats["deferred_hot_loop"] += 1
        return True

    def _ensure_irq_symbol_table(self) -> Dict[int, str]:
        if self._irq_symbol_by_addr is None:
            inverse: Dict[int, str] = {}
            try:
                for symbol_name, symbol_address in self._load_elf_all_symbols_by_name().items():
                    inverse.setdefault(int(symbol_address) & ~1, str(symbol_name))
            except Exception:
                inverse = {}
            self._irq_symbol_by_addr = inverse
            self._irq_symbol_starts_cache = sorted(inverse)
        return self._irq_symbol_by_addr

    def _symbol_name_for_bb(self, bb: int) -> str:
        address = int(bb) & ~1
        if address in self._irq_sink_symbol_cache:
            return self._irq_sink_symbol_cache[address]
        name = str((getattr(self, "symbols_by_addr", {}) or {}).get(address, "") or "")
        if not name:
            table = self._ensure_irq_symbol_table()
            name = str(table.get(address, "") or "")
            if not name:
                # Symbolize by the nearest preceding symbol start: a `b .`
                # inside a function belongs to that function (e.g. the zombie
                # branch at __port_exit_from_isr+2).
                starts = self._irq_symbol_starts_cache or []
                if starts:
                    import bisect

                    index = bisect.bisect_right(starts, address) - 1
                    if index >= 0:
                        name = str(table.get(starts[index], "") or "")
        self._irq_sink_symbol_cache[address] = name
        return name

    def _is_interruptible_wait_sink(self, bb: int) -> bool:
        """True for RTOS wait/idle self-loops, False for fault terminals."""
        symbol = self._symbol_name_for_bb(bb).lower()
        if not symbol:
            return False
        return any(token in symbol for token in self.IRQ_WAIT_SINK_SYMBOL_TOKENS)

    def _irq_boundary_crossed(self) -> bool:
        """Consume a pending interrupt-entry/exit boundary marker."""
        if not getattr(self, "irq_delivery_enabled", False):
            return False
        serial = int(getattr(self, "irq_delivery_boundary", 0) or 0)
        if serial == int(self._irq_delivery_boundary_seen or 0):
            return False
        self._irq_delivery_boundary_seen = serial
        return True

    def _irq_delivery_has_pending_event(self) -> bool:
        controller = getattr(self, "irq_delivery_controller", None)
        if controller is None:
            return False
        try:
            return bool(controller.has_pending_event())
        except Exception:
            return False

    def _irq_delivery_defers_terminal_stop(self, bb: int) -> bool:
        """Should a terminal self-loop be treated as a wait instead of a sink?

        Two independent conditions, both required: the sink must be an RTOS
        wait/idle loop (not a fault terminal), and a modelled hardware event
        must actually be outstanding or imminently deliverable.
        """
        if not getattr(self, "irq_delivery_enabled", False):
            return False
        if not self._is_interruptible_wait_sink(bb):
            self.irq_delivery_deferral_stats["fault_terminal_sink"] += 1
            return False
        if not self._irq_delivery_has_pending_event():
            self.irq_delivery_deferral_stats["wait_sink_no_event"] += 1
            return False
        self.irq_delivery_deferral_stats["deferred_wait_sink"] += 1
        return True

    def _irq_delivery_watchdog_tick(self, deferred: bool) -> bool:
        """Independent no-progress fuse.  Returns True when it tripped."""
        if not getattr(self, "irq_delivery_enabled", False):
            return False
        if not deferred:
            self.irq_delivery_watchdog_spins = 0
            return False
        self.irq_delivery_watchdog_spins += 1
        limit = int(getattr(self, "irq_delivery_watchdog_limit", 0) or 0)
        if limit <= 0 or self.irq_delivery_watchdog_spins < limit:
            return False
        logger.info(
            "无进展看门狗熔断: 连续 %d 个BB在等待模型事件但无投递/无新覆盖",
            self.irq_delivery_watchdog_spins,
        )
        self.stop_requested_reason = "no_progress_watchdog"
        self.uc.emu_stop()
        return True

    def bb_hook(self, uc, address, size, user_data):
        """
        基本块Hook - 智能版

        核心逻辑：
        1. 记录执行
        2. 软时钟自增
        3. 处理时间函数返回
        4. 智能循环分类
        5. 根据循环类型决定是否干预
        6. 记录分支点（用于路径探索）
        7. 在分支点保存快照（关键优化）
        """
        if self._consume_deferred_retry_block_skip(address):
            return
        retrying_block_callback = self._consume_deferred_retry_block_reentry(
            address
        )

        # 【修复】确保address是BB起始地址
        # 检查address是否在static_bbs中，如果不在，尝试找到对应的BB起始地址
        bb_start_addr = self._ensure_dynamic_basic_block(address)
        if self._external_rom_summary_target == (int(address) & ~1):
            self.instruction_count += 1
            self.time_handler.tick()
            self.time_handler.handle_function_return(uc, address)
            self._external_rom_summary_target = None
            return
        if address not in self.static_bbs:
            bb_start_addr = self.instruction_to_bb.get(address, bb_start_addr)
        if bb_start_addr == address and address not in self.static_bbs:
            # 尝试找到包含这个地址的BB
            for bb_addr, instructions in self.static_bbs.items():
                if instructions and len(instructions) > 0:
                    first_addr = instructions[0].get('address', bb_addr)
                    last_addr = instructions[-1].get('address', bb_addr)
                    if first_addr <= address <= last_addr:
                        bb_start_addr = bb_addr
                        break

        previous_bb = self.bb_history[-1] if self.bb_history else None
        internal_unicorn_fragment = bool(
            not retrying_block_callback
            and self._is_internal_unicorn_bb_fragment(
                address,
                bb_start_addr,
                previous_bb,
            )
        )
        if internal_unicorn_fragment:
            self.internal_unicorn_bb_fragments += 1
            return

        if not retrying_block_callback:
            self.instruction_count += 1

            # 软时钟自增（每个规范化BB自增）
            self.time_handler.tick()

            # 处理时间函数返回
            self.time_handler.handle_function_return(uc, address)

            # r31：中断在BB中间夺走CPU后，ISR的第一个BB不能被记成「被打断BB的
            # 后继分支」，异常返回后线程恢复的第一个BB同理。边界标记在进/出各
            # bump 一次（见 IrqDeliveryController），这里消费它来切断假边。
            irq_boundary = self._irq_boundary_crossed()
            edge_previous_bb = None if irq_boundary else previous_bb
            if irq_boundary:
                self.irq_delivery_deferral_stats["branch_edges_cut"] += 1

            # 从入口重放路径时，分支方向只在真实到达对应分支后才约束。
            if (
                not internal_unicorn_fragment
                and (self.forced_branch_choices or self.forced_branch_sequence)
                and edge_previous_bb is not None
            ):
                if self._verify_pending_forced_branch(edge_previous_bb, bb_start_addr):
                    return

            if self.enable_branch_snapshot and not internal_unicorn_fragment:
                self._save_branch_entry_snapshot_if_needed(bb_start_addr)

            # 记录BB（使用BB起始地址）
            is_new_bb = bb_start_addr not in self.bb_addr_set
            if is_new_bb:
                self.bb_addr_set.add(bb_start_addr)
                self._no_new_bb_run = 0
                # r35 T1：新增覆盖即视为真实前进，未用完的出口宽限作废。
                self._quiescence_exit_grace_remaining = 0
            else:
                self._no_new_bb_run += 1

            # 记录分支点（在添加到history之前）
            if (
                not internal_unicorn_fragment
                and self.path_explorer is not None
                and edge_previous_bb is not None
            ):
                self._record_branch_if_needed(edge_previous_bb, bb_start_addr, size)

            # 【关键优化】在分支点保存快照
            if (
                not internal_unicorn_fragment
                and self.enable_branch_snapshot
                and edge_previous_bb is not None
            ):
                self._save_branch_snapshot_if_needed(edge_previous_bb, bb_start_addr)

            self._append_bb_history(bb_start_addr)

        if self._try_lzo1x_decompress_summary(bb_start_addr):
            return

        if bb_start_addr in self.terminal_self_loop_bbs:
            # r31 P4：只有「RTOS 等待/空转自环 + 确有未决模型事件」才不判停。
            # 真错误汇点（_unhandled_exception/_exit/NMI_Handler…）的符号不对号，
            # 照旧停；无事件时照旧停。判停/不判停都进无进展看门狗账。
            deferred_sink_stop = self._irq_delivery_defers_terminal_stop(bb_start_addr)
            if self._irq_delivery_watchdog_tick(deferred_sink_stop):
                return
            if not deferred_sink_stop:
                logger.info("✓ 进入terminal self-loop sink @ 0x%08x，结束当前replay", bb_start_addr)
                self.stop_requested_reason = "fatal_sink_terminal"
                self.uc.emu_stop()
                return

        if not retrying_block_callback:
            # 更新MMIO处理器
            self.mmio_handler.update_phase(bb_start_addr)

            # 记录到循环分类器
            self.loop_classifier.record_execution(
                bb_start_addr,
                self._get_registers(),
            )

            # 无论是否是全局新增覆盖，都让快照管理器自行决定是否需要
            # 为当前重放路径重新创建上下文快照。
            self._create_snapshot(bb_start_addr)

        # 智能循环检测。多BB/间接循环由 loop_classifier 归一到
        # active_loop_head；当前 BB 可能只是循环体中的一个 phase。
        loop_check_head = bb_start_addr
        active_loop_head = getattr(self.loop_classifier, "active_loop_head", None)
        active_loop_body = set(getattr(self.loop_classifier, "active_loop_body", []) or [])
        if (
            active_loop_head is not None
            and bb_start_addr in active_loop_body
            and active_loop_head in self.loop_classifier.loop_heads
        ):
            loop_check_head = int(active_loop_head)

        # r15 D1：force-free 翻转重放（dfs_flip_deterministic_fast_forward）在
        # enable_loop_intervention=False 下仍允许「确定性快转族」（纯内存写 +
        # 确定退出 + 不改控制流方向，见 _record_loop_fast_forward 家族）。边界：
        # 环被分类器标记 has_mmio_access 时一律不快转——外设读写副作用不可
        # O(1) 物化。主跑路径（enable_loop_intervention=True）行为不变。
        flip_fast_forward_active = self._flip_fast_forward_eligible(loop_check_head)
        if (
            (self.enable_loop_intervention or flip_fast_forward_active)
            and self.early_byte_copy_fast_forward
            and loop_check_head in self.loop_classifier.loop_heads
        ):
            loop_info = self.loop_classifier.loop_heads[loop_check_head]
            if int(getattr(loop_info, "iteration_count", 0) or 0) >= 1:
                if flip_fast_forward_active:
                    # D1 豁免集＝确定性快转族。init/copy 只在翻转路径早开
                    # （主跑仍走干预路径阈值后触发，顺序语义不变）。
                    if self._handle_finite_memory_initialization_loop(loop_check_head):
                        self._record_intervention_stats(
                            loop_check_head, family="loop_fast_forward_emulation"
                        )
                        return
                    if self._handle_finite_memory_copy_loop(loop_check_head):
                        self._record_intervention_stats(
                            loop_check_head, family="loop_fast_forward_emulation"
                        )
                        return
                    # r35 T2：memset 族（后索引存储 + 指针即计数器 + 无条件
                    # 回跳）。只在翻转 force-free 路径早开，主跑顺序语义不变。
                    if self._handle_pointer_limit_store_loop(loop_check_head):
                        self._record_intervention_stats(
                            loop_check_head, family="loop_fast_forward_emulation"
                        )
                        return
                if self._handle_finite_byte_copy_loop(loop_check_head):
                    self._record_intervention_stats(
                        loop_check_head, family="loop_fast_forward_emulation"
                    )
                    return
                if self._handle_postindexed_store_count_loop(loop_check_head):
                    self._record_intervention_stats(
                        loop_check_head, family="loop_fast_forward_emulation"
                    )
                    return
                if self._handle_postindexed_store_inc_cmp_loop(loop_check_head):
                    self._record_intervention_stats(
                        loop_check_head, family="loop_fast_forward_emulation"
                    )
                    return

        should_intervene, reason = (
            self.loop_classifier.should_intervene(loop_check_head)
            if self.enable_loop_intervention
            else (False, "")
        )

        handled_simple_wait = False
        if should_intervene:
            previous_intervention_iteration = self.last_intervention_iterations.get(
                loop_check_head
            )
            previous_intervention_count = int(self.intervention_count)
            previous_intervention_by_type = Counter(self.intervention_by_type)
            previous_intervention_labels = Counter(self.intervention_event_labels)
            if self._should_run_intervention(loop_check_head):
                logger.warning(f"\n⚠️ 需要干预: {reason}")
                self._log_loop_intervention_diagnostics(loop_check_head)
                try:
                    self._handle_intervention(loop_check_head)
                except _DeferredMemoryMapping:
                    # The retry must see the same scheduler state. Otherwise
                    # _should_run_intervention() suppresses the very repair
                    # whose mapping precondition caused this safe-point stop.
                    if previous_intervention_iteration is None:
                        self.last_intervention_iterations.pop(
                            loop_check_head,
                            None,
                        )
                    else:
                        self.last_intervention_iterations[loop_check_head] = (
                            previous_intervention_iteration
                        )
                    self.intervention_count = previous_intervention_count
                    self.intervention_by_type.clear()
                    self.intervention_by_type.update(
                        previous_intervention_by_type
                    )
                    # r9：家族标签账本随干预计数一并回滚，保持一一对应。
                    self.intervention_event_labels.clear()
                    self.intervention_event_labels.update(
                        previous_intervention_labels
                    )
                    self.memory_mapping_stats[
                        "block_intervention_state_rollbacks"
                    ] += 1
                    raise

        # 额外检查：即使没有达到干预阈值，也检查简单等待循环
        # 但已知会自然退出的 RAM self-loop 不要过早改写状态。
        if self.enable_loop_intervention and loop_check_head in self.loop_classifier.loop_heads:
            loop_info = self.loop_classifier.loop_heads[loop_check_head]
            if loop_info.iteration_count >= 100:
                simple_wait_threshold = max(100, self._get_loop_intervention_soft_floor(loop_check_head))
                if loop_info.iteration_count >= simple_wait_threshold:
                    if self.wait_loop_handler.is_wait_loop(loop_check_head):
                        last_iteration = self.last_intervention_iterations.get(loop_check_head)
                        allow_retry = (
                            last_iteration is None
                            or loop_info.iteration_count - last_iteration >= self.simple_wait_retry_gap
                        )
                        if not handled_simple_wait and allow_retry:
                            if self.wait_loop_handler.handle_wait_loop(self.uc, loop_check_head):
                                self.last_intervention_iterations[loop_check_head] = loop_info.iteration_count
        elif loop_check_head in self.loop_classifier.loop_heads:
            loop_info = self.loop_classifier.loop_heads[loop_check_head]
            if self._should_halt_hot_loop_without_intervention(loop_check_head, loop_info.iteration_count):
                # r31 P4：有未决模型事件时「稳定尾部」不成立，但由无进展看门狗兜住。
                deferred_hot_loop = self._irq_delivery_defers_hot_loop_halt()
                if self._irq_delivery_watchdog_tick(deferred_hot_loop):
                    return
                if not deferred_hot_loop:
                    logger.info(
                        "热循环稳定尾部熔断: loop @ 0x%08x iter=%d no_new=%d",
                        loop_check_head,
                        loop_info.iteration_count,
                        self._no_new_bb_run,
                    )
                    self.stop_requested_reason = "hot_loop_stable_tail"
                    self.uc.emu_stop()
                    return

        defer_quiescence_for_progress_loop = False
        if self.stop_after_no_new_bbs and self._no_new_bb_run >= self.stop_after_no_new_bbs:
            loop_head_for_quiescence = bb_start_addr
            loop_info = self.loop_classifier.loop_heads.get(loop_head_for_quiescence)
            if loop_info is None:
                for candidate_head, candidate_info in self.loop_classifier.loop_heads.items():
                    try:
                        body = self.loop_classifier._get_loop_body_bbs(int(candidate_head))
                    except Exception:
                        body = list(getattr(candidate_info, "loop_body", []) or [])
                    if bb_start_addr in body:
                        loop_head_for_quiescence = int(candidate_head)
                        loop_info = candidate_info
                        break
            if loop_info is not None:
                try:
                    loop_type = self.loop_classifier.classify_loop(loop_head_for_quiescence)
                except Exception:
                    loop_type = None
                try:
                    initialization_threshold = max(1, int(os.environ.get("LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "10000000")))
                except ValueError:
                    initialization_threshold = 10000000
                try:
                    delay_threshold = max(1, int(os.environ.get("LSGEMU_DELAY_LOOP_THRESHOLD", "100000")))
                except ValueError:
                    delay_threshold = 100000
                if (
                    loop_type in {LoopType.INITIALIZATION, LoopType.DELAY}
                    and bool(getattr(loop_info, "has_exit_condition", False))
                    and (
                        bool(getattr(loop_info, "has_counter_increment", False))
                        or bool(getattr(loop_info, "register_changes", {}))
                    )
                    and int(getattr(loop_info, "iteration_count", 0) or 0) < (
                        delay_threshold if loop_type == LoopType.DELAY else initialization_threshold
                    )
                ):
                    defer_quiescence_for_progress_loop = True

        if (
            self.stop_after_no_new_bbs
            and self._no_new_bb_run >= self.stop_after_no_new_bbs
            and not defer_quiescence_for_progress_loop
        ):
            # r35 T1（默认关）：环出口 fallthrough 的有界宽限。命中时不立即
            # 收口，而是继续执行至多 N 个 BB 后复核；宽限内若出现新增覆盖，
            # _no_new_bb_run 归零并作废剩余宽限（见 is_new_bb 分支）。
            grace_budget = int(getattr(self, "quiescence_loop_exit_grace_budget", 0) or 0)
            if grace_budget > 0:
                if self._quiescence_exit_grace_remaining > 0:
                    self._quiescence_exit_grace_remaining -= 1
                    self.quiescence_loop_exit_grace_stats["consumed_bbs"] += 1
                    return
                grace_head = self._quiescence_loop_exit_fallthrough_head(bb_start_addr)
                if grace_head is not None:
                    self._quiescence_exit_grace_remaining = grace_budget
                    self.quiescence_loop_exit_grace_stats["granted"] += 1
                    logger.info(
                        "r35 T1：静止尾部落在循环 0x%08x 的出口 fallthrough 0x%08x，"
                        "给予 %d 个 BB 的复核宽限（no_new=%d）",
                        int(grace_head),
                        int(bb_start_addr),
                        int(grace_budget),
                        self._no_new_bb_run,
                    )
                    return
            logger.info(
                "连续 %d 个BB没有新增覆盖，认为当前路径已进入稳定尾部，停止仿真",
                self._no_new_bb_run,
            )
            self.stop_requested_reason = "quiescence_no_new_bbs"
            self.uc.emu_stop()
            return

    def _quiescence_loop_exit_fallthrough_head(self, bb_start_addr: int) -> Optional[int]:
        """r35 T1：判断当前 BB 是否为某个已分类有限循环的出口 fallthrough。

        命中的充要条件（缺一不可）：
          1. 存在环头 H（``loop_classifier.loop_heads`` 中已成形）使得当前 BB
             正好是 H 的**条件分支落空后继**（H 静态末条指令有谓词条件，
             落空地址 = 末条地址 + 长度），即环的正常退出边；
          2. 当前 BB 不在 H 的环体内（否则是环中相位，不是出口）；
          3. H 被分类为 INITIALIZATION/DELAY，且有退出条件、有计数/寄存器
             变化，迭代数仍低于对应阈值——与既有 quiescence 豁免同口径，
             只认「会自己退出的有限环」，无限自环（如 ``b .``）天然无落空
             后继，不会被豁免。

        开关关闭时本函数不会被调用；命中只影响「是否立即以静默收口」，不
        改写 PC/寄存器，也不改变终止原因集合。
        """
        try:
            loop_heads = getattr(self.loop_classifier, "loop_heads", None) or {}
        except Exception:
            return None
        target = int(bb_start_addr) & ~1
        for candidate_head, candidate_info in list(loop_heads.items()):
            head = int(candidate_head) & ~1
            if head == target:
                continue
            instructions = self.static_bbs.get(head) or self.static_bbs.get(int(candidate_head)) or []
            if not instructions:
                continue
            last_insn = instructions[-1]
            if self._dispatch_condition_for_instruction(last_insn) is None:
                # 无条件分支（含 `b .` 自环）没有落空后继，永不豁免。
                continue
            try:
                fallthrough = int(last_insn.get("address", head) or head) + int(
                    last_insn.get("size", 2) or 2
                )
            except Exception:
                continue
            if (fallthrough & ~1) != target:
                continue
            try:
                body = {
                    int(b) & ~1
                    for b in (
                        self.loop_classifier._get_loop_body_bbs(head)
                        or getattr(candidate_info, "loop_body", [])
                        or []
                    )
                }
            except Exception:
                body = {int(b) & ~1 for b in (getattr(candidate_info, "loop_body", []) or [])}
            if target in body:
                continue
            try:
                loop_type = self.loop_classifier.classify_loop(head)
            except Exception:
                loop_type = None
            if loop_type not in {LoopType.INITIALIZATION, LoopType.DELAY}:
                continue
            if not bool(getattr(candidate_info, "has_exit_condition", False)):
                continue
            if not (
                bool(getattr(candidate_info, "has_counter_increment", False))
                or bool(getattr(candidate_info, "register_changes", {}))
            ):
                continue
            try:
                initialization_threshold = max(
                    1, int(os.environ.get("LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "10000000"))
                )
            except ValueError:
                initialization_threshold = 10000000
            try:
                delay_threshold = max(1, int(os.environ.get("LSGEMU_DELAY_LOOP_THRESHOLD", "100000")))
            except ValueError:
                delay_threshold = 100000
            limit = delay_threshold if loop_type == LoopType.DELAY else initialization_threshold
            if int(getattr(candidate_info, "iteration_count", 0) or 0) >= limit:
                continue
            return head
        return None

    def _should_run_intervention(self, loop_head: int) -> bool:
        """限制同一个循环的干预频率，避免每次迭代都调用LLM/回退。"""
        loop_info = self.loop_classifier.loop_heads.get(loop_head)
        if not loop_info:
            return True

        current_iteration = loop_info.iteration_count
        soft_floor = self._get_loop_intervention_soft_floor(loop_head)
        if current_iteration < soft_floor:
            return False

        last_iteration = self.last_intervention_iterations.get(loop_head)
        if last_iteration is not None and current_iteration - last_iteration < 100:
            return False

        self.last_intervention_iterations[loop_head] = current_iteration
        return True

    def _get_loop_intervention_soft_floor(self, loop_head: int) -> int:
        cached = self.loop_intervention_threshold_cache.get(loop_head)
        if cached is not None:
            return cached

        soft_floor = 0
        observed_hint = self.loop_exit_iteration_hints.get(loop_head)
        if observed_hint is not None:
            soft_floor = max(soft_floor, int(observed_hint) + self.observed_loop_exit_margin)

        if soft_floor == 0:
            snapshot = self._build_current_loop_snapshot(loop_head)
            if snapshot is not None:
                try:
                    analysis = self.code_analyzer._analyze_self_loop_constraint(snapshot)
                except Exception as e:
                    logger.debug("self-loop soft-floor分析失败 @ 0x%08x: %s", loop_head, e)
                    analysis = None
                if analysis is not None and float(getattr(analysis, "confidence", 0.0) or 0.0) >= 0.8:
                    for constraint in getattr(analysis, "suggested_constraints", []) or []:
                        if str(constraint.get("type", "")).lower() != "memory":
                            continue
                        try:
                            addr = int(constraint.get("address", 0) or 0)
                        except Exception:
                            continue
                        if 0x20000000 <= addr < 0x40000000:
                            soft_floor = max(soft_floor, self.ram_self_loop_soft_threshold)
                            break

        self.loop_intervention_threshold_cache[loop_head] = soft_floor
        return soft_floor

    def _should_halt_hot_loop_without_intervention(self, loop_head: int, iteration_count: int) -> bool:
        """
        Replay/ISR paths often disable loop intervention to avoid LLM-side effects,
        but they still need a deterministic way to stop spinning in a stable tail.
        Only halt when the loop is clearly hot *and* coverage has been flat for a while.
        """
        if self.enable_loop_intervention:
            return False
        if self.hot_loop_halt_threshold <= 0 or self.hot_loop_min_no_new_bbs <= 0:
            return False
        if iteration_count < self.hot_loop_halt_threshold:
            return False
        if self._no_new_bb_run < self.hot_loop_min_no_new_bbs:
            return False
        if self._initialization_loop_within_budget(loop_head, iteration_count):
            # r14：带退出条件的 INITIALIZATION/DELAY 环是有界工作，不是稳定
            # 尾部——现场：_crt0_entry 的 .bss 清零环（.bss=93092B ⇒ 23273 次
            # 迭代，0x08005062 实测 classification=initialization/
            # has_exit_condition=True）在 force-free 翻转重放里于第 20000 次
            # 被熔断，重放永远进不了 main ⇒ 16/16 prefix_divergence。与
            # stop_after_no_new_bbs 判停路径同款豁免；指令数/超时预算仍在外
            # 层封顶，WAIT/POLLING/UNKNOWN 环照旧熔断。
            return False
        if not self.semantic_obligation_enabled:
            # The semantic-obligation ablation uses only the generic iteration
            # and quiescence bounds; do not consult loop-analysis soft floors.
            return True
        soft_floor = self._get_loop_intervention_soft_floor(loop_head)
        if soft_floor > 0 and iteration_count < soft_floor:
            return False
        return True

    def _initialization_loop_within_budget(
        self, loop_head: int, iteration_count: int
    ) -> bool:
        """带退出条件的 INITIALIZATION/DELAY 环豁免（r14）。

        与 stop_after_no_new_bbs 判停路径的豁免同口径（同用
        LSGEMU_INITIALIZATION_LOOP_THRESHOLD、同「先 classify 后读
        has_exit_condition」次序——该字段由指令分析推导，分析前是陈旧
        False）；裸壳/分类器缺失时不豁免（getattr 兜底，消融单测的
        object.__new__ 语义不受影响）。
        """
        try:
            threshold = max(
                1,
                int(
                    os.environ.get(
                        "LSGEMU_INITIALIZATION_LOOP_THRESHOLD", "10000000"
                    )
                ),
            )
        except ValueError:
            threshold = 10000000
        if iteration_count >= threshold:
            return False
        classifier = getattr(self, "loop_classifier", None)
        loop_heads = getattr(classifier, "loop_heads", None) or {}
        loop_info = loop_heads.get(loop_head) if classifier is not None else None
        if loop_info is None:
            return False
        try:
            loop_type = classifier.classify_loop(loop_head)
        except Exception:
            return False
        if loop_type not in {LoopType.INITIALIZATION, LoopType.DELAY}:
            return False
        return bool(getattr(loop_info, "has_exit_condition", False))

    def _count_intervention_event(self, family: str) -> None:
        """r9 裁定：一次干预事件记一个家族标签（与 intervention_count 同步增减）。"""
        key = str(family or "loop_intervention")
        self.intervention_event_labels[key] += 1

    def _reclassify_last_intervention_event(self, family: str) -> None:
        """把入口处记的 pessimistic loop_intervention 标签改记为具体家族。

        仅在家族 handler 成功路径调用；对干预未产生效果（返回 False /
        抛 _DeferredMemoryMapping 回滚）的事件保持 loop_intervention 诊断口径。
        """
        key = str(family or "")
        if not key or key == "loop_intervention":
            return
        if int(self.intervention_event_labels.get("loop_intervention", 0) or 0) > 0:
            self.intervention_event_labels["loop_intervention"] -= 1
        self.intervention_event_labels[key] += 1

    def _record_intervention_stats(
        self, loop_head: int, family: str = "loop_intervention"
    ):
        """为不经过 _handle_intervention 的快捷修复路径补齐统计。"""
        self.intervention_count += 1
        self._count_intervention_event(family)
        loop_type = self.loop_classifier.classify_loop(loop_head)
        self.intervention_by_type[loop_type] += 1

    def _log_loop_intervention_diagnostics(self, loop_head: int) -> None:
        if not (
            os.environ.get("LSGEMU_LOOP_DIAGNOSTICS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            return
        loop_info = self.loop_classifier.loop_heads.get(loop_head)
        if loop_info is None:
            logger.warning("loop诊断: loop=0x%08x no_loop_info", loop_head)
            return
        try:
            loop_type = self.loop_classifier.classify_loop(loop_head)
            body = self.loop_classifier._get_loop_body_bbs(loop_head)
        except Exception as e:
            logger.warning("loop诊断失败: loop=0x%08x error=%s", loop_head, e)
            return
        logger.warning(
            "loop诊断: loop=0x%08x type=%s iter=%d body=%s features=read:%s write:%s mmio:%s counter:%s cond:%s exit:%s",
            loop_head,
            loop_type.value,
            int(loop_info.iteration_count),
            ",".join(f"0x{int(bb):08x}" for bb in body[:16]),
            bool(loop_info.has_memory_read),
            bool(loop_info.has_memory_write),
            bool(loop_info.has_mmio_access),
            bool(loop_info.has_counter_increment),
            bool(loop_info.has_conditional_branch),
            bool(loop_info.has_exit_condition),
        )
        for bb in body[:8]:
            insn_text = []
            for insn in self.static_bbs.get(int(bb), [])[:8]:
                insn_text.append(
                    f"0x{int(insn.get('address') or 0):08x}:{insn.get('mnemonic', '')} {insn.get('operands', '')}".strip()
                )
            logger.warning("loop诊断BB: 0x%08x %s", int(bb), " | ".join(insn_text))

    def _record_branch_if_needed(self, prev_address: int, curr_address: int, size: int):
        """
        记录分支点（如果前一个BB包含分支指令）

        Args:
            prev_address: 前一个BB地址
            curr_address: 当前BB地址
            size: BB大小
        """
        # 获取前一个BB的指令
        bb_instructions = self.static_bbs.get(prev_address, [])

        if not bb_instructions:
            return

        # 检查最后一条指令是否是分支
        last_insn = bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
        condition = self._branch_condition_for_mnemonic(mnemonic)
        if condition:
            # 这是一个条件分支
            # 获取目标地址
            operands = last_insn.get('operands', '')

            try:
                # 解析目标地址
                target = self._parse_branch_target(operands)
                if target is None:
                    # 可能是相对地址或标签，跳过
                    return

                # 计算fallthrough地址（不跳转时的地址）
                fallthrough = last_insn.get('address', prev_address) + last_insn.get('size', 2)

                # 判断是否跳转
                target_bb = self._resolve_bb_start(target)
                taken = (curr_address == target_bb)

                # 记录分支
                self.path_explorer.record_branch(
                    address=prev_address,
                    insn_mnemonic=mnemonic,
                    taken=taken,
                    target=target,
                    fallthrough=fallthrough
                )

            except Exception as e:
                logger.debug(f"解析分支失败 @ 0x{prev_address:08x}: {e}")

    def _get_conditional_branch_info(self, bb_address: int) -> Optional[Dict[str, int]]:
        """返回BB末尾条件分支信息。"""
        bb_instructions = self.static_bbs.get(bb_address, [])
        if not bb_instructions:
            return None

        last_insn = bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
        condition = self._dispatch_condition_for_instruction(last_insn)
        if condition is None:
            return None

        operands = last_insn.get('operands', '')

        branch_pc = last_insn.get('address', bb_address)
        fallthrough = branch_pc + last_insn.get('size', 2)
        if str(condition).startswith("IT"):
            it_info = self._get_it_predicate_info(bb_address)
            if it_info is None:
                return None
            return it_info
        if self._is_switch_dispatch_condition(condition):
            targets = (
                self._read_ldr_pc_switch_targets(self.uc, bb_address, last_insn)
                if condition == 'LDRPC'
                else self._read_switch_targets(self.uc, bb_address, last_insn)
            )
            return {
                "branch_pc": branch_pc,
                "target": 0,
                "fallthrough": 0,
                "targets": targets,
            }

        if mnemonic in {'BX', 'BXJ'} or (mnemonic == 'BLX' and self._parse_dispatch_target_register(operands)):
            reg_name = self._parse_dispatch_target_register(operands)
            target = 0
            if reg_name is not None:
                reg_map = {
                    'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1,
                    'r2': UC_ARM_REG_R2, 'r3': UC_ARM_REG_R3,
                    'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
                    'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7,
                    'r8': UC_ARM_REG_R8, 'r9': UC_ARM_REG_R9,
                    'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
                    'r12': UC_ARM_REG_R12, 'lr': UC_ARM_REG_LR,
                    'pc': UC_ARM_REG_PC,
                }
                reg_id = reg_map.get(reg_name.lower())
                if reg_id is not None:
                    try:
                        target = self.uc.reg_read(reg_id) & ~1
                    except Exception:
                        target = 0
            return {
                "branch_pc": branch_pc,
                "target": target,
                "fallthrough": fallthrough if mnemonic == 'BLX' else 0,
            }

        target = self._parse_branch_target(operands)
        if target is None:
            return None
        return {
            "branch_pc": branch_pc,
            "target": target,
            "fallthrough": fallthrough,
        }

    def _build_branch_pc_lookup(self) -> Dict[int, int]:
        """构建真实分支/IT指令PC到BB起点的索引，用于 forced choice。"""
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            last_insn = instructions[-1]
            condition = self._dispatch_condition_for_instruction(last_insn)
            if condition is None:
                continue
            if str(condition).startswith("IT"):
                it_info = self._get_it_predicate_info(bb_addr)
                if it_info is not None:
                    lookup[int(it_info["branch_pc"])] = bb_addr
                continue
            lookup[last_insn.get('address', bb_addr)] = bb_addr
        return lookup

    def _build_branch_entry_lookup(self) -> Dict[int, int]:
        """Map static frontier BB entries that may sit inside a Unicorn block."""
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            last_insn = instructions[-1]
            is_frontier = self._dispatch_condition_for_instruction(last_insn) is not None
            is_frontier = is_frontier or self._is_direct_call_mnemonic(last_insn.get("mnemonic", ""))
            if not is_frontier:
                continue
            first_pc = int(instructions[0].get("address", bb_addr) or bb_addr)
            lookup[first_pc] = int(bb_addr)
        return lookup

    def _build_instruction_to_bb_lookup(self) -> Dict[int, int]:
        """构建指令地址到BB起点的索引，避免运行时反复扫描所有BB。"""
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                address = insn.get('address')
                if address is not None:
                    lookup[address] = bb_addr
        return lookup

    def _dynamic_block_terminates(self, insn: Dict[str, object]) -> bool:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        if self._dispatch_condition_for_instruction(insn) is not None:
            return True
        if mnemonic in {"B", "BAL", "BL", "BLX", "BX", "BXJ"}:
            return True
        operands = str(insn.get("operands", "") or "").lower()
        if mnemonic in {"POP", "LDM", "LDMIA"} and "pc" in operands:
            return True
        return False

    def _trim_dynamic_block_for_internal_target(
        self,
        start_address: int,
        instructions: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """
        Keep runtime-discovered BBs aligned with real control-flow targets.

        Capstone decodes linearly from a dynamic-code entry until the first
        branch.  A common firmware pattern is a prologue followed by a local
        polling loop whose back-edge targets an address inside that decoded
        range:

            prologue...
        loop:
            ldr ...
            cmp ...
            beq loop

        If the whole range is registered under the prologue address, later
        Unicorn callbacks at `loop` are normalized back to the prologue and the
        loop solver sees a polluted window.  Split the prefix here; when
        execution really reaches the internal target, it will be decoded and
        registered as its own BB.
        """
        if len(instructions) < 2:
            return instructions

        first = int(start_address) & ~1
        last_insn = instructions[-1]
        target = self._parse_branch_target(last_insn.get("operands", ""))
        if target is None:
            return instructions
        target = int(target) & ~1
        if target == first:
            return instructions

        target_index = None
        for index, insn in enumerate(instructions):
            if int(insn.get("address", 0) or 0) == target:
                target_index = index
                break
        if target_index is None or target_index <= 0:
            return instructions

        trimmed = instructions[:target_index]
        if not trimmed:
            return instructions

        self.dynamic_static_bb_split_count += 1
        logger.debug(
            "动态BB内部目标拆分: entry=0x%08x target=0x%08x prefix_insns=%d suffix_pending=%d",
            first,
            target,
            len(trimmed),
            len(instructions) - target_index,
        )
        return trimmed

    def _ensure_dynamic_basic_block(self, address: int) -> int:
        """Decode runtime-reached code islands absent from a raw BIN static view."""
        address = int(address) & ~1
        owner = self.instruction_to_bb.get(address)
        if owner is not None:
            return int(owner)
        if address in self.static_bbs:
            return address
        if (
            not self.dynamic_static_bb_enabled
            or self.dynamic_static_bbs_added >= self.dynamic_static_bb_max
            or not self._contains_mapped_range(address, 1)
        ):
            return address
        provenance = self._dynamic_code_provenance(address)
        if provenance is None:
            if self._summarize_external_rom_call(address):
                return address
            self.dynamic_static_bb_rejections += 1
            normalized_address = int(address) & 0xFFFFFFFF
            self.unproven_dynamic_code_targets[normalized_address] += 1
            _GLOBAL_UNPROVEN_DYNAMIC_CODE_TARGETS[normalized_address] += 1
            if self.stop_on_unproven_dynamic_code:
                self.stop_requested_reason = "unproven_dynamic_code_target"
                target_count = _GLOBAL_UNPROVEN_DYNAMIC_CODE_TARGETS[normalized_address]
                if target_count <= 3 or target_count in {5, 10, 25, 50, 100, 250, 500}:
                    logger.warning(
                        "停止未证明动态代码执行 @ 0x%08x (第%d次)",
                        address,
                        target_count,
                    )
                self.uc.emu_stop()
            return address

        try:
            from capstone import CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_BIG_ENDIAN, CS_MODE_LITTLE_ENDIAN, CS_MODE_THUMB, Cs

            mode = CS_MODE_THUMB if self.execution_thumb else CS_MODE_ARM
            if str(getattr(getattr(self, "arch_info", None), "endianness", "")).lower().endswith("big"):
                mode |= CS_MODE_BIG_ENDIAN
            else:
                mode |= CS_MODE_LITTLE_ENDIAN
            max_bytes = 0x100 if self.execution_thumb else 0x100
            code = bytes(self.uc.mem_read(address, max_bytes))
            md = Cs(CS_ARCH_ARM, mode)
            instructions = []
            for index, insn in enumerate(md.disasm(code, address)):
                normalized = {
                    "address": int(insn.address),
                    "mnemonic": str(insn.mnemonic),
                    "operands": str(insn.op_str),
                    "size": int(insn.size),
                    "bytes": bytes(insn.bytes),
                }
                instructions.append(normalized)
                if self._dynamic_block_terminates(normalized) or index >= 63:
                    break
        except Exception as exc:
            logger.debug("动态BB反汇编失败 @ 0x%08x: %s", address, exc)
            return address
        instructions = self._trim_dynamic_block_for_internal_target(address, instructions)
        if not instructions:
            return address

        self.loop_classifier.add_dynamic_basic_block(address, instructions)
        self.static_bbs.setdefault(address, instructions)
        self.dynamic_static_bb_starts.add(address)
        for insn in instructions:
            self.instruction_to_bb[int(insn["address"])] = address
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic in {"MRS", "MSR", "MRC", "MCR", "ISB", "DSB", "DMB", "CPSID", "CPSIE"}:
                self.cortex_m_system_instruction_map[int(insn["address"])] = insn
            if mnemonic in {"SVC", "SWI"}:
                self.svc_instruction_pcs.add(int(insn["address"]))
            if self._is_mapped_mmio_load_candidate(insn):
                self.mapped_mmio_load_instruction_map[int(insn["address"])] = insn
        if hasattr(self.code_analyzer, "add_dynamic_basic_block"):
            self.code_analyzer.add_dynamic_basic_block(address, instructions)
        self._activate_pending_persisted_memory_constraints()

        last_insn = instructions[-1]
        condition = self._dispatch_condition_for_instruction(last_insn)
        if condition is not None:
            self.branch_pc_to_bb[int(last_insn["address"])] = address
            self.branch_entry_pc_to_bb[address] = address
        mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic", ""))
        if mnemonic in {"B", "BAL"}:
            target = self._parse_branch_target(last_insn.get("operands", ""))
            if target is not None and (int(target) & ~1) == address:
                self.terminal_self_loop_bbs.add(address)

        self.dynamic_static_bbs_added += 1
        logger.debug(
            "动态补齐BB @ 0x%08x (%d instructions, provenance=%s)",
            address,
            len(instructions),
            provenance,
        )
        return address

    def _get_it_predicate_info(self, bb_address: int) -> Optional[Dict[str, int]]:
        """Model a Ghidra-split Thumb IT block as a strict, flag-controlled frontier."""
        instructions = self.static_bbs.get(int(bb_address), [])
        if not instructions:
            return None
        last_insn = instructions[-1]
        predicate = self._predicated_condition_for_mnemonic(last_insn.get("mnemonic", ""))
        if predicate is None:
            return None

        it_insn = None
        for insn in reversed(instructions[:-1]):
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic.startswith("IT"):
                it_insn = insn
                break
        if it_insn is None:
            return None

        it_mnemonic = self._normalize_mnemonic(it_insn.get("mnemonic", ""))
        predicated_count = max(1, len(it_mnemonic) - 1)
        predicated_seen = 0
        last_predicated = None
        for insn in instructions:
            if int(insn.get("address", 0) or 0) <= int(it_insn.get("address", 0) or 0):
                continue
            if self._predicated_condition_for_mnemonic(insn.get("mnemonic", "")) is None:
                continue
            predicated_seen += 1
            last_predicated = insn

        if last_predicated is None:
            return None
        target = int(last_predicated.get("address", bb_address) or bb_address) + int(last_predicated.get("size", 2) or 2)
        fallthrough = target
        remaining = max(0, predicated_count - predicated_seen)
        cursor = target
        while remaining > 0:
            owner = self.instruction_to_bb.get(cursor)
            next_insn = None
            if owner is not None:
                for candidate in self.static_bbs.get(owner, []):
                    if int(candidate.get("address", 0) or 0) == cursor:
                        next_insn = candidate
                        break
            if next_insn is None:
                break
            fallthrough = cursor + int(next_insn.get("size", 2) or 2)
            cursor = fallthrough
            remaining -= 1

        return {
            "branch_pc": int(it_insn.get("address", bb_address) or bb_address),
            "target": int(target),
            "fallthrough": int(fallthrough),
        }

    def _build_cortex_m_system_instruction_map(self) -> Dict[int, Dict[str, object]]:
        """Collect Cortex-M system instructions that Unicorn may not decode."""
        system_mnemonics = {
            'MRS', 'MSR', 'MRC', 'MCR', 'ISB', 'DSB', 'DMB', 'CPSID', 'CPSIE',
        }
        result: Dict[int, Dict[str, object]] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                mnemonic = self._normalize_mnemonic(insn.get('mnemonic', ''))
                if mnemonic in system_mnemonics:
                    try:
                        result[int(insn['address'])] = insn
                    except Exception:
                        continue
        return result

    def _build_svc_instruction_pcs(self) -> Set[int]:
        """Collect SVC/SWI sites that need an OS/monitor model."""
        pcs: Set[int] = set()
        for instructions in self.static_bbs.values():
            for insn in instructions:
                if self._normalize_mnemonic(insn.get("mnemonic", "")) in {"SVC", "SWI"}:
                    try:
                        pcs.add(int(insn["address"]))
                    except Exception:
                        continue
        return pcs

    def _build_thumb_indirect_branch_instruction_map(self) -> Dict[int, Dict[str, object]]:
        """Collect register-indirect BLX/BX sites for Thumb target-bit repair."""
        result: Dict[int, Dict[str, object]] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
                if mnemonic not in {"BLX", "BX", "BXJ"}:
                    continue
                operands = self._split_operands(insn.get("operands", ""))
                if not operands:
                    continue
                if self._arm_reg_id(operands[0]) is None:
                    continue
                try:
                    result[int(insn["address"])] = insn
                except Exception:
                    continue
        return result

    def _is_mapped_mmio_load_candidate(self, insn: Dict[str, object]) -> bool:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        if not (mnemonic.startswith("LDR") or mnemonic in {"LDRB", "LDRH", "LDRSB", "LDRSH"}):
            return False
        if mnemonic in {"LDRD", "LDREX", "LDREXB", "LDREXH"}:
            return False
        operands = self._split_operands(insn.get("operands", ""))
        if len(operands) < 2:
            return False
        if operands[0].strip().lower() == "pc":
            return False
        return "[" in operands[1] and "]" in operands[1]

    def _build_mapped_mmio_load_instruction_map(self) -> Dict[int, Dict[str, object]]:
        result: Dict[int, Dict[str, object]] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                if not self._is_mapped_mmio_load_candidate(insn):
                    continue
                try:
                    result[int(insn["address"])] = insn
                except Exception:
                    continue
        return result

    def _build_static_mmio_prediction_stats(self) -> Dict[str, object]:
        accesses = list(getattr(self, "static_mmio_accesses", []) or [])
        exact = [
            item for item in accesses
            if isinstance(item, dict) and item.get("address") is not None
        ]
        base_only = [
            item for item in accesses
            if isinstance(item, dict) and item.get("address") is None
        ]
        reads = [item for item in accesses if str(item.get("access_type", "")) == "read"]
        writes = [item for item in accesses if str(item.get("access_type", "")) == "write"]
        exact_addresses = {
            int(item.get("address", 0)) & 0xFFFFFFFF
            for item in exact
            if item.get("address") is not None
        }
        summaries = getattr(self, "function_mmio_summaries", {}) or {}
        effectful_functions = 0
        for summary in summaries.values():
            if not isinstance(summary, dict):
                continue
            if summary.get("reads") or summary.get("writes") or summary.get("unresolved_reads") or summary.get("unresolved_writes"):
                effectful_functions += 1
        return {
            "total_accesses": len(accesses),
            "exact_accesses": len(exact),
            "base_only_accesses": len(base_only),
            "read_accesses": len(reads),
            "write_accesses": len(writes),
            "unique_exact_addresses": len(exact_addresses),
            "function_summaries": len(summaries),
            "effectful_functions": effectful_functions,
        }

    def _mmio_seed_report_stats(self) -> Dict[str, object]:
        """Seed table size + hit counters for the run report (design §2.3)."""
        handler_stats = dict(
            getattr(getattr(self, "mmio_handler", None), "mmio_seed_stats", {}) or {}
        )
        applied = int(handler_stats.get("applied", 0) or 0)
        table_size = len(getattr(self, "mmio_seed_values", {}) or {})
        return {
            "enabled": bool(getattr(self, "enable_static_mmio_seeds", False)),
            "table_size": table_size,
            **dict(getattr(self, "mmio_seed_derivation_meta", {}) or {}),
            "applied_first_reads": applied,
            "hit_rate": (round(applied / table_size, 4) if table_size else 0.0),
        }

    def _top_function_mmio_summaries(self, limit: int = 16) -> List[Dict[str, object]]:
        summaries = getattr(self, "function_mmio_summaries", {}) or {}
        ranked: List[Tuple[int, Dict[str, object]]] = []
        for entry, summary in summaries.items():
            if not isinstance(summary, dict):
                continue
            score = (
                len(summary.get("reads") or [])
                + len(summary.get("writes") or [])
                + int(summary.get("unresolved_reads", 0) or 0)
                + int(summary.get("unresolved_writes", 0) or 0)
            )
            if score <= 0:
                continue
            item = dict(summary)
            try:
                item["entry_addr"] = f"0x{int(item.get('entry_addr', entry)) & 0xFFFFFFFF:08x}"
            except Exception:
                item["entry_addr"] = f"0x{int(entry) & 0xFFFFFFFF:08x}"
            item["reads"] = [f"0x{int(value) & 0xFFFFFFFF:08x}" for value in (item.get("reads") or [])[:32]]
            item["writes"] = [f"0x{int(value) & 0xFFFFFFFF:08x}" for value in (item.get("writes") or [])[:32]]
            item["callees"] = [f"0x{int(value) & 0xFFFFFFFF:08x}" for value in (item.get("callees") or [])[:32]]
            item["callsites"] = [f"0x{int(value) & 0xFFFFFFFF:08x}" for value in (item.get("callsites") or [])[:32]]
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("entry_addr", ""))))
        return [item for _score, item in ranked[:max(0, int(limit))]]

    def _load_size_for_instruction(self, insn: Dict[str, object]) -> int:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        if mnemonic in {"LDRB", "LDRSB"}:
            return 1
        if mnemonic in {"LDRH", "LDRSH"}:
            return 2
        return 4

    def _extend_loaded_value(self, insn: Dict[str, object], value: int) -> int:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        value = int(value) & 0xFFFFFFFF
        if mnemonic == "LDRSB":
            value &= 0xFF
            if value & 0x80:
                value |= 0xFFFFFF00
        elif mnemonic == "LDRSH":
            value &= 0xFFFF
            if value & 0x8000:
                value |= 0xFFFF0000
        return value & 0xFFFFFFFF

    def _effective_address_for_load_operand(self, uc, operand: str, insn_address: int) -> Optional[int]:
        text = str(operand or "")
        match = re.search(r"\[([^\]]+)\]", text)
        if not match:
            return None
        parts = self._split_operands(match.group(1))
        if not parts:
            return None
        base_name = parts[0].strip().lower().rstrip("!")
        base_reg = self._arm_reg_id(base_name)
        if base_reg is None:
            return None
        try:
            if base_name == "pc":
                base = (int(insn_address) + (4 if self.execution_thumb else 8)) & 0xFFFFFFFF
            else:
                base = int(uc.reg_read(base_reg)) & 0xFFFFFFFF
        except Exception:
            return None

        if len(parts) < 2:
            return base

        offset_text = str(parts[1] or "").strip().lower()
        sign = 1
        if offset_text.startswith("-"):
            sign = -1
            offset_text = offset_text[1:].strip()
        elif offset_text.startswith("+"):
            offset_text = offset_text[1:].strip()
        offset_text = offset_text.replace("#", "").strip()

        offset = None
        try:
            offset = int(offset_text, 0)
        except Exception:
            index_reg = self._arm_reg_id(offset_text)
            if index_reg is None:
                return None
            try:
                offset = int(uc.reg_read(index_reg)) & 0xFFFFFFFF
                self.mapped_mmio_preload_stats["indexed_checked"] += 1
            except Exception:
                return None
            if len(parts) >= 3:
                shift_text = str(parts[2] or "").strip().lower().replace("#", "")
                shift_match = re.match(r"(lsl|lsr|asr)\s+(\d+)$", shift_text)
                if shift_match:
                    amount = int(shift_match.group(2))
                    if amount < 0 or amount > 31:
                        return None
                    if shift_match.group(1) == "lsl":
                        offset = (offset << amount) & 0xFFFFFFFF
                    else:
                        offset = (offset & 0xFFFFFFFF) >> amount
                else:
                    return None
        return (base + sign * offset) & 0xFFFFFFFF

    def _mapped_mmio_preload_hook(self, uc, address, size, user_data):
        insn = self.mapped_mmio_load_instruction_map.get(int(address))
        if insn is None:
            return
        if self._predicated_condition_for_mnemonic(insn.get("mnemonic", "")) is not None:
            return
        self.mapped_mmio_preload_stats["checked"] += 1
        try:
            operands = self._split_operands(insn.get("operands", ""))
            if len(operands) < 2:
                return
            dest_reg = self._arm_reg_id(operands[0])
            if dest_reg is None:
                return
            effective = self._effective_address_for_load_operand(uc, operands[1], int(address))
            if effective is None:
                return
            load_size = self._load_size_for_instruction(insn)
            decoded_alias = self._decode_bitband_alias(effective)
            is_mmio_effective = bool(
                self._is_mmio_address(effective)
                or (
                    decoded_alias is not None
                    and self._is_mmio_address(decoded_alias[0])
                )
            )
            if not is_mmio_effective:
                return

            # Do all address-space work before allocating an occurrence or
            # invoking the stateful handler.  If a backing page is absent, the
            # deferred signal aborts this callback and the same instruction is
            # retried after both the alias and backing ranges are mapped.
            if decoded_alias is not None:
                if not self._ensure_bitband_access_mapped(effective, load_size):
                    return
            elif not self._ensure_memory_mapped(effective, load_size):
                return

            # This path writes the destination register and advances PC inside
            # the code hook, so Unicorn never emits the ordinary MEM_READ
            # event.  Allocate the same causal occurrence that a normal MMIO
            # load would have received before serving the modeled value.
            causal_input_token = self._begin_causal_input_read(
                kind="mmio",
                pc=int(address),
                address=int(effective),
                size=load_size,
            )
            causal_input_token["delivery"] = "mapped_mmio_preload"
            input_source = "mapped_mmio_preload"

            alias_value = self._apply_bitband_alias_read(effective, int(address))
            if alias_value is not None:
                value = int(alias_value) & ((1 << (load_size * 8)) - 1)
                self.mapped_mmio_preload_stats["alias_applied"] += 1
                input_source = "mapped_mmio_preload_bitband_alias"
            else:
                value = int(self.mmio_handler.handle_read(effective, int(address), load_size))
            mask = (1 << (load_size * 8)) - 1
            uc.mem_write(effective, (value & mask).to_bytes(load_size, "little"))
            sampled_value = int(value) & mask
            self.loop_classifier.record_mmio_access(int(address), int(effective), True, sampled_value)
            self._record_mmio_access_history(int(address), int(effective), True, sampled_value)
            uc.reg_write(dest_reg, self._extend_loaded_value(insn, value))
            next_pc = int(address) + max(2, int(size or insn.get("size", 2) or 2))
            uc.reg_write(UC_ARM_REG_PC, (next_pc | 1) if self.execution_thumb else (next_pc & ~1))
            self._finish_causal_input_read(
                causal_input_token,
                value=sampled_value,
                source=input_source,
            )
            self.mapped_mmio_preload_stats["applied"] += 1
        except Exception as exc:
            self.mapped_mmio_preload_stats["failed"] += 1
            logger.debug("mapped MMIO preload失败 @ 0x%08x: %s", int(address), exc)

    def _it_state_guard_check(self, uc, address) -> None:
        """Detect stale ITSTATE sitting on an instruction that cannot be in an IT block.

        The predicate is purely architectural -- non-zero IT bits in xPSR plus an
        instruction of the unconditional ``B``/``B.W``/``BX``/``BLX``/``POP{pc}``
        family -- so it holds for any firmware and needs no address whitelist.

        On a hit we only raise a *flag* and break the translation block.  The IT
        bits cannot be cleared here: Unicorn writes ``condexec`` back to ``env``
        when the TB exits, so a clear performed inside the callback is undone.
        ``_managed_emu_start`` does the clear and the in-place re-entry instead.
        """
        if not getattr(self, "it_state_guard_enabled", False):
            return
        try:
            xpsr = int(uc.reg_read(UC_ARM_REG_XPSR)) & 0xFFFFFFFF
        except Exception:
            return
        if not (xpsr & IT_STATE_MASK):
            return
        self.it_state_guard_stats["checked"] += 1
        # 只取指令自己决定是否属于受保护族；判定不看地址白名单。
        try:
            hw1 = int.from_bytes(bytes(uc.mem_read(int(address) & ~1, 2)), "little")
            hw2 = None
            # 11101/11110/11111 前缀 ⇒ 32 位指令（0xE8BD 的 POP.W 也落在 0xE800 桶里）。
            if (hw1 & 0xF800) in (0xE800, 0xF000, 0xF800):
                hw2 = int.from_bytes(bytes(uc.mem_read((int(address) & ~1) + 2, 2)), "little")
        except Exception:
            self.it_state_guard_stats["failed"] += 1
            return
        kind = thumb_it_unpredictable_kind(hw1, hw2)
        if kind is None:
            return
        pc = int(address) & ~1
        self.it_state_guard_stats["trips"] += 1
        self._it_state_guard_sites[pc] += 1
        self._it_state_guard_kinds[kind] += 1
        self._it_state_guard_pending = True
        logger.debug(
            "脏ITSTATE守卫触发: pc=0x%08x kind=%s xpsr=0x%08x", pc, kind, xpsr
        )
        uc.emu_stop()

    def _repair_stale_itstate(self) -> bool:
        """Clear stale IT bits; return True when the caller should re-enter.

        Must be called *outside* ``uc.emu_start``: Unicorn writes ``condexec``
        back to ``env`` on translation-block exit, so a clear issued from inside a
        callback is overwritten (r28 段E).  Only the IT state bits are touched --
        no register value is injected and no branch direction is chosen.
        """
        if not getattr(self, "_it_state_guard_pending", False):
            return False
        self._it_state_guard_pending = False
        uc = getattr(self, "uc", None)
        if uc is None:
            return False
        try:
            xpsr = int(uc.reg_read(UC_ARM_REG_XPSR)) & 0xFFFFFFFF
            if xpsr & IT_STATE_MASK:
                uc.reg_write(UC_ARM_REG_XPSR, xpsr & ~IT_STATE_MASK)
            self.it_state_guard_stats["repaired"] += 1
        except Exception as exc:
            self.it_state_guard_stats["failed"] += 1
            logger.debug("脏ITSTATE清理失败: %s", exc)
            return False
        return True

    def _thumb_state_guard_hook(self, uc, address, size, user_data):
        """Keep Cortex-M ELF execution in Thumb state across modeled BLX/BX flows."""
        if not self.execution_thumb:
            return
        self._it_state_guard_check(uc, address)
        normalized = int(address) & ~1
        if not self._is_executable_address(normalized):
            return
        self.thumb_state_guard_stats["checked"] += 1
        try:
            cpsr = int(uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFF
            if cpsr & 0x20:
                return
            uc.reg_write(UC_ARM_REG_CPSR, cpsr | 0x20)
            self.thumb_state_guard_stats["repaired"] += 1
            logger.debug("修复Thumb状态位: pc=0x%08x cpsr=0x%08x", normalized, cpsr)
        except Exception as exc:
            self.thumb_state_guard_stats["failed"] += 1
            logger.debug("Thumb状态守护失败 @ 0x%08x: %s", normalized, exc)

    def _instruction_for_address(self, address: int) -> Optional[Dict[str, object]]:
        address = int(address) & ~1
        insn = self.code_analyzer.instruction_index.get(address)
        if insn is not None:
            return insn
        owner = self.instruction_to_bb.get(address)
        if owner is None:
            return None
        for candidate in self.static_bbs.get(owner, []):
            if int(candidate.get("address", 0) or 0) == address:
                return candidate
        return None

    def _try_recover_invalid_thumb_instruction(self, exc: Exception) -> bool:
        """Resume after Unicorn rejects a static-valid Thumb branch instruction."""
        if not self.execution_thumb:
            return False
        if "UC_ERR_INSN_INVALID" not in str(exc):
            return False
        self.invalid_thumb_recovery_stats["attempted"] += 1
        try:
            pc = int(self.uc.reg_read(UC_ARM_REG_PC)) & ~1
            insn = self._instruction_for_address(pc)
            if insn is None:
                self.invalid_thumb_recovery_stats["failed"] += 1
                return False
            condition = self._branch_condition_for_mnemonic(insn.get("mnemonic", ""))
            if condition is None or condition in {"CBZ", "CBNZ", "TBB", "TBH"}:
                self.invalid_thumb_recovery_stats["failed"] += 1
                return False
            target = self._parse_branch_target(insn.get("operands", ""))
            if target is None:
                self.invalid_thumb_recovery_stats["failed"] += 1
                return False
            cpsr = int(self.uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFF
            taken = self._condition_result_from_cpsr(condition, cpsr)
            if taken is None:
                self.invalid_thumb_recovery_stats["failed"] += 1
                return False
            fallthrough = pc + max(2, int(insn.get("size", 2) or 2))
            next_pc = int(target if taken else fallthrough) & ~1
            if not self._is_executable_address(next_pc):
                self.invalid_thumb_recovery_stats["failed"] += 1
                return False
            self.uc.reg_write(UC_ARM_REG_CPSR, cpsr | 0x20)
            self.uc.reg_write(UC_ARM_REG_PC, next_pc | 1)
            self.invalid_thumb_recovery_stats["recovered"] += 1
            logger.debug(
                "恢复Thumb条件分支解码异常: pc=0x%08x condition=%s taken=%s next=0x%08x",
                pc,
                condition,
                bool(taken),
                next_pc,
            )
            return True
        except Exception as recover_exc:
            self.invalid_thumb_recovery_stats["failed"] += 1
            logger.debug("Thumb非法指令恢复失败: %s", recover_exc)
            return False

    def _svc_number_from_instruction(self, insn: Optional[Dict[str, object]]) -> int:
        if not insn:
            return 0
        value = self._parse_int(str(insn.get("operands", "")).strip().lstrip("#"))
        return int(value or 0) & 0xFFFFFFFF

    def _thumb_indirect_branch_target_hook(self, uc, address, size, user_data):
        if not self.execution_thumb:
            return
        insn = self.thumb_indirect_branch_instruction_map.get(int(address))
        if insn is None:
            return
        operands = self._split_operands(insn.get("operands", ""))
        if not operands:
            return
        reg_id = self._arm_reg_id(operands[0])
        if reg_id is None:
            return
        self.thumb_indirect_branch_repair_stats["checked"] += 1
        try:
            target = int(uc.reg_read(reg_id)) & 0xFFFFFFFF
            if target & 1:
                return
            if not self._is_executable_address(target):
                self.thumb_indirect_branch_repair_stats["skipped_non_executable"] += 1
                return
            # ELF Cortex-M function pointers should carry the Thumb low bit.
            # Some static tables lose it under rehosting; repairing the bit keeps
            # execution in the same real target instead of decoding Thumb bytes
            # as ARM instructions.
            uc.reg_write(reg_id, target | 1)
            self.thumb_indirect_branch_repair_stats["repaired"] += 1
            logger.debug(
                "修复Thumb间接跳转低位: pc=0x%08x reg=%s target=0x%08x",
                int(address),
                operands[0],
                target | 1,
            )
        except Exception as exc:
            self.thumb_indirect_branch_repair_stats["failed"] += 1
            logger.debug("Thumb间接跳转修复失败 @ 0x%08x: %s", int(address), exc)

    def _svc_instruction_hook(self, uc, address, size, user_data):
        address = int(address) & ~1
        if address not in self.svc_instruction_pcs:
            return
        insn = self.code_analyzer.instruction_index.get(address)
        if insn is None:
            owner = self.instruction_to_bb.get(address)
            for candidate in self.static_bbs.get(int(owner or 0), []):
                if int(candidate.get("address", 0) or 0) == address:
                    insn = candidate
                    break
        svc_no = self._svc_number_from_instruction(insn)
        self.svc_stats["handled"] += 1
        by_number = self.svc_stats.setdefault("by_number", {})
        key = f"0x{svc_no:x}"
        by_number[key] = int(by_number.get(key, 0)) + 1

        # r32 D3：先把「执行到 svc 指令」这个架构事件交给上层派发器。派发器
        # （IrqDeliveryController）用 build_exception_entry 做一次真实异常入口，
        # 而不是跳过指令——判据是本条 svc 真的被执行到，与超时/判停规则无关。
        dispatcher = getattr(self, "svc_dispatch_handler", None)
        if callable(dispatcher):
            try:
                if dispatcher(address, insn, size, svc_no):
                    return
            except Exception as exc:  # 派发失败退回既有 no-op 语义
                self.svc_stats["dispatch_errors"] = (
                    int(self.svc_stats.get("dispatch_errors", 0)) + 1
                )
                logger.debug("SVC 派发失败 @ 0x%08x: %s", address, exc)

        # Raw vendor/RTOS firmware often uses SVC as a synchronous scheduler or
        # monitor service. Without the OS vector table Unicorn raises
        # UC_ERR_EXCEPTION. Treating it as a completed service preserves
        # entry-derived execution and lets the caller continue at the real
        # architectural return PC.
        insn_size = max(2, int(size or (insn or {}).get("size", 4) or 4))
        next_pc = address + insn_size
        if self.execution_thumb:
            next_pc |= 1
        else:
            next_pc &= ~1
        uc.reg_write(UC_ARM_REG_PC, next_pc)
        logger.debug("SVC/SWI no-op模型: pc=0x%08x svc=0x%x next=0x%08x", address, svc_no, next_pc)

    @staticmethod
    def _system_operand_tokens(operands: object) -> List[str]:
        return [
            token.strip().lower().lstrip('#')
            for token in str(operands or '').split(',')
            if token.strip()
        ]

    def _arm_reg_id(self, name: str) -> Optional[int]:
        normalized = str(name or '').strip().lower()
        return {
            'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1,
            'r2': UC_ARM_REG_R2, 'r3': UC_ARM_REG_R3,
            'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
            'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7,
            'r8': UC_ARM_REG_R8, 'r9': UC_ARM_REG_R9,
            'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
            'r12': UC_ARM_REG_R12, 'sp': UC_ARM_REG_SP,
            'sb': UC_ARM_REG_R9, 'sl': UC_ARM_REG_R10,
            'fp': UC_ARM_REG_R11, 'ip': UC_ARM_REG_R12,
            'lr': UC_ARM_REG_LR, 'pc': UC_ARM_REG_PC,
        }.get(normalized)

    def _read_operand_value(self, uc, operand: str) -> int:
        operand = str(operand or '').strip().lower()
        reg_id = self._arm_reg_id(operand)
        if reg_id is not None:
            return int(uc.reg_read(reg_id)) & 0xFFFFFFFF
        try:
            return int(operand.lstrip('#'), 0) & 0xFFFFFFFF
        except Exception:
            return 0

    @staticmethod
    def _cp15_key(operands: List[str]) -> str:
        if len(operands) < 6:
            return "cp15"
        return ",".join([operands[0], operands[1], operands[3], operands[4], operands[5]])

    def _cortex_m_system_instruction_hook(self, uc, address, size, user_data):
        insn = self.cortex_m_system_instruction_map.get(int(address))
        if insn is None:
            return

        mnemonic = self._normalize_mnemonic(insn.get('mnemonic', ''))
        operands = self._system_operand_tokens(insn.get('operands', ''))
        try:
            if mnemonic == 'MRS' and len(operands) >= 2:
                dest_reg = self._arm_reg_id(operands[0])
                sysreg = operands[1]
                if sysreg in {'msp', 'psp'}:
                    value = int(self.cortex_m_system_registers.get(sysreg, uc.reg_read(UC_ARM_REG_SP))) & 0xFFFFFFFF
                else:
                    value = int(self.cortex_m_system_registers.get(sysreg, 0)) & 0xFFFFFFFF
                if dest_reg is not None:
                    uc.reg_write(dest_reg, value)
            elif mnemonic == 'MSR' and len(operands) >= 2:
                sysreg = operands[0]
                value = self._read_operand_value(uc, operands[1])
                self.cortex_m_system_registers[sysreg] = value
                if sysreg in {'msp', 'psp'}:
                    # r32 D3：``msr psp, rX`` 在架构上**总是**写 PSP 这一 bank，
                    # 与当前模式无关。旧实现只写 UC_ARM_REG_SP——handler 模式下
                    # SP 就是 MSP，于是 `msr psp` 写错了 bank：SVC_Handler
                    # (0x0812ED34) 的 `mrs r3,psp / adds r3,#104 / msr psp,r3`
                    # 之后 unicorn 的 PSP 仍指向 SVC 自己的帧，异常返回会回到
                    # svc 指令形成自环。这里按 bank 写，并在写入的 bank 就是
                    # 当前 bank 时同步 SP（thread 态 = PSP，handler 态 = MSP）。
                    bank_reg = UC_ARM_REG_MSP if sysreg == 'msp' else UC_ARM_REG_PSP
                    uc.reg_write(bank_reg, value)
                    try:
                        in_handler = int(uc.reg_read(UC_ARM_REG_IPSR)) & 0x1FF != 0
                    except Exception:
                        in_handler = False
                    if (sysreg == 'msp') == in_handler:
                        uc.reg_write(UC_ARM_REG_SP, value)
                elif sysreg == 'control' and (value & 0x2) and 'psp' in self.cortex_m_system_registers:
                    uc.reg_write(UC_ARM_REG_SP, self.cortex_m_system_registers['psp'])
            elif mnemonic == 'MRC' and len(operands) >= 6 and operands[0] == 'p15':
                dest_reg = self._arm_reg_id(operands[2])
                key = self._cp15_key(operands)
                value = int(self.cortex_m_system_registers.get(key, 0)) & 0xFFFFFFFF
                if dest_reg is not None:
                    uc.reg_write(dest_reg, value)
            elif mnemonic == 'MCR' and len(operands) >= 6 and operands[0] == 'p15':
                src_reg = self._arm_reg_id(operands[2])
                key = self._cp15_key(operands)
                value = int(uc.reg_read(src_reg)) & 0xFFFFFFFF if src_reg is not None else 0
                self.cortex_m_system_registers[key] = value
            elif mnemonic == 'CPSID':
                self.cortex_m_system_registers['primask'] = 1
            elif mnemonic == 'CPSIE':
                self.cortex_m_system_registers['primask'] = 0
        except Exception as exc:
            logger.debug(f"系统指令仿真失败 @ 0x{int(address):08x}: {exc}")

        next_pc = int(address) + max(2, int(size or insn.get('size', 2) or 2))
        if self.execution_thumb:
            next_pc |= 1
        else:
            next_pc &= ~1
        uc.reg_write(UC_ARM_REG_PC, next_pc)

    def _branch_entry_instruction_hook(self, uc, address, size, user_data):
        """
        Save snapshots for static dispatch BBs that Ghidra split inside one
        Unicorn translation block.  Only capture when the instruction address
        is exactly the static BB start; otherwise root validation would reject
        the snapshot because PC would point at a mid-BB branch instruction.
        """
        if not self.enable_branch_snapshot:
            return
        bb_addr = self.branch_entry_pc_to_bb.get(int(address))
        if bb_addr is None or int(bb_addr) != int(address):
            return
        self._save_branch_entry_snapshot_if_needed(int(bb_addr))

    def _runtime_loop_branch_force_hook(self, uc, address, size, user_data):
        """
        Apply a proven MMIO loop-exit choice at the branch instruction itself.

        Some Unicorn memory-read hooks update the modeled MMIO value correctly
        but the current translated block can still observe the old flags/value.
        This hook is only installed after a local load/test/branch chain proves
        which branch outcome corresponds to the synthesized MMIO value, so it
        remains an entry-derived execution, not cold coverage injection.
        """
        force = self.runtime_loop_branch_forces.get(int(address))
        if not force:
            return
        condition = str(force.get("condition") or "")
        take_branch = bool(force.get("take_branch"))
        branch_bb = int(force.get("branch_bb") or 0)
        instructions = self.static_bbs.get(branch_bb, [])
        if not instructions:
            self.runtime_loop_branch_force_stats["failed"] += 1
            return
        branch_pc = int(force.get("branch_pc") or address)
        branch_insn = None
        for insn in instructions:
            if int(insn.get("address", 0) or 0) == branch_pc:
                branch_insn = insn
                break
        if branch_insn is None:
            branch_insn = instructions[-1]
        try:
            if condition in {"CBZ", "CBNZ"}:
                applied = self._force_cbz_cbnz_register(uc, branch_insn, take_branch)
            else:
                applied = self.branch_snapshot_manager._modify_cpsr(
                    uc,
                    self._condition_from_dispatch_token(condition),
                    take_branch,
                )
        except Exception:
            applied = False
        if applied:
            self.runtime_loop_branch_force_stats["applied"] += 1
        else:
            self.runtime_loop_branch_force_stats["failed"] += 1

    def _build_terminal_self_loop_bbs(self) -> Set[int]:
        """检测明显的终止sink，如 `_Error_Handler: b .` / `_exit: b .`。"""
        sinks: Set[int] = set()
        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            last_insn = instructions[-1]
            mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
            if mnemonic not in {'B', 'BAL'}:
                continue
            target = self._parse_branch_target(last_insn.get('operands', ''))
            if target is None:
                continue
            target_bb = self._resolve_bb_start(target)
            if target_bb == bb_addr:
                sinks.add(int(bb_addr))
        return sinks

    def _forced_branch_instruction_hook(self, uc, address, size, user_data):
        """在分支指令执行前设置CPSR，使分支按路径约束自然执行。"""
        if getattr(self, "_forced_branch_disabled", False):
            # r38 保险丝：即使有别的路径塞进 forced_branch_choices，
            # 总开关生效时也不改 CPSR、不写 PC，直接让位真实执行。
            self.forced_branch_disabled_blocks["hook"] += 1
            return
        bb_addr = self.branch_pc_to_bb.get(address)
        if bb_addr is None:
            return

        occurrence_index = self.forced_branch_encounters.get(bb_addr, 0) + 1
        self.forced_branch_encounters[bb_addr] = occurrence_index

        sequence_index = None
        if self.forced_branch_sequence:
            if self.forced_branch_sequence_index >= len(self.forced_branch_sequence):
                return
            expected_key, take_branch = self.forced_branch_sequence[self.forced_branch_sequence_index]
            if isinstance(expected_key, tuple):
                expected_bb = int(expected_key[0])
                expected_occurrence = int(expected_key[1])
                if bb_addr != expected_bb or occurrence_index != expected_occurrence:
                    return
            else:
                expected_bb = int(expected_key)
                if bb_addr != expected_bb:
                    return
            sequence_index = self.forced_branch_sequence_index
            choice_key = ("sequence", sequence_index, bb_addr, occurrence_index)
        else:
            event_key = (bb_addr, occurrence_index)
            if event_key in self.forced_branch_choices:
                choice_key = event_key
            elif bb_addr in self.forced_branch_choices and bb_addr not in self.forced_branch_hits:
                choice_key = bb_addr
            else:
                return
            take_branch = self.forced_branch_choices[choice_key]

        bb_instructions = self.static_bbs.get(bb_addr, [])
        if not bb_instructions:
            return

        last_insn = bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get('mnemonic', ''))
        condition = self._dispatch_condition_for_instruction(last_insn)
        if not condition:
            return

        switch_reg_name = None
        switch_reg_before = None
        if self._is_switch_dispatch_condition(condition):
            if condition == 'LDRPC':
                parsed_switch = self._parse_ldr_pc_switch_operands(last_insn.get('operands', ''))
                switch_reg_name = parsed_switch[1] if parsed_switch else None
            else:
                switch_reg_name = self._parse_switch_index_register(last_insn.get('operands', ''))
            if switch_reg_name is not None:
                try:
                    reg_lookup = {
                        'r0': UC_ARM_REG_R0, 'r1': UC_ARM_REG_R1, 'r2': UC_ARM_REG_R2,
                        'r3': UC_ARM_REG_R3, 'r4': UC_ARM_REG_R4, 'r5': UC_ARM_REG_R5,
                        'r6': UC_ARM_REG_R6, 'r7': UC_ARM_REG_R7, 'r8': UC_ARM_REG_R8,
                        'r9': UC_ARM_REG_R9, 'r10': UC_ARM_REG_R10, 'r11': UC_ARM_REG_R11,
                        'r12': UC_ARM_REG_R12,
                    }
                    reg_id = reg_lookup.get(switch_reg_name.lower())
                    if reg_id is not None:
                        switch_reg_before = uc.reg_read(reg_id) & 0xFFFFFFFF
                except Exception:
                    switch_reg_before = None

        if self._is_call_dispatch_condition(condition):
            forced = self._force_call_dispatch_target(uc, last_insn, take_branch)
        elif condition in {'CBZ', 'CBNZ'}:
            forced = self._force_cbz_cbnz_register(uc, last_insn, take_branch)
        elif self._is_switch_dispatch_condition(condition):
            forced = (
                self._force_ldr_pc_switch_index(uc, last_insn, take_branch)
                if condition == 'LDRPC'
                else self._force_switch_index(uc, last_insn, take_branch)
            )
        else:
            forced = self.branch_snapshot_manager._modify_cpsr(
                uc,
                self._condition_from_dispatch_token(condition),
                take_branch,
            )

        if forced:
            branch_info = self._get_conditional_branch_info(bb_addr)
            if not branch_info:
                return
            if self._is_call_dispatch_condition(condition):
                desired_addr = int(take_branch) & ~1
            elif self._is_switch_dispatch_condition(condition):
                switch_targets = branch_info.get("targets", [])
                try:
                    target_index = int(take_branch)
                except (TypeError, ValueError):
                    return
                if target_index < 0 or target_index >= len(switch_targets):
                    return
                desired_addr = switch_targets[target_index]
                if desired_addr == 0:
                    return
            else:
                desired_addr = branch_info["target"] if take_branch else branch_info["fallthrough"]
            desired_bb = self._resolve_bb_start(desired_addr)
            self.forced_branch_pending[bb_addr] = {
                "desired_bb": desired_bb,
                "direction": int(take_branch),
                "occurrence_index": occurrence_index,
                "choice_key": choice_key,
                "branch_pc": address,
            }
            if self._is_switch_dispatch_condition(condition):
                self.forced_branch_pending[bb_addr]["requested_index"] = int(take_branch)
                if switch_reg_name is not None:
                    self.forced_branch_pending[bb_addr]["switch_reg_name"] = switch_reg_name
                if switch_reg_before is not None:
                    self.forced_branch_pending[bb_addr]["switch_reg_before"] = int(switch_reg_before)
            if sequence_index is not None:
                self.forced_branch_pending[bb_addr]["sequence_index"] = sequence_index
                self.forced_branch_sequence_index += 1
            self.forced_branch_hits.add(choice_key)
            trace_item = {
                "branch": bb_addr,
                "direction": int(take_branch),
                "branch_pc": address,
                "occurrence_index": occurrence_index,
                "verified": False,
            }
            if sequence_index is not None:
                trace_item["sequence_index"] = sequence_index
            if self._is_switch_dispatch_condition(condition):
                trace_item["requested_index"] = int(take_branch)
                if switch_reg_name is not None:
                    trace_item["switch_reg_name"] = switch_reg_name
                if switch_reg_before is not None:
                    trace_item["switch_reg_before"] = int(switch_reg_before)
                trace_item["desired_bb"] = int(desired_bb)
            self.forced_branch_trace.append(trace_item)
            self.forced_branch_pending[bb_addr]["trace_index"] = len(self.forced_branch_trace) - 1

    def _resolve_bb_start(self, address: int) -> int:
        """把指令地址解析为静态BB起始地址。"""
        if address in self.static_bbs:
            return address
        bb_addr = self.instruction_to_bb.get(address)
        if bb_addr is not None:
            return bb_addr
        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            first_addr = instructions[0].get('address', bb_addr)
            last_addr = instructions[-1].get('address', bb_addr)
            if first_addr <= address <= last_addr:
                return bb_addr
        return address

    def _verify_pending_forced_branch(self, prev_bb: int, current_bb: int) -> bool:
        """验证刚约束过的分支是否走到目标方向；不符合则终止当前路径。"""
        pending = self.forced_branch_pending.pop(prev_bb, None)
        if not pending:
            return False
        desired_bb = pending["desired_bb"]

        if current_bb == desired_bb:
            choice_key = pending.get("choice_key")
            if choice_key is not None:
                self.forced_branch_hits.add(choice_key)
            trace_index = pending.get("trace_index")
            if isinstance(trace_index, int) and 0 <= trace_index < len(self.forced_branch_trace):
                self.forced_branch_trace[trace_index]["verified"] = True
                self.forced_branch_trace[trace_index]["actual_bb"] = int(current_bb)
            else:
                trace_item = {
                    "branch": prev_bb,
                    "direction": int(pending.get("direction", 0)),
                    "branch_pc": int(pending.get("branch_pc", prev_bb)),
                    "occurrence_index": int(pending.get("occurrence_index", 1)),
                    "verified": True,
                    "actual_bb": int(current_bb),
                }
                if "sequence_index" in pending:
                    trace_item["sequence_index"] = int(pending["sequence_index"])
                self.forced_branch_trace.append(trace_item)
            return False

        trace_index = pending.get("trace_index")
        if isinstance(trace_index, int) and 0 <= trace_index < len(self.forced_branch_trace):
            self.forced_branch_trace[trace_index]["verified"] = False
            self.forced_branch_trace[trace_index]["actual_bb"] = int(current_bb)
            self.forced_branch_trace[trace_index]["desired_bb"] = int(desired_bb)
        logger.debug(
            f"路径约束方向不一致 @ 0x{prev_bb:08x}: "
            f"expected 0x{desired_bb:08x}, got 0x{current_bb:08x}，终止当前路径"
        )
        self.uc.emu_stop()
        return True

    def _save_branch_snapshot_if_needed(self, prev_address: int, curr_address: int):
        """
        在分支点保存快照（关键优化）

        Args:
            prev_address: 前一个BB地址（可能是分支点）
            curr_address: 当前BB地址
        """
        # 获取前一个BB的指令
        bb_instructions = self.static_bbs.get(prev_address, [])
        if not bb_instructions:
            return

        # 检查最后一条指令是否是可控分支/动态分发点。
        last_insn = bb_instructions[-1]
        condition = self._dispatch_condition_for_instruction(last_insn)

        if condition is None:
            direct_call_info = self._get_direct_call_info(prev_address)
            if direct_call_info is not None:
                try:
                    self.path_explorer.record_branch(
                        address=prev_address,
                        insn_mnemonic="BL",
                        taken=True,
                        target=direct_call_info["target"],
                        fallthrough=direct_call_info["fallthrough"],
                    )
                except Exception:
                    pass
            return

        # 解析目标地址
        operands = last_insn.get('operands', '')
        try:
            # Register-indirect BX/BXJ/BLX targets are only valid once the
            # dispatch BB has actually executed. Record the concrete successor
            # observed by Unicorn so later frontier replays remain entry-derived.
            if self._is_call_dispatch_condition(condition):
                branch_pc = last_insn.get('address', prev_address)
                fallthrough = branch_pc + last_insn.get('size', 2)
                target = curr_address
                target_bb = self._resolve_bb_start(target)
                fallthrough_bb = self._resolve_bb_start(fallthrough)
                original_taken = curr_address != fallthrough_bb
                event = self.branch_snapshot_manager.record_event(
                    address=prev_address,
                    branch_pc=branch_pc,
                    target=target_bb if target_bb is not None else target,
                    fallthrough=fallthrough if condition == 'BLX' else 0,
                    condition=condition,
                    original_taken=original_taken,
                    depth=self._bb_history_depth(),
                    alternatives=[
                        int(target_bb if target_bb is not None else target),
                        int(fallthrough),
                    ] if condition == 'BLX' else [
                        int(target_bb if target_bb is not None else target),
                    ],
                    original_index=0,
                    context_signature=self.causal_context.branch_signature(),
                )

                has_snapshot = self.branch_snapshot_manager.has_snapshot(prev_address)
                if has_snapshot:
                    return

                self.branch_snapshot_manager.save_snapshot(
                    self.uc,
                    prev_address,
                    target_bb if target_bb is not None else target,
                    fallthrough if condition == 'BLX' else 0,
                    condition,
                    original_taken=original_taken,
                    depth=self._bb_history_depth(),
                    mmio_state=self._current_mmio_state(),
                    alternatives=[
                        int(target_bb if target_bb is not None else target),
                        int(fallthrough),
                    ] if condition == 'BLX' else [
                        int(target_bb if target_bb is not None else target),
                    ],
                    original_index=0,
                    occurrence_index=getattr(event, "occurrence_index", 1),
                )
                logger.debug(f"保存动态分发快照 @ 0x{prev_address:08x} ({condition})")
                return

            # TBB/TBH and LDR-PC jump tables are multi-way dispatches.  Record
            # the concrete table and the index observed on an entry-derived run.
            if self._is_switch_dispatch_condition(condition):
                branch_pc = last_insn.get('address', prev_address)
                targets = (
                    self._read_ldr_pc_switch_targets(self.uc, prev_address, last_insn)
                    if condition == 'LDRPC'
                    else self._read_switch_targets(self.uc, prev_address, last_insn)
                )
                if not targets:
                    return

                original_index = None
                actual_target = 0
                for index, target_candidate in enumerate(targets):
                    if target_candidate == 0:
                        continue
                    if curr_address == self._resolve_bb_start(target_candidate & ~1):
                        original_index = index
                        actual_target = target_candidate
                        break
                if original_index is None:
                    return

                event = self.branch_snapshot_manager.record_event(
                    address=prev_address,
                    branch_pc=branch_pc,
                    target=actual_target,
                    fallthrough=0,
                    condition=condition,
                    original_taken=True,
                    depth=self._bb_history_depth(),
                    alternatives=targets,
                    original_index=original_index,
                    context_signature=self.causal_context.branch_signature(),
                )

                has_snapshot = self.branch_snapshot_manager.has_snapshot(prev_address)
                if has_snapshot:
                    return

                self.branch_snapshot_manager.save_snapshot(
                    self.uc,
                    prev_address,
                    actual_target,
                    0,
                    condition,
                    original_taken=True,
                    depth=self._bb_history_depth(),
                    mmio_state=self._current_mmio_state(),
                    alternatives=targets,
                    original_index=original_index,
                    occurrence_index=getattr(event, "occurrence_index", 1),
                )
                logger.debug(f"保存switch快照 @ 0x{prev_address:08x} ({condition}, index={original_index})")
                return

            if str(condition).startswith("IT"):
                branch_info = self._get_conditional_branch_info(prev_address)
                if not branch_info:
                    return
                target = int(branch_info.get("target", 0) or 0)
                fallthrough = int(branch_info.get("fallthrough", 0) or 0)
                target_bb = self._resolve_bb_start(target)
                fallthrough_bb = self._resolve_bb_start(fallthrough)
                taken = curr_address == target_bb
                if curr_address not in {target_bb, fallthrough_bb}:
                    taken = curr_address == target

                event = self.branch_snapshot_manager.record_event(
                    address=prev_address,
                    branch_pc=int(branch_info.get("branch_pc", prev_address) or prev_address),
                    target=target,
                    fallthrough=fallthrough,
                    condition=condition,
                    original_taken=taken,
                    depth=self._bb_history_depth(),
                    context_signature=self.causal_context.branch_signature(),
                )

                has_snapshot = self.branch_snapshot_manager.has_snapshot(prev_address)
                if has_snapshot:
                    return

                self.branch_snapshot_manager.save_snapshot(
                    self.uc,
                    prev_address,
                    target,
                    fallthrough,
                    condition,
                    original_taken=taken,
                    depth=self._bb_history_depth(),
                    mmio_state=self._current_mmio_state(),
                    occurrence_index=getattr(event, "occurrence_index", 1),
                )
                logger.debug(f"保存IT谓词快照 @ 0x{prev_address:08x} ({condition})")
                return

            target = self._parse_branch_target(operands)
            if target is None:
                return

            # 计算fallthrough地址
            branch_pc = last_insn.get('address', prev_address)
            fallthrough = branch_pc + last_insn.get('size', 2)
            target_bb = self._resolve_bb_start(target)
            fallthrough_bb = self._resolve_bb_start(fallthrough)
            taken = curr_address == target_bb
            if curr_address not in {target_bb, fallthrough_bb}:
                taken = curr_address == target

            event = self.branch_snapshot_manager.record_event(
                address=prev_address,
                branch_pc=branch_pc,
                target=target,
                fallthrough=fallthrough,
                condition=condition,
                original_taken=taken,
                depth=self._bb_history_depth(),
                context_signature=self.causal_context.branch_signature(),
            )

            # 当前快照仍按地址保留第一份稳定入口态；历史快照保留少量不同 occurrence/前缀状态。
            has_snapshot = self.branch_snapshot_manager.has_snapshot(prev_address)
            if has_snapshot:
                return

            # 保存快照
            self.branch_snapshot_manager.save_snapshot(
                self.uc,
                prev_address,
                target,
                fallthrough,
                condition,
                original_taken=taken,
                depth=self._bb_history_depth(),
                mmio_state=self._current_mmio_state(),
                occurrence_index=getattr(event, "occurrence_index", 1),
            )

            logger.debug(f"保存分支快照 @ 0x{prev_address:08x} ({condition})")

        except Exception as e:
            logger.debug(f"保存分支快照失败 @ 0x{prev_address:08x}: {e}")

    def _save_branch_entry_snapshot_if_needed(self, bb_address: int):
        """在进入条件分支BB时立刻保存入口态，供后续真正从该BB重放。"""
        branch_info = self._get_conditional_branch_info(bb_address)
        if not branch_info:
            self._save_direct_call_entry_snapshot_if_needed(bb_address)
            return

        bb_instructions = self.static_bbs.get(bb_address, [])
        if not bb_instructions:
            return

        last_insn = bb_instructions[-1]
        condition = self._dispatch_condition_for_instruction(last_insn)
        if not condition:
            return

        has_snapshot = self.branch_snapshot_manager.has_snapshot(bb_address)
        guided_replay_active = bool(self.forced_branch_choices or self.forced_branch_sequence)
        after_forced_divergence = bool(self.forced_branch_trace)
        refresh_switch_snapshot = (
            guided_replay_active
            and self._is_switch_dispatch_condition(condition)
            and (
                after_forced_divergence
                or bb_address in self.forced_branch_target_bbs
            )
        )
        should_refresh = bb_address in self.branch_snapshot_hotset or refresh_switch_snapshot
        predicted_occurrence = self.branch_snapshot_manager.occurrence_counts.get(bb_address, 0) + 1
        save_history_entry = (
            has_snapshot
            and not should_refresh
            and self.branch_snapshot_manager.should_save_history_snapshot(
                bb_address,
                predicted_occurrence,
            )
        )
        if has_snapshot and not should_refresh and not save_history_entry:
            return

        target = branch_info.get("target", 0)
        fallthrough = branch_info.get("fallthrough", 0)
        targets = None
        if self._is_switch_dispatch_condition(condition):
            targets = branch_info.get("targets", [])
            target = targets[0] if targets else 0
            fallthrough = 0
        elif self._is_call_dispatch_condition(condition):
            target = branch_info.get("target", 0)
            fallthrough = branch_info.get("fallthrough", 0)
            if condition == 'BLX':
                targets = [target, fallthrough] if fallthrough else [target]
            else:
                targets = [target] if target else []

        try:
            self.branch_snapshot_manager.save_snapshot(
                self.uc,
                bb_address,
                target,
                fallthrough,
                condition,
                original_taken=False,
                depth=self._bb_history_depth(),
                preserve_order=has_snapshot or should_refresh,
                mmio_state=self._current_mmio_state(),
                alternatives=targets if self._is_switch_dispatch_condition(condition) or self._is_call_dispatch_condition(condition) else None,
                original_index=None,
                occurrence_index=predicted_occurrence,
                update_current=not save_history_entry,
            )
            logger.debug(f"保存分支入口快照 @ 0x{bb_address:08x} ({condition})")
        except Exception as e:
            logger.debug(f"保存分支入口快照失败 @ 0x{bb_address:08x}: {e}")

    def _save_direct_call_entry_snapshot_if_needed(self, bb_address: int):
        """Save direct BL entry state for later continuation replay from an entry-derived context."""
        call_info = self._get_direct_call_info(bb_address)
        if not call_info:
            return

        target = int(call_info.get("target", 0) or 0)
        fallthrough = int(call_info.get("fallthrough", 0) or 0)
        if target <= 0 and fallthrough <= 0:
            return
        predicted_occurrence = self.branch_snapshot_manager.occurrence_counts.get(bb_address, 0) + 1
        has_snapshot = self.branch_snapshot_manager.has_snapshot(bb_address)
        save_history_entry = (
            has_snapshot
            and self.branch_snapshot_manager.should_save_history_snapshot(
                bb_address,
                predicted_occurrence,
            )
        )
        if has_snapshot and not save_history_entry:
            try:
                existing = self.branch_snapshot_manager.get_snapshot(bb_address)
                if self._normalize_mnemonic(getattr(existing, "condition", "")) == "BL":
                    return
            except Exception:
                return
        try:
            self.branch_snapshot_manager.save_snapshot(
                self.uc,
                bb_address,
                target,
                fallthrough,
                "BL",
                original_taken=True,
                depth=self._bb_history_depth(),
                mmio_state=self._current_mmio_state(),
                alternatives=[
                    item for item in (target, fallthrough) if item
                ],
                original_index=0,
                occurrence_index=predicted_occurrence,
                update_current=not save_history_entry,
            )
            logger.debug(f"保存direct-call入口快照 @ 0x{bb_address:08x} (BL)")
        except Exception as e:
            logger.debug(f"保存direct-call入口快照失败 @ 0x{bb_address:08x}: {e}")

    def _execution_counter_snapshot(self) -> Dict[str, object]:
        """Capture numeric intervention counters for one-run delta accounting."""
        nested_names = (
            "runtime_loop_branch_force_stats",
            "skip_function_stats",
            "lzo_decompress_summary_stats",
            "external_rom_call_stats",
            "svc_stats",
            "thumb_state_guard_stats",
            "it_state_guard_stats",
            "invalid_thumb_recovery_stats",
            "thumb_indirect_branch_repair_stats",
            "execution_preflight_stats",
            "environment_input_delivery_stats",
            # r9 裁定：干预事件家族标签（与 intervention_count 一一对应）。
            "intervention_event_labels",
        )
        result: Dict[str, object] = {
            "intervention_count": int(getattr(self, "intervention_count", 0) or 0),
            "forced_branch_trace_count": len(
                list(getattr(self, "forced_branch_trace", []) or [])
            ),
            "loop_unresolved_limit_trips": int(
                getattr(self, "loop_unresolved_limit_trips", 0) or 0
            ),
        }
        for name in nested_names:
            value = getattr(self, name, {})
            if not isinstance(value, dict):
                result[name] = {}
                continue
            numeric = {}
            for key, item in value.items():
                try:
                    numeric[str(key)] = int(item or 0)
                except (TypeError, ValueError, OverflowError):
                    continue
            result[name] = numeric
        # RX/input-ready is maintained by the primary stateful MMIO model,
        # not as a counter on the emulator itself.  Include its cumulative
        # counters here so direct ``emu_start`` replays and normal ``run()``
        # executions can both compute an exact per-attempt delta.
        # A replay overlay may be passed around as ``mmio_handler`` by the
        # runner, but it is not the semantic owner of the Unicorn read.  Use
        # the registered primary handler whenever available so normal
        # ``run()`` records and direct-replay records share one accounting
        # source.  Construction happens before registration, hence the
        # emulator-owned handler remains the compatibility fallback.
        primary_handler = get_primary_mmio_handler(getattr(self, "uc", None))
        peripheral_owner = primary_handler or getattr(self, "mmio_handler", None)
        peripheral_stats = getattr(
            peripheral_owner,
            "peripheral_input_stats",
            {},
        )
        if isinstance(peripheral_stats, Mapping):
            result["peripheral_input_stats"] = {
                str(key): int(value or 0)
                for key, value in peripheral_stats.items()
                if isinstance(value, (int, float))
            }
        else:
            result["peripheral_input_stats"] = {}
        return result

    @staticmethod
    def _execution_counter_delta(
        current: Dict[str, object],
        baseline: Dict[str, object],
    ) -> Dict[str, object]:
        """Subtract two compact counter snapshots without changing their shape."""
        result: Dict[str, object] = {}
        for key, value in current.items():
            old = baseline.get(key, 0)
            if isinstance(value, dict):
                old_map = old if isinstance(old, dict) else {}
                nested = {}
                for nested_key, nested_value in value.items():
                    try:
                        delta = int(nested_value or 0) - int(old_map.get(nested_key, 0) or 0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if delta > 0:
                        nested[nested_key] = delta
                result[key] = nested
                continue
            try:
                delta = int(value or 0) - int(old or 0)
            except (TypeError, ValueError, OverflowError):
                delta = 0
            result[key] = max(0, delta)
        return result

    def _current_execution_provenance(self) -> Dict[str, object]:
        """Return capture-time provenance, including the restored prefix lineage."""
        current = self._execution_counter_snapshot()
        delta = self._execution_counter_delta(
            current,
            dict(getattr(self, "_execution_counter_baseline", {}) or {}),
        )
        trace = list(getattr(self, "forced_branch_trace", []) or [])
        trace_start = max(0, int(getattr(self, "_forced_trace_baseline_len", 0) or 0))
        trace_since_start = trace[trace_start:]
        if trace_since_start:
            delta["forced_branch_trace"] = trace_since_start
        actual_reasons = execution_intervention_reasons(delta)

        ancestor = dict(getattr(self, "_active_prefix_provenance", {}) or {})
        ancestor_reasons = [
            str(reason)
            for reason in (
                ancestor.get("reasons")
                or ancestor.get("intervention_reasons")
                or ancestor.get("prefix_intervention_reasons")
                or ()
            )
            if str(reason)
        ]
        combined_reasons = list(dict.fromkeys(ancestor_reasons + actual_reasons))
        ancestor_invalidated = coerce_bool(
            ancestor.get("provenance_invalidated"),
            False,
        )
        ancestor_invalidation_reasons = [
            str(reason)
            for reason in (
                ancestor.get("provenance_invalidation_reasons") or ()
            )
            if str(reason)
        ]
        if ancestor_invalidated and "provenance_invalidated" not in combined_reasons:
            combined_reasons.append("provenance_invalidated")
        combined_reasons.extend(
            reason
            for reason in ancestor_invalidation_reasons
            if reason not in combined_reasons
        )
        ancestor_status = str(
            ancestor.get("status") or ancestor.get("classification") or ""
        ).strip().lower()
        ancestor_finalization_present = "provenance_finalized" in ancestor
        ancestor_finalized = coerce_bool(
            ancestor.get("provenance_finalized"),
            False,
        )
        if ancestor_status in {
            "diagnostic",
            "pending",
            "unverified",
            "unknown",
            "unspecified",
            "incomplete",
            "invalid",
            "",
        } and not combined_reasons and ancestor:
            combined_reasons.append("prefix_ancestor_not_validated")
        if ancestor and not ancestor_finalization_present:
            combined_reasons.append("prefix_provenance_finalization_missing")
        elif ancestor and not ancestor_finalized:
            combined_reasons.append("prefix_provenance_not_finalized")
        telemetry_complete = coerce_bool(
            ancestor.get("telemetry_complete"),
            True,
        )
        execution_active = bool(getattr(self, "_execution_active", False))
        # A live run has observed a concrete prefix, but its lineage record is
        # not final until the run reaches the result boundary.  Snapshot
        # capture uses this bit to defer promotion without changing execution.
        if execution_active:
            status = "pending"
            prefix_telemetry_complete = False
        else:
            status = (
                "diagnostic"
                if combined_reasons or not telemetry_complete or ancestor_invalidated
                else "validated"
            )
            prefix_telemetry_complete = bool(telemetry_complete)
        configured = configured_intervention_counts({
            "forced_branch_choices_configured": len(
                dict(getattr(self, "forced_branch_choices", {}) or {})
            ),
            "forced_choices_requested": len(
                list(getattr(self, "forced_branch_sequence", []) or [])
            ),
            "runtime_loop_branch_force_stats": getattr(
                self, "runtime_loop_branch_force_stats", {}
            ),
            "skip_function_stats": getattr(self, "skip_function_stats", {}),
        })
        return {
            "status": status,
            "reasons": combined_reasons,
            "intervention_reasons": list(actual_reasons),
            "prefix_intervention_reasons": ancestor_reasons,
            "telemetry_complete": telemetry_complete,
            "prefix_telemetry_complete": prefix_telemetry_complete,
            "execution_active": execution_active,
            "prefix_observed": True,
            "provenance_invalidated": bool(ancestor_invalidated),
            "provenance_invalidation_reasons": tuple(
                ancestor_invalidation_reasons
            ),
            "snapshot_ancestor_validated": bool(
                bool(ancestor)
                and ancestor_status == "validated"
                and ancestor_finalized
                and telemetry_complete
                and not ancestor_reasons
                and not ancestor_invalidated
            ),
            "prefix_provenance_status": (
                ancestor_status if ancestor else None
            ),
            "snapshot_source_execution_id": (
                str(ancestor.get("execution_id") or "") if ancestor else ""
            ),
            "provenance_finalized": bool(
                not execution_active
                and (not ancestor or ancestor_finalized)
            ),
            "execution_id": str(getattr(self, "execution_id", "") or ""),
            "configured_intervention_counts": dict(configured),
            # r38 审计面：随 provenance 透传到相位记录，campaign 级可核验
            # （总开关是否生效 + 该仿真器上被挡下的 force 请求累计）。
            "forced_branch_disabled": bool(
                getattr(self, "_forced_branch_disabled", False)
            ),
            "forced_branch_disabled_blocks": {
                str(key): int(value)
                for key, value in dict(
                    getattr(self, "forced_branch_disabled_blocks", {}) or {}
                ).items()
            },
        }

    def _advance_active_prefix_provenance_from_run(
        self,
        run_result: Mapping[str, object],
    ) -> None:
        """Carry one completed ``run()`` into a later continuation.

        Some replay stages intentionally stop at a callsite and then invoke
        ``run(preserve_cpu_state=True)`` again after materializing a return
        value.  The second invocation must inherit the first invocation's
        intervention facts; otherwise a summary/skip used in the prefix can
        disappear from the final suffix witness.  This only updates lineage
        metadata and never changes CPU, MMIO, or scheduling state.
        """
        if not isinstance(run_result, Mapping):
            return
        provenance = run_result.get("execution_provenance")
        provenance_map = dict(provenance) if isinstance(provenance, Mapping) else {}
        reasons: List[str] = []
        for source in (
            provenance_map.get("reasons"),
            provenance_map.get("intervention_reasons"),
            run_result.get("execution_intervention_reasons"),
        ):
            if isinstance(source, Mapping):
                values = source.keys()
            elif isinstance(source, str):
                values = (source,)
            else:
                values = source or ()
            for reason in values:
                text = str(reason)
                if (
                    text
                    and not is_environment_fact_reason(text)
                    and text not in reasons
                ):
                    reasons.append(text)

        execution_failed = coerce_bool(run_result.get("execution_failed"), False)
        preflight_failed = coerce_bool(run_result.get("preflight_failed"), False)
        run_invalidated = coerce_bool(
            run_result.get("provenance_invalidated"),
            False,
        )
        telemetry_complete = coerce_bool(
            run_result.get("execution_telemetry_complete"), False
        )
        if execution_failed or preflight_failed or not telemetry_complete:
            failure_reason = str(
                run_result.get("failure_reason")
                or run_result.get("stop_reason")
                or "execution_prefix_incomplete"
            )
            marker = f"execution_prefix_incomplete:{failure_reason}"
            if marker not in reasons:
                reasons.append(marker)
        if run_invalidated and "provenance_invalidated" not in reasons:
            reasons.append("provenance_invalidated")

        previous = dict(getattr(self, "_active_prefix_provenance", {}) or {})
        previous_status = str(
            previous.get("status") or previous.get("provenance_status") or ""
        ).strip().lower()
        previous_reasons = [
            str(reason)
            for reason in (
                previous.get("reasons")
                or previous.get("intervention_reasons")
                or previous.get("prefix_intervention_reasons")
                or ()
            )
            if str(reason) and not is_environment_fact_reason(reason)
        ]
        previous_invalidated = coerce_bool(
            previous.get("provenance_invalidated"),
            False,
        )
        previous_invalidation_reasons = [
            str(reason)
            for reason in (
                previous.get("provenance_invalidation_reasons") or ()
            )
            if str(reason)
        ]
        if previous_invalidated and "provenance_invalidated" not in previous_reasons:
            previous_reasons.append("provenance_invalidated")
        previous_reasons.extend(
            reason
            for reason in previous_invalidation_reasons
            if reason not in previous_reasons
        )
        all_reasons = list(dict.fromkeys(previous_reasons + reasons))
        previous_complete = bool(
            not previous
            or (
                previous_status == "validated"
                and coerce_bool(previous.get("telemetry_complete"), False)
                and coerce_bool(previous.get("provenance_finalized"), False)
                and not previous_reasons
            )
        )
        whole_prefix_complete = bool(telemetry_complete and previous_complete)
        status = (
            "validated"
            if whole_prefix_complete and not all_reasons and not previous_invalidated
            else "diagnostic"
        )
        self._active_prefix_provenance = {
            "status": status,
            "reasons": tuple(all_reasons),
            "prefix_intervention_reasons": tuple(all_reasons),
            "telemetry_complete": whole_prefix_complete,
            "provenance_finalized": True,
            "provenance_invalidated": bool(previous_invalidated or run_invalidated),
            "provenance_invalidation_reasons": tuple(
                dict.fromkeys(
                    previous_invalidation_reasons
                    + (["provenance_invalidated"] if run_invalidated else [])
                )
            ),
            "execution_id": str(run_result.get("execution_id") or ""),
        }

    def _set_active_prefix_provenance(self, snapshot: object) -> None:
        """Remember the lineage of a restored snapshot for the next replay."""
        if snapshot is None:
            self._active_prefix_provenance = {}
            return
        provenance = snapshot_provenance_record(snapshot)
        self._active_prefix_provenance = {
            "status": str(provenance.get("status") or "unverified"),
            "reasons": tuple(provenance.get("reasons", []) or []),
            "prefix_intervention_reasons": tuple(
                provenance.get("prefix_intervention_reasons", []) or []
            ),
            "telemetry_complete": bool(
                provenance.get("telemetry_complete", False)
            ),
            "provenance_finalized": bool(
                provenance.get("provenance_finalized", False)
            ),
            "provenance_invalidated": bool(
                provenance.get("provenance_invalidated", False)
            ),
            "provenance_invalidation_reasons": tuple(
                provenance.get("provenance_invalidation_reasons", []) or []
            ),
            "execution_id": str(provenance.get("execution_id") or ""),
        }

    def _snapshot_external_state(self) -> Dict[str, object]:
        """Return causal-input watermarks and mutable environment-model state."""
        compact_state_enabled = str(
            os.environ.get("LSGEMU_SNAPSHOT_STATE_COMPRESSION", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        shared_causal_context = False
        try:
            shared_causal_context = (
                getattr(self.mmio_handler, "causal_context", None)
                is self.causal_context
            )
        except Exception:
            shared_causal_context = False
        try:
            if compact_state_enabled:
                try:
                    mmio_runtime = self.mmio_handler.snapshot_runtime_state(
                        copy_values=False,
                        include_causal_context=not shared_causal_context,
                    )
                except TypeError:
                    try:
                        # Keep compatibility with handlers that adopted the
                        # view API but not the optional context argument.
                        mmio_runtime = self.mmio_handler.snapshot_runtime_state(
                            copy_values=False
                        )
                    except TypeError:
                        # Keep compatibility with external/test handlers that
                        # still expose the historical no-argument method.
                        mmio_runtime = self.mmio_handler.snapshot_runtime_state()
                if shared_causal_context and isinstance(mmio_runtime, dict):
                    # The owning emulator stores this exact context below.
                    # Remove only the duplicate field; all independent MMIO
                    # state remains part of the retained replay contract.
                    mmio_runtime = dict(mmio_runtime)
                    mmio_runtime.pop("causal_context", None)
            else:
                mmio_runtime = self.mmio_handler.snapshot_runtime_state()
        except Exception:
            mmio_runtime = {}
        try:
            causal_runtime = self.causal_context.snapshot_runtime_state(
                copy_values=False if compact_state_enabled else True
            )
        except TypeError:
            # Compatibility with a specialized context implementation that
            # still exposes only the historical detached-snapshot signature.
            causal_runtime = self.causal_context.snapshot_runtime_state()
        # P0-D（cycle3 k.5 C3）：按逻辑态写 identity 四元组 (era, bb, c2, c3)。
        # era=恢复代数；bb=本次捕获地址（经管理器的 pending 捕获上下文，
        # 零参 provider 协议下唯一的传递通道）；c2=本实例单调捕获序号；
        # c3=``(era, bb)`` 限定键下的捕获计数（= 本 era 内该 bb 的本地出现
        # 数）。``root_occurrence_local`` = 本 era 被恢复根快照的本地出现数
        # （恢复时从根自身 identity 的 c3 继承）。键追加不改老读者（R4
        # 全量 grep 零消费者），落盘通道 snapshot_memory.py 透传不动。
        capture_context = (
            getattr(
                getattr(self, "branch_snapshot_manager", None),
                "_pending_capture_context",
                None,
            )
            or {}
        )
        try:
            capture_bb = int(capture_context.get("address", 0) or 0) & 0xFFFFFFFF
        except Exception:
            capture_bb = 0
        era = int(getattr(self, "_snapshot_identity_era", 0))
        self._snapshot_identity_seq = seq = (
            int(getattr(self, "_snapshot_identity_seq", 0)) + 1
        )
        occurrences = getattr(self, "_snapshot_identity_occurrences", None)
        if not isinstance(occurrences, dict):
            occurrences = {}
            self._snapshot_identity_occurrences = occurrences
        occurrence_key = (era, capture_bb)
        local_occurrence = int(occurrences.get(occurrence_key, 0)) + 1
        occurrences[occurrence_key] = local_occurrence
        external_model_state = {
            "snapshot_identity": (era, capture_bb, seq, local_occurrence),
            "root_occurrence_local": getattr(
                self, "_root_occurrence_local", None
            ),
            "mmio_handler": mmio_runtime,
            "stream_input_state": (
                getattr(self, "stream_input_state", {}) or {}
                if compact_state_enabled
                else dict(getattr(self, "stream_input_state", {}) or {})
            ),
            "cortex_m_system_registers": (
                getattr(self, "cortex_m_system_registers", {}) or {}
                if compact_state_enabled
                else dict(getattr(self, "cortex_m_system_registers", {}) or {})
            ),
            "skip_function_state": (
                getattr(self, "skip_function_state", {}) or {}
                if compact_state_enabled
                else copy.deepcopy(getattr(self, "skip_function_state", {}) or {})
            ),
            "model_heap_next": int(getattr(self, "model_heap_next", 0) or 0),
            "model_heap_allocations": (
                getattr(self, "model_heap_allocations", {}) or {}
                if compact_state_enabled
                else dict(getattr(self, "model_heap_allocations", {}) or {})
            ),
            "external_memory_input_addresses": (
                getattr(self, "external_memory_input_addresses", set()) or set()
                if compact_state_enabled
                else set(
                    getattr(self, "external_memory_input_addresses", set()) or set()
                )
            ),
            "time_soft_tick": int(getattr(self.time_handler, "soft_tick", 0) or 0),
            "time_call_count": int(getattr(self.time_handler, "call_count", 0) or 0),
            "causal_context": causal_runtime,
        }
        return {
            "input_event_index": int(getattr(self, "causal_input_event_count", 0) or 0),
            "input_occurrence_counts": (
                getattr(self, "input_occurrence_counts", {}) or {}
                if compact_state_enabled
                else dict(getattr(self, "input_occurrence_counts", {}) or {})
            ),
            "external_model_state": external_model_state,
            "execution_provenance": _safe_execution_provenance(self),
        }

    def restore_snapshot_external_state(self, snapshot: object) -> bool:
        """Restore the environment state carried by a branch/input snapshot.

        The boolean result is part of the replay contract.  Partial external
        state restoration can otherwise make a subsequent Unicorn execution
        look like a valid suffix while it is actually based on a mixed state.
        """
        restore_errors: List[str] = []
        # P0-D（cycle3 k.5 C3）：恢复即开新 era（恢复代数单调 +1）；本 era
        # 的 root_occurrence_local 先清空，若被恢复快照带 identity 则从其
        # c3（该快照捕获时的本地出现数）继承——join 主键的 root 面。
        self._snapshot_identity_era = (
            int(getattr(self, "_snapshot_identity_era", 0)) + 1
        )
        self._root_occurrence_local = None
        try:
            self._set_active_prefix_provenance(snapshot)
        except Exception as exc:
            restore_errors.append(f"prefix_provenance_restore_failed:{type(exc).__name__}")
        external_model_state = _snapshot_field(snapshot, "external_model_state", None)
        state_blob = (
            external_model_state
            if isinstance(external_model_state, SnapshotStateBlob)
            else None
        )
        retain_decoded_state = str(
            os.environ.get("LSGEMU_SNAPSHOT_STATE_CACHE_DECODED", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            # Avoid invoking Mapping.__len__ through ``or {}`` before the
            # explicit conversion; one materialization is enough for a
            # restore, and it can be released immediately afterward.
            state = (
                dict(state_blob)
                if state_blob is not None
                else dict(external_model_state or {})
            )
            if state:
                try:
                    restore_result = self.mmio_handler.restore_runtime_state(
                        state.get("mmio_handler")
                    )
                    if restore_result is False:
                        restore_errors.append("mmio_state_restore_returned_false")
                except Exception as exc:
                    restore_errors.append(f"mmio_state_restore_failed:{type(exc).__name__}")
                self.stream_input_state = {
                    str(key): int(value)
                    for key, value in dict(state.get("stream_input_state", {}) or {}).items()
                }
                self.cortex_m_system_registers = {
                    str(key): int(value) & 0xFFFFFFFF
                    for key, value in dict(
                        state.get("cortex_m_system_registers", {}) or {}
                    ).items()
                }
                self.skip_function_state = copy.deepcopy(
                    state.get("skip_function_state", {}) or {}
                )
                self.model_heap_next = int(
                    state.get("model_heap_next", getattr(self, "model_heap_next", 0)) or 0
                )
                self.model_heap_allocations = {
                    int(address): int(size)
                    for address, size in dict(
                        state.get("model_heap_allocations", {}) or {}
                    ).items()
                }
                if "external_memory_input_addresses" in state:
                    self.external_memory_input_addresses = {
                        int(address) & 0xFFFFFFFF
                        for address in set(
                            state.get("external_memory_input_addresses", set()) or set()
                        )
                    }
                self.time_handler.soft_tick = max(
                    0,
                    int(state.get("time_soft_tick", 0) or 0),
                )
                self.time_handler.call_count = max(
                    0,
                    int(state.get("time_call_count", 0) or 0),
                )
                if "causal_context" in state:
                    self.causal_context.restore_runtime_state(
                        state.get("causal_context")
                    )
                restored_identity = state.get("snapshot_identity")
                if (
                    isinstance(restored_identity, (tuple, list))
                    and len(restored_identity) == 4
                ):
                    try:
                        self._root_occurrence_local = int(restored_identity[3])
                    except (TypeError, ValueError):
                        self._root_occurrence_local = None
        except Exception as exc:
            restore_errors.append(
                f"external_state_restore_failed:{type(exc).__name__}"
            )
        finally:
            if state_blob is not None and not retain_decoded_state:
                state_blob.release_materialized()

        if restore_errors:
            current = dict(getattr(self, "_active_prefix_provenance", {}) or {})
            reasons = list(current.get("reasons", ()) or ())
            reasons.extend(restore_errors)
            current["reasons"] = tuple(dict.fromkeys(str(item) for item in reasons if str(item)))
            current["status"] = "diagnostic"
            current["telemetry_complete"] = False
            self._active_prefix_provenance = current

        self.last_snapshot_restore_errors = list(
            dict.fromkeys(str(item) for item in restore_errors if str(item))
        )
        self.last_snapshot_restore = {
            "success": not bool(self.last_snapshot_restore_errors),
            "errors": list(self.last_snapshot_restore_errors),
            "snapshot_address": int(
                _snapshot_field(snapshot, "address", 0) or 0
            )
            & 0xFFFFFFFF,
        }

        self.causal_input_event_count = max(
            0,
            int(_snapshot_field(snapshot, "input_event_index", 0) or 0),
        )
        self.input_occurrence_counts = {
            (int(key[0]), int(key[1])): int(value)
            for key, value in dict(
                _snapshot_field(snapshot, "input_occurrence_counts", {}) or {}
            ).items()
            if isinstance(key, tuple) and len(key) == 2
        }
        self.memory_read_occurrence_counts = Counter(
            self.input_occurrence_counts
        )
        self.causal_input_events = [
            event
            for event in list(getattr(self, "causal_input_events", []) or [])
            if isinstance(event, Mapping)
            and int(event.get("event_id", 0) or 0) <= self.causal_input_event_count
        ]
        return not bool(self.last_snapshot_restore_errors)

    def _begin_causal_input_read(
        self,
        *,
        kind: str,
        pc: int,
        address: int,
        size: int,
    ) -> Dict[str, object]:
        """Capture a bounded pre-read checkpoint and allocate an input occurrence."""
        pc = int(pc) & 0xFFFFFFFF
        address = int(address) & 0xFFFFFFFF
        site = (pc, address)
        occurrence = int(self.input_occurrence_counts.get(site, 0) or 0) + 1
        event_id = int(self.causal_input_event_count or 0) + 1
        trace_event_id = int(
            getattr(self, "causal_input_trace_sequence", 0) or 0
        ) + 1
        self.causal_input_trace_sequence = trace_event_id
        branch_sequence_start = self._branch_occurrence_event_cursor()
        snapshot = None
        snapshot_site = (str(kind or "input"), pc, address)
        site_snapshot_count = int(
            self.causal_input_snapshot_site_counts[snapshot_site]
        )
        sparse_sequence_checkpoint = bool(
            self.causal_input_sequence_checkpoints
            and occurrence > self.max_causal_input_snapshots_per_site
            and (occurrence & (occurrence - 1)) == 0
        )
        should_snapshot = bool(
            self.enable_branch_snapshot
            and self.max_causal_input_snapshots > 0
            and len(self.causal_input_snapshots) < self.max_causal_input_snapshots
            and (
                site_snapshot_count < self.max_causal_input_snapshots_per_site
                or sparse_sequence_checkpoint
            )
        )
        if should_snapshot:
            bb_addr = int(self.instruction_to_bb.get(pc, pc)) & ~1
            try:
                snapshot = self.branch_snapshot_manager.save_snapshot(
                    self.uc,
                    bb_addr,
                    pc,
                    pc,
                    f"INPUT:{str(kind or 'input').upper()}",
                    original_taken=True,
                    depth=self._bb_history_depth(),
                    mmio_state=self._current_mmio_state(),
                    alternatives=[pc],
                    original_index=0,
                    occurrence_index=occurrence,
                    update_current=False,
                )
                self.causal_input_snapshot_site_counts[snapshot_site] += 1
                self.causal_input_snapshots.append({
                    "event_id": event_id,
                    "trace_event_id": trace_event_id,
                    "kind": str(kind or "input"),
                    "pc": pc,
                    "address": address,
                    "size": max(1, int(size or 1)),
                    "occurrence": occurrence,
                    "checkpoint_kind": (
                        "sparse_sequence"
                        if sparse_sequence_checkpoint
                        else "initial_site"
                    ),
                    "snapshot": snapshot,
                })
                self._record_causal_input_retention("snapshots_retained")
            except Exception as exc:
                self._record_causal_input_retention("snapshot_capture_failures")
                logger.debug(
                    "保存输入前因果快照失败 PC=0x%08x Addr=0x%08x: %s",
                    pc,
                    address,
                    exc,
                )
        elif self.enable_branch_snapshot and self.max_causal_input_snapshots > 0:
            if len(self.causal_input_snapshots) >= self.max_causal_input_snapshots:
                self._record_causal_input_retention("snapshot_total_limit_skips")
            elif (
                site_snapshot_count >= self.max_causal_input_snapshots_per_site
                and not sparse_sequence_checkpoint
            ):
                self._record_causal_input_retention("snapshot_site_limit_skips")

        self.causal_input_event_count = event_id
        self.input_occurrence_counts[site] = occurrence
        return {
            "event_id": event_id,
            "trace_event_id": trace_event_id,
            "kind": str(kind or "input"),
            "pc": pc,
            "address": address,
            "size": max(1, int(size or 1)),
            "occurrence": occurrence,
            "branch_event_sequence_start": branch_sequence_start,
            "snapshot_capture_order": (
                int(getattr(snapshot, "capture_order", 0) or 0)
                if snapshot is not None
                else None
            ),
        }

    def _finish_causal_input_read(
        self,
        token: Optional[Dict[str, object]],
        *,
        value: int,
        source: str,
    ) -> None:
        if not token:
            return
        event = dict(token)
        size = max(1, int(event.get("size", 1) or 1))
        event["value"] = int(value) & ((1 << (size * 8)) - 1)
        event["source"] = str(source or "environment")
        retained_branch_events = len(
            self.branch_snapshot_manager.get_ordered_occurrence_events()
        )
        branch_sequence_end = self._branch_occurrence_event_cursor()
        # Keep the historical field name, but make it a true monotonic index.
        # The retained count is reported separately for compatibility audits.
        event["branch_event_index"] = branch_sequence_end
        event["branch_event_sequence_end"] = branch_sequence_end
        event["retained_branch_event_count"] = retained_branch_events
        event["branch_event_stream_truncated"] = bool(
            branch_sequence_end > retained_branch_events
        )
        if len(self.causal_input_events) < self.max_causal_input_events:
            self.causal_input_events.append(event)
            self._record_causal_input_retention("events_retained")
        else:
            self._record_causal_input_retention("event_limit_skips")
        causal_context = getattr(self, "causal_context", None)
        if causal_context is not None:
            causal_context.record_external_input(
                kind=str(event.get("kind") or "external"),
                pc=int(event.get("pc", 0) or 0),
                address=int(event.get("address", 0) or 0),
                occurrence=max(1, int(event.get("occurrence", 1) or 1)),
                trace_event_id=max(0, int(event.get("trace_event_id", 0) or 0)),
                value=int(event.get("value", 0) or 0),
                size=size,
            )
        self._notify_external_input_observers(event)

    def _branch_occurrence_event_cursor(self) -> int:
        """Return a capacity-independent branch occurrence cursor."""
        manager = getattr(self, "branch_snapshot_manager", None)
        getter = getattr(manager, "get_occurrence_event_cursor", None)
        if callable(getter):
            try:
                return max(0, int(getter()))
            except Exception:
                pass
        retained_getter = getattr(manager, "get_ordered_occurrence_events", None)
        if callable(retained_getter):
            try:
                return len(retained_getter())
            except Exception:
                pass
        return 0

    def _record_causal_input_retention(self, name: str, count: int = 1) -> None:
        """Update retention telemetry for normal and legacy ``__new__`` instances."""
        stats = getattr(self, "causal_input_retention_stats", None)
        if not isinstance(stats, Counter):
            stats = Counter(stats or {})
            self.causal_input_retention_stats = stats
        stats[str(name)] += max(0, int(count))

    def add_external_input_observer(
        self,
        observer: Callable[[Dict[str, object]], None],
    ) -> bool:
        """Attach a transient listener for concrete external input events."""
        if not callable(observer):
            return False
        observers = list(getattr(self, "external_input_observers", []) or [])
        if any(existing is observer for existing in observers):
            return False
        observers.append(observer)
        self.external_input_observers = observers
        stats = getattr(self, "external_input_observer_stats", None)
        if not isinstance(stats, dict):
            stats = {"registered": 0, "notifications": 0, "failures": 0}
            self.external_input_observer_stats = stats
        stats["registered"] = len(observers)
        return True

    def remove_external_input_observer(
        self,
        observer: Callable[[Dict[str, object]], None],
    ) -> bool:
        observers = list(getattr(self, "external_input_observers", []) or [])
        retained = [existing for existing in observers if existing is not observer]
        if len(retained) == len(observers):
            return False
        self.external_input_observers = retained
        stats = getattr(self, "external_input_observer_stats", None)
        if isinstance(stats, dict):
            stats["registered"] = len(retained)
        return True

    def _notify_external_input_observers(self, event: Dict[str, object]) -> None:
        observers = list(getattr(self, "external_input_observers", []) or [])
        if not observers:
            return
        stats = getattr(self, "external_input_observer_stats", None)
        if not isinstance(stats, dict):
            stats = {"registered": len(observers), "notifications": 0, "failures": 0}
            self.external_input_observer_stats = stats
        for observer in observers:
            try:
                observer(dict(event))
                stats["notifications"] = int(stats.get("notifications", 0) or 0) + 1
            except Exception as exc:
                stats["failures"] = int(stats.get("failures", 0) or 0) + 1
                logger.debug("外部输入观察器回调失败: %s", exc)

    def _current_mmio_state(self) -> Dict[int, int]:
        """Capture concrete MMIO register values at the current dynamic state."""
        state: Dict[int, int] = {}
        for mmio_addr, mmio_state in getattr(self.mmio_handler, "mmio_states", {}).items():
            try:
                state[int(mmio_addr)] = int(getattr(mmio_state, "current_value", 0)) & 0xFFFFFFFF
            except Exception:
                continue
        return state

    def _get_registers(self) -> Dict[str, int]:
        """获取寄存器状态"""
        registers = {}
        try:
            for i in range(13):
                reg_id = UC_ARM_REG_R0 + i
                registers[f"r{i}"] = self.uc.reg_read(reg_id)
            registers['sp'] = self.uc.reg_read(UC_ARM_REG_SP)
            registers['lr'] = self.uc.reg_read(UC_ARM_REG_LR)
            registers['pc'] = self.uc.reg_read(UC_ARM_REG_PC)
        except Exception as e:
            logger.debug(f"读取寄存器失败: {e}")
        return registers

    def _create_snapshot(self, bb_address: int):
        """创建快照"""
        bb_instructions = self.static_bbs.get(bb_address, [])

        mmio_values = {}
        for mmio_addr, state in self.mmio_handler.mmio_states.items():
            mmio_values[mmio_addr] = state.current_value

        if self.snapshot_manager.should_create_snapshot(bb_address):
            try:
                self.snapshot_manager.create_snapshot(
                    uc=self.uc,
                    bb_address=bb_address,
                    mmio_values=mmio_values,
                    mmio_history=self.mmio_access_history.copy(),
                    execution_path=self.bb_history.copy(),
                    loop_counters=self.mmio_handler.loop_counters.copy(),
                    instruction_count=self.instruction_count,
                    bb_instructions=bb_instructions,
                    dirty_pages=set(getattr(self, "runtime_written_pages", set())),
                )
                self.snapshot_capture_stats["unique_bb_succeeded"] += 1
            except SnapshotCaptureError as exc:
                self.snapshot_manager.seen_bbs.discard(bb_address)
                self.snapshot_capture_stats["unique_bb_failed"] += 1
                logger.debug("唯一BB快照捕获失败 @ 0x%08x: %s", bb_address, exc)

    def _handle_intervention(self, loop_head: int):
        """
        处理干预

        根据循环类型采取不同策略
        """
        # 获取干预策略
        strategy = self.loop_classifier.get_intervention_strategy(loop_head)
        action = strategy.get("action")

        symbol = self.symbols_by_addr.get(int(loop_head) & ~1, "")
        if symbol and self._default_skip_return_for_symbol(symbol) is not None:
            if self._apply_symbol_summary_return(loop_head, symbol):
                self.loop_intervention_failures.pop(loop_head, None)
                return

        logger.info(f"\n{'='*80}")
        logger.info(f"干预策略")
        logger.info(f"{'='*80}")
        logger.info(f"循环头: 0x{loop_head:08x}")
        logger.info(f"动作: {action}")
        logger.info(f"理由: {strategy.get('reason')}")

        self.intervention_count += 1
        # r9 裁定：入口先按 loop_intervention（diagnostic）记账；家族 handler
        # 成功后改记为具体家族（快进/轮询MMIO/等待处理），未产生效果的干预
        # 保持诊断口径不变——只拆标签，不放宽任何禁令。
        self._count_intervention_event("loop_intervention")

        # 获取循环类型
        loop_type = self.loop_classifier.classify_loop(loop_head)
        self.intervention_by_type[loop_type] += 1

        before_constraint_count = self._runtime_constraint_count()
        if self._handle_finite_memory_initialization_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_fast_forward_emulation")
            return
        if self._handle_finite_memory_copy_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_fast_forward_emulation")
            return
        if self._handle_postindexed_store_count_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_fast_forward_emulation")
            return
        if self._handle_postindexed_store_inc_cmp_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_fast_forward_emulation")
            return
        if self._handle_finite_byte_copy_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_fast_forward_emulation")
            return
        if self._handle_predicated_status_wait_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_wait_handled")
            return
        if self._handle_status_register_exit_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_wait_handled")
            return

        is_simple_wait_loop = self.wait_loop_handler.is_wait_loop(loop_head)
        if (self.local_self_loop_before_wait or not is_simple_wait_loop) and self._handle_local_self_loop_constraint(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            return
        if self._handle_memory_mapped_wait_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_wait_handled")
            return

        if is_simple_wait_loop:
            if self.wait_loop_handler.handle_wait_loop(self.uc, loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_wait_handled")
                logger.info("✓ 简单等待循环已处理，继续执行")
                return

        if not self.local_self_loop_before_wait and self._handle_local_self_loop_constraint(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            return
        if self._handle_memory_mapped_wait_loop(loop_head):
            self.loop_intervention_failures.pop(loop_head, None)
            self._reclassify_last_intervention_event("loop_wait_handled")
            return

        intervention_applied = False
        if action == "adjust_mmio":
            # 轮询循环：调整MMIO值
            intervention_applied = self._handle_polling_loop(loop_head, strategy)
            if intervention_applied:
                # 写 MMIO 让轮询环退出 = 外设状态变化（外部输入，用户原则 1）。
                self._reclassify_last_intervention_event("loop_mmio_adjust")

        elif action == "llm_inference":
            # 使用LLM推断（新策略）
            self._handle_llm_inference(loop_head, strategy)

        elif action == "rollback":
            # 死循环：回退并使用LLM分析（旧策略，保留兼容）
            self._handle_deadlock_rollback(loop_head, strategy)

        elif action == "check_config":
            # 初始化循环：检查配置
            if self._handle_finite_memory_initialization_loop(loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_fast_forward_emulation")
                return
            if self._handle_finite_memory_copy_loop(loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_fast_forward_emulation")
                return
            if self._handle_postindexed_store_count_loop(loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_fast_forward_emulation")
                return
            if self._handle_postindexed_store_inc_cmp_loop(loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_fast_forward_emulation")
                return
            if self._handle_finite_byte_copy_loop(loop_head):
                self.loop_intervention_failures.pop(loop_head, None)
                self._reclassify_last_intervention_event("loop_fast_forward_emulation")
                return
            if self._force_byte_copy_loop_fallthrough(loop_head):
                # 强制 fallthrough 属强制转分支族，保持 loop_intervention 诊断口径。
                self.loop_intervention_failures.pop(loop_head, None)
                return
            logger.warning("初始化循环执行时间过长，可能配置值不正确")
            logger.warning("建议检查: 数据段大小、BSS段大小等配置")
            self.uc.emu_stop()

        elif action == "skip":
            # 未知循环：强制停止
            logger.warning(f"未知循环类型，强制停止执行")
            self.uc.emu_stop()

        else:
            # 默认：强制停止
            logger.warning(f"未知动作: {action}，强制停止")
            self.uc.emu_stop()

        after_constraint_count = self._runtime_constraint_count()
        if after_constraint_count > before_constraint_count or intervention_applied:
            self.loop_intervention_failures.pop(loop_head, None)
        elif action in {"adjust_mmio", "llm_inference", "rollback"}:
            self._record_unresolved_loop_intervention(loop_head)

    def _handle_finite_memory_initialization_loop(self, loop_head: int) -> bool:
        """Fast-forward bounded memset/BSS clear loops instead of stopping.

        Raw ARM bootloaders often start with a large finite RAM clear:

            stm   r0!, {r1-r12}
            cmp   r0, lr
            blt   loop

        Executing every iteration is valid but wastes the whole instruction
        budget before any protocol/update code is reached.  This handler only
        applies when the current loop has a concrete cursor register, concrete
        upper bound, and a memory write with write-back/post-index progress.
        It preserves entry-derived execution by materializing the skipped
        writes and then placing PC at the branch fallthrough.
        """
        try:
            loop_body = self.loop_classifier._get_loop_body_bbs(loop_head)
        except Exception:
            loop_body = [loop_head]
        instructions: List[Dict[str, object]] = []
        seen_addresses: Set[int] = set()
        for bb in loop_body or [loop_head]:
            for insn in self.static_bbs.get(int(bb), []) or []:
                address = int(insn.get("address", 0) or 0)
                if address in seen_addresses:
                    continue
                seen_addresses.add(address)
                instructions.append(dict(insn))
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if not instructions or len(instructions) > 32:
            return False

        branch_insn = instructions[-1]
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if int(target_bb or -1) != int(loop_head):
            return False
        if branch_mnemonic not in {"BNE", "BLT", "BLO", "BCC", "BLS", "BLE"}:
            return False

        compare_insn = None
        for candidate in reversed(instructions[:-1]):
            if self._normalize_mnemonic(candidate.get("mnemonic", "")) in {"CMP", "CMN"}:
                compare_insn = candidate
                break
        if compare_insn is None:
            return False
        compare_operands = self._split_operands(compare_insn.get("operands", ""))
        if len(compare_operands) < 2:
            return False
        cursor_reg = compare_operands[0].strip().lower()
        if self._arm_reg_id(cursor_reg) is None:
            return False
        limit_value = self._read_operand_value(self.uc, compare_operands[1])
        cursor_value = int(self.uc.reg_read(self._arm_reg_id(cursor_reg))) & 0xFFFFFFFF
        if limit_value <= cursor_value:
            return False

        writer = None
        for candidate in instructions[:-1]:
            mnemonic = self._normalize_mnemonic(candidate.get("mnemonic", ""))
            operands = str(candidate.get("operands", "") or "").lower()
            if mnemonic.startswith("STM") and re.search(rf"\b{re.escape(cursor_reg)}\s*!", operands):
                writer = candidate
                break
            if mnemonic.startswith("STR") and re.search(rf"\[\s*{re.escape(cursor_reg)}\s*\]\s*,", operands):
                writer = candidate
                break
        if writer is None:
            return False

        pattern = self._memory_init_write_pattern(writer)
        if not pattern:
            return False
        stride = len(pattern)
        remaining = int(limit_value - cursor_value)
        if remaining <= 0:
            return False
        max_fast_forward = max(0x1000, int(os.environ.get("LSGEMU_INIT_LOOP_FAST_FORWARD_MAX_BYTES", "33554432")))
        if remaining > max_fast_forward:
            logger.warning(
                "初始化循环fast-forward拒绝: loop=0x%08x remaining=0x%x max=0x%x",
                loop_head,
                remaining,
                max_fast_forward,
            )
            return False

        write_size = remaining - (remaining % stride)
        if write_size <= 0:
            write_size = remaining
        try:
            self._write_repeated_pattern(cursor_value, pattern, write_size)
            cursor_id = self._arm_reg_id(cursor_reg)
            if cursor_id is not None:
                self.uc.reg_write(cursor_id, int(limit_value) & 0xFFFFFFFF)
            fallthrough = int(branch_insn.get("address", loop_head) or loop_head) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            self.stop_requested_reason = None
            logger.info(
                "✓ 初始化内存循环fast-forward: loop=0x%08x %s=0x%08x->0x%08x bytes=0x%x fallthrough=0x%08x",
                loop_head,
                cursor_reg,
                cursor_value,
                limit_value,
                write_size,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "finite_memory_initialization_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug("初始化循环fast-forward失败 @ 0x%08x: %s", loop_head, exc)
            return False

    def _memory_init_write_pattern(self, insn: Dict[str, object]) -> Optional[bytes]:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        operands = str(insn.get("operands", "") or "")
        if mnemonic.startswith("STM"):
            register_text = operands[operands.find("{") + 1 : operands.find("}")] if "{" in operands and "}" in operands else ""
            regs = [part.strip().lower() for part in register_text.split(",") if part.strip()]
            if not regs:
                return None
            chunks = []
            for reg in regs:
                reg_id = self._arm_reg_id(reg)
                if reg_id is None:
                    return None
                value = int(self.uc.reg_read(reg_id)) & 0xFFFFFFFF
                chunks.append(value.to_bytes(4, "little"))
            return b"".join(chunks)

        if mnemonic.startswith("STR"):
            parts = self._split_operands(operands)
            if not parts:
                return None
            value = self._read_operand_value(self.uc, parts[0])
            width = 4
            if mnemonic in {"STRB"}:
                width = 1
            elif mnemonic in {"STRH"}:
                width = 2
            return int(value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
        return None

    def _write_repeated_pattern(self, address: int, pattern: bytes, size: int) -> None:
        if not pattern or size <= 0:
            return
        max_chunk = 0x10000
        offset = 0
        while offset < size:
            chunk_size = min(max_chunk, size - offset)
            current = int(address) + offset
            # Preflight one bounded write transaction.  A missing page cannot
            # leave a half-written chunk before the callback is retried.
            if not self._ensure_memory_ranges_mapped([(current, chunk_size)]):
                raise RuntimeError(f"cannot map 0x{current:08x}+0x{chunk_size:x}")
            repeats = (chunk_size + len(pattern) - 1) // len(pattern)
            payload = (pattern * repeats)[:chunk_size]
            self.uc.mem_write(current, payload)
            offset += chunk_size

    def _handle_finite_memory_copy_loop(self, loop_head: int) -> bool:
        """Fast-forward bounded block-copy loops in raw ARM startup code.

        The Honeywell scanner images use a hand-written memcpy-like loop during
        relocation:

            ldm   ip!, {r0-r10}
            stm   fp!, {r0-r10}
            sub   lr, lr, #0x2c
            cmp   lr, #0x2c
            bge   loop

        This handler copies the skipped bytes in Unicorn memory, updates the
        source/destination/count registers, and resumes at the branch
        fallthrough. It only applies to concrete post-increment LDM/STM loops
        with a matching decrement and compare threshold.
        """
        try:
            loop_body = self.loop_classifier._get_loop_body_bbs(loop_head)
        except Exception:
            loop_body = [loop_head]
        instructions: List[Dict[str, object]] = []
        seen_addresses: Set[int] = set()
        for bb in loop_body or [loop_head]:
            for insn in self.static_bbs.get(int(bb), []) or []:
                address = int(insn.get("address", 0) or 0)
                if address in seen_addresses:
                    continue
                seen_addresses.add(address)
                instructions.append(dict(insn))
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 5 or len(instructions) > 24:
            return False

        branch_insn = instructions[-1]
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        if branch_mnemonic not in {"BGE", "BHS", "BCS"}:
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if int(target_bb or -1) != int(loop_head):
            return False

        ldm_insn = None
        stm_insn = None
        sub_insn = None
        cmp_insn = None
        for insn in instructions[:-1]:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if ldm_insn is None and mnemonic.startswith("LDM"):
                ldm_insn = insn
            elif stm_insn is None and mnemonic.startswith("STM"):
                stm_insn = insn
            elif mnemonic in {"SUB", "SUBS"}:
                sub_insn = insn
            elif mnemonic == "CMP":
                cmp_insn = insn
        if not (ldm_insn and stm_insn and sub_insn and cmp_insn):
            return False

        ldm_base, ldm_regs = self._parse_block_transfer_writeback(ldm_insn)
        stm_base, stm_regs = self._parse_block_transfer_writeback(stm_insn)
        if not ldm_base or not stm_base or not ldm_regs or ldm_regs != stm_regs:
            return False
        chunk_size = 4 * len(ldm_regs)
        if chunk_size <= 0:
            return False

        sub_parts = self._split_operands(sub_insn.get("operands", ""))
        cmp_parts = self._split_operands(cmp_insn.get("operands", ""))
        if len(sub_parts) < 3 or len(cmp_parts) < 2:
            return False
        count_reg = sub_parts[0].strip().lower()
        if sub_parts[1].strip().lower() != count_reg:
            return False
        if cmp_parts[0].strip().lower() != count_reg:
            return False
        decrement = self._parse_immediate_operand(sub_parts[2])
        threshold = self._parse_immediate_operand(cmp_parts[1])
        if decrement is None or threshold is None:
            return False
        decrement = int(decrement)
        threshold = int(threshold)
        if decrement <= 0 or threshold < 0 or decrement != chunk_size:
            return False

        src_reg_id = self._arm_reg_id(ldm_base)
        dst_reg_id = self._arm_reg_id(stm_base)
        count_reg_id = self._arm_reg_id(count_reg)
        if src_reg_id is None or dst_reg_id is None or count_reg_id is None:
            return False
        src = int(self.uc.reg_read(src_reg_id)) & 0xFFFFFFFF
        dst = int(self.uc.reg_read(dst_reg_id)) & 0xFFFFFFFF
        count = int(self.uc.reg_read(count_reg_id)) & 0xFFFFFFFF
        if count < threshold:
            return False
        iterations = ((count - threshold) // decrement) + 1
        copy_size = int(iterations) * chunk_size
        max_fast_forward = max(0x1000, int(os.environ.get("LSGEMU_INIT_LOOP_FAST_FORWARD_MAX_BYTES", "33554432")))
        if copy_size <= 0 or copy_size > max_fast_forward:
            return False

        try:
            if not self._ensure_memory_ranges_mapped(
                [(src, copy_size), (dst, copy_size)]
            ):
                return False
            max_chunk = 0x10000
            offset = 0
            while offset < copy_size:
                chunk = min(max_chunk, copy_size - offset)
                payload = bytes(self.uc.mem_read(src + offset, chunk))
                self.uc.mem_write(dst + offset, payload)
                offset += chunk
            self.uc.reg_write(src_reg_id, (src + copy_size) & 0xFFFFFFFF)
            self.uc.reg_write(dst_reg_id, (dst + copy_size) & 0xFFFFFFFF)
            self.uc.reg_write(count_reg_id, (count - copy_size) & 0xFFFFFFFF)
            fallthrough = int(branch_insn.get("address", loop_head) or loop_head) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            logger.info(
                "✓ 初始化拷贝循环fast-forward: loop=0x%08x src=%s:0x%08x dst=%s:0x%08x count=%s:0x%x bytes=0x%x fallthrough=0x%08x",
                loop_head,
                ldm_base,
                src,
                stm_base,
                dst,
                count_reg,
                count,
                copy_size,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "finite_memory_copy_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug("初始化拷贝循环fast-forward失败 @ 0x%08x: %s", loop_head, exc)
            return False

    def _handle_finite_byte_copy_loop(self, loop_head: int) -> bool:
        """Fast-forward bounded byte-copy loops.

        Many decompressor stubs use very small loops:

            ldrb  tmp, [src], #1
            subs  count, count, #1
            strb  tmp, [dst], #1
            bne   loop

        They are finite and concrete, but executing every byte burns the budget
        before command/update code is reached. This handler materializes the
        remaining copy and resumes at the branch fallthrough.
        """
        try:
            loop_body = self.loop_classifier._get_loop_body_bbs(loop_head)
        except Exception:
            loop_body = [loop_head]
        instructions: List[Dict[str, object]] = []
        seen_addresses: Set[int] = set()
        for bb in loop_body or [loop_head]:
            for insn in self.static_bbs.get(int(bb), []) or []:
                address = int(insn.get("address", 0) or 0)
                if address in seen_addresses:
                    continue
                seen_addresses.add(address)
                instructions.append(dict(insn))
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 4 or len(instructions) > 16:
            instructions = [dict(insn) for insn in self.static_bbs.get(int(loop_head), []) or []]
            instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
            if len(instructions) < 4 or len(instructions) > 16:
                return False

        branch_insn = instructions[-1]
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        if branch_mnemonic != "BNE":
            instructions = [dict(insn) for insn in self.static_bbs.get(int(loop_head), []) or []]
            instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
            if len(instructions) < 4 or len(instructions) > 16:
                return False
            branch_insn = instructions[-1]
            branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
            if branch_mnemonic != "BNE":
                return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if int(target_bb or -1) != int(loop_head):
            instructions = [dict(insn) for insn in self.static_bbs.get(int(loop_head), []) or []]
            instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
            if len(instructions) < 4 or len(instructions) > 16:
                return False
            branch_insn = instructions[-1]
            branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
            branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
            target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
            if branch_mnemonic != "BNE" or int(target_bb or -1) != int(loop_head):
                return False

        load_insn = None
        store_insn = None
        sub_insn = None
        load_info = None
        store_info = None
        for insn in instructions[:-1]:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if load_insn is None and mnemonic == "LDRB":
                parsed = self._parse_post_indexed_byte_transfer(insn)
                if parsed is not None:
                    load_insn = insn
                    load_info = parsed
                    continue
            if store_insn is None and mnemonic == "STRB":
                parsed = self._parse_post_indexed_byte_transfer(insn)
                if parsed is not None:
                    store_insn = insn
                    store_info = parsed
                    continue
            if sub_insn is None and mnemonic in {"SUB", "SUBS"}:
                sub_insn = insn

        if not (load_insn and store_insn and sub_insn and load_info and store_info):
            return False
        load_reg, src_reg, load_stride = load_info
        store_reg, dst_reg, store_stride = store_info
        if load_reg != store_reg or int(load_stride) != 1 or int(store_stride) != 1:
            return False

        sub_parts = self._split_operands(sub_insn.get("operands", ""))
        if len(sub_parts) < 3:
            return False
        count_reg = sub_parts[0].strip().lower()
        if sub_parts[1].strip().lower() != count_reg:
            return False
        decrement = self._parse_immediate_operand(sub_parts[2])
        if decrement != 1:
            return False

        src_reg_id = self._arm_reg_id(src_reg)
        dst_reg_id = self._arm_reg_id(dst_reg)
        count_reg_id = self._arm_reg_id(count_reg)
        tmp_reg_id = self._arm_reg_id(load_reg)
        if src_reg_id is None or dst_reg_id is None or count_reg_id is None:
            return False

        src = int(self.uc.reg_read(src_reg_id)) & 0xFFFFFFFF
        dst = int(self.uc.reg_read(dst_reg_id)) & 0xFFFFFFFF
        count = int(self.uc.reg_read(count_reg_id)) & 0xFFFFFFFF
        if count == 0:
            fallthrough = int(branch_insn.get("address", loop_head) or loop_head) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            logger.info(
                "✓ 零长度字节拷贝循环fast-forward: loop=0x%08x count=%s:0 fallthrough=0x%08x",
                loop_head,
                count_reg,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "finite_byte_copy_loop", loop_head=loop_head
            )
        max_fast_forward = max(
            0x1000,
            int(os.environ.get("LSGEMU_BYTE_COPY_FAST_FORWARD_MAX_BYTES", "16777216")),
        )
        if count > max_fast_forward:
            logger.warning(
                "字节拷贝循环fast-forward拒绝: loop=0x%08x count=0x%x max=0x%x",
                loop_head,
                count,
                max_fast_forward,
            )
            return False

        try:
            if not self._ensure_memory_ranges_mapped([(src, count), (dst, count)]):
                return False
            max_chunk = 0x10000
            offset = 0
            last_byte = None
            while offset < count:
                chunk = min(max_chunk, count - offset)
                try:
                    payload = bytes(self.uc.mem_read(src + offset, chunk))
                except Exception:
                    payload = b"\x00" * chunk
                self.uc.mem_write(dst + offset, payload)
                if payload:
                    last_byte = payload[-1]
                offset += chunk

            self.uc.reg_write(src_reg_id, (src + count) & 0xFFFFFFFF)
            self.uc.reg_write(dst_reg_id, (dst + count) & 0xFFFFFFFF)
            self.uc.reg_write(count_reg_id, 0)
            if tmp_reg_id is not None and last_byte is not None:
                self.uc.reg_write(tmp_reg_id, int(last_byte) & 0xFF)

            fallthrough = int(branch_insn.get("address", loop_head) or loop_head) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            logger.info(
                "✓ 字节拷贝循环fast-forward: loop=0x%08x src=%s:0x%08x dst=%s:0x%08x count=%s:0x%x fallthrough=0x%08x",
                loop_head,
                src_reg,
                src,
                dst_reg,
                dst,
                count_reg,
                count,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "postindexed_store_count_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug("字节拷贝循环fast-forward失败 @ 0x%08x: %s", loop_head, exc)
            return False

    def _handle_postindexed_store_count_loop(self, loop_head: int) -> bool:
        """Fast-forward memset/BSS loops in statically known or dynamic code.

        Matches reached loops of the form:

            str{b,h} value, [cursor], #stride
            subs     count, count, #decrement
            bne      loop

        This pattern appears after Honeywell's LZO stage in the decompressed
        image. It is a concrete finite initialization loop, not an external
        state wait, so the faithful action is to materialize the writes and
        resume at the branch fallthrough.
        """
        owner = self._ensure_dynamic_basic_block(loop_head)
        instructions = [dict(insn) for insn in self.static_bbs.get(int(owner), []) or []]
        if not instructions and int(owner) != int(loop_head):
            instructions = [dict(insn) for insn in self.static_bbs.get(int(loop_head), []) or []]
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 3 or len(instructions) > 8:
            return False

        branch_insn = instructions[-1]
        if self._normalize_mnemonic(branch_insn.get("mnemonic", "")) != "BNE":
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if (int(target_bb or -1) & ~1) != (int(owner) & ~1):
            return False

        store_insn = None
        store_info = None
        for index, insn in enumerate(instructions[:-1]):
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if store_insn is None and mnemonic in {"STR", "STRB", "STRH"}:
                parsed = self._parse_post_indexed_store_transfer(insn)
                if parsed is not None:
                    store_insn = insn
                    store_info = parsed
                    store_index = index
                    break

        if not (store_insn and store_info):
            return False
        sub_insn = None
        for insn in instructions[store_index + 1:-1]:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic not in {"SUB", "SUBS"}:
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if len(parts) < 3:
                continue
            if parts[0].strip().lower() != parts[1].strip().lower():
                continue
            if self._parse_immediate_operand(parts[2]) is None:
                continue
            sub_insn = insn
            break
        if sub_insn is None:
            return False
        value_reg, cursor_reg, stride, width = store_info
        sub_parts = self._split_operands(sub_insn.get("operands", ""))
        if len(sub_parts) < 3:
            return False
        count_reg = sub_parts[0].strip().lower()
        if sub_parts[1].strip().lower() != count_reg:
            return False
        decrement = self._parse_immediate_operand(sub_parts[2])
        if decrement is None or decrement <= 0:
            return False
        if stride <= 0 or width <= 0:
            return False
        if decrement < width:
            return False

        value_reg_id = self._arm_reg_id(value_reg)
        cursor_reg_id = self._arm_reg_id(cursor_reg)
        count_reg_id = self._arm_reg_id(count_reg)
        if value_reg_id is None or cursor_reg_id is None or count_reg_id is None:
            return False

        cursor = int(self.uc.reg_read(cursor_reg_id)) & 0xFFFFFFFF
        count = int(self.uc.reg_read(count_reg_id)) & 0xFFFFFFFF
        if count == 0:
            return False
        if count >= 0x80000000:
            return False

        iterations = (count + int(decrement) - 1) // int(decrement)
        write_size = iterations * int(width)
        cursor_advance = iterations * int(stride)
        count_after = max(0, count - iterations * int(decrement))
        try:
            max_fast_forward = max(
                0x1000,
                int(os.environ.get("LSGEMU_STORE_LOOP_FAST_FORWARD_MAX_BYTES", "67108864")),
            )
        except ValueError:
            max_fast_forward = 64 * 1024 * 1024
        if write_size <= 0 or write_size > max_fast_forward:
            logger.warning(
                "store初始化循环fast-forward拒绝: loop=0x%08x bytes=0x%x max=0x%x",
                int(loop_head),
                int(write_size),
                int(max_fast_forward),
            )
            return False

        value = int(self.uc.reg_read(value_reg_id)) & 0xFFFFFFFF
        pattern = (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
        try:
            self._write_strided_pattern(cursor, pattern, int(stride), int(iterations))
            self.uc.reg_write(cursor_reg_id, (cursor + cursor_advance) & 0xFFFFFFFF)
            self.uc.reg_write(count_reg_id, count_after & 0xFFFFFFFF)
            fallthrough = int(branch_insn.get("address", owner) or owner) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            logger.info(
                "✓ store初始化循环fast-forward: loop=0x%08x %s=0x%08x %s=0x%08x count=%s:0x%x bytes=0x%x fallthrough=0x%08x",
                int(owner) & ~1,
                cursor_reg,
                cursor,
                value_reg,
                value,
                count_reg,
                count,
                write_size,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "postindexed_store_inc_cmp_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug("store初始化循环fast-forward失败 @ 0x%08x: %s", int(loop_head), exc)
            return False

    def _handle_postindexed_store_inc_cmp_loop(self, loop_head: int) -> bool:
        """Fast-forward store loops using ADD counter + CMP limit + BNE.

        Example:

            str  r2, [r1], #4
            add  r0, r0, #1
            cmp  r3, r0
            bne  loop
        """
        owner = self._ensure_dynamic_basic_block(loop_head)
        instructions = [dict(insn) for insn in self.static_bbs.get(int(owner), []) or []]
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 4 or len(instructions) > 96:
            return False

        branch_insn = instructions[-1]
        if self._normalize_mnemonic(branch_insn.get("mnemonic", "")) != "BNE":
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if (int(target_bb or -1) & ~1) != (int(owner) & ~1):
            return False

        store_insn = None
        store_info = None
        store_index = -1
        for index, insn in enumerate(instructions[:-1]):
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic not in {"STR", "STRB", "STRH"}:
                continue
            parsed = self._parse_post_indexed_store_transfer(insn)
            if parsed is None:
                continue
            store_insn = insn
            store_info = parsed
            store_index = index
            break
        if store_insn is None or store_info is None:
            return False

        add_insn = None
        cmp_insn = None
        counter_reg = None
        increment = None
        for insn in instructions[store_index + 1:-1]:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if add_insn is None and mnemonic in {"ADD", "ADDS"}:
                parts = self._split_operands(insn.get("operands", ""))
                if len(parts) >= 3 and parts[0].strip().lower() == parts[1].strip().lower():
                    inc = self._parse_immediate_operand(parts[2])
                    if inc is not None and inc > 0:
                        add_insn = insn
                        counter_reg = parts[0].strip().lower()
                        increment = int(inc)
                        continue
            if add_insn is not None and mnemonic == "CMP":
                parts = self._split_operands(insn.get("operands", ""))
                if len(parts) >= 2 and counter_reg in {parts[0].strip().lower(), parts[1].strip().lower()}:
                    cmp_insn = insn
                    break
        if add_insn is None or cmp_insn is None or counter_reg is None or increment is None:
            return False

        cmp_parts = self._split_operands(cmp_insn.get("operands", ""))
        if len(cmp_parts) < 2:
            return False
        left = cmp_parts[0].strip().lower()
        right = cmp_parts[1].strip().lower()
        if left == counter_reg:
            limit_value = self._read_operand_value(self.uc, right)
        elif right == counter_reg:
            limit_value = self._read_operand_value(self.uc, left)
        else:
            return False

        counter_reg_id = self._arm_reg_id(counter_reg)
        if counter_reg_id is None:
            return False
        counter = int(self.uc.reg_read(counter_reg_id)) & 0xFFFFFFFF
        limit = int(limit_value) & 0xFFFFFFFF
        if limit >= 0x80000000 or counter >= 0x80000000:
            return False
        if counter >= limit:
            return False

        value_reg, cursor_reg, stride, width = store_info
        value_reg_id = self._arm_reg_id(value_reg)
        cursor_reg_id = self._arm_reg_id(cursor_reg)
        if value_reg_id is None or cursor_reg_id is None:
            return False
        iterations = (limit - counter + int(increment) - 1) // int(increment)
        if iterations <= 0:
            return False
        write_size = int(iterations) * int(width)
        try:
            max_fast_forward = max(
                0x1000,
                int(os.environ.get("LSGEMU_STORE_LOOP_FAST_FORWARD_MAX_BYTES", "67108864")),
            )
        except ValueError:
            max_fast_forward = 64 * 1024 * 1024
        if write_size > max_fast_forward:
            return False

        cursor = int(self.uc.reg_read(cursor_reg_id)) & 0xFFFFFFFF
        value = int(self.uc.reg_read(value_reg_id)) & 0xFFFFFFFF
        pattern = (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
        try:
            self._write_strided_pattern(cursor, pattern, int(stride), int(iterations))
            self.uc.reg_write(cursor_reg_id, (cursor + int(iterations) * int(stride)) & 0xFFFFFFFF)
            self.uc.reg_write(counter_reg_id, limit & 0xFFFFFFFF)
            fallthrough = int(branch_insn.get("address", owner) or owner) + int(branch_insn.get("size", 4) or 4)
            if self.execution_thumb:
                fallthrough |= 1
            else:
                fallthrough &= ~1
            self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
            logger.info(
                "✓ store计数循环fast-forward: loop=0x%08x %s=0x%08x %s=0x%08x %s:0x%x->0x%x bytes=0x%x fallthrough=0x%08x",
                int(owner) & ~1,
                cursor_reg,
                cursor,
                value_reg,
                value,
                counter_reg,
                counter,
                limit,
                write_size,
                fallthrough & ~1,
            )
            return self._record_loop_fast_forward(
                "postindexed_store_limit_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug("store计数循环fast-forward失败 @ 0x%08x: %s", int(loop_head), exc)
            return False

    def _handle_pointer_limit_store_loop(self, loop_head: int) -> bool:
        """r35 T2：memset 族（后索引存储 + 指针即计数器 + 无条件回跳）O(1) 物化。

        形态（环头/环体两个 BB）：

            head:  cmp  cursor, limit
                   bne  body          ; 落空后继 = 环的正常出口
            body:  str{b,h} value, [cursor], #stride
                   b    head

        与 ``_handle_postindexed_store_count_loop``（SUBS 计数器）和
        ``_handle_postindexed_store_inc_cmp_loop``（独立 ADD 计数器）不同，本族
        **没有独立计数器寄存器**：指针自身就是计数器，退出条件是 ``cmp`` 的
        相等比较（ARMv7-M ``memset`` 的典型形态，见 0x0813a9e8）。

        force-free 边界（与 r15 D1 确定性快转族同口径，见 ``_flip_fast_forward_eligible``）：
          * 只由 ``flip_fast_forward_active`` 分支调用，即 ``dfs_flip`` 重放路径
            且 ``enable_loop_intervention=False``；触 MMIO 的环已被资格判据挡掉，
            这里再按目标地址范围二次确认；
          * **终止性必须可证**：stride > 0 且 ``(limit - cursor) % stride == 0``。
            相等比较下若不整除，循环永不退出——此时一律拒绝，退回逐条真实执行
            （这就是「不终止的指针环不得物化」的负对照边界）；
          * 只写内存 + 把指针寄存器的值推到 limit，**不写 PC、不选分支方向**：
            环头的 ``cmp/bne`` 随后照常执行，条件成立后自然落空退出。

        家族内开关 ``LSGEMU_POINTER_LIMIT_STORE_FF``（默认 ``1``）可单独关掉本形态，
        供 A/B 对照。家族本身仍受 ``dfs_flip_deterministic_fast_forward`` 门控
        （默认 False，只在 DFS 翻转重放路径打开），故 canonical 主跑路径不受影响。
        """
        if os.environ.get("LSGEMU_POINTER_LIMIT_STORE_FF", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        owner = self._ensure_dynamic_basic_block(loop_head)
        head_instructions = [
            dict(insn) for insn in self.static_bbs.get(int(owner), []) or []
        ]
        head_instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if not (2 <= len(head_instructions) <= 6):
            return False
        branch_insn = head_instructions[-1]
        if self._normalize_mnemonic(branch_insn.get("mnemonic", "")) != "BNE":
            return False
        cmp_insn = None
        for insn in reversed(head_instructions[:-1]):
            if self._normalize_mnemonic(insn.get("mnemonic", "")) == "CMP":
                cmp_insn = insn
                break
        if cmp_insn is None:
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        if branch_target is None:
            return False
        body_bb = int(
            self.instruction_to_bb.get(int(branch_target), int(branch_target)) or 0
        ) & ~1
        owner_bb = int(owner) & ~1
        if body_bb == owner_bb:
            return False
        body = [dict(insn) for insn in self.static_bbs.get(body_bb, []) or []]
        if not (2 <= len(body) <= 4):
            return False
        body.sort(key=lambda item: int(item.get("address", 0) or 0))
        back_insn = body[-1]
        if self._normalize_mnemonic(back_insn.get("mnemonic", "")) not in {"B", "BAL"}:
            return False
        back_target = self._parse_branch_target(back_insn.get("operands", ""))
        if back_target is None:
            return False
        back_bb = int(
            self.instruction_to_bb.get(int(back_target), int(back_target)) or 0
        ) & ~1
        if back_bb != owner_bb:
            return False
        store_info = None
        for insn in body[:-1]:
            if self._normalize_mnemonic(insn.get("mnemonic", "")) in {"STR", "STRB", "STRH"}:
                parsed = self._parse_post_indexed_store_transfer(insn)
                if parsed is not None:
                    store_info = parsed
                    break
        if store_info is None:
            return False
        value_reg, cursor_reg, stride, width = store_info
        if stride <= 0 or width <= 0 or value_reg == cursor_reg:
            return False
        cmp_parts = self._split_operands(cmp_insn.get("operands", ""))
        if len(cmp_parts) < 2:
            return False
        left = cmp_parts[0].strip().lower()
        right = cmp_parts[1].strip().lower()
        if cursor_reg not in {left, right}:
            return False
        limit_text = right if left == cursor_reg else left
        cursor_reg_id = self._arm_reg_id(cursor_reg)
        value_reg_id = self._arm_reg_id(value_reg)
        if cursor_reg_id is None or value_reg_id is None:
            return False
        try:
            limit_value = self._read_operand_value(self.uc, limit_text)
        except Exception:
            return False
        if limit_value is None:
            return False
        cursor = int(self.uc.reg_read(cursor_reg_id)) & 0xFFFFFFFF
        limit = int(limit_value) & 0xFFFFFFFF
        if cursor >= 0x80000000 or limit >= 0x80000000:
            return False
        if limit <= cursor:
            return False
        span = limit - cursor
        if span % int(stride) != 0:
            # 相等比较永不成立 ⇒ 不可证终止，绝不物化。
            self.pointer_limit_store_ff_declines["non_terminating"] += 1
            logger.debug(
                "指针计数 store 循环拒绝（不可证终止）: loop=0x%08x span=0x%x stride=0x%x",
                int(owner) & ~1,
                span,
                int(stride),
            )
            return False
        iterations = span // int(stride)
        if iterations <= 0:
            return False
        write_size = int(iterations) * int(width)
        try:
            max_fast_forward = max(
                0x1000,
                int(os.environ.get("LSGEMU_STORE_LOOP_FAST_FORWARD_MAX_BYTES", "67108864")),
            )
        except ValueError:
            max_fast_forward = 64 * 1024 * 1024
        if write_size > max_fast_forward:
            return False
        last_address = cursor + (iterations - 1) * int(stride)
        if (
            self._is_mmio_address(cursor)
            or self._is_mmio_address(last_address)
            or self._is_mmio_address(cursor + write_size - 1)
        ):
            self.pointer_limit_store_ff_declines["mmio_target"] += 1
            return False
        value = int(self.uc.reg_read(value_reg_id)) & 0xFFFFFFFF
        pattern = (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
        try:
            self._write_strided_pattern(cursor, pattern, int(stride), int(iterations))
            # 只把指针推到终点：环头的 cmp/bne 随后自然落空退出。
            self.uc.reg_write(cursor_reg_id, limit & 0xFFFFFFFF)
            self.pointer_limit_store_ff_declines["granted"] += 1
            logger.info(
                "✓ 指针即计数器 store 循环 fast-forward: loop=0x%08x %s:0x%08x->0x%08x "
                "bytes=0x%x iterations=%d",
                int(owner) & ~1,
                cursor_reg,
                cursor,
                limit,
                write_size,
                int(iterations),
            )
            return self._record_loop_fast_forward(
                "pointer_limit_store_loop", loop_head=loop_head
            )
        except Exception as exc:
            logger.debug(
                "指针计数 store 循环 fast-forward 失败 @ 0x%08x: %s", int(loop_head), exc
            )
            return False

    def _parse_post_indexed_store_transfer(self, insn: Dict[str, object]) -> Optional[Tuple[str, str, int, int]]:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        width = {"STR": 4, "STRH": 2, "STRB": 1}.get(mnemonic)
        if width is None:
            return None
        parts = self._split_operands(insn.get("operands", ""))
        if len(parts) != 3:
            return None
        value_reg = parts[0].strip().lower()
        cursor_text = parts[1].strip().lower()
        if not (cursor_text.startswith("[") and cursor_text.endswith("]")):
            return None
        cursor_reg = cursor_text[1:-1].strip().lower()
        stride = self._parse_immediate_operand(parts[2])
        if stride is None:
            return None
        if self._arm_reg_id(value_reg) is None or self._arm_reg_id(cursor_reg) is None:
            return None
        return value_reg, cursor_reg, int(stride), int(width)

    def _write_strided_pattern(self, address: int, pattern: bytes, stride: int, iterations: int) -> None:
        if not pattern or iterations <= 0 or stride <= 0:
            return
        if stride == len(pattern):
            self._write_repeated_pattern(address, pattern, iterations * len(pattern))
            return
        # Batch a finite number of sparse destinations.  Each batch is fully
        # mapped before its first write; this bounds temporary Python state and
        # keeps retry behaviour deterministic for very large startup loops.
        batch_size = 256
        for batch_start in range(0, iterations, batch_size):
            batch_end = min(iterations, batch_start + batch_size)
            ranges = [
                (int(address) + index * int(stride), len(pattern))
                for index in range(batch_start, batch_end)
            ]
            if not self._ensure_memory_ranges_mapped(ranges):
                first = ranges[0][0]
                raise RuntimeError(f"cannot map strided write batch @ 0x{first:08x}")
            for current, _size in ranges:
                self.uc.mem_write(current, pattern)

    def _parse_post_indexed_byte_transfer(self, insn: Dict[str, object]) -> Optional[Tuple[str, str, int]]:
        operands = str(insn.get("operands", "") or "")
        match = re.match(
            r"\s*([a-z0-9]+)\s*,\s*\[\s*([a-z0-9]+)\s*\]\s*,\s*#?(-?(?:0x[0-9a-fA-F]+|\d+))\s*$",
            operands,
            re.I,
        )
        if not match:
            return None
        data_reg = match.group(1).strip().lower()
        base_reg = match.group(2).strip().lower()
        stride = self._parse_int(match.group(3))
        if stride is None:
            return None
        if self._arm_reg_id(data_reg) is None or self._arm_reg_id(base_reg) is None:
            return None
        return data_reg, base_reg, int(stride)

    def _force_byte_copy_loop_fallthrough(self, loop_head: int) -> bool:
        """Last-resort exit for a proven single-BB byte-copy self loop.

        Used only after the normal concrete fast-forward path fails at an
        intervention threshold. It does not discover new code by PC injection;
        it exits a reached finite-copy micro-loop at its architectural
        fallthrough when polluted decompressor state would otherwise underflow
        the counter and spin forever.
        """
        instructions = [dict(insn) for insn in self.static_bbs.get(int(loop_head), []) or []]
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 4 or len(instructions) > 16:
            return False
        branch_insn = instructions[-1]
        if self._normalize_mnemonic(branch_insn.get("mnemonic", "")) != "BNE":
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        target_bb = self.instruction_to_bb.get(int(branch_target or 0), branch_target)
        if int(target_bb or -1) != int(loop_head):
            return False

        has_load = False
        has_store = False
        count_reg = None
        for insn in instructions[:-1]:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic == "LDRB" and self._parse_post_indexed_byte_transfer(insn) is not None:
                has_load = True
            elif mnemonic == "STRB" and self._parse_post_indexed_byte_transfer(insn) is not None:
                has_store = True
            elif mnemonic in {"SUB", "SUBS"}:
                parts = self._split_operands(insn.get("operands", ""))
                if len(parts) >= 3 and parts[0].strip().lower() == parts[1].strip().lower():
                    decrement = self._parse_immediate_operand(parts[2])
                    if decrement == 1:
                        count_reg = parts[0].strip().lower()
        if not (has_load and has_store and count_reg):
            return False

        count_reg_id = self._arm_reg_id(count_reg)
        if count_reg_id is not None:
            try:
                self.uc.reg_write(count_reg_id, 0)
            except Exception:
                pass
        fallthrough = int(branch_insn.get("address", loop_head) or loop_head) + int(branch_insn.get("size", 4) or 4)
        if self.execution_thumb:
            fallthrough |= 1
        else:
            fallthrough &= ~1
        self.uc.reg_write(UC_ARM_REG_PC, fallthrough)
        logger.info(
            "✓ 字节拷贝循环fallthrough兜底: loop=0x%08x count=%s fallthrough=0x%08x",
            int(loop_head),
            count_reg,
            fallthrough & ~1,
        )
        return True

    def _parse_block_transfer_writeback(self, insn: Dict[str, object]) -> Tuple[Optional[str], List[str]]:
        operands = str(insn.get("operands", "") or "")
        match = re.search(r"^\s*([a-z0-9]+)\s*!", operands, re.I)
        if not match:
            return None, []
        base = match.group(1).strip().lower()
        if self._arm_reg_id(base) is None:
            return None, []
        if "{" not in operands or "}" not in operands:
            return None, []
        register_text = operands[operands.find("{") + 1 : operands.find("}")]
        regs = [part.strip().lower() for part in register_text.split(",") if part.strip()]
        if not regs or any(self._arm_reg_id(reg) is None for reg in regs):
            return None, []
        return base, regs

    def _runtime_constraint_count(self) -> int:
        return len(self.memory_read_constraints) + len(getattr(self.mmio_handler, "static_constraints", {}))

    def _current_input_read_occurrence(
        self,
        read_pc: Optional[int],
        address: int,
    ) -> Optional[int]:
        """Return the concrete input occurrence reached by the live execution."""
        if read_pc is None:
            return None
        try:
            occurrence = int(
                getattr(self, "input_occurrence_counts", {}).get(
                    (int(read_pc) & 0xFFFFFFFF, int(address) & 0xFFFFFFFF),
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            return None
        return occurrence if occurrence > 0 else None

    def _record_unresolved_loop_intervention(self, loop_head: int):
        failures = self.loop_intervention_failures.get(loop_head, 0) + 1
        self.loop_intervention_failures[loop_head] = failures
        if failures < self.max_unresolved_loop_interventions:
            return

        # r9 裁定：兜底熔断单列（loop_unresolved_limit，保持 diagnostic）。
        self.loop_unresolved_limit_trips += 1

        self.stop_requested_reason = "unresolved_loop_intervention_limit"
        logger.warning(
            "循环 0x%08x 连续 %d 次干预未产生新约束，停止当前路径",
            loop_head,
            failures,
        )
        self.uc.emu_stop()

    def _handle_local_self_loop_constraint(self, loop_head: int) -> bool:
        """Solve concrete self-loop load/compare branches without calling LLM."""
        try:
            loop_body = [
                int(bb)
                for bb in self.loop_classifier._get_loop_body_bbs(loop_head)
                if int(bb) in self.static_bbs
            ]
        except Exception:
            loop_body = [int(loop_head)] if int(loop_head) in self.static_bbs else []
        loop_info = self.loop_classifier.loop_heads.get(loop_head)
        mmio_addresses = sorted(getattr(loop_info, "mmio_addresses_accessed", set()) or [])
        for mmio_addr in mmio_addresses:
            value = self._infer_local_loop_exit_mmio_value(loop_body, loop_head, int(mmio_addr))
            if value is None:
                continue
            read_pc = self._find_recent_mmio_read_pc(loop_body, int(mmio_addr))
            constraint_pc = self._find_loop_constraint_pc(loop_body) or loop_head
            if not self._validate_loop_exit_mmio_value(
                loop_body,
                loop_head,
                int(mmio_addr),
                int(value),
                read_pc=read_pc,
                constraint_pc=constraint_pc,
            ):
                continue
            before = self._runtime_constraint_count()
            normalized = {
                "type": "mmio",
                "read_pc": read_pc,
                "address": int(mmio_addr) & 0xFFFFFFFF,
                "value": int(value) & 0xFFFFFFFF,
                "constraint_pc": constraint_pc,
                "description": f"Local loop-exit MMIO constraint from loop @ 0x{loop_head:08x}",
            }
            read_occurrence = self._current_input_read_occurrence(
                read_pc,
                int(mmio_addr),
            )
            if read_occurrence is not None:
                normalized["read_occurrence"] = read_occurrence
            self._apply_constraint(normalized)
            self._install_runtime_loop_branch_force(loop_body, loop_head, int(mmio_addr), int(value))
            applied = self._runtime_constraint_count() > before or bool(self.runtime_loop_branch_forces)
            if applied:
                logger.info(
                    "✓ 本地MMIO循环求解 @ 0x%08x: read_pc=%s addr=0x%08x value=0x%08x",
                    loop_head,
                    f"0x{read_pc:08x}" if read_pc is not None else "None",
                    int(mmio_addr) & 0xFFFFFFFF,
                    int(value) & 0xFFFFFFFF,
                )
                return True

        snapshot = self._build_current_loop_snapshot(loop_head)
        if snapshot is None:
            return self._handle_local_short_memory_loop_constraint(loop_head)

        try:
            analysis = self.code_analyzer._analyze_self_loop_constraint(snapshot)
        except Exception as e:
            logger.debug("本地self-loop求解失败 @ 0x%08x: %s", loop_head, e)
            return self._handle_local_short_memory_loop_constraint(loop_head)

        if float(getattr(analysis, "confidence", 0.0) or 0.0) < 0.8:
            return self._handle_local_short_memory_loop_constraint(loop_head)
        constraints = list(getattr(analysis, "suggested_constraints", []) or [])
        if not constraints:
            return self._handle_local_short_memory_loop_constraint(loop_head)

        logger.info(
            "✓ 本地self-loop求解 @ 0x%08x: %s",
            loop_head,
            getattr(analysis, "reason", ""),
        )
        applied = False
        for constraint in constraints:
            constraint_for_normalize = dict(constraint)
            if str(constraint_for_normalize.get("type", "")).lower() == "memory":
                address = self._parse_int(constraint_for_normalize.get("address"))
                read_pc = self._parse_int(constraint_for_normalize.get("read_pc"))
                if (
                    address is not None
                    and read_pc is not None
                    and self._is_modeled_async_state_source(loop_body, int(address), int(read_pc))
                ):
                    constraint_for_normalize["modeled_async_state"] = True
            normalized = self._normalize_runtime_constraint(constraint_for_normalize)
            if normalized is None:
                continue
            dynamic_rule = (
                self._derive_self_loop_dynamic_memory_rule(snapshot, normalized)
                if self.enable_dynamic_memory_constraints
                else None
            )
            if dynamic_rule is not None:
                dynamic_rule = self._validated_dynamic_memory_rule(
                    dynamic_rule,
                    normalized,
                    source="self-loop",
                )
            if dynamic_rule is not None:
                normalized["dynamic"] = dynamic_rule
            before = self._runtime_constraint_count()
            self._apply_constraint(normalized)
            if not bool(normalized.get("modeled_async_state")):
                self._install_runtime_self_loop_branch_force(snapshot, loop_head, normalized)
            applied = True
            applied = self._runtime_constraint_count() > before or applied
        return applied

    def _handle_local_short_memory_loop_constraint(self, loop_head: int) -> bool:
        """
        Solve short natural loops whose load/compare/exit branch spans several
        BBs. This covers table scans and external-memory waits such as:

            loop: ldr r3, [r1]
                  ...
            test: cmp r3, r0
                  bne next_iteration

        The fix is still concrete execution: write the load source value and
        optionally force the proven branch flags at the real branch PC.
        """
        candidate = self._derive_short_loop_load_constraint(loop_head)
        if not candidate:
            return False

        constraint = candidate.get("constraint")
        if not isinstance(constraint, dict):
            return False
        normalized = self._normalize_runtime_constraint(constraint)
        if normalized is None:
            return False

        accepted, reason = self._validate_analysis_constraint(
            normalized,
            candidate.get("branch_pc"),
            candidate.get("desired_branch_taken"),
        )
        if accepted is False:
            self._record_constraint_validation(
                "local_short_loop",
                loop_head,
                normalized,
                accepted,
                reason,
                control_branch_pc=candidate.get("branch_pc"),
                desired_branch_taken=candidate.get("desired_branch_taken"),
            )
            logger.debug("短自然循环约束未通过语义校验: %s (%s)", normalized, reason)
            return False

        dynamic_rule = candidate.get("dynamic")
        if self.enable_dynamic_memory_constraints and dynamic_rule is not None:
            dynamic_rule = self._validated_dynamic_memory_rule(
                dynamic_rule,
                normalized,
                source="short-loop",
            )
            if dynamic_rule is not None:
                normalized["dynamic"] = dynamic_rule

        before = self._runtime_constraint_count()
        self._apply_constraint(normalized)
        branch_pc = self._parse_int(candidate.get("branch_pc"))
        desired_taken = candidate.get("desired_branch_taken")
        if (
            branch_pc is not None
            and desired_taken is not None
            and not bool(normalized.get("modeled_async_state"))
        ):
            self._install_runtime_loop_branch_force_at(
                branch_pc,
                bool(desired_taken),
                int(loop_head),
                int(normalized.get("address") or 0),
                int(normalized.get("value") or 0),
            )
        applied = self._runtime_constraint_count() > before or bool(self.runtime_loop_branch_forces.get(int(branch_pc or 0)))
        if applied:
            logger.info(
                "✓ 本地短自然循环求解 @ 0x%08x: read_pc=%s addr=%s value=%s branch=%s desired=%s",
                loop_head,
                self._format_optional_hex(normalized.get("read_pc")),
                self._format_optional_hex(normalized.get("address")),
                self._format_optional_hex(normalized.get("value")),
                self._format_optional_hex(branch_pc),
                "taken" if desired_taken else "not-taken",
            )
        return applied

    def _handle_memory_mapped_wait_loop(self, loop_head: int) -> bool:
        """
        Solve reached hardware/status waits that use non-standard mapped
        addresses rather than the Cortex-M 0x40000000 MMIO window.

        This is still entry-derived execution: the loop must already be hot,
        the read PC must be inside the reached loop body, and the value is
        derived from the local load/flag/branch chain. The handler writes only
        a read-PC-scoped memory constraint and optionally fixes CPSR at the
        proven branch PC to avoid stale translated-block flags.
        """
        loop_info = self.loop_classifier.loop_heads.get(loop_head)
        if loop_info is None:
            return False

        try:
            loop_body = [
                int(bb)
                for bb in self.loop_classifier._get_loop_body_bbs(loop_head)
                if int(bb) in self.static_bbs
            ]
        except Exception:
            loop_body = [int(loop_head)] if int(loop_head) in self.static_bbs else []
        if not loop_body or len(loop_body) > 16:
            return False

        # Finite data loops should be handled by the fast-forward paths. This
        # handler is for small status/control register sets with no cursor or
        # counter progress.
        if bool(getattr(loop_info, "has_counter_increment", False)):
            return False
        memory_address_count = len(getattr(loop_info, "memory_addresses_accessed", set()) or set())
        if memory_address_count <= 0 or memory_address_count > 16:
            return False

        branch_pc = self._find_loop_branch_pc(loop_body, loop_head)
        if branch_pc is None:
            return False

        desired_taken = self._desired_loop_exit_branch_direction(loop_body, loop_head, branch_pc)
        if desired_taken is None:
            return False

        candidates = self._recent_loop_memory_read_candidates(loop_body)
        if not candidates:
            return False

        for read_pc, address in candidates:
            if not self._memory_wait_address_allowed(address):
                continue

            value = self._infer_local_loop_exit_mmio_value(
                loop_body,
                loop_head,
                int(address),
                read_pc=int(read_pc),
            )
            if value is None:
                continue

            write_size = self._constraint_write_size(int(read_pc))
            mask = (1 << (max(1, min(4, write_size)) * 8)) - 1
            normalized = {
                "type": "memory",
                "read_pc": int(read_pc),
                "address": int(address) & 0xFFFFFFFF,
                "value": int(value) & mask,
                "constraint_pc": self._find_loop_constraint_pc(loop_body) or int(branch_pc),
                "external_memory": True,
                "description": (
                    f"Local mapped-memory wait exit constraint from loop @ 0x{loop_head:08x}"
                ),
            }

            before = self._runtime_constraint_count()
            self._apply_constraint(normalized)
            forced = self._install_runtime_loop_branch_force_at(
                int(branch_pc),
                bool(desired_taken),
                int(loop_head),
                int(address),
                int(value),
            )
            applied = self._runtime_constraint_count() > before or forced
            if not applied:
                continue

            logger.info(
                "✓ 本地映射内存等待循环求解 @ 0x%08x: read_pc=0x%08x addr=0x%08x value=0x%08x branch=0x%08x desired=%s",
                int(loop_head),
                int(read_pc),
                int(address) & 0xFFFFFFFF,
                int(value) & mask,
                int(branch_pc),
                "taken" if desired_taken else "not-taken",
            )
            return True
        return False

    def _arm_base_mnemonic_and_predicate(self, mnemonic: object) -> Tuple[str, Optional[str]]:
        """Split ARM predicated mnemonics such as LDRLT/CMPLT/MOVNE."""
        raw = str(mnemonic or "").strip().upper()
        if not raw:
            return "", None
        raw = raw.split(".")[0]
        normalized = self._normalize_mnemonic(raw)
        if self._branch_condition_for_mnemonic(normalized) is not None or normalized in {"B", "BAL", "BL", "BLX", "BX", "BXJ"}:
            return normalized, None

        predicable_bases = {
            "ADC", "ADD", "AND", "ASR", "BIC", "CMN", "CMP", "EOR", "LDR",
            "LDRB", "LDRH", "LDRSB", "LDRSH", "LSL", "LSR", "MOV", "MVN",
            "ORR", "ROR", "RSB", "SBC", "STR", "STRB", "STRH", "SUB",
            "TEQ", "TST", "UXTB", "UXTH",
        }
        for condition in sorted(THUMB_CONDITION_CODES, key=len, reverse=True):
            if not raw.endswith(condition):
                continue
            base = raw[: -len(condition)]
            if base in predicable_bases:
                return base, condition
        return normalized, None

    def _load_width_for_base_mnemonic(self, mnemonic: str) -> int:
        mnemonic = str(mnemonic or "").upper()
        if mnemonic in {"LDRB", "LDRSB"}:
            return 1
        if mnemonic in {"LDRH", "LDRSH"}:
            return 2
        return 4

    def _runtime_register_values(self) -> Dict[str, int]:
        values = self._get_registers()
        return {str(key).lower(): int(value) & 0xFFFFFFFF for key, value in values.items()}

    def _condition_value_for_shifted_zero_compare(
        self,
        condition: str,
        shift_left: int,
        width: int,
    ) -> Optional[int]:
        """Return a raw load value that makes `(value << shift) cmp #0` satisfy condition."""
        condition = str(condition or "").upper()
        shift_left = int(shift_left or 0)
        width = max(1, min(4, int(width or 4)))
        raw_mask = (1 << (width * 8)) - 1
        if condition in {"LT", "MI"}:
            bit = 31 - shift_left
            if 0 <= bit < width * 8:
                return (1 << bit) & raw_mask
            return None
        if condition in {"GE", "PL", "EQ"}:
            return 0
        if condition == "NE":
            return 1 & raw_mask
        return None

    def _handle_predicated_status_wait_loop(self, loop_head: int) -> bool:
        """
        Solve ARM predicated status waits that encode a compound condition.

        Example reached in Honeywell scanner firmware:

            ldr     r0, [r1, #0xd0]
            lsl     r0, r0, #0x19
            cmp     r0, #0          ; require LT => source bit 6 set
            ldrlt   r0, [r1, #0x98]
            lsllt   r0, r0, #0x1c
            cmplt   r0, #0          ; require LT => source bit 3 set
            movlt   r0, #1
            blt     success
            mov     r0, #0
        success:
            cmp     r0, #0
            beq     loop

        This remains entry-derived execution: the loop has already been
        reached, and the synthesized values are read-PC scoped status values
        required for the architectural branch chain to fall through.
        """
        try:
            loop_body = [
                int(bb)
                for bb in self.loop_classifier._get_loop_body_bbs(loop_head)
                if int(bb) in self.static_bbs
            ]
        except Exception:
            loop_body = [int(loop_head)] if int(loop_head) in self.static_bbs else []
        if not loop_body or len(loop_body) > 8:
            return False

        # This handler models compact ARM predicated instruction sequences.
        # The recent dynamic order at the intervention point is often
        # back-edge order (tail -> head); use local address order to recover
        # the actual fallthrough/skip structure.
        ordered_bbs = sorted(int(bb) for bb in loop_body)
        instructions: List[Dict[str, object]] = []
        for bb in ordered_bbs:
            instructions.extend(self.static_bbs.get(int(bb), []) or [])
        if len(instructions) < 6 or len(instructions) > 64:
            return False

        branch_index = None
        branch_insn = None
        branch_pc = None
        for index in range(len(instructions) - 1, -1, -1):
            insn = instructions[index]
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if self._branch_condition_for_mnemonic(mnemonic) is None:
                continue
            target = self._parse_branch_target(insn.get("operands", ""))
            if target is None:
                continue
            normalized_target = self.instruction_to_bb.get(int(target), int(target))
            if int(normalized_target) != int(loop_head):
                continue
            branch_index = index
            branch_insn = insn
            branch_pc = int(insn.get("address", 0) or 0)
            break
        if branch_index is None or branch_insn is None:
            return False

        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        if branch_target is None:
            return False
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))
        loop_back_taken = self.instruction_to_bb.get(int(branch_target), int(branch_target)) in loop_targets
        desired_branch_taken = not loop_back_taken
        if branch_mnemonic not in {"BEQ", "BNE"}:
            return False

        final_cmp_index = None
        final_reg = None
        for index in range(branch_index - 1, max(-1, branch_index - 8), -1):
            insn = instructions[index]
            base, _predicate = self._arm_base_mnemonic_and_predicate(insn.get("mnemonic", ""))
            if base != "CMP":
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if len(parts) < 2:
                continue
            compare_value = self._parse_immediate_operand(parts[1])
            reg = self._normalize_register_name(parts[0])
            if reg is None or compare_value != 0:
                continue
            final_cmp_index = index
            final_reg = reg
            break
        if final_cmp_index is None or final_reg is None:
            return False

        # For a loop-back BEQ, exiting means final_reg != 0.  For a loop-back
        # BNE, exiting means final_reg == 0.  The pattern below targets the
        # common success case where a predicated MOV sets a non-zero sentinel
        # and a same-condition branch skips the zero assignment.
        need_nonzero = (
            (branch_mnemonic == "BEQ" and not desired_branch_taken)
            or (branch_mnemonic == "BNE" and desired_branch_taken)
        )
        if not need_nonzero:
            return False

        success_condition = None
        success_branch_pc = None
        setter_index = None
        final_cmp_pc = int(instructions[final_cmp_index].get("address", 0) or 0)
        for index in range(final_cmp_index - 1, max(-1, final_cmp_index - 16), -1):
            candidate = instructions[index]
            candidate_mnemonic = self._normalize_mnemonic(candidate.get("mnemonic", ""))
            condition = self._branch_condition_for_mnemonic(candidate_mnemonic)
            if condition is None:
                continue
            target = self._parse_branch_target(candidate.get("operands", ""))
            if int(target or 0) != final_cmp_pc:
                continue
            for setter_candidate_index in range(index - 1, max(-1, index - 5), -1):
                setter = instructions[setter_candidate_index]
                base, predicate = self._arm_base_mnemonic_and_predicate(setter.get("mnemonic", ""))
                if base != "MOV" or predicate != condition:
                    continue
                parts = self._split_operands(setter.get("operands", ""))
                if len(parts) < 2 or self._normalize_register_name(parts[0]) != final_reg:
                    continue
                immediate = self._parse_immediate_operand(parts[1])
                if immediate is None or int(immediate) == 0:
                    continue
                success_condition = condition
                success_branch_pc = int(candidate.get("address", 0) or 0)
                setter_index = setter_candidate_index
                break
            if success_condition is not None:
                break
        if success_condition is None or setter_index is None or success_branch_pc is None:
            return False

        reg_values = self._runtime_register_values()
        reg_exprs: Dict[str, Dict[str, object]] = {}
        derived_constraints: List[Dict[str, object]] = []
        for index, insn in enumerate(instructions[: setter_index + 1]):
            base, predicate = self._arm_base_mnemonic_and_predicate(insn.get("mnemonic", ""))
            if predicate is not None and predicate != success_condition:
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if not parts:
                continue

            if base.startswith("LDR") and len(parts) >= 2:
                dest = self._normalize_register_name(parts[0])
                if dest is None:
                    continue
                memory_operand = ", ".join(parts[1:])
                address = self.code_analyzer.resolve_memory_operand_for_instruction(
                    insn,
                    memory_operand,
                    reg_values,
                )
                if address is None:
                    reg_exprs.pop(dest, None)
                    continue
                reg_exprs[dest] = {
                    "type": "memory",
                    "read_pc": int(insn.get("address", 0) or 0),
                    "address": int(address) & 0xFFFFFFFF,
                    "width": self._load_width_for_base_mnemonic(base),
                    "shift_left": 0,
                }
                continue

            if base in {"LSL", "LSLS"} and len(parts) >= 3:
                dest = self._normalize_register_name(parts[0])
                src = self._normalize_register_name(parts[1])
                shift = self._parse_immediate_operand(parts[2])
                if dest is None or src is None or shift is None or src not in reg_exprs:
                    if dest is not None:
                        reg_exprs.pop(dest, None)
                    continue
                expr = dict(reg_exprs[src])
                expr["shift_left"] = int(expr.get("shift_left", 0) or 0) + int(shift)
                reg_exprs[dest] = expr
                continue

            if base == "CMP" and len(parts) >= 2:
                src = self._normalize_register_name(parts[0])
                compare_value = self._parse_immediate_operand(parts[1])
                if src is None or compare_value != 0 or src not in reg_exprs:
                    continue
                expr = reg_exprs[src]
                value = self._condition_value_for_shifted_zero_compare(
                    success_condition,
                    int(expr.get("shift_left", 0) or 0),
                    int(expr.get("width", 4) or 4),
                )
                if value is None:
                    continue
                constraint = {
                    "type": "memory",
                    "read_pc": int(expr["read_pc"]),
                    "address": int(expr["address"]) & 0xFFFFFFFF,
                    "value": int(value) & 0xFFFFFFFF,
                    "constraint_pc": int(insn.get("address", 0) or 0),
                    "modeled_async_state": True,
                    "description": (
                        f"Predicated status wait exit from loop @ 0x{loop_head:08x}; "
                        f"make {base}{success_condition} true before 0x{success_branch_pc:08x}"
                    ),
                }
                key = (constraint["read_pc"], constraint["address"])
                if not any((item["read_pc"], item["address"]) == key for item in derived_constraints):
                    derived_constraints.append(constraint)
                continue

            dest = self._normalize_register_name(parts[0])
            if dest is not None and base not in {"CMP", "CMN", "TST", "TEQ", "B", "BLT", "MOV"}:
                reg_exprs.pop(dest, None)

        if not derived_constraints:
            return False

        applied = False
        for constraint in derived_constraints:
            before = self._runtime_constraint_count()
            self._apply_constraint(constraint)
            applied = applied or self._runtime_constraint_count() > before

        if derived_constraints:
            first = derived_constraints[0]
            applied = self._install_runtime_loop_branch_force_at(
                int(success_branch_pc),
                True,
                int(loop_head),
                int(first["address"]),
                int(first["value"]),
            ) or applied
            applied = self._install_runtime_loop_branch_force_at(
                int(branch_pc),
                bool(desired_branch_taken),
                int(loop_head),
                int(first["address"]),
                int(first["value"]),
            ) or applied

        if applied:
            logger.info(
                "✓ ARM条件执行状态等待循环求解 @ 0x%08x: condition=%s constraints=%s final_branch=0x%08x desired=%s",
                int(loop_head),
                success_condition,
                ", ".join(
                    f"read_pc=0x{int(item['read_pc']):08x} addr=0x{int(item['address']):08x} value=0x{int(item['value']) & 0xFFFFFFFF:08x}"
                    for item in derived_constraints
                ),
                int(branch_pc),
                "taken" if desired_branch_taken else "not-taken",
            )
        return applied

    def _handle_status_register_exit_loop(self, loop_head: int) -> bool:
        """
        Solve compact multi-BB state/status loops without cold PC injection.

        A common RTOS/driver pattern repeatedly tries to update an object,
        sets a local status register on success, and loops while the status is
        zero:

            ... load state field
            cmp state, expected
            bne retry_or_alt
            mov status, #2
            b final_test
        retry_or_alt:
            ldr state, [object, #off]
            cmp state, #0
            moveq status, #1
        final_test:
            cmp status, #0
            beq loop_head

        The solver derives a read-PC scoped memory/MMIO value that makes one of
        the already-reached success setters execute, then forces only the
        proven branch flags at the real branch instructions.
        """
        loop_body, instructions = self._status_loop_body_and_instructions(loop_head)
        if not instructions:
            return False

        self.status_loop_solver_stats["attempted"] += 1
        final = self._find_status_loop_final_branch(instructions, loop_body, loop_head)
        if final is None:
            self.status_loop_solver_stats["rejected"] += 1
            return False
        final_branch_index, final_branch_insn, desired_final_taken = final

        final_compare = self._find_status_loop_final_compare(instructions, final_branch_index)
        if final_compare is None:
            self.status_loop_solver_stats["rejected"] += 1
            return False
        final_cmp_index, status_reg = final_compare

        candidates = []
        for setter_index in range(final_cmp_index - 1, -1, -1):
            setter = instructions[setter_index]
            base, predicate = self._arm_base_mnemonic_and_predicate(setter.get("mnemonic", ""))
            if base != "MOV":
                continue
            parts = self._split_operands(setter.get("operands", ""))
            if len(parts) < 2:
                continue
            if self._normalize_register_name(parts[0]) != status_reg:
                continue
            status_value = self._parse_immediate_operand(parts[1])
            if status_value is None:
                continue
            if not self._status_value_exits_final_branch(
                int(status_value),
                instructions[final_cmp_index],
                final_branch_insn,
                bool(desired_final_taken),
                status_reg,
            ):
                continue

            if predicate is not None:
                candidate = self._derive_predicated_status_setter_candidate(
                    instructions,
                    setter_index,
                    predicate,
                    final_cmp_index,
                    loop_body,
                    loop_head,
                )
            else:
                candidate = self._derive_unconditional_status_setter_candidate(
                    instructions,
                    setter_index,
                    final_cmp_index,
                    loop_body,
                    loop_head,
                )
            if candidate is not None:
                candidate["setter_pc"] = int(setter.get("address", 0) or 0)
                candidate["status_value"] = int(status_value) & 0xFFFFFFFF
                candidates.append(candidate)

        if not candidates:
            self.status_loop_solver_stats["rejected"] += 1
            return False

        # Prefer candidates whose load was observed recently; they are less
        # likely to be stale static alternatives in a large loop body.
        recent_read_pcs = {
            int(pc)
            for pc, _addr, is_read, _value in self.memory_access_history[-256:]
            if is_read
        }
        candidates.sort(
            key=lambda item: (
                0 if int((item.get("constraint") or {}).get("read_pc") or 0) in recent_read_pcs else 1,
                int(item.get("setter_pc") or 0),
            )
        )

        final_branch_pc = int(final_branch_insn.get("address", 0) or 0)
        final_addr = 0
        final_value = 0
        for candidate in candidates:
            constraint = candidate.get("constraint")
            if not isinstance(constraint, dict):
                continue
            normalized = self._normalize_runtime_constraint(constraint)
            if normalized is None:
                continue
            before = self._runtime_constraint_count()
            self._apply_constraint(normalized)
            applied = self._runtime_constraint_count() > before

            control_branch_pc = self._parse_int(candidate.get("control_branch_pc"))
            control_desired = candidate.get("control_branch_taken")
            if control_branch_pc is not None and control_desired is not None:
                applied = self._install_runtime_loop_branch_force_at(
                    int(control_branch_pc),
                    bool(control_desired),
                    int(loop_head),
                    int(normalized.get("address") or 0),
                    int(normalized.get("value") or 0),
                ) or applied

            applied = self._install_runtime_loop_branch_force_at(
                int(final_branch_pc),
                bool(desired_final_taken),
                int(loop_head),
                int(normalized.get("address") or 0),
                int(normalized.get("value") or 0),
            ) or applied

            if not applied:
                continue

            self.status_loop_solver_stats["applied"] += 1
            final_addr = int(normalized.get("address") or 0) & 0xFFFFFFFF
            final_value = int(normalized.get("value") or 0) & 0xFFFFFFFF
            logger.info(
                "✓ 状态寄存器循环求解 @ 0x%08x: setter=0x%08x read_pc=%s addr=0x%08x value=0x%08x control=%s desired=%s final=0x%08x final_desired=%s",
                int(loop_head),
                int(candidate.get("setter_pc") or 0),
                self._format_optional_hex(normalized.get("read_pc")),
                final_addr,
                final_value,
                self._format_optional_hex(control_branch_pc),
                "taken" if control_desired else "not-taken" if control_desired is not None else "none",
                int(final_branch_pc),
                "taken" if desired_final_taken else "not-taken",
            )
            return True

        self.status_loop_solver_stats["rejected"] += 1
        return False

    def _status_loop_body_and_instructions(self, loop_head: int) -> Tuple[List[int], List[Dict[str, object]]]:
        try:
            loop_body = [
                int(bb)
                for bb in self.loop_classifier._get_loop_body_bbs(loop_head)
                if int(bb) in self.static_bbs
            ]
        except Exception:
            loop_body = []

        owner = self._ensure_dynamic_basic_block(loop_head)
        if int(owner) in self.static_bbs and int(owner) not in loop_body:
            loop_body.append(int(owner))
        if int(loop_head) in self.static_bbs and int(loop_head) not in loop_body:
            loop_body.append(int(loop_head))

        if not loop_body or len(loop_body) > 24:
            return loop_body, []
        loop_body = self._expand_local_status_loop_cfg(loop_body, loop_head)
        if not loop_body or len(loop_body) > 24:
            return loop_body, []

        seen_addresses: Set[int] = set()
        instructions: List[Dict[str, object]] = []
        for bb in loop_body:
            for insn in self.static_bbs.get(int(bb), []) or []:
                address = int(insn.get("address", 0) or 0)
                if address in seen_addresses:
                    continue
                seen_addresses.add(address)
                instructions.append(dict(insn))
        instructions.sort(key=lambda item: int(item.get("address", 0) or 0))
        if len(instructions) < 5 or len(instructions) > 160:
            return loop_body, []
        return loop_body, instructions

    def _expand_local_status_loop_cfg(self, seeds: List[int], loop_head: int) -> List[int]:
        """
        Expand a reached loop head into nearby branch successors for analysis.

        Runtime-discovered BBs stop at the first branch.  Multi-BB state loops
        often place the success setter and the final loop-back test in local
        branch successors, so a single-BB view loses the constraint relation.
        This only decodes local successors in a small window around the reached
        loop; it does not mark them covered or set PC to them.
        """
        window_start = max(0, (int(loop_head) & ~1) - 0x100)
        window_end = (int(loop_head) & ~1) + 0x200
        ordered: List[int] = []
        seen: Set[int] = set()
        queue: List[int] = []
        for seed in seeds:
            seed = int(seed) & ~1
            if seed in seen:
                continue
            seen.add(seed)
            ordered.append(seed)
            queue.append(seed)

        while queue and len(ordered) < 24:
            bb = queue.pop(0)
            instructions = self.static_bbs.get(int(bb), []) or []
            if not instructions:
                continue
            last = instructions[-1]
            last_pc = int(last.get("address", bb) or bb)
            last_size = int(last.get("size", 4) or 4)
            mnemonic = self._normalize_mnemonic(last.get("mnemonic", ""))
            condition = self._dispatch_condition_for_instruction(last)
            successor_pcs: List[int] = []
            target = self._parse_branch_target(last.get("operands", ""))

            if condition is not None and not self._is_switch_dispatch_condition(condition) and not self._is_call_dispatch_condition(condition):
                if target is not None:
                    successor_pcs.append(int(target) & ~1)
                successor_pcs.append((last_pc + last_size) & ~1)
            elif mnemonic in {"B", "BAL"} and target is not None:
                successor_pcs.append(int(target) & ~1)
            elif mnemonic not in {"BX", "BXJ", "POP", "LDM", "LDMIA"}:
                successor_pcs.append((last_pc + last_size) & ~1)

            for pc in successor_pcs:
                if not (window_start <= int(pc) <= window_end):
                    continue
                succ = self._ensure_dynamic_basic_block(int(pc)) & ~1
                if succ not in self.static_bbs:
                    continue
                if not (window_start <= int(succ) <= window_end):
                    continue
                if succ in seen:
                    continue
                seen.add(succ)
                ordered.append(succ)
                queue.append(succ)

        ordered.sort()
        return ordered

    def _find_status_loop_final_branch(
        self,
        instructions: List[Dict[str, object]],
        loop_body: List[int],
        loop_head: int,
    ) -> Optional[Tuple[int, Dict[str, object], bool]]:
        if not instructions:
            return None
        min_pc = min(int(insn.get("address", 0) or 0) for insn in instructions)
        max_pc = max(int(insn.get("address", 0) or 0) for insn in instructions)
        instruction_pcs = {int(insn.get("address", 0) or 0) for insn in instructions}
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))

        for index in range(len(instructions) - 1, -1, -1):
            insn = instructions[index]
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            condition = self._branch_condition_for_mnemonic(mnemonic)
            if condition is None or condition in {"TBB", "TBH"}:
                continue
            target = self._parse_branch_target(insn.get("operands", ""))
            if target is None:
                continue
            branch_pc = int(insn.get("address", 0) or 0)
            normalized_target = self.instruction_to_bb.get(int(target), int(target))
            target_in_loop = normalized_target in loop_targets or int(target) in instruction_pcs
            local_backward = int(target) < branch_pc and (min_pc - 0x100) <= int(target) <= max_pc
            if target_in_loop or local_backward:
                # The conditional branch is the loop-back edge; exiting means
                # the branch must not be taken.
                return index, insn, False
        return None

    def _find_status_loop_final_compare(
        self,
        instructions: List[Dict[str, object]],
        branch_index: int,
    ) -> Optional[Tuple[int, str]]:
        for index in range(branch_index - 1, max(-1, branch_index - 8), -1):
            insn = instructions[index]
            base, _predicate = self._arm_base_mnemonic_and_predicate(insn.get("mnemonic", ""))
            if base not in {"CMP", "CMN", "TST", "TEQ"}:
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if len(parts) < 2:
                continue
            reg = self._normalize_register_name(parts[0])
            if reg is None:
                continue
            return index, reg
        return None

    def _status_value_exits_final_branch(
        self,
        status_value: int,
        compare_insn: Dict[str, object],
        branch_insn: Dict[str, object],
        desired_branch_taken: bool,
        status_reg: str,
    ) -> bool:
        base, _predicate = self._arm_base_mnemonic_and_predicate(compare_insn.get("mnemonic", ""))
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        parts = self._split_operands(compare_insn.get("operands", ""))
        if len(parts) < 2:
            return False

        reg_values = self._runtime_register_values()

        def operand_value(token: str) -> Optional[int]:
            reg = self._normalize_register_name(token)
            if reg == status_reg:
                return int(status_value) & 0xFFFFFFFF
            if reg is not None:
                return int(reg_values.get(reg, 0)) & 0xFFFFFFFF
            immediate = self._parse_immediate_operand(token)
            if immediate is not None:
                return int(immediate) & 0xFFFFFFFF
            return None

        left = operand_value(parts[0])
        right = operand_value(parts[1])
        if left is None or right is None:
            return False
        actual_taken = self._branch_condition_result_from_compare(
            base,
            branch_mnemonic,
            int(left),
            int(right),
        )
        return actual_taken is not None and bool(actual_taken) == bool(desired_branch_taken)

    @staticmethod
    def _signed32(value: int) -> int:
        value = int(value) & 0xFFFFFFFF
        return value - 0x100000000 if value & 0x80000000 else value

    def _branch_condition_result_from_compare(
        self,
        compare_mnemonic: str,
        branch_mnemonic: str,
        left: int,
        right: int,
    ) -> Optional[bool]:
        compare_mnemonic = self._normalize_mnemonic(compare_mnemonic)
        branch_mnemonic = self._normalize_mnemonic(branch_mnemonic)
        condition = self._branch_condition_for_mnemonic(branch_mnemonic)
        if condition is None:
            return None

        left &= 0xFFFFFFFF
        right &= 0xFFFFFFFF
        if compare_mnemonic == "CMP":
            if condition == "EQ":
                return left == right
            if condition == "NE":
                return left != right
            if condition in {"HI"}:
                return left > right
            if condition in {"HS", "CS"}:
                return left >= right
            if condition in {"LO", "CC"}:
                return left < right
            if condition == "LS":
                return left <= right
            signed_left = self._signed32(left)
            signed_right = self._signed32(right)
            if condition == "GT":
                return signed_left > signed_right
            if condition == "GE":
                return signed_left >= signed_right
            if condition == "LT":
                return signed_left < signed_right
            if condition == "LE":
                return signed_left <= signed_right
            result = (left - right) & 0xFFFFFFFF
            if condition == "MI":
                return bool(result & 0x80000000)
            if condition == "PL":
                return not bool(result & 0x80000000)
            return None

        if compare_mnemonic == "CMN":
            result = (left + right) & 0xFFFFFFFF
            if condition == "EQ":
                return result == 0
            if condition == "NE":
                return result != 0
            if condition == "MI":
                return bool(result & 0x80000000)
            if condition == "PL":
                return not bool(result & 0x80000000)
            return None

        if compare_mnemonic == "TST":
            result = left & right
            if condition == "EQ":
                return result == 0
            if condition == "NE":
                return result != 0
            if condition == "MI":
                return bool(result & 0x80000000)
            if condition == "PL":
                return not bool(result & 0x80000000)
            return None

        if compare_mnemonic == "TEQ":
            result = left ^ right
            if condition == "EQ":
                return result == 0
            if condition == "NE":
                return result != 0
            return None
        return None

    def _derive_predicated_status_setter_candidate(
        self,
        instructions: List[Dict[str, object]],
        setter_index: int,
        predicate: str,
        final_cmp_index: int,
        loop_body: List[int],
        loop_head: int,
    ) -> Optional[Dict[str, object]]:
        compare_index = self._find_nearest_flag_compare_index(instructions, setter_index)
        if compare_index is None:
            return None
        branch_mnemonic = f"B{predicate.upper()}"
        constraint = self._derive_memory_constraint_from_compare(
            instructions,
            compare_index,
            branch_mnemonic,
            True,
            loop_body,
            loop_head,
        )
        if constraint is None:
            return None

        control_branch_pc = None
        control_branch_taken = None
        for index in range(setter_index + 1, final_cmp_index):
            insn = instructions[index]
            condition = self._branch_condition_for_mnemonic(self._normalize_mnemonic(insn.get("mnemonic", "")))
            if condition != predicate.upper():
                continue
            target = self._parse_branch_target(insn.get("operands", ""))
            if target is None:
                continue
            target_index = self._instruction_index_for_address(instructions, int(target))
            if target_index is not None and target_index <= final_cmp_index:
                control_branch_pc = int(insn.get("address", 0) or 0)
                control_branch_taken = True
                break

        if control_branch_pc is None and self._register_written_between(
            instructions,
            self._normalize_register_name(self._split_operands(instructions[setter_index].get("operands", ""))[0]) or "",
            setter_index + 1,
            final_cmp_index,
        ):
            return None

        return {
            "constraint": constraint,
            "control_branch_pc": control_branch_pc,
            "control_branch_taken": control_branch_taken,
        }

    def _derive_unconditional_status_setter_candidate(
        self,
        instructions: List[Dict[str, object]],
        setter_index: int,
        final_cmp_index: int,
        loop_body: List[int],
        loop_head: int,
    ) -> Optional[Dict[str, object]]:
        setter_pc = int(instructions[setter_index].get("address", 0) or 0)
        final_cmp_pc = int(instructions[final_cmp_index].get("address", 0) or 0)
        status_reg = self._normalize_register_name(self._split_operands(instructions[setter_index].get("operands", ""))[0])
        if status_reg is None:
            return None

        branch_candidate = None
        for branch_index in range(setter_index - 1, -1, -1):
            branch = instructions[branch_index]
            branch_mnemonic = self._normalize_mnemonic(branch.get("mnemonic", ""))
            if self._branch_condition_for_mnemonic(branch_mnemonic) is None:
                continue
            target = self._parse_branch_target(branch.get("operands", ""))
            if target is None:
                continue
            target = int(target)
            target_skips_setter = setter_pc < target <= final_cmp_pc
            target_enters_setter = int(branch.get("address", 0) or 0) < target <= setter_pc
            if not (target_skips_setter or target_enters_setter):
                continue
            desired_taken = bool(target_enters_setter)
            constraint = self._derive_memory_constraint_for_branch(
                instructions,
                branch_index,
                desired_taken,
                loop_body,
                loop_head,
            )
            if constraint is None:
                continue
            branch_candidate = {
                "constraint": constraint,
                "control_branch_pc": int(branch.get("address", 0) or 0),
                "control_branch_taken": bool(desired_taken),
            }
            break

        if branch_candidate is None:
            return None

        has_jump_to_final = False
        for index in range(setter_index + 1, final_cmp_index):
            insn = instructions[index]
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            target = self._parse_branch_target(insn.get("operands", ""))
            if mnemonic in {"B", "BAL"} and target is not None:
                target_index = self._instruction_index_for_address(instructions, int(target))
                if target_index is not None and target_index <= final_cmp_index:
                    has_jump_to_final = True
                    break

        if not has_jump_to_final and self._register_written_between(
            instructions,
            status_reg,
            setter_index + 1,
            final_cmp_index,
        ):
            return None
        return branch_candidate

    def _find_nearest_flag_compare_index(
        self,
        instructions: List[Dict[str, object]],
        before_index: int,
    ) -> Optional[int]:
        for index in range(before_index - 1, max(-1, before_index - 16), -1):
            base, _predicate = self._arm_base_mnemonic_and_predicate(instructions[index].get("mnemonic", ""))
            if base in {"CMP", "CMN", "TST", "TEQ"}:
                return index
        return None

    def _derive_memory_constraint_for_branch(
        self,
        instructions: List[Dict[str, object]],
        branch_index: int,
        desired_taken: bool,
        loop_body: List[int],
        loop_head: int,
    ) -> Optional[Dict[str, object]]:
        branch = instructions[branch_index]
        branch_mnemonic = self._normalize_mnemonic(branch.get("mnemonic", ""))

        if branch_mnemonic in {"CBZ", "CBNZ"}:
            snapshot = self._build_status_loop_analysis_snapshot(loop_head, instructions)
            if snapshot is None:
                return None
            reg_values, reg_sources = self.code_analyzer._evaluate_snapshot_registers(
                snapshot,
                end_index=branch_index - 1,
            )
            parts = self._split_operands(branch.get("operands", ""))
            source_reg = self._normalize_register_name(parts[0]) if parts else None
            source = reg_sources.get(source_reg) if source_reg else None
            if source is None:
                return None
            desired = self.code_analyzer._infer_cbz_cbnz_value(
                branch_mnemonic,
                bool(desired_taken),
                source_reg,
                reg_values,
                int(source.get("width", 4) or 4),
            )
            if desired is None:
                return None
            return self._constraint_from_register_source(
                source,
                int(desired),
                int(branch.get("address", 0) or 0),
                loop_body,
                loop_head,
                f"Status-loop CBZ/CBNZ exit via 0x{int(branch.get('address', 0) or 0):08x}",
            )

        compare_index = self._find_nearest_flag_compare_index(instructions, branch_index)
        if compare_index is None:
            return None
        return self._derive_memory_constraint_from_compare(
            instructions,
            compare_index,
            branch_mnemonic,
            bool(desired_taken),
            loop_body,
            loop_head,
        )

    def _derive_memory_constraint_from_compare(
        self,
        instructions: List[Dict[str, object]],
        compare_index: int,
        branch_mnemonic: str,
        desired_taken: bool,
        loop_body: List[int],
        loop_head: int,
    ) -> Optional[Dict[str, object]]:
        compare_insn = instructions[compare_index]
        compare_base, _predicate = self._arm_base_mnemonic_and_predicate(compare_insn.get("mnemonic", ""))
        if compare_base not in {"CMP", "CMN", "TST", "TEQ"}:
            return None
        compare_parts = self._split_operands(compare_insn.get("operands", ""))
        if len(compare_parts) < 2:
            return None

        snapshot = self._build_status_loop_analysis_snapshot(loop_head, instructions)
        if snapshot is None:
            return None
        reg_values, reg_sources = self.code_analyzer._evaluate_snapshot_registers(
            snapshot,
            end_index=compare_index - 1,
        )
        source_reg, source = self.code_analyzer._select_constraint_source(compare_parts, reg_sources)
        if source is None:
            source_reg, source = self._fallback_memory_source_before_compare(
                instructions,
                compare_index,
                compare_parts,
                reg_values,
            )
        if source is None or source_reg is None:
            return None

        width = int(source.get("width", 4) or 4)
        desired = self.code_analyzer._infer_branch_condition_value(
            self._normalize_mnemonic(branch_mnemonic),
            compare_base,
            source_reg,
            compare_parts,
            reg_values,
            width,
            bool(desired_taken),
        )
        if desired is None:
            return None

        return self._constraint_from_register_source(
            source,
            int(desired),
            int(compare_insn.get("address", 0) or 0),
            loop_body,
            loop_head,
            (
                f"Status-loop exit constraint from {self._normalize_mnemonic(branch_mnemonic)} "
                f"using 0x{int(compare_insn.get('address', 0) or 0):08x}"
            ),
        )

    def _fallback_memory_source_before_compare(
        self,
        instructions: List[Dict[str, object]],
        compare_index: int,
        compare_parts: List[str],
        reg_values: Dict[str, int],
    ) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
        candidate_regs = [
            self._normalize_register_name(part)
            for part in compare_parts[:2]
        ]
        candidate_regs = [reg for reg in candidate_regs if reg is not None]
        if not candidate_regs:
            return None, None

        for source_reg in candidate_regs:
            for index in range(compare_index - 1, max(-1, compare_index - 24), -1):
                insn = instructions[index]
                base, _predicate = self._arm_base_mnemonic_and_predicate(insn.get("mnemonic", ""))
                if base in {"BL", "BLX", "BX", "BXJ"}:
                    break
                if self._is_ldr_pc_dispatch_instruction(insn):
                    break
                if not base.startswith("LDR"):
                    continue
                parts = self._split_operands(insn.get("operands", ""))
                if len(parts) < 2 or self._normalize_register_name(parts[0]) != source_reg:
                    continue
                memory_operand = ", ".join(parts[1:])
                address = self.code_analyzer.resolve_memory_operand_for_instruction(
                    insn,
                    memory_operand,
                    reg_values,
                )
                if address is None:
                    continue
                return source_reg, {
                    "type": "mmio" if self._is_mmio_address(int(address)) else "memory",
                    "address": int(address) & 0xFFFFFFFF,
                    "read_pc": int(insn.get("address", 0) or 0),
                    "width": self._load_width_for_base_mnemonic(base),
                }
        return None, None

    def _constraint_from_register_source(
        self,
        source: Dict[str, object],
        desired_value: int,
        constraint_pc: int,
        loop_body: List[int],
        loop_head: int,
        description: str,
    ) -> Optional[Dict[str, object]]:
        source_type = str(source.get("type", "") or "").lower()
        address = self._parse_int(source.get("address"))
        read_pc = self._parse_int(source.get("read_pc"))
        if address is None or read_pc is None:
            return None
        width = int(source.get("width", 4) or 4)
        mask = (1 << (max(1, min(4, width)) * 8)) - 1

        if source_type == "mmio":
            if not self._is_mmio_address(int(address)):
                return None
            return {
                "type": "mmio",
                "read_pc": int(read_pc),
                "address": int(address) & 0xFFFFFFFF,
                "value": int(desired_value) & mask,
                "constraint_pc": int(constraint_pc),
                "description": f"{description}; loop @ 0x{int(loop_head):08x}",
            }

        flags = self._memory_constraint_flags_for_status_loop(loop_body, int(address), int(read_pc))
        if not self._memory_constraint_address_allowed(int(address), flags):
            return None
        return {
            "type": "memory",
            "read_pc": int(read_pc),
            "address": int(address) & 0xFFFFFFFF,
            "value": int(desired_value) & mask,
            "constraint_pc": int(constraint_pc),
            "external_memory": bool(flags.get("external_memory")),
            "modeled_async_state": bool(flags.get("modeled_async_state")),
            "description": f"{description}; loop @ 0x{int(loop_head):08x}",
        }

    def _memory_constraint_flags_for_status_loop(
        self,
        loop_body: List[int],
        address: int,
        read_pc: int,
    ) -> Dict[str, bool]:
        address = int(address) & 0xFFFFFFFF
        flags = {
            "external_memory": bool(address in self.external_memory_input_addresses),
            "modeled_async_state": False,
        }
        if self._is_modeled_async_state_source(loop_body, address, int(read_pc)):
            flags["modeled_async_state"] = True
        elif (
            self._page_down(address) in self.runtime_written_pages
            and self._is_mapped_non_mmio_data_address(address, 1)
        ):
            # In reached state-machine loops, runtime-written object fields can
            # legitimately be advanced by timers/IRQs or prior task phases.
            flags["modeled_async_state"] = True
        return flags

    def _build_status_loop_analysis_snapshot(
        self,
        loop_head: int,
        instructions: List[Dict[str, object]],
    ):
        memory_regions: Dict[Tuple[int, int], bytes] = {}
        for start, size in getattr(self.snapshot_manager, "memory_regions", []) or []:
            try:
                memory_regions[(int(start), int(size))] = bytes(self.uc.mem_read(int(start), int(size)))
            except Exception:
                continue

        self._add_pc_literal_pages_to_analysis_snapshot(memory_regions, instructions)
        candidate_addresses = {
            int(address) & 0xFFFFFFFF
            for _pc, address, is_read, _value in self.memory_access_history[-512:]
            if is_read
        }
        candidate_addresses.update(
            int(address) & 0xFFFFFFFF
            for _pc, address, _is_read, _value in self.memory_access_history[-512:]
        )
        for address in candidate_addresses:
            if self._is_mmio_address(address) or not self._is_mapped_non_mmio_data_address(address, 1):
                continue
            page = self._page_down(address)
            key = (page, 0x1000)
            if key in memory_regions:
                continue
            try:
                memory_regions[key] = bytes(self.uc.mem_read(page, 0x1000))
            except Exception:
                continue

        return SimpleNamespace(
            bb_address=int(loop_head),
            instruction_count=self.instruction_count,
            cpu_state=self._get_registers(),
            pc=int(loop_head),
            flags=int(self.uc.reg_read(UC_ARM_REG_CPSR)) if self.uc is not None else 0,
            memory_regions=memory_regions,
            mmio_values=dict(self._current_mmio_state()),
            mmio_access_history=list(self.mmio_access_history[-32:]),
            bb_instructions=list(instructions),
        )

    @staticmethod
    def _instruction_index_for_address(
        instructions: List[Dict[str, object]],
        address: int,
    ) -> Optional[int]:
        address = int(address) & ~1
        best_index = None
        for index, insn in enumerate(instructions):
            insn_addr = int(insn.get("address", 0) or 0) & ~1
            if insn_addr == address:
                return index
            if insn_addr < address:
                best_index = index
        return best_index

    def _register_written_between(
        self,
        instructions: List[Dict[str, object]],
        reg_name: str,
        start_index: int,
        end_index: int,
    ) -> bool:
        reg_name = str(reg_name or "").lower()
        if not reg_name:
            return True
        for index in range(max(0, start_index), min(len(instructions), end_index)):
            insn = instructions[index]
            base, _predicate = self._arm_base_mnemonic_and_predicate(insn.get("mnemonic", ""))
            if base in {"CMP", "CMN", "TST", "TEQ", "B", "BL", "BLX", "BX", "BXJ"}:
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if not parts:
                continue
            if self._normalize_register_name(parts[0]) == reg_name:
                return True
        return False

    def _desired_loop_exit_branch_direction(
        self,
        loop_body: List[int],
        loop_head: int,
        branch_pc: int,
    ) -> Optional[bool]:
        branch_bb = self.instruction_to_bb.get(int(branch_pc))
        if branch_bb is None:
            return None
        branch_insn = None
        for insn in self.static_bbs.get(int(branch_bb), []):
            if int(insn.get("address", 0) or 0) == int(branch_pc):
                branch_insn = insn
                break
        if branch_insn is None:
            return None
        condition = self._dispatch_condition_for_instruction(branch_insn)
        if condition is None or self._is_switch_dispatch_condition(condition) or self._is_call_dispatch_condition(condition):
            return None
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        if branch_target is None:
            return None
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))
        normalized_target = self.instruction_to_bb.get(int(branch_target), int(branch_target))
        loop_back_taken = int(normalized_target) in loop_targets
        return not loop_back_taken

    def _recent_loop_memory_read_candidates(self, loop_body: List[int]) -> List[Tuple[int, int]]:
        loop_ranges: List[Tuple[int, int]] = []
        for bb_addr in loop_body:
            instructions = self.static_bbs.get(int(bb_addr), [])
            if instructions:
                first = int(instructions[0].get("address", bb_addr) or bb_addr)
                last = instructions[-1]
                end = int(last.get("address", first) or first) + int(last.get("size", 4) or 4)
            else:
                first = int(bb_addr)
                end = int(bb_addr) + 4
            loop_ranges.append((first, end))

        def pc_in_loop(pc: int) -> bool:
            return any(start <= int(pc) < end for start, end in loop_ranges)

        candidates: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for pc, address, is_read, _value in reversed(self.memory_access_history[-512:]):
            if not is_read or not pc_in_loop(int(pc)):
                continue
            key = (int(pc), int(address) & 0xFFFFFFFF)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(key)

        for pc, addresses in getattr(self.loop_classifier, "memory_reads", {}).items():
            if not pc_in_loop(int(pc)):
                continue
            for address in reversed(list(addresses)[-32:]):
                key = (int(pc), int(address) & 0xFFFFFFFF)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(key)
        return candidates

    def _loaded_firmware_image_contains(self, address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        try:
            file_size = os.path.getsize(self.firmware_path)
        except Exception:
            file_size = int(getattr(self.arch_info, "code_size", 0) or 0)
        if file_size > 0:
            for start in (getattr(self, "base_addr", None), getattr(self, "raw_load_base", None)):
                if start is None:
                    continue
                start = int(start) & 0xFFFFFFFF
                if start <= address < start + int(file_size):
                    return True

        for segment in self._load_segments():
            start = int(segment.get("vaddr", 0) or 0)
            end = start + max(int(segment.get("filesz", 0) or 0), int(segment.get("memsz", 0) or 0))
            if start <= address < end:
                return True
        return False

    def _memory_wait_address_allowed(self, address: int) -> bool:
        address = int(address) & 0xFFFFFFFF
        if self._is_mmio_address(address):
            return False
        if self._plausible_stack_pointer(address):
            return False
        if self._loaded_firmware_image_contains(address):
            return False
        if self._is_executable_address(address):
            return False
        if address in self.external_memory_input_addresses:
            return True
        if self._is_memory_mapped(address, 1):
            return True
        if self._page_down(address) in getattr(self, "runtime_written_pages", set()):
            return True
        return False

    def _derive_short_loop_load_constraint(self, loop_head: int) -> Optional[Dict[str, object]]:
        try:
            loop_body = self.loop_classifier._get_loop_body_bbs(loop_head)
        except Exception:
            loop_body = [loop_head]
        loop_body = [int(bb) for bb in (loop_body or [loop_head]) if int(bb) in self.static_bbs]
        if not loop_body or len(loop_body) > 16:
            return None

        instructions: List[Dict[str, object]] = []
        for bb in loop_body:
            instructions.extend(self.static_bbs.get(int(bb), []))
        instructions.sort(key=lambda insn: int(insn.get("address", 0) or 0))
        if not instructions or len(instructions) > 96:
            return None

        loop_set = {int(bb) for bb in loop_body}
        snapshot = self._build_short_loop_analysis_snapshot(loop_head, instructions)
        if snapshot is None:
            return None

        conditional_branches = {
            "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
            "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
            "CBZ", "CBNZ",
        }

        # Prefer later exits: they usually represent the semantic success path
        # after sentinel/early-error checks.
        for branch_index in range(len(instructions) - 1, -1, -1):
            branch_insn = instructions[branch_index]
            branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
            if branch_mnemonic not in conditional_branches:
                continue
            branch_pc = self._parse_int(branch_insn.get("address"))
            if branch_pc is None:
                continue

            target = self._parse_branch_target(branch_insn.get("operands", ""))
            target_bb = self.instruction_to_bb.get(target, target) if target is not None else None
            fallthrough_pc = branch_pc + int(branch_insn.get("size", 2) or 2)
            fallthrough_bb = self.instruction_to_bb.get(fallthrough_pc, fallthrough_pc if fallthrough_pc in self.static_bbs else None)
            target_in_loop = target_bb in loop_set
            fallthrough_in_loop = fallthrough_bb in loop_set
            if target_in_loop == fallthrough_in_loop:
                continue
            desired_taken = bool(not target_in_loop and fallthrough_in_loop)
            if target_in_loop and not fallthrough_in_loop:
                desired_taken = False

            if branch_mnemonic in {"CBZ", "CBNZ"}:
                parts = self._split_operands(branch_insn.get("operands", ""))
                if not parts:
                    continue
                source_reg = self._normalize_register_name(parts[0])
                reg_values, reg_sources = self.code_analyzer._evaluate_snapshot_registers(
                    snapshot,
                    end_index=branch_index - 1,
                )
                source = reg_sources.get(source_reg) if source_reg else None
                desired = self.code_analyzer._infer_cbz_cbnz_value(
                    branch_mnemonic,
                    desired_taken,
                    source_reg,
                    reg_values,
                    int(source.get("width", 4) if source else 4),
                )
                compare_pc = branch_pc
                compare_mnemonic = branch_mnemonic
                compare_parts = parts
                rhs_operand = "#0"
            else:
                compare_index = None
                compare_insn = None
                for index in range(branch_index - 1, max(-1, branch_index - 12), -1):
                    candidate = instructions[index]
                    if self._normalize_mnemonic(candidate.get("mnemonic", "")) in {"CMP", "CMN", "TST", "TEQ"}:
                        compare_index = index
                        compare_insn = candidate
                        break
                if compare_insn is None or compare_index is None:
                    continue
                reg_values, reg_sources = self.code_analyzer._evaluate_snapshot_registers(
                    snapshot,
                    end_index=compare_index - 1,
                )
                compare_mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
                compare_parts = self._split_operands(compare_insn.get("operands", ""))
                if len(compare_parts) < 2:
                    continue
                source_reg, source = self.code_analyzer._select_constraint_source(compare_parts, reg_sources)
                if source is None:
                    continue
                desired = self.code_analyzer._infer_branch_condition_value(
                    branch_mnemonic,
                    compare_mnemonic,
                    source_reg,
                    compare_parts,
                    reg_values,
                    int(source.get("width", 4) or 4),
                    desired_taken,
                )
                compare_pc = self._parse_int(compare_insn.get("address")) or branch_pc
                source_on_left = self._normalize_register_name(compare_parts[0]) == source_reg
                rhs_operand = compare_parts[1] if source_on_left and len(compare_parts) >= 2 else compare_parts[0]

            if source is None or desired is None:
                continue
            source_type = str(source.get("type", "")).lower()
            address = self._parse_int(source.get("address"))
            read_pc = self._parse_int(source.get("read_pc"))
            if address is None or read_pc is None:
                continue
            if source_type != "memory":
                continue
            modeled_async_state = self._is_modeled_async_state_source(
                loop_body,
                int(address),
                int(read_pc),
            )
            if not self._memory_constraint_address_allowed(
                address,
                {
                    "external_memory": address in self.external_memory_input_addresses,
                    "modeled_async_state": modeled_async_state,
                },
            ):
                continue
            width = int(source.get("width", 4) or 4)
            mask = (1 << (max(1, min(4, width)) * 8)) - 1
            dynamic = self._derive_short_loop_dynamic_memory_rule(
                branch_pc,
                branch_mnemonic,
                compare_pc,
                compare_mnemonic,
                rhs_operand,
                instructions[:branch_index],
                width,
            )
            return {
                "branch_pc": branch_pc,
                "desired_branch_taken": bool(desired_taken),
                "dynamic": dynamic,
                "constraint": {
                    "type": "memory",
                    "read_pc": read_pc,
                    "address": int(address) & 0xFFFFFFFF,
                    "value": int(desired) & mask,
                    "constraint_pc": compare_pc,
                    "external_memory": bool(address in self.external_memory_input_addresses),
                    "modeled_async_state": bool(modeled_async_state),
                    "description": (
                        f"Local short-loop exit constraint from branch 0x{branch_pc:08x} "
                        f"({branch_mnemonic} -> {'taken' if desired_taken else 'not taken'})"
                    ),
                },
            }
        return None

    def _is_modeled_async_state_source(self, loop_body: List[int], address: int, read_pc: int) -> bool:
        """
        Allow a dynamic RAM value only when it behaves like an asynchronous
        producer (for example an ISR-updated tick): it is read inside the wait
        loop and was also read before entering the loop.
        """
        address = int(address) & 0xFFFFFFFF
        if (
            self._is_mmio_address(address)
            or self._page_down(address) not in self.runtime_written_pages
            or not self._is_mapped_non_mmio_data_address(address, 1)
        ):
            return False
        loop_pcs = {
            int(insn.get("address", 0) or 0)
            for bb in loop_body
            for insn in self.static_bbs.get(int(bb), [])
        }
        inside_read = False
        outside_read = False
        for pc, values in self.loop_classifier.memory_reads.items():
            if address not in {int(value) & 0xFFFFFFFF for value in values}:
                continue
            if int(pc) == int(read_pc) or int(pc) in loop_pcs:
                inside_read = True
            else:
                outside_read = True
        return inside_read and outside_read

    def _snapshot_region_contains(
        self,
        memory_regions: Dict[Tuple[int, int], bytes],
        address: int,
        size: int = 1,
    ) -> bool:
        address = int(address) & 0xFFFFFFFF
        end = address + max(1, int(size or 1))
        for start, region_size in memory_regions:
            if int(start) <= address and end <= int(start) + int(region_size):
                return True
        return False

    def _add_memory_page_to_analysis_snapshot(
        self,
        memory_regions: Dict[Tuple[int, int], bytes],
        address: int,
    ) -> None:
        address = int(address) & 0xFFFFFFFF
        if self._snapshot_region_contains(memory_regions, address, 4):
            return
        page = self._page_down(address)
        key = (page, 0x1000)
        if key in memory_regions:
            return
        try:
            memory_regions[key] = bytes(self.uc.mem_read(page, 0x1000))
        except Exception:
            return

    def _add_pc_literal_pages_to_analysis_snapshot(
        self,
        memory_regions: Dict[Tuple[int, int], bytes],
        instructions: List[Dict[str, object]],
    ) -> None:
        for insn in instructions or []:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if not mnemonic.startswith("LDR"):
                continue
            parts = self._split_operands(insn.get("operands", ""))
            if len(parts) < 2:
                continue
            literal_addr = self.code_analyzer.resolve_memory_operand_for_instruction(
                insn,
                parts[1],
                {},
            )
            if literal_addr is None:
                continue
            self._add_memory_page_to_analysis_snapshot(memory_regions, literal_addr)

    def _build_short_loop_analysis_snapshot(self, loop_head: int, instructions: List[Dict[str, object]]):
        base = self._build_current_loop_snapshot(loop_head)
        if base is None:
            return None
        memory_regions = dict(getattr(base, "memory_regions", {}) or {})
        self._add_pc_literal_pages_to_analysis_snapshot(memory_regions, instructions)
        instruction_pcs = {
            int(insn.get("address", 0) or 0)
            for insn in instructions
            if int(insn.get("address", 0) or 0)
        }
        candidate_addresses = {
            int(addr) & 0xFFFFFFFF
            for _pc, addr, is_read, _value in self.memory_access_history[-128:]
            if is_read
        }
        for pc in instruction_pcs:
            candidate_addresses.update(
                int(address) & 0xFFFFFFFF
                for address in self.loop_classifier.memory_reads.get(pc, [])
            )
        for address in candidate_addresses:
            if not (
                address in self.external_memory_input_addresses
                or (
                    self._page_down(address) in self.runtime_written_pages
                    and self._is_mapped_non_mmio_data_address(address, 1)
                )
            ):
                continue
            page = address & ~0xFFF
            key = (page, 0x1000)
            if key in memory_regions:
                continue
            try:
                memory_regions[key] = bytes(self.uc.mem_read(page, 0x1000))
            except Exception:
                continue
        return SimpleNamespace(
            bb_address=int(loop_head),
            instruction_count=self.instruction_count,
            cpu_state=self._get_registers(),
            pc=getattr(base, "pc", loop_head),
            flags=getattr(base, "flags", 0),
            memory_regions=memory_regions,
            mmio_values=dict(getattr(base, "mmio_values", {}) or {}),
            mmio_access_history=list(self.mmio_access_history[-32:]),
            bb_instructions=list(instructions),
        )

    def _derive_short_loop_dynamic_memory_rule(
        self,
        branch_pc: int,
        branch_mnemonic: str,
        compare_pc: int,
        compare_mnemonic: str,
        rhs_operand: str,
        instructions_before_branch: List[Dict[str, object]],
        width: int,
    ) -> Optional[Dict[str, object]]:
        immediate = self._parse_immediate_operand(rhs_operand)
        source_reg = None
        source_mask = None
        if immediate is None:
            source_reg, source_mask, immediate = self._trace_rhs_operand_source(
                rhs_operand,
                instructions_before_branch,
            )
        if immediate is None and source_reg is None:
            return None
        return {
            "kind": "self_loop_exit",
            "branch_pc": int(branch_pc),
            "branch_mnemonic": self._normalize_mnemonic(branch_mnemonic),
            "compare_pc": int(compare_pc),
            "compare_mnemonic": self._normalize_mnemonic(compare_mnemonic),
            "rhs_operand": str(rhs_operand or ""),
            "source_reg": source_reg,
            "source_mask": source_mask,
            "immediate": immediate,
            "width": int(width if width in {1, 2, 4} else 4),
        }

    def _flip_fast_forward_eligible(self, loop_check_head: int) -> bool:
        """r15 D1：翻转重放路径的确定性快转资格判据。

        必要条件（缺一不可）：
        1. ``dfs_flip_deterministic_fast_forward`` 打开（仅翻转重放路径）且
           ``enable_loop_intervention`` 关（force-free）；
        2. 环已在分类器成形（``iteration_count >= 1``，至少跑过一轮）；
        3. 环不触 MMIO（``has_mmio_access`` 为假）——外设读写副作用不可
           O(1) 物化，触外设的环逐条真实执行。

        快转族本身（init/copy/byte_copy/count/inc_cmp）另有各自的模式
        匹配（确定退出 + 纯内存写 + PC 只落 fallthrough）。
        """
        if not (
            self.dfs_flip_deterministic_fast_forward
            and not self.enable_loop_intervention
            and self.early_byte_copy_fast_forward
        ):
            return False
        info = self.loop_classifier.loop_heads.get(int(loop_check_head))
        if info is None:
            return False
        if int(getattr(info, "iteration_count", 0) or 0) < 1:
            return False
        if bool(getattr(info, "has_mmio_access", False)):
            return False
        return True

    def _record_loop_fast_forward(
        self, handler_name: str, loop_head: Optional[int] = None
    ) -> bool:
        """r7 口径 2：快进仿真成功路径的单列计数（不改变既有返回语义）。

        r15 D1：``loop_head`` 可选——传入时同时记入逐条事件表
        （``loop_fast_forward_events``，封顶 64 条）供翻转重放证据审计。
        """
        stats = self.loop_fast_forward_emulation
        stats["total"] = int(stats.get("total", 0)) or 0
        stats["total"] += 1
        by_handler = stats.setdefault("by_handler", {})
        by_handler[handler_name] = int(by_handler.get(handler_name, 0) or 0) + 1
        if loop_head is not None:
            if len(self.loop_fast_forward_events) < 64:
                self.loop_fast_forward_events.append(
                    (int(loop_head) & 0xFFFFFFFF, str(handler_name))
                )
        return True

    def _install_runtime_self_loop_branch_force(self, snapshot, loop_head: int, constraint: Dict) -> bool:
        # r7 / R6-D2：runtime_loop_branch_force 属强制转分支，默认关。
        # 需要诊断臂时显式 LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE=1。
        if os.environ.get("LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        instructions = list(getattr(snapshot, "bb_instructions", []) or [])
        if not instructions:
            return False
        branch_insn = instructions[-1]
        condition = self._dispatch_condition_for_instruction(branch_insn)
        if condition is None or self._is_switch_dispatch_condition(condition) or self._is_call_dispatch_condition(condition):
            return False
        branch_pc = self._parse_int(branch_insn.get("address"))
        if branch_pc is None:
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        if branch_target is None:
            return False
        normalized_target = self.instruction_to_bb.get(branch_target, branch_target)
        loop_back_taken = normalized_target == int(loop_head)
        if not loop_back_taken:
            return False
        address = self._parse_int(constraint.get("address"))
        value = self._parse_int(constraint.get("value"))
        self.runtime_loop_branch_forces[int(branch_pc)] = {
            "condition": condition,
            "take_branch": False,
            "branch_pc": int(branch_pc),
            "branch_bb": int(loop_head),
            "loop_head": int(loop_head),
            "mmio_addr": int(address or 0),
            "value": int(value or 0) & 0xFFFFFFFF,
        }
        self.runtime_loop_branch_force_stats["installed"] += 1
        return True

    def _derive_self_loop_dynamic_memory_rule(self, snapshot, constraint: Dict) -> Optional[Dict[str, object]]:
        if constraint.get("type") != "memory":
            return None
        read_pc = self._parse_int(constraint.get("read_pc"))
        if read_pc is None:
            return None
        instructions = list(getattr(snapshot, "bb_instructions", []) or [])
        if not instructions:
            return None

        read_index = None
        for index, insn in enumerate(instructions):
            if int(insn.get("address", 0) or 0) == read_pc:
                read_index = index
                break
        if read_index is None:
            return None

        branch_insn = instructions[-1]
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        branch_pc = int(branch_insn.get("address", 0) or 0)
        compare_index = None
        compare_insn = None
        for index in range(len(instructions) - 2, read_index, -1):
            candidate = instructions[index]
            if self._normalize_mnemonic(candidate.get("mnemonic", "")) in {"CMP", "CMN", "TST", "TEQ"}:
                compare_index = index
                compare_insn = candidate
                break
        if compare_insn is None or compare_index is None:
            return None

        compare_mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
        compare_parts = self._split_operands(compare_insn.get("operands", ""))
        if len(compare_parts) < 2:
            return None
        read_parts = self._split_operands(instructions[read_index].get("operands", ""))
        read_reg = self._normalize_register_name(read_parts[0]) if read_parts else None
        if compare_mnemonic in {"TST", "TEQ"} and read_reg is not None:
            lhs_reg = self._normalize_register_name(compare_parts[0])
            rhs_reg = self._normalize_register_name(compare_parts[1])
            if lhs_reg == read_reg:
                rhs_operand = compare_parts[1]
            elif rhs_reg == read_reg:
                rhs_operand = compare_parts[0]
            else:
                return None
        else:
            rhs_operand = compare_parts[1]
        immediate = self._parse_immediate_operand(rhs_operand)
        source_reg = None
        source_mask = None
        if immediate is None:
            source_reg, source_mask, immediate = self._trace_rhs_operand_source(
                rhs_operand,
                instructions[:compare_index],
            )
        if immediate is None and source_reg is None:
            return None

        read_width = self._constraint_write_size(read_pc)
        return {
            "kind": "self_loop_exit",
            "branch_pc": branch_pc,
            "branch_mnemonic": branch_mnemonic,
            "compare_pc": int(compare_insn.get("address", 0) or 0),
            "compare_mnemonic": compare_mnemonic,
            "rhs_operand": rhs_operand,
            "source_reg": source_reg,
            "source_mask": source_mask,
            "immediate": immediate,
            "width": read_width,
        }

    def _trace_rhs_operand_source(
        self,
        rhs_operand: str,
        instructions: List[Dict[str, object]],
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        target_reg = self._normalize_register_name(rhs_operand)
        if target_reg is None:
            return None, None, self._parse_immediate_operand(rhs_operand)

        constant_regs = self._constant_register_values(instructions)
        if target_reg in constant_regs:
            return None, None, constant_regs[target_reg]

        source_reg = target_reg
        source_mask = None
        immediate = None
        for insn in reversed(instructions):
            parts = self._split_operands(insn.get("operands", ""))
            if not parts:
                continue
            dest = self._normalize_register_name(parts[0])
            if dest != target_reg:
                continue

            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic == "UXTB" and len(parts) >= 2:
                source_reg = self._normalize_register_name(parts[1])
                source_mask = 0xFF
                immediate = None
                break
            if mnemonic == "UXTH" and len(parts) >= 2:
                source_reg = self._normalize_register_name(parts[1])
                source_mask = 0xFFFF
                immediate = None
                break
            if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
                immediate = self._parse_immediate_operand(parts[1])
                source_reg = None if immediate is not None else self._normalize_register_name(parts[1])
                source_mask = None
                break
            if mnemonic in {"AND", "ANDS"} and len(parts) >= 3:
                source_reg = self._normalize_register_name(parts[1])
                source_mask = self._parse_immediate_operand(parts[2])
                immediate = None
                break

        return source_reg, source_mask, immediate

    def _constant_register_values(self, instructions: List[Dict[str, object]]) -> Dict[str, int]:
        """Best-effort constant propagation inside one basic block."""
        constants: Dict[str, int] = {}
        for insn in instructions:
            self._update_constant_register_values(constants, insn)
        return constants

    def _operand_constant_value(self, operand: str, constants: Dict[str, int]) -> Optional[int]:
        immediate = self._parse_immediate_operand(operand)
        if immediate is not None:
            return int(immediate) & 0xFFFFFFFF
        reg = self._normalize_register_name(operand)
        if reg is None:
            return None
        value = constants.get(reg)
        return None if value is None else int(value) & 0xFFFFFFFF

    def _operand_constant_or_runtime_value(
        self,
        operand: str,
        constants: Dict[str, int],
        *,
        excluded_reg: Optional[str] = None,
    ) -> Optional[int]:
        value = self._operand_constant_value(operand, constants)
        if value is not None:
            return value
        reg = self._normalize_register_name(operand)
        if reg is None or (excluded_reg is not None and reg == excluded_reg):
            return None
        return self._read_register_by_name(getattr(self, "uc", None), reg)

    def _update_constant_register_values(
        self,
        constants: Dict[str, int],
        insn: Dict[str, object],
    ) -> None:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
        parts = [part.strip().lower() for part in self._split_operands(insn.get("operands", ""))]
        if not mnemonic or not parts:
            return

        no_write_mnemonics = {
            "CMP", "CMN", "TST", "TEQ", "B", "BEQ", "BNE", "BCS", "BCC",
            "BHS", "BLO", "BMI", "BPL", "BVS", "BVC", "BHI", "BLS", "BGE",
            "BLT", "BGT", "BLE", "CBZ", "CBNZ", "STR", "STRB", "STRH",
            "STM", "STMIA", "PUSH",
        }
        if mnemonic in no_write_mnemonics:
            return

        if mnemonic == "BL":
            for reg in ("r0", "r1", "r2", "r3", "r12", "lr"):
                constants.pop(reg, None)
            return

        dest = self._normalize_register_name(parts[0])
        if dest is None:
            return

        def set_dest(value: Optional[int]) -> None:
            if value is None:
                constants.pop(dest, None)
            else:
                constants[dest] = int(value) & 0xFFFFFFFF

        if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
            set_dest(self._operand_constant_value(parts[1], constants))
            return

        if mnemonic == "MOVT" and len(parts) >= 2:
            upper = self._parse_immediate_operand(parts[1])
            current = constants.get(dest)
            set_dest(((current or 0) & 0xFFFF) | ((int(upper) & 0xFFFF) << 16) if upper is not None and current is not None else None)
            return

        if mnemonic in {"MVN", "MVNS"} and len(parts) >= 2:
            value = self._operand_constant_value(parts[1], constants)
            set_dest((~value) & 0xFFFFFFFF if value is not None else None)
            return

        if mnemonic in {"UXTB", "UXTH", "SXTB", "SXTH"} and len(parts) >= 2:
            value = self._operand_constant_value(parts[1], constants)
            if value is None:
                set_dest(None)
                return
            if mnemonic == "UXTB":
                set_dest(value & 0xFF)
            elif mnemonic == "UXTH":
                set_dest(value & 0xFFFF)
            elif mnemonic == "SXTB":
                byte = value & 0xFF
                set_dest(byte | 0xFFFFFF00 if byte & 0x80 else byte)
            else:
                half = value & 0xFFFF
                set_dest(half | 0xFFFF0000 if half & 0x8000 else half)
            return

        if mnemonic in {"ADD", "ADDS", "SUB", "SUBS", "ORR", "ORRS", "EOR", "EORS", "AND", "ANDS", "BIC", "BICS"}:
            if len(parts) >= 3:
                lhs = self._operand_constant_value(parts[1], constants)
                rhs = self._operand_constant_value(parts[2], constants)
            elif len(parts) >= 2:
                lhs = constants.get(dest)
                rhs = self._operand_constant_value(parts[1], constants)
            else:
                lhs = rhs = None
            if lhs is None or rhs is None:
                set_dest(None)
                return
            if mnemonic in {"ADD", "ADDS"}:
                set_dest(lhs + rhs)
            elif mnemonic in {"SUB", "SUBS"}:
                set_dest(lhs - rhs)
            elif mnemonic in {"ORR", "ORRS"}:
                set_dest(lhs | rhs)
            elif mnemonic in {"EOR", "EORS"}:
                set_dest(lhs ^ rhs)
            elif mnemonic in {"AND", "ANDS"}:
                set_dest(lhs & rhs)
            else:
                set_dest(lhs & (~rhs & 0xFFFFFFFF))
            return

        if mnemonic in {"LSL", "LSLS", "LSR", "LSRS"}:
            if len(parts) >= 3:
                value = self._operand_constant_value(parts[1], constants)
                shift = self._operand_constant_value(parts[2], constants)
            elif len(parts) >= 2:
                value = constants.get(dest)
                shift = self._operand_constant_value(parts[1], constants)
            else:
                value = shift = None
            if value is None or shift is None or not (0 <= int(shift) <= 31):
                set_dest(None)
                return
            if mnemonic in {"LSL", "LSLS"}:
                set_dest(value << int(shift))
            else:
                set_dest(value >> int(shift))
            return

        # Loads and other unmodelled writers make the destination non-constant.
        set_dest(None)

    def _parse_immediate_operand(self, value) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if text.startswith("#"):
            text = text[1:].strip()
        return self._parse_int(text)

    def _build_current_loop_snapshot(self, loop_head: int):
        instructions = self.static_bbs.get(loop_head, [])
        if not instructions:
            return None

        memory_regions = {}
        for start, size in getattr(self.snapshot_manager, "memory_regions", []) or []:
            try:
                memory_regions[(start, size)] = bytes(self.uc.mem_read(start, size))
            except Exception:
                continue
        self._add_pc_literal_pages_to_analysis_snapshot(memory_regions, instructions)

        mmio_values = {}
        for mmio_addr, state in getattr(self.mmio_handler, "mmio_states", {}).items():
            try:
                mmio_values[int(mmio_addr)] = int(state.current_value) & 0xFFFFFFFF
            except Exception:
                continue

        try:
            pc = self.uc.reg_read(UC_ARM_REG_PC)
            flags = self.uc.reg_read(UC_ARM_REG_CPSR)
        except Exception:
            pc = loop_head
            flags = 0

        return SimpleNamespace(
            bb_address=loop_head,
            instruction_count=self.instruction_count,
            cpu_state=self._get_registers(),
            pc=pc,
            flags=flags,
            memory_regions=memory_regions,
            mmio_values=mmio_values,
            mmio_access_history=list(self.mmio_access_history[-32:]),
            bb_instructions=list(instructions),
        )

    def _handle_llm_inference(self, loop_head: int, strategy: Dict):
        """
        使用LLM推断（新策略）

        输入：最近3个唯一BB
        让LLM分析并给出解决方案

        Args:
            loop_head: 循环头地址
            strategy: 干预策略
        """
        logger.info("使用LLM推断...")

        # 获取最近的3个唯一BB快照
        recent_snapshots = self.snapshot_manager.get_recent_snapshots(n=5)
        logger.info("最近快照链: %s", self._format_snapshot_chain(recent_snapshots))

        if len(recent_snapshots) < 3:
            logger.warning("快照数量不足，使用回退策略")
            self._handle_deadlock_rollback(loop_head, strategy)
            return

        # 使用LLM分析
        if strategy.get("use_llm", False):
            self._mark_deadlock_attempt_failed(loop_head)
            analysis = self.code_analyzer.analyze_deadlock(
                recent_snapshots=recent_snapshots,
                current_bb=loop_head,
                loop_count=self.loop_classifier.loop_heads[loop_head].iteration_count,
                branch_snapshots=self.branch_snapshot_manager.snapshots,
                excluded_branch_directions=self.deadlock_failed_directions.get(loop_head),
            )

            logger.info(f"\nLLM分析结果:")
            logger.info(f"  问题类型: {analysis.problem_type}")
            logger.info(f"  置信度: {analysis.confidence:.2f}")
            if analysis.critical_pc:
                logger.info(f"  关键PC: 0x{analysis.critical_pc:08x}")
            if analysis.critical_instruction:
                logger.info(f"  关键指令: {analysis.critical_instruction}")
            if analysis.reason:
                logger.info(f"  原因: {analysis.reason}")
            if analysis.terminal_path:
                logger.info("  路径属性: terminal error path")

            if self._stop_on_terminal_error_path(loop_head, analysis):
                self._register_branch_snapshot_interest_from_analysis(analysis)
                return

            if analysis.confidence >= 0.6 and analysis.need_rollback:
                self._register_branch_snapshot_interest_from_analysis(analysis)
                applied_constraints = self._apply_analysis_constraints(
                    loop_head,
                    analysis,
                    source="llm_inference",
                )
                if applied_constraints > 0:
                    self._remember_deadlock_attempt(loop_head, analysis)
                elif analysis.suggested_constraints:
                    logger.warning(
                        "LLM分析给出的约束全部被本地语义校验拒绝 @ 0x%08x",
                        loop_head,
                    )

                if analysis.restore_branch_bb is not None and self._restore_to_branch_context(analysis.restore_branch_bb):
                    return

                # 回退
                rollback_levels = analysis.rollback_levels
            else:
                # LLM置信度不足，使用默认回退
                rollback_levels = strategy.get("rollback_levels", 2)
        else:
            rollback_levels = strategy.get("rollback_levels", 2)

        self._rollback_to_previous_snapshot(rollback_levels)

    def _handle_deadlock_rollback(self, loop_head: int, strategy: Dict):
        """
        处理死循环：回退并分析

        Args:
            loop_head: 循环头地址
            strategy: 干预策略
        """
        logger.info("检测到死循环，执行回退...")

        # 获取最近的快照
        recent_snapshots = self.snapshot_manager.get_recent_snapshots(n=5)
        logger.info("最近快照链: %s", self._format_snapshot_chain(recent_snapshots))

        if len(recent_snapshots) < 2:
            logger.warning("快照数量不足，无法回退")
            self.uc.emu_stop()
            return

        # 使用LLM分析
        if strategy.get("use_llm", False):
            self._mark_deadlock_attempt_failed(loop_head)
            analysis = self.code_analyzer.analyze_deadlock(
                recent_snapshots=recent_snapshots,
                current_bb=loop_head,
                loop_count=self.loop_classifier.loop_heads[loop_head].iteration_count,
                branch_snapshots=self.branch_snapshot_manager.snapshots,
                excluded_branch_directions=self.deadlock_failed_directions.get(loop_head),
            )

            logger.info(f"\nLLM分析结果:")
            logger.info(f"  问题类型: {analysis.problem_type}")
            logger.info(f"  置信度: {analysis.confidence:.2f}")
            if analysis.critical_pc:
                logger.info(f"  关键PC: 0x{analysis.critical_pc:08x}")
            if analysis.critical_instruction:
                logger.info(f"  关键指令: {analysis.critical_instruction}")
            if analysis.reason:
                logger.info(f"  原因: {analysis.reason}")
            if analysis.terminal_path:
                logger.info("  路径属性: terminal error path")

            if self._stop_on_terminal_error_path(loop_head, analysis):
                self._register_branch_snapshot_interest_from_analysis(analysis)
                return

            if analysis.confidence >= 0.6 and analysis.need_rollback:
                self._register_branch_snapshot_interest_from_analysis(analysis)
                applied_constraints = self._apply_analysis_constraints(
                    loop_head,
                    analysis,
                    source="deadlock_rollback",
                )
                if applied_constraints > 0:
                    self._remember_deadlock_attempt(loop_head, analysis)
                elif analysis.suggested_constraints:
                    logger.warning(
                        "deadlock rollback分析给出的约束全部被本地语义校验拒绝 @ 0x%08x",
                        loop_head,
                    )

                if analysis.restore_branch_bb is not None and self._restore_to_branch_context(analysis.restore_branch_bb):
                    return

                # 回退
                rollback_levels = analysis.rollback_levels
            else:
                # LLM置信度不足，使用默认回退
                rollback_levels = strategy.get("rollback_levels", 2)
        else:
            rollback_levels = strategy.get("rollback_levels", 2)

        self._rollback_to_previous_snapshot(rollback_levels)

    def _rollback_to_previous_snapshot(self, rollback_levels: int) -> bool:
        """执行快照回退并恢复状态，不再让回退失败直接终止仿真"""
        for i in range(rollback_levels):
            target_snapshot = self.snapshot_manager.rollback_one_level()
            if not target_snapshot:
                break

        if self.snapshot_manager.snapshots:
            target_snapshot = self.snapshot_manager.snapshots[-1]
            if not self.snapshot_manager.restore_snapshot(self.uc, target_snapshot):
                logger.warning("拒绝恢复不完整的唯一BB快照 #%s", target_snapshot.snapshot_id)
                return False
            self._restore_snapshot_dynamic_code_provenance(target_snapshot)
            self.snapshot_manager.rearm_from_retained_snapshots()

            for mmio_addr, value in target_snapshot.mmio_values.items():
                if mmio_addr in self.mmio_handler.mmio_states:
                    self.mmio_handler.mmio_states[mmio_addr].current_value = value

            logger.info(f"✓ 回退完成 -> BB 0x{target_snapshot.bb_address:08x}")
            return True
        else:
            logger.warning("没有可用的快照")
            return False

    def _restore_to_branch_context(self, branch_bb: int) -> bool:
        """恢复到可安全重放的分支BB入口，避免因唯一BB快照窗口过短而错过前驱分支。"""
        snapshot = self.branch_snapshot_manager.get_snapshot(branch_bb)
        if snapshot is None:
            logger.warning("缺少分支上下文快照: 0x%08x", branch_bb)
            return False

        if not self.branch_snapshot_manager.restore_snapshot(self.uc, snapshot):
            return False
        if not self.restore_snapshot_external_state(snapshot):
            logger.warning("拒绝恢复外部状态失败的分支上下文 @ 0x%08x", branch_bb)
            return False
        self._restore_snapshot_dynamic_code_provenance(snapshot)

        try:
            restored_pc = (
                (int(branch_bb) | 1)
                if self.execution_thumb
                else (int(branch_bb) & ~1)
            )
            self.uc.reg_write(UC_ARM_REG_PC, restored_pc)
        except Exception as e:
            logger.warning("恢复分支上下文后设置PC失败 @ 0x%08x: %s", branch_bb, e)
            return False

        self.snapshot_manager.reset_history()

        logger.info("✓ 恢复到分支上下文 -> BB 0x%08x", branch_bb)
        return True

    def _restore_snapshot_dynamic_code_provenance(self, snapshot: object) -> None:
        """Restore runtime-written page provenance carried by branch/unique snapshots."""
        try:
            self.runtime_written_pages.update(
                int(page) & ~0xFFF
                for page in (getattr(snapshot, "dirty_pages", set()) or set())
            )
        except Exception:
            pass

    def _mark_deadlock_attempt_failed(self, loop_head: int):
        if not self.semantic_obligation_enabled:
            self.last_deadlock_attempt.pop(loop_head, None)
            return
        attempt = self.last_deadlock_attempt.pop(loop_head, None)
        if attempt is None:
            return
        branch_pc, desired_taken = attempt
        failed = self.deadlock_failed_directions.setdefault(loop_head, {})
        failed.setdefault(branch_pc, set()).add(bool(desired_taken))

    def _remember_deadlock_attempt(self, loop_head: int, analysis):
        if not self.semantic_obligation_enabled:
            self.last_deadlock_attempt.pop(loop_head, None)
            return
        branch_pc = getattr(analysis, "control_branch_pc", None)
        desired_taken = getattr(analysis, "desired_branch_taken", None)
        if branch_pc is None or desired_taken is None:
            self.last_deadlock_attempt.pop(loop_head, None)
            return
        self.last_deadlock_attempt[loop_head] = (int(branch_pc), bool(desired_taken))

    def _register_branch_snapshot_interest_from_analysis(self, analysis):
        if not self.semantic_obligation_enabled:
            return
        if getattr(analysis, "suggested_constraints", None):
            return
        critical_pc = getattr(analysis, "critical_pc", None)
        if critical_pc is None:
            return
        bb_addr = self.branch_pc_to_bb.get(int(critical_pc))
        if bb_addr is None:
            bb_addr = self.instruction_to_bb.get(int(critical_pc))
        if bb_addr is None:
            return
        if not self._get_conditional_branch_info(bb_addr):
            return
        self.branch_snapshot_hotset.add(bb_addr)

    def _stop_on_terminal_error_path(self, loop_head: int, analysis) -> bool:
        """已确认是终止错误路径时，直接结束当前replay，避免在sink上空转。

        r31 P4：本判据与 `terminal_self_loop_bbs` 无关，单独条件化——只有当
        loop_head 是 RTOS 等待/空转自环且确有未决模型事件时才让路；真错误
        处理器（`_unhandled_exception` / `_exit` …）符号不对号，照旧停。
        """
        if self._irq_delivery_defers_terminal_stop(loop_head):
            return False
        if analysis.problem_type != "error_handler":
            return False
        if analysis.suggested_constraints:
            return False
        if not analysis.terminal_path:
            return False
        if float(getattr(analysis, "confidence", 0.0) or 0.0) < 0.7:
            return False

        self.stop_requested_reason = "fatal_sink_terminal"
        logger.info(
            "✓ 终止当前replay: fatal sink @ 0x%08x, 无可恢复约束",
            loop_head,
        )
        self.uc.emu_stop()
        return True

    def _handle_polling_loop(self, loop_head: int, strategy: Dict) -> bool:
        """
        处理轮询循环：优先使用LLM求解器

        Args:
            loop_head: 循环头地址
            strategy: 干预策略
        """
        logger.info("检测到轮询循环，使用LLM求解...")

        # 获取循环特征
        loop_info = self.loop_classifier.loop_heads.get(loop_head)
        if not loop_info:
            return False

        loop_body = self.loop_classifier._get_loop_body_bbs(loop_head)
        # Mapping is cheaper than repeating static inference or an LLM
        # fallback after a block-hook retry. Establish every already-known
        # candidate register page before either solver observes or mutates
        # candidate state.
        preflight_addresses = {
            int(address) & 0xFFFFFFFF
            for address in (loop_info.mmio_addresses_accessed or set())
        }
        for raw_address in strategy.get("mmio_addresses", []) or []:
            parsed_address = self._parse_int(raw_address)
            if parsed_address is not None:
                preflight_addresses.add(int(parsed_address) & 0xFFFFFFFF)
        if preflight_addresses and not self._ensure_memory_ranges_mapped(
            [(address, 4) for address in sorted(preflight_addresses)]
        ):
            logger.warning(
                "轮询求解预检无法映射MMIO页: loop=0x%08x",
                loop_head,
            )
            return False

        inferred_values = self.mmio_inferencer.analyze_polling_loop(
            loop_bbs=loop_body,
            static_bbs=self.static_bbs,
            mmio_accesses=self.mmio_access_history
        )

        # The generic inferencer can over-approximate masked tests, e.g.
        # `and #0x0c; cmp #0x08; bne loop` as `0x0c`.  Prefer the local
        # load/transform/branch chain solver whenever it can recover a
        # concrete raw MMIO byte/word value for the current loop.
        if loop_info.mmio_addresses_accessed:
            inferred_values = dict(inferred_values or {})
            for mmio_addr in sorted(loop_info.mmio_addresses_accessed):
                value = self._infer_local_loop_exit_mmio_value(loop_body, loop_head, mmio_addr)
                if value is not None:
                    inferred_values[int(mmio_addr)] = int(value) & 0xFFFFFFFF

        # LLM只作为保底，避免常见TST/CMP轮询也反复调用大模型。
        if not inferred_values and self.deadlock_solver and loop_info.mmio_addresses_accessed:
            inferred_values = self.deadlock_solver.solve_deadlock(
                loop_head,
                loop_info.iteration_count,
                loop_info.mmio_addresses_accessed,
                loop_bbs=loop_body,
                mmio_access_history=self.mmio_access_history[-512:],
            )

        # 如果还是失败，使用默认值
        if not inferred_values:
            logger.warning("无法推断MMIO值，使用默认策略")
            mmio_addresses = strategy.get("mmio_addresses", [])
            suggested_values = strategy.get("suggested_values", [0x1, 0x80, 0x100])

            inferred_values = {}
            for mmio_addr in mmio_addresses:
                value = suggested_values[0] if suggested_values else 0x1
                inferred_values[mmio_addr] = value

        # Establish all candidate register pages before candidate rotation,
        # branch-force installation, rule learning, or persistence.  A page
        # discovered here can therefore trigger a transparent retry without
        # consuming a candidate or leaving a partially installed repair.
        if inferred_values and not self._ensure_memory_ranges_mapped(
            [(int(mmio_addr), 4) for mmio_addr in inferred_values]
        ):
            logger.warning("轮询修复所需MMIO页无法映射: loop=0x%08x", loop_head)
            return False

        applied_any = False
        rejected_count = 0

        # 应用推断的值
        for mmio_addr, value in inferred_values.items():
            read_pc = self._find_recent_mmio_read_pc(loop_body, mmio_addr)
            constraint_pc = self._find_loop_constraint_pc(loop_body) or loop_head
            locally_inferred = self._infer_local_loop_exit_mmio_value(loop_body, loop_head, mmio_addr)
            value, candidates_exhausted = self._select_polling_candidate_value(
                loop_head,
                int(mmio_addr),
                int(value) & 0xFFFFFFFF,
                locally_verified=locally_inferred is not None,
                suggested_values=strategy.get("suggested_values", []),
            )
            if not self._validate_loop_exit_mmio_value(
                loop_body,
                loop_head,
                mmio_addr,
                value,
                read_pc=read_pc,
                constraint_pc=constraint_pc,
            ):
                logger.warning(
                    "拒绝不满足本地loop-exit语义的MMIO值: addr=0x%08x value=0x%08x read_pc=%s constraint_pc=%s",
                    mmio_addr,
                    value & 0xFFFFFFFF,
                    f"0x{read_pc:08x}" if read_pc is not None else "None",
                    f"0x{constraint_pc:08x}" if constraint_pc is not None else "None",
                )
                self._record_constraint_validation(
                    "polling_loop",
                    loop_head,
                    {
                        "type": "mmio",
                        "read_pc": read_pc,
                        "address": mmio_addr,
                        "value": value,
                        "constraint_pc": constraint_pc,
                    },
                    False,
                    "rejected_by_loop_exit_semantics",
                    control_branch_pc=self._find_loop_branch_pc(loop_body, loop_head),
                )
                rejected_count += 1
                continue
            if locally_inferred is not None and (int(locally_inferred) & 0xFFFFFFFF) == (int(value) & 0xFFFFFFFF):
                applied_any = self._install_runtime_loop_branch_force(loop_body, loop_head, mmio_addr, value) or applied_any
                semantic_mask = self._infer_loop_constraint_mask(
                    loop_body,
                    loop_head,
                    mmio_addr,
                    read_pc=read_pc,
                    constraint_pc=constraint_pc,
                )
                try:
                    self.mmio_handler.learn_validated_polling_rule(
                        mmio_addr=int(mmio_addr),
                        value=int(value) & 0xFFFFFFFF,
                        read_pc=read_pc,
                        constraint_pc=constraint_pc,
                        loop_head=loop_head,
                        mask=semantic_mask,
                        source="local_loop_solver",
                        confidence=1.0,
                    )
                except Exception as exc:
                    logger.debug("通用MMIO语义规则学习失败: %s", exc)
            elif candidates_exhausted:
                logger.warning(
                    "轮询候选已耗尽，安装loop-back分支退出兜底: loop=0x%08x MMIO=0x%08x",
                    loop_head,
                    int(mmio_addr) & 0xFFFFFFFF,
                )
                applied_any = self._install_runtime_loop_branch_force(loop_body, loop_head, mmio_addr, value) or applied_any
            logger.info(f"✓ 设置MMIO 0x{mmio_addr:08x} = 0x{value:08x}")

            # 关键：立即修改Unicorn内存中的值
            memory_written = False
            memory_changed = False
            try:
                # 写入新值到内存
                value_bytes = (value & 0xFFFFFFFF).to_bytes(4, 'little')
                old_bytes = None
                if self._is_memory_mapped(mmio_addr, len(value_bytes)):
                    try:
                        old_bytes = bytes(self.uc.mem_read(mmio_addr, len(value_bytes)))
                    except Exception:
                        old_bytes = None
                self.uc.mem_write(mmio_addr, value_bytes)
                memory_written = True
                memory_changed = old_bytes != value_bytes
                logger.info(f"  ✓ 已写入Unicorn内存")

            except Exception as e:
                logger.error(f"  ❌ 写入Unicorn内存失败: {e}")

            # 添加到MMIO约束（供后续使用）
            before_value = self.mmio_handler.static_constraints.get((read_pc or 0, mmio_addr))
            _external_input_applier = getattr(
                self.mmio_handler, "apply_external_loop_exit_input", None
            )
            if callable(_external_input_applier):
                # r9：外部输入穿透 overlay（固件写镜像会遮蔽主表约束），
                # 否则轮询环读到的仍是旧值、环永远退不出去（A 件根因）。
                _external_input_applier(read_pc or 0, mmio_addr, value)
            else:
                self.mmio_handler.static_constraints[(read_pc or 0, mmio_addr)] = value
            applied_any = applied_any or before_value != value or memory_changed

            # 如果MMIO状态已存在，更新当前值
            if mmio_addr in self.mmio_handler.mmio_states:
                previous_state_value = self.mmio_handler.mmio_states[mmio_addr].current_value
                self.mmio_handler.mmio_states[mmio_addr].current_value = value
                applied_any = applied_any or previous_state_value != value

            # 持久化约束
            if self.constraint_json_path and read_pc is not None and locally_inferred is not None:
                persisted_constraint = {
                    "type": "mmio",
                    "read_pc": read_pc,
                    "address": mmio_addr,
                    "value": value,
                    "constraint_pc": constraint_pc,
                    "description": f"Inferred from polling loop @ 0x{loop_head:08x}"
                }
                read_occurrence = self._current_input_read_occurrence(
                    read_pc,
                    int(mmio_addr),
                )
                if read_occurrence is not None:
                    persisted_constraint["read_occurrence"] = read_occurrence
                self._save_constraint_to_json(persisted_constraint)
            elif self.constraint_json_path:
                logger.debug(
                    "跳过持久化未本地验证的探索MMIO约束: loop=0x%08x addr=0x%08x read_pc=%s value=0x%08x",
                    loop_head,
                    int(mmio_addr) & 0xFFFFFFFF,
                    f"0x{read_pc:08x}" if read_pc is not None else "None",
                    int(value) & 0xFFFFFFFF,
                )

        if applied_any:
            logger.info("✓ MMIO值已调整")
        else:
            logger.warning(
                "轮询循环 0x%08x 未应用有效MMIO约束: candidates=%d rejected=%d",
                loop_head,
                len(inferred_values),
                rejected_count,
            )
        return applied_any

    def _select_polling_candidate_value(
        self,
        loop_head: int,
        mmio_addr: int,
        inferred_value: int,
        *,
        locally_verified: bool,
        suggested_values: List[int],
    ) -> Tuple[int, bool]:
        """Rotate exploratory polling values when LLM/default value does not exit."""
        inferred_value = int(inferred_value) & 0xFFFFFFFF
        if locally_verified:
            return inferred_value, False

        candidates = [
            inferred_value,
            inferred_value ^ 0x1,
            0x00000001,
            0x00000000,
            0xFFFFFFFF,
            0x00000080,
            0x00000100,
            0x0000FFFF,
            0xFFFF0000,
        ]
        for item in suggested_values or []:
            parsed = self._parse_int(item)
            if parsed is not None:
                candidates.append(parsed & 0xFFFFFFFF)

        ordered = []
        seen = set()
        for candidate in candidates:
            candidate = int(candidate) & 0xFFFFFFFF
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)

        key = (int(loop_head), int(mmio_addr) & 0xFFFFFFFF)
        attempted = self.polling_value_attempts.setdefault(key, set())
        for candidate in ordered:
            if candidate not in attempted:
                attempted.add(candidate)
                self.polling_value_exhausted.discard(key)
                if candidate != inferred_value:
                    logger.info(
                        "轮询候选轮换: loop=0x%08x MMIO=0x%08x inferred=0x%08x try=0x%08x",
                        loop_head,
                        mmio_addr,
                        inferred_value,
                        candidate,
                    )
                return candidate, False
        self.polling_value_exhausted.add(key)
        return (ordered[-1] if ordered else inferred_value), True

    def _install_runtime_loop_branch_force(
        self,
        loop_body: List[int],
        loop_head: int,
        mmio_addr: int,
        value: int,
    ) -> bool:
        # r7 / R6-D2：runtime_loop_branch_force 属强制转分支，默认关。
        # 需要诊断臂时显式 LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE=1。
        if os.environ.get("LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        branch_pc = self._find_loop_branch_pc(loop_body, loop_head)
        if branch_pc is None:
            return False
        branch_bb = self.instruction_to_bb.get(branch_pc)
        if branch_bb is None:
            return False
        instructions = self.static_bbs.get(branch_bb, [])
        if not instructions:
            return False
        branch_insn = None
        for insn in instructions:
            if int(insn.get("address", 0) or 0) == int(branch_pc):
                branch_insn = insn
                break
        if branch_insn is None:
            return False
        condition = self._dispatch_condition_for_instruction(branch_insn)
        if condition is None or self._is_switch_dispatch_condition(condition) or self._is_call_dispatch_condition(condition):
            return False
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        if branch_target is None:
            return False
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))
        normalized_target = self.instruction_to_bb.get(branch_target, branch_target)
        loop_back_taken = normalized_target in loop_targets
        take_branch = not loop_back_taken
        self.runtime_loop_branch_forces[int(branch_pc)] = {
            "condition": condition,
            "take_branch": bool(take_branch),
            "branch_pc": int(branch_pc),
            "branch_bb": int(branch_bb),
            "loop_head": int(loop_head),
            "mmio_addr": int(mmio_addr),
            "value": int(value) & 0xFFFFFFFF,
        }
        self.runtime_loop_branch_force_stats["installed"] += 1
        return True

    def _install_runtime_loop_branch_force_at(
        self,
        branch_pc: int,
        take_branch: bool,
        loop_head: int,
        address: int,
        value: int,
    ) -> bool:
        # r7 / R6-D2：runtime_loop_branch_force 属强制转分支，默认关。
        # 需要诊断臂时显式 LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE=1。
        if os.environ.get("LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        branch_pc = int(branch_pc) & 0xFFFFFFFF
        branch_bb = self.instruction_to_bb.get(branch_pc)
        if branch_bb is None:
            return False
        branch_insn = None
        for insn in self.static_bbs.get(int(branch_bb), []):
            if int(insn.get("address", 0) or 0) == branch_pc:
                branch_insn = insn
                break
        if branch_insn is None:
            return False
        condition = self._dispatch_condition_for_instruction(branch_insn)
        if condition is None or self._is_switch_dispatch_condition(condition) or self._is_call_dispatch_condition(condition):
            return False
        self.runtime_loop_branch_forces[branch_pc] = {
            "condition": condition,
            "take_branch": bool(take_branch),
            "branch_pc": int(branch_pc),
            "branch_bb": int(branch_bb),
            "loop_head": int(loop_head),
            "mmio_addr": int(address) & 0xFFFFFFFF,
            "value": int(value) & 0xFFFFFFFF,
        }
        self.runtime_loop_branch_force_stats["installed"] += 1
        return True

    def _find_recent_mmio_read_pc(self, loop_body: List[int], mmio_addr: int) -> Optional[int]:
        """Find the concrete MMIO load PC inside the current loop body."""
        for pc, addr, is_read, _ in reversed(self.mmio_access_history):
            if addr != mmio_addr or not is_read:
                continue
            for bb_addr in loop_body:
                instructions = self.static_bbs.get(bb_addr, [])
                if instructions:
                    first = int(instructions[0].get("address", bb_addr) or bb_addr)
                    last = instructions[-1]
                    bb_end = int(last.get("address", first) or first) + int(last.get("size", 4) or 4)
                else:
                    first = int(bb_addr)
                    bb_end = int(bb_addr) + 4
                if first <= pc < bb_end:
                    return pc
        for pc, addr, is_read, _ in reversed(self.mmio_access_history):
            if addr == mmio_addr and is_read:
                return pc
        latest = self.latest_mmio_read_by_address.get(int(mmio_addr) & 0xFFFFFFFF)
        if latest is not None:
            return int(latest[0])
        return None

    def _find_loop_constraint_pc(self, loop_body: List[int]) -> Optional[int]:
        """Return the compare/test PC that constrains the loop branch, if known."""
        for bb_addr in reversed(loop_body):
            instructions = self.static_bbs.get(bb_addr, [])
            for insn in reversed(instructions):
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
                if mnemonic in {"CMP", "CMN", "TST", "TEQ", "LSLS", "LSL", "LSRS", "LSR"}:
                    return insn.get("address")
        return None

    def _infer_loop_constraint_mask(
        self,
        loop_body: List[int],
        loop_head: int,
        mmio_addr: int,
        *,
        read_pc: Optional[int] = None,
        constraint_pc: Optional[int] = None,
    ) -> Optional[int]:
        """
        Return the bit mask controlled by a validated polling condition.

        This is deliberately conservative: only obvious `TST #mask` and
        `AND #mask; CMP ...` chains are promoted into cross-PC semantic rules.
        """
        candidate_read_pc = self._parse_int(read_pc) or self._find_recent_mmio_read_pc(loop_body, mmio_addr)
        compare_pc = self._parse_int(constraint_pc) or self._find_loop_constraint_pc(loop_body)
        branch_pc = self._find_loop_branch_pc(loop_body, loop_head)
        if candidate_read_pc is None or compare_pc is None or branch_pc is None:
            return None

        flow = self._loop_value_flow_instructions(loop_body, candidate_read_pc, branch_pc)
        if flow is None:
            return None
        _prefix, instructions = flow
        compare_index = None
        compare_insn = None
        for index, insn in enumerate(instructions):
            if int(insn.get("address", 0) or 0) == int(compare_pc):
                compare_index = index
                compare_insn = insn
                break
        if compare_index is None or compare_insn is None:
            return None

        mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
        parts = [part.strip().lower() for part in self._split_operands(compare_insn.get("operands", ""))]
        if mnemonic == "TST" and len(parts) >= 2:
            for operand in parts[1:]:
                immediate = self._parse_immediate_operand(operand)
                if immediate is not None:
                    return int(immediate) & 0xFFFFFFFF
            return None

        if mnemonic not in {"CMP", "CMN", "TEQ"}:
            return None
        compare_regs = {
            self._normalize_register_name(part)
            for part in parts
            if self._normalize_register_name(part) is not None
        }
        for prev in reversed(instructions[max(0, compare_index - 6):compare_index]):
            prev_mnemonic = self._normalize_mnemonic(prev.get("mnemonic", ""))
            if prev_mnemonic not in {"AND", "ANDS"}:
                continue
            prev_parts = [part.strip().lower() for part in self._split_operands(prev.get("operands", ""))]
            if len(prev_parts) < 3:
                continue
            dest_reg = self._normalize_register_name(prev_parts[0])
            if dest_reg is None or dest_reg not in compare_regs:
                continue
            mask = self._parse_immediate_operand(prev_parts[2])
            if mask is not None:
                return int(mask) & 0xFFFFFFFF
        return None

    def _infer_local_loop_exit_mmio_value(
        self,
        loop_body: List[int],
        loop_head: int,
        mmio_addr: int,
        read_pc: Optional[int] = None,
        allow_runtime_operands: bool = True,
    ) -> Optional[int]:
        """Infer common load/test/branch polling values without LLM."""
        read_pc = self._parse_int(read_pc) or self._find_recent_mmio_read_pc(loop_body, mmio_addr)
        branch_pc = self._find_loop_branch_pc(loop_body, loop_head)
        if read_pc is None or branch_pc is None:
            return None

        branch_bb = self.instruction_to_bb.get(branch_pc)
        read_bb = self.instruction_to_bb.get(read_pc)
        if branch_bb is None or read_bb is None:
            return None

        flow = self._loop_value_flow_instructions(loop_body, read_pc, branch_pc)
        if flow is None:
            return None
        loop_prefix_instructions, instructions = flow
        read_index = branch_index = None
        read_insn = branch_insn = None
        for index, insn in enumerate(instructions):
            addr = int(insn.get("address", 0) or 0)
            if addr == read_pc:
                read_index = index
                read_insn = insn
            if addr == branch_pc:
                branch_index = index
                branch_insn = insn
        if read_index is None or branch_index is None or read_insn is None or branch_insn is None:
            return None

        read_parts = self._split_operands(read_insn.get("operands", ""))
        if not read_parts:
            return None
        tracked_reg = read_parts[0].strip().lower()
        prefix_instructions = self._linear_fallthrough_predecessor_instructions(
            int(read_pc),
            int(read_bb),
        )
        constants = self._constant_register_values(
            prefix_instructions + loop_prefix_instructions + instructions[:read_index]
        )
        constants.pop(tracked_reg, None)
        tracked_mask: Optional[int] = None
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))
        loop_back_taken = False
        if branch_target is not None:
            normalized_target = self.instruction_to_bb.get(branch_target, branch_target)
            loop_back_taken = normalized_target in loop_targets
        desired_branch_taken = not loop_back_taken
        conditional_branches = {
            "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
            "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
            "CBZ", "CBNZ",
        }

        def resolve_operand(operand: str, *, excluded_reg: Optional[str] = None) -> Optional[int]:
            if allow_runtime_operands:
                return self._operand_constant_or_runtime_value(
                    operand,
                    constants,
                    excluded_reg=excluded_reg,
                )
            immediate = self._parse_immediate_operand(operand)
            if immediate is not None:
                return int(immediate) & 0xFFFFFFFF
            reg = self._normalize_register_name(operand)
            if reg is None or (excluded_reg is not None and reg == excluded_reg):
                return None
            value = constants.get(reg)
            return None if value is None else int(value) & 0xFFFFFFFF

        def condition_branch_context(condition_index: int) -> Tuple[str, bool]:
            for candidate in instructions[condition_index + 1:branch_index + 1]:
                candidate_mnemonic = self._normalize_mnemonic(candidate.get("mnemonic", ""))
                if candidate_mnemonic not in conditional_branches:
                    continue
                candidate_target = self._parse_branch_target(candidate.get("operands", ""))
                if candidate_target is None:
                    continue
                normalized_candidate_target = self.instruction_to_bb.get(candidate_target, candidate_target)
                candidate_loops = normalized_candidate_target in loop_targets
                return candidate_mnemonic, not candidate_loops
            return branch_mnemonic, desired_branch_taken

        for relative_index, insn in enumerate(instructions[read_index + 1:branch_index], start=read_index + 1):
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            parts = [part.strip().lower() for part in self._split_operands(insn.get("operands", ""))]
            if not parts:
                continue

            if mnemonic in {"MOV", "MOVS"} and len(parts) >= 2 and parts[1] == tracked_reg:
                tracked_reg = parts[0]
                self._update_constant_register_values(constants, insn)
                continue

            if mnemonic in {"UXTB", "UXTH", "SXTB", "SXTH"} and len(parts) >= 2 and parts[1] == tracked_reg:
                tracked_reg = parts[0]
                self._update_constant_register_values(constants, insn)
                continue

            if mnemonic in {"AND", "ANDS", "BIC", "BICS"} and len(parts) >= 3 and parts[1] == tracked_reg:
                mask_value = resolve_operand(parts[2], excluded_reg=tracked_reg)
                if mask_value is None:
                    self._update_constant_register_values(constants, insn)
                    continue
                tracked_reg = parts[0]
                if mnemonic in {"AND", "ANDS"}:
                    tracked_mask = int(mask_value) & 0xFFFFFFFF
                else:
                    tracked_mask = None
                self._update_constant_register_values(constants, insn)
                continue

            if mnemonic in {"LSLS", "LSL"} and len(parts) >= 3 and parts[1] == tracked_reg:
                local_branch_mnemonic, local_desired_branch_taken = condition_branch_context(relative_index)
                shift = self._parse_immediate_operand(parts[2])
                if shift is None or not (0 <= shift <= 31):
                    self._update_constant_register_values(constants, insn)
                    continue
                source_bit = 31 - int(shift)
                if local_branch_mnemonic == "BPL":
                    desired_negative = not local_desired_branch_taken
                    return (1 << source_bit) if desired_negative else 0
                if local_branch_mnemonic == "BMI":
                    desired_negative = local_desired_branch_taken
                    return (1 << source_bit) if desired_negative else 0
                self._update_constant_register_values(constants, insn)

            if mnemonic == "TST" and len(parts) >= 2 and tracked_reg in {parts[0], parts[1]}:
                local_branch_mnemonic, local_desired_branch_taken = condition_branch_context(relative_index)
                other_operand = parts[1] if parts[0] == tracked_reg else parts[0]
                mask = resolve_operand(other_operand, excluded_reg=tracked_reg)
                if mask is None:
                    self._update_constant_register_values(constants, insn)
                    continue
                if local_branch_mnemonic == "BEQ":
                    return 0 if local_desired_branch_taken else int(mask)
                if local_branch_mnemonic == "BNE":
                    return int(mask) if local_desired_branch_taken else 0

            if mnemonic == "CMP" and len(parts) >= 2 and tracked_reg in {parts[0], parts[1]}:
                local_branch_mnemonic, local_desired_branch_taken = condition_branch_context(relative_index)
                other_operand = parts[1] if parts[0] == tracked_reg else parts[0]
                compare_value = resolve_operand(other_operand, excluded_reg=tracked_reg)
                if compare_value is None:
                    self._update_constant_register_values(constants, insn)
                    continue
                if tracked_mask is not None and int(compare_value) == 0:
                    if local_branch_mnemonic == "BEQ":
                        return 0 if local_desired_branch_taken else int(tracked_mask)
                    if local_branch_mnemonic == "BNE":
                        return int(tracked_mask) if local_desired_branch_taken else 0
                if local_branch_mnemonic == "BEQ":
                    return int(compare_value) if local_desired_branch_taken else ((int(compare_value) + 1) & 0xFFFFFFFF)
                if local_branch_mnemonic == "BNE":
                    return ((int(compare_value) + 1) & 0xFFFFFFFF) if local_desired_branch_taken else int(compare_value)

            if (
                self._normalize_register_name(parts[0]) == tracked_reg
                and mnemonic not in {
                    "CMP", "CMN", "TST", "TEQ", "B", "BEQ", "BNE", "BCS", "BCC",
                    "BHS", "BLO", "BMI", "BPL", "BVS", "BVC", "BHI", "BLS",
                    "BGE", "BLT", "BGT", "BLE", "CBZ", "CBNZ", "STR", "STRB",
                    "STRH", "STM", "STMIA", "PUSH",
                }
            ):
                return None
            self._update_constant_register_values(constants, insn)

        return None

    def _loop_body_execution_order(self, loop_body: List[int]) -> List[int]:
        """Recover the current dynamic order of a loop body from recent BB history."""
        loop_set = {int(bb) for bb in loop_body or []}
        if not loop_set:
            return []

        recent = [int(bb) for bb in self.bb_history[-512:] if int(bb) in loop_set]
        ordered_reversed: List[int] = []
        seen: Set[int] = set()
        for bb in reversed(recent):
            if bb in seen:
                if seen >= loop_set:
                    break
                continue
            seen.add(bb)
            ordered_reversed.append(bb)
            if seen >= loop_set:
                break

        ordered = list(reversed(ordered_reversed))
        for bb in loop_body:
            bb = int(bb)
            if bb not in seen:
                seen.add(bb)
                ordered.append(bb)
        return ordered

    def _loop_value_flow_instructions(
        self,
        loop_body: List[int],
        read_pc: int,
        branch_pc: int,
    ) -> Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
        """
        Return (prefix_before_read, read_to_branch_instructions) in dynamic loop
        order. This is needed when a polling read lives in a callee BB and the
        exit branch lives in the caller's continuation BB.
        """
        loop_body_ordered: List[int] = []
        loop_body_set: Set[int] = set()
        for bb in loop_body or []:
            bb = int(bb)
            if bb in loop_body_set:
                continue
            loop_body_set.add(bb)
            loop_body_ordered.append(bb)
        read_bb = self.instruction_to_bb.get(int(read_pc))
        branch_bb = self.instruction_to_bb.get(int(branch_pc))
        if read_bb is not None and int(read_bb) not in loop_body_set:
            loop_body_set.add(int(read_bb))
            loop_body_ordered.append(int(read_bb))
        if branch_bb is not None and int(branch_bb) not in loop_body_set:
            loop_body_set.add(int(branch_bb))
            loop_body_ordered.append(int(branch_bb))
        if not loop_body_set:
            return None

        ordered_bbs = self._loop_body_execution_order(loop_body_ordered)
        flat: List[Dict[str, object]] = []
        for bb in ordered_bbs:
            flat.extend(self.static_bbs.get(int(bb), []))
        if not flat:
            return None

        read_index = None
        branch_index = None
        for index, insn in enumerate(flat):
            address = int(insn.get("address", 0) or 0)
            if address == int(read_pc):
                read_index = index
            if address == int(branch_pc):
                branch_index = index
        if read_index is None or branch_index is None:
            return None

        prefix = list(flat[:read_index])
        if read_index <= branch_index:
            return prefix, list(flat[read_index:branch_index + 1])
        return prefix, list(flat[read_index:] + flat[:branch_index + 1])

    def _linear_fallthrough_predecessor_instructions(
        self,
        read_pc: int,
        current_bb: int,
    ) -> List[Dict[str, object]]:
        """
        Return a unique immediately preceding fallthrough BB, if it is safe to
        treat as a linear prefix for local constant recovery.
        """
        branch_like = {
            "B", "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
            "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE", "BL",
            "BLX", "BX", "BXJ", "CBZ", "CBNZ", "TBB", "TBH", "LDRPC",
        }
        candidates: List[List[Dict[str, object]]] = []
        for bb_addr, bb_instructions in self.static_bbs.items():
            if int(bb_addr) == int(current_bb) or not bb_instructions:
                continue
            last_insn = bb_instructions[-1]
            last_addr = self._parse_int(last_insn.get("address"))
            if last_addr is None:
                continue
            # Thumb instructions are 2 or 4 bytes; require direct fallthrough.
            if not (0 < int(read_pc) - int(last_addr) <= 4):
                continue
            if self._normalize_mnemonic(last_insn.get("mnemonic", "")) in branch_like:
                continue
            candidates.append(list(bb_instructions[-16:]))
        if len(candidates) != 1:
            return []
        return candidates[0]

    def _find_loop_branch_pc(self, loop_body: List[int], loop_head: int) -> Optional[int]:
        """Locate the branch that most likely jumps back into the current loop."""
        loop_targets = {int(bb) for bb in loop_body}
        loop_targets.add(int(loop_head))
        branch_mnemonics = {
            "BEQ", "BNE", "BCS", "BCC", "BHS", "BLO", "BMI", "BPL",
            "BVS", "BVC", "BHI", "BLS", "BGE", "BLT", "BGT", "BLE",
            "CBZ", "CBNZ", "B",
        }
        for bb_addr in reversed(loop_body):
            instructions = self.static_bbs.get(bb_addr, [])
            for insn in reversed(instructions):
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
                if mnemonic not in branch_mnemonics:
                    continue
                target = self._parse_branch_target(insn.get("operands", ""))
                if target is None:
                    continue
                normalized_target = self.instruction_to_bb.get(target, target)
                if normalized_target in loop_targets:
                    return self._parse_int(insn.get("address"))
        return None

    def _validate_loop_exit_mmio_value(
        self,
        loop_body: List[int],
        loop_head: int,
        mmio_addr: int,
        value: int,
        *,
        read_pc: Optional[int] = None,
        constraint_pc: Optional[int] = None,
        allow_runtime_operands: bool = True,
    ) -> bool:
        """
        Guard synthesized MMIO values with local loop-exit semantics when a
        simple load/test/branch chain can be recovered from the current loop.
        """
        candidate_read_pc = self._parse_int(read_pc) or self._find_recent_mmio_read_pc(loop_body, mmio_addr)
        branch_pc = self._find_loop_branch_pc(loop_body, loop_head)
        compare_pc = self._parse_int(constraint_pc) or self._find_loop_constraint_pc(loop_body)
        if candidate_read_pc is None or branch_pc is None or compare_pc is None:
            return True

        # Prefer the same chain-aware local solver used for synthesis.  The
        # older CMP-only heuristic below cannot validate common forms such as
        # `ldr; and #mask; cmp #0; beq loop`, where the raw MMIO value must be
        # the mask rather than `1`.
        locally_inferred = self._infer_local_loop_exit_mmio_value(
            loop_body,
            loop_head,
            mmio_addr,
            read_pc=candidate_read_pc,
            allow_runtime_operands=allow_runtime_operands,
        )
        if locally_inferred is not None:
            write_size = self._constraint_write_size(candidate_read_pc)
            value_mask = (1 << (max(1, min(4, write_size)) * 8)) - 1
            return (int(locally_inferred) & value_mask) == (int(value) & value_mask)

        branch_bb = self.instruction_to_bb.get(branch_pc)
        compare_bb = self.instruction_to_bb.get(compare_pc)
        if branch_bb is None or compare_bb is None or branch_bb != compare_bb:
            return True

        instructions = self.static_bbs.get(branch_bb, [])
        compare_insn = None
        branch_insn = None
        read_index = compare_index = branch_index = None
        for index, insn in enumerate(instructions):
            addr = int(insn.get("address", 0) or 0)
            if addr == candidate_read_pc:
                read_index = index
            if addr == compare_pc:
                compare_index = index
                compare_insn = insn
            if addr == branch_pc:
                branch_index = index
                branch_insn = insn
        if (
            read_index is None
            or compare_index is None
            or branch_index is None
            or compare_insn is None
            or branch_insn is None
            or read_index > compare_index
            or compare_index > branch_index
        ):
            return True

        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        branch_target = self._parse_branch_target(branch_insn.get("operands", ""))
        loop_back_taken = False
        if branch_target is not None:
            normalized_target = self.instruction_to_bb.get(branch_target, branch_target)
            loop_back_taken = normalized_target in set(loop_body) | {int(loop_head)}

        expected_target_direction = not loop_back_taken
        test_str = f"{compare_insn.get('mnemonic', '')} {compare_insn.get('operands', '')}".strip()
        inferred = self.mmio_inferencer._heuristic_infer(
            test_str,
            branch_mnemonic,
            expected_target_direction,
        )
        if inferred is None:
            return True
        return (int(inferred) & 0xFFFFFFFF) == (int(value) & 0xFFFFFFFF)

    def _latest_branch_snapshot(self, branch_bb: int):
        """Return the most recent saved branch snapshot for a BB if available."""
        try:
            history = self.branch_snapshot_manager.get_ordered_snapshot_history()
        except Exception:
            history = []
        for snapshot in reversed(history):
            if int(getattr(snapshot, "address", 0) or 0) == int(branch_bb):
                return snapshot
        return self.branch_snapshot_manager.get_snapshot(branch_bb)

    def _build_validation_snapshot_from_branch(self, branch_bb: int):
        """Adapt a branch snapshot into the shape expected by LLMCodeAnalyzer helpers."""
        snapshot = self._latest_branch_snapshot(branch_bb)
        if snapshot is None:
            return None
        memory_regions = {}
        try:
            raw_regions = getattr(snapshot, "memory_regions", None) or {}
            if raw_regions:
                for (start, size), data in raw_regions.items():
                    memory_regions[(int(start), int(size))] = bytes(data)
            else:
                memory_regions[(int(snapshot.memory_base), int(snapshot.memory_size))] = bytes(snapshot.memory_data)
        except Exception:
            memory_regions = {}
        return SimpleNamespace(
            bb_address=int(branch_bb),
            instruction_count=self.instruction_count,
            cpu_state=dict(getattr(snapshot, "registers", {}) or {}),
            pc=int((getattr(snapshot, "registers", {}) or {}).get("pc", branch_bb)),
            flags=int(getattr(snapshot, "cpsr", 0) or 0),
            memory_regions=memory_regions,
            mmio_values=dict(getattr(snapshot, "mmio_state", {}) or {}),
            mmio_access_history=list(self.mmio_access_history[-32:]),
            bb_instructions=list(self.static_bbs.get(branch_bb, [])),
        )

    def _build_validation_snapshot_from_runtime(self, branch_bb: int):
        """Build a best-effort current snapshot for branch validation."""
        instructions = self.static_bbs.get(branch_bb, [])
        if not instructions:
            return None

        memory_regions = {}
        for start, size in getattr(self.snapshot_manager, "memory_regions", []) or []:
            try:
                memory_regions[(start, size)] = bytes(self.uc.mem_read(start, size))
            except Exception:
                continue

        try:
            pc = self.uc.reg_read(UC_ARM_REG_PC)
            flags = self.uc.reg_read(UC_ARM_REG_CPSR)
        except Exception:
            pc = branch_bb
            flags = 0

        return SimpleNamespace(
            bb_address=int(branch_bb),
            instruction_count=self.instruction_count,
            cpu_state=self._get_registers(),
            pc=pc,
            flags=flags,
            memory_regions=memory_regions,
            mmio_values=self._current_mmio_state(),
            mmio_access_history=list(self.mmio_access_history[-32:]),
            bb_instructions=list(instructions),
        )

    def _branch_validation_snapshot(self, branch_bb: int):
        """Prefer a nearby branch-entry snapshot; fall back to current runtime state only when local."""
        snapshot = self._build_validation_snapshot_from_branch(branch_bb)
        if snapshot is not None:
            return snapshot
        current_pc = None
        try:
            current_pc = self.uc.reg_read(UC_ARM_REG_PC)
        except Exception:
            current_pc = None
        current_bb = self.instruction_to_bb.get(current_pc, current_pc) if current_pc is not None else None
        if current_bb == branch_bb:
            return self._build_validation_snapshot_from_runtime(branch_bb)
        return None

    def _record_constraint_validation(
        self,
        source: str,
        loop_head: int,
        constraint: Dict,
        accepted: Optional[bool],
        reason: str,
        *,
        control_branch_pc: Optional[int] = None,
        desired_branch_taken: Optional[bool] = None,
    ):
        if accepted is True:
            status = "accepted"
        elif accepted is False:
            status = "rejected"
        else:
            status = "skipped"
        self.constraint_validation_stats[status] = self.constraint_validation_stats.get(status, 0) + 1
        self.constraint_validation_history.append({
            "source": source,
            "loop_head": self._format_optional_hex(loop_head),
            "status": status,
            "reason": reason,
            "control_branch_pc": self._format_optional_hex(control_branch_pc),
            "desired_branch_taken": desired_branch_taken,
            "constraint": {
                "type": constraint.get("type"),
                "read_pc": self._format_optional_hex(constraint.get("read_pc")),
                "address": self._format_optional_hex(constraint.get("address")),
                "value": self._format_optional_hex(constraint.get("value")),
                "constraint_pc": self._format_optional_hex(constraint.get("constraint_pc")),
            },
        })
        if len(self.constraint_validation_history) > self.constraint_validation_history_limit:
            overflow = len(self.constraint_validation_history) - self.constraint_validation_history_limit
            if overflow > 0:
                del self.constraint_validation_history[:overflow]

    def _validate_analysis_constraint(
        self,
        constraint: Dict,
        control_branch_pc: Optional[int],
        desired_branch_taken: Optional[bool],
    ) -> Tuple[Optional[bool], str]:
        """
        Validate whether a proposed constraint is consistent with the intended
        branch direction. Return (True/False/None, reason), where None means
        local evidence is insufficient and the caller should allow it.
        """
        branch_pc = self._parse_int(control_branch_pc)
        if branch_pc is None or desired_branch_taken is None:
            return None, "missing control_branch_pc/desired_branch_taken"

        branch_bb = self.branch_pc_to_bb.get(branch_pc)
        if branch_bb is None:
            branch_bb = self.instruction_to_bb.get(branch_pc)
        if branch_bb is None:
            return None, f"cannot resolve branch bb for 0x{branch_pc:08x}"

        instructions = self.static_bbs.get(branch_bb, [])
        if not instructions:
            return None, f"missing static instructions for branch bb 0x{branch_bb:08x}"

        branch_index = None
        branch_insn = None
        for index, insn in enumerate(instructions):
            insn_addr = self._parse_int(insn.get("address"))
            if insn_addr == branch_pc:
                branch_index = index
                branch_insn = insn
                break
        if branch_index is None or branch_insn is None:
            return None, f"cannot find branch instruction @ 0x{branch_pc:08x}"

        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic", ""))
        if branch_mnemonic in {"TBB", "TBH", "LDRPC"}:
            return None, f"switch branch {branch_mnemonic} currently not validated locally"

        snapshot = self._branch_validation_snapshot(branch_bb)
        if snapshot is None:
            return None, f"no validation snapshot for branch bb 0x{branch_bb:08x}"

        reg_values, reg_sources = self.code_analyzer._evaluate_snapshot_registers(
            snapshot,
            end_index=branch_index - 1,
        )
        width = 4
        expected_value = None
        expected_source = None
        compare_insn = None

        if branch_mnemonic in {"CBZ", "CBNZ"}:
            source_reg = self._parse_cbz_register(branch_insn.get("operands", ""))
            expected_source = reg_sources.get(source_reg) if source_reg else None
            if expected_source is None:
                return None, f"cannot recover source for {branch_mnemonic} @ 0x{branch_pc:08x}"
            width = int(expected_source.get("width", 4) or 4)
            expected_value = self.code_analyzer._infer_cbz_cbnz_value(
                branch_mnemonic,
                bool(desired_branch_taken),
                source_reg,
                reg_values,
                width,
            )
        else:
            compare_insn = self.code_analyzer._find_compare_instruction_before(instructions, branch_index)
            if compare_insn is None:
                return None, f"cannot find compare/test before branch @ 0x{branch_pc:08x}"
            compare_mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
            compare_parts = self.code_analyzer._split_operands(compare_insn.get("operands", ""))
            if len(compare_parts) < 2:
                return None, f"incomplete compare operands at 0x{self._parse_int(compare_insn.get('address')) or 0:08x}"
            source_reg, expected_source = self.code_analyzer._select_constraint_source(compare_parts, reg_sources)
            if expected_source is None:
                return None, f"cannot recover compare source for branch @ 0x{branch_pc:08x}"
            width = int(expected_source.get("width", 4) or 4)
            expected_value = self.code_analyzer._infer_branch_condition_value(
                branch_mnemonic,
                compare_mnemonic,
                source_reg,
                compare_parts,
                reg_values,
                width,
                bool(desired_branch_taken),
            )

        if expected_source is None or expected_value is None:
            return None, f"cannot infer expected value for branch @ 0x{branch_pc:08x}"

        expected_mask = (1 << (max(1, min(4, width)) * 8)) - 1
        actual_type = str(constraint.get("type", "")).lower()
        actual_address = self._parse_int(constraint.get("address"))
        actual_read_pc = self._parse_int(constraint.get("read_pc"))
        actual_value = self._parse_int(constraint.get("value"))

        if actual_type != str(expected_source.get("type", "")).lower():
            return False, (
                f"constraint type mismatch: expected {expected_source.get('type')} "
                f"for branch 0x{branch_pc:08x}, got {actual_type}"
            )
        if actual_address != int(expected_source.get("address", -1)):
            return False, (
                f"constraint address mismatch: expected 0x{int(expected_source.get('address', 0)) & 0xFFFFFFFF:08x}, "
                f"got {self._format_optional_hex(actual_address)}"
            )
        expected_read_pc = self._parse_int(expected_source.get("read_pc"))
        if expected_read_pc is not None and actual_read_pc is not None and actual_read_pc != expected_read_pc:
            return False, (
                f"constraint read_pc mismatch: expected 0x{expected_read_pc:08x}, "
                f"got 0x{actual_read_pc:08x}"
            )
        if actual_value is None:
            return False, "constraint value is not a concrete integer"
        if (actual_value & expected_mask) != (int(expected_value) & expected_mask):
            compare_text = ""
            if compare_insn is not None:
                compare_text = (
                    f" compare={compare_insn.get('mnemonic', '')} {compare_insn.get('operands', '')}".strip()
                )
            return False, (
                f"value mismatch for branch 0x{branch_pc:08x}: expected 0x{int(expected_value) & expected_mask:08x}, "
                f"got 0x{actual_value & expected_mask:08x}; branch={branch_mnemonic}{compare_text}"
            )
        return True, (
            f"validated against branch 0x{branch_pc:08x} "
            f"({'taken' if desired_branch_taken else 'not taken'})"
        )

    def _apply_analysis_constraints(self, loop_head: int, analysis, source: str) -> int:
        """Apply analysis-produced constraints only after local semantic validation."""
        applied = 0
        for raw_constraint in getattr(analysis, "suggested_constraints", []) or []:
            normalized = self._normalize_runtime_constraint(raw_constraint)
            if normalized is None:
                reason = "runtime normalization rejected the constraint"
                self._record_constraint_validation(
                    source,
                    loop_head,
                    raw_constraint,
                    False,
                    reason,
                    control_branch_pc=getattr(analysis, "control_branch_pc", None),
                    desired_branch_taken=getattr(analysis, "desired_branch_taken", None),
                )
                logger.warning("拒绝无效分析约束: %s", raw_constraint)
                continue

            if str(normalized.get("type", "")).lower() == "mmio":
                try:
                    loop_body = [
                        int(bb)
                        for bb in self.loop_classifier._get_loop_body_bbs(loop_head)
                        if int(bb) in self.static_bbs
                    ]
                except Exception:
                    loop_body = [int(loop_head)] if int(loop_head) in self.static_bbs else []
                mmio_addr = self._parse_int(normalized.get("address"))
                read_pc = self._parse_int(normalized.get("read_pc"))
                if mmio_addr is not None and loop_body:
                    local_value = self._infer_local_loop_exit_mmio_value(
                        loop_body,
                        loop_head,
                        int(mmio_addr),
                        read_pc=read_pc,
                    )
                    if local_value is not None:
                        write_size = self._constraint_write_size(read_pc)
                        mask = (1 << (max(1, min(4, write_size)) * 8)) - 1
                        if (int(normalized.get("value", 0)) & mask) != (int(local_value) & mask):
                            logger.warning(
                                "修正分析约束以匹配本地loop-exit语义: loop=0x%08x addr=0x%08x old=0x%08x new=0x%08x",
                                loop_head,
                                int(mmio_addr) & 0xFFFFFFFF,
                                int(normalized.get("value", 0)) & 0xFFFFFFFF,
                                int(local_value) & 0xFFFFFFFF,
                            )
                            normalized["value"] = int(local_value) & 0xFFFFFFFF
                            normalized["description"] = (
                                f"Corrected by local loop-exit semantics @ 0x{loop_head:08x}; "
                                + str(normalized.get("description", ""))
                            )

            accepted, reason = self._validate_analysis_constraint(
                normalized,
                getattr(analysis, "control_branch_pc", None),
                getattr(analysis, "desired_branch_taken", None),
            )
            self._record_constraint_validation(
                source,
                loop_head,
                normalized,
                accepted,
                reason,
                control_branch_pc=getattr(analysis, "control_branch_pc", None),
                desired_branch_taken=getattr(analysis, "desired_branch_taken", None),
            )
            if accepted is False:
                logger.warning("拒绝不满足本地branch语义的约束: %s (%s)", normalized, reason)
                continue
            if accepted is None:
                logger.debug("约束缺少足够本地证据，宽松放行: %s (%s)", normalized, reason)
            self._apply_constraint(normalized)
            applied += 1
        return applied

    def _is_memory_mapped(self, address: int, size: int = 1) -> bool:
        """Return True if the full address range is already mapped in Unicorn."""
        try:
            cursor = int(address)
            end_addr = int(address) + max(1, int(size))
            # Most calls address ELF/RAM ranges established during setup.  The
            # Python-side index avoids a costly ``mem_regions`` round trip on
            # every mapped MMIO/SRAM write, while the native query below
            # remains the source of truth for newly discovered ranges.
            if self._contains_mapped_range(cursor, max(1, int(size))):
                return True
            regions = sorted((int(start), int(end) + 1) for start, end, _perms in self.uc.mem_regions())
            for start, end in regions:
                if end <= cursor:
                    continue
                if start > cursor:
                    return False
                cursor = max(cursor, end)
                if cursor >= end_addr:
                    return True
        except Exception:
            pass
        return False

    def _execution_preflight(self, start_pc: int) -> Tuple[bool, Dict[str, object]]:
        """Validate native-engine invariants immediately before ``emu_start``."""
        record: Dict[str, object] = {
            "schema": "lsgemu.execution_preflight.v1",
            "start_pc": f"0x{int(start_pc) & 0xFFFFFFFF:08x}",
            "expected_thumb": bool(int(start_pc) & 1),
            "corrections": [],
        }
        uc = getattr(self, "uc", None)
        if uc is None:
            record.update({"ok": False, "reason": "closed_unicorn_instance"})
            self.execution_preflight_stats["closed_unicorn_instance"] += 1
            self.last_execution_preflight = record
            return False, record

        primary = get_primary_mmio_handler(uc)
        owned_handler = getattr(self, "mmio_handler", None)
        if primary is not None and owned_handler is not None and primary is not owned_handler:
            record.update({"ok": False, "reason": "primary_mmio_owner_mismatch"})
            self.execution_preflight_stats["primary_mmio_owner_mismatch"] += 1
            self.last_execution_preflight = record
            return False, record

        target_pc = int(start_pc) & ~1
        instruction_size = 2 if bool(int(start_pc) & 1) else 4
        if not self._is_memory_mapped(target_pc, instruction_size):
            record.update({
                "ok": False,
                "reason": "entry_pc_unmapped",
                "target_pc": f"0x{target_pc & 0xFFFFFFFF:08x}",
            })
            self.execution_preflight_stats["entry_pc_unmapped"] += 1
            self.last_execution_preflight = record
            return False, record
        try:
            uc.mem_read(target_pc, instruction_size)
        except Exception as exc:
            record.update({"ok": False, "reason": "entry_pc_unreadable", "error": str(exc)})
            self.execution_preflight_stats["entry_pc_unreadable"] += 1
            self.last_execution_preflight = record
            return False, record

        try:
            sp = int(uc.reg_read(UC_ARM_REG_SP)) & 0xFFFFFFFF
        except Exception as exc:
            record.update({"ok": False, "reason": "stack_pointer_unreadable", "error": str(exc)})
            self.execution_preflight_stats["stack_pointer_unreadable"] += 1
            self.last_execution_preflight = record
            return False, record
        record["sp"] = f"0x{sp:08x}"
        stack_probe = (sp - 4) & 0xFFFFFFFF if sp >= 4 else sp
        if sp == 0 or not self._is_memory_mapped(stack_probe, 4):
            record.update({
                "ok": False,
                "reason": "stack_pointer_unmapped",
                "stack_probe": f"0x{stack_probe:08x}",
            })
            self.execution_preflight_stats["stack_pointer_unmapped"] += 1
            self.last_execution_preflight = record
            return False, record

        try:
            cpsr = int(uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFF
        except Exception as exc:
            record.update({"ok": False, "reason": "cpsr_unreadable", "error": str(exc)})
            self.execution_preflight_stats["cpsr_unreadable"] += 1
            self.last_execution_preflight = record
            return False, record
        expected_thumb = bool(int(start_pc) & 1)
        actual_thumb = bool(cpsr & 0x20)
        record["cpsr_before"] = f"0x{cpsr:08x}"
        if actual_thumb != expected_thumb:
            corrected = (cpsr | 0x20) if expected_thumb else (cpsr & ~0x20)
            try:
                uc.reg_write(UC_ARM_REG_CPSR, corrected)
            except Exception as exc:
                record.update({"ok": False, "reason": "thumb_state_uncorrectable", "error": str(exc)})
                self.execution_preflight_stats["thumb_state_uncorrectable"] += 1
                self.last_execution_preflight = record
                return False, record
            record["corrections"].append("cpsr_thumb_bit")
            record["cpsr_after"] = f"0x{corrected & 0xFFFFFFFF:08x}"
            self.execution_preflight_stats["thumb_state_corrected"] += 1

        record.update({"ok": True, "reason": "ok"})
        self.execution_preflight_stats["passed"] += 1
        self.last_execution_preflight = record
        return True, record

    def _ensure_memory_ranges_mapped(
        self,
        ranges: List[Tuple[int, int]],
        *,
        permissions: Optional[int] = None,
        origin: str = "unknown",
        retry_block_callback: bool = False,
        verify_native: bool = False,
    ) -> bool:
        """Ensure several ranges are mapped without mutating native topology.

        When called from a Unicorn callback, all missing pages are collected in
        one request and a private control-flow signal aborts the current native
        slice.  The owner retries the slice after applying the batch at depth
        zero.  This also makes multi-buffer summaries atomic with respect to
        mapping: no partial synthetic write is committed before a later page
        is discovered missing.
        """
        page_size = 0x1000
        missing: Set[int] = set()
        max_pages = 65536
        try:
            max_pages = max(
                1,
                int(os.environ.get("LSGEMU_MEMORY_MAP_MAX_REQUEST_PAGES", "65536")),
            )
        except ValueError:
            pass

        for raw_address, raw_size in ranges or []:
            try:
                address = int(raw_address)
                size = max(1, int(raw_size))
            except (TypeError, ValueError, OverflowError):
                return False
            if address < 0 or address > 0xFFFFFFFF:
                return False
            end_address = address + size
            if end_address <= address or end_address > 0x100000000:
                return False
            start_page = address & ~(page_size - 1)
            end_page = (end_address - 1) & ~(page_size - 1)
            page = start_page
            while page <= end_page:
                page_is_mapped = (
                    self._native_range_is_mapped(page, 1)
                    if verify_native
                    else self._is_memory_mapped(page, 1)
                )
                if not page_is_mapped:
                    missing.add(page)
                    if len(missing) > max_pages:
                        self.memory_mapping_stats["request_too_large"] += 1
                        return False
                page += page_size

        if not missing:
            return all(
                (
                    self._native_range_is_mapped(
                        int(address),
                        max(1, int(size)),
                    )
                    if verify_native
                    else self._is_memory_mapped(
                        int(address),
                        max(1, int(size)),
                    )
                )
                for address, size in (ranges or [])
            )

        with self._native_state_lock():
            native_depth = int(getattr(self, "_native_emulation_depth", 0) or 0)
            owner_thread_id = getattr(self, "_native_emulation_thread_id", None)
        if native_depth > 0:
            if owner_thread_id != threading.get_ident():
                self.memory_mapping_stats["cross_thread_requests_rejected"] += 1
                raise RuntimeError(
                    "memory mapping requested by a non-owner thread during emulation"
                )
            self._queue_deferred_memory_mapping(
                missing,
                permissions=permissions,
                origin=origin,
                retry_block_callback=retry_block_callback,
            )

        if not self._map_memory_pages_now(
            {page: int(permissions) if permissions is not None else 0 for page in missing}
        ):
            return False
        return all(
            (
                self._native_range_is_mapped(
                    int(address),
                    max(1, int(size)),
                )
                if verify_native
                else self._is_memory_mapped(
                    int(address),
                    max(1, int(size)),
                )
            )
            for address, size in (ranges or [])
        )

    def _ensure_memory_mapped(
        self,
        address: int,
        size: int = 4,
        *,
        origin: str = "unknown",
        retry_block_callback: bool = False,
        verify_native: bool = False,
    ) -> bool:
        """Map missing pages before writing synthesized constraints."""
        return self._ensure_memory_ranges_mapped(
            [(int(address), int(size))],
            origin=origin,
            retry_block_callback=retry_block_callback,
            verify_native=verify_native,
        )

    def _managed_mem_map(
        self,
        address: int,
        size: int,
        *,
        perms: Optional[int] = None,
    ) -> bool:
        """Compatibility adapter used by auxiliary Unicorn components."""
        return self._ensure_memory_ranges_mapped(
            [(int(address), int(size))],
            permissions=perms,
            origin="component",
        )

    def _managed_mem_unmap(self, address: int, size: int) -> None:
        """Unmap only after native execution has fully unwound."""
        with self._native_state_lock():
            if int(getattr(self, "_native_emulation_depth", 0) or 0) > 0:
                self.memory_mapping_stats["unsafe_unmap_rejected"] += 1
                raise RuntimeError(
                    "cannot unmap memory during native Unicorn execution"
                )
        self.uc.mem_unmap(int(address), int(size))
        self._forget_mapped_range(int(address), int(size))
        self.memory_mapping_stats["safe_unmaps"] += 1

    def _memory_constraint_crosses_call_boundary(
        self,
        read_pc: Optional[int],
        constraint_pc: Optional[int],
    ) -> bool:
        """Reject local memory dependencies that cross a call/return boundary."""
        read_pc = self._parse_int(read_pc)
        constraint_pc = self._parse_int(constraint_pc)
        if read_pc is None or constraint_pc is None or read_pc == constraint_pc:
            return False
        lo = min(int(read_pc), int(constraint_pc))
        hi = max(int(read_pc), int(constraint_pc))
        if hi - lo > 0x400:
            return False

        for pc in sorted(addr for addr in self.code_analyzer.instruction_index if lo < int(addr) < hi):
            insn = self.code_analyzer.instruction_index.get(pc) or {}
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic", ""))
            if mnemonic in {"BL", "BLX", "BX", "BXJ"}:
                return True
            if self._is_ldr_pc_dispatch_instruction(insn):
                return True
            parts = [part.strip().lower() for part in self._split_operands(insn.get("operands", ""))]
            if mnemonic in {"POP", "LDM", "LDMIA"} and any(part in {"pc", "{pc}"} or "pc" in part for part in parts):
                return True
        return False

    def _is_plausible_pointer_value(self, value: int) -> bool:
        value = int(value) & 0xFFFFFFFF
        if value in {0, 0xFFFFFFFF}:
            return False
        if self._is_executable_address(value):
            return True
        if self._loaded_firmware_image_contains(value):
            return True
        if self._is_mapped_non_mmio_data_address(value, 1):
            return True
        return 0x20000000 <= value < 0x40000000

    def _read_concrete_u32(self, address: int) -> Optional[int]:
        try:
            if not self._is_memory_mapped(int(address), 4):
                return None
            raw = bytes(self.uc.mem_read(int(address) & 0xFFFFFFFF, 4))
            return int.from_bytes(raw, "little") & 0xFFFFFFFF
        except Exception:
            return None

    def _memory_constraint_safety_reason(self, constraint: Dict[str, object]) -> Optional[str]:
        """Return a rejection reason for memory constraints that would corrupt concrete state."""
        if str((constraint or {}).get("type", "")).lower() != "memory":
            return None
        read_pc = self._parse_int(constraint.get("read_pc"))
        constraint_pc = self._parse_int(constraint.get("constraint_pc"))
        address = self._parse_int(constraint.get("address"))
        value = self._parse_int(constraint.get("value"))
        if address is None or value is None:
            return "missing_address_or_value"

        if self._memory_constraint_crosses_call_boundary(read_pc, constraint_pc):
            return "call_boundary_between_read_and_constraint"

        width = self._constraint_write_size(read_pc)
        if width != 4:
            return None
        if bool(constraint.get("external_memory")):
            return None

        old_value = self._read_concrete_u32(int(address))
        if old_value is None:
            return None
        if self._is_plausible_pointer_value(old_value) and not self._is_plausible_pointer_value(int(value)):
            return "would_overwrite_plausible_pointer_field"
        return None

    def _record_memory_constraint_guard(
        self,
        source: str,
        read_pc: Optional[int],
        address: int,
        value: int,
        reason: str,
    ) -> None:
        key = "rejected_other"
        if "call_boundary" in str(reason):
            key = "rejected_call_boundary"
        elif "pointer" in str(reason):
            key = "rejected_pointer_field"
        self.memory_constraint_guard_stats[key] = int(self.memory_constraint_guard_stats.get(key, 0) or 0) + 1
        event = {
            "source": str(source or "unknown"),
            "read_pc": self._format_optional_hex(read_pc),
            "address": self._format_optional_hex(address),
            "value": self._format_optional_hex(value),
            "reason": str(reason),
        }
        self.memory_constraint_guard_history.append(event)
        if len(self.memory_constraint_guard_history) > 256:
            del self.memory_constraint_guard_history[: len(self.memory_constraint_guard_history) - 256]
        logger.warning(
            "拒绝危险memory约束: source=%s read_pc=%s address=0x%08x value=0x%08x reason=%s",
            source,
            self._format_optional_hex(read_pc),
            int(address) & 0xFFFFFFFF,
            int(value) & 0xFFFFFFFF,
            reason,
        )

    def _apply_constraint(self, constraint: Dict):
        """应用约束"""
        requested_occurrence = self._parse_int(
            constraint.get("read_occurrence")
            if isinstance(constraint, dict)
            else None
        )
        normalized = self._normalize_runtime_constraint(constraint)
        if normalized is None:
            logger.warning(f"跳过无效约束: {constraint}")
            return
        if requested_occurrence is not None and requested_occurrence > 0:
            normalized["read_occurrence"] = int(requested_occurrence)

        constraint_type = normalized.get("type")
        address = normalized.get("address")
        value = normalized.get("value")
        description = normalized.get("description", "")

        if address is None or value is None:
            return
        value &= 0xFFFFFFFF

        read_pc = self._parse_int(normalized.get("read_pc"))
        if constraint_type == "mmio" and read_pc is None:
            read_pc = 0
        write_size = self._constraint_write_size(read_pc)
        value_mask = (1 << (write_size * 8)) - 1
        value_bytes = (value & value_mask).to_bytes(write_size, 'little')

        # Mapping is part of the transaction precondition.  Do it before any
        # constraint dictionary, modeled async marker, or persistent artifact
        # is updated, so a deferred callback retry cannot expose half-applied
        # state to subsequent candidate generation.
        if not self._ensure_memory_mapped(address, len(value_bytes)):
            logger.warning(
                "跳过无法映射的%s约束: 0x%08x = 0x%08x (%s)",
                constraint_type,
                int(address) & 0xFFFFFFFF,
                int(value) & 0xFFFFFFFF,
                description,
            )
            return

        if constraint_type == "memory":
            if read_pc is not None:
                self.memory_read_constraints[(read_pc, address)] = value
                if bool(normalized.get("modeled_async_state")):
                    key = (int(read_pc), int(address) & 0xFFFFFFFF)
                    if key not in self.modeled_async_memory_constraints:
                        self.modeled_async_memory_constraints.add(key)
                        self.modeled_async_memory_stats["installed"] += 1
            try:
                self.uc.mem_write(address, value_bytes)
                logger.info(f"✓ 应用内存约束: 0x{address:08x} = 0x{value:08x}")
                logger.info(f"  理由: {description}")
            except Exception as e:
                logger.error(f"❌ 应用内存约束失败: {e}")

        elif constraint_type == "mmio":
            _external_input_applier = getattr(
                self.mmio_handler, "apply_external_loop_exit_input", None
            )
            if callable(_external_input_applier):
                # r9：同轮询处理——外部输入约束须穿透 overlay 才能被读到。
                _external_input_applier(read_pc or 0, address, value)
            else:
                self.mmio_handler.static_constraints[(read_pc, address)] = value
            try:
                self.uc.mem_write(address, value_bytes)
            except Exception as e:
                logger.debug(f"写入MMIO约束到Unicorn内存失败: {e}")
            logger.info(f"✓ 应用MMIO约束: 0x{address:08x} = 0x{value:08x}")
            logger.info(f"  理由: {description}")

        # 持久化
        constraint = {
            **normalized,
            "address": address,
            "value": value,
        }
        dynamic_rule = (
            self._serialize_dynamic_memory_rule(normalized.get("dynamic"))
            if self.enable_dynamic_memory_constraints
            else None
        )
        if constraint_type == "memory" and dynamic_rule is not None:
            read_pc = self._parse_int(normalized.get("read_pc"))
            if read_pc is not None:
                self.dynamic_memory_read_constraints[(read_pc, address)] = dynamic_rule
                constraint["dynamic"] = dynamic_rule
        if self.constraint_json_path:
            self._save_constraint_to_json(constraint)

    def _constraint_write_size(self, read_pc: Optional[int]) -> int:
        if read_pc is None:
            return 4
        insn = self.code_analyzer.instruction_index.get(int(read_pc))
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic", "")) if insn else ""
        if mnemonic.startswith("LDRB"):
            return 1
        if mnemonic.startswith("LDRH"):
            return 2
        return 4

    def _normalize_runtime_constraint(self, constraint: Dict) -> Optional[Dict]:
        """运行时约束校验，避免错误LLM结果污染后续重放。"""
        if not isinstance(constraint, dict):
            return None

        constraint_type = str(constraint.get("type", "")).lower()
        if constraint_type == "mmio":
            return self.code_analyzer.normalize_constraint(constraint)

        if constraint_type != "memory":
            return None

        address = self._parse_int(constraint.get("address"))
        value = self._parse_int(constraint.get("value"))
        if address is None or value is None:
            return None
        if not self._memory_constraint_address_allowed(address, constraint):
            logger.debug("拒绝不可变/非外部输入地址的memory约束: %s", constraint)
            return None

        normalized = {
            "type": "memory",
            "address": address & 0xFFFFFFFF,
            "value": value & 0xFFFFFFFF,
            "description": constraint.get("description", ""),
            "read_pc": self._parse_int(constraint.get("read_pc")),
            "constraint_pc": self._parse_int(constraint.get("constraint_pc")),
        }
        if bool(constraint.get("external_memory")):
            normalized["external_memory"] = True
        if bool(constraint.get("modeled_async_state")):
            normalized["modeled_async_state"] = True
        dynamic_rule = (
            self._normalize_dynamic_memory_rule(constraint.get("dynamic"))
            if self.enable_dynamic_memory_constraints
            else None
        )
        if dynamic_rule is not None:
            normalized["dynamic"] = dynamic_rule
        if normalized["read_pc"] is not None:
            insn = self.code_analyzer.instruction_index.get(normalized["read_pc"])
            mnemonic = str(insn.get("mnemonic", "")).upper() if insn else ""
            if not mnemonic.startswith("LDR"):
                logger.debug("拒绝read_pc不是load的memory约束: %s", constraint)
                return None
        guard_reason = self._memory_constraint_safety_reason(normalized)
        if guard_reason is not None:
            self._record_memory_constraint_guard(
                "normalize",
                normalized.get("read_pc"),
                normalized["address"],
                normalized["value"],
                guard_reason,
            )
            return None
        return normalized

    def _memory_constraint_address_allowed(self, address: int, constraint: Optional[Dict] = None) -> bool:
        address = int(address) & 0xFFFFFFFF
        runtime_written_pages = getattr(self, "runtime_written_pages", set())
        if (
            self._plausible_stack_pointer(address)
            or self._is_mmio_address(address)
            or 0x20000000 <= address < 0x40000000
        ):
            return True
        if (
            isinstance(constraint, dict)
            and bool(constraint.get("modeled_async_state"))
            and self._page_down(address) in runtime_written_pages
            and self._is_mapped_non_mmio_data_address(address, 1)
        ):
            return True
        if (
            isinstance(constraint, dict)
            and bool(constraint.get("modeled_async_state"))
            and bool(constraint.get("persisted_modeled_async_state"))
            and constraint.get("read_pc") is not None
            and not self._is_mmio_address(address)
            and not self._is_executable_address(address)
        ):
            return True
        if not self.allow_external_memory_constraints:
            return False
        if address in self.external_memory_input_addresses:
            return True
        if isinstance(constraint, dict) and bool(constraint.get("external_memory")):
            self.external_memory_input_addresses.add(address)
            return True
        return False

    def _format_snapshot_chain(self, snapshots: List[object]) -> str:
        if not snapshots:
            return "[]"
        return " -> ".join(f"0x{snapshot.bb_address:08x}" for snapshot in reversed(list(snapshots)))

    def _save_constraint_to_json(self, constraint: Dict):
        """保存约束到JSON"""
        try:
            if os.path.exists(self.constraint_json_path):
                with open(self.constraint_json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {"constraints": []}

            new_item = {
                "type": constraint.get("type"),
                "read_pc": self._format_optional_hex(constraint.get("read_pc")),
                "address": f"0x{constraint.get('address'):08x}",
                "value": f"0x{constraint.get('value'):08x}",
                "constraint_pc": self._format_optional_hex(constraint.get("constraint_pc")),
                "description": constraint.get("description", ""),
                "added_by": "intelligent_emulator",
                "iteration": self.iteration
            }
            read_occurrence = self._parse_int(constraint.get("read_occurrence"))
            if read_occurrence is not None and read_occurrence > 0:
                new_item["read_occurrence"] = int(read_occurrence)
            dynamic_rule = (
                self._serialize_dynamic_memory_rule(constraint.get("dynamic"))
                if self.enable_dynamic_memory_constraints
                else None
            )
            if dynamic_rule is not None:
                new_item["dynamic"] = dynamic_rule
            if bool(constraint.get("external_memory")):
                new_item["external_memory"] = True
            if bool(constraint.get("modeled_async_state")):
                new_item["modeled_async_state"] = True

            rewritten_constraints = []
            updated = False
            for item in data["constraints"]:
                same_read_site = (
                    item.get("type") == new_item["type"]
                    and item.get("address") == new_item["address"]
                    and (item.get("read_pc") or item.get("pc")) == new_item["read_pc"]
                    and int(self._parse_int(item.get("read_occurrence")) or 0)
                    == int(new_item.get("read_occurrence") or 0)
                )
                legacy_broad_loop_constraint = bool(
                    new_item.get("read_occurrence")
                    and item.get("type") == new_item["type"]
                    and item.get("address") == new_item["address"]
                    and (item.get("read_pc") or item.get("pc")) == new_item["read_pc"]
                    and self._parse_int(item.get("read_occurrence")) is None
                    and str(item.get("added_by") or "") == "intelligent_emulator"
                    and "loop" in str(item.get("description") or "").lower()
                )
                if legacy_broad_loop_constraint:
                    continue
                if same_read_site:
                    if not updated:
                        rewritten_constraints.append(new_item)
                        updated = True
                    continue
                rewritten_constraints.append(item)
            if not updated:
                rewritten_constraints.append(new_item)
            data["constraints"] = rewritten_constraints

            atomic_json_dump(data, self.constraint_json_path, indent=2)

            logger.info(f"✓ 约束已保存到: {self.constraint_json_path}")

        except Exception as e:
            logger.error(f"❌ 保存约束失败: {e}")

    def _format_optional_hex(self, value):
        parsed = self._parse_int(value)
        if parsed is None:
            return None
        return f"0x{parsed:08x}"

    def _record_unmapped_access(
        self,
        access: int,
        address: int,
        size: int,
        value: int,
        pc: int,
    ) -> None:
        """Record the exact fault-time unmapped access for crash triage."""
        access_name = {
            UC_MEM_READ_UNMAPPED: "read_unmapped",
            UC_MEM_WRITE_UNMAPPED: "write_unmapped",
        }.get(access, str(access))
        self.last_unmapped_access = {
            "access": access_name,
            "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
            "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
            "size": int(size or 0),
            "value": f"0x{int(value or 0) & 0xFFFFFFFF:08x}",
            "registers": {
                name: f"0x{int(reg_value) & 0xFFFFFFFF:08x}"
                for name, reg_value in self._get_registers().items()
            },
        }

    def mmio_unmapped_hook(self, uc, access, address, size, value, user_data):
        """
        MMIO访问Hook

        改进：对于未映射的地址，先检查是否是MMIO，如果是则返回1
        """
        pc = int(uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
        address = int(address) & 0xFFFFFFFF
        size = max(1, int(size or 1))
        self._record_unmapped_access(access, address, size, value, pc)

        # An invalid-memory callback is a discovery boundary, not the place to
        # commit a modeled read/write.  Map the complete dependency first and
        # let the original guest instruction execute again after the safe-point
        # retry.  The mapped read/write hooks then apply the normal MMIO,
        # external-input, and bit-band semantics exactly once.
        decoded_alias = self._decode_bitband_alias(address)
        try:
            if decoded_alias is not None:
                mapped = self._ensure_bitband_access_mapped(
                    address,
                    size,
                    verify_native=True,
                )
            else:
                mapped = self._ensure_memory_mapped(
                    address,
                    size,
                    verify_native=True,
                )
        except _DeferredMemoryMapping:
            if (
                access == UC_MEM_WRITE_UNMAPPED
                and not bool(getattr(self, "mapped_write_hook_enabled", True))
                and (
                    decoded_alias is not None
                    or self._is_mmio_address(address)
                )
            ):
                self._mark_pending_unmapped_write_replay(
                    pc=pc,
                    address=address,
                    size=size,
                    value=value,
                )
            if decoded_alias is None and not self._is_mmio_address(address):
                # This is queue metadata, not a modeled read/write. It becomes
                # visible to mapped-memory hooks only after the map succeeds.
                self._mark_pending_external_memory_address(address)
            raise
        if (
            mapped
            and decoded_alias is None
            and not self._is_mmio_address(address)
        ):
            self.external_memory_input_addresses.add(address)
        return bool(mapped)

    def run(
        self,
        entry_point=None,
        max_instructions=5000000,
        timeout_seconds: Optional[float] = None,
        preserve_cpu_state: bool = False,
        stop_before_pc: Optional[int] = None,
    ):
        """运行仿真"""
        start_time = time.time()

        # A failed snapshot/external-state restore can leave native emulator
        # state only partially restored.  Such an engine is quarantined by
        # the replay layer and must never cross back into Unicorn.  Keep this
        # guard before any register write, hook installation, or emu_start so
        # callers receive an explicit failed execution record instead of a
        # second native crash from a poisoned state.
        if bool(getattr(self, "replay_state_poisoned", False)):
            self._execution_sequence = int(
                getattr(self, "_execution_sequence", 0) or 0
            ) + 1
            self.execution_id = f"emu-{id(self):x}-{self._execution_sequence}"
            poisoned_result = {
                "execution_attempted": True,
                "execution_started": False,
                "execution_completed_normally": False,
                "execution_failed": True,
                "preflight_failed": True,
                "execution_telemetry_complete": False,
                "stop_reason": "replay_state_poisoned",
                "execution_error": (
                    "replay engine state was poisoned after restore failure"
                ),
                "preserve_cpu_state": bool(preserve_cpu_state),
                "entry_derivation": (
                    "entry_derived_continuation"
                    if preserve_cpu_state
                    else "reset_entry"
                ),
                "execution_id": self.execution_id,
                "unique_bbs": 0,
                "instruction_count": 0,
                "covered_bb_list": [],
                "execution_intervention_reasons": [],
                "execution_intervention_reasons_authoritative": True,
                "configured_intervention_counts": {},
                **self._forced_branch_audit_fields(),
            }
            self.last_run_result = poisoned_result
            return poisoned_result

        self._execution_sequence = int(getattr(self, "_execution_sequence", 0) or 0) + 1
        self.execution_id = f"emu-{id(self):x}-{self._execution_sequence}"
        # r38：run() 入口再刷一次总开关，兜住 setter 之后才设 env 的场景；
        # 已装钩子的残留 force 由钩子保险丝挡下并计数。
        self._refresh_forced_branch_disabled_flag()
        self._execution_counter_baseline = self._execution_counter_snapshot()
        self._forced_trace_baseline_len = len(
            list(getattr(self, "forced_branch_trace", []) or [])
        )
        self._stream_summary_event_baseline_len = len(
            list(getattr(self, "stream_input_summary_events", []) or [])
        )
        self._stream_payload_write_baseline_len = len(
            list(getattr(self, "stream_input_payload_writes", []) or [])
        )
        # Snapshot hooks run synchronously inside Unicorn.  Mark the native
        # execution interval explicitly so snapshots captured by those hooks
        # remain pending until this invocation has a final outcome.
        self._execution_active = True
        if not preserve_cpu_state:
            # A reset/entry execution starts a new provenance chain.  Snapshot
            # restoration sets this field again before preserve-state replay.
            self._active_prefix_provenance = {}
        self.stop_requested_reason = None
        previous_stop_before_pc = self.stop_before_pc
        self.stop_before_pc = int(stop_before_pc) & ~1 if stop_before_pc is not None else None

        # These fields describe this invocation only.  In particular, a
        # replay that fails after entering Unicorn must not inherit the
        # previous invocation's ``last_run_result`` or be reported as a
        # complete execution merely because it has a stop reason.
        execution_started = False
        execution_failed = False
        preflight_failed = False
        execution_completed_normally = False
        execution_error: Optional[str] = None
        stop_reason = "not_started"
        runtime_monitor = None

        try:
            if preserve_cpu_state:
                entry_point = self.uc.reg_read(UC_ARM_REG_PC) & ~1
            else:
                if entry_point is None:
                    entry_point = self.uc.reg_read(UC_ARM_REG_PC) & ~1
                if entry_point & 1:
                    self.execution_thumb = True
                    entry_point = entry_point & ~1

                self.uc.reg_write(UC_ARM_REG_PC, entry_point)

                # 注意：不要覆盖SP，它已经在load_firmware中正确设置了
                # 如果SP为0，说明load_firmware失败，使用默认值
                current_sp = self.uc.reg_read(UC_ARM_REG_SP)
                if current_sp == 0:
                    self.uc.reg_write(UC_ARM_REG_SP, 0x20005000)
                    logger.warning("SP为0，使用默认值0x20005000")

            count_limit = max(0, int(max_instructions or 0))
            count_label = "unbounded" if count_limit <= 0 else f"{count_limit:,}"
            logger.info(f"开始仿真: 入口点 0x{entry_point:08x}, 最大指令数 {count_label}")

            timeout_us = 0
            if timeout_seconds is not None and timeout_seconds > 0:
                timeout_us = int(timeout_seconds * 1_000_000)

            start_pc = (entry_point | 1) if self.execution_thumb else (entry_point & ~1)
            stop_pc = 0
            if (start_pc & ~1) == 0:
                code_size = int(getattr(self.arch_info, "code_size", 0) or 0)
                stop_pc = (int(self.base_addr) + code_size) & 0xFFFFFFFF if code_size else 0xFFFFFFFF
                if stop_pc == 0:
                    stop_pc = 0xFFFFFFFF
            recovery_count = 0
            preflight_ok, preflight_record = self._execution_preflight(start_pc)
            if not preflight_ok:
                preflight_failed = True
                execution_failed = True
                stop_reason = f"preflight_failed:{preflight_record.get('reason', 'unknown')}"
            else:
                runtime_monitor = self._install_runtime_crash_monitor()
                while True:
                    try:
                        # Mark the native boundary before calling into
                        # Unicorn.  If it raises, the invocation is still a
                        # failed attempted execution, even when no
                        # instruction was ultimately retired.
                        execution_started = True
                        self._managed_emu_start(
                            start_pc,
                            stop_pc,
                            timeout=timeout_us,
                            count=count_limit,
                        )
                        if runtime_monitor is not None and getattr(runtime_monitor, "event", None) is not None:
                            stop_reason = str(runtime_monitor.event.stop_reason or "runtime_crash_monitor_event")
                            execution_failed = True
                        elif self.stop_requested_reason:
                            stop_reason = self.stop_requested_reason
                        elif timeout_us and count_limit > 0:
                            stop_reason = "timeout_or_max_instructions_reached"
                        elif timeout_us:
                            stop_reason = "timeout_reached"
                        elif count_limit > 0:
                            stop_reason = "max_instructions_reached"
                        else:
                            stop_reason = "completed"
                        execution_completed_normally = not execution_failed
                        break
                    except Exception as e:
                        if recovery_count < 128 and self._try_recover_invalid_thumb_instruction(e):
                            recovery_count += 1
                            start_pc = int(self.uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
                            if self.execution_thumb:
                                start_pc |= 1
                            retry_ok, retry_record = self._execution_preflight(start_pc)
                            if not retry_ok:
                                preflight_failed = True
                                execution_failed = True
                                stop_reason = f"preflight_failed:{retry_record.get('reason', 'unknown')}"
                                break
                            continue
                        if runtime_monitor is not None and getattr(runtime_monitor, "event", None) is not None:
                            stop_reason = str(runtime_monitor.event.stop_reason or e)
                        else:
                            stop_reason = str(e)
                        execution_failed = True
                        execution_error = str(e)[:2000]
                        break
        except Exception as exc:
            # Setup failures (register access, monitor installation, hook
            # lifecycle errors, etc.) used to escape before a run result was
            # materialized.  Preserve the failure as explicit telemetry so a
            # parent replay can classify it without losing the whole phase.
            execution_failed = True
            execution_error = str(exc)[:2000]
            stop_reason = f"execution_setup_exception:{type(exc).__name__}"
        finally:
            runtime_monitor_record = None
            if 'runtime_monitor' in locals() and runtime_monitor is not None:
                try:
                    runtime_monitor_record = runtime_monitor.as_record()
                except Exception as exc:
                    runtime_monitor_record = {"error": str(exc)}
                try:
                    runtime_monitor.uninstall()
                except Exception:
                    pass
            self.stop_before_pc = previous_stop_before_pc
            self._execution_active = False
            # Finalize capture-time lineage even if result assembly below is
            # interrupted by an unexpected Python-side error.  The manager
            # only updates snapshots belonging to this execution id; an
            # intervention that occurs after a snapshot was captured is not
            # retroactively copied into that prefix.
            try:
                final_counter_delta = self._execution_counter_delta(
                    self._execution_counter_snapshot(),
                    dict(getattr(self, "_execution_counter_baseline", {}) or {}),
                )
                final_trace = list(
                    list(getattr(self, "forced_branch_trace", []) or [])[
                        max(0, int(getattr(self, "_forced_trace_baseline_len", 0) or 0)):
                    ]
                )
                if final_trace:
                    final_counter_delta["forced_branch_trace"] = final_trace
                self.branch_snapshot_manager.finalize_snapshot_provenance(
                    self.execution_id,
                    successful=bool(
                        execution_completed_normally
                        and not execution_failed
                        and not preflight_failed
                    ),
                    telemetry_complete=bool(
                        execution_started
                        and execution_completed_normally
                        and not execution_failed
                        and not preflight_failed
                    ),
                    actual_intervention_reasons=execution_intervention_reasons(
                        final_counter_delta
                    ),
                )
            except Exception as exc:
                logger.debug("快照provenance收尾失败: %s", exc)

        elapsed = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info(f"仿真完成")
        logger.info(f"{'='*80}")
        logger.info(f"停止原因: {stop_reason}")
        logger.info(f"执行时间: {elapsed:.2f}s")
        logger.info(f"唯一BB: {len(self.bb_addr_set)}")
        logger.info(f"指令数: {self.instruction_count:,}")
        logger.info(
            "MMIO访问: retained=%d total=%d discarded=%d",
            len(self.mmio_access_history),
            int(getattr(self, "mmio_access_history_total", len(self.mmio_access_history)) or 0),
            int(getattr(self, "mmio_access_history_entries_discarded", 0) or 0),
        )
        logger.info(
            "memory访问: retained=%d total=%d discarded=%d",
            len(self.memory_access_history),
            int(getattr(self, "memory_access_history_total", len(self.memory_access_history)) or 0),
            int(getattr(self, "memory_access_history_entries_discarded", 0) or 0),
        )
        logger.info(f"干预次数: {self.intervention_count}")

        # 输出最后10个执行的PC
        logger.info(f"\n最后10个执行的PC:")
        last_10_pcs = self.bb_history[-10:] if len(self.bb_history) >= 10 else self.bb_history
        for i, pc in enumerate(last_10_pcs, 1):
            logger.info(f"  {i}. 0x{pc:08x}")

        # 干预统计
        logger.info(f"\n干预统计:")
        for loop_type, count in self.intervention_by_type.items():
            if count > 0:
                logger.info(f"  {loop_type.value}: {count}次")

        # 模块统计
        loop_classifier_stats = self.loop_classifier.get_statistics()
        logger.info(f"\n模块统计:")
        logger.info(f"  循环分类器: 检测到 {len(self.loop_classifier.loop_heads)} 个循环")
        logger.info(f"  循环分类详情: {loop_classifier_stats.get('by_type')}")
        logger.info(f"  mapped写入观测: {self.mapped_memory_write_stats}")
        logger.info(
            "  动态代码BB: added=%d rejected=%d split=%d written_pages=%d",
            self.dynamic_static_bbs_added,
            self.dynamic_static_bb_rejections,
            self.dynamic_static_bb_split_count,
            len(self.runtime_written_pages),
        )
        logger.info(f"  时间函数处理器: {self.time_handler.get_statistics()}")
        logger.info(f"  等待循环处理器: {self.wait_loop_handler.get_statistics()}")
        logger.info(f"  快照管理器: {self.snapshot_manager.get_statistics()}")
        logger.info(f"  代码分析器: {self.code_analyzer.get_statistics()}")
        if any(self.runtime_loop_branch_force_stats.values()):
            logger.info(f"  运行时loop分支约束: {self.runtime_loop_branch_force_stats}")
        if any(self.modeled_async_memory_stats.values()):
            logger.info(f"  异步内存状态模型: {self.modeled_async_memory_stats}")
        if self.static_mmio_prediction_stats.get("total_accesses"):
            logger.info(f"  静态MMIO预测: {self.static_mmio_prediction_stats}")
        seed_stats = self._mmio_seed_report_stats()
        if seed_stats.get("table_size"):
            logger.info(
                f"  静态MMIO初始种子: 表规模={seed_stats['table_size']} "
                f"立即数线索={seed_stats.get('from_immediate', 0)} "
                f"命中首读={seed_stats.get('applied_first_reads', 0)}"
            )
        if any(self.mapped_mmio_preload_stats.values()):
            logger.info(f"  mapped MMIO预装载: {self.mapped_mmio_preload_stats}")
        if any(self.status_loop_solver_stats.values()):
            logger.info(f"  状态寄存器循环求解: {self.status_loop_solver_stats}")
        if any(self.constraint_validation_stats.values()):
            logger.info(f"  约束语义校验: {self.constraint_validation_stats}")
        if any(self.memory_constraint_guard_stats.values()):
            logger.info(f"  memory约束安全门: {self.memory_constraint_guard_stats}")
        if self.svc_stats.get("handled"):
            logger.info(f"  SVC/SWI模型: {self.svc_stats}")
        if any(self.thumb_indirect_branch_repair_stats.values()):
            logger.info(f"  Thumb间接跳转修复: {self.thumb_indirect_branch_repair_stats}")
        if any(self.thumb_state_guard_stats.values()):
            logger.info(f"  Thumb状态守护: {self.thumb_state_guard_stats}")
        if any(self.invalid_thumb_recovery_stats.values()):
            logger.info(f"  Thumb非法指令恢复: {self.invalid_thumb_recovery_stats}")
        if any(self.skip_function_stats.values()):
            logger.info(f"  函数快进: {self.skip_function_stats}")
        if any(self.lzo_decompress_summary_stats.values()):
            logger.info(f"  LZO解压摘要: {self.lzo_decompress_summary_stats}")
        if any(self.external_rom_call_stats.values()):
            logger.info(f"  外部ROM调用摘要: {self.external_rom_call_stats}")
        if self.internal_unicorn_bb_fragments:
            logger.info(
                "  Unicorn内部BB片段: ignored_transitions=%d",
                self.internal_unicorn_bb_fragments,
            )

        final_registers = self._get_registers()
        mmio_access_tail = [
            {
                "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
                "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
                "is_read": bool(is_read),
                "value": f"0x{int(value or 0) & 0xFFFFFFFF:08x}",
            }
            for pc, address, is_read, value in list(self.mmio_access_history[-32:])
        ]
        memory_access_tail = [
            {
                "pc": f"0x{int(pc) & 0xFFFFFFFF:08x}",
                "address": f"0x{int(address) & 0xFFFFFFFF:08x}",
                "is_read": bool(is_read),
                "value": f"0x{int(value or 0) & 0xFFFFFFFF:08x}",
            }
            for pc, address, is_read, value in list(self.memory_access_history[-32:])
        ]

        run_counter_delta = self._execution_counter_delta(
            self._execution_counter_snapshot(),
            dict(getattr(self, "_execution_counter_baseline", {}) or {}),
        )
        run_trace_since_start = list(
            list(getattr(self, "forced_branch_trace", []) or [])[
                max(0, int(getattr(self, "_forced_trace_baseline_len", 0) or 0)):
            ]
        )
        if run_trace_since_start:
            run_counter_delta["forced_branch_trace"] = run_trace_since_start
        run_intervention_reasons = execution_intervention_reasons(run_counter_delta)
        stream_event_start = max(
            0,
            int(getattr(self, "_stream_summary_event_baseline_len", 0) or 0),
        )
        stream_events = list(
            getattr(self, "stream_input_summary_events", []) or []
        )[stream_event_start:]
        payload_write_start = max(
            0,
            int(getattr(self, "_stream_payload_write_baseline_len", 0) or 0),
        )
        stream_payload_writes = list(
            getattr(self, "stream_input_payload_writes", []) or []
        )[payload_write_start:]
        run_environment_facts = environment_input_facts(
            {
                "environment_input_delivery_stats": run_counter_delta.get(
                    "environment_input_delivery_stats", {}
                ),
                "peripheral_input_stats": run_counter_delta.get(
                    "peripheral_input_stats", {}
                ),
                "stream_input_summary_events": stream_events,
                "stream_input_payload_writes": stream_payload_writes,
            }
        )
        run_environment_diagnostics = environment_model_diagnostics(
            {
                "peripheral_input_stats": run_counter_delta.get(
                    "peripheral_input_stats", {}
                ),
                "stream_input_summary_events": stream_events,
            }
        )
        run_configured_intervention_counts = configured_intervention_counts({
            "forced_branch_choices_configured": len(
                dict(getattr(self, "forced_branch_choices", {}) or {})
            ),
            "forced_choices_requested": len(
                list(getattr(self, "forced_branch_sequence", []) or [])
            ),
            "runtime_loop_branch_force_stats": getattr(
                self, "runtime_loop_branch_force_stats", {}
            ),
            "skip_function_stats": getattr(self, "skip_function_stats", {}),
        })

        self.last_run_result = {
            'unique_bbs': len(self.bb_addr_set),
            'instruction_count': self.instruction_count,
            # P1-D（cycle3 k.5 C7）：instruction_count 按**规范化 BB** 自增
            #（块回调），单位由本键显式化（报告计量坑：phase_metadata 的
            # run_result 即本字典）。
            'is_bb_counted': True,
            'access_history_retention': {
                'mmio_limit': int(getattr(self, 'mmio_access_history_limit', 0) or 0),
                'mmio_retained': len(getattr(self, 'mmio_access_history', []) or []),
                'mmio_total': int(
                    getattr(self, 'mmio_access_history_total', len(getattr(self, 'mmio_access_history', []) or []))
                    or 0
                ),
                'mmio_discarded': int(
                    getattr(self, 'mmio_access_history_entries_discarded', 0) or 0
                ),
                'memory_limit': int(getattr(self, 'memory_access_history_limit', 0) or 0),
                'memory_retained': len(getattr(self, 'memory_access_history', []) or []),
                'memory_total': int(
                    getattr(self, 'memory_access_history_total', len(getattr(self, 'memory_access_history', []) or []))
                    or 0
                ),
                'memory_discarded': int(
                    getattr(self, 'memory_access_history_entries_discarded', 0) or 0
                ),
            },
            'bb_history': {
                'limit': int(getattr(self, 'bb_history_limit', 0) or 0),
                'trim_trigger': int(getattr(self, 'bb_history_limit', 0) or 0) * 2,
                'retained': len(getattr(self, 'bb_history', []) or []),
                'total': int(self._bb_history_depth()),
                'discarded': int(
                    getattr(self, 'bb_history_entries_discarded', 0) or 0
                ),
                'successor_edge_count': len(
                    getattr(self, 'runtime_successor_edges', set()) or set()
                ),
            },
            'internal_unicorn_bb_fragments': int(self.internal_unicorn_bb_fragments),
            'stop_reason': stop_reason,
            'elapsed_time': elapsed,
            'execution_attempted': bool(
                preflight_failed or execution_started or execution_failed
            ),
            'execution_started': bool(execution_started),
            'execution_failed': bool(execution_failed),
            'preflight_failed': bool(preflight_failed),
            'execution_completed_normally': bool(execution_completed_normally),
            'preserve_cpu_state': bool(preserve_cpu_state),
            'entry_derivation': (
                'entry_derived_continuation'
                if preserve_cpu_state
                else 'reset_entry'
            ),
            'execution_error': execution_error,
            'registers': {
                name: f"0x{int(value) & 0xFFFFFFFF:08x}"
                for name, value in final_registers.items()
            },
            'intervention_count': self.intervention_count,
            'intervention_by_type': {k.value: v for k, v in self.intervention_by_type.items()},
            # r38 审计面：总开关是否生效 + 被挡下的 force 请求计数
            # （setter 入口挡下的装配请求 / 钩子保险丝挡下的残留应用）。
            **self._forced_branch_audit_fields(),
            # r9 裁定：家族标签（快进/轮询MMIO/等待处理 = 工程优化·外部输入，
            # 不入干预集合；可见性靠这两个字段，判定见 evidence_contract）。
            'intervention_event_labels': {
                str(k): int(v) for k, v in dict(self.intervention_event_labels).items()
            },
            'loop_unresolved_limit_trips': int(
                getattr(self, 'loop_unresolved_limit_trips', 0) or 0
            ),
            'execution_id': str(getattr(self, 'execution_id', '') or ''),
            'execution_telemetry_complete': bool(
                execution_started
                and execution_completed_normally
                and not execution_failed
                and not preflight_failed
            ),
            'execution_intervention_reasons': run_intervention_reasons,
            'execution_intervention_reasons_authoritative': True,
            # intervention_reasons() 返回纯字符串列表; dict() 会把每个字符串
            # 当 (k, v) 拆包而抛 ValueError, 这里按"类别 -> 出现次数(>=1)"组装。
            'execution_intervention_counts': dict.fromkeys(
                execution_intervention_reasons(run_counter_delta), 1
            ),
            'environment_facts': dict(run_environment_facts),
            'environment_input_facts': dict(run_environment_facts),
            'environment_input_delivery_stats': dict(
                run_counter_delta.get("environment_input_delivery_stats", {})
                or {}
            ),
            'peripheral_input_stats': dict(
                run_counter_delta.get("peripheral_input_stats", {}) or {}
            ),
            'environment_model_diagnostics': dict(run_environment_diagnostics),
            'stream_input_summary_events': [
                dict(item) for item in stream_events if isinstance(item, dict)
            ][-256:],
            'stream_input_payload_writes': [
                dict(item) for item in stream_payload_writes if isinstance(item, dict)
            ][-256:],
            'configured_intervention_counts': dict(
                run_configured_intervention_counts
            ),
            'execution_provenance': _safe_execution_provenance(self),
            'loop_classifier_stats': loop_classifier_stats,
            'snapshot_manager_stats': self.snapshot_manager.get_statistics(),
            'branch_snapshot_manager_stats': self.branch_snapshot_manager.get_statistics(),
            'causal_input_stats': {
                'logical_event_count': int(
                    getattr(self, 'causal_input_event_count', 0) or 0
                ),
                'trace_event_sequence': int(
                    getattr(self, 'causal_input_trace_sequence', 0) or 0
                ),
                'retained_events': len(
                    list(getattr(self, 'causal_input_events', []) or [])
                ),
                'retained_snapshots': len(
                    list(getattr(self, 'causal_input_snapshots', []) or [])
                ),
                'retention': dict(
                    getattr(self, 'causal_input_retention_stats', {}) or {}
                ),
                'event_tail': [
                    dict(event)
                    for event in list(
                        getattr(self, 'causal_input_events', []) or []
                    )[-128:]
                    if isinstance(event, dict)
                ],
            },
            'causal_context': self.causal_context.fingerprint_components(),
            'snapshot_capture_stats': dict(self.snapshot_capture_stats),
            'snapshot_page_store': self.snapshot_page_store.get_statistics(),
            'mapped_memory_write_stats': dict(self.mapped_memory_write_stats),
            'dynamic_static_bb_stats': {
                'added': int(self.dynamic_static_bbs_added),
                'rejected_unproven': int(self.dynamic_static_bb_rejections),
                'split_internal_targets': int(self.dynamic_static_bb_split_count),
                'runtime_written_pages': len(self.runtime_written_pages),
                'unproven_target_counts': {
                    f"0x{int(address) & 0xFFFFFFFF:08x}": int(count)
                    for address, count in self.unproven_dynamic_code_targets.most_common(32)
                },
                'external_rom_call_stats': dict(self.external_rom_call_stats),
                'external_rom_call_targets': {
                    f"0x{int(address) & 0xFFFFFFFF:08x}": int(count)
                    for address, count in self.external_rom_call_targets.most_common(32)
                },
            },
            'last_10_pcs': [f"0x{pc:08x}" for pc in last_10_pcs],
            'runtime_loop_branch_force_stats': dict(self.runtime_loop_branch_force_stats),
            'modeled_async_memory_stats': dict(self.modeled_async_memory_stats),
            'static_mmio_prediction_stats': dict(self.static_mmio_prediction_stats),
            'mmio_seed_stats': self._mmio_seed_report_stats(),
            'static_mmio_access_tail': [
                dict(item)
                for item in list(getattr(self, "static_mmio_accesses", []) or [])[-32:]
                if isinstance(item, dict)
            ],
            'function_mmio_summary_top': self._top_function_mmio_summaries(),
            'mapped_mmio_preload_stats': dict(self.mapped_mmio_preload_stats),
            'status_loop_solver_stats': dict(self.status_loop_solver_stats),
            'constraint_validation_stats': dict(self.constraint_validation_stats),
            'constraint_validation_history': list(self.constraint_validation_history),
            'memory_constraint_guard_stats': dict(self.memory_constraint_guard_stats),
            'memory_constraint_guard_history': list(self.memory_constraint_guard_history[-128:]),
            'execution_preflight': dict(self.last_execution_preflight),
            'execution_preflight_stats': dict(self.execution_preflight_stats),
            'last_unmapped_access': dict(self.last_unmapped_access or {}),
            'memory_mapping_lifecycle': {
                'pending_pages': len(
                    getattr(self, '_pending_memory_map_pages', {}) or {}
                ),
                'pending_origins': sorted(
                    getattr(self, '_pending_memory_map_origins', set()) or set()
                ),
                'pending_external_addresses': len(
                    getattr(
                        self,
                        '_pending_memory_map_external_addresses',
                        set(),
                    )
                    or set()
                ),
                'pending_write_replays': len(
                    getattr(self, '_pending_memory_map_write_replays', []) or []
                ),
                'active_write_retry_hooks': len(
                    getattr(self, '_deferred_write_retry_hooks', []) or []
                ),
                'block_callback_retry_pending': bool(
                    getattr(
                        self,
                        '_pending_memory_map_retries_block_callback',
                        False,
                    )
                ),
                'retry_block_marker': (
                    f"0x{int(self._deferred_retry_skip_block_pc) & 0xFFFFFFFF:08x}"
                    if getattr(self, '_deferred_retry_skip_block_pc', None)
                    is not None
                    else None
                ),
                'retry_block_reentry_marker': (
                    f"0x{int(self._deferred_retry_reenter_block_pc) & 0xFFFFFFFF:08x}"
                    if getattr(self, '_deferred_retry_reenter_block_pc', None)
                    is not None
                    else None
                ),
                'native_emulation_depth': int(
                    getattr(self, '_native_emulation_depth', 0) or 0
                ),
                'stats': dict(getattr(self, 'memory_mapping_stats', {}) or {}),
                'last_mapping': dict(
                    getattr(self, 'last_memory_mapping', {}) or {}
                ),
            },
            'svc_stats': {
                'handled': int(self.svc_stats.get('handled', 0)),
                'by_number': dict(self.svc_stats.get('by_number', {})),
            },
            'thumb_indirect_branch_repair_stats': dict(self.thumb_indirect_branch_repair_stats),
            'thumb_state_guard_stats': dict(self.thumb_state_guard_stats),
            # r29: 脏 ITSTATE 守卫审计面——站点与指令族计数，干预必须可追溯。
            'it_state_guard_stats': {
                **dict(self.it_state_guard_stats),
                'enabled': bool(self.it_state_guard_enabled),
                'sites': {
                    f'0x{pc:08x}': int(n)
                    for pc, n in self._it_state_guard_sites.most_common(32)
                },
                'kinds': dict(self._it_state_guard_kinds),
            },
            'invalid_thumb_recovery_stats': dict(self.invalid_thumb_recovery_stats),
            'skip_function_stats': dict(self.skip_function_stats),
            'lzo_decompress_summary_stats': dict(self.lzo_decompress_summary_stats),
            'mmio_access_tail': mmio_access_tail,
            'memory_access_tail': memory_access_tail,
            'watch_memory_events': list(self.watch_memory_events[-128:]),
            'watch_pc_events': list(self.watch_pc_events[-128:]),
            'hook_lifecycle': {
                'active_owned_hooks': len(getattr(self, '_owned_hooks', []) or []),
                'pending_deletions': len(
                    getattr(self, '_pending_owned_hook_removals', []) or []
                ),
                'native_emulation_depth': int(
                    getattr(self, '_native_emulation_depth', 0) or 0
                ),
                'stats': dict(getattr(self, 'hook_lifecycle_stats', {}) or {}),
            },
        }
        if runtime_monitor_record is not None:
            self.last_run_result['runtime_monitor'] = runtime_monitor_record
            if isinstance(runtime_monitor_record, dict) and runtime_monitor_record.get("runtime_crash_event"):
                self.last_run_result['runtime_crash_event'] = runtime_monitor_record.get("runtime_crash_event")
                self.last_run_result.setdefault("source_layer", "runtime_monitor")
        self._attach_crash_triage(self.last_run_result)
        self._advance_active_prefix_provenance_from_run(self.last_run_result)
        return self.last_run_result
