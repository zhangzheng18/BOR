"""DFS 回退锚点池（r7 P0-B）.

深度优先 + 自底向上翻转需要一个不受常规前缀快照 FIFO 淘汰影响的锚点层：
主干上按 K 个分支点保留一个可回退锚点，热层驻内存、冷层写文件（复用
``_AppendOnlyFileStore``，页级去重由 P0-A 接通的 SnapshotPageStore 页池承担），
锚点必须携带血统（``snapshot_provenance_record`` 解析）与前缀约束表快照
（R5 审计 §4.3 缺口 4：重放等价性要求前缀期内学到且后缀仍应生效的输入假设
随锚点保存），血统不干净或缺约束表的锚点禁止用于翻转（fail closed，不降级）。

该模块刻意不依赖 HistoricalRunner，以便独立单测；runner 侧的接线见
``historical_runner.HistoricalRunner._dfs_consider_prefix_anchor``。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .analysis.snapshot_memory import _AppendOnlyFileStore
from .evidence_contract import snapshot_provenance_record
from .runner_models import PrefixReplaySnapshot

__all__ = [
    "DFSAnchorEntry",
    "DFSColdAnchorStore",
    "DFSAnchorPool",
    "anchor_lineage_status",
    "dfs_anchor_stride",
    "dfs_anchor_limit",
    "dfs_anchor_hot_limit",
    "dfs_anchor_fallback_deepest",
]


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return int(default)
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def dfs_anchor_stride() -> int:
    """主干上每 K 个分支点保留一个锚点（K 默认 8）。"""
    return max(1, _env_int("LSGEMU_DFS_ANCHOR_STRIDE", 8))


def dfs_anchor_limit() -> int:
    """锚点总数上限（热+冷，默认 256；0 关闭锚点捕获）。"""
    return max(0, _env_int("LSGEMU_DFS_ANCHOR_LIMIT", 256))


def dfs_anchor_hot_limit() -> int:
    """热层内存保留的最近锚点数（默认 32；更早的降级到冷层）。"""
    return max(1, _env_int("LSGEMU_DFS_ANCHOR_HOT", 32))


def dfs_anchor_fallback_deepest() -> bool:
    """r32 D1：无候选跨过 stride 时的兜底登记开关（默认开）。

    「主干每 K 个分支点一个锚点」在诚实场景下可能整臂成立不了（r32 现场：
    4 小时 327 个候选前缀深度最大 7 < stride 8 ⇒ 锚点池全空 ⇒ 翻转 pass
    计划恒 0 ⇒ 其上的投递/SVC 机制一次都执行不到）。兜底只改「登记与否」，
    不改血统与 ``flip_eligible`` 的判定口径：不合格的候选照旧不可翻转。
    """
    return _env_flag("LSGEMU_DFS_ANCHOR_FALLBACK_DEEPEST", True)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def anchor_lineage_status(entry: Any) -> Tuple[bool, Tuple[str, ...], Dict[str, object]]:
    """解析锚点血统并给出翻转资格。

    翻转（P2）的合法性依赖「从该锚点重放 = 入口点可达」的认证：只有
    ``snapshot_provenance_record`` 解析为 validated 的血统才允许参与翻转，
    其余一律 fail closed。
    """
    record = snapshot_provenance_record(entry)
    status = str(record.get("status") or "unverified")
    reasons = tuple(str(item) for item in (record.get("reasons") or ()))
    eligible = status == "validated"
    summary: Dict[str, object] = {
        "status": status,
        "execution_id": str(record.get("execution_id") or ""),
        "finalized": bool(record.get("finalized")),
    }
    return eligible, reasons, summary


@dataclass
class DFSAnchorEntry:
    """单个回退锚点：前缀快照 + 血统 + 前缀约束表 + 冷层位置。"""

    anchor_key: Tuple[tuple, Tuple[int, int]]
    depth: int
    sequence: int
    entry: Optional[PrefixReplaySnapshot]
    prefix_constraint_tables: Optional[Dict[str, object]]
    flip_eligible: bool = False
    ineligibility_reasons: Tuple[str, ...] = field(default_factory=tuple)
    lineage: Dict[str, object] = field(default_factory=dict)
    # 冷层位置：(offset, length, sha256)。热锚点为 None。
    cold_location: Optional[Tuple[int, int, str]] = None
    # 热层 LRU 序：复活/新登记时刷新；降冷淘汰按 touch 而非锚点序。
    touch: int = 0

    @property
    def is_hot(self) -> bool:
        return self.entry is not None and self.cold_location is None

    def _constraint_table_sizes(self) -> Dict[str, int]:
        tables = self.prefix_constraint_tables
        if not isinstance(tables, dict):
            return {}
        sizes: Dict[str, int] = {}
        for name, items in tables.items():
            try:
                sizes[str(name)] = len(items)
            except TypeError:
                sizes[str(name)] = 1
        return sizes

    def describe(self) -> Dict[str, object]:
        return {
            "depth": int(self.depth),
            "sequence": int(self.sequence),
            "hot": bool(self.is_hot),
            "cold": bool(self.cold_location is not None),
            "flip_eligible": bool(self.flip_eligible),
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "lineage": dict(self.lineage),
            "has_constraint_tables": self.prefix_constraint_tables is not None,
            "constraint_table_sizes": self._constraint_table_sizes(),
        }


class _RestrictedAnchorUnpickler(pickle.Unpickler):
    """冷锚点文件的白名单 unpickler（与静态缓存同款防御，允许面更窄）。

    冷锚点文件与静态缓存一样落在磁盘上，任何有写权限的进程都可替换；
    白名单外的 GLOBAL 一律拒绝。
    """

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_ANCHOR_GLOBALS:
            raise pickle.UnpicklingError(
                f"forbidden global in dfs anchor payload: {module}.{name}"
            )
        return super().find_class(module, name)


_ALLOWED_ANCHOR_GLOBALS = frozenset({
    ("lsgemu.dfs_anchor_pool", "DFSAnchorEntry"),
    ("lsgemu.runner_models", "PrefixReplaySnapshot"),
    ("lsgemu.runner_models", "BranchConstraintCandidate"),
    ("lsgemu.analysis.branch_snapshot_manager", "BranchSnapshot"),
    ("lsgemu.analysis.snapshot_memory", "PagedMemory"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotStateBlob"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotPage"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    # r22：真实载荷的 __reduce__ 引用集合内建（dict/frozenset 重建字段，
    # getattr 来自 reduce 协议兼容路径）。均为数据重建原语，无 I/O、无
    # 执行面；与静态缓存白名单同信任级（文件在 run 专属目录内）。
    ("builtins", "dict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "bytes"),
    ("builtins", "getattr"),
})

_COLD_MAGIC = b"LSGEMU_DFS_ANCHOR_V1\n"


class DFSColdAnchorStore:
    """冷锚点载荷存储：append-only 文件 + sha256 完整性 + 白名单反序列化。"""

    def __init__(self, directory):
        self._store = _AppendOnlyFileStore(directory, "dfs-anchor")

    @property
    def directory(self):
        return self._store.directory

    @staticmethod
    def payload_bytes(anchor: DFSAnchorEntry) -> bytes:
        """把锚点编成自包含记录（magic + sha256 + 长度 + pickle 体）。"""
        body = pickle.dumps(anchor, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(body).hexdigest()
        return _COLD_MAGIC + struct.pack(">64sQ", digest.encode(), len(body)) + body

    def append_payload(self, payload: bytes) -> Tuple[int, int, str]:
        """追加一条已编码记录，返回 (offset, length, sha256)。"""
        header_len = len(_COLD_MAGIC) + 72
        if len(payload) < header_len or not payload.startswith(_COLD_MAGIC):
            raise ValueError("dfs anchor payload header mismatch")
        digest = struct.unpack(">64sQ", payload[len(_COLD_MAGIC):header_len])[0].decode()
        offset, length = self._store.append(payload)
        return offset, length, digest

    def read_payload(self, location: Tuple[int, int, str]) -> bytes:
        """按位置读回整条记录字节（含 magic 头，可原样再追加）。"""
        offset, length, _digest = location
        return self._store.read(offset, length)

    def serialize(self, anchor: DFSAnchorEntry) -> Tuple[int, int, str]:
        offset, length, digest = self.append_payload(self.payload_bytes(anchor))
        return offset, length, digest

    def deserialize(self, location: Tuple[int, int, str]) -> DFSAnchorEntry:
        offset, length, digest = location
        payload = self._store.read(offset, length)
        header_len = len(_COLD_MAGIC) + 72  # 64 字节 hexdigest + 8 字节长度
        if len(payload) < header_len or not payload.startswith(_COLD_MAGIC):
            raise ValueError("dfs anchor payload header mismatch")
        stored_digest = struct.unpack(">64sQ", payload[len(_COLD_MAGIC):header_len])[0].decode()
        body = payload[header_len:]
        actual = hashlib.sha256(body).hexdigest()
        if stored_digest != digest or actual != digest:
            raise ValueError("dfs anchor payload digest mismatch")
        anchor = _RestrictedAnchorUnpickler(_BytesReader(body)).load()
        if not isinstance(anchor, DFSAnchorEntry):
            raise ValueError("dfs anchor payload has unexpected type")
        return anchor

    def statistics(self) -> Dict[str, object]:
        return dict(self._store.statistics())


class _BytesReader:
    """让白名单 Unpickler 直接读内存字节（Unpickler 需要文件式对象）。"""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk, self._payload = self._payload, b""
            return chunk
        chunk = self._payload[:size]
        self._payload = self._payload[size:]
        return chunk

    def readline(self, size: int = -1) -> bytes:
        chunk, sep, rest = self._payload.partition(b"\n")
        self._payload = rest if sep else b""
        return chunk + (sep if sep else b"")


class DFSAnchorPool:
    """锚点单列存储：不参与 reservoir 前缀快照池的 FIFO 淘汰与阶段 reset。"""

    def __init__(
        self,
        *,
        stride: Optional[int] = None,
        hot_limit: Optional[int] = None,
        total_limit: Optional[int] = None,
        cold_store: Optional[DFSColdAnchorStore] = None,
    ):
        self.stride = dfs_anchor_stride() if stride is None else max(1, int(stride))
        self.hot_limit = dfs_anchor_hot_limit() if hot_limit is None else max(1, int(hot_limit))
        self.total_limit = dfs_anchor_limit() if total_limit is None else max(0, int(total_limit))
        self.cold_store = cold_store
        # key → DFSAnchorEntry，插入序即锚点序（热在前不保证；按 sequence 判新旧）。
        self.anchors: Dict[Tuple[tuple, Tuple[int, int]], DFSAnchorEntry] = {}
        self._next_sequence = 1
        # r10 ③：深度对齐窗口 [k*stride, (k+1)*stride) 的占用登记，与候选
        # 到达顺序无关（替代旧 _next_anchor_depth 单调门槛——首个跨门槛候选
        # 若来自深任务会把后续浅窗口永久屏蔽）。
        self._anchor_by_bucket: Dict[int, DFSAnchorEntry] = {}
        self._touch_counter = 0
        # r32 D1：兜底候选槽（只留「当前最深」一个；不登记进池，不占额度）
        # 与兜底锚点键集合（用于区分「池里只有兜底锚点」这一状态）。
        self._deepest_below_stride: Optional[Dict[str, object]] = None
        self._fallback_keys: Set[Tuple[tuple, Tuple[int, int]]] = set()
        self.stats: Dict[str, int] = {
            "considered": 0,
            "registered": 0,
            "demoted_to_cold": 0,
            "resurrected_from_cold": 0,
            "cold_demote_failures": 0,
            "dropped_over_limit": 0,
            "ineligible_lineage": 0,
            "ineligible_missing_tables": 0,
            # r9 C 件：空前缀（入口快照）不登记不占额度（成不了翻转主干）。
            "skipped_empty_prefix": 0,
            # r10 ③：步进拒绝显式记账（90min 现场 263 considered / 1 registered，
            # 其余 262 个静默丢失无法审计）+ 候选深度上界可观测（区分「主干
            # 太浅」与「步进误拒」）。
            "skipped_below_stride": 0,
            "skipped_stride_window_filled": 0,
            "window_upgrades": 0,
            "max_candidate_depth": 0,
            # r32 D1：兜底登记（r31 现场 327 候选全在 stride 下 ⇒ 池空转）。
            "fallback_promotions": 0,
            "fallback_upgrades": 0,
            "fallback_rejected_by_flag": 0,
            # r32 D1b：消费时点重导资格（源快照定稿后才可能 validated）。
            "eligibility_refreshes": 0,
            "eligibility_upgrades": 0,
            "eligibility_downgrades": 0,
        }
        # r32 D1：considered 候选中**非空前缀**的深度直方图（{深度字符串: 计数}）。
        # 「为什么 4 小时最大深度只有 7」这类问题原先只能看到上界一个数字，
        # 无法区分「全是 1-2」与「密集堆在 7」。直方图随任意锚点统计落盘。
        self.candidate_depth_histogram: Dict[str, int] = {}

    # -- 注册 ------------------------------------------------------------

    def consider(
        self,
        entry: PrefixReplaySnapshot,
        *,
        depth: int,
        prefix_constraint_tables: Optional[Dict[str, object]],
        anchor_key: Optional[Tuple[tuple, Tuple[int, int]]] = None,
    ) -> Optional[DFSAnchorEntry]:
        """按深度步进决定是否把一个前缀快照登记为锚点。

        ``anchor_key`` 缺省用 ``(entry.prefix_signature, entry.next_branch_key)``。
        步进语义（r10 ③，前缀长度口径）：深度对齐窗口 [k*stride,(k+1)*stride)
        至多保留一个锚点（「主干每 K 个分支点一个」），深度 < stride 的候选
        不登记。窗口占用与候选到达顺序无关；同窗口内血统不合格的锚点可被
        后到的合格候选顶替（不放宽血统判定，只在该窗口的稀疏预算内优先
        可用于翻转的锚点）。这是「主干每 K 个分支点一个锚点」的深度代理——
        严格的单一主干 T 由 P1 主干优先调度保证，这里不假设路径唯一性，
        翻转侧用祖先匹配过滤。
        """
        self.stats["considered"] += 1
        depth = int(depth)
        if depth > int(self.stats.get("max_candidate_depth", 0) or 0):
            self.stats["max_candidate_depth"] = depth
        if entry is not None:
            key_count = str(depth)
            self.candidate_depth_histogram[key_count] = (
                int(self.candidate_depth_histogram.get(key_count, 0)) + 1
            )
        if self.total_limit <= 0:
            return None
        if depth < self.stride:
            self.stats["skipped_below_stride"] += 1
            self._note_below_stride(depth, entry, prefix_constraint_tables)
            return None
        bucket = depth // self.stride
        existing = self._anchor_by_bucket.get(bucket)
        if existing is not None:
            # 顶替仅限「既有不合格、新候选合格」；血统/约束表判定本身不变。
            if existing.flip_eligible or prefix_constraint_tables is None:
                self.stats["skipped_stride_window_filled"] += 1
                return None
            eligible_new, _reasons_new, _lineage_new = anchor_lineage_status(entry)
            if not eligible_new:
                self.stats["skipped_stride_window_filled"] += 1
                return None
            self.anchors.pop(existing.anchor_key, None)
            self.stats["window_upgrades"] += 1
        if entry is None:
            return None
        return self._register(
            entry,
            depth=depth,
            prefix_constraint_tables=prefix_constraint_tables,
            anchor_key=anchor_key,
            bucket=bucket,
        )

    # -- 登记（consider 与兜底登记共用同一资格判定） -----------------------

    def _register(
        self,
        entry: PrefixReplaySnapshot,
        *,
        depth: int,
        prefix_constraint_tables: Optional[Dict[str, object]],
        anchor_key: Optional[Tuple[tuple, Tuple[int, int]]],
        bucket: int,
        fallback: bool = False,
    ) -> DFSAnchorEntry:
        """构造并登记一个锚点（血统/约束表判定与 ``consider`` 完全同口径）。

        ``fallback`` 只影响记账（``_fallback_keys``）；``flip_eligible``
        的计算路径一字不改——兜底登记不放宽血统，不合格照旧不可翻转。
        """
        key = anchor_key or (
            tuple(entry.prefix_signature or tuple()),
            (int(entry.next_branch_key[0]), int(entry.next_branch_key[1])),
        )
        eligible, reasons, lineage = anchor_lineage_status(entry)
        ineligibility: List[str] = []
        if not eligible:
            ineligibility.extend(reasons or ("ancestor_not_validated",))
            self.stats["ineligible_lineage"] += 1
        if prefix_constraint_tables is None:
            # 缺前缀约束表 ⇒ 拒绝翻转（R5 §4.3 缺口 4），不降级凑合。
            ineligibility.append("prefix_constraint_tables_missing")
            self.stats["ineligible_missing_tables"] += 1
        anchor = DFSAnchorEntry(
            anchor_key=key,
            depth=depth,
            sequence=self._next_sequence,
            entry=entry,
            prefix_constraint_tables=prefix_constraint_tables,
            flip_eligible=eligible and prefix_constraint_tables is not None,
            ineligibility_reasons=tuple(dict.fromkeys(ineligibility)),
            lineage=lineage,
        )
        self._next_sequence += 1
        self._touch_counter += 1
        anchor.touch = self._touch_counter
        self._anchor_by_bucket[bucket] = anchor
        self.anchors[key] = anchor
        if fallback:
            # 兜底登记不记入 `registered`：保持 consider 漏斗恒等式
            # considered == registered + skipped_below_stride + window_filled
            # 在兜底介入后依然成立（`registered` = 步进通过的登记数）。
            self._fallback_keys.add(key)
        else:
            self.stats["registered"] += 1
        self._enforce_limits()
        return anchor

    # -- r32 D1：stride 之下的兜底候选 ------------------------------------

    def _note_below_stride(
        self,
        depth: int,
        entry: Optional[PrefixReplaySnapshot],
        prefix_constraint_tables: Optional[Dict[str, object]],
    ) -> None:
        """记住「当前最深」的被步进拒绝的候选（只留一个，不占额度）。

        直存 ``PrefixReplaySnapshot`` 与约束表快照（不深拷）：该对象同时也
        被常规前缀快照池引用，这里多持有一个引用不产生额外快照拷贝。
        """
        if entry is None:
            return
        current = self._deepest_below_stride
        if current is not None and int(current.get("depth", 0) or 0) >= int(depth):
            return
        self._deepest_below_stride = {
            "depth": int(depth),
            "entry": entry,
            "prefix_constraint_tables": prefix_constraint_tables,
        }

    def deepest_below_stride(self) -> Optional[Dict[str, object]]:
        pending = self._deepest_below_stride
        return dict(pending) if pending is not None else None

    def refresh_eligibility(self) -> Dict[str, int]:
        """在**消费时点**重导每个锚点的翻转资格。

        锚点是在 ``remember_prefix_snapshot`` 期间登记的，那一刻源重放还没收尾
        （``provenance_finalized`` 未置位）⇒ ``snapshot_provenance_record`` 只能
        解析出未定稿状态，首判必然不是 validated。r32 现场实测：30 min 臂里
        兜底锚点 ``flip_eligible=False``（``ineligible_lineage=1``）。
        翻转 pass 在阶段末尾消费锚点，此时源重放已定稿，重导才是同一契约在
        **正确时点**的求值。判定口径一字不改：只有 validated（且约束表非空）
        才 ``flip_eligible``，其余仍 fail closed；冷锚点不复活，保持既有判定。
        """
        upgraded = downgraded = 0
        for anchor in self.anchors.values():
            if anchor.entry is None:
                continue
            eligible, reasons, lineage = anchor_lineage_status(anchor.entry)
            refreshed = bool(
                eligible and anchor.prefix_constraint_tables is not None
            )
            if refreshed and not anchor.flip_eligible:
                upgraded += 1
            elif not refreshed and anchor.flip_eligible:
                downgraded += 1
            anchor.flip_eligible = refreshed
            summary = dict(lineage)
            summary["eligibility_refreshed"] = True
            anchor.lineage = summary
            if not eligible:
                anchor.ineligibility_reasons = tuple(
                    dict.fromkeys(
                        list(anchor.ineligibility_reasons)
                        + list(reasons or ("ancestor_not_validated",))
                    )
                )
        self.stats["eligibility_refreshes"] = (
            int(self.stats.get("eligibility_refreshes", 0)) + 1
        )
        self.stats["eligibility_upgrades"] = (
            int(self.stats.get("eligibility_upgrades", 0)) + upgraded
        )
        self.stats["eligibility_downgrades"] = (
            int(self.stats.get("eligibility_downgrades", 0)) + downgraded
        )
        return {"upgraded": upgraded, "downgraded": downgraded}

    def promote_deepest_below_stride(self) -> Optional[DFSAnchorEntry]:
        """兜底登记：没有任何候选跨过 stride 时，登记「当前最深可用」候选。

        触发条件（全部满足）：
          * 开关 ``LSGEMU_DFS_ANCHOR_FALLBACK_DEEPEST`` 开着；
          * 池里没有**非兜底**锚点（有正常锚点 ⇒ 步进语义成立，兜底不介入）；
          * 存在被 ``depth < stride`` 拒绝的最深候选；
          * 该候选比既有兜底锚点更深（或池为空）——兜底只做「升级」不扩量，
            池里兜底锚点恒 ≤ 1 个。

        资格判定复用 ``_register``（= ``consider`` 同口径）：血统不合格的候选
        照旧 ``flip_eligible=False``，绝不为了凑出可翻转锚点而放宽血统。
        """
        pending = self._deepest_below_stride
        if pending is None:
            return None
        if not dfs_anchor_fallback_deepest():
            self.stats["fallback_rejected_by_flag"] += 1
            return None
        if any(key not in self._fallback_keys for key in self.anchors):
            return None
        depth = int(pending.get("depth", 0) or 0)
        existing_fallback = [
            anchor
            for key, anchor in self.anchors.items()
            if key in self._fallback_keys
        ]
        if existing_fallback:
            deepest_existing = max(int(a.depth) for a in existing_fallback)
            if depth <= deepest_existing:
                return None
        bucket = depth // self.stride
        for anchor in existing_fallback:
            # 同深度窗口的旧兜底锚点让位（升级语义，池内仍只有 1 个兜底）。
            self.anchors.pop(anchor.anchor_key, None)
            self._fallback_keys.discard(anchor.anchor_key)
            if self._anchor_by_bucket.get(bucket) is anchor:
                self._anchor_by_bucket.pop(bucket, None)
        anchor = self._register(
            pending.get("entry"),
            depth=depth,
            prefix_constraint_tables=pending.get("prefix_constraint_tables"),
            anchor_key=None,
            bucket=bucket,
            fallback=True,
        )
        if existing_fallback:
            self.stats["fallback_upgrades"] += 1
        else:
            self.stats["fallback_promotions"] += 1
        return anchor

    # -- 查找 ------------------------------------------------------------

    def lookup(
        self, anchor_key: Tuple[tuple, Tuple[int, int]]
    ) -> Optional[DFSAnchorEntry]:
        anchor = self.anchors.get(anchor_key)
        if anchor is None:
            return None
        if not anchor.is_hot:
            if not self._resurrect(anchor):
                return None
        return anchor

    def nearest_ancestor(
        self, path_items: List[Tuple[Tuple[int, int], object]]
    ) -> Optional[DFSAnchorEntry]:
        """给定一条有序选择路径，返回作为其前缀的最深锚点（P2 翻转回退点）。

        锚点的前缀签名必须与路径的前缀逐项相等（含 occurrence 键与方向）。
        """
        best: Optional[DFSAnchorEntry] = None
        for anchor in self.anchors.values():
            signature = list(anchor.anchor_key[0])
            if not signature or len(signature) > len(path_items):
                continue
            if any(
                _ordered_key(item[0]) != _ordered_key(sig_item[0])
                or item[1] != sig_item[1]
                for sig_item, item in zip(signature, path_items)
            ):
                continue
            if best is None or anchor.depth > best.depth:
                best = anchor
        if best is not None and not best.is_hot:
            if not self._resurrect(best):
                return None
        return best

    # -- 冷热分层 --------------------------------------------------------

    def _enforce_limits(self) -> None:
        while len(self.anchors) > self.total_limit:
            oldest_key = min(self.anchors, key=lambda k: self.anchors[k].sequence)
            self.anchors.pop(oldest_key, None)
            self.stats["dropped_over_limit"] += 1
        hot = [a for a in self.anchors.values() if a.is_hot]
        hot.sort(key=lambda a: a.touch)
        while len(hot) > self.hot_limit:
            victim = hot.pop(0)
            if not self._demote(victim):
                # 冷层不可用时保持热态（内存换正确性），计数暴露该退化。
                hot.insert(0, victim)
                break

    def _demote(self, anchor: DFSAnchorEntry) -> bool:
        if self.cold_store is None or anchor.entry is None:
            self.stats["cold_demote_failures"] += 1
            return False
        try:
            location = self.cold_store.serialize(anchor)
        except (OSError, ValueError, TypeError, pickle.PicklingError):
            self.stats["cold_demote_failures"] += 1
            return False
        anchor.cold_location = location
        anchor.entry = None
        self.stats["demoted_to_cold"] += 1
        return True

    def _resurrect(self, anchor: DFSAnchorEntry) -> bool:
        if anchor.cold_location is None or self.cold_store is None:
            return False
        try:
            restored = self.cold_store.deserialize(anchor.cold_location)
        except (OSError, ValueError, pickle.UnpicklingError):
            self.stats["cold_demote_failures"] += 1
            return False
        anchor.entry = restored.entry
        anchor.prefix_constraint_tables = restored.prefix_constraint_tables
        anchor.cold_location = None
        self.stats["resurrected_from_cold"] += 1
        # 复活即视为最近使用：刷新 touch，避免立刻被 _enforce_limits 挤回冷层。
        self._touch_counter += 1
        anchor.touch = self._touch_counter
        # 复活后可能需要把别的热锚点挤到冷层。
        self._enforce_limits()
        return True

    # -- 观测 ------------------------------------------------------------

    def dump(self, store: "DFSColdAnchorStore") -> Dict[str, int]:
        """池级落盘（r22：非 campaign 场景的锚点持久化通路）。

        热锚点整体序列化（含快照与前缀约束表）；冷锚点按原载荷字节复制
        ——冷锚点的 ``entry`` 已置 None，直接重序列化会丢快照。落盘不改变
        池内状态（不降级、不清 entry）；载荷新位置不回写 ``cold_location``，
        ``load`` 按文件扫描重建，不依赖任何索引文件。

        r23（P1）：每桶落盘上限 ``LSGEMU_DFS_ANCHOR_DUMP_BUCKET_LIMIT``
        （默认 4）——桶 = 锚点的下一分支 bb（``anchor_key[1][0]``）。同一
        bb 在环/重复路径上会产生大量不同深度的锚点（r22 采集 2048 ×
        ~1.6MB = 3.2GB 的主因），试翻只消费「每站最深可翻祖先」，桶内按
        （可翻优先、更深优先）截断即可保住全部消费面。0 = 不设限。
        """
        stats = {"dumped": 0, "dump_errors": 0, "bucket_capped": 0}
        bucket_limit = _env_int("LSGEMU_DFS_ANCHOR_DUMP_BUCKET_LIMIT", 4)
        anchors = list(self.anchors.values())
        if bucket_limit > 0:
            buckets: Dict[int, List[DFSAnchorEntry]] = {}
            for anchor in anchors:
                try:
                    bucket_key = int(anchor.anchor_key[1][0]) & 0xFFFFFFFF
                except Exception:
                    bucket_key = -1
                buckets.setdefault(bucket_key, []).append(anchor)
            selected: List[DFSAnchorEntry] = []
            for bucket in buckets.values():
                bucket.sort(
                    key=lambda a: (a.flip_eligible, int(a.depth)),
                    reverse=True,
                )
                stats["bucket_capped"] += max(0, len(bucket) - bucket_limit)
                selected.extend(bucket[:bucket_limit])
            # 落盘顺序保持深度单调（load 侧统计与日志可读性）。
            anchors = sorted(selected, key=lambda a: int(a.depth))
        for anchor in anchors:
            try:
                if anchor.is_hot:
                    store.append_payload(DFSColdAnchorStore.payload_bytes(anchor))
                elif (
                    anchor.cold_location is not None
                    and self.cold_store is not None
                ):
                    # 原载荷在本池自己的冷层文件里（降冷时的 store）。
                    store.append_payload(
                        self.cold_store.read_payload(anchor.cold_location)
                    )
                else:
                    stats["dump_errors"] += 1
                    continue
                stats["dumped"] += 1
            except Exception:
                # 单锚点失败只记账：dump 是尽力持久化，不中断整池。
                stats["dump_errors"] += 1
        stats["buckets"] = len({a.anchor_key[1][0] for a in anchors}) if anchors else 0
        return stats

    @classmethod
    def load(
        cls,
        directory: object,
        *,
        stride: Optional[int] = None,
        hot_limit: Optional[int] = None,
        total_limit: Optional[int] = None,
        cold_store: Optional["DFSColdAnchorStore"] = None,
    ) -> Tuple["DFSAnchorPool", Dict[str, int]]:
        """从落盘载荷重建锚点池（跨进程复用；与 ``dump`` 成对）。

        扫描目录内 ``dfs-anchor-*.bin`` 文件，逐条校验 sha256 后按白名单
        unpickler 反序列化。血统/约束表判定不放宽：不合规锚点照常登记但
        ``flip_eligible=False``（fail closed，与 ``consider`` 同口径）。
        深度窗口冲突时按「可翻转优先、更深优先」择一保留。
        """
        pool = cls(
            stride=stride,
            hot_limit=hot_limit,
            total_limit=total_limit,
            cold_store=cold_store,
        )
        stats = {"files": 0, "records": 0, "anchors": 0, "errors": 0}
        root = Path(str(directory)).expanduser()
        header_len = len(_COLD_MAGIC) + 72
        for path in sorted(root.glob("dfs-anchor-*.bin")):
            stats["files"] += 1
            try:
                data = path.read_bytes()
            except OSError:
                stats["errors"] += 1
                continue
            position = 0
            while position + header_len <= len(data):
                if not data.startswith(_COLD_MAGIC, position):
                    break
                digest_raw, body_len = struct.unpack(
                    ">64sQ", data[position + len(_COLD_MAGIC):position + header_len]
                )
                body_start = position + header_len
                body_end = body_start + int(body_len)
                if body_end > len(data):
                    stats["errors"] += 1
                    break
                body = data[body_start:body_end]
                stats["records"] += 1
                if hashlib.sha256(body).hexdigest() != digest_raw.decode():
                    stats["errors"] += 1
                    position = body_end
                    continue
                try:
                    anchor = _RestrictedAnchorUnpickler(_BytesReader(body)).load()
                except (pickle.UnpicklingError, OSError, ValueError, TypeError):
                    stats["errors"] += 1
                    position = body_end
                    continue
                if isinstance(anchor, DFSAnchorEntry) and anchor.anchor_key:
                    pool._register_loaded(anchor)
                    stats["anchors"] += 1
                position = body_end
        pool.stats["loaded_from_disk"] = stats["anchors"]
        return pool, stats

    def _register_loaded(self, anchor: DFSAnchorEntry) -> None:
        """登记一条落盘锚点：重导血统/约束表判定，窗口冲突择优。"""
        bucket = int(anchor.depth) // self.stride
        existing = self._anchor_by_bucket.get(bucket)
        if existing is not None and existing.anchor_key != anchor.anchor_key:
            new_better = (
                anchor.flip_eligible and not existing.flip_eligible
            ) or (
                anchor.flip_eligible == existing.flip_eligible
                and int(anchor.depth) > int(existing.depth)
            )
            if not new_better:
                return
            self.anchors.pop(existing.anchor_key, None)
        # 血统不随序列化漂移，但重导一次保持与 consider 完全同口径。
        eligible, reasons, lineage = anchor_lineage_status(anchor.entry)
        if not eligible:
            anchor.ineligibility_reasons = tuple(
                dict.fromkeys(
                    list(anchor.ineligibility_reasons)
                    + list(reasons or ("ancestor_not_validated",))
                )
            )
            anchor.flip_eligible = False
        if anchor.prefix_constraint_tables is None:
            anchor.ineligibility_reasons = tuple(
                dict.fromkeys(
                    list(anchor.ineligibility_reasons)
                    + ["prefix_constraint_tables_missing"]
                )
            )
            anchor.flip_eligible = False
        anchor.lineage = lineage
        self._next_sequence += 1
        anchor.sequence = self._next_sequence
        self._touch_counter += 1
        anchor.touch = self._touch_counter
        self._anchor_by_bucket[bucket] = anchor
        self.anchors[anchor.anchor_key] = anchor

    def statistics(self) -> Dict[str, object]:
        hot = sum(1 for a in self.anchors.values() if a.is_hot)
        return {
            "stride": self.stride,
            "hot_limit": self.hot_limit,
            "total_limit": self.total_limit,
            "anchors_total": len(self.anchors),
            "anchors_hot": hot,
            "anchors_cold": len(self.anchors) - hot,
            "flip_eligible": sum(1 for a in self.anchors.values() if a.flip_eligible),
            "max_depth": max((a.depth for a in self.anchors.values()), default=0),
            "fallback_anchors": sum(
                1 for key in self.anchors if key in self._fallback_keys
            ),
            "deepest_below_stride": (
                int(self._deepest_below_stride.get("depth", 0) or 0)
                if self._deepest_below_stride is not None
                else 0
            ),
            "candidate_depth_histogram": dict(
                sorted(
                    self.candidate_depth_histogram.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "stats": dict(self.stats),
            "cold_store": (
                self.cold_store.statistics() if self.cold_store is not None else None
            ),
        }

    def depth_curve(self) -> List[Dict[str, object]]:
        return [
            anchor.describe()
            for anchor in sorted(self.anchors.values(), key=lambda a: a.sequence)
        ]


def _ordered_key(key: Tuple[int, int]) -> Tuple[int, int]:
    return (int(key[0]), int(key[1]))
