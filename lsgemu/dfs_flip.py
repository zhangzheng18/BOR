"""DFS 翻转（r7 P2）：求解 → 回填输入 → 从锚点/入口重放 → 真实执行自证。

用户算法（逐字）：
  2. 到达主干终点后，从最底部的分支点开始，逐个分支点「翻转」（走另一边）
  3. 翻转的实现 = 改变外部 MMIO 输入：求解 → 回填输入 → 从锚点/入口重放
     → 由真实执行证明方向确实变了
  4. 一个分支点两个方向都走过 ⇒ 上退一层（自底向上）
  5. 全程只认「真实可达」：不得用 CPSR/PC 强制、不得改分支方向、不得靠函数摘要跳过

本模块持纯逻辑（规划/护栏/账本）；重放原语在
``HistoricalRunner._dfs_flip_replay_once``（需要大量 runner 内部设施）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .path_naturalization import (
    EVIDENCE_E1,
    EVIDENCE_UNKNOWN,
    derived_replay_evidence,
    match_path_signature,
)

__all__ = [
    "DFSFlipTask",
    "DFSFlipGuardrails",
    "DFSFlipLedger",
    "DFSFlipPlanner",
    "dfs_flip_enabled",
    "dfs_flip_fast_forward_enabled",
    "dfs_flip_max_attempts",
    "dfs_flip_replay_instructions",
    "dfs_flip_timeout_seconds",
    "dfs_preflight_repair_limit",
]


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return int(default)
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def dfs_flip_enabled() -> bool:
    """翻转默认关；e2e/短跑显式开（LSGEMU_DFS_FLIP_ENABLED=1）。"""
    return _env_flag("LSGEMU_DFS_FLIP_ENABLED")


def dfs_flip_max_attempts() -> int:
    return max(0, _env_int("LSGEMU_DFS_FLIP_MAX_ATTEMPTS", 24))


def dfs_flip_replay_instructions() -> int:
    return max(1000, _env_int("LSGEMU_DFS_FLIP_REPLAY_INSTRUCTIONS", 180000))


def dfs_flip_timeout_seconds() -> float:
    return max(0.5, _env_int("LSGEMU_DFS_FLIP_TIMEOUT_SECONDS", 20))


def dfs_flip_fast_forward_enabled() -> bool:
    """r15 D1：翻转重放是否启用确定性循环快转族（默认开）。

    边界：纯内存写 + 确定退出 + 不改控制流方向的环才可 O(1) 物化；
    触 MMIO 的环一律不快转（模拟器侧强制）。快转事件按
    ``loop_fast_forward_emulation`` 家族单列，不计入护栏的方向性强制
    （见 ``DFSFlipGuardrails.evaluate``）。A/B 对照可置
    ``LSGEMU_DFS_FLIP_FAST_FORWARD=0`` 关闭。
    """
    return _env_flag("LSGEMU_DFS_FLIP_FAST_FORWARD", "1")


def dfs_preflight_repair_limit() -> int:
    """口径 1：单次翻转重放允许的 Thumb 状态修复次数上限（默认 4）。

    execution_preflight_repair 是快照编码的确定性函数（重放前把 Thumb 位
    纠正到可执行状态，不选方向），默认按环境事实豁免；但单次重放超过该
    上限说明恢复链异常，该次重放不入账。
    """
    return max(0, _env_int("LSGEMU_DFS_PREFLIGHT_REPAIR_LIMIT", 4))


BranchKey = Tuple[int, int]
PathItem = Tuple[BranchKey, object]


@dataclass
class DFSFlipTask:
    """一次待尝试的翻转：分支点 b 的未尝试方向 dir。"""

    branch_key: BranchKey
    target_direction: object
    depth: int
    anchor_key: object = None
    # 从锚点到 b 的有序后缀（不含 b），重放后需逐项匹配证明仍在主干上。
    suffix_signature: Tuple[PathItem, ...] = field(default_factory=tuple)
    anchor_prefix_items: Tuple[PathItem, ...] = field(default_factory=tuple)


class DFSFlipGuardrails:
    """翻转重放的强制零方向性干预计据（缺一不可，否则该次重放不得入账）。

    r15 D1（用户裁定 docs/DECISIONS_ledger_reclass_20260920.md）：确定性
    循环快转（``loop_fast_forward_emulation`` 家族）属工程优化，豁免于本
    护栏；方向性强制（其余干预家族、循环强转、强制分支表、函数摘要跳过）
    依旧零容忍。
    """

    @staticmethod
    def evaluate(
        *,
        emulator=None,
        run_result: Optional[Dict[str, object]] = None,
    ) -> Tuple[bool, Dict[str, object]]:
        result = run_result or {}
        violations: Dict[str, object] = {}
        intervention_count = int(result.get("intervention_count", 0) or 0)
        if emulator is not None:
            intervention_count = max(
                intervention_count,
                int(getattr(emulator, "intervention_count", 0) or 0),
            )
        # r15 D1：确定性循环快转（loop_fast_forward_emulation 家族）按用户
        # 裁定（docs/DECISIONS_ledger_reclass_20260920.md）属工程优化，不计
        # 方向性强制；其余家族（本地约束/LLM/回退/等待处理/MMIO 调整）与
        # 裸计数（无可依标签）全部照旧拒绝——「不得强制转分支」没有放宽。
        fast_forward_exempt = 0
        if emulator is not None:
            labels = dict(
                getattr(emulator, "intervention_event_labels", None) or {}
            )
            fast_forward_exempt = int(
                labels.get("loop_fast_forward_emulation", 0) or 0
            )
        directional_interventions = max(0, intervention_count - fast_forward_exempt)
        if directional_interventions != 0:
            violations["intervention_count"] = directional_interventions
            if fast_forward_exempt:
                violations["intervention_count_detail"] = {
                    "total": intervention_count,
                    "fast_forward_exempt": fast_forward_exempt,
                }
        forced_trace = int(result.get("forced_branch_trace_count", 0) or 0)
        if forced_trace != 0:
            violations["forced_branch_trace_count"] = forced_trace
        forced_configured = int(
            result.get("forced_branch_choices_configured", 0) or 0
        )
        if forced_configured != 0:
            violations["forced_branch_choices_configured"] = forced_configured
        if emulator is not None:
            loop_forces = getattr(emulator, "runtime_loop_branch_forces", None)
            loop_stats = getattr(emulator, "runtime_loop_branch_force_stats", None) or {}
            # 「装载数」＝实际武装/应用过的循环强转；register_hooks 预装的休眠
            # hook 句柄（零武装、零应用）不算干预。
            installed_loop_forces = (
                len(dict(loop_forces or {}))
                + int(loop_stats.get("installed", 0) or 0)
                + int(loop_stats.get("applied", 0) or 0)
            )
            if installed_loop_forces != 0:
                violations["runtime_loop_branch_force_installed"] = (
                    installed_loop_forces
                )
            skip_stats = getattr(emulator, "skip_function_stats", None) or {}
            nonzero_skips = {
                str(name): int(count)
                for name, count in dict(skip_stats).items()
                if int(count or 0) != 0
            }
            if nonzero_skips:
                violations["skip_function_stats"] = nonzero_skips
            # 口径 1：Thumb 状态修复按环境事实豁免，但单次重放超上限 ⇒ 异常。
            preflight = dict(
                getattr(emulator, "execution_preflight_stats", None) or {}
            )
            thumb_repaired = int(preflight.get("thumb_state_corrected", 0) or 0)
            if thumb_repaired > dfs_preflight_repair_limit():
                violations["execution_preflight_repair_over_limit"] = thumb_repaired
        return (not violations), violations


class DFSFlipLedger:
    """翻转记账：explored / no_external_input / unresolved / proven_unreachable 四类。

    r11 口径 ④：「求解器无候选」≠「输入不可达」。
    - ``no_external_input_directions``：工具限制——当前外部输入模型（MMIO 读值
      overlay）下分支条件数据流锥内无外部输入位点；附锥分类证据。
    - ``unresolved_directions``：有候选、重放过、方向没翻成（含预算耗尽）。
    - ``unreachable_directions``：**仅收有证明的不可达**（proven_unreachable_*）。
      没有约束求解/等价证明之前，「无候选」不得写进该清单。
    """

    def __init__(self) -> None:
        # ((addr, occ), dir, evidence)
        self.explored_direction_edges: List[Tuple[BranchKey, object, str]] = []
        self.seen_edges: set = set()
        self.unreachable_directions: List[Dict[str, object]] = []
        self.no_external_input_directions: List[Dict[str, object]] = []
        self.seen_no_candidate_edges: set = set()
        self.unresolved_directions: List[Dict[str, object]] = []
        self.attempt_stats: Dict[str, int] = {
            "planned": 0,
            "attempts": 0,
            "replays": 0,
            "guardrail_rejections": 0,
            "candidate_rejections": 0,
            "anchor_restore_failures": 0,
            "prefix_divergences": 0,
            "direction_not_changed": 0,
            "flips_succeeded": 0,
            "no_candidates": 0,
            "no_candidate_cached": 0,
        }
        self.attempt_log: List[Dict[str, object]] = []

    def record_success(
        self, branch_key: BranchKey, direction: object, evidence: str
    ) -> None:
        edge = (branch_key, direction, evidence)
        if (branch_key, direction) in self.seen_edges:
            return
        self.seen_edges.add((branch_key, direction))
        self.explored_direction_edges.append(edge)
        self.attempt_stats["flips_succeeded"] += 1

    def record_no_external_input(
        self,
        branch_key: BranchKey,
        direction: object,
        evidence: Optional[Dict[str, object]] = None,
    ) -> None:
        """「无外部输入候选」分类（工具限制，不是不可达）。幂等去重。"""
        edge = (branch_key, direction)
        if edge in self.seen_no_candidate_edges:
            return
        self.seen_no_candidate_edges.add(edge)
        self.no_external_input_directions.append(
            {
                "branch": f"0x{int(branch_key[0]) & 0xFFFFFFFF:08x}",
                "occurrence": int(branch_key[1]),
                "direction": self._serialize_direction(direction),
                "reason": "no_external_input_candidates",
                "evidence": dict(evidence or {}),
            }
        )

    def no_candidate_edge_set(self) -> set:
        """已判「无外部输入候选」的 (key, dir) 集合：翻转 pass 据此跳过重规划。"""
        return set(self.seen_no_candidate_edges)

    def record_unresolved(
        self, branch_key: BranchKey, direction: object, reason: str
    ) -> None:
        """有候选但重放后方向未翻成（含预算耗尽）——非不可达、非无候选。"""
        self.unresolved_directions.append(
            {
                "branch": f"0x{int(branch_key[0]) & 0xFFFFFFFF:08x}",
                "occurrence": int(branch_key[1]),
                "direction": self._serialize_direction(direction),
                "reason": str(reason or "unknown"),
            }
        )

    def record_unreachable(
        self, branch_key: BranchKey, direction: object, proof: str
    ) -> None:
        """仅收**有证明**的不可达（proof 须以 proven_unreachable 开头并说明方法）。

        没有证明的方向请走 record_no_external_input（工具限制）或
        record_unresolved（重放失败）；把两者混进本清单会破坏论文口径。
        """
        normalized = str(proof or "")
        if not normalized.startswith("proven_unreachable"):
            raise ValueError(
                "unreachable_directions requires a proof "
                f"(proven_unreachable_*); got: {normalized!r}"
            )
        self.unreachable_directions.append(
            {
                "branch": f"0x{int(branch_key[0]) & 0xFFFFFFFF:08x}",
                "occurrence": int(branch_key[1]),
                "direction": self._serialize_direction(direction),
                "reason": normalized,
            }
        )

    @staticmethod
    def _serialize_direction(direction: object) -> object:
        if isinstance(direction, bool):
            return "taken" if direction else "fallthrough"
        return direction

    def record_attempt(self, entry: Dict[str, object]) -> None:
        self.attempt_log.append(entry)
        if len(self.attempt_log) > 512:
            del self.attempt_log[:256]

    def report_payload(self) -> Dict[str, object]:
        return {
            "explored_direction_edges": [
                {
                    "branch": f"0x{int(key[0]) & 0xFFFFFFFF:08x}",
                    "occurrence": int(key[1]),
                    "direction": self._serialize_direction(direction),
                    "evidence": evidence,
                }
                for key, direction, evidence in self.explored_direction_edges
            ],
            "unreachable_directions": list(self.unreachable_directions),
            "no_external_input_directions": list(self.no_external_input_directions),
            "unresolved_directions": list(self.unresolved_directions),
            "direction_counts": {
                "explored": len(self.explored_direction_edges),
                "no_external_input": len(self.no_external_input_directions),
                "unresolved": len(self.unresolved_directions),
                "proven_unreachable": len(self.unreachable_directions),
            },
            "attempt_stats": dict(self.attempt_stats),
            "attempts": list(self.attempt_log),
        }


class DFSFlipPlanner:
    """自底向上规划：主干分支点按深度降序，逐点产出未尝试方向。"""

    @staticmethod
    def plan(
        trunk_items: Sequence[PathItem],
        *,
        anchor_by_index: Optional[Dict[int, object]] = None,
        already_explored: Optional[set] = None,
        trunk_directions: Optional[Dict[BranchKey, object]] = None,
    ) -> List[DFSFlipTask]:
        """从深到浅列出翻转任务。

        ``trunk_items``：有序主干 ((addr,occ), choice) 列表（choice=主干方向）。
        ``anchor_by_index``：trunk 索引 → 可用祖先锚点（该索引之前最近的锚点）。
        ``already_explored``：{(addr, occ), dir} 已真实走过 ⇒ 跳过。
        ``trunk_directions``：锚点 next_branch_key 的主干方向（bool）。
        """
        explored = already_explored or set()
        directions = dict(trunk_directions or {})
        tasks: List[DFSFlipTask] = []
        # 深到浅：含锚点自身的 next_branch_key（最深可达点，depth=len(trunk)）。
        ordered_points: List[Tuple[BranchKey, object, int]] = [
            (key, choice, index) for index, (key, choice) in enumerate(trunk_items)
        ]
        for key, choice in trunk_items:
            if isinstance(choice, bool) and key not in directions:
                directions[key] = choice
        # next_branch_key 点（若提供）排在最深处。
        for key, direction in (directions or {}).items():
            if all(key != point_key for point_key, _c, _i in ordered_points):
                ordered_points.append((key, direction, len(trunk_items)))
        ordered_points.sort(key=lambda item: -item[2])
        for key, trunk_choice, index in ordered_points:
            if not isinstance(trunk_choice, bool):
                continue  # 非二方向（switch/indirect）不适用「走另一边」
            other = not trunk_choice
            if (key, trunk_choice) not in explored and (
                key,
                other,
            ) not in explored:
                # 主干方向本身尚未真实记账时优先补记（通常已由主干执行覆盖）。
                tasks.append(
                    DFSFlipTask(
                        branch_key=key,
                        target_direction=trunk_choice,
                        depth=index,
                    )
                )
            if (key, other) in explored:
                continue
            anchor = (anchor_by_index or {}).get(index)
            suffix = tuple(trunk_items[:index]) if anchor is None else tuple(())
            prefix_items = tuple(trunk_items[:index])
            if anchor is not None:
                anchor_prefix_len = int(getattr(anchor, "anchor_prefix_len", 0) or 0)
                prefix_items = tuple(trunk_items[:anchor_prefix_len])
                suffix = tuple(trunk_items[anchor_prefix_len:index])
            tasks.append(
                DFSFlipTask(
                    branch_key=key,
                    target_direction=other,
                    depth=index,
                    anchor_key=getattr(anchor, "anchor_key", None),
                    suffix_signature=suffix,
                    anchor_prefix_items=prefix_items,
                )
            )
        return tasks


def verify_flip_by_real_execution(
    task: DFSFlipTask,
    replay_events: Sequence[object],
) -> Tuple[str, Dict[str, object]]:
    """自证：真实执行轨迹里 b 前的主干后缀逐项吻合，且 b 实际走了 dir。

    返回 (verdict, detail)：verdict ∈
      flipped / direction_not_changed / prefix_diverged /
      branch_occurrence_not_reached / 其他 mismatch 原因
    """
    expected: List[PathItem] = [
        *((key, direction) for key, direction in (task.suffix_signature or tuple())),
        (task.branch_key, task.target_direction),
    ]
    match = match_path_signature(tuple(expected), list(replay_events or ()))
    detail = {
        "matched": int(match.matched_prefix_len),
        "total": int(match.expected_count),
        "mismatch_reason": match.mismatch_reason,
        "observed_choice": match.observed_choice,
    }
    if match.complete:
        return "flipped", detail
    reason = str(match.mismatch_reason or "")
    if reason == "wrong_natural_choice":
        # 到达了 b 但方向没变（或后缀中途拐弯——matched 指明位置）。
        if match.mismatch_key == task.branch_key:
            return "direction_not_changed", detail
        return "prefix_diverged", detail
    if reason == "branch_occurrence_not_reached":
        if match.mismatch_key == task.branch_key:
            return "branch_occurrence_not_reached", detail
        return "prefix_diverged", detail
    return reason or "unmatched", detail


def flip_edge_evidence(
    *, forced_control: bool = False, consumed_environment_fact: bool = True
) -> str:
    """翻转成功边的证据等级：护栏全零 + 候选被真实消费 ⇒ E1。"""
    if forced_control:
        # 理论不可达：护栏层已拒绝入账；仅供诊断调用。
        return derived_replay_evidence(
            EVIDENCE_UNKNOWN, forced_control=True
        )
    return derived_replay_evidence(
        EVIDENCE_E1, consumed_environment_fact=consumed_environment_fact
    )
