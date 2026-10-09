#!/usr/bin/env python3
"""Obligation-scoped concrete input recovery for ARM branch predicates.

This module deliberately does not execute firmware and does not force control
flow.  It builds a small bit-vector slice from one external read to the first
unmatched branch, then returns concrete input hypotheses.  The caller remains
responsible for validating every hypothesis with a force-free replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .runner_models import constraint_delivery_site_identity

try:
    import z3  # type: ignore
except Exception:  # pragma: no cover - exercised on minimal deployments
    z3 = None


U32_MASK = 0xFFFFFFFF

# r42：候选生成消融臂（C2 配对对照）。通用候选策略的固定 PRNG 种子——
# 提前写死、可复现，不因目标/站点而变。
GENERIC_CANDIDATE_PRNG_SEED = 0xC0FFEE

_GENERIC_ENV_SWITCH = "LSGEMU_DISABLE_CONSTRAINT_GUIDED_CANDIDATES"


def env_flag_enabled(name: str, default: str = "") -> bool:
    """Truthiness of an environment switch using the project's 1/true/yes/on set."""
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def generic_candidate_values(
    width: int,
    *,
    seed_values: Iterable[int] = (),
    limit: int = 4,
    avoid: Iterable[int] = (),
    prng_seed: int = GENERIC_CANDIDATE_PRNG_SEED,
) -> List[Tuple[int, str]]:
    """Fixed generic candidate strategy for the candidate-generation ablation.

    Deliberately closed over the target predicate: the only inputs are the
    recovered access width, runtime-observed seed values, the shared candidate
    limit, and previously failed values.  The signature must never accept (and
    callers must never pass) the symbolic problem, compare immediates, masks,
    or partner constants — that is the red line of the ablation design.

    Strategy order is fixed in advance: width-derived boundary band, then a
    fixed mutation sequence over the observed seeds, then seeded random
    sampling.  Values already emitted or marked avoid (failed attempts at the
    same site) are skipped, so the sequence advances deeper across retries.
    """
    width = min(32, max(1, int(width or 32)))
    mask = (1 << width) - 1 if width < 32 else U32_MASK
    avoid_values = {int(value) & mask for value in avoid or ()}
    emitted: List[Tuple[int, str]] = []
    seen = set()

    def offer(value: object, strategy: str) -> bool:
        try:
            normalized = int(value) & mask
        except (TypeError, ValueError):
            return False
        if normalized in seen or normalized in avoid_values:
            return False
        seen.add(normalized)
        emitted.append((normalized, strategy))
        return len(emitted) >= max(1, int(limit or 1))

    # 1. Boundary values for width W (fixed table; independent of the target).
    boundaries: List[Tuple[int, str]] = [
        (0, "generic_boundary_zero"),
        (1, "generic_boundary_one"),
        (2, "generic_boundary_two"),
        (mask, "generic_boundary_max_all_ones"),
        (mask - 1, "generic_boundary_max_minus_one"),
        (1 << (width - 1), "generic_boundary_sign_bit"),
        (1 << (width // 2), "generic_boundary_half_position_bit"),
        (width // 2, "generic_boundary_half_width_value"),
    ]
    for value, strategy in boundaries:
        if offer(value, strategy):
            return emitted

    # 2. Seed mutation over runtime-observed values (fixed sequence).
    seeds: List[int] = []
    for value in seed_values or ():
        try:
            normalized = int(value) & mask
        except (TypeError, ValueError):
            continue
        if normalized not in seeds:
            seeds.append(normalized)
    byte_count = max(1, width // 8)
    for seed in seeds:
        if offer((seed - 1) & mask, "generic_seed_minus_one"):
            return emitted
        if offer((seed + 1) & mask, "generic_seed_plus_one"):
            return emitted
        if offer((seed - 2) & mask, "generic_seed_minus_two"):
            return emitted
        if offer((seed + 2) & mask, "generic_seed_plus_two"):
            return emitted
        for k in range(0, min(width, 8)):
            if offer((seed - (1 << k)) & mask, "generic_seed_minus_pow2"):
                return emitted
            if offer((seed + (1 << k)) & mask, "generic_seed_plus_pow2"):
                return emitted
        for bit in range(width):
            if offer(seed ^ (1 << bit), "generic_seed_bit_flip"):
                return emitted
        for byte_index in range(byte_count):
            shift = 8 * byte_index
            byte_mask = 0xFF << shift
            if offer(seed & ~byte_mask & mask, "generic_seed_byte_zero"):
                return emitted
            if offer(seed | byte_mask, "generic_seed_byte_ones"):
                return emitted
            if offer(seed ^ byte_mask, "generic_seed_byte_invert"):
                return emitted

    # 3. Seeded random sampling (fixed PRNG; reproducible across runs).  The
    # draw count is bounded so a fully exhausted small domain cannot spin.
    prng = random.Random(int(prng_seed))
    max_draws = 64 * max(1, int(limit or 1)) + 64
    draws = 0
    while len(emitted) < max(1, int(limit or 1)) and draws < max_draws:
        draws += 1
        offer(prng.getrandbits(width), "generic_random_sample")
    return emitted


# ---------------------------------------------------------------------------
# ThumbExpandImm_C 全模式解码（r19 P0，设计 /tmp/dfs_r18b.md §5.2.4）
#
# ARM ARM DDI 0403E.e A5.3.2 ThumbExpandImm_C 伪代码（原文，/tmp/r18b 提取）：
#   if imm12<11:10> == '00' then
#       case imm12<9:8> of            -- imm8 = imm12<7:0>
#           '00': imm32 = imm8        -- 未旋转
#           '01': imm32 = 0x00XY00XY  -- imm8 复制到 23:16 与 7:0
#           '10': imm32 = 0xXY00XY00  -- imm8 复制到 31:24 与 15:8
#           '11': imm32 = 0xXYXYXYXY  -- imm8 复制到全部四字节
#       carry_out = carry_in          -- 非旋转形不写 C
#   else
#       unrotated_value = ZeroExtend('1':imm12<6:0>, 32)   -- 低 8 位
#       (imm32, carry_out) = ROR_C(unrotated_value, UInt(imm12<11:7>))
# ROR_C 的 carry_out = result<31>（DDI 0403E Appx：carry_out = result<N-1> 对
# N=32 即 bit31），与 A5.3.2 表注「C = 修改后立即数 bit[31]」一致。
# 前向解码对账 capstone：ardupilot_Pixhawk1 全部 1091 条 .w imm 形 DP 指令
# 零失配；imm32→imm12 反解在可编码域（4093/4096，剔除 3 个 UNPREDICTABLE
# 零编码）上单射——非旋转复制形与旋转形值域不相交。
# ---------------------------------------------------------------------------

def expand_imm_c(imm12, carry_in=None):
    """ThumbExpandImm_C：返回 (imm32, carry)。

    carry 为 None 表示非旋转形（C 不变）；0/1 为旋转形的解码期常量
    （= imm32 bit[31]）。与 prepared_firmware._producer_imm12_from_value 互逆。
    """
    imm12 = int(imm12) & 0xFFF
    if (imm12 >> 10) & 3 == 0:
        imm8 = imm12 & 0xFF
        mode = (imm12 >> 8) & 3
        if mode == 0:
            value = imm8
        elif mode == 1:
            value = (imm8 << 16) | imm8
        elif mode == 2:
            value = (imm8 << 24) | (imm8 << 8)
        else:
            value = imm8 * 0x01010101
        return value & U32_MASK, None
    rotation = (imm12 >> 7) & 0x1F
    unrotated = 0x80 | (imm12 & 0x7F)
    value = ((unrotated >> rotation) | (unrotated << (32 - rotation))) & U32_MASK
    return value, (value >> 31) & 1


def imm12_from_value(value):
    """imm32 → imm12 反解（可编码域上单射）；不可编码返回 None。"""
    value = int(value) & U32_MASK
    low = value & 0xFF
    if value == low:
        return low
    if low and value == ((low << 16) | low):
        return 0x100 | low
    mid = (value >> 8) & 0xFF
    if mid and value == ((mid << 24) | (mid << 8)):
        return 0x200 | mid
    if low and value == low * 0x01010101:
        return 0x300 | low
    for shift in range(1, 25):
        if value & ((1 << shift) - 1):
            continue
        unit = (value >> shift) & 0xFF
        if unit >= 0x80 and ((unit << shift) & U32_MASK) == value:
            return ((32 - shift) << 7) | (unit & 0x7F)
    return None


@dataclass(frozen=True)
class ValueHypothesis:
    value: int
    strategy: str
    exact_local_model: bool = False

    def normalized(self, width: int = 32) -> "ValueHypothesis":
        effective_width = min(32, max(1, int(width or 32)))
        mask = (1 << effective_width) - 1 if effective_width < 32 else U32_MASK
        return ValueHypothesis(
            value=int(self.value) & mask,
            strategy=str(self.strategy or "unknown"),
            exact_local_model=bool(self.exact_local_model),
        )


@dataclass(frozen=True)
class LocalRecoveryResult:
    hypotheses: Tuple[ValueHypothesis, ...]
    symbolic_slice: bool
    solver_backend: str
    source_width: int
    reason: str
    slice_description: str = ""
    # r42：候选生成模式审计位（constraint_guided=谓词规则+Z3 现状；
    # generic=固定通用策略的消融对照）。
    candidate_generation_mode: str = "constraint_guided"


@dataclass(frozen=True)
class ReplayFailure:
    category: str
    retry_values: bool
    retry_site: bool
    retry_compound: bool


@dataclass(frozen=True)
class _SymbolicProblem:
    variable: object
    predicate: object
    source_width: int
    description: str
    # r19：切片内最近比较立即数（种子表用）与产生者 pc（描述/诊断用）。
    compare_imm: Optional[int] = None
    producer_pcs: Tuple[int, ...] = ()


def constraint_site_identity(candidate) -> Tuple[object, ...]:
    """Return the dynamic external-read identity without its candidate value."""
    return constraint_delivery_site_identity(candidate)


def classify_replay_failure(
    evaluation: Optional[Dict[str, object]],
    constraint_feedback: Optional[Dict[str, object]],
    *,
    previous_prefix_len: int = 0,
    result_prefix_len: int = 0,
    previous_discovery_prefix_len: int = 0,
    result_discovery_prefix_len: int = 0,
) -> ReplayFailure:
    """Classify a rejected candidate into a scheduling action.

    The classification is intentionally conservative.  In particular, a value
    mutation is not retried when the configured read was never consumed.
    """
    evaluation = dict(evaluation or {})
    feedback = dict(constraint_feedback or {})
    all_consumed = bool(feedback.get("all_constraint_reads_matched"))
    address_reads = int(feedback.get("matched_address_reads", 0) or 0)
    exact_reads = int(feedback.get("matched_constraint_reads", 0) or 0)
    requested_occurrences = int(
        feedback.get("matched_requested_occurrences", 0) or 0
    )
    delivery_mismatches = int(
        feedback.get("value_delivery_mismatches", 0) or 0
    )

    if not all_consumed:
        if delivery_mismatches > 0 or requested_occurrences > exact_reads:
            return ReplayFailure(
                "configured_value_not_materialized",
                retry_values=False,
                retry_site=True,
                retry_compound=False,
            )
        if address_reads > requested_occurrences:
            return ReplayFailure(
                "wrong_read_occurrence",
                retry_values=False,
                retry_site=True,
                retry_compound=False,
            )
        return ReplayFailure(
            "input_site_not_reached_or_context_missing",
            retry_values=False,
            retry_site=True,
            retry_compound=False,
        )

    if (
        int(result_prefix_len) < int(previous_prefix_len)
        or int(result_discovery_prefix_len) < int(previous_discovery_prefix_len)
    ):
        return ReplayFailure(
            "upstream_prefix_interference",
            retry_values=True,
            retry_site=True,
            retry_compound=False,
        )

    result_match = dict(evaluation.get("result_match", {}) or {})
    mismatch_reason = str(result_match.get("mismatch_reason") or "")
    if mismatch_reason == "wrong_natural_choice":
        return ReplayFailure(
            "consumed_but_predicate_unchanged",
            retry_values=True,
            retry_site=False,
            retry_compound=True,
        )
    if mismatch_reason == "branch_occurrence_not_reached":
        return ReplayFailure(
            "consumed_but_downstream_context_missing",
            retry_values=True,
            retry_site=True,
            retry_compound=True,
        )

    reason = str(evaluation.get("reason") or "candidate_rejected")
    if reason in {"target_not_reached", "candidate_no_target_delta"}:
        return ReplayFailure(
            "predicate_or_downstream_state_unsatisfied",
            retry_values=True,
            retry_site=False,
            retry_compound=True,
        )
    return ReplayFailure(
        "candidate_rejected",
        retry_values=True,
        retry_site=False,
        retry_compound=True,
    )


class LocalConstraintRecovery:
    """Recover a bounded concrete domain for one external input occurrence."""

    _LOAD_WIDTHS = {
        "LDR": (32, False),
        "LDRB": (8, False),
        "LDRH": (16, False),
        "LDRSB": (8, True),
        "LDRSH": (16, True),
    }

    # These S-form ALU writers set N/Z from the computed result, so a branch on
    # N/Z is solvable from the slice without any CMP/TST (r16: the
    # `ands r2,r2,#imm; bne` and `lsls r2,r2,#imm; bpl` peripheral-poll
    # families previously died with `missing_compare_instruction`).
    # r19：补 SBCS/ADCS/ORNS（符号标志槽全面接管谓词构建后，此表只用于
    # 索引未命中时的降级启发 `_find_flag_setting_instruction`）。
    _FLAG_SETTING_MNEMONICS = {
        "ANDS", "BICS", "ORRS", "ORNS", "EORS", "LSLS", "LSRS", "ASRS", "RORS",
        "MOVS", "MVNS", "ADDS", "SUBS", "RSBS", "SBCS", "ADCS", "MULS",
    }

    # r19：NZCV 产生者语义表（与 prepared_firmware._producer_flag_writes 一致）。
    # 32 位乘法族永不写标志；MULS 仅 IT 块外写 N/Z；CLZ/SXTB/REV*/SSAT/USAT
    # 不写（不在表中自动排除）。
    _ARITH_FLAG_MNEMONICS = frozenset({
        "CMP", "CMN", "ADDS", "ADCS", "SUBS", "SBCS", "RSBS",
        "RSBCS", "RSBCSS", "RSCS",  # Ghidra 对 RSC S 形的个别渲染
    })
    _LOGIC_FLAG_MNEMONICS = frozenset({
        "TST", "TEQ", "ANDS", "ORRS", "EORS", "BICS", "ORNS", "MOVS", "MVNS",
    })
    _SHIFT_FLAG_MNEMONICS = frozenset({"LSLS", "LSRS", "ASRS", "RORS", "RRXS"})
    _EXPLICIT_FLAG_MNEMONICS = frozenset({"VMRS", "MSR"})

    # 逐条件码标志需求（与 prepared_firmware._PRODUCER_COND_FLAGS 一致）
    _COND_FLAG_REQUIREMENTS = {
        "BEQ": "Z", "BNE": "Z",
        "BCS": "C", "BHS": "C", "BCC": "C", "BLO": "C",
        "BMI": "N", "BPL": "N", "BVS": "V", "BVC": "V",
        "BHI": "CZ", "BLS": "CZ",
        "BGE": "NV", "BLT": "NV", "BGT": "NZV", "BLE": "NZV",
    }

    _SHIFT_SUFFIX = re.compile(r"^(lsl|lsr|asr|ror|rrx)\b", re.IGNORECASE)

    def __init__(
        self,
        static_bbs: Dict[int, List[Dict[str, object]]],
        *,
        instruction_to_bb: Optional[Dict[int, int]] = None,
        compare_lookup: Optional[Dict[int, Dict[str, object]]] = None,
        producer_index: Optional[Dict[int, object]] = None,
        solver_timeout_ms: int = 40,
        generic_candidates: Optional[bool] = None,
    ) -> None:
        self.static_bbs = static_bbs or {}
        self.instruction_to_bb = instruction_to_bb or self._build_instruction_to_bb()
        self.compare_lookup = compare_lookup or {}
        # r19：NZCV 产生者索引（prepared_firmware._build_producer_index 产物）。
        # 缺省 None（离线/单测环境）时走 legacy 降级路径。
        self.producer_index = producer_index or None
        self.solver_timeout_ms = max(1, int(solver_timeout_ms or 40))
        # r42：generic_candidates=True 时 recover_values 走固定通用策略，
        # 不构造/求解任何符号问题。None 时回落到等价环境开关。
        self.generic_candidates = (
            env_flag_enabled(_GENERIC_ENV_SWITCH)
            if generic_candidates is None
            else bool(generic_candidates)
        )

    def recover_values(
        self,
        *,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        read_pc: Optional[int],
        primary_values: Iterable[int] = (),
        observed_values: Iterable[int] = (),
        avoid_values: Iterable[int] = (),
        max_values: int = 4,
        compare_partner_values: Iterable[int] = (),
    ) -> LocalRecoveryResult:
        limit = max(1, int(max_values or 1))
        avoid = {int(value) & U32_MASK for value in avoid_values or ()}
        if self.generic_candidates:
            # r42 消融臂：只替换候选生成——出现身份与恢复出的宽度照旧，
            # 候选来自 generic_candidate_values 的固定通用策略。
            return self._recover_generic_values(
                read_pc=read_pc,
                observed_values=observed_values,
                avoid_values=avoid,
                max_values=limit,
            )
        problem, build_reason = self._build_symbolic_problem(
            int(branch_pc),
            str(branch_condition or ""),
            bool(target_direction),
            int(read_pc) if read_pc is not None else None,
        )
        if problem is None:
            problem = self._build_direct_problem(
                int(branch_pc),
                str(branch_condition or ""),
                bool(target_direction),
            )
            symbolic_slice = False
        else:
            symbolic_slice = True

        width = int(problem.source_width if problem is not None else 32)
        mask = (1 << width) - 1 if width < 32 else U32_MASK
        ordered_seeds: List[Tuple[int, str]] = []

        def seed(value: object, strategy: str) -> None:
            try:
                normalized = int(value) & mask
            except Exception:
                return
            if all(existing != normalized for existing, _source in ordered_seeds):
                ordered_seeds.append((normalized, strategy))

        # r16: values reverse-solved from the branch predicate come first —
        # they are the physically plausible bit patterns the condition asks
        # for (e.g. bit17 for `lsls #14; bpl`), ahead of rule heuristics.
        for value, strategy in self._condition_derived_values(problem):
            seed(value, strategy)
        # r22: register-register compares against a statically resolved
        # constant partner (literal pool / constant chain, supplied by the
        # caller from the precise-branch reverse window).  For an equality
        # compare the partner constant is *the* physically plausible input
        # value, so it seeds ahead of the rule heuristics, same layer as the
        # r16 condition-derived values.  No guessing happens here: values
        # without a static provenance are never passed in.
        for value in compare_partner_values or ():
            seed(value, "literal_partner_register")
        for value in primary_values or ():
            seed(value, "primary_inference")
        for value in observed_values or ():
            seed(value, "observed_environment_value")
        # r16 fallback set: complement of every observed read (a status
        # register that never changed is the classic un-flipped poll input).
        for value in observed_values or ():
            seed((~int(value)) & mask, "observed_environment_complement")
        for value, strategy in self._semantic_boundary_values(
            int(branch_pc),
            str(branch_condition or ""),
            bool(target_direction),
            compare_imm=(problem.compare_imm if problem is not None else None),
            compare_partner_values=tuple(compare_partner_values or ()),
        ):
            seed(value, strategy)

        hypotheses: List[ValueHypothesis] = []
        seen = set()

        def accept(value: int, strategy: str, exact: bool) -> None:
            normalized = int(value) & mask
            if normalized in avoid or normalized in seen or len(hypotheses) >= limit:
                return
            seen.add(normalized)
            hypotheses.append(
                ValueHypothesis(normalized, strategy, exact).normalized(width)
            )

        for value, strategy in ordered_seeds:
            if problem is not None and not self._problem_accepts(problem, value):
                continue
            accept(value, strategy, bool(symbolic_slice))

        solver_backend = "z3" if z3 is not None else "unavailable"
        if problem is not None and len(hypotheses) < limit:
            for value, strategy in self._solver_models(
                problem,
                excluded=avoid | seen,
                max_values=limit - len(hypotheses),
            ):
                accept(value, strategy, bool(symbolic_slice))

        # Preserve the old rule candidate when the slice is unsupported.  It is
        # still only a hypothesis and cannot receive coverage credit by itself.
        if not hypotheses and problem is None:
            for value in primary_values or ():
                accept(int(value), "primary_unvalidated_fallback", False)
                if hypotheses:
                    break
        if not hypotheses and problem is None:
            for value in (0, 1, mask):
                accept(value, "bounded_unknown_fallback", False)

        return LocalRecoveryResult(
            hypotheses=tuple(hypotheses),
            symbolic_slice=bool(symbolic_slice),
            solver_backend=solver_backend,
            source_width=width,
            reason=("symbolic_slice" if symbolic_slice else build_reason),
            slice_description=(problem.description if problem is not None else ""),
            candidate_generation_mode="constraint_guided",
        )

    def _recover_generic_values(
        self,
        *,
        read_pc: Optional[int],
        observed_values: Iterable[int],
        avoid_values: Iterable[int],
        max_values: int,
    ) -> LocalRecoveryResult:
        """Candidate-generation ablation arm: fixed generic strategy only.

        Same occurrence and same recovered width as the guided arm; the
        symbolic problem, predicate constants, and compare immediates are
        never constructed or consulted here (``generic_candidate_values``
        cannot even receive them by signature).
        """
        width = self._external_read_source_width(read_pc)
        mask = (1 << width) - 1 if width < 32 else U32_MASK
        hypotheses = [
            ValueHypothesis(value, strategy, False).normalized(width)
            for value, strategy in generic_candidate_values(
                width,
                seed_values=observed_values,
                limit=max_values,
                avoid={int(value) & mask for value in avoid_values or ()},
            )
        ]
        return LocalRecoveryResult(
            hypotheses=tuple(hypotheses),
            symbolic_slice=False,
            solver_backend="not_consulted",
            source_width=width,
            reason="generic_candidates",
            candidate_generation_mode="generic",
        )

    def _external_read_source_width(self, read_pc: Optional[int]) -> int:
        """Recovered access width of the external read (range recovery).

        Static instruction lookup only — deliberately shares nothing with
        ``_build_symbolic_problem`` beyond the read's load mnemonic, so the
        generic arm keeps the recovered width without building a predicate.
        """
        if read_pc is None:
            return 32
        bb_start = self.instruction_to_bb.get(int(read_pc))
        if bb_start is None:
            return 32
        for insn in self.static_bbs.get(bb_start, ()) or ():
            try:
                if int(insn.get("address", 0) or 0) != int(read_pc):
                    continue
            except (TypeError, ValueError):
                continue
            width_signed = self._LOAD_WIDTHS.get(
                self._normalize_mnemonic(insn.get("mnemonic"))
            )
            return int(width_signed[0]) if width_signed else 32
        return 32

    def _build_instruction_to_bb(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for bb_start, instructions in self.static_bbs.items():
            for insn in instructions or ():
                try:
                    lookup[int(insn.get("address", 0) or 0)] = int(bb_start)
                except Exception:
                    continue
        return lookup

    @staticmethod
    def _normalize_mnemonic(value: object) -> str:
        raw = str(value or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B" and parts[1] not in {"N", "W", "NW"}:
            return f"B{parts[1]}"
        return parts[0]

    @staticmethod
    def _split_operands(value: object) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for char in str(value or ""):
            if char in "[{":
                depth += 1
            elif char in "]}":
                depth = max(0, depth - 1)
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    @staticmethod
    def _register(value: object) -> Optional[str]:
        match = re.fullmatch(
            r"(?:r(?:1[0-5]|[0-9])|sp|lr|pc)",
            str(value or "").strip().lower(),
        )
        return match.group(0) if match else None

    @staticmethod
    def _immediate(value: object) -> Optional[int]:
        text = str(value or "").strip().lower()
        if text.startswith("#"):
            text = text[1:]
        if not re.fullmatch(r"[-+]?(?:0x[0-9a-f]+|\d+)", text):
            return None
        try:
            return int(text, 0) & U32_MASK
        except ValueError:
            return None

    def _operand_expr(self, operand: object, registers: Dict[str, object]):
        immediate = self._immediate(operand)
        if immediate is not None:
            return z3.BitVecVal(immediate, 32)
        register = self._register(operand)
        if register is not None:
            return registers.get(register)
        return None

    @staticmethod
    def _source_expr(variable, width: int, signed: bool):
        if width >= 32:
            return variable
        low = z3.Extract(width - 1, 0, variable)
        return z3.SignExt(32 - width, low) if signed else z3.ZeroExt(32 - width, low)

    def _build_symbolic_problem(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        read_pc: Optional[int],
    ) -> Tuple[Optional[_SymbolicProblem], str]:
        if z3 is None:
            return None, "z3_unavailable"
        if read_pc is None:
            return None, "missing_read_pc"
        bb_start = self.instruction_to_bb.get(int(branch_pc))
        if bb_start is None or self.instruction_to_bb.get(int(read_pc)) != bb_start:
            return None, "read_and_branch_not_in_same_basic_block"
        instructions = list(self.static_bbs.get(bb_start, ()) or ())
        indexes = {
            int(insn.get("address", 0) or 0): index
            for index, insn in enumerate(instructions)
        }
        read_index = indexes.get(int(read_pc))
        branch_index = indexes.get(int(branch_pc))
        if read_index is None or branch_index is None or read_index >= branch_index:
            return None, "invalid_local_slice_order"

        read_insn = instructions[read_index]
        read_mnemonic = self._normalize_mnemonic(read_insn.get("mnemonic"))
        width_signed = self._LOAD_WIDTHS.get(read_mnemonic)
        read_parts = self._split_operands(read_insn.get("operands"))
        read_dest = self._register(read_parts[0]) if read_parts else None
        if width_signed is None or read_dest is None:
            return None, "unsupported_external_read_instruction"
        source_width, signed = width_signed
        variable = z3.BitVec("external_input", 32)
        registers: Dict[str, object] = {}
        flags: Dict[str, object] = {}
        compare_imm: Optional[int] = None

        condition = self._normalize_mnemonic(branch_condition)
        record = None
        legacy_compare_pc: Optional[int] = None
        if condition not in {"CBZ", "CBNZ"} and self.producer_index:
            record = self.producer_index.get(int(branch_pc))
        if record is not None:
            # r19：索引优先（r18b §5.2.2）。索引 miss 的 Bcc 站点（运行时吸收的
            # BB 等）走下方 legacy 路径，与旧行为一致。
            producers, paths, calls_between = record
            if paths == "label_fp_opaque":
                return None, "fp_opaque_producer"
            if paths == "label_msr":
                return None, "msr_flag_producer"
            if calls_between:
                return None, "cross_call_flag_producer"
            if any(producer[4] for producer in producers):
                return None, "unsupported_predicated_slice"
            if paths == "partial":
                return None, "producer_path_dependent"
            if paths == "live_in":
                return None, "flags_live_in"
            if paths == "method_limited":
                return None, "producer_window_exceeded"
            slice_pcs = {
                int(insn.get("address", 0) or 0)
                for insn in instructions[read_index + 1:branch_index]
            }
            for producer in producers:
                if int(producer[0]) not in slice_pcs:
                    # 产生者在读之前（含跨 BB all_covered）——标志不依赖本次
                    # 外部读，静态反解不可行。
                    return None, "producer_outside_local_slice"
        else:
            # legacy 降级（索引未命中）：沿用 compare_lookup/最近设标志指令
            # 定位比较，仅用于描述与种子；谓词本身由标志槽统一构建。
            compare_insn = self.compare_lookup.get(int(branch_pc))
            if compare_insn is None:
                compare_insn = self._find_flag_setting_instruction(
                    instructions, read_index, branch_index
                )
            if compare_insn is None and condition not in {"CBZ", "CBNZ"}:
                return None, "missing_compare_instruction"
            if compare_insn is not None:
                legacy_compare_pc = int(compare_insn.get("address", 0) or 0)

        # IT 块谓词化执行静态不确定：切片前缀出现任何 IT 即拒绝（r16 起政策，
        # 对齐 :439 的 unsupported_predicated_slice；索引模式下产生者的 in_it
        # 已在上方先行拒绝）。
        for insn in instructions[:branch_index]:
            if self._normalize_mnemonic(insn.get("mnemonic")).startswith("IT"):
                return None, "unsupported_predicated_slice"

        for index in range(branch_index):
            insn = instructions[index]
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
            parts = self._split_operands(insn.get("operands"))
            if not parts:
                continue
            dest = self._register(parts[0])
            pc = int(insn.get("address", 0) or 0)
            if pc == int(read_pc):
                registers[read_dest] = self._source_expr(variable, source_width, signed)
                continue
            # 标志槽先于寄存器写：2 操作数 S 形（adds r0,#1）的左值是旧 dest。
            applied_imm = self._apply_flag_effects(
                insn, mnemonic, parts, registers, flags
            )
            if applied_imm is not None:
                compare_imm = applied_imm
            if mnemonic in self._LOAD_WIDTHS:
                if dest is not None:
                    registers.pop(dest, None)
                continue
            self._apply_symbolic_instruction(mnemonic, parts, registers)

        if condition in {"CBZ", "CBNZ"}:
            branch_parts = self._split_operands(instructions[branch_index].get("operands"))
            checked = self._operand_expr(branch_parts[0], registers) if branch_parts else None
            if checked is None:
                return None, "unresolved_cbz_operand"
            taken = checked == z3.BitVecVal(0, 32)
            if condition == "CBNZ":
                taken = z3.Not(taken)
        else:
            taken = self._condition_from_flags(condition, flags)
            if taken is None:
                if not flags:
                    # 与旧口径同名：切片内没有任何设标志指令
                    return None, "missing_compare_instruction"
                # 与旧口径同名：有产生者但所需标志位的值不可静态导出
                #（C 未写保持旧值、读前定义、寄存器移位置等）
                return None, "unsupported_compare_slice"

        predicate = taken if target_direction else z3.Not(taken)
        anchor_pc = int(branch_pc)
        if record is not None and record[0]:
            anchor_pc = int(record[0][0][0])
        elif legacy_compare_pc is not None:
            anchor_pc = legacy_compare_pc
        description = (
            f"0x{int(read_pc) & U32_MASK:08x}->"
            f"0x{int(anchor_pc) & U32_MASK:08x}->"
            f"0x{int(branch_pc) & U32_MASK:08x}:{condition}:"
            f"{'taken' if target_direction else 'not_taken'}"
        )
        return (
            _SymbolicProblem(
                variable,
                predicate,
                source_width,
                description,
                compare_imm=compare_imm,
                producer_pcs=tuple(int(p[0]) for p in record[0]) if record is not None else (),
            ),
            "symbolic_slice",
        )

    def _build_direct_problem(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
    ) -> Optional[_SymbolicProblem]:
        if z3 is None:
            return None
        variable = z3.BitVec("external_input_direct", 32)
        condition = self._normalize_mnemonic(branch_condition)
        if condition in {"CBZ", "CBNZ"}:
            taken = variable == z3.BitVecVal(0, 32)
            if condition == "CBNZ":
                taken = z3.Not(taken)
            predicate = taken if target_direction else z3.Not(taken)
            return _SymbolicProblem(variable, predicate, 32, "direct_cbz_model")

        compare_insn = self.compare_lookup.get(int(branch_pc))
        if compare_insn is None:
            return None
        parts = self._split_operands(compare_insn.get("operands"))
        if len(parts) < 2:
            return None
        immediate = self._immediate(parts[1])
        if immediate is None:
            return None
        registers = {"r0": variable}
        synthetic = dict(compare_insn)
        synthetic["operands"] = f"r0, #{immediate}"
        taken = self._compare_predicate(synthetic, condition, registers)
        if taken is None:
            return None
        predicate = taken if target_direction else z3.Not(taken)
        return _SymbolicProblem(variable, predicate, 32, "direct_compare_model")

    def _apply_symbolic_instruction(
        self,
        mnemonic: str,
        parts: Sequence[str],
        registers: Dict[str, object],
    ) -> None:
        dest = self._register(parts[0]) if parts else None
        if dest is None:
            return

        def clear() -> None:
            registers.pop(dest, None)

        if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
            value = self._operand_expr(parts[1], registers)
            if value is None:
                clear()
            else:
                registers[dest] = value
            return
        if mnemonic == "MOVT" and len(parts) >= 2:
            high = self._immediate(parts[1])
            low_expr = registers.get(dest)
            if high is None or low_expr is None or not z3.is_bv_value(z3.simplify(low_expr)):
                clear()
            else:
                low = z3.simplify(low_expr).as_long() & 0xFFFF
                registers[dest] = z3.BitVecVal(((high & 0xFFFF) << 16) | low, 32)
            return
        if mnemonic in {"MVN", "MVNS"} and len(parts) >= 2:
            value = self._operand_expr(parts[1], registers)
            if value is None:
                clear()
            else:
                registers[dest] = ~value
            return
        if mnemonic in {"UXTB", "UXTH", "SXTB", "SXTH"} and len(parts) >= 2:
            value = self._operand_expr(parts[1], registers)
            if value is None:
                clear()
                return
            width = 8 if mnemonic.endswith("B") else 16
            low = z3.Extract(width - 1, 0, value)
            registers[dest] = (
                z3.SignExt(32 - width, low)
                if mnemonic.startswith("SX")
                else z3.ZeroExt(32 - width, low)
            )
            return
        if mnemonic == "UBFX" and len(parts) >= 4:
            source = self._operand_expr(parts[1], registers)
            lsb = self._immediate(parts[2])
            width = self._immediate(parts[3])
            if source is None or lsb is None or width is None or width <= 0 or lsb + width > 32:
                clear()
                return
            field = z3.Extract(lsb + width - 1, lsb, source)
            registers[dest] = z3.ZeroExt(32 - width, field)
            return

        binary_ops = {
            "ADD", "ADDS", "SUB", "SUBS", "RSB", "RSBS",
            "AND", "ANDS", "ORR", "ORRS", "EOR", "EORS", "BIC", "BICS",
            "LSL", "LSLS", "LSR", "LSRS", "ASR", "ASRS", "ROR", "RORS",
            "MUL", "MULS",
        }
        if mnemonic in binary_ops:
            if len(parts) == 2:
                left = registers.get(dest)
                right = self._operand_expr(parts[1], registers)
            elif len(parts) >= 3:
                left = self._operand_expr(parts[1], registers)
                right = self._operand_expr(parts[2], registers)
            else:
                left = right = None
            if left is None or right is None:
                clear()
                return
            result = self._binary_op_expr(mnemonic, left, right)
            if result is None:
                clear()
                return
            registers[dest] = z3.simplify(result)
            return

        # Unknown register writes invalidate provenance instead of silently
        # carrying a stale symbolic value through the slice.
        if mnemonic not in {
            "CMP", "CMN", "TST", "TEQ", "STR", "STRB", "STRH",
            "PUSH", "POP", "B", "BL", "BLX", "BX",
        } and not mnemonic.startswith("B"):
            clear()

    @staticmethod
    def _binary_op_expr(mnemonic: str, left, right):
        """Shared ALU semantics for slice execution and flag predicates."""
        if mnemonic in {"ADD", "ADDS"}:
            return left + right
        if mnemonic in {"SUB", "SUBS"}:
            return left - right
        if mnemonic in {"RSB", "RSBS"}:
            return right - left
        if mnemonic in {"AND", "ANDS"}:
            return left & right
        if mnemonic in {"ORR", "ORRS"}:
            return left | right
        if mnemonic in {"EOR", "EORS"}:
            return left ^ right
        if mnemonic in {"BIC", "BICS"}:
            return left & ~right
        if mnemonic in {"MUL", "MULS"}:
            return left * right
        simplified_right = z3.simplify(right)
        if not z3.is_bv_value(simplified_right):
            return None
        shift = simplified_right.as_long() & 0x1F
        if mnemonic in {"LSL", "LSLS"}:
            return left << shift
        if mnemonic in {"LSR", "LSRS"}:
            return z3.LShR(left, shift)
        if mnemonic in {"ASR", "ASRS"}:
            return left >> shift
        if mnemonic in {"ROR", "RORS"}:
            return z3.RotateRight(left, shift)
        return None

    def _find_flag_setting_instruction(
        self,
        instructions: Sequence[Dict[str, object]],
        read_index: int,
        branch_index: int,
    ) -> Optional[Dict[str, object]]:
        """Deprecated (r19)：生产谓词构建已由符号标志槽接管（多产生者 1.02%
        与部分覆盖语义由槽模型修正）；本函数仅保留为 producer_index 未命中
        时的降级定位（`_build_symbolic_problem` legacy 路径）。
        """
        for index in range(branch_index - 1, read_index, -1):
            mnemonic = self._normalize_mnemonic(instructions[index].get("mnemonic"))
            if (
                mnemonic in {"CMP", "CMN", "TST", "TEQ"}
                or mnemonic in self._FLAG_SETTING_MNEMONICS
            ):
                return instructions[index]
        return None

    def _flag_setting_predicate(
        self,
        flag_insn: Dict[str, object],
        condition: str,
        registers: Dict[str, object],
    ):
        """Deprecated (r19)：由符号标志槽 + `_condition_from_flags` 取代。

        保留函数体供索引未命中且无 z3 槽路径的降级调试使用；生产调用链
        不再经过这里（多产生者/部分覆盖语义已由槽模型修正）。
        """
        if z3 is None:
            return None
        mnemonic = self._normalize_mnemonic(flag_insn.get("mnemonic"))
        parts = self._split_operands(flag_insn.get("operands"))
        if not parts or self._register(parts[0]) is None:
            return None
        result = self._alu_result_expr(mnemonic, parts, registers)
        if result is None:
            return None
        zero = result == z3.BitVecVal(0, 32)
        negative = z3.Extract(31, 31, result) == z3.BitVecVal(1, 1)
        condition = self._normalize_mnemonic(condition)
        if condition == "BEQ":
            return zero
        if condition == "BNE":
            return z3.Not(zero)
        if condition == "BMI":
            return negative
        if condition == "BPL":
            return z3.Not(negative)
        return None

    # ------------------------------------------------------------------
    # r19：符号标志槽（/tmp/dfs_r18b.md §5.2）
    # ------------------------------------------------------------------

    @staticmethod
    def _mnemonic_is_predicated(raw_mnemonic: object) -> bool:
        raw = str(raw_mnemonic or "").strip().upper()
        if not raw or "." not in raw:
            return False
        parts = [part for part in raw.split(".") if part]
        if len(parts) < 2 or parts[0] in {"B", "BL", "BLX", "BX", "BXJ"}:
            return False
        if parts[0].startswith("IT"):
            return False
        return any(
            part in {"EQ", "NE", "CS", "HS", "CC", "LO", "MI", "PL",
                     "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE"}
            for part in parts[1:]
        )

    @staticmethod
    def _shift_suffix(token: str):
        """解析移位后缀 "lsr #0x14"/"rrx"/"ror r3" → (op, amount, is_register)。"""
        match = re.match(r"^(lsl|lsr|asr|ror|rrx)\b\s*(.*)$", str(token or "").strip(), re.IGNORECASE)
        if not match:
            return None
        op = match.group(1).lower()
        rest = match.group(2).strip()
        if op == "rrx":
            return op, None, False
        if rest.startswith("#"):
            try:
                return op, int(rest[1:], 0) & 0xFFFFFFFF, False
            except ValueError:
                return None
        if re.fullmatch(r"(?:r(?:1[0-5]|[0-9])|sp|lr|pc)", rest.lower()):
            return op, None, True  # 寄存器移位置：静态不可定
        if rest == "":
            return op, None, False
        return None

    def _shift_result(self, op: str, amount, source):
        """移位结果表达式（支持 imm 0..32；寄存器量返回 None）。"""
        if source is None or amount is None:
            return None
        if op == "lsl":
            if amount == 0:
                return source
            if amount >= 32:
                return z3.BitVecVal(0, 32)
            return source << amount
        if op == "lsr":
            if amount == 0:
                return source
            if amount >= 32:
                return z3.BitVecVal(0, 32)
            return z3.LShR(source, amount)
        if op == "asr":
            if amount == 0:
                return source
            if amount >= 32:
                return z3.SignExt(31, z3.Extract(31, 31, source))
            return source >> amount
        if op == "ror":
            if amount == 0:
                return source
            if amount >= 32:
                return None  # ROR #32 形态按不可达处理
            return z3.RotateRight(source, amount)
        return None

    def _shift_carry(self, op: str, amount, source):
        """移位 C 公式（DDI 0403E Shift_C）：LSL #n → src<32-n>，LSR/ASR/ROR #n → src<n-1>。

        n=0（LSL/LSR #0）：C 不变（None 哨兵）；RRX 单独处理；寄存器量 None。
        """
        if source is None or amount is None or amount < 1 or amount > 32:
            return None
        if op == "lsl":
            bit = 32 - amount
        elif op in {"lsr", "asr", "ror"}:
            bit = amount - 1
        else:
            return None
        return z3.Extract(bit, bit, source) == z3.BitVecVal(1, 1)

    @staticmethod
    def _arith_carry_overflow(op: str, left, right, carry_in=None):
        """算术 S 形的 C/V（从旧 :738-752 的 CMP/CMN 公式提取并泛化）。

        op ∈ {'add','sub','rsb'}；carry_in 为 ADC/SBC 链式借位/进位读的 C 槽
        （Bool 表达式；None 表示首次或槽未知）。返回 (carry, overflow)，
        任一不可建模时该位为 None。
        """
        if op == "add":
            left33 = z3.ZeroExt(1, left)
            right33 = z3.ZeroExt(1, right)
            total = left33 + right33
            if carry_in is not None:
                total = total + z3.If(carry_in, z3.BitVecVal(1, 33), z3.BitVecVal(0, 33))
                result = left + right + z3.If(carry_in, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
            else:
                result = left + right
            carry = z3.Extract(32, 32, total) == z3.BitVecVal(1, 1)
            overflow = z3.And(
                z3.Extract(31, 31, left) == z3.Extract(31, 31, right),
                z3.Extract(31, 31, result) != z3.Extract(31, 31, left),
            )
            return carry, overflow
        if op == "rsb":
            minuend, subtrahend = right, left  # result = right - left
        else:
            minuend, subtrahend = left, right
        minuend33 = z3.ZeroExt(1, minuend)
        subtrahend33 = z3.ZeroExt(1, subtrahend)
        if carry_in is not None:
            # SBC：borrow = ¬C；carry = 无借位 = a >= b + ¬cin
            borrow33 = z3.If(carry_in, z3.BitVecVal(0, 33), z3.BitVecVal(1, 33))
            carry = z3.UGE(minuend33, subtrahend33 + borrow33)
            result = minuend - subtrahend - z3.If(
                carry_in, z3.BitVecVal(0, 32), z3.BitVecVal(1, 32)
            )
        else:
            carry = z3.UGE(minuend33, subtrahend33)
            result = minuend - subtrahend
        overflow = z3.And(
            z3.Extract(31, 31, minuend) != z3.Extract(31, 31, subtrahend),
            z3.Extract(31, 31, result) != z3.Extract(31, 31, minuend),
        )
        return carry, overflow

    def _resolve_alu_operands(self, mnemonic: str, parts: Sequence[str], registers):
        """算术/逻辑族的 (left, right, shift_op, shift_amount, shift_source)。

        CMP/CMN/TST/TEQ 无目的寄存器；3 操作数 S 形 left=parts[1]；2 操作数
        S 形 left=旧 dest 值。移位后缀（"lsr #0x14"）应用在 operand2 上——
        这同时修正了旧 `_alu_result_expr` 丢弃寄存器移位的近似
        （`rsbs r5, r4, r5, lsr #0x15` 形态）。
        """
        compare_like = mnemonic in {"CMP", "CMN", "TST", "TEQ"}
        if compare_like:
            if len(parts) < 2:
                return None
            left = self._operand_expr(parts[0], registers)
            rest = parts[1:]
        elif len(parts) >= 3:
            left = self._operand_expr(parts[1], registers)
            rest = parts[2:]
        elif len(parts) == 2:
            dest = self._register(parts[0])
            left = registers.get(dest) if dest is not None else None
            rest = parts[1:]
        else:
            return None

        shift_op = shift_amount = None
        shift_source = None
        right = None
        suffix = self._shift_suffix(rest[-1]) if len(rest) >= 2 else None
        if suffix is not None:
            shift_op, shift_amount, register_amount = suffix
            if register_amount:
                return None  # 寄存器移位置：值与 C 均不可静态定
            source = self._operand_expr(rest[0], registers)
            if source is None:
                return None
            shift_source = source
            right = self._shift_result(shift_op, shift_amount, source)
            if right is None:
                return None
        else:
            right = self._operand_expr(rest[0], registers) if rest else None
        if left is None or right is None:
            return None
        return left, right, shift_op, shift_amount, shift_source

    def _apply_flag_effects(
        self,
        insn: Dict[str, object],
        mnemonic: str,
        parts: Sequence[str],
        registers: Dict[str, object],
        flags: Dict[str, object],
    ) -> Optional[int]:
        """对一条指令应用 NZCV 槽写（r19：逐产生者只覆写其 covers 位）。

        返回比较类产生者的 operand2 立即数（种子表用），否则 None。
        槽值为 z3 Bool 表达式；None 表示未知/不变（依赖它的条件拒绝求解）。
        """
        if z3 is None or not parts:
            return None
        raw_mnemonic = str(insn.get("mnemonic") or "")
        wide = int(insn.get("size") or 2) == 4 or ".W" in raw_mnemonic.upper()
        operands = str(insn.get("operands") or "")
        compare_imm: Optional[int] = None

        def set_nz(result) -> bool:
            if result is None:
                flags["N"] = None
                flags["Z"] = None
                return False
            flags["N"] = z3.Extract(31, 31, result) == z3.BitVecVal(1, 1)
            flags["Z"] = result == z3.BitVecVal(0, 32)
            return True

        if mnemonic in self._EXPLICIT_FLAG_MNEMONICS:
            # VMRS APSR / MSR APSR：显式四标志写者。FP 操作数不可静态建模
            # （r18b §4.2：可建模 0/58179）——索引命中时整站跳过；降级路径
            # 遇到时置四槽为未知，绝不沿用旧值。
            for flag in "NZCV":
                flags[flag] = None
            return None

        if mnemonic == "MULS":
            if self._mnemonic_is_predicated(raw_mnemonic):
                return None  # IT 块内 MULS 不写标志（A7.7.84）
            if not set_nz(self._alu_result_expr(mnemonic, parts, registers)):
                pass
            # C/V 槽不动（UNCHANGED）
            return None

        if mnemonic in self._ARITH_FLAG_MNEMONICS:
            resolved = self._resolve_alu_operands(mnemonic, parts, registers)
            chain_carry = None
            chain = mnemonic in {"ADCS", "SBCS", "RSBCS", "RSBCSS", "RSCS"}
            if resolved is None:
                if chain:
                    for flag in "NZCV":
                        flags[flag] = None
                else:
                    set_nz(None)
                return None
            left, right, shift_op, shift_amount, shift_source = resolved
            if mnemonic in {"RSBS", "RSBCS", "RSBCSS", "RSCS"}:
                arith_op = "rsb"
            elif mnemonic in {"CMN", "ADDS", "ADCS"}:
                arith_op = "add"
            else:  # CMP/SUBS/SBCS
                arith_op = "sub"
            if chain:
                chain_carry = flags.get("C")
                if chain_carry is None:
                    # C 槽未知：链式结果本身不确定，N/Z 也不可模型
                    for flag in "NZCV":
                        flags[flag] = None
                    return None
                # ADC 加进位项 / SBC 减借位项（¬C）：N/Z 结果必须含该项
                if arith_op == "add":
                    term = z3.If(chain_carry, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
                else:
                    term = z3.If(chain_carry, z3.BitVecVal(0, 32), z3.BitVecVal(1, 32))
                if arith_op == "rsb":
                    result = (right - left) - term
                elif arith_op == "add":
                    result = (left + right) + term
                else:
                    result = (left - right) - term
            elif arith_op == "rsb":
                result = right - left
            elif arith_op == "add":
                result = left + right
            else:
                result = left - right
            carry, overflow = self._arith_carry_overflow(arith_op, left, right, chain_carry)
            set_nz(result)
            flags["C"] = carry
            flags["V"] = overflow
            # 种子表用：比较类/减法比较惯用形的 operand2 立即数
            if mnemonic in {"CMP", "CMN", "SUBS", "ADDS", "RSBS"}:
                if mnemonic in {"CMP", "CMN"} or len(parts) == 2:
                    candidate = parts[1] if len(parts) >= 2 else None
                else:
                    candidate = parts[2] if len(parts) >= 3 else None
                if candidate is not None and self._shift_suffix(candidate) is None:
                    value = self._immediate(candidate)
                    if value is not None:
                        compare_imm = value
            return compare_imm

        if mnemonic in self._LOGIC_FLAG_MNEMONICS:
            resolved = self._resolve_alu_operands(mnemonic, parts, registers)
            if resolved is None:
                set_nz(None)
                return None
            left, right, shift_op, shift_amount, shift_source = resolved
            if mnemonic == "TST":
                result = left & right
            elif mnemonic == "TEQ":
                result = left ^ right
            elif mnemonic in {"MOVS", "MVNS"}:
                result = right if mnemonic == "MOVS" else ~right
            else:
                base = {"ANDS": "AND", "ORRS": "ORR", "EORS": "EOR", "BICS": "BIC", "ORNS": "ORN"}
                op = base.get(mnemonic, "AND")
                if op == "ORN":
                    result = left | ~right
                elif op == "BIC":
                    result = left & ~right
                else:
                    result = self._binary_op_expr(op, left, right)
            set_nz(result)
            if shift_op is not None:
                # 带移位寄存器形：C = 移位进位（LSL #0 → C 不变 → None 哨兵）
                carry = self._shift_carry(shift_op, shift_amount, shift_source)
                if mnemonic == "MOVS" and shift_op == "rrx":
                    # MOVS rX, rY, rrx 少见形态：按 RRX 公式
                    carry = (
                        z3.Extract(0, 0, shift_source) == z3.BitVecVal(1, 1)
                        if shift_source is not None else None
                    )
                if carry is not None:
                    flags["C"] = carry
                # shift_amount == 0：C 不变（不动槽）
            else:
                # 立即数形态：仅 .w imm 走 ThumbExpandImm_C 全模式；
                # 16 位 imm / 纯寄存器 / 非旋转复制形 → C 不变（不动槽）。
                if len(parts) == 2:
                    candidate = parts[1]
                elif len(parts) >= 3:
                    candidate = parts[2]
                else:
                    candidate = None
                if candidate is not None:
                    imm = self._immediate(candidate)
                    if imm is not None and wide:
                        imm12 = imm12_from_value(imm)
                        if imm12 is not None and ((imm12 >> 10) & 3) != 0:
                            _, carry_const = expand_imm_c(imm12)
                            flags["C"] = z3.BoolVal(bool(carry_const))
            if mnemonic in {"TST", "TEQ"}:
                candidate = parts[1] if len(parts) >= 2 else None
                if candidate is not None and self._shift_suffix(candidate) is None:
                    value = self._immediate(candidate)
                    if value is not None:
                        compare_imm = value
            return compare_imm

        if mnemonic in self._SHIFT_FLAG_MNEMONICS:
            # LSLS/LSRS/ASRS/RORS：parts = [dest, src, amount]；RRXS：[dest, src]
            if mnemonic == "RRXS":
                source = self._operand_expr(parts[1], registers) if len(parts) >= 2 else None
                if source is None:
                    set_nz(None)
                    flags["C"] = None
                    return None
                carry_in = flags.get("C")
                if carry_in is None:
                    shifted = z3.LShR(source, 1)
                    flags["C"] = z3.Extract(0, 0, source) == z3.BitVecVal(1, 1)
                    set_nz(shifted)
                else:
                    top = z3.If(carry_in, z3.BitVecVal(0x80000000, 32), z3.BitVecVal(0, 32))
                    shifted = z3.LShR(source, 1) | top
                    flags["C"] = z3.Extract(0, 0, source) == z3.BitVecVal(1, 1)
                    set_nz(shifted)
                return None
            source = self._operand_expr(parts[1], registers) if len(parts) >= 3 else None
            amount = self._immediate(parts[2]) if len(parts) >= 3 else None
            if source is None or amount is None:
                # 寄存器移位置：C 不可定；N/Z 用掩码量的旧近似结果
                set_nz(self._alu_result_expr(mnemonic, parts, registers))
                flags["C"] = None
                return None
            op = mnemonic[:-1].lower()  # LSLS → lsl
            shifted = self._shift_result(op, amount, source)
            set_nz(shifted)
            carry = self._shift_carry(op, amount, source)
            if carry is not None:
                flags["C"] = carry
            # amount == 0：C 不变（不动槽）；C/V 恒不动
            return None

        return None

    def _condition_from_flags(self, condition: str, flags: Dict[str, object]):
        """从标志槽构建条件码谓词（ARM ARM 条件码表）。

        任一所需槽为 None（未知/读前定义）→ None（不可解）。组合公式与
        旧 `_compare_predicate` :769-792 一致，只是槽来源统一。
        """
        if z3 is None:
            return None
        zero = flags.get("Z")
        carry = flags.get("C")
        negative = flags.get("N")
        overflow = flags.get("V")

        def both(a, b):
            if a is None or b is None:
                return None
            return z3.And(a, b)

        if condition == "BEQ":
            return zero
        if condition == "BNE":
            return z3.Not(zero) if zero is not None else None
        if condition in {"BCS", "BHS"}:
            return carry
        if condition in {"BCC", "BLO"}:
            return z3.Not(carry) if carry is not None else None
        if condition == "BMI":
            return negative
        if condition == "BPL":
            return z3.Not(negative) if negative is not None else None
        if condition == "BVS":
            return overflow
        if condition == "BVC":
            return z3.Not(overflow) if overflow is not None else None
        if condition == "BHI":
            if carry is None or zero is None:
                return None
            return z3.And(carry, z3.Not(zero))
        if condition == "BLS":
            if carry is None or zero is None:
                return None
            return z3.Or(z3.Not(carry), zero)
        if condition == "BGE":
            if negative is None or overflow is None:
                return None
            return negative == overflow
        if condition == "BLT":
            if negative is None or overflow is None:
                return None
            return negative != overflow
        if condition == "BGT":
            if negative is None or overflow is None or zero is None:
                return None
            return z3.And(z3.Not(zero), negative == overflow)
        if condition == "BLE":
            if negative is None or overflow is None or zero is None:
                return None
            return z3.Or(zero, negative != overflow)
        return None

    def _alu_result_expr(
        self,
        mnemonic: str,
        parts: Sequence[str],
        registers: Dict[str, object],
    ):
        """Result expression of `mnemonic` over its operands, or None."""
        dest = self._register(parts[0]) if parts else None
        if dest is None:
            return None
        if len(parts) == 2:
            left = registers.get(dest)
            right = self._operand_expr(parts[1], registers)
        elif len(parts) >= 3:
            left = self._operand_expr(parts[1], registers)
            right = self._operand_expr(parts[2], registers)
        else:
            return None
        if left is None or right is None:
            return None
        return self._binary_op_expr(mnemonic, left, right)

    def _compare_predicate(
        self,
        compare_insn: Dict[str, object],
        condition: str,
        registers: Dict[str, object],
    ):
        mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic"))
        parts = self._split_operands(compare_insn.get("operands"))
        if len(parts) < 2:
            return None
        left = self._operand_expr(parts[0], registers)
        right = self._operand_expr(parts[1], registers)
        if left is None or right is None:
            return None

        if mnemonic == "CMP":
            result = left - right
            carry = z3.UGE(left, right)
            overflow = z3.And(
                z3.Extract(31, 31, left) != z3.Extract(31, 31, right),
                z3.Extract(31, 31, result) != z3.Extract(31, 31, left),
            )
        elif mnemonic == "CMN":
            result = left + right
            extended = z3.ZeroExt(1, left) + z3.ZeroExt(1, right)
            carry = z3.Extract(32, 32, extended) == z3.BitVecVal(1, 1)
            overflow = z3.And(
                z3.Extract(31, 31, left) == z3.Extract(31, 31, right),
                z3.Extract(31, 31, result) != z3.Extract(31, 31, left),
            )
        elif mnemonic == "TST":
            result = left & right
            carry = overflow = None
        elif mnemonic == "TEQ":
            result = left ^ right
            carry = overflow = None
        else:
            return None

        zero = result == z3.BitVecVal(0, 32)
        negative = z3.Extract(31, 31, result) == z3.BitVecVal(1, 1)
        condition = self._normalize_mnemonic(condition)
        if condition == "BEQ":
            return zero
        if condition == "BNE":
            return z3.Not(zero)
        if condition in {"BCS", "BHS"}:
            return carry
        if condition in {"BCC", "BLO"}:
            return z3.Not(carry) if carry is not None else None
        if condition == "BMI":
            return negative
        if condition == "BPL":
            return z3.Not(negative)
        if condition == "BVS":
            return overflow
        if condition == "BVC":
            return z3.Not(overflow) if overflow is not None else None
        if condition == "BHI":
            return z3.And(carry, z3.Not(zero)) if carry is not None else None
        if condition == "BLS":
            return z3.Or(z3.Not(carry), zero) if carry is not None else None
        if condition == "BGE":
            return negative == overflow if overflow is not None else None
        if condition == "BLT":
            return negative != overflow if overflow is not None else None
        if condition == "BGT":
            return z3.And(z3.Not(zero), negative == overflow) if overflow is not None else None
        if condition == "BLE":
            return z3.Or(zero, negative != overflow) if overflow is not None else None
        return None

    def _condition_derived_values(
        self,
        problem: Optional[_SymbolicProblem],
        *,
        max_single_bits: int = 4,
    ) -> List[Tuple[int, str]]:
        """Reverse-solve minimal bit patterns the branch predicate accepts.

        Single-bit values first (lowest bit first) — these are the physically
        plausible "set exactly this status bit" models (bit17 for an
        `lsls #14; bpl` HSERDY poll) — then the complement of the first two.
        Returns [] when no symbolic predicate exists: derivation never guesses.
        """
        if problem is None:
            return []
        width = min(32, max(1, int(problem.source_width or 32)))
        mask = (1 << width) - 1 if width < 32 else U32_MASK
        single_bits: List[int] = []
        for bit in range(width):
            candidate = 1 << bit
            if self._problem_accepts(problem, candidate):
                single_bits.append(candidate)
                if len(single_bits) >= max(1, int(max_single_bits)):
                    break
        derived = [
            (value, "condition_derived_single_bit") for value in single_bits
        ]
        for value in single_bits[:2]:
            complement = (~value) & mask
            if not complement:
                continue
            if any(complement == existing for existing, _strategy in derived):
                continue
            if self._problem_accepts(problem, complement):
                derived.append((complement, "condition_derived_complement"))
        return derived

    def _semantic_boundary_values(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        compare_imm: Optional[int] = None,
        compare_partner_values: Iterable[int] = (),
    ) -> List[Tuple[int, str]]:
        values: List[Tuple[int, str]] = [
            (0, "zero_boundary"),
            (1, "unit_boundary"),
            (0xFF, "byte_boundary"),
            (0xFFFF, "halfword_boundary"),
            (U32_MASK, "all_bits_boundary"),
            (0x7FFFFFFF, "signed_max_boundary"),
            (0x80000000, "signed_min_boundary"),
        ]
        condition = self._normalize_mnemonic(branch_condition)
        compare_insn = self.compare_lookup.get(int(branch_pc))
        immediate = compare_imm if compare_imm is not None else None
        mnemonic = ""
        if compare_insn is not None:
            mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic"))
            if immediate is None:
                parts = self._split_operands(compare_insn.get("operands"))
                immediate = self._immediate(parts[1]) if len(parts) >= 2 else None
        partner_values: List[int] = []
        for value in compare_partner_values or ():
            try:
                normalized = int(value) & U32_MASK
            except (TypeError, ValueError):
                continue
            if normalized not in partner_values:
                partner_values.append(normalized)
        if immediate is None:
            # r22：寄存器-寄存器比较没有立即数可围——静态解析出的伙伴常量
            # 顶替立即数位生成同款 ±1 边界带（精确值 + relational 条件所需
            # 的邻界；种子层去重保证高优 literal_partner_register 不被覆盖）。
            for partner in partner_values:
                values.extend([
                    (partner, "literal_partner_register"),
                    ((partner - 1) & U32_MASK, "literal_partner_minus_one"),
                    ((partner + 1) & U32_MASK, "literal_partner_plus_one"),
                ])
            return values
        immediate = int(immediate) & U32_MASK
        if condition in {"BHI", "BLS", "BHS", "BLO"}:
            # r19：无符号条件——围绕比较立即数的边界带
            values.extend([
                ((immediate - 1) & U32_MASK, "unsigned_minus_one"),
                (immediate, "compare_boundary"),
                ((immediate + 1) & U32_MASK, "unsigned_plus_one"),
            ])
        elif condition in {"BCS", "BCC"}:
            # r19：C 类借位边界（BCC/BLO 挂 CMP/SUBS imm：imm-1 首个借位值）
            values.extend([
                ((immediate - 1) & U32_MASK, "borrow_first_below"),
                (immediate, "borrow_at_boundary"),
            ])
        elif condition in {"BGE", "BLT", "BGT", "BLE"}:
            # r19：有符号条件——imm±1（±极值已在基础表）
            values.extend([
                ((immediate - 1) & U32_MASK, "signed_minus_one"),
                ((immediate + 1) & U32_MASK, "signed_plus_one"),
            ])
        else:
            values.extend([
                (immediate, "compare_boundary"),
                ((immediate - 1) & U32_MASK, "compare_minus_one"),
                ((immediate + 1) & U32_MASK, "compare_plus_one"),
            ])
        if mnemonic == "TST":
            least_bit = immediate & -immediate if immediate else 1
            values.extend([
                (least_bit, "mask_least_bit"),
                (immediate, "mask_full"),
                ((~immediate) & U32_MASK, "mask_complement"),
            ])
        return values

    def _problem_accepts(self, problem: _SymbolicProblem, value: int) -> bool:
        if z3 is None:
            return False
        concrete = z3.simplify(
            z3.substitute(
                problem.predicate,
                (problem.variable, z3.BitVecVal(int(value) & U32_MASK, 32)),
            )
        )
        if z3.is_true(concrete):
            return True
        if z3.is_false(concrete):
            return False
        solver = z3.Solver()
        solver.set(timeout=self.solver_timeout_ms)
        solver.add(problem.predicate)
        solver.add(problem.variable == z3.BitVecVal(int(value) & U32_MASK, 32))
        if problem.source_width < 32:
            solver.add(z3.ULT(problem.variable, z3.BitVecVal(1 << problem.source_width, 32)))
        return solver.check() == z3.sat

    def _solver_models(
        self,
        problem: _SymbolicProblem,
        *,
        excluded: Iterable[int],
        max_values: int,
    ) -> List[Tuple[int, str]]:
        if z3 is None or max_values <= 0:
            return []
        excluded_values = {int(value) & U32_MASK for value in excluded or ()}
        results: List[Tuple[int, str]] = []

        def base_constraints() -> List[object]:
            constraints: List[object] = [problem.predicate]
            if problem.source_width < 32:
                constraints.append(
                    z3.ULT(
                        problem.variable,
                        z3.BitVecVal(1 << problem.source_width, 32),
                    )
                )
            for value in sorted(excluded_values):
                constraints.append(problem.variable != z3.BitVecVal(value, 32))
            return constraints

        solver = z3.Solver()
        solver.set(timeout=self.solver_timeout_ms)
        solver.set(random_seed=0)
        solver.add(*base_constraints())
        while len(results) < max_values and solver.check() == z3.sat:
            model = solver.model()
            value_expr = model.eval(problem.variable, model_completion=True)
            if not z3.is_bv_value(value_expr):
                break
            value = value_expr.as_long() & U32_MASK
            solver.add(problem.variable != z3.BitVecVal(value, 32))
            if value in excluded_values:
                continue
            excluded_values.add(value)
            results.append((value, "smt_diverse_model"))
        return results
