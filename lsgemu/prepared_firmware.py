#!/usr/bin/env python3
"""Firmware preparation and static-view construction for LSGEmu.

This module owns the static analysis cache, refined basic-block view, raw ARM
fallback decoding, valid-BB metadata, symbol loading, and emulator construction.
The historical runner imports these types for backward compatibility but keeps
runtime scheduling separate from firmware preparation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple
import logging
import pickle
import re
import subprocess
import hashlib
import json

from .angr_reachability import load_angr_reachable_bb_set
from .artifact_io import atomic_pickle_dump
from .analysis.disasm.ghidra_disassembler_with_bb import BasicBlock
from .analysis.firmware_analyzer import FirmwareAnalyzer
from .analysis.intelligent_emulator import IntelligentEmulator
from .analysis.lightweight_mmio_analysis import LIGHTWEIGHT_MMIO_ANALYSIS_VERSION, analyze_static_mmio
from .analysis.snapshot_memory import (
    SnapshotBlobStore,
    SnapshotMetadataStore,
    SnapshotPageStore,
)
from .instruction_utils import (
    CALL_SPLIT_MNEMONICS,
    COMPARE_MNEMONICS,
    CONDITIONAL_BRANCH_MNEMONICS,
    _is_return_like_instruction,
    _normalize_mnemonic,
    _parse_direct_target,
    _predicated_condition_for_mnemonic,
    _should_split_after_mnemonic,
)
from .runner_common import (
    _instruction_dict,
    _loop_exit_hints_enabled,
    _resolve_valid_bb_metadata,
    _static_cache_path,
)
from .toolchain_fingerprint import collect_toolchain_fingerprint, sha256_file


logger = logging.getLogger(__name__)
# v15 (r19)：static views 增加 producer_index（NZCV 产生者一等对象索引）。
# 版本翻位使全部 v14 旧缓存判废并全量重建（identity_hash 不含该字段，靠版本号失效）。
STATIC_CACHE_VERSION = 15
STATIC_ANALYSIS_SCHEMA = "lsgemu.prepared_firmware.v15"
_ALLOWED_STATIC_CACHE_GLOBALS = frozenset({
    ("lsgemu.prepared_firmware", "PreparedFirmware"),
    ("lsgemu.analysis.firmware_analyzer", "AnalysisResult"),
    ("lsgemu.analysis.disasm.ghidra_disassembler_with_bb", "Instruction"),
    ("lsgemu.analysis.disasm.ghidra_disassembler_with_bb", "BasicBlock"),
    ("lsgemu.analysis.file_parser.architecture", "ArchInfo"),
    ("lsgemu.analysis.file_parser.architecture", "Architecture"),
    ("lsgemu.analysis.file_parser.architecture", "Endianness"),
    ("lsgemu.analysis.mmio_identifier", "MMIOAccess"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotPageStore"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotMetadataStore"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotBlobStore"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotStateBlob"),
    ("lsgemu.analysis.snapshot_memory", "PagedMemory"),
    ("lsgemu.analysis.snapshot_memory", "SnapshotPage"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("types", "SimpleNamespace"),
})


class _RestrictedStaticCacheUnpickler(pickle.Unpickler):
    """Whitelist-restricted unpickler for prepared-firmware static caches.

    Static cache files live on disk and can be swapped by anything with write
    access to the cache directory; a bare ``pickle.load`` would then execute
    arbitrary code at load time. Only the globals listed above are permitted.
    """

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_STATIC_CACHE_GLOBALS:
            raise pickle.UnpicklingError(
                f"forbidden global in prepared-firmware cache: {module}.{name}"
            )
        return super().find_class(module, name)


def _load_static_cache_pickle(path: Path):
    with path.open("rb") as handle:
        return _RestrictedStaticCacheUnpickler(handle).load()


@dataclass
class StaticViews:
    static_bbs: Dict[int, List[Dict[str, object]]]
    static_bb_set: Set[int]
    instruction_to_bb: Dict[int, int]
    instruction_lookup: Dict[int, Dict[str, object]]
    branch_instruction_by_bb: Dict[int, Dict[str, object]]
    compare_lookup: Dict[int, Dict[str, object]]
    static_successors: Dict[int, Set[int]]
    refined_basic_blocks: List[BasicBlock]
    conditional_branch_bbs: List[BasicBlock]
    producer_index: Dict[int, Tuple[Tuple[Tuple[int, str, str, str, bool, int, int], ...], str, int]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# NZCV 产生者索引（r19 P0，设计见 /tmp/dfs_r18b.md §5.1）
#
# 语义裁定（r18b 三重证据：DDI 0403E 手册原文 + qemu/unicorn 源码 + 差分实测）：
# - 32 位乘法族（MUL/MLA/MLS/SMULL/UMULL/SMLAL/UMLAL 及半字并行乘）**永不**写
#   任何标志（A7.7.84 等 setflags = FALSE），不得进入产生者集合；
# - MULS 仅 16 位形态且仅 IT 块外写 N/Z（setflags = !InITBlock()）；
# - CLZ/SXTB/SXTH/UXTB/UXTH/REV*/SSAT/USAT 不写 NZCV（不在表中即自动排除）；
# - VMRS APSR / MSR APSR 族是显式四标志写者，但 FP 操作数不可静态建模
#   （r18b §4：可建模 0/58179）——只标签化（label_fp_opaque/label_msr），
#   值合成通道直接跳过；
# - 逻辑 S 形的 C 按 ThumbExpandImm_C 全模式判定：非旋转（imm8/复制形）不写，
#   旋转形写 C = imm32<31>（ARM ARM A5.3.2 + 本轮 1091 条真实指令对账 capstone）。
# 键 = branch_pc（仅 Bcc；CBZ/CBNZ 走值通道、TBB/TBH 无标志依赖，均不入索引）。
# 窗口（r18b §5.5 已定）：同 BB ≤16 条 / 跨 BB ≤3BB·24 条 / 前驱扇出 ≤6 /
# 结点预算 64；跨调用不拒绝但记录 calls_between（消费侧拒绝）。
# ---------------------------------------------------------------------------

_PRODUCER_ARITH_NZCV = frozenset({
    "CMP", "CMN", "ADDS", "ADCS", "SUBS", "SBCS", "RSBS",
    # Ghidra 在个别现场把 RSC 的 S 形渲染为 rsbcs/rsbcss（全固件 2 条），
    # 语义是「带借位的反向减、写 NZCV」，按算术族处理。
    "RSBCS", "RSBCSS", "RSCS",
})
_PRODUCER_LOGIC = frozenset({
    "TST", "TEQ", "ANDS", "ORRS", "EORS", "BICS", "ORNS", "MOVS", "MVNS",
})
_PRODUCER_SHIFT_NZC = frozenset({"LSLS", "LSRS", "ASRS", "RORS", "RRXS"})

# 逐条件码的标志需求（ARM ARM 条件码表；HI/LS/GT/LE 需多位）
_PRODUCER_COND_FLAGS = {
    "BEQ": "Z", "BNE": "Z",
    "BCS": "C", "BHS": "C", "BCC": "C", "BLO": "C",
    "BMI": "N", "BPL": "N", "BVS": "V", "BVC": "V",
    "BHI": "CZ", "BLS": "CZ",
    "BGE": "NV", "BLT": "NV", "BGT": "NZV", "BLE": "NZV",
}
_PRODUCER_INDEXED_BRANCH_MNEMONICS = frozenset(
    CONDITIONAL_BRANCH_MNEMONICS - {"CBZ", "CBNZ", "TBB", "TBH"}
)

# 回溯窗口（r18b §5.5 定论：16 条 / 跨 BB ≤3BB·24 条）
_PRODUCER_WINDOW_SAME_BB = 16
_PRODUCER_WINDOW_TOTAL = 24
_PRODUCER_MAX_BBS = 3
_PRODUCER_MAX_PRED_FAN = 6
_PRODUCER_NODE_BUDGET = 200

_PRODUCER_SHIFT_TOKEN = re.compile(r"\b(lsl|lsr|asr|ror|rrx)\b", re.IGNORECASE)
_PRODUCER_IMM_TOKEN = re.compile(r"#-?(?:0x[0-9a-f]+|\d+)", re.IGNORECASE)


def _producer_imm32_from_operands(operands: str) -> Optional[int]:
    """最后一级 #imm 操作数（Ghidra 渲染的是 ThumbExpandImm 展开后的 imm32）。"""
    match = None
    for match in _PRODUCER_IMM_TOKEN.finditer(str(operands or "")):
        pass
    if match is None:
        return None
    try:
        return int(match.group(0)[1:], 0) & 0xFFFFFFFF
    except ValueError:
        return None


def _producer_imm12_from_value(value: int) -> Optional[int]:
    """imm32 → imm12 反解（T32 modified immediate 在可编码域上单射）。

    与 local_constraint_recovery.expand_imm_c 的前向解码互逆；三类非旋转
    复制形与旋转形（u<<s, u≥0x80, s∈1..24）值域不相交，反解唯一。
    """
    value &= 0xFFFFFFFF
    low = value & 0xFF
    if value == low:
        return low  # '00'：imm8
    if low and value == ((low << 16) | low):
        return 0x100 | low  # '01'：imm8 复制到 23:16 与 7:0
    mid = (value >> 8) & 0xFF
    if mid and value == ((mid << 24) | (mid << 8)):
        return 0x200 | mid  # '10'：imm8 复制到 31:24 与 15:8
    if low and value == low * 0x01010101:
        return 0x300 | low  # '11'：imm8 复制到全部四字节
    for shift in range(1, 25):
        if value & ((1 << shift) - 1):
            continue
        unit = (value >> shift) & 0xFF
        if unit >= 0x80 and ((unit << shift) & 0xFFFFFFFF) == value:
            return ((32 - shift) << 7) | (unit & 0x7F)
    return None


def _producer_flag_writes(insn: Dict[str, object]) -> Tuple[str, str, str, bool]:
    """单指令 NZCV 写语义（r18b 修正版）。

    返回 (covers ⊂ "NZCV", kind, c_source, in_it)；covers 为空表示不是产生者。
    c_source ∈ {'arith','shifter','imm_expand','explicit', None}，供谓词构建
    分派 C 公式（local_constraint_recovery._apply_symbolic_instruction）。
    """
    raw_mnemonic = str(insn.get("mnemonic") or "")
    mnemonic = _normalize_mnemonic(raw_mnemonic)
    operands = str(insn.get("operands") or "")
    in_it = _predicated_condition_for_mnemonic(raw_mnemonic) is not None

    if mnemonic in _PRODUCER_ARITH_NZCV:
        return "NZCV", "arith", "arith", in_it
    if mnemonic in _PRODUCER_SHIFT_NZC:
        return "NZC", "shift", "shifter", in_it
    if mnemonic == "MULS":
        # 16 位 MULS 仅 IT 块外写 N/Z（A7.7.84 setflags = !InITBlock()；
        # 多槽 IT 内实测不写）。32 位乘法族在解码层就不到这里。
        if in_it:
            return "", "", None, in_it
        return "NZ", "mul16", None, in_it
    if mnemonic in _PRODUCER_LOGIC:
        has_shift = bool(_PRODUCER_SHIFT_TOKEN.search(operands))
        if has_shift:
            return "NZC", "logic", "shifter", in_it
        wide = int(insn.get("size") or 2) == 4 or ".W" in raw_mnemonic.upper()
        if wide:
            imm32 = _producer_imm32_from_operands(operands)
            if imm32 is not None:
                imm12 = _producer_imm12_from_value(imm32)
                if imm12 is not None and ((imm12 >> 10) & 3) != 0:
                    # 旋转形：C = imm32<31>（解码期常量）
                    return "NZC", "logic", "imm_expand", in_it
                # 非旋转 imm / 复制形：C 不变
                return "NZ", "logic", None, in_it
        # 纯寄存器 / 16 位 imm：C 不变
        return "NZ", "logic", None, in_it
    if mnemonic == "VMRS":
        if "apsr" in operands.lower():
            return "NZCV", "explicit", "explicit", in_it
        return "", "", None, in_it
    if mnemonic == "MSR":
        first = operands.split(",")[0].strip().lower()
        if first.startswith(("apsr", "iapsr", "eapsr", "xpsr")) and not first.endswith("_g"):
            return "NZCV", "explicit", "explicit", in_it
        return "", "", None, in_it
    return "", "", None, in_it


def _build_producer_index(
    static_bbs: Dict[int, List[Dict[str, object]]],
    static_successors: Dict[int, Set[int]],
) -> Dict[int, Tuple[Tuple[Tuple[int, str, str, str, bool, int, int], ...], str, int]]:
    """对全部 Bcc 站点做「逐标志最近写者胜」回溯，产出 branch_pc → ProducerRecord。

    算法移植自 r18a/r18b census（scripts/audit_20260921/r18a_claude_census.py
    的 analyze_branch + r18b_recensus 的乘法族修正），但运行在 prepared 的
    refined 静态视图上（static_bbs / static_successors 反转的前驱表，自环边
    按 r14 语义排除）——不另建 CFG、不重新反汇编，天然满足节口径（Ghidra
    只反汇编 SHF_EXECINSTR 节，段内节外伪分支不进视图）。
    """
    predecessors: Dict[int, List[int]] = {}
    for src, dsts in (static_successors or {}).items():
        for dst in dsts or ():
            if int(dst) == int(src):
                continue  # r14：自环是回边，不是到达路径
            predecessors.setdefault(int(dst), []).append(int(src))
    for plist in predecessors.values():
        plist.sort()

    index: Dict[int, Tuple[Tuple[Tuple[int, str, str, str, bool, int, int], ...], str, int]] = {}

    for bb_addr in sorted(static_bbs):
        instructions = static_bbs.get(bb_addr) or ()
        if not instructions:
            continue
        branch_insn = instructions[-1]
        branch_pc = int(branch_insn.get("address", 0) or 0)
        mnemonic = _normalize_mnemonic(branch_insn.get("mnemonic"))
        if mnemonic not in _PRODUCER_INDEXED_BRANCH_MNEMONICS:
            continue
        required = set(_PRODUCER_COND_FLAGS.get(mnemonic, ""))

        budget = {"nodes": 0}

        def finish(
            status: str,
            covered: Dict[str, int],
            producers: List[Dict[str, object]],
            bbs_crossed: int,
            calls: int,
        ) -> Dict[str, object]:
            return {
                "status": status,
                "covered": covered,
                "producers": producers,
                "bbs": bbs_crossed,
                "calls": calls,
            }

        def walk(
            bb: int,
            stop_index: int,
            steps: int,
            bbs_crossed: int,
            calls: int,
            covered: Dict[str, int],
            visited: frozenset,
        ) -> Dict[str, object]:
            budget["nodes"] += 1
            local = dict(covered)
            producers: List[Dict[str, object]] = []
            addrs = static_bbs.get(bb) or ()
            segment_start = steps
            for i in range(stop_index, -1, -1):
                steps += 1
                if steps > _PRODUCER_WINDOW_TOTAL or (steps - segment_start) > _PRODUCER_WINDOW_SAME_BB:
                    return finish("ml_window", local, producers, bbs_crossed, calls)
                if budget["nodes"] > _PRODUCER_NODE_BUDGET:
                    return finish("ml_budget", local, producers, bbs_crossed, calls)
                insn = addrs[i]
                pc = int(insn.get("address", 0) or 0)
                insn_mnemonic = _normalize_mnemonic(insn.get("mnemonic"))
                if insn_mnemonic in {"BL", "BLX"}:
                    calls += 1
                covers, kind, c_source, in_it = _producer_flag_writes(insn)
                if covers:
                    newly = [f for f in covers if f in required and f not in local]
                    if newly:
                        producers.append({
                            "p_pc": pc,
                            "mnemonic": insn_mnemonic,
                            "kind": kind,
                            "covers": "".join(sorted(newly)),
                            "c_source": c_source,
                            "in_it": bool(in_it),
                            "dist": steps - 1,
                            "bbs": bbs_crossed,
                        })
                        for f in newly:
                            local[f] = pc
                if len(local) == len(required):
                    return finish("covered", local, producers, bbs_crossed, calls)
            plist = predecessors.get(bb, ())
            if bb in visited:
                return finish("ml_loop", local, producers, bbs_crossed, calls)
            if not plist:
                return finish("live_in", local, producers, bbs_crossed, calls)
            if bbs_crossed >= _PRODUCER_MAX_BBS:
                return finish("ml_bbs", local, producers, bbs_crossed, calls)
            if len(plist) > _PRODUCER_MAX_PRED_FAN:
                return finish("ml_preds", local, producers, bbs_crossed, calls)
            visited = visited | {bb}
            outs = [
                walk(int(p), len(static_bbs.get(int(p)) or ()) - 1, steps, bbs_crossed + 1, calls, local, visited)
                for p in plist
            ]
            covered_outs = [o for o in outs if o["status"] == "covered"]
            if len(covered_outs) == len(outs):
                best = max(covered_outs, key=lambda o: len(o["producers"]))
                best["bbs"] = max(o["bbs"] for o in outs)
                return best
            if covered_outs:
                mixed = max(covered_outs, key=lambda o: len(o["producers"]))
                mixed["status"] = "partial"
                return mixed
            if all(o["status"] == "live_in" for o in outs):
                return max(outs, key=lambda o: len(o["producers"]))
            statuses = {o["status"] for o in outs}
            rep = max(outs, key=lambda o: len(o["producers"]))
            if len(statuses) > 1:
                rep["status"] = "ml_mixed"
            return rep

        result = walk(bb_addr, len(instructions) - 2, 0, 0, 0, {}, frozenset())
        status = result["status"]
        if status == "covered":
            paths = "same_bb" if result["bbs"] == 0 else "all_covered"
        elif status == "live_in":
            paths = "live_in"
        elif status == "partial":
            paths = "partial"
        else:
            paths = "method_limited"

        producers_tuple = tuple(
            (
                int(p["p_pc"]),
                str(p["mnemonic"]),
                str(p["covers"]),
                str(p["c_source"]) if p["c_source"] is not None else "",
                bool(p["in_it"]),
                int(p["dist"]),
                int(p["bbs"]),
            )
            for p in result["producers"]
        )
        # VMRS/MSR 显式写者中标：保留 producers 元数据，paths 打标签，
        # 值合成通道据此跳过（r18b §4.3 裁定：z3 FP 可建模 0/58179，不做）。
        producer_mnemonics = {p[1] for p in producers_tuple}
        if "VMRS" in producer_mnemonics:
            paths = "label_fp_opaque"
        elif "MSR" in producer_mnemonics:
            paths = "label_msr"
        index[branch_pc] = (producers_tuple, paths, int(result["calls"]))

    return index


@dataclass
class PreparedFirmware:
    firmware_path: Path
    result: object
    static_bbs: Dict[int, List[Dict[str, object]]]
    static_bb_set: Set[int]
    instruction_to_bb: Dict[int, int]
    instruction_lookup: Dict[int, Dict[str, object]]
    branch_instruction_by_bb: Dict[int, Dict[str, object]]
    compare_lookup: Dict[int, Dict[str, object]]
    static_successors: Dict[int, Set[int]]
    refined_basic_blocks: List[BasicBlock]
    conditional_branch_bbs: List[BasicBlock]
    ghidra_total_bbs: int
    # r19：Bcc 站点 → NZCV 产生者记录（见 _build_producer_index）。纯固件静态
    # 属性，随 cache_identity 走；运行时按 producer_index.get(branch_pc) 惰性取。
    producer_index: Dict[int, Tuple[Tuple[Tuple[int, str, str, str, bool, int, int], ...], str, int]] = field(
        default_factory=dict
    )
    valid_bb_path: Optional[Path] = None
    valid_bb_set: Set[int] = field(default_factory=set)
    # angr 从入口可达 BB 基线（可选分母，见 lsgemu.angr_reachability）：
    # LSGEMU_ANGR_REACHABILITY=1 且 <stem>_angr_reachable.txt 存在时非 None。
    # None/空集表示未启用，报告字段保持 null（向后兼容）。
    angr_reachable_bb_path: Optional[Path] = None
    angr_reachable_bb_set: Optional[Set[int]] = None
    loop_exit_iteration_hints: Dict[int, int] = field(default_factory=dict)
    symbols_by_addr: Dict[int, str] = field(default_factory=dict)
    symbols_by_name: Dict[str, int] = field(default_factory=dict)
    static_mmio_accesses: List[Dict[str, object]] = field(default_factory=list)
    function_mmio_summaries: Dict[int, Dict[str, object]] = field(default_factory=dict)
    static_mmio_analysis_version: int = 0
    raw_load_base: Optional[int] = None
    execution_thumb_override: Optional[bool] = None
    firmware_sha256: str = ""
    toolchain_fingerprint: Dict[str, object] = field(default_factory=dict)
    static_cache_identity: Dict[str, object] = field(default_factory=dict)
    snapshot_page_store: SnapshotPageStore = field(
        default_factory=SnapshotPageStore,
        repr=False,
        compare=False,
    )
    snapshot_metadata_store: SnapshotMetadataStore = field(
        default_factory=SnapshotMetadataStore,
        repr=False,
        compare=False,
    )

    def rebuild_raw_arm_static_view(
        self,
        entry_point: int,
        thumb: bool = False,
        load_base: Optional[int] = None,
    ) -> int:
        """Rebuild static BBs for raw BINs when Ghidra decoded a vendor prefix.

        Some raw MCU images are not Cortex-M vector-table images. They carry a
        vendor prefix and then plain ARM code at a later file offset. Ghidra's
        raw import can decode the prefix as Thumb and produce a static view that
        does not contain the real entry PC. In that case dynamic execution works
        but coverage and branch reasoning cannot map PCs back to BBs. This
        fallback builds a conservative Capstone CFG from the explicit entry.
        """
        try:
            from capstone import CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_LITTLE_ENDIAN, CS_MODE_THUMB, Cs
        except Exception:
            return 0

        entry = int(entry_point) & ~1
        try:
            data = self.firmware_path.read_bytes()
        except Exception:
            return 0
        file_offset = entry
        if load_base is not None:
            base = int(load_base) & 0xFFFFFFFF
            if base <= entry < base + len(data):
                file_offset = entry - base
        if file_offset < 0 or file_offset >= len(data):
            return 0

        mode = CS_MODE_THUMB if thumb else CS_MODE_ARM
        mode |= CS_MODE_LITTLE_ENDIAN
        md = Cs(CS_ARCH_ARM, mode)

        instruction_by_addr: Dict[int, SimpleNamespace] = {}
        max_instructions = max(1, min(250000, (len(data) - file_offset) // (2 if thumb else 4)))
        max_seed_bytes = max(0x1000, min(0x20000, len(data) - file_offset))

        def image_offset(address: int) -> Optional[int]:
            address = int(address) & ~1
            offset = address
            if load_base is not None:
                base = int(load_base) & 0xFFFFFFFF
                if base <= address < base + len(data):
                    offset = address - base
            if 0 <= offset < len(data):
                return offset
            return None

        def enqueue_target(queue: deque, target: Optional[int]) -> None:
            if target is None:
                return
            target = int(target) & ~1
            if image_offset(target) is not None:
                queue.append(target)

        decode_queue: deque[int] = deque([entry])
        decoded_seeds: Set[int] = set()
        while decode_queue and len(instruction_by_addr) < max_instructions:
            seed = int(decode_queue.popleft()) & ~1
            if seed in decoded_seeds:
                continue
            seed_offset = image_offset(seed)
            if seed_offset is None:
                continue
            decoded_seeds.add(seed)
            remaining = len(data) - seed_offset
            window = data[seed_offset:seed_offset + min(max_seed_bytes, remaining)]
            previous_end = None
            for insn in md.disasm(window, seed):
                address = int(insn.address)
                size = int(insn.size or (2 if thumb else 4))
                if previous_end is not None and previous_end < address:
                    break
                previous_end = address + size

                if address not in instruction_by_addr:
                    instruction_by_addr[address] = SimpleNamespace(
                        address=address,
                        mnemonic=str(insn.mnemonic),
                        op_str=str(insn.op_str),
                        size=size,
                        bytes=bytes(insn.bytes),
                    )
                    if len(instruction_by_addr) >= max_instructions:
                        break

                mnemonic = _normalize_mnemonic(insn.mnemonic)
                target = _parse_direct_target(insn.op_str)
                fallthrough = address + size
                branch_like = (
                    mnemonic in CONDITIONAL_BRANCH_MNEMONICS
                    or mnemonic in CALL_SPLIT_MNEMONICS
                    or mnemonic in {"B", "BX", "BLX", "BXJ"}
                )

                if target is not None and branch_like:
                    enqueue_target(decode_queue, target)
                if mnemonic in CONDITIONAL_BRANCH_MNEMONICS:
                    enqueue_target(decode_queue, fallthrough)
                    continue
                if mnemonic in CALL_SPLIT_MNEMONICS:
                    enqueue_target(decode_queue, fallthrough)
                    # Direct calls return to fallthrough, so keep decoding the
                    # current island. Register-indirect BLX cannot be followed.
                    if target is None and mnemonic == "BLX":
                        break
                    continue
                if mnemonic == "B":
                    break
                if mnemonic in {"BX", "BXJ"} or _is_return_like_instruction(_instruction_dict(insn)):
                    break
        if not instruction_by_addr:
            return 0

        instructions = [instruction_by_addr[address] for address in sorted(instruction_by_addr)]
        addresses = [int(insn.address) for insn in instructions]
        next_by_addr = {}
        for index in range(len(addresses) - 1):
            current = instruction_by_addr[addresses[index]]
            expected_next = addresses[index] + int(getattr(current, "size", 2 if thumb else 4) or (2 if thumb else 4))
            if expected_next == addresses[index + 1]:
                next_by_addr[addresses[index]] = addresses[index + 1]

        leaders: Set[int] = {entry}
        for insn in instructions:
            address = int(insn.address)
            mnemonic = _normalize_mnemonic(insn.mnemonic)
            target = _parse_direct_target(insn.op_str)
            fallthrough = next_by_addr.get(address)

            if target is not None and int(target) in instruction_by_addr:
                if mnemonic in CONDITIONAL_BRANCH_MNEMONICS or mnemonic in CALL_SPLIT_MNEMONICS or mnemonic in {"B"}:
                    leaders.add(int(target))
            if fallthrough is not None and (
                mnemonic in CONDITIONAL_BRANCH_MNEMONICS
                or mnemonic in CALL_SPLIT_MNEMONICS
                or mnemonic in {"B", "BX", "BLX"}
                or _is_return_like_instruction(_instruction_dict(insn))
            ):
                leaders.add(int(fallthrough))

        leaders = {leader for leader in leaders if leader in instruction_by_addr}
        leader_list = sorted(leaders)
        index_by_addr = {address: index for index, address in enumerate(addresses)}
        basic_blocks: List[BasicBlock] = []

        for leader_index, leader in enumerate(leader_list):
            start_index = index_by_addr.get(leader)
            if start_index is None:
                continue
            next_leader = leader_list[leader_index + 1] if leader_index + 1 < len(leader_list) else None
            cursor = start_index
            block_insns = []
            while cursor < len(instructions):
                insn = instructions[cursor]
                address = int(insn.address)
                if next_leader is not None and address >= next_leader:
                    break
                block_insns.append(insn)
                mnemonic = _normalize_mnemonic(insn.mnemonic)
                if (
                    mnemonic in CONDITIONAL_BRANCH_MNEMONICS
                    or mnemonic in CALL_SPLIT_MNEMONICS
                    or mnemonic in {"B", "BX", "BLX"}
                    or _is_return_like_instruction(_instruction_dict(insn))
                ):
                    break
                cursor += 1

            if not block_insns:
                continue

            last = block_insns[-1]
            last_addr = int(last.address)
            fallthrough = next_by_addr.get(last_addr)
            target = _parse_direct_target(last.op_str)
            mnemonic = _normalize_mnemonic(last.mnemonic)
            successors: List[int] = []

            def add_successor(value: Optional[int]) -> None:
                if value is None:
                    return
                value = int(value) & ~1
                if value in instruction_by_addr and value not in successors:
                    successors.append(value)

            if mnemonic in CONDITIONAL_BRANCH_MNEMONICS:
                add_successor(target)
                add_successor(fallthrough)
            elif mnemonic in CALL_SPLIT_MNEMONICS:
                add_successor(fallthrough)
                add_successor(target)
            elif mnemonic == "B":
                add_successor(target)
            elif mnemonic in {"BX", "BLX"} or _is_return_like_instruction(_instruction_dict(last)):
                pass
            else:
                add_successor(fallthrough)

            basic_blocks.append(BasicBlock(
                start_addr=int(block_insns[0].address),
                end_addr=last_addr,
                size=(last_addr - int(block_insns[0].address)) + int(last.size),
                instruction_count=len(block_insns),
                successors=successors,
                predecessors=[],
            ))

        if not basic_blocks:
            return 0

        result_view = SimpleNamespace(
            instructions=instructions,
            basic_blocks=basic_blocks,
        )
        static_views = self._build_static_views(result_view, self.firmware_path)
        self.static_bbs = static_views.static_bbs
        self.static_bb_set = static_views.static_bb_set
        self.instruction_to_bb = static_views.instruction_to_bb
        self.instruction_lookup = static_views.instruction_lookup
        self.branch_instruction_by_bb = static_views.branch_instruction_by_bb
        self.compare_lookup = static_views.compare_lookup
        self.static_successors = static_views.static_successors
        self.refined_basic_blocks = static_views.refined_basic_blocks
        self.conditional_branch_bbs = static_views.conditional_branch_bbs
        self.producer_index = static_views.producer_index
        self.ghidra_total_bbs = len(basic_blocks)
        self.valid_bb_set = {bb for bb in self.valid_bb_set if bb in self.static_bb_set}

        self.result.instructions = instructions
        self.result.basic_blocks = basic_blocks
        self.result.total_instructions = len(instructions)
        self.result.total_basic_blocks = len(basic_blocks)
        self.result.min_address = min(self.instruction_lookup)
        self.result.max_address = max(self.instruction_lookup)
        self.result.arch_info.entry_point = entry
        self.result.arch_info.base_addr = int(getattr(self.result.arch_info, "base_addr", 0) or 0)
        self.result.arch_info.code_size = len(data)
        self.raw_load_base = int(load_base) & 0xFFFFFFFF if load_base is not None else None
        return len(self.static_bb_set)

    @classmethod
    def from_firmware(
        cls,
        firmware_path: str | Path,
        use_ghidra: bool = True,
        *,
        entry_point_override: Optional[int] = None,
        load_base_override: Optional[int] = None,
        raw_static_mode: str = "auto",
        execution_thumb_override: Optional[bool] = None,
        resolve_valid_bb_metadata: bool = True,
    ) -> "PreparedFirmware":
        firmware_path = Path(firmware_path).resolve()
        cache_path = _static_cache_path(firmware_path)
        stat = firmware_path.stat()
        firmware_sha256 = sha256_file(firmware_path)
        if resolve_valid_bb_metadata:
            valid_bb_path, valid_bb_set = _resolve_valid_bb_metadata(firmware_path)
        else:
            valid_bb_path, valid_bb_set = None, set()
        # angr 可达基线按需懒加载：开关关闭/文件缺失时为 (None, None)。
        # 该视图不影响静态分析结果，故不参与缓存 identity（hydrate 时刷新）。
        angr_reachable_path, angr_reachable_set = load_angr_reachable_bb_set(
            firmware_path
        )
        toolchain = collect_toolchain_fingerprint(valid_bb_path=valid_bb_path)
        analysis_config = {
            "use_ghidra": bool(use_ghidra),
            "entry_point_override": (
                int(entry_point_override) & 0xFFFFFFFF
                if entry_point_override is not None
                else None
            ),
            "load_base_override": (
                int(load_base_override) & 0xFFFFFFFF
                if load_base_override is not None
                else None
            ),
            "raw_static_mode": str(raw_static_mode or "auto").lower(),
        }
        if execution_thumb_override is not None:
            analysis_config["execution_thumb_override"] = bool(
                execution_thumb_override
            )
        if not resolve_valid_bb_metadata:
            analysis_config.update({
                "resolve_valid_bb_metadata": False,
                "static_view_valid_hint_policy": "disabled_v1",
            })
        identity_payload = {
            "schema": STATIC_ANALYSIS_SCHEMA,
            "firmware_path": str(firmware_path),
            "firmware_sha256": firmware_sha256,
            "firmware_size": int(stat.st_size),
            "valid_bb_path": str(valid_bb_path) if valid_bb_path else None,
            "valid_bb_sha256": (
                str((toolchain.get("valid_basic_blocks", {}) or {}).get("sha256") or "")
            ),
            "analysis_config": analysis_config,
            "toolchain_fingerprint": str(toolchain.get("fingerprint") or ""),
            "lightweight_mmio_analysis_version": LIGHTWEIGHT_MMIO_ANALYSIS_VERSION,
        }
        identity_payload["identity_hash"] = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

        def cache_payload(prepared: "PreparedFirmware") -> Dict[str, object]:
            return {
                "version": STATIC_CACHE_VERSION,
                "schema": STATIC_ANALYSIS_SCHEMA,
                "firmware_mtime_ns": int(stat.st_mtime_ns),
                "firmware_size": int(stat.st_size),
                "firmware_sha256": firmware_sha256,
                "cache_identity": identity_payload,
                "prepared": prepared,
            }

        def hydrate_cached_prepared(prepared: "PreparedFirmware") -> bool:
            changed = False
            for attribute, value in (
                ("firmware_sha256", firmware_sha256),
                ("toolchain_fingerprint", dict(toolchain)),
                ("static_cache_identity", dict(identity_payload)),
                ("valid_bb_path", valid_bb_path),
                ("valid_bb_set", set(valid_bb_set)),
                ("angr_reachable_bb_path", angr_reachable_path),
                ("angr_reachable_bb_set", angr_reachable_set),
                (
                    "execution_thumb_override",
                    bool(execution_thumb_override)
                    if execution_thumb_override is not None
                    else None,
                ),
            ):
                if getattr(prepared, attribute, None) != value:
                    setattr(prepared, attribute, value)
                    changed = True
            if not hasattr(prepared, "loop_exit_iteration_hints"):
                prepared.loop_exit_iteration_hints = {}
                changed = True
            if not hasattr(prepared, "snapshot_page_store"):
                prepared.snapshot_page_store = SnapshotPageStore()
            if not hasattr(prepared, "snapshot_metadata_store"):
                prepared.snapshot_metadata_store = SnapshotMetadataStore()
            if (
                not hasattr(prepared, "symbols_by_addr")
                or not getattr(prepared, "symbols_by_addr", None)
                or not hasattr(prepared, "symbols_by_name")
                or not getattr(prepared, "symbols_by_name", None)
            ):
                prepared.symbols_by_addr, prepared.symbols_by_name = cls._load_elf_symbols(
                    firmware_path
                )
                changed = True
            if not hasattr(prepared, "raw_load_base"):
                prepared.raw_load_base = None
                changed = True
            previous_mmio_version = int(
                getattr(prepared, "static_mmio_analysis_version", 0) or 0
            )
            prepared._ensure_lightweight_mmio_analysis()
            if int(getattr(prepared, "static_mmio_analysis_version", 0) or 0) != previous_mmio_version:
                changed = True
            prepared._ensure_producer_index()
            return changed

        if cache_path.exists():
            try:
                payload = _load_static_cache_pickle(cache_path)
                if (
                    payload.get("version") == STATIC_CACHE_VERSION
                    and payload.get("schema") == STATIC_ANALYSIS_SCHEMA
                    and payload.get("firmware_mtime_ns") == int(stat.st_mtime_ns)
                    and payload.get("firmware_size") == int(stat.st_size)
                    and payload.get("firmware_sha256") == firmware_sha256
                    and (payload.get("cache_identity") or {}).get("identity_hash")
                    == identity_payload["identity_hash"]
                    and getattr(payload.get("prepared"), "total_bbs", 0) > 0
                ):
                    prepared = payload["prepared"]
                    if hydrate_cached_prepared(prepared):
                        try:
                            atomic_pickle_dump(cache_payload(prepared), cache_path)
                        except Exception:
                            pass
                    return prepared

                legacy_cache_valid = bool(
                    payload.get("version") == STATIC_CACHE_VERSION - 1
                    and payload.get("firmware_mtime") == stat.st_mtime
                    and payload.get("firmware_size") == stat.st_size
                    and payload.get("firmware_sha256") == firmware_sha256
                    and getattr(payload.get("prepared"), "total_bbs", 0) > 0
                )
                if legacy_cache_valid:
                    prepared = payload["prepared"]
                    hydrate_cached_prepared(prepared)
                    atomic_pickle_dump(cache_payload(prepared), cache_path)
                    logger.info(
                        "Upgraded legacy static cache v%d -> v%d: %s",
                        STATIC_CACHE_VERSION - 1,
                        STATIC_CACHE_VERSION,
                        cache_path,
                    )
                    return prepared
            except Exception as cache_exc:
                # 缓存损坏/校验失败必须可见：静默吞掉会让每次运行都
                # 重做全量 Ghidra 分析（大固件 20+ 分钟），且无从排查。
                logger.warning(
                    "静态缓存加载失败，将重做全量分析 (%s): %s",
                    type(cache_exc).__name__,
                    cache_exc,
                )

        analyzer = FirmwareAnalyzer(use_ghidra=use_ghidra)
        result = analyzer.analyze(str(firmware_path))
        static_views = cls._build_static_views(
            result,
            firmware_path,
            valid_hint_set=(valid_bb_set if resolve_valid_bb_metadata else set()),
        )
        if not result.basic_blocks:
            raise RuntimeError(f"静态分析未产生基本块，拒绝缓存失败结果: {firmware_path}")

        symbols_by_addr, symbols_by_name = cls._load_elf_symbols(firmware_path)

        prepared = cls(
            firmware_path=firmware_path,
            result=result,
            static_bbs=static_views.static_bbs,
            static_bb_set=static_views.static_bb_set,
            instruction_to_bb=static_views.instruction_to_bb,
            instruction_lookup=static_views.instruction_lookup,
            branch_instruction_by_bb=static_views.branch_instruction_by_bb,
            compare_lookup=static_views.compare_lookup,
            static_successors=static_views.static_successors,
            refined_basic_blocks=static_views.refined_basic_blocks,
            conditional_branch_bbs=static_views.conditional_branch_bbs,
            producer_index=static_views.producer_index,
            ghidra_total_bbs=len(result.basic_blocks),
            valid_bb_path=valid_bb_path,
            valid_bb_set=valid_bb_set,
            angr_reachable_bb_path=angr_reachable_path,
            angr_reachable_bb_set=angr_reachable_set,
            symbols_by_addr=symbols_by_addr,
            symbols_by_name=symbols_by_name,
            execution_thumb_override=(
                bool(execution_thumb_override)
                if execution_thumb_override is not None
                else None
            ),
            firmware_sha256=firmware_sha256,
            toolchain_fingerprint=dict(toolchain),
            static_cache_identity=dict(identity_payload),
        )
        prepared._ensure_lightweight_mmio_analysis()
        try:
            atomic_pickle_dump(cache_payload(prepared), cache_path)
        except Exception:
            pass
        return prepared

    def _ensure_producer_index(self) -> None:
        """Lazily (re)build the NZCV producer index from the current views.

        Fresh v15 builds already carry it; this covers hand-modified caches and
        any future view mutation path that skips _build_static_views.
        absorb_runtime_basic_blocks() merges runtime-discovered BBs into the
        views without rebuilding the index — new branch sites simply miss the
        index and consumers fall back to their legacy heuristics.
        """
        existing = getattr(self, "producer_index", None)
        if isinstance(existing, dict) and existing:
            return
        try:
            self.producer_index = _build_producer_index(
                self.static_bbs, getattr(self, "static_successors", {}) or {}
            )
        except Exception as exc:
            logger.debug("NZCV 产生者索引构建失败: %s", exc)
            self.producer_index = {}

    def _ensure_lightweight_mmio_analysis(self) -> None:
        if not hasattr(self, "static_mmio_accesses") or self.static_mmio_accesses is None:
            self.static_mmio_accesses = []
        if not hasattr(self, "function_mmio_summaries") or self.function_mmio_summaries is None:
            self.function_mmio_summaries = {}
        if not hasattr(self, "static_mmio_analysis_version"):
            self.static_mmio_analysis_version = 0
        if (
            int(getattr(self, "static_mmio_analysis_version", 0) or 0) >= LIGHTWEIGHT_MMIO_ANALYSIS_VERSION
            and (self.static_mmio_accesses or self.function_mmio_summaries)
        ):
            return
        try:
            accesses, summaries = analyze_static_mmio(
                self.static_bbs,
                instruction_lookup=self.instruction_lookup,
                symbols_by_addr=getattr(self, "symbols_by_addr", {}) or {},
                thumb_mode=self._infer_static_thumb_mode(),
                firmware_path=self.firmware_path,
                raw_load_base=getattr(self, "raw_load_base", None),
            )
            self.static_mmio_accesses = accesses
            self.function_mmio_summaries = summaries
            self.static_mmio_analysis_version = LIGHTWEIGHT_MMIO_ANALYSIS_VERSION
        except Exception as exc:
            logger.debug("轻量静态MMIO分析失败: %s", exc)
            self.static_mmio_accesses = []
            self.function_mmio_summaries = {}
            self.static_mmio_analysis_version = 0

    def _infer_static_thumb_mode(self) -> bool:
        if getattr(self, "execution_thumb_override", None) is not None:
            return bool(self.execution_thumb_override)
        try:
            entry = int(getattr(getattr(self.result, "arch_info", None), "entry_point", 0) or 0)
        except Exception:
            entry = 0
        if entry & 1:
            return True
        # An even entry point is an ARM-state entry for both raw BIN and ELF
        # images. Do not force Thumb merely from the .elf suffix: ARM-state
        # ELF firmware would be statically disassembled with the wrong ISA.
        return False

    @staticmethod
    def _load_elf_symbols(firmware_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
        symbols_by_addr: Dict[int, str] = {}
        symbols_by_name: Dict[str, int] = {}
        tools = ("arm-none-eabi-nm", "llvm-nm", "nm")
        output = ""
        for tool in tools:
            try:
                completed = subprocess.run(
                    [tool, "-n", str(firmware_path)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=30,
                )
            except Exception:
                continue
            if completed.returncode == 0 and completed.stdout:
                output = completed.stdout
                break
        if not output:
            return symbols_by_addr, symbols_by_name
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                address = int(parts[0], 16)
            except ValueError:
                continue
            symbol_type = parts[1]
            name = parts[2]
            if symbol_type.lower() not in {"t", "w"}:
                continue
            normalized = int(address) & ~1
            symbols_by_addr.setdefault(normalized, name)
            symbols_by_name.setdefault(name, normalized)
        return symbols_by_addr, symbols_by_name

    @staticmethod
    def _build_static_views(
        result,
        firmware_path: Optional[Path] = None,
        valid_hint_set: Optional[Set[int]] = None,
    ) -> StaticViews:
        # Build a refined static BB view:
        # 1. preserve original Ghidra BB starts,
        # 2. split at post-call fallthroughs,
        # 3. split at valid_basic_blocks.txt hints when available.
        basic_blocks = sorted(result.basic_blocks, key=lambda bb: bb.start_addr)
        instructions = sorted(result.instructions, key=lambda insn: insn.address)
        normalized_instructions = [_instruction_dict(insn) for insn in instructions]
        instruction_lookup = {
            normalized["address"]: normalized
            for normalized in normalized_instructions
        }
        if valid_hint_set is None:
            resolved_hints: Set[int] = set()
            if firmware_path is not None:
                _valid_bb_path, resolved_hints = _resolve_valid_bb_metadata(
                    firmware_path
                )
        else:
            resolved_hints = set(valid_hint_set)
        valid_hint_set = {
            addr for addr in resolved_hints
            if addr in instruction_lookup
        }

        original_bb_instructions: Dict[int, List[Dict[str, object]]] = {}
        insn_index = 0
        total_instructions = len(instructions)
        for bb in basic_blocks:
            while insn_index < total_instructions and instructions[insn_index].address < bb.start_addr:
                insn_index += 1

            cursor = insn_index
            bb_instructions: List[Dict[str, object]] = []
            while cursor < total_instructions and instructions[cursor].address <= bb.end_addr:
                normalized = normalized_instructions[cursor]
                bb_instructions.append(normalized)
                cursor += 1
            original_bb_instructions[bb.start_addr] = bb_instructions
            insn_index = cursor

        refined_segments: List[Tuple[int, int, List[Dict[str, object]], List[int]]] = []
        for bb in basic_blocks:
            bb_instructions = original_bb_instructions.get(bb.start_addr, [])
            if not bb_instructions:
                continue

            index_by_address = {
                insn["address"]: index
                for index, insn in enumerate(bb_instructions)
            }
            split_points = {bb.start_addr}

            for index, insn in enumerate(bb_instructions[:-1]):
                if _should_split_after_mnemonic(insn.get("mnemonic")):
                    split_points.add(bb_instructions[index + 1]["address"])

            for insn in bb_instructions[1:]:
                address = insn["address"]
                if address in valid_hint_set:
                    split_points.add(address)

            ordered_starts = [
                insn["address"]
                for insn in bb_instructions
                if insn["address"] in split_points
            ]
            ordered_starts.sort()

            def contiguous_fallthrough(last_insn: Dict[str, object]) -> Optional[int]:
                natural = (
                    int(last_insn.get("address", 0) or 0)
                    + int(last_insn.get("size", 2) or 2)
                )
                if natural in instruction_lookup:
                    return natural
                return None

            def semantic_successors(
                last_insn: Dict[str, object],
                default_successors: List[int],
                fallthrough_addr: Optional[int],
            ) -> List[int]:
                mnemonic = _normalize_mnemonic(last_insn.get("mnemonic"))
                parsed_target = _parse_direct_target(last_insn.get("operands"))
                ordered: List[int] = []

                def append_successor(value: Optional[int]):
                    if value is None:
                        return
                    value = int(value)
                    if value not in ordered:
                        ordered.append(value)

                if mnemonic in {"TBB", "TBH"}:
                    for successor in default_successors:
                        append_successor(successor)
                    return ordered

                if mnemonic in CALL_SPLIT_MNEMONICS:
                    append_successor(fallthrough_addr)
                    append_successor(parsed_target)
                    if ordered:
                        return ordered

                if mnemonic in CONDITIONAL_BRANCH_MNEMONICS:
                    append_successor(parsed_target)
                    append_successor(fallthrough_addr)
                    if ordered:
                        return ordered

                if mnemonic == "B":
                    append_successor(parsed_target)
                    if ordered:
                        return ordered

                if _is_return_like_instruction(last_insn):
                    return ordered

                if fallthrough_addr is not None:
                    append_successor(fallthrough_addr)
                    return ordered

                for successor in default_successors:
                    append_successor(successor)
                return ordered

            for index, start_addr in enumerate(ordered_starts):
                start_index = index_by_address[start_addr]
                end_index = (
                    index_by_address[ordered_starts[index + 1]] - 1
                    if index + 1 < len(ordered_starts)
                    else len(bb_instructions) - 1
                )
                segment_instructions = bb_instructions[start_index:end_index + 1]
                if not segment_instructions:
                    continue
                default_successors = (
                    [ordered_starts[index + 1]]
                    if index + 1 < len(ordered_starts)
                    else list(getattr(bb, "successors", []))
                )
                last_segment_insn = segment_instructions[-1]
                fallthrough_addr = contiguous_fallthrough(last_segment_insn)
                raw_successors = semantic_successors(
                    last_segment_insn,
                    list(default_successors),
                    fallthrough_addr,
                )
                refined_segments.append((
                    segment_instructions[0]["address"],
                    segment_instructions[-1]["address"],
                    segment_instructions,
                    raw_successors,
                ))

        static_bbs: Dict[int, List[Dict[str, object]]] = {
            start_addr: segment_instructions
            for start_addr, _end_addr, segment_instructions, _successors in refined_segments
        }
        static_bb_set = set(static_bbs.keys())
        instruction_to_bb: Dict[int, int] = {}
        for bb_addr, bb_instructions in static_bbs.items():
            for insn in bb_instructions:
                instruction_to_bb[insn["address"]] = bb_addr

        static_successors: Dict[int, Set[int]] = {bb_addr: set() for bb_addr in static_bbs}
        for bb_addr, _end_addr, _segment_instructions, raw_successors in refined_segments:
            for successor in raw_successors:
                resolved = instruction_to_bb.get(successor)
                if resolved is None and successor in static_bb_set:
                    resolved = successor
                if resolved is not None and resolved in static_bb_set:
                    static_successors[bb_addr].add(resolved)

        block_objects: Dict[int, BasicBlock] = {}
        for bb_addr, bb_instructions in static_bbs.items():
            last_insn = bb_instructions[-1]
            end_addr = last_insn["address"]
            block_size = (end_addr - bb_addr) + int(last_insn.get("size", 2) or 2)
            block_objects[bb_addr] = BasicBlock(
                start_addr=bb_addr,
                end_addr=end_addr,
                size=block_size,
                instruction_count=len(bb_instructions),
                successors=sorted(static_successors.get(bb_addr, set())),
                predecessors=[],
            )

        for bb_addr, successors in static_successors.items():
            for successor in successors:
                predecessor_list = block_objects[successor].predecessors
                if bb_addr not in predecessor_list:
                    predecessor_list.append(bb_addr)
        for block in block_objects.values():
            block.predecessors.sort()

        branch_instruction_by_bb: Dict[int, Dict[str, object]] = {}
        compare_lookup: Dict[int, Dict[str, object]] = {}
        for bb_addr, bb_instructions in static_bbs.items():
            if not bb_instructions:
                continue
            branch_instruction_by_bb[bb_addr] = bb_instructions[-1]
            for previous in reversed(bb_instructions[max(0, len(bb_instructions) - 11):-1]):
                if _normalize_mnemonic(previous.get("mnemonic")) in COMPARE_MNEMONICS:
                    compare_lookup[bb_instructions[-1]["address"]] = previous
                    break

        refined_basic_blocks = [block_objects[addr] for addr in sorted(block_objects)]
        conditional_branch_bbs = [
            block
            for block in refined_basic_blocks
            if len(block.successors) == 2
            and _normalize_mnemonic(branch_instruction_by_bb.get(block.start_addr, {}).get("mnemonic")) in CONDITIONAL_BRANCH_MNEMONICS
        ]

        return StaticViews(
            static_bbs=static_bbs,
            static_bb_set=static_bb_set,
            instruction_to_bb=instruction_to_bb,
            instruction_lookup=instruction_lookup,
            branch_instruction_by_bb=branch_instruction_by_bb,
            compare_lookup=compare_lookup,
            static_successors=static_successors,
            refined_basic_blocks=refined_basic_blocks,
            conditional_branch_bbs=conditional_branch_bbs,
            producer_index=_build_producer_index(static_bbs, static_successors),
        )

    def new_emulator(
        self,
        max_snapshots: int = 5,
        iteration: int = 0,
        llm_config_path: Optional[str | Path] = None,
        constraint_json_path: Optional[str | Path] = None,
        branch_mmio_file_mode: Optional[str] = None,
    ) -> IntelligentEmulator:
        shared_page_store = getattr(self, "snapshot_page_store", None)
        if not isinstance(shared_page_store, SnapshotPageStore):
            shared_page_store = SnapshotPageStore()
            self.snapshot_page_store = shared_page_store
        # P0-A (r7): a store constructed before the entrypoint configured the
        # run-level snapshot storage env (e.g. PreparedFirmware restored from
        # the static-analysis pickle cache) never re-checked that env.  Retry
        # disk backing here so captured pages can spill to
        # .snapshot_store/<token>/snapshot-pages-*.bin; idempotent when the
        # store already enabled backing.
        shared_page_store.configure_disk_backing()
        shared_metadata_store = getattr(self, "snapshot_metadata_store", None)
        if not isinstance(shared_metadata_store, SnapshotMetadataStore):
            shared_metadata_store = SnapshotMetadataStore()
            self.snapshot_metadata_store = shared_metadata_store
        shared_blob_store = SnapshotBlobStore.from_environment()
        emulator = None
        try:
            emulator = IntelligentEmulator(
                firmware_path=str(self.firmware_path),
                static_bbs=self.static_bbs,
                max_snapshots=max_snapshots,
                iteration=iteration,
                llm_config_path=str(llm_config_path) if llm_config_path else None,
                constraint_json_path=str(constraint_json_path) if constraint_json_path else None,
                branch_mmio_file_mode=branch_mmio_file_mode,
                raw_load_base=self.raw_load_base,
                static_mmio_accesses=list(getattr(self, "static_mmio_accesses", []) or []),
                function_mmio_summaries=dict(getattr(self, "function_mmio_summaries", {}) or {}),
                execution_thumb_override=self.execution_thumb_override,
                snapshot_page_store=shared_page_store,
                snapshot_metadata_store=shared_metadata_store,
                snapshot_blob_store=shared_blob_store,
            )
            emulator.symbols_by_addr = dict(getattr(self, "symbols_by_addr", {}) or {})
            emulator.loop_exit_iteration_hints = (
                dict(getattr(self, "loop_exit_iteration_hints", {}) or {})
                if _loop_exit_hints_enabled()
                else {}
            )
            emulator.loop_intervention_threshold_cache.clear()
            emulator.setup_memory()
            emulator.load_firmware()
            emulator.register_hooks()
            return emulator
        except BaseException:
            # ``setup_memory``, firmware loading, or partial hook registration
            # may fail after the Unicorn engine has been allocated.  Close the
            # partially initialized owner here so a failed replay cannot leave
            # a native engine outside the runner's lifecycle registry.
            if emulator is not None:
                try:
                    close = getattr(emulator, "close", None)
                    if callable(close):
                        close()
                except Exception as close_error:
                    logger.debug(
                        "failed to close partially initialized emulator: %s",
                        close_error,
                    )
            raise

    @property
    def entry_point(self) -> int:
        return self.result.arch_info.entry_point

    @property
    def total_bbs(self) -> int:
        return len(self.static_bb_set)

    @property
    def valid_total_bbs(self) -> int:
        return len(self.valid_bb_set)

    @property
    def reachable_total_bbs(self) -> Optional[int]:
        """angr 可达基线 BB 总数；未启用时为 None（报告字段保持 null）。"""
        reachable = getattr(self, "angr_reachable_bb_set", None)
        return len(reachable) if reachable else None

    def resolve_basic_block(self, address: int) -> Optional[int]:
        if address in self.instruction_to_bb:
            return self.instruction_to_bb[address]
        if address in self.static_bb_set:
            return address
        return None

    def absorb_runtime_basic_blocks(self, emulator: IntelligentEmulator) -> int:
        """Merge runtime-decoded raw BIN code islands into coverage/branch metadata."""
        runtime_bbs = getattr(emulator, "static_bbs", {}) or {}
        new_starts = sorted(set(runtime_bbs) - self.static_bb_set)
        if not new_starts:
            return 0

        for bb_addr in new_starts:
            instructions = list(runtime_bbs.get(bb_addr, []) or [])
            if not instructions:
                continue
            self.static_bbs[int(bb_addr)] = instructions
            self.static_bb_set.add(int(bb_addr))
            for insn in instructions:
                address = int(insn.get("address", 0) or 0)
                if not address:
                    continue
                self.instruction_lookup[address] = insn
                self.instruction_to_bb[address] = int(bb_addr)
            self.branch_instruction_by_bb[int(bb_addr)] = instructions[-1]
            for previous in reversed(instructions[max(0, len(instructions) - 11):-1]):
                if _normalize_mnemonic(previous.get("mnemonic")) in COMPARE_MNEMONICS:
                    self.compare_lookup[int(instructions[-1].get("address", bb_addr) or bb_addr)] = previous
                    break

        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            successors = self.static_successors.setdefault(int(bb_addr), set())
            last_insn = instructions[-1]
            mnemonic = _normalize_mnemonic(last_insn.get("mnemonic"))
            branch_pc = int(last_insn.get("address", bb_addr) or bb_addr)
            fallthrough = branch_pc + int(last_insn.get("size", 2) or 2)
            target = _parse_direct_target(last_insn.get("operands"))

            def add_resolved(value: Optional[int]) -> None:
                if value is None:
                    return
                resolved = self.resolve_basic_block(int(value) & ~1)
                if resolved is not None:
                    successors.add(int(resolved))

            if mnemonic in CONDITIONAL_BRANCH_MNEMONICS:
                add_resolved(target)
                add_resolved(fallthrough)
            elif mnemonic in CALL_SPLIT_MNEMONICS:
                add_resolved(fallthrough)
                add_resolved(target)
            elif mnemonic == "B":
                add_resolved(target)
            elif mnemonic not in {"BX", "BXJ"} and not _is_return_like_instruction(last_insn):
                add_resolved(fallthrough)

        existing_blocks = {int(block.start_addr): block for block in self.refined_basic_blocks}
        for bb_addr in new_starts:
            instructions = self.static_bbs.get(int(bb_addr), [])
            if not instructions or int(bb_addr) in existing_blocks:
                continue
            last_insn = instructions[-1]
            block = BasicBlock(
                start_addr=int(bb_addr),
                end_addr=int(last_insn.get("address", bb_addr) or bb_addr),
                size=(
                    int(last_insn.get("address", bb_addr) or bb_addr)
                    - int(bb_addr)
                    + int(last_insn.get("size", 2) or 2)
                ),
                instruction_count=len(instructions),
                successors=sorted(self.static_successors.get(int(bb_addr), set())),
                predecessors=[],
            )
            existing_blocks[int(bb_addr)] = block

        for block in existing_blocks.values():
            block.successors = sorted(self.static_successors.get(int(block.start_addr), set()))
            block.predecessors = []
        for bb_addr, successors in self.static_successors.items():
            for successor in successors:
                block = existing_blocks.get(int(successor))
                if block is not None and int(bb_addr) not in block.predecessors:
                    block.predecessors.append(int(bb_addr))
        for block in existing_blocks.values():
            block.predecessors.sort()

        self.refined_basic_blocks = [existing_blocks[addr] for addr in sorted(existing_blocks)]
        self.conditional_branch_bbs = [
            block
            for block in self.refined_basic_blocks
            if len(block.successors) == 2
            and _normalize_mnemonic(self.branch_instruction_by_bb.get(block.start_addr, {}).get("mnemonic"))
            in CONDITIONAL_BRANCH_MNEMONICS
        ]
        self.result.basic_blocks = self.refined_basic_blocks
        self.result.total_basic_blocks = len(self.static_bb_set)
        self.result.total_instructions = len(self.instruction_lookup)
        return len(new_starts)

    def get_branch_instruction(self, bb_start: int) -> Optional[Dict[str, object]]:
        return self.branch_instruction_by_bb.get(bb_start)

    def valid_coverage(self, covered_bbs: Iterable[int]) -> Set[int]:
        if not self.valid_bb_set:
            return set()
        return {bb for bb in covered_bbs if bb in self.valid_bb_set}

    def reachable_coverage(self, covered_bbs: Iterable[int]) -> Set[int]:
        """计算 angr 可达基线中被动态覆盖命中的 BB 集合。

        粒度对齐：angr CFG 与 Ghidra refined BB 的切分点不完全一致，除
        "angr BB 起始地址本身就是已覆盖 BB 起点"外，angr BB 起始地址若落在
        某个已覆盖静态 BB 的内部（通过 instruction_to_bb 归属），同样计为
        该 angr BB 已覆盖——避免因两边 BB 边界差一两条指令而系统性低估。
        """
        reachable = getattr(self, "angr_reachable_bb_set", None)
        if not reachable:
            return set()
        covered = set(covered_bbs)
        hits: Set[int] = set()
        for addr in reachable:
            if addr in covered:
                hits.add(addr)
                continue
            owner = self.instruction_to_bb.get(int(addr))
            if owner is not None and owner in covered:
                hits.add(int(addr))
        return hits
