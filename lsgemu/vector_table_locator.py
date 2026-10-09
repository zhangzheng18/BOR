#!/usr/bin/env python3
"""Cortex-M 向量表自动定位（纯函数模块，与仿真器解耦）。

背景：部分无人机固件 ELF 的第一个 PT_LOAD ``p_offset=0``，ELF 文件头（52
字节）随段映射进 flash，真实向量表位于文件偏移 0x4000（flash 0x08004000）。
只尝试"段 vaddr / base_addr"的传统启发式会在 0x08000000 读到 ELF 魔数
``0x464c457f`` 当作初始 SP，随后静默回退默认 SP，导致仿真从根上跑错。

本模块只依赖镜像字节与显式参数（不依赖 Uc 实例），对镜像内每个 4 字节
对齐偏移打分，输出按置信度排序的向量表候选列表：

- word[0]（初始 SP）：4 字节对齐且落在已知 Cortex-M RAM 窗口内；
- word[1]（Reset）：Thumb（bit0=1）且落在 [load_base, load_base+len) 内；
- word[2..16)（NMI/HardFault/.../SysTick）：越多"落在镜像内且 Thumb"
  （保留槽位为 0 视为一致）得分越高；全部为 0 或全指向同一地址属于弱
  信号，会拉低 distinct 填充项得分；
- ELF 头污染（``0x464c457f``、``0x00010101``）直接判负，不产出候选。

load_base 未知时按常见 Cortex-M flash 基址逐一假设，并用 Reset 指针的
扇区对齐值补充候选（覆盖 bootloader 布局，例如应用镜像基址 0x08004000），
取能产生最多一致判据的组合。随机数据仅靠 SP+Reset 巧合通过时得分约
0.60，远低于 :data:`CONFIDENT_MIN_SCORE`（0.75），可用 word[2..] 的
一致性拉开差距。
"""

from __future__ import annotations

import logging
import sys
from array import array
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 已知 Cortex-M RAM 窗口（SP 判据），参考各 MCU 手册：
#   0x20000000-0x20300000  STM32 F1/F4 主 SRAM 窗口
#   0x10000000-0x10040000  STM32 F4 CCM RAM
#   0x1FFF0000-0x20000000  系统/备份 SRAM（F4 Ethernet SRAM、L4 SRAM2 等）
#   0x24000000-0x24080000  STM32H7 AXI SRAM (D1)
#   0x30000000-0x30080000  STM32H7 SRAM1-3 (D2，含 AXI 重映射视角)
RAM_WINDOWS: Tuple[Tuple[int, int], ...] = (
    (0x20000000, 0x20300000),
    (0x10000000, 0x10040000),
    (0x1FFF0000, 0x20000000),
    (0x24000000, 0x24080000),
    (0x30000000, 0x30080000),
)

# load_base 未知时逐一尝试的常见 Cortex-M flash 基址：
#   0x08000000  STM32 / 多数 Cortex-M3/M4/M7
#   0x00000000  部分芯片（LPC、Kinetis 等低地址映射）
#   0x02000000  部分厂商双 bank / 远端 flash
#   0x00400000  部分 TI / 重映射布局
COMMON_FLASH_BASES: Tuple[int, ...] = (0x08000000, 0x00000000, 0x02000000, 0x00400000)

# 从 Reset 指针反推候选基址时使用的扇区对齐粒度（覆盖 bootloader 应用
# 布局：16KB/32KB/64KB 扇区边界，例如 0x08004000）。
_SECTOR_ALIGN_MASKS: Tuple[int, ...] = (0x3FFF, 0x7FFF, 0xFFFF)

# ELF 头特征 word：word[0]=0x464c457f（"\x7fELF" 小端魔数）、后续
# 0x00010101（EI_CLASS/EI_DATA/EI_VERSION）。这些值被当作 SP/Reset 是
# 本次事故的直接特征，直接判负。
_ELF_HEADER_MARKERS: frozenset = frozenset({0x464C457F, 0x00010101})

# 评分权重（合计 1.0）：
#   sp_ram_window        SP 落在已知 RAM 窗口（必要条件）
#   sp_8byte_aligned     SP 满足 AAPCS 8 字节对齐（常见但非强制）
#   reset_thumb_in_image Reset 为 Thumb 且落在镜像内（必要条件）
#   vector_consistency   word[2..16) 落在镜像内且 Thumb（保留槽位 0 视为一致）
#   vector_distinct      非零向量去重比例（全 0/全同址的弱信号降分）
#   vector_nonzero_fill  非零向量填充比例
_WEIGHT_SP_RAM_WINDOW = 0.30
_WEIGHT_SP_8BYTE_ALIGNED = 0.05
_WEIGHT_RESET_THUMB_IN_IMAGE = 0.25
_WEIGHT_VECTOR_CONSISTENCY = 0.20
_WEIGHT_VECTOR_DISTINCT = 0.10
_WEIGHT_VECTOR_NONZERO_FILL = 0.10

# 置信阈值：真实向量表（含 bootloader 弱表）通常 >= 0.85，随机数据仅靠
# SP+Reset 巧合通过约 0.60，阈值取 0.75 以隔离两类分布。
CONFIDENT_MIN_SCORE: float = 0.75

# 全量扫描的镜像大小上限；更大的镜像只扫描 [0, 0x20000) 与 64KB 边界
# 附近（bootloader 布局几乎总是落在这两类窗口内）。
_FULL_SCAN_MAX_IMAGE = 4 * 1024 * 1024
_LARGE_IMAGE_HEAD_WINDOW = 0x20000
_LARGE_IMAGE_BOUNDARY_RADIUS = 0x400

# 候选列表软上限，防御病态输入（例如大片重复数据）撑爆返回值。
_MAX_RETURNED_CANDIDATES = 256

# 评分使用 word[2..16)（NMI..SysTick 共 14 个系统异常槽位）。
_VECTOR_SCORE_FIRST = 2
_VECTOR_SCORE_LAST = 16


@dataclass(frozen=True)
class VectorTableCandidate:
    """一个向量表候选及其判定依据。

    Attributes:
        offset: 向量表起点在镜像内的偏移（BIN 字节偏移 = flash 地址 - load_base）。
        load_base: 建议加载基址（镜像首字节对应的总线地址）。
        initial_sp: word[0] 解析出的初始 SP。
        reset_pc: word[1] 解析出的 Reset 入口（含 Thumb bit）。
        score: 0~1 置信度得分。
        signals: 命中的判据标签。
    """

    offset: int
    load_base: int
    initial_sp: int
    reset_pc: int
    score: float
    signals: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def vector_table_address(self) -> int:
        """向量表在总线上的绝对地址。"""
        return (int(self.load_base) + int(self.offset)) & 0xFFFFFFFF

    @property
    def confident(self) -> bool:
        """是否达到置信阈值。"""
        return float(self.score) >= CONFIDENT_MIN_SCORE


def _in_ram_window(value: int) -> bool:
    """SP 是否落在任一已知 RAM 窗口内。"""
    return any(start <= value < end for start, end in RAM_WINDOWS)


def _iter_scan_offsets(image_len: int) -> Iterable[int]:
    """产出候选起点偏移：小镜像全扫，大镜像扫头部与 64KB 边界附近。

    终点保证 offset+4 处的 Reset word 仍可读（至少需要 SP+Reset 两项）。
    """
    scan_limit = image_len - 7  # 最后一个可判定 SP+Reset 的起点
    if image_len <= _FULL_SCAN_MAX_IMAGE:
        yield from range(0, max(scan_limit, 0), 4)
        return
    yielded = set()
    head_limit = min(scan_limit, _LARGE_IMAGE_HEAD_WINDOW)
    for offset in range(0, head_limit, 4):
        yielded.add(offset)
        yield offset
    boundary = _LARGE_IMAGE_HEAD_WINDOW
    while boundary < scan_limit:
        low = max(0, boundary - _LARGE_IMAGE_BOUNDARY_RADIUS) & ~3
        high = min(scan_limit, boundary + _LARGE_IMAGE_BOUNDARY_RADIUS)
        for offset in range(low, high, 4):
            if offset not in yielded:
                yielded.add(offset)
                yield offset
        boundary += 0x10000


def _decode_words(image: bytes, endian: str) -> array:
    """把镜像解码为 32 位 word 数组（截断尾部不足 4 字节的部分）。"""
    usable = len(image) & ~3
    words = array("I")
    words.frombytes(image[:usable])
    little_endian_host = sys.byteorder == "little"
    if (endian == "little") != little_endian_host:
        words.byteswap()
    return words


def _elf_header_contaminated(sp_word: int, reset_word: int) -> bool:
    """识别 ELF 头特征污染（例如 PT_LOAD 把文件头映射进 flash 的产物）。"""
    return sp_word in _ELF_HEADER_MARKERS or reset_word in _ELF_HEADER_MARKERS


def _self_consistent_prefilter(words: array, word_index: int, reset_target: int) -> bool:
    """基址无关的自一致性预筛：进一步压缩随机巧合幸存者。

    要求 word[2..16) 中至少一个非零 Thumb 值与 Reset 指针位于同一
    64KB 块（启动代码的异常向量通常与 Reset 相邻），或全部为 0（保留
    极简弱表的检测能力，交给评分阶段降权）。该条件与加载基址无关，
    可在逐基址评分之前执行。
    """
    block = reset_target & ~0xFFFF
    saw_nonzero = False
    for index in range(word_index + _VECTOR_SCORE_FIRST, word_index + _VECTOR_SCORE_LAST):
        if index >= len(words):
            return True  # 镜像截断，交由评分阶段处理
        value = words[index] & 0xFFFFFFFF
        if value == 0:
            continue
        saw_nonzero = True
        if value & 1 and ((value & ~1) & ~0xFFFF) == block:
            return True
    # 全零弱表（无任何非零向量）放行，由评分阶段降权。
    return not saw_nonzero


def _derived_load_bases(
    sp_survivors: Sequence[Tuple[int, int, int]],
) -> List[int]:
    """从双筛幸存者的 Reset 指针反推扇区对齐基址。

    幸存者已经过 SP+Reset+自一致性三重预筛，数量与真实向量表同量级，
    派生基址（16KB/32KB/64KB 对齐）不会随镜像内 RAM 常量数量爆炸。
    例如 reset=0x08004fb5 补充 0x08004000（bootloader 应用镜像基址）。
    """
    derived: List[int] = []
    seen = set(COMMON_FLASH_BASES)
    for _offset, _sp_word, reset_word in sp_survivors:
        if not reset_word:
            continue
        target = reset_word & ~1
        for mask in _SECTOR_ALIGN_MASKS:
            base = target & ~mask
            if 0 < base <= target and base not in seen:
                seen.add(base)
                derived.append(base)
    return derived


def _score_candidate(
    offset: int,
    load_base: int,
    words: array,
    image_len: int,
    max_entries: int,
) -> Optional[VectorTableCandidate]:
    """对一个 (offset, load_base) 组合评分；必要条件不满足返回 None。"""
    sp_word = words[offset // 4]
    reset_word = words[offset // 4 + 1]
    initial_sp = sp_word & 0xFFFFFFFF
    reset_pc = reset_word & 0xFFFFFFFF

    if _elf_header_contaminated(sp_word, reset_word):
        # ELF 头特征：直接判负，不产出候选（正是本次事故的失败模式）。
        logger.debug(
            "向量表扫描: 偏移 0x%x 命中 ELF 头特征 (sp=0x%08x reset=0x%08x)，判负",
            offset,
            initial_sp,
            reset_pc,
        )
        return None
    if initial_sp & 0x3 or not _in_ram_window(initial_sp):
        return None

    reset_target = reset_pc & ~1
    if not (reset_pc & 1) or not (load_base <= reset_target < load_base + image_len):
        return None

    signals: List[str] = ["sp_ram_window", "reset_thumb_in_image"]
    score = _WEIGHT_SP_RAM_WINDOW + _WEIGHT_RESET_THUMB_IN_IMAGE
    if initial_sp & 0x7 == 0:
        signals.append("sp_8byte_aligned")
        score += _WEIGHT_SP_8BYTE_ALIGNED

    vector_last = min(_VECTOR_SCORE_LAST, max(max_entries, _VECTOR_SCORE_FIRST + 1))
    consistent = 0.0
    nonzero = 0
    distinct_values = set()
    available = 0
    for index in range(offset // 4 + _VECTOR_SCORE_FIRST, offset // 4 + vector_last):
        if index >= len(words):
            break
        available += 1
        value = words[index] & 0xFFFFFFFF
        if value == 0:
            # 保留槽位为 0 视为半权重一致（例如 Cortex-M3 的 UsageFault
            # 保留区）；全零表因此落到置信线下，避免数据区巧合冒充向量表。
            consistent += 0.5
            continue
        target = value & ~1
        if value & 1 and load_base <= target < load_base + image_len:
            consistent += 1.0
            nonzero += 1
            distinct_values.add(value)

    if available:
        consistency_ratio = consistent / available
        nonzero_ratio = nonzero / available
        distinct_ratio = (len(distinct_values) / nonzero) if nonzero else 0.0
        score += (
            _WEIGHT_VECTOR_CONSISTENCY * consistency_ratio
            + _WEIGHT_VECTOR_NONZERO_FILL * nonzero_ratio
            + _WEIGHT_VECTOR_DISTINCT * distinct_ratio
        )
        if consistency_ratio >= 0.99:
            signals.append("vectors_consistent")
        if nonzero and distinct_ratio >= 0.5:
            signals.append("vectors_distinct")
        if nonzero == 0:
            signals.append("vectors_all_zero_weak")

    score = min(1.0, score)
    return VectorTableCandidate(
        offset=offset,
        load_base=load_base,
        initial_sp=initial_sp,
        reset_pc=reset_pc,
        score=round(score, 4),
        signals=tuple(signals),
    )


def scan_vector_table(
    image: bytes,
    *,
    arch: str = "arm",
    endian: str = "little",
    load_base_hint: Optional[int] = None,
    max_entries: int = 48,
) -> List[VectorTableCandidate]:
    """扫描镜像，返回按置信度降序排列的向量表候选。

    Args:
        image: 镜像字节（BIN 内容，或 ELF 按 PT_LOAD 展开后的内存镜像）。
        arch: 目标架构，当前仅支持 ``arm``（Cortex-M）。
        endian: word 字节序，``little`` 或 ``big``。
        load_base_hint: 已知加载基址时只验证该基址；None 时按常见
            Cortex-M flash 基址逐一假设并取一致判据最多的组合。
        max_entries: 向量表一致性检查的最大 word 数上限（评分覆盖
            word[2..min(16, max_entries))），防御截断镜像。

    Returns:
        候选列表，按 (score 降序, reset 偏移升序, 镜像偏移升序) 排序；
        同分时偏好 Reset 距表更近的紧凑布局（例如区分完整 flash 镜像
        0x08000000 与应用镜像 0x08004000）。无候选时返回空列表。
    """
    if str(arch).lower() != "arm":
        raise ValueError(f"vector_table_locator 目前仅支持 arm/Cortex-M, 收到 {arch!r}")
    if endian not in {"little", "big"}:
        raise ValueError(f"endian 必须是 little/big, 收到 {endian!r}")
    if len(image) < 8:
        return []

    words = _decode_words(image, endian)
    if len(words) < 2:
        return []

    # 第一遍预筛与 load_base 无关：SP 对齐 + RAM 窗口（随机数据在此
    # 以 >99.9% 的比例淘汰）+ Reset 为 Thumb + 向量自一致性，幸存者
    # 数量与真实向量表同量级，才进入逐基址评分。
    sp_survivors: List[Tuple[int, int, int]] = []
    for offset in _iter_scan_offsets(len(image)):
        word_index = offset // 4
        sp_word = words[word_index] & 0xFFFFFFFF
        if sp_word & 0x3 or not _in_ram_window(sp_word):
            continue
        reset_word = words[word_index + 1] & 0xFFFFFFFF
        if _elf_header_contaminated(sp_word, reset_word):
            continue
        if not (reset_word & 1):
            continue
        if not _self_consistent_prefilter(words, word_index, reset_word & ~1):
            continue
        sp_survivors.append((offset, sp_word, reset_word))

    if not sp_survivors:
        return []

    image_len = len(image)

    def _evaluate(load_bases: Sequence[int]) -> List[VectorTableCandidate]:
        scored: List[VectorTableCandidate] = []
        for load_base in load_bases:
            for offset, _sp_word, _reset_word in sp_survivors:
                candidate = _score_candidate(offset, load_base, words, image_len, max_entries)
                if candidate is not None:
                    scored.append(candidate)
        return scored

    if load_base_hint is not None:
        candidates = _evaluate([int(load_base_hint) & 0xFFFFFFFF])
    else:
        # 常见 flash 基址 + 幸存 Reset 指针的扇区对齐基址（覆盖 bootloader
        # 应用镜像布局，例如小应用镜像在 0x08000000 假设下 Reset 越界、
        # 只有 0x08004000 假设成立）。
        load_bases = list(COMMON_FLASH_BASES) + _derived_load_bases(sp_survivors)
        candidates = _evaluate(load_bases)

    candidates.sort(
        key=lambda item: (
            -item.score,
            (item.reset_pc & ~1) - item.load_base,
            item.offset,
            -item.load_base,
        )
    )
    return candidates[:_MAX_RETURNED_CANDIDATES]


__all__ = [
    "COMMON_FLASH_BASES",
    "CONFIDENT_MIN_SCORE",
    "RAM_WINDOWS",
    "VectorTableCandidate",
    "scan_vector_table",
]
