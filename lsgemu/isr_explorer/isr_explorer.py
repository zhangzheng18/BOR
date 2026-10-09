#!/usr/bin/env python3
"""
ISR Explorer - 中断服务程序探索器（改进版）

核心改进:
1. 模拟真实的 CPU 压栈过程
2. 设置常见中断标志位
3. 支持 MMIO 处理器集成
"""

import logging
import os
from collections import Counter
from typing import Callable, Dict, Optional, Set, List, Tuple
from unicorn import *
from unicorn.arm_const import *

from ..causal_context import CausalExecutionContext
from ..hook_lifecycle import (
    managed_emu_start,
    managed_hook_add,
    managed_hook_del,
    managed_mem_map,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ARMv7-M exception entry / exit (architectural, zero free parameters)
# ---------------------------------------------------------------------------
#
# EXC_RETURN encoding (ARMv7-M ARM, B1.5.8):
#   bit0     = 1                 (always set; a zero value would be a valid PC)
#   bit1     = 0                 (reserved, MBZ)
#   bit2     = return stack: 1 = PSP, 0 = MSP
#   bit3     = return mode:  1 = Thread, 0 = Handler
#   bit4     = frame type:   0 = extended (FPU) frame, 1 = basic frame
#   bits31:5 = 1                 (0xFFFFFF.. prefix)
#
# ``0xFFFFFFED`` (return to Thread/PSP, extended frame) is what this firmware
# actually produces: ``__port_irq_epilogue`` writes ``subs r4, #104`` and
# ``str FPDSCR, [r4, #0x60]`` — 104 = sizeof(struct port_extctx).
EXC_RETURN_HANDLER_MSP = 0xFFFFFFF1
EXC_RETURN_HANDLER_PSP = 0xFFFFFFF5
EXC_RETURN_THREAD_MSP = 0xFFFFFFF9
EXC_RETURN_THREAD_PSP = 0xFFFFFFFD
EXC_RETURN_THREAD_MSP_FPU = 0xFFFFFFE9
EXC_RETURN_THREAD_PSP_FPU = 0xFFFFFFED

EXC_RETURN_VALUES = (
    EXC_RETURN_HANDLER_MSP,
    EXC_RETURN_HANDLER_PSP,
    EXC_RETURN_THREAD_MSP,
    EXC_RETURN_THREAD_PSP,
    EXC_RETURN_THREAD_MSP_FPU,
    EXC_RETURN_THREAD_PSP_FPU,
)

FRAME_FORMAT_BASIC32 = "basic32"
FRAME_FORMAT_EXT104 = "ext104"
FRAME_SIZE_BASIC = 32
FRAME_SIZE_EXTENDED = 104

# Byte offsets inside a Cortex-M exception frame (R0 first, low address).
FRAME_OFFSET_PC = 0x18
FRAME_OFFSET_XPSR = 0x1C
FRAME_OFFSET_S0 = 0x20
FRAME_OFFSET_FPSCR = 0x60
FRAME_OFFSET_EXT_PAD = 0x64

# APSR flag bits inside xPSR (N,Z,C,V,Q).
XPSR_FLAG_MASK = 0xF8000000
XPSR_THUMB_BIT = 0x01000000
XPSR_IPSR_MASK = 0x000001FF


def legacy_isr_frame_enabled() -> bool:
    """r40 P3/C7：LSGEMU_ISR_LEGACY_FRAME（缺省 1 = 旧语义）。

    ``=0`` 时 ``_setup_isr_context`` 改走共享构造器
    （``_setup_isr_context_shared``）。A/B 一致后才删旧体（k.4 P1-2）。
    """
    return os.environ.get("LSGEMU_ISR_LEGACY_FRAME", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def exception_frame_size(exc_return: int) -> int:
    """Frame size in bytes, from EXC_RETURN bit4 (0 = extended 104B, 1 = basic 32B)."""
    return FRAME_SIZE_BASIC if (int(exc_return) & 0x10) else FRAME_SIZE_EXTENDED


def exception_is_extended_frame(exc_return: int) -> bool:
    return (int(exc_return) & 0x10) == 0


def exception_uses_psp(exc_return: int) -> bool:
    """True when the EXC_RETURN restores Thread mode on the process stack."""
    return bool(int(exc_return) & 0x04)


def exception_returns_to_thread(exc_return: int) -> bool:
    """True when the EXC_RETURN leaves Handler mode (bit3)."""
    return bool(int(exc_return) & 0x08)


def _frame_format_to_exc_return(frame_format: str) -> int:
    if frame_format == FRAME_FORMAT_EXT104:
        return EXC_RETURN_THREAD_PSP_FPU
    if frame_format == FRAME_FORMAT_BASIC32:
        return EXC_RETURN_THREAD_PSP
    raise ValueError(f"unknown frame_format: {frame_format!r}")


def _stack_bank_register(exc_return: int) -> int:
    return UC_ARM_REG_PSP if exception_uses_psp(exc_return) else UC_ARM_REG_MSP


def _read_u32(uc, address: int) -> int:
    return int.from_bytes(uc.mem_read(int(address) & 0xFFFFFFFF, 4), "little")


def _write_u32(uc, address: int, value: int) -> None:
    uc.mem_write(
        int(address) & 0xFFFFFFFF, (int(value) & 0xFFFFFFFF).to_bytes(4, "little")
    )


def build_exception_entry(
    uc,
    *,
    irq: int,
    vector: int,
    at_pc: Optional[int] = None,
    frame_format: str = FRAME_FORMAT_EXT104,
    exc_return: Optional[int] = None,
    stack_pointer: Optional[int] = None,
    stack_fixer: Optional[Callable[[int], Tuple[int, str]]] = None,
    enter_handler_mode: bool = True,
) -> Dict[str, object]:
    """Synthesize the architectural effect of *taking* an exception.

    Unicorn never delivers an NVIC interrupt, so the delivery has to be written
    by hand.  This function is the faithful counterpart of the hardware push:

    * the frame lands on the stack bank selected by ``EXC_RETURN.bit2``
      (PSP for Thread/PSP, MSP otherwise) and its base is ``(SP - size) & ~7``;
    * the eight core words are ``R0, R1, R2, R3, R12, LR, PC, xPSR``;
    * for the extended (104B) format S0-S15 and FPSCR are also stacked;
    * ``LR`` is set to ``EXC_RETURN``, ``IPSR`` to the IRQ number, and the CPU
      enters Handler mode.

    ``frame_format`` selects how many bytes are *written*; ``exc_return`` selects
    what the *unstack* will consume.  They are kept independent so a deliberate
    mismatch can be exercised as a counterexample arm.

    Returns an audit dict; see ``entry`` key fields used by the replay evidence.
    """
    if frame_format not in (FRAME_FORMAT_BASIC32, FRAME_FORMAT_EXT104):
        raise ValueError(f"unknown frame_format: {frame_format!r}")
    if exc_return is None:
        exc_return = _frame_format_to_exc_return(frame_format)
    exc_return = int(exc_return) & 0xFFFFFFFF

    write_size = (
        FRAME_SIZE_EXTENDED if frame_format == FRAME_FORMAT_EXT104 else FRAME_SIZE_BASIC
    )

    stack_source = "explicit"
    if stack_pointer is None:
        stack_pointer = int(uc.reg_read(_stack_bank_register(exc_return))) & 0xFFFFFFFF
        stack_source = "bank"
    if stack_fixer is not None:
        # r40 P3/C7（Side B 缺陷二）：修复窗按本次实写的帧长开（旧行为恒按
        # 32B —— 104B 扩展帧会写出修复窗 72B）。fixer 返回的是帧基址本身。
        fixed_frame_start, fix_source = stack_fixer(
            int(stack_pointer) & 0xFFFFFFFF, write_size
        )
        stack_pointer = (int(fixed_frame_start) + write_size) & 0xFFFFFFFF
        stack_source = str(fix_source)

    interrupted_pc = (
        int(at_pc) if at_pc is not None else int(uc.reg_read(UC_ARM_REG_PC))
    )
    interrupted_xpsr = int(uc.reg_read(UC_ARM_REG_CPSR)) & 0xFFFFFFFF
    # The interrupted context was Thread mode: its xPSR had IPSR == 0 and T == 1.
    stacked_xpsr = (interrupted_xpsr & ~XPSR_IPSR_MASK) | XPSR_THUMB_BIT

    frame_base = (int(stack_pointer) - write_size) & ~7
    _write_u32(uc, frame_base + 0x00, int(uc.reg_read(UC_ARM_REG_R0)))
    _write_u32(uc, frame_base + 0x04, int(uc.reg_read(UC_ARM_REG_R1)))
    _write_u32(uc, frame_base + 0x08, int(uc.reg_read(UC_ARM_REG_R2)))
    _write_u32(uc, frame_base + 0x0C, int(uc.reg_read(UC_ARM_REG_R3)))
    _write_u32(uc, frame_base + 0x10, int(uc.reg_read(UC_ARM_REG_R12)))
    _write_u32(uc, frame_base + 0x14, int(uc.reg_read(UC_ARM_REG_LR)))
    _write_u32(uc, frame_base + FRAME_OFFSET_PC, (interrupted_pc & ~1) | 1)
    _write_u32(uc, frame_base + FRAME_OFFSET_XPSR, stacked_xpsr)

    if write_size == FRAME_SIZE_EXTENDED:
        for index in range(16):
            _write_u32(
                uc,
                frame_base + FRAME_OFFSET_S0 + 4 * index,
                int(uc.reg_read(UC_ARM_REG_S0 + index)),
            )
        _write_u32(uc, frame_base + FRAME_OFFSET_FPSCR, int(uc.reg_read(UC_ARM_REG_FPSCR)))
        _write_u32(uc, frame_base + FRAME_OFFSET_EXT_PAD, 0)

    # The exception pushes on its own bank; Thread mode keeps the PSP, Handler
    # mode runs on the MSP.  Writing the bank here matches "the pushed SP".
    uc.reg_write(_stack_bank_register(exc_return), frame_base)
    uc.reg_write(UC_ARM_REG_LR, exc_return)
    if enter_handler_mode:
        current_cpsr = int(uc.reg_read(UC_ARM_REG_CPSR))
        uc.reg_write(UC_ARM_REG_CPSR, (current_cpsr & ~0x1F) | 0x13)
    uc.reg_write(UC_ARM_REG_IPSR, int(irq) & XPSR_IPSR_MASK)
    uc.reg_write(UC_ARM_REG_PC, (int(vector) & ~1) | 1)

    return {
        "irq": int(irq),
        "vector": f"0x{int(vector) & 0xFFFFFFFF:08x}",
        "frame_format": frame_format,
        "frame_size": write_size,
        "exc_return": f"0x{exc_return:08x}",
        "exc_return_frame_size": exception_frame_size(exc_return),
        "frame_consistent": exception_frame_size(exc_return) == write_size,
        "frame_base": f"0x{frame_base & 0xFFFFFFFF:08x}",
        "stack_pointer_before": f"0x{int(stack_pointer) & 0xFFFFFFFF:08x}",
        "stack_source": stack_source,
        "interrupted_pc": f"0x{interrupted_pc & 0xFFFFFFFF:08x}",
        "stacked_xpsr": f"0x{stacked_xpsr & 0xFFFFFFFF:08x}",
    }


def _profile_range_constraints(constraints) -> Dict[str, int]:
    """Filter a constraint table down to the interrupt-profile ranges."""
    if not isinstance(constraints, dict):
        return {}
    result: Dict[str, int] = {}
    for key, value in constraints.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        pc, address = key
        try:
            pc = int(pc)
            address = int(address)
        except (TypeError, ValueError):
            continue
        if (
            0x40000C00 <= address < 0x40001000
            or address == 0xE000ED04
            or 0x50000000 <= address < 0x50040000
        ):
            result[f"0x{pc:08x}:0x{address:08x}"] = int(value)
    return result


def unstack_exception(
    uc,
    *,
    exc_return: Optional[int] = None,
    frame_base: Optional[int] = None,
    restore_flags: bool = True,
) -> Dict[str, object]:
    """Drive the architectural tail of ``do_v7m_exception_exit`` by hand.

    Unicorn raises ``EXCP_EXCEPTION_EXIT`` (``UC_HOOK_INTR`` intno=8) when a
    ``BX``/``POP {pc}`` targets an EXC_RETURN value but does *not* pop the frame,
    so the replay path has to do it.

    Two hard-won rules are encoded here:

    1. The frame base is the **current PSP/MSP**, never a remembered address.
       ``__port_irq_epilogue`` builds a second 104B frame below the PSP and
       ``msr psp`` moves the bank, so the frame actually being returned from is
       the one the PSP points at *now*.
    2. The frame type comes from the **PC**, never from ``LR``.  Unicorn rewrites
       the PC to ``EXC_RETURN & ~1`` when the exception exit is raised, while the
       LR at that moment may have been clobbered by an intervening ``bl``
       (``chSchIsPreemptionRequired`` does exactly that at ``0x0812EDC8``).

    ``IPSR`` is cleared *before* the banked SP is written: leaving Handler mode
    is what makes the SP bank switch observable, so the order is load-bearing.
    """
    if exc_return is None:
        exc_return = (int(uc.reg_read(UC_ARM_REG_PC)) | 1) & 0xFFFFFFFF
    exc_return = int(exc_return) & 0xFFFFFFFF
    size = exception_frame_size(exc_return)
    bank = _stack_bank_register(exc_return)
    if frame_base is None:
        frame_base = int(uc.reg_read(bank)) & 0xFFFFFFFF
    frame_base = int(frame_base) & 0xFFFFFFFF

    core = [_read_u32(uc, frame_base + 4 * index) for index in range(8)]
    stacked_pc = core[6]
    stacked_xpsr = core[7]

    uc.reg_write(UC_ARM_REG_R0, core[0])
    uc.reg_write(UC_ARM_REG_R1, core[1])
    uc.reg_write(UC_ARM_REG_R2, core[2])
    uc.reg_write(UC_ARM_REG_R3, core[3])
    uc.reg_write(UC_ARM_REG_R12, core[4])
    uc.reg_write(UC_ARM_REG_LR, core[5])

    fpu_restored = 0
    if size == FRAME_SIZE_EXTENDED:
        for index in range(16):
            uc.reg_write(
                UC_ARM_REG_S0 + index, _read_u32(uc, frame_base + FRAME_OFFSET_S0 + 4 * index)
            )
            fpu_restored += 1
        uc.reg_write(UC_ARM_REG_FPSCR, _read_u32(uc, frame_base + FRAME_OFFSET_FPSCR))

    flags_restored = False
    if restore_flags:
        try:
            uc.reg_write(UC_ARM_REG_APSR, stacked_xpsr & XPSR_FLAG_MASK)
            flags_restored = True
        except Exception:
            flags_restored = False

    saved_ipsr = 0
    try:
        saved_ipsr = int(uc.reg_read(UC_ARM_REG_IPSR)) & XPSR_IPSR_MASK
    except Exception:
        saved_ipsr = 0

    post_sp = (frame_base + size) & 0xFFFFFFFF
    # Leave Handler mode first, then publish the restored stack pointer.
    uc.reg_write(UC_ARM_REG_IPSR, 0)
    uc.reg_write(bank, post_sp)
    uc.reg_write(UC_ARM_REG_SP, post_sp)
    uc.reg_write(UC_ARM_REG_PC, (stacked_pc & ~1) | 1)

    return {
        "exc_return": f"0x{exc_return:08x}",
        "frame_size": size,
        "extended_frame": size == FRAME_SIZE_EXTENDED,
        "ipsp": f"0x{frame_base:08x}",
        "base": f"0x{frame_base:08x}",
        "base_value": frame_base,
        "stacked_pc": f"0x{stacked_pc & 0xFFFFFFFF:08x}",
        "stacked_xpsr": f"0x{stacked_xpsr & 0xFFFFFFFF:08x}",
        "post_unstack_sp": f"0x{post_sp:08x}",
        "post_unstack_sp_value": post_sp,
        "core": [f"0x{value:08x}" for value in core],
        "fpu_restored": fpu_restored,
        "flags_restored": flags_restored,
        "restored_to_thread": exception_returns_to_thread(exc_return),
        "restored_psp": exception_uses_psp(exc_return),
        "saved_ipsr": f"0x{saved_ipsr:08x}",
    }


# ---------------------------------------------------------------------------
# r40 P3/C7：异常栈窗修复的唯一实现（explorer 冷启动/上下文注入与投递/重放
# 两侧共用）。原先只有 ``ISRExplorer._ensure_exception_stack_frame`` 一份，
# 投递路径（``IrqDeliveryController._deliver``）完全不带栈修复（Side B
# 缺陷一），且 ``build_exception_entry`` 的 stack_fixer 合约按 32B 基本帧
# 假设（Side B 缺陷二：104B 扩展帧会写出修复窗 72B 之外）。
# ---------------------------------------------------------------------------

def _uc_memory_regions(uc) -> List[Tuple[int, int, int]]:
    try:
        return [
            (int(start), int(end), int(perms))
            for start, end, perms in uc.mem_regions()
        ]
    except Exception:
        return []


def _uc_range_in_region(
    uc, address: int, size: int, *, require_write: bool = False
) -> bool:
    if size <= 0:
        return True
    start = int(address)
    end = start + int(size) - 1
    if start < 0 or end > 0xFFFFFFFF or end < start:
        return False
    for region_start, region_end, perms in _uc_memory_regions(uc):
        if region_start <= start and end <= region_end:
            return not require_write or bool(perms & UC_PROT_WRITE)
    return False


def _uc_is_writable_range(uc, address: int, size: int) -> bool:
    if _uc_range_in_region(uc, address, size, require_write=True):
        return True
    try:
        original = bytes(uc.mem_read(address, size))
        uc.mem_write(address, original)
        return True
    except Exception:
        return False


def _uc_is_plausible_stack_pointer(uc, sp: int) -> bool:
    sp = int(sp) & 0xFFFFFFFF
    if sp in {0, 0xFFFFFFFF} or sp < 0x80:
        return False
    if _uc_range_in_region(uc, sp - 32, 32, require_write=True):
        return True
    # Cortex-M stacks normally live in SRAM/TCM/CCM-like regions. Keep this
    # broad enough for non-STM32 MCUs, while rejecting flash/MMIO/system space.
    return 0x10000080 <= sp <= 0x3FFFFFFF or 0x60000080 <= sp <= 0x9FFFFFFF


def _uc_map_writable_pages(uc, address: int, size: int) -> bool:
    if size <= 0:
        return True
    start = int(address)
    end = start + int(size)
    if start < 0 or end <= start or end > 0x100000000:
        return False
    page = start & ~0xFFF
    last = (end + 0xFFF) & ~0xFFF
    while page < last:
        if not _uc_range_in_region(uc, page, 1, require_write=False):
            try:
                managed_mem_map(uc, page, 0x1000, UC_PROT_ALL)
            except Exception:
                if not _uc_range_in_region(uc, page, 1, require_write=True):
                    return False
        elif not _uc_range_in_region(uc, page, 1, require_write=True):
            return False
        page += 0x1000
    return _uc_is_writable_range(uc, address, size)


def _uc_read_vector_initial_sp(uc, vtor: int) -> Optional[int]:
    try:
        return int.from_bytes(uc.mem_read(int(vtor), 4), "little") & 0xFFFFFFFF
    except Exception:
        return None


def _uc_mapped_writable_stack_top(uc) -> Optional[int]:
    candidates = []
    for start, end, perms in _uc_memory_regions(uc):
        if not (perms & UC_PROT_WRITE):
            continue
        # Prefer RAM/TCM-like regions; avoid MMIO/system mappings created
        # for status registers during ISR exploration.
        if not (
            0x10000000 <= start <= 0x3FFFFFFF
            or 0x60000000 <= start <= 0x9FFFFFFF
        ):
            continue
        if end - start + 1 < 0x80:
            continue
        candidates.append((end + 1) & ~7)
    return max(candidates) if candidates else None


def ensure_writable_exception_frame(
    uc,
    requested_sp: int,
    frame_size: int = FRAME_SIZE_BASIC,
    *,
    vtor: int = 0x08000000,
    repair_stats: Optional[Counter] = None,
    invalid_samples: Optional[List[Dict[str, object]]] = None,
) -> Tuple[int, str]:
    """Return a writable Cortex-M exception-frame base for ISR injection.

    r40 P3/C7 从 ``ISRExplorer._ensure_exception_stack_frame`` 下沉的唯一
    实现：冷启动注入（explorer）与模型驱动投递/重放（controller）共用同
    一条修复级联（native SP → 当前 SP 可映射 → 向量表初值 SP → 已映射
    可写 RAM 顶 → 缺省 SRAM 栈）。``frame_size`` 按本次入口要写的帧长
    （32/104B）开窗——旧行为恒按 32B 开窗，104B 扩展帧会写出修复窗。
    """
    try:
        window = int(frame_size)
    except (TypeError, ValueError):
        window = FRAME_SIZE_BASIC
    if window <= 0:
        window = FRAME_SIZE_BASIC

    original_sp = int(requested_sp) & 0xFFFFFFFF
    aligned_sp = original_sp & ~7
    frame_start = aligned_sp - window
    if frame_start >= 0 and _uc_is_writable_range(uc, frame_start, window):
        if repair_stats is not None:
            repair_stats["stack_frame_native"] += 1
        return frame_start, "native_sp"

    candidates: List[Tuple[str, Optional[int]]] = [
        (
            "current_sp_mapped",
            aligned_sp if _uc_is_plausible_stack_pointer(uc, aligned_sp) else None,
        ),
        ("vector_initial_sp", _uc_read_vector_initial_sp(uc, vtor)),
        ("mapped_writable_ram", _uc_mapped_writable_stack_top(uc)),
        ("default_sram_stack", 0x20010000),
    ]
    for source, candidate in candidates:
        if candidate is None:
            continue
        sp = int(candidate) & 0xFFFFFFFF
        sp &= ~7
        frame_start = sp - window
        if frame_start < 0:
            continue
        if _uc_map_writable_pages(uc, frame_start, window):
            if repair_stats is not None:
                repair_stats["stack_frame_repaired"] += 1
                repair_stats[f"stack_source_{source}"] += 1
            if invalid_samples is not None:
                invalid_samples.append({
                    "original_sp": f"0x{original_sp:08x}",
                    "selected_sp": f"0x{sp:08x}",
                    "source": source,
                })
                del invalid_samples[:-16]
            return frame_start, source

    if repair_stats is not None:
        repair_stats["stack_frame_unrepairable"] += 1
    if invalid_samples is not None:
        invalid_samples.append({
            "original_sp": f"0x{original_sp:08x}",
            "selected_sp": f"0x{aligned_sp:08x}",
            "source": "unrepairable",
        })
        del invalid_samples[:-16]
    return frame_start, "unrepairable"


class ISRExplorer:
    """ISR 探索器（改进版）"""

    def __init__(self, uc, static_bbs, vtor=0x08000000, mmio_handler=None,
                 instruction_to_bb: Optional[Dict[int, int]] = None,
                 broad_interrupt_flags: bool = False,
                 code_ranges: Optional[List[Tuple[int, int]]] = None,
                 start_validator: Optional[Callable[[int], object]] = None,
                 causal_context: Optional[CausalExecutionContext] = None):
        """
        初始化

        Args:
            uc: Unicorn 实例
            static_bbs: 静态基本块
            vtor: 向量表基址
            mmio_handler: MMIO 处理器（可选）
            instruction_to_bb: 指令地址到基本块的映射（可选）
            start_validator: Unicorn 启动前验证回调（可选）。返回 bool，或
                以 bool 为首项的 tuple；省略时保持原有启动行为。
        """
        self.uc = uc
        self.static_bbs = static_bbs
        self.vtor = vtor
        self.mmio_handler = mmio_handler
        self.instruction_to_bb = instruction_to_bb or self._build_instruction_map()
        self.broad_interrupt_flags = broad_interrupt_flags
        self.code_ranges = code_ranges or [(0x08000000, 0x08100000), (0x00000000, 0x00100000)]
        self.start_validator = start_validator
        self.causal_context = causal_context
        self._active_irq: Optional[int] = None
        self.execution_start_stats = Counter()

        # ISR 信息
        self.isr_addresses = {}  # {irq_num: isr_addr}
        self.isr_coverage = {}   # {irq_num: set(covered_bbs)}

        # 常见中断号对应的 MMIO 地址（STM32 为例）
        self.common_irq_mmio = {
            -1: 0x40021000,   # SysTick: STK_CTRL
            37: 0x40013800,   # USART1: USART_SR
            38: 0x40004400,   # USART2: USART_SR
            39: 0x40004800,   # USART3: USART_SR
            28: 0x40000010,   # TIM2: TIM_SR
            29: 0x40000410,   # TIM3: TIM_SR
            30: 0x40000810,   # TIM4: TIM_SR
        }

        self.interrupt_flag_registers = {
            # NVIC enable/pending registers. Manually invoking the ISR does not
            # require these, but some firmware checks them before dispatching.
            0xE000E100: 0xFFFFFFFF,
            0xE000E104: 0xFFFFFFFF,
            0xE000E200: 0xFFFFFFFF,
            0xE000E204: 0xFFFFFFFF,
            # STM32F1-style peripheral interrupt/status flags.
            0x40021008: 0xFFFFFFFF,  # RCC_CIR
            0x40010414: 0xFFFFFFFF,  # EXTI_PR
            0x40020000: 0xFFFFFFFF,  # DMA1_ISR
            0x40020400: 0xFFFFFFFF,  # DMA2_ISR
            0x40000010: 0x0000001F,  # TIM2_SR
            0x40000410: 0x0000001F,  # TIM3_SR
            0x40000810: 0x0000001F,  # TIM4_SR
            0x40000C10: 0x0000001F,  # TIM5_SR
            0x40001010: 0x0000001F,  # TIM6_SR
            0x40001410: 0x0000001F,  # TIM7_SR
            0x40013800: 0x000000E0,  # USART1_SR
            0x40004400: 0x000000E0,  # USART2_SR
            0x40004800: 0x000000E0,  # USART3_SR
            0x40013008: 0x000000FF,  # SPI1_SR
            0x40003808: 0x000000FF,  # SPI2_SR
            0x40005414: 0x0000FFFF,  # I2C1_SR1
            0x40005814: 0x0000FFFF,  # I2C2_SR1
            0x40012400: 0xFFFFFFFF,  # ADC_SR
        }
        self.irq_event_history: List[Dict[str, object]] = []
        self.irq_register_observations: Dict[int, int] = {}
        self.irq_event_counts = Counter()
        self.stack_repair_stats = Counter()
        self.invalid_stack_samples: List[Dict[str, object]] = []
        self._stack_warning_emitted = False
        self.nvic_enabled_irqs: Set[int] = set()
        self.nvic_pending_irqs: Set[int] = set()
        self.status_register_hits: Counter[int] = Counter()

        # 解析向量表
        self._parse_vector_table()
        self.observe_mmio_state(getattr(mmio_handler, "mmio_state", {}) if mmio_handler is not None else {})

    def observe_mmio_state(self, mmio_state: Optional[Dict[int, int]]) -> None:
        """Learn IRQ enable/pending/status hints from an existing MMIO state map."""
        for address, value in (mmio_state or {}).items():
            try:
                self._record_irq_event(int(address), int(value))
            except Exception:
                continue

    def _build_instruction_map(self) -> Dict[int, int]:
        """构建指令地址到基本块的映射"""
        mapping: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                mapping[insn['address']] = bb_addr
        return mapping

    def _parse_vector_table(self):
        """解析向量表，提取 ISR 地址"""
        logger.info(f"[ISR] 解析向量表 @ {hex(self.vtor)}")

        try:
            # Cortex-M vector indices:
            #   0 = initial SP, 1 = reset, 2..15 = system exceptions,
            #   16..255 = external IRQs. CMSIS IRQn values are index - 16,
            # so SysTick is -1 and the first external IRQ is 0.
            #
            # Earlier versions only parsed external IRQs. Static reachability
            # already counted system-exception handlers, which left NMI,
            # HardFault, SVC, PendSV and SysTick as permanent vector-only gaps.
            for vector_index in range(2, 16 + 240):
                irq_num = vector_index - 16
                offset = vector_index * 4

                try:
                    isr_addr_bytes = self.uc.mem_read(self.vtor + offset, 4)
                    isr_addr = int.from_bytes(isr_addr_bytes, 'little') & ~1

                    if self._is_code_address(isr_addr):
                        self.isr_addresses[irq_num] = isr_addr

                except Exception:
                    pass

            logger.info(f"[ISR] 找到 {len(self.isr_addresses)} 个 ISR")

        except Exception as e:
            logger.error(f"[ISR] 解析向量表失败: {e}")

    def _is_code_address(self, address: int) -> bool:
        target = int(address) & ~1
        # A zero vector is an uninstalled handler, not executable code. This
        # matters for raw images whose default code range includes address 0.
        if target == 0:
            return False
        if target in self.static_bbs or target in self.instruction_to_bb:
            return True
        return any(int(start) <= target < int(end) for start, end in self.code_ranges)

    def _memory_regions(self) -> List[Tuple[int, int, int]]:
        # r40 P3/C7：栈窗修复已下沉到模块级唯一实现（explorer 与投递/重放
        # 共用）；这些薄委托保留既有方法面，行为逐字不变。
        return _uc_memory_regions(self.uc)

    def _range_in_region(self, address: int, size: int, *, require_write: bool = False) -> bool:
        return _uc_range_in_region(self.uc, address, size, require_write=require_write)

    def _is_writable_range(self, address: int, size: int) -> bool:
        return _uc_is_writable_range(self.uc, address, size)

    def _is_plausible_stack_pointer(self, sp: int) -> bool:
        return _uc_is_plausible_stack_pointer(self.uc, sp)

    def _map_writable_pages(self, address: int, size: int) -> bool:
        return _uc_map_writable_pages(self.uc, address, size)

    def _read_vector_initial_sp(self) -> Optional[int]:
        return _uc_read_vector_initial_sp(self.uc, self.vtor)

    def _mapped_writable_stack_top(self) -> Optional[int]:
        return _uc_mapped_writable_stack_top(self.uc)

    def _ensure_exception_stack_frame(
        self, requested_sp: int, frame_size: int = FRAME_SIZE_BASIC
    ) -> Tuple[int, str]:
        """Return a writable Cortex-M exception-frame base for ISR injection.

        r40 P3/C7：委托到模块级 ``ensure_writable_exception_frame``（唯一
        实现）。``frame_size`` 由 ``build_exception_entry`` 的 stack_fixer
        合约传入（32/104B 按本次入口实写的帧长开窗）；缺省 32B = 旧语义。
        """
        return ensure_writable_exception_frame(
            self.uc,
            requested_sp,
            frame_size,
            vtor=self.vtor,
            repair_stats=self.stack_repair_stats,
            invalid_samples=self.invalid_stack_samples,
        )

    def explore_all_isrs(self, max_instructions=10000) -> Dict[int, Set[int]]:
        """探索所有 ISR"""
        logger.info(f"\n[ISR] 开始探索 {len(self.isr_addresses)} 个 ISR")

        for irq_num, isr_addr in sorted(self.isr_addresses.items()):
            logger.info(f"\n[ISR] 探索 IRQ {irq_num} @ {hex(isr_addr)}")

            covered = self._explore_single_isr(irq_num, isr_addr, max_instructions)
            self.isr_coverage[irq_num] = covered

            logger.info(f"  覆盖了 {len(covered)} 个 BB")

        return self.isr_coverage

    def _explore_single_isr(self, irq_num: int, isr_addr: int, max_instructions: int) -> Set[int]:
        """探索单个 ISR"""
        covered_bbs = set()

        # 保存当前状态
        saved_context = self._save_context()
        saved_causal_context = (
            self.causal_context.snapshot_runtime_state()
            if self.causal_context is not None
            else None
        )

        try:
            # 设置 ISR 执行环境（传入 irq_num）
            self._setup_isr_context(isr_addr, irq_num)

            # 注册 hook 记录覆盖
            def code_hook(uc, address, size, user_data):
                bb_addr = self.instruction_to_bb.get(address)
                if bb_addr is not None:
                    covered_bbs.add(bb_addr)
                elif address in self.static_bbs:
                    covered_bbs.add(address)

            hook = managed_hook_add(self.uc, UC_HOOK_CODE, code_hook)

            # 执行 ISR
            try:
                self.start_isr_execution(
                    isr_addr | 1,
                    timeout=1000000,
                    count=max_instructions
                )
            except Exception:
                pass

            managed_hook_del(self.uc, hook)

        finally:
            self._restore_context(saved_context)
            if self.causal_context is not None and saved_causal_context is not None:
                self.causal_context.restore_runtime_state(saved_causal_context)

        return covered_bbs

    def _execution_start_is_valid(self, start_pc: int) -> bool:
        """Run the owning emulator's preflight without coupling to its class."""
        if self.start_validator is None:
            self.execution_start_stats["validator_not_configured"] += 1
            return True
        try:
            result = self.start_validator(int(start_pc))
            valid = bool(result[0]) if isinstance(result, tuple) else bool(result)
        except Exception as exc:
            self.execution_start_stats["validator_error"] += 1
            logger.warning("[ISR] 启动前检查异常 @ 0x%08x: %s", int(start_pc) & 0xFFFFFFFF, exc)
            return False
        self.execution_start_stats["validated" if valid else "rejected"] += 1
        return valid

    def start_isr_execution(self, start_pc: int, *, timeout: int, count: int) -> bool:
        """Start one ISR only after validating the native-engine invariants."""
        if not self._execution_start_is_valid(start_pc):
            return False
        self.execution_start_stats["started"] += 1
        try:
            managed_emu_start(
                self.uc,
                int(start_pc),
                0,
                timeout=int(timeout),
                count=int(count),
            )
        except Exception:
            self.execution_start_stats["engine_errors"] += 1
            raise
        finally:
            if self.causal_context is not None and self._active_irq is not None:
                self.causal_context.record_irq_return(
                    self._active_irq,
                    pc=int(start_pc) & 0xFFFFFFFF,
                )
                self._active_irq = None
        self.execution_start_stats["completed"] += 1
        return True

    def build_exception_entry(self, **kwargs) -> Dict[str, object]:
        """Architectural exception entry; shares this explorer's stack repair.

        r40 P3/C7：这一包装现在是 explorer 侧的**唯一**冷启动入口构造器
        （``LSGEMU_ISR_LEGACY_FRAME=0`` 时 ``_setup_isr_context`` 走这里），
        与投递/重放侧（``IrqDeliveryController``）共用模块级
        ``build_exception_entry`` + ``ensure_writable_exception_frame``。
        旧体（``0xFFFFFFF9`` / 32B / 手写假寄存器）只在缺省 legacy 臂保留。
        """
        kwargs.setdefault("stack_fixer", self._ensure_exception_stack_frame)
        return build_exception_entry(self.uc, **kwargs)

    def unstack_exception(self, **kwargs) -> Dict[str, object]:
        """Architectural exception return (see module-level ``unstack_exception``)."""
        return unstack_exception(self.uc, **kwargs)

    def _setup_isr_context_shared(
        self,
        isr_addr: int,
        irq_num: int,
        *,
        preserve_registers: bool = False,
        return_pc: Optional[int] = None,
    ) -> Dict[str, object]:
        """r40 P3/C7（P1-2 合并臂）：经共享构造器的 ISR 上下文注入。

        与旧体（``_setup_isr_context``）的差异只在入口构造这一步：
        基本帧 32B、EXC_RETURN=0xFFFFFFF9（Thread/MSP 基本帧，与旧体一致）、
        栈窗修复同源；栈上保存**真实**当前寄存器（旧体写假值 0/0x08001004），
        且额外写 IPSR=irq（旧体不写）。A/B 一致后才删旧体
        （``LSGEMU_ISR_LEGACY_FRAME`` 缺省 = 1 = 旧语义）。
        """
        current_pc = int(self.uc.reg_read(UC_ARM_REG_PC)) & 0xFFFFFFFF
        entry = self.build_exception_entry(
            irq=int(irq_num),
            vector=int(isr_addr) & ~1,
            at_pc=(int(return_pc) & 0xFFFFFFFF) if return_pc is not None else current_pc,
            frame_format=FRAME_FORMAT_BASIC32,
            exc_return=0xFFFFFFF9,
            enter_handler_mode=True,
        )
        # 旧体的探索确定性步骤：冷启动清零寄存器（保留被中断现场时跳过）。
        if not preserve_registers:
            for i in range(13):
                self.uc.reg_write(UC_ARM_REG_R0 + i, 0)
        self._setup_hardware_state(irq_num)
        self._active_irq = int(irq_num)
        if self.causal_context is not None:
            self.causal_context.record_irq_deliver(
                int(irq_num),
                pc=int(isr_addr) & 0xFFFFFFFF,
                source="isr_context_replay",
            )
        self.uc.reg_write(UC_ARM_REG_PC, int(isr_addr) | 1)
        return entry

    def _setup_isr_context(self, isr_addr: int, irq_num: int,
                           preserve_registers: bool = False,
                           return_pc: Optional[int] = None):
        """
        设置 ISR 执行环境（改进版：模拟真实的 CPU 压栈）

        Args:
            isr_addr: ISR 入口地址
            irq_num: 中断号

        r40 P3/C7：``LSGEMU_ISR_LEGACY_FRAME=0`` 显式关闭时走共享构造器
        （``_setup_isr_context_shared``）；缺省保持本旧体逐位不变。
        """
        if not legacy_isr_frame_enabled():
            self._setup_isr_context_shared(
                isr_addr,
                irq_num,
                preserve_registers=preserve_registers,
                return_pc=return_pc,
            )
            return
        interrupted = {}
        if preserve_registers:
            try:
                for i in range(13):
                    interrupted[f"r{i}"] = self.uc.reg_read(UC_ARM_REG_R0 + i)
                interrupted["lr"] = self.uc.reg_read(UC_ARM_REG_LR)
                interrupted["pc"] = self.uc.reg_read(UC_ARM_REG_PC)
                interrupted["cpsr"] = self.uc.reg_read(UC_ARM_REG_CPSR)
            except Exception:
                interrupted = {}

        # 1. 获取当前 SP，并确保异常栈帧所在窗口可写。上下文 ISR 会从
        # snapshot 恢复 SP；如果该 SP 已失效或未映射，先修复栈窗口，避免
        # ISR 注入退化为 UC_ERR_WRITE_UNMAPPED。
        original_sp = self.uc.reg_read(UC_ARM_REG_SP)
        sp, stack_source = self._ensure_exception_stack_frame(original_sp)

        # 3. 构造栈帧
        fake_xpsr = interrupted.get("cpsr", 0x01000000) | 0x01000000  # T 位 = 1（Thumb 模式）
        fake_pc = (return_pc if return_pc is not None else interrupted.get("pc", 0x08001000)) & ~1
        fake_lr = interrupted.get("lr", 0x08001004)
        fake_r12 = interrupted.get("r12", 0)
        fake_r3 = interrupted.get("r3", 0)
        fake_r2 = interrupted.get("r2", 0)
        fake_r1 = interrupted.get("r1", 0)
        fake_r0 = interrupted.get("r0", 0)

        # 4. 写入栈（小端序：R0, R1, R2, R3, R12, LR, PC, xPSR）
        stack_write_ok = True
        try:
            self.uc.mem_write(sp,      fake_r0.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 4,  fake_r1.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 8,  fake_r2.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 12, fake_r3.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 16, fake_r12.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 20, fake_lr.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 24, fake_pc.to_bytes(4, 'little'))
            self.uc.mem_write(sp + 28, fake_xpsr.to_bytes(4, 'little'))
        except Exception as e:
            stack_write_ok = False
            self.stack_repair_stats["stack_push_failures"] += 1
            if not self._stack_warning_emitted:
                logger.warning(
                    "[ISR] 压栈失败: %s (original_sp=%s, frame_sp=%s, source=%s)",
                    e,
                    hex(int(original_sp) & 0xFFFFFFFF),
                    hex(int(sp) & 0xFFFFFFFF),
                    stack_source,
                )
                self._stack_warning_emitted = True
        else:
            self.stack_repair_stats["stack_push_success"] += 1

        # 5. 更新 SP
        if stack_write_ok:
            self.uc.reg_write(UC_ARM_REG_SP, sp)

        # 6. 设置 LR 为 EXC_RETURN
        self.uc.reg_write(UC_ARM_REG_LR, 0xFFFFFFF9)

        # 7. 设置 CPSR 为 Handler 模式
        old_cpsr = self.uc.reg_read(UC_ARM_REG_CPSR)
        new_cpsr = (old_cpsr & ~0x1F) | 0x13  # 0x13 = Supervisor 模式
        self.uc.reg_write(UC_ARM_REG_CPSR, new_cpsr)

        # 8. 冷启动 ISR 探索保持确定性；上下文触发时保留被中断现场。
        if not preserve_registers:
            for i in range(13):
                self.uc.reg_write(UC_ARM_REG_R0 + i, 0)

        # 9. 设置硬件状态（中断标志位）
        self._setup_hardware_state(irq_num)

        self._active_irq = int(irq_num)
        if self.causal_context is not None:
            self.causal_context.record_irq_deliver(
                int(irq_num),
                pc=int(isr_addr) & 0xFFFFFFFF,
                source="isr_context_replay",
            )

        # 10. 设置 PC
        self.uc.reg_write(UC_ARM_REG_PC, isr_addr | 1)

        logger.debug(
            f"[ISR] 设置上下文完成: SP={hex(sp)}, PC={hex(isr_addr)}, IRQ={irq_num}, stack_source={stack_source}"
        )

    def _setup_hardware_state(self, irq_num: int):
        """
        设置硬件状态（改进版：设置中断标志位）

        Args:
            irq_num: 中断号
        """
        flags = dict(self.interrupt_flag_registers) if self.broad_interrupt_flags else {}
        if not self.broad_interrupt_flags:
            flags.update(self.irq_register_observations)
        if irq_num in self.common_irq_mmio:
            flags[self.common_irq_mmio[irq_num]] = 0xFFFFFFFF

        for mmio_addr, flag_value in flags.items():
            try:
                self._write_flag_value(mmio_addr, flag_value)
                logger.debug(f"[ISR] 设置 IRQ {irq_num} 硬件标志位 @ {hex(mmio_addr)} = {hex(flag_value)}")
            except Exception as e:
                logger.debug(f"[ISR] 设置硬件状态失败 @ {hex(mmio_addr)}: {e}")

    def _write_flag_value(self, address: int, value: int):
        """Map and seed an interrupt/status register for ISR path exploration."""
        page_start = address & ~0xfff
        try:
            managed_mem_map(self.uc, page_start, 0x1000)
        except Exception:
            pass

        if self.mmio_handler is not None and hasattr(self.mmio_handler, "mmio_state"):
            self.mmio_handler.mmio_state[address] = value & 0xFFFFFFFF
            marker = getattr(self.mmio_handler, "mark_mmio_state_explicit", None)
            if callable(marker):
                marker([address])

        self._record_irq_event(address, value)
        self.uc.mem_write(address, (value & 0xFFFFFFFF).to_bytes(4, 'little'))

    def _record_irq_event(self, address: int, value: int) -> None:
        addr = int(address) & 0xFFFFFFFF
        value = int(value) & 0xFFFFFFFF
        if addr in {0xE000E100, 0xE000E104}:
            kind = "enable"
        elif addr in {0xE000E200, 0xE000E204}:
            kind = "pending"
        elif addr in self.interrupt_flag_registers or addr in self.common_irq_mmio.values():
            kind = "status"
        else:
            kind = "other"
        self.irq_event_counts[kind] += 1
        self.irq_event_history.append({
            "kind": kind,
            "address": f"0x{addr:08x}",
            "value": f"0x{value:08x}",
        })
        if kind in {"enable", "pending", "status"}:
            self.irq_register_observations[addr] = value
        if kind in {"enable", "pending"}:
            base = 0xE000E100 if kind == "enable" else 0xE000E200
            irq_base = ((addr - base) // 4) * 32
            for bit in range(32):
                if value & (1 << bit):
                    irq = int(irq_base + bit)
                    if kind == "enable":
                        self.nvic_enabled_irqs.add(irq)
                    else:
                        self.nvic_pending_irqs.add(irq)
        elif kind == "status":
            self.status_register_hits[addr] += 1
        if self.causal_context is not None:
            self.causal_context.record_mmio_write(
                pc=0,
                address=addr,
                value=value,
                size=4,
            )

    def learned_irq_candidates(self, limit: int = 0) -> List[Dict[str, object]]:
        """Return IRQ candidates learned from observed NVIC/status state."""
        candidates = []
        all_irqs = set(self.isr_addresses)
        enabled = set(self.nvic_enabled_irqs)
        pending = set(self.nvic_pending_irqs)
        for irq_num in sorted(all_irqs):
            score = 0
            reasons = []
            if irq_num in pending:
                score += 100
                reasons.append("nvic_pending")
            if irq_num in enabled:
                score += 80
                reasons.append("nvic_enabled")
            if irq_num in self.common_irq_mmio and self.common_irq_mmio[irq_num] in self.irq_register_observations:
                score += 60
                reasons.append("status_observed")
            if score <= 0:
                continue
            candidates.append({
                "irq": int(irq_num),
                "isr_addr": f"0x{int(self.isr_addresses[irq_num]) & 0xFFFFFFFF:08x}",
                "score": int(score),
                "reasons": reasons,
            })
        candidates.sort(key=lambda item: (int(item["score"]), -int(item["irq"])), reverse=True)
        if limit and limit > 0:
            return candidates[: int(limit)]
        return candidates

    def _save_context(self) -> Dict:
        """保存 CPU 上下文"""
        context = {}

        try:
            for i in range(13):
                context[f'r{i}'] = self.uc.reg_read(UC_ARM_REG_R0 + i)

            context['sp'] = self.uc.reg_read(UC_ARM_REG_SP)
            context['lr'] = self.uc.reg_read(UC_ARM_REG_LR)
            context['pc'] = self.uc.reg_read(UC_ARM_REG_PC)
            context['cpsr'] = self.uc.reg_read(UC_ARM_REG_CPSR)

        except Exception as e:
            logger.warning(f"保存上下文失败: {e}")

        return context

    def _restore_context(self, context: Dict):
        """恢复 CPU 上下文"""
        try:
            for i in range(13):
                if f'r{i}' in context:
                    self.uc.reg_write(UC_ARM_REG_R0 + i, context[f'r{i}'])

            if 'sp' in context:
                self.uc.reg_write(UC_ARM_REG_SP, context['sp'])
            if 'lr' in context:
                self.uc.reg_write(UC_ARM_REG_LR, context['lr'])
            if 'pc' in context:
                self.uc.reg_write(UC_ARM_REG_PC, context['pc'])
            if 'cpsr' in context:
                self.uc.reg_write(UC_ARM_REG_CPSR, context['cpsr'])

        except Exception as e:
            logger.warning(f"恢复上下文失败: {e}")

    def get_total_coverage(self) -> Set[int]:
        """获取所有 ISR 的总覆盖"""
        total = set()
        for covered in self.isr_coverage.values():
            total.update(covered)
        return total

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_coverage = self.get_total_coverage()

        return {
            'total_isrs': len(self.isr_addresses),
            'explored_isrs': len(self.isr_coverage),
            'total_coverage': len(total_coverage),
            'isr_details': {
                irq_num: len(covered)
                for irq_num, covered in self.isr_coverage.items()
            },
            'irq_event_counts': dict(self.irq_event_counts),
            'irq_event_history_sample': self.irq_event_history[-16:],
            'irq_register_observations': {
                f"0x{address:08x}": f"0x{value & 0xFFFFFFFF:08x}"
                for address, value in sorted(self.irq_register_observations.items())
            },
            'nvic_enabled_irqs': sorted(self.nvic_enabled_irqs)[:64],
            'nvic_pending_irqs': sorted(self.nvic_pending_irqs)[:64],
            'status_register_hits': {
                f"0x{address:08x}": int(count)
                for address, count in self.status_register_hits.most_common(32)
            },
            'learned_irq_candidates': self.learned_irq_candidates(limit=32),
            'isr_stack': dict(self.stack_repair_stats),
            'invalid_stack_samples': list(self.invalid_stack_samples),
            'execution_start': dict(self.execution_start_stats),
        }


# ---------------------------------------------------------------------------
# r31: model-driven interrupt delivery for the replay path
# ---------------------------------------------------------------------------
class IrqDeliveryController:
    """Deliver model-driven interrupts while a replay is running.

    This is the production counterpart of the r30b probe.  It never patches the
    MMIO handler: the device state lives in registered semantic profiles
    (read-path priority 3) and the ticks come from the executed-instruction
    count, so the same input sequence always produces the same delivery log.

    Ownership of the pieces:

    * ``TIM5SemanticProfile`` decides *whether* an event is due (firmware-written
      ``CR1/DIER/PSC/ARR/CCR1``), and owns the virtual clock and its remainder.
    * ``CortexMSCSInterruptProfile`` derives ``ICSR.RETTOBASE`` from the active
      exception stack this controller maintains.
    * ``build_exception_entry`` / ``unstack_exception`` supply the architectural
      entry and return.
    * The emulator only learns about *boundaries* (``irq_delivery_boundary``) so
      branch/coverage bookkeeping does not invent an edge across the hijack.

    Nothing here is enabled unless :meth:`install` is called.
    """

    TIM5_IRQ = 50
    TIM5_VECTOR = 0x08135AB8
    # r32 D3：SVCall 是异常 11（ARMv7-M 架构固定编号）。ChibiOS 的
    # ``__port_exit_from_isr``(0x08005106) 用 ``svc #0`` 请求一次「从 ISR
    # 返回线程」的上下文切换；不派发异常 11 就必然落到紧随的 ``b .``
    # (0x08005108) 终止汇点。
    SVCALL_IRQ = 11
    SVCALL_NUMBER = 0

    def __init__(
        self,
        emulator,
        *,
        irq: int = TIM5_IRQ,
        vector: Optional[int] = None,
        mmio_handler=None,
        frame_format: str = FRAME_FORMAT_EXT104,
        exc_return: int = EXC_RETURN_THREAD_PSP_FPU,
        enable_otgfs: bool = True,
    ):
        self.emu = emulator
        self.uc = getattr(emulator, "uc", None)
        self.mmio_handler = mmio_handler if mmio_handler is not None else getattr(
            emulator, "mmio_handler", None
        )
        self.registry = getattr(self.mmio_handler, "semantic_profiles", None)
        self.irq = int(irq)
        self.vector = None if vector is None else int(vector) & ~1
        self.frame_format = frame_format
        self.exc_return = int(exc_return)
        self.enable_otgfs = bool(enable_otgfs)

        self.installed = False
        self.vector_table_base: Optional[int] = None
        self.instructions = 0
        self.deliveries: List[Dict[str, object]] = []
        self.unstacks: List[Dict[str, object]] = []
        self.errors: Counter = Counter()
        self.intr_counts: Counter = Counter()
        self.boundary_serial = 0
        self._hooks: List[object] = []
        self._cached_profiles_list: Optional[object] = None
        self._tim5 = None
        self._scs = None
        self._otgfs = None
        self._dwt = None
        self.otgfs_reads: List[Dict[str, object]] = []
        # r32 D3：SVC #0 派发（默认关，与投递同一控制器生命周期）。派发判据 =
        # 真的执行到 ``svc #0`` 这条指令（架构事件），不是超时/判停放宽。
        self.svc_enabled = os.environ.get("LSGEMU_SVC_DISPATCH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.svc_vector: Optional[int] = None
        self.svc_exc_return = EXC_RETURN_THREAD_PSP_FPU
        self.svcs: List[Dict[str, object]] = []
        self.svc_stats: Counter = Counter()
        # 异常活动栈：投递与 SVC 共用，保证解栈时 pop 的是对应那层。
        self._active_stack: List[int] = []
        self._prev_svc_dispatch_handler: Optional[object] = None
        self._svc_dispatch_callable: Optional[object] = None
        # r40 P3/C7（Side B 缺陷一/三）：投递入口带上与 explorer 同源的栈窗
        # 修复（此前 ``_deliver`` 完全不带 stack_fixer，SP 失效即入口失败）。
        self.stack_repair_stats: Counter = Counter()
        self.invalid_stack_samples: List[Dict[str, object]] = []

    # -- profile access ---------------------------------------------------
    def _resolve_profiles(self) -> None:
        """Re-resolve on profile-list identity change.

        ``SemanticProfileRegistry.restore_runtime_state`` replaces the profile
        objects with deepcopies, so a cached instance would silently diverge
        from the registry after any snapshot restore.
        """
        registry = self.registry
        if registry is None:
            return
        current = registry.profiles
        if current is self._cached_profiles_list:
            return
        self._cached_profiles_list = current
        self._tim5 = registry.profile_by_name("tim5")
        self._scs = registry.profile_by_name("cortexm_scs_irq")
        self._otgfs = registry.profile_by_name("otgfs_reset")
        self._dwt = registry.profile_by_name("cortexm_dwt")

    @property
    def tim5(self):
        self._resolve_profiles()
        return self._tim5

    @property
    def scs(self):
        self._resolve_profiles()
        return self._scs

    @property
    def otgfs(self):
        self._resolve_profiles()
        return self._otgfs

    @property
    def dwt(self):
        self._resolve_profiles()
        return self._dwt

    # -- lifecycle --------------------------------------------------------
    def _is_plausible_handler(self, address: int) -> bool:
        """Reject a vector entry that cannot be code.

        A wrong VTOR would otherwise send the CPU to a data word; failing the
        install is the honest outcome, not a silent jump into garbage.
        """
        address = int(address) & ~1
        if address == 0:
            return False
        for attr in ("static_bbs", "instruction_to_bb"):
            table = getattr(self.emu, attr, None)
            if isinstance(table, dict) and address in table:
                return True
        return 0x08000000 <= address < 0x08200000

    def _vector_table_base(self) -> int:
        vtor = int(getattr(self.emu, "vector_table_base", 0) or 0)
        if not vtor:
            getter = getattr(self.emu, "_vector_table_base", None)
            if callable(getter):
                try:
                    vtor = int(getter()) & 0xFFFFFFFF
                except Exception:
                    vtor = 0
        return vtor or 0x08000000

    def _read_vector_index(self, index: int) -> Optional[int]:
        """按**向量表下标**（= ARMv7-M 异常号）读一项并做可信性检查。

        下标语义是本文件的活口子：外部 IRQ n 的异常号 = 16 + n，而 SVCall 是
        **系统异常 11**（不是外部 IRQ 11）⇒ 下标 11、字节偏移 0x2C。
        r32 实测踩过这个坑：用 ``(16+11)*4 = 0x6C`` 取到的是外部 IRQ11 的
        处理函数（0x0812F54C），SVC 会被派发进一个外设向量，SVC_Handler
        0x0812ED34 永远不进覆盖集。
        """
        vtor = self._vector_table_base()
        try:
            offset = int(index) * 4
            raw = bytes(self.uc.mem_read(vtor + offset, 4))
            candidate = int.from_bytes(raw, "little") & ~1
        except Exception as exc:
            self.errors[f"vector_read:{type(exc).__name__}"] += 1
            return None
        if not self._is_plausible_handler(candidate):
            self.errors["vector_implausible"] += 1
            logger.warning(
                "[r31] 向量[%d] 不可信: vtor=0x%08x -> 0x%08x",
                int(index),
                vtor,
                candidate,
            )
            return None
        return candidate

    def _read_irq_vector(self, irq: int) -> Optional[int]:
        """外部 IRQ ``irq`` 的处理函数（向量表下标 = 16 + irq）。"""
        return self._read_vector_index(16 + int(irq))

    def _resolve_vector(self) -> Optional[int]:
        if self.vector is not None:
            return self.vector
        candidate = self._read_irq_vector(self.irq)
        if candidate is None:
            return None
        self.vector = candidate
        self.vector_table_base = self._vector_table_base()
        return candidate

    def install(self) -> bool:
        if self.installed:
            return True
        from ..analysis.peripheral_semantic_profiles import (
            CortexMDWTProfile,
            CortexMSCSInterruptProfile,
            OTGFSResetProfile,
            TIM5SemanticProfile,
        )

        # Resolve the vector *before* touching the registry: a failed install
        # must leave the MMIO read path exactly as it was.
        if self._resolve_vector() is None:
            return False
        if self.registry is not None:
            # P0-B（cycle3 k.5 C1）：TIM5 CEN 血统闭合。install() 在快照恢复
            # 之后执行（historical_runner 装配序），blob 普查 0/1796 条快照
            # ``tim5`` profile_state 带 CEN、4190/4470 条 MMIO 窗带 ⇒ 唯一
            # 权威 CEN 源是 ``mmio_handler`` 已恢复的窗（``peek_register_value``
            # → ``_read_stored_register_value``，stateful_mmio_handler.py）。
            # 硬条款：窗缺（返回 None）⇒ 不写该寄存器（armed() 维持 False）；
            # **禁播 CNT**——虚拟时间由模型自持 ``self.cnt`` 驱动，播 CNT 即
            # 双计（peripheral_semantic_profiles.py CNT/时间基注释）。
            tim5_profile = TIM5SemanticProfile()
            seed_reader = getattr(self.mmio_handler, "peek_register_value", None)
            if callable(seed_reader):
                for _offset in (TIM5SemanticProfile.CR1, TIM5SemanticProfile.DIER):
                    _seeded = seed_reader(TIM5SemanticProfile.BASE + _offset, 4)
                    if _seeded is not None:
                        tim5_profile.regs[_offset] = int(_seeded) & 0xFFFFFFFF
            self.registry.register_profile(tim5_profile)
            self.registry.register_profile(CortexMSCSInterruptProfile())
            # DWT_CYCCNT is the polled-delay time base.  Without it
            # chSysPolledDelayX (0x08133CDC) never terminates and
            # usb_lld_start stalls between the CSRST write and its wait.
            self.registry.register_profile(CortexMDWTProfile())
            if self.enable_otgfs:
                self.registry.register_profile(OTGFSResetProfile())
        self._resolve_profiles()
        import unicorn

        self._hooks.append(
            self.uc.hook_add(unicorn.UC_HOOK_CODE, self._code_hook)
        )
        self._hooks.append(
            self.uc.hook_add(unicorn.UC_HOOK_INTR, self._intr_hook)
        )
        if self.svc_enabled:
            # SVC 派发挂在 emulator 自己的 ``_svc_instruction_hook`` 上
            # （``svc_dispatch_handler`` 协议）：那条 hook 就是原本会「写 PC
            # 跳过 SVC」的地方，改由这里做真实异常入口，避免两条 CODE hook
            # 争抢 PC。emulator 未启用该 hook（LSGEMU_HANDLE_SVC_AS_NOOP=0）
            # 时明确报不可用，不假装支持。
            if not getattr(self.emu, "handle_svc_as_noop", False):
                self.errors["svc_dispatch_unavailable"] += 1
                logger.warning("[r32] SVC 派发不可用：emulator 未注册 svc hook")
            else:
                # SVCall = 系统异常 11 ⇒ 向量表下标 11（不是外部 IRQ 11）。
                self.svc_vector = self._read_vector_index(self.SVCALL_IRQ)
                if self.svc_vector is None:
                    self.errors["svc_vector_unavailable"] += 1
                    logger.warning("[r32] SVC 派发不可用：向量表 vec[11] 不可信")
                else:
                    self._prev_svc_dispatch_handler = getattr(
                        self.emu, "svc_dispatch_handler", None
                    )
                    # 绑定方法每次取属性都是新对象，比对必须用身份稳定的引用。
                    self._svc_dispatch_callable = self._dispatch_svc
                    self.emu.svc_dispatch_handler = self._svc_dispatch_callable
        self.installed = True
        self.emu.irq_delivery_enabled = True
        self.emu.irq_delivery_controller = self
        self.emu.irq_delivery_boundary = self.boundary_serial
        self.emu._irq_delivery_boundary_seen = self.boundary_serial
        return True

    def uninstall(self) -> None:
        for hook in self._hooks:
            try:
                self.uc.hook_del(hook)
            except Exception:
                continue
        self._hooks = []
        mounted = getattr(self, "_svc_dispatch_callable", None)
        if mounted is not None and getattr(
            self.emu, "svc_dispatch_handler", None
        ) is mounted:
            self.emu.svc_dispatch_handler = self._prev_svc_dispatch_handler
        self._svc_dispatch_callable = None
        self.installed = False
        self.emu.irq_delivery_enabled = False
        self.emu.irq_delivery_controller = None

    # -- event state ------------------------------------------------------
    def has_pending_event(self) -> bool:
        """True when a modelled hardware event is due or still coming."""
        tim5 = self.tim5
        if tim5 is not None:
            if tim5.due():
                return True
            if tim5.armed() and not tim5.delivered_pending:
                return True
        context = getattr(self.emu, "causal_context", None)
        pending = getattr(context, "pending_irqs", None)
        if pending:
            return True
        return False

    def _bump_boundary(self) -> None:
        self.boundary_serial += 1
        self.emu.irq_delivery_boundary = self.boundary_serial

    def _sync_stack_shadow(self) -> None:
        """让 emulator 的 MSR/MRS 软件影子寄存器跟真实栈指针对齐。

        ``IntelligentEmulator`` 用 ``cortex_m_system_registers`` 影子字典解释
        ``mrs rX, psp``；而 ``build_exception_entry`` / ``unstack_exception``
        直接写 unicorn 的 PSP/MSP，影子不会被更新。``SVC_Handler``
        (0x0812ED34) 正是 ``mrs r3, psp`` + ``adds r3, #104`` + ``msr psp, r3``
        ——影子过期会把 SVC 返回帧指到错误地址。入口/解栈之后各同步一次。
        """
        shadow = getattr(self.emu, "cortex_m_system_registers", None)
        if not isinstance(shadow, dict):
            return
        for name, reg in (("psp", UC_ARM_REG_PSP), ("msp", UC_ARM_REG_MSP)):
            if name not in shadow:
                continue
            try:
                shadow[name] = int(self.uc.reg_read(reg)) & 0xFFFFFFFF
            except Exception:
                continue

    # -- delivery ---------------------------------------------------------
    def _stack_fixer(self, requested_sp: int, frame_size: int) -> Tuple[int, str]:
        """r40 P3/C7：投递路径共用 ``ensure_writable_exception_frame``。

        与 explorer 冷启动注入同一条修复级联（native SP → 向量表初值 →
        已映射可写 RAM → 缺省 SRAM 栈）；vtor 用安装期解析出的向量表基址。
        """
        return ensure_writable_exception_frame(
            self.uc,
            requested_sp,
            frame_size,
            vtor=int(self.vector_table_base or 0x08000000),
            repair_stats=self.stack_repair_stats,
            invalid_samples=self.invalid_stack_samples,
        )

    def _deliver(self, at_pc: int) -> Optional[Dict[str, object]]:
        tim5 = self.tim5
        scs = self.scs
        try:
            entry = build_exception_entry(
                self.uc,
                irq=self.irq,
                vector=self.vector,
                at_pc=at_pc,
                frame_format=self.frame_format,
                exc_return=self.exc_return,
                stack_fixer=self._stack_fixer,
            )
        except Exception as exc:
            self.errors[f"entry:{type(exc).__name__}:{exc}"] += 1
            return None

        self._sync_stack_shadow()
        if tim5 is not None:
            tim5.note_delivered(pc=at_pc, instructions=self.instructions)
        if scs is not None:
            scs.push_active(self.irq)
        self._active_stack.append(self.irq)
        self._bump_boundary()
        # A delivery is progress: the no-progress watchdog starts over.
        self.emu.irq_delivery_watchdog_spins = 0

        derivation = getattr(scs, "last_derivation", None) if scs is not None else None
        record: Dict[str, object] = {
            "irq": self.irq,
            "vector": f"0x{int(self.vector) & 0xFFFFFFFF:08x}",
            "delivered_at_pc": f"0x{int(at_pc) & 0xFFFFFFFF:08x}",
            "ipsr_before": "0x00000000",
            "instructions": int(self.instructions),
            "frame_base": entry["frame_base"],
            "frame_size": entry["frame_size"],
            "exc_return": entry["exc_return"],
            "frame_format": entry["frame_format"],
            "stacked_pc": entry["interrupted_pc"],
            "active_exceptions": list(getattr(scs, "active_exceptions", []) or []),
            "rettobase_derivation": derivation,
            "cnt": None if tim5 is None else f"0x{int(tim5.cnt) & 0xFFFFFFFF:08x}",
            "ccr1": None if tim5 is None else f"0x{tim5.deadline():08x}",
            "sr": None
            if tim5 is None
            else f"0x{int(tim5.regs.get(tim5.SR, 0)) | (tim5.SR_CC1IF if tim5.cc1if else 0):08x}",
            "dier": None
            if tim5 is None
            else f"0x{int(tim5.regs.get(tim5.DIER, 0)) & 0xFFFFFFFF:08x}",
            "source": "model_driven_tim5",
        }
        self.deliveries.append(record)

        context = getattr(self.emu, "causal_context", None)
        if context is not None:
            try:
                context.record_irq_deliver(
                    self.irq, pc=int(at_pc) & 0xFFFFFFFF, source="model_driven_tim5"
                )
            except Exception:
                pass
        logger.info(
            "[r31] 投递 IRQ%d @ pc=0x%08x 帧=0x%s(%dB) exc_return=0x%s",
            self.irq,
            int(at_pc) & 0xFFFFFFFF,
            str(entry["frame_base"])[2:],
            entry["frame_size"],
            str(entry["exc_return"])[2:],
        )
        return record

    def _on_exception_exit(self) -> None:
        result = unstack_exception(self.uc)
        self._sync_stack_shadow()
        scs = self.scs
        # 解栈对应的是「最近一次入口」那层：投递与 SVC 共用这一个栈，
        # 混用 self.irq 会在 SVC 返回时错误地 pop 掉 TIM5 那层。
        irq = self._active_stack.pop() if self._active_stack else self.irq
        if scs is not None:
            scs.pop_active(irq)
        self._bump_boundary()
        record = dict(result)
        record["irq"] = int(irq)
        record["instructions"] = int(self.instructions)
        record["active_exceptions"] = list(
            getattr(scs, "active_exceptions", []) or []
        )
        self.unstacks.append(record)
        context = getattr(self.emu, "causal_context", None)
        if context is not None:
            try:
                context.record_irq_return(int(irq), pc=int(self.uc.reg_read(UC_ARM_REG_PC)))
            except Exception:
                pass

    # -- r32 D3: SVC (#0 -> exception 11) dispatch -------------------------
    def _dispatch_svc(self, address, insn, size, svc_no) -> bool:
        """把一条**正在执行**的 ``svc #0`` 变成一次真实的异常入口。

        判据是架构事件本身（执行到 svc 指令且 svc 号 = 0），与超时/判停规则
        无关。入口用 ``build_exception_entry``（104B 扩展帧、EXC_RETURN =
        Thread/PSP 扩展帧 0xFFFFFFED）；返回走既有的 ``unstack_exception``
        （unicorn 在 ``bx lr`` 目标为 EXC_RETURN 时抛 intno=8）。

        只在 Thread 模式派发：``SVC_Handler``(0x0812ED34) 的语义是
        ``mrs r3, psp`` + ``adds r3, #104``（丢弃异常帧）+ ``bx lr``——
        只有 Thread/PSP 入口才与它一致。Handler 模式下的 svc 保持既有 no-op
        语义并显式计数，不假装派发过。

        返回 True 表示「本条 svc 已按异常派发处理」，调用方不得再写 PC 跳过。
        """
        if int(svc_no) != self.SVCALL_NUMBER:
            self.svc_stats["other_number"] += 1
            return False
        if self.svc_vector is None:
            self.svc_stats["vector_unavailable"] += 1
            return False
        try:
            ipsr = int(self.uc.reg_read(UC_ARM_REG_IPSR)) & 0x1FF
        except Exception:
            ipsr = -1
        if ipsr != 0:
            self.svc_stats["handler_mode_skipped"] += 1
            return False
        at_pc = int(address) & 0xFFFFFFFF
        # r32 D3 证据：入口前同时记录两个栈，事后可验证「帧落在 PSP」这一
        # 架构前提（SVC_Handler 的 psp += 104 依赖它）。
        try:
            psp_before = int(self.uc.reg_read(UC_ARM_REG_PSP)) & 0xFFFFFFFF
            msp_before = int(self.uc.reg_read(UC_ARM_REG_MSP)) & 0xFFFFFFFF
        except Exception:
            psp_before = msp_before = 0
        try:
            entry = build_exception_entry(
                self.uc,
                irq=self.SVCALL_IRQ,
                vector=self.svc_vector,
                at_pc=at_pc,
                frame_format=FRAME_FORMAT_EXT104,
                exc_return=self.svc_exc_return,
                stack_fixer=self._stack_fixer,
            )
        except Exception as exc:
            self.errors[f"svc_entry:{type(exc).__name__}:{exc}"] += 1
            return False
        self._sync_stack_shadow()
        scs = self.scs
        if scs is not None:
            scs.push_active(self.SVCALL_IRQ)
        self._active_stack.append(self.SVCALL_IRQ)
        self._bump_boundary()
        # 一次派发 = 一次进展；无进展看门狗重新起算。
        self.emu.irq_delivery_watchdog_spins = 0
        record: Dict[str, object] = {
            "irq": self.SVCALL_IRQ,
            "svc_number": int(svc_no),
            "dispatched_at_pc": f"0x{at_pc:08x}",
            "vector": f"0x{int(self.svc_vector) & 0xFFFFFFFF:08x}",
            "instructions": int(self.instructions),
            "frame_base": entry["frame_base"],
            "frame_size": entry["frame_size"],
            "exc_return": entry["exc_return"],
            "frame_format": entry["frame_format"],
            "stacked_pc": entry["interrupted_pc"],
            "psp_before": f"0x{psp_before:08x}",
            "msp_before": f"0x{msp_before:08x}",
            "ipsr_before": 0,
            "source": "model_driven_svc",
        }
        self.svcs.append(record)
        context = getattr(self.emu, "causal_context", None)
        if context is not None:
            try:
                context.record_irq_deliver(
                    self.SVCALL_IRQ, pc=at_pc, source="model_driven_svc"
                )
            except Exception:
                pass
        logger.info(
            "[r32] 派发 SVC#%d @ pc=0x%08x -> handler=0x%08x 帧=0x%s(%dB) exc_return=0x%s",
            int(svc_no),
            at_pc,
            int(self.svc_vector) & 0xFFFFFFFF,
            str(entry["frame_base"])[2:],
            entry["frame_size"],
            str(entry["exc_return"])[2:],
        )
        return True

    # -- hooks ------------------------------------------------------------
    def _code_hook(self, uc, address, size, user_data):
        self.instructions += 1
        tim5 = self.tim5
        dwt = self.dwt
        if dwt is not None:
            dwt.advance_by_instructions(1)
        if tim5 is None:
            return
        tim5.advance_by_instructions(1)

        # r32 D3：若同一条指令上更早的 hook（emulator 的 svc 派发）已经改写了
        # PC，本条 hook 就不再按"当前 PC = address"做任何投递判定——否则会把
        # 刚落地的 SVC 异常入口用一次投递覆盖掉。
        try:
            if (int(uc.reg_read(UC_ARM_REG_PC)) & ~1) != (int(address) & ~1):
                return
        except Exception:
            pass

        # While the CPU parks on an RTOS wait sink, let virtual time pass to the
        # next compare event instead of burning instructions.  This is a time
        # skip, not a control-flow skip: the delivered interrupt still has to
        # wake the scheduler on its own.
        try:
            mapped_bb = self.emu.instruction_to_bb.get(int(address), int(address))
        except Exception:
            mapped_bb = int(address)
        if (
            self.emu.terminal_self_loop_bbs
            and mapped_bb in self.emu.terminal_self_loop_bbs
            and self.emu._is_interruptible_wait_sink(mapped_bb)
        ):
            try:
                tim5.fast_forward_to_deadline()
            except Exception:
                self.errors["fast_forward"] += 1

        try:
            if int(uc.reg_read(UC_ARM_REG_IPSR)) != 0:
                return
        except Exception:
            return
        if not tim5.due():
            return
        self._deliver(address)

    def _intr_hook(self, uc, intno, user_data):
        self.intr_counts[int(intno)] += 1
        if int(intno) != 8:  # not EXCP_EXCEPTION_EXIT
            # Never swallow: an unrelated exception keeps unicorn's own handling.
            return False
        try:
            self._on_exception_exit()
        except Exception as exc:
            self.errors[f"unstack:{type(exc).__name__}:{exc}"] += 1
        return True

    # -- audit ------------------------------------------------------------
    def audit_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "enabled": bool(self.installed),
            "irq": self.irq,
            "vector": None
            if self.vector is None
            else f"0x{self.vector & 0xFFFFFFFF:08x}",
            "vector_table_base": None
            if self.vector_table_base is None
            else f"0x{self.vector_table_base & 0xFFFFFFFF:08x}",
            "frame_format": self.frame_format,
            "exc_return": f"0x{self.exc_return:08x}",
            "instructions": int(self.instructions),
            "delivery_count": len(self.deliveries),
            "unstack_count": len(self.unstacks),
            "errors": {str(key): int(value) for key, value in self.errors.items()},
            "intr_counts": {str(key): int(value) for key, value in self.intr_counts.items()},
            # r40 P3/C7：投递入口的栈窗修复审计（与 explorer 同源的唯一实现）。
            "stack_repair_stats": {
                str(key): int(value)
                for key, value in self.stack_repair_stats.items()
            },
            "invalid_stack_samples": list(self.invalid_stack_samples)[-4:],
            "deferrals": {
                str(key): int(value)
                for key, value in getattr(
                    self.emu, "irq_delivery_deferral_stats", Counter()
                ).items()
            },
            "watchdog": {
                "limit": int(getattr(self.emu, "irq_delivery_watchdog_limit", 0) or 0),
                "spins": int(getattr(self.emu, "irq_delivery_watchdog_spins", 0) or 0),
            },
            "deliveries": list(self.deliveries)[:32],
            "unstacks": list(self.unstacks)[:32],
            "svc_dispatch_enabled": bool(self.svc_enabled),
            "svc_handler_vector": None
            if self.svc_vector is None
            else f"0x{int(self.svc_vector) & 0xFFFFFFFF:08x}",
            "svc_dispatch_count": len(self.svcs),
            "svc_stats": {str(k): int(v) for k, v in self.svc_stats.items()},
            "svc_dispatches": list(self.svcs)[:32],
            "otgfs_reads": list(self.otgfs_reads)[:8],
            # Fake-edge accounting: edges actually recorded vs edges suppressed
            # because an interrupt boundary sat between the two BBs.
            "branch_edges_recorded": len(
                list(getattr(getattr(self.emu, "path_explorer", None), "branch_points", []) or [])
            ),
            "branch_edges_cut": int(
                getattr(self.emu, "irq_delivery_deferral_stats", Counter()).get(
                    "branch_edges_cut", 0
                )
            ),
            "branch_snapshot_edges": len(
                list(
                    getattr(
                        getattr(self.emu, "branch_snapshot_manager", None), "snapshots", {}
                    )
                    or {}
                )
            ),
            "coverage_count": len(getattr(self.emu, "bb_addr_set", set()) or set()),
            # Read-path attribution for the interrupt registers: proves the
            # registered profile (priority 3) was not masked by a firmware write
            # mirror (priority 1) or a file constraint (priority 2).
            "read_sources": (
                self.mmio_handler.interrupt_register_read_sources()
                if callable(
                    getattr(self.mmio_handler, "interrupt_register_read_sources", None)
                )
                else {}
            ),
            "mmio_handler_class": type(self.mmio_handler).__name__
            if self.mmio_handler is not None
            else None,
            "static_constraints_on_profiles": _profile_range_constraints(
                getattr(self.mmio_handler, "static_constraints", None)
            ),
        }
        for name, profile in (
            ("tim5", self.tim5),
            ("scs", self.scs),
            ("otgfs", self.otgfs),
            ("dwt", self.dwt),
        ):
            summary = getattr(profile, "summary", None)
            if callable(summary):
                try:
                    payload[f"{name}_summary"] = summary()
                except Exception:
                    continue
            # r41 D1：四个器件面键无条件在（profile 未注册 = None，不是缺席）
            # ——零投递审计的消费者按键取 tim5_summary 时不需要猜缺键含义。
            payload.setdefault(f"{name}_summary", None)
        return payload

    def run_result_fields(self) -> Dict[str, object]:
        """Fields the replay result carries so evidence_contract can credit it.

        ``irq_event_delivered`` is an *environment input fact* (see
        ``evidence_contract.environment_input_facts``), not a control
        intervention; the accompanying audit blob is what makes the credit
        checkable.

        r41 D1：零投递不再整体 fail-silent。审计面（装到没装、跑没跑到、
        TIM5 有无 armed、错误是什么）无条件随 ``run_result`` 走相位聚合，
        使 ``instructions > 0 ∧ delivery_count > 0`` 这类机器判据在字段
        层面可判；但 ``irq_event_delivered`` / ``svc_dispatch`` 仍然只在
        真投递/真派发时出现——字段缺席 = 未发生，不给证据面送 0 值填充。
        """
        if not self.deliveries and not self.svcs:
            return {
                "irq_delivery": {
                    "source": "model_driven_tim5",
                    "knob": False,
                    "delivery_count": 0,
                    "unstack_count": 0,
                    "unstack_count_all_exceptions": len(self.unstacks),
                    "audit": self.audit_payload(),
                },
            }
        fields: Dict[str, object] = {}
        if self.svcs:
            # r32 D3：SVC 派发证据（不是环境输入事实，只是重放审计字段；
            # 覆盖记账仍由真实执行 + capture_coverage 决定）。
            first_svc = self.svcs[0]
            fields["svc_dispatch"] = {
                "source": "model_driven_svc",
                "irq": self.SVCALL_IRQ,
                "svc_number": int(first_svc.get("svc_number", 0)),
                "dispatch_count": len(self.svcs),
                "handler_vector": first_svc.get("vector"),
                "dispatched_at_pc": first_svc.get("dispatched_at_pc"),
                "frame_size": first_svc.get("frame_size"),
                "exc_return": first_svc.get("exc_return"),
                "unstack_count": sum(
                    1
                    for record in self.unstacks
                    if int(record.get("irq", -1)) == self.SVCALL_IRQ
                ),
                "stats": {str(k): int(v) for k, v in self.svc_stats.items()},
                "dispatches": list(self.svcs)[:8],
            }
        if not self.deliveries:
            return fields
        first = self.deliveries[0]
        fields.update({
            "irq_event_delivered": "interrupt_delivery",
            "irq": self.irq,
            "isr_address": None if self.vector is None else f"0x{self.vector:08x}",
            "irq_delivery": {
                "source": "model_driven_tim5",
                "knob": False,
                "delivery_count": len(self.deliveries),
                # r32 D3：解栈计数按异常号归属。SVC 派发也复用同一条
                # unstack 通路，混计会让「4 投 4 解」这类口径失真。
                "unstack_count": sum(
                    1
                    for record in self.unstacks
                    if int(record.get("irq", -1)) == self.irq
                ),
                "unstack_count_all_exceptions": len(self.unstacks),
                "vector": first.get("vector"),
                "frame_base": first.get("frame_base"),
                "frame_size": first.get("frame_size"),
                "exc_return": first.get("exc_return"),
                "delivered_at_pc": first.get("delivered_at_pc"),
                "rettobase_derivation": first.get("rettobase_derivation"),
                "audit": self.audit_payload(),
            },
        })
        return fields
