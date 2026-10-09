#!/usr/bin/env python3
"""K1: architectural exception entry / unstack round-trip.

The fixtures here are pure architecture: a mapped stack window, known register
values, and an explicit EXC_RETURN.  No firmware, no MMIO, no server.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  (installs the unicorn binding patch)
import unicorn
from unicorn import UC_ARCH_ARM, UC_MODE_THUMB, UC_PROT_ALL
from unicorn.arm_const import (
    UC_ARM_REG_APSR,
    UC_ARM_REG_CPSR,
    UC_ARM_REG_FPSCR,
    UC_ARM_REG_IPSR,
    UC_ARM_REG_LR,
    UC_ARM_REG_MSP,
    UC_ARM_REG_PC,
    UC_ARM_REG_PSP,
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_R3,
    UC_ARM_REG_R12,
    UC_ARM_REG_S0,
    UC_ARM_REG_SP,
    UC_CPU_ARM_CORTEX_M4,
)

from lsgemu.isr_explorer.isr_explorer import (
    EXC_RETURN_THREAD_PSP,
    EXC_RETURN_THREAD_PSP_FPU,
    FRAME_FORMAT_BASIC32,
    FRAME_FORMAT_EXT104,
    FRAME_OFFSET_FPSCR,
    FRAME_OFFSET_PC,
    FRAME_OFFSET_S0,
    FRAME_OFFSET_XPSR,
    FRAME_SIZE_BASIC,
    FRAME_SIZE_EXTENDED,
    build_exception_entry,
    exception_frame_size,
    unstack_exception,
)

STACK_BASE = 0x20000000
STACK_TOP = 0x20001000
CODE_BASE = 0x08000000
CODE_SIZE = 0x1000
VECTOR_CODE = 0x08000400
INTERRUPTED_PC = 0x08000200
XPSR_FLAGS = 0xA0000000  # N=1, C=1


def _new_cpu():
    uc = unicorn.Uc(UC_ARCH_ARM, UC_MODE_THUMB)
    uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M4)
    uc.mem_map(STACK_BASE, 0x10000, UC_PROT_ALL)
    uc.mem_map(CODE_BASE, CODE_SIZE, UC_PROT_ALL | unicorn.UC_PROT_EXEC)
    return uc


def _read_u32(uc, address):
    return int.from_bytes(uc.mem_read(address, 4), "little")


def _pc(uc):
    """Unicorn normalises the Thumb bit away on PC reads."""
    return int(uc.reg_read(UC_ARM_REG_PC)) & ~1


class ExceptionUnstackRoundTripTests(unittest.TestCase):
    def _seed_thread_context(self, uc, sp):
        for index, value in enumerate(
            (0x11111111, 0x22222222, 0x33333333, 0x44444444)
        ):
            uc.reg_write(UC_ARM_REG_R0 + index, value)
        uc.reg_write(UC_ARM_REG_R12, 0xCCCCCCCC)
        uc.reg_write(UC_ARM_REG_LR, 0x08000101)
        uc.reg_write(UC_ARM_REG_PC, INTERRUPTED_PC | 1)
        uc.reg_write(UC_ARM_REG_PSP, sp)
        uc.reg_write(UC_ARM_REG_SP, sp)
        uc.reg_write(UC_ARM_REG_MSP, STACK_TOP)
        uc.reg_write(UC_ARM_REG_CPSR, XPSR_FLAGS | 0x01000000)
        for index in range(16):
            uc.reg_write(UC_ARM_REG_S0 + index, 0x5A5A0000 + index)
        uc.reg_write(UC_ARM_REG_FPSCR, 0x00000010)

    def test_ext104_round_trip_restores_every_stacked_field(self):
        uc = _new_cpu()
        original_sp = (STACK_TOP - 0x40) & ~7
        self._seed_thread_context(uc, original_sp)

        entry = build_exception_entry(
            uc,
            irq=50,
            vector=VECTOR_CODE,
            frame_format=FRAME_FORMAT_EXT104,
            exc_return=EXC_RETURN_THREAD_PSP_FPU,
            at_pc=INTERRUPTED_PC,
        )

        self.assertTrue(entry["frame_consistent"])
        self.assertEqual(entry["frame_size"], FRAME_SIZE_EXTENDED)
        frame_base = int(entry["frame_base"], 16)
        self.assertEqual(original_sp - FRAME_SIZE_EXTENDED, frame_base)
        # Hardware state after the push: PSP owns the frame, IPSR names the IRQ.
        self.assertEqual(uc.reg_read(UC_ARM_REG_PSP), frame_base)
        self.assertEqual(uc.reg_read(UC_ARM_REG_IPSR), 50)
        self.assertEqual(_pc(uc), VECTOR_CODE & ~1)
        self.assertEqual(uc.reg_read(UC_ARM_REG_LR), EXC_RETURN_THREAD_PSP_FPU)
        # The frame itself: core words at their architectural offsets.
        self.assertEqual(_read_u32(uc, frame_base + 0x00), 0x11111111)
        self.assertEqual(_read_u32(uc, frame_base + 0x10), 0xCCCCCCCC)
        self.assertEqual(_read_u32(uc, frame_base + FRAME_OFFSET_PC), INTERRUPTED_PC | 1)
        stacked_xpsr = _read_u32(uc, frame_base + FRAME_OFFSET_XPSR)
        self.assertEqual(stacked_xpsr & 0xF8000000, XPSR_FLAGS)
        self.assertEqual(stacked_xpsr & 0x01000000, 0x01000000)  # T bit
        self.assertEqual(stacked_xpsr & 0x000001FF, 0)  # interrupted in Thread mode
        self.assertEqual(_read_u32(uc, frame_base + FRAME_OFFSET_S0), 0x5A5A0000)
        self.assertEqual(_read_u32(uc, frame_base + FRAME_OFFSET_FPSCR), 0x00000010)

        # Scramble the live context so the restore cannot pass by accident.
        for index in range(13):
            uc.reg_write(UC_ARM_REG_R0 + index, 0xDEADBEEF)
        for index in range(16):
            uc.reg_write(UC_ARM_REG_S0 + index, 0)
        uc.reg_write(UC_ARM_REG_FPSCR, 0)

        result = unstack_exception(uc, exc_return=EXC_RETURN_THREAD_PSP_FPU)

        self.assertEqual(result["frame_size"], FRAME_SIZE_EXTENDED)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R0), 0x11111111)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R1), 0x22222222)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R2), 0x33333333)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R3), 0x44444444)
        self.assertEqual(uc.reg_read(UC_ARM_REG_R12), 0xCCCCCCCC)
        self.assertEqual(uc.reg_read(UC_ARM_REG_LR), 0x08000101)
        self.assertEqual(_pc(uc), INTERRUPTED_PC & ~1)
        self.assertEqual(uc.reg_read(UC_ARM_REG_IPSR), 0)
        for index in range(16):
            self.assertEqual(
                uc.reg_read(UC_ARM_REG_S0 + index), 0x5A5A0000 + index
            )
        self.assertEqual(uc.reg_read(UC_ARM_REG_FPSCR), 0x00000010)
        self.assertEqual(uc.reg_read(UC_ARM_REG_APSR) & 0xF8000000, XPSR_FLAGS)
        # SP round-trips exactly to the pre-entry value.
        self.assertEqual(result["post_unstack_sp_value"], original_sp)
        self.assertEqual(uc.reg_read(UC_ARM_REG_SP), original_sp)
        self.assertEqual(uc.reg_read(UC_ARM_REG_PSP), original_sp)

    def test_retto_base_invariant_after_port_stack_reframe(self):
        """Two-frame sequence: the port epilogue reframes below the PSP.

        ``__port_irq_epilogue`` builds a second 104B frame at ``PSP - 104`` and
        ``msr psp``.  Unstacking *that* frame must land the PSP back on the
        injected frame base — the r30b invariant.
        """
        uc = _new_cpu()
        thread_sp = (STACK_TOP - 0x40) & ~7
        self._seed_thread_context(uc, thread_sp)

        entry = build_exception_entry(
            uc,
            irq=50,
            vector=VECTOR_CODE,
            frame_format=FRAME_FORMAT_EXT104,
            exc_return=EXC_RETURN_THREAD_PSP_FPU,
            at_pc=INTERRUPTED_PC,
        )
        injected_frame = int(entry["frame_base"], 16)

        # Port epilogue: new frame at PSP - 104 with PC = __port_switch_from_isr.
        reframed = (injected_frame - FRAME_SIZE_EXTENDED) & ~7
        uc.mem_write(
            reframed + FRAME_OFFSET_PC, (0x08005103).to_bytes(4, "little")
        )
        uc.mem_write(reframed + FRAME_OFFSET_XPSR, (0x01000000).to_bytes(4, "little"))
        uc.reg_write(UC_ARM_REG_PSP, reframed)

        result = unstack_exception(uc, exc_return=EXC_RETURN_THREAD_PSP_FPU)

        self.assertEqual(result["base_value"], reframed)
        self.assertEqual(result["post_unstack_sp_value"], injected_frame)
        self.assertEqual(_pc(uc), 0x08005102)
        self.assertNotEqual(result["post_unstack_sp_value"], reframed)

    def test_frame_size_comes_from_exc_return_bit4(self):
        self.assertEqual(exception_frame_size(EXC_RETURN_THREAD_PSP_FPU), 104)
        self.assertEqual(exception_frame_size(EXC_RETURN_THREAD_PSP), 32)
        self.assertEqual(exception_frame_size(0xFFFFFFF9), 32)

    def test_counterexample_basic_frame_with_extended_unstack_off_by_72(self):
        """Anti-example arm: 32B push consumed as a 104B frame.

        The invariant must be *violated* and the delta must be exactly
        ``104 - 32 == 72``.
        """
        uc = _new_cpu()
        thread_sp = (STACK_TOP - 0x40) & ~7
        self._seed_thread_context(uc, thread_sp)

        entry = build_exception_entry(
            uc,
            irq=50,
            vector=VECTOR_CODE,
            frame_format=FRAME_FORMAT_BASIC32,
            exc_return=EXC_RETURN_THREAD_PSP,
            at_pc=INTERRUPTED_PC,
        )
        self.assertTrue(entry["frame_consistent"])
        frame_base = int(entry["frame_base"], 16)
        self.assertEqual(frame_base, thread_sp - FRAME_SIZE_BASIC)

        # The unstack is told (via the PC-derived EXC_RETURN) that the frame is
        # the extended one — the exact confusion a wrong LR would cause.
        result = unstack_exception(uc, exc_return=EXC_RETURN_THREAD_PSP_FPU)

        self.assertEqual(result["frame_size"], FRAME_SIZE_EXTENDED)
        self.assertNotEqual(result["post_unstack_sp_value"], thread_sp)
        self.assertEqual(
            result["post_unstack_sp_value"] - thread_sp,
            FRAME_SIZE_EXTENDED - FRAME_SIZE_BASIC,
        )
        self.assertEqual(result["post_unstack_sp_value"] - thread_sp, 72)

    def test_exc_return_is_read_from_pc_not_lr(self):
        """F3: the clobbered-LR case.  LR holds a return address, PC holds EXC_RETURN."""
        uc = _new_cpu()
        thread_sp = (STACK_TOP - 0x40) & ~7
        self._seed_thread_context(uc, thread_sp)
        build_exception_entry(
            uc,
            irq=50,
            vector=VECTOR_CODE,
            frame_format=FRAME_FORMAT_EXT104,
            exc_return=EXC_RETURN_THREAD_PSP_FPU,
            at_pc=INTERRUPTED_PC,
        )
        # Unicorn rewrites PC to EXC_RETURN & ~1 when raising the exit; an
        # intervening `bl` (chSchIsPreemptionRequired) leaves a stale LR.
        uc.reg_write(UC_ARM_REG_PC, EXC_RETURN_THREAD_PSP_FPU & ~1)
        uc.reg_write(UC_ARM_REG_LR, 0x0812EDCD)

        result = unstack_exception(uc)

        self.assertEqual(result["frame_size"], FRAME_SIZE_EXTENDED)
        self.assertEqual(result["exc_return"], "0xffffffed")
        self.assertEqual(_pc(uc), INTERRUPTED_PC & ~1)

    def test_double_unstack_survives_two_consecutive_exception_returns(self):
        """F1 by-product: both return branches land in a port function first."""
        uc = _new_cpu()
        thread_sp = (STACK_TOP - 0x40) & ~7
        self._seed_thread_context(uc, thread_sp)
        entry = build_exception_entry(
            uc,
            irq=50,
            vector=VECTOR_CODE,
            frame_format=FRAME_FORMAT_EXT104,
            exc_return=EXC_RETURN_THREAD_PSP_FPU,
            at_pc=INTERRUPTED_PC,
        )
        injected_frame = int(entry["frame_base"], 16)

        reframed = (injected_frame - FRAME_SIZE_EXTENDED) & ~7
        uc.mem_write(reframed + FRAME_OFFSET_PC, (0x08005103).to_bytes(4, "little"))
        uc.mem_write(reframed + FRAME_OFFSET_XPSR, (0x01000000).to_bytes(4, "little"))
        uc.reg_write(UC_ARM_REG_PSP, reframed)
        uc.reg_write(UC_ARM_REG_PC, EXC_RETURN_THREAD_PSP_FPU & ~1)

        first = unstack_exception(uc)
        self.assertEqual(first["stacked_pc"], "0x08005103")

        # Second return: the injected frame is now the one the PSP points at.
        uc.reg_write(UC_ARM_REG_PC, EXC_RETURN_THREAD_PSP_FPU & ~1)
        second = unstack_exception(uc)
        self.assertEqual(second["base_value"], injected_frame)
        self.assertEqual(second["stacked_pc"], f"0x{INTERRUPTED_PC | 1:08x}")
        self.assertEqual(_pc(uc), INTERRUPTED_PC & ~1)
        self.assertEqual(uc.reg_read(UC_ARM_REG_SP), thread_sp)


if __name__ == "__main__":
    unittest.main()
