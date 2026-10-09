"""Thumb ITSTATE helpers: detect stale IT residue and restore architectural semantics.

Background (r28 findings, ``docs/GDB_DEBUG_R28_20260926.md``)
------------------------------------------------------------
The Cortex-M IT state machine lives in ``env->condexec_bits`` and is exposed
through ``xPSR[26:25] + xPSR[15:10]``.  Under Unicorn the state is written back
to ``env`` only when a translation block exits, and for some firmware the
bookkeeping leaves a *non-zero* ITSTATE behind at an address that is outside the
declared IT block.  A concrete instance on ``ardupilot_Pixhawk1_STM32F427``:

    0x0806D3A2  itett mi          ; 4 slots: 0x3A4 / 0x3A6 / 0x3AA / 0x3AC
    0x0806D3B0  b 0x0806D3C4      ; OUTSIDE the block, yet ITSTATE != 0 here
    0x0806D3C4  pop {r4,r5,r6,pc} ; executed as *conditional*, skipped => SP leak

Masking the residue away is architecturally legitimate because a number of
Thumb instructions are **not permitted inside an IT block**: the architecture
classifies them as UNPREDICTABLE there.  QEMU's own translation loop documents
that "perform the cc check and update the IT bits state machine ... is a
permitted CONSTRAINED UNPREDICTABLE choice for those situations" -- i.e. it is
one legal resolution among several.  This module implements the *other* legal
resolution for the narrow class of instructions a compiler never emits inside an
IT block: treat them per their own encoding (these are all unconditional in
their base encoding) instead of applying the stale predicate.

This is deliberately **not** a branch-forcing intervention: no condition and no
branch target is ever synthesised, and no register value is injected.  The
predicate that is dropped cannot legitimately apply to the instruction it sits
on; removing it restores the encoding the compiler emitted.

The classifier is a pure function of the two instruction halfwords so it can be
unit-tested without an emulator and is independent of any address whitelist.
"""

from __future__ import annotations

from typing import Optional

# xPSR bits carrying the IT state: [26:25] plus [15:10].
#
# Derivation (qemu/target/arm/cpu.h): the CPU field is
#   ``uint32_t condexec_bits;  /* IT bits. cpsr[15:10,26:25]. */``
# and ``xpsr_read()`` exposes it as
#   ``(condexec_bits & 3) << 25 | (condexec_bits & 0xfc) << 8``
# i.e. condexec_bits[1:0] -> xPSR[26:25] and condexec_bits[7:2] -> xPSR[15:10].
# So "ITSTATE != 0" is exactly "xPSR & 0x0600FC00" != 0.  (r28's probe used
# 0x0400FC00, which covers bit 26 but drops bit 25 -- one bit of condexec_bits
# was therefore invisible to it.)
IT_STATE_MASK = 0x0600FC00

# Human-readable labels for the instruction classes the guard will act on.
KIND_B = "B"
KIND_BW = "B.W"
KIND_BX = "BX"
KIND_BLX_REG = "BLX"
KIND_BLX_IMM = "BLX.W"
KIND_POP_PC = "POP{pc}"
KIND_POPW_PC = "POP.W{pc}"


def thumb_it_unpredictable_kind(hw1: int, hw2: Optional[int] = None) -> Optional[str]:
    """Classify an instruction that is UNPREDICTABLE inside an IT block.

    Returns a short label when the instruction belongs to the *unconditional*
    family that must not appear in an IT block (``B`` / ``B.W`` / ``BX`` /
    ``BLX`` / ``POP{...,pc}``), otherwise ``None``.

    ``hw1`` is the first halfword; ``hw2`` is the second halfword for 32-bit
    encodings (``None`` when the caller only has the first halfword, in which
    case 32-bit candidates report no match).

    Deliberately excluded:

    * conditional branches (T2 ``B``, T3 ``B.W``) -- these are *permitted* in an
      IT block, so a non-zero ITSTATE on them is meaningful;
    * ``BL`` (immediate, the function-call form) -- syntactically adjacent to
      ``BLX <imm>`` but a call rather than a branch, and no measurement has ever
      shown stale IT residue on a call site; keeping it out bounds the reach of
      the guard to the classes with evidence behind them;
    * ``CBZ`` / ``CBNZ`` / ``TBB`` / ``TBH`` -- not part of the specified set.
    """
    hw1 &= 0xFFFF

    # --- 16-bit encodings -------------------------------------------------
    # T1 B (unconditional): 11100 imm11
    if (hw1 & 0xF800) == 0xE000:
        return KIND_B
    # T1 BX:  0100 0111 0 Rm 000   /  T1 BLX (register): 0100 0111 1 Rm 000
    # bits[2:0] are the register field and must be zero for both.
    if (hw1 & 0xFF87) == 0x4700:
        return KIND_BX
    if (hw1 & 0xFF87) == 0x4780:
        return KIND_BLX_REG
    # T1 POP{...,pc}: 1011 110 P reglist, P=1 means PC is in the list.
    if (hw1 & 0xFF00) == 0xBD00:
        return KIND_POP_PC

    # --- 32-bit encodings (need the second halfword) ----------------------
    if hw2 is None:
        return None
    hw2 &= 0xFFFF

    # T2 POP{...,pc}: LDM.W with Rn=SP; bit15 (P) set means PC is in the list.
    # Checked before the 11110-prefix gate because its first halfword is 0xE8BD.
    if hw1 == 0xE8BD and (hw2 & 0x8000):
        return KIND_POPW_PC

    if (hw1 & 0xF800) != 0xF000:
        return None

    # For 11110-prefixed encodings hw2 bits {15,14,12} select the family:
    #   10 x 0 -> B.W T3 (conditional, permitted in IT)
    #   10 x 1 -> B.W T4 (unconditional)
    #   11 x 0 -> BLX <imm> (T2)
    #   11 x 1 -> BL      (the call form, deliberately out of scope)
    family = hw2 & 0xD000
    if family == 0x8000:                   # B.W T3 (conditional) -> permitted
        return None
    if family == 0x9000:                   # B.W T4 (unconditional)
        return KIND_BW
    if family == 0xC000:                   # BLX <imm>
        return KIND_BLX_IMM
    if family == 0xD000:                   # BL -> excluded
        return None

    return None
