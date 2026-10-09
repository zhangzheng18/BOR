#!/usr/bin/env python3
"""
Local disassembler subset used by LSGEmu.
"""

__all__ = [
    "GhidraDisassemblerWithBB",
    "UniversalDisassembler",
    "Instruction",
    "BasicBlock",
]


def __getattr__(name):
    if name in {"GhidraDisassemblerWithBB", "Instruction", "BasicBlock"}:
        from .ghidra_disassembler_with_bb import GhidraDisassemblerWithBB, Instruction, BasicBlock

        exports = {
            "GhidraDisassemblerWithBB": GhidraDisassemblerWithBB,
            "Instruction": Instruction,
            "BasicBlock": BasicBlock,
        }
        return exports[name]

    if name == "UniversalDisassembler":
        from .universal_disassembler import UniversalDisassembler

        return UniversalDisassembler

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
