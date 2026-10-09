#!/usr/bin/env python3
"""Compatibility name for the retired ISR explorer implementation.

The former ``ISRExplorerSmart`` implementation was an unreferenced prototype
with a separate vector parser, fixed address assumptions, and weaker context
and stack handling than :class:`ISRExplorer`.  Keeping a small compatibility
subclass avoids breaking external scripts that still import the old name while
ensuring that all ISR exploration uses the maintained implementation.
"""

from __future__ import annotations

from .isr_explorer import ISRExplorer


class ISRExplorerSmart(ISRExplorer):
    """Backward-compatible alias for :class:`ISRExplorer`.

    The constructor and inherited public methods remain compatible with the
    old two-required-argument form (``uc`` and ``static_bbs``), while callers
    also gain the maintained explorer's optional MMIO and validation hooks.
    """


__all__ = ["ISRExplorerSmart"]
