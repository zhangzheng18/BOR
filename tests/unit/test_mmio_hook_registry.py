#!/usr/bin/env python3
"""Contracts for the single-primary-MMIO-hook registry and replay bridge."""

from __future__ import annotations

import unittest

from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler
from lsgemu.mmio_handler.mmio_hook_registry import (
    get_primary_mmio_handler,
    register_primary_mmio_handler,
    unregister_primary_mmio_handler,
)


class _FakeUnicorn:
    def hook_add(self, *_args, **_kwargs):
        raise AssertionError("replay overlay installed a second Unicorn hook")


class _PrimaryHandler:
    def __init__(self) -> None:
        self.overlays = []

    def push_mmio_overlay(self, overlay) -> None:
        self.overlays.append(overlay)

    def pop_mmio_overlay(self, overlay) -> None:
        self.overlays.remove(overlay)


class MMIOHookRegistryTests(unittest.TestCase):
    def test_enhanced_handler_bridges_without_installing_second_hook(self):
        uc = _FakeUnicorn()
        primary = _PrimaryHandler()
        overlay = EnhancedMMIOHandler(uc)
        register_primary_mmio_handler(uc, primary)
        try:
            overlay.start_hooking()
            self.assertTrue(overlay._bridged_to_primary)
            self.assertEqual([overlay], primary.overlays)
            self.assertEqual([], overlay.read_hooks)
            self.assertEqual([], overlay.write_hooks)

            overlay.stop_hooking()
            self.assertFalse(overlay._bridged_to_primary)
            self.assertEqual([], primary.overlays)
        finally:
            unregister_primary_mmio_handler(uc, primary)

    def test_non_weakref_unicorn_fallback_is_identity_checked_and_unregistered(self):
        uc = object()
        handler = object()
        other_handler = object()
        register_primary_mmio_handler(uc, handler)
        try:
            self.assertIs(handler, get_primary_mmio_handler(uc))
            unregister_primary_mmio_handler(uc, other_handler)
            self.assertIs(handler, get_primary_mmio_handler(uc))
        finally:
            unregister_primary_mmio_handler(uc, handler)
        self.assertIsNone(get_primary_mmio_handler(uc))


if __name__ == "__main__":
    unittest.main()
