"""Regression tests for shared Unicorn hook ownership and deferred deletion."""

from __future__ import annotations

from collections import Counter
import threading
import unittest

from unicorn import UC_HOOK_CODE

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.hook_lifecycle import (
    get_hook_owner,
    managed_emu_start,
    managed_hook_add,
    managed_hook_del,
    register_hook_owner,
    unregister_hook_owner,
)


class _FakeUnicorn:
    def __init__(self) -> None:
        self.next_handle = 1
        self.added = []
        self.deleted = []
        self.stop_calls = 0

    def hook_add(self, *args, **kwargs):
        handle = self.next_handle
        self.next_handle += 1
        self.added.append((handle, args, kwargs))
        return handle

    def hook_del(self, handle):
        self.deleted.append(handle)

    def emu_start(self, *args, **kwargs):
        return (args, kwargs)

    def emu_stop(self):
        self.stop_calls += 1


def _owner_for(uc: _FakeUnicorn) -> IntelligentEmulator:
    owner = IntelligentEmulator.__new__(IntelligentEmulator)
    owner.uc = uc
    owner._owned_hooks = []
    owner._owned_hook_records = {}
    owner._pending_owned_hook_removals = []
    owner._native_emulation_depth = 0
    owner._native_emulation_thread_id = None
    owner._close_requested = False
    owner._close_completed = False
    owner._closed_uc = None
    owner.hook_lifecycle_stats = Counter()
    return owner


class HookLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uc = _FakeUnicorn()
        self.owner = _owner_for(self.uc)
        register_hook_owner(self.uc, self.owner)

    def tearDown(self) -> None:
        unregister_hook_owner(self.uc, self.owner)

    def test_components_route_add_and_delete_through_owner(self):
        handle = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)
        self.assertEqual([handle], self.owner._owned_hooks)
        self.assertIs(self.owner, get_hook_owner(self.uc))

        managed_hook_del(self.uc, handle)

        self.assertEqual([handle], self.uc.deleted)
        self.assertEqual([], self.owner._owned_hooks)

    def test_delete_is_deferred_until_native_emulation_returns(self):
        handle = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)

        with self.owner._native_emulation_scope():
            managed_hook_del(self.uc, handle)
            self.assertEqual([], self.uc.deleted)
            self.assertEqual([handle], self.owner._pending_owned_hook_removals)

        self.assertEqual([handle], self.uc.deleted)
        self.assertEqual([], self.owner._pending_owned_hook_removals)

    def test_exception_also_drains_deferred_deletion(self):
        handle = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with self.owner._native_emulation_scope():
                self.owner._remove_owned_hook(handle)
                raise RuntimeError("stop")

        self.assertEqual([handle], self.uc.deleted)

    def test_emulation_calls_share_the_same_lifecycle_scope(self):
        handle = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)

        result = managed_emu_start(self.uc, 0x1000, 0, count=1)

        self.assertEqual(((0x1000, 0), {"count": 1}), result)
        self.assertEqual(0, self.owner._native_emulation_depth)
        self.owner._remove_owned_hook(handle)
        self.assertEqual([handle], self.uc.deleted)

    def test_clear_unregisters_owner_and_deletes_remaining_hooks(self):
        first = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)
        second = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)

        self.owner._clear_owned_hooks(self.uc)

        self.assertEqual([second, first], self.uc.deleted)
        self.assertIsNone(get_hook_owner(self.uc))

    def test_close_requested_inside_native_scope_finishes_after_unwind(self):
        handle = managed_hook_add(self.uc, UC_HOOK_CODE, lambda *_: None)

        with self.owner._native_emulation_scope():
            self.owner.close()
            self.assertIs(self.uc, self.owner.uc)
            self.assertEqual([handle], self.owner._pending_owned_hook_removals)
            self.assertIs(self.owner, get_hook_owner(self.uc))

        self.assertIsNone(self.owner.uc)
        self.assertIs(self.uc, self.owner._closed_uc)
        self.assertEqual([handle], self.uc.deleted)
        self.assertIsNone(get_hook_owner(self.uc))
        self.assertTrue(self.owner._close_completed)

        # Cleanup is idempotent after the deferred close has completed.
        self.owner.close()
        self.assertEqual([handle], self.uc.deleted)
        self.assertEqual(1, self.owner.hook_lifecycle_stats["close_completed"])

    def test_reentrant_native_scope_is_rejected_by_default(self):
        with self.owner._native_emulation_scope():
            with self.assertRaisesRegex(RuntimeError, "reentrant emulation"):
                with self.owner._native_emulation_scope():
                    pass

        self.assertEqual(
            1,
            self.owner.hook_lifecycle_stats["reentrant_emu_start_rejected"],
        )

    def test_cross_thread_native_scope_is_rejected(self):
        entered = threading.Event()
        release = threading.Event()
        result = []

        def hold_scope():
            with self.owner._native_emulation_scope():
                entered.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=hold_scope)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        try:
            with self.assertRaisesRegex(RuntimeError, "concurrent emulation"):
                with self.owner._native_emulation_scope():
                    result.append(True)
        finally:
            release.set()
            thread.join(timeout=2)

        self.assertEqual([], result)
        self.assertEqual(
            1,
            self.owner.hook_lifecycle_stats["cross_thread_emu_start_rejected"],
        )

    def test_cross_thread_close_defers_native_stop_to_emulation_owner(self):
        entered = threading.Event()
        release = threading.Event()

        def hold_scope():
            with self.owner._native_emulation_scope():
                entered.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=hold_scope)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        try:
            self.owner.close()
            self.assertEqual(0, self.uc.stop_calls)
            self.assertFalse(self.owner._close_completed)
            self.assertEqual(
                1,
                self.owner.hook_lifecycle_stats["cross_thread_close_deferred"],
            )
        finally:
            release.set()
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(self.owner._close_completed)
        self.assertIsNone(self.owner.uc)


if __name__ == "__main__":
    unittest.main()
