"""Regression tests for safe-point memory mapping and instruction retry."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import threading
import unittest

from unicorn import (
    UC_ARCH_ARM,
    UC_HOOK_MEM_WRITE_UNMAPPED,
    UC_MODE_THUMB,
    UC_PROT_ALL,
    UC_PROT_READ,
    Uc,
)
from unicorn.arm_const import (
    UC_ARM_REG_PC,
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_R3,
)

from lsgemu.analysis.intelligent_emulator import (
    IntelligentEmulator,
    _DeferredMemoryMapping,
)


class _FakeUnicorn:
    def __init__(self) -> None:
        self.regions = []
        self.map_calls = []
        self.map_depths = []
        self.starts = []
        self.stop_calls = 0
        self.current_pc = 0x1001
        self.owner = None
        self.first_start = None
        self.fail_mapping = False
        self.registers = {}
        self.next_hook = 1
        self.hooks = {}
        self.deleted_hooks = []

    def mem_regions(self):
        return list(self.regions)

    def mem_map(self, address, size, perms=UC_PROT_ALL):
        depth = int(getattr(self.owner, "_native_emulation_depth", -1))
        self.map_depths.append(depth)
        self.map_calls.append((int(address), int(size), int(perms)))
        if self.fail_mapping:
            raise RuntimeError("injected mapping failure")
        self.regions.append(
            (int(address), int(address) + int(size) - 1, int(perms))
        )

    def mem_unmap(self, address, size):
        start = int(address)
        end = start + int(size)
        self.regions = [
            region
            for region in self.regions
            if int(region[1]) + 1 <= start or int(region[0]) >= end
        ]

    def emu_start(self, begin, until, timeout=0, count=0):
        self.current_pc = int(begin)
        self.starts.append((int(begin), int(until), int(timeout), int(count)))
        callback = self.first_start
        self.first_start = None
        if callback is not None:
            callback()
        return "completed"

    def emu_stop(self):
        self.stop_calls += 1

    def hook_add(self, *args, **kwargs):
        handle = self.next_hook
        self.next_hook += 1
        self.hooks[handle] = (args, kwargs)
        return handle

    def hook_del(self, handle):
        self.deleted_hooks.append(handle)
        self.hooks.pop(handle, None)

    def reg_read(self, register):
        if register == UC_ARM_REG_PC:
            return self.current_pc
        return self.registers.get(register, 0)


def _bare_owner(uc) -> IntelligentEmulator:
    owner = IntelligentEmulator.__new__(IntelligentEmulator)
    owner.uc = uc
    owner.execution_thumb = True
    owner.mapped_ranges = []
    owner._owned_hooks = []
    owner._owned_hook_records = {}
    owner._pending_owned_hook_removals = []
    owner._native_emulation_depth = 0
    owner._native_emulation_thread_id = None
    owner._native_emulation_state_lock = threading.RLock()
    owner._close_requested = False
    owner._close_completed = False
    owner._closed_uc = None
    owner.hook_lifecycle_stats = Counter()
    owner._pending_memory_map_pages = {}
    owner._pending_memory_map_origins = set()
    owner._pending_memory_map_external_addresses = set()
    owner._pending_memory_map_write_replays = []
    owner._pending_memory_map_retries_block_callback = False
    owner._deferred_retry_skip_block_pc = None
    owner._deferred_retry_reenter_block_pc = None
    owner._deferred_write_retry_hooks = []
    owner.memory_mapping_stats = Counter()
    owner.last_memory_mapping = {
        "status": "idle",
        "requested_pages": [],
        "mapped_pages": [],
        "error": None,
    }
    owner.external_memory_input_addresses = set()
    owner.mapped_write_hook_enabled = True
    if hasattr(uc, "owner"):
        uc.owner = owner
    return owner


class MemoryMappingLifecycleTests(unittest.TestCase):
    def test_production_native_topology_calls_are_confined_to_safe_boundaries(self):
        package_root = Path(__file__).resolve().parents[2] / "lsgemu"
        expected = {
            "mem_map": {
                ("analysis/intelligent_emulator.py", "_map_memory_pages_now"),
                ("analysis/intelligent_emulator.py", "setup_memory"),
                ("analysis/intelligent_emulator.py", "load_firmware"),
                ("hook_lifecycle.py", "managed_mem_map"),
            },
            "mem_unmap": {
                ("analysis/intelligent_emulator.py", "_managed_mem_unmap"),
            },
            "emu_start": {
                ("analysis/intelligent_emulator.py", "_managed_emu_start"),
                ("hook_lifecycle.py", "managed_emu_start"),
            },
        }
        observed = {name: set() for name in expected}
        for source_path in package_root.rglob("*.py"):
            if (
                source_path.name.startswith("test_")
                or source_path.name.startswith("validate_")
            ):
                continue
            relative_path = source_path.relative_to(package_root).as_posix()
            tree = ast.parse(source_path.read_text(encoding="utf-8"))

            class Visitor(ast.NodeVisitor):
                def __init__(self):
                    self.function_stack = []

                def visit_FunctionDef(self, node):
                    self.function_stack.append(node.name)
                    self.generic_visit(node)
                    self.function_stack.pop()

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_Call(self, node):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in observed
                    ):
                        observed[node.func.attr].add(
                            (
                                relative_path,
                                self.function_stack[-1]
                                if self.function_stack
                                else "<module>",
                            )
                        )
                    self.generic_visit(node)

            Visitor().visit(tree)
        for operation, allowed_calls in expected.items():
            self.assertEqual(
                allowed_calls,
                observed[operation],
                operation,
            )

    def test_depth_zero_mapping_is_immediate(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)

        self.assertTrue(owner._ensure_memory_mapped(0x2004, 4))

        self.assertEqual([(0x2000, 0x1000, UC_PROT_ALL)], uc.map_calls)
        self.assertEqual([0], uc.map_depths)
        self.assertEqual({}, owner._pending_memory_map_pages)

    def test_native_callback_maps_only_after_emu_start_unwinds(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)

        def request_missing_page():
            owner._ensure_memory_mapped(0x3004, 4, origin="memory")

        uc.first_start = request_missing_page
        result = owner._managed_emu_start(0x1001, 0, timeout=1000, count=4)

        self.assertEqual("completed", result)
        self.assertEqual([0], uc.map_depths)
        self.assertEqual(1, uc.stop_calls)
        self.assertEqual([0x1001, 0x1001], [item[0] for item in uc.starts])
        self.assertEqual(1, owner.memory_mapping_stats["retries"])
        self.assertEqual({}, owner._pending_memory_map_pages)
        self.assertEqual(set(), owner._pending_memory_map_origins)

    def test_contiguous_pages_are_mapped_as_one_run(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)

        self.assertTrue(
            owner._map_memory_pages_now({0x4000: 0, 0x5000: 0, 0x6000: 0})
        )

        self.assertEqual([(0x4000, 0x3000, UC_PROT_ALL)], uc.map_calls)
        self.assertEqual(1, owner.memory_mapping_stats["map_runs"])
        self.assertEqual(3, owner.memory_mapping_stats["pages_mapped"])

    def test_fault_path_bypasses_a_stale_python_mapping_index(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        owner.mapped_ranges = [(0xB000, 0xC000)]

        self.assertTrue(
            owner._ensure_memory_mapped(
                0xB004,
                4,
                verify_native=True,
            )
        )

        self.assertEqual([(0xB000, 0x1000, UC_PROT_ALL)], uc.map_calls)

    def test_unmap_is_rejected_during_native_execution(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        uc.mem_map(0xD000, 0x1000)
        owner.mapped_ranges = [(0xD000, 0xE000)]
        owner._native_emulation_depth = 1

        with self.assertRaisesRegex(RuntimeError, "cannot unmap"):
            owner._managed_mem_unmap(0xD000, 0x1000)

        self.assertTrue(owner._native_range_is_mapped(0xD000, 1))
        self.assertEqual([(0xD000, 0xE000)], owner.mapped_ranges)

    def test_missing_page_stack_classifies_block_callback_without_hot_path_wrapper(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        owner._native_emulation_depth = 1
        owner._native_emulation_thread_id = threading.get_ident()

        def bb_hook():
            owner._ensure_memory_mapped(0xC004, 4)

        with self.assertRaises(_DeferredMemoryMapping):
            bb_hook()

        self.assertEqual({"block"}, owner._pending_memory_map_origins)
        self.assertTrue(owner._pending_memory_map_retries_block_callback)
        owner._discard_pending_memory_maps("test_cleanup")

    def test_default_permissions_dominate_an_explicit_subset(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        owner._native_emulation_depth = 1
        owner._native_emulation_thread_id = threading.get_ident()
        owner._pending_memory_map_pages[0x7000] = UC_PROT_READ

        with self.assertRaises(_DeferredMemoryMapping):
            owner._queue_deferred_memory_mapping({0x7000}, permissions=None)

        self.assertEqual(0, owner._pending_memory_map_pages[0x7000])

    def test_mapping_failure_is_explicit_and_does_not_commit_external_state(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        uc.fail_mapping = True
        owner._pending_memory_map_pages = {0x8000: 0}
        owner._pending_memory_map_origins = {"invalid_memory"}
        owner._pending_memory_map_external_addresses = {0x8004}

        self.assertFalse(owner._drain_pending_memory_maps())

        self.assertEqual(set(), owner.external_memory_input_addresses)
        self.assertEqual("failed", owner.last_memory_mapping["status"])
        self.assertEqual(["invalid_memory"], owner.last_memory_mapping["origins"])

    def test_retry_block_marker_is_one_shot_and_not_armed_for_block_origin(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)

        owner._set_deferred_retry_block_skip(0x9001, False)
        self.assertTrue(owner._consume_deferred_retry_block_skip(0x9000))
        self.assertFalse(owner._consume_deferred_retry_block_skip(0x9000))

        owner._set_deferred_retry_block_skip(0xA001, True)
        self.assertFalse(owner._consume_deferred_retry_block_skip(0xA000))
        self.assertTrue(owner._consume_deferred_retry_block_reentry(0xA000))
        self.assertFalse(owner._consume_deferred_retry_block_reentry(0xA000))

    def test_stream_cursor_and_snapshot_do_not_advance_before_mapping(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        uc.registers.update(
            {
                UC_ARM_REG_R0: 0,
                UC_ARM_REG_R1: 0x20002000,
                UC_ARM_REG_R2: 8,
                UC_ARM_REG_R3: 0,
            }
        )
        snapshots = []
        payload_calls = []
        owner._save_summary_entry_snapshot_if_needed = (
            lambda address, kind: snapshots.append((address, kind))
        )
        owner._stream_input_payload = (
            lambda key, size: payload_calls.append((key, size)) or b"A" * size
        )
        owner._ensure_memory_mapped = lambda _address, _size: (_ for _ in ()).throw(
            _DeferredMemoryMapping((0x20002000,))
        )

        with self.assertRaises(_DeferredMemoryMapping):
            owner._apply_stream_input_summary(
                uc,
                {"kind": "stream_read", "arg_style": "posix", "symbol": "read"},
                0x08001000,
            )

        self.assertEqual([], snapshots)
        self.assertEqual([], payload_calls)

    def test_disabled_global_write_hook_gets_one_shot_model_replay(self):
        uc = _FakeUnicorn()
        owner = _bare_owner(uc)
        owner.mapped_write_hook_enabled = False
        observed = []
        owner.mem_write_hook = lambda *args: observed.append(args[2:5])

        self.assertTrue(
            owner._install_deferred_write_retry_hooks(
                [
                    {
                        "pc": 0x1000,
                        "address": 0x40001000,
                        "size": 4,
                        "value": 0xA5,
                    }
                ]
            )
        )
        self.assertEqual(1, len(uc.hooks))
        handle, (args, _kwargs) = next(iter(uc.hooks.items()))
        callback = args[1]
        uc.current_pc = 0x1001
        callback(uc, 0, 0x40001000, 4, 0xA5, None)

        self.assertEqual([(0x40001000, 4, 0xA5)], observed)
        self.assertIn(handle, uc.deleted_hooks)
        self.assertEqual(1, owner.memory_mapping_stats["write_replays_applied"])

    def test_real_thumb_store_is_retried_and_committed(self):
        code = 0x1000
        target = 0x3000
        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        uc.mem_map(code, 0x1000, UC_PROT_ALL)
        # str r0, [r1]; movs r2, #7
        uc.mem_write(code, b"\x08\x60\x07\x22")
        uc.reg_write(UC_ARM_REG_R0, 0x12345678)
        uc.reg_write(UC_ARM_REG_R1, target)
        owner = _bare_owner(uc)
        owner.mapped_ranges = [(code, code + 0x1000)]
        map_depths = []
        original_map_now = owner._map_memory_pages_now

        def observed_map_now(pages):
            map_depths.append(owner._native_emulation_depth)
            return original_map_now(pages)

        owner._map_memory_pages_now = observed_map_now

        def on_unmapped(_uc, _access, address, size, _value, _user_data):
            return owner._ensure_memory_mapped(
                address,
                size,
                origin="invalid_memory",
            )

        hook = uc.hook_add(UC_HOOK_MEM_WRITE_UNMAPPED, on_unmapped)
        try:
            owner._managed_emu_start(code | 1, code + 4, count=2)
        finally:
            uc.hook_del(hook)

        self.assertEqual(b"\x78\x56\x34\x12", bytes(uc.mem_read(target, 4)))
        self.assertEqual(7, uc.reg_read(UC_ARM_REG_R2))
        self.assertEqual([0], map_depths)
        self.assertEqual(1, owner.memory_mapping_stats["retries"])

    def test_real_thumb_mmio_store_uses_one_shot_hook_when_global_hook_is_off(self):
        code = 0x1000
        target = 0x40001000
        uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        uc.mem_map(code, 0x1000, UC_PROT_ALL)
        # str r0, [r1]; movs r2, #9
        uc.mem_write(code, b"\x08\x60\x09\x22")
        uc.reg_write(UC_ARM_REG_R0, 0xCAFEBABE)
        uc.reg_write(UC_ARM_REG_R1, target)
        owner = _bare_owner(uc)
        owner.mapped_ranges = [(code, code + 0x1000)]
        owner.mapped_write_hook_enabled = False
        observed = []
        owner.mem_write_hook = lambda *args: observed.append(
            (int(args[2]), int(args[3]), int(args[4]))
        )

        def on_unmapped(_uc, _access, address, size, value, _user_data):
            try:
                return owner._ensure_memory_mapped(
                    address,
                    size,
                    origin="invalid_memory",
                    verify_native=True,
                )
            except _DeferredMemoryMapping:
                owner._mark_pending_unmapped_write_replay(
                    pc=int(uc.reg_read(UC_ARM_REG_PC)),
                    address=address,
                    size=size,
                    value=value,
                )
                raise

        hook = uc.hook_add(UC_HOOK_MEM_WRITE_UNMAPPED, on_unmapped)
        try:
            owner._managed_emu_start(code | 1, code + 4, count=2)
        finally:
            uc.hook_del(hook)

        self.assertEqual(b"\xbe\xba\xfe\xca", bytes(uc.mem_read(target, 4)))
        self.assertEqual(9, uc.reg_read(UC_ARM_REG_R2))
        self.assertEqual([(target, 4, 0xCAFEBABE)], observed)
        self.assertEqual(1, owner.memory_mapping_stats["write_replays_applied"])


if __name__ == "__main__":
    unittest.main()
