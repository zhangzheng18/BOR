#!/usr/bin/env python3
"""Contracts for deduplicated and integrity-checked execution snapshots."""

from __future__ import annotations

import gc
import os
import pickle
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager
from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.snapshot_memory import (
    SnapshotMetadataStore,
    PagedMemory,
    SnapshotCaptureError,
    SnapshotBlobStore,
    SnapshotPageStore,
    SnapshotStateBlob,
)
from lsgemu.analysis.unique_bb_snapshot_manager import UniqueBBSnapshotManager
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.causal_context import CausalExecutionContext
from lsgemu.prepared_firmware import PreparedFirmware


class DummyUC:
    def __init__(self, memory: bytes, *, fail_reads: bool = False):
        self.memory = bytes(memory)
        self.fail_reads = fail_reads
        self.registers = {}
        self.memory_writes = []

    def reg_read(self, register):
        return self.registers.get(register, 0)

    def reg_write(self, register, value):
        self.registers[register] = int(value)

    def mem_read(self, _address, size):
        if self.fail_reads:
            raise RuntimeError("injected memory read failure")
        return self.memory[: int(size)]

    def mem_write(self, address, data):
        self.memory_writes.append((int(address), bytes(data)))


class SnapshotMemoryContractTests(unittest.TestCase):
    def test_page_disk_backing_can_be_disabled_independently_of_state_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LSGEMU_SNAPSHOT_DISK_BACKING": "1",
                    "LSGEMU_SNAPSHOT_PAGE_DISK_BACKING": "0",
                    "LSGEMU_SNAPSHOT_STORAGE_DIR": directory,
                },
            ):
                page_store = SnapshotPageStore(page_size=256)
                blob_store = SnapshotBlobStore.from_environment()

                self.assertFalse(page_store.get_statistics()["disk_backed"])
                self.assertIsNotNone(blob_store)

    def test_disk_index_reuses_page_after_weak_page_object_is_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotPageStore(page_size=256, storage_dir=directory)
            raw = bytes(range(256))
            first = store.intern_region(raw)
            page_ref = weakref.ref(next(first.iter_page_objects()))
            initial_size = store.get_statistics()["disk_store"]["file_size"]

            del first
            gc.collect()
            self.assertIsNone(page_ref())

            second = store.intern_region(raw)
            stats = store.get_statistics()
            self.assertEqual(raw, bytes(second))
            self.assertEqual(initial_size, stats["disk_store"]["file_size"])
            self.assertEqual(1, stats["indexed_unique_pages"])
            self.assertEqual(1, stats["disk_index_reuses"])

    def test_disk_backed_paged_memory_outlives_temporary_page_store(self):
        raw = bytes(range(256)) * 2
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LSGEMU_SNAPSHOT_DISK_BACKING": "1",
                    "LSGEMU_SNAPSHOT_STORAGE_DIR": directory,
                },
            ):
                memory = PagedMemory.from_bytes(raw, page_size=256)
                gc.collect()
                self.assertEqual(raw, bytes(memory))

    def test_blob_store_environment_registry_does_not_extend_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LSGEMU_SNAPSHOT_DISK_BACKING": "1",
                    "LSGEMU_SNAPSHOT_STORAGE_DIR": directory,
                },
            ):
                store = SnapshotBlobStore.from_environment()
                self.assertIsNotNone(store)
                store_ref = weakref.ref(store)
                del store
                gc.collect()
                self.assertIsNone(store_ref())

    def test_disk_backed_pages_preserve_bytes_and_share_page_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotPageStore(page_size=256, storage_dir=directory)
            raw = bytes(range(256)) * 3
            first = store.intern_region(raw)
            second = store.intern_region(raw)

            self.assertEqual(raw, bytes(first))
            self.assertEqual(raw[250:270], first[250:270])
            self.assertEqual(first, second)
            stats = store.get_statistics()
            self.assertTrue(stats["disk_backed"])
            self.assertEqual(1, stats["live_unique_pages"])
            self.assertEqual(256, stats["live_unique_bytes"])
            self.assertEqual(256, stats["disk_store"]["file_size"])

    def test_disk_backed_state_blob_round_trips_without_retaining_payload_bytes(self):
        payload = {
            "events": [{"pc": 0x1000 + index, "value": index} for index in range(256)]
        }
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotBlobStore(directory)
            blob = SnapshotStateBlob.from_mapping(
                payload,
                min_raw_bytes=0,
                compression_level=1,
                store=store,
            )

            self.assertIsInstance(blob, SnapshotStateBlob)
            self.assertTrue(blob.is_disk_backed)
            self.assertIsNone(blob._payload)
            self.assertEqual(payload, dict(blob))
            blob.release_materialized()
            self.assertFalse(blob.is_materialized)
            self.assertGreater(store.get_statistics()["file_size"], 0)

    def test_state_blob_write_failure_falls_back_to_exact_memory_payload(self):
        payload = {
            "events": [{"pc": 0x1000 + index, "value": index} for index in range(256)]
        }
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotBlobStore(directory)
            with patch.object(store, "append", side_effect=OSError("disk full")):
                blob = SnapshotStateBlob.from_mapping(
                    payload,
                    min_raw_bytes=0,
                    compression_level=1,
                    store=store,
                )

            self.assertIsInstance(blob, SnapshotStateBlob)
            self.assertFalse(blob.is_disk_backed)
            self.assertEqual(payload, dict(blob))

    def test_blob_store_deduplicates_identical_serialized_payloads(self):
        payload = {"events": [{"pc": 0x1000, "value": 7}] * 128}
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotBlobStore(directory)
            first = SnapshotStateBlob.from_mapping(
                payload,
                min_raw_bytes=0,
                compression_level=1,
                store=store,
            )
            second = SnapshotStateBlob.from_mapping(
                payload,
                min_raw_bytes=0,
                compression_level=1,
                store=store,
            )

            stats = store.get_statistics()
            self.assertEqual(first.stored_size, second.stored_size)
            self.assertEqual(first._file_offset, second._file_offset)
            self.assertEqual(first._file_size, second._file_size)
            self.assertEqual(first.stored_size, stats["file_size"])
            self.assertEqual(1, stats["deduplicated_reuses"])
            self.assertEqual(1, stats["indexed_payloads"])
            self.assertEqual(payload, dict(second))

    def test_auto_storage_directory_changes_between_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LSGEMU_SNAPSHOT_DISK_BACKING": "1",
                },
                clear=False,
            ):
                for key in (
                    "LSGEMU_SNAPSHOT_STORAGE_DIR",
                    "LSGEMU_SNAPSHOT_STORAGE_AUTO",
                    "LSGEMU_SNAPSHOT_STORAGE_RUN_KEY",
                ):
                    os.environ.pop(key, None)
                from lsgemu.analysis.snapshot_memory import (
                    configure_snapshot_storage_for_run,
                )

                first = configure_snapshot_storage_for_run(directory, run_id="one")
                second = configure_snapshot_storage_for_run(directory, run_id="two")

            self.assertNotEqual(first, second)
            self.assertTrue(str(first).endswith("/one"))
            self.assertTrue(str(second).endswith("/two"))

    def test_page_write_failure_falls_back_to_exact_memory_page(self):
        raw = bytes(range(256)) * 2
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotPageStore(page_size=256, storage_dir=directory)
            with patch.object(
                store._file_store,
                "append",
                side_effect=OSError("disk full"),
            ):
                memory = store.intern_region(raw)

            self.assertEqual(raw, bytes(memory))
            self.assertFalse(next(memory.iter_page_objects()).is_disk_backed)
            self.assertEqual(
                1,
                store.get_statistics()["disk_fallback_pages"],
            )

    def test_branch_snapshot_uses_shared_disk_blob_store(self):
        raw = b"F" * 512
        expected = {
            "mmio_handler": {
                "events": [{"pc": 0x1000, "value": index} for index in range(512)]
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotBlobStore(directory)
            manager = BranchSnapshotManager(blob_store=store)
            manager.set_memory_regions([(0x20000000, len(raw))])
            manager.set_external_state_provider(
                lambda: {"external_model_state": expected}
            )
            snapshot = manager.save_snapshot(
                DummyUC(raw),
                0x1000,
                0x1100,
                0x1004,
                "BNE",
            )

            self.assertTrue(snapshot.external_model_state.is_disk_backed)
            self.assertEqual(expected, dict(snapshot.external_model_state))
            self.assertEqual(
                1,
                manager.get_statistics()["blob_store"]["read_operations"],
            )

    def test_nested_replay_state_round_trips_without_retaining_object_graph(self):
        payload = {
            "mmio_handler": {
                "mmio_states": {
                    address: {
                        "value": address ^ 0x55AA,
                        "history": [address & 0xFF] * 32,
                    }
                    for address in range(0x40000000, 0x40000040, 4)
                },
                "semantic_profiles": {
                    "events": [
                        {"address": 0x40000000, "value": 1, "is_read": True}
                    ] * 128,
                },
            },
            "stream_input_state": {"uart0": 17},
            "causal_context": {"active_task": "main", "pending_events": []},
        }

        blob = SnapshotStateBlob.from_mapping(
            payload,
            min_raw_bytes=0,
            compression_level=1,
        )
        self.assertIsInstance(blob, SnapshotStateBlob)
        self.assertGreater(blob.raw_size, blob.stored_size)
        self.assertEqual(payload, dict(blob))

        # Capture is immutable even if the caller mutates its temporary graph.
        payload["stream_input_state"]["uart0"] = 99
        self.assertEqual(17, dict(blob)["stream_input_state"]["uart0"])

        restored = pickle.loads(pickle.dumps(blob))
        self.assertIsInstance(restored, SnapshotStateBlob)
        self.assertEqual(dict(blob), dict(restored))

    def test_nested_replay_state_supports_serialization_without_compression(self):
        payload = {
            "events": [
                {"pc": 0x1000 + index, "value": index}
                for index in range(512)
            ]
        }
        blob = SnapshotStateBlob.from_mapping(
            payload,
            min_raw_bytes=0,
            compression_level=0,
        )

        self.assertIsInstance(blob, SnapshotStateBlob)
        self.assertFalse(blob.is_compressed)
        self.assertEqual(blob.raw_size, blob.stored_size)
        self.assertEqual(payload, dict(blob))

    def test_branch_snapshot_uses_compact_state_without_changing_fields(self):
        raw = b"D" * 512
        uc = DummyUC(raw)
        manager = BranchSnapshotManager()
        manager.set_memory_regions([(0x20000000, len(raw))])
        expected = {
            "mmio_handler": {
                "mmio_states": {index: {"value": index} for index in range(256)},
                "events": [{"pc": 0x1000, "value": 1}] * 1024,
            },
            "stream_input_state": {"uart0": 3},
        }
        manager.set_external_state_provider(
            lambda: {
                "input_event_index": 4,
                "input_occurrence_counts": {(0x1000, 0x40000000): 2},
                "external_model_state": expected,
            }
        )

        snapshot = manager.save_snapshot(
            uc,
            0x1000,
            0x1100,
            0x1004,
            "BNE",
        )

        self.assertIsInstance(snapshot.external_model_state, SnapshotStateBlob)
        self.assertEqual(expected, dict(snapshot.external_model_state))
        self.assertGreater(manager.get_statistics()["capture"]["external_state_raw_bytes"], 0)

    def test_compact_state_can_be_disabled_without_changing_payload(self):
        raw = b"E" * 512
        uc = DummyUC(raw)
        manager = BranchSnapshotManager()
        manager.set_memory_regions([(0x20000000, len(raw))])
        expected = {
            "mmio_handler": {
                "events": [{"pc": 0x1000, "value": index} for index in range(512)]
            }
        }
        manager.set_external_state_provider(
            lambda: {"external_model_state": expected}
        )

        with patch.dict(os.environ, {"LSGEMU_SNAPSHOT_STATE_COMPRESSION": "0"}):
            snapshot = manager.save_snapshot(
                uc,
                0x1000,
                0x1100,
                0x1004,
                "BNE",
            )

        self.assertIsInstance(snapshot.external_model_state, dict)
        self.assertEqual(expected, snapshot.external_model_state)
        self.assertEqual(
            1,
            manager.get_statistics()["capture"][
                "external_state_compression_disabled"
            ],
        )

    def test_lazy_state_is_consumed_by_the_existing_restore_contract(self):
        payload = {
            "stream_input_state": {"uart0": 4},
            "cortex_m_system_registers": {"icsr": 0x80},
            "skip_function_state": {0x1000: {"calls": 2}},
            "model_heap_next": 0x20001000,
            "model_heap_allocations": {0x20000000: 32},
            "external_memory_input_addresses": {0x20000010},
            "time_soft_tick": 7,
            "time_call_count": 3,
            "causal_context": {"active_task": "main"},
        }
        blob = SnapshotStateBlob.from_mapping(payload, min_raw_bytes=0)
        calls = []
        fake = SimpleNamespace(
            mmio_handler=SimpleNamespace(
                restore_runtime_state=lambda value: calls.append(value)
            ),
            stream_input_state={},
            cortex_m_system_registers={},
            skip_function_state={},
            model_heap_next=0,
            model_heap_allocations={},
            external_memory_input_addresses=set(),
            time_handler=SimpleNamespace(soft_tick=0, call_count=0),
            causal_context=SimpleNamespace(
                restore_runtime_state=lambda value: calls.append(value)
            ),
            causal_input_events=[],
        )
        snapshot = SimpleNamespace(
            external_model_state=blob,
            input_event_index=3,
            input_occurrence_counts={(0x1000, 0x40000000): 2},
        )

        IntelligentEmulator.restore_snapshot_external_state(fake, snapshot)

        # Restoring consumes the decoded graph but does not retain it in the
        # long-lived snapshot.  The compressed payload remains available for
        # a later replay.
        self.assertFalse(blob.is_materialized)
        self.assertEqual({"uart0": 4}, fake.stream_input_state)
        self.assertEqual({"icsr": 0x80}, fake.cortex_m_system_registers)
        self.assertEqual({0x1000: {"calls": 2}}, fake.skip_function_state)
        self.assertEqual(0x20001000, fake.model_heap_next)
        self.assertEqual({0x20000000: 32}, fake.model_heap_allocations)
        self.assertEqual({0x20000010}, fake.external_memory_input_addresses)
        self.assertEqual(7, fake.time_handler.soft_tick)
        self.assertEqual(3, fake.time_handler.call_count)
        self.assertEqual(3, fake.causal_input_event_count)
        self.assertEqual(2, fake.input_occurrence_counts[(0x1000, 0x40000000)])
        self.assertEqual(2, len(calls))

    def test_shared_causal_context_is_stored_once_in_compact_state(self):
        context = CausalExecutionContext()
        handler = StatefulMMIOHandler(causal_context=context)
        handler.get_or_create_state(0x40000000).current_value = 7

        view = handler.snapshot_runtime_state(
            copy_values=False,
            include_causal_context=False,
        )

        self.assertNotIn("causal_context", view)
        self.assertEqual(7, view["mmio_states"][0x40000000].current_value)

        detached = context.snapshot_runtime_state(copy_values=True)
        borrowed = context.snapshot_runtime_state(copy_values=False)
        restored = CausalExecutionContext()
        restored.restore_runtime_state(dict(borrowed))
        self.assertEqual(detached["active_task"], restored.active_task)
        self.assertEqual(detached["sequence"], restored.sequence)
        self.assertEqual(detached["event_counts"], dict(restored.event_counts))
        self.assertEqual(detached["pending_events"], list(restored.pending_events))

    def test_emulator_compact_capture_deduplicates_shared_context(self):
        context = CausalExecutionContext()
        context.active_task = "worker"
        handler = StatefulMMIOHandler(causal_context=context)
        handler.get_or_create_state(0x40000000).current_value = 7
        fake = SimpleNamespace(
            mmio_handler=handler,
            causal_context=context,
            stream_input_state={"uart0": 2},
            cortex_m_system_registers={},
            skip_function_state={},
            model_heap_next=0,
            model_heap_allocations={},
            external_memory_input_addresses=set(),
            time_handler=SimpleNamespace(soft_tick=0, call_count=0),
            causal_input_event_count=0,
            input_occurrence_counts={},
            causal_input_events=[],
        )

        with patch.dict(os.environ, {"LSGEMU_SNAPSHOT_STATE_COMPRESSION": "1"}):
            captured = IntelligentEmulator._snapshot_external_state(fake)

        external = captured["external_model_state"]
        self.assertIn("causal_context", external)
        self.assertNotIn("causal_context", external["mmio_handler"])

        blob = SnapshotStateBlob.from_mapping(external, min_raw_bytes=0)
        self.assertIsInstance(blob, SnapshotStateBlob)
        snapshot = SimpleNamespace(
            external_model_state=blob,
            input_event_index=captured["input_event_index"],
            input_occurrence_counts=captured["input_occurrence_counts"],
        )
        context.active_task = "mutated"
        handler.mmio_states[0x40000000].current_value = 99

        IntelligentEmulator.restore_snapshot_external_state(fake, snapshot)

        self.assertEqual("worker", context.active_task)
        self.assertEqual(7, handler.mmio_states[0x40000000].current_value)
        self.assertFalse(blob.is_materialized)

    def test_paged_memory_preserves_bytes_interface_and_reuses_pages(self):
        store = SnapshotPageStore(page_size=256)
        raw = bytes(range(256)) * 3
        first = store.intern_region(raw)
        second = store.intern_region(raw)

        self.assertIsInstance(first, PagedMemory)
        self.assertEqual(raw, bytes(first))
        self.assertEqual(len(raw), len(first))
        self.assertEqual(raw[250:270], first[250:270])
        self.assertEqual(raw[-1], first[-1])
        self.assertEqual(first, second)
        stats = store.get_statistics()
        self.assertGreaterEqual(stats["page_reuses"], 3)
        self.assertEqual(1, stats["live_unique_pages"])

    def test_required_capture_failure_never_creates_zero_filled_snapshot(self):
        manager = BranchSnapshotManager()
        manager.set_memory_regions([(0x20000000, 512)])
        uc = DummyUC(b"\x00" * 512, fail_reads=True)

        with self.assertRaises(SnapshotCaptureError):
            manager.save_snapshot(
                uc,
                0x08001000,
                0x08001010,
                0x08001004,
                "BNE",
            )

        self.assertEqual({}, manager.snapshots)
        self.assertEqual(1, manager.get_statistics()["capture"]["capture_failures"])

    def test_shared_store_deduplicates_branch_and_unique_bb_snapshots(self):
        store = SnapshotPageStore(page_size=256)
        raw = b"A" * 512
        uc = DummyUC(raw)
        branch = BranchSnapshotManager(page_store=store)
        branch.set_memory_regions([(0x20000000, len(raw))])
        unique = UniqueBBSnapshotManager(
            max_snapshots=2,
            memory_regions=[(0x20000000, len(raw))],
            page_store=store,
        )

        branch.save_snapshot(uc, 0x1000, 0x1100, 0x1004, "BNE")
        unique.create_snapshot(uc, 0x1000)

        stats = store.get_statistics()
        self.assertEqual(1, stats["live_unique_pages"])
        self.assertGreaterEqual(stats["page_reuses"], 3)

    def test_snapshot_metadata_store_reuses_immutable_flat_mappings(self):
        store = SnapshotMetadataStore()
        first = store.intern_mapping({("pc", 0x1000): 2, ("pc", 0x2000): 1})
        second = store.intern_mapping({("pc", 0x1000): 2, ("pc", 0x2000): 1})

        self.assertIs(first, second)
        self.assertEqual(2, store.get_statistics()["entries_avoided"])
        with self.assertRaises(TypeError):
            first[("pc", 0x3000)] = 1

    def test_new_emulator_enables_page_disk_backing_from_late_env(self):
        """P0-A (r7): a page store built before the run storage env exists must
        pick up disk backing when new_emulator retries configure_disk_backing,
        so campaign runs actually produce snapshot-pages-*.bin payloads."""

        class FakeEmulator:
            def __init__(self):
                self.loop_intervention_threshold_cache = {}

            def setup_memory(self):
                return None

            def load_firmware(self):
                return None

            def register_hooks(self):
                return None

        def build_prepared() -> PreparedFirmware:
            return PreparedFirmware(
                firmware_path=SimpleNamespace(),
                result=SimpleNamespace(arch_info=SimpleNamespace(entry_point=0)),
                static_bbs={},
                static_bb_set=set(),
                instruction_to_bb={},
                instruction_lookup={},
                branch_instruction_by_bb={},
                compare_lookup={},
                static_successors={},
                refined_basic_blocks=[],
                conditional_branch_bbs=[],
                ghidra_total_bbs=0,
            )

        with patch.dict(
            os.environ,
            {"LSGEMU_SNAPSHOT_PAGE_DISK_BACKING": "0"},
            clear=False,
        ):
            os.environ.pop("LSGEMU_SNAPSHOT_STORAGE_DIR", None)
            prepared = build_prepared()
            self.assertFalse(
                prepared.snapshot_page_store.get_statistics()["disk_backed"]
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LSGEMU_SNAPSHOT_PAGE_DISK_BACKING": "1",
                    "LSGEMU_SNAPSHOT_STORAGE_DIR": directory,
                },
                clear=False,
            ):
                with patch(
                    "lsgemu.prepared_firmware.IntelligentEmulator",
                    return_value=FakeEmulator(),
                ):
                    prepared.new_emulator()
                stats = prepared.snapshot_page_store.get_statistics()
                self.assertTrue(stats["disk_backed"])
                self.assertEqual(stats["disk_store"]["file_size"], 0)
                # Idempotent: a second emulator keeps the same backing file.
                with patch(
                    "lsgemu.prepared_firmware.IntelligentEmulator",
                    return_value=FakeEmulator(),
                ):
                    prepared.new_emulator()
                self.assertTrue(
                    prepared.snapshot_page_store.get_statistics()["disk_backed"]
                )

    def test_prepared_firmware_shares_pages_across_replay_emulators(self):
        class FakeEmulator:
            def __init__(self):
                self.loop_intervention_threshold_cache = {}

            def setup_memory(self):
                return None

            def load_firmware(self):
                return None

            def register_hooks(self):
                return None

        prepared = PreparedFirmware(
            firmware_path=SimpleNamespace(),
            result=SimpleNamespace(arch_info=SimpleNamespace(entry_point=0)),
            static_bbs={},
            static_bb_set=set(),
            instruction_to_bb={},
            instruction_lookup={},
            branch_instruction_by_bb={},
            compare_lookup={},
            static_successors={},
            refined_basic_blocks=[],
            conditional_branch_bbs=[],
            ghidra_total_bbs=0,
        )
        created = []

        def constructor(**kwargs):
            created.append(kwargs)
            return FakeEmulator()

        with patch("lsgemu.prepared_firmware.IntelligentEmulator", side_effect=constructor):
            prepared.new_emulator()
            prepared.new_emulator()

        self.assertIs(
            created[0]["snapshot_page_store"],
            created[1]["snapshot_page_store"],
        )
        self.assertIs(
            prepared.snapshot_page_store,
            created[0]["snapshot_page_store"],
        )
        self.assertIs(
            created[0]["snapshot_metadata_store"],
            created[1]["snapshot_metadata_store"],
        )
        self.assertIs(
            prepared.snapshot_metadata_store,
            created[0]["snapshot_metadata_store"],
        )
        restored = pickle.loads(pickle.dumps(prepared))
        self.assertIsInstance(restored.snapshot_page_store, SnapshotPageStore)
        self.assertIsInstance(restored.snapshot_metadata_store, SnapshotMetadataStore)

    def test_prepared_firmware_closes_engine_when_initialization_fails(self):
        class FailingEmulator:
            def __init__(self):
                self.loop_intervention_threshold_cache = {}
                self.closed = 0

            def setup_memory(self):
                return None

            def load_firmware(self):
                raise RuntimeError("injected load failure")

            def register_hooks(self):
                self.fail("register_hooks must not run after load failure")

            def close(self):
                self.closed += 1

            def fail(self, message):
                raise AssertionError(message)

        prepared = PreparedFirmware(
            firmware_path=SimpleNamespace(),
            result=SimpleNamespace(arch_info=SimpleNamespace(entry_point=0)),
            static_bbs={},
            static_bb_set=set(),
            instruction_to_bb={},
            instruction_lookup={},
            branch_instruction_by_bb={},
            compare_lookup={},
            static_successors={},
            refined_basic_blocks=[],
            conditional_branch_bbs=[],
            ghidra_total_bbs=0,
        )
        engine = FailingEmulator()
        with patch(
            "lsgemu.prepared_firmware.IntelligentEmulator",
            return_value=engine,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected load failure"):
                prepared.new_emulator()

        self.assertEqual(1, engine.closed)

    def test_restore_rejects_truncated_region_before_any_memory_write(self):
        raw = b"B" * 512
        capture_uc = DummyUC(raw)
        manager = BranchSnapshotManager()
        manager.set_memory_regions([(0x20000000, len(raw))])
        snapshot = manager.save_snapshot(
            capture_uc,
            0x1000,
            0x1100,
            0x1004,
            "BNE",
        )
        snapshot.memory_regions[(0x20000000, len(raw))] = b"short"
        restore_uc = DummyUC(raw)

        self.assertFalse(manager.restore_snapshot(restore_uc, snapshot))
        self.assertEqual([], restore_uc.memory_writes)
        self.assertEqual(
            1,
            manager.get_statistics()["capture"]["restore_integrity_failures"],
        )

    def test_current_snapshot_limit_uses_lru_without_truncating_event_catalog(self):
        raw = b"C" * 256
        uc = DummyUC(raw)
        manager = BranchSnapshotManager(max_current_snapshots=2)
        manager.set_memory_regions([(0x20000000, len(raw))])
        addresses = (0x08001000, 0x08002000, 0x08003000)

        manager.save_snapshot(
            uc,
            addresses[0],
            addresses[0] + 0x10,
            addresses[0] + 4,
            "BNE",
        )
        manager.record_event(
            addresses[0], addresses[0], addresses[0] + 0x10,
            addresses[0] + 4, "BNE", True, 0,
        )
        manager.save_snapshot(
            uc,
            addresses[1],
            addresses[1] + 0x10,
            addresses[1] + 4,
            "BNE",
        )
        manager.record_event(
            addresses[1], addresses[1], addresses[1] + 0x10,
            addresses[1] + 4, "BNE", True, 1,
        )

        self.assertIsNotNone(manager.get_snapshot(addresses[0]))
        manager.save_snapshot(
            uc,
            addresses[2],
            addresses[2] + 0x10,
            addresses[2] + 4,
            "BNE",
        )
        manager.record_event(
            addresses[2], addresses[2], addresses[2] + 0x10,
            addresses[2] + 4, "BNE", True, 2,
        )

        self.assertEqual({addresses[0], addresses[2]}, set(manager.snapshots))
        self.assertIsNone(manager.get_snapshot(addresses[1]))
        self.assertEqual(3, len(manager.get_ordered_events()))
        self.assertEqual(
            1,
            manager.get_statistics()["current_snapshot_evictions"],
        )


if __name__ == "__main__":
    unittest.main()
