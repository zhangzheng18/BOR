#!/usr/bin/env python3
"""r7 P0-B 单测：DFS 回退锚点池（FIFO 免疫 / 跨阶段存活 / 冷热分层等价恢复 / 血统门控）。"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshot
from lsgemu.dfs_anchor_pool import (
    DFSColdAnchorStore,
    DFSAnchorPool,
    anchor_lineage_status,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.runner_models import BranchConstraintCandidate, PrefixReplaySnapshot


def _validated_snapshot(reg0: int = 0x11, memory: bytes = bytes(range(256)) * 16) -> BranchSnapshot:
    snap = BranchSnapshot(
        address=0x1000,
        target=0x2000,
        fallthrough=0x1004,
        condition="eq",
        original_taken=True,
        order=0,
        depth=0,
        registers={"r0": reg0, "pc": 0x1000, "sp": 0x20080000},
        cpsr=0x13,
        memory_data=memory,
        memory_base=0x20000000,
        memory_size=len(memory),
    )
    snap.provenance_status = "validated"
    snap.provenance_finalized = True
    snap.prefix_telemetry_complete = True
    snap.source_execution_id = "exec-anchor-test"
    return snap


def _validated_entry(depth_prefix, next_key, snapshot, branch_depth):
    entry = PrefixReplaySnapshot(
        prefix_signature=tuple(depth_prefix),
        next_branch_key=next_key,
        snapshot=snapshot,
        branch_depth=branch_depth,
        provenance_status="validated",
        source_execution_id="exec-anchor-test",
    )
    entry.provenance_finalized = True
    entry.prefix_telemetry_complete = True
    return entry


def _tables(items=None):
    return {
        "runner_scoped_constraints": list(
            items
            if items is not None
            else [
                BranchConstraintCandidate(
                    constraint_type="mmio_read",
                    address=0x40000000,
                    value=0x55,
                    read_pc=0x8000100,
                )
            ]
        )
    }


class DFSAnchorPoolTests(unittest.TestCase):
    def test_stride_gate_registers_every_k_branch_points(self):
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16, cold_store=None)
        snap = _validated_snapshot()
        registered = []
        for depth in range(0, 12):
            entry = _validated_entry(
                (((0x800 + depth, 1), True),), (0x1000 + depth, 1), snap, depth
            )
            anchor = pool.consider(entry, depth=depth, prefix_constraint_tables=_tables())
            if anchor is not None:
                registered.append(depth)
        self.assertEqual(registered, [4, 8])
        stats = pool.statistics()
        self.assertEqual(stats["anchors_total"], 2)
        self.assertEqual(stats["flip_eligible"], 2)

    def test_stride_windows_independent_of_arrival_order(self):
        """r10 ③：深候选先到不得永久屏蔽其后到达的浅窗口。

        旧实现（_next_anchor_depth 单调门槛）：depth 9 先登记后 next=13，
        其后到达的 depth 4-7 永远被拒——回退梯度被到达顺序锁死。
        """
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16, cold_store=None)
        snap = _validated_snapshot()
        deep = pool.consider(
            _validated_entry((((0x900, 1), True),) * 9, (0x1900, 1), snap, 9),
            depth=9,
            prefix_constraint_tables=_tables(),
        )
        self.assertIsNotNone(deep)
        shallow = pool.consider(
            _validated_entry((((0x800, 1), True),) * 5, (0x1500, 1), snap, 5),
            depth=5,
            prefix_constraint_tables=_tables(),
        )
        self.assertIsNotNone(shallow, "浅窗口必须仍可登记（与到达顺序无关）")
        same_window = pool.consider(
            _validated_entry((((0x860, 1), True),) * 6, (0x1600, 1), snap, 6),
            depth=6,
            prefix_constraint_tables=_tables(),
        )
        self.assertIsNone(same_window)
        stats = pool.statistics()
        self.assertEqual(stats["anchors_total"], 2)
        self.assertEqual(stats["stats"]["skipped_stride_window_filled"], 1)

    def test_stride_rejection_funnel_closes(self):
        """r10 ③：considered = registered + 显式拒绝桶（无静默丢失）。"""
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16, cold_store=None)
        snap = _validated_snapshot()
        for depth in range(0, 12):
            pool.consider(
                _validated_entry(
                    (((0x800 + depth, 1), True),), (0x1000 + depth, 1), snap, depth
                ),
                depth=depth,
                prefix_constraint_tables=_tables(),
            )
        stats = pool.statistics()["stats"]
        self.assertEqual(stats["considered"], 12)
        self.assertEqual(stats["registered"], 2)
        self.assertEqual(stats["skipped_below_stride"], 4)  # depth 0-3
        self.assertEqual(stats["skipped_stride_window_filled"], 6)  # 5,6,7,9,10,11
        self.assertEqual(
            stats["considered"],
            stats["registered"]
            + stats["skipped_below_stride"]
            + stats["skipped_stride_window_filled"],
        )
        self.assertEqual(stats["max_candidate_depth"], 11)

    def test_eligible_candidate_upgrades_dirty_anchor_in_same_window(self):
        """r10 ③：同窗口内血统不合格锚点让位给合格候选（判定本身不放宽）。"""
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16, cold_store=None)
        dirty_snap = _validated_snapshot()
        dirty_snap.provenance_status = "diagnostic"
        dirty = pool.consider(
            _validated_entry((((0x800, 1), True),) * 4, (0x1400, 1), dirty_snap, 4),
            depth=4,
            prefix_constraint_tables=_tables(),
        )
        self.assertIsNotNone(dirty)
        self.assertFalse(dirty.flip_eligible)
        clean = pool.consider(
            _validated_entry(
                (((0x820, 1), True),) * 6, (0x1600, 1), _validated_snapshot(), 6
            ),
            depth=6,
            prefix_constraint_tables=_tables(),
        )
        self.assertIsNotNone(clean)
        self.assertTrue(clean.flip_eligible)
        stats = pool.statistics()
        self.assertEqual(stats["anchors_total"], 1)  # 顶替而非并存
        self.assertEqual(stats["flip_eligible"], 1)
        self.assertEqual(stats["stats"]["window_upgrades"], 1)
        self.assertNotIn(dirty.anchor_key, pool.anchors)

    def test_dirty_lineage_and_missing_tables_refuse_flip(self):
        pool = DFSAnchorPool(stride=1, hot_limit=8, total_limit=16)
        snap = _validated_snapshot()
        snap.provenance_status = "diagnostic"
        dirty = _validated_entry((((0x800, 1), True),), (0x1000, 1), snap, 1)
        anchor = pool.consider(dirty, depth=1, prefix_constraint_tables=_tables())
        self.assertIsNotNone(anchor)
        self.assertFalse(anchor.flip_eligible)
        self.assertTrue(anchor.ineligibility_reasons)

        clean = _validated_entry(
            (((0x900, 1), False),), (0x1100, 1), _validated_snapshot(), 2
        )
        no_tables = pool.consider(clean, depth=2, prefix_constraint_tables=None)
        self.assertIsNotNone(no_tables)
        self.assertFalse(no_tables.flip_eligible)
        self.assertIn(
            "prefix_constraint_tables_missing", no_tables.ineligibility_reasons
        )

    def test_anchors_survive_reservoir_fifo_pressure_and_reset(self):
        """验收口径：FIFO 挤压 + 阶段 reset 后锚点仍可命中。"""
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.dfs_anchor_pool = None
        runner.snapshot_resource_stats = {}
        runner.reservoir_prefix_snapshots = {}
        with patch.object(
            runner,
            "_matching_scoped_branch_constraints",
            return_value=[],
            create=True,
        ), patch.dict(os.environ, {"LSGEMU_DFS_ANCHOR_STRIDE": "2"}):
            # r9 C 件：步进深度=前缀长度（branch_depth 事件深度已不参与门控，
            # 故意给 2729 证明不被轮询环污染）。前缀长度 2 ≥ stride 2 ⇒ 登记。
            entry = _validated_entry(
                (((0x800, 1), True), ((0x900, 1), False)),
                (0x1000, 1),
                _validated_snapshot(),
                2729,
            )
            anchor = runner._dfs_consider_prefix_anchor(entry)
            self.assertIsNotNone(anchor)

            # FIFO 挤压：把常规前缀快照池灌爆（模拟 96 上限淘汰）。
            for index in range(500):
                runner.reservoir_prefix_snapshots[
                    (((0x800 + index, 1), True),), (0x1000 + index, 1)
                ] = entry
            runner.reservoir_prefix_snapshots.pop(
                (((0x800, 1), True),), (0x1000, 1)
            )
            self.assertNotIn(
                (((0x800, 1), True),), (0x1000, 1), runner.reservoir_prefix_snapshots
            )
            self.assertIn(
                anchor.anchor_key, runner.dfs_anchor_snapshots
            )

            # 阶段 reset：调度派发状态清空，锚点池必须存活。
            runner.reservoir_task_queue = ["stale"]
            runner.reservoir_deferred_tasks = ["stale"]
            runner.reservoir_scoped_probe_tasks = []
            runner.reservoir_explored_paths = set()
            runner.reservoir_incomplete_paths = set()
            runner.reservoir_pending_paths = set()
            runner.reservoir_task_contexts = {}
            runner.reservoir_explored_edges = set()
            runner.reservoir_root_budget_used = {}
            runner.reservoir_roots_started = set()
            runner.reservoir_discovered_branch_keys = set()
            runner.reservoir_new_branch_points = set()
            runner.reservoir_discovery_events = []  # k5 C4 发现点遥测（reset 面）
            runner.reservoir_discovery_events_truncated = 0
            runner.reservoir_state_signatures = set()
            runner.reservoir_saturated_edges = set()
            runner.reservoir_interrupt_contexts = {}
            runner.reservoir_hotset_retried_paths = set()
            runner.deadlock_failed_directions = set()
            runner.reservoir_initialized = True
            runner._reset_reservoir_dispatch_state()
            self.assertFalse(runner.reservoir_initialized)
            self.assertEqual(runner.reservoir_task_queue, [])
            self.assertIn(anchor.anchor_key, runner.dfs_anchor_snapshots)

    def test_cold_anchor_round_trip_restores_equivalent_state(self):
        """验收口径：冷锚点落盘后恢复出等价状态（寄存器 + 内存摘要）。"""
        memory = bytes(range(256)) * 16
        snap = _validated_snapshot(memory=memory)
        entry = _validated_entry((((0x800, 1), True),), (0x1000, 1), snap, 1)
        with tempfile.TemporaryDirectory() as directory:
            pool = DFSAnchorPool(
                stride=1, hot_limit=1, total_limit=8, cold_store=DFSColdAnchorStore(directory)
            )
            first = pool.consider(entry, depth=1, prefix_constraint_tables=_tables())
            deeper = _validated_entry(
                (((0x800, 1), True), ((0x900, 1), False)),
                (0x1100, 1),
                _validated_snapshot(),
                2,
            )
            pool.consider(deeper, depth=2, prefix_constraint_tables=_tables())
            self.assertFalse(first.is_hot)  # 已降冷
            self.assertTrue(first.cold_location is not None)

            restored = pool.lookup(first.anchor_key)
            self.assertIsNotNone(restored)
            self.assertTrue(restored.is_hot)
            restored_snapshot = restored.entry.snapshot
            self.assertEqual(restored_snapshot.registers, snap.registers)
            self.assertEqual(restored_snapshot.cpsr, snap.cpsr)
            restored_memory = bytes(restored_snapshot.memory_data)
            self.assertEqual(
                hashlib.sha256(restored_memory).hexdigest(),
                hashlib.sha256(memory).hexdigest(),
            )
            self.assertEqual(
                restored.prefix_constraint_tables["runner_scoped_constraints"][0].address,
                0x40000000,
            )
            stats = pool.statistics()
            self.assertGreaterEqual(stats["stats"]["resurrected_from_cold"], 1)

    def test_cold_payload_integrity_and_whitelist(self):
        snap = _validated_snapshot()
        entry = _validated_entry((((0x800, 1), True),), (0x1000, 1), snap, 1)
        with tempfile.TemporaryDirectory() as directory:
            store = DFSColdAnchorStore(directory)
            pool = DFSAnchorPool(
                stride=1, hot_limit=1, total_limit=8, cold_store=store
            )
            first = pool.consider(entry, depth=1, prefix_constraint_tables=_tables())
            pool.consider(
                _validated_entry(
                    (((0x800, 1), True), ((0x900, 1), False)),
                    (0x1100, 1),
                    snap,
                    2,
                ),
                depth=2,
                prefix_constraint_tables=_tables(),
            )
            offset, length, digest = first.cold_location
            # 篡改载荷 ⇒ digest 拒绝。
            file_path = next(store.directory.iterdir())
            with file_path.open("r+b") as handle:
                handle.seek(offset + length - 4)
                original = handle.read(4)
                handle.seek(offset + length - 4)
                handle.write(b"\x00\x00\x00\x00")
            with self.assertRaises(ValueError):
                store.deserialize((offset, length, digest))
            with file_path.open("r+b") as handle:
                handle.seek(offset + length - 4)
                handle.write(original)

            # 白名单外的 GLOBAL ⇒ 拒绝反序列化。
            from lsgemu.dfs_anchor_pool import _RestrictedAnchorUnpickler, _BytesReader

            payload = pickle.dumps(os.system)
            with self.assertRaises(pickle.UnpicklingError):
                _RestrictedAnchorUnpickler(_BytesReader(payload)).load()

    def test_unpicklable_payload_keeps_anchor_hot(self):
        pool = DFSAnchorPool(stride=1, hot_limit=1, total_limit=8, cold_store=None)
        snap = _validated_snapshot()
        first = pool.consider(
            _validated_entry((((0x800, 1), True),), (0x1000, 1), snap, 1),
            depth=1,
            prefix_constraint_tables=_tables(),
        )
        pool.consider(
            _validated_entry((((0x900, 1), True),), (0x1200, 1), snap, 2),
            depth=2,
            prefix_constraint_tables=_tables(),
        )
        # 冷层不可用 ⇒ 计数暴露退化，锚点保持热态可用。
        stats = pool.statistics()
        self.assertGreaterEqual(stats["stats"]["cold_demote_failures"], 1)
        self.assertIsNotNone(pool.lookup(first.anchor_key))

    def test_total_limit_drops_oldest_anchor(self):
        pool = DFSAnchorPool(stride=1, hot_limit=8, total_limit=2)
        snap = _validated_snapshot()
        anchors = []
        for depth in (1, 2, 3):
            anchors.append(
                pool.consider(
                    _validated_entry(
                        (((0x800 + depth, 1), True),), (0x1000 + depth, 1), snap, depth
                    ),
                    depth=depth,
                    prefix_constraint_tables=_tables(),
                )
            )
        self.assertIsNone(pool.lookup(anchors[0].anchor_key))
        self.assertIsNotNone(pool.lookup(anchors[1].anchor_key))
        self.assertIsNotNone(pool.lookup(anchors[2].anchor_key))

    def test_nearest_ancestor_picks_deepest_matching_prefix(self):
        pool = DFSAnchorPool(stride=1, hot_limit=8, total_limit=16)
        snap = _validated_snapshot()
        path_a = (((0x800, 1), True), ((0x900, 1), False), ((0xA00, 1), True))
        pool.consider(
            _validated_entry(path_a[:1], (0x900, 1), snap, 1),
            depth=1,
            prefix_constraint_tables=_tables(),
            anchor_key=(path_a[:1], (0x900, 1)),
        )
        pool.consider(
            _validated_entry(path_a[:2], (0xA00, 1), snap, 2),
            depth=2,
            prefix_constraint_tables=_tables(),
            anchor_key=(path_a[:2], (0xA00, 1)),
        )
        pool.consider(
            _validated_entry(
                (((0xF00, 1), True),), (0xF10, 1), snap, 9
            ),  # 深度更大但不同路径
            depth=9,
            prefix_constraint_tables=_tables(),
            anchor_key=((((0xF00, 1), True),), (0xF10, 1)),
        )
        ancestor = pool.nearest_ancestor(list(path_a))
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.depth, 2)
        self.assertEqual(ancestor.anchor_key[1], (0xA00, 1))

    def test_lineage_status_uses_shared_evidence_contract(self):
        snap = _validated_snapshot()
        entry = _validated_entry((((0x800, 1), True),), (0x1000, 1), snap, 1)
        eligible, reasons, summary = anchor_lineage_status(entry)
        self.assertTrue(eligible)
        self.assertEqual(reasons, ())
        self.assertEqual(summary["status"], "validated")
        self.assertEqual(summary["execution_id"], "exec-anchor-test")


class PrefixAnchorHookTests(unittest.TestCase):
    """r9 C 件：空前缀拒绝 + 步进深度改用前缀长度 + planned=0 可观测。"""

    PATH4 = (
        ((0x800, 1), True),
        ((0x900, 1), False),
        ((0xA00, 1), True),
        ((0xB00, 1), True),
    )

    def _runner_with_pool(self, stride=4):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.dfs_anchor_pool = DFSAnchorPool(
            stride=stride, hot_limit=8, total_limit=16
        )
        runner._matching_scoped_branch_constraints = lambda signature: []
        return runner

    def test_empty_prefix_entry_skipped_without_consuming_stride(self):
        runner = self._runner_with_pool(stride=4)
        snap = _validated_snapshot()
        # 90min 长跑形态：事件深度被轮询环顶到 2729 的空前缀入口快照。
        empty = _validated_entry((), (0x081357AA, 2729), snap, 2729)
        self.assertIsNone(runner._dfs_consider_prefix_anchor(empty))
        pool = runner.dfs_anchor_pool
        self.assertEqual(1, pool.stats["skipped_empty_prefix"])
        self.assertEqual(0, pool.stats["considered"])
        self.assertEqual({}, pool.anchors)
        # 额度未被占用：前缀长度 4 的正常锚点仍可登记。
        anchor = runner._dfs_consider_prefix_anchor(
            _validated_entry(self.PATH4, (0xC00, 1), snap, 4)
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(4, anchor.depth)

    def test_depth_is_prefix_length_not_event_depth(self):
        runner = self._runner_with_pool(stride=4)
        snap = _validated_snapshot()
        # branch_depth=2729（事件深度，被环迭代污染）但前缀长度 4 ⇒ 深度=4。
        anchor = runner._dfs_consider_prefix_anchor(
            _validated_entry(self.PATH4, (0xC00, 1), snap, 2729)
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(4, anchor.depth)

    def test_flip_pass_reports_planned_zero_reasons(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.dfs_anchor_pool = DFSAnchorPool(
            stride=4, hot_limit=8, total_limit=16
        )
        payload = runner._dfs_execute_flip_pass(known_coverage=None)
        self.assertEqual(0, payload["attempt_stats"]["planned"])
        self.assertEqual("no_anchors", payload["planned_zero_reason"])
        self.assertEqual(0, payload["trunk_depth"])

        # 直接塞一个空签名锚点（绕过钩子）：主干为空 ⇒ 早退原因可读。
        snap = _validated_snapshot()
        runner.dfs_anchor_pool.consider(
            _validated_entry((), (0xC00, 1), snap, 2729),
            depth=4,
            prefix_constraint_tables=_tables(),
            anchor_key=((), (0xC00, 1)),
        )
        payload = runner._dfs_execute_flip_pass(known_coverage=None)
        self.assertEqual(0, payload["attempt_stats"]["planned"])
        self.assertEqual("empty_trunk_prefix", payload["planned_zero_reason"])
        self.assertEqual(0, payload["trunk_depth"])


if __name__ == "__main__":
    unittest.main()
