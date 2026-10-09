#!/usr/bin/env python3
"""cycle3 k.5 C4（P1-F）：发现点 oracle 事件 + 同域 join。

FINAL_PLAN_CYCLE3 §4 C4 断言面：
1. 合成 mini-catalog：主键 ``(bb, local_occ, identity)`` 命中（多事件 bb
   也能精确命中，不被 K0 式 (bb,occ) 塌缩误拼）；legacy 回退 K1=(bb,)+
   单边唯一；多事件未命中 ⇒ ambiguous；零事件 ⇒ oracle_unavailable 子标。
2. 遗留 fixture（真 armP3 catalog/report 同源投影，2026-10-09 复跑逐位
   重现 docs/k5_assets/c3r4_t2_join.py 输出）：joined=76, ambiguous=0,
   oracle_unavailable=6（4×rv_root_no_direction + 2×no_root_at_all）。
3. ``_record_reservoir_discovery_event`` 字段（bb/local_occ/original_taken/
   direction_provenance/identity）+ 截断。
4. ``save_branch_catalog`` 序列化携带 ``reservoir_discovery_events``（列表）
   与 ``reservoir_discovery_events_truncated``。

硬条款：join **禁用 catalog 全局 occ 键**（R2/R4 裁定双域机械根因）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import (
    HistoricalRunner,
    join_discovery_direction_oracle,
)

# -- 遗留 fixture（同源投影）------------------------------------------------
# 投影基（final_4h_armP3_on_340_20260930_092027，与 c3r4_t2_join.py 同输入）：
# 82 条 entry_derived 快照记录落在 23 个 bb；catalog main_path_events 在其中
# 20 个 bb 恰 1 条事件（覆盖 76 条记录）、3 个 bb 0 条事件（6 条记录）。
# (bb, 记录数)；occ 全部按 K1 语义不影响结果，16 条 miss 的真实 occ 见
# test_snapshot_identity_telemetry.K0_MISS_RECORDS。
LEGACY_RECORD_BB_MULTIPLICITY = [
    ("0x080050f0", 16), ("0x08007934", 2), ("0x08087932", 2),
    ("0x080a51e6", 1), ("0x080a51f4", 1), ("0x080c44c4", 5),
    ("0x080da7ac", 14), ("0x081235aa", 14), ("0x08133410", 2),
    ("0x08133454", 2), ("0x08133460", 2), ("0x08133a88", 2),
    ("0x08134030", 1), ("0x08134986", 2), ("0x081349d8", 2),
    ("0x08134a2c", 2), ("0x0813517c", 2), ("0x08135188", 1),
    ("0x0813569a", 2), ("0x081357d8", 2), ("0x08136024", 2),
    ("0x08136032", 2), ("0x08138578", 1),
]
# 3 个零事件 bb：前两个在 rv 变体表（rv_root_no_direction），后一个不在。
LEGACY_UNAVAILABLE_BBS = ["0x08133454", "0x08134986", "0x0813569a"]
# 20 个单事件 bb = 记录 bb 集 − 零事件 bb。
LEGACY_SINGLE_EVENT_BBS = [
    bb for bb, _ in LEGACY_RECORD_BB_MULTIPLICITY
    if bb not in LEGACY_UNAVAILABLE_BBS
]


def _legacy_records():
    return [
        {"bb": bb, "root_occurrence": 1, "identity": None}
        for bb, count in LEGACY_RECORD_BB_MULTIPLICITY
        for _ in range(count)
    ]


def _legacy_events():
    return [
        {"bb": bb, "local_occ": 1, "identity": None}
        for bb in LEGACY_SINGLE_EVENT_BBS
    ]


class JoinDiscoveryDirectionOracleTests(unittest.TestCase):
    def test_synthetic_mini_catalog_primary_key_hits(self):
        bb_hot = 0x08134030  # 多事件 bb：两条事件同 occ 不同 identity
        events = [
            {
                "bb": f"0x{bb_hot:08x}",
                "local_occ": 1,
                "identity": [0, bb_hot, 11, 1],
                "original_taken": True,
            },
            {
                "bb": f"0x{bb_hot:08x}",
                "local_occ": 1,
                "identity": [1, bb_hot, 42, 1],
                "original_taken": False,
            },
            # 单事件 bb：K1 回退的用武之地。
            {"bb": "0x080050f0", "local_occ": 3, "identity": None},
        ]
        records = [
            # 主键命中：多事件 bb 上精确命中第二条（original_taken=False）。
            {"bb": f"0x{bb_hot:08x}", "root_occurrence": 1,
             "identity": [1, bb_hot, 42, 1]},
            # 同 bb、identity 未命中且多事件 ⇒ ambiguous（不硬拼）。
            {"bb": f"0x{bb_hot:08x}", "root_occurrence": 1,
             "identity": [9, bb_hot, 99, 1]},
            # legacy 无 identity、单事件 bb ⇒ K1 单边唯一。
            {"bb": "0x080050f0", "root_occurrence": 1, "identity": None},
            # 零事件 bb ⇒ oracle_unavailable 子标（rv 与非 rv 各一）。
            {"bb": "0x08134986", "root_occurrence": 1, "identity": None},
            {"bb": "0x0813569a", "root_occurrence": 1, "identity": None},
        ]
        result = join_discovery_direction_oracle(
            events, records, rv_branches=[0x08134986]
        )
        self.assertEqual(result["joined"], 2)
        self.assertEqual(result["ambiguous"], 1)
        self.assertEqual(result["oracle_unavailable"], 2)
        self.assertEqual(result["by_source"]["primary"], 1)
        self.assertEqual(result["by_source"]["k1_single_edge"], 1)
        # 同 bb 多条记录：按位置断言（dict 按 bb 会折叠）。
        record_labels = [item["label"] for item in result["records"]]
        self.assertEqual(record_labels[0], "primary")
        self.assertEqual(record_labels[1], "ambiguous")
        self.assertEqual(record_labels[2], "k1_single_edge")
        self.assertEqual(
            record_labels[3], "oracle_unavailable:rv_root_no_direction"
        )
        self.assertEqual(
            record_labels[4], "oracle_unavailable:no_root_at_all"
        )

    def test_primary_key_rejects_global_occ_hard_join(self):
        # K0 形态反例：record 的 occ 与事件 occ 不同域时，(bb, occ) 硬拼
        # 不得作为命中通道——本函数的命中必须同时携带 identity 三元组。
        events = [
            {"bb": "0x08134030", "local_occ": 24, "identity": [0, 0x08134030, 5, 1]},
        ]
        records = [
            # occ 恰好相同（24）但 identity 缺失：多事件才歧义，单事件走 K1；
            # 关键是它不能因为 occ 相同就被标成 primary。
            {"bb": "0x08134030", "root_occurrence": 24, "identity": None},
        ]
        result = join_discovery_direction_oracle(events, records)
        self.assertEqual(result["by_source"].get("primary", 0), 0)
        self.assertEqual(result["joined"], 1)  # 经 K1 单边唯一

    def test_legacy_fixture_joined_76_ambiguous_0_unavailable_6(self):
        result = join_discovery_direction_oracle(
            _legacy_events(),
            _legacy_records(),
            rv_branches=[int(bb, 16) for bb in LEGACY_UNAVAILABLE_BBS[:2]],
        )
        self.assertEqual(sum(count for _, count in LEGACY_RECORD_BB_MULTIPLICITY), 82)
        self.assertEqual(result["joined"], 76)
        self.assertEqual(result["ambiguous"], 0)
        self.assertEqual(result["oracle_unavailable"], 6)
        self.assertEqual(
            result["by_source"]["oracle_unavailable:rv_root_no_direction"], 4
        )
        self.assertEqual(
            result["by_source"]["oracle_unavailable:no_root_at_all"], 2
        )


def _shell_runner() -> HistoricalRunner:
    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.reservoir_discovery_events = []
    runner.reservoir_discovery_events_truncated = 0
    return runner


class RecordReservoirDiscoveryEventTests(unittest.TestCase):
    def test_fields_and_identity_extraction(self):
        runner = _shell_runner()
        child = SimpleNamespace(
            address=0x08134030,
            occurrence_index=7,
            original_taken=True,
            direction_provenance="unicorn_execution",
        )
        snapshot = SimpleNamespace(
            external_model_state={"snapshot_identity": (2, 0x08134030, 15, 7)}
        )
        runner._record_reservoir_discovery_event(
            child, {0x08134030: snapshot}
        )
        self.assertEqual(
            runner.reservoir_discovery_events,
            [{
                "bb": "0x08134030",
                "local_occ": 7,
                "original_taken": True,
                "direction_provenance": "unicorn_execution",
                "identity": [2, 0x08134030, 15, 7],
            }],
        )

    def test_legacy_snapshot_yields_identity_none(self):
        runner = _shell_runner()
        child = SimpleNamespace(
            address=0x080050f0,
            occurrence_index=1,
            original_taken=False,
            direction_provenance="unicorn_execution",
        )
        runner._record_reservoir_discovery_event(
            child, {0x080050f0: SimpleNamespace(external_model_state={})}
        )
        self.assertIsNone(runner.reservoir_discovery_events[0]["identity"])

    def test_truncation_beyond_limit(self):
        runner = _shell_runner()
        child = SimpleNamespace(
            address=0x080050f0, occurrence_index=1, original_taken=True,
            direction_provenance="unicorn_execution",
        )
        limit = HistoricalRunner.RESERVOIR_DISCOVERY_EVENT_LIMIT
        for _ in range(limit + 3):
            runner._record_reservoir_discovery_event(child, {})
        self.assertEqual(len(runner.reservoir_discovery_events), limit)
        self.assertEqual(runner.reservoir_discovery_events_truncated, 3)


class SaveBranchCatalogKeysTests(unittest.TestCase):
    def test_payload_carries_discovery_event_keys(self):
        import tempfile

        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(
            branch_instruction_by_bb={}, firmware_path="/tmp/fake.elf"
        )
        runner.emulator = SimpleNamespace(
            branch_snapshot_manager=SimpleNamespace(get_ordered_events=lambda: [])
        )
        runner.known_branch_root_snapshots = {}
        runner.known_main_branch_events = {}
        runner.reservoir_discovered_branch_keys = {(0x080050F0, 1)}
        runner.reservoir_new_branch_points = {0x080050F0}
        runner.reservoir_discovery_events = [{
            "bb": "0x080050f0", "local_occ": 1, "original_taken": True,
            "direction_provenance": "unicorn_execution", "identity": None,
        }]
        runner.reservoir_discovery_events_truncated = 2

        def _variant_count():
            return 0

        runner._known_branch_root_snapshot_variant_count = _variant_count
        with tempfile.TemporaryDirectory(prefix="lsgemu_k5_c4_") as tmp:
            out = Path(tmp) / "catalog.json"
            payload = runner.save_branch_catalog(str(out))
            on_disk = json.loads(out.read_text())
        for target in (payload, on_disk):
            self.assertEqual(
                target["reservoir_discovered_branch_events"], 1
            )
            self.assertEqual(len(target["reservoir_discovery_events"]), 1)
            self.assertEqual(
                target["reservoir_discovery_events"][0]["bb"], "0x080050f0"
            )
            self.assertEqual(target["reservoir_discovery_events_truncated"], 2)


if __name__ == "__main__":
    unittest.main()
