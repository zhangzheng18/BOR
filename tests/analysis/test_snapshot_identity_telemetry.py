#!/usr/bin/env python3
"""cycle3 k.5 C3（P0-D）：blob identity 遥测——(era, bb, c2, c3) 四元组。

FINAL_PLAN_CYCLE3 §4 C3 断言面：
1. 两逻辑态**仅 era 异** ⇒ 四元组互异（era 是塌缩消毒的第一维）；
2. 同 era 同 bb 重复捕获 ⇒ c3 递增、c2（seq）递增 ⇒ 互异（现库 identity
   塌缩最大簇 3070 的机械根因：计数跨域混并 + 无序号）；
3. 16 条 K0 miss 样本（真 armP3 join miss 表内嵌）逐条走「restore 根 →
   捕获」产线路径 ⇒ 16/16 四元组互异；
4. ``root_occurrence_local`` 从被恢复快照 identity 的 c3 继承；legacy 快照
   （无 identity）恢复后保持 None。

样本表来源：docs/k5_assets/c3r4_t2_join.py 对
``final_4h_armP3_on_340_20260930_092027`` 的 K0 miss 输出
（``miss bb/occ: [...]``，2026-10-09 本轮复跑逐位重现）。
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  (unicorn 绑定补丁：先 import lsgemu 再建 Uc)

from lsgemu.dfs_flip_kit import FLIP_BRANCH_BB, build_flip_emulator, build_flip_image

# 16 条 K0 miss 记录（bb 十进制=产线 JSON 原值，occ=root_occurrence）。
# 8 个 bb：7 个 occ=1×2 重复（K0 (bb,occ) 塌缩面）+ 0x081349d8 occ∈{1,24}。
K0_MISS_RECORDS = [
    (135476240, 1), (135476240, 1),  # 0x08133410
    (135476308, 1), (135476308, 1),  # 0x08133454
    (135476320, 1), (135476320, 1),  # 0x08133460
    (135481734, 1), (135481734, 1),  # 0x08134986
    (135481816, 1), (135481816, 24),  # 0x081349d8
    (135485082, 1), (135485082, 1),  # 0x0813569a
    (135487524, 1), (135487524, 1),  # 0x08136024
    (135487538, 1), (135487538, 1),  # 0x08136032
]

RAM_REGION = (0x20000000, 0x1000)


class SnapshotIdentityTelemetryTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_k5_c3_")
        self.firmware_path = str(Path(self._temporary.name) / "flip.bin")
        build_flip_image(self.firmware_path)
        self.emulator = build_flip_emulator(self.firmware_path)
        self.manager = self.emulator.branch_snapshot_manager
        self.manager.set_memory_regions([RAM_REGION])
        self.addCleanup(self._teardown)

    def _teardown(self):
        with contextlib.suppress(Exception):
            self.emulator.uc.close()
        self._temporary.cleanup()

    def _capture(self, bb: int):
        return self.manager.save_snapshot(
            self.emulator.uc, bb, bb + 0x10, bb + 4, "EQ"
        )

    @staticmethod
    def _identity(snapshot):
        state = snapshot.external_model_state
        return tuple(state["snapshot_identity"])

    def test_states_differing_only_in_era_get_distinct_quadruples(self):
        first = self._capture(FLIP_BRANCH_BB)
        id_before = self._identity(first)
        self.assertEqual(id_before[0], 0)  # 全新实例 era=0
        self.assertEqual(id_before[1], FLIP_BRANCH_BB)

        # 恢复即开新 era；root_occurrence_local 从被恢复快照的 c3 继承。
        self.assertTrue(
            self.emulator.restore_snapshot_external_state(first)
        )
        second = self._capture(FLIP_BRANCH_BB)
        id_after = self._identity(second)

        self.assertEqual(id_after[0], 1)  # 仅 era 异
        self.assertEqual(id_after[1], id_before[1])
        self.assertNotEqual(id_before, id_after)
        self.assertEqual(
            second.external_model_state["root_occurrence_local"], 1
        )

    def test_same_era_same_bb_occurrence_and_seq_increment(self):
        first = self._capture(FLIP_BRANCH_BB)
        second = self._capture(FLIP_BRANCH_BB)
        third = self._capture(FLIP_BRANCH_BB + 0x100)
        id1, id2, id3 = (
            self._identity(first),
            self._identity(second),
            self._identity(third),
        )
        # 同 era 同 bb：c3 1→2，c2（seq）递增 ⇒ 互异。
        self.assertEqual(id1[0], id2[0])
        self.assertEqual(id1[1], id2[1])
        self.assertEqual(id1[3], 1)
        self.assertEqual(id2[3], 2)
        self.assertGreater(id2[2], id1[2])
        self.assertNotEqual(id1, id2)
        # 同 era 换 bb：(era,bb) 限定键重新计数。
        self.assertEqual(id3[1], FLIP_BRANCH_BB + 0x100)
        self.assertEqual(id3[3], 1)

    def test_legacy_restore_without_identity_keeps_root_none(self):
        first = self._capture(FLIP_BRANCH_BB)
        legacy = dataclasses.replace(first, external_model_state={})
        self.assertTrue(
            self.emulator.restore_snapshot_external_state(legacy)
        )
        after = self._capture(FLIP_BRANCH_BB)
        self.assertIsNone(after.external_model_state["root_occurrence_local"])

    def test_sixteen_miss_records_get_distinct_quadruples(self):
        # 形态 A（产线节奏）：每条记录 = 独立重放任务（restore 根 → 捕获）。
        # 根 = 本 era 内 bb 的第 occ 次捕获；随后 restore 它并捕获继任快照。
        identities = []
        for bb, occ in K0_MISS_RECORDS:
            root = None
            for _ in range(occ):
                root = self._capture(bb)
            self.assertEqual(self._identity(root)[3], occ)
            self.assertTrue(
                self.emulator.restore_snapshot_external_state(root)
            )
            successor = self._capture(bb + 0x1000)
            identities.append(self._identity(successor))
            self.assertEqual(
                successor.external_model_state["root_occurrence_local"], occ
            )
        self.assertEqual(len(identities), 16)
        self.assertEqual(len(set(identities)), 16, "16 条逻辑态四元组必须互异")

        # 形态 B（最紧臂）：16 条全部落在同一 era（无中间 restore），
        # 仅靠 c2/c3 消毒——重复 (bb, occ=1) 对靠递增计数区分。
        tight = []
        for bb, occ in K0_MISS_RECORDS:
            last = None
            for _ in range(occ):
                last = self._capture(bb)
            tight.append(self._identity(last))
        self.assertEqual(len(tight), 16)
        self.assertEqual(len(set(tight)), 16)


if __name__ == "__main__":
    unittest.main()
