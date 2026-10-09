#!/usr/bin/env python3
"""r7 P1 单测：主干优先调度键（LSGEMU_DFS_TRUNK_FIRST，默认关）。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from lsgemu.historical_runner import HistoricalRunner


def _historical_score(uncovered: int, depth: int, attempts: int = 0, root_key=(0x1000, 1)):
    """按 retry_dispatch_queue_item_score 的历史 17 元组语义构造打分。"""
    return (
        uncovered,          # event_uncovered_successor_count（历史主键）
        2,                  # direct_target_fanout
        0,                  # successor in target_bb_set
        1,                  # successor uncovered
        1,                  # root_attempts == 0
        0,                  # root_new_bbs > 0
        0, 0, 0, 0, 0,      # successor_priority[0..4]
        0,                  # -root_invalid_stops
        0,                  # -successor_zero_new_tasks
        0,                  # -root_zero_productivity_penalty
        -attempts,          # -root_attempts
        -depth,             # -len(prefix)：历史语义同分时浅者胜
        root_key,
    )


class DFSTrunkFirstScheduleKeyTests(unittest.TestCase):
    def test_flag_defaults_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LSGEMU_DFS_TRUNK_FIRST", None)
            self.assertFalse(HistoricalRunner._dfs_trunk_first_enabled())

    def test_flag_reads_env(self):
        for raw, expected in (("1", True), ("true", True), ("0", False), ("", False)):
            with patch.dict(os.environ, {"LSGEMU_DFS_TRUNK_FIRST": raw}):
                self.assertEqual(
                    HistoricalRunner._dfs_trunk_first_enabled(), expected, raw
                )

    def test_trunk_first_off_keeps_historical_tuple(self):
        score = _historical_score(uncovered=5, depth=3)
        self.assertEqual(
            HistoricalRunner._retry_dispatch_order_score(
                score, depth=3, trunk_first=False
            ),
            score,
        )

    def test_trunk_first_promotes_depth_over_uncovered_count(self):
        shallow_rich = _historical_score(uncovered=5, depth=2, root_key=(0x1000, 1))
        deep_poor = _historical_score(uncovered=1, depth=9, root_key=(0x2000, 1))
        shallow_key = HistoricalRunner._retry_dispatch_order_score(
            shallow_rich, depth=2, trunk_first=True
        )
        deep_key = HistoricalRunner._retry_dispatch_order_score(
            deep_poor, depth=9, trunk_first=True
        )
        # 取 max（queue_policy.pop_best）时深度主键胜出。
        self.assertGreater(deep_key, shallow_key)
        self.assertEqual(deep_key[0], 9)
        self.assertEqual(deep_key[1], 1)  # 未覆盖后继数降为次键

    def test_trunk_first_breaks_ties_by_uncovered_then_root_key(self):
        a = HistoricalRunner._retry_dispatch_order_score(
            _historical_score(uncovered=1, depth=4, root_key=(0x1000, 1)),
            depth=4,
            trunk_first=True,
        )
        b = HistoricalRunner._retry_dispatch_order_score(
            _historical_score(uncovered=3, depth=4, root_key=(0x1000, 1)),
            depth=4,
            trunk_first=True,
        )
        self.assertGreater(b, a)  # 同深度 ⇒ 覆盖率增益次键生效

        c = HistoricalRunner._retry_dispatch_order_score(
            _historical_score(uncovered=1, depth=4, root_key=(0x900, 1)),
            depth=4,
            trunk_first=True,
        )
        self.assertGreater(a, c)  # 全同级 ⇒ root_key 决胜仍在尾部

    def test_trunk_first_drops_contradictory_shallow_tail(self):
        score = _historical_score(uncovered=1, depth=7)
        reordered = HistoricalRunner._retry_dispatch_order_score(
            score, depth=7, trunk_first=True
        )
        self.assertNotIn(-7, reordered)
        self.assertEqual(reordered[-1], score[-1])
        # 尾部 -depth 挪为头部主键 depth，元素数不变。
        self.assertEqual(len(reordered), len(score))
        self.assertEqual(reordered[0], 7)


if __name__ == "__main__":
    unittest.main()
