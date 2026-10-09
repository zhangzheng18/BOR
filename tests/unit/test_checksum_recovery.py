#!/usr/bin/env python3
from __future__ import annotations

import unittest

from lsgemu.checksum_recovery import (
    ChecksumInputSite,
    generate_checksum_hypotheses,
)


class ChecksumRecoveryTests(unittest.TestCase):
    @staticmethod
    def _site(index: int, value: int) -> ChecksumInputSite:
        return ChecksumInputSite(
            site_index=index,
            kind="external_memory",
            address=0x20000000 + index,
            read_pc=0x1000 + index * 2,
            occurrence=1,
            width=8,
            observed_value=value,
            event_index=index + 1,
            trace_event_id=index + 1,
        )

    def test_generates_suffix_xor_and_sum_candidates(self):
        hypotheses = generate_checksum_hypotheses([
            self._site(0, 0x10),
            self._site(1, 0x11),
            self._site(2, 0xFF),
        ], max_hypotheses=16)
        by_algorithm = {
            item.algorithm: item
            for item in hypotheses
            if item.field_location == "suffix"
        }
        self.assertEqual(0x01, by_algorithm["xor8"].assignments[0].value)
        self.assertEqual(0x21, by_algorithm["sum8"].assignments[0].value)

    def test_candidates_are_bounded_and_only_change_checksum_sites(self):
        hypotheses = generate_checksum_hypotheses(
            [self._site(index, index) for index in range(12)],
            max_hypotheses=3,
        )
        self.assertLessEqual(len(hypotheses), 3)
        self.assertTrue(hypotheses)
        self.assertEqual({1, 2, 4}, {
            hypothesis.checksum_width for hypothesis in hypotheses
        })
        for hypothesis in hypotheses:
            self.assertTrue(hypothesis.assignments)
            self.assertTrue(all(
                assignment.site_index in {8, 9, 10, 11}
                for assignment in hypothesis.assignments
            ))

    def test_unchanged_checksum_is_not_emitted_as_assignment(self):
        hypotheses = generate_checksum_hypotheses([
            self._site(0, 0x12),
            self._site(1, 0x34),
            self._site(2, 0x26),
        ], max_hypotheses=16)
        xor_candidates = [
            item for item in hypotheses
            if item.algorithm == "xor8" and item.field_location == "suffix"
        ]
        self.assertFalse(xor_candidates)

    def test_selection_offset_rotates_through_bounded_algorithm_schedule(self):
        sites = [self._site(index, index + 1) for index in range(12)]
        first = generate_checksum_hypotheses(
            sites,
            max_hypotheses=3,
            selection_offset=0,
        )
        second = generate_checksum_hypotheses(
            sites,
            max_hypotheses=3,
            selection_offset=len(first),
        )
        first_identity = {
            (item.algorithm, item.field_location, item.strategy)
            for item in first
        }
        second_identity = {
            (item.algorithm, item.field_location, item.strategy)
            for item in second
        }
        self.assertTrue(first_identity)
        self.assertTrue(second_identity)
        self.assertTrue(first_identity.isdisjoint(second_identity))


if __name__ == "__main__":
    unittest.main()
