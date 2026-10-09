"""Regression tests for the run-summary ``execution_intervention_counts`` field.

``IntelligentEmulator.run()`` publishes that field as a ``reason -> count``
mapping, but it was assembled as ``dict(execution_intervention_reasons(...))``
while the helper returns a plain ``list[str]``.  ``dict()`` unpacks each string
as a ``(key, value)`` pair, so the run summary raised ``ValueError`` the first
time a baseline replay actually had an intervention (campaign attempt5).  These
tests pin the old expression's failure mode and the structure the fixed
``dict.fromkeys`` expression must produce, including how
``evidence_contract`` consumes the mapping downstream.
"""

import inspect
import json
import unittest
from collections import Counter

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.evidence_contract import (
    _add_explicit_reasons,
    _intervention_counter_from_mapping,
    execution_intervention_reasons,
)


class InterventionCountsDictFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shape-identical to the per-run delta built inside run(): scalar
        # counters plus nested per-stat deltas, with one counter deliberately
        # unchanged so the delta keeps zero entries like the real thing.
        cls.run_counter_delta = IntelligentEmulator._execution_counter_delta(
            {
                "intervention_count": 5,
                "forced_branch_trace_count": 2,
                "skip_function_stats": {"applied": 3, "installed": 4},
                "svc_stats": {"handled": 1},
            },
            {
                "intervention_count": 2,
                "forced_branch_trace_count": 2,
                "skip_function_stats": {"applied": 1, "installed": 4},
                "svc_stats": {"handled": 0},
            },
        )

    def test_delta_yields_nonempty_plain_string_reasons(self):
        reasons = execution_intervention_reasons(self.run_counter_delta)
        self.assertTrue(reasons)
        self.assertTrue(all(isinstance(reason, str) for reason in reasons))
        self.assertEqual(
            {"loop_intervention", "function_summary_or_skip", "svc_intervention"},
            set(reasons),
        )

    def test_old_dict_wrapping_raises_value_error(self):
        # The pre-fix expression: dict() over a list of plain strings tries to
        # unpack each reason of length != 2 and raises.
        with self.assertRaises(ValueError):
            dict(execution_intervention_reasons(self.run_counter_delta))

    def test_fixed_fromkeys_construction_shape(self):
        reasons = execution_intervention_reasons(self.run_counter_delta)
        counts = dict.fromkeys(reasons, 1)
        self.assertEqual({reason: 1 for reason in reasons}, counts)
        # The run summary is serialized into campaign JSON reports.
        self.assertEqual(counts, json.loads(json.dumps(counts)))

    def test_counts_feed_explicit_reasons_consumer(self):
        counts = dict.fromkeys(
            execution_intervention_reasons(self.run_counter_delta), 1
        )
        counter = Counter()
        _add_explicit_reasons(counter, counts)
        self.assertEqual(Counter(counts), counter)

    def test_counts_support_authoritative_counter_fallback(self):
        # evidence_contract reads ``execution_intervention_counts`` when an
        # adapter emits the authoritative per-run result under the counter
        # name only; the {reason: >=1} shape keeps that extraction working.
        counts = dict.fromkeys(
            execution_intervention_reasons(self.run_counter_delta), 1
        )
        extracted = _intervention_counter_from_mapping({
            "execution_intervention_reasons_authoritative": True,
            "execution_intervention_counts": counts,
        })
        self.assertEqual(set(counts), set(extracted))
        self.assertTrue(all(count >= 1 for count in extracted.values()))

    def test_run_summary_uses_fromkeys_construction(self):
        # The construction lives inline in run(); pin the source shape so the
        # behavioral tests above cannot drift from the production line.
        source = inspect.getsource(IntelligentEmulator.run)
        self.assertRegex(
            source,
            r"'execution_intervention_counts':\s*dict\.fromkeys\(\s*"
            r"execution_intervention_reasons\(run_counter_delta\),\s*1",
        )
        self.assertNotRegex(
            source,
            r"'execution_intervention_counts':\s*dict\(\s*"
            r"execution_intervention_reasons",
        )


if __name__ == "__main__":
    unittest.main()
