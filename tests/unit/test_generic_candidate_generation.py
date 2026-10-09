#!/usr/bin/env python3
"""Contracts for the without-constraint-guided-candidate-generation ablation.

The generic control arm replaces only the candidate generation step: the same
input occurrence, the same recovered read width, and the same replay
validation are retained, but candidates come from a fixed generic strategy
(boundary values, seed mutation, seeded random sampling) that never reads the
target predicate's constants, masks, or any symbolic problem intermediate.
"""

import unittest
from unittest import mock

from lsgemu.local_constraint_recovery import LocalConstraintRecovery


def byte_poll_recovery(**kwargs):
    """LDRB + TST #0x20 + BNE poll slice: guided mode answers "set bit5"."""
    instructions = [
        {"address": 0x2000, "mnemonic": "LDRB", "operands": "r2, [r3]"},
        {"address": 0x2002, "mnemonic": "TST", "operands": "r2, #0x20"},
        {"address": 0x2004, "mnemonic": "BNE", "operands": "0x2020"},
    ]
    engine_kwargs = dict(
        generic_candidates=True,
        compare_lookup={0x2004: instructions[1]},
    )
    engine_kwargs.update(kwargs)
    return LocalConstraintRecovery(
        {0x2000: instructions},
        instruction_to_bb={0x2000: 0x2000, 0x2002: 0x2000, 0x2004: 0x2000},
        **engine_kwargs,
    )


def equality_compare_recovery(**kwargs):
    """LDRB + CMP #0x37 + BEQ slice: guided mode answers exactly 0x37."""
    instructions = [
        {"address": 0x2000, "mnemonic": "LDRB", "operands": "r2, [r3]"},
        {"address": 0x2002, "mnemonic": "CMP", "operands": "r2, #0x37"},
        {"address": 0x2004, "mnemonic": "BEQ", "operands": "0x2020"},
    ]
    engine_kwargs = dict(
        generic_candidates=True,
        compare_lookup={0x2004: instructions[1]},
    )
    engine_kwargs.update(kwargs)
    return LocalConstraintRecovery(
        {0x2000: instructions},
        instruction_to_bb={0x2000: 0x2000, 0x2002: 0x2000, 0x2004: 0x2000},
        **engine_kwargs,
    )


class GenericCandidateGenerationTests(unittest.TestCase):
    def test_generic_mode_labels_result_and_skips_symbolic_problem(self):
        engine = byte_poll_recovery()
        with mock.patch.object(
            engine,
            "_build_symbolic_problem",
            side_effect=AssertionError("symbolic problem built"),
        ), mock.patch.object(
            engine,
            "_build_direct_problem",
            side_effect=AssertionError("direct problem built"),
        ), mock.patch.object(
            engine,
            "_condition_derived_values",
            side_effect=AssertionError("predicate derivation used"),
        ), mock.patch.object(
            engine,
            "_semantic_boundary_values",
            side_effect=AssertionError("compare-immediate boundaries used"),
        ):
            result = engine.recover_values(
                branch_pc=0x2004,
                branch_condition="BNE",
                target_direction=True,
                read_pc=0x2000,
                primary_values=(0x20,),
                max_values=4,
            )
        self.assertEqual("generic", result.candidate_generation_mode)
        self.assertFalse(result.symbolic_slice)
        self.assertNotEqual("z3", result.solver_backend)
        self.assertTrue(result.hypotheses)

    def test_generic_candidates_do_not_satisfy_the_target_predicate(self):
        # The fixed generic strategy must not know the compare constant: for
        # `CMP #0x37; BEQ` the guided arm answers exactly 0x37, while the
        # width-8 generic boundary band is 0,1,2,0xFF — the predicate
        # constant never appears, and no value satisfies the equality.
        result = equality_compare_recovery().recover_values(
            branch_pc=0x2004,
            branch_condition="BEQ",
            target_direction=True,
            read_pc=0x2000,
            max_values=4,
        )
        values = [item.value for item in result.hypotheses]
        self.assertEqual([0, 1, 2, 0xFF], values)
        self.assertNotIn(0x37, values)
        self.assertTrue(all(value != 0x37 for value in values))

    def test_generic_mode_keeps_recovered_read_width(self):
        result = byte_poll_recovery().recover_values(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            max_values=8,
        )
        self.assertEqual(8, result.source_width)
        self.assertTrue(all(0 <= item.value <= 0xFF for item in result.hypotheses))

    def test_generic_mode_mutates_observed_seeds(self):
        result = byte_poll_recovery().recover_values(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            observed_values=(0x41,),
            max_values=32,
        )
        values = [item.value for item in result.hypotheses]
        # boundary band first, then +/-1, +/-2, +/-(1<<k) of the observed
        # seed (single-bit flips dedup into the +/-(1<<k) family), then
        # byte-level ops and random samples.
        self.assertIn(0x40, values)
        self.assertIn(0x42, values)
        self.assertIn(0x43, values)  # seed with bit0 flipped
        self.assertIn(0xBE, values)  # byte inversion of 0x41
        strategies = {item.strategy for item in result.hypotheses}
        self.assertIn("generic_seed_minus_one", strategies)
        self.assertIn("generic_seed_plus_pow2", strategies)
        self.assertIn("generic_seed_byte_invert", strategies)

    def test_generic_mode_is_deterministic_across_calls(self):
        engine = byte_poll_recovery()
        kwargs = dict(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            observed_values=(0x41,),
            max_values=32,
        )
        first = engine.recover_values(**kwargs)
        second = engine.recover_values(**kwargs)
        self.assertEqual(
            [(item.value, item.strategy) for item in first.hypotheses],
            [(item.value, item.strategy) for item in second.hypotheses],
        )
        strategies = {item.strategy for item in first.hypotheses}
        self.assertIn("generic_random_sample", strategies)

    def test_generic_mode_respects_limit_and_avoid_values(self):
        result = byte_poll_recovery().recover_values(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            avoid_values=(0, 1),
            max_values=3,
        )
        values = [item.value for item in result.hypotheses]
        self.assertEqual(3, len(values))
        self.assertNotIn(0, values)
        self.assertNotIn(1, values)

    def test_default_mode_stays_constraint_guided(self):
        engine = byte_poll_recovery(generic_candidates=None)
        result = engine.recover_values(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            max_values=4,
        )
        self.assertEqual("constraint_guided", result.candidate_generation_mode)
        self.assertTrue(
            all(item.value & 0x20 for item in result.hypotheses),
            "guided mode must still answer the predicate",
        )

    def test_env_switch_enables_generic_mode(self):
        with mock.patch.dict(
            "os.environ", {"LSGEMU_DISABLE_CONSTRAINT_GUIDED_CANDIDATES": "1"}
        ):
            engine = byte_poll_recovery(generic_candidates=None)
            result = engine.recover_values(
                branch_pc=0x2004,
                branch_condition="BNE",
                target_direction=True,
                read_pc=0x2000,
                max_values=4,
            )
        self.assertEqual("generic", result.candidate_generation_mode)

    def test_env_switch_off_by_default(self):
        import os

        env = {
            key: value
            for key, value in os.environ.items()
            if key != "LSGEMU_DISABLE_CONSTRAINT_GUIDED_CANDIDATES"
        }
        with mock.patch.dict("os.environ", env, clear=True):
            engine = byte_poll_recovery(generic_candidates=None)
            self.assertFalse(engine.generic_candidates)


class CandidateGenerationMechanismMetricsTests(unittest.TestCase):
    """The report-level four-metric block aggregates phase metadata."""

    def _metrics(self, phase_map):
        from lsgemu.historical_runner import (
            candidate_generation_mechanism_metrics,
        )

        return candidate_generation_mechanism_metrics(phase_map)

    def test_four_mechanism_metrics_aggregate_across_phases(self):
        phase_map = {
            "path_naturalization": {
                "promoted_paths": 3,
                "candidate_replays": 11,
                "bootstrap_candidate_replays": 5,
                "compound_candidate_replays": 2,
                "failure_category_counts": {
                    "upstream_prefix_interference": 4,
                    "candidate_rejected": 6,
                },
                "local_constraint_recovery": {
                    "calls": 9,
                    "elapsed_seconds": 1.5,
                    "candidate_generation_mode_generic": 9,
                    "hypotheses": 27,
                },
            },
            "stream_input_replay": {
                "candidate_replays": 2,
                "failure_category_counts": {"upstream_prefix_interference": 1},
            },
        }
        metrics = self._metrics(phase_map)
        self.assertEqual(3, metrics["original_target_reached_count"])
        self.assertEqual(20, metrics["candidate_attempt_count"])
        self.assertEqual(5, metrics["early_deviation_count"])
        self.assertEqual(1.5, metrics["recovery_elapsed_seconds"])
        self.assertEqual(
            {
                "note": "denominator includes failed attempts",
                "candidates_generated": 27,
                "recovery_calls": 9,
                "mode_counts": {"constraint_guided": 0, "generic": 9},
            },
            metrics["generation"],
        )

    def test_metrics_absent_metadata_defaults_to_zero(self):
        metrics = self._metrics({})
        self.assertEqual(0, metrics["original_target_reached_count"])
        self.assertEqual(0, metrics["candidate_attempt_count"])
        self.assertEqual(0, metrics["early_deviation_count"])
        self.assertEqual(0.0, metrics["recovery_elapsed_seconds"])


class CandidateGenerationAblationTableTests(unittest.TestCase):
    def test_ablation_variants_registered(self):
        from evaluation.run_rq_experiments import (
            ABLATIONS,
            ABLATION_ENV_OVERRIDES,
            DEFAULT_RQ2_ABLATIONS,
        )

        self.assertEqual(
            ["--disable-constraint-guided-candidates"],
            ABLATIONS["no_constraint_guided_candidates"],
        )
        self.assertEqual([], ABLATIONS["full_llm_off"])
        self.assertIn("full_llm_off", DEFAULT_RQ2_ABLATIONS)
        self.assertIn("no_constraint_guided_candidates", DEFAULT_RQ2_ABLATIONS)
        # LLM must be actually switched off (0 means "no cap" in llm_guide,
        # so the env must carry a real disable switch too).
        self.assertEqual("1", ABLATION_ENV_OVERRIDES["full_llm_off"]["LSGEMU_DISABLE_LLM"])
        self.assertEqual(
            "0", ABLATION_ENV_OVERRIDES["full_llm_off"]["LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS"]
        )


class LlmOffSwitchTests(unittest.TestCase):
    def test_env_report_cards_disable_llm(self):
        from lsgemu.cached_interleaved_runner import llm_disabled_by_env

        for value in ("1", "true", "yes", "on"):
            with mock.patch.dict("os.environ", {"LSGEMU_DISABLE_LLM": value}):
                self.assertTrue(llm_disabled_by_env())
        for value in ("", "0", "false", "off"):
            with mock.patch.dict("os.environ", {"LSGEMU_DISABLE_LLM": value}):
                self.assertFalse(llm_disabled_by_env())


if __name__ == "__main__":
    unittest.main()
