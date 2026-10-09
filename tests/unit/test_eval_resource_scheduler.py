#!/usr/bin/env python3
"""Contracts for resource-aware evaluation admission control."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_rq_experiments as runner


class EvaluationResourceSchedulerTests(unittest.TestCase):
    def test_completed_worker_peak_remains_in_admission_projection(self):
        args = SimpleNamespace(
            scheduler_per_job_memory_gib=2.0,
            scheduler_peak_multiplier=1.25,
            _scheduler_observed_peak_bytes=4 * runner.BYTES_PER_GIB,
        )

        projected = runner.compute_observed_per_job_bytes(args, [])

        self.assertEqual(5 * runner.BYTES_PER_GIB, projected)

    def test_auto_jobs_uses_configured_safe_cap(self):
        args = SimpleNamespace(
            jobs=0,
            max_safe_jobs=8,
            allow_high_jobs=False,
        )

        runner.normalize_jobs(args, target_count=23)

        self.assertEqual(8, args.jobs)

    def test_ramp_adds_one_slot_per_campaign_interval(self):
        args = SimpleNamespace(
            jobs=8,
            scheduler_initial_jobs=2,
            scheduler_ramp_seconds=300,
            scheduler_min_free_memory_gib=19.0,
            scheduler_per_job_memory_gib=1.0,
            scheduler_peak_multiplier=1.25,
            scheduler_max_swap_used_gib=4.0,
        )
        running = [SimpleNamespace(peak_tree_rss_bytes=0) for _ in range(2)]

        with (
            patch.object(runner, "available_memory_bytes", return_value=80 * runner.BYTES_PER_GIB),
            patch.object(runner, "swap_used_bytes", return_value=0),
            patch.object(runner.time, "time", return_value=1000.0),
        ):
            held, reason = runner.can_start_more(args, running, last_start_time=999.0)
            admitted, admitted_reason = runner.can_start_more(
                args,
                running,
                last_start_time=699.0,
            )

        self.assertFalse(held)
        self.assertEqual("ramp_limit", reason)
        self.assertTrue(admitted)
        self.assertEqual("ok", admitted_reason)

    def test_parser_uses_deployment_scheduler_defaults(self):
        args = runner.build_arg_parser().parse_args([])

        self.assertEqual(
            float(runner.SCHEDULER_DEFAULTS["min_free_memory_gib"]),
            args.scheduler_min_free_memory_gib,
        )
        self.assertEqual(
            float(runner.SCHEDULER_DEFAULTS["per_job_memory_gib"]),
            args.scheduler_per_job_memory_gib,
        )


if __name__ == "__main__":
    unittest.main()
