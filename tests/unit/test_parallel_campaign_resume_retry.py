import tempfile
import unittest
from types import SimpleNamespace

from lsgemu.run_elfmultifuzz_interleaved_strict_parallel_campaign import (
    should_low_coverage_retry,
    should_resume_skip_report,
)


def _long_budget_args(**overrides):
    base = dict(
        firmware_minutes=1440,
        low_coverage_retry_minutes=0,
        low_coverage_retry_rate=0.8,
        low_coverage_retry_min_elapsed_ratio=0.85,
        low_coverage_retry_max_attempts=0,
        retry_full_duration_low_coverage=False,
        disable_low_coverage_retry=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _watchdog_stopped_result(**overrides):
    base = dict(
        status="completed_strict_checkpoint_union",
        # MCUdatabase-shaped: reachable/valid rates are 0.0 because the
        # valid-BB denominator is not wired up on this chain.
        valid_entry_plus_vector_reachable_coverage_rate=0.0,
        valid_coverage_rate=0.0,
        execution_time_seconds=16_200.0,  # 4.5h of a 24h budget -> ratio 0.19
        stall_watchdog_stop_reason="stall_no_new_bb_watchdog",
        stall_watchdog=None,
    )
    base.update(overrides)
    return base


class ParallelCampaignLowCoverageRetryWatchdogTest(unittest.TestCase):
    def test_watchdog_early_stop_never_feeds_low_coverage_retry(self):
        args = _long_budget_args()
        result = _watchdog_stopped_result()
        retry, diagnostics = should_low_coverage_retry(
            args=args,
            result=result,
            retry_count=0,
            resource_failure=False,
            segment_index=0,
            total_segments=1,
        )
        self.assertFalse(retry)
        self.assertTrue(diagnostics["watchdog_stopped"])
        # Without the short-circuit this exact case WOULD have retried
        # (short elapsed ratio + zero reachable rate): the suppression, not
        # an unrelated gate, is what stops the retry.
        self.assertTrue(diagnostics["would_retry_without_watchdog"])
        self.assertAlmostEqual(0.1875, diagnostics["elapsed_ratio"], places=4)

    def test_same_shape_without_watchdog_still_retries(self):
        args = _long_budget_args()
        result = _watchdog_stopped_result(
            stall_watchdog_stop_reason=None,
        )
        retry, diagnostics = should_low_coverage_retry(
            args=args,
            result=result,
            retry_count=0,
            resource_failure=False,
            segment_index=0,
            total_segments=1,
        )
        self.assertTrue(retry)
        self.assertFalse(diagnostics["watchdog_stopped"])

    def test_full_duration_flag_bypass_still_blocked_after_watchdog(self):
        args = _long_budget_args(retry_full_duration_low_coverage=True)
        result = _watchdog_stopped_result()
        retry, _ = should_low_coverage_retry(
            args=args,
            result=result,
            retry_count=0,
            resource_failure=False,
            segment_index=0,
            total_segments=1,
        )
        self.assertFalse(retry)

    def test_auto_full_duration_bypass_still_blocked_after_watchdog(self):
        # Short-budget campaigns enable the full-duration auto bypass by
        # default; a watchdog stop must suppress it just the same.
        import os
        from unittest.mock import patch

        args = _long_budget_args(firmware_minutes=20)
        result = _watchdog_stopped_result(execution_time_seconds=240.0)
        with patch.dict(
            os.environ,
            {"LSGEMU_AUTO_FULL_DURATION_LOW_COVERAGE_RETRY": "1"},
            clear=False,
        ):
            from lsgemu.run_elfmultifuzz_interleaved_strict_parallel_campaign import (
                auto_retry_full_duration_low_coverage,
            )

            self.assertTrue(auto_retry_full_duration_low_coverage(args))
            retry, _ = should_low_coverage_retry(
                args=args,
                result=result,
                retry_count=0,
                resource_failure=False,
                segment_index=0,
                total_segments=1,
            )
        self.assertFalse(retry)

    def test_watchdog_status_dict_alone_blocks_retry(self):
        # Union paths can rebuild the result without the flat stop-reason
        # field; the embedded watchdog status must be honored too.
        result = _watchdog_stopped_result(
            stall_watchdog_stop_reason=None,
            stall_watchdog={"stop_requested": True, "stop_reason": "x"},
        )
        retry, diagnostics = should_low_coverage_retry(
            args=_long_budget_args(),
            result=result,
            retry_count=0,
            resource_failure=False,
            segment_index=0,
            total_segments=1,
        )
        self.assertFalse(retry)
        self.assertTrue(diagnostics["watchdog_stopped"])

    def test_non_completed_status_or_resource_failure_unchanged(self):
        result = _watchdog_stopped_result(status="failed")
        retry, _ = should_low_coverage_retry(
            args=_long_budget_args(),
            result=result,
            retry_count=0,
            resource_failure=False,
            segment_index=0,
            total_segments=1,
        )
        self.assertFalse(retry)
        retry, _ = should_low_coverage_retry(
            args=_long_budget_args(),
            result=_watchdog_stopped_result(),
            retry_count=0,
            resource_failure=True,
            segment_index=0,
            total_segments=1,
        )
        self.assertFalse(retry)


class ParallelCampaignResumeRetryTest(unittest.TestCase):
    def test_resume_skip_report_is_limited_to_initial_attempt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            report_path = Path(tmp_dir) / "firmware_interleaved_report.json"
            report_path.write_text("{}", encoding="utf-8")

            self.assertTrue(
                should_resume_skip_report(
                    resume=True,
                    retry_count=0,
                    segment_index=0,
                    report_path=report_path,
                )
            )
            self.assertFalse(
                should_resume_skip_report(
                    resume=True,
                    retry_count=1,
                    segment_index=0,
                    report_path=report_path,
                )
            )
            self.assertFalse(
                should_resume_skip_report(
                    resume=True,
                    retry_count=0,
                    segment_index=1,
                    report_path=report_path,
                )
            )
            self.assertFalse(
                should_resume_skip_report(
                    resume=False,
                    retry_count=0,
                    segment_index=0,
                    report_path=report_path,
                )
            )


if __name__ == "__main__":
    unittest.main()
