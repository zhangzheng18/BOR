#!/usr/bin/env python3
from __future__ import annotations

import unittest

from lsgemu.causal_context import CausalExecutionContext, compare_branch_contexts


class CausalExecutionContextTests(unittest.TestCase):
    def test_snapshot_restore_preserves_context_identity(self):
        context = CausalExecutionContext(max_transitions=32)
        context.record_mmio_write(0x1000, 0xE000E100, 0x4, 4)
        context.record_irq_pending(2, pc=0x1002)
        context.record_irq_deliver(2, pc=0x1004)
        context.record_input_ready("uart0", pc=0x1006)
        context.record_input_consume("uart0", pc=0x1008, value=0x41)
        context.record_dma_start("dma0", pc=0x100A, count=16)

        expected = context.branch_signature()
        snapshot = context.snapshot_runtime_state()
        context.record_irq_return(2, pc=0x100C)
        context.record_dma_complete("dma0", pc=0x100E)
        context.restore_runtime_state(snapshot)

        self.assertEqual(expected, context.branch_signature())
        self.assertEqual([2], context.irq_stack)
        self.assertEqual("active", context.dma_channels["dma0"].phase)

    def test_bounded_transition_log_keeps_monotonic_counts(self):
        context = CausalExecutionContext(max_transitions=16)
        for index in range(40):
            context.record_mmio_read(index, 0x40000000, index, 4, source="test")
        self.assertEqual(16, len(context.transitions))
        self.assertEqual(24, context.dropped_transitions)
        self.assertEqual(40, context.event_counts["mmio_read"])

    def test_context_comparison_rejects_irq_or_task_lineage_change(self):
        control = CausalExecutionContext()
        candidate = CausalExecutionContext()
        self.assertTrue(compare_branch_contexts(
            control.branch_signature(), candidate.branch_signature()
        )["compatible"])

        candidate.set_active_task("worker", pc=0x2000)
        mismatch = compare_branch_contexts(
            control.branch_signature(), candidate.branch_signature()
        )
        self.assertFalse(mismatch["compatible"])
        self.assertIn("active_task", mismatch["hard_mismatches"])

    def test_input_lineage_ignores_candidate_value_but_not_delivery_site(self):
        control = CausalExecutionContext()
        candidate = CausalExecutionContext()
        for context, value in ((control, 0x10), (candidate, 0x20)):
            context.record_external_input(
                kind="mmio",
                pc=0x2100,
                address=0x40001000,
                occurrence=1,
                trace_event_id=7,
                value=value,
                size=1,
            )
        self.assertTrue(compare_branch_contexts(
            control.branch_signature(), candidate.branch_signature()
        )["compatible"])

        different_site = CausalExecutionContext()
        different_site.record_external_input(
            kind="mmio",
            pc=0x2100,
            address=0x40001004,
            occurrence=1,
            trace_event_id=7,
            value=0x20,
            size=1,
        )
        mismatch = compare_branch_contexts(
            control.branch_signature(), different_site.branch_signature()
        )
        self.assertFalse(mismatch["compatible"])
        self.assertIn("input_lineage_sha256", mismatch["hard_mismatches"])

    def test_pending_event_state_is_hard_context_and_ready_is_coalesced(self):
        control = CausalExecutionContext()
        candidate = CausalExecutionContext()
        candidate.record_input_ready("uart0", pc=0x2200)
        candidate.record_input_ready("uart0", pc=0x2202)
        self.assertEqual(1, candidate.sequence)
        self.assertEqual(1, candidate.event_counts["input_ready_reobserved"])
        mismatch = compare_branch_contexts(
            control.branch_signature(), candidate.branch_signature()
        )
        self.assertFalse(mismatch["compatible"])
        self.assertIn("pending_events_sha256", mismatch["hard_mismatches"])

    def test_pending_delivery_removes_repeated_irq_and_task_state(self):
        context = CausalExecutionContext()
        context.record_irq_pending(5, pc=0x2300)
        context.record_irq_pending(5, pc=0x2302)
        self.assertEqual(1, len(context.pending_events))
        context.record_irq_deliver(5, pc=0x2304)
        self.assertFalse(context.pending_events)

        context.record_task_wakeup("worker", pc=0x2306)
        context.record_task_wakeup("worker", pc=0x2308)
        self.assertEqual(1, len(context.pending_events))
        context.set_active_task("worker", pc=0x230A)
        self.assertFalse(context.pending_events)

    def test_missing_context_is_backward_compatible_but_audited(self):
        result = compare_branch_contexts(None, {"active_task": "reset"})
        self.assertFalse(result["known"])
        self.assertTrue(result["compatible"])

        incomplete = compare_branch_contexts(
            {"schema": CausalExecutionContext.SCHEMA, "active_task": "reset"},
            CausalExecutionContext().branch_signature(),
        )
        self.assertFalse(incomplete["known"])
        self.assertTrue(incomplete["compatible"])
        self.assertEqual("context_signature_incomplete", incomplete["reason"])


if __name__ == "__main__":
    unittest.main()
