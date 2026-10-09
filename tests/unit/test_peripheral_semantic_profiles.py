#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.peripheral_semantic_profiles import (
    CortexMDWTProfile,
    CortexMSCSInterruptProfile,
    SVDRegisterIndex,
    OTGFSResetProfile,
    SemanticProfileRegistry,
    TIM5SemanticProfile,
)
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler


class PeripheralSemanticProfileTests(unittest.TestCase):
    class _Overlay:
        def __init__(self, values):
            self.values = dict(values)

        def resolve_bridge_read(self, pc, address, size):
            key = (int(pc), int(address))
            if key not in self.values:
                return False, 0
            return True, self.values[key]

        def record_bridge_read(self, pc, address, size, value):
            return None

    def test_unmasked_rule_stays_read_pc_scoped(self):
        registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([]))
        registry.learn_polling_exit_rule(
            status_address=0x40000010,
            value=0x1234,
            read_pc=0x08000100,
            constraint_pc=0x08000108,
            loop_head=0x080000f0,
            mask=None,
            source="test",
        )

        self.assertEqual(registry.model_read(0x40000010, 0x08000100, 4, 0), 0x1234)
        self.assertIsNone(registry.model_read(0x40000010, 0x08000200, 4, 0))

    def test_masked_rule_promotes_after_distinct_read_pc_evidence(self):
        registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([]))
        registry.promote_evidence = 2
        for read_pc in (0x08000100, 0x08000200):
            registry.learn_polling_exit_rule(
                status_address=0x40000010,
                value=0x20,
                read_pc=read_pc,
                constraint_pc=read_pc + 4,
                loop_head=read_pc - 0x10,
                mask=0x20,
                source="test",
            )

        self.assertEqual(registry.model_read(0x40000010, 0x08000300, 4, 0x5), 0x25)
        self.assertGreaterEqual(registry.summary()["rules_promoted"], 1)

    def test_svd_status_register_supports_single_evidence_promotion(self):
        svd_text = """<?xml version="1.0"?>
<device>
  <peripherals>
    <peripheral>
      <name>TESTUART</name>
      <baseAddress>0x40001000</baseAddress>
      <registers>
        <register>
          <name>SR</name>
          <addressOffset>0x04</addressOffset>
          <size>32</size>
          <fields>
            <field>
              <name>RDY</name>
              <bitOffset>7</bitOffset>
              <bitWidth>1</bitWidth>
              <description>ready flag</description>
            </field>
          </fields>
        </register>
      </registers>
    </peripheral>
  </peripherals>
</device>
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.svd"
            path.write_text(svd_text)
            registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([str(path)]))
            registry.promote_evidence = 99
            registry.learn_polling_exit_rule(
                status_address=0x40001004,
                value=0x80,
                read_pc=0x08000100,
                constraint_pc=0x08000108,
                loop_head=0x080000f0,
                mask=0x80,
                source="test",
            )
            self.assertEqual(registry.model_read(0x40001004, 0x08000900, 4, 0), 0x80)

    def test_stateful_handler_uses_validated_semantic_rule(self):
        handler = StatefulMMIOHandler()
        handler.handle_write(0x40002000, 0x08000080, 0x1, 4)
        handler.learn_validated_polling_rule(
            mmio_addr=0x40002004,
            value=0x2,
            read_pc=0x08000100,
            constraint_pc=0x08000104,
            loop_head=0x080000f0,
            mask=0x2,
            source="test",
        )

        self.assertEqual(handler.handle_read(0x40002004, 0x08000100, 4), 0x2)

    def test_transaction_rule_requires_recent_matching_write(self):
        registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([]))
        registry.observe_write(0x08000080, 0x40002000, 0x1, 4, 1)
        registry.learn_polling_exit_rule(
            status_address=0x40002004,
            value=0x2,
            read_pc=0x08000100,
            constraint_pc=0x08000104,
            loop_head=0x080000f0,
            mask=0x2,
            source="test",
        )
        registry.events.clear()
        self.assertIsNone(registry.model_read(0x40002004, 0x08000100, 4, 0))
        registry.observe_write(0x08000090, 0x40002000, 0x1, 4, 2)
        self.assertEqual(registry.model_read(0x40002004, 0x08000100, 4, 0), 0x2)
        summary = registry.summary()
        self.assertEqual(summary["rule_kind_counts"]["write_then_status"], 1)
        self.assertTrue(summary["transaction_signatures"])

    def test_builtin_stm32_rcc_profile_sets_ready_bits(self):
        handler = StatefulMMIOHandler()
        handler.handle_write(0x40021000, 0x08000080, 0x01000000, 4)
        value = handler.handle_read(0x40021000, 0x08000084, 4)
        self.assertEqual(value & 0x03000000, 0x03000000)

    def test_builtin_stm32_usart_profile_sets_ack_and_tx_ready(self):
        handler = StatefulMMIOHandler()
        handler.handle_write(0x40004400, 0x08000080, 0xD, 4)  # UE | TE | RE
        value = handler.handle_read(0x4000441C, 0x08000084, 4)
        self.assertTrue(value & (1 << 7))
        self.assertTrue(value & (1 << 6))
        self.assertTrue(value & (1 << 21))
        self.assertTrue(value & (1 << 22))

    def test_uart_overlay_preserves_ready_and_data_side_effects(self):
        handler = StatefulMMIOHandler()
        status_pc = 0x08000084
        data_pc = 0x08000088
        handler.push_mmio_overlay(
            self._Overlay(
                {
                    (status_pc, 0x40004400): (1 << 5) | (1 << 7) | (1 << 6),
                    (data_pc, 0x40004404): 0x5A,
                }
            )
        )

        status = handler.handle_read(0x40004400, status_pc, 4)
        self.assertTrue(status & (1 << 5))
        self.assertTrue(handler._peripheral_input_pending)

        data = handler.handle_read(0x40004404, data_pc, 4)
        self.assertEqual(0x5A, data)
        self.assertFalse(handler._peripheral_input_pending)
        self.assertEqual(0, handler.peek_register_value(0x40004400, 4) & (1 << 5))
        self.assertEqual(0x5A, handler.peek_register_value(0x40004404, 4))
        self.assertEqual(1, handler.peripheral_input_stats["rx_data_reads"])

    def test_uart_data_read_without_ready_is_diagnostic(self):
        handler = StatefulMMIOHandler()

        value = handler.handle_read(
            0x40004404,
            0x08000088,
            4,
        )

        self.assertIsInstance(value, int)
        self.assertEqual(
            1,
            handler.peripheral_input_stats["rx_data_reads_without_ready"],
        )
        self.assertEqual(1, handler.peripheral_input_stats["rx_data_reads"])
        self.assertFalse(handler._peripheral_input_pending)

    def test_uart_repeated_data_read_consumes_each_byte_once(self):
        handler = StatefulMMIOHandler()
        handler.handle_read(0x40004400, 0x08000084, 4)
        first = handler.handle_read(0x40004404, 0x08000088, 4)
        second = handler.handle_read(0x40004404, 0x0800008C, 4)

        self.assertIsInstance(first, int)
        self.assertIsInstance(second, int)
        self.assertEqual(2, handler.peripheral_input_stats["rx_data_reads"])
        self.assertEqual(1, handler.peripheral_input_stats["rx_data_reads_without_ready"])
        self.assertEqual(2, handler.peripheral_input_stats["rx_bytes_consumed"])
        self.assertFalse(handler._peripheral_input_pending)

    def test_builtin_stm32_bxcan_profile_sets_mode_ack(self):
        handler = StatefulMMIOHandler()
        handler.handle_write(0x40006400, 0x08000080, 0x1, 4)
        value = handler.handle_read(0x40006404, 0x08000084, 4)
        self.assertEqual(value & 0x1, 0x1)

    def test_builtin_cortexm_systick_profile_is_vendor_independent(self):
        handler = StatefulMMIOHandler()
        handler.handle_write(0xE000E014, 0x08000080, 0x10, 4)
        value = handler.handle_read(0xE000E014, 0x08000084, 4)
        self.assertEqual(value, 0x10)
        calib = handler.handle_read(0xE000E01C, 0x08000088, 4)
        self.assertEqual(calib, 0x270F)

    def test_svd_write_one_to_clear_and_read_clear_are_applied(self):
        svd_text = """<?xml version="1.0"?>
<device><peripherals><peripheral>
  <name>TEST</name><baseAddress>0x50000000</baseAddress>
  <registers>
    <register><name>FLAGS</name><addressOffset>0x0</addressOffset><size>32</size>
      <fields><field><name>IRQ</name><bitOffset>0</bitOffset><bitWidth>4</bitWidth>
        <modifiedWriteValues>oneToClear</modifiedWriteValues>
      </field></fields>
    </register>
    <register><name>LATCH</name><addressOffset>0x4</addressOffset><size>32</size>
      <fields><field><name>READY</name><bitOffset>0</bitOffset><bitWidth>2</bitWidth>
        <readAction>clear</readAction>
      </field></fields>
    </register>
  </registers>
</peripheral></peripherals></device>
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "effects.svd"
            path.write_text(svd_text)
            handler = StatefulMMIOHandler()
            handler.semantic_profiles = SemanticProfileRegistry(
                svd_index=SVDRegisterIndex([str(path)]),
                causal_context=handler.causal_context,
            )
            handler._set_word_register_value(0x50000000, 0xF)
            handler.handle_write(0x50000000, 0x08000100, 0x5, 4)
            self.assertEqual(0xA, handler.peek_register_value(0x50000000, 4))

            handler._set_word_register_value(0x50000004, 0x3)
            self.assertEqual(0x3, handler.handle_read(0x50000004, 0x08000104, 4))
            self.assertEqual(0, handler.peek_register_value(0x50000004, 4))

    def test_new_conflicting_control_write_invalidates_transaction_rule(self):
        registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([]))
        registry.observe_write(0x08000080, 0x40002000, 0x1, 4, 1)
        registry.learn_polling_exit_rule(
            status_address=0x40002004,
            value=0x2,
            read_pc=0x08000100,
            constraint_pc=0x08000104,
            loop_head=0x080000F0,
            mask=0x2,
            source="test",
        )
        registry.observe_write(0x08000084, 0x40002000, 0x0, 4, 2)
        self.assertIsNone(
            registry.model_read(0x40002004, 0x08000100, 4, 0)
        )
        self.assertEqual(1, registry.summary()["transaction_conflicting_writes"])

    def test_transaction_window_counts_mmio_events_not_instruction_timestamps(self):
        registry = SemanticProfileRegistry(svd_index=SVDRegisterIndex([]))
        registry.observe_write(0x08000080, 0x40002000, 0x1, 4, 1)
        registry.learn_polling_exit_rule(
            status_address=0x40002004,
            value=0x2,
            read_pc=0x08000100,
            constraint_pc=0x08000104,
            loop_head=0x080000F0,
            mask=0x2,
            source="test",
        )
        registry.observe_read(0x080000F8, 0x40002004, 0, 4, 10000)
        self.assertEqual(
            0x2,
            registry.model_read(0x40002004, 0x08000100, 4, 0),
        )


class _TimerHarness:
    """Drive a TIM5 profile through the real MMIO read/write ladder."""

    def __init__(self, us_per_insn=0.05):
        self.handler = StatefulMMIOHandler({}, None)
        self.tim5 = self.handler.semantic_profiles.register_profile(
            TIM5SemanticProfile(us_per_insn=us_per_insn)
        )
        self.scs = self.handler.semantic_profiles.register_profile(
            CortexMSCSInterruptProfile()
        )

    def write(self, offset, value, pc=0x08001000):
        self.handler.handle_write(TIM5SemanticProfile.BASE + offset, pc, value, 4)

    def read(self, offset, pc=0x08001000):
        return self.handler.handle_read(TIM5SemanticProfile.BASE + offset, pc, 4)

    def arm(self, ccr1, *, cr1=1, dier=2, cnt=0):
        self.write(TIM5SemanticProfile.CR1, cr1)
        self.write(TIM5SemanticProfile.DIER, dier)
        self.write(TIM5SemanticProfile.CCR1, ccr1)
        self.write(TIM5SemanticProfile.CNT, cnt)

    def resolve_tim5(self):
        """Profiles are replaced by deepcopy on restore: always re-resolve."""
        return self.handler.semantic_profiles.profile_by_name(TIM5SemanticProfile.name)


class TIM5SemanticProfileTests(unittest.TestCase):
    def test_cnt_is_a_function_of_the_instruction_count(self):
        for chunks in ((1000,), (400, 600), (1,) * 10 + (990,)):
            harness = _TimerHarness()
            harness.arm(ccr1=0x100000)
            for chunk in chunks:
                harness.resolve_tim5().advance_by_instructions(chunk)
            self.assertEqual(
                int(sum(chunks) * harness.tim5.us_per_insn),
                harness.resolve_tim5().cnt,
                f"chunking {chunks} must not change CNT",
            )

    def test_cc1if_latches_on_crossing_and_clears_on_rc_w0(self):
        harness = _TimerHarness()
        harness.arm(ccr1=100)
        tim5 = harness.resolve_tim5()
        tim5.advance_by_instructions(1000)  # 1000 * 0.05 = 50 us -> short of 100
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 0)
        tim5.advance_by_instructions(1200)  # crosses CCR1
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 2)
        self.assertTrue(tim5.due())

        # Firmware clear: SR = ~(SR & DIER) -> bit1 written 0.
        harness.write(TIM5SemanticProfile.SR, 0xFFFFFFFD)
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 0)
        self.assertFalse(tim5.due())
        # It must not re-latch while CNT stays past the compare value.
        tim5.advance_by_instructions(2000)
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 0)

    def test_device_owned_register_beats_the_firmware_write_mirror(self):
        """r31 P2 读路径优先级裁定：器件自有寄存器不被固件写镜像遮蔽。

        The firmware writes TIM5 SR/DIER, so the replay overlay (priority 1)
        holds a mirror of those stores.  Letting the mirror win hid the latched
        CC1IF and silenced the tick handler; the profile must be the authority.
        """
        class _Mirror:
            def __init__(self, values):
                self.values = dict(values)

            def resolve_bridge_read(self, pc, address, size):
                key = (int(pc), int(address))
                if key not in self.values:
                    return False, 0
                return True, self.values[key]

            def record_bridge_read(self, pc, address, size, value):
                return None

            def record_bridge_write(self, pc, address, size, value):
                return None

        status_pc = 0x0813661C
        harness = _TimerHarness()
        harness.handler.push_mmio_overlay(
            _Mirror(
                {
                    (status_pc, TIM5SemanticProfile.BASE + TIM5SemanticProfile.SR): 0xFFFFFFFD,
                    (status_pc, TIM5SemanticProfile.BASE + TIM5SemanticProfile.DIER): 0x0,
                    (status_pc, 0x40000C00 + 0x40): 0xDEADBEEF,
                }
            )
        )
        harness.arm(ccr1=10)
        harness.resolve_tim5().advance_by_instructions(1000)  # CC1IF latched

        self.assertEqual(harness.read(TIM5SemanticProfile.SR, pc=status_pc) & 2, 2)
        self.assertEqual(harness.read(TIM5SemanticProfile.DIER, pc=status_pc), 2)
        # A register the profile does not own still falls through to the mirror.
        self.assertEqual(harness.read(0x40, pc=status_pc), 0xDEADBEEF)

    def test_dier_zero_disarms_the_event(self):
        harness = _TimerHarness()
        harness.arm(ccr1=10)
        harness.write(TIM5SemanticProfile.DIER, 0)  # stStopAlarm
        harness.resolve_tim5().advance_by_instructions(10000)
        self.assertFalse(harness.resolve_tim5().due())
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 0)

    def test_delivery_is_latched_until_the_firmware_acknowledges(self):
        harness = _TimerHarness()
        harness.arm(ccr1=10)
        tim5 = harness.resolve_tim5()
        tim5.advance_by_instructions(1000)
        self.assertTrue(tim5.due())
        tim5.note_delivered(pc=0x081360C4, instructions=1000)
        self.assertFalse(tim5.due())  # hardware does not clear the flag
        self.assertEqual(harness.read(TIM5SemanticProfile.SR) & 2, 2)
        harness.write(TIM5SemanticProfile.SR, 0xFFFFFFFD)
        self.assertFalse(tim5.due())

    def test_dual_rate_keeps_which_event_fires(self):
        """A different rate changes *when* the event fires, not *which* event.

        The CNT overshoot past CCR1 is bounded by one clock granularity, so the
        firing CNT is *not* rate-invariant; the identity of the event is.
        """
        observed = []
        chunk = 500
        for rate in (0.05, 0.5):
            harness = _TimerHarness(us_per_insn=rate)
            harness.arm(ccr1=0x200)
            tim5 = harness.resolve_tim5()
            instructions = 0
            fired = None
            while instructions < 200000:
                tim5.advance_by_instructions(chunk)
                instructions += chunk
                if tim5.due():
                    fired = {
                        "ccr1": tim5.deadline(),
                        "cnt_at_delivery": tim5.cnt,
                        "sr": harness.read(TIM5SemanticProfile.SR) & 2,
                        "instructions": instructions,
                        "granularity_us": int(chunk * rate),
                    }
                    break
            self.assertIsNotNone(fired, f"rate {rate} never fired")
            observed.append(fired)
        # Same event (CCR1 boundary, same SR bit) is reachable at both rates.
        self.assertEqual(observed[0]["ccr1"], observed[1]["ccr1"])
        self.assertEqual(observed[0]["sr"], 0x2)
        self.assertEqual(observed[1]["sr"], 0x2)
        for item in observed:
            self.assertGreaterEqual(item["cnt_at_delivery"], 0x200)
            self.assertLess(
                item["cnt_at_delivery"] - 0x200, item["granularity_us"] + 2
            )
        # Only the timing detail moves: the faster clock needs fewer instructions.
        self.assertLess(observed[1]["instructions"], observed[0]["instructions"])

    def test_snapshot_round_trip_reproduces_the_delivery_log(self):
        """K2 determinism: continuous run vs capture->restore->continue."""

        def script(harness, start, stop):
            log = []
            tim5 = harness.resolve_tim5()
            for step in range(start, stop):
                tim5.advance_by_instructions(700)
                sr = harness.read(TIM5SemanticProfile.SR)
                if tim5.due():
                    log.append(
                        (
                            step,
                            tim5.cnt,
                            tim5.deadline(),
                            tim5.regs.get(TIM5SemanticProfile.DIER, 0),
                            sr & 2,
                            harness.scs.active_exceptions,
                        )
                    )
                    tim5.note_delivered(pc=0x081360C4, instructions=step)
                    harness.scs.push_active(50)
                if step % 5 == 4:
                    harness.write(TIM5SemanticProfile.SR, 0xFFFFFFFD)
                    harness.scs.pop_active(50)
            return log

        continuous = _TimerHarness()
        continuous.arm(ccr1=0x120)
        full_log = script(continuous, 0, 40)

        resumed = _TimerHarness()
        resumed.arm(ccr1=0x120)
        first_half = script(resumed, 0, 20)
        state = resumed.handler.snapshot_runtime_state(copy_values=False)
        script(resumed, 20, 40)  # run past the capture point
        resumed.handler.restore_runtime_state(state)
        # Restore replaces profile objects; the harness must re-resolve.
        self.assertIsNot(
            resumed.tim5, resumed.resolve_tim5(),
            "restore must hand out fresh profile instances",
        )
        second_half = script(resumed, 20, 40)

        self.assertEqual(first_half + second_half, full_log)
        self.assertTrue(full_log, "the fixture must actually deliver something")

    def test_snapshot_alias_counterexample(self):
        """Capture, mutate the live profile list in place, then restore."""
        harness = _TimerHarness()
        harness.arm(ccr1=0x400)
        harness.resolve_tim5().advance_by_instructions(4000)
        harness.scs.push_active(50)
        captured = {
            "cnt": harness.tim5.cnt,
            "time_acc": harness.tim5.time_acc,
            "active": list(harness.scs.active_exceptions),
        }
        state = harness.handler.snapshot_runtime_state(copy_values=False)
        # The compressed path does hand out the live list — that is precisely
        # the hazard the value-captured profile_states side table works around.
        self.assertIs(
            state["semantic_profiles"]["profiles"],
            harness.handler.semantic_profiles.profiles,
        )

        harness.resolve_tim5().advance_by_instructions(9999)
        harness.scs.push_active(51)
        harness.scs.push_active(52)
        self.assertNotEqual(harness.tim5.cnt, captured["cnt"])

        harness.handler.restore_runtime_state(state)
        restored = harness.resolve_tim5()
        restored_scs = harness.handler.semantic_profiles.profile_by_name(
            CortexMSCSInterruptProfile.name
        )
        self.assertEqual(restored.cnt, captured["cnt"])
        self.assertEqual(restored.time_acc, captured["time_acc"])
        self.assertEqual(restored_scs.active_exceptions, captured["active"])


class CortexMSCSInterruptProfileTests(unittest.TestCase):
    def setUp(self):
        self.handler = StatefulMMIOHandler({}, None)
        self.scs = self.handler.semantic_profiles.register_profile(
            CortexMSCSInterruptProfile()
        )

    def _read_icsr(self):
        return self.handler.handle_read(CortexMSCSInterruptProfile.ICSR, 0x0812EDA4, 4)

    def test_retto_base_is_derived_from_the_active_exception_stack(self):
        # Thread mode: no active exception -> RETTOBASE reads 1.
        self.assertTrue(self._read_icsr() & (1 << 11))
        # The delivered IRQ50 is the only active exception -> still 1.
        self.scs.push_active(50)
        self.assertTrue(self._read_icsr() & (1 << 11))
        # A preempting exception makes it 0.
        self.scs.push_active(12)
        self.assertFalse(self._read_icsr() & (1 << 11))
        self.scs.pop_active(12)
        self.assertTrue(self._read_icsr() & (1 << 11))
        self.scs.pop_active(50)
        self.assertTrue(self._read_icsr() & (1 << 11))

    def test_retto_base_is_never_a_hard_coded_constant(self):
        self.scs.push_active(50)
        self.scs.push_active(12)
        # A firmware write of 0xFFFFFFFF to ICSR must not leak a hard-coded bit11.
        self.handler.handle_write(
            CortexMSCSInterruptProfile.ICSR, 0x08001000, 0xFFFFFFFF, 4
        )
        self.assertEqual(self._read_icsr() & (1 << 11), 0)
        self.scs.pop_active(12)
        self.assertEqual(self._read_icsr() & (1 << 11), 1 << 11)

    def test_other_scs_addresses_are_not_claimed(self):
        self.assertIsNone(
            self.handler.semantic_profiles.model_read(
                0xE000E100, 0x08001000, 4, 0, backend=self.handler
            )
        )


class OTGFSResetProfileTests(unittest.TestCase):
    """P5: GRSTCTL AHBIDLE / CSRST handshake, driven only through MMIO reads."""

    GRSTCTL = OTGFSResetProfile.BASE + OTGFSResetProfile.GRSTCTL

    def setUp(self):
        self.handler = StatefulMMIOHandler({}, None)
        self.otgfs = self.handler.semantic_profiles.register_profile(
            OTGFSResetProfile()
        )

    def _read(self, pc=0x081303D0):
        return self.handler.handle_read(self.GRSTCTL, pc, 4)

    def test_ahbidle_reads_one_at_rest(self):
        self.assertTrue(self._read() & OTGFSResetProfile.AHBIDLE)

    def test_usb_lld_start_handshake_completes(self):
        # 0x081303D0: wait for AHBIDLE (bit31) to read 1.
        self.assertTrue(self._read(pc=0x081303D0) & OTGFSResetProfile.AHBIDLE)
        # 0x081303D8: CSRST = 1.
        self.handler.handle_write(self.GRSTCTL, 0x081303D8, 1, 4)
        self.assertTrue(self.otgfs.reset_pending)
        # 0x081303E0: lsls #31 puts CSRST (bit0) into N, so the loop waits for
        # CSRST to self-clear.  The first read reports the reset in flight...
        first = self._read(pc=0x081303E0)
        self.assertTrue(first & OTGFSResetProfile.CSRST)
        self.assertFalse(first & OTGFSResetProfile.AHBIDLE)
        # ...and the following read reports the core idle again.
        second = self._read(pc=0x081303E0)
        self.assertFalse(second & OTGFSResetProfile.CSRST)
        self.assertTrue(second & OTGFSResetProfile.AHBIDLE)
        self.assertFalse(self.otgfs.reset_pending)

    def test_other_otg_addresses_are_not_claimed(self):
        self.assertIsNone(
            self.handler.semantic_profiles.model_read(
                OTGFSResetProfile.BASE + 0x14, 0x081303D0, 4, 0, backend=self.handler
            )
        )


class CortexMDWTProfileTests(unittest.TestCase):
    """The polled-delay time base: CYCCNT must actually advance."""

    CYCCNT = CortexMDWTProfile.DWT_CYCCNT

    def setUp(self):
        self.handler = StatefulMMIOHandler({}, None)
        self.dwt = self.handler.semantic_profiles.register_profile(
            CortexMDWTProfile(cycles_per_insn=1.0)
        )

    def _read(self, pc):
        return self.handler.handle_read(self.CYCCNT, pc, 4)

    def test_polled_delay_loop_terminates(self):
        """Replicates chSysPolledDelayX @ 0x08133CDC: `bhi` while cycles > now-start."""
        cycles = 12
        start = self._read(0x08133CDE)
        iterations = 0
        while True:
            now = self._read(0x08133CE0)
            if not (cycles > (now - start) & 0xFFFFFFFF):
                break
            self.dwt.advance_by_instructions(5)  # loop body
            iterations += 1
            self.assertLess(iterations, 100, "the delay loop must terminate")
        self.assertGreater(iterations, 0)
        self.assertGreaterEqual((now - start) & 0xFFFFFFFF, cycles)

    def test_cyccnt_is_a_function_of_the_instruction_count(self):
        harness = StatefulMMIOHandler({}, None)
        a = harness.semantic_profiles.register_profile(CortexMDWTProfile(cycles_per_insn=1.0))
        b = harness.semantic_profiles.register_profile(CortexMDWTProfile(cycles_per_insn=1.0))
        for chunk in (1000,):
            a.advance_by_instructions(chunk)
        for chunk in (400, 600):
            b.advance_by_instructions(chunk)
        self.assertEqual(a.cycles, b.cycles)

    def test_only_the_dwt_registers_are_owned(self):
        self.assertTrue(self.dwt.owns_read(self.CYCCNT))
        self.assertTrue(self.dwt.owns_read(CortexMDWTProfile.DWT_CTRL))
        self.assertFalse(self.dwt.owns_read(0xE000ED04))


if __name__ == "__main__":
    unittest.main()
