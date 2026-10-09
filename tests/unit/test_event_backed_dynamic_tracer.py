#!/usr/bin/env python3
"""Regression tests for MMIO reads serviced before Unicorn executes LDR."""

import unittest

from unicorn.arm_const import UC_ARM_REG_CPSR, UC_ARM_REG_PC

from lsgemu.register_tracer.register_tracer import RegisterTracer


class DummyUC:
    def __init__(self):
        self._next_hook = 1
        self._registers = {
            UC_ARM_REG_CPSR: 1 << 30,
            UC_ARM_REG_PC: 0,
        }

    def hook_add(self, *args, **kwargs):
        handle = self._next_hook
        self._next_hook += 1
        return handle

    def hook_del(self, handle):
        return None

    def reg_read(self, register):
        return int(self._registers.get(register, 0))


class ExternalInputEventSource:
    def __init__(self):
        self.observers = []

    def add_external_input_observer(self, observer):
        if not any(existing is observer for existing in self.observers):
            self.observers.append(observer)
            return True
        return False

    def remove_external_input_observer(self, observer):
        retained = [existing for existing in self.observers if existing is not observer]
        removed = len(retained) != len(self.observers)
        self.observers = retained
        return removed

    def emit(self, event):
        for observer in list(self.observers):
            observer(dict(event))


class EventBackedDynamicTracerTests(unittest.TestCase):
    def test_preloaded_mmio_leaf_survives_load_code_hook(self):
        static_bbs = {
            0x1000: [
                {
                    "address": 0x1000,
                    "mnemonic": "LDRB",
                    "operands": "r0, [r1]",
                    "size": 2,
                },
                {
                    "address": 0x1002,
                    "mnemonic": "CMP",
                    "operands": "r0, #0x41",
                    "size": 2,
                },
                {
                    "address": 0x1004,
                    "mnemonic": "BEQ",
                    "operands": "0x1010",
                    "size": 2,
                },
            ]
        }
        uc = DummyUC()
        source = ExternalInputEventSource()
        tracer = RegisterTracer(
            uc,
            static_bbs,
            external_input_event_source=source,
            dynamic_graph_enabled=True,
        )

        tracer.start_tracing()
        source.emit({
            "event_id": 1,
            "kind": "mmio",
            "pc": 0x1000,
            "address": 0x40001000,
            "size": 1,
            "occurrence": 1,
            "value": 0x41,
            "source": "mapped_mmio_preload",
            "delivery": "mapped_mmio_preload",
        })

        input_node = tracer.dynamic_graph.register_nodes.get("r0")
        self.assertIsNotNone(input_node)
        self.assertEqual(0x40001000, tracer.register_sources["r0"].mmio_address)

        tracer._code_hook(uc, 0x1000, 2, None)
        self.assertEqual(input_node, tracer.dynamic_graph.register_nodes.get("r0"))
        self.assertEqual(1, tracer.external_input_event_stats["preloaded_loads_preserved"])

        tracer._code_hook(uc, 0x1002, 2, None)
        tracer._code_hook(uc, 0x1004, 2, None)
        self.assertEqual(1, len(tracer.dynamic_graph.branch_order))

        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1004,
            occurrence=1,
            target_taken=False,
            max_models=1,
        )
        self.assertEqual("dynamic_models", result.reason)
        self.assertEqual(1, len(result.models))
        self.assertEqual(1, result.models[0].assignments[0].occurrence)
        self.assertNotEqual(0x41, result.models[0].assignments[0].value)

        tracer.stop_tracing()
        self.assertFalse(source.observers)


if __name__ == "__main__":
    unittest.main()
