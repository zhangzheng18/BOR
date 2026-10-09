#!/usr/bin/env python3
"""Contracts for input provenance, constraint identity, and fact scope."""

from __future__ import annotations

import unittest

from lsgemu.constraint_utils import constraint_identity
from lsgemu.path_naturalization import (
    NaturalizationFact,
    PathNaturalizationLedger,
    merge_constraint_sets,
)
from lsgemu.register_tracer.register_tracer import RegisterSource, RegisterTracer
from lsgemu.runner_models import BranchConstraintCandidate


class DummyUC:
    pass


def candidate(*, occurrence: int = 1, width: int = 32, group: str = "g"):
    return BranchConstraintCandidate(
        constraint_type="mmio",
        address=0x40001000,
        value=1,
        read_pc=0x08000020,
        read_occurrence=occurrence,
        input_kind="mmio",
        externally_controllable=True,
        dependency_group=group,
        width=width,
    )


class ProvenanceScopeContractTests(unittest.TestCase):
    def test_unknown_destination_write_kills_stale_source(self):
        tracer = RegisterTracer(DummyUC(), {})
        tracer.register_sources["r0"] = RegisterSource(
            register="r0",
            source_type="mmio",
            mmio_address=0x40001000,
        )
        tracer.reg_mmio_map["r0"] = 0x40001000

        tracer._propagate_register_sources({
            "mnemonic": "MUL",
            "operands": "r0, r1, r2",
        })

        self.assertNotIn("r0", tracer.register_sources)
        self.assertNotIn("r0", tracer.reg_mmio_map)
        self.assertEqual(
            1,
            tracer.get_statistics()["provenance_kill_stats"]["unknown_destination_write"],
        )

    def test_non_writing_compare_preserves_source(self):
        tracer = RegisterTracer(DummyUC(), {})
        source = RegisterSource(register="r0", source_type="mmio", mmio_address=0x40001000)
        tracer.register_sources["r0"] = source

        tracer._propagate_register_sources({
            "mnemonic": "CMP",
            "operands": "r0, #1",
        })

        self.assertIs(source, tracer.register_sources["r0"])

    def test_indirect_load_preserves_pointer_input_as_opaque(self):
        tracer = RegisterTracer(DummyUC(), {})
        pointer = RegisterSource(
            register="r1",
            source_type="mmio",
            mmio_address=0x40002000,
            source_pc=0x08000100,
            dependency_sites=(("mmio", 0x40002000, 0x08000100, 0x08000100),),
        )
        tracer.register_sources["r1"] = pointer
        instruction = {
            "mnemonic": "LDRB",
            "operands": "r0, [r1, #4]",
        }
        self.assertEqual(["r1"], tracer._memory_address_registers(instruction))
        combined = tracer._combine_indirect_memory_source(
            "r0",
            None,
            [pointer],
            "LDRB r0, [r1, #4]",
            access_pc=0x08000104,
            address=0x20001004,
            size=1,
            operation="indirect_load",
        )
        self.assertIsNotNone(combined)
        self.assertTrue(combined.opaque_dependency)
        self.assertFalse(combined.causal_complete)
        self.assertEqual(
            {("mmio", 0x40002000)},
            {
                (kind, address)
                for kind, address, _source_pc, _origin_pc
                in tracer._dependency_sites_for_source(combined)
            },
        )

    def test_constraint_identity_tracks_runtime_delivery_not_provenance_labels(self):
        base = {
            "type": "mmio",
            "read_pc": "0x08000020",
            "address": "0x40001000",
            "constraint_pc": "0x08000024",
            "read_occurrence": 1,
            "input_kind": "mmio",
            "dependency_group": "g1",
            "width": 8,
        }
        occurrence_variant = dict(base)
        occurrence_variant["read_occurrence"] = 2
        self.assertNotEqual(
            constraint_identity(base),
            constraint_identity(occurrence_variant),
        )

        for key, value in (
            ("input_kind", "stream"),
            ("dependency_group", "g2"),
            ("width", 16),
            ("constraint_pc", "0x08000080"),
        ):
            item = dict(base)
            item[key] = value
            self.assertEqual(constraint_identity(base), constraint_identity(item))

        merged, reason = merge_constraint_sets(
            (candidate(width=8),),
            (candidate(width=16),),
        )
        self.assertIsNone(reason)
        self.assertEqual(1, len(merged or ()))

        conflicting, reason = merge_constraint_sets(
            (candidate(width=8),),
            (
                BranchConstraintCandidate(
                    **{
                        **candidate(width=16, group="other").__dict__,
                        "value": 2,
                    }
                ),
            ),
        )
        self.assertIsNone(conflicting)
        self.assertEqual("conflicting_constraint_values", reason)

        global_first = dict(base, read_pc=None, read_occurrence=1)
        global_second = dict(base, read_pc=None, read_occurrence=9)
        self.assertEqual(
            constraint_identity(global_first),
            constraint_identity(global_second),
        )

    def test_nonempty_fact_scope_must_match_obligation_prefix(self):
        ledger = PathNaturalizationLedger()
        obligation = ledger.register_obligation(
            (
                ((0x1000, 1), False),
                ((0x2000, 1), True),
            ),
            target_bbs={0x2100},
        )
        self.assertIsNotNone(obligation)
        ledger.register_fact(NaturalizationFact(
            branch_key=(0x2000, 1),
            choice=True,
            constraints=(candidate(),),
            scope_signature=(((0x3000, 1), False),),
            paired_control=True,
            candidate_consumed=True,
            local_success=True,
        ))

        extensions, reason = ledger.candidate_extensions(
            obligation,
            (0x2000, 1),
            True,
        )
        self.assertEqual([], extensions)
        self.assertEqual("scope_mismatch", reason)

        ledger.register_fact(NaturalizationFact(
            branch_key=(0x2000, 1),
            choice=True,
            constraints=(candidate(group="global"),),
            scope_signature=tuple(),
            paired_control=True,
            candidate_consumed=True,
            local_success=True,
        ))
        extensions, reason = ledger.candidate_extensions(
            obligation,
            (0x2000, 1),
            True,
        )
        self.assertEqual("ready", reason)
        self.assertTrue(extensions)

    def test_fact_scope_matches_concrete_discovery_prefix(self):
        ledger = PathNaturalizationLedger()
        obligation = ledger.register_obligation(
            (((0x3000, 1), True),),
            target_bbs={0x3100},
            discovery_trace_signature=(
                ((0x1000, 1), False),
                ((0x3000, 1), True),
            ),
        )
        self.assertIsNotNone(obligation)
        ledger.register_fact(NaturalizationFact(
            branch_key=(0x3000, 1),
            choice=True,
            constraints=(candidate(),),
            scope_signature=(((0x1000, 1), False),),
            paired_control=True,
            candidate_consumed=True,
            local_success=True,
        ))
        extensions, reason = ledger.candidate_extensions(
            obligation,
            (0x3000, 1),
            True,
        )
        self.assertEqual("ready", reason)
        self.assertEqual(1, len(extensions))


if __name__ == "__main__":
    unittest.main()
