#!/usr/bin/env python3
"""Regression tests for obligation-scoped local constraint recovery."""

import unittest
from unittest import mock

from lsgemu.local_constraint_recovery import (
    LocalConstraintRecovery,
    classify_replay_failure,
)


def recovery_for(instructions):
    bb_start = int(instructions[0]["address"])
    branch_pc = int(instructions[-1]["address"])
    compare = instructions[-2]
    return LocalConstraintRecovery(
        {bb_start: instructions},
        instruction_to_bb={int(item["address"]): bb_start for item in instructions},
        compare_lookup={branch_pc: compare},
    )


class LocalConstraintRecoveryTests(unittest.TestCase):
    def test_recovers_raw_value_through_shift_and_mask(self):
        instructions = [
            {"address": 0x1000, "mnemonic": "LDR", "operands": "r0, [r1]"},
            {"address": 0x1002, "mnemonic": "LSR", "operands": "r0, r0, #4"},
            {"address": 0x1004, "mnemonic": "AND", "operands": "r0, r0, #3"},
            {"address": 0x1006, "mnemonic": "CMP", "operands": "r0, #2"},
            {"address": 0x1008, "mnemonic": "BEQ", "operands": "0x1020"},
        ]
        result = recovery_for(instructions).recover_values(
            branch_pc=0x1008,
            branch_condition="BEQ",
            target_direction=True,
            read_pc=0x1000,
            primary_values=(2,),
            max_values=4,
        )
        values = [item.value for item in result.hypotheses]
        self.assertTrue(result.symbolic_slice)
        self.assertNotIn(2, values)
        self.assertIn(0x20, values)
        self.assertTrue(all(((value >> 4) & 3) == 2 for value in values))

    def test_byte_status_mask_candidates_respect_read_width(self):
        instructions = [
            {"address": 0x2000, "mnemonic": "LDRB", "operands": "r2, [r3]"},
            {"address": 0x2002, "mnemonic": "TST", "operands": "r2, #0x20"},
            {"address": 0x2004, "mnemonic": "BNE", "operands": "0x2020"},
        ]
        result = recovery_for(instructions).recover_values(
            branch_pc=0x2004,
            branch_condition="BNE",
            target_direction=True,
            read_pc=0x2000,
            primary_values=(1,),
            max_values=4,
        )
        values = [item.value for item in result.hypotheses]
        self.assertEqual(8, result.source_width)
        self.assertIn(0x20, values)
        self.assertNotIn(1, values)
        self.assertTrue(all(0 <= value <= 0xFF for value in values))
        self.assertTrue(all(value & 0x20 for value in values))

    def test_failure_classification_separates_site_and_value_failures(self):
        missing = classify_replay_failure(
            {"reason": "candidate_not_consumed"},
            {
                "all_constraint_reads_matched": False,
                "matched_address_reads": 0,
                "matched_constraint_reads": 0,
            },
        )
        self.assertEqual(
            "input_site_not_reached_or_context_missing",
            missing.category,
        )
        self.assertFalse(missing.retry_values)
        self.assertTrue(missing.retry_site)

        wrong_occurrence = classify_replay_failure(
            {"reason": "candidate_not_consumed"},
            {
                "all_constraint_reads_matched": False,
                "matched_address_reads": 1,
                "matched_requested_occurrences": 0,
                "matched_constraint_reads": 0,
            },
        )
        self.assertEqual("wrong_read_occurrence", wrong_occurrence.category)
        self.assertFalse(wrong_occurrence.retry_values)
        self.assertTrue(wrong_occurrence.retry_site)

        delivery_failure = classify_replay_failure(
            {"reason": "candidate_not_consumed"},
            {
                "all_constraint_reads_matched": False,
                "matched_address_reads": 1,
                "matched_requested_occurrences": 1,
                "matched_constraint_reads": 0,
                "value_delivery_mismatches": 1,
            },
        )
        self.assertEqual(
            "configured_value_not_materialized",
            delivery_failure.category,
        )

        consumed = classify_replay_failure(
            {
                "reason": "wrong_natural_choice",
                "result_match": {"mismatch_reason": "wrong_natural_choice"},
            },
            {
                "all_constraint_reads_matched": True,
                "matched_address_reads": 1,
                "matched_constraint_reads": 1,
            },
        )
        self.assertEqual("consumed_but_predicate_unchanged", consumed.category)
        self.assertTrue(consumed.retry_values)
        self.assertTrue(consumed.retry_compound)

    def test_missing_z3_falls_back_to_bounded_concrete_values(self):
        instructions = [
            {"address": 0x3000, "mnemonic": "LDR", "operands": "r0, [r1]"},
            {"address": 0x3002, "mnemonic": "CMP", "operands": "r0, #7"},
            {"address": 0x3004, "mnemonic": "BEQ", "operands": "0x3020"},
        ]
        with mock.patch("lsgemu.local_constraint_recovery.z3", None):
            result = recovery_for(instructions).recover_values(
                branch_pc=0x3004,
                branch_condition="BEQ",
                target_direction=True,
                read_pc=0x3000,
                primary_values=(7,),
                max_values=3,
            )
        self.assertFalse(result.symbolic_slice)
        self.assertEqual("unavailable", result.solver_backend)
        self.assertEqual(7, result.hypotheses[0].value)


class ChainedCarryFlagTests(unittest.TestCase):
    """r19：SBCS/ADCS 链式借位/进位（64 位比较惯用语 subs;sbcs 两站式）。"""

    def test_sbcs_chained_borrow_respects_first_subs_carry(self):
        # movs r1,#0 置 Z/N；subs r0,r0,#0x10 的 C=r0>=0x10；sbcs r1,r1,r0? 用
        # 经典两站式：subs + sbcs（借位 = r0 < imm）→ bcc 取「首个借位值 imm-1」
        instructions = [
            {"address": 0x4000, "mnemonic": "LDR", "operands": "r0, [r4]", "size": 2},
            {"address": 0x4002, "mnemonic": "MOVS", "operands": "r1, #0", "size": 2},
            {"address": 0x4004, "mnemonic": "SUBS", "operands": "r0, r0, #0x10", "size": 2},
            {"address": 0x4006, "mnemonic": "SBCS", "operands": "r1, r1, #0", "size": 4},
            {"address": 0x4008, "mnemonic": "BCC", "operands": "0x4020", "size": 2},
        ]
        result = recovery_for(instructions).recover_values(
            branch_pc=0x4008,
            branch_condition="BCC",
            target_direction=True,
            read_pc=0x4000,
            max_values=4,
        )
        self.assertTrue(result.symbolic_slice)
        self.assertTrue(result.hypotheses)
        # BCC(taken) ⇔ 借位 ⇔ r0 < 0x10（sbcs 链式读 subs 的 C 槽）
        for item in result.hypotheses:
            self.assertLess(item.value, 0x10, hex(item.value))

    def test_adcs_chained_carry_from_subs(self):
        # subs 置 C = UGE(r0,0x10)（无借位）；mvns r1,#0 → r1=0xFFFFFFFF；
        # adcs r1,r1,#0：0xFFFFFFFF+0+cin 的进位恰等于 cin，C 经 adcs 链式
        # 透传 → bcs 读到 subs 的 C。
        instructions = [
            {"address": 0x5000, "mnemonic": "LDR", "operands": "r0, [r4]", "size": 2},
            {"address": 0x5002, "mnemonic": "MVNS", "operands": "r1, #0", "size": 2},
            {"address": 0x5004, "mnemonic": "SUBS", "operands": "r0, r0, #0x10", "size": 2},
            {"address": 0x5006, "mnemonic": "ADCS", "operands": "r1, r1, #0", "size": 4},
            {"address": 0x5008, "mnemonic": "BCS", "operands": "0x5020", "size": 2},
        ]
        result = recovery_for(instructions).recover_values(
            branch_pc=0x5008,
            branch_condition="BCS",
            target_direction=True,
            read_pc=0x5000,
            max_values=4,
        )
        self.assertTrue(result.symbolic_slice)
        self.assertTrue(result.hypotheses)
        # BCS(taken) ⇔ C 经 adcs(0xFFFFFFFF+0+cin) 透传 ⇔ r0 >= 0x10
        for item in result.hypotheses:
            self.assertGreaterEqual(item.value, 0x10, hex(item.value))

    def test_sbcs_with_unknown_carry_slot_rejected(self):
        # C 槽读前未知（无前置算术写 C）→ 链式结果不可模型，拒绝而非猜。
        instructions = [
            {"address": 0x6000, "mnemonic": "LDR", "operands": "r0, [r4]", "size": 2},
            {"address": 0x6002, "mnemonic": "SBCS", "operands": "r1, r1, r0", "size": 4},
            {"address": 0x6006, "mnemonic": "BCS", "operands": "0x6020", "size": 2},
        ]
        result = recovery_for(instructions).recover_values(
            branch_pc=0x6006,
            branch_condition="BCS",
            target_direction=True,
            read_pc=0x6000,
            max_values=4,
        )
        self.assertFalse(result.symbolic_slice)
        self.assertEqual(result.reason, "unsupported_compare_slice")


class SeedTableTests(unittest.TestCase):
    """r19：按条件族的种子表增补（max_values=4 与 accept 限流不变）。"""

    def test_blo_seeds_include_imm_minus_one(self):
        instructions = [
            {"address": 0x1000, "mnemonic": "LDRB", "operands": "r2, [r3]", "size": 2},
            {"address": 0x1002, "mnemonic": "CMP", "operands": "r2, #0x20", "size": 2},
            {"address": 0x1004, "mnemonic": "BLO", "operands": "0x1020", "size": 2},
        ]
        recovery = recovery_for(instructions)
        seeds = recovery._semantic_boundary_values(0x1004, "BLO", True, compare_imm=0x20)
        # BLO ∈ 无符号族（r18b §5.2.5：HI/LS/HS/LO）——imm-1/imm/imm+1 带
        self.assertIn((0x1F, "unsigned_minus_one"), seeds)
        self.assertIn((0x20, "compare_boundary"), seeds)
        # BCC（同条件码的借位族命名）走 borrow 边界
        bcc_seeds = recovery._semantic_boundary_values(0x1004, "BCC", True, compare_imm=0x20)
        self.assertIn((0x1F, "borrow_first_below"), bcc_seeds)
        self.assertIn((0x20, "borrow_at_boundary"), bcc_seeds)
        result = recovery.recover_values(
            branch_pc=0x1004,
            branch_condition="BLO",
            target_direction=True,
            read_pc=0x1000,
            primary_values=(0,),
            max_values=6,
        )
        self.assertTrue(result.symbolic_slice)
        # imm-1=0x1f 满足 BLO(taken)（r2<0x20）——条件反解单区位之后作为
        # 无符号族边界种子出现
        self.assertIn(0x1F, [item.value for item in result.hypotheses])

    def test_bge_seeds_include_signed_extremes(self):
        instructions = [
            {"address": 0x2000, "mnemonic": "LDR", "operands": "r0, [r1]", "size": 2},
            {"address": 0x2002, "mnemonic": "CMP", "operands": "r0, #0", "size": 2},
            {"address": 0x2004, "mnemonic": "BGE", "operands": "0x2020", "size": 2},
        ]
        recovery = recovery_for(instructions)
        seeds = recovery._semantic_boundary_values(0x2004, "BGE", True, compare_imm=0)
        self.assertIn((0x7FFFFFFF, "signed_max_boundary"), seeds)
        self.assertIn((0x80000000, "signed_min_boundary"), seeds)
        # BGE(taken) ⇔ N==V ⇔ 非负；0x80000000 不满足，0x7fffffff 满足
        result = recovery.recover_values(
            branch_pc=0x2004,
            branch_condition="BGE",
            target_direction=True,
            read_pc=0x2000,
            primary_values=(0x80000000,),
            max_values=4,
        )
        values = [item.value for item in result.hypotheses]
        self.assertTrue(result.symbolic_slice)
        self.assertNotIn(0x80000000, values)
        for value in values:
            self.assertLess(value, 0x80000000, hex(value))

    def test_unsigned_family_seeds_around_imm(self):
        recovery = LocalConstraintRecovery({}, compare_lookup={})
        seeds = recovery._semantic_boundary_values(0, "BHI", True, compare_imm=10)
        strategies = {strategy for _value, strategy in seeds}
        self.assertIn("unsigned_minus_one", strategies)
        self.assertIn("unsigned_plus_one", strategies)
        self.assertIn("compare_boundary", strategies)


if __name__ == "__main__":
    unittest.main()
