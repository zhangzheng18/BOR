#!/usr/bin/env python3
"""r19 P0 回归：NZCV 产生者索引 + 符号标志槽（/tmp/dfs_r18b.md §5.4 清单 1–11）。

语义护栏（防回潮）：
- 32 位乘法族（MUL/MLA/MLS/SMULL/UMULL/SMLAL/UMLAL/半字并行乘）永不写标志；
- MULS 仅 16 位形态且仅 IT 块外写 N/Z；
- 逻辑 imm 的 C 按 ThumbExpandImm_C 全模式：非旋转/复制形不变，旋转形
  C = imm32<31>（解码期常量）；
- VMRS/MSR 显式写者只标签化（label_fp_opaque/label_msr），值合成通道跳过；
- in_it 产生者拒绝；partial/live_in 拒绝并标记；CBZ/CBNZ 不入索引。
"""

from __future__ import annotations

import glob
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.local_constraint_recovery import (
    LocalConstraintRecovery,
    expand_imm_c,
    imm12_from_value,
)
from lsgemu.prepared_firmware import (
    STATIC_CACHE_VERSION,
    _build_producer_index,
    _producer_flag_writes,
)


def _insn(address: int, mnemonic: str, operands: str, size: int = 2) -> dict:
    return {"address": address, "mnemonic": mnemonic, "operands": operands, "size": size}


def _recovery(static_bbs, successors, *, branch_pc, read_pc, compare_lookup=None):
    instruction_to_bb = {
        int(insn["address"]): bb for bb, insns in static_bbs.items() for insn in insns
    }
    return LocalConstraintRecovery(
        static_bbs,
        instruction_to_bb=instruction_to_bb,
        compare_lookup=dict(compare_lookup or {}),
        producer_index=_build_producer_index(static_bbs, successors),
    )


class ProducerSemanticsTests(unittest.TestCase):
    """清单 1–2：乘法族语义（r18b 三重证据：手册 × qemu × 差分）。"""

    def test_mul32_family_never_producers(self):
        for mnemonic in (
            "MUL", "MLA", "MLS", "SMULL", "UMULL", "SMLAL", "UMLAL",
            "SMULBB", "SMULTT",
        ):
            for raw in (mnemonic, f"{mnemonic}.W", f"{mnemonic}.NE"):
                covers, kind, _c_source, _in_it = _producer_flag_writes(
                    _insn(0x1000, raw, "r0, r1, r2", size=4)
                )
                self.assertEqual(
                    covers, "", f"{raw} 不得是 NZCV 产生者（32 位乘法族 setflags=FALSE）"
                )
                self.assertEqual(kind, "")

    def test_muls_outside_it_is_nz_producer(self):
        covers, kind, c_source, in_it = _producer_flag_writes(
            _insn(0x1000, "MULS", "r3, r4")
        )
        self.assertEqual(covers, "NZ")
        self.assertEqual(kind, "mul16")
        self.assertFalse(in_it)

    def test_muls_inside_multi_slot_it_not_producer(self):
        # A7.7.84: setflags = !InITBlock() —— it ne; muls; muls 的第二槽不写。
        covers, _kind, _c_source, in_it = _producer_flag_writes(
            _insn(0x1000, "MULS.NE", "r3, r4")
        )
        self.assertEqual(covers, "")
        self.assertTrue(in_it)

    def test_muls_single_slot_it_al_emulator_divergence(self):
        # emulator_divergence 记录（r18b §1.3）：单槽 it al 后 unicorn 实测写
        # N/Z（condexec_mask=0），ARM 对 IT AL 本身 UNPREDICTABLE。Ghidra 对该
        # 形态渲染为无后缀 muls（AL 不在条件码集合、也无 .AL 后缀）——等价按
        # IT 块外处理为 N/Z 产生者，与被模拟的 unicorn 行为一致。
        covers, kind, _c_source, in_it = _producer_flag_writes(
            _insn(0x1000, "MULS.AL", "r3, r4")
        )
        self.assertEqual((covers, kind, in_it), ("NZ", "mul16", False))
        covers, kind, _c_source, in_it = _producer_flag_writes(
            _insn(0x1000, "MULS", "r3, r4")
        )
        self.assertEqual((covers, kind, in_it), ("NZ", "mul16", False))

    def test_clz_sat_rev_never_producers(self):
        for mnemonic in ("CLZ", "SXTB", "SXTH", "UXTB", "UXTH", "REV", "REV16",
                         "REVSH", "SSAT", "USAT", "SDIV", "UDIV"):
            covers, _kind, _c_source, _in_it = _producer_flag_writes(
                _insn(0x1000, mnemonic, "r0, r1")
            )
            self.assertEqual(covers, "", mnemonic)

    def test_vmrs_msr_labeled(self):
        covers, kind, c_source, _in_it = _producer_flag_writes(
            _insn(0x1000, "VMRS", "apsr, fpscr", size=4)
        )
        self.assertEqual((covers, kind, c_source), ("NZCV", "explicit", "explicit"))
        covers, _kind, _c_source, _in_it = _producer_flag_writes(
            _insn(0x1000, "VMRS", "r1, fpscr", size=4)
        )
        self.assertEqual(covers, "")
        covers, _kind, _c_source, _in_it = _producer_flag_writes(
            _insn(0x1000, "MSR", "basepri, r5", size=4)
        )
        self.assertEqual(covers, "")

    def test_logic_imm_covers_full_mode(self):
        # 旋转形（0xff000000 = (0x80|0x7f)<<24）：NZC + imm_expand
        covers, kind, c_source, _ = _producer_flag_writes(
            _insn(0x1000, "ANDS.W", "r2, r2, #0xff000000", size=4)
        )
        self.assertEqual((covers, c_source), ("NZC", "imm_expand"))
        # 非旋转 imm8：NZ、C 不变
        covers, kind, c_source, _ = _producer_flag_writes(
            _insn(0x1000, "ANDS.W", "r2, r2, #0x30", size=4)
        )
        self.assertEqual((covers, c_source), ("NZ", None))
        # 复制形 0x00ff00ff：NZ、C 不变
        covers, kind, c_source, _ = _producer_flag_writes(
            _insn(0x1000, "ANDS.W", "r2, r2, #0x00ff00ff", size=4)
        )
        self.assertEqual((covers, c_source), ("NZ", None))
        # 16 位 imm：NZ、C 不变
        covers, kind, c_source, _ = _producer_flag_writes(
            _insn(0x1000, "ANDS", "r2, r2, #0x30")
        )
        self.assertEqual((covers, c_source), ("NZ", None))
        # 纯寄存器：NZ、C 不变；带移位寄存器：NZC + shifter
        self.assertEqual(
            _producer_flag_writes(_insn(0x1000, "ANDS.W", "r1, r3, r5", size=4))[:3],
            ("NZ", "logic", None),
        )
        self.assertEqual(
            _producer_flag_writes(_insn(0x1000, "ANDS.W", "r4, r12, r1, lsr #0x14", size=4))[:3],
            ("NZC", "logic", "shifter"),
        )


class ExpandImmCTests(unittest.TestCase):
    """清单 3：ThumbExpandImm_C 全模式表驱动（≥30 断言）。"""

    IMM8_SAMPLES = (0x00, 0x01, 0x37, 0x7F, 0x80, 0xC3, 0xFF)

    def test_nonrotation_modes_leave_carry_none(self):
        for imm8 in self.IMM8_SAMPLES:
            value, carry = expand_imm_c(imm8)
            self.assertEqual(value, imm8)
            self.assertIsNone(carry)
            value, carry = expand_imm_c(0x100 | imm8)
            self.assertEqual(value, (imm8 << 16) | imm8)
            self.assertIsNone(carry)
            value, carry = expand_imm_c(0x200 | imm8)
            self.assertEqual(value, (imm8 << 24) | (imm8 << 8))
            self.assertIsNone(carry)
            value, carry = expand_imm_c(0x300 | imm8)
            self.assertEqual(value, imm8 * 0x01010101)
            self.assertIsNone(carry)

    def test_rotation_modes_set_carry_to_bit31(self):
        # 旋转 n = UInt(imm12<11:7>) ∈ 8..31；unrotated = 0x80|imm7（低 8 位）
        for rotation in range(8, 32):
            unit = 0x80 | ((rotation * 37) % 0x80)
            imm12 = (rotation << 7) | (unit & 0x7F)
            value, carry = expand_imm_c(imm12)
            expected = ((unit >> rotation) | (unit << (32 - rotation))) & 0xFFFFFFFF
            self.assertEqual(value, expected, f"imm12=0x{imm12:03x}")
            # A5.3.2 表注：C = 修改后立即数 bit[31]；ROR_C carry = result<31>
            self.assertEqual(carry, (value >> 31) & 1, f"imm12=0x{imm12:03x}")

    def test_known_encodings(self):
        # 真实固件样本（capstone 对账）：
        self.assertEqual(expand_imm_c(0x980), (0x00100000, 0))    # cmp.w r1,#0x100000
        self.assertEqual(expand_imm_c(0x880), (0x00400000, 0))    # cmn.w r2,#0x400000
        self.assertEqual(expand_imm_c(0xF00), (0x00000200, 0))    # cmp.w r6,#0x200
        self.assertEqual(expand_imm_c(0x47F), (0xFF000000, 1))    # bics r2,#0xff000000
        self.assertEqual(expand_imm_c(0x30F), (0x0F0F0F0F, None))  # 复制形 '11'
        self.assertEqual(expand_imm_c(0x3FF), (0xFFFFFFFF, None))  # 0xFF 复制形
        self.assertEqual(expand_imm_c(0x2F0), (0xF000F000, None))  # 复制形 '10'

    def test_reverse_mapping_roundtrip_full_domain(self):
        skipped = {0x100, 0x200, 0x300}  # UNPREDICTABLE 零编码（三态映射到 0）
        for imm12 in range(0x1000):
            if imm12 in skipped:
                continue
            value, _carry = expand_imm_c(imm12)
            self.assertEqual(imm12_from_value(value), imm12, f"imm12=0x{imm12:03x}")

    def test_reverse_mapping_matches_prepared_module(self):
        from lsgemu.prepared_firmware import _producer_imm12_from_value

        for imm12 in range(0x1000):
            if imm12 in {0x100, 0x200, 0x300}:
                continue
            value, _carry = expand_imm_c(imm12)
            self.assertEqual(_producer_imm12_from_value(value), imm12)


class FlagSlotSolverTests(unittest.TestCase):
    """清单 4–8：索引门控 + 标志槽求解。"""

    def test_multi_producer_hi_solves_sat(self):
        # HI 需要 C+Z：C 由更早 LSRS（移位 C），Z 由最近 ANDS（寄存器形 C 不变）
        # —— r18b §5.2 的「后写只覆写其 covers 位」多产生者语义。
        static_bbs = {
            0x3000: [
                _insn(0x3000, "LDR", "r2, [r4]"),
                _insn(0x3002, "LSRS", "r2, r2, #4"),
                _insn(0x3004, "ANDS", "r2, r2"),
                _insn(0x3006, "BHI", "0x3020"),
            ],
        }
        index = _build_producer_index(static_bbs, {0x3000: {0x3020}})
        producers, paths, _calls = index[0x3006]
        self.assertEqual(paths, "same_bb")
        self.assertEqual([p[1] for p in producers], ["ANDS", "LSRS"])
        self.assertEqual(producers[0][2], "Z")   # ANDS 只覆盖所需 Z
        self.assertEqual(producers[1][2], "C")   # LSRS 补 C
        recovery = _recovery(static_bbs, {0x3000: {0x3020}}, branch_pc=0x3006, read_pc=0x3000)
        result = recovery.recover_values(
            branch_pc=0x3006, branch_condition="BHI",
            target_direction=True, read_pc=0x3000, max_values=4,
        )
        self.assertTrue(result.symbolic_slice)
        self.assertTrue(result.hypotheses)
        # BHI(taken) ⇔ C(bit3 of raw) ∧ ¬Z((raw>>4)&(raw>>4)≠0) ⇔ raw ≥ 0x18
        for item in result.hypotheses:
            self.assertGreaterEqual(item.value, 0x18, hex(item.value))

    def test_cross_bb_all_covered_record_and_mmio_cone(self):
        # 索引层：唯一前驱链上凑齐标志 → all_covered，产生者 pc 在前驱 BB。
        static_bbs = {
            0x08004000: [
                _insn(0x08004000, "LDR", "r3, [pc, #4]"),
                _insn(0x08004002, "LDR", "r2, [r3]"),
                _insn(0x08004004, "CMP", "r2, #0"),
                _insn(0x08004006, "B", "0x08004100"),
            ],
            0x08004100: [_insn(0x08004100, "BNE", "0x08004200")],
        }
        successors = {0x08004000: {0x08004100}, 0x08004100: {0x08004200}}
        index = _build_producer_index(static_bbs, successors)
        producers, paths, _calls = index[0x08004100]
        self.assertEqual(paths, "all_covered")
        self.assertEqual(producers[0][1], "CMP")
        self.assertEqual(producers[0][6], 1)  # bbs_crossed

        # 求解器侧：读与分支须同 BB（跨 BB 读先被既有护栏拒绝）
        recovery = _recovery(
            static_bbs, successors, branch_pc=0x08004100, read_pc=0x08004002
        )
        result = recovery.recover_values(
            branch_pc=0x08004100, branch_condition="BNE",
            target_direction=True, read_pc=0x08004002, max_values=4,
        )
        self.assertFalse(result.symbolic_slice)
        self.assertEqual(result.reason, "read_and_branch_not_in_same_basic_block")

        # 同 BB 内产生者在读之前（same_bb 记录）→ producer_outside_local_slice
        local_bbs = {
            0x08004000: [
                _insn(0x08004000, "CMP", "r1, #0"),
                _insn(0x08004002, "LDR", "r2, [r3]"),
                _insn(0x08004004, "BNE", "0x08004200"),
            ],
        }
        local_succ = {0x08004000: {0x08004200}}
        local_index = _build_producer_index(local_bbs, local_succ)
        self.assertEqual(local_index[0x08004004][1], "same_bb")
        local_recovery = _recovery(
            local_bbs, local_succ, branch_pc=0x08004004, read_pc=0x08004002
        )
        local_result = local_recovery.recover_values(
            branch_pc=0x08004004, branch_condition="BNE",
            target_direction=True, read_pc=0x08004002, max_values=4,
        )
        self.assertFalse(local_result.symbolic_slice)
        self.assertEqual(local_result.reason, "producer_outside_local_slice")

        # 精确分析侧：跨 BB 产生者沿唯一前驱链定位，MMIO 锥可达（r19 接线）。
        from tests.unit._firmware_shell import _firmware_shell

        runner = _firmware_shell(
            static_bbs,
            compare_lookup={},
            successors=successors,
            literals={(0x08004000 + 4 + 4) & ~3: 0x40000000},
        )
        runner.prepared.producer_index = index
        analysis = runner._precise_branch_mmio_analysis(0x08004100)
        self.assertEqual(analysis["candidates"], [0x40000000])
        self.assertEqual(analysis["read_pc_by_addr"], {0x40000000: 0x08004002})
        self.assertGreaterEqual(analysis["scan_bb_count"], 2)

    def test_partial_rejected_with_marker(self):
        # 双前驱：一条路径覆盖、一条不覆盖 → producer_path_dependent。
        static_bbs = {
            0x1000: [
                _insn(0x1000, "CMP", "r1, #0"),
                _insn(0x1002, "B", "0x3000"),
            ],
            0x2000: [
                _insn(0x2000, "MOV", "r0, r0"),
                _insn(0x2002, "B", "0x3000"),
            ],
            0x3000: [
                _insn(0x3000, "LDR", "r2, [r3]"),
                _insn(0x3002, "BNE", "0x4000"),
            ],
        }
        successors = {0x1000: {0x3000}, 0x2000: {0x3000}, 0x3000: {0x4000}}
        index = _build_producer_index(static_bbs, successors)
        self.assertEqual(index[0x3002][1], "partial")
        recovery = _recovery(static_bbs, successors, branch_pc=0x3002, read_pc=0x3000)
        result = recovery.recover_values(
            branch_pc=0x3002, branch_condition="BNE",
            target_direction=True, read_pc=0x3000, max_values=4,
        )
        self.assertFalse(result.symbolic_slice)
        self.assertEqual(result.reason, "producer_path_dependent")

    def test_live_in_marked(self):
        static_bbs = {
            0x1000: [
                _insn(0x1000, "LDR", "r2, [r3]"),
                _insn(0x1002, "MOV", "r0, r0"),
                _insn(0x1004, "BNE", "0x2000"),
            ],
        }
        successors = {0x1000: {0x2000}}
        index = _build_producer_index(static_bbs, successors)
        self.assertEqual(index[0x1004][1], "live_in")
        recovery = _recovery(static_bbs, successors, branch_pc=0x1004, read_pc=0x1000)
        result = recovery.recover_values(
            branch_pc=0x1004, branch_condition="BNE",
            target_direction=True, read_pc=0x1000, max_values=4,
        )
        self.assertEqual(result.reason, "flags_live_in")

    def test_in_it_producer_rejected(self):
        static_bbs = {
            0x1000: [
                _insn(0x1000, "LDR", "r2, [r3]"),
                _insn(0x1002, "LSLS.NE", "r2, r2, #1"),
                _insn(0x1004, "BNE", "0x2000"),
            ],
        }
        successors = {0x1000: {0x2000}}
        index = _build_producer_index(static_bbs, successors)
        self.assertTrue(any(p[4] for p in index[0x1004][0]))
        recovery = _recovery(static_bbs, successors, branch_pc=0x1004, read_pc=0x1000)
        result = recovery.recover_values(
            branch_pc=0x1004, branch_condition="BNE",
            target_direction=True, read_pc=0x1000, max_values=4,
        )
        self.assertEqual(result.reason, "unsupported_predicated_slice")

    def test_vmrs_site_labeled_and_skipped(self):
        static_bbs = {
            0x1000: [
                _insn(0x1000, "LDR", "r2, [r3]"),
                _insn(0x1002, "VMRS", "apsr, fpscr", size=4),
                _insn(0x1006, "BNE", "0x2000"),
            ],
        }
        successors = {0x1000: {0x2000}}
        index = _build_producer_index(static_bbs, successors)
        producers, paths, _calls = index[0x1006]
        self.assertEqual(paths, "label_fp_opaque")
        self.assertEqual(producers[0][1], "VMRS")
        recovery = _recovery(static_bbs, successors, branch_pc=0x1006, read_pc=0x1000)
        result = recovery.recover_values(
            branch_pc=0x1006, branch_condition="BNE",
            target_direction=True, read_pc=0x1000, max_values=4,
        )
        self.assertEqual(result.reason, "fp_opaque_producer")
        # 跳过 = 不产生「条件反解/求解器」候选：无 symbolic slice、无
        # condition_derived_*/smt_* 策略（规则兜底值仍允许，交重放验证）。
        self.assertFalse(result.symbolic_slice)
        for item in result.hypotheses:
            self.assertFalse(item.exact_local_model)
            self.assertFalse(str(item.strategy).startswith(("condition_derived", "smt_")))

        # 运行时侧（精确分析）：label 站不产生 MMIO 候选。
        from tests.unit._firmware_shell import _firmware_shell

        runner = _firmware_shell(static_bbs, compare_lookup={}, successors=successors)
        runner.prepared.producer_index = index
        analysis = runner._precise_branch_mmio_analysis(0x1006)
        self.assertEqual(analysis["candidates"], [])

    def test_cbz_not_in_index(self):
        static_bbs = {
            0x1000: [
                _insn(0x1000, "LDR", "r2, [r3]"),
                _insn(0x1002, "CBZ", "r2, 0x2000"),
            ],
        }
        index = _build_producer_index(static_bbs, {0x1000: {0x2000}})
        self.assertNotIn(0x1002, index)


class IndexRegressionTests(unittest.TestCase):
    """清单 10：成本/键空间/版本回归。"""

    def test_build_idempotent(self):
        static_bbs = {
            0x1000: [
                _insn(0x1000, "LDR", "r2, [r3]"),
                _insn(0x1002, "CMP", "r2, #5"),
                _insn(0x1004, "BNE", "0x2000"),
            ],
        }
        successors = {0x1000: {0x2000}}
        self.assertEqual(
            _build_producer_index(static_bbs, successors),
            _build_producer_index(static_bbs, successors),
        )

    def test_static_cache_version_bumped(self):
        # v15：producer_index 进 static cache；旧 v14 缓存必须判废重建。
        self.assertEqual(STATIC_CACHE_VERSION, 15)

    def test_key_space_is_branch_pcs(self):
        # 键 = 分支指令地址（与 _precise_branch_mmio_analysis 的 branch_pc 同一
        # 键空间）；值形状 = (producers, paths, calls_between)。
        static_bbs = {
            0x1000: [
                _insn(0x1000, "CMP", "r1, #0"),
                _insn(0x1002, "BEQ", "0x3000"),
            ],
            0x2000: [
                _insn(0x2000, "LDR", "r2, [r3]"),
                _insn(0x2002, "SUBS", "r2, r2, #1"),
                _insn(0x2004, "BLO", "0x3000"),
            ],
            0x2100: [_insn(0x2100, "B", "0x1000")],
        }
        successors = {0x1000: {0x3000}, 0x2000: {0x3000}, 0x2100: {0x1000}}
        index = _build_producer_index(static_bbs, successors)
        self.assertEqual(set(index), {0x1002, 0x2004})
        for value in index.values():
            producers, paths, calls_between = value
            self.assertIsInstance(producers, tuple)
            self.assertIsInstance(paths, str)
            self.assertIsInstance(calls_between, int)
            for p_pc, mnemonic, covers, c_source, in_it, dist, bbs in producers:
                self.assertIsInstance(p_pc, int)
                self.assertIsInstance(mnemonic, str)
                self.assertTrue(set(covers) <= set("NZCV"))
                self.assertIn(c_source, {"arith", "shifter", "imm_expand", "explicit", ""})
                self.assertIsInstance(in_it, bool)
                self.assertGreaterEqual(dist, 0)
                self.assertIsInstance(bbs, int)


class RealFirmwareSiteTests(unittest.TestCase):
    """清单 11：Pixhawk1 真实站点快测（v15 静态缓存存在时执行）。"""

    ELF_MARK = "ardupilot_Pixhawk1_STM32F427"
    CACHE_GLOB = os.path.join(
        str(PROJECT_ROOT), ".lsgemu_cache", f"{ELF_MARK}*_static_cache.pkl"
    )

    @classmethod
    def _load_v15_cache(cls):
        from lsgemu.prepared_firmware import _load_static_cache_pickle

        best = None
        for path in sorted(glob.glob(cls.CACHE_GLOB), key=os.path.getmtime):
            try:
                payload = _load_static_cache_pickle(Path(path))
            except Exception:
                continue
            if payload.get("version") == STATIC_CACHE_VERSION:
                best = payload
        return best

    def test_real_sites(self):
        payload = self._load_v15_cache()
        if payload is None:
            self.skipTest("Pixhawk1 v15 静态缓存不存在（需先完成一次 v15 重建）")
        prepared = payload["prepared"]
        index = prepared.producer_index
        self.assertTrue(index)

        # r18b §4.2 抽验的 VMRS 家族站（0x80207f0 族在 px4_fmu-v3；Pixhawk1 上
        # 以任一 label_fp_opaque 站验证同族语义）。
        opaque_pcs = [pc for pc, v in index.items() if v[1] == "label_fp_opaque"]
        self.assertTrue(opaque_pcs)
        record = index[opaque_pcs[0]]
        self.assertTrue(any(p[1] == "VMRS" for p in record[0]))

        covered = [
            (pc, v) for pc, v in index.items()
            if v[1] == "all_covered" and v[0] and v[0][0][1] == "CMP"
        ]
        self.assertTrue(covered)
        pc, (producers, paths, _calls) = covered[0]
        self.assertEqual(paths, "all_covered")
        self.assertGreater(producers[0][6], 0)  # 跨 BB
        self.assertEqual(
            prepared.instruction_to_bb.get(producers[0][0]) is not None, True
        )

        partial = [(pc, v) for pc, v in index.items() if v[1] == "partial"]
        if partial:  # Pixhawk1 实测 2 个；形状护栏
            _pc, (producers, paths, _calls) = partial[0]
            self.assertEqual(paths, "partial")

        # 求解器与索引同键空间：任取 same_bb 站可直接 .get(branch_pc)
        same_bb_pc = next(pc for pc, v in index.items() if v[1] == "same_bb")
        self.assertIn(same_bb_pc, index)


if __name__ == "__main__":
    unittest.main()
