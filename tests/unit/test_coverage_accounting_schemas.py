#!/usr/bin/env python3
"""cycle4 k.4 回归测试 v4（k.5 落地形态，暂存 /tmp；落地路径 lsgemu/test_coverage_accounting_schemas.py）。

v3（R4）+ R5 裁决：
  - census 注册表升级为 R5-A(b) 严格路径正则判据（容器层失配即落精确表/
    UNREGISTERED，fail-closed），并预注册 k.4 自身新增的 C2 双写键
    validated_replay_lower_bound_bbs（=|V|）与 C1′ counterfactual_entry_validated_bbs
    （=|V∩cf|）——v3 注册表会把 k.4 自己的新键判 UNREGISTERED（R5-C 自纠错）。
  - R5-B ② 证伪臂：派生槽改回旧值 1287 → RED；锚名
    coverage_evidence_summary/validated_replay_bbs 1555→1448 → RED。
  - D∩cf≠∅ 构造（① 号用例）与 validated 撞名 census（⑦ 号用例）沿用 v3。
"""

import re
import unittest

from lsgemu.historical_runner import (
    coverage_accounting_three_number,
    termination_merged_denominator,
)

VALIDATED_BBS_CALIBER = (
    "validated_replay_minus_direction_confirmed_counterfactual_only"
)

# ---- R5-A(b) v4 注册表判据（与 /tmp/c4r5_validated_census.py 同源）----
PATH_PATTERNS = [
    (re.compile(r"^canonical_evidence_by_phase/[^/]+/validated_bbs$"), "phase_scoped"),
    (re.compile(r"^stall_watchdog/ledger_watermarks/[^/]+$"), "watermark"),
]
BASE_PATTERNS = [
    re.compile(r"reachable"),
    re.compile(r"^child_validated_execution_count$"),
    re.compile(r"^provenance_finalized_validated$"),
    re.compile(r"^validated$"),
]
EXACT = {  # 末段键名精确注册：期望值类别
    "validated_replay_bbs": "V",
    "canonical_validated_replay_bbs": "V",
    "L3_validated_replay_bbs": "V",
    "validated_replay_lower_bound_bbs": "V",          # C2 双写（k.4 新）
    "validated_bbs": "V_minus_cf",
    "validated_direction_confirmed_bbs": "V_minus_cf",
    "counterfactual_entry_validated_bbs": "V_cap_cf",  # C1′（k.4 新）
}


def validated_scalars(node, path=""):
    """census：递归枚举键名含 validated 的标量 int 键（不进 list 元素）。"""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            q = f"{path}/{k}" if path else str(k)
            if "validated" in str(k).lower() and isinstance(v, int) and not isinstance(v, bool):
                out.append((q, v))
            if isinstance(v, dict):
                out.extend(validated_scalars(v, q))
    return out


def census_violations(report, v_total, v_minus_cf, v_cap_cf):
    """k.4 v4 注册表：路径正则豁免相位/水位层（严格形状）；末段判据豁免可达/
    记录层；精确表校验锚名==|V|（含 C2 双写键）、派生==|V∖cf|、C1′==|V∩cf|；
    其余 validated 名族标量一律 UNREGISTERED。"""
    want = {"V": v_total, "V_minus_cf": v_minus_cf, "V_cap_cf": v_cap_cf}
    bad = []
    for p, v in validated_scalars(report):
        q = p.lstrip("/")
        base = q.rsplit("/", 1)[-1]
        if any(rx.match(q) for rx, _ in PATH_PATTERNS):
            continue  # 相位作用域（实测 25 键 8 值）/ 运行最大值（236≠161 可合法）
        if any(rx.search(base) for rx in BASE_PATTERNS):
            continue  # 可达子集/记录计数层：非 BB 覆盖计数
        if base in EXACT:
            if v != want[EXACT[base]]:
                bad.append((p.lstrip("/"), v, f"{EXACT[base]} != {want[EXACT[base]]}"))
        else:
            bad.append((p.lstrip("/"), v, "UNREGISTERED validated-named scalar"))
    return bad


class ThreeNumberCaliberTests(unittest.TestCase):
    def test_cf_not_subset_of_validated_round_shape(self):
        # 构造本轮形态：V=20, D=10, U=0, covered=30, cf 脸=12（cf∩V=7, cf∩D=5）。
        # 旧水位口径会发 {validated 8, cf 12, gap 10}：恒等式 30=8+12+10 全过，
        # 但 8 不对应任何集合基数（V∖cf=13 才是）。新口径断言 {13, 7, 10}。
        tn = coverage_accounting_three_number(
            covered_bbs=30,
            validated_bbs=20,
            canonical_counterfactual_only_bbs=7,   # = len(cf∩V)
            counterfactual_face_bbs=12,            # = len(cf) 只读披露
        )
        self.assertEqual(tn["validated_bbs"], 13)                    # |V∖cf|
        self.assertEqual(tn["direction_confirmed_counterfactual_only_bbs"], 7)
        self.assertEqual(tn["fidelity_gap_bbs"], 10)                 # |covered∖V|
        self.assertEqual(tn["covered_bbs"], 30)
        self.assertEqual(
            tn["validated_bbs"]
            + tn["direction_confirmed_counterfactual_only_bbs"]
            + tn["fidelity_gap_bbs"],
            30,
        )
        self.assertEqual(
            tn["counterfactual_only_source"], "canonical_cf_intersect_validated"
        )
        self.assertEqual(tn["counterfactual_face_bbs"], 12)

    def test_baseline_shape_where_calibers_coincide(self):
        # k.4 复算 armP3 基线形态：1450 = 1448 + 2 + 0（D=0 → cf⊆V，两口径重合）。
        # 防过度修正：新口径在基线形态必须与 C5 冻结样例逐字相同。
        tn = coverage_accounting_three_number(
            covered_bbs=1450,
            validated_bbs=1450,
            canonical_counterfactual_only_bbs=2,
            counterfactual_face_bbs=2,
        )
        self.assertEqual(
            (
                tn["validated_bbs"],
                tn["direction_confirmed_counterfactual_only_bbs"],
                tn["fidelity_gap_bbs"],
            ),
            (1448, 2, 0),
        )

    def test_ledger_watermark_plumbing_is_gone(self):
        # R3-A 裁决的「删支」半边：水位优先支与其参数一并死亡，
        # 防止跨轴喂数以任何形式回流进三数函数。
        with self.assertRaises(TypeError):
            coverage_accounting_three_number(
                covered_bbs=30,
                validated_bbs=20,
                canonical_counterfactual_only_bbs=7,
                ledger_watermarks={"counterfactual_only_bbs": 12},
            )

    def test_identity_is_blind_guard_is_not(self):
        # 恒等式对「x⊄V」全盲（cycle4 R2 §1(b) 恒真证明）：错喂 cf 脸当 slot
        # 时恒等式仍成立；守卫（调用点在集合作用域内复核 emitted slot ==
        # len(cf∩V)）必须能抓住 feeder 回归。
        cf_face = set(range(12))
        V = set(range(7)) | set(range(20, 33))       # |V|=20, cf∩V={0..6}=7
        emitted = coverage_accounting_three_number(
            covered_bbs=30,
            validated_bbs=20,
            canonical_counterfactual_only_bbs=len(cf_face & V),
            counterfactual_face_bbs=len(cf_face),
        )
        guard_ok = (
            emitted["direction_confirmed_counterfactual_only_bbs"]
            == len(cf_face & V)
        )
        self.assertTrue(guard_ok)
        wrong = coverage_accounting_three_number(    # feeder 回归：错喂整脸
            covered_bbs=30,
            validated_bbs=20,
            canonical_counterfactual_only_bbs=len(cf_face),
            counterfactual_face_bbs=len(cf_face),
        )
        self.assertEqual(
            wrong["validated_bbs"]
            + wrong["direction_confirmed_counterfactual_only_bbs"]
            + wrong["fidelity_gap_bbs"],
            30,
        )                                           # 恒等式照过（盲区证明）
        guard_bad = (
            wrong["direction_confirmed_counterfactual_only_bbs"]
            == len(cf_face & V)
        )
        self.assertFalse(guard_bad)                 # 守卫抓到


class ValidatedNameCensusTests(unittest.TestCase):
    """R4-A：validated 撞名 census（同报告同名多值，census 实测六层 54 键）。"""

    def _k4_shape_report(self):
        # V=20, cf=12（V∩cf=7 → V∖cf=13）, D=10, U=0, covered=30。
        tn = coverage_accounting_three_number(
            covered_bbs=30,
            validated_bbs=20,
            canonical_counterfactual_only_bbs=7,
            counterfactual_face_bbs=12,
        )
        tm = termination_merged_denominator(
            covered_bbs=30,
            validated_bbs=20,
            counterfactual_only_bbs=tn[
                "direction_confirmed_counterfactual_only_bbs"
            ],
            exploration_only_bbs=10,
            unclassified_bbs=0,
        )
        return {
            "validated_replay_bbs": 20,                      # 锚名层
            "canonical_coverage_summary": {"validated_replay_bbs": 20},
            "coverage_accounting_three_number": tn,          # 派生层
            "denominator_disclosure": {
                "termination_merged_denominator": tm,
                "L3_validated_replay_bbs": 20,
            },
            "counterfactual_entry_validated_bbs": 7,         # C1′ 层
            "validated_replay_lower_bound_bbs": 20,          # C2 双写（v4 注册）
            "canonical_evidence_by_phase": {                 # 相位同名层（豁免）
                "baseline": {"validated_bbs": 13},
            },
            "stall_watchdog": {                              # 水位层（豁免）
                "ledger_watermarks": {"validated_replay_bbs": 20},
            },
            "validated_reachable_bbs": 5,                    # 可达层（豁免）
        }

    def test_validated_name_collision_census(self):
        report = self._k4_shape_report()
        tn = report["coverage_accounting_three_number"]
        # (i) R4-A (a)：三数槽必须携带口径键（不改名，遵 C5 冻结面）。
        self.assertEqual(tn.get("validated_bbs_caliber"), VALIDATED_BBS_CALIBER)
        # (ii) census 注册表在 k.4 形态下零违规。
        self.assertEqual(census_violations(report, 20, 13, 7), [])
        # (iii) 牙齿一：口径键缺失必须被抓（census/口径断言二选一抓到）。
        stripped = {
            "coverage_accounting_three_number": {
                k: v for k, v in tn.items() if k != "validated_bbs_caliber"
            }
        }
        self.assertNotEqual(
            stripped["coverage_accounting_three_number"].get("validated_bbs_caliber"),
            VALIDATED_BBS_CALIBER,
        )
        # (iv) 牙齿二：旧 1287 形态（V−整脸=8，本轮真实缺陷形态）census 必抓。
        report["coverage_accounting_three_number"]["validated_bbs"] = 8
        self.assertNotEqual(census_violations(report, 20, 13, 7), [])
        # (v) 牙齿三：未注册的 validated 名族新键必须被抓。
        #     （k.5 修正：v3/v4 暂存稿此处 assertIn("UNREGISTERED", ...) 按
        #     列表元素精确匹配恒假（census 实际标签是更长的
        #     "UNREGISTERED validated-named scalar"），此前被 6 个 TypeError
        #     掩盖从未执行到本行；改子串匹配，断言意图不变。）
        report["coverage_accounting_three_number"]["validated_bbs"] = 13
        report["validated_magic_bbs"] = 999
        self.assertTrue(
            any(
                "UNREGISTERED" in r[2]
                for r in census_violations(report, 20, 13, 7)
            )
        )


class CensusFalsificationAndPatternTests(unittest.TestCase):
    """R5-B ② 证伪臂 + R5-A(b) 路径模式严格性（纯字典，不依赖产线签名）。"""

    def _arm_p3_shape_report(self):
        # armP3 实测数：|V|=1555, |V∖cf|=1448, |V∩cf|=107, lep.cf 脸=268。
        return {
            "validated_replay_bbs": 1555,
            "coverage_evidence_summary": {"validated_replay_bbs": 1555},
            "coverage_accounting_three_number": {
                "validated_bbs": 1448,
                "direction_confirmed_counterfactual_only_bbs": 107,
            },
            "denominator_disclosure": {
                "termination_merged_denominator": {
                    "validated_direction_confirmed_bbs": 1448,
                },
            },
            "counterfactual_entry_validated_bbs": 107,
            "validated_replay_lower_bound_bbs": 1555,
            "canonical_evidence_by_phase": {
                "baseline": {"validated_bbs": 1448},
            },
            "stall_watchdog": {
                "ledger_watermarks": {"validated_replay_bbs": 1555},
            },
        }

    def test_falsification_arm_derived_slot_back_to_1287(self):
        # R5-B ② ①：派生槽改回旧值 1287（=|V|−整脸 268 的旧水位口径）→ RED。
        report = self._arm_p3_shape_report()
        report["coverage_accounting_three_number"]["validated_bbs"] = 1287
        viol = census_violations(report, 1555, 1448, 107)
        self.assertIn(
            ("coverage_accounting_three_number/validated_bbs", 1287,
             "V_minus_cf != 1448"),
            viol,
        )

    def test_falsification_arm_anchor_1555_to_1448(self):
        # R5-B ② ②：锚名 coverage_evidence_summary/validated_replay_bbs
        # 由 1555 改 1448（锚名错值）→ RED。
        report = self._arm_p3_shape_report()
        report["coverage_evidence_summary"]["validated_replay_bbs"] = 1448
        viol = census_violations(report, 1555, 1448, 107)
        self.assertIn(
            ("coverage_evidence_summary/validated_replay_bbs", 1448,
             "V != 1555"),
            viol,
        )

    def test_path_pattern_strictness(self):
        # R5-A(b)：豁免=严格路径正则，非容器前缀。
        # (i) v3 漏洞回归：相位容器内锚名键错值必须被抓（v3 前缀形态静默豁免）。
        report = self._arm_p3_shape_report()
        report["canonical_evidence_by_phase"]["baseline"][
            "validated_replay_bbs"] = 12345
        self.assertIn(
            ("canonical_evidence_by_phase/baseline/validated_replay_bbs",
             12345, "V != 1555"),
            census_violations(report, 1555, 1448, 107),
        )
        # (ii) 主代理反例：不在任何模式、不在锚/派生层的 validated_* → RED。
        report["coverage_evidence_summary"]["validated_magic_bbs"] = 999
        self.assertIn(
            ("coverage_evidence_summary/validated_magic_bbs", 999,
             "UNREGISTERED validated-named scalar"),
            census_violations(report, 1555, 1448, 107),
        )
        # (iii) 合法相位/水位形状仍豁免（25 键 8 值全靠此模式）。
        report["coverage_evidence_summary"].pop("validated_magic_bbs", None)
        report["canonical_evidence_by_phase"]["baseline"] = {"validated_bbs": 1448}
        report["stall_watchdog"]["ledger_watermarks"][
            "diagnostic_replay_bbs"] = 236  # E-r4-1：水位≠报告时分区可合法
        self.assertEqual(census_violations(report, 1555, 1448, 107), [])
        # (iv) 更深层嵌套失配（[^/]+ 只允许一层相位名）→ fail-closed 落精确表。
        report["canonical_evidence_by_phase"]["baseline"] = {
            "nested": {"validated_bbs": 13},
        }
        self.assertIn(
            ("canonical_evidence_by_phase/baseline/nested/validated_bbs",
             13, "V_minus_cf != 1448"),
            census_violations(report, 1555, 1448, 107),
        )


class MergedDenominatorLockstepTests(unittest.TestCase):
    def test_lockstep_with_three_number_cf_slot(self):
        # 本轮实测形态：1716 = 1448 + 107 + 161（三数）/ {1448,107,161,0}（五数）。
        tn = coverage_accounting_three_number(
            covered_bbs=1716,
            validated_bbs=1555,
            canonical_counterfactual_only_bbs=107,
            counterfactual_face_bbs=268,
        )
        tm = termination_merged_denominator(
            covered_bbs=1716,
            validated_bbs=1555,
            counterfactual_only_bbs=tn[
                "direction_confirmed_counterfactual_only_bbs"
            ],
            exploration_only_bbs=161,
            unclassified_bbs=0,
        )
        self.assertEqual(
            (
                tm["validated_direction_confirmed_bbs"],
                tm["direction_confirmed_counterfactual_only_bbs"],
                tm["exploration_only_bbs"],
                tm["fidelity_gap_bbs"],
            ),
            (1448, 107, 161, 0),
        )
        self.assertTrue(tm["identity_holds"])
        # Q4 裁决：三数 gap 是五数末两桶（exploration∪unclassified）的粗口径
        # 合并——结构恒等（三.gap=|cov∖V|=|D∪U|），验收脚本以 lockstep 锁死。
        self.assertEqual(
            tn["fidelity_gap_bbs"],
            tm["exploration_only_bbs"] + tm["fidelity_gap_bbs"],
        )
        # face 对账（Q1 只读披露的验收形态）：268 = 107（进 validated）
        # + 161（diagnostic）+ 0（unclassified）。
        self.assertEqual(
            tn["counterfactual_face_bbs"],
            tm["direction_confirmed_counterfactual_only_bbs"] + 161 + 0,
        )

    def test_unclassified_gt0_gap_stays_reconcilable(self):
        # Q4 边界：U>0 时三数 gap 并桶（D∪U）、五数单列——接受粗口径，
        # 但 lockstep 恒等必须继续成立（30 = 20 = 10+5 等）。
        tn = coverage_accounting_three_number(
            covered_bbs=40,
            validated_bbs=25,
            canonical_counterfactual_only_bbs=5,
            counterfactual_face_bbs=9,
        )
        tm = termination_merged_denominator(
            covered_bbs=40,
            validated_bbs=25,
            counterfactual_only_bbs=tn[
                "direction_confirmed_counterfactual_only_bbs"
            ],
            exploration_only_bbs=10,
            unclassified_bbs=5,
        )
        self.assertEqual(tn["fidelity_gap_bbs"], 15)   # |D∪U| = 10+5
        self.assertEqual(tm["fidelity_gap_bbs"], 5)    # 五数单列 unclassified
        self.assertEqual(
            tn["fidelity_gap_bbs"],
            tm["exploration_only_bbs"] + tm["fidelity_gap_bbs"],
        )


if __name__ == "__main__":
    unittest.main()
