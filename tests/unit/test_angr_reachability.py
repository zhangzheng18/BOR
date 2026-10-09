#!/usr/bin/env python3
"""angr 从入口可达 BB 基线（angr_reachability）契约测试。

三层验证：
1. 纯逻辑层：十六进制行解析、懒加载开关/文件名约定、JSON/BB 文件格式
   （合成 ReachabilityReport，不依赖 angr，任何环境可跑）；
2. PreparedFirmware 集成：reachable_total_bbs / reachable_coverage 的
   粒度对齐规则（angr BB 起始落在已覆盖静态 BB 内部同样计命中）；
3. 真实固件层（依赖 angr 与 Heat_Press.elf，缺失时跳过）：
   可达数 > 0 且 < Ghidra 全量静态 BB；CLI 产物格式正确。

angr 全程 lazy import：本文件自身不 import angr，纯逻辑用例不付 10 秒
导入代价；只有真实固件用例内部触发。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.angr_reachability import (
    CFG_EMULATED_SIZE_LIMIT,
    REACHABILITY_SCHEMA,
    AngrReachabilityError,
    ReachabilityReport,
    angr_reachability_enabled,
    compute_reachable_bbs,
    load_angr_reachable_bb_set,
    main as reachability_cli_main,
    parse_reachable_bb_lines,
    report_to_dict,
    write_bb_file,
)

# 真实固件回归样本（构建产物缺失时跳过对应用例）。
HEAT_PRESS_ELF = Path(
    "/opt/artifact/benchmarks/elfmultifuzz/P2IM/Heat_Press/Heat_Press.elf"
)


def _angr_available() -> bool:
    try:
        import angr  # noqa: F401
        return True
    except Exception:
        return False


def synthetic_report() -> ReachabilityReport:
    """不依赖 angr 的合成报告（格式契约用）。"""
    return ReachabilityReport(
        elf_path="/tmp/fake.elf",
        entry=0x08004FB5,
        bb_addrs=[0x08000000, 0x0800000C, 0x0800001A],
        node_count=5,
        elapsed_seconds=1.5,
        entry_source="e_entry",
        method="cfg_emulated",
        degrade_reason=None,
        angr_version="9.2.14",
    )


class ParseAndFormatTests(unittest.TestCase):
    """纯逻辑：解析与输出格式。"""

    def test_parse_reachable_bb_lines_accepts_mixed_tokens(self):
        addrs = parse_reachable_bb_lines(
            "08000000\n\n800f4\n  08000000 \n0x0800000c\nbad-line\n"
        )
        self.assertEqual(addrs, {0x08000000, 0x800F4, 0x0800000C})

    def test_report_to_dict_hex_format(self):
        payload = report_to_dict(synthetic_report())
        self.assertEqual(payload["schema"], REACHABILITY_SCHEMA)
        self.assertEqual(payload["bb_count"], 3)
        self.assertEqual(payload["entry"], 0x08004FB5)
        self.assertEqual(payload["entry_hex"], "0x08004fb5")
        self.assertEqual(
            payload["bb_addrs"], ["0x08000000", "0x0800000c", "0x0800001a"]
        )
        # 全量 JSON 可序列化
        json.dumps(payload)

    def test_write_bb_file_matches_valid_basic_blocks_contract(self):
        # 与数据集侧 valid_basic_blocks.txt 同构：小写 8 位十六进制、无 0x 前缀，
        # 且能被 runner_common._load_valid_bbs 的 int(line, 16) 口径读回。
        with tempfile.TemporaryDirectory(prefix="lsgemu_reach_fmt_") as tmp:
            output = Path(tmp) / "fw_angr_reachable.txt"
            resolved = write_bb_file(synthetic_report(), output)
            lines = resolved.read_text().splitlines()
            self.assertEqual(lines, ["08000000", "0800000c", "0800001a"])
            from lsgemu.runner_common import _load_valid_bbs

            self.assertEqual(
                _load_valid_bbs(resolved),
                {0x08000000, 0x0800000C, 0x0800001A},
            )

    def test_size_limit_constant(self):
        # 大固件自动降级阈值：5MB（CFGEmulated 复杂度超线性）。
        self.assertEqual(CFG_EMULATED_SIZE_LIMIT, 5 * 1024 * 1024)


class LazyLoadTests(unittest.TestCase):
    """纯逻辑：LSGEMU_ANGR_REACHABILITY 开关与 <stem>_angr_reachable.txt 约定。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_reach_lazy_")
        self.dir = Path(self._temporary.name)
        (self.dir / "demo_fw.elf").write_bytes(b"\x7fELF-placeholder")
        (self.dir / "demo_fw_angr_reachable.txt").write_text(
            "08000000\n800f4\n0x08000010\n"
        )

    def tearDown(self):
        self._temporary.cleanup()

    def test_disabled_env_returns_none_pair(self):
        with patch.dict("os.environ", {"LSGEMU_ANGR_REACHABILITY": ""}, clear=False):
            path, addrs = load_angr_reachable_bb_set(self.dir / "demo_fw.elf")
        self.assertIsNone(path)
        self.assertIsNone(addrs)
        self.assertFalse(angr_reachability_enabled())

    def test_enabled_env_loads_convention_file(self):
        with patch.dict("os.environ", {"LSGEMU_ANGR_REACHABILITY": "1"}, clear=False):
            self.assertTrue(angr_reachability_enabled())
            path, addrs = load_angr_reachable_bb_set(self.dir / "demo_fw.elf")
        self.assertEqual(path, self.dir / "demo_fw_angr_reachable.txt")
        self.assertEqual(addrs, {0x08000000, 0x800F4, 0x08000010})

    def test_explicit_env_file_overrides_convention(self):
        explicit = self.dir / "custom.txt"
        explicit.write_text("08000100\n")
        with patch.dict(
            "os.environ",
            {"LSGEMU_ANGR_REACHABILITY": "true", "LSGEMU_ANGR_REACHABLE_FILE": str(explicit)},
            clear=False,
        ):
            path, addrs = load_angr_reachable_bb_set(self.dir / "elsewhere.bin")
        self.assertEqual(path, explicit)
        self.assertEqual(addrs, {0x08000100})

    def test_enabled_but_file_missing_returns_none_pair(self):
        with patch.dict("os.environ", {"LSGEMU_ANGR_REACHABILITY": "1"}, clear=False):
            path, addrs = load_angr_reachable_bb_set(self.dir / "no_such_fw.elf")
        self.assertIsNone(path)
        self.assertIsNone(addrs)

    def test_truthy_values(self):
        for value in ("1", "true", "YES", " on "):
            with patch.dict("os.environ", {"LSGEMU_ANGR_REACHABILITY": value}, clear=False):
                self.assertTrue(angr_reachability_enabled(), value)
        for value in ("", "0", "no", "off"):
            with patch.dict("os.environ", {"LSGEMU_ANGR_REACHABILITY": value}, clear=False):
                self.assertFalse(angr_reachability_enabled(), value)


class PreparedFirmwareReachableCoverageTests(unittest.TestCase):
    """PreparedFirmware 集成：可选分母与粒度对齐规则。"""

    def _prepared(self):
        from lsgemu.prepared_firmware import PreparedFirmware

        return PreparedFirmware(
            firmware_path=Path("/tmp/demo_fw.elf"),
            result=object(),
            static_bbs={},
            static_bb_set=set(),
            # 0x104 是 0x100 BB 的内部指令：模拟 angr/静态 BB 边界错位。
            instruction_to_bb={0x100: 0x100, 0x104: 0x100, 0x108: 0x108},
            instruction_lookup={},
            branch_instruction_by_bb={},
            compare_lookup={},
            static_successors={},
            refined_basic_blocks=[],
            conditional_branch_bbs=[],
            ghidra_total_bbs=0,
        )

    def test_reachable_coverage_aligns_grain_via_instruction_to_bb(self):
        prepared = self._prepared()
        prepared.angr_reachable_bb_set = {0x100, 0x104, 0x200}
        self.assertEqual(prepared.reachable_total_bbs, 3)
        # 0x100 直接命中；0x104 落在已覆盖静态 BB 0x100 内部 -> 计命中；
        # 0x200 所在静态 BB 未覆盖 -> 不计。
        hits = prepared.reachable_coverage({0x100, 0x108})
        self.assertEqual(hits, {0x100, 0x104})

    def test_reachable_fields_absent_when_disabled(self):
        prepared = self._prepared()
        prepared.angr_reachable_bb_set = None
        self.assertIsNone(prepared.reachable_total_bbs)
        self.assertEqual(prepared.reachable_coverage({0x100}), set())
        # 缓存反序列化路径（旧 pickle 无该属性）也要安全：
        del prepared.angr_reachable_bb_set
        self.assertIsNone(prepared.reachable_total_bbs)
        self.assertEqual(prepared.reachable_coverage({0x100}), set())


class R40DenominatorDisclosureTests(unittest.TestCase):
    """r40 P4：分母可溯源 + 缺分母显式降级（不许静默 null / 空集冒充数字）。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_r40_denom_")
        self.dir = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_missing_reachable_file_logs_degradation_warning(self):
        missing_fw = self.dir / "no_such_fw.elf"
        with patch.dict(
            "os.environ", {"LSGEMU_ANGR_REACHABILITY": "1"}, clear=False
        ):
            with self.assertLogs(
                "lsgemu.angr_reachability", level="WARNING"
            ) as captured:
                path, addrs = load_angr_reachable_bb_set(missing_fw)
        self.assertIsNone(path)
        self.assertIsNone(addrs)
        self.assertIn(
            "denominator=unavailable", "\n".join(captured.output)
        )

    def test_valid_bb_denominator_unavailable_logs_warning(self):
        from lsgemu.runner_common import (
            _VALID_BB_RESOLUTION_CACHE,
            _resolve_valid_bb_metadata,
        )

        missing_fw = (self.dir / "r40_missing_fw.elf").resolve()
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("LSGEMU_VALID_BB_ROOT", None)
            with self.assertLogs(
                "lsgemu.runner_common", level="WARNING"
            ) as captured:
                path, addrs = _resolve_valid_bb_metadata(missing_fw)
        self.assertIsNone(path)
        self.assertEqual(addrs, set())
        self.assertIn(
            "denominator=unavailable", "\n".join(captured.output)
        )
        # 清缓存，避免该固件路径的降级结果泄漏到其它测试。
        _VALID_BB_RESOLUTION_CACHE.pop(str(missing_fw), None)

    def test_reachable_bb_file_sha256_helper(self):
        import hashlib
        from types import SimpleNamespace

        from lsgemu.historical_runner import HistoricalRunner

        payload = b"08000000\n08000010\n"
        reachable_file = self.dir / "frozen_angr_reachable.txt"
        reachable_file.write_bytes(payload)
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(
            angr_reachable_bb_path=reachable_file
        )
        self.assertEqual(
            runner._reachable_bb_file_sha256(),
            hashlib.sha256(payload).hexdigest(),
        )
        # 无分母文件 ⇒ None（与 reachable_* 字段 null 口径一致）。
        runner.prepared = SimpleNamespace(angr_reachable_bb_path=None)
        self.assertIsNone(runner._reachable_bb_file_sha256())


@unittest.skipIf(not _angr_available(), "本机缺少 angr")
@unittest.skipIf(not HEAT_PRESS_ELF.is_file(), f"缺少回归固件 {HEAT_PRESS_ELF}")
class RealFirmwareReachabilityTests(unittest.TestCase):
    """真实固件：Heat_Press.elf 的 angr 可达基线与 CLI 产物。"""

    @classmethod
    def setUpClass(cls):
        # 一次分析，多个用例共享（CFGEmulated 约十余秒）。
        cls.report = compute_reachable_bbs(HEAT_PRESS_ELF, timeout_seconds=240)

    def test_reachable_bb_count_positive_and_below_ghidra_total(self):
        report = self.report
        self.assertGreater(report.bb_count, 0)
        self.assertEqual(len(set(report.bb_addrs)), report.bb_count)  # 去重
        self.assertEqual(report.bb_addrs, sorted(report.bb_addrs))  # 升序
        self.assertTrue(all(addr % 2 == 0 for addr in report.bb_addrs))  # Thumb 位已清
        self.assertGreater(report.node_count, 0)
        self.assertEqual(report.method, "cfg_emulated")  # 小固件不应降级
        self.assertIsNone(report.degrade_reason)
        self.assertTrue(report.angr_version)

        # 分母对照：直接读静态缓存 pickle（避免测试内触发 Ghidra 全量重跑）。
        # 缓存缺失或为旧格式（受限 unpickler 拒载）时退回 valid_basic_blocks.txt
        # 计数做宽松上界；严格"< Ghidra 全量"对照由端到端验证兜底。
        from lsgemu.runner_common import DEFAULT_STATIC_CACHE_DIR, _load_valid_bbs

        ghidra_total = 0
        caches = sorted(
            DEFAULT_STATIC_CACHE_DIR.glob("Heat_Press_*_static_cache.pkl")
        )
        if caches:
            from lsgemu.prepared_firmware import _load_static_cache_pickle

            try:
                payload = _load_static_cache_pickle(caches[0])
                ghidra_total = int(getattr(payload["prepared"], "total_bbs", 0) or 0)
            except Exception:
                ghidra_total = 0
        denominator = ghidra_total
        if denominator <= 0:
            valid_path = HEAT_PRESS_ELF.parent / "valid_basic_blocks.txt"
            if valid_path.is_file():
                denominator = len(_load_valid_bbs(valid_path))
        self.assertGreater(
            denominator, 0, "缺少可用的静态分母（缓存与 valid BB 文件均不可读）"
        )
        self.assertLess(
            report.bb_count,
            denominator,
            "可达基线应显著小于静态全量 BB（分母修正的前提）",
        )

    def test_cli_writes_json_and_bb_file(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_reach_cli_") as tmp:
            output = Path(tmp) / "report.json"
            bb_file = Path(tmp) / "reachable.txt"
            reachability_cli_main(
                [
                    str(HEAT_PRESS_ELF),
                    "--output",
                    str(output),
                    "--bb-file-format",
                    str(bb_file),
                    "--timeout",
                    "240",
                ]
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["schema"], REACHABILITY_SCHEMA)
            self.assertEqual(payload["bb_count"], len(payload["bb_addrs"]))
            self.assertEqual(payload["bb_count"], self.report.bb_count)
            self.assertTrue(payload["bb_addrs"])
            for token in payload["bb_addrs"][:5]:
                self.assertRegex(token, r"^0x[0-9a-f]{8}$")

            lines = bb_file.read_text().splitlines()
            self.assertEqual(len(lines), self.report.bb_count)
            for token in lines[:5]:
                self.assertRegex(token, r"^[0-9a-f]{8}$")

    def test_missing_input_raises(self):
        with self.assertRaises(AngrReachabilityError):
            compute_reachable_bbs("/no/such/firmware.elf")


if __name__ == "__main__":
    unittest.main()
