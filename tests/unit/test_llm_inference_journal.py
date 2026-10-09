#!/usr/bin/env python3
"""Regression tests for the incremental LLM inference journal.

Campaign attempts are killed with SIGTERM once ``--firmware-minutes`` expires,
which bypasses the runner's finalize stage — the only place that used to write
``<firmware>_llm_history.json``.  The journal appends every inference record to
``<firmware>_llm_history.jsonl`` the moment it is produced, so the history
survives SIGTERM/SIGKILL/native crashes.  These tests verify:

* a journal line exists on disk immediately after each inference (no flush by
  the test, no finalize);
* a subprocess killed with SIGTERM/SIGKILL leaves a complete, parseable
  journal behind;
* journal open/write failures never break the inference path;
* ``save_inference_history`` still produces the full JSON at finalize.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.llm_guide.llm_guide import LLMGuide

REPO_ROOT = Path(__file__).resolve().parents[1]

STATIC_BBS = {
    0x08001230: [
        {"address": 0x08001230, "mnemonic": "LDR", "operands": "R0, [R1, #0x04]"},
        {"address": 0x08001234, "mnemonic": "CMP", "operands": "R0, #0x20"},
        {"address": 0x08001238, "mnemonic": "BEQ", "operands": "0x08001260"},
    ],
    0x08001240: [
        {"address": 0x08001240, "mnemonic": "LDR", "operands": "R3, [R2, #0x10]"},
        {"address": 0x08001244, "mnemonic": "TST", "operands": "R3, #0x2000000"},
        {"address": 0x08001248, "mnemonic": "BNE", "operands": "0x0800124c"},
    ],
    0x08001250: [
        {"address": 0x08001250, "mnemonic": "CBZ", "operands": "R0, 0x08001258"},
    ],
}


def _read_journal_lines(path: Path) -> list[dict]:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


class IncrementalJournalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="lsgemu_journal_")
        self.journal_path = Path(self._tmp.name) / "fw_llm_history.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _guide(self) -> LLMGuide:
        return LLMGuide(STATIC_BBS, use_llm=False)

    def test_record_exists_on_disk_immediately_after_inference(self):
        guide = self._guide()
        self.assertTrue(guide.enable_inference_journal(self.journal_path))
        value = guide.infer_constraint(
            branch_pc=0x08001238,
            branch_condition="BEQ",
            target_direction=True,
            mmio_addr=0x40010004,
        )
        self.assertIsNotNone(value)
        # 此处不调用任何 flush/close：文件必须已经在磁盘上。
        self.assertTrue(self.journal_path.exists())
        lines = _read_journal_lines(self.journal_path)
        self.assertEqual(lines[0]["journal_event"], "session_start")
        self.assertEqual(lines[0]["records_before_journal"], 0)
        record = lines[1]
        self.assertEqual(record["method"], "rules")
        self.assertEqual(record["seq"], 0)
        self.assertEqual(record["branch_pc"], hex(0x08001238))
        self.assertEqual(record["mmio_addr"], hex(0x40010004))
        self.assertEqual(record["inferred_value"], hex(value))
        self.assertIn("session_id", record)
        # journal 行与内存历史一致（去掉 journal 元数据字段后）
        in_memory = dict(guide.inference_history[0])
        for key, item in record.items():
            if key not in {"seq", "session_id"}:
                self.assertEqual(in_memory[key], item)

    def test_each_inference_appends_one_line_with_monotonic_seq(self):
        guide = self._guide()
        guide.enable_inference_journal(self.journal_path)
        for pc, condition, direction in (
            (0x08001238, "BEQ", True),
            (0x08001248, "BNE", True),
            (0x08001250, "CBZ", True),
        ):
            guide.infer_constraint(pc, condition, direction, 0x40010004)
        records = [line for line in _read_journal_lines(self.journal_path)
                   if "journal_event" not in line]
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])
        self.assertEqual(len(guide.inference_history), 3)

    def test_save_inference_history_still_writes_full_json_at_finalize(self):
        guide = self._guide()
        guide.enable_inference_journal(self.journal_path)
        guide.infer_constraint(0x08001238, "BEQ", True, 0x40010004)
        guide.infer_constraint(0x08001248, "BNE", True, 0x40010004)
        full_json_path = Path(self._tmp.name) / "fw_llm_history.json"
        guide.save_inference_history(str(full_json_path))
        payload = json.loads(full_json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["method"], "rules")
        # finalize 后关闭 journal（幂等），journal 文件保持完整。
        guide.close_inference_journal()
        guide.close_inference_journal()
        self.assertEqual(
            len(_read_journal_lines(self.journal_path)), 3
        )

    def test_clear_writes_marker_and_restarts_seq(self):
        guide = self._guide()
        guide.enable_inference_journal(self.journal_path)
        guide.infer_constraint(0x08001238, "BEQ", True, 0x40010004)
        guide.clear()
        guide.infer_constraint(0x08001248, "BNE", True, 0x40010004)
        lines = _read_journal_lines(self.journal_path)
        events = [line.get("journal_event") for line in lines]
        self.assertIn("session_start", events)
        self.assertIn("clear", events)
        clear_line = next(line for line in lines if line.get("journal_event") == "clear")
        self.assertEqual(clear_line["cleared_records"], 1)
        records = [line for line in lines if "journal_event" not in line]
        self.assertEqual([r["seq"] for r in records], [0, 0])

    def test_llm_methods_are_journalled_too(self):
        guide = self._guide()
        guide.enable_inference_journal(self.journal_path)
        ok_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content='{"analysis": "a", "mmio_value": "0x20", '
                            '"confidence": 0.9, "root_cause": "unknown", '
                            '"recommended_action": "use_value"}',
                    reasoning_content="",
                ),
                finish_reason="stop",
            )],
        )
        with unittest.mock.patch.object(guide, "_call_llm_json", return_value=ok_response):
            value = guide._infer_with_llm(0x08001238, "BEQ", True, 0x40010004)
        self.assertEqual(value, 0x20)
        with unittest.mock.patch.object(
            guide, "_call_llm_json", side_effect=RuntimeError("boom")
        ):
            guide._infer_with_llm(0x08001248, "BNE", True, 0x40010004)
        records = [line for line in _read_journal_lines(self.journal_path)
                   if "journal_event" not in line]
        methods = [r["method"] for r in records]
        self.assertIn("llm", methods)
        self.assertIn("llm_error", methods)
        self.assertIn("rules", methods)  # llm_error 回退规则推断也被记录
        self.assertEqual(
            methods, [r["method"] for r in guide.inference_history]
        )

    def test_unopenable_journal_path_disables_journal_without_breaking_inference(self):
        blocker = Path(self._tmp.name) / "not_a_dir"
        blocker.write_text("occupied", encoding="utf-8")
        guide = self._guide()
        enabled = guide.enable_inference_journal(blocker / "journal.jsonl")
        self.assertFalse(enabled)
        value = guide.infer_constraint(0x08001238, "BEQ", True, 0x40010004)
        self.assertIsNotNone(value)
        self.assertEqual(len(guide.inference_history), 1)

    def test_write_failure_after_open_disables_journal_without_breaking_inference(self):
        guide = self._guide()
        guide.enable_inference_journal(self.journal_path)

        class _BrokenHandle:
            def write(self, *_args, **_kwargs):
                raise OSError("disk gone")

            def flush(self):
                raise OSError("disk gone")

            def close(self):
                pass

        guide._inference_journal_handle = _BrokenHandle()
        value = guide.infer_constraint(0x08001238, "BEQ", True, 0x40010004)
        self.assertIsNotNone(value)
        self.assertEqual(len(guide.inference_history), 1)
        self.assertTrue(guide._inference_journal_failed)
        self.assertIsNone(guide._inference_journal_handle)
        # 后续推断不受影响
        value2 = guide.infer_constraint(0x08001248, "BNE", True, 0x40010004)
        self.assertIsNotNone(value2)


_CHILD_SCRIPT = r"""
import sys
import time
from pathlib import Path

sys.path.insert(0, {repo_root!r})
from lsgemu.llm_guide.llm_guide import LLMGuide

static_bbs = {static_bbs!r}
guide = LLMGuide(static_bbs, use_llm=False)
assert guide.enable_inference_journal({journal_path!r})
for pc, condition, direction, mmio in [
    (0x08001238, "BEQ", True, 0x40010004),
    (0x08001248, "BNE", True, 0x40010004),
    (0x08001250, "CBZ", True, 0x40010004),
]:
    guide.infer_constraint(pc, condition, direction, mmio)
print("JOURNAL_READY", flush=True)
time.sleep(60)
"""


class SigtermDurabilityTests(unittest.TestCase):
    """模拟 campaign driver 的超时终止：子进程推断后被 SIGTERM/SIGKILL。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="lsgemu_journal_sig_")
        self.journal_path = Path(self._tmp.name) / "fw_llm_history.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _run_child_until_ready(self) -> subprocess.Popen:
        script = _CHILD_SCRIPT.format(
            repo_root=str(REPO_ROOT),
            static_bbs={
                int(addr): [dict(insn) for insn in insns]
                for addr, insns in STATIC_BBS.items()
            },
            journal_path=str(self.journal_path),
        )
        return subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

    def _assert_journal_complete(self):
        lines = _read_journal_lines(self.journal_path)
        self.assertEqual(lines[0]["journal_event"], "session_start")
        records = [line for line in lines if "journal_event" not in line]
        self.assertEqual(len(records), 3)
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])
        self.assertEqual([r["method"] for r in records], ["rules", "rules", "rules"])

    def test_sigterm_preserves_journal(self):
        process = self._run_child_until_ready()
        try:
            ready = process.stdout.readline()
            self.assertIn("JOURNAL_READY", ready)
            process.send_signal(signal.SIGTERM)
            return_code = process.wait(timeout=15)
            self.assertNotEqual(return_code, 0)
            self._assert_journal_complete()
        finally:
            if process.poll() is None:
                process.kill()

    def test_sigkill_preserves_journal(self):
        process = self._run_child_until_ready()
        try:
            ready = process.stdout.readline()
            self.assertIn("JOURNAL_READY", ready)
            process.kill()
            return_code = process.wait(timeout=15)
            self.assertNotEqual(return_code, 0)
            self._assert_journal_complete()
        finally:
            if process.poll() is None:
                process.kill()


if __name__ == "__main__":
    unittest.main()
