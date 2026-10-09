#!/usr/bin/env python3
"""Regression tests for reasoning-model responses and LLM config path resolution.

Covers two campaign failures observed with DeepSeek v4-flash:
1. `LSGEMU_LLM_CONFIG` relative paths not found from a non-repo cwd.
2. Reasoning models returning `reasoning_content` while `content` is empty
   (the completion-token budget is shared between chain of thought and the
   final answer), which used to surface as `ValueError("empty response")`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu import llm_json_utils
from lsgemu.analysis.llm_code_analyzer import LLMCodeAnalyzer
from lsgemu.deployment_config import configured_path
from lsgemu.llm_json_utils import (
    DEFAULT_LLM_MAX_TOKENS,
    REASONING_MODEL_MIN_MAX_TOKENS,
    OpenAICompatibleHTTPClient,
    call_llm_json,
    extract_response_reasoning,
    extract_response_text,
    looks_like_reasoning_model,
    parse_json_object,
    reasoning_budget_ceiling,
    resolve_request_max_tokens,
)
from lsgemu.llm_guide.llm_guide import LLMGuide

REPO_ROOT = Path(__file__).resolve().parents[2]


def _response(content, reasoning="", finish_reason="stop", model="deepseek-v4-flash"):
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model)


class _RecordingClient:
    """Minimal stand-in for an OpenAI-compatible client."""

    def __init__(self, content="", reasoning=""):
        self._content = content
        self._reasoning = reasoning
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content, reasoning_content=self._reasoning)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], model=kwargs.get("model"))


class ExtractResponseTextTests(unittest.TestCase):
    def setUp(self):
        llm_json_utils._EMPTY_CONTENT_WARNED.clear()

    def test_content_with_json_wins_over_reasoning(self):
        response = _response('{"a": 1}', reasoning="很长的思维链，里面甚至有 {\"干扰\": \"括号\"}")
        self.assertEqual(parse_json_object(extract_response_text(response)), {"a": 1})
        self.assertEqual(extract_response_reasoning(response).startswith("很长"), True)

    def test_empty_content_salvages_json_object_from_reasoning(self):
        reasoning = (
            "分支是 CBZ，等待 PLL 就绪位。先看 0x40021000 的 bit0……"
            '结论是返回 {"mmio_value": "0x1", "confidence": 0.9} 即可让分支 taken。'
            "再检查一遍 32 位范围没有问题。"
        )
        response = _response("", reasoning=reasoning, finish_reason="length")
        parsed = parse_json_object(extract_response_text(response))
        self.assertEqual(parsed["mmio_value"], "0x1")

    def test_both_empty_returns_empty_string(self):
        self.assertEqual(extract_response_text(_response("", "")), "")
        self.assertEqual(extract_response_text(_response(None, None)), "")

    def test_unsalvageable_reasoning_returns_empty_and_logs_diagnosis(self):
        response = _response("", reasoning="纯文字思维链，没有任何 JSON。", finish_reason="length")
        with self.assertLogs("lsgemu.llm_json_utils", level="ERROR") as captured:
            text = extract_response_text(response)
        self.assertEqual(text, "")
        joined = "\n".join(captured.output)
        self.assertIn("reasoning_content长度=", joined)
        self.assertIn("finish_reason=length", joined)
        self.assertIn("deepseek-v4-flash", joined)

    def test_structurally_broken_response_returns_empty(self):
        self.assertEqual(extract_response_text(SimpleNamespace(choices=[])), "")
        self.assertEqual(extract_response_text(object()), "")


class ReasoningMaxTokensTests(unittest.TestCase):
    def test_reasoning_model_detection(self):
        self.assertTrue(looks_like_reasoning_model("deepseek-v4-flash"))
        self.assertTrue(looks_like_reasoning_model("deepseek-reasoner"))
        self.assertTrue(looks_like_reasoning_model("QwQ-32B"))
        self.assertTrue(looks_like_reasoning_model("o1-mini"))
        self.assertFalse(looks_like_reasoning_model("qwen-plus"))
        self.assertFalse(looks_like_reasoning_model("gpt-4o"))
        self.assertFalse(looks_like_reasoning_model("qwen3-32b"))

    def test_unset_budget_defaults_to_large_value(self):
        self.assertEqual(resolve_request_max_tokens("deepseek-v4-flash", None), DEFAULT_LLM_MAX_TOKENS)
        self.assertEqual(resolve_request_max_tokens("qwen-plus", 0), DEFAULT_LLM_MAX_TOKENS)

    def test_reasoning_model_small_budget_is_raised(self):
        self.assertEqual(
            resolve_request_max_tokens("deepseek-v4-flash", 500),
            REASONING_MODEL_MIN_MAX_TOKENS,
        )

    def test_explicit_large_budget_and_non_reasoning_models_are_kept(self):
        # 高于自适应下限的显式预算保持不变（下限本身由
        # test_reasoning_floor_raised_for_real_scale_chains 验证 ≥8192）。
        self.assertEqual(resolve_request_max_tokens("deepseek-v4-flash", 16000), 16000)
        self.assertEqual(resolve_request_max_tokens("qwen-plus", 500), 500)

    def test_call_llm_json_sends_adjusted_budget(self):
        client = _RecordingClient(content='{"a": 1}')
        call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=500,
            temperature=0.0,
        )
        self.assertEqual(client.calls[0]["max_tokens"], REASONING_MODEL_MIN_MAX_TOKENS)


class MockedClientResponseFlowTests(unittest.TestCase):
    """The exact two-response scenario from the campaign failure."""

    def setUp(self):
        llm_json_utils._EMPTY_CONTENT_WARNED.clear()

    def test_content_present_parses_successfully(self):
        client = _RecordingClient(content='{"a": 1}', reasoning="思维链……")
        response = call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=2000,
            temperature=0.0,
            repair_prompt="repair",
            parse_response=parse_json_object,
        )
        self.assertEqual(parse_json_object(extract_response_text(response)), {"a": 1})

    def test_content_empty_raises_with_diagnostic_log(self):
        client = _RecordingClient(content="", reasoning="没有 JSON 的思维链")
        with self.assertLogs("lsgemu.llm_json_utils", level="ERROR") as captured:
            with self.assertRaises(ValueError) as ctx:
                response = call_llm_json(
                    client=client,
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=2000,
                    temperature=0.0,
                    repair_prompt="repair",
                    parse_response=parse_json_object,
                )
                parse_json_object(extract_response_text(response))
        self.assertIn("empty response", str(ctx.exception))
        self.assertTrue(
            any("reasoning_content长度=" in line for line in captured.output),
            captured.output,
        )

    def test_content_empty_with_salvageable_reasoning_parses(self):
        client = _RecordingClient(content="", reasoning='分析……最终 {"mmio_value": "0x20"} 完毕')
        response = call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=2000,
            temperature=0.0,
            repair_prompt="repair",
            parse_response=parse_json_object,
        )
        self.assertEqual(parse_json_object(extract_response_text(response))["mmio_value"], "0x20")


class HTTPFallbackReasoningTests(unittest.TestCase):
    """The stdlib HTTP fallback must keep reasoning_content instead of dropping it."""

    class _Handler(BaseHTTPRequestHandler):
        requests = []

        def log_message(self, fmt, *args):  # pragma: no cover - keep output stable
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8")
            type(self).requests.append(json.loads(body))
            response = {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "推理链…… 最终结论 {\"value\": \"0x00000020\"}。",
                        },
                        "finish_reason": "length",
                    }
                ]
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    def test_reasoning_content_is_preserved_and_salvaged(self):
        server = HTTPServer(("127.0.0.1", 0), self._Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleHTTPClient(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                timeout=5.0,
            )
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=500,
            )
        finally:
            server.shutdown()
            thread.join(timeout=2.0)

        self.assertEqual(response.choices[0].message.content, "")
        self.assertIn("0x00000020", response.choices[0].message.reasoning_content)
        self.assertEqual(response.choices[0].finish_reason, "length")
        parsed = parse_json_object(extract_response_text(response))
        self.assertEqual(parsed["value"], "0x00000020")


class LLMGuideMaxTokensTests(unittest.TestCase):
    def test_reads_max_tokens_from_config(self):
        guide = LLMGuide({}, use_llm=False, llm_config={"llm": {"model": "m", "max_tokens": 777}})
        self.assertEqual(guide.llm_max_tokens, 777)

    def test_flat_config_shape_is_also_supported(self):
        guide = LLMGuide({}, use_llm=False, llm_config={"max_tokens": 888})
        self.assertEqual(guide.llm_max_tokens, 888)

    def test_defaults_when_unset(self):
        self.assertEqual(LLMGuide({}, use_llm=False).llm_max_tokens, DEFAULT_LLM_MAX_TOKENS)


class LoadConfigPathResolutionTests(unittest.TestCase):
    """Bug 1: relative config paths must resolve regardless of process cwd."""

    @staticmethod
    def _make_analyzer():
        return LLMCodeAnalyzer.__new__(LLMCodeAnalyzer)

    def test_relative_path_resolves_from_repo_root_when_cwd_differs(self):
        config_name = "LLM.test_relative_resolution_tmp.yaml"
        config_file = REPO_ROOT / config_name
        config_file.write_text("llm:\n  model: test-relative-model\n  max_tokens: 123\n", encoding="utf-8")
        original_cwd = os.getcwd()
        tmp_cwd = tempfile.mkdtemp(prefix="lsgemu_llm_cfg_")
        try:
            os.chdir(tmp_cwd)
            config = self._make_analyzer()._load_config(config_name)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp_cwd, ignore_errors=True)
            config_file.unlink(missing_ok=True)
        self.assertEqual(config.get("model"), "test-relative-model")
        self.assertEqual(config.get("max_tokens"), 123)

    def test_absolute_path_loads_from_any_cwd(self):
        fd, path = tempfile.mkstemp(prefix="lsgemu_llm_abs_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("llm:\n  model: test-absolute-model\n")
            original_cwd = os.getcwd()
            tmp_cwd = tempfile.mkdtemp(prefix="lsgemu_llm_cfg_")
            try:
                os.chdir(tmp_cwd)
                config = self._make_analyzer()._load_config(path)
            finally:
                os.chdir(original_cwd)
                shutil.rmtree(tmp_cwd, ignore_errors=True)
        finally:
            os.unlink(path)
        self.assertEqual(config.get("model"), "test-absolute-model")

    def test_missing_config_logs_error_with_candidate_paths(self):
        missing_name = "LLM.definitely_missing_for_regression_test.yaml"
        original_cwd = os.getcwd()
        tmp_cwd = tempfile.mkdtemp(prefix="lsgemu_llm_cfg_")
        try:
            os.chdir(tmp_cwd)
            with self.assertLogs("lsgemu.analysis.llm_code_analyzer", level="ERROR") as captured:
                config = self._make_analyzer()._load_config(missing_name)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp_cwd, ignore_errors=True)
        self.assertEqual(config, {})
        joined = "\n".join(captured.output)
        self.assertIn("配置文件不存在", joined)
        self.assertIn(str(REPO_ROOT / missing_name), joined)

    def test_empty_config_path_returns_empty_dict(self):
        self.assertEqual(self._make_analyzer()._load_config(""), {})


class _ScriptedClient:
    """Returns one scripted response per call, recording every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected extra LLM call")
        return self._responses.pop(0)


def _length_exhausted_response(reasoning="很长的思维链……没有完整 JSON……"):
    message = SimpleNamespace(content="", reasoning_content=reasoning)
    choice = SimpleNamespace(message=message, finish_reason="length")
    return SimpleNamespace(choices=[choice], model="deepseek-v4-flash")


class ReasoningBudgetEscalationTests(unittest.TestCase):
    """Empty content + finish_reason=length must retry with a doubled budget."""

    def setUp(self):
        llm_json_utils._BUDGET_ESCALATION_WARNED.clear()

    def test_budget_doubles_on_length_exhausted_empty_content(self):
        client = _ScriptedClient([
            _length_exhausted_response(),
            _response('{"a": 1}'),
        ])
        response = call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=16000,
            temperature=0.0,
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 16000)
        self.assertEqual(client.calls[1]["max_tokens"], 32000)
        self.assertEqual(parse_json_object(extract_response_text(response)), {"a": 1})

    def test_retry_budget_is_capped_by_ceiling_env(self):
        client = _ScriptedClient([
            _length_exhausted_response(),
            _response('{"a": 1}'),
        ])
        with patch.dict(os.environ, {"LSGEMU_REASONING_MAX_TOKENS_CEILING": "20000"}):
            response = call_llm_json(
                client=client,
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=16000,
                temperature=0.0,
            )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["max_tokens"], 20000)
        self.assertIsNotNone(response)

    def test_no_retry_when_budget_already_at_ceiling(self):
        client = _ScriptedClient([_length_exhausted_response()])
        response = call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=reasoning_budget_ceiling(),
            temperature=0.0,
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(extract_response_text(response), "")

    def test_no_retry_when_content_present(self):
        client = _ScriptedClient([_response('{"a": 1}', reasoning="思维链")])
        call_llm_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=16000,
            temperature=0.0,
        )
        self.assertEqual(len(client.calls), 1)

    def test_no_retry_for_non_reasoning_models(self):
        client = _ScriptedClient([_length_exhausted_response()])
        call_llm_json(
            client=client,
            model="qwen-plus",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=16000,
            temperature=0.0,
        )
        self.assertEqual(len(client.calls), 1)

    def test_reasoning_floor_raised_for_real_scale_chains(self):
        # 真实 campaign 观测 ~29k 字符思维链，2000 的自适应下限远远不够。
        self.assertGreaterEqual(REASONING_MODEL_MIN_MAX_TOKENS, 8192)
        self.assertEqual(
            resolve_request_max_tokens("deepseek-v4-flash", 500),
            REASONING_MODEL_MIN_MAX_TOKENS,
        )


class LLMGuideReasoningSalvageTests(unittest.TestCase):
    """LLMGuide must salvage fields from a truncated chain of thought."""

    def test_mmio_value_salvaged_from_unbalanced_truncated_reasoning(self):
        guide = LLMGuide({}, use_llm=False)
        reasoning = (
            "分支等待的是状态位。先检查 TST 掩码 0x2000000……"
            "结论应该是 mmio_value: 0x20 才能让 CMP 相等成立，"
            "然后继续验证 width……"  # 无平衡 JSON，模拟 finish_reason=length 截断
        )
        response = _response("", reasoning=reasoning, finish_reason="length")
        with patch.object(guide, "_call_llm_json", return_value=response):
            value = guide._infer_with_llm(0x08001238, "BEQ", True, 0x40010004)
        self.assertEqual(value, 0x20)
        history = guide.get_inference_history()
        self.assertEqual(history[-1]["method"], "llm")
        self.assertIn("0x20", history[-1]["llm_full_response"])

    def test_unsalvageable_reasoning_still_falls_back_to_rules(self):
        guide = LLMGuide({}, use_llm=False)
        response = _response("", reasoning="纯文字思维链，没有任何字段。", finish_reason="length")
        with patch.object(guide, "_call_llm_json", return_value=response):
            value = guide._infer_with_llm(0x08001238, "BEQ", True, 0x40010004)
        self.assertIsNotNone(value)
        methods = [item["method"] for item in guide.get_inference_history()]
        self.assertIn("llm_error", methods)
        self.assertIn("rules", methods)


class DeepSeekConfigRegressionTests(unittest.TestCase):
    """The campaign's live DeepSeek config must keep the verified settings."""

    def test_model_is_stable_alias_and_budget_has_headroom(self):
        config = yaml.safe_load((REPO_ROOT / "LLM.deepseek.yaml").read_text(encoding="utf-8"))
        llm = config["llm"]
        # 项目负责人指定使用别名 "deepseek-v4.1-flash-expires-on-0910"：服务端会把它
        # 回显为规范名 deepseek-flash（GET /v1/models 只列出 deepseek-flash /
        # deepseek-v4-pro），两者是同一模型。此断言锁定"campaign 配置里必须是这个
        # 已实测可用的模型串"，防止被静默换名字（2026-09-18 决断）。
        self.assertEqual(llm["model"], "deepseek-v4.1-flash-expires-on-0910")
        # 实测观测到的真实任务推理链 ~29k 字符，预算必须显著高于 16000。
        self.assertGreaterEqual(int(llm["max_tokens"]), 65536)


class ConfiguredPathResolutionTests(unittest.TestCase):
    def test_relative_env_value_falls_back_to_repo_root_when_missing_from_cwd(self):
        config_name = "LLM.test_configured_path_tmp.yaml"
        config_file = REPO_ROOT / config_name
        config_file.write_text("llm:\n  model: x\n", encoding="utf-8")
        original_cwd = os.getcwd()
        tmp_cwd = tempfile.mkdtemp(prefix="lsgemu_llm_cfg_")
        try:
            os.chdir(tmp_cwd)
            with patch.dict(os.environ, {"LSGEMU_TEST_LLM_CONFIG": config_name}):
                resolved = configured_path("LSGEMU_TEST_LLM_CONFIG", "fallback.yaml")
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp_cwd, ignore_errors=True)
            config_file.unlink(missing_ok=True)
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, config_file.resolve())

    def test_absolute_env_value_is_returned_resolved(self):
        with patch.dict(os.environ, {"LSGEMU_TEST_LLM_CONFIG": str(REPO_ROOT / "LLM.deepseek.yaml")}):
            resolved = configured_path("LSGEMU_TEST_LLM_CONFIG", "fallback.yaml")
        self.assertTrue(resolved.is_absolute())

    def test_fallback_is_resolved_absolute(self):
        resolved = configured_path("LSGEMU_TEST_UNSET_ENV_KEY_XYZ", "relative_fallback.yaml")
        self.assertTrue(resolved.is_absolute())


if __name__ == "__main__":
    unittest.main()
