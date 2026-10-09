#!/usr/bin/env python3
"""Shared helpers for robust JSON-oriented LLM calls."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Optional

import yaml

_LOG = logging.getLogger(__name__)

_JSON_MODE_WARNED: set[str] = set()
_CLIENT_INIT_WARNED: set[str] = set()
_EMPTY_CONTENT_WARNED: set[str] = set()
_REASONING_TOKENS_WARNED: set[str] = set()

# Reasoning models (DeepSeek v4/reasoner, qwq, o1/o3/o4 …) spend completion
# tokens on the chain of thought before the final answer, and providers count
# both against `max_tokens`. A small budget therefore truncates `content` to
# an empty string while `reasoning_content` holds a long chain of thought.
# Real campaign prompts produced ~29k characters of chain of thought, which
# already exceeds a 16000-token budget, so the adaptive floor and the budget
# escalation retry below both operate on larger scales.
DEFAULT_LLM_MAX_TOKENS = 4000
REASONING_MODEL_MIN_MAX_TOKENS = 8192
REASONING_MODEL_MAX_TOKENS_CEILING = 65536
_BUDGET_ESCALATION_WARNED: set[str] = set()
_REASONING_MODEL_MARKERS = (
    "reasoner",
    "thinking",
    "qwq",
    "deepseek-r1",
    "deepseek-v4",
)


def looks_like_reasoning_model(model: str) -> bool:
    lowered = str(model or "").lower()
    if any(marker in lowered for marker in _REASONING_MODEL_MARKERS):
        return True
    # o1/o3/o4 need boundary checks so names like "qwen3" or "gpt-4o" don't match.
    return bool(re.search(r"(?:^|[^a-z0-9])(o1|o3|o4)(?:[^a-z0-9]|$)", lowered))


class OpenAICompatibleHTTPClient:
    """Small stdlib client for OpenAI-compatible chat/completions services.

    The project should not silently disable LLM reasoning just because the
    optional OpenAI SDK is unavailable or incompatible with the Python runtime.
    This client intentionally implements only the subset used by
    `call_llm_json`: `client.chat.completions.create(...)`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.timeout = float(timeout or 60.0)
        self.extra_headers = dict(extra_headers or {})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _create_chat_completion(self, **kwargs):
        if not self.api_key:
            raise RuntimeError("missing API key for OpenAI-compatible LLM client")
        if not self.base_url:
            raise RuntimeError("missing base_url for OpenAI-compatible LLM client")

        payload = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        encoded = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.extra_headers)
        request = urllib.request.Request(
            self._endpoint(),
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM HTTP request failed: status={exc.code} body={body[:500]}"
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM HTTP request failed: {exc}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM HTTP response was not JSON: {exc}: {raw[:500]!r}")
        try:
            message_data = data["choices"][0]["message"]
        except Exception as exc:
            raise RuntimeError(f"LLM HTTP response missing choices[0].message: {data!r}") from exc

        # Reasoning models put the chain of thought in `reasoning_content`
        # and may return `content: null` when the token budget runs out.
        # Keep both fields so extract_response_text can diagnose/salvage.
        message = SimpleNamespace(
            content=str(message_data.get("content") or ""),
            reasoning_content=str(message_data.get("reasoning_content") or ""),
        )
        try:
            finish_reason = str(data["choices"][0].get("finish_reason") or "")
        except Exception:
            finish_reason = ""
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], raw_response=data)


def _flat_llm_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    nested = config.get("llm")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(config)


def _config_or_env(config: Dict[str, Any], key: str, env_names: Iterable[str]) -> str:
    value = config.get(key)
    if value:
        return str(value)
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            return str(env_value)
    return ""


def create_openai_compatible_client(
    config: Optional[Dict[str, Any]],
    *,
    logger=None,
    warn_key: str = "llm",
    default_model: str = "qwen-plus",
):
    """Create an SDK or stdlib OpenAI-compatible client.

    Returns `(client, model, backend)`.  `client` is `None` when the config is
    unusable.  The fallback is important on older Python environments where the
    OpenAI SDK dependency chain may not import cleanly.
    """
    cfg = _flat_llm_config(config)
    model = str(cfg.get("model") or default_model)
    configured_api_key_env = str(cfg.get("api_key_env") or "").strip()
    api_key_env_names = tuple(dict.fromkeys(
        name
        for name in (
            configured_api_key_env,
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "OPENAI_API_KEY",
        )
        if name
    ))
    api_key = _config_or_env(
        cfg,
        "api_key",
        api_key_env_names,
    )
    api_base = _config_or_env(
        cfg,
        "api_base",
        ("OPENAI_BASE_URL", "OPENAI_API_BASE", "DASHSCOPE_API_BASE"),
    )
    timeout = cfg.get("timeout", cfg.get("request_timeout", 60.0))

    if not api_key:
        key = f"{warn_key}:missing_api_key"
        if key not in _CLIENT_INIT_WARNED:
            if logger is not None:
                logger.warning("LLM配置缺少api_key，禁用真实LLM调用")
            _CLIENT_INIT_WARNED.add(key)
        return None, model, "disabled_missing_api_key"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_base or None)
        return client, model, "openai_sdk"
    except Exception as exc:
        key = f"{warn_key}:sdk:{type(exc).__name__}:{exc}"
        if key not in _CLIENT_INIT_WARNED:
            if logger is not None:
                logger.warning("OpenAI SDK不可用，改用标准库HTTP兼容客户端: %s", exc)
            _CLIENT_INIT_WARNED.add(key)

    if not api_base:
        key = f"{warn_key}:missing_api_base"
        if key not in _CLIENT_INIT_WARNED:
            if logger is not None:
                logger.warning("LLM配置缺少api_base，无法使用HTTP fallback")
            _CLIENT_INIT_WARNED.add(key)
        return None, model, "disabled_missing_api_base"

    try:
        client = OpenAICompatibleHTTPClient(
            api_key=api_key,
            base_url=api_base,
            timeout=float(timeout or 60.0),
        )
        return client, model, "http_fallback"
    except Exception as exc:
        key = f"{warn_key}:fallback:{type(exc).__name__}:{exc}"
        if key not in _CLIENT_INIT_WARNED:
            if logger is not None:
                logger.warning("LLM HTTP fallback初始化失败: %s", exc)
            _CLIENT_INIT_WARNED.add(key)
        return None, model, "disabled_init_failed"


def extract_response_reasoning(response) -> str:
    """Return the reasoning model's chain-of-thought text, if any."""
    try:
        return str(response.choices[0].message.reasoning_content or "").strip()
    except Exception:
        return ""


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced JSON object found in `text`, or "".

    Unlike the greedy `\\{.*\\}` regex in parse_json_object this respects
    braces inside strings and stops at the first complete object, which is
    what salvaging a chain of thought needs.
    """
    source = str(text or "")
    start = source.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(source)):
            ch = source[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[start:idx + 1]
        start = source.find("{", start + 1)
    return ""


def _diagnose_empty_content_once(response, reasoning: str) -> None:
    """Log a clear 'reasoning consumed the budget' diagnosis (once per cause)."""
    model = ""
    finish_reason = ""
    try:
        model = str(getattr(response, "model", "") or "")
        finish_reason = str(response.choices[0].finish_reason or "")
    except Exception:
        pass
    key = f"{model}:{finish_reason}"
    if key in _EMPTY_CONTENT_WARNED:
        return
    _EMPTY_CONTENT_WARNED.add(key)
    _LOG.error(
        "LLM返回content为空: reasoning_content长度=%d, finish_reason=%s, model=%s "
        "— 推理模型的思维链与最终答案共用max_tokens预算，content为空通常意味着"
        "预算被reasoning耗尽（finish_reason=length）或响应被截断，请调大max_tokens",
        len(reasoning),
        finish_reason or "unknown",
        model or "unknown",
    )


def extract_response_text(response) -> str:
    """Extract the model's final answer text from a chat-completions response.

    `content` wins. When it is empty but a reasoning model left a chain of
    thought in `reasoning_content`, only the embedded JSON object is salvaged
    — never the whole chain — and a diagnosis is logged when even that fails.
    """
    try:
        content = str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if content:
        return content
    reasoning = extract_response_reasoning(response)
    if not reasoning:
        return ""
    salvaged = _extract_first_json_object(reasoning)
    if salvaged:
        return salvaged
    _diagnose_empty_content_once(response, reasoning)
    return ""


def _strip_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def repair_json_like_text(text: str) -> Optional[Dict[str, Any]]:
    candidate = _strip_fences(text)
    if not candidate:
        return None
    brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0).strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)
    repaired = repaired.replace("\r", " ").replace("\n", " ")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)", r'\1"\2"\3', repaired)
    repaired = re.sub(r":\s*0x([0-9a-fA-F]+)(\s*[,}])", r': "0x\1"\2', repaired)

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(repaired)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _yaml_parse_is_faithful(parsed: Dict[str, Any], candidate: str) -> bool:
    """Guard the YAML fallback in parse_json_object.

    YAML semantics are far looser than JSON: values containing ``#`` are
    truncated as comments, ``no/yes/on/off`` are reinterpreted as booleans,
    and unterminated structures are silently "repaired". Accept a YAML result
    only when it round-trips back to JSON whose keys all appear verbatim in
    the original candidate text — anything else is likely content the LLM
    never produced.
    """
    try:
        rendered = json.dumps(parsed, ensure_ascii=False)
        reparsed = json.loads(rendered)
    except (TypeError, ValueError):
        return False
    if not isinstance(reparsed, dict):
        return False
    _YAML_BOOL_WORDS = {"true", "false", "yes", "no", "on", "off", "y", "n"}
    for key, value in reparsed.items():
        if not isinstance(key, str) or key not in candidate:
            return False
        # Reject YAML boolean coercion: an unquoted yes/no/on/off/off-style
        # word in the source becomes a Python bool, which JSON would never
        # produce from the same text.
        if isinstance(value, bool):
            return False
        # Also require every string leaf value to appear verbatim in the
        # original candidate: YAML truncates values at '#' comments and
        # coerces yes/no/on/off to booleans, so a faithful parse cannot
        # invent or shorten strings the LLM actually emitted.
        if isinstance(value, str) and value not in candidate:
            return False
    return True


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")

    candidates = [_strip_fences(text), str(text).strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", str(text), re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    braced = re.search(r"\{.*\}", str(text), re.DOTALL)
    if braced:
        candidates.append(braced.group(0).strip())

    seen = set()
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = yaml.safe_load(candidate)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and _yaml_parse_is_faithful(parsed, candidate):
                return parsed
            repaired = repair_json_like_text(candidate)
            if isinstance(repaired, dict):
                return repaired
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"failed to parse JSON object from response: {str(text)[:400]!r}")


def extract_json_string_field(text: str, field_name: str) -> Optional[str]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"'
    match = re.search(pattern, str(text), re.IGNORECASE)
    if not match:
        return None

    start = match.end()
    chars = []
    escaped = False
    source = str(text)
    for idx in range(start, len(source)):
        ch = source[idx]
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            break
        chars.append(ch)

    parsed = "".join(chars).strip()
    if not parsed:
        return None
    return parsed.replace("\n", " ").replace("\r", " ")


def extract_unquoted_field_text(text: str, field_name: str) -> Optional[str]:
    patterns = (
        rf'["\']?{re.escape(field_name)}["\']?\s*:\s*([^,\n\r}}]+)',
        rf'{re.escape(field_name)}\s*[:=]\s*([^,\n\r}}]+)',
    )
    source = str(text)
    for pattern in patterns:
        match = re.search(pattern, source, re.IGNORECASE)
        if not match:
            continue
        parsed = match.group(1).strip().strip('"\'')
        parsed = re.sub(r"\s+", " ", parsed)
        if parsed:
            return parsed
    return None


def salvage_named_fields(
    text: str,
    *,
    numeric_fields: Iterable[str] = (),
    text_fields: Iterable[str] = (),
) -> Optional[Dict[str, Any]]:
    source = str(text or "")
    result: Dict[str, Any] = {}

    for field in numeric_fields:
        field_match = None
        patterns = (
            rf'["\']?{re.escape(field)}["\']?\s*:\s*["\']?(-?0x[0-9a-fA-F]+|-?\d+)["\']?',
            rf'{re.escape(field)}\s*[:=]\s*["\']?(-?0x[0-9a-fA-F]+|-?\d+)["\']?',
        )
        for pattern in patterns:
            field_match = re.search(pattern, source, re.IGNORECASE)
            if field_match:
                break
        if field_match:
            result[field] = field_match.group(1)

    for field in text_fields:
        parsed = extract_json_string_field(source, field)
        if not parsed:
            parsed = extract_unquoted_field_text(source, field)
        if parsed:
            result[field] = parsed

    return result or None


def resolve_request_max_tokens(model: str, max_tokens: Optional[int], logger=None) -> int:
    """Return the max_tokens actually sent for `model`.

    Unset/invalid values default to DEFAULT_LLM_MAX_TOKENS. For reasoning
    models the budget is shared between the chain of thought and the final
    answer, so a small configured value is adaptively raised to
    REASONING_MODEL_MIN_MAX_TOKENS (with a one-shot warning) — otherwise the
    request is near-guaranteed to come back with an empty `content`.
    """
    try:
        configured = int(max_tokens) if max_tokens is not None else 0
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        return DEFAULT_LLM_MAX_TOKENS
    if looks_like_reasoning_model(model) and configured < REASONING_MODEL_MIN_MAX_TOKENS:
        key = f"{model}:{configured}"
        if key not in _REASONING_TOKENS_WARNED:
            _REASONING_TOKENS_WARNED.add(key)
            if logger is not None:
                logger.warning(
                    "推理模型 %s 的 max_tokens=%d 过小（思维链与最终答案共用预算，"
                    "content 会被截断成空），自适应提升到 %d",
                    model,
                    configured,
                    REASONING_MODEL_MIN_MAX_TOKENS,
                )
            else:
                _LOG.warning(
                    "推理模型 %s 的 max_tokens=%d 过小（思维链与最终答案共用预算，"
                    "content 会被截断成空），自适应提升到 %d",
                    model,
                    configured,
                    REASONING_MODEL_MIN_MAX_TOKENS,
                )
        return REASONING_MODEL_MIN_MAX_TOKENS
    return configured


def reasoning_budget_ceiling() -> int:
    """Upper bound for the automatic reasoning-budget escalation retry."""
    raw = str(os.environ.get("LSGEMU_REASONING_MAX_TOKENS_CEILING") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return REASONING_MODEL_MAX_TOKENS_CEILING


def _hit_token_limit_without_content(response) -> bool:
    """Detect the 'reasoning consumed the whole budget' response signature."""
    try:
        content = str(response.choices[0].message.content or "").strip()
        finish_reason = str(response.choices[0].finish_reason or "").strip().lower()
    except Exception:
        return False
    return not content and finish_reason == "length"


def _create_completion(
    client,
    kwargs: Dict[str, Any],
    *,
    model: str,
    warn_key: Optional[str],
    logger=None,
):
    """create() with json_object mode, falling back when unsupported."""
    try:
        return client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except TypeError:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "response_format" in message or "json_object" in message:
            key = warn_key or f"{model}:{message}"
            if key not in _JSON_MODE_WARNED:
                if logger is not None:
                    logger.warning("JSON模式不可用，回退普通completion: %s", exc)
                else:
                    _LOG.warning("JSON模式不可用，回退普通completion: %s", exc)
                _JSON_MODE_WARNED.add(key)
            return client.chat.completions.create(**kwargs)
        raise


def call_llm_json(
    *,
    client,
    model: str,
    messages,
    max_tokens: Optional[int],
    temperature: float,
    repair_prompt: Optional[str] = None,
    parse_response: Optional[Callable[[str], Any]] = None,
    logger=None,
    warn_key: Optional[str] = None,
):
    effective_max_tokens = resolve_request_max_tokens(model, max_tokens, logger=logger)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
    }

    response = _create_completion(client, kwargs, model=model, warn_key=warn_key, logger=logger)

    # Reasoning-budget escalation: when the chain of thought consumed the
    # whole budget (empty content + finish_reason=length), retry once with a
    # doubled budget (bounded by the ceiling). DeepSeek v4-flash needed
    # >16000 completion tokens on real campaign prompts, so a configured
    # budget that looks generous can still be exhausted.
    if (
        looks_like_reasoning_model(model)
        and _hit_token_limit_without_content(response)
        and effective_max_tokens < reasoning_budget_ceiling()
    ):
        escalated = min(reasoning_budget_ceiling(), effective_max_tokens * 2)
        key = f"{model}:{effective_max_tokens}->{escalated}"
        if key not in _BUDGET_ESCALATION_WARNED:
            _BUDGET_ESCALATION_WARNED.add(key)
            (logger or _LOG).warning(
                "推理模型 %s 的 max_tokens=%d 被思维链耗尽"
                "（content为空, finish_reason=length），翻倍到 %d 重试",
                model,
                effective_max_tokens,
                escalated,
            )
        kwargs["max_tokens"] = escalated
        response = _create_completion(client, kwargs, model=model, warn_key=warn_key, logger=logger)

    if not repair_prompt:
        return response

    parser = parse_response or parse_json_object
    content = extract_response_text(response)
    if not content:
        return response

    try:
        parser(content)
        return response
    except Exception:
        repair_messages = list(messages) + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": repair_prompt},
        ]
        repair_kwargs = dict(kwargs)
        repair_kwargs["messages"] = repair_messages
        repair_kwargs["temperature"] = 0.0
        return client.chat.completions.create(**repair_kwargs)
