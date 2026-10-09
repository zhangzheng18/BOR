#!/usr/bin/env python3
"""Contracts for explicit LLM enablement and environment-owned credentials."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from lsgemu.analysis.llm_code_analyzer import LLMCodeAnalyzer
from lsgemu.llm_json_utils import create_openai_compatible_client


class LLMConfigContractTests(unittest.TestCase):
    def test_none_config_disables_code_analyzer_client(self):
        analyzer = LLMCodeAnalyzer(config_path=None, static_bbs={})

        self.assertEqual({}, analyzer.config)
        self.assertIsNone(analyzer.client)

    def test_named_api_key_environment_variable_is_honored(self):
        config = {
            "model": "test-model",
            "api_base": "http://127.0.0.1:1/v1",
            "api_key_env": "LSGEMU_TEST_LLM_KEY",
        }
        with patch.dict(
            os.environ,
            {"LSGEMU_TEST_LLM_KEY": "test-environment-key"},
            clear=True,
        ):
            client, model, backend = create_openai_compatible_client(config)

        self.assertIsNotNone(client)
        self.assertEqual("test-model", model)
        self.assertIn(backend, {"openai_sdk", "http_fallback"})
        self.assertEqual("test-environment-key", getattr(client, "api_key", ""))


if __name__ == "__main__":
    unittest.main()
