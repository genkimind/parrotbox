import os
import unittest
from unittest.mock import patch

import llm_provider


class LLMProviderTests(unittest.TestCase):
    def resolve_with(self, **environment: str) -> llm_provider.LLMProvider:
        with patch.dict(os.environ, environment, clear=True):
            return llm_provider.resolve_llm_provider()

    def test_defaults_to_deepseek(self):
        provider = self.resolve_with()
        self.assertIs(provider, llm_provider.DEEPSEEK)
        self.assertEqual(provider.default_model, "deepseek-flash")

    def test_custom_url_selects_provider_neutral_profile(self):
        provider = self.resolve_with(
            LLM_BASE_URL="http://127.0.0.1:8000/v1",
            LLM_MODEL="local-model",
        )
        self.assertIs(provider, llm_provider.CUSTOM)
        self.assertEqual(provider.extra_body, {})

    def test_explicit_provider_wins(self):
        provider = self.resolve_with(
            LLM_PROVIDER="deepseek",
            LLM_BASE_URL="http://127.0.0.1:8000/v1",
        )
        self.assertIs(provider, llm_provider.DEEPSEEK)

    def test_rejects_unknown_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "expected one of"):
                llm_provider.resolve_llm_provider()

    def test_generic_key_overrides_profile_key(self):
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "generic", "DEEPSEEK_API_KEY": "specific"},
            clear=True,
        ):
            self.assertEqual(llm_provider.llm_api_key(llm_provider.DEEPSEEK), "generic")

    def test_unauthenticated_custom_endpoint_gets_client_placeholder(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_provider.llm_api_key(llm_provider.CUSTOM), "local")

    def test_custom_endpoint_requires_url_and_model(self):
        with patch.dict(os.environ, {}, clear=True):
            errors = llm_provider.llm_config_errors(llm_provider.CUSTOM)
        self.assertEqual(
            errors,
            [
                "LLM_BASE_URL is required for the custom provider",
                "LLM_MODEL is required for the custom provider",
            ],
        )

    def test_hosted_profile_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            errors = llm_provider.llm_config_errors(llm_provider.DEEPSEEK)
        self.assertEqual(len(errors), 1)
        self.assertIn("DEEPSEEK_API_KEY", errors[0])


if __name__ == "__main__":
    unittest.main()
