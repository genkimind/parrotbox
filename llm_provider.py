"""Resolve the OpenAI-compatible LLM endpoint from the environment.

Two hosted profiles are selected by API key (or explicitly via
``LLM_PROVIDER``):

  * `deepseek`   — the default: api.deepseek.com direct. Lowest measured TTFT
    from this network (~1.05 s p50, see ARCHITECTURE.md latency budget).
  * `openrouter` — openrouter.ai gateway: one key routes to any of their
    models; useful as the low-TTFT fallback path or to A/B providers without
    touching code. Model slugs are provider-prefixed
    (`deepseek/deepseek-v4-flash`).

Set ``LLM_BASE_URL`` and ``LLM_MODEL`` to use any other OpenAI-compatible
endpoint, including a local server. ``LLM_API_KEY`` supplies generic auth and
overrides a profile-specific key. Local servers that do not authenticate are
given the harmless placeholder key required by the OpenAI client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProvider:
    name: str
    api_key_env: str
    base_url: str
    default_model: str
    # Request tweaks that disable "thinking"/reasoning tokens — they add
    # hundreds of ms before the first audible word and buy nothing in a voice
    # turn. Each provider spells this differently; models without thinking
    # ignore the parameter.
    extra_body: dict


DEEPSEEK = LLMProvider(
    name="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    default_model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

OPENROUTER = LLMProvider(
    name="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
    default_model="deepseek/deepseek-v4-flash",
    extra_body={"reasoning": {"enabled": False}},
)

CUSTOM = LLMProvider(
    name="custom",
    api_key_env="LLM_API_KEY",
    base_url="",
    default_model="",
    extra_body={},
)

PROVIDERS = {"custom": CUSTOM, "deepseek": DEEPSEEK, "openrouter": OPENROUTER}


def resolve_llm_provider() -> LLMProvider:
    """Pick a provider: explicit choice, custom endpoint, then available key.

    With both keys set, DeepSeek-direct stays the measured default unless
    OpenRouter is named explicitly.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER={explicit!r} — expected one of {', '.join(sorted(PROVIDERS))}"
            )
        return PROVIDERS[explicit]
    if os.environ.get("LLM_BASE_URL"):
        return CUSTOM
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DEEPSEEK
    if os.environ.get("OPENROUTER_API_KEY"):
        return OPENROUTER
    return DEEPSEEK


def llm_base_url(provider: LLMProvider) -> str:
    return os.environ.get("LLM_BASE_URL") or provider.base_url


def llm_model(provider: LLMProvider) -> str:
    return os.environ.get("LLM_MODEL") or provider.default_model


def llm_api_key(provider: LLMProvider) -> str | None:
    """Return generic/profile auth, or a placeholder for an unauthenticated local server."""
    key = os.environ.get("LLM_API_KEY") or os.environ.get(provider.api_key_env)
    if key:
        return key
    if provider is CUSTOM:
        return "local"
    return None


def llm_config_errors(provider: LLMProvider) -> list[str]:
    """Describe missing configuration without importing the heavyweight client."""
    errors: list[str] = []
    if not llm_base_url(provider):
        errors.append("LLM_BASE_URL is required for the custom provider")
    if not llm_model(provider):
        errors.append("LLM_MODEL is required for the custom provider")
    if not llm_api_key(provider):
        errors.append(
            f"{provider.api_key_env} is not set (or use the generic LLM_API_KEY override)"
        )
    return errors
