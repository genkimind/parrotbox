"""Which hosted LLM endpoint the agent talks to, resolved from the environment.

Two built-in profiles, both OpenAI-compatible and chosen by which API key is
present (or explicitly via `LLM_PROVIDER`):

  * `deepseek`   — the default: api.deepseek.com direct. Lowest measured TTFT
    from this network (~1.05 s p50, see ARCHITECTURE.md latency budget).
  * `openrouter` — openrouter.ai gateway: one key routes to any of their
    models; useful as the low-TTFT fallback path or to A/B providers without
    touching code. Model slugs are provider-prefixed
    (`deepseek/deepseek-v4-flash`).

`LLM_MODEL` overrides the default model either way; `LLM_BASE_URL` overrides
the endpoint for any custom OpenAI-compatible host.
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

PROVIDERS = {"deepseek": DEEPSEEK, "openrouter": OPENROUTER}


def resolve_llm_provider() -> LLMProvider:
    """Pick a provider: `LLM_PROVIDER` wins, else whichever key is set.

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
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DEEPSEEK
    if os.environ.get("OPENROUTER_API_KEY"):
        return OPENROUTER
    return DEEPSEEK


def llm_base_url(provider: LLMProvider) -> str:
    return os.environ.get("LLM_BASE_URL") or provider.base_url


def llm_model(provider: LLMProvider) -> str:
    return os.environ.get("LLM_MODEL") or provider.default_model
