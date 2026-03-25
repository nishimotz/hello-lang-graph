"""Shared runtime configuration for the exercises.

The workshop supports two chat backends:
- LM Studio (local OpenAI-compatible endpoint)
- OpenRouter (hosted OpenAI-compatible endpoint)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ChatRuntimeConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    app_name: str
    default_headers: dict[str, str] | None = None

def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_chat_config() -> ChatRuntimeConfig:
    provider = _env("LLM_PROVIDER", default="lmstudio").strip().lower()

    if provider == "openrouter":
        api_key = _env("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        return ChatRuntimeConfig(
            provider=provider,
            base_url=_env("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"),
            model=_env(
                "OPENROUTER_MODEL",
                "LLM_CHAT_MODEL",
                default="meta-llama/llama-3.1-8b-instruct",
            ),
            api_key=api_key,
            app_name="OpenRouter",
            default_headers=_build_openrouter_headers(),
        )

    return ChatRuntimeConfig(
        provider="lmstudio",
        base_url=_env("LM_STUDIO_BASE_URL", "OPENAI_BASE_URL", default="http://localhost:1234/v1"),
        model=_env("LM_STUDIO_CHAT_MODEL", "LLM_CHAT_MODEL", default="gpt-oss-20b"),
        api_key=_env("LM_STUDIO_API_KEY", "OPENAI_API_KEY", default="lm-studio"),
        app_name="LM Studio",
    )


def _build_openrouter_headers() -> dict[str, str] | None:
    headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME", "hello-lang-graph")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers or None


def build_chat_llm(*, temperature: float = 0.8, streaming: bool = False) -> ChatOpenAI:
    config = get_chat_config()
    kwargs: dict[str, object] = {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
        "temperature": temperature,
        "streaming": streaming,
    }
    if config.default_headers:
        kwargs["default_headers"] = config.default_headers
    return ChatOpenAI(**kwargs)
