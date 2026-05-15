"""Provider factory — build a pydantic_ai Model for any supported backend.

The goal is a single entry point `build_model("openai", "gpt-4o-mini")` that
returns a Model instance ready to hand to `pydantic_ai.Agent`, regardless of
which backend is selected. Ollama is wired through the OpenAI-compatible
endpoint at `http://localhost:11434/v1`.

Environment variables (read lazily when a model is built):
- OPENAI_API_KEY, OPENAI_BASE_URL
- ANTHROPIC_API_KEY
- GEMINI_API_KEY (AI Studio) or GOOGLE_APPLICATION_CREDENTIALS (Vertex)
- OLLAMA_BASE_URL (default: http://localhost:11434/v1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "anthropic", "gemini", "ollama"]

# Sensible defaults per provider. Override by passing `model` to build_model.
DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
}


@dataclass(frozen=True)
class ModelSpec:
    """Resolved provider + model name pair, used for logging/debugging."""

    provider: Provider
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _require_env(key: str, provider: Provider) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"{provider} provider requires env var {key}. "
            f"Set it in your environment or .env file."
        )
    return value


def build_model(
    provider: Provider,
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Return a pydantic_ai Model for the given provider.

    Args:
        provider: one of "openai", "anthropic", "gemini", "ollama".
        model: model name (e.g. "gpt-4o-mini"). Falls back to DEFAULT_MODELS.
        api_key: overrides the provider's env-based key.
        base_url: overrides the provider's default endpoint (useful for
            OpenAI-compatible proxies or self-hosted Ollama).
    """
    model_name = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        key = api_key or _require_env("OPENAI_API_KEY", "openai")
        url = base_url or os.getenv("OPENAI_BASE_URL")
        prov = OpenAIProvider(api_key=key, base_url=url) if url else OpenAIProvider(api_key=key)
        return OpenAIModel(model_name, provider=prov)

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        key = api_key or _require_env("ANTHROPIC_API_KEY", "anthropic")
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=key))

    if provider == "gemini":
        # Prefer the newer GoogleModel (google-genai) for both AI Studio and Vertex.
        try:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider
        except ImportError as e:
            raise ImportError(
                "Install the google extra: pip install 'pydantic-ai-slim[google]'"
            ) from e

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            prov = GoogleProvider(api_key=key)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # Vertex AI via ADC
            prov = GoogleProvider(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            raise RuntimeError(
                "Gemini provider needs GEMINI_API_KEY (AI Studio) or "
                "GOOGLE_APPLICATION_CREDENTIALS (Vertex)."
            )
        return GoogleModel(model_name, provider=prov)

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API at /v1 — reuse OpenAIModel.
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        # Ollama ignores the key but OpenAI client requires a non-empty value.
        prov = OpenAIProvider(api_key=api_key or "ollama", base_url=url)
        return OpenAIModel(model_name, provider=prov)

    raise ValueError(f"Unknown provider: {provider!r}")


def list_known_models() -> dict[Provider, str]:
    """Return the default model mapping — handy for CLI/demo scripts."""
    return dict(DEFAULT_MODELS)
