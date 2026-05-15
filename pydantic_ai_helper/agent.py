"""Agent factory — a thin wrapper around pydantic_ai.Agent."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic_ai import Agent

from .providers import Provider, build_model


def make_agent(
    provider: Provider,
    model: str | None = None,
    *,
    system_prompt: str | Sequence[str] | None = None,
    output_type: Any = str,
    tools: Iterable = (),
    retries: int = 2,
    deps_type: Any = None,
    **model_kwargs: Any,
) -> Agent:
    """Build a configured pydantic_ai Agent.

    Args:
        provider: backend ("openai", "anthropic", "gemini", "ollama").
        model: model name; uses provider default if omitted.
        system_prompt: string or list of strings concatenated by pydantic_ai.
        output_type: type used for structured output (defaults to `str`).
            Pass a pydantic BaseModel (or dataclass) to force structured output.
        tools: callables — plain functions work; use `@tool` decorator or
            `Agent.tool` for richer metadata.
        retries: number of reask retries on validation failure.
        deps_type: optional `deps_type` for dependency injection into tools.
        model_kwargs: forwarded to `build_model` (api_key, base_url).
    """
    llm = build_model(provider, model, **model_kwargs)
    kwargs: dict[str, Any] = {
        "model": llm,
        "output_type": output_type,
        "tools": list(tools),
        "retries": retries,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if deps_type is not None:
        kwargs["deps_type"] = deps_type
    return Agent(**kwargs)


async def arun(agent: Agent, prompt: str, **kwargs: Any):
    """Async run — returns the full AgentRunResult."""
    return await agent.run(prompt, **kwargs)


def run(agent: Agent, prompt: str, **kwargs: Any):
    """Sync run — returns the full AgentRunResult."""
    return agent.run_sync(prompt, **kwargs)
