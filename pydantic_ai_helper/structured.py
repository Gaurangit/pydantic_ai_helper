"""One-shot structured output — hand a prompt + schema, get a typed object back."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .agent import make_agent
from .providers import Provider

T = TypeVar("T", bound=BaseModel)


def structured_output(
    schema: type[T],
    prompt: str,
    *,
    provider: Provider = "openai",
    model: str | None = None,
    system_prompt: str | None = None,
    retries: int = 2,
) -> T:
    """Run a single prompt and return an instance of `schema`.

    Convenience wrapper — for repeated calls build one agent with `make_agent`
    and reuse it instead.
    """
    agent = make_agent(
        provider,
        model,
        system_prompt=system_prompt,
        output_type=schema,
        retries=retries,
    )
    result = agent.run_sync(prompt)
    return result.output


async def astructured_output(
    schema: type[T],
    prompt: str,
    *,
    provider: Provider = "openai",
    model: str | None = None,
    system_prompt: str | None = None,
    retries: int = 2,
) -> T:
    """Async variant of `structured_output`."""
    agent = make_agent(
        provider,
        model,
        system_prompt=system_prompt,
        output_type=schema,
        retries=retries,
    )
    result = await agent.run(prompt)
    return result.output
