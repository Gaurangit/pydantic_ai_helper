"""Structured output + tools against a local Ollama model.

Prerequisite: `ollama pull llama3.1` (or whatever model you pass).

    python examples/ollama_local.py
    python examples/ollama_local.py qwen2.5
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from pydantic_ai_helper import make_agent, tool


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    line_items: list[str]


@tool
def normalize_currency(code: str) -> str:
    """Upper-case and trim an ISO 4217 currency code."""
    return code.strip().upper()


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1"
    agent = make_agent(
        "ollama",
        model,
        system_prompt="Extract structured invoice data from the user's text.",
        output_type=Invoice,
        tools=[normalize_currency],
    )
    text = (
        "Invoice from Acme Corp — 2x widgets and 1x gizmo, total 199.95 usd."
    )
    result = agent.run_sync(text)
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
