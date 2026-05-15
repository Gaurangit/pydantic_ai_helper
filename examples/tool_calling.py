"""Tool calling — agent decides which Python function to invoke.

    python examples/tool_calling.py anthropic
    python examples/tool_calling.py gemini gemini-2.5-flash
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from pydantic_ai_helper import make_agent, tool


@tool
def get_current_utc_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def add(a: float, b: float) -> float:
    """Add two numbers. Use this instead of mental arithmetic."""
    return a + b


@tool
def word_count(text: str) -> int:
    """Count words in the supplied text."""
    return len(text.split())


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    model = sys.argv[2] if len(sys.argv) > 2 else None

    agent = make_agent(
        provider,  # type: ignore[arg-type]
        model,
        system_prompt=(
            "You are a careful assistant. When a tool can answer the user, "
            "call it rather than guessing."
        ),
        tools=[get_current_utc_time, add, word_count],
    )

    prompt = (
        "What is 17.5 + 4.25, and how many words are in the sentence "
        "'pydantic ai makes structured llm calls easy'? Also tell me the current UTC time."
    )
    result = agent.run_sync(prompt)
    print("Answer:\n", result.output)
    print("\nMessages (for debugging):")
    for m in result.all_messages():
        print(" -", type(m).__name__)


if __name__ == "__main__":
    main()
