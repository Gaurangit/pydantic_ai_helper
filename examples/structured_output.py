"""Structured output — pass a pydantic model, get a typed instance back.

    python examples/structured_output.py openai
    python examples/structured_output.py ollama llama3.1
"""

from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai_helper import structured_output


class BookSummary(BaseModel):
    title: str
    author: str
    year: int = Field(description="Year of first publication.")
    genre: Literal["fiction", "non-fiction", "poetry", "other"]
    one_line_pitch: str


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    model = sys.argv[2] if len(sys.argv) > 2 else None

    result = structured_output(
        BookSummary,
        prompt="Summarize 'The Left Hand of Darkness' by Ursula K. Le Guin.",
        provider=provider,  # type: ignore[arg-type]
        model=model,
        system_prompt="You are a precise literary reference assistant.",
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
