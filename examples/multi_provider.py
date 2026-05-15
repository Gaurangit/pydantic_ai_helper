"""Run the same structured-output task against every configured provider.

Handy for benchmarking or for sanity-checking that your credentials work.

    python examples/multi_provider.py
"""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_ai_helper import list_known_models, structured_output


class Sentiment(BaseModel):
    label: str  # "positive" | "negative" | "neutral"
    confidence: float
    rationale: str


PROMPT = "Classify: 'The new release is buggy but the team is responsive.'"


def main() -> None:
    for provider in list_known_models():
        print(f"\n=== {provider} ===")
        try:
            out = structured_output(
                Sentiment,
                PROMPT,
                provider=provider,  # type: ignore[arg-type]
                system_prompt="You are a sentiment classifier. Return concise rationale.",
            )
            print(out.model_dump_json(indent=2))
        except Exception as e:  # keep going so the other providers still run
            print(f"  skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
