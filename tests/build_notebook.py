"""Builder for test_helpers.ipynb — keeps cell authoring readable in .py form."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src.strip("\n"))


CELLS = [
    md("""
# pydantic_ai_helper — integration tests

End-to-end checks of the helper package. Each provider section auto-skips if
its credentials aren't set. Ollama runs locally and is the default backend.

Sections
1. Environment probe
2. Provider factory (`build_model`)
3. Simple text generation (`make_agent`)
4. Structured output (`structured_output`, `output_type=`)
5. Tool calling (`@tool`, `make_agent(tools=...)`)
6. Structured + tools together
7. Async variant
8. `collect_tools` + cross-provider dispatch
    """),

    md("## 0. Setup"),
    code("""
import os, sys, asyncio, json
from pathlib import Path

# nest_asyncio lets pydantic_ai's `run_sync` work inside Jupyter's running loop.
import nest_asyncio
nest_asyncio.apply()

# Load .env so shell-level keys flow through (harcover_api/.env is a fallback).
from dotenv import load_dotenv
for env_path in [".env", "../.env", "../../harcover_api/.env"]:
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)
        print("loaded", env_path)

sys.path.insert(0, str(Path("..").resolve()))

import pydantic_ai
from pydantic_ai_helper import (
    build_model, make_agent, structured_output, astructured_output,
    tool, collect_tools, list_known_models,
)

print("pydantic_ai:", pydantic_ai.__version__)
print("defaults:", list_known_models())
    """),

    md("## 1. Environment probe — which providers can we actually test?"),
    code("""
import httpx

def _has(var: str) -> bool:
    val = os.getenv(var) or ""
    return bool(val.strip())

def _ollama_up(url: str = "http://localhost:11434") -> bool:
    try:
        return httpx.get(f"{url}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False

AVAILABLE = {
    "openai":    _has("OPENAI_API_KEY"),
    "anthropic": _has("ANTHROPIC_API_KEY"),
    "gemini":    _has("GEMINI_API_KEY") or _has("GOOGLE_API_KEY") or _has("GOOGLE_APPLICATION_CREDENTIALS"),
    "ollama":    _ollama_up(),
}
for k, v in AVAILABLE.items():
    print(f"  {k:10s} {'✓' if v else '—'}")

# Pick the primary backend used for the core correctness tests.
# Prefer cheap-and-fast hosted; fall back to Ollama.
PRIMARY = next((p for p in ("openai", "anthropic", "gemini", "ollama") if AVAILABLE[p]), None)
assert PRIMARY, "No providers available — set at least one API key or start Ollama."
print("\\nPRIMARY backend:", PRIMARY)

# For Ollama we need a real locally-pulled model. Try a sensible default.
OLLAMA_MODEL = None
if AVAILABLE["ollama"]:
    tags = httpx.get("http://localhost:11434/api/tags").json()["models"]
    names = [m["name"] for m in tags]
    # Prefer small/fast chat-capable model.
    for pref in ("qwen3:32b", "llama3.3:70b", "gpt-oss:20b", "gemma3:27b"):
        if pref in names:
            OLLAMA_MODEL = pref
            break
    OLLAMA_MODEL = OLLAMA_MODEL or names[0]
    print("OLLAMA_MODEL:", OLLAMA_MODEL)
    """),

    md("## 2. Provider factory — `build_model`"),
    code("""
# Build a model for every available provider and sanity-check the class.
built = {}
for provider, ok in AVAILABLE.items():
    if not ok:
        continue
    try:
        m = build_model(provider, OLLAMA_MODEL if provider == "ollama" else None)
        built[provider] = m
        print(f"  {provider:10s} → {type(m).__name__}  model={getattr(m, 'model_name', '?')}")
    except Exception as e:
        print(f"  {provider:10s} FAILED: {type(e).__name__}: {e}")

assert built, "No models could be built"
    """),

    md("## 3. Simple text generation"),
    code("""
kwargs = {"model": OLLAMA_MODEL} if PRIMARY == "ollama" else {}
agent = make_agent(
    PRIMARY,
    system_prompt="Answer with one lowercase word, no punctuation.",
    **kwargs,
)
result = agent.run_sync("Capital of Japan?")
print("output:", repr(result.output))
assert "tokyo" in result.output.lower(), f"Expected tokyo, got {result.output!r}"
print("✓ simple generation works")
    """),

    md("""
## 4. Structured output

Pass a pydantic schema; the helper coerces the model's reply into a typed instance.
    """),
    code("""
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

out = structured_output(
    Sentiment,
    "Classify the sentiment: 'shipping was late but support was friendly'",
    provider=PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    system_prompt="You are a sentiment classifier. Keep rationale to one sentence.",
)
print(out.model_dump_json(indent=2))
assert isinstance(out, Sentiment)
assert out.label in {"positive", "negative", "neutral", "mixed"}
assert 0.0 <= out.confidence <= 1.0
print("✓ structured output works")
    """),

    code("""
# Nested schema — make sure the helper handles lists-of-models.
class LineItem(BaseModel):
    name: str
    qty: int
    unit_price: float

class Invoice(BaseModel):
    vendor: str
    currency: str
    line_items: list[LineItem]
    total: float

text = (
    "Invoice from Acme Corp. 2 widgets at $4.99 each, 1 gizbox at $189.97. "
    "Currency USD. Total $199.95."
)

agent = make_agent(
    PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    output_type=Invoice,
    system_prompt="Extract the invoice into the schema. Use the exact numbers in the text.",
)
inv = agent.run_sync(text).output
print(inv.model_dump_json(indent=2))
assert inv.vendor.lower().startswith("acme")
assert len(inv.line_items) >= 2
print("✓ nested schema works")
    """),

    md("## 5. Tool calling"),
    code("""
from datetime import datetime, timezone

_calls = {"add": 0, "now": 0, "wc": 0}

@tool
def add(a: float, b: float) -> float:
    \"\"\"Add two numbers. Prefer this over mental arithmetic.\"\"\"
    _calls["add"] += 1
    return a + b

@tool
def current_utc() -> str:
    \"\"\"Return the current UTC time in ISO 8601.\"\"\"
    _calls["now"] += 1
    return datetime.now(timezone.utc).isoformat()

@tool
def word_count(text: str) -> int:
    \"\"\"Count words in the given text.\"\"\"
    _calls["wc"] += 1
    return len(text.split())

agent = make_agent(
    PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    system_prompt=(
        "When a tool applies, call it rather than guessing. "
        "After tool calls, give a concise final answer."
    ),
    tools=[add, current_utc, word_count],
)

prompt = (
    "Compute 17.5 + 4.25 using the add tool, "
    "count the words in the sentence 'pydantic ai makes llm calls easy', "
    "and give me the current UTC time."
)
result = agent.run_sync(prompt)
print("OUTPUT:\\n", result.output)
print("\\ntool call counts:", _calls)

assert _calls["add"] >= 1, "add tool was never called"
assert _calls["wc"] >= 1,  "word_count tool was never called"
assert "21.75" in result.output or "21,75" in result.output
print("✓ tool calling works")
    """),

    md("## 6. Structured output AND tools together"),
    code("""
class MathAnswer(BaseModel):
    result: float
    steps: list[str]

_calls2 = {"mul": 0, "add": 0}

@tool
def mul(a: float, b: float) -> float:
    \"\"\"Multiply two numbers.\"\"\"
    _calls2["mul"] += 1
    return a * b

@tool
def add2(a: float, b: float) -> float:
    \"\"\"Add two numbers.\"\"\"
    _calls2["add"] += 1
    return a + b

agent = make_agent(
    PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    output_type=MathAnswer,
    tools=[mul, add2],
    system_prompt="Use the tools for any arithmetic. Record each tool call as a step.",
)
out = agent.run_sync("Compute (3 * 4) + 5 using the tools.").output
print(out.model_dump_json(indent=2))
assert abs(out.result - 17) < 1e-6
assert _calls2["mul"] >= 1 and _calls2["add"] >= 1
print("✓ structured + tools works")
    """),

    md("## 7. Async variant"),
    code("""
async def _run():
    out = await astructured_output(
        Sentiment,
        "Classify: 'absolutely delighted with the new update'",
        provider=PRIMARY,
        model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
        system_prompt="You are a sentiment classifier.",
    )
    return out

out = asyncio.get_event_loop().run_until_complete(_run())
print(out)
assert out.label in {"positive", "neutral", "mixed"}
print("✓ async works")
    """),

    md("## 8. `collect_tools` + cross-provider dispatch"),
    code("""
# `collect_tools` should pick up only @tool-tagged callables.
def not_a_tool(x): return x

@tool
def tagged_a(x: int) -> int:
    \"\"\"a\"\"\"
    return x

@tool
def tagged_b(y: int) -> int:
    \"\"\"b\"\"\"
    return y

tools = collect_tools([not_a_tool, tagged_a, tagged_b])
assert [t.__name__ for t in tools] == ["tagged_a", "tagged_b"]
print("✓ collect_tools filters correctly")
    """),
    code("""
# Same prompt, each available provider. Useful smoke test before a bigger run.
for provider, ok in AVAILABLE.items():
    if not ok:
        print(f"  {provider:10s} skipped (no creds)")
        continue
    try:
        a = make_agent(
            provider,
            model=OLLAMA_MODEL if provider == "ollama" else None,
            system_prompt="Reply in exactly one lowercase word.",
        )
        r = a.run_sync("Capital of France?")
        print(f"  {provider:10s} → {r.output.strip()!r}")
    except Exception as e:
        print(f"  {provider:10s} ERROR {type(e).__name__}: {e}")
    """),

    md("""
## Summary

If every cell above ran without an `AssertionError`, the helper package is wired
up correctly for the providers your environment has credentials for. To exercise
the providers marked `—` in the probe, set the matching env var and re-run.
    """),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = Path(__file__).parent / "test_helpers.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
