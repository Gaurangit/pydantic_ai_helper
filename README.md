# pydantic_ai_helper

Thin ergonomics layer over [`pydantic_ai`](https://ai.pydantic.dev/) for the
two things you'll actually do most: **structured output** and **tool calling**,
across **OpenAI / Anthropic / Gemini / Ollama** with one API.
(I update this quite frequently, last updated April, 2026)

## Install

```bash
pip install -r requirements.txt
# or, as a package:
pip install -e .
```



**Using from a Jupyter notebook?** Add this once at the top of the notebook —
pydantic_ai's `run_sync` won't work otherwise because the kernel already has
a running event loop:

```python
import nest_asyncio; nest_asyncio.apply()
```

(Install with `pip install nest_asyncio` or `pip install -e '.[notebook]'`.)

## Providers

| provider    | auth                                    | default model           |
|-------------|-----------------------------------------|-------------------------|
| `openai`    | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | `gpt-4o-mini`    |
| `anthropic` | `ANTHROPIC_API_KEY`                     | `claude-sonnet-4-5`     |
| `gemini`    | `GEMINI_API_KEY` **or** Vertex ADC      | `gemini-2.5-flash`      |
| `ollama`    | local — `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `llama3.1` |

Ollama uses the OpenAI-compatible endpoint under the hood, so any tool or
structured-output pattern that works with OpenAI works locally too.

## 1. Structured output

```python
from pydantic import BaseModel
from pydantic_ai_helper import structured_output

class Sentiment(BaseModel):
    label: str
    confidence: float

out = structured_output(
    Sentiment,
    "Classify: 'shipping was late but support was great'",
    provider="anthropic",
)
print(out.label, out.confidence)
```

## 2. Tool calling

```python
from pydantic_ai_helper import make_agent, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

agent = make_agent("openai", tools=[add])
print(agent.run_sync("What is 17.5 + 4.25?").output)
```

Tools are plain Python functions — `pydantic_ai` reads the signature and
docstring to build the schema. The `@tool` decorator is optional; it just
marks functions so `collect_tools(module)` can pick them all up at once.

## 3. Structured output + tools together

```python
from pydantic import BaseModel
from pydantic_ai_helper import make_agent, tool

class Answer(BaseModel):
    sum: float
    reasoning: str

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

agent = make_agent("gemini", output_type=Answer, tools=[add])
print(agent.run_sync("What is 2+2? Show reasoning.").output)
```

## 4. Swap providers without touching the rest

```python
from pydantic_ai_helper import make_agent

for provider in ("openai", "anthropic", "gemini", "ollama"):
    agent = make_agent(provider, system_prompt="Reply in one word.")
    print(provider, "→", agent.run_sync("Capital of Japan?").output)
```

See `examples/` for runnable scripts (`structured_output.py`,
`tool_calling.py`, `multi_provider.py`, `ollama_local.py`).

## Layout

```
pydantic_ai_helper/
├── providers.py   # build_model(provider, model, ...) → pydantic_ai Model
├── agent.py       # make_agent(...) → Agent
├── structured.py  # structured_output / astructured_output one-shots
├── tools.py       # @tool + collect_tools
└── __init__.py
```
