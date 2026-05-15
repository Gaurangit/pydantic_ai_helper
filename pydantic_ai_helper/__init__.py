"""pydantic_ai_helper — thin ergonomics over pydantic_ai.

Exposes a unified way to build agents across providers (OpenAI, Anthropic,
Gemini, Ollama) and run them with structured output or tool calling.
"""

from .providers import Provider, build_model, list_known_models
from .agent import make_agent, arun, run
from .structured import structured_output, astructured_output
from .tools import tool, collect_tools

__all__ = [
    "Provider",
    "build_model",
    "list_known_models",
    "make_agent",
    "run",
    "arun",
    "structured_output",
    "astructured_output",
    "tool",
    "collect_tools",
]
