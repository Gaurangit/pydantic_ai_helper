"""Tool-calling helpers.

pydantic_ai already introspects Python callables (name, docstring, type hints)
to build the tool schema — there's nothing to reinvent. What this module adds:

- `@tool` decorator that simply tags a function so `collect_tools` can pick it
  up when scanning a module or list. Useful when you want to keep tool
  implementations in one file and hand them all to an agent in one line.
- `collect_tools(module_or_iterable)` returns the tagged callables.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any, Callable, Iterable

_TOOL_MARKER = "__pydantic_ai_helper_tool__"


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Tag `func` as a tool so `collect_tools` discovers it.

    The function is returned unchanged — pydantic_ai reads its signature and
    docstring directly when the Agent is constructed.
    """
    setattr(func, _TOOL_MARKER, True)
    return func


def collect_tools(source: ModuleType | Iterable[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """Gather tagged tools from a module or an iterable of callables.

    Passing a module: returns every top-level function in it that was tagged
    with `@tool`. Passing an iterable: filters to the tagged entries.
    """
    if isinstance(source, ModuleType):
        members = (obj for _, obj in inspect.getmembers(source, inspect.isfunction))
    else:
        members = source
    return [f for f in members if getattr(f, _TOOL_MARKER, False)]
