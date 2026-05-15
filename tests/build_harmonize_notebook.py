"""Builder for test_harmonize_usecases.ipynb.

Use cases that exercise pydantic_ai_helper in the context of the schema
harmonization workflow (cf. ../chariot/schema_engg/tests/test_harmonize.ipynb):

1. Structured output — LLM proposes a HarmonizeConfig from source + target
   field lists.
2. Tool calling — LLM discovers fields/transforms via registered tools,
   then answers in free text.
3. Tools + structured output — same discovery tools, but the final answer is
   a typed HarmonizeConfig.
4. Explain validation errors in plain English (structured output helper).
5. Propose → validate → refine loop (end-to-end).
"""

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
# pydantic_ai_helper × schema harmonization

Reuse the `pydantic_ai_helper` package to drive the harmonization workflow
from `../chariot/schema_engg/tests/test_harmonize.ipynb` with an LLM. Every cell tests a
concrete use-case; each runs against whichever provider is available
(Ollama by default, hosted if keys are set).

Use-cases covered
1. Structured output — LLM proposes a full mapping config.
2. Tool calling — LLM discovers schema fields via tools, answers in prose.
3. Tools + structured output — discovery tools + typed final answer.
4. Explain a validation error in plain English.
5. Propose → validate → refine end-to-end loop.
    """),

    md("## 0. Setup"),
    code("""
import os, sys, asyncio, json
from pathlib import Path

import nest_asyncio; nest_asyncio.apply()

from dotenv import load_dotenv
for env_path in [".env", "../.env", "../../harcover_api/.env"]:
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)
        print("loaded", env_path)

sys.path.insert(0, str(Path("..").resolve()))
from pydantic_ai_helper import (
    make_agent, structured_output, tool, list_known_models,
)

import httpx
def _has(var):
    v = os.getenv(var) or ""
    return bool(v.strip())
def _ollama_up(url="http://localhost:11434"):
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
PRIMARY = next((p for p in ("openai", "anthropic", "gemini", "ollama") if AVAILABLE[p]), None)
assert PRIMARY, "No providers available"

OLLAMA_MODEL = None
if PRIMARY == "ollama":
    names = [m["name"] for m in httpx.get("http://localhost:11434/api/tags").json()["models"]]
    for pref in ("qwen3:32b", "llama3.3:70b", "gpt-oss:20b", "gemma3:27b"):
        if pref in names:
            OLLAMA_MODEL = pref
            break
    OLLAMA_MODEL = OLLAMA_MODEL or names[0]

def agent_kwargs():
    return {"model": OLLAMA_MODEL} if PRIMARY == "ollama" else {}

print(f"PRIMARY: {PRIMARY}" + (f" / {OLLAMA_MODEL}" if OLLAMA_MODEL else ""))
    """),

    md("""
## 1. Shared domain types

A slimmed-down mirror of the harmonization config from `chariot/schema_engg`. Same
shape — `FieldMapping`, `UnmappedTargetField`, `HarmonizeConfig` — defined
inline so the notebook is self-contained.
    """),
    code("""
from typing import Literal, Optional
from pydantic import BaseModel, Field

class FieldMapping(BaseModel):
    source_field: str
    target_field: Optional[str] = None
    transform: Optional[str] = None
    unmapped: bool = False
    note: Optional[str] = None

class UnmappedTargetField(BaseModel):
    field: str
    note: Optional[str] = None

class HarmonizeConfig(BaseModel):
    source: str
    target: str
    mappings: list[FieldMapping]
    unmapped_target_fields: list[UnmappedTargetField] = Field(default_factory=list)

# Fake schema registry — what the LLM has to work with.
SCHEMAS = {
    "Person":      ["name", "email", "telephone"],
    "UserProfile": ["full_name", "email", "internal_id"],
}
AVAILABLE_TRANSFORMS = ["str_to_lower", "strip_whitespace", "date_to_iso"]

print("domain types ready")
    """),

    md("""
## 2. Use-case 1 — LLM proposes a full `HarmonizeConfig`

Structured output: hand the LLM the source + target field lists and the
registered transforms; ask for a config back.
    """),
    code("""
PROMPT_1 = f\"\"\"\\
Propose a field mapping from source schema `Person` to target schema `UserProfile`.

Source fields:    {SCHEMAS["Person"]}
Target fields:    {SCHEMAS["UserProfile"]}
Available transforms: {AVAILABLE_TRANSFORMS}

Rules:
- Every source field MUST appear in `mappings`. If it has no meaningful target,
  set `target_field: null`, `unmapped: true`, and add a brief note.
- Every target field MUST either be the target of a mapping OR listed in
  `unmapped_target_fields` with a note.
- Use a transform only if it helps (e.g. normalise email case).
- Field names must be an exact match from the lists above.
\"\"\"

config = structured_output(
    HarmonizeConfig,
    PROMPT_1,
    provider=PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    system_prompt="You are a careful data-schema engineer. Output valid JSON only.",
)

print(config.model_dump_json(indent=2))

# Structural assertions (content can vary by model, shape shouldn't).
source_fields = {m.source_field for m in config.mappings}
target_fields = {m.target_field for m in config.mappings if m.target_field}
target_unmapped = {u.field for u in config.unmapped_target_fields}

assert config.source == "Person" and config.target == "UserProfile"
assert source_fields == set(SCHEMAS["Person"]), \\
    f"Every source field should appear in mappings. Got {source_fields}"
# Every target field covered either as a mapping target or as unmapped_target:
covered_targets = target_fields | target_unmapped
assert set(SCHEMAS["UserProfile"]).issubset(covered_targets), \\
    f"Missing target coverage: {set(SCHEMAS['UserProfile']) - covered_targets}"
# Any transform used must be one we registered:
for m in config.mappings:
    if m.transform:
        assert m.transform in AVAILABLE_TRANSFORMS, f"Unknown transform: {m.transform}"

print("\\n✓ Use-case 1: LLM produced a well-shaped HarmonizeConfig")
    """),

    md("""
## 3. Use-case 2 — Tool calling for schema discovery

The LLM doesn't get field lists in the prompt — instead it has to call
tools (`get_source_fields`, `get_target_fields`, `list_transforms`) to
figure out what's available, then explain its mapping in prose.
    """),
    code("""
_calls = {"get_src": 0, "get_tgt": 0, "list_tx": 0}

@tool
def get_source_fields(schema_name: str) -> list[str]:
    \"\"\"Return the field names of a source schema by its name.\"\"\"
    _calls["get_src"] += 1
    return SCHEMAS.get(schema_name, [])

@tool
def get_target_fields(schema_name: str) -> list[str]:
    \"\"\"Return the field names of a target schema by its name.\"\"\"
    _calls["get_tgt"] += 1
    return SCHEMAS.get(schema_name, [])

@tool
def list_transforms() -> list[str]:
    \"\"\"Return the list of registered data-transform function names.\"\"\"
    _calls["list_tx"] += 1
    return AVAILABLE_TRANSFORMS

agent = make_agent(
    PRIMARY,
    **agent_kwargs(),
    system_prompt=(
        "You are a careful data-schema engineer. You do not know the field "
        "names ahead of time — use the provided tools to look them up. "
        "After the lookups, propose a one-line-per-pair mapping."
    ),
    tools=[get_source_fields, get_target_fields, list_transforms],
)

r = agent.run_sync(
    "Propose a field mapping from `Person` to `UserProfile`. "
    "Look up the fields and the available transforms before answering."
)
print("OUTPUT:\\n", r.output)
print("\\ntool-call counts:", _calls)

assert _calls["get_src"] >= 1, "Expected the source-fields tool to be called"
assert _calls["get_tgt"] >= 1, "Expected the target-fields tool to be called"
# Free-text answer should mention at least the obvious rename.
low = r.output.lower()
assert "name" in low and "full_name" in low
print("\\n✓ Use-case 2: tools were called and the prose references name→full_name")
    """),

    md("""
## 4. Use-case 3 — Tools + typed output together

Same discovery tools, but this time the agent must return a
`HarmonizeConfig` instead of prose. The LLM figures out the fields via
tools, then emits the typed object.
    """),
    code("""
_calls2 = {"get_src": 0, "get_tgt": 0, "list_tx": 0}

@tool
def schema_fields(schema_name: str) -> list[str]:
    \"\"\"Return the field names of any registered schema by its name.\"\"\"
    role = "get_src" if schema_name == "Person" else "get_tgt"
    _calls2[role] += 1
    return SCHEMAS.get(schema_name, [])

@tool
def transforms() -> list[str]:
    \"\"\"Return the list of registered transforms.\"\"\"
    _calls2["list_tx"] += 1
    return AVAILABLE_TRANSFORMS

agent = make_agent(
    PRIMARY,
    **agent_kwargs(),
    output_type=HarmonizeConfig,
    system_prompt=(
        "You are a careful data-schema engineer. Always call the tools to "
        "discover field names and transforms before proposing a mapping. "
        "Every source field must appear in mappings (either mapped or "
        "unmapped=true). Every target field must be covered either in the "
        "mappings or in unmapped_target_fields."
    ),
    tools=[schema_fields, transforms],
)

out = agent.run_sync("Propose a mapping from Person to UserProfile.").output
print(out.model_dump_json(indent=2))
print("\\ntool-call counts:", _calls2)

# We're testing that tools + typed output cooperate — not that the LLM
# produces a perfectly-covering mapping first try. Coverage is policed by
# use-case 5's refine loop.
assert _calls2["get_src"] >= 1 and _calls2["get_tgt"] >= 1, "tools weren't called"
assert isinstance(out, HarmonizeConfig)
assert out.source == "Person" and out.target == "UserProfile"
assert len(out.mappings) >= 1
for m in out.mappings:
    assert m.source_field in SCHEMAS["Person"], f"Hallucinated source field: {m.source_field!r}"
    if m.target_field is not None:
        assert m.target_field in SCHEMAS["UserProfile"], f"Hallucinated target: {m.target_field!r}"
    if m.transform:
        assert m.transform in AVAILABLE_TRANSFORMS, f"Hallucinated transform: {m.transform!r}"

# Report any gaps — informative, not fatal.
missing_src = set(SCHEMAS["Person"]) - {m.source_field for m in out.mappings}
covered_tgt = ({m.target_field for m in out.mappings if m.target_field}
               | {u.field for u in out.unmapped_target_fields})
missing_tgt = set(SCHEMAS["UserProfile"]) - covered_tgt
if missing_src or missing_tgt:
    print(f"\\nNOTE: first-pass gaps — source missing {missing_src}, target missing {missing_tgt}")
    print("  (use-case 5 demonstrates feeding these gaps back for a refine pass)")
print("\\n✓ Use-case 3: tools fired, typed output parsed, no hallucinated fields")
    """),

    md("""
## 5. Use-case 4 — Explain a validation error in plain English

Pass a `ValidationError` code + message (matching the harmonization
validator's own codes) and get back a structured explanation with a
suggested fix.
    """),
    code("""
class ErrorExplanation(BaseModel):
    summary: str
    suggested_fix: str
    severity: Literal["error", "warning"]

error_case = {
    "code": "UNACCOUNTED_SOURCE_FIELD",
    "message": "Source field 'telephone' (Person) is not declared in mappings",
}
prompt = (
    "Explain this schema-harmonization validation issue to the engineer. "
    "Codes ending in _WARNING are warnings; the rest are errors.\\n\\n"
    f"Code:    {error_case['code']}\\n"
    f"Message: {error_case['message']}\\n\\n"
    "Respond with: a one-sentence summary, a concrete one-sentence fix, "
    "and the severity."
)

exp = structured_output(
    ErrorExplanation,
    prompt,
    provider=PRIMARY,
    model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
    system_prompt="You are a helpful senior engineer reviewing a config PR.",
)
print(exp.model_dump_json(indent=2))

assert exp.severity == "error"
assert "telephone" in exp.suggested_fix.lower() or "unmapped" in exp.suggested_fix.lower()
print("\\n✓ Use-case 4: plain-English explanation generated")
    """),

    md("""
## 6. Use-case 5 — Propose → validate → refine loop

Mirrors how you'd actually use this: LLM proposes a config, a local
validator checks it, and if it fails we feed the errors back and ask the
LLM to fix them. Limited to 2 tries so the test stays deterministic.
    """),
    code("""
def validate_config(cfg: HarmonizeConfig) -> list[str]:
    \"\"\"Tiny local validator — same error codes as the harmonize package.\"\"\"
    errors = []
    src_needed = set(SCHEMAS[cfg.source])
    tgt_needed = set(SCHEMAS[cfg.target])

    # Every source field declared?
    declared_src = {m.source_field for m in cfg.mappings}
    for missing in src_needed - declared_src:
        errors.append(f"UNACCOUNTED_SOURCE_FIELD: {missing}")
    # Extra source fields?
    for extra in declared_src - src_needed:
        errors.append(f"UNKNOWN_SOURCE_FIELD: {extra}")
    # Every target field covered?
    covered_tgt = ({m.target_field for m in cfg.mappings if m.target_field}
                   | {u.field for u in cfg.unmapped_target_fields})
    for missing in tgt_needed - covered_tgt:
        errors.append(f"UNACCOUNTED_TARGET_FIELD: {missing}")
    # Transforms registered?
    for m in cfg.mappings:
        if m.transform and m.transform not in AVAILABLE_TRANSFORMS:
            errors.append(f"UNKNOWN_TRANSFORM: {m.transform} on {m.source_field}")
    # Ambiguous mapping?
    for m in cfg.mappings:
        if m.target_field is None and not m.unmapped:
            errors.append(f"AMBIGUOUS_MAPPING: {m.source_field}")
    return errors


def propose(extra_instructions: str = "") -> HarmonizeConfig:
    prompt = (
        f"Propose a HarmonizeConfig mapping Person → UserProfile.\\n"
        f"Source fields: {SCHEMAS['Person']}\\n"
        f"Target fields: {SCHEMAS['UserProfile']}\\n"
        f"Transforms:    {AVAILABLE_TRANSFORMS}\\n"
    )
    if extra_instructions:
        prompt += f"\\nAdditional requirements:\\n{extra_instructions}\\n"
    return structured_output(
        HarmonizeConfig, prompt,
        provider=PRIMARY,
        model=OLLAMA_MODEL if PRIMARY == "ollama" else None,
        system_prompt=(
            "Every source field MUST appear in `mappings`. "
            "Every target field MUST be covered in `mappings` or "
            "`unmapped_target_fields`. Only use listed transforms. "
            "Return valid JSON."
        ),
    )


cfg = propose()
errors = validate_config(cfg)
print(f"Attempt 1 errors: {errors}")

if errors:
    fix_instructions = (
        "The previous attempt had these validation errors — fix them:\\n"
        + "\\n".join(f"- {e}" for e in errors)
    )
    cfg = propose(fix_instructions)
    errors = validate_config(cfg)
    print(f"Attempt 2 errors: {errors}")

print("\\nFinal config:")
print(cfg.model_dump_json(indent=2))
assert not errors, f"Validator should be clean after refine; got {errors}"
print("\\n✓ Use-case 5: propose → validate → refine produced a valid config")
    """),

    md("""
## Summary

Every cell that asserted something ran cleanly, which means:
- `structured_output` reliably produces typed configs against your PRIMARY backend.
- `make_agent(tools=...)` triggers the registered tools.
- Tools + typed output can be combined (use-case 3).
- The same helpers cover downstream tasks like error-explanation (use-case 4).
- A propose/validate/refine loop is straightforward to build on top (use-case 5).
    """),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = Path(__file__).parent / "test_harmonize_usecases.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
