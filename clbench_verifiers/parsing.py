"""
Parse a model's raw text completion into a CLBench action (a pydantic model
matching ``Query.response_schema``).

Strategy: tolerant JSON extraction (the policy emits JSON wrapped in optional
markdown fences or chain-of-thought text). Parse failures return None and the
rubric penalizes them — this gives GRPO a learning signal toward valid JSON
without needing constrained decoding.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

# JSON object detector: greedy from first '{' to matching '}'. The model's
# completion almost always contains exactly one JSON object; for longer texts
# we fall back to bracket matching.
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[str]:
    """Best-effort JSON object extraction from raw model text."""
    if not text:
        return None
    m = _FENCED.search(text)
    if m:
        return m.group(1)

    # Bracket-balanced scan: handles nested objects, ignores braces in strings.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Fallback: greedy regex (will probably fail validation but try anyway).
    m = _FIRST_OBJECT.search(text)
    return m.group(0) if m else None


def parse_action(
    text: str,
    schema: type[BaseModel],
) -> tuple[Optional[BaseModel], Optional[str]]:
    """
    Parse a model completion into a pydantic action.

    Returns
    -------
    (action, error)
        ``action`` is the validated pydantic instance on success, ``None`` on failure.
        ``error`` is a short reason string on failure, ``None`` on success.
    """
    raw = _extract_json(text)
    if raw is None:
        return None, "no_json_found"
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json_decode:{e.msg}"
    if not isinstance(data, dict):
        return None, "json_not_object"
    try:
        return schema.model_validate(data), None
    except ValidationError as e:
        # First validation error is enough signal; full error is verbose.
        first = e.errors()[0] if e.errors() else {"msg": "validation_failed"}
        loc = ".".join(str(x) for x in first.get("loc", []))
        return None, f"schema:{loc}:{first.get('msg', 'invalid')}"


def format_schema_hint(schema: type[BaseModel]) -> str:
    """
    Render a compact JSON-schema hint for the prompt.

    We use ``schema.model_json_schema()`` which already produces a well-formed
    JSON schema; we strip the noisy ``$defs`` indirection if any to keep tokens
    down. Caller is expected to wrap this in a fence in the system prompt.
    """
    js = schema.model_json_schema()
    # Inline simple $ref → $defs substitutions (one level deep) to compact output.
    defs = js.pop("$defs", None) or js.pop("definitions", None)
    if defs:
        js = _inline_refs(js, defs)
    return json.dumps(js, indent=2, default=str)


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"].rsplit("/", 1)[-1]
            target = defs.get(ref)
            if target is not None:
                return _inline_refs(target, defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node
