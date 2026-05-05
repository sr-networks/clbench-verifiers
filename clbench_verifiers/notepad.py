"""
icl_notepad-style schema augmentation.

We extend the task's per-turn ``response_schema`` with an optional
``notepad_update`` field. When the model emits a non-null value, we treat it
as the new full notepad text (overwrite, not append — same convention as
clbench's ``icl_notepad`` system). The notepad is then prepended to the first
user message of every subsequent instance within the same rollout.

This is the "memory as a structured-output field" pattern, not a separate tool
call. The same reward signal that shapes the action also shapes notepad
content because both share token credit in the GRPO completion.

Why we keep the augmentation here rather than in env.py: it's a self-contained
pydantic transform plus a couple of helpers, and isolating it makes the env
file easier to follow.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from pydantic import BaseModel, Field, create_model


_NOTEPAD_FIELD_DESC = (
    "Optional: update your notepad with new observations, patterns, or "
    "learned facts that will help on future instances. Provide the FULL "
    "updated notepad content (overwrites the previous notepad). Leave as "
    "null to keep the notepad unchanged. Keep notes terse; the entire "
    "notepad is replayed at the start of every new instance."
)


def build_schema_with_notepad(task_schema: Type[BaseModel]) -> Type[BaseModel]:
    """
    Return a pydantic model identical to ``task_schema`` plus an optional
    ``notepad_update: Optional[str]`` field.

    If the schema already has ``notepad_update`` (e.g. a future task ships
    with native notepad support), return it unchanged.
    """
    if "notepad_update" in task_schema.model_fields:
        return task_schema

    fields: dict[str, Any] = {}
    for name, info in task_schema.model_fields.items():
        fields[name] = (info.annotation, info)
    fields["notepad_update"] = (
        Optional[str],
        Field(default=None, description=_NOTEPAD_FIELD_DESC),
    )

    return create_model(
        f"{task_schema.__name__}WithNotepad",
        **fields,
    )


def split_notepad_action(
    parsed: BaseModel,
    task_schema: Type[BaseModel],
) -> tuple[BaseModel, Optional[str]]:
    """
    Given a parsed instance of a notepad-augmented schema, return ``(task_action,
    notepad_update_or_None)``. ``task_action`` is a pure instance of the
    original task schema (so the task layer never sees ``notepad_update``).
    """
    notepad_update = getattr(parsed, "notepad_update", None)
    if notepad_update == "":
        notepad_update = None
    if "notepad_update" in task_schema.model_fields:
        # Native support — task wants to see the field itself.
        return parsed, notepad_update

    # Build a clean task action by copying the task fields.
    task_data = {
        name: getattr(parsed, name) for name in task_schema.model_fields.keys()
    }
    task_action = task_schema(**task_data)
    return task_action, notepad_update


def render_notepad_for_prompt(notepad: str) -> str:
    """Wrap the notepad in clear markers for the prompt. Empty → empty string."""
    if not notepad:
        return ""
    return f"=== YOUR NOTEPAD ===\n{notepad}\n==================="
