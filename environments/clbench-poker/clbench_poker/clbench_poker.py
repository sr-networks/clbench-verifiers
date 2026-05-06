"""
``load_environment`` factory required by Prime's hosted training service.

Prime's env-server invokes this with the ``args`` dict from the [[env]] block
of the training TOML, expanded as kwargs. We forward everything to
``clbench_verifiers.build_clbench_env`` so the env exposes the same knobs in
hosted training as it does in local/Colab training.

Examples
--------

In ``rl.toml``:

    [[env]]
    id = "sr-networks/clbench-poker"
    args = { max_instances_per_rollout = 1 }

Notepad-mode training:

    [[env]]
    id = "sr-networks/clbench-poker"
    args = {
        use_notepad = true,
        max_instances_per_rollout = 4,
        task_kwargs = { num_instances = 4, opponent_policy = "calling_station" },
    }
"""

from __future__ import annotations

from typing import Any, Optional

# Re-exported from the glue package, which is a runtime dep of this env
# (declared in pyproject.toml).
from clbench_verifiers import build_clbench_env


# ``task_name`` is fixed for this env package — Prime's env id namespacing
# already encodes the task. Other CLBench tasks would get their own env
# packages (e.g. ``clbench-database-exploration``) when we wrap them.
_TASK_NAME = "exploitable_poker"

# Conservative defaults. Override via [[env]].args in the training TOML.
_DEFAULT_TASK_KWARGS: dict[str, Any] = {
    "num_instances": 5,
    "opponent_policy": "calling_station",
    "seed": 0,
}


def load_environment(
    *,
    task_kwargs: Optional[dict[str, Any]] = None,
    max_instances_per_rollout: int = 1,
    max_turns: int = 16,
    max_input_tokens_per_rollout: int = 8000,
    parse_failure_penalty: float = -1.0,
    format_shaping_weight: float = 0.1,
    end_on_parse_failure: bool = True,
    schema_hint_in_system: bool = True,
    use_notepad: bool = False,
    notepad_max_chars: int = 4000,
    enable_guided_json: bool = True,
    **_unused: Any,
):
    """
    Build a verifiers ``Environment`` for ``exploitable_poker``.

    Defaults are tuned for cold-start safety on an untrained base model
    (low ``max_turns``, hard ``max_input_tokens_per_rollout`` cap, and
    ``end_on_parse_failure=True``). Once the policy emits valid JSON
    consistently, raise the caps via ``[[env]].args`` in the training TOML.

    All parameters are optional and forwarded to ``build_clbench_env``. Keyword
    args we don't recognize are accepted-and-ignored (``**_unused``) so future
    additions to the training TOML don't break older env images.
    """
    merged_task_kwargs = dict(_DEFAULT_TASK_KWARGS)
    if task_kwargs:
        merged_task_kwargs.update(task_kwargs)

    return build_clbench_env(
        task_name=_TASK_NAME,
        task_kwargs=merged_task_kwargs,
        max_instances_per_rollout=max_instances_per_rollout,
        max_turns=max_turns,
        max_input_tokens_per_rollout=max_input_tokens_per_rollout,
        parse_failure_penalty=parse_failure_penalty,
        format_shaping_weight=format_shaping_weight,
        end_on_parse_failure=end_on_parse_failure,
        schema_hint_in_system=schema_hint_in_system,
        use_notepad=use_notepad,
        notepad_max_chars=notepad_max_chars,
        enable_guided_json=enable_guided_json,
    )
