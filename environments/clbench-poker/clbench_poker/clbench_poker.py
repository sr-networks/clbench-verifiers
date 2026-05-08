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

import logging
from pathlib import Path
from typing import Any, Optional

# Re-exported from the glue package, which is a runtime dep of this env
# (declared in pyproject.toml).
from clbench_verifiers import build_clbench_env

logger = logging.getLogger(__name__)


def _bundled_questions_path(filename: str) -> Optional[Path]:
    """Return path to a questions file bundled inside this env package, or None."""
    candidate = Path(__file__).resolve().parent / "data" / filename
    return candidate if candidate.is_file() else None


def _maybe_setup_task(task_name: str) -> None:
    """If the task class has ``setup()``, call it. No-op when it doesn't."""
    try:
        from src.registry import get_task_class  # type: ignore
        task_cls = get_task_class(task_name)
    except Exception as exc:  # pragma: no cover - clbench import failure
        logger.warning("could not resolve task '%s' for setup: %s", task_name, exc)
        return
    if not getattr(task_cls, "has_setup", False):
        return
    try:
        task_cls.setup(force=False)
    except Exception as exc:  # pragma: no cover - network/HF errors
        logger.warning("task '%s' setup() failed: %s", task_name, exc)


# Default task — kept for backwards-compat with existing poker configs.
# Override with ``task_name`` in [[env]].args to point at any other CLBench
# task that's registered in the same package (e.g. ``database_exploration``).
_DEFAULT_TASK_NAME = "exploitable_poker"

# Per-task default ``task_kwargs``. If ``task_name`` matches a key here we
# seed those defaults; otherwise no defaults are applied and the caller is
# expected to provide everything via ``task_kwargs`` in the training TOML.
_TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "exploitable_poker": {
        "num_instances": 5,
        "opponent_policy": "calling_station",
        "seed": 0,
    },
    "database_exploration": {
        "num_instances": 2,
        "num_questions": 2,
        "seed": 0,
    },
}


def load_environment(
    *,
    task_name: str = _DEFAULT_TASK_NAME,
    task_kwargs: Optional[dict[str, Any]] = None,
    max_instances_per_rollout: int = 1,
    max_turns: int = 16,
    max_input_tokens_per_rollout: int = 8000,
    parse_failure_penalty: float = -1.0,
    format_shaping_weight: float = 0.1,
    query_efficiency_weight: float = 0.3,
    end_on_parse_failure: bool = True,
    schema_hint_in_system: bool = True,
    use_notepad: bool = False,
    notepad_max_chars: int = 4000,
    clear_history_between_instances: bool = False,
    final_instance_reward_weight: float = 0.0,
    mean_instance_reward_weight: float = 1.0,
    enable_guided_json: bool = True,
    dataset_size: int = 256,
    **_unused: Any,
):
    """
    Build a verifiers ``Environment`` wrapping a CLBench task.

    Despite the package name (``clbench-poker``, kept for back-compat), the
    same env package now drives any CLBench task: pass ``task_name`` in
    [[env]].args. Default is ``exploitable_poker``.

    Defaults are tuned for cold-start safety on an untrained base model
    (low ``max_turns``, hard ``max_input_tokens_per_rollout`` cap, and
    ``end_on_parse_failure=True``). Once the policy emits valid JSON
    consistently, raise the caps via ``[[env]].args`` in the training TOML.

    All parameters are optional and forwarded to ``build_clbench_env``. Keyword
    args we don't recognize are accepted-and-ignored (``**_unused``) so future
    additions to the training TOML don't break older env images.
    """
    merged_task_kwargs = dict(_TASK_DEFAULTS.get(task_name, {}))
    if task_kwargs:
        merged_task_kwargs.update(task_kwargs)

    # Some CLBench tasks have a ``setup()`` classmethod that downloads
    # large artifacts from HuggingFace. Run it before the env-server
    # starts handing out rollouts so the first task instantiation
    # doesn't block on a 400MB download. (database_exploration ships
    # ``products.db`` and ``products_drifted.db`` this way.)
    _maybe_setup_task(task_name)

    # database_exploration's questions.json must be generated from the
    # downloaded DB; CLBench's ``setup()`` only downloads the .db files.
    # We bundle a pre-generated questions.json with the env wheel so
    # Prime's env-server doesn't need to run the generator script. Point
    # ``questions_path`` at the bundled file unless the caller already
    # provided one.
    if task_name == "database_exploration" and "questions_path" not in merged_task_kwargs:
        bundled = _bundled_questions_path("database_exploration_questions.json")
        if bundled is not None:
            merged_task_kwargs["questions_path"] = str(bundled)

    return build_clbench_env(
        task_name=task_name,
        task_kwargs=merged_task_kwargs,
        max_instances_per_rollout=max_instances_per_rollout,
        max_turns=max_turns,
        max_input_tokens_per_rollout=max_input_tokens_per_rollout,
        parse_failure_penalty=parse_failure_penalty,
        format_shaping_weight=format_shaping_weight,
        query_efficiency_weight=query_efficiency_weight,
        end_on_parse_failure=end_on_parse_failure,
        schema_hint_in_system=schema_hint_in_system,
        use_notepad=use_notepad,
        notepad_max_chars=notepad_max_chars,
        clear_history_between_instances=clear_history_between_instances,
        final_instance_reward_weight=final_instance_reward_weight,
        mean_instance_reward_weight=mean_instance_reward_weight,
        enable_guided_json=enable_guided_json,
        dataset_size=dataset_size,
    )
