"""
CLBench task → verifiers MultiTurnEnv adapter.

The verifiers Environment loop drives the conversation:

    1. setup_state(state)            → initialize per-rollout state, including
                                        the CLBench task instance and a fresh
                                        first user message built from the
                                        first Query.
    2. model speaks                   → assistant message appended.
    3. env_response(messages, state) → we parse the assistant message into a
                                        pydantic action via parse_action(),
                                        feed it into task.step(), and return
                                        a user message that contains the next
                                        Query.prompt plus any feedback.
    4. is_completed()                 → True when the task signals done OR we
                                        have hit max_instances per rollout.

Reward is computed by the rubric (see rubric.py) reading the per-rollout
state's accumulated InstanceOutcomes.

CLBench's package layout exposes ``src.*`` rather than a normal package name
(see its pyproject), so the imports below use that. We re-import lazily inside
the adapter to keep ``import clbench_verifiers`` cheap and to give a clean
error if CLBench is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Lazy-imported below to keep ``import clbench_verifiers`` light.
_clbench_imports: dict[str, Any] = {}


def _default_seed_dataset():
    """Minimal dataset for verifiers versions that require one on env init."""
    row = {
        "prompt": [{"role": "user", "content": "<begin clbench rollout>"}],
        "answer": "",
        "info": {"source": "clbench-verifiers-env-seed"},
    }
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        return [row]
    return Dataset.from_list([row])


def _load_clbench() -> dict[str, Any]:
    """Import clbench symbols on first use."""
    if _clbench_imports:
        return _clbench_imports
    try:
        from src.interface import (  # type: ignore
            ContinualLearningTask,
            InstanceOutcome,
            Observation,
            Query,
            Response,
            TaskStepResult,
        )
        from src.registry import get_task_class  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import clbench (`src.*`). Install via "
            "`pip install --ignore-requires-python "
            "git+https://github.com/pgasawa/continual-learning-bench.git`. "
            f"Original error: {exc}"
        ) from exc
    _clbench_imports.update(
        {
            "ContinualLearningTask": ContinualLearningTask,
            "InstanceOutcome": InstanceOutcome,
            "Observation": Observation,
            "Query": Query,
            "Response": Response,
            "TaskStepResult": TaskStepResult,
            "get_task_class": get_task_class,
        }
    )
    return _clbench_imports


def _load_verifiers():
    try:
        import verifiers as vf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "verifiers not installed. Install with `pip install verifiers[all]` "
            "or use the [train] extra of this package."
        ) from exc
    return vf


@dataclass
class CLBenchRolloutState:
    """Per-rollout state carried in verifiers' ``state`` dict."""

    task: Any  # ContinualLearningTask instance
    pending_query: Any  # Query: the next user-prompt to deliver to the model
    task_schema: Any  # type[BaseModel]: the *task-native* schema for this turn
    prompt_schema: Any  # type[BaseModel]: schema we actually present to the model
    # (== task_schema if notepad disabled, else task_schema + notepad_update)
    notepad: str = ""  # current notepad content (icl_notepad-style memory)
    instance_outcomes: list = field(default_factory=list)
    parse_failures: int = 0
    notepad_updates: int = 0
    turns: int = 0
    instances_completed: int = 0
    instance_started: bool = True  # next user-msg builder prepends notepad if True
    finished: bool = False
    last_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_clbench_env(
    task_name: str,
    *,
    task_kwargs: Optional[dict[str, Any]] = None,
    max_instances_per_rollout: int = 1,
    max_turns: int = 64,
    schema_hint_in_system: bool = True,
    parse_failure_penalty: float = -1.0,
    end_on_parse_failure: bool = False,
    use_notepad: bool = False,
    notepad_max_chars: int = 4000,
    timeout_seconds: float | None = None,
    rubric_funcs: Optional[list[Callable]] = None,
):
    """
    Construct a ``CLBenchEnv`` with a ``Rubric`` already attached.

    Parameters
    ----------
    task_name
        Name registered in ``clbench`` (e.g. ``"exploitable_poker"``).
    task_kwargs
        Forwarded to the task constructor (e.g. ``{"num_instances": 10}``).
    max_instances_per_rollout
        How many CLBench *instances* to run inside one rollout. ``1`` (default)
        gives plain GRPO with no cross-instance learning. ``>1`` enables
        continual-learning mode where memory carries across instances inside
        the same rollout — closer to the bench's eval intent but more compute.
    max_turns
        Hard turn cap for verifiers' rollout loop. Should comfortably exceed
        ``max_instances_per_rollout`` × (turns per instance).
    schema_hint_in_system
        If True, prepend a JSON-schema hint to the system prompt so the model
        knows what shape its action must take. Recommended on for training.
    parse_failure_penalty
        Scalar reward delta applied per turn where the model emits unparseable
        output. Used by the default rubric (see ``rubric.py``).
    end_on_parse_failure
        If True, a parse failure terminates the rollout. If False (default),
        we continue and re-prompt the model — gives the policy a chance to
        recover and learn the correct format.
    use_notepad
        If True, augment the response schema with an optional ``notepad_update``
        field (icl_notepad style). The notepad persists across instances within
        a single rollout and is prepended to the first turn of every new
        instance. Only meaningful with ``max_instances_per_rollout > 1``.
    notepad_max_chars
        Soft cap on notepad length. If the model writes more, we truncate
        (taking the head) before injecting. Prevents context blow-up.
    """
    from .rubric import build_clbench_rubric

    rubric = build_clbench_rubric(
        parse_failure_penalty=parse_failure_penalty,
        extra_funcs=rubric_funcs or [],
    )
    return CLBenchEnv(
        task_name=task_name,
        task_kwargs=task_kwargs or {},
        max_instances_per_rollout=max_instances_per_rollout,
        schema_hint_in_system=schema_hint_in_system,
        end_on_parse_failure=end_on_parse_failure,
        use_notepad=use_notepad,
        notepad_max_chars=notepad_max_chars,
        rubric=rubric,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )


def _make_env_class():
    """
    Build the CLBenchEnv class lazily so that ``import clbench_verifiers.env``
    does not require verifiers to be installed (handy for unit tests on
    machines without GPU stack).
    """
    vf = _load_verifiers()

    class _CLBenchEnv(vf.MultiTurnEnv):
        """
        verifiers MultiTurnEnv that drives a CLBench ``ContinualLearningTask``.
        """

        def __init__(
            self,
            task_name: str,
            task_kwargs: dict[str, Any],
            max_instances_per_rollout: int,
            schema_hint_in_system: bool,
            end_on_parse_failure: bool,
            use_notepad: bool,
            notepad_max_chars: int,
            rubric,
            max_turns: int,
            timeout_seconds: float | None = None,
            **kwargs,
        ):
            self.task_name = task_name
            self.task_kwargs = dict(task_kwargs)
            self.max_instances_per_rollout = max_instances_per_rollout
            self.schema_hint_in_system = schema_hint_in_system
            self.end_on_parse_failure = end_on_parse_failure
            self.use_notepad = use_notepad
            self.notepad_max_chars = notepad_max_chars

            if use_notepad and max_instances_per_rollout < 2:
                logger.warning(
                    "use_notepad=True with max_instances_per_rollout=%d means "
                    "the notepad never gets replayed (it's reset every rollout). "
                    "You probably want max_instances_per_rollout >= 2.",
                    max_instances_per_rollout,
                )

            # We construct the task once eagerly to fail-fast if the task name
            # is wrong or its extras are missing. Rollout-time tasks are fresh
            # instances built per-rollout in setup_state.
            self._validate_task()

            # verifiers >=0.1.7 validates that every Environment has at least
            # one dataset. CLBenchEnv generates the real first prompt in
            # setup_state(), so this seed row is only there to satisfy that
            # outer Environment contract.
            kwargs.setdefault("dataset", _default_seed_dataset())

            super().__init__(
                rubric=rubric,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

        def _validate_task(self) -> None:
            cl = _load_clbench()
            cl["get_task_class"](self.task_name)  # raises if unknown

        def _new_task(self):
            cl = _load_clbench()
            task_cls = cl["get_task_class"](self.task_name)
            return task_cls(**self.task_kwargs)

        def _build_user_message(
            self,
            query,
            *,
            prompt_schema=None,
            notepad: str = "",
            instance_started: bool = False,
            error: Optional[str] = None,
        ) -> dict:
            from .parsing import format_schema_hint
            from .notepad import render_notepad_for_prompt

            parts: list[str] = []
            if instance_started and notepad:
                parts.append(render_notepad_for_prompt(notepad))
            if query.feedback is not None and query.feedback.content:
                parts.append(f"=== FEEDBACK ===\n{query.feedback.content}")
            if error:
                parts.append(
                    f"=== PARSE ERROR ===\n{error}\n"
                    "Respond again with a single JSON object that matches the "
                    "schema. Do not add prose outside the JSON."
                )
            parts.append(query.prompt)
            schema = prompt_schema or query.response_schema
            if not self.schema_hint_in_system and schema is not None:
                parts.append(
                    "Respond with a JSON object matching this schema:\n```json\n"
                    + format_schema_hint(schema)
                    + "\n```"
                )
            return {"role": "user", "content": "\n\n".join(parts)}

        def _system_prompt(self, schema) -> str:
            from .parsing import format_schema_hint

            base = (
                "You are an agent solving a continual-learning benchmark task. "
                "Respond every turn with a single JSON object — no prose, no "
                "markdown fences — matching the schema for the current turn."
            )
            if self.use_notepad:
                base += (
                    " You may write to a persistent notepad via the optional "
                    "`notepad_update` field. The notepad is shown at the start "
                    "of every new task instance, so use it to record durable, "
                    "transferable observations rather than per-turn scratch work."
                )
            if not self.schema_hint_in_system or schema is None:
                return base
            return (
                base
                + "\n\nSchema:\n```json\n"
                + format_schema_hint(schema)
                + "\n```"
            )

        # ------------------------------------------------------------------
        # verifiers MultiTurnEnv hooks
        # ------------------------------------------------------------------

        async def setup_state(self, state):
            """
            Initialize per-rollout state and seed ``state["prompt"]`` with the
            CLBench task's first user-side messages.

            The base ``Environment.init_state`` already populated
            ``state["prompt"]`` from the dataset row before we get here. We
            replace it with the actual task framing so the model never sees
            the placeholder seed prompt.

            Must return ``state`` — verifiers' rollout assigns the result back.
            """
            from .notepad import build_schema_with_notepad

            task = self._new_task()
            query = task.reset()
            task_schema = query.response_schema
            prompt_schema = (
                build_schema_with_notepad(task_schema)
                if self.use_notepad and task_schema is not None
                else task_schema
            )

            rs = CLBenchRolloutState(
                task=task,
                pending_query=query,
                task_schema=task_schema,
                prompt_schema=prompt_schema,
                instance_started=True,
            )
            state["clbench"] = rs

            sys_msg = {"role": "system", "content": self._system_prompt(prompt_schema)}
            user_msg = self._build_user_message(
                query,
                prompt_schema=prompt_schema,
                notepad=rs.notepad,
                instance_started=True,
            )
            state["prompt"] = [sys_msg, user_msg]
            # Mirror under ``state["messages"]`` for back-compat with the local
            # smoke tests (which use a stand-in MultiTurnEnv that doesn't run
            # the full rollout pipeline).
            state["messages"] = list(state["prompt"])
            rs.instance_started = False
            return state

        # NOTE: ``Environment.is_completed`` is ``@final`` and walks all
        # ``@vf.stop``-decorated methods to decide termination. We register
        # our condition that way instead of overriding ``is_completed``.

        @vf.stop
        async def clbench_done(self, state, **kwargs) -> bool:
            rs: Optional[CLBenchRolloutState] = state.get("clbench")
            if rs is None:
                return False
            return rs.finished or rs.instances_completed >= self.max_instances_per_rollout

        async def env_response(self, messages, state, **kwargs):
            """
            Process the latest assistant message and return the next user
            message(s).
            """
            cl = _load_clbench()
            Response = cl["Response"]

            rs: CLBenchRolloutState = state["clbench"]
            rs.turns += 1

            assistant_text = self._latest_assistant_text(messages)

            from .parsing import parse_action
            from .notepad import split_notepad_action

            parsed, error = parse_action(assistant_text, rs.prompt_schema)
            if parsed is None:
                rs.parse_failures += 1
                rs.last_error = error
                if self.end_on_parse_failure:
                    rs.finished = True
                    return [
                        {
                            "role": "user",
                            "content": (
                                f"Parse error: {error}. Ending rollout."
                            ),
                        }
                    ]
                # Re-prompt with the same query and an explicit error note.
                return [
                    self._build_user_message(
                        rs.pending_query,
                        prompt_schema=rs.prompt_schema,
                        notepad=rs.notepad,
                        instance_started=False,
                        error=error,
                    )
                ]

            # Strip out notepad_update before handing the action to the task.
            if self.use_notepad:
                action, notepad_update = split_notepad_action(parsed, rs.task_schema)
                if notepad_update is not None:
                    rs.notepad = self._maybe_truncate(notepad_update)
                    rs.notepad_updates += 1
            else:
                action = parsed

            response = Response(action=action, metadata={})
            try:
                step_result = rs.task.step(response)
            except Exception as exc:  # pragma: no cover - task bug
                logger.exception("CLBench task.step raised: %s", exc)
                rs.finished = True
                rs.last_error = f"task_exception:{type(exc).__name__}:{exc}"
                return [
                    {
                        "role": "user",
                        "content": f"Task error: {rs.last_error}. Ending rollout.",
                    }
                ]

            obs = step_result.observation
            instance_complete = bool(getattr(obs, "instance_complete", True))

            # Capture instance outcomes as they appear (more reliable than
            # waiting for end-of-rollout because some tasks finalize lazily).
            new_outcomes = list(rs.task.get_instance_outcomes())
            if len(new_outcomes) > len(rs.instance_outcomes):
                rs.instance_outcomes = new_outcomes

            if instance_complete:
                rs.instances_completed += 1

            # Decide whether to continue.
            done_from_task = bool(step_result.done)
            done_from_budget = rs.instances_completed >= self.max_instances_per_rollout
            if done_from_task or done_from_budget:
                rs.finished = True
                # No further user message needed; verifiers will exit on is_completed.
                # But verifiers expects env_response to return at least one message
                # in many code paths, so emit a brief terminal note.
                return [
                    {
                        "role": "user",
                        "content": f"=== ROLLOUT END ===\n{obs.content}",
                    }
                ]

            # Otherwise: hand the next Query to the model.
            next_query = step_result.next_query
            if next_query is None:
                # Task says not done but provided no next query — treat as done.
                rs.finished = True
                return [
                    {
                        "role": "user",
                        "content": "=== ROLLOUT END (no next query) ===",
                    }
                ]

            # Update schemas if the task swapped them mid-rollout.
            from .notepad import build_schema_with_notepad

            if next_query.response_schema is not None:
                rs.task_schema = next_query.response_schema
                rs.prompt_schema = (
                    build_schema_with_notepad(rs.task_schema)
                    if self.use_notepad
                    else rs.task_schema
                )

            rs.pending_query = next_query
            instance_started = bool(instance_complete)
            return [
                self._build_user_message(
                    next_query,
                    prompt_schema=rs.prompt_schema,
                    notepad=rs.notepad,
                    instance_started=instance_started,
                )
            ]

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _maybe_truncate(self, text: str) -> str:
            if not text or len(text) <= self.notepad_max_chars:
                return text
            head = text[: self.notepad_max_chars]
            return head + "\n[... notepad truncated ...]"

        @staticmethod
        def _latest_assistant_text(messages) -> str:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        # OpenAI-style content blocks.
                        return "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
            return ""

    return _CLBenchEnv


# Lazy class — verifiers is a heavy import.
class _CLBenchEnvProxy:
    """Indirection so ``CLBenchEnv(...)`` works without importing verifiers eagerly."""

    _impl = None

    def __call__(self, *args, **kwargs):
        if self._impl is None:
            self._impl = _make_env_class()
        return self._impl(*args, **kwargs)


CLBenchEnv = _CLBenchEnvProxy()
