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


def _default_seed_dataset(num_rows: int = 256):
    """
    Dataset of varying seeds for per-step prompt diversity.

    Each row carries a distinct ``info.seed`` that ``setup_state`` threads
    into the task's ``task_kwargs.seed``. With this, the 8 rollouts within
    a GRPO step share the same hand (proper group-relative advantage), but
    each step trains on a different hand (the policy generalises across
    scenarios). With a 1-row dataset every step would replay the same
    hand — the bug we hit in earlier runs.
    """
    rows = [
        {
            "prompt": [{"role": "user", "content": f"<begin clbench rollout seed={i}>"}],
            "answer": "",
            "info": {"seed": i, "source": "clbench-verifiers-env-seed"},
        }
        for i in range(max(1, num_rows))
    ]
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        return rows
    return Dataset.from_list(rows)


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


def _format_score(text: str, schema, *, parsed_ok: bool) -> float:
    """
    Heuristic 0..1 score: how close is ``text`` to a valid action JSON?

    Used to break zero-advantage on cold-start GRPO groups where no rollout
    fully parses. Awarded points (out of 1.0):

    - 0.5 if the schema validates (full credit; in practice the rubric's
      mean_instance_reward will dominate once parses happen).
    - 0.2 if there is a ``{ ... }`` block.
    - up to 0.3 prorated by how many of the schema's *required* field
      names appear (case-insensitive) in the text.

    Cheap and deliberately permissive so the gradient signal kicks in early.
    """
    if parsed_ok:
        return 1.0
    if not text:
        return 0.0

    score = 0.0
    if "{" in text and "}" in text:
        score += 0.2

    fields: list[str] = []
    if schema is not None:
        try:
            fields = list(schema.model_fields.keys())
        except Exception:
            fields = []
    if fields:
        text_low = text.lower()
        hits = sum(1 for f in fields if f.lower() in text_low)
        score += 0.3 * hits / len(fields)

    return min(score, 0.5)


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
    # Best per-turn "did the output look like the schema?" score in [0, 1].
    # Updated on every assistant message; surfaced as a small-weight rubric
    # component so cold-start GRPO gets advantage variance even when no
    # rollout in the group manages to fully parse. See `_format_score`.
    best_format_score: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_clbench_env(
    task_name: str,
    *,
    task_kwargs: Optional[dict[str, Any]] = None,
    max_instances_per_rollout: int = 1,
    max_turns: int = 16,
    max_input_tokens_per_rollout: int = 8000,
    schema_hint_in_system: bool = True,
    parse_failure_penalty: float = -1.0,
    format_shaping_weight: float = 0.1,
    end_on_parse_failure: bool = True,
    use_notepad: bool = False,
    notepad_max_chars: int = 4000,
    clear_history_between_instances: bool = False,
    final_instance_reward_weight: float = 0.0,
    mean_instance_reward_weight: float = 1.0,
    enable_guided_json: bool = True,
    dataset_size: int = 256,
    timeout_seconds: float | None = None,
    rubric_funcs: Optional[list[Callable]] = None,
):
    """
    Construct a ``CLBenchEnv`` with a ``Rubric`` already attached.

    Defaults are tuned for **early training** with an untrained base model:
    aggressive turn and token caps and immediate exit on parse failure, so
    cold-start cost spirals (an untrained policy that emits gibberish for
    30+ re-prompted turns) cannot blow up. Once a policy emits valid actions
    reliably, raise ``max_turns`` and set ``end_on_parse_failure=False``.

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
        the same rollout.
    max_turns
        Hard turn cap for verifiers' rollout loop. Default ``16`` keeps cold-
        start rollouts bounded. Raise once the policy is producing valid
        actions; lower is fine for evaluation rollouts that don't need much
        depth.
    max_input_tokens_per_rollout
        Per-rollout cap on cumulative *input* tokens fed to the policy
        (read from verifiers' usage tracker). Default ``8000`` keeps bad
        rollouts from blowing up context quadratically when each turn
        replays the prior conversation. Set to ``0`` to disable.
    schema_hint_in_system
        If True, prepend a JSON-schema hint to the system prompt so the model
        knows what shape its action must take. Recommended on for training.
    parse_failure_penalty
        Scalar reward delta applied per turn where the model emits unparseable
        output. Used by the default rubric (see ``rubric.py``).
    end_on_parse_failure
        If True (the new default), a parse failure terminates the rollout
        immediately, which is the correct behaviour during cost-sensitive
        early training. Set to ``False`` once your policy emits valid JSON
        reliably; the re-prompt path is more sample-efficient but only when
        the model can actually recover within a few turns.
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
        format_shaping_weight=format_shaping_weight,
        mean_instance_reward_weight=mean_instance_reward_weight,
        final_instance_reward_weight=final_instance_reward_weight,
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
        max_input_tokens_per_rollout=max_input_tokens_per_rollout,
        clear_history_between_instances=clear_history_between_instances,
        enable_guided_json=enable_guided_json,
        dataset_size=dataset_size,
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
            max_input_tokens_per_rollout: int,
            clear_history_between_instances: bool,
            enable_guided_json: bool,
            rubric,
            max_turns: int,
            timeout_seconds: float | None = None,
            dataset_size: int = 256,
            **kwargs,
        ):
            self.task_name = task_name
            self.task_kwargs = dict(task_kwargs)
            self.max_instances_per_rollout = max_instances_per_rollout
            self.schema_hint_in_system = schema_hint_in_system
            self.end_on_parse_failure = end_on_parse_failure
            self.use_notepad = use_notepad
            self.notepad_max_chars = notepad_max_chars
            self.max_input_tokens_per_rollout = max_input_tokens_per_rollout
            self.clear_history_between_instances = clear_history_between_instances
            self.enable_guided_json = enable_guided_json

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
            kwargs.setdefault("dataset", _default_seed_dataset(num_rows=dataset_size))

            super().__init__(
                rubric=rubric,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

        def _validate_task(self) -> None:
            cl = _load_clbench()
            cl["get_task_class"](self.task_name)  # raises if unknown

        def _new_task(self, seed_override: Optional[int] = None):
            cl = _load_clbench()
            task_cls = cl["get_task_class"](self.task_name)
            kwargs = dict(self.task_kwargs)
            if seed_override is not None:
                kwargs["seed"] = int(seed_override)
            return task_cls(**kwargs)

        @staticmethod
        def _extract_dataset_seed(state) -> Optional[int]:
            """Pull ``info.seed`` from the rollout state, tolerant to the few
            shapes verifiers exposes (top-level info, nested under input)."""
            info = state.get("info") if isinstance(state, dict) else None
            if not isinstance(info, dict):
                inp = state.get("input") if isinstance(state, dict) else None
                if isinstance(inp, dict):
                    info = inp.get("info")
            if isinstance(info, dict):
                seed = info.get("seed")
                if isinstance(seed, int):
                    return seed
                if isinstance(seed, str) and seed.lstrip("-").isdigit():
                    return int(seed)
            return None

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

            Reads ``state["info"]["seed"]`` (or ``state["input"]["info"]["seed"]``)
            and threads it into the task's ``task_kwargs.seed``. This is what
            gives us per-step prompt diversity — the 8 rollouts in a GRPO
            group share the same seed (= same hand), but each training step
            picks a different dataset row (= different seed = different hand)
            so the policy generalises across scenarios rather than memorising
            seed=0.

            Must return ``state`` — verifiers' rollout assigns the result back.
            """
            from .notepad import build_schema_with_notepad

            seed_override = self._extract_dataset_seed(state)
            task = self._new_task(seed_override=seed_override)
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

            # Inject guided_json into sampling_args so vLLM constrains the
            # policy's output to valid JSON matching the response schema.
            # This is the single biggest fix for cold-start training: without
            # it, the policy can emit truncated/malformed JSON and most of the
            # gradient signal goes to "fix syntax", not "play better poker".
            if self.enable_guided_json and prompt_schema is not None:
                self._inject_guided_json(state, prompt_schema)

            rs.instance_started = False
            return state

        def _inject_guided_json(self, state, schema) -> None:
            """Merge a guided_json schema into ``state["sampling_args"]``.

            See ``_apply_constraint`` for the actual merge logic; this just
            persists into rollout state for paths that read sampling_args
            from there.
            """
            sampling_args = state.get("sampling_args")
            if not isinstance(sampling_args, dict):
                sampling_args = {}
            state["sampling_args"] = self._apply_constraint(dict(sampling_args), schema)

        @staticmethod
        def _apply_constraint(sampling_args: dict, schema) -> dict:
            """Inject both vLLM-native (``extra_body.guided_json``) and
            OpenAI-standard (``response_format``) JSON-mode constraints into a
            sampling_args dict. We set both forms because Prime's hosted
            inference pool may strip one and forward the other; whichever
            survives, vLLM constrains generation to valid JSON.
            """
            try:
                schema_dict = schema.model_json_schema()
            except Exception as exc:
                logger.warning("guided_json: schema serialization failed: %s", exc)
                return sampling_args

            extra_body = sampling_args.get("extra_body")
            extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
            extra_body["guided_json"] = schema_dict
            sampling_args["extra_body"] = extra_body

            sampling_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema_dict,
                    "strict": True,
                },
            }
            return sampling_args

        # Note: an earlier version of this class overrode `get_model_response`
        # to force-inject the guided_json constraint regardless of caller, but
        # that broke Prime's TITO (token-in-token-out) call path which doesn't
        # go through the standard chat-completions surface. The constraint
        # now flows from `[sampling.extra_body.guided_json]` in the Prime
        # training TOML, with `state["sampling_args"]` injection as a backup
        # for non-Prime callers.

        async def get_prompt_messages(self, state):
            """
            Override to optionally wipe within-instance conversation history
            at instance boundaries (CLBench's `icl_notepad` semantics).

            With ``clear_history_between_instances=True``, when a turn
            completes an instance, the next turn's prompt becomes
            ``[system_message, instance_N+1_first_user_message]`` — the
            notepad (if enabled) is the only thing carrying state forward.
            Without the flag, the verifiers default applies: every prior
            turn's history accumulates in context (CLBench's plain `icl`).
            """
            if not self.clear_history_between_instances:
                return await super().get_prompt_messages(state)

            if not state.get("trajectory"):
                return state["prompt"]

            from verifiers.utils.message_utils import (  # type: ignore
                concat_messages,
                maybe_normalize_messages,
            )

            rs: Optional[CLBenchRolloutState] = state.get("clbench")
            prev_step = state["trajectory"][-1]
            prev_messages = concat_messages([prev_step["prompt"], prev_step["completion"]])

            instances_before = rs.instances_completed if rs is not None else 0

            new_user_msgs = await self.env_response(prev_messages, state)
            new_user_msgs = maybe_normalize_messages(
                new_user_msgs, field_name="env_response"
            )

            instance_just_completed = (
                rs is not None and rs.instances_completed > instances_before
            )

            if instance_just_completed and rs is not None and not rs.finished:
                # Build a fresh prompt: only the system message survives the
                # boundary; the new user message already includes the notepad
                # (env_response prepends it via _build_user_message when
                # instance_started=True).
                system_msgs = [
                    m for m in state["prompt"]
                    if isinstance(m, dict) and m.get("role") == "system"
                ]
                if system_msgs:
                    return concat_messages([system_msgs, new_user_msgs])

            # Default path: same as the parent's get_prompt_messages.
            return concat_messages([prev_messages, new_user_msgs])

        # NOTE: ``Environment.is_completed`` is ``@final`` and walks all
        # ``@vf.stop``-decorated methods to decide termination. We register
        # our condition that way instead of overriding ``is_completed``.

        @vf.stop
        async def clbench_done(self, state, **kwargs) -> bool:
            rs: Optional[CLBenchRolloutState] = state.get("clbench")
            if rs is None:
                return False
            return rs.finished or rs.instances_completed >= self.max_instances_per_rollout

        @vf.stop
        async def input_token_budget_exceeded(self, state, **kwargs) -> bool:
            """Hard cap on cumulative input tokens fed to the policy this rollout.

            Without this cap, a base model that emits unparseable output keeps
            getting re-prompted with a growing conversation, so input tokens
            grow quadratically with turn count. We read from verifiers'
            built-in usage tracker; if it isn't populated yet, we fall through.
            """
            if self.max_input_tokens_per_rollout <= 0:
                return False
            try:
                usage = self.get_state_usage(state)
            except Exception:
                return False
            if usage is None:
                return False
            return float(usage.get("input_tokens", 0)) >= self.max_input_tokens_per_rollout

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
            # Update partial-format score for this turn (0..1). Provides reward
            # variance during cold-start even when no rollout fully parses.
            score = _format_score(assistant_text, rs.prompt_schema, parsed_ok=parsed is not None)
            if score > rs.best_format_score:
                rs.best_format_score = score
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
            """
            Return the most recent assistant message's text.

            Thinking models (Qwen3.5, Nemotron, GPT-OSS at high effort) split
            their output into ``content`` (final answer) and ``reasoning_content``
            (chain of thought). Cold-start policies often dump *everything* in
            ``reasoning_content`` and leave ``content`` null. We accept either —
            and concatenate both when available — so the parser sees whatever
            the model actually emitted.

            We also handle OpenAI-style content blocks (``[{type: text, text: ...}]``).
            """
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue

                parts: list[str] = []
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                elif isinstance(content, list):
                    parts.append(
                        "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    )

                reasoning = msg.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    parts.append(reasoning)

                joined = "\n".join(p for p in parts if p)
                if joined:
                    return joined
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
