"""
CLBench System adapter that talks to a vLLM OpenAI-compatible server.

Use case: after GRPO training spits out a checkpoint, you serve it with
``vllm serve <ckpt> --port 8000`` and run

    clbench run exploitable_poker --system vllm_local \\
        --system.base_url http://localhost:8000/v1 \\
        --system.model my-trained-checkpoint

to get the official benchmark scores. This system is intentionally minimal —
no memory, no notepad, no retrieval — so it measures the *raw* policy. Pair
it with the ``icl_notepad`` schema augmentation in a follow-up if you want
notepad-style memory at eval time.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# These imports are deferred to runtime so this module can be imported on
# machines without clbench installed (e.g. during local lint/test).
try:
    from src.interface import (  # type: ignore
        ContinualLearningSystem,
        Query,
        Response,
    )
    from src.registry import register_system  # type: ignore
    from src.usage import UsageEvent  # type: ignore

    _CLBENCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CLBENCH_AVAILABLE = False
    ContinualLearningSystem = object  # type: ignore
    register_system = lambda name: (lambda cls: cls)  # noqa: E731
    Query = Response = UsageEvent = None  # type: ignore


from .notepad import (
    build_schema_with_notepad,
    render_notepad_for_prompt,
    split_notepad_action,
)
from .parsing import format_schema_hint, parse_action


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agent solving a continual-learning benchmark task. "
    "Respond every turn with a single JSON object — no prose, no markdown "
    "fences — matching the schema for the current turn."
)
_NOTEPAD_SYSTEM_PROMPT_SUFFIX = (
    " You may write to a persistent notepad via the optional `notepad_update` "
    "field. The notepad is shown at the start of every new task instance, so "
    "use it to record durable, transferable observations rather than per-turn "
    "scratch work."
)


@register_system("vllm_local")
class VLLMClientSystem(ContinualLearningSystem):  # type: ignore[misc]
    """
    CLBench system that calls a local vLLM OpenAI-compatible server.

    The CLBench harness drives the rollout (single threaded, sync); we just
    need to translate (Query → chat messages → completion → parsed action).
    """

    supports_baseline = True

    def __init__(
        self,
        name: str = "vllm_local",
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        api_key: str = "EMPTY",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        keep_history: bool = True,
        max_history_messages: int = 200,
        use_notepad: bool = False,
        notepad_max_chars: int = 4000,
        clear_notepad_between_instances: bool = False,
        clear_context_between_instances: bool = False,
        enable_guided_json: bool = False,
    ):
        if not _CLBENCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "clbench is not installed; cannot construct VLLMClientSystem."
            )
        super().__init__()
        self._name = name
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.keep_history = keep_history
        self.max_history_messages = max_history_messages
        self.use_notepad = use_notepad
        self.notepad_max_chars = notepad_max_chars
        self.clear_notepad_between_instances = clear_notepad_between_instances
        self.clear_context_between_instances = clear_context_between_instances
        self.enable_guided_json = enable_guided_json

        # Lazy client — construct on first use so module imports stay cheap.
        self._client = None
        self._messages: list[dict[str, Any]] = []
        self._current_schema_id: Optional[str] = None
        self._notepad: str = ""
        self._instance_started: bool = True
        self._last_instance_id: Optional[str] = None

    # --- CLBench plumbing -------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._messages = []
        self._current_schema_id = None
        self._notepad = ""
        self._instance_started = True
        self._last_instance_id = None

    # --- Core respond loop ------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "openai package required for VLLMClientSystem. "
                    "Install with `pip install openai`."
                ) from exc
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout,
            )
        return self._client

    def _system_prompt(self, schema) -> str:
        base = _DEFAULT_SYSTEM_PROMPT
        if self.use_notepad:
            base += _NOTEPAD_SYSTEM_PROMPT_SUFFIX
        if schema is None:
            return base
        return base + "\n\nSchema:\n```json\n" + format_schema_hint(schema) + "\n```"

    def _format_user_message(self, query) -> dict:
        parts: list[str] = []
        if self.use_notepad and self._instance_started and self._notepad:
            parts.append(render_notepad_for_prompt(self._notepad))
        if query.feedback is not None and query.feedback.content:
            parts.append(f"=== FEEDBACK ===\n{query.feedback.content}")
        parts.append(query.prompt)
        return {"role": "user", "content": "\n\n".join(parts)}

    def _maybe_swap_system_prompt(self, schema) -> None:
        """Replace the leading system message when the response schema changes."""
        schema_id = (
            getattr(schema, "__qualname__", None)
            or getattr(schema, "__name__", None)
            or repr(schema)
        )
        if schema_id == self._current_schema_id and self._messages:
            return
        self._current_schema_id = schema_id
        sys_msg = {"role": "system", "content": self._system_prompt(schema)}
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0] = sys_msg
        else:
            self._messages.insert(0, sys_msg)

    def _truncate_history(self) -> None:
        if len(self._messages) <= self.max_history_messages:
            return
        # Keep the system message at the head, drop the oldest non-system messages.
        head = self._messages[:1] if self._messages[0].get("role") == "system" else []
        tail = self._messages[-(self.max_history_messages - len(head)) :]
        self._messages = head + tail

    def respond(self, query) -> Any:
        client = self._ensure_client()

        # Detect instance boundary by instance_id transition + feedback flag.
        feedback_complete = (
            query.feedback is not None
            and bool(getattr(query.feedback, "instance_complete", True))
        )
        instance_id_changed = (
            query.instance_id is not None
            and query.instance_id != self._last_instance_id
        )
        self._instance_started = (
            instance_id_changed or feedback_complete or self._last_instance_id is None
        )
        if (
            self._instance_started
            and self.clear_notepad_between_instances
            and self._last_instance_id is not None
        ):
            self._notepad = ""
        # Mirror the training-time history wipe at instance boundaries: only
        # the system message survives, so the policy's only memory channel
        # is the notepad (when use_notepad=True). Without this the eval
        # distribution diverges from training and the comparison is unfair.
        if (
            self._instance_started
            and self.clear_context_between_instances
            and self._last_instance_id is not None
        ):
            head = (
                self._messages[:1]
                if self._messages and self._messages[0].get("role") == "system"
                else []
            )
            self._messages = list(head)
        self._last_instance_id = query.instance_id

        # Pick the schema we'll show / parse against.
        task_schema = query.response_schema
        prompt_schema = (
            build_schema_with_notepad(task_schema)
            if self.use_notepad and task_schema is not None
            else task_schema
        )

        self._maybe_swap_system_prompt(prompt_schema)
        self._messages.append(self._format_user_message(query))
        self._truncate_history()

        # Build sampling kwargs. When guided_json is enabled we pass the
        # Pydantic-derived schema with notepad_update force-marked required
        # (same logic used during training in env._apply_constraint), so
        # the eval policy is held to the same grammar as training.
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=self._messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.enable_guided_json and prompt_schema is not None:
            try:
                schema_dict = prompt_schema.model_json_schema()
                props = schema_dict.get("properties") if isinstance(schema_dict, dict) else None
                if isinstance(props, dict) and "notepad_update" in props:
                    required = list(schema_dict.get("required") or [])
                    if "notepad_update" not in required:
                        required.append("notepad_update")
                        schema_dict["required"] = required
                kwargs["extra_body"] = {"guided_json": schema_dict}
            except Exception:  # pragma: no cover - schema serialisation rare
                pass

        completion = client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""

        # Record token usage if the server reports it.
        usage = getattr(completion, "usage", None)
        if usage is not None and UsageEvent is not None:
            self.record_usage_event(
                UsageEvent(
                    provider="vllm_local",
                    model=self.model,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    cost_usd=0.0,
                )
            )

        parsed, error = parse_action(text, prompt_schema)
        if parsed is None:
            try:
                action: Any = task_schema()  # type: ignore[call-arg]
            except Exception:
                action = _coerce_minimal_action(task_schema, text)
            notepad_update = None
        else:
            if self.use_notepad and task_schema is not None:
                action, notepad_update = split_notepad_action(parsed, task_schema)
            else:
                action, notepad_update = parsed, None

        if notepad_update is not None:
            if len(notepad_update) > self.notepad_max_chars:
                notepad_update = (
                    notepad_update[: self.notepad_max_chars] + "\n[... truncated ...]"
                )
            self._notepad = notepad_update

        if self.keep_history:
            self._messages.append({"role": "assistant", "content": text})

        return Response(
            action=action,
            metadata={
                "parse_error": error,
                "raw_text": text,
                "notepad_updated": notepad_update is not None,
                "notepad_length": len(self._notepad),
            },
        )


def _coerce_minimal_action(schema, raw_text: str):
    """Build any schema instance, even if validators would normally reject it."""
    try:
        return schema.model_construct()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        # If even model_construct fails, embed the raw text in a dict and let
        # the task surface the resulting validation error downstream.
        return {"raw_text": raw_text}
