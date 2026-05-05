"""
Smoke test runnable on a CPU-only machine (no verifiers / no vLLM required).
Exercises the parser, the lazy clbench import path, and that we can drive a
poker task end-to-end through ``setup_state`` + ``env_response`` using the
*inner* MultiTurnEnv class directly.

Usage (from inside the clbench venv with cl-benchmark installed):

    python -m pytest tests/test_env_smoke.py -v

Or just run as a script:

    python tests/test_env_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the package importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_parse_action_extracts_fenced_json():
    from pydantic import BaseModel
    from clbench_verifiers.parsing import parse_action

    class A(BaseModel):
        action: str
        amount: int = 0

    text = '''Sure, here's my move:
```json
{"action": "RAISE", "amount": 50}
```
'''
    a, err = parse_action(text, A)
    assert err is None, err
    assert a.action == "RAISE"
    assert a.amount == 50


def test_parse_action_handles_inline_json():
    from pydantic import BaseModel
    from clbench_verifiers.parsing import parse_action

    class A(BaseModel):
        action: str

    a, err = parse_action('Going to {"action": "FOLD"} now.', A)
    assert err is None
    assert a.action == "FOLD"


def test_parse_action_rejects_garbage():
    from pydantic import BaseModel
    from clbench_verifiers.parsing import parse_action

    class A(BaseModel):
        action: str

    a, err = parse_action("FOLD!", A)
    assert a is None
    assert err is not None


def test_format_schema_hint_renders_pydantic_schema():
    from pydantic import BaseModel, Field
    from clbench_verifiers.parsing import format_schema_hint

    class A(BaseModel):
        action: str = Field(description="Pick one.")
        amount: int = 0

    s = format_schema_hint(A)
    assert "action" in s and "amount" in s


def _build_local_env_class():
    """
    Build the inner MultiTurnEnv class without importing verifiers, by
    monkey-patching a minimal stand-in that supplies the surface we use.
    """
    import clbench_verifiers.env as env_mod

    # Stand-in for vf.MultiTurnEnv that records init args and ignores Rubric.
    class _StandinMultiTurnEnv:
        def __init__(self, rubric=None, max_turns=64, timeout_seconds=None, **kw):
            self.rubric = rubric
            self.max_turns = max_turns
            self.timeout_seconds = timeout_seconds
            self.kw = kw

    class _StandinVF:
        MultiTurnEnv = _StandinMultiTurnEnv

        # parse_action and the env wrapper don't construct other vf objects.
        class Rubric:
            def __init__(self, funcs=None, weights=None):
                self.funcs = funcs or []
                self.weights = weights or []

    # Patch the module-level cache so _load_verifiers returns our standin.
    original_loader = env_mod._load_verifiers
    env_mod._load_verifiers = lambda: _StandinVF  # type: ignore
    env_mod._CLBenchEnvProxy._impl = None  # force rebuild

    try:
        # Force lazy class build now using the standin.
        cls = env_mod._make_env_class()
        return cls
    finally:
        env_mod._load_verifiers = original_loader


async def _drive_one_turn(env, valid_action_text: str):
    state = {"messages": []}
    await env.setup_state(state)

    # First user message contains the prompt and (in default config) the schema.
    first_user = state["messages"][-1]
    assert first_user["role"] == "user"
    assert first_user["content"]

    # Simulate model emitting the action text.
    state["messages"].append({"role": "assistant", "content": valid_action_text})
    new_msgs = await env.env_response(state["messages"], state)

    rs = state["clbench"]
    return rs, new_msgs


def test_env_drives_poker_task_smoke():
    """
    Build the env (using a verifiers stand-in) and step it through one turn
    of exploitable_poker. Requires cl-benchmark + texasholdem installed.
    """
    cls = _build_local_env_class()

    # Build a tiny rubric stand-in that exposes .funcs/.weights so the env
    # __init__ doesn't blow up.
    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={
            "num_instances": 2,
            "opponent_policy": "calling_station",
            "seed": 0,
        },
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        rubric=_Rubric(),
        max_turns=32,
    )

    # Fully valid poker action JSON.
    valid_text = (
        '```json\n{"thinking": "small ev call", "action": "FOLD"}\n```'
    )

    async def go():
        return await _drive_one_turn(env, valid_text)

    rs, new_msgs = asyncio.run(go())

    assert rs.parse_failures == 0
    assert rs.turns == 1
    assert isinstance(new_msgs, list) and len(new_msgs) >= 1
    # Either we ended this hand (instance) or got a follow-up query.
    assert new_msgs[0]["role"] == "user"


def test_env_handles_parse_failure():
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 2, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        rubric=_Rubric(),
        max_turns=32,
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        state["messages"].append({"role": "assistant", "content": "FOLD now"})
        msgs = await env.env_response(state["messages"], state)
        return state["clbench"], msgs

    rs, msgs = asyncio.run(go())
    assert rs.parse_failures == 1
    # Re-prompt should mention parse error.
    assert any("PARSE ERROR" in m.get("content", "") for m in msgs)


if __name__ == "__main__":
    test_parse_action_extracts_fenced_json()
    test_parse_action_handles_inline_json()
    test_parse_action_rejects_garbage()
    test_format_schema_hint_renders_pydantic_schema()
    test_env_drives_poker_task_smoke()
    test_env_handles_parse_failure()
    print("All smoke tests passed.")
