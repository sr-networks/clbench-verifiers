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

        # @vf.stop is used to register termination conditions — for tests we
        # treat it as an identity decorator (the standin never drives the
        # real is_completed loop, so registration is a no-op).
        @staticmethod
        def stop(arg=None, **_kwargs):
            if callable(arg):
                return arg

            def _wrap(func):
                return func

            return _wrap

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
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
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
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
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


def test_notepad_schema_augmentation():
    from pydantic import BaseModel
    from clbench_verifiers.notepad import (
        build_schema_with_notepad,
        split_notepad_action,
    )

    class A(BaseModel):
        action: str
        amount: int = 0

    Aug = build_schema_with_notepad(A)
    assert "notepad_update" in Aug.model_fields
    # Round-trip with notepad.
    parsed = Aug(action="FOLD", notepad_update="opponent calls a lot")
    task_action, np_text = split_notepad_action(parsed, A)
    assert task_action.action == "FOLD"
    assert np_text == "opponent calls a lot"
    # Round-trip without notepad.
    parsed2 = Aug(action="CALL")
    task_action2, np_text2 = split_notepad_action(parsed2, A)
    assert task_action2.action == "CALL"
    assert np_text2 is None


def test_env_notepad_mode_persists_across_instances():
    """
    Drive a notepad-enabled env through one parse-success turn that also
    writes the notepad, then verify the notepad is preserved on rollout state.
    """
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={
            "num_instances": 4,
            "opponent_policy": "calling_station",
            "seed": 0,
        },
        max_instances_per_rollout=4,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=True,
        notepad_max_chars=200,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=64,
    )

    valid_text = (
        '```json\n'
        '{"thinking": "fold weak", "action": "FOLD", '
        '"notepad_update": "opponent is a calling station; value-bet wide"}\n'
        '```'
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        # System prompt should mention the notepad.
        sys_content = state["messages"][0]["content"]
        assert "notepad" in sys_content.lower()
        state["messages"].append({"role": "assistant", "content": valid_text})
        await env.env_response(state["messages"], state)
        return state["clbench"]

    rs = asyncio.run(go())
    assert rs.notepad_updates == 1
    assert "calling station" in rs.notepad
    assert rs.parse_failures == 0


def test_env_notepad_truncates_oversized():
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 2, "seed": 0},
        max_instances_per_rollout=2,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=True,
        notepad_max_chars=50,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=32,
    )

    big_note = "x" * 500
    text = (
        '{"thinking": "test", "action": "FOLD", "notepad_update": '
        f'"{big_note}"}}'
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        state["messages"].append({"role": "assistant", "content": text})
        await env.env_response(state["messages"], state)
        return state["clbench"]

    rs = asyncio.run(go())
    assert len(rs.notepad) <= 50 + len("\n[... notepad truncated ...]") + 5


def test_guided_json_injected_into_sampling_args():
    """When enable_guided_json=True, setup_state must populate
    state['sampling_args']['extra_body']['guided_json'] with the
    response schema's JSON schema dict."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=True,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    async def go():
        # Pre-populate sampling_args to confirm we merge rather than overwrite.
        state = {"messages": [], "sampling_args": {"temperature": 0.7}}
        await env.setup_state(state)
        return state

    state = asyncio.run(go())
    sa = state.get("sampling_args")
    assert isinstance(sa, dict)
    assert sa.get("temperature") == 0.7, "must preserve pre-existing keys"
    extra = sa.get("extra_body")
    assert isinstance(extra, dict)
    schema = extra.get("guided_json")
    assert isinstance(schema, dict)
    # exploitable_poker's PokerAction schema has these required fields.
    assert "action" in (schema.get("properties") or {})
    assert "thinking" in (schema.get("properties") or {})


def test_guided_json_off_when_disabled():
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        return state

    state = asyncio.run(go())
    sa = state.get("sampling_args") or {}
    extra = sa.get("extra_body") or {}
    assert "guided_json" not in extra


def test_latest_assistant_text_reads_reasoning_content():
    """
    Thinking models (Qwen3.5, Nemotron) put output in `reasoning_content`
    not `content`. The env must read both so cold-start policies don't appear
    to be emitting empty text.
    """
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    # Empty content but JSON in reasoning_content — must be picked up.
    msgs_reasoning_only = [
        {"role": "user", "content": "play"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": '{"thinking":"x","action":"FOLD"}',
        },
    ]
    text = env._latest_assistant_text(msgs_reasoning_only)
    assert "FOLD" in text

    # Both populated → joined (so parser sees whatever the model wrote).
    msgs_both = [
        {"role": "user", "content": "play"},
        {
            "role": "assistant",
            "content": "final answer",
            "reasoning_content": "thinking out loud",
        },
    ]
    text2 = env._latest_assistant_text(msgs_both)
    assert "final answer" in text2 and "thinking out loud" in text2

    # Plain content path still works.
    msgs_content_only = [
        {"role": "user", "content": "play"},
        {"role": "assistant", "content": '{"action":"CALL"}'},
    ]
    assert "CALL" in env._latest_assistant_text(msgs_content_only)


def test_legal_actions_are_extracted_from_situation_line():
    from clbench_verifiers.env import _extract_legal_actions

    preflop_facing_raise = (
        "Hand #1 - PREFLOP\n"
        "Pot: 15\n"
        "Situation: Opponent raised to 10 (you need 5 to call)\n"
        "What's your action?"
    )
    flop_check_to_you = (
        "Hand #1 - FLOP\n"
        "Situation: Action to you (you can check or raise)\n"
    )
    out_pf = _extract_legal_actions(preflop_facing_raise)
    out_fl = _extract_legal_actions(flop_check_to_you)
    assert out_pf is not None and "CHECK" not in out_pf and "CALL" in out_pf
    assert out_fl is not None and "CHECK" in out_fl


def test_legal_actions_prefix_appears_in_user_message():
    """User message should carry an explicit `=== LEGAL ACTIONS THIS TURN ===`
    header parsed from the task's prompt."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0, "opponent_policy": "calling_station"},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        return state["prompt"][-1]["content"]

    first_user = asyncio.run(go())
    assert "LEGAL ACTIONS THIS TURN" in first_user
    # Preflop facing a raise → CALL/FOLD/RAISE legal, CHECK not.
    assert "CALL" in first_user and "FOLD" in first_user
    # CHECK should not be in the legal-actions block (it'll still appear in the
    # task's own action menu below it; we only check the LEGAL ACTIONS block).
    legal_block = first_user.split("LEGAL ACTIONS THIS TURN", 1)[1].split("\n\n")[0]
    assert "CHECK" not in legal_block


def test_observation_propagated_as_feedback_at_instance_boundary():
    """When a turn ends an instance and the rollout continues to a next
    instance, the next user message must include the just-finished
    observation as `=== FEEDBACK ===`. Without this the model can't
    write notes about prior-hand outcomes."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={
            "num_instances": 4,
            "opponent_policy": "calling_station",
            "seed": 0,
        },
        max_instances_per_rollout=4,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=True,
        notepad_max_chars=500,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=True,
        rubric=_Rubric(),
        max_turns=64,
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        # Drive successive FOLD actions — each one ends the current poker
        # hand quickly, so we cross instance boundaries.
        valid_action = (
            '{"thinking": "fold weak", "action": "FOLD", '
            '"notepad_update": "opponent calls a lot"}'
        )
        boundary_msgs = []
        for _ in range(8):  # generous; usually only need 1-3 turns per FOLD
            state["messages"].append({"role": "assistant", "content": valid_action})
            new_msgs = await env.env_response(state["messages"], state)
            state["messages"].extend(new_msgs)
            rs = state["clbench"]
            content = new_msgs[-1].get("content", "")
            # Detect a boundary: instance just bumped AND new prompt contains FEEDBACK.
            if rs.instances_completed >= 1 and "=== FEEDBACK ===" in content:
                boundary_msgs.append(content)
            if rs.instances_completed >= 2:
                break
        return boundary_msgs

    boundaries = asyncio.run(go())
    assert boundaries, "expected at least one instance-boundary message with FEEDBACK"
    # Outcome wording from the poker task includes 'Hand' and either 'WON' or 'LOST'.
    bm = boundaries[0]
    assert "Hand" in bm
    assert "WON" in bm or "LOST" in bm or "complete" in bm.lower()


def test_dataset_seed_is_threaded_into_task_kwargs():
    """state['info']['seed'] should override task_kwargs.seed in setup_state."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0, "opponent_policy": "calling_station"},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=False,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    async def go(seed):
        state = {"messages": [], "info": {"seed": seed}}
        await env.setup_state(state)
        return state["clbench"].task

    # Two rollouts with different seeds → different Poker tasks initialized.
    t0 = asyncio.run(go(0))
    t7 = asyncio.run(go(7))
    # Poker stores its seed; check we threaded it through.
    assert getattr(t0, "seed", None) == 0
    assert getattr(t7, "seed", None) == 7


def test_final_instance_reward_returns_last_outcome():
    """final_instance_reward should return the last completed instance's reward,
    independent of what the mean is."""
    import asyncio as _asyncio
    from types import SimpleNamespace
    from clbench_verifiers.rubric import final_instance_reward

    rs = SimpleNamespace(
        instance_outcomes=[
            SimpleNamespace(reward=-1.0),
            SimpleNamespace(reward=-2.0),
            SimpleNamespace(reward=+5.0),  # final
        ]
    )
    state = {"clbench": rs}
    val = _asyncio.run(final_instance_reward(state=state))
    assert val == 5.0

    empty = {"clbench": SimpleNamespace(instance_outcomes=[])}
    assert _asyncio.run(final_instance_reward(state=empty)) == 0.0


def test_format_score_increases_with_partial_compliance():
    """Gibberish < {-only < schema-field-mention < parsed."""
    from pydantic import BaseModel
    from clbench_verifiers.env import _format_score

    class A(BaseModel):
        action: str
        amount: int = 0

    s_garbage = _format_score("FOLD now", A, parsed_ok=False)
    s_braces_only = _format_score("{ what }", A, parsed_ok=False)
    s_one_field_only = _format_score('"action" goes here', A, parsed_ok=False)
    s_two_fields_braces = _format_score('{ "action": x, "amount": y }', A, parsed_ok=False)
    s_parsed = _format_score('{"action":"FOLD"}', A, parsed_ok=True)
    assert s_garbage == 0.0
    assert 0 < s_braces_only          # 0.2
    assert 0 < s_one_field_only       # 0.15
    assert s_two_fields_braces > max(s_braces_only, s_one_field_only)  # combined
    assert s_parsed > s_two_fields_braces  # full parse beats partial
    assert s_parsed == 1.0


def test_env_records_best_format_score_per_rollout():
    """env_response should bump rs.best_format_score on every assistant msg."""
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
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    async def go():
        state = {"messages": []}
        await env.setup_state(state)
        # Looks-like-JSON gibberish — should produce a non-zero partial score.
        # exploitable_poker schema fields are {thinking, action, amount}; this
        # text has braces + the word "action" + the word "thinking" but isn't
        # valid JSON.
        state["messages"].append(
            {"role": "assistant", "content": "{ thinking ... action ... }"}
        )
        await env.env_response(state["messages"], state)
        rs = state["clbench"]
        return rs.best_format_score

    score = asyncio.run(go())
    assert 0.0 < score < 1.0


def test_input_token_budget_disabled_by_default_in_tests():
    """When the cap is set to 0, the @vf.stop must always return False."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=True,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=0,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    # Stand-in env doesn't expose get_state_usage; the @vf.stop should still
    # short-circuit on max_input_tokens_per_rollout == 0 without raising.
    async def go():
        return await env.input_token_budget_exceeded({"clbench": None})

    assert asyncio.run(go()) is False


def test_input_token_budget_fires_when_exceeded():
    """When the usage tracker reports input_tokens >= cap, return True."""
    cls = _build_local_env_class()

    class _Rubric:
        funcs: list = []
        weights: list = []

    env = cls(
        task_name="exploitable_poker",
        task_kwargs={"num_instances": 1, "seed": 0},
        max_instances_per_rollout=1,
        schema_hint_in_system=True,
        end_on_parse_failure=True,
        use_notepad=False,
        notepad_max_chars=4000,
        max_input_tokens_per_rollout=1000,
        enable_guided_json=False,
        clear_history_between_instances=False,
        rubric=_Rubric(),
        max_turns=8,
    )

    # Monkey-patch get_state_usage to simulate the framework's usage tracker.
    env.get_state_usage = lambda state: {"input_tokens": 1500, "output_tokens": 100}

    async def fires():
        return await env.input_token_budget_exceeded({})

    env.get_state_usage = lambda state: {"input_tokens": 500, "output_tokens": 100}

    async def passes():
        return await env.input_token_budget_exceeded({})

    assert asyncio.run(passes()) is False
    env.get_state_usage = lambda state: {"input_tokens": 1500, "output_tokens": 100}
    assert asyncio.run(fires()) is True


if __name__ == "__main__":
    test_parse_action_extracts_fenced_json()
    test_parse_action_handles_inline_json()
    test_parse_action_rejects_garbage()
    test_format_schema_hint_renders_pydantic_schema()
    test_env_drives_poker_task_smoke()
    test_env_handles_parse_failure()
    test_notepad_schema_augmentation()
    test_env_notepad_mode_persists_across_instances()
    test_env_notepad_truncates_oversized()
    test_input_token_budget_disabled_by_default_in_tests()
    test_input_token_budget_fires_when_exceeded()
    test_format_score_increases_with_partial_compliance()
    test_env_records_best_format_score_per_rollout()
    test_latest_assistant_text_reads_reasoning_content()
    test_guided_json_injected_into_sampling_args()
    test_guided_json_off_when_disabled()
    test_final_instance_reward_returns_last_outcome()
    test_dataset_seed_is_threaded_into_task_kwargs()
    test_observation_propagated_as_feedback_at_instance_boundary()
    test_legal_actions_are_extracted_from_situation_line()
    test_legal_actions_prefix_appears_in_user_message()
    print("All smoke tests passed.")
