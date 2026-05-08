"""
Rewards for GRPO training: read accumulated CLBench instance outcomes from the
rollout state and emit scalar rewards.

Two reward components are wired up by default:

- ``mean_instance_reward``: arithmetic mean of ``InstanceOutcome.reward`` values
  collected during the rollout. This is the "task" reward — what CLBench
  actually scores systems on.
- ``parse_failure_penalty``: a small negative bonus per turn the model emitted
  unparseable output. Kicks in even on rollouts that complete successfully so
  the policy converges toward valid JSON quickly.

Add custom shaping (turn count, latency, memory-token usage, etc.) by passing
``extra_funcs`` to ``build_clbench_rubric``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

# Reward functions in verifiers can be sync or async; we use sync-then-await
# wrappers to keep the surface uniform.

RewardFn = Callable[..., Awaitable[float]]


def _get_state(state: Any) -> Any:
    """Find the CLBench rollout-state object under various verifiers signatures."""
    if state is None:
        return None
    if hasattr(state, "get") and "clbench" in state:
        return state["clbench"]
    return getattr(state, "clbench", None)


async def mean_instance_reward(*, state=None, **_kwargs) -> float:
    """Mean reward across all completed CLBench instances in this rollout."""
    rs = _get_state(state)
    if rs is None or not rs.instance_outcomes:
        return 0.0
    rewards = [float(o.reward) for o in rs.instance_outcomes if o.reward is not None]
    return sum(rewards) / len(rewards) if rewards else 0.0


async def final_instance_reward(*, state=None, **_kwargs) -> float:
    """Reward of the *last* completed instance in this rollout.

    Useful for memory-augmented training where the policy sets up earlier
    instances primarily to make the final instance succeed (the notepad /
    history-wipe configuration). Weighting this > 0 gives every token in
    the rollout (including notepad-write tokens in early instances) credit
    proportional to the final outcome.

    Returns 0.0 if no instances completed.
    """
    rs = _get_state(state)
    if rs is None or not rs.instance_outcomes:
        return 0.0
    last = rs.instance_outcomes[-1]
    return float(last.reward) if last.reward is not None else 0.0


def make_parse_failure_penalty(penalty_per_failure: float) -> RewardFn:
    """Factory: scalar penalty applied per parse failure observed in the rollout."""

    async def parse_failure_penalty(*, state=None, **_kwargs) -> float:
        rs = _get_state(state)
        if rs is None:
            return 0.0
        return penalty_per_failure * rs.parse_failures

    parse_failure_penalty.__name__ = "parse_failure_penalty"
    return parse_failure_penalty


async def num_instances_completed(*, state=None, **_kwargs) -> float:
    """Diagnostic — not weighted by default; useful for logging."""
    rs = _get_state(state)
    return float(rs.instances_completed) if rs else 0.0


async def num_notepad_updates(*, state=None, **_kwargs) -> float:
    """Diagnostic — counts how often the model wrote to the notepad."""
    rs = _get_state(state)
    return float(getattr(rs, "notepad_updates", 0)) if rs else 0.0


async def notepad_length_chars(*, state=None, **_kwargs) -> float:
    """Diagnostic — final notepad length in characters."""
    rs = _get_state(state)
    return float(len(getattr(rs, "notepad", ""))) if rs else 0.0


async def query_efficiency_reward(*, state=None, **_kwargs) -> float:
    """
    Variance-injecting shaping for tasks where CLBench's per-instance reward
    is 0/1 and the cold-start policy bottoms out at 0.

    Returns the mean over instances of ``(max_queries - num_queries) /
    max_queries``: 1.0 when the rollout submitted answers without exploring
    (probably wrong but quick), 0.0 when the model used the full query
    budget on every instance, linear in between. ``database_exploration``
    populates ``InstanceOutcome.metadata.num_queries``; tasks that don't
    expose this metadata produce 0.0 (no contribution).

    Why this is needed: CLBench's database_exploration reward is
    ``1 − num_queries / max_queries`` for correct answers and ``0`` for
    wrong answers. Cold-start: every rollout answers everything wrong,
    every reward is 0, every group has zero advantage, GRPO has no
    gradient. This component varies with how many queries each rollout
    used — different exploration strategies → different rollout rewards
    → non-zero advantages — so gradient flows even before the policy
    starts answering correctly.

    Weighted at 0.3 by default in ``build_clbench_rubric`` (small enough
    to not dominate once correct answers start appearing; large enough
    to push the cold-start group apart).
    """
    rs = _get_state(state)
    if rs is None or not rs.instance_outcomes:
        return 0.0
    contributions: list[float] = []
    for outcome in rs.instance_outcomes:
        meta = getattr(outcome, "metadata", None) or {}
        nq = meta.get("num_queries")
        if nq is None:
            continue
        # Be defensive about the budget — some tasks may expose a
        # per-question budget; fall back to a sane default.
        max_q = meta.get("max_queries_per_question") or 15
        if max_q <= 0:
            continue
        contributions.append(max(0.0, (float(max_q) - float(nq)) / float(max_q)))
    if not contributions:
        return 0.0
    return sum(contributions) / len(contributions)


async def best_format_score(*, state=None, **_kwargs) -> float:
    """
    Best per-turn partial-format score (0..1). Used at small positive weight
    so cold-start GRPO groups in which no rollout fully parses still get
    advantage variance from "less bad" outputs vs "more bad" outputs.

    Set the weight to 0 once your policy reliably produces valid JSON; this
    component should be a tiny perturbation by then anyway.
    """
    rs = _get_state(state)
    return float(getattr(rs, "best_format_score", 0.0)) if rs else 0.0


def build_clbench_rubric(
    *,
    parse_failure_penalty: float = -1.0,
    format_shaping_weight: float = 0.1,
    query_efficiency_weight: float = 0.3,
    mean_instance_reward_weight: float = 1.0,
    final_instance_reward_weight: float = 0.0,
    extra_funcs: Optional[list[RewardFn]] = None,
):
    """
    Build a verifiers ``Rubric`` for CLBench.

    Reward components (weights in parentheses):
      - ``mean_instance_reward`` (``mean_instance_reward_weight``, default 1.0) —
        mean per-instance reward across all completed instances.
      - ``final_instance_reward`` (``final_instance_reward_weight``, default 0.0) —
        only the final instance's reward. Use this for memory-augmented
        training where early instances primarily exist to set up the last
        one (notepad mode). Set ``mean_instance_reward_weight=0`` and
        ``final_instance_reward_weight=1`` for pure last-instance credit.
      - ``parse_failure_penalty`` (1.0) — weighted -penalty × #parse failures.
      - ``best_format_score`` (``format_shaping_weight``, default 0.1) — small
        positive shaping for cold-start.
      - Diagnostics (weight 0): ``num_instances_completed``,
        ``num_notepad_updates``, ``notepad_length_chars``.

    The Rubric API is imported lazily so this module can be used in tests
    without verifiers installed.
    """
    try:
        import verifiers as vf  # type: ignore
    except ImportError:  # pragma: no cover
        return _MockRubric(parse_failure_penalty=parse_failure_penalty)

    funcs: list[RewardFn] = [
        mean_instance_reward,
        final_instance_reward,
        make_parse_failure_penalty(parse_failure_penalty),
        best_format_score,
        query_efficiency_reward,
        num_instances_completed,
        num_notepad_updates,
        notepad_length_chars,
    ]
    if extra_funcs:
        funcs.extend(extra_funcs)

    weights = [
        mean_instance_reward_weight,
        final_instance_reward_weight,
        1.0,                         # parse_failure_penalty
        format_shaping_weight,
        query_efficiency_weight,     # query_efficiency_reward
        0.0, 0.0, 0.0,              # diagnostics
    ] + [1.0] * len(extra_funcs or [])
    return vf.Rubric(funcs=funcs, weights=weights)


class _MockRubric:  # pragma: no cover - only used when verifiers absent
    """No-op stand-in so this module can be imported without verifiers."""

    def __init__(self, parse_failure_penalty: float):
        self.parse_failure_penalty = parse_failure_penalty
        self.funcs = []
        self.weights = []
