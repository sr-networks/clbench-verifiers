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
    extra_funcs: Optional[list[RewardFn]] = None,
):
    """
    Build a verifiers ``Rubric`` for CLBench.

    Reward components (weights in parentheses):
      - ``mean_instance_reward`` (1.0) — the real task reward.
      - ``parse_failure_penalty`` (1.0) — weighted -penalty × #parse failures.
      - ``best_format_score`` (``format_shaping_weight``, default 0.1) — small
        positive shaping: a 0..1 score for "did the output look like the
        schema?". Lets cold-start GRPO have advantage variance on groups
        where no rollout fully parses. Set the weight to 0 once your
        policy emits valid JSON reliably.
      - Diagnostics (weight 0): ``num_instances_completed``,
        ``num_notepad_updates``, ``notepad_length_chars``.

    The Rubric API is imported lazily so this module can be used in tests
    without verifiers installed (the rubric will only be needed at training
    time anyway).
    """
    try:
        import verifiers as vf  # type: ignore
    except ImportError:  # pragma: no cover
        return _MockRubric(parse_failure_penalty=parse_failure_penalty)

    funcs: list[RewardFn] = [
        mean_instance_reward,
        make_parse_failure_penalty(parse_failure_penalty),
        best_format_score,          # cold-start shaping (small weight)
        num_instances_completed,    # diagnostic; weight 0
        num_notepad_updates,        # diagnostic; weight 0
        notepad_length_chars,       # diagnostic; weight 0
    ]
    if extra_funcs:
        funcs.extend(extra_funcs)

    weights = [
        1.0,                         # mean_instance_reward
        1.0,                         # parse_failure_penalty
        format_shaping_weight,       # best_format_score
        0.0, 0.0, 0.0,              # diagnostics
    ] + [1.0] * len(extra_funcs or [])
    return vf.Rubric(funcs=funcs, weights=weights)


class _MockRubric:  # pragma: no cover - only used when verifiers absent
    """No-op stand-in so this module can be imported without verifiers."""

    def __init__(self, parse_failure_penalty: float):
        self.parse_failure_penalty = parse_failure_penalty
        self.funcs = []
        self.weights = []
