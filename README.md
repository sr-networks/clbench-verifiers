# clbench-verifiers

Glue package that runs [Continual Learning Bench](https://github.com/pgasawa/continual-learning-bench)
tasks as [verifiers](https://github.com/willccbb/verifiers) `MultiTurnEnv`s,
so a small HuggingFace policy can be GRPO-trained on them via `vf.RLTrainer`
and the trained checkpoint can be evaluated back through `clbench run` with
official benchmark scoring.

The package is intentionally thin (a wrapper, a parser, a rubric, an eval
system, and two entry points). Both upstream repos move fast and have
different release cadences, so we keep the integration as a third-party shim
rather than forking either one.

## Status

- **Milestone 1 (current):** `exploitable_poker`, with optional `icl_notepad`-style memory.
- **Milestone 2:** `database_exploration` (similar shape, no Docker, higher memory payoff).
- **Milestone 3:** Docker-sandboxed tasks (`codebase_adaptation`, `sales_prediction`)
  via a remote sandbox (Modal/e2b), since nested Docker in Colab is not really viable.

## Architecture at a glance

```
                                     train.py (clbv-train)
                                          │
                                          ▼
       ┌────────────────────────── vf.RLTrainer (GRPO) ──────────────────────┐
       │                                                                     │
       │                       drives rollouts                               │
       │                                                                     │
       │   ┌──────────────────────────────────────────────────────────────┐  │
       │   │ CLBenchEnv  (vf.MultiTurnEnv)                                │  │
       │   │                                                              │  │
       │   │  setup_state ─► task.reset() ─► first user msg               │  │
       │   │  env_response(messages, state)                               │  │
       │   │     ├─ parse last assistant msg → pydantic action            │  │
       │   │     │   (with optional notepad_update field)                 │  │
       │   │     ├─ task.step(Response) → (Observation, next_query)       │  │
       │   │     ├─ accumulate InstanceOutcome.reward                     │  │
       │   │     └─ build next user msg (prepend notepad on inst boundary)│  │
       │   │  is_completed ─► done from task or budget hit                │  │
       │   │                                                              │  │
       │   │  Rubric: mean_instance_reward + parse_failure_penalty        │  │
       │   │          + diagnostics (instances, notepad updates, length)  │  │
       │   └──────────────────────────────────────────────────────────────┘  │
       │                                ▲                                    │
       │                                │ vLLM-batched G samples / prompt    │
       │                                │ (verifiers ↔ vLLM OpenAI server)   │
       └────────────────────────────────┴────────────────────────────────────┘
                                          │
                                          ▼
                                save LoRA / model checkpoint
                                          │
                                          ▼
                              eval.py (clbv-eval)
                                          │
       ┌──────────────────────────────────┴────────────────────────────────┐
       │   spawns vllm serve <ckpt>                                        │
       │   shells out to:                                                  │
       │       clbench run exploitable_poker --system vllm_local           │
       │           --system.base_url http://127.0.0.1:8000/v1              │
       │           --system.model trained-policy                           │
       │   (optionally --system.use_notepad true for notepad parity)       │
       └───────────────────────────────────────────────────────────────────┘
```

Two control loops, separated cleanly:
1. **Training** — verifiers' RL trainer drives, vLLM samples, the env wraps a
   fresh CLBench task per rollout. Reward is computed from the per-rollout
   state, not from the bench's run harness (we bypass that part for speed).
2. **Evaluation** — the official `clbench run` drives, our `vllm_local`
   System adapter sits in for the policy. This gives apples-to-apples scoring
   against the public leaderboard's metric definitions.

## Modules

| File | Purpose |
|---|---|
| `clbench_verifiers/env.py` | `CLBenchEnv` — wraps any `ContinualLearningTask` as a `vf.MultiTurnEnv`. One rollout = N CLBench instances (configurable), 1 by default. |
| `clbench_verifiers/parsing.py` | Tolerant JSON-from-text parser. Extracts the first balanced `{...}` (handles fences, prose, nested objects), then validates against the response schema. Parse failures are caught and penalized rather than crashed. |
| `clbench_verifiers/notepad.py` | icl_notepad-style schema augmentation. Adds an optional `notepad_update: str` to the response schema so the policy can write structured memory inline with its action. |
| `clbench_verifiers/rubric.py` | `vf.Rubric` with `mean_instance_reward` (the real reward) + `parse_failure_penalty` + diagnostics (`num_instances_completed`, `num_notepad_updates`, `notepad_length_chars`). |
| `clbench_verifiers/system.py` | `vllm_local` CLBench System adapter. Talks to a vLLM OpenAI-compatible server. Supports `use_notepad` for eval parity. |
| `clbench_verifiers/train.py` | `clbv-train` CLI. Loads a TOML config, builds env + model + GRPO trainer, runs N steps, saves a checkpoint. |
| `clbench_verifiers/eval.py` | `clbv-eval` CLI. Spawns vLLM, shells out to `clbench run`, tears down. |
| `configs/poker_qwen2_5_1_5b.toml` | Plain GRPO config (1 instance / rollout, no notepad). |
| `configs/poker_qwen2_5_1_5b_notepad.toml` | Notepad-mode config (4 instances / rollout, `use_notepad=true`). |
| `notebooks/train_poker.ipynb` | End-to-end Colab notebook. |
| `scripts/colab_setup.sh` | Idempotent Colab bootstrap. |
| `tests/test_env_smoke.py` | CPU-only smoke tests. Drives a real poker task end-to-end with a verifiers stand-in. |

## How the env wrapper works

`CLBenchEnv` extends `vf.MultiTurnEnv`, which gives us the standard verifiers
chat-format rollout loop. The mapping is:

| verifiers concept | CLBench equivalent |
|---|---|
| `setup_state(state)` initial messages | `task.reset() → Query` plus a system prompt embedding the JSON schema |
| `env_response(messages, state)` | Parse last assistant msg → `PokerAction` → `task.step(Response)` → format `Observation.content` + next `Query.prompt` as next user message |
| `is_completed(messages, state)` | Task's `done` flag or `instances_completed >= max_instances_per_rollout` |
| Rubric reward | Mean of accumulated `InstanceOutcome.reward` values |

Per-rollout state lives in `state["clbench"]` (a `CLBenchRolloutState`
dataclass) so the rubric and any verifiers introspection can read it.

## Notepad mode (icl_notepad-style memory)

Set `use_notepad = true` in your config (or `--system.use_notepad true` at
eval time) to enable persistent memory. The wrapper:

1. Runs `notepad.build_schema_with_notepad(task_schema)` to add an optional
   `notepad_update: Optional[str]` field to the response schema.
2. Updates the system prompt to mention the notepad and what it's for.
3. Each turn, if the parsed action has `notepad_update` set, **overwrites**
   `state["clbench"].notepad` with that string (truncated to
   `notepad_max_chars`).
4. At every **instance boundary** within the rollout, prepends the current
   notepad — wrapped in `=== YOUR NOTEPAD === ... ===` markers — to the
   first user message of the new instance.

Why a structured-output field rather than a separate "memory tool":
- No tool-call tokens needed — works with any vanilla chat model.
- The notepad write is part of the same completion that emits the action,
  so the GRPO advantage signal credits useful memory writes via the same
  reward that credits good actions.
- Matches the upstream `icl_notepad` system in CLBench, so eval-time
  comparisons are like-for-like.

The notepad only persists *within* a rollout, not across rollouts. This
respects GRPO's i.i.d. assumption between samples. The reward signal that
makes the notepad useful is the within-rollout improvement: in
`max_instances_per_rollout=4` mode, the policy that writes a useful note in
instance 1 wins higher reward in instances 2–4, raising the mean reward of
that rollout, raising its group-relative advantage.

> **Practical note:** notepad mode only makes sense with
> `max_instances_per_rollout >= 2`. The wrapper warns you if you set
> `use_notepad=True` with a single-instance rollout.

## Quickstart on Colab

Spin up a Colab with an A100 (or H100 if available), then in a single cell:

```bash
!nvidia-smi
%cd /content
!git clone https://github.com/sr-networks/clbench-verifiers.git
%cd clbench-verifiers
!bash scripts/colab_setup.sh

# Train (plain GRPO, ~1 hr on A100):
!clbv-train --config configs/poker_qwen2_5_1_5b.toml

# Or, train with notepad memory (~2× longer per step due to multi-instance rollouts):
!clbv-train --config configs/poker_qwen2_5_1_5b_notepad.toml

# Evaluate the trained checkpoint via the official bench:
!clbv-eval --checkpoint ./checkpoints/poker_qwen2_5_1_5b/final \
    --task exploitable_poker --schedule quick_test
```

The `notebooks/train_poker.ipynb` notebook does the same in step-by-step form
with a CPU smoke-test cell at the start so you can sanity-check the env
without burning GPU time.

## Local development

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .                        # core; pulls clbench from git
uv pip install -e ".[poker,dev]"           # add poker extra + dev tools
python tests/test_env_smoke.py             # CPU-only smoke tests
```

You don't need verifiers, vLLM, or torch on your laptop — those are in the
`[train]` extra and only required at training/eval time.

## Configuration reference

All training knobs live in `clbench_verifiers.train.TrainConfig` (a dataclass
loaded from TOML). The fields you'll touch most often:

| Field | Default | Notes |
|---|---|---|
| `task_name` | `exploitable_poker` | Any registered CLBench task. |
| `task_kwargs` | `{}` | Forwarded to the task constructor. |
| `max_instances_per_rollout` | `1` | Set ≥ 2 to enable continual mode (required for notepad). |
| `use_notepad` | `false` | Adds `notepad_update` field; persists notepad across instances within a rollout. |
| `notepad_max_chars` | `4000` | Soft cap; head-truncated when exceeded. |
| `parse_failure_penalty` | `-1.0` | Per-failure reward delta. |
| `model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Any HF chat model. |
| `use_lora` | `true` | LoRA reduces memory ~5×. Disable for full FT. |
| `rollouts_per_step` | `16` | GRPO group size G. |
| `num_train_steps` | `200` | Total GRPO steps. |
| `learning_rate` | `1e-6` | Conservative for 1.5B; safe to push to `2e-6` once stable. |
| `vllm_gpu_memory_utilization` | `0.45` | Drop to `0.3` on T4-16GB; `0.7` on H100. |

## Caveats

1. **`vf.RLTrainer` API.** This package was written against `verifiers`
   `>=0.1.7`, where `RLTrainer` (the "nano" trainer) replaces the older
   `GRPOTrainer`. The trainer-construction line in `train.py` is the only
   place exposed to upstream API drift; if it breaks after a verifiers
   bump, the fix is typically a one- or two-arg rename in
   `_build_grpo_config`.
2. **Python 3.13.** CLBench pins `requires-python>=3.13` in its pyproject.
   Colab is typically on 3.10/3.11, so `colab_setup.sh` installs CLBench with
   `--ignore-requires-python`. The actual code runs fine on 3.10+; the
   version pin is more aspirational than load-bearing.
3. **No constrained decoding.** We rely on prompt + tolerant parse rather
   than vLLM's `guided_json`. Trade-off: simpler, gives the policy a
   learning signal toward valid output, slightly noisier early in training.
   Adding constrained decoding is a 20-line change in `train.py` if you
   want it.
4. **One reward per trajectory.** Multi-turn credit assignment is flat —
   the trajectory's mean instance reward is assigned to every emitted
   token. Fine for short rollouts (poker hands are ~5 turns each) but
   blurs gradient on longer ones (codebase_adaptation, 20+ steps). Fix is
   GAE-per-turn or shaped step rewards from `Observation.content`, both
   non-trivial — out of scope for milestone 1.
5. **Notepad doesn't persist across rollouts** (by design — see "Notepad
   mode" above). If you want long-term persistent memory across the whole
   training run, that's a different system (mem0-style retrieval). Out of
   scope here.

## Roadmap

- [ ] Run `poker_qwen2_5_1_5b.toml` end-to-end on Colab A100, confirm the
      trained checkpoint beats CLBench's `reset_each_instance` baseline
      (current `gpt-5.4` baseline ≈ 1.295).
- [ ] Run `poker_qwen2_5_1_5b_notepad.toml` and compare against the
      no-notepad checkpoint at equal compute.
- [ ] Wrap `database_exploration` (no Docker, similar action shape).
- [ ] Constrained-decoding option in `train.py` for faster early-training
      convergence.
- [ ] Remote-sandbox option (Modal) for `codebase_adaptation` /
      `sales_prediction`.

## License

Apache-2.0.
