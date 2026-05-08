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

## What we're actually trying to do (plain English)

When an AI plays a series of related rounds — say, several poker hands
against the same opponent — a smart agent should *remember* what it
learned earlier and use it later. Most language models don't do this on
their own: each round looks fresh to them. A common workaround is to
give the model a "notepad": a small text buffer it can write to during
one round and read from on the next.

Our question is: **can we train a small language model, using reinforcement
learning, to actually *use* a notepad well?** That is, can it learn
*on its own* — without us hand-coding the rules — to jot down useful
observations about its opponent in early rounds and exploit those
observations in later rounds?

The setup: a 2-billion-parameter open-source model (Qwen3.5-2B) plays
poker against a deliberately exploitable opponent (the "calling
station," who calls every bet). A model that spots the exploit early
and writes it down should beat a model that has no memory at all. We
test that by training the same model two ways — with and without the
notepad — and comparing them on a held-out evaluation.

The harder version of the question is: even if the model with a
notepad ends up winning, **is it winning *because* of the notepad, or
just because the extra training made it generically better at poker?**
A lot of the engineering effort in this repo is about answering that
second question cleanly.

## Current state of experiments (2026-05-08)

**Headline so far on `exploitable_poker`:** the pipeline works, the
schema-fix bug is shipped, the policy trains. But at this scale (Qwen3.5-2B,
100-300 GRPO steps, 2-hand rollouts, calling-station opponent) the
*memory contribution* at eval time is statistically indistinguishable
from zero. We're pivoting to a task where one-shot observations
(database schema discovery) plug in much more directly — see the
"Pivot to `database_exploration`" section below.

**Earlier headline (v0.1.14, kept here for context):** the notepad
memory channel was silently broken — `notepad_update` was nominally
required by the TOML schema but the env's own injection clobbered the
constraint, so vLLM let the model omit the field. Fixed in v0.1.16.

| Run | Env ver | Config | Final reward Δ (start → end) | Mean reward Δ | Note |
|---|---|---|---|---|---|
| `z3f9leb8…` notepad-on | 0.1.14 | 2-hand, history wipe, final-instance reward | −0.74 → −0.78 (flat) | −1.54 → −0.66 (+0.88) | Memory channel inactive — `notepad_update` field omitted on most turns |
| `ond5k5m2…` no-memory | 0.1.14 | same minus notepad | −0.91 → −0.53 (**+0.38**) | −2.56 → −0.56 (+2.00) | Improves more — no schema tax, all gradient on poker play |
| `wvajchd9…` notepad-on | 0.1.15 | render fix only | (training math identical to v0.1.14) | — | Dashboard fix; same memory-channel bug |
| `qa93yufj…` notepad-on | **0.1.16** | schema enforcement fix | ⏳ in progress | ⏳ | First run with the memory channel actually working |

### What's fixed at v0.1.16

1. **Pydantic-derived schema `required` list now includes `notepad_update`.**
   Pydantic's `Optional[str] = None` produced a JSON schema where the
   field was in `properties` but not `required`. Post-process the dict
   after `model_json_schema()` to add it. `anyOf=[string, null]` still
   permits null mid-instance; only completely-absent gets rejected by
   vLLM's grammar. (`clbench_verifiers/env.py::_apply_constraint`)
2. **Env-side guided_json injection no longer clobbers TOML overrides.**
   Skip injection when `state["sampling_args"]["extra_body"]["guided_json"]`
   already exists. The TOML version (with `maxLength` caps and explicit
   `required` lists) wins.
3. **Dashboard rendering shows all hands chronologically** (v0.1.15).
   With `clear_history_between_instances=True`, verifiers' default
   `render_completion` only saw the last instance's tail. Override it
   to walk every trajectory step and append new content in order. Pure
   cosmetic — doesn't change reward, advantage, or training tokens.

### Eval harness ready

`scripts/eval_remote.py` runs CLBench's official `run_benchmark` against
any OpenAI-compatible endpoint (Prime deployments, vLLM serve, anywhere).
Outputs `mean_reward`, `baseline_mean` (stateless reset), and
`mean_gain` (stateful − stateless = "did memory help?").

`VLLMClientSystem` extended with two flags so eval matches training
distribution:
- `clear_context_between_instances=True` — mirrors the training-time
  history wipe
- `enable_guided_json=True` — sends the same `notepad_update`-required
  schema vLLM enforced during sampling

```bash
prime deployments create <checkpoint_id>          # → endpoint URL
uv run --python 3.12 python scripts/eval_remote.py \
  --base-url <url> --model <name> --api-key-env PRIME_API_KEY \
  --use-notepad --enable-guided-json \
  --clear-context-between-instances \
  --num-instances 10 --runs 5 \
  --task-arg opponent_policy=calling_station
```

### Training results

Final-hand reward per rollout, averaged over the last 10 training
steps. The two `v0.1.14` rows are kept for context (broken memory
channel + matched control); the v0.1.16 rows are the runs with a
working memory channel:

| Run | Env ver | Start (steps 0-9) | End (last 10) | Δ |
|---|---|---|---|---|
| **ON memory, 300 steps (v0.1.16)** | 0.1.16 | −1.17 chips/hand | (smoothed) **−0.21** | **see plot** |
| **ON memory, 100 steps (v0.1.16)** | 0.1.16 | −1.17 | **+0.28** | **+1.45** |
| OFF memory, 100 steps (v0.1.14, control) | 0.1.14 | −0.91 | −0.53 | +0.38 |
| ON memory, 100 steps (v0.1.14, broken schema) | 0.1.14 | −0.74 | −0.78 | flat |

The broken-schema ON run ended slightly worse than where it started
(paid the schema tax with no working memory channel). After the
v0.1.16 schema fix, the policy trends upward; the 300-step extension
lowers per-rollout variance but does not push the absolute reward
materially below the 100-step end-state. Plot: `docs/training_curve.png`.

### Held-out eval (CLBench `run_benchmark` via `scripts/eval_remote.py`)

ON-memory checkpoints deployed on Prime Inference, evaluated against
the `calling_station` opponent at **20 task seeds**, with
`num_instances=2` (matches the training distribution) and `runs=5`
(5 permutation orderings per seed). Stateful = notepad active across
instances; stateless baseline = system reset between instances.
`mean_gain = stateful − stateless` is "did the notepad help at
inference?".

| Checkpoint | Stateful mean | Stateless baseline | **`mean_gain`** | SE | 95% CI |
|---|---|---|---|---|---|
| **300-step v0.1.16** | −0.54 | −0.26 | **−0.28** | **0.27** | **[−0.82, +0.27]** |
| 100-step v0.1.16 | −0.54 | −0.75 | +0.21 | 0.45 | [−0.70, +1.11] |

Both straddle zero. **Neither distinguishes memory-helps from
memory-doesn't-help.** With 3× more training the SE tightened ~40%
(0.45 → 0.27) but the point estimate flipped sign in the noise. Per-seed
variance dominates — a single all-in pot can swing a 12-hand mean by
±5 chips. The 300-step policy plays *much more consistently* (per-seed
std dropped from 1.25 to 0.44) but at a slightly lower absolute reward
— it converges to a conservative "fold often, raise small" strategy
that misses the upside the noisier baseline occasionally catches.

### What we can claim, what we can't

- ✅ **Schema-fix bug found and shipped (v0.1.16)**: ``notepad_update``
  is now actually required by vLLM's grammar; notepad-write count
  per rollout climbed from 4.5 (broken) to 6.5 (working).
- ✅ **Pipeline + cost story is solid**: end-to-end training on Prime
  at ≈$3.40 per 100 steps; eval harness (`scripts/eval_remote.py`)
  produces clean `mean_gain` numbers; reproducible across seeds.
- ✅ **Notepad-flavoured training produces a more consistent
  policy**: 300-step stateful std fell to 0.44 chips/hand (vs 1.25
  at 100 steps).
- ⚠️ **Cannot claim memory helps on this task at this scale**:
  100-step gain `+0.21 ± 0.45`, 300-step gain `−0.28 ± 0.27`.
  Both inside noise; tighter SE didn't lift the signal.
- ❌ **Cannot claim memory hurts**: same noise band; the 300-step
  point estimate is below zero but the 95% CI is `[−0.82, +0.27]`.
- 🔍 **Diagnosed why**: against a deterministic exploitable opponent,
  the within-rollout memory signal needs *many* hands to extract a
  pattern. Two hands gives at most one observation, which doesn't
  translate into a hand-2 EV improvement large enough to surface
  above ±5-chip per-hand variance.

### Pivot to `database_exploration`

The diagnosis above motivates switching to a CLBench task where
**one observation in instance 1 directly transforms instance 2+**.
`database_exploration` is exactly that:

- Sequential natural-language questions about a SQLite database
  the agent doesn't know.
- Per question, the agent can issue SQL `QUERY` actions to explore,
  then submit `ANSWER`.
- Primary CLBench metric: **reduction in exploratory queries over
  time** as the agent learns the schema.
- The "right" memory move: instance 1 issues `PRAGMA table_info(...)`,
  writes "tables: products(id, name, price, category, stock); price
  and stock are floats" to the notepad. Instance 2+ skips the
  exploration step.
- Bonus: a schema-drift variant (`products.db → products_drifted.db`
  mid-sequence) tests whether the agent *updates* its notes vs.
  trusting them blindly.

Why this should give a measurable signal where poker doesn't:

| | Poker (calling station) | `database_exploration` |
|---|---|---|
| Useful instance-1 signal | Statistical pattern across hands | Schema is fixed, fully discoverable in 1-2 queries |
| Translates to instance-2 reward | Marginal (small EV per hand) | Huge (skip 4 exploratory queries, answer directly) |
| Per-instance reward variance | ±5+ chips (deck shuffles) | ±~0.1 (correct/incorrect score) |

### Next

Within the remaining ~$8 budget, ranked:

1. **Smoke-test `database_exploration` with our env wrapper** (~free):
   confirm `CLBenchEnv` handles the new task's schema (`DatabaseAction`
   has `action` + `content` fields) and that the rollout completes.
   Will require gating poker-specific bits in `env.py`
   (`_extract_legal_actions`, system-prompt language) on
   `task_name == "exploitable_poker"`.
2. **Push env v0.1.17** with the gated poker-specific bits (and any
   small fixes from the smoke).
3. **100-step pilot training run** on `database_exploration` with
   notepad-on, 2 instances per rollout (~$3-4).
4. **20-seed eval** on the resulting checkpoint (~$2). Headline:
   does `mean_gain` clear zero on a task where one-shot memory
   should actually pay?
5. **(stretch) Add the differential reward** (`last_hand − mean(prev)`)
   you suggested earlier as a knob in `rubric.py`. Out of budget for
   this sprint but trivial to land for the next one.
6. **(stretch) Schema-drift variant** to test whether the trained
   policy *updates* its notepad when info goes stale.

### Earlier candidate next steps (kept for record, no longer plan-A)

- Longer training at v0.1.16 settings on poker. 300 steps tightened
  the eval CI but didn't move the signal — likely a task-fit issue,
  not an under-training issue.
- Larger eval N (60-80 task seeds): would tighten poker's `mean_gain`
  CI further but doesn't change the underlying.
- Harder poker opponent: `calling_station` is so exploitable a
  constant strategy already does well. Switching to
  `tight_aggressive` or a mixed schedule would punish a
  forgetful policy more, raising the value of working memory.

## Historical status (2026-05-06, evening)

| Component | State | Notes |
|---|---|---|
| `exploitable_poker` env wrapper | ✅ working end-to-end | `setup_state` / `env_response` / `@vf.stop` aligned with verifiers `0.1.12`; reads `content` AND `reasoning_content` so thinking models like Qwen3.5 are handled correctly |
| Cost caps (`max_input_tokens_per_rollout`, lower `max_turns`) | ✅ landed | ~80× cost reduction vs uncapped run |
| Partial-format reward shaping | ✅ landed | weight 0.1; breaks zero-advantage at cold-start |
| Constrained decoding (`guided_json`) | ✅ landed via `[sampling.extra_body.guided_json]` (v0.1.8) | parse_failure_penalty drops to 0; +0.56 chip/hand vs the broken-extraction baseline |
| Per-instance history wipe (`clear_history_between_instances`) | ✅ landed (v0.1.9) | overrides `get_prompt_messages` so notepad mode actually requires the notepad — see "memory experiments" below |
| Last-instance reward weighting (`final_instance_reward_weight`) | ✅ landed (v0.1.9) | Lets us train pure last-hand objective so memory writes get clean credit |
| Local CPU smoke tests | ✅ all 18 passing | parser, notepad, format-score, guided_json, final-instance reward, reasoning_content |
| Prime env push | ✅ `sr-networks/clbench-poker@0.1.9` PUBLIC | CI green; live at <https://app.primeintellect.ai/dashboard/environments/sr-networks/clbench-poker> |
| Prime full plain GRPO (200 steps, single hand/rollout) | ✅ completed | `g0g0j6hkxuoct7ipyd8faaau`: $4.59, mean reward -1.27 (vs -1.83 in pre-guided run); +0.56/hand uplift over the broken-extraction run, **but no within-run learning trend** (slope -0.0025/step) |
| Prime ICL baseline (200 steps, 4 hands/rollout, no notepad) | ⏳ in flight | `ktr3ksy46hxljeicgboqrnq4`; tests if multi-hand context alone is useful before we add notepad on top |
| Prime notepad GRPO (4 hands/rollout, history wiped, last-instance reward) | ⏳ queued after ICL baseline lands | notepad-config v0.1.9 |
| Local Colab GRPO (`clbv-train`) | ⚠️ scaffolded, not yet run end-to-end | first real Colab run still pending |
| `database_exploration` env | ⏳ not started | low-effort follow-up; same wrapper shape |
| Docker-sandboxed tasks | ⏳ not started | needs Modal/e2b sandbox for `codebase_adaptation` + `sales_prediction` |

### What we're actually training (and what CLBench measures)

CLBench scores systems on two things: **aggregate reward** across a
sequence of N task instances (e.g., 120 poker hands), and **continual
gain** — `aggregate_reward(stateful) − aggregate_reward(reset_each_instance)`,
i.e. how much memory across instances improved performance.

The crucial piece is that the *continual gain* is what makes the
benchmark non-trivial. For our task pick (`exploitable_poker` against the
`calling_station` opponent) there is a learnable opponent-pattern that a
memory-augmented system should exploit better than a no-memory one.

Our **plain config** (`prime_qwen3_5_2b.toml`) trains with one hand per
rollout and a fixed seed, so all 1,600 training hands are basically the
same scenario. That:
- proves the pipeline (env wrapper, reward shape, cost caps, guided_json),
- delivers a measurable single-hand uplift,
- but does **not** engage with the continual-learning question at all —
  the policy never sees more than one hand at a time.

The **ICL baseline** (`prime_qwen3_5_2b_icl.toml`, currently running)
moves to 4 hands per rollout with full conversation history kept across
hands. That's the upstream `icl` system in CLBench's leaderboard. It
tells us whether GRPO+ICL alone (no curated memory, just raw history)
already beats single-hand training.

The **notepad config** (`prime_qwen3_5_2b_notepad.toml`, queued) is the
memory-augmented experiment: 4 hands per rollout, `clear_history_between_instances=true`
so within-hand history is wiped at each boundary and only the notepad
survives, plus `final_instance_reward_weight=1.0` so all earlier-hand
tokens (including notepad writes) get their advantage from the **last
hand's** outcome.

### Memory experiments — design notes

Two non-obvious things came up while wiring this up:

1. **Verifiers' default rollout keeps full history within a multi-instance
   rollout.** If we just turned on `use_notepad=true` without history-wiping,
   the model would see *both* the notepad *and* the raw conversation from
   prior hands, making the notepad redundant. The new `clear_history_between_instances`
   flag overrides `get_prompt_messages` to wipe the within-rollout history at
   each instance boundary; this matches CLBench's upstream `icl_notepad`
   semantics (only the notepad survives). The flag defaults to `False`, so
   plain ICL multi-instance runs work too.
2. **GRPO advantage is per-rollout, not per-turn.** With 4 hands per rollout
   and `final_instance_reward_weight=1.0`, every action token in instances
   1–3 (including the `notepad_update` writes) gets credit/blame proportional
   to instance 4's chip outcome. This is the cleanest credit-assignment for
   "did your earlier-hand notes help the last hand?". The default
   (`mean_instance_reward_weight=1.0`, `final_instance_reward_weight=0.0`)
   preserves the old behavior of averaging across all instances.

### Cost is controlled. Learning is the next frontier.

Cost control was the explicit goal of the first push, and it's solved:
the original uncapped 2-step smoke burned **$3.26** on 30+-turn gibberish
loops; the current-config 2-step smoke costs **$0.04** and runs to
completion with rollouts that actually finish poker hands. A 200-step
plain run lands at **$4.59** — predictable and capped.

But the 200-step plain GRPO run shows **no within-run learning trend**:
mean reward -1.27 (peak window -1.06 around step 75-99, late drift to
-1.42 by step 199). Compared to the same config without guided_json
(-1.83 across all windows), guided_json delivered a clean +0.56 chip/hand
absolute uplift, but the gradient isn't pulling the policy reliably
upward over training. Likely structural causes:

1. **Single-scenario training distribution.** Same fixed seed every
   rollout = same opening hand replayed 1,600 times. The policy is
   over-specifying on one hand instead of learning poker.
2. **Reward variance dominates the gradient at LR=1e-6 on LoRA.** Real
   poker hands swing -10 to +10 chips; a 0.3-chip EV improvement is
   buried in noise. Either bigger groups (16+ rollouts), lower LR, or
   higher SNR shaping is needed.

Both motivate the multi-instance experiments (broader distribution,
lower-variance per-rollout-mean) and ultimately the notepad config
(direct credit on the final-hand outcome).

### Cost-control journey

| Run | Config delta | Outcome | Cost |
|---|---|---|---|
| `w4o10b2y9cjav8dj1b6yhx1h` | original (no caps) | COMPLETED but every rollout was 30+ turns of gibberish | **$3.26** |
| `ceizjz5hxeamdvlwke4kd0km` | + `max_turns=8`, `max_input_tokens_per_rollout=4000`, `end_on_parse_failure=true` | FAILED — `zero_advantage=2/2` every group ⇒ orchestrator crash | $0.005 |
| `r9zjmjt6ci90y4jtsm8c5qsf` | + partial-format reward shaping (0..1, weight 0.1) | FAILED — same crash, 2 rollouts/group still too few | $0.006 |
| `u46xq9y4fzp2mb93ngoyc9sx` | + `rollouts_per_example=4`, `batch_size=4` | step 0 ✓, **step 1 zero-advantage crash** (post-update output collapse) | $0.01 |
| `lmqvmi9ah7m6yhxgca98wbql` | + `end_on_parse_failure=false` (rely on token cap, not turn-1 abort) | ✅ both steps completed but `best_format_score=0`! | $0.04 |
| **`akfq9ddm5bn59dgae172ek5d`** | + env reads `content` *and* `reasoning_content` (v0.1.5) + `enable_thinking=false` | ✅ instances actually completing, real reward signal | **$0.05** |
| `bhj1q8wm3aaxs3t4brdipaaw` | full 200-step run on the broken-extraction config (pre-v0.1.5) | reward stuck at -5.0 floor for 51 steps; stopped early | $1.58 |
| `p8hx0n3n5gxu0dbjs5s7f4nc` | full 200-step run on v0.1.5 + `enable_thinking=false` | runs cleanly 200/200; **no learning trend** (rewards in [-3, -1] throughout) | $5.59 |
| `kwklvvrxoisl8bh4cbvb8pp5` | env-side `state["sampling_args"]["extra_body"]["guided_json"]` injection (v0.1.6) | constraint never reached vLLM; parse failures still ~3/turn | $0.06 |
| `hlrfpn58m3secgdzlm61bld7` | `get_model_response` override forcing constraint per-call (v0.1.7) | broke Prime's TITO client path; orchestrator hung 11 min before crashing | $0.00 |
| **`zco930bwwpqgjd2cm3hxj80v`** | guided_json via `[sampling.extra_body.guided_json]` in TOML (v0.1.8) | ✅ **parse_failure_penalty=0, instance completion 100%, step 1 reward +0.10** | **$0.04** |
| `g0g0j6hkxuoct7ipyd8faaau` | **full 200-step plain GRPO on v0.1.8 (guided_json)** | ✅ COMPLETED 200/200; **mean reward -1.27** vs -1.83 in pre-guided run (+0.56/hand uplift); peak window -1.06 (steps 75-99); late drift to -1.42; **slope = -0.0025/step (no within-run trend)** | **$4.59** |
| `ktr3ksy46hxljeicgboqrnq4` | ICL baseline: 4 hands/rollout, no notepad, history kept (v0.1.9) | ⏳ in flight — first multi-instance run | TBD |

Lessons baked in:

1. The right cold-start config is "tight token budget + generous turn
   budget + format-shaping reward + group size ≥ 4". Aborting on parse
   failure breaks GRPO's variance assumption.
2. Thinking models put their output in `reasoning_content`, not `content`.
   A wrapper that reads only `content` will see empty strings and report
   100% parse failure even when the model is producing valid output —
   that was the entire signal-loss in the first 50-step run. The env now
   reads both fields.
3. A green pipeline is not a learning pipeline. Watch reward dynamics,
   not just cost.
4. Constrained decoding (`guided_json`) is the right structural fix for
   small-model JSON tool-use. With Prime, the load-bearing surface is
   the TOML's `[sampling.extra_body.guided_json]` block — env-side
   `state["sampling_args"]` injection and runtime `get_model_response`
   overrides do *not* flow through Prime's TITO client path.
5. Multi-instance memory experiments require **explicit history wiping**
   in the env wrapper (verifiers concatenates by default). Without it,
   the model has full prior-hand context and the notepad is decorative.
6. For pure memory-augmented training, **last-instance reward weighting**
   gives cleaner credit assignment than mean-instance: notepad-write
   tokens in early hands get blamed/credited by the final-hand outcome,
   not diluted by the noise of their own immediate hand.

## Two ways to run training

| Path | Best for | Where it lives |
|---|---|---|
| **Local / Colab** with our `clbv-train` script | Iterating, full control, free-tier T4 / A100 / H100 | Step-by-step Colab walkthrough below in this README. |
| **Prime Intellect Hosted Training** with `prime train` | Cheap *once* the policy emits valid actions; Prime runs the GRPO loop server-side | See [`docs/PRIME.md`](docs/PRIME.md). The env package lives in [`environments/clbench-poker/`](environments/clbench-poker/) and gets pushed to Prime's hub via `prime env push`. |

Both paths use the same `CLBenchEnv` wrapper and the same notepad logic;
they differ only in *who drives the GRPO loop*. The Colab path uses
`vf.RLTrainer` directly; Prime's hosted path uses Prime's training service
which speaks to the env over the verifiers env-server protocol.

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

## Run it on Colab (step-by-step)

### 1. Open a notebook with A100 attached

Go to <https://colab.research.google.com> → **File → New notebook**.

Then **Runtime → Change runtime type → Hardware accelerator: A100 GPU → Save**.
(H100 works too on Pro+; free-tier T4 will technically run but you'll need to
drop `vllm_gpu_memory_utilization` to ~0.30 in the TOML and expect it to be slow.)

Wait until the runtime indicator says **Connected to A100**.

### 2. Confirm the GPU

```python
!nvidia-smi
```

You should see an A100 with ~40 GB. If it says "Tesla T4" or "no GPU", the
runtime didn't switch — go back to step 1.

### 3. Clone the repo and run setup

```python
%cd /content
!git clone https://github.com/sr-networks/clbench-verifiers.git
%cd clbench-verifiers
!bash scripts/colab_setup.sh
```

Setup takes ~5–8 min (vLLM is the slow install). The script is idempotent —
re-running is safe after a runtime restart.

### 4. Smoke-test on CPU (no GPU work, ~10 sec)

```python
!python tests/test_env_smoke.py
```

You should see `All smoke tests passed.` This proves the env wrapper, parser,
notepad augmentation, and the CLBench poker task all wire up correctly
**before** you burn GPU time on training. If anything errors here, fix it
before going further.

### 5. Train — pick one of these two cells

**Plain GRPO** (single-instance rollouts, no memory, ~1 hr on A100):

```python
!clbv-train --config configs/poker_qwen2_5_1_5b.toml
```

**Notepad GRPO** (4-instance rollouts with icl_notepad-style memory,
~2 hr on A100):

```python
!clbv-train --config configs/poker_qwen2_5_1_5b_notepad.toml
```

Watch the `mean_instance_reward` column in the per-step output — it should
drift upward over the first ~30 steps. If it stays flat at 0 or strongly
negative the whole run, something is mis-wired; see "When something breaks"
below.

Checkpoints save every 50 steps under `./checkpoints/poker_qwen2_5_1_5b/`
(or `..._notepad/`).

### 6. Evaluate via the official CLBench harness

For the plain checkpoint:

```python
!clbv-eval --checkpoint ./checkpoints/poker_qwen2_5_1_5b/final \
    --task exploitable_poker --schedule quick_test
```

For the notepad checkpoint, pass the same flag at eval time so the bench's
view of the policy matches what was trained:

```python
!clbv-eval --checkpoint ./checkpoints/poker_qwen2_5_1_5b_notepad/final \
    --task exploitable_poker --schedule quick_test \
    --clbench-arg=--system.use_notepad=true
```

`clbv-eval` spawns a vLLM server, shells out to `clbench run …`, and tears the
server down. You'll see the standard CLBench rollout output and a final
mean-reward summary.

### 7. Compare against the published baseline

```python
import gzip, json
from pathlib import Path
p = Path('continual-learning-bench/final_results/runs/icl-gpt-5.4/tasks/exploitable_poker.json.gz')
with gzip.open(p) as f:
    d = json.load(f)
print('icl-gpt-5.4 mean reward:', d['summary']['aggregate']['score']['mean'])
print('reset_each_instance baseline:', d['baseline_trace']['result']['score'])
```

The target is to clear the **`reset_each_instance` baseline** (≈ 1.29 for
poker on gpt-5.4 — that's "same big model with no memory"). A 1.5B Qwen
clearing it would be a clear win for GRPO + memory; it's a stretch goal but
a credible target.

### Or just open the notebook directly

`notebooks/train_poker.ipynb` is steps 1–7 in cell form. In Colab:
**File → Open notebook → GitHub tab → enter `sr-networks/clbench-verifiers` →
pick `notebooks/train_poker.ipynb`**.

### When something breaks

In rough order of likelihood:

1. **`vf.RLTrainer` argument-name drift.** The `_build_grpo_config` function
   in `train.py` is the only piece not fully verified against current
   verifiers source — if upstream renamed a field, the override warning will
   be silent and the trainer will crash on construction. Fix is usually a
   one- or two-line rename.
2. **vLLM out-of-memory.** Drop `vllm_gpu_memory_utilization` in the TOML to
   0.35 or 0.30, and/or `rollouts_per_step` to 4. On notepad config,
   `max_prompt_tokens` is the other knob to lower.
3. **`ModuleNotFoundError: src`.** Means CLBench didn't install. Run
   `!pip show cl-benchmark` to confirm; if missing, re-run `colab_setup.sh`
   or install manually with
   `!pip install --ignore-requires-python git+https://github.com/pgasawa/continual-learning-bench.git`.
4. **`texasholdem` import error during smoke test.** Run
   `!pip install --ignore-requires-python texasholdem==0.11.0`.
5. **Eval hangs after "vLLM serving started".** vLLM took longer than the
   600 s startup timeout. Check `!nvidia-smi` for memory pressure; if
   another process is holding the GPU, restart the runtime.

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

## Contributing — board of next steps

The project is set up so most of the work below is **independent**, can be
picked up by a single contributor end-to-end, and only requires the parts
of the stack you already touch. Bullets are roughly ordered by what unblocks
the most things downstream.

Tag legend:

- 🔥 **Cost-blocking** — required before more Prime training runs make sense.
- ⭐ **Good first issue** — small, scoped, no deep familiarity with verifiers/Prime needed.
- 🎯 **Research-y** — the result is itself the contribution.

### Phase A — Cost control & training stability

Cost-control items: ✅ shipped. Learning-stability items below are now the
critical path before another full run is worth the spend.

- [x] 🔥 ⭐ **Cap context blowup on parse failures.** ([5561849](https://github.com/sr-networks/clbench-verifiers/commit/5561849))
  `max_turns 64→16`, new `@vf.stop input_token_budget_exceeded` reading
  verifiers' usage tracker, default cap 8000 input tokens/rollout.
- [x] ⭐ **Two-phase reward shaping.** ([e50c78a](https://github.com/sr-networks/clbench-verifiers/commit/e50c78a))
  Partial-format heuristic in `_format_score(text, schema, parsed_ok)`,
  tracked as `best_format_score` and weighted 0.1 by default. Used to break
  zero-advantage on cold-start groups.
- [x] **Read `reasoning_content` from thinking models.** ([bb759e3](https://github.com/sr-networks/clbench-verifiers/commit/bb759e3))
  Qwen3.5 / Nemotron / GPT-OSS-thinking split output across two fields.
  The wrapper now reads both. Configs default `enable_thinking=false` for
  Qwen3.5 to also save the reasoning-channel tokens.

Now the open items, in order of how directly they unblock real learning:

- [x] 🔥 **Constrained decoding via `guided_json`.** ([84d3312](https://github.com/sr-networks/clbench-verifiers/commit/84d3312))
  Configured via Prime's `[sampling.extra_body.guided_json]` TOML block (env
  v0.1.8). Smoke `zco930bwwpqgjd2cm3hxj80v` confirmed the constraint reaches
  vLLM; full plain run `g0g0j6hkxuoct7ipyd8faaau` showed +0.56 chip/hand
  uplift over the broken-extraction run.
- [x] 🔥 **Per-instance history wipe + last-instance reward.** ([3d09fbc](https://github.com/sr-networks/clbench-verifiers/commit/3d09fbc))
  New env knob `clear_history_between_instances` (overrides
  `get_prompt_messages` to wipe within-instance history at instance
  boundaries) plus `final_instance_reward_weight`/`mean_instance_reward_weight`
  rubric weights for clean credit assignment. Required for the notepad
  config to actually be testing what it claims to test.
- [ ] 🔥 ⭐ **Train on more diverse hand seeds.**
  Currently `task_kwargs.seed=0` is shared across all rollouts in all
  steps; the policy sees the same opening hand 1,600 times in a 200-step
  training run. Threading a per-rollout seed into the env (or calling
  `Poker(num_instances=N, seed=...)` with rotating seeds) would broaden
  the training distribution. Cleanest fix to "no within-run learning trend"
  hypothesis #1.
- [ ] 🔥 🎯 **SFT bootstrap before GRPO.**
  Generate a few hundred valid `PokerAction` JSONs (random legal actions +
  short fake-thinking strings), SFT Qwen3.5-2B on them for 100–200 steps,
  *then* GRPO. Particularly useful for tasks without `guided_json` support,
  and a cheap way to lift the model off the cold-start floor before RL.
- [ ] ⭐ **Bump `max_tokens` to 2048 in Prime configs.**
  Qwen3.5-2B is verbose; with 1024 it sometimes spends the entire output
  budget on the `thinking` field. With guided_json on this matters less
  (output is shape-bounded), but raising the limit removes another
  spurious failure mode. Trade-off: cost per rollout ≈ 1.5× — still in
  budget envelope.
- [ ] ⭐ **Schema description tightening.**
  Add `description="Brief reasoning, ≤30 words"` (or similar) to the
  `thinking` field in CLBench's `PokerAction`. Smaller deltas are likely
  but it's the cheapest way to discourage rambling. Upstream PR to
  `pgasawa/continual-learning-bench`.
- [ ] ⭐ **Per-turn rollout diagnostics in env state.**
  Track per-turn: output tokens, prompt-vs-completion ratio, time. Surface
  as weight-0 rubric components so they show up in
  `prime train metrics --output json` and we can spot regressions like
  the "all thinking, no action" attractor without manually inspecting
  rollouts.

### Phase B — More tasks

Each new task wrapped buys us a new training distribution and exercises a
different part of the bench's continual-learning signal.

- [ ] 🎯 **Wrap `database_exploration`.**
  Similar shape to poker (no Docker, multi-turn, structured action). Action
  format is text-prefix (`QUERY <sql>` / `ANSWER <text>`) rather than JSON,
  so this also exercises a different parser path. Add
  `environments/clbench-database-exploration/` mirroring the poker package.
  Memory payoff is high (schema reuse), so notepad mode should shine here.
- [ ] **Wrap `cohort_studies`.**
  Pure tool-calling task (typed pydantic discriminated union of 6 tools).
  Forces the env wrapper to round-trip OpenAI-style tool calls rather than
  inline JSON. Likely needs a small `tool_call` extension to the env.
- [ ] **Wrap `blind_spectrum_monitoring`.**
  Single-shot structured output per instance. Easiest of the remaining
  tasks. Mostly tests the wrapper for `instance_complete=True` immediately.
- [ ] 🎯 **Wrap `codebase_adaptation` + `sales_prediction` via remote sandbox.**
  These need Docker per turn. Two viable backends:
    - **Modal** — easy, high-quality SDK, $/min pricing.
    - **e2b** — purpose-built for AI sandboxes, cheaper at scale.
  Add a `clbench_verifiers/sandbox/` module that adapts CLBench's
  in-task Docker calls to a remote sandbox. Open question: is per-instance
  container reuse worth the complexity, or do we re-spawn each instance?

### Phase C — Memory & training research

- [ ] 🎯 **Compare ICL / icl_notepad / mem0 / no-memory at equal compute.**
  Replicate (a subset of) CLBench's leaderboard study but with our trained
  policy as the underlying LM. Each system is selected via `[[env]].args`
  (notepad on/off) and Prime training config. Result is a small table that
  could go in a blog post / paper.
- [ ] 🎯 **Per-turn vs trajectory rewards.**
  Right now the trajectory's mean instance reward is assigned to every
  emitted token. For poker (5 turns) that's fine; for `database_exploration`
  (15+ turns) the gradient is blurred. Implement GAE-per-turn or use
  `Observation.content` as a step reward source.
- [ ] 🎯 **Cross-rollout memory (for notepad mode).**
  Currently the notepad resets between rollouts to keep GRPO i.i.d.
  Experiment with carrying notepad state across rollouts in a group (so
  rollout 7 of 8 can read what rollout 1 wrote). This is a small departure
  from canonical GRPO but lets the policy learn longer-horizon memory use.
- [ ] **Curriculum: easier opponent → harder opponent.**
  Train against `calling_station` (current default), then mix in
  `fit_or_fold` and `loose_aggressive`. Compare to flat-distribution training.

### Phase D — Engineering / DX

- [ ] ⭐ **Pull a Prime checkpoint and run `clbv-eval` on it.**
  End-to-end "trained model → official CLBench score" path. Add a
  `clbv-eval --prime-run <run_id>` shortcut that downloads the latest
  checkpoint and shells out to existing eval code.
- [ ] ⭐ **Replace `requires-python>=3.13` workaround with an upstream PR.**
  Submit a PR to `pgasawa/continual-learning-bench` relaxing the pin so we
  can drop the fork at `sr-networks/continual-learning-bench`.
- [ ] ⭐ **CI on this repo.**
  GitHub Actions running `pytest tests/test_env_smoke.py` on push. The
  smoke tests don't need a GPU and should stay green.
- [ ] **Push `clbench-verifiers` (the glue package) to PyPI.**
  Currently we install via git URL. Pinning to a PyPI version makes Prime
  CI builds more reproducible.
- [ ] **Pre-commit hooks (ruff + black) and a CONTRIBUTING.md.**

### How to claim work

1. Open a GitHub issue mirroring the bullet (e.g. *"Phase A — cap context
   blowup on parse failures"*) and link this section.
2. Drop a comment claiming it before starting so we don't double-up.
3. Open the PR against `main`. Add or update a smoke test if you touch the
   env wrapper or the rubric — anything else needs at least manual
   reproduction notes.

For Prime-side changes (env CI / training cost / hub layout), include a
short `prime train metrics` snapshot in the PR description so reviewers
can see the cost effect without having to launch a run themselves.

## License

Apache-2.0.
