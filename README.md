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

## Status (2026-05-06)

| Component | State | Notes |
|---|---|---|
| `exploitable_poker` env wrapper | ✅ working end-to-end | `setup_state` / `env_response` / `@vf.stop` aligned with verifiers `0.1.12`; reads `content` AND `reasoning_content` so thinking models like Qwen3.5 are handled correctly |
| `icl_notepad`-style memory | ✅ working in env + `vllm_local` system | tested with multi-instance rollouts |
| Cost caps (`max_input_tokens_per_rollout`, lower `max_turns`) | ✅ landed | ~80× cost reduction vs uncapped run, see table below |
| Partial-format reward shaping | ✅ landed | weight 0.1; breaks zero-advantage at cold-start |
| Local CPU smoke tests | ✅ all 14 passing | drives a real poker task through the wrapper without GPU/verifiers; covers reasoning_content and format-score paths |
| Local Colab GRPO (`clbv-train`) | ⚠️ scaffolded, **not yet run end-to-end** | trainer config tracks verifiers `RLConfig` API drift; first real run still pending |
| Prime env push | ✅ `sr-networks/clbench-poker@0.1.8` PUBLIC | CI green; live at <https://app.primeintellect.ai/dashboard/environments/sr-networks/clbench-poker> |
| Constrained decoding (`guided_json`) | ✅ landed via `[sampling.extra_body.guided_json]` | `zco930bwwpqgjd2cm3hxj80v`: parse_failure_penalty 0.0, format_score 1.0, instance completion 1.0, **first positive step reward (+0.10)** |
| Prime Hosted Training smoke | ✅ green pipeline | `zco930bwwpqgjd2cm3hxj80v`: 2 steps × 4 rollouts, all parse, all complete, $0.04 |
| Prime full plain GRPO (200 steps) | ⏳ ready to relaunch on the new clean signal | `p8hx0n3n5gxu0dbjs5s7f4nc` ran without guided_json and showed no learning trend ($5.59) |
| Prime notepad GRPO | ⏳ ready to launch after plain | notepad config now ships with the matching schema-aware `guided_json` block |
| `database_exploration` env | ⏳ not started | low-effort follow-up; same wrapper shape |
| Docker-sandboxed tasks | ⏳ not started | needs Modal/e2b sandbox for `codebase_adaptation` + `sales_prediction` |

### Cost is controlled. Learning is the next frontier.

Cost control was the explicit goal of the first push, and it's solved:
the original uncapped 2-step smoke burned **$3.26** on 30+-turn gibberish
loops; the current-config 2-step smoke costs **$0.05** and runs to
completion with rollouts that actually finish poker hands. A 200-step
plain run (8 rollouts/step) lands somewhere around **$2–4**, predictable
and capped.

What we *also* learned in the process is that a green pipeline isn't the
same as a learning pipeline. The current 200-step run's reward keeps
oscillating around -1.8 with no upward trend through ~90 steps. Two real
issues remain:

1. **Mid-output truncation under `max_tokens=1024`.** Qwen3.5-2B is verbose;
   it spends its entire output budget on the `thinking` field, leaving the
   `action` / `amount` fields off the end. The JSON is incomplete, the
   action either fails to parse or fails task validation, and the rollout
   wastes turns re-prompting until the input-token cap fires.
2. **`num_instances_completed` is a free lunch.** It exists as a weight-0
   diagnostic. When a rollout never finishes a hand, `mean_instance_reward`
   is treated as `0` (no instances to average over), which is *better than*
   the typical `−1` of an actual losing hand. A policy that emits valid-
   shape JSON without playing therefore beats a policy that plays badly.
   This is a reward-hacking attractor reachable from the current init.

Both have one-line fixes; we landed cost control first because it was
blocking us from finding these. The fixes — `max_tokens=2048` and
weight-1 on `num_instances_completed` — are proposed but not yet shipped
as of this README write.

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

- [ ] 🔥 ⭐ **Penalize "didn't finish a hand".**
  `num_instances_completed` is a weight-0 diagnostic. A rollout that emits
  valid-shape JSON without ever finishing an instance gets
  `mean_instance_reward = 0`, which is *better* than playing badly. Bump
  the weight to ~1.0 in `clbench_verifiers/rubric.py::build_clbench_rubric`
  so the model can't reward-hack the metric. Add a smoke test that fails
  if a rollout with zero completions outscores one that completed and lost.
- [ ] 🔥 ⭐ **Bump `max_tokens` to 2048 in Prime configs.**
  Qwen3.5-2B is verbose; with 1024 it spends the entire output budget on
  the `thinking` field and never emits `action`/`amount`. Touches all three
  `configs/prime_*.toml` `[sampling]` blocks. Trade-off: cost per rollout
  ≈ 2× — still in our budget envelope.
- [x] 🔥 **Constrained decoding via `guided_json`.** ([84d3312](https://github.com/sr-networks/clbench-verifiers/commit/84d3312))
  Configured via Prime's `[sampling.extra_body.guided_json]` TOML block (env
  v0.1.8). Smoke `zco930bwwpqgjd2cm3hxj80v` confirms the constraint reaches
  vLLM and the policy now produces fully valid JSON every turn. Two earlier
  attempts (env-side `state["sampling_args"]` injection, runtime
  `get_model_response` override) didn't make it through Prime's TITO client
  path; the TOML-driven path does.
- [ ] 🔥 🎯 **SFT bootstrap before GRPO.**
  Generate a few hundred valid `PokerAction` JSONs (random legal actions +
  short fake-thinking strings), SFT Qwen3.5-2B on them for 100–200 steps,
  *then* GRPO. Cleanest fix to cold-start; alternative to constrained
  decoding for environments where `guided_json` is unavailable.
- [ ] ⭐ **Schema description tightening.**
  Add `description="Brief reasoning, ≤30 words"` (or similar) to the
  `thinking` field in CLBench's `PokerAction`. Smaller deltas are likely
  but it's the cheapest way to discourage the rambling that hits
  `max_tokens`. Upstream PR to `pgasawa/continual-learning-bench`.
- [ ] ⭐ **Per-turn rollout diagnostics in env state.**
  Track per-turn: output tokens, prompt-vs-completion ratio, time. Surface
  as weight-0 rubric components so they show up in
  `prime train metrics --output json` and we can spot regressions like
  the "all thinking, no action" attractor without having to manually
  inspect rollouts.

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
