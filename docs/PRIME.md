# Running on Prime Intellect (Hosted Training)

This page covers the **Prime Hosted Training** path: Prime runs the GRPO
loop server-side and bills per token, so we don't rent a pod. Our job is
to push the env to Prime's registry and submit a training config.

The Colab/local path in the main README still works and is independent —
this is a parallel option, not a replacement.

## Prerequisites

```bash
# Install the CLI
uv tool install -U prime

# Authenticate
prime login

# Confirm you're logged in
prime --plain whoami
```

If `whoami` shows `Username: Not set`, **set it on Prime's web dashboard**
(<https://app.primeintellect.ai/dashboard/account>) before continuing.
Env ids are `<username>/<env-name>`, so without a username `env push`
won't go anywhere useful.

The configs in this repo assume the username is `sr-networks`; if yours
differs, update the `id` field in each `configs/prime_*.toml` and in the
`prime env push` commands below.

## 1. Push the env to Prime's registry

```bash
cd environments/clbench-poker
prime --plain env push --visibility PUBLIC
```

This runs hatchling to build the env package, uploads it to Prime, and
provisions an env-server image. First push takes a few minutes; subsequent
pushes are diffs.

To verify:

```bash
prime --plain env list --output json | jq '.environments[] | select(.id | contains("clbench-poker"))'
prime --plain env info sr-networks/clbench-poker
```

## 2. Smoke run (cents)

Validate the wiring with a 2-step run before launching the real one.

```bash
prime --plain train configs/prime_qwen3_5_2b_smoke.toml
```

The CLI prints a `run_id`. To watch:

```bash
prime --plain train logs <run_id>
prime --plain train get <run_id>
prime --plain train metrics <run_id> --output json
```

If the smoke run finishes with status `completed` and the metrics show any
non-zero `mean_instance_reward`, the pipeline is working. If it errors,
check the env-server logs via `prime train components <run_id>` and
`prime train logs <run_id> --component env-server-0`.

## 3. Full plain-GRPO run (~$1.20–$1.80)

```bash
prime --plain train configs/prime_qwen3_5_2b.toml
```

200 steps, batch size 8, group size 8. Watch the reward distribution:

```bash
prime --plain train metrics <run_id> --output json | jq '.metrics[] | select(.name | contains("reward"))'
prime --plain train distributions <run_id>
```

## 4. Notepad-GRPO run (~$5–$8)

```bash
prime --plain train configs/prime_qwen3_5_2b_notepad.toml
```

This runs 4 CLBench instances per rollout with `icl_notepad`-style memory
enabled. Token cost is roughly 4× the plain run; halve `max_steps` to
`100` for a cheaper first attempt.

## 5. Pull a checkpoint and evaluate

After a run completes:

```bash
prime --plain train checkpoints <run_id>
# Pick a checkpoint id, then:
prime --plain train get-checkpoint <checkpoint_id> --download-path ./prime-ckpt/
```

Then evaluate via the official CLBench harness using our `vllm_local`
adapter — same as in the main README, just point `--checkpoint` at the
downloaded path. For notepad-trained checkpoints add the matching flag:

```bash
clbv-eval --checkpoint ./prime-ckpt/ --task exploitable_poker --schedule quick_test
# Notepad variant:
clbv-eval --checkpoint ./prime-ckpt/ --task exploitable_poker --schedule quick_test \
    --clbench-arg=--system.use_notepad=true
```

(Prime's hosted `inference` service can also serve the trained adapter;
see `prime inference --help` once you want to skip the local vLLM step.)

## Cost reference

From `prime train models --output json` at the time these configs were
written:

| Model | Train $/Mtok | Inference $/Mtok (in / out) |
|---|---|---|
| Qwen/Qwen3.5-0.8B | 0.06 | 0.02 / 0.06 |
| **Qwen/Qwen3.5-2B** | **0.15** | **0.05 / 0.15** |
| Qwen/Qwen3.5-4B | 0.30 | 0.10 / 0.30 |
| Qwen/Qwen3.5-9B | 0.60 | 0.20 / 0.60 |

Re-check before launching:

```bash
prime --plain train models --output json | jq '.models[] | select(.name == "Qwen/Qwen3.5-2B")'
```

## Troubleshooting

- **`env push` fails with "owner not found"** — your username isn't set
  on the Prime dashboard. Fix it on the web UI, run `prime whoami` to
  confirm, retry.
- **`train` rejects the config** — `prime train configs --output json`
  lists all valid fields and types; the most common cause of rejection
  is a renamed field after a CLI upgrade.
- **`mean_instance_reward` is flat at zero across the smoke run** —
  the env is parsing or the policy never emits valid JSON. Look at
  `prime train rollouts <run_id>` to inspect a few sampled completions;
  the diagnostic counters (`num_instances_completed`, `parse_failures`)
  should tell you which.
- **Env-server image fails to build** — usually a dependency conflict.
  `prime env action` lists the CI runs for your env; check the failed
  one for the pip resolver output.
