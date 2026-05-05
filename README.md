# clbench-verifiers

Glue package that runs [Continual Learning Bench](https://github.com/pgasawa/continual-learning-bench)
tasks as [verifiers](https://github.com/willccbb/verifiers) `MultiTurnEnv`s,
so you can train policies on them with GRPO via `vf.RLTrainer` and evaluate
the trained checkpoints back through `clbench run`.

The package itself is small (a wrapper, a parser, a rubric, an eval system,
and a trainer entry point) and intentionally lives outside both upstream repos
so they can be updated independently.

## What you get

- `clbench_verifiers.env.CLBenchEnv` — a `vf.MultiTurnEnv` subclass that
  drives any `ContinualLearningTask` through verifiers' chat-format rollout
  loop. Each "rollout" can be one CLBench instance (default) or N instances
  with memory carried across (continual mode).
- `clbench_verifiers.system.VLLMClientSystem` — a CLBench `System` that
  talks to a vLLM OpenAI-compatible server, so you can plug a trained
  checkpoint into `clbench run` for the official benchmark scoring.
- `clbench_verifiers.train` / `clbench_verifiers.eval` — entry points
  wired up for Google Colab A100/H100.

## Quickstart on Colab

```bash
# In a Colab cell with an A100/H100 attached
!git clone https://github.com/sr-networks/clbench-verifiers.git
!bash clbench-verifiers/scripts/colab_setup.sh
!clbv-train --config clbench-verifiers/configs/poker_qwen2_5_1_5b.toml
```

See `notebooks/train_poker.ipynb` for a step-by-step version.

## Status

Milestone 1: `exploitable_poker` only. Other CLBench tasks
(`codebase_adaptation`, `sales_prediction`, `database_exploration`, etc.)
are wrappable but require either Docker-in-Colab or a remote sandbox and
are left for milestone 2.

## Why a glue package and not a fork

Both upstream repos move fast and have different release cadences. Keeping
the integration as a thin third-party shim means upstream verifiers /
CLBench changes are an SDK bump, not a merge conflict.
