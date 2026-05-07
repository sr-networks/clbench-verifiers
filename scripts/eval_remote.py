"""
Run CLBench's official ``run_benchmark`` against a remote OpenAI-compatible
endpoint (Prime deployment, vLLM serve, anything else exposing the chat
completions surface).

Why this exists: ``clbench-verifiers/eval.py`` spins up a local vLLM and
shells out to the ``clbench`` CLI. That doesn't fit Prime-trained
checkpoints, which live in cloud storage and are exposed via
``prime deployments create``. This driver skips local serving entirely:
point ``--base-url`` at the deployment URL and the script runs
N rollouts + a stateless baseline, then prints the aggregate scores.

The system class registered on CLBench is
``clbench_verifiers.system.VLLMClientSystem``; importing this module
ensures the ``vllm_local`` system name is registered before
``run_benchmark`` resolves it.

Outputs the headline numbers per CLBench convention:
  - ``mean_reward``: mean of all instance rewards across rollout runs.
  - ``mean_baseline``: mean of stateless-reset baseline rewards.
  - ``mean_gain``: stateful minus stateless = "did memory help?"

Use ``--no-baseline`` to skip the baseline phase (faster but you lose the
gain metric).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any


def _import_clbench():
    """Import clbench bits + register VLLMClientSystem.

    Importing ``clbench_verifiers.system`` triggers the
    ``@register_system("vllm_local")`` side effect. We call it before
    ``run_benchmark`` resolves the system name.
    """
    try:
        import clbench_verifiers.system  # noqa: F401  (registers vllm_local)
        from src.registry import get_task_class  # type: ignore
        from src.runs.benchmark import run_benchmark  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"clbench / clbench-verifiers not importable: {exc}.\n"
            "Run from the project's venv: "
            "`uv run --python 3.12 python scripts/eval_remote.py ...`"
        ) from exc
    return get_task_class, run_benchmark, clbench_verifiers.system.VLLMClientSystem


def _aggregate_rewards(task_results) -> dict[str, float]:
    """Pull instance rewards out of TaskResults and summarise."""
    all_rewards: list[float] = []
    per_run_means: list[float] = []
    for result in task_results.results:
        run_rewards = [float(o.reward) for o in result.instance_outcomes]
        if not run_rewards:
            continue
        all_rewards.extend(run_rewards)
        per_run_means.append(statistics.mean(run_rewards))
    out = {
        "n_rewards": float(len(all_rewards)),
        "n_runs": float(len(per_run_means)),
    }
    if all_rewards:
        out["mean"] = statistics.mean(all_rewards)
        out["std"] = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0
        out["min"] = min(all_rewards)
        out["max"] = max(all_rewards)
    if per_run_means:
        out["per_run_mean"] = statistics.mean(per_run_means)
        out["per_run_std"] = (
            statistics.stdev(per_run_means) if len(per_run_means) > 1 else 0.0
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible endpoint base URL (e.g. https://.../v1).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name string the endpoint accepts in `model` field.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Env var holding the API key. Use 'EMPTY' literal for keyless servers.",
    )
    parser.add_argument("--task", default="exploitable_poker")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Instances per run (CLBench task_kwargs.num_instances).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent rollout runs (different seeds).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel rollout workers. 1 is safest for HTTP-bound endpoints.",
    )
    parser.add_argument(
        "--use-notepad",
        action="store_true",
        help="Enable icl_notepad-style memory (matches notepad-on training).",
    )
    parser.add_argument(
        "--clear-context-between-instances",
        action="store_true",
        help="Wipe within-instance history at instance boundaries (matches "
        "notepad training; the notepad becomes the ONLY memory channel).",
    )
    parser.add_argument(
        "--enable-guided-json",
        action="store_true",
        help="Send extra_body.guided_json so the endpoint constrains JSON output. "
        "Use this to match training-time grammar enforcement.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip stateless baseline phase (faster, but no gain metric).",
    )
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra task_kwargs entry (repeatable). E.g. --task-arg "
        'opponent_policy=calling_station --task-arg seed=0',
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the headline numbers as JSON.",
    )
    args = parser.parse_args(argv)

    api_key = (
        args.api_key_env if args.api_key_env == "EMPTY" else os.environ.get(args.api_key_env)
    )
    if not api_key:
        raise SystemExit(
            f"API key not found in env var '{args.api_key_env}'. "
            "Use --api-key-env EMPTY for keyless local servers."
        )

    get_task_class, run_benchmark, VLLMClientSystem = _import_clbench()

    task_kwargs: dict[str, Any] = {"num_instances": args.num_instances}
    for entry in args.task_arg:
        if "=" not in entry:
            raise SystemExit(f"Bad --task-arg {entry!r}; expected KEY=VALUE")
        key, _, value = entry.partition("=")
        # Heuristic int-coerce: most CLBench task params are int.
        try:
            value_typed: Any = int(value)
        except ValueError:
            try:
                value_typed = float(value)
            except ValueError:
                value_typed = value
        task_kwargs[key] = value_typed

    system_params = dict(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_notepad=args.use_notepad,
        clear_context_between_instances=args.clear_context_between_instances,
        enable_guided_json=args.enable_guided_json,
    )

    task_class = get_task_class(args.task)
    print(
        f"[eval] task={args.task} runs={args.runs} instances/run={args.num_instances} "
        f"use_notepad={args.use_notepad} guided_json={args.enable_guided_json}",
        file=sys.stderr,
    )

    baseline_info, task_results, _ = run_benchmark(
        task_class=task_class,
        task_params=task_kwargs,
        system_class=VLLMClientSystem,
        system_params=system_params,
        runs=args.runs,
        max_workers=args.max_workers,
        system_name="vllm_local",
        task_name=args.task,
        include_baseline=not args.no_baseline,
        verbose_runs=False,
    )

    rollout_stats = _aggregate_rewards(task_results)
    out: dict[str, Any] = {
        "task": args.task,
        "model": args.model,
        "use_notepad": args.use_notepad,
        "enable_guided_json": args.enable_guided_json,
        "num_runs": args.runs,
        "num_instances_per_run": args.num_instances,
        "rollout": rollout_stats,
    }

    if baseline_info is not None:
        # ``run_benchmark`` returns
        # (baseline_index, baseline_task_result, baseline_metrics, _opts)
        _, baseline_task_result, _, _ = baseline_info
        baseline_rewards = [float(o.reward) for o in baseline_task_result.instance_outcomes]
        if baseline_rewards:
            out["baseline_mean"] = statistics.mean(baseline_rewards)
            out["baseline_n"] = len(baseline_rewards)
            mean_rollout = rollout_stats.get("mean")
            if mean_rollout is not None:
                out["mean_gain"] = mean_rollout - out["baseline_mean"]

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
