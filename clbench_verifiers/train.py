"""
GRPO training entry point.

Uses verifiers' ``RLTrainer`` to GRPO-train a small HF policy on a CLBench
task wrapped as a ``MultiTurnEnv``. Designed to run on a single Colab A100/H100.

The verifiers RL stack expects:

- a ``vf.Environment`` (with a Rubric attached) — provided by ``build_clbench_env``
- a tokenizer and model, optionally with LoRA
- a small "seed" dataset whose prompts are unused at rollout time (the env
  generates its own first message in ``setup_state``) but RLTrainer needs a
  Dataset of the right length to drive its outer loop. We synthesize one.

If the verifiers API drifts, the small set of `vf.*` calls below is the only
place to adjust.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All training knobs in one dataclass; load from a TOML/JSON file or kwargs."""

    # Task / env
    task_name: str = "exploitable_poker"
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    max_instances_per_rollout: int = 1
    max_turns: int = 64
    parse_failure_penalty: float = -1.0
    end_on_parse_failure: bool = False
    use_notepad: bool = False
    notepad_max_chars: int = 4000

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    bf16: bool = True

    # GRPO / training
    output_dir: str = "./checkpoints/poker_qwen2_5_1_5b"
    num_train_steps: int = 200
    rollouts_per_step: int = 16     # G in GRPO (group size)
    batch_size: int = 16            # prompts per step (often == rollouts_per_step)
    learning_rate: float = 1e-6
    grad_accum: int = 1
    warmup_steps: int = 5
    save_every: int = 50
    log_every: int = 1
    max_prompt_tokens: int = 4096
    max_completion_tokens: int = 1024

    # vLLM inference for rollouts
    vllm_gpu_memory_utilization: float = 0.45
    vllm_dtype: str = "bfloat16"
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.95
    sampling_top_k: int = 50

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_level: str = "INFO"

    @staticmethod
    def from_toml(path: str) -> "TrainConfig":
        try:
            import tomllib  # py311+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return TrainConfig(**data)


# ---------------------------------------------------------------------------
# Build the components
# ---------------------------------------------------------------------------


def _build_seed_dataset(num_rows: int, env_name: str):
    """
    verifiers' RLTrainer needs a Dataset whose length determines how many
    "samples" exist; for env-driven rollouts the prompt content is irrelevant
    because ``setup_state`` builds the actual first message. We supply a
    placeholder.
    """
    from datasets import Dataset  # type: ignore

    # The placeholder prompt is what the env will see in messages BEFORE
    # setup_state runs — most envs ignore it, but we keep it informative so
    # any inspection or logging shows what's going on.
    seed_msg = [
        {"role": "user", "content": f"<begin {env_name} rollout>"}
    ]
    rows = [
        {
            "prompt": seed_msg,
            "answer": "",  # required by some verifiers code paths
            "info": {"rollout_index": i},
        }
        for i in range(num_rows)
    ]
    return Dataset.from_list(rows)


def _build_model_and_tokenizer(cfg: TrainConfig):
    import verifiers as vf  # type: ignore

    # LoRA is configured on RLConfig in verifiers 0.1.7. Keep model loading
    # plain here and let RLTrainer wrap it with PEFT.
    model_kwargs: dict[str, Any] = {}
    if cfg.bf16:
        import torch  # type: ignore

        model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["attn_implementation"] = "sdpa"
    model_kwargs["use_cache"] = False

    try:
        model, tokenizer = vf.get_model_and_tokenizer(
            cfg.model_name,
            use_liger=False,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        model, tokenizer = vf.get_model_and_tokenizer(
            cfg.model_name,
            model_kwargs=model_kwargs,
        )
    return model, tokenizer


def _build_env(cfg: TrainConfig):
    from .env import build_clbench_env

    return build_clbench_env(
        task_name=cfg.task_name,
        task_kwargs=cfg.task_kwargs,
        max_instances_per_rollout=cfg.max_instances_per_rollout,
        max_turns=cfg.max_turns,
        parse_failure_penalty=cfg.parse_failure_penalty,
        end_on_parse_failure=cfg.end_on_parse_failure,
        use_notepad=cfg.use_notepad,
        notepad_max_chars=cfg.notepad_max_chars,
    )


def _build_grpo_config(cfg: TrainConfig):
    import verifiers as vf  # type: ignore

    try:
        base = vf.RLConfig()
    except AttributeError:
        base = vf.grpo_defaults()
    if cfg.use_lora and hasattr(base, "lora_config"):
        from peft import LoraConfig  # type: ignore

        base.lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # ``GRPOConfig`` is a dataclass / TrainingArguments-derived. We update it
    # in-place with our knobs. Names follow the TRL/verifiers GRPOConfig
    # convention; if the upstream API renames fields, adjust here.
    overrides = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=1,
        max_steps=cfg.num_train_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        save_steps=cfg.save_every,
        logging_steps=cfg.log_every,
        bf16=cfg.bf16,
        report_to=("wandb",) if cfg.wandb_project else (),
        run_name=cfg.wandb_run_name,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        # GRPO-specific
        num_generations=cfg.rollouts_per_step,
        rollouts_per_example=cfg.rollouts_per_step,
        micro_batch_size=cfg.batch_size,
        max_prompt_length=cfg.max_prompt_tokens,
        max_prompt_len=cfg.max_prompt_tokens,
        max_completion_length=cfg.max_completion_tokens,
        max_tokens=cfg.max_completion_tokens,
        max_seq_len=cfg.max_prompt_tokens + cfg.max_completion_tokens,
        temperature=cfg.sampling_temperature,
        top_p=cfg.sampling_top_p,
        top_k=cfg.sampling_top_k,
        # vLLM rollout backend
        use_vllm=True,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_dtype=cfg.vllm_dtype,
    )
    for k, v in overrides.items():
        if hasattr(base, k):
            setattr(base, k, v)
        else:
            logger.warning("GRPOConfig has no field %r; skipping override.", k)
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> str:
    """Run training; return path to final checkpoint."""
    logging.basicConfig(level=cfg.log_level, format="%(levelname)s %(name)s %(message)s")

    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    import verifiers as vf  # type: ignore

    logger.info("Building env: %s", cfg.task_name)
    env = _build_env(cfg)

    logger.info("Building model: %s (lora=%s)", cfg.model_name, cfg.use_lora)
    model, tokenizer = _build_model_and_tokenizer(cfg)

    grpo_cfg = _build_grpo_config(cfg)

    logger.info("Output dir: %s", cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # vf.RLTrainer is verifiers' GRPO trainer (the "nano" trainer that
    # replaced vf.GRPOTrainer in v0.1.7). It takes the env so it can drive
    # rollouts internally.
    try:
        trainer = vf.RLTrainer(
            model=model,
            processing_class=tokenizer,
            env=env,
            args=grpo_cfg,
        )
    except TypeError:
        logger.info("Building seed dataset (%d rows)", cfg.num_train_steps * cfg.batch_size)
        train_dataset = _build_seed_dataset(
            num_rows=cfg.num_train_steps * cfg.batch_size,
            env_name=cfg.task_name,
        )
        trainer = vf.RLTrainer(
            model=model,
            processing_class=tokenizer,
            env=env,
            args=grpo_cfg,
            train_dataset=train_dataset,
        )

    logger.info("Starting GRPO training for %d steps", cfg.num_train_steps)
    trainer.train()

    final_path = str(Path(cfg.output_dir) / "final")
    trainer.save_model(final_path)
    logger.info("Saved final checkpoint to %s", final_path)
    return final_path


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="GRPO-train a CLBench policy with verifiers.")
    p.add_argument("--config", type=str, help="Path to TOML config file.")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="CLBench task name (overrides config).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model name or path (overrides config).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of GRPO steps (overrides config).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint output dir (overrides config).",
    )
    args = p.parse_args(argv)

    if args.config:
        cfg = TrainConfig.from_toml(args.config)
    else:
        cfg = TrainConfig()

    if args.task is not None:
        cfg.task_name = args.task
    if args.model is not None:
        cfg.model_name = args.model
    if args.steps is not None:
        cfg.num_train_steps = args.steps
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    logger.info("Effective config:\n%s", _pretty_config(cfg))
    train(cfg)


def _pretty_config(cfg: TrainConfig) -> str:
    import json

    return json.dumps(asdict(cfg), indent=2, default=str)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
