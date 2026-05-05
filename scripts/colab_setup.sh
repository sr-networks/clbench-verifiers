#!/usr/bin/env bash
# Colab bootstrap for clbench-verifiers (A100/H100 image).
#
# Idempotent: rerun is safe.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CLBENCH_DIR="${CLBENCH_DIR:-$REPO_DIR/continual-learning-bench}"

echo ">> Installing system extras (build deps for vLLM)…"
pip install -q --upgrade pip wheel setuptools

echo ">> Installing CLBench (with --ignore-requires-python because its pyproject pins >=3.13 but Colab is 3.10/3.11)…"
if [[ ! -d "$CLBENCH_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/pgasawa/continual-learning-bench.git "$CLBENCH_DIR"
fi
pip install --ignore-requires-python -e "$CLBENCH_DIR"
# Task extras for the milestone-1 task.
pip install --ignore-requires-python "texasholdem==0.11.0"

echo ">> Running CLBench setup (downloads any task assets)…"
# poker has no setup, but this validates the install.
( cd "$CLBENCH_DIR" && python -m src.cli list )

echo ">> Installing verifiers + RL stack…"
pip install -q "verifiers[rl]>=0.1.7"
pip install -q "trl>=0.12" "transformers>=4.45" "accelerate>=0.34" "datasets>=2.20" "peft>=0.12"

# vLLM is the slow one; pin a known-good version compatible with current TRL/verifiers.
pip install -q "vllm>=0.6.3"

echo ">> Installing this glue package in editable mode…"
# Dependencies are installed explicitly above. Use --no-deps here so pip does
# not re-resolve cl-benchmark and reject Colab's Python despite the earlier
# --ignore-requires-python install.
pip install --no-deps -e "$REPO_DIR"

echo ">> Done. Try:"
echo "   clbv-train --config $REPO_DIR/configs/poker_qwen2_5_1_5b.toml"
