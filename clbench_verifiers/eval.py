"""
Evaluate a trained checkpoint via the official CLBench harness.

Workflow:
1. Spin up vLLM serving the checkpoint on localhost:<port>.
2. Wait until ``/v1/models`` responds.
3. Shell out to ``clbench run`` against ``--system vllm_local`` so the bench
   uses the same exact System adapter the user would use to register the
   trained policy in CLBench.
4. Tear down the vLLM process.

Why shell out to clbench rather than calling its Python API: the bench has
a lot of run-orchestration plumbing (manifests, traces, baselines, viewer
artifacts) that's easier to invoke via the CLI than to wire up by hand.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_server(base_url: str, timeout_s: float = 600.0, poll_s: float = 2.0) -> None:
    """Poll ``/v1/models`` until reachable or timeout."""
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(poll_s)
    raise RuntimeError(
        f"vLLM server at {base_url} did not become ready within "
        f"{timeout_s:.0f}s (last error: {last_err})"
    )


@contextmanager
def vllm_server(
    model_path: str,
    *,
    served_model_name: str,
    port: Optional[int] = None,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    extra_args: Optional[list[str]] = None,
):
    """
    Context manager: launch ``vllm serve`` in a subprocess, yield the base URL.
    """
    port = port or _free_port()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]
    if extra_args:
        cmd += extra_args

    logger.info("Launching vLLM: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        _wait_for_server(base_url, timeout_s=600.0)
        yield base_url
    finally:
        logger.info("Terminating vLLM (pid=%s)", proc.pid)
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=30)
        except Exception:  # pragma: no cover
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass


def run_clbench(
    *,
    task: str,
    base_url: str,
    served_model_name: str,
    schedule: Optional[str] = None,
    extra_clbench_args: Optional[list[str]] = None,
) -> int:
    """Invoke ``clbench run <task> --system vllm_local --system.* …``."""
    cmd = [
        "clbench",
        "run",
        task,
        "--system",
        "vllm_local",
        "--system.base_url",
        base_url,
        "--system.model",
        served_model_name,
    ]
    if schedule:
        cmd += ["--schedule", schedule]
    if extra_clbench_args:
        cmd += extra_clbench_args
    logger.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained CLBench policy.")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved HF model directory (or a HF hub name).",
    )
    p.add_argument(
        "--task",
        default="exploitable_poker",
        help="CLBench task to evaluate.",
    )
    p.add_argument(
        "--schedule",
        default=None,
        help="Schedule to use (e.g. quick_test, default).",
    )
    p.add_argument(
        "--served-model-name",
        default="trained-policy",
        help="Name to use in the OpenAI API model field.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
    )
    p.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help="Extra raw arg to pass to vllm (repeatable).",
    )
    p.add_argument(
        "--clbench-arg",
        action="append",
        default=[],
        help="Extra raw arg to pass to clbench run (repeatable).",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s %(message)s")

    ckpt = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint

    with vllm_server(
        ckpt,
        served_model_name=args.served_model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        extra_args=args.vllm_arg,
    ) as base_url:
        rc = run_clbench(
            task=args.task,
            base_url=base_url,
            served_model_name=args.served_model_name,
            schedule=args.schedule,
            extra_clbench_args=args.clbench_arg,
        )
    sys.exit(rc)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
