#!/usr/bin/env python3
"""Run the Colab GRPO training path with a compatible verifiers vLLM server."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

import requests


def _load_hf_token() -> None:
    try:
        from google.colab import userdata  # type: ignore
    except Exception as exc:
        print(f"Could not import Colab userdata: {exc}")
        return

    token = userdata.get("HF_token") or userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("HF token loaded from Colab secrets")
    else:
        print("HF token secret not found; continuing unauthenticated")


def run(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=os.environ.copy(), **kwargs)


def main() -> None:
    _load_hf_token()
    repo = pathlib.Path("/content/clbench-verifiers")
    os.chdir(repo)

    run(["git", "pull", "--ff-only"])
    run([sys.executable, "-m", "pip", "install", "-q", "verifiers==0.1.7"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "trl>=0.12",
            "transformers>=4.45",
            "accelerate>=0.34",
            "datasets>=2.20",
            "peft>=0.12",
            "liger-kernel>=0.5.10",
            "deepspeed",
            "torchao>=0.16.0",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"])
    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."])
    run(
        [
            sys.executable,
            "-c",
            "from vllm.utils import FlexibleArgumentParser; import vllm; print(vllm.__version__)",
        ]
    )

    for pattern in ["vllm.entrypoints.openai.api_server", "verifiers.rl.inference.server", "vf-vllm"]:
        subprocess.run(["pkill", "-f", pattern], check=False)
    time.sleep(3)

    log_path = pathlib.Path("/content/vf_vllm_clbench.log")
    log = log_path.open("w")
    server_cmd = [
        "vf-vllm",
        "--model",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "--served-model-name",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.45",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    print("\n$ " + " ".join(server_cmd), flush=True)
    server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())

    url = "http://0.0.0.0:8000/health"
    for i in range(180):
        if server.poll() is not None:
            log.flush()
            log.close()
            print(log_path.read_text()[-12000:])
            raise RuntimeError(f"vf-vllm exited early with code {server.returncode}")
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("vf-vllm health check passed")
                break
        except Exception:
            pass
        if i % 15 == 0:
            print(f"waiting for vf-vllm ({i * 2}s)", flush=True)
        time.sleep(2)
    else:
        log.flush()
        log.close()
        print(log_path.read_text()[-12000:])
        raise RuntimeError("vf-vllm did not become healthy")

    print(log_path.read_text()[-4000:])
    run([sys.executable, "-m", "clbench_verifiers.train", "--config", "configs/poker_qwen2_5_1_5b.toml"])


if __name__ == "__main__":
    main()
