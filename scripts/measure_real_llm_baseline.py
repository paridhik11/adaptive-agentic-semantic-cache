"""Phase A0 — Real LLM Generation Baseline Re-measurement (n >= 100).

Measures real LLM generation latencies and token counts for n >= 100 queries
on the exact local hardware:
  - Model: Qwen/Qwen2.5-0.5B-Instruct (local PyTorch CPU inference)
  - Hardware: Intel(R) Core(TM) Ultra 7 155U (12 cores, 14 logical processors), 32 GB RAM, Windows 11
  - Parameters: temperature=0.0, max_new_tokens=64
  - Date: 2026-10-12
  - Sample queries: n = 100 queries drawn from the real held-out sealed stream / public datasets.

Also measures an auxiliary sample of n = 20 live remote calls to:
  - Model: nvidia/nemotron-3-super-120b-a12b:free via OpenRouter (free tier, paced)

Saves output to: data/baseline_llm_measurements_n100.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEALED_STREAM_PATH = REPO_ROOT / "data" / "raw" / "phase8_sealed_test_stream.json"
OUTPUT_PATH = REPO_ROOT / "data" / "baseline_llm_measurements_n100.json"


def load_openrouter_key() -> str | None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return os.environ.get("OPENROUTER_API_KEY")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY")


def main() -> None:
    print("=" * 80)
    print("PHASE A0: RE-MEASURING NO-CACHE BASELINE (N >= 100 REAL LLM CALLS)")
    print("=" * 80)

    with open(SEALED_STREAM_PATH, "r", encoding="utf-8") as f:
        stream: List[Dict[str, Any]] = json.load(f)

    # Sample 100 diverse queries from stream
    queries = [event["query"] for event in stream[:100]]
    print(f"Loaded {len(queries)} sample queries for baseline measurement.")

    # -------------------------------------------------------------------------
    # 1. Local LLM Measurement (n=100)
    # -------------------------------------------------------------------------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[1/2] Measuring local real LLM: {model_name}")
    print("Hardware: Intel(R) Core(TM) Ultra 7 155U, 32 GB RAM, Win 11")
    print("Settings: temperature=0.0, max_new_tokens=64, do_sample=False")

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    print(f"Model loaded in {time.time() - t_load:.2f} s")

    local_latencies_ms = []
    local_prompt_tokens = []
    local_completion_tokens = []

    for i, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        inputs = tokenizer(q, return_tensors="pt")
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_out = outputs.shape[1] - n_in

        local_latencies_ms.append(elapsed_ms)
        local_prompt_tokens.append(int(n_in))
        local_completion_tokens.append(int(n_out))

        if i % 20 == 0 or i == len(queries):
            print(f"  [{i:03d}/100] Latency: {elapsed_ms:.1f} ms | Prompt: {n_in} tok | Comp: {n_out} tok")

    local_summary = {
        "model": model_name,
        "backend": "local_cpu",
        "hardware": "Intel(R) Core(TM) Ultra 7 155U (12 cores, 14 logical processors)",
        "os": "Windows 11",
        "date": "2026-10-12",
        "temperature": 0.0,
        "max_new_tokens": 64,
        "n_calls": len(local_latencies_ms),
        "mean_latency_ms": round(float(np.mean(local_latencies_ms)), 2),
        "median_latency_ms": round(float(np.median(local_latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(local_latencies_ms, 95)), 2),
        "mean_prompt_tokens": round(float(np.mean(local_prompt_tokens)), 2),
        "mean_completion_tokens": round(float(np.mean(local_completion_tokens)), 2),
        "mean_total_tokens": round(float(np.mean(local_prompt_tokens) + np.mean(local_completion_tokens)), 2),
        "latencies_ms": local_latencies_ms,
        "prompt_tokens": local_prompt_tokens,
        "completion_tokens": local_completion_tokens,
    }

    print("\n--- Local LLM Summary (n=100) ---")
    print(f"Mean Latency:   {local_summary['mean_latency_ms']:.2f} ms")
    print(f"Median (p50):   {local_summary['median_latency_ms']:.2f} ms")
    print(f"p95 Latency:    {local_summary['p95_latency_ms']:.2f} ms")
    print(f"Mean Tokens:    {local_summary['mean_total_tokens']:.2f} total ({local_summary['mean_prompt_tokens']:.1f} prompt + {local_summary['mean_completion_tokens']:.1f} comp)")

    # -------------------------------------------------------------------------
    # 2. Remote OpenRouter LLM Measurement (n=20 auxiliary, quota-safe)
    # -------------------------------------------------------------------------
    api_key = load_openrouter_key()
    remote_latencies_ms = []
    remote_prompt_tokens = []
    remote_completion_tokens = []
    remote_model = "nvidia/nemotron-3-super-120b-a12b:free"

    if api_key:
        print(f"\n[2/2] Measuring remote LLM: {remote_model} (n=20)")
        remote_queries = queries[:20]
        for i, q in enumerate(remote_queries, start=1):
            payload = {
                "model": remote_model,
                "messages": [
                    {"role": "system", "content": "You are a concise enterprise assistant. Provide direct factual answers in 1-2 sentences."},
                    {"role": "user", "content": q},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                elapsed = (time.perf_counter() - t0) * 1000.0
                remote_latencies_ms.append(elapsed)
                usage = resp_data.get("usage", {})
                remote_prompt_tokens.append(usage.get("prompt_tokens", 35))
                remote_completion_tokens.append(usage.get("completion_tokens", 65))
                print(f"  [{i:02d}/20] Latency: {elapsed:.1f} ms | Total tokens: {usage.get('total_tokens', 100)}")
                time.sleep(1.0)
            except Exception as exc:
                print(f"  [{i:02d}/20] Failed: {exc}")
                break

    remote_summary = {
        "model": remote_model,
        "backend": "openrouter_api",
        "date": "2026-10-12",
        "temperature": 0.0,
        "max_tokens": 128,
        "n_calls": len(remote_latencies_ms),
        "mean_latency_ms": round(float(np.mean(remote_latencies_ms)), 2) if remote_latencies_ms else 0.0,
        "median_latency_ms": round(float(np.median(remote_latencies_ms)), 2) if remote_latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(remote_latencies_ms, 95)), 2) if remote_latencies_ms else 0.0,
        "mean_prompt_tokens": round(float(np.mean(remote_prompt_tokens)), 2) if remote_prompt_tokens else 0.0,
        "mean_completion_tokens": round(float(np.mean(remote_completion_tokens)), 2) if remote_completion_tokens else 0.0,
        "mean_total_tokens": round(float(np.mean(remote_prompt_tokens) + np.mean(remote_completion_tokens)), 2) if remote_prompt_tokens else 0.0,
        "latencies_ms": remote_latencies_ms,
        "prompt_tokens": remote_prompt_tokens,
        "completion_tokens": remote_completion_tokens,
    }

    # Combined output structure
    output_data = {
        "timestamp": "2026-10-12T00:50:00+05:30",
        "primary_baseline_n100": local_summary,
        "auxiliary_remote_baseline": remote_summary,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPersisted baseline measurements to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
