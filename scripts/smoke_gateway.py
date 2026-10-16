"""Run a real local HTTP smoke test with a SIMULATED fixture provider only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("choose a new output file; existing evidence is not overwritten")
    root = Path(__file__).resolve().parents[1]
    token_a, token_b, admin = (secrets.token_urlsafe(32) for _ in range(3))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "GATEWAY_PROVIDER": "fixture", "GATEWAY_CACHE_POLICY": "rules",
           "GATEWAY_TENANTS_JSON": json.dumps({token_a: "smoke-a", token_b: "smoke-b"}),
           "GATEWAY_ADMIN_TOKEN": admin, "GATEWAY_MAX_INFLIGHT": "64",
           "GATEWAY_TTL_SECONDS": "3600", "GATEWAY_MAX_ENTRIES": "1024",
           "GATEWAY_MAX_CACHE_BYTES": "67108864", "GATEWAY_MAX_OUTPUT_TOKENS": "4096"}
    env.pop("GATEWAY_CACHE_DB", None)
    env.pop("GATEWAY_CACHE_VERSION", None)
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.gateway:app_from_env", "--factory",
                                "--host", "127.0.0.1", "--port", str(port), "--no-access-log",
                                "--log-level", "error", "--limit-concurrency", "128"],
                               cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    body = {"model": "fixture-model", "messages": [{"role": "user", "content": "What is photosynthesis?"}],
            "temperature": 0, "max_tokens": 64}
    def call(path, *, token=None, method="GET", data=None, extra=None):
        headers = {"Content-Type": "application/json", **(extra or {})}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = Request(f"http://127.0.0.1:{port}" + path, method=method, headers=headers,
                      data=json.dumps(data).encode() if data is not None else None)
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(req, timeout=10) as resp:
                return resp.status, dict(resp.headers), resp.read().decode()
        except HTTPError as exc:
            with exc:
                return exc.code, dict(exc.headers), exc.read().decode()
    def complete(token, expected=None, extra=None):
        status, headers, payload = call("/v1/chat/completions", token=token, method="POST", data=body, extra=extra)
        lower = {k.lower(): v for k, v in headers.items()}
        assert status == 200, (status, payload)
        assert lower["x-evaluation-mode"] == "simulated_fixture"
        if expected:
            assert lower["x-cache"] == expected, lower
        assert json.loads(payload)["choices"][0]["message"]["content"].startswith("SIMULATED FIXTURE:")
        return lower["x-cache"]
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise RuntimeError("gateway startup failed: " + process.stderr.read().decode()[-2000:])
            try:
                if call("/health")[0] == 200:
                    break
            except (URLError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("gateway startup timed out")
            time.sleep(0.05)
        assert call("/v1/chat/completions", method="POST", data=body)[0] == 401
        statuses = [complete(token_a, "MISS"), complete(token_a, "HIT"), complete(token_b, "MISS")]
        barrier = threading.Barrier(50)
        def burst(index):
            barrier.wait(timeout=10)
            return complete(token_a if index % 2 else token_b, "HIT")
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(burst, i) for i in range(50)]
            statuses += [f.result(timeout=15) for f in futures]
        statuses.append(complete(token_a, "BYPASS", {"X-Cache-Mode": "bypass"}))
        status, _, payload = call("/v1/cache", token=token_a, method="DELETE")
        assert status == 200 and json.loads(payload)["removed"] == 1
        statuses += [complete(token_b, "HIT"), complete(token_a, "MISS")]
        status, _, metrics = call("/metrics", token=admin)
        assert status == 200 and "cache_gateway_upstream_attempts_total 4\n" in metrics
        assert "cache_gateway_upstream_errors_total 0\n" in metrics
        assert call("/metrics", token=token_a)[0] == 401
        status, headers, stream = call("/v1/chat/completions",token=token_a,method="POST",data={**body,"stream":True})
        assert status==200 and stream.endswith("data: [DONE]\n\n")
        assert {k.lower():v for k,v in headers.items()}["x-cache"]=="BYPASS"
        statuses.append("BYPASS")
        status, _, metrics = call("/metrics",token=admin)
        assert "cache_gateway_stream_completed_total 1\n" in metrics
        assert "cache_gateway_upstream_attempts_total 5\n" in metrics
        report = {"evaluation_mode": "simulated_functional_smoke", "transport": "loopback HTTP with uvicorn",
                  "provider": "deterministic fixture, not an LLM", "external_model_calls": 0,
                  "authenticated_completion_requests": len(statuses), "concurrent_clients": 50,
                  "concurrency_method": "50 client threads released through a barrier; server peak concurrency not measured",
                  "cache_status_counts": dict(Counter(statuses)), "fixture_upstream_attempts": 5, "streaming_completion": "passed",
                  "tenant_isolation": "passed", "tenant_invalidation": "passed",
                  "authentication": "passed", "admin_metrics_access": "passed",
                  "semantic_accuracy_measured": False, "real_model_savings_measured": False}
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
        print(rendered, end="")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


if __name__ == "__main__":
    main()
