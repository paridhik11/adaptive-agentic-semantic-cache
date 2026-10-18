"""Exercise the full R3 lifecycle over loopback HTTP with fixture answers only."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api.providers import FixtureChatProvider
from src.policy.semantic_runtime import KnowledgeScope, approval_binding, digest

PROMPT = "Answer using only this fixed fixture reference: Python tuples are immutable sequences. State when evidence is missing."
SEED = {"model": "fixture-model", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": "Explain Python tuple immutability."}]}
VARIANT = {**SEED, "messages": [SEED["messages"][0], {"role": "user", "content": "Describe Python tuple immutability."}]}
SCOPE = KnowledgeScope("fixture-python-reference", "fixture-revision-a", "public-fixture", "fixture-policy-a")


@contextmanager
def server(workdir, *, mode, tokens, admin, revision=SCOPE.revision):
    workdir.mkdir(exist_ok=True)
    (workdir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    binding = approval_binding(VARIANT, SEED, FixtureChatProvider()(SEED), tenant_id="smoke-a", scope=asdict(SCOPE))
    (workdir / "approvals.json").write_text(json.dumps({"schema_version": "answer-bound-approvals-v1",
         "provenance": "authored_fixture", "approvals": [{**binding, "approved": True}]}), encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "GATEWAY_PROVIDER": "fixture", "GATEWAY_CACHE_POLICY": "pinned_context",
           "GATEWAY_CACHE_MODE": mode, "GATEWAY_TENANTS_JSON": json.dumps(tokens), "GATEWAY_ADMIN_TOKEN": admin,
           "GATEWAY_DOCUMENT_PROMPT_FILE": str(workdir / "prompt.txt"), "GATEWAY_KNOWLEDGE_BASE": SCOPE.knowledge_base,
           "GATEWAY_KNOWLEDGE_REVISION": revision, "GATEWAY_ACCESS_SCOPE": SCOPE.access_scope,
           "GATEWAY_SCOPE_POLICY_VERSION": SCOPE.policy_revision, "GATEWAY_SEMANTIC_ENCODER": "lexical",
           "GATEWAY_SEMANTIC_THRESHOLD": "0.5", "GATEWAY_SEMANTIC_MARGIN": "0.05",
           "GATEWAY_SEMANTIC_APPROVALS": str(workdir / "approvals.json"),
           "GATEWAY_CACHE_DB": str(workdir / "cache.sqlite"), "GATEWAY_CACHE_VERSION": "r3-fixture-v1",
           "GATEWAY_TTL_SECONDS": "3600", "GATEWAY_MAX_ENTRIES": "1024", "GATEWAY_MAX_CACHE_BYTES": "67108864",
           "GATEWAY_MAX_OUTPUT_TOKENS": "4096", "GATEWAY_MAX_INFLIGHT": "32",
           "GATEWAY_SEMANTIC_MAX_ENTRIES": "256", "GATEWAY_SEMANTIC_MAX_BYTES": "8388608"}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.gateway:app_from_env", "--factory",
                                "--host", "127.0.0.1", "--port", str(port), "--no-access-log", "--log-level", "error"],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    def call(path, *, token=None, method="GET", body=None, bypass=False):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if bypass:
            headers["X-Cache-Mode"] = "bypass"
        request = Request(f"http://127.0.0.1:{port}" + path, method=method, headers=headers,
                          data=json.dumps(body).encode() if body is not None else None)
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read().decode()
        except HTTPError as error:
            with error:
                return error.code, dict(error.headers), error.read().decode()
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise RuntimeError("fixture server startup failed: " + process.stderr.read().decode()[-2000:])
            try:
                if call("/health")[0] == 200:
                    break
            except (URLError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("fixture server startup timeout")
            time.sleep(0.05)
        yield call
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def run():
    token_a, token_b, admin = (secrets.token_urlsafe(32) for _ in range(3))
    tokens = {token_a: "smoke-a", token_b: "smoke-b"}
    rows, checks = [], {}
    def complete(call, label, body=SEED, *, token=token_a, expected, tier, bypass=False):
        status, headers, text = call("/v1/chat/completions", token=token, method="POST", body=body, bypass=bypass)
        assert status == 200, (label, status)
        response = json.loads(text)
        assert response["choices"][0]["message"]["content"].startswith("SIMULATED FIXTURE:")
        assert headers["x-evaluation-mode"] == "simulated_fixture"
        assert headers["x-cache"] == expected, (label, headers)
        assert headers["x-cache-tier"] == tier
        rows.append({"case": label, "status": expected, "tier": tier,
                     "provider_attempts": int(headers["x-provider-calls"]), "response_sha256": digest(response)})
        return response
    with tempfile.TemporaryDirectory(prefix="r3-semantic-smoke-") as temp:
        root = Path(temp)
        with server(root / "off", mode="off", tokens=tokens, admin=admin) as call:
            complete(call, "cache_off_seed", expected="BYPASS", tier="PROVIDER")
            complete(call, "cache_off_repeat", expected="BYPASS", tier="PROVIDER")
        with server(root / "exact", mode="exact", tokens=tokens, admin=admin) as call:
            complete(call, "exact_seed", expected="MISS", tier="PROVIDER")
            complete(call, "exact_repeat", expected="HIT", tier="EXACT")
            complete(call, "exact_nonidentical", VARIANT, expected="MISS", tier="PROVIDER")
        with server(root / "exact", mode="exact", tokens=tokens, admin=admin) as call:
            complete(call, "persistent_exact_after_restart", expected="HIT", tier="EXACT")
            checks["persistent_exact_restart"] = "passed"
        with server(root / "shadow", mode="shadow", tokens=tokens, admin=admin) as call:
            complete(call, "shadow_seed", expected="MISS", tier="PROVIDER")
            complete(call, "shadow_nonidentical", VARIANT, expected="MISS", tier="PROVIDER")
            metrics = call("/metrics", token=admin)[2]
            assert "cache_gateway_semantic_shadow_candidates_total 1\n" in metrics
            checks["shadow_records_without_serving"] = "passed"
        with server(root / "guarded", mode="guarded", tokens=tokens, admin=admin) as call:
            assert call("/v1/chat/completions", method="POST", body=SEED)[0] == 401
            seed = complete(call, "guarded_seed", expected="MISS", tier="PROVIDER")
            complete(call, "guarded_exact", expected="HIT", tier="EXACT")
            semantic = complete(call, "guarded_approved_nonidentical", VARIANT, expected="HIT", tier="SEMANTIC")
            assert seed == semantic
            near_miss = {**VARIANT, "messages": [SEED["messages"][0], {"role": "user", "content": "Describe Python tuple mutability."}]}
            complete(call, "guarded_unapproved_near_miss", near_miss, expected="MISS", tier="PROVIDER")
            complete(call, "other_tenant", VARIANT, token=token_b, expected="MISS", tier="PROVIDER")
            complete(call, "explicit_bypass", bypass=True, expected="BYPASS", tier="PROVIDER")
            assert call("/v1/cache", token=token_a, method="DELETE")[0] == 200
            complete(call, "after_tenant_invalidation", expected="MISS", tier="PROVIDER")
            checks.update(authentication="passed", answer_binding="passed", tenant_isolation="passed", invalidation="passed")
        with server(root / "guarded", mode="guarded", tokens=tokens, admin=admin, revision="fixture-revision-b") as call:
            complete(call, "source_revision_changed", expected="MISS", tier="PROVIDER")
            complete(call, "old_approval_under_new_revision", VARIANT, expected="MISS", tier="PROVIDER")
            checks["source_revision_isolation"] = "passed"
    return {"schema_version": "r3-semantic-serving-smoke-v1", "evaluation_mode": "simulated_functional_smoke",
            "transport": "loopback HTTP; separate uvicorn processes; SQLite exact persistence",
            "provider": "deterministic fixture, not an LLM", "encoder": "lexical hashing baseline",
            "approvals": "authored fixture bindings; not independently reviewed answer labels",
            "external_model_calls": 0, "semantic_quality_measured": False, "live_latency_or_savings_measured": False,
            "checks": checks, "completion_requests": len(rows),
            "fixture_provider_attempts": sum(row["provider_attempts"] for row in rows),
            "status_counts": dict(Counter(row["status"] for row in rows)), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("existing evidence is not overwritten; choose a new output")
    report = run()
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
