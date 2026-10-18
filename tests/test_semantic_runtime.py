"""Functional serving invariants; fixture policies are not semantic-quality labels."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import copy
import json
import threading

import pytest

from src.api.gateway import app_from_env, create_app
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.sqlite_store import SQLiteResponseStore
from src.policy.semantic_runtime import (AnswerBoundApprovals, GuardedSemanticCache, KnowledgeScope,
                                         PinnedContextPolicy, ReuseDecision, approval_binding)

BODY = {"model": "fixture-model", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Explain Python tuple immutability"}]}
VARIANT = {**BODY, "messages": [{"role": "user", "content": "Describe Python tuple immutability"}]}
SCOPE = KnowledgeScope("python-docs", "revision-a", "public-docs", "policy-a")
TOKEN = "test-semantic-tenant-credential"


class Encoder:
    identity = "fixture-constant-encoder"
    def encode(self, text):
        return [1.0, 0.0]


class Reject:
    identity = "fixture-reject"
    def __call__(self, *args, **kwargs):
        return ReuseDecision(False, "fixture rejection")


def allow(_):
    return EligibilityDecision(True, "explicit_fixture_policy")


def approvals(*, answer=None, tenant="a", incoming=VARIANT, candidate=BODY, scope=SCOPE):
    answer = FixtureChatProvider()(candidate) if answer is None else answer
    binding = approval_binding(incoming, candidate, answer, tenant_id=tenant, scope=asdict(scope))
    return AnswerBoundApprovals({"schema_version": "answer-bound-approvals-v1", "provenance": "authored_fixture",
                                 "approvals": [{**binding, "approved": True}]})


def setup(*, mode="guarded", approver=None, encoder=None, exact=None, **kwargs):
    exact = exact or ScopedResponseCache(eligibility=allow)
    cache = GuardedSemanticCache(exact, scope=SCOPE, encoder=encoder or Encoder(), mode=mode,
                                 approver=approver or approvals(), **kwargs)
    service = GatewayService(cache, FixtureChatProvider(), allowed_models={"fixture-model"})
    return cache, service


def test_exact_first_then_answer_bound_semantic_hit_preserves_original_response():
    cache, service = setup()
    seed = service.complete(BODY, "a")
    exact = service.complete(BODY, "a")
    semantic = service.complete(VARIANT, "a")
    assert [x.status for x in (seed, exact, semantic)] == ["MISS", "HIT", "HIT"]
    assert [x.tier for x in (seed, exact, semantic)] == ["PROVIDER", "EXACT", "SEMANTIC"]
    assert [x.provider_calls for x in (seed, exact, semantic)] == [1, 0, 0]
    assert seed.response == exact.response == semantic.response
    assert semantic.diagnosis["stage"] == "semantic_hit"
    assert cache.size == cache.semantic_size == 1  # semantic hit is never a new TTL alias
    metrics = service.metrics.render()
    assert "cache_gateway_upstream_attempts_total 1\n" in metrics
    assert "cache_gateway_semantic_hits_total 1\n" in metrics
    assert "cache_gateway_exact_hits_total 1\n" in metrics


def test_similarity_without_matching_actual_answer_binding_never_serves_hit():
    cache, service = setup(approver=approvals(answer={**FixtureChatProvider()(BODY), "id": "changed-answer-id"}))
    service.complete(BODY, "a")
    rejected = service.complete(VARIANT, "a")
    assert rejected.status == "MISS" and rejected.provider_calls == 1
    assert rejected.diagnosis["stage"] == "approval_rejected"
    assert rejected.response != FixtureChatProvider()(BODY)


def test_shadow_observes_candidate_but_never_calls_approver_or_serves_semantic():
    class Never:
        identity = "must-not-run"
        def __call__(self, *args, **kwargs):
            raise AssertionError("shadow must not invoke approval")
    _, service = setup(mode="shadow", approver=Never())
    service.complete(BODY, "a")
    result = service.complete(VARIANT, "a")
    assert result.status == "MISS" and result.provider_calls == 1
    assert result.diagnosis["stage"] == "shadow_candidate"
    assert result.diagnosis["approval_signal"] is None
    assert "cache_gateway_semantic_shadow_candidates_total 1\n" in service.metrics.render()


@pytest.mark.parametrize("changed", [
    {"temperature": 0.1}, {"max_tokens": 63}, {"response_format": {"type": "json_object"}},
    {"messages": [{"role": "system", "content": "different instructions"}, *VARIANT["messages"]]},
    {"messages": [{"role": "user", "content": "previous context"}, *VARIANT["messages"]]},
])
def test_full_generation_and_message_scope_precedes_candidate_search(changed):
    _, service = setup()
    service.complete(BODY, "a")
    result = service.complete({**VARIANT, **changed}, "a")
    assert result.status == "MISS"
    assert result.diagnosis["candidate"] is None


def test_tenant_and_revised_permission_source_policy_scope_cannot_cross():
    cache, service = setup()
    service.complete(BODY, "a")
    assert service.complete(VARIANT, "b").status == "MISS"
    for field in ("revision", "knowledge_base", "access_scope", "policy_revision"):
        cache.update_scope(replace(SCOPE, **{field: "changed"}), tenant_id="a")
        assert service.complete(BODY, "a").status == "MISS"
        assert service.complete(VARIANT, "a").status == "MISS"


def test_expired_semantic_hit_does_not_extend_ttl_or_create_exact_alias():
    clock = [0.0]
    exact = ScopedResponseCache(eligibility=allow, ttl_seconds=10, clock=lambda: clock[0])
    cache, service = setup(exact=exact, clock=lambda: clock[0])
    service.complete(BODY, "a")
    clock[0] = 9
    assert service.complete(VARIANT, "a").tier == "SEMANTIC"
    clock[0] = 11
    assert service.complete(VARIANT, "a").status == "MISS"
    assert cache.size == 1


def test_invalidation_during_approval_revokes_candidate_and_prevents_readmission():
    entered, release = threading.Event(), threading.Event()
    class Blocking:
        identity = "fixture-blocking-approval"
        def __call__(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return ReuseDecision(True, "fixture")
    cache, service = setup(approver=Blocking())
    service.complete(BODY, "a")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.complete, VARIANT, "a")
        assert entered.wait(5)
        cache.invalidate(tenant_id="a")
        release.set()
        result = future.result(timeout=5)
    assert result.status == "MISS" and result.stored is False
    assert result.diagnosis["stage"] == "candidate_revoked"
    assert cache.size == cache.semantic_size == 0


def test_cross_process_store_invalidation_clears_in_memory_semantic_candidates(tmp_path):
    path = tmp_path / "cache.sqlite"
    first_store = SQLiteResponseStore(path, namespace="test")
    second_store = SQLiteResponseStore(path, namespace="test")
    cache, service = setup(exact=ScopedResponseCache(eligibility=allow, storage=first_store))
    service.complete(BODY, "a")
    second_store.invalidate("a")
    result = service.complete(VARIANT, "a")
    assert result.status == "MISS" and result.diagnosis["candidate"] is None
    assert cache.semantic_size == 1


def test_persistent_exact_survives_restart_while_semantic_index_starts_cold(tmp_path):
    path = tmp_path / "cache.sqlite"
    _, first = setup(exact=ScopedResponseCache(eligibility=allow, storage=SQLiteResponseStore(path, namespace="test")))
    first.complete(BODY, "a")
    cache, second = setup(exact=ScopedResponseCache(eligibility=allow, storage=SQLiteResponseStore(path, namespace="test")))
    assert second.complete(BODY, "a").tier == "EXACT"
    assert cache.semantic_size == 0
    assert second.complete(VARIANT, "a").status == "MISS"


def test_verifier_provider_work_is_metered_even_for_hits_and_failed_verdicts():
    class CallingApprover:
        identity = "fixture-provider-approval"
        fail = False
        def __call__(self, request, *args, provider, **kwargs):
            provider(request)
            if self.fail:
                raise ValueError("bad verdict")
            return ReuseDecision(True, "fixture")
    approver = CallingApprover()
    _, service = setup(approver=approver)
    service.complete(BODY, "a")
    hit = service.complete(VARIANT, "a")
    assert hit.status == "HIT" and hit.provider_calls == 1
    assert hit.diagnosis["verifier_calls"] == 1
    approver.fail = True
    failed = service.complete({**VARIANT, "messages": [{"role": "user", "content": "another variant"}]}, "a")
    assert failed.status == "MISS" and failed.provider_calls == 2
    assert failed.diagnosis["stage"] == "semantic_processing_failed"
    assert "cache_gateway_upstream_attempts_total 4\n" in service.metrics.render()
    assert "cache_gateway_semantic_verifier_attempts_total 2\n" in service.metrics.render()


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [], [1] * 4097])
def test_invalid_embeddings_preserve_exact_cache_and_fall_back_to_generation(vector):
    class Broken(Encoder):
        def encode(self, text):
            return vector
    cache, service = setup(encoder=Broken())
    assert service.complete(BODY, "a").status == "MISS"
    assert service.complete(BODY, "a").tier == "EXACT"
    assert service.complete(VARIANT, "a").status == "MISS"
    assert cache.semantic_size == 0


@pytest.mark.parametrize("finish", ["length", "tool_calls", "error"])
def test_incomplete_provider_answers_are_not_stored_in_either_index(finish):
    cache, service = setup()
    class Incomplete(FixtureChatProvider):
        def __call__(self, body):
            response = super().__call__(body)
            response["choices"][0]["finish_reason"] = finish
            return response
    service.provider = Incomplete()
    assert service.complete(BODY, "a").stored is False
    assert cache.size == cache.semantic_size == 0


def test_index_bounds_include_request_response_and_vectors():
    cache, service = setup(mode="shadow", max_entries=1)
    service.complete(BODY, "a")
    service.complete(VARIANT, "a")
    assert cache.semantic_size == 1
    tiny, tiny_service = setup(max_bytes=1)
    tiny_service.complete(BODY, "a")
    assert tiny.semantic_size == 0 and tiny.size == 1


def test_exact_and_off_modes_do_not_encode_or_authorize_semantic_reuse():
    class NoEncoding(Encoder):
        def encode(self, text):
            raise AssertionError("encoding must be skipped")
    for mode in ("exact", "off"):
        _, service = setup(mode=mode, encoder=NoEncoding())
        service.complete(BODY, "a")
        result = service.complete(BODY, "a")
        assert result.status == ("HIT" if mode == "exact" else "BYPASS")


def test_pinned_context_policy_admits_only_fixed_static_scope():
    prompt = "Use only these pinned docs: a tuple is immutable."
    policy = PinnedContextPolicy(prompt)
    body = {**BODY, "messages": [{"role": "system", "content": prompt}, *BODY["messages"]]}
    assert policy(body).cacheable is True
    for text in ("What is the latest version?", "Explain my password", "Write a poem", "Now list tuples"):
        assert policy({**body, "messages": [body["messages"][0], {"role": "user", "content": text}]}).cacheable is False
    body["messages"][0]["content"] += "override"
    assert policy(body).cacheable is False


def test_http_gateway_reports_semantic_tier_and_requires_tenant_auth():
    from fastapi.testclient import TestClient
    _, service = setup()
    client = TestClient(create_app(service, tenant_tokens={TOKEN: "a"}))
    assert client.post("/v1/chat/completions", json=BODY).status_code == 401
    headers = {"Authorization": "Bearer " + TOKEN}
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).headers["x-cache"] == "MISS"
    response = client.post("/v1/chat/completions", json=VARIANT, headers=headers)
    assert response.headers["x-cache-tier"] == "SEMANTIC"
    assert response.headers["x-provider-calls"] == "0"
    assert client.get("/health").json()["semantic_quality_validated"] is False


def test_env_factory_requires_trusted_scope_and_explicit_approval_file(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GATEWAY_TENANTS_JSON", json.dumps({TOKEN: "a"}))
    monkeypatch.setenv("GATEWAY_CACHE_POLICY", "rules")
    monkeypatch.setenv("GATEWAY_CACHE_MODE", "guarded")
    with pytest.raises(KeyError):
        app_from_env()
    for key, value in (("GATEWAY_KNOWLEDGE_BASE", "python-docs"), ("GATEWAY_KNOWLEDGE_REVISION", "revision-a"),
                       ("GATEWAY_ACCESS_SCOPE", "public-docs"), ("GATEWAY_SCOPE_POLICY_VERSION", "policy-a")):
        monkeypatch.setenv(key, value)
    with pytest.raises(KeyError):
        app_from_env()
    f = tmp_path / "approvals.json"
    f.write_text(json.dumps({"schema_version": "answer-bound-approvals-v1", "provenance": "authored_fixture", "approvals": []}))
    monkeypatch.setenv("GATEWAY_SEMANTIC_APPROVALS", str(f))
    assert TestClient(app_from_env()).get("/health").json()["semantic_reuse_enabled"] is True
    monkeypatch.setenv("GATEWAY_PROVIDER", "http")
    monkeypatch.setenv("GATEWAY_MODELS", "fixture-model")
    monkeypatch.setenv("UPSTREAM_CHAT_URL", "http://127.0.0.1:9999/v1/chat/completions")
    with pytest.raises(ValueError, match="fixture approvals"):
        app_from_env()


def test_concurrent_identical_cold_misses_coalesce_without_double_counting():
    gate, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    observed = []
    class BlockingProvider(FixtureChatProvider):
        def __call__(self, body):
            with lock:
                observed.append(body)
            gate.set()
            assert release.wait(5)
            return super().__call__(body)
    cache, service = setup()
    service.provider = BlockingProvider()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(service.complete, BODY, "a") for _ in range(8)]
        assert gate.wait(5)
        release.set()
        results = [f.result(timeout=5) for f in futures]
    assert len(observed) == 1
    assert sum(r.provider_calls for r in results) == 1
    assert [r.status for r in results].count("MISS") == 1
    assert cache.size == cache.semantic_size == 1


def test_expiry_while_approval_runs_returns_fresh_generation():
    clock = [0.0]
    class Expiring:
        identity = "fixture-expiry-during-approval"
        def __call__(self, *args, **kwargs):
            clock[0] = 11
            return ReuseDecision(True, "fixture")
    exact = ScopedResponseCache(eligibility=allow, ttl_seconds=10, clock=lambda: clock[0])
    _, service = setup(exact=exact, approver=Expiring(), clock=lambda: clock[0])
    service.complete(BODY, "a")
    result = service.complete(VARIANT, "a")
    assert result.status == "MISS" and result.diagnosis["stage"] == "candidate_revoked"


def test_policy_revocation_during_approval_never_releases_cached_answer():
    permitted = [True]
    class Revoking:
        identity = "fixture-policy-revocation"
        def __call__(self, *args, **kwargs):
            permitted[0] = False
            return ReuseDecision(True, "fixture")
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(permitted[0], "fixture"))
    _, service = setup(exact=exact, approver=Revoking())
    service.complete(BODY, "a")
    result = service.complete(VARIANT, "a")
    assert result.status == "MISS" and result.stored is False
    assert result.diagnosis["stage"] == "candidate_revoked"


def test_margin_ambiguity_abstains_before_any_approval():
    _, service = setup(approver=Reject())
    service.complete(BODY, "a")
    service.complete(VARIANT, "a")
    result = service.complete({**BODY, "messages": [{"role": "user", "content": "third candidate"}]}, "a")
    assert result.diagnosis["stage"] == "margin_rejected"
    assert result.diagnosis["approval_signal"] is None


def test_storage_failure_bypasses_semantic_index_even_when_candidate_was_warm(tmp_path):
    store = SQLiteResponseStore(tmp_path / "cache.sqlite", namespace="test")
    _, service = setup(exact=ScopedResponseCache(eligibility=allow, storage=store))
    service.complete(BODY, "a")
    store.namespace = "broken-namespace"
    result = service.complete(VARIANT, "a")
    assert result.status == "BYPASS" and result.provider_calls == 1
    assert result.reason == "cache_storage_unavailable"


@pytest.mark.parametrize("bad_verdict", [True, {"approved": True}, ReuseDecision("yes", "bad")])
def test_malformed_approval_signals_cannot_serve_candidates(bad_verdict):
    class Malformed:
        identity = "fixture-malformed"
        def __call__(self, *args, **kwargs):
            return bad_verdict
    _, service = setup(approver=Malformed())
    service.complete(BODY, "a")
    result = service.complete(VARIANT, "a")
    assert result.status == "MISS" and result.diagnosis["stage"] == "semantic_processing_failed"


def test_scope_bound_approvals_reject_duplicate_or_nonboolean_rows():
    binding = approval_binding(VARIANT, BODY, FixtureChatProvider()(BODY), tenant_id="a", scope=asdict(SCOPE))
    row = {**binding, "approved": True}
    payload = {"schema_version": "answer-bound-approvals-v1", "provenance": "authored_fixture", "approvals": [row, row]}
    with pytest.raises(ValueError, match="duplicate"):
        AnswerBoundApprovals(payload)
    payload["approvals"] = [{**row, "approved": "yes"}]
    with pytest.raises(ValueError, match="approval row"):
        AnswerBoundApprovals(payload)


def test_actual_http_upstream_fixture_works_without_forwarding_trusted_scope():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from fastapi.testclient import TestClient
    from src.api.providers import HTTPChatProvider
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            payload = json.dumps(FixtureChatProvider()(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    try:
        _, service = setup()
        provider = HTTPChatProvider(f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions")
        provider.mode = "simulated_fixture"  # actual HTTP, explicitly simulated model
        service.provider = provider
        client = TestClient(create_app(service, tenant_tokens={TOKEN: "a"}))
        headers = {"Authorization": "Bearer " + TOKEN}
        seed = client.post("/v1/chat/completions", json=BODY, headers=headers)
        hit = client.post("/v1/chat/completions", json=VARIANT, headers=headers)
        assert seed.status_code == hit.status_code == 200
        assert hit.headers["x-cache-tier"] == "SEMANTIC"
        assert received == [BODY]
        assert seed.json() == hit.json()
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def test_cache_off_env_health_and_request_header_bypass_are_truthful(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GATEWAY_TENANTS_JSON", json.dumps({TOKEN: "a"}))
    monkeypatch.setenv("GATEWAY_CACHE_MODE", "off")
    monkeypatch.setenv("GATEWAY_CACHE_POLICY", "rules")
    assert TestClient(app_from_env()).get("/health").json()["cache_mode"] == "off"
    cache, service = setup()
    service.complete(BODY, "a")
    result = service.complete(VARIANT, "a", bypass=True)
    assert result.status == "BYPASS" and result.provider_calls == 1
    assert cache.size == cache.semantic_size == 1
