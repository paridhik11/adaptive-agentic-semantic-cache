"""HTTP boundary, provider work accounting, and cross-tenant regressions."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading

import pytest
from fastapi.testclient import TestClient

from src.api.gateway import app_from_env, create_app
from src.api.providers import FixtureChatProvider, ProviderError
from src.api.service import ConservativeRulesPolicy, GatewayBusy, GatewayService
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache

TOKEN_A = "test-fixture-tenant-a-credential"
TOKEN_B = "test-fixture-tenant-b-credential"
ADMIN = "test-fixture-admin-credential"
BODY = {"model": "fixture-model", "messages": [{"role": "user", "content": "What is photosynthesis?"}],
        "temperature": 0, "max_tokens": 64}


def setup(provider=None, policy=None, **kwargs):
    cache = ScopedResponseCache(eligibility=policy or ConservativeRulesPolicy())
    service = GatewayService(cache, provider or FixtureChatProvider(), allowed_models={"fixture-model"}, **kwargs)
    app = create_app(service, tenant_tokens={TOKEN_A: "tenant-a", TOKEN_B: "tenant-b"}, admin_token=ADMIN)
    return service, TestClient(app)


def headers(token=TOKEN_A):
    return {"Authorization": "Bearer " + token}


def send(client, body=None, token=TOKEN_A, **extra):
    return client.post("/v1/chat/completions", json=BODY if body is None else body,
                       headers={**headers(token), **extra})


def test_exact_hit_preserves_answer_but_does_not_double_charge_usage():
    service, client = setup()
    first, second = send(client), send(client)
    assert first.status_code == second.status_code == 200
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert first.json() == second.json()
    assert second.headers["x-provider-calls"] == "0"
    assert second.headers["x-evaluation-mode"] == "simulated_fixture"
    metrics = client.get("/metrics", headers=headers(ADMIN)).text
    assert "cache_gateway_upstream_attempts_total 1\n" in metrics
    assert f'cache_gateway_upstream_prompt_tokens_total {first.json()["usage"]["prompt_tokens"]}\n' in metrics
    assert 'cache_gateway_processing_seconds_count{status="HIT"} 1' in metrics


def test_authenticated_tenants_cannot_spoof_scope_and_invalidate_each_other():
    _, client = setup()
    assert send(client).headers["x-cache"] == "MISS"
    assert send(client, token=TOKEN_B, **{"X-Tenant-ID": "tenant-a"}).headers["x-cache"] == "MISS"
    assert client.delete("/v1/cache", headers=headers()).json() == {"removed": 1}
    assert send(client, token=TOKEN_B).headers["x-cache"] == "HIT"
    assert send(client).headers["x-cache"] == "MISS"


def test_auth_required_and_metrics_are_admin_only():
    service, client = setup()
    assert client.post("/v1/chat/completions", json=BODY).status_code == 401
    assert send(client, token="wrong-test-credential").status_code == 401
    assert client.delete("/v1/cache").status_code == 401
    assert client.get("/metrics", headers=headers()).status_code == 401
    assert client.get("/metrics", headers=headers(ADMIN)).status_code == 200
    assert service.cache.size == 0


def test_duplicate_auth_headers_are_rejected():
    _, client = setup()
    response = client.post("/v1/chat/completions", json=BODY,
                           headers=[("Authorization", "Bearer " + TOKEN_A), ("Authorization", "Bearer " + TOKEN_B)])
    assert response.status_code == 401


def test_bypass_header_executes_provider_without_invalidating_existing_cache():
    _, client = setup()
    assert send(client).headers["x-cache"] == "MISS"
    response = send(client, **{"X-Cache-Mode": "bypass"})
    assert response.headers["x-cache"] == "BYPASS"
    assert response.headers["x-provider-calls"] == "1"
    assert send(client).headers["x-cache"] == "HIT"


@pytest.mark.parametrize("query", ["What is the weather today?", "Write a poem about gravity",
                                   "What is photosynthesis? Email me at person@example.org",
                                   "What is photosynthesis? " + "word " * 129])
def test_dynamic_creative_personal_and_long_queries_bypass(query):
    _, client = setup()
    body = copy.deepcopy(BODY)
    body["messages"][0]["content"] = query
    assert send(client, body).headers["x-cache"] == "BYPASS"
    assert send(client, body).headers["x-cache"] == "BYPASS"


def test_context_default_bypass_and_custom_policy_still_isolates_full_history():
    _, client = setup()
    body = copy.deepcopy(BODY)
    body["messages"].insert(0, {"role": "system", "content": "Explain briefly"})
    assert send(client, body).headers["x-cache"] == "BYPASS"
    _, client = setup(policy=lambda _: EligibilityDecision(True, "fixture"))
    assert send(client, body).headers["x-cache"] == "MISS"
    assert send(client, body).headers["x-cache"] == "HIT"
    body["messages"][0]["content"] = "Explain in detail"
    assert send(client, body).headers["x-cache"] == "MISS"


@pytest.mark.parametrize("update", [{"stream": True, "stream_options":{"include_usage":False}}, {"stream": "false"}, {"n": True}, {"n": 2},
                                    {"model": "not-enabled"}, {"temperature": True}, {"top_p": 2},
                                    {"max_tokens": 999999}, {"max_tokens": 0}, {"tools": []},
                                    {"messages": []}, {"response_format": {"type": "unsupported"}},
                                    {"max_completion_tokens": 100}, {"seed": False}])
def test_invalid_requests_never_call_provider(update):
    service, client = setup()
    body = {**BODY, **update}
    assert send(client, body).status_code == 400
    assert "cache_gateway_upstream_attempts_total 0\n" in service.metrics.render()


@pytest.mark.parametrize("raw", ['{"model":"a","model":"b"}', '{"temperature":NaN}', '[1,2]', '{broken'])
def test_json_validation(raw):
    _, client = setup()
    assert client.post("/v1/chat/completions", content=raw,
                       headers={**headers(), "Content-Type": "application/json"}).status_code == 400


def test_actual_body_limit_does_not_trust_content_length():
    service, _ = setup()
    client = TestClient(create_app(service, tenant_tokens={TOKEN_A: "tenant-a"}, max_request_bytes=32))
    response = client.post("/v1/chat/completions", content=b"x" * 100,
                           headers={**headers(), "Content-Type": "application/json", "Content-Length": "1"})
    assert response.status_code == 413


def test_provider_failure_is_sanitized_counted_and_never_cached():
    def failing(_):
        raise ProviderError("upstream_rate_limited", 503)
    service, client = setup(provider=failing)
    assert send(client).status_code == 503
    assert send(client).status_code == 503
    assert service.cache.size == 0
    assert "cache_gateway_upstream_attempts_total 2\n" in service.metrics.render()
    assert "cache_gateway_upstream_errors_total 2\n" in service.metrics.render()


def test_unexpected_provider_error_does_not_expose_details():
    def failing(_):
        raise RuntimeError("credential-that-must-not-be-returned")
    _, client = setup(provider=failing)
    result = send(client)
    assert result.status_code == 502
    assert "credential-that-must-not-be-returned" not in result.text


def test_incomplete_responses_are_returned_but_not_admitted():
    def truncated(body):
        response = FixtureChatProvider()(body)
        response["choices"][0]["finish_reason"] = "length"
        return response
    service, client = setup(provider=truncated)
    assert send(client).headers["x-cache"] == "MISS"
    assert send(client).headers["x-cache"] == "MISS"
    assert service.cache.size == 0


def test_concurrent_identical_requests_make_one_provider_call():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed(body):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return FixtureChatProvider()(body)
    service, _ = setup(provider=delayed)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(service.complete, copy.deepcopy(BODY), "tenant-a") for _ in range(8)]
        assert entered.wait(5)
        release.set()
        results = [f.result(timeout=5) for f in futures]
    assert len(calls) == 1
    assert sum(r.status == "HIT" for r in results) == 7


def test_busy_limit_releases_capacity_after_failure():
    entered, release = threading.Event(), threading.Event()
    def failing(_):
        entered.set()
        assert release.wait(5)
        raise ProviderError("upstream_unavailable")
    service, _ = setup(provider=failing, max_inflight=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(service.complete, BODY, "tenant-a")
        assert entered.wait(5)
        try:
            with pytest.raises(GatewayBusy):
                service.complete(BODY, "tenant-b")
        finally:
            release.set()
        with pytest.raises(ProviderError):
            pending.result(timeout=5)
    service.provider = FixtureChatProvider()
    assert service.complete(BODY, "tenant-a").status == "MISS"


def test_env_factory_defaults_to_fixture_and_cache_disabled(monkeypatch):
    monkeypatch.setenv("GATEWAY_TENANTS_JSON", '{"' + TOKEN_A + '":"tenant-a"}')
    monkeypatch.delenv("GATEWAY_PROVIDER", raising=False)
    monkeypatch.delenv("GATEWAY_CACHE_POLICY", raising=False)
    client = TestClient(app_from_env())
    assert client.get("/health").json()["provider_mode"] == "simulated_fixture"
    assert send(client).headers["x-cache"] == "BYPASS"


def test_short_credentials_fail_at_startup():
    service, _ = setup()
    with pytest.raises(ValueError):
        create_app(service, tenant_tokens={"short": "tenant-a"})


def test_persistent_env_restart_and_version_guard(monkeypatch, tmp_path):
    monkeypatch.setenv('GATEWAY_TENANTS_JSON', '{"' + TOKEN_A + '":"tenant-a"}')
    monkeypatch.setenv('GATEWAY_PROVIDER', 'fixture')
    monkeypatch.setenv('GATEWAY_CACHE_POLICY', 'rules')
    monkeypatch.setenv('GATEWAY_CACHE_DB', str(tmp_path / 'answers.sqlite'))
    monkeypatch.delenv('GATEWAY_CACHE_VERSION', raising=False)
    with pytest.raises(ValueError, match='GATEWAY_CACHE_VERSION'):
        app_from_env()
    monkeypatch.setenv('GATEWAY_CACHE_VERSION', 'fixture-v1')
    assert send(TestClient(app_from_env())).headers['x-cache'] == 'MISS'
    assert send(TestClient(app_from_env())).headers['x-cache'] == 'HIT'
    monkeypatch.setenv('GATEWAY_CACHE_VERSION', 'fixture-v2')
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError, match='namespace mismatch'):
        app_from_env()


def test_failed_invalidation_returns_503(monkeypatch):
    import sqlite3
    service, client = setup()
    def fail(**kwargs):
        raise sqlite3.OperationalError('private database path')
    monkeypatch.setattr(service.cache, 'invalidate', fail)
    result = client.delete('/v1/cache', headers={'Authorization': 'Bearer ' + TOKEN_A})
    assert result.status_code == 503
    assert 'private database path' not in result.text


def test_dashboard_is_static_and_metrics_remain_protected():
    _, client = setup()
    page = client.get('/dashboard')
    assert page.status_code == 200
    assert 'Operational counters' in page.text
    assert 'localStorage' not in page.text
    assert TOKEN_A not in page.text
    assert "frame-ancestors 'none'" in page.headers['content-security-policy']
    assert page.headers['cache-control'] == 'no-store'
    assert client.get('/metrics').status_code == 401
