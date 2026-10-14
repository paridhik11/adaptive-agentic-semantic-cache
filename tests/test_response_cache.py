"""Hermetic runtime regressions; provider responses here are test fixtures.

These checks establish application behavior, not semantic accuracy or savings
on a real workload. They deliberately avoid the historical model harnesses.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.policy.response_cache import EligibilityDecision, ScopedResponseCache


def request():
    return {"model": "model-a", "temperature": 0.0, "messages": [
        {"role": "system", "content": "Answer precisely."},
        {"role": "user", "content": "What is binary search?"},
    ]}


def response(content="A search over a sorted collection."):
    return {"id": "fixture-generation", "model": "model-a", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content,
        }}], "usage": {"prompt_tokens": 10, "completion_tokens": 8}}


class Provider:
    def __init__(self, output=None):
        self.calls = 0
        self.output = response() if output is None else output

    def __call__(self, body):
        self.calls += 1
        return copy.deepcopy(self.output)


def stable(body):
    return EligibilityDecision(True, "stable_fixture")


def run(cache, body, provider, tenant="tenant-a"):
    return cache.process(body, tenant_id=tenant, provider=provider)


def test_miss_generates_and_hit_returns_complete_stored_answer():
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    first = run(cache, request(), provider)
    second = run(cache, request(), provider)
    assert (first.status, first.provider_calls, first.stored) == ("MISS", 1, True)
    assert (second.status, second.provider_calls) == ("HIT", 0)
    assert second.response == first.response == provider.output
    assert provider.calls == 1
    assert first.total_latency_ms >= 0 and second.total_latency_ms >= 0


def test_response_byte_budget_evicts_even_below_entry_limit():
    cache = ScopedResponseCache(eligibility=stable, max_entries=10, max_bytes=400)
    provider = Provider(response("a" * 100))
    first = request()
    second = {**request(), "seed": 2}
    assert run(cache, first, provider).stored
    assert run(cache, second, provider).stored
    assert cache.size == 1
    assert run(cache, second, provider).status == "HIT"
    assert run(cache, first, provider).status == "MISS"


def test_byte_budget_counts_utf8_and_does_not_admit_oversized_response():
    cache = ScopedResponseCache(eligibility=stable, max_bytes=400)
    provider = Provider(response("界" * 100))
    assert not run(cache, request(), provider).stored
    assert cache.size == 0
    assert run(cache, request(), provider).status == "MISS"


def test_invalidation_and_expiry_release_payload_budget():
    now = [0.0]
    cache = ScopedResponseCache(eligibility=stable, max_bytes=400, ttl_seconds=1, clock=lambda: now[0])
    provider = Provider(response("a" * 100))
    assert run(cache, request(), provider).stored
    assert cache.invalidate() == 1
    assert run(cache, request(), provider).stored
    now[0] = 2.0
    assert cache.size == 0
    assert run(cache, request(), provider).stored
    assert run(cache, request(), provider).status == "HIT"


def test_client_bypass_never_reads_or_admits_cache():
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    assert run(cache, request(), provider).stored
    bypassed = cache.process(request(), tenant_id="tenant-a", provider=provider, bypass=True)
    assert bypassed.status == "BYPASS" and bypassed.provider_calls == 1
    assert run(cache, request(), provider).status == "HIT"
    with pytest.raises(ValueError):
        cache.process(request(), tenant_id="tenant-a", provider=provider, bypass="true")


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_byte_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        ScopedResponseCache(eligibility=stable, max_bytes=budget)


def test_tenants_never_share_even_identical_requests():
    cache = ScopedResponseCache(eligibility=stable)
    a = Provider(response("Tenant A's answer"))
    b = Provider(response("Tenant B's answer"))
    run(cache, request(), a, "tenant-a")
    result = run(cache, request(), b, "tenant-b")
    assert result.status == "MISS"
    assert result.response["choices"][0]["message"]["content"] == "Tenant B's answer"
    assert run(cache, request(), a, "tenant-a").status == "HIT"


@pytest.mark.parametrize("field,value", [
    ("model", "model-b"), ("temperature", 0.1), ("top_p", 0.5),
    ("max_tokens", 100), ("seed", 42), ("stop", ["END"]),
    ("response_format", {"type": "json_object"}),
    ("metadata", {"feature_version": "changed"}),
])
def test_generation_parameters_and_other_request_fields_are_scoped(field, value):
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    body = request()
    run(cache, body, provider)
    changed = copy.deepcopy(body)
    changed[field] = value
    assert run(cache, changed, provider).status == "MISS"
    assert provider.calls == 2


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_full_context_changes_never_share_entries(role):
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    body = request()
    run(cache, body, provider)
    changed = copy.deepcopy(body)
    changed["messages"].insert(0, {"role": role, "content": "A different context."})
    assert run(cache, changed, provider).status == "MISS"


@pytest.mark.parametrize("query", ["What is Binary Search?", "binary search?", "Explain binary search."])
def test_no_unverified_normalization_shortcut(query):
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    body = request()
    run(cache, body, provider)
    body["messages"][-1]["content"] = query
    assert run(cache, body, provider).status == "MISS"


def test_dynamic_requests_bypass_lookup_and_admission():
    cache = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "dynamic"))
    provider = Provider()
    body = request()
    body["messages"][-1]["content"] = "What is the weather now?"
    assert run(cache, body, provider).status == "BYPASS"
    assert run(cache, body, provider).status == "BYPASS"
    assert provider.calls == 2 and cache.size == 0


def test_policy_revocation_prevents_an_existing_exact_hit():
    allowed = True
    cache = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(allowed, "revoked"))
    provider = Provider()
    run(cache, request(), provider)
    allowed = False
    assert run(cache, request(), provider).status == "BYPASS"
    assert provider.calls == 2


@pytest.mark.parametrize("decision", [None, True, {"cacheable": True}, EligibilityDecision(1, "invalid")])
def test_invalid_policy_outputs_fail_closed(decision):
    cache = ScopedResponseCache(eligibility=lambda _: decision)
    assert run(cache, request(), Provider()).status == "BYPASS"
    assert cache.size == 0


def test_policy_exception_does_not_approve_reuse():
    def broken(_):
        raise RuntimeError("policy unavailable")
    cache = ScopedResponseCache(eligibility=broken)
    result = run(cache, request(), Provider())
    assert result.status == "BYPASS"
    assert result.reason == "eligibility_policy_failed"


@pytest.mark.parametrize("temperature", [None, True, -1, 0.9])
def test_unknown_or_ineligible_temperature_bypasses(temperature):
    cache = ScopedResponseCache(eligibility=stable)
    body = request()
    if temperature is None:
        del body["temperature"]
    else:
        body["temperature"] = temperature
    assert run(cache, body, Provider()).status == "BYPASS"
    assert cache.size == 0


@pytest.mark.parametrize("field", ["tools", "functions"])
def test_tool_requests_not_cached(field):
    cache = ScopedResponseCache(eligibility=stable)
    body = request()
    body[field] = [{"name": "search"}]
    assert run(cache, body, Provider()).status == "BYPASS"


def test_multimodal_content_bypasses():
    cache = ScopedResponseCache(eligibility=stable)
    body = request()
    body["messages"][-1]["content"] = [{"type": "text", "text": "Hello"}]
    assert run(cache, body, Provider()).status == "BYPASS"


@pytest.mark.parametrize("kind", ["length", "tool_calls", "error", "empty", "refusal", "missing_model", "wrong_count"])
def test_only_complete_successful_text_responses_are_admitted(kind):
    output = response()
    if kind == "length":
        output["choices"][0]["finish_reason"] = "length"
    elif kind == "tool_calls":
        output["choices"][0]["message"]["tool_calls"] = [{"id": "tool-fixture"}]
    elif kind == "error":
        output["error"] = {"message": "failed"}
    elif kind == "empty":
        output["choices"][0]["message"]["content"] = ""
    elif kind == "refusal":
        output["choices"][0]["message"]["refusal"] = "refused"
    elif kind == "missing_model":
        del output["model"]
    else:
        output["choices"] = []
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider(output)
    assert not run(cache, request(), provider).stored
    assert run(cache, request(), provider).status == "MISS"
    assert provider.calls == 2 and cache.size == 0


def test_provider_failure_is_propagated_and_never_stored():
    cache = ScopedResponseCache(eligibility=stable)
    def broken(_):
        raise TimeoutError("upstream timed out")
    with pytest.raises(TimeoutError):
        run(cache, request(), broken)
    assert cache.size == 0
    assert run(cache, request(), Provider()).status == "MISS"


def test_request_and_response_mutations_cannot_change_stored_data():
    cache = ScopedResponseCache(eligibility=stable)
    body = request()
    def mutating_provider(snapshot):
        snapshot["messages"][-1]["content"] = "mutated"
        return response()
    first = run(cache, body, mutating_provider)
    first.response["choices"][0]["message"]["content"] = "corrupted by caller"
    assert body == request()
    hit = run(cache, request(), Provider())
    assert hit.status == "HIT" and hit.response == response()
    hit.response["usage"]["prompt_tokens"] = 999
    assert run(cache, request(), Provider()).response == response()


def test_expiry_is_checked_at_the_boundary_and_ttl_starts_after_generation():
    now = [0.0]
    cache = ScopedResponseCache(eligibility=stable, ttl_seconds=10, clock=lambda: now[0])
    def slow(_):
        now[0] = 100.0
        return response()
    run(cache, request(), slow)
    now[0] = 109.9
    assert run(cache, request(), Provider()).status == "HIT"
    now[0] = 110.0
    assert cache.size == 0
    assert run(cache, request(), Provider()).status == "MISS"


def test_capacity_is_bounded_and_recently_used_entries_survive_eviction():
    cache = ScopedResponseCache(eligibility=stable, max_entries=2)
    provider = Provider()
    a, b, c = request(), request(), request()
    b["messages"][-1]["content"] = "Query B"
    c["messages"][-1]["content"] = "Query C"
    run(cache, a, provider)
    run(cache, b, provider)
    assert run(cache, a, provider).status == "HIT"
    run(cache, c, provider)
    assert cache.size == 2
    assert run(cache, a, provider).status == "HIT"
    assert run(cache, b, provider).status == "MISS"


def test_tenant_invalidation_preserves_other_tenants():
    cache = ScopedResponseCache(eligibility=stable)
    provider = Provider()
    run(cache, request(), provider, "a")
    run(cache, request(), provider, "b")
    assert cache.invalidate(tenant_id="a") == 1
    assert run(cache, request(), provider, "b").status == "HIT"
    assert run(cache, request(), provider, "a").status == "MISS"
    assert cache.invalidate() == 2
    assert cache.size == 0


def test_parallel_identical_misses_make_one_provider_call():
    cache = ScopedResponseCache(eligibility=stable)
    started = threading.Event()
    release = threading.Event()
    calls = []
    def provider(_):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return response()
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(run, cache, request(), provider)
        assert started.wait(5)
        rest = [pool.submit(run, cache, request(), provider) for _ in range(7)]
        release.set()
        results = [first.result()] + [f.result() for f in rest]
    assert len(calls) == 1
    assert sum(r.provider_calls for r in results) == 1
    assert [r.status for r in results].count("HIT") == 7


def test_invalidation_during_generation_prevents_repopulation():
    cache = ScopedResponseCache(eligibility=stable)
    started, release = threading.Event(), threading.Event()
    def provider(_):
        started.set()
        assert release.wait(5)
        return response()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(run, cache, request(), provider)
        assert started.wait(5)
        assert cache.invalidate(tenant_id="tenant-a") == 0
        release.set()
        assert not pending.result().stored
    assert cache.size == 0
    assert run(cache, request(), Provider()).status == "MISS"


def test_policy_revocation_during_generation_prevents_admission():
    allowed = True
    cache = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(allowed, "revoked"))
    def provider(_):
        nonlocal allowed
        allowed = False
        return response()
    assert not run(cache, request(), provider).stored
    assert cache.size == 0


@pytest.mark.parametrize("tenant", ["", " ", None])
def test_tenant_is_required(tenant):
    with pytest.raises(ValueError):
        run(ScopedResponseCache(eligibility=stable), request(), Provider(), tenant)


def test_streaming_is_explicitly_unsupported():
    body = request()
    body["stream"] = True
    provider = Provider()
    with pytest.raises(ValueError, match="streaming"):
        run(ScopedResponseCache(eligibility=stable), body, provider)
    assert provider.calls == 0


def test_invalid_json_parameters_are_not_hashed_or_sent():
    body = request()
    body["temperature"] = float("nan")
    provider = Provider()
    with pytest.raises(ValueError):
        run(ScopedResponseCache(eligibility=stable), body, provider)
    assert provider.calls == 0
