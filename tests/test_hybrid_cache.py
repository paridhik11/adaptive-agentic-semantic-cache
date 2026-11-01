"""Comprehensive unit and safety tests for ConservativeHybridCache.

Verifies:
- Scoped exact-cache lookup
- Dense semantic top-k retrieval
- Context, direction, version and constraint guards
- Answer-aware verification
- Cache-entry-specific veto and approval decisions
- Candidate-promotion queue (unverified candidates never served)
- Explicit revocation and TTL handling
- Clean fallback on uncertainty
"""
import copy
from dataclasses import asdict
import re
import time
import pytest

from src.policy.encoders import DenseEncoder
from src.policy.hybrid_cache import (
    CandidatePromotionItem, CandidatePromotionQueue,
    ConservativeHybridCache, EntryReviewRecord,
)
from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION,
    PASSAGE_MODEL, PASSAGE_REVISION,
    LocalPassageScorer,
)
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope, digest


class MockPassageScorer:
    identity = "mock-passage-scorer-v1"
    def score(self, query: str, passage: str):
        # Word overlap based scoring using regex
        q_words = set(re.findall(r"\w+", query.lower()))
        p_words = set(re.findall(r"\w+", passage.lower()))
        overlap = len(q_words & p_words) / max(1, len(q_words))
        logit = 5.0 if overlap >= 0.30 else -5.0
        prob = 0.99 if overlap >= 0.30 else 0.01
        return {"logit": logit, "probability": prob, "input_tokens": 10}



class MockEncoder:
    identity = "mock-dense-encoder-v1"
    max_dimensions = 384
    def encode(self, text: str):
        # Deterministic 384-d normalized vector based on tokens
        vec = [0.0] * 384
        for w in re.findall(r"\w+", text.lower()):
            idx = int(digest(w)[:4], 16) % 384
            vec[idx] += 1.0
        norm = sum(x*x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else [1.0] + [0.0]*383



def make_request(query: str, system_prompt: str = "Answer accurately using documentation.") -> dict:
    return {
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    }


def make_response(answer: str) -> dict:
    return {
        "id": "resp-test",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": answer},
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
    }


@pytest.fixture
def hybrid_cache():
    exact = ScopedResponseCache(eligibility=lambda req: EligibilityDecision(True, "test"))
    scope = KnowledgeScope("test-corpus", "rev-1", "public", "test-p1")
    return ConservativeHybridCache(
        exact,
        scope=scope,
        encoder=MockEncoder(),
        passage_scorer=MockPassageScorer(),
        retrieval_threshold=0.50,
        verification_threshold=0.85,
    )


def test_exact_cache_hit_returns_zero_calls(hybrid_cache):
    req = make_request("How to convert a list to a tuple in Python?")
    resp = make_response("Use tuple(my_list) to create a tuple from a list.")

    provider_calls = 0
    def mock_provider(r):
        nonlocal provider_calls
        provider_calls += 1
        return resp

    # First call: cold miss -> populates exact cache and candidate
    res1 = hybrid_cache.process(req, tenant_id="t1", provider=mock_provider)
    assert res1.status == "MISS"
    assert provider_calls == 1

    # Second call (identical): exact hit -> 0 provider calls
    res2 = hybrid_cache.process(req, tenant_id="t1", provider=mock_provider)
    assert res2.status == "HIT"
    assert res2.tier == "EXACT"
    assert res2.provider_calls == 0
    assert provider_calls == 1


def test_verified_semantic_hit_on_safe_paraphrase(hybrid_cache):
    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list) to convert a list to a tuple in Python.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    # Populate cache with seed
    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)
    assert calls == 1

    # Safe paraphrase
    req_para = make_request("How can I turn a list into a tuple in Python?")
    res_para = hybrid_cache.process(req_para, tenant_id="t1", provider=provider)

    assert res_para.status == "HIT"
    assert res_para.tier == "SEMANTIC"
    assert res_para.provider_calls == 0
    assert calls == 1  # Zero additional provider calls


def test_guard_rejects_direction_reversal(hybrid_cache):
    req_seed = make_request("How do I convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list) to convert a list to a tuple.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return make_response("Use list(my_tuple) to convert tuple to list.")

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)
    assert calls == 1

    # Reversed direction: tuple to list
    req_rev = make_request("How do I convert a tuple to a list in Python?")
    res_rev = hybrid_cache.process(req_rev, tenant_id="t1", provider=provider)

    assert res_rev.status == "MISS"
    assert calls == 2  # Fresh provider call executed
    assert any("guard_rejected:conversion_direction_or_entity_changed" in g for g in res_rev.diagnosis["guard_rejections"])


def test_guard_rejects_negation_constraint(hybrid_cache):
    req_seed = make_request("How do I sort a list in Python?")
    resp_seed = make_response("Use list.sort() or the sorted() built-in.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)

    # Added negation: without using built-ins
    req_neg = make_request("How do I sort a list without using any built-ins in Python?")
    res_neg = hybrid_cache.process(req_neg, tenant_id="t1", provider=provider)

    assert res_neg.status == "MISS"
    assert calls == 2
    assert any("guard_rejected:added_explicit_requirement" in g for g in res_neg.diagnosis["guard_rejections"])


def test_guard_rejects_numeric_version_mismatch(hybrid_cache):
    req_seed = make_request("What is new in Python 3.12?")
    resp_seed = make_response("Python 3.12 adds improved error messages and f-strings.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)

    # Different version: Python 2.7
    req_num = make_request("What is new in Python 2.7?")
    res_num = hybrid_cache.process(req_num, tenant_id="t1", provider=provider)

    assert res_num.status == "MISS"
    assert calls == 2
    assert any("guard_rejected:numeric_or_version_requirement_changed" in g for g in res_num.diagnosis["guard_rejections"])


def test_entry_specific_veto_blocks_reuse(hybrid_cache):
    req_seed = make_request("Explain list conversion in Python.")
    resp_seed = make_response("Explain list conversion in Python: use tuple() constructor to convert a list.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)

    req_new = make_request("Can you explain list conversion in Python?")
    # Pre-record an entry veto on the binding
    binding = {
        "scope_sha256": digest({"tenant_id": "t1", "scope": asdict(hybrid_cache.get_scope("t1"))}),
        "incoming_request_sha256": digest(req_new),
        "candidate_request_sha256": digest(req_seed),
        "candidate_response_sha256": digest(resp_seed),
    }
    binding_h = digest(binding)
    hybrid_cache.record_entry_decision(binding_h, "cand-1", approved=False, reason="human_flagged_unsafe")

    res = hybrid_cache.process(req_new, tenant_id="t1", provider=provider)
    assert res.status == "MISS"
    assert any("entry_veto:human_flagged_unsafe" in g or "entry_revoked" in g for g in res.diagnosis["guard_rejections"])


def test_entry_specific_approval_grants_verified_hit(hybrid_cache):
    req_seed = make_request("Explain list conversion in Python.")
    resp_seed = make_response("Explain list conversion in Python: use tuple() constructor to convert a list.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)

    req_new = make_request("Can you explain list conversion in Python?")
    binding = {
        "scope_sha256": digest({"tenant_id": "t1", "scope": asdict(hybrid_cache.get_scope("t1"))}),
        "incoming_request_sha256": digest(req_new),
        "candidate_request_sha256": digest(req_seed),
        "candidate_response_sha256": digest(resp_seed),
    }
    binding_h = digest(binding)
    hybrid_cache.record_entry_decision(binding_h, "cand-1", approved=True, reason="human_certified_exact")

    res = hybrid_cache.process(req_new, tenant_id="t1", provider=provider)
    assert res.status == "HIT"
    assert res.tier == "SEMANTIC"
    assert res.diagnosis["entry_decision"] == "human_certified_exact"


def test_unverified_candidates_never_served_from_promotion_queue(hybrid_cache):
    hybrid_cache.async_promotion = True
    req = make_request("Explain dictionary comprehension.")
    resp = make_response("Dictionary comprehension explanation: use {k: v for k, v in items} syntax.")

    # Admit via async queue
    enqueued = hybrid_cache.admit_candidate(req, resp, tenant_id="t1")
    assert enqueued is True
    assert hybrid_cache.promotion_queue.qsize() == 1

    # Invariant check: candidate is in queue, NOT yet in serving cache
    assert len(hybrid_cache._serving_candidates) == 0

    # Querying right now must NOT serve the unverified candidate
    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp

    res = hybrid_cache.process(req, tenant_id="t1", provider=provider)
    # Exact hit won't fire because exact.admit wasn't called yet; semantic won't fire because candidate is unverified!
    assert res.status == "MISS"
    assert calls == 1

    # Drain and verify queue
    admitted = hybrid_cache.process_promotion_queue()
    assert admitted >= 1
    assert len(hybrid_cache._serving_candidates) >= 1
    assert next(iter(hybrid_cache._serving_candidates.values())).verified is True



def test_tenant_revocation_invalidates_candidates(hybrid_cache):
    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list) to convert a list to a tuple in Python.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="tenant-x", provider=provider)
    assert calls == 1

    # Safe paraphrase hits
    req_para = make_request("How can I turn a list into a tuple in Python?")
    res1 = hybrid_cache.process(req_para, tenant_id="tenant-x", provider=provider)
    assert res1.status == "HIT"
    assert calls == 1

    # Explicit tenant revocation
    hybrid_cache.revoke_tenant("tenant-x")

    # Subsequent request misses because tenant epoch incremented and candidates purged
    res2 = hybrid_cache.process(req_para, tenant_id="tenant-x", provider=provider)
    assert res2.status == "MISS"
    assert calls == 2


def test_ttl_expiry_causes_fallback(hybrid_cache):
    fake_clock = 1000.0
    hybrid_cache.clock = lambda: fake_clock
    hybrid_cache.ttl_seconds = 10.0

    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list) to convert a list to a tuple in Python.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    hybrid_cache.process(req_seed, tenant_id="t1", provider=provider)
    assert calls == 1

    # Advance clock past TTL
    fake_clock += 20.0

    req_para = make_request("How can I turn a list into a tuple in Python?")
    res = hybrid_cache.process(req_para, tenant_id="t1", provider=provider)

    assert res.status == "MISS"
    assert calls == 2
