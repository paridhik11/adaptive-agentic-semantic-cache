"""Unit and regression tests for AnswerAwareSemanticPolicy."""
import copy
import pytest

from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision
from src.api.service import GatewayService

CONTEXT = "Pinned Python 3.12 documentation: tuple(seq) converts sequence to tuple."
SEED = {
    "model": "test-model",
    "temperature": 0.0,
    "max_tokens": 128,
    "messages": [
        {"role": "system", "content": CONTEXT},
        {"role": "user", "content": "How do I convert a list into a tuple?"},
    ],
}
PARAPHRASE = {
    "model": "test-model",
    "temperature": 0.0,
    "max_tokens": 128,
    "messages": [
        {"role": "system", "content": CONTEXT},
        {"role": "user", "content": "How can I make a tuple from an existing Python list?"},
    ],
}
ANSWER = {
    "model": "test-model",
    "choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "Use the tuple() constructor: tuple(my_list)."},
    }],
}


class MockPassageScorer:
    """Deterministic passage scorer for fast regression tests."""
    identity = "mock-passage-scorer-v1"

    def __init__(self, score_prob: float = 0.95):
        self.score_prob = score_prob
        self.calls = []

    def score(self, query: str, passage: str) -> dict:
        self.calls.append((query, passage))
        return {
            "logit": 3.0 if self.score_prob >= 0.5 else -3.0,
            "probability": self.score_prob,
            "input_tokens": len(query.split()) + len(passage.split()),
        }


class MockEncoder:
    identity = "mock-encoder-v1"
    def encode(self, text: str):
        # Deterministic 4-dim unit vector
        return [0.5, 0.5, 0.5, 0.5]


def test_answer_aware_disabled_by_default():
    scorer = MockPassageScorer(score_prob=0.95)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=False)
    verdict = policy(PARAPHRASE, SEED, ANSWER, binding={})
    assert verdict.approved is False
    assert verdict.local_policy_recommendation is True
    assert verdict.reason == "verified_answer_reuse"
    assert verdict.local_verifier_attempts == 1


def test_answer_aware_enabled_approves_safe_paraphrase():
    scorer = MockPassageScorer(score_prob=0.95)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=True)
    verdict = policy(PARAPHRASE, SEED, ANSWER, binding={})
    assert verdict.approved is True
    assert verdict.local_policy_recommendation is True
    assert verdict.reason == "verified_answer_reuse"


def test_guard_rejects_numeric_version_mismatch():
    scorer = MockPassageScorer(score_prob=0.99)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=True)
    req_v2 = copy.deepcopy(SEED)
    req_v2["messages"][-1]["content"] = "How do I convert a list into a tuple in Python 2?"
    req_v3 = copy.deepcopy(SEED)
    req_v3["messages"][-1]["content"] = "How do I convert a list into a tuple in Python 3?"
    verdict = policy(req_v2, req_v3, ANSWER, binding={})
    assert verdict.approved is False
    assert verdict.local_policy_recommendation is False
    assert "numeric_or_version_requirement_changed" in verdict.reason
    assert len(scorer.calls) == 0  # Guard fires before neural scoring!


def test_guard_rejects_direction_reversal():
    scorer = MockPassageScorer(score_prob=0.99)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=True)
    req_rev = copy.deepcopy(SEED)
    req_rev["messages"][-1]["content"] = "How do I convert a tuple into a list?"
    verdict = policy(req_rev, SEED, ANSWER, binding={})
    assert verdict.approved is False
    assert verdict.local_policy_recommendation is False
    assert "conversion_direction_or_entity_changed" in verdict.reason
    assert len(scorer.calls) == 0


def test_guard_rejects_negation_and_added_constraint():
    scorer = MockPassageScorer(score_prob=0.99)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=True)
    req_neg = copy.deepcopy(SEED)
    req_neg["messages"][-1]["content"] = "How do I convert a list into a tuple without using tuple()?"
    verdict = policy(req_neg, SEED, ANSWER, binding={})
    assert verdict.approved is False
    assert verdict.local_policy_recommendation is False
    assert "added_explicit_requirement" in verdict.reason
    assert len(scorer.calls) == 0


def test_verifier_rejects_low_scoring_answer():
    scorer = MockPassageScorer(score_prob=0.20)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, threshold=0.85, enabled=True)
    verdict = policy(PARAPHRASE, SEED, ANSWER, binding={})
    assert verdict.approved is False
    assert verdict.local_policy_recommendation is False
    assert verdict.reason == "verifier_threshold_rejected"
    assert len(scorer.calls) == 1


def test_context_mismatch_blocks_evaluation():
    scorer = MockPassageScorer(score_prob=0.99)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, enabled=True)
    req_other_ctx = copy.deepcopy(PARAPHRASE)
    req_other_ctx["messages"][0]["content"] = "Different context: unrelated documentation."
    verdict = policy(req_other_ctx, SEED, ANSWER, binding={})
    assert verdict.approved is False
    assert "outside_pinned_context" in verdict.reason
    assert len(scorer.calls) == 0


def test_guarded_cache_integration_serves_verified_semantic_hit():
    scorer = MockPassageScorer(score_prob=0.95)
    policy = AnswerAwareSemanticPolicy(system_prompt=CONTEXT, scorer=scorer, threshold=0.85, enabled=True)
    encoder = MockEncoder()
    exact_cache = ScopedResponseCache(eligibility=lambda req: EligibilityDecision(True, "test"))
    scope = KnowledgeScope("python-docs", "rev-test-1", "public", "p1")
    cache = GuardedSemanticCache(
        exact_cache,
        scope=scope,
        encoder=encoder,
        approver=policy,
        mode="guarded",
        threshold=0.50,
        margin=0.0,
    )
    
    calls = 0
    def mock_provider(req):
        nonlocal calls
        calls += 1
        return copy.deepcopy(ANSWER)
        
    service = GatewayService(cache, mock_provider, allowed_models={"test-model"})
    
    # First request: Cache miss -> provider called
    res1 = service.complete(copy.deepcopy(SEED), tenant_id="tenant-1")
    assert res1.status == "MISS"
    assert calls == 1
    
    # Second request: Exact repeat -> exact hit
    res2 = service.complete(copy.deepcopy(SEED), tenant_id="tenant-1")
    assert res2.status == "HIT"
    assert getattr(res2, "tier", None) == "EXACT"
    assert calls == 1
    
    # Third request: Safe paraphrase -> semantic hit
    res3 = service.complete(copy.deepcopy(PARAPHRASE), tenant_id="tenant-1")
    assert res3.status == "HIT"
    assert getattr(res3, "tier", None) == "SEMANTIC"
    assert calls == 1  # No new provider call!
    
    # Fourth request: Direction reversal -> rejected by guard -> provider called
    req_rev = copy.deepcopy(SEED)
    req_rev["messages"][-1]["content"] = "How do I convert a tuple into a list?"
    res4 = service.complete(req_rev, tenant_id="tenant-1")
    assert res4.status == "MISS"
    assert calls == 2  # Provider called safely!
