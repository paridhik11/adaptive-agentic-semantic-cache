"""Functional safeguards only; fake scores do not establish model quality."""
from dataclasses import replace
import copy

import pytest

from src.policy.answer_applicability import DiagnosticAnswerPolicy, LocalYesNoScorer, LocalChatYesNoScorer
from src.api.service import GatewayService
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision

CONTEXT = "Authored fixture: tuple(items) returns a tuple."
SEED = {"model": "fixture", "temperature": 0, "max_tokens": 64, "messages": [
    {"role": "system", "content": CONTEXT}, {"role": "user", "content": "Convert list to tuple."}]}
VARIANT = {**SEED, "messages": [SEED["messages"][0], {"role": "user", "content": "Make a tuple from a list."}]}
ANSWER = {"model": "fixture", "choices": [{"finish_reason": "stop",
           "message": {"role": "assistant", "content": "Use tuple(items)."}}]}


class Scorer:
    identity = "test-local-scorer"
    def __init__(self, yes=.99, no=.005):
        self.prompts = []
        self.yes, self.no = yes, no
    def score(self, prompt):
        self.prompts.append(prompt)
        return {"yes_probability": self.yes, "no_probability": self.no,
                "yes_no_mass": self.yes+self.no,
                "conditional_yes": self.yes/(self.yes+self.no)}


class Encoder:
    identity = "constant-fixture"
    def encode(self, text):
        return [1., 0.]


def policy(scorer=None):
    return DiagnosticAnswerPolicy(system_prompt=CONTEXT, scorer=scorer or Scorer())


def test_positive_diagnostic_cannot_enable_serving_or_invoke_remote_provider():
    obj = policy()
    verdict = obj(VARIANT, SEED, ANSWER, binding={},
                  provider=lambda _: pytest.fail("remote verifier must not run"))
    assert verdict.approved is False and verdict.local_policy_recommendation is True
    assert verdict.local_verifier_attempts == 2


def test_actual_answer_and_context_are_in_both_checks_and_changed_answer_is_bound():
    scorer = Scorer()
    obj = policy(scorer)
    first = obj.assess(VARIANT, SEED, ANSWER)
    changed = copy.deepcopy(ANSWER)
    changed["choices"][0]["message"]["content"] = "Deliberately wrong changed answer."
    second = obj.assess(VARIANT, SEED, changed)
    assert all("Use tuple(items)." in p and CONTEXT in p and "Make a tuple" in p for p in scorer.prompts[:2])
    assert all("wrong changed answer" in p for p in scorer.prompts[2:])
    assert first.answer_sha256 != second.answer_sha256
    assert first.incoming_sha256 == second.incoming_sha256


def test_renormalized_yes_no_confidence_cannot_hide_low_full_vocabulary_mass():
    result = policy(Scorer(yes=.001, no=.000001)).assess(VARIANT, SEED, ANSWER)
    assert result.scores["sufficiency"]["conditional_yes"] > .98
    assert not result.recommended_reuse


@pytest.mark.parametrize("changed", [
    {"model": "different"}, {"temperature": .1},
    {"messages": [{"role": "system", "content": "changed source"}, VARIANT["messages"][1]]},
    {"messages": None}, {"n": True},
])
def test_context_or_parameter_changes_abstain_before_local_work(changed):
    scorer = Scorer()
    result = policy(scorer).assess({**VARIANT, **changed}, SEED, ANSWER)
    assert not result.recommended_reuse and result.local_verifier_attempts == 0
    assert not scorer.prompts


def test_incomplete_answer_is_not_assessed():
    answer = copy.deepcopy(ANSWER)
    answer["choices"][0]["finish_reason"] = "length"
    assert policy().assess(VARIANT, SEED, answer).reason == "incomplete_or_unsupported_answer"


def test_model_failure_abstains_and_records_attempt():
    class Broken(Scorer):
        def score(self, prompt):
            raise RuntimeError("local inference failure")
    result = policy(Broken()).assess(VARIANT, SEED, ANSWER)
    assert not result.recommended_reuse and result.local_verifier_attempts == 1
    assert result.reason == "local_check_failed"


def test_nonfinite_or_inconsistent_scores_abstain():
    class Broken(Scorer):
        def score(self, prompt):
            return {"yes_probability": .99, "no_probability": .001,
                    "conditional_yes": float("nan"), "yes_no_mass": .991}
    assert not policy(Broken()).assess(VARIANT, SEED, ANSWER).recommended_reuse


def test_cache_exact_path_skips_local_model_and_diagnostic_positive_is_a_miss():
    scorer = Scorer()
    obj = policy(scorer)
    calls = []
    def provider(body):
        calls.append(body)
        return ANSWER
    cache = GuardedSemanticCache(ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "fixture")),
        scope=KnowledgeScope("fixture", "1", "public", "diagnostic"),
        encoder=Encoder(), mode="guarded", approver=obj)
    cache.process(SEED, tenant_id="a", provider=provider)
    exact = cache.process(SEED, tenant_id="a", provider=provider)
    assert exact.tier == "EXACT" and not scorer.prompts
    variant = cache.process(VARIANT, tenant_id="a", provider=provider)
    assert variant.status == "MISS" and len(calls) == 2
    assert variant.diagnosis["local_policy_recommendation"] is True
    assert variant.diagnosis["local_verifier_attempts"] == 2
    assert variant.diagnosis["verifier_calls"] == 0


@pytest.mark.parametrize("field,value", [("local_verifier_attempts", True), ("local_verifier_ms", float("nan")),
    ("local_policy_recommendation", "yes")])
def test_invalid_work_metadata_cannot_authorize_hit(field, value):
    class Invalid:
        identity = "invalid-test"
        def __call__(self, *args, **kwargs):
            return replace(ReuseDecision(True, "test"), **{field: value})
    cache = GuardedSemanticCache(ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "fixture")),
        scope=KnowledgeScope("fixture", "1", "public", "test"), encoder=Encoder(), mode="guarded", approver=Invalid())
    cache.process(SEED, tenant_id="a", provider=lambda _: ANSWER)
    result = cache.process(VARIANT, tenant_id="a", provider=lambda _: ANSWER)
    assert result.status == "MISS" and result.diagnosis["stage"] == "semantic_processing_failed"


def test_mutable_model_revision_fails_before_model_loading():
    with pytest.raises(ValueError, match="immutable"):
        LocalYesNoScorer(revision="main")
    with pytest.raises(ValueError, match="immutable"):
        LocalChatYesNoScorer(revision="main")


def test_gateway_reports_local_work_separately_from_remote_verifier_work():
    cache = GuardedSemanticCache(ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "fixture")),
        scope=KnowledgeScope("fixture", "1", "public", "test"), encoder=Encoder(), mode="guarded", approver=policy())
    gateway = GatewayService(cache, lambda _: ANSWER, allowed_models={"fixture"})
    gateway.complete(SEED, "a")
    gateway.complete(VARIANT, "a")
    metrics = gateway.metrics.render()
    assert "cache_gateway_semantic_local_verifier_attempts_total 2\n" in metrics
    assert "cache_gateway_semantic_local_recommendations_total 1\n" in metrics
    assert "cache_gateway_semantic_verifier_attempts_total 0\n" in metrics
