"""Exercise context isolation with several seeded answers in one runtime."""
import pytest

from scripts.risk_v3_context import ReportingCoordinates, process_report
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope


class FixtureEncoder:
    identity = "fixed-encoder"

    def encode(self, text):
        return [1.0, 0.0]


class FixtureScorer:
    identity = "fixed-scorer"

    def score(self, query, answer):
        return {"probability": 1.0}


def request(text):
    return {"model": "fixture", "temperature": 0,
            "messages": [{"role": "system", "content": "approved data only"}, {"role": "user", "content": text}]}


def answer(text):
    return {"model": "fixture", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}},
    ]}


def test_context_isolates_exact_and_semantic_hits_in_the_same_cache():
    cache = ConservativeHybridCache(
        ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "fixture")),
        scope=KnowledgeScope("oversight", "v1", "test", "reporting"),
        encoder=FixtureEncoder(), passage_scorer=FixtureScorer(), mode="guarded",
    )
    a = ReportingCoordinates("source-a", "2025-01", "all")
    b = ReportingCoordinates("source-b", "2025-01", "all")
    different_month = ReportingCoordinates("source-a", "2025-02", "all")
    seed = request("What was the pass rate?")
    expected_a = answer("The pass rate was high.")
    expected_b = answer("The pass rate was low.")
    fresh = answer("Fresh response.")
    assert process_report(cache, seed, provider=lambda _: expected_a, coordinates=a).stored
    assert process_report(cache, seed, provider=lambda _: expected_b, coordinates=b).stored

    paraphrase = request("Tell me the pass rate.")
    for coords, expected in ((a, expected_a), (b, expected_b)):
        exact = process_report(cache, seed, provider=lambda _: pytest.fail("unexpected fallback"), coordinates=coords)
        semantic = process_report(cache, paraphrase, provider=lambda _: pytest.fail("unexpected fallback"), coordinates=coords)
        assert (exact.tier, exact.response) == ("EXACT", expected)
        assert (semantic.tier, semantic.response) == ("SEMANTIC", expected)

    miss = process_report(cache, seed, provider=lambda _: fresh, coordinates=different_month)
    assert miss.provider_calls == 1 and miss.response == fresh
    for _ in range(2):
        unresolved = process_report(cache, seed, provider=lambda _: fresh, coordinates=None)
        assert unresolved.status == "BYPASS" and not unresolved.stored and unresolved.provider_calls == 1
    assert seed["messages"][0]["content"] == "approved data only"


@pytest.mark.parametrize("coordinates", [("source-a", "2025-13", "all"), ("a:b", "2025-01", "all")])
def test_invalid_coordinates_fail_before_cache_boundary(coordinates):
    with pytest.raises(ValueError):
        ReportingCoordinates(*coordinates)
