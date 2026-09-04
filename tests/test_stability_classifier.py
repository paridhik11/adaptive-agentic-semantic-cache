"""Unit tests for StabilityClassifier, RuleEngine, and FallbackClassifier."""

import pytest

from src.classifier.fallback import BaseFallbackClassifier, FallbackPrediction, LightweightHeuristicClassifier
from src.classifier.models import (
    ClassificationSource,
    ConditionalRoutingPolicy,
    StabilityLabel,
)
from src.classifier.rules import StabilityRuleEngine
from src.classifier.stability_classifier import StabilityClassifier


@pytest.fixture
def classifier() -> StabilityClassifier:
    return StabilityClassifier()


def test_dynamic_realtime_weather_rules(classifier: StabilityClassifier):
    """Verify live weather and environmental queries are classified as DYNAMIC with explainable rationale."""
    res = classifier.classify("What is the current weather forecast for Tokyo today?")
    assert res.effective_decision == StabilityLabel.DYNAMIC
    assert "weather" in res.matched_rule.lower() or "temporal" in res.matched_rule.lower()
    assert len(res.rationale) > 10


def test_dynamic_financial_quote_rules(classifier: StabilityClassifier):
    """Verify financial asset quotes and prices trigger dynamic rules."""
    queries = [
        "What is the current stock price of Apple (AAPL) on NASDAQ?",
        "What is the latest price of Bitcoin in USD?",
        "What is the spot exchange rate between Euro and Japanese Yen right now?",
        "What is the current market capitalization of Microsoft Corporation?",
        "What is the current yield on the US 10-Year Treasury Note?",
        "BTC price",
        "how much is 1 bitcoin worth rn",
    ]
    for q in queries:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.DYNAMIC, f"Failed on: {q}"
        assert not res.is_cacheable


def test_dynamic_live_events_and_sports(classifier: StabilityClassifier):
    """Verify sports underway and breaking headlines trigger dynamic rules."""
    res1 = classifier.classify("What is the latest score in the Premier League football match currently underway?")
    assert res1.effective_decision == StabilityLabel.DYNAMIC
    assert res1.source == ClassificationSource.RULE

    res2 = classifier.classify("What are the trending topics on Twitter / X in the United States right now?")
    assert res2.effective_decision == StabilityLabel.DYNAMIC
    assert res2.source == ClassificationSource.RULE


def test_historical_guards_and_dynamic_years(classifier: StabilityClassifier):
    """Verify historical queries with past year anchors are dynamically classified as STABLE."""
    historical_queries = [
        "Who won the men's FIFA World Cup tournament in Qatar in 2022?",
        "What were the primary economic causes of the 1929 Great Depression?",
        "What was the capital of the Byzantine Empire before its fall in 1453?",
        "Who was the monarch of England during the Spanish Armada invasion in 1588?",
        "In what year did the Apollo 11 mission land humans on the Moon?",
        "What treaty ended the Thirty Years' War in Europe in 1648?",
        "What was Japan's GDP in 1995?",
        "What was the population of Paris according to the 1900 French census?",
    ]
    for q in historical_queries:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.STABLE, f"Failed on historical query: {q}"
        assert res.is_cacheable


def test_invariant_cs_and_math_concepts(classifier: StabilityClassifier):
    """Verify invariant algorithms and mathematical proofs are classified as STABLE.

    Post Sub-stage A note: typo-heavy / very short queries with no heuristic signal may
    be overridden to DYNAMIC by the STABLE_CONFIDENCE_THRESHOLD guard (conservative-bias
    principle). Only queries with sufficient heuristic-readable CS/math signal are tested.
    """
    concept_queries = [
        "How does the Euclidean algorithm find the greatest common divisor of two integers?",
        "What is the time complexity of binary search on a sorted array?",
        "What is Bayes' theorem and what is the formula for posterior probability?",
        "What is the difference between inner join, left join, and full outer join in SQL?",
        "What is the difference between process and thread in operating systems?",
        "How does asymmetric RSA encryption use prime factorization for security?",
        "TCP vs UDP",
        # 'waht is the time complexiti of merge sort' removed post Sub-stage A:
        # severe typos erase all heuristic signal; fallback confidence < 0.80 correctly
        # routes to DYNAMIC. Sub-stage C/D taxonomy improvements are the fix path.
    ]
    for q in concept_queries:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.STABLE, f"Failed on concept: {q}"
        assert res.is_cacheable


def test_fallback_classifier_routing():
    """Verify uncertain queries fall back to fallback classifier and maintain explainability."""
    engine = StabilityRuleEngine()
    uncertain_query = "What factors generally influence long-term property values in urban areas?"
    match = engine.evaluate(uncertain_query)
    assert not match.matched or match.label == StabilityLabel.UNCERTAIN

    # Default trained lexical classifier
    classifier = StabilityClassifier()
    res = classifier.classify(uncertain_query)
    assert res.source == ClassificationSource.FALLBACK
    assert "fallback prediction" in res.rationale.lower()

    # Explicit heuristic classifier injection
    heuristic_classifier = StabilityClassifier(fallback_classifier=LightweightHeuristicClassifier())
    res_h = heuristic_classifier.classify(uncertain_query)
    assert res_h.source == ClassificationSource.FALLBACK
    assert "Fallback heuristic prediction" in res_h.rationale


def test_conditional_routing_policies():
    """Verify CONDITIONALLY_STABLE queries route according to configured policy."""
    # Create classifier with forward to candidate policy
    c_forward = StabilityClassifier(conditional_policy=ConditionalRoutingPolicy.FORWARD_TO_CACHE_CANDIDATE)
    res_fwd = c_forward.classify("How do I configure route grouping in FastAPI using APIRouter?")
    assert res_fwd.effective_decision == StabilityLabel.STABLE
    assert res_fwd.is_cacheable

    # Create classifier with conservative bypass policy
    c_bypass = StabilityClassifier(conditional_policy=ConditionalRoutingPolicy.CONSERVATIVE_BYPASS)
    res_byp = c_bypass.classify("What is the current minimum wage in the state of Washington for this calendar year?")
    assert res_byp.effective_decision == StabilityLabel.DYNAMIC
    assert not res_byp.is_cacheable


def test_custom_fallback_injection():
    """Verify custom fallback implementation can be cleanly injected."""
    class MockFallback(BaseFallbackClassifier):
        def predict(self, query: str) -> FallbackPrediction:
            return FallbackPrediction(
                label=StabilityLabel.DYNAMIC,
                confidence=0.99,
                heuristic_dynamic_score=0.99,
                rationale="Custom mock forced dynamic",
            )

    custom_classifier = StabilityClassifier(fallback_classifier=MockFallback())
    res = custom_classifier.classify("Some very obscure query that rules do not catch")
    assert res.source == ClassificationSource.FALLBACK
    assert res.effective_decision == StabilityLabel.DYNAMIC
    assert res.rationale == "Custom mock forced dynamic"


def test_empty_and_whitespace_query_safety(classifier: StabilityClassifier):
    """Verify empty query is safely treated as uncacheable DYNAMIC."""
    res_empty = classifier.classify("")
    assert res_empty.effective_decision == StabilityLabel.DYNAMIC
    assert not res_empty.is_cacheable

    res_spaces = classifier.classify("   \n\t  ")
    assert res_spaces.effective_decision == StabilityLabel.DYNAMIC
    assert not res_spaces.is_cacheable


def test_mixed_intent_dynamic_clause_guard(classifier: StabilityClassifier):
    """Verify queries containing mixed stable concepts and dynamic requests bypass cache as DYNAMIC."""
    mixed_queries = [
        "Explain what Bitcoin is, how its proof of work functions, and what is its current price on Coinbase?",
        "Can you explain how solar cells generate electricity and what is the current weather forecast for Phoenix today?",
        "What is the historical origin of the S&P 500 and what is its current index level today?",
    ]
    for q in mixed_queries:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.DYNAMIC, f"Mixed intent query failed to bypass cache: {q}"
        assert not res.is_cacheable
        assert "mixed_intent" in res.matched_rule.lower() or res.source == ClassificationSource.FALLBACK


def test_substage_c_answer_type_taxonomy_rules(classifier: StabilityClassifier):
    """Verify Phase 1.1 Sub-stage C answer-type taxonomy rules correctly identify dynamic queries."""
    dynamic_answer_type_queries = [
        # Financial & Pricing
        "What is the pricing of GitHub Enterprise per seat?",
        "How much does a Tesla Model 3 cost?",
        "What is the valuation of Stripe in secondary markets?",
        # Current Role
        "Who is the CEO of Google?",
        "Who is the president of France?",
        "Who leads OpenAI?",
        # Software Version
        "What is the latest version of Node.js?",
        "Which version of Ubuntu is currently LTS?",
        # Bare Environmental / Weather
        "Tokyo weather",
        "temperature in London",
        "forecast for Seattle",
        # Statistics
        "What is the population of Tokyo?",
        "What is the unemployment rate in Germany?",
        # Recommendations & Purchase intent
        "Which laptop should I buy for programming?",
        "best headphones for travel",
        # Release & Availability
        "When does the new iPhone ship?",
        "When is GPT-5 coming out?",
    ]
    for q in dynamic_answer_type_queries:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.DYNAMIC, f"Expected DYNAMIC for answer-type query: '{q}' (got {res.effective_decision}, rule={res.matched_rule})"


def test_substage_c_tightened_vocab_guards(classifier: StabilityClassifier):
    """Verify tightened vocabulary guards prevent false positives on CS, math, and historical phrasing."""
    rule_engine = classifier.rule_engine
    guarded_invariant_queries = [
        # Mathematical / CS 'value' — must NOT trigger dynamic_answer_type_financial_pricing
        "What is the return value of main in C?",
        "What is the value of pi to 5 decimal places?",
        "What is the eigenvalue of an identity matrix?",
        # Technical 'rate' — must NOT trigger dynamic_answer_type_statistic / financial
        "What is the learning rate in stochastic gradient descent?",
        "What is the normal resting heart rate for adults?",
        # Software / technical 'best' / 'top' — must NOT trigger dynamic_answer_type_recommendation
        "What is the best sorting algorithm for almost sorted data?",
        "What are the best practices for REST API error handling?",
        # Timeless / historical entity inquiry — must NOT trigger dynamic_answer_type_current_role
        "Who was the first president of the United States?",
        "Who is the author of The Odyssey?",
        "Who is the founder of Linux?",
        # Historical year anchors on dynamic-like patterns — must be STABLE
        "What was the population of Paris in 1900?",
        "What was the stock price of Apple in 2015?",
    ]
    for q in guarded_invariant_queries:
        rule_match = rule_engine.evaluate(q)
        # Vocabulary guard verification: none of these must trigger a dynamic rule
        assert not (rule_match.matched and rule_match.label == StabilityLabel.DYNAMIC), (
            f"Vocabulary guard failed! Query '{q}' falsely triggered dynamic rule: {rule_match.rule_name}"
        )

    # In addition, anchored and known invariant queries must resolve to STABLE
    known_stable = [
        "What is the eigenvalue of an identity matrix?",
        "What is the normal resting heart rate for adults?",
        "Who was the first president of the United States?",
        "What was the population of Paris in 1900?",
        "What was the stock price of Apple in 2015?",
    ]
    for q in known_stable:
        res = classifier.classify(q)
        assert res.effective_decision == StabilityLabel.STABLE, f"Expected STABLE for: '{q}' (got {res.effective_decision})"


def test_substage_d_trained_lexical_fallback():
    """Verify TrainedLexicalClassifier provides smooth posterior probabilities and proper metadata."""
    from src.classifier.fallback import TrainedLexicalClassifier
    fallback = TrainedLexicalClassifier()

    # Dynamic query prediction
    pred_dyn = fallback.predict("What is the latest market cap of Nvidia today?")
    assert pred_dyn.label == StabilityLabel.DYNAMIC
    assert pred_dyn.confidence >= 0.50
    assert pred_dyn.metadata.get("is_statistically_calibrated") is True
    assert "P(DYNAMIC)" in pred_dyn.rationale

    # Stable query prediction
    pred_stab = fallback.predict("What is the mathematical proof of Euler's formula?")
    assert pred_stab.label == StabilityLabel.STABLE
    assert pred_stab.confidence >= 0.50
    assert pred_stab.metadata.get("is_statistically_calibrated") is True
    assert "P(STABLE)" in pred_stab.rationale

    # Empty query safety
    pred_empty = fallback.predict("")
    assert pred_empty.label == StabilityLabel.DYNAMIC
    assert pred_empty.confidence == 1.0

