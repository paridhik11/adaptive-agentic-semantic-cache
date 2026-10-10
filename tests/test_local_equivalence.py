"""Unit tests for Phase 8 local equivalence layer and rule guards."""

import pytest
from unittest.mock import MagicMock

from src.decision.local_equivalence import (
    evaluate_rule_guards,
    extract_numbers,
    extract_negations,
    extract_proper_entities,
    LocalEquivalenceDecisionStep,
    LocalEquivalenceResult,
)


def test_number_rule_guard():
    """Rule guard catches numeric differences."""
    # Matching numbers
    ok, reason = evaluate_rule_guards(
        "Top 10 vacation spots in 2024",
        "Best 10 vacation destinations for 2024"
    )
    assert ok is True
    assert reason == "GUARDS_PASSED"

    # Differing numbers
    ok, reason = evaluate_rule_guards(
        "Top 5 vacation spots in 2024",
        "Top 10 vacation spots in 2024"
    )
    assert ok is False
    assert "NUMBER_MISMATCH" in reason

    # One has number, one does not
    ok, reason = evaluate_rule_guards(
        "iPhone 15 pro review",
        "iPhone pro review"
    )
    assert ok is False
    assert "NUMBER_MISMATCH" in reason


def test_negation_rule_guard():
    """Rule guard catches polarity flips."""
    # Both affirmative
    ok, reason = evaluate_rule_guards(
        "Is caffeine safe during pregnancy?",
        "Can pregnant women safely consume caffeine?"
    )
    assert ok is True

    # One negated, one affirmative
    ok, reason = evaluate_rule_guards(
        "Is caffeine safe during pregnancy?",
        "Why is caffeine not safe during pregnancy?"
    )
    assert ok is False
    assert "NEGATION_MISMATCH" in reason

    # Contraction negation
    ok, reason = evaluate_rule_guards(
        "Can I drive after taking antihistamines?",
        "Why can't I drive after taking antihistamines?"
    )
    assert ok is False
    assert "NEGATION_MISMATCH" in reason


def test_entity_rule_guard():
    """Rule guard catches conflicting proper entities."""
    ok, reason = evaluate_rule_guards(
        "Who is the president of France?",
        "Who is the president of Germany?"
    )
    assert ok is False
    assert "ENTITY_MISMATCH" in reason


def test_local_equivalence_mock_model():
    """LocalEquivalenceDecisionStep routes based on score and guards."""
    mock_model = MagicMock()
    # High score
    mock_model.predict.return_value = [0.95]

    step = LocalEquivalenceDecisionStep(
        threshold=0.70,
        fallback_to_llm=False,
        model_instance=mock_model
    )

    # Valid pair with matching guards and high score -> REUSE
    res = step.decide(
        "How do I sort a list in Python?",
        "Python list sorting syntax"
    )
    assert res.decision == "REUSE"
    assert res.is_safe is True
    assert res.guards_passed is True
    assert res.fallback_to_llm_used is False

    # Guard violation overrides high model score -> BYPASS
    res_guard_fail = step.decide(
        "Top 5 sorting algorithms in Python",
        "Top 10 sorting algorithms in Python"
    )
    assert res_guard_fail.decision == "BYPASS"
    assert res_guard_fail.is_safe is False
    assert res_guard_fail.guards_passed is False

    # Low score -> BYPASS without LLM fallback
    mock_model.predict.return_value = [0.25]
    res_low = step.decide(
        "What is merge sort?",
        "What is quick sort?"
    )
    assert res_low.decision == "BYPASS"
    assert res_low.is_safe is False
    assert res_low.fallback_to_llm_used is False
