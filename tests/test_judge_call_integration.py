"""Integration tests for Phase 3 LLM Judge — REQUIRES LIVE GEMINI API KEY.

These tests invoke the actual Google Gemini model via the Google GenAI API to verify:
  - End-to-end round trip with structured outputs.
  - Correct detection of algorithmic intent inversion on PAIR-035 (string-to-int vs int-to-string).
  - Correct detection of sort order inversion on PAIR-002 (ascending vs descending).
  - Correct approval of true equivalence on PAIR-004 (quicksort runtime).
  - Explicit verification of fail-closed behavior on an invalid API key.
  - Real token counts, authentic model string, and real measured latency.

To run these tests:
    pytest -m requires_llm_api tests/test_judge_call_integration.py

To exclude these tests (default pytest invocation):
    pytest    # automatically excluded by pyproject.toml addopts
"""

from __future__ import annotations

import os
import pytest

from src.decision.judge_call import LLMJudge, DEFAULT_JUDGE_MODEL, _resolve_api_key


# Mark all tests in this module as requiring live LLM API access
pytestmark = pytest.mark.requires_llm_api


@pytest.fixture(scope="module")
def judge() -> LLMJudge:
    """Instantiate live judge; skips if GEMINI_API_KEY or GOOGLE_API_KEY is not set."""
    api_key = _resolve_api_key()
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not configured in environment — skipping live API integration tests.")
    return LLMJudge(model=DEFAULT_JUDGE_MODEL, api_key=api_key)


class TestLLMJudgeLiveAPI:
    def test_explicit_fail_closed_on_invalid_api_key(self):
        """CRITICAL INVARIANT: A bad key or failed API call must strictly return BYPASS with fallback_triggered=True."""
        bad_judge = LLMJudge(model=DEFAULT_JUDGE_MODEL, api_key="INVALID_TEST_KEY_FOR_FAIL_CLOSED")
        result = bad_judge.judge(
            query_a="What is quicksort Big-O?",
            query_b="What is quicksort average runtime?",
            domain="computer_science",
            similarity_score=0.8931,
            stability_confidence=1.0,
            category_history_rate=0.50,
        )

        assert result.decision == "BYPASS"
        assert result.is_safe is False
        assert result.fallback_triggered is True
        assert result.error is not None
        assert result.cost_usd == 0.0

    def test_catches_intent_inversion_pair_035(self, judge: LLMJudge):
        """PAIR-035 has cosine similarity 0.9961 but inverted intent.

        Query A: "How do I convert a string to an integer in Python?"
        Query B: "How do I convert an integer to a string in Python?"
        The Gemini judge must correctly catch this intent inversion and return BYPASS.
        """
        result = judge.judge(
            query_a="How do I convert a string to an integer in Python?",
            query_b="How do I convert an integer to a string in Python?",
            domain="computer_science",
            similarity_score=0.9961,
            stability_confidence=0.8522,
            category_history_rate=0.50,
        )

        assert result.decision == "BYPASS"
        assert result.is_safe is False
        assert result.fallback_triggered is False
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.latency_ms > 0.0
        assert result.cost_usd == 0.0
        assert "gemini" in result.model.lower()
        assert len(result.rationale) > 0

    def test_catches_sort_order_inversion_pair_002(self, judge: LLMJudge):
        """PAIR-002 has cosine similarity 0.9645 with inverted sorting order.

        Query A: "How do I sort a list of integers in ascending order in Python?"
        Query B: "How do I sort a list of integers in descending order in Python?"
        The Gemini judge must correctly identify ascending vs descending and return BYPASS.
        """
        result = judge.judge(
            query_a="How do I sort a list of integers in ascending order in Python?",
            query_b="How do I sort a list of integers in descending order in Python?",
            domain="computer_science",
            similarity_score=0.9645,
            stability_confidence=0.8257,
            category_history_rate=0.50,
        )

        assert result.decision == "BYPASS"
        assert result.is_safe is False
        assert result.fallback_triggered is False
        assert result.prompt_tokens > 0
        assert result.latency_ms > 0.0
        assert "gemini" in result.model.lower()

    def test_approves_safe_equivalent_pair_004(self, judge: LLMJudge):
        """PAIR-004 has cosine similarity 0.8931 and is semantically safe.

        Query A: "What is the time complexity of quicksort in the average and worst case?"
        Query B: "What is the Big-O runtime of quicksort on average and in the worst case?"
        The Gemini judge must confirm equivalence and return REUSE.
        """
        result = judge.judge(
            query_a="What is the time complexity of quicksort in the average and worst case?",
            query_b="What is the Big-O runtime of quicksort on average and in the worst case?",
            domain="computer_science",
            similarity_score=0.8931,
            stability_confidence=1.0,
            category_history_rate=0.50,
        )

        assert result.decision == "REUSE"
        assert result.is_safe is True
        assert result.confidence >= 0.70
        assert result.fallback_triggered is False
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.latency_ms > 0.0
        assert result.cost_usd == 0.0
        assert "gemini" in result.model.lower()
