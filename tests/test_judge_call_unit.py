"""Network-free unit tests for Phase 3 LLM Judge (OpenRouter & Gemini Backends) and JudgeDecisionStep.

Validates:
  - Safety-fallback-on-error path: any timeout, API error, malformed output,
    rate-limit exhaustion, safety filter block, or missing client/key strictly
    returns BYPASS with fallback_triggered=True.
  - Prompt template renders all 6 required fields verbatim.
  - Structured output parsing on OpenRouter mock responses (REUSE and BYPASS).
  - OpenRouter 1-retry behavior on malformed JSON schema before fail-closed.
  - Conservative safety override (even if model outputs REUSE, if is_safe=False,
    decision is forced to BYPASS).
  - Safety filter refusal detection (content_filter / safety block -> defaults to BYPASS).
  - Request ID / Generation ID capture.
  - JudgeCallable compatibility hook (__call__ returns float in [0, 1]).
  - Backward-compatibility with Gemini backend.

pytest marker: (none) — runs unconditionally in the default network-free test suite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.cache.semantic_cache import CacheDecision, CacheLookupResult
from src.classifier.models import ClassificationSource, StabilityLabel, StabilityResult
from src.decision.decision_step import CategoryHistory
from src.decision.judge_call import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    JudgeDecisionEnum,
    JudgeOutputSchema,
    JudgeResult,
    LLMJudge,
    PRICE_PER_1M_INPUT_TOKENS,
    PRICE_PER_1M_OUTPUT_TOKENS,
)
from src.decision.decision_step import JudgeDecisionResult, JudgeDecisionStep


# ---------------------------------------------------------------------------
# Fixtures & Mock Helpers
# ---------------------------------------------------------------------------

def _mock_openrouter_response(
    decision: str = "REUSE",
    is_safe: bool = True,
    confidence: float = 0.95,
    rationale: str = "Queries are semantically identical.",
    prompt_tokens: int = 240,
    completion_tokens: int = 45,
    model_version: str = "nvidia/nemotron-3-super-120b-a12b:free",
    finish_reason: str = "stop",
    generation_id: str = "gen-test-openrouter-12345",
) -> MagicMock:
    """Create a mock OpenAI/OpenRouter ChatCompletion object."""
    mock_resp = MagicMock()
    mock_resp.id = generation_id
    mock_resp.model = model_version

    mock_payload = {
        "decision": decision,
        "is_safe": is_safe,
        "confidence": confidence,
        "rationale": rationale,
    }

    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_resp.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens
    mock_resp.usage = mock_usage

    return mock_resp


def _mock_gemini_response(
    decision: str = "REUSE",
    is_safe: bool = True,
    confidence: float = 0.95,
    rationale: str = "Queries are semantically identical.",
    prompt_tokens: int = 240,
    completion_tokens: int = 45,
    model_version: str = "gemini-2.0-flash",
    finish_reason: str = "STOP",
) -> MagicMock:
    """Create a mock Google GenAI GenerateContentResponse."""
    mock_resp = MagicMock()
    mock_resp.parsed = JudgeOutputSchema(
        decision=JudgeDecisionEnum(decision),
        is_safe=is_safe,
        confidence=confidence,
        rationale=rationale,
    )
    mock_resp.text = mock_resp.parsed.model_dump_json()
    mock_resp.model_version = model_version

    mock_cand = MagicMock()
    mock_cand.finish_reason = finish_reason
    mock_resp.candidates = [mock_cand]

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = completion_tokens
    mock_usage.total_token_count = prompt_tokens + completion_tokens
    mock_resp.usage_metadata = mock_usage

    return mock_resp


def _dummy_cache_result(score: float = 0.85) -> CacheLookupResult:
    return CacheLookupResult(
        decision=CacheDecision.HIT if score >= 0.85 else CacheDecision.MISS,
        similarity_score=score,
        threshold=0.85,
        embed_latency_ms=1.0,
        search_latency_ms=0.1,
        total_latency_ms=1.1,
        store_size=1,
    )


def _dummy_stability_result(conf: float = 0.85) -> StabilityResult:
    return StabilityResult(
        query="test",
        predicted_label=StabilityLabel.STABLE,
        effective_decision=StabilityLabel.STABLE,
        is_cacheable=True,
        confidence=conf,
        source=ClassificationSource.RULE,
        matched_rule="test_rule",
        rationale="unit test",
    )


# ---------------------------------------------------------------------------
# 1. Prompt Template Rendering
# ---------------------------------------------------------------------------

class TestPromptTemplateRendering:
    def test_renders_all_required_context_fields(self):
        judge = LLMJudge()
        prompt = judge.format_prompt(
            query_a="How do I convert a string to an integer in Python?",
            query_b="How do I convert an integer to a string in Python?",
            domain="computer_science",
            similarity_score=0.9961,
            stability_confidence=0.8522,
            category_history_rate=0.5250,
        )

        assert 'Cached Query (Query A): "How do I convert a string to an integer in Python?"' in prompt
        assert 'Incoming Query (Query B): "How do I convert an integer to a string in Python?"' in prompt
        assert "Domain: computer_science" in prompt
        assert "Embedding Similarity Score: 0.9961" in prompt
        assert "Stability Confidence: 0.8522" in prompt
        assert "Domain Historical Reuse Success Rate: 0.5250" in prompt


# ---------------------------------------------------------------------------
# 2. Structured Output Parsing (OpenRouter Backend)
# ---------------------------------------------------------------------------

class TestStructuredOutputParsing:
    def test_mock_reuse_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="REUSE",
            is_safe=True,
            confidence=0.98,
            rationale="Identical computational goal.",
            prompt_tokens=250,
            completion_tokens=50,
            model_version="nvidia/nemotron-3-super-120b-a12b:free",
            generation_id="gen-openrouter-abc123",
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="What is quicksort Big-O?",
            query_b="What is quicksort average runtime?",
            domain="computer_science",
            similarity_score=0.8931,
            stability_confidence=1.0,
            category_history_rate=0.50,
        )

        assert res.decision == "REUSE"
        assert res.is_safe is True
        assert res.confidence == pytest.approx(0.98, abs=1e-3)
        assert res.rationale == "Identical computational goal."
        assert res.prompt_tokens == 250
        assert res.completion_tokens == 50
        assert res.total_tokens == 300
        assert res.model == "nvidia/nemotron-3-super-120b-a12b:free"
        assert res.request_id == "gen-openrouter-abc123"
        assert res.fallback_triggered is False
        assert res.error is None
        assert res.cost_usd == 0.0

    def test_mock_bypass_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="BYPASS",
            is_safe=False,
            confidence=0.99,
            rationale="Inverted intent: string-to-int vs int-to-string.",
            prompt_tokens=260,
            completion_tokens=40,
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="How do I convert a string to an integer in Python?",
            query_b="How do I convert an integer to a string in Python?",
            domain="computer_science",
            similarity_score=0.9961,
            stability_confidence=0.8522,
            category_history_rate=0.50,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.confidence == pytest.approx(0.99, abs=1e-3)
        assert "Inverted intent" in res.rationale
        assert res.fallback_triggered is False

    def test_conservative_safety_override(self):
        """If model returns decision='REUSE' but is_safe=False, force decision to BYPASS."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="REUSE",
            is_safe=False,  # conflicting signal
            confidence=0.50,
            rationale="Uncertain safety.",
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="computer_science",
            similarity_score=0.75,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False

    def test_retry_on_malformed_json_succeeds(self):
        """If initial completion returns non-JSON, 1 retry asking for format succeeds."""
        mock_bad = MagicMock()
        mock_bad.id = "gen-bad"
        mock_bad.model = "nvidia/nemotron-3-super-120b-a12b:free"
        bad_choice = MagicMock()
        bad_choice.finish_reason = "stop"
        bad_choice.message = MagicMock(content="Sure, here is the answer: REUSE is safe because queries match.")
        mock_bad.choices = [bad_choice]
        mock_bad.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)

        mock_good = _mock_openrouter_response(
            decision="REUSE",
            is_safe=True,
            confidence=0.95,
            rationale="Semantically identical.",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [mock_bad, mock_good]

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="computer_science",
            similarity_score=0.88,
        )

        assert res.decision == "REUSE"
        assert res.is_safe is True
        assert res.fallback_triggered is False
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# 3. Fail-Closed Safety Fallback on Errors & Safety Filters
# ---------------------------------------------------------------------------

class TestFailClosedSafetyFallback:
    def test_safety_fallback_on_timeout(self):
        """On timeout, decision MUST be BYPASS (fail-closed, never fail-open)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out after 15.0s")

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="computer_science",
            similarity_score=0.95,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.confidence == 0.0
        assert res.fallback_triggered is True
        assert "TimeoutError" in res.error
        assert res.prompt_tokens == 0
        assert res.completion_tokens == 0

    def test_safety_fallback_on_api_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("503 Service Unavailable")

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="system_operations",
            similarity_score=0.80,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.fallback_triggered is True
        assert "503 Service Unavailable" in res.error

    def test_safety_fallback_on_safety_filter_block(self):
        """If provider blocks content with finish_reason='content_filter', safely default to BYPASS."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="REUSE",
            is_safe=True,
            finish_reason="content_filter",
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="computer_science",
            similarity_score=0.88,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.fallback_triggered is True
        assert "Candidate blocked" in res.error
        assert "CONTENT_FILTER" in res.error

    def test_safety_fallback_on_malformed_output_after_retry(self):
        """If response output is malformed after retry, decision must fall back to BYPASS."""
        mock_bad = MagicMock()
        mock_bad.id = "gen-bad"
        mock_bad.model = "nvidia/nemotron-3-super-120b-a12b:free"
        bad_choice = MagicMock()
        bad_choice.finish_reason = "stop"
        bad_choice.message = MagicMock(content="NOT VALID JSON")
        mock_bad.choices = [bad_choice]
        mock_bad.usage = MagicMock(prompt_tokens=100, completion_tokens=10, total_tokens=110)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [mock_bad, mock_bad]

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="mathematics",
            similarity_score=0.88,
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.fallback_triggered is True
        assert "invalid JSON after retry" in res.error

    def test_safety_fallback_when_client_missing(self):
        """When no API key or client is provided, judge safely defaults to BYPASS."""
        with patch("src.decision.judge_call._resolve_api_key", return_value=None):
            judge = LLMJudge(client=None, api_key=None, backend="openrouter")
            res = judge.judge(
                query_a="Query A",
                query_b="Query B",
                domain="finance_economics",
                similarity_score=0.70,
            )

            assert res.decision == "BYPASS"
            assert res.is_safe is False
            assert res.fallback_triggered is True
            assert "Missing API key" in res.error


# ---------------------------------------------------------------------------
# 4. Gemini Backend Backward-Compatibility Tests
# ---------------------------------------------------------------------------

class TestGeminiBackwardCompatibility:
    def test_gemini_backend_reuse(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_gemini_response(
            decision="REUSE",
            is_safe=True,
            confidence=0.98,
            model_version="gemini-2.5-flash",
        )

        judge = LLMJudge(client=mock_client, backend="gemini", rate_limit_delay_seconds=0.0)
        res = judge.judge(
            query_a="Query A",
            query_b="Query B",
            domain="computer_science",
            similarity_score=0.90,
        )

        assert res.decision == "REUSE"
        assert res.is_safe is True
        assert res.model == "gemini-2.5-flash"
        assert res.fallback_triggered is False


# ---------------------------------------------------------------------------
# 5. JudgeCallable Hook Compatibility
# ---------------------------------------------------------------------------

class TestJudgeCallableCompatibility:
    def test_call_signature_matches_callable(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="REUSE",
            is_safe=True,
            confidence=0.92,
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)

        # Call with query_a and query_b provided
        score = judge(
            similarity_score=0.85,
            stability_confidence=0.90,
            domain="computer_science",
            query_a="How do I reverse a string in Python?",
            query_b="Python string reversal syntax",
        )
        assert isinstance(score, float)
        assert score == pytest.approx(0.92, abs=1e-3)

    def test_call_without_queries_returns_zero(self):
        """Without query texts, intent analysis is impossible -> returns 0.0 safe fallback."""
        mock_client = MagicMock()
        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)

        score = judge(
            similarity_score=0.95,
            stability_confidence=0.95,
            domain="computer_science",
            query_a="",
            query_b="",
        )
        assert score == 0.0
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# 6. JudgeDecisionStep Routing and History Updating
# ---------------------------------------------------------------------------

class TestJudgeDecisionStepIntegration:
    def test_ambiguous_safe_pair_produces_reuse(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="REUSE",
            is_safe=True,
            confidence=0.94,
            rationale="Semantically identical prompt intent.",
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        history = CategoryHistory()
        step = JudgeDecisionStep(judge=judge, history=history)

        cache_res = _dummy_cache_result(score=0.89)
        stab_res = _dummy_stability_result(conf=0.88)

        res: JudgeDecisionResult = step.decide(
            query_a="Cached query A",
            query_b="Incoming query B",
            cache_result=cache_res,
            stability_result=stab_res,
            domain="computer_science",
        )

        assert res.decision == "REUSE"
        assert res.is_safe is True
        assert res.similarity_score == 0.89
        assert res.stability_confidence == 0.88
        assert res.domain == "computer_science"
        assert res.judge_result.decision == "REUSE"
        assert res.judge_result.is_safe is True

    def test_ambiguous_unsafe_pair_produces_bypass(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openrouter_response(
            decision="BYPASS",
            is_safe=False,
            confidence=0.99,
            rationale="Inverted intent: sort ascending vs descending.",
        )

        judge = LLMJudge(client=mock_client, backend="openrouter", rate_limit_delay_seconds=0.0)
        step = JudgeDecisionStep(judge=judge)

        cache_res = _dummy_cache_result(score=0.96)
        stab_res = _dummy_stability_result(conf=0.85)

        res: JudgeDecisionResult = step.decide(
            query_a="sort ascending",
            query_b="sort descending",
            cache_result=cache_res,
            stability_result=stab_res,
            domain="computer_science",
        )

        assert res.decision == "BYPASS"
        assert res.is_safe is False
        assert res.judge_result.decision == "BYPASS"
