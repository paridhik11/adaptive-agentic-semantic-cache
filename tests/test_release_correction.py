"""Tests for Final Release Correction — P2 Findings & P3-P6 Integration.

Validates:
1. Safe default exact-only serving mode.
2. Automatic neural semantic reuse shadow mode (diagnostic only, no substitution).
3. Guarded semantic reuse under explicit opt-in.
4. Request/answer coverage checks for explicit executable-example requirements.
5. Direction and inverse operation compatibility guardrails.
6. Live runner zero-output-ceiling validation with mocked transport.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import pytest

from scripts.run_live_benchmark import (
    FREE_MODELS,
    LiveBenchmarkRunner,
    sanitize_outbound_request,
)
from src.policy.decomposed_applicability import (
    answer_satisfies_requirements,
    contains_executable_example,
    inverse_operation_mismatch,
    requirement_guard,
    requires_executable_example,
)
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope


class MockEncoder:
    identity = "mock-encoder@v1"

    def encode(self, text: str):
        vec = [0.0] * 384
        for w in text.lower().split():
            idx = abs(hash(w)) % 384
            vec[idx] += 1.0
        norm = sum(x*x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else [1.0] + [0.0] * 383


class MockPassageScorer:
    identity = "mock-passage-scorer@v1"

    def score(self, query: str, passage: str):
        return {"probability": 0.95, "logit": 3.0, "input_tokens": 12}


def make_request(user_content: str, model: str = "test-model") -> dict:
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": user_content},
        ],
    }


def make_response(content: str, model: str = "test-model") -> dict:
    return {
        "id": "resp-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }


# =============================================================================
# 1. Safe Serving Mode Tests (Exact Default, Shadow Diagnostic, Guarded Opt-in)
# =============================================================================

def test_default_mode_is_exact():
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    scope = KnowledgeScope("docs", "v1", "test-tenant", "rel-v1")
    cache = ConservativeHybridCache(
        exact,
        scope=scope,
        encoder=MockEncoder(),
        passage_scorer=MockPassageScorer(),
    )
    # Default mode must be 'exact'
    assert cache.mode == "exact"

    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list).")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    # 1. Seed query: cold miss
    r1 = cache.process(req_seed, tenant_id="test-tenant", provider=provider)
    assert r1.status == "MISS"
    assert calls == 1

    # 2. Identical repeat: exact hit (0 provider calls)
    r2 = cache.process(req_seed, tenant_id="test-tenant", provider=provider)
    assert r2.status == "HIT"
    assert r2.tier == "EXACT"
    assert r2.provider_calls == 0
    assert calls == 1

    # 3. Safe paraphrase: MUST MISS under safe default exact mode (no semantic substitution)
    req_para = make_request("How can I turn a list into a tuple in Python?")
    r3 = cache.process(req_para, tenant_id="test-tenant", provider=provider)
    assert r3.status == "MISS"
    assert r3.tier == "NONE"
    assert calls == 2  # Provider called fresh for paraphrase


def test_shadow_mode_performs_diagnostic_without_substituting_answer():
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    scope = KnowledgeScope("docs", "v1", "test-tenant", "rel-v1")
    cache = ConservativeHybridCache(
        exact,
        scope=scope,
        encoder=MockEncoder(),
        passage_scorer=MockPassageScorer(),
        mode="shadow",
    )
    assert cache.mode == "shadow"

    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list).")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    # 1. Seed query: cold miss (admitted for shadow tracking)
    cache.process(req_seed, tenant_id="test-tenant", provider=provider)
    assert calls == 1

    # 2. Exact repeat still hits exact tier
    r_exact = cache.process(req_seed, tenant_id="test-tenant", provider=provider)
    assert r_exact.status == "HIT"
    assert r_exact.tier == "EXACT"

    # 3. Paraphrase: performs diagnostic evaluation but returns MISS (provider called)
    req_para = make_request("How can I turn a list into a tuple in Python?")
    r_para = cache.process(req_para, tenant_id="test-tenant", provider=provider)

    assert r_para.status == "MISS"
    assert calls == 2  # Provider was called fresh; no unqualified substitution
    # Diagnostic details must reflect shadow candidate evaluation
    assert r_para.diagnosis.get("stage") == "shadow_candidate"
    assert r_para.diagnosis.get("local_policy_recommendation") is True
    assert r_para.diagnosis.get("tier") == "SEMANTIC"


def test_guarded_mode_explicit_opt_in_serves_verified_semantic_hit():
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    scope = KnowledgeScope("docs", "v1", "test-tenant", "rel-v1")
    cache = ConservativeHybridCache(
        exact,
        scope=scope,
        encoder=MockEncoder(),
        passage_scorer=MockPassageScorer(),
        mode="guarded",
    )
    assert cache.mode == "guarded"

    req_seed = make_request("How to convert a list to a tuple in Python?")
    resp_seed = make_response("Use tuple(my_list) to convert a list.")

    calls = 0
    def provider(r):
        nonlocal calls
        calls += 1
        return resp_seed

    cache.process(req_seed, tenant_id="test-tenant", provider=provider)
    assert calls == 1

    # Safe paraphrase under guarded mode: returns verified semantic hit
    req_para = make_request("How can I turn a list into a tuple in Python?")
    r_para = cache.process(req_para, tenant_id="test-tenant", provider=provider)
    assert r_para.status == "HIT"
    assert r_para.tier == "SEMANTIC"
    assert r_para.provider_calls == 0
    assert calls == 1  # Zero additional calls


# =============================================================================
# 2. Executable Example Requirement Coverage Tests
# =============================================================================

def test_requires_executable_example_detection():
    # True positives (development examples)
    assert requires_executable_example("How do I split a string? Provide an executable example.")
    assert requires_executable_example("Show me a code snippet for reading files in Python.")
    assert requires_executable_example("Give a working code example of list comprehension.")
    assert requires_executable_example("Include runnable code for converting dict to json.")
    assert requires_executable_example("How to parse dates with a code example in Python?")

    # True negatives (no explicit executable example requirement)
    assert not requires_executable_example("How do I split a string in Python?")
    assert not requires_executable_example("What is the difference between list and tuple?")
    assert not requires_executable_example("Why does dict preserve insertion order?")


def test_contains_executable_example_detection():
    # Markdown fenced code block
    fenced = "You can split a string as follows:\n```python\ntext = 'a,b,c'\nparts = text.split(',')\n```\nDone."
    assert contains_executable_example(fenced)

    # REPL session
    repl = "In Python:\n>>> 'hello world'.split()\n['hello', 'world']"
    assert contains_executable_example(repl)

    # Inline call syntax
    inline = "You can call `str.split(',')` directly on your string."
    assert contains_executable_example(inline)

    # Conceptual explanation without code
    prose = "The split method breaks a string into a list using a delimiter."
    assert not contains_executable_example(prose)


def test_answer_satisfies_requirements_rejects_missing_example():
    q_with_ex = "How do I split a string? Include an executable example."
    ans_prose = "Use the split method of Python string objects."
    ans_code = "Use split():\n```python\ns = 'a,b'.split(',')\n```"

    # Rejection when code example is required but answer lacks it
    assert answer_satisfies_requirements(q_with_ex, ans_prose) == "missing_explicit_executable_example"
    # Acceptance when answer contains code block
    assert answer_satisfies_requirements(q_with_ex, ans_code) is None


def test_requirement_guard_rejects_added_example_requirement():
    q_orig = "How do I split a string in Python?"
    q_inc = "How do I split a string in Python? Provide an executable example."

    guard = requirement_guard(q_orig, q_inc)
    assert guard == "added_explicit_executable_example_requirement"


# =============================================================================
# 3. Direction and Inverse Operation Compatibility Tests
# =============================================================================

def test_inverse_operation_detection_rejects_incompatible_pairs():
    # Encodings & Serialization
    assert inverse_operation_mismatch("How to encode text to bytes?", "How to decode bytes to text?")
    assert inverse_operation_mismatch("How to serialize a dict to json?", "How to deserialize json to a dict?")
    assert inverse_operation_mismatch("How to compress data?", "How to decompress data?")
    assert inverse_operation_mismatch("How to encrypt a message?", "How to decrypt a message?")
    assert inverse_operation_mismatch("How to import a module?", "How to export a module?")
    assert inverse_operation_mismatch("How to join a list of strings?", "How to split a string into a list?")

    # Lifecycle & System
    assert inverse_operation_mismatch("How to install a package?", "How to uninstall a package?")
    assert inverse_operation_mismatch("How to enable debug logging?", "How to disable debug logging?")
    assert inverse_operation_mismatch("How to lock a mutex?", "How to unlock a mutex?")
    assert inverse_operation_mismatch("How to connect to the database?", "How to disconnect from the database?")
    assert inverse_operation_mismatch("How to sort in ascending order?", "How to sort in descending order?")

    # Symmetrical / compatible queries should NOT trigger mismatch
    assert not inverse_operation_mismatch("How to encode utf-8?", "How do I encode text with utf-8?")
    assert not inverse_operation_mismatch("How to split a string?", "How can I split text?")


def test_requirement_guard_rejects_inverse_operations():
    orig = "How do I encode a string to utf-8 in Python?"
    incoming = "How do I decode bytes from utf-8 in Python?"
    assert requirement_guard(orig, incoming) == "inverse_or_opposing_operation_mismatch"


# =============================================================================
# 4. Live Benchmark Runner Zero-Output-Ceiling Behavior (Mocked Transport)
# =============================================================================

def test_sanitize_outbound_request_rejects_zero_or_negative_max_tokens():
    req = {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "messages": [{"role": "user", "content": "hello"}],
    }
    # Zero ceiling must raise ValueError
    with pytest.raises(ValueError, match="max_output_tokens ceiling must be a positive integer"):
        sanitize_outbound_request(req, "nvidia/nemotron-3-super-120b-a12b:free", max_output_tokens=0)

    # Negative ceiling must raise ValueError
    with pytest.raises(ValueError, match="max_output_tokens ceiling must be a positive integer"):
        sanitize_outbound_request(req, "nvidia/nemotron-3-super-120b-a12b:free", max_output_tokens=-10)

    # Request with max_tokens: 0 must be rejected
    req_zero = dict(req, max_tokens=0)
    with pytest.raises(ValueError, match="Invalid max_tokens value"):
        sanitize_outbound_request(req_zero, "nvidia/nemotron-3-super-120b-a12b:free", max_output_tokens=1024)


def test_live_runner_zero_output_ceiling_blocked_in_preflight():
    with tempfile.TemporaryDirectory() as tmp:
        journal_path = Path(tmp) / "journal.db"
        journal_path.touch()

        free_model = list(FREE_MODELS)[0]
        runner = LiveBenchmarkRunner(
            model=free_model,
            execute_live=True,
            max_output_tokens=0,
            journal_path=journal_path,
        )
        preflight = runner.preflight_check()
        assert preflight["status"] == "BLOCKED"
        assert any("max_output_tokens ceiling must be greater than zero" in b for b in preflight["blockers"])

        # Attempting execute_request must raise RuntimeError without dispatching network calls
        req = {"model": free_model, "messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(RuntimeError, match="Live execution blocked"):
            runner.execute_request(req, "req-1")
