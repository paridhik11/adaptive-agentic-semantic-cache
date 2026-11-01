"""Deterministic regression and acceptance tests for P1.1 repairs.

Directly verifies that all defects demonstrated in the independent review:
1. Queued candidate promotion after revocation is prevented.
2. Cross-tier revocation consistency (revoked binding invalidates exact entry).
3. Request compatibility: model identity and response format mismatch rejected.
4. Conversation history compatibility: intermediate message mismatch rejected.
5. Verifier failure abstains safely and returns fresh response without crashing.
6. Synchronous admission latency is accounted in total request latency.
7. Millisecond units are correctly scaled across all tiers.
8. Live runner safeguards: verified free-only allowlist, journal enforcement, max token ceiling.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import pytest

from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope, approval_binding, digest
from src.policy.sqlite_store import SQLiteResponseStore
from tests.test_hybrid_cache import (
    MockEncoder,
    make_request,
    make_response,
)
from scripts.run_live_benchmark import LiveBenchmarkRunner, FREE_MODELS


class ConstantScorer:
    identity = "constant-scorer"

    def __init__(self, prob: float = 0.99):
        self.prob = prob

    def score(self, q: str, a: str) -> dict:
        return {"probability": self.prob, "logit": 5.0, "input_tokens": 10}


def test_queued_candidate_prevented_from_promotion_after_revocation():
    """Defect A1: A candidate enqueued before tenant revocation must be dropped."""
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=ConstantScorer(prob=0.99),
        async_promotion=True,
    )
    req = make_request("Explain list conversion in Python.")
    resp = make_response("Use tuple() for list conversion.")
    para = make_request("Can you explain list conversion in Python?")

    # Enqueue candidate while tenant is valid
    cache.admit_candidate(req, resp, tenant_id="tenant-t")
    assert cache.promotion_queue.qsize() == 1

    # Revoke tenant before queue is processed
    cache.revoke_tenant("tenant-t")

    # Processing queue must drop the item because epoch changed
    promoted = cache.process_promotion_queue()
    assert promoted == 0
    assert len(cache._serving_candidates) == 0

    # Incoming query must not hit semantic cache
    calls = []
    res = cache.process(
        para,
        tenant_id="tenant-t",
        provider=lambda r: (calls.append(1) or resp),
    )
    assert res.status == "MISS"
    assert len(calls) == 1


def test_revoked_binding_suppressed_in_exact_cache():
    """Defect A3: Revoking an entry binding must invalidate the exact tier as well."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "cache.db"
        store = SQLiteResponseStore(db_path, namespace="test")
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "test"),
            storage=store,
        )
        cache = ConservativeHybridCache(
            exact,
            scope=KnowledgeScope("docs", "v1", "public", "p1"),
            encoder=MockEncoder(),
            passage_scorer=ConstantScorer(prob=0.99),
        )

        req = make_request("How to sort a list in Python?")
        resp = make_response("Use list.sort() or sorted().")

        # Process to populate exact cache
        r1 = cache.process(req, tenant_id="tenant-t", provider=lambda r: resp)
        assert r1.status == "MISS"
        assert r1.provider_calls == 1

        # Second request is an exact HIT
        r2 = cache.process(req, tenant_id="tenant-t", provider=lambda r: pytest.fail("Should hit exact"))
        assert r2.status == "HIT"
        assert r2.tier == "EXACT"

        # Operator revokes the binding
        scope = asdict(cache.get_scope("tenant-t"))
        binding = digest(approval_binding(req, req, resp, tenant_id="tenant-t", scope=scope))
        cache.revoke_entry(binding)

        # Subsequent request must NOT hit exact cache; must fall back to fresh provider call
        calls = []
        r3 = cache.process(req, tenant_id="tenant-t", provider=lambda r: (calls.append(1) or resp))
        assert r3.status == "MISS"
        assert len(calls) == 1


def test_changed_model_and_format_rejects_semantic_hit():
    """Defect A2: Requests requesting different models or output formats must not hit."""
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=ConstantScorer(prob=0.99),
    )
    req = make_request("Explain list conversion in Python.")
    resp = make_response("Use tuple() for list conversion.")
    cache.admit_candidate(req, resp, tenant_id="tenant-t")

    para = make_request("Can you explain list conversion in Python?")
    para["model"] = "different-model-id"
    para["response_format"] = {"type": "json_object"}

    calls = []
    res = cache.process(
        para,
        tenant_id="tenant-t",
        provider=lambda r: (calls.append(1) or resp),
    )
    assert res.status == "MISS"
    assert len(calls) == 1


def test_changed_conversation_history_rejects_semantic_hit():
    """Defect A2: Requests with different prior conversational history must not hit."""
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=ConstantScorer(prob=0.99),
    )

    req = make_request("How to convert a list?")
    req["messages"].insert(1, {"role": "user", "content": "Only discuss lists."})
    resp = make_response("Use tuple() to convert.")
    cache.admit_candidate(req, resp, tenant_id="tenant-t")

    # Incoming request has different conversational context
    para = make_request("How can I convert a list?")
    para["messages"].insert(1, {"role": "user", "content": "Only discuss sets."})

    calls = []
    res = cache.process(
        para,
        tenant_id="tenant-t",
        provider=lambda r: (calls.append(1) or resp),
    )
    assert res.status == "MISS"
    assert len(calls) == 1


def test_post_provider_verifier_error_falls_back_safely():
    """Defect A4: Scorer/verifier exceptions must abstain safely without crashing."""
    class FailingScorer:
        identity = "failing-scorer"
        def score(self, q, a):
            raise ValueError("diagnostic input window overflow")

    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=FailingScorer(),
    )

    req = make_request("How to sort a list?")
    resp = make_response("Use sorted(list).")

    calls = []
    # Must not raise ValueError; must return provider response safely
    res = cache.process(
        req,
        tenant_id="tenant-t",
        provider=lambda r: (calls.append(1) or resp),
    )
    assert res.status == "MISS"
    assert res.provider_calls == 1
    assert len(calls) == 1
    assert res.response["choices"][0]["message"]["content"] == resp["choices"][0]["message"]["content"]


def test_admission_timing_includes_synchronous_admission():
    """Defect B1: Total request latency must include synchronous admission time."""
    class SlowScorer:
        identity = "slow-scorer"
        def score(self, q, a):
            time.sleep(0.030)  # 30 ms
            return {"probability": 0.99, "logit": 5.0, "input_tokens": 10}

    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=SlowScorer(),
        async_promotion=False,
    )

    req = make_request("How to read a file?")
    resp = make_response("Use with open(...) as f:")

    t0 = time.perf_counter()
    res = cache.process(req, tenant_id="tenant-t", provider=lambda r: resp)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    assert res.total_latency_ms >= 25.0
    assert abs(wall_ms - res.total_latency_ms) < 20.0


def test_semantic_hit_latency_in_milliseconds():
    """Defect B2: Semantic hit latency must be reported in milliseconds, not seconds."""
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "test"))
    cache = ConservativeHybridCache(
        exact,
        scope=KnowledgeScope("docs", "v1", "public", "p1"),
        encoder=MockEncoder(),
        passage_scorer=ConstantScorer(prob=0.99),
    )

    req = make_request("How to sort a list?")
    resp = make_response("Use list.sort().")
    cache.admit_candidate(req, resp, tenant_id="tenant-t")

    para = make_request("How can I sort a list?")
    res = cache.process(para, tenant_id="tenant-t", provider=lambda r: resp)
    assert res.status == "HIT"
    assert res.tier == "SEMANTIC"
    # Latency should be non-trivial milliseconds, never sub-microsecond seconds
    assert res.total_latency_ms > 0.05


def test_live_runner_safeguards_allowlist_and_journal():
    """Defect C1, C2, C3, C4: Live runner must enforce free models, journal, token ceiling."""
    with tempfile.TemporaryDirectory() as tmp:
        missing_journal = Path(tmp) / "missing_journal.json"
        body = {"model": "diagnostic-paid-model", "messages": [{"role": "user", "content": "test"}]}

        # 1. Blocks if journal missing
        runner = LiveBenchmarkRunner(execute_live=True, journal_path=missing_journal)
        pre = runner.preflight_check()
        assert pre["status"] == "BLOCKED"
        assert any("Durable quota journal does not exist" in b for b in pre["blockers"])
        with pytest.raises(RuntimeError, match="Live execution blocked"):
            runner.execute_request(body, "req-1")

        # 2. Rejects unverified paid model immediately at construction
        from src.benchmark.free_pilot import PilotJournal
        valid_journal = Path(tmp) / "valid_journal.db"
        PilotJournal(valid_journal, run_id="init-test", fingerprint="fp")
        with pytest.raises(ValueError, match="not in the free-only allowlist"):
            LiveBenchmarkRunner(
                model="diagnostic-paid-model",
                execute_live=True,
                journal_path=valid_journal,
            )

        # 3. Request body bound to free allowlist and max_output_tokens enforced
        free_model = list(FREE_MODELS)[0]
        runner_free = LiveBenchmarkRunner(
            model=free_model,
            execute_live=True,
            journal_path=valid_journal,
            max_output_tokens=512,
            output_dir=Path(tmp) / "receipts",
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "OFFLINE_PLACEHOLDER"}):
            assert runner_free.preflight_check()["status"] == "READY_OPT_IN"

            class MockResponse:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def read(self):
                    return b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":5,"completion_tokens":5}}'

            with patch("urllib.request.urlopen", return_value=MockResponse()) as mock_urlopen:
                raw_resp = runner_free.execute_request(
                    {"model": "diagnostic-paid-model", "messages": [{"role": "user", "content": "hi"}]},
                    "test-req-1",
                )
                # Request model must be overridden to verified free model
                sent_data = json.loads(mock_urlopen.call_args.args[0].data)
                assert sent_data["model"] == free_model
                assert sent_data["max_tokens"] == 512
                receipt = runner_free.receipts[-1]
                assert receipt["status"] == "SUCCESS"
                assert receipt["spend_usd"] == 0.0
