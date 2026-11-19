from concurrent.futures import ThreadPoolExecutor
import re
import sqlite3
from threading import Event

import pytest
from fastapi.testclient import TestClient

from src.api.live_chat import create_live_chat_app, RUN_LIMIT
from src.api.providers import ProviderError

QUESTION = "Explain the difference between mean and median."


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.failure = False

    def __call__(self, body):
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}
        self.requests.append(body)
        if self.failure:
            raise ProviderError("upstream_unavailable")
        return {
            "model": body["model"],
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": f"Answer {len(self.requests)}"}}],
            "usage": {"completion_tokens": 2},
        }


def client(tmp_path, transport, semantic_models=None):
    app = create_live_chat_app(tmp_path / "chat", journal_path=tmp_path / "quota.sqlite",
                               api_key="test-placeholder", transport=transport, semantic_models=semantic_models)
    c = TestClient(app)
    page = c.get("/")
    c.headers["x-demo-session"] = re.search(r'name="demo-session" content="([^"]+)"', page.text)[1]
    return c


def test_live_chat_miss_exact_hit_and_restart(tmp_path):
    transport = FakeTransport()
    with client(tmp_path, transport) as c:
        first = c.post("/api/chat", json={"question": QUESTION}).json()
        assert first["cache_status"] == "MISS" and first["provider_calls"] == 1
        assert first["provider_ms"] > 0 and first["latency_ms"] >= first["provider_ms"]
        assert first["generated_tokens"] == 2 and first["stored"]
        repeat = c.post("/api/chat", json={"question": QUESTION}).json()
        assert repeat["cache_status"] == "HIT" and repeat["provider_calls"] == 0
        assert repeat["latency_ms"] >= 0 and repeat["provider_ms"] is None and repeat["generated_tokens"] == 0
        assert repeat["answer"] == first["answer"] and repeat["used"] == 1
        assert len(transport.requests) == 1
    with client(tmp_path, transport) as c:
        saved = c.post("/api/chat", json={"question": QUESTION}).json()
        assert saved["answer"] == first["answer"] and saved["provider_calls"] == 0
        new = c.post("/api/chat", json={"question": "Define entropy."}).json()
        assert new["provider_calls"] == 1 and new["used"] == 2


def test_general_chat_reuses_a_plain_language_risk_definition(tmp_path):
    transport = FakeTransport()
    question = "what do u mean by risk management ?"
    with client(tmp_path, transport) as c:
        fresh = c.post("/api/chat", json={"question": question}).json()
        repeat = c.post("/api/chat", json={"question": question}).json()
        assert transport.requests[0]["messages"][-1]["content"] == question
        assert fresh["provider_calls"] == 1 and fresh["stored"]
        assert repeat["provider_calls"] == 0 and repeat["cache_tier"] == "EXACT"


def test_default_chat_reuses_risk_definition_across_typos_and_shorthand(tmp_path):
    first_question = "what is risk managemnet"
    definition = "**Risk management** is the process of identifying, assessing, and controlling risks."

    class DefinitionTransport(FakeTransport):
        def __call__(self, body):
            result = super().__call__(body)
            if body["messages"][-1]["content"] == first_question:
                result["choices"][0]["message"]["content"] = definition
            return result

    transport = DefinitionTransport()
    with client(tmp_path, transport) as c:
        first = c.post("/api/chat", json={"question": first_question}).json()
        assert first["provider_calls"] == 1 and first["stored"]
    with client(tmp_path, transport) as c:
        for query in ("what do u mean by risk managemgcnt", "risk management?",
                      "What's risk management?", "Explain what risk management means."):
            hit = c.post("/api/chat", json={"question": query}).json()
            assert hit["cache_tier"] == "SEMANTIC" and hit["provider_calls"] == 0, hit
            assert hit["answer"] == definition and hit["generated_tokens"] == 0
            assert hit["semantic_match"] == {"matched_question": first_question, "similarity": None}
        assert len(transport.requests) == 1
        for query in ("What is risk measurement?", "What is market risk management?",
                      "What is risk management? Give a worked example."):
            fresh = c.post("/api/chat", json={"question": query}).json()
            assert fresh["provider_calls"] == 1 and fresh["cache_tier"] is None
        unconfirmed = c.post("/api/chat", json={"question": "risk treatment?"}).json()
        assert unconfirmed["provider_calls"] == 1 and not unconfirmed["stored"]


def test_live_chat_context_and_time_sensitive_questions_bypass(tmp_path):
    transport = FakeTransport()
    with client(tmp_path, transport) as c:
        initial = c.post("/api/chat", json={"question": QUESTION}).json()
        context = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": initial["answer"]}]
        followup = c.post("/api/chat", json={"question": "Give an example.", "history": context}).json()
        assert followup["cache_status"] == "BYPASS" and followup["provider_calls"] == 1
        assert transport.requests[-1]["messages"][1:-1] == context
        for question in ("What is the weather today?", "What is the weather today?"):
            fresh = c.post("/api/chat", json={"question": question}).json()
            assert fresh["cache_reason"] == "freshness_requires_provider" and fresh["provider_calls"] == 1
        manual = c.post("/api/chat", json={"question": QUESTION, "cache": False}).json()
        assert manual["cache_reason"] == "cache_disabled_by_caller" and manual["provider_calls"] == 1
        assert c.post("/api/chat", json={"question": QUESTION, "history": [{"role": "system", "content": "unsafe"}]}).status_code == 400
        assert c.post("/api/chat", headers={"Origin": "https://elsewhere.test"}, json={"question": QUESTION}).status_code == 403


@pytest.mark.parametrize(("left", "right", "different"), [
    ("precision", "recall", "specificity"),
    ("bonds", "equities", "derivatives"),
    ("binary search", "linear search", "hash lookup"),
    ("variance", "standard deviation", "interquartile range"),
])
def test_comparison_reuse_across_domains_preserves_order_and_scope(tmp_path, left, right, different):
    seed = f"Compare {left} and {right}."
    answer = f"**{left}** and **{right}** describe different concepts. Here is the distinction."

    class ComparisonTransport(FakeTransport):
        def __call__(self, body):
            result = super().__call__(body)
            if body["messages"][-1]["content"] == seed:
                result["choices"][0]["message"]["content"] = answer
            return result

    with client(tmp_path, ComparisonTransport()) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["provider_calls"] == 1
        for wording in (f"{left} vs. {right}?", f"What's the difference between {left} and {right}?"):
            hit = c.post("/api/chat", json={"question": wording}).json()
            assert hit["answer"] == answer and hit["semantic_decision"] == "comparison_verified", hit
            assert hit["provider_calls"] == 0
        for wording in (f"{right} vs {left}?", f"{left} vs {different}?",
                        f"{left} vs {right} for retail clients?",
                        f"Compare {left} and {right} as of Q3."):
            miss = c.post("/api/chat", json={"question": wording}).json()
            assert miss["provider_calls"] == 1 and miss["cache_tier"] is None, miss


def test_cached_answer_and_status_can_finish_while_provider_is_busy(tmp_path):
    started = Event()
    release = Event()

    class BlockingTransport(FakeTransport):
        def __call__(self, body):
            if body["messages"][-1]["content"] != QUESTION:
                started.set()
                assert release.wait(timeout=5)
            return super().__call__(body)

    with client(tmp_path, BlockingTransport()) as c, ThreadPoolExecutor(max_workers=3) as pool:
        assert c.post("/api/chat", json={"question": QUESTION}).json()["provider_calls"] == 1
        # A fresh creative request bypasses the cache, isolating the provider lock.
        slow = pool.submit(c.post, "/api/chat", json={"question": "Draft a weekly exercise plan."})
        try:
            assert started.wait(timeout=3)
            info = pool.submit(c.get, "/api/chat").result(timeout=2).json()
            assert info["used"] == 2  # includes the outstanding reservation
            repeat = pool.submit(c.post, "/api/chat", json={"question": QUESTION}).result(timeout=2).json()
            assert repeat["cache_tier"] == "EXACT" and repeat["provider_calls"] == 0
        finally:
            release.set()
        assert slow.result(timeout=3).json()["provider_calls"] == 1


def test_live_chat_records_incomplete_answer_as_spent_error_and_allows_manual_retry(tmp_path):
    class IncompleteTransport(FakeTransport):
        def __call__(self, body):
            result = super().__call__(body)
            if len(self.requests) == 1:
                result["choices"][0]["finish_reason"] = "length"
            return result

    with client(tmp_path, IncompleteTransport()) as c:
        failure = c.post("/api/chat", json={"question": QUESTION})
        assert failure.status_code == 503 and failure.json()["provider_calls"] == 1
        assert "shorter question" in failure.json()["error"]
        retry = c.post("/api/chat", json={"question": QUESTION}).json()
        assert retry["provider_calls"] == 1 and retry["used"] == 2
        with sqlite3.connect(tmp_path / "quota.sqlite") as db:
            assert db.execute("SELECT state FROM pilot_calls ORDER BY ordinal").fetchall() == [
                ("error",), ("complete",)]


def test_invalid_unicode_does_not_use_provider_quota(tmp_path):
    transport = FakeTransport()
    with client(tmp_path, transport) as c:
        result = c.post("/api/chat", content=b'{"question":"bad \\ud800"}',
                        headers={"Content-Type": "application/json"})
        assert result.status_code == 400
        assert c.get("/api/chat").json()["used"] == 0 and not transport.requests


def test_live_chat_failure_is_not_cached_and_hits_survive_quota(tmp_path):
    transport = FakeTransport()
    with client(tmp_path, transport) as c:
        transport.failure = True
        denied = c.post("/api/chat", json={"question": QUESTION})
        assert denied.status_code == 503 and denied.json()["provider_calls"] == 1
        transport.failure = False
        assert c.post("/api/chat", json={"question": QUESTION}).json()["provider_calls"] == 1
        with sqlite3.connect(tmp_path / "quota.sqlite") as db:
            db.executemany(
                "INSERT INTO pilot_calls VALUES ('general-chat-v1', ?, 'sha', 0, 'complete', '{}')",
                ((ordinal,) for ordinal in range(2, RUN_LIMIT)),
            )
        miss = c.post("/api/chat", json={"question": "Define entropy."})
        assert miss.status_code == 503 and miss.json()["provider_calls"] == 0
        repeat = c.post("/api/chat", json={"question": QUESTION}).json()
        assert repeat["cache_status"] == "HIT" and repeat["provider_calls"] == 0
        assert len(transport.requests) == 2


def test_live_chat_requires_key_and_explicit_free_model(tmp_path):
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_live_chat_app(tmp_path, journal_path=tmp_path / "quota.sqlite", api_key=None)
    with pytest.raises(ValueError, match="free model IDs"):
        create_live_chat_app(tmp_path, journal_path=tmp_path / "quota.sqlite", api_key="test-placeholder",
                             model="some/model")


class Encoder:
    identity = "fixture-encoder"

    def encode(self, text):
        return [1.0, 0.0]


class PairScorer:
    identity = "fixture-pair"

    def score(self, question, original):
        return {"pair_score": 0.1 if "photon" in question or "photon" in original else 0.99}


class AnswerScorer:
    identity = "fixture-answer"

    def score(self, question, answer):
        return {"probability": 0.99}


def test_semantic_hit_persists_and_exact_stays_first(tmp_path):
    models = Encoder(), PairScorer(), AnswerScorer()
    transport = FakeTransport()
    with client(tmp_path, transport) as c:
        first = c.post("/api/chat", json={"question": "what is a photo"}).json()
        assert first["provider_calls"] == 1
    with client(tmp_path, transport, models) as c:
        warmed = c.post("/api/chat", json={"question": "what is a photo"}).json()
        assert warmed["cache_tier"] == "EXACT" and warmed["provider_calls"] == 0
    with client(tmp_path, transport, models) as c:
        semantic = c.post("/api/chat", json={"question": "what is a photograph"}).json()
        assert semantic["cache_tier"] == "SEMANTIC" and semantic["provider_calls"] == 0
        assert semantic["answer"] == first["answer"] and semantic["generated_tokens"] == 0
        assert semantic["semantic_match"]["matched_question"] == "what is a photo"
        assert semantic["semantic_hits"] == 1 and len(transport.requests) == 1
        assert c.post("/api/chat", json={"question": "what is a photo"}).json()["cache_tier"] == "EXACT"


def test_similarity_alone_is_insufficient_and_bypass_is_preserved(tmp_path):
    transport = FakeTransport()
    with client(tmp_path, transport, (Encoder(), PairScorer(), AnswerScorer())) as c:
        c.post("/api/chat", json={"question": "what is a photo"})
        changed = c.post("/api/chat", json={"question": "what is a digital photo"}).json()
        assert changed["provider_calls"] == 1 and changed["cache_tier"] is None
        for text, extra in [
            ("what is a photograph", {"cache": False}),
            ("what is a photograph", {"history": [{"role": "user", "content": "In optics"},
                                                   {"role": "assistant", "content": "Okay."}]}),
            ("what is a photo without a lens", {}),
            ("what is a photon", {}),
        ]:
            fresh = c.post("/api/chat", json={"question": text, **extra}).json()
            assert fresh["provider_calls"] == 1 and fresh["cache_tier"] is None


@pytest.mark.requires_model
def test_photo_paraphrase_with_pinned_local_models(tmp_path):
    from src.api.chat_semantic import load_chat_models

    class RealisticTransport(FakeTransport):
        def __call__(self, body):
            result = super().__call__(body)
            if body["messages"][-1]["content"] == "what is a photo":
                result["choices"][0]["message"]["content"] = (
                    "A photo, short for photograph, is an image made when light falls "
                    "on photographic film or an electronic image sensor."
                )
            return result

    transport = RealisticTransport()
    with client(tmp_path, transport, load_chat_models()) as c:
        first = c.post("/api/chat", json={"question": "what is a photo"}).json()
        for text in ("what is a photograph", "define a photograph", "what does photograph mean"):
            paraphrase = c.post("/api/chat", json={"question": text}).json()
            assert paraphrase["cache_tier"] == "SEMANTIC" and paraphrase["provider_calls"] == 0, paraphrase
            assert paraphrase["answer"] == first["answer"]
        for text in ("what is a photon", "what is photography", "what is a digital photo"):
            different = c.post("/api/chat", json={"question": text}).json()
            assert different["provider_calls"] == 1 and different["cache_tier"] is None
