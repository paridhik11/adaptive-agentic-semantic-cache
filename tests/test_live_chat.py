import re
import sqlite3

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
