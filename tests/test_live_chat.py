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


def client(tmp_path, transport):
    app = create_live_chat_app(tmp_path / "chat", journal_path=tmp_path / "quota.sqlite",
                               api_key="test-placeholder", transport=transport)
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
