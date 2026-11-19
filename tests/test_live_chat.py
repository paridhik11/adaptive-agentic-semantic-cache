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
        for wording in (f"{left} vs. {right}?", f"What's the difference between {left} and {right}?",
                        f"How does {left} differ from {right}?"):
            hit = c.post("/api/chat", json={"question": wording}).json()
            assert hit["answer"] == answer and hit["semantic_decision"] == "comparison_verified", hit
            assert hit["provider_calls"] == 0
        for wording in (f"{right} vs {left}?", f"How does {right} differ from {left}?", f"{left} vs {different}?",
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


# Each group gets an empty store and independent transport. The seed returns
# the supported answer. Every provider call for other prompts returns a
# different string so the replay cannot confuse a miss with a cached answer.
FACT_REPLAY = [
    ("1st prime minister of india?", "Jawaharlal Nehru was the first prime minister of India.",
     ["india's first prime ministerr?", "Who was the first PM of India?"],
     ["1st prime minister of Pakistan?", "Who is the current prime minister of India?",
      "2nd prime minister of India?"]),
    ("Who was the second president of Ireland?", "Seán T. O'Kelly.",
     ["Ireland's 2nd president?", "2nd president of Ireland?"],
     ["Ireland's first president?", "Second president of Iceland?"]),
    ("Who was the first governor of California?", "Peter Hardeman Burnett.",
     ["California's 1st governor?", "1st governor of California?"],
     ["Who was the first female governor of California?", "First governor of Nevada?"]),
    ("Where was Ada Lovelace born?", "London, England.",
     ["What was Ada Lovelace's birthplace?", "Place of birth of Ada Lovelace?"],
     ["When was Ada Lovelace born?", "Where was Mary Shelley born?"]),
    ("When was Alan Turing born?", "23 June 1912.",
     ["Alan Turing's date of birth?", "What was Alan Turing's birth date?"],
     ["Alan Turing's place of birth?", "When was Alan Kay born?"]),
    ("Who wrote The Hobbit?", "J. R. R. Tolkien.",
     ["Who is the author of The Hobbit?", "Who was The Hobbit written by?"],
     ["Who wrote The Silmarillion?", "Who translated The Hobbit?"]),
    ("How do I calculate sample variance?", "Divide the sum of squared deviations from the mean by n minus one.",
     ["How can I compute sample variance?", "How is sample variance calculated?"],
     ["How is population variance calculated?", "How do I calculate sample variance in Excel?"]),
    ("How do I calculate expected credit loss?", "A simple estimate multiplies PD by LGD and EAD.",
     ["How can I compute expected credit loss?", "How is expected credit loss calculated?"],
     ["How do I calculate unexpected credit loss?", "How is expected credit loss calculated for secured loans?"]),
]


class AnswersTransport(FakeTransport):
    def __init__(self, answers):
        super().__init__()
        self.answers = answers

    def __call__(self, body):
        response = super().__call__(body)
        question = body["messages"][-1]["content"]
        if question in self.answers:
            response["choices"][0]["message"]["content"] = self.answers[question]
        return response


@pytest.mark.parametrize(("seed", "answer", "rephrases", "changed"), FACT_REPLAY)
def test_factual_semantic_reuse_and_changed_requirements(tmp_path, seed, answer, rephrases, changed):
    transport = AnswersTransport({seed: answer})
    with client(tmp_path, transport) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["stored"]
    with client(tmp_path, transport) as c:
        for question in rephrases:
            hit = c.post("/api/chat", json={"question": question}).json()
            assert hit["cache_tier"] == "SEMANTIC" and hit["provider_calls"] == 0, hit
            assert hit["answer"] == answer
            assert hit["semantic_match"]["matched_question"] == seed
        for question in changed:
            response = c.post("/api/chat", json={"question": question}).json()
            assert response["provider_calls"] == 1 and response["cache_tier"] is None
            assert response["answer"] != answer
        assert len(transport.requests) == 1 + len(changed)


def test_ranked_role_context_is_explicit_and_requires_a_scope(tmp_path):
    seed = "1st prime minister of india?"
    answer = "Jawaharlal Nehru."
    ambiguous = "1st prime minister?"
    transport = AnswersTransport({seed: answer})
    with client(tmp_path, transport) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["provider_calls"] == 1
        hit = c.post("/api/chat", json={"question": ambiguous, "previous_question": seed}).json()
        assert hit["answer"] == answer and hit["cache_tier"] == "SEMANTIC" and hit["provider_calls"] == 0
        assert hit["resolved_question"] == "Who was the 1st prime minister of india?"
        for previous in ("", "Who was the first president of France?", "Define portfolio risk."):
            ask = c.post("/api/chat", json={"question": ambiguous, "previous_question": previous}).json()
            assert ask["cache_status"] == "CLARIFY" and ask["provider_calls"] == 0
        reply = c.post("/api/chat", json={"question": "India", "previous_question": ambiguous}).json()
        assert reply["answer"] == answer and reply["provider_calls"] == 0
        changed_rank = c.post("/api/chat", json={"question": "2nd PM?", "previous_question": seed}).json()
        assert changed_rank["provider_calls"] == 1
        assert transport.requests[-1]["messages"][-1]["content"] == "Who was the 2nd prime minister of india?"
        assert len(transport.requests) == 2


def test_explicit_aliases_and_polite_wrappers_reuse_without_models(tmp_path):
    transport = AnswersTransport({
        "what is a photo?": "A photo (short for photograph) is an image captured with light.",
        "What is VIF?": "VIF (short for variance inflation factor) is a measure of multicollinearity.",
    })
    with client(tmp_path, transport) as c:
        c.post("/api/chat", json={"question": "what is a photo?"})
        for question in ("What is a photograph?", "Could you tell me what is a photo?"):
            r = c.post("/api/chat", json={"question": question}).json()
            assert r["cache_tier"] == "SEMANTIC" and r["provider_calls"] == 0
        assert c.post("/api/chat", json={"question": "What is a photon?"}).json()["provider_calls"] == 1
        c.post("/api/chat", json={"question": "What is VIF?"})
        r = c.post("/api/chat", json={"question": "Define variance inflation factor"}).json()
        assert r["cache_tier"] == "SEMANTIC" and r["provider_calls"] == 0


def test_event_time_preserves_precision_date_and_case_sensitive_measure(tmp_path):
    transport = AnswersTransport({"When did the Berlin Wall fall?": "1989."})
    with client(tmp_path, transport) as c:
        c.post("/api/chat", json={"question": "When did the Berlin Wall fall?"})
        year = c.post("/api/chat", json={"question": "In which year did the Berlin Wall fall?"}).json()
        assert year["cache_tier"] == "SEMANTIC" and year["answer"] == "1989."
        date = c.post("/api/chat", json={"question": "On what date did the Berlin Wall fall?"}).json()
        assert date["provider_calls"] == 1 and date["cache_tier"] is None
        c.post("/api/chat", json={"question": "How do I calculate VaR?"})
        var = c.post("/api/chat", json={"question": "How can I compute VAR?"}).json()
        assert var["provider_calls"] == 1 and var["cache_tier"] is None


KEYWORD_REPLAY = [
    ("Who designed the Sydney Opera House?", "Jørn Utzon designed the Sydney Opera House.",
     ["Who was the Sydney Opera House designed by?", "Could you tell me who designed the Sydney Opera House?"],
     ["Who was the Sydney Opera House named by?", "Who designed the Oslo Opera House?",
      "Who was the Sydney Opera House designed for?"]),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan.",
     ["Japan's capital city?", "Capital city of Japan?"],
     ["What is the capital of China?", "What was the capital of Japan?", "What was the capital of Japan in 1600?",
      "What is the largest city in Japan?"]),
    ("Why is ice less dense than liquid water?", "Ice forms an open crystal lattice.",
     ["Explain why ice is less dense than liquid water.", "Could you explain why ice is less dense than liquid water?"],
     ["Why is liquid water less dense than ice?", "Why is ice denser than liquid water?",
      "What is ice less dense than?"]),
    ("How does a heat pump work?", "A heat pump moves heat by compressing and expanding refrigerant.",
     ["Explain how a heat pump operates.", "How does the heat pump function?"],
     ["How does a ground-source heat pump work?", "What is a heat pump used for?",
      "How does a heat pump work below 0°C?"]),
    ("How do I sort rows by revenue descending?", "Sort the rows using revenue as a descending sort key.",
     ["Show me how to sort rows by revenue descending.", "How can one sort rows by revenue descending?"],
     ["How to sort rows by revenue ascending?", "How to sort rows by descending revenue then region?",
      "How to sort rows by revenue descending in Excel?"]),
    ("How to convert a tuple to a list?", "Pass the tuple to list() to create a list.",
     ["Can you explain how to convert from a tuple to a list?",
      "How can I convert to a list from a tuple?"],
     ["How do I convert a list to a tuple?", "How to convert a tuple to a list without copying?",
      "How to convert a tuple to a list with an example?"]),
    ("What is the purpose of an index fund?", "An index fund aims to track a market index.",
     ["What is an index fund used for?", "What do people use an index fund for?"],
     ["What is an index fund?", "What is the purpose of a bond index fund?",
      "What is the purpose of an index fund today?"]),
    ("What is the formula for weighted average?", "Divide the sum of weighted values by the sum of the weights.",
     ["Give me the formula for weighted average.", "How is the formula for weighted average written?"],
     ["Give me the formula for weighted median.", "Give me the formula for weighted average with an example.",
      "Give me the formula for weighted average of returns."]),
    ("Who was India's first prime minister?", "Jawaharlal Nehru was the first prime minister of India.",
     ["Who served as the first prime minister of India?", "Name the first person to serve as India's prime minister."],
     ["Who served as the second prime minister of India?", "Who served as the first prime minister of Pakistan?",
      "Who serves as the current prime minister of India?"]),
]


@pytest.mark.parametrize(("seed", "answer", "rephrases", "changed"), KEYWORD_REPLAY)
def test_ordered_keyword_reuse(tmp_path, seed, answer, rephrases, changed):
    transport = AnswersTransport({seed: answer})
    with client(tmp_path, transport) as c:
        first = c.post("/api/chat", json={"question": seed}).json()
        assert first["stored"] and first["provider_calls"] == 1, first
    with client(tmp_path, transport) as c:
        for question in rephrases:
            result = c.post("/api/chat", json={"question": question}).json()
            assert result["cache_tier"] == "SEMANTIC" and result["provider_calls"] == 0, result
            assert result["answer"] == answer
            assert result["semantic_match"]["matched_question"] == seed
        for question in changed:
            result = c.post("/api/chat", json={"question": question}).json()
            assert result["provider_calls"] == 1 and result["cache_tier"] is None, result
        assert len(transport.requests) == 1 + len(changed)


def test_keyword_upgrade_backfills_existing_answers_and_respects_source_expiry(tmp_path):
    seed = "How does a heat pump work?"
    paraphrase = "Explain how a heat pump operates."
    answer = "A heat pump moves heat across a temperature gradient."
    transport = AnswersTransport({seed: answer})
    with client(tmp_path, transport) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["stored"]
    database = tmp_path / "chat" / "chat.sqlite"
    with sqlite3.connect(database) as db:
        source = db.execute("SELECT key, expires, source_token FROM entries").fetchone()
        db.execute("DROP INDEX semantic_candidates_lookup_tag")
        db.execute("ALTER TABLE semantic_candidates DROP COLUMN lookup_tag")
    with client(tmp_path, transport) as c:
        result = c.post("/api/chat", json={"question": paraphrase}).json()
        assert result["cache_reason"] == "ordered_keyword_hit" and result["provider_calls"] == 0
        assert result["answer"] == answer
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT key, expires, source_token FROM entries").fetchone() == source
            assert db.execute("SELECT lookup_tag FROM semantic_candidates").fetchone()[0]
            db.execute("UPDATE entries SET expires = 0")
        result = c.post("/api/chat", json={"question": paraphrase}).json()
        assert result["provider_calls"] == 1 and result["cache_tier"] is None


def test_keyword_order_retains_primary_sort_key(tmp_path):
    seed = "How do I sort rows by region then revenue?"
    transport = AnswersTransport({seed: "Sort by region first, then revenue within each region."})
    with client(tmp_path, transport) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["stored"]
        reverse = c.post("/api/chat", json={"question": "How to sort rows by revenue then region?"}).json()
        assert reverse["provider_calls"] == 1 and reverse["cache_tier"] is None
        match = c.post("/api/chat", json={"question": "Show me how to sort rows by region then revenue."}).json()
        assert match["provider_calls"] == 0 and match["semantic_match"]["matched_question"] == seed



def test_keyword_definition_keeps_case_sensitive_abbreviations(tmp_path):
    seed = "Define VaR."
    transport = AnswersTransport({seed: "VaR is a quantile-based measure of portfolio losses."})
    with client(tmp_path, transport) as c:
        assert c.post("/api/chat", json={"question": seed}).json()["stored"]
        result = c.post("/api/chat", json={"question": "What is VAR?"}).json()
        assert result["provider_calls"] == 1 and result["cache_tier"] is None
