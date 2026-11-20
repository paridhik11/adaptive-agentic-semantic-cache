"""Regression checks for experimental scoring and the answer-level denominators."""

from pathlib import Path

import pytest


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_independent_semantics

    return evaluate_independent_semantics


def candidate(**changes):
    return {
        "cosine": 0.96,
        "pair_forward": 0.99,
        "pair_reverse": 0.96,
        "passage": 0.04,
        "passage_logit": -3.18,
        "source_passage_logit": -3.2,
        "guard": None,
        "extra_guard": None,
        "answer_words": 2,
        **changes,
    }


@pytest.mark.parametrize(
    "original,incoming",
    [
        ("Convert a tuple to a list", "Convert a list to a tuple"),
        ("Use version 3.10", "Use version 3.12"),
        ("Use the list in place", "Do not use the list in place"),
        ("Show FY 2024 revenue", "Show FY 2025 revenue"),
    ],
)
def test_relative_score_cannot_override_requirements(study, original, incoming):
    failure = study.requirement_guard(original, incoming)
    assert failure
    assert study.reject(candidate(guard=failure), relative=True, typed=False) == failure


def test_actual_answer_and_both_directions_matter(study):
    assert study.reject(candidate(), True, False) is None
    assert study.reject(candidate(), False, False) == "passage"
    for changes in [
        {"pair_reverse": 0.94},
        {"answer_words": 13},
        {"passage": 0.009},
        {"source_passage_logit": -2.4},
    ]:
        assert study.reject(candidate(**changes), True, False) == "passage"
    reason = study.semantic_answer_reason(
        "Profile my Python script",
        "Profile my Python script using cProfile",
        "The tool cProfile exists.",
        None,
    )
    assert reason
    assert study.reject(candidate(extra_guard=reason), True, True) == reason


def test_ranked_candidate_and_ambiguity(study):
    order = [1, 0, 2]
    scores = {1: 0.97, 0: 0.96, 2: 0.6}
    row = (0, 0, order, order, scores, scores)
    trace = {(0, j): candidate(cosine=scores[j]) for j in order}
    assert study.decision(row, trace, ["relative5"]) == (None, "ambiguous")
    trace[0, 1]["guard"] = "entity_changed"
    assert study.decision(row, trace, ["relative5"]) == (0, "accepted")


def test_unscored_exact_repeat_does_not_inflate_semantic_precision(study):
    events = [
        dict(kind="present", tier="SEMANTIC", correct=True, reason="accepted"),
        dict(kind="present", tier=None, correct=False, reason="passage"),
        dict(kind="absent", tier="SEMANTIC", correct=False, reason="accepted"),
        dict(kind="exact", tier="EXACT", correct=True, reason="exact"),
    ]
    result = study.summarize(events)
    assert result["recall"]["rate"] == 0.5
    assert result["precision"]["rate"] == 0.5
    assert result["absent_error"]["n"] == 1
    assert result["exact_hits"] == 1
    assert study.score_record(0, 0)["rate"] is None


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "exact"),
        (["--guarded-rules"], "guarded"),
        (["--experimental-semantic"], "guarded"),
    ],
)
def test_live_launcher_requires_explicit_semantic_opt_in(
    study, monkeypatch, tmp_path, options, expected
):
    import sys
    import uvicorn
    import run_demo
    from src.api import chat_semantic, live_chat
    from fastapi.testclient import TestClient

    factory = live_chat.create_live_chat_app
    monkeypatch.setattr(chat_semantic, "load_chat_models", lambda: None)

    def local_factory(*args, **kwargs):
        def fixture(_):
            return {
                "model": live_chat.DEFAULT_MODEL,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "A photo (short for photograph) is an image recorded by a camera.",
                        },
                    }
                ],
            }

        return factory(*args, transport=fixture, **kwargs)

    monkeypatch.setattr(live_chat, "create_live_chat_app", local_factory)

    def inspect(app, **kwargs):
        import re

        with TestClient(app) as client:
            client.headers["x-demo-session"] = re.search(
                r'name="demo-session" content="([^"]+)"', client.get("/").text
            )[1]

            def ask(q):
                res = client.post("/api/chat", json={"question": q})
                res.raise_for_status()
                return res.json()

            assert client.get("/api/chat").json()["semantic_mode"] == expected
            assert ask("What is a photo?")["provider_calls"] == 1
            assert ask("What is a photo?")["cache_tier"] == "EXACT"
            response = ask("What is a photograph?")
            assert response["provider_calls"] == (expected == "exact")
            assert (response["cache_tier"] == "SEMANTIC") == (expected == "guarded")

    monkeypatch.setattr(uvicorn, "run", inspect)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo",
            "--directory",
            str(tmp_path),
            "--journal",
            str(tmp_path / "fixture.db"),
            *options,
        ],
    )
    run_demo.main()
