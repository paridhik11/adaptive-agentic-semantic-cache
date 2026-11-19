"""Benchmark cache retrieval and reuse with annotated ComQA short answers."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from urllib.parse import unquote

import numpy as np
from fastapi.testclient import TestClient
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.chat_semantic import ANSWER_MIN, PAIR_MIN, SIMILARITY_MIN, load_chat_models
from src.api.live_chat import DEFAULT_MODEL, create_live_chat_app
from src.policy.decomposed_applicability import answer_satisfies_requirements, requirement_guard

TEST_SHA = "b81e19c4108198781167ddb30d26172571e76cc8e684aabb181537a45e816867"


def fold(text):
    return " ".join(text.casefold().split())


def prepare(raw):
    if sha256(raw).hexdigest() != TEST_SHA:
        raise ValueError("Use the pinned public ComQA test JSON.")
    groups = {}
    seen = set()
    for row in json.loads(raw):
        question = row["question"].strip()
        answers = frozenset(
            fold(unquote(text.strip().removeprefix("https://en.wikipedia.org/wiki/")).replace("_", " "))
            for text in row["answers"] if text.strip()
        )
        if not question or not answers or fold(question) in seen:
            continue
        seen.add(fold(question))
        groups.setdefault(answers, []).append(question)
    gallery = [(questions[0], answers) for answers, questions in groups.items()]
    probes = [(questions[1], answers, i) for i, (answers, questions) in enumerate(groups.items())
              if len(questions) >= 2]
    return gallery, probes


def answer_text(answers):
    return ", ".join(sorted(answers))


def api_pair(seed, probe, models):
    # The annotation is the full answer returned by the fixture. The live app
    # still runs its regular policy, durable cache and local-model code.
    answers = {seed[0]: answer_text(seed[1]), probe[0]: answer_text(probe[1])}

    def transport(body):
        return {"model": DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": answers[body["messages"][-1]["content"]]}}]}

    with TemporaryDirectory() as tmp:
        app = create_live_chat_app(tmp, journal_path=Path(tmp) / "quota.sqlite",
                                   api_key="offline-fixture", transport=transport, semantic_models=models)
        with TestClient(app) as client:
            client.headers["x-demo-session"] = re.search(
                r'name="demo-session" content="([^"]+)"', client.get("/").text)[1]
            for question in answers:
                response = client.post("/api/chat", json={"question": question})
                response.raise_for_status()
                result = response.json()
    return result


def band(n, total):
    if not total:
        return None
    ci = binomtest(n, total).proportion_ci()
    return [round(ci.low, 6), round(ci.high, 6)]


def summarize(counts, n):
    return {**counts,
            "hit_interval_per_probe": band(counts["supported"], n),
            "wrong_interval_per_probe": band(counts["wrong"], n),
            "wrong_interval_per_hit": band(counts["wrong"], counts["supported"] + counts["wrong"]),
            "overlap_interval_per_probe": band(counts["overlapping"], n)}


def evaluate(raw):
    gallery, probes = prepare(raw)
    models = load_chat_models()
    encoder, pair, passage = models
    texts = [seed[0] for seed in gallery] + [probe[0] for probe in probes]
    token_ids = encoder.model.tokenizer(texts, truncation=False)["input_ids"]
    if any(len(ids) > encoder.model.max_seq_length for ids in token_ids):
        raise ValueError("Input exceeds the pinned encoder window.")
    vectors = encoder.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    cosine = vectors[len(gallery):] @ vectors[:len(gallery)].T
    results = {arm: {cohort: Counter(supported=0, wrong=0, overlapping=0)
                     for cohort in cohorts}
               for arm, cohorts in {
                   "default_live_chat": ("paired", "hard_negative"),
                   "optional_live_chat": ("paired", "hard_negative"),
                   "full_question_cosine": ("gallery_top1", "hard_negative"),
                   "full_question_verified": ("gallery_top1", "hard_negative"),
               }.items()}
    ranking = Counter()
    guard_failures = Counter()
    unique_controls = set()

    def outcome(seed, probe):
        if seed[1] == probe[1]:
            return "supported"
        return "wrong" if seed[1].isdisjoint(probe[1]) else "overlapping"

    def verify(seed, probe):
        if requirement_guard(seed[0], probe[0]) or answer_satisfies_requirements(probe[0], answer_text(seed[1])):
            return "requirements"
        if min(pair.score(seed[0], probe[0])["pair_score"], pair.score(probe[0], seed[0])["pair_score"]) < PAIR_MIN:
            return "question_pair"
        if passage.score(probe[0], answer_text(seed[1]))["probability"] < ANSWER_MIN:
            return "answer_relevance"
        return "accepted"

    for scores, probe in zip(cosine, probes, strict=True):
        order = np.argsort(-scores, kind="stable")
        top = int(order[0])
        negative = next(int(i) for i in order if gallery[i][1].isdisjoint(probe[1]))
        unique_controls.add(negative)
        ranking["top1_same_answers"] += top == probe[2]
        ranking["top5_same_answers"] += probe[2] in order[:5]
        verdicts = {}
        for cohort, i in (("gallery_top1", top), ("hard_negative", negative)):
            if scores[i] < SIMILARITY_MIN:
                continue
            label = outcome(gallery[i], probe)
            results["full_question_cosine"][cohort][label] += 1
            if i not in verdicts:
                verdicts[i] = verify(gallery[i], probe)
            verdict = verdicts[i]
            guard_failures[f"{cohort}:{label}:{verdict}"] += 1
            if verdict == "accepted":
                results["full_question_verified"][cohort][label] += 1
        for arm, model_option in (("default_live_chat", None), ("optional_live_chat", models)):
            for cohort, i in (("paired", probe[2]), ("hard_negative", negative)):
                result = api_pair(gallery[i], probe, model_option)
                if result["cache_tier"] is not None:
                    label = outcome(gallery[i], probe)
                    assert result["answer"] == answer_text(gallery[i][1])
                    results[arm][cohort][label] += 1
    n = len(probes)
    return {
        "source_sha256": TEST_SHA, "gallery_seeds": len(gallery), "probes": n,
        "unique_hard_negative_seeds": len(unique_controls), "ranking": dict(ranking),
        "thresholds": {"cosine": SIMILARITY_MIN, "pair": PAIR_MIN, "passage": ANSWER_MIN},
        "model_ids": [model.identity for model in models], "guard_outcomes": dict(guard_failures),
        "results": {arm: {cohort: summarize(counts, n) for cohort, counts in rows.items()}
                    for arm, rows in results.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset.read_bytes()), indent=2))
