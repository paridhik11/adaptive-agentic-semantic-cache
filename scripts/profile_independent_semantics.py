"""Time each frozen research arm independently and count offline fallback calls."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate_independent_semantics import (
    ARMS,
    BASE,
    CAP,
    decision,
    dump,
    elapsed,
    embed,
    load_chat_models,
    question_type,
)
from independent_semantic_data import comqa, pinned, python_docs
from src.api.chat_keywords import keyword_signature
from src.api.chat_reuse_checks import semantic_answer_reason
from src.policy.decomposed_applicability import (
    answer_satisfies_requirements,
    requirement_guard,
)


def timed_candidate(q, seed, answer, cosine, config, models, reference):
    # Fail early; fields below a rejected stage are ignored by decision().
    c = dict(
        cosine=cosine,
        guard=None,
        extra_guard=None,
        pair_forward=0.0,
        pair_reverse=0.0,
        passage=0.0,
        passage_logit=-100.0,
        source_passage_logit=reference,
        answer_words=len(answer.split()),
    )
    if cosine < 0.84:
        return c
    c["guard"] = requirement_guard(seed, q) or answer_satisfies_requirements(q, answer)
    if c["guard"]:
        return c
    if "typed5" in config:
        a, b = question_type(seed), question_type(q)
        c["extra_guard"] = (
            "answer_type_changed"
            if a and b and a != b
            else semantic_answer_reason(seed, q, answer, keyword_signature(q))
        )
        if c["extra_guard"]:
            return c
    pair, passage = models[1:]
    c["pair_forward"] = pair.score(q, seed)["pair_score"]
    if c["pair_forward"] < 0.80:
        return c
    c["pair_reverse"] = pair.score(seed, q)["pair_score"]
    if c["pair_reverse"] >= 0.80:
        scores = passage.score(q, answer)
        c.update(passage=scores["probability"], passage_logit=scores["logit"])
    return c


def profile(rows, config, models, expected):
    encoder, _, passage = models
    obs, verification, latency = Counter(), [], []
    priming_ms = 0.0

    def fresh():
        obs["fixture_calls"] += 1
        return row[4]

    expected = {(e["id"], e["kind"]): e["source"] for e in expected}
    for shard in (0, 1):
        t = perf_counter()
        ids = list(range(shard, len(rows), 2))
        seeds = [rows[j][2] for j in ids]
        exact = {rows[j][2]: rows[j][4] for j in ids}
        vectors = embed(encoder, seeds)
        refs = (
            {j: passage.score(rows[j][2], rows[j][4])["logit"] for j in ids}
            if "relative5" in config
            else {}
        )
        if "hybrid5" in config:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
            words = tfidf.fit_transform(seeds)
        priming_ms += elapsed(t)
        for i, row in enumerate(rows):
            if i in ids:
                assert exact[row[2]] == row[4]
                obs["exact_hits"] += 1
            q = row[3]
            start = perf_counter()
            scores = vectors @ encoder.encode(q)
            cosine = dict(zip(ids, scores, strict=True))
            dense = sorted(ids, key=lambda j: -cosine[j])
            order, blended = dense, cosine
            if "hybrid5" in config:
                lexical = (tfidf.transform([q]) @ words.T).toarray()[0]
                blended = dict(zip(ids, 0.75 * scores + 0.25 * lexical, strict=True))
                order = sorted(ids, key=lambda j: -blended[j])
            k = 1 if config == [BASE] else 5
            traced = {
                (i, j): timed_candidate(
                    q,
                    rows[j][2],
                    rows[j][4],
                    cosine[j],
                    config,
                    models,
                    refs.get(j, 0.0),
                )
                for j in order[:k]
            }
            j, _ = decision((i, shard, dense, order, cosine, blended), traced, config)
            verification.append(elapsed(start))
            result = rows[j][4] if j is not None else fresh()
            latency.append(elapsed(start))
            kind = "present" if i in ids else "absent"
            actual_source = rows[j][0] if j is not None else None
            if actual_source != expected[row[0], kind]:
                raise ValueError(
                    f"Online and batch decisions disagree on {row[0]} / {kind}"
                )
            if j is not None:
                obs["semantic_hits"] += 1
                obs["correct"] += result == row[4]
    return dict(
        **obs,
        probes=len(latency),
        total_requests=len(latency) + len(rows),
        calls_avoided=len(latency) + len(rows) - obs["fixture_calls"],
        priming_ms=priming_ms,
        verification_p50_p95_ms=np.percentile(verification, [50, 95]).tolist(),
        fixture_end_to_end_p50_p95_ms=np.percentile(latency, [50, 95]).tolist(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--split", choices=["development", "holdout"], required=True)
    args = ap.parse_args()
    pinned(args.directory, False)
    models = load_chat_models()
    selection = json.loads((args.output / "selection.json").read_bytes())
    with gzip.open(args.output / (args.split + "-traces.json.gz")) as f:
        traces = json.load(f)
    result = {}
    for name, get_data in [("comqa", comqa), ("python", python_docs)]:
        rows = [r for r in get_data(args.directory) if r[1] == args.split][: CAP[name]]
        result[name] = {}
        for arm in [*ARMS, "combined"]:
            config = selection["combined"] if arm == "combined" else [arm]
            result[name][arm] = profile(
                rows, config, models, traces[name]["events"][arm]
            )
    dump(args.output / (args.split + "-profile.json"), result)


if __name__ == "__main__":
    main()
