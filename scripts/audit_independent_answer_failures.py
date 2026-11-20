"""Stress frozen policies with wrong stored answers and audit the spent 168 cases."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluate_chat_cosine import prepare, answer_text
from evaluate_independent_semantics import (
    ARMS,
    CAP,
    batch_scores,
    dump,
    embed,
    fixture_app,
    load_chat_models,
    question_type,
    reject,
    score_record,
)
from independent_semantic_data import comqa, pinned, python_docs
from src.api.chat_keywords import keyword_signature
from src.api.chat_reuse_checks import semantic_answer_reason
from src.policy.decomposed_applicability import (
    answer_satisfies_requirements,
    requirement_guard,
)


def corrupted(rows, models, selection):
    encoder, pair, passage = models
    seeds = [r[2] for r in rows]
    vectors = embed(encoder, seeds)
    incoming = embed(encoder, [r[3] for r in rows])
    sims = vectors @ vectors.T
    cohort = []
    for i, r in enumerate(rows):
        other = [j for j in range(len(rows)) if j != i]
        same_type = [
            j
            for j in other
            if question_type(r[2]) and question_type(r[2]) == question_type(seeds[j])
        ]
        j = max(same_type or other, key=lambda n: sims[i, n])
        assert rows[j][4] != r[4]
        cohort.append(j)
    pairs = [(r[3], r[2]) for r in rows]
    _, p, _ = batch_scores(pair, pairs + [(b, a) for a, b in pairs])
    forward, reverse = np.split(p, 2)
    logs, probs, _ = batch_scores(
        passage, [(r[3], rows[j][4]) for r, j in zip(rows, cohort)]
    )
    ref, _, _ = batch_scores(
        passage, [(r[2], rows[j][4]) for r, j in zip(rows, cohort)]
    )
    results, counts = [], Counter()
    for i, (r, j) in enumerate(zip(rows, cohort, strict=True)):
        seed, q, wrong = r[2], r[3], rows[j][4]
        a, b = question_type(seed), question_type(q)
        trace = dict(
            cosine=float(incoming[i] @ vectors[i]),
            guard=requirement_guard(seed, q) or answer_satisfies_requirements(q, wrong),
            extra_guard="answer_type_changed"
            if a and b and a != b
            else semantic_answer_reason(seed, q, wrong, keyword_signature(q)),
            answer_words=len(wrong.split()),
            pair_forward=float(forward[i]),
            pair_reverse=float(reverse[i]),
            passage_logit=float(logs[i]),
            passage=float(probs[i]),
            source_passage_logit=float(ref[i]),
        )
        decisions = {}
        for arm in [*ARMS, "combined"]:
            config = selection if arm == "combined" else [arm]
            decisions[arm] = (
                reject(trace, "relative5" in config, "typed5" in config) or "accepted"
            )
        for arm, mode in [("live_exact", "exact"), ("live_optional", "guarded")]:
            with (
                TemporaryDirectory() as tmp,
                fixture_app(
                    Path(tmp),
                    Path(tmp) / "journal.db",
                    models,
                    mode,
                    {seed: wrong, q: r[4]},
                ) as ask,
            ):
                ask(seed)
                reply = ask(q)
            decisions[arm] = (
                "accepted" if reply["cache_tier"] else reply["cache_reason"]
            )
            if reply["cache_tier"]:
                assert reply["answer"] == wrong
        counts.update(arm for arm, reason in decisions.items() if reason == "accepted")
        results.append(
            dict(
                probe=r[0], wrong_answer_source=rows[j][0], decisions=decisions, **trace
            )
        )
    return {
        "wrong_reuse": {arm: score_record(counts[arm], len(rows)) for arm in decisions},
        "records": results,
    }


def spent(raw, models):
    seeds, probes = prepare(raw)
    encoder, pair, passage = models
    vs = embed(encoder, [r[0] for r in seeds] + [r[0] for r in probes])
    sims = vs[len(seeds) :] @ vs[: len(seeds)].T
    items = []
    for i, (q, labels, _) in enumerate(probes):
        j = int(np.argmax(sims[i]))
        s, stored = seeds[j]
        if (
            labels != stored
            or sims[i, j] < 0.84
            or requirement_guard(s, q)
            or answer_satisfies_requirements(q, answer_text(stored))
        ):
            continue
        items.append((i, j, q, s, answer_text(stored)))
    paired = [(q, s) for _, _, q, s, _ in items]
    _, both, _ = batch_scores(pair, paired + [(b, a) for a, b in paired])
    low = np.minimum(*np.split(both, 2))
    _, p, _ = batch_scores(passage, [(q, answer) for _, _, q, _, answer in items])
    records = [
        dict(
            probe_index=i,
            gallery_index=j,
            cosine=float(sims[i, j]),
            pair_min=float(v),
            passage=float(score),
            answer_words=len(answer.split()),
        )
        for (i, j, _, _, answer), v, score in zip(items, low, p, strict=True)
        if v >= 0.80 and score < 0.85
    ]
    if len(records) != 168:
        raise ValueError(f"Historical passage cohort changed: {len(records)}")
    return {
        "matched_rejections": len(records),
        "passage_quartiles": np.percentile(
            [r["passage"] for r in records], [25, 50, 75]
        ).tolist(),
        "at_most_three_answer_words": sum(r["answer_words"] <= 3 for r in records),
        "records": records,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--split", choices=["development", "holdout", "spent"], required=True
    )
    args = ap.parse_args()
    pinned(args.directory, False)
    models = load_chat_models()
    if args.split == "spent":
        out = spent((args.directory / "comqa_test.json").read_bytes(), models)
    else:
        selection = json.loads((args.output / "selection.json").read_bytes())[
            "combined"
        ]
        out = {
            name: corrupted(
                [r for r in getter(args.directory) if r[1] == args.split][: CAP[name]],
                models,
                selection,
            )
            for name, getter in [("comqa", comqa), ("python", python_docs)]
        }
    dump(args.output / (args.split + "-answer-audit.json"), out)
