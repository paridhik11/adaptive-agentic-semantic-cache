"""Offline answer-level and pair-equivalence evaluation. See the frozen protocol."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer

from independent_semantic_data import (
    ROOT,
    INPUTS,
    comqa,
    python_docs,
    pair_data,
    pinned,
)
from evaluate_semantic_upgrade import interval

sys.path.insert(0, str(ROOT))
from src.api.chat_keywords import keyword_signature
from src.api.chat_reuse_checks import semantic_answer_reason
from src.api.chat_semantic import load_chat_models
from src.api.live_chat import DEFAULT_MODEL, create_live_chat_app
from src.policy.decomposed_applicability import (
    answer_satisfies_requirements,
    requirement_guard,
)

BASE = "top1"
ARMS = [BASE, "dense5", "hybrid5", "relative5", "typed5"]
CAP = {"comqa": 128, "python": 40}


def score_record(n, d):
    return {"n": n, "d": d, "rate": n / d if d else None, "ci95": interval(n, d)}


def dump(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def elapsed(t):
    return (perf_counter() - t) * 1000


def question_type(question):
    for name, pattern in [
        ("time", r"when|what year|what date"),
        ("person", r"who"),
        ("quantity", r"how many|how much"),
        ("place", r"where"),
    ]:
        if re.match(r"(?:" + pattern + r")\b", question.strip(), re.I):
            return name
    return None


def batch_scores(model, items):
    # Exactly the pinned wrapper's untruncated CPU logits and sigmoid, batched.
    from scipy.special import expit

    t = perf_counter()
    tokenizer = model.model.tokenizer
    a, b = zip(*items)
    tokens = tokenizer(list(a), list(b), truncation=False)["input_ids"]
    max_len = min(tokenizer.model_max_length, model.model.max_seq_length)
    if any(len(x) > max_len for x in tokens):
        raise ValueError("Cross-encoder window exceeded")
    logits = np.asarray(
        model.model.predict(items, batch_size=16, show_progress_bar=False), dtype=float
    )
    if not np.isfinite(logits).all():
        raise ValueError("Invalid model scores")
    return logits, expit(logits), elapsed(t)


def equivalence(directory, split, models):
    reports = {}
    encoder, pair, _ = models
    for name in ["paws", "qqp"]:
        rows = pair_data(directory, name, split)
        pairs = [(a, b) for _, a, b, _ in rows]
        texts = [q for p in pairs for q in p]
        vectors = embed(encoder, texts)
        cos = np.einsum("ij,ij->i", vectors[::2], vectors[1::2])
        _, scores, ms = batch_scores(pair, pairs + [(b, a) for a, b in pairs])
        p = np.minimum(*np.split(scores, 2))
        obs = [r[-1] for r in rows]
        policies = {
            "cosine": cos >= 0.84,
            "pair": p >= 0.8,
            "guarded_pair": [
                (s >= 0.84 and v >= 0.8 and not requirement_guard(a, b))
                for s, v, (a, b) in zip(cos, p, pairs, strict=True)
            ],
        }
        metrics = {}
        for arm, predicted in policies.items():
            tp, fp, tn, fn = [
                sum(
                    int(label == actual and bool(guess) == expected)
                    for label, guess in zip(obs, predicted, strict=True)
                )
                for actual, expected in [(1, True), (0, True), (0, False), (1, False)]
            ]
            metrics[arm] = dict(
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                precision=score_record(tp, tp + fp),
                recall=score_record(tp, tp + fn),
                specificity=score_record(tn, tn + fp),
            )
        reports[name] = {
            "ids": [r[0] for r in rows],
            "metrics": metrics,
            "bidirectional_batch_ms": ms,
        }
    return reports


def embed(encoder, texts):
    ids = encoder.model.tokenizer(texts, truncation=False)["input_ids"]
    if max(map(len, ids)) > encoder.model.max_seq_length:
        raise ValueError("Embedding window exceeded")
    return encoder.model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )


def features(rows, models):
    encoder, pair, passage = models
    t = perf_counter()
    seeds, probes = [r[2] for r in rows], [r[3] for r in rows]
    n = len(rows)
    vectors = embed(encoder, seeds + probes)
    cosines = vectors[n:] @ vectors[:n].T
    embed_ms = elapsed(t)
    plan, candidates = [], set()
    for shard in (0, 1):
        ids = list(range(shard, n, 2))
        tfidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        lex = tfidf.fit_transform([seeds[j] for j in ids])
        lexical = (tfidf.transform(probes) @ lex.T).toarray()
        for i in range(n):
            cosine = {j: float(cosines[i, j]) for j in ids}
            hybrid = {
                j: 0.75 * cosine[j] + 0.25 * float(lexical[i, k])
                for k, j in enumerate(ids)
            }
            dense_order = sorted(ids, key=lambda j: -cosine[j])
            hybrid_order = sorted(ids, key=lambda j: -hybrid[j])
            selected = set(dense_order[:5] + hybrid_order[:5])
            if i in ids:
                selected.add(i)
            candidates.update((i, j) for j in selected)
            plan.append((i, shard, dense_order, hybrid_order, cosine, hybrid))
    candidates = sorted(candidates)
    pairs = [(probes[i], seeds[j]) for i, j in candidates]
    _, probs, pair_ms = batch_scores(pair, pairs + [(b, a) for a, b in pairs])
    forward, reverse = np.split(probs, 2)
    logits, ans_probs, passage_ms = batch_scores(
        passage, [(probes[i], rows[j][4]) for i, j in candidates]
    )
    src_logits, _, source_ms = batch_scores(passage, [(r[2], r[4]) for r in rows])
    trace = {}
    for k, (i, j) in enumerate(candidates):
        q, s, answer = probes[i], seeds[j], rows[j][4]
        # Baseline uses the same guards as the published full-question study.
        guard = requirement_guard(s, q) or answer_satisfies_requirements(q, answer)
        mismatch = (
            question_type(s)
            and question_type(q)
            and question_type(s) != question_type(q)
        )
        extra = (
            "answer_type_changed"
            if mismatch
            else semantic_answer_reason(s, q, answer, keyword_signature(q))
        )
        trace[i, j] = dict(
            source=rows[j][0],
            matches_annotation=i == j,
            guard=guard,
            extra_guard=extra,
            cosine=float(cosines[i, j]),
            pair_forward=float(forward[k]),
            pair_reverse=float(reverse[k]),
            passage=float(ans_probs[k]),
            passage_logit=float(logits[k]),
            source_passage_logit=float(src_logits[j]),
            answer_words=len(answer.split()),
        )
    timing = {
        "embedding_batch_ms": embed_ms,
        "bidirectional_batch_ms": pair_ms,
        "passage_batch_ms": passage_ms,
        "source_passage_batch_ms": source_ms,
        "unique_scored_pairs": len(candidates),
    }
    return plan, trace, timing


def reject(c, relative, typed):
    if c["cosine"] < 0.84:
        return "cosine"
    if c["guard"]:
        return c["guard"]
    if typed and c["extra_guard"]:
        return c["extra_guard"]
    pair = min(c["pair_forward"], c["pair_reverse"])
    if pair < 0.80:
        return "question_pair"
    if c["passage"] >= 0.85:
        return None
    if (
        relative
        and c["answer_words"] <= 12
        and pair >= 0.95
        and c["passage"] >= 0.01
        and c["passage_logit"] >= c["source_passage_logit"] - 0.5
    ):
        return None
    return "passage"


def decision(row, trace, config):
    i, _, dense, hybrid, cosines, blended = row
    order, scores = (hybrid, blended) if "hybrid5" in config else (dense, cosines)
    k = 1 if config == [BASE] else 5
    outcomes = {
        j: reject(trace[i, j], "relative5" in config, "typed5" in config)
        for j in order[:k]
    }
    eligible = [j for j in outcomes if outcomes[j] is None]
    if eligible:
        first = eligible[0]
        rivals = eligible[1:2] + order[k : k + 1] if k > 1 else []
        if any(scores[first] - scores[j] < 0.035 for j in rivals):
            return None, "ambiguous"
        return first, "accepted"
    if i in order and i not in outcomes:
        return None, "outside_top_k"
    return None, outcomes.get(i, outcomes[order[0]])


@contextmanager
def fixture_app(directory, journal, models, mode, answers):
    def transport(body):
        q = body["messages"][-1]["content"]
        return {
            "model": DEFAULT_MODEL,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": answers[q]},
                }
            ],
        }

    app = create_live_chat_app(
        directory,
        journal_path=journal,
        api_key="offline-fixture",
        transport=transport,
        semantic_models=models,
        semantic_mode=mode,
    )
    with TestClient(app) as client:
        client.headers["x-demo-session"] = re.search(
            r'name="demo-session" content="([^"]+)"', client.get("/").text
        )[1]

        def ask(q):
            res = client.post("/api/chat", json={"question": q})
            if res.is_error:
                raise RuntimeError(
                    f"Fixture request failed: {res.status_code} {res.json()}"
                )
            return res.json()

        yield ask


def runtime(rows, models, mode):
    # Snapshot each primed database through SQLite's backup API (WAL-aware).
    # Isolate every measured request, including exact controls, from probe writes.
    answers = {q: r[4] for r in rows for q in r[2:4]}
    events = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for shard in (0, 1):
            primed = root / str(shard)
            for seq, j in enumerate(range(shard, len(rows), 2)):
                # Each local-only journal has its normal limit; real journals are never opened.
                with fixture_app(
                    primed, root / f"seed-{shard}-{seq}.db", models, mode, answers
                ) as ask:
                    ask(rows[j][2])
            snapshot = root / f"{shard}.db"
            with (
                sqlite3.connect(primed / "chat.sqlite") as source,
                sqlite3.connect(snapshot) as dest,
            ):
                source.backup(dest)
            for i, r in enumerate(rows):
                kinds = ["present", "exact"] if i % 2 == shard else ["absent"]
                for kind in kinds:
                    with TemporaryDirectory(dir=root) as child:
                        shutil.copyfile(snapshot, Path(child) / "chat.sqlite")
                        with fixture_app(
                            child, Path(child) / "fixture.db", models, mode, answers
                        ) as ask:
                            response = ask(r[2] if kind == "exact" else r[3])
                    tier = response["cache_tier"]
                    if tier and response["answer"] not in {
                        s[4] for s in rows[shard::2]
                    }:
                        raise ValueError("Hit outside primed shard")
                    diag = response.get("semantic_verification", {})
                    events.append(
                        dict(
                            id=r[0],
                            kind=kind,
                            tier=tier,
                            correct=bool(tier and response["answer"] == r[4]),
                            reason=response.get("semantic_decision")
                            or response["cache_reason"],
                            verification_ms=diag.get("verification_ms", 0),
                            latency_ms=response["latency_ms"],
                            calls=response["provider_calls"],
                            state=response["cache_status"],
                            live_candidates=diag.get("candidates", []),
                            live_rejections=diag.get("rejections", {}),
                            verification_state=diag.get("state"),
                        )
                    )
    return events


def summarize(events):
    probes = [e for e in events if e["kind"] != "exact"]
    supported = [e for e in probes if e["kind"] == "present"]
    hits = [e for e in probes if e["tier"] == "SEMANTIC"]
    correct = sum(e["correct"] for e in hits)
    return dict(
        probes=len(probes),
        correct=correct,
        incorrect=len(hits) - correct,
        precision=score_record(correct, len(hits)),
        recall=score_record(correct, len(supported)),
        coverage=score_record(len(hits), len(probes)),
        fallback=score_record(len(probes) - len(hits), len(probes)),
        absent_error=score_record(
            sum(e["kind"] == "absent" for e in hits), len(probes) - len(supported)
        ),
        exact_hits=sum(e["tier"] == "EXACT" for e in events),
        potential_semantic_calls_avoided=len(hits),
        reasons_present=dict(
            Counter(e["reason"] for e in supported if not e["correct"])
        ),
        reasons_absent=dict(
            Counter(e["reason"] for e in probes if e["kind"] == "absent")
        ),
    )


def replay(rows, plan, trace, config):
    events = []
    for p in plan:
        i, shard, *_ = p
        j, reason = decision(p, trace, config)
        events.append(
            dict(
                id=rows[i][0],
                kind="present" if i % 2 == shard else "absent",
                tier="SEMANTIC" if j is not None else None,
                correct=i == j,
                reason=reason,
                source=rows[j][0] if j is not None else None,
            )
        )
    return events


def improves(panels, config):
    gained = 0
    for rows, plan, trace in panels.values():
        normal = replay(rows, plan, trace, [BASE])
        alt = replay(rows, plan, trace, config)
        # No newly incorrect decision, including within the same aggregate count.
        if any(
            y["tier"]
            and not y["correct"]
            and (not x["tier"] or y["source"] != x["source"])
            for x, y in zip(normal, alt, strict=True)
        ):
            return False
        delta = sum(e["correct"] for e in alt) - sum(e["correct"] for e in normal)
        if delta < 0:
            return False
        gained += delta
    return gained > 0


def runtime_summary(events):
    metrics = summarize(events)
    probes = [e for e in events if e["kind"] != "exact"]
    metrics.update(
        measured_fixture_calls=sum(e["calls"] for e in events),
        measured_probe_calls=sum(e["calls"] for e in probes),
        probe_latency_p50_p95_ms=np.percentile(
            [e["latency_ms"] for e in probes], [50, 95]
        ).tolist(),
        mean_probe_verification_ms=np.mean([e["verification_ms"] for e in probes]),
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--split", choices=["development", "holdout"], required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument(
        "--default-only",
        action="store_true",
        help="Replay the unmodified lightweight live policy",
    )
    args = ap.parse_args()
    pinned(args.directory, args.download)
    args.output.mkdir(parents=True, exist_ok=True)
    split = args.split
    lock = args.output / "selection.json"
    if split == "holdout" and not lock.is_file():
        ap.error("Generate development selection before holdout")
    selection = json.loads(lock.read_bytes()) if split == "holdout" else {}
    if args.default_only:
        result = {}
        for name, get_data in [("comqa", comqa), ("python", python_docs)]:
            rows = [r for r in get_data(args.directory) if r[1] == split][: CAP[name]]
            events = runtime(rows, None, "guarded")
            result[name] = {"metrics": runtime_summary(events), "events": events}
        dump(args.output / (split + "-live-default.json"), result)
        return
    models = load_chat_models()
    panels, details, result = (
        {},
        {},
        {"models": [m.identity for m in models], "inputs": INPUTS, "split": split},
    )
    result["equivalence"] = equivalence(args.directory, split, models)
    for name, get_data in [("comqa", comqa), ("python", python_docs)]:
        rows = [r for r in get_data(args.directory) if r[1] == split][: CAP[name]]
        plan, trace, times = features(rows, models)
        panels[name] = (rows, plan, trace)
        details[name] = {"candidates": [], "events": {}}
        for i, shard, dense, hybrid, _, blended in plan:
            tracked = set(dense[:5] + hybrid[:5] + ([i] if i in dense else []))
            for j in sorted(tracked):
                details[name]["candidates"].append(
                    dict(
                        probe=rows[i][0],
                        shard=shard,
                        dense_rank=dense.index(j) + 1,
                        hybrid_rank=hybrid.index(j) + 1,
                        hybrid_score=blended[j],
                        **trace[i, j],
                    )
                )
        result[name] = {"groups": len(rows), "scoring": times, "arms": {}}
    if split == "development":
        components = [arm for arm in ARMS[1:] if improves(panels, [arm])]
        combined = components if components and improves(panels, components) else [BASE]
        selection = {
            "individually_qualified": components,
            "combined": combined,
            "baseline": BASE,
        }
        dump(lock, selection)
    for name, (rows, plan, trace) in panels.items():
        for arm in ["live_exact", "live_optional", *ARMS, "combined"]:
            if arm.startswith("live"):
                mode = "exact" if arm == "live_exact" else "guarded"
                events = runtime(rows, models, mode)
                metrics = runtime_summary(events)
            else:
                config = selection["combined"] if arm == "combined" else [arm]
                t = perf_counter()
                events = replay(rows, plan, trace, config)
                scoring_ms = elapsed(t)
                metrics = summarize(events)
                # Offline retrieval arms only process nonexact probes. No simulated transport.
                metrics.update(
                    exact_hits=None,
                    measured_fixture_calls=None,
                    shared_score_decision_ms=scoring_ms,
                )
            details[name]["events"][arm] = events
            result[name]["arms"][arm] = metrics
    dump(args.output / (split + ".json"), result)
    import gzip

    with (args.output / (split + "-traces.json.gz")).open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as stream:
            stream.write(
                json.dumps(details, sort_keys=True, separators=(",", ":")).encode()
            )
    print(split, selection)


if __name__ == "__main__":
    main()
