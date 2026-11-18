"""Phase 1 diagnostic: score all 30 exposed questions against their 3 source seeds.

Produces concrete score distributions for MiniLM cosine, Quora logit/pair_score,
and a simple combined policy. No threshold tuning, no serving activation.
Compares against human (Y/N/?) and model-assisted labels to characterize failures.

Evidence type: model_measurement_on_exposed_development_packet
Human labels: declared_human_review (not independently verified)
"""
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def block_network():
    def forbidden(*args, **kwargs):
        raise RuntimeError("network connection forbidden during local diagnostic")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/semantic-dev/phase1-score-diagnostic.json")
    parser.add_argument("--dense-threshold", type=float, default=0.85)
    parser.add_argument("--pair-threshold", type=float, default=0.5)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists, choose a fresh path: {args.output}")

    sources = json.loads((ROOT / "data/document_product_review/v1/sources.json").read_text())
    template = [json.loads(r) for r in
                (ROOT / "data/document_product_review/human_v1/human-responses.template.jsonl")
                .read_text().splitlines() if r.strip()]
    model_responses = {
        r["item_id"]: r for r in
        (json.loads(row) for row in
         (ROOT / "data/document_product_review/model_assisted_v1/responses.jsonl")
         .read_text().splitlines() if row.strip())
    }
    labels_raw = json.loads(
        (ROOT / "data/document_product_review/human_return_v1/labels-from-chat.json").read_text())
    h_to_item = {row["human_id"]: row["item_id"] for row in template}
    human_labels = {h_to_item[hid]: v["judgment"] for hid, v in labels_raw.items()}

    block_network()

    import torch
    torch.set_num_threads(2)

    from src.policy.encoders import DenseEncoder
    from src.policy.local_models import (ENCODER_MODEL, ENCODER_REVISION,
                                          PAIR_MODEL, PAIR_REVISION, LocalPairScorer)

    t0 = time.perf_counter()
    print("Loading encoder and pair scorer...", flush=True)
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    pair_scorer = LocalPairScorer(PAIR_MODEL, PAIR_REVISION)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  loaded in {load_ms:.0f} ms", flush=True)

    # Embed all source seeds
    seed_vectors = {s["seed_id"]: encoder.encode(s["original_question"]) for s in sources}
    seed_by_id = {s["seed_id"]: s for s in sources}

    rows = []
    for row in template:
        item_id = row["item_id"]
        human_q = row["new_question"]
        seed_id = row["seed_id"]
        human_label = human_labels.get(item_id)
        model_row = model_responses.get(item_id, {})
        model_label = model_row.get("answer_sufficient")

        t1 = time.perf_counter()
        q_vector = encoder.encode(human_q)
        embed_ms = (time.perf_counter() - t1) * 1000

        # Score against ALL seeds (not just the designated one)
        candidate_scores = []
        for s in sources:
            sv = seed_vectors[s["seed_id"]]
            cosine = sum(a * b for a, b in zip(q_vector, sv))
            t2 = time.perf_counter()
            try:
                pair = pair_scorer.score(human_q, s["original_question"])
                pair_ms = (time.perf_counter() - t2) * 1000
            except ValueError as exc:
                pair = {"logit": None, "pair_score": None, "input_tokens": None, "error": str(exc)}
                pair_ms = (time.perf_counter() - t2) * 1000
            candidate_scores.append({
                "seed_id": s["seed_id"],
                "dense_cosine": cosine,
                "pair_ms": pair_ms,
                **pair,
            })

        # Rank by dense cosine
        dense_ranked = sorted(candidate_scores, key=lambda c: -c["dense_cosine"])
        # Rank by pair_score (where available)
        pair_valid = [c for c in candidate_scores if c.get("pair_score") is not None]
        pair_ranked = sorted(pair_valid, key=lambda c: -c["pair_score"]) if pair_valid else []

        designated = next(c for c in candidate_scores if c["seed_id"] == seed_id)
        dense_top = dense_ranked[0]
        pair_top = pair_ranked[0] if pair_ranked else None

        # Policy decisions
        dense_accept = designated["dense_cosine"] >= args.dense_threshold
        pair_accept = (designated.get("pair_score") is not None and
                       designated["pair_score"] >= args.pair_threshold)
        # Combined: pair-ranked top has sufficient dense score AND pair score AND margin
        combined_accept = False
        combined_reason = "no_candidates"
        if pair_ranked:
            best = pair_ranked[0]
            second_score = pair_ranked[1]["pair_score"] if len(pair_ranked) > 1 else 0.0
            margin_ok = best["pair_score"] - second_score >= args.pair_margin
            combined_accept = (best["dense_cosine"] >= args.dense_threshold and
                               best["pair_score"] >= args.pair_threshold and
                               margin_ok)
            combined_reason = "accepted" if combined_accept else (
                "margin_fail" if not margin_ok else
                "dense_fail" if best["dense_cosine"] < args.dense_threshold else
                "pair_fail"
            )

        rows.append({
            "item_id": item_id,
            "human_id": row["human_id"],
            "seed_id": seed_id,
            "question": human_q,
            "human_label": human_label,
            "model_label": model_label,
            "embed_ms": embed_ms,
            "designated_scores": designated,
            "dense_top": dense_top,
            "pair_top": pair_top,
            "all_scores": candidate_scores,
            "policy": {
                "dense_only_accept": dense_accept,
                "pair_only_accept": pair_accept,
                "combined_accept": combined_accept,
                "combined_reason": combined_reason,
            },
        })
        status = f"human={human_label or '?'} dense={designated['dense_cosine']:.3f}"
        if designated.get("pair_score") is not None:
            status += f" pair={designated['pair_score']:.3f}(logit={designated['logit']:.2f})"
        print(f"  {row['human_id']} {combined_reason:12s} {status}", flush=True)

    # Compute summary distributions
    def policy_summary(key, label_key="human_label"):
        accepts = [r for r in rows if r["policy"][key] and r[label_key] in ("Y", "N")]
        tp = sum(1 for r in accepts if r[label_key] == "Y")
        fp = sum(1 for r in accepts if r[label_key] == "N")
        negatives = [r for r in rows if r[label_key] == "N"]
        positives = [r for r in rows if r[label_key] == "Y"]
        coverage = tp / len(positives) if positives else None
        irr = fp / len(accepts) if accepts else None
        return {"accepted": len(accepts), "tp": tp, "fp": fp,
                "irr_cache": irr, "coverage": coverage,
                "abstained_uncertain": sum(1 for r in rows if r["policy"][key] and r[label_key] == "?")}

    summary = {
        "dense_only": policy_summary("dense_only_accept"),
        "pair_only": policy_summary("pair_only_accept"),
        "combined": policy_summary("combined_accept"),
    }

    # Score distributions for designated pairs
    dense_scores = [r["designated_scores"]["dense_cosine"] for r in rows]
    pair_scores = [r["designated_scores"]["pair_score"] for r in rows
                   if r["designated_scores"].get("pair_score") is not None]

    result = {
        "schema": "phase1-score-diagnostic-v1",
        "evidence_type": "model_measurement_on_exposed_development_packet",
        "human_label_provenance": "declared_human_review_not_independently_verified",
        "model_label_provenance": "model_assisted_self_review",
        "n_questions": len(rows),
        "n_sources": len(sources),
        "models": {
            "encoder": ENCODER_MODEL + "@" + ENCODER_REVISION,
            "pair": PAIR_MODEL + "@" + PAIR_REVISION,
        },
        "config": {
            "dense_threshold": args.dense_threshold,
            "pair_threshold": args.pair_threshold,
            "pair_margin": args.pair_margin,
        },
        "load_ms": load_ms,
        "score_distributions": {
            "dense_cosine": {
                "min": min(dense_scores),
                "max": max(dense_scores),
                "mean": sum(dense_scores) / len(dense_scores),
                "human_Y": [r["designated_scores"]["dense_cosine"] for r in rows if r["human_label"] == "Y"],
                "human_N": [r["designated_scores"]["dense_cosine"] for r in rows if r["human_label"] == "N"],
                "human_uncertain": [r["designated_scores"]["dense_cosine"] for r in rows if r["human_label"] == "?"],
            },
            "pair_score": {
                "min": min(pair_scores) if pair_scores else None,
                "max": max(pair_scores) if pair_scores else None,
                "mean": sum(pair_scores) / len(pair_scores) if pair_scores else None,
                "human_Y": [r["designated_scores"]["pair_score"] for r in rows
                            if r["human_label"] == "Y" and r["designated_scores"].get("pair_score") is not None],
                "human_N": [r["designated_scores"]["pair_score"] for r in rows
                            if r["human_label"] == "N" and r["designated_scores"].get("pair_score") is not None],
            },
        },
        "policy_summary": summary,
        "rows": rows,
        "limitations": [
            "30 correlated exposed model-authored questions over 3 FAQ sources; no fresh final evaluation.",
            "Pair scores are question-similarity, not answer-applicability.",
            "MiniLM was trained on MS MARCO, not FAQ answer reuse.",
            "Quora CrossEncoder trained on Quora Question Pairs (QQP), measures question duplicate intent.",
            "No serving activation. Human label provenance declared, not independently verified.",
            "This packet is spent/exposed and must not be reused as a final test.",
        ],
        "serving_enabled": False,
        "external_generation_calls": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "policy_summary": summary}))


if __name__ == "__main__":
    main()
