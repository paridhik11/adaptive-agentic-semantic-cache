"""Phase 4 — LLM Judge Feedback Labeling on Synthetic Query Pairs.

Uses OpenRouter (nvidia/nemotron-3-super-120b-a12b:free) via LLMJudge
to label all 108 synthetic query pairs with a simulated production feedback signal.

Integrity rules:
- Blind labeling: all labels are assigned BEFORE any threshold sweep.
- Paced execution: >= 2.5s delay to strictly comply with OpenRouter's 20 RPM limit.
- Resumable: saves progress incrementally to data/openrouter_synthetic_feedback_telemetry.json.
- Full telemetry: records latency, tokens, decision, confidence, and reasoning.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.cache.embedding import QueryEmbedder
from src.decision.judge_call import DEFAULT_JUDGE_MODEL, LLMJudge

DATASET_PATH = REPO_ROOT / "data" / "raw" / "synthetic_query_pair_feedback.json"
OUTPUT_TELEMETRY_PATH = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"


def main() -> None:
    target_model = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("OPENROUTER_MODEL") or DEFAULT_JUDGE_MODEL)
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

    print("=" * 80)
    print("PHASE 4: LLM JUDGE FEEDBACK LABELING (OPENROUTER)")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {target_model}")
    print(f"Inter-call delay: {delay}s")
    print(f"Output: {OUTPUT_TELEMETRY_PATH}")
    print("=" * 80)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Pre-embed pairs to obtain exact cosine similarity scores
    print("\nComputing embeddings with all-MiniLM-L6-v2...")
    embedder = QueryEmbedder()
    pair_sims: List[float] = []
    for p in pairs:
        emb_a = embedder.encode(p["query_a"])
        emb_b = embedder.encode(p["query_b"])
        sim = float(np.dot(emb_a, emb_b))
        pair_sims.append(sim)
    print(f"Computed similarity scores for {len(pairs)} pairs.")

    # Initialize judge
    judge = LLMJudge(
        model=target_model,
        backend="openrouter",
        rate_limit_delay_seconds=delay,
    )

    if not judge.api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    # Check for existing telemetry to support resuming
    completed_records: Dict[str, Dict[str, Any]] = {}
    if OUTPUT_TELEMETRY_PATH.exists():
        try:
            with open(OUTPUT_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    for r in saved:
                        if "pair_id" in r and not r.get("fallback_triggered"):
                            completed_records[r["pair_id"]] = r
            print(f"Loaded {len(completed_records)} previously completed evaluations from telemetry.")
        except Exception as e:
            print(f"Could not load existing telemetry ({e}), starting fresh.")

    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, (pair, sim) in enumerate(zip(pairs, pair_sims), start=1):
        pid = pair["id"]
        qa = pair["query_a"]
        qb = pair["query_b"]
        dom = pair["domain"]
        authored_safe = pair["is_reuse_safe"]

        if pid in completed_records:
            print(f"[{idx:>3}/{len(pairs)}] {pid} ({dom:<20}): CACHED -> decision={completed_records[pid]['judge_decision']}")
            results.append(completed_records[pid])
            continue

        print(f"[{idx:>3}/{len(pairs)}] {pid} ({dom:<20}): Calling OpenRouter (sim={sim:.4f})...", end="", flush=True)
        res = judge.judge(
            query_a=qa,
            query_b=qb,
            domain=dom,
            similarity_score=sim,
            stability_confidence=1.0,
            category_history_rate=0.5,
        )

        record = {
            "pair_id": pid,
            "query_a": qa,
            "query_b": qb,
            "domain": dom,
            "similarity_score": round(sim, 4),
            "authored_is_reuse_safe": authored_safe,
            "judge_decision": res.decision,
            "judge_is_safe": res.is_safe,
            "confidence": round(res.confidence, 4),
            "rationale": res.rationale,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.total_tokens,
            "latency_ms": round(res.latency_ms, 2),
            "model": res.model,
            "fallback_triggered": res.fallback_triggered,
            "error": res.error,
            "raw_response": res.raw_response,
            "retries_used": res.retries_used,
            "request_id": res.request_id,
        }
        results.append(record)
        print(f" -> {res.decision:<6} (conf={res.confidence:.2f}, {res.latency_ms:.0f}ms)")

        # Save progress incrementally every 5 pairs
        if idx % 5 == 0 or idx == len(pairs):
            with open(OUTPUT_TELEMETRY_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    total_duration = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print("FEEDBACK LABELING COMPLETED")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.2f} min)")
    print("=" * 80)

    # Compute agreement summary
    total = len(results)
    agreements = sum(1 for r in results if r["authored_is_reuse_safe"] == r["judge_is_safe"])
    fallbacks = sum(1 for r in results if r.get("fallback_triggered", False))

    print(f"Total pairs evaluated: {total}")
    print(f"Author-Judge Agreement: {agreements}/{total} ({agreements/total*100:.2f}%)")
    print(f"Judge Fallbacks / Errors: {fallbacks}")

    # Domain breakdown
    from collections import Counter
    by_dom = {}
    for r in results:
        d = r["domain"]
        by_dom.setdefault(d, []).append(r)

    print("\nPer-Domain Judge Label Breakdown:")
    for dom in sorted(by_dom.keys()):
        d_recs = by_dom[dom]
        d_tot = len(d_recs)
        d_reuse = sum(1 for r in d_recs if r["judge_decision"] == "REUSE")
        d_bypass = d_tot - d_reuse
        d_agree = sum(1 for r in d_recs if r["authored_is_reuse_safe"] == r["judge_is_safe"])
        print(f"  - {dom:<24}: Total={d_tot:>2} | Judge REUSE={d_reuse:>2} | Judge BYPASS={d_bypass:>2} | Agreement={d_agree}/{d_tot} ({d_agree/d_tot*100:.1f}%)")

    # Final write to ensure all results saved
    with open(OUTPUT_TELEMETRY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full telemetry to {OUTPUT_TELEMETRY_PATH}")


if __name__ == "__main__":
    main()
