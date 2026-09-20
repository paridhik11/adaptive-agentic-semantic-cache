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
    max_calls = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("OPENROUTER_MAX_DAILY_CALLS", "50"))

    print("=" * 80)
    print("PHASE 4: LLM JUDGE FEEDBACK LABELING (OPENROUTER)")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {target_model}")
    print(f"Inter-call delay: {delay}s")
    print(f"Daily call cap: {max_calls}")
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

    # Check for existing telemetry to support resuming and merging
    completed_records: Dict[str, Dict[str, Any]] = {}
    all_records_by_id: Dict[str, Dict[str, Any]] = {}
    if OUTPUT_TELEMETRY_PATH.exists():
        try:
            with open(OUTPUT_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    for r in saved:
                        if "pair_id" in r:
                            all_records_by_id[r["pair_id"]] = r
                            if not r.get("fallback_triggered"):
                                completed_records[r["pair_id"]] = r
            print(f"Loaded {len(completed_records)} previously completed genuine evaluations from telemetry.")
            print(f"Total existing records in telemetry: {len(all_records_by_id)}.")
        except Exception as e:
            print(f"Could not load existing telemetry ({e}), starting fresh.")

    def save_merged_telemetry() -> None:
        """Write all 108 records preserving dataset ordering and merging genuine updates."""
        merged_list = [all_records_by_id[p["id"]] for p in pairs if p["id"] in all_records_by_id]
        with open(OUTPUT_TELEMETRY_PATH, "w", encoding="utf-8") as tf:
            json.dump(merged_list, tf, indent=2)

    calls_attempted_this_session = 0
    genuine_labeled_this_session = 0
    quota_exhausted = False
    stop_reason = ""
    t_start = time.perf_counter()

    for idx, (pair, sim) in enumerate(zip(pairs, pair_sims), start=1):
        pid = pair["id"]
        qa = pair["query_a"]
        qb = pair["query_b"]
        dom = pair["domain"]
        authored_safe = pair["is_reuse_safe"]

        if pid in completed_records:
            print(f"[{idx:>3}/{len(pairs)}] {pid} ({dom:<22}): CACHED -> decision={completed_records[pid]['judge_decision']}")
            continue

        if calls_attempted_this_session >= max_calls:
            quota_exhausted = True
            stop_reason = f"Daily session cap reached ({max_calls} calls)."
            print(f"\n[STOP] {stop_reason}")
            break

        print(f"[{idx:>3}/{len(pairs)}] {pid} ({dom:<22}): Calling OpenRouter (sim={sim:.4f})...", end="", flush=True)
        res = judge.judge(
            query_a=qa,
            query_b=qb,
            domain=dom,
            similarity_score=sim,
            stability_confidence=1.0,
            category_history_rate=0.5,
        )
        calls_attempted_this_session += 1

        if res.fallback_triggered:
            print(f" -> FALLBACK ({res.error})")
            # Check if 429 or quota limit hit
            err_str = str(res.error).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                quota_exhausted = True
                stop_reason = f"OpenRouter quota / 429 limit encountered on {pid}: {res.error}"
                print(f"\n[STOP] {stop_reason}")
                break
            else:
                # Non-quota error, record fallback and continue
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
                    "fallback_triggered": True,
                    "error": res.error,
                    "raw_response": res.raw_response,
                    "retries_used": res.retries_used,
                    "request_id": res.request_id,
                }
                all_records_by_id[pid] = record
                save_merged_telemetry()
                continue

        # Genuine call with non-null response and request_id
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
            "fallback_triggered": False,
            "error": None,
            "raw_response": res.raw_response,
            "retries_used": res.retries_used,
            "request_id": res.request_id,
        }
        all_records_by_id[pid] = record
        completed_records[pid] = record
        genuine_labeled_this_session += 1
        print(f" -> {res.decision:<6} (safe={res.is_safe}, conf={res.confidence:.2f}, {res.latency_ms:.0f}ms, req_id={res.request_id})")

        # Save merged telemetry incrementally after each successful pair
        save_merged_telemetry()

    total_duration = time.perf_counter() - t_start
    save_merged_telemetry()

    # Final calculations for session report
    all_records = [all_records_by_id[p["id"]] for p in pairs if p["id"] in all_records_by_id]
    total_genuine = sum(1 for r in all_records if not r.get("fallback_triggered", False))
    total_pending = sum(1 for r in all_records if r.get("fallback_triggered", False))

    print("\n" + "=" * 80)
    print("SESSION SUMMARY REPORT")
    print("=" * 80)
    print(f"Session execution time: {total_duration:.1f}s ({total_duration/60:.2f} min)")
    print(f"Calls attempted this session: {calls_attempted_this_session}")
    print(f"New genuine labels obtained this session: {genuine_labeled_this_session}")
    print(f"Total genuine labels in telemetry: {total_genuine} / {len(pairs)}")
    print(f"Total fallback / pending records remaining: {total_pending} / {len(pairs)}")
    if stop_reason:
        print(f"Session termination note: {stop_reason}")

    # Per-domain breakdown
    by_dom: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        d = r["domain"]
        by_dom.setdefault(d, []).append(r)

    print("\nPer-Domain Telemetry Breakdown (Genuine vs. Pending):")
    print(f"{'Domain':<26} | {'Total':>5} | {'Genuine':>7} | {'Pending':>7} | {'Status':<15}")
    print("-" * 70)
    for dom in sorted(by_dom.keys()):
        d_recs = by_dom[dom]
        d_tot = len(d_recs)
        d_gen = sum(1 for r in d_recs if not r.get("fallback_triggered", False))
        d_pend = d_tot - d_gen
        status = "COMPLETE" if d_pend == 0 else f"{d_pend} pending"
        print(f"{dom:<26} | {d_tot:>5} | {d_gen:>7} | {d_pend:>7} | {status:<15}")

    print(f"\nTelemetry saved to {OUTPUT_TELEMETRY_PATH}")


if __name__ == "__main__":
    main()
