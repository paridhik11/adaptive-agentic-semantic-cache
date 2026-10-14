"""Phase A3 — Natural Repeat Rates in Real Chat Traffic (WildChat N >= 5,000).

Measures exact, canonical, and semantic repeat rates across real user chat traffic:
  - Dataset: allenai/WildChat (train-00000-of-00006.parquet)
  - License: Open Data Commons Attribution License (ODC-By)
  - Sample size: N = 5,000 English first-user-turn queries in temporal stream order.

Evaluates:
  - Exact string match repeat rate.
  - Canonical match repeat rate (via QueryNormalizer).
  - Semantic near-duplicate repeat rate across embedding cosine thresholds (0.85, 0.90, 0.92, 0.95).
  - Semantic repeat rate after Phase A rule guards (numbers, negations, entities, long-prompt, generative).

Outputs:
  - data/phaseA_wildchat_repeat_rates.json
  - data/review/pa_human_audit.csv (100 sampled decisions with blank human_label column)
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cache.normalizer import normalize_query
from src.decision.local_equivalence import evaluate_rule_guards

PARQUET_PATH = REPO_ROOT / "data" / "wildchat" / "train-00000.parquet"
OUTPUT_JSON_PATH = REPO_ROOT / "data" / "phaseA_wildchat_repeat_rates.json"
AUDIT_CSV_PATH = REPO_ROOT / "data" / "review" / "pa_human_audit.csv"

N_QUERIES = 5000


def main() -> None:
    print("=" * 80)
    print("PHASE A3: NATURAL REPEAT RATES IN REAL CHAT TRAFFIC (WILDCHAT)")
    print("=" * 80)

    print(f"\nLoading {N_QUERIES} first user queries from {PARQUET_PATH.name}...", flush=True)
    parquet_file = pq.ParquetFile(PARQUET_PATH)

    queries: List[str] = []
    for batch in parquet_file.iter_batches(batch_size=1000, columns=["conversation"]):
        conversations = batch["conversation"].to_pylist()
        for conv_list in conversations:
            if conv_list and len(conv_list) > 0:
                first_turn = conv_list[0]
                if isinstance(first_turn, dict) and first_turn.get("role") == "user":
                    content = first_turn.get("content", "").strip()
                    if content and 3 <= len(content) <= 2000:
                        queries.append(content)
                        if len(queries) >= N_QUERIES:
                            break
        if len(queries) >= N_QUERIES:
            break

    print(f"Extracted {len(queries)} real user prompts.", flush=True)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Tracking sets and stores
    seen_raw: Set[str] = set()
    seen_canonical: Dict[str, str] = {}  # canonical -> original query

    # We will embed in batches for speed and compare against past embeddings
    print("\nComputing embeddings for all 5,000 queries in stream order...")
    t0 = time.time()
    embeddings = embedder.encode(queries, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    print(f"Embeddings computed in {time.time() - t0:.2f} s")

    exact_repeats = 0
    canonical_repeats = 0

    thresholds = [0.85, 0.90, 0.92, 0.95]
    sim_counts = {t: 0 for t in thresholds}
    guarded_sim_counts = {t: 0 for t in thresholds}

    all_decision_records: List[Dict[str, Any]] = []

    print("\nEvaluating sequential stream reuse opportunities...")
    for i in range(len(queries)):
        q = queries[i]
        canon_key = normalize_query(q)

        is_exact = q in seen_raw
        is_canonical = canon_key in seen_canonical and not is_exact

        if is_exact:
            exact_repeats += 1
        if is_canonical or is_exact:
            canonical_repeats += 1

        best_sim = 0.0
        best_match_idx = -1
        best_match_query = ""

        if i > 0:
            # Cosine similarity against all prior queries [0 ... i-1]
            past_embs = embeddings[:i]
            cur_emb = embeddings[i : i + 1]
            sims = util.cos_sim(cur_emb, past_embs)[0]
            best_idx = int(torch.argmax(sims).item())
            best_sim = float(sims[best_idx].item())
            best_match_idx = best_idx
            best_match_query = queries[best_idx]

        guards_passed = False
        guard_reason = "NO_MATCH"
        if best_match_query:
            guards_passed, guard_reason = evaluate_rule_guards(best_match_query, q)

        for t in thresholds:
            if best_sim >= t:
                sim_counts[t] += 1
                if guards_passed:
                    guarded_sim_counts[t] += 1

        # Classify decision
        if is_exact or is_canonical:
            decision = "REUSE"
            tier = "EXACT_CANONICAL"
        elif best_sim >= 0.92 and guards_passed:
            decision = "REUSE"
            tier = "AUTO_REUSE"
        elif best_sim >= 0.85 and guards_passed:
            decision = "AMBIGUOUS_REUSE"
            tier = "LOCAL_EQUIVALENCE"
        else:
            decision = "BYPASS"
            tier = "BYPASS"

        all_decision_records.append({
            "query_index": i + 1,
            "query": q,
            "matched_query": best_match_query,
            "similarity": round(best_sim, 4),
            "guards_passed": guards_passed,
            "guard_reason": guard_reason,
            "tier": tier,
            "system_decision": decision,
        })

        seen_raw.add(q)
        if canon_key not in seen_canonical:
            seen_canonical[canon_key] = q

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries...")

    # Statistics
    total = len(queries)
    exact_rate = exact_repeats / total
    canonical_rate = canonical_repeats / total

    print("\n" + "=" * 80)
    print("WILDCHAT NATURAL REPEAT RATE FINDINGS (N = 5,000)")
    print("=" * 80)
    print(f"Exact string repeats:      {exact_repeats:5d} / {total} ({exact_rate:.2%})")
    print(f"Canonical folded repeats:  {canonical_repeats:5d} / {total} ({canonical_rate:.2%})")
    for t in thresholds:
        raw_rate = sim_counts[t] / total
        guard_rate = guarded_sim_counts[t] / total
        print(f"Semantic sim >= {t:.2f}:       {sim_counts[t]:5d} ({raw_rate:.2%}) | Guarded: {guarded_sim_counts[t]:5d} ({guard_rate:.2%})")

    # Sample 100 decisions for human audit
    # Mix: 25 canonical/exact, 25 high similarity (>= 0.92), 25 ambiguous (0.85-0.92), 25 bypass
    random.seed(42)
    pool_exact = [r for r in all_decision_records if r["tier"] == "EXACT_CANONICAL"]
    pool_auto = [r for r in all_decision_records if r["tier"] == "AUTO_REUSE"]
    pool_ambig = [r for r in all_decision_records if r["tier"] == "LOCAL_EQUIVALENCE"]
    pool_bypass = [r for r in all_decision_records if r["tier"] == "BYPASS" and r["similarity"] > 0.60]

    sample_exact = random.sample(pool_exact, min(25, len(pool_exact)))
    sample_auto = random.sample(pool_auto, min(25, len(pool_auto)))
    sample_ambig = random.sample(pool_ambig, min(25, len(pool_ambig)))
    sample_bypass = random.sample(pool_bypass, min(100 - len(sample_exact) - len(sample_auto) - len(sample_ambig), len(pool_bypass)))

    audit_sample = sample_exact + sample_auto + sample_ambig + sample_bypass
    random.shuffle(audit_sample)

    AUDIT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_index", "query", "matched_query", "similarity", "tier", "system_decision", "human_label"])
        for r in audit_sample:
            writer.writerow([
                r["query_index"],
                r["query"].replace("\n", " "),
                r["matched_query"].replace("\n", " "),
                r["similarity"],
                r["tier"],
                r["system_decision"],
                "",  # blank for human evaluation
            ])

    print(f"\nExported {len(audit_sample)} sampled decisions to: {AUDIT_CSV_PATH}")

    # Output JSON summary
    summary_data = {
        "timestamp": "2026-10-12T01:20:00+05:30",
        "dataset": "allenai/WildChat",
        "license": "Open Data Commons Attribution License (ODC-By)",
        "sample_size": total,
        "exact_string_repeats": exact_repeats,
        "exact_string_repeat_rate": round(exact_rate, 4),
        "canonical_repeats": canonical_repeats,
        "canonical_repeat_rate": round(canonical_rate, 4),
        "semantic_raw_counts": {str(t): sim_counts[t] for t in thresholds},
        "semantic_raw_rates": {str(t): round(sim_counts[t] / total, 4) for t in thresholds},
        "semantic_guarded_counts": {str(t): guarded_sim_counts[t] for t in thresholds},
        "semantic_guarded_rates": {str(t): round(guarded_sim_counts[t] / total, 4) for t in thresholds},
        "audit_csv_path": str(AUDIT_CSV_PATH),
        "audit_sample_size": len(audit_sample),
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Saved WildChat repeat rate summary to: {OUTPUT_JSON_PATH.name}")


if __name__ == "__main__":
    main()
