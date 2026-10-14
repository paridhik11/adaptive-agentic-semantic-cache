"""Phase A2 — Domain-Level Calibration on CQADupStack Multi-Domain Data.

Uses CQADupStack (StackExchange community duplicate questions across subforums)
as real multi-domain data to evaluate whether per-category thresholds can fire
under the strict safety gates:
  - MINIMUM_CATEGORY_N = 20
  - MINIMUM_MINORITY_CLASS_N = 10
  - Global Fallback = 0.85

Dataset:
  - Source: mteb/cqadupstack-retrieval
  - License: CC BY-SA 3.0 / 4.0 (Stack Exchange user contribution license)
  - Subforums: android, english, gaming, gis, mathematica, physics,
               programmers, stats, tex, unix, webmasters, wordpress

Outputs:
  - data/phaseA_cqadupstack_calibration.json
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.decision.adaptive_threshold_engine import (
    AdaptiveThresholdEngine,
    CategoryThreshold,
    MINIMUM_CATEGORY_N,
    MINIMUM_MINORITY_CLASS_N,
    FALLBACK_THRESHOLD,
)

CACHE_DIR = REPO_ROOT / "data" / "cqadupstack"
OUTPUT_PATH = REPO_ROOT / "data" / "phaseA_cqadupstack_calibration.json"

SUBFORUMS = [
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
]


def download_subforum_data(subforum: str) -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{subforum}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://huggingface.co/datasets/mteb/cqadupstack-retrieval/resolve/main/data/{subforum}-00000-of-00001.json.gz"
    print(f"Downloading {subforum} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw_gz = resp.read()
    data = json.loads(gzip.decompress(raw_gz).decode("utf-8"))

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def build_labeled_pairs_for_subforum(
    subforum: str,
    data: Dict[str, Any],
    n_pos: int = 25,
    n_neg: int = 25,
    seed: int = 42,
) -> List[Tuple[str, str, bool]]:
    """Build positive and negative query pairs for validation tuning."""
    random.seed(seed + hash(subforum) % 1000)

    corpus_map = dict(zip(data["corpus"]["ids"], data["corpus"]["sentences"]))
    query_map = dict(zip(data["queries"]["ids"], data["queries"]["sentences"]))
    rel_map = dict(zip(data["relevant_docs"]["ids"], data["relevant_docs"]["relevant_docs"]))

    # Find valid positive pairs
    pos_pairs: List[Tuple[str, str, bool]] = []
    all_query_ids = list(query_map.keys())
    random.shuffle(all_query_ids)

    for qid in all_query_ids:
        if qid in rel_map and rel_map[qid]:
            doc_id = rel_map[qid][0]
            if doc_id in corpus_map:
                q_text = query_map[qid].strip()
                doc_text = corpus_map[doc_id].strip()
                if q_text and doc_text and q_text.lower() != doc_text.lower():
                    pos_pairs.append((q_text, doc_text, True))
                    if len(pos_pairs) >= n_pos:
                        break

    # Build negative pairs from distinct queries in the same subforum
    neg_pairs: List[Tuple[str, str, bool]] = []
    shuffled_corpus = list(corpus_map.items())
    random.shuffle(shuffled_corpus)

    for qid in all_query_ids:
        q_text = query_map[qid].strip()
        rel_docs = set(rel_map.get(qid, []))
        for doc_id, doc_text in shuffled_corpus:
            if doc_id not in rel_docs and doc_id != qid:
                doc_clean = doc_text.strip()
                if doc_clean and q_text:
                    neg_pairs.append((q_text, doc_clean, False))
                    break
        if len(neg_pairs) >= n_neg:
            break

    return pos_pairs + neg_pairs


def main() -> None:
    print("=" * 80)
    print("PHASE A2: DOMAIN-LEVEL CALIBRATION ON CQADUPSTACK")
    print("=" * 80)

    print("\nLoading Phase 2 Bi-Encoder model for embedding cosine similarities...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_labeled_samples: List[Tuple[str, float, bool]] = []
    subforum_pair_details: Dict[str, Any] = {}

    for subforum in SUBFORUMS:
        print(f"\nProcessing subforum: {subforum}...")
        raw_data = download_subforum_data(subforum)
        pairs = build_labeled_pairs_for_subforum(subforum, raw_data, n_pos=25, n_neg=25, seed=42)
        print(f"  Constructed {len(pairs)} validation pairs (balanced pos/neg)")

        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]

        embs_a = embedder.encode(texts_a, convert_to_tensor=True, show_progress_bar=False)
        embs_b = embedder.encode(texts_b, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(embs_a, embs_b).diagonal().cpu().tolist()

        n_pos = sum(1 for l in labels if l)
        n_neg = sum(1 for l in labels if not l)

        for sim, label in zip(sims, labels):
            all_labeled_samples.append((subforum, float(sim), bool(label)))

        subforum_pair_details[subforum] = {
            "n_pairs": len(pairs),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "mean_sim_positive": round(float(np.mean([s for s, l in zip(sims, labels) if l])), 4),
            "mean_sim_negative": round(float(np.mean([s for s, l in zip(sims, labels) if not l])), 4),
        }

    # Fit AdaptiveThresholdEngine
    print("\n" + "=" * 80)
    print("FITTING ADAPTIVE THRESHOLD ENGINE WITH DUAL GATES (N>=20, Minority>=10)...")
    print("=" * 80)

    engine = AdaptiveThresholdEngine.fit(
        labeled_data=all_labeled_samples,
        minimum_category_n=MINIMUM_CATEGORY_N,
        minimum_minority_class_n=MINIMUM_MINORITY_CLASS_N,
        fallback_threshold=FALLBACK_THRESHOLD,
    )

    domain_results: Dict[str, Any] = {}
    n_moved = 0
    n_fallback = 0

    print(f"\n{'Subforum':<15} | {'N':<4} | {'Minority':<8} | {'Status':<10} | {'OperThresh':<10} | {'CI 95%':<18} | {'Reason'}")
    print("-" * 95)

    for subforum in SUBFORUMS:
        ct: CategoryThreshold = engine.get_threshold(subforum)
        ci_str = f"[{ct.ci_lower:.4f}, {ct.ci_upper:.4f}]" if ct.ci_lower is not None else "N/A"
        status = "FALLBACK" if ct.is_fallback else "FITTED"
        if ct.is_fallback:
            n_fallback += 1
        else:
            n_moved += 1

        print(f"{subforum:<15} | {ct.sample_size:<4} | {ct.minority_count:<8} | {status:<10} | {ct.operating_threshold:<10.4f} | {ci_str:<18} | {ct.fallback_reason or 'Gates passed; fitted logistic'}")

        domain_results[subforum] = {
            "sample_size": ct.sample_size,
            "minority_count": ct.minority_count,
            "is_fallback": ct.is_fallback,
            "operating_threshold": round(float(ct.operating_threshold), 4),
            "point_estimate": round(float(ct.threshold), 4) if ct.threshold is not None else None,
            "ci_lower": round(float(ct.ci_lower), 4) if ct.ci_lower is not None else None,
            "ci_upper": round(float(ct.ci_upper), 4) if ct.ci_upper is not None else None,
            "fallback_reason": ct.fallback_reason,
            "pair_stats": subforum_pair_details[subforum],
        }

    summary = {
        "timestamp": "2026-10-12T01:15:00+05:30",
        "dataset": "mteb/cqadupstack-retrieval",
        "license": "CC BY-SA 3.0 / 4.0",
        "gates": {
            "minimum_category_n": MINIMUM_CATEGORY_N,
            "minimum_minority_class_n": MINIMUM_MINORITY_CLASS_N,
            "fallback_threshold": FALLBACK_THRESHOLD,
        },
        "total_subforums": len(SUBFORUMS),
        "subforums_fitted": n_moved,
        "subforums_fallback": n_fallback,
        "domain_thresholds": domain_results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved CQADupStack Calibration Results to: {OUTPUT_PATH.name}")
    print(f"Summary: {n_moved} subforums moved off fallback, {n_fallback} remained on fallback 0.85.")


if __name__ == "__main__":
    main()
