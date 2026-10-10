"""Phase 8 — Dataset Preparation and Splitting.

Fetches slices from three publicly licensed benchmark datasets:
  1. PAWS (Paraphrase Adversaries from Word Scrambling) — Google Research
     Validation split (labeled_final/validation-00000-of-00001.parquet)
     License: CC-BY-SA 4.0
  2. QQP (Quora Question Pairs) — GLUE / NYU MLL mirror
     Validation split (qqp/validation-00000-of-00001.parquet)
     License: Quora Terms of Service (Research evaluation)
  3. MS MARCO (Microsoft Machine Reading Comprehension) — Microsoft
     Validation split (v1.1/validation-00000-of-00001.parquet)
     License: Microsoft Research License / CC-BY 4.0

Outputs:
  - data/phase8_tuning_val.json: 160 pairs for threshold calibration and rule guard evaluation.
  - data/raw/phase8_sealed_test_stream.json: 200 query events for sealed simulation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent

# Output paths
TUNING_VAL_PATH = REPO_ROOT / "data" / "phase8_tuning_val.json"
SEALED_TEST_PATH = REPO_ROOT / "data" / "raw" / "phase8_sealed_test_stream.json"

PRIOR_DATASET_PATHS = [
    REPO_ROOT / "data" / "raw" / "query_pair_reuse_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_heldout.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_final_test.json",
    REPO_ROOT / "data" / "raw" / "query_stability_human_credibility.json",
    REPO_ROOT / "data" / "raw" / "synthetic_query_pair_feedback.json",
    REPO_ROOT / "data" / "raw" / "load_test_query_stream.json",
    REPO_ROOT / "data" / "raw" / "new_dataset_v3.json",
    REPO_ROOT / "data" / "raw" / "phase6_blind_eval_dataset.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_prior_queries() -> Set[str]:
    priors: Set[str] = set()
    for p in PRIOR_DATASET_PATHS:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("queries", [])
            for item in items:
                for k in ["query", "query_a", "query_b", "sentence1", "sentence2"]:
                    if k in item and isinstance(item[k], str):
                        priors.add(item[k].strip().lower())
        except Exception:
            continue
    return priors


def main() -> None:
    print("=" * 80)
    print("PHASE 8: PREPARING REAL BENCHMARK DATASETS & SPLITS")
    print("=" * 80)

    prior_queries = load_prior_queries()
    print(f"Loaded {len(prior_queries)} prior query strings for leakage exclusion.")

    # 1. Download raw validation parquet files
    print("\nDownloading public validation parquets via Hugging Face Hub...")
    paws_path = Path(hf_hub_download(
        repo_id="google-research-datasets/paws",
        filename="labeled_final/validation-00000-of-00001.parquet",
        repo_type="dataset",
    ))
    qqp_path = Path(hf_hub_download(
        repo_id="nyu-mll/glue",
        filename="qqp/validation-00000-of-00001.parquet",
        repo_type="dataset",
    ))
    ms_path = Path(hf_hub_download(
        repo_id="microsoft/ms_marco",
        filename="v1.1/validation-00000-of-00001.parquet",
        repo_type="dataset",
    ))

    # Compute and display checksums
    paws_hash = sha256_file(paws_path)
    qqp_hash = sha256_file(qqp_path)
    ms_hash = sha256_file(ms_path)

    print("\nSource Dataset Checksums (Validation splits):")
    print(f"  PAWS:     {paws_path.name} | SHA256: {paws_hash}")
    print(f"  QQP:      {qqp_path.name}  | SHA256: {qqp_hash}")
    print(f"  MS MARCO: {ms_path.name}  | SHA256: {ms_hash}")

    # Read tables
    paws_t = pq.read_table(paws_path)
    qqp_t = pq.read_table(qqp_path)
    ms_t = pq.read_table(ms_path)

    paws_rows = paws_t.to_pylist()
    qqp_rows = qqp_t.to_pylist()
    ms_rows = ms_t.to_pylist()

    print(f"\nRaw row counts:")
    print(f"  PAWS validation:     {len(paws_rows)}")
    print(f"  QQP validation:      {len(qqp_rows)}")
    print(f"  MS MARCO validation: {len(ms_rows)}")

    # -------------------------------------------------------------------------
    # Build Tuning Validation Split (160 pairs)
    # -------------------------------------------------------------------------
    print("\nBuilding Tuning Validation Split (N=160 pairs)...")
    tuning_pairs: List[Dict[str, Any]] = []

    # 60 QQP positive pairs (label == 1)
    qqp_pos_idx = 0
    for r in qqp_rows:
        if r["label"] == 1:
            q1, q2 = r["question1"].strip(), r["question2"].strip()
            if q1.lower() in prior_queries or q2.lower() in prior_queries or not q1 or not q2:
                continue
            tuning_pairs.append({
                "id": f"tune_qqp_pos_{qqp_pos_idx:03d}",
                "query_a": q1,
                "query_b": q2,
                "is_equivalent": True,
                "dataset": "qqp",
                "original_label": 1,
            })
            qqp_pos_idx += 1
            if qqp_pos_idx >= 60:
                break

    # 40 PAWS adversarial negative pairs (label == 0: word-swapped/adversarial)
    paws_neg_idx = 0
    for r in paws_rows:
        if r["label"] == 0:
            s1, s2 = r["sentence1"].strip(), r["sentence2"].strip()
            if s1.lower() in prior_queries or s2.lower() in prior_queries or not s1 or not s2:
                continue
            tuning_pairs.append({
                "id": f"tune_paws_neg_{paws_neg_idx:03d}",
                "query_a": s1,
                "query_b": s2,
                "is_equivalent": False,
                "dataset": "paws",
                "original_label": 0,
            })
            paws_neg_idx += 1
            if paws_neg_idx >= 40:
                break

    # 30 QQP negative pairs (label == 0: standard non-duplicates)
    qqp_neg_idx = 0
    for r in qqp_rows:
        if r["label"] == 0:
            q1, q2 = r["question1"].strip(), r["question2"].strip()
            if q1.lower() in prior_queries or q2.lower() in prior_queries or not q1 or not q2:
                continue
            tuning_pairs.append({
                "id": f"tune_qqp_neg_{qqp_neg_idx:03d}",
                "query_a": q1,
                "query_b": q2,
                "is_equivalent": False,
                "dataset": "qqp",
                "original_label": 0,
            })
            qqp_neg_idx += 1
            if qqp_neg_idx >= 30:
                break

    # 30 MS MARCO random query pairs (label == 0: distinct organic queries)
    ms_idx = 0
    for i in range(0, len(ms_rows) - 1, 2):
        qa = ms_rows[i]["query"].strip()
        qb = ms_rows[i+1]["query"].strip()
        if qa.lower() in prior_queries or qb.lower() in prior_queries or not qa or not qb:
            continue
        tuning_pairs.append({
            "id": f"tune_ms_neg_{ms_idx:03d}",
            "query_a": qa,
            "query_b": qb,
            "is_equivalent": False,
            "dataset": "ms_marco",
            "original_label": 0,
        })
        ms_idx += 1
        if ms_idx >= 30:
            break

    print(f"Total tuning pairs assembled: {len(tuning_pairs)}")
    with open(TUNING_VAL_PATH, "w", encoding="utf-8") as f:
        json.dump(tuning_pairs, f, indent=2)
    print(f"Saved tuning validation set to: {TUNING_VAL_PATH}")

    # Track queries used in tuning so test split is strictly disjoint
    used_in_tuning: Set[str] = set()
    for item in tuning_pairs:
        used_in_tuning.add(item["query_a"].lower())
        used_in_tuning.add(item["query_b"].lower())

    # -------------------------------------------------------------------------
    # Build Sealed Test Split (N=200 query stream)
    # -------------------------------------------------------------------------
    print("\nBuilding Sealed Evaluation Test Stream (N=200 query events)...")
    # Stream composition:
    #   - 50 cold seeds (from MS MARCO / QQP, indexable)
    #   - 40 exact/canonical repeats of seeds (testing normalizer & exact match)
    #   - 40 semantic paraphrases of seeds (testing vector search + local model)
    #   - 30 adversarial negatives of seeds (from PAWS)
    #   - 40 unrelated organic queries (from MS MARCO)
    test_stream: List[Dict[str, Any]] = []

    # Select 50 seed pairs from QQP validation (label == 1) strictly disjoint from tuning
    seed_pairs: List[Tuple[str, str]] = []
    qqp_offset = 200
    for r in qqp_rows[qqp_offset:]:
        if r["label"] == 1:
            q1, q2 = r["question1"].strip(), r["question2"].strip()
            if (q1.lower() in prior_queries or q2.lower() in prior_queries or
                q1.lower() in used_in_tuning or q2.lower() in used_in_tuning or
                not q1 or not q2 or len(q1.split()) < 3):
                continue
            seed_pairs.append((q1, q2))
            if len(seed_pairs) >= 50:
                break

    # Select 30 adversarial pairs from PAWS validation (label == 0) strictly disjoint from tuning
    paws_offset = 150
    adv_pairs: List[Tuple[str, str]] = []
    for r in paws_rows[paws_offset:]:
        if r["label"] == 0:
            s1, s2 = r["sentence1"].strip(), r["sentence2"].strip()
            if (s1.lower() in prior_queries or s2.lower() in prior_queries or
                s1.lower() in used_in_tuning or s2.lower() in used_in_tuning or
                not s1 or not s2):
                continue
            adv_pairs.append((s1, s2))
            if len(adv_pairs) >= 30:
                break

    # Select 40 unrelated organic queries from MS MARCO validation
    ms_offset = 150
    unrelated_queries: List[str] = []
    for r in ms_rows[ms_offset:]:
        mq = r["query"].strip()
        if (mq.lower() in prior_queries or mq.lower() in used_in_tuning or
            not mq or len(mq.split()) < 3):
            continue
        unrelated_queries.append(mq)
        if len(unrelated_queries) >= 40:
            break

    # Assemble 200-query stream
    # 1. 50 Cold Seeds
    for i, (q1, _) in enumerate(seed_pairs, start=1):
        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": f"P8-SEED-{i:03d}",
            "query": q1,
            "query_type": "COLD_SEED",
            "reference_seed_id": None,
            "is_reuse_safe": False,
            "domain": "general_qa",
            "source": "qqp_val",
            "note": "Initial unseen cold seed query; expected MISS",
        })

    # 2. 40 Exact / Canonical Repeats (20 verbatim repeats, 20 canonical question frame folds)
    for i in range(40):
        seed_id = f"P8-SEED-{(i + 1):03d}"
        seed_q = seed_pairs[i][0]
        if i < 20:
            # Verbatim repeat
            repeat_q = seed_q
            q_type = "VERBATIM_REPEAT"
        else:
            # Canonical frame variation (e.g., adding or stripping question mark / "What is")
            if seed_q.lower().startswith("what is "):
                # Strip "what is " and add '?'
                repeat_q = seed_q[8:].strip()
                if not repeat_q.endswith("?"):
                    repeat_q += "?"
            else:
                # Add trailing question mark or normalize
                repeat_q = seed_q.rstrip("?") + "?"
            q_type = "CANONICAL_REPEAT"

        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": f"P8-REP-{(i + 1):03d}",
            "query": repeat_q,
            "query_type": q_type,
            "reference_seed_id": seed_id,
            "is_reuse_safe": True,
            "domain": "general_qa",
            "source": "qqp_val",
            "note": f"Foldable repeat of {seed_id}; expected AUTO_REUSE HIT",
        })

    # 3. 40 Semantic Paraphrases (testing vector search + local equivalence model)
    for i in range(40):
        seed_id = f"P8-SEED-{(i + 1):03d}"
        paraphrase_q = seed_pairs[i][1]  # q2 from positive QQP pair
        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": f"P8-PARA-{(i + 1):03d}",
            "query": paraphrase_q,
            "query_type": "SEMANTIC_PARAPHRASE",
            "reference_seed_id": seed_id,
            "is_reuse_safe": True,
            "domain": "general_qa",
            "source": "qqp_val",
            "note": f"Genuine semantic paraphrase of {seed_id}; safe reuse HIT",
        })

    # 4. 30 Adversarial Negatives (PAWS pairs: s2 presented after s1 indexed, unsafe reuse)
    for i, (s1, s2) in enumerate(adv_pairs, start=1):
        # Index s1 as adversarial seed
        seed_id = f"P8-ADV-SEED-{i:03d}"
        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": seed_id,
            "query": s1,
            "query_type": "ADVERSARIAL_SEED",
            "reference_seed_id": None,
            "is_reuse_safe": False,
            "domain": "adversarial",
            "source": "paws_val",
            "note": "Adversarial negative base sentence; expected MISS",
        })
        # Present s2 (word-swapped / meaning shifted)
        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": f"P8-ADV-PROBE-{i:03d}",
            "query": s2,
            "query_type": "ADVERSARIAL_TRAP",
            "reference_seed_id": seed_id,
            "is_reuse_safe": False,
            "domain": "adversarial",
            "source": "paws_val",
            "note": f"Adversarial word-scrambled probe against {seed_id}; UNSAFE reuse -> must BYPASS",
        })

    # 5. 40 Unrelated Organic Queries (MS MARCO)
    for i, mq in enumerate(unrelated_queries, start=1):
        test_stream.append({
            "event_idx": len(test_stream) + 1,
            "id": f"P8-MS-{i:03d}",
            "query": mq,
            "query_type": "UNRELATED_ORGANIC",
            "reference_seed_id": None,
            "is_reuse_safe": False,
            "domain": "bing_organic",
            "source": "ms_marco_val",
            "note": "Unrelated organic search query; expected BYPASS / MISS",
        })

    print(f"Total stream events assembled: {len(test_stream)}")
    # Ensure stream directory exists
    SEALED_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEALED_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_stream, f, indent=2)

    sealed_hash = sha256_file(SEALED_TEST_PATH)
    print(f"\nSealed Test Stream saved to: {SEALED_TEST_PATH}")
    print(f"Sealed Test Stream SHA-256: {sealed_hash}")
    print(f"Stream total events: {len(test_stream)}")
    print(f"  Safe reuse queries (TP target): {sum(1 for e in test_stream if e['is_reuse_safe'])}")
    print(f"  Unsafe / Miss queries (FP if reused): {sum(1 for e in test_stream if not e['is_reuse_safe'])}")


if __name__ == "__main__":
    main()
