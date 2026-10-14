"""Download training slices and prepare 2,000 pairs for fine-tuning candidate (b).

Sources:
  - QQP train split: 1,000 positive pairs (label 1.0) + 500 standard negative pairs (label 0.0)
  - PAWS train split: 500 hard adversarial negative pairs (label 0.0)
Total: 2,000 pairs (balanced 1,000 positive / 1,000 negative).

Strict data integrity:
  - Excludes any query appearing in data/phase8_tuning_val.json or data/raw/phase8_sealed_test_stream.json.
  - Fixes random seed = 42 for reproducibility.
  - Outputs to: data/candidate_b_train_pairs.json
"""

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_PATH = REPO_ROOT / "data" / "candidate_b_train_pairs.json"
VAL_PATH = REPO_ROOT / "data" / "phase8_tuning_val.json"
SEALED_PATH = REPO_ROOT / "data" / "raw" / "phase8_sealed_test_stream.json"


def load_exclusion_strings() -> Set[str]:
    """Load all queries from validation and test splits to strictly prevent leakage."""
    exclusions = set()
    if VAL_PATH.exists():
        for d in json.load(open(VAL_PATH, encoding="utf-8")):
            exclusions.add(d["query_a"].strip().lower())
            exclusions.add(d["query_b"].strip().lower())
    if SEALED_PATH.exists():
        for d in json.load(open(SEALED_PATH, encoding="utf-8")):
            exclusions.add(d["query"].strip().lower())
    return exclusions


def main():
    random.seed(42)
    exclusions = load_exclusion_strings()
    print(f"Loaded {len(exclusions)} strings to exclude from training set.")

    # 1. Download QQP train parquet
    print("\n[1/2] Downloading QQP train parquet...")
    qqp_file = Path(hf_hub_download(
        repo_id="nyu-mll/glue",
        filename="qqp/train-00000-of-00001.parquet",
        repo_type="dataset",
    ))
    qqp_table = pq.read_table(qqp_file)
    qqp_rows = qqp_table.to_pylist()
    print(f"QQP train total rows: {len(qqp_rows)}")

    # 2. Download PAWS train parquet
    print("\n[2/2] Downloading PAWS train parquet...")
    paws_file = Path(hf_hub_download(
        repo_id="google-research-datasets/paws",
        filename="labeled_final/train-00000-of-00001.parquet",
        repo_type="dataset",
    ))
    paws_table = pq.read_table(paws_file)
    paws_rows = paws_table.to_pylist()
    print(f"PAWS train total rows: {len(paws_rows)}")

    # Collect QQP positive pairs (label == 1)
    qqp_pos = []
    for r in qqp_rows:
        q1 = str(r.get("question1", "")).strip()
        q2 = str(r.get("question2", "")).strip()
        label = int(r.get("label", 0))
        if label == 1 and q1 and q2:
            if q1.lower() not in exclusions and q2.lower() not in exclusions:
                qqp_pos.append({"query_a": q1, "query_b": q2, "label": 1.0, "source": "qqp_train_pos"})
                if len(qqp_pos) >= 1000:
                    break

    # Collect QQP negative pairs (label == 0)
    qqp_neg = []
    for r in qqp_rows:
        q1 = str(r.get("question1", "")).strip()
        q2 = str(r.get("question2", "")).strip()
        label = int(r.get("label", 0))
        if label == 0 and q1 and q2:
            if q1.lower() not in exclusions and q2.lower() not in exclusions:
                qqp_neg.append({"query_a": q1, "query_b": q2, "label": 0.0, "source": "qqp_train_neg"})
                if len(qqp_neg) >= 500:
                    break

    # Collect PAWS hard negative pairs (label == 0)
    paws_neg = []
    for r in paws_rows:
        s1 = str(r.get("sentence1", "")).strip()
        s2 = str(r.get("sentence2", "")).strip()
        label = int(r.get("label", 1))
        if label == 0 and s1 and s2:
            if s1.lower() not in exclusions and s2.lower() not in exclusions:
                paws_neg.append({"query_a": s1, "query_b": s2, "label": 0.0, "source": "paws_train_hard_neg"})
                if len(paws_neg) >= 500:
                    break

    print(f"\nCollected training pairs:")
    print(f"  QQP Positives:       {len(qqp_pos)}")
    print(f"  QQP Negatives:       {len(qqp_neg)}")
    print(f"  PAWS Hard Negatives: {len(paws_neg)}")

    train_data = qqp_pos + qqp_neg + paws_neg
    random.shuffle(train_data)
    print(f"Total training pairs: {len(train_data)} (balanced: {sum(1 for d in train_data if d['label'] == 1.0)} pos, {sum(1 for d in train_data if d['label'] == 0.0)} neg)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    # Compute sha256
    h = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"Saved {OUTPUT_PATH.name} (SHA-256: {h})")


if __name__ == "__main__":
    main()
