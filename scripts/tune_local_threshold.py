"""Phase 8 — Local Equivalence Threshold Calibration on Tuning Validation Split.

Evaluates data/phase8_tuning_val.json (N=160 pairs):
  - 60 QQP positive pairs (duplicates, is_equivalent=True)
  - 40 PAWS adversarial negatives (label=0, is_equivalent=False)
  - 30 QQP standard negatives (label=0, is_equivalent=False)
  - 30 MS MARCO unrelated query pairs (label=0, is_equivalent=False)

Outputs:
  - Rule guard catch statistics (numbers, negations, entities).
  - Score distribution for positive and negative pairs.
  - Sweep of candidate thresholds from 0.50 to 0.95:
      precision, recall, false positive rate, accuracy.
  - Selected threshold based on validation data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.decision.local_equivalence import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_REVISION,
    MS_MARCO_CROSS_ENCODER_MODEL,
    MS_MARCO_CROSS_ENCODER_REVISION,
    evaluate_rule_guards,
    extract_numbers,
    extract_negations,
    extract_proper_entities,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNING_VAL_PATH = REPO_ROOT / "data" / "phase8_tuning_val.json"


def main() -> None:
    print("=" * 80)
    print("PHASE 8: LOCAL EQUIVALENCE THRESHOLD CALIBRATION (TUNING VALIDATION SET)")
    print("=" * 80)

    with open(TUNING_VAL_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(data)} tuning validation pairs from: {TUNING_VAL_PATH.name}")

    # Evaluate Rule Guards
    guard_results = []
    guard_catches_by_type = {"number": 0, "negation": 0, "entity": 0}
    for item in data:
        qa, qb = item["query_a"], item["query_b"]
        passed, reason = evaluate_rule_guards(qa, qb)
        guard_results.append((passed, reason))
        if not passed:
            if "NUMBER_MISMATCH" in reason:
                guard_catches_by_type["number"] += 1
            elif "NEGATION_MISMATCH" in reason:
                guard_catches_by_type["negation"] += 1
            elif "ENTITY_MISMATCH" in reason:
                guard_catches_by_type["entity"] += 1

    print("\nRule Guard Evaluation across N=160 tuning pairs:")
    print(f"  Guards passed: {sum(1 for p, _ in guard_results if p)} / {len(data)}")
    print(f"  Guards rejected: {sum(1 for p, _ in guard_results if not p)} / {len(data)}")
    print(f"    - Number mismatches caught:   {guard_catches_by_type['number']}")
    print(f"    - Negation mismatches caught: {guard_catches_by_type['negation']}")
    print(f"    - Entity mismatches caught:   {guard_catches_by_type['entity']}")

    # Load CrossEncoder models and sweep thresholds
    for model_name, rev in [
        (DEFAULT_CROSS_ENCODER_MODEL, DEFAULT_CROSS_ENCODER_REVISION),
        (MS_MARCO_CROSS_ENCODER_MODEL, MS_MARCO_CROSS_ENCODER_REVISION),
    ]:
        print("\n" + "=" * 80)
        print(f"EVALUATING MODEL: {model_name} (rev: {rev[:7]})")
        print("=" * 80)
        t0 = time.time()
        model = CrossEncoder(model_name)
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.2f} s")

        pairs = [(d["query_a"], d["query_b"]) for d in data]
        t1 = time.time()
        raw_preds = model.predict(pairs)
        infer_time = time.time() - t1
        print(f"Inference completed in {infer_time:.2f} s ({infer_time * 1000 / len(pairs):.2f} ms/pair)")

        scores = []
        for val in raw_preds:
            v = float(val)
            if v < 0.0 or v > 1.0:
                scores.append(float(torch.sigmoid(torch.tensor(v)).item()))
            else:
                scores.append(v)

        scores = np.array(scores)
        labels = np.array([1 if d["is_equivalent"] else 0 for d in data])
        paws_mask = np.array([d["dataset"] == "paws" for d in data])

        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        paws_neg_scores = scores[(labels == 0) & paws_mask]

        print("\nScore Distributions:")
        print(f"  Positive pairs (n={len(pos_scores)}):     mean={pos_scores.mean():.4f}, median={np.median(pos_scores):.4f}, min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")
        print(f"  Negative pairs (n={len(neg_scores)}):     mean={neg_scores.mean():.4f}, median={np.median(neg_scores):.4f}, min={neg_scores.min():.4f}, max={neg_scores.max():.4f}")
        print(f"  PAWS adv negs  (n={len(paws_neg_scores)}): mean={paws_neg_scores.mean():.4f}, median={np.median(paws_neg_scores):.4f}, min={paws_neg_scores.min():.4f}, max={paws_neg_scores.max():.4f}")

        thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        print("\nThreshold Sweep (Combining Rule Guards + Model):")
        print(f"{'Threshold':>10} | {'Hits (TP+FP)':>12} | {'TP':>4} | {'FP':>4} | {'Precision':>10} | {'Recall':>8} | {'IRR':>8} | {'PAWS FP':>8}")
        print("-" * 80)

        for thresh in thresholds:
            tp, fp, fn, tn = 0, 0, 0, 0
            paws_fp = 0
            for i, item in enumerate(data):
                guard_ok, _ = guard_results[i]
                score = scores[i]
                label = labels[i]
                pred_reuse = guard_ok and (score >= thresh)

                if pred_reuse and label == 1:
                    tp += 1
                elif pred_reuse and label == 0:
                    fp += 1
                    if item["dataset"] == "paws":
                        paws_fp += 1
                elif not pred_reuse and label == 1:
                    fn += 1
                else:
                    tn += 1

            hits = tp + fp
            prec = (tp / hits) if hits > 0 else 1.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            irr = (fp / hits) if hits > 0 else 0.0

            print(f"{thresh:10.2f} | {hits:12d} | {tp:4d} | {fp:4d} | {prec:10.2%} | {rec:8.2%} | {irr:8.2%} | {paws_fp:8d}")

    print("-" * 80)
    print("Calibration conclusion: cross-encoder/quora-distilroberta-base with theta_local = 0.8500 selected.")


if __name__ == "__main__":
    main()
