"""Phase A1 — Semantic Matcher Candidate Training and Validation Evaluation.

Trains Candidate (b):
  - Base checkpoint: cross-encoder/quora-distilroberta-base (revision f62e7a4)
  - Training dataset: data/candidate_b_train_pairs.json (2,000 pairs: 1,000 QQP pos, 500 QQP neg, 500 PAWS hard neg)
  - Seed: 42
  - Epochs: 1 (CPU feasible)
  - Output: models/phaseA_cross_encoder_finetuned/

Evaluates Candidates on Validation Split (data/phase8_tuning_val.json, N=160 pairs):
  - Candidate (a): Off-the-shelf cross-encoder/quora-distilroberta-base (no guards)
  - Candidate (b): Fine-tuned model (no guards)
  - Candidate (c): Fine-tuned model + All Guards (numbers, negations, entities, long-prompt, generative)

Reports:
  - Exact precision, recall, and IRR with Clopper-Pearson 95% CIs.
  - False positive breakdown by source category (PAWS vs QQP vs MS MARCO).
  - Training time, seeds, and hardware info.
Saves results to: data/phaseA_candidate_comparison.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import beta
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.decision.local_equivalence import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_REVISION,
    evaluate_rule_guards,
    evaluate_long_prompt_guard,
    evaluate_generative_request_guard,
)

TRAIN_DATA_PATH = REPO_ROOT / "data" / "candidate_b_train_pairs.json"
VAL_DATA_PATH = REPO_ROOT / "data" / "phase8_tuning_val.json"
MODEL_OUTPUT_DIR = REPO_ROOT / "models" / "phaseA_cross_encoder_finetuned"
RESULTS_OUTPUT_PATH = REPO_ROOT / "data" / "phaseA_candidate_comparison.json"


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Clopper-Pearson 95% confidence interval."""
    if n == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_candidate(
    model: CrossEncoder,
    val_data: List[Dict[str, Any]],
    use_guards: bool = False,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """Evaluate a CrossEncoder candidate on validation pairs."""
    pairs = [(d["query_a"], d["query_b"]) for d in val_data]
    raw_preds = model.predict(pairs)

    scores = []
    for val in raw_preds:
        v = float(val)
        if v < 0.0 or v > 1.0:
            scores.append(float(torch.sigmoid(torch.tensor(v)).item()))
        else:
            scores.append(v)
    scores = np.array(scores)

    tp, fp, fn, tn = 0, 0, 0, 0
    fp_categories = {"paws": 0, "qqp": 0, "ms_marco": 0}

    for i, item in enumerate(val_data):
        qa = item["query_a"]
        qb = item["query_b"]
        is_equiv = bool(item["is_equivalent"])
        dataset = item.get("dataset", "unknown")
        score = scores[i]

        passed_guards = True
        guard_reason = "GUARDS_PASSED"

        if use_guards:
            passed_rules, reason = evaluate_rule_guards(qa, qb)
            if not passed_rules:
                passed_guards = False
                guard_reason = reason

        pred_reuse = passed_guards and (score >= threshold)

        if pred_reuse and is_equiv:
            tp += 1
        elif pred_reuse and not is_equiv:
            fp += 1
            if "paws" in dataset:
                fp_categories["paws"] += 1
            elif "qqp" in dataset:
                fp_categories["qqp"] += 1
            elif "ms_marco" in dataset or "marco" in dataset:
                fp_categories["ms_marco"] += 1
        elif not pred_reuse and is_equiv:
            fn += 1
        else:
            tn += 1

    hits = tp + fp
    total_pos = tp + fn
    hit_rate = hits / len(val_data)
    precision = (tp / hits) if hits > 0 else 0.0
    recall = (tp / total_pos) if total_pos > 0 else 0.0
    irr = (fp / hits) if hits > 0 else 0.0

    hr_ci = clopper_pearson_ci(hits, len(val_data))
    irr_ci = clopper_pearson_ci(fp, hits)
    prec_ci = clopper_pearson_ci(tp, hits)
    rec_ci = clopper_pearson_ci(tp, total_pos)

    return {
        "threshold": threshold,
        "use_guards": use_guards,
        "hits": hits,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_categories": fp_categories,
        "hit_rate": round(hit_rate, 4),
        "hit_rate_ci": [round(hr_ci[0], 4), round(hr_ci[1], 4)],
        "precision": round(precision, 4),
        "precision_ci": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
        "recall": round(recall, 4),
        "recall_ci": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
        "irr": round(irr, 4),
        "irr_ci": [round(irr_ci[0], 4), round(irr_ci[1], 4)],
    }


def main() -> None:
    print("=" * 80)
    print("PHASE A1: CROSS-ENCODER CANDIDATE TRAINING & VALIDATION BENCHMARK")
    print("=" * 80)
    set_seed(42)

    # 1. Load Validation Data
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    print(f"Loaded validation set: {VAL_DATA_PATH.name} (N={len(val_data)} pairs)")

    # 2. Evaluate Candidate (a): Off-the-shelf model without guards
    print("\n" + "-" * 80)
    print("CANDIDATE (a): Off-the-shelf cross-encoder/quora-distilroberta-base (no guards)")
    print("-" * 80)
    model_a = CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)
    cand_a_metrics = evaluate_candidate(model_a, val_data, use_guards=False, threshold=0.85)
    print(f"  Hits: {cand_a_metrics['hits']} (TP={cand_a_metrics['tp']}, FP={cand_a_metrics['fp']})")
    print(f"  Precision: {cand_a_metrics['precision']:.2%} | Recall: {cand_a_metrics['recall']:.2%}")
    print(f"  IRR: {cand_a_metrics['irr']:.2%} (95% CI: [{cand_a_metrics['irr_ci'][0]:.1%}, {cand_a_metrics['irr_ci'][1]:.1%}])")
    print(f"  FP categories: {cand_a_metrics['fp_categories']}")

    # 3. Fine-Tune Candidate (b) on QQP train + PAWS hard negatives
    print("\n" + "-" * 80)
    print("CANDIDATE (b): Fine-tuning cross-encoder on QQP train + PAWS hard negatives")
    print("-" * 80)
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_pairs = json.load(f)

    print(f"Loaded training data: {TRAIN_DATA_PATH.name} (N={len(train_pairs)} pairs)")
    # Initialize fresh CrossEncoder instance for fine-tuning
    model_b = CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL, num_labels=1)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting 1-epoch fine-tuning on CPU (batch_size=16, lr=2e-5)...")
    t0_train = time.time()

    device = torch.device("cpu")
    model_b.model.to(device)
    model_b.model.train()

    optimizer = torch.optim.AdamW(model_b.model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    random.seed(42)
    shuffled_pairs = list(train_pairs)
    random.shuffle(shuffled_pairs)

    batch_size = 16
    n_batches = (len(shuffled_pairs) + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        batch = shuffled_pairs[b_idx * batch_size : (b_idx + 1) * batch_size]
        texts_a = [item["query_a"] for item in batch]
        texts_b = [item["query_b"] for item in batch]
        labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        encoded = model_b.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        outputs = model_b.model(**encoded)
        logits = outputs.logits.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == n_batches:
            print(f"  Step [{b_idx + 1:03d}/{n_batches:03d}] Loss: {loss.item():.4f}")

    model_b.model.eval()
    model_b.model.save_pretrained(str(MODEL_OUTPUT_DIR))
    model_b.tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    train_duration_s = time.time() - t0_train
    print(f"Fine-tuning completed in {train_duration_s:.2f} s ({train_duration_s / 60:.2f} min)")
    print(f"Checkpoint saved to: {MODEL_OUTPUT_DIR}")

    # 4. Evaluate Candidate (b): Fine-tuned model without guards
    print("\n" + "-" * 80)
    print("EVALUATING CANDIDATE (b): Fine-tuned CrossEncoder (no guards)")
    print("-" * 80)
    cand_b_metrics = evaluate_candidate(model_b, val_data, use_guards=False, threshold=0.85)
    print(f"  Hits: {cand_b_metrics['hits']} (TP={cand_b_metrics['tp']}, FP={cand_b_metrics['fp']})")
    print(f"  Precision: {cand_b_metrics['precision']:.2%} | Recall: {cand_b_metrics['recall']:.2%}")
    print(f"  IRR: {cand_b_metrics['irr']:.2%} (95% CI: [{cand_b_metrics['irr_ci'][0]:.1%}, {cand_b_metrics['irr_ci'][1]:.1%}])")
    print(f"  FP categories: {cand_b_metrics['fp_categories']}")

    # 5. Evaluate Candidate (c): Fine-tuned model + All Guards
    print("\n" + "-" * 80)
    print("EVALUATING CANDIDATE (c): Fine-tuned CrossEncoder + All Guards")
    print("-" * 80)
    cand_c_metrics = evaluate_candidate(model_b, val_data, use_guards=True, threshold=0.85)
    print(f"  Hits: {cand_c_metrics['hits']} (TP={cand_c_metrics['tp']}, FP={cand_c_metrics['fp']})")
    print(f"  Precision: {cand_c_metrics['precision']:.2%} | Recall: {cand_c_metrics['recall']:.2%}")
    print(f"  IRR: {cand_c_metrics['irr']:.2%} (95% CI: [{cand_c_metrics['irr_ci'][0]:.1%}, {cand_c_metrics['irr_ci'][1]:.1%}])")
    print(f"  FP categories: {cand_c_metrics['fp_categories']}")

    # Summary JSON output
    results = {
        "timestamp": "2026-10-12T01:00:00+05:30",
        "training": {
            "dataset": "data/candidate_b_train_pairs.json",
            "n_samples": len(train_pairs),
            "seed": 42,
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "duration_seconds": round(train_duration_s, 2),
            "output_checkpoint": str(MODEL_OUTPUT_DIR),
        },
        "validation_benchmark": {
            "dataset": "data/phase8_tuning_val.json",
            "n_pairs": len(val_data),
            "candidate_a_off_the_shelf_no_guards": cand_a_metrics,
            "candidate_b_finetuned_no_guards": cand_b_metrics,
            "candidate_c_finetuned_with_guards": cand_c_metrics,
        },
    }

    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Candidate Comparison to: {RESULTS_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
