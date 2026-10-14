import json
import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decision.local_equivalence import (
    DEFAULT_CROSS_ENCODER_MODEL,
    extract_numbers,
    extract_negations,
    extract_proper_entities,
)

data = json.load(open("data/phase8_tuning_val.json", encoding="utf-8"))
print(f"Loaded {len(data)} pairs from data/phase8_tuning_val.json")

model = CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)
pairs = [(d["query_a"], d["query_b"]) for d in data]
raw_preds = model.predict(pairs)

scores = []
for val in raw_preds:
    v = float(val)
    if v < 0.0 or v > 1.0:
        scores.append(float(torch.sigmoid(torch.tensor(v)).item()))
    else:
        scores.append(v)
scores = np.array(scores)
labels = np.array([1 if d["is_equivalent"] else 0 for d in data])

def check_guards(qa, qb, use_num=True, use_neg=True, use_ent=True):
    if use_num:
        na, nb = extract_numbers(qa), extract_numbers(qb)
        if na != nb:
            return False, "NUMBER_MISMATCH"
    if use_neg:
        ga, gb = extract_negations(qa), extract_negations(qb)
        if (len(ga) % 2) != (len(gb) % 2):
            return False, "NEGATION_MISMATCH"
    if use_ent:
        ea, eb = extract_proper_entities(qa), extract_proper_entities(qb)
        if ea and eb and not (ea & eb):
            return False, "ENTITY_MISMATCH"
    return True, "PASSED"

configs = [
    ("1. No guards (raw CrossEncoder only)", False, False, False),
    ("2. Numbers guard only", True, False, False),
    ("3. Negations guard only", False, True, False),
    ("4. Entities guard only", False, False, True),
    ("5. Numbers + Negations", True, True, False),
    ("6. All guards active (Numbers + Negations + Entities)", True, True, True),
]

thresh = 0.85
print(f"\nVALIDATION SET GUARD ABLATION AT THRESHOLD {thresh}:")
print(f"{'Configuration':<45} | Hits |  TP |  FP | Prec (%) | Rec (%) | IRR (%) | PAWS FP")
print("-" * 105)

for name, unum, uneg, uent in configs:
    tp, fp, fn, tn = 0, 0, 0, 0
    paws_fp = 0
    for i, item in enumerate(data):
        passed, reason = check_guards(item["query_a"], item["query_b"], unum, uneg, uent)
        score = scores[i]
        label = labels[i]
        pred_reuse = passed and (score >= thresh)

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
    prec = (tp / hits * 100.0) if hits > 0 else 0.0
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    irr = (fp / hits * 100.0) if hits > 0 else 0.0
    print(f"{name:<45} | {hits:4d} | {tp:3d} | {fp:3d} | {prec:7.2f}% | {rec:6.2f}% | {irr:6.2f}% | {paws_fp:7d}")
