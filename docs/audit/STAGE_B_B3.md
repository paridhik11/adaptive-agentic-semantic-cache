# B3 — held-out generalization failure

Train four-class fit was 100% for every seed; held-out dev agreement with the synthetic oracle is only 326/704 (seed 11), 515/704 (29), 449/704 (47). This is a large gap, with domain shift confounded with formatter/lineage holdout. No natural-language generalization claim follows. Labels are Codex-authored programmatic synthetic evidence, not human ground truth.

| Seed | Fixed diagnostic threshold | Safe accepts / actual positive opportunities | Unsafe / unknown accepts | Retrieval recall@3 | Additional veto rejections |
| --- | --- | --- | --- | --- | --- |
| 11 | 0.5 (not selected) | 4/256 | 0 / 0 | 1.0 | {'paraphrase': 1} |
| 29 | 0.5 (not selected) | 66/256 | 0 / 0 | 1.0 | {'added_requirement': 5} |
| 47 | 0.5 (not selected) | 60/256 | 0 / 0 | 1.0 | {} |

B3 scores actual candidates and their actual relations from the scoped bank. The metadata oracle is used ONLY for measurement, never acceptance. Exact controls bypass semantic decisions. Similarity retrieves top-3; it never directly approves reuse. Classifier chooses first; competing candidates abstain; the signature veto can only reject the chosen candidate. Unknown labels stay null.

All 256 nonexact positive opportunities have a suitable top-3 candidate. Primary-seed misses at 0.5: 251 classifier abstentions and 1 veto. Seed 29 has 190 classifier abstentions and seed 47 has 196. These are diagnostic counts, not final selected-policy results or independent accuracy. Seed 11 stays primary; no best-seed substitution.

41 baseline points use the same actual MiniLM candidate pool and original Stage A grids. Redis embedding is a cosine threshold comparator on that pool, not full Redis retrieval. Redis low-threshold sweep is same-set diagnostic. Quora/Redis/Quora-derived classifier carry possible pretraining overlap with public pair data. No public pairs used.

Source/receipts: artifacts/stage_b/dev/{scored.jsonl.gz,receipt.json,baseline-comparison.json,baseline-errors.jsonl.gz,classifier-veto-diagnostic.json,designated-classification.json}. Every baseline unsafe/unknown accept and missed positive is preserved. No threshold selected in B3. Model/veto/competition rules remain frozen for B4.

Commands:

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/score_stage_b_candidates.py --split dev --output-dir artifacts/stage_b/dev
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split dev
```

Natural review request and rubric: data/intent_stage_b/v2/natural-review-request.jsonl and docs/STAGE_B_NATURAL_REVIEW.md. The request is blank; independent human labels = 0. Known oracle/parser limitations are listed in artifacts/stage_b/EVIDENCE_LABELS.json. General serving remains disabled. No final read/evaluation, live/provider calls, or production changes.
