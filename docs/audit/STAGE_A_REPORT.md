# Stage A report — stop after A4

Evidence: developer-authored and template-generated development set. Not independent human review, calibration or final evaluation. No threshold selected; no trained model. Same-set supplementary Redis diagnostics are labeled below.

## Files and preservation

`data/intent_stage_a/` contains the dataset, split answer stores, manifests and dev metadata-completion provenance. `configs/intent_stage_a*.json` freeze the primary and supplementary grids. `scripts/build_stage_a_intents.py`, `scripts/complete_stage_a_dev_signatures.py`, `scripts/run_stage_a_baselines.py`, `scripts/report_stage_a.py`, and `src/evaluation/intent_stage_a.py` implement generation, inference and policy-agnostic metrics. `artifacts/stage_a/` preserves original scores, candidates, metrics, logs, corrected metrics and all decisions. Nothing was committed, pushed or branched.

## Dataset counts

| Split | Groups | Source lineages | Rows | New-question rows |
| --- | ---: | ---: | ---: | ---: |
| train | 96 | 8 | 1248 | 1152 |
| dev | 59 | 24 | 727 | 668 |
| calibration | 36 | 3 | 468 | 432 |
| final | 24 | 2 | 312 | 288 |

Groups are task contracts, not independent observations. Generated groups vary operation, input container and a material response requirement. Shared operation lineage is kept within one split. Formatter templates are shared across splits, so split isolation does not establish broad linguistic generalization.

All 75 old cases are spent dev data. Historical wrong answers remain available for audit and can have same-intent pair labels. Generated answers are authored code examples, not measured LLM generations; presentation/version compliance is not independently reviewed. No public corpus was imported: existing historical sealed/FAQ/pilot material was excluded, and no vetted license-checked local public-pair corpus was selected.

Final manifest SHA-256: `1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af`. Written into already tracked `docs/FACTS.md` at creation. Final questions and answers were not evaluated; the harness rejects `--split final` before data/model loading. Only final manifest metadata/hashes are checked. Train/calibration received no model evaluation.

### Counts per split, family and provenance

```json
{
  "train": {
    "families": {
      "cached_control": 96,
      "natural_paraphrase": 288,
      "exact_control": 96,
      "added_requirement": 96,
      "removed_requirement": 96,
      "negation_polarity": 96,
      "entity_function_swap": 96,
      "direction_role_inversion": 96,
      "quantity_version_unit": 96,
      "negative_control": 96,
      "ambiguity": 96
    },
    "provenance": {
      "template-generated": 1248
    },
    "relations": {
      "same_intent": 480,
      "new_more_specific": 192,
      "new_more_general": 96,
      "different_intent": 384,
      "None": 96
    }
  },
  "dev": {
    "families": {
      "cached_control": 59,
      "natural_paraphrase": 225,
      "direction_role_inversion": 52,
      "added_requirement": 63,
      "negation_polarity": 60,
      "quantity_version_unit": 37,
      "entity_function_swap": 64,
      "exact_control": 59,
      "removed_requirement": 36,
      "negative_control": 36,
      "ambiguity": 36
    },
    "provenance": {
      "authored": 121,
      "template-generated": 606
    },
    "relations": {
      "same_intent": 343,
      "different_intent": 213,
      "new_more_specific": 99,
      "new_more_general": 36,
      "None": 36
    }
  },
  "calibration": {
    "families": {
      "cached_control": 36,
      "natural_paraphrase": 108,
      "exact_control": 36,
      "added_requirement": 36,
      "removed_requirement": 36,
      "negation_polarity": 36,
      "entity_function_swap": 36,
      "direction_role_inversion": 36,
      "quantity_version_unit": 36,
      "negative_control": 36,
      "ambiguity": 36
    },
    "provenance": {
      "template-generated": 468
    },
    "relations": {
      "same_intent": 180,
      "new_more_specific": 72,
      "new_more_general": 36,
      "different_intent": 144,
      "None": 36
    }
  },
  "final": {
    "families": {
      "cached_control": 24,
      "natural_paraphrase": 72,
      "exact_control": 24,
      "added_requirement": 24,
      "removed_requirement": 24,
      "negation_polarity": 24,
      "entity_function_swap": 24,
      "direction_role_inversion": 24,
      "quantity_version_unit": 24,
      "negative_control": 24,
      "ambiguity": 24
    },
    "provenance": {
      "template-generated": 312
    },
    "relations": {
      "same_intent": 120,
      "new_more_specific": 48,
      "new_more_general": 24,
      "different_intent": 96,
      "None": 24
    }
  }
}
```

## A4 primary fixed-grid baseline table

DESIGNATED PAIRS: every policy sees the same cached/new question pair. These are not serving hit-rate measurements. Exact controls are excluded from semantic precision/coverage. Precision is null when no non-exact accepts exist or accepted labels are unknown. Unknowns are never imputed safe.

| Baseline | Threshold | Non-exact accepts | Pair precision | Positive coverage | Unsafe | Unknown | Overlap flag* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| always_miss | None | 0 | null | 0.0000 | 0 | 0 | not asserted |
| exact_only | None | 0 | null | 0.0000 | 0 | 0 | not asserted |
| minilm_cosine | 0.5 | 476 | 0.4706 | 1.0000 | 252 | 0 | not asserted |
| redis_embedding_cosine | 0.5 | 495 | null | 1.0000 | 269 | 2 | yes |
| minilm_cosine | 0.6 | 473 | 0.4693 | 0.9911 | 251 | 0 | not asserted |
| redis_embedding_cosine | 0.6 | 481 | 0.4657 | 1.0000 | 257 | 0 | yes |
| minilm_cosine | 0.7 | 460 | 0.4587 | 0.9420 | 249 | 0 | not asserted |
| redis_embedding_cosine | 0.7 | 466 | 0.4764 | 0.9911 | 244 | 0 | yes |
| minilm_cosine | 0.8 | 446 | 0.4484 | 0.8929 | 246 | 0 | not asserted |
| redis_embedding_cosine | 0.8 | 445 | 0.4697 | 0.9330 | 236 | 0 | yes |
| minilm_cosine | 0.9 | 356 | 0.5169 | 0.8214 | 172 | 0 | not asserted |
| redis_embedding_cosine | 0.9 | 333 | 0.5465 | 0.8125 | 151 | 0 | yes |
| minilm_cosine | 0.95 | 277 | 0.6101 | 0.7545 | 108 | 0 | not asserted |
| redis_embedding_cosine | 0.95 | 258 | 0.6667 | 0.7679 | 86 | 0 | yes |
| quora | 0.5 | 371 | 0.4906 | 0.8125 | 189 | 0 | yes |
| redis_reranker | 0.5 | 0 | null | 0.0000 | 0 | 0 | yes |
| existing_local_equivalence | 0.5 | 98 | 0.4694 | 0.2054 | 52 | 0 | yes |
| quora | 0.7 | 348 | 0.5172 | 0.8036 | 168 | 0 | yes |
| redis_reranker | 0.7 | 0 | null | 0.0000 | 0 | 0 | yes |
| existing_local_equivalence | 0.7 | 85 | 0.5294 | 0.2009 | 40 | 0 | yes |
| quora | 0.9 | 287 | 0.6028 | 0.7723 | 114 | 0 | yes |
| redis_reranker | 0.9 | 0 | null | 0.0000 | 0 | 0 | yes |
| existing_local_equivalence | 0.9 | 57 | 0.7193 | 0.1830 | 16 | 0 | yes |
| quora | 0.95 | 209 | 0.6746 | 0.6295 | 68 | 0 | yes |
| redis_reranker | 0.95 | 0 | null | 0.0000 | 0 | 0 | yes |
| existing_local_equivalence | 0.95 | 45 | 0.8667 | 0.1741 | 6 | 0 | yes |
| quora | 0.99 | 1 | 0.0000 | 0.0000 | 1 | 0 | yes |
| redis_reranker | 0.99 | 0 | null | 0.0000 | 0 | 0 | yes |
| existing_local_equivalence | 0.99 | 0 | null | 0.0000 | 0 | 0 | yes |

*Flag text: **possible pretraining overlap with public pair data**. Applied to Quora, both Redis models, and the Quora-based existing guard stack; no public pairs are used in this dataset. The flag must stay in future public-pair evaluation tables. Unflagged does not mean proven contamination-free.

### Redis supplementary grid — SAME-SET diagnostic, no new inference

The primary 0.5–0.99 grid rejected everything. A broader low-score grid and publisher-provided thresholds are applied to the saved logits. This is exposed-dev descriptive analysis, not an independently selected operating point or calibration result.

| Baseline | Threshold | Non-exact accepts | Pair precision | Positive coverage | Unsafe | Unknown | Overlap flag* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| redis_reranker | 0.0 | 608 | null | 1.0000 | 348 | 36 | yes |
| redis_reranker | 0.01 | 608 | null | 1.0000 | 348 | 36 | yes |
| redis_reranker | 0.025 | 608 | null | 1.0000 | 348 | 36 | yes |
| redis_reranker | 0.05 | 397 | null | 1.0000 | 137 | 36 | yes |
| redis_reranker | 0.07275390625 | 328 | null | 1.0000 | 69 | 35 | yes |
| redis_reranker | 0.075 | 322 | null | 0.9955 | 66 | 33 | yes |
| redis_reranker | 0.0888671875 | 300 | null | 0.9866 | 52 | 27 | yes |
| redis_reranker | 0.1 | 290 | null | 0.9821 | 46 | 24 | yes |
| redis_reranker | 0.2 | 190 | null | 0.7812 | 11 | 4 | yes |
| redis_reranker | 0.3 | 0 | null | 0.0000 | 0 | 0 | yes |
| redis_reranker | 0.4 | 0 | null | 0.0000 | 0 | 0 | yes |
| redis_reranker | 0.5 | 0 | null | 0.0000 | 0 | 0 | yes |

### Retrieval, separately

```json
{
  "denominator_nonexact_same_intent": 224,
  "recall_at_k": {
    "1": 0.9866071428571429,
    "3": 1.0,
    "5": 1.0
  },
  "note": "Retrieval against dev bank, separate from designated-pair decision metrics."
}
```

Every dev query has an immutable saved MiniLM top-5 candidate list. Decision baselines are compared on designated pairs; they do not rescore the retrieved bank in this Stage A study. A later integrated policy must be evaluated on actual retrieved candidates and their actual relations. Positive recall alone does not establish negative rejection.

## Group/lineage endpoints and uncertainty

For each table row, verified `metrics.json` includes accepted-group and accepted-lineage counts, all-accepted-safe outcomes, failed/unknown outcomes and exact one-sided 95% Clopper–Pearson bounds. A single definitive unsafe accept fails its group even when another label is unknown. Answer-level binomial intervals are intentionally absent because paraphrases are correlated. Group and lineage bounds are explicitly conditional on independent Bernoulli sampling, which these purposive templates do not establish. None is a population-safety guarantee.

## Failures and unavailable work

- Historical parsed-call policy: **NOT RUN**. Requires `deepset/minilm-uncased-squad2@934656cdda79824eabf503ed56e15c01ddbdbe3f` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. They are absent and not approved for download here. No helper was substituted as that baseline.
- Baselines show substantial unsafe accepts or coverage collapse. No policy is selected or enabled from these results. Every unsafe accept and missed positive is listed per baseline in verified metrics and decisions JSONL, with case family and cause.
- Existing local_equivalence has an unpinned lazy loader and a heuristic logit/probability conversion. The harness injects actual pinned Quora probabilities into the unchanged stack and disables its remote fallback. Model and guard timings here are diagnostic, not a serving-cost comparison.
- The initial harness preflight joined on event `case_id` instead of logical `cached_case_id`; fixed before inference. Original error log preserved. These are separate schema identifiers, not duplicate final records.
- Legacy signatures were completed before inference, and two metadata-completion receipts preserve that process. Any remaining unverified signature metadata is explicitly marked and is not a policy input; Stage B veto construction must review it.
- Generated labels are builder-authored assumptions; ambiguous relation is null. The locked template-generated final is not independent human ground truth.

## Redis artifact identities and licenses

- `redis/langcache-embed-v1@2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a`: current main at metadata check; card license: None (not_declared_in_pinned_card_metadata).
- `redis/langcache-reranker-v2-bce@f59539f4208f9005e10e45160a379441bdb42e9b`: current main at metadata check; card license: apache-2.0 (declared).

Newer embedding versions `redis/langcache-embed-v2` and `redis/langcache-embed-v3-small` exist. No later numbered BCE reranker was listed in the checked Redis repositories; `redis/langcache-reranker-v2-mnrl` is a later-created alternative loss variant, not a newer BCE revision. All metadata IDs and timestamps are in the receipt. No newer model files were downloaded. The embed-v1 card metadata does not declare a license; it is recorded as unknown rather than inferred from its base model.

## Exact commands

Run from the repository root. Dependencies are already installed under the local runtime path here. Use a fresh output directory when repeating inference; never overwrite a baseline run.

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/download_redis_stage_a.py --output docs/audit/STAGE_A_REDIS_RECEIPT.json
python scripts/inventory_intent_artifacts.py --download-receipt docs/audit/STAGE_A_DOWNLOAD_RECEIPT.json --extra-receipt docs/audit/STAGE_A_REDIS_RECEIPT.json --output docs/audit/STAGE_A_INVENTORY.json --replace-existing
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/build_stage_a_intents.py
python scripts/complete_stage_a_dev_signatures.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/run_stage_a_baselines.py --split dev --output-dir artifacts/stage_a/baselines-v1
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_a.py
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py
```

Generation refuses an existing dataset directory; do not regenerate the locked final. Restore the archived dataset/manifests for exact replication. Download/report commands also preserve existing receipts/output directories. Original output paths above identify the actual run; use fresh paths or a clean clone for recreation.

## Stop / next approval

Stage B training, veto construction, calibration and Stage C integration were not performed. Final evaluation, live/provider calls, UI/agent changes, serving enablement, Git commits/pushes/branches were not performed. No trained model exists to export. If Stage B is authorized later, its learned artifact must be exported as safetensors with hash, base revision, seed and exact training command, and added to the retrievable backup.

Minimum user action: back up the supplied archive, review this Stage A report, then give the explicit Stage B go-ahead if desired. Completing the missing historical parsed-call comparison additionally needs separate approval for its two pinned checkpoints; otherwise that comparator remains unavailable and B5 cannot claim to beat it.

## Review limits and schema recommendations

- `case_id` identifies a dataset event; `cached_case_id` identifies its logical cached question. They are deliberately distinct and join through the cached-role record's `cached_case_id`.
- Ambiguity has a null relation and explicit label status. Preserve that distinction; a forced four-class label would invent certainty.
- The current calibration/final group and lineage counts cannot establish the earlier 125 accepted-group final gate. Do not treat this locked generated split as ready for that claim. A separately designed, adequately sized independent final cohort would be required before any final run; do not expand a spent final after a failure.
- All sigmoid and cosine outputs are uncalibrated scores. No classifier threshold was selected, trained or calibrated in Stage A.

## Backup before any reset

Download `artifacts/stage_a/stage-a-backup.zip`. Its `BACKUP_MANIFEST.json` lists every included path, byte size and SHA-256. It contains all uncommitted scripts/configs/tests, the complete dataset and split answer stores (including the locked final for preservation), manifests, receipts, baseline scores/candidates/metrics/decisions/logs, this report and modified tracked docs. Extract over a checkout of the base Git commit recorded in the manifest. Do not open/evaluate final records as part of review.

If copying manually, preserve `data/intent_stage_a/`, `artifacts/stage_a/`, `configs/intent_stage_a*.json`, every `scripts/*stage_a*.py`, `scripts/inventory_intent_artifacts.py`, `src/evaluation/intent_stage_a.py`, `tests/test_intent_stage_a.py`, `docs/STAGE_A_PROTOCOL.md`, `docs/audit/STAGE_A*`, `docs/FACTS.md` and `docs/CURRENT_STATUS.md`. The authoritative exact file list is the backup manifest.

Pretrained weights and the Python environment are not in the backup. They are reproducible from the approved pinned download scripts, per-file receipts and `artifacts/stage_a/runtime-summary.json`; retain the external Hugging Face cache separately if offline restoration is required. No learned checkpoint exists. The archive remains uncommitted in the hosted workspace until you download/copy it; it has not been pushed.

Create and verify the backup with:

```bash
python scripts/backup_stage_a.py
```

Validation: `python -m pytest -q tests/test_intent_stage_a.py` passed 9 tests (see `artifacts/stage_a/tests.txt`). These check joins, split isolation, final refusal, answer/pair label separation and metric denominators, not model precision.

Authored provenance here means authored by the Codex implementation agent, not human-labeled. No independent model review or human review is claimed.
