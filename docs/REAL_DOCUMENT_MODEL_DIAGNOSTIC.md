# Actual local-model diagnostic — 2026-11-01

User priority: actual project work; GitHub postponed. No further publication
attempts. Earlier uploaded blob objects are not a published branch head.

## Operational repair

Hugging Face was reachable through normal access. Restored exactly:

- MiniLM `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Quora `cross-encoder/quora-distilroberta-base` at
  `f62e7a4b20b97195c2868e53ec59126df5eac743`.

Pins came from the approved user instructions and repository constants in
`src/policy/local_models.py`, not memory-based guesses. The existing download
script validates both revisions, uses explicit `snapshot_download` revisions and
an allowlist, and refuses pickle fallback or revision substitution. Both pinned
model cards declare Apache-2.0; every file's path, size, SHA-256 and license is in
`artifacts/documentation_rebuild/group4/model-restore-v1.json`. Weights reside in
the expected Hugging Face cache outside Git; the inventory verified safetensors
ignore rules. Download and local inference occurred in this same workspace.
No Redis/MS MARCO/other checkpoint or new training data was downloaded.

MiniLM loaded and produced 384-dimensional vectors. Quora initially failed because
the existing constructor used `activation_fn`, unsupported by installed
SentenceTransformers 3.4.1. Inspection of its actual constructor confirmed
`default_activation_function`; the wrapper now uses explicit Identity activation
and `use_safetensors=True`. This preserves raw-logit scoring and the existing
single sigmoid. It is a runtime repair, not a model error or threshold adjustment.
Quora then executed successfully. Torch is 2.6.0 CPU; two threads were used.

## Diagnostic, not fresh final evaluation

The configuration was written before scoring and uses pre-existing diagnostic
settings: cosine .8, pair score .7, pair margin .05. No grid or threshold selection.
The scorer reads blank human templates/questions and three source candidates,
not human judgments. It writes predictions and their hash before the separate
reporter reads labels. The builder has already seen these items; this is still
an exposed diagnostic and must not be advertised as a fresh held-out test.

| Fixed recommendation method | Recommended / 30 | Human sufficient | Human insufficient | Unknown selected answer | Abstained |
| --- | --- | --- | --- | --- | --- |
| Dense-only | 3 | 2 | 1 | 0 | 27 |
| Quora-only | 0 | 0 | 0 | 0 | 30 |
| Combined + margin | 0 | 0 | 0 | 0 | 30 |

The unsafe dense recommendation H12 asks to convert a list of lists into a tuple
of tuples. Its selected answer covers ordinary outer conversion, not nested
conversion. Cosine was 0.9234989537385168. A high similarity score is therefore
insufficient here. Zero pair/combined coverage is a failed useful-coverage result,
not evidence of high precision. Scores are not calibrated correctness probabilities.
Quora retains the possible-pretraining-overlap-with-public-pair-data flag; this
cohort itself is model-authored over exposed public FAQ excerpts, not public QQP.

Labels apply only when the actual selected seed matches the reviewed source.
Other selected answers would remain unknown. All three recommendations matched
reviewed sources. Thirty correlated, model-authored questions over three FAQ
topics do not establish population accuracy. Counts only are reported: no
precision percentage, safety guarantee, representative coverage or savings claim.
These recommendations did not serve answers and changed no serving policy.

90 pair scorer invocations and 33 dense encoding calls were made locally.
No remote planning/generation/review, provider API or fresh agent-answer generation.
120 focused regression checks passed in 3.36s, including constructor compatibility,
window refusal, explicit sigmoid behavior and cache/agent invariants.

## Reproduce (fresh output paths)

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/download_stage_a_checkpoints.py --output NEW_DOWNLOAD_RECEIPT.json
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/inventory_intent_artifacts.py --output NEW_INVENTORY.json --download-receipt NEW_DOWNLOAD_RECEIPT.json
PYTHONPATH=.venv/lib/python3.12/site-packages HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/score_document_diagnostic.py --config configs/document_product/exposed_diagnostic_v1.json --output-dir NEW_PREDICTIONS
python scripts/report_document_diagnostic.py --predictions NEW_PREDICTIONS/predictions.jsonl --human artifacts/documentation_rebuild/group4/human-review-v1/human-responses.jsonl --output NEW_COMPARISON.json
```

The download is not needed if those exact snapshots remain cached. The historical
missing-weight preflight is retained; a new restored preflight is separate.
All output commands refuse overwrites. Model hashes, runtime and source identities
are retained beside the sealed predictions. Existing Stage A/B datasets, models,
thresholds and historical results were untouched; locked final records were not
opened or evaluated. No training, serving promotion or GitHub branch publication.

The unresolved product task is actual-answer applicability for unfamiliar queries.
This diagnostic supplies concrete failure evidence; it does not complete that
task. Do not solve it by lowering thresholds against this reviewed cohort.
