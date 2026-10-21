# Stage B closeout — STOP after B5

**GENERALIZATION FAILURE / TEMPLATE-MEMORIZATION WARNING.** Primary-seed train four-class accuracy is 100.00%; held-out dev is 46.31%, calibration 45.45%. These are synthetic designated-pair diagnostics, not reuse precision or independent accuracy. The unchanged guarded Quora baseline at 0.7 achieves 128/256 safe dev reuses with 0 observed unsafe accepts under the synthetic oracle.

Labels are programmatic synthetic contracts authored by Codex (model-assisted construction), not human review. General semantic serving remains **DISABLED**. No human-written natural set has been returned. Nothing here is independent human ground truth. Models, generated contracts and the bounded veto vocabulary were authored by the same builder; template holdout alone does not remove that dependence.

## 1. Local branches and commits

Stage A: `6cea9ec` harness → `ec77788` data → `699223a` results. Stage B: `stage-b/classifier` from results. B0–B4 commits below; B5 is the report commit immediately following this generation. No push, merge, main rebase or history rewrite.

```text
ca1fe33 research(stage-b): B4 calibrate frozen policies and preserve resumable evidence
2058360 research(stage-b): B3 measure actual-candidate failures on held-out dev
4426088 research(stage-b): B2 add question-only veto and quarantine unverified metadata
a259797 research(stage-b): B1 train and export three pinned LoRA classifier seeds
9c07956 research(stage-b): B0 freeze diverse template-held-out cohorts and protocol
```

After B5: `git log --oneline stage-a/results..stage-b/classifier` gives the complete six-commit Stage B series. The backup includes the final Git bundle.

## 2. Files and experimental scope

New versioned datasets: data/intent_stage_b/v2. Unmeasured v1 construction draft and its generator are preserved. Train 96 groups/24 operation lineages, dev 64/8, calibration 256/8; whole question formatters and requirement/negative wrappers are disjoint across splits. All families and provenance counts are in each manifest. More numeric/version calibration variants increase group count, not independent lineage diversity.

Harness: build_stage_b_intents.py, stage_b_common.py, train_stage_b_classifier.py, diagnose_stage_b_train.py, review_stage_b_veto.py, score_stage_b_candidates.py, report_stage_b_dev.py, calibrate_stage_b_policy.py, benchmark_stage_b_local.py, report_stage_b_final.py, backup_stage_b.py. Model/veto/policy are isolated in src/evaluation/; production code is unchanged. Configs and protocol record all settings. tests/test_intent_stage_b.py covers isolation, nulls, actual-candidate attribution and veto/competition invariants.

Scored candidate files, predictions, model receipts, baseline grids, selected policies, comparisons, every selected-policy failure, per-family/template/lineage tables and timing observations are under artifacts/stage_b/. Raw large logs use deterministic lossless gzip. Model updates remain ignored by Git.

## 3. Exact commands

Run in an isolated Linux/WSL reproduction checkout with the pinned runtime and empty generated output directories. Native Windows parallel-runner support is not validated (worker locks use fcntl). Byte-identical original Stage A generator replay was verified on Linux; platform newline behavior elsewhere is not guaranteed. Existing evidence is deliberately not overwritten.

```bash
python scripts/build_stage_b_intents.py --output-dir scratch/stage-b-rebuild
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/train_stage_b_classifier.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/diagnose_stage_b_train.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/review_stage_b_veto.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/score_stage_b_candidates.py --split dev --output-dir artifacts/stage_b/dev
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split dev
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/run_stage_b_scoring.py --split calibration --output-dir artifacts/stage_b/calibration
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/calibrate_stage_b_policy.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split calibration
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/benchmark_stage_b_local.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_final.py
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py tests/test_intent_stage_b.py
```

Pretrained cache restoration uses only scripts/download_stage_a_checkpoints.py and scripts/download_redis_stage_a.py with their already-approved exact revisions. The restored inventory and receipts are preserved. Stage B restores and experiments share the same current environment. The original Stage A evidence came from an earlier host; all baseline methods were rerun within Stage B on its current cohorts. See environment-provenance.json; no cross-machine latency claim. No pickle or remote code was loaded.

## 4. Results — synthetic template-held-out actual-candidate evidence

| Split | Seed | Threshold | Safe / accepted | Unsafe / unknown | Precision | Safe coverage | Retrieval recall@3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 11 | None | 0/0 | 0 / 0 | null | 0.00% (0/256) | 100.00% |
| dev | 29 | None | 0/0 | 0 / 0 | null | 0.00% (0/256) | 100.00% |
| dev | 47 | None | 0/0 | 0 / 0 | null | 0.00% (0/256) | 100.00% |
| calibration | 11 | None | 0/0 | 0 / 0 | null | 0.00% (0/1024) | 99.02% |
| calibration | 29 | None | 0/0 | 0 / 0 | null | 0.00% (0/1024) | 99.02% |
| calibration | 47 | None | 0/0 | 0 / 0 | null | 0.00% (0/1024) | 99.02% |

Safe coverage counts actual cache-bank matches, including a designated negative that safely matches another cached question. A retrieval hit does not imply classifier acceptance. Exact hits are excluded from semantic precision and coverage. Precision with no accepts is null. Quora-derived policies and both Redis baselines carry **possible pretraining overlap with public pair data**; public pairs were not used.

Calibration used only calibration labels. Threshold/config hashes, adapter/base/embedding hashes, veto version, seed and abstention settings are frozen per policy. Exact one-sided group bounds use alpha=0.05/30 for the prespecified threshold/seed family. Pointwise group/lineage bounds are also available in baseline/subgroup tables. These are conditional arithmetic under independent Bernoulli cluster assumptions not established by purposive shared-template/lineage sets. No population guarantee follows.

| Split | Seed | Accepted groups / failures | Adjusted group error upper | Accepted lineages / failures | Adjusted lineage error upper |
| --- | --- | --- | --- | --- | --- |
| dev | 11 | 0 / 0 | null | 0 / 0 | null |
| dev | 29 | 0 / 0 | null | 0 / 0 | null |
| dev | 47 | 0 / 0 | null | 0 / 0 | null |
| calibration | 11 | 0 / 0 | null | 0 / 0 | null |
| calibration | 29 | 0 / 0 | null | 0 / 0 | null |
| calibration | 47 | 0 / 0 | null | 0 / 0 | null |

Why calibration failed: at the fixed 0.5 diagnostic threshold, seed 11 accepts 0 safe cases across 0 groups; seed 29 accepts 30 safe cases across 16 groups; seed 47 accepts 28 safe cases across 16 groups. No fixed-grid point meets the adjusted group-bound requirement. This is chiefly lack of acceptance evidence, not a discovered zero-error guarantee. Raising thresholds cannot repair missing useful coverage.


### Fixed Stage A baseline settings on the new cohorts

Evidence: actual retrieved-candidate synthetic development/calibration diagnostics, not human truth. These are the previously fixed baseline settings, not newly selected operating points.

| Split | Baseline / threshold | Safe / accepted | Unsafe / unknown | Safe coverage | Accepted groups / failures | Pointwise group error upper 95% |
| --- | --- | --- | --- | --- | --- | --- |
| dev | minilm_cosine / 0.8 | 256/661 | 405 / 0 | 100.00% | 64 / 64 | 100.00% |
| dev | redis_embedding_cosine / 0.8 | 252/684 | 432 / 0 | 98.44% | 64 / 64 | 100.00% |
| dev | quora / 0.7 | 224/661 | 437 / 0 | 87.50% | 64 / 64 | 100.00% |
| dev | existing_guard_stack / 0.7 | 128/128 | 0 / 0 | 50.00% | 64 / 0 | 4.57% |
| dev | redis_reranker_same_set_diagnostic / 0.0888671875 | 182/626 | 444 / 0 | 71.09% | 64 / 64 | 100.00% |
| calibration | minilm_cosine / 0.8 | 611/2564 | 1953 / 0 | 59.67% | 256 / 256 | 100.00% |
| calibration | redis_embedding_cosine / 0.8 | 1009/2698 | 1689 / 0 | 98.54% | 256 / 256 | 100.00% |
| calibration | quora / 0.7 | 1013/2514 | 1501 / 0 | 98.93% | 256 / 256 | 100.00% |
| calibration | existing_guard_stack / 0.7 | 1014/1014 | 0 / 0 | 99.02% | 256 / 0 | 1.16% |
| calibration | redis_reranker_same_set_diagnostic / 0.0888671875 | 1014/2272 | 1258 / 0 | 99.02% | 256 / 256 | 100.00% |

These baseline bounds are pointwise and conditional on the unestablished independent-cluster assumptions; they are not the multiplicity-adjusted trained-policy gate. Lineage bounds and every fixed grid point are in each baseline-comparison.json. No baseline is promoted or substituted for the prespecified primary seed.

### Baseline comparison at matched observed unsafe/unknown budget

| Split, primary seed | Comparator | Policy safe coverage | Best comparator coverage within budget | Strict improvement? |
| --- | --- | --- | --- | --- |
| dev, 11 | always_miss | 0.00% | 0.00% | False |
| dev, 11 | exact_only | 0.00% | 0.00% | False |
| dev, 11 | existing_guard_stack | 0.00% | 50.00% | False |
| dev, 11 | minilm_cosine | 0.00% | null | False |
| dev, 11 | quora | 0.00% | 0.78% | False |
| dev, 11 | redis_embedding_cosine | 0.00% | null | False |
| dev, 11 | redis_reranker | 0.00% | 0.00% | False |
| dev, 11 | redis_reranker_same_set_diagnostic | 0.00% | 0.00% | False |
| calibration, 11 | always_miss | 0.00% | 0.00% | False |
| calibration, 11 | exact_only | 0.00% | 0.00% | False |
| calibration, 11 | existing_guard_stack | 0.00% | 99.02% | False |
| calibration, 11 | minilm_cosine | 0.00% | null | False |
| calibration, 11 | quora | 0.00% | 0.00% | False |
| calibration, 11 | redis_embedding_cosine | 0.00% | null | False |
| calibration, 11 | redis_reranker | 0.00% | 0.00% | False |
| calibration, 11 | redis_reranker_same_set_diagnostic | 0.00% | 0.00% | False |

All seed comparisons and fixed baseline points are saved, including negative results. Redis embedding thresholds use the common MiniLM candidate pool, not an independently optimized Redis retrieval system. The Redis low-threshold sweep is explicitly a same-set diagnostic. Baseline selectors retain their original candidate guard/ranking behavior. The new policy has additional competition abstention and a top-choice veto without fallback, so these are whole-policy comparisons, not an isolated causal effect of fine-tuning. Historical parsed-call comparator: **UNAVAILABLE / NOT RUN**, as requested.

### Unsafe accepts by requested family (primary seed)

| Split | Direction inversion | Negation | Entity swap | Added requirement | Quantity | Version | Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

These are observed unsafe selected-candidate counts, not independent error-rate estimates. Every unsafe acceptance and every missed positive for ALL selected seed policies is listed in calibration-v1/all-selected-failures.jsonl.gz (full questions, retrieved candidates, probabilities and veto reasons in selected-failures-detailed.jsonl.gz), with actual selected relation, group, lineage, template, family and cause. Baseline errors are likewise saved per split. Per-family conditional cluster bounds and denominators are in per-family.json. Hard-negative suite group/lineage denominators and exact conditional zero-error bounds are separately saved in hard-negative-safety.json; they include all tested negative clusters, not only accepted clusters.

### Abstention and retrieval cost

| Split | Seed | Retrieval-missed positives | Decision-lost retrievable positives | Miss causes |
| --- | --- | --- | --- | --- |
| dev | 11 | 0 | 256 | {"no_calibrated_threshold": 256} |
| dev | 29 | 0 | 256 | {"no_calibrated_threshold": 256} |
| dev | 47 | 0 | 256 | {"no_calibrated_threshold": 256} |
| calibration | 11 | 10 | 1014 | {"no_calibrated_threshold": 1014, "retrieval_miss": 10} |
| calibration | 29 | 10 | 1014 | {"no_calibrated_threshold": 1014, "retrieval_miss": 10} |
| calibration | 47 | 10 | 1014 | {"no_calibrated_threshold": 1014, "retrieval_miss": 10} |

Veto-dev-summary.json reports designated-pair potential rejections per family; dev/classifier-veto-diagnostic.json reports additional rejections and wrongly killed true positives conditional on the fixed 0.5 classifier diagnostic. The veto never resolves a tie or promotes a runner-up. Unverified Stage A metadata is quarantined, reviewed in legacy-signature-review.json, and never used as policy input.

### Seed spread and template memorization

| Seed | Post-fit train four-class accuracy | Dev held-out accuracy | Calibration held-out accuracy | Training wall seconds |
| --- | --- | --- | --- | --- |
| 11 | 100.00% | 46.31% | 45.45% | 197.406 |
| 29 | 100.00% | 73.15% | 69.35% | 222.955 |
| 47 | 100.00% | 63.78% | 56.71% | 212.832 |

Classification-generalization.json preserves all confusion counts and per-held-out-template/per-lineage classification counts. Per-template-lineage.json preserves selected-policy reuse metrics and bounds for every template and lineage, including zero-accept rows. No best seed is substituted: seed 11 was fixed as primary before training. Natural-wording performance: **NOT RUN**, blank request pending.

### Local latency — component measurements only

Hardware: `{"platform": "Linux-6.18.55-x86_64-with-glibc2.39", "python": "3.12.14", "torch": "2.6.0+cpu", "logical_cpus": 9, "torch_threads": 2, "cpu_model": "AMD EPYC 9V74 80-Core Processor", "memory_total": "MemTotal:       10206484 kB", "device": "cpu", "cgroup_cpu_max": "800000 100000", "cgroup_memory_max_bytes": "8589934592"}`.

Cold definition: fresh model objects in a new process; OS disk/page cache not controlled, tokenizer/library imports may be warm; bank embedding precedes first query.

Model loading / bank preparation milliseconds: `{"embedding_load_ms": 70.70969399956084, "classifier_load_ms": 99.06762399987201, "bank_build_ms": 479.99229399920296}`.

| Stage | Warm median ms | Warm p95 ms | Observations |
| --- | --- | --- | --- |
| embedding_ms | 14.098 | 16.820 | 14 dependent local probes |
| search_ms | 0.127 | 0.188 | 14 dependent local probes |
| classifier_ms | 152.295 | 210.276 | 14 dependent local probes |
| veto_ms | 0.864 | 1.080 | 14 dependent local probes |

Calibration scoring had one interrupted process with no traceback; its log and failure metadata are preserved. Recovery added atomic identity-bound numeric checkpoints and bounded concurrent CPU workers, without changing models, data, veto or acceptance logic. Scoring batch times can include worker contention and are not fair application-latency comparisons. The standalone local probe ran separately after scoring.

Full first-forward and warm observations are saved in local-latency-v1.json. Per-model batch/load times for every seed and baseline are in each score receipt; batch throughput is not singleton query latency. No latency percentage, monetary savings or application benefit is claimed.

Validation: `artifacts/stage_b/tests-final.txt` records the targeted harness tests; closeout-integrity.json verifies frozen data, models and decision functions plus the unchanged final-manifest hash. Tests verify implementation invariants, not model accuracy.

## 5. Failures and unknowns

- An always-miss exact-control bookkeeping bug was corrected; original B3 aggregates are retained and every prior semantic metric is identical across all 41 points. See exact-accounting-correction.md and exact-accounting-invariance.json.
- One calibration process stopped without a traceback before raw results were written; the cause is unknown. Its lost scores were recomputed under the frozen contract with resumable checkpoints. See calibration-interruption-v1.json.
- Synthetic data, shared structured task language, builder-authored labels and domain-aligned rules can inflate apparent safety. Whole-template holdout does not establish natural-language robustness.
- The synthetic four-class oracle treats ASCII and Unicode input kinds as incomparable. ASCII is a subset of Unicode, so specificity ordering for that contrast is a known unverified simplification; four-class accuracy is oracle agreement, not independently established relation accuracy.
- Training operations are sequence-focused; held-out operations are string-focused. Lineage holdout is confounded with input-domain shift, so a gap cannot be attributed to template memorization alone.
- Operation matching is a bounded lexical vocabulary, not a general semantic intent parser. Legitimate paraphrases using unsupported operation wording will abstain; useful natural-wording coverage is unmeasured.
- The veto output type is domain-derived, not a general parser for arbitrary requested output representations; generic signature safety remains unproven.
- Few independent operation lineages limit evidence regardless of group count. Conditional group/lineage bounds do not certify a real user workload.
- A threshold with no qualifying calibration bound becomes always-abstain; this is a failed useful-coverage result, not 100% precision.
- Answer correctness/sufficiency is NOT measured. Question equivalence alone cannot make an incorrect cached answer correct.
- The full original 125-accepted-group final-evidence contract remains unresolved; none of these development results replaces final or human review.
- PAWS approval is pending; Stack Exchange duplicate licensing/Python filtering is unresolved. No public data has been used.

## 6. Blockers and minimum owner actions

Return data/intent_stage_b/v2/natural-review-request.jsonl after 50 pairs have been independently written and labeled. Declare human or model-assisted provenance and reviewer independence. Until then, no natural-wording claim. Optional PAWS approval details are in docs/PUBLIC_DATA_APPROVAL_STAGE_B.md; no download occurs without approval. The concrete follow-up data correction is a NEW versioned cohort that mixes sequence, text and conversion domains in training while holding out wording templates and operation lineages separately. Review specificity labels such as ASCII versus Unicode before training. Preserve this spent study and do not tune on the future natural set. No follow-up study or Stage C has been executed; no promotion decision is made for the owner.

## 7. Not done

The locked Stage A final records were NOT opened, trained on, tuned on or evaluated during Stage B. Only its manifest hash was checked; Part 1 performed the separately authorized isolated regeneration/hash test. No final inference or per-case review occurred. No live/provider calls, historical parsed-call comparator, public-pair download, independent human ground truth, UI/agent changes, Stage C integration, semantic serving enablement, remote judge, push, merge, rebase or history rewrite.

## 8. Files to back up and learned model retrieval

Download artifacts/stage_b/stage-b-complete-backup.zip. It includes the local Git bundle, working-tree datasets/manifests/configs/harness/reports, all raw baseline and policy outputs, the excluded Stage A final stores, and all three ignored learned safetensors adapters. BACKUP_FILES.json gives every file size and SHA-256; the ZIP contains restore instructions. No pretrained weights, Python environment, transient inference caches or secrets are included.

The three adapters are at artifacts/stage_b/models/{11,29,47}/adapter.safetensors. All learned tensors are exported; reconstruct with the pinned Quora safetensors base via load_adapter. Exact commands, seeds, base pin, learned/base SHA-256 and hardware are in training-v1.json and model-receipts/. The frozen policies identify each adapter hash. Preserve the archive before environment reset; a commit does not preserve ignored learned exports.
