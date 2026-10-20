# Answer-aware local policy development

October 21, 2026. Development work, not a fresh final evaluation. The final
quality contract and historical failed gates remain unchanged.

## Policy and bounded experiment

The old question-pair model is not an answer-applicability verifier. The new
diagnostic policy supplies the actual complete cached answer, original and new
questions, and trusted source/system context to two local checks: whether the
answer fully satisfies the new question, and whether its factual claims are
supported. It rechecks request/context boundaries and complete-response admission.
It refuses oversized input rather than truncating. There is no remote fallback.

Two explicitly pinned off-the-shelf baselines are investigated: FLAN-T5-base
and Qwen3-0.6B with thinking disabled. Preparation downloads public artifacts;
model construction is local-only and disallows remote custom code. CPU forward
passes yield yes/no token scores. Qwen aggregates case/space variants of each
label; FLAN uses its single lowercase labels. Neither output is a calibrated
correctness probability and the two scoring adapters are not a controlled
model-only comparison.

The initial diagnostic rule requires BOTH checks to have a conditional yes
score at least 0.98 AND absolute yes mass at least 0.90. Recording full-vocabulary
mass prevents a tiny yes score winning merely because no is even less likely.
These fixed initial development thresholds are not a calibrated serving rule.
They are not lowered to rescue the exposed development examples.

All fifteen cases are authored development material, immediately exposed to
the builder. Four positives and eleven negatives span three correlated intent
groups, not fifteen independent observations. Wrong, unsupported, irrelevant and
incomplete actual answers are included alongside changed requests. Expectations
are authored fixture assertions, not independent human or model review labels.
No population precision or safety confidence bound is derived from these cases.
Do not relabel them to agree with the model.

The FLAN run abstained on every case, including all four authored positives.
That is an unusable-coverage finding, not a passed safety evaluation. The second
baseline tests whether stronger local instruction following improves separation;
its result, including any unsafe recommendations, is retained alongside FLAN.
This milestone does not search further models until a desired result appears.

The Qwen run recommended reuse for all four positives but also eight of eleven
authored negatives. It approved wrong and irrelevant actual answers and changed
units/direction cases. This baseline fails the development usefulness/safety
check and is not eligible for serving or calibration into a claimed safe policy.
Do not present the four positive recommendations as successful cache hits.
The blocking diagnostic interface prevented all recommendations from serving.

## Serving and accounting

`DiagnosticAnswerPolicy.assess` returns recommendations with answer/request hashes,
per-check scores and local work. Its runtime approval interface ALWAYS returns
false, even for positive recommendations. It has no serving-enable argument.
It cannot silently activate semantic reuse in the gateway. Existing authored
answer approvals and all default modes remain unchanged.

Local verifier attempts and elapsed policy milliseconds are separate from remote
verifier calls. Attempts count check invocations, including validation/inference
failures; they are not necessarily completed model forward passes. The gateway
exports separate bounded-cardinality local-work counters. Successful exact hits
skip both local checks and remote work. The integration probe uses real dense
retrieval, but its provider answers are scripted development fixtures.
Gateway semantic-stage counters cover returned requests; a later upstream
exception can prevent the stage diagnosis from reaching the service counters.
These counters are not a failure-complete end-to-end work ledger. The local
development assessment itself records failed local checks as attempted work.

Actual safe semantic serving remains a subsequent milestone: independent
development red-teaming, calibration on separate groups, source/answer provenance,
persistence and evaluation must support a versioned enabling decision. If this
experiment cannot separate safe from unsafe reuse, retain abstention and report
the limitation rather than ship a permissive threshold.

## Reproduce

```bash
python scripts/probe_answer_applicability.py --prepare --scorer flan --output runs/semantic-dev/flan.json
python scripts/probe_answer_applicability.py --prepare --scorer qwen --output runs/semantic-dev/qwen.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/probe_answer_applicability.py --scorer qwen --output runs/semantic-dev/qwen-offline.json
python -m pytest tests/test_answer_applicability.py tests/test_semantic_runtime.py tests/test_semantic_preparation.py -q
```

After preparation, the runner blocks Python socket connections for inference and
the cache integration probe. No generation key or external generation attempt is
needed. Local weights can be lost when the hosted workspace is recreated; the
published revisions and explicit preparation command restore them. Record model
artifact hashes and library versions with each milestone's aggregate evidence.

Primary model/API sources:
https://huggingface.co/google/flan-t5-base
https://huggingface.co/Qwen/Qwen3-0.6B
https://huggingface.co/docs/transformers/model_doc/t5
