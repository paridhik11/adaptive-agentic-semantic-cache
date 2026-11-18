# Decomposed local verifier: development result

The verifier remains unsuitable for general serving: it accepts only 2 of 14
valid authored paraphrases. It made no unsafe recommendations on 31 authored
negative cases. Neither observation estimates population precision or safety.

## What changed

Pinned local DeBERTa NLI checks bidirectional request entailment, actual-answer
coverage of a selected source span, and source support for each answer claim.
Generic guards reject explicit direction, quantity and added-constraint changes.
Evidence selection uses the original question only. Dense and NLI inputs that
exceed their model windows abstain instead of silently truncating.

The application policy always rejects serving, including positive diagnostics.
Only the isolated socket-blocked fixture runner turns recommendations into cache
approvals. There is no production enable flag and no approved query-pair list.

## Recorded outcomes

| Authored cohort | Positive recommendations | Unsafe negative recommendations |
| --- | --- | --- |
| Existing 15 cases, 3 intent groups | 0/4 | 0/11 |
| Additional 30 cases, 10 intent groups | 2/10 | 0/20 |

The accepted paraphrases concern integer-to-string conversion and JSON parsing.
Across all 45 actual cache experiments: 2 semantic hits, 0 unsafe semantic hits,
1 exact hit, and 87 scripted provider invocations including seeding. The exact
hit reused a deliberately incomplete/wrong seed: existing complete-response
admission checks response structure, not factual sufficiency. This is retained
as a failure boundary, not counted as a successful semantic decision.

Source indexing required 20 initial embedding calls per run. Detailed results
include per-request local embeddings, NLI attempts and elapsed policy time.
These are local diagnostic timings; no paired application latency, monetary
savings or live answer-quality claim follows. Python socket connections were
blocked during inference. External generation calls and human reviews: zero.

## Why coverage failed and what follows

Request descriptions are not reliable NLI hypotheses for all paraphrases. Some
source spans describe a broader fact than the answer needs to cover; requiring
the answer to entail that entire span rejects valid answers. Merely lowering
the threshold would hide these failures. Next development should address task
representation and question-relative sufficiency, using separate development
and calibration groups before a frozen evaluation. Persistent semantic metadata
and pre-agent integration remain later milestones. Independent reviewers and
final cohort sufficiency are still unresolved.

The initial direct run and expanded cache replay are preserved together in
[raw evidence](DECOMPOSED_POLICY_RESULT.json). The second run kept all cases and
thresholds unchanged; it expanded engineering integration coverage only. Both
are spent, builder-authored development data. Earlier FLAN/Qwen failures remain
unchanged. The final evaluation contract and failed historical gate are unchanged.

## Validation

Full offline suite: 744 passed, 23 deselected in 126.00 seconds. Focused suite:
80 passed. These verify engineering behavior, not model accuracy. Both recorded
runs have identical case hashes, thresholds, recommendations and cohort counts.
