# Audit of supplied Phase A reports

Reviewed 2026-10-14. Sources: user-supplied `Pasted markdown.md` (FACTS ledger last updated 2026-10-01) and `Pasted markdown (2).md` (Phase A walkthrough dated 2026-10-12).

These are reports, not a source snapshot or event-level evidence. This audit checks reported arithmetic and interpretation; it does not certify execution, split isolation, hash verification, model provenance, or the reported 281-test result. The referenced preregistration commit `d1f9d97` was not available in the previously fetched public repository.

## Findings

| Finding | Evidence and consequence |
| --- | --- |
| Gate A failed | The report gives 21 incorrect non-exact hits among 75 accepted non-exact hits: 28%, with reported exact 95% interval 18.24–39.56%. Both the 10% point ceiling and minimum 100-hit volume fail. |
| Canonical reuse also requires repair | Overall TP/FP are 60/26; non-exact TP/FP are 54/21. Subtraction implies canonical TP/FP are 6/5 among 11 hits. This inference depends on those categories partitioning the same accepted-hit population. Canonical normalization cannot be assumed safe. |
| Seed and probe denominators differ | TP + FP + FN + TN = 1,000, consistent with 1,000 probes and 1,200 cold seeds. The 2,200-event hit rate and 1,000-probe confusion matrix must be labeled separately. |
| Useful hits are lower than accepted hits | Accepted hits: 86/2,200 = 3.91%. Report-labeled correct hits: 60/2,200 = 2.73%. Neither establishes actual cached-answer correctness without answer-level evidence. |
| Zero accepted hits have undefined incorrect-reuse rate | WildChat reports zero hits and 0.0% IRR. With no accepted hits the denominator is zero: report N/A, not a demonstrated zero error rate. |
| Routing counts need event-level reconciliation | CANONICAL_EXACT path count is 30, but reported canonical accepted hits are 11. A path may include rejected or missed lookups; these are not necessarily contradictory, but the remaining 19 events need dispositions. |
| The repeat-share table does not establish p50 | Its numbers follow a linear weighted projection: 8,169.31 − 0.0086 × 8,153.50 ≈ 8,099.19 ms. A weighted latency estimate cannot be labeled a measured median. Quantiles require an event-latency distribution. |
| Savings are projections | Calls/tokens avoided use accepted hits and a baseline token average. They include incorrect reuse and do not establish measured net savings after embedding, verification, retries, and real generation on misses. |
| Small calibration samples cannot certify the ceiling | With only 25 positive examples per domain, even 25 accepted positives and zero errors have a two-sided exact 95% upper bound of approximately 13.72%. This cannot certify a 10% upper-bound gate. |
| The supplied ledger is older than Phase A | The October 1 ledger cannot substantiate October 12 execution. Its earlier 1/12 point estimate below 10% is not statistical certification, and its routing-latency comparisons are not end-to-end service savings. |

## Required evidence before revising Phase A implementation

Obtain the actual source revision used for the run, including git history if available, plus:

- `docs/phaseA_preregistration.md` and the frozen run configuration;
- `data/raw/phaseA_sealed_test_stream.json` and its manifest;
- raw per-event results with candidate identity, source/split lineage, decision path, labels, and timing;
- baseline generation records and summary-generation scripts;
- model/checkpoint identifiers and training/calibration split manifests.

A source archive must exclude secrets, virtual environments, and unnecessary model weights. Do not overwrite the newer repository ledger with the older uploaded ledger. Preserve the spent sealed test as an evaluation result; do not tune on it and relabel it as untouched evidence.

## Engineering consequence

Continue building the scoped actual-response runtime with strict exact matching and explicit eligibility. Keep semantic reuse disabled in that runtime until development-set work and a new frozen evaluation support admission. The fixture tests for the exact runtime are engineering checks, not semantic accuracy or production savings measurements. Gateway, persistence, and observability work can proceed without declaring semantic Gate A passed.
