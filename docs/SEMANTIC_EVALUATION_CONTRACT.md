# Semantic answer-reuse development and evaluation contract

Adopted work direction: October 20, 2026. This is a prospective development
contract, **not a frozen final evaluation or a passed gate**. Historical failed
gates, spent datasets and live attempt budgets remain unchanged.

## Product contract

Project 7 semantic answer caching is the foundation; the bounded Project 15
agent is a client. For eligible fixed-context requests, lookup precedes agent
planning: scoped exact lookup, local dense retrieval, local actual-answer
applicability check, then reuse or abstention. A successful hit makes zero
remote generation, planning, reviewer or judge calls. Count local inference and
lookup overhead. Remote generation remains available on misses under an explicit
budget; no live calls are authorized by this preparation work.

Full tenant/access, system/source revision, model and generation-parameter
boundaries remain binding. Do not weaken scope to manufacture hits. A question
duplicate score is only one possible development feature: it does not verify
that an actual cached answer satisfies the incoming question. No similarity-only
serving policy, pair allowlist presented as generalization, or hidden remote judge.

## Evidence targets and distinct denominators

The selected target is the tolerant group gate. Include **all** accepted groups
in the fixed final cohort, never a favorable subset or just the first 125.

| Endpoint | Denominator and proposed pass condition |
| --- | --- |
| Group safety | At least 125 independently constructed intent groups with one or more accepted non-exact reuses; one-sided 95% exact lower bound on group safety at least 95% |
| Group outcome | A group succeeds only if every accepted reuse in that group is applicable; one incorrect accepted answer fails the group |
| Actual-answer precision | Correct, sufficient, supported accepted non-exact answers / all accepted non-exact answers; observed precision at least 98%, with uncertainty and grouping reported separately |
| Positive coverage | Accepted non-exact positive queries / independently labeled eligible positive queries; at least 30%; exact repeats excluded |
| Hard negatives | Zero accepted reuses of the predeclared unsafe candidate answers across a separate fixed suite; count query and group denominators and report a one-sided 95% false-acceptance upper bound |
| Hit-path work | Zero actual remote calls on each successful hit; verify all stages, including planning and review |

At exactly 125 accepted groups, two failures give a 95.0492% one-sided lower
bound; three fail the group gate. These are hypothetical arithmetic, not observed
quality. For larger accepted samples the exact bound determines the allowed
errors; the answer-level precision gate is still separate.

Binomial bounds require defensible independence and a sampling interpretation.
Multiple paraphrases are not independent trials. Shared source/topic dependence
must be disclosed and grouped; report answer precision with an intent-group
bootstrap when appropriate. Purposive stress-suite bounds describe the sampled
suite under stated assumptions, not a guaranteed production error rate.

No missing actual-answer labels are imputed correct. Any unreviewed accepted
reuse makes the independently reviewed quality gate inconclusive. Always report
the observed errors and missing counts even when the gate cannot be assessed.

## Cohort sizing and review capacity

Positive *group* acceptance and positive *query* coverage are different rates.
For planning, assume two positive paraphrases per group, but estimate group
acceptance using development/calibration data before freezing the final cohort.
Several paraphrases cannot be counted as distinct intent groups.

With an unmeasured 30% group acceptance assumption, 417 positive groups gives an
expected 125 accepted groups. Under an additional independent common-probability
assumption, 470 positive groups gives at least a 95% probability of obtaining
125 accepted groups. Neither is a guarantee. At 470 groups and two paraphrases,
there are 940 positive queries requiring eligibility/reference labels, before
reviewing actual accepted answers or adding hard negatives. This workload does
not fit a casual one-day review session.

The hard-negative design target is 150 distinct groups, two variants per group,
with 25 groups each for direction reversal, polarity, added constraints, changed
entities, units and versions. At zero group false acceptances, the one-sided
95% upper bound is 1.9773%, conditional on the stated sampling assumptions.
Each negative must specify an actual candidate answer that is unsafe for the
incoming question; a textual difference alone is not a negative label.

**Current reviewer status: no independent human reviewers are confirmed.**
Do not invent availability, a review booking or an independent final dataset.
Until reviewer capacity and sufficient distinct fresh intents exist, these are
sizing targets, not a locked final cohort or a promise of completion in ten days.
The default fallback is a clearly labeled model-assisted diagnostic study with
human validation and the final quality gate explicitly unestablished.

Do not start final evaluation until a separate freeze records:

- fixed cohort size, group/variant IDs, construction method and source licenses;
- immutable seed questions, complete answers, system/source context and hashes;
- eligibility/reference labels produced independently of policy construction;
- pinned model artifacts, policy revision, thresholds and abstention rules;
- reviewer assignments, blinding, adjudication and missing-label rules;
- workload/repeat mix, arm order, sample sizes and stopping rules.

Keep development, calibration and final groups separate, including related
topics and paraphrases. Previously exposed reserved data is development data,
not sealed. Never add groups after seeing a final failure to obtain a pass.
If the fixed cohort yields too few accepted groups, report insufficient evidence.

## Written actual-answer rubric

Review the **new question and complete cached answer**, with the original
question, governing system instructions and fixed source/version context.
Mask arm, model scores, thresholds and serving decision wherever feasible.
All judgments bind to immutable incoming/candidate request and answer hashes.

For each item record separately: eligibility, factual correctness, source
support, sufficiency, applicability, ambiguity and a short justification.
Allow `true`, `false` and `null` (insufficient information); never silently turn
`null` into success.

An answer is applicable only when it is correct and supported in the governing
context and fully addresses the incoming request's material requirements:

- Preserve conversion direction, polarity, entities, quantities, units and
  requested software/version behavior.
- Honor new constraints, requested output and material prerequisites. A generic
  answer that omits a newly required recursive conversion is insufficient.
- Do not demand identical wording or an identical method when the cached answer
  already supplies a valid, sufficient solution.
- Additional accurate context is acceptable unless it contradicts instructions,
  confuses the result or violates a stated constraint.
- Related subject matter, high retrieval similarity and duplicate-question
  scores do not establish applicability.
- An inaccurate original answer remains inaccurate when reused. Correctness
  against a reference and transfer applicability are reported separately.
- If a question or source is ambiguous, mark ambiguity and the appropriate
  unknown labels. Any exclusion must follow the frozen rule and be reported.

Two independent human reviews of accepted answers, followed by recorded human
adjudication of disagreements, are the preferred final process. If this cannot
be staffed, state the limitation before collection and do not imply that model
consensus replaces independent human evidence. Never rewrite historical labels.

## Work split and order

Codex builds retrieval, the applicability policy, calibration/abstention,
persistence, pre-agent integration, full work accounting and reproducible runners.
Codex does not author final correctness labels for its own policy. Antigravity
can independently author development adversarial cases without seeing policy
internals, red-team development, check the no-remote-call path and test a fresh
installation. Its labels remain model-assisted, including any optional final
diagnostic labels. Human reviewers own final human judgments. Claude can audit
a user-approved published milestone against its exact commit; do not send
messages or data to another service without the user's instruction.

Execution order: verify weights and rubric now; demonstrate real local retrieval;
build and calibrate a generalizing answer-aware policy; integrate before agent
planning; freeze only after data/review capacity is ready; evaluate once; finish
browser/install/demo packaging. The live comparison is the first item dropped
if provider reliability or budget is insufficient. A working product may ship
with quality evidence explicitly inconclusive.

The user authorizes Codex to commit and push substantial validated milestones on
the task branch, not each small step. No main push or merge. Never publish raw
private answers, review mappings or user data as part of aggregate evidence.

## Reproduction and model fallback

The missing historical fine-tuned checkpoint is not a prerequisite. Explicitly
prepare the existing pinned off-the-shelf MiniLM and Quora pair model, record
weight SHA-256 and local library versions, then run inference with offline flags.
Quora duplicate scoring is a diagnostic baseline, not a validated applicability
policy; poor results require development, not a lower threshold chosen to pass.

```bash
python scripts/run_semantic_probe.py
python scripts/run_semantic_probe.py --prepare --probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_semantic_probe.py --probe --output runs/semantic-dev/offline-probe.json
python scripts/plan_semantic_review.py --accepted-groups 125 --group-acceptance-rate .30
```

`--prepare` permits public artifact downloads only; no generation-provider call.
The default is inventory only. Probe examples are exposed authored development
material, not fresh final evaluation or independent quality observations.

Primary API reference: https://www.sbert.net/docs/package_reference/cross_encoder/model.html
and exact confidence interval reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
