# Answer-bound benchmark correction — 2026-11-02

Base: `8a272ff483f7dc8d9f97fd633847ea7b86e4d4bf`, branch
`integration/documentation-rebuild`. This is a completed evidence-correction
milestone, not completion of semantic quality or live utility evaluation.

## What was inspected

All four supplied handoff Markdown files were read. GitHub branch metadata and a
fresh single-branch clone confirmed the base. The linked raw endpoint was too
large for the connector fetch; its exact committed file was instead read from
the checkout. All 1,100 event records (220 ordered events per arm) were reconciled.
The spent workload/results and sealed final data were not changed or rerun.

The historical trace contains no returned-response payloads, but accepted semantic
hits carry full candidate-response SHA-256 bindings. Reconstructing the original
deterministic ReferenceProvider outputs recovered all 129 semantic-hit payloads
with matching full-response hashes: dense 79, combined 38, answer-aware 12.
Recovery verifies identity, not correctness. It uses the committed fixture code,
not a reconstructed live LLM response. The audit receipt includes input hashes.

## Actual returned-answer review

`answer_aware_model_reviews_v2.json` declares source-reading ChatGPT model-assisted
review, not independent human ground truth. Every one of the 12 answer-aware hits
was read in full. The rubric requires addressing the incoming request without a
materially false behavioral claim. It was explicitly recorded with each label.
These are retrospective development annotations; no fresh evaluation is claimed.

11 answers are sufficient under that rubric. The reverse-iteration answer gives
correct usable loop syntax but falsely says `reversed(sequence)` builds a new
copy. A local Python check constructs a reverse iterator, mutates the last element,
and observes the changed element through the iterator. The answer is marked
insufficient for whole-answer correctness, with the specific rationale preserved.
All 12 answers address their request topic; this one correctness failure is not
evidence of a wrong-topic cache match. The same erroneous reference is returned
by the scripted fresh provider, so it does not establish cache-induced degradation.
Source-bound admission and exact reuse also do not establish factual correctness.

For completeness, the descriptive independent-binomial Clopper–Pearson interval
for 11/12 successes is [0.6152038348, 0.9978924068] (two-sided 95%). The records are
selected, template-generated, exposed development cases; this interval is NOT a
valid generalization guarantee or release gate. The 12 source groups do not make
this a representative independent sample. No population accuracy claim follows.

The same exact request/response/scope bindings can be reviewed across arms. Dense
retains 67 unknowns and combined 26; neither receives an overall precision claim.
The historical family-derived TP/FP values are preserved in the old raw file, but
are not used as correctness labels in the new audit or corrected benchmark.

## Directly observed offline work and latency

New replay: same 220 ordered existing development events in independently fresh
no-cache and exact-only caches, 20 ms scripted delay, same fixture settings.
Timing surrounds `GatewayService.complete`, including cache work and fallback;
it excludes HTTP, agent orchestration and initial model construction. Quantiles
are nearest-rank. Hardware/runtime differs from the historical Windows run; do
not compare new Linux timings against historical semantic-arm timings for a
speedup claim.

| New offline arm | Completed | Counted scripted attempts | Failures | Gateway p50 ms | Gateway p95 ms |
|---|---:|---:|---:|---:|---:|
| No cache | 220 | 220 | 0 | 20.386301 | 20.636422 |
| Exact only | 220 | 198 | 0 | 20.442820 | 20.614063 |

Exact-hit path: 22 observations, p50 0.160151 ms / p95 0.299983 ms.
No API traffic occurred. These are fixture timings, not HTTP/whole-agent/live LLM
end-to-end timings. Historical answer-aware calls remain inferred: 186 versus
198 exact-only, because the old trace did not directly log invocation boundaries.
Historical quantiles are also recomputed with a declared nearest-rank method;
small differences from old summaries follow from that method, not edited data.

The corrected provider wrapper logs every invocation, elapsed time, error and
available usage at its boundary, including failed invocations. Cached response
usage is never counted as fresh work. ReferenceProvider whitespace counts are
explicitly synthetic. Billable prompt/completion tokens, monetary cost and live
savings are null. A dated rate table and authorized real usage are absent.
Existing production WorkLedger accounting of planning/generation/review remains
unchanged; this gateway replay does not measure the whole agent's efficiency.

## Implementation and integration result

- Family-derived correctness removed; reviews bind actual request, returned full
  response, tenant and source revision. Missing or changed bindings stay unknown.
- Zero-hit/zero-reviewed precision is null, not 100%. Unknown coverage is explicit.
- Selected candidate request/response, decision, existing binding and diagnosis
  are exported in newly executed semantic traces without changing policy behavior.
- Output files use exclusive creation. Prior receipts cannot be overwritten.
- The MS MARCO implementation is now accurately documented. Its relevance score
  is not a correctness probability. Defaults, models and thresholds are unchanged.
- Existing gateway integration passes fixture tests. Policy construction is found
  in benchmark/tests, not a production factory or agent path. Real-model serving
  and whole-agent integration remain unverified.
- Neural rerun was not attempted: torch/SentenceTransformers are absent in this
  environment. No model download, training, new provider call or paid API was used.
- Human approval, tools, role routing, memory, persistence and invalidation remain
  intact. General semantic serving is not promoted.

## Reproduction

Use new output filenames; these commands intentionally reject existing outputs.

```bash
python scripts/audit_phase3_trace.py --raw artifacts/documentation_rebuild/semantic_p0/phase3_benchmark_results.json --workload artifacts/documentation_rebuild/semantic_p0/phase3_dev_workload.json --reviews artifacts/documentation_rebuild/semantic_p0/answer_aware_model_reviews_v2.json --output /tmp/phase3-reviewed-NEW.json
python scripts/phase3_development_benchmark.py --workload artifacts/documentation_rebuild/semantic_p0/phase3_dev_workload.json --arms A_no_cache B_exact_cache --output /tmp/phase3-exact-NEW.json
python -m pytest -q tests/test_answer_evidence.py tests/test_answer_aware_policy.py tests/test_response_cache.py tests/test_sqlite_response_cache.py tests/test_semantic_runtime.py tests/test_durable_semantic_rebuild.py tests/test_before_agent_cache.py tests/test_document_orchestration_memory.py tests/test_document_agent.py tests/test_agent_workflow.py tests/test_static_document_stage.py
```

Validation: 250 passed, one Starlette/httpx deprecation warning, 3.82s.
Secret-pattern scan covered all 15 changed/new text files before the environment
receipt, with no key/token/private-key-header matches. This is a scoped pattern
scan, not a proof of absence. Runtime versions are in answer_evidence_environment_v2.json.

Validation receipt: `artifacts/documentation_rebuild/semantic_p0/answer_evidence_tests_v2.txt`.
Next evidence needs independent answer review beyond these model-assisted labels,
local pinned-model availability for a matched neural replay, and explicit live
provider authorization before any live token/cost/latency evaluation. No positive
live efficiency result can be inferred from the completed correction.
