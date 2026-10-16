# Answer-bound development evaluation

The recovery benchmark evaluates complete returned answers and actual provider
work. It is separate from the historical Phase A pair/stream harnesses. It does
not change the failed safety gate or label a development run as sealed evidence.

## Run without an API key

```bash
python scripts/benchmark_answers.py run \
  --workload examples/benchmark/workload.fixture.json \
  --output-dir runs/fixture-demo
```

Choose a new directory each time. Open `review.html` locally. It contains the
actual requests, served answers, and fresh baseline answers. Select correctness
and reuse labels, add references/reasoning, and download the review JSON. A fresh
model answer is a comparator, not ground truth. Do not mark an echoed fixture
answer as correct simply because it matches another fixture answer.

```bash
python scripts/benchmark_answers.py report \
  --trace runs/fixture-demo/trace.json \
  --reviews answer-reviews.json \
  --output runs/fixture-demo/report.reviewed.json
```

Outputs include a trace, blank review template, unreviewed report, offline review
page, and content-hash manifest. Existing evidence is not overwritten. These
files contain request/answer content; `runs/` is excluded from Git. Review UI
labels remain only in page memory until downloaded. Use access-controlled local
storage for private workloads.

## Prepare a larger development stream

```bash
python scripts/prepare_answer_workload.py \
  --pairs data/candidate_b_train_pairs.json \
  --output runs/workload.dev.json \
  --license-note "Existing local development data; verify upstream terms before redistribution"
python scripts/benchmark_answers.py run \
  --workload runs/workload.dev.json --output-dir runs/dev-fixture
```

Create the `runs` parent directory first. The converter preserves source labels,
input content hash, and original pair indices. It builds connected components
of shared normalized query text for grouping. All outputs remain development
only: these queries were already exposed in training. The query-a/query-b order
is constructed and can inflate reuse relative to organic traffic. Pair labels
are retained as provenance and **never copied into answer correctness labels**.
No thread-level or semantic deduplication is claimed. Group IDs must be globally
consistent across sources. Supplied workloads reject group or normalized-request
leakage across train/dev/test splits, but this cannot detect unrecorded exposure.

The bundled development data covers QQP and PAWS, not every target workload.
A credible future evaluation also needs technical/domain queries, contextual and
time-sensitive questions, adversarial near-matches, and genuine repeated-traffic
samples with explicit provenance and source terms. New independent test data
must be frozen only after development decisions are complete.

## Live development run

For the current zero-budget project, use the [free-only pilot](free_pilot.md).
The generic HTTP runner below does not enforce OpenRouter free-model quotas
or zero-price routing.

The fixture runner uses no external model. Live runs are explicit opt-in; set
`UPSTREAM_CHAT_URL` and optionally `UPSTREAM_API_KEY` in the environment, use the
provider's model name in the workload, then choose `--provider http`,
`--allow-live-calls`, and `--max-live-calls` covering twice the selected request
count. Each request has one fresh baseline call and at most one cache-arm call.
The runner validates all requests before starting calls, requires an output-token
limit, and does not retry failures. The call budget bounds attempts, not currency
or upstream billing. No live calls were used for the committed fixture evidence.

The runner currently refuses test-split execution. A separately pre-registered
protocol and freeze/checksum workflow are still needed before a new sealed test.

## What the report means

- Each review binds the trace hash, event ID, request hash, actual served-answer
  hash, fresh-answer hash and retrieved candidate event ID. Missing, duplicated,
  stale, wrong-answer and wrong-candidate labels cannot silently alter the score.
- A hit must reference a previous generated response from the same tenant. The
  returned answer hash must match that candidate. Exact/non-exact partitioning
  compares complete request hashes, not normalized text or designated pairs.
- Accepted and reviewed denominators are separate. Unreviewed hits are unknown;
  their best/worst-case error bounds are shown. Zero hits produce null rates.
- Reviewed incorrect-reuse rates include two-sided Clopper–Pearson intervals.
  They assume independent trials. Group bootstrap intervals resample complete
  query groups; few groups or all-identical outcomes remain weak evidence.
- Quality compares paired served/fresh correctness labels. Latency uses raw
  complete-answer samples and reports empirical p50/p95/p99, plus a paired mean
  difference with exploratory group bootstrap. It never averages percentiles.
- Timing includes lookup, policy, generation and admission inside the runtime,
  but excludes HTTP gateway transport. Arms run serially in seeded randomized
  order per request. It is not a concurrency/load benchmark.
- Tokens count only newly executed calls. Cached response usage is not counted
  as fresh work. Failed attempts and missing usage remain visible; token savings
  become null when either arm has unreported work. No monetary saving is inferred.
- Recall is omitted because no independently labeled opportunity denominator
  exists. Safety certification remains false; semantic serving is still disabled.

## Remaining measurement work

Use independently reviewed real answers and references, multiple domains, enough
accepted non-exact hits, grouped uncertainty analysis, predefined quality margins,
and new independent held-out groups. Compare no-cache, exact, similarity-only,
verified semantic and adaptive policies under one complete-answer protocol.
The runner implements no-cache, exact, and experimental similarity/verified arms;
none of the semantic policies are certified. Adaptive policy learning remains
unimplemented. Reviewer identity/method are recorded, not
authenticated; label files require review before scientific claims.

## Experimental semantic policies and independent review

The benchmark now adds similarity-only and answer-verified retrieval with
separate verifier work accounting. Two human review files can be reconciled
without silently resolving disagreements. See [the implementation guide](semantic_streaming_agents.md).
Semantic serving remains disabled and no new safety gate has passed.
