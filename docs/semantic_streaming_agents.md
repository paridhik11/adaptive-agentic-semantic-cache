# Semantic experiments, streaming and agent integration

## Current boundary

Semantic retrieval is implemented in the development benchmark, not enabled in
the serving gateway. Gateway reuse remains exact and eligibility-gated. The
experimental cache is serial, in-memory and scoped by tenant, all generation
parameters and the complete preceding message context. Only the final user text
can vary during retrieval. It has TTL, entry/payload limits and LRU eviction;
request/vector memory is additionally bounded by entry count, not payload bytes.
It is not a concurrency-ready replacement for the gateway cache.

## Semantic policy comparisons

```bash
python scripts/benchmark_answers.py run --workload runs/workload.dev.json \
  --policy similarity --encoder lexical --threshold .85 \
  --output-dir runs/similarity-dev
python scripts/benchmark_answers.py run --workload runs/workload.dev.json \
  --policy verified --encoder lexical --threshold .85 \
  --output-dir runs/verified-fixture-dev
```

`similarity` is an intentionally unsafe ablation. The lexical encoder uses
unigram/bigram hashing and cosine similarity; it is not a learned semantic model.
The `dense` encoder loads an explicitly pinned, already-local SentenceTransformer
snapshot with `--embedding-model` and a full `--embedding-revision` commit hash.
It never downloads model weights implicitly. A separate marked integration test
can download the existing repository-pinned model. A missing local snapshot is
an explicit setup failure, not a silent lexical fallback.

Both policies apply the conservative eligibility gate before lookup/admission,
require the configured top-score threshold and runner-up margin, and preserve
actual candidate-event provenance. The threshold/margin are development settings,
not calibrated or certified values. Similarity alone cannot establish answer
correctness or request equivalence.

The verified policy supplies the incoming request, candidate request and actual
cached answer to an explicitly configured `AnswerVerifier`. It accepts only a
complete, strictly parsed boolean JSON verdict; uncertainty, malformed output,
exceptions and rejection all generate a fresh answer. Policy-judge outputs are
never converted into evaluation labels. No adaptive threshold learning is added
until simpler policies have credible evaluation evidence.

With a fixture provider, verification deliberately abstains on every non-exact
candidate. That exercises failure behavior, not verifier intelligence. With
`--provider http`, use `--verifier-model`, explicit live-call opt-in and a budget
covering up to three calls per request (fresh baseline, verifier, generation).
The current CLI uses the configured generation endpoint for the policy verifier;
independent human evaluation remains separate. Judge attempts, missing usage,
errors and token counts are included in net cache work. A generation failure
after verification does not erase already-spent verifier work.

The report audits generation/context scope as well as answer hashes. Non-exact
hits under a verified policy require an affirmative recorded verdict. This is
trace consistency, not proof that the verdict was correct.

## Independent answer review

Have two reviewers separately fill review templates bound to the same trace.
Review actual answers using references where appropriate. Reviewer identities
are declared identifiers, not authenticated identities; separate names alone do
not prove independence. Avoid showing one reviewer's decisions to the other.

```bash
python scripts/reconcile_answer_reviews.py --trace runs/live-dev/trace.json \
  --first reviewer-a.json --second reviewer-b.json \
  --output runs/live-dev/agreement.json \
  --consensus-output runs/live-dev/consensus.json
python scripts/benchmark_answers.py report --trace runs/live-dev/trace.json \
  --reviews runs/live-dev/consensus.json --output runs/live-dev/reviewed-report.json
```

The reconciliation validates every binding, requires distinct declared human
reviewers, reports overlap/agreement/Cohen's kappa and exports explicit conflicts.
Only matching double-reviewed labels become consensus. Disagreements and
single-review labels remain unknown. Adjudication needs an explicit new review;
this tool never manufactures a tie-breaking judgment. Kappa is undefined when
expected agreement is one. Reviews are not blinded by the existing UI; formal
studies should control that source of bias.

No live-model endpoint or credentials were available during this implementation.
No independent humans have reviewed these new fixture outputs. Live-answer
quality, net real-model savings and a new sealed safety gate remain unestablished.

## Streaming gateway

The existing authenticated `/v1/chat/completions` endpoint now accepts
`"stream": true` and optional `"stream_options": {"include_usage": true}`.
Streaming always bypasses both cache reads and admission. The server requests
terminal usage from the provider. The HTTP adapter requires an SSE response and
forwards validated text-completion events; tools/multiple choices remain unsupported.

Limits: the configured response-byte limit, a bounded event size, a stream
read-duration limit, and the configured upstream socket timeout. The duration
limit is checked between bounded chunk reads; a blocked read may overshoot it
by a socket-timeout interval. Downstream backpressure can also delay completion.
A terminal stop event with text followed by `[DONE]` is required for success.
Truncation, malformed JSON, missing completion, refusal and tool deltas produce
a sanitized SSE error without a success marker. Failures after HTTP headers
cannot change the HTTP status; clients must inspect terminal stream events.

Processing slots are held through streaming and released on completion,
disconnection or error. Cancellation during connection opening schedules cleanup
of the eventual source; response cleanup is shielded from cancellation. No
implicit upstream retry occurs. Closing a blocking socket can wait for its
configured timeout; this is not instantaneous cancellation or a throughput SLA.

Metrics distinguish completion, cancellation, failure, missing usage and cleanup
errors. The first-text sum/count measures time until the service sees its first
text delta, not client-observed time to first token. It does not supply p95/p99
first-token latency. Cached response usage cannot be counted because stream
requests never read the cache.

## Bounded agent workflow

```bash
python scripts/demo_agent_workflow.py --output runs/agent-fixture.json
```

`AgentWorkflow` routes deterministically to a coding or explanation specialist,
then invokes a reviewer. This is a small explicit supervisor/specialist/reviewer
integration, not a claim of learned routing or a general autonomous agent system.
The caller supplies tenant identity and bounded user/assistant conversation context;
workflow version, specialist instructions and context are in the full cache key.
The reviewer always bypasses caching. A rejected review returns
`NEEDS_HUMAN_REVIEW` with a draft and no released answer. Malformed review or
provider failure returns `FAILED`; there are no hidden retries. Failures record
unknown actual provider-call counts and a conservative charged budget rather
than inventing exact work counts.

The demo uses an explicitly permissive fixture eligibility policy and scripted
approval to test orchestration and isolation. It demonstrates repeated specialist
reuse while rerunning review and separating tenants. Production's conservative
policy currently bypasses contextual specialist prompts. To use real generation,
use the demo’s `--provider http --model MODEL --allow-live-calls --max-live-calls 6`
options with the upstream environment configuration, or construct the workflow
with a `GatewayService` and `HTTPChatProvider`. The live demo uses the conservative
policy; credentials belong to the service configuration. There is no model-controlled shell, arbitrary
URL tool, external write capability or automatic human approval. Caller-supplied
context is the memory interface; persistent agent session management is not
implemented. Workflow review is not independent evaluation ground truth.
