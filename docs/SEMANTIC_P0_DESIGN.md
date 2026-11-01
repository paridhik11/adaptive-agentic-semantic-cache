# Implemented semantic P0 design — corrected 2026-11-02

This describes `8a272ff` source, replacing an inaccurate FLAN-T5 design description.
No serving promotion or policy-threshold change is made by the evidence correction.

`GuardedSemanticCache` checks scoped exact lookup first. On an eligible miss it
embeds the incoming request and ranks stored candidates. The runtime selects the
top candidate, subject to its retrieval threshold and margin. It does not iterate
through every candidate with a verifier. Tenant, context and source scope remain
part of the cache boundary.

`AnswerAwareSemanticPolicy` checks the two-message pinned context, equal system
prompts and response admissibility, then applies `requirement_guard`. If the guard
passes, `LocalPassageScorer` scores the incoming question against the actual cached
answer text. It runs `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, locally on CPU, safetensors only,
without remote code. Over-window input abstains; there is no silent truncation.

The model supplies a relevance logit and sigmoid score. This is not calibrated
answer correctness, entailment or contradiction probability. Threshold remains
0.85. There is no separate neural contradiction check in this implementation.
`enabled=False` is the policy default. The development benchmark explicitly uses
`enabled=True` and retrieval threshold 0.60/margin 0.0; it is an experimental arm.
No general production safety has been established.

The optional promoter callback runs synchronously inside `__call__`; it is not an
implemented asynchronous promotion queue or a demonstrated SQLite admission path.
No production factory or bounded-agent code currently constructs this policy:
construction appears in the benchmark and tests. Existing gateway integration is
fixture-tested; real-model agent integration remains unverified. Existing bounded
agent, source tools, reviewer, memory, approvals and durable lifecycle are preserved.

The corrected benchmark exports actual request/response/scope bindings, selected
candidate assessments and provider-boundary invocation records. Review labels
remain unknown unless a declaration matches the returned response hash, request
hash, tenant and source revision. Family labels never establish correctness.
Shadow/disabled semantic serving remains appropriate pending independent fresh
answer-level evaluation; this correction invents no new release gate.
