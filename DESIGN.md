# Cache gateway and bounded documentation-agent design

Current status: [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md).
Evidence: [docs/FACTS.md](docs/FACTS.md) and phase audit artifacts.
This document describes implemented contracts; quality and utility gates remain open.

## Serving architecture

GatewayService validates requests and invokes the existing scoped response cache.
The exact key binds the complete generation request and trusted tenant/context.
Only eligible, complete admissible responses are stored. TTL, bounded payloads,
SQLite persistence and invalidation are explicit. Identical cold misses are
coordinated within a process, not across a distributed deployment.

GuardedSemanticCache wraps that cache. Modes are off, exact, shadow and guarded.
Exact lookup is first. Semantic candidates are constrained by trusted knowledge,
source revision, access and policy scope. Shadow mode records candidates without
semantic serving. Guarded mode checks retrieval scores and explicit approvals
bound to the incoming request, candidate request, actual response and scope.
The current approval registry is an allowlist, not a learned equivalence model.
Similarity or a classifier's STABLE label alone cannot certify answer reuse.

Complete exact responses may persist in SQLite. The semantic candidate index is
process-local and cold after restart. Streaming bypasses caching. Bounds describe
configured payload/storage limits, not measured total resident memory or throughput.

## Agent architecture

DocumentAgent is a client of GatewayService. A bounded JSON plan selects local
allowlisted search/read operations. Draft and review stages use actual collected
evidence. SQLite persists checkpoints, budgets, trace, memory and human decisions.
Human approve/modify/reject is distinct from independent benchmark answer review.

The current implementation includes prior-task memories in draft inputs and only
drafting is cache-eligible. The R5 source-replay experiment found no provider-call
reduction in the agent arms; see [closeout](docs/audit/R5_CLOSEOUT.md). Its model
plans, answers and reviews were scripted fixtures. Live agent quality is unmeasured.
The [R6 proposal](docs/audit/R6_AGENT_CACHE_PREREG.md) is not implemented yet.

## Failure and security boundaries

Malformed, refused or incomplete outputs are not admitted as complete answers.
Provider exceptions and invalid approval contracts abstain/fail rather than grant
reuse. This is a handling contract: a plausible but wrong model judgment is not
necessarily detectable. No claim that all model hallucinations are caught is made.

Trusted callers must supply actual tenant/principal permissions and a source
mapping they may read. The CLI is not a multi-user authentication server. Public
knowledge can be shared only within the declared deployment scope. Task deletion
and gateway cache invalidation are separate operations. Crash-time inference can
be repeated under durable reservations; exactly-once inference is not promised.
Secret-pattern scans reduce risk but do not certify supply-chain security.

## Historical experiments and interpretation

The original pipeline used a stability classifier, dense embeddings, FAISS,
auto-reuse thresholds, local equivalence and a remote ambiguous-tier judge.
These components remain for research/history; they are not all active in the
current serving path. Historical settings/results are in FACTS.md, not operating
safety guarantees. A query-dependent LLM judgment is not evidence of learning.
The early threshold fitting stayed on fallback; later domain fitting was reported,
but neither establishes useful deployed adaptive behaviour.

Historical AUTO_REUSE-versus-judge timings do not measure full response latency
against fresh generation. Avoiding a judge can mean bypassing cache and still
calling generation, so local routing resolution is not generation-call savings.
Zero observed benchmark errors do not certify safety. A cryptographic seal records
bytes/timing; it does not establish group independence or eliminate leakage.
Overlapping marginal confidence intervals alone are not a paired significance test.

The original full design is retained as a [superseded snapshot](docs/archive/DESIGN.before_claim_hygiene.md).
Corrections and evidence rows are listed in the [claim audit](docs/audit/CLAIM_HYGIENE_AUDIT.md).
