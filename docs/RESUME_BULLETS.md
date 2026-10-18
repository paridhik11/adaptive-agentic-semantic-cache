# Resume wording supported by current evidence

Status: [CURRENT_STATUS.md](CURRENT_STATUS.md). Evidence: [FACTS.md](FACTS.md).
The project has not established representative end-to-end latency/cost savings
or safe general semantic reuse. These bullets describe implemented engineering.

## Recommended entry

**Adaptive Agentic Semantic Cache** | Python, FastAPI, SQLite, Pytest

- Built a scoped LLM response-cache gateway with authentication, persistent exact
  responses, TTL, invalidation and guarded semantic candidate/answer bindings.
- Implemented a bounded documentation agent with validated plans, allowlisted
  document tools, persisted checkpoints and resumable human approve/modify/reject.
- Developed paired evaluation and answer-bound review workflows that track model
  attempts, missing labels and orchestration overhead; documented negative results.

Do not imply production adoption, validated semantic accuracy, learned adaptation
or observed bill savings. FAISS and cross-encoders belong to the historical
research pipeline; the current serving wrapper is not that entire pipeline.

## Optional historical component-timing bullet

> Measured historical AUTO_REUSE stage latencies 99.20%–99.84% below remote LLM
> judge adjudication latency (not fresh answer generation or end-to-end application
> latency); AUTO_REUSE hit counts were n=20, n=1 and n=1 in the N=70, N=338 and
> N=175 runs respectively. Singleton timing samples do not establish a stable
> latency distribution; no sampling interval was established for that reduction.

Evidence: FACTS.md LAT-01–LAT-06, P5A-06–P5A-08, P5B-08–P5B-10, P6-09–P6-11.
These are historical measured component timings, not a proposed new percentage.
This is an optional technical detail, not the recommended headline resume claim.

## Optional historical safety observation

> Recorded no labeled incorrect reuse among n=8 accepted hits in the historical
> N=175 sealed benchmark: observed error rate 0.00%, exact two-sided 95%
> Clopper–Pearson interval [0.00%, 36.94%]. This underpowered observation does
> not establish safe semantic reuse or validate the later serving implementation.

Evidence: FACTS.md P6-01–P6-05. Independent actual-answer review of current live
serving remains pending. Historical benchmark labels are not silently promoted
into independent answer-level ground truth.

Prior wording is preserved in [the superseded snapshot](archive/RESUME_BULLETS.before_claim_hygiene.md).
See [claim audit](audit/CLAIM_HYGIENE_AUDIT.md) for errors removed, including the
unqualified latency headline, cross-cohort classifier latency, leakage guarantees
and unsupported security/production assertions.
