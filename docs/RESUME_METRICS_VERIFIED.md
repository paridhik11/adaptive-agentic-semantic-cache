# Resume evidence status — corrected 2026-11-02

Do not describe the five-arm reference replay as measured-live provider evidence.
It combines local-model execution with a scripted 20 ms provider; its whitespace
usage fields are not billable tokens. The original family-derived 12/12 is not
verified answer precision. Historical data remains unchanged.

Current actual-answer review is explicitly model-assisted, not human ground truth:
11 of 12 answer-aware reused answers are sufficient under a strict whole-answer
rubric; one gives a useful reverse loop but falsely claims `reversed` builds a
new copy. No population safety estimate or deployment qualification follows.
The other semantic arms retain unknown reviews. See
[BENCHMARK_EVIDENCE_CORRECTION.md](BENCHMARK_EVIDENCE_CORRECTION.md).

Supported historical engineering wording remains in `Resume_Ready_Project.md`:

- Tenant-scoped durable caching with TTL/invalidation; historical local SQLite
  warm-hit p95 1.79 ms across 5,000 requests, not end-to-end LLM time.
- Process-local coordination: 64 simultaneous constructed cold requests made
  one counted stub-provider invocation, not demonstrated paid-API savings.
- Bounded planning, source tools, review, memory and durable approval, validated
  with scoped offline regressions, not natural-answer accuracy observations.

The corrected 440-request offline gateway replay directly counts 220 no-cache
versus 198 exact-only scripted attempts, with zero completion failures. No live
provider calls were made. Billable tokens, API cost and live end-to-end efficiency
remain unverified; no headline application savings percentage is supported.
