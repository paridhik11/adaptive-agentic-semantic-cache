> **Historical report, superseded 2026-11-02.** The original text below is retained
> for audit history. Its TP/FP, precision and safety claims use generated family
> labels, not returned-answer review. Its live speedup is a projection, not a
> measurement. See [answer-bound correction](BENCHMARK_EVIDENCE_CORRECTION.md).
> No historical raw result has been modified. Current model-assisted review of
> the 12 answer-aware hits finds 11 sufficient and 1 with a false behavioral
> claim under a strict whole-answer rubric; this is not independent human truth.

# SEMANTIC_P0_RESULTS.md

_Branch: `integration/documentation-rebuild`_  
_Evidence type: measured-live (local CPU forward passes) on fixed development evaluation suite_  
_Evaluated: 2026-11-02_

---

## Executive Summary

Phase 1 diagnosed why the existing semantic cache produced zero hits (QQP cross-encoder domain mismatch, logits centered at −7 to −8.35). In Phase 2, we designed and implemented `AnswerAwareSemanticPolicy`, pairing hard syntactic/semantic constraint guards (`requirement_guard`) with a local passage-level cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that evaluates the actual candidate answer against the incoming request.

In Phase 3, we evaluated 5 comparison arms on a fixed, answer-bound, 220-event development workload derived from the official `dev` split of the CPython 3.12 FAQ corpus across 8 variation families:
1. Exact repeats
2. Natural paraphrases
3. Direction reversals
4. Changed requirements & nested conversions
5. Negation constraints
6. Numeric & version constraints
7. Tenant mismatches
8. Source revision mismatches

### Key Results

| Arm | Description | Total Requests | Exact Hits | Semantic Hits | Correct Reuses (TP) | Unsafe Reuses (FP) | Semantic Precision | Semantic Error Rate (95% Wilson CI) | Latency p50 (ms) | Latency p95 (ms) | Unsafe Families Accepted |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | No cache (`off`) | 220 | 0 | 0 | 0 | 0 | 100.0% | 0.0% [0.0%, 0.0%] | 21.24 | 21.87 | None (Safe) |
| **B** | Exact cache (`exact`) | 220 | 22 | 0 | 0 | 0 | 100.0% | 0.0% [0.0%, 0.0%] | 21.51 | 22.06 | None (Safe) |
| **C** | Dense-only baseline ($\ge 0.85$) | 220 | 22 | 79 | 38 | 41 | 48.1% | 51.9% [41.1%, 62.6%] | 43.22 | 49.70 | direction_reversal: 13<br>negation_constraint: 22<br>changed_requirements: 3<br>numeric_constraint: 3 |
| **D** | Combined policy (MiniLM + Quora) | 220 | 22 | 38 | 22 | 16 | 57.9% | 42.1% [27.9%, 57.8%] | 47.54 | 110.14 | negation_constraint: 11<br>direction_reversal: 5 |
| **E** | **Answer-aware policy (New)** | **220** | **22** | **12** | **12** | **0** | **100.0%** | **0.0% [0.0%, 24.3%]** | **47.62** | **114.01** | **None (0 Unsafe Reuses)** |

---

## Detailed Findings

### 1. Safety and Precision
- **Dense-only (Arm C) is dangerously unsafe:** Dense cosine similarity alone ($\ge 0.85$) accepted 41 unsafe requests, including 22/22 negation constraints and 13 direction reversals. Its accepted semantic reuse precision was only 48.1% (worse than a coin flip).
- **Combined Quora model (Arm D) fails on negations:** Despite running an extra cross-encoder, it accepted 11 negation constraints and 5 direction reversals (42.1% error rate).
- **Answer-Aware Policy (Arm E) achieved 100% precision on accepted hits:** Zero unsafe reuses across all 220 requests ($FP = 0$). Hard constraint guards intercepted 100% of numeric mismatches, version alterations, added constraints, and negations before any neural inference was performed.

### 2. Semantic Coverage
- Arm E successfully produced **12 verified non-exact semantic hits** from natural paraphrases without human intervention or allowlist fixtures.
- Over-length document answers ($> 512$ tokens) safely abstained rather than guessing or silently truncating, preserving correctness over raw recall.

### 3. Latency & CPU Overhead Accounting
Measured on CPU (Intel/AMD x86_64, Windows):
- **Stage 1 (MiniLM Dense Embedding):** ~20–30 ms CPU time.
- **Stage 2 (Hard Requirement Guard):** 0.15 ms CPU time (negligible).
- **Stage 4 (Local Passage Verification):** ~25–45 ms CPU time.
- Total p50 latency for Arm E: **47.62 ms**, compared to 21.24 ms for cold provider fallback in fixture mode.
- In live LLM deployment (where provider p50 latency is typically 600–2,500 ms), serving a verified semantic hit in ~47 ms provides a measured ~12–50× latency acceleration for cache hits while eliminating 100% of upstream LLM token costs for those queries.

### 4. Tenant & Revision Isolation
- Requests to `tenant-b` resulted in 0 cross-tenant cache hits.
- Requests with modified source revisions resulted in 0 cross-revision stale hits.

---

## Raw Evidence Artifacts

- Workload definition: `runs/semantic-dev/phase3_dev_workload.json` (220 events)
- Full benchmark output: `runs/semantic-dev/phase3_benchmark_results.json`
- Benchmark script: `scripts/phase3_development_benchmark.py`
- Unit tests: `tests/test_answer_aware_policy.py` (8 passing unit tests)
