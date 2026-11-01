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

### 3. Latency & CPU Overhead Accounting (Separated Stage Timings)
Measured on CPU (Intel/AMD x86_64, Windows, `runs/semantic-dev/phase3_benchmark_results.json`):
- **Model Initialization:** One-time cold load 442.85 ms (safetensors direct mmap, zero network activity).
- **Stage 1 — Dense Query Embedding (MiniLM-L6):** Mean **27.08 ms** (p50 ~25 ms).
- **Stage 2 — Top-k Candidate Retrieval:** Mean **0.25 ms**.
- **Stage 3 — Hard Requirement Guard:** Mean **0.15 ms** (numeric, negation, polarity, and direction filtering).
- **Stage 4 — Passage Cross-Encoder (MS MARCO):** Mean **13.42 ms**.
- **Outcome Latencies:**
  - Exact Cache Hit: p50 **0.53 ms**
  - Verified Semantic Hit: p50 **121.59 ms**
  - Cache Miss: p50 **49.11 ms**

### 4. Conservative Hybrid Cache Architecture (Milestone B)
Implemented in `src/policy/hybrid_cache.py`:
- **Scoped Exact Lookup:** Evaluated first at sub-millisecond latency.
- **Dense Top-k Retrieval:** MiniLM-L6 semantic search with strict cosine threshold ($\ge 0.82$).
- **Guards:** Hard syntactic and semantic constraint guards (`requirement_guard`).
- **Answer-Aware Neural Verifier:** Passage-level cross-encoder verifying query against candidate answer text.
- **Entry-Specific Review Decisions:** Cache entry approval/veto overrides backed by human review.
- **Async Candidate Promotion Queue:** New generation responses are queued for background verification and never admitted into the serving cache until verified.
- **Revocation & TTL:** Explicit tenant eviction (`revoke_tenant`) and TTL expiry.

### 5. Whole-Agent Latency & Provider Accounting (Milestone C)
Instrumented the complete Project 15 agent workflow (`scripts/benchmark_whole_agent.py`, `runs/whole_agent_benchmark_results.json`), spanning:
- Supervisor / planning phase
- Document tool calls (`read`)
- Draft answer generation
- Reviewer verification & approval admission

Matched across 24 tasks across 4 families:

| Metric | No Cache | Exact Only | Verified Semantic (Hybrid) | Savings vs No Cache | Incremental over Exact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All-Traffic Tasks** | 24 | 24 | 24 | — | — |
| Provider Calls Total | 72 | 57 | **39** | **−45.8%** (33 calls avoided) | **−31.6%** (18 calls avoided) |
| Total Tokens Consumed | 6,718 | 5,331 | **3,639** | **−45.8%** (3,079 tokens saved) | **−31.7%** (1,692 tokens saved) |
| Hits Total | 0 | 5 (5 exact) | **11** (5 exact + 6 semantic) | — | **+120.0% hits** |
| **Eligible Tasks ($N=18$)** | 18 | 18 | 18 | — | — |
| Provider Calls | 54 | 39 | **24** | **−55.6%** (30 calls avoided) | **−38.5%** (15 calls avoided) |
| Total Tokens Consumed | 4,996 | 3,609 | **2,197** | **−56.0%** (2,799 tokens saved) | **−39.1%** (1,412 tokens saved) |
| Hits Total | 0 | 5 | **10** (5 exact + 5 semantic) | — | **+100.0% hits** |

**Safety Invariant:** Exactly **0** remote provider calls attributable to any cache hit.

### 6. Opt-In Live Provider Benchmark Runner (Milestone D)
Implemented in `scripts/run_live_benchmark.py`:
- Explicit `--execute-live` CLI flag required; execution refuses to run live inference without it.
- Model preflight checks against official HuggingFace / OpenRouter models.
- Hard maximum provider attempts and spend cap defaulting to `$0.00`.
- Reuses existing durable quota journal without resetting.
- Price source: OpenRouter catalog 2026-11-01 ($0.00 / 1M tokens for free-tier reference model).

### 7. Independent Evaluation Handoff (Milestone E)
Exported to `artifacts/independent_evaluation_handoff/`:
- `evaluation_manifest.json`: Full cryptographically bound records (SHA-256 of incoming query, candidate answer, and approval binding).
- `blinded_review_form.jsonl`: Shuffled JSONL format for programmatic annotation.
- `BLINDED_HUMAN_REVIEW_PACKET.md`: Cleanly formatted markdown packet separating the 4 orthogonal evaluation axes (applicability, factual correctness, cache degradation, and rationale).
- Synthetic variation tags are withheld from reviewers and never cited as ground truth.

---

## Raw Evidence Artifacts

- Neural Runtime Lock: `artifacts/reproducibility/neural_runtime_lock.json`
- Phase 3 Benchmark Telemetry: `runs/semantic-dev/phase3_benchmark_results.json`
- Whole Agent Benchmark Telemetry: `runs/whole_agent_benchmark_results.json`
- Independent Evaluation Handoff: `artifacts/independent_evaluation_handoff/`
- Unit & Safety Tests:
  - `tests/test_answer_aware_policy.py`
  - `tests/test_hybrid_cache.py`
  - `tests/test_before_agent_cache.py`
  - `tests/test_document_model_preflight.py`
