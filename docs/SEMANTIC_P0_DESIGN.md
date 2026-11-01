# SEMANTIC_P0_DESIGN.md

_Branch: `integration/documentation-rebuild`_  
_Written: 2026-11-02_

---

## Design Goal

Produce real non-exact semantic cache hits that are safe to serve, using only
local models and existing infrastructure. Do not invent positive results or weaken
safety requirements.

---

## Non-Design (Preserved Without Change)

- FastAPI gateway, exact SQLite cache, scope/tenant/TTL safety — unchanged.
- `GuardedSemanticCache` exact-first lookup and shadow/guarded mode — unchanged.
- `AnswerBoundApprovals` allowlist path — unchanged.
- Bounded documentation agent, reviewer, scoped memory, human approvals — unchanged.
- All telemetry, audit trail, and `FACTS.md` ledger — unchanged.

---

## Pipeline Design

```
Incoming request
     │
     ▼
[Stage 0] Exact key lookup (ScopedResponseCache)
     │ HIT → serve immediately
     │ MISS ↓
     ▼
[Stage 1] MiniLM top-k retrieval (existing DenseEncoder + GuardedSemanticCache._entries)
     │ no candidates → provider call
     │ candidates ↓
     ▼
[Stage 2] Hard constraint guards (requirement_guard from decomposed_applicability)
          • numeric/version equality
          • negation polarity
          • conversion direction
          • explicit added constraints
     │ any guard fires → provider call
     │ all pass ↓
     ▼
[Stage 3] Context boundary check (PinnedContextPolicy scope validation)
     │ out of scope → provider call
     │ in scope ↓
     ▼
[Stage 4] Local answer-aware verification (LocalYesNoScorer or DecomposedDiagnosticPolicy)
          • DiagnosticAnswerPolicy: flan-t5 yes/no on "does the cached answer fully
            address the new question?" — 2 forward passes per candidate
          • DecomposedDiagnosticPolicy: NLI entailment chain on actual answer claims
            against source document, plus intent NLI on incoming vs original request
     │ verifier abstains or recommends=False → provider call
     │ verifier recommends=True ↓
     ▼
[Stage 5] Final contradiction guard (checks answer doesn't contradict verifier)
     │ contradiction detected → provider call
     │ all pass ↓
     ▼
[Stage 6] Serve cached answer — record as verified semantic hit
     │ simultaneously (async, non-blocking, does not affect latency of current response):
     └─► [Optional] Asynchronous verified promotion:
           • store confirmed (incoming_request, cached_answer) binding in SQLite
           • binding includes verifier identity and recommendation score
           • future exact lookups can find this binding without re-running verifier
           • binding is versioned; source revision change invalidates it
```

---

## Design Decisions

### 1. Answer-aware verification model choice

Two existing options:

**A. `DiagnosticAnswerPolicy` (`answer_applicability.py`):**
- Uses `LocalYesNoScorer` (flan-t5-base) or `LocalChatYesNoScorer` (Qwen3-0.6B)
- Constructs a yes/no prompt with the full context, original question, new question, and response
- Checks sufficiency and source support as separate criteria
- 2 forward passes, ~50–400 ms CPU depending on context length
- Does not require source span registry

**B. `DecomposedDiagnosticPolicy` (`decomposed_applicability.py`):**
- Uses `LocalEntailmentScorer` (DeBERTa NLI model)
- Requires pre-registered source spans (EvidenceIndex)
- Checks: intent alignment (NLI), source evidence coverage, claim entailment
- More compute per call (~300–800 ms) but more structured
- Requires configuring source spans at startup

**Selected for Phase 2:** `DiagnosticAnswerPolicy` with `LocalYesNoScorer` (flan-t5-base)
because:
- No pre-registered span registry needed (works on any cached answer)
- Lower model load overhead (90M params vs 184M for DeBERTa)
- Already tested in `probe_answer_applicability.py`

`DecomposedDiagnosticPolicy` used as optional secondary check for high-stakes cases
where source spans are available.

### 2. No per-request remote judge for accepted hits

The Phase 8 pipeline (`src/decision/local_equivalence.py`) used a remote judge.
This design uses only local models. No API key or remote calls for accepted hits.

### 3. Quora CrossEncoder

The Quora pair scorer (`local_models.py`, `LocalPairScorer`) is **removed from the
primary serving path** based on Phase 1 diagnostic: it produces near-zero logits
(−7.5 to −3.1) for clear Python-domain paraphrases. It can remain as an optional
diagnostic comparator.

MiniLM cosine is used for retrieval only (top-k candidates). The answer-aware
verifier decides whether to serve, not the pair scorer.

### 4. Safety invariants

- Similarity alone never authorizes serving.
- `requirement_guard` is mandatory and fires before any model call.
- All verify fails fall through to fresh provider call.
- Serving path disabled by default; requires explicit mode change after evaluation qualifies it.
- `local_policy_recommendation` is a diagnostic signal; the serving gate is separate.

### 5. Draw from vCache / Krites ideas (without claiming reproduction)

- **vCache:** Per-entry adaptive decision rather than a single global threshold.
  Our approach uses a verifier decision per candidate, not a fixed similarity cutoff.
  We do not implement vCache's statistical calibration or its exact algorithms.
- **Krites:** Asynchronous verified promotion: confirmed hits are stored for future
  exact lookup without re-running the verifier.
  We implement a simplified version using the existing SQLite cache; we do not
  implement Krites' distributed consensus or probabilistic guarantees.

### 6. CPU overhead accounting

Expected per-call overhead when a semantic candidate is available:
- MiniLM embedding: ~30–60 ms (from existing FACTS rows P8-P06)
- `requirement_guard`: < 1 ms (pure Python regex)
- `DiagnosticAnswerPolicy` (flan-t5): ~100–500 ms CPU depending on context + answer length
- Total overhead for a cache miss path: ~130–560 ms additional vs no-cache path
- Total overhead for a cache hit path: same overhead as miss (verifier runs before serving)

These estimates are from historical component measurements
(FACTS P8-P05–P8-P08, P8-B01–P8-B03) and are not end-to-end latency guarantees.

---

## New File: `src/policy/answer_aware_policy.py`

A thin wiring layer that connects the existing components:

```
AnswerAwareSemanticPolicy(approver)
  .assess(request, candidate_request, answer)
    → ReuseDecision
```

The policy:
1. Calls `requirement_guard` — fail if any hard constraint fires
2. Calls `approver.assess(...)` (DiagnosticAnswerPolicy)
3. Returns `ReuseDecision(approved=verifier_recommendation, ...)`

This is NOT a new verifier — it is a wiring layer that allows the existing
`DiagnosticAnswerPolicy` to plug in as a proper `GuardedSemanticCache` approver
with `approved=True` when the local verifier recommends it.

Currently `DiagnosticAnswerPolicy.__call__` always returns `approved=False`
(diagnostic-only). The new policy enables serving while preserving all safety
constraints and keeping serving disabled by default.

---

## Integration: `GuardedSemanticCache`

No changes to `GuardedSemanticCache` are needed. The new policy is passed as the
`approver` argument:

```python
from src.policy.answer_applicability import DiagnosticAnswerPolicy, LocalYesNoScorer
from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy

base_policy = DiagnosticAnswerPolicy(system_prompt=..., scorer=LocalYesNoScorer())
approver = AnswerAwareSemanticPolicy(base_policy)  # adds requirement_guard

cache = GuardedSemanticCache(
    exact_cache,
    scope=...,
    encoder=DenseEncoder(ENCODER_MODEL, ENCODER_REVISION),
    mode="shadow",  # shadow until evaluation qualifies serving
    approver=approver,
    threshold=0.80,
    margin=0.05,
)
```

The `mode="shadow"` ensures no actual serving occurs until explicitly changed after
evaluation.

---

## Serving Enablement Gate

The new policy remains in `mode="shadow"` until a **fresh, non-exposed** evaluation:
1. Uses a dataset NOT in the 30-question exposed packet
2. Shows IRR_cache < 10% at 95% Clopper-Pearson CI upper bound
3. Has been reviewed by at least one qualified human reviewer

Only after meeting this gate will `mode="guarded"` be permitted for production use.

---

## Phase 3 Benchmark Design (Summary)

**Development benchmark** using answer-bound, source-version-aware records.

Splits:
- Group A: Natural paraphrases (within same seed)
- Group B: Exact repeats
- Group C: Direction reversals
- Group D: Changed requirements and nested conversions
- Group E: Negations, numeric constraints, and changed context
- Group F: Source-version and tenant mismatches
- Group G: Multiple candidate competition
- Group H: Different intent groups (cross-seed)

Comparison arms:
- A: No cache
- B: Exact cache
- C: Dense-only baseline (MiniLM cosine ≥ 0.85)
- D: Existing combined policy (dense + Quora pair + margin)
- E: New answer-aware policy (dense + requirement_guard + DiagnosticAnswerPolicy)

Required metrics:
- Semantic accepted-answer precision (TP / (TP + FP))
- Correct semantic reuse count and safe coverage
- Unsafe accepted reuses (by family)
- Cache hits breakdown (exact vs semantic)
- End-to-end p50/p95 latency (measured, not estimated)
- Local embedding and verification overhead (measured separately)
- Bootstrap CIs where justified

**Isolation:** Each arm uses an independently reset cache namespace. Same workload
for all arms. No test data from the exposed 30-question packet.

---

## Known Limitations

1. `DiagnosticAnswerPolicy` threshold (0.98 conditional_yes, 0.90 minimum_yes_mass)
   is calibrated on authored development examples — not on the target document domain.
2. Local flan-t5 yes/no scores are uncalibrated; they are not correctness probabilities.
3. CPU-only inference: flan-t5 inference 100–500 ms per candidate pair on this hardware.
4. The async promotion path requires SQLite write capability in the serving thread.
5. This design has not been evaluated on a fresh dataset yet. Serving remains disabled.
