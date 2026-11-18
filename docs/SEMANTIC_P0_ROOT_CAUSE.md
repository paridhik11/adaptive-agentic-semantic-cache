# SEMANTIC_P0_ROOT_CAUSE.md

_Branch: `integration/documentation-rebuild`_  
_Evidence type: historical component measurement (local model probe + full diagnostic run)_  
_Written: 2026-11-02_

---

## Summary

The existing semantic cache produces **zero non-exact hits** on the 30-question exposed
development packet. Root causes are concrete, reproducible, and measured.

---

## Measured Evidence (2026-11-02)

### Full 30-question diagnostic run (`scripts/phase1_score_diagnostic.py`)

| Policy | Accepted | TP | FP | IRR_cache | Coverage |
|---|---:|---:|---:|---:|---:|
| Dense-only (threshold=0.85) | 3 | 2 | 1 | 33.3% | 14.3% |
| Pair-only (any threshold) | **0** | 0 | 0 | N/A | 0% |
| Combined (dense+pair+margin) | **0** | 0 | 0 | N/A | 0% |

Source: `runs/semantic-dev/phase1-score-diagnostic.json` (historical component measurement, 2026-11-02).  
Human labels: 14 Y, 15 N, 1 ? (declared human review, not independently verified).  
30-question packet is exposed/spent; these numbers must not be used as a final test.

### Per-question Quora pair scores (all against designated seed)

| H-code | Human | Dense | Quora logit | pair_score | Fail reason |
|---|---|---:|---:|---:|---|
| H01 | N | 0.749 | −5.39 | 0.005 | margin_fail |
| H02 | N | 0.645 | −8.33 | 0.000 | margin_fail |
| H03 | **Y** | 0.770 | −4.44 | 0.012 | margin_fail |
| H04 | N | 0.706 | −5.53 | 0.004 | margin_fail |
| H05 | ? | 0.666 | −3.97 | 0.019 | margin_fail |
| H06 | N | 0.605 | −8.26 | 0.000 | margin_fail |
| H07 | N | 0.634 | −2.72 | 0.062 | dense_fail |
| H08 | **Y** | 0.427 | −8.32 | 0.000 | margin_fail |
| H09 | **Y** | 0.670 | −8.30 | 0.000 | margin_fail |
| H10 | **Y** | 0.870 | −1.21 | **0.230** | pair_fail |
| H11 | N | 0.706 | −8.26 | 0.000 | margin_fail |
| H12 | N | **0.923** | −5.51 | 0.004 | margin_fail |
| H13 | **Y** | 0.687 | −4.26 | 0.014 | margin_fail |
| H14 | N | 0.775 | −7.06 | 0.001 | margin_fail |
| H15 | **Y** | 0.560 | −8.35 | 0.000 | margin_fail |
| H16 | N | 0.765 | −6.20 | 0.002 | margin_fail |
| H17 | **Y** | 0.590 | −8.06 | 0.000 | margin_fail |
| H18 | N | 0.704 | −5.76 | 0.003 | margin_fail |
| H19 | **Y** | 0.890 | −7.93 | 0.000 | margin_fail |
| H20 | **Y** | 0.707 | −8.24 | 0.000 | margin_fail |
| H21 | N | 0.311 | −8.34 | 0.000 | margin_fail |
| H22 | N | 0.586 | −8.10 | 0.000 | margin_fail |
| H23 | N | 0.254 | −8.34 | 0.000 | margin_fail |
| H24 | **Y** | 0.772 | −6.73 | 0.001 | margin_fail |
| H25 | **Y** | 0.609 | −8.35 | 0.000 | margin_fail |
| H26 | **Y** | 0.783 | −4.48 | 0.011 | margin_fail |
| H27 | N | 0.356 | −8.29 | 0.000 | margin_fail |
| H28 | N | 0.387 | −8.34 | 0.000 | margin_fail |
| H29 | **Y** | 0.307 | −8.30 | 0.000 | margin_fail |
| H30 | **Y** | 0.332 | −8.32 | 0.000 | margin_fail |

**Key observation:** 27 of 30 questions have logit ≤ −3.97. H10 is the highest at
−1.21 (pair_score=0.230) — still well below any plausible pair threshold. Quora
CrossEncoder never fires. The dense cosines for H21–H30 (negative-index source)
are uniformly 0.25–0.59 — the seed question "What's a negative index?" does not
embed close to the new questions.

### 8-probe mini-diagnostic (`run_semantic_probe.py`)

| Pair type | Quora logit | pair_score |
|---|---:|---:|
| paraphrase (list→tuple) | **−7.51** | **0.0005** |
| paraphrase (tuple→list) | **−3.76** | **0.023** |
| paraphrase (Celsius→Fahrenheit) | **+2.18** | **0.898** ✓ |
| direction_flip (exact match) | **+4.50** | **0.989** |
| constraint (nested) | **−3.09** | **0.044** |
| negation | **−5.29** | **0.005** |
| unit_change (→Kelvin) | **−3.76** | **0.023** |
| constraint (sort order) | **−2.33** | **0.088** |

Temperature paraphrase scores 0.898 — model works for high-lexical-overlap cases.
Python type-conversion paraphrases score near zero — task-mismatch failure.

---

## Root Cause 1: Quora CrossEncoder Task Mismatch

**File:** [`src/policy/local_models.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/local_models.py) L13–14, L26–46  
**Model:** `cross-encoder/quora-distilroberta-base` (revision `f62e7a4b...`)

The model is trained on Quora Question Pairs (QQP): user-forum exact-duplicate detection.
It is being used to score question paraphrases for programmatic FAQ retrieval.

**Preprocessing:** Correct. Raw question text, no normalization. Token counts 20–30 (well within limit).  
**Activation:** Correct. `Identity()` suppresses built-in sigmoid; manual sigmoid on L45 is verified (temperature example: sigmoid(+2.18)=0.898 ✓).  
**Revision:** Correct. Weights loaded and producing finite outputs.

**Root cause is task mismatch, not a code bug.** The QQP model predicts
"are these the same user question posted twice?" not "does this question ask
for the same information as that question?" For "How do I convert a Python list
into a tuple?" vs "How can I make a tuple from an existing Python list?", the
model sees low lexical overlap and predicts "not duplicate" (logit=−7.51).

---

## Root Cause 2: MiniLM Cosine Cannot Distinguish Applicable vs Inapplicable

**File:** [`src/policy/encoders.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/encoders.py) L32–43  
**Config:** threshold=0.85, `semantic_runtime.py` L145

Dense cosine similarity retrieves by topic proximity, not answer applicability:
- H12 (list→tuple→list deep copy, human=N): cosine=**0.923** — above threshold
- H19 (list↔tuple, human=Y): cosine=0.890 — similar score to unsafe case H12

A safety policy that gates on cosine alone would serve incorrect answers.

H10 ("Can you show how to assign…") dense=0.870 — only question that passes
the dense threshold among the human-Y cases scored against the designated seed.
But its pair_score=0.230 still fails a reasonable pair threshold.

---

## Root Cause 3: Combined Policy Candidate-Selection Defect

**File:** [`scripts/score_document_diagnostic.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/scripts/score_document_diagnostic.py) L36–44

Combined policy ranks by pair score then checks dense+pair+margin. When all pair
scores are near zero (as observed), the "top" pair-ranked candidate is noise.
The margin check (`pair_score_best − pair_score_second ≥ 0.05`) is trivially
satisfied when both scores are 0.000 vs 0.001.

For the 30-question packet this produces 0 accepted of 30, including 0 of 14 human-Y cases.

---

## Root Cause 4: No Answer-Aware Verification in Active Pipeline

**File:** [`src/policy/semantic_runtime.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/semantic_runtime.py) L241–306

The active semantic pipeline:
1. Retrieves by MiniLM cosine (L255–261)
2. Applies similarity threshold and margin (L268–273)
3. Computes approval binding (L275–276)
4. Calls the configured `approver` (L282–283)

The `approver` in shadow/guarded mode is a configured object. In the observed
diagnostic runs, this is `AnswerBoundApprovals` (allowlist) — which cannot approve
questions it hasn't seen. There is no local answer-aware check.

**Existing code that could fix this:**
- [`src/policy/answer_applicability.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/answer_applicability.py): `DiagnosticAnswerPolicy` reads the actual answer and scores applicability
- [`src/policy/decomposed_applicability.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/decomposed_applicability.py): `DecomposedDiagnosticPolicy` applies NLI entailment chain
- [`src/policy/question_conditioned.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/question_conditioned.py): `QuestionConditionedPolicy` combines QA span extraction with NLI support

Both call `requirement_guard` from `decomposed_applicability.py` — which correctly
catches numeric changes, negations, direction reversals, and added constraints.

**These policies always abstain at the serving interface** (return `ReuseDecision(approved=False, ...)`).
They produce a `local_policy_recommendation` signal. Enabling serving requires
separately qualifying their recommendation quality on a fresh (non-exposed) dataset.

---

## Root Cause 5: Deprecation Warnings (Non-Breaking)

**File:** [`src/policy/local_models.py`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/src/policy/local_models.py) L26–29

`sentence-transformers 6.0.1` renamed:
- `automodel_args` → `model_kwargs`
- `default_activation_function` → `activation_fn`

Both deprecated args work in current version. Should be fixed to avoid future breakage.

---

## Root Cause Summary Table

| # | Root Cause | Code Location | Evidence |
|---|---|---|---|
| RC-1 | Quora CrossEncoder task mismatch: near-zero logits (−7.5 to −1.2) for FAQ paraphrases | `local_models.py` L13–14 | Measured-live: 30/30 questions fail pair threshold |
| RC-2 | MiniLM cosine doesn't distinguish applicable vs inapplicable (direction flip cosine=0.923) | `encoders.py`, `semantic_runtime.py` L145 | Measured-live: H12 (N) has cosine=0.923 > H19 (Y) cosine=0.890 |
| RC-3 | Combined policy selects by pair rank when all pair scores are ~0; margin check trivially satisfied | `score_document_diagnostic.py` L36–44 | 0/14 human-Y accepted |
| RC-4 | No answer-aware verification in active pipeline; existing `answer_applicability.py` diagnostic-only | `semantic_runtime.py` L241–306 | Design gap; not a code bug |
| RC-5 | Deprecation warnings in `LocalPairScorer` (non-breaking) | `local_models.py` L26–29 | Probe stderr |

---

## What Is NOT a Root Cause

- **Threshold choice:** Even at threshold=0.01, MiniLM produces false positives for
  direction flips (H12 cosine=0.923, human=N). Lowering the threshold is unsafe.
- **Model weights or revision:** Verified present and producing finite outputs.
- **Preprocessing:** Raw text is correct for QQP model input.
- **Activation bug:** Verified correct from temperature probe (sigmoid(+2.18)=0.898 ✓).

---

## Recommended Fix (Phase 2)

Reuse existing answer-aware code:

1. Retrieve top-k by MiniLM cosine (existing, working).
2. Apply `requirement_guard` hard constraint filters (existing, catches most unsafe cases).
3. Run `DiagnosticAnswerPolicy` or `DecomposedDiagnosticPolicy` on the actual cached answer.
4. Promote to `approved=True` only when local verifier recommends AND serving guards pass.
5. Keep shadow-only until a **fresh** (non-exposed) evaluation establishes safety.
6. Remove Quora CrossEncoder from the primary scoring path; retain for optional cross-validation.
