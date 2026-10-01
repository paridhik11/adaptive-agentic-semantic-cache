# Master Project Report: Adaptive Agentic Semantic Cache

> **Document Status:** Authoritative Master Project Report for `adaptive-agentic-semantic-cache`.  
> **Source Rule:** Every metric, threshold, latency timing, sample count, percentage, and confidence interval in this document cites a specific verified row in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).  
> **Stale Report Exclusion:** Neither `comprehensive_project_status_report.md` nor `Adaptive_Agentic_Semantic_Cache_Master_Report.pdf` was used.

---

## 1. Executive Summary

This project designed, implemented, and empirically evaluated a three-tier semantic caching architecture for large language model (LLM) query pipelines. Standard semantic caches rely on fixed embedding similarity thresholds, which encounter a severe failure mode: queries with inverted intent, altered technical parameters, or volatile temporal requirements often produce high cosine similarity, causing semantic cache hazards (incorrectly serving cached answers).

The system resolves this through three coordinated stages:
1. A fast local stability classifier that intercepts volatile queries before vector search.
2. A FAISS vector store that indexes temporal cache candidates using dense sentence embeddings.
3. A multi-tier routing gateway that cleanly bifurcates traffic into immediate local vector reuse (`AUTO_REUSE`), immediate cache bypass (`BYPASS`), and an ambiguous adjudication tier (`AMBIGUOUS`) evaluated by an external LLM judge.

### The Five Strongest Verified Findings

1. **Massive Latency Reduction on Cache Hits:** Serving cached results via `AUTO_REUSE` reduced query latency by **99.20% to 99.84%** across all load test runs (a speedup of **124.6x to 622.7x**; `AUTO_REUSE` mean latency of **10.56–61.91 ms** versus remote judge latencies of **6,577.94–9,291.35 ms**; cited in [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190), [`LAT-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191), [`LAT-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L192), [`LAT-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L193)). *Caveat:* The scaled run ($N=338$) and blind evaluation ($N=175$) each produced $n=1$ `AUTO_REUSE` hit; the $N=70$ pilot run produced $n=20$ hits (mean **19.18 ms**) and provides the most representative latency distribution ([`LAT-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L194)).
2. **Zero False Positives on Sealed Blind Evaluation:** In a strictly blind evaluation on a pre-sealed dataset of 175 entries across 7 domains, the cache committed **0 false positives** across 8 admitted cache hits ($k=0$, $n=8$; empirical $IRR_{\text{cache}} = \mathbf{0.00\%}$, exact 95% Clopper-Pearson CI **[0.00%, 36.94%]**; cited in [`P6-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L158), [`P6-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L159), [`P6-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L160)).
3. **Zero Dangerous False Positives on Held-Out Challenge Set:** On the operationally pristine held-out challenge benchmark ($N=60$), the two-stage `StabilityClassifier` achieved **91.67%** accuracy (55/60 correct; [`P1-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L70)) and committed **0 dangerous errors** across all 24 dynamic queries (FP=0; [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71)), verifying that temporal volatility is reliably intercepted before cache lookup under held-out evaluation conditions.
4. **Generalization on Final Test Benchmark:** On the independent 60-query final test set, the classifier demonstrated identical generalization with **91.67%** accuracy (55/60 correct; [`P1-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L68)) and **0 dangerous errors** on 23 dynamic queries ([`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69)), maintaining a low conservative false rejection rate of 13.51% (5/37 stable queries routed to dynamic; [`P1-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L72)).
5. **High Local Resolution Rate Resolved Without a Judge Call:** Across all three end-to-end load tests, **86.39% to 89.71%** of queries were resolved locally on-device without invoking an external LLM judge call (87.14% in the $N=70$ pilot, 86.39% in the $N=338$ scaled run, and 89.71% in the $N=175$ blind evaluation; cited in [`LRR-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L202), [`LRR-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L203), [`LRR-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L204), [`LRR-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L205)).

### Baseline Comparison & Calibration Finding (Same-Set)

- **Ambiguous-Band LLM Judge Point Estimate:** On the 120-pair baseline benchmark (`data/raw/query_pair_reuse_benchmark.json`), the best fixed cosine similarity threshold (0.85) failed the pre-stated safety ceiling ($IRR_{\text{cache}} < 10\%$) with an empirical hazard rate of **20.83%** (5 FP / 24 hits; [`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L74)). Integrating an LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`) for ambiguous pairs reduced the same-set point estimate to **8.33%** (1 FP / 12 hits, 95% Clopper-Pearson CI **[0.21%, 38.48%]**; [`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92)), achieving an actual reuse rate of 10.00% (12/120; [`P3-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L93)). However, because the exact 95% Clopper-Pearson confidence interval spans [0.21%, 38.48%], this same-set point estimate does not certify the 10% ceiling.


### Honest Limitations

- **Statistical Power Limits:** To mathematically guarantee that the true cache hazard rate does not exceed 10% under a 95% Clopper-Pearson confidence interval when observing zero errors ($k=0$), an evaluation must observe at least **$n \ge 36$ hits** ([`P4-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L105), [`KL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L214)). The blind run yielded $n=8$ hits (upper bound 36.94%; [`P6-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L160)), and pooling the scaled run and blind run yielded $n=22$ hits (upper bound 15.44%; [`POOL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L174)). No single blind evaluation achieved $n \ge 36$; all individual blind intervals remain underpowered.
- **Adaptive Mechanism Fallback:** The per-category adaptive threshold engine never altered an operating threshold in production; all 7 categories remained on the global 0.85 fallback due to finite-sample safety gates ([`P4-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L102), [`KL-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L223)).
- **Uncorrected Benchmark Collisions:** 6 undetected near-duplicate collisions identified in Phase 5 remain uncorrected in `data/raw/new_dataset_v3.json` ([`P5B-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L141), [`KL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L213)).
- **Synthetic Distributions:** Evaluations utilized structured synthetic streams and StackExchange seeds rather than organic enterprise customer traffic ([`KL-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L220)).
- **Free-Tier Infrastructure:** The production judge relies on OpenRouter's free tier (`nvidia/nemotron-3-super-120b-a12b:free`), which imposes a 50 requests-per-day quota limit ([`KL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L215)) and introduces dual-role bias in Phase 4 calibration ([`KL-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L216)).
- **In-Memory Store:** The FAISS index operates strictly in-memory without disk persistence across restarts ([`KL-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L219)).

---

## 2. System Overview

### Architecture

The system executes sequentially across three components:

```
Incoming Query
      │
      ▼
[ Stage 1: StabilityClassifier ] ── (conf < 0.80 or DYNAMIC) ──► BYPASS (Generate Fresh)
      │
      ▼ (STABLE, conf >= 0.80)
[ Stage 2: SemanticCache (FAISS FlatIP) ]
      │
      ▼ (Cosine Similarity Score)
[ Stage 3: TierRouter ]
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
AUTO_REUSE Tier                AMBIGUOUS Tier                  BYPASS Tier
(sim >= 0.92, conf >= 0.90)    (borderline similarity)        (sim < 0.50 or conf < 0.80)
      │                              │                              │
      ▼                              ▼                              ▼
Instant Local Hit (~10-60 ms)   LLM Judge Call (6,577.94–9,291.35 ms)    Instant Bypass (Generate)
```

### The Tier Rules As Coded

The exact routing rules implemented in `TierRouter.route()` (`src/decision/tier_router.py`) operate as follows:

- **`AUTO_REUSE` (Instant Cache Reuse):**
  - Requires `similarity_score >= auto_reuse_sim_floor` (**0.92**; [`C-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L34)), AND
  - `stability_confidence >= auto_reuse_conf_floor` (**0.90**; [`C-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L35)), AND
  - `effective_decision == StabilityLabel.STABLE`.
  - Serviced immediately from vector store memory without calling an LLM.

- **`BYPASS` (Instant Cache Miss):**
  - Triggered if `effective_decision != StabilityLabel.STABLE`, OR
  - `similarity_score < bypass_sim_ceiling` (**0.50**; [`C-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L36)), OR
  - `stability_confidence < bypass_conf_ceiling` (**0.80**; [`C-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L37)).
  - Bypasses cache immediately without calling an LLM.

- **`AMBIGUOUS` (Agentic Adjudication):**
  - Captures all intermediate queries where similarity falls in `[0.50, 0.92)` or stability confidence falls in `[0.80, 0.90)` ([`C-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L34), [`C-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L35), [`C-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L36), [`C-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L37)).
  - Dispatched to `ProductionDecisionStep` / `JudgeDecisionStep` (`src/decision/decision_step.py`), which calls `LLMJudge` (`src/decision/judge_call.py`).
  - Evaluated via `nvidia/nemotron-3-super-120b-a12b:free` on OpenRouter ([`C-14`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L42)).
  - **Fail-Closed Invariant:** On any timeout (10 seconds), network disconnect, rate limit exhaustion, malformed response, or missing API key, the judge strictly returns `decision=BYPASS`, `is_safe=False`, `confidence=0.0`, and `fallback_triggered=True` ([`C-16`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L44)). It never fails open.

### Key System Constants (Code-Sourced)

| Fact ID | Parameter Description | Exact Coded Value | Source Code Location |
| :---: | :--- | :--- | :--- |
| [`C-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L29) | Dense embedding model name | `all-MiniLM-L6-v2` | `src/cache/embedding.py` line 52 |
| [`C-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L30) | Embedding model git revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `src/cache/embedding.py` line 63 |
| [`C-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L31) | Embedding vector dimension | 384 | `src/cache/embedding.py` line 53 |
| [`C-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L32) | FAISS index type | `faiss.IndexFlatIP` (cosine similarity on normalized vectors) | `src/cache/vector_store.py` line 54 |
| [`C-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L33) | Stability classifier confidence override floor | 0.80 | `src/classifier/stability_classifier.py` line 71 |
| [`C-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L34) | TierRouter AUTO_REUSE similarity floor | 0.92 | `src/decision/tier_router.py` line 210 |
| [`C-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L35) | TierRouter AUTO_REUSE confidence floor | 0.90 | `src/decision/tier_router.py` line 211 |
| [`C-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L36) | TierRouter BYPASS similarity ceiling | 0.50 | `src/decision/tier_router.py` line 213 |
| [`C-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L37) | TierRouter BYPASS confidence ceiling | 0.80 | `src/decision/tier_router.py` line 214 |
| [`C-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L38) | Adaptive engine minimum domain sample gate | 20 | `src/decision/adaptive_threshold_engine.py` line 78 |
| [`C-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L39) | Adaptive engine minimum minority class gate | 10 | `src/decision/adaptive_threshold_engine.py` line 79 |
| [`C-12`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L40) | Adaptive engine global fallback threshold | 0.85 | `src/decision/adaptive_threshold_engine.py` line 80 |
| [`C-13`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L41) | Pre-stated safety ceiling $IRR_{\text{cache}}$ | 0.10 (10%) | `src/decision/adaptive_threshold_engine.py` line 82 |
| [`C-14`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L42) | Production LLM judge model | `nvidia/nemotron-3-super-120b-a12b:free` | `src/decision/judge_call.py` line 34 |
| [`C-15`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L43) | Initial judge attempted then abandoned | `gemini-2.5-flash` (halted after 17 pairs by 20 RPD daily quota) | `src/decision/judge_call.py` line 36 |
| [`C-16`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L44) | Judge fail-closed safety fallback | `decision=BYPASS`, `is_safe=False`, `confidence=0.0` | `src/decision/judge_call.py` lines 8–14, 356, 391–405 |

---

## 3. Dataset Roles Table

| Role ID | Dataset Role | Dataset File | $N$ | Composition / Breakdown | Evaluation Nature | Sourced Facts |
| :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| [`DROLE-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L231) | Classifier Train / Dev | `data/raw/query_stability_benchmark.json` + `data/raw/query_stability_human_credibility.json` | 245 | 160 dev queries + 85 human credibility queries | Same-set development and diagnostic tuning | [`D-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L53), [`D-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L56) |
| [`DROLE-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L232) | Held-Out Challenge | `data/raw/query_stability_benchmark_heldout.json` | 60 | 31 STABLE, 24 DYNAMIC, 5 CONDITIONALLY_STABLE | Held-out challenge (unseen during prompt/rule tuning) | [`D-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L54), [`P1-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L70), [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71) |
| [`DROLE-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L233) | Final Test | `data/raw/query_stability_benchmark_final_test.json` | 60 | 37 STABLE, 23 DYNAMIC | Clean for tuning; seen once by pytest ([`KL-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L222)) | [`D-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L55), [`P1-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L68), [`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69) |
| [`DROLE-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L234) | Cache Pipeline Calibration | `data/raw/query_pair_reuse_benchmark.json` | 120 pairs | 62 SAFE, 58 UNSAFE | Same-set baseline sweep & Phase 3 evaluation ([`KL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L218)) | [`D-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52), [`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L74)–[`P2-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L79), [`P3-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L87)–[`P3-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L94) |
| [`DROLE-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L235) | Cache Pipeline Calibration | `data/raw/synthetic_query_pair_feedback.json` | 108 pairs | 108 targeted synthetic pairs across 7 domains | Same-set calibration for adaptive threshold sweep | [`D-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L57), [`P4-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L103) |
| [`DROLE-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L236) | Design-Informed Pilot | `data/raw/load_test_query_stream.json` | 70 queries | Hand-crafted synthetic pilot stream | Design-informed pilot run | [`D-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L58), [`P5A-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L119)–[`P5A-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L127) |
| [`DROLE-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L237) | Design-Informed Pilot | `data/raw/new_dataset_v3.json` | 338 queries | 210 StackExchange cold seeds + 128 hand-authored | Scaled pilot (contains labeling collisions; [`KL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L213)) | [`D-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L59), [`P5B-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L135)–[`P5B-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L145) |
| [`DROLE-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L238) | Blind Evaluation | `data/raw/phase6_blind_eval_dataset.json` | 175 entries | 175 entries across 7 domains (sealed commit `fc35744`) | Strictly blind evaluation (unseen by pipeline/prompts) | [`D-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L60), [`P6-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L156)–[`P6-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L166) |

---

## 4. Phase Reports

### Phase 1: Query Stability Classifier
*RECONSTRUCTED (no Phase 1 walkthrough exists)*

#### Objective
Determine whether incoming queries can be reliably classified into temporally stable versus volatile/dynamic requests before cache lookup, ensuring volatile requests bypass caching with zero dangerous false admissions.

#### Method / steps
1. Built a two-stage classifier architecture in `src/classifier/`:
   - Stage 1: `StabilityRuleEngine` (`src/classifier/rule_engine.py`) using regex pattern matching for temporal triggers, live data, and volatile phrases.
   - Stage 2: `TrainedLexicalClassifier` (`src/classifier/trained_fallback.py`, trained via `scripts/train_fallback.py`) utilizing TF-IDF feature extraction and logistic regression.
2. Implemented Sub-stage A conservative uncertainty override in `src/classifier/stability_classifier.py`: any prediction with stability confidence $< 0.80$ is automatically overridden to `DYNAMIC` (cache bypass; [`C-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L33)).
3. Developed evaluation harness in `src/evaluation/stability_evaluator.py`.
4. Evaluated against held-out challenge queries (`scripts/step0b_heldout_test_eval.py`) and final test queries (`scripts/step0_final_test_eval.py`).

#### Data used
- Development set: 160 queries (`data/raw/query_stability_benchmark.json`; [`D-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L53)).
- Human credibility diagnostic set: 85 queries (`data/raw/query_stability_human_credibility.json`; [`D-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L56)).
- Final test set: 60 queries (37 STABLE, 23 DYNAMIC; `data/raw/query_stability_benchmark_final_test.json`; [`D-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L55)).
- Held-out challenge set: 60 queries (31 STABLE, 24 DYNAMIC, 5 CONDITIONALLY_STABLE; `data/raw/query_stability_benchmark_heldout.json`; [`D-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L54)).

#### Results
- Final test accuracy: **91.67%** (55/60 correct; [`P1-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L68); *held-out*).
- Final test dangerous errors: **0** (FP=0 on 23 dynamic queries; [`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69); *held-out*).
- Final test conservative error rate: **13.51%** (FN=5/37 stable queries; [`P1-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L72); *held-out*).
- Classifier latency percentiles (pristine final test): mean **2.29 ms** (2.2875 ms), median **0.25 ms** (0.2517 ms), P95 **8.70 ms** (8.7019 ms) ([`P1-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L74); *held-out*).
- Classifier pipeline routing resolution: Stage 1 Rule Engine **40** queries (**66.7%**), Stage 2 Fallback (`TrainedLexicalClassifier`) **20** queries (**33.3%**) ([`P1-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L75); *held-out*).
- Held-out challenge accuracy: **91.67%** (55/60 correct; [`P1-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L70); *held-out*).
- Held-out challenge dangerous errors: **0** (FP=0 on 24 dynamic queries; [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71); *held-out*).
- Held-out challenge conservative error rate: **13.89%** (FN=5/36 stable queries; [`P1-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L73); *held-out*).

#### Issues faced
In initial testing on the credibility set, queries regarding software versions (e.g. "Python version") produced confidence scores escaping to STABLE due to strong programming cues. Furthermore, uncertain queries below the 0.80 boundary caused unacceptable volatility risks ([`C-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L33)).

#### How each issue was resolved
The Sub-stage A uncertainty override was codified in commit `c8bebb6` and documented in `59efc32`, fixing `STABLE_CONFIDENCE_THRESHOLD = 0.80` ([`C-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L33)). Any query with confidence $< 0.80$ is forced to `DYNAMIC`. Dataset leakage between benchmark subsets was resolved in commit `56ce16d`.

#### Decision and why
The two-stage classifier successfully prevented all dangerous false positives across both 60-query benchmark sets ([`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69), [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71)) while resolving at local classifier speed. The classifier was frozen to serve as the gateway filter for Phase 2.

#### What it does NOT prove
Does not prove query equivalence or safe semantic reuse between pairs; only assesses single-query temporal stability. Furthermore, the final test set was executed once against a pytest assertion (`assert metrics.accuracy >= 0.85` in `tests/test_stability_evaluation.py` line 74) prior to formal evaluation ([`KL-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L222)).

---

### Phase 2: Semantic Cache Core & Threshold Sweep

#### Objective
Determine whether fixed-threshold cosine similarity on dense sentence embeddings can satisfy an enterprise safety ceiling of $IRR_{\text{cache}} < 10\%$ ([`C-13`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L41), [`P2-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L75)) while achieving meaningful cache hit yield.

#### Method / steps
1. Verified pair reuse dataset schema and disjunction against Phase 1 sets in `docs/phase2_walkthrough.md`.
2. Implemented embedding module `src/cache/embedding.py` using `sentence-transformers` model `all-MiniLM-L6-v2` pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` ([`C-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L29), [`C-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L30)).
3. Implemented in-memory vector store `src/cache/vector_store.py` using `faiss.IndexFlatIP` ([`C-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L32)).
4. Implemented evaluation runner `src/evaluation/cache_evaluator.py` and sweep script `scripts/run_cache_threshold_sweep.py`.
5. Pre-stated the safety criterion ($IRR_{\text{cache}} < 10\%$; [`P2-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L75)) prior to running the sweep.
6. Executed systematic threshold sweep across the 120 pairs ([`P2-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L79)).

#### Data used
- 120 query pairs (62 SAFE, 58 UNSAFE) from `data/raw/query_pair_reuse_benchmark.json` ([`D-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52), [`P2-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L79)).

#### Results
- Best fixed-threshold hazard rate: **20.83%** (5 FP / 24 hits) at threshold **0.85** ([`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L76); *same-set*).
- Pre-stated safety criterion: $IRR_{\text{cache}} < 10\%$ ([`P2-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L77)).
- Does any fixed threshold satisfy $IRR_{\text{cache}} < 10\%$? **NO** — at every threshold where hits $> 0$, $IRR_{\text{cache}} \ge 20.83\%$ ([`P2-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L78); *same-set*).
- Reference threshold for comparison: 0.85 (ARR=20.00%, CRR=79.17%, IRR=20.83%; [`P2-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L79); *same-set*).
- False Rejection Rate at reference threshold 0.85: **69.35%** (FN=43/62 missed safe reuses; [`P2-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L82); *same-set*).
- Mean embed + search latency: **13.9–14.9 ms** per pair ([`P2-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L80); *same-set*).

#### Issues faced
The embedding model proved blind to subtle programmatic and semantic inversions. Queries with inverted intent (e.g., ascending vs. descending sort, string-to-int vs. int-to-string) generated cosine similarities exceeding 0.92 ([`C-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L34)). Offline model loading initially failed due to unpinned HuggingFace revisions.

#### How each issue was resolved
Offline reproducibility was resolved in commit `7ea482b` by pinning `all-MiniLM-L6-v2` to commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (`docs/phase2_reproducibility_fix.md`). The inability of fixed thresholds to satisfy $IRR_{\text{cache}} < 10\%$ was accepted as an empirical finding establishing that embedding cosine similarity alone is structurally insufficient for enterprise cache safety.

#### Decision and why
No fixed threshold was recommended for production. Threshold 0.85 was designated as an unvalidated reference baseline ([`P2-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L79)). Unlocked Phase 3 to develop an agentic decision layer capable of adjudicating ambiguous pairs.

#### What it does NOT prove
Does not demonstrate generalization to unseen traffic distributions (evaluated strictly on the 120 calibration pairs; [`KL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L218)).

---

### Phase 3: Reuse Decision Layer & LLM Judge Selection

#### Objective
Determine whether a tiered routing gateway and ambiguous-band adjudication (comparing heuristics, adaptive thresholding, trivial bypass, and LLM judges) can clear the $IRR_{\text{cache}} < 10\%$ safety ceiling ([`C-13`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L41)) on ambiguous pairs.

#### Method / steps
1. Justified tier boundaries using Phase 1 and 2 distributions prior to coding (`auto_reuse_sim_floor=0.92` [`C-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L34), `auto_reuse_conf_floor=0.90` [`C-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L35), `bypass_sim_ceiling=0.50` [`C-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L36), `bypass_conf_ceiling=0.80` [`C-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L37)).
2. Implemented `TierRouter` (`src/decision/tier_router.py`), `DecisionStep` (`src/decision/decision_step.py`), and `DecisionEvaluator` (`src/evaluation/decision_evaluator.py`).
3. Evaluated linear combination heuristic (weighting similarity, confidence, and category history).
4. Evaluated adaptive threshold engine (`AdaptiveThresholdEngine` in `src/decision/adaptive_threshold_engine.py`) using logistic regression upper confidence bounds ($\theta_{\text{upper}}$).
5. Evaluated trivial bypass-all-ambiguous baseline.
6. Evaluated Google Gemini judge (`gemini-2.5-flash`) via `scripts/run_gemini_eval.py`.
7. Evaluated OpenRouter judge (`nvidia/nemotron-3-super-120b-a12b:free`) via `scripts/run_openrouter_eval.py`.

#### Data used
- 120 query pairs from `data/raw/query_pair_reuse_benchmark.json` ([`D-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52)).

#### Results
- Ambiguous tier traffic fraction: **26.7%** (32/120 pairs; AUTO_REUSE=2 [1.7%], BYPASS=86 [71.7%]; [`P3-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L94); *same-set*).
- AUTO_REUSE tier hazard: **0.00%** (TP=2, FP=0; [`P3-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L89); *same-set*).
- Linear combination non-LLM decision step: $IRR_{\text{cache}} = \mathbf{52.17\%}$ (12 FP / 23 hits; AMBIGUOUS tier hazard **57.1%**; [`P3-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L87), [`P3-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L88); *same-set*).
- Adaptive threshold engine ($\theta_{\text{upper}}$): $IRR_{\text{cache}} = \mathbf{23.08\%}$ (3 FP / 13 hits, ARR = **10.83%**, CRR = 76.92%; failed the 10% ceiling; [`P3-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L96), [`P3-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L97); *same-set*).
- Trivial bypass-all ambiguous baseline: $IRR_{\text{cache}} = \mathbf{0.00\%}$ (0 FP / 2 hits, ARR = **1.67%**, CRR = 100.00%; [`P3-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L98), [`P3-12`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L99); *same-set*).
- Gemini LLM judge (`gemini-2.5-flash`): $IRR_{\text{cache}} = \mathbf{0.00\%}$ (0 FP / 5 hits, ARR = 4.17%; partial run: 17 evaluated, 15 fell back on quota; [`P3-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L90), [`P3-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L91); *same-set*).
- OpenRouter LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`): $IRR_{\text{cache}} = \mathbf{8.33\%}$ (1 FP / 12 hits, ARR = **10.00%**, CRR = 91.67%, exact 95% Clopper-Pearson CI **[0.21%, 38.48%]**; [`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92), [`P3-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L93); *same-set*).

#### Issues faced
- **Non-LLM Failure:** Linear combination score blending failed disastrously at 52.17% IRR ([`P3-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L87)) because continuous weighting cannot detect discrete semantic polarity flips.
- **Adaptive Failure:** Adaptive logistic upper-bound thresholds failed the safety ceiling with 3 FP out of 13 hits ([`P3-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L96)) because syntactic similarity in CS pairs masked algorithmic inversions.
- **Gemini Quota Exhaustion:** `gemini-2.5-flash` hit Google AI Studio's strict free-tier limit of 20 requests per day (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), completing only 17 of 32 ambiguous pairs before the remaining 15 were forced into fail-closed BYPASS ([`C-15`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L43)).
- **Rejection of Simulated Results:** Any mocked, simulated, or fabricated judge completions were explicitly discarded; only authentic API responses were accepted for the comparative benchmark.

#### How each issue was resolved
Switched to OpenRouter's free tier with `nvidia/nemotron-3-super-120b-a12b:free` in commit `ab033c0`. Under paced execution, it completed all 32 ambiguous-tier evaluations with 0 fallbacks, achieving an empirical hazard rate of 8.33% (1 FP on a superset query; [`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92)) and clearing the 10% ceiling. `JudgeDecisionStep` was wired as the default production ambiguous path.

#### Decision and why
Adopted `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter as the production ambiguous-band judge: 12 hits of which 11 true (1 FP), versus 2 hits for the trivial baseline, same-set ([`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92), [`P3-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L98)).

#### What it does NOT prove
Does not demonstrate generalization to unseen queries; all six candidate strategies were evaluated on the same 120 calibration pairs ([`KL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L218)).

---

### Phase 4: Adaptive Policy & Calibration Loop

#### Objective
Determine whether domain-specific similarity thresholds can be safely tuned below the 0.85 global fallback using synthetic query pairs and judge feedback under dual sample-size gates and Clopper-Pearson confidence bounds.

#### Method / steps
1. Generated 108 synthetic query pairs across 7 domains (`scripts/build_and_validate_synthetic_pairs.py`).
2. Evaluated pairs with OpenRouter judge across multiple quota days (`scripts/label_synthetic_feedback.py`, `scripts/retry_fallback_pairs.py`).
3. Ran calibration sweep (`scripts/run_phase4_calibration.py`) computing Delta-method logistic CIs on threshold location $\theta$ and exact Clopper-Pearson CIs on $IRR_{\text{cache}}$.
4. Enforced dual sample gates: `MINIMUM_CATEGORY_N = 20` ([`C-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L38)) and `MINIMUM_MINORITY_CLASS_N = 10` ([`C-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L39)).
5. Enforced safety criterion: operating threshold is the lowest threshold where hits $> 0$ and the Clopper-Pearson 95% CI upper bound on $IRR_{\text{cache}} \le 10\%$.

#### Data used
- 108 synthetic pairs (`data/raw/synthetic_query_pair_feedback.json`; [`D-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L57)) combined with 120 benchmark pairs ([D-01](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52)).

#### Results
- Categories moved off 0.85 fallback: **0 of 7** — all 7 categories remain on the **0.8500** global fallback ([`P4-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L102), [`P4-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L104); *same-set*).
- Genuine judge labels obtained: **108 of 108** (100% across 3 sessions; [`P4-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L103)).
- Multi-session labeling breakdown: Session 0 completed **22** genuine evaluations (and 86 fallback stubs); Session 1 added **47** genuine evaluations (cumulative **69** of **108**; 39 stubs); Session 2 completed all **39** remaining evaluations, reaching **108 of 108** genuine labels (**100%**) with zero stubs remaining ([`P4-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L103), [`P4-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L108)).
- Mathematical minimum hits to prove $IRR_{\text{cache}} \le 10\%$ ($k=0$ errors, 95% Clopper-Pearson CI): **$n \ge 36$ hits** ([`P4-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L105)).

#### Issues faced
- **Stub Contamination Bug:** In Session 0, 86 rate-limited fallback stubs (`fallback_triggered=True`) were erroneously ingested by `load_combined_data()` as genuine negative labels, distorting initial threshold fits ([`P4-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L106)).
- **Quota Throttling:** OpenRouter's 50 requests-per-day free-tier ceiling required splitting labeling across 3 quota days.
- **Dual-Role Bias:** `nvidia/nemotron-3-super-120b-a12b:free` acted as both production judge and ground-truth labeler ([`P4-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L107), [`KL-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L216)).

#### How each issue was resolved
The contamination bug was fixed in commit `810f63e` by filtering out all records where `fallback_triggered=True`. Paced labeling completed 108/108 genuine labels in commits `3fb6fad` and `a765bbe`.

#### Decision and why
Because finite sample statistics prove that $n \ge 36$ hits with zero errors are mathematically required to bound $IRR_{\text{cache}} \le 10\%$, no domain had sufficient empirical mass to lower its threshold safely. Retaining the 0.85 fallback across all 7 categories was the mathematically sound decision.

#### What it does NOT prove
Does not demonstrate that adaptive thresholding improves hit yield in production; the mechanism never fired and all categories remained on fallback ([`KL-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L223)).

---

### Phase 5: Scaled Load Testing & Collision Audit

#### Objective
Evaluate the complete end-to-end pipeline under continuous load to measure latency speedups, local offloading rates, and empirical cache hazard rates.

#### Method / steps
1. Evaluated frozen `StabilityClassifier` on final test and held-out challenge benchmarks (`scripts/step0_final_test_eval.py`, `scripts/step0b_heldout_test_eval.py`).
2. Executed N=70 pilot load test (`scripts/run_load_test.py`, `data/raw/load_test_query_stream.json`) and generated Streamlit dashboard (`dashboard/app.py`).
3. Sourced scaled 338-query stream (`data/raw/new_dataset_v3.json`) combining 210 StackExchange seeds and 128 authored queries.
4. Executed scaled load test (`scripts/run_new_dataset_load_test.py`) and recorded telemetry.
5. Audited 10 unexpected false positive outcomes and executed intra-dataset collision sweep.

#### Data used
- Pilot stream: 70 queries (`data/raw/load_test_query_stream.json`; [`D-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L58)).
- Scaled stream: 338 queries (`data/raw/new_dataset_v3.json`; [`D-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L59)).

#### Results
- **N=70 Pilot Run:**
  - Cache hits: **22** total (**20 AUTO_REUSE + 2 judge-approved**, FP=0; [`P5A-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L120), [`P5A-02b`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L121); *design-informed*).
  - Empirical $IRR_{\text{cache}}$: **0.00%** (k=0 FP / n=22 hits; 95% Clopper-Pearson CI **[0.00%, 15.44%]**; [`P5A-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L122), [`P5A-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L123); *design-informed*).
  - Local resolution rate: **87.14%** (61/70 without remote call; [`P5A-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L124), [`LRR-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L202)).
  - Latency: `AUTO_REUSE` mean **19.18 ms** (n=20) vs AMBIGUOUS judge mean **9,291.35 ms**; speedup **484.5x** (**99.79%** reduction; [`P5A-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L125), [`P5A-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L126), [`P5A-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L127), [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189)).
- **N=338 Scaled Run:**
  - As originally labeled: $IRR_{\text{cache}} = \mathbf{71.43\%}$ (10 FP / 14 hits, 95% CI **[41.90%, 91.61%]**; [`P5B-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L135), [`P5B-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L138); *design-informed*).
  - After ground-truth collision correction: $IRR_{\text{cache}} = \mathbf{0.00\%}$ (0 FP / 14 hits: **1 AUTO_REUSE + 13 judge-approved**, 95% CI **[0.00%, 23.16%]**; [`P5B-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L136), [`P5B-02b`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L137), [`P5B-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L139); *design-informed*).
  - Local resolution rate: **86.39%** (292/338; [`P5B-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L142), [`LRR-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L203)).
  - Latency: `AUTO_REUSE` mean **61.91 ms** (n=1) vs AMBIGUOUS judge mean **7,715.62 ms** (46 calls); speedup **124.6x** (**99.20%** reduction; [`P5B-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L143), [`P5B-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L144), [`P5B-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L145), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190)).

#### Issues faced
- **Annotation Collisions:** Automated scoring against `new_dataset_v3.json` produced an apparent 71.43% hazard rate because 10 queries labeled `MISS` were exact duplicates or direct paraphrases of earlier StackExchange cold seeds. The judge correctly recognized their equivalence and permitted reuse, which the benchmark miscounted as false positives.
- **Uncorrected Near-Duplicates:** An audit of 27 collision candidates identified 6 additional near-duplicates that remained undetected during execution (masked by BYPASS or classifier) and remain uncorrected in the raw file ([`P5B-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L141), [`KL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L213)).
- **Documentation Calculation Error:** Documentation originally misreported the N=338 latency reduction as 99.79% (copying N=70) instead of the true value of 99.20% ([`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190), [`LAT-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L193)).

#### How each issue was resolved
All 10 collision queries were audited in commit `f9e1545` and relabeled in `scaled_load_test_telemetry_corrected.json`. Dual reporting was mandated to present both perspectives. The internal collision checker `check_internal_collisions()` was permanently added to `scripts/run_new_dataset_load_test.py`. An erratum was added to `docs/phase5_walkthrough.md` in commit `cb8946b` establishing the true 99.20% reduction ([`LAT-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L193)).

#### Decision and why
Confirmed the architectural invariant that $> 86\%$ of traffic resolves locally while validating sub-100 ms local cache speedups. Revealed the critical need for a pre-sealed blind evaluation dataset constructed with rigorous collision checking before execution.

#### What it does NOT prove
Both datasets were design-informed rather than blind. The scaled run produced only $n=1$ `AUTO_REUSE` hit ([`LAT-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L194)), and its Clopper-Pearson CI upper bound (23.16%) remains underpowered ([`KL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L214)).

---

### Phase 6: Blind Evaluation

#### Objective
Evaluate the entire cache decision pipeline against a genuinely blind, pre-sealed dataset never previously seen by designers or algorithms.

#### Method / steps
1. Authored 175-entry dataset using `scripts/build_phase6_sealed_dataset.py`, enforcing automated internal collision checks and 9-way prior leakage checks.
2. Committed and sealed dataset at commit `fc35744` ([`P6-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L155)) before running the pipeline.
3. Executed a single, uninspected run via `scripts/run_new_dataset_load_test.py`.
4. Evaluated telemetry `data/phase6_telemetry.json` and computed exact Clopper-Pearson intervals.

#### Data used
- 175 entries across 7 domains (`data/raw/phase6_blind_eval_dataset.json`; [`D-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L60), [`P6-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L156)).

#### Results
- Hit count: **8** total (**1 AUTO_REUSE + 7 judge-approved**, FP=0; [`P6-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L158); *blind*).
- Empirical $IRR_{\text{cache}}$: **0.00%** (k=0 FP / n=8 hits; [`P6-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L159); *blind*).
- Exact 95% Clopper-Pearson CI: **[0.00%, 36.94%]** (k=0, n=8; [`P6-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L160); *blind*).
- Local resolution rate: **89.71%** (89.71% without remote judge; [`P6-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L161); *blind*).
- Judge calls made: 18 ([`P6-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L162)).
- Judge outcome breakdown: **TP=7, TN=9, FN=2, FP=0** ([`P6-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L163)).
- Latency: `AUTO_REUSE` mean **10.56 ms** (n=1) vs AMBIGUOUS judge mean **6,577.94 ms** (n=18); speedup **622.7x** (**99.84%** reduction; [`P6-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L164), [`P6-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L165), [`P6-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L166), [`LAT-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191)).
- **Pooled Phase 5 Corrected + Phase 6 Blind:** Total $k=0$ FP ([`POOL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L174)), $n=22$ hits (14 + 8; [`POOL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L175)), exact 95% Clopper-Pearson upper bound **15.44%** ([`POOL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L176)). Does not clear the 10% ceiling ([`POOL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L178)). *(Sensitivity including N=70 pilot yields $n=44$, CI upper bound **8.04%** ([`POOL-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L179)), but pilot was design-informed, not blind).*

#### Issues faced
- **Underpowered Sample:** The blind dataset produced only 8 cache hits, resulting in an exact 95% CI upper bound of 36.94% ([`P6-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L158), [`P6-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L160)). Even pooled with Phase 5 ($n=22$), the upper bound (15.44%) remains above 10% ([`POOL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L175), [`POOL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L176), [`POOL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L178)).
- **Section 6 Walkthrough Discrepancy:** The Phase 6 walkthrough text initially reported TP=6, TN=10 for the judge tier, whereas raw telemetry recorded TP=7, TN=9 (18 calls total; [`P6-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L163)).

#### How each issue was resolved
The Section 6 discrepancy was resolved via Erratum (2026-10-01) in `docs/phase6_walkthrough.md` (commit `b8f78e7`) and marked `RESOLVED` in FACTS row [`P6-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L163) (commit `4b42532`), confirming raw telemetry as authoritative truth. Underpowered intervals were explicitly disclosed as known limitations rather than overstated.

#### Decision and why
Proved zero cache hazard on a sealed blind dataset. Formally closed the blind evaluation phase and established the honest finite-sample statistical boundaries.

#### What it does NOT prove
Does not mathematically prove $IRR_{\text{cache}} \le 10\%$ at 95% confidence due to small hit count ($n=8$, pooled $n=22$; [`KL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L214)). `AUTO_REUSE` latency is based on $n=1$ hit ([`LAT-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L194)).

---

### Phase 7: Repository Audit, Credential Hygiene & Verification Ledger

#### Objective
Audit repository security, verify absence of secrets in git history, construct a single authoritative fact ledger, resolve cross-document discrepancies, and build strict token-level verification tooling.

#### Method / steps
- **Phase 7a:** Scanned 23 commits across git history with Gitleaks (`docs/phase7a_secret_scan_raw.txt`). Grepped working tree for secret patterns. Extended `.gitignore` to cover `.env`, private keys, and local telemetry. Created `.env.example`. Documented findings in `docs/phase7a_walkthrough.md`.
- **Phase 7b:** Constructed single verified ledger [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) recording every constant, dataset size, empirical metric, CI, and limitation with explicit row IDs.
- **Phase 7c:** Audited all documented claims across `README.md`, `DESIGN.md`, and `demo/SCRIPT.md` in `docs/phase7c_claim_check.md`. Built strict token check `scripts/check_numeric_tokens.py` and built interactive demonstration `demo/run_demo.py`.

#### Data used
- Full git commit history (all 23 commits up to `57f4060` at scan time).
- All raw telemetry files (`data/load_test_telemetry.json`, `data/scaled_load_test_telemetry_corrected.json`, `data/phase6_telemetry.json`).

#### Results
- Git history secret scan: **0 leaks found** across all commits (exit code 0; `docs/phase7a_walkthrough.md`).
- Claim check verification: **all documented claims verified compliant** (100%) against [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) with **0 unledgered numeric tokens** across all audited docs (`docs/phase7c_claim_check.md`).

#### Issues faced
- **Credential Hygiene Risk:** A live OpenRouter API key had been present in `.env` within the local workspace. While Gitleaks proved it was never committed to git history, it had been visible in the local editing environment (`docs/phase7a_walkthrough.md` ss4, ss8).
- **Cross-Document Discrepancies:** Multiple documents contained outdated latency reduction percentages (99.79% instead of 99.20%; [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190), [`LAT-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L193)), mismatched confusion matrix counts (Phase 6 TP=6 vs TP=7; [`P6-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L163)), and broken code fence formatting in FACTS.md Appendix A.

#### How each issue was resolved
Recommended rotating the OpenRouter API key and committed hardened `.gitignore` rules in commit `2d282dd`. Corrected Phase 5 and Phase 6 walkthrough errata in commits `cb8946b` and `b8f78e7`. Reconciled FACTS.md telemetry and code fences in commits `8fa04c9` and `1efb9fc`. Implemented strict token verification in commit `48e2f5a`.

#### Decision and why
Anchored all project documentation to a single, immutable verified fact ledger, establishing complete scientific integrity and eliminating unverified claims.

#### What it does NOT prove
Auditing confirms the internal consistency and provenance of existing measurements; it does not collect new empirical query traffic or expand sample sizes.

---

## 5. Audit and Integrity Log

The following table documents every bug, discrepancy, and labeling artifact caught by auditing across the project lifecycle:

| Audit Item | How It Was Found | Before (Error State) | After (Corrected State) | Fix Commit | FACTS Row |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Discarded Fabricated/Simulated Phase 3 Results** | Recorded from auditor notes, not documented in repo | Discarded fabricated or simulated judge evaluations during early Phase 3 exploration | Only authentic live OpenRouter API responses accepted; zero simulation retained in repo | None (recorded from auditor notes, not documented in repo) | `docs/phase3_judge_call_walkthrough.md` |
| **Phase 4 Fallback Stub Contamination** | Data inspection of Session 0 synthetic feedback | 86 fail-closed fallback stubs (`fallback_triggered=True`) ingested as genuine negative labels | `load_combined_data()` filters fallback stubs; 108/108 genuine judge labels collected | `810f63e` | [`P4-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L106) |
| **Phase 5 Ground-Truth Labeling Collisions** | Error analysis of 10 false positives in N=338 scaled run | $IRR_{\text{cache}} = \mathbf{71.43\%}$ (10 FP / 14 hits) due to benchmark labeling collisions against seeds | 10 collisions relabeled MISS $\to$ HIT; $IRR_{\text{cache}} = \mathbf{0.00\%}$ (0 FP / 14 hits) | `f9e1545` | [`P5B-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L135), [`P5B-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L136), [`P5B-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L140) |
| **Phase 5 Scaled Run Latency Reduction Error** | Independent calculation from raw walkthrough means | Erroneously reported as **99.79%** (copied from N=70 pilot) | Recomputed from raw means (61.91 ms vs 7715.62 ms) to exact **99.20%** | `cb8946b` | [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190), [`LAT-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L193) |
| **Phase 6 Confusion Matrix Documentation Error** | Cross-checking walkthrough Section 6 text against raw telemetry | Walkthrough text stated **TP=6, TN=10** across 18 calls | Raw telemetry verified as source of truth with **TP=7, TN=9, FN=2, FP=0** | `b8f78e7`, `4b42532` | [`P6-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L163) |
| **Credential Hygiene & Secret Scan** | Phase 7a Gitleaks audit across full git history | `OPENROUTER_API_KEY` was present on local filesystem in `.env` and read during audit session | Full git history verified 100% clean (0 leaks across all 23 commits); `.gitignore` hardened and `.env.example` added in commit `2d282dd`; key rotation recommended at openrouter.ai as standard hygiene | `2d282dd` | `docs/phase7a_walkthrough.md` ss4, ss8 |

---

## 6. Timeline

The complete chronological development timeline across all phases, derived from git commit history:

- **Phase 1: Stability Classification Pipeline**
  - Commit `c851157`: Initial commit.
  - Commit `8590e18`: Project structure and test environment setup.
  - Commit `60fb844`: Reuse safety benchmarks and evaluation metrics definition.
  - Commit `25d54cd`: Test package marker addition.
  - Commit `c8bebb6`: Sub-stage A uncertainty-default safety fix (`STABLE_CONFIDENCE_THRESHOLD = 0.80`).
  - Commit `59efc32`: Threshold sweep documentation in confidence docblock.
  - Commit `dcf98f2`: Stability classification pipeline and evaluation suite.
  - Commit `56ce16d`: Dataset leakage fix and cross-dataset validation.
- **Phase 2: Semantic Cache Baseline**
  - Commit `07a36f7`: Semantic cache baseline, FAISS vector store, and threshold sweep.
  - Commit `7ea482b`: Offline model reproducibility fix pinning commit hash.
- **Phase 3: Reuse Decision Layer**
  - Commit `f555a93`: TierRouter implementation and linear combination evaluation.
  - Commit `ab033c0`: OpenRouter LLM judge adoption after evaluating 6 candidate strategies.
  - Commit `7f06e09`: Runtime dependency declarations.
- **Phase 4: Adaptive Policy & Calibration**
  - Commit `727837a`: Adaptive policy engine and Clopper-Pearson decision formulation.
  - Commit `810f63e`: Calibration label contamination fix and latency benchmark.
  - Commit `3fb6fad`: Real judge labeling session 1.
  - Commit `a765bbe`: Finalize 108/108 genuine judge labels and calibration verdict.
- **Phase 5: Scaled Load Testing & Collision Correction**
  - Commit `cad551e`: Pilot load test ($N=70$) and Streamlit dashboard.
  - Commit `9591101`: Step 0 pristine final test classifier evaluation.
  - Commit `2d123f3`: Scaled dataset ($N=338$) runner and benchmark generation.
  - Commit `f9e1545`: Corrected scaled load test and dual collision reporting.
  - Commit `c9d6a18`: Step 0b held-out challenge benchmark evaluation.
- **Phase 6: Blind Evaluation**
  - Commit `fc35744`: Author, collision-check, and seal blind evaluation dataset ($N=175$).
  - Commit `57f4060`: Blind evaluation results and walkthrough.
- **Phase 7: Security Audit, Verified Fact Ledger & Documentation Claim Check**
  - Commit `2d282dd`: Secret scan with Gitleaks, `.gitignore` hardening, and `.env.example`.
  - Commit `58d0e6d`: Verified fact ledger [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).
  - Commit `8fa04c9`: FACTS.md telemetry reconciliation and Appendix A code fence fix.
  - Commit `1efb9fc`: POOL numbering and 99.79 precision finalization.
  - Commit `b8f78e7`: Phase 6 walkthrough erratum (TP=7, TN=9).
  - Commit `cb8946b`: Phase 5 walkthrough erratum (99.20% latency reduction).
  - Commit `4b42532`: Resolve P6-08 in FACTS.md citing erratum.
  - Commit `875308d`: Append KL-10, KL-11, and Dataset Roles table to FACTS.md.
  - Commit `5914533`: Architecture design document `DESIGN.md`.
  - Commit `60c1977`: Comprehensive `README.md` rewrite with verified claims.
  - Commit `db39fd7`: Reconcile `DESIGN.md` numbers with FACTS.md.
  - Commit `e8daf17`: Reconcile `README.md` numbers with FACTS.md.
  - Commit `0e3e5c4`: Add `docs/phase7c_claim_check.md` verification matrix.
  - Commit `856a5df`: Add FACTS.md rows P5A-02b and P5B-02b with recount commands.
  - Commit `7194993`: Correct AUTO_REUSE latency range in README.md.
  - Commit `2f8c320`: Correct AUTO_REUSE latency range in DESIGN.md.
  - Commit `33adf6e`: Update claim check rows for latency and hit breakdowns.
  - Commit `090760f`: Refine AMBIGUOUS tier definition in DESIGN.md.
  - Commit `8e7de6e`: Interactive pipeline walkthrough `demo/run_demo.py`.
  - Commit `5ab093f`: Interview walkthrough script `demo/SCRIPT.md`.
  - Commit `00a6f5e`: MIT license and packaging configuration.
  - Commit `4681031`: Pin tier router citations in DESIGN.md.
  - Commit `480ba11`: Align SCRIPT.md latency with run_demo.py.
  - Commit `48e2f5a`: Enforce strict token-level numeric ledger verification.

> [!NOTE]
> **Commit SHA Normalization Note:** All git commit hashes in this report are normalized to 7 characters. For four commits (`4681031`, `5914533`, `7194993`, and `9591101`), their 7-character hexadecimal short forms consist entirely of decimal digits (`0-9`), colliding with digit-only numeric token extraction. Rather than altering or artificially lengthening these hashes, these four commit identifiers are explicitly ledgered in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L260) (`SHA-01` through `SHA-04`).

---

## 7. Consolidated Metrics Table

Every headline metric across the project lifecycle, strictly cited from [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md):

| Metric Description | Value | 95% Confidence Interval | Sample Size ($n$) | Evaluation Nature | FACTS Row ID |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Classifier Final Test Accuracy | **91.67%** | — | 60 | held-out | [`P1-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L68) |
| Classifier Final Test Dangerous Errors | **0** | — | 23 dynamic | held-out | [`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69) |
| Classifier Held-Out Challenge Accuracy | **91.67%** | — | 60 | held-out | [`P1-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L70) |
| Classifier Held-Out Challenge Dangerous Errors | **0** | — | 24 dynamic | held-out | [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71) |
| Classifier Final Test Conservative Error Rate | **13.51%** | — | 37 stable | held-out | [`P1-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L72) |
| Classifier Held-Out Challenge Conservative Error Rate | **13.89%** | — | 36 stable | held-out | [`P1-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L73) |
| Best Fixed Threshold Hazard Rate (@ 0.85) | **20.83%** | — | 24 hits (120 pairs) | same-set | [`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L74) |
| Best Fixed Threshold Actual Reuse Rate (@ 0.85) | **20.00%** | — | 120 pairs | same-set | [`P2-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L77) |
| Best Fixed Threshold Correct Reuse Precision (@ 0.85) | **79.17%** | — | 24 hits | same-set | [`P2-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L77) |
| Non-LLM Decision Step Hazard Rate | **52.17%** | — | 23 hits (120 pairs) | same-set | [`P3-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L87) |
| Ambiguous-Tier Non-LLM Hazard Rate | **57.1%** | — | 21 hits | same-set | [`P3-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L88) |
| AUTO_REUSE Tier Hazard Rate (Phase 3) | **0.00%** | — | 2 hits | same-set | [`P3-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L89) |
| Gemini Judge Hazard Rate (Partial: 17/32) | **0.00%** | — | 5 hits (120 pairs) | same-set | [`P3-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L90) |
| Gemini Judge Actual Reuse Rate (Partial) | **4.17%** | — | 120 pairs | same-set | [`P3-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L91) |
| OpenRouter Judge Hazard Rate ($IRR_{\text{cache}}$) | **8.33%** | [0.21%, 38.48%] | 12 hits (120 pairs) | same-set | [`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92) |
| OpenRouter Judge Actual Reuse Rate | **10.00%** | — | 120 pairs | same-set | [`P3-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L93) |
| Ambiguous Tier Traffic Fraction (Phase 3) | **26.7%** | — | 120 pairs | same-set | [`P3-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L94) |
| Pilot Run Hit Count (20 auto + 2 judge) | **22** | — | 70 queries | design-informed | [`P5A-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L120), [`P5A-02b`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L121) |
| Pilot Run Empirical Hazard Rate | **0.00%** | [0.00%, 15.44%] | 22 hits | design-informed | [`P5A-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L122), [`P5A-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L123) |
| Pilot Run Local Resolution Rate | **87.14%** | — | 70 queries | design-informed | [`P5A-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L124), [`LRR-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L202) |
| Pilot Run AUTO_REUSE Mean Latency | **19.18 ms** | — | 20 hits | design-informed | [`P5A-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L125), [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189) |
| Pilot Run AMBIGUOUS Judge Mean Latency | **9,291.35 ms** | — | 9 calls | design-informed | [`P5A-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L126), [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189) |
| Pilot Run Latency Reduction | **99.79%** (484.5x) | — | 70 queries | design-informed | [`P5A-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L127), [`LAT-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L189) |
| Scaled Run Pre-Correction Hazard Rate | **71.43%** | [41.90%, 91.61%] | 14 hits | design-informed | [`P5B-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L135), [`P5B-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L138) |
| Scaled Run Corrected Hit Count (1 auto + 13 judge) | **14** | — | 338 queries | design-informed | [`P5B-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L136), [`P5B-02b`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L137) |
| Scaled Run Post-Correction Hazard Rate | **0.00%** | [0.00%, 23.16%] | 14 hits | design-informed | [`P5B-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L136), [`P5B-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L139) |
| Scaled Run Local Resolution Rate | **86.39%** | — | 338 queries | design-informed | [`P5B-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L142), [`LRR-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L203) |
| Scaled Run AUTO_REUSE Mean Latency | **61.91 ms** | — | 1 hit | design-informed | [`P5B-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L143), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190) |
| Scaled Run AMBIGUOUS Judge Mean Latency | **7,715.62 ms** | — | 46 calls | design-informed | [`P5B-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L144), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190) |
| Scaled Run Latency Reduction | **99.20%** (124.6x) | — | 338 queries | design-informed | [`P5B-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L145), [`LAT-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L190) |
| Blind Run Hit Count (1 auto + 7 judge) | **8** | — | 175 entries | blind | [`P6-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L158) |
| Blind Run Empirical Hazard Rate ($IRR_{\text{cache}}$) | **0.00%** | [0.00%, 36.94%] | 8 hits | blind | [`P6-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L159), [`P6-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L160) |
| Blind Run Local Resolution Rate | **89.71%** | — | 175 entries | blind | [`P6-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L161), [`LRR-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L204) |
| Blind Run AUTO_REUSE Mean Latency | **10.56 ms** | — | 1 hit | blind | [`P6-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L164), [`LAT-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191) |
| Blind Run AMBIGUOUS Judge Mean Latency | **6,577.94 ms** | — | 18 calls | blind | [`P6-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L165), [`LAT-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191) |
| Blind Run Latency Reduction | **99.84%** (622.7x) | — | 175 entries | blind | [`P6-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L166), [`LAT-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191) |
| Pooled Phase 5 Corrected + Phase 6 Blind Hazard | **0.00%** | [0.00%, 15.44%] | 22 hits ($k=0$) | design-informed + blind | [`POOL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L174), [`POOL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L175), [`POOL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L176) |
| Sensitivity Three-Run Pooled Hazard (Pilot+Scaled+Blind) | **0.00%** | [0.00%, 8.04%] | 44 hits ($k=0$) | multi-run sensitivity (NOT BLIND) | [`POOL-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L179) |

---

## 8. Candidate Additional Metrics

Under strict governance rules, additional candidate metrics may only be ledgered if they meet four criteria:
1. Computable from existing raw telemetry files.
2. The comparison baseline exists within the same dataset (zero cross-dataset comparisons).
3. The definition is written out, a computation script is committed under `scripts/`, and output is added to `docs/FACTS.md` before use.
4. The evaluation label and confidence interval are stated.

Pursuant to Section 8 governance rules, the relative hazard reduction script has been committed under `scripts/compute_hazard_reduction.py` and its verified output ledgered in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L250) (`HR-01` through `HR-04`):

1. **Relative Reduction in Cache Hazard Rate (Same-Set Benchmark):**
   - *Definition:* Relative reduction in $IRR_{\text{cache}}$ from the Phase 2 best fixed threshold (0.85) to the Phase 3 OpenRouter LLM judge on the exact same 120 query pairs:
     $$\text{Relative Hazard Reduction} = \frac{IRR_{\text{fixed}} - IRR_{\text{judge}}}{IRR_{\text{fixed}}} = \frac{20.83\% - 8.33\%}{20.83\%} = 60.00\%$$
   - *Computation Script:* `scripts/compute_hazard_reduction.py` (reads Phase 2 and Phase 3 results on the same 120 pairs from `data/raw/query_pair_reuse_benchmark.json`).
   - *Baseline (Phase 2 Fixed 0.85):* 5 FP / 24 hits = **20.83%**, exact 95% Clopper-Pearson CI **[7.13%, 42.15%]** ([`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L76), [`HR-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L254)).
   - *Intervention (Phase 3 OpenRouter Judge):* 1 FP / 12 hits = **8.33%**, exact 95% Clopper-Pearson CI **[0.21%, 38.48%]** ([`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92)).
   - *Point Difference:* **12.50 percentage points** reduction ([`HR-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L255)).
   - *Relative Reduction:* **60.00%** ([`HR-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L256)).
   - *Confidence Interval Overlap & Statistical Significance:* The 95% Clopper-Pearson confidence intervals overlap ([7.13%, 42.15%] vs [0.21%, 38.48%]; [`HR-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L257)). The difference is **not statistically established** at 95% confidence.
   - *Evaluation Nature:* **same-set, CIs overlap**.

2. **Disqualified Candidates (Negative Rules):**
   - Any metric lacking a baseline within the same dataset (such as "judge invocations avoided per 100 queries", which has no comparative baseline in our experiments).
   - Any cost-in-dollars projection (e.g. hypothetical dollar savings).
   - Any unverified "percent safer than industry standard" assertion.
   - Any synthetic projection to live enterprise production volume.
   - Any comparison mixing denominators across different evaluation sets.

---

## 9. Limitations & Resolution Roadmap

The complete set of Known Limitations from [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) along with the engineering roadmap required to resolve each:

### [`KL-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L213): 6 Uncorrected Near-Duplicate Collisions in `new_dataset_v3.json`
- **Ledgered Detail:** Phase 5 collision analysis identified 27 pairwise collision candidates. 10 were surfaced as false positives and relabeled. Six additional near-duplicates remained undetected during execution (masked by BYPASS or StabilityClassifier uncertainty override): `cs_552`/`cs_005`, `fin_548`/`fin_006`, `fin_550`/`fin_003`, `sys_548`/`sys_004`, `sci_544`/`sci_002`, `fin_546`/`fin_005`. These were classified TN and not relabeled. The uncorrected records remain in `data/raw/new_dataset_v3.json`.
- **What is Needed to Close:** Run `scripts/run_new_dataset_load_test.py --dataset data/raw/new_dataset_v3.json --check-internal-collisions` to generate an updated `new_dataset_v4.json` where all 6 records are either explicitly re-annotated as positive reuse candidates or replaced with strictly disjoint queries.

### [`KL-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L214): Underpowered Confidence Intervals Throughout
- **Ledgered Detail:** Minimum $n \ge 36$ hits with $k=0$ FP is required to prove $IRR_{\text{cache}} \le 10\%$ at 95% Clopper-Pearson confidence. No single blind run achieved this: N=338 corrected has $n=14$, Phase 6 has $n=8$, pooled design-informed + blind has $n=22$. All blind CIs are underpowered. The three-run sensitivity ([`POOL-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L179), $n=44$) clears the ceiling but includes the development-stage N=70 pilot and is not a headline figure.
- **What is Needed to Close:** Execute an extended blind load test with a sealed query stream sized to produce at least 50 cache hits (e.g. an extended query stream containing at least 50 safe repeats and 50 near-boundary traps) to achieve $n \ge 36$ hits under blind conditions.

### [`KL-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L215): Free-Tier Judge Model Dependencies
- **Ledgered Detail:** All production LLM judge calls use `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter free tier. Subject to: (a) 50 RPD account limit, (b) provider-level edge caching, (c) possible model version updates at OpenRouter without notice. Paid-tier or self-hosted models were not evaluated.
- **What is Needed to Close:** Deploy a self-hosted open-weight judge model (e.g. vLLM instance serving Llama-3-70B or Nemotron locally) with deterministic seed pinning, fixed quantization, and zero external network latency.

### [`KL-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L216): Dual-Role Bias in Phase 4
- **Ledgered Detail:** `nvidia/nemotron-3-super-120b-a12b:free` was used both as the production ambiguous-tier decision judge (Phase 3 onward) AND as the ground-truth label generator for Phase 4 calibration data. Any systematic inductive bias propagates into both production inference and the training signal, without an independent external referee.
- **What is Needed to Close:** Procure human expert ground-truth annotations or an ensemble of frontier models (e.g. frontier model ensembles) as an independent calibration standard disjoint from the production judge.

### [`KL-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L217): No Blind Held-Out Set for Cache Pipeline Prior to Phase 6
- **Ledgered Detail:** The `StabilityClassifier` has a held-out final test set ($n=60$). The cache-reuse decision pipeline (`TierRouter` thresholds, similarity threshold, `LLMJudge` logic) had never been evaluated on data structurally unavailable to its designers before Phase 6. Phase 6 is the first genuinely blind evaluation but uses structured synthetic benchmarks, not organic query logs.
- **What is Needed to Close:** Ingest a production log stream of live multi-turn user queries from an external conversational agent deployment.

### [`KL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L218): Same-Set Evaluation for Phases 2 to 4
- **Ledgered Detail:** All of Phase 2, Phase 3, and Phase 4 were evaluated on `query_pair_reuse_benchmark.json` ($N=120$), which also informed boundary selection and calibration decisions. Results describe within-sample consistency, not generalization.
- **What is Needed to Close:** Retain `query_pair_reuse_benchmark.json` strictly as training/calibration material, and evaluate all pipeline versions on a secondary held-out set of query pairs.

### [`KL-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L219): FAISS Index Non-Persistence
- **Ledgered Detail:** `FlatVectorStore` resets between pair evaluations and does not persist across process restarts. Production deployments require index serialization.
- **What is Needed to Close:** Implement index serialization (`faiss.write_index` / `faiss.read_index`) coupled with an append-only write-ahead log (WAL) and key-value metadata store (e.g. SQLite or Redis).

### [`KL-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L220): Synthetic Workload Distributions
- **Ledgered Detail:** All load tests ($N=70$, $N=338$, $N=175$) use hand-crafted or StackExchange-sourced queries with intentional splits. Phase 6 walkthrough states: "whether these distributions match any specific production deployment is unknown and unverified."
- **What is Needed to Close:** Validate on a public real-world query trace (such as the LMSYS Chatbot Arena or ShareGPT traces) to evaluate performance under organic traffic distributions.

### [`KL-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L221): Diagnostic Set Leakage
- **Ledgered Detail:** 3 query strings overlap between `query_stability_benchmark.json` (dev/diagnostic, $N=160$) and `query_pair_reuse_benchmark.json`. Zero overlap with any pristine evaluation set. Documented and tested but not corrected.
- **What is Needed to Close:** Remove the 3 overlapping queries from the development set and re-verify cross-dataset disjunction.

### [`KL-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L222): Final Test Set Seen by Pytest Assertion
- **Ledgered Detail:** The classifier final test set's accuracy was "seen" once by a pytest assertion (`assert metrics.accuracy >= 0.85`) in `tests/test_stability_evaluation.py` (line 74) before the formal evaluation in `scripts/step0_final_test_eval.py`. The dataset was clean for hyperparameter tuning (no model weights or thresholds were fitted to it), but it was not strictly blind.
- **What is Needed to Close:** Generate a third, sealed test set for the stability classifier that is never checked into automated test assertions.

### [`KL-11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L223): Adaptive Threshold Mechanism Never Fired
- **Ledgered Detail:** All 7 categories stayed on the 0.85 fallback throughout calibration and evaluation. The per-category threshold fitting mechanism never lowered or altered an operating threshold in production (cite [`P4-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L102)).
- **What is Needed to Close:** Collect at least 100 labeled pairs per domain category to provide the sample size required for Clopper-Pearson upper bounds to clear 10% on permissive thresholds.

---

## 10. Reproduction Guide

To reproduce every headline metric recorded in this report, execute the following commands from the repository root in a verified Python environment ([`ENV-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L246)):

### Phase 1: Stability Classifier Verification
```powershell
# Final test evaluation (91.67% accuracy, 0 dangerous FP; P1-01, P1-02, P1-05)
python scripts/step0_final_test_eval.py

# Held-out challenge evaluation (91.67% accuracy, 0 dangerous FP; P1-03, P1-04, P1-06)
python scripts/step0b_heldout_test_eval.py
```

### Phase 2: Semantic Cache Baseline Sweep
```powershell
# Threshold sweep on 120 pairs (Best fixed threshold 0.85 -> 20.83% IRR; P2-01..P2-06)
python scripts/run_cache_threshold_sweep.py
```

### Phase 3: Ambiguous-Band Decision Layer Evaluation
```powershell
# Live OpenRouter judge evaluation on 120 pairs (8.33% IRR, 10.00% ARR; P3-06, P3-07)
python scripts/run_openrouter_eval.py

# Clopper-Pearson CI calculation for OpenRouter judge (k=1, n=12 -> [0.21%, 38.48%]; P3-06)
python -c "from scipy.stats import beta; print(f'[{beta.ppf(0.025,1,12)*100:.2f}%, {beta.ppf(0.975,2,11)*100:.2f}%]')"
```

### Phase 4: Adaptive Calibration Verification
```powershell
# Execute per-category calibration sweep (All 7 categories on 0.85 fallback; P4-01, P4-03)
python scripts/run_phase4_calibration.py

# Mathematical proof of n >= 36 hit floor for 10% ceiling at 95% confidence (P4-04)
python -c "import math; print(f'Required n: {math.ceil(math.log(0.025) / math.log(0.90))}')"
```

### Phase 5: Load Test & Collision Audit
```powershell
# Recompute N=70 pilot outcomes from raw telemetry (20 AUTO_REUSE + 2 judge; P5A-02, P5A-02b)
python -c "import json; from collections import Counter; d=json.load(open('data/load_test_telemetry.json')); print(Counter((e['tier'], e['outcome_type']) for e in d if e['outcome_type']=='TP'))"

# Recompute N=70 pilot 95% Clopper-Pearson CI (k=0, n=22 -> [0.00%, 15.44%]; P5A-04)
python -c "from scipy.stats import beta; print(f'[0.00%, {beta.ppf(0.975,1,22)*100:.2f}%]')"

# Scaled load test pre-correction evaluation (71.43% IRR; P5B-01)
python scripts/evaluate_load_test.py --telemetry data/scaled_load_test_telemetry.json

# Scaled load test post-correction evaluation (0.00% IRR; P5B-02)
python scripts/evaluate_load_test.py --telemetry data/scaled_load_test_telemetry_corrected.json

# Recompute N=338 scaled run corrected breakdown (1 AUTO_REUSE + 13 judge; P5B-02b)
python -c "import json; from collections import Counter; d=json.load(open('data/scaled_load_test_telemetry_corrected.json')); print(Counter((e['tier'], e['outcome_type']) for e in d if e['outcome_type']=='TP'))"

# Recompute N=338 scaled run 95% Clopper-Pearson CI (k=0, n=14 -> [0.00%, 23.16%]; P5B-04)
python -c "from scipy.stats import beta; print(f'[0.00%, {beta.ppf(0.975,1,14)*100:.2f}%]')"

# Execute internal collision checker on new_dataset_v3.json (Surfaces 27 candidates; P5B-06, KL-01)
python scripts/run_new_dataset_load_test.py --dataset data/raw/new_dataset_v3.json --check-internal-collisions
```

### Phase 6: Blind Evaluation & Pooled CIs
```powershell
# Recompute Phase 6 blind hit breakdown (1 AUTO_REUSE + 7 judge; P6-03)
python -c "import json; from collections import Counter; d=json.load(open('data/phase6_telemetry.json')); print(Counter((e['tier'], e['outcome_type']) for e in d if e['outcome_type']=='TP'))"

# Recompute Phase 6 blind 95% Clopper-Pearson CI (k=0, n=8 -> [0.00%, 36.94%]; P6-05)
python -c "from scipy.stats import beta; print(f'[0.00%, {beta.ppf(0.975,1,8)*100:.2f}%]')"

# Recompute Phase 6 judge tier outcome breakdown (TP=7, TN=9, FN=2, FP=0; P6-08)
python -c "import json; from collections import Counter; d=json.load(open('data/phase6_telemetry.json')); print(Counter(e['outcome_type'] for e in d if e['tier']=='AMBIGUOUS'))"

# Recompute Pooled Phase 5 + Phase 6 CI (k=0, n=22 -> [0.00%, 15.44%]; POOL-03)
python -c "from scipy.stats import beta; print(f'[0.00%, {beta.ppf(0.975,1,22)*100:.2f}%]')"

# Recompute Three-Run Sensitivity CI (k=0, n=44 -> [0.00%, 8.04%]; POOL-07)
python -c "from scipy.stats import beta; print(f'[0.00%, {beta.ppf(0.975,1,44)*100:.2f}%]')"
```

### Full Hermetic Test Suite
```powershell
# Run full unit and integration test suite (verified hermetic test suite, excluding model-integration tests)
pytest tests/ -m "not model_integration"
```
