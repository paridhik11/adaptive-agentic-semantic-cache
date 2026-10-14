# Phase A Pre-Registration Document

**Study Title:** Rigorous Out-of-Distribution Evaluation of Semantic Equivalence Matching on Unseen Multi-Domain Query Streams  
**Protocol Version:** 1.0.0  
**Registration Date:** 2026-10-12  
**Evaluation Mode:** Sealed Out-of-Sample Test (Single Run, Zero Tuning, Strict Fail-Closed Verification)  
**Authors:** Adaptive Agentic Semantic Cache Engineering Team  

---

## 1. Background and Motivation

In Phase 8, the caching pipeline folded exact query repetitions with 0% error, but non-exact semantic matching demonstrated vulnerabilities:
1. Under adversarial word-order permutations (PAWS), bi-encoder cosine similarities exceeded 0.92, bypassing rule guards in the `AUTO_REUSE` tier and producing 3 false positives.
2. Bag-of-words token set guards (numbers, negations, entity sets) were structurally unable to distinguish word-scrambled negatives with identical multisets.
3. The Phase 8 evaluation was conducted on a self-constructed test stream ($N=230$) where 82% of hits were exact/canonical folds, yielding an overall IRR of 6.0% with a 95% Clopper-Pearson CI of [1.3%, 16.5%], which did not formally establish the 10% safety ceiling.

Phase A establishes a sound semantic verification architecture:
- Candidate (c): Fine-tuned pairwise CrossEncoder (`models/phaseA_cross_encoder_finetuned`) trained on QQP and PAWS hard negatives, combined with deterministic rule guards (numbers, negations, entities, long-prompt window limits, and open-ended generative request bypass).
- All prospective reuses—including candidates with similarity $\ge 0.92$—must satisfy all rule guards and pairwise local equivalence checks.
- Pre-registering endpoints, frozen configurations, and statistical hypotheses prevents p-hacking, boundary shifting, or post-hoc data curation.

---

## 2. Hypotheses and Primary/Secondary Endpoints

### 2.1 Primary Endpoint: Non-Exact Incorrect-Reuse Rate (Safety Criterion)
- **Definition:** The proportion of false positive reuses among all cache hits that were **not** resolved by trivial exact string match or Stage 0 canonical frame folding:
  $$\text{IRR}_{\text{non-exact}} = \frac{\text{FP}_{\text{non-exact}}}{\text{Hits}_{\text{non-exact}}} = \frac{\text{FP}_{\text{non-exact}}}{\text{TP}_{\text{non-exact}} + \text{FP}_{\text{non-exact}}}$$
- **Statistical Metric:** Exact two-sided 95% Clopper-Pearson confidence interval:
  $$\text{CI}_{95\%} = \left[ \text{Beta}^{-1}(0.025; k, n - k + 1), \; \text{Beta}^{-1}(0.975; k + 1, n - k) \right]$$
- **Pre-Registered Acceptance Criteria:**
  - **Gate A Safety Minimum:** Point estimate $\text{IRR}_{\text{non-exact}} \le 10\%$.
  - **Ideal Statistical Target:** Upper bound of the 95% Clopper-Pearson CI on non-exact incorrect reuse $\le 10\%$.
  - **Statistical Power Requirement:** At least 100 non-exact hits ($n \ge 100$) must be evaluated on the sealed stream to ensure the evaluation is adequately powered.

### 2.2 Secondary Endpoints: Utility and Efficiency
1. **Overall Hit Rate:**
   $$\text{Hit Rate} = \frac{\text{Total Hits}}{N_{\text{stream}}}$$
   Reported with exact 95% Clopper-Pearson CI.
2. **Workload Avoidance:**
   - LLM generation calls avoided per 1,000 incoming queries.
   - LLM prompt and completion tokens avoided per 1,000 incoming queries (referenced to measured $n=100$ baseline: 72.92 tokens/query on local Qwen2.5-0.5B-Instruct).
3. **End-to-End Latency vs. No-Cache Baseline:**
   - End-to-end p50 and p95 latency across all stream queries (decision time for hits; decision time + generation latency for misses).
   - Local path latency percentiles by resolution path (`CANONICAL_EXACT`, `AUTO_REUSE`, `LOCAL_EQUIVALENCE`, `BYPASS`).
4. **Break-Even Analysis:**
   - Break-even hit rate where caching computational overhead equals total LLM generation time saved.
   - Hit rate vs. repeat-share curve under 0%, 10%, 20%, 40%, and 60% workload repeat shares.

---

## 3. Sealed Test Stream Specification

The sealed stream ($N = 2,200$ queries) is constructed exclusively from public, out-of-sample held-out splits that were strictly withheld from all prior prompt tuning, boundary sweeps, and model fine-tuning:

| Slice | Source Dataset | Split / Slice Details | Queries | Intended Role |
| :--- | :--- | :--- | :---: | :--- |
| **QQP Held-Out** | Quora Question Pairs (`nyu-mll/glue`, `qqp`) | Validation slice (pairs 500–1000) not used in A1 tuning | **1,000** (500 pairs) | Diverse natural question paraphrases and non-equivalent pairs |
| **PAWS Test Split** | PAWS (`google-research-datasets/paws`) | Official test split (`test.tsv`), pristine | **600** (300 pairs) | 150 real paraphrases + 150 adversarial word-scramble negatives |
| **CQADupStack** | CQADupStack (`mteb/cqadupstack-retrieval`) | Held-out threads across 12 subforums | **400** (200 pairs) | Multi-domain community duplicate technical questions |
| **WildChat Traffic** | WildChat (`allenai/WildChat`) | Sampled organic first-turn user turns | **200** (100 pairs) | Natural conversational user prompt distribution |
| **Total Stream** | **Public Held-Out Multi-Domain** | **100% Out-of-Sample Held-Out** | **2,200** | **Target >= 200 non-exact hits possible** |

### Leakage and Independence Guarantees
1. Zero training overlap: No pair in the sealed stream appears in `data/candidate_b_train_pairs.json` or `data/phase8_tuning_val.json`.
2. Hashing and pre-commitment: The sealed stream file (`data/raw/phaseA_sealed_test_stream.json`) is cryptographically digested (SHA-256) and committed to git together with this pre-registration document before any pipeline run is executed.
3. Single-execution policy: The pipeline is executed exactly once on this stream. No parameter tuning, threshold changes, or post-hoc query exclusions are permitted.

---

## 4. Frozen Pipeline Configuration

The evaluation pipeline is frozen to the following exact components and parameter configurations:

```json
{
  "phase": "A",
  "pipeline_name": "Phase A Local Equivalence Verified Pipeline",
  "bi_encoder": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "similarity_metric": "cosine"
  },
  "normalizer": {
    "enabled": true,
    "strategy": "definitional_frame_stripping_and_edge_punct",
    "class": "CanonicalCache"
  },
  "rule_guards": {
    "number_guard": true,
    "negation_guard": true,
    "entity_guard": true,
    "long_prompt_guard": {
      "enabled": true,
      "max_tokens": 128,
      "fallback_policy": "exact_or_canonical_only"
    },
    "generative_request_guard": {
      "enabled": true,
      "benchmark_precision": 1.0,
      "policy": "fail_closed_bypass"
    }
  },
  "tier_router": {
    "auto_reuse_threshold": 0.92,
    "auto_reuse_requires_guards": true,
    "ambiguous_lower_bound": 0.50,
    "bypass_below": 0.50
  },
  "cross_encoder": {
    "model_path": "models/phaseA_cross_encoder_finetuned",
    "base_architecture": "cross-encoder/quora-distilroberta-base",
    "decision_threshold": 0.85,
    "requires_guards": true,
    "fallback_to_llm": false
  },
  "baseline_generation_reference": {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "mean_latency_ms": 8997.49,
    "median_latency_ms": 8153.50,
    "p95_latency_ms": 12763.96,
    "mean_tokens": 72.92
  }
}
```

---

## 5. Execution and Analysis Protocol

1. **Pipeline Execution Command:**
   ```powershell
   python scripts/run_phaseA_sealed_evaluation.py
   ```
2. **Analysis Output Artifacts:**
   - Telemetry JSON: `data/phaseA_sealed_evaluation_telemetry.json`
   - Evaluation Summary JSON: `data/phaseA_sealed_evaluation_summary.json`
   - Walkthrough Report: `docs/phaseA_walkthrough.md`
   - Claim Check Matrix: `docs/claim_check_A.md`
   - Phase A Audit Pack: `docs/audit/AUDIT_PACK_A.md`

3. **Cryptographic Checksums (Frozen Commit Artifacts):**
   - Sealed Stream (`data/raw/phaseA_sealed_test_stream.json`): `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991`
   - Frozen Config (`data/phaseA_frozen_config.json`): `a1e0dc2a0a2e89e8c29a357597264f2d47fbf556f3f531b9a0b5c760f2fd8466`

4. **Stopping Condition (Gate A):**
   - If non-exact incorrect reuse point estimate exceeds 10%, or fewer than 100 non-exact hits occur: write `docs/BLOCKED_A.md` and immediately halt execution.
   - If all Gate A criteria pass: document results, update `docs/STATUS.md`, and advance to Phase B.
