# Adaptive Agentic Semantic Cache

> **Authoritative Documentation:** For the comprehensive project report, see [`docs/MASTER_REPORT.md`](docs/MASTER_REPORT.md). Every metric, threshold, latency measurement, and sample count in this repository originates directly from the authoritative ledger in [`docs/FACTS.md`](docs/FACTS.md). Historical or unverified status reports are not authoritative.

An adaptive semantic response reuse engine designed to eliminate redundant LLM computation safely. While naive similarity caches suffer high false-positive reuse rates by confusing syntactically similar prompts with semantically identical ones, this system guards cache access through a gated architecture: a lexical/rule-based **StabilityClassifier** that filters volatile and time-sensitive queries, a dense **SemanticCache** using `all-MiniLM-L6-v2` embeddings in FAISS, a dual-signal **TierRouter** evaluating both similarity and stability confidence, and an **LLMJudge** (`nvidia/nemotron-3-super-120b-a12b:free`) that reasons over ambiguous-tier queries with a strict fail-closed contract.

---

## Key Results

The five strongest empirical findings from the consolidated metrics table in [`docs/MASTER_REPORT.md`](docs/MASTER_REPORT.md), ordered strictly by evidentiary strength. Every metric preserves its verified evaluation nature label:

1. **Massive Latency Reduction on Cache Hits** (**design-informed** & **blind**; [`LAT-05`](docs/FACTS.md))
   When queries meet high-confidence cache reuse criteria (`AUTO_REUSE`), vector retrieval delivers an end-to-end response in **10.56–61.91 ms** compared to **6,577.94–9,291.35 ms** for ambiguous-tier remote LLM judge calls, yielding a **99.20% to 99.84% latency reduction** (**124.6x to 622.7x speedup**) across all three load test runs ([`LAT-01`..`LAT-05`](docs/FACTS.md)).
   > *Sample Size Caveat ([`LAT-06`](docs/FACTS.md)):* In the scaled $N=338$ and blind $N=175$ runs, exactly $n=1$ query triggered the `AUTO_REUSE` path in each run (61.91 ms and 10.56 ms, respectively). The Phase 5 $N=70$ pilot produced $n=20$ `AUTO_REUSE` hits (mean 19.18 ms) and represents our most statistically reliable latency distribution.

2. **Zero False Positives on Sealed Blind Evaluation** (**blind**; [`P6-04`](docs/FACTS.md))
   Across 175 blind evaluation queries across 7 domains sealed prior to execution at commit `fc35744` ([`D-09`](docs/FACTS.md), [`P6-01`](docs/FACTS.md), [`P6-02`](docs/FACTS.md)), the pipeline admitted 8 cache hits (1 `AUTO_REUSE`, 7 judge-approved; [`P6-03`](docs/FACTS.md)) with **0 false positives**, yielding an empirical hazard rate ($IRR_{\text{cache}}$) of **0.00%** with an exact 95% Clopper-Pearson confidence interval of **[0.00%, 36.94%]** ([`P6-05`](docs/FACTS.md)).

3. **Same-Set Incorrect-Reuse Reduction (Calibration Benchmark)** (**same-set**; [`P2-01`](docs/FACTS.md), [`P3-06`](docs/FACTS.md))
   On the 120-pair baseline calibration benchmark, the best fixed cosine similarity threshold (0.85) failed the 10% safety ceiling with an empirical hazard rate of **20.83%** (5 FP / 24 hits, 95% CI [7.13%, 42.15%]; [`P2-01`](docs/FACTS.md), [`HR-01`](docs/FACTS.md)). Integrating the OpenRouter LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`) for ambiguous pairs reduced the same-set point estimate to **8.33%** (1 FP / 12 hits, 95% CI [0.21%, 38.48%]; [`P3-06`](docs/FACTS.md)), achieving a 12.50 percentage point reduction ([`HR-02`](docs/FACTS.md)) and a 60.00% relative hazard reduction ([`HR-03`](docs/FACTS.md)).
   > *Statistical Significance & Calibration Note ([`HR-04`](docs/FACTS.md), [`KL-06`](docs/FACTS.md)):* The 95% Clopper-Pearson confidence intervals overlap ([7.13%, 42.15%] vs [0.21%, 38.48%]), meaning the reduction is **same-set** and **not statistically established** at 95% confidence. This point estimate served architecture selection, not a proven blind safety bound.

4. **Classifier Held-Out Accuracy and Zero Dangerous Errors** (**held-out**; [`P1-01`..`P1-04`](docs/FACTS.md))
   The two-stage `StabilityClassifier` achieved **91.67% accuracy** (55/60 correct) on both the pristine held-out challenge benchmark ([`P1-03`](docs/FACTS.md)) and the final test benchmark ([`P1-01`](docs/FACTS.md)), committing **0 dangerous errors** across all dynamic queries in both sets (FP=0 on 24 challenge dynamic queries, FP=0 on 23 final test dynamic queries; [`P1-02`](docs/FACTS.md), [`P1-04`](docs/FACTS.md)), while maintaining a low conservative false rejection rate (13.51% final test, 13.89% challenge; [`P1-05`](docs/FACTS.md), [`P1-06`](docs/FACTS.md)).

5. **High Local Resolution Rate Resolved Without a Judge Call** (**design-informed** & **blind**; [`LRR-04`](docs/FACTS.md))
   Across all three load tests, **86.39% to 89.71%** of queries were resolved locally on-device without invoking a slow, costly remote LLM judge call (87.14% in the $N=70$ pilot, 86.39% in the $N=338$ scaled run, and 89.71% in the $N=175$ blind evaluation; [`LRR-01`..`LRR-03`](docs/FACTS.md)).

---

## Honest Limitations

- **Statistical Power Limits:** To mathematically guarantee that the true cache hazard rate does not exceed 10% under a 95% Clopper-Pearson confidence interval when observing zero errors ($k=0$), an evaluation must observe at least **$n \ge 36$ hits** ([`P4-04`](docs/FACTS.md)). The blind run yielded $n=8$ hits (upper bound 36.94%; [`P6-05`](docs/FACTS.md)), and pooling the scaled run ($n=14$) and blind run ($n=8$) yielded $n=22$ hits with $k=0$ (upper bound **15.44%**; [`POOL-01`..`POOL-03`](docs/FACTS.md)). Because 15.44% remains above the 10% safety ceiling ([`C-13`](docs/FACTS.md)), the 10% ceiling is not certified under blind evaluation.
- **Adaptive Mechanism Fallback:** The per-category adaptive threshold engine never altered an operating threshold in production; all 7 categories remained on the global 0.85 fallback due to finite-sample safety gates ([`P4-01`](docs/FACTS.md), [`KL-11`](docs/FACTS.md)).
- **Uncorrected Benchmark Collisions:** 6 undetected near-duplicate collisions identified in Phase 5 remain uncorrected in `data/raw/new_dataset_v3.json` ([`P5B-06`](docs/FACTS.md), [`KL-01`](docs/FACTS.md)).
- **Synthetic Distributions:** Evaluations utilized structured synthetic streams and StackExchange seeds rather than organic enterprise customer traffic ([`KL-08`](docs/FACTS.md)).
- **Free-Tier Infrastructure:** The production judge relies on OpenRouter's free tier (`nvidia/nemotron-3-super-120b-a12b:free`), which imposes a 50 requests-per-day quota limit ([`KL-03`](docs/FACTS.md)) and introduces dual-role bias in Phase 4 calibration ([`KL-04`](docs/FACTS.md)).
- **In-Memory Store:** The FAISS index operates strictly in-memory without disk persistence across restarts ([`KL-07`](docs/FACTS.md)).

---

## Verified Results Summary

The table below summarizes measured empirical results across all phases of the project. Each evaluation is explicitly labeled by the nature of its data exposure: **same-set** (development/calibration data), **held-out** (unseen during tuning), or **blind** (operationally sealed prior to execution).

| Pipeline Stage / Run | Dataset Role | N | Cache Hits (n) | False Positives (k) | Hazard Rate ($IRR_{\text{cache}}$) | 95% Clopper-Pearson CI | Local Resolution Rate | FACTS.md Rows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **StabilityClassifier Final Test** | **Held-out** (seen once by a pytest assertion, see [`KL-10`](docs/FACTS.md)) | 60 | — | 0 dangerous FP | 0.00% FP on dynamic | — | — | [`P1-01`](docs/FACTS.md), [`P1-02`](docs/FACTS.md), [`D-04`](docs/FACTS.md) |
| **StabilityClassifier Challenge** | **Held-out** | 60 | — | 0 dangerous FP | 0.00% FP on dynamic | — | — | [`P1-03`](docs/FACTS.md), [`P1-04`](docs/FACTS.md), [`D-03`](docs/FACTS.md) |
| **Fixed Threshold Sweep (Best: 0.85)** | **Same-set** | 120 pairs | 24 | 5 | 20.83% | — | — | [`P2-01`](docs/FACTS.md), [`P2-04`](docs/FACTS.md), [`D-01`](docs/FACTS.md) |
| **Phase 3 Non-LLM Decision Step** | **Same-set** | 120 pairs | 23 | 12 | 52.17% | — | — | [`P3-01`](docs/FACTS.md), [`D-01`](docs/FACTS.md) |
| **Phase 3 OpenRouter Nemotron Judge** | **Same-set** (calibration benchmark) | 120 pairs | 12 | 1 | 8.33% | [0.21%, 38.48%] | — | [`P3-06`](docs/FACTS.md), [`P3-07`](docs/FACTS.md), [`KL-06`](docs/FACTS.md) |
| **Phase 5 Pilot Load Test** | **Same-set** (design-informed) | 70 | 22 (20 auto, 2 judge) | 0 | 0.00% | [0.00%, 15.44%] | 87.14% | [`P5A-01`..`P5A-05`](docs/FACTS.md), [`D-07`](docs/FACTS.md) |
| **Phase 5 Scaled Load Test (Initial)** | **Same-set** (design-informed) | 338 | 14 | 10 | 71.43% | [41.90%, 91.61%] | 86.39% | [`P5B-01`](docs/FACTS.md), [`P5B-03`](docs/FACTS.md), [`D-08`](docs/FACTS.md) |
| **Phase 5 Scaled Load Test (Corrected)** | **Same-set** (collision-audited) | 338 | 14 (1 auto, 13 judge) | 0 | 0.00% | [0.00%, 23.16%] | 86.39% | [`P5B-02`](docs/FACTS.md), [`P5B-04`](docs/FACTS.md), [`P5B-07`](docs/FACTS.md) |
| **Phase 6 Blind Evaluation** | **Blind** (sealed commit `fc35744`, [`P6-02`](docs/FACTS.md)) | 175 | 8 (1 auto, 7 judge) | 0 | **0.00%** | **[0.00%, 36.94%]** | **89.71%** | [`P6-01`..`P6-06`](docs/FACTS.md), [`D-09`](docs/FACTS.md) |
| **Pooled Blind + Corrected Pilot** | **Design-informed + blind** (Phase 5 corr. + Phase 6 blind) | — | 22 | 0 | **0.00%** | **[0.00%, 15.44%]** | — | [`POOL-01`..`POOL-06`](docs/FACTS.md) |

---

## Quickstart

### 1. Installation

Install dependencies into a Python virtual environment (tested and verified with Python 3.14.5 in fresh-clone test):

```bash
git clone https://github.com/paridhik11/adaptive-agentic-semantic-cache.git
cd adaptive-agentic-semantic-cache
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Environment Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```

Edit `.env` to provide your OpenRouter API key for remote judge evaluation (optional for replay demo and fast tests):
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```
*(If unset or invalid, the pipeline adheres to its fail-closed invariant and routes ambiguous queries safely to BYPASS.)*

### 3. Run Pipeline Demo & Fast Test Suite

Run the interactive demonstration in replay mode (no API key required):
```bash
python demo/run_demo.py
```

Run the default hermetic test suite (network-free):
```bash
pytest
```

Run the Phase 5 load test demonstration:
```bash
python scripts/run_load_test.py
```

Evaluate the resulting telemetry:
```bash
python scripts/evaluate_load_test.py --telemetry data/load_test_telemetry.json
```

---

## Repository Layout

```text
adaptive-agentic-semantic-cache/
├── DESIGN.md                                 # Full architecture, design decisions, and math
├── README.md                                 # Project overview and empirical results
├── pyproject.toml                            # Dependencies and test markers
├── .env.example                              # Environment variable configuration template
├── src/
│   ├── classifier/                           # StabilityClassifier (rules + lexical model)
│   ├── cache/                                # Sentence-transformers embedding & FAISS store
│   ├── decision/                             # TierRouter, JudgeDecisionStep, AdaptiveThresholdEngine
│   └── evaluation/                           # Metrics calculation & Clopper-Pearson routines
├── data/
│   ├── raw/                                  # Pristine and synthetic benchmark datasets
│   └── *.json                                # Telemetry logs from experimental runs
├── docs/
│   ├── FACTS.md                              # Single authoritative ledger of verified facts
│   ├── phase7c_claim_check.md                # Mapping of every claim to its FACTS.md row
│   ├── phase5_walkthrough.md                 # Execution log for Phase 5 load tests
│   └── phase6_walkthrough.md                 # Execution log for Phase 6 blind evaluation
├── scripts/                                  # Reproducible execution scripts per phase
└── tests/                                    # Unit, integration, and evaluation test suites
```

---

## How to Reproduce Each Phase

Every phase of the project is backed by standalone reproduction scripts cited in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md):

- **Phase 1 — Classifier Evaluation:**
  ```bash
  python scripts/step0_final_test_eval.py      # Final test benchmark (91.67% accuracy; FACTS P1-01)
  python scripts/step0b_heldout_test_eval.py    # Held-out challenge benchmark (91.67% accuracy; FACTS P1-03)
  ```
- **Phase 2 — Fixed Threshold Sweep:**
  ```bash
  python scripts/run_cache_threshold_sweep.py   # Baseline sweep across 120 pairs (best 20.83% IRR; FACTS P2-01)
  ```
- **Phase 3 — Ambiguous-Tier Judge Evaluation:**
  ```bash
  python scripts/run_openrouter_eval.py         # Evaluates OpenRouter judge on 120 pairs (8.33% IRR; FACTS P3-06)
  ```
- **Phase 4 — Adaptive Threshold Calibration:**
  ```bash
  python scripts/run_phase4_calibration.py      # Sweeps 7 categories (all stay on 0.85 fallback; FACTS P4-01)
  ```
- **Phase 5 — Load Tests & Scaled Evaluation:**
  ```bash
  python scripts/run_load_test.py               # N=70 pilot load test (FACTS P5A-01)
  python scripts/evaluate_load_test.py --telemetry data/load_test_telemetry.json
  python scripts/evaluate_load_test.py --telemetry data/scaled_load_test_telemetry_corrected.json # N=338 (FACTS P5B-02)
  ```
- **Phase 6 — Sealed Blind Evaluation:**
  ```bash
  python scripts/evaluate_load_test.py --telemetry data/phase6_telemetry.json # N=175 blind run (FACTS P6-04)
  ```

---

## Further Documentation

- [`docs/MASTER_REPORT.md`](docs/MASTER_REPORT.md): Authoritative Master Project Report containing the full project narrative, complete phase-by-phase walkthroughs, consolidated metrics, audited commit log, and limitations.
- [`docs/FACTS.md`](docs/FACTS.md): Authoritative fact ledger with exact line citations and CLI reproduction steps for every metric.
- [`DESIGN.md`](DESIGN.md): Detailed discussion of the failure modes of naive caching, four-stage architecture, tier routing rules, the adaptive mechanism analysis, self-audited bugs, and complete limitation disclosures.
- [`docs/phase7c_claim_check.md`](docs/phase7c_claim_check.md): Comprehensive verification matrix mapping all numbers in documentation to [`docs/FACTS.md`](docs/FACTS.md).