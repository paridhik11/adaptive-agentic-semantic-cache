# Adaptive Agentic Semantic Cache

> **Authoritative Fact Source:** Every metric, threshold, latency measurement, and sample count in this repository originates directly from [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md). Historical or unverified status reports are not authoritative.

An adaptive semantic response reuse engine designed to eliminate redundant LLM computation safely. While naive similarity caches suffer high false-positive reuse rates by confusing syntactically similar prompts with semantically identical ones, this system guards cache access through a gated architecture: a lexical/rule-based **StabilityClassifier** that filters volatile and time-sensitive queries, a dense **SemanticCache** using `all-MiniLM-L6-v2` embeddings in FAISS, a dual-signal **TierRouter** evaluating both similarity and stability confidence, and an **LLMJudge** (`nvidia/nemotron-3-super-120b-a12b:free`) that reasons over ambiguous-tier queries with a strict fail-closed contract.

---

## Verified Results Summary

The table below summarizes measured empirical results across all phases of the project. Each evaluation is explicitly labeled by the nature of its data exposure: **same-set** (development/calibration data), **held-out** (unseen during tuning), or **blind** (operationally sealed prior to execution).

| Pipeline Stage / Run | Dataset Role | N | Cache Hits (n) | False Positives (k) | Hazard Rate ($IRR_{\text{cache}}$) | 95% Clopper-Pearson CI | Local Resolution Rate | FACTS.md Rows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **StabilityClassifier Final Test** | **Held-out** (seen once by a pytest assertion, see [`KL-10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L220)) | 60 | — | 0 dangerous FP | 0.00% FP on dynamic | — | — | [`P1-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L68), [`P1-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L69), [`D-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L55) |
| **StabilityClassifier Challenge** | **Held-out** | 60 | — | 0 dangerous FP | 0.00% FP on dynamic | — | — | [`P1-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L70), [`P1-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L71), [`D-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L54) |
| **Fixed Threshold Sweep (Best: 0.85)** | **Same-set** | 120 pairs | 24 | 5 | 20.83% | — | — | [`P2-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L74), [`P2-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L77), [`D-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52) |
| **Phase 3 Non-LLM Decision Step** | **Same-set** | 120 pairs | 23 | 12 | 52.17% | — | — | [`P3-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L87), [`D-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L52) |
| **Phase 3 OpenRouter Nemotron Judge** | **Same-set** (calibration benchmark) | 120 pairs | 12 | 1 | 8.33% | [0.21%, 38.48%] | — | [`P3-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92), [`P3-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L93), [`KL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L216) |
| **Phase 5 Pilot Load Test** | **Same-set** (design-informed) | 70 | 22 (20 auto, 2 judge) | 0 | 0.00% | [0.00%, 15.44%] | 87.14% | [`P5A-01`..`P5A-05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L119-L123), [`D-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L58) |
| **Phase 5 Scaled Load Test (Initial)** | **Same-set** (design-informed) | 338 | 14 | 10 | 71.43% | [41.90%, 91.61%] | 86.39% | [`P5B-01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L134), [`P5B-03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L136), [`D-08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L59) |
| **Phase 5 Scaled Load Test (Corrected)** | **Same-set** (collision-audited) | 338 | 14 (1 auto, 13 judge) | 0 | 0.00% | [0.00%, 23.16%] | 86.39% | [`P5B-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L135), [`P5B-04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L137), [`P5B-07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L140) |
| **Phase 6 Blind Evaluation** | **Blind** (sealed commit `fc35744`, [`P6-02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L155)) | 175 | 8 (1 auto, 7 judge) | 0 | **0.00%** | **[0.00%, 36.94%]** | **89.71%** | [`P6-01`..`P6-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L154-L159), [`D-09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L60) |
| **Pooled Blind + Corrected Pilot** | **Pooled** (Phase 5 corr. + Phase 6 blind) | — | 22 | 0 | **0.00%** | **[0.00%, 15.44%]** | — | [`POOL-01`..`POOL-06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L172-L176) |

---

## Headline Findings

1. **Latency Reduction: 99.20% to 99.84%** ([`FACTS.md` row LAT-05](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L191))
   When queries meet the high-confidence auto-reuse criteria, vector cache retrieval delivers an end-to-end response in **10.56–61.91 ms** (means of the three runs: LAT-03, LAT-01, LAT-02; [`FACTS.md` rows LAT-01..LAT-03](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L187-L189)) compared to 6,577.94–9,291.35 ms for ambiguous-tier remote LLM judge calls, yielding speedups of **124.6x to 622.7x** ([`FACTS.md` rows LAT-01..LAT-04](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L187-L190)).
   > **Sample Size Caveat ([`FACTS.md` row LAT-06](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L192)):** In the scaled N=338 and blind N=175 runs, exactly $n=1$ query triggered the AUTO_REUSE path in each run (61.91 ms and 10.56 ms, respectively). The Phase 5 N=70 pilot produced $n=20$ AUTO_REUSE hits (mean 19.18 ms) and represents our most statistically reliable latency benchmark.
2. **Zero False Positives on Blind Data ($IRR_{\text{cache}} = 0.00\%$)** ([`FACTS.md` row P6-04](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L157))
   Across 175 blind evaluation queries sealed prior to execution ([`FACTS.md` rows D-09, P6-01, P6-02](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L60)), the pipeline produced 8 cache hits (1 AUTO_REUSE, 7 judge-approved; [`FACTS.md` row P6-03](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L156)) with **0 false positives**, yielding an empirical hazard rate of 0.00% with exact 95% Clopper-Pearson CI **[0.00%, 36.94%]** ([`FACTS.md` row P6-05](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L158)).
3. **Pooled CI [0.00%, 15.44%] Remains Above the 10% Safety Ceiling** ([`FACTS.md` rows POOL-03, POOL-06](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L174-L176))
   Combining verified cache hits from the Phase 5 corrected run ($n=14$) and Phase 6 blind run ($n=8$) yields $n=22$ hits with $k=0$ false positives ([`FACTS.md` rows POOL-01, POOL-02](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L172-L173)). The exact 95% Clopper-Pearson upper bound is **15.44%**. While zero empirical errors occurred, **15.44% is still above the 10% safety ceiling** ([`FACTS.md` row C-13](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L41)) because proving an upper bound $\le 10\%$ mathematically requires observing at least $n \ge 36$ hits with $k=0$ ([`FACTS.md` row P4-04](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L105)).
4. **Judge 8.33% IRR is Same-Set Calibration, Not a Proven Blind Safety Bound**
   In Phase 3, the OpenRouter Nemotron judge achieved an $IRR_{\text{cache}}$ of **8.33%** (1 FP / 12 hits; [`FACTS.md` row P3-06](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92)). This is a **same-set point estimate** measured on the 120-pair development benchmark ([`FACTS.md` row KL-06](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L216)), with an exact 95% Clopper-Pearson CI of **[0.21%, 38.48%]** ([`FACTS.md` row P3-06](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L92)). It was used to select the judge architecture over non-LLM baselines, but must not be cited as a proven blind safety result.

---

## Quickstart

### 1. Installation

Install dependencies into a Python virtual environment:

```bash
git clone https://github.com/paridhik11/adaptive-agentic-semantic-cache.git
cd adaptive-agentic-semantic-cache
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

To include development and test dependencies:
```bash
pip install -e .[dev]
```

### 2. Environment Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```

Edit `.env` to provide your OpenRouter API key for remote judge evaluation:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```
*(If unset or invalid, the pipeline adheres to its fail-closed invariant and routes ambiguous queries safely to BYPASS.)*

### 3. Run Pipeline Demo & Fast Test Suite

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

- [`DESIGN.md`](DESIGN.md): Detailed discussion of the failure modes of naive caching, four-stage architecture, tier routing rules, the adaptive mechanism analysis, self-audited bugs, and complete limitation disclosures.
- [`docs/FACTS.md`](docs/FACTS.md): Authoritative fact ledger with exact line citations and CLI reproduction steps for every metric.
- [`docs/phase7c_claim_check.md`](docs/phase7c_claim_check.md): Comprehensive verification matrix mapping all numbers in documentation to [`docs/FACTS.md`](docs/FACTS.md).