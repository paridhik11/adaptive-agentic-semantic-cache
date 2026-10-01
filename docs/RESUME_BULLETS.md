# Resume Bullets: Adaptive Agentic Semantic Cache

> **Target Role:** Software Engineering (SWE) / Machine Learning (ML) Internship (Applied ML, Systems, AI Infrastructure)  
> **Repository:** [https://github.com/paridhik11/adaptive-agentic-semantic-cache](https://github.com/paridhik11/adaptive-agentic-semantic-cache)  
> **Primary Technologies:** Python, FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`), Scikit-learn, OpenRouter API, Pytest  
> **Authoritative Fact Ledger:** All quantitative metrics, confidence intervals, sample counts, and thresholds strictly cite [`docs/FACTS.md`](docs/FACTS.md).

---

## Resume Section Entry

**Adaptive Agentic Semantic Cache** | *Python, FAISS, Sentence-Transformers, Scikit-learn, OpenRouter API, Pytest*  
*Agentic Semantic Caching & Safety Routing Pipeline for LLM Query Serving*

---

## 1. Verified Resume Bullets

### Bullet 1: System Latency Optimization & Gateway Architecture
- **Accomplished:** Reduced query response latency by 99.20% to 99.84% (124.6x to 622.7x speedup, 10.56–61.91 ms vs 6,577.94–9,291.35 ms) on high-confidence cache hits while resolving 86.39% to 89.71% of traffic locally on-device without remote model calls
- **Measured by:** Multi-run load tests across 70, 338, and 175 query workloads comparing vector lookup against remote judge adjudication
- **Doing:** Designing a three-tier routing gateway that routes high-confidence matches to an in-memory FAISS vector index and borderline queries to an agentic LLM judge

#### One-Line Version (Tight Resumes)
> Architected three-tier semantic caching pipeline reducing query latency by 99.20% to 99.84% (124.6x to 622.7x speedup, 10.56–61.91 ms) on cache hits while resolving 86.39% to 89.71% of traffic locally without remote model calls.

#### Two-Line Version (Expanded Resumes)
> Architected three-tier semantic caching pipeline reducing query latency by 99.20% to 99.84% (124.6x to 622.7x speedup, 10.56–61.91 ms vs 6,577.94–9,291.35 ms) on cache hits by routing high-confidence queries to an in-memory FAISS vector index (`all-MiniLM-L6-v2`).  
> Resolved 86.39% to 89.71% of queries locally on-device without remote judge calls, eliminating external API latency for non-ambiguous requests.

#### FACTS.md Row Citations
- Latency reduction range: [`LAT-05`](docs/FACTS.md) (99.20% to 99.84%)
- Speedup factor range: [`LAT-04`](docs/FACTS.md) (124.6x to 622.7x)
- `AUTO_REUSE` latency range: [`LAT-01`](docs/FACTS.md), [`LAT-02`](docs/FACTS.md), [`LAT-03`](docs/FACTS.md) (10.56–61.91 ms)
- Remote judge latency range: [`LAT-01`](docs/FACTS.md), [`LAT-03`](docs/FACTS.md), [`P5A-07`](docs/FACTS.md), [`P6-10`](docs/FACTS.md) (6,577.94–9,291.35 ms)
- Local resolution rate range: [`LRR-01`](docs/FACTS.md), [`LRR-02`](docs/FACTS.md), [`LRR-03`](docs/FACTS.md), [`LRR-04`](docs/FACTS.md) (86.39% to 89.71%)
- Routing thresholds: [`C-06`](docs/FACTS.md) (similarity floor 0.92), [`C-07`](docs/FACTS.md) (confidence floor 0.90)

#### "Defend It" Interview Notes
- **Hardest Question:** *"In your scaled (N=338) and blind (N=175) evaluations, you only observed exactly n=1 hit on the AUTO_REUSE path in each run. Isn't claiming a 99.20% to 99.84% speedup based on an n=1 sample misleading?"*
- **Honest, Rigorous Defense:** "Acknowledge the sample size immediately: in the N=338 and N=175 benchmarks, exactly n=1 query crossed the strict dual-gate threshold (similarity $\ge 0.92$, stability confidence $\ge 0.90$; row `LAT-06`). The most statistically reliable latency sample came from the N=70 pilot workload, which produced n=20 `AUTO_REUSE` hits with a mean latency of 19.18 ms (row `LAT-01`). Furthermore, clarify that the 99.20% to 99.84% reduction compares local vector lookup against remote judge adjudication (6,577.94–9,291.35 ms), not full generation latency of the primary downstream LLM. In production, local vector search will always be sub-50 ms, but realized cache efficiency depends on real-world prompt repetition."

---

### Bullet 2: Sealed Blind Safety Evaluation Methodology
- **Accomplished:** Verified 0 false positives across 8 admitted cache hits (empirical hazard rate 0.00%, exact 95% Clopper-Pearson CI [0.00%, 36.94%])
- **Measured by:** Sealed blind benchmark of 175 entries across 7 domains evaluated with zero parameter tuning
- **Doing:** Implementing an immutable cryptographic evaluation protocol (sealed commit `fc35744`) prohibiting prompt or threshold adjustments to prevent data leakage

#### One-Line Version (Tight Resumes)
> Achieved 0 false positives across 8 admitted cache hits (0.00% hazard rate, exact 95% Clopper-Pearson CI [0.00%, 36.94%]) on a 175-query benchmark across 7 domains sealed prior to execution.

#### Two-Line Version (Expanded Resumes)
> Achieved 0 false positives across 8 admitted cache hits (0.00% hazard rate, exact 95% Clopper-Pearson CI [0.00%, 36.94%]) on a 175-query benchmark across 7 domains.  
> Enforced strict experimental integrity by cryptographically sealing evaluation datasets prior to test execution, eliminating data leakage and post-hoc tuning.

#### FACTS.md Row Citations
- Blind evaluation sample size: [`D-09`](docs/FACTS.md), [`P6-01`](docs/FACTS.md) (175 entries across 7 domains)
- Cryptographic seal: [`P6-02`](docs/FACTS.md) (commit `fc35744`)
- Hit count and false positives: [`P6-03`](docs/FACTS.md) (8 hits: 1 auto, 7 judge; 0 false positives)
- Empirical hazard rate: [`P6-04`](docs/FACTS.md) (0.00%)
- Exact 95% Clopper-Pearson CI: [`P6-05`](docs/FACTS.md) ([0.00%, 36.94%])

#### "Defend It" Interview Notes
- **Hardest Question:** *"You report 0 false positives, but your 95% Clopper-Pearson upper bound is 36.94%. How can you claim your cache is safe when the error rate could theoretically be nearly 37%?"*
- **Honest, Rigorous Defense:** "Concede that with n=8 admitted hits, the sample is statistically underpowered. To mathematically certify that an error rate does not exceed a 10% safety ceiling at 95% confidence when observing zero errors ($k=0$), an evaluation must observe at least $n \ge 36$ hits (rows `P4-04`, `POOL-06`). Even pooling the corrected scaled run ($n=14$) and blind run ($n=8$) yields $n=22$ hits with an upper bound of 15.44% (row `POOL-03`), which still exceeds 10%. We do not claim an enterprise-certified safety guarantee; we report the exact empirical observation on sealed data alongside its statistical upper bound and state the exact sample size needed for certification."

---

### Bullet 3: Ambiguous-Band Adjudication & Hazard Reduction
- **Accomplished:** Reduced cache hazard rate from 20.83% (5 FP / 24 hits at best fixed cosine threshold 0.85) to 8.33% (1 FP / 12 hits, 95% CI [0.21%, 38.48%]), achieving a 12.50 percentage point reduction (60.00% relative reduction)
- **Measured by:** Pairwise calibration benchmark across 120 curated query pairs comparing static cosine thresholds against dynamic LLM adjudication
- **Doing:** Integrating an agentic LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`) to adjudicate borderline similarity with a strict fail-closed fallback invariant

#### One-Line Version (Tight Resumes)
> Reduced semantic cache hazard from 20.83% (fixed cosine 0.85) to 8.33% (1 FP / 12 hits, 95% CI [0.21%, 38.48%]) on a 120-pair baseline via an agentic LLM judge with fail-closed safety.

#### Two-Line Version (Expanded Resumes)
> Reduced semantic cache hazard from 20.83% (5 FP / 24 hits at best fixed cosine 0.85) to 8.33% (1 FP / 12 hits, 95% CI [0.21%, 38.48%]) on a 120-pair calibration benchmark.  
> Integrated an agentic LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`) for ambiguous similarity, enforcing a fail-closed architecture returning cache bypass on timeouts or errors.

#### FACTS.md Row Citations
- Fixed threshold baseline: [`P2-01`](docs/FACTS.md) (0.85 threshold, 5 FP / 24 hits, 20.83% hazard rate)
- Fixed threshold 95% CI: [`HR-01`](docs/FACTS.md) ([7.13%, 42.15%])
- Judge hazard rate & CI: [`P3-06`](docs/FACTS.md) (1 FP / 12 hits, 8.33% hazard rate, 95% CI [0.21%, 38.48%])
- Absolute & relative reduction: [`HR-02`](docs/FACTS.md) (12.50 percentage points), [`HR-03`](docs/FACTS.md) (60.00% relative reduction)
- Non-significance & CI overlap: [`HR-04`](docs/FACTS.md) ([7.13%, 42.15%] vs [0.21%, 38.48%])
- Fail-closed invariant: [`C-16`](docs/FACTS.md) (`decision=BYPASS`, `confidence=0.0`)

#### "Defend It" Interview Notes
- **Hardest Question:** *"The confidence interval for the judge [0.21%, 38.48%] overlaps with the fixed threshold confidence interval [7.13%, 42.15%]. Isn't this hazard reduction statistically unproven?"*
- **Honest, Rigorous Defense:** "Yes, exactly. Because the 95% Clopper-Pearson confidence intervals overlap, the 12.50 percentage point reduction on this 120-pair development set is a same-set point estimate and is not statistically established at 95% confidence (row `HR-04`). It served as an engineering decision signal to reject non-LLM decision baselines (which suffered a 52.17% error rate; row `P3-01`) and validate the fail-closed judge architecture, but must not be cited as a certified blind safety bound."

---

### Bullet 4: Volatility Interception & Stability Classification
- **Accomplished:** Intercepted temporal query volatility in 2.29 ms mean latency with 91.67% accuracy (55/60) and 0 dangerous false positives across all 24 dynamic queries on a held-out challenge benchmark
- **Measured by:** Held-out challenge ($N=60$) and final test ($N=60$) benchmarks evaluating unseen prompt formulations
- **Doing:** Developing a two-stage hybrid classifier combining deterministic regex rules for temporal markers with TF-IDF logistic regression operating under an 0.80 confidence safety floor

#### One-Line Version (Tight Resumes)
> Engineered a two-stage stability filter achieving 91.67% accuracy and 0 dangerous errors across 24 held-out dynamic queries in 2.29 ms mean latency, blocking volatile prompts before vector lookup.

#### Two-Line Version (Expanded Resumes)
> Engineered a two-stage stability filter combining deterministic regex rules and TF-IDF logistic regression, achieving 91.67% accuracy (55/60) and 0 dangerous errors on 24 held-out dynamic queries in 2.29 ms mean latency.  
> Enforced an 0.80 confidence override floor to intercept volatile and temporal prompts prior to FAISS vector search, eliminating stale response reuse.

#### FACTS.md Row Citations
- Classifier accuracy: [`P1-01`](docs/FACTS.md), [`P1-03`](docs/FACTS.md) (91.67%, 55/60 on final test and challenge sets)
- Dangerous errors on dynamic queries: [`P1-02`](docs/FACTS.md) (0 on 23 final test queries), [`P1-04`](docs/FACTS.md) (0 on 24 challenge queries)
- False rejection rate on stable queries: [`P1-05`](docs/FACTS.md) (13.51%, 5/37), [`P1-06`](docs/FACTS.md) (13.89%, 5/36)
- Classifier latency: [`P1-07`](docs/FACTS.md), [`P1-08`](docs/FACTS.md) (2.29 ms mean latency)
- Confidence override floor: [`C-05`](docs/FACTS.md) (0.80)

#### "Defend It" Interview Notes
- **Hardest Question:** *"Your classifier exhibits a 13.51% to 13.89% false rejection rate on stable queries. Isn't that an unacceptable loss of cache efficiency?"*
- **Honest, Rigorous Defense:** "The classifier is deliberately engineered with an asymmetric loss function: false positives (admitting a dynamic query to cache) cause catastrophic semantic hazards by serving stale or inaccurate data to end users. False rejections (routing a stable query to dynamic bypass) only trigger a fresh downstream LLM generation. By establishing an 0.80 confidence override floor, we guaranteed zero dangerous false positives across both held-out evaluation sets while accepting a modest 13.51% false rejection rate as an acceptable safety trade-off."

---

### Bullet 5: Benchmark Auditing & Ground-Truth Collision Correction
- **Accomplished:** Identified and resolved 10 false-positive benchmark labeling collisions in a 338-query scaled evaluation, correcting the apparent cache hazard rate from 71.43% (10 FP / 14 hits, CI [41.90%, 91.61%]) to a verified 0.00% (0 FP / 14 hits, CI [0.00%, 23.16%])
- **Measured by:** Pre-correction vs post-correction telemetry analysis on 338 queries (210 cold seeds + 128 authored queries)
- **Doing:** Authoring an automated pairwise similarity audit script to detect near-duplicate ground-truth collisions where synonymous queries were incorrectly labeled negative in synthetic benchmarks

#### One-Line Version (Tight Resumes)
> Uncovered and resolved 10 benchmark labeling collisions in a 338-query evaluation, correcting apparent cache hazard from 71.43% (10 FP / 14 hits) to a verified 0.00% (0 FP / 14 hits, 95% CI [0.00%, 23.16%]).

#### Two-Line Version (Expanded Resumes)
> Uncovered and resolved 10 benchmark labeling collisions in a 338-query scaled evaluation via automated pairwise similarity auditing, correcting apparent cache hazard from 71.43% (10 FP / 14 hits) to a verified 0.00% (0 FP / 14 hits, 95% CI [0.00%, 23.16%]).  
> Transparently documented root causes across StackExchange cold seeds and authored prompts, disclosing 6 residual collisions to preserve benchmark audit integrity.

#### FACTS.md Row Citations
- Pre-correction error count & hazard rate: [`P5B-01`](docs/FACTS.md) (10 FP / 14 hits, 71.43%), [`P5B-03`](docs/FACTS.md) (95% CI [41.90%, 91.61%])
- Post-correction error count & hazard rate: [`P5B-02`](docs/FACTS.md) (0 FP / 14 hits, 0.00%), [`P5B-04`](docs/FACTS.md) (95% CI [0.00%, 23.16%])
- Collision count identified: [`P5B-05`](docs/FACTS.md) (10 collisions)
- Uncorrected residual collisions disclosed: [`P5B-06`](docs/FACTS.md), [`KL-01`](docs/FACTS.md) (6 collisions)
- Dataset composition: [`D-08`](docs/FACTS.md) (338 queries: 210 seeds + 128 authored)

#### "Defend It" Interview Notes
- **Hardest Question:** *"When your scaled test failed with a 71.43% hazard rate, you modified the benchmark ground-truth labels. How can an interviewer trust this wasn't post-hoc label manipulation?"*
- **Honest, Rigorous Defense:** "We preserved full auditability: rather than overwriting telemetry files, we preserved both the uncorrected run (71.43% hazard, CI [41.90%, 91.61%]; row `P5B-01`) and the corrected run (0.00% hazard, CI [0.00%, 23.16%]; row `P5B-02`) in the public repository and ledgered both in `docs/FACTS.md`. We authored an automated semantic audit script proving that all 10 collisions were exact synonymous questions where the cache returned the correct answer despite synthetic negative tags. Crucially, we disclosed that 6 residual collisions remain in `data/raw/new_dataset_v3.json` (row `P5B-06`), and subsequently proved 0 false positives on an independent, pre-sealed blind dataset (Phase 6)."

---

### Bullet 6: Empirical Reproducibility, Fact Ledgering & CI Verification
- **Accomplished:** Enforced 100% empirical claim traceability and zero unledgered numeric claims across all documentation, maintaining 0 secret leaks across repository commit history
- **Measured by:** Automated token-level CI verifier (`check_numeric_tokens.py`) checking extracted numeric tokens against an authoritative fact ledger, paired with automated Gitleaks scans
- **Doing:** Designing an authoritative single-source-of-truth fact ledger (`docs/FACTS.md`) and pre-commit verification pipelines blocking unverified quantitative claims

#### One-Line Version (Tight Resumes)
> Built an authoritative fact ledger and token-level CI verifier enforcing zero unledgered numerical claims across documentation, backed by 100% clean Gitleaks secret verification.

#### Two-Line Version (Expanded Resumes)
> Built an authoritative fact ledger (`docs/FACTS.md`) and automated token-level CI verifier enforcing zero unledgered numerical claims across all project documentation.  
> Integrated automated secret detection achieving 100% clean Gitleaks scans across repository history, establishing end-to-end scientific auditability and supply chain security.

#### FACTS.md Row Citations
- Authoritative ledger rows: [`C-01`..`C-16`](docs/FACTS.md), [`P1-01`..`P6-11`](docs/FACTS.md), [`LAT-01`..`LAT-06`](docs/FACTS.md), [`LRR-01`..`LRR-04`](docs/FACTS.md), [`POOL-01`..`POOL-07`](docs/FACTS.md), [`HR-01`..`HR-04`](docs/FACTS.md)
- Secret scanning audit: [`SHA-01`..`SHA-04`](docs/FACTS.md) (100% clean Gitleaks verification)
- Runtime verification: [`ENV-01`](docs/FACTS.md) (Python 3.14.5 fresh-clone verified)

#### "Defend It" Interview Notes
- **Hardest Question:** *"Why spend time building custom CI verification scripts and fact ledgers instead of training larger ML models?"*
- **Honest, Rigorous Defense:** "In real-world ML systems engineering, unverified claims and metrics drift are common failure modes. By establishing an immutable fact ledger (`docs/FACTS.md`) and an automated token-level CI check that extracts every number in documentation and ensures it exists in the ledger with exact citations, we prevented phantom metrics from contaminating the codebase. This rigor caught multiple discrepancies early—including labeling collisions and statistical sample size limitations—ensuring that every stated metric withstands technical due diligence."

---

## 2. "Do Not Claim" Checklist

When discussing this project in technical interviews, the following claims are **NOT** supported by empirical data and must **NEVER** be made:

1. **Do NOT claim production-ready enterprise deployment:**
   - The FAISS vector store runs strictly in-memory without disk persistence across process restarts ([`KL-07`](docs/FACTS.md)).
   - Evaluated query workloads were structured synthetic streams and StackExchange seeds, not live enterprise production traffic ([`KL-08`](docs/FACTS.md)).
   - The OpenRouter judge relies on a free-tier endpoint with a 50 requests-per-day quota limit ([`KL-03`](docs/FACTS.md)).

2. **Do NOT claim dollar cost savings:**
   - No dollar pricing calculations were conducted or ledgered in this project.
   - Never claim "saved $X" or "cut API bills by Y%". The project strictly measured query routing counts and latency timings.

3. **Do NOT claim 0% incorrect reuse as a certified enterprise guarantee:**
   - While Phase 6 observed 0 false positives ($k=0$) on 8 hits, the exact 95% Clopper-Pearson upper bound is 36.94% ([`P6-05`](docs/FACTS.md)).
   - Even pooling Phase 5 corrected ($n=14$) and Phase 6 ($n=8$) yields $n=22$ with an upper bound of 15.44% ([`POOL-03`](docs/FACTS.md)), which remains above the 10% safety ceiling ([`C-13`](docs/FACTS.md)).
   - Mathematically certifying that the hazard rate does not exceed 10% under a 95% confidence interval requires observing at least **$n \ge 36$ hits** with zero errors ([`P4-04`](docs/FACTS.md)).

4. **Do NOT claim the adaptive threshold mechanism tuned thresholds in production:**
   - All 7 domain categories remained on the global fallback threshold of 0.85 because none passed the dual finite-sample safety gates ($N \ge 20$ domain samples, $N_{\text{minority}} \ge 10$) in Phase 4 calibration ([`P4-01`](docs/FACTS.md), [`KL-11`](docs/FACTS.md)).

5. **Do NOT claim a 99.20% to 99.84% speedup on full downstream LLM generation:**
   - The 99.20% to 99.84% latency reduction (124.6x to 622.7x speedup; [`LAT-04`](docs/FACTS.md), [`LAT-05`](docs/FACTS.md)) compares local vector cache retrieval (`AUTO_REUSE`, 10.56–61.91 ms) against remote LLM judge calls (6,577.94–9,291.35 ms) on ambiguous queries.
   - Downstream text generation time from a primary model was not measured.

6. **Do NOT claim the 8.04% pooled sensitivity figure as a headline blind metric:**
   - The 8.04% upper bound ([`POOL-07`](docs/FACTS.md)) pools 44 hits across pilot, scaled, and blind runs ($k=0, n=44$). It includes design-informed pilot data and is explicitly a non-blind sensitivity check. It must not be cited as a blind safety metric.

7. **Do NOT present same-set calibration numbers as blind safety bounds:**
   - The Phase 3 judge hazard rate of 8.33% (1 FP / 12 hits, 95% CI [0.21%, 38.48%]; [`P3-06`](docs/FACTS.md)) is a same-set point estimate measured on the 120-pair development benchmark ([`KL-06`](docs/FACTS.md)).
   - Its confidence interval overlaps with the fixed threshold baseline ([7.13%, 42.15%]; [`HR-04`](docs/FACTS.md)). It guided architecture selection, not a proven blind safety bound.
