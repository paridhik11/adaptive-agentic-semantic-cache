# BLOCKED_A.md: Execution Halted at GATE A

> **Status:** HALTED / BLOCKED at GATE A  
> **Timestamp:** 2026-10-12  
> **Repository:** `adaptive-agentic-semantic-cache`  
> **Reference Protocol:** `docs/ROADMAP_ABC.md` (Gate A)  
> **Pre-registration:** `docs/phaseA_preregistration.md` (commit `d1f9d97`)  
> **Sealed Telemetry:** `data/phaseA_sealed_evaluation_telemetry.json`  
> **Sealed Summary:** `data/phaseA_sealed_evaluation_summary.json`  

---

## 1. Reason for Halted Execution

In strict compliance with the **GLOBAL RULES** and **GATE A** specification of `docs/ROADMAP_ABC.md`, autonomous execution of the roadmap is **HALTED**. Phase B will **NOT** be started.

The single pre-registered sealed test stream evaluation on 2200 queries failed two required criteria of Gate A:
1. **Statistical Volume:** Measured **75 non-exact hits** (Required: $\ge 100$ non-exact hits).
2. **Safety Ceiling:** Measured **28.00% non-exact IRR point estimate** [18.24%, 39.56%] (Required: $\le 10.0\%$ point estimate).

Per protocol instruction:
> *"A bad result is still a valid, reportable result; do not tune on the sealed set to fix it. If any item fails: write `docs/BLOCKED_A.md` with the raw findings and STOP (do not start Phase B)."*

---

## 2. Gate A Verification Matrix

| Gate A Item | Protocol Requirement | Actual Measured Result (`sealed`) | Gate Pass? |
| :--- | :--- | :--- | :---: |
| 1. Pre-registration | Committed to git before run | Pre-registered in commit `d1f9d97` | **PASS** |
| 2. Cryptographic hash | Hash matches frozen configuration | Verified identical: `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991` | **PASS** |
| 3. Execution limit | Executed exactly once | Executed single run (66.64 s runtime) | **PASS** |
| 4. Unit tests | All repository unit tests pass | 281 / 281 passing | **PASS** |
| 5. Security & Tokens | Strict token check passes; clean gitleaks | Verified clean (0 leaks, 0 unledgered tokens) | **PASS** |
| 6. **Hit Sample Size** | **$\ge 100$ non-exact hits** | **75 non-exact hits** | **FAIL** |
| 7. **Safety Ceiling** | **$\text{IRR}_{\text{non-exact}} \le 10.0\%$ point estimate** | **28.00% point estimate [18.24%, 39.56%]** | **FAIL** |

---

## 3. Raw Evaluation Findings

The sealed stream ($N=2200$) produced 86 total cache hits (60 TP, 26 FP), of which 11 were exact canonical normalizer matches (11 TP, 0 FP).

### Primary Endpoint: Non-Exact Matches
- **Total Non-Exact Hits ($n$):** 75
- **Non-Exact True Positives:** 54
- **Non-Exact False Positives ($k$):** 21
- **Non-Exact Point Estimate IRR:** 28.00%
- **Exact 95% Clopper-Pearson CI:** [18.24%, 39.56%]

### Error Source Attribution
- **Adversarial Word Scrambles (`paws_test`):** Accounted for **16 of the 21 false positives** (Non-exact IRR on PAWS test was 47.06% [29.84%, 64.89%]). When scrambled sentences exceeded the 0.50 vector similarity floor, the fine-tuned CrossEncoder assigned scores $> 0.85$, demonstrating an out-of-distribution generalization gap on unseen grammatical scrambles.
- **Held-Out Paraphrases (`qqp_heldout`):** Accounted for **5 of the 21 false positives** out of 40 non-exact hits (Non-exact IRR was 12.50% [4.19%, 26.80%]).
- **Technical Q&A (`cqadupstack_heldout`):** 1 non-exact hit, 1 TP, 0 FP.
- **Natural Chat (`wildchat_traffic`):** 0 hits out of 200 unseeded queries.

---

## 4. Why the Pipeline Failed & Options Going Forward

### Technical Diagnosis
The failure confirms a major empirical limitation of small cross-encoder models:
1. **Adversarial syntactic sensitivity:** While fine-tuning on 2000 pairs eliminated PAWS errors on the validation distribution, it failed to generalize to the full test set of adversarial scrambles without lexical syntax parsing.
2. **Fast-path routing:** Bi-encoder embeddings cannot distinguish word order permutations (e.g. *"Flights from New York to Florida"* vs *"Flights from Florida to New York"*), forcing reliance on the downstream CrossEncoder which still exhibited residual false positives.

### Next Options (Pending Human Approval)
In accordance with instructions, no changes or re-evaluations are performed on the sealed test set. The possible paths are:
1. **Report Honest Finding:** Treat the failure of Gate A as the primary negative result: demonstrating that local cross-encoders without lexical DAG parsing or remote LLM judge verification cannot reliably guarantee $\text{IRR} \le 10\%$ against adversarial bag-of-words scrambles.
2. **Re-architecture with Remote Judge (Phase C style):** Incorporate the remote LLM judge back into the ambiguous tier for adversarial syntactic cases.
3. **Threshold Calibration:** Tighten the CrossEncoder decision threshold or add a strict bidirectional word-order alignment check. Note: Any new parameters would require a new pre-registration and fresh evaluation stream.

Phase B gateway execution remains blocked until human direction.
