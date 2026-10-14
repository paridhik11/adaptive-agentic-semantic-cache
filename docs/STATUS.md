# Project Status Ledger: Roadmap ABC

> Authoritative status tracking for Phases A, B, and C as defined in `docs/ROADMAP_ABC.md`.
> Last updated: 2026-10-12

---

## Roadmap Overview

| Roadmap Phase | Description | Status | Gate Status | Audit Pack |
| :--- | :--- | :---: | :---: | :---: |
| **Phase A** | Trustworthy Matching, Domain Calibration, Real Chat Repeat Rates, Sealed Evaluation | **EVALUATED** | **FAILED** ([`docs/BLOCKED_A.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/BLOCKED_A.md)) | [`docs/audit/AUDIT_PACK_A.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/audit/AUDIT_PACK_A.md) |
| **Phase B** | Drop-in FastAPI Cache Gateway, Isolation, Streaming, Load Test | **BLOCKED** | Execution Halted at Gate A | Pending |
| **Phase C** | LangGraph Agent System, Human Feedback Calibration, Evaluation | **BLOCKED** | Execution Halted at Gate A | Pending |

---

## Phase A Execution Summary

- **A0. Preflight and Repairs:**
  - [x] Clean working tree & unit tests green (281 passing).
  - [x] Phase 8 pipeline reproduced from scratch (50 hits, 21.7% [16.6%, 27.6%], IRR 6.0% [1.3%, 16.5%], FP=3).
  - [x] Erratum on safety ceiling claim (6.0% point estimate does not establish 10% ceiling; CI upper is 16.5%).
  - [x] Category breakdown of Phase 8 test stream (FP=3 all from PAWS val routed to AUTO_REUSE).
  - [x] Validation set (39.62% IRR) vs test stream (6.0% IRR) divergence explained with guard ablation data.
  - [x] Retracted unanchored cost estimate; costs reported exclusively in avoided tokens and LLM calls.
  - [x] Re-measured baseline ($n=100$ real calls on local hardware: mean 8997.49 ms, p50 8153.50 ms, p95 12763.96 ms).
- **A1. Better Semantic Matching:**
  - [x] Implemented long prompt guard and generative request guard (100.0% accuracy on generative benchmark).
  - [x] Fine-tuned CrossEncoder on QQP+PAWS hard negatives (Candidate b & c, 2000 pairs, 449.57 s CPU).
  - [x] Selected Candidate (c) with validation IRR of 2.50% [0.06%, 13.16%] (PAWS FPs reduced from 34 to 0).
- **A2. Domain-Level Calibration (CQADupStack):**
  - [x] Evaluated 12 subforums from `mteb/cqadupstack-retrieval` (CC BY-SA 3.0 / 4.0).
  - [x] All 12 subforums passed sample size ($N=50 \ge 20$) and minority class ($25 \ge 10$) gates.
- **A3. Natural Repeat Rates (WildChat):**
  - [x] Analyzed 5000 first turns from `allenai/WildChat` (ODC-By).
  - [x] Measured exact string repeats: 490 / 5000 (9.80%); canonical repeats: 529 / 5000 (10.58%).
  - [x] Measured guarded semantic reuse: 986 / 5000 (19.72%) at 0.85; 777 / 5000 (15.54%) at 0.92.
  - [x] Exported 100 sample decisions to `data/review/pa_human_audit.csv`.
- **A4. Pre-Registration & Sealed Test Stream Evaluation:**
  - [x] Pre-registration committed (`docs/phaseA_preregistration.md`, commit `d1f9d97`).
  - [x] Frozen config SHA-256 and sealed test stream SHA-256 committed prior to run.
  - [x] Executed single run on 2200 queries (runtime: 66.64 s).
  - [x] Measured primary endpoint non-exact IRR: 28.00% [18.24%, 39.56%] on 75 non-exact hits.
- **A5. Reporting & Gate A Decision:**
  - [x] Gate A evaluated: **FAILED** (75 hits < 100 target; 28.00% IRR > 10.0% target).
  - [x] Written `docs/phaseA_walkthrough.md`, `docs/claim_check_A.md`, `docs/BLOCKED_A.md`, and `docs/audit/AUDIT_PACK_A.md`.

---

## Gate Status
- **GATE A:** **FAILED** (documented in `docs/BLOCKED_A.md`).
- **GATE B:** **BLOCKED** (halted per protocol; awaiting human auditor review of Audit Pack A).
- **GATE C:** **BLOCKED**.
