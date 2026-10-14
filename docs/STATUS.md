# Project Status Ledger: Roadmap ABC

> Authoritative status tracking for Phases A, B, and C as defined in `docs/ROADMAP_ABC.md`.
> Last updated: 2026-10-12

---

## Overall Roadmap Completion: 15%

| Roadmap Phase | Description | Status | Gate Status | Audit Pack |
| :--- | :--- | :---: | :---: | :---: |
| **Phase A** | Trustworthy Matching, Domain Calibration, Real Chat Repeat Rates, Sealed Evaluation | **IN PROGRESS** | Evaluating Gate A | `docs/audit/AUDIT_PACK_A.md` (pending) |
| **Phase B** | Drop-in FastAPI Cache Gateway, Isolation, Streaming, Load Test | **PENDING** | Not started | `docs/audit/AUDIT_PACK_B.md` (pending) |
| **Phase C** | LangGraph Agent System, Human Feedback Calibration, Evaluation | **PENDING** | Not started | `docs/audit/AUDIT_PACK_C.md` (pending) |

---

## Current Sub-stage: Phase A (A0 & A1 Execution)

- **A0. Preflight and Repairs:**
  - [x] Clean working tree & tests green (279 passed).
  - [x] Phase 8 pipeline reproduced from scratch (50 hits, 21.7% [16.6%, 27.6%], IRR 6.0% [1.3%, 16.5%], FP=3).
  - [x] Erratum on safety ceiling claim (6.0% point estimate does not establish 10% ceiling; CI upper is 16.5%).
  - [x] Category breakdown of Phase 8 test stream (FP=3 all from PAWS val routed to AUTO_REUSE).
  - [x] Validation set (39.6% IRR) vs test stream (6.0% IRR) divergence explained with guard ablation data.
  - [x] $0.011 cost estimate retracted; costs reported exclusively in avoided tokens and LLM calls.
  - [x] Re-measuring no-cache baseline ($n \ge 100$ real calls on local hardware + auxiliary remote sample).
- **A1. Better Semantic Matching:** In progress.
- **A2. Domain-Level Thresholds:** In progress.
- **A3. Natural Repeat Rates (WildChat):** In progress.
- **A4. Pre-Registration & Sealed Test Stream:** In progress.
- **A5. Reporting & Gate A Check:** In progress.

---

## Gate Status
- **GATE A:** Evaluating (blocked if gates fail).
- **GATE B:** Pending.
- **GATE C:** Pending.
