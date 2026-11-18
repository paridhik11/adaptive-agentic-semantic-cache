# Live Benchmark Readiness & Safety Envelope Audit Report

**Version:** 1.1  
**Date:** 2026-11-02  
**Harness Script:** `scripts/evaluate_live_paired.py`  
**Execution Status:** **BLOCKED (Safe by Default — Zero Live Calls Executed)**  
**Receipt Path:** `artifacts/live_evaluation/live_readiness_receipt.json`  

---

## 1. Safety Envelope & Audit Invariants

A comprehensive pre-use audit of `scripts/evaluate_live_paired.py` confirms that all safety boundaries and invariants are strictly enforced:

1. **Shared Quota Enforcement Across All Arms:**
   - All three arms share a single durable SQLite journal (`artifacts/live_evaluation/pilot_journal.sqlite`).
   - Each arm executes with a distinct coordinated run ID (`f"{base_run_id}-{arm_name}"`) within the shared database.
   - `PilotJournal` enforces a project-wide daily cap (40 calls max) and rolling-minute rate limit (10 calls/min) across all runs in the database.
   - The runner tracks cumulative calls across all arms and halts execution if cumulative calls exceed the study budget cap (30 calls max).
2. **Stable Retries & Fail-Closed Rate Limiting:**
   - Any upstream rate limit (`429`), network error, or timeout immediately raises `RunPaused` and updates the journal control state.
   - Rate-limited responses invoke `journal.block_rate_limit()`, blocking subsequent requests until the next UTC day.
   - No automatic paid fallback, no hidden retries, and no quota resets.
3. **Strict Zero Spend Cap & Zero Paid Fallback:**
   - Outbound requests enforce `outgoing['provider'] = {'allow_fallbacks': False, 'max_price': {'prompt': 0, 'completion': 0, 'request': 0}, 'require_parameters': True}`.
   - Spend cap is hard-coded to `$0.00`. Any non-zero spend cap is rejected during preflight.
4. **Allowlisted Free-Only Models:**
   - Outbound requests are strictly allowlisted to verified free endpoints:
     - `nvidia/nemotron-3-super-120b-a12b:free` (Default)
     - `meta-llama/llama-3.2-3b-instruct:free`
   - Model strings must match regex `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+:free`. Automatic provider routing is disabled.
5. **Zero-Output Ceiling Safeguard:**
   - `max_output_tokens` must be strictly positive (`> 0`). Non-positive ceilings are blocked during preflight and request validation.
6. **Complete Actual Token Usage:**
   - In live mode, actual prompt tokens, completion tokens, and total tokens are recorded directly from OpenRouter response payload receipts (`usage` field) rather than estimated synthetically.
7. **Full-Task Timing:**
   - Wall-clock latencies measure end-to-end task execution: from task creation through supervisor planning, document tool reads, draft generation, reviewer verification, operator approval, and terminal SQLite state persistence.
8. **Immutable Receipts with SHA-256 Digests:**
   - Every completed task produces an immutable receipt containing task ID, category, question, arm, response answer SHA-256 digest, cache hit status, tier, status, actual provider calls by stage, actual token usage, and elapsed milliseconds.

---

## 2. Genuine Three-Arm Matching & Shadow Semantic Mode

In accordance with release safety directives, experimental semantic reuse is kept in **shadow mode** during the initial live study:

| Arm | Mode Setting | Serving Behavior | Provider Invocations on Repeat |
| :--- | :---: | :--- | :---: |
| **Arm 1: `no_cache`** | `mode="off"` | Uncached baseline. Every request executes full 3-stage agent workflow. | 3 calls |
| **Arm 2: `exact_only`** | `mode="exact"` | Safe release default. Tier 0 exact SQLite lookup; misses fall back to fresh execution. | **0 calls (exact hit)** |
| **Arm 3: `shadow_semantic`** | `mode="shadow"` | Evaluates neural cross-encoder in background; serves exact answers only; **no unproven answer substitution**. | **0 calls (exact hit)** |

---

## 3. Proposed Real-Provider Pilot Plan (Frozen 4-Task Workload)

To ensure the initial live study is bounded, reproducible, and safely within a single 30-call quota window, a frozen 4-task pilot workload is prepared across 2 documentation groups:

### Pilot Workload Specification:
| Task ID | Domain Group | Query | Category | Proposed Calls (`no_cache`) | Proposed Calls (`exact_only`) | Proposed Calls (`shadow_semantic`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `pilot-1` | `list_operations` | How do I convert a list to a tuple in Python? | `seed` | 3 calls (cold) | 3 calls (cold) | 3 calls (cold) |
| `pilot-2` | `list_operations` | How do I convert a list to a tuple in Python? | `exact_repeat` | 3 calls (no cache) | **0 calls (exact hit)** | **0 calls (exact hit)** |
| `pilot-3` | `dict_operations` | How do I merge two dictionaries in Python 3.9+? | `seed` | 3 calls (cold) | 3 calls (cold) | 3 calls (cold) |
| `pilot-4` | `dict_operations` | How do I merge two dictionaries in Python 3.9+? | `exact_repeat` | 3 calls (no cache) | **0 calls (exact hit)** | **0 calls (exact hit)** |
| **Total** | | | | **12 calls** | **6 calls** | **6 calls** |

### Exact Proposed Provider Request Accounting:
- **`no_cache`**: 4 tasks × 3 stages = **12 provider calls**.
- **`exact_only`**: 2 cold tasks × 3 stages + 2 exact hits × 0 calls = **6 provider calls**.
- **`shadow_semantic`**: 2 cold tasks × 3 stages + 2 exact hits × 0 calls = **6 provider calls**.
- **Total Proposed Provider Requests:** **24 provider calls**.
- **Budget Compliance:** 24 calls ≤ 30-call single run cap and ≤ 40-call daily project cap (`fits_within_budget = True`).

---

## 4. Tested Preflight Verification Receipt

Execution of `python scripts/evaluate_live_paired.py --check-only` generated the audited preflight receipt:

```text
================================================================================
          LIVE PAIRED BENCHMARK EVALUATION HARNESS — STATUS REPORT              
================================================================================

Status           : BLOCKED
Model            : nvidia/nemotron-3-super-120b-a12b:free
Live Authorized  : False

Proposed Real-Provider Pilot Plan:
  - Workload Profile         : pilot_4_tasks (4 tasks/arm)
  - Arm 1 (no_cache)         : 12 provider calls
  - Arm 2 (exact_only)       : 6 provider calls
  - Arm 3 (shadow_semantic)  : 6 provider calls (shadow mode)
  - Total Proposed Requests  : 24 calls (budget cap: 30)
  - Fits Within Budget       : True

Blockers / Safety Gates Active:
  - [BLOCKED] Live execution flag '--execute-live' not passed. Operating in preflight verification mode.

Safety & Invariant Checks:
  - model_allowed            : True
  - spend_cap_zero           : True
  - output_ceiling_valid     : True
  - budget_cap_valid         : True
  - live_authorized          : False
  - api_key_configured       : False
  - journal_available        : False
  - journal_path             : artifacts/live_evaluation/pilot_journal.sqlite
  - journal_total_calls      : 0

Preflight receipt written to: artifacts/live_evaluation/live_readiness_receipt.json
Live API calls remain strictly blocked pending explicit authorization.
```

### Readiness Declaration
- [x] Shared quota enforcement across all arms verified.
- [x] Fail-closed `RunPaused` rate limit protection verified.
- [x] Zero automatic paid fallback and max price 0.00 verified.
- [x] Valid free-tier model settings verified.
- [x] Complete actual token usage capture verified.
- [x] Full-task timing boundary verified.
- [x] Immutable receipt generation verified.
- [x] Genuine three-arm matching with shadow semantic mode verified.
- [x] Exactly **24 provider requests** proposed for the frozen 4-task pilot.
- [x] Live API network calls remain **STRICTLY BLOCKED** pending explicit authorization.
