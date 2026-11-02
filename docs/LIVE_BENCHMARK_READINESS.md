# Live Benchmark Readiness & Safety Envelope Report

**Version:** 1.0  
**Date:** 2026-11-02  
**Harness Script:** `scripts/evaluate_live_paired.py`  
**Execution Status:** **BLOCKED (Safe by Default)**  

---

## 1. Safety Envelope & Invariants

To comply with standing project rules and scientific evaluation protocols, all live inference operations are bound by hard safety invariants:

1. **Explicit Multi-Gate Authorization:**
   Live provider network dispatches are strictly disabled by default. Execution requires the explicit command-line flag `--execute-live`. Without this flag, the runner operates strictly in preflight verification mode.
2. **Hard Zero Spend Cap:**
   All live operations are restricted to free-tier model routes. The hard spend cap defaults to `$0.00`. Any non-zero spend cap or paid model route is immediately rejected during outbound request sanitization.
3. **Allowlisted Free-Only Models:**
   Outbound requests are strictly allowlisted to verified free-tier endpoints on OpenRouter:
   - `nvidia/nemotron-3-super-120b-a12b:free` (Default)
   - `meta-llama/llama-3.2-3b-instruct:free`
   Paid fallbacks, caller-controlled routing parameters, and unverified third-party endpoints are prohibited.
4. **Zero-Output Ceiling Safeguard:**
   Preflight checks and request sanitizers validate that `max_output_tokens > 0`. Requests with non-positive ceilings are rejected before network transmission.
5. **Durable Pilot Journal Quota:**
   Calls are governed by an immutable SQLite journal (`PilotJournal`). Quota reservations prevent duplicate executions, ensure recovery across crashes, and prohibit hidden retries or quota resets.
6. **Immutable Replay Preservation:**
   When live execution is authorized, actual returned prompt/completion tokens, request latency, agent workflow latency, and complete response payloads are written to durable receipts (`artifacts/live_evaluation/`) to power offline reproducible replays.

---

## 2. Frozen 3-Arm Workload Specification

The live evaluation harness exercises the complete `DocumentAgent` (supervisor planning, tool retrieval, draft generation, reviewer verification, and operator approval) across a frozen 12-task workload:

| Task ID | Domain Group | Query | Category | Expected Behavior |
| :---: | :--- | :--- | :---: | :--- |
| `live-1` | `list_operations` | How do I convert a list to a tuple in Python? | `seed` | Cold execution (3 provider calls, admitted upon approval) |
| `live-2` | `list_operations` | How do I convert a list to a tuple in Python? | `exact_repeat` | Exact hit (0 provider calls in exact & semantic arms) |
| `live-3` | `list_operations` | How can I turn a Python list into a tuple? | `natural_paraphrase` | Semantic hit (0 provider calls in semantic arm) |
| `live-4` | `list_operations` | How do I convert a tuple to a list in Python? | `adversarial` | Fail closed (3 provider calls, blocked by direction guard) |
| `live-5` | `dict_operations` | How do I merge two dictionaries in Python 3.9+? | `seed` | Cold execution (3 provider calls, admitted upon approval) |
| `live-6` | `dict_operations` | How do I merge two dictionaries in Python 3.9+? | `exact_repeat` | Exact hit (0 provider calls in exact & semantic arms) |
| `live-7` | `dict_operations` | What is the operator syntax to combine two dicts in Python 3.9? | `natural_paraphrase` | Semantic hit (0 provider calls in semantic arm) |
| `live-8` | `dict_operations` | How do I merge two lists into a dictionary in Python? | `adversarial` | Fail closed (3 provider calls, blocked by entity guard) |
| `live-9` | `file_io` | How do I read a file line by line in Python? | `seed` | Cold execution (3 provider calls, admitted upon approval) |
| `live-10` | `file_io` | How do I read a file line by line in Python? | `exact_repeat` | Exact hit (0 provider calls in exact & semantic arms) |
| `live-11` | `file_io` | What is the standard way to iterate through lines in a file in Python? | `natural_paraphrase` | Semantic hit (0 provider calls in semantic arm) |
| `live-12` | `file_io` | How do I write a list of strings line by line to a file in Python? | `adversarial` | Fail closed (3 provider calls, blocked by action guard) |

---

## 3. Preflight Readiness Verification Results

Execution of `python scripts/evaluate_live_paired.py --check-only` confirmed that all safety gates are actively enforcing fail-closed protection:

```text
================================================================================
          LIVE PAIRED BENCHMARK EVALUATION HARNESS — STATUS REPORT              
================================================================================

Status           : BLOCKED
Model            : nvidia/nemotron-3-super-120b-a12b:free
Live Authorized  : False

Blockers / Safety Gates:
  - [BLOCKED] Live execution flag '--execute-live' not passed. Operating in preflight verification mode.

Checks:
  - model_allowed            : True
  - spend_cap_zero           : True
  - output_ceiling_valid     : True
  - live_authorized          : False
  - api_key_configured       : False
  - journal_available        : False
  - journal_quota_remaining  : 0
```

### Readiness Assessment
- [x] Preflight checks block unauthorized live execution.
- [x] Model routing restricted to free-only model.
- [x] Zero spend cap enforced ($0.00).
- [x] Output token ceiling validated.
- [x] Live API network calls remain **STRICTLY BLOCKED** pending explicit user instruction.
