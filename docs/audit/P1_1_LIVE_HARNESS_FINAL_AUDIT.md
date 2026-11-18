# P1.1 — Final Live Evaluation Harness Repair & Acceptance Audit Report

**Branch:** `fix/p1-1-live-harness-acceptance`  
**Base Branch:** `integration/documentation-rebuild`  
**Base Commit:** `f9e9b4f7b41e60a1f483ed34fe06a618516f177f`  
**Audit Date:** 2026-11-09  
**Status:** **PASS** (Zero Live Calls Executed, Safe by Default)

---

## 1. Audit Hypotheses Investigation Matrix

| # | Suspected Defect | File(s) Investigated | Audit Finding | Resolution Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Quota accounting double-count:** Check whether `already_spent + cumulative_calls` double-counts journaled calls and incorrectly blocks later benchmark arms. | `scripts/evaluate_live_paired.py` (`run_paired_study`) | **CONFIRMED DEFECT.** When Arm 1 ran, calls were written to `pilot_calls`. Before Arm 2, `already_spent` (count from DB) was added to `cumulative_calls` (study tally). Before Arm 3, `18 + 18 = 36 >= 30`, which falsely blocked Arm 3 (`shadow_semantic`) with `RunPaused` even though total study usage was only 18/30 calls. | **REPAIRED.** Reconciled budget guard to track `initial_journal_calls` and `total_effective_spent = max(current_journal_calls, initial_journal_calls + cumulative_calls)` without double-counting study's own completed calls. |
| **2** | **Completion status false success:** Ensure FAILED, REJECTED, paused, and nonterminal tasks cannot cause a false successful-study declaration. | `scripts/evaluate_live_paired.py` (`run_paired_study`) | **CONFIRMED DEFECT.** `TERMINAL` was set to `{"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}`. If any task failed or was rejected, it was considered terminal, causing `final_status` to blindly report `"COMPLETED"`. | **REPAIRED.** Explicitly partitioned task outcomes into `failed_tasks`, `rejected_tasks`, and `non_terminal_tasks`. Only when 100% of tasks across all arms are `COMPLETED` can study status be `COMPLETED`; otherwise status surfaces as `FAILED`, `REJECTED`, or `INCOMPLETE`. |
| **3** | **Provider token accounting:** Handle partially missing usage receipts without reporting incomplete totals as valid measured usage. | `scripts/evaluate_live_paired.py` (`_extract_usage_from_trace`, `_run_arm`) | **CONFIRMED DEFECT.** If some provider calls had usage receipts and others lacked usage receipts, the function summed available receipts and reported `source: "trace_entries"` as if it were complete measured usage. Arm totals also reported incomplete sums as integers. | **REPAIRED.** When any provider call is missing a usage receipt, token usage is reported as `"PARTIAL"` with `is_complete: False` and `source: "partial_usage_in_trace"`. Arm totals flag `token_accounting_complete: False` and report `"PARTIAL"`. |
| **4** | **Provider attempt tracing:** Ensure every actual provider attempt, including unsuccessful attempts, can be traced and reconciled with the durable journal. | `src/benchmark/free_pilot.py` (`JournaledProvider`), `src/agents/document_agent.py`, `scripts/evaluate_live_paired.py` | **CONFIRMED DEFECT.** `JournaledProvider` did not attach its unique attempt ID `f"{run_id}:{ordinal}"` to responses, and `DocumentAgent` did not record `attempt_id` in trace, leaving `provider_attempt_ids` empty (`[]`) in task records. | **REPAIRED.** `JournaledProvider` attaches `provider_attempt_id` to responses and maintains internal `attempts` log. `DocumentAgent` propagates attempt ID into trace. `evaluate_live_paired.py` reconciles every event's `provider_attempt_ids` directly with rows in `pilot_calls`. |
| **5** | **Benchmark isolation:** Confirm no-cache, exact-only, and shadow-semantic arms begin with correct independent cache states and identical workload inputs. | `scripts/evaluate_live_paired.py` (`_run_arm`) | **PARTIAL DEFECT (HYGIENE).** Stale `answers.db` and `tasks.db` were unlinked, but auxiliary SQLite files (`-wal`, `-shm`, `-journal`) could persist across runs. Also ensured workload task dictionaries are deep-copied per arm to avoid mutation leakage. | **REPAIRED.** Cleaned up all auxiliary SQLite journal and WAL files. Workload inputs are deep-copied per arm. Cache modes cleanly set (`off`, `exact`, `shadow`). |
| **6** | **Receipts & reproducible metadata:** Ensure immutable results, correct latency units, actual responses, terminal states, and reproducible execution metadata. | `scripts/evaluate_live_paired.py` (`run_paired_study`) | **PARTIAL DEFECT (HYGIENE).** Base run ID, journal path, and max study budget were missing from top-level receipt metadata. Cache lookup tier extraction was hardcoded to `trace[0]`. | **REPAIRED.** Added `base_run_id`, `journal_path`, `max_study_budget`, `failed_tasks`, `rejected_tasks`, and `non_terminal_tasks` to receipt. Cache lookup tier dynamically resolved. Timestamps and millisecond wall-clock latencies verified. |

---

## 2. Minimal Implementation Fixes

### A. `src/benchmark/free_pilot.py`
- Added unique attempt identifier generation `attempt_id = f"{self.journal.run_id}:{ordinal}"` in `JournaledProvider.__call__`.
- Attached `attempt_id` and `provider_attempt_id` to all fresh and replayed response payloads.
- Maintained durable attempts history `self.attempts` tracking `attempt_id`, `ordinal`, `state`, and `replayed` flags.

### B. `src/agents/document_agent.py`
- Propagated `attempt_id` / `provider_attempt_id` from model response payloads into task trace entries for planning, drafting, and reviewing stages.

### C. `scripts/evaluate_live_paired.py`
- Fixed quota double-counting in `run_paired_study` by inspecting `initial_journal_calls` and evaluating `max(current_journal_calls, initial_journal_calls + cumulative_calls)` against `self.max_study_budget`.
- Replaced permissive `TERMINAL` check with strict validation: only all-`COMPLETED` tasks produce `final_status = "COMPLETED"`; failed tasks produce `"FAILED"`, rejected tasks produce `"REJECTED"`, non-terminal tasks produce `"INCOMPLETE"`.
- Refined `_extract_usage_from_trace`: skips non-provider trace stages (e.g. `human_decision`); detects partially missing receipts and returns `"PARTIAL"`, `is_complete: False`, and `source: "partial_usage_in_trace"`.
- Enhanced `_run_arm` cold-start isolation to unlink SQLite `-wal`, `-shm`, and `-journal` auxiliary files.
- Reconciled task `provider_attempt_ids` with journal `pilot_calls` records.

---

## 3. Test Verification & Results

### Dedicated P1.1 Acceptance Suite (`tests/test_p1_1_harness_acceptance.py`)
```powershell
python -m pytest tests/test_p1_1_harness_acceptance.py -v
```
**Result:** **16 passed in 24.79s** (100% pass rate, zero live API calls):
1. `TestQuotaAccounting::test_quota_accounting_all_three_arms_complete_without_double_counting` PASSED
2. `TestQuotaAccounting::test_quota_guard_blocks_when_prior_external_calls_reach_budget` PASSED
3. `TestCompletionStatusIntegrity::test_failed_task_marks_study_failed` PASSED
4. `TestCompletionStatusIntegrity::test_rejected_task_marks_study_rejected` PASSED
5. `TestCompletionStatusIntegrity::test_awaiting_approval_task_marks_study_incomplete` PASSED
6. `TestProviderTokenAccounting::test_partially_missing_usage_reported_as_partial_not_complete` PASSED
7. `TestProviderTokenAccounting::test_all_missing_usage_reported_as_unknown` PASSED
8. `TestProviderTokenAccounting::test_complete_usage_reported_as_valid_integers` PASSED
9. `TestProviderTokenAccounting::test_cache_hit_zero_calls_reported_as_zero_tokens` PASSED
10. `TestProviderAttemptTracing::test_journaled_provider_records_attempt_id_and_tracks_attempts` PASSED
11. `TestProviderAttemptTracing::test_journaled_provider_records_error_attempt` PASSED
12. `TestProviderAttemptTracing::test_replayed_calls_reconciled_with_attempt_id` PASSED
13. `TestProviderAttemptTracing::test_full_agent_workflow_attempt_reconciliation` PASSED
14. `TestBenchmarkIsolation::test_cold_start_auxiliary_sqlite_files_cleaned` PASSED
15. `TestBenchmarkIsolation::test_workload_tasks_not_mutated_across_arms` PASSED
16. `TestReceiptIntegrity::test_study_receipt_contains_all_execution_metadata` PASSED

### Prior 8-Issue Acceptance Suite (`tests/test_harness_acceptance_8.py`)
```powershell
python -m pytest tests/test_harness_acceptance_8.py -v
```
**Result:** **24 passed in 143.31s** (100% pass rate).

### P1 Acceptance Suite (`tests/test_p1_acceptance.py`)
```powershell
python -m pytest tests/test_p1_acceptance.py -v
```
**Result:** **14 passed in 10.63s** (100% pass rate).

### Full Regression Suite (15 Test Files)
```powershell
python -m pytest -q -p no:cacheprovider tests/test_hybrid_cache.py tests/test_answer_evidence.py tests/test_answer_aware_policy.py tests/test_response_cache.py tests/test_sqlite_response_cache.py tests/test_semantic_runtime.py tests/test_durable_semantic_rebuild.py tests/test_before_agent_cache.py tests/test_document_orchestration_memory.py tests/test_document_agent.py tests/test_agent_workflow.py tests/test_static_document_stage.py tests/test_p1_acceptance.py tests/test_harness_acceptance_8.py tests/test_p1_1_harness_acceptance.py
```
**Result:** **321 passed, 2 warnings in 38.51s** (100% pass rate).

---

## 4. Preflight Readiness Verification

Preflight execution of `python scripts/evaluate_live_paired.py --check-only` verified the safety envelope:
- **Status:** `BLOCKED` (Safety default active; live calls require explicit `--execute-live` flag)
- **Spend Cap:** `$0.00` (Strictly zero paid fallback)
- **Model Allowlist:** `nvidia/nemotron-3-super-120b-a12b:free`
- **Output Token Ceiling:** 2048 (> 0)
- **Proposed Pilot Calls:** Exactly 24 requests across 3 arms (≤ 30 budget cap)
- **Receipts Persisted:**
  - `artifacts/live_evaluation/live_readiness_receipt.json`
  - `artifacts/live_evaluation/live_readiness_receipt_20261109T122827Z.json`

---

## 5. Secret Scan & Hygiene Disclosures

- `git diff` scan scope: all modifications across `scripts/evaluate_live_paired.py`, `src/agents/document_agent.py`, `src/benchmark/free_pilot.py`, `tests/test_p1_1_harness_acceptance.py`, and `docs/audit/`.
- Zero credentials, API keys, or raw provider secrets found.
- Zero live network API calls executed during evaluation or tests.

---

## 6. Handoff & Acceptance Verdict

- **P1.1 Acceptance Verdict:** **PASS**
- **Remaining Blockers:** None for Phase 1.1. Live evaluation harness is verified, reconcilable, and fail-closed.
- **Next Phase:** Live pilot execution remains BLOCKED pending explicit operator authorization.
