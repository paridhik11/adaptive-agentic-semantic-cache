# ROADMAP_ABC.md: Run book for Antigravity

Project: adaptive-agentic-semantic-cache (public repo, MIT).
Goal: a safe semantic-cache gateway that cuts LLM calls and latency for real traffic and for agent workloads, validated on public datasets, with every number traceable.

Phases 1 to 8 are finished history. Do not rename or rewrite them. New work is called **Phase A, Phase B, Phase C**. New FACTS.md row IDs use the prefixes `PA-`, `PB-`, `PC-`.

---

## HOW TO USE THIS FILE (for the human)

1. Put this file at `docs/ROADMAP_ABC.md` in the repo and commit it to main.
2. Paste the **Starter message** (bottom of this file) into Antigravity.
3. Antigravity runs Phase A, then B, then C without asking questions, but only while the **gates** pass. If a gate fails it writes `docs/BLOCKED_<phase>.md` and stops.
4. At the end of each phase it writes `docs/audit/AUDIT_PACK_<phase>.md`. Paste that file to Claude (your auditor). Recommended: paste Pack A before letting B start, because A decides whether B and C make sense. B and C can then run back to back.
5. Do not edit the resume or the README headline until Claude has seen the audit packs.

---

## GLOBAL RULES (apply to every phase)

### Git and security
- Work directly on `main`. No branches.
- Small commits, one logical unit each, descriptive messages.
- Before every push: `gitleaks protect --staged --no-banner`. Before the final push of each phase also run `gitleaks detect --source . --no-git` and `gitleaks detect --source . --log-opts "--all"`. (Gitleaks is v8.18.4 at `%TEMP%\gitleaks_bin\gitleaks.exe`; the `gitleaks dir` command does not exist in this version.) Never use `.gitleaks.toml` allowlists to hide a finding.
- Never write secrets into any file. Use `.env` (ignored) and `.env.example`.
- Write files with the file-edit tool or Python `open(..., encoding="utf-8")`. Do NOT use PowerShell here-strings (they corrupted a code fence earlier).

### Truthfulness rules
- Every number in any document must have a row in `docs/FACTS.md` (ID, value, source file, exact command to recompute) BEFORE it appears elsewhere.
- Verify with `python scripts/check_numeric_tokens.py` (strict token set difference). Never use substring matching to "pass" the check.
- Never delete a documented number to pass the check. Ledger it from its source or write "not ledgered" with no number.
- Every result carries an evaluation label: `same-set`, `validation`, `held-out`, `sealed`, or `simulated`. Rates carry n and an exact Clopper-Pearson 95% CI.
- No claim stronger than the evidence. A point estimate below a target is not "clearing" the target unless the CI upper bound is below it.
- Do not edit results of Phases 1 to 8. Errors are corrected only with a dated erratum plus a FACTS row.
- Record failures, discarded attempts and surprises in the phase walkthrough. A phase with no failures recorded is suspicious.
- Do not fabricate, simulate or "estimate" a result and present it as measured. Simulated components must be labelled `simulated` everywhere they appear.
- Dollar figures: only tokens avoided x a published price with source and date, labelled "estimate". Otherwise leave dollars out.

### Data rules
- Use public datasets only. Before use, read the official dataset card, record licence and access terms in FACTS.md. If a dataset is gated, forbids use, or is unavailable, document it in `docs/DECISIONS.md` and substitute; do not scrape websites.
- Do not commit large data. Commit download scripts plus SHA-256 checksums.
- Splits: train, validation (all tuning happens here), sealed test. A sealed test file is hashed and committed together with its pre-registration BEFORE any run on it. It is run once. Any re-run is logged as a deviation.
- If a pretrained checkpoint was trained on a dataset, say so and never evaluate on that dataset's training portion.

### Autonomy rules
- If something is ambiguous and a reasonable assumption exists, take it, write it in `docs/DECISIONS.md` (date, assumption, reason), and continue.
- If a gate fails, a licence blocks the plan, or a quota makes the plan impossible, write `docs/BLOCKED_<phase>.md` (what happened, raw evidence, options) and STOP.
- Free-tier API quotas apply (about 50 requests/day on the OpenRouter free tier). Spread calls over days if needed, log the dates, and never fabricate fallback results.

### Audit pack (end of every phase)
Create `docs/audit/AUDIT_PACK_<A|B|C>.md` with raw output, not summaries:
1. `git log --oneline -20` and `git status`.
2. Every new FACTS row, with the recompute command AND its raw output.
3. Raw output of: pytest summary, strict token check, all gitleaks commands.
4. Files created and modified.
5. Failures, discarded attempts, deviations from the plan, and every assumption in DECISIONS.md.
6. A section titled "Claims I am least sure about".
7. A section titled "What the auditor should check first".
Also update `docs/STATUS.md` (overall % of this roadmap complete, current phase, gate status).

---

# PHASE A: Make the matching trustworthy and measure it on real data

Why: the Phase 8 prototype folded exact repeats well, but only 9 of its 50 hits came from semantic matching and 3 of those 9 were wrong. It was evaluated on a stream we built. Phase A fixes the matcher and tests it on public data we did not design.

## A0. Preflight and Phase 8 repairs
1. Confirm clean tree, tests green. Re-run Phase 8's pipeline from scratch and confirm it reproduces its FACTS rows (`P8-*`). Report any mismatch.
2. Replace the wording "6.0% clears the 10% ceiling" everywhere with the CI-based statement (CI [1.3%, 16.5%] does not establish the ceiling). Use an erratum, not a silent edit.
3. Break down the Phase 8 test stream by category (cold seeds, exact repeats, paraphrases, PAWS adversarial, MS MARCO): hits, false positives, per category. State which category produced each of the 3 false positives, with the query pairs shown.
4. Explain, with data, why the validation set at threshold 0.85 had about 39.6% incorrect reuse while the test stream had 6.0% (composition, guards, which hits were trivial). Run the pipeline on the validation set with each guard on/off.
5. Remove the "$0.011 per 1,000" figure unless it is recomputed with a price source and date for the same model class.
6. Re-measure the no-cache baseline with n >= 100 real LLM calls (state the model, temperature, max_tokens, dates, hardware). Report p50/p95 with the sample size.

## A1. Better semantic matching
1. Candidates (choose on validation only, never on sealed): (a) off-the-shelf `cross-encoder/quora-distilroberta-base`; (b) the same model fine-tuned on QQP train plus hard negatives from PAWS train; (c) (b) plus the guards. Document training data and seeds. Keep training to CPU-feasible sizes and record time.
2. Guards (numbers, negations, entities) stay. Add a **long-prompt guard**: if either prompt exceeds the embedding model's input window, compare a full-text SHA-256 and the tail, and only reuse on exact or canonical equality.
3. Add a **generative-request guard**: prompts asking for creative or open-ended generation ("write a poem...", "give me ideas...") are BYPASS. Evaluate it on a small hand-built labelled set (>= 100 prompts, written by you, committed).
4. Report precision, recall and the false-positive categories for each candidate on validation, with CIs.

## A2. Domain-level thresholds that can actually fire
1. Use CQADupStack (StackExchange duplicate questions across subforums) as the real multi-domain data. Treat each subforum as a category. Verify access and licence.
2. Using validation only, run the existing AdaptiveThresholdEngine per category with its gates (>= 20 samples, >= 10 minority). Report which categories now move off the 0.85 fallback and which still do not. If none fire, say so; do not weaken the gates to make them fire.

## A3. Natural repeat rates in real chat traffic
1. Verify access and licence for an open chat log (for example WildChat; the LMSYS-Chat-1M dataset is gated). If neither is usable, record the reason and use the best open alternative.
2. Measure, on a sample of >= 5,000 first user turns, how many are exact-canonical repeats and how many are semantic near-duplicates under each threshold, WITHOUT claiming correctness (no labels). Write `data/review/pa_human_audit.csv` with 100 sampled reuse decisions (query, matched query, decision, blank `human_label` column) for a human to fill later. Do not wait for it.

## A4. Pre-registration and the sealed test
1. Write `docs/phaseA_preregistration.md` BEFORE building the sealed set. Contents: primary endpoint = incorrect-reuse rate among NON-exact hits, with exact 95% CI; secondary endpoints = overall hit rate, LLM calls avoided per 1,000, end-to-end p50/p95 latency vs the no-cache baseline; frozen config (threshold values, model revisions, guard versions) and its hash; the stated acceptance criteria below; the analysis scripts.
2. Build the sealed stream (>= 2,000 queries) from public held-out slices not used in any tuning: QQP validation-or-held-out portion, PAWS test, CQADupStack held-out threads, MS MARCO queries, and the chat-log sample. State composition per source. Aim for >= 200 non-exact hits to be possible.
3. Commit the sealed file, its SHA-256, the frozen config, and the pre-registration in ONE commit before running anything on it.
4. Run once. Log the command and raw output.

Acceptance criteria to pre-register (report the outcome either way, never adjust afterward):
- Safety: upper bound of the 95% CI on non-exact incorrect reuse <= 10%.
- Utility: LLM calls avoided per 1,000 queries reported with CI, with the stream's repeat share stated.

## A5. Reporting
1. Baselines: (a) no cache; (b) fixed cosine only (0.85 and 0.92); (c) Phase 8 pipeline; (d) Phase A pipeline. Ablation: normalizer, + embedding, + guards, + cross-encoder (off-the-shelf), + cross-encoder (fine-tuned), + per-domain thresholds.
2. Metrics with n and CI: hit rate; incorrect-reuse among hits and among non-exact hits; per-source breakdown (including PAWS); calls and tokens avoided per 1,000 queries; end-to-end p50/p95 (decision time + LLM time for misses) vs no-cache; local-path latency by path with hardware stated.
3. A **hit-rate vs repeat-share curve**: from the stored per-query outcomes of the single sealed run, re-weight categories to repeat shares of 0%, 10%, 20%, 40%, 60% and plot/tabulate hit rate and end-to-end latency. Compute the **break-even hit rate** at which the cache's overhead equals the LLM time saved.
4. Docs: `docs/phaseA_walkthrough.md` (steps, results, issues faced and fixed, discarded attempts), FACTS rows `PA-*`, `docs/claim_check_A.md`. Do NOT change the README headline yet; add a "Phase A (in progress)" link only.

## GATE A (stop here unless all pass)
- Sealed set was committed before the run and was run exactly once.
- Strict token check passes; all gitleaks scans clean; tests green.
- >= 100 non-exact hits on the sealed run.
- Non-exact incorrect-reuse POINT estimate <= 10%.
If any item fails: write `docs/BLOCKED_A.md` with the raw findings and STOP (do not start Phase B). A bad result is still a valid, reportable result; do not tune on the sealed set to fix it.

Then write `AUDIT_PACK_A.md` and `STATUS.md`. If the gate passes, continue to Phase B.

---

# PHASE B: A drop-in gateway people can actually run

Why: a project is more convincing when a reviewer can point an OpenAI-compatible client at it and watch hits and misses.

## B1. Gateway service
1. FastAPI service exposing `POST /v1/chat/completions` in the OpenAI request/response format, plus `GET /health` and `GET /metrics` (Prometheus).
2. Providers: OpenRouter (real, free tier) and a **mock provider** whose latency and token counts are drawn from the measured baseline distribution (Phase A0, n >= 100). Everything produced with the mock provider is labelled `simulated` in every report.
3. Response headers on every call: `X-Cache: HIT|MISS|BYPASS`, `X-Cache-Tier`, `X-Cache-Reason`.

## B2. Cache key and safety policy (write each as a test first)
1. Cache scope key = tenant id + system-prompt hash + model + generation parameters. Different scope keys never share entries. **Tenant isolation test**: tenant A's cached answer is never returned to tenant B, including for identical prompts.
2. Caching policy by parameters: by default only cache when temperature <= a configurable cap (default 0.2); a request header can opt in or out. Document the policy.
3. Multi-turn: semantic reuse only for single-turn requests; for multi-turn, exact match on a hash of the full message history only.
4. Use the Phase A guards: long-prompt guard, generative-request guard, stability classifier (dynamic or time-sensitive prompts are BYPASS).
5. Skip caching on responses with finish_reason other than `stop`, tool calls, or errors. Never cache partial streams.
6. Simple privacy rule: do not cache prompts matching email or phone-number patterns; document it as a minimal measure, not a guarantee.

## B3. Streaming, TTL, invalidation
1. Hits return immediately. If the client asked for `stream=true`, replay the stored response as SSE chunks. Misses stream from the provider while buffering; store only complete successful responses.
2. TTL tiers from the stability classifier (stable: long TTL, uncertain: short TTL, dynamic: no cache). Configurable.
3. Invalidation endpoints: by tenant, by system-prompt hash, by model, by prefix/tag. Tests for each.

## B4. Observability and packaging
1. Prometheus metrics: hit/miss/bypass counts per tier, latency histograms per path, tokens avoided, similarity score histogram for hits and near-misses. Add a near-miss analyzer (queries just below threshold) and use it only to propose, never to auto-change, thresholds.
2. `docker-compose.yml` with the gateway, Prometheus and Grafana, with a provisioned dashboard. README section: run in two commands.

## B5. Load test (>= 2,000 requests)
1. Workload = the Phase A sealed-stream distribution (do not invent a new mix), run through (i) a no-cache pass-through gateway and (ii) the cache gateway, both with the real provider on a small sample (>= 100 requests, quota permitting) and with the mock provider for the full run (`simulated`).
2. Report: hit rate, p50/p95/p99 end-to-end latency, LLM calls and tokens avoided, the incorrect-reuse rate on the labelled subset, and the break-even computed in Phase A. State clearly what is simulated.
3. Concurrency test: 50 parallel clients, no cross-tenant leaks, no crashes. Record results.

## B6. Docs
`docs/phaseB_walkthrough.md`, FACTS rows `PB-*`, `docs/claim_check_B.md`, DESIGN.md gateway section. README: add a "Gateway" quickstart section but do not change the headline.

## GATE B
- Tenant isolation, streaming, invalidation and concurrency tests pass.
- Docker compose starts from a fresh clone (show the transcript).
- Load test completed and logged; strict token check passes; gitleaks clean.
Else write `docs/BLOCKED_B.md` and STOP.

Then `AUDIT_PACK_B.md` and `STATUS.md`. If the gate passes, continue to Phase C.

---

# PHASE C: The agentic layer, human feedback, and the final story

Why: agent workloads repeat sub-queries and tool calls, which is where a safe cache is genuinely useful. Human review also supplies the labels the adaptive thresholds lacked.

Scope limit: small and finished beats large and broken. Skip Celery, Postgres, a replay UI and a full review UI. Use LangGraph.

## C1. Minimal agent system
1. LangGraph graph: supervisor (decomposes a task into 2 to 4 subtasks), two specialists (research with a search tool, analysis/calculation), and a reviewer that accepts or rejects. All LLM calls go through the Phase B gateway.
2. Tools: Wikipedia/arXiv APIs with **recorded responses** for reproducibility (live mode optional), and a calculator. Tool-result cache: key = tool name + normalized arguments, with TTL; log hits and misses.
3. Logging: OpenTelemetry spans for each node and LLM call with attributes: cache tier, similarity, tokens, tokens avoided.

## C2. Human feedback loop
1. For AMBIGUOUS or borderline reuse decisions, write them to a review file (`data/review/pc_review_queue.csv`). A tiny CLI or Streamlit page shows both queries, the cached answer, and approve/reject buttons. Reviews are saved as labels.
2. Show offline that these labels feed per-category threshold calibration, using the existing engine and its gates. If labels are too few for any category to move, report that honestly. Provide the interface and the pipeline; do not invent labels. Seed it with the Phase A audit CSV if the human has filled it.

## C3. Agent evaluation (preregister first)
1. Tasks: 150 multi-step questions from HotpotQA (verify licence) plus 50 paraphrased variants of the same questions (to create realistic reuse opportunities). Pre-register the design in `docs/phaseC_preregistration.md` before running: arms (cache on vs off), metrics, analysis plan.
2. Run both arms on the same tasks, same order, same seeds. Report per task: LLM calls, tokens, wall-clock, task correctness (exact match and F1 vs the gold answer).
3. Primary comparison: task correctness with the cache on vs off, as a paired comparison (report the difference with a bootstrap CI; use McNemar's test for correctness). Secondary: calls, tokens and time per task with CIs. State that the model is a free-tier model and the sample is small.
4. Hand-check: write 40 randomly chosen cache-on answers that differed from cache-off answers into `data/review/pc_manual_check.csv` for a human to review. Do not wait for it.

## C4. Final documents (only from FACTS rows)
1. `docs/phaseC_walkthrough.md`, `docs/claim_check_C.md`, FACTS rows `PC-*`.
2. Update `DESIGN.md` (merged architecture: agents, gateway, matcher, human feedback), the demo (replay mode, no API key needed), and a README rewrite whose headline results come ONLY from Phase A/B/C FACTS rows, each with its evaluation label and CI. Keep the Phase 1 to 8 table as "foundation results". Keep a limitations section.
3. `docs/MASTER_REPORT.md`: bring it up to date (this time every metric sentence carries a FACTS ID; fix any wording that implies a CI-overlapping difference is established).
4. Write `docs/RESUME_DRAFT.md` marked "NOT APPROVED, for auditor review". Include only metrics with FACTS rows, the "defend it" notes, and a "do not claim" list.
5. Repo hygiene: add GitHub About description and topics (in a checklist for the human, since this cannot be set from git), check the Actions workflow is green, remove stale scratch files.

## GATE C
- Agent tests (decomposition, tool use, reviewer rejection, escalation to the review queue, failure recovery) pass; the cache-on arm does not reduce task correctness by more than the pre-registered margin (pre-register the margin; suggested 3 percentage points) OR the report states plainly that it did.
- Strict token check passes on every changed document; gitleaks clean; fresh-clone test passes (clone, install, run the demo).
Else write `docs/BLOCKED_C.md` and STOP.

Then `AUDIT_PACK_C.md` and `STATUS.md`.

---

## Starter message (paste into Antigravity)

```
Read docs/ROADMAP_ABC.md completely. Follow its GLOBAL RULES strictly.
Execute Phase A now, then Phase B, then Phase C in order, without asking me
questions. After each phase write docs/audit/AUDIT_PACK_<phase>.md and update
docs/STATUS.md. Check each GATE; if a gate fails, write docs/BLOCKED_<phase>.md
and stop. Do not change the README headline or write resume content as final
until I approve. Work directly on main. Report only when a gate fails, a phase
completes, or you are blocked.
```

## What to paste back to Claude

- After Phase A: the full text of `docs/audit/AUDIT_PACK_A.md` (and `docs/BLOCKED_A.md` if it exists).
- After B and C: the two remaining audit packs and `docs/STATUS.md`.
- If you can, also paste `docs/phaseA_preregistration.md`, so the auditor can check the targets were fixed before the run.
