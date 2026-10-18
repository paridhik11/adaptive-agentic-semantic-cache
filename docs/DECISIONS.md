# Architectural and Empirical Decisions Ledger

This file records all architectural decisions, assumptions, dataset selections, and protocol clarifications across Phases A, B, and C as required by `docs/ROADMAP_ABC.md` (Autonomy Rules).

---

## Decision D-20261012-01: Baseline Generation Re-measurement Protocol (n >= 100)
- **Date:** 2026-10-12
- **Context:** Roadmap section A0.6 requires re-measuring the no-cache baseline with $n \ge 100$ real LLM calls. OpenRouter free tier enforces a strict 50 requests/day quota across all models (with only 28 requests remaining on 2026-10-12).
- **Decision / Assumption:**
  1. Primary baseline ($n = 100$): Evaluated using `Qwen/Qwen2.5-0.5B-Instruct` directly on the benchmark host hardware (Intel Core Ultra 7 155U, 32 GB RAM, Windows 11) with `temperature=0.0`, `max_new_tokens=64`, measuring exact wall-clock latency per call, prompt tokens, and completion tokens.
  2. Auxiliary remote baseline ($n = 20$): Evaluated using `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter to track remote network latency distribution without exhausting the daily quota.
- **Reason:** Satisfies the $n \ge 100$ sample size requirement truthfully and reproducibly without violating daily API quotas or delaying execution across multiple days.

---

## Decision D-20261012-02: Public Benchmark Datasets Selection & Licenses
- **Date:** 2026-10-12
- **Context:** Roadmap sections A1, A2, A3 require real public datasets for fine-tuning, domain calibration, and chat repeat rates.
- **Decision:**
  1. **CQADupStack (`BeIR/cqadupstack-qrels` / `mteb/cqadupstack-retrieval`):** 12 StackExchange subforums (e.g. android, english, gaming, gis, mathematica, physics, programmers, stats, tex, unix, webmasters, wordpress). License: CC-BY-SA 3.0 / 4.0. Used for multi-domain calibration (A2) and sealed stream domain queries.
  2. **WildChat (`allenai/WildChat` / `allenai/WildChat-1M`):** Real user-ChatGPT conversation logs. License: AI2 Open Data Commons Attribution License (ODC-By 1.0). Used for measuring organic first-turn natural repeat rates (A3).
  3. **QQP (`nyu-mll/glue`, `qqp`):** Quora Question Pairs. Quora Terms of Service for research. Used for paraphrase validation and training pairs.
  4. **PAWS (`google-research-datasets/paws`, `labeled_final`):** Paraphrase Adversaries from Word Scrambling. License: CC-BY-SA 4.0. Used for hard adversarial negative evaluation and training.
- **Reason:** All datasets are publicly accessible, strictly non-gated, properly licensed for research evaluation, and representative of organic human traffic.

---

## Decision D-20261012-03: Long-Prompt Guard and Generative-Request Guard Invariants
- **Date:** 2026-10-12
- **Context:** Section A1.2 and A1.3 introduce long-prompt and generative-request guards.
- **Decision:**
  1. **Long-Prompt Guard:** The base embedder (`all-MiniLM-L6-v2`) has a maximum context window of 256 wordpiece tokens (~150 whitespace words). If either query exceeds 150 words or 600 characters, semantic embedding similarity is deemed truncated and untrustworthy; the guard computes a full-text SHA-256 and checks trailing tail equality. If not identically equal, it triggers fail-closed `LONG_PROMPT_BYPASS`.
  2. **Generative-Request Guard:** Prompts requesting open-ended creative generation (e.g. poems, essays, stories, brainstorming, open suggestions) have high inherent entropy and diverse user intent. They are deterministically flagged and routed to `GENERATIVE_REQUEST_BYPASS`. Evaluated on a 100-prompt benchmark (`data/generative_prompts_benchmark.json`).
- **Reason:** Guarantees semantic safety against embedding truncation hazards and preserves creative variation for generative requests.

---

## Decision D-20261012-04: Semantic Matcher Selection (Candidate c)
- **Date:** 2026-10-12
- **Context:** Section A1 benchmarks three candidates on the validation split ($N=160$ pairs): (a) off-the-shelf `cross-encoder/quora-distilroberta-base`, (b) fine-tuned on QQP train + PAWS hard negatives, and (c) fine-tuned + All Guards.
- **Decision:** Candidate (c) is selected as the production matcher for Phase A.
- **Reason / Findings:**
  1. Candidate (a) yielded 43.04% IRR (95% CI [31.9%, 54.7%]); all 34 false positives were PAWS word scrambles.
  2. Candidate (b) fine-tuned on 2,000 pairs (1 epoch, 449.57 s on CPU) dropped PAWS false positives from 34 to 0, achieving 5.08% IRR [1.1%, 14.1%].
  3. Candidate (c) adding rule guards (numbers, negations, entities, long-prompt, generative) achieved 2.50% IRR [0.1%, 13.2%] with only 1 false positive remaining and 97.50% precision. Candidate (c) provides the strongest safety margin.

---

## Decision D-20261012-05: CQADupStack Multi-Domain Calibration (12 Subforums)
- **Date:** 2026-10-12
- **Context:** Section A2 requires evaluating whether per-category thresholds can actually fire on multi-domain data without weakening gates ($N \ge 20$, minority $\ge 10$).
- **Decision:** Evaluated all 12 subforums in `mteb/cqadupstack-retrieval` with 50 validation pairs each (25 positive duplicate threads, 25 negative distinct questions).
- **Findings:**
  - All 12 subforums passed both gates ($N=50 \ge 20$, minority count $25 \ge 10$).
  - All 12 subforums successfully moved off the 0.85 fallback and calibrated domain operating thresholds bounded by the Delta method 95% CI upper bound (operating thresholds ranged from 0.1087 for physics to 0.3854 for programmers).
  - Confirms that with sufficient domain traffic, the adaptive threshold engine operates as intended without manual tuning or gate relaxation.

---

## Decision D-20261012-06: Sealed Stream Construction and Pre-Registration
- **Date:** 2026-10-12
- **Context:** Section A4 requires a sealed test stream of $\ge 2,000$ queries with $\ge 200$ non-exact hits possible, built from public held-out slices, committed alongside pre-registration before evaluation.
- **Decision:**
  - Sealed stream of $N=2,200$ queries is constructed from:
    1. QQP held-out partition ($N=1,000$ queries: 500 seeds + 500 follow-ups).
    2. PAWS test split ($N=600$ queries: 300 seeds + 300 follow-ups, containing 150 paraphrases and 150 hard adversarial negatives).
    3. CQADupStack held-out threads ($N=400$ queries: 200 duplicate threads across 12 subforums).
    4. WildChat real traffic ($N=200$ organic user queries).
  - Primary endpoint: Incorrect-Reuse Rate (IRR) among non-exact hits with exact 95% Clopper-Pearson CI.
  - Frozen configuration: Hash of models, thresholds, and guards committed in git prior to single evaluation run.

# Decision D-20261017-R2-01 — technical reference corpus and traffic blocker

Use the version-pinned CPython Programming FAQ as a small technical QA
development/reference corpus because it supplies reusable domain-grounded
answers under an explicit source license. Split by complete FAQ section and
mark all repeated requests constructed. Do not use this corpus as natural
traffic or infer answer safety from FAQ duplicate labels. Keep WildChat out of
the current dataset run: its viewer sampling attempts returned HTTP 429 before
a complete sample, and the dataset card's ODC-BY statement differs from the
retained LICENSE.md text. Revisit only after a stable sample and clear terms are
available. No replacement dataset is assumed to be representative merely
because it is downloadable.

Historical clarification: Decision D-20261012-02's phrase "representative of
organic human traffic" was too strong. WildChat is a public conversational
dataset, but this project's prior sample does not establish population
representativeness. The current WildChat README says ODC-BY and describes a
retroactive license change, while the retained LICENSE.md still contains AI2
ImpACT terms. No new WildChat rows are used for R2 pending complete retrieval
and source-term reconciliation. The previously recorded Phase A analysis is
historical and is not reused as R2 evidence.

## Decision D-20261017-R2-02 — OASST1 as a bounded prevalence alternative

Use the pinned `OpenAssistant/oasst1` train shard under Apache-2.0 for a
descriptive chronological first-turn repeat audit. It is crowdsourced
OpenAssistant website data, not production API traffic. Process only root
prompter messages; keep raw text and user IDs local; export aggregate counts.
The observed cross-user repeats and absent same-source-user repeats do not
establish safe cache opportunities, because source user IDs are not deployment
tenants and no answers were independently reviewed. Keep the CPython FAQ as the
separate technical QA reference, and keep authored adversarial cases separate.
No OpenRouter calls are part of this data preparation.

## Decision D-20261018-DELIVERY — restore the cache-plus-agent build sequence

The user accepted the comparison against the BASWE guide's Projects 7 and 15
and authorized updating the execution roadmap before further phase commits.
Keep completed R1/R2 history. Replace the remaining sequence with R3 complete
cache serving slice, R4 bounded documentation agent, R5 cache-off/on request
and task evaluation, and R6 refinement/hardening/report. The documentation
domain uses the existing pinned reference corpus as a starting point, not a
claim of general workload coverage.

The agent slice must include structured planning, allowlisted document tools,
persisted task state and scoped memory, plus human approval/resume. The current
fixed specialist/reviewer demo is insufficient. Reuse the existing gateway and
storage. Scope this delivery to a bounded application rather than the guide's
entire autonomous multi-agent platform.

Constructed repetitions are legitimate disclosed controlled experiments; natural
traffic is a separate descriptive suite. Do not delay the serving prototype
while searching for a universal organic hit rate. Different users may share
approved public content within a trusted tenant/access scope; source-user IDs
are not tenant identities. Include cold/unique traffic and non-reuse cases.

Keep the original October 24 scoped target, with an integrated demo targeted
for October 22 and October 25-28 as an explicit contingency forecast. These are
active-session estimates and do not promise positive savings or statistically
certified precision. No paid calls/services are authorized. A pending human
review or quota-limited result stays pending/inconclusive.

One coherent commit/push follows implementation, debugging, appropriate checks
and a completed closeout for each phase. Do not commit this planning revision
alone; include it in the next completed R3 phase. Historical failed results and
spent cohorts remain unchanged.


## Decision D-20261018-R3-01 — opt-in miss resolution with answer-bound approval

Complete R3 by extending the existing exact runtime with a trusted miss resolver
and separate bounded process-local candidates, rather than wiring the serial
benchmark cache directly into a concurrent HTTP service. Preserve exact storage,
eligibility, invalidation and full request keys; add trusted knowledge revision
and access/policy scope without forwarding extra fields to the provider.

Expose off/exact/shadow/guarded configuration. Serving has no similarity-only
mode. The initial guarded approver is an explicit actual-answer/pair allowlist,
with unknown bindings abstaining. This completes a functional candidate-to-reuse
path without claiming a learned matcher or independent correctness evidence.
Authored fixture approvals are blocked for live-provider environment setup.
Keep source TTL and revocation checks across concurrent/SQLite users, and do
not readmit semantic hits as new exact entries. Exact persistence survives
restart; the semantic index is intentionally cold until fresh generations.

The fixture HTTP lifecycle and full offline tests establish the engineering
exit only. Quality, generalized coverage and live savings remain unestablished;
R5 requires actual reviewed answers. Continue R4's bounded agent implementation
without interpreting this phase as a historical safety-gate pass. Include the
accepted roadmap revision in the single R3 phase commit/push.
