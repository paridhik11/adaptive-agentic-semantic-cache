# Project 7 + Project 15 — final handoff for the next chat

Prepared 2 November 2026 (Asia/Kolkata). This is a handoff, not a claim that the project is finished.

## 1. Read this first

The user needs defensible resume metrics urgently and wants continuation of the existing project, not another redesign. Substantial systems engineering exists. **Automatic, useful semantic answer reuse is still unresolved.** Do not present test counts, exact hits or approval-bound semantic fixtures as solving it. Do not repeatedly ask the user to restate the project.

Repository: `paridhik11/adaptive-agentic-semantic-cache`.
Working branch: `integration/documentation-rebuild`.
Verified implementation head before this report: `c8083960892b9818856ab40a7dbd692d8011fb9e`.

The user explicitly authorizes publishing committed history INCLUDING review data to this repository/branch. Do not merge main, force-push or rewrite local history. Earlier GitHub pauses are superseded by that explicit authorization. Nevertheless, automated approval review has rejected connector uploads citing the earlier pause. Respect actual tool rejections; do not bypass them.

## 2. Exactly what the product is

A caching gateway in front of a bounded, version-aware documentation assistant. After an answer is generated and admitted, an eligible equivalent request should reuse that actual answer without remote planning, generation or judging. Changed direction, constraints, source versions, tenant or context must not receive an inappropriate cached answer. A miss executes fresh agent work.

Project 7 supplies scoped exact and semantic lookup, persistence, expiry, invalidation, concurrency coordination and measured work avoidance. Project 15 supplies supervisor/role routing, specialists, tools, reviewer, working/task memory, human approval and execution traces. The agent is part of the combined product, not an optional unrelated demo.

Answer caching and task memory are distinct: the former can avoid generation; the latter provides context and may still need generation. A similarity score alone does not establish that an answer satisfies the new question.

Example: list-to-tuple paraphrases may share an answer. Tuple-to-list reverses direction. Nested list-to-nested tuple conversion adds a requirement that the ordinary conversion answer may not meet.

## 3. Roadmap provenance and lost material

The user remembers a six/seven-group combined plan associated with reported commit `bd2782b` and links titled “Implementation plan” and “Complete backup”. That exact plan and commit were NOT recovered in the available saved history. Do not invent their contents or deny the user's recollection.

The later repository organizes implementation into four groups: durable caching; before-agent lookup/accounting; orchestration/memory; evaluation/release. This is not proof that the earlier plan had only four groups. Earlier ten-day scheduling and Stage A/B research are separate structures. A prior claim of 243 tests at `f9fcc9b` was not recovered; the independently rebuilt Group 1 reports 242 tests. Preserve that distinction.

## 4. Implemented work

| Area | Delivered | Limits |
|---|---|---|
| Gateway | Python/FastAPI gateway, provider adapter, scoped exact cache, eligibility, complete-response admission, streaming bypass | Streaming responses are not cached; no production certification |
| Durable lifecycle | SQLite exact storage and optional durable semantic candidates; source admission identity, TTL, eviction, replacement, invalidation, final revocation checks | General semantic approval still unqualified; physical DB disk quota not established |
| Concurrency | Process-local coordination of identical misses | Not distributed single-flight |
| Before-agent lookup | Completed-answer lookup before planning on explicit standalone tasks; full context/version scope | Contextual tasks still run fresh workflow |
| Approval | Durable approve/modify/reject; fresh approval task even on reused answer; rejection/edit epochs revoke stale reuse | Human edits not silently promoted as validated answers |
| Work accounting | Actual provider-stage attempts, failures and available usage; local stages and tool work | Provider-internal behavior and missing token usage cannot be inferred |
| Agent | Bounded planner, source researcher, answer writer, reviewer, typed read-only search/read tools, checkpoints and traces | Deterministic role routing; not a demonstrated unrestricted autonomous multi-agent system |
| Memory | Scoped persistent vector index over approved source tasks, frozen context snapshots, authenticated list/clear, revocation | Demo uses lexical hashing; genuine dense semantic-memory quality not established |
| UI | Local question/source/answer/approval workbench, matched-question and mode labels, context controls | Browser visual/responsive QA and public deployment unfinished |
| Packaging | Installed wheel tested outside source checkout in fresh minimal web environment; packaged assets and HTTP flow passed | Full research/ML dependency installation not verified |

Primary implementation: `src/policy/{response_cache,sqlite_store,semantic_runtime,work_ledger}.py`, `src/agents/{document_agent,document_answer_cache,document_orchestration,task_memory}.py`, `src/api/demo.py` and workbench assets.

## 5. What was attempted and did not establish success

- Historical Phase A/earlier matching studies failed safety/generalization gates. Do not reuse old advertised latency percentages as application savings.
- Stage A established reproducible intent baselines and artifact inventories; it did not qualify safe semantic serving.
- Stage B used synthetic Python-operation cohorts, template-held-out evaluation and three LoRA seeds (11,29,47). Train fit reached 100%; primary held-out four-class agreement was about 46.31% dev / 45.45% calibration. These are synthetic diagnostic labels, not human accuracy. Prespecified calibrated policies had zero useful semantic coverage. No promotion.
- Approved MiniLM and Quora checkpoints were restored and actually executed. A constructor incompatibility was repaired: explicit raw-logit activation and one sigmoid, safetensors only.
- Exposed 30-question real-model diagnostic: fixed dense .8 / pair .7 / margin .05. Dense-only recommended 3 answers: 2 human-sufficient, 1 insufficient. Quora-only and combined recommended zero. H12 nested conversion was the unsafe dense match, cosine 0.9234989537385168. This is not a fresh final evaluation or population precision estimate.
- Zero coverage is a failed utility result, not a safety success. No thresholds were lowered on the reviewed set.
- Earlier live pilots were incomplete (including quota exhaustion, output truncation and provider failure). They do not establish a completed live semantic savings comparison. Do not resume/reset their journals.
- Chromium browser installation returned an invalid ZIP archive. Browser QA has not passed.
- Git publication remains incomplete. CLI push lacks credentials; connected GitHub uploads have also been blocked by automated review despite explicit authorization. Some unreferenced blobs were uploaded, but no integration branch head was published at the last verified remote check.
- The assistant repeatedly confused roadmap versions and emphasized infrastructure progress while the main semantic feature remained unresolved. Do not repeat this reporting failure.

## 6. Data, models and review provenance

Current product diagnostic uses three bundled CPython 3.12 FAQ excerpts: octal/hex literals, tuple/list conversion and negative indexing. Source revision: `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`. This is narrow documentation coverage, not arbitrary-domain validation.

30 questions were model-authored and self-reviewed separately. The user then reviewed a shuffled form: **14 sufficient / 15 insufficient / 1 uncertain**. All 29 definite human labels agree with the model labels; H05 remains uncertain. This is **human review of model-authored questions**, not human-authored independent ground truth, and agreement is not cache-policy precision. Preserve rationales and nulls. No further labels are needed to complete this already returned packet.

Data: `data/document_product_review/`; reconciliation: `docs/HUMAN_MODEL_REVIEW.md`; raw results: `artifacts/documentation_rebuild/group4/human-review-v1/` and `real-model-diagnostic-v1/`.

Pinned approved model revisions:
- sentence-transformers/all-MiniLM-L6-v2: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- cross-encoder/quora-distilroberta-base: `f62e7a4b20b97195c2868e53ec59126df5eac743`
- Earlier baseline-only approvals: redis/langcache-embed-v1 `2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a`; redis/langcache-reranker-v2-bce `f59539f4208f9005e10e45160a379441bdb42e9b`. Redis was not rerun in current product diagnostic.

No additional checkpoint approval is implied. Keep trust_remote_code off, safetensors only, revisions explicit. Quora/Redis public-pair comparisons carry possible-pretraining-overlap flags. Model weights can disappear on workspace reset; verify availability before claiming readiness.

## 7. Defensible metrics available now

| Observation | Evidence type and scope | Source |
|---|---|---|
| SQLite cache-hit p95 1.78591505 ms (resume: 1.79 ms), p50 1.273999 ms, 5,000 hits | Measured local component; five runs × 1,000 warm hits; three keys; single managed container | `artifacts/resume/cache-benchmark-v1.json` |
| In-memory p95 .039598 ms over another 5,000 hits | Same local component workload; not HTTP or end-to-end LLM time | Same receipt |
| 64 concurrent identical cold requests → 1 counted stub-provider invocation, 1 miss, 63 hits per store | Constructed process-local concurrency fixture; not paid LLM savings | Same receipt |
| Four request fixture: off/exact/guarded 4/3/2 calls | Offline mechanics including restart; explicit semantic approvals | `docs/DOCUMENTATION_REBUILD.md` |
| Four whole-agent tasks: off/exact/guarded 12/9/6 calls | Scripted provider and approval fixture, not automatic natural-language savings | `docs/BEFORE_AGENT_CACHE.md` |
| 320 targeted regressions passed in 6.09s | Software engineering checks, not 320 correct answers | `artifacts/resume/regression-tests.txt` |
| 174 fresh-environment release tests passed in 3.53s; one deprecation warning | Subset/overlapping tests, NOT additive to 320 | `artifacts/documentation_rebuild/group4/clean-install-v1/` |
| Clean installed-wheel HTTP flow: 3 then 0 scripted calls; fresh approvals and memory clear work | Offline functional integration; not browser visual QA | Same clean-install receipts |

Do not sum historical test counts. Do not claim live dollars saved, production latency, general semantic precision, end-to-end speedup or FAANG acceptance from this evidence.

## 8. Metrics to improve and how to measure them

Priority is **useful safe semantic coverage**, not maximizing similarity or lowering thresholds.

1. Semantic accepted-answer precision: sufficient actual reused answers / accepted semantic reuses. Report wrong and unknown cases separately, sample size, grouping and appropriate uncertainty. Zero accepts means precision is undefined.
2. Safe semantic coverage/recall: correct semantic reuses / labeled reusable opportunities. Separate retrieval misses from verifier abstentions. Count actual selected candidate relations.
3. Unsafe accepts by family: direction, negation, entities, quantities/units/versions, added requirements. These must not be hidden by exact repeats.
4. Actual remote calls and tokens avoided: matched off/exact/semantic workloads; include planning, drafting, review, failures, retries and verification. Report all traffic and eligible traffic separately.
5. End-to-end completed-answer p50/p95 and local verification overhead: comparable workloads, provider/settings/cache state; retain raw timings. Component lookup timing is a separate metric.
6. Reliability: restart, invalidation, stale-source rejection, scope isolation and duplicate-miss coordination; state workload and process count.

No new numerical quality gate is invented by this handoff. Existing frozen protocols remain unchanged. A new claim needs a prespecified fresh evaluation protocol; the exposed 30-question packet must not be tuned on or called fresh.

## 9. Exact next work and deadline boundary

The user's urgent resume deadline was described as tomorrow; confirm calendar timing only if scheduling matters. Do not promise an unmeasured positive result by then.

**Next technical block:** inspect actual pair scoring, input construction, score distributions and policy wiring. Use existing development records and known model contracts. Identify a reproducible implementation defect or establish that model/task mismatch is the blocker. Fix concrete defects only; do not expand training or lower thresholds on the spent diagnostic.

Then run one declared DEVELOPMENT comparison on actual retrieved answers with fixed settings. Report correct/incorrect/unknown reuses and misses for off/exact/semantic. If no improvement, report that failure and stop that experiment; do not quietly rename exact reuse as solved semantic caching.

Independently finish browser QA when a working browser is available, then deployment verification. A new live utility run needs explicit provider budget and journal authorization. Existing approvals to publish code are not authorization for live inference or paid compute.

For the immediate resume, use the measured engineering metrics already available. Additional credible live savings cannot be guaranteed. Do not spend the deadline collecting more interchangeable unit-test counts.

## 10. Resume wording supported now

**LLM Caching Gateway & Agentic Documentation Assistant — Python, FastAPI, SQLite, SentenceTransformers**

- Built a tenant-scoped response-caching gateway with TTL, invalidation and durable storage; measured 1.79 ms p95 SQLite cache-hit processing across 5,000 warm local requests.
- Implemented a bounded documentation agent with planning, read-only source tools, review and durable human approval; validated caching and workflow behavior with 320 targeted offline regression tests.

Interview caveat: semantic retrieval exists, but automatic answer applicability is not validated. Explain that failure directly. See `docs/Resume_Ready_Project.md`.

## 11. Commands and durable recovery

Normal local demo (no key, explicitly scripted):

```bash
python -m pip install -r deploy/requirements-gateway.txt
python scripts/run_demo.py --whole-answer-cache --task-memory
```

Measured component benchmark, NEW output only:

```bash
python scripts/benchmark_resume_cache.py --output NEW_BENCHMARK.json
```

Clean installed-wheel reproduction commands: `docs/RELEASE_48H.md`.
On this hosted runtime, some old commands need `PYTHONPATH=.venv/lib/python3.12/site-packages`; do not assume that path exists on a user's machine.

Latest complete implementation checkpoint before this report:
https://chatgpt.com/library?file_id=libfile_59fb96b3583081919f7c5aac4ef37ebe&account_id=personal
Filename: documentation-product-clean-install.zip
SHA-256: `f115a90c9dd46a9cf8a0a99e7479cd081947bc8d4e85c6a6fcd01655b3a3f1e6`
Contains `project.bundle`, MANIFEST, RESTORE and opaque Stage B archive. The latter preserves trained exports and excluded final records; do not open final records. Pretrained models, environments, credentials and application databases are excluded.

After extraction and manifest verification, restore in a NEW folder:

```bash
git bundle verify project.bundle
git clone -b integration/documentation-rebuild project.bundle restored-project
```

A new checkpoint accompanies this handoff. Use its bundle to retain the report commit too. Workspace resets have repeatedly exposed old Stage B state; do not assume that old checkout is the newest work. Restore the latest bundle instead of rebuilding completed groups.

Locked final manifest SHA-256: `1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af`. Do not evaluate or inspect final records. Preserve historical evidence and FACTS append-only discipline. Keep important source/data/reports committed and durably retrievable.

## 12. Publication outcome and commit inventory

Latest CLI push attempt during handoff preparation failed: `fatal: could not read Username for https://github.com: No such device or address`. No successful push is claimed. Prior connected-GitHub attempt uploaded a few unreferenced blobs but was stopped by automatic approval review. No new integration branch was present at the last verified remote check. Do not substitute replayed commits with new SHAs for an exact history push without explicitly explaining that difference.

When an authenticated Git environment is available, the authorized command is:

```bash
git push origin HEAD:refs/heads/integration/documentation-rebuild
```

Do not put credentials in commands or ask the user to paste API keys. Publishing remains unfinished, not silently delegated as completed.

Local milestone commits after existing audit base `9cb9083` (the report commit follows these):

```text
6cea9ec research(stage-a): preserve pinned intent baseline harness and hash checks
ec77788 research(stage-a): record authored splits and reproducible final lock
699223a research(stage-a): preserve baseline evidence and lossless decision log
9c07956 research(stage-b): B0 freeze diverse template-held-out cohorts and protocol
a259797 research(stage-b): B1 train and export three pinned LoRA classifier seeds
4426088 research(stage-b): B2 add question-only veto and quarantine unverified metadata
2058360 research(stage-b): B3 measure actual-candidate failures on held-out dev
ca1fe33 research(stage-b): B4 calibrate frozen policies and preserve resumable evidence
847d3f9 research(stage-b): B5 report generalization failure and comparative evidence
f65ea9b feat(docs-cache): rebuild source-bound durable semantic candidates
06a996c feat(documentation): reuse qualified answers before agent execution
d68173d feat(documentation): integrate role routing and scoped task memory
5057657 chore(evaluation): inventory product data and prepare independent review
14c1827 feat(review): separate model diagnostics from blinded human judgments
0724442 chore(review): preserve complete human labels and model comparison
097d046 feat(workbench): expose context controls and clarify answer reuse
ff129e1 chore(release): add model preflight and manual product browser CI
f2c56f7 fix(models): restore pinned execution and measure real document diagnostics
0696b37 perf(cache): measure persistent hit overhead and prepare honest resume entry
c808396 release: verify clean gateway wheel and reconcile product status
```

## 13. Instructions to the next chat

Read this file and the latest source before reporting progress. Retain the combined Project 7 + 15 scope. Do not start over, invent recovered roadmap groups, claim background work, repeat failed upload routes around approval blocks, enable unqualified semantic serving or fabricate performance. First verify HEAD and recover if necessary; then investigate the specific semantic scoring/policy failure. Give the user a concrete finding and outcome, not another vague roadmap.
