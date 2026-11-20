# Semantic reuse upgrade

The upgrade adds ordered container conversions, procedural evidence checks and bounded candidate verification. Keep exact-only as the release default for general neural reuse. The live chat retains its scoped deterministic rules and optional short-definition neural path.

## Separate execution paths

| Path | Actual route |
| --- | --- |
| Live general chat | Eligibility → scoped exact store → keyword signatures and proven definition variants → optional MiniLM retrieval for supported short definitions → local verification → reuse or provider. SQLite binds an index record to the stored entry and its expiry. |
| Documentation workbench | Before-agent lookup of an approved answer bound to the workflow, source revision, tenant, principal and task context. On a hit, the agent revalidates the stored source evidence and follows the task’s approval policy. A miss enters planning, tools, drafting and review. |
| Experimental hybrid and answer-aware policies | `ConservativeHybridCache` handles scoped candidates and its own structural checks. `GuardedSemanticCache` delegates to its configured approver. `AnswerAwareSemanticPolicy` uses passage relevance and defaults to diagnostic operation. The typed workbench gate does not replace these policies. |
| Historical interview campaign | Recorded offline no-cache, exact, guarded and shadow executions. The report joins matching requests and reconciles call and token counts by segment. |

The earlier P0 design describes a different runtime from live chat. Its single-candidate selection and approval wiring should not be read as the live chat design. The stored policy artifacts remain interpretable under their original versions. [P0 design](SEMANTIC_P0_DESIGN.md) · [Historical interview report](INTERVIEW_V1.md)

## What caused misses and bad hits

- The directional conversion request “make a list from a tuple” had no scoped signature. The new grammar records source and destination in the same order as “convert a tuple to a list.” It preserves extra text outside this narrow grammar. A semantic hit now requires text that demonstrates the corresponding constructor.
- A related answer could mention cProfile and still omit its use. The semantic serving paths now check whether the source request included the tool requirement and look for a supported Python invocation or CLI example in the actual answer. The Python check inspects syntax without running it. Other valid profiling idioms can still fall back.
- The optional live verifier previously stopped after rejection by the first candidate or a close-score veto. It now verifies up to three eligible candidates at the original thresholds. A lower-ranked candidate can win after a higher one fails. Close qualified alternatives and unexamined alternatives within the margin still force fresh execution. This was verified with deterministic ranking fixtures.
- The earlier public benchmark found 168 annotation-matched candidates rejected at the passage stage after passing the earlier checks. The rerun reproduced the prior ComQA aggregates, including 0 supported hits in either live API mode and 9/422 in the experimental full-question path. The passage model measures relevance and its score has no calibrated correctness interpretation. The unchanged off-the-shelf model and broad full-question route remain a coverage limitation. [ComQA protocol and counts](LIVE_CHAT_COSINE_EVALUATION.md)

The live API records whether checks ran, rejection counts from scoped retrieval and structural filters, the rank and cosine of each considered neural candidate, bidirectional pair scores, passage scores, and elapsed verification work. Tenant filtering happens before candidates are exposed. The trace uses candidate IDs and omits candidate prompt and answer text. Candidate answer sufficiency remains unknown in the live trace. The fixture replay assigns the authored label only to the paired probe. The document receipt includes reasons for profiling fallbacks.

## Historical campaign reconciliation

The report checker recomputed aggregates from the historical receipts. The original replay had 1,000 request records across 4,000 executions.

| Workload | No cache calls | Exact calls | Guarded calls | Shadow calls |
| --- | ---: | ---: | ---: | ---: |
| Repeat-heavy, 600 requests per policy | 1,800 | 948 | 927 | 948 |
| Broader, 400 requests per policy | 1,200 | 1,023 | 975 | 1,023 |

Exact reuse saved 47.33% and 14.75% of scripted calls and 46.57% and 14.56% of synthetic tokens, respectively. Guarded saved 21 and 48 further scripted calls. The broader arm's 48 includes three calls avoided by the documented cProfile wrong reuse. These are historical scripted and synthetic results. [Source CSV](../dashboard/interview/comparison.csv)

## Before and after on authored reserved groups

There are 12 probes, five suitable and seven unsuitable, plus 12 exact repeats per arm. Counts exclude cache priming. Labels apply to the probes. Some deliberately insufficient seed answers are present to test semantic rejection. The exact-repeat count is an execution metric. Before uses checkout `98bd87e`; after uses the upgrade. All calls use a fixture.

| Arm | Exact hits | Supported semantic hits | Unsuitable semantic hits | Probe fallbacks | Scripted calls | Calls avoided vs no cache | Synthetic tokens avoided | p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no cache before | 0 | 0 | 0 | 12 | 24 | 0 | 0 | 7.98 / 18.01 |
| exact before | 10 | 0 | 0 | 12 | 14 | 10 | 94 | 10.04 / 48.00 |
| guarded before | 10 | 4 | 2 | 6 | 8 | 16 | 148 | 4.71 / 30.06 |
| shadow before | 10 | 0 | 0 | 12 | 14 | 10 | 94 | 7.87 / 45.10 |
| no cache after | 0 | 0 | 0 | 12 | 24 | 0 | 0 | 4.60 / 5.92 |
| exact after | 10 | 0 | 0 | 12 | 14 | 10 | 94 | 4.78 / 8.12 |
| guarded after | 10 | 5 | 0 | 7 | 9 | 15 | 145 | 3.67 / 13.29 |
| shadow after | 10 | 0 | 0 | 12 | 14 | 10 | 94 | 4.47 / 13.14 |

The upgrade captured **5/5** suitable reserved probes versus **4/5** before. It blocked both previously accepted unsuitable answers. The guarded arm's extra fresh call is the net result of blocking two unsuitable hits and gaining one supported hit. Shadow made the same recommendations without serving them. In development, supported reuse increased from 2 to 3 and unsuitable reuse fell from 2 to 0. Rejection totals count candidate attempts and may include multiple checks of a single stored answer. The key failure reasons in both splits were missing conversion evidence and name-only cProfile answers.

The exact 95% interval for the reserved 5/5 supported captures is 47.82%–100%. For 0/7 unsuitable reuses it is 0%–40.96%. These are descriptive under independence. Related requests share groups, and the fixture labels are authored. [Before receipts](../artifacts/semantic_upgrade/reserved_before.json) · [After receipts](../artifacts/semantic_upgrade/reserved_after.json) · [Protocol](SEMANTIC_UPGRADE_PROTOCOL.md)

The small replay stores peaked at 28 KiB per SQLite cache file in both versions. The handler times above include lookup, local work and the fixture call when needed. They are single-run measurements on a shared host. One separately executed local neural definition probe recorded 11.03 ms for embedding, 112.84 ms for cross-encoders and 129.75 ms total verification. This is a single measured stage receipt, with provider generation simulated by a fixture. Representative throughput, cache RAM scaling, production latency, real provider tokens and cost are **NOT MEASURED**. [Local stage receipt](../artifacts/semantic_upgrade/local_model_trace.json)

## Reproduce

Use a virtual environment with the gateway, test and SciPy dependencies. From the current repository, point the runner at an immutable baseline checkout.

```bash
git worktree add --detach ../semantic-upgrade-baseline 98bd87e
python scripts/evaluate_semantic_upgrade.py --repo ../semantic-upgrade-baseline --revision 98bd87e --split reserved
python scripts/evaluate_semantic_upgrade.py --revision upgrade-v1 --split reserved
python scripts/report_interview.py --check
python scripts/report_workload_population.py --check
python -m pytest -q tests/test_live_chat.py tests/test_before_agent_cache.py tests/test_durable_semantic_rebuild.py
```

Repeat with `--split development` for the exposed development groups. The [ComQA runbook](LIVE_CHAT_COSINE_EVALUATION.md#reproduce) installs and pins the local model stack and the public test file.

## Interview explanation

“I compared the same requests across cache policies and checked answer suitability separately from call counts. A phrase such as ‘make a list from a tuple’ can reuse a scoped conversion answer, while reversing direction or omitting the required profiler example forces a fresh answer. I expanded verification to a bounded shortlist while preserving uncertainty fallback. The fixture replay increased supported hits and blocked two unsuitable reuses. Broad semantic coverage still needs independent answer-level evaluation, so my release recommendation is exact reuse with guarded semantic experiments.”

Verification included a scan of added staged lines for provider keys, GitHub tokens and private-key headers.
