# Claim-hygiene audit

Scope: README.md, DESIGN.md and docs/RESUME_BULLETS.md as of parent commit
`4fbe298`; current status consolidation and standing instructions. This is an
offline documentation audit, not a new model/evaluation run. Original documents
are retained under docs/archive/ with superseded banners and in Git history.

The table lists distinct overclaim families and all affected document locations
by original section. Repeated one-line/two-line/interview versions are covered
by the same row. Correct source observations are retained in FACTS.md; no old
telemetry, labels, protocol or gate was modified.

| Original location / claim | Evidence limit | Correction |
| --- | --- | --- |
| README key result; RESUME bullet 1 variants: query latency reduced 99.20%–99.84% | LAT-01–LAT-06 compare AUTO_REUSE against remote judge, not fresh generation; hit counts n=20,1,1 | Remove application headline; optional historical bullet includes baseline, cohorts and counts in the bullet itself |
| Same bullet: resolves traffic locally without remote model calls | LRR counts absence of judge calls; bypass may still require answer generation | Remove conflation; DESIGN explicitly distinguishes routing, judge work and generation |
| RESUME bullet 1 interview: local vector search always sub-50 ms | LAT includes a larger scaled mean and singleton runs; no universal SLA | Remove universal timing promise |
| README/RESUME use tiny timing cohorts as stable performance | LAT-06 shows two singleton AUTO_REUSE cohorts | Put sample sizes directly beside comparison; no fabricated timing interval |
| README/design history presented as active serving architecture | R3 serving wrapper and historical FAISS/CrossEncoder pipeline differ | Document current gateway/agent, link separate research history |
| README classifier zero-error rates without denominators/CI in table | P1 cohorts have finite dynamic-query subsets | Remove abbreviated rate claims from current headline; raw historical observations remain ledgered |
| README benchmark table several error rates without CI | Ledger has sample roles; subset estimates do not establish population safety | Current README uses evidence links rather than truncated metric leaderboard |
| README Phase 8 local-resolution/call column can imply no generation | P8-R04/P8-P09 are simulation/projection; zero remote judging is not zero generation | Explicit source replay/projected-versus-live distinction |
| README/DESIGN: fixed thresholds cannot separate safe paraphrases universally | P2 sweep only tests a finite curated benchmark | Remove universal impossibility claim; retain scoped historical failure in ledger |
| README/DESIGN statistical ceiling described as mathematically guaranteed | Finite-sample confidence statements require assumptions; grouped hits need dependence treatment | No guarantee wording; current quality gate remains open |
| README/RESUME: overlap of marginal CIs proves lack of significance | HR-04 provides marginal intervals, not a paired hypothesis test | Do not infer a significance test from overlap; DESIGN names this limitation |
| RESUME bullet 2: cryptographic seal eliminates leakage | Seal proves recorded bytes; cannot prove split/lineage independence | Remove leakage guarantee; Phase A source audit remains authoritative correction |
| RESUME bullet 3: hazard reduction looks like validated general improvement | HR/P3 are same-set estimates with small denominators and different accepted sets | Remove from recommended headline; no general semantic benefit claimed |
| RESUME bullet 4: challenge accuracy/dynamic errors combined with 2.29 ms | P1-07 timing belongs to final-test cohort, not challenge cohort | Remove cross-cohort composite bullet |
| RESUME bullet 4: guaranteed zero dangerous errors / eliminates stale reuse | P1 observations on small cohorts are not guarantees | Remove causal safety guarantee |
| RESUME bullet 5: collision relabeling produces verified safety | P5B corrections are historical design-informed labels; current actual-answer independent review pending | Do not use retrospective label corrections as present safety evidence |
| RESUME bullet 6: token check validates every claim / scientific auditability | Numeric token matching proves vocabulary inclusion, not support/meaning/denominators | Remove semantic-verification implication; retain auditability tooling as engineering |
| RESUME bullet 6: clean scans establish supply-chain security | Scan findings are tool/time/pattern scoped; SHA rows are commit identifiers | Remove security certification and incorrect evidence attribution |
| RESUME limitations: downstream generation never measured / no pricing computations anywhere | P8 baseline and Phase A baseline records exist; historical pricing estimates were recorded/retracted | Scope absence to representative paired application evidence; avoid claiming no historical baseline/pricing work existed |
| DESIGN: detects model hallucinations / never fails open as correctness guarantee | C-16 covers handling timeouts, parsing and other detected errors; plausible wrong verdict can pass | Separate error handling from model correctness |
| DESIGN: judge is primary adaptive mechanism operating in production | No demonstrated production deployment or learning outcome | Describe query-dependent judgment without deployment/learning claim |
| DESIGN: fallback threshold is safe | C-12 is a configured fallback; P2 does not certify safety | Remove safe-threshold wording |
| DESIGN: adaptive fitting never fired (unqualified across whole project) | P4-01/KL-11 describe early cohorts; PA-C02 later reports domain fits | Scope early history; later fits also do not prove deployed utility |
| README limits: all storage is in-memory | Historical KL-07 describes FAISS; newer exact runtime supports SQLite | Distinguish persistent exact storage from cold semantic index |
| README historical fixed free-tier quota as current guarantee | Account/provider limits can change; project durable caps are local configuration | Link current quota-controlled pilot; no account availability guarantee |
| Multiple authoritative status reports conflict with R4/R5 | CURRENT_STATUS and R3–R5 closeouts supersede older summaries | One current page; retain old bodies under visible superseded banners, including HTML |

## What did not change

No runtime code, model, threshold, data split, archived evaluation result or R5
raw output changed. No live calls or new replay evaluation were run. The R5
agent result remains the negative baseline. Check/recompute details are recorded
in CLAIM_HYGIENE_VALIDATION.json. Model-assisted review remains separate from
human labels; user approval of this proposal is not answer-quality ground truth.
