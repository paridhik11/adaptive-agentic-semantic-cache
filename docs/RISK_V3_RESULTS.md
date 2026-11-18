# Reporting context risk and efficiency

**Decision.** Advance the context control to follow-up testing for applications that resolve trusted report coordinates. Keep exact-only as the release default.

On **192 matched probes across 24 held-out intent groups**, adding report coordinates reduced unsuitable semantic accepts from **70 to 0**. For changed periods, entities and populations, the context arm accepted **0/72** requests (95% exact event-level upper bound **5.0%**). It lost **0** suitable reuses when coordinates were resolved. It gave up **24** suitable reuses overall.

## Four configurations on the same requests

| Configuration | Unsuitable / semantic accepts | Changed-context accepts | Resolved eligible non-exact capture | Unresolved eligible capture | Gross / eligible scripted calls avoided |
| --- | --- | --- | --- | --- | --- |
| Exact only | — | 0/72 (0.0%–5.0% CI) | 0/48 (0.0%–7.4% CI) | 0/24 (0.0%–14.2% CI) | 24 / 24 |
| Current guarded | 94/166 (48.7%–64.3% CI) | 70/72 (90.3%–99.7% CI) | 48/48 (92.6%–100.0% CI) | 24/24 (85.8%–100.0% CI) | 190 / 96 |
| Guarded + V2 reporting check | 70/142 (40.8%–57.8% CI) | 70/72 (90.3%–99.7% CI) | 48/48 (92.6%–100.0% CI) | 24/24 (85.8%–100.0% CI) | 166 / 96 |
| V2 + verified context | 0/48 (0.0%–7.4% CI) | 0/72 (0.0%–5.0% CI) | 48/48 (92.6%–100.0% CI) | 0/24 (0.0%–14.2% CI) | 72 / 72 |

The context policy avoided **72 suitable scripted calls** versus **24 for exact-only**. Gross savings credit every hit. Eligible savings credit only reuse of suitable fixture answers. The denominator for the scripted benchmark is one call per probe without caching, excluding priming.

## Exceptions and fallbacks

| Request scenario | Requests | V2 guarded accepts | Context-control accepts |
| --- | ---: | ---: | ---: |
| Entity | 24 | 24 | 0 |
| Exact repeat | 24 | 24 | 24 |
| Missing artifact | 24 | 0 | 0 |
| Paraphrase a | 24 | 24 | 24 |
| Paraphrase b | 24 | 24 | 24 |
| Period | 24 | 24 | 0 |
| Scope | 24 | 22 | 0 |
| Unresolved | 24 | 24 | 0 |

At the intent-group level, **0/24** had any changed-context exception (95% exact upper bound **14.2%**). All intervals use exact Clopper–Pearson 95% bounds. Event-level intervals are descriptive because related probes share intents.

The data are synthetic. Coordinates are supplied by a trusted caller in this harness. The evaluation tests enforcing those coordinates at lookup. It does not measure extraction or authorization from free-form user text. The held-out intent groups use shared wording patterns. No billable inference or end-to-end production latency was measured here.

## Reproduce and audit

Use the pinned local MiniLM checkpoints from the [V2 setup instructions](RISK_V2_RESULTS.md#reproduce).

```bash
python scripts/evaluate_risk_v3.py calibration --output calibration-replay-v3.json
python scripts/evaluate_risk_v3.py holdout --output holdout-replay-v3.json
python scripts/report_risk_v3.py --check
```

The first holdout is spent. A replay reproduces the frozen comparison. The report checker independently pairs request receipts in SQLite, checks each served-answer digest against the fixture, and reconciles aggregate metrics.

[Protocol](RISK_V3_PROTOCOL.md) · [Frozen source hashes](../artifacts/risk_v3/seal.json) · [Holdout receipts](../artifacts/risk_v3/holdout.json) · [Original workload benchmarks](INTERVIEW_V1.md#numbers-to-defend)
