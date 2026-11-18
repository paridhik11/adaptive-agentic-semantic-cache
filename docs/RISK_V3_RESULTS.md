# Reporting context risk and efficiency

**Decision.** Advance the context control to follow-up testing for applications that resolve trusted report coordinates. Keep exact-only as the release default.

On **192 matched probes across 24 held-out intent groups**, adding report coordinates reduced unsuitable semantic accepts from **70 to 0**. For changed periods, entities and populations, the context arm accepted **0/72** requests (95% exact event-level upper bound **5.0%**). It lost **0** suitable reuses when coordinates were resolved. It gave up **24** suitable reuses overall.

## Four configurations on the same requests

| Configuration | Unsuitable / semantic accepts | Changed-context accepts | Resolved eligible non-exact capture | Unresolved eligible capture | Gross / eligible scripted calls avoided |
| --- | --- | --- | --- | --- | --- |
| Exact only | — | 0/72 (0.0%–5.0% CI) | 0/48 (0.0%–7.4% CI) | 0/24 (0.0%–14.2% CI) | 24 / 24 |
| Current guarded | 94/166 (48.7%–64.3% CI) | 70/72 (90.3%–99.7% CI) | 48/48 (92.6%–100.0% CI) | 24/24 (85.8%–100.0% CI) | 190 / 96 |
| Guarded + V2 reporting check | 70/142 (40.8%–57.8% CI) | 70/72 (90.3%–99.7% CI) | 48/48 (92.6%–100.0% CI) | 24/24 (85.8%–100.0% CI) | 166 / 96 |
| V2 + trusted context | 0/48 (0.0%–7.4% CI) | 0/72 (0.0%–5.0% CI) | 48/48 (92.6%–100.0% CI) | 0/24 (0.0%–14.2% CI) | 72 / 72 |

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

The data are synthetic. Coordinates are supplied by a trusted caller in this harness. The evaluation tests enforcing those coordinates at lookup. Extraction and authorization from free-form user text are outside this evaluation. The held-out intent groups use shared wording patterns. No billable inference or end-to-end production latency was measured here.

## Measured CPU pilot on mixed requests

On **24 paired requests** spanning exact repeats, paraphrases, changed months and unresolved coordinates, local generation calls fell from **24 to 12**. Generated completion tokens fell from **757 to 384**.

Summed wall time was **110.99 s fresh** versus **57.05 s gated** (**57.30 s** including cache preparation). Median request latency was **4.62 s fresh** and **2.21 s gated**.

| Timing scenario | Pairs | Fresh / gated generation calls | Fresh / gated completion tokens |
| --- | ---: | ---: | ---: |
| Exact repeat | 6 | 6 / 0 | 192 / 0 |
| Paraphrase a | 6 | 6 / 0 | 181 / 0 |
| Period | 6 | 6 / 6 | 192 / 192 |
| Unresolved | 6 | 6 / 6 | 192 / 192 |

This separate timing replay uses a single local CPU model with fixed prewritten seed answers. Preparation excludes generating those seeds. The results describe this request mix and hardware.

[Timing protocol](RISK_V3_MIXED_TIMING.md) · [Per-request timing and token receipts](../artifacts/risk_v3/mixed_timing.json)

## Reproduce and audit

Use the pinned local MiniLM checkpoints from the [V2 setup instructions](RISK_V2_RESULTS.md#reproduce).

```bash
python scripts/evaluate_risk_v3.py calibration --output calibration-replay-v3.json
python scripts/evaluate_risk_v3.py holdout --output holdout-replay-v3.json
python scripts/report_risk_v3.py --check
```

The first holdout is spent. A replay reproduces the frozen comparison. The report checker independently pairs request receipts in SQLite, checks each served-answer digest against the fixture, and reconciles aggregate metrics.

[Protocol](RISK_V3_PROTOCOL.md) · [Frozen source hashes](../artifacts/risk_v3/seal.json) · [Holdout receipts](../artifacts/risk_v3/holdout.json) · [Original workload benchmarks](INTERVIEW_V1.md#numbers-to-defend)

With the pinned Qwen checkpoint from the [V2 timing setup](RISK_V2_RESULTS.md#reproduce) available locally:

```bash
python scripts/measure_risk_v3_mixed.py --output mixed-timing-replay.json
```
