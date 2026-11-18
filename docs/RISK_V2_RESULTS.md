# Cache reuse risk oversight

**Decision.** Keep exact reuse as the default. The reporting check met the fixture criteria for follow-up testing. Remaining failures need stronger controls for changes to period, scope and entity.

Across **120 matched requests in 24 intent groups**, the reporting check prevented **24 unsuitable reuses** and lost **0 suitable reuses** versus the same guarded pipeline. It introduced **0 unsuitable reuses**. Both guarded configurations avoided **72 scripted calls on eligible requests**.

## Holdout scorecard

| Configuration | Unsuitable answer / semantic accepts | Eligible non-exact capture | Challenge accepts | Gross / eligible scripted calls avoided |
| --- | --- | --- | --- | --- |
| Exact only | — (0 accepts) | 0/48 (0.0%, 95% CI 0.0%–7.4%) | 0/48 (0.0%, 95% CI 0.0%–7.4%) | 24 / 24 |
| Current guarded | 38/86 (44.2%, 95% CI 33.5%–55.3%) | 48/48 (100.0%, 95% CI 92.6%–100.0%) | 38/48 (79.2%, 95% CI 65.0%–89.5%) | 110 / 72 |
| Guarded + reporting check | 14/62 (22.6%, 95% CI 12.9%–35.0%) | 48/48 (100.0%, 95% CI 92.6%–100.0%) | 14/48 (29.2%, 95% CI 17.0%–44.1%) | 86 / 72 |

Gross savings include work avoided by returning unsuitable answers. Eligible savings count only scripted calls avoided on suitable fixture answers. The no-cache denominator is one scripted call per probe. Priming is excluded from these savings.

## Exception drilldown

| Challenge | Requests | Guarded accepts | Accepts with reporting check |
| --- | ---: | ---: | ---: |
| Missing artifact | 24 | 24 | 0 |
| Negation | 5 | 0 | 0 |
| Numeric requirement | 5 | 0 | 0 |
| Period change | 5 | 5 | 5 |
| Scope change | 5 | 5 | 5 |
| Entity change | 4 | 4 | 4 |

Inputs are fictional operational reporting fixtures. Labels apply to the particular cached answer identified by its hash in each receipt. Intent groups contain related variants. Exact intervals are event-level and descriptive under the binomial assumption. The holdout is now spent for development, and repeats reproduce the frozen run. The experiment leaves serving policy decisions outside the harness.

## Reproduce

With the exact MiniLM encoder and passage scorer checkpoints already in the local Hugging Face cache:

```bash
python scripts/evaluate_risk_v2.py calibration --output calibration-replay.json
python scripts/evaluate_risk_v2.py holdout --output holdout-replay.json
python scripts/report_risk_v2.py --check
```

[Frozen protocol](RISK_V2_PROTOCOL.md) · [Input hashes](../artifacts/risk_v2/seal.json) · [Holdout receipts](../artifacts/risk_v2/holdout.json)
