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

Inputs are fictional operational reporting fixtures. Labels apply to the particular cached answer identified by its hash in each receipt. Calibration and holdout use different base intents and share request templates. The results test those wording patterns. The added rule checks labels, so it cannot establish the factual accuracy or provenance of their contents. Intent groups contain related variants, so event-level exact intervals are descriptive under the binomial assumption. The holdout is now spent for development, and repeats reproduce the frozen run.

## Paired local timing pilot

On **8 warm literal repeats** using Qwen2.5-0.5B-Instruct on CPU, fresh generation measured **3307.35 ms p50** and exact reuse **0.141 ms p50**. The fresh arm generated **200 completion tokens**, and the reuse arm generated zero. The pilot verifies identical returned answers for each pair. Priming is measured separately. This is a local warm-cache component measurement with machine-specific timing.

## Reproduce

Run from the repository root with Python 3.11 or newer. Install dependencies in a virtual environment and download the pinned checkpoints once:

```bash
python -m pip install -e ".[dev]"
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --include "*.json" --include "*.txt" --include "*.safetensors"
hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --include "*.json" --include "*.txt" --include "*.safetensors"
python scripts/evaluate_risk_v2.py calibration --output calibration-replay.json
python scripts/evaluate_risk_v2.py holdout --output holdout-replay.json
python scripts/report_risk_v2.py --check
```

[Frozen protocol](RISK_V2_PROTOCOL.md) · [Input hashes](../artifacts/risk_v2/seal.json) · [Holdout receipts](../artifacts/risk_v2/holdout.json) · [SQLite validation](../scripts/report_risk_v2.py)

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --include "*.json" --include "*.txt" --include "model.safetensors"
python scripts/measure_risk_v2_exact.py --output local-pilot-replay.json
```

[Timing protocol](RISK_V2_LATENCY_PROTOCOL.md) · [Local timing receipts](../artifacts/risk_v2/local_exact_pilot.json)
