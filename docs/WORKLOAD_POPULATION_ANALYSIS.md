# Workload population and cache eligibility

**Decision.** Monitor the request mix and trusted-context coverage alongside cache savings. Exact caching remains the default and the context control has controlled evaluation evidence.

Exact caching avoided **47.33%** of baseline scripted calls in the repeat-heavy workload and **14.75%** in the broader workload. The **32.58 percentage-point gap** decomposes into **+25.58 points from the request mix** and **+7.00 points from different within-segment savings**.

## Profile the observed workloads

Executions are matched on workload and request ID. The historical population has **1,000 request records** and **4,000 executions** across four policies. The unit in the table is a request. The savings denominator is baseline scripted calls. Exact duplicates can occur inside other request categories.

| Request type | Repeat-heavy requests | Broader requests | Repeat-heavy exact calls avoided | Broader exact calls avoided |
| --- | ---: | ---: | ---: | ---: |
| adversarial challenge | 120 | 125 | 0 | 0 |
| exact repeat | 240 | 60 | 708 | 177 |
| natural paraphrase | 180 | 90 | 144 | 0 |
| seed | 60 | 125 | 0 | 0 |

For each request type, the mix term is the change in share of baseline calls multiplied by the average segment savings rate. The within-segment term uses the change in segment rate multiplied by the average share. This symmetric decomposition reconciles the arithmetic gap. Causal attribution would require stronger assumptions about which requests enter each workload.

## Compare suitability, risk, and fallbacks

The separate synthetic risk holdout has **192 requests** across **24 intent groups**. It contains 24 literal repeats, 48 eligible resolved paraphrases, 24 eligible unresolved paraphrases, and 96 unsuitable challenges.

| Policy | Suitable scripted calls avoided | Unsuitable reuses | Requests routed to fallback |
| --- | ---: | ---: | ---: |
| Exact only | 24 | 0 | 168 |
| Current guarded | 96 | 94 | 2 |
| Reporting check | 96 | 70 | 26 |
| Trusted context | 72 | 0 | 120 |

## Workload sensitivity scenarios

With exact repeats held at **15%**, increasing trusted-context coverage among eligible paraphrases from **25% to 100%** moves the context-control projection from **241 to 514 suitable scripted calls avoided per 1,000 requests**.

| Exact repeats | Trusted context among eligible non-exact requests | Suitable calls avoided per 1,000 | Fallbacks per 1,000 |
| ---: | ---: | ---: | ---: |
| 5% | 25% | 151.8 | 848.2 |
| 5% | 50% | 253.6 | 746.4 |
| 5% | 75% | 355.4 | 644.6 |
| 5% | 100% | 457.1 | 542.9 |
| 15% | 25% | 241.1 | 758.9 |
| 15% | 50% | 332.1 | 667.9 |
| 15% | 75% | 423.2 | 576.8 |
| 15% | 100% | 514.3 | 485.7 |
| 30% | 25% | 375.0 | 625.0 |
| 30% | 50% | 450.0 | 550.0 |
| 30% | 75% | 525.0 | 475.0 |
| 30% | 100% | 600.0 | 400.0 |

Scenario math uses the observed outcomes within four strata. The eligible share among non-exact requests stays fixed at 72/168. Changing trusted-context coverage reallocates eligible paraphrases between resolved and unresolved strata. Each request can avoid one scripted call. These are conditional projections from synthetic inputs. Segment outcomes and challenge composition could change in a real population. The V3 scorecard gives observed denominators and interval bounds for the holdout.

## Interview walkthrough

> “I defined the grain as one request per policy, reconciled it back to the benchmark, and split the savings gap into changes in traffic mix and changes within each segment. Then I kept the repeat share fixed and varied trusted-context coverage. That showed why data completeness is a prerequisite for useful semantic reuse. I would monitor suitable calls avoided, unsuitable reuses and fresh fallbacks together.”

**Architecture in plain terms.** One agent workflow plans, drafts and reviews. Exact reuse can skip those steps for a matching approved response. Semantic reuse tests whether a paraphrase can reuse an answer within trusted reporting context. Routing adapts by request and falls back when suitability cannot be established.

## Reproduce

```bash
python scripts/report_workload_population.py
python scripts/report_workload_population.py --check
```

Use Python 3.11+ with SciPy installed for the existing holdout reconciliation. The report command operates on committed receipts and requires no model weights. For the interactive view, start the standard interview workbench with `python scripts/run_demo.py --interview --fresh` and open [Workload population](http://127.0.0.1:8765/benchmark/population.html).

[SQL views](../scripts/workload_population.sql) · [Observed population CSV](../artifacts/workload_analysis/observed_population.csv) · [Savings-gap CSV](../artifacts/workload_analysis/savings_gap.csv) · [Risk population CSV](../artifacts/workload_analysis/risk_population.csv) · [Sensitivity CSV](../artifacts/workload_analysis/sensitivity.csv) · [Risk scorecard](RISK_V3_RESULTS.md)
