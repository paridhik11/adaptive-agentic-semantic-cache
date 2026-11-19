"""Build a reproducible SQL population analysis from committed cache receipts."""
import argparse
from collections import Counter
from contextlib import closing
import csv
import io
import json
import math
from pathlib import Path
import sqlite3

from report_risk_v3 import DATA, reconcile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/workload_analysis"
REPORT = ROOT / "docs/WORKLOAD_POPULATION_ANALYSIS.md"
INTERVIEW = ROOT / "dashboard/interview"
POLICIES = ("exact", "guarded", "reporting", "context")
STRATA = ("exact_repeat", "resolved_paraphrase", "unresolved_paraphrase", "challenge")
HISTORICAL_FIELDS = (
    "workload", "event_id", "request_type", "no_cache_calls", "exact_calls", "guarded_calls",
    "shadow_calls", "guarded_false_accept",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_history(db):
    for row in read_csv(INTERVIEW / "paired-events.csv"):
        strings = tuple(row[key] for key in HISTORICAL_FIELDS[:3])
        counts = tuple(int(row[key]) for key in HISTORICAL_FIELDS[3:])
        if any(val < 0 for val in counts) or counts[-1] not in (0, 1):
            raise ValueError("historical calls or challenge labels are invalid")
        if counts[-1] and strings[-1] != "adversarial_challenge":
            raise ValueError("challenge label outside its population")
        db.execute("INSERT INTO historical VALUES (?,?,?,?,?,?,?,?)", (*strings, *counts))


def load_risk(db):
    holdout = json.loads((DATA / "holdout.json").read_text(encoding="utf-8"))
    reconcile(holdout)
    population = Counter()
    for row in holdout["receipts"]:
        if not row["eligible"]:
            stratum = "challenge"
        elif row["type"] == "exact_repeat":
            stratum = "exact_repeat"
        else:
            stratum = "resolved_paraphrase" if row["resolved"] else "unresolved_paraphrase"
        db.execute("INSERT INTO risk VALUES (?,?,?,?,?,?,?)",
                   (row["group"], row["type"], row["policy"], stratum, row["eligible"], row["served"], row["calls"]))
        if row["policy"] == "context":
            population[stratum] += 1
    total = sum(population.values())
    nonexact = population["resolved_paraphrase"] + population["unresolved_paraphrase"]
    observed_repeat = population["exact_repeat"] / total
    observed_resolved = population["resolved_paraphrase"] / nonexact
    db.execute("INSERT INTO scenario_inputs VALUES (?,?,?)", ("observed_mix", observed_repeat, observed_resolved))
    for repeats in (5, 15, 30):
        for resolved in (25, 50, 75, 100):
            db.execute("INSERT INTO scenario_inputs VALUES (?,?,?)",
                       (f"repeat_{repeats:02}_context_{resolved:03}", repeats / 100, resolved / 100))
    return holdout, population


def query(db, sql):
    return [dict(row) for row in db.execute(sql)]


def compute():
    with closing(sqlite3.connect(":memory:")) as db:
        db.row_factory = sqlite3.Row
        db.executescript(Path(__file__).with_name("workload_population.sql").read_text(encoding="utf-8"))
        load_history(db)
        holdout, population = load_risk(db)
        history = query(db, "SELECT * FROM historical_segments ORDER BY workload, request_type")
        gap = query(db, "SELECT * FROM exact_gap ORDER BY request_type")
        risk = query(db, "SELECT * FROM risk_segments ORDER BY policy, stratum")
        scenarios = query(db, "SELECT * FROM sensitivity ORDER BY scenario, policy")
        reference = {
            (row["workload"], row["configuration"]): (int(row["events"]), int(row["scripted_calls"]))
            for row in read_csv(INTERVIEW / "comparison.csv")
        }
        actual = {
            (workload, arm): (n, calls) for workload, arm, n, calls in db.execute("""
              WITH agg AS (
                  SELECT workload, SUM(requests) n, SUM(baseline_calls) b,
                         SUM(exact_calls) e, SUM(guarded_calls) g, SUM(shadow_calls) s
                  FROM historical_segments GROUP BY workload
              )
              SELECT workload, 'no_cache', n, b FROM agg
              UNION ALL SELECT workload, 'exact_only', n, e FROM agg
              UNION ALL SELECT workload, 'verified_semantic', n, g FROM agg
              UNION ALL SELECT workload, 'shadow_semantic', n, s FROM agg
            """)
        }
    if actual != reference:
        raise ValueError("request counts or calls disagree with the historical comparison")
    if {arm: {r["stratum"] for r in risk if r["policy"] == arm} for arm in POLICIES} != {
            arm: set(STRATA) for arm in POLICIES}:
        raise ValueError("scenario weights require one populated row for every policy and stratum")
    repeat, broad = (
        100 * (1 - actual[(workload, "exact_only")][1] / actual[(workload, "no_cache")][1])
        for workload in ("repeat_rich", "lower_repeat")
    )
    mix = sum(row["mix_gap_pp"] for row in gap)
    within = sum(row["within_segment_gap_pp"] for row in gap)
    if not math.isclose(repeat-broad, mix+within, abs_tol=1e-9):
        raise ValueError("gap decomposition does not sum to the observed difference")
    for row in scenarios:
        if not all(math.isclose(left, right, abs_tol=1e-9) for left, right in (
            (row["weight_sum"], 1),
            (row["suitable_avoided_per_1000"]+row["unsuitable_per_1000"], row["gross_avoided_per_1000"]),
            (row["gross_avoided_per_1000"]+row["fallback_per_1000"], 1000),
        )):
            raise ValueError("scenario counts do not reconcile")
        if row["scenario"] == "observed_mix":
            metrics = holdout["arms"][row["policy"]]
            requests = metrics["requests"]
            gross = metrics["scripted_calls_avoided"]
            observed = {
                "gross_avoided_per_1000": gross,
                "suitable_avoided_per_1000": metrics["calls_avoided_on_eligible_rows"],
                "unsuitable_per_1000": metrics["challenge_accepts"]["count"],
                "fallback_per_1000": requests - gross,
            }
            if any(not math.isclose(row[key], 1000 * count / requests) for key, count in observed.items()):
                raise ValueError("observed weights failed to reproduce the holdout")
    return history, gap, risk, scenarios, population, (repeat, broad, mix, within)


def csv_content(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: f"{val:.6f}" if isinstance(val, float) else val for key, val in row.items()}
                     for row in rows)
    return out.getvalue()


def render(history, gap, risk, scenarios, population, metrics):
    repeat, broad, mix, within = metrics
    context = {row["scenario"]: row for row in scenarios if row["policy"] == "context"}
    low, high = (context[key]["suitable_avoided_per_1000"]
                 for key in ("repeat_15_context_025", "repeat_15_context_100"))
    lines = [
        "# Workload population and cache eligibility",
        "",
        "**Decision.** Monitor the request mix and trusted-context coverage alongside cache savings. "
        "Exact caching remains the default and the context control has controlled evaluation evidence.",
        "",
        f"Exact caching avoided **{repeat:.2f}%** of baseline scripted calls in the repeat-heavy workload "
        f"and **{broad:.2f}%** in the broader workload. The **{repeat-broad:.2f} percentage-point gap** decomposes "
        f"into **{mix:+.2f} points from the request mix** and **{within:+.2f} points from different within-segment savings**.",
        "",
        "## Profile the observed workloads",
        "",
        "Executions are matched on workload and request ID. The historical population has **1,000 request records** "
        "and **4,000 executions** across four policies. The unit in the table is a request. The savings denominator is "
        "baseline scripted calls. Exact duplicates can occur inside other request categories.",
        "",
        "| Request type | Repeat-heavy requests | Broader requests | Repeat-heavy exact calls avoided | Broader exact calls avoided |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    segments = {(r["workload"], r["request_type"]): r for r in history}
    for row in gap:
        kind = row["request_type"]
        left, right = (segments[(w, kind)] for w in ("repeat_rich", "lower_repeat"))
        lines.append(f"| {kind.replace('_',' ')} | {left['requests']} | {right['requests']} | "
                     f"{left['exact_calls_avoided']} | {right['exact_calls_avoided']} |")
    lines += [
        "",
        "For each request type, the mix term is the change in share of baseline calls multiplied by the average "
        "segment savings rate. The within-segment term uses the change in segment rate multiplied by the average "
        "share. This symmetric decomposition reconciles the arithmetic gap. Causal attribution would require "
        "stronger assumptions about which requests enter each workload.",
        "",
        "## Compare suitability, risk, and fallbacks",
        "",
        f"The separate synthetic risk holdout has **{sum(population.values())} requests** across **24 intent groups**. "
        f"It contains {population['exact_repeat']} literal repeats, {population['resolved_paraphrase']} eligible resolved paraphrases, "
        f"{population['unresolved_paraphrase']} eligible unresolved paraphrases, and {population['challenge']} unsuitable challenges.",
        "",
        "| Policy | Suitable scripted calls avoided | Unsuitable reuses | Requests routed to fallback |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {"exact": "Exact only", "guarded": "Current guarded", "reporting": "Reporting check", "context": "Trusted context"}
    for policy, label in labels.items():
        batch = [r for r in risk if r["policy"] == policy]
        values = [sum(row[field] for row in batch)
                  for field in ("suitable_calls_avoided", "unsuitable_reuses", "fallback_requests")]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "## Workload sensitivity scenarios",
        "",
        f"With exact repeats held at **15%**, increasing trusted-context coverage among eligible paraphrases "
        f"from **25% to 100%** moves the context-control projection from **{low:.0f} to {high:.0f} suitable "
        "scripted calls avoided per 1,000 requests**.",
        "",
        "| Exact repeats | Trusted context among eligible non-exact requests | Suitable calls avoided per 1,000 | Fallbacks per 1,000 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for repeats in (5, 15, 30):
        for resolved in (25, 50, 75, 100):
            row = context[f"repeat_{repeats:02}_context_{resolved:03}"]
            lines.append(f"| {repeats}% | {resolved}% | {row['suitable_avoided_per_1000']:.1f} | "
                         f"{row['fallback_per_1000']:.1f} |")
    lines += [
        "",
        "Scenario math uses the observed outcomes within four strata. The eligible share among non-exact requests "
        "stays fixed at 72/168. Changing trusted-context coverage reallocates eligible paraphrases between resolved "
        "and unresolved strata. Each request can avoid one scripted call. These are conditional projections "
        "from synthetic inputs. Segment outcomes and challenge composition could change in a real population. "
        "The V3 scorecard gives observed denominators and interval bounds for the holdout.",
        "",
        "## Interview walkthrough",
        "",
        '> “I defined the grain as one request per policy, reconciled it back to the benchmark, and split the '
        'savings gap into changes in traffic mix and changes within each segment. Then I kept the repeat share '
        'fixed and varied trusted-context coverage. That showed why data completeness is a prerequisite for '
        'useful semantic reuse. I would monitor suitable calls avoided, unsuitable reuses and fresh fallbacks together.”',
        "",
        "**Architecture in plain terms.** One agent workflow plans, drafts and reviews. Exact reuse can skip those "
        "steps for a matching approved response. Semantic reuse tests whether a paraphrase can reuse an answer within trusted "
        "reporting context. Routing adapts by request and falls back when suitability cannot be established.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/report_workload_population.py",
        "python scripts/report_workload_population.py --check",
        "```",
        "",
        "Use Python 3.11+ with SciPy installed for the existing holdout reconciliation. The report command operates "
        "on committed receipts and requires no model weights. For the interactive view, start the standard interview workbench "
        "with `python scripts/run_demo.py --interview --fresh` and open [Workload population](http://127.0.0.1:8765/benchmark/population.html).",
        "",
        "[SQL views](../scripts/workload_population.sql) · [Observed population CSV](../artifacts/workload_analysis/observed_population.csv) "
        "· [Savings-gap CSV](../artifacts/workload_analysis/savings_gap.csv) · [Risk population CSV](../artifacts/workload_analysis/risk_population.csv) "
        "· [Sensitivity CSV](../artifacts/workload_analysis/sensitivity.csv) · [Risk scorecard](RISK_V3_RESULTS.md)",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files against the committed receipts")
    args = parser.parse_args()
    history, gap, risk, scenarios, population, metrics = compute()
    outputs = {
        OUTPUT / "observed_population.csv": csv_content(history),
        OUTPUT / "savings_gap.csv": csv_content(gap),
        OUTPUT / "risk_population.csv": csv_content(risk),
        OUTPUT / "sensitivity.csv": csv_content(scenarios),
        REPORT: render(history, gap, risk, scenarios, population, metrics),
        INTERVIEW / "workload-population.json": json.dumps(
            {
                "historical": history, "gap": gap, "risk": risk, "scenarios": scenarios,
                "totals": dict(zip(("repeat_heavy_pct", "broader_pct", "mix_pp", "within_segment_pp"), metrics)),
            },
            indent=2,
        ) + "\n",
    }
    for path, value in outputs.items():
        if args.check:
            if path.read_bytes() != value.encode("utf-8"):
                raise SystemExit(f"Generated report differs: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
