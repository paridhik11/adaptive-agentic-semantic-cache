"""Reconcile sealed context-control evidence and generate a short scorecard."""
import argparse
from contextlib import closing
import hashlib
import json
import sqlite3

from evaluate_risk_v3 import ARMS, CHANGED, DATA, ROOT, compare, hashes, load, proportion, summarize

REPORT = ROOT / "docs/RISK_V3_RESULTS.md"
LABELS = {
    "exact": "Exact only", "guarded": "Current guarded",
    "reporting": "Guarded + V2 reporting check", "context": "V2 + verified context",
}


def reconcile(report):
    if report["split"] != "holdout" or report["sealed_inputs"] != hashes():
        raise ValueError("holdout receipt differs from the source seal")
    expected = {
        (group["id"], probe["type"], arm): (probe, hashlib.sha256(group["answer"].encode()).hexdigest())
        for group in load() if group["split"] == "holdout"
        for probe in group["probes"] for arm in ARMS
    }
    fallback = hashlib.sha256(b"Controlled fallback marker").hexdigest()
    arms = {arm: [] for arm in ARMS}
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("""
          CREATE TABLE receipt (
            intent TEXT, scenario TEXT, arm TEXT, eligible INTEGER, served INTEGER,
            PRIMARY KEY (intent, scenario, arm))
        """)
        for row in report["receipts"]:
            key = row["group"], row["type"], row["policy"]
            probe, answer = expected.pop(key)
            served = row["served"]
            expected_tier = ("EXACT" if key[1] == "exact_repeat" else "SEMANTIC") if served else "NONE"
            if (type(served) is not bool or row["eligible"] != probe["eligible"]
                    or row["resolved"] != (probe["context"] is not None)
                    or row["response_sha256"] != (answer if served else fallback)
                    or row["calls"] != 1 - served or row["served_tier"] != expected_tier):
                raise ValueError(f"malformed receipt: {key}")
            db.execute("INSERT INTO receipt VALUES (?,?,?,?,?)", (*key, row["eligible"], served))
            arms[row["policy"]].append(row)
        if expected:
            raise ValueError("missing paired requests")

        for arm, rows in arms.items():
            stored = report["arms"][arm]
            recomputed = summarize(rows)
            for name, subset in (
                ("resolved_nonexact_capture", [r for r in rows if r["eligible"] and r["resolved"] and r["type"] != "exact_repeat"]),
                ("unresolved_capture", [r for r in rows if not r["resolved"]]),
                ("changed_context_accepts", [r for r in rows if r["type"] in CHANGED]),
            ):
                recomputed[name] = proportion(sum(r["served"] for r in subset), len(subset))
            if stored != recomputed:
                raise ValueError(f"aggregate failed reconciliation for {arm}")

        paired = compare(arms["reporting"], arms["context"])
        if paired != report["paired_reporting_vs_context"]:
            raise ValueError("paired policy transitions failed reconciliation")
        table = db.execute("""
          SELECT scenario, COUNT(*), SUM(b.served), SUM(c.served)
          FROM receipt b JOIN receipt c USING (intent, scenario)
          WHERE b.arm = 'reporting' AND c.arm = 'context'
          GROUP BY scenario ORDER BY scenario
        """).fetchall()
        groups = db.execute("""
          SELECT arm, COUNT(DISTINCT intent), COUNT(DISTINCT CASE WHEN served = 1 THEN intent END)
          FROM receipt WHERE scenario IN ('entity', 'period', 'scope') GROUP BY arm
        """).fetchall()
    return table, {arm: proportion(affected, n) for arm, n, affected in groups}


def display(metric):
    if metric["n"] == 0:
        return "—"
    lo, hi = metric["exact_95_ci"]
    return f"{metric['count']}/{metric['n']} ({lo:.1%}–{hi:.1%} CI)"


def render(report, table, groups):
    arms = report["arms"]
    before, after = arms["reporting"], arms["context"]
    delta = report["paired_reporting_vs_context"]
    material = after["changed_context_accepts"]
    followup = delta["material_unsuitable_prevented"] > 0 and not (
        delta["unsuitable_gained"] or delta["resolved_suitable_lost"]
    )
    if report["meets_followup_criterion"] != followup:
        raise ValueError("follow-up decision disagrees with paired results")
    decision = (
        "Advance the context control to follow-up testing for applications that resolve trusted report coordinates."
        if followup else "Review the context control against the failed follow-up criterion before further testing."
    )
    heading = [
        "# Reporting context risk and efficiency",
        "",
        f"**Decision.** {decision} Keep exact-only as the release default.",
        "",
        f"On **{after['requests']} matched probes across {after['groups']} held-out intent groups**, "
        f"adding report coordinates reduced unsuitable semantic accepts from **{before['semantic_incorrect_reuse']['count']} "
        f"to {after['semantic_incorrect_reuse']['count']}**. For changed periods, entities and populations, "
        f"the context arm accepted **{material['count']}/{material['n']}** requests "
        f"(95% exact event-level upper bound **{material['exact_95_ci'][1]:.1%}**). "
        f"It lost **{delta['resolved_suitable_lost']}** suitable reuses when coordinates were resolved. "
        f"It gave up **{delta['suitable_lost']}** suitable reuses overall.",
        "",
        "## Four configurations on the same requests",
        "",
        "| Configuration | Unsuitable / semantic accepts | Changed-context accepts | Resolved eligible non-exact capture | Unresolved eligible capture | Gross / eligible scripted calls avoided |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in LABELS.items():
        arm = arms[key]
        heading.append(
            f"| {label} | {display(arm['semantic_incorrect_reuse'])} | {display(arm['changed_context_accepts'])} | "
            f"{display(arm['resolved_nonexact_capture'])} | {display(arm['unresolved_capture'])} | "
            f"{arm['scripted_calls_avoided']} / {arm['calls_avoided_on_eligible_rows']} |"
        )
    heading += [
        "",
        f"The context policy avoided **{after['calls_avoided_on_eligible_rows']} suitable scripted calls** versus "
        f"**{arms['exact']['calls_avoided_on_eligible_rows']} for exact-only**. Gross savings credit every hit. "
        "Eligible savings credit only reuse of suitable fixture answers. The denominator for the scripted benchmark is one call per probe without caching, excluding priming.",
        "",
        "## Exceptions and fallbacks",
        "",
        "| Request scenario | Requests | V2 guarded accepts | Context-control accepts |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scenario, n, current, proposed in table:
        heading.append(f"| {scenario.replace('_', ' ').capitalize()} | {n} | {current} | {proposed} |")
    heading += [
        "",
        f"At the intent-group level, **{groups['context']['count']}/{groups['context']['n']}** had any changed-context exception "
        f"(95% exact upper bound **{groups['context']['exact_95_ci'][1]:.1%}**). "
        "All intervals use exact Clopper–Pearson 95% bounds. Event-level intervals are descriptive because related probes share intents.",
        "",
        "The data are synthetic. Coordinates are supplied by a trusted caller in this harness. The evaluation tests enforcing those coordinates at lookup. "
        "It does not measure extraction or authorization from free-form user text. The held-out intent groups use shared wording patterns. "
        "No billable inference or end-to-end production latency was measured here.",
        "",
        "## Reproduce and audit",
        "",
        "Use the pinned local MiniLM checkpoints from the [V2 setup instructions](RISK_V2_RESULTS.md#reproduce).",
        "",
        "```bash",
        "python scripts/evaluate_risk_v3.py calibration --output calibration-replay-v3.json",
        "python scripts/evaluate_risk_v3.py holdout --output holdout-replay-v3.json",
        "python scripts/report_risk_v3.py --check",
        "```",
        "",
        "The first holdout is spent. A replay reproduces the frozen comparison. The report checker independently pairs request receipts in SQLite, "
        "checks each served-answer digest against the fixture, and reconciles aggregate metrics.",
        "",
        "[Protocol](RISK_V3_PROTOCOL.md) · [Frozen source hashes](../artifacts/risk_v3/seal.json) · [Holdout receipts](../artifacts/risk_v3/holdout.json) · [Original workload benchmarks](INTERVIEW_V1.md#numbers-to-defend)",
    ]
    return "\n".join(heading) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if json.loads((DATA / "seal.json").read_text(encoding="utf-8")) != hashes():
        raise SystemExit("source seal changed")
    result = json.loads((DATA / "holdout.json").read_text(encoding="utf-8"))
    text = render(result, *reconcile(result))
    if args.check:
        if REPORT.read_text(encoding="utf-8") != text:
            raise SystemExit("report differs from evidence")
    else:
        REPORT.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
