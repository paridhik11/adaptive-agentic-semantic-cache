"""Reconcile frozen policy receipts with SQLite and render the oversight report."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from evaluate_risk_v2 import ROOT, SEAL, inputs, load_groups, proportion

DATA = ROOT / "artifacts" / "risk_v2"
REPORT = ROOT / "docs" / "RISK_V2_RESULTS.md"
ARMS = {"exact_only": "Exact only", "current_guarded": "Current guarded", "reporting_control": "Guarded + reporting check"}


def reconcile(result):
    expected = {
        (group["id"], probe["type"], arm): (
            probe["eligible"], hashlib.sha256(group["answer"].encode("utf-8")).hexdigest()
        )
        for group in load_groups() if group["split"] == "holdout"
        for probe in group["probes"] for arm in ARMS
    }
    if result["split"] != "holdout" or result["sealed_inputs"] != inputs():
        raise ValueError("Holdout receipt differs from source seal")
    receipts = result["receipts"]
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("""
          CREATE TABLE probes (
            intent TEXT, scenario TEXT, arm TEXT, eligible INTEGER, served INTEGER,
            calls INTEGER, tier TEXT, PRIMARY KEY (intent, scenario, arm)
          )
        """)
        fallback_hash = hashlib.sha256(b"Controlled fallback marker").hexdigest()
        for row in receipts:
            key = row["group"], row["type"], row["policy"]
            eligible, answer_hash = expected.pop(key)
            served, tier = row["served"], row["served_tier"]
            if (eligible != row["eligible"] or row["calls"] != 1 - served
                    or row["answer_sha256"] != (answer_hash if served else fallback_hash)
                    or tier not in ("EXACT", "SEMANTIC", "NONE")
                    or served != (tier != "NONE") or (tier == "EXACT") != (key[1] == "exact_repeat")):
                raise ValueError(f"Invalid receipt for {key}")
            conn.execute("INSERT INTO probes VALUES (?,?,?,?,?,?,?)", (*key, eligible, served, row["calls"], tier))
        if expected:
            raise ValueError("Missing paired probe receipts")
        arm_rows = conn.execute("""
          SELECT arm, COUNT(*), SUM(1 - calls), SUM(eligible * served),
                 SUM((1 - eligible) * served), SUM(1 - eligible),
                 SUM(tier = 'SEMANTIC'), SUM((tier = 'SEMANTIC') * (1 - eligible)),
                 SUM(eligible * (scenario != 'exact_repeat')),
                 SUM(eligible * (scenario != 'exact_repeat') * served)
          FROM probes GROUP BY arm
        """).fetchall()
        for arm, count, gross, eligible, incorrect, challenges, semantic, semantic_wrong, opportunities, capture in arm_rows:
            metric = result["arms"][arm]
            if (count, gross, eligible, proportion(incorrect, challenges), proportion(semantic_wrong, semantic),
                    proportion(capture, opportunities)) != (
                metric["requests"], metric["scripted_calls_avoided"], metric["calls_avoided_on_eligible_rows"],
                metric["challenge_accepts"], metric["semantic_incorrect_reuse"], metric["eligible_non_exact_capture"],
            ):
                raise ValueError(f"SQL reconciliation failed for {arm}")
        deltas = conn.execute("""
          SELECT b.eligible, b.served, c.served, COUNT(*)
          FROM probes b JOIN probes c USING (intent, scenario)
          WHERE b.arm = 'current_guarded' AND c.arm = 'reporting_control'
            AND b.served != c.served
          GROUP BY b.eligible, b.served, c.served
        """).fetchall()
        counts = {(eligible, before, after): count for eligible, before, after, count in deltas}
        for eligible in (0, 1):
            for before in (0, 1):
                key = ("valid" if eligible else "incorrect") + ("_removed" if before else "_added")
                if result["paired_changes"][key] != counts.get((eligible, before, 1 - before), 0):
                    raise ValueError(f"SQL reconciliation failed for {key}")
        return conn.execute("""
          SELECT b.scenario, COUNT(*), SUM(b.served), SUM(c.served)
          FROM probes b JOIN probes c USING (intent, scenario)
          WHERE b.arm = 'current_guarded' AND c.arm = 'reporting_control' AND b.eligible = 0
          GROUP BY b.scenario ORDER BY COUNT(*) DESC, b.scenario
        """).fetchall()


def rate(metric):
    if metric["n"] == 0:
        return "— (0 accepts)"
    low, high = metric["exact_95_ci"]
    return f"{metric['count']}/{metric['n']} ({metric['rate']:.1%}, 95% CI {low:.1%}–{high:.1%})"


def render(holdout, scenarios, pilot):
    changes = holdout["paired_changes"]
    arms = holdout["arms"]
    examples = arms["exact_only"]
    lines = [
        "# Cache reuse risk oversight",
        "",
        "**Decision.** Keep exact reuse as the default. The reporting check met the fixture criteria for follow-up testing. Remaining failures need stronger controls for changes to period, scope and entity.",
        "",
        f"Across **{examples['requests']} matched requests in {examples['groups']} intent groups**, the reporting check prevented "
        f"**{changes['incorrect_removed']} unsuitable reuses** and lost **{changes['valid_removed']} suitable reuses** versus the same guarded pipeline. "
        f"It introduced **{changes['incorrect_added']} unsuitable reuses**. Both guarded configurations avoided "
        f"**{arms['reporting_control']['calls_avoided_on_eligible_rows']} scripted calls on eligible requests**.",
        "",
        "## Holdout scorecard",
        "",
        "| Configuration | Unsuitable answer / semantic accepts | Eligible non-exact capture | Challenge accepts | Gross / eligible scripted calls avoided |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in ARMS.items():
        row = arms[key]
        lines.append(
            f"| {label} | {rate(row['semantic_incorrect_reuse'])} | {rate(row['eligible_non_exact_capture'])} | "
            f"{rate(row['challenge_accepts'])} | {row['scripted_calls_avoided']} / {row['calls_avoided_on_eligible_rows']} |"
        )
    lines += [
        "",
        "Gross savings include work avoided by returning unsuitable answers. Eligible savings count only scripted calls avoided on suitable fixture answers. "
        "The no-cache denominator is one scripted call per probe. Priming is excluded from these savings.",
        "",
        "## Exception drilldown",
        "",
        "| Challenge | Requests | Guarded accepts | Accepts with reporting check |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scenario, n, before, after in scenarios:
        lines.append(f"| {scenario.replace('_', ' ').capitalize()} | {n} | {before} | {after} |")
    lines += [
        "",
        "Inputs are fictional operational reporting fixtures. Labels apply to the particular cached answer identified by its hash in each receipt. "
        "Intent groups contain related variants. Exact intervals are event-level and descriptive under the binomial assumption. "
        "The holdout is now spent for development, and repeats reproduce the frozen run. The experiment leaves serving policy decisions outside the harness.",
    ]
    if pilot is not None:
        lines += [
            "",
            "## Paired local timing pilot",
            "",
            f"On **{pilot['pairs']} warm literal repeats** using Qwen2.5-0.5B-Instruct on CPU, "
            f"fresh generation measured **{pilot['fresh_p50_ms']:.2f} ms p50** and exact reuse **{pilot['cached_p50_ms']:.3f} ms p50**. "
            f"The fresh arm generated **{pilot['fresh_completion_tokens']} completion tokens**, and the reuse arm generated zero. "
            "The pilot verifies identical returned answers for each pair. Priming is measured separately. "
            "This is a local warm-cache component measurement with machine-specific timing.",
        ]
    lines += [
        "",
        "## Reproduce",
        "",
        "With the exact MiniLM encoder and passage scorer checkpoints already in the local Hugging Face cache:",
        "",
        "```bash",
        "python scripts/evaluate_risk_v2.py calibration --output calibration-replay.json",
        "python scripts/evaluate_risk_v2.py holdout --output holdout-replay.json",
        "python scripts/report_risk_v2.py --check",
        "```",
        "",
        "[Frozen protocol](RISK_V2_PROTOCOL.md) · [Input hashes](../artifacts/risk_v2/seal.json) · [Holdout receipts](../artifacts/risk_v2/holdout.json)",
    ]
    if pilot is not None:
        lines += [
            "",
            "With the pinned Qwen checkpoint already cached, run `python scripts/measure_risk_v2_exact.py --output local-pilot-replay.json`.",
            "",
            "[Timing protocol](RISK_V2_LATENCY_PROTOCOL.md) · [Local timing receipts](../artifacts/risk_v2/local_exact_pilot.json)",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if json.loads(SEAL.read_text(encoding="utf-8")) != inputs():
        raise SystemExit("Frozen source or fixture changed")
    holdout = json.loads((DATA / "holdout.json").read_text(encoding="utf-8"))
    path = DATA / "local_exact_pilot.json"
    pilot = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    report = render(holdout, reconcile(holdout), pilot)
    if args.check:
        if REPORT.read_text(encoding="utf-8") != report:
            raise SystemExit("Risk report differs from its reconciled source")
    else:
        REPORT.write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
