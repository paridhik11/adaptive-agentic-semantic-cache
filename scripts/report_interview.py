"""Reconcile the committed offline campaign and export the interview dashboard data."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts" / "metrics_campaign"
OUTPUT = ROOT / "dashboard" / "interview"
PROFILES = ("repeat_rich", "lower_repeat")
ARMS = ("no_cache", "exact_only", "verified_semantic", "shadow_semantic")
CATEGORIES = ("seed", "exact_repeat", "natural_paraphrase", "adversarial_challenge")
MATCH_FIELDS = (
    "intent_id", "group_id", "category", "challenge_type", "question"
)
REQUIRED = set(MATCH_FIELDS) | {
    "arm", "event_id", "is_hit", "tier", "provider_calls", "total_tokens_synthetic", "workload_expected_reuse",
    "elapsed_ms", "avoided_calls", "terminal_status", "matched_question", "actual_response_answer",
}
FAILURE_ID = "async_05:challenge_added_requirements"


def read_events(raw):
    arms = {arm: {} for arm in ARMS}
    for number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"line {number}: missing fields {sorted(missing)}")
        arm, event = row["arm"], row["event_id"]
        if arm not in arms or not isinstance(event, str) or not event:
            raise ValueError(f"line {number}: invalid arm or event ID")
        if event in arms[arm]:
            raise ValueError(f"line {number}: duplicate event {arm}/{event}")
        for key in ("provider_calls", "total_tokens_synthetic", "avoided_calls"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"line {number}: invalid {key}")
        elapsed = row["elapsed_ms"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"line {number}: invalid elapsed_ms")
        for key in ("is_hit", "workload_expected_reuse"):
            if type(row[key]) is not bool:
                raise ValueError(f"line {number}: invalid {key}")
        if (row["category"] not in CATEGORIES or row["terminal_status"] != "COMPLETED"
                or row["tier"] not in ("NONE", "EXACT", "SEMANTIC")
                or row["is_hit"] != (row["tier"] != "NONE")):
            raise ValueError(f"line {number}: invalid category, terminal status or cache tier")
        if row["workload_expected_reuse"] != (row["is_hit"] and row["category"] != "adversarial_challenge"):
            raise ValueError(f"line {number}: eligibility flag disagrees with the workload policy")
        arms[arm][event] = row
    return arms


def validate_pairs(arms, manifest):
    baseline = arms["no_cache"]
    if not baseline:
        raise ValueError("missing no-cache baseline")
    for arm, rows in arms.items():
        missing, extra = baseline.keys() - rows.keys(), rows.keys() - baseline.keys()
        if missing or extra:
            raise ValueError(f"{arm}: missing event IDs {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")
        for event, reference in baseline.items():
            row = rows[event]
            for field in MATCH_FIELDS:
                if row[field] != reference[field]:
                    raise ValueError(f"{arm}/{event}: mismatched {field}")
            if reference["provider_calls"] - row["provider_calls"] != row["avoided_calls"]:
                raise ValueError(f"{arm}/{event}: avoided calls disagree with paired baseline")
            if row["is_hit"] and (arm == "no_cache" or (arm != "verified_semantic" and row["tier"] == "SEMANTIC")):
                raise ValueError(f"{arm}/{event}: cache tier incompatible with arm")
    observed = {
        "total_events": len(baseline),
        "executed_source_groups_count": len({row["group_id"] for row in baseline.values()}),
        "executed_base_intents_count": len({row["intent_id"] for row in baseline.values()}),
        "event_category_distribution": Counter(row["category"] for row in baseline.values()),
    }
    for field, value in observed.items():
        if value != manifest[field]:
            raise ValueError(f"{field}: events give {value}, manifest gives {manifest[field]}")


def percent(numerator, denominator):
    return round(numerator / denominator * 100, 4) if denominator else None


def latency(rows):
    times = [row["elapsed_ms"] for row in rows]
    if not times:
        return {"n": 0, "p50": None, "p95": None}
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    return {"n": len(times), "p50": round(statistics.median(times), 4), "p95": round(p95, 4)}


def proportion(hits, total):
    interval = None
    if total:
        ci = binomtest(hits, total).proportion_ci(confidence_level=0.95, method="exact")
        interval = [round(ci.low * 100, 4), round(ci.high * 100, 4)]
    return {"hits": hits, "n": total, "pct": percent(hits, total), "exact_95_ci_pct": interval}


def summarize(rows, baseline_calls, baseline_tokens):
    calls = sum(row["provider_calls"] for row in rows)
    tokens = sum(row["total_tokens_synthetic"] for row in rows)
    challenge = [row for row in rows if row["category"] == "adversarial_challenge"]
    hits = [row for row in rows if row["is_hit"]]
    coverage = {}
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        eligible = [row for row in group if row["category"] != "adversarial_challenge"]
        coverage[category] = {
            "hits": sum(row["is_hit"] for row in group), "requests": len(group),
            "eligible_hits": sum(row["is_hit"] for row in eligible), "eligible": len(eligible),
        }
    return {
        "calls": calls, "calls_saved": baseline_calls - calls,
        "calls_saved_pct": percent(baseline_calls - calls, baseline_calls),
        "calls_saved_on_eligible_hits": sum(row["avoided_calls"] for row in hits if row["workload_expected_reuse"]),
        "calls_saved_on_ineligible_hits": sum(row["avoided_calls"] for row in hits if not row["workload_expected_reuse"]),
        "synthetic_tokens": tokens, "synthetic_tokens_saved_pct": percent(baseline_tokens - tokens, baseline_tokens),
        "latency_ms": {"all": latency(rows), "hit": latency(hits), "miss": latency([r for r in rows if not r["is_hit"]])},
        "answer_review": {
            "total_accepts": len(hits), "reviewed_accepts": 0, "incorrect": None,
            "incorrect_rate_pct": None, "exact_95_ci_pct": None, "label_source": "unadjudicated",
        },
        "adversarial_false_accepts": proportion(sum(row["is_hit"] for row in challenge), len(challenge)),
        "coverage": coverage,
    }


def profile_report(directory, profile):
    raw = (directory / "offline_results.jsonl").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["workload_profile"] != profile:
        raise ValueError(f"manifest says {manifest['workload_profile']}, expected {profile}")
    arms = read_events(raw)
    validate_pairs(arms, manifest)
    baseline_calls = sum(row["provider_calls"] for row in arms["no_cache"].values())
    baseline_tokens = sum(row["total_tokens_synthetic"] for row in arms["no_cache"].values())
    metrics = {arm: summarize(list(rows.values()), baseline_calls, baseline_tokens) for arm, rows in arms.items()}
    exact, semantic = arms["exact_only"], arms["verified_semantic"]
    differences = [exact[event]["provider_calls"] - row["provider_calls"] for event, row in semantic.items()]
    failures = []
    if profile == "lower_repeat":
        failure = semantic[FAILURE_ID]
        if failure["category"] != "adversarial_challenge" or not failure["is_hit"]:
            raise ValueError(f"documented challenge no longer reconciles: {FAILURE_ID}")
        failures.append({
            key: failure[key] for key in
            ("event_id", "question", "matched_question", "actual_response_answer", "tier", "provider_calls")
        } | {"exact_only_calls": exact[FAILURE_ID]["provider_calls"]})
    return {
        "events_per_arm": manifest["total_events"], "source_groups": manifest["executed_source_groups_count"],
        "intents": manifest["executed_base_intents_count"],
        "source": f"artifacts/metrics_campaign/{profile}/offline_results.jsonl",
        "sha256": hashlib.sha256(raw).hexdigest(), "arms": metrics, "failures": failures,
        "semantic_increment": {
            "calls_saved": sum(differences), "pct_of_baseline_calls": percent(sum(differences), baseline_calls),
            "events_with_fewer_calls": sum(delta > 0 for delta in differences),
            "events_with_more_calls": sum(delta < 0 for delta in differences),
            "calls_saved_on_eligible_hits": metrics["verified_semantic"]["calls_saved_on_eligible_hits"]
            - metrics["exact_only"]["calls_saved_on_eligible_hits"],
        },
    }


def export(campaign):
    reports = {}
    for profile in PROFILES:
        try:
            reports[profile] = profile_report(campaign / profile, profile)
        except (ValueError, KeyError, OSError, AttributeError) as error:
            raise ValueError(f"{profile}: {error}") from error
    report = {
        "schema_version": 1,
        "evidence": "offline replay: scripted provider calls, synthetic tokens, historical wall-clock latency",
        "events_replayed": sum(profile["events_per_arm"] for profile in reports.values()),
        "executions": sum(profile["events_per_arm"] * len(ARMS) for profile in reports.values()),
        "profiles": reports,
    }
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(("workload", "configuration", "events", "scripted_calls", "calls_saved", "calls_saved_pct",
                     "synthetic_tokens", "synthetic_tokens_saved_pct", "latency_p50_ms", "latency_p95_ms",
                     "false_accepts", "adversarial_challenges", "reviewed_accepts", "total_accepts"))
    for title, profile in reports.items():
        for arm, row in profile["arms"].items():
            timing, challenge, review = row["latency_ms"]["all"], row["adversarial_false_accepts"], row["answer_review"]
            writer.writerow((title, arm, profile["events_per_arm"], row["calls"], row["calls_saved"], row["calls_saved_pct"],
                             row["synthetic_tokens"], row["synthetic_tokens_saved_pct"], timing["p50"], timing["p95"],
                             challenge["hits"], challenge["n"], review["reviewed_accepts"], review["total_accepts"]))
    return {"report.json": json.dumps(report, indent=2) + "\n", "comparison.csv": csv_buffer.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if checked-in dashboard exports have drifted")
    args = parser.parse_args()
    try:
        files = export(CAMPAIGN)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, contents in files.items():
            path = args.output_dir / name
            if args.check:
                if path.read_bytes() != contents.encode("utf-8"):
                    parser.exit(1, f"{path} differs from the telemetry. Regenerate it and review the diff.\n")
            else:
                path.write_text(contents, encoding="utf-8", newline="\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Benchmark reconciliation failed: {error}\n")
    print(files["comparison.csv"], end="")


if __name__ == "__main__":
    main()
