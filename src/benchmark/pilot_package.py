"""Validate and redact a user-exported live-pilot result package."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.benchmark.schema import digest, read_json


def summarize_pilot_package(path: str | Path) -> dict:
    """Return an answer-free, secret-free summary after reconciling evidence.

    The source package may contain prompts and full model answers. This function
    deliberately exports only aggregate counts, hashes, configuration labels,
    token usage, and timing summaries.
    """
    source = Path(path)
    raw = source.read_bytes()
    package = read_json(source)
    if package.get("schema_version") != "pilot-result-package-v1":
        raise ValueError("unsupported pilot result package")

    trace = package.get("trace")
    report = package.get("report")
    journal = package.get("journal_diagnostics")
    if not all(isinstance(item, dict) for item in (trace, report, journal)):
        raise ValueError("package must include trace, report, and journal diagnostics")
    trace_hash = digest(trace)
    if package.get("trace_sha256") != trace_hash or report.get("trace_sha256") != trace_hash:
        raise ValueError("trace digest binding mismatch")
    if trace.get("evaluation_mode") != "live":
        raise ValueError("expected a live evaluation package")
    events = trace.get("events")
    calls = journal.get("calls")
    if not isinstance(events, list) or not isinstance(calls, list):
        raise ValueError("events and journal calls must be lists")

    status_counts: dict[str, int] = {}
    arm_attempts = {"cache": 0, "fresh": 0}
    arm_tokens = {"cache": {"prompt": 0, "completion": 0},
                  "fresh": {"prompt": 0, "completion": 0}}
    for event in events:
        status = event.get("status")
        if status not in ("HIT", "MISS", "BYPASS"):
            raise ValueError("event has unknown terminal cache status")
        status_counts[status] = status_counts.get(status, 0) + 1
        for arm in ("cache", "fresh"):
            observation = event.get(arm)
            if not isinstance(observation, dict):
                raise ValueError("paired event is missing an arm")
            attempts = observation.get("provider_calls")
            if type(attempts) is not int or attempts < 0:
                raise ValueError("provider call count must be a nonnegative integer")
            arm_attempts[arm] += attempts
            for usage in observation.get("usage", []):
                arm_tokens[arm]["prompt"] += int(usage.get("prompt_tokens", 0))
                arm_tokens[arm]["completion"] += int(usage.get("completion_tokens", 0))

    if status_counts != report.get("status_counts"):
        raise ValueError("event statuses do not reconcile with report")
    for arm in ("cache", "fresh"):
        summary = report.get(f"{arm}_arm", {})
        if arm_attempts[arm] != summary.get("provider_attempts"):
            raise ValueError(f"{arm} provider attempts do not reconcile")
        if arm_tokens[arm]["prompt"] != summary.get("observed_prompt_tokens"):
            raise ValueError(f"{arm} prompt token usage does not reconcile")
        if arm_tokens[arm]["completion"] != summary.get("observed_completion_tokens"):
            raise ValueError(f"{arm} completion token usage does not reconcile")
    if len(calls) != arm_attempts["cache"] + arm_attempts["fresh"] + int(
            report.get("verifier_work", {}).get("provider_attempts", 0)):
        raise ValueError("journal call count does not reconcile with serving report")
    if any(call.get("state") != "complete" for call in calls):
        raise ValueError("journal contains incomplete calls")

    model = trace.get("pilot_config", {}).get("models", [None])
    model = model[0] if model else None
    return {
        "schema_version": "redacted-live-pilot-audit-v1",
        "source_package_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_sha256": trace_hash,
        "run_id": journal.get("run_id"),
        "evaluation_mode": trace.get("evaluation_mode"),
        "execution_mode": trace.get("execution_mode"),
        "workload": {
            "name": trace.get("workload_name"),
            "sha256": trace.get("workload_sha256"),
            "nature": trace.get("nature"),
            "split": trace.get("split"),
            "provenance": trace.get("provenance"),
            "requests": len(events),
            "working_tree_dirty": trace.get("working_tree_dirty"),
            "code_tree": trace.get("code_tree"),
        },
        "model": model,
        "cache_policy": trace.get("config", {}).get("policy"),
        "latency_valid": trace.get("latency_valid"),
        "status_counts": status_counts,
        "nonexact_reuse": report.get("nonexact_reuse"),
        "provider_attempts": {
            "cache_arm": arm_attempts["cache"],
            "fresh_arm": arm_attempts["fresh"],
            "verifier": report.get("verifier_work", {}).get("provider_attempts", 0),
            "avoided": report.get("provider_attempts_avoided"),
            "journaled_live_calls": len(calls),
        },
        "provider_tokens": {
            "cache_arm": arm_tokens["cache"],
            "fresh_arm": arm_tokens["fresh"],
            "fresh_minus_cache": report.get("token_difference_fresh_minus_cache"),
        },
        "latency_ms": {
            "cache_arm_mean": report.get("cache_arm", {}).get("all_attempt_latency_ms_mean"),
            "fresh_arm_mean": report.get("fresh_arm", {}).get("all_attempt_latency_ms_mean"),
            "paired_mean_cache_minus_fresh": report.get("paired_latency", {}).get("mean_cache_minus_fresh_ms"),
            "paired_group_bootstrap_95": report.get("paired_latency", {}).get("group_bootstrap_95"),
            "paired_comparisons": report.get("paired_latency", {}).get("complete_pairs"),
        },
        "answer_review": {
            "reviewed_hits": report.get("overall_reuse", {}).get("reviewed"),
            "unreviewed_hits": report.get("overall_reuse", {}).get("unreviewed"),
            "safety_certified": report.get("safety_certified"),
            "quality_pairs_reviewed": report.get("paired_answer_quality", {}).get("reviewed_pairs"),
        },
        "limitations": report.get("limitations"),
    }
