import json

import pytest

from src.benchmark.pilot_package import summarize_pilot_package
from src.benchmark.schema import digest


def _package():
    events = [
        {"status": "MISS", "cache": {"provider_calls": 1, "usage": [{"prompt_tokens": 2, "completion_tokens": 3}]},
         "fresh": {"provider_calls": 1, "usage": [{"prompt_tokens": 2, "completion_tokens": 3}]}},
        {"status": "HIT", "cache": {"provider_calls": 0, "usage": []},
         "fresh": {"provider_calls": 1, "usage": [{"prompt_tokens": 2, "completion_tokens": 3}]}},
    ]
    trace = {"evaluation_mode": "live", "execution_mode": "live_uninterrupted", "events": events,
             "workload_name": "authored", "workload_sha256": "a" * 64, "nature": "synthetic_fixture",
             "split": "dev", "provenance": "constructed", "working_tree_dirty": False,
             "code_tree": "b" * 40, "latency_valid": True, "pilot_config": {"models": ["free-model"]},
             "config": {"policy": "exact"}}
    trace_hash = digest(trace)
    report = {"trace_sha256": trace_hash, "status_counts": {"MISS": 1, "HIT": 1},
              "cache_arm": {"provider_attempts": 1, "observed_prompt_tokens": 2, "observed_completion_tokens": 3,
                            "all_attempt_latency_ms_mean": 20},
              "fresh_arm": {"provider_attempts": 2, "observed_prompt_tokens": 4, "observed_completion_tokens": 6,
                            "all_attempt_latency_ms_mean": 30},
              "provider_attempts_avoided": 1, "token_difference_fresh_minus_cache": {"prompt_tokens": 2, "completion_tokens": 3},
              "verifier_work": {"provider_attempts": 0}, "nonexact_reuse": {"accepted": 0},
              "paired_latency": {"mean_cache_minus_fresh_ms": -10, "group_bootstrap_95": [-12, 2], "complete_pairs": 2},
              "overall_reuse": {"reviewed": 0, "unreviewed": 1},
              "paired_answer_quality": {"reviewed_pairs": 0}, "safety_certified": False, "limitations": []}
    return {"schema_version": "pilot-result-package-v1", "trace": trace, "report": report,
            "trace_sha256": trace_hash,
            "journal_diagnostics": {"run_id": "test", "calls": [{"state": "complete"}] * 3}}


def test_redacted_summary_reconciles_and_omits_raw_answers(tmp_path):
    package = _package()
    # Raw answer-like content must never be copied into the audit projection.
    package["trace"]["events"][0]["private_answer"] = "do not export this answer"
    package["trace_sha256"] = digest(package["trace"])
    package["report"]["trace_sha256"] = package["trace_sha256"]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(package), encoding="utf-8")

    result = summarize_pilot_package(path)

    assert result["status_counts"] == {"MISS": 1, "HIT": 1}
    assert result["provider_attempts"]["avoided"] == 1
    assert result["provider_tokens"]["fresh_minus_cache"]["completion_tokens"] == 3
    assert "private_answer" not in json.dumps(result)
    assert "do not export" not in json.dumps(result)


@pytest.mark.parametrize("tamper", ["package_hash", "report_hash", "status", "attempts", "tokens", "journal"])
def test_rejects_unreconciled_evidence(tmp_path, tamper):
    package = _package()
    if tamper == "package_hash":
        package["trace_sha256"] = "0" * 64
    elif tamper == "report_hash":
        package["report"]["trace_sha256"] = "0" * 64
    elif tamper == "status":
        package["report"]["status_counts"]["HIT"] = 0
    elif tamper == "attempts":
        package["report"]["cache_arm"]["provider_attempts"] += 1
    elif tamper == "tokens":
        package["report"]["fresh_arm"]["observed_prompt_tokens"] += 1
    else:
        package["journal_diagnostics"]["calls"].pop()
    path = tmp_path / "results.json"
    path.write_text(json.dumps(package), encoding="utf-8")

    with pytest.raises(ValueError):
        summarize_pilot_package(path)
