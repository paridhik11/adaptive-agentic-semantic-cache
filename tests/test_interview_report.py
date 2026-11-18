from copy import deepcopy
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from scripts.report_interview import ARMS, CAMPAIGN, OUTPUT, export, latency, read_events, validate_pairs
from src.api.demo import create_demo_app


@pytest.fixture
def pairs():
    rows = [
        dict(arm=arm, event_id="test", intent_id="intent", group_id="group", category="exact_repeat",
             challenge_type=None, question="test?", is_hit=arm == "exact_only",
             tier="EXACT" if arm == "exact_only" else "NONE", provider_calls=0 if arm == "exact_only" else 3,
             total_tokens_synthetic=0 if arm == "exact_only" else 10, elapsed_ms=5,
             avoided_calls=3 if arm == "exact_only" else 0, terminal_status="COMPLETED",
             workload_expected_reuse=arm == "exact_only", matched_question=None, actual_response_answer=None)
        for arm in ARMS
    ]
    return rows


def encoded(rows):
    return "\n".join(map(json.dumps, rows)).encode()


def test_matching_rejects_duplicates_missing_and_changed_events(pairs):
    manifest = dict(total_events=1, executed_source_groups_count=1, executed_base_intents_count=1,
                    event_category_distribution={"exact_repeat": 1})
    validate_pairs(read_events(encoded(pairs)), manifest)
    with pytest.raises(ValueError, match="duplicate event"):
        read_events(encoded(pairs + [pairs[0]]))
    with pytest.raises(ValueError, match="missing event IDs"):
        validate_pairs(read_events(encoded(pairs[:-1])), manifest)
    mutated = deepcopy(pairs)
    mutated[-1]["question"] = "other question?"
    with pytest.raises(ValueError, match="mismatched question"):
        validate_pairs(read_events(encoded(mutated)), manifest)
    del mutated[-1]["elapsed_ms"]
    with pytest.raises(ValueError, match="missing fields.*elapsed_ms"):
        read_events(encoded(mutated))


def test_latency_sample_size():
    assert latency([]) == {"n": 0, "p50": None, "p95": None}
    assert latency([{"elapsed_ms": 2}, {"elapsed_ms": 4}]) == {"n": 2, "p50": 3, "p95": 4}


def test_campaign_reconciles_resume_and_incremental_savings():
    files = export(CAMPAIGN)
    for name, content in files.items():
        assert (OUTPUT / name).read_bytes() == content.encode("utf-8")
    report = json.loads(files["report.json"])
    assert (report["events_replayed"], report["executions"]) == (1000, 4000)
    repeat, broad = (report["profiles"][profile] for profile in ("repeat_rich", "lower_repeat"))
    for profile, calls, tokens, extra, challenges in (
        (repeat, 47.33, 46.57, 21, (0, 120)), (broad, 14.75, 14.56, 48, (1, 125))
    ):
        exact = profile["arms"]["exact_only"]
        guarded = profile["arms"]["verified_semantic"]
        assert round(exact["calls_saved_pct"], 2) == calls
        assert round(exact["synthetic_tokens_saved_pct"], 2) == tokens
        assert profile["semantic_increment"]["calls_saved"] == extra
        adverse = guarded["adversarial_false_accepts"]
        assert (adverse["hits"], adverse["n"]) == challenges
        assert adverse["exact_95_ci_pct"][1] > 0
        assert guarded["answer_review"]["incorrect_rate_pct"] is None


def test_paired_sql_drilldown_matches_summary():
    files = export(CAMPAIGN)
    events = list(csv.DictReader(io.StringIO(files["paired-events.csv"])))
    assert len({(e["workload"], e["event_id"]) for e in events}) == len(events) == 1000
    segment_rows = list(csv.DictReader(io.StringIO(files["segments.csv"])))
    report = json.loads(files["report.json"])
    for profile, expected_outcomes in (("repeat_rich", (7, 593, 0)), ("lower_repeat", (16, 384, 0))):
        inc = report["profiles"][profile]["semantic_increment"]
        rows = [e for e in events if e["workload"] == profile]
        deltas = [int(e["exact_calls"]) - int(e["guarded_calls"]) for e in rows]
        assert (sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)) == expected_outcomes
        assert sum(deltas) == inc["calls_saved"]
        for dimension in ("request_type", "domain"):
            groups = [r for r in segment_rows if r["workload"] == profile and r["dimension"] == dimension]
            assert sum(int(group["requests"]) for group in groups) == len(rows)
            assert sum(int(group["guarded_increment"]) for group in groups) == inc["calls_saved"]
    false_accept, = [e for e in events if e["guarded_false_accept"] == "1"]
    assert false_accept["event_id"] == "async_05:challenge_added_requirements"
    assert [false_accept[k] for k in ("no_cache_calls", "exact_calls", "guarded_calls", "shadow_calls")] == ["3", "3", "0", "3"]


def test_workbench_links_and_serves_benchmark(tmp_path):
    with TestClient(create_demo_app(tmp_path, benchmark_directory=OUTPUT)) as client:
        assert 'href="/benchmark/"' in client.get("/").text
        assert "Cache benchmark" in client.get("/benchmark/").text
        assert client.get("/benchmark/report.json").json()["executions"] == 4000
        assert client.get("/benchmark/comparison.csv").status_code == 200
        assert client.get("/benchmark/segments.csv").status_code == 200
        assert len(client.get("/benchmark/paired-events.json").json()["lower_repeat"]["rows"]) == 400
    with TestClient(create_demo_app(tmp_path)) as client:
        assert "__BENCHMARK_LINK__" not in client.get("/").text
        assert client.get("/benchmark/").status_code == 404
