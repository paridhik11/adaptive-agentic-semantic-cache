"""Regression tests for candidate-bound labels and artifact integrity."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "phase_a_audit", Path(__file__).resolve().parents[1] / "scripts/audit_phase_a_artifacts.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    stream = [
        {"id": "s1", "query": "Mars", "source": "toy", "is_seed": True,
         "is_reuse_safe": False, "pair_id": "p1", "matched_seed_id": None},
        {"id": "s2", "query": "Venus", "source": "toy", "is_seed": True,
         "is_reuse_safe": False, "pair_id": "p2", "matched_seed_id": None},
        {"id": "q1", "query": "Venus", "source": "toy", "is_seed": False,
         "is_reuse_safe": False, "pair_id": "p1", "matched_seed_id": "s1"},
    ]
    telemetry = []
    for event in stream:
        probe = not event["is_seed"]
        telemetry.append({**event, "final_decision": "REUSE" if probe else "BYPASS",
                          "path": "CANONICAL_EXACT" if probe else "BYPASS",
                          "outcome": "FP" if probe else "SEED_INDEXED",
                          "matched_id": "s2" if probe else None,
                          "matched_text": "Venus" if probe else "",
                          "similarity": 1.0, "stability_confidence": 0.9,
                          "local_score": 0.0, "latency_ms": 0.1})
    summary = {"n_queries": 3,
               "secondary_endpoints": {"total_hits": 1, "total_tp": 0, "total_fp": 1,
                                       "total_fn": 0, "total_tn": 0, "exact_canonical_hits": 1},
               "primary_endpoint_non_exact_irr": {"non_exact_hits": 0, "non_exact_tp": 0, "non_exact_fp": 0},
               "path_resolution": {"counts": {"BYPASS": 2, "CANONICAL_EXACT": 1}},
               "source_breakdown": {"toy": {"hits": 1, "tp": 0, "fp": 1, "non_exact_hits": 0, "non_exact_fp": 0}}}
    return stream, telemetry, summary


def test_identical_other_candidate_does_not_inherit_negative_pair_label():
    stream, telemetry, summary = fixture()
    originals = copy.deepcopy((stream, telemetry, summary))
    result = audit.audit_records(stream, telemetry, summary, [], [])
    assert result["historical_counts_reconcile"]
    assert result["historical_outcomes"]["FP"] == 1
    assert result["accepted_candidate_partitions"] == {"identical_text_other_candidate": 1}
    assert result["historical_fp_with_identical_retrieved_text_ids"] == ["q1"]
    assert result["historical_non_exact"]["irr"] is None
    assert not result["actual_answer_safety_measured"]
    assert (stream, telemetry, summary) == originals


def test_nonidentical_other_candidate_is_unlabeled_not_reclassified():
    stream, telemetry, summary = fixture()
    stream[-1]["query"] = telemetry[-1]["query"] = "Jupiter"
    result = audit.audit_records(stream, telemetry, summary, [], [])
    assert result["unlabeled_candidate_event_ids"] == ["q1"]
    assert result["accepted_candidate_partitions"] == {"unlabeled_other_candidate": 1}
    assert not result["candidate_aware_safety_certified"]


def test_designated_seed_has_pair_label_provenance():
    stream, telemetry, summary = fixture()
    telemetry[-1]["matched_id"], telemetry[-1]["matched_text"] = "s1", "Mars"
    result = audit.audit_records(stream, telemetry, summary, [], [])
    assert result["accepted_candidate_partitions"] == {"designated_seed": 1}


@pytest.mark.parametrize("change", [
    lambda s, t: s[1].update(id="s1"),
    lambda s, t: t.reverse(),
    lambda s, t: t[-1].update(matched_id="q1"),
    lambda s, t: t[-1].update(matched_id="missing"),
    lambda s, t: t[-1].update(matched_id=None),
    lambda s, t: t[-1].update(matched_text="tampered"),
    lambda s, t: t[-1].update(outcome="TP"),
    lambda s, t: t[-1].update(final_decision="OTHER"),
    lambda s, t: t[-1].update(latency_ms=float("nan")),
    lambda s, t: t[-1].update(latency_ms=-1),
    lambda s, t: t[-1].update(similarity=float("inf")),
    lambda s, t: s[-1].update(matched_seed_id="q1"),
    lambda s, t: s[-1].update(pair_id="wrong"),
    lambda s, t: (s[-1].update(is_seed=0), t[-1].update(is_seed=0)),
])
def test_rejects_invalid_provenance_or_records(change):
    stream, telemetry, summary = fixture()
    change(stream, telemetry)
    with pytest.raises(ValueError):
        audit.audit_records(stream, telemetry, summary, [], [])


def test_missing_record_rejected():
    stream, telemetry, summary = fixture()
    with pytest.raises(ValueError):
        audit.audit_records(stream, telemetry[:-1], summary, [], [])


def test_summary_mismatch_is_reported():
    stream, telemetry, summary = fixture()
    summary["secondary_endpoints"]["total_fp"] = 0
    result = audit.audit_records(stream, telemetry, summary, [], [])
    assert result["summary_mismatches"]["total_fp"] == {"reported": 0, "recomputed": 1}
    assert not result["historical_counts_reconcile"]


def test_overlap_detects_reversed_casefolded_pairs_and_shared_text():
    stream, telemetry, summary = fixture()
    result = audit.audit_records(stream, telemetry, summary,
                                 [{"query_a": " VENUS ", "query_b": "mars"}],
                                 [{"query_a": "Elsewhere", "query_b": "Mars"}])
    assert result["training_pair_overlap_probe_ids"] == ["q1"]
    assert result["training_text_overlap_event_ids"] == ["s1", "s2", "q1"]
    assert result["validation_pair_overlap_probe_ids"] == []
    assert result["validation_text_overlap_event_ids"] == ["s1"]


def test_legacy_empty_lookup_infinity_is_flagged_and_output_is_strict_json():
    stream, telemetry, summary = fixture()
    telemetry[0]["similarity"] = float("-inf")
    result = audit.audit_records(stream, telemetry, summary, [], [])
    assert result["legacy_nonfinite_numeric_fields"] == [{"id": "s1", "field": "similarity"}]
    json.dumps(result, allow_nan=False)


def test_no_hits_irr_is_undefined():
    assert audit.rate(0, 0) is None
    assert audit.rate(0, 10) == 0


def test_repository_artifacts_reconcile_without_inference():
    root = Path(__file__).resolve().parents[1]
    result = audit.audit_directory(root)
    assert result["historical_counts_reconcile"]
    assert result["canonical_path_outcomes"] == {"TP": 6, "FP": 5, "SEED_INDEXED": 19}
    assert result["accepted_candidate_partitions"] == {
        "designated_seed": 67, "identical_text_other_candidate": 10, "unlabeled_other_candidate": 9}
    assert result["stream_hash_audit"]["matching_variants"]
    assert result["sealed_inference_rerun"] is False
