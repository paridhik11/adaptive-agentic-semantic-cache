"""Audit saved Phase A artifacts without running models or changing sealed results.

Historical outcomes describe each probe's designated seed. They are not gold
labels for every candidate retrieved from the growing cache, or for answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def text_key(text: str) -> str:
    return text.strip().casefold()


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((text_key(a), text_key(b))))


def audit_records(stream: list[dict], telemetry: list[dict], summary: dict,
                  training: list[dict], validation: list[dict]) -> dict[str, Any]:
    """Reconcile historical counts; expose candidate-label and split limitations.

    Reject broken joins or impossible cache provenance instead of silently
    dropping records. Off-seed candidates are never assigned the seed's label.
    """
    if len(stream) != len(telemetry):
        raise ValueError("Stream and telemetry lengths differ")
    ids = [e["id"] for e in stream]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate stream event IDs")
    by_id = {e["id"]: e for e in stream}
    indexed: set[str] = set()
    outcomes: Counter = Counter()
    paths: Counter = Counter()
    canonical: Counter = Counter()
    partitions: Counter = Counter()
    sources: dict[str, dict] = {}
    unresolved_ids = []
    canonical_fp_ids = []
    identical_fp_ids = []
    nonfinite_fields = []
    seed_reuses = 0
    nonexact_tp = nonexact_fp = 0
    for event, record in zip(stream, telemetry):
        if record["id"] != event["id"]:
            raise ValueError("Telemetry order/IDs differ from stream")
        for field in ("query", "source", "is_seed", "is_reuse_safe"):
            if record[field] != event[field]:
                raise ValueError(f"Telemetry field {field} differs for {event['id']}")
        if type(event["is_seed"]) is not bool or type(event["is_reuse_safe"]) is not bool:
            raise ValueError("Seed and label fields must be booleans")
        if record["final_decision"] not in ("REUSE", "BYPASS"):
            raise ValueError("Unknown final decision")
        if record["path"] not in ("CANONICAL_EXACT", "AUTO_REUSE", "LOCAL_EQUIVALENCE", "BYPASS"):
            raise ValueError("Unknown decision path")
        hit = record["final_decision"] == "REUSE"
        if hit and record["path"] == "BYPASS":
            raise ValueError("REUSE cannot have BYPASS path")
        if record["path"] in ("CANONICAL_EXACT", "AUTO_REUSE") and not hit:
            raise ValueError("Canonical and auto-reuse paths must accept reuse")
        candidate = record["matched_id"]
        if candidate is not None:
            if candidate not in indexed:
                raise ValueError("Candidate was not previously indexed")
            if record["matched_text"] != by_id[candidate]["query"]:
                raise ValueError("Candidate text and ID disagree")
        if hit and candidate is None:
            raise ValueError("Accepted hit has no candidate identity")
        for field in ("similarity", "stability_confidence", "local_score", "latency_ms"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid numeric field {field}")
            if not math.isfinite(value):
                nonfinite_fields.append({"id": event["id"], "field": field})
                if field != "similarity" or hit:
                    raise ValueError("Non-finite latency/score or accepted similarity")
        if record["latency_ms"] < 0:
            raise ValueError("Negative latency")
        paths[record["path"]] += 1
        expected = "SEED_INDEXED" if event["is_seed"] else (
            "TP" if hit and event["is_reuse_safe"] else
            "FP" if hit else "FN" if event["is_reuse_safe"] else "TN")
        if record["outcome"] != expected:
            raise ValueError("Historical outcome disagrees with designated-pair label")
        outcomes[expected] += 1
        if record["path"] == "CANONICAL_EXACT":
            canonical[expected] += 1
        source = sources.setdefault(event["source"], {
            "events": 0, "probes": 0, "hits": 0, "tp": 0, "fp": 0,
            "non_exact_hits": 0, "non_exact_fp": 0})
        source["events"] += 1
        if event["is_seed"]:
            seed_reuses += int(hit)
        else:
            source["probes"] += 1
            designated = event["matched_seed_id"]
            if designated not in indexed or not by_id[designated]["is_seed"]:
                raise ValueError("Probe's designated seed was not previously indexed")
            if by_id[designated]["pair_id"] != event["pair_id"]:
                raise ValueError("Designated seed belongs to a different pair")
            if hit:
                source["hits"] += 1
                source[expected.lower()] += 1
                nonexact = record["path"] != "CANONICAL_EXACT"
                if nonexact:
                    source["non_exact_hits"] += 1
                    source["non_exact_fp"] += int(expected == "FP")
                    nonexact_tp += int(expected == "TP")
                    nonexact_fp += int(expected == "FP")
                if candidate == designated:
                    partitions["designated_seed"] += 1
                elif event["query"] == record["matched_text"]:
                    partitions["identical_text_other_candidate"] += 1
                else:
                    partitions["unlabeled_other_candidate"] += 1
                    unresolved_ids.append(event["id"])
                if expected == "FP" and event["query"] == record["matched_text"]:
                    identical_fp_ids.append(event["id"])
                if expected == "FP" and not nonexact:
                    canonical_fp_ids.append(event["id"])
        if event["is_seed"] or not hit:
            indexed.add(event["id"])
    for source in sources.values():
        source["historical_irr"] = rate(source["fp"], source["hits"])
    hits = outcomes["TP"] + outcomes["FP"]
    nonexact_hits = nonexact_tp + nonexact_fp
    secondary = summary["secondary_endpoints"]
    primary = summary["primary_endpoint_non_exact_irr"]
    checks = {
        "n_queries": (summary["n_queries"], len(stream)),
        "total_hits": (secondary["total_hits"], hits),
        "total_tp": (secondary["total_tp"], outcomes["TP"]),
        "total_fp": (secondary["total_fp"], outcomes["FP"]),
        "total_fn": (secondary["total_fn"], outcomes["FN"]),
        "total_tn": (secondary["total_tn"], outcomes["TN"]),
        "exact_canonical_hits": (secondary["exact_canonical_hits"], canonical["TP"] + canonical["FP"]),
        "non_exact_hits": (primary["non_exact_hits"], nonexact_hits),
        "non_exact_tp": (primary["non_exact_tp"], nonexact_tp),
        "non_exact_fp": (primary["non_exact_fp"], nonexact_fp),
        "path_counts": (summary["path_resolution"]["counts"], dict(paths)),
    }
    for name, reported in summary["source_breakdown"].items():
        for field, value in reported.items():
            checks[f"source.{name}.{field}"] = (value, sources.get(name, {}).get(field))
    mismatches = {k: {"reported": a, "recomputed": b}
                  for k, (a, b) in checks.items() if a != b}
    train_pairs = {pair_key(r["query_a"], r["query_b"]) for r in training}
    val_pairs = {pair_key(r["query_a"], r["query_b"]) for r in validation}
    train_texts = {text_key(r[k]) for r in training for k in ("query_a", "query_b")}
    val_texts = {text_key(r[k]) for r in validation for k in ("query_a", "query_b")}
    probes = [e for e in stream if not e["is_seed"]]
    def pair_overlaps(keys: set) -> list[str]:
        return [e["id"] for e in probes if pair_key(
            by_id[e["matched_seed_id"]]["query"], e["query"]) in keys]
    return {
        "audit_kind": "offline_artifact_reconciliation_not_new_evaluation",
        "historical_counts_reconcile": not mismatches,
        "summary_mismatches": mismatches,
        "denominators": {"stream_events": len(stream), "seeds": len(stream) - len(probes), "probes": len(probes)},
        "historical_outcomes": dict(outcomes),
        "historical_non_exact": {"hits": nonexact_hits, "tp": nonexact_tp, "fp": nonexact_fp,
                                  "irr": rate(nonexact_fp, nonexact_hits)},
        "canonical_path_outcomes": dict(canonical),
        "source_counts": sources,
        "seed_decisions_reusing_existing_cache": seed_reuses,
        "accepted_candidate_partitions": dict(partitions),
        "unlabeled_candidate_event_ids": unresolved_ids,
        "historical_canonical_fp_event_ids": canonical_fp_ids,
        "historical_fp_with_identical_retrieved_text_ids": identical_fp_ids,
        "legacy_nonfinite_numeric_fields": nonfinite_fields,
        "overlap_check_normalization": "strip_and_casefold_unordered_pairs",
        "training_pair_overlap_probe_ids": pair_overlaps(train_pairs),
        "validation_pair_overlap_probe_ids": pair_overlaps(val_pairs),
        "training_text_overlap_event_ids": [e["id"] for e in stream if text_key(e["query"]) in train_texts],
        "validation_text_overlap_event_ids": [e["id"] for e in stream if text_key(e["query"]) in val_texts],
        "actual_answer_safety_measured": False,
        "candidate_aware_safety_certified": False,
    }


def audit_directory(root: Path) -> dict:
    data = root / "data"
    def read(name: str) -> Any:
        return json.loads((data / name).read_text(encoding="utf-8"))
    result = audit_records(read("raw/phaseA_sealed_test_stream.json"),
                           read("phaseA_sealed_evaluation_telemetry.json"),
                           read("phaseA_sealed_evaluation_summary.json"),
                           read("candidate_b_train_pairs.json"), read("phase8_tuning_val.json"))
    cfg = read("phaseA_frozen_config.json")
    raw = (data / "raw/phaseA_sealed_test_stream.json").read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    expected = cfg["sealed_stream"]["sha256"]
    variants = {"checked_out_bytes": raw, "lf": lf, "crlf": lf.replace(b"\n", b"\r\n")}
    result["stream_hash_audit"] = {
        "declared": expected,
        "variants": {k: hashlib.sha256(v).hexdigest() for k, v in variants.items()},
        "matching_variants": [k for k, v in variants.items() if hashlib.sha256(v).hexdigest() == expected],
        "historical_file_not_modified": True,
    }
    model_dir = root / cfg["cross_encoder"]["model_path"]
    result["checkpoint_weights_present"] = any(
        model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    result["sealed_inference_rerun"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional NEW audit JSON; never overwrite historical inputs")
    args = parser.parse_args()
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            parser.error("Audit output already exists; choose a new path")
        if destination.is_relative_to((args.repo_root / "data").resolve()):
            parser.error("Audit output must be outside historical data directory")
    try:
        result = audit_directory(args.repo_root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Artifact audit failed: {exc}\n")
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    if result["summary_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
