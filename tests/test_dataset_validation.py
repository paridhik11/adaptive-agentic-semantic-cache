"""Validation tests for Phase 0 datasets, schemas, and dataset quality."""

import json
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
STABILITY_DATASET_PATH = DATA_DIR / "query_stability_benchmark.json"
REUSE_PAIR_DATASET_PATH = DATA_DIR / "query_pair_reuse_benchmark.json"

VALID_DOMAINS = {
    "computer_science",
    "mathematics",
    "history_geography",
    "science_medicine",
    "finance_economics",
    "realtime_news_weather",
    "system_operations",
    "legal_compliance",
}

VALID_STABILITY_LABELS = {"STABLE", "DYNAMIC", "CONDITIONALLY_STABLE"}
VALID_TEMPORAL_SENSITIVITY = {"EVERGREEN", "SLOW_DECAY", "FAST_DECAY", "EVENT_DRIVEN"}
VALID_SIMILARITY_LEVELS = {"HIGH", "MEDIUM", "LOW"}
VALID_TAXONOMY_CLASSES = {
    "SAFE_EQUIVALENT",
    "UNSAFE_DIFFERENT_INTENT",
    "UNSAFE_SCOPE_MISMATCH",
    "UNSAFE_CONTEXT_MISMATCH",
    "UNSAFE_DYNAMIC_TEMPORAL",
    "UNSAFE_DIFFERENT_TOPIC",
}


def test_stability_benchmark_structure_and_counts():
    """Verify stability benchmark exists, is valid JSON, meets count requirement, and conforms to schema.

    Accepts both the legacy bare-array format AND the Phase 1.1 B.1 metadata-wrapped format:
      Bare array:    [ { "id": "STAB-001", ... }, ... ]
      Wrapped:       { "dataset_role": "...", "metadata": {...}, "queries": [ ... ] }

    The heldout and final_test datasets are NOT touched here; their format tests are separate.
    """
    assert STABILITY_DATASET_PATH.exists(), f"Missing file: {STABILITY_DATASET_PATH}"
    
    with open(STABILITY_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Normalise: support both bare array and metadata-wrapped object
    if isinstance(raw, list):
        data = raw
        dataset_role = None
    elif isinstance(raw, dict):
        # Metadata-wrapped format introduced in Phase 1.1 Sub-stage B
        assert "queries" in raw, (
            "Metadata-wrapped stability dataset must contain a 'queries' key. "
            f"Found keys: {list(raw.keys())}"
        )
        data = raw["queries"]
        dataset_role = raw.get("dataset_role")
        assert isinstance(data, list), "'queries' must be a JSON array"
        # If dataset_role is present, it must be the canonical dev/diagnostic marker
        if dataset_role is not None:
            assert dataset_role == "development_diagnostic_do_not_use_as_final_eval", (
                f"Unexpected dataset_role value: {dataset_role!r}"
            )
    else:
        raise AssertionError(
            f"Stability dataset must be a JSON array or metadata-wrapped object, got {type(raw).__name__}"
        )
    
    assert len(data) >= 150, f"Expected at least 150 queries, found {len(data)}"
    assert len(data) <= 200, f"Expected at most 200 queries, found {len(data)}"
    
    seen_ids = set()
    domain_counts = {}
    label_counts = {}
    has_historical_stable = False
    
    for item in data:
        required_keys = {
            "id",
            "query",
            "domain",
            "stability_label",
            "temporal_sensitivity",
            "has_explicit_temporal_marker",
            "implicit_dependencies",
            "rationale",
        }
        assert required_keys.issubset(item.keys()), f"Missing keys in {item.get('id')}: {required_keys - set(item.keys())}"
        
        # ID validation
        item_id = item["id"]
        assert item_id.startswith("STAB-"), f"Invalid ID format: {item_id}"
        assert item_id not in seen_ids, f"Duplicate ID detected: {item_id}"
        seen_ids.add(item_id)
        
        # Query validation
        assert isinstance(item["query"], str) and len(item["query"].strip()) >= 5
        
        # Domain validation
        domain = item["domain"]
        assert domain in VALID_DOMAINS, f"Invalid domain '{domain}' in {item_id}"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Stability label validation
        label = item["stability_label"]
        assert label in VALID_STABILITY_LABELS, f"Invalid stability label '{label}' in {item_id}"
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check for historical stable nuance (has explicit temporal marker but is STABLE)
        if item["has_explicit_temporal_marker"] and label == "STABLE":
            has_historical_stable = True
        
        # Temporal sensitivity
        temp_sens = item["temporal_sensitivity"]
        assert temp_sens in VALID_TEMPORAL_SENSITIVITY, f"Invalid temporal sensitivity '{temp_sens}' in {item_id}"
        
        # Boolean marker
        assert isinstance(item["has_explicit_temporal_marker"], bool), f"has_explicit_temporal_marker must be bool in {item_id}"
        
        # Implicit dependencies
        assert isinstance(item["implicit_dependencies"], list), f"implicit_dependencies must be list in {item_id}"
        
        # Rationale
        assert isinstance(item["rationale"], str) and len(item["rationale"].strip()) >= 10, f"Rationale too short in {item_id}"

    # Quality check: All domains represented
    assert len(domain_counts) == len(VALID_DOMAINS), "All 8 domains must be represented in stability dataset"
    
    # Quality check: Stability distributions
    assert label_counts.get("STABLE", 0) >= 30, "Must have substantial STABLE queries"
    assert label_counts.get("DYNAMIC", 0) >= 30, "Must have substantial DYNAMIC queries"
    assert label_counts.get("CONDITIONALLY_STABLE", 0) >= 10, "Must have CONDITIONALLY_STABLE queries"
    
    # Quality check: Nuanced temporal cases exist
    assert has_historical_stable, "Should contain nuanced stable queries with explicit temporal/historical anchors"


def test_query_pair_reuse_benchmark_structure_and_counts():
    """Verify query-pair benchmark schema, count requirements, and realistic distributions."""
    assert REUSE_PAIR_DATASET_PATH.exists(), f"Missing file: {REUSE_PAIR_DATASET_PATH}"
    
    with open(REUSE_PAIR_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert isinstance(data, list), "Query pair dataset must be a JSON array"
    assert len(data) >= 100, f"Expected at least 100 pairs, found {len(data)}"
    
    seen_ids = set()
    safe_count = 0
    unsafe_count = 0
    similarity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    high_sim_unsafe_count = 0
    high_sim_safe_count = 0
    med_sim_safe_count = 0
    med_sim_unsafe_count = 0
    low_sim_unsafe_count = 0
    temporal_count = 0
    
    for item in data:
        required_keys = {
            "id",
            "query_a",
            "query_b",
            "domain",
            "semantic_similarity_level",
            "is_reuse_safe",
            "taxonomy_class",
            "rejection_reason",
            "rationale",
        }
        assert required_keys.issubset(item.keys()), f"Missing keys in {item.get('id')}: {required_keys - set(item.keys())}"
        
        # ID validation
        item_id = item["id"]
        assert item_id.startswith("PAIR-"), f"Invalid ID format: {item_id}"
        assert item_id not in seen_ids, f"Duplicate ID detected: {item_id}"
        seen_ids.add(item_id)
        
        # Queries validation
        assert isinstance(item["query_a"], str) and len(item["query_a"].strip()) >= 5
        assert isinstance(item["query_b"], str) and len(item["query_b"].strip()) >= 5
        
        # Domain validation
        assert item["domain"] in VALID_DOMAINS, f"Invalid domain in {item_id}"
        
        # Similarity level
        sim = item["semantic_similarity_level"]
        assert sim in VALID_SIMILARITY_LEVELS, f"Invalid similarity level '{sim}' in {item_id}"
        similarity_counts[sim] += 1
        
        # Reuse safety
        is_safe = item["is_reuse_safe"]
        assert isinstance(is_safe, bool), f"is_reuse_safe must be boolean in {item_id}"
        
        if is_safe:
            safe_count += 1
            assert item["rejection_reason"] is None, f"Safe pair {item_id} must have null rejection_reason"
            if sim == "HIGH":
                high_sim_safe_count += 1
            elif sim == "MEDIUM":
                med_sim_safe_count += 1
        else:
            unsafe_count += 1
            assert isinstance(item["rejection_reason"], str) and len(item["rejection_reason"].strip()) >= 10, (
                f"Unsafe pair {item_id} must have meaningful rejection_reason"
            )
            if sim == "HIGH":
                high_sim_unsafe_count += 1
            elif sim == "MEDIUM":
                med_sim_unsafe_count += 1
            elif sim == "LOW":
                low_sim_unsafe_count += 1
        
        # Taxonomy class
        tax = item["taxonomy_class"]
        assert tax in VALID_TAXONOMY_CLASSES, f"Invalid taxonomy class '{tax}' in {item_id}"
        if tax == "UNSAFE_DYNAMIC_TEMPORAL":
            temporal_count += 1
        
        # Rationale
        assert isinstance(item["rationale"], str) and len(item["rationale"].strip()) >= 10, f"Rationale too short in {item_id}"

    # Quality check: Balance between safe and unsafe
    assert safe_count >= 30, f"Expected at least 30 safe pairs, got {safe_count}"
    assert unsafe_count >= 30, f"Expected at least 30 unsafe pairs, got {unsafe_count}"
    
    # Quality check: Realistic distribution across HIGH, MEDIUM, and LOW
    assert similarity_counts["HIGH"] >= 30, "Must have substantial HIGH similarity pairs"
    assert similarity_counts["MEDIUM"] >= 25, "Must have substantial MEDIUM similarity pairs"
    assert similarity_counts["LOW"] >= 10, "Must have LOW similarity rejection pairs"
    
    # Quality check: High-similarity deceptive traps exist
    assert high_sim_unsafe_count >= 15, f"Expected at least 15 HIGH similarity UNSAFE pairs, found {high_sim_unsafe_count}"
    assert high_sim_safe_count >= 20, f"Expected at least 20 HIGH similarity SAFE pairs, found {high_sim_safe_count}"
    
    # Quality check: Medium similarity has both safe and unsafe examples
    assert med_sim_safe_count >= 10, f"Expected at least 10 MEDIUM similarity SAFE pairs, found {med_sim_safe_count}"
    assert med_sim_unsafe_count >= 10, f"Expected at least 10 MEDIUM similarity UNSAFE pairs, found {med_sim_unsafe_count}"
    
    # Quality check: Dynamic/temporal cases represented
    assert temporal_count >= 3, f"Expected at least 3 temporal dynamic unsafe pairs, found {temporal_count}"


def test_heldout_stability_benchmark_structure_and_counts():
    """Verify held-out stability benchmark exists, is valid JSON, and conforms to schema."""
    heldout_path = DATA_DIR / "query_stability_benchmark_heldout.json"
    assert heldout_path.exists(), f"Missing file: {heldout_path}"

    with open(heldout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Held-out dataset must be a JSON array"
    assert len(data) >= 50, f"Expected at least 50 held-out queries, found {len(data)}"

    seen_ids = set()
    domain_counts = {}
    label_counts = {}

    for item in data:
        required_keys = {
            "id",
            "query",
            "domain",
            "stability_label",
            "temporal_sensitivity",
            "has_explicit_temporal_marker",
            "implicit_dependencies",
            "rationale",
        }
        assert required_keys.issubset(item.keys()), f"Missing keys in {item.get('id')}"

        item_id = item["id"]
        assert item_id.startswith("STAB-"), f"Invalid ID format: {item_id}"
        assert item_id not in seen_ids, f"Duplicate ID detected: {item_id}"
        seen_ids.add(item_id)

        assert isinstance(item["query"], str) and len(item["query"].strip()) >= 5
        domain = item["domain"]
        assert domain in VALID_DOMAINS, f"Invalid domain '{domain}' in {item_id}"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        label = item["stability_label"]
        assert label in VALID_STABILITY_LABELS, f"Invalid stability label '{label}' in {item_id}"
        label_counts[label] = label_counts.get(label, 0) + 1

        temp_sens = item["temporal_sensitivity"]
        assert temp_sens in VALID_TEMPORAL_SENSITIVITY, f"Invalid temporal sensitivity in {item_id}"
        assert isinstance(item["has_explicit_temporal_marker"], bool)
        assert isinstance(item["implicit_dependencies"], list)
        assert isinstance(item["rationale"], str) and len(item["rationale"].strip()) >= 10

    # Ensure all domains represented
    assert len(domain_counts) == len(VALID_DOMAINS), "All 8 domains must be represented in held-out dataset"
    assert label_counts.get("STABLE", 0) >= 15
    assert label_counts.get("DYNAMIC", 0) >= 15


def test_final_test_stability_benchmark_structure_and_counts():
    """Verify final test stability benchmark exists, is valid JSON, and conforms to schema."""
    final_test_path = DATA_DIR / "query_stability_benchmark_final_test.json"
    assert final_test_path.exists(), f"Missing file: {final_test_path}"

    with open(final_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Final test dataset must be a JSON array"
    assert len(data) >= 50, f"Expected at least 50 final test queries, found {len(data)}"

    seen_ids = set()
    domain_counts = {}
    label_counts = {}

    for item in data:
        required_keys = {
            "id",
            "query",
            "domain",
            "stability_label",
            "temporal_sensitivity",
            "has_explicit_temporal_marker",
            "implicit_dependencies",
            "rationale",
        }
        assert required_keys.issubset(item.keys()), f"Missing keys in {item.get('id')}"

        item_id = item["id"]
        assert item_id.startswith("STAB-"), f"Invalid ID format: {item_id}"
        assert item_id not in seen_ids, f"Duplicate ID detected: {item_id}"
        seen_ids.add(item_id)

        assert isinstance(item["query"], str) and len(item["query"].strip()) >= 5
        domain = item["domain"]
        assert domain in VALID_DOMAINS, f"Invalid domain '{domain}' in {item_id}"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        label = item["stability_label"]
        assert label in VALID_STABILITY_LABELS, f"Invalid stability label '{label}' in {item_id}"
        label_counts[label] = label_counts.get(label, 0) + 1

        temp_sens = item["temporal_sensitivity"]
        assert temp_sens in VALID_TEMPORAL_SENSITIVITY, f"Invalid temporal sensitivity in {item_id}"
        assert isinstance(item["has_explicit_temporal_marker"], bool)
        assert isinstance(item["implicit_dependencies"], list)
        assert isinstance(item["rationale"], str) and len(item["rationale"].strip()) >= 10

    # Ensure all domains represented
    assert len(domain_counts) == len(VALID_DOMAINS), "All 8 domains must be represented in final test dataset"
    assert label_counts.get("STABLE", 0) >= 15
    assert label_counts.get("DYNAMIC", 0) >= 15


def test_human_credibility_dataset_structure_and_counts():
    """Verify human credibility audit dataset exists, is valid metadata-wrapped JSON, and conforms to expected structure."""
    human_path = DATA_DIR / "query_stability_human_credibility.json"
    assert human_path.exists(), f"Missing file: {human_path}"

    with open(human_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Human credibility dataset must be a metadata-wrapped JSON object"
    assert "metadata" in data, "Missing 'metadata' key in human credibility dataset"
    assert "queries" in data, "Missing 'queries' key in human credibility dataset"
    assert isinstance(data["queries"], list), "'queries' must be a list"
    assert len(data["queries"]) >= 50, f"Expected at least 50 queries, found {len(data['queries'])}"

    seen_ids = set()
    for item in data["queries"]:
        assert isinstance(item, dict)
        assert "id" in item and item["id"].startswith("HUMAN-")
        assert item["id"] not in seen_ids
        seen_ids.add(item["id"])
        assert "query" in item and isinstance(item["query"], str) and len(item["query"].strip()) >= 3
        assert "expected_label" in item and item["expected_label"] in ("STABLE", "DYNAMIC", "CONDITIONALLY_STABLE")



