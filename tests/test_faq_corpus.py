import json
from pathlib import Path

import pytest

from src.benchmark.faq_corpus import (
    build_constructed_workload,
    parse_programming_faq,
    validate_group_splits,
)
from src.benchmark.schema import validate_workload


FAQ_FIXTURE = """General Questions
=================

How can I inspect a value?
--------------------------

Use the built-in repr function to inspect a value.

Core Language
=============

What is a generator?
--------------------

A generator yields values lazily.
"""


def test_parses_source_questions_and_preserves_topic_split(tmp_path):
    path = tmp_path / "programming.rst"
    path.write_text(FAQ_FIXTURE, encoding="utf-8")

    corpus = parse_programming_faq(path)
    checked = validate_group_splits(corpus)

    assert len(corpus["items"]) == 2
    assert [row["split"] for row in corpus["items"]] == ["train", "train"]
    assert corpus["items"][1]["reference_answer"] == "A generator yields values lazily."
    assert checked["question_count"] == 2
    assert all(row["answer_reuse_gold"] is None for row in corpus["items"])


def test_constructed_repeat_workload_is_valid_and_explicitly_not_natural(tmp_path):
    path = tmp_path / "programming.rst"
    path.write_text(FAQ_FIXTURE, encoding="utf-8")
    corpus = parse_programming_faq(path)

    workload = validate_workload(build_constructed_workload(corpus))

    assert workload["nature"] == "synthetic_fixture"
    assert len(workload["requests"]) == 4
    assert [row["constructed_repeat"] for row in workload["requests"]] == [False, True, False, True]
    assert workload["requests"][0]["request"] == workload["requests"][1]["request"]


def test_split_validation_rejects_topic_and_question_leakage(tmp_path):
    path = tmp_path / "programming.rst"
    path.write_text(FAQ_FIXTURE, encoding="utf-8")
    corpus = parse_programming_faq(path)
    corpus["items"][1]["group_id"] = corpus["items"][0]["group_id"]
    corpus["items"][1]["split"] = "dev"

    with pytest.raises(ValueError, match="topic group crosses splits"):
        validate_group_splits(corpus)


def test_official_manifest_has_pinned_hashes_and_complete_group_mapping():
    manifest = json.loads(Path("examples/benchmark/python_faq.sources.json").read_text(encoding="utf-8"))

    assert manifest["revision"] == "0fb18b02c8ad56299d6a2910be0bab8ad601ef24"
    assert {row["path"] for row in manifest["files"]} == {"LICENSE", "Doc/faq/programming.rst"}
    assert {section for sections in manifest["group_split"].values() for section in sections} == {
        "General Questions", "Core Language", "Numbers and strings", "Performance",
        "Sequences (Tuples/Lists)", "Objects", "Modules",
    }


def test_authored_adversarial_workload_only_requests_misses_or_bypasses():
    workload = json.loads(Path("examples/benchmark/python_faq_adversarial.json").read_text(encoding="utf-8"))

    assert workload["nature"] == "authored_stress_cases_not_accuracy_labels"
    assert len(workload["cases"]) == 12
    assert len({row["case_id"] for row in workload["cases"]}) == 12
    assert all(row["expected_reuse"] is False for row in workload["cases"])
    assert {row["category"] for row in workload["cases"]} >= {
        "entity", "negation", "quantity", "unit", "direction", "code_input",
        "version", "temporal", "requested_depth", "context", "conversation_context",
        "generative_request",
    }
