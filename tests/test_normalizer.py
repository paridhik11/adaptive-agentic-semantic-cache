"""Unit tests for Phase 8 normalizer."""

import pytest
from src.cache.normalizer import normalize_query, CanonicalCache


def test_what_is_and_question_mark_fold():
    """Unit test: 'what is X' and 'X?' fold together."""
    canonical_1 = normalize_query("what is Python")
    canonical_2 = normalize_query("Python?")
    canonical_3 = normalize_query("What is Python?")
    canonical_4 = normalize_query("what's Python")
    canonical_5 = normalize_query("define Python")

    assert canonical_1 == "python"
    assert canonical_2 == "python"
    assert canonical_3 == "python"
    assert canonical_4 == "python"
    assert canonical_5 == "python"
    assert canonical_1 == canonical_2 == canonical_3 == canonical_4 == canonical_5


def test_what_is_vs_how_does_work_do_not_fold():
    """Unit test: 'what is X' vs 'how does X work' do NOT fold."""
    what_is = normalize_query("what is Python")
    how_work = normalize_query("how does Python work")
    how_work_q = normalize_query("how does Python work?")

    assert what_is == "python"
    assert how_work == "how does python work"
    assert how_work_q == "how does python work"
    assert what_is != how_work


def test_leading_articles_stripped_after_frame():
    """Definitional frames strip 'a', 'an', 'the'."""
    assert normalize_query("What is a neural network?") == "neural network"
    assert normalize_query("What is the transformer architecture?") == "transformer architecture"
    assert normalize_query("Explain an autoencoder") == "autoencoder"


def test_whitespace_and_punctuation_handling():
    """Normalizes extraneous spaces and edge punctuation."""
    assert normalize_query("   what is    recursion?!!  ") == "recursion"
    assert normalize_query('"what is recursion?"') == "recursion"
    assert normalize_query("") == ""


def test_canonical_cache():
    """CanonicalCache stores and retrieves entries by canonical key without embedding."""
    cache = CanonicalCache()
    cache.put("What is Python?", {"response": "A high-level programming language."})

    hit_1 = cache.get("Python?")
    hit_2 = cache.get("what is python")
    miss = cache.get("How does Python work?")

    assert hit_1 is not None
    assert hit_1["response"] == "A high-level programming language."
    assert hit_2 is not None
    assert hit_2["response"] == "A high-level programming language."
    assert miss is None
