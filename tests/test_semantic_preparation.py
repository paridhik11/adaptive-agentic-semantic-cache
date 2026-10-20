"""Prevent misleading probe/model and prospective sample planning behavior."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.policy.local_models import LocalPairScorer


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scorer(logit=0.0, limit=10):
    obj = LocalPairScorer.__new__(LocalPairScorer)
    obj.model = SimpleNamespace(tokenizer=Tokenizer(), max_length=limit,
                               predict=lambda *args, **kwargs: [logit])
    return obj


class Tokenizer:
    model_max_length = 20
    def __call__(self, first, second, **kwargs):
        assert kwargs == {"truncation": False}
        return {"input_ids": list(range(len((first+" "+second).split())))}


def test_pair_window_overflow_abstains_before_model_prediction():
    obj = scorer(limit=2)
    obj.model.predict = lambda *a, **kw: pytest.fail("truncated inference must not run")
    with pytest.raises(ValueError, match="window"):
        obj.score("one two", "three")


@pytest.mark.parametrize("logit,expected", [(0, .5), (1000, 1.0), (-1000, 0.0)])
def test_pair_logit_conversion_is_explicit_and_numerically_stable(logit, expected):
    assert scorer(logit).score("first", "second")["pair_score"] == expected


def test_nonfinite_pair_predictions_cannot_appear_as_valid_scores():
    with pytest.raises(ValueError, match="nonfinite"):
        scorer(float("nan")).score("first", "second")


def test_mutable_pair_revision_rejected_before_any_model_import():
    with pytest.raises(ValueError, match="immutable"):
        LocalPairScorer(revision="main")


def test_exact_group_gate_distinguishes_one_error_in_sixty_from_two_in_125():
    module = script("plan_semantic_review")
    assert module.lower_bound(60, 1) < .95
    assert module.lower_bound(125, 2) > .95
    assert module.lower_bound(125, 3) < .95


def test_expected_sample_size_is_not_mistaken_for_sufficiency_assurance():
    result = script("plan_semantic_review").plan(125, .3)
    assert result["positive_groups_for_expected_count"] == 417
    assert result["positive_groups_for_95pct_sufficiency_under_assumption"] > 417
    assert result["evidence_type"] == "hypothetical_planning_arithmetic"
