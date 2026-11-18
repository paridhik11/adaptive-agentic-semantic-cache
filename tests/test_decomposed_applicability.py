"""Functional boundaries, not model-quality labels."""
from types import SimpleNamespace

import pytest

from src.policy.decomposed_applicability import (DecomposedDiagnosticPolicy, EvidenceIndex,
    LocalEntailmentScorer, SourceSpan, requirement_guard, split_claims)
from src.policy.encoders import DenseEncoder

CONTEXT = "tuple(items) converts the items to a tuple."
ORIGINAL = "How do I convert a list to a tuple?"
INCOMING = "How can I make a tuple from a list?"


class Encoder:
    identity = "constant-fixture-only"
    def __init__(self):
        self.inputs = []
    def encode(self, text):
        self.inputs.append(text)
        return [1., 0.]


class Scorer:
    identity = "fixture-nli"
    def __init__(self):
        self.inputs = []
    def score(self, premise, hypothesis):
        self.inputs.append((premise, hypothesis))
        return {"entailment":.999,"neutral":.0005,"contradiction":.0005}


def request(query=ORIGINAL, context=CONTEXT):
    return {"model":"fixture","temperature":0,"messages":[
        {"role":"system","content":context},{"role":"user","content":query}]}


def response(text="Use tuple(items) to convert the list to a tuple."):
    return {"model":"fixture","choices":[{"finish_reason":"stop",
        "message":{"role":"assistant","content":text}}]}


def setup(context=CONTEXT, scorer=None):
    encoder = Encoder()
    index = EvidenceIndex(context=context,spans=[SourceSpan(0,len(context))],encoder=encoder)
    return DecomposedDiagnosticPolicy(evidence_index=index,scorer=scorer or Scorer()),encoder


@pytest.mark.parametrize("incoming", [
    "How do I convert a tuple to a list?",
    "How do I recursively convert a list to a tuple?",
    "How do I convert a list to a tuple without allocating an object?",
    "Why should I not convert a list to a tuple?",
    "How do I convert a list to a tuple in Python 3.12?",
])
def test_changed_requirements_abstain_before_embedding_or_nli(incoming):
    policy,encoder=setup()
    result=policy.assess(request(incoming),request(),response())
    assert not result.recommended_reuse
    assert result.work['embedding_attempts']==result.work['nli_attempts']==0
    assert len(encoder.inputs)==1  # source initialization only


def test_generic_direction_guard_is_not_a_question_allowlist():
    first="How do I convert red cuboids into blue spheres?"
    assert requirement_guard(first,"How can I make blue spheres from red cuboids?") is None
    assert requirement_guard(first,"How do I convert blue spheres into red cuboids?") == "conversion_direction_or_entity_changed"


def test_source_selection_cannot_be_steered_by_incoming_or_wrong_answer():
    policy,encoder=setup()
    result=policy.assess(request(INCOMING),request(),response("Use list(items) to make a tuple."))
    assert result.reason=="required_source_api_missing"
    assert encoder.inputs[-1]==ORIGINAL
    assert result.work['nli_attempts']==0


def test_missing_formula_constant_rejects_incomplete_answer_before_nli():
    context="F = C * 9 / 5 + 32 converts Celsius C to Fahrenheit F."
    policy,_=setup(context)
    result=policy.assess(request("How can I convert Celsius to Fahrenheit?",context),
        request("How do I convert Celsius to Fahrenheit?",context),response("Multiply C by 9 / 5."))
    assert result.reason=="required_source_quantity_missing"


def test_every_check_must_pass_and_actual_answer_is_bound():
    policy,_=setup()
    result=policy.assess(request(INCOMING),request(),response())
    assert result.recommended_reuse
    assert set(result.scores)=={"original_to_incoming","incoming_to_original","evidence_coverage","claim_support_0"}
    assert result.work['nli_attempts']==4
    changed=policy.assess(request(INCOMING),request(),response("Use tuple(items). It returns a list."))
    assert changed.binding['candidate_response_sha256']!=result.binding['candidate_response_sha256']


def test_neutral_entailment_abstains_without_running_later_checks():
    class Neutral(Scorer):
        def score(self,*args):
            return {"entailment":.01,"neutral":.98,"contradiction":.01}
    policy,_=setup(scorer=Neutral())
    result=policy.assess(request(INCOMING),request(),response())
    assert result.reason=="original_to_incoming_not_entailed"
    assert result.work['nli_attempts']==1
    assert "claim_support_0" not in result.scores


def test_invalid_score_distribution_fails_closed():
    class Invalid(Scorer):
        def score(self,*args):
            return {"entailment":.99,"neutral":.99,"contradiction":.99}
    policy,_=setup(scorer=Invalid())
    assert policy.assess(request(INCOMING),request(),response()).reason=="local_check_failed"


def test_source_offsets_and_ambiguity_are_enforced():
    with pytest.raises(ValueError):
        EvidenceIndex(context=CONTEXT,spans=[SourceSpan(0,len(CONTEXT)+1)],encoder=Encoder())
    index=EvidenceIndex(context="First fact. Second fact.",
        spans=[SourceSpan(0,11),SourceSpan(12,24)],encoder=Encoder())
    assert index.select(ORIGINAL) is None


def test_scope_change_and_incomplete_answer_skip_local_checks():
    policy,_=setup()
    result=policy.assess({**request(INCOMING),"temperature":.1},request(),response())
    assert result.reason=="request_scope_changed" and result.work['embedding_attempts']==0
    incomplete=response();incomplete['choices'][0]['finish_reason']='length'
    assert policy.assess(request(INCOMING),request(),incomplete).reason=="incomplete_or_unsupported_answer"


def test_positive_diagnostic_cannot_serve_or_call_remote_provider():
    policy,_=setup()
    result=policy(request(INCOMING),request(),response(),binding={},
        provider=lambda _:pytest.fail("remote provider must not run"))
    assert result.approved is False and result.local_policy_recommendation is True
    assert result.local_verifier_attempts==5


def test_claim_split_preserves_api_names_and_decimal_values():
    assert split_claims("Use os.path.basename(path). The value is 273.15.")==[
        "Use os.path.basename(path).","The value is 273.15."]


def test_dense_overflow_is_rejected_before_inference():
    encoder=DenseEncoder.__new__(DenseEncoder)
    encoder.model=SimpleNamespace(max_seq_length=2,
        tokenizer=lambda *a,**k:{"input_ids":[1,2,3]},
        encode=lambda *a,**k:pytest.fail("truncated inference must not run"))
    with pytest.raises(ValueError,match="window"):
        encoder.encode("too long")


def test_nli_overflow_is_rejected_before_inference():
    class Tokenizer:
        model_max_length=512
        def __call__(self,*args,**kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids':SimpleNamespace(shape=(1,513))}
    scorer=LocalEntailmentScorer.__new__(LocalEntailmentScorer)
    scorer.tokenizer=Tokenizer()
    with pytest.raises(ValueError,match="too long"):
        scorer.score("long premise","long hypothesis")


def test_mutable_nli_revision_is_rejected_before_loading():
    with pytest.raises(ValueError,match="immutable"):
        LocalEntailmentScorer(revision="main")


def test_complete_diagnostics_keep_rejection_and_count_later_checks():
    class FirstNeutral(Scorer):
        def score(self, *args):
            v = super().score(*args)
            return {'entailment': .01, 'neutral': .98, 'contradiction': .01} if len(self.inputs)==1 else v
    policy, _ = setup(scorer=FirstNeutral())
    a = policy.assess(request(INCOMING),request(),response(),collect_all_scores=True)
    assert not a.recommended_reuse and a.reason=='original_to_incoming_not_entailed'
    assert len(a.scores)==a.work['nli_attempts']==4


def test_complete_diagnostics_preserve_hard_guards():
    policy, _ = setup()
    a = policy.assess(request('How do I convert a tuple to a list?'),request(),response(),collect_all_scores=True)
    assert not a.recommended_reuse and a.work['nli_attempts']==0


def test_complete_diagnostics_later_error_abstains():
    class LaterError(Scorer):
        def score(self, *args):
            super().score(*args)
            if len(self.inputs)>1:
                raise RuntimeError('fixture')
            return {'entailment': .01, 'neutral': .98, 'contradiction': .01}
    policy, _ = setup(scorer=LaterError())
    a = policy.assess(request(INCOMING),request(),response(),collect_all_scores=True)
    assert not a.recommended_reuse and a.reason=='local_check_failed'
    assert a.work['nli_attempts']==2
