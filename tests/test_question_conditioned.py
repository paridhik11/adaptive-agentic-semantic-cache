"""Engineering boundaries; fixture model outputs do not measure model quality."""
import copy
from types import SimpleNamespace

import pytest

from src.policy.question_conditioned import (ExtractedSpan, LocalSpanReader, QuestionConditionedPolicy,
    decode_span, recommend, span_key)
from tests.test_decomposed_applicability import Scorer, request, response, CONTEXT, INCOMING


class Reader:
    identity = 'fixture-reader'
    def __init__(self):
        self.calls = []
    def extract(self, question, text):
        self.calls.append((question, text))
        start = text.index('tuple(items)')
        return ExtractedSpan('tuple(items)', start, start+12, 4., 30)


def setup(reader=None, scorer=None):
    return QuestionConditionedPolicy(context=CONTEXT, reader=reader or Reader(), scorer=scorer or Scorer())


def test_question_conditioned_check_binds_actual_answer_and_records_all_work():
    policy = setup()
    a = policy.assess(request(INCOMING), request(), response())
    assert a.recommended_reuse
    assert a.work['qa_attempts'] == 3 and a.work['nli_attempts'] == 1
    assert policy.reader.calls[1][0] == INCOMING
    assert policy.reader.calls[2][1] == response()['choices'][0]['message']['content']
    assert a.binding['candidate_response_sha256']


def test_positive_diagnostic_never_calls_remote_provider_or_serves():
    policy = setup()
    def forbidden(*args):
        raise AssertionError('remote call forbidden')
    d = policy(request(INCOMING), request(), response(), binding={}, provider=forbidden)
    assert not d.approved and d.local_policy_recommendation
    assert d.local_verifier_attempts == 4


@pytest.mark.parametrize('change', ['direction', 'context', 'model', 'incomplete'])
def test_scope_requirements_and_incomplete_answer_stop_before_model_work(change):
    policy = setup()
    incoming, answer = request(INCOMING), response()
    if change == 'direction':
        incoming = request('How do I convert a tuple to a list?')
    elif change == 'context':
        incoming['messages'][0]['content'] += ' Changed.'
    elif change == 'model':
        incoming['model'] = 'different'
    else:
        answer['choices'][0]['finish_reason'] = 'length'
    a = policy.assess(incoming, request(), answer)
    assert not a.recommended_reuse and a.work['qa_attempts'] == a.work['nli_attempts'] == 0


def test_low_null_margin_and_unsupported_additional_claim_reject():
    class NoAnswer(Reader):
        def extract(self, *args):
            s = super().extract(*args)
            return ExtractedSpan(s.text, s.start, s.end, -1., s.input_tokens)
    a = setup(reader=NoAnswer()).assess(request(INCOMING), request(), response())
    assert a.reason == 'qa_prefers_no_answer' and not a.recommended_reuse
    class Contradiction(Scorer):
        def score(self, *args):
            return {'entailment': .001, 'neutral': .001, 'contradiction': .998}
    a = setup(scorer=Contradiction()).assess(request(INCOMING), request(), response())
    assert a.reason == 'answer_claim_not_supported' and not a.recommended_reuse


def test_mismatched_span_is_not_rescued_by_high_model_confidence():
    class Mismatch(Reader):
        def extract(self, question, text):
            if text.startswith('Use'):
                return ExtractedSpan(text[:3], 0, 3, 99., 30)
            return super().extract(question, text)
    a = setup(reader=Mismatch()).assess(request(INCOMING), request(), response())
    assert a.reason == 'extracted_answers_disagree' and not a.recommended_reuse
    assert not recommend(a.features, -5., .5)


@pytest.mark.parametrize('bad', ['offset', 'nan', 'error'])
def test_invalid_reader_results_fail_closed_with_attempt_accounting(bad):
    class Broken(Reader):
        def extract(self, *args):
            if bad == 'error':
                raise RuntimeError('fixture')
            if bad == 'nan':
                return ExtractedSpan('x', 0, 1, float('nan'), 4)
            return ExtractedSpan('invented', 0, 4, 99., 4)
    a = setup(reader=Broken()).assess(request(INCOMING), request(), response())
    assert not a.recommended_reuse and not a.features['complete']
    assert a.work['qa_attempts'] == 1 and a.work['nli_attempts'] == 0


def test_decoder_excludes_question_tokens_and_handles_no_answer_margin():
    # A huge score on question tokens must not become an answer span.
    span = decode_span('abc def', [(0,0),(0,3),(0,3),(4,7),(0,0)],
                       [None,0,1,1,None], [10,100,3,1,0], [10,100,3,1,0], 0)
    assert span.text == 'abc' and span.margin == -14
    with pytest.raises(ValueError):
        decode_span('x', [(0,1)], [1], [float('nan')], [0], 0)


def test_span_comparison_preserves_operators_case_and_directional_code():
    assert span_key(' int( text ) ') == span_key('int(text)')
    assert span_key('int(text)') != span_key('str(text)')
    assert span_key('A') != span_key('a')
    assert span_key('x+1') != span_key('x-1')


def test_reader_rejects_mutable_revision_before_import_or_download():
    with pytest.raises(ValueError, match='immutable'):
        LocalSpanReader(revision='main')


def test_reader_rejects_overflow_before_model_inference():
    class Encoded(dict):
        def sequence_ids(self, _):
            raise AssertionError('must refuse overflow first')
    reader = object.__new__(LocalSpanReader)
    reader.tokenizer = lambda *args, **kwargs: Encoded(input_ids=SimpleNamespace(shape=(1, 385)))
    reader.tokenizer.model_max_length = 512
    with pytest.raises(ValueError, match='window'):
        reader.extract('question', 'context')


def test_call_anchoring_fixes_boundaries_without_dropping_arguments_or_keywords():
    from src.policy.question_conditioned import anchored_key
    def key(text, term):
        start=text.index(term)
        return anchored_key(text,ExtractedSpan(term,start,start+len(term),1.,10))
    assert key('str(value) converts values.', 'str') == key('Use str(value).','Use str(value)')
    assert key('sorted(values, reverse=True)', 'sorted') != key('sorted(values)', 'sorted')
    assert key('str(value)', 'str') != key('int(value)', 'int')
    assert key('f(a + 1)', 'f') != key('f(a - 1)', 'f')
    # A span containing two unrelated calls is ambiguous and cannot select one.
    assert key('f(a) and g(b)', 'f(a) and g(b)')[0]=='text'
    # Nested calls preserve their entire expression.
    assert key('list(reversed(items))','reversed')[0]=='python-call'
    assert key('list(reversed(items))','reversed') != key('list(items)','list')


def test_generic_function_question_form_preserves_material_requirements():
    from src.policy.question_conditioned import extraction_question
    q='How can I recursively convert a list to a tuple without copying in Python 3.12?'
    assert extraction_question(q,'function')=='Which function can recursively convert a list to a tuple without copying in Python 3.12?'
    assert extraction_question(q)==q
    other='What function turns list elements into a tuple?'
    assert extraction_question(other,'function')==other
    with pytest.raises(ValueError):
        extraction_question(q,'invented')


def test_question_form_cannot_change_request_bindings_or_scope_guards():
    policy=QuestionConditionedPolicy(context=CONTEXT,reader=Reader(),scorer=Scorer(),question_form='function')
    a=policy.assess(request(INCOMING),request(),response())
    raw=setup().assess(request(INCOMING),request(),response())
    assert a.binding['incoming_request_sha256']==raw.binding['incoming_request_sha256']
    assert policy.reader.calls[1][0].startswith('Which function can ')
    blocked=policy.assess(request('How do I convert a tuple to a list?'),request(),response())
    assert blocked.work['qa_attempts']==0 and not blocked.recommended_reuse


def test_hybrid_intent_gate_rejects_supported_but_different_request():
    class IntentNeutral(Scorer):
        def score(self, premise, hypothesis):
            if premise.startswith('The requested'):
                self.inputs.append((premise,hypothesis))
                return {'entailment':.01,'neutral':.98,'contradiction':.01}
            return super().score(premise,hypothesis)
    policy=QuestionConditionedPolicy(context=CONTEXT,reader=Reader(),scorer=IntentNeutral(),intent_gate=True)
    a=policy.assess(request(INCOMING),request(),response())
    assert a.reason=='request_intent_not_preserved' and not a.recommended_reuse
    assert a.work['qa_attempts']==3 and a.work['nli_attempts']==3
    assert len(a.features['intent_scores'])==2
    assert not recommend(a.features,-5.,.5)


def test_hybrid_gate_records_positive_diagnostic_without_serving():
    policy=QuestionConditionedPolicy(context=CONTEXT,reader=Reader(),scorer=Scorer(),intent_gate=True)
    a=policy.assess(request(INCOMING),request(),response())
    assert a.recommended_reuse and a.features['intent_preserved']
    d=policy(request(INCOMING),request(),response(),binding={},provider=lambda _:pytest.fail('remote'))
    assert not d.approved


def test_span_formatting_never_concatenates_separate_words_or_numbers():
    assert span_key('1 0') != span_key('10')
    assert span_key('data base') != span_key('database')
    assert span_key('f( x )') == span_key('f(x)')
