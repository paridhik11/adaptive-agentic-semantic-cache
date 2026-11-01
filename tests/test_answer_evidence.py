import copy
import json
import pytest
from src.benchmark.answer_evidence import AnswerReviews, ObservedProvider, review_binding, summarize
from scripts.phase3_development_benchmark import ReferenceProvider, run_benchmark
from scripts.audit_phase3_trace import audit


def test_review_is_bound_to_actual_answer_and_scope():
    binding = review_binding({'q':'q'}, {'answer':'right'}, 'tenant-a', 'rev1')
    reviews=AnswerReviews([{'binding':binding,'label':'sufficient','reviewer':'model-assisted-test', 'rationale':'fixture'}])
    assert reviews.lookup(binding)['label']=='sufficient'
    for changed in [review_binding({'q':'q'}, {'answer':'wrong'}, 'tenant-a','rev1'),
                    review_binding({'q':'q2'}, {'answer':'right'}, 'tenant-a','rev1'),
                    review_binding({'q':'q'}, {'answer':'right'}, 'tenant-b','rev1'),
                    review_binding({'q':'q'}, {'answer':'right'}, 'tenant-a','rev2')]:
        assert reviews.lookup(changed)['label']=='unknown'
    with pytest.raises(ValueError): AnswerReviews([{'binding':binding,'label':'yes'}])


def test_observer_records_failures_and_missing_usage():
    def failure(request): raise RuntimeError('fixture')
    observed=ObservedProvider(failure,usage_kind='provider_reported')
    with pytest.raises(RuntimeError): observed({})
    assert observed.events[0]['error']=='RuntimeError'
    assert observed.events[0]['usage'] is None
    assert observed.events[0]['elapsed_ms']>=0


def test_unknown_and_empty_precision_not_perfect():
    row={'is_hit':True,'tier':'SEMANTIC','answer_review':{'label':'unknown'},'provider_events':[], 'elapsed_ms':1,'error':None}
    s=summarize([row]); assert s['semantic_precision_labeled'] is None
    assert s['semantic_precision_all_accepted'] is None
    assert summarize([])['semantic_precision_labeled'] is None
    assert summarize([])['latency_p95_ms'] is None


def test_actual_review_overrides_generated_family_and_cached_usage_not_counted(tmp_path):
    workload=json.load(open('artifacts/documentation_rebuild/semantic_p0/phase3_dev_workload.json'))[:2]
    # Even the known exact-repeat family cannot manufacture a positive answer label.
    response=ReferenceProvider(0)(workload[0]['request'])
    binding=review_binding(workload[1]['request'], response, workload[1]['tenant'],workload[1]['revision'])
    reviews=[{'binding':binding,'label':'insufficient','reviewer':'model-assisted-fixture','rationale':'deliberate negative review'}]
    result=run_benchmark(workload,tmp_path/'new.json',arms=['B_exact_cache'],reviews=reviews,delay_ms=0)
    rows=result['results']['B_exact_cache']; s=result['summary']['B_exact_cache']
    assert rows[1]['correct_reuse'] is False
    assert rows[1]['unsafe_reuse'] is True
    assert rows[1]['provider_events']==[]
    assert s['provider_attempts']==1
    assert s['billable_prompt_tokens'] is None
    assert s['monetary_cost'] is None
    with pytest.raises(FileExistsError): run_benchmark(workload,tmp_path/'new.json',arms=['B_exact_cache'],delay_ms=0)


def test_spent_trace_hash_recovery_and_binding_tamper():
    raw=json.load(open('artifacts/documentation_rebuild/semantic_p0/phase3_benchmark_results.json'))
    workload=json.load(open('artifacts/documentation_rebuild/semantic_p0/phase3_dev_workload.json'))
    result=audit(raw,workload)
    assert result['summary']['E_answer_aware_policy']['semantic_answers_hash_recovered']==12
    assert len(result['review_packet'])==129
    assert all(r['label']=='unknown' for r in result['review_packet'])
    bad=copy.deepcopy(raw)
    row=next(r for r in bad['results']['E_answer_aware_policy'] if r['tier']=='SEMANTIC')
    row['diagnosis']['binding']['incoming_request_sha256']='wrong'
    with pytest.raises(ValueError,match='binding mismatch'): audit(bad,workload)


def test_recovered_reverse_answer_has_false_copy_claim():
    # Python's reverse iterator observes element replacement after construction.
    values=[1,2,3]; reverse=reversed(values); values[-1]=9
    assert next(reverse)==9
    reviews=json.load(open('artifacts/documentation_rebuild/semantic_p0/answer_aware_model_reviews_v2.json'))
    row=next(r for r in reviews if '1189' in r['event_id'])
    assert row['label']=='insufficient'
    assert 'model-assisted' in row['reviewer']


def test_observed_approver_keeps_candidate_and_verdict():
    from src.benchmark.answer_evidence import ObservedApprover
    from src.policy.semantic_runtime import ReuseDecision
    class Policy:
        identity='test'
        def __call__(self,*args,**kwargs):return ReuseDecision(False,'fixture-abstention')
    recorder=ObservedApprover(Policy())
    answer={'text':'actual candidate'}
    verdict=recorder({}, {'q':'old'}, answer, binding={'hash':'test'})
    answer['text']='later mutation'
    assert verdict.approved is False
    assert recorder.events[0]['candidate_response']['text']=='actual candidate'
    assert recorder.events[0]['decision']['reason']=='fixture-abstention'
