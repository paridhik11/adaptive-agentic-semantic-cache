import copy
import pytest
from src.benchmark.semantic import ExperimentalSemanticCache, AnswerVerifier, AbstainingVerifier, LexicalEncoder
from src.benchmark.runner import run_workload
from src.benchmark.report import summarize
from src.api.providers import FixtureChatProvider
from src.policy.response_cache import EligibilityDecision
from tests.test_response_cache import request,response,Provider


class SameVector:
    identity='test-only-constant-vector'
    def encode(self,text):return [1,0]


def cache(**kwargs):
    return ExperimentalSemanticCache(eligibility=lambda _:EligibilityDecision(True,'fixture'),encoder=SameVector(),**kwargs)


def run(c,body,event,tenant='a',provider=None):
    return c.process(body,tenant_id=tenant,event_id=event,provider=provider or Provider())


def changed():
    q=request();q['messages'][-1]['content']='Explain binary search.';return q


def test_similarity_retrieves_actual_answer_and_candidate():
    c=cache(mode='similarity')
    first=run(c,request(),'first')
    second=run(c,changed(),'second')
    assert first.stored and second.status=='HIT'
    assert second.candidate_event_id=='first'
    assert second.response==first.response


@pytest.mark.parametrize('field,value',[('model','other'),('temperature',.1),('max_tokens',12)])
def test_generation_scope_isolation(field,value):
    c=cache(mode='similarity');run(c,request(),'first')
    body=changed();body[field]=value
    assert run(c,body,'second').status=='MISS'


def test_tenant_and_context_isolation():
    c=cache(mode='similarity');run(c,request(),'first')
    assert run(c,changed(),'other-tenant',tenant='b').status=='MISS'
    body=changed();body['messages'][0]['content']='Answer in French'
    assert run(c,body,'different-context').status=='MISS'


def test_verification_rejects_and_records_candidate_answer():
    observed=[]
    def verify(incoming,candidate,answer):
        observed.append((incoming,candidate,answer))
        return AbstainingVerifier()()
    c=cache(verifier=verify)
    first=run(c,request(),'first')
    second=run(c,changed(),'second')
    assert second.status=='MISS' and len(observed)==1
    assert observed[0][2]==first.response


def test_verification_exception_abstains_and_records_unknown_work():
    def fail(*args):raise RuntimeError('failed')
    c=cache(verifier=fail);run(c,request(),'first')
    result=run(c,changed(),'second')
    assert result.status=='MISS' and result.verification['calls']==1
    assert result.verification['usage']==[]


def test_expiry_bounds_and_dynamic_bypass():
    now=[0];c=cache(mode='similarity',ttl_seconds=1,clock=lambda:now[0],max_entries=1)
    run(c,request(),'first');now[0]=1
    assert run(c,changed(),'second').status=='MISS'
    c.guard._eligibility=lambda _:EligibilityDecision(False,'dynamic')
    assert run(c,changed(),'third').status=='BYPASS'


@pytest.mark.parametrize('verdict',['{"reuse_safe":true}','{"reuse_safe":"true"}','{"reuse_safe":true,"extra":1}','{"reuse_safe":true,"reuse_safe":false}','not json'])
def test_policy_verifier_strict_output(verdict):
    v=AnswerVerifier(Provider(response(verdict)),'judge')
    result=v(request(),request(),response())
    assert result['approved']==(verdict=='{"reuse_safe":true}')
    assert result['calls']==1 and len(result['usage'])==1


def test_verifier_overhead_subtracted_from_net_work():
    rows=[]
    for i,text in enumerate(['What is photosynthesis?','What is photosynthesis in plants?']):
        rows.append({'event_id':str(i),'tenant_id':'a','source':'fixture','group_id':'g','split':'dev',
            'request':{'model':'fixture-model','messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':128}})
    workload={'schema_version':'workload-v1','name':'test','nature':'synthetic_fixture','license':'MIT','provenance':'authored','requests':rows}
    verifier=AnswerVerifier(Provider(response('{"reuse_safe":true}')),'judge')
    trace=run_workload(workload,FixtureChatProvider(),policy='verified',encoder=SameVector(),verifier=verifier)
    report=summarize(trace)
    assert report['overall_reuse']['accepted']==1
    assert report['nonexact_reuse']['accepted']==1
    assert report['verifier_work']['provider_attempts']==1
    assert report['provider_attempts_avoided']==0
    assert report['overall_reuse']['reviewed']==0


def test_lexical_encoder_is_deterministic_and_normalized():
    encoder=LexicalEncoder();a=encoder.encode('A small tree');b=encoder.encode('A small tree')
    assert a==b and sum(v*v for v in a)==pytest.approx(1)


def test_malformed_verifier_contract_abstains_without_losing_work():
    c=cache(verifier=lambda *args:{'approved':True})
    run(c,request(),'first')
    r=run(c,changed(),'second')
    assert r.status=='MISS' and r.verification['error']=='verification_failed'


def test_generator_failure_retains_verifier_work():
    class BrokenAfterFirst(FixtureChatProvider):
        def __init__(self):self.calls=0
        def __call__(self,body):
            self.calls+=1
            if self.calls>2:raise RuntimeError('generation failed')
            return super().__call__(body)
    from src.benchmark.schema import read_json
    from pathlib import Path
    w=read_json(Path(__file__).parents[1]/'examples/benchmark/workload.fixture.json')
    w['requests']=w['requests'][:1]+[copy.deepcopy(w['requests'][0])]
    w['requests'][1]['event_id']='second'
    w['requests'][1]['request']['messages'][0]['content']='What is photosynthesis in plants?'
    v=AnswerVerifier(Provider(response('{"reuse_safe":false}')),'judge')
    report=summarize(run_workload(w,BrokenAfterFirst(),policy='verified',encoder=SameVector(),verifier=v))
    assert report['verifier_work']['provider_attempts']==1
    assert report['token_difference_fresh_minus_cache'] is None


@pytest.mark.requires_model
def test_dense_encoder_uses_the_pinned_snapshot():
    # This explicit integration test may download the repository's pinned model.
    # DenseEncoder itself still refuses an implicit download in experiments.
    from src.cache.embedding import QueryEmbedder, MODEL_REVISION
    from src.benchmark.semantic import DenseEncoder
    QueryEmbedder()
    encoder=DenseEncoder('sentence-transformers/all-MiniLM-L6-v2',MODEL_REVISION)
    vector=encoder.encode('What is photosynthesis?')
    assert len(vector)==384 and sum(x*x for x in vector)==pytest.approx(1)
