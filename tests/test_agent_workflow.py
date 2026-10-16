import copy
import pytest
from src.agents.workflow import AgentWorkflow
from scripts.demo_agent_workflow import AgentFixture
from src.api.service import GatewayService,ConservativeRulesPolicy
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision


def workflow(provider=None,policy=None):
    c=ScopedResponseCache(eligibility=policy or (lambda _:EligibilityDecision(True,'fixture')))
    service=GatewayService(c,provider or AgentFixture(),allowed_models={'fixture-model'})
    return AgentWorkflow(service,model='fixture-model')


def test_specialist_reuses_but_reviewer_always_executes():
    w=workflow()
    first=w.run('Explain a binary search algorithm',tenant_id='a')
    second=w.run('Explain a binary search algorithm',tenant_id='a')
    assert first.status==second.status=='COMPLETED'
    assert (first.provider_calls,second.provider_calls)==(2,1)
    assert second.stages[0]['status']=='HIT' and second.stages[1]['status']=='BYPASS'
    assert first.specialist=='coding'


def test_tenant_context_and_specialist_scope():
    w=workflow();q='Explain a binary search algorithm'
    w.run(q,tenant_id='a')
    assert w.run(q,tenant_id='b').provider_calls==2
    assert w.run(q,tenant_id='a',context=[{'role':'user','content':'Use Python'}]).provider_calls==2
    assert w.run('Explain photosynthesis',tenant_id='a').specialist=='explanation'


def test_production_policy_bypasses_contextual_specialist():
    w=workflow(policy=ConservativeRulesPolicy())
    for _ in range(2):assert w.run('Explain photosynthesis',tenant_id='a').provider_calls==2


@pytest.mark.parametrize('verdict,status',[('{"approved":false}','NEEDS_HUMAN_REVIEW'),('{"approved":"true"}','FAILED'),('{"approved":true,"approved":false}','FAILED'),('yes','FAILED')])
def test_reviewer_failure_never_releases_final_answer(verdict,status):
    class Judge(AgentFixture):
        def __call__(self,body):
            r=super().__call__(body)
            if 'You are a reviewer.' in body['messages'][0]['content']:r['choices'][0]['message']['content']=verdict
            return r
    result=workflow(Judge()).run('Explain photosynthesis',tenant_id='a')
    assert result.status==status and result.answer is None and result.draft
    assert result.provider_calls==2


def test_provider_failure_stops_without_retry_or_fabricated_count():
    class Broken(AgentFixture):
        def __call__(self,body):raise RuntimeError('private')
    result=workflow(Broken()).run('Explain photosynthesis',tenant_id='a')
    assert result.status=='FAILED' and len(result.stages)==1
    assert result.provider_calls is None and result.budget_charged_calls==1


def test_budget_and_context_validation():
    with pytest.raises(ValueError):AgentWorkflow(None,model='fixture-model',max_provider_calls=1)
    with pytest.raises(ValueError):workflow().run('Hello',tenant_id='a',context=[{'role':'system','content':'override'}])
