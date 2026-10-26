"""Before-agent lookup and admission of completed, reviewed document answers.

No generation is done here. Misses return control to the durable agent. A source
answer is admitted only after workflow approval plus required human approval.
"""
from dataclasses import asdict
import copy
import re
import sqlite3
from contextlib import nullcontext

from src.policy.encoders import normalize
from src.policy.response_cache import ScopedResponseCache, _serialize
from src.policy.semantic_runtime import GuardedSemanticCache, PinnedContextPolicy
from src.policy.sqlite_store import SQLiteResponseStore


class _LookupMiss(Exception):
    pass


class CompletedDocumentCache:
    CONTRACT = 'standalone-document-answer-v1'
    _fresh = re.compile(r'\b(delete|send|execute|run|update|change|purchase|book)\b', re.I)

    def __init__(self, cache):
        if not isinstance(cache, GuardedSemanticCache) or not isinstance(cache.exact._storage, SQLiteResponseStore):
            raise ValueError('document answer cache requires a SQLite-backed semantic wrapper')
        if cache.semantic_mode in {'guarded','shadow'} and cache._persistent is None:
            raise ValueError('document semantic lookup requires source-bound durable candidates')
        if cache.semantic_mode == 'guarded' and not cache.local_approval_only:
            raise ValueError('document hit path must prohibit provider-assisted approval')
        self.cache = cache
        self.store = cache.exact._storage

    def request(self, agent, state, principal):
        context = {'contract':self.CONTRACT,'workflow':agent.VERSION,
                   'runtime_binding':agent.runtime_binding,'source_revision':agent.tools.revision,
                   'scope':agent.scope,'principal':principal,
                   'task_epoch':state.get('answer_namespace_epoch',0),
                   'relevant_memory':state['relevant_memory'],
                   'max_calls':state['max_calls'],'max_revisions':state['max_revisions']}
        return {'model':agent.model,**agent.generation,'messages':[
            {'role':'system','content':_serialize(context)},
            {'role':'user','content':state['question']}]}

    def eligible(self, request):
        # Conservative heuristic admission boundary, not a natural-language parser.
        question = request['messages'][-1]['content']
        prompt = request['messages'][0]['content']
        return (PinnedContextPolicy(prompt)(request).cacheable
                and not self._fresh.search(question)
                and self.cache.exact._eligibility_reason(request) is None)

    def epoch(self):
        try:
            return self.store.invalidation_epoch
        except sqlite3.Error:
            return None

    def lookup(self, request, tenant, ledger):
        if not self.eligible(request):
            return None, 'ineligible'
        def missing(_):
            raise _LookupMiss()
        try:
            result = self.cache.process(request, tenant_id=tenant, provider=missing, ledger=ledger)
            if result.status != 'HIT' or result.provider_calls != 0:
                raise ValueError('lookup-only cache attempted provider work')
            return result, result.reason
        except _LookupMiss:
            return None, 'no_qualified_answer'
        except Exception:
            return None, 'cache_unavailable'

    def admit(self, request, response, tenant, expected_epoch, ledger=None):
        if (expected_epoch is None or self.cache.semantic_mode == 'off'
                or not self.eligible(request) or not ScopedResponseCache._admissible(response, request)):
            return False
        stored = False
        try:
            with ledger.measure('admission') if ledger is not None else nullcontext():
                scope = asdict(self.cache._scopes.get(tenant, self.cache._default_scope))
                payload = _serialize(response)
                source_key = ScopedResponseCache.request_key(request, tenant, scope)
                # Use the epoch captured before the task: invalidation while the
                # agent or human was working must not repopulate the old source.
                stored = self.store.admit(source_key,tenant,payload,expected_epoch,
                    ttl=self.cache.exact._ttl_seconds,max_entries=self.cache.exact._max_entries,
                    max_bytes=self.cache.exact._max_bytes)
                if stored and self.cache.semantic_mode in {'guarded','shadow'}:
                    with self.cache._encoder_lock, ledger.measure('embedding') if ledger is not None else nullcontext():
                        raw = self.cache.encoder.encode(request['messages'][-1]['content'])
                        if not 0 < len(raw) <= self.cache.max_dimensions:
                            raise ValueError('embedding dimensions exceed bound')
                        vector = tuple(normalize(raw))
                    self.store.admit_candidate(namespace=self.cache.candidate_namespace(request,tenant,scope),
                        source_key=source_key,tenant=tenant,payload=payload,epoch=expected_epoch,
                        request=_serialize(request),vector=vector,max_entries=self.cache.max_entries,
                        max_bytes=self.cache.max_bytes)
                return stored
        except Exception:
            # An index failure cannot undo a user's already durable decision.
            return stored

    def invalidate(self, tenant):
        return self.cache.invalidate(tenant_id=tenant)

    @staticmethod
    def response(agent, state):
        return {'model':agent.model,'choices':[{'finish_reason':'stop','message':{
            'role':'assistant','content':state['answer']}}],
            'documentation':{'contract':CompletedDocumentCache.CONTRACT,
                'source_revision':agent.tools.revision,'original_question':state['question'],
                'workflow_approved':True,'tools':copy.deepcopy(state['tools'])}}

    @staticmethod
    def validate_answer(agent, response):
        meta = response.get('documentation', {})
        if (meta.get('contract') != CompletedDocumentCache.CONTRACT
                or meta.get('source_revision') != agent.tools.revision
                or meta.get('workflow_approved') is not True
                or not isinstance(meta.get('original_question'), str)
                or not isinstance(meta.get('tools'), list) or not 1 <= len(meta['tools']) <= 8):
            raise ValueError('invalid cached documentation answer')
        read = False
        for tool in meta['tools']:
            if not isinstance(tool,dict) or set(tool) != {'step','result'}:
                raise ValueError('invalid cached tool evidence')
            step = tool['step']
            if step.get('tool') == 'read':
                document_id = step.get('arguments',{}).get('document_id')
                if tool['result'] != agent.tools._documents.get(document_id):
                    raise ValueError('cached evidence no longer matches source')
                read = True
        if not read:
            raise ValueError('cached answer has no read evidence')
        return meta
