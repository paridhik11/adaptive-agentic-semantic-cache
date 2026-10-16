import io
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import pytest
from fastapi.testclient import TestClient
from src.api.streaming import SSEReader,encode_event
from src.api.providers import HTTPChatProvider,FixtureChatProvider,ProviderError
from src.api.service import GatewayService,GatewayBusy
from src.api.gateway import create_app
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision

TOKEN='stream-test-token-long-enough'
BODY={'model':'fixture-model','messages':[{'role':'user','content':'What is photosynthesis?'}],'temperature':0,'max_tokens':128,'stream':True}


def service(provider=None,max_inflight=1):
    c=ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'test'))
    return GatewayService(c,provider or FixtureChatProvider(),allowed_models={'fixture-model'},max_inflight=max_inflight)


def events():
    return [{'choices':[{'index':0,'delta':{'content':'Hello'},'finish_reason':None}]},
            {'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':1}},'[DONE]']


def test_fixture_stream_complete_bypasses_all_cache_storage():
    s=service();client=TestClient(create_app(s,tenant_tokens={TOKEN:'a'}))
    for _ in range(2):
        result=client.post('/v1/chat/completions',json=BODY,headers={'Authorization':'Bearer '+TOKEN})
        assert result.status_code==200 and result.headers['x-cache']=='BYPASS'
        assert result.text.endswith('data: [DONE]\n\n')
        assert 'SIMULATED' in result.text
    assert s.cache.size==0
    metrics=s.metrics.render()
    assert 'cache_gateway_stream_completed_total 2' in metrics
    assert 'cache_gateway_upstream_attempts_total 2' in metrics
    assert 'cache_gateway_upstream_usage_missing_total 0' in metrics


def test_stream_holds_slot_until_closed_and_counts_cancellation_once():
    s=service();session=s.open_stream(BODY,'a')
    with pytest.raises(GatewayBusy):s.open_stream(BODY,'b')
    next(session);session.close();session.close()
    other=s.open_stream(BODY,'b');other.close()
    assert 'cache_gateway_stream_cancelled_total 2' in s.metrics.render()
    assert s.cache.size==0


@pytest.mark.parametrize('bad',[
    [],events()[:-1],['[DONE]'],[{'error':'private secret'}],
    [{'choices':[{'delta':{'content':'x'},'finish_reason':'length'}]},'[DONE]'],
    [{'choices':[{'delta':{'tool_calls':[{}]},'finish_reason':None}]}],
])
def test_partial_or_malformed_stream_never_claims_completion(bad):
    class Broken(FixtureChatProvider):
        def open_stream(self,body):return iter(bad)
    s=service(Broken());session=s.open_stream(BODY,'a')
    wire=b''.join(session)
    assert b'upstream_stream_failed' in wire and b'[DONE]' not in wire
    assert b'private secret' not in wire
    assert s.cache.size==0
    assert 'cache_gateway_stream_failed_total 1' in s.metrics.render()
    session=s.open_stream(BODY,'a');session.close() # slot was released


def test_open_failure_releases_slot():
    class Fail(FixtureChatProvider):
        def open_stream(self,body):raise ProviderError('upstream_timeout',504)
    s=service(Fail());client=TestClient(create_app(s,tenant_tokens={TOKEN:'a'}))
    for _ in range(2):
        response=client.post('/v1/chat/completions',json=BODY,headers={'Authorization':'Bearer '+TOKEN})
        assert response.status_code==504


@pytest.mark.parametrize('wire', [b'data: not-json\n\n',b'data: {"x":NaN}\n\n',b'data: '+b'x'*100+b'\n\n'])
def test_parser_bounds_and_invalid_json(wire):
    reader=SSEReader(io.BytesIO(wire),max_event_bytes=64)
    with pytest.raises((ValueError,ProviderError)):list(reader)
    reader.close()


def test_multiline_crlf_comment_and_unicode():
    payload=b': heartbeat\r\n\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]\r\n\r\n'
    assert list(SSEReader(io.BytesIO(payload)))==[{'choices':[]},'[DONE]']


@contextmanager
def upstream(payload):
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            self.wfile.write(payload);self.wfile.flush()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),seen
    finally:server.shutdown();server.server_close();thread.join()


def test_real_http_stream_transport_and_usage():
    payload=b''.join(encode_event(e) for e in events())
    with upstream(payload) as (url,seen):
        s=service(HTTPChatProvider(url));wire=b''.join(s.open_stream(BODY,'a'))
        assert wire==payload
        assert seen[0]['stream_options']=={'include_usage':True}
        assert 'cache_gateway_upstream_prompt_tokens_total 3' in s.metrics.render()


def test_stream_never_reads_existing_exact_entry():
    s=service();body={**BODY,'stream':False}
    assert s.complete(body,'a').stored
    assert b'[DONE]' in b''.join(s.open_stream(BODY,'a'))
    assert s.cache.size==1
    assert s.complete(body,'a').status=='HIT'


def test_deadline_checked_between_partial_network_reads():
    now=[0]
    class Trickle(io.BytesIO):
        def read1(self,n):
            now[0]+=1
            return b'x'
    reader=SSEReader(Trickle(),max_seconds=2,clock=lambda:now[0])
    with pytest.raises(ProviderError,match='deadline'):list(reader)


def test_asgi_disconnect_closes_source_and_releases_slot():
    import asyncio
    from src.api.streaming import ManagedStreamingResponse
    s=service();session=s.open_stream(BODY,'a')
    async def check():
        async def receive():return {'type':'http.disconnect'}
        async def send(message):
            if message['type']=='http.response.body':raise OSError('client disconnected')
        from starlette.requests import ClientDisconnect
        with pytest.raises(ClientDisconnect):
            await ManagedStreamingResponse(session)({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
    asyncio.run(check())
    assert session.closed
    other=s.open_stream(BODY,'a');other.close()


def test_cleanup_error_does_not_leak_capacity():
    class Source:
        def __iter__(self):return iter(events())
        def close(self):raise OSError('close failed')
    class Provider(FixtureChatProvider):
        def open_stream(self,body):return Source()
    s=service(Provider());session=s.open_stream(BODY,'a')
    assert b'[DONE]' in b''.join(session)
    assert 'cache_gateway_stream_cleanup_errors_total 1' in s.metrics.render()
    s.open_stream(BODY,'a').close()
