"""Bounded response diagnostics without exporting answer/reasoning text."""
import hashlib
from src.policy.response_cache import ScopedResponseCache
from src.benchmark.schema import canonical


def response_diagnostics(response,request):
    try:
        response_hash=hashlib.sha256(canonical(response).encode()).hexdigest()
    except (TypeError,ValueError,UnicodeError):response_hash=None
    data=response if isinstance(response,dict) else {}
    choices=data.get('choices')
    summaries=[]
    if isinstance(choices,list):
        for choice in choices[:4]:
            choice=choice if isinstance(choice,dict) else {}
            message=choice.get('message')
            message=message if isinstance(message,dict) else {}
            text=message.get('content');reasoning=message.get('reasoning')
            finish=choice.get('finish_reason')
            summaries.append({'finish_reason':finish[:64] if isinstance(finish,str) else None,
                'assistant_text_characters':len(text) if isinstance(text,str) else None,
                'reasoning_text_characters':len(reasoning) if isinstance(reasoning,str) else None,
                'has_tool_calls':bool(message.get('tool_calls') or message.get('function_call')),
                'has_refusal':bool(message.get('refusal'))})
    admitted=response_hash is not None and isinstance(response,dict) and ScopedResponseCache._admissible(response,request)
    reason=None
    if not admitted:
        if response_hash is None:reason='invalid_response_json'
        elif not isinstance(choices,list) or not choices:reason='invalid_response_shape'
        elif any(c['finish_reason']=='length' for c in summaries):reason='output_token_limit'
        elif any(c['finish_reason']!='stop' for c in summaries):reason='completion_not_stopped'
        elif any(c['has_tool_calls'] or c['has_refusal'] for c in summaries):reason='tool_or_refusal_response'
        elif any(not c['assistant_text_characters'] for c in summaries):reason='empty_assistant_answer'
        else:reason='response_not_admissible'
    usage=data.get('usage');usage=usage if isinstance(usage,dict) else {}
    tokens={k:usage[k] for k in ('prompt_tokens','completion_tokens') if type(usage.get(k)) is int and usage[k]>=0}
    details=usage.get('completion_tokens_details')
    if isinstance(details,dict) and type(details.get('reasoning_tokens')) is int and details['reasoning_tokens']>=0:
        tokens['reasoning_tokens']=details['reasoning_tokens']
    model=data.get('model')
    return {'response_sha256':response_hash,'model':model[:256] if isinstance(model,str) else None,
            'admissible_complete_answer':admitted,'rejection_reason':reason,'choices':summaries,'usage':tokens}
