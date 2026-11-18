"""Research-inspired QA diagnostic. Never authorizes application serving.

This is NOT a reproduction of Q2/QAFactEval or a calibrated correctness model.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import math
import re
import threading
import time

from src.policy.decomposed_applicability import call_names, numbers, requirement_guard, split_claims, task_statement
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import ReuseDecision, digest

QA_MODEL = 'deepset/minilm-uncased-squad2'
QA_REVISION = '934656cdda79824eabf503ed56e15c01ddbdbe3f'
QA_MARGINS = (-5., 0., 2., 5., 10.)
SUPPORT_THRESHOLDS = (.5, .75, .9, .95, .99)


@dataclass(frozen=True)
class ExtractedSpan:
    text: str
    start: int
    end: int
    margin: float
    input_tokens: int


def decode_span(context, offsets, sequence_ids, starts, ends, cls_index, max_answer_tokens=40):
    """Bounded context-only extraction, retaining signed null-margin evidence."""
    size = len(offsets)
    if not (len(sequence_ids) == len(starts) == len(ends) == size and 0 <= cls_index < size):
        raise ValueError('invalid QA output shape')
    if not all(math.isfinite(float(v)) for v in [*starts, *ends]):
        raise ValueError('nonfinite QA logits')
    best = None
    for i in range(size):
        if sequence_ids[i] != 1 or offsets[i][0] >= offsets[i][1]:
            continue
        for j in range(i, min(size, i + max_answer_tokens)):
            if sequence_ids[j] != 1 or offsets[j][0] >= offsets[j][1]:
                break
            score = float(starts[i]) + float(ends[j])
            if best is None or score > best[0]:
                best = (score, offsets[i][0], offsets[j][1])
    if best is None or not 0 <= best[1] < best[2] <= len(context):
        raise ValueError('no valid QA context span')
    return ExtractedSpan(context[best[1]:best[2]], best[1], best[2],
                         best[0] - float(starts[cls_index]) - float(ends[cls_index]), size)


class LocalSpanReader:
    def __init__(self, model=QA_MODEL, revision=QA_REVISION):
        if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError('immutable QA revision required')
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
            local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model, revision=revision,
            local_files_only=True, trust_remote_code=False, use_safetensors=True).to('cpu').eval()
        self.identity = model + '@' + revision + ':context-span-40-window-384-v1'
        self._lock = threading.Lock()

    def extract(self, question, context):
        if not all(isinstance(t, str) and t.strip() for t in (question, context)):
            raise ValueError('nonempty QA texts required')
        encoded = self.tokenizer(question, context, truncation=False,
                                  return_offsets_mapping=True, return_tensors='pt')
        size = encoded['input_ids'].shape[-1]
        if size > min(self.tokenizer.model_max_length, 384):
            raise ValueError('QA window exceeded; no truncation')
        ids = encoded.sequence_ids(0)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        tokens = encoded['input_ids'][0].tolist()
        cls = tokens.index(self.tokenizer.cls_token_id)
        with self._lock, self.torch.inference_mode():
            output = self.model(**encoded)
        return decode_span(context, offsets, ids, output.start_logits[0].float().tolist(),
                           output.end_logits[0].float().tolist(), cls)


def span_key(text):
    # Ignore formatting whitespace without merging distinct words or numbers.
    # Preserve operators and case; parsed call anchors preserve string literals.
    return tuple(re.findall(r'\w+|[^\w\s]', text.strip().strip('`')))


def extraction_question(question, form='raw'):
    if form not in ('raw', 'function'):
        raise ValueError('unknown QA question form')
    if form=='function':
        return re.sub(r'^How (?:do|can|should) I\s+', 'Which function can ', question, flags=re.I)
    return question


def anchored_key(text, span):
    """Expand a QA boundary to one overlapping maximal Python call, if unique.

    ASTs preserve arguments/operators/keywords. No approved functions or query
    pairs are registered. Non-code or ambiguous extractions retain strict keys.
    """
    calls = []
    for match in re.finditer(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(', text):
        start = match.start()
        for end in range(match.end(), min(len(text), start+512)+1):
            if text[end-1] != ')':
                continue
            try:
                node = ast.parse(text[start:end], mode='eval').body
            except (SyntaxError, ValueError, RecursionError):
                continue
            if isinstance(node, ast.Call):
                calls.append((start, end, ast.dump(node, include_attributes=False)))
                break
    maximal = [row for row in calls if not any(a <= row[0] and row[1] <= b
               and (a,b) != row[:2] for a,b,_ in calls)]
    overlaps = [row for row in maximal if row[0] < span.end and span.start < row[1]]
    return ('python-call', overlaps[0][2]) if len(overlaps)==1 else ('text', span_key(span.text))


def enclosing_sentence(text, start, end):
    for match in re.finditer(r'.+?(?:\.(?=\s|$)|$)', text):
        if match.start() <= start < end <= match.end():
            return match.group(), match.start(), match.end()
    raise ValueError('span crosses sentence boundary')


@dataclass(frozen=True)
class QAAssessment:
    recommended_reuse: bool
    reason: str
    features: dict
    work: dict
    binding: dict


def recommend(features, qa_margin=0., support_threshold=.95):
    if (type(qa_margin) not in (int, float) or not math.isfinite(qa_margin)
            or type(support_threshold) not in (int, float)
            or not .5 <= support_threshold <= 1):
        raise ValueError('invalid diagnostic thresholds')
    if not features.get('complete'):
        return False
    return (features['span_agreement'] is True and features['source_requirements_present'] is True
            and features.get('intent_preserved', True) is True
            and features['min_qa_margin'] >= qa_margin
            and features['min_support'] >= support_threshold)


class QuestionConditionedPolicy:
    def __init__(self, *, context, reader, scorer, span_mode='strict', question_form='raw', intent_gate=False):
        if not isinstance(context, str) or not context.strip():
            raise ValueError('fixed context required')
        if span_mode not in ('strict', 'api-anchor'):
            raise ValueError('unknown span representation')
        extraction_question('', question_form)
        if type(intent_gate) is not bool:
            raise ValueError('intent gate must be boolean')
        for obj, method in [(reader, 'extract'), (scorer, 'score')]:
            if not callable(getattr(obj, method, None)) or not isinstance(getattr(obj, 'identity', None), str) or not obj.identity:
                raise ValueError('identified local model required')
        self.context, self.reader, self.scorer = context, reader, scorer
        self.span_mode = span_mode
        self.question_form = question_form
        self.intent_gate = intent_gate
        self.identity = 'question-conditioned-diagnostic-v1:' + digest({
            'context': context, 'reader': reader.identity, 'scorer': scorer.identity,
            'qa_margin': 0., 'support_threshold': .95, 'span_mode': span_mode,
            'question_form': question_form, 'intent_gate': intent_gate})

    def assess(self, request, candidate_request, answer):
        started = time.perf_counter()
        features, qa, nli = {'complete': False}, 0, 0
        def result(reason):
            approved = recommend(features)
            return QAAssessment(approved, reason, copy.deepcopy(features),
                {'qa_attempts': qa, 'nli_attempts': nli, 'elapsed_ms': (time.perf_counter()-started)*1000},
                {'incoming_request_sha256': digest(request), 'candidate_request_sha256': digest(candidate_request),
                 'candidate_response_sha256': digest(answer), 'context_sha256': digest(self.context),
                 'policy_identity': self.identity})
        for body in (request, candidate_request):
            if not isinstance(body, dict):
                return result('unsupported_scope')
            messages = body.get('messages')
            if (not isinstance(messages, list) or len(messages) != 2
                    or messages[0] != {'role': 'system', 'content': self.context}
                    or not isinstance(messages[1], dict) or messages[1].get('role') != 'user'
                    or not isinstance(messages[1].get('content'), str) or not messages[1]['content'].strip()
                    or type(body.get('n', 1)) is not int or body.get('n', 1) != 1):
                return result('unsupported_scope')
        first, second = copy.deepcopy(request), copy.deepcopy(candidate_request)
        for body in (first, second):
            body['messages'][-1]['content'] = '<query>'
        if first != second:
            return result('request_scope_changed')
        if not isinstance(answer, dict) or not ScopedResponseCache._admissible(answer, candidate_request):
            return result('incomplete_or_unsupported_answer')
        incoming, original = (b['messages'][-1]['content'] for b in (request, candidate_request))
        guard = requirement_guard(original, incoming)
        if guard:
            return result(guard)
        content = answer['choices'][0]['message']['content']
        claims = split_claims(content)
        if len(claims) > 8:
            return result('too_many_claims')
        try:
            spans = {}
            for name, question, text in [('source_original', original, self.context),
                                         ('source_incoming', incoming, self.context),
                                         ('answer_incoming', incoming, content)]:
                qa += 1
                span = self.reader.extract(extraction_question(question, self.question_form), text)
                if (not isinstance(span, ExtractedSpan) or type(span.start) is not int
                        or type(span.end) is not int or not 0 <= span.start < span.end <= len(text)
                        or span.text != text[span.start:span.end] or not span_key(span.text)
                        or type(span.margin) not in (int, float) or not math.isfinite(span.margin)):
                    raise ValueError('invalid extracted span')
                spans[name] = span
            features['spans'] = {name: vars(s) for name, s in spans.items()}
            texts = {'source_original': self.context, 'source_incoming': self.context,
                     'answer_incoming': content}
            keys = {name: anchored_key(texts[name], s) if self.span_mode=='api-anchor'
                    else ('text',span_key(s.text)) for name,s in spans.items()}
            features['span_keys'] = keys
            features['span_agreement'] = len(set(keys.values())) == 1
            features['min_qa_margin'] = min(s.margin for s in spans.values())
            s = spans['source_incoming']
            evidence, start, end = enclosing_sentence(self.context, s.start, s.end)
            features['evidence'] = {'text': evidence, 'start': start, 'end': end, 'sha256': digest(evidence)}
            features['source_requirements_present'] = (numbers(evidence) <= numbers(content)
                                                        and call_names(evidence) <= call_names(content))
            supports = []
            features['claim_support'] = supports
            for claim in claims:
                nli += 1
                values = self.scorer.score(self.context, claim)
                probs = [values[k] for k in ('entailment', 'neutral', 'contradiction')]
                if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs)
                        or not math.isclose(sum(probs), 1., abs_tol=1e-5)):
                    raise ValueError('invalid NLI distribution')
                supports.append(dict(values))
            features['min_support'] = min(v['entailment'] for v in supports)
            if self.intent_gate:
                intent = []
                features['intent_scores'] = intent
                for premise, hypothesis in [(original, incoming), (incoming, original)]:
                    nli += 1
                    values = self.scorer.score(task_statement(premise), task_statement(hypothesis))
                    probs = [values[k] for k in ('entailment', 'neutral', 'contradiction')]
                    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs)
                            or not math.isclose(sum(probs), 1., abs_tol=1e-5)):
                        raise ValueError('invalid intent NLI distribution')
                    intent.append(dict(values))
                features['intent_preserved'] = min(v['entailment'] for v in intent) >= .95
            features['complete'] = True
        except Exception:
            return result('local_check_failed')
        if not features['span_agreement']:
            return result('extracted_answers_disagree')
        if not features.get('intent_preserved', True):
            return result('request_intent_not_preserved')
        if not features['source_requirements_present']:
            return result('required_source_tokens_missing')
        if features['min_qa_margin'] < 0:
            return result('qa_prefers_no_answer')
        if features['min_support'] < .95:
            return result('answer_claim_not_supported')
        return result('diagnostic_positive')

    def __call__(self, request, candidate_request, answer, *, binding, provider):
        assessment = self.assess(request, candidate_request, answer)
        return ReuseDecision(False, 'diagnostic_only:' + assessment.reason,
            local_verifier_attempts=assessment.work['qa_attempts'] + assessment.work['nli_attempts'],
            local_verifier_ms=assessment.work['elapsed_ms'],
            local_policy_recommendation=assessment.recommended_reuse)
