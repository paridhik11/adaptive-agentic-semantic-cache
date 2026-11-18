"""Question-only bounded-domain rejection rules. Never an acceptance oracle.

The domain vocabulary is authored and aligned with the synthetic task domain.
It is not an independent learned semantic parser or evidence of natural accuracy.
No generator signature, cached answer, label or oracle is an input here.
"""
import re

VERSION = 'question-veto-v1'
OPERATIONS = {
    'sort': r'\bsort\b.*ascending', 'reverse': r'\breverse the order',
    'deduplicate': r'remove repeated values', 'chunk': r'divide.*blocks of length',
    'window': r'overlapping windows', 'interleave': r'\binterleave\b',
    'cartesian': r'Cartesian product', 'permutations': r'ordered pairs of distinct',
    'combinations': r'unordered pairs of distinct', 'frequency': r'count occurrences',
    'cumulative-sum': r'cumulative sums', 'rotate': r'\brotate\b',
    'positive-filter': r'retain only positive', 'even-filter': r'retain only even',
    'double': r'\bdouble each', 'square': r'\bsquare each', 'absolute': r'absolute value',
    'negative-filter': r'retain only negative', 'odd-filter': r'retain only odd',
    'running-product': r'cumulative products', 'adjacent-difference': r'subtract each element',
    'enumerate': r'attach the zero-based index', 'nonzero-filter': r'retain only nonzero',
    'stable-partition': r'separate.*even and odd', 'lowercase': r'letters.*lowercase',
    'uppercase': r'letters.*uppercase', 'capitalize': r'capitalize only the first',
    'titlecase': r'capitalize the beginning of every word', 'word-count': r'count whitespace-separated words',
    'strip-space': r'remove leading and trailing whitespace', 'reverse-text': r'reverse the characters',
    'split-lines': r'split.*line boundaries', 'encode-utf8': r'encode.*UTF-8 bytes',
    'encode-base64': r'encode.*Base64 text', 'encode-json': r'serialize.*JSON string',
    'html-escape': r'escape HTML special', 'count-digits': r'count digit characters',
    'count-spaces': r'count space characters', 'remove-vowels': r'remove vowel characters',
    'swap-case': r'swap uppercase and lowercase',
}
TYPES = {'list': r'\blist of integers\b', 'tuple': r'\btuple of integers\b',
         'unicode': r'\bUnicode string\b', 'ascii': r'\bASCII string\b'}
REQUIREMENTS = {
    'code': r'include python code|python implementation is required|supply executable python',
    'example': r'worked example|concrete input and output|demonstrate with an example',
    'complexity': r'explain time complexity|discuss the time cost|running-time complexity',
    'proof': r'justify correctness|correctness argument|why the method is correct',
}


def signature(question):
    operations = [key for key, pattern in OPERATIONS.items() if re.search(pattern, question, re.I)]
    types = [key for key, pattern in TYPES.items() if re.search(pattern, question, re.I)]
    requirements = sorted(key for key, pattern in REQUIREMENTS.items() if re.search(pattern, question, re.I))
    units = sorted(set(x.lower() for x in re.findall(r'\b(?:milliseconds?|seconds?|ms|s)\b', question, re.I)))
    normalized_units = sorted({'ms' if x in {'ms','millisecond','milliseconds'} else 's' for x in units})
    inverse = bool(re.search(r'recover the original input|undo this operation|reconstruct the source', question, re.I))
    negated = bool(re.search(r'\bdo not carry out|avoid doing this|refrain from performing', question, re.I))
    numbers = sorted(re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)*(?![A-Za-z0-9])', question))
    versions = sorted(re.findall(r'\b3\.\d+\b', question))
    unknown_constraint = bool(re.search(r'\balso\b|\bwithout\b|\bmust\b|\bexcept\b|constant memory|no recursion', question, re.I))
    operation = operations[0] if len(operations)==1 else None
    return {'operation': operation, 'input_type': types[0] if len(types)==1 else None,
            # Output signature is domain-derived, not read from labeled data.
            'output_type': 'result:'+operation if operation else None,
            'direction': 'inverse' if inverse else 'forward', 'negated': negated,
            'numbers': numbers, 'versions': versions, 'units': normalized_units,
            'requirements': requirements, 'unknown_constraint': unknown_constraint}


def veto(cached_question, new_question):
    cached, new = signature(cached_question), signature(new_question)
    reasons = []
    if any(s['operation'] is None or s['input_type'] is None for s in [cached,new]):
        reasons.append('unparsed_or_ambiguous_question')
    if cached['unknown_constraint'] or new['unknown_constraint']:
        reasons.append('unsupported_constraint')
    for field in ['operation','input_type','output_type','direction','negated','numbers','versions','units','requirements']:
        if cached[field] != new[field]:
            reasons.append(field+'_mismatch')
    return {'reject': bool(reasons), 'reasons': reasons, 'version': VERSION}
