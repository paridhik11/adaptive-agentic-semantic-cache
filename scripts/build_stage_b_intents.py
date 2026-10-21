"""Versioned synthetic B0 data; no Stage A/final records are read."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Operation lineages, not formatter IDs, determine the split. All operations
# contain positives and every definite relation; operation cannot encode label.
OPERATIONS = [
    ('sort', 'sort {input} in ascending order', 'sequence', 'sorted(values)'),
    ('reverse', 'reverse the order of {input}', 'sequence', 'list(reversed(values))'),
    ('deduplicate', 'remove repeated values from {input} while preserving first occurrences', 'sequence', 'list(dict.fromkeys(values))'),
    ('chunk', 'divide {input} into blocks of length 3', 'sequence', '[values[i:i+3] for i in range(0,len(values),3)]'),
    ('window', 'produce overlapping windows of length 3 from {input}', 'sequence', '[values[i:i+3] for i in range(len(values)-2)]'),
    ('interleave', 'interleave {input} with a second sequence until either ends', 'sequence', '[x for pair in zip(values, other) for x in pair]'),
    ('cartesian', 'take the Cartesian product of {input} with a second sequence', 'sequence', 'list(itertools.product(values, other))'),
    ('permutations', 'enumerate ordered pairs of distinct positions from {input}', 'sequence', 'list(itertools.permutations(values,2))'),
    ('combinations', 'enumerate unordered pairs of distinct positions from {input}', 'sequence', 'list(itertools.combinations(values,2))'),
    ('frequency', 'count occurrences of every value in {input}', 'sequence', 'collections.Counter(values)'),
    ('cumulative-sum', 'calculate cumulative sums of {input}', 'sequence', 'list(itertools.accumulate(values))'),
    ('rotate', 'rotate {input} one place to the left', 'sequence', 'values[1:]+values[:1]'),
    ('positive-filter', 'retain only positive values in {input}', 'sequence', '[x for x in values if x>0]'),
    ('even-filter', 'retain only even values in {input}', 'sequence', '[x for x in values if x%2==0]'),
    ('double', 'double each value in {input}', 'sequence', '[2*x for x in values]'),
    ('square', 'square each value in {input}', 'sequence', '[x*x for x in values]'),
    ('absolute', 'take the absolute value of each element in {input}', 'sequence', '[abs(x) for x in values]'),
    ('negative-filter', 'retain only negative values in {input}', 'sequence', '[x for x in values if x<0]'),
    ('odd-filter', 'retain only odd values in {input}', 'sequence', '[x for x in values if x%2!=0]'),
    ('running-product', 'calculate cumulative products of {input}', 'sequence', 'list(itertools.accumulate(values,operator.mul))'),
    ('adjacent-difference', 'subtract each element of {input} from its successor', 'sequence', '[b-a for a,b in itertools.pairwise(values)]'),
    ('enumerate', 'attach the zero-based index to every value in {input}', 'sequence', 'list(enumerate(values))'),
    ('nonzero-filter', 'retain only nonzero values in {input}', 'sequence', '[x for x in values if x!=0]'),
    ('stable-partition', 'separate {input} into even and odd values preserving order', 'sequence', '([x for x in values if x%2==0],[x for x in values if x%2!=0])'),
    ('lowercase', 'convert all letters in {input} to lowercase', 'string', 'values.lower()'),
    ('uppercase', 'convert all letters in {input} to uppercase', 'string', 'values.upper()'),
    ('capitalize', 'capitalize only the first character of {input}', 'string', 'values.capitalize()'),
    ('titlecase', 'capitalize the beginning of every word in {input}', 'string', 'values.title()'),
    ('word-count', 'count whitespace-separated words in {input}', 'string', 'len(values.split())'),
    ('strip-space', 'remove leading and trailing whitespace from {input}', 'string', 'values.strip()'),
    ('reverse-text', 'reverse the characters in {input}', 'string', 'values[::-1]'),
    ('split-lines', 'split {input} at line boundaries', 'string', 'values.splitlines()'),
    ('encode-utf8', 'encode {input} as UTF-8 bytes', 'string', 'values.encode("utf-8")'),
    ('encode-base64', 'encode {input} as Base64 text', 'string', 'base64.b64encode(values.encode()).decode()'),
    ('encode-json', 'serialize {input} as a JSON string', 'string', 'json.dumps(values)'),
    ('html-escape', 'escape HTML special characters in {input}', 'string', 'html.escape(values)'),
    ('count-digits', 'count digit characters in {input}', 'string', 'sum(x.isdigit() for x in values)'),
    ('count-spaces', 'count space characters in {input}', 'string', 'values.count(" ")'),
    ('remove-vowels', 'remove vowel characters from {input}', 'string', '"".join(x for x in values if x.lower() not in "aeiou")'),
    ('swap-case', 'swap uppercase and lowercase letters in {input}', 'string', 'values.swapcase()'),
]
TYPES = {'sequence': ['a list of integers', 'a tuple of integers'],
         'string': ['a Unicode string', 'an ASCII string']}
REQUIREMENTS = ['code', 'example', 'complexity', 'proof']
REQUIREMENT_TEXT = {
    'train': {'code': 'Include Python code.', 'example': 'Include a worked example.',
              'complexity': 'Explain time complexity.', 'proof': 'Justify correctness.'},
    'dev': {'code': 'A Python implementation is required.', 'example': 'Show a concrete input and output.',
            'complexity': 'Discuss the time cost.', 'proof': 'Provide a correctness argument.'},
    'calibration': {'code': 'Supply executable Python.', 'example': 'Demonstrate with an example.',
                    'complexity': 'State the running-time complexity.', 'proof': 'Explain why the method is correct.'},
}
FORMATTERS = {
    'train': ['How do I {task}? {requirements}', 'I need to {task}. {requirements}',
              'Please show how to {task}. {requirements}', 'What method can {task}? {requirements}'],
    'dev': ['My task is to {task}; what would you suggest? {requirements}',
            'Could you help me {task}? {requirements}',
            'The result I need: {task}. How can I obtain it? {requirements}',
            'I am looking for an approach to {task}. {requirements}'],
    'calibration': ['Can you walk through a solution to {task}? {requirements}',
                    'For my program, the goal is to {task}. {requirements}',
                    'Which approach would let me {task}? {requirements}',
                    'I would like guidance to {task}. {requirements}'],
}
EXTRAS = {
    'train': {'version': 'Use Python {version}.', 'quantity': 'Include {quantity} test cases.',
              'negation': 'Do not carry out the following operation: {task}.',
              'inverse': 'Recover the original input from the result of this operation: {task}.',
              'units': 'Express timing figures in milliseconds instead of seconds.'},
    'dev': {'version': 'Target Python release {version}.', 'quantity': 'Supply {quantity} tests.',
            'negation': 'I want to avoid doing this: {task}.',
            'inverse': 'Undo this operation to reconstruct its input: {task}.',
            'units': 'Report durations in ms rather than s.'},
    'calibration': {'version': 'Python version {version} is required.', 'quantity': 'Show {quantity} test examples.',
                    'negation': 'Refrain from performing this: {task}.',
                    'inverse': 'Reconstruct the source by reversing this operation: {task}.',
                    'units': 'Use milliseconds, not seconds, for runtime measurements.'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relation(new, cached):
    """Synthetic oracle over authored contracts, NEVER a policy input."""
    if new is None or cached is None:
        return None
    for key in ['operation', 'input_type', 'output_type', 'direction', 'version', 'quantity', 'negated']:
        if new[key] != cached[key]:
            return 'different_intent'
    a, b = set(new['requirements']), set(cached['requirements'])
    if a == b:
        return 'same_intent'
    if a > b:
        return 'new_more_specific'
    if a < b:
        return 'new_more_general'
    return 'different_intent'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/intent_stage_b/v2')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to overwrite a dataset')
    rows = {s: [] for s in FORMATTERS}
    answers = {s: [] for s in FORMATTERS}
    for index, (op, description, kind, expression) in enumerate(OPERATIONS):
        split = 'train' if index < 24 else 'dev' if index < 32 else 'calibration'
        # Two requirement variants per lineage; all four appear in every split.
        profiles = ([REQUIREMENTS[index % 4], REQUIREMENTS[(index+1) % 4]]
                    if split == 'train' else REQUIREMENTS)
        for input_type in TYPES[kind]:
            for profile in profiles:
                variants = [('3.10', 3), ('3.10', 5), ('3.11', 3), ('3.11', 5)] if split == 'calibration' else [('3.11', 3)]
                for version, quantity in variants:
                    gid = f'b2:{op}:{TYPES[kind].index(input_type)}:{profile}:{version}:{quantity}'
                    signature = dict(operation=op, input_type=input_type, output_type=f'result:{op}',
                                     direction='forward', version=version, quantity=quantity, negated=False,
                                     requirements=[profile])
                    task = description.format(input=input_type)

                    def render(sig, formatter=0, task_override=None):
                        task_text = task if task_override is None else task_override
                        if sig['negated']:
                            task_text = EXTRAS[split]['negation'].format(task=task_text)
                        if sig['direction'] != 'forward':
                            task_text = EXTRAS[split]['inverse'].format(task=task_text)
                        reqs = [REQUIREMENT_TEXT[split][x] for x in sig['requirements'] if x != 'millisecond-timing']
                        reqs += [EXTRAS[split]['version'].format(**sig), EXTRAS[split]['quantity'].format(**sig)]
                        if 'millisecond-timing' in sig['requirements']:
                            reqs.append(EXTRAS[split]['units'])
                        return FORMATTERS[split][formatter].format(task=task_text, requirements=' '.join(reqs))

                    cached_question = render(signature)
                    answer_id = gid+':answer'
                    answers[split].append({'cached_answer_id': answer_id,
                        'text': f'# Python sketch; imports and presentation obligations are unverified.\nresult = {expression}\n',
                        'provenance': 'builder_authored_synthetic_not_human_reviewed',
                        'answer_quality': 'NOT EVALUATED; relation labels are question-only'})

                    def add(sig, family, formatter=0, question=None):
                        rs = rows[split]
                        rs.append({'case_id': gid+':'+family+':'+str(formatter), 'group_id': gid,
                            'cached_case_id': gid+':cached', 'cached_answer_id': answer_id,
                            'source_lineage_id': 'operation:'+op, 'split': split,
                            'role': 'cached' if family == 'cached_control' else 'new',
                            'question': question if question is not None else render(sig, formatter),
                            'intent_signature': sig, 'relation_to_cached': relation(sig, signature),
                            'case_family': family, 'template_id': split+':formatter-'+str(formatter)+':'+family,
                            'formatter_id': split+':formatter-'+str(formatter),
                            'scope_id': 'python-fixed-reference-v1', 'label_provenance': 'builder_authored_synthetic',
                            'source': 'scripts/build_stage_b_intents.py', 'license': 'MIT (project-authored)',
                            'is_exact': family == 'exact_control', 'independent_human_ground_truth': False})
                    add(signature, 'cached_control')
                    for formatter in [1, 2, 3]:
                        add(signature, 'paraphrase', formatter)
                    add(signature, 'exact_control')
                    extra = next(x for x in REQUIREMENTS if x not in signature['requirements'])
                    add({**signature, 'requirements': signature['requirements']+[extra]}, 'added_requirement', 1)
                    add({**signature, 'requirements': []}, 'removed_requirement', 2)
                    add({**signature, 'negated': True}, 'negation', 3)
                    add({**signature, 'direction': 'inverse'}, 'direction_inversion', 1)
                    # Same operation, materially different source representation.
                    other_type = next(x for x in TYPES[kind] if x != input_type)
                    add({**signature, 'input_type': other_type}, 'entity_swap', 2,
                        render({**signature, 'input_type': other_type}, 2,
                               description.format(input=other_type)))
                    add({**signature, 'version': '3.13'}, 'version', 3)
                    add({**signature, 'quantity': quantity+1}, 'quantity', 1)
                    add({**signature, 'requirements': signature['requirements']+['millisecond-timing']}, 'unit', 2)
                    add(None, 'ambiguity', 3, question={'train': 'How should I do that?',
                        'dev': 'Can you handle it for me?', 'calibration': 'Please perform that task.'}[split])
    args.output_dir.mkdir(parents=True)
    summary = {'schema': 'stage-b-v2', 'evidence_type': 'synthetic_template_held_out_not_human_ground_truth',
               'generator_sha256': sha(__file__), 'seed': 'deterministic enumeration; no PRNG',
               'final_split_created_read_or_evaluated': False, 'public_data_used': False,
               'template_holdout': 'whole question wrappers, requirements and negative wrappers disjoint; task wording remains structured',
               'splits': {}}
    for split, rs in rows.items():
        path = args.output_dir/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rs))
        (args.output_dir/(split+'.answers.jsonl')).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in answers[split]))
        summary['splits'][split] = {'sha256': sha(path), 'rows': len(rs),
            'groups': len({r['group_id'] for r in rs}), 'lineages': dict(Counter(r['source_lineage_id'] for r in rs)),
            'families': dict(Counter(r['case_family'] for r in rs)),
            'templates': dict(Counter(r['template_id'] for r in rs)),
            'formatters': dict(Counter(r['formatter_id'] for r in rs)),
            'provenance': dict(Counter(r['label_provenance'] for r in rs)),
            'relations': dict(Counter(str(r['relation_to_cached']) for r in rs))}
    (args.output_dir/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    request = args.output_dir/'natural-review-request.jsonl'
    families = ['paraphrase','direction_inversion','negation','entity_swap','added_requirement','quantity','version','unit','removed_requirement','ambiguity']
    request.write_text(''.join(json.dumps({'slot_id': f'natural-{i+1:02d}',
        'suggested_family': families[i % len(families)], 'cached_question': '', 'new_question': '',
        'relation': None, 'human_or_model_assisted': '', 'reviewer_id': '', 'rationale': '',
        'instructions': 'Write BOTH questions independently of generated templates and policy outputs. Label same_intent/new_more_specific/new_more_general/different_intent, or leave null for ambiguity. Declare human vs model-assisted and reviewer independence. Do not copy public pairs. No labels or questions have been supplied by the builder.'})+'\n' for i in range(50)))
    print(json.dumps({s:{k:v for k,v in d.items() if k in ['rows','groups','relations']} for s,d in summary['splits'].items()},indent=2))


if __name__ == '__main__':
    main()
