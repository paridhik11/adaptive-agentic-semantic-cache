# Independent wording and answer-applicability request

Write the 30 new questions in requests.jsonl yourself, using sources.json.
Do not inspect classifier code, thresholds, fixture approvals or predictions.
Use natural wording and rearranged meaning, not just spelling changes. For each
source, aim for five equivalent requests and five plausible mismatches: changed
conversion direction, added requirement, different type, changed number base or
index/slice boundary as applicable. Authoring requests are NOT assigned labels.

Judge the actual complete cached_answer against the new question and source:
answer_sufficient=true only if it directly resolves every requested requirement,
is supported by this source, and needs no changed code, facts or extra reasoning.
Set false if a material requirement is absent or contradicted; leave null if
uncertain and explain. A source passage that merely contains related information
is not automatically sufficient. Relation: same_intent, more_specific,
more_general, different_intent or ambiguous. Keep identifiers unchanged.

Fill author_id, reviewer_id and rationale. Prefer a second person for review;
declare if author and reviewer are the same. evidence_type must be human_authored
or model_assisted. Model-assisted work is NOT independent human ground truth.
No fake human identities. Do not include personal data or API keys.

Return the completed file; keep sources unchanged. This is a small purposive
diagnostic of reference-excerpt answers, NOT a final split or representative
savings workload. No actual new LLM-generated answer was collected here. These
three source topics cannot establish broad paraphrase safety. Missing labels
remain null; builder must not fill them and claim independent evidence.
