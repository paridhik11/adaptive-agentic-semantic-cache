# Complete human review and separate model diagnostics

2026-11-01. Latest complete user-chat review supersedes prior partial responses.
The original human form, model questions and model judgments remain unchanged.

| Human judgment of the actual supplied excerpt answer | Items |
| --- | --- |
| Sufficient (Y) | 14 |
| Insufficient (N) | 15 |
| Uncertain (?) | 1 |
| Missing | 0 |

Evidence type: declared human review of model-authored questions over three
correlated exposed FAQ sources. This is not independently human-authored wording,
new LLM-generated answers, a representative workload or a final evaluation.
Human identity, independence and blinding are declared/not externally verified.

The separate model-assisted author/self-review agrees on 29 definite human
judgments. No opposite binary judgments are observed. H05 remains unresolved:
the human notes that “octal ten” could mean the numeral `0o10` or decimal ten
written as `0o12`; the model interprets it as `0o10` and judges the supplied
example sufficient. Human label stays null. The model label stays true. Neither
is overwritten to improve agreement. This is uncertainty, not an observed
binary human/model contradiction.

These are answer-applicability labels, not system acceptance predictions.
There is no measured cache-policy accuracy, precision, coverage or savings in
this comparison. Agreement counts are descriptive; three builder-exposed source
topics and model-authored sampling do not establish population independence.
No accuracy percentage/confidence guarantee is inferred.

Machine-readable human rows and reasons:
`artifacts/documentation_rebuild/group4/human-review-v1/human-responses.jsonl`.
The provided N/? explanations and H17 explanation are preserved. Where the user
gave no reason, a bookkeeping note explicitly says no rationale was supplied;
`human_rationale_provided=false` prevents that note becoming invented human text.
Relations remain null rather than copying model judgments. Source/question IDs
and hashes bind each H-code to the original form. No model labels were imported
into human fields. Raw records are kept locally/private; no push was performed.

Reproduce into a fresh directory:

```bash
python scripts/reconcile_document_reviews.py --packet data/document_product_review/human_v1 --labels data/document_product_review/human_return_v1/labels-from-chat.json --reviewer user-declared-human-reviewer --model data/document_product_review/model_assisted_v1/responses.jsonl --output-dir NEW_RECONCILIATION
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_document_review_reconciliation.py tests/test_document_review_packet.py
```

16 focused engineering/schema tests passed in 0.35s. They test preservation of
uncertainty, no invented rationale, ID/binding mismatches and provenance. They
are not 16 answer-quality observations. Comparison, schema validation and input
hashes are retained under `human-review-v1/`.

No more answers are needed from the user for this packet. General unseen-pair
approval still needs an applicable candidate policy and evaluation; these labels
do not automatically create one. No threshold/model change, training, fresh-set
tuning, live call, serving promotion or locked-final evaluation occurred here.
This exposed diagnostic must not be retuned and then advertised as a fresh final
test. Product QA can continue separately from that policy gap.
