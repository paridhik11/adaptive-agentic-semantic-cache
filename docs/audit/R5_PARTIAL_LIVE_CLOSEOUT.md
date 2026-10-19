# R5 partial live evidence inspection — October 20, 2026

The uploaded archive is internally consistent: all manifest entries, pinned
request bindings, response diagnostics, blank review records and private review
mappings verify. The raw journal was not uploaded, so its original snapshot hash
is not independently recomputed. This is verification of user-supplied live
response records, not independent authentication of provider execution.

| Run | Saved provider attempts | Complete responses | Truncated responses |
| --- | ---: | ---: | ---: |
| Original v1 | 6 | 5 | 1 |
| Revised v2 | 10 | 10 | 0 |

The original run had already spent six attempts, not just the final truncated
attempt shown in the console. The revision's combined-cap rule therefore left
ten attempts. Its expected successful fixture path needs fourteen provider calls
(eight off and six exact), so it could not finish within that remaining allowance.
The shortfall should have been surfaced before execution. This is a planning/
preflight limitation, not evidence that the shared quota journal malfunctioned.
The old cap is not retroactively increased, and no fresh run is authorized here.

The revised sample contains ten complete responses with no recorded truncation.
That does not establish reliable completion on new requests. The different prompt
profiles remain separate; do not pool them into a completed paired comparison.
No live latency or call-savings result is inferred from the export.

Fifteen complete returned responses yield twelve unique request-bound review
items. Identical answers for the same request are deduplicated; these are not
independent safety observations. Every human label remains null. The reviewer
packet preserves the original template bytes and adds plain-language instructions
and a readable copy, without run metadata, private mappings or model judgments.
Raw answers and the review packet are kept private, not committed to GitHub.

## Model-assisted observations, not human ground truth

The inspecting assistant saw metadata and all answers. This is neither blinded
nor independent human review. No binary correctness/support labels were assigned.

- Both completed authored-negation answers need review. One treats the existence
  of the reference answer as invalidating the user's need to ask the question.
  The other imports claims about when non-decimal notation is necessary, which
  are not supported by the supplied excerpt.
- The synthetic negation wrapper itself is awkward: the underlying how-to
  question does not assert that non-decimal notation is necessary. Review should
  consider prompt ambiguity, not invent a known semantic-negation gold label.
- Straightforward syntax and tuple/list conversion answers contain the expected
  core information, but some add claims absent from the narrow reference. Human
  review must distinguish factual correctness from source support.

## Verification and reproduction

`scripts/audit_r5_partial.py` checks the uploaded artifact and writes only aggregate
results. It rejects changed manifest bytes, changed blank labels, altered request
bindings, altered responses and changed summary counts in five negative checks
on temporary copies. Original input bytes are unchanged. No external model calls.
The full runtime suite was not rerun for this read-only evidence audit.

```bash
python scripts/audit_r5_partial.py --archive PATH_TO_PRIVATE_ZIP --corpus runs/r5-corpus/corpus.json --output runs/r5-partial-audit-recomputed.json
python scripts/check_numeric_tokens.py
git diff --check
```

Aggregate evidence: `R5_PARTIAL_LIVE_AUDIT.json`. Source archive SHA-256 is retained
there; the private archive is not in Git. Next step: review the twelve items, keep
self/model-assisted review distinct from independent human review, then define
any further collection budget/protocol prospectively. Do not rerun stopped pilots.
