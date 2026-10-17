# R2 human label rubric for answer reuse

This rubric is for future labels on actual prior-answer opportunities. The
current reference corpus has no generated answer-review labels.

## Unit and label

One unit is an incoming request, the exact prior request and response that the
cache would retrieve, the fresh response for the incoming request, and the
request context available to the serving system at that moment. Reviewers see
all relevant system/user messages and generation scope, but cache arm and
matcher score should be hidden where practical.

Each reviewer records:

- `reuse_label`: `safe`, `unsafe`, or `uncertain`.
- `fresh_correct`: `yes`, `no`, or `uncertain`, assessed independently against
  the domain reference or explicit task rubric.
- `cached_correct`: the same values, assessed against the incoming request.
- `material_difference`: a short reason such as entity, quantity/unit,
  direction/negation, version/time, code/input, requested format/depth, missing
  conversation context, or factual error.
- `reviewer_id`, `review_timestamp`, and `notes`.

## Decision rules

Mark `safe` only when the cached answer satisfies the incoming request with no
material loss in facts, constraints, freshness, context, or requested output.
Topic similarity and agreement with the fresh model response are insufficient.
The cached answer may be unsafe even when the fresh answer is also wrong. Use
`uncertain` when the source, answer, prompt context, or currentness cannot be
verified. Do not convert missing answers or incomplete generations into safe
labels.

## Sampling and adjudication

Sample accepted hits and candidate misses, including near-threshold rejections,
by topic and source. Preserve their selection probabilities so any weighted
estimate can account for unequal sampling. Do not review only accepted hits.
Two reviewers label independently; disagreements go to a third reviewer or
remain uncertain. Record agreement and coverage separately from precision.
Group repeated paraphrases by their source answer so confidence intervals do
not treat variants of one answer as independent observations.

The future evaluation protocol must freeze the workload, sampling plan, rubric,
blind, review counts and analysis before the untouched evaluation begins.
