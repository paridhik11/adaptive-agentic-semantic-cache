# Dataset roles for credible recovery measurements

This is the acquisition/design plan, not a claim that fresh evaluation data has
been downloaded, annotated or sealed. The existing converted development stream
is already exposed QQP/PAWS training data. Never promote it to a test set.

| Source | Intended role | What its labels do not establish | Primary source |
| --- | --- | --- | --- |
| QQP / Quora duplicate questions | Natural question paraphrases and hard negatives for development; retain original question IDs where available | A duplicate label does not prove that a particular generated answer is safe to reuse | [Sentence Transformers dataset card](https://huggingface.co/datasets/sentence-transformers/quora-duplicates/blob/main/README.md) |
| PAWS | Structural/word-order adversarial checks; partition source sentences and all derived variants together | Pair paraphrase labels are not actual-answer judgments; PAWS-QQP distribution differs from PAWS-Wiki | [Original repository](https://github.com/google-research-datasets/paws), [paper](https://arxiv.org/abs/1904.01130) |
| CQADupStack / BEIR | Technical-domain retrieval probes; reserve entire forums before tuning when testing domain transfer | Retrieval relevance or duplicate links are not sufficient answer-equivalence labels | [Maintainer dataset card](https://huggingface.co/datasets/BeIR/cqadupstack), [BEIR registry](https://github.com/beir-cellar/beir/wiki/Datasets-available) |
| Super-NaturalInstructions | Candidate reference-answer tasks to complement query-pair datasets; select tasks where output correctness can be assessed against supplied references | Many tasks have multiple valid outputs; reference equality alone need not establish quality, and repetitions must not be described as organic traffic | [Original repository](https://github.com/allenai/natural-instructions), [paper](https://arxiv.org/abs/2204.07705) |

Primary source pages checked during this implementation. The PAWS maintainers
note that original PAWS-QQP examples require reconstruction because the original
QQP text is not directly redistributed there. Natural Instructions task instances
retain their underlying dataset terms. Record source-specific license information
before redistributing samples; a converter's `license` string is provenance, not
verification or permission.

## Required acquisition manifest

For each downloaded artifact record its URL, immutable revision, retrieval date,
byte SHA-256, source terms, configuration/split, original IDs and transformations.
Record all prior training/calibration exposure. Current legacy files lack complete
original IDs, so normalized-text groups are a development fallback only.

Group all variants of a source question/thread together before allocating splits.
Use the union of known training/development text and IDs when checking leakage.
Do not randomly sample the same forum/pool twice and call those sets disjoint.
Keep domain-held-out and within-domain conclusions separate.

## Answer annotation and comparison

Generate and save actual candidate answers, then evaluate the answer actually
retrieved for each incoming request. Include context, model configuration and
fresh-baseline answer in each review binding. Reference answers guide human
review; independently configured model judges can assist but cannot silently
replace independent ground truth. Review a sample for agreement and resolve
contradictions before final scoring.

Keep natural traffic, constructed repeat streams, and adversarial cases in
separate result strata. A synthetic repeat distribution can test mechanics but
cannot estimate customer savings. Predefine the number of groups, annotation
budget, non-exact accepted-hit requirement, confidence bound, quality margin,
endpoint definitions and stopping rule before acquiring a new sealed evaluation.

The immediate engineering decision from the development fixture run is to keep
semantic serving disabled and use this evaluation contract for future candidates.
The tiny exact-only hit count does not justify a speedup or cost-saving claim.
