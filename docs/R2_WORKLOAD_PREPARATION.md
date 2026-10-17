# R2 workload fit and data preparation

## Decision

The three suites stay separate. OASST1 supplies chronological, crowdsourced
first-turn prompts for descriptive prevalence; the CPython Programming FAQ
supplies a versioned technical Q&A reference set for development; a handwritten
suite supplies adversarial stress cases. WildChat could not be retrieved
completely and its dataset card and retained `LICENSE.md` show different
license text, so it is not used. None of these sources alone represents general
production API traffic.

## Source and rights

The FAQ source and full CPython PSF license are version-pinned and checked by
SHA-256. The official Python licensing page identifies PSF License Version 2
for Python documentation and dual licensing for code examples. The derived
corpus preserves question and answer text, including code examples, while
normalizing reStructuredText inline markup and adding FAQ IDs, source-line
references, group/split metadata and workload wrappers. A short change notice
is included here and the upstream license is retained at
`data/licenses/CPYTHON_PSF_LICENSE.txt`.

For the prevalence-only alternative, OASST1 is pinned by commit and the
Apache-2.0 license file and selected Parquet shard are checked by SHA-256. The
dataset card describes conversations collected on the OpenAssistant website;
the data contains message-tree IDs, timestamps, user IDs and a synthetic flag.
The audit uses root prompter messages, excludes deleted or synthetic rows,
sorts by timestamp, and deduplicates by message tree. It exports aggregate
counts only; raw text and user IDs stay in the local ignored source directory.

WildChat's incomplete viewer attempts were not used. Its README currently says
ODC-BY and says the change retroactively covers previous downloads, while the
repository's retained `LICENSE.md` still contains AI2 ImpACT text. We do not
redistribute or analyze new rows until that discrepancy is resolved. Attempts
are recorded in `docs/audit/R2_COLLECTION_ATTEMPTS.json`.

## Grouping and splits

Preparation finds FAQ headings directly in the pinned reStructuredText file
and assigns entire topical sections to one split. The current mapping places
General Questions, Core Language and Performance in train; Numbers and strings
and Sequences in dev; Objects and Modules in reserved test. A question, its
reference answer, and its constructed repeat always share one FAQ group and one
split. This prevents those generated repeats from leaking across splits. The
section-level partition is intentionally visible and small; it is a grouped
development/reference design, not a statistical guarantee of independence.

The script emits 1 seed and 1 constructed exact-repeat request for each FAQ
entry. This 50% designed repeat share is only a cache-mechanics workload. It
must never be pooled with natural traffic or used as an expected deployment hit
rate. `answer_reuse_gold` remains null because a question pair or source answer
does not prove that a live model response is safely substitutable. No pretrained
checkpoint exposure is asserted; possible pretraining contamination from
public Python documentation is unknown.

The authored adversarial suite covers changed entities, negation, quantities,
units, direction, code input, versions, temporal requests, requested depth,
conversation context and open-ended generation. Expected non-reuse labels are
test intentions written by the project author, not independently validated
correctness labels. The suite is stored separately from the FAQ reference
workload.

## Natural-repeat observation

The pinned OASST1 train shard contains 9,845 eligible first-turn roots after
tree deduplication. In timestamp order across the shard's available period,
there are 66 byte-for-byte repeated prompts and 83 repeats after NFKC, casefold
and whitespace normalization. The normalized prompts include 3,482 English
entries and prompts in other languages. None of the repeated normalized
prompts occur under the same source user ID in this corpus. These are raw text
repeat opportunities only. A chronology check finds 49 exact and 57 normalized
repeat occurrences whose earlier source thread has an assistant reply record
timestamped before the repeat. This indicates response-record availability
only; no answer text was reviewed or compared, so none are verified safe-reuse
opportunities. Even a hypothetical single shared scope has at most 83
normalized repeat opportunities in this shard before context checks,
correctness review, eligibility, eviction or miss behavior, and 57 have an
earlier response record available by timestamp. The zero repeats within source
user IDs further weakens the case for a personal-user cache in this dataset,
although those IDs are not deployment tenants. No answers were reviewed, and
the dataset is crowdsourced rather than production API traffic. This is not a
safe hit rate or expected savings estimate. Raw prompt strings and IDs are
omitted from the committed summary.

## Reproduction

```bash
python scripts/prepare_python_faq.py \
  --source-dir runs/r2-python-source-recompute \
  --output-dir runs/r2-python-recompute \
  --download
```

The downloader fetches only the pinned FAQ and license files, verifies their
byte lengths and SHA-256 values, checks group leakage, and writes the derived
corpus, workload and summary under the ignored `runs/` directory. It makes no
model/API calls. The local output directory used for this phase is
`runs/r2-python-prepared-v1`; its corpus/workload hashes and per-split counts
are ledgered in `docs/FACTS.md`.

The OASST1 prevalence audit uses the optional Parquet dependency and writes no
prompt text or user ID to the summary:

```bash
pip install '.[data]'
python scripts/prepare_oasst_natural.py \
  --source-dir runs/r2-oasst-source-recompute \
  --output-dir runs/r2-oasst-recompute \
  --limit 9845 \
  --download
```

The resulting counts-only summary is `docs/audit/R2_OASST_NATURAL_REPETITION.json`.

## What remains unknown

There is no production traffic sample, no independent review of a generated
answer against these reference answers, no safe-reuse precision, and
no credible latency or token-saving estimate for this domain. We have not yet
sampled both accepted and missed prior-answer opportunities. The WildChat
collection and license-file mismatch remain a documented source blocker;
OASST1 is a descriptive alternative, not production traffic. R3 can use the
grouped FAQ material for matcher development, but any quality result must be
called development evidence and the future
untouched evaluation still requires a separate pre-registration.
