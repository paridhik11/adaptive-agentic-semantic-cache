# Day 3 research and development protocol

October 21, 2026. Recorded before the new QA experiment. These are development
experiments, not calibration, final evaluation, human review, or a passed gate.

## Related work and decisions

| Primary source | Relevant mechanism and limitation | Decision for this project |
| --- | --- | --- |
| [GPTCache, NLP-OSS 2023](https://aclanthology.org/2023.nlposs-1.24/) and [project](https://github.com/zilliztech/GPTCache) | Separates embedding retrieval, storage and similarity evaluation. The paper's discussion reports that available similarity evaluators do not adequately distinguish good and bad hits. | Retain modular retrieval/approval. Do not treat retrieval relevance as answer validity. No dependency transplant or paper performance claim. |
| [vCache, ICLR 2026, arXiv v5](https://arxiv.org/html/2502.03771v5) and [project](https://github.com/vcache-project/vCache) | Learns per-entry thresholds using response-equivalence observations from exploration. Open-ended comparison uses an LLM judge; analysis assumes iid observations and a sigmoid family. Its guarantee concerns agreement with fresh-model responses, not independently established factual truth or our accepted-group endpoint. | Useful later for calibration/feedback design. It cannot supply missing trustworthy labels or repair an unsuitable verifier by itself. Do not claim its guarantee for our implementation. |
| [Krites, arXiv 2602.13165v1](https://arxiv.org/html/2602.13165v1) | Promotes vetted static answers after off-path verification. Trace evaluation uses equivalence-class oracle labels and assumes verifier fidelity. | Preserve explicit provenance and admission quality. Do not import an oracle as a real verifier or make background remote calls. |
| [SmartCache, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fb74b63d225f846e6032bf3e3ab0f4ec-Paper-Conference.pdf) | Retrieves through conversational context and integrates KV state. | Retain full context/access scope; fixed-context text-answer caching is the current bounded product. KV integration is outside Day 3. |
| [QAFactEval, NAACL 2022](https://aclanthology.org/2022.naacl-main.187/) | QA-based factual consistency; answerability components matter, and QA/NLI signals can complement one another. | Test incoming-question-conditioned extraction rather than bidirectional entailment of artificial task statements. |
| [Q2, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.619/) | Generates questions and compares answer spans using NLI for grounded dialogue consistency. | Motivates source/answer comparison, but our smaller diagnostic does not reproduce Q2: it uses the incoming question and strict span agreement. |

This synthesis is a design inference from the sources, not proof that these
methods solve our cache. No third-party implementation is copied. Paper metrics
are not transferred to this project.

## Candidate fixed before execution

Use pinned `deepset/minilm-uncased-squad2` revision
`934656cdda79824eabf503ed56e15c01ddbdbe3f` for extractive QA; its training task is
SQuAD 2.0, not semantic-cache approval. Retain pinned DeBERTa NLI for actual
answer-claim support and MiniLM for cache retrieval.

1. Enforce identical tenant/request/system/source/generation scope and complete
   response admission; reject explicit changed requirements with existing guards.
2. Extract source answers for the original and incoming questions and extract an
   answer from the actual cached response for the incoming question. Three local
   QA calls. Source extraction never reads the cached answer.
3. Require nonempty normalized spans to agree, preserving code, numbers and
   direction-sensitive tokens. This is a restrictive diagnostic, not a theorem
   of answer equivalence. No request-pair registry or approved alias list.
4. Select the sentence enclosing the incoming source span; require its API calls
   and numeric constants in the cached answer. This can overreject alternative
   valid methods and does not prove complete applicability.
5. Check all actual answer claims against the full context using local NLI.
   Record all checks, including after rejection; no silent input truncation.
6. Fixed initial diagnostic point: QA span-versus-null logit margin >=0 for all
   three extractions, each support entailment >=0.95. Record curves on the fixed
   Cartesian grid QA margin [-5,0,2,5,10], NLI [0.5,0.75,0.9,0.95,0.99]. The
   curve is descriptive development evidence; no best point is promoted from it.

The application-facing policy always abstains. A socket-blocked fixture runner
may explicitly map recommendations to decisions in isolated cache experiments.
This does not enable production serving. Local work and setup costs are counted.
No live generation or remote judge calls.

## Data and exit accounting

Replay all 45 previous authored development cases unchanged. Add 30 authored
cases in 10 separate development groups (lowercase, uppercase, strip, split,
join, replace, absolute value, rounding, reversed copy, all-elements predicate),
with one paraphrase, one changed-task negative and one wrong-answer negative
each. These are builder-authored expectations and source fixtures, not final
labels, independent review, authenticated documentation, or a sealed set.
Report cohorts separately. Preserve all failures and old baselines.

Day 3 requires: researched candidate implementation; measured development
precision/coverage counts across predeclared settings; complete failure categories;
actual cached-answer bindings; no similarity-only serving; regression checks and
reproduction commands. Record separately whether the candidate is useful enough
to advance. Engineering completion cannot establish quality completion. If the
candidate fails, say so and keep Day 4 selection/calibration blocked; do not
rename a failed model as a working general semantic cache.

Day 10 remains a delivery target, with installation/demo/documentation and honest
supported scope. It is not a promise of general-domain accuracy, independent
reviewer availability or a passing final evaluation.

## Development revision after QA v1 failure

The first run accepted zero of 24 positives and zero of 51 negatives. Its strict
text spans are not stable representations of Python calls: extraction can return
`str`, `str(value)` or `Use str(value)`. Preserve `day3-qa-v1.json` unchanged.

Test a separate `api-anchor` representation, with the same QA margin 0 and NLI
0.95 initial point and the same grid. When an extracted span overlaps exactly
one maximal parsable call expression, compare that whole AST expression. Preserve
function names, arguments, constants and keywords. Ambiguous or non-code spans
retain strict comparison. Use no function/question allowlist. This is a declared
engineering development replay of the same exposed cases, not fresh evidence.
This change does not cure wrong extraction, low answerability margins, changed
requirements or unsupported claims. All such failures must remain visible.

## Procedural-question representation hypothesis

The call-anchor variant accepted 5/24 positives and 0/51 negatives at the fixed
initial point; the relaxed QA margin -5 / NLI 0.95 accepted 10 positives and 3
negatives. Do not promote that relaxed setting. Many remaining extractions treat
procedural questions as unanswerable; SQuAD extraction is not trained as a
procedural-answer judge.

Test a separate generic `function` question form: rewrite only the leading
`How do/can/should I <verb phrase>?` as `Which function can <verb phrase>?`.
Other forms remain unchanged. Keep raw questions for scope, guards, hashes and
cache retrieval. Apply the same rewrite to all three QA requests; do not insert
an expected answer, API name, domain alias or per-case exception. Retain the
same initial thresholds, grid, fixtures, call anchoring and claim checks.
This remains an exposed development replay, not a new generalization test.

## Hybrid preservation gate hypothesis

The generic function-question variant accepted 13 positives and 4 changed-task
negatives at its initial point; actual fixture retrieval served 7 semantic hits,
2 unsafe. It fails. Asking an extractive reader to name a function can make an
unrelated source look answerable; source support does not establish intent.

Test one hybrid with an additional mandatory original/incoming bidirectional
NLI task-entailment gate at the unchanged 0.95 threshold, using the existing
`task_statement` transformation. Retain all QA and answer-claim checks. Observe
both intent scores even after rejection. Grid plots vary QA margin and answer
support only; the intent floor stays fixed at 0.95. This tests whether combining
the two failing representations recovers useful development coverage safely.
Do not tune intent thresholds on these results or claim either representation
is now validated. If no candidate is suitable, Day 3 quality remains blocked.
