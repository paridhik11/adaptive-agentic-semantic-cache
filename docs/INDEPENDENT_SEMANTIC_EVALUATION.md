# When can a semantic cache safely reuse an answer?

**Use exact reuse for the interview release.** In the reserved ComQA panel, a relative relevance rule recovered **20/128** eligible paraphrases versus **4/128** for the existing full-question verifier. It also introduced **one wrong substitution** and accepted **11/128** deliberately corrupted entries. The branch launcher now selects exact-only live reuse by default. The narrower pre-existing matchers remain available with explicit flags. The broader relative rule is available only in the offline study. [Held-out decisions](../artifacts/independent_semantic/holdout.json) · [Wrong-answer stress](../artifacts/independent_semantic/holdout-answer-audit.json)

## What was tested

The live chat can call a provider on a miss. Its lightweight policy recognizes specific question forms. Its optional neural route retrieves short definitions and verifies candidates locally. The documentation workbench has a separate before-agent lookup that binds approved answers to principal, tenant, task context and source revision. A miss runs planning, document tools, drafting and review. The research arms in this study search full questions in a fixed snapshot with trusted input labels. They do not serve users. The historical 4,000 executions measure a third workload with scripted calls and synthetic tokens. [Runtime architecture](SEMANTIC_UPGRADE_RESULTS.md) · [Historical campaign](INTERVIEW_V1.md)

The hypotheses were that the intended question might rank below first place, that word overlap might help retrieval, and that a passage scorer trained for search might undervalue short factual answers. A fourth hypothesis was that preserving answer type and tool requirements would reject a substitution that question similarity alone would accept.

## Data and labels

| Panel | Unit of analysis | What counts as success |
| --- | --- | --- |
| ComQA | Two different questions in a publisher paraphrase cluster, with the actual normalized annotated answer stored | Return that answer from a primed shard. In the other shard all annotated answer sets are disjoint. Alias and date-granularity errors remain possible. |
| PAWS-Wiki and GLUE QQP | A publisher-labeled pair of sentences or questions | Correctly classify paraphrase or duplicate status. Neither panel supplies a saved answer. |
| Python 3.12.10 published examples | A literal expression, its documented displayed output, and the same expression enclosed in parentheses | Return the documented output with the version and operation preserved. These are controlled syntax variants. |
| Python 3.12.10 FAQ supplement | A publisher question with its full section answer, plus a controlled swap of “How do I” and “How can I” | Reuse the full answer. This positive-only, single-entry panel measures coverage on technical guidance. |
| Corrupted-entry stress | A matching ComQA or Python question deliberately primed with another group's answer | Abstain. The donor is the nearest other seed by cosine, preferring the same explicit interrogative class where possible. |

ComQA's 3,966 official training clusters supplied the new sample. The public development and test files were used to exclude prior questions and answer strings. Eligible clusters contribute their first two distinct questions. Greedy deduplication prevents a selected question or answer from crossing groups. A salted hash assigns entire clusters to development or holdout. There are 128 groups per split and 64 seeds per shard. Each group contributes a positive probe, the same probe against the other shard, and an exact-repeat control. A live policy can refuse a priming request. A research policy receives the reference gallery by construction. All arms start from the same input seeds and each probe reads its original snapshot. This measures the impact of admission as well as lookup. [ComQA publisher and paper](https://qa.mpi-inf.mpg.de/comqa/) · [Manifest and input hashes](../artifacts/independent_semantic/manifest.json)

PAWS contributes 8,000 source test rows and QQP contributes 40,430 validation rows. The sampler joins pairs that share either normalized sentence into connected components, excludes components intersecting four known repository training or evaluation files, and takes at most one pair per component. Each panel has 128 positives and 128 negatives per split. Precision therefore reflects a balanced sample. Prior repository use and possible upstream model-training overlap, especially the Quora-trained pair scorer, limit claims of independence. ComQA has a publisher research download without an identified redistribution licence. GLUE refers users to the original Quora terms. Google's PAWS terms permit use with acknowledgement appreciated. Only derived scores, IDs and aggregates are distributed here. [PAWS terms](https://github.com/google-research-datasets/paws/blob/master/LICENSE) · [GLUE card](https://huggingface.co/datasets/nyu-mll/glue) · [Quora release](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

The Python data comes directly from versioned CPython documentation. Distinct source documents stay in one split. The output panel contains 25 development and nine holdout examples drawn from seven and six documents respectively. It excludes names, function calls and context-dependent inputs and uses the published literal output. No documentation code executes. The FAQ supplement selects 40 complete answers under a length cap of 850 characters and preserves the title's subject and constraints. It was added as a separate diagnostic after development selection. Natural Python support traffic and independent human grading of generated procedural answers remain open. [CPython documentation release](https://github.com/python/cpython/tree/v3.12.10/Doc) · [Python licence and examples](https://docs.python.org/3.12/license.html)

## Design and decision rule

No models were trained. MiniLM encodes full questions as normalized vectors. Their dot product is cosine similarity. The pair cross-encoder reads the two questions jointly in both orders. The passage cross-encoder reads the incoming question and the stored answer together. All three checkpoint revisions are in the result JSON. Their scores have no calibrated interpretation as the probability that a cached answer is correct.

The full-question baseline checks the nearest seed, cosine ≥0.84, both pair scores ≥0.80, structural requirements and passage score ≥0.85. Development compared a top-five dense search, a hybrid ranking of 75% dense and 25% word TF-IDF, a relative passage gate, and additional answer-type/tool checks. The multi-candidate arms abstain with a close independently qualified alternative or an unexamined candidate within 0.035 of the selected score. Each policy preserves the baseline requirement checks. The relative gate accepts an additional short answer only when it has at most 12 words, both pair scores reach 0.95, passage score reaches 0.01 and its logit is within 0.5 below the original question's score on that answer. This last check tests transfer of relevance. It assumes the saved source answer was adequate.

Only the relative gate qualified on development. The protocol was committed at ee0772a and the selection at ba3755b before holdout scoring. No holdout thresholds were changed. The candidate remained an offline experiment. The serving decision required zero newly incorrect decisions, a lower exact 95% precision bound of at least 98%, reproducible source suitability, and rejection of all deliberately corrupted answers. [Frozen protocol](INDEPENDENT_SEMANTIC_PROTOCOL.md) · [Development selection](../artifacts/independent_semantic/selection.json)

## Answer-level results

Each ComQA split has 128 probes with a known suitable answer in scope and 128 with that answer absent. Precision divides correct semantic hits by all semantic hits. Safe recall divides correct semantic hits by 128 available answers. Coverage counts any semantic hit across all 256 probes. The dash indicates no semantic hits. Correctness on the corruption panel uses a separate denominator.

| Policy | Correct development /128 | Correct holdout /128 | Wrong holdout /256 | Precision | Safe recall | Coverage / fallback | Corrupted hits /128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Live exact | 0 | 0 | 0 | — | 0.0% | 0.0% / 100.0% | 0 |
| Live lightweight rules | 0 | 3 | 0 | 100.0% | 2.3% | 1.2% / 98.8% | unmeasured |
| A. Live optional models | 0 | 3 | 0 | 100.0% | 2.3% | 1.2% / 98.8% | 2 |
| Full-question top one | 2 | 4 | 0 | 100.0% | 3.1% | 1.6% / 98.4% | 0 |
| B. Dense top five | 2 | 4 | 0 | 100.0% | 3.1% | 1.6% / 98.4% | 0 |
| C. Hybrid top five | 2 | 4 | 0 | 100.0% | 3.1% | 1.6% / 98.4% | 0 |
| D. Relative passage | 20 | 20 | 1 | 95.2% | 15.6% | 8.2% / 91.8% | 11 |
| E. Answer type and tool | 2 | 4 | 0 | 100.0% | 3.1% | 1.6% / 98.4% | 0 |
| F. Development selection (D) | 20 | 20 | 1 | 95.2% | 15.6% | 8.2% / 91.8% | 11 |

The selected rule gained **12.5 percentage points of safe recall**. Its exact 95% recall interval was **9.81%–23.10%** versus **0.86%–7.81%** for top one. Precision was 20/21, **95.24%** with a **76.18%–99.88%** interval. The baseline's 4/4 precision has a **39.76%–100%** interval. Baseline 0/128 absent-shard errors leaves a **2.84%** upper bound. The corruption result was **8.59%** (11/128, 95% interval **4.37%–14.86%**). Development had already shown 9/128 corrupted hits against 0/128 for the top-one verifier. These are two-sided Clopper–Pearson intervals under independence. Shard sharing, related questions and paired conditions introduce dependence. The constructed 50% answer-availability mix also affects precision. [Development](../artifacts/independent_semantic/development.json) · [Holdout](../artifacts/independent_semantic/holdout.json) · [Development corruption](../artifacts/independent_semantic/development-answer-audit.json)

## Where the decisions failed

- **Retrieval.** The intended answer ranked first for 127/128 held-out questions with the answer present, and second for the remaining question. Top-five and hybrid retrieval each captured the same four correct answers as the top-one verifier. This 64-seed gallery leaves limited room to extrapolate to large indexes.
- **Verification.** The original top-one misses comprised 73 passage rejections, 27 question-pair rejections, 18 low cosines, five numeric or version checks and one intended candidate outside top one. With the relative rule the counts were 57 passage, 28 pair, 18 cosine and five numeric or version. For the Budapest river paraphrase, the intended answer ranked first, cosine was 0.979, both pair scores exceeded 0.995 and the bare answer’s passage score was 0.085. The relative gate recovered it.
- **Wrong substitution.** A probe asking when Jefferson served as vice president under Adams retrieved a question about Adams himself serving as vice president. It scored 0.857 cosine and 0.976 / 0.989 on the question pair. The stored answer reached 0.286 passage relevance. The relative gate accepted it because the original question also scored poorly on that answer. The structural guard missed the change in who held office. The candidate's question type remained time-related.
- **Wrong stored content.** A matching question with a substituted answer can satisfy every question-side check. Eleven relative-gate stress hits passed all its thresholds. The existing optional live path also reused two corrupted entries. In the clean replay those same two pairs used a structured fact match and an ordered keyword match respectively. The tests exposed that a plausible date or name can have the right form and still be wrong. Exact-mode zeros on this *nonexact* stress panel do not validate an answer originally stored for an exact repeat.
- **Live routing.** With optional models, 55 present probes encountered the static-topic eligibility rule, six encountered freshness bypass, 13 received a fresh response that was not admitted, 45 generated and stored a fresh response, six reported no close candidate and three hit. Final status codes describe the route taken. The live trace supplies executed candidate checks. The separate full-gallery trace scores the intended candidate even if live routing skipped neural retrieval.
- **Historical 168.** Replaying the spent ComQA protocol exactly reproduced its aggregate artifact. Of its 168 matched candidates rejected at the passage stage, 130 had at most three answer words. The median passage score was 0.00169, with quartiles 0.00015 and 0.01995. That is evidence of a short-answer scoring mismatch. The corruption test shows why accepting a candidate solely because the original short answer also scored poorly creates risk.

The compressed traces identify the probe, shard and source, ranks under each retrieval method, cosine, guard result, bidirectional pair scores, source and incoming passage logits, annotation agreement and final policy reason. Join an event to the candidate records by probe and shard. The intended positive candidate has `matches_annotation=true`. Live-model checks that never ran remain absent. Independent human suitability for arbitrary generated answers remains unknown. [Holdout traces](../artifacts/independent_semantic/holdout-traces.json.gz) · [Spent 168-case audit](../artifacts/independent_semantic/spent-answer-audit.json)

## Python technical answers

The full-question baseline and selected variant captured **27/40** full FAQ answers on single-entry wording changes (67.5%, exact 95% interval 50.87%–81.43%). The existing optional live route captured 3/40. Of the 13 research misses, 12 failed the passage check and one failed numeric evidence. On the separate output panel **all research arms captured 0/9 held-out positive probes**. The numeric guard requires the answer text to repeat numbers from the request, including the Python version. A correct displayed value can omit that version. Precision was undefined with zero hits, and the recall interval extends up to 33.63%. A better test of version applicability would bind it to trusted source metadata. That design still needs validation. [FAQ evidence](../artifacts/independent_semantic/python-faq.json) · [Output-panel evidence](../artifacts/independent_semantic/holdout.json)

## Question equivalence

Each dataset below supplies 128 positive and 128 negative holdout pairs. The PAWS result explains why a high cosine or duplicate score can be risky when the same words occupy different roles.

| Holdout panel | Classifier | True positives | False positives | Precision | Recall | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PAWS | Cosine ≥0.84 | 126 | 124 | 50.4% | 98.4% | 3.1% |
| PAWS | Bidirectional pair ≥0.80 | 103 | 108 | 48.8% | 80.5% | 15.6% |
| PAWS | Cosine + pair + requirements | 101 | 103 | 49.5% | 78.9% | 19.5% |
| QQP | Cosine ≥0.84 | 85 | 20 | 81.0% | 66.4% | 84.4% |
| QQP | Bidirectional pair ≥0.80 | 104 | 6 | 94.5% | 81.2% | 95.3% |
| QQP | Cosine + pair + requirements | 68 | 4 | 94.4% | 53.1% | 96.9% |

The artifacts also contain development results, true negatives, false negatives and exact intervals for every precision, recall and specificity entry. These labels measure equivalence of the pair. They do not evaluate a saved answer. [Pair results and intervals](../artifacts/independent_semantic/holdout.json)

## Verification cost and fixture calls

The ComQA table counts three requests per group, excluding priming. A one-call fresh fixture would make 384 calls. The API rows execute the actual handler with a local transport and its admission restrictions. The research arms have 128 primed exact controls by construction. Their profile reruns real local models per request, counts genuine calls into an in-process fixture on fallback, and checks each decision against the batched score trace. Latencies below cover the 256 nonexact probes. Research timings exclude model loading and gallery priming. Concurrent CPU work makes these single-run timings descriptive. D and F are independent timing runs of the same decision policy.

| Policy | Exact /128 | Fixture calls /384 | Calls avoided /384 | Probe verification p50 / p95 ms | Probe end-to-end p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Live exact | 51 | 333 | 51 | 0.0 / 0.0 | 7.3 / 15.4 |
| Live lightweight rules | 51 | 330 | 54 | 0.3 / 45.1 | 20.3 / 77.3 |
| A. Live optional models | 51 | 330 | 54 | 0.3 / 18.7 | 11.1 / 32.5 |
| Full-question top one | 128 | 252 | 132 | 43.9 / 200.9 | 43.9 / 200.9 |
| B. Dense top five | 128 | 252 | 132 | 31.8 / 177.4 | 31.8 / 177.4 |
| C. Hybrid top five | 128 | 252 | 132 | 45.6 / 220.5 | 45.6 / 220.5 |
| D. Relative passage | 128 | 235 | 149 | 44.6 / 201.4 | 44.6 / 201.4 |
| E. Answer type and tool | 128 | 252 | 132 | 48.7 / 301.5 | 49.5 / 301.5 |
| F. Development selection (D) | 128 | 235 | 149 | 45.5 / 269.0 | 45.5 / 269.0 |

Compared within the research harness, the selected rule saved **17 additional fixture calls**, including **one on an incorrect answer**. It recovered 16 additional correct reuses. Broader real-world efficiency remains unmeasured. The study issued zero external inference calls. Provider latency, actual token savings, production throughput and monetary savings have no estimate here. Development and Python per-policy call counts and timing distributions are in the profile receipts. [Holdout profile](../artifacts/independent_semantic/holdout-profile.json) · [Development profile](../artifacts/independent_semantic/development-profile.json) · [Lightweight live baseline](../artifacts/independent_semantic/holdout-live-default.json)

## Replication and checks

The historical ComQA aggregate reproduced byte for byte. The interview report checker reconciled the original 4,000 scripted executions. The original 12-probe authored reserved fixture reproduced 24/14/9/14 transport calls for no-cache/exact/guarded/shadow and five suitable guarded hits with zero unsuitable hits. The 47.33% and 14.75% historical scripted-call reductions belong to the exact policy on repeat-heavy and broader workloads. [Historical receipts](SEMANTIC_UPGRADE_RESULTS.md) · [Historical cosine study](LIVE_CHAT_COSINE_EVALUATION.md)

The release regression suite covers paraphrases, reversed operations, tools, negation, numbers and versions, entity changes, reporting context, stale answers, source and tenant separation, ambiguity, revocation and expiry. New study tests cover relative scoring, guard precedence, ranking, ambiguity, denominators and the launch modes. A browser smoke test exercised fresh, exact repeat and reworded fresh requests in exact mode. The final local run passed 1,077 tests including offline model integrations. Four live-provider tests were excluded. The added study files, compressed traces and changed source were scanned for common token formats and private-key headers.

## Reproduce on Windows PowerShell

From your project checkout, fetch the experiment and install the study dependencies. Use the pinned model setup to populate the local cache once. The evaluation commands operate offline after the public inputs and checkpoints are downloaded. Each command writes into a separate local results directory to preserve the committed evidence.

```powershell
git fetch origin experiment/independent-semantic-evaluation
git switch experiment/independent-semantic-evaluation
py -3.12 -m venv .venv
$p = ".\.venv\Scripts\python.exe"
& $p -m pip install -r deploy\requirements-gateway.txt
& $p -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $p -m pip install "sentence-transformers>=5,<7" scipy scikit-learn pyarrow httpx pytest
& $p scripts\setup_live_chat_semantic.py
$data = "runs\independent-inputs"
$out = "runs\independent-results"
New-Item -Force -ItemType Directory $out | Out-Null
& $p scripts\independent_semantic_data.py $data "$out\manifest.json" --download
$env:HF_HUB_OFFLINE = "1"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
foreach ($split in "development","holdout") {
    & $p scripts\evaluate_independent_semantics.py $data $out --split $split
    & $p scripts\evaluate_independent_semantics.py $data $out --split $split --default-only
    & $p scripts\profile_independent_semantics.py $data $out --split $split
    & $p scripts\audit_independent_answer_failures.py $data $out --split $split
}
& $p scripts\evaluate_independent_python_faq.py $data $out
& $p scripts\audit_independent_answer_failures.py $data $out --split spent
& $p scripts\evaluate_chat_cosine.py "$data\comqa_test.json"
& $p scripts\evaluate_semantic_upgrade.py --revision study-reproduction --split reserved
& $p scripts\report_interview.py --check
```

For the full project test suite install the development dependencies. The explicit marker selection includes offline model integration tests.

```powershell
& $p -m pip install -e ".[dev]"
& $p -m pytest -o "addopts=" -m "not requires_llm_api" -q
```

## A 45-second interview explanation

“I built a cache for an LLM agent, then investigated why it kept regenerating answers to paraphrases. I measured whether we retrieved the right question and whether its saved answer still applied. On 128 held-out questions with a valid saved answer, retrieval ranked the right one first 127 times. Changing how we verified short answers raised correct reuse from four to twenty. But it also mixed up two vice presidents, and accepted eleven of 128 deliberately wrong saved answers. I kept exact reuse as the release default. The analysis showed which stage to improve and the risk of treating every saved call as a success.”

## Five interviewer questions

1. **Why use cosine and a second model?** Cosine gives an inexpensive nearest-question ranking once the gallery is embedded. A cross-encoder can read both inputs jointly. In this sample the right question almost always ranked first, yet passage verification rejected many suitable short answers.
2. **What exactly is the denominator of 15.6%?** Twenty correct semantic hits divided by 128 requests whose annotated answer existed in the shard. Across all 256 nonexact requests, the rule served 21 hits. One was wrong.
3. **How did you avoid tuning on the test?** I grouped by source, removed known prior exposure, specified candidate policies, selected on development and committed the rule before holdout scoring. Public pretraining exposure and unobserved aliases remain possible. No fresh human judgment was collected.
4. **Why would a risk analyst reject a fivefold increase in correct hits?** The held-out error changed the person whose term was being reported. Wrong saved content also reached users in the corruption stress simulation. The lower bound on precision was 76.18%, below the predeclared 98% requirement. Accuracy of the underlying stored record matters as much as retrieval.
5. **Did this cut real latency or cost?** I measured local verification time and counted fixture invocations. The rule avoided 17 additional fixture calls and one came from a wrong answer. Estimating production impact would require paired live measurements on representative traffic with independently reviewed answers.

For the published interview demo, `scripts/run_demo.py --interview --fresh` opens the offline documentation workbench. On this experiment branch `scripts/run_demo.py --live-chat` starts exact-only general chat. `--guarded-rules` explicitly enables the older lightweight rules, and `--experimental-semantic` enables the older guarded local models. The relative-score research policy runs only through the offline evaluator. [Launch instructions](LIVE_CHAT.md)
