# Live chat cosine evaluation

This offline evaluation uses the [ComQA public test split](https://qa.mpi-inf.mpg.de/comqa/comqa_test.json). The earlier live-chat replay used the development split and its annotated paraphrase clusters. The public test file provides individual questions and annotated answer lists. Exact agreement between these lists measures reuse of short annotated answers, including cases where different questions happen to have the same answer. This protocol evaluates answer reuse on the fixed dataset.

## Protocol

Pin the public test JSON by SHA-256. Strip surrounding whitespace and discard blank answers. Use Wikipedia-link titles, Unicode case folding and whitespace folding when forming answer sets. Keep the original question text, retaining the first occurrence of a case- and whitespace-normalized duplicate. For each distinct nonempty answer set, take its first question as a gallery seed. Take the second distinct question, if present, as its probe. Include singleton gallery seeds as distractors. For each probe, the hard negative is the gallery seed with highest cosine among those with disjoint answer sets. Source order resolves ties. Report the number of unique controls because several probes can share a hard negative.

Compare default live chat, live chat with its existing optional local models, top-one full-question cosine pooling, and full-question cosine plus verification. The application replay starts with an empty cache per probe. It submits its paired seed and then its probe, and separately pairs each hard negative's seed with that same probe in a second empty cache. The fixture returns the dataset's short answer. The cosine experiment uses one common gallery and evaluates its nearest seed as well as each probe's nearest disjoint-answer seed. Model checkpoints and the cosine, pair and passage cutoffs come from the current live chat. The full-question comparison uses cosine at least 0.84, the existing deterministic requirement guard, bidirectional pair scores at least 0.80, and a passage score at least 0.85 against the actual seed answer. The experiment scores single candidates, so it omits the second-candidate margin and live-chat eligibility rules. Keep the original cutoffs and report both coverage and wrong-answer reuse. Overlapping but unequal answer sets are recorded separately.

Use no provider calls. This is a controlled replay with annotated short answers. Natural answer generation, freshness, representative cache traffic, and production error rates require separate data. Grouped queries and reused controls limit the interpretation of binomial confidence intervals.

[ComQA paper](https://aclanthology.org/N19-1027/) · [RedisVL vector cache documentation](https://redis.io/docs/latest/develop/ai/redisvl/api/cache/) · [vCache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9559cd2116de7a8f5672eac3fcd232cc-Abstract-Conference.html)

## Results on `49fa7de`

The gallery had 1,434 seed questions. The 422 probes drew 362 unique hard-negative seeds. The app rows use a paired cache with one seed. The full-question rows search the common gallery.

| Retrieval and verification | Supported reuse out of 422 probes | Disjoint-answer hits in the main run | Disjoint-answer hits out of 422 hard negatives |
| --- | ---: | ---: | ---: |
| Default live chat | 0 | 0 | 1 |
| Optional local models in live chat | 0 | 0 | 1 |
| Full-question cosine at 0.84 | 279 | 19 | 47 |
| Full-question cosine with verification | 9 | 0 | 0 |

Full-question cosine had three additional hits with overlapping answer sets. One clearly incompatible match scored 0.9453: a question about where Janis Joplin died was retrieved for a question about her age at death. The verified experiment accepted 9/422 probes, with an exact 95% interval of 0.98%–4.01% under independence. Its zero observed errors on 422 hard negatives give a two-sided upper bound of 0.87% under independence. Among the nine accepted hits the corresponding upper bound is 33.63%. The cosine-only hard-negative rate was 47/422, with an exact 95% interval of 8.30%–14.53%. [Aggregate results and other intervals](../artifacts/live_chat/comqa_cosine_49fa7de.json)

The app's flagged reuse was a precision change from a date to its year. The stored ISO date was 1553-05-25 and the probe explicitly requested the year, whose annotation was 1553. The existing date-precision rule accepted that reuse. The benchmark retains the disjoint-string label to preserve the protocol. Fields named `wrong` in the JSON record annotated disjointness, and this review shows why the count needs interpretation.

The passage relevance cutoff rejected 168 candidates that had already passed both the cosine and question-pair checks and whose annotated answers matched. That identifies a coverage bottleneck for this short-answer workload. A broader cache should use cosine to retrieve candidates while preserving model, context and freshness boundaries. Its verifier needs fresh evaluation with actual stored answers and checks for changed entities, numbers, direction and answer requirements before a wider serving rollout.

## Reproduce

Set up the pinned local models using the instructions in [Live chat](LIVE_CHAT.md#enable-semantic-reuse-for-paraphrases). Then run from the repository in PowerShell.

```powershell
Invoke-WebRequest https://qa.mpi-inf.mpg.de/comqa/comqa_test.json -OutFile comqa_test.json
.\.venv\Scripts\python.exe -m pip install scipy httpx
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts\evaluate_chat_cosine.py comqa_test.json
```

The runner validates SHA-256 `b81e19c4108198781167ddb30d26172571e76cc8e684aabb181537a45e816867`. It writes aggregate counts and Clopper–Pearson intervals to stdout. The wrong-answer interval among hits uses only hits with disjoint or identical answer sets. The release artifact retains ambiguous overlaps as a separate count.
