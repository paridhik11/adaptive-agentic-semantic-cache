# Live general chat

Ask about risk management, SQL, or any other topic. New questions go to a free OpenRouter model. The local SQLite cache stores complete eligible answers for one hour. Automatic context handles short follow-ups such as “Give me an example” and can recover the scope of a ranked officeholder question from the preceding turn. Set **Context** to **Include recent turns** or **Standalone question** when you want explicit control.

## PowerShell

In your existing repository checkout:

```powershell
git fetch origin optimize-v2:refs/remotes/origin/optimize-v2
git switch optimize-v2
git pull --ff-only origin optimize-v2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-gateway.txt
.\.venv\Scripts\python.exe scripts\run_demo.py
```

If you already created `.venv`, keep it and run the install line after updating the branch to pick up the spelling dictionary. Stop any earlier server with **Ctrl+C** first. The launcher uses `OPENROUTER_API_KEY` when set, or prompts for it with hidden input. Open [Live chat](http://127.0.0.1:8765/) and look for **Live** next to the model name.

1. Ask **what is risk managemnet**. A cache miss shows **1 provider call**, elapsed time and completion tokens if reported by OpenRouter.
2. Select **Repeat as new question**. An eligible exact hit shows **0 provider calls**, **0 generated tokens**, and the new response time alongside the earlier fresh response.
3. Ask **what do u mean by risk managemgcnt** and then **risk management?** as standalone questions. If the stored answer opens by defining risk management, both show **Semantic hit · 0 provider calls**. An earlier exact answer enters the definition index when its original question is asked again.
4. Ask **What is risk measurement?** or **What is risk management? Give a worked example.** These take the fresh path. Try **Compare precision and recall** followed by **Precision vs recall?** to show comparison reuse. Reversing the terms, changing a term, or adding a population or period asks the model again.
5. Select **Benchmark analysis** to open the historical offline evidence and workload projections.

Definition reuse is enabled by default. The cache checks that the saved answer opens by defining the same short topic. It can account for a spelling variation in one long word and a singular or plural form of the final noun in the same prompt. For rare spellings outside the local dictionary, the answer opening can corroborate the intended topic and an incoming prompt can reuse it with the same spelling or the spelling used in the answer. The chat falls back to the model when the match is ambiguous.

If the three prompts already have separate exact answers, add `--directory runs\definition-reuse-demo` when starting the server to demonstrate the sequence with a separate cache. The provider quota journal stays shared.

The default is `nvidia/nemotron-3.5-lightning:free`. If its availability changes, use `--model` followed by an explicit model ID ending in `:free`. The provider enforces zero price and disables automatic fallback. The persistent quota journal uses the existing free-only allowance (30 chat attempts with shared daily and rolling-minute caps). Use `--journal <path>` to share your existing quota journal. Reopening the chat preserves the allowance and cache. Eligible cached answers can still be served when the quota is exhausted.

Live chat uses the model's knowledge without web search. Response time covers server-side processing through answer availability, including time waiting for the provider slot. Provider time starts after the slot opens and includes the quota journal. Cache hits for other questions can complete while the provider works.

## Check a spelling and plural variation

Ask **what do u mean by bioluminiscent beaches** and then **what is a bioluminiscent beach?**. When the first answer opens by describing a bioluminescent beach, the second uses that stored answer and shows **Semantic hit · 0 provider calls**. The text similarity shown on the receipt compares the ordered topic phrases. The cache also verifies the defined topic against the answer. **What is a phosphorescent beach?** asks for a different concept and calls the model.

The offline regression replay reused the seed answer on all 11 tested rephrasings. All 11 changed requests took the fresh path. [Recorded results](FACTS.md#live-chat-definition-replay)

## See a semantic hit in the default chat

Leave **Context** on **Automatic** and keep cache reuse enabled.

1. Ask **1st prime minister of India?** In a cold cache the model answers once.
2. Ask **1st prime minister?** The chat fills in India from the preceding question and displays the wording it used. Ask **India's first prime ministerr?** to see the same answer through an explicitly scoped paraphrase. Both semantic hits show **0 provider calls**.
3. Ask **Who was the first prime minister of Pakistan?** or **Who is the current prime minister of India?** These require fresh generation. In **Standalone question** mode, **1st prime minister?** asks for the missing country or organization. Choose **Automatic** to reply with the name.
4. Ask **How do I calculate sample variance?** and then **How is sample variance calculated?** The second reuses the first answer. Changing **sample** to **population** requires fresh generation.

The local checks compare the requested fact, named subject, rank, qualifiers and answer precision. They also recognize alternate names explicitly stated at the start of a stored definition, such as “A photo (short for photograph) is…”. Arbitrary rephrasings can still miss. The context handling follows the approach of making an incomplete question self-contained before retrieval. [Conversational query rewriting research](https://aclanthology.org/2022.emnlp-main.679/) · [Scoped semantic cache retrieval documentation](https://docs.redisvl.com/en/stable/api/cache.html)

## Keyword lookup

The default chat indexes the request type and ordered keywords when it saves an eligible answer. It recognizes common changes in framing, such as “How does a heat pump work?” and “Explain how a heat pump operates.” It also recognizes grammatical changes that preserve roles, such as asking who designed a building in either the active or passive voice.

Try **How to convert a tuple to a list?** and then **How can I convert to a list from a tuple?**. The cache recognizes the same source and destination and shows **Semantic hit · 0 provider calls** with the original question. **How to convert a list to a tuple?** reverses the operation and gets a fresh answer. Changed values, named subjects, word order that changes the request, and added instructions can also require a fresh answer.

On startup, the app adds keyword entries for eligible questions already in the semantic index. Those answers retain their original expiration. An older answer with only an exact-cache record can be indexed when you ask its original question again. No new dependencies or launcher flags are required.

### Replay evidence

A constructed offline replay ran the default chat against 41 requests in eight isolated groups. Relative to commit `1c595e4`, it moved from 0/16 to 16/16 semantic hits on intended paraphrases. All 17 changed requests called the fixture provider. Total fixture provider calls fell from 41 to 25. Reproduce the current behavior with `python -m pytest -q tests/test_live_chat.py -k factual_semantic_reuse`. The groups share an answer within each group, and their check results describe these fixtures.

In an authored keyword replay, the app matched 18/18 rephrasings across nine independent cache setups. The preceding revision `facd79c` matched 0/18. All 28 changed requests made fresh calls in both revisions. Total fixture calls fell from 55 to 37. Reproduce with `python -m pytest -q tests/test_live_chat.py -k "ordered_keyword or keyword_order or keyword_upgrade"`. The questions in each group share an answer. These deterministic counts describe the fixtures. [Recorded results](FACTS.md#live-chat-keyword-replay)

The ComQA development replay used the first two distinct questions in each of 135 eligible clusters. It used their supplied short answers and the most lexically similar question with disjoint annotated answers as a control per group. The keyword update matched **1/135 paraphrases**, up from **0/135** on `facd79c`. It returned **0/135 cache hits** on controls. The two-sided exact 95% intervals under independence are 0.02%–4.06% for 1/135 and 0%–2.70% for 0/135. Controls may recur across groups. ComQA includes complex natural factoid questions, and most still miss. [ComQA paper](https://aclanthology.org/N19-1027/) · [Dataset and annotations](https://qa.mpi-inf.mpg.de/comqa/)

```powershell
Invoke-WebRequest https://qa.mpi-inf.mpg.de/comqa/comqa_dev.json -OutFile comqa_dev.json
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe scripts\evaluate_chat_paraphrases.py comqa_dev.json
```

## Recover an interrupted provider request

If the browser reports an interrupted attempt, first stop every server using the same journal. Inspect the reservations without making a provider call:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_pilot_journal.py --journal runs\live-quota.sqlite --run-id general-chat-v1
```

For a confirmed interrupted attempt whose state is `reserved`, pass its ordinal. The command requires the reservation to be at least two minutes old and records it as spent.

```powershell
$ordinal = Read-Host 'Interrupted attempt ordinal'
.\.venv\Scripts\python.exe scripts\run_demo.py --journal runs\live-quota.sqlite --resolve-interrupted $ordinal
```

## Enable semantic reuse for paraphrases

Local models handle broader rephrasings such as different words for the same topic.

Run this one-time setup in the same PowerShell environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "sentence-transformers>=5,<7"
.\.venv\Scripts\python.exe scripts\setup_live_chat_semantic.py
```

Stop the previous server with **Ctrl+C**, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --live-chat --experimental-semantic
```

The header should read **local models on**. Ask **what is a photo**, then enter **what is a photograph** as a standalone question. The answer shows **Semantic hit**, the matched saved question, and **0 provider calls**. A request for **what is a photon** takes the fresh path.

Exact matches have priority. If both prompts are already saved as exact answers, use an empty cache directory to demonstrate the semantic path:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --live-chat --experimental-semantic --directory runs\chat-semantic-demo
```

This chooses separate cache storage and keeps the shared quota journal. Existing exact answers can enter the semantic index when you ask their original questions again. The local policy evaluates short standalone definitions using dense retrieval, a duplicate-question check in both directions, and relevance to the cached answer. It sends requests to the model when a local check cannot establish reuse. A similarity score is a retrieval signal. The pinned-model smoke tests cover the photo example, different phrasings, and nearby concepts. Wider quality evaluation is available in the benchmark dashboard.
