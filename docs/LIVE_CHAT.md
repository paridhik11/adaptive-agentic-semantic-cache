# Live general chat

Ask about risk management, SQL, or any other topic. New questions go to a free OpenRouter model. The local SQLite cache stores complete eligible answers for one hour. Automatic context covers short follow-ups such as “Give me an example.” Set **Context** to **Include recent turns** or **Standalone question** when you want explicit control.

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

Definition reuse is enabled by default. The cache checks that the saved answer opens by defining the same short topic. It permits an unrecognized typo in one long word when the other topic words agree, and falls back to the model when the match is ambiguous. The default check uses a local spelling dictionary.

If the three prompts already have separate exact answers, add `--directory runs\definition-reuse-demo` when starting the server to demonstrate the sequence with a separate cache. The provider quota journal stays shared.

The default is `nvidia/nemotron-3.5-lightning:free`. If its availability changes, use `--model` followed by an explicit model ID ending in `:free`. The provider enforces zero price and disables automatic fallback. The persistent quota journal uses the existing free-only allowance (30 chat attempts with shared daily and rolling-minute caps). Use `--journal <path>` to share your existing quota journal. Reopening the chat preserves the allowance and cache. Eligible cached answers can still be served when the quota is exhausted.

Live chat uses the model's knowledge without web search. Response time covers server-side processing through answer availability, including time waiting for the provider slot. Provider time starts after the slot opens and includes the quota journal. Cache hits for other questions can complete while the provider works.

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
