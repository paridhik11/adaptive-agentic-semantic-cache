# Live chat with exact response reuse

Live chat accepts general questions and sends fresh requests to an explicit free OpenRouter model. Ask a stable standalone question twice to compare a provider call with a cache hit. The local SQLite cache stores complete eligible answers for one hour. Include recent turns for a follow-up, or turn off exact reuse to force a fresh request.

## PowerShell

In your existing repository checkout:

```powershell
git fetch origin optimize-v2:refs/remotes/origin/optimize-v2
git switch optimize-v2
git pull --ff-only origin optimize-v2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-gateway.txt
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'OpenRouter API key' -AsSecureString)).Password
.\.venv\Scripts\python.exe scripts\run_demo.py --live-chat
```

If you already created `.venv` and installed the requirements, start at the key prompt. Keep PowerShell running and open [Live chat](http://127.0.0.1:8765/). The key prompt keeps the credential out of the command text. The environment variable is scoped to the current PowerShell process and processes it launches.

1. Ask **What is the difference between INNER JOIN and LEFT JOIN?** A cache miss shows **1 provider call**, elapsed time and completion tokens if reported by OpenRouter.
2. Select **Repeat as new question**. An eligible exact hit shows **0 provider calls**, **0 generated tokens**, and the new response time alongside the earlier fresh response.
3. Ask a different question. For a contextual question such as **Give me a small example**, select **Include recent turns**. The follow-up sends up to three recent turns to the provider.
4. Select **Benchmark analysis** to open the historical offline evidence and workload projections.

The default is `nvidia/nemotron-3.5-lightning:free`. If its availability changes, use `--model` followed by an explicit model ID ending in `:free`. The provider enforces zero price and disables automatic fallback. The persistent quota journal uses the existing free-only allowance (30 chat attempts with shared daily and rolling-minute caps). Use `--journal <path>` to share your existing quota journal. Reopening the chat preserves the allowance and cache. Eligible cached answers can still be served when the quota is exhausted.

Live chat uses the model's knowledge without web search. Response time includes server-side lookup and local verification through answer availability.

## Enable semantic reuse for paraphrases

Run this one-time setup in the same PowerShell environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "sentence-transformers>=5,<7"
.\.venv\Scripts\python.exe scripts\setup_live_chat_semantic.py
```

Stop the previous server with **Ctrl+C**, keep your OpenRouter key set, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --live-chat --experimental-semantic
```

The header should read **exact + semantic**. Ask **what is a photo**, then enter **what is a photograph** as a new standalone question with recent turns unchecked. The answer shows **Semantic hit**, the matched saved question, and **0 provider calls**. A request for **what is a photon** takes the fresh path.

Exact matches have priority. If both prompts are already saved as exact answers, use an empty cache directory to demonstrate the semantic path:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --live-chat --experimental-semantic --directory runs\chat-semantic-demo
```

This chooses separate cache storage and keeps the shared quota journal. Existing exact answers can enter the semantic index when you ask their original questions again. The local policy evaluates short standalone definitions using dense retrieval, a duplicate-question check in both directions, and relevance to the cached answer. It sends requests to the model when a local check cannot establish reuse. A similarity score is a retrieval signal. The pinned-model smoke tests cover the photo example, different phrasings, and nearby concepts. Wider quality evaluation is available in the benchmark dashboard.
