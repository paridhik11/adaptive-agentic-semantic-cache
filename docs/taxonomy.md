# Query and Reuse-Safety Taxonomy

## 1. Project Goal & Core Problem

The primary objective of the **Adaptive Agentic Semantic Cache** is:

> **Reduce unnecessary LLM calls, token usage, cost, and latency by reusing previous answers only when that reuse is logically safe.**

A semantic cache looks up previously answered queries that are semantically close to an incoming query. However:

$$\text{Semantic Similarity is NOT Proof of Safe Answer Reuse}$$

Semantic similarity measures whether two queries are topically or linguistically related in vector embedding space. It does not guarantee that an answer generated for Query A can accurately, safely, and completely answer Query B.

---

## 2. The Four Pillars of Safe Reuse

To make a safe reuse decision, the system distinguishes between four key concepts:

```
                      1. Information Stability & Freshness
                      (Is the underlying knowledge static or changing?)
                                     │
                                     ▼
                      2. Semantic Candidate Retrieval
                      (Is there a closely related cached query?)
                                     │
                                     ▼
                      3. Intent, Scope & Context Compatibility
                      (Do the queries request the exact same thing?)
                                     │
                                     ▼
                      4. Answer Reuse Safety Decision
                      (Does Answer A safely and accurately satisfy Query B?)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                     [REUSE]                 [REGENERATE]
               (Safe cache hit)          (Call LLM / bypass)
```

1. **Information Stability / Freshness:** Is the information timeless (e.g., algorithm runtime, historical date) or dynamic and time-sensitive (e.g., stock price, weather, latest package version, current executive)? Dynamic information decays quickly and should not be served from a stale cache.
2. **Semantic Similarity as Candidate Signal:** Embedding similarity is used solely to retrieve potential candidate queries from the cache. High similarity is a starting point for evaluation, not a conclusive decision.
3. **Intent, Scope & Context Compatibility:** Even when two queries share high similarity, their underlying intent (explain vs debug), scope (broad summary vs specific detail), or context (Python vs JavaScript, Linux vs Windows, version differences) may conflict.
4. **Answer Reuse Safety:** An answer is safe to reuse when it correctly and completely satisfies the user's prompt without misleading instructions, missing constraints, or stale facts.

---

## 3. Practical Taxonomy Classes

To support the cache decision pipeline, query relationships are categorized into clear, practical classes:

| Taxonomy Class | Description | Reuse Decision | Primary Risk / Failure Mode |
| :--- | :--- | :--- | :--- |
| `SAFE_EQUIVALENT` | Phrasing variants, paraphrases, and synonyms that ask for the same core information under identical constraints. | **REUSE** | None; answer fully satisfies incoming query. |
| `UNSAFE_DIFFERENT_INTENT` | High similarity in subject matter, but requests a fundamentally different action or goal (e.g., create vs delete, explain vs debug, serialize vs parse). | **REGENERATE** | Answering the wrong question; misleading the user. |
| `UNSAFE_SCOPE_MISMATCH` | Scope divergence where the cached answer is too narrow (omitting required information) or too broad/unfocused for a specific question. | **REGENERATE** | Incomplete or unhelpful response. |
| `UNSAFE_CONTEXT_MISMATCH` | Incompatible technical environment, language version, operating system, or mutually exclusive assumptions (e.g., in-place vs new copy). | **REGENERATE** | Syntax errors, runtime failures, broken code. |
| `UNSAFE_DYNAMIC_TEMPORAL` | Time-sensitive information where a cached answer would provide stale, inaccurate, or obsolete data. | **REGENERATE** | Stale data served as current truth (prices, news, versions). |
| `UNSAFE_DIFFERENT_TOPIC` | Low similarity queries with distinct subject matter. | **REGENERATE** | Irrelevant answer (standard cache miss). |

---

## 4. Concrete Examples & Failure Modes

### 4.1 Safe Reuse (`SAFE_EQUIVALENT`)
- **Query A (Cached):** `"How do I reverse a string in Python using slicing?"`
- **Query B (Incoming):** `"Python string reversal with slice notation"`
- **Why Safe:** Both queries request the exact same code pattern (`[::-1]`) with identical constraints in Python.

- **Query A (Cached):** `"What is the average and worst-case time complexity of quicksort?"`
- **Query B (Incoming):** `"Quicksort Big-O runtime average and worst case"`
- **Why Safe:** Mathematical properties of quicksort are invariant; both queries seek the exact same theoretical values ($O(n \log n)$ and $O(n^2)$).

---

### 4.2 Different Intent (`UNSAFE_DIFFERENT_INTENT`)
- **Query A (Cached):** `"How do I create a virtual environment with venv in Python?"`
- **Query B (Incoming):** `"How do I delete a virtual environment created with venv in Python?"`
- **Similarity:** High lexical overlap (Python, virtual environment, venv).
- **Why Unsafe:** Creation (`python -m venv .venv`) does the opposite of deletion. Reusing Answer A causes task failure.

- **Query A (Cached):** `"How does quicksort partition logic work step by step?"`
- **Query B (Incoming):** `"Why is my quicksort implementation hitting maximum recursion depth?"`
- **Similarity:** High topical overlap.
- **Why Unsafe:** Conceptual algorithm explanation does not debug the user's missing base case or recursive pivot bug.

---

### 4.3 Scope Mismatch (`UNSAFE_SCOPE_MISMATCH`)
- **Query A (Cached):** `"What are the three main properties in the CAP theorem?"`
- **Query B (Incoming):** `"Explain partition tolerance in the CAP theorem with split-brain network examples."`
- **Why Unsafe:** Answer A gives a high-level summary of C, A, and P. It lacks the deep network split examples and partition handling details requested in Query B.

- **Query A (Cached):** `"How do I sort a list of tuples by the second element in Python?"`
- **Query B (Incoming):** `"How do I sort a list of objects by multiple attributes in Python?"`
- **Why Unsafe:** Single-tuple index lambda does not answer multi-attribute sorting (`(x.attr1, -x.attr2)` or `itemgetter`).

---

### 4.4 Context and Environment Mismatch (`UNSAFE_CONTEXT_MISMATCH`)
- **Query A (Cached):** `"How do I kill a process on port 8080?"` *(Answer generated for Linux: `lsof -i :8080 -t | xargs kill -9`)*
- **Query B (Incoming):** `"How to kill process on port 8080 in PowerShell?"`
- **Similarity:** High semantic proximity.
- **Why Unsafe:** Linux bash commands fail with syntax errors in Windows PowerShell.

- **Query A (Cached):** `"How do I merge two dictionaries in Python using the union operator?"` *(Shows `d1 | d2`)*
- **Query B (Incoming):** `"How do I merge two dictionaries in Python 3.5?"`
- **Why Unsafe:** Python 3.9+ syntax `d1 | d2` throws a fatal `TypeError` in Python 3.5.

- **Query A (Cached):** `"How do I remove duplicates from a list in Python?"` *(Shows `list(set(x))`)*
- **Query B (Incoming):** `"How do I remove duplicates from a list in Python while preserving insertion order?"`
- **Why Unsafe:** `set()` destroys ordering. Serving Answer A introduces subtle ordering bugs.

---

### 4.5 Dynamic / Temporal Staleness (`UNSAFE_DYNAMIC_TEMPORAL`)
- **Query A (Cached):** `"What is the current stock price of Apple?"` *(Answered yesterday: \$180)*
- **Query B (Incoming):** `"Apple stock price right now"`
- **Why Unsafe:** Equity prices change continuously; cached response serves stale financial data.

- **Query A (Cached):** `"What is the latest stable release version of Node.js?"` *(Answered in 2023: Node 20)*
- **Query B (Incoming):** `"Current Node.js LTS version"`
- **Why Unsafe:** Software LTS releases rotate periodically; cached response becomes obsolete.

---

### 4.6 Nuanced Temporal Edge Cases: Completed Historical Periods
Not all queries with temporal terms are dynamic:

- **Query:** `"Who won the Men's FIFA World Cup in Qatar in 2022?"`
- **Analysis:** Contains temporal wording ("2022", "in Qatar"), but refers to a completed historical event. The answer (Argentina) is permanently fixed and **safe to cache/reuse**.

- **Query:** `"What was the highest US inflation rate during the 1970s?"`
- **Analysis:** Historical macroeconomic period; data is settled and **stable**.

---

## 5. Decision Rules for the Pipeline

1. **Stability / Freshness Gate:** If a query is about dynamic, fast-changing state, route directly to LLM generation (bypass cache).
2. **Similarity as Filter Only:** Vector similarity retrieves candidate answers, but must never trigger automatic reuse without compatibility validation.
3. **Intent, Scope, Context Validation:** For candidates above the retrieval threshold, verify that intent, scope, and technical assumptions match before deciding to reuse.
4. **Safety Over Hit Rate:** It is always better to unnecessarily regenerate an answer (costing tokens) than to serve an incorrect or stale cached answer (causing user harm).
