# Dataset Schema Documentation

This document defines the formal data schemas, enumerations, and validation rules for the evaluation datasets used in the **Adaptive Agentic Semantic Cache** project.

---

## 1. Query Stability Benchmark Schema (`data/raw/query_stability_benchmark.json`)

The stability benchmark evaluates whether a standalone user query is about timeless/stable information or dynamic/time-sensitive state.

### 1.1 JSON Schema Definition

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "QueryStabilityBenchmark",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "id",
      "query",
      "domain",
      "stability_label",
      "temporal_sensitivity",
      "has_explicit_temporal_marker",
      "implicit_dependencies",
      "rationale"
    ],
    "properties": {
      "id": {
        "type": "string",
        "pattern": "^STAB-[0-9]{3,}$",
        "description": "Unique identifier for the stability benchmark sample."
      },
      "query": {
        "type": "string",
        "minLength": 5,
        "description": "The exact user prompt/query string."
      },
      "domain": {
        "type": "string",
        "enum": [
          "computer_science",
          "mathematics",
          "history_geography",
          "science_medicine",
          "finance_economics",
          "realtime_news_weather",
          "system_operations",
          "legal_compliance"
        ],
        "description": "Primary domain of the query."
      },
      "stability_label": {
        "type": "string",
        "enum": [
          "STABLE",
          "DYNAMIC",
          "CONDITIONALLY_STABLE"
        ],
        "description": "Ground-truth stability classification."
      },
      "temporal_sensitivity": {
        "type": "string",
        "enum": [
          "EVERGREEN",
          "SLOW_DECAY",
          "FAST_DECAY",
          "EVENT_DRIVEN"
        ],
        "description": "Rate of ground-truth decay or volatility."
      },
      "has_explicit_temporal_marker": {
        "type": "boolean",
        "description": "True if query includes overt temporal keywords or dates (e.g., 'latest', 'today', '2024', 'current')."
      },
      "implicit_dependencies": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Underlying environmental or contextual dependencies (e.g. 'python_version', 'os_flavor'). Empty array if none."
      },
      "rationale": {
        "type": "string",
        "minLength": 10,
        "description": "Detailed justification for the stability classification."
      }
    },
    "additionalProperties": false
  }
}
```

---

## 2. Query-Pair Reuse-Safety Benchmark Schema (`data/raw/query_pair_reuse_benchmark.json`)

The query-pair benchmark evaluates the core caching decision: **Given cached Query A with Answer A, can Answer A safely and accurately satisfy incoming Query B?**

### 2.1 JSON Schema Definition

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "QueryPairReuseSafetyBenchmark",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "id",
      "query_a",
      "query_b",
      "domain",
      "semantic_similarity_level",
      "is_reuse_safe",
      "taxonomy_class",
      "rejection_reason",
      "rationale"
    ],
    "properties": {
      "id": {
        "type": "string",
        "pattern": "^PAIR-[0-9]{3,}$",
        "description": "Unique identifier for the query pair benchmark sample."
      },
      "query_a": {
        "type": "string",
        "minLength": 5,
        "description": "The pre-existing cached query for which Answer A was generated."
      },
      "query_b": {
        "type": "string",
        "minLength": 5,
        "description": "The incoming user query being evaluated for potential cache reuse."
      },
      "domain": {
        "type": "string",
        "enum": [
          "computer_science",
          "mathematics",
          "history_geography",
          "science_medicine",
          "finance_economics",
          "realtime_news_weather",
          "system_operations",
          "legal_compliance"
        ]
      },
      "semantic_similarity_level": {
        "type": "string",
        "enum": ["HIGH", "MEDIUM", "LOW"],
        "description": "Expected conceptual embedding similarity level between Query A and Query B for evaluation."
      },
      "is_reuse_safe": {
        "type": "boolean",
        "description": "Ground truth: True if Answer A accurately and completely satisfies Query B without errors or stale information."
      },
      "taxonomy_class": {
        "type": "string",
        "enum": [
          "SAFE_EQUIVALENT",
          "UNSAFE_DIFFERENT_INTENT",
          "UNSAFE_SCOPE_MISMATCH",
          "UNSAFE_CONTEXT_MISMATCH",
          "UNSAFE_DYNAMIC_TEMPORAL",
          "UNSAFE_DIFFERENT_TOPIC"
        ],
        "description": "Taxonomy classification of the relationship between Query A and Query B."
      },
      "rejection_reason": {
        "type": ["string", "null"],
        "description": "Concise summary of why reuse is unsafe. Must be null if is_reuse_safe is true."
      },
      "rationale": {
        "type": "string",
        "minLength": 10,
        "description": "Detailed explanation justifying the reuse safety decision."
      }
    },
    "additionalProperties": false
  }
}
```

> **Note on LOW Similarity Cases:** All `LOW` semantic similarity cases are intentionally labeled `is_reuse_safe = false` with taxonomy class `UNSAFE_DIFFERENT_TOPIC`. They serve as an explicit, necessary baseline representing obvious candidate rejections.

---

## 3. Pipeline Alignment & Explicit Decision Terminology

The schema directly maps to the runtime cache decision pipeline:

```
Incoming Query (Q_B)
      │
      ▼
1. Candidate Retrieval: Query vector search in cache
   - If similarity < threshold ───► REGENERATE (Retrieval Miss)
      │ (Candidate Found: r_i = 1)
      ▼
2. Safety & Compatibility Check: Stability, Intent, Scope, Context
   - If dynamic OR incompatible ──► REGENERATE (Candidate Rejected / Safe Bypass)
   - If compatible & safe ────────► REUSE (Safe Cache Hit)
      │
      ▼
3. Log Outcome for Policy Tuning
```

### Critical Conceptual Distinction: Retrieval vs. Reuse Decision

> **Core Rule:** **Finding a cached candidate does NOT mean a cached answer was reused.**
> - **Candidate Retrieval ($r_i = 1$):** A semantically similar cached query was found.
> - **Reuse Decision ($\hat{y}_i = 1$):** The system actively decided to reuse the cached response.
> - **Regeneration / Rejection ($\hat{y}_i = 0$):** A candidate may have been retrieved, but was rejected as unsafe, prompting a fresh LLM call.

### Confusion Matrix Definitions (Based on Reuse Decision $\hat{y}_i$, NOT Retrieval)

| Pipeline Decision ($\hat{y}$) \ Ground Truth ($y$) | `is_reuse_safe == true` ($y = 1$) | `is_reuse_safe == false` ($y = 0$) |
| :--- | :--- | :--- |
| **System Decides: `REUSE` ($\hat{y} = 1$)** | **True Positive (TP)**<br>*(Safe Hit: reused answer and saved LLM call)* | **False Positive (FP)**<br>*(Hazard: reused incorrect or stale answer)* |
| **System Decides: `REGENERATE` ($\hat{y} = 0$)** | **False Negative (FN)**<br>*(Missed saving: unnecessarily called LLM)* | **True Negative (TN)**<br>*(Correct Bypass: rejected unsafe candidate / called LLM)* |
