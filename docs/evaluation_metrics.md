# Formal Evaluation Metrics

This document defines the mathematical formulas, definitions, numerators, denominators, and boundary conditions for evaluating the **Adaptive Agentic Semantic Cache**.

Metric definitions are established prior to experiments to ensure consistent evaluation and prevent metric gaming.

---

## 1. Ground Truth & Confusion Matrix Definitions

Let $N$ denote the total number of incoming evaluation requests:

$$\mathcal{D} = \{(q_i, c_i, y_i)\}_{i=1}^N$$

where:
- $q_i$: Incoming query.
- $c_i$: Retrieved candidate query from cache (or $\emptyset$ if none found above retrieval threshold).
- $r_i \in \{0, 1\}$: **Candidate Retrieval Indicator** ($r_i = 1$ if a candidate $c_i$ was retrieved above the similarity threshold, $0$ otherwise).
- $y_i \in \{0, 1\}$: **Ground Truth Reuse Safety** ($y_i = 1$ if safe to reuse candidate answer for $q_i$, $0$ otherwise).
- $\hat{y}_i \in \{0, 1\}$: **System Reuse Decision** ($\hat{y}_i = 1$ if system decides to **REUSE** cached answer, $0$ if system decides to **REGENERATE** via LLM).

---

### Critical Operational Principle

> **Finding a cached candidate does NOT mean a cached answer was reused.**
> 
> The pipeline explicitly separates:
> 1. **Candidate Retrieval ($r_i = 1$):** A candidate exists above the vector search similarity threshold.
> 2. **Reuse Decision ($\hat{y}_i = 1$):** The system passes stability/intent/scope checks and actually serves the cached answer.
> 3. **Regeneration / Rejection ($\hat{y}_i = 0$):** A candidate may be found ($r_i = 1$) but rejected as unsafe, triggering a fresh LLM generation.

---

### Confusion Matrix Counts (Evaluated on Reuse Decision $\hat{y}$, NOT Retrieval)

The standard classification counts evaluate the **system's reuse decision $\hat{y}_i$ against ground-truth safety $y_i$**:

| Count Symbol | Operational Meaning | Formula |
| :--- | :--- | :--- |
| $N$ | Total Incoming Queries | $\sum_{i=1}^N 1$ |
| $N_{\text{retrieved}}$ | Queries with a Candidate Retrieved from Cache | $\sum_{i=1}^N \mathbb{I}(r_i = 1)$ |
| $N_{\text{reuse}}$ | Queries where System Decided to Reuse Cached Answer | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 1)$ |
| $N_{\text{regen}}$ | Queries where System Regenerated via LLM | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 0)$ |
| $N_{\text{safe}}$ | Ground Truth Safely Reusable Queries | $\sum_{i=1}^N \mathbb{I}(y_i = 1)$ |
| **$TP$** | **True Positive (Correct Reuse):** System reused answer and reuse was safe. | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 1 \land y_i = 1)$ |
| **$FP$** | **False Positive (Hazardous Reuse):** System reused answer but reuse was unsafe. | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 1 \land y_i = 0)$ |
| **$TN$** | **True Negative (Safe Bypass / Rejection):** System regenerated and reuse was unsafe. | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 0 \land y_i = 0)$ |
| **$FN$** | **False Negative (Missed Reuse):** System regenerated but reuse was safe. | $\sum_{i=1}^N \mathbb{I}(\hat{y}_i = 0 \land y_i = 1)$ |

---

## 2. Core Safety & Cache Performance Metrics

### 2.1 Candidate / Cache Retrieval Rate ($\text{CRR}_{\text{retrieval}}$)
Measures how often vector similarity search finds a relevant candidate query in the cache above the retrieval threshold:

$$\text{CRR}_{\text{retrieval}} = \frac{N_{\text{retrieved}}}{N} = \frac{\sum_{i=1}^N \mathbb{I}(r_i = 1)}{N}$$

- **Range:** $[0.0, 1.0]$
- **Role in Pipeline:** Represents candidate generation coverage before safety and compatibility checks.

---

### 2.2 Actual Reuse Rate ($\text{ARR}$)
The proportion of total incoming traffic where the system actually decides to reuse a cached answer:

$$\text{ARR} = \frac{N_{\text{reuse}}}{N} = \frac{TP + FP}{N}$$

- **Range:** $[0.0, 1.0]$
- **Role in Pipeline:** Reflects actual traffic offloaded from the primary LLM. Note that $\text{ARR} \le \text{CRR}_{\text{retrieval}}$ since unsafe retrieved candidates are rejected.

---

### 2.3 Correct Reuse Rate / Precision ($\text{CRR}$)
The proportion of reused answers that were genuinely safe and accurate:

$$\text{CRR} = \frac{TP}{N_{\text{reuse}}} = \frac{TP}{TP + FP}$$

- **Range:** $[0.0, 1.0]$
- **Experimental Target:** Aim to maximize toward $1.0$ (minimizing bad reuses).
- **Edge Case:** If $N_{\text{reuse}} = 0$, $\text{CRR} \triangleq 1.0$ (no unsafe answers were served).

---

### 2.4 Incorrect Reuse / Hazard Rate ($\text{IRR}$)
The proportion of reused answers that were incorrect, stale, or misleading (the critical safety metric):

$$\text{IRR}_{\text{cache}} = \frac{FP}{N_{\text{reuse}}} = \frac{FP}{TP + FP} = 1 - \text{CRR}$$

Across the full traffic volume:

$$\text{IRR}_{\text{traffic}} = \frac{FP}{N}$$

- **Experimental Target:** Minimize toward $0.0$.
- **Edge Case:** If $N_{\text{reuse}} = 0$, $\text{IRR}_{\text{cache}} \triangleq 0.0$.

---

### 2.5 False Rejection Rate ($\text{FRR}$)
The proportion of safely reusable opportunities that the system unnecessarily regenerated (opportunity cost):

$$\text{FRR} = \frac{FN}{N_{\text{safe}}} = \frac{FN}{TP + FN}$$

- **Range:** $[0.0, 1.0]$
- **Role in Pipeline:** Measures optimization efficiency without sacrificing safety.
- **Edge Case:** If $N_{\text{safe}} = 0$, $\text{FRR} \triangleq 0.0$.

---

## 3. Resource & Performance Optimization Metrics

### 3.1 Token Savings ($\Delta T$)

Let:
- $T_{\text{in}}(q_i)$: Input prompt tokens for query $q_i$.
- $T_{\text{out}}(q_i)$: Output response tokens for query $q_i$.
- $T_{\text{agent}}(q_i)$: Overhead tokens consumed by the decision agent (0 for direct hits and direct bypasses).

#### Absolute Tokens Saved ($\Delta T$):
$$\Delta T = \sum_{i=1}^N \left( \hat{y}_i \cdot \left[ T_{\text{in}}(q_i) + T_{\text{out}}(q_i) \right] - T_{\text{agent}}(q_i) \right)$$

#### Percentage Token Reduction ($\text{PTR}$):
$$\text{PTR} = \frac{\Delta T}{\sum_{i=1}^N [T_{\text{in}}(q_i) + T_{\text{out}}(q_i)]}$$

---

### 3.2 Monetary Cost Savings ($\text{CS}$)

Let:
- $P_{\text{in}}, P_{\text{out}}$: Cost per 1k input/output tokens for main LLM.
- $P_{\text{agent}}$: Token pricing for decision agent model (if invoked).
- $C_{\text{embed}}$: Cost of query embedding lookup.

#### Baseline Cost without Cache ($C_{\text{baseline}}$):
$$C_{\text{baseline}} = \sum_{i=1}^N \left( \frac{T_{\text{in}}(q_i)}{1000} P_{\text{in}} + \frac{T_{\text{out}}(q_i)}{1000} P_{\text{out}} \right)$$

#### Total System Cost with Cache ($C_{\text{cache}}$):
$$C_{\text{cache}} = \sum_{i=1}^N \left( C_{\text{embed}}(q_i) + (1 - \hat{y}_i)\left[ \frac{T_{\text{in}}(q_i)}{1000} P_{\text{in}} + \frac{T_{\text{out}}(q_i)}{1000} P_{\text{out}} \right] + C_{\text{agent}}(q_i) \right)$$

#### Net Cost Reduction Percentage ($\text{NCR}$):
$$\text{NCR} = \frac{C_{\text{baseline}} - C_{\text{cache}}}{C_{\text{baseline}}} \times 100\%$$

---

### 3.3 Latency by Execution Path

The pipeline evaluates latency across three distinct paths:

1. **Direct Reuse Path ($L_{\text{direct}}$):** High similarity, high stability confidence $\to$ immediate cache hit.
   $$L_{\text{direct}} = L_{\text{embed}} + L_{\text{lookup}}$$
2. **Decision / Ambiguous Path ($L_{\text{agent}}$):** Borderline similarity $\to$ lightweight decision agent verification $\to$ reuse or regenerate.
   $$L_{\text{agent}} = L_{\text{embed}} + L_{\text{lookup}} + L_{\text{agent\_call}} + (1 - \hat{y}_i) L_{\text{gen}}$$
3. **Regeneration / Miss Path ($L_{\text{miss}}$):** Dynamic query or low similarity $\to$ full LLM generation.
   $$L_{\text{miss}} = L_{\text{embed}} + L_{\text{lookup}} + L_{\text{gen}}$$

#### Latency Reduction Comparison:
Report P50, P95, and P99 latency comparisons between:
- $\text{Latency}_{\text{baseline}}$ (100% LLM generation)
- $\text{Latency}_{\text{system}}$ (Adaptive cache pipeline)

---

## 4. Summary Table of Metrics

| Metric | Symbol | Goal | Formula / Basis |
| :--- | :--- | :--- | :--- |
| **Candidate Retrieval Rate** | $\text{CRR}_{\text{retrieval}}$ | Track Coverage | $N_{\text{retrieved}} / N$ |
| **Actual Reuse Rate** | $\text{ARR}$ | Maximize Safely | $(TP + FP) / N$ |
| **Correct Reuse Rate (Precision)** | $\text{CRR}$ | Maximize ($\to 1.0$) | $TP / (TP + FP)$ |
| **Cache Hazard Rate** | $\text{IRR}_{\text{cache}}$ | Minimize ($\to 0.0$) | $FP / (TP + FP)$ |
| **False Rejection Rate** | $\text{FRR}$ | Minimize | $FN / (TP + FN)$ |
| **Net Token Savings** | $\text{PTR}$ | Maximize | $(\text{Tokens Saved} - \text{Agent Overhead}) / \text{Baseline Tokens}$ |
| **Net Cost Reduction** | $\text{NCR}$ | Maximize | $(C_{\text{baseline}} - C_{\text{cache}}) / C_{\text{baseline}}$ |
| **Latency Reduction** | $\Delta L_{\text{P50/P95}}$ | Maximize | $L_{\text{baseline}} - L_{\text{system}}$ |

---

## 5. Evaluation Invariants

1. **Safety Priority:** Serving an incorrect cached answer ($\text{IRR} > 0$) is treated as a critical failure. Maximizing hit rate at the expense of correctness is invalid.
2. **Account for Overhead:** Agent tokens, agent latency, and embedding costs must be included in all cost/token/latency calculations.
3. **No Retrospective Metric Manipulation:** Formulas remain constant across experimental iterations.
