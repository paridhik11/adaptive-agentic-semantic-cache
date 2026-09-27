"""Phase 6 — Sealed Dataset Builder and Verifier.

Constructs and verifies data/raw/phase6_blind_eval_dataset.json:
- Target N = 175 entries across 7 domains matching new_dataset_v3 proportions.
- Real Stack Exchange cold seeds with verified source_url and se_score.
- Hand-authored pairs:
  - AUTO_REUSE near-exact pairs
  - AMBIGUOUS paraphrase pairs (safe reuse -> HIT)
  - AMBIGUOUS adversarial trap pairs (unsafe reuse -> BYPASS)
- Standalone realtime_news_weather entries (temporal -> BYPASS).
- 0 overlap with all prior datasets in data/raw/ (1,211+ queries).
- 0 internal annotation collisions across MISS queries.
"""

from __future__ import annotations

import difflib
import gzip
import html
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DATASET_PATH = REPO_ROOT / "data" / "raw" / "phase6_blind_eval_dataset.json"

# All prior datasets in data/raw
PRIOR_DATASET_PATHS = [
    REPO_ROOT / "data" / "raw" / "query_pair_reuse_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_heldout.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_final_test.json",
    REPO_ROOT / "data" / "raw" / "query_stability_human_credibility.json",
    REPO_ROOT / "data" / "raw" / "synthetic_query_pair_feedback.json",
    REPO_ROOT / "data" / "raw" / "load_test_query_stream.json",
    REPO_ROOT / "data" / "raw" / "new_dataset_v3.json",
    REPO_ROOT / "data" / "raw" / "new_dataset_v2.json",
]


def load_all_prior_queries() -> Set[str]:
    """Collect every query string across all previous benchmark and load test files."""
    prior: Set[str] = set()
    for p in PRIOR_DATASET_PATHS:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        items = data.get("queries", data.get("items", data.get("data", []))) if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for k in ["query", "query_a", "query_b", "prompt", "text"]:
                        v = item.get(k, "")
                        if isinstance(v, str) and v.strip():
                            prior.add(v.strip().lower())
    return prior


def fetch_se_questions(
    site: str,
    min_score: int = 50,
    max_score: int = 400,
    pages: Tuple[int, ...] = (2, 3),
    pagesize: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch live questions from public Stack Exchange API."""
    items_out = []
    for page in pages:
        url = (
            f"https://api.stackexchange.com/2.3/questions"
            f"?page={page}&pagesize={pagesize}&order=desc&sort=votes"
            f"&min={min_score}&max={max_score}&site={site}&filter=default"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Phase6BlindEval/1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=12.0) as res:
                content = res.read()
                if res.info().get("Content-Encoding") == "gzip":
                    content = gzip.decompress(content)
                payload = json.loads(content.decode("utf-8"))
                for q in payload.get("items", []):
                    title = html.unescape(q.get("title", "")).strip()
                    if title and len(title) >= 15:
                        items_out.append({
                            "query": title,
                            "source_url": q.get("link"),
                            "se_score": q.get("score"),
                            "tags": q.get("tags", []),
                        })
            time.sleep(0.4)
        except Exception as exc:
            print(f"Warning: Failed to fetch {site} p{page}: {exc}")
    return items_out


def build_hand_authored_pairs() -> Dict[str, List[Dict[str, Any]]]:
    """Carefully curated hand-authored pairs across all domains.
    
    Contains:
    - 1 AUTO_REUSE pair per domain (near-exact duplicate of stable query)
    - 2-3 AMBIGUOUS paraphrase pairs per domain (same intent, alternate phrasing -> HIT)
    - 2 AMBIGUOUS adversarial trap pairs per domain (contrasting intent/operation -> BYPASS)
    """
    pairs: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Computer Science (6 pairs -> 12 entries)
    pairs["computer_science"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_cs_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "What is the time complexity of QuickSort in the average case?",
            "follow_on": "What is the average case time complexity of QuickSort?",
            "behavior": "HIT",
            "tags": ["algorithm", "quicksort", "time-complexity"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_cs_pair_02",
            "type": "PARAPHRASE",
            "anchor": "How do you detect memory leaks in a C++ application on Linux?",
            "follow_on": "What tools or techniques can identify memory leaks in C++ programs under Linux?",
            "behavior": "HIT",
            "tags": ["c++", "linux", "memory-leaks", "valgrind"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_cs_pair_03",
            "type": "PARAPHRASE",
            "anchor": "What is the difference between synchronous and asynchronous programming?",
            "follow_on": "Can you explain how asynchronous execution differs from synchronous execution in programming?",
            "behavior": "HIT",
            "tags": ["concurrency", "async", "sync"],
        },
        # Pair 4: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_cs_pair_04",
            "type": "PARAPHRASE",
            "anchor": "Why is immutability important in functional programming?",
            "follow_on": "What advantages does data immutability offer in functional programming paradigms?",
            "behavior": "HIT",
            "tags": ["functional-programming", "immutability"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_cs_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How does symmetric encryption work and what are common symmetric ciphers?",
            "follow_on": "How does asymmetric encryption work and what are common asymmetric ciphers?",
            "behavior": "BYPASS",
            "tags": ["cryptography", "encryption", "security"],
        },
        # Pair 6: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_cs_pair_06",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "Explain the difference between call-by-value and call-by-reference.",
            "follow_on": "Explain the difference between call-by-name and call-by-need.",
            "behavior": "BYPASS",
            "tags": ["compiler", "semantics", "parameter-passing"],
        },
    ]

    # 2. System Operations (6 pairs -> 12 entries)
    pairs["system_operations"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_sys_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "How do I check available disk space in Linux from command line?",
            "follow_on": "How to check available disk space in Linux using command line?",
            "behavior": "HIT",
            "tags": ["linux", "bash", "disk-space", "df"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_sys_pair_02",
            "type": "PARAPHRASE",
            "anchor": "How can I find all files modified in the last 24 hours on Ubuntu?",
            "follow_on": "What is the command to list files that were changed in the past 24 hours in Ubuntu?",
            "behavior": "HIT",
            "tags": ["linux", "find", "filesystem"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_sys_pair_03",
            "type": "PARAPHRASE",
            "anchor": "What is the purpose of the sticky bit on Linux directory permissions?",
            "follow_on": "Why is the sticky bit permission set on directories in Linux?",
            "behavior": "HIT",
            "tags": ["linux", "permissions", "chmod", "security"],
        },
        # Pair 4: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_sys_pair_04",
            "type": "PARAPHRASE",
            "anchor": "How do you configure an Nginx reverse proxy to forward client real IP headers?",
            "follow_on": "How can Nginx be configured as a reverse proxy to pass the real client IP address?",
            "behavior": "HIT",
            "tags": ["nginx", "reverse-proxy", "http-headers"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_sys_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How to enable IP forwarding in Linux kernel using sysctl?",
            "follow_on": "How to disable IP forwarding in Linux kernel using sysctl?",
            "behavior": "BYPASS",
            "tags": ["linux", "networking", "sysctl", "routing"],
        },
        # Pair 6: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_sys_pair_06",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How to mount an NFS export with read-write permissions on Linux?",
            "follow_on": "How to mount an NFS export in read-only mode on Linux?",
            "behavior": "BYPASS",
            "tags": ["nfs", "mount", "storage", "linux"],
        },
    ]

    # 3. Mathematics (6 pairs -> 12 entries)
    pairs["mathematics"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_math_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "What is Euler's formula for planar graphs connecting vertices, edges, and faces?",
            "follow_on": "What is Euler's formula relating vertices, edges, and faces in planar graphs?",
            "behavior": "HIT",
            "tags": ["graph-theory", "planar-graphs", "topology"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_math_pair_02",
            "type": "PARAPHRASE",
            "anchor": "How do you find the eigenvalues and eigenvectors of a 2x2 matrix?",
            "follow_on": "What is the procedure for computing the eigenvalues and corresponding eigenvectors for a two by two matrix?",
            "behavior": "HIT",
            "tags": ["linear-algebra", "eigenvalues", "matrices"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_math_pair_03",
            "type": "PARAPHRASE",
            "anchor": "Can you explain why the harmonic series diverges?",
            "follow_on": "What is a straightforward proof that the sum of 1/n diverges as n goes to infinity?",
            "behavior": "HIT",
            "tags": ["calculus", "series", "divergence"],
        },
        # Pair 4: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_math_pair_04",
            "type": "PARAPHRASE",
            "anchor": "What does it mean for a group to be abelian in abstract algebra?",
            "follow_on": "In abstract algebra, what is the definition and significance of an abelian group?",
            "behavior": "HIT",
            "tags": ["abstract-algebra", "group-theory"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_math_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "What is the formula to compute the volume of a sphere of radius r?",
            "follow_on": "What is the formula to compute the surface area of a sphere of radius r?",
            "behavior": "BYPASS",
            "tags": ["geometry", "sphere", "mensuration"],
        },
        # Pair 6: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_math_pair_06",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How do you determine if a given square matrix is positive definite?",
            "follow_on": "How do you determine if a given square matrix is negative definite?",
            "behavior": "BYPASS",
            "tags": ["linear-algebra", "matrices", "positive-definite"],
        },
    ]

    # 4. Finance & Economics (5 pairs -> 10 entries)
    pairs["finance_economics"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_fin_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "What is the compound annual growth rate formula and how is CAGR calculated?",
            "follow_on": "What is the formula for calculating Compound Annual Growth Rate (CAGR)?",
            "behavior": "HIT",
            "tags": ["investing", "cagr", "returns"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_fin_pair_02",
            "type": "PARAPHRASE",
            "anchor": "What is the difference between a traditional IRA and a Roth IRA?",
            "follow_on": "How do Roth IRAs compare to traditional IRAs regarding tax treatment upon contribution and withdrawal?",
            "behavior": "HIT",
            "tags": ["retirement", "ira", "taxes"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_fin_pair_03",
            "type": "PARAPHRASE",
            "anchor": "How does the discount rate set by a central bank influence commercial bank lending rates?",
            "follow_on": "What is the mechanism by which central bank policy interest rates affect retail interest rates?",
            "behavior": "HIT",
            "tags": ["monetary-policy", "central-banks", "interest-rates"],
        },
        # Pair 4: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_fin_pair_04",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How does purchasing power parity explain long-term exchange rate movements?",
            "follow_on": "How does interest rate parity explain short-term exchange rate movements?",
            "behavior": "BYPASS",
            "tags": ["economics", "foreign-exchange", "parity"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_fin_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "What are the rights and payout priority of preferred stock vs common stock?",
            "follow_on": "What are the rights and payout priority of corporate debt vs equity shares in liquidation?",
            "behavior": "BYPASS",
            "tags": ["corporate-finance", "stocks", "bonds"],
        },
    ]

    # 5. Science & Medicine (5 pairs -> 10 entries)
    pairs["science_medicine"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_sci_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "What is the function of mitochondria in eukaryotic cellular respiration?",
            "follow_on": "What role do mitochondria play in cellular respiration in eukaryotic cells?",
            "behavior": "HIT",
            "tags": ["biology", "cell-biology", "mitochondria"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_sci_pair_02",
            "type": "PARAPHRASE",
            "anchor": "Why is the sky blue during midday according to physics?",
            "follow_on": "What physical mechanism, such as Rayleigh scattering, causes the sky to appear blue?",
            "behavior": "HIT",
            "tags": ["physics", "optics", "scattering"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_sci_pair_03",
            "type": "PARAPHRASE",
            "anchor": "How do vaccines stimulate the adaptive immune system to produce antibodies?",
            "follow_on": "What is the biological mechanism by which immunizations trigger memory B-cell and antibody formation?",
            "behavior": "HIT",
            "tags": ["immunology", "vaccines", "medicine"],
        },
        # Pair 4: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_sci_pair_04",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "What happens during the mitotic phase of somatic cell division?",
            "follow_on": "What happens during the meiotic phase of germ cell division?",
            "behavior": "BYPASS",
            "tags": ["genetics", "mitosis", "meiosis", "cell-division"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_sci_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "Explain the difference between exothermic and endothermic chemical reactions.",
            "follow_on": "Explain the difference between exergonic and endergonic thermodynamic reactions.",
            "behavior": "BYPASS",
            "tags": ["chemistry", "thermodynamics", "reactions"],
        },
    ]

    # 6. History & Geography (5 pairs -> 10 entries)
    pairs["history_geography"] = [
        # Pair 1: AUTO_REUSE
        {
            "pair_id": "p6_hist_pair_01",
            "type": "AUTO_REUSE",
            "anchor": "What were the main causes of the outbreak of the Peloponnesian War?",
            "follow_on": "What were the primary historical causes of the Peloponnesian War?",
            "behavior": "HIT",
            "tags": ["ancient-history", "greece", "war"],
        },
        # Pair 2: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_hist_pair_02",
            "type": "PARAPHRASE",
            "anchor": "How did the invention of the printing press by Gutenberg impact literacy in Europe?",
            "follow_on": "In what ways did Johannes Gutenberg's movable type printing press transform European literacy levels?",
            "behavior": "HIT",
            "tags": ["european-history", "printing-press", "renaissance"],
        },
        # Pair 3: Paraphrase (AMBIGUOUS -> HIT)
        {
            "pair_id": "p6_hist_pair_03",
            "type": "PARAPHRASE",
            "anchor": "What geographical factors influenced the development of early Mesopotamian civilization?",
            "follow_on": "How did rivers and geography shape the emergence of civilization in ancient Mesopotamia?",
            "behavior": "HIT",
            "tags": ["geography", "ancient-mesopotamia", "civilization"],
        },
        # Pair 4: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_hist_pair_04",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "What were the terms and territorial concessions of the Treaty of Versailles in 1919?",
            "follow_on": "What were the terms and territorial concessions of the Treaty of Brest-Litovsk in 1918?",
            "behavior": "BYPASS",
            "tags": ["world-war-one", "treaties", "diplomacy"],
        },
        # Pair 5: Adversarial Trap (AMBIGUOUS -> BYPASS)
        {
            "pair_id": "p6_hist_pair_05",
            "type": "ADVERSARIAL_TRAP",
            "anchor": "How did plate tectonics create the Himalayan mountain range?",
            "follow_on": "How did plate tectonics create the Marianas trench oceanic depression?",
            "behavior": "BYPASS",
            "tags": ["geology", "plate-tectonics", "earth-science"],
        },
    ]

    return pairs


def build_realtime_queries() -> List[Dict[str, Any]]:
    """Hand-authored volatile/temporal queries for realtime_news_weather (all BYPASS)."""
    return [
        {
            "id": "p6_rt_001",
            "query": "What is the current temperature and weather forecast for Chicago right now?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_001",
            "tags": ["weather", "forecast", "realtime"],
            "se_score": None,
            "note": "Temporal weather query requiring fresh fetch",
        },
        {
            "id": "p6_rt_002",
            "query": "What is the live trading price of Bitcoin in USD at this moment?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_002",
            "tags": ["cryptocurrency", "realtime", "market-price"],
            "se_score": None,
            "note": "Volatile cryptocurrency market data",
        },
        {
            "id": "p6_rt_003",
            "query": "Who is currently leading the 2026 Formula 1 World Drivers Championship standings?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_003",
            "tags": ["sports", "formula1", "standings", "2026"],
            "se_score": None,
            "note": "Live seasonal sports leaderboard query",
        },
        {
            "id": "p6_rt_004",
            "query": "What are the latest breaking global news headlines reported in the past hour?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_004",
            "tags": ["breaking-news", "realtime", "world-news"],
            "se_score": None,
            "note": "Temporal breaking news event",
        },
        {
            "id": "p6_rt_005",
            "query": "What is the current status of traffic delays on the I-95 corridor today?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_005",
            "tags": ["traffic", "transit", "realtime"],
            "se_score": None,
            "note": "Live roadway transit conditions",
        },
        {
            "id": "p6_rt_006",
            "query": "What are today's live currency exchange rates between the Euro and Japanese Yen?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_006",
            "tags": ["forex", "exchange-rate", "realtime"],
            "se_score": None,
            "note": "Intraday foreign exchange rate",
        },
        {
            "id": "p6_rt_007",
            "query": "Which team won yesterday's Champions League football match?",
            "domain": "realtime_news_weather",
            "behavior": "BYPASS",
            "source": "hand_authored",
            "source_url": None,
            "pair_id": "p6_rt_007",
            "tags": ["sports", "football", "match-results"],
            "se_score": None,
            "note": "Recent sporting event result",
        },
    ]


def run_checks(records: List[Dict[str, Any]], prior_queries: Set[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Check external leakage and internal collision."""
    # 1. External leakage check
    overlaps = []
    for r in records:
        q = r["query"].strip().lower()
        if q in prior_queries:
            overlaps.append(f"[{r['id']}] {r['query']}")

    # 2. Internal collision check among MISS entries
    miss_records = [r for r in records if r.get("behavior", "").upper() == "MISS"]
    texts = [r["query"] for r in miss_records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_mat = vectorizer.fit_transform(texts)
    sim_mat = cosine_similarity(tfidf_mat)

    collisions = []
    n = len(miss_records)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = miss_records[i]
            r_j = miss_records[j]
            seq_ratio = difflib.SequenceMatcher(None, r_i["query"].lower(), r_j["query"].lower()).ratio()
            tfidf_sim = float(sim_mat[i, j])
            if tfidf_sim >= 0.70 or seq_ratio >= 0.70:
                collisions.append({
                    "id_a": r_i["id"],
                    "id_b": r_j["id"],
                    "domain_a": r_i["domain"],
                    "domain_b": r_j["domain"],
                    "tfidf_sim": round(tfidf_sim, 4),
                    "fuzzy_ratio": round(seq_ratio, 4),
                    "query_a": r_i["query"],
                    "query_b": r_j["query"],
                })

    return overlaps, collisions


def main():
    print("=" * 80)
    print("PHASE 6 SEALED DATASET GENERATOR")
    print("=" * 80)

    prior_queries = load_all_prior_queries()
    print(f"Loaded {len(prior_queries)} prior unique queries for external leakage checking.")

    # Fetch candidates from Stack Exchange for 6 cold seed domains
    site_map = {
        "computer_science": [("stackoverflow", 30)],
        "system_operations": [("serverfault", 20), ("superuser", 20)],
        "mathematics": [("math", 35)],
        "finance_economics": [("money", 35)],
        "science_medicine": [("biology", 15), ("physics", 15), ("chemistry", 15)],
        "history_geography": [("history", 35)],
    }

    target_se_counts = {
        "computer_science": 18,
        "system_operations": 18,
        "mathematics": 17,
        "finance_economics": 17,
        "science_medicine": 17,
        "history_geography": 15,
    }

    selected_se_entries: Dict[str, List[Dict[str, Any]]] = {}

    for domain, site_specs in site_map.items():
        domain_cands = []
        for site, count in site_specs:
            fetched = fetch_se_questions(site, min_score=60, max_score=350, pages=(2, 3), pagesize=count)
            domain_cands.extend(fetched)

        # Filter out any query overlapping prior datasets
        clean_cands = []
        for c in domain_cands:
            q_norm = c["query"].strip().lower()
            if q_norm not in prior_queries:
                clean_cands.append(c)

        # Select target count ensuring diversity
        target_n = target_se_counts[domain]
        chosen: List[Dict[str, Any]] = []
        for c in clean_cands:
            # Check similarity against already chosen in this domain to avoid redundant topics
            is_dup = False
            for ch in chosen:
                ratio = difflib.SequenceMatcher(None, c["query"].lower(), ch["query"].lower()).ratio()
                if ratio > 0.55:
                    is_dup = True
                    break
            if not is_dup:
                chosen.append(c)
                if len(chosen) >= target_n:
                    break

        if len(chosen) < target_n:
            raise RuntimeError(f"Could not find enough clean SE questions for {domain}: got {len(chosen)}, need {target_n}")

        selected_se_entries[domain] = chosen
        print(f"Domain {domain}: selected {len(chosen)} verified SE cold seeds.")

    # Build hand-authored pairs
    hand_pairs = build_hand_authored_pairs()
    realtime_items = build_realtime_queries()

    # Now assemble final dataset with interleaved stream ordering
    # Structure:
    # We will assemble records domain-by-domain or interleaved, with pair anchors placed before follow-ons
    all_records: List[Dict[str, Any]] = []

    # Format domain codes
    dom_prefix = {
        "computer_science": "p6_cs",
        "system_operations": "p6_sys",
        "mathematics": "p6_math",
        "finance_economics": "p6_fin",
        "science_medicine": "p6_sci",
        "history_geography": "p6_hist",
        "realtime_news_weather": "p6_rt",
    }

    # For each domain, construct anchor records, follow-on records, and cold seeds
    domain_batches: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]] = {}

    for dom, prefix in dom_prefix.items():
        if dom == "realtime_news_weather":
            continue
        se_items = selected_se_entries[dom]
        pairs_list = hand_pairs[dom]

        cold_records = []
        for i, se in enumerate(se_items, start=1):
            qid = f"{prefix}_se_{i:03d}"
            cold_records.append({
                "id": qid,
                "query": se["query"],
                "domain": dom,
                "behavior": "MISS",
                "source": "stackexchange",
                "source_url": se["source_url"],
                "pair_id": qid,
                "tags": se.get("tags", []),
                "se_score": se.get("se_score"),
                "note": None,
            })

        anchor_records = []
        followon_records = []
        for p_idx, p in enumerate(pairs_list, start=1):
            pid = p["pair_id"]
            aid = f"{prefix}_anc_{p_idx:02d}"
            fid = f"{prefix}_fol_{p_idx:02d}"

            anchor_records.append({
                "id": aid,
                "query": p["anchor"],
                "domain": dom,
                "behavior": "MISS",
                "source": "hand_authored",
                "source_url": None,
                "pair_id": pid,
                "tags": p["tags"],
                "se_score": None,
                "note": f"Anchor for {p['type']}",
            })

            followon_records.append({
                "id": fid,
                "query": p["follow_on"],
                "domain": dom,
                "behavior": p["behavior"],
                "source": "hand_authored",
                "source_url": None,
                "pair_id": pid,
                "tags": p["tags"],
                "se_score": None,
                "note": f"Follow-on for {p['type']} ({p['behavior']})",
            })

        domain_batches[dom] = (cold_records, anchor_records, followon_records)

    # Sequence records into a realistic production load-test stream:
    # 1. First half of cold seeds and all anchors (seeds the cache)
    # 2. Second half of cold seeds mixed with follow-ons and realtime queries
    first_wave: List[Dict[str, Any]] = []
    second_wave: List[Dict[str, Any]] = []

    for dom, (colds, anchors, followons) in domain_batches.items():
        mid = len(colds) // 2
        first_wave.extend(colds[:mid])
        first_wave.extend(anchors)
        second_wave.extend(colds[mid:])
        second_wave.extend(followons)

    # Interleave realtime queries throughout second wave
    combined_stream = first_wave + second_wave + realtime_items

    print(f"\nTotal assembled queries: {len(combined_stream)}")

    # Run leakage and collision checks
    overlaps, collisions = run_checks(combined_stream, prior_queries)

    print(f"External Leakage Check: {len(overlaps)} overlaps found.")
    if overlaps:
        for o in overlaps[:10]:
            print(f"  Overlap: {o}")
        raise ValueError("Leakage check failed!")

    print(f"Internal Collision Check: {len(collisions)} collision candidates (sim >= 0.70).")
    if collisions:
        for c in collisions:
            print(f"  Collision: [{c['id_a']}] vs [{c['id_b']}] tfidf={c['tfidf_sim']} fuzzy={c['fuzzy_ratio']}")
            print(f"    A: {c['query_a']}")
            print(f"    B: {c['query_b']}")
        raise ValueError("Internal collision check failed!")

    # Write out JSON
    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_stream, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote sealed dataset to {OUTPUT_DATASET_PATH}")
    print("\nSummary Statistics:")
    print(f"  Total records (N): {len(combined_stream)}")
    print(f"  Domain breakdown: {dict(sorted(Counter(r['domain'] for r in combined_stream).items()))}")
    print(f"  Behavior breakdown: {dict(sorted(Counter(r['behavior'] for r in combined_stream).items()))}")
    print(f"  Source breakdown: {dict(sorted(Counter(r['source'] for r in combined_stream).items()))}")


if __name__ == "__main__":
    main()
