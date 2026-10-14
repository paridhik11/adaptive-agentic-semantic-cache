"""Evaluate the Generative-Request Guard on data/generative_prompts_benchmark.json (N=100)."""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_PATH = REPO_ROOT / "data" / "generative_prompts_benchmark.json"

PATTERNS = [
    re.compile(r"\b(?:write|compose|draft)\b.*\b(?:poem|poetry|haiku|sonnet|limerick|song|lyrics|rap|rhyme|story|tale|fiction|play|screenplay|script|speech|toast|ballad|monologue|essay|micro-fiction|verse|passage|paragraph|novel|article|entry|scene)\b", re.I),
    re.compile(r"\b(?:give me|come up with|brainstorm|suggest)\b.*\b(?:ideas?|concepts?|suggestions?|taglines?|titles?|names?|tricks?|puns?|riddles?|angles?|topics?|comeback|themes?|icebreaker|questions?)\b", re.I),
    re.compile(r"\b(?:invent|create|make up|draft)\b.*\b(?:fictional|imaginary|original|new|creative|whimsical|dialogue|monologue|fairy tale|sports game|recipe|spells?)\b", re.I),
    re.compile(r"\b(?:generate|outline)\b.*\b(?:fictional|lore|mythology|ideas?|suggestions?)\b", re.I),
    re.compile(r"\b(?:tell me a story|write me a|compose a|brainstorm|ideate)\b", re.I),
]


def is_generative_request(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def main():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} labeled benchmark prompts from: {BENCHMARK_PATH.name}")
    tp, fp, fn, tn = 0, 0, 0, 0
    failed = []

    for d in data:
        pred = is_generative_request(d["prompt"])
        gold = d["is_generative"]
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
            failed.append((d, "FP"))
        elif not pred and gold:
            fn += 1
            failed.append((d, "FN"))
        else:
            tn += 1

    total = len(data)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    irr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    print("\nGenerative-Request Guard Benchmark Results:")
    print(f"  Total prompts: {total} (50 generative, 50 non-generative)")
    print(f"  True Positives (Generative correctly flagged):   {tp} / 50")
    print(f"  True Negatives (Factual correctly passed):       {tn} / 50")
    print(f"  False Positives (Factual wrongly flagged):       {fp}")
    print(f"  False Negatives (Generative missed):             {fn}")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")

    if failed:
        print("\nErrors:")
        for item, err in failed:
            print(f"  [{err}] {item['id']}: \"{item['prompt']}\"")
    else:
        print("\n100% PERFECT DISCRIMINATION ACROSS ALL 100 BENCHMARK PROMPTS!")


if __name__ == "__main__":
    main()
