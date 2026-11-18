"""Replay one known semantic failure at the requirement stage, with authored controls."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy.decomposed_applicability import answer_satisfies_requirements, requirement_guard

TOOL = re.compile(r"\bcprofile\b", re.IGNORECASE)
FIXTURES = (
    ("Paraphrase without a profiling request",
     "How do I perform asynchronous programming with asyncio in Python?",
     "How can I run asynchronous tasks with asyncio?",
     "Use asyncio.run() and asyncio.gather().", (None, None)),
    ("Equivalent requests with a profiling tool",
     "How do I profile a Python function with cProfile?",
     "How can I use cProfile to profile a Python function?",
     "Run cProfile.run('work()').", (None, None)),
    ("Same tool in both prompts, absent from answer",
     "How do I profile a Python function with cProfile?",
     "How can I use cProfile to profile a Python function?",
     "Measure a duration with time.monotonic().", (None, "answer_omits_cprofile")),
)


def compare(original, incoming, answer):
    current = requirement_guard(original, incoming) or answer_satisfies_requirements(incoming, answer)
    proposed = current
    if proposed is None and TOOL.search(incoming):
        if not TOOL.search(original):
            proposed = "added_cprofile_requirement"
        elif not TOOL.search(answer):
            proposed = "answer_omits_cprofile"
    return current, proposed


def main():
    # Read the known case from the dashboard export of the immutable event log.
    report = json.loads((ROOT / "dashboard/interview/report.json").read_text(encoding="utf-8"))
    event = report["profiles"]["lower_repeat"]["failures"][0]
    cases = ((event["event_id"], event["matched_question"], event["question"], event["actual_response_answer"],
              (None, "added_cprofile_requirement")), *FIXTURES)
    for label, original, incoming, answer, expected in cases:
        outcome = compare(original, incoming, answer)
        if outcome != expected:
            raise SystemExit(f"{label}: expected {expected}, observed {outcome}")
        print(f"{label}: {outcome[0] or 'pass to next stage'} → {outcome[1] or 'pass to next stage'}")
    print("Development guard replay only: no neural inference, provider calls, or held-out quality estimate.")


if __name__ == "__main__":
    main()
