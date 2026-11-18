"""Evaluate the frozen requirement-stage check on the separate authored fixture."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from interview_guard_experiment import compare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "interview"
PROTOCOL = ROOT / "docs" / "INTERVIEW_GUARD_PROTOCOL.md"
POLICY = Path(__file__).with_name("interview_guard_experiment.py")
GROUP_DESIGN = {"adequate_paraphrase": "allow", "added_tool": "block", "omitted_tool": "block"}
STRESS_DESIGN = {"name_only": "block"}


def evaluate():
    policy_hash = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    if f"`{policy_hash}`" not in PROTOCOL.read_text(encoding="utf-8"):
        raise ValueError("candidate implementation differs from the frozen protocol")
    raw_fixtures = (DATA / "guard_fixtures.json").read_bytes()
    cases = json.loads(raw_fixtures)["fixtures"]
    groups = defaultdict(dict)
    outcomes = []
    seen = set()
    for case in cases:
        group, kind, expected = (case[key] for key in ("group", "type", "expected"))
        if case["id"] in seen or kind in groups[group]:
            raise ValueError(f"duplicate fixture: {case['id']}")
        seen.add(case["id"])
        groups[group][kind] = expected
        current, proposed = compare(case["original"], case["incoming"], case["answer"])
        outcomes.append({
            "id": case["id"], "group": group, "type": kind, "expected": expected,
            "current_block_reason": current, "proposed_block_reason": proposed,
        })
    if list(groups.values()).count(GROUP_DESIGN) != 8 or list(groups.values()).count(STRESS_DESIGN) != 4 or len(groups) != 12:
        raise ValueError("fixture composition differs from the frozen protocol")

    inadequate = [row for row in outcomes if row["expected"] == "block"]
    adequate = [row for row in outcomes if row["expected"] == "allow"]
    by_type = {}
    for kind in GROUP_DESIGN | STRESS_DESIGN:
        rows = [row for row in outcomes if row["type"] == kind]
        by_type[kind] = {
            "fixtures": len(rows),
            "current_blocked": sum(row["current_block_reason"] is not None for row in rows),
            "proposed_blocked": sum(row["proposed_block_reason"] is not None for row in rows),
        }
    return {
        "evidence": "authored requirement-stage fixture, no neural or live evaluation",
        "policy_sha256": policy_hash,
        "fixture_sha256": hashlib.sha256(raw_fixtures).hexdigest(),
        "intent_groups": len(groups),
        "cases": len(outcomes),
        "adequate": len(adequate),
        "inadequate": len(inadequate),
        "newly_blocked_inadequate": sum(
            row["current_block_reason"] is None and row["proposed_block_reason"] is not None for row in inadequate
        ),
        "newly_blocked_adequate": sum(
            row["current_block_reason"] is None and row["proposed_block_reason"] is not None for row in adequate
        ),
        "remaining_inadequate_passes": [row["id"] for row in inadequate if row["proposed_block_reason"] is None],
        "by_type": by_type,
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the committed fixture receipt")
    args = parser.parse_args()
    result = evaluate()
    contents = json.dumps(result, indent=2) + "\n"
    path = DATA / "guard_fixture_results.json"
    if args.check:
        if path.read_bytes() != contents.encode("utf-8"):
            parser.exit(1, "Fixture receipt differs. Inspect the policy and fixtures.\n")
    else:
        path.write_text(contents, encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
