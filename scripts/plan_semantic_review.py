"""Prospective review workload arithmetic, never a project quality result."""
from __future__ import annotations

import argparse
import json
import math


def lower_bound(groups, errors, confidence=0.95):
    from scipy.stats import beta
    if (type(groups) is not int or type(errors) is not int or
            not 0 <= errors <= groups or groups < 1 or not 0 < confidence < 1):
        raise ValueError("invalid group counts or confidence")
    return 0.0 if errors == groups else float(beta.ppf(1-confidence, groups-errors, errors+1))


def plan(accepted_groups=125, acceptance_rate=0.30):
    from scipy.stats import binom
    if type(accepted_groups) is not int or accepted_groups < 1 or not 0 < acceptance_rate <= 1:
        raise ValueError("positive count and acceptance rate in (0, 1] required")
    expected_size = math.ceil(accepted_groups / acceptance_rate)
    # Binary search the fixed positive cohort giving 95% sufficiency probability
    # ONLY under an assumed common independent group acceptance probability.
    low, high = accepted_groups, expected_size
    while binom.sf(accepted_groups-1, high, acceptance_rate) < .95:
        high *= 2
    while low < high:
        mid = (low + high)//2
        if binom.sf(accepted_groups-1, mid, acceptance_rate) >= .95:
            high = mid
        else:
            low = mid+1
    return {"evidence_type": "hypothetical_planning_arithmetic",
            "required_accepted_groups": accepted_groups, "assumed_group_acceptance_rate": acceptance_rate,
            "positive_groups_for_expected_count": expected_size,
            "positive_groups_for_95pct_sufficiency_under_assumption": low,
            "two_error_lower_bound_at_required_count": lower_bound(accepted_groups, min(2, accepted_groups)),
            "warning": "Acceptance assumption is unmeasured. Neither cohort size guarantees enough accepted groups. Freeze before evaluation; never collect until passing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-groups", type=int, default=125)
    parser.add_argument("--group-acceptance-rate", type=float, default=.30)
    args = parser.parse_args()
    print(json.dumps(plan(args.accepted_groups, args.group_acceptance_rate), indent=2))


if __name__ == "__main__":
    main()
