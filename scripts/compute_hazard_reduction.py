"""Compute relative hazard reduction and confidence interval overlap.

Compares Phase 2 best fixed threshold (0.85) against Phase 3 OpenRouter
LLM judge on the exact same 120 query pairs from query_pair_reuse_benchmark.json.
Prints exact Clopper-Pearson 95% binomial confidence intervals, point difference,
relative reduction, and interval overlap status.
"""

from scipy.stats import beta


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95):
    """Compute exact two-sided Clopper-Pearson confidence interval."""
    alpha = 1.0 - confidence
    if k == 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        high = 1.0
    else:
        high = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def main():
    # Phase 2 baseline at threshold 0.85 (query_pair_reuse_benchmark.json, N=120)
    p2_k, p2_n = 5, 24
    p2_rate = p2_k / p2_n
    p2_low, p2_high = clopper_pearson_ci(p2_k, p2_n)

    # Phase 3 OpenRouter judge (query_pair_reuse_benchmark.json, N=120)
    p3_k, p3_n = 1, 12
    p3_rate = p3_k / p3_n
    p3_low, p3_high = clopper_pearson_ci(p3_k, p3_n)

    # Differences
    abs_diff = p2_rate - p3_rate
    rel_red = (p2_rate - p3_rate) / p2_rate
    overlap = not (p2_high < p3_low or p3_high < p2_low)

    print("=== Hazard Reduction Analysis (Same-Set N=120 pairs) ===")
    print(f"Phase 2 Best Fixed (0.85): {p2_k}/{p2_n} = {p2_rate*100:.2f}%, 95% CP CI [{p2_low*100:.2f}%, {p2_high*100:.2f}%]")
    print(f"Phase 3 OpenRouter Judge:  {p3_k}/{p3_n} = {p3_rate*100:.2f}%, 95% CP CI [{p3_low*100:.2f}%, {p3_high*100:.2f}%]")
    print(f"Absolute Difference:       {abs_diff*100:.2f} percentage points")
    print(f"Relative Hazard Reduction: {rel_red*100:.2f}%")
    print(f"Intervals Overlap:         {overlap}")
    if overlap:
        print("Conclusion: CIs overlap; difference is not statistically established at 95% confidence.")
    else:
        print("Conclusion: CIs do not overlap; difference is statistically established.")


if __name__ == "__main__":
    main()
