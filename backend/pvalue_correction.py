"""Multiple-testing correction shared by the p-value based projections
(kendall, spearman, pearson, odds ratio)."""

VALID_METHODS = ('none', 'fdr_bh', 'bonferroni')

def correct(cand, m, alpha, method='fdr_bh'):
    """Filter candidate significant pairs by multiple-testing correction.

    Args:
        cand:   list of tuples whose LAST element is the raw p-value. Callers
                pass only pairs with p < alpha, which is a safe superset for all
                three methods (none/bonferroni thresholds are <= alpha, and BH
                can never keep a pair with p >= alpha), so the omitted pairs can
                never survive. Their count must still be reflected in `m`.
        m:      total number of hypothesis tests performed (>= len(cand)).
        alpha:  significance level.
        method: 'none' (raw p < alpha), 'fdr_bh' (Benjamini-Hochberg step-up),
                or 'bonferroni' (p < alpha / m).

    Returns:
        The kept subset of `cand`.
    """
    if m <= 0 or not cand:
        return []
    method = (method or 'fdr_bh').lower()
    if method == 'none':
        return [e for e in cand if e[-1] < alpha]
    if method == 'bonferroni':
        thr = alpha / m
        return [e for e in cand if e[-1] < thr]
    if method == 'fdr_bh':
        cand.sort(key=lambda e: e[-1])
        kept = 0
        for rank, e in enumerate(cand, start=1):
            if e[-1] <= (rank / m) * alpha:
                kept = rank
        return cand[:kept]
    raise ValueError(f"unknown correction method: {method!r}")


if __name__ == "__main__":
    # m=10, alpha=0.05: BH keeps p<=k/m*alpha -> [.001<=.005, .008<=.01, .03>.015]
    c = [(0, 1, 0.9, 0.001), (0, 2, 0.8, 0.008), (0, 3, 0.7, 0.03)]
    assert [e[-1] for e in correct(list(c), 10, 0.05, 'fdr_bh')] == [0.001, 0.008]
    # none keeps everything below alpha (all three are < .05)
    assert len(correct(list(c), 10, 0.05, 'none')) == 3
    # bonferroni: alpha/m = .005 -> only .001
    assert [e[-1] for e in correct(list(c), 10, 0.05, 'bonferroni')] == [0.001]
    print("pvalue_correction self-check OK")
