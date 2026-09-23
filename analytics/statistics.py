"""
Statistical functions for condition comparison (Phase 4.5).

All functions are cross-checked against scipy/statsmodels reference implementations
in tests/unit/test_statistics.py.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Any, Callable, Optional, Callable, Optional


def mann_whitney_u(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Non-parametric test for whether two independent samples come from
    the same distribution.

    Args:
        x: First sample (1D array).
        y: Second sample (1D array).
        alternative: "two-sided", "less", or "greater".

    Returns:
        Tuple of (U statistic, p-value).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Use scipy's implementation as reference (cross-checked in tests)
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(u_stat), float(p_value)


def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test on paired samples.

    The paired analogue of Mann-Whitney U: tests whether the paired
    differences x - y are symmetrically distributed around zero. Use this
    (not mann_whitney_u) whenever the two samples share units - e.g. the
    same frozen genome scored on train vs. held-out layouts, or conditions
    compared seed-by-seed with a common initial population.

    Args:
        x: First sample (1D array), paired element-wise with y.
        y: Second sample (1D array), same length as x.

    Returns:
        Tuple of (W statistic, two-sided p-value). Returns (0.0, 1.0)
        when all paired differences are exactly zero.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"paired samples must have equal length: {x.size} != {y.size}")
    if x.size == 0:
        raise ValueError("paired samples must be non-empty")

    diffs = x - y
    if np.all(diffs == 0.0):
        return 0.0, 1.0
    w_stat, p_value = stats.wilcoxon(x, y, alternative="two-sided",
                                      zero_method="wilcox")
    return float(w_stat), float(p_value)


def matched_pairs_rank_biserial(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Matched-pairs rank-biserial correlation (effect size for the
    Wilcoxon signed-rank test).

    r = (sum of positive signed ranks - sum of |negative| signed ranks)
        / (total rank sum of non-zero differences)

    Range [-1, 1]: +1 = x consistently larger, -1 = y consistently larger,
    0 = no systematic direction. Zero-difference pairs are excluded.

    Args:
        x: First sample, paired element-wise with y.
        y: Second sample, same length as x.

    Returns:
        Matched-pairs rank-biserial correlation. NaN if all differences
        are zero or the arrays are empty.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    diffs = x - y
    diffs = diffs[diffs != 0.0]
    if diffs.size == 0:
        return float("nan")

    ranks = stats.rankdata(np.abs(diffs))
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    return float((pos - neg) / ranks.sum())


def compare_conditions_paired(
    condition_results: Dict[str, np.ndarray],
    metric: str = "fitness",
    alpha: float = 0.05,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """
    Paired counterpart of compare_conditions: every condition must supply
    one score per seed, aligned by index, and every pairwise comparison is
    a paired (within-seed) test.

    Runs Wilcoxon signed-rank per pair, Holm-Bonferroni correction across
    the pair family, matched-pairs rank-biserial effect sizes, and
    bootstrap CIs on the per-pair differences.

    Args:
        condition_results: Dict condition_name -> 1D array of per-seed
            scores. All arrays must have equal length (aligned by seed).
        metric: Metric name (for labeling).
        alpha: Family-wise error rate for Holm-Bonferroni.
        n_resamples: Bootstrap resamples for difference CIs.

    Returns:
        Dict with keys:
            - "pairwise": list of per-pair dicts (p_value, effect_size,
              mean_diff, diff_ci95)
            - "significant_pairs": pair labels surviving Holm correction
            - "bootstrap_cis": per-condition score CIs
            - "n_seeds": number of aligned scores per condition

    Raises:
        ValueError: If fewer than 2 conditions or array lengths differ.
    """
    names = list(condition_results.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 conditions for comparison")
    arrays = {k: np.asarray(v, dtype=float).ravel()
              for k, v in condition_results.items()}
    lengths = {v.size for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("All conditions must have equal-length "
                         f"(seed-aligned) score arrays, got {lengths}")
    n = arrays[names[0]].size

    results: Dict[str, Any] = {
        "metric": metric,
        "alpha": alpha,
        "pairwise": [],
        "significant_pairs": [],
        "bootstrap_cis": {},
        "n_seeds": n,
    }
    for name in names:
        results["bootstrap_cis"][name] = bootstrap_ci(
            arrays[name], n_resamples=n_resamples,
            rng=np.random.default_rng(100))

    p_values: List[float] = []
    pair_labels: List[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            label = f"{names[i]}_vs_{names[j]}"
            a, b = arrays[names[i]], arrays[names[j]]
            diffs = a - b
            _, p_val = wilcoxon_signed_rank(a, b)
            effect = matched_pairs_rank_biserial(a, b)
            diff_ci = bootstrap_ci(diffs, n_resamples=n_resamples,
                                    rng=np.random.default_rng(101))
            p_values.append(p_val)
            pair_labels.append(label)
            results["pairwise"].append({
                "condition_a": names[i],
                "condition_b": names[j],
                "mean_diff": float(diffs.mean()),
                "p_value": float(p_val),
                "matched_pairs_rank_biserial": effect,
                "diff_ci95": diff_ci,
            })

    rejections = holm_bonferroni(p_values, alpha=alpha)
    for label, reject in zip(pair_labels, rejections):
        if reject:
            results["significant_pairs"].append(label)
    return results


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        data: Input data (1D array).
        statistic: Function to compute on each resample (default: mean).
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        method: "percentile" or "basic" bootstrap method.
        rng: Optional random generator for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If method is not "percentile" or "basic".
    """
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0:
        return (np.nan, np.nan)

    if method not in ("percentile", "basic"):
        raise ValueError(f"method must be 'percentile' or 'basic', got '{method}'")

    if rng is None:
        rng = np.random.default_rng()

    n = data.size

    # Vectorized resampling: shape (n_resamples, n)
    resamples = rng.choice(data, size=(n_resamples, n), replace=True)
    bootstrap_stats = np.apply_along_axis(statistic, 1, resamples)

    alpha = 1 - confidence_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    if method == "percentile":
        lower = np.percentile(bootstrap_stats, lower_pct)
        upper = np.percentile(bootstrap_stats, upper_pct)
    else:  # basic method
        stat_val = statistic(data)
        lower = 2 * stat_val - np.percentile(bootstrap_stats, upper_pct)
        upper = 2 * stat_val - np.percentile(bootstrap_stats, lower_pct)

    return float(lower), float(upper)


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Controls family-wise error rate (FWER) at level alpha.
    More powerful than Bonferroni correction.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Family-wise error rate.

    Returns:
        List of booleans: True if hypothesis is rejected (significant).

    Raises:
        ValueError: If any p-value is NaN.
    """
    p_values = np.asarray(p_values, dtype=float)
    if np.any(np.isnan(p_values)):
        raise ValueError("p_values contains NaN")
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values, keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Step-down procedure
    rejections = np.zeros(n, dtype=bool)
    for i, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        if p <= adjusted_alpha:
            rejections[orig_idx] = True
        else:
            # Step-down: once we fail to reject, all remaining fail
            break

    return rejections.tolist()


def rank_biserial_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Rank-biserial correlation (effect size for Mann-Whitney U).

    Measures the degree of overlap between two distributions.
    Range: [-1, 1], where 0 = no effect, +1 = complete separation (all x > y),
    -1 = complete separation (all x < y).

    Formula: r = 1 - (2 * U_less) / (n_x * n_y)
    where U_less is the Mann-Whitney U statistic for the "less" alternative
    (number of pairs where x < y).

    Args:
        x: First sample (1D array).
        y: Second sample (1D array).

    Returns:
        Rank-biserial correlation coefficient.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n_x, n_y = x.size, y.size
    if n_x == 0 or n_y == 0:
        return np.nan

    # U_greater: number of pairs where x > y
    # scipy's alternative="less" counts x > y pairs
    u_greater, _ = stats.mannwhitneyu(x, y, alternative="less")
    r = (2 * u_greater) / (n_x * n_y) - 1
    return float(r)


def compare_conditions(
    condition_results: Dict[str, np.ndarray],
    metric: str = "fitness",
    alpha: float = 0.05,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """
    End-to-end condition comparison pipeline.

    Given seed-level results per condition, runs all pairwise comparisons,
    applies Holm-Bonferroni correction, computes effect sizes, and returns
    a complete summary.

    Args:
        condition_results: Dict mapping condition_name -> array of seed-level metrics.
        metric: Name of the metric (for labeling).
        alpha: Family-wise error rate for Holm-Bonferroni.
        n_resamples: Bootstrap resamples for CI.

    Returns:
        Dict with keys:
            - "pairwise": list of comparison dicts
            - "significant_pairs": list of (cond_a, cond_b) that are significant
            - "effect_sizes": dict mapping (cond_a, cond_b) -> rank_biserial
            - "bootstrap_cis": dict mapping condition_name -> (lower, upper)
    """
    condition_names = list(condition_results.keys())
    if len(condition_names) < 2:
        raise ValueError("Need at least 2 conditions for comparison")

    results = {
        "metric": metric,
        "alpha": alpha,
        "pairwise": [],
        "significant_pairs": [],
        "effect_sizes": {},
        "bootstrap_cis": {},
    }

    # Bootstrap CIs for each condition
    for name, values in condition_results.items():
        ci = bootstrap_ci(np.asarray(values), n_resamples=n_resamples)
        results["bootstrap_cis"][name] = ci

    # Pairwise comparisons
    p_values = []
    pair_names = []
    effect_sizes = {}

    for i in range(len(condition_names)):
        for j in range(i + 1, len(condition_names)):
            name_a, name_b = condition_names[i], condition_names[j]
            data_a = np.asarray(condition_results[name_a])
            data_b = np.asarray(condition_results[name_b])

            # Mann-Whitney U
            u_stat, p_val = mann_whitney_u(data_a, data_b)
            p_values.append(p_val)
            pair_names.append((name_a, name_b))

            # Effect size
            r_rb = rank_biserial_correlation(data_a, data_b)
            effect_sizes[f"{name_a}_vs_{name_b}"] = r_rb

            # Store raw comparison
            results["pairwise"].append({
                "condition_a": name_a,
                "condition_b": name_b,
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "rank_biserial": float(r_rb),
                "mean_a": float(np.mean(data_a)),
                "mean_b": float(np.mean(data_b)),
            })

    # Holm-Bonferroni correction
    rejections = holm_bonferroni(p_values, alpha=alpha)
    for (name_a, name_b), reject in zip(pair_names, rejections):
        if reject:
            results["significant_pairs"].append(f"{name_a}_vs_{name_b}")

    results["effect_sizes"] = effect_sizes
    return results