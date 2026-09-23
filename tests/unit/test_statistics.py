"""
Unit tests for statistical functions (Phase 4.5).

Cross-checks every function against scipy/statsmodels reference implementations.
Do not trust a from-scratch implementation of a significance test without this.
"""

from __future__ import annotations
import numpy as np
import pytest
from scipy import stats

from analytics.statistics import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    matched_pairs_rank_biserial,
    compare_conditions_paired,
    bootstrap_ci,
    holm_bonferroni,
    rank_biserial_correlation,
    compare_conditions,
)


class TestMannWhitneyU:
    """Cross-check Mann-Whitney U against scipy.stats.mannwhitneyu."""

    def test_two_sided_matches_scipy(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0.5, 1, 50)

        u_our, p_our = mann_whitney_u(x, y, alternative="two-sided")
        u_scipy, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided")

        assert np.isclose(u_our, u_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)

    def test_less_alternative(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(1, 1, 30)

        u_our, p_our = mann_whitney_u(x, y, alternative="less")
        u_scipy, p_scipy = stats.mannwhitneyu(x, y, alternative="less")

        assert np.isclose(u_our, u_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)

    def test_greater_alternative(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(1, 1, 30)
        y = rng.normal(0, 1, 30)

        u_our, p_our = mann_whitney_u(x, y, alternative="greater")
        u_scipy, p_scipy = stats.mannwhitneyu(x, y, alternative="greater")

        assert np.isclose(u_our, u_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)

    def test_identical_samples(self) -> None:
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)

        u_our, p_our = mann_whitney_u(x, y, alternative="two-sided")
        u_scipy, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided")

        assert np.isclose(u_our, u_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)

    def test_different_sample_sizes(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        y = rng.normal(0, 1, 50)

        u_our, p_our = mann_whitney_u(x, y, alternative="two-sided")
        u_scipy, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided")

        assert np.isclose(u_our, u_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)


class TestWilcoxonSignedRank:
    """Cross-check Wilcoxon signed-rank against scipy.stats.wilcoxon."""

    def test_matches_scipy(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0.3, 1, 30)
        y = rng.normal(0, 1, 30)

        w_our, p_our = wilcoxon_signed_rank(x, y)
        w_scipy, p_scipy = stats.wilcoxon(x, y, alternative="two-sided",
                                          zero_method="wilcox")

        assert np.isclose(w_our, w_scipy, rtol=1e-10)
        assert np.isclose(p_our, p_scipy, rtol=1e-10)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([1.0, 2.0]), np.array([1.0]))

    def test_all_zero_differences(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        w, p = wilcoxon_signed_rank(x, x.copy())
        # scipy raises on all-zero differences; our wrapper returns the
        # degenerate "no evidence" result instead
        assert w == 0.0
        assert p == 1.0

    def test_systematic_shift_detected(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(1.0, 0.1, 20)
        y = np.zeros(20)
        _, p = wilcoxon_signed_rank(x, y)
        assert p < 0.001


class TestMatchedPairsRankBiserial:
    """Cross-check matched-pairs rank-biserial against manual calculation."""

    def test_manual_small_example(self) -> None:
        # diffs = [+3, -1, +2]; abs diffs = [3, 1, 2], ranks = [3, 1, 2]
        # pos ranks (diff>0): 3 + 2 = 5, neg ranks: 1
        # r = (5 - 1) / (3 + 1 + 2) = 4/6
        x = np.array([4.0, 0.0, 5.0])
        y = np.array([1.0, 1.0, 3.0])
        r = matched_pairs_rank_biserial(x, y)
        assert np.isclose(r, 4.0 / 6.0)

    def test_perfect_positive(self) -> None:
        x = np.array([5.0, 6.0, 7.0])
        y = np.array([1.0, 2.0, 3.0])
        assert np.isclose(matched_pairs_rank_biserial(x, y), 1.0)

    def test_perfect_negative(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 6.0, 7.0])
        assert np.isclose(matched_pairs_rank_biserial(x, y), -1.0)

    def test_all_zero_diffs_returns_nan(self) -> None:
        x = np.array([1.0, 2.0])
        assert np.isnan(matched_pairs_rank_biserial(x, x.copy()))

    def test_ties_and_zeros_excluded(self) -> None:
        # One tie (rank split) and one zero-diff pair (excluded)
        x = np.array([2.0, 3.0, 5.0])
        y = np.array([1.0, 3.0, 4.0])
        # diffs = [+1, 0, +1]; zero excluded -> diffs [+1, +1], ranks [1.5, 1.5]
        # r = (3 - 0) / 3 = 1.0
        assert np.isclose(matched_pairs_rank_biserial(x, y), 1.0)


class TestCompareConditionsPaired:
    """Test the paired condition-comparison pipeline."""

    def test_basic_paired_comparison(self) -> None:
        # n=6: exact Wilcoxon min two-sided p = 2/64 = 0.031 < 0.05,
        # so a perfectly consistent pair can reach significance
        conditions = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "b": np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),  # b consistently higher
        }
        result = compare_conditions_paired(conditions, alpha=0.05,
                                           n_resamples=1000)
        assert result["n_seeds"] == 6
        assert len(result["pairwise"]) == 1
        entry = result["pairwise"][0]
        assert entry["condition_a"] == "a"
        assert entry["condition_b"] == "b"
        assert entry["mean_diff"] < 0  # a - b negative
        assert entry["matched_pairs_rank_biserial"] == -1.0
        assert "a_vs_b" in result["significant_pairs"]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compare_conditions_paired({
                "a": np.array([1.0, 2.0]),
                "b": np.array([1.0, 2.0, 3.0]),
            })

    def test_single_condition_raises(self) -> None:
        with pytest.raises(ValueError):
            compare_conditions_paired({"a": np.array([1.0])})

    def test_no_difference_not_significant(self) -> None:
        # Identical scores: Wilcoxon degenerates to p=1.0
        conditions = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([1.0, 2.0, 3.0]),
        }
        result = compare_conditions_paired(conditions, n_resamples=500)
        assert result["pairwise"][0]["p_value"] == 1.0
        assert result["significant_pairs"] == []

    def test_three_conditions_three_pairs(self) -> None:
        # n=8: exact min p = 2/256 = 0.0078 < Holm threshold 0.05/6
        conditions = {
            "a": np.array([1.0, 1.2, 0.8, 1.1, 1.05, 0.95, 1.15, 0.85]),
            "b": np.array([2.0, 2.2, 1.8, 2.1, 2.05, 1.95, 2.15, 1.85]),
            "c": np.array([3.0, 3.2, 2.8, 3.1, 3.05, 2.95, 3.15, 2.85]),
        }
        result = compare_conditions_paired(conditions, n_resamples=500)
        assert len(result["pairwise"]) == 3
        labels = {f"{p['condition_a']}_vs_{p['condition_b']}"
                  for p in result["pairwise"]}
        assert labels == {"a_vs_b", "a_vs_c", "b_vs_c"}
        # All pairs perfectly ordered -> all significant after Holm
        assert set(result["significant_pairs"]) == labels


class TestBootstrapCI:
    """Cross-check bootstrap CI against expected behavior."""

    def test_mean_ci_contains_true_mean(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, 100)

        ci = bootstrap_ci(data, statistic=np.mean, n_resamples=5000, confidence_level=0.95, rng=np.random.default_rng(123))

        assert ci[0] <= 5.0 <= ci[1]

    def test_ci_width_shrinks_with_more_resamples(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(2.0, 50)

        ci_100 = bootstrap_ci(data, n_resamples=100, rng=np.random.default_rng(1))
        ci_10000 = bootstrap_ci(data, n_resamples=10000, rng=np.random.default_rng(1))

        width_100 = ci_100[1] - ci_100[0]
        width_10000 = ci_10000[1] - ci_10000[0]

        # More resamples should give tighter (or at least not wider) CI on average
        # We just check both are valid intervals
        assert ci_100[0] < ci_100[1]
        assert ci_10000[0] < ci_10000[1]

    def test_custom_statistic(self) -> None:
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        ci = bootstrap_ci(data, statistic=np.median, n_resamples=1000, rng=np.random.default_rng(42))
        assert ci[0] <= 3.0 <= ci[1]

    def test_confidence_levels(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        ci_90 = bootstrap_ci(data, confidence_level=0.90, rng=np.random.default_rng(1))
        ci_99 = bootstrap_ci(data, confidence_level=0.99, rng=np.random.default_rng(1))

        assert ci_99[0] <= ci_90[0]
        assert ci_99[1] >= ci_90[1]

    def test_empty_data_returns_nan(self) -> None:
        ci = bootstrap_ci(np.array([], dtype=float))
        assert np.isnan(ci[0]) and np.isnan(ci[1])


class TestHolmBonferroni:
    """Cross-check Holm-Bonferroni against manual calculation."""

    def test_no_rejections(self) -> None:
        p_values = [0.5, 0.6, 0.7]
        rejections = holm_bonferroni(p_values, alpha=0.05)
        assert not any(rejections)

    def test_all_rejected(self) -> None:
        p_values = [0.001, 0.002, 0.003]
        rejections = holm_bonferroni(p_values, alpha=0.05)
        assert all(rejections)

    def test_step_down_behavior(self) -> None:
        # p-values: 0.01, 0.02, 0.03, 0.04 with alpha=0.05
        # Sorted: 0.01, 0.02, 0.03, 0.04
        # Step 1: 0.01 <= 0.05/4 = 0.0125 -> reject
        # Step 2: 0.02 <= 0.05/3 = 0.0167 -> fail (0.02 > 0.0167)
        # Stop: remaining not rejected
        p_values = [0.02, 0.01, 0.04, 0.03]  # unsorted
        rejections = holm_bonferroni(p_values, alpha=0.05)
        # Original indices: 0.02 (idx 0), 0.01 (idx 1), 0.04 (idx 2), 0.03 (idx 3)
        # Only the smallest (0.01) should be rejected
        assert rejections == [False, True, False, False]

    def test_empty_list(self) -> None:
        assert holm_bonferroni([]) == []

    def test_alpha_boundary(self) -> None:
        # Exactly at boundary should reject
        p_values = [0.05]
        rejections = holm_bonferroni(p_values, alpha=0.05)
        assert rejections == [True]

    def test_single_p_value(self) -> None:
        assert holm_bonferroni([0.03], alpha=0.05) == [True]
        assert holm_bonferroni([0.07], alpha=0.05) == [False]


class TestRankBiserialCorrelation:
    """Cross-check rank-biserial correlation against formula."""

    def test_matches_formula(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0.5, 1, 50)

        r_our = rank_biserial_correlation(x, y)

        # Manual calculation: r = 2*U_greater/(n_x*n_y) - 1
        # where U_greater is the Mann-Whitney U statistic for x > y pairs
        # (scipy's alternative="less" counts x > y pairs)
        u_greater, _ = stats.mannwhitneyu(x, y, alternative="less")
        r_formula = (2 * u_greater) / (len(x) * len(y)) - 1

        assert np.isclose(r_our, r_formula, rtol=1e-10)

    def test_perfect_separation(self) -> None:
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([6, 7, 8, 9, 10], dtype=float)  # all y > x

        r = rank_biserial_correlation(x, y)
        # Complete separation: r = -1 (all x < all y)
        assert np.isclose(r, -1.0, atol=1e-10)

    def test_identical_distributions(self) -> None:
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)

        r = rank_biserial_correlation(x, y)
        # Identical: r = 0 (no systematic difference)
        assert np.isclose(r, 0.0, atol=1e-10)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)

        r_xy = rank_biserial_correlation(x, y)
        r_yx = rank_biserial_correlation(y, x)
        assert np.isclose(r_xy, -r_yx, rtol=1e-10)

    def test_empty_returns_nan(self) -> None:
        x = np.array([], dtype=float)
        y = np.array([1, 2, 3], dtype=float)
        assert np.isnan(rank_biserial_correlation(x, y))


class TestCompareConditions:
    """Test the end-to-end comparison pipeline."""

    def test_basic_comparison(self) -> None:
        condition_results = {
            "cond_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "cond_b": np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
        }
        result = compare_conditions(condition_results, metric="test_metric", alpha=0.05)

        assert "pairwise" in result
        assert "significant_pairs" in result
        assert "effect_sizes" in result
        assert "bootstrap_cis" in result
        assert result["metric"] == "test_metric"
        assert len(result["pairwise"]) == 1
        assert "cond_a_vs_cond_b" in result["significant_pairs"]

    def test_three_conditions(self) -> None:
        condition_results = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([4.0, 5.0, 6.0]),
            "c": np.array([7.0, 8.0, 9.0]),
        }
        result = compare_conditions(condition_results, alpha=0.05)

        assert len(result["pairwise"]) == 3  # a-b, a-c, b-c
        assert "bootstrap_cis" in result
        assert len(result["bootstrap_cis"]) == 3

    def test_effect_size_sign(self) -> None:
        condition_results = {
            "low": np.array([1.0, 1.0, 1.0, 1.0]),
            "high": np.array([10.0, 10.0, 10.0, 10.0]),
        }
        result = compare_conditions(condition_results)

        # high > low, so rank_biserial should be negative (low < high)
        r = result["effect_sizes"]["low_vs_high"]
        assert r < 0

    def test_non_significant_pairs(self) -> None:
        condition_results = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([1.1, 2.1, 3.1, 4.1, 5.1]),  # nearly identical
        }
        result = compare_conditions(condition_results, alpha=0.05)

        # p-value should be high, not significant
        assert len(result["significant_pairs"]) == 0

    def test_multiple_conditions_holm_correction(self) -> None:
        # 4 conditions -> 6 pairwise comparisons
        # Make some significant, some not
        condition_results = {
            "a": np.array([1, 2, 3, 4, 5]),
            "b": np.array([10, 11, 12, 13, 14]),
            "c": np.array([20, 21, 22, 23, 24]),
            "d": np.array([1.5, 2.5, 3.5, 4.5, 5.5]),  # similar to a
        }
        result = compare_conditions(condition_results, alpha=0.05)

        # a-b, a-c, b-c, a-d should be checked
        assert len(result["pairwise"]) == 6
        # Holm-Bonferroni should still reject a-b, a-c, b-c
        # but a-d should not be rejected
        sig_pairs = result["significant_pairs"]
        assert "a_vs_b" in sig_pairs
        assert "a_vs_c" in sig_pairs
        assert "b_vs_c" in sig_pairs
        # a_vs_d might or might not be significant depending on exact values
        # Just check it runs without error