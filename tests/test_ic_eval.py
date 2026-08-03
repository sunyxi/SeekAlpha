"""Tests for src/orb/features/ic_eval.py.

Covers: forward return computation, Spearman rank IC, IC summary,
Benjamini-Hochberg FDR correction, and edge cases.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from orb.features.ic_eval import (
    _cs_rank,
    fdr_correct,
    fwd_return,
    ic_summary,
    ic_summary_hac,
    ic_summary_nonoverlap,
    rank_ic_series,
)


# ---------------------------------------------------------------------------
# _cs_rank — cross-sectional rank helper
# ---------------------------------------------------------------------------

class TestCsRank(unittest.TestCase):

    def test_range_0_to_1(self):
        x = np.array([[3.0, 1.0, 4.0, 1.5, 2.0]])
        r = _cs_rank(x)
        self.assertTrue(np.all(r >= 0.0 - 1e-9))
        self.assertTrue(np.all(r <= 1.0 + 1e-9))

    def test_max_is_1(self):
        x = np.array([[5.0, 2.0, 8.0, 1.0]])
        r = _cs_rank(x)
        self.assertAlmostEqual(r[0].max(), 1.0)

    def test_min_is_0(self):
        x = np.array([[5.0, 2.0, 8.0, 1.0]])
        r = _cs_rank(x)
        self.assertAlmostEqual(r[0].min(), 0.0)

    def test_nan_input_row_produces_nan(self):
        x = np.array([[np.nan, np.nan]])
        r = _cs_rank(x)
        self.assertTrue(np.all(np.isnan(r)))

    def test_single_finite_value_produces_nan_row(self):
        # n < 2 → can't rank
        x = np.array([[np.nan, 5.0]])
        r = _cs_rank(x)
        self.assertTrue(np.all(np.isnan(r)))

    def test_ordering_preserved(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        r = _cs_rank(x)
        # Ranks should be strictly increasing left-to-right
        self.assertTrue(np.all(np.diff(r[0]) > 0))


# ---------------------------------------------------------------------------
# fwd_return
# ---------------------------------------------------------------------------

class TestFwdReturn(unittest.TestCase):

    def setUp(self):
        # Simple 5-day, 2-asset close panel
        self.close = np.array([
            [100.0, 200.0],
            [110.0, 210.0],
            [105.0, 220.0],
            [115.0, 215.0],
            [120.0, 225.0],
        ])

    def test_shape_preserved(self):
        fr = fwd_return(self.close, 1)
        self.assertEqual(fr.shape, self.close.shape)

    def test_h1_last_row_is_nan(self):
        fr = fwd_return(self.close, 1)
        self.assertTrue(np.all(np.isnan(fr[-1])))

    def test_h1_values(self):
        fr = fwd_return(self.close, 1)
        # Row 0: (110-100)/100 = 0.10, (210-200)/200 = 0.05
        np.testing.assert_allclose(fr[0], [0.10, 0.05])

    def test_h2_last_two_rows_are_nan(self):
        fr = fwd_return(self.close, 2)
        self.assertTrue(np.all(np.isnan(fr[-2:])))

    def test_h2_values(self):
        fr = fwd_return(self.close, 2)
        # Row 0: (105-100)/100=0.05, (220-200)/200=0.10
        np.testing.assert_allclose(fr[0], [0.05, 0.10])

    def test_negative_return(self):
        close = np.array([[100.0], [90.0]])
        fr = fwd_return(close, 1)
        self.assertAlmostEqual(fr[0, 0], -0.10)


# ---------------------------------------------------------------------------
# rank_ic_series — Spearman cross-sectional IC
# ---------------------------------------------------------------------------

class TestRankIcSeries(unittest.TestCase):

    def _make_panel(self, T=50, N=10, seed=42):
        rng = np.random.RandomState(seed)
        close = np.cumprod(1 + rng.randn(T, N) * 0.01, axis=0) * 100.0
        factor = rng.randn(T, N)
        return factor, close

    def test_shape(self):
        factor, close = self._make_panel()
        fr = fwd_return(close, 1)
        ic = rank_ic_series(factor, fr)
        self.assertEqual(ic.shape, (factor.shape[0],))

    def test_last_row_is_nan_for_h1(self):
        factor, close = self._make_panel()
        fr = fwd_return(close, 1)
        ic = rank_ic_series(factor, fr)
        self.assertTrue(np.isnan(ic[-1]))

    def test_range_minus1_to_plus1(self):
        factor, close = self._make_panel()
        fr = fwd_return(close, 1)
        ic = rank_ic_series(factor, fr)
        finite = ic[np.isfinite(ic)]
        self.assertTrue(np.all(finite >= -1.0 - 1e-9))
        self.assertTrue(np.all(finite <= +1.0 + 1e-9))

    def test_perfect_positive_correlation(self):
        """If factor = fwd_ret (perfectly sorted), IC must be +1 each row."""
        T, N = 10, 8
        # Create strictly increasing fwd_ret per row
        rng = np.random.RandomState(0)
        base = rng.randn(T, N)
        fr = np.sort(base, axis=1)          # sorted ascending
        factor = np.sort(base, axis=1)       # identical ranks
        ic = rank_ic_series(factor, fr)
        finite = ic[np.isfinite(ic)]
        np.testing.assert_allclose(finite, 1.0, atol=1e-8)

    def test_perfect_negative_correlation(self):
        T, N = 10, 8
        rng = np.random.RandomState(1)
        base = np.sort(rng.randn(T, N), axis=1)
        factor = base
        fr = base[:, ::-1]  # reversed ranks → IC = -1
        ic = rank_ic_series(factor, fr)
        finite = ic[np.isfinite(ic)]
        np.testing.assert_allclose(finite, -1.0, atol=1e-8)

    def test_nan_in_factor_row_produces_nan_ic(self):
        T, N = 5, 4
        factor = np.ones((T, N))
        factor[2] = np.nan  # entire row is NaN
        fr = np.ones((T, N)) * 0.01
        fr[-1] = np.nan     # last row NaN from fwd_return
        ic = rank_ic_series(factor, fr)
        self.assertTrue(np.isnan(ic[2]))


# ---------------------------------------------------------------------------
# ic_summary
# ---------------------------------------------------------------------------

class TestIcSummary(unittest.TestCase):

    def test_zero_mean_ic(self):
        rng = np.random.RandomState(0)
        ic = rng.randn(200)
        s = ic_summary(ic)
        self.assertAlmostEqual(s["mean_ic"], float(np.mean(ic)), places=10)

    def test_known_mean_std(self):
        ic = np.array([0.1, 0.2, 0.3])
        s = ic_summary(ic)
        self.assertAlmostEqual(s["mean_ic"], 0.2, places=10)
        self.assertAlmostEqual(s["std_ic"],  float(np.std(ic, ddof=1)), places=10)

    def test_ic_ir(self):
        ic = np.array([0.1, 0.1, 0.1])
        s = ic_summary(ic)
        # std_ic = 0 → IR and t_stat may be inf or nan; mean_ic should be 0.1
        self.assertAlmostEqual(s["mean_ic"], 0.1, places=10)

    def test_n_obs_ignores_nan(self):
        ic = np.array([0.1, np.nan, 0.2, np.nan, 0.3])
        s = ic_summary(ic)
        self.assertEqual(s["n_obs"], 3)

    def test_p_value_in_range(self):
        ic = np.linspace(-0.3, 0.3, 100)
        s = ic_summary(ic)
        self.assertGreaterEqual(s["p_value"], 0.0)
        self.assertLessEqual(s["p_value"],    1.0)

    def test_high_ic_low_pvalue(self):
        # Strong positive IC → p-value should be very small
        ic = np.full(200, 0.3) + np.random.RandomState(0).randn(200) * 0.01
        s = ic_summary(ic)
        self.assertLess(s["p_value"], 0.001)

    def test_keys_present(self):
        s = ic_summary(np.linspace(-0.1, 0.1, 50))
        for key in ("mean_ic", "std_ic", "ic_ir", "t_stat", "p_value", "n_obs"):
            self.assertIn(key, s)


# ---------------------------------------------------------------------------
# fdr_correct — Benjamini-Hochberg
# ---------------------------------------------------------------------------

class TestFdrCorrect(unittest.TestCase):

    def test_all_null(self):
        """Uniform p-values → BH should reject nothing."""
        rng = np.random.RandomState(7)
        p = rng.uniform(0.1, 1.0, 100)
        rejected, _ = fdr_correct(p, q=0.05)
        # Not guaranteed zero, but most should pass
        self.assertLess(rejected.sum(), 10)

    def test_all_signal(self):
        """Very small p-values → BH should reject all."""
        p = np.full(20, 1e-10)
        rejected, bh_p = fdr_correct(p, q=0.05)
        self.assertTrue(np.all(rejected))

    def test_mixed(self):
        """5 tiny p-values mixed with 95 large → at least the tiny ones rejected."""
        p = np.concatenate([np.full(5, 1e-8), np.full(95, 0.9)])
        rejected, _ = fdr_correct(p, q=0.05)
        # First 5 should be rejected
        self.assertTrue(np.all(rejected[:5]))

    def test_bh_adjusted_leq_1(self):
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        _, bh_p = fdr_correct(p, q=0.05)
        self.assertTrue(np.all(bh_p <= 1.0 + 1e-12))
        self.assertTrue(np.all(bh_p >= 0.0))

    def test_bh_adjusted_monotone_with_original(self):
        """BH-adjusted p-values must be >= original p-values."""
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        _, bh_p = fdr_correct(p, q=0.05)
        self.assertTrue(np.all(bh_p >= p - 1e-12))

    def test_single_p_value(self):
        rejected, bh_p = fdr_correct(np.array([0.03]), q=0.05)
        self.assertTrue(rejected[0])
        self.assertAlmostEqual(bh_p[0], 0.03)

    def test_known_result(self):
        """Manually verified BH example from Benjamini & Hochberg (1995)."""
        # m=10 tests; 4 have tiny p-values → all 4 rejected at q=0.05
        p = np.array([0.001, 0.002, 0.006, 0.019, 0.100,
                      0.200, 0.300, 0.400, 0.500, 0.600])
        rejected, bh_p = fdr_correct(p, q=0.05)
        # At rank k=4: threshold = 0.05*4/10 = 0.02; p[3]=0.019 < 0.02 → reject
        # At rank k=5: threshold = 0.05*5/10 = 0.025; p[4]=0.10 > 0.025 → accept
        self.assertEqual(rejected.sum(), 4)
        self.assertTrue(np.all(rejected[:4]))
        self.assertFalse(np.any(rejected[4:]))

    def test_output_shapes(self):
        p = np.random.RandomState(0).uniform(0, 1, 30)
        rejected, bh_p = fdr_correct(p)
        self.assertEqual(rejected.shape, p.shape)
        self.assertEqual(bh_p.shape, p.shape)
        self.assertEqual(rejected.dtype, bool)


# ---------------------------------------------------------------------------
# ic_summary_hac — Newey-West HAC standard error
# ---------------------------------------------------------------------------

class TestHacIcSummary(unittest.TestCase):

    def test_known_answer_hac_tstat(self):
        """Hand-computed HAC t for a small IC series with Bartlett lag=1."""
        ic = np.array([0.1, -0.2, 0.15, 0.3, -0.05])
        n = len(ic)
        mean = float(ic.mean())        # 0.06
        u = ic - mean
        g0 = float(np.dot(u, u)) / (n - 1)          # 0.03675
        g1 = float(np.dot(u[1:], u[:-1])) / (n - 1) # −0.00965
        S_hat = g0 + 2.0 * 0.5 * g1                 # 0.0271  (w(1)=0.5)
        t_expected = mean * math.sqrt(n) / math.sqrt(S_hat)  # ≈ 0.8150

        s = ic_summary_hac(ic, lag=1)
        self.assertAlmostEqual(s["hac_t_stat"], t_expected, places=8)

    def test_degenerate_lag0_equals_ols(self):
        """When lag=0, HAC t must be numerically identical to OLS t."""
        rng = np.random.RandomState(42)
        ic = rng.randn(100)
        s_ols = ic_summary(ic)
        s_hac = ic_summary_hac(ic, lag=0)
        self.assertAlmostEqual(s_hac["hac_t_stat"], s_ols["t_stat"], places=10)

    def test_iid_hac_close_to_ols(self):
        """For large i.i.d. IC series, HAC t ≈ OLS t (within 5 % relative)."""
        rng = np.random.RandomState(7)
        ic = rng.randn(600)
        s_ols = ic_summary(ic)
        s_hac = ic_summary_hac(ic, lag=4)  # lag = H-1 for H=5
        if math.isfinite(s_ols["t_stat"]) and math.isfinite(s_hac["hac_t_stat"]):
            ratio = abs(s_hac["hac_t_stat"]) / abs(s_ols["t_stat"])
            self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_overlapping_hac_substantially_smaller_than_ols(self):
        """IC from overlapping-window returns: |HAC t| < |OLS t| by a clear margin."""
        rng = np.random.RandomState(13)
        H = 20
        # Simulate IC series with strong positive autocorrelation:
        # running mean of WN over H bars mimics IC from 20d overlapping returns
        n_raw = 3000
        raw = rng.randn(n_raw)
        ic = np.convolve(raw, np.ones(H) / H, mode="valid") + 0.05

        s_ols = ic_summary(ic)
        s_hac = ic_summary_hac(ic, lag=H - 1)
        if math.isfinite(s_ols["t_stat"]) and math.isfinite(s_hac["hac_t_stat"]):
            # OLS over-counts due to autocorrelation; HAC must be at least 30% smaller
            self.assertGreater(abs(s_ols["t_stat"]), abs(s_hac["hac_t_stat"]) * 1.3)

    def test_determinism(self):
        """Same input → identical HAC output on two independent calls."""
        ic = np.array([0.05, -0.1, 0.08, 0.12, -0.03, 0.07, 0.02, -0.05,
                       0.11, 0.06, -0.07, 0.09])
        s1 = ic_summary_hac(ic, lag=2)
        s2 = ic_summary_hac(ic, lag=2)
        for key in s1:
            v1, v2 = s1[key], s2[key]
            if isinstance(v1, float) and math.isfinite(v1):
                self.assertEqual(v1, v2, msg=f"non-deterministic key: {key}")
            else:
                self.assertEqual(v1, v2)

    def test_hac_returns_required_keys(self):
        ic = np.linspace(-0.1, 0.1, 50)
        s = ic_summary_hac(ic, lag=3)
        for key in ("mean_ic", "std_ic", "t_stat", "p_value", "n_obs",
                    "hac_t_stat", "hac_p_value"):
            self.assertIn(key, s)

    def test_empty_ic_returns_nan(self):
        ic = np.array([np.nan, np.nan])
        s = ic_summary_hac(ic, lag=1)
        self.assertEqual(s["n_obs"], 0)
        self.assertTrue(math.isnan(s["hac_t_stat"]))


# ---------------------------------------------------------------------------
# ic_summary_nonoverlap — non-overlapping subsample t-test
# ---------------------------------------------------------------------------

class TestNonOverlapIcSummary(unittest.TestCase):

    def test_h1_equals_full_ols(self):
        """For h=1 the subsampled t-stat matches ic_summary on the full series."""
        rng = np.random.RandomState(5)
        ic = rng.randn(100)
        s_ols = ic_summary(ic)
        s_no  = ic_summary_nonoverlap(ic, h=1)
        self.assertAlmostEqual(s_no["t_stat"], s_ols["t_stat"], places=10)

    def test_h_greater_than_1_reduces_n(self):
        """Subsampling by h reduces n_obs to ceil(T / h)."""
        ic = np.ones(100) * 0.05
        h = 5
        s = ic_summary_nonoverlap(ic, h=h)
        expected_n = len(np.ones(100)[::h])  # 20
        self.assertEqual(s["n_obs"], expected_n)

    def test_returns_required_keys(self):
        ic = np.linspace(-0.1, 0.1, 60)
        s = ic_summary_nonoverlap(ic, h=10)
        for key in ("mean_ic", "std_ic", "t_stat", "p_value", "n_obs"):
            self.assertIn(key, s)

    def test_nan_excluded_before_subsample(self):
        """NaN values in the input are excluded before subsampling."""
        ic = np.array([0.1, np.nan, 0.2, np.nan, 0.3, np.nan, 0.4, np.nan])
        s = ic_summary_nonoverlap(ic, h=1)
        self.assertEqual(s["n_obs"], 4)  # 4 finite values


if __name__ == "__main__":
    unittest.main()
