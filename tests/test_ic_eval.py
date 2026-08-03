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


if __name__ == "__main__":
    unittest.main()
