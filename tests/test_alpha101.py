"""Tests for Alpha-101 factor library (src/orb/features/alpha101.py).

Verifies: output shape, warm-up NaN, finite values after warm-up,
no-future-data (point-in-time) property, and determinism.
"""

from __future__ import annotations

import unittest

import numpy as np

from orb.features.alpha101 import Alpha101, _rank, _delay, _delta, _ts_corr, _ts_std


# ---------------------------------------------------------------------------
# Shared synthetic panel
# ---------------------------------------------------------------------------

T, N = 310, 8   # T > longest window (250+); N assets

def _make_panel(seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    close = np.cumprod(1 + rng.randn(T, N) * 0.015, axis=0) * 50.0
    open_ = close * (1 + rng.randn(T, N) * 0.005)
    high  = np.maximum(close, open_) * (1 + np.abs(rng.randn(T, N)) * 0.003)
    low   = np.minimum(close, open_) * (1 - np.abs(rng.randn(T, N)) * 0.003)
    volume = np.abs(rng.randn(T, N)) * 1e6 + 1e5
    return dict(open_=open_, high=high, low=low, close=close, volume=volume)


_PANEL = _make_panel()
_ALPHA = Alpha101(**_PANEL)
_ALL_NAMES = Alpha101.all_alpha_names()


# ---------------------------------------------------------------------------
# Helper: perturb the last row to test no-look-ahead
# ---------------------------------------------------------------------------

def _perturbed_alpha() -> Alpha101:
    """Return an Alpha101 whose last row is wildly different."""
    p = {k: v.copy() for k, v in _PANEL.items()}
    p["close"][-1]  *= 99.0
    p["open_"][-1]  *= 99.0
    p["high"][-1]   *= 99.0
    p["low"][-1]    *= 99.0
    p["volume"][-1] *= 99.0
    return Alpha101(**p)


_ALPHA_PERTURBED = _perturbed_alpha()


# ---------------------------------------------------------------------------
# Tests: output shape
# ---------------------------------------------------------------------------

class TestOutputShape(unittest.TestCase):

    def _check(self, name: str) -> None:
        out = getattr(_ALPHA, name)()
        self.assertEqual(out.shape, (T, N),
                         f"{name}: expected ({T},{N}), got {out.shape}")

    def test_all_shapes(self):
        for name in _ALL_NAMES:
            with self.subTest(alpha=name):
                self._check(name)


# Alphas that require no historical lookback — purely elementwise or
# cross-sectional on the current row.  Row 0 is legitimately finite.
_INSTANTANEOUS = frozenset({"alpha033", "alpha041", "alpha042", "alpha054"})


# ---------------------------------------------------------------------------
# Tests: warm-up NaN (first rows must be NaN for every factor)
# ---------------------------------------------------------------------------

class TestWarmUpNaN(unittest.TestCase):

    def test_row0_is_nan_for_rolling_alphas(self):
        """Rolling alphas must be NaN at row 0 (no history available)."""
        for name in _ALL_NAMES:
            if name in _INSTANTANEOUS:
                continue
            out = getattr(_ALPHA, name)()
            with self.subTest(alpha=name):
                self.assertTrue(
                    np.all(np.isnan(out[0])),
                    f"{name}: row 0 should be all-NaN (rolling alpha, no prior data)",
                )

    def test_row0_finite_for_instantaneous_alphas(self):
        """Instantaneous alphas (no lookback) must have finite values at row 0."""
        for name in _INSTANTANEOUS:
            out = getattr(_ALPHA, name)()
            with self.subTest(alpha=name):
                self.assertTrue(
                    np.any(np.isfinite(out[0])),
                    f"{name}: row 0 should be finite (no rolling dependency)",
                )

    def test_late_rows_have_finite_values(self):
        """After sufficient warm-up, each alpha must produce some finite values."""
        for name in _ALL_NAMES:
            out = getattr(_ALPHA, name)()
            with self.subTest(alpha=name):
                finite_count = np.isfinite(out).sum()
                self.assertGreater(
                    finite_count, 0,
                    f"{name}: no finite values at all",
                )

    def test_last_row_has_finite_values(self):
        """The last row (T-1) should contain at least some finite values for all alphas."""
        for name in _ALL_NAMES:
            out = getattr(_ALPHA, name)()
            with self.subTest(alpha=name):
                self.assertTrue(
                    np.any(np.isfinite(out[-1])),
                    f"{name}: last row is all-NaN (warm-up too long for T={T})",
                )


# ---------------------------------------------------------------------------
# Tests: no future data (point-in-time property)
# ---------------------------------------------------------------------------

class TestNoFutureData(unittest.TestCase):
    """Perturbing only the last row of inputs must not affect earlier outputs."""

    def test_all_alphas_no_lookahead(self):
        for name in _ALL_NAMES:
            out_orig = getattr(_ALPHA, name)()
            out_pert = getattr(_ALPHA_PERTURBED, name)()
            with self.subTest(alpha=name):
                # All rows except the last must be identical
                np.testing.assert_array_equal(
                    out_orig[:-1],
                    out_pert[:-1],
                    err_msg=f"{name}: look-ahead detected — row t-1 changed when row t was perturbed",
                )


# ---------------------------------------------------------------------------
# Tests: determinism (no RNG)
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_same_input_same_output(self):
        a1 = Alpha101(**_PANEL)
        a2 = Alpha101(**_PANEL)
        for name in _ALL_NAMES:
            out1 = getattr(a1, name)()
            out2 = getattr(a2, name)()
            with self.subTest(alpha=name):
                np.testing.assert_array_equal(
                    out1, out2,
                    err_msg=f"{name}: non-deterministic output",
                )


# ---------------------------------------------------------------------------
# Tests: factor count
# ---------------------------------------------------------------------------

class TestFactorCount(unittest.TestCase):

    def test_at_least_40_factors(self):
        self.assertGreaterEqual(
            len(_ALL_NAMES), 40,
            f"Expected >= 40 factors; got {len(_ALL_NAMES)}: {_ALL_NAMES}",
        )


# ---------------------------------------------------------------------------
# Tests: internal operator correctness
# ---------------------------------------------------------------------------

class TestOperators(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(7)
        self.x = self.rng.randn(20, 4)

    def test_rank_range(self):
        """Cross-sectional rank must lie in [1/N, 1] for finite values."""
        r = _rank(self.x)
        finite = r[np.isfinite(r)]
        self.assertTrue(np.all(finite >= 1.0 / 4 - 1e-9))
        self.assertTrue(np.all(finite <= 1.0 + 1e-9))

    def test_rank_row_mean_approx_half(self):
        """Average cross-sectional rank ≈ (N+1)/(2N) for large N."""
        big = self.rng.randn(100, 20)
        r = _rank(big)
        row_means = np.nanmean(r, axis=1)
        expected = (20 + 1) / (2 * 20)
        np.testing.assert_allclose(row_means, expected, atol=0.01)

    def test_delay_shifts_correctly(self):
        x = np.arange(10).reshape(5, 2).astype(float)
        d = _delay(x, 2)
        self.assertTrue(np.all(np.isnan(d[:2])))
        np.testing.assert_array_equal(d[2:], x[:-2])

    def test_delta_is_difference(self):
        x = np.arange(10).reshape(5, 2).astype(float)
        d = _delta(x, 1)
        np.testing.assert_array_equal(d[1:], x[1:] - x[:-1])

    def test_ts_corr_perfect_correlation(self):
        """A series perfectly correlated with itself should give corr = 1."""
        x = np.cumsum(self.rng.randn(50, 3), axis=0)
        c = _ts_corr(x, x, 10)
        finite = c[np.isfinite(c)]
        np.testing.assert_allclose(finite, 1.0, atol=1e-8)

    def test_ts_std_nonnegative(self):
        x = self.rng.randn(30, 5)
        s = _ts_std(x, 5)
        finite = s[np.isfinite(s)]
        self.assertTrue(np.all(finite >= 0))

    def test_delay_zero_returns_copy(self):
        x = self.rng.randn(10, 3)
        np.testing.assert_array_equal(_delay(x, 0), x)


if __name__ == "__main__":
    unittest.main()
