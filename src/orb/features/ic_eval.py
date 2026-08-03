"""Rank IC evaluation and Benjamini-Hochberg FDR correction.

Implements Spearman rank IC against multi-horizon forward returns, IC summary
statistics (mean, std, IR, t-stat, two-sided p-value via normal approximation),
and Benjamini-Hochberg FDR correction — all without scipy.

Public API
----------
fwd_return(close, h)             -> ndarray (T, N)  forward return at horizon h
rank_ic_series(factor, fwd_ret)  -> ndarray (T,)    Spearman IC per row
ic_summary(ic)                   -> dict             mean, std, IR, t_stat, p_value
fdr_correct(p_values, q=0.05)    -> (rejected, bh_p_adjusted)
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Cross-sectional rank helper

def _cs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [0, 1] per row (0 = lowest, 1 = highest).

    Rows with fewer than 2 finite values are returned as all-NaN.
    """
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=float)
    for t in range(T):
        row = x[t]
        mask = np.isfinite(row)
        n = int(mask.sum())
        if n < 2:
            continue
        # argsort of argsort gives rank in {0, 1, ..., n-1}; normalise to [0, 1]
        out[t, mask] = np.argsort(np.argsort(row[mask])) / (n - 1.0)
    return out


# ---------------------------------------------------------------------------
# Forward return

def fwd_return(close: np.ndarray, h: int) -> np.ndarray:
    """Compute h-day forward return for each (t, asset).

    forward_return[t] = (close[t+h] - close[t]) / close[t].
    Last h rows are NaN (future not yet available).
    """
    close = np.asarray(close, dtype=float)
    T, N = close.shape
    out = np.full((T, N), np.nan, dtype=float)
    if h >= T:
        return out
    denom = np.where(close[:-h] != 0, close[:-h], np.nan)
    out[:-h] = (close[h:] - close[:-h]) / denom
    return out


# ---------------------------------------------------------------------------
# Rank IC series (Spearman)

def rank_ic_series(
    factor: np.ndarray,   # (T, N)
    fwd_ret: np.ndarray,  # (T, N)
) -> np.ndarray:          # (T,)
    """Compute cross-sectional Spearman rank IC at each row.

    Returns NaN for rows where either array has fewer than 2 jointly-finite values.
    """
    factor  = np.asarray(factor,  dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    T, N = factor.shape
    rf = _cs_rank(factor)
    rr = _cs_rank(fwd_ret)
    ic = np.full(T, np.nan, dtype=float)
    for t in range(T):
        mask = np.isfinite(rf[t]) & np.isfinite(rr[t])
        n = int(mask.sum())
        if n < 2:
            continue
        a = rf[t, mask]
        b = rr[t, mask]
        a = a - a.mean()
        b = b - b.mean()
        num = float((a * b).sum())
        den = float(math.sqrt((a * a).sum() * (b * b).sum()))
        ic[t] = num / den if den > 1e-12 else 0.0
    return ic


# ---------------------------------------------------------------------------
# IC summary statistics

def ic_summary(ic: np.ndarray) -> dict:
    """Summarise an IC time series.

    Returns a dict with:
      mean_ic  — time-series mean of IC (NaN excluded)
      std_ic   — sample std (ddof=1, NaN excluded)
      ic_ir    — information ratio = mean_ic / std_ic
      t_stat   — t = mean_ic / (std_ic / sqrt(n_obs))
      p_value  — two-sided p-value via normal approx (valid for n >= 30)
      n_obs    — number of non-NaN IC observations
    """
    ic = np.asarray(ic, dtype=float)
    finite = ic[np.isfinite(ic)]
    n = len(finite)

    if n == 0:
        return dict(mean_ic=float("nan"), std_ic=float("nan"),
                    ic_ir=float("nan"), t_stat=float("nan"),
                    p_value=float("nan"), n_obs=0)

    mean = float(finite.mean())
    std  = float(finite.std(ddof=1)) if n > 1 else float("nan")

    if n > 1 and std > 0:
        ic_ir  = mean / std
        t_stat = mean / (std / math.sqrt(n))
        # Two-sided p-value: 2*(1-Φ(|t|)) = erfc(|t|/sqrt(2))
        p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    else:
        ic_ir  = float("nan")
        t_stat = float("nan")
        p_value = float("nan")

    return dict(
        mean_ic=mean,
        std_ic=std,
        ic_ir=float(ic_ir),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction

def fdr_correct(
    p_values: np.ndarray,
    q: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (no scipy).

    Parameters
    ----------
    p_values : ndarray, shape (m,)
    q : float
        False discovery rate threshold.

    Returns
    -------
    rejected : bool ndarray, shape (m,)
        True where the null hypothesis is rejected at FDR level q.
    bh_p : float ndarray, shape (m,)
        BH-adjusted p-values (step-up; always >= original p-values; capped at 1).
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    # Sort ascending
    order = np.argsort(p_values)
    ranked_p = p_values[order]

    # BH critical values: p(k) * m / k  (1-indexed k)
    ranks = np.arange(1, m + 1, dtype=float)
    adj = ranked_p * m / ranks

    # Step-up: enforce non-decreasing from right (take cummin from right)
    for i in range(m - 2, -1, -1):
        if adj[i] > adj[i + 1]:
            adj[i] = adj[i + 1]
    np.minimum(adj, 1.0, out=adj)

    # Map back to original order
    bh_p = np.empty(m, dtype=float)
    bh_p[order] = adj

    return bh_p <= q, bh_p
