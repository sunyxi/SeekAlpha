"""Rank IC evaluation and Benjamini-Hochberg FDR correction.

Implements Spearman rank IC against multi-horizon forward returns, IC summary
statistics (mean, std, IR, t-stat, two-sided p-value via normal approximation),
Newey-West HAC standard errors, non-overlapping subsample t-statistics,
and Benjamini-Hochberg FDR correction — all without scipy.

Public API
----------
fwd_return(close, h)               -> ndarray (T, N)  forward return at horizon h
rank_ic_series(factor, fwd_ret)    -> ndarray (T,)    Spearman IC per row
ic_summary(ic)                     -> dict             mean, std, IR, t_stat, p_value
ic_summary_hac(ic, lag)            -> dict             OLS stats + HAC t / p
ic_summary_nonoverlap(ic, h)       -> dict             OLS stats on every-h subsample
fdr_correct(p_values, q=0.05)      -> (rejected, bh_p_adjusted)
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
# HAC (Newey-West) IC summary

def ic_summary_hac(ic: np.ndarray, lag: int) -> dict:
    """IC summary with Newey-West (Bartlett-kernel) HAC standard error.

    Parameters
    ----------
    ic  : ndarray (T,) — IC time series; NaN values are excluded.
    lag : int          — Bartlett kernel bandwidth (= H - 1 for horizon H days).
                         lag=0 → HAC t equals OLS t exactly (no autocorrelation
                         correction applied).

    Returns
    -------
    dict with all keys from ic_summary plus:
      hac_t_stat  — HAC-corrected t-statistic
      hac_p_value — two-sided p-value from hac_t_stat (normal approximation)

    Notes
    -----
    Autocovariances use ddof=1 so that lag=0 gives HAC t == OLS t exactly:
        γ(k) = (1/(n-1)) Σ_{t=k}^{n-1} u_t · u_{t-k},  u_t = ic_t − mean
        S_hat = γ(0) + 2 Σ_{k=1}^{lag} (1 − k/(lag+1)) γ(k)
        t_HAC = mean · √n / √S_hat
    """
    ic = np.asarray(ic, dtype=float)
    finite = ic[np.isfinite(ic)]
    n = len(finite)

    base = ic_summary(ic)

    if n < 2:
        return {**base, "hac_t_stat": float("nan"), "hac_p_value": float("nan")}

    mean = float(finite.mean())
    u = finite - mean

    def _autocov(k: int) -> float:
        if k == 0:
            return float(np.dot(u, u)) / (n - 1)
        if k >= n:
            return 0.0
        return float(np.dot(u[k:], u[:-k])) / (n - 1)

    S = _autocov(0)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        S += 2.0 * w * _autocov(k)

    if S <= 0:
        hac_t = float("nan")
        hac_p = float("nan")
    else:
        hac_t = mean * math.sqrt(n) / math.sqrt(S)
        hac_p = math.erfc(abs(hac_t) / math.sqrt(2))

    return {**base, "hac_t_stat": float(hac_t), "hac_p_value": float(hac_p)}


# ---------------------------------------------------------------------------
# Non-overlapping subsample IC summary

def ic_summary_nonoverlap(ic: np.ndarray, h: int) -> dict:
    """IC summary on a non-overlapping subsample (every h-th finite observation).

    Parameters
    ----------
    ic : ndarray (T,) — IC time series; NaN values are excluded first.
    h  : int          — subsample stride (= horizon H in trading days).
                        h=1 → full series → identical to ic_summary(ic).

    Returns
    -------
    dict with the same keys as ic_summary, computed on the subsampled series.
    """
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    ic = np.asarray(ic, dtype=float)
    finite = ic[np.isfinite(ic)]
    return ic_summary(finite[::h])


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
