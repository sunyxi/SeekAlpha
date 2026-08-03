"""Alpha-101 cross-sectional factor library.

Reference: Kakushadze, Z. (2016). 101 Formulaic Alphas.
Wilmott, 2016(84), 72-81.  https://arxiv.org/abs/1601.00991

All inputs are 2-D NumPy arrays with shape (T, N):
  T = number of trading days (oldest first)
  N = number of assets

Each ``alpha0NN()`` method returns an array of shape (T, N).  Rows in
the warm-up window contain NaN.  All operators are point-in-time: the
value at row t depends only on rows 0 … t.

Install requirements (per ADR-003):
    pip install -e ".[features]"   # or .[test]
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

# ---------------------------------------------------------------------------
# Internal operators — all shape (T, N) → (T, N)
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [1/N, 1] per row. NaN propagates."""
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=float)
    for t in range(T):
        row = x[t]
        mask = np.isfinite(row)
        n = int(mask.sum())
        if n < 2:
            continue
        vals = row[mask]
        out[t, mask] = (vals.argsort().argsort() + 1).astype(float) / n
    return out


def _delay(x: np.ndarray, d: int) -> np.ndarray:
    """Shift x forward by d rows; fill leading rows with NaN."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if d <= 0:
        return x.copy()
    out[:d] = np.nan
    out[d:] = x[:-d]
    return out


def _delta(x: np.ndarray, d: int) -> np.ndarray:
    return np.asarray(x, dtype=float) - _delay(x, d)


def _sign(x: np.ndarray) -> np.ndarray:
    return np.sign(np.asarray(x, dtype=float))


def _log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log(np.where(x > 0, x, np.nan))


def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x, dtype=float))


def _signedpower(x: np.ndarray, a: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** a


def _scale(x: np.ndarray) -> np.ndarray:
    """Cross-sectional: divide each row by sum(|row|). NaN-safe."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        s = np.nansum(np.abs(x[t]))
        if s > 0:
            out[t] = x[t] / s
    return out


def _wins(x: np.ndarray, d: int):
    """Sliding windows of shape (T-d+1, N, d), or None if d > T."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if d > T:
        return None
    return _swv(x, d, axis=0)   # (T-d+1, N, d)


def _pad(x: np.ndarray, w, d: int) -> np.ndarray:
    """Fill a (T, N) output: NaN for first d-1 rows, then w result."""
    out = np.full_like(x, np.nan, dtype=float)
    if w is not None:
        out[d - 1:] = w
    return out


def _ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.sum(axis=-1) if w is not None else None, d)


def _ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.mean(axis=-1) if w is not None else None, d)


def _ts_std(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.std(axis=-1, ddof=1) if w is not None else None, d)


def _ts_min(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.min(axis=-1) if w is not None else None, d)


def _ts_max(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.max(axis=-1) if w is not None else None, d)


def _ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.argmax(axis=-1).astype(float) if w is not None else None, d)


def _ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    w = _wins(x, d)
    return _pad(x, w.argmin(axis=-1).astype(float) if w is not None else None, d)


def _ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Fraction of window values ≤ last value (time-series rank in [1/d, 1])."""
    w = _wins(x, d)
    if w is None:
        return _pad(x, None, d)
    last = w[..., -1:]
    return _pad(x, (w <= last).sum(axis=-1) / float(d), d)


def _ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling d-period time-series correlation per asset."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    wx, wy = _wins(x, d), _wins(y, d)
    if wx is None:
        return _pad(x, None, d)
    xm = wx - wx.mean(axis=-1, keepdims=True)
    ym = wy - wy.mean(axis=-1, keepdims=True)
    num = (xm * ym).sum(axis=-1)
    den = np.sqrt((xm ** 2).sum(axis=-1) * (ym ** 2).sum(axis=-1))
    safe_den = np.where(den > 1e-12, den, 1.0)
    return _pad(x, np.where(den > 1e-12, num / safe_den, 0.0), d)


def _ts_cov(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling d-period sample covariance per asset."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    wx, wy = _wins(x, d), _wins(y, d)
    if wx is None:
        return _pad(x, None, d)
    xm = wx - wx.mean(axis=-1, keepdims=True)
    ym = wy - wy.mean(axis=-1, keepdims=True)
    return _pad(x, (xm * ym).sum(axis=-1) / (d - 1), d)


def _decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Linearly decaying WMA: weights 1, 2, …, d (oldest→newest), normalised."""
    w = _wins(x, d)
    if w is None:
        return _pad(x, None, d)
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    return _pad(x, (w * weights).sum(axis=-1), d)


# ---------------------------------------------------------------------------
# Alpha101 class — 42 factors
# ---------------------------------------------------------------------------

class Alpha101:
    """42 Alpha-101 cross-sectional factors over daily OHLCV panel data.

    Parameters
    ----------
    open_, high, low, close, volume : ndarray, shape (T, N)
        Daily OHLCV. T rows (oldest first), N assets.
    vwap : ndarray, shape (T, N), optional
        Defaults to (open_ + high + low + close) / 4.
    """

    def __init__(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        vwap: np.ndarray | None = None,
    ) -> None:
        self.o = np.asarray(open_,  dtype=float)
        self.h = np.asarray(high,   dtype=float)
        self.l = np.asarray(low,    dtype=float)
        self.c = np.asarray(close,  dtype=float)
        self.v = np.asarray(volume, dtype=float)
        self.vwap = (
            np.asarray(vwap, dtype=float) if vwap is not None
            else (self.o + self.h + self.l + self.c) / 4.0
        )
        prev = _delay(self.c, 1)
        self.returns = np.where(prev > 0, (self.c - prev) / prev, np.nan)

    # ---------------------------------------------------------------- factors

    def alpha001(self) -> np.ndarray:
        """Rank(ArgMax(SignedPower(ret<0?Std(ret,20):close, 2), 5)) - 0.5
        (Kakushadze #1)"""
        x = np.where(self.returns < 0,
                     _ts_std(self.returns, 20),
                     self.c)
        return _rank(_ts_argmax(_signedpower(x, 2.0), 5)) - 0.5

    def alpha002(self) -> np.ndarray:
        """-Corr(Rank(Delta(Log(vol),2)), Rank((close-open)/open), 6)
        (Kakushadze #2)"""
        return -_ts_corr(
            _rank(_delta(_log(self.v), 2)),
            _rank((self.c - self.o) / np.where(self.o != 0, self.o, np.nan)),
            6,
        )

    def alpha003(self) -> np.ndarray:
        """-Corr(Rank(open), Rank(volume), 10)
        (Kakushadze #3)"""
        return -_ts_corr(_rank(self.o), _rank(self.v), 10)

    def alpha004(self) -> np.ndarray:
        """-TsRank(Rank(low), 9)
        (Kakushadze #4)"""
        return -_ts_rank(_rank(self.l), 9)

    def alpha005(self) -> np.ndarray:
        """Rank(open - Mean(vwap,10)) * -Abs(Rank(close - vwap))
        (Kakushadze #5)"""
        return _rank(self.o - _ts_mean(self.vwap, 10)) * (-_abs(_rank(self.c - self.vwap)))

    def alpha006(self) -> np.ndarray:
        """-Corr(open, volume, 10)
        (Kakushadze #6)"""
        return -_ts_corr(self.o, self.v, 10)

    def alpha008(self) -> np.ndarray:
        """-Rank(Sum(open,5)*Sum(ret,5) - Delay(...,10))
        (Kakushadze #8)"""
        x = _ts_sum(self.o, 5) * _ts_sum(self.returns, 5)
        return -_rank(x - _delay(x, 10))

    def alpha009(self) -> np.ndarray:
        """Conditional delta(close,1) vs -delta(close,1) via ts_min/max window 5.
        (Kakushadze #9)"""
        d1 = _delta(self.c, 1)
        return np.where(
            _ts_min(d1, 5) > 0, d1,
            np.where(_ts_max(d1, 5) < 0, d1, -d1),
        )

    def alpha010(self) -> np.ndarray:
        """Rank(alpha009 logic with window 4).
        (Kakushadze #10)"""
        d1 = _delta(self.c, 1)
        return _rank(np.where(
            _ts_min(d1, 4) > 0, d1,
            np.where(_ts_max(d1, 4) < 0, d1, -d1),
        ))

    def alpha011(self) -> np.ndarray:
        """(Rank(TsMax(vwap-close,3)) + Rank(TsMin(vwap-close,3))) * Rank(Delta(vol,3))
        (Kakushadze #11)"""
        spread = self.vwap - self.c
        return (_rank(_ts_max(spread, 3)) + _rank(_ts_min(spread, 3))) * _rank(_delta(self.v, 3))

    def alpha012(self) -> np.ndarray:
        """Sign(Delta(vol,1)) * -Delta(close,1)
        (Kakushadze #12)"""
        return _sign(_delta(self.v, 1)) * (-_delta(self.c, 1))

    def alpha013(self) -> np.ndarray:
        """-Rank(Cov(Rank(close), Rank(vol), 5))
        (Kakushadze #13)"""
        return -_rank(_ts_cov(_rank(self.c), _rank(self.v), 5))

    def alpha014(self) -> np.ndarray:
        """-Rank(Delta(ret,3)) * Corr(open, vol, 10)
        (Kakushadze #14)"""
        return -_rank(_delta(self.returns, 3)) * _ts_corr(self.o, self.v, 10)

    def alpha015(self) -> np.ndarray:
        """-Sum(Rank(Corr(Rank(high), Rank(vol), 3)), 3)
        (Kakushadze #15)"""
        return -_ts_sum(_rank(_ts_corr(_rank(self.h), _rank(self.v), 3)), 3)

    def alpha016(self) -> np.ndarray:
        """-Rank(Cov(Rank(high), Rank(vol), 5))
        (Kakushadze #16)"""
        return -_rank(_ts_cov(_rank(self.h), _rank(self.v), 5))

    def alpha018(self) -> np.ndarray:
        """-Rank(Std(|close-open|,5) + (close-open) + Corr(close,open,10))
        (Kakushadze #18)"""
        return -_rank(
            _ts_std(_abs(self.c - self.o), 5)
            + (self.c - self.o)
            + _ts_corr(self.c, self.o, 10)
        )

    def alpha019(self) -> np.ndarray:
        """-Sign(close-Delay(close,7)+Delta(close,7)) * (1+Rank(1+Sum(ret,250)))
        (Kakushadze #19)"""
        return (
            -_sign((self.c - _delay(self.c, 7)) + _delta(self.c, 7))
            * (1.0 + _rank(1.0 + _ts_sum(self.returns, 250)))
        )

    def alpha020(self) -> np.ndarray:
        """-Rank(open-Delay(high,1)) * Rank(open-Delay(close,1)) * Rank(open-Delay(low,1))
        (Kakushadze #20)"""
        return (
            -_rank(self.o - _delay(self.h, 1))
            * _rank(self.o - _delay(self.c, 1))
            * _rank(self.o - _delay(self.l, 1))
        )

    def alpha022(self) -> np.ndarray:
        """-(Delta(Corr(high,vol,5),5) * Rank(Std(close,20)))
        (Kakushadze #22)"""
        return -(_delta(_ts_corr(self.h, self.v, 5), 5) * _rank(_ts_std(self.c, 20)))

    def alpha023(self) -> np.ndarray:
        """If Mean(high,20) < high: -Delta(high,2), else 0.
        (Kakushadze #23)"""
        ma = _ts_mean(self.h, 20)
        result = np.where(ma < self.h, -_delta(self.h, 2), 0.0)
        return np.where(np.isfinite(ma), result, np.nan)

    def alpha024(self) -> np.ndarray:
        """If Delta(Mean(close,100),100)/Delay(close,100) < 0.05: -(close-TsMin(close,100))
        else -Delta(close,3).
        (Kakushadze #24)"""
        ma100 = _ts_mean(self.c, 100)
        denom = np.where(_delay(self.c, 100) != 0, _delay(self.c, 100), np.nan)
        cond = _delta(ma100, 100) / denom
        return np.where(cond < 0.05, -(self.c - _ts_min(self.c, 100)), -_delta(self.c, 3))

    def alpha026(self) -> np.ndarray:
        """-TsMax(Corr(TsRank(vol,5), TsRank(high,5), 5), 3)
        (Kakushadze #26)"""
        return -_ts_max(_ts_corr(_ts_rank(self.v, 5), _ts_rank(self.h, 5), 5), 3)

    def alpha027(self) -> np.ndarray:
        """If Rank(Sum(Corr(Rank(vol),Rank(vwap),6),2)/2) > 0.5: -1, else 1.
        (Kakushadze #27)"""
        inner = _ts_sum(_ts_corr(_rank(self.v), _rank(self.vwap), 6), 2) / 2.0
        r = _rank(inner)
        return np.where(np.isfinite(r), np.where(r > 0.5, -1.0, 1.0), np.nan)

    def alpha030(self) -> np.ndarray:
        """(1-Rank(ΔC-sign-sum)) * Sum(vol,5) / Sum(vol,20)
        (Kakushadze #30)"""
        s = (_sign(self.c - _delay(self.c, 1))
             + _sign(_delay(self.c, 1) - _delay(self.c, 2))
             + _sign(_delay(self.c, 2) - _delay(self.c, 3)))
        denom = np.where(_ts_sum(self.v, 20) != 0, _ts_sum(self.v, 20), np.nan)
        return (1.0 - _rank(s)) * _ts_sum(self.v, 5) / denom

    def alpha032(self) -> np.ndarray:
        """Scale(Mean(close,7)-close) + 20*Scale(Corr(vwap,Delay(close,5),230))
        (Kakushadze #32)"""
        return (
            _scale(_ts_mean(self.c, 7) - self.c)
            + 20.0 * _scale(_ts_corr(self.vwap, _delay(self.c, 5), 230))
        )

    def alpha033(self) -> np.ndarray:
        """Rank(-(1-open/close))
        (Kakushadze #33)"""
        denom = np.where(self.c != 0, self.c, np.nan)
        return _rank(-(1.0 - self.o / denom))

    def alpha034(self) -> np.ndarray:
        """Rank((1-Rank(Std(ret,2)/Std(ret,5))) + (1-Rank(Delta(close,1))))
        (Kakushadze #34)"""
        ratio = _ts_std(self.returns, 2) / np.where(
            _ts_std(self.returns, 5) != 0, _ts_std(self.returns, 5), np.nan
        )
        return _rank((1.0 - _rank(ratio)) + (1.0 - _rank(_delta(self.c, 1))))

    def alpha035(self) -> np.ndarray:
        """TsRank(vol,32) * (1-TsRank(close+high-low,16)) * (1-TsRank(ret,32))
        (Kakushadze #35)"""
        return (
            _ts_rank(self.v, 32)
            * (1.0 - _ts_rank(self.c + self.h - self.l, 16))
            * (1.0 - _ts_rank(self.returns, 32))
        )

    def alpha037(self) -> np.ndarray:
        """Rank(Corr(Delay(open-close,1), close, 200)) + Rank(open-close)
        (Kakushadze #37)"""
        return (
            _rank(_ts_corr(_delay(self.o - self.c, 1), self.c, 200))
            + _rank(self.o - self.c)
        )

    def alpha038(self) -> np.ndarray:
        """-Rank(TsRank(close,10)) * Rank(close/open)
        (Kakushadze #38)"""
        denom = np.where(self.o != 0, self.o, np.nan)
        return -_rank(_ts_rank(self.c, 10)) * _rank(self.c / denom)

    def alpha040(self) -> np.ndarray:
        """-Rank(Std(high,10)) * Corr(high, vol, 10)
        (Kakushadze #40)"""
        return -_rank(_ts_std(self.h, 10)) * _ts_corr(self.h, self.v, 10)

    def alpha041(self) -> np.ndarray:
        """(high * low)^0.5 - vwap
        (Kakushadze #41)"""
        return np.sqrt(np.where(self.h * self.l >= 0, self.h * self.l, np.nan)) - self.vwap

    def alpha042(self) -> np.ndarray:
        """Rank(vwap-close) / Rank(vwap+close)
        (Kakushadze #42)"""
        denom = _rank(self.vwap + self.c)
        return _rank(self.vwap - self.c) / np.where(denom != 0, denom, np.nan)

    def alpha044(self) -> np.ndarray:
        """-Corr(high, Rank(vol), 5)
        (Kakushadze #44)"""
        return -_ts_corr(self.h, _rank(self.v), 5)

    def alpha045(self) -> np.ndarray:
        """-(Rank(Mean(Delay(close,5),20)) * Corr(close,vol,2))
              * Rank(Corr(Sum(close,5), Sum(close,20), 2))
        (Kakushadze #45)"""
        return -(
            (_rank(_ts_mean(_delay(self.c, 5), 20)) * _ts_corr(self.c, self.v, 2))
            * _rank(_ts_corr(_ts_sum(self.c, 5), _ts_sum(self.c, 20), 2))
        )

    def alpha046(self) -> np.ndarray:
        """Momentum-reversal: diff>0.25→-1; diff<0→1; else -delta(close,1).
        diff = (Delay(c,20)-Delay(c,10))/10 - (Delay(c,10)-close)/10.
        (Kakushadze #46)"""
        diff = ((_delay(self.c, 20) - _delay(self.c, 10)) / 10.0
                - (_delay(self.c, 10) - self.c) / 10.0)
        return np.where(diff > 0.25, -1.0, np.where(diff < 0.0, 1.0, -_delta(self.c, 1)))

    def alpha049(self) -> np.ndarray:
        """If diff < -0.1: 1, else -delta(close,1).  diff as in alpha046.
        (Kakushadze #49)"""
        diff = ((_delay(self.c, 20) - _delay(self.c, 10)) / 10.0
                - (_delay(self.c, 10) - self.c) / 10.0)
        return np.where(diff < -0.1, 1.0, -_delta(self.c, 1))

    def alpha050(self) -> np.ndarray:
        """-TsMax(Rank(Corr(Rank(vol), Rank(vwap), 5)), 5)
        (Kakushadze #50)"""
        return -_ts_max(_rank(_ts_corr(_rank(self.v), _rank(self.vwap), 5)), 5)

    def alpha051(self) -> np.ndarray:
        """If diff < -0.05: 1, else -delta(close,1).  diff as in alpha046.
        (Kakushadze #51)"""
        diff = ((_delay(self.c, 20) - _delay(self.c, 10)) / 10.0
                - (_delay(self.c, 10) - self.c) / 10.0)
        return np.where(diff < -0.05, 1.0, -_delta(self.c, 1))

    def alpha053(self) -> np.ndarray:
        """-Delta((close-low-(high-close))/(close-low), 9)
        (Kakushadze #53)"""
        hl = np.where(self.c - self.l != 0, self.c - self.l, np.nan)
        return -_delta((self.c - self.l - (self.h - self.c)) / hl, 9)

    def alpha054(self) -> np.ndarray:
        """-(low-close)*(open^5) / ((low-high)*(close^5))
        (Kakushadze #54)"""
        num = -(self.l - self.c) * (self.o ** 5)
        den = (self.l - self.h) * (self.c ** 5)
        return num / np.where(den != 0, den, np.nan)

    def alpha055(self) -> np.ndarray:
        """-Corr(Rank((close-TsMin(low,12))/(TsMax(high,12)-TsMin(low,12))),Rank(vol),6)
        (Kakushadze #55)"""
        rng = _ts_max(self.h, 12) - _ts_min(self.l, 12)
        x = (self.c - _ts_min(self.l, 12)) / np.where(rng != 0, rng, np.nan)
        return -_ts_corr(_rank(x), _rank(self.v), 6)

    # ---------------------------------------------------------------- helpers

    @classmethod
    def all_alpha_names(cls) -> list[str]:
        """Return sorted list of all alpha method names."""
        return sorted(
            name for name in dir(cls)
            if name.startswith("alpha") and callable(getattr(cls, name))
        )
