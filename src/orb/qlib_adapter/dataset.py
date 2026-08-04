"""Convert SeekAlpha daily panels to Qlib-compatible tabular data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orb.features.panel_builder import DailyPanel

_QLIB_COLUMNS = ("$open", "$high", "$low", "$close", "$volume")


def to_qlib_frame(panel: DailyPanel) -> Any:
    """Return a deterministic ``(datetime, instrument)`` OHLCV DataFrame.

    The function preserves the panel's values, including NaNs and zero-volume
    forward-filled rows. It does not initialise Qlib, download data, or mutate
    the source panel.
    """
    expected_shape = (len(panel.symbols), len(panel.dates), len(_QLIB_COLUMNS))
    if panel.panel.shape != expected_shape:
        raise ValueError(
            f"panel shape {panel.panel.shape} does not match metadata {expected_shape}"
        )

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        from .runtime import QlibUnavailableError

        raise QlibUnavailableError(
            "Qlib adapter dependencies are unavailable; run: pip install -e .[qlib]"
        ) from exc

    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(panel.dates), panel.symbols],
        names=("datetime", "instrument"),
    )
    values = panel.panel.transpose(1, 0, 2).reshape(len(index), len(_QLIB_COLUMNS))
    return pd.DataFrame(values, index=index, columns=_QLIB_COLUMNS)
