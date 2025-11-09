# SeekAlpha

This repository contains a TradingView Pine Script strategy that combines multiple trend, momentum, volatility, and volume indicators to highlight high-confluence trading opportunities.

## Files

- `pine/seekalpha_multi_indicator_strategy.pine` &mdash; Pine Script v6 strategy that fuses DMI/ADX, Ichimoku Cloud, TTM Squeeze, SuperTrend, OBV, and MFI to produce long/short signals along with alert conditions and visual aids.

## Usage

1. Open TradingView and create a new strategy script.
2. Copy the contents of `pine/seekalpha_multi_indicator_strategy.pine` into the editor.
3. Click **Add to chart** to compile and apply the strategy.
4. Adjust the input parameters in the script settings to fit the asset and timeframe you trade.
5. Use the built-in alert conditions (`Composite Bullish Entry`, `Composite Bearish Entry`) to receive notifications when all confirmation criteria align.

> **Note:** This strategy is intended for educational purposes. Always validate signals with your own analysis and risk controls before trading live capital.
