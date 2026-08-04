# Qlib Research Adapter POC

This POC converts SeekAlpha's existing daily `DailyPanel` into a deterministic
Qlib-compatible pandas DataFrame. The index is `(datetime, instrument)` and the
columns are `$open`, `$high`, `$low`, `$close`, and `$volume`.

Architecture constraints are defined in
[ADR-006](adr/ADR-006-optional-qlib-adapter.md).

## CLI Usage

Install the optional runtime:

```bash
pip install -e ".[qlib]"
```

Export from the existing minute cache without downloading data:

```bash
python3 scripts/qlib_poc.py \
  --cache-dir data \
  --start 2021-01-04 \
  --end 2026-06-30 \
  --output derived/qlib-daily.csv \
  --verify-qlib
```

Omit `--verify-qlib` when only the tabular export is required. The exporter
still requires pandas through the `qlib` optional dependency group.

## Operations

- Treat `data/` as read-only input. The CLI reads existing
  `*_1min.csv.gz` files and never calls Alpaca or another data provider.
- Write exports outside `data/`, for example under `derived/`.
- Outputs are create-only. Choose a new path for every run and retain the
  command, Git commit, source-cache hashes, and output hash with the experiment.
- Run `python3 -m pytest tests/test_qlib_adapter.py tests/test_qlib_poc.py
  tests/test_qlib_runtime_integration.py -v` after environment changes.
- On Apple Silicon, Qlib may require the system OpenMP runtime. Install it only
  if the Qlib installation reports a missing OpenMP library.

## Limitations

- This POC validates a tabular integration boundary; it does not run Alpha158,
  train a model, tune parameters, or prove improved returns.
- The source panel is daily OHLCV derived from the existing IEX minute cache.
  It does not contain a consolidated US order book or full-market volume.
- `build_panel` currently scans each selected cache file completely before
  applying the requested date range, so narrow exports over multi-year cache
  files can still take several minutes.
- Missing pre-observation rows remain NaN. Existing post-observation forward
  fills from `DailyPanel` are preserved exactly.
- Qlib backtests do not replace SeekAlpha's frozen ORB lifecycle, cost scenarios,
  nested walk-forward, or independent execution validation.
- The Qlib runtime fixture is skipped when `pyqlib` is not installed and must be
  reported as skipped, not passed.
- With pyqlib 0.9.7, the runtime fixture currently emits three upstream numpy
  `Timedelta` deprecation warnings from `qlib.constant`; the fixture still
  passes, but a future numpy release may require a Qlib upgrade.

## Rollback

Remove `src/orb/qlib_adapter/`, `scripts/qlib_poc.py`, the `qlib` optional
dependency group, Qlib tests, ADR-006, and the three localized POC documents.
No cache, ORB core, walk-forward report, frozen factor list, or broker-facing
asset needs migration or rollback.
