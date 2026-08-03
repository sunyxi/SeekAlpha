# Known Limitations

## Raw report verification in CI

Research JSON outputs under `reports/*.json` are create-only local artifacts and
are gitignored, while `reports/manifest.json` is committed. A normal GitHub
Actions checkout therefore validates the manifest structure but cannot verify
the SHA-256 or schema of absent raw reports. `scripts/validate_manifest.py`
reports this distinction explicitly as `manifest_status: passed` and
`raw_reports_status: not-run`. Raw-report validation is `passed` only in an
environment where every manifest entry is present, hash-matched, and every
available `wf_select` report passes schema validation.

The committed Markdown summaries and manifest hashes are audit pointers, not a
substitute for retaining the corresponding immutable raw JSON in controlled
research storage.

## Data pump non-idempotency (IEX backfill)

### Symptom

Two independent runs of `scripts/local_pump.py` over the same `--start`/`--end`
range can produce different trade counts.  The known instance is:

| Run | Trade count | Report |
|-----|-------------|--------|
| M2 baseline | 959 233 | `reports/orb045-wf.json` |
| M4 meta-label | 959 221 | `reports/orb045-ml.json` |

The 12-trade discrepancy causes `scripts/compare_runs.py` to reject the two
reports as mismatched inputs unless `--allow-input-mismatch` is supplied.
Fold 14 also shows different `rejection_reasons` between the two runs, which
indicates that at least one candidate crossed a selection gate in one run but
not the other — consistent with the trade count difference.

### Root cause

Alpaca's IEX feed is a **single-exchange print feed** with delayed backfill.
Bars for the most recent weeks (sometimes months) of the requested window are
still being adjusted as late prints arrive.  When a subsequent run re-downloads
the same date range, it fetches marginally different 1-minute bars for the
tail of the window, which propagates through 5-minute consolidation to produce
slightly different RVOL/ATR values and, in edge cases, different breakout
signals.

The `cache_manifest.json` written by `local_pump.py` into each shard directory
records the SHA-256 of every `data/<SYMBOL>_<start>_<end>_1min.csv.gz` file
used for that run, so any two shard directories can be compared to confirm
whether they were built from exactly the same data snapshot.

### Impact on conclusions

- **M2 (No-Go) and M4 (No-Go) decisions are unaffected.**  The outer-test
  metrics differ by less than 0.1% across all cost scenarios; neither run
  approaches any decision gate.
- The 12-trade difference is a data-quality footnote, not a reproducibility
  failure in the strategy evaluation sense.

### Mitigation

1. **Freeze data before running ML variants.**  After the M2 baseline run,
   keep the cache directory read-only (`chmod -R a-w data/`).  Pass the same
   `--cache-dir` to the M4 run so it reuses the frozen files without
   re-downloading.
2. **Use `cache_manifest.json` as a provenance check.**  Before calling
   `compare_runs.py`, verify that the SHA-256 of each symbol file matches
   between the two shard directories.  If they differ, the comparison result
   should be treated with caution and `--allow-input-mismatch` documented
   explicitly in the report narrative.
3. **Pin the data window.**  Set `--end` to a date at least 30 calendar days
   in the past so IEX backfill for the tail of the window is complete before
   the first download.

### Related

- `reports/orb045-ml-summary.md` — notes the 959 233 vs 959 221 discrepancy
- `scripts/compare_runs.py` — `check_input_meta()` detects `trade_count`
  mismatches and exits non-zero without `--allow-input-mismatch`
- `tests/test_compare_runs.py::TestInputMetaCheck::test_regression_wf_vs_ml_reports_rejected`
  — regression test that confirms the known mismatch is detected
