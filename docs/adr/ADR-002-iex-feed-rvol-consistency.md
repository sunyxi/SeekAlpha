# ADR-002: Alpaca IEX feed for RVOL self-consistency

**Date**: 2026-08-03  
**Status**: Accepted  
**Deciders**: project lead

## Context

Relative Volume (RVOL) compares the current session's cumulative volume at a
given intraday minute mark against the median of the same mark across the last
20 trading days. The validity of this ratio depends on the numerator and
denominator being drawn from the same data source and the same feed type.

Two feed options are available through the Alpaca free tier:
- **IEX**: a single-exchange feed (Investors Exchange). Absolute volume is
  lower than the consolidated tape because it reflects only IEX prints.
- **SIP** (consolidated tape): full market volume, requires a paid subscription.

## Decision

Use the Alpaca IEX free feed (`feed="iex"`) for all 1-minute bar downloads.

## Consequences

**Easier**:
- No Alpaca subscription cost; free paper-account keys are sufficient.
- RVOL is internally consistent: both the "today so far" numerator and the
  historical-median denominator come from the same IEX feed, so the ratio
  correctly measures relative activity on that exchange.
- Reproducible: anyone with a free Alpaca account can regenerate the data.

**Harder**:
- Absolute volume figures are lower than the full consolidated tape. This means
  RVOL values are valid for ranking and filtering within this dataset but are
  **not directly comparable** to RVOL values computed from consolidated-tape
  data or published by other vendors.
- Any report that cites RVOL thresholds must state the IEX feed limitation.
  This is enforced by the "Known limitations" section in the README.

## Alternatives considered

- **SIP feed**: more representative of true market activity but requires a paid
  Alpaca subscription. Rejected to keep the project free and reproducible for
  anyone with a paper account.
- **Mixing feeds** (IEX for intraday, SIP for history): rejected because it
  would break the self-consistency invariant that makes RVOL meaningful.
- **Yahoo Finance / other free sources**: rejected due to lack of a supported
  Python client with consistent minute-bar delivery and adjustment semantics.
