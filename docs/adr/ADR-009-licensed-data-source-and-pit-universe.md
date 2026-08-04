# ADR-009: Licensed data source policy and point-in-time universe

**Date**: 2026-08-04

**Status**: Accepted with blockers

**Deciders**: SeekAlpha research maintainers

## Context

The existing Alpaca IEX cache is useful for a low-cost ORB baseline, but a new
strategy study must not use current constituents as historical membership or
silently drop delisted instruments. A source decision must also account for
historical bars, quotes, corporate actions, symbol changes, and licensing.

## Decision

Select Databento as the provisional primary source and Alpaca as the fallback.
Databento's official corporate-action documentation describes listed and
delisted security coverage and continuous listing records. Alpaca's official
historical bars and corporate-actions documentation supports the existing
fallback OHLCV/quote workflow. These sources are role selections, not an
authorization to download data.

The source-of-truth policy is `src/orb/data_source_policy.json`. It records the
capability matrix, required contracts, blockers, and prohibited shortcuts.
Membership is reconstructed by `instrument_id` and half-open listing intervals;
ticker symbols are aliases valid only within their interval. Common stocks are
the initial asset type; ETFs and current-only constituent lists are excluded.

## Blockers

Before bulk download, the project must obtain a Databento non-display research
entitlement, verify historical sector classification coverage, and record
provider costs and retention rights. Until then the decision remains
`selected_with_blockers` and no credentials or raw data may enter the repository.

## Consequences

Historical membership can be reconstructed without survivorship shortcuts. The
policy adds licensing and provenance work and deliberately prevents a convenient
but invalid fallback to current constituents.

## Alternatives considered

- Alpaca-only was rejected because the policy cannot establish point-in-time
  delisted membership or historical sector history from the existing contract.
- Current index/portfolio constituents were rejected because they introduce
  survivorship bias.
- Unlicensed web scraping was rejected for legal, reproducibility, and schema
  stability reasons.

## References

- [Databento corporate actions and reference data](https://databento.com/docs/venues-and-datasets/corporate-actions)
- [Alpaca historical stock bars](https://docs.alpaca.markets/us/v1.4.2/reference/stockbars)
- [Alpaca corporate actions update](https://docs.alpaca.markets/us/changelog/2026-06-03-market-data-9dddd18)
