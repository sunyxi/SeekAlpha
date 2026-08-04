# Research Protocol

## Purpose

Issue #53 freezes the research boundary before new strategy experiments. The
source of truth is `src/orb/research_protocol.json`; do not copy its values into
strategy code or change it after results are observed.

## Frozen controls

- Development: 2021-01-04 through 2024-12-31.
- Outer test: 2025-01-01 through 2026-06-30.
- Retention: 2026-07-01 through 2026-12-31; available after 2027-01-01.
- Maximum three experiments per strategy family, twelve total.
- Maximum 192 parameter trials and 50 model trials per experiment.
- Walk-forward controls: 252 train days, 63 validation days, 63 outer-test
  days, 63-day step, 20-day purge, and 5-day embargo.
- Costs: zero, baseline 2.5, and double 5.0 bps per side.
- Every stochastic component must record a declared seed.

## CLI Usage

```bash
python3 scripts/validate_research_protocol.py --json
```

The command must print the protocol ID and SHA-256 hash. A retention evaluator
must call `RetentionLedger.read_once(experiment_id, loader)`. The ledger reserves
access before invoking `loader`; a failed loader remains consumed.

## Operations

1. Validate the protocol before an experiment.
2. Record its protocol hash, data manifest hash, search-space hash, seed, and
   experiment ID in the report.
3. Keep the ledger directory in immutable research storage.
4. Do not read retention before `retention_available_after`.

## Limitations

The protocol does not prove data quality, prevent a user with filesystem write
access from deleting markers, or provide a distributed audit service. Those
controls belong to later data and experiment-harness issues.

## Rollback

Do not edit the accepted JSON in place. If the protocol is wrong before any
result or retention access, revert the feature branch. After results exist,
create a new protocol ID and new experiment IDs, document the invalidation, and
rerun from development. Never delete a consumed ledger marker.
