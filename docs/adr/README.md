# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the SeekAlpha
project. An ADR documents a significant architectural decision: what was chosen,
why, and what alternatives were considered.

## Process

1. **When to write one**: any introduction of a new third-party dependency,
   any change to a frozen invariant, or any significant structural decision
   that future contributors should understand without reading the full history.

2. **File naming**: `ADR-NNN-short-kebab-title.md`, where `NNN` is a
   zero-padded three-digit sequence number.

3. **Status lifecycle**: `Proposed → Accepted | Rejected | Superseded`

4. **Immutability**: once `Accepted`, the file body must not be edited.
   Superseding decisions create a new ADR and update the `Status` field of
   the old one to `Superseded by ADR-NNN`.

## Template

```markdown
# ADR-NNN: Title

**Date**: YYYY-MM-DD  
**Status**: Proposed | Accepted | Rejected | Superseded by ADR-NNN  
**Deciders**: (names or roles)

## Context

What situation or constraint prompted this decision?

## Decision

What was decided?

## Consequences

What becomes easier or harder as a result?

## Alternatives considered

What other options were evaluated, and why were they rejected?
```

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-stdlib-only-core.md) | stdlib-only core and walk-forward evaluator | Accepted |
| [ADR-002](ADR-002-iex-feed-rvol-consistency.md) | Alpaca IEX feed for RVOL self-consistency | Accepted |
| ADR-003 | Introduce numpy in the feature layer (M3) | Pending |
| ADR-004 | ML classifier dependency for meta-labeling (M4) | Pending |
| ADR-005 | Optuna for Bayesian hyperparameter search (M4) | Pending |
| [ADR-006](ADR-006-optional-qlib-adapter.md) | Optional Microsoft Qlib research adapter | Accepted |
| [ADR-007](ADR-007-research-protocol-retention-ledger.md) | Freeze research protocol and enforce read-once retention access | Accepted |
| [ADR-008](ADR-008-strategy-spec-and-decision-report-contract.md) | Strategy specification and decision report contract | Accepted |
| [ADR-009](ADR-009-licensed-data-source-and-pit-universe.md) | Licensed data source policy and point-in-time universe | Accepted with blockers |
| [ADR-010](ADR-010-immutable-research-cache.md) | Immutable incremental research cache | Accepted |
