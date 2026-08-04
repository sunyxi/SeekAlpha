# ADR-007: Freeze research protocol and enforce read-once retention access

**Date**: 2026-08-04

**Status**: Accepted

**Deciders**: SeekAlpha research maintainers

## Context

The strategy-discovery WBS needs a reproducible boundary before additional
experiments are run. Existing ORB and ML controls are valid for their existing
reports, but they do not provide one machine-readable protocol for new strategy
families or a durable audit record for final retention access.

## Decision

Store the new-strategy protocol in
`src/orb/research_protocol.json`. It declares disjoint development, outer-test,
and retention windows; experiment and parameter budgets; walk-forward purge and
embargo; cost scenarios; random seeds; promotion gates; and invalidation rules.

Expose the protocol through a stdlib-only loader that validates the contract and
computes a canonical SHA-256 hash. Retention access is reserved through an
atomic create-only filesystem marker keyed by protocol ID and experiment ID.
The marker is written before the loader runs, so a failed read still consumes
the one allowed access and requires a new experiment ID after investigation.

## Consequences

Experiments become auditable and comparable before strategy results are seen.
The filesystem ledger is portable and has no service dependency, but it must be
stored with the research artifacts and protected from deletion or manual edits.
The protocol does not download data, run models, or promote a strategy.

## Alternatives considered

- An in-memory flag was rejected because it does not survive process restarts.
- A mutable JSONL ledger was rejected because concurrent appenders could race.
- A database was deferred because the local research workflow needs a
  dependency-free, create-only control first.
