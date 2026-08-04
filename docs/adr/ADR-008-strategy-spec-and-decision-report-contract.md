# ADR-008: Strategy specification and decision report contract

**Date**: 2026-08-04

**Status**: Accepted

**Deciders**: SeekAlpha research maintainers

## Context

Strategy research needs a falsifiable hypothesis and provenance boundary before
implementation. Existing walk-forward reports validate ORB-specific fields but
do not bind a new strategy result to its predeclared universe, features, label,
costs, budget, protocol, and data snapshot.

## Decision

Use `src/orb/strategy_spec_template.json` as the source-of-truth field contract.
`StrategySpec` validates a concrete specification and computes a canonical
SHA-256 `spec_hash`. `DecisionReport` accepts only Candidate, No-Go,
Exploratory, or Invalid states and requires matching `spec_hash`, protocol hash,
data-manifest hash, code commit, budget, evidence, reasons, and gate statuses.

Reports are published create-only. Markdown summaries are rendered from the
validated JSON objects and the tracked template; generated output is never
hand-edited.

## Consequences

Hypotheses and results are auditable and cannot silently drift apart. This adds
validation work before experiments and does not provide strategy logic, data
access, or parameter tuning.

## Alternatives considered

- Extending the ORB report validator was rejected because it would couple a
  generic contract to ORB-only fields.
- A third-party JSON Schema dependency was deferred; the stdlib validator keeps
  the contract usable in the existing research environment.
