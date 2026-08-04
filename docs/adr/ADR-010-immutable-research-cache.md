# ADR-010: Immutable incremental research cache

**Date**: 2026-08-04

**Status**: Accepted

**Deciders**: SeekAlpha research maintainers

## Context

The legacy ORB cache updater extends files by creating a replacement and
deleting the old file. New strategy research must preserve every data snapshot
used by an experiment, support interrupted ingestion, and prove the exact files
behind a snapshot.

## Decision

Add `ImmutableResearchCache` as an isolated cache boundary. Partitions are stored
under `providers/<provider>/partitions/`, with deterministic normalized JSONL,
per-chunk checkpoints, schema validation, duplicate-key conflict detection, and
SHA-256 file manifests. Snapshots aggregate partition manifests and are published
create-only. Existing ORB cache behavior remains unchanged.

## Consequences

Interrupted jobs can resume from a deterministic checkpoint, and prior snapshots
cannot be silently overwritten or deleted. Storage grows with each snapshot and
garbage collection is intentionally out of scope; operators must retain manifests
with the experiments that reference them.

## Alternatives considered

- Replacing the existing cache file was rejected because it destroys provenance.
- A mutable database was deferred; create-only filesystem artifacts are sufficient
  for the current local research workflow and require no new dependency.
