# Immutable Research Cache

## CLI Usage

```bash
python3 scripts/validate_snapshot.py \
  --cache-root research-cache --snapshot-id daily-panel-2026-06-30
```

## Operations

Ingest through `ImmutableResearchCache.ingest_partition` with a provider name,
request, time range, symbol list, schema, chunk count, and chunk loader. A
successful chunk updates a deterministic `.partial` checkpoint. Retryable loader
failures are retried; an interruption leaves the checkpoint for a later run.
Publish a snapshot only after all partitions are complete. Store the snapshot ID
and manifest hash in the experiment report.

## Limitations

This issue provides local filesystem storage, not distributed locking or cloud
retention. It does not download provider data, validate provider licensing, or
merge incompatible schemas. The legacy ORB cache updater remains a separate
compatibility path and must not be used to create new strategy snapshots.

## Rollback

Do not delete or overwrite a partition referenced by an experiment. To reject a
bad ingestion, mark its snapshot invalid and create a new partition/snapshot ID
after correction. Preserve the partial checkpoint for diagnosis unless it
contains a secret, in which case follow the repository secret-rotation process.
