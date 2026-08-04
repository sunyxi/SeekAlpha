"""Provider-isolated, immutable research cache with deterministic manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class ImmutableResearchCache:
    """Append-by-new-snapshot cache; existing partitions are never replaced."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.providers = self.root / "providers"
        self.snapshots = self.root / "snapshots"

    def ingest_partition(
        self,
        *,
        partition_id: str,
        provider: str,
        source: str,
        request: Mapping[str, Any],
        time_range: Mapping[str, str],
        symbols: Iterable[str],
        schema: list[str],
        chunk_count: int,
        chunk_loader: Callable[[int], Iterable[Mapping[str, Any]]],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Ingest chunks into a checkpoint, then publish immutable artifacts."""
        if not partition_id or not provider or not source:
            raise ValueError("partition_id, provider, and source are required")
        if chunk_count <= 0 or max_retries <= 0:
            raise ValueError("chunk_count and max_retries must be positive")
        schema = list(schema)
        if len(schema) != len(set(schema)) or not {"instrument_id", "timestamp"}.issubset(schema):
            raise ValueError("schema must be unique and include instrument_id,timestamp")
        symbols = sorted(set(symbols))
        if not symbols:
            raise ValueError("symbols must be non-empty")
        partition_root = self.providers / provider / "partitions"
        partition_root.mkdir(parents=True, exist_ok=True)
        data_path = partition_root / f"{partition_id}.jsonl"
        manifest_path = partition_root / f"{partition_id}.manifest.json"
        partial_path = partition_root / f"{partition_id}.partial"
        if data_path.exists() or manifest_path.exists():
            raise FileExistsError(f"partition already exists: {partition_id}")

        rows = self._read_partial(partial_path, schema)
        for chunk_index in range(chunk_count):
            last_error: Exception | None = None
            for _attempt in range(max_retries):
                try:
                    chunk_rows = self._normalize_rows(chunk_loader(chunk_index), schema)
                    self._merge_rows(rows, chunk_rows)
                    self._write_partial(partial_path, rows)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error

        payload = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
        _atomic_create(data_path, payload)
        manifest = {
            "schema_version": 1,
            "kind": "partition",
            "partition_id": partition_id,
            "provider": provider,
            "source": source,
            "request": json.loads(json.dumps(request, sort_keys=True)),
            "time_range": dict(time_range),
            "symbols": symbols,
            "schema": schema,
            "files": [{"path": str(data_path.relative_to(self.root)), "sha256": _sha256(data_path)}],
            "row_count": len(rows),
        }
        _atomic_create(manifest_path, _canonical(manifest) + b"\n")
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass
        return manifest

    def publish_snapshot(self, snapshot_id: str, partition_ids: Iterable[str]) -> dict[str, Any]:
        if not snapshot_id:
            raise ValueError("snapshot_id is required")
        partition_ids = sorted(set(partition_ids))
        if not partition_ids:
            raise ValueError("snapshot must contain at least one partition")
        manifest_path = self.snapshots / f"{snapshot_id}.json"
        if manifest_path.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot_id}")
        partitions = []
        files = []
        for partition_id in partition_ids:
            matches = list(self.providers.glob(f"*/partitions/{partition_id}.manifest.json"))
            if not matches:
                raise FileNotFoundError(f"partition manifest missing: {partition_id}")
            if len(matches) > 1:
                raise ValueError(f"partition ID is ambiguous across providers: {partition_id}")
            path = matches[0]
            partition = json.loads(path.read_text(encoding="utf-8"))
            partitions.append(partition)
            files.extend(partition["files"])
        snapshot = {
            "schema_version": 1,
            "kind": "snapshot",
            "snapshot_id": snapshot_id,
            "partitions": partitions,
            "files": sorted(files, key=lambda item: item["path"]),
            "manifest_sha256": hashlib.sha256(_canonical({
                "schema_version": 1, "kind": "snapshot", "snapshot_id": snapshot_id,
                "partitions": partitions, "files": sorted(files, key=lambda item: item["path"]),
            })).hexdigest(),
        }
        _atomic_create(manifest_path, _canonical(snapshot) + b"\n")
        return snapshot

    def resolve_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        path = self.snapshots / f"{snapshot_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"snapshot not found: {snapshot_id}")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        expected = dict(snapshot)
        actual_hash = expected.pop("manifest_sha256")
        if hashlib.sha256(_canonical(expected)).hexdigest() != actual_hash:
            raise ValueError(f"snapshot manifest hash mismatch: {snapshot_id}")
        return snapshot

    @staticmethod
    def _normalize_rows(rows: Iterable[Mapping[str, Any]], schema: list[str]) -> list[dict[str, Any]]:
        normalized = []
        for row in rows:
            if set(row) != set(schema):
                raise ValueError("row schema differs from partition schema")
            normalized.append({key: row[key] for key in schema})
        return sorted(normalized, key=lambda row: (str(row["instrument_id"]), str(row["timestamp"])))

    @staticmethod
    def _merge_rows(target: dict[tuple[str, str], dict[str, Any]], rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            key = (str(row["instrument_id"]), str(row["timestamp"]))
            previous = target.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate row: {key}")
            target[key] = row

    @staticmethod
    def _read_partial(path: Path, schema: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if set(row) != set(schema):
                    raise ValueError("checkpoint schema differs from requested schema")
                ImmutableResearchCache._merge_rows(rows, [row])
        return rows

    @staticmethod
    def _write_partial(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
        ordered = sorted(rows.values(), key=lambda row: (str(row["instrument_id"]), str(row["timestamp"])))
        content = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in ordered)
        path.write_text(content, encoding="utf-8")
