import json
import tempfile
import unittest
from pathlib import Path

from orb.immutable_cache import ImmutableResearchCache


SCHEMA = ["instrument_id", "timestamp", "open", "high", "low", "close", "volume"]


def row(instrument, ts, close):
    return {"instrument_id": instrument, "timestamp": ts, "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "volume": 100}


class ImmutableCacheTests(unittest.TestCase):
    def metadata(self, partition="p1"):
        return {
            "partition_id": partition,
            "provider": "fixture",
            "source": "fixture-bars",
            "request": {"feed": "daily", "adjustment": "raw"},
            "time_range": {"start": "2024-01-01", "end": "2024-01-03"},
            "symbols": ["AAA", "BBB"],
            "schema": SCHEMA,
        }

    def test_interruption_leaves_checkpoint_and_resume_publishes_same_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ImmutableResearchCache(tmp)
            chunks = [[row("A", "2024-01-01T00:00:00Z", 10)], [row("A", "2024-01-02T00:00:00Z", 11)]]
            calls = []

            def interrupted(index):
                calls.append(index)
                if index == 1:
                    raise RuntimeError("interrupted fixture")
                return chunks[index]

            with self.assertRaisesRegex(RuntimeError, "interrupted fixture"):
                cache.ingest_partition(**self.metadata(), chunk_count=2, chunk_loader=interrupted)
            self.assertTrue(list(Path(tmp).rglob("*.partial")))

            def resumed(index):
                return chunks[index]

            manifest = cache.ingest_partition(**self.metadata(), chunk_count=2, chunk_loader=resumed)
            self.assertEqual(manifest["row_count"], 2)
            self.assertFalse(list(Path(tmp).rglob("*.partial")))

            snapshot = cache.publish_snapshot("snapshot-1", ["p1"])
            self.assertEqual(cache.resolve_snapshot("snapshot-1")["snapshot_id"], snapshot["snapshot_id"])

    def test_duplicate_rows_are_deduplicated_but_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ImmutableResearchCache(tmp)
            duplicate = row("A", "2024-01-01T00:00:00Z", 10)
            manifest = cache.ingest_partition(
                **self.metadata(), chunk_count=2,
                chunk_loader=lambda index: [duplicate] if index == 0 else [duplicate],
            )
            self.assertEqual(manifest["row_count"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            cache = ImmutableResearchCache(tmp)
            conflicting = row("A", "2024-01-01T00:00:00Z", 12)
            with self.assertRaises(ValueError):
                cache.ingest_partition(
                    **self.metadata(), chunk_count=2,
                    chunk_loader=lambda index: [duplicate] if index == 0 else [conflicting],
                )

    def test_schema_change_and_attempted_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ImmutableResearchCache(tmp)
            cache.ingest_partition(**self.metadata(), chunk_count=1, chunk_loader=lambda _index: [row("A", "2024-01-01T00:00:00Z", 10)])
            with self.assertRaises(FileExistsError):
                cache.ingest_partition(**self.metadata(), chunk_count=1, chunk_loader=lambda _index: [])
            changed = self.metadata("p2")
            changed["schema"] = SCHEMA + ["vwap"]
            with self.assertRaises(ValueError):
                cache.ingest_partition(**changed, chunk_count=1, chunk_loader=lambda _index: [row("A", "2024-01-01T00:00:00Z", 10)])

    def test_manifest_is_deterministic_and_snapshot_is_create_only(self):
        def build(root):
            cache = ImmutableResearchCache(root)
            cache.ingest_partition(**self.metadata(), chunk_count=1, chunk_loader=lambda _index: [row("A", "2024-01-01T00:00:00Z", 10)])
            return cache.publish_snapshot("snapshot-1", ["p1"])

        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            self.assertEqual(build(left), build(right))
            cache = ImmutableResearchCache(left)
            with self.assertRaises(FileExistsError):
                cache.publish_snapshot("snapshot-1", ["p1"])
            manifest_path = Path(left) / "snapshots" / "snapshot-1.json"
            self.assertEqual(json.loads(manifest_path.read_text())["snapshot_id"], "snapshot-1")


if __name__ == "__main__":
    unittest.main()
