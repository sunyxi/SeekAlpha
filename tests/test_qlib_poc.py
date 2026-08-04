"""End-to-end fixture tests for the read-only Qlib POC CLI."""

from __future__ import annotations

import csv
import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "qlib_poc.py"
sys.path.insert(0, str(_ROOT / "scripts"))

import qlib_poc
from orb.qlib_adapter import QlibUnavailableError


def _write_cache(path: Path) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        writer.writerow(["2021-01-04T14:30:00+00:00", 10, 11, 9, 10.5, 1000])


class TestQlibPocCli(unittest.TestCase):
    def test_verify_runtime_fails_before_reading_cache_or_writing_output(self):
        with patch.object(
            sys,
            "argv",
            [
                "qlib_poc.py",
                "--cache-dir",
                "unused",
                "--start",
                "2021-01-04",
                "--end",
                "2021-01-04",
                "--output",
                "unused.csv",
                "--verify-qlib",
            ],
        ), patch.object(
            qlib_poc,
            "require_qlib",
            side_effect=QlibUnavailableError("missing qlib"),
        ), patch.object(qlib_poc, "build_panel") as build_panel:
            with self.assertRaisesRegex(QlibUnavailableError, "missing qlib"):
                qlib_poc.main()

        build_panel.assert_not_called()

    def test_exports_existing_cache_without_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            source = cache / "AAA_2021-01-04_2021-01-04_1min.csv.gz"
            _write_cache(source)
            before = source.read_bytes()
            output = root / "derived" / "qlib.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--cache-dir",
                    str(cache),
                    "--start",
                    "2021-01-04",
                    "--end",
                    "2021-01-04",
                    "--output",
                    str(output),
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertEqual(source.read_bytes(), before, "source cache must remain unchanged")
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]),
                ["datetime", "instrument", "$open", "$high", "$low", "$close", "$volume"],
            )
            self.assertEqual(rows[0]["instrument"], "AAA")
            self.assertEqual(rows[0]["$close"], "10.5")

    def test_refuses_to_overwrite_derived_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            _write_cache(cache / "AAA_2021-01-04_2021-01-04_1min.csv.gz")
            output = root / "qlib.csv"
            output.write_text("existing\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--cache-dir",
                    str(cache),
                    "--start",
                    "2021-01-04",
                    "--end",
                    "2021-01-04",
                    "--output",
                    str(output),
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refuses overwrite", result.stderr)
            self.assertEqual(output.read_text(), "existing\n")


if __name__ == "__main__":
    unittest.main()
