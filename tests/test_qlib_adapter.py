"""Contract tests for the optional Qlib research adapter."""

from __future__ import annotations

import ast
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from orb.features.panel_builder import DailyPanel
from orb.qlib_adapter import QlibUnavailableError, require_qlib, to_qlib_frame


class TestToQlibFrame(unittest.TestCase):
    def setUp(self):
        self.panel = DailyPanel(
            symbols=("AAA", "BBB"),
            dates=(date(2021, 1, 4), date(2021, 1, 5)),
            panel=np.array(
                [
                    [
                        [10.0, 11.0, 9.0, 10.5, 1000.0],
                        [10.5, 10.5, 10.5, 10.5, 0.0],
                    ],
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        [20.0, 21.0, 19.0, 20.5, 2000.0],
                    ],
                ],
                dtype=float,
            ),
        )

    def test_frame_uses_qlib_index_and_field_names(self):
        frame = to_qlib_frame(self.panel)

        self.assertEqual(frame.index.names, ["datetime", "instrument"])
        self.assertEqual(
            list(frame.columns),
            ["$open", "$high", "$low", "$close", "$volume"],
        )
        self.assertEqual(
            list(frame.index),
            [
                (datetime(2021, 1, 4), "AAA"),
                (datetime(2021, 1, 4), "BBB"),
                (datetime(2021, 1, 5), "AAA"),
                (datetime(2021, 1, 5), "BBB"),
            ],
        )

    def test_frame_preserves_values_and_missing_observations(self):
        frame = to_qlib_frame(self.panel)

        self.assertEqual(frame.loc[(datetime(2021, 1, 4), "AAA"), "$close"], 10.5)
        self.assertEqual(frame.loc[(datetime(2021, 1, 5), "BBB"), "$volume"], 2000.0)
        self.assertTrue(
            frame.loc[(datetime(2021, 1, 4), "BBB")].isna().all(),
            "pre-listing NaN values must not be forward-filled by the adapter",
        )

    def test_rejects_panel_shape_that_disagrees_with_metadata(self):
        malformed = DailyPanel(
            symbols=("AAA", "BBB"),
            dates=(date(2021, 1, 4),),
            panel=np.zeros((1, 1, 5), dtype=float),
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            to_qlib_frame(malformed)


class TestQlibRuntimeGuard(unittest.TestCase):
    def test_missing_qlib_has_actionable_optional_dependency_error(self):
        with patch(
            "orb.qlib_adapter.runtime.import_module",
            side_effect=ModuleNotFoundError("No module named 'qlib'"),
        ):
            with self.assertRaisesRegex(
                QlibUnavailableError,
                r"pip install -e .\[qlib\]",
            ):
                require_qlib()


class TestQlibIsolation(unittest.TestCase):
    def test_stdlib_only_paths_do_not_import_qlib_adapter_or_pandas(self):
        root = Path(__file__).resolve().parent.parent
        forbidden = {"pandas", "qlib"}
        paths = [root / "src/orb/core/orb_core.py", root / "scripts/wf_select.py"]

        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imported_roots = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported_roots.update(
                (node.module or "").split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
            with self.subTest(path=path):
                self.assertTrue(forbidden.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
