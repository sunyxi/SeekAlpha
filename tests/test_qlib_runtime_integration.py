"""Optional integration fixture against an installed pyqlib runtime."""

from __future__ import annotations

import importlib.util
import unittest
from datetime import date

import numpy as np

from orb.features.panel_builder import DailyPanel
from orb.qlib_adapter import require_qlib, to_qlib_frame


@unittest.skipUnless(importlib.util.find_spec("qlib"), "pyqlib optional extra is not installed")
class TestInstalledQlibRuntime(unittest.TestCase):
    def test_frame_is_selectable_by_qlib_datetime_utility(self):
        from qlib.data.dataset.utils import fetch_df_by_index

        panel = DailyPanel(
            symbols=("AAA",),
            dates=(date(2021, 1, 4), date(2021, 1, 5)),
            panel=np.array(
                [[[10, 11, 9, 10.5, 1000], [11, 12, 10, 11.5, 1200]]],
                dtype=float,
            ),
        )
        frame = to_qlib_frame(panel)

        qlib = require_qlib()
        selected = fetch_df_by_index(
            frame,
            slice("2021-01-05", "2021-01-05"),
            level="datetime",
        )

        self.assertTrue(getattr(qlib, "__version__", None))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["$close"], 11.5)
