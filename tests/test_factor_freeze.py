"""Tests for M3-4: factor development period boundary and list freeze.

Verifies:
  1. factor_list.json exists and contains the required fields.
  2. factor_list.json SHA-256 matches the frozen value recorded in CI.
  3. The dev_cutoff date is correct and parseable.
  4. The factor list contains exactly the expected 6 survivors.
  5. run_ic_eval._DEV_CUTOFF matches the dev_cutoff in factor_list.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from datetime import date
from pathlib import Path

# The frozen SHA-256 of src/orb/features/factor_list.json.
# Any modification to that file must update this constant (and justify the change).
_FROZEN_HASH = "41788b5b97629965a52799936f691e1bde391801d6df995712cd301df59d80ec"

_FACTOR_LIST_PATH = Path(__file__).resolve().parent.parent / "src" / "orb" / "features" / "factor_list.json"

_EXPECTED_FACTORS = frozenset({
    "alpha003", "alpha024", "alpha032", "alpha035", "alpha042", "alpha050",
})


class TestFactorListIntegrity(unittest.TestCase):

    def setUp(self):
        self.assertTrue(_FACTOR_LIST_PATH.exists(),
                        f"factor_list.json not found at {_FACTOR_LIST_PATH}")
        with open(_FACTOR_LIST_PATH) as f:
            self.data = json.load(f)

    def test_sha256_matches_frozen_hash(self):
        """Detect any unauthorised modification to factor_list.json."""
        with open(_FACTOR_LIST_PATH, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual, _FROZEN_HASH,
            f"factor_list.json has been modified!\n"
            f"  frozen : {_FROZEN_HASH}\n"
            f"  actual : {actual}\n"
            "Update _FROZEN_HASH in this test only after explicit justification.",
        )

    def test_dev_cutoff_field_present(self):
        self.assertIn("dev_cutoff", self.data,
                      "factor_list.json must contain a dev_cutoff field")

    def test_dev_cutoff_is_valid_date(self):
        cutoff = self.data["dev_cutoff"]
        try:
            d = date.fromisoformat(cutoff)
        except ValueError:
            self.fail(f"dev_cutoff={cutoff!r} is not a valid ISO date")
        self.assertEqual(d, date(2026, 6, 30),
                         "dev_cutoff must be 2026-06-30 (factor development period boundary)")

    def test_schema_version_is_1(self):
        self.assertEqual(self.data.get("schema_version"), 1)

    def test_fdr_q_is_0_05(self):
        self.assertAlmostEqual(self.data.get("fdr_q", -1), 0.05)

    def test_horizons_are_1_5_20(self):
        self.assertEqual(self.data.get("horizons_days"), [1, 5, 20])

    def test_exactly_6_factors(self):
        factors = self.data.get("factors", [])
        self.assertEqual(len(factors), 6,
                         f"expected 6 surviving factors, got {len(factors)}: {factors}")

    def test_factors_match_expected_set(self):
        factors = frozenset(self.data.get("factors", []))
        self.assertEqual(factors, _EXPECTED_FACTORS,
                         f"factor set changed!\n  expected: {sorted(_EXPECTED_FACTORS)}\n  got: {sorted(factors)}")

    def test_factors_are_sorted(self):
        factors = self.data.get("factors", [])
        self.assertEqual(factors, sorted(factors),
                         "factor list must be in sorted order")


class TestDevCutoffEnforcement(unittest.TestCase):
    """Verify run_ic_eval.py enforces the development cutoff."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    def test_dev_cutoff_constant_matches_factor_list(self):
        """_DEV_CUTOFF in run_ic_eval must match factor_list.json dev_cutoff."""
        import run_ic_eval
        with open(_FACTOR_LIST_PATH) as f:
            data = json.load(f)
        expected = date.fromisoformat(data["dev_cutoff"])
        self.assertEqual(run_ic_eval._DEV_CUTOFF, expected,
                         "_DEV_CUTOFF in run_ic_eval.py is out of sync with factor_list.json")

    def test_frozen_hash_constant_is_sha256_length(self):
        self.assertEqual(len(_FROZEN_HASH), 64,
                         "frozen hash must be a full 64-char SHA-256 hex digest")


if __name__ == "__main__":
    unittest.main()
