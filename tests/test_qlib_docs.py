"""Documentation localization contract for the Qlib POC."""

from __future__ import annotations

import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_DOCS = (
    _ROOT / "docs/qlib-poc.en.md",
    _ROOT / "docs/qlib-poc.ja.md",
    _ROOT / "docs/qlib-poc.zh-CN.md",
)
_SECTIONS = ("CLI Usage", "Operations", "Limitations", "Rollback")


class TestQlibDocumentationLocalization(unittest.TestCase):
    def test_all_required_locales_exist(self):
        for path in _DOCS:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing localized document: {path}")

    def test_each_locale_keeps_governance_section_anchors(self):
        for path in _DOCS:
            if not path.exists():
                continue
            text = path.read_text()
            for section in _SECTIONS:
                with self.subTest(path=path.name, section=section):
                    self.assertIn(f"## {section}", text)
            self.assertIn("ADR-006", text)

    def test_adr_006_exists(self):
        self.assertTrue((_ROOT / "docs/adr/ADR-006-optional-qlib-adapter.md").is_file())


if __name__ == "__main__":
    unittest.main()
