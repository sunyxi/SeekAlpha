from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.validate_manifest import validate_manifest


def _write_manifest(tmp_path: Path, reports: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-08-04T00:00:00+09:00",
                "reports": reports,
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(filename: str, sha256: str) -> dict[str, object]:
    return {
        "filename": filename,
        "kind": "compare_runs",
        "decision": "No-Go",
        "generated_at": "2026-08-04T00:00:00+09:00",
        "sha256": sha256,
    }


def test_missing_raw_reports_are_reported_as_not_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_entry("missing.json", "0" * 64)])

    result = validate_manifest(manifest)

    assert result.manifest_status == "passed"
    assert result.raw_reports_status == "not-run"
    assert result.present_report_count == 0
    assert result.missing_report_count == 1


def test_present_report_with_matching_hash_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"decision":"No-Go"}', encoding="utf-8")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    manifest = _write_manifest(tmp_path, [_entry(report.name, digest)])

    result = validate_manifest(manifest)

    assert result.manifest_status == "passed"
    assert result.raw_reports_status == "passed"
    assert result.present_report_count == 1
    assert result.missing_report_count == 0


def test_present_report_with_wrong_hash_fails(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"decision":"No-Go"}', encoding="utf-8")
    manifest = _write_manifest(tmp_path, [_entry(report.name, "0" * 64)])

    result = validate_manifest(manifest)

    assert result.manifest_status == "failed"
    assert result.raw_reports_status == "failed"
    assert any("SHA-256 mismatch" in error for error in result.errors)


def test_empty_manifest_fails_instead_of_vacuously_passing(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [])

    result = validate_manifest(manifest)

    assert result.manifest_status == "failed"
    assert result.raw_reports_status == "not-run"
    assert any("must not be empty" in error for error in result.errors)
