#!/usr/bin/env python3
"""Validate the report manifest and truthfully classify raw-report coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .validate_report import validate_file as validate_wf_report
except ImportError:
    from validate_report import validate_file as validate_wf_report


_REQUIRED_ENTRY_FIELDS = frozenset(
    {"filename", "kind", "generated_at", "sha256"}
)
_HEX = frozenset(string.hexdigits)


@dataclass(frozen=True)
class ManifestValidationResult:
    manifest_status: str
    raw_reports_status: str
    manifest_entry_count: int
    present_report_count: int
    missing_report_count: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: Path) -> ManifestValidationResult:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ManifestValidationResult(
            manifest_status="failed",
            raw_reports_status="not-run",
            manifest_entry_count=0,
            present_report_count=0,
            missing_report_count=0,
            errors=(f"could not read manifest: {exc}",),
        )

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        errors.append("manifest schema_version must equal 1")
    reports = payload.get("reports") if isinstance(payload, dict) else None
    if not isinstance(reports, list) or not reports:
        errors.append("manifest reports must not be empty")
        reports = []

    present = 0
    missing = 0
    seen: set[str] = set()
    report_errors = False
    for index, entry in enumerate(reports):
        prefix = f"reports[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            report_errors = True
            continue
        missing_fields = _REQUIRED_ENTRY_FIELDS - set(entry)
        if missing_fields:
            errors.append(f"{prefix} missing fields: {sorted(missing_fields)}")
            report_errors = True
            continue
        filename = entry["filename"]
        expected_hash = entry["sha256"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            errors.append(f"{prefix}.filename must be a plain file name")
            report_errors = True
            continue
        if filename in seen:
            errors.append(f"duplicate report filename: {filename}")
            report_errors = True
            continue
        seen.add(filename)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in _HEX for character in expected_hash)
        ):
            errors.append(f"{prefix}.sha256 must be 64 hexadecimal characters")
            report_errors = True
            continue

        report_path = path.parent / filename
        if not report_path.exists():
            missing += 1
            continue
        present += 1
        if _sha256(report_path) != expected_hash.lower():
            errors.append(f"SHA-256 mismatch: {filename}")
            report_errors = True
            continue
        if entry["kind"] == "wf_select":
            schema_errors = validate_wf_report(report_path)
            if schema_errors:
                errors.extend(
                    f"{filename}: {schema_error}"
                    for schema_error in schema_errors
                )
                report_errors = True

    manifest_failed = bool(errors)
    if report_errors or (present > 0 and missing > 0):
        raw_status = "failed"
        if present > 0 and missing > 0:
            errors.append("raw report set is only partially present")
    elif present == 0:
        raw_status = "not-run"
    else:
        raw_status = "passed"
    return ManifestValidationResult(
        manifest_status="failed" if manifest_failed else "passed",
        raw_reports_status=raw_status,
        manifest_entry_count=len(reports),
        present_report_count=present,
        missing_report_count=missing,
        errors=tuple(errors),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a report manifest and available raw reports."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    result = validate_manifest(args.manifest)
    print(json.dumps(result.to_dict(), indent=2))
    if result.manifest_status == "failed" or result.raw_reports_status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
