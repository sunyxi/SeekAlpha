"""Frozen research protocol and single-use retention access controls."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


T = TypeVar("T")
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> "DateWindow":
        try:
            start = date.fromisoformat(value["start"])
            end = date.fromisoformat(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name} date window") from exc
        if start > end:
            raise ValueError(f"{name} window starts after it ends")
        return cls(start, end)


@dataclass(frozen=True)
class ProtocolWindows:
    development: DateWindow
    outer_test: DateWindow
    retention: DateWindow


@dataclass(frozen=True)
class ExperimentBudget:
    max_experiments_per_strategy_family: int
    max_total_experiments: int
    max_parameter_trials_per_experiment: int
    max_model_trials_per_experiment: int


@dataclass(frozen=True)
class ValidationControls:
    train_days: int
    validation_days: int
    outer_test_days: int
    step_days: int
    purge_days: int
    embargo_days: int
    random_seed_required: bool


@dataclass(frozen=True)
class ResearchProtocol:
    schema_version: int
    protocol_id: str
    declared_at: date
    retention_available_after: date
    windows: ProtocolWindows
    budget: ExperimentBudget
    validation: ValidationControls
    costs: dict[str, float]
    random_seeds: dict[str, int]
    gates: dict[str, float | int]
    invalidation_rules: tuple[str, ...]
    protocol_hash: str

    @classmethod
    def load(cls, path: Path) -> "ResearchProtocol":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResearchProtocol":
        required = {
            "schema_version", "protocol_id", "declared_at",
            "retention_available_after", "date_windows", "experiment_budget",
            "validation", "costs", "random_seeds", "gates", "invalidation_rules",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"protocol missing fields: {', '.join(missing)}")
        if raw["schema_version"] != 1:
            raise ValueError("unsupported research protocol schema")
        if not isinstance(raw["protocol_id"], str) or not raw["protocol_id"]:
            raise ValueError("protocol_id must be a non-empty string")

        windows_raw = raw["date_windows"]
        if set(windows_raw) != {"development", "outer_test", "retention"}:
            raise ValueError("date_windows must contain development, outer_test, retention")
        windows = ProtocolWindows(
            *(DateWindow.from_mapping(windows_raw[name], name)
              for name in ("development", "outer_test", "retention"))
        )
        if windows.development.end >= windows.outer_test.start:
            raise ValueError("development and outer_test windows overlap")
        if windows.outer_test.end >= windows.retention.start:
            raise ValueError("outer_test and retention windows overlap")

        budget = _build_budget(raw["experiment_budget"])
        validation = _build_validation(raw["validation"])
        costs = _build_costs(raw["costs"])
        seeds = _build_seeds(raw["random_seeds"])
        gates = _build_gates(raw["gates"])
        invalidations = raw["invalidation_rules"]
        if not isinstance(invalidations, list) or not invalidations or not all(
            isinstance(rule, str) and rule for rule in invalidations
        ):
            raise ValueError("invalidation_rules must be a non-empty list of strings")
        declared_at = _parse_date(raw["declared_at"], "declared_at")
        available_after = _parse_date(
            raw["retention_available_after"], "retention_available_after"
        )
        if available_after <= windows.retention.end:
            raise ValueError("retention_available_after must follow retention window")

        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            schema_version=1,
            protocol_id=raw["protocol_id"],
            declared_at=declared_at,
            retention_available_after=available_after,
            windows=windows,
            budget=budget,
            validation=validation,
            costs=costs,
            random_seeds=seeds,
            gates=gates,
            invalidation_rules=tuple(invalidations),
            protocol_hash=digest,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "declared_at": self.declared_at.isoformat(),
            "retention_available_after": self.retention_available_after.isoformat(),
            "date_windows": {
                name: {"start": window.start.isoformat(), "end": window.end.isoformat()}
                for name, window in (
                    ("development", self.windows.development),
                    ("outer_test", self.windows.outer_test),
                    ("retention", self.windows.retention),
                )
            },
            "experiment_budget": self.budget.__dict__,
            "validation": self.validation.__dict__,
            "costs": self.costs,
            "random_seeds": self.random_seeds,
            "gates": self.gates,
            "invalidation_rules": list(self.invalidation_rules),
        }


def _parse_date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _positive_ints(raw: Mapping[str, Any], names: tuple[str, ...], section: str) -> list[int]:
    values = []
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{section}.{name} must be a positive integer")
        values.append(value)
    return values


def _build_budget(raw: Mapping[str, Any]) -> ExperimentBudget:
    values = _positive_ints(raw, (
        "max_experiments_per_strategy_family", "max_total_experiments",
        "max_parameter_trials_per_experiment", "max_model_trials_per_experiment",
    ), "experiment_budget")
    if values[0] > values[1]:
        raise ValueError("per-family experiment budget exceeds total budget")
    return ExperimentBudget(*values)


def _build_validation(raw: Mapping[str, Any]) -> ValidationControls:
    values = _positive_ints(raw, ("train_days", "validation_days", "outer_test_days", "step_days"), "validation")
    for name in ("purge_days", "embargo_days"):
        if isinstance(raw.get(name), bool) or not isinstance(raw.get(name), int) or raw[name] < 0:
            raise ValueError(f"validation.{name} must be a non-negative integer")
    if raw.get("random_seed_required") is not True:
        raise ValueError("validation.random_seed_required must be true")
    return ValidationControls(*values, raw["purge_days"], raw["embargo_days"], True)


def _build_costs(raw: Mapping[str, Any]) -> dict[str, float]:
    names = ("zero_bps_per_side", "baseline_bps_per_side", "double_bps_per_side")
    if set(raw) != set(names):
        raise ValueError("costs must declare zero, baseline, and double scenarios")
    values = {name: raw[name] for name in names}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in values.values()):
        raise ValueError("cost scenarios must be non-negative numbers")
    if values["zero_bps_per_side"] != 0 or values["double_bps_per_side"] <= values["baseline_bps_per_side"]:
        raise ValueError("cost scenarios must be ordered zero < baseline < double")
    return {name: float(value) for name, value in values.items()}


def _build_seeds(raw: Mapping[str, Any]) -> dict[str, int]:
    if not raw or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw.values()):
        raise ValueError("random_seeds must contain non-negative integers")
    return dict(raw)


def _build_gates(raw: Mapping[str, Any]) -> dict[str, float | int]:
    if not raw or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw.values()):
        raise ValueError("gates must contain numeric values")
    return dict(raw)


class RetentionAlreadyConsumedError(RuntimeError):
    """Raised when an experiment tries to access retention a second time."""


class RetentionNotAvailableError(RuntimeError):
    """Raised when retention is accessed before the protocol release date."""


class RetentionLedger:
    """Filesystem-backed, atomic, read-once retention access ledger."""

    def __init__(
        self,
        directory: str | Path,
        protocol: ResearchProtocol,
        *,
        as_of: date | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        self.as_of = as_of or date.today()

    def read_once(
        self,
        experiment_id: str,
        loader: Callable[[DateWindow], T],
        *,
        reader: str = "unknown",
    ) -> T:
        if self.as_of < self.protocol.retention_available_after:
            raise RetentionNotAvailableError(
                "retention is unavailable before "
                f"{self.protocol.retention_available_after.isoformat()}"
            )
        self._reserve(experiment_id, reader)
        return loader(self.protocol.windows.retention)

    def _reserve(self, experiment_id: str, reader: str) -> Path:
        if not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError("experiment_id contains unsupported characters")
        marker = self.directory / f"{self.protocol.protocol_id}--{experiment_id}.json"
        audit = {
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "experiment_id": experiment_id,
            "reader": reader,
            "retention_window": {
                "start": self.protocol.windows.retention.start.isoformat(),
                "end": self.protocol.windows.retention.end.isoformat(),
            },
            "consumed_at": datetime.now(timezone.utc).isoformat(),
            "status": "consumed",
        }
        fd, temp_name = tempfile.mkstemp(prefix=".retention-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(audit, stream, sort_keys=True, indent=2)
                stream.write("\n")
            try:
                os.link(temp_name, marker)
            except FileExistsError as exc:
                raise RetentionAlreadyConsumedError(
                    f"retention already consumed for experiment_id={experiment_id}"
                ) from exc
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        return marker
