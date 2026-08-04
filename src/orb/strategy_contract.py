"""Versioned strategy hypotheses and decision reports.

This module deliberately contains no strategy logic, data access, or model
search. It validates pre-declared contracts and renders an audit summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SCHEMA_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")
_REPORT_STATES = frozenset({"Candidate", "No-Go", "Exploratory", "Invalid"})
_GATE_STATUSES = frozenset({"passed", "failed", "not-run", "skipped"})
_SPEC_REQUIRED = frozenset({
    "schema_version", "spec_id", "strategy_family", "hypothesis", "universe",
    "features", "label", "holding_period", "parameters", "costs", "protocol",
    "data", "experiment_budget", "invalidation",
})
_REPORT_REQUIRED = frozenset({
    "schema_version", "report_id", "spec_id", "strategy_family", "spec_hash",
    "protocol_hash", "data_manifest_hash", "code_commit", "experiment_budget",
    "decision", "evidence",
})


def _canonical_hash(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class StrategySpec:
    """A pre-registered strategy hypothesis with its provenance boundary."""

    data: dict[str, Any]
    spec_hash: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StrategySpec":
        missing = sorted(_SPEC_REQUIRED - raw.keys())
        if missing:
            raise ValueError(f"strategy spec missing fields: {', '.join(missing)}")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported strategy spec schema")
        _require_nonempty_string(raw.get("spec_id"), "spec_id")
        _require_nonempty_string(raw.get("strategy_family"), "strategy_family")
        _require_nonempty_string(raw.get("hypothesis"), "hypothesis")

        universe = raw["universe"]
        if not isinstance(universe, dict) or universe.get("point_in_time") is not True:
            raise ValueError("universe must declare point_in_time=true")
        features = raw["features"]
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a non-empty list")
        for feature in features:
            if not isinstance(feature, dict):
                raise ValueError("each feature declaration must be an object")
            for field in ("name", "source", "as_of", "transform"):
                _require_nonempty_string(feature.get(field), f"features[].{field}")

        label = raw["label"]
        if not isinstance(label, dict):
            raise ValueError("label must be an object")
        _require_nonempty_string(label.get("name"), "label.name")
        _require_nonempty_string(label.get("definition"), "label.definition")
        if not isinstance(label.get("horizon_days"), int) or label["horizon_days"] <= 0:
            raise ValueError("label.horizon_days must be a positive integer")
        if label.get("cost_included") is not True:
            raise ValueError("label.cost_included must be true")

        holding = raw["holding_period"]
        if not isinstance(holding, dict):
            raise ValueError("holding_period must be an object")
        for field in ("min_days", "max_days"):
            if not isinstance(holding.get(field), int) or holding[field] <= 0:
                raise ValueError(f"holding_period.{field} must be a positive integer")
        if holding["min_days"] > holding["max_days"]:
            raise ValueError("holding_period min_days exceeds max_days")
        if not isinstance(holding.get("overnight_allowed"), bool):
            raise ValueError("holding_period.overnight_allowed must be boolean")

        parameters = raw["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        _require_hash(parameters.get("search_space_hash"), "parameters.search_space_hash")
        if not isinstance(parameters.get("declared"), dict) or not parameters["declared"]:
            raise ValueError("parameters.declared must be a non-empty object")

        costs = raw["costs"]
        if not isinstance(costs, dict) or not isinstance(costs.get("scenarios"), dict):
            raise ValueError("costs.scenarios must be an object")
        scenarios = costs["scenarios"]
        if set(scenarios) != {"zero_bps_per_side", "baseline_bps_per_side", "double_bps_per_side"}:
            raise ValueError("costs.scenarios must declare zero, baseline, and double")
        if not all(isinstance(value, (int, float)) and value >= 0 for value in scenarios.values()):
            raise ValueError("cost scenarios must be non-negative numbers")
        _require_nonempty_string(costs.get("commission_model"), "costs.commission_model")

        protocol = raw["protocol"]
        if not isinstance(protocol, dict):
            raise ValueError("protocol must be an object")
        _require_nonempty_string(protocol.get("protocol_id"), "protocol.protocol_id")
        protocol_hash = _require_hash(protocol.get("protocol_hash"), "protocol.protocol_hash")

        data = raw["data"]
        if not isinstance(data, dict):
            raise ValueError("data must be an object")
        data_hash = _require_hash(data.get("manifest_hash"), "data.manifest_hash")
        _require_nonempty_string(data.get("snapshot_id"), "data.snapshot_id")

        budget = raw["experiment_budget"]
        if not isinstance(budget, dict):
            raise ValueError("experiment_budget must be an object")
        for field in ("max_parameter_trials", "max_model_trials"):
            if isinstance(budget.get(field), bool) or not isinstance(budget.get(field), int) or budget[field] <= 0:
                raise ValueError(f"experiment_budget.{field} must be a positive integer")

        invalidation = raw["invalidation"]
        if not isinstance(invalidation, list) or not invalidation or not all(
            isinstance(value, str) and value for value in invalidation
        ):
            raise ValueError("invalidation must be a non-empty list of strings")

        data_copy = json.loads(json.dumps(raw, sort_keys=True))
        return cls(data=data_copy, spec_hash=_canonical_hash(data_copy))

    @classmethod
    def load(cls, path: Path) -> "StrategySpec":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @property
    def spec_id(self) -> str:
        return self.data["spec_id"]

    @property
    def strategy_family(self) -> str:
        return self.data["strategy_family"]

    @property
    def universe(self) -> dict[str, Any]:
        return dict(self.data["universe"])

    @property
    def protocol_hash(self) -> str:
        return self.data["protocol"]["protocol_hash"]

    @property
    def data_manifest_hash(self) -> str:
        return self.data["data"]["manifest_hash"]

    @property
    def experiment_budget(self) -> dict[str, int]:
        return dict(self.data["experiment_budget"])

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data, sort_keys=True))


@dataclass(frozen=True)
class DecisionReport:
    data: dict[str, Any]

    @classmethod
    def validate(cls, raw: Mapping[str, Any], spec: StrategySpec) -> "DecisionReport":
        missing = sorted(_REPORT_REQUIRED - raw.keys())
        if missing:
            raise ValueError(f"decision report missing fields: {', '.join(missing)}")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported decision report schema")
        if raw.get("spec_id") != spec.spec_id:
            raise ValueError("decision report spec_id does not match strategy spec")
        if raw.get("strategy_family") != spec.strategy_family:
            raise ValueError("decision report strategy_family does not match strategy spec")
        if raw.get("spec_hash") != spec.spec_hash:
            raise ValueError("decision report spec_hash does not match strategy spec")
        if raw.get("protocol_hash") != spec.protocol_hash:
            raise ValueError("decision report protocol_hash does not match strategy spec")
        if raw.get("data_manifest_hash") != spec.data_manifest_hash:
            raise ValueError("decision report data_manifest_hash does not match strategy spec")
        _require_nonempty_string(raw.get("report_id"), "report_id")
        _require_nonempty_string(raw.get("code_commit"), "code_commit")
        if raw.get("experiment_budget") != spec.experiment_budget:
            raise ValueError("decision report experiment_budget differs from predeclared budget")

        decision = raw["decision"]
        if not isinstance(decision, dict) or decision.get("state") not in _REPORT_STATES:
            raise ValueError(f"decision.state must be one of {sorted(_REPORT_STATES)}")
        if not isinstance(decision.get("reasons"), list) or not decision["reasons"]:
            raise ValueError("decision.reasons must be a non-empty list")
        gates = decision.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError("decision.gates must be a non-empty list")
        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("each decision gate must be an object")
            _require_nonempty_string(gate.get("name"), "decision.gates[].name")
            if gate.get("status") not in _GATE_STATUSES:
                raise ValueError(f"decision.gates[].status must be one of {sorted(_GATE_STATUSES)}")
            for field in ("threshold", "observed"):
                if not isinstance(gate.get(field), (int, float)):
                    raise ValueError(f"decision.gates[].{field} must be numeric")
        if not isinstance(raw["evidence"], dict) or not raw["evidence"]:
            raise ValueError("evidence must be a non-empty object")
        return cls(data=json.loads(json.dumps(raw, sort_keys=True)))

    @property
    def decision_state(self) -> str:
        return self.data["decision"]["state"]

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data, sort_keys=True))


def create_only_write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically publish a report and refuse to overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=True, sort_keys=True, indent=2)
            stream.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_SUMMARY_TEMPLATE = Path(__file__).resolve().parents[2] / "docs" / "strategy-decision-summary.template.md"


def render_decision_summary(spec: StrategySpec, report: DecisionReport) -> str:
    """Render the tracked summary template from validated source objects."""
    template = _SUMMARY_TEMPLATE.read_text(encoding="utf-8")
    decision = report.data["decision"]
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"])
    gates = "\n".join(
        f"| {gate['name']} | {gate['threshold']} | {gate['observed']} | {gate['status']} |"
        for gate in decision["gates"]
    )
    replacements = {
        "{{REPORT_ID}}": report.data["report_id"],
        "{{SPEC_ID}}": spec.spec_id,
        "{{STRATEGY_FAMILY}}": spec.strategy_family,
        "{{DECISION_STATE}}": report.decision_state,
        "{{SPEC_HASH}}": spec.spec_hash,
        "{{PROTOCOL_HASH}}": spec.protocol_hash,
        "{{DATA_MANIFEST_HASH}}": spec.data_manifest_hash,
        "{{CODE_COMMIT}}": report.data["code_commit"],
        "{{REASONS}}": reasons,
        "{{GATES}}": gates,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template
