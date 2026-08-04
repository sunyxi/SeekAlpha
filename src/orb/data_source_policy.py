"""Licensed data-source policy and point-in-time universe primitives."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9./-]*$")
_PROVIDERS = {"databento", "alpaca"}
_REQUIRED_CAPABILITIES = {
    "historical_daily_ohlcv", "historical_minute_ohlcv", "corporate_actions",
    "delisted_instruments", "symbol_changes", "bid_ask",
    "point_in_time_listing_reference", "historical_sector_history",
}


@dataclass(frozen=True)
class SourceSelection:
    provider: str
    capabilities: dict[str, bool]
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class UniverseContract:
    market: str
    asset_types: tuple[str, ...]
    identity_key: str
    interval_end_semantics: str
    minimum_history_start: date
    minimum_daily_universe_size: int
    minimum_regime_years: int
    minute_subset_max_size: int


@dataclass(frozen=True)
class DataSourcePolicy:
    policy_id: str
    decision: str
    primary: SourceSelection
    fallback: SourceSelection
    universe: UniverseContract
    licensing: dict[str, Any]
    blockers: tuple[str, ...]
    prohibited_shortcuts: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "DataSourcePolicy":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DataSourcePolicy":
        required = {
            "schema_version", "policy_id", "decision", "primary", "fallback",
            "universe", "licensing", "blockers", "prohibited_shortcuts",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"data source policy missing fields: {', '.join(missing)}")
        if raw["schema_version"] != 1:
            raise ValueError("unsupported data source policy schema")
        if raw["decision"] != "selected_with_blockers":
            raise ValueError("policy must explicitly select sources with blockers")
        policy_id = _nonempty(raw["policy_id"], "policy_id")
        primary = _source(raw["primary"], "primary")
        fallback = _source(raw["fallback"], "fallback")
        if primary.provider == fallback.provider:
            raise ValueError("primary and fallback providers must differ")
        universe = _universe(raw["universe"])
        licensing = raw["licensing"]
        if not isinstance(licensing, dict) or licensing.get("credentials_from_environment_only") is not True:
            raise ValueError("licensing must require environment-only credentials")
        if licensing.get("raw_data_commit_forbidden") is not True:
            raise ValueError("raw_data_commit_forbidden must be true")
        blockers = _strings(raw["blockers"], "blockers")
        shortcuts = _strings(raw["prohibited_shortcuts"], "prohibited_shortcuts")
        if not any(not primary.capabilities[name] for name in _REQUIRED_CAPABILITIES):
            raise ValueError("policy must expose any unresolved provider capability as a blocker")
        return cls(policy_id, raw["decision"], primary, fallback, universe, dict(licensing), blockers, shortcuts)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return tuple(value)


def _source(raw: Mapping[str, Any], name: str) -> SourceSelection:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    provider = _nonempty(raw.get("provider"), f"{name}.provider").lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"unsupported {name} provider: {provider}")
    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != _REQUIRED_CAPABILITIES:
        raise ValueError(f"{name}.capabilities must declare the complete capability matrix")
    if not all(isinstance(value, bool) for value in capabilities.values()):
        raise ValueError(f"{name}.capabilities values must be boolean")
    urls = _strings(raw.get("source_urls"), f"{name}.source_urls")
    return SourceSelection(provider, dict(capabilities), urls)


def _universe(raw: Mapping[str, Any]) -> UniverseContract:
    if not isinstance(raw, dict):
        raise ValueError("universe must be an object")
    if raw.get("membership_basis") != "point_in_time_listing_intervals":
        raise ValueError("universe must use point-in-time listing intervals")
    if raw.get("symbol_is_alias_only") is not True:
        raise ValueError("symbol_is_alias_only must be true")
    if raw.get("interval_end_semantics") != "exclusive":
        raise ValueError("universe interval end semantics must be exclusive")
    start = date.fromisoformat(raw["minimum_history_start"])
    ints = ("minimum_daily_universe_size", "minimum_regime_years", "minute_subset_max_size")
    values = []
    for name in ints:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"universe.{name} must be a positive integer")
        values.append(value)
    return UniverseContract(
        market=_nonempty(raw.get("market"), "universe.market"),
        asset_types=tuple(_strings(raw.get("asset_types"), "universe.asset_types")),
        identity_key=_nonempty(raw.get("identity_key"), "universe.identity_key"),
        interval_end_semantics=raw["interval_end_semantics"],
        minimum_history_start=start,
        minimum_daily_universe_size=values[0],
        minimum_regime_years=values[1],
        minute_subset_max_size=values[2],
    )


@dataclass(frozen=True)
class _Listing:
    instrument_id: str
    symbol: str
    asset_type: str
    listed_from: date
    listed_to: date | None

    def contains(self, as_of: date) -> bool:
        return self.listed_from <= as_of and (self.listed_to is None or as_of < self.listed_to)


class PointInTimeUniverse:
    """Reconstruct membership from listing intervals, never current constituents."""

    def __init__(self, records: tuple[_Listing, ...]) -> None:
        self._records = records

    @classmethod
    def from_records(cls, records: list[Mapping[str, Any]]) -> "PointInTimeUniverse":
        parsed = []
        for raw in records:
            instrument = _nonempty(raw.get("instrument_id"), "instrument_id")
            symbol = _nonempty(raw.get("symbol"), "symbol")
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f"invalid symbol: {symbol!r}")
            asset_type = _nonempty(raw.get("asset_type"), "asset_type")
            try:
                listed_from = date.fromisoformat(raw["listed_from"])
                listed_to = None if raw.get("listed_to") is None else date.fromisoformat(raw["listed_to"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid listing interval") from exc
            if listed_to is not None and listed_from >= listed_to:
                raise ValueError(f"invalid listing interval for {instrument}")
            parsed.append(_Listing(instrument, symbol, asset_type, listed_from, listed_to))
        for instrument in {record.instrument_id for record in parsed}:
            intervals = sorted((record for record in parsed if record.instrument_id == instrument), key=lambda item: item.listed_from)
            for previous, current in zip(intervals, intervals[1:]):
                if previous.listed_to is None or current.listed_from < previous.listed_to:
                    raise ValueError(f"overlapping listing intervals for {instrument}")
        return cls(tuple(parsed))

    def members_at(self, as_of: date) -> tuple[str, ...]:
        members = {
            record.instrument_id
            for record in self._records
            if record.asset_type == "common_stock" and record.contains(as_of)
        }
        return tuple(sorted(members))

    def resolve_symbol(self, symbol: str, as_of: date) -> str:
        matches = [record.instrument_id for record in self._records if record.symbol == symbol and record.contains(as_of)]
        if len(matches) != 1:
            raise ValueError(f"symbol {symbol!r} is not uniquely resolvable at {as_of.isoformat()}")
        return matches[0]
