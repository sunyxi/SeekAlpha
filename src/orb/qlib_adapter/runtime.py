"""Lazy import guard for the optional pyqlib runtime."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


class QlibUnavailableError(RuntimeError):
    """Raised when an explicitly requested Qlib operation lacks pyqlib."""


def require_qlib() -> ModuleType:
    """Import and return Qlib, or raise an actionable optional-dependency error."""
    try:
        return import_module("qlib")
    except ModuleNotFoundError as exc:
        raise QlibUnavailableError(
            "Qlib is unavailable; run: pip install -e .[qlib]"
        ) from exc
