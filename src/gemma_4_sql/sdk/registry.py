"""
Backend registry.
"""

from __future__ import annotations

import importlib.metadata
import sys
from typing import cast

from .protocols import BackendProtocol

_ENTRY_POINTS: dict[str, importlib.metadata.EntryPoint] | None = None


def get_backend(name: str) -> BackendProtocol:
    """
    Get backend by name.
    """
    global _ENTRY_POINTS
    if _ENTRY_POINTS is None:
        _ENTRY_POINTS = {}
        if sys.version_info >= (3, 10):  # noqa: UP036
            eps = importlib.metadata.entry_points(group="gemma_4_sql.backends")
        else:
            eps = importlib.metadata.entry_points().get("gemma_4_sql.backends", [])

        for ep in eps:
            _ENTRY_POINTS[ep.name] = ep

    if name not in _ENTRY_POINTS:
        raise ValueError(f"Unknown backend: {name}")

    return cast(BackendProtocol, _ENTRY_POINTS[name].load())
