"""Backend registry for gemma-4-sql."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .protocols import BackendProtocol
ENTRY_POINTS: dict[str, importlib.metadata.EntryPoint] = {}


def get_backend(name: str) -> BackendProtocol:
    """Get backend by name.

    Args:
        name: The string representing the name.

    Returns:
        The execution result.
    """
    if not ENTRY_POINTS:
        eps_all = importlib.metadata.entry_points()
        eps = eps_all.get("gemma_4_sql.backends", []) if isinstance(eps_all, dict) else eps_all.select(group="gemma_4_sql.backends")
        for ep in eps:
            ENTRY_POINTS[ep.name] = ep
    if name not in ENTRY_POINTS:
        msg = f"Unknown backend: {name}"
        raise ValueError(msg)
    return cast("BackendProtocol", ENTRY_POINTS[name].load())
