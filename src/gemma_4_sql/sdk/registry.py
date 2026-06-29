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
    ----
        name: The name of the backend to load (e.g., 'jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        The loaded BackendProtocol implementation.

    Raises:
    ------
        ValueError: If the specified backend name is not found in the entry points.

    """
    if not ENTRY_POINTS:
        eps = importlib.metadata.entry_points(group="gemma_4_sql.backends")
        for ep in eps:
            ENTRY_POINTS[ep.name] = ep
    if name not in ENTRY_POINTS:
        msg = f"Unknown backend: {name}"
        raise ValueError(msg)
    return cast("BackendProtocol", ENTRY_POINTS[name].load())
