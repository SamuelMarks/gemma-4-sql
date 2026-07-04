# Copyright 2024
"""SDK Export module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def export_model(model_name: str, export_path: str, backend: str = "jax") -> JSONDict:
    """Export a trained Text-to-SQL model.

    Args:
    ----
        model_name: The name or path of the model.
        export_path: The filesystem path to export the checkpoint.
        backend: The backend framework ('jax', 'keras', or 'maxtext').

    Returns:
    -------
        Export results dictionary.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).export_model(model_name, export_path)
