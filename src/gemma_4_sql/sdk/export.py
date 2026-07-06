"""SDK Export module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def export_model(model_name: str, export_path: str, backend: str = "jax") -> JSONDict:
    """Export a trained Text-to-SQL model.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.
        backend: The backend framework to use.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).export_model(model_name, export_path)
