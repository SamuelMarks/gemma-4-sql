"""SDK Export module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import ExportResultDict


def export_model(model_name: str, export_path: str, backend: str = "jax", **kwargs: Any) -> ExportResultDict:
    """Export a trained Text-to-SQL model.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.
        backend: The backend framework to use.
        **kwargs: Optional additional keyword arguments passed to the backend export.

    Returns:
        A dictionary containing the results.

    """

    return get_backend(backend).export_model(model_name, export_path, **kwargs)


__all__ = ["export_model"]
