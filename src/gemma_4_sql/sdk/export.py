"""
SDK Export module.
"""

from __future__ import annotations

from typing import Any


def export_model(
    model_name: str, export_path: str, backend: str = "jax"
) -> dict[str, Any]:
    """
    Exports a trained Text-to-SQL model.

    Args:
        model_name: The name or path of the model.
        export_path: The filesystem path to export the checkpoint.
        backend: The backend framework ('jax', 'keras', or 'maxtext').

    Returns:
        Export results dictionary.
    """
    from gemma_4_sql.sdk.registry import get_backend
    return get_backend(backend).export_model(model_name, export_path)
