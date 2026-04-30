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
    if backend == "jax":
        from gemma_4_sql.backends.jax.export import export_model as jax_export

        return jax_export(model_name, export_path)
    elif backend == "keras":
        from gemma_4_sql.backends.keras.export import (
            export_model as keras_export,
        )

        return keras_export(model_name, export_path)
    elif backend == "maxtext":
        from gemma_4_sql.backends.maxtext.export import export_model as maxtext_export

        return maxtext_export(model_name, export_path)
    elif backend == "pytorch":
        from gemma_4_sql.backends.pytorch.export import (
            export_model as pytorch_export,
        )

        return pytorch_export(model_name, export_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")
