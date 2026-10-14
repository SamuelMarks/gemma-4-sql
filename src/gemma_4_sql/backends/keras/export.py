"""Keras-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, ExportError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import ExportResultDict

keras = optional_import("keras")
GemmaCausalLM: Any = optional_attr("keras_nlp.models", "GemmaCausalLM")


def export_model(model_name: str, export_path: str, **_kwargs: object) -> ExportResultDict:
    """Export a Text-to-SQL model using the Keras backend.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.
        **_kwargs: Additional export keyword arguments.

    Returns:
        A structured `ExportResultDict` containing the export metadata.

    Raises:
        DependencyMissingError: If Keras dependencies are missing for export.
        ExportError: If loading or exporting the model fails.

    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing for export.", backend="keras")

    try:
        gemma_causal_lm_cls = GemmaCausalLM
        if gemma_causal_lm_cls is None:
            gemma_causal_lm_cls = optional_attr("keras_nlp.models", "GemmaCausalLM")
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        model = gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError, AttributeError) as e:
        msg = f"Failed to load model {model_name}"
        raise ExportError(msg, backend="keras", cause=e) from e

    file_path = Path(export_path) / "model.keras"
    model.save(file_path)
    status = "exported_with_keras"

    return {
        "backend": "keras",
        "model": model_name,
        "export_path": export_path,
        "file_path": str(file_path),
        "status": status,
        "format": "keras_v3/keras_tensor",
    }
