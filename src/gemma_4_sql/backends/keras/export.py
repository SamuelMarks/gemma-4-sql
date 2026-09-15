"""Keras-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    import keras as _keras

    keras: Any = _keras
except (ImportError, AttributeError):
    keras = None


def export_model(model_name: str, export_path: str) -> JSONDict:
    """Export a Text-to-SQL model using the Keras backend.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If Keras dependencies are missing for export.
        ValueError: If loading the model fails.
    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing for export.")

    try:
        gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
        model = gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError) as e:
        msg = f"Failed to load model {model_name}"
        raise ValueError(msg) from e

    file_path = Path(export_path) / "model.keras"
    model.save(file_path)
    status = "exported_with_keras"

    return {"backend": "keras", "model": model_name, "export_path": export_path, "file_path": str(file_path), "status": status, "format": "keras_v3/keras_tensor"}
