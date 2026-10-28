"""Keras-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import ExportResultDict, KerasCausalLMProtocol


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


keras = optional_import("keras")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = optional_attr(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def export_model(model_name: str, export_path: str, **_kwargs: StatusValue) -> ExportResultDict:
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
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing for export.", backend="keras")

    def _execute() -> tuple[Path, str]:
        """Load the Keras model and save it to `model.keras`."""
        try:
            gemma_causal_lm_cls = GemmaCausalLM
            if gemma_causal_lm_cls is None:
                gemma_causal_lm_cls = optional_attr(
                    "keras_nlp.models",
                    "GemmaCausalLM",
                    _GemmaCausalLMLoaderProtocol,
                )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model = gemma_causal_lm_cls.from_preset(model_name)
        except (ImportError, ValueError, AttributeError) as e:
            msg = f"Failed to load model {model_name}"
            raise ExportError(msg, backend="keras", cause=e) from e

        file_path = Path(export_path) / "model.keras"
        model.save(file_path)
        return (file_path, "exported_with_keras")

    return run_export_wrapper("keras", model_name, export_path, "keras_v3/keras_tensor", _execute)
