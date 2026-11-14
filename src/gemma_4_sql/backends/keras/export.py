"""Keras-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend
from gemma_4_sql.backends.keras._runtime import load_keras_gemma_model, load_preset_with_hf_fallback
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import ExportResultDict, KerasCausalLMProtocol


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


keras = optional_import("keras")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = import_optional_symbol(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def export_model(model_name: str, export_path: str, **kwargs: StatusValue) -> ExportResultDict:
    """Export a Text-to-SQL model using the Keras backend (`keras_jax`, `keras_tensorflow`, `keras_pytorch`).

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.
        **kwargs: Additional export keyword arguments (`backend_alias`, `keras_backend`).

    Returns:
        A structured `ExportResultDict` containing the export metadata.

    Raises:
        DependencyMissingError: If Keras dependencies are missing for export.
        ExportError: If loading or exporting the model fails.

    """
    backend_alias = str(kwargs.get("backend_alias", "keras"))
    explicit_kb = (
        str(kwargs["keras_backend"]) if "keras_backend" in kwargs and kwargs["keras_backend"] is not None else None
    )
    active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing for export.", backend=backend_alias)

    def _execute() -> tuple[Path, str]:
        """Load the Keras model and save it to `model.keras`.

        Returns:
            Tuple of `(saved_keras_file_path, export_status_message)`.

        Raises:
            ImportError: Internally raised and wrapped if KerasHub / KerasNLP `GemmaCausalLM` is unavailable.
            ExportError: If loading the Keras model fails.

        """
        try:
            gemma_causal_lm_cls = load_keras_gemma_model(
                model_name,
                _GemmaCausalLMLoaderProtocol,
                explicit_cls=GemmaCausalLM,
                symbol_importer=import_optional_symbol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_hub / keras_nlp GemmaCausalLM is not available.")
            model = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
        except (ImportError, ValueError, AttributeError) as e:
            msg = f"Failed to load model {model_name}"
            raise ExportError(msg, backend=backend_alias, cause=e) from e

        file_path = Path(export_path) / "model.keras"
        model.save(file_path)
        return (file_path, "exported_with_keras")

    res = run_export_wrapper(backend_alias, model_name, export_path, "keras_v3/keras_tensor", _execute)
    res["keras_backend"] = active_kb
    return res
