"""MLX-specific model export pipeline."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import ExportResultDict, JSONValue, MlxCausalLMProtocol, TokenizerProtocol


@runtime_checkable
class _MlxLoadCallableProtocol(Protocol):
    """Protocol for `mlx_lm.load(model_name)`."""

    def __call__(self, model_name: str, /) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None]:
        """Load an MLX model and tokenizer pair."""
        ...


mx = optional_import("mlx.core")
mlx_load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None]] | None = optional_attr(
    "mlx_lm",
    "load",
    _MlxLoadCallableProtocol,
)


def export_model(model_name: str, export_path: str, **_kwargs: StatusValue) -> ExportResultDict:
    """Export a Text-to-SQL model using the MLX backend.

    Saves model weights in safetensors format alongside configuration metadata (`config.json`)
    to enable standalone re-loading with `mlx_lm`.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.
        **_kwargs: Additional export keyword arguments.

    Returns:
        A structured `ExportResultDict` containing the export metadata.

    Raises:
        DependencyMissingError: If MLX is not installed.
        ExportError: If loading or exporting the model fails.

    """
    if mx is None:
        raise DependencyMissingError("MLX is not installed, cannot export model.", backend="mlx")

    def _execute() -> tuple[Path, str]:
        """Load MLX model parameters and write safetensors and `config.json`."""
        assert mx is not None
        export_dir = Path(export_path)
        try:
            load_fn = mlx_load if mlx_load is not None else optional_attr("mlx_lm", "load", _MlxLoadCallableProtocol)
            if load_fn is None:
                raise ImportError("mlx_lm.load is not available.")
            (model, _tokenizer) = load_fn(model_name)
            tensors = dict(model.parameters())
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError, OSError) as e:
            raise ExportError(f"Failed to load MLX model {model_name}", backend="mlx", cause=e) from e

        file_path = export_dir / "model.safetensors"
        mx.save_safetensors(str(file_path), tensors)

        config_path = export_dir / "config.json"
        cfg_dict: dict[str, JSONValue] = {"model_type": "gemma4", "model_name": model_name}
        with contextlib.suppress(AttributeError, TypeError):
            raw_cfg = model.config  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            cfg_dict = dict(raw_cfg) if isinstance(raw_cfg, Mapping) else dataclasses.asdict(raw_cfg)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

        return (file_path, "exported_with_safetensors")

    return run_export_wrapper("mlx", model_name, export_path, "safetensors", _execute)
