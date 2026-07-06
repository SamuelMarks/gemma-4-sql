"""MLX-specific model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
mx = None
with catch_optional_imports():
    import mlx.core as mx


def export_model(model_name: str, export_path: str) -> JSONDict:
    """Export a Text-to-SQL model using the MLX backend.

    Args:
        model_name: The name of the target model.
        export_path: The path where the model will be exported.

    Returns:
        A dictionary containing the results.
    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if mx is None:
        raise RuntimeError("MLX is not installed, cannot export model.")

    try:
        load = __import__("mlx_lm", fromlist=["load"]).load
        (model, _) = load(model_name)
        tensors = dict(model.parameters())
    except (ImportError, ValueError, RuntimeError, TypeError, AttributeError, OSError) as e:
        raise ValueError(f"Failed to load MLX model {model_name}") from e

    file_path = Path(export_path) / "model.safetensors"
    mx.save_safetensors(str(file_path), tensors)
    status = "exported_with_safetensors"

    return {"backend": "mlx", "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "safetensors"}
