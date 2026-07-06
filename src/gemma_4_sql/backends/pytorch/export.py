"""PyTorch-specific model export pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
torch = None
save_file = None
with catch_optional_imports():
    import torch
    from safetensors.torch import save_file


def _is_rank_zero() -> bool:
    """Check if current process is rank 0 in distributed training.

    Returns:
        A boolean indicating the result of the operation.
    """
    if torch is None:
        return True
    try:
        dist = __import__("torch.distributed", fromlist=[""])
        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError) as e:
        logger = logging.getLogger(__name__)
        logger.debug("Distributed not available or not initialized: %s", e)
    return True


def _save_real_model(model_name: str, export_path: str, *, is_rank_zero: bool = True) -> tuple[Path, str]:
    """Save a real PyTorch model using safetensors.

    Returns:
        object: The resulting output from the operation.

    """
    try:
        gemma4_for_causal_lm_cls = __import__("transformers.models.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
        model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
        tensors = model.state_dict()
    except (ImportError, ValueError) as e:
        msg = f"Failed to load model {model_name}"
        raise ValueError(msg) from e
    file_path = Path(export_path) / "model.safetensors"
    if is_rank_zero:  # pragma: no cover
        save_file(tensors, file_path)
    status = "exported_with_safetensors" if is_rank_zero else "skipped_non_rank_zero"
    return (file_path, status)


def export_model(model_name: str, export_path: str) -> JSONDict:
    """Export a Text-to-SQL model using the PyTorch backend.

    Args:
    ----
        model_name: The name of the model to export.
        export_path: The destination path for the checkpoint.

    Returns:
    -------
        A dictionary containing export metadata.

    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    is_rank_zero = _is_rank_zero()
    if torch is None or save_file is None:
        raise RuntimeError("PyTorch or safetensors missing, cannot export model.")
    (file_path, status) = _save_real_model(model_name, export_path, is_rank_zero=is_rank_zero)
    return {"backend": "pytorch", "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "safetensors"}
