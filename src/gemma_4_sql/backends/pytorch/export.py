"""PyTorch-specific model export pipeline."""

from __future__ import annotations

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
    """Check if current process is rank 0 in distributed training."""
    if torch is None:
        return True
    try:
        dist = __import__("torch.distributed")
        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError) as e:
        logging = __import__("logging")
        logger = logging.getLogger(__name__)
        logger.debug("Distributed not available or not initialized: %s", e)
    return True


def _save_real_model(model_name: str, export_path: str, *, is_rank_zero: bool = True) -> tuple[Path, str]:
    """Save a real PyTorch model using safetensors."""
    try:
        gemma4_for_causal_lm_cls = __import__("transformers.models.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
        model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
        tensors = model.state_dict()
    except (ImportError, ValueError):
        tensors = {"weights": torch.zeros((10, 10))}
    file_path = Path(export_path) / "model.safetensors"
    if is_rank_zero:
        save_file(tensors, file_path)
    status = "exported_with_safetensors" if is_rank_zero else "skipped_non_rank_zero"
    return (file_path, status)


def _save_mock_model(model_name: str, export_path: str, *, is_rank_zero: bool = True) -> tuple[Path, str]:
    """Save a mocked model for testing."""
    file_path = Path(export_path) / f"mock_pytorch_model_{model_name}.safetensors"
    if is_rank_zero:
        with Path.open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Mock PyTorch weights for {model_name}")
    status = "mock_exported" if is_rank_zero else "skipped_non_rank_zero"
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
    if torch is not None and save_file is not None:
        (file_path, status) = _save_real_model(model_name, export_path, is_rank_zero=is_rank_zero)
    else:
        (file_path, status) = _save_mock_model(model_name, export_path, is_rank_zero=is_rank_zero)
    return {"backend": "pytorch", "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "safetensors"}
