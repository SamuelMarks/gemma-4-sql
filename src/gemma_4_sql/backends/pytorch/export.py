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


def _save_real_model(model_name: str, export_path: str, *, is_rank_zero: bool = True, backend_alias: str = "pytorch", **kwargs: object) -> tuple[Path, str]:
    """Save a real PyTorch model using safetensors.

    Returns:
        object: The resulting output from the operation.

    """
    try:
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4Config
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4

            cfg = kwargs.get("config") or (Gemma4Config(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128) if kwargs.get("test_mode") else Gemma4Config())
            model = NativeGemma4(cfg)
            tensors = {k: v.clone() if k == "lm_head.weight" else v for k, v in model.state_dict().items()}
        else:
            gemma4_for_causal_lm_cls = __import__("transformers.models.gemma4", fromlist=["Gemma4ForCausalLM"]).Gemma4ForCausalLM
            model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
            tensors = model.state_dict()
    except (ImportError, ValueError, OSError) as e:
        msg = f"Failed to load model {model_name}"
        raise ValueError(msg) from e
    file_path = Path(export_path) / "model.safetensors"
    if is_rank_zero:  # pragma: no cover
        save_file(tensors, file_path)
    status = "exported_with_safetensors" if is_rank_zero else "skipped_non_rank_zero"
    return (file_path, status)


def export_model(model_name: str, export_path: str, **kwargs: object) -> JSONDict:
    """Export a Text-to-SQL model using the PyTorch backend.

    Args:
        model_name: The name or path of the model.
        export_path: Destination path for exported artifacts.
        **kwargs: Optional keyword arguments like backend_alias.

    Returns:
        Dictionary indicating status and file path.
    """
    if torch is None or save_file is None:
        raise RuntimeError("PyTorch or safetensors missing, cannot export model.")
    Path(export_path).mkdir(parents=True, exist_ok=True)
    is_rank_0 = _is_rank_zero()
    backend_alias = str(kwargs.get("backend_alias", kwargs.get("backend", "pytorch")))
    save_kwargs = dict(kwargs)
    save_kwargs.pop("backend_alias", None)
    save_kwargs.pop("backend", None)
    (file_path, status) = _save_real_model(model_name, export_path, is_rank_zero=is_rank_0, backend_alias=backend_alias, **save_kwargs)
    return {"backend": backend_alias, "model": model_name, "export_path": export_path, "file_path": file_path, "status": status, "format": "safetensors"}
