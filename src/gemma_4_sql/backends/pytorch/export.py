"""PyTorch-specific model export pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as NativeGemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import ExportResultDict, TensorProtocol, TorchCausalLMProtocol, TorchTensorProtocol


@runtime_checkable
class _ClonableTensorProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a tensor supporting `.clone()`."""

    def clone(self) -> TorchTensorProtocol:
        """Return a copy of the tensor."""
        ...


torch = optional_import("torch")
_raw_save_file = optional_attr("safetensors.torch", "save_file")
save_file: Callable[[Mapping[str, TorchTensorProtocol | TensorProtocol], str], None] | None = (  # ty: ignore[invalid-assignment]
    _raw_save_file if callable(_raw_save_file) else None
)


def _is_rank_zero() -> bool:
    """Check if current process is rank 0 in distributed training.

    Returns:
        `True` if distributed is uninitialized or current process is rank 0.

    """
    if torch is None:
        return True
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError) as e:
        logger = logging.getLogger(__name__)
        logger.debug("Distributed not available or not initialized: %s", e)
    return True


def _save_real_model(
    model_name: str,
    export_path: str,
    *,
    is_rank_zero: bool = True,
    backend_alias: str = "pytorch",
    **kwargs: StatusValue | NativeGemma4Config,
) -> tuple[Path, str]:
    """Save a real PyTorch model using safetensors.

    Args:
        model_name: Model identifier or path.
        export_path: Destination path.
        is_rank_zero: Whether the current process is rank 0.
        backend_alias: Target backend alias.
        **kwargs: Additional model and export keyword arguments.

    Returns:
        A tuple of `(file_path, status)`.

    Raises:
        ExportError: If loading the model fails.

    """
    try:
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4Config
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4

            raw_cfg = kwargs.get("config")
            cfg = raw_cfg if isinstance(raw_cfg, NativeGemma4Config) else Gemma4Config.from_preset(model_name)
            model: TorchCausalLMProtocol = NativeGemma4(cfg)
            tensors: dict[str, TorchTensorProtocol | TensorProtocol] = {k: v.clone() if k == "lm_head.weight" and isinstance(v, _ClonableTensorProtocol) else v for k, v in model.state_dict().items()}
        else:
            from transformers.models.gemma4 import Gemma4ForCausalLM

            gemma4_for_causal_lm_cls = Gemma4ForCausalLM
            model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
            tensors = dict(model.state_dict())
    except (ImportError, ValueError, OSError) as e:
        msg = f"Failed to load model {model_name}"
        raise ExportError(msg, backend=backend_alias, cause=e) from e

    out_dir = Path(export_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / "model.safetensors"
    if kwargs.get("export_type") == "adapter":
        model.save_pretrained(export_path)  # pyright: ignore[reportAttributeAccessIssue]
        file_path = out_dir / "adapter_model.safetensors"
    elif is_rank_zero and save_file is not None:
        save_file(tensors, str(file_path))
    status = "exported_with_safetensors" if is_rank_zero else "skipped_non_rank_zero"
    return (file_path, status)


def export_model(
    model_name: str,
    export_path: str,
    **kwargs: StatusValue | NativeGemma4Config,
) -> ExportResultDict:
    """Export a Text-to-SQL model using the PyTorch backend.

    Args:
        model_name: The name or path of the model.
        export_path: Destination path for exported artifacts.
        **kwargs: Optional keyword arguments like `backend_alias` or `config`.

    Returns:
        Dictionary indicating status and file path.

    Raises:
        DependencyMissingError: If PyTorch or safetensors are missing.

    """
    if torch is None or save_file is None:
        raise DependencyMissingError("PyTorch or safetensors missing, cannot export model.", backend="pytorch")
    is_rank_0 = _is_rank_zero()
    backend_alias = str(kwargs.get("backend_alias", kwargs.get("backend", "pytorch")))
    save_kwargs = dict(kwargs)
    save_kwargs.pop("backend_alias", None)
    save_kwargs.pop("backend", None)
    return run_export_wrapper(
        backend_alias,
        model_name,
        export_path,
        "safetensors",
        lambda: _save_real_model(
            model_name,
            export_path,
            is_rank_zero=is_rank_0,
            backend_alias=backend_alias,
            **save_kwargs,
        ),
    )
