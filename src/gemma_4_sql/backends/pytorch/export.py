"""PyTorch-specific model export pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_export import run_export_wrapper
from gemma_4_sql.backends.common_inference import make_model_cache_key
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as NativeGemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import ExportResultDict, TensorProtocol, TorchCausalLMProtocol, TorchTensorProtocol


@runtime_checkable
class _ClonableTensorProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a tensor supporting `.clone()`."""

    def clone(self) -> TorchTensorProtocol:
        """Return a copy of the tensor.

        Returns:
            Cloned `TorchTensorProtocol` tensor.

        """
        ...


@runtime_checkable
class _HasDeviceAndToProtocol(Protocol):
    """Protocol for tensors exposing `.device` and `.to('cpu')` for XLA host staging."""

    @property
    def device(self) -> str | Path | None:
        """Provide tensor device descriptor."""
        ...

    def to(self, target: str, /) -> TorchTensorProtocol:
        """Move tensor to `target` device.

        Args:
            target: Target device string (`'cpu'`).

        Returns:
            Host-staged tensor.

        """
        ...


def _stage_xla_tensor_to_cpu[TensorT: TorchTensorProtocol | TensorProtocol | _HasDeviceAndToProtocol](
    tensor: TensorT,
) -> TorchTensorProtocol | TensorProtocol | TensorT:
    """Move `tensor` from an `xla:*` device to `'cpu'` prior to `safetensors` serialization.

    Args:
        tensor: Parameter tensor from `state_dict()`.

    Returns:
        CPU-staged tensor when originating on XLA, otherwise original `tensor`.

    """
    if isinstance(tensor, _HasDeviceAndToProtocol) and str(tensor.device).startswith("xla"):
        return tensor.to("cpu")
    return tensor


@runtime_checkable
class _SafetensorsSaveFileProtocol(Protocol):
    """Protocol for `safetensors.torch.save_file`."""

    def __call__(
        self,
        tensors: Mapping[str, TorchTensorProtocol | TensorProtocol],
        filename: str,
        /,
    ) -> None:
        """Serialize `tensors` to a safetensors file at `filename`.

        Args:
            tensors: Mapping of parameter names to tensors.
            filename: Output `.safetensors` file path.

        """
        ...


@runtime_checkable
class _SavePretrainedModelProtocol(TorchCausalLMProtocol, Protocol):
    """Protocol for a PyTorch model supporting `.save_pretrained(export_path)`."""

    def save_pretrained(self, save_directory: str, /) -> str | None:
        """Persist model or adapter weights to `save_directory`.

        Args:
            save_directory: Destination directory path.

        Returns:
            Saved file path or `None`.

        """
        ...


@runtime_checkable
class _NativeFromPretrainedProtocol(Protocol):
    """Protocol for `NativeGemma4` exposing `.from_pretrained(model_name, config=...)`."""

    def from_pretrained(
        self,
        model_name_or_path: str | Path,
        config: NativeGemma4Config | None = None,
    ) -> _SavePretrainedModelProtocol:
        """Load or instantiate a `Gemma4ForCausalLM` from checkpoint path or preset.

        Args:
            model_name_or_path: Checkpoint path or preset identifier.
            config: Optional explicit `Gemma4Config`.

        Returns:
            Loaded `_SavePretrainedModelProtocol` instance.

        """
        ...


torch = optional_import("torch")
save_file: _SafetensorsSaveFileProtocol | None = import_optional_symbol(
    "safetensors.torch",
    "save_file",
    _SafetensorsSaveFileProtocol,
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


def _resolve_native_export_model(
    model_name: str,
    raw_cfg: StatusValue | NativeGemma4Config | None,
) -> _SavePretrainedModelProtocol:
    """Resolve a cached, checkpoint-restored, or configured native PyTorch Gemma 4 model.

    Args:
        model_name: Model cache key, checkpoint path, or preset name.
        raw_cfg: Optional explicit `Gemma4Config` passed by caller.

    Returns:
        Resolved `_SavePretrainedModelProtocol` model ready for state-dict serialization.

    """
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
    from gemma_4_sql.backends.pytorch.inference import MODEL_CACHE

    cfg_opt = raw_cfg if isinstance(raw_cfg, NativeGemma4Config) else None
    direct_cached = MODEL_CACHE.get_model(model_name)
    if direct_cached is not None and isinstance(direct_cached[0], _SavePretrainedModelProtocol):
        return direct_cached[0]
    from_pretrained_ref = (
        NativeGemma4.from_pretrained if isinstance(NativeGemma4, _NativeFromPretrainedProtocol) else None
    )
    keyed_cache_id = make_model_cache_key(
        model_name,
        backend_name="native",
        config=cfg_opt,
        factories=(from_pretrained_ref,) if from_pretrained_ref is not None else (),
    )
    keyed_cached = MODEL_CACHE.get_model(keyed_cache_id)
    if keyed_cached is not None and isinstance(keyed_cached[0], _SavePretrainedModelProtocol):
        return keyed_cached[0]

    model_path = Path(model_name)
    if model_path.exists() and isinstance(NativeGemma4, _NativeFromPretrainedProtocol):
        return NativeGemma4.from_pretrained(model_name, config=cfg_opt)

    cfg = cfg_opt if cfg_opt is not None else Gemma4Config.from_preset(model_name)
    if isinstance(NativeGemma4, _NativeFromPretrainedProtocol):
        return NativeGemma4.from_pretrained(model_name, config=cfg)
    instantiated: _SavePretrainedModelProtocol = NativeGemma4(cfg)
    return instantiated


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
            native_model = _resolve_native_export_model(model_name, kwargs.get("config"))
            tensors: dict[str, TorchTensorProtocol | TensorProtocol] = {
                k: _stage_xla_tensor_to_cpu(
                    v.clone() if k == "lm_head.weight" and isinstance(v, _ClonableTensorProtocol) else v
                )
                for k, v in native_model.state_dict().items()
            }
            save_pretrained_fn: Callable[[str], str | None] = native_model.save_pretrained
        else:

            @runtime_checkable
            class _ExportModelInstanceProtocol(Protocol):
                """Protocol for a loaded Hugging Face causal LM supporting `state_dict` and `save_pretrained`."""

                def state_dict(self) -> Mapping[str, TorchTensorProtocol | TensorProtocol]:
                    """Return model parameter state dictionary."""
                    ...

                def save_pretrained(self, save_directory: str, /) -> str | None:
                    """Persist model weights/adapters to `save_directory`."""
                    ...

            @runtime_checkable
            class _ExportCausalLMLoaderProtocol(Protocol):
                """Protocol for `Gemma4ForCausalLM` exposing `from_pretrained`."""

                def from_pretrained(self, model_id: str, /) -> _ExportModelInstanceProtocol:
                    """Load pretrained `Gemma4ForCausalLM`."""
                    ...

            gemma4_for_causal_lm_cls = import_optional_symbol(
                "transformers.models.gemma4",
                "Gemma4ForCausalLM",
                _ExportCausalLMLoaderProtocol,
            )
            if gemma4_for_causal_lm_cls is None:
                msg = (
                    f"Failed to load model {model_name}: transformers.models.gemma4.Gemma4ForCausalLM is not available."
                )
                raise ExportError(msg, backend=backend_alias)
            hf_model = gemma4_for_causal_lm_cls.from_pretrained(model_name)
            tensors = {k: _stage_xla_tensor_to_cpu(v) for k, v in hf_model.state_dict().items()}
            save_pretrained_fn = hf_model.save_pretrained
    except (ImportError, ValueError, OSError) as e:
        msg = f"Failed to load model {model_name}"
        raise ExportError(msg, backend=backend_alias, cause=e) from e

    out_dir = Path(export_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / "model.safetensors"
    if kwargs.get("export_type") == "adapter":
        save_pretrained_fn(export_path)
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
