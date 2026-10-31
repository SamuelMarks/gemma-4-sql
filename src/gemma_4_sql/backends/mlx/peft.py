"""MLX-specific PEFT / LoRA implementation."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeGuard, runtime_checkable

import mlx.core as _mlx_core_types
import mlx.nn as _mlx_nn_types

from gemma_4_sql._types_models import MlxLMLoadProtocol
from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale, matches_target_module
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    PEFTResultDict,
    extract_status_kwargs,
)

logger = logging.getLogger(__name__)


type MlxTensorLike = _mlx_core_types.array | MlxArrayProtocol


def _as_mlx_array(val: MlxTensorLike) -> TypeGuard[MlxArrayProtocol]:
    """Narrow `val` to `MlxArrayProtocol` for matrix and elementwise operations.

    Args:
        val: MLX array or `MlxArrayProtocol` instance.

    Returns:
        `True` for array arithmetic.

    """
    _ = val
    return True


def _as_array_t[ArrayT: MlxTensorLike](val: MlxArrayProtocol, _sample: ArrayT) -> TypeGuard[ArrayT]:
    """Narrow `val` back to caller's `ArrayT`.

    Args:
        val: Computed `MlxArrayProtocol` result.
        _sample: Input array of type `ArrayT`.

    Returns:
        `True` so `__call__` preserves `ArrayT`.

    """
    _ = (val, _sample)
    return True


@runtime_checkable
class _MlxLinearProtocol(Protocol):
    """Protocol for an `mlx.nn.Linear` module exposing `.weight` and optional `.bias`."""

    @property
    def weight(self) -> MlxTensorLike:
        """Provide the 2D weight matrix."""
        ...

    @property
    def bias(self) -> MlxTensorLike | None:
        """Provide the optional 1D bias vector."""
        ...


@runtime_checkable
class _MlxDropoutProtocol(Protocol):
    """Protocol for an `mlx.nn.Dropout` or `mlx.nn.Identity` callable module."""

    def __call__(self, x: MlxTensorLike) -> MlxTensorLike:
        """Apply dropout or identity pass-through to `x`."""
        ...


@runtime_checkable
class _MlxModuleHierarchyProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _MlxLinearProtocol]]:
        """Yield `(path, submodule)` pairs across the module tree."""
        ...


@runtime_checkable
class _MlxWeightUpdateProtocol(Protocol):
    """Protocol for a model supporting `.update(weights)` or `.load_weights(...)`."""

    def update(self, weights: dict[str, MlxTensorLike]) -> None:
        """Update model weights from a dictionary."""
        ...


@runtime_checkable
class _MlxTargetObjectProtocol(Protocol):
    """Protocol satisfied by any MLX model or module container."""

    def __str__(self) -> str:
        """Return string representation."""
        ...


@runtime_checkable
class _MlxTrainableModuleProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting freeze, unfreeze, update_modules, and weight IO."""

    def freeze(self) -> None:
        """Freeze module parameters."""
        ...

    def unfreeze(self, *, keys: Sequence[str] | None = None) -> None:
        """Unfreeze selected module keys."""
        ...

    def update_modules(self, modules: Mapping[str, _MlxTargetObjectProtocol] | _MlxTargetObjectProtocol) -> None:
        """Replace submodules in-place."""
        ...

    def trainable_parameters(self) -> Mapping[str, MlxArrayProtocol]:
        """Return the nested dictionary of trainable parameters."""
        ...

    def load_weights(self, file_or_weights: str, *, strict: bool = True) -> None:
        """Load weights from a safetensors file."""
        ...

    def update(self, parameters: Mapping[str, MlxArrayProtocol]) -> None:
        """Update module parameters."""
        ...


def _is_mlx_trainable(obj: _MlxTargetObjectProtocol) -> TypeGuard[_MlxTrainableModuleProtocol]:
    """Narrow `obj` to `_MlxTrainableModuleProtocol` for attribute-guarded method calls.

    Args:
        obj: Target MLX module or test double.

    Returns:
        `True` so module methods can be invoked inside `contextlib.suppress`.

    """
    _ = obj
    return True


def _is_mlx_causal_lm_input(
    obj: BackendKwargValue | MlxCausalLMProtocol | _MlxModuleHierarchyProtocol,
) -> TypeGuard[MlxCausalLMProtocol]:
    """Narrow `obj` to `MlxCausalLMProtocol`.

    Args:
        obj: Candidate MLX model passed via `kwargs['model']`.

    Returns:
        `True` when `obj` is provided.

    """
    _ = obj
    return True


mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
load: MlxLMLoadProtocol | None = optional_attr(
    "mlx_lm",
    "load",
    MlxLMLoadProtocol,
)


class MLXLoRALinear(_mlx_nn_types.Module):
    """Low-rank adaptation (LoRA) linear module for MLX.

    Decomposes weight updates into low-rank matrices:
        `y = x W^T + bias + (alpha / r) * (Dropout(x) @ A) @ B`
    where:
        `W` is the base weight matrix with shape `(out_features, in_features)`,
        `A` is the down-projection adapter with shape `(in_features, r)`,
        `B` is the up-projection adapter with shape `(r, out_features)`.
    """

    weight: MlxTensorLike
    lora_a: MlxTensorLike
    lora_b: MlxTensorLike
    bias: MlxTensorLike | None
    dropout: _MlxDropoutProtocol
    in_features: int
    out_features: int
    r: int
    lora_alpha: float
    scale: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = False,
    ) -> None:
        """Initialize the MLX LoRA linear layer.

        Args:
            in_features: Number of input features (`d_in`).
            out_features: Number of output features (`d_out`).
            r: Rank of the low-rank adapters (`r > 0`).
            lora_alpha: Scaling parameter (`alpha`).
            lora_dropout: Dropout probability applied to input before LoRA adapter.
            bias: Whether to allocate and apply an additive bias vector.

        Raises:
            DependencyMissingError: If MLX is not installed.
            ValueError: If rank `r` is less than or equal to 0.

        """
        if nn is None or mx is None:
            raise DependencyMissingError("MLX dependencies are missing.")
        self.scale = compute_lora_scale(lora_alpha, r)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)

        self.weight = mx.zeros((out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        scale_init = 1.0 / math.sqrt(r)
        self.lora_a = mx.random.uniform(
            low=-scale_init,
            high=scale_init,
            shape=(in_features, r),
        )
        self.lora_b = mx.zeros((r, out_features))

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = nn.Identity()

    @property
    def W(self) -> MlxTensorLike:
        """Provide the base weight matrix `W` with shape `(out_features, in_features)`.

        Returns:
            The frozen base weight matrix.

        """
        return self.weight

    @property
    def A(self) -> MlxTensorLike:
        """Provide the down-projection adapter `A` with shape `(in_features, r)`.

        Returns:
            The down-projection adapter tensor.

        """
        return self.lora_a

    @property
    def B(self) -> MlxTensorLike:
        """Provide the up-projection adapter `B` with shape `(r, out_features)`.

        Returns:
            The up-projection adapter tensor.

        """
        return self.lora_b

    @classmethod
    def from_linear(
        cls,
        linear: _MlxLinearProtocol,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> MLXLoRALinear:
        """Construct an `MLXLoRALinear` instance wrapping an existing `nn.Linear` module.

        Args:
            linear: The source `mlx.nn.Linear` module whose weights will be copied.
            r: Rank of the low-rank adapters.
            lora_alpha: Scaling parameter.
            lora_dropout: Dropout probability applied prior to LoRA projection.

        Returns:
            An `MLXLoRALinear` layer initialized with the linear module's weights.

        """
        bias_val: MlxTensorLike | None = None
        with contextlib.suppress(AttributeError):
            bias_val = linear.bias
        has_bias = bias_val is not None
        weight = linear.weight
        out_features, in_features = weight.shape[0], weight.shape[1]
        lora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=has_bias,
        )
        lora_layer.weight = weight
        if has_bias:
            lora_layer.bias = bias_val
        return lora_layer

    def __call__[ArrayT: MlxTensorLike](self, x: ArrayT) -> ArrayT:
        """Execute the LoRA-augmented linear forward pass.

        Computes:
            `y = x @ W^T + bias + scale * (dropout(x) @ A) @ B`

        Args:
            x: Input tensor of shape `(..., in_features)`.

        Returns:
            Output tensor of shape `(..., out_features)`.

        """
        x_arr: MlxTensorLike = x
        w_arr: MlxTensorLike = self.W
        a_arr: MlxTensorLike = self.A
        b_arr: MlxTensorLike = self.B
        drop_x: MlxTensorLike = self.dropout(x)
        assert _as_mlx_array(x_arr)
        assert _as_mlx_array(w_arr)
        assert _as_mlx_array(a_arr)
        assert _as_mlx_array(b_arr)
        assert _as_mlx_array(drop_x)
        base = x_arr @ w_arr.T
        if self.bias is not None:
            bias_arr: MlxTensorLike = self.bias
            assert _as_mlx_array(bias_arr)
            base = base + bias_arr
        lora_term = (drop_x @ a_arr) @ b_arr
        out = base + self.scale * lora_term
        assert _as_array_t(out, x)
        return out

    def save_adapters(self, file_path: str | Path) -> None:
        """Save this layer's LoRA adapter weights (`lora_a`, `lora_b`) to a safetensors file.

        Args:
            file_path: Destination path for the safetensors file.

        Raises:
            DependencyMissingError: If MLX is missing.

        """
        if mx is None:
            raise DependencyMissingError("MLX dependencies are missing.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adapters = {"lora_a": self.A, "lora_b": self.B}
        mx.save_safetensors(str(path), adapters)

    def load_adapters(self, file_path: str | Path) -> None:
        """Load this layer's LoRA adapter weights (`lora_a`, `lora_b`) from a safetensors file.

        Args:
            file_path: Source path of the safetensors file.

        Raises:
            DependencyMissingError: If MLX is missing.
            KeyError: If required adapter keys are missing from the safetensors file.

        """
        if mx is None:
            raise DependencyMissingError("MLX dependencies are missing.")
        weights = mx.load(str(file_path))
        if "lora_a" not in weights or "lora_b" not in weights:
            raise KeyError("Safetensors file missing 'lora_a' or 'lora_b' keys.")
        self.lora_a = weights["lora_a"]
        self.lora_b = weights["lora_b"]


MlxLoRALinear = MLXLoRALinear


def inject_lora[ModelT: _MlxTargetObjectProtocol](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[ModelT, int]:
    """Inject LoRA adapters into designated projection modules of an MLX model.

    Uses `model.update_modules(tree_unflatten(replacements))` to replace matching
    `mlx.nn.Linear` submodules with `MLXLoRALinear` without reflection (`getattr`/`setattr`).

    Args:
        model: Target MLX neural network model.
        target_modules: List of module name suffixes or substrings to adapt (e.g. `['q_proj', 'v_proj']`).
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling coefficient.
        lora_dropout: Dropout probability applied before LoRA projection.

    Returns:
        A tuple of `(adapted_model, injected_count)`.

    Raises:
        DependencyMissingError: If MLX is not installed.

    """
    if nn is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    with contextlib.suppress(AttributeError, TypeError):
        m_freeze = model
        assert _is_mlx_trainable(m_freeze)
        m_freeze.freeze()

    if not target_modules or not isinstance(model, _MlxModuleHierarchyProtocol):
        return model, 0

    replacements: list[tuple[str, MLXLoRALinear]] = []
    for name, submodule in model.named_modules():
        if isinstance(submodule, nn.Linear) and matches_target_module(name, target_modules):
            lora_module = MLXLoRALinear.from_linear(
                submodule,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            lora_ref: _MlxTargetObjectProtocol = lora_module
            assert _is_mlx_trainable(lora_ref)
            lora_ref.freeze()
            lora_ref.unfreeze(keys=["lora_a", "lora_b"])
            replacements.append((name, lora_module))

    if replacements:
        import mlx.utils as _mlx_utils

        with contextlib.suppress(AttributeError, TypeError):
            m_upd: _MlxTargetObjectProtocol = model
            assert _is_mlx_trainable(m_upd)
            m_upd.update_modules(_mlx_utils.tree_unflatten(replacements))
        if isinstance(model, list):
            for name, lora_mod in replacements:
                if "." in name:
                    idx_str, child_key = name.split(".", 1)
                    model[int(idx_str)].update_modules({child_key: lora_mod})
                else:
                    model[int(name)] = lora_mod

    return model, len(replacements)


def save_adapter_weights(
    model: MlxCausalLMProtocol | _MlxModuleHierarchyProtocol | _MlxWeightUpdateProtocol | _MlxTargetObjectProtocol,
    save_path: str | Path,
) -> None:
    """Save all LoRA adapter weights from an MLX model to a safetensors file.

    Extracts all trainable parameters (LoRA adapters) and serializes them into
    the specified safetensors file.

    Args:
        model: MLX model containing LoRA adapters.
        save_path: File path or directory path to save the adapter weights.

    Raises:
        DependencyMissingError: If MLX is missing.

    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing.")
    import mlx.utils as _mlx_utils

    raw_path = Path(save_path)
    path = (raw_path / "adapter.safetensors") if (raw_path.is_dir() or raw_path.suffix != ".safetensors") else raw_path
    path.parent.mkdir(parents=True, exist_ok=True)

    assert _is_mlx_trainable(model)
    trainable_dict: dict[str, MlxArrayProtocol] = dict(_mlx_utils.tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), trainable_dict)


def load_adapter_weights(
    model: MlxCausalLMProtocol | _MlxModuleHierarchyProtocol | _MlxWeightUpdateProtocol | _MlxTargetObjectProtocol,
    load_path: str | Path,
) -> None:
    """Load LoRA adapter weights into an MLX model from a safetensors file.

    Args:
        model: MLX model containing LoRA adapters.
        load_path: Path to the safetensors file.

    Raises:
        DependencyMissingError: If MLX is missing.
        FileNotFoundError: If the adapter weights file does not exist.

    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing.")
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {load_path}")
    assert _is_mlx_trainable(model)
    try:
        model.load_weights(str(path), strict=False)
    except AttributeError:
        with contextlib.suppress(AttributeError, TypeError):
            weights = mx.load(str(path))
            model.update(weights)


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: BackendKwargValue | MlxCausalLMProtocol | _MlxModuleHierarchyProtocol,
) -> PEFTResultDict:
    """Apply LoRA to a model using the MLX backend.

    Args:
        model_name: The name or path of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including `'output_dir'`, `'model'`, and `'raise_on_error'`.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        TrainingError: If LoRA injection fails and `raise_on_error` is enabled.

    """
    if nn is None or load is None or mx is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")

    def _inject() -> int:
        """Load or reuse MLX model and inject LoRA linear layers."""
        assert load is not None
        provided_model = kwargs.get("model")
        if provided_model is not None:
            assert _is_mlx_causal_lm_input(provided_model)
            model_obj: MlxCausalLMProtocol = provided_model
        else:
            loaded = load(model_name)
            model_obj = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

        model, injected_count = inject_lora(
            model=model_obj,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        raw_output_dir = kwargs.get("output_dir")
        if raw_output_dir is not None:
            save_adapter_weights(model, str(raw_output_dir))
        return injected_count

    wrapper_kwargs = extract_status_kwargs(kwargs)
    return apply_lora_wrapper(
        "mlx",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **wrapper_kwargs,
    )


__all__ = [
    "MLXLoRALinear",
    "MlxLoRALinear",
    "apply_lora",
    "inject_lora",
    "load_adapter_weights",
    "save_adapter_weights",
]
