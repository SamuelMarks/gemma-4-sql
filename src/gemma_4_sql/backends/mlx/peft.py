"""MLX-specific PEFT / LoRA implementation."""

from __future__ import annotations

import contextlib
import logging
import math
from collections import UserDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale, matches_target_module
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    PEFTResultDict,
    TokenizerProtocol,
)

logger = logging.getLogger(__name__)


type MlxTensorLike = MlxArrayProtocol | SupportsFloat


@runtime_checkable
class _MlxLinearProtocol(Protocol):
    """Protocol for an `mlx.nn.Linear` module exposing `.weight` and optional `.bias`."""

    @property
    def weight(self) -> MlxTensorLike:
        """Provide the 2D weight matrix."""
        ...


@runtime_checkable
class _MlxDropoutProtocol(Protocol):
    """Protocol for an `mlx.nn.Dropout` or `mlx.nn.Identity` callable module."""

    def __call__[ArrayT: MlxTensorLike](self, x: ArrayT) -> ArrayT:
        """Apply dropout or identity pass-through to `x`."""
        ...


@runtime_checkable
class _MlxModuleHierarchyProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _MlxLinearProtocol]]:
        """Yield `(path, submodule)` pairs across the module tree."""
        ...


mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
_raw_load = optional_attr("mlx_lm", "load")
load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | None] | MlxCausalLMProtocol] | None = (  # ty: ignore[invalid-assignment]
    _raw_load if callable(_raw_load) else None
)


class _FallbackModuleBase(UserDict[str, MlxArrayProtocol]):
    """Fallback base class when `mlx.nn` is not installed."""


_ModuleBase = nn.Module if nn is not None else _FallbackModuleBase


class MLXLoRALinear(_ModuleBase):  # pyright: ignore[reportUntypedBaseClass,reportGeneralTypeIssues]
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
    def W(self) -> MlxArrayProtocol:
        """Provide the base weight matrix `W` with shape `(out_features, in_features)`.

        Returns:
            The frozen base weight matrix.

        """
        self["weight"] = self.pop("W", self["weight"])  # pyright: ignore[reportIndexIssue,reportAttributeAccessIssue]
        return self["weight"]  # pyright: ignore[reportIndexIssue]

    @W.deleter
    def W(self) -> None:
        """No-op deleter required by `mlx.nn.Module` before storing `self['W']`."""

    @property
    def A(self) -> MlxArrayProtocol:
        """Provide the down-projection adapter `A` with shape `(in_features, r)`.

        Returns:
            The down-projection adapter tensor.

        """
        self["lora_a"] = self.pop("A", self["lora_a"])  # pyright: ignore[reportIndexIssue,reportAttributeAccessIssue]
        return self["lora_a"]  # pyright: ignore[reportIndexIssue]

    @A.deleter
    def A(self) -> None:
        """No-op deleter required by `mlx.nn.Module` before storing `self['A']`."""

    @property
    def B(self) -> MlxArrayProtocol:
        """Provide the up-projection adapter `B` with shape `(r, out_features)`.

        Returns:
            The up-projection adapter tensor.

        """
        self["lora_b"] = self.pop("B", self["lora_b"])  # pyright: ignore[reportIndexIssue,reportAttributeAccessIssue]
        return self["lora_b"]  # pyright: ignore[reportIndexIssue]

    @B.deleter
    def B(self) -> None:
        """No-op deleter required by `mlx.nn.Module` before storing `self['B']`."""

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
            bias_val = linear.bias  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        has_bias = bias_val is not None
        weight = linear.weight
        out_features, in_features = weight.shape  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
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
        base = x @ self.W.T  # pyright: ignore[reportOperatorIssue]  # ty: ignore[unsupported-operator]
        if self.bias is not None:
            base = base + self.bias  # pyright: ignore[reportOperatorIssue]
        lora_term = (self.dropout(x) @ self.A) @ self.B  # pyright: ignore[reportOperatorIssue]  # ty: ignore[unsupported-operator]
        res: ArrayT = base + self.scale * lora_term  # pyright: ignore[reportOperatorIssue,reportAssignmentType]
        return res

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


def inject_lora[ModelT](
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
        model.freeze()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]

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
            lora_module.freeze()  # pyright: ignore[reportAttributeAccessIssue]
            lora_module.unfreeze(keys=["lora_a", "lora_b"])  # pyright: ignore[reportAttributeAccessIssue]
            replacements.append((name, lora_module))

    if replacements:
        import mlx.utils as _mlx_utils

        with contextlib.suppress(AttributeError, TypeError):
            model.update_modules(_mlx_utils.tree_unflatten(replacements))  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        if isinstance(model, list):
            for name, lora_mod in replacements:
                if "." in name:
                    idx_str, child_key = name.split(".", 1)
                    model[int(idx_str)].update_modules({child_key: lora_mod})
                else:
                    model[int(name)] = lora_mod

    return model, len(replacements)


def save_adapter_weights(model: MlxCausalLMProtocol | _MlxModuleHierarchyProtocol, save_path: str | Path) -> None:
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

    trainable_dict: dict[str, MlxArrayProtocol] = dict(
        _mlx_utils.tree_flatten(model.trainable_parameters())  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    )
    mx.save_safetensors(str(path), trainable_dict)


def load_adapter_weights(model: MlxCausalLMProtocol | _MlxModuleHierarchyProtocol, load_path: str | Path) -> None:
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
    try:
        model.load_weights(str(path), strict=False)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    except AttributeError:
        with contextlib.suppress(AttributeError, TypeError):
            weights = mx.load(str(path))
            model.update(weights)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]


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
            model_obj: MlxCausalLMProtocol = provided_model  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
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

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
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
