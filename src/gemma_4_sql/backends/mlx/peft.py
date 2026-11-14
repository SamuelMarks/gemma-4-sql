"""MLX-specific PEFT / LoRA implementation with `MLXLoRALinear(mlx.nn.Module)`."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from pathlib import Path
from typing import Protocol, Self, overload, runtime_checkable

from gemma_4_sql._types_models import MlxLMLoadProtocol
from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale, matches_target_module
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    PEFTResultDict,
    TensorProtocol,
    extract_status_kwargs,
)

logger = logging.getLogger(__name__)

type _MlxListOrScalar = bool | int | float | complex | list[_MlxListOrScalar]


@runtime_checkable
class _MlxTensorProtocol(Protocol):
    """Structural protocol satisfied by `mlx.core.array`, `MlxArrayProtocol`, and `_TinyMlxArray`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    @property
    def T(self) -> Self:
        """Provide the transpose of the array."""
        ...

    def flatten(self) -> Self:
        """Return a 1D flattened array view."""
        ...

    def tolist(self) -> Sequence[float] | _MlxListOrScalar:
        """Convert the array into a Python list or scalar."""
        ...

    def __add__(self, other: Self | float, /) -> Self:
        """Add an array or scalar to this array."""
        ...

    def __radd__(self, other: Self | float, /) -> Self:
        """Right-add an array or scalar to this array."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply this array by an array or scalar."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply this array by an array or scalar."""
        ...

    def __matmul__(self, other: Self, /) -> Self:
        """Matrix-multiply two arrays."""
        ...


type MlxTensorLike = _MlxTensorProtocol


@runtime_checkable
class _MlxLinearProtocol(Protocol):
    """Protocol for an `mlx.nn.Linear` module exposing `.weight`."""

    @property
    def weight(self) -> MlxTensorLike:
        """Provide the 2D weight matrix."""
        ...


@runtime_checkable
class _MlxBiasedLinearProtocol(Protocol):
    """Protocol for an `mlx.nn.Linear` module also exposing `.bias`."""

    @property
    def bias(self) -> MlxTensorLike | None:
        """Provide the optional 1D bias vector."""
        ...


@runtime_checkable
class _MlxDropoutProtocol(Protocol):
    """Protocol for an `mlx.nn.Dropout` or `mlx.nn.Identity` callable module."""

    def __call__[ArrT: MlxTensorLike](self, x: ArrT) -> ArrT:
        """Apply dropout or identity pass-through to `x`.

        Args:
            x: Input array.

        Returns:
            Dropped-out or unchanged array of the same type.

        """
        ...


@runtime_checkable
class _MlxModuleHierarchyProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _MlxLinearProtocol]]:
        """Yield `(path, submodule)` pairs across the module tree."""
        ...


@runtime_checkable
class _MlxFreezableProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting `.freeze()`."""

    def freeze(self) -> Self | None:
        """Freeze module parameters.

        Returns:
            Frozen module or `None`.

        """
        ...


@runtime_checkable
class _MlxUpdateModulesProtocol(Protocol):
    """Protocol for an `mlx.nn.Module` supporting `.update_modules(modules)`."""

    def update_modules(self, modules: Mapping[str, MLXLoRALinear]) -> Self | None:
        """Replace child submodules in-place.

        Args:
            modules: Mapping from child attribute name to replacement `MLXLoRALinear`.

        Returns:
            Updated module or `None`.

        """
        ...


@runtime_checkable
class _MlxWeightUpdateProtocol(Protocol):
    """Protocol for a non-dict model supporting `.update(parameters)`."""

    def update(
        self,
        parameters: Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]],
    ) -> Self | None:
        """Update model weights from a dictionary.

        Args:
            parameters: Mapping of parameter names to MLX arrays.

        Returns:
            Optional self reference or `None`.

        """
        ...


type MlxPEFTTarget = (
    MlxCausalLMProtocol
    | _MlxModuleHierarchyProtocol
    | _MlxWeightUpdateProtocol
    | _MlxUpdateModulesProtocol
    | _MlxLinearProtocol
    | MLXLoRALinear
    | MutableMapping[str, "MlxPEFTTarget"]
    | MutableSequence["MlxPEFTTarget"]
    | Mapping[str, "MlxPEFTTarget"]
    | Sequence["MlxPEFTTarget"]
)


@runtime_checkable
class _MlxRandomNamespaceProtocol(Protocol):
    """Protocol for `mlx.core.random` exposing `uniform`."""

    def uniform(
        self,
        *,
        low: float,
        high: float,
        shape: tuple[int, int],
    ) -> _MlxTensorProtocol:
        """Sample uniform random float values in `[low, high)`.

        Args:
            low: Lower bound.
            high: Upper bound.
            shape: 2D `(in_features, r)` shape tuple.

        Returns:
            Sampled `_MlxTensorProtocol` matrix.

        """
        ...


@runtime_checkable
class _MlxPEFTCoreProtocol(Protocol):
    """Protocol for `mlx.core` operations used by MLX LoRA PEFT."""

    @property
    def random(self) -> _MlxRandomNamespaceProtocol:
        """Provide `mlx.core.random` namespace."""
        ...

    def array(self, val: MlxTensorLike, /) -> MlxTensorLike:
        """Convert `val` into an MLX array.

        Args:
            val: Input sequence or tensor.

        Returns:
            Converted `MlxTensorLike` array.

        """
        ...

    def zeros(self, shape: tuple[int] | tuple[int, int], /) -> _MlxTensorProtocol:
        """Construct a zero-filled 1D or 2D `_MlxTensorProtocol`.

        Args:
            shape: 1D or 2D shape tuple.

        Returns:
            Zero-initialized `_MlxTensorProtocol`.

        """
        ...

    def save_safetensors(self, file_path: str, arrays: Mapping[str, _MlxTensorProtocol], /) -> None:
        """Save named adapter weight arrays to a `.safetensors` file.

        Args:
            file_path: Destination `.safetensors` path string.
            arrays: Mapping from adapter parameter names to `_MlxTensorProtocol`.

        """
        ...

    def load(self, file_path: str, /) -> Mapping[str, MlxArrayProtocol]:
        """Load named adapter weight arrays from a `.safetensors` file.

        Args:
            file_path: Source `.safetensors` path string.

        Returns:
            Mapping from adapter parameter names to `MlxArrayProtocol`.

        """
        ...


@runtime_checkable
class _MlxPEFTNNProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `Dropout`, `Identity`, and `Linear`."""

    @property
    def Linear(self) -> type:
        """Provide `mlx.nn.Linear` class for `isinstance` checks."""
        ...

    def Dropout(self, *, p: float) -> _MlxDropoutProtocol:
        """Construct an MLX `Dropout` layer.

        Args:
            p: Dropout probability.

        Returns:
            Initialized `_MlxDropoutProtocol`.

        """
        ...

    def Identity(self) -> _MlxDropoutProtocol:
        """Construct an MLX `Identity` pass-through layer.

        Returns:
            Initialized `_MlxDropoutProtocol`.

        """
        ...


mx: _MlxPEFTCoreProtocol | None = optional_import("mlx.core", _MlxPEFTCoreProtocol)
nn: _MlxPEFTNNProtocol | None = optional_import("mlx.nn", _MlxPEFTNNProtocol)
load: MlxLMLoadProtocol | None = import_optional_symbol(
    "mlx_lm",
    "load",
    MlxLMLoadProtocol,
)


@runtime_checkable
class _MxArrayFactoryProtocol(Protocol):
    """Protocol for `mlx.core` exposing `array(...)`."""

    def array(self, val: MlxTensorLike, /) -> MlxTensorLike:
        """Convert `val` to an MLX array."""
        ...


@runtime_checkable
class _MlxModuleNamespaceCheckProtocol(Protocol):
    """Protocol verifying `nn` is the real `mlx.nn` module exposing `Module`."""

    @property
    def Module(self) -> type:
        """Provide `mlx.nn.Module` class."""
        ...


_NATIVE_LORA_CLS_CACHE: list[type[MLXLoRALinear]] = []


def _get_native_mlx_lora_cls() -> type[MLXLoRALinear] | None:
    """Return a cached `MLXLoRALinear` subclass inheriting `mlx.nn.Module` when `mlx.nn` is installed.

    Returns:
        Subclass of `(MLXLoRALinear, mlx.nn.Module)` or `None` if `mlx.nn` is unavailable.

    """
    if _NATIVE_LORA_CLS_CACHE:
        return _NATIVE_LORA_CLS_CACHE[0]
    try:
        from mlx import nn as _real_nn
    except ImportError:
        return None

    class _NativeMLXLoRALinear(MLXLoRALinear, _real_nn.Module):
        """Concrete `MLXLoRALinear` bound to `mlx.nn.Module` for parameter & gradient traversal."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: float = 16.0,
            lora_dropout: float = 0.05,
            bias: bool = False,
        ) -> None:
            """Initialize `mlx.nn.Module` dictionary storage and `MLXLoRALinear` weights."""
            _real_nn.Module.__init__(self)
            self._init_lora_fields(
                in_features=in_features,
                out_features=out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
            )
            self._freeze_base_weights(has_bias=bias)

        def _freeze_base_weights(self, *, has_bias: bool) -> None:
            """Freeze `weight` and optional `bias` keys on the underlying `mlx.nn.Module`."""
            _real_nn.Module.freeze(
                self,
                recurse=False,
                keys=["weight", "bias"] if has_bias else ["weight"],
            )

        def parameters(self) -> dict[str, _MlxTensorProtocol]:
            """Delegate parameter tree extraction to `mlx.nn.Module.parameters`.

            Returns:
                Dictionary mapping parameter names to MLX tensor instances.

            """
            params_fn: Callable[[MLXLoRALinear], Mapping[str, MlxPEFTTarget | _MlxTensorProtocol]] = vars(
                _real_nn.Module
            )["parameters"]
            raw_p = params_fn(self)
            return {str(k): v for k, v in raw_p.items() if isinstance(v, _MlxTensorProtocol)}

        def trainable_parameters(self) -> dict[str, _MlxTensorProtocol]:
            """Delegate trainable parameter tree extraction to `mlx.nn.Module.trainable_parameters`.

            Returns:
                Dictionary mapping trainable parameter names to MLX tensor instances.

            """
            train_params_fn: Callable[[MLXLoRALinear], Mapping[str, MlxPEFTTarget | _MlxTensorProtocol]] = vars(
                _real_nn.Module
            )["trainable_parameters"]
            raw_tp = train_params_fn(self)
            return {str(k): v for k, v in raw_tp.items() if isinstance(v, _MlxTensorProtocol)}

    _NATIVE_LORA_CLS_CACHE.append(_NativeMLXLoRALinear)
    return _NativeMLXLoRALinear


class MLXLoRALinear:
    """Low-rank adaptation (LoRA) linear module subclassing `mlx.nn.Module` when MLX is available.

    Decomposes weight updates into low-rank matrices:
        `y = x W^T + bias + (alpha / r) * (Dropout(x) @ A) @ B`
    where `W` and optional `bias` are frozen on the `mlx.nn.Module` parameter tree while
    `lora_a` and `lora_b` remain trainable for `model.trainable_parameters()` and `nn.value_and_grad`.

    Attributes:
        weight: Frozen base weight matrix of shape `(out_features, in_features)`.
        lora_a: Trainable down-projection adapter of shape `(in_features, r)`.
        lora_b: Trainable up-projection adapter of shape `(r, out_features)`.
        bias: Optional frozen bias vector of shape `(out_features,)`.
        dropout: Dropout or identity module applied before `lora_a`.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        r: LoRA rank.
        lora_alpha: LoRA scaling numerator.
        scale: Computed `lora_alpha / r` multiplier.

    """

    weight: _MlxTensorProtocol
    lora_a: _MlxTensorProtocol
    lora_b: _MlxTensorProtocol
    bias: _MlxTensorProtocol | None
    dropout: _MlxDropoutProtocol
    in_features: int
    out_features: int
    r: int
    lora_alpha: float
    scale: float

    def __new__(
        cls,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = False,
    ) -> Self | MLXLoRALinear:
        """Instantiate an `mlx.nn.Module`-backed subclass when real `mlx.nn` is active."""
        _ = (in_features, out_features, r, lora_alpha, lora_dropout, bias)
        native_cls = (
            _get_native_mlx_lora_cls()
            if cls is MLXLoRALinear and isinstance(nn, _MlxModuleNamespaceCheckProtocol)
            else None
        )
        if native_cls is not None:
            return super(MLXLoRALinear, native_cls).__new__(native_cls)
        return super().__new__(cls)

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
        self._init_lora_fields(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )

    def _init_lora_fields(
        self,
        *,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        bias: bool,
    ) -> None:
        """Populate LoRA weights, dropout, and parameter freeze mask.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            r: Rank of the low-rank adapters.
            lora_alpha: LoRA scaling parameter.
            lora_dropout: Dropout probability applied before the LoRA adapter.
            bias: Whether to allocate an additive bias vector.

        Raises:
            DependencyMissingError: If MLX dependencies (`mlx.core` or `mlx.nn`) are missing.

        """
        if nn is None or mx is None:
            raise DependencyMissingError("MLX dependencies are missing.")
        self.scale = compute_lora_scale(lora_alpha, r)
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

    def _freeze_base_weights(self, *, has_bias: bool) -> None:
        """Freeze base `weight` and optional `bias` parameters when backed by `mlx.nn.Module`."""
        _ = has_bias

    @property
    def W(self) -> _MlxTensorProtocol:
        """Frozen base weight matrix `W` with shape `(out_features, in_features)`."""
        return self.weight

    @property
    def A(self) -> _MlxTensorProtocol:
        """Trainable down-projection adapter `A` with shape `(in_features, r)`."""
        return self.lora_a

    @property
    def B(self) -> _MlxTensorProtocol:
        """Trainable up-projection adapter `B` with shape `(r, out_features)`."""
        return self.lora_b

    def parameters(self) -> dict[str, _MlxTensorProtocol]:
        """Return full parameter dictionary (`weight`, optional `bias`, `lora_a`, `lora_b`).

        Returns:
            Dictionary of all layer parameter tensors.

        """
        params: dict[str, _MlxTensorProtocol] = {
            "weight": self.weight,
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
        }
        if self.bias is not None:
            params["bias"] = self.bias
        return params

    def trainable_parameters(self) -> dict[str, _MlxTensorProtocol]:
        """Return trainable LoRA adapter parameter dictionary (`lora_a`, `lora_b`).

        Returns:
            Dictionary of trainable adapter parameter tensors.

        """
        return {"lora_a": self.lora_a, "lora_b": self.lora_b}

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
            An `MLXLoRALinear` layer initialized with the linear module's weights frozen.

        """
        assert mx is not None
        bias_val: MlxTensorLike | None = None
        if isinstance(linear, _BiasMappingLinearProtocol) and "bias" in linear:
            cand_bias = linear["bias"]
            bias_val = cand_bias if isinstance(cand_bias, _MlxTensorProtocol) else None
        elif isinstance(linear, _MlxBiasedLinearProtocol):
            bias_val = linear.bias
        has_bias = bias_val is not None
        weight = linear.weight
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        lora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=has_bias,
        )
        lora_layer.weight = mx.array(weight)
        if bias_val is not None:
            lora_layer.bias = mx.array(bias_val)
        lora_layer._freeze_base_weights(has_bias=has_bias)
        return lora_layer

    @overload
    def __call__[ArrT: MlxArrayProtocol](self, x: ArrT) -> ArrT:
        """Execute LoRA forward pass preserving concrete `MlxArrayProtocol` (`mx.array`) type."""

    @overload
    def __call__[ArrT: _MlxTensorProtocol](self, x: ArrT) -> ArrT:
        """Execute LoRA forward pass preserving concrete `_MlxTensorProtocol` type."""

    def __call__(self, x: _MlxTensorProtocol) -> _MlxTensorProtocol:
        """Execute the LoRA-augmented linear forward pass.

        Computes:
            `y = x @ W^T + bias + scale * (dropout(x) @ A) @ B`

        Args:
            x: Input tensor of shape `(..., in_features)`.

        Returns:
            Output tensor of shape `(..., out_features)`.

        """
        drop_x = self.dropout(x)
        base = x @ self.weight.T
        if self.bias is not None:
            base += self.bias
        lora_term = (drop_x @ self.lora_a) @ self.lora_b
        return base + self.scale * lora_term

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
        adapters = {"lora_a": self.lora_a, "lora_b": self.lora_b}
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


@runtime_checkable
class _BiasMappingLinearProtocol(Protocol):
    """Non-generic protocol for dictionary-backed `mlx.nn.Linear` exposing `'bias'` key lookup."""

    def __contains__(self, key: str, /) -> bool:
        """Check whether `key` is stored on the linear module."""
        ...

    def __getitem__(self, key: str, /) -> _MlxTensorProtocol | None:
        """Retrieve parameter tensor at `key`."""
        ...


@runtime_checkable
class _SubmoduleDictProtocol(Protocol):
    """Non-generic mutable string-keyed submodule dictionary protocol."""

    def keys(self) -> Iterable[str]:
        """Yield string keys (distinguishing mappings from sequences)."""
        ...

    def __getitem__(self, key: str, /) -> MlxPEFTTarget:
        """Retrieve child submodule at `key`."""
        ...

    def __setitem__(self, key: str, value: MlxPEFTTarget, /) -> None:
        """Assign child submodule `value` at `key`."""
        ...


@runtime_checkable
class _SubmoduleListProtocol(Protocol):
    """Non-generic mutable integer-indexed submodule sequence protocol."""

    def insert(self, index: int, value: MlxPEFTTarget, /) -> None:
        """Insert `value` at `index` (distinguishing mutable sequences from mappings)."""
        ...

    def __getitem__(self, index: int, /) -> MlxPEFTTarget:
        """Retrieve child submodule at `index`."""
        ...

    def __setitem__(self, index: int, value: MlxPEFTTarget, /) -> None:
        """Assign child submodule `value` at `index`."""
        ...


def _set_mlx_submodule(
    root: MlxPEFTTarget,
    dotted_path: str,
    lora_mod: MLXLoRALinear,
) -> None:
    """Replace a submodule at `dotted_path` inside `root` using dictionary/sequence indexing.

    Args:
        root: Root MLX model (`dict` or `list` container).
        dotted_path: Dot-delimited submodule path from `named_modules()`.
        lora_mod: Replacement `MLXLoRALinear` instance.

    """
    parts = dotted_path.split(".")
    curr: MlxPEFTTarget = root
    for part in parts[:-1]:
        if isinstance(curr, _SubmoduleDictProtocol):
            curr = curr[part]
        elif isinstance(curr, _SubmoduleListProtocol):
            curr = curr[int(part)]
    last = parts[-1]
    if isinstance(curr, _SubmoduleDictProtocol):
        curr[last] = lora_mod
    elif isinstance(curr, _SubmoduleListProtocol):
        curr[int(last)] = lora_mod
    elif isinstance(curr, _MlxUpdateModulesProtocol):
        curr.update_modules({last: lora_mod})


def _iter_lora_layers(
    node: MlxPEFTTarget,
    prefix: str = "",
) -> Iterator[tuple[str, MLXLoRALinear]]:
    """Yield `(dotted_prefix, lora_layer)` pairs across a nested MLX model hierarchy.

    Args:
        node: Root or child MLX module, mapping, sequence, or `MLXLoRALinear`.
        prefix: Current dot-delimited path prefix.

    Yields:
        Tuples of `(dotted_prefix, lora_layer)`.

    """
    if isinstance(node, MLXLoRALinear):
        yield (prefix, node)
    elif isinstance(node, Mapping):
        for k, v in node.items():
            child_key = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_lora_layers(v, child_key)
    elif isinstance(node, (list, tuple)):
        for idx, item in enumerate(node):
            yield from _iter_lora_layers(item, f"{prefix}.{idx}")


def inject_lora[ModelT: MlxPEFTTarget](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[ModelT, int]:
    """Inject `MLXLoRALinear` adapters into designated projection modules of an MLX model.

    Freezes the base model parameters first so only injected `lora_a` and `lora_b` tensors
    remain trainable in `model.trainable_parameters()`.

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

    if isinstance(model, _MlxFreezableProtocol) and callable(model.freeze):
        model.freeze()

    if not target_modules or not isinstance(model, _MlxModuleHierarchyProtocol):
        return model, 0

    replacements: list[tuple[str, MLXLoRALinear]] = []
    for name, submodule in model.named_modules():
        if (
            isinstance(submodule, nn.Linear)
            and not isinstance(submodule, MLXLoRALinear)
            and matches_target_module(name, target_modules)
        ):
            lora_module = MLXLoRALinear.from_linear(
                submodule,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            replacements.append((name, lora_module))

    for name, lora_mod in replacements:
        _set_mlx_submodule(model, name, lora_mod)

    return model, len(replacements)


def save_adapter_weights(
    model: MlxPEFTTarget,
    save_path: str | Path,
) -> None:
    """Save all LoRA adapter weights from an MLX model to a safetensors file.

    Args:
        model: MLX model containing LoRA adapters.
        save_path: File path or directory path to save the adapter weights.

    Raises:
        DependencyMissingError: If MLX is missing.

    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    raw_path = Path(save_path)
    path = (raw_path / "adapter.safetensors") if (raw_path.is_dir() or raw_path.suffix != ".safetensors") else raw_path
    path.parent.mkdir(parents=True, exist_ok=True)

    trainable_dict: dict[str, _MlxTensorProtocol] = {}
    for prefix, lora_mod in _iter_lora_layers(model):
        key_a = f"{prefix}.lora_a" if prefix else "lora_a"
        key_b = f"{prefix}.lora_b" if prefix else "lora_b"
        trainable_dict[key_a] = lora_mod.lora_a
        trainable_dict[key_b] = lora_mod.lora_b
    mx.save_safetensors(str(path), trainable_dict)


def load_adapter_weights(
    model: MlxPEFTTarget,
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
    weights = mx.load(str(path))
    found_lora = False
    for prefix, lora_mod in _iter_lora_layers(model):
        found_lora = True
        key_a = f"{prefix}.lora_a" if prefix else "lora_a"
        key_b = f"{prefix}.lora_b" if prefix else "lora_b"
        lora_mod.lora_a = weights[key_a]
        lora_mod.lora_b = weights[key_b]
    if (
        not found_lora
        and not isinstance(model, Mapping)
        and isinstance(model, _MlxWeightUpdateProtocol)
        and callable(model.update)
    ):
        model.update(weights)


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: JSONValue | MlxCausalLMProtocol | _MlxModuleHierarchyProtocol,
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
        """Load or reuse MLX model and inject LoRA linear layers.

        Returns:
            Number of injected `MLXLoRALinear` adapter layers.

        """
        assert load is not None
        provided_model = kwargs.get("model")
        if (
            not isinstance(provided_model, (str, int, float, bool, list, tuple))
            and provided_model is not None
            and isinstance(provided_model, (MlxCausalLMProtocol, _MlxModuleHierarchyProtocol))
        ):
            model_obj: MlxCausalLMProtocol | _MlxModuleHierarchyProtocol = provided_model
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
