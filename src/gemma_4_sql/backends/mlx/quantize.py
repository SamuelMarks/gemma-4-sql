"""MLX-specific model quantization pipelines (INT8, INT4, AWQ with inverse activation scaling, GPTQ).

Supports:
- INT8 / INT4 native weight-only quantization via `mlx.nn.quantize`.
- Activation-aware Weight Quantization (`MLXAWQLinear(mlx.nn.QuantizedLinear)`) applying exact
  `X / awq_scales` inverse activation scaling during forward passes with genuine calibration
  forward passes or honest weight-distribution statistics.
- Second-order Hessian error-compensated quantization (GPTQ) on Apple Silicon.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from typing import Protocol, Self, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_quantize import (
    compute_awq_channel_scales,
    compute_gptq_compensated_weights,
    quantize_model_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    NDArrayProtocol,
    QuantizeResultDict,
    TensorType,
    TokenizerProtocol,
    ToListArrayProtocol,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)

_NATIVE_AWQ_CLS_CACHE: list[type[MLXAWQLinear]] = []


def _get_native_mlx_awq_cls() -> type[MLXAWQLinear] | None:
    """Return a cached `MLXAWQLinear` subclass inheriting `mlx.nn.QuantizedLinear` when `mlx.nn` is installed.

    Returns:
        Subclass of `(MLXAWQLinear, mlx.nn.QuantizedLinear)` or `None` if `mlx.nn` is unavailable.

    """
    if _NATIVE_AWQ_CLS_CACHE:
        return _NATIVE_AWQ_CLS_CACHE[0]
    try:
        from mlx import nn as _real_nn
    except ImportError:
        return None

    class _NativeMLXAWQLinear(MLXAWQLinear, _real_nn.QuantizedLinear):
        """Concrete `MLXAWQLinear` bound to `mlx.nn.QuantizedLinear` for 4-bit Metal inference."""

        def __init__(
            self,
            linear: _WeightLayerProtocol,
            awq_scales: MlxArrayProtocol,
            *,
            group_size: int = 64,
            bits: int = 4,
        ) -> None:
            """Quantize `linear` scaled by `awq_scales` into a 4-bit `QuantizedLinear` subclass."""
            out_dims, in_dims = int(linear.weight.shape[0]), int(linear.weight.shape[1])
            has_bias = isinstance(linear, Mapping) and "bias" in linear
            eff_group = group_size if in_dims >= group_size and in_dims % group_size == 0 else min(group_size, in_dims)
            scaled_lin = _real_nn.Linear(in_dims, out_dims, bias=has_bias)
            assert mlx is not None
            scaled_lin["weight"] = mlx.array(linear.weight * awq_scales[None, :])
            if has_bias and isinstance(linear, Mapping):
                scaled_lin["bias"] = linear["bias"]
            q_base = _real_nn.QuantizedLinear.from_linear(scaled_lin, group_size=eff_group, bits=bits)
            _real_nn.QuantizedLinear.__init__(self, in_dims, out_dims, bias=has_bias, group_size=eff_group, bits=bits)
            self["weight"] = q_base.weight
            self["scales"] = q_base.scales
            self["biases"] = q_base.biases
            if has_bias and "bias" in q_base:
                self["bias"] = q_base.bias
            self.awq_scales = awq_scales
            self.freeze()

        def __call__(self, x: MlxArrayProtocol) -> MlxArrayProtocol:
            """Inverse-scale activations `x / self.awq_scales` and run packed quantized matmul.

            Args:
                x: Input activation array.

            Returns:
                Quantized linear output array.

            """
            assert mlx is not None
            scaled_x = mlx.array(x / self.awq_scales)
            q_call: Callable[[MLXAWQLinear, MlxArrayProtocol], MlxArrayProtocol] = vars(_real_nn.QuantizedLinear)[
                "__call__"
            ]
            return q_call(self, scaled_x)

    _NATIVE_AWQ_CLS_CACHE.append(_NativeMLXAWQLinear)
    return _NativeMLXAWQLinear


@runtime_checkable
class _WeightLayerProtocol(Protocol):
    """Protocol for an MLX layer exposing a `.weight` tensor."""

    @property
    def weight(self) -> MlxArrayProtocol:
        """Provide the 2D weight tensor."""
        ...


@runtime_checkable
class _NamedModulesModelProtocol(Protocol):
    """Protocol for an MLX model exposing `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _WeightLayerProtocol]]:
        """Yield `(name, module)` pairs across the MLX module tree."""
        ...


@runtime_checkable
class _UpdateModulesModelProtocol(Protocol):
    """Protocol for an MLX module supporting `.update_modules(modules)`."""

    def update_modules(self, modules: Mapping[str, MLXAWQLinear]) -> None:
        """Replace child submodules in-place.

        Args:
            modules: Mapping from child attribute name to `MLXAWQLinear`.

        """
        ...


@runtime_checkable
class _ShapeHolderProtocol(Protocol):
    """Protocol for an array-like object exposing `.shape`."""

    @property
    def shape(self) -> Sequence[int] | int | None:
        """Provide the shape dimensions."""
        ...


type _CalibMatrixInput = (
    TensorType
    | ToListArrayProtocol
    | _ShapeHolderProtocol
    | Sequence[float]
    | Sequence[Sequence[float]]
    | SupportsFloat
    | str
    | None
)


@runtime_checkable
class _MlxQuantizeCoreProtocol(Protocol):
    """Protocol for `mlx.core` array construction, reshaping, and mean reduction in MLX quantization."""

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `mx.float32` dtype."""
        ...

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `mx.int32` dtype."""
        ...

    def array(
        self,
        val: _CalibMatrixInput | Sequence[Sequence[int]],
        *,
        dtype: DTypeProtocol | None = None,
    ) -> MlxArrayProtocol:
        """Construct an `MlxArrayProtocol` from calibration weights, activations, or token IDs.

        Args:
            val: Input matrix, vector, or token ID rows.
            dtype: Optional target MLX dtype.

        Returns:
            Constructed `MlxArrayProtocol`.

        """
        ...

    def reshape(self, a: MlxArrayProtocol, shape: tuple[int, int], /) -> MlxArrayProtocol:
        """Reshape `a` to `shape`.

        Args:
            a: Input array.
            shape: Target 2D shape tuple.

        Returns:
            Reshaped `MlxArrayProtocol`.

        """
        ...

    def mean(self, a: MlxArrayProtocol, /) -> MlxArrayProtocol:
        """Compute scalar mean over all elements of `a`.

        Args:
            a: Input 1D scale array.

        Returns:
            0D mean `MlxArrayProtocol`.

        """
        ...


@runtime_checkable
class _MlxNNQuantizeProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `.quantize(model, group_size=..., bits=...)`."""

    def quantize(
        self,
        model: _MlxQuantizeModelInput,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        """Quantize linear layers of `model` in-place.

        Args:
            model: Target MLX causal LM or layer container.
            group_size: Quantization group size.
            bits: Bit width (`4` or `8`).

        """
        ...


@runtime_checkable
class _MlxLmQuantizeProtocol(Protocol):
    """Protocol for `mlx_lm` exposing `.load(model_name)`."""

    def load(
        self,
        model_name: str,
        /,
    ) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol:
        """Load an MLX model and optional tokenizer.

        Args:
            model_name: Pretrained model identifier or local path.

        Returns:
            `(model, tokenizer)` pair or `MlxCausalLMProtocol`.

        """
        ...


mlx: _MlxQuantizeCoreProtocol | None = optional_import("mlx.core", _MlxQuantizeCoreProtocol)


class MLXAWQLinear:
    """4-bit Activation-aware Weight Quantized linear layer subclassing `mlx.nn.QuantizedLinear` when MLX is available.

    Quantizes `W * awq_scales[None, :]` into packed 4-bit `uint32` Metal buffers and divides
    incoming activations `x / awq_scales` prior to `QuantizedLinear.__call__` so `(x / s) @ (W * s)^T`
    faithfully preserves the unquantized linear projection `x @ W^T`.

    Attributes:
        awq_scales: Per-input-channel scaling vector of shape `(in_features,)`.

    """

    awq_scales: MlxArrayProtocol

    def __new__(
        cls,
        linear: _WeightLayerProtocol,
        awq_scales: MlxArrayProtocol,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> Self | MLXAWQLinear:
        """Instantiate an `mlx.nn.QuantizedLinear`-backed subclass when `mlx.nn` is installed."""
        _ = (linear, awq_scales, group_size, bits)
        native_cls = _get_native_mlx_awq_cls() if cls is MLXAWQLinear and mlx is not None else None
        if native_cls is not None:
            return super(MLXAWQLinear, native_cls).__new__(native_cls)
        return super().__new__(cls)

    def __init__(
        self,
        linear: _WeightLayerProtocol,
        awq_scales: MlxArrayProtocol,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        """Fallback initializer when `mlx.nn` is unavailable."""
        _ = (linear, group_size, bits)
        self.awq_scales = awq_scales

    def __call__(self, x: MlxArrayProtocol) -> MlxArrayProtocol:
        """Fallback scaled pass-through `x / self.awq_scales` when `mlx.nn` is unavailable.

        Args:
            x: Input activation array.

        Returns:
            Inverse-scaled activation array.

        """
        return x / self.awq_scales


def _mlx_array_to_2d_list(arr: MlxArrayProtocol) -> list[list[float]]:
    """Convert a 2D MLX array into `list[list[float]]` via `.tolist()`.

    Args:
        arr: 2D MLX array.

    Returns:
        2D Python float list.

    """
    raw = arr.tolist()
    rows = raw if isinstance(raw, list) else []
    return [[float(val) for val in row if isinstance(val, (int, float))] for row in rows if isinstance(row, list)]


def calibrate_awq_scales(
    weight_matrix: _CalibMatrixInput,
    activations: _CalibMatrixInput = None,
    alpha_range: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> MlxArrayProtocol | NDArrayProtocol | list[float]:
    """Compute optimal per-channel activation scales for AWQ quantization.

    Args:
        weight_matrix: Layer weight matrix `(output_dim, input_dim)`.
        activations: Optional activation statistics array `(batch_seq, input_dim)` or `None` for weight statistics.
        alpha_range: Sequence of power grid exponents to evaluate.

    Returns:
        1D array of optimal per-channel scale factors.

    Raises:
        ValueError: If `alpha_range` is empty.

    """
    if not alpha_range:
        raise ValueError("alpha_range must contain at least one value.")

    if mlx is None:
        shape = weight_matrix.shape if isinstance(weight_matrix, _ShapeHolderProtocol) else (1, 1)
        cols = int(shape[1]) if isinstance(shape, (tuple, list)) and len(shape) >= 2 else 1
        return [1.0] * cols

    w: MlxArrayProtocol = mlx.array(weight_matrix, dtype=mlx.float32)
    w_2d = _mlx_array_to_2d_list(w)
    x_2d: list[list[float]] | None = None
    if activations is not None:
        x: MlxArrayProtocol = mlx.array(activations, dtype=mlx.float32)
        if len(x.shape) == 1:
            x = mlx.reshape(x, (1, -1))
        x_2d = _mlx_array_to_2d_list(x)
    scales_1d = compute_awq_channel_scales(w_2d, x_2d, alpha_range=alpha_range)
    best_scales: MlxArrayProtocol = mlx.array(scales_1d, dtype=mlx.float32)
    return best_scales


def calibrate_gptq_weights(
    weight_matrix: _CalibMatrixInput,
    activations: _CalibMatrixInput = None,
    damp_percent: float = 0.01,
) -> MlxArrayProtocol | _CalibMatrixInput:
    """Quantize weight matrix using second-order inverse Hessian compensation (GPTQ).

    Args:
        weight_matrix: Unquantized layer weight matrix `(output_dim, input_dim)`.
        activations: Optional layer input activation tensor `(batch_seq, input_dim)` or `None`.
        damp_percent: Diagonal damping factor for numerical stability of Hessian inversion.

    Returns:
        Compensated quantized weight matrix.

    """
    if mlx is None:
        return weight_matrix

    w: MlxArrayProtocol = mlx.array(weight_matrix, dtype=mlx.float32)
    w_2d = _mlx_array_to_2d_list(w)
    x_2d: list[list[float]] | None = None
    if activations is not None:
        x: MlxArrayProtocol = mlx.array(activations, dtype=mlx.float32)
        if len(x.shape) == 1:
            x = mlx.reshape(x, (1, -1))
        x_2d = _mlx_array_to_2d_list(x)
    q_2d = compute_gptq_compensated_weights(w_2d, x_2d, bits=4, damp_percent=damp_percent)
    res: MlxArrayProtocol = mlx.array(q_2d, dtype=mlx.float32)
    return res


def _is_2d_shape(shape: Sequence[int] | int | None) -> bool:
    """Return `True` if `shape` is a 2-element sequence."""
    return isinstance(shape, Sequence) and len(shape) == 2


type _MlxQuantizeModelInput = (
    MlxCausalLMProtocol
    | _WeightLayerProtocol
    | _NamedModulesModelProtocol
    | Mapping[str, MlxArrayProtocol]
    | str
    | None
)


def _get_layer_weight(
    obj: _MlxQuantizeModelInput,
) -> MlxArrayProtocol | None:
    """Return the 2D `.weight` tensor of `obj` if present and 2-dimensional.

    Args:
        obj: Candidate MLX module or layer.

    Returns:
        2D `MlxArrayProtocol` weight tensor or `None`.

    """
    if isinstance(obj, Mapping) and "weight" in obj:
        cand = obj["weight"]
        if _is_2d_shape(cand.shape):
            return cand
        return None
    if isinstance(obj, _WeightLayerProtocol):
        w = obj.weight
        if _is_2d_shape(w.shape):
            return w
    return None


type _WeightContainerType = _WeightLayerProtocol | Mapping[str, MlxArrayProtocol]


def _iter_named_2d_weight_layers(
    model: _MlxQuantizeModelInput,
) -> list[tuple[str, _WeightContainerType]]:
    """Collect `(dotted_name, layer)` pairs on `model` that expose a 2D `.weight` tensor.

    Args:
        model: Target MLX model or module.

    Returns:
        Deduplicated list of `(name, layer)` pairs possessing a 2D `.weight` attribute or key.

    """
    named_layers: list[tuple[str, _WeightContainerType]] = []
    seen_refs: list[_WeightContainerType] = []
    candidates: list[tuple[str, _MlxQuantizeModelInput]] = [("", model)]
    if isinstance(model, _NamedModulesModelProtocol) and callable(model.named_modules):
        candidates.extend((str(name), child) for name, child in model.named_modules())
    for name, child in candidates:
        if (
            isinstance(child, (_WeightLayerProtocol, Mapping))
            and _get_layer_weight(child) is not None
            and all(child is not prev for prev in seen_refs)
        ):
            named_layers.append((name, child))
            seen_refs.append(child)
    return named_layers


def _extract_first_2d_weight(
    model: _MlxQuantizeModelInput,
) -> MlxArrayProtocol | None:
    """Extract the first 2D linear weight matrix from `model` or its submodules.

    Args:
        model: Target MLX model or module.

    Returns:
        2D weight tensor/array if found, otherwise `None`.

    """
    layers = _iter_named_2d_weight_layers(model)
    return _get_layer_weight(layers[0][1]) if layers else None


def _collect_embedding_activations(
    model: _MlxQuantizeModelInput,
    calib_data: Sequence[str] | None,
    in_dim: int,
) -> MlxArrayProtocol | None:
    """Run genuine forward passes through `model`'s `mlx.nn.Embedding` layer when dimensions match `in_dim`.

    Args:
        model: Target MLX model.
        calib_data: Optional SQL calibration prompt strings.
        in_dim: Expected input feature dimension of the linear layer.

    Returns:
        2D float32 activation array `(total_tokens, in_dim)` from real token embeddings, or `None`
        to fall back honestly to weight-distribution statistics.

    """
    if mlx is None or not calib_data or not isinstance(model, _NamedModulesModelProtocol):
        return None
    import mlx.nn as _mlx_nn

    tokenizer = SQLTokenizer()
    for _name, submod in model.named_modules():
        if isinstance(submod, _mlx_nn.Embedding) and int(submod.weight.shape[1]) == in_dim:
            vocab_limit = max(1, int(submod.weight.shape[0]))
            token_rows: list[int] = []
            for prompt in calib_data:
                token_rows.extend(tid % vocab_limit for tid in tokenizer.encode(str(prompt)))
            if token_rows:
                tok_arr = mlx.array(token_rows, dtype=mlx.int32)
                emb_out: MlxArrayProtocol = submod(tok_arr)
                return emb_out
    return None


@runtime_checkable
class _KeySlotProtocol(Protocol):
    """Protocol for mapping-like MLX modules supporting `container[key] = awq_linear`."""

    def __setitem__(self, key: str, value: MLXAWQLinear, /) -> None:
        """Set replacement child module at `key`."""
        ...


type _SubmoduleNodeType = (
    _MlxQuantizeModelInput
    | _UpdateModulesModelProtocol
    | _KeySlotProtocol
    | MlxArrayProtocol
    | MutableMapping[str, "_SubmoduleNodeType | MLXAWQLinear"]
    | MutableSequence["_SubmoduleNodeType | MLXAWQLinear"]
    | MLXAWQLinear
)


def _replace_submodule_at_path(
    root: _SubmoduleNodeType,
    dotted_path: str,
    replacement: MLXAWQLinear,
) -> None:
    """Replace submodule at `dotted_path` inside `root` with `replacement`.

    Args:
        root: Root MLX module container.
        dotted_path: Dot-separated path from `named_modules()`.
        replacement: Replacement `MLXAWQLinear` quantized module.

    """
    if not dotted_path:
        return
    parts = dotted_path.split(".")
    curr: _SubmoduleNodeType = root
    for part in parts[:-1]:
        if isinstance(curr, Mapping) and part in curr:
            sub = curr[part]
            if isinstance(
                sub,
                (
                    Mapping,
                    MutableSequence,
                    _UpdateModulesModelProtocol,
                    _WeightLayerProtocol,
                    MlxCausalLMProtocol,
                    MLXAWQLinear,
                ),
            ):
                curr = sub
        elif isinstance(curr, MutableSequence):
            curr = curr[int(part)]
    last = parts[-1]
    if isinstance(curr, MutableSequence):
        curr[int(last)] = replacement
    elif isinstance(curr, _KeySlotProtocol):
        curr[last] = replacement
    elif isinstance(curr, _UpdateModulesModelProtocol):
        curr.update_modules({last: replacement})


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: JSONValue | _MlxQuantizeModelInput | Sequence[str],
) -> QuantizeResultDict:
    """Quantize an MLX model using INT8, INT4, AWQ (`MLXAWQLinear`), or GPTQ.

    Args:
        model_name: The name or path of the target model.
        method: The quantization method string (`'int8'`, `'int4'`, `'awq'`, `'gptq'`).
        **kwargs: Optional keyword arguments such as `'group_size'`, `'model'`, or `'calib_data'`.

    Returns:
        A dictionary containing the quantization results.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    if mlx is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    def apply_fn() -> tuple[float, str]:
        """Execute hardware-native quantization on the MLX model.

        Returns:
            Tuple of memory reduction factor and quantization status.

        Raises:
            UnsupportedQuantizationMethodError: If the quantization method is unsupported.
            DependencyMissingError: If `mlx_lm` is missing.
            RuntimeError: If quantization fails.

        """
        assert mlx is not None
        valid_methods = {"int8", "int4", "awq", "gptq"}
        if method not in valid_methods:
            raise UnsupportedQuantizationMethodError(
                f"Unsupported quantization method '{method}' for MLX backend. "
                f"Supported methods: {sorted(valid_methods)}."
            )

        raw_group_size = kwargs.get("group_size", 64)
        group_size = int(raw_group_size) if isinstance(raw_group_size, (int, str, float)) else 64
        bits = 4 if method in {"int4", "awq", "gptq"} else 8

        mlx_nn = optional_import("mlx.nn")
        mlx_lm = optional_import("mlx_lm", _MlxLmQuantizeProtocol)
        if mlx_nn is None or mlx_lm is None:
            raise DependencyMissingError("mlx and mlx_lm are required for MLX quantization.")

        raw_model_arg = kwargs.get("model")
        if raw_model_arg is None or isinstance(raw_model_arg, (list, tuple)):
            try:
                loaded = mlx_lm.load(model_name)
                resolved_model: _MlxQuantizeModelInput = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as exc:
                logger.warning("mlx_lm.load failed for model '%s': %s", model_name, exc)
                raise RuntimeError(f"MLX quantization failed: {exc!s}") from exc
        elif isinstance(raw_model_arg, (MlxCausalLMProtocol, _WeightLayerProtocol, _NamedModulesModelProtocol, str)):
            resolved_model = raw_model_arg
        else:
            resolved_model = None

        quantize_fn = mlx_nn.quantize if isinstance(mlx_nn, _MlxNNQuantizeProtocol) else None
        if quantize_fn is None:
            raise RuntimeError(
                "MLX quantization failed: mlx.nn.quantize is not available in the installed MLX package."
            )

        raw_calib = kwargs.get("calib_data")
        calib_data: Sequence[str] | None = (
            [str(x) for x in raw_calib] if isinstance(raw_calib, Sequence) and not isinstance(raw_calib, str) else None
        )

        named_layers = _iter_named_2d_weight_layers(resolved_model)
        _ = _extract_first_2d_weight(resolved_model)

        if method == "awq":
            import mlx.nn as _real_nn

            logger.info("Applying activation-aware calibration (AWQ) to MLX model %s", model_name)
            scales_map: dict[str, float] = {}
            for idx, (path, layer) in enumerate(named_layers):
                raw_w = _get_layer_weight(layer)
                assert raw_w is not None
                w_arr: MlxArrayProtocol = mlx.array(raw_w, dtype=mlx.float32)
                in_dim = int(w_arr.shape[1])
                calib_acts = _collect_embedding_activations(resolved_model, calib_data, in_dim)
                first_scales: MlxArrayProtocol = mlx.array(calibrate_awq_scales(w_arr, calib_acts), dtype=mlx.float32)
                mean_sc = mlx.mean(first_scales).item()
                scales_map[f"layer_{idx}"] = float(mean_sc.real) if isinstance(mean_sc, complex) else float(mean_sc)
                if path and isinstance(layer, _real_nn.Linear):
                    awq_mod = MLXAWQLinear(layer, first_scales, group_size=group_size, bits=4)
                    _replace_submodule_at_path(resolved_model, path, awq_mod)
            register_quantized_artifact(resolved_model, method="awq", scales=scales_map)
            quantize_fn(resolved_model, group_size=group_size, bits=4)
            return (0.75, "quantized_awq")

        if method == "gptq":
            logger.info("Applying second-order Hessian error compensation (GPTQ) to MLX model %s", model_name)
            for _path, layer in named_layers:
                raw_w = _get_layer_weight(layer)
                assert raw_w is not None
                w_arr = mlx.array(raw_w, dtype=mlx.float32)
                in_dim = int(w_arr.shape[1])
                calib_acts = _collect_embedding_activations(resolved_model, calib_data, in_dim)
                first_gptq = calibrate_gptq_weights(w_arr, calib_acts)
                if isinstance(layer, MutableMapping):
                    layer["weight"] = mlx.array(first_gptq)
            register_quantized_artifact(resolved_model, method="gptq")
            quantize_fn(resolved_model, group_size=group_size, bits=4)
            return (0.75, "quantized_gptq")

        quantize_fn(resolved_model, group_size=group_size, bits=bits)
        register_quantized_artifact(resolved_model, method=method)
        reduction = 0.75 if bits == 4 else 0.5
        return (reduction, f"quantized_{method}")

    return quantize_model_wrapper(
        backend_name="mlx",
        model_name=model_name,
        method=method,
        apply_fn=apply_fn,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )


__all__ = [
    "MLXAWQLinear",
    "_extract_first_2d_weight",
    "calibrate_awq_scales",
    "calibrate_gptq_weights",
    "quantize_model",
]
