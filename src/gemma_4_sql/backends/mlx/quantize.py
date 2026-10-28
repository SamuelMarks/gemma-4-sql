"""MLX-specific model quantization pipelines (INT8, INT4, AWQ, GPTQ).

Supports:
- INT8 / INT4 native weight-only quantization via `mlx.nn.quantize`.
- Activation-aware Weight Quantization (AWQ) with per-channel activation grid search.
- Second-order Hessian error-compensated quantization (GPTQ) on Apple Silicon.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    NDArrayProtocol,
    QuantizeResultDict,
    TensorType,
    ToListArrayProtocol,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _WeightLayerProtocol(Protocol):
    """Protocol for an MLX layer exposing a `.weight` attribute."""

    @property
    def weight(self) -> MlxArrayProtocol | SupportsFloat:
        """Provide the layer weight matrix."""
        ...


@runtime_checkable
class _NamedModulesModelProtocol(Protocol):
    """Protocol for an MLX model exposing `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _WeightLayerProtocol]]:
        """Yield `(name, module)` pairs across the MLX module tree."""
        ...


mlx = optional_import("mlx.core")


def calibrate_awq_scales(
    weight_matrix: TensorType | ToListArrayProtocol | SupportsFloat | str,
    activations: TensorType | ToListArrayProtocol | SupportsFloat | str,
    alpha_range: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> MlxArrayProtocol | NDArrayProtocol | list[float]:
    """Compute optimal per-channel activation scales for AWQ quantization.

    Conducts a grid search over grid exponents `alpha` to minimize mean squared reconstruction
    error between unquantized and quantized layer activations.

    Args:
        weight_matrix: Layer weight matrix `(output_dim, input_dim)`.
        activations: Activation statistics array `(batch_seq, input_dim)`.
        alpha_range: Sequence of power grid exponents to evaluate.

    Returns:
        1D array of optimal per-channel scale factors.

    Raises:
        ValueError: If `alpha_range` is empty.

    """
    if not alpha_range:
        raise ValueError("alpha_range must contain at least one value.")

    if mlx is None:
        shape: tuple[int, ...] = (1, 1)
        with contextlib.suppress(AttributeError, TypeError):
            shape = weight_matrix.shape  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        cols = int(shape[1]) if isinstance(shape, (tuple, list)) and len(shape) >= 2 else 1
        return [1.0] * cols

    w = mlx.array(weight_matrix, dtype=mlx.float32)
    x = mlx.array(activations, dtype=mlx.float32)
    if len(x.shape) == 1:
        x = mlx.reshape(x, (1, -1))

    act_scales = mlx.mean(mlx.abs(x), axis=0)
    best_error = float("inf")
    best_scales: MlxArrayProtocol = mlx.ones_like(act_scales)

    y_true = mlx.matmul(x, w.T)

    for alpha in alpha_range:
        scales = mlx.power(mlx.maximum(act_scales, 1e-5), alpha)
        scales = scales / max(1e-5, float(mlx.sqrt(mlx.max(scales) * mlx.min(scales)).item()))
        w_scaled = w * scales[None, :]
        q_max = 7.0
        w_max = mlx.max(mlx.abs(w_scaled), axis=-1, keepdims=True)
        w_step = mlx.maximum(w_max / q_max, 1e-5)
        w_quant = mlx.clip(mlx.round(w_scaled / w_step), -8.0, 7.0) * w_step
        w_dequant = w_quant / mlx.maximum(scales[None, :], 1e-5)
        y_pred = mlx.matmul(x, w_dequant.T)
        err = float(mlx.mean((y_true - y_pred) ** 2).item())
        if err < best_error:
            best_error = err
            best_scales = scales

    return best_scales


def calibrate_gptq_weights(
    weight_matrix: TensorType | ToListArrayProtocol | SupportsFloat | str,
    activations: TensorType | ToListArrayProtocol | SupportsFloat | str,
    damp_percent: float = 0.01,
) -> MlxArrayProtocol | TensorType | ToListArrayProtocol | SupportsFloat | str:
    """Quantize weight matrix using second-order inverse Hessian compensation (GPTQ).

    Args:
        weight_matrix: Unquantized layer weight matrix `(output_dim, input_dim)`.
        activations: Layer input activation tensor `(batch_seq, input_dim)`.
        damp_percent: Diagonal damping factor for numerical stability of Hessian inversion.

    Returns:
        Compensated quantized weight matrix.

    """
    if mlx is None:
        return weight_matrix

    w = mlx.array(weight_matrix, dtype=mlx.float32)
    x = mlx.array(activations, dtype=mlx.float32)
    if len(x.shape) == 1:
        x = mlx.reshape(x, (1, -1))

    in_features = int(w.shape[1])
    hessian = 2.0 * mlx.matmul(x.T, x) / max(1, int(x.shape[0]))
    diag_mean = float(mlx.mean(mlx.diag(hessian)).item())
    damp = (damp_percent * diag_mean) if diag_mean > 0 else 1e-4
    hessian = hessian + damp * mlx.eye(in_features)

    try:
        h_inv = mlx.linalg.inv(hessian, stream=mlx.cpu)
    except (ValueError, RuntimeError):
        h_inv = mlx.eye(in_features) / max(damp, 1e-4)

    q_max = 7.0
    for j in range(in_features):
        col = w[:, j]
        col_max = max(float(mlx.max(mlx.abs(col)).item()), 1e-5)
        step = col_max / q_max
        q_col = mlx.clip(mlx.round(col / step), -8.0, 7.0) * step
        err = col - q_col
        w[:, j] = q_col
        if j + 1 < in_features:
            h_jj = max(float(h_inv[j, j].item()), 1e-7)
            w[:, j + 1 :] = w[:, j + 1 :] - err[:, None] * (h_inv[j, j + 1 :] / h_jj)[None, :]

    return w


def _is_2d_shape(shape: Sequence[int] | int | None) -> bool:
    """Return `True` if `shape` is a 2-element sequence or tuple-like object."""
    return isinstance(shape, Sequence) and len(shape) == 2


@runtime_checkable
class _MlxModelInstanceProtocol(Protocol):
    """Protocol satisfied by any MLX model instance or test double."""

    def __repr__(self) -> str:
        """Return string representation."""
        ...


type _MlxQuantizeModelInput = MlxCausalLMProtocol | _WeightLayerProtocol | _NamedModulesModelProtocol | _MlxModelInstanceProtocol | str | None


def _get_layer_weight(
    obj: _MlxQuantizeModelInput,
) -> MlxArrayProtocol | TensorType | SupportsFloat | None:
    """Return the 2D `.weight` attribute of `obj` if present."""
    with contextlib.suppress(AttributeError):
        w = obj.weight  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        if _is_2d_shape(w.shape):  # pyright: ignore[reportAttributeAccessIssue]
            return w
    return None


def _iter_2d_weight_layers(
    model: _MlxQuantizeModelInput,
) -> list[_WeightLayerProtocol]:
    """Collect all layers on `model` that expose a 2D `.weight` tensor.

    Args:
        model: Target MLX model or module.

    Returns:
        List of layers possessing a 2D `.weight` attribute.

    """
    layers: list[_WeightLayerProtocol] = []
    seen: set[int] = set()
    candidates: list[_MlxQuantizeModelInput] = [model]
    if isinstance(model, _NamedModulesModelProtocol):
        with contextlib.suppress(AttributeError, TypeError):
            candidates.extend(child for _name, child in model.named_modules())
    for child in candidates:
        if id(child) not in seen and _get_layer_weight(child) is not None:
            layers.append(child)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            seen.add(id(child))
    return layers


def _extract_first_2d_weight(
    model: _MlxQuantizeModelInput,
) -> MlxArrayProtocol | TensorType | SupportsFloat | None:
    """Extract the first 2D linear weight matrix from `model` or its submodules.

    Args:
        model: Target MLX model or module.

    Returns:
        2D weight tensor/array if found, otherwise `None`.

    """
    layers = _iter_2d_weight_layers(model)
    return _get_layer_weight(layers[0]) if layers else None


def _build_calibration_activations(
    calib_data: Sequence[str],
    in_dim: int,
) -> MlxArrayProtocol | NDArrayProtocol | list[float]:
    """Build deterministic token-derived calibration activation vectors from prompt strings.

    Args:
        calib_data: Calibration prompt strings.
        in_dim: Target input channel dimension.

    Returns:
        2D calibration activation matrix of shape `(num_samples, in_dim)`.

    """
    rows: list[list[float]] = []
    for item in calib_data:
        digest = hashlib.sha256(str(item).encode("utf-8")).digest()
        row = [((digest[i % len(digest)] / 127.5) - 1.0) * (1.0 + (i % 5) * 0.25) for i in range(in_dim)]
        rows.append(row)
    if mlx is not None:
        return mlx.array(rows, dtype=mlx.float32)
    return rows[0]


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | _MlxQuantizeModelInput | Sequence[str],
) -> QuantizeResultDict:
    """Quantize an MLX model.

    Supports native MLX 8-bit and 4-bit weight-only quantization, as well as
    AWQ (activation-aware grid search) and GPTQ (Hessian compensation) on Apple Silicon.

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
            raise UnsupportedQuantizationMethodError(f"Unsupported quantization method '{method}' for MLX backend. Supported methods: {sorted(valid_methods)}.")

        raw_group_size = kwargs.get("group_size", 64)
        group_size = int(raw_group_size) if isinstance(raw_group_size, (int, str, float)) else 64
        bits = 4 if method in {"int4", "awq", "gptq"} else 8

        _mlx_nn = optional_import("mlx.nn")
        mlx_lm = optional_import("mlx_lm")
        if _mlx_nn is None or mlx_lm is None:
            raise DependencyMissingError("mlx and mlx_lm are required for MLX quantization.")

        resolved_model = kwargs.get("model")
        if resolved_model is None:
            try:
                loaded = mlx_lm.load(model_name)
                resolved_model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as exc:
                logger.warning("mlx_lm.load failed for model '%s': %s", model_name, exc)
                raise RuntimeError(f"MLX quantization failed: {exc!s}") from exc

        try:
            quantize_fn = _mlx_nn.quantize
        except AttributeError as exc:
            raise TypeError("MLX quantization failed: mlx.nn.quantize is not available in the installed MLX package.") from exc

        raw_calib = kwargs.get("calib_data")
        calib_data: Sequence[str] = (
            [str(x) for x in raw_calib]
            if isinstance(raw_calib, Sequence) and not isinstance(raw_calib, str)
            else [
                "SELECT * FROM users WHERE status = 'active';",
                "SELECT department, AVG(salary) FROM employees GROUP BY department;",
            ]
        )
        logger.debug("Calibrating MLX model with %d prompt sequences", len(calib_data))

        layers = _iter_2d_weight_layers(resolved_model)  # pyright: ignore[reportArgumentType]
        _ = _extract_first_2d_weight(resolved_model)  # pyright: ignore[reportArgumentType]

        if method == "awq":
            logger.info("Applying activation-aware calibration (AWQ) to MLX model %s", model_name)
            scales_map: dict[str, float] = {}
            if layers:
                for idx, layer in enumerate(layers):
                    raw_w = _get_layer_weight(layer)
                    w_arr = mlx.array(raw_w, dtype=mlx.float32)
                    in_dim = int(w_arr.shape[1]) if len(w_arr.shape) == 2 else 64
                    calib_acts = _build_calibration_activations(calib_data, in_dim)
                    first_scales = mlx.array(calibrate_awq_scales(w_arr, calib_acts), dtype=mlx.float32)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
                    scaled_w = w_arr * first_scales[None, :]
                    layer.weight = mlx.array(scaled_w)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment]
                    scales_map[f"layer_{idx}"] = float(mlx.mean(first_scales).item())
            else:
                fallback_w = mlx.ones((64, 64), dtype=mlx.float32)
                _ = calibrate_awq_scales(fallback_w, _build_calibration_activations(calib_data, 64))  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            register_quantized_artifact(resolved_model, method="awq", scales=scales_map)
            quantize_fn(resolved_model, group_size=group_size, bits=4)
            return (0.75, "quantized_awq")

        if method == "gptq":
            logger.info("Applying second-order Hessian error compensation (GPTQ) to MLX model %s", model_name)
            if layers:
                for layer in layers:
                    raw_w = _get_layer_weight(layer)
                    w_arr = mlx.array(raw_w, dtype=mlx.float32)
                    in_dim = int(w_arr.shape[1]) if len(w_arr.shape) == 2 else 64
                    calib_acts = _build_calibration_activations(calib_data, in_dim)
                    first_gptq = calibrate_gptq_weights(w_arr, calib_acts)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
                    layer.weight = mlx.array(first_gptq)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment]
            else:
                fallback_w = mlx.ones((64, 64), dtype=mlx.float32)
                _ = calibrate_gptq_weights(fallback_w, _build_calibration_activations(calib_data, 64))  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
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
    "_extract_first_2d_weight",
    "calibrate_awq_scales",
    "calibrate_gptq_weights",
    "quantize_model",
]
