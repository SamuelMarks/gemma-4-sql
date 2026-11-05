"""Common quantization utility for backends."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Sequence
from typing import Protocol, TypeGuard, runtime_checkable

from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    QuantizationError,
    StatusValue,
    UnsupportedQuantizationMethodError,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    DTypeSpec,
    JSONValue,
    ModelType,
    QuantizableModelProtocol,
    QuantizeResultDict,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class BitsAndBytesConfigProtocol(Protocol):
    """Protocol for `transformers.BitsAndBytesConfig` constructor."""

    def __call__(self, **kwargs: bool | float | str | list[str] | DTypeProtocol | DTypeSpec) -> JSONValue:
        """Construct a `BitsAndBytesConfig` instance."""
        ...


@runtime_checkable
class QuantizableConfigHolderProtocol(Protocol):
    """Protocol for a model config object supporting a mutable `quantization_config` attribute."""

    quantization_config: JSONValue | None


class ModelWithQuantizableConfigProtocol(Protocol):
    """Protocol for a model exposing a `.config` attribute conforming to `QuantizableConfigHolderProtocol`."""

    @property
    def config(self) -> QuantizableConfigHolderProtocol:
        """Provide the model's configuration object."""
        ...


def _is_config_model(
    model: ModelType | QuantizableModelProtocol | ModelWithQuantizableConfigProtocol | None,
) -> TypeGuard[ModelWithQuantizableConfigProtocol]:
    """Narrow `model` to `ModelWithQuantizableConfigProtocol`."""
    return model is not None and not isinstance(model, (str, int, float, bool))


type BitsAndBytesFactory = BitsAndBytesConfigProtocol | Callable[..., JSONValue] | type


def apply_bits_and_bytes_quantization(
    method: str,
    bits_and_bytes_config_cls: BitsAndBytesFactory | None,
    float16_dtype: DTypeProtocol | DTypeSpec = None,
    *,
    model: ModelType | QuantizableModelProtocol | ModelWithQuantizableConfigProtocol | None = None,
    llm_int8_threshold: float = 6.0,
    llm_int8_skip_modules: list[str] | None = None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> tuple[float, str]:
    """Apply quantization using BitsAndBytes config mapping.

    Supports native INT8 and INT4 (NF4/FP4) bitsandbytes configurations with double quantization,
    supporting both in-memory model transformations and Hugging Face model loading.

    Args:
        method: The string representing the quantization method ('int8', 'int4').
        bits_and_bytes_config_cls: The bits and bytes config class.
        float16_dtype: Optional float16 dtype for 4-bit compute.
        model: Optional model instance to attach quantization metadata to.
        llm_int8_threshold: Threshold for 8-bit outlier feature detection.
        llm_int8_skip_modules: Optional list of module name substrings to skip in 8-bit.
        bnb_4bit_quant_type: 4-bit quantization datatype ('nf4' or 'fp4').
        bnb_4bit_use_double_quant: Whether to compress quantization constants (double quantization).

    Returns:
        A tuple of `(memory_reduction_factor, status_string)`.

    Raises:
        DependencyMissingError: If `bits_and_bytes_config_cls` is None.
        UnsupportedQuantizationMethodError: If `method` is not `'int8'` or `'int4'`.

    """
    if bits_and_bytes_config_cls is None:
        msg = "bitsandbytes and transformers are required for BitsAndBytes quantization. Install with `pip install bitsandbytes transformers`."
        raise DependencyMissingError(msg, dependency_name="bitsandbytes", backend="pytorch")

    if method == "int8":
        if llm_int8_skip_modules is not None:
            bnb_config = bits_and_bytes_config_cls(
                load_in_8bit=True,
                llm_int8_threshold=llm_int8_threshold,
                llm_int8_skip_modules=llm_int8_skip_modules,
            )
        else:
            bnb_config = bits_and_bytes_config_cls(
                load_in_8bit=True,
                llm_int8_threshold=llm_int8_threshold,
            )
        memory_reduction = 0.5
    elif method == "int4":
        bnb_config = bits_and_bytes_config_cls(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=float16_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )
        memory_reduction = 0.75
    else:
        logger.warning("Unsupported quantization method for BitsAndBytes: %s", method)
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method for BitsAndBytes: {method}",
            method=method,
            supported_methods=("int8", "int4"),
            backend="pytorch",
        )

    if _is_config_model(model):
        with contextlib.suppress(AttributeError, TypeError):
            model.config.quantization_config = bnb_config
        register_quantized_artifact(model, method=method)

    return (memory_reduction, f"quantized_{method}")


def quantize_model_wrapper(
    backend_name: str,
    model_name: str,
    method: str,
    apply_fn: Callable[..., tuple[float, str]],
    *,
    default_reduction: float = 0.0,
    raise_on_error: bool | None = None,
    **kwargs: StatusValue,
) -> QuantizeResultDict:
    """Wrap and standardize execution and error handling for quantization.

    Args:
        backend_name: Name of the backend.
        model_name: Name of the model.
        method: The quantization method.
        apply_fn: Function that applies quantization and returns `(memory_reduction, status)`.
        default_reduction: Default memory reduction factor used when `apply_fn` fails without raising.
        raise_on_error: Whether to raise `QuantizationError` on runtime failure.
        **kwargs: Additional keyword options inspected by `resolve_raise_on_error`.

    Returns:
        A dictionary containing the quantization results.

    Raises:
        DependencyMissingError: If raised by `apply_fn`.
        UnsupportedQuantizationMethodError: If raised by `apply_fn`.
        QuantizationError: If `raise_on_error` is True and `apply_fn` fails.

    """
    should_raise = resolve_raise_on_error(raise_on_error=raise_on_error, kwargs=dict(kwargs))
    error_message: str | None = None
    try:
        (memory_reduction, status) = apply_fn()
        if status.startswith("unsupported"):
            status_code = ExecutionStatus.UNSUPPORTED.value
        else:
            status_code = ExecutionStatus.COMPLETED.value
            logger.info("Loading model %s with %s quantization...", model_name, method)
    except (DependencyMissingError, UnsupportedQuantizationMethodError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
        logger.exception("Failed to quantize: ")
        if should_raise:
            raise QuantizationError(
                f"Quantization failed on {backend_name}: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="quantize",
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value
        memory_reduction = default_reduction

    result: QuantizeResultDict = {
        "backend": backend_name,
        "model": model_name,
        "method": method,
        "status": status,
        "status_code": status_code,
        "memory_reduction_factor": float(memory_reduction),
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


def compute_awq_channel_scales(
    weight_2d: Sequence[Sequence[float]],
    activations_2d: Sequence[Sequence[float]] | None = None,
    alpha_range: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[float]:
    """Compute optimal per-channel activation scales for AWQ quantization via grid search.

    Evaluates candidate scale vectors `s_j = (mean(|X_j|)^alpha) / (mean(|W_j|)^(1 - alpha))`
    across `alpha_range` and selects the scale vector minimizing 4-bit quantization reconstruction MSE.

    Args:
        weight_2d: 2D weight matrix of shape `(out_features, in_features)`.
        activations_2d: Optional 2D calibration activation matrix of shape `(tokens, in_features)`.
        alpha_range: Sequence of grid exponents in `[0.0, 1.0]` to evaluate.

    Returns:
        1D list of optimal per-channel scale factors of length `in_features`.

    Raises:
        ValueError: If `alpha_range` is empty or `weight_2d` is empty.

    """
    if not alpha_range:
        raise ValueError("alpha_range must contain at least one value.")
    if not weight_2d or not weight_2d[0]:
        raise ValueError("weight_2d must be a non-empty 2D matrix.")

    out_dim = len(weight_2d)
    in_dim = len(weight_2d[0])

    w_col_mean = [max(1e-5, sum(abs(float(weight_2d[r][c])) for r in range(out_dim)) / float(out_dim)) for c in range(in_dim)]
    if activations_2d and len(activations_2d) > 0 and len(activations_2d[0]) == in_dim:
        n_tok = len(activations_2d)
        x_col_mean = [max(1e-5, sum(abs(float(activations_2d[t][c])) for t in range(n_tok)) / float(n_tok)) for c in range(in_dim)]
    else:
        x_col_mean = list(w_col_mean)

    best_error = float("inf")
    best_scales = [1.0] * in_dim
    q_max = 7.0

    for alpha in alpha_range:
        raw_scales = [(x_col_mean[c] ** alpha) / max(1e-5, w_col_mean[c] ** (1.0 - alpha)) for c in range(in_dim)]
        norm = max(1e-5, math.sqrt(max(raw_scales) * min(raw_scales)))
        scales = [max(1e-5, s / norm) for s in raw_scales]

        total_sq_err = 0.0
        for r in range(out_dim):
            row_scaled = [float(weight_2d[r][c]) * scales[c] for c in range(in_dim)]
            row_max = max(1e-5, max(abs(v) for v in row_scaled))
            step = row_max / q_max
            for c in range(in_dim):
                q_val = max(-8.0, min(7.0, round(row_scaled[c] / step))) * step
                recon = q_val / scales[c]
                diff = float(weight_2d[r][c]) - recon
                total_sq_err += diff * diff

        mse = total_sq_err / float(out_dim * in_dim)
        if mse < best_error:
            best_error = mse
            best_scales = scales

    return best_scales


def _invert_spd_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Invert a symmetric positive-definite matrix using Gauss-Jordan elimination with diagonal fallback.

    Args:
        matrix: Square 2D matrix of shape `(n, n)`.

    Returns:
        Inverse matrix of shape `(n, n)`.

    """
    n = len(matrix)
    aug = [[float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(abs(aug[col][col]), 1e-8)
        inv_pivot = 1.0 / pivot if aug[col][col] >= 0.0 else -1.0 / pivot
        for j in range(2 * n):
            aug[col][j] *= inv_pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def compute_gptq_compensated_weights(
    weight_2d: Sequence[Sequence[float]],
    activations_2d: Sequence[Sequence[float]] | None = None,
    bits: int = 4,
    damp_percent: float = 0.01,
) -> list[list[float]]:
    """Quantize a 2D weight matrix using second-order Hessian column-wise error compensation (GPTQ).

    Args:
        weight_2d: 2D weight matrix of shape `(out_features, in_features)`.
        activations_2d: Optional 2D calibration activations of shape `(tokens, in_features)`.
        bits: Target quantization bit-width (default 4).
        damp_percent: Diagonal damping factor for Hessian positive-definiteness.

    Returns:
        2D list of GPTQ error-compensated quantized-and-dequantized weights.

    """
    out_dim = len(weight_2d)
    in_dim = len(weight_2d[0]) if out_dim > 0 else 0
    w = [[float(weight_2d[r][c]) for c in range(in_dim)] for r in range(out_dim)]

    if activations_2d and len(activations_2d) > 0 and len(activations_2d[0]) == in_dim:
        n_tok = len(activations_2d)
        scale_factor = 2.0 / float(max(1, n_tok))
        h = [[scale_factor * sum(float(activations_2d[t][i]) * float(activations_2d[t][j]) for t in range(n_tok)) for j in range(in_dim)] for i in range(in_dim)]
    else:
        h = [[1.0 if i == j else 0.0 for j in range(in_dim)] for i in range(in_dim)]

    diag_mean = sum(h[i][i] for i in range(in_dim)) / float(max(1, in_dim))
    damp = max(1e-4, damp_percent * (diag_mean if diag_mean > 0.0 else 1.0))
    for i in range(in_dim):
        h[i][i] += damp

    h_inv = _invert_spd_matrix(h)
    q_max = float((1 << (max(2, bits) - 1)) - 1)
    q_min = -float(1 << (max(2, bits) - 1))
    row_scales = [max(1e-5, max((abs(v) for v in w[r]), default=1e-5)) / q_max for r in range(out_dim)]

    for col in range(in_dim):
        d_ii = max(abs(h_inv[col][col]), 1e-6)
        for r in range(out_dim):
            scale = row_scales[r]
            w_val = w[r][col]
            q_val = max(q_min, min(q_max, round(w_val / scale))) * scale
            err = (w_val - q_val) / d_ii
            w[r][col] = q_val
            if col + 1 < in_dim:
                for k in range(col + 1, in_dim):
                    w[r][k] -= err * h_inv[col][k]

    return w
