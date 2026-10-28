"""JAX-specific model quantization logic and AWQ implementation."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    JaxArrayProtocol,
    JaxCausalLMProtocol,
    QuantizeResultDict,
    register_quantized_artifact,
)

MIN_NDIM_FOR_QUANTIZATION = 2
logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig


@runtime_checkable
class _MetadataParamProtocol(Protocol):
    """Protocol for an NNX `Param` or variable supporting `.set_metadata(key, value)`."""

    def set_metadata(self, key: str, value: float | str | Array, /) -> None:
        """Store quantization metadata on the parameter."""
        ...


def quantize_int8(tensor: Array | JaxArrayProtocol) -> tuple[Array, Array]:
    """Quantize a tensor to uniform symmetric int8.

    Applies the transformation:
        `scale = max(|tensor|) / 127.0`
        `q_tensor = clip(round(tensor / scale), -128, 127).astype(int8)`

    Args:
        tensor: The input floating-point tensor.

    Returns:
        A tuple of `(q_tensor, scale)` where `q_tensor` is int8 and `scale` is float.

    Raises:
        DependencyMissingError: If JAX is required but missing.

    """
    if jnp is None:
        raise DependencyMissingError("JAX is required for quantize_int8.")
    arr = jnp.asarray(tensor)
    max_val = jnp.max(jnp.abs(arr))
    scale = jnp.where(max_val == 0.0, 1.0, max_val / 127.0)
    q_tensor = jnp.clip(jnp.round(arr / scale), -128, 127).astype(jnp.int8)
    return (q_tensor, scale)


def compute_channel_activation_statistics(activations: Array | JaxArrayProtocol) -> Array:
    """Compute per-channel average absolute activation magnitude across tokens.

    For input activations `X` of shape `(..., channels)`, computes:
        `s_c = mean(|X_{..., c}|)` across all preceding batch and sequence dimensions.

    Args:
        activations: JAX array of activations with shape `(..., channels)`.

    Returns:
        1D JAX array of shape `(channels,)` with mean absolute activation magnitude.

    Raises:
        DependencyMissingError: If JAX is required but missing.

    """
    if jnp is None:
        raise DependencyMissingError("JAX is required for compute_channel_activation_statistics.")
    abs_act = jnp.abs(jnp.asarray(activations))
    if abs_act.ndim == 1:
        return abs_act
    reduce_axes = tuple(range(abs_act.ndim - 1))
    res: Array = jnp.mean(abs_act, axis=reduce_axes)
    return res


def compute_salient_mask(activation_magnitude: Array | JaxArrayProtocol, salient_ratio: float = 0.01) -> Array:
    """Compute a binary boolean mask identifying salient channels based on activation magnitudes.

    Identifies top-k channels with largest activation magnitudes:
        `k = max(1, int(channels * salient_ratio))`

    Args:
        activation_magnitude: 1D array of shape `(channels,)` with activation magnitudes.
        salient_ratio: Fraction of channels to mark as salient (must be in `(0.0, 1.0]`).

    Returns:
        1D boolean array of shape `(channels,)` where `True` indicates a salient channel.

    Raises:
        DependencyMissingError: If JAX is required but missing.
        ValueError: If `salient_ratio` is not within `(0.0, 1.0]`.

    """
    if jnp is None:
        raise DependencyMissingError("JAX is required for compute_salient_mask.")
    if salient_ratio <= 0.0 or salient_ratio > 1.0:
        raise ValueError(f"salient_ratio must be in (0.0, 1.0], got {salient_ratio}")

    mag_arr = jnp.asarray(activation_magnitude)
    channels = int(mag_arr.shape[0])
    num_salient = max(1, int(channels * salient_ratio))
    top_indices = jnp.argsort(mag_arr)[-num_salient:]
    mask = jnp.zeros((channels,), dtype=bool)
    res: Array = mask.at[top_indices].set(True)
    return res


def quantize_awq(
    tensor: Array | JaxArrayProtocol,
    channel_activations: Array | JaxArrayProtocol | None = None,
    salient_ratio: float = 0.01,
) -> tuple[Array, Array, Array, Array]:
    """Quantize a weight tensor using Activation-aware Weight Quantization (AWQ).

    Args:
        tensor: 2D or higher dimensional weight tensor.
        channel_activations: Optional calibration activation array or channel magnitude vector.
        salient_ratio: Proportion of channels to protect as salient (default `0.01`).

    Returns:
        Tuple of `(q_tensor, quant_scale, salient_weights, salient_mask)`.

    Raises:
        DependencyMissingError: If JAX is required but missing.

    """
    if jnp is None:
        raise DependencyMissingError("JAX is required for quantize_awq.")

    arr = jnp.asarray(tensor)
    in_channels = arr.shape[0]

    act_magnitude: Array | None = None
    if channel_activations is not None:
        calib_arr = jnp.asarray(channel_activations)
        if calib_arr.ndim > 1 and calib_arr.shape[-1] == in_channels:
            act_magnitude = compute_channel_activation_statistics(calib_arr)
        elif calib_arr.shape[0] == in_channels:
            act_magnitude = calib_arr

    if act_magnitude is None:
        reduce_axes = tuple(range(1, arr.ndim))
        act_magnitude = jnp.mean(jnp.abs(arr), axis=reduce_axes)

    salient_mask_1d = compute_salient_mask(act_magnitude, salient_ratio=salient_ratio)

    broadcast_shape = [in_channels] + [1] * (arr.ndim - 1)
    mask_nd = jnp.reshape(salient_mask_1d, broadcast_shape)

    salient_weights = jnp.where(mask_nd, arr, jnp.zeros_like(arr))
    non_salient_weights = jnp.where(mask_nd, jnp.zeros_like(arr), arr)

    max_val = jnp.max(jnp.abs(non_salient_weights))
    scale = jnp.where(max_val == 0.0, 1.0, max_val / 127.0)
    q_tensor = jnp.clip(jnp.round(non_salient_weights / scale), -128, 127).astype(jnp.int8)

    return (q_tensor, scale, salient_weights, salient_mask_1d)


def dequantize_awq(
    q_tensor: Array | JaxArrayProtocol,
    scale: Array | JaxArrayProtocol | float,
    salient_weights: Array | JaxArrayProtocol,
) -> Array:
    """Dequantize an AWQ-quantized tensor back into floating-point representation.

    Computes:
        `W_reconstructed = (q_tensor * scale) + salient_weights`

    Args:
        q_tensor: Int8 quantized tensor of non-salient weights.
        scale: Floating-point quantization scale factor.
        salient_weights: Preserved floating-point salient weights.

    Returns:
        Reconstructed floating-point tensor.

    Raises:
        DependencyMissingError: If JAX is required but missing.

    """
    if jnp is None:
        raise DependencyMissingError("JAX is required for dequantize_awq.")
    q_arr = jnp.asarray(q_tensor)
    sal_arr = jnp.asarray(salient_weights)
    scale_arr = jnp.asarray(scale, dtype=sal_arr.dtype)
    res: Array = (q_arr.astype(sal_arr.dtype) * scale_arr) + sal_arr
    return res


@runtime_checkable
class _AnyParamObjProtocol(Protocol):
    """Protocol satisfied by any parameter or mock parameter object."""

    def __repr__(self) -> str:
        """Return string representation."""
        ...


def _set_param_metadata(
    param: nnx.Param[Array] | _MetadataParamProtocol | _AnyParamObjProtocol | str | None,
    key: str,
    value: float | str | Array,
) -> None:
    """Attach quantization metadata to a Flax NNX `Param` supporting `set_metadata`.

    Args:
        param: Target parameter object.
        key: Metadata key name.
        value: Metadata value to store.

    """
    with contextlib.suppress(AttributeError, TypeError):
        param.set_metadata(key, value)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]


def _apply_quantization_to_model(
    model: JaxCausalLMProtocol | Gemma4ForCausalLM,
    method: str,
    calibration_samples: Sequence[Array | JaxArrayProtocol] | None = None,
    salient_ratio: float = 0.01,
) -> tuple[str, float, int]:
    """Apply quantization to the model graph and preserve scale factors.

    Args:
        model: The model object to quantize.
        method: The quantization method string (`'int8'`, `'awq'`).
        calibration_samples: Optional sequence of calibration activation arrays.
        salient_ratio: Ratio of salient channels to preserve in AWQ.

    Returns:
        A tuple of `(status_string, memory_reduction_factor, count_of_quantized_parameters)`.

    """
    quantized_params = 0
    if method == "int8":
        scales: dict[str, float] = {}
        for path, param in nnx.graph.iter_graph(model):
            if isinstance(param, nnx.Param):
                val = param[...]
                if isinstance(val, JaxArrayProtocol) and val.ndim >= MIN_NDIM_FOR_QUANTIZATION:
                    (q_tensor, scale) = quantize_int8(val)
                    param[...] = q_tensor
                    scale_float = float(scale)
                    _set_param_metadata(param, "quant_scale", scale_float)
                    scales[str(path)] = scale_float
                    quantized_params += 1
        register_quantized_artifact(model, method=method, scales=scales)
        status = "quantized_int8"
        memory_reduction = 0.5
        logger.info("Quantized %d parameters using uniform int8", quantized_params)
    elif method == "awq":
        scales = {}
        for idx, (path, param) in enumerate(nnx.graph.iter_graph(model)):
            if isinstance(param, nnx.Param):
                val = param[...]
                if isinstance(val, JaxArrayProtocol) and val.ndim >= MIN_NDIM_FOR_QUANTIZATION:
                    calib_act = calibration_samples[idx % len(calibration_samples)] if calibration_samples else None
                    (q_tensor, scale, salient_weights, salient_mask) = quantize_awq(
                        val,
                        channel_activations=calib_act,
                        salient_ratio=salient_ratio,
                    )
                    param[...] = q_tensor
                    scale_float = float(scale)
                    _set_param_metadata(param, "quant_scale", scale_float)
                    _set_param_metadata(param, "salient_weights", salient_weights)
                    _set_param_metadata(param, "salient_mask", salient_mask)
                    scales[str(path)] = scale_float
                    quantized_params += 1
        register_quantized_artifact(model, method=method, scales=scales)
        status = "quantized_awq"
        memory_reduction = 0.7
        logger.info("Quantized %d parameters using AWQ (salient_ratio=%.3f)", quantized_params, salient_ratio)
    else:
        status = f"unsupported_method_{method}"
        memory_reduction = 0.0
    return (status, memory_reduction, quantized_params)


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | Sequence[Array | JaxArrayProtocol] | JaxCausalLMProtocol,
) -> QuantizeResultDict:
    """Quantize a JAX model using int8 or AWQ.

    Args:
        model_name: The name of the model to quantize.
        method: The quantization method (`'int8'`, `'awq'`, etc.).
        **kwargs: Additional backend keyword arguments (such as `calibration_samples`, `salient_ratio`, `strict_method=True`).

    Returns:
        A dictionary containing quantization status and metadata.

    Raises:
        DependencyMissingError: If JAX quantization dependencies are missing.
        UnsupportedQuantizationMethodError: If `strict_method=True` and `method` is unsupported.

    """
    if jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX dependencies are missing.", backend="jax")

    if kwargs.get("strict_method") and method not in {"int8", "awq"}:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for JAX backend.",
            method=method,
            supported_methods=("int8", "awq"),
            backend="jax",
        )

    raw_samples = kwargs.get("calibration_samples")
    calibration_samples: list[Array | JaxArrayProtocol] | None = [item for item in raw_samples if isinstance(item, JaxArrayProtocol)] if isinstance(raw_samples, (list, tuple)) else None
    raw_ratio = kwargs.get("salient_ratio", 0.01)
    salient_ratio = float(raw_ratio) if isinstance(raw_ratio, (int, float)) else 0.01
    provided_model = kwargs.get("model")

    def _apply() -> tuple[float, str]:
        """Instantiate or resolve JAX model and apply uniform INT8 or AWQ quantization."""
        cached = provided_model if isinstance(provided_model, (JaxCausalLMProtocol, Gemma4ForCausalLM)) else _MODEL_CACHE.get_model(model_name)
        if cached is not None:
            model: JaxCausalLMProtocol | Gemma4ForCausalLM = cached
        else:
            cfg = resolve_gemma4_config(Gemma4Config, model_name)
            model = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
        (status, memory_reduction, _) = _apply_quantization_to_model(
            model,
            method,
            calibration_samples=calibration_samples,
            salient_ratio=salient_ratio,
        )
        if not status.startswith("unsupported"):
            _MODEL_CACHE.put_model(model_name, model)  # pyright: ignore[reportArgumentType]
        return (float(memory_reduction), status)

    return quantize_model_wrapper(
        backend_name="jax",
        model_name=model_name,
        method=method,
        apply_fn=_apply,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )
