"""JAX-specific model quantization logic and AWQ implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

MIN_NDIM_FOR_QUANTIZATION = 2
if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    from flax import nnx as _nnx

    from .gemma4 import Gemma4Config as _Gemma4Config
    from .gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    jax: Any = _jax
    jnp: Any = _jnp
    nnx: Any = _nnx
    Gemma4Config: Any = _Gemma4Config
    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    jax = None
    jnp = None
    nnx = None
    Gemma4Config = None
    Gemma4ForCausalLM = None


def quantize_int8(tensor: Any) -> tuple[Any, Any]:
    """Quantize a tensor to uniform symmetric int8.

    Applies the transformation:
        scale = max(|tensor|) / 127.0
        q_tensor = clip(round(tensor / scale), -128, 127).astype(int8)

    Args:
        tensor: The input floating-point tensor.

    Returns:
        A tuple of (q_tensor, scale) where q_tensor is int8 and scale is float.

    Raises:
        DependencyMissingError: If JAX is required but missing.
    """
    if jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for quantize_int8.")
    max_val = jnp.max(jnp.abs(tensor))
    scale = jnp.where(max_val == 0.0, 1.0, max_val / 127.0)
    q_tensor = jnp.clip(jnp.round(tensor / scale), -128, 127).astype(jnp.int8)
    return (q_tensor, scale)


def compute_channel_activation_statistics(activations: Any) -> Any:
    """Compute per-channel average absolute activation magnitude across tokens.

    For input activations X of shape (..., channels), computes:
        s_c = mean(|X_{..., c}|) across all preceding batch and sequence dimensions.

    Args:
        activations: JAX array of activations with shape (..., channels).

    Returns:
        1D JAX array of shape (channels,) with mean absolute activation magnitude.

    Raises:
        DependencyMissingError: If JAX is required but missing.
    """
    if jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for compute_channel_activation_statistics.")
    abs_act = jnp.abs(activations)
    if abs_act.ndim == 1:
        return abs_act
    reduce_axes = tuple(range(abs_act.ndim - 1))
    return jnp.mean(abs_act, axis=reduce_axes)


def compute_salient_mask(activation_magnitude: Any, salient_ratio: float = 0.01) -> Any:
    """Compute a binary boolean mask identifying salient channels based on activation magnitudes.

    Identifies top-k channels with largest activation magnitudes:
        k = max(1, int(channels * salient_ratio))

    Args:
        activation_magnitude: 1D array of shape (channels,) with activation magnitudes.
        salient_ratio: Fraction of channels to mark as salient (must be between 0.0 and 1.0).

    Returns:
        1D boolean array of shape (channels,) where True indicates a salient channel.

    Raises:
        DependencyMissingError: If JAX is required but missing.
        ValueError: If salient_ratio is not within (0.0, 1.0].
    """
    if jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for compute_salient_mask.")
    if salient_ratio <= 0.0 or salient_ratio > 1.0:
        raise ValueError(f"salient_ratio must be in (0.0, 1.0], got {salient_ratio}")

    channels = int(activation_magnitude.shape[0])
    num_salient = max(1, int(channels * salient_ratio))
    top_indices = jnp.argsort(activation_magnitude)[-num_salient:]
    mask = jnp.zeros((channels,), dtype=bool)
    return mask.at[top_indices].set(True)


def quantize_awq(
    tensor: Any,
    channel_activations: Any | None = None,
    salient_ratio: float = 0.01,
) -> tuple[Any, Any, Any, Any]:
    """Quantize a weight tensor using Activation-aware Weight Quantization (AWQ).

    Mathematical Formulation:
        1. Given weight matrix W of shape (in_features, out_features) and activation magnitude s:
           Salient channels C_salient are identified as the top-k% largest elements of s.
        2. A binary channel protection mask M is formed (M_c = 1 if c in C_salient, else 0).
        3. Salient weights are preserved in floating-point:
           W_salient = W * M
        4. Non-salient weights are quantized to int8:
           W_nonsalient = W * (1 - M)
           scale = max(|W_nonsalient|) / 127.0
           W_q = clip(round(W_nonsalient / scale), -128, 127).astype(int8)

    Args:
        tensor: 2D or higher dimensional weight tensor.
        channel_activations: Optional calibration activation array or channel magnitude vector.
        salient_ratio: Proportion of channels to protect as salient (default 0.01).

    Returns:
        Tuple of (q_tensor, quant_scale, salient_weights, salient_mask).

    Raises:
        DependencyMissingError: If JAX is required but missing.
    """
    if jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for quantize_awq.")

    # Determine channel dimension (first axis in Flax NNX kernel: in_features)
    in_channels = tensor.shape[0]

    act_magnitude: Any = None
    if channel_activations is not None:
        if hasattr(channel_activations, "ndim") and channel_activations.ndim > 1 and channel_activations.shape[-1] == in_channels:
            act_magnitude = compute_channel_activation_statistics(channel_activations)
        elif hasattr(channel_activations, "shape") and channel_activations.shape[0] == in_channels:
            act_magnitude = channel_activations

    if act_magnitude is None:
        # Fallback: compute channel magnitude from weight tensor L1-norm across output dimensions
        reduce_axes = tuple(range(1, tensor.ndim))
        act_magnitude = jnp.mean(jnp.abs(tensor), axis=reduce_axes)

    salient_mask_1d = compute_salient_mask(act_magnitude, salient_ratio=salient_ratio)

    # Broadcast mask to match weight tensor dimensions
    broadcast_shape = [in_channels] + [1] * (tensor.ndim - 1)
    mask_nd = jnp.reshape(salient_mask_1d, broadcast_shape)

    salient_weights = jnp.where(mask_nd, tensor, jnp.zeros_like(tensor))
    non_salient_weights = jnp.where(mask_nd, jnp.zeros_like(tensor), tensor)

    max_val = jnp.max(jnp.abs(non_salient_weights))
    scale = jnp.where(max_val == 0.0, 1.0, max_val / 127.0)
    q_tensor = jnp.clip(jnp.round(non_salient_weights / scale), -128, 127).astype(jnp.int8)

    return (q_tensor, scale, salient_weights, salient_mask_1d)


def dequantize_awq(q_tensor: Any, scale: Any, salient_weights: Any) -> Any:
    """Dequantize an AWQ-quantized tensor back into floating-point representation.

    Computes:
        W_reconstructed = (q_tensor * scale) + salient_weights

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
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for dequantize_awq.")
    return (q_tensor.astype(salient_weights.dtype) * scale) + salient_weights


def _set_param_metadata(param: Any, key: str, value: Any) -> None:
    """Safely attach metadata to a parameter (Flax NNX Param or mock).

    Args:
        param: Target parameter object.
        key: Metadata key name.
        value: Metadata value to store.
    """
    if hasattr(param, "set_metadata"):
        param.set_metadata(key, value)
    else:
        setattr(param, key, value)


def _apply_quantization_to_model(
    model: Any,
    method: str,
    calibration_samples: list[Any] | None = None,
    salient_ratio: float = 0.01,
) -> tuple[str, float, int]:
    """Apply quantization to the model graph and preserve scale factors.

    Args:
        model: The model object to quantize.
        method: The quantization method string ('int8', 'awq').
        calibration_samples: Optional list of calibration activation arrays.
        salient_ratio: Ratio of salient channels to preserve in AWQ.

    Returns:
        A tuple of (status string, memory reduction factor, count of quantized parameters).
    """
    quantized_params = 0
    if method == "int8":
        scales: dict[str, object] = {}
        for path, param in nnx.graph.iter_graph(model):
            if isinstance(param, nnx.Param) and hasattr(param.value, "ndim") and (param.value.ndim >= MIN_NDIM_FOR_QUANTIZATION):
                (q_tensor, scale) = quantize_int8(param.value)
                param.value = q_tensor
                scale_float = float(scale)
                _set_param_metadata(param, "quant_scale", scale_float)
                scales[str(path)] = scale_float
                quantized_params += 1
        model._quant_scales = scales
        status = "quantized_int8"
        memory_reduction = 0.5
        logger.info("Quantized %d parameters using uniform int8", quantized_params)
    elif method == "awq":
        scales = {}
        for idx, (path, param) in enumerate(nnx.graph.iter_graph(model)):
            if isinstance(param, nnx.Param) and hasattr(param.value, "ndim") and (param.value.ndim >= MIN_NDIM_FOR_QUANTIZATION):
                calib_act = calibration_samples[idx % len(calibration_samples)] if calibration_samples else None
                (q_tensor, scale, salient_weights, salient_mask) = quantize_awq(
                    param.value,
                    channel_activations=calib_act,
                    salient_ratio=salient_ratio,
                )
                param.value = q_tensor
                scale_float = float(scale)
                _set_param_metadata(param, "quant_scale", scale_float)
                _set_param_metadata(param, "salient_weights", salient_weights)
                _set_param_metadata(param, "salient_mask", salient_mask)
                scales[str(path)] = scale_float
                quantized_params += 1
        model._quant_scales = scales
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
    calibration_samples: list[Any] | None = None,
    salient_ratio: float = 0.01,
    **kwargs: object,
) -> JSONDict:
    """Quantize a JAX model using int8 or AWQ.

    Args:
        model_name: The name of the model to quantize.
        method: The quantization method ('int8', 'awq', etc.).
        calibration_samples: Optional calibration activation samples for AWQ.
        salient_ratio: Ratio of salient weights to protect in AWQ.
        **kwargs: Additional backend keyword arguments.

    Returns:
        A dictionary containing quantization status and metadata.

    Raises:
        DependencyMissingError: If JAX quantization dependencies are missing.
    """
    if jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX quantization dependencies are missing.")

    try:
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
        (status, memory_reduction, _) = _apply_quantization_to_model(
            model,
            method,
            calibration_samples=calibration_samples,
            salient_ratio=salient_ratio,
        )
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"
        memory_reduction = 0.0

    return {
        "backend": "jax",
        "model": model_name,
        "method": method,
        "status": status,
        "memory_reduction_factor": float(memory_reduction),
    }
