"""JAX-specific model quantization logic with `QuantizedLinear` and `AWQLinear` NNX modules."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import jax
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.jax._protocols import is_jnp_missing
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.config import JaxDTypeLike, ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.layers import ConstVar, LinearLayerProtocol
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
QUANTIZABLE_LINEAR_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)
logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig


@runtime_checkable
class _MetadataParamProtocol(Protocol):
    """Protocol for an NNX `Param` or variable supporting `.set_metadata(key, value)`."""

    def set_metadata(self, key: str, value: float | str | Array, /) -> None:
        """Store quantization metadata on the parameter.

        Args:
            key: Metadata key string.
            value: Scalar or array metadata value.

        """
        ...


@runtime_checkable
class _LinearProjectionInjectableProtocol(Protocol):
    """Protocol for NNX models supporting in-place linear projection replacement."""

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Replace matching `nnx.Linear` projections with `LinearLayerProtocol` modules.

        Args:
            target_modules: Target projection attribute names.
            adapter_factory: Factory mapping `nnx.Linear` to a quantized `LinearLayerProtocol`.

        Returns:
            Count of replaced linear modules.

        """
        ...


class QuantizedLinear(nnx.Module):
    """Uniform symmetric INT8 linear projection module in Flax NNX.

    Stores `kernel` as `int8` and multiplies `(x @ kernel) * quant_scale` during forward passes
    so activations remain properly scaled instead of exploding by `127x` per layer.

    Attributes:
        kernel: Quantized `int8` weight parameter of shape `(in_features, out_features)`.
        bias: Optional additive floating-point bias parameter of shape `(out_features,)`.
        quant_scale: Scalar floating-point dequantization scale factor stored as `ConstVar`.
        dtype: Output floating-point computation dtype.

    """

    kernel: nnx.Param[Array]
    bias: nnx.Param[Array] | None
    quant_scale: ConstVar
    dtype: JaxDTypeLike

    def __init__(
        self,
        q_kernel: Array,
        scale: Array,
        bias: nnx.Param[Array] | None = None,
        dtype: JaxDTypeLike = jnp.float32,
    ) -> None:
        """Initialize an INT8 `QuantizedLinear` projection.

        Args:
            q_kernel: Quantized `int8` kernel of shape `(in_features, out_features)`.
            scale: Scalar floating-point scale tensor.
            bias: Optional additive bias parameter.
            dtype: Output computation dtype.

        """
        self.kernel = nnx.Param(q_kernel)
        self.bias = bias
        self.quant_scale = ConstVar(jnp.asarray(scale, dtype=jnp.float32))
        self.dtype = dtype

    @classmethod
    def from_linear(cls, linear: nnx.Linear) -> tuple[QuantizedLinear, float]:
        """Quantize an `nnx.Linear` module into a `QuantizedLinear` instance.

        Args:
            linear: Source floating-point `nnx.Linear` layer.

        Returns:
            Tuple of `(quantized_layer, scale_float)`.

        """
        kernel_val = linear.kernel[...]
        q_tensor, scale = quantize_int8(kernel_val)
        out_dtype: JaxDTypeLike = kernel_val.dtype
        q_layer = cls(q_tensor, scale, bias=linear.bias, dtype=out_dtype)
        scale_float = float(scale)
        _set_param_metadata(q_layer.kernel, "quant_scale", scale_float)
        return q_layer, scale_float

    def __call__(self, x: Array) -> Array:
        """Compute scaled INT8 linear projection `(x @ kernel) * quant_scale + bias`.

        Args:
            x: Input activation tensor of shape `(..., in_features)`.

        Returns:
            Projected floating-point tensor of shape `(..., out_features)`.

        """
        compute_dtype = x.dtype
        scale_val = self.quant_scale[...].astype(compute_dtype)
        out = jnp.dot(x, self.kernel[...].astype(compute_dtype)) * scale_val
        if self.bias is not None:
            out += self.bias[...].astype(compute_dtype)
        res: Array = out.astype(self.dtype)
        return res


class AWQLinear(nnx.Module):
    """Activation-aware Weight Quantization (AWQ) linear module in Flax NNX.

    Stores non-salient weights in `int8` (`kernel`) alongside compact floating-point
    salient input channel rows (`salient_indices`, `salient_rows`), computing:
        `y = (x @ kernel) * quant_scale + x[..., salient_indices] @ salient_rows + bias`

    Attributes:
        kernel: Quantized `int8` non-salient weight parameter of shape `(in_features, out_features)`.
        bias: Optional additive floating-point bias parameter of shape `(out_features,)`.
        quant_scale: Scalar floating-point dequantization scale factor stored as `ConstVar`.
        salient_indices: 1D `int32` indices of protected salient input channels `(num_salient,)`.
        salient_rows: 2D floating-point weights for salient input channels `(num_salient, out_features)`.
        salient_mask: 1D boolean mask over `in_features` marking salient channels.
        dtype: Output floating-point computation dtype.

    """

    kernel: nnx.Param[Array]
    bias: nnx.Param[Array] | None
    quant_scale: ConstVar
    salient_indices: ConstVar
    salient_rows: ConstVar
    salient_mask: ConstVar
    dtype: JaxDTypeLike

    def __init__(
        self,
        q_kernel: Array,
        scale: Array,
        salient_indices: Array,
        salient_rows: Array,
        salient_mask: Array,
        bias: nnx.Param[Array] | None = None,
        dtype: JaxDTypeLike = jnp.float32,
    ) -> None:
        """Initialize an AWQ mixed-precision linear projection.

        Args:
            q_kernel: Quantized `int8` non-salient kernel `(in_features, out_features)`.
            scale: Scalar floating-point scale tensor for `q_kernel`.
            salient_indices: 1D `int32` index array of salient input channels `(k,)`.
            salient_rows: Preserved floating-point rows `(k, out_features)`.
            salient_mask: 1D boolean salient mask `(in_features,)`.
            bias: Optional additive bias parameter.
            dtype: Output computation dtype.

        """
        self.kernel = nnx.Param(q_kernel)
        self.bias = bias
        self.quant_scale = ConstVar(jnp.asarray(scale, dtype=jnp.float32))
        self.salient_indices = ConstVar(jnp.asarray(salient_indices, dtype=jnp.int32))
        self.salient_rows = ConstVar(salient_rows)
        self.salient_mask = ConstVar(salient_mask)
        self.dtype = dtype

    @classmethod
    def from_linear(
        cls,
        linear: nnx.Linear,
        channel_activations: Array | JaxArrayProtocol | None = None,
        salient_ratio: float = 0.01,
    ) -> tuple[AWQLinear, float]:
        """Quantize an `nnx.Linear` module into an `AWQLinear` instance.

        Args:
            linear: Source floating-point `nnx.Linear` layer.
            channel_activations: Optional calibration activation array.
            salient_ratio: Fraction of salient input channels preserved in full precision.

        Returns:
            Tuple of `(awq_layer, scale_float)`.

        """
        kernel_val = linear.kernel[...]
        q_tensor, scale, salient_weights, salient_mask = quantize_awq(
            kernel_val,
            channel_activations=channel_activations,
            salient_ratio=salient_ratio,
        )
        num_salient = max(1, int(jnp.sum(salient_mask.astype(jnp.int32))))
        salient_idx = jnp.nonzero(salient_mask, size=num_salient)[0].astype(jnp.int32)
        salient_rows = kernel_val[salient_idx]
        out_dtype: JaxDTypeLike = kernel_val.dtype
        awq_layer = cls(
            q_tensor,
            scale,
            salient_idx,
            salient_rows,
            salient_mask,
            bias=linear.bias,
            dtype=out_dtype,
        )
        scale_float = float(scale)
        _set_param_metadata(awq_layer.kernel, "quant_scale", scale_float)
        _set_param_metadata(awq_layer.kernel, "salient_weights", salient_weights)
        _set_param_metadata(awq_layer.kernel, "salient_mask", salient_mask)
        return awq_layer, scale_float

    def __call__(self, x: Array) -> Array:
        """Compute AWQ projection combining scaled INT8 non-salient channels and FP salient channels.

        Args:
            x: Input activation tensor of shape `(..., in_features)`.

        Returns:
            Projected floating-point tensor of shape `(..., out_features)`.

        """
        compute_dtype = x.dtype
        scale_val = self.quant_scale[...].astype(compute_dtype)
        non_salient_out = jnp.dot(x, self.kernel[...].astype(compute_dtype)) * scale_val
        x_salient = jnp.take(x, self.salient_indices[...], axis=-1)
        salient_out = jnp.dot(x_salient, self.salient_rows[...].astype(compute_dtype))
        out = non_salient_out + salient_out
        if self.bias is not None:
            out += self.bias[...].astype(compute_dtype)
        res: Array = out.astype(self.dtype)
        return res


def _resolve_quantize_target(
    model_name: str,
    provided_model: BackendKwargValue | None,
) -> JaxCausalLMProtocol | Gemma4ForCausalLM:
    """Resolve an in-memory, cached, or freshly initialized JAX model for quantization.

    Args:
        model_name: Target model preset or checkpoint identifier.
        provided_model: Optional caller-supplied model instance.

    Returns:
        Resolved `JaxCausalLMProtocol` or `Gemma4ForCausalLM` instance.

    """
    if isinstance(provided_model, JaxCausalLMProtocol):
        return provided_model
    raw_cached = _MODEL_CACHE.get_model(model_name)
    if isinstance(raw_cached, Gemma4ForCausalLM):
        return raw_cached
    if isinstance(raw_cached, JaxCausalLMProtocol):
        return raw_cached
    cfg = resolve_gemma4_config(Gemma4Config, model_name)
    return Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))


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
    if is_jnp_missing(jnp):
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
    if is_jnp_missing(jnp):
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
    if is_jnp_missing(jnp):
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
    if is_jnp_missing(jnp):
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
    if is_jnp_missing(jnp):
        raise DependencyMissingError("JAX is required for dequantize_awq.")
    q_arr = jnp.asarray(q_tensor)
    sal_arr = jnp.asarray(salient_weights)
    scale_arr = jnp.asarray(scale, dtype=sal_arr.dtype)
    res: Array = (q_arr.astype(sal_arr.dtype) * scale_arr) + sal_arr
    return res


def _set_param_metadata(
    param: nnx.Param[Array] | _MetadataParamProtocol | JaxArrayProtocol | str | None,
    key: str,
    value: float | str | Array,
) -> None:
    """Attach quantization metadata to a Flax NNX `Param` supporting `set_metadata`.

    Args:
        param: Target parameter object.
        key: Metadata key name.
        value: Metadata value to store.

    """
    if isinstance(param, _MetadataParamProtocol):
        param.set_metadata(key, value)


def _apply_quantization_to_model(
    model: JaxCausalLMProtocol | Gemma4ForCausalLM,
    method: str,
    calibration_samples: Sequence[Array | JaxArrayProtocol] | None = None,
    salient_ratio: float = 0.01,
) -> tuple[str, float, int]:
    """Replace `nnx.Linear` projections with `QuantizedLinear` or `AWQLinear` modules.

    Args:
        model: The model object to quantize.
        method: The quantization method string (`'int8'`, `'awq'`).
        calibration_samples: Optional sequence of calibration activation arrays.
        salient_ratio: Ratio of salient channels to preserve in AWQ.

    Returns:
        A tuple of `(status_string, memory_reduction_factor, count_of_quantized_parameters)`.

    """
    quantized_params = 0
    target_names = list(QUANTIZABLE_LINEAR_TARGETS)
    if method == "int8":
        scales: dict[str, float] = {}
        if isinstance(model, _LinearProjectionInjectableProtocol):

            def _make_int8(linear: nnx.Linear) -> LinearLayerProtocol:
                """Wrap `linear` into `QuantizedLinear` and record its scale.

                Args:
                    linear: Base Flax NNX `Linear` projection module.

                Returns:
                    Uniform int8 `QuantizedLinear` replacement layer.

                """
                nonlocal quantized_params
                q_layer, scale_float = QuantizedLinear.from_linear(linear)
                scales[f"linear_{quantized_params}"] = scale_float
                quantized_params += 1
                return q_layer

            model.inject_lora_projections(target_names, _make_int8)
        register_quantized_artifact(model, method=method, scales=scales)
        status = "quantized_int8"
        memory_reduction = 0.5
        logger.info("Quantized %d linear modules using uniform int8", quantized_params)
    elif method == "awq":
        scales = {}
        if isinstance(model, _LinearProjectionInjectableProtocol):

            def _make_awq(linear: nnx.Linear) -> LinearLayerProtocol:
                """Wrap `linear` into `AWQLinear` using optional calibration activations.

                Args:
                    linear: Base Flax NNX `Linear` projection module.

                Returns:
                    Activation-aware `AWQLinear` replacement layer.

                """
                nonlocal quantized_params
                calib_act = (
                    calibration_samples[quantized_params % len(calibration_samples)] if calibration_samples else None
                )
                awq_layer, scale_float = AWQLinear.from_linear(
                    linear,
                    channel_activations=calib_act,
                    salient_ratio=salient_ratio,
                )
                scales[f"linear_{quantized_params}"] = scale_float
                quantized_params += 1
                return awq_layer

            model.inject_lora_projections(target_names, _make_awq)
        register_quantized_artifact(model, method=method, scales=scales)
        status = "quantized_awq"
        memory_reduction = 0.7
        logger.info("Quantized %d linear modules using AWQ (salient_ratio=%.3f)", quantized_params, salient_ratio)
    else:
        status = f"unsupported_method_{method}"
        memory_reduction = 0.0
    return (status, memory_reduction, quantized_params)


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | Sequence[Array | JaxArrayProtocol] | JaxCausalLMProtocol,
) -> QuantizeResultDict:
    """Quantize a JAX model using int8 (`QuantizedLinear`) or AWQ (`AWQLinear`).

    Args:
        model_name: The name of the model to quantize.
        method: The quantization method (`'int8'`, `'awq'`, etc.).
        **kwargs: Additional backend keyword arguments (such as `calibration_samples`,
            `salient_ratio`, `strict_method=True`).

    Returns:
        A dictionary containing quantization status and metadata.

    Raises:
        DependencyMissingError: If JAX quantization dependencies are missing.
        UnsupportedQuantizationMethodError: If `strict_method=True` and `method` is unsupported.

    """
    if jax is None or is_jnp_missing(jnp) or nnx is None or not callable(Gemma4ForCausalLM):
        raise DependencyMissingError("JAX dependencies are missing.", backend="jax")

    if kwargs.get("strict_method") and method not in {"int8", "awq"}:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for JAX backend.",
            method=method,
            supported_methods=("int8", "awq"),
            backend="jax",
        )

    raw_samples = kwargs.get("calibration_samples")
    calibration_samples: list[Array | JaxArrayProtocol] | None = (
        [item for item in raw_samples if isinstance(item, JaxArrayProtocol)]
        if isinstance(raw_samples, (list, tuple))
        else None
    )
    raw_ratio = kwargs.get("salient_ratio", 0.01)
    salient_ratio = float(raw_ratio) if isinstance(raw_ratio, (int, float)) else 0.01
    provided_model = kwargs.get("model")

    def _apply() -> tuple[float, str]:
        """Instantiate or resolve JAX model and apply uniform INT8 or AWQ quantization.

        Returns:
            Tuple of `(memory_reduction_factor, status_string)`.

        """
        model = _resolve_quantize_target(model_name, provided_model)
        (status, memory_reduction, _) = _apply_quantization_to_model(
            model,
            method,
            calibration_samples=calibration_samples,
            salient_ratio=salient_ratio,
        )
        if not status.startswith("unsupported"):
            _MODEL_CACHE.put_model(model_name, model)
        return (float(memory_reduction), status)

    return quantize_model_wrapper(
        backend_name="jax",
        model_name=model_name,
        method=method,
        apply_fn=_apply,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )


__all__ = [
    "AWQLinear",
    "Gemma4Config",
    "QuantizedLinear",
    "compute_channel_activation_statistics",
    "compute_salient_mask",
    "dequantize_awq",
    "quantize_awq",
    "quantize_int8",
    "quantize_model",
]
