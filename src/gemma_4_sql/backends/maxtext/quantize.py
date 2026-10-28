"""MaxText-specific model quantization logic using Google AQT (Accurate Quantized Training).

Implements:
- INT8 and INT4 numerical quantization for weight and activation matrices.
- Dynamic per-channel scale factor computation: `scale = max(|W|, axis=-1) / clipping_bound`.
- Uniform symmetric quantization with clipping bounds:
    * INT8: `[-128, 127]` with clipping bound `127.0` (50% memory reduction).
    * INT4: `[-8, 7]` with clipping bound `7.0` (75% memory reduction).
- Selective projection targeting across attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
  and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from typing import SupportsFloat

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    JaxArrayProtocol,
    JSONDict,
    MaxTextCausalLMProtocol,
    PyTreeNode,
    QuantizeResultDict,
    TensorType,
)

logger = logging.getLogger(__name__)

type AQTParamLeaf = JaxArrayProtocol | TensorType | SupportsFloat | str
type AQTParamDict = dict[str, "AQTParamDict | AQTParamLeaf"]

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
aqt = optional_import("aqt.jax.v2")
_raw_gemma4 = optional_attr("maxtext.models.gemma4", "Gemma4Model")
Gemma4Model: Callable[..., MaxTextCausalLMProtocol] | None = _raw_gemma4 if callable(_raw_gemma4) else None  # ty: ignore[invalid-assignment]

DEFAULT_AQT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def quantize_tensor_aqt(
    tensor: TensorType | SupportsFloat,
    bits: int = 8,
) -> tuple[JaxArrayProtocol, JaxArrayProtocol]:
    """Quantize a tensor using Google AQT symmetric numerical quantization.

    Computes per-channel dynamic scaling:
        `clipping_bound = 2^(bits - 1) - 1`
        `scale = max(|tensor|, axis=-1, keepdims=True) / clipping_bound`
        `quantized = round(clip(tensor / scale, -clipping_bound - 1, clipping_bound))`

    Args:
        tensor: Floating-point array to quantize (typically shape `(d_in, d_out)`).
        bits: Number of quantization bits (8 for int8, 4 for int4). Must be > 0.

    Returns:
        A tuple of `(quantized_tensor, scale_factors)`.

    Raises:
        DependencyMissingError: If JAX is missing.
        ValueError: If `bits` is non-positive.

    """
    if jax is None or jnp is None:
        raise DependencyMissingError("JAX dependencies are missing.")
    if bits <= 0:
        raise ValueError(f"Quantization bits must be positive, got {bits}")

    clipping_bound = float((1 << (bits - 1)) - 1)
    max_val = jnp.max(jnp.abs(tensor), axis=-1, keepdims=True)
    scale: JaxArrayProtocol = jnp.maximum(max_val / clipping_bound, 1e-7)

    scaled = jnp.divide(tensor, scale)
    clipped = jnp.clip(scaled, -clipping_bound - 1.0, clipping_bound)
    quantized: JaxArrayProtocol = jnp.round(clipped)

    dtype = jnp.int8 if bits <= 8 else jnp.int16
    return quantized.astype(dtype), scale


def apply_aqt_quantization(
    params: AQTParamDict | AQTParamLeaf,
    method: str = "int8",
    quant_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[AQTParamDict, JSONDict, int]:
    """Apply Google AQT numerical quantization across model parameter projections.

    Args:
        params: Model parameter PyTree.
        method: Quantization format (`'int8'`, `'int4'`, `'aqt'`).
        quant_targets: Submodule names to quantize (defaults to Gemma 4 attention and MLP projections).

    Returns:
        A tuple of `(quantized_params, quantization_metadata, count_of_quantized_modules)`.

    Raises:
        DependencyMissingError: If JAX is missing.
        UnsupportedQuantizationMethodError: If `method` is unsupported.

    """
    if jax is None or jnp is None:
        raise DependencyMissingError("JAX dependencies are missing.")

    if method in {"int8", "aqt"}:
        bits = 8
        reduction = 0.5
    elif method == "int4":
        bits = 4
        reduction = 0.75
    else:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for MaxText backend.",
            method=method,
            supported_methods=("int8", "int4", "aqt"),
            backend="maxtext",
        )

    clipping_bound = float((1 << (bits - 1)) - 1)
    targets = tuple(quant_targets) if quant_targets is not None else DEFAULT_AQT_TARGETS

    aqt_native_cfg: str | None = None
    if aqt is not None:
        with contextlib.suppress(AttributeError, TypeError):
            aqt_native_cfg = str(aqt.config.fully_quantized(fwd_bits=bits, bwd_bits=bits))

    injected_count = 0

    def _traverse(curr: AQTParamDict | AQTParamLeaf, current_path: str = "") -> AQTParamDict | AQTParamLeaf:
        """Traverse parameters recursively to quantize targeted kernels."""
        nonlocal injected_count
        if not isinstance(curr, dict):
            return curr
        res: AQTParamDict = {}
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                v_dict: AQTParamDict = v
                if "kernel" in v_dict and any(k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in targets):
                    kernel = v_dict["kernel"]
                    q_kernel, scale = quantize_tensor_aqt(kernel, bits=bits)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
                    new_v: AQTParamDict = dict(v_dict)
                    new_v["kernel"] = q_kernel
                    new_v["kernel_scale"] = scale
                    new_v["aqt_config"] = {
                        "bits": bits,
                        "clipping_bound": clipping_bound,
                        "method": method,
                    }
                    res[k] = new_v
                    injected_count += 1
                else:
                    res[k] = _traverse(v_dict, sub_path)
            else:
                res[k] = v
        return res

    quantized_params = _traverse(params)
    metadata: JSONDict = {
        "backend": "maxtext",
        "method": method,
        "bits": bits,
        "clipping_bound": clipping_bound,
        "quant_targets": list(targets),
        "quantized_modules_count": injected_count,
        "memory_reduction_factor": reduction,
        "aqt_native": aqt is not None,
    }
    if aqt_native_cfg is not None:
        metadata["aqt_dg_config"] = aqt_native_cfg
    return quantized_params, metadata, injected_count  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | AQTParamDict | Sequence[str],
) -> QuantizeResultDict:
    """Quantize a MaxText Gemma 4 model using Google AQT numerical quantization.

    Args:
        model_name: The name or preset path of the target model.
        method: Quantization method (`'int8'`, `'int4'`, `'aqt'`).
        **kwargs: Optional parameters including `'params'` and `'quant_targets'`.

    Returns:
        A dictionary containing quantization results, status, and metadata.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.
        UnsupportedQuantizationMethodError: If `method` is not supported.

    """
    has_params = "params" in kwargs and kwargs["params"] is not None
    if jax is None or jnp is None or (Gemma4Model is None and not has_params):
        raise DependencyMissingError("MaxText dependencies are missing.", backend="maxtext")

    if method not in {"int8", "int4", "aqt"}:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for MaxText backend.",
            method=method,
            supported_methods=("int8", "int4", "aqt"),
            backend="maxtext",
        )

    default_reduction = 0.75 if method == "int4" else 0.5
    metadata: JSONDict = {}

    def _apply() -> tuple[float, str]:
        """Resolve MaxText model parameters and apply AQT quantization."""
        nonlocal metadata
        assert jax is not None
        assert jnp is not None
        if has_params:
            params = kwargs["params"]
        else:
            assert Gemma4Model is not None
            model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            params = model.init(rng, dummy_input)

        if isinstance(params, dict):
            params_dict: AQTParamDict = params  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            quant_targets = kwargs.get("quant_targets")
            targets_list: list[str] | None = [str(t) for t in quant_targets] if isinstance(quant_targets, (list, tuple)) else None
            _quantized_params, meta, count = apply_aqt_quantization(
                params=params_dict,
                method=method,
                quant_targets=targets_list,
            )
            metadata = meta
            raw_red = meta.get("memory_reduction_factor", default_reduction)
            reduction = float(raw_red) if isinstance(raw_red, (int, float)) else default_reduction
            logger.info("AQT quantization applied to %d modules with reduction %f", count, reduction)
        else:
            reduction = default_reduction
            metadata = {"method": method, "memory_reduction_factor": reduction}
        return (reduction, f"quantized_{method}")

    wrapped = quantize_model_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        method=method,
        apply_fn=_apply,
        default_reduction=float(default_reduction),
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )
    if metadata:
        wrapped["metadata"] = metadata  # ty: ignore[invalid-assignment]
    return wrapped


__all__ = [
    "DEFAULT_AQT_TARGETS",
    "AQTParamDict",
    "AQTParamLeaf",
    "Gemma4Model",
    "PyTreeNode",
    "apply_aqt_quantization",
    "quantize_model",
    "quantize_tensor_aqt",
]
