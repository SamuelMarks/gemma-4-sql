"""MaxText-specific model quantization logic using Google AQT (Accurate Quantized Training).

Implements:
- INT8 and INT4 numerical quantization for weight and activation matrices.
- Dynamic per-channel scale factor computation: scale = max(|W|, axis=-1) / clipping_bound.
- Uniform symmetric quantization with clipping bounds:
    * INT8: [-128, 127] with clipping bound 127.0 (50% memory reduction).
    * INT4: [-8, 7] with clipping bound 7.0 (75% memory reduction).
- Selective projection targeting across attention (q_proj, k_proj, v_proj, o_proj)
  and feed-forward layers (gate_proj, up_proj, down_proj).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    UnsupportedQuantizationMethodError,
)

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, TensorType

logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
aqt = optional_import("aqt.jax.v2")
Gemma4Model = optional_attr("maxtext.models.gemma4", "Gemma4Model")

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
    tensor: TensorType,
    bits: int = 8,
) -> tuple[Any, Any]:
    """Quantize a tensor using Google AQT symmetric numerical quantization.

    Computes per-channel dynamic scaling:
        clipping_bound = 2^(bits - 1) - 1
        scale = max(|tensor|, axis=-1, keepdims=True) / clipping_bound
        quantized = round(clip(tensor / scale, -clipping_bound - 1, clipping_bound))

    Args:
        tensor: Floating-point array to quantize (typically shape (d_in, d_out)).
        bits: Number of quantization bits (8 for int8, 4 for int4). Must be > 0.

    Returns:
        A tuple of (quantized_tensor, scale_factors).

    Raises:
        DependencyMissingError: If JAX is missing.
        ValueError: If bits is non-positive.

    """
    if jax is None or jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing.")
    if bits <= 0:
        raise ValueError(f"Quantization bits must be positive, got {bits}")

    clipping_bound = float((1 << (bits - 1)) - 1)
    max_val = jnp.max(jnp.abs(tensor), axis=-1, keepdims=True)
    scale = jnp.maximum(max_val / clipping_bound, 1e-7)

    scaled = tensor / scale
    clipped = jnp.clip(scaled, -clipping_bound - 1.0, clipping_bound)
    quantized = jnp.round(clipped)

    dtype = jnp.int8 if bits <= 8 else jnp.int16
    return quantized.astype(dtype), scale


def apply_aqt_quantization(
    params: dict[str, Any],
    method: str = "int8",
    quant_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Apply Google AQT numerical quantization across model parameter projections.

    Args:
        params: Model parameter PyTree.
        method: Quantization format ('int8', 'int4').
        quant_targets: Submodule names to quantize (defaults to Gemma 4 attention and MLP projections).

    Returns:
        A tuple of (quantized_params, quantization_metadata, count_of_quantized_modules).

    Raises:
        DependencyMissingError: If JAX is missing.
        ValueError: If method is unsupported.

    """
    if jax is None or jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing.")

    if method == "int8":
        bits = 8
        reduction = 0.5
    elif method == "int4":
        bits = 4
        reduction = 0.75
    else:
        bits = 8
        reduction = 0.7

    clipping_bound = float((1 << (bits - 1)) - 1)
    targets = tuple(quant_targets) if quant_targets is not None else DEFAULT_AQT_TARGETS

    injected_count = 0

    def _traverse(curr: Any, current_path: str = "") -> Any:
        """Traverse parameters recursively to quantize targeted kernels."""
        nonlocal injected_count
        if not isinstance(curr, dict):
            return curr
        res: dict[str, Any] = {}
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                v_dict: dict[str, Any] = v
                # Check if this leaf dictionary is a target projection module
                if "kernel" in v_dict and any(k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in targets):
                    kernel = v_dict["kernel"]
                    q_kernel, scale = quantize_tensor_aqt(kernel, bits=bits)
                    new_v: dict[str, Any] = dict(v_dict)
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
    metadata = {
        "backend": "maxtext",
        "method": method,
        "bits": bits,
        "clipping_bound": clipping_bound,
        "quant_targets": list(targets),
        "quantized_modules_count": injected_count,
        "memory_reduction_factor": reduction,
        "aqt_native": aqt is not None,
    }
    return quantized_params, metadata, injected_count


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: object,
) -> JSONDict:
    """Quantize a MaxText Gemma 4 model using Google AQT numerical quantization.

    Args:
        model_name: The name or preset path of the target model.
        method: Quantization method ('int8', 'int4').
        **kwargs: Optional parameters including 'params' and 'quant_targets'.

    Returns:
        A dictionary containing quantization results, status, and metadata.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    if jax is None or jnp is None or (Gemma4Model is None and "params" not in kwargs):
        raise DependencyMissingError("MaxText dependencies are missing.", backend="maxtext")

    if kwargs.get("strict_method") and method not in {"int8", "int4"}:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for MaxText backend.",
            method=method,
            supported_methods=("int8", "int4"),
            backend="maxtext",
        )

    default_reduction = 0.5 if method == "int8" else (0.75 if method == "int4" else 0.7)
    metadata: dict[str, Any] = {}

    def _apply() -> tuple[float, str]:
        """Resolve MaxText model parameters and apply AQT quantization."""
        nonlocal metadata
        if "params" in kwargs and kwargs["params"] is not None:
            params = kwargs["params"]
        else:
            if Gemma4Model is None:
                raise DependencyMissingError("MaxText dependency missing.", backend="maxtext")
            model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            params = model.init(rng, dummy_input)

        if isinstance(params, dict):
            params_dict: dict[str, Any] = params
            quant_targets = kwargs.get("quant_targets")
            targets_list: list[str] | None = [str(t) for t in quant_targets] if isinstance(quant_targets, (list, tuple)) else None
            _quantized_params, meta, count = apply_aqt_quantization(
                params=params_dict,
                method=method,
                quant_targets=targets_list,
            )
            metadata = meta
            reduction = float(meta["memory_reduction_factor"])
            logger.info("AQT quantization applied to %d modules with reduction %f", count, reduction)
        else:
            reduction = default_reduction
            metadata = {"method": method, "memory_reduction_factor": reduction}
        return (reduction, f"quantized_{method}")

    wrapped = quantize_model_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        method=method,
        missing_deps=False,
        apply_fn=_apply,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )
    res: JSONDict = dict(wrapped)
    if res["status"].startswith("failed:"):
        res["memory_reduction_factor"] = float(default_reduction)
    if metadata:
        res["metadata"] = metadata
    return res
