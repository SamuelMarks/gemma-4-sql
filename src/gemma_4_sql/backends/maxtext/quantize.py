"""MaxText-specific model quantization logic using Google AQT (Accurate Quantized Training).

Implements:
- INT8 and INT4 numerical quantization for weight and activation matrices.
- Dynamic per-channel scale factor computation: `scale = max(|W|, axis=-1) / clipping_bound`.
- Uniform symmetric quantization with clipping bounds:
    * INT8: `[-128, 127]` with clipping bound `127.0` (50% memory reduction).
    * INT4: `[-8, 7]` with clipping bound `7.0` (75% memory reduction).
- Selective projection targeting across attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
  and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`).
- Explicit `QuantizedModelArtifact` storage and optional Orbax persistence so quantized
  PyTrees are preserved for downstream `export_model` and `generate_sql` execution.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, overload, runtime_checkable

from gemma_4_sql._types_configs import QuantizedModelArtifact, register_quantized_artifact
from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql.backends.common_inference import BoundedLRUCache
from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import JaxPRNGHolderProtocol, JnpZerosInt32Protocol
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    DTypeProtocol,
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


@runtime_checkable
class _StringableProtocol(Protocol):
    """Protocol for any value supporting `str(...)`."""

    def __str__(self) -> str:
        """Format value as string.

        Returns:
            String representation.

        """
        ...


@runtime_checkable
class _AQTConfigNamespaceProtocol(Protocol):
    """Protocol for `aqt.jax.v2.config` exposing `.fully_quantized(...)`."""

    def fully_quantized(self, *, fwd_bits: int, bwd_bits: int) -> _StringableProtocol:
        """Create an AQT dot-general configuration for `fwd_bits` and `bwd_bits`.

        Args:
            fwd_bits: Forward quantization bit width.
            bwd_bits: Backward quantization bit width.

        Returns:
            AQT configuration object.

        """
        ...


@runtime_checkable
class _AQTModuleProtocol(Protocol):
    """Protocol for `aqt.jax.v2` exposing `.config`."""

    @property
    def config(self) -> _AQTConfigNamespaceProtocol:
        """Provide the AQT configuration namespace.

        Returns:
            `_AQTConfigNamespaceProtocol` object.

        """
        ...


@runtime_checkable
class _AQTItemsMappingProtocol(Protocol):
    """Non-generic protocol for iterating `(key, value)` pairs of AQT parameter mappings."""

    def items(self) -> Iterable[tuple[str, BackendKwargValue | AQTParamDict | PyTreeNode | None]]:
        """Yield `(key, value)` pairs."""
        ...


def _coerce_aqt_param_dict(
    raw: BackendKwargValue | AQTParamDict | Sequence[str] | PyTreeNode | None,
) -> AQTParamDict | None:
    """Recursively coerce a mapping into a typed `AQTParamDict`.

    Args:
        raw: Candidate parameter dictionary or scalar.

    Returns:
        Coerced `AQTParamDict`, or `None` when `raw` is not a `Mapping`.

    """
    if not isinstance(raw, Mapping):
        return None
    assert isinstance(raw, _AQTItemsMappingProtocol)
    out: AQTParamDict = {}
    for k, v in raw.items():
        key_s = str(k)
        sub = _coerce_aqt_param_dict(v)
        if sub is not None:
            out[key_s] = sub
        elif isinstance(v, (JaxArrayProtocol, SupportsFloat, str, int, float)):
            out[key_s] = v
    return out


@runtime_checkable
class _JnpQuantizeModuleProtocol(JnpZerosInt32Protocol, Protocol):
    """Protocol for `jax.numpy` symmetric integer quantization and dtype operations."""

    @property
    def int8(self) -> DTypeProtocol:
        """Provide `jnp.int8` dtype."""
        ...

    @property
    def int16(self) -> DTypeProtocol:
        """Provide `jnp.int16` dtype."""
        ...

    def abs(self, x: JaxArrayProtocol | TensorType | SupportsFloat, /) -> JaxArrayProtocol:
        """Compute elementwise absolute value.

        Args:
            x: Input tensor.

        Returns:
            Magnitude `JaxArrayProtocol`.

        """
        ...

    def max(self, a: JaxArrayProtocol, *, axis: int = -1, keepdims: bool = False) -> JaxArrayProtocol:
        """Compute maximum along `axis`.

        Args:
            a: Input tensor.
            axis: Reduction axis index.
            keepdims: Whether to keep reduced dimension of size 1.

        Returns:
            Reduced maximum `JaxArrayProtocol`.

        """
        ...

    def maximum(self, x1: JaxArrayProtocol, x2: float, /) -> JaxArrayProtocol:
        """Compute elementwise `max(x1, x2)`.

        Args:
            x1: Input scale array.
            x2: Minimum epsilon bound.

        Returns:
            Clamped scale `JaxArrayProtocol`.

        """
        ...

    def divide(
        self,
        x1: JaxArrayProtocol | TensorType | SupportsFloat,
        x2: JaxArrayProtocol,
        /,
    ) -> JaxArrayProtocol:
        """Divide `x1` by `x2` elementwise.

        Args:
            x1: Numerator tensor.
            x2: Scale denominator tensor.

        Returns:
            Scaled `JaxArrayProtocol`.

        """
        ...

    def clip(self, a: JaxArrayProtocol, a_min: float, a_max: float, /) -> JaxArrayProtocol:
        """Clamp `a` into `[a_min, a_max]`.

        Args:
            a: Scaled tensor.
            a_min: Lower bound.
            a_max: Upper bound.

        Returns:
            Clamped `JaxArrayProtocol`.

        """
        ...

    def round(self, a: JaxArrayProtocol, /) -> JaxArrayProtocol:
        """Round elements to nearest integers.

        Args:
            a: Clamped floating-point tensor.

        Returns:
            Rounded `JaxArrayProtocol`.

        """
        ...


jax: JaxPRNGHolderProtocol | None = optional_import("jax", JaxPRNGHolderProtocol)
jnp: _JnpQuantizeModuleProtocol | None = optional_import("jax.numpy", _JnpQuantizeModuleProtocol)
aqt: _AQTModuleProtocol | ModuleNamespaceProtocol | None = optional_import("aqt.jax.v2")
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)

DEFAULT_AQT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_QUANTIZED_ARTIFACT_CACHE: BoundedLRUCache[QuantizedModelArtifact[AQTParamDict]] = BoundedLRUCache(maxsize=8)


def get_quantized_maxtext_artifact(model_name: str) -> QuantizedModelArtifact[AQTParamDict] | None:
    """Retrieve the cached `QuantizedModelArtifact` for `model_name` if available.

    Args:
        model_name: Model identifier passed to `quantize_model`.

    Returns:
        Cached `QuantizedModelArtifact[AQTParamDict]`, or `None` if absent.

    """
    return _QUANTIZED_ARTIFACT_CACHE.get_model(model_name)


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


@overload
def apply_aqt_quantization(
    params: AQTParamDict,
    method: str = "int8",
    quant_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[AQTParamDict, JSONDict, int]:
    """Apply Google AQT numerical quantization to an `AQTParamDict`.

    Args:
        params: Model parameter PyTree dictionary.
        method: Quantization format (`'int8'`, `'int4'`, `'aqt'`).
        quant_targets: Submodule names to quantize.

    Returns:
        A tuple of `(quantized_params, quantization_metadata, count_of_quantized_modules)`.

    """


@overload
def apply_aqt_quantization(
    params: AQTParamDict | AQTParamLeaf,
    method: str = "int8",
    quant_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[AQTParamDict | AQTParamLeaf, JSONDict, int]:
    """Apply Google AQT numerical quantization to a parameter PyTree or leaf.

    Args:
        params: Model parameter PyTree or leaf.
        method: Quantization format (`'int8'`, `'int4'`, `'aqt'`).
        quant_targets: Submodule names to quantize.

    Returns:
        A tuple of `(quantized_params, quantization_metadata, count_of_quantized_modules)`.

    """


def apply_aqt_quantization(
    params: AQTParamDict | AQTParamLeaf,
    method: str = "int8",
    quant_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[AQTParamDict | AQTParamLeaf, JSONDict, int]:
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
    if isinstance(aqt, _AQTModuleProtocol):
        aqt_native_cfg = str(aqt.config.fully_quantized(fwd_bits=bits, bwd_bits=bits))

    injected_count = 0

    def _traverse(curr: AQTParamDict | AQTParamLeaf, current_path: str = "") -> AQTParamDict | AQTParamLeaf:
        """Traverse parameters recursively to quantize targeted kernels.

        Args:
            curr: Current parameter dictionary or leaf.
            current_path: Dotted key path prefix.

        Returns:
            Quantized parameter dictionary or leaf.

        """
        nonlocal injected_count
        if not isinstance(curr, dict):
            return curr
        res: AQTParamDict = {}
        for k, v in curr.items():
            sub_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, dict):
                v_dict: AQTParamDict = v
                raw_kernel = v_dict.get("kernel")
                if (
                    raw_kernel is not None
                    and not isinstance(raw_kernel, (dict, str))
                    and any(k == t or sub_path.endswith(f".{t}") or f".{t}." in sub_path for t in targets)
                ):
                    q_kernel, scale = quantize_tensor_aqt(raw_kernel, bits=bits)
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
    return quantized_params, metadata, injected_count


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | AQTParamDict | Sequence[str],
) -> QuantizeResultDict:
    """Quantize a MaxText Gemma 4 model using Google AQT, caching and optionally exporting the quantized PyTree.

    Args:
        model_name: The name or preset path of the target model.
        method: Quantization method (`'int8'`, `'int4'`, `'aqt'`).
        **kwargs: Optional parameters including `'params'`, `'quant_targets'`, and `'export_path'`.

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
    exported_ckpt_path: str | None = None

    def _apply() -> tuple[float, str]:
        """Resolve MaxText model parameters, apply AQT quantization, and persist/cache quantized PyTree.

        Returns:
            Tuple of `(memory_reduction_factor, quantization_status)`.

        """
        nonlocal exported_ckpt_path
        from gemma_4_sql.backends.maxtext.export import export_model
        from gemma_4_sql.backends.maxtext.inference import cache_maxtext_params, get_cached_maxtext_params

        assert jax is not None
        assert jnp is not None
        loaded_model: MaxTextCausalLMProtocol | None = None
        if has_params:
            raw_params = kwargs["params"]
        else:
            cached_params = get_cached_maxtext_params(model_name)
            if cached_params is not None:
                raw_params = cached_params
            else:
                assert Gemma4Model is not None
                cand_model = Gemma4Model(model_name)
                rng = jax.random.PRNGKey(0)
                dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
                raw_params = dict(cand_model.init(rng, dummy_input))
                loaded_model = cand_model

        params_dict = _coerce_aqt_param_dict(raw_params)
        if params_dict is not None:
            quant_targets = kwargs.get("quant_targets")
            targets_list: list[str] | None = (
                [str(t) for t in quant_targets] if isinstance(quant_targets, (list, tuple)) else None
            )
            quantized_params, meta, count = apply_aqt_quantization(
                params=params_dict,
                method=method,
                quant_targets=targets_list,
            )
            artifact = register_quantized_artifact(
                quantized_params,
                method=method,
                scales={"quantized_modules_count": float(count)},
            )
            _QUANTIZED_ARTIFACT_CACHE.put_model(model_name, artifact)
            cache_maxtext_params(model_name, quantized_params, model=loaded_model)

            raw_export_path = kwargs.get("export_path")
            if isinstance(raw_export_path, str) and raw_export_path:
                exp_res = export_model(model_name, raw_export_path, params=quantized_params)
                exported_ckpt_path = str(exp_res.get("file_path", raw_export_path))

            metadata.update(meta)
            raw_red = meta.get("memory_reduction_factor", default_reduction)
            reduction = float(raw_red) if isinstance(raw_red, (int, float)) else default_reduction
            logger.info("AQT quantization applied to %d modules with reduction %f", count, reduction)
        else:
            reduction = default_reduction
            metadata.update({"method": method, "memory_reduction_factor": reduction})
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
        wrapped["metadata"] = metadata
    if exported_ckpt_path is not None:
        wrapped["export_path"] = exported_ckpt_path
    return wrapped


__all__ = [
    "DEFAULT_AQT_TARGETS",
    "AQTParamDict",
    "AQTParamLeaf",
    "Gemma4Model",
    "PyTreeNode",
    "apply_aqt_quantization",
    "get_quantized_maxtext_artifact",
    "quantize_model",
    "quantize_tensor_aqt",
]
