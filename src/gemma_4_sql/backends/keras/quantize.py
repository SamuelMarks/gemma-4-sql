"""Keras-specific model quantization pipelines (native Keras 3 INT8/INT4 and explicit simulated QAT).

Supports:
- Native Keras 3 `layer.quantize('int8' | 'int4')` / `model.quantize('int8' | 'int4')` storing genuine
  `int8` / packed 4-bit `int8` weight variables and `float32` `kernel_scale` variables.
- Explicit `simulated=True` quantization-aware dequantization reporting `memory_reduction_factor = 0.0`
  and `status = 'simulated_quantized_int8' | 'simulated_quantized_int4'`.
- Fail-fast `QuantizationError` when model preset loading fails or returns `None`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    QuantizationError,
    UnsupportedQuantizationMethodError,
)
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    CausalLMProtocol,
    DTypeProtocol,
    KerasCausalLMProtocol,
    KerasLayerProtocol,
    KerasVariableProtocol,
    QuantizeResultDict,
    TensorType,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol | None:
        """Load a `GemmaCausalLM` model from a preset identifier.

        Args:
            model_name: Preset identifier string.

        Returns:
            Loaded `KerasCausalLMProtocol` or `None`.

        """
        ...


@runtime_checkable
class _NativeQuantizableProtocol(Protocol):
    """Protocol for Keras 3 models or layers exposing native `.quantize(mode)`."""

    def quantize(self, mode: str, /) -> None:
        """Quantize weights in-place to genuine `int8` or packed `int4` variables.

        Args:
            mode: Quantization mode (`'int8'` or `'int4'`).

        """
        ...


@runtime_checkable
class _KerasTensorShapeProtocol(Protocol):
    """Protocol for Keras tensors exposing `.shape`."""

    @property
    def shape(self) -> Sequence[int | None] | None:
        """Provide tensor shape."""
        ...


@runtime_checkable
class _KerasLayersContainerProtocol(Protocol):
    """Protocol for any Keras model exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol | _WeightsOwnerProtocol]:
        """Provide the model's layers.

        Returns:
            Sequence of Keras layers or weight owners.

        """
        ...


@runtime_checkable
class _WeightsOwnerProtocol(Protocol):
    """Protocol for a layer exposing `.weights`."""

    @property
    def weights(self) -> Sequence[KerasVariableProtocol | _KerasTensorShapeProtocol | str]:
        """Provide layer weight variables.

        Returns:
            Sequence of weight variables.

        """
        ...


@runtime_checkable
class _SavableModelProtocol(Protocol):
    """Protocol for a Keras model supporting `.save(filepath)`."""

    def save(self, filepath: str, /) -> None:
        """Save quantized model artifact to `filepath`.

        Args:
            filepath: Output `.keras` file path.

        """
        ...


@runtime_checkable
class _KerasConfigNamespaceProtocol(Protocol):
    """Protocol for `keras.config` exposing `dtype_policy()` and `set_dtype_policy(...)`."""

    def dtype_policy(self) -> DTypeProtocol | str | None:
        """Return current global Keras dtype policy."""
        ...

    def set_dtype_policy(self, policy: DTypeProtocol | str, /) -> None:
        """Set global Keras dtype policy."""
        ...


@runtime_checkable
class _KerasWithConfigProtocol(Protocol):
    """Protocol for `keras` module exposing `.config`."""

    @property
    def config(self) -> _KerasConfigNamespaceProtocol:
        """Provide `keras.config` namespace."""
        ...


@runtime_checkable
class _KerasDTypePoliciesNamespaceProtocol(Protocol):
    """Protocol for `keras.dtype_policies` exposing `set_dtype_policy(...)`."""

    def set_dtype_policy(self, policy: DTypeProtocol | str, /) -> None:
        """Set global Keras dtype policy."""
        ...


@runtime_checkable
class _KerasWithDTypePoliciesProtocol(Protocol):
    """Protocol for `keras` module exposing `.dtype_policies`."""

    @property
    def dtype_policies(self) -> _KerasDTypePoliciesNamespaceProtocol:
        """Provide `keras.dtype_policies` namespace."""
        ...


@runtime_checkable
class _QuantOpsTensorProtocol(Protocol):
    """Protocol for a Keras ops tensor supporting `.shape`, scalar `float()`, `/` and `*` scaling."""

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Provide tensor dimensions."""
        ...

    def __float__(self) -> float:
        """Convert 0D scalar tensor to Python `float`."""
        ...

    def __truediv__(self, denom: float, /) -> _QuantOpsTensorProtocol:
        """Divide tensor elements by scalar `denom`."""
        ...

    def __mul__(self, scale: float, /) -> TensorType:
        """Multiply tensor elements by scalar `scale`.

        Args:
            scale: Scalar quantization scale factor.

        Returns:
            Dequantized `TensorType` tensor.

        """
        ...


@runtime_checkable
class _KerasOpsQuantizeProtocol(Protocol):
    """Protocol for `keras.ops` math operations used during simulated weight quantization."""

    def stop_gradient(
        self,
        variable: KerasVariableProtocol | _KerasTensorShapeProtocol | str,
        /,
    ) -> _QuantOpsTensorProtocol:
        """Detach `variable` from gradient tracking.

        Args:
            variable: Keras weight variable or tensor.

        Returns:
            Detached `_QuantOpsTensorProtocol`.

        """
        ...

    def abs(self, x: _QuantOpsTensorProtocol, /) -> _QuantOpsTensorProtocol:
        """Compute elementwise absolute value.

        Args:
            x: Input tensor.

        Returns:
            Magnitude tensor.

        """
        ...

    def max(self, x: _QuantOpsTensorProtocol, /) -> SupportsFloat:
        """Compute maximum scalar element across `x`.

        Args:
            x: Input tensor.

        Returns:
            Scalar maximum convertible via `float(...)`.

        """
        ...

    def round(self, x: _QuantOpsTensorProtocol, /) -> _QuantOpsTensorProtocol:
        """Round tensor elements to nearest integers.

        Args:
            x: Scaled floating-point tensor.

        Returns:
            Rounded tensor.

        """
        ...

    def clip(self, x: _QuantOpsTensorProtocol, x_min: float, x_max: float, /) -> _QuantOpsTensorProtocol:
        """Clamp tensor elements into `[x_min, x_max]`.

        Args:
            x: Rounded tensor.
            x_min: Lower bound.
            x_max: Upper bound.

        Returns:
            Clamped tensor.

        """
        ...

    def cast(self, x: _QuantOpsTensorProtocol, dtype: str, /) -> _QuantOpsTensorProtocol:
        """Cast tensor `x` to `dtype`.

        Args:
            x: Input tensor.
            dtype: Target dtype name (`'int8'` or `'float32'`).

        Returns:
            Cast `_QuantOpsTensorProtocol`.

        """
        ...


keras = optional_import("keras")
ops: _KerasOpsQuantizeProtocol | None = optional_import("keras.ops", _KerasOpsQuantizeProtocol)

type _QuantizeLayerInput = (
    KerasLayerProtocol | _KerasLayersContainerProtocol | _WeightsOwnerProtocol | CausalLMProtocol | str | None
)


def quantize_layer_weights(
    layer: _QuantizeLayerInput,
    method: str = "int8",
    *,
    simulated: bool = False,
) -> int:
    """Quantize kernel weights of a Keras layer into native `int8`/`int4` variables or simulated QAT.

    Uses Keras 3 native `layer.quantize(method)` when available (creating genuine `int8` / packed
    4-bit `int8` kernel variables and `float32` `kernel_scale` variables), falling back to explicit
    simulated quantization when `simulated=True` or on custom layer containers without `.quantize`.

    Args:
        layer: Keras `Layer` instance containing weights.
        method: Quantization method (`'int8'` or `'int4'`).
        simulated: Whether to perform simulated float32 quantize-dequantize instead of native integer storage.

    Returns:
        Number of weight matrices quantized within this layer.

    """
    if ops is None or layer is None or isinstance(layer, str):
        return 0

    if not simulated and isinstance(layer, _NativeQuantizableProtocol) and callable(layer.quantize):
        try:
            layer.quantize(method)
            register_quantized_artifact(layer, method=method)
            return 1
        except (NotImplementedError, ValueError, TypeError, RuntimeError):
            pass

    if not isinstance(layer, _WeightsOwnerProtocol):
        return 0

    max_q = 127.0 if method == "int8" else 7.0
    q_min, q_max = (-128.0, 127.0) if method == "int8" else (-8.0, 7.0)
    quantized_count = 0
    layer_scales: list[float] = []

    for w in layer.weights:
        try:
            val = ops.stop_gradient(w)
            if len(val.shape) >= 2:
                max_abs = float(ops.max(ops.abs(val)))
                scale = max_abs / max_q if max_abs > 0.0 else 1.0
                q_int = ops.cast(ops.clip(ops.round(val / scale), q_min, q_max), "int8")
                quantized = ops.cast(q_int, "float32") * scale
                if isinstance(w, KerasVariableProtocol) and callable(w.assign):
                    w.assign(quantized)
                layer_scales.append(scale)
                quantized_count += 1
        except (RuntimeError, ValueError, TypeError, AttributeError):
            pass

    if quantized_count > 0:
        register_quantized_artifact(
            layer,
            method=f"simulated_{method}",
            scales={f"w_{i}": s for i, s in enumerate(layer_scales)},
        )

    return quantized_count


def _get_current_keras_policy() -> DTypeProtocol | str | None:
    """Retrieve the active global Keras dtype policy so it can be restored after quantization.

    Returns:
        Current Keras dtype policy object or string name, or `None` if unavailable.

    """
    if isinstance(keras, _KerasWithConfigProtocol) and callable(keras.config.dtype_policy):
        res: DTypeProtocol | str | None = keras.config.dtype_policy()
        return res
    return None


def _set_keras_policy(policy: DTypeProtocol | str) -> None:
    """Set the global Keras dtype policy via `keras.config` or `keras.dtype_policies`.

    Args:
        policy: Policy string or `DTypePolicy` object.

    """
    if isinstance(keras, _KerasWithConfigProtocol) and callable(keras.config.set_dtype_policy):
        keras.config.set_dtype_policy(policy)
        return
    if isinstance(keras, _KerasWithDTypePoliciesProtocol) and callable(keras.dtype_policies.set_dtype_policy):
        keras.dtype_policies.set_dtype_policy(policy)


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue
    | KerasCausalLMProtocol
    | KerasLayerProtocol
    | _KerasLayersContainerProtocol
    | _NativeQuantizableProtocol
    | _SavableModelProtocol,
) -> QuantizeResultDict:
    """Quantize a Keras model into genuine `int8`/`int4` weights or explicit simulated QAT.

    Fails fast with `QuantizationError` when no in-memory `model` is provided and
    `GemmaCausalLM.from_preset(model_name)` fails or returns `None`.

    Args:
        model_name: The name or path of the target model.
        method: The quantization method (`'int8'` or `'int4'`).
        **kwargs: Optional keyword arguments (`'model'`, `'export_path'`, `'simulated'`, `'raise_on_error'`).

    Returns:
        A dictionary containing the quantization status, memory reduction metrics, and export path.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        UnsupportedQuantizationMethodError: If an unsupported quantization method is requested.
        QuantizationError: If `raise_on_error=True` and model loading or quantization fails.

    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.", backend="keras")

    if method not in {"int8", "int4"}:
        raise UnsupportedQuantizationMethodError(
            f"Unsupported quantization method '{method}' for Keras backend. Supported methods are 'int8' and 'int4'.",
            method=method,
            supported_methods=("int8", "int4"),
            backend="keras",
        )

    simulated = bool(kwargs.get("simulated"))
    target_reduction = 0.5 if method == "int8" else 0.75
    policy_name = f"{method}_from_float32"
    quantized_layers_count = 0
    export_file: Path | None = None

    def _apply() -> tuple[float, str]:
        """Load or validate Keras model, quantize weights, and optionally save `.keras` artifact.

        Returns:
            Tuple of `(memory_reduction_factor, quantization_status)`.

        Raises:
            QuantizationError: If `GemmaCausalLM` is unavailable or preset loading fails.

        """
        nonlocal quantized_layers_count, export_file
        prev_policy = _get_current_keras_policy()
        logger.info("Setting Keras model dtype policy to %s", policy_name)
        _set_keras_policy(policy_name)
        try:
            model = kwargs.get("model")
            if model is None:
                try:
                    from gemma_4_sql.backends.keras._runtime import (
                        load_keras_gemma_model,
                        load_preset_with_hf_fallback,
                    )

                    gemma_causal_lm_cls = load_keras_gemma_model(
                        model_name,
                        _GemmaCausalLMLoaderProtocol,
                        symbol_importer=import_optional_symbol,
                    )
                    if gemma_causal_lm_cls is None:
                        raise QuantizationError(
                            f"keras_hub / keras_nlp GemmaCausalLM is unavailable to load '{model_name}'.",
                            backend="keras",
                            model_name=model_name,
                        )
                    model = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
                except QuantizationError:
                    raise
                except (ImportError, ValueError, RuntimeError, AttributeError, OSError) as err:
                    raise QuantizationError(
                        f"Failed to load Keras model '{model_name}' for quantization: {err}",
                        backend="keras",
                        model_name=model_name,
                        cause=err,
                    ) from err

                if model is None:
                    raise QuantizationError(
                        f"GemmaCausalLM.from_preset('{model_name}') returned None.",
                        backend="keras",
                        model_name=model_name,
                    )

            native_used = False
            layers: Sequence[KerasLayerProtocol | _WeightsOwnerProtocol] = (
                model.layers if isinstance(model, _KerasLayersContainerProtocol) else ()
            )
            for layer in layers:
                before_native = not simulated and isinstance(layer, _NativeQuantizableProtocol)
                added = quantize_layer_weights(layer, method=method, simulated=simulated)
                quantized_layers_count += added
                if before_native and added > 0:
                    native_used = True

            if (
                not simulated
                and quantized_layers_count == 0
                and isinstance(model, _NativeQuantizableProtocol)
                and callable(model.quantize)
            ):
                model.quantize(method)
                native_used = True
                quantized_layers_count = 1

            register_quantized_artifact(model, method=f"simulated_{method}" if simulated else method)

            raw_export_path = kwargs.get("export_path")
            if raw_export_path:
                export_dir = Path(str(raw_export_path))
                export_dir.mkdir(parents=True, exist_ok=True)
                export_file = export_dir / f"{Path(model_name).name or 'model'}_{method}.keras"
                if isinstance(model, _SavableModelProtocol) and callable(model.save):
                    model.save(str(export_file))
                    logger.info("Saved quantized Keras model artifact to %s", export_file)

            if simulated:
                return (0.0, f"simulated_quantized_{method}")
            if not native_used and len(layers) > 0 and quantized_layers_count > 0:
                return (0.0, f"simulated_quantized_{method}")
            return (target_reduction, f"quantized_{method}")
        finally:
            if prev_policy is not None:
                _set_keras_policy(prev_policy)

    from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend

    backend_alias = str(kwargs.get("backend_alias", "keras"))
    explicit_kb = (
        str(kwargs["keras_backend"]) if "keras_backend" in kwargs and kwargs["keras_backend"] is not None else None
    )
    active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)
    wrapped = quantize_model_wrapper(
        backend_name=backend_alias,
        model_name=model_name,
        method=method,
        apply_fn=_apply,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )
    wrapped["keras_backend"] = active_kb
    if export_file is not None:
        wrapped["export_path"] = str(export_file)
    if quantized_layers_count > 0:
        wrapped["quantized_layers_count"] = quantized_layers_count
    return wrapped
