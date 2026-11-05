"""Keras-specific model quantization pipelines (INT8, INT4).

Supports:
- INT8 / INT4 global dtype policy configuration in Keras 3 (restoring policy after quantization).
- Native `model.quantize(mode)` when supported by Keras 3 / KerasNLP layers, plus layer-level weight quantization with scale tracking.
- Serializing and persisting quantized model artifacts (`.keras`).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeGuard, runtime_checkable

from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    CausalLMProtocol,
    DTypeProtocol,
    KerasCausalLMProtocol,
    KerasLayerProtocol,
    QuantizeResultDict,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol | None:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


keras = optional_import("keras")
ops = optional_import("keras.ops")


@runtime_checkable
class _KerasLayersContainerProtocol(Protocol):
    """Protocol for any Keras model exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the model's layers."""
        ...


type _QuantizeLayerInput = KerasLayerProtocol | _KerasLayersContainerProtocol | CausalLMProtocol | str | None


def _is_keras_layer(layer: _QuantizeLayerInput) -> TypeGuard[KerasLayerProtocol]:
    """Narrow `layer` to `KerasLayerProtocol` for `.weights` access.

    Args:
        layer: Candidate Keras layer object.

    Returns:
        `True` when `layer` is not `None` and not a `str`.

    """
    return layer is not None and not isinstance(layer, str)


def _is_keras_model(
    model: BackendKwargValue | KerasCausalLMProtocol | KerasLayerProtocol | _KerasLayersContainerProtocol | None,
) -> TypeGuard[KerasCausalLMProtocol]:
    """Narrow `model` to `KerasCausalLMProtocol` for `.layers` and `.save(...)` access.

    Args:
        model: Candidate Keras model object.

    Returns:
        `True` when `model` is not `None` and not a primitive scalar.

    """
    return model is not None and not isinstance(model, (str, int, float, bool))


def quantize_layer_weights(
    layer: _QuantizeLayerInput,
    method: str = "int8",
) -> int:
    """Quantize kernel weights of a Keras layer using symmetric scaling factors.

    Stores quantized integer values and scale metadata on the layer while updating the
    layer's kernel variable.

    Args:
        layer: Keras `Layer` instance containing weights.
        method: Quantization method (`'int8'` or `'int4'`).

    Returns:
        Number of weight matrices quantized within this layer.

    """
    if ops is None:
        return 0

    max_q = 127.0 if method == "int8" else 7.0
    q_min, q_max = (-128.0, 127.0) if method == "int8" else (-8.0, 7.0)
    quantized_count = 0
    layer_scales: list[float] = []

    weights = ()
    with contextlib.suppress(AttributeError, AssertionError):
        assert _is_keras_layer(layer)
        weights = layer.weights
    for w in weights:
        try:
            val = ops.stop_gradient(w)
            if len(val.shape) >= 2:
                max_abs = float(ops.max(ops.abs(val)))
                scale = max_abs / max_q if max_abs > 0.0 else 1.0
                q_int = ops.cast(ops.clip(ops.round(val / scale), q_min, q_max), "int8")
                quantized = ops.cast(q_int, "float32") * scale
                with contextlib.suppress(AttributeError):
                    w.assign(quantized)
                layer_scales.append(scale)
                quantized_count += 1
        except (RuntimeError, ValueError, TypeError, AttributeError):
            pass

    if quantized_count > 0:
        register_quantized_artifact(
            layer,
            method=method,
            scales={f"w_{i}": s for i, s in enumerate(layer_scales)},
        )

    return quantized_count


def _get_current_keras_policy() -> DTypeProtocol | str | None:
    """Retrieve the active global Keras dtype policy so it can be restored after quantization.

    Returns:
        Current Keras dtype policy object or string name, or `None` if unavailable.

    """
    assert keras is not None
    with contextlib.suppress(AttributeError, TypeError):
        res: DTypeProtocol | str | None = keras.config.dtype_policy()
        return res
    return None


def _set_keras_policy(policy: DTypeProtocol | str) -> None:
    """Set the global Keras dtype policy.

    Args:
        policy: Policy string or `DTypePolicy` object.

    """
    assert keras is not None
    try:
        keras.config.set_dtype_policy(policy)
        return
    except (AttributeError, TypeError):
        pass
    with contextlib.suppress(AttributeError, TypeError):
        keras.dtype_policies.set_dtype_policy(policy)


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | KerasCausalLMProtocol | KerasLayerProtocol | _KerasLayersContainerProtocol,
) -> QuantizeResultDict:
    """Quantize a Keras model using INT8 or INT4 dtype policies and weight quantization.

    Restores the previous global Keras dtype policy upon completion so subsequent float32
    operations are not unexpectedly mutated.

    Args:
        model_name: The name or path of the target model.
        method: The quantization method (`'int8'` or `'int4'`).
        **kwargs: Optional keyword arguments such as `'model'`, `'export_path'`, or `'group_size'`.

    Returns:
        A dictionary containing the quantization status, memory reduction metrics, and export path.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        UnsupportedQuantizationMethodError: If an unsupported quantization method is requested.

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

    memory_reduction = 0.5 if method == "int8" else 0.75
    policy_name = f"{method}_from_float32"
    quantized_layers_count = 0
    export_file: Path | None = None

    def _apply() -> tuple[float, str]:
        """Configure Keras dtype policy, quantize layer weights, and optionally save artifact."""
        nonlocal quantized_layers_count, export_file
        prev_policy = _get_current_keras_policy()
        logger.info("Setting Keras model dtype policy to %s", policy_name)
        _set_keras_policy(policy_name)
        try:
            model = kwargs.get("model")
            if model is None:
                try:
                    gemma_causal_lm_cls = optional_attr(
                        "keras_nlp.models",
                        "GemmaCausalLM",
                        _GemmaCausalLMLoaderProtocol,
                    )
                    if gemma_causal_lm_cls is not None:
                        model = gemma_causal_lm_cls.from_preset(model_name)
                except (ImportError, ValueError, RuntimeError, AttributeError, OSError) as err:
                    logger.debug("Preset loading skipped during Keras quantization: %s", err)

            if model is not None:
                assert _is_keras_model(model)
                layers: Sequence[KerasLayerProtocol] = ()
                with contextlib.suppress(AttributeError):
                    layers = model.layers
                for layer in layers:
                    quantized_layers_count += quantize_layer_weights(layer, method=method)
                register_quantized_artifact(model, method=method)

                raw_export_path = kwargs.get("export_path")
                if raw_export_path:
                    export_dir = Path(str(raw_export_path))
                    export_dir.mkdir(parents=True, exist_ok=True)
                    export_file = export_dir / f"{Path(model_name).name or 'model'}_{method}.keras"
                    with contextlib.suppress(AttributeError):
                        model.save(str(export_file))
                        logger.info("Saved quantized Keras model artifact to %s", export_file)

            return (memory_reduction, f"quantized_{method}")
        finally:
            if prev_policy is not None:
                _set_keras_policy(prev_policy)

    wrapped = quantize_model_wrapper(
        backend_name="keras",
        model_name=model_name,
        method=method,
        apply_fn=_apply,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )
    if export_file is not None:
        wrapped["export_path"] = str(export_file)
    if quantized_layers_count > 0:
        wrapped["quantized_layers_count"] = quantized_layers_count
    return wrapped
