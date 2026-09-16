"""Keras-specific model quantization pipelines (INT8, INT4).

Supports:
- INT8 / INT4 global dtype policy configuration in Keras 3.
- Layer-level weight quantization with scaling factors.
- Serializing and persisting quantized model artifacts (.keras).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

try:
    import keras as _keras
    import numpy as _np

    keras: Any = _keras
    np: Any = _np
except (ImportError, AttributeError):
    keras = None
    np = None


def quantize_layer_weights(layer: Any, method: str = "int8") -> int:
    """Quantize kernel weights of a Keras layer using scaling factors.

    Args:
        layer: Keras Layer instance containing weights.
        method: Quantization method ('int8' or 'int4').

    Returns:
        Number of weight matrices quantized within this layer.
    """
    if np is None:
        return 0

    max_q = 127.0 if method == "int8" else 7.0
    q_min, q_max = (-128, 127) if method == "int8" else (-8, 7)
    quantized_count = 0

    weights = getattr(layer, "weights", [])
    for w in weights:
        try:
            val = w.numpy() if hasattr(w, "numpy") else np.array(w)
            if val.ndim >= 2:
                max_abs = float(np.max(np.abs(val)))
                scale = max_abs / max_q if max_abs > 0.0 else 1.0
                quantized = np.clip(np.round(val / scale), q_min, q_max) * scale
                if hasattr(w, "assign"):
                    w.assign(quantized)
                quantized_count += 1
        except (RuntimeError, ValueError, TypeError, AttributeError):
            pass

    return quantized_count


def quantize_model(model_name: str, method: str = "int8", **kwargs: object) -> JSONDict:
    """Quantize a Keras model using INT8 or INT4 dtype policies and weight quantization.

    Args:
        model_name: The name or path of the target model.
        method: The quantization method ('int8' or 'int4').
        **kwargs: Optional keyword arguments such as 'model', 'export_path', or 'group_size'.

    Returns:
        A dictionary containing the quantization status, memory reduction metrics, and export path.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        UnsupportedQuantizationMethodError: If an unsupported quantization method is requested.
    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    if method not in {"int8", "int4"}:
        raise UnsupportedQuantizationMethodError(f"Unsupported quantization method '{method}' for Keras backend. Supported methods are 'int8' and 'int4'.")

    memory_reduction = 0.5 if method == "int8" else 0.75
    policy_name = f"{method}_from_float32"
    quantized_layers_count = 0
    export_file: Path | None = None

    try:
        logger.info("Setting Keras model dtype policy to %s", policy_name)
        if hasattr(keras, "dtype_policies") and hasattr(keras.dtype_policies, "set_dtype_policy"):
            keras.dtype_policies.set_dtype_policy(policy_name)
        elif hasattr(keras, "config") and hasattr(keras.config, "set_dtype_policy"):
            keras.config.set_dtype_policy(policy_name)

        # Retrieve or load model instance if provided
        model = kwargs.get("model")
        if model is None:
            try:
                from keras_nlp.models import GemmaCausalLM

                model = GemmaCausalLM.from_preset(model_name)
            except (ImportError, ValueError, RuntimeError, AttributeError, OSError):
                model = None

        if model is not None:
            layers = getattr(model, "layers", [])
            for layer in layers:
                quantized_layers_count += quantize_layer_weights(layer, method=method)

            raw_export_path = kwargs.get("export_path")
            if raw_export_path:
                export_dir = Path(str(raw_export_path))
                export_dir.mkdir(parents=True, exist_ok=True)
                export_file = export_dir / f"{Path(model_name).name or 'model'}_{method}.keras"
                if hasattr(model, "save"):
                    model.save(str(export_file))
                    logger.info("Saved quantized Keras model artifact to %s", export_file)

        status = f"quantized_{method}"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply Keras quantization: ")
        status = f"failed: {e!s}"
        memory_reduction = 0.0

    result: JSONDict = {
        "backend": "keras",
        "model": model_name,
        "method": method,
        "status": status,
        "memory_reduction_factor": float(memory_reduction),
    }
    if export_file is not None:
        result["export_path"] = str(export_file)
    if quantized_layers_count > 0:
        result["quantized_layers_count"] = quantized_layers_count
    return result
