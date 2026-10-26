"""Common quantization utility for backends."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from gemma_4_sql.exceptions import (
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    GemmaSQLError,
    QuantizationError,
    UnsupportedQuantizationMethodError,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import ModelType, QuantizeResultDict, register_quantized_artifact

logger = logging.getLogger(__name__)


def apply_bits_and_bytes_quantization(
    method: str,
    bits_and_bytes_config_cls: type | None,
    float16_dtype: object = None,
    *,
    model: ModelType | None = None,
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

    """
    if bits_and_bytes_config_cls is None:
        msg = "bitsandbytes and transformers are required for BitsAndBytes quantization. Install with `pip install bitsandbytes transformers`."
        raise DependencyMissingError(msg, dependency_name="bitsandbytes", backend="pytorch")

    bnb_config: Any
    if method == "int8":
        bnb_kwargs: dict[str, Any] = {
            "load_in_8bit": True,
            "llm_int8_threshold": llm_int8_threshold,
        }
        if llm_int8_skip_modules is not None:
            bnb_kwargs["llm_int8_skip_modules"] = llm_int8_skip_modules
        bnb_config = bits_and_bytes_config_cls(**bnb_kwargs)
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

    if model is not None:
        model_any: Any = model
        if hasattr(model_any, "config") and model_any.config is not None:
            model_any.config.quantization_config = bnb_config
        register_quantized_artifact(model, method=method)

    return (memory_reduction, f"quantized_{method}")


def quantize_model_wrapper(
    backend_name: str,
    model_name: str,
    method: str,
    missing_deps: bool,
    apply_fn: Callable[..., tuple[float, str]],
    *,
    raise_on_error: bool | None = None,
    **kwargs: object,
) -> QuantizeResultDict:
    """Wrap and standardize execution and error handling for quantization.

    Args:
        backend_name: Name of the backend.
        model_name: Name of the model.
        method: The quantization method.
        missing_deps: Whether dependencies are missing.
        apply_fn: Function that applies quantization and returns `(memory_reduction, status)`.
        raise_on_error: Whether to raise `QuantizationError` on runtime failure.
        **kwargs: Additional keyword options inspected by `resolve_raise_on_error`.

    Returns:
        A dictionary containing the quantization results.

    Raises:
        DependencyMissingError: If `missing_deps` is True or raised by `apply_fn`.
        UnsupportedQuantizationMethodError: If raised by `apply_fn`.
        QuantizationError: If `raise_on_error` is True and `apply_fn` fails.

    """
    if missing_deps:
        raise DependencyMissingError(
            f"Dependencies missing for backend: {backend_name}",
            backend=backend_name,
        )

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
    except (GemmaSQLError, RuntimeError, ValueError, TypeError, OSError, LookupError, ArithmeticError, AttributeError, ImportError) as e:
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
        memory_reduction = 0.0

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
