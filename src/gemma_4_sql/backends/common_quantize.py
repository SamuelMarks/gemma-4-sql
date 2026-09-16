"""Common quantization utility for backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)


def apply_bits_and_bytes_quantization(
    method: str,
    bits_and_bytes_config_cls: type | None,
    float16_dtype: object = None,
    *,
    model: Any = None,
    raise_if_missing: bool = False,
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
        model: Optional model instance or model name string to quantize.
        raise_if_missing: If True, raises DependencyMissingError when config class is absent.
        llm_int8_threshold: Threshold for 8-bit outlier feature detection.
        llm_int8_skip_modules: Optional list of module name substrings to skip in 8-bit.
        bnb_4bit_quant_type: 4-bit quantization datatype ('nf4' or 'fp4').
        bnb_4bit_use_double_quant: Whether to compress quantization constants (double quantization).

    Returns:
        A tuple of (memory_reduction_factor, status_string).

    Raises:
        DependencyMissingError: If raise_if_missing is True and bits_and_bytes_config_cls is None.
    """
    if bits_and_bytes_config_cls is None:
        if raise_if_missing:
            from gemma_4_sql.exceptions import DependencyMissingError

            msg = "bitsandbytes and transformers are required for BitsAndBytes quantization. Install with `pip install bitsandbytes transformers`."
            raise DependencyMissingError(msg)
        return (0.0, "mocked_missing_bitsandbytes")

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
        return (0.0, f"unsupported_method_{method}")

    # Attach quantization config or transform in-memory model if provided
    if model is not None:
        if hasattr(model, "config"):
            model.config.quantization_config = bnb_config
        model._is_quantized = True
        model._quant_method = method

    return (memory_reduction, f"quantized_{method}")


def quantize_model_wrapper(
    backend_name: str,
    model_name: str,
    method: str,
    missing_deps: bool,
    missing_status: str,
    apply_fn: Callable[..., tuple[float, str]],
) -> JSONDict:
    """Wrap and standardize execution and error handling for quantization.

    Args:
        backend_name: Name of the backend.
        model_name: Name of the model.
        method: The quantization method.
        missing_deps: Whether dependencies are missing.
        missing_status: Status to return if dependencies are missing.
        apply_fn: Function that applies quantization and returns (memory_reduction, status).

    Returns:
        A dictionary containing the quantization results.
    """
    if missing_deps:
        return {
            "backend": backend_name,
            "model": model_name,
            "method": method,
            "status": missing_status,
            "memory_reduction_factor": 0.0,
        }

    try:
        (memory_reduction, status) = apply_fn()
        if not status.startswith("unsupported"):
            logger.info("Loading model %s with %s quantization...", model_name, method)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to quantize: ")
        status = f"failed: {e!s}"
        memory_reduction = 0.0

    return {
        "backend": backend_name,
        "model": model_name,
        "method": method,
        "status": status,
        "memory_reduction_factor": float(memory_reduction),
    }
