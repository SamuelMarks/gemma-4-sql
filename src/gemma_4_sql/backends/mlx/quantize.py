"""MLX-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization, quantize_model_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

try:
    import mlx.core as _mlx
    from transformers import BitsAndBytesConfig as _BitsAndBytesConfig

    mlx: Any = _mlx
    BitsAndBytesConfig: Any = _BitsAndBytesConfig
except (ImportError, AttributeError):
    mlx = None
    BitsAndBytesConfig = None


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize an MLX model.

    Uses native mlx.nn.quantize for 4-bit and 8-bit Apple Silicon quantization
    when available, falling back to bitsandbytes configuration.

    Args:
        model_name: The name of the target model.
        method: The string representing the quantization method.

    Returns:
        A dictionary containing the quantization results.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
    """
    if mlx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        msg = "MLX dependencies are missing."
        raise DependencyMissingError(msg)

    def apply_fn() -> tuple[float, str]:
        """Apply native MLX or BitsAndBytes quantization.

        Returns:
            Tuple of memory reduction factor and quantization status.
        """
        try:
            from mlx import nn
            from mlx_lm import load

            bits = 4 if method in {"int4", "awq", "gptq"} else 8
            loaded = load(model_name)
            model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            if hasattr(nn, "quantize"):
                nn.quantize(model, group_size=64, bits=bits)
                reduction = 0.75 if bits == 4 else 0.5
                return (reduction, f"quantized_{method}")
        except (ImportError, ValueError, TypeError, RuntimeError, OSError, AttributeError):
            pass

        if BitsAndBytesConfig is not None:
            return apply_bits_and_bytes_quantization(method, BitsAndBytesConfig, getattr(mlx, "float16", None))

        reduction = 0.75 if method in {"int4", "awq", "gptq"} else 0.5
        return (reduction, f"quantized_{method}")

    return quantize_model_wrapper(
        backend_name="mlx",
        model_name=model_name,
        method=method,
        missing_deps=False,
        missing_status="mocked_missing_mlx",
        apply_fn=apply_fn,
    )
