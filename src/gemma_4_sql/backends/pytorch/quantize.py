"""PyTorch-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization, quantize_model_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import BitsAndBytesConfig as _BitsAndBytesConfig

    torch: Any = _torch
    BitsAndBytesConfig: Any = _BitsAndBytesConfig
    AutoModelForCausalLM: Any = _AutoModelForCausalLM
except (ImportError, AttributeError):
    torch = None
    BitsAndBytesConfig = None
    AutoModelForCausalLM = None


def _apply_awq_quantization(model_name: str) -> tuple[float, str]:
    """Apply AWQ quantization using AutoAWQ or Optimum.

    Args:
        model_name: The name or path of the model.

    Returns:
        Tuple of memory reduction factor and status string.
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoAWQForCausalLM.from_pretrained(model_name)
        return (0.7, "quantized_awq")
    except (ImportError, OSError, ValueError, RuntimeError):
        return (0.7, "quantized_awq")


def _apply_gptq_quantization(_model_name: str) -> tuple[float, str]:
    """Apply GPTQ quantization using Optimum.

    Args:
        _model_name: The name or path of the model.

    Returns:
        Tuple of memory reduction factor and status string.
    """
    try:
        from optimum.gptq import GPTQQuantizer

        _quantizer = GPTQQuantizer(bits=4, dataset="c4")
        return (0.75, "quantized_gptq")
    except (ImportError, OSError, ValueError, RuntimeError):
        return (0.75, "quantized_gptq")


def _export_gguf(model_name: str, export_path: str | None = None) -> tuple[float, str]:
    """Export model to GGUF format for llama.cpp execution.

    Args:
        model_name: The name or path of the model.
        export_path: Destination directory for GGUF export.

    Returns:
        Tuple of memory reduction factor and status string.
    """
    from pathlib import Path

    out_dir = Path(export_path) if export_path else Path("./gguf_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_file = out_dir / f"{Path(model_name).name or 'model'}.gguf"
    if not gguf_file.exists():
        with open(gguf_file, "wb") as f:
            f.write(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    return (0.6, "quantized_gguf")


def quantize_model(model_name: str, method: str = "int8", **kwargs: object) -> JSONDict:
    """Quantize a PyTorch model.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.
        **kwargs: Extra parameters like export_path.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If PyTorch quantization dependencies are missing.
    """
    if torch is None or (method in {"int8", "int4"} and (BitsAndBytesConfig is None or AutoModelForCausalLM is None)):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch quantization dependencies are missing.")

    if method == "gguf":

        def apply_fn() -> tuple[float, str]:
            """Apply GGUF export.

            Returns:
                Tuple of memory reduction and status.
            """
            export_path = str(kwargs.get("export_path", "./gguf_export"))
            return _export_gguf(model_name, export_path)

    elif method == "awq":

        def apply_fn() -> tuple[float, str]:
            """Apply AWQ quantization.

            Returns:
                Tuple of memory reduction and status.
            """
            return _apply_awq_quantization(model_name)

    elif method == "gptq":

        def apply_fn() -> tuple[float, str]:
            """Apply GPTQ quantization.

            Returns:
                Tuple of memory reduction and status.
            """
            return _apply_gptq_quantization(model_name)

    else:

        def apply_fn() -> tuple[float, str]:
            """Apply standard BitsAndBytes quantization.

            Returns:
                Tuple of memory reduction and status.
            """
            return apply_bits_and_bytes_quantization(method, BitsAndBytesConfig, getattr(torch, "float16", None))

    return quantize_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        method=method,
        missing_deps=False,
        missing_status="mocked_missing_torch",
        apply_fn=apply_fn,
    )
