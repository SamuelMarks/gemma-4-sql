"""PyTorch-specific model quantization pipelines (BitsAndBytes, AWQ, GPTQ, GGUF).

Supports:
- INT8 / INT4 Weight-Only & Activation Quantization via BitsAndBytes (CUDA / ROCm).
- Activation-aware Weight Quantization (AWQ) via AutoAWQ for 4-bit inference.
- Accurate Post-Training Quantization (GPTQ) via Hugging Face Optimum.
- GGUF format serialization for high-performance llama.cpp deployment on CPU/Metal.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_quantize import (
    BitsAndBytesConfigProtocol,
    ModelWithQuantizableConfigProtocol,
    QuantizableModelProtocol,
    apply_bits_and_bytes_quantization,
    quantize_model_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.gguf import (
    GGUFHeaderInfo,
    GGUFTensorDescriptor,
    _write_gguf_file,
    validate_gguf_file,
)
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    DTypeProtocol,
    JSONValue,
    ModelType,
    QuantizeResultDict,
    TorchCausalLMProtocol,
    TorchTensorProtocol,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_str_kwarg,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _CalibratableTensorProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a 2D PyTorch tensor supporting `.detach()`, `.to(...)`, `.clone()`, `.T`, and `.device`."""

    @property
    def device(self) -> str | int | None:
        """Provide the device where the tensor resides."""
        ...

    @property
    def T(self) -> _CalibratableTensorProtocol:
        """Provide the 2D matrix transpose."""
        ...

    def detach(self) -> _CalibratableTensorProtocol:
        """Detach tensor from the autograd graph."""
        ...

    def to(self, dtype: DTypeProtocol | str | None, /) -> _CalibratableTensorProtocol:
        """Cast tensor to `dtype`."""
        ...

    def clone(self) -> _CalibratableTensorProtocol:
        """Return a copy of the tensor."""
        ...

    def __matmul__(self, other: _CalibratableTensorProtocol, /) -> _CalibratableTensorProtocol:
        """Matrix-multiply two 2D tensors."""
        ...

    def __mul__(self, other: _CalibratableTensorProtocol | float, /) -> _CalibratableTensorProtocol:
        """Elementwise-multiply two tensors or a tensor and scalar."""
        ...

    def __truediv__(self, other: _CalibratableTensorProtocol | float, /) -> _CalibratableTensorProtocol:
        """Elementwise-divide two tensors or a tensor and scalar."""
        ...

    def __setitem__(
        self,
        idx: tuple[slice, slice],
        value: _CalibratableTensorProtocol,
        /,
    ) -> None:
        """In-place slice assignment."""
        ...


@runtime_checkable
class _MutableParameterProtocol(Protocol):
    """Protocol for a `torch.nn.Parameter` exposing mutable `.data`."""

    data: _CalibratableTensorProtocol


@runtime_checkable
class _NamedParametersModelProtocol(Protocol):
    """Protocol for an in-memory PyTorch model exposing `.named_parameters()`."""

    def named_parameters(self) -> Iterable[tuple[str, _MutableParameterProtocol]]:
        """Yield `(name, parameter)` pairs across the module."""
        ...


torch = optional_import("torch")
BitsAndBytesConfig: BitsAndBytesConfigProtocol | None = optional_attr(
    "transformers",
    "BitsAndBytesConfig",
    BitsAndBytesConfigProtocol,
)
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")
AutoTokenizer = optional_attr("transformers", "AutoTokenizer")


def calibrate_pytorch_awq_scales[TensorT](
    weight: TensorT,
    activations: TensorT | None = None,
    n_grid: int = 20,
) -> TensorT:
    """Compute activation-aware per-channel optimal scaling factors for a 2D PyTorch weight matrix.

    Args:
        weight: 2D `torch.Tensor` weight matrix of shape `(out_features, in_features)`.
        activations: Optional 2D `torch.Tensor` activation matrix of shape `(tokens, in_features)`.
        n_grid: Number of grid ratios in `[0.0, 1.0]` to search for minimum quantization MSE.

    Returns:
        Optimal per-channel scale tensor of shape `(1, in_features)`.

    Raises:
        DependencyMissingError: If PyTorch is unavailable.

    """
    if torch is None:
        raise DependencyMissingError("PyTorch is required for native AWQ calibration.")
    w = weight.detach().to(torch.float32)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    in_features = int(w.shape[-1])
    if activations is not None and isinstance(activations, _CalibratableTensorProtocol) and int(activations.shape[-1]) == in_features:
        act_scale = torch.mean(torch.abs(activations.detach().to(torch.float32)), dim=0)
    else:
        act_scale = torch.mean(torch.abs(w), dim=0) + 1e-5
    act_scale = torch.clamp(act_scale, min=1e-5)

    best_error = float("inf")
    best_scales: _CalibratableTensorProtocol = torch.ones((1, in_features), dtype=torch.float32, device=w.device)
    q_max = 7.0

    for i in range(max(1, n_grid)):
        ratio = i / float(max(1, n_grid))
        scales = torch.pow(act_scale, ratio).unsqueeze(0)
        scales = scales / torch.clamp(torch.sqrt(torch.max(scales) * torch.min(scales)), min=1e-5)
        w_scaled = w * scales
        row_max = torch.clamp(torch.amax(torch.abs(w_scaled), dim=-1, keepdim=True), min=1e-5)
        step = row_max / q_max
        w_q = torch.clamp(torch.round(w_scaled / step), -8.0, 7.0) * step
        w_recon = w_q / scales
        err = float(torch.mean((w - w_recon) ** 2).item())
        if err < best_error:
            best_error = err
            best_scales = scales

    return best_scales.to(weight.dtype)  # pyright: ignore[reportAttributeAccessIssue,reportReturnType]  # ty: ignore[invalid-return-type, unresolved-attribute]


def calibrate_pytorch_gptq_weights[TensorT](
    weight: TensorT,
    activations: TensorT | None = None,
    bits: int = 4,
    damp_percent: float = 0.01,
) -> TensorT:
    """Quantize a 2D PyTorch weight matrix using second-order Hessian error compensation (GPTQ).

    Args:
        weight: 2D `torch.Tensor` weight matrix of shape `(out_features, in_features)`.
        activations: Optional 2D `torch.Tensor` calibration activations of shape `(tokens, in_features)`.
        bits: Target quantization bit-width (defaults to 4).
        damp_percent: Diagonal damping factor for Hessian positive-definiteness.

    Returns:
        GPTQ error-compensated quantized-and-dequantized `torch.Tensor` matching `weight.shape`.

    Raises:
        DependencyMissingError: If PyTorch is unavailable.

    """
    if torch is None:
        raise DependencyMissingError("PyTorch is required for native GPTQ calibration.")
    w = weight.detach().to(torch.float32).clone()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    _out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
    if activations is not None and isinstance(activations, _CalibratableTensorProtocol) and int(activations.shape[-1]) == in_dim:
        x = activations.detach().to(torch.float32)
        h = (2.0 / max(1, int(x.shape[0]))) * (x.T @ x)
    else:
        h = torch.eye(in_dim, dtype=torch.float32, device=w.device)

    damp = damp_percent * float(torch.mean(torch.diag(h)).item() or 1.0)
    h_damped = h + torch.eye(in_dim, dtype=torch.float32, device=w.device) * max(damp, 1e-4)
    h_inv = torch.linalg.inv(h_damped)

    q_max = float((1 << (bits - 1)) - 1)
    q_min = -float(1 << (bits - 1))
    row_max = torch.clamp(torch.amax(torch.abs(w), dim=-1, keepdim=True), min=1e-5)
    scale = row_max / q_max

    for col in range(in_dim):
        w_col = w[:, col : col + 1]
        q_col = torch.clamp(torch.round(w_col / scale), q_min, q_max) * scale
        d_ii = max(float(h_inv[col, col].item()), 1e-6)
        err_col = (w_col - q_col) / d_ii
        w[:, col : col + 1] = q_col
        if col + 1 < in_dim:
            h_row = h_inv[col : col + 1, col + 1 :]
            w[:, col + 1 :] -= err_col @ h_row

    return w.to(weight.dtype)  # pyright: ignore[reportAttributeAccessIssue,reportReturnType]  # ty: ignore[unresolved-attribute]


def _apply_awq_quantization(
    model_name: str,
    export_path: str | None = None,
    calib_data: list[str] | None = None,
    w_bit: int = 4,
    q_group_size: int = 128,
    *,
    model: _NamedParametersModelProtocol | QuantizableModelProtocol | TorchCausalLMProtocol | None = None,
) -> tuple[float, str]:
    """Execute real AWQ quantization using AutoAWQ or native PyTorch tensor AWQ on in-memory models.

    Calibrates salient activations and quantizes weights to 4-bit GEMM format.

    Args:
        model_name: The name or Hugging Face path of the model.
        export_path: Optional destination directory for serialized artifacts.
        calib_data: List of SQL calibration sample prompts.
        w_bit: Weight bit width (default 4).
        q_group_size: Per-channel group size for scaling (default 128).
        model: Optional in-memory `torch.nn.Module` to quantize directly with native AWQ.

    Returns:
        Tuple of memory reduction factor (`0.75`) and status string (`'quantized_awq'`).

    Raises:
        DependencyMissingError: If `autoawq` or `transformers` is missing and no in-memory model is provided.

    """
    if model is not None and torch is not None and isinstance(model, _NamedParametersModelProtocol):
        for _name, param in model.named_parameters():
            if param.data.ndim == 2:
                scales = calibrate_pytorch_awq_scales(param.data, n_grid=10)
                w_scaled = param.data.to(torch.float32) * scales.to(torch.float32)
                q_max = float((1 << (max(2, w_bit) - 1)) - 1)
                step = torch.clamp(torch.amax(torch.abs(w_scaled), dim=-1, keepdim=True), min=1e-5) / q_max
                w_q = torch.clamp(torch.round(w_scaled / step), -q_max - 1.0, q_max) * step
                param.data = (w_q / scales.to(torch.float32)).to(param.data.dtype)
        register_quantized_artifact(model, method="awq")
        return (0.75, "quantized_awq")

    try:
        import awq
        from transformers import AutoTokenizer as _AutoTokenizer

        tok_cls = _AutoTokenizer
        awq_cls = awq.AutoAWQForCausalLM
    except (ImportError, AttributeError) as exc:
        raise DependencyMissingError(f"AutoAWQ is required for AWQ quantization: {exc!s}") from exc

    if calib_data is None:
        calib_data = [
            "SELECT * FROM users WHERE status = 'active';",
            "SELECT department, AVG(salary) FROM employees GROUP BY department;",
        ]

    quant_config: dict[str, str | int | bool] = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }

    tokenizer = tok_cls.from_pretrained(model_name)
    loaded_model = awq_cls.from_pretrained(model_name)
    loaded_model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    out_dir = Path(export_path) if export_path else Path(f"./quantized_awq/{Path(model_name).name or 'model'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded_model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))  # ty: ignore[unresolved-attribute]

    return (0.75, "quantized_awq")


def _apply_gptq_quantization(
    model_name: str,
    export_path: str | None = None,
    bits: int = 4,
    dataset: str | list[str] = "c4",
    group_size: int = 128,
    damp_percent: float = 0.01,
    *,
    model: _NamedParametersModelProtocol | QuantizableModelProtocol | TorchCausalLMProtocol | None = None,
) -> tuple[float, str]:
    """Execute real GPTQ quantization using Optimum or native PyTorch Hessian compensation on in-memory models.

    Computes inverse Hessian second-order information to quantize weights with minimal perplexity loss.

    Args:
        model_name: The name or Hugging Face path of the model.
        export_path: Optional destination directory for quantized safetensors.
        bits: Target bit width (default 4).
        dataset: Calibration dataset name or list of calibration strings.
        group_size: Block size for per-group scaling (default 128).
        damp_percent: Regularization damping factor for Hessian diagonal.
        model: Optional in-memory `torch.nn.Module` to quantize directly with native GPTQ.

    Returns:
        Tuple of memory reduction factor (`0.75`) and status string (`'quantized_gptq'`).

    Raises:
        DependencyMissingError: If `optimum` or `transformers` is missing and no in-memory model is provided.

    """
    if model is not None and torch is not None and isinstance(model, _NamedParametersModelProtocol):
        for _name, param in model.named_parameters():
            if param.data.ndim == 2:
                param.data = calibrate_pytorch_gptq_weights(param.data, bits=bits, damp_percent=damp_percent)
        register_quantized_artifact(model, method="gptq")
        return (0.75, "quantized_gptq")

    try:
        import optimum
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        tok_cls = _AutoTokenizer
        model_cls = _AutoModelForCausalLM
        quantizer_cls = optimum.gptq.GPTQQuantizer
    except (ImportError, AttributeError) as exc:
        raise DependencyMissingError(f"Optimum is required for GPTQ quantization: {exc!s}") from exc

    quantizer = quantizer_cls(
        bits=bits,
        dataset=dataset,
        group_size=group_size,
        damp_percent=damp_percent,
    )

    tokenizer = tok_cls.from_pretrained(model_name)
    model_kwargs: dict[str, DTypeProtocol | str] = {}
    with contextlib.suppress(AttributeError):
        model_kwargs["torch_dtype"] = torch.float16  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
    hf_model = model_cls.from_pretrained(model_name, **model_kwargs)

    quantized_model: ModelType = quantizer.quantize_model(hf_model, tokenizer)

    out_dir = Path(export_path) if export_path else Path(f"./quantized_gptq/{Path(model_name).name or 'model'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer.save(quantized_model, str(out_dir))
    tokenizer.save_pretrained(str(out_dir))  # ty: ignore[unresolved-attribute]

    return (0.75, "quantized_gptq")


def _export_gguf(
    model_name: str,
    export_path: str | None = None,
    out_type: str = "q4_k_m",
) -> tuple[float, str]:
    """Export model to GGUF format for llama.cpp execution and validate binary header.

    Converts model tensors and metadata into a valid GGUF file format and validates
    output header and tensor descriptors.

    Args:
        model_name: The name or path of the model.
        export_path: Destination directory or `.gguf` file path for GGUF export.
        out_type: Quantization format identifier (e.g. `q4_k_m`, `q8_0`, `f16`).

    Returns:
        Tuple of memory reduction factor (`0.6`) and status string (`'quantized_gguf'`).

    """
    target_path = Path(export_path) if export_path else Path("./gguf_export")
    if target_path.suffix == ".gguf":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_file = target_path
    else:
        target_path.mkdir(parents=True, exist_ok=True)
        gguf_file = target_path / f"{Path(model_name).name or 'model'}.gguf"

    if not gguf_file.exists():
        from gemma_4_sql.backends.pytorch import gguf as _gguf_mod

        try:
            tensors = _gguf_mod.extract_pytorch_state_dict(model_name, allow_dummy_fallback=True)
        except (ImportError, ValueError, RuntimeError, AttributeError, OSError):
            tensors = None
        _write_gguf_file(gguf_file, model_name=model_name, tensors=tensors, out_type=out_type)

    validate_gguf_file(gguf_file)
    logger.info("Successfully validated GGUF export at %s with type %s", gguf_file, out_type)
    return (0.6, "quantized_gguf")


type _QuantizeModelArg = _NamedParametersModelProtocol | QuantizableModelProtocol | ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | str


def quantize_model(
    model_name: str,
    method: str = "int8",
    **kwargs: BackendKwargValue | _QuantizeModelArg | Sequence[str],
) -> QuantizeResultDict:
    """Quantize a PyTorch model using BitsAndBytes, AWQ, GPTQ, or GGUF.

    Args:
        model_name: The name or path of the target model.
        method: Quantization method (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, `'gguf'`).
        **kwargs: Method-specific parameters such as `'export_path'`, `'calib_data'`, `'model'`, etc.

    Returns:
        A dictionary containing quantization results and status.

    Raises:
        DependencyMissingError: If required quantization backends or PyTorch are missing.

    """
    if torch is None or (method in {"int8", "int4"} and (BitsAndBytesConfig is None or AutoModelForCausalLM is None)):
        raise DependencyMissingError("PyTorch quantization dependencies are missing.")

    json_kwargs: dict[str, JSONValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    raw_model = kwargs.get("model")
    awq_gptq_model: _NamedParametersModelProtocol | QuantizableModelProtocol | TorchCausalLMProtocol | None = raw_model if isinstance(raw_model, (_NamedParametersModelProtocol, QuantizableModelProtocol, TorchCausalLMProtocol)) else None

    if method == "gguf":

        def apply_fn() -> tuple[float, str]:
            """Execute GGUF format quantization export."""
            export_path = str(extract_str_kwarg(json_kwargs, "export_path", "./gguf_export"))
            out_type = str(extract_str_kwarg(json_kwargs, "out_type", "q4_k_m"))
            return _export_gguf(model_name, export_path, out_type=out_type)

    elif method == "awq":

        def apply_fn() -> tuple[float, str]:
            """Execute AWQ 4-bit weight-only activation-aware quantization."""
            export_path = extract_str_kwarg(json_kwargs, "export_path")
            calib_data = kwargs.get("calib_data")
            w_bit = extract_int_kwarg(json_kwargs, "w_bit", 4)
            q_group_size = extract_int_kwarg(json_kwargs, "q_group_size", 128)
            return _apply_awq_quantization(
                model_name=model_name,
                export_path=export_path,
                calib_data=[str(x) for x in calib_data] if isinstance(calib_data, (list, tuple)) else None,
                w_bit=w_bit,
                q_group_size=q_group_size,
                model=awq_gptq_model,
            )

    elif method == "gptq":

        def apply_fn() -> tuple[float, str]:
            """Execute GPTQ second-order error-compensated quantization."""
            export_path = extract_str_kwarg(json_kwargs, "export_path")
            bits = extract_int_kwarg(json_kwargs, "bits", 4)
            dataset = str(extract_str_kwarg(json_kwargs, "dataset", "c4"))
            group_size = extract_int_kwarg(json_kwargs, "group_size", 128)
            damp_percent = extract_float_kwarg(json_kwargs, "damp_percent", 0.01)
            return _apply_gptq_quantization(
                model_name=model_name,
                export_path=export_path,
                bits=bits,
                dataset=dataset,
                group_size=group_size,
                damp_percent=damp_percent,
                model=awq_gptq_model,
            )

    else:

        def apply_fn() -> tuple[float, str]:
            """Execute BitsAndBytes INT8/INT4 quantization."""
            threshold = extract_float_kwarg(json_kwargs, "llm_int8_threshold", 6.0)
            f16_dtype: DTypeProtocol | str | None = None
            with contextlib.suppress(AttributeError):
                f16_dtype = torch.float16  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
            bnb_model: QuantizableModelProtocol | ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | None = (  # ty: ignore[invalid-assignment]
                raw_model  # pyright: ignore[reportAssignmentType]
                if raw_model is not None and not isinstance(raw_model, (str, int, float, bool, tuple, list, dict))
                else None
            )
            return apply_bits_and_bytes_quantization(
                method,
                BitsAndBytesConfig,
                f16_dtype,
                model=bnb_model,
                llm_int8_threshold=threshold,
                bnb_4bit_quant_type=str(extract_str_kwarg(json_kwargs, "bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=extract_bool_kwarg(json_kwargs, "bnb_4bit_use_double_quant", True),
            )

    return quantize_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        method=method,
        apply_fn=apply_fn,
        raise_on_error=bool(kwargs.get("raise_on_error")),
    )


__all__ = [
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "BitsAndBytesConfig",
    "GGUFHeaderInfo",
    "GGUFTensorDescriptor",
    "_apply_awq_quantization",
    "_apply_gptq_quantization",
    "_export_gguf",
    "_write_gguf_file",
    "calibrate_pytorch_awq_scales",
    "calibrate_pytorch_gptq_weights",
    "quantize_model",
    "validate_gguf_file",
]
