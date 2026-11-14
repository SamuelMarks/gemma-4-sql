"""PyTorch-specific model quantization pipelines (BitsAndBytes, AWQ, GPTQ, GGUF).

Supports:
- INT8 / INT4 Weight-Only & Activation Quantization via BitsAndBytes (CUDA / ROCm).
- Activation-aware Weight Quantization (AWQ) via AutoAWQ for 4-bit inference.
- Accurate Post-Training Quantization (GPTQ) via Hugging Face Optimum.
- GGUF format serialization for high-performance llama.cpp deployment on CPU/Metal.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, overload, runtime_checkable

from gemma_4_sql.backends.common_quantize import (
    BitsAndBytesConfigProtocol,
    ModelWithQuantizableConfigProtocol,
    apply_bits_and_bytes_quantization,
    compute_awq_channel_scales,
    compute_gptq_compensated_weights,
    quantize_model_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.pytorch.gguf import (
    GGUFHeaderInfo,
    GGUFTensorDescriptor,
    TensorInput,
    _StateDictModelProtocol,
    _write_gguf_file,
    validate_gguf_file,
)
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, QuantizationError
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    DTypeProtocol,
    JSONValue,
    ModelType,
    QuantizeResultDict,
    ToListArrayProtocol,
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
class _Float16TorchModuleProtocol(Protocol):
    """Protocol for a `torch` module exposing `.float16`."""

    @property
    def float16(self) -> DTypeProtocol:
        """Provide `torch.float16` dtype.

        Returns:
            `torch.float16` dtype descriptor.

        """
        ...


@runtime_checkable
class _MutableParameterProtocol(Protocol):
    """Protocol for a `torch.nn.Parameter` exposing mutable `.data`."""

    data: TorchTensorProtocol


@runtime_checkable
class _NamedParametersModelProtocol(Protocol):
    """Protocol for an in-memory PyTorch model exposing `.named_parameters()`."""

    def named_parameters(self) -> Iterable[tuple[str, _MutableParameterProtocol]]:
        """Yield `(name, parameter)` pairs across the module.

        Returns:
            Iterable of `(parameter_name, parameter)` pairs.

        """
        ...


torch = optional_import("torch")
BitsAndBytesConfig: BitsAndBytesConfigProtocol | None = import_optional_symbol(
    "transformers",
    "BitsAndBytesConfig",
    BitsAndBytesConfigProtocol,
)
AutoModelForCausalLM = import_optional_symbol("transformers", "AutoModelForCausalLM", type)
AutoTokenizer = import_optional_symbol("transformers", "AutoTokenizer", type)


@runtime_checkable
class _ToListOnlyProtocol(Protocol):
    """Minimal structural protocol for objects exposing `.tolist()`."""

    def tolist(self) -> list[list[float]] | list[float] | float:
        """Convert array to nested Python float list or scalar.

        Returns:
            Nested float list or scalar float.

        """
        ...


def _tensor_to_2d_list(tensor: ToListArrayProtocol | _ToListOnlyProtocol | TorchTensorProtocol) -> list[list[float]]:
    """Convert a 2D PyTorch tensor into `list[list[float]]` via `.tolist()`.

    Args:
        tensor: 2D calibratable tensor.

    Returns:
        Nested 2D Python float list.

    """
    raw = tensor.tolist()
    if isinstance(raw, list):
        return [[float(val) for val in row if isinstance(val, (int, float))] for row in raw if isinstance(row, list)]
    return []


@overload
def calibrate_pytorch_awq_scales[TensorT: TorchTensorProtocol](
    weight: TensorT,
    activations: None = None,
    n_grid: int = 20,
) -> TensorT:
    """Compute AWQ channel scales without activation tensor."""


@overload
def calibrate_pytorch_awq_scales[TensorT: TorchTensorProtocol](
    weight: TensorT,
    activations: TensorT,
    n_grid: int = 20,
) -> TensorT:
    """Compute AWQ channel scales using calibration `activations`."""


def calibrate_pytorch_awq_scales(
    weight: TorchTensorProtocol,
    activations: TorchTensorProtocol | None = None,
    n_grid: int = 20,
) -> TorchTensorProtocol:
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
    import torch as _torch

    assert isinstance(weight, _torch.Tensor)
    w = weight.detach().to(_torch.float32)
    w_2d = _tensor_to_2d_list(w)
    acts_2d: list[list[float]] | None = None
    if activations is not None:
        assert isinstance(activations, _torch.Tensor)
        act_t = activations.detach().to(_torch.float32)
        acts_2d = _tensor_to_2d_list(act_t)
    grid = [i / float(max(1, n_grid)) for i in range(max(1, n_grid))]
    scales_1d = compute_awq_channel_scales(w_2d, acts_2d, alpha_range=grid)
    res: TorchTensorProtocol = _torch.tensor([scales_1d], dtype=_torch.float32, device=weight.device).to(weight.dtype)
    return res


@overload
def calibrate_pytorch_gptq_weights[TensorT: TorchTensorProtocol](
    weight: TensorT,
    activations: None = None,
    bits: int = 4,
    damp_percent: float = 0.01,
) -> TensorT:
    """Compute GPTQ error-compensated weights without activation tensor."""


@overload
def calibrate_pytorch_gptq_weights[TensorT: TorchTensorProtocol](
    weight: TensorT,
    activations: TensorT,
    bits: int = 4,
    damp_percent: float = 0.01,
) -> TensorT:
    """Compute GPTQ error-compensated weights using calibration `activations`."""


def calibrate_pytorch_gptq_weights(
    weight: TorchTensorProtocol,
    activations: TorchTensorProtocol | None = None,
    bits: int = 4,
    damp_percent: float = 0.01,
) -> TorchTensorProtocol:
    """Quantize a 2D PyTorch weight matrix using second-order Hessian error compensation (GPTQ).

    Args:
        weight: 2D `torch.Tensor` weight matrix of shape `(out_features, in_features)`.
        activations: Optional 2D `torch.Tensor` calibration activations of shape `(tokens, in_features)`.
        bits: Target quantization bit-width (default 4).
        damp_percent: Diagonal damping factor for Hessian positive-definiteness.

    Returns:
        GPTQ error-compensated quantized-and-dequantized `torch.Tensor` matching `weight.shape`.

    Raises:
        DependencyMissingError: If PyTorch is unavailable.

    """
    if torch is None:
        raise DependencyMissingError("PyTorch is required for native GPTQ calibration.")
    import torch as _torch

    assert isinstance(weight, _torch.Tensor)
    w = weight.detach().to(_torch.float32)
    w_2d = _tensor_to_2d_list(w)
    acts_2d: list[list[float]] | None = None
    if activations is not None:
        assert isinstance(activations, _torch.Tensor)
        act_t = activations.detach().to(_torch.float32)
        acts_2d = _tensor_to_2d_list(act_t)
    q_2d = compute_gptq_compensated_weights(w_2d, acts_2d, bits=bits, damp_percent=damp_percent)
    res: TorchTensorProtocol = _torch.tensor(q_2d, dtype=_torch.float32, device=weight.device).to(weight.dtype)
    return res


def _apply_awq_quantization(
    model_name: str,
    export_path: str | None = None,
    calib_data: list[str] | None = None,
    w_bit: int = 4,
    q_group_size: int = 128,
    *,
    model: _NamedParametersModelProtocol | ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | None = None,
) -> tuple[float, str]:
    """Execute real AWQ quantization using AutoAWQ or native PyTorch tensor AWQ on in-memory models.

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
        import torch as _torch

        for _name, param in model.named_parameters():
            pdata = param.data
            if isinstance(pdata, _torch.Tensor) and pdata.ndim == 2:
                scales = calibrate_pytorch_awq_scales(pdata, n_grid=10)
                w_scaled = pdata.to(_torch.float32) * scales.to(_torch.float32)
                q_max = float((1 << (max(2, w_bit) - 1)) - 1)
                step = _torch.clamp(_torch.amax(_torch.abs(w_scaled), dim=-1, keepdim=True), min=1e-5) / q_max
                w_q = _torch.clamp(_torch.round(w_scaled / step), -q_max - 1.0, q_max) * step
                param.data = (w_q / scales.to(_torch.float32)).to(pdata.dtype)
        register_quantized_artifact(model, method="awq")
        return (0.75, "quantized_awq")

    @runtime_checkable
    class _QuantSaveTokenizerProtocol(Protocol):
        """Protocol for a Hugging Face tokenizer supporting `save_pretrained`."""

        def save_pretrained(self, save_directory: str, /) -> tuple[str, ...] | None:
            """Save tokenizer files to `save_directory`."""
            ...

    @runtime_checkable
    class _QuantAutoTokenizerProtocol(Protocol):
        """Protocol for `AutoTokenizer` exposing `from_pretrained`."""

        def from_pretrained(self, model_name: str, /) -> _QuantSaveTokenizerProtocol | None:
            """Load pretrained tokenizer."""
            ...

    @runtime_checkable
    class _AWQLoadedModelProtocol(Protocol):
        """Protocol for an `AutoAWQForCausalLM` instance."""

        def quantize(
            self,
            tokenizer: _QuantSaveTokenizerProtocol,
            *,
            quant_config: dict[str, str | int | bool],
            calib_data: list[str],
        ) -> None:
            """Run AWQ calibration and quantization."""
            ...

        def save_quantized(self, save_dir: str, /) -> None:
            """Persist AWQ quantized model."""
            ...

    @runtime_checkable
    class _AWQModelClassProtocol(Protocol):
        """Protocol for `awq.AutoAWQForCausalLM`."""

        def from_pretrained(self, model_name: str, /) -> _AWQLoadedModelProtocol:
            """Load model for AWQ quantization."""
            ...

    @runtime_checkable
    class _AWQModuleProtocol(Protocol):
        """Protocol for `awq` module exposing `AutoAWQForCausalLM`."""

        @property
        def AutoAWQForCausalLM(self) -> _AWQModelClassProtocol:
            """Provide `AutoAWQForCausalLM` loader."""
            ...

    awq_mod = optional_import("awq", _AWQModuleProtocol)
    tok_cls = import_optional_symbol("transformers", "AutoTokenizer", _QuantAutoTokenizerProtocol)
    if awq_mod is None or tok_cls is None:
        raise DependencyMissingError("AutoAWQ is required for AWQ quantization: missing awq or transformers")
    awq_cls = awq_mod.AutoAWQForCausalLM

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
    assert tokenizer is not None
    loaded_model = awq_cls.from_pretrained(model_name)
    loaded_model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    out_dir = Path(export_path) if export_path else Path(f"./quantized_awq/{Path(model_name).name or 'model'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded_model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return (0.75, "quantized_awq")


def _apply_gptq_quantization(
    model_name: str,
    export_path: str | None = None,
    bits: int = 4,
    dataset: str | list[str] = "c4",
    group_size: int = 128,
    damp_percent: float = 0.01,
    *,
    model: _NamedParametersModelProtocol | ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | None = None,
) -> tuple[float, str]:
    """Execute real GPTQ quantization using Optimum or native PyTorch Hessian compensation on in-memory models.

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

    @runtime_checkable
    class _GPTQTokenizerProtocol(Protocol):
        """Protocol for a Hugging Face tokenizer supporting `save_pretrained`."""

        def save_pretrained(self, save_directory: str, /) -> tuple[str, ...] | None:
            """Save tokenizer files to `save_directory`."""
            ...

    @runtime_checkable
    class _GPTQAutoTokenizerProtocol(Protocol):
        """Protocol for `AutoTokenizer` exposing `from_pretrained`."""

        def from_pretrained(self, model_name: str, /) -> _GPTQTokenizerProtocol | None:
            """Load pretrained tokenizer."""
            ...

    @runtime_checkable
    class _GPTQAutoModelProtocol(Protocol):
        """Protocol for `AutoModelForCausalLM` exposing `from_pretrained`."""

        def from_pretrained(self, model_name: str, **kwargs: DTypeProtocol | str) -> ModelType:
            """Load pretrained causal LM."""
            ...

    @runtime_checkable
    class _GPTQQuantizerInstanceProtocol(Protocol):
        """Protocol for an initialized `optimum.gptq.GPTQQuantizer`."""

        def quantize_model(self, model: ModelType, tokenizer: _GPTQTokenizerProtocol, /) -> ModelType:
            """Quantize `model` using `tokenizer`."""
            ...

        def save(self, model: ModelType, save_dir: str, /) -> None:
            """Persist quantized GPTQ model."""
            ...

    @runtime_checkable
    class _GPTQNamespaceProtocol(Protocol):
        """Protocol for `optimum.gptq` exposing `GPTQQuantizer`."""

        def GPTQQuantizer(
            self,
            *,
            bits: int,
            dataset: str | list[str],
            group_size: int,
            damp_percent: float,
        ) -> _GPTQQuantizerInstanceProtocol:
            """Construct a `GPTQQuantizer`."""
            ...

    @runtime_checkable
    class _OptimumModuleProtocol(Protocol):
        """Protocol for `optimum` exposing `.gptq`."""

        @property
        def gptq(self) -> _GPTQNamespaceProtocol:
            """Provide `optimum.gptq` namespace."""
            ...

    opt_mod = optional_import("optimum", _OptimumModuleProtocol)
    tok_cls = import_optional_symbol("transformers", "AutoTokenizer", _GPTQAutoTokenizerProtocol)
    model_cls = import_optional_symbol("transformers", "AutoModelForCausalLM", _GPTQAutoModelProtocol)
    if opt_mod is None or tok_cls is None or model_cls is None:
        raise DependencyMissingError("Optimum is required for GPTQ quantization: missing optimum or transformers")
    quantizer_cls = opt_mod.gptq.GPTQQuantizer

    quantizer = quantizer_cls(
        bits=bits,
        dataset=dataset,
        group_size=group_size,
        damp_percent=damp_percent,
    )

    tokenizer = tok_cls.from_pretrained(model_name)
    assert tokenizer is not None
    model_kwargs: dict[str, DTypeProtocol | str] = {}
    if isinstance(torch, _Float16TorchModuleProtocol):
        model_kwargs["torch_dtype"] = torch.float16
    hf_model = model_cls.from_pretrained(model_name, **model_kwargs)

    quantized_model: ModelType = quantizer.quantize_model(hf_model, tokenizer)

    out_dir = Path(export_path) if export_path else Path(f"./quantized_gptq/{Path(model_name).name or 'model'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer.save(quantized_model, str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return (0.75, "quantized_gptq")


@runtime_checkable
class _QuantXLADeviceTensorProtocol(Protocol):
    """Protocol for XLA tensors exposing `.device` and `.to('cpu')`."""

    @property
    def device(self) -> DTypeProtocol | str | int:
        """Provide tensor device handle."""
        ...

    def to(self, target: str, /) -> TorchTensorProtocol:
        """Move tensor to `target` device.

        Args:
            target: Target device string (`'cpu'`).

        Returns:
            CPU-staged tensor.

        """
        ...


def _stage_xla_gguf_tensor[TensorT: TensorInput | _QuantXLADeviceTensorProtocol](
    tensor: TensorT,
) -> TensorInput | TensorT:
    """Stage `xla:*` tensors onto `'cpu'` prior to GGUF binary serialization.

    Args:
        tensor: Extracted state-dict tensor or buffer.

    Returns:
        CPU-staged tensor when originating on XLA, otherwise original `tensor`.

    """
    if isinstance(tensor, _QuantXLADeviceTensorProtocol) and str(tensor.device).startswith("xla"):
        return tensor.to("cpu")
    return tensor


def _export_gguf(
    model_name: str,
    export_path: str | None = None,
    out_type: str = "q4_k_m",
    *,
    allow_dummy_fallback: bool = False,
    model: _StateDictModelProtocol | TorchCausalLMProtocol | None = None,
) -> tuple[float, str]:
    """Export model to GGUF format for llama.cpp execution and validate binary header.

    Fails fast with `QuantizationError` when model weights cannot be resolved unless
    `allow_dummy_fallback=True` is explicitly passed.

    Args:
        model_name: The name or path of the model.
        export_path: Destination directory or `.gguf` file path for GGUF export.
        out_type: Quantization format identifier (e.g. `q4_k_m`, `q8_0`, `f16`).
        allow_dummy_fallback: Whether to permit synthetic dummy weights if `model_name` cannot be found.
        model: Optional in-memory model exposing `.state_dict()`.

    Returns:
        Tuple of memory reduction factor (`0.6`) and status string (`'quantized_gguf'`).

    Raises:
        QuantizationError: If model weights cannot be extracted and `allow_dummy_fallback` is `False`.

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

        source_arg: _StateDictModelProtocol | TorchCausalLMProtocol | str = model if model is not None else model_name
        try:
            raw_tensors = _gguf_mod.extract_pytorch_state_dict(
                source_arg,
                allow_dummy_fallback=allow_dummy_fallback,
            )
            tensors = {k: _stage_xla_gguf_tensor(v) for k, v in raw_tensors.items()}
        except (ExportError, ImportError, ValueError, RuntimeError, AttributeError, OSError) as exc:
            raise QuantizationError(
                f"Failed to extract PyTorch weights for GGUF export of '{model_name}': {exc}",
                backend="pytorch",
                model_name=model_name,
                cause=exc,
            ) from exc
        _write_gguf_file(gguf_file, model_name=model_name, tensors=tensors, out_type=out_type)

    validate_gguf_file(gguf_file)
    logger.info("Successfully validated GGUF export at %s with type %s", gguf_file, out_type)
    return (0.6, "quantized_gguf")


type _QuantizeModelArg = (
    _NamedParametersModelProtocol
    | _StateDictModelProtocol
    | ModelWithQuantizableConfigProtocol
    | TorchCausalLMProtocol
    | str
)


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

    json_kwargs: dict[str, JSONValue] = {
        k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None
    }
    raw_model = kwargs.get("model")
    awq_gptq_model: (
        _NamedParametersModelProtocol | ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | None
    ) = (
        raw_model
        if isinstance(
            raw_model, (_NamedParametersModelProtocol, ModelWithQuantizableConfigProtocol, TorchCausalLMProtocol)
        )
        else None
    )
    if awq_gptq_model is None and kwargs.get("backend_alias") == "pytorch_native" and method in {"awq", "gptq", "gguf"}:
        from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

        awq_gptq_model = Gemma4ForCausalLM(Gemma4Config.from_preset(model_name))

    if method == "gguf":
        gguf_model: _StateDictModelProtocol | TorchCausalLMProtocol | None = (
            raw_model
            if isinstance(raw_model, (_StateDictModelProtocol, TorchCausalLMProtocol))
            else (
                awq_gptq_model if isinstance(awq_gptq_model, (_StateDictModelProtocol, TorchCausalLMProtocol)) else None
            )
        )
        allow_dummy_fb = extract_bool_kwarg(json_kwargs, "allow_dummy_fallback", False)

        def apply_fn() -> tuple[float, str]:
            """Execute GGUF format quantization export.

            Returns:
                Tuple of `(memory_reduction_ratio, status)`.

            """
            export_path = str(extract_str_kwarg(json_kwargs, "export_path", "./gguf_export"))
            out_type = str(extract_str_kwarg(json_kwargs, "out_type", "q4_k_m"))
            return _export_gguf(
                model_name,
                export_path,
                out_type=out_type,
                allow_dummy_fallback=allow_dummy_fb,
                model=gguf_model,
            )

    elif method == "awq":

        def apply_fn() -> tuple[float, str]:
            """Execute AWQ 4-bit weight-only activation-aware quantization.

            Returns:
                Tuple of `(memory_reduction_ratio, status)`.

            """
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
            """Execute GPTQ second-order error-compensated quantization.

            Returns:
                Tuple of `(memory_reduction_ratio, status)`.

            """
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
            """Execute BitsAndBytes INT8/INT4 quantization.

            Returns:
                Tuple of `(memory_reduction_ratio, status)`.

            """
            threshold = extract_float_kwarg(json_kwargs, "llm_int8_threshold", 6.0)
            f16_dtype: DTypeProtocol | str | None = (
                torch.float16 if isinstance(torch, _Float16TorchModuleProtocol) else None
            )
            bnb_model: ModelWithQuantizableConfigProtocol | TorchCausalLMProtocol | None = (
                raw_model
                if isinstance(
                    raw_model,
                    (ModelWithQuantizableConfigProtocol, TorchCausalLMProtocol),
                )
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
        backend_name=str(kwargs.get("backend_alias", "pytorch")),
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
