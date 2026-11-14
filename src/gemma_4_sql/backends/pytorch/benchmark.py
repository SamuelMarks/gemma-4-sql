"""PyTorch-specific benchmarking pipeline."""

from __future__ import annotations

import contextvars
import logging
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.backends.common_benchmark import (
    BenchmarkTelemetry,
    DetailedLoopMetrics,
    execute_detailed_benchmark_loop,
    get_current_rss_mb,
    normalize_benchmark_mode,
    normalize_hardware_family,
    run_benchmark_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.pytorch.runtime_adapter import (
    TorchModuleInput,
    TorchRandintProtocol,
    TorchRuntimeAdapter,
)
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    CausalLMProtocol,
    DTypeProtocol,
    ForwardArg,
    ForwardOutput,
    JSONValue,
    ModelConfigProtocol,
    TorchCausalLMProtocol,
    TorchTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _BenchmarkCallableProtocol(Protocol):
    """Protocol for a callable benchmark model."""

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> ForwardOutput:
        """Execute a forward pass.

        Args:
            *args: Positional forward arguments.
            **kwargs: Keyword forward arguments.

        Returns:
            Model forward output.

        """
        ...


@runtime_checkable
class _ConfigVocabProtocol(Protocol):
    """Protocol for a config object exposing `.vocab_size`."""

    @property
    def vocab_size(self) -> int | str | None:
        """Provide the vocabulary size.

        Returns:
            Integer or string vocabulary size.

        """
        ...


@runtime_checkable
class _HasConfigProtocol(Protocol):
    """Protocol for a model exposing a `.config` attribute."""

    @property
    def config(self) -> _ConfigVocabProtocol | ModelConfigProtocol | None:
        """Provide the model configuration object.

        Returns:
            Model configuration or `None`.

        """
        ...


@runtime_checkable
class _GenerateCapableProtocol(Protocol):
    """Protocol for a PyTorch model supporting `.generate(...)`."""

    def generate(
        self,
        input_ids: TorchTensorProtocol,
        *,
        max_new_tokens: int = 20,
        min_new_tokens: int = 0,
    ) -> TorchTensorProtocol | ForwardOutput:
        """Generate token sequences autoregressively.

        Args:
            input_ids: Prompt token ID tensor.
            max_new_tokens: Maximum new tokens to decode.
            min_new_tokens: Minimum new tokens to decode.

        Returns:
            Generated token sequence tensor.

        """
        ...


@runtime_checkable
class _DeviceMovableOnlyProtocol(Protocol):
    """Protocol for a PyTorch model supporting `.to(...)`."""

    def to(self, device_or_dtype: DTypeProtocol | str, /) -> Self:
        """Move or cast the model to `device_or_dtype`.

        Args:
            device_or_dtype: Target device string or dtype.

        Returns:
            Self after device/dtype placement.

        """
        ...


@runtime_checkable
class _EvalOnlyProtocol(Protocol):
    """Protocol for models exposing `.eval()`."""

    def eval(self) -> Self:
        """Set evaluation mode.

        Returns:
            Self in evaluation mode.

        """
        ...


@runtime_checkable
class _HFAutoModelLoaderProtocol(Protocol):
    """Protocol for `transformers.AutoModelForCausalLM` exposing `.from_pretrained(...)`."""

    def from_pretrained(
        self,
        model_name: str,
        *,
        torch_dtype: DTypeProtocol | str | None = None,
    ) -> TorchCausalLMProtocol:
        """Load a pretrained Hugging Face causal language model.

        Args:
            model_name: Pretrained model identifier or path.
            torch_dtype: Optional loading dtype.

        Returns:
            Loaded `TorchCausalLMProtocol` instance.

        """
        ...


_raw_torch = optional_import("torch")
torch: TorchModuleInput | None = _raw_torch if isinstance(_raw_torch, TorchRandintProtocol) else None
AutoModelForCausalLM: _HFAutoModelLoaderProtocol | None = import_optional_symbol(
    "transformers",
    "AutoModelForCausalLM",
    _HFAutoModelLoaderProtocol,
)


def _runtime() -> TorchRuntimeAdapter:
    """Construct a `TorchRuntimeAdapter` bound to the current `torch` module.

    Returns:
        Configured `TorchRuntimeAdapter` instance.

    """
    return TorchRuntimeAdapter(torch)


def _get_device(hardware: str) -> str:
    """Provide the correct device based on hardware string.

    Args:
        hardware: Target hardware string.

    Returns:
        Device identifier string (`'cpu'`, `'cuda'`, or `'mps'`).

    """
    return _runtime().get_device(hardware)


def _move_and_eval(
    model: TorchCausalLMProtocol | _BenchmarkCallableProtocol,
    device: str,
    torch_dtype: DTypeProtocol | str | None = None,
) -> None:
    """Move `model` to `torch_dtype` and `device`, then set evaluation mode when supported.

    Args:
        model: Target PyTorch causal language model.
        device: Target device string.
        torch_dtype: Optional resolved torch dtype.

    """
    if isinstance(model, _DeviceMovableOnlyProtocol) and callable(model.to):
        if torch_dtype is not None:
            model.to(torch_dtype)
        model.to(device)
    if isinstance(model, _EvalOnlyProtocol) and callable(model.eval):
        model.eval()


def _load_pytorch_model_and_device(
    model_name: str,
    hardware: str,
    *,
    dtype: str = "bfloat16",
    backend_alias: str = "pytorch",
    strict_hardware: bool = False,
) -> tuple[TorchCausalLMProtocol, str]:
    """Load the model and determine device.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        dtype: The string representing dtype.
        backend_alias: The alias used (`'pytorch'`, `'pytorch_hf'`, `'hf_transformers'`, `'pytorch_native'`).
        strict_hardware: Whether to raise `HardwareUnavailableError` on accelerator fallback.

    Returns:
        A tuple of `(model, device)`.

    """
    adapter = _runtime()
    device = adapter.get_device(hardware, strict_hardware=strict_hardware)
    torch_dtype = adapter.resolve_dtype(dtype)

    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4Config, Gemma4ForCausalLM

        config = Gemma4Config.from_preset(model_name)
        model: TorchCausalLMProtocol = Gemma4ForCausalLM(config)
        _move_and_eval(model, device, torch_dtype)
    else:
        assert AutoModelForCausalLM is not None
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        _move_and_eval(model, device)

    compiled = adapter.compile_model(model, device=device)
    return (compiled, device)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA or MPS streams if using an accelerator.

    Args:
        device: The string representing the device.

    """
    _runtime().synchronize(device)


type _BenchmarkModelInput = TorchCausalLMProtocol | CausalLMProtocol | _BenchmarkCallableProtocol | str | None


def _get_memory_mb(model: _BenchmarkModelInput, device: str) -> float:
    """Provide max memory allocated in MB.

    Args:
        model: The model.
        device: The string representing the device.

    Returns:
        The max memory in MB.

    """
    _ = model
    return _runtime().max_memory_mb(device, get_current_rss_mb())


def _resolve_vocab_bound(model: _BenchmarkModelInput) -> int:
    """Extract positive integer `vocab_size` from `model.config` or default to `256000`.

    Args:
        model: Target PyTorch causal language model.

    Returns:
        Positive integer vocabulary upper bound for dummy input sampling.

    """
    if isinstance(model, _HasConfigProtocol):
        cfg = model.config
        if isinstance(cfg, _ConfigVocabProtocol):
            raw_vocab = cfg.vocab_size
            if isinstance(raw_vocab, int) and raw_vocab > 1:
                return raw_vocab
    return 256000


def _call_decode_generate(
    model: _BenchmarkModelInput,
    dummy_inputs: TorchTensorProtocol,
    tokens: int,
) -> None:
    """Invoke `model.generate` for decode benchmarking when supported.

    Args:
        model: Benchmark model instance.
        dummy_inputs: Synthetic input token tensor.
        tokens: Number of tokens to generate.

    """
    if isinstance(model, _GenerateCapableProtocol) and callable(model.generate):
        _ = model.generate(dummy_inputs, max_new_tokens=tokens, min_new_tokens=tokens)


_LAST_LOOP_METRICS: contextvars.ContextVar[DetailedLoopMetrics | None] = contextvars.ContextVar(
    "pytorch_last_loop_metrics",
    default=None,
)


def _run_benchmark_pass(
    model: TorchCausalLMProtocol | CausalLMProtocol | _BenchmarkCallableProtocol,
    device: str,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
) -> tuple[float, float, float]:
    """Execute synchronized warmup and per-step timed benchmark runs.

    Args:
        model: Loaded PyTorch causal LM instance.
        device: Target execution device string.
        batch_size: Batch size for synthetic benchmark inputs.
        num_runs: Number of timed benchmark iterations.
        warmup_steps: Number of untimed warmup iterations.
        mode: Benchmark mode (`'prefill'`, `'decode'`, `'generate'`, or `'end-to-end'`).
        max_new_tokens: Number of tokens to generate in non-prefill modes.

    Returns:
        Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    """
    canon_mode = normalize_benchmark_mode(mode, backend="pytorch")
    adapter = _runtime()
    adapter.manual_seed(42)

    vocab_bound = _resolve_vocab_bound(model)
    dummy_inputs = adapter.randint(1, vocab_bound, (batch_size, 32), device)
    adapter.reset_peak_memory_stats(device)

    def _run_pt_step(gen_tokens: int) -> None:
        """Execute one PyTorch prefill or decode step under `no_grad`."""
        with adapter.no_grad():
            if canon_mode == "prefill":
                _ = model(dummy_inputs)
            else:
                _call_decode_generate(model, dummy_inputs, gen_tokens)

    # On XLA TPUs, warmup must use `max_new_tokens` so XLA compiles the exact decode graph shape.
    warmup_gen_tokens = max_new_tokens if device.startswith("xla") else 2
    metrics = execute_detailed_benchmark_loop(
        lambda: _run_pt_step(max_new_tokens),
        warmup_steps=warmup_steps,
        num_runs=num_runs,
        tokens_per_iter=32 if canon_mode == "prefill" else max_new_tokens,
        batch_size=batch_size,
        warmup_step_fn=lambda: _run_pt_step(warmup_gen_tokens),
        sync_fn=lambda: _sync_cuda(device),
    )
    _LAST_LOOP_METRICS.set(metrics)

    memory_mb = _get_memory_mb(model, device)
    return (float(metrics.tokens_per_sec), float(metrics.latency_ms), float(memory_mb))


def benchmark_model(
    model_name: str,
    hardware: str = "gpu",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the PyTorch backend (`hf_transformers` or `pytorch_native`).

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'tpu-v6e'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`, `warmup_steps`, `mode`, `strict_hardware`, `backend_alias`.

    Returns:
        A dictionary containing benchmark metrics and status.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.

    """
    backend_alias = str(kwargs.get("backend_alias", "pytorch"))
    strict_hw = bool(kwargs.get("strict_hardware", False))
    missing_hf = backend_alias != "pytorch_native" and AutoModelForCausalLM is None
    if torch is None or missing_hf:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.", backend=backend_alias)

    def _run() -> BenchmarkTelemetry:
        """Load PyTorch model, execute synchronized benchmark pass, and collect XLA/GPU telemetry.

        Returns:
            `BenchmarkTelemetry` across timed runs.

        """
        dtype = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))
        num_runs = int(str(kwargs.get("num_runs", 5)))

        (model, device) = _load_pytorch_model_and_device(
            model_name,
            hardware,
            dtype=dtype,
            backend_alias=backend_alias,
            strict_hardware=strict_hw,
        )
        ctx_tok = _LAST_LOOP_METRICS.set(None)
        try:
            tps, lat_ms, memory_mb = _run_benchmark_pass(
                model,
                device,
                batch_size,
                num_runs,
                warmup_steps,
                mode,
                max_new_tokens,
            )
            loop_metrics = _LAST_LOOP_METRICS.get()
        finally:
            _LAST_LOOP_METRICS.reset(ctx_tok)
        dev_count = _runtime().get_device_count(hardware) if normalize_hardware_family(hardware) != "cpu" else 1
        peak_hbm = memory_mb if device.startswith(("xla", "cuda")) else None
        return BenchmarkTelemetry(
            tokens_per_sec=tps,
            latency_ms=lat_ms,
            memory_mb=memory_mb,
            resolved_device=device,
            device_count=dev_count,
            compile_latency_ms=loop_metrics.compile_latency_ms if loop_metrics is not None else None,
            peak_hbm_mb=peak_hbm,
            latency_samples=loop_metrics.latency_samples if loop_metrics is not None else (),
        )

    return run_benchmark_wrapper(
        backend_name=backend_alias,
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
        strict_hardware=strict_hw,
        raise_on_error=bool(kwargs["raise_on_error"]) if "raise_on_error" in kwargs else None,
    )


__all__ = [
    "_get_device",
    "_get_memory_mb",
    "_load_pytorch_model_and_device",
    "_resolve_vocab_bound",
    "_run_benchmark_pass",
    "_sync_cuda",
    "benchmark_model",
]
