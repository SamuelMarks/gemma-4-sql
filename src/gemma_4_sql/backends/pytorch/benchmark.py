"""PyTorch-specific benchmarking pipeline."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_benchmark import get_current_rss_mb, run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRuntimeAdapter
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    CausalLMProtocol,
    DTypeProtocol,
    ForwardArg,
    ForwardOutput,
    JSONValue,
    TorchCausalLMProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _BenchmarkCallableProtocol(Protocol):
    """Protocol for a callable benchmark model."""

    def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> ForwardOutput:
        """Execute a forward pass."""
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
        """Load a pretrained Hugging Face causal language model."""
        ...


torch = optional_import("torch")
AutoModelForCausalLM: _HFAutoModelLoaderProtocol | None = optional_attr(
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
    model: TorchCausalLMProtocol,
    device: str,
    torch_dtype: DTypeProtocol | str | None = None,
) -> None:
    """Move `model` to `torch_dtype` and `device`, then set evaluation mode when supported.

    Args:
        model: Target PyTorch causal language model.
        device: Target device string.
        torch_dtype: Optional resolved torch dtype.

    """
    with contextlib.suppress(AttributeError, TypeError):
        if torch_dtype is not None:
            model.to(torch_dtype)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        model.to(device)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    with contextlib.suppress(AttributeError, TypeError):
        model.eval()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]


def _load_pytorch_model_and_device(
    model_name: str,
    hardware: str,
    *,
    dtype: str = "bfloat16",
    backend_alias: str = "pytorch",
) -> tuple[TorchCausalLMProtocol, str]:
    """Load the model and determine device.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        dtype: The string representing dtype.
        backend_alias: The alias used (`'pytorch'`, `'pytorch_hf'`, `'pytorch_native'`).

    Returns:
        A tuple of `(model, device)`.

    """
    adapter = _runtime()
    device = adapter.get_device(hardware)
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

    model = adapter.compile_model(model)
    return (model, device)


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
    with contextlib.suppress(AttributeError, TypeError):
        raw_vocab = model.config.vocab_size  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
        if isinstance(raw_vocab, int) and raw_vocab > 1:
            return raw_vocab
    return 256000


def _run_benchmark_pass(
    model: TorchCausalLMProtocol | CausalLMProtocol | _BenchmarkCallableProtocol,
    device: str,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: Loaded PyTorch causal LM instance.
        device: Target execution device string.
        batch_size: Batch size for synthetic benchmark inputs.
        num_runs: Number of timed benchmark iterations.
        warmup_steps: Number of untimed warmup iterations.
        mode: Benchmark mode (`'prefill'` or `'decode'`).
        max_new_tokens: Number of tokens to generate in decode mode.

    Returns:
        Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    """
    adapter = _runtime()
    adapter.manual_seed(42)

    vocab_bound = _resolve_vocab_bound(model)
    dummy_inputs = adapter.randint(1, vocab_bound, (batch_size, 32), device)
    adapter.reset_peak_memory_stats(device)

    for _ in range(warmup_steps):
        with adapter.no_grad():
            if mode == "prefill":
                _ = model(dummy_inputs)  # pyright: ignore[reportCallIssue]
            else:
                with contextlib.suppress(AttributeError, TypeError):
                    _ = model.generate(dummy_inputs, max_new_tokens=2, min_new_tokens=2)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    _sync_cuda(device)

    start_time = time.time()
    for _ in range(num_runs):
        with adapter.no_grad():
            if mode == "prefill":
                _ = model(dummy_inputs)  # pyright: ignore[reportCallIssue]
            else:
                with contextlib.suppress(AttributeError, TypeError):
                    _ = model.generate(dummy_inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    _sync_cuda(device)
    end_time = time.time()

    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(1, num_runs)

    if mode == "prefill":
        tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    else:
        tokens_per_sec = max_new_tokens * batch_size * num_runs / max(end_time - start_time, 1e-09)

    memory_mb = _get_memory_mb(model, device)
    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(
    model_name: str,
    hardware: str = "gpu",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the PyTorch backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
        A dictionary containing benchmark metrics and status.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.

    """
    backend_alias = str(kwargs.get("backend_alias", "pytorch"))

    def _run() -> tuple[float, float, float]:
        """Load PyTorch model and execute benchmark pass.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        dtype = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))

        (model, device) = _load_pytorch_model_and_device(model_name, hardware, dtype=dtype, backend_alias=backend_alias)
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, device, batch_size, num_runs, warmup_steps, mode, max_new_tokens)

    if torch is None or AutoModelForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name=backend_alias,
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
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
