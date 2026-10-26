"""PyTorch-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from gemma_4_sql.backends.common_benchmark import get_current_rss_mb, run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.type_hints import BenchmarkResultDict, JSONValue, TorchCausalLMProtocol

logger = logging.getLogger(__name__)

torch = optional_import("torch")
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")


def _get_device(hardware: str) -> str:
    """Provide the correct device based on hardware string.

    Args:
        hardware: Target hardware string.

    Returns:
        Device identifier string (cpu, cuda, or mps).

    """
    if hardware == "cpu":
        return "cpu"
    if getattr(torch, "cuda", None) and getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    if getattr(torch, "backends", None) and getattr(torch.backends, "mps", None) and getattr(torch.backends.mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _load_pytorch_model_and_device(model_name: str, hardware: str, *, dtype: str = "bfloat16", backend_alias: str = "pytorch") -> tuple[Any, str]:
    """Load the model and determine device.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        dtype: The string representing dtype.
        backend_alias: The alias used (pytorch, pytorch_hf, pytorch_native).

    Returns:
        A tuple containing the results.

    """
    device = _get_device(hardware)
    torch_dtype = getattr(torch, dtype, None) if hasattr(torch, dtype) else getattr(torch, "float32", None)

    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4Config, Gemma4ForCausalLM

        # In a real scenario, we might want to load weights or adjust config based on model_name
        config = Gemma4Config()
        model = Gemma4ForCausalLM(config)
        if hasattr(model, "to"):
            if torch_dtype is not None:
                model.to(torch_dtype)
            model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        if hasattr(model, "to"):
            model.to(device)

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("torch.compile failed: %s", e)
    return (model, device)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA if using GPU.

    Args:
        device: The string representing the device.

    """
    if device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _get_memory_mb(model: TorchCausalLMProtocol, device: str) -> float:
    """Provide max memory allocated in MB.

    Args:
        model: The model.
        device: The string representing the device.

    Returns:
        The max memory.

    """
    _ = model
    if device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "max_memory_allocated"):
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        return float(torch.mps.driver_allocated_memory() / (1024 * 1024))
    return get_current_rss_mb()


def _run_benchmark_pass(model: TorchCausalLMProtocol, device: str, batch_size: int, num_runs: int, warmup_steps: int, mode: str, max_new_tokens: int) -> tuple[float, float, float]:
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
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(42)

    dummy_inputs = torch.randint(1, 256000, (batch_size, 32), dtype=getattr(torch, "long", None))
    if hasattr(dummy_inputs, "to"):
        dummy_inputs = dummy_inputs.to(device)

    if device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup_steps):
        if hasattr(torch, "no_grad"):
            with torch.no_grad():
                if mode == "prefill":
                    _ = model(dummy_inputs)
                elif hasattr(model, "generate"):
                    _ = model.generate(dummy_inputs, max_new_tokens=2, min_new_tokens=2)
    _sync_cuda(device)

    start_time = time.time()
    for _ in range(num_runs):
        if hasattr(torch, "no_grad"):
            with torch.no_grad():
                if mode == "prefill":
                    _ = model(dummy_inputs)
                elif hasattr(model, "generate"):
                    _ = model.generate(dummy_inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens)
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


def benchmark_model(model_name: str, hardware: str = "gpu", batch_size: int = 32, **kwargs: JSONValue) -> BenchmarkResultDict:
    """Benchmark a model using the PyTorch backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'gpu', 'tpu', 'cpu').
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
        missing_deps=False,
        benchmark_fn=_run,
    )
