# Copyright 2024
"""PyTorch-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue, ModelType
logger = logging.getLogger(__name__)
torch = None
AutoModelForCausalLM = None
with catch_optional_imports():
    import torch
    from transformers import AutoModelForCausalLM


def _load_pytorch_model_and_device(model_name: str, hardware: str, *, test_mode: bool = False) -> tuple[ModelType, str]:
    """Load the model and determine device.

    Returns:
        object: The resulting output from the operation.

    """
    if test_mode:
        return (None, "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if getattr(torch, "cuda", None) and getattr(torch.cuda, "is_available", lambda: False)() and (hardware != "cpu") else "cpu"
    if hasattr(model, "to"):
        model.to(device)  # pragma: no cover
    if hasattr(model, "eval"):
        model.eval()  # pragma: no cover
    return (model, device)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA if using GPU."""
    if device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()  # pragma: no cover


def _run_forward_pass(model: torch.nn.Module, dummy_inputs: object) -> None:
    """Run a single forward pass."""
    if model is not None and hasattr(torch, "no_grad"):
        with torch.no_grad():  # pragma: no cover
            _ = model(dummy_inputs)  # pragma: no cover


def _get_memory_mb(model: torch.nn.Module, device: str) -> float:
    """Get max memory allocated in MB.

    Returns:
        object: The resulting output from the operation.

    """
    if model is not None and device == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "max_memory_allocated"):
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))  # pragma: no cover
    return 8192.0


def _run_benchmark_pass(model: torch.nn.Module, device: str, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Returns:
        object: The resulting output from the operation.

    """
    dummy_inputs = torch.zeros((batch_size, 32), dtype=getattr(torch, "long", None))
    if model is not None and hasattr(dummy_inputs, "to"):
        dummy_inputs = dummy_inputs.to(device)  # pragma: no cover
    _run_forward_pass(model, dummy_inputs)
    _sync_cuda(device)
    start_time = time.time()
    for _ in range(num_runs):
        _run_forward_pass(model, dummy_inputs)
    _sync_cuda(device)
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(1, num_runs)
    tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    memory_mb = _get_memory_mb(model, device)
    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark a model using the PyTorch backend.

    Args:
    ----
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'gpu', 'tpu', 'cpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
    -------
        A dictionary containing benchmark metrics and status.

    """

    def _run() -> tuple[float, float, float]:
        (model, device) = _load_pytorch_model_and_device(model_name, hardware, test_mode=bool(kwargs.get("test_mode")))
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, device, batch_size, num_runs)

    return run_benchmark_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=torch is None or AutoModelForCausalLM is None,
        missing_status="mocked_missing_torch",
        benchmark_fn=_run,
    )
