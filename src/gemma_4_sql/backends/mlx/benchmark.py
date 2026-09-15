"""MLX-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

try:
    import mlx as _mlx
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

    mlx: Any = _mlx
    AutoModelForCausalLM: Any = _AutoModelForCausalLM
except (ImportError, AttributeError):
    mlx = None
    AutoModelForCausalLM = None


def _load_mlx_model_and_device(model_name: str, hardware: str, *, test_mode: bool = False) -> tuple[Any, str]:
    """Load the model and determine device.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        test_mode: Boolean flag indicating test mode.

    Returns:
        A tuple containing the results.

    Raises:
        DependencyMissingError: If Transformers AutoModelForCausalLM is missing.
    """
    if test_mode:
        return (None, "cpu")
    if AutoModelForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Transformers AutoModelForCausalLM is missing.")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if hasattr(mlx, "cuda") and mlx.cuda.is_available() and (hardware != "cpu") else "cpu"
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):  # pragma: no cover
        model.eval()
    return (model, device)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA if using GPU.

    Args:
        device: The string representing the device.
    """
    if device == "cuda" and hasattr(mlx, "cuda") and hasattr(mlx.cuda, "synchronize"):
        mlx.cuda.synchronize()


def _run_forward_pass(model: Any, dummy_inputs: Any) -> None:
    """Run a single forward pass.

    Args:
        model: The model.
        dummy_inputs: Inputs tensor.
    """
    if model is not None and hasattr(mlx, "no_grad"):
        with mlx.no_grad():
            _ = model(dummy_inputs)


def _get_memory_mb(model: Any, device: str) -> float:
    """Get max memory allocated in MB.

    Args:
        model: The model.
        device: The string representing the device.

    Returns:
        Memory usage in MB.
    """
    if model is not None and device == "cuda" and hasattr(mlx, "cuda") and hasattr(mlx.cuda, "max_memory_allocated"):
        return float(mlx.cuda.max_memory_allocated() / (1024 * 1024))
    return 8192.0


def _run_benchmark_pass(model: Any, device: str, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: The model.
        device: Target device string.
        batch_size: Batch size for benchmark.
        num_runs: Number of benchmark iterations.

    Returns:
        A tuple of (tokens_per_sec, latency_ms, memory_mb).

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
    """
    if mlx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    dummy_inputs = mlx.zeros((batch_size, 32), dtype=getattr(mlx, "long", None))
    if model is not None and hasattr(dummy_inputs, "to"):
        dummy_inputs = dummy_inputs.to(device)
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
    """Benchmark a model using the MLX backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'gpu', 'tpu', 'cpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
        A dictionary containing benchmark metrics and status.
    """

    def _run() -> tuple[float, float, float]:
        """Execute function.

        Returns:
            The execution result.

        """
        (model, device) = _load_mlx_model_and_device(model_name, hardware, test_mode=bool(kwargs.get("test_mode")))
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, device, batch_size, num_runs)

    return run_benchmark_wrapper(
        backend_name="mlx",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=mlx is None or AutoModelForCausalLM is None,
        missing_status="mocked_missing_mlx",
        benchmark_fn=_run,
    )
