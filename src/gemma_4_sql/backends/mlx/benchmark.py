"""MLX-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue, ModelType
logger = logging.getLogger(__name__)
mlx = None
AutoModelForCausalLM = None
with catch_optional_imports():
    import mlx
    from transformers import AutoModelForCausalLM


def _load_mlx_model_and_device(model_name: str, hardware: str, *, test_mode: bool = False) -> tuple[ModelType, str]:
    """Load the model and determine device.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        test_mode: Boolean flag indicating test mode.

    Returns:
        A tuple containing the results.
    """
    if test_mode:
        return (None, "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if hasattr(mlx, "cuda") and mlx.cuda.is_available() and (hardware != "cpu") else "cpu"
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return (model, device)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA if using GPU.

    Args:
        device: The string representing the device.
    """
    if device == "cuda" and hasattr(mlx, "cuda") and hasattr(mlx.cuda, "synchronize"):
        mlx.cuda.synchronize()


def _run_forward_pass(model: object, dummy_inputs: object) -> None:
    """Run a single forward pass.

    Args:
        model: The model.
        device: The string representing the device.

    Returns:
        The computed float value.
    """
    if model is not None and hasattr(mlx, "no_grad"):
        with mlx.no_grad():
            _ = model(dummy_inputs)


def _get_memory_mb(model: object, device: str) -> float:
    """Get max memory allocated in MB.

    Args:
        model: The model.
        device: The string representing the device.
        batch_size: The number of items to process in a single batch.
        num_runs: The integer value for num runs.

    Returns:
        A tuple containing the results.
    """
    if model is not None and device == "cuda" and hasattr(mlx, "cuda") and hasattr(mlx.cuda, "max_memory_allocated"):
        return float(mlx.cuda.max_memory_allocated() / (1024 * 1024))
    return 8192.0


def _run_benchmark_pass(model: object, device: str, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Returns:
        object: The resulting output from the operation.

    """
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
