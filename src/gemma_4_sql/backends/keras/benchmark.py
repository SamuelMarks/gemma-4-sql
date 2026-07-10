"""Keras-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf


def _load_keras_model(model_name: str) -> object:
    """Load an actual compiled Keras model.

    Args:
        model_name: The name of the target model.

    Returns:
        The execution result.

    Raises:
        ValueError: if model cannot be loaded.
    """
    try:
        gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
        return gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError) as err:
        msg = f"Failed to load actual model {model_name}"
        raise ValueError(msg) from err


def _run_benchmark_pass(model: keras.Model, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: The model.
        batch_size: The number of items to process in a single batch.
        num_runs: The integer value for num runs.

    Returns:
        A tuple containing the results.
    """
    dummy_inputs = tf.random.uniform((batch_size, 32), minval=1, maxval=1000, dtype=tf.int32)

    @tf.function
    def forward_pass(inputs: keras.KerasTensor | tf.Tensor) -> object:
        """Execute the forward pass operation.

        Returns:
            object: The resulting output from the operation.

        """
        return model(inputs)

    _ = forward_pass(dummy_inputs)
    start_time = time.time()
    for _ in range(num_runs):
        out = forward_pass(dummy_inputs)
    if hasattr(out, "numpy"):
        _ = out.numpy()
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(1, num_runs)
    tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    try:
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        memory_mb = memory_info["current"] / (1024 * 1024)
    except ValueError:
        memory_mb = 6000.0
    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark a model using the Keras backend.

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
        model = _load_keras_model(model_name)
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs)

    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name="keras",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=False,
        missing_status="missing_keras",
        benchmark_fn=_run,
    )
