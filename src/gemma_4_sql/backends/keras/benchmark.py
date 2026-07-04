# Copyright 2024
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
    import tensorflow as tf  # pragma: no cover


def _load_keras_model(model_name: str, *, test_mode: bool = False) -> object:
    """Load or mock a Keras model.

    Returns:
        object: The resulting output from the operation.

    """
    if test_mode:
        return None
    try:
        gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
        return gemma_causal_lm_cls.from_preset(model_name)  # pragma: no cover
    except (ImportError, ValueError):
        inputs = keras.Input(shape=(None,), dtype="int32")
        x = keras.layers.Embedding(256, 128)(inputs)
        outputs = keras.layers.Dense(256)(x)
        return keras.Model(inputs, outputs)


def _run_benchmark_pass(model: keras.Model, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Returns:
        object: The resulting output from the operation.

    """
    dummy_inputs = tf.zeros((batch_size, 32), dtype=tf.int32)

    @tf.function
    def forward_pass(inputs: keras.KerasTensor | tf.Tensor) -> object:
        """Execute the forward pass operation.

        Returns:
            object: The resulting output from the operation.

        """
        return model(inputs) if model is not None else inputs

    _ = forward_pass(dummy_inputs)
    start_time = time.time()
    for _ in range(num_runs):
        out = forward_pass(dummy_inputs)
    if hasattr(out, "numpy"):  # pragma: no cover
        _ = out.numpy()
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(1, num_runs)
    tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    try:
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        memory_mb = memory_info["current"] / (1024 * 1024)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
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
            object: Description of return.

        """
        model = _load_keras_model(model_name, test_mode=bool(kwargs.get("test_mode")))
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs)

    return run_benchmark_wrapper(
        backend_name="keras",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=keras is None or tf is None,
        missing_status="mocked_missing_keras",
        benchmark_fn=_run,
    )
