"""Keras-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.type_hints import BenchmarkResultDict, JSONValue, KerasCausalLMProtocol, TensorType

logger = logging.getLogger(__name__)

keras = optional_import("keras")
tf = optional_import("tensorflow")


def _load_keras_model(model_name: str, dtype: str) -> KerasCausalLMProtocol:
    """Load and configure a compiled KerasNLP `GemmaCausalLM` model for benchmarking.

    Args:
        model_name: Target KerasNLP preset or model identifier.
        dtype: Floating-point precision policy string (e.g., `'bfloat16'`, `'float32'`).

    Returns:
        Loaded `GemmaCausalLM` model instance.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        InferenceError: If the model preset cannot be loaded.

    """
    if keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")
    try:
        keras.config.set_floatx(dtype)
        gemma_causal_lm_cls = optional_attr("keras_nlp.models", "GemmaCausalLM")
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        loaded_model: KerasCausalLMProtocol = gemma_causal_lm_cls.from_preset(model_name)
        return loaded_model
    except (ImportError, ValueError) as err:
        from gemma_4_sql.exceptions import InferenceError

        msg = f"Failed to load actual model {model_name}"
        raise InferenceError(msg, backend="keras", model_name=model_name, cause=err) from err


def _get_device_str(hardware: str) -> str:
    """Map hardware target identifier to a TensorFlow device placement string.

    Args:
        hardware: Target hardware identifier (`'cpu'`, `'gpu'`, `'tpu'`).

    Returns:
        TensorFlow device specification string (`'/CPU:0'`, `'/GPU:0'`, or `'/TPU:0'`).

    """
    if tf is None:
        return "/CPU:0"
    if hardware == "cpu":
        return "/CPU:0"
    if hardware == "gpu" and tf.config.list_physical_devices("GPU"):
        return "/GPU:0"
    if hardware == "tpu" and tf.config.list_physical_devices("TPU"):
        return "/TPU:0"
    return "/CPU:0"


def _run_benchmark_pass(
    model: Any,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
    hardware: str,
) -> tuple[float, float, float]:
    """Execute the warmup and timed benchmark pass for a Keras model.

    Args:
        model: Instantiated Keras model.
        batch_size: Number of sequences processed per batch.
        num_runs: Number of timed benchmark iterations.
        warmup_steps: Number of untimed warmup iterations before timing begins.
        mode: Execution mode (`'prefill'` or `'generate'`).
        max_new_tokens: Number of tokens generated in `'generate'` mode.
        hardware: Target hardware identifier (`'cpu'`, `'gpu'`, `'tpu'`).

    Returns:
        A 3-tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    Raises:
        DependencyMissingError: If TensorFlow dependencies are missing.

    """
    if tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorFlow dependencies are missing.")

    tf.random.set_seed(42)
    device_str = _get_device_str(hardware)

    with tf.device(device_str):
        dummy_inputs = tf.random.uniform((batch_size, 32), minval=1, maxval=256000, dtype=tf.int32)

        @tf.function(jit_compile=True)
        def forward_pass(inputs: TensorType) -> object:
            """Run JIT-compiled forward pass.

            Args:
                inputs: Input token tensor.

            Returns:
                Output logits tensor.

            """
            return model(inputs)

        if "GPU" in device_str:
            try:
                tf.config.experimental.reset_memory_stats(device_str.replace("/", ""))
            except ValueError:
                pass

        for _ in range(warmup_steps):
            if mode == "prefill":
                out = forward_pass(dummy_inputs)
                if hasattr(out, "numpy"):
                    _ = out.numpy()
            else:
                if hasattr(model, "generate"):
                    out = model.generate(dummy_inputs, max_length=32 + 2)
                    if hasattr(out, "numpy"):
                        _ = out.numpy()

        start_time = time.perf_counter()
        for _ in range(num_runs):
            if mode == "prefill":
                out = forward_pass(dummy_inputs)
                if hasattr(out, "numpy"):
                    _ = out.numpy()
            else:
                if hasattr(model, "generate"):
                    out = model.generate(dummy_inputs, max_length=32 + max_new_tokens)
                    if hasattr(out, "numpy"):
                        _ = out.numpy()
        end_time = time.perf_counter()

        elapsed_s = max(end_time - start_time, 1e-09)
        total_time_ms = elapsed_s * 1000.0
        latency_ms = total_time_ms / max(1, num_runs)

        if mode == "prefill":
            tokens_per_sec = 32 * batch_size * num_runs / elapsed_s
        else:
            tokens_per_sec = max_new_tokens * batch_size * num_runs / elapsed_s

        try:
            if "GPU" in device_str:
                memory_info = tf.config.experimental.get_memory_info(device_str.replace("/", ""))
                memory_mb = memory_info.get("peak", memory_info.get("current", 0.0)) / (1024 * 1024)
            else:
                memory_mb = 0.0
        except ValueError:
            memory_mb = 0.0

        return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32, **kwargs: JSONValue) -> BenchmarkResultDict:
    """Benchmark a model using the Keras backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional options such as `num_runs`, `dtype`, `mode`, `max_new_tokens`, `warmup_steps`.

    Returns:
        Structured `BenchmarkResultDict` containing throughput, latency, memory, and status.

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.

    """

    def _run() -> tuple[float, float, float]:
        """Load Keras model and execute benchmark pass.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        dtype = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))

        model = _load_keras_model(model_name, dtype=dtype)
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs, warmup_steps, mode, max_new_tokens, hardware)

    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name="keras",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=False,
        benchmark_fn=_run,
    )
