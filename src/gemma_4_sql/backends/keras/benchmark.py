"""Keras-specific benchmarking pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import KerasScoreType
from gemma_4_sql.backends.common_benchmark import (
    execute_timed_benchmark_loop,
    get_current_rss_mb,
    run_benchmark_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    DTypeProtocol,
    ForwardArg,
    ForwardOutput,
    JSONValue,
    KerasCausalLMProtocol,
    TensorProtocol,
    TensorType,
    TfModuleProtocol,
    TfTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier.

        Args:
            model_name: Preset identifier string.

        Returns:
            Loaded `KerasCausalLMProtocol` instance.

        """
        ...


@runtime_checkable
class _KerasCallableModelProtocol(Protocol):
    """Protocol for a callable Keras benchmark model."""

    def __call__(self, inputs: TensorType, /) -> ForwardOutput | str:
        """Execute forward pass on `inputs`.

        Args:
            inputs: Input token tensor.

        Returns:
            Output logits tensor or string.

        """
        ...


@runtime_checkable
class _KerasGenerateModelProtocol(Protocol):
    """Protocol for a Keras model supporting `.generate(inputs, max_length=...)`."""

    def generate(
        self,
        *args: ForwardArg | Sequence[str],
        **kwargs: ForwardArg,
    ) -> (
        TensorProtocol
        | TfTensorProtocol
        | Sequence[int]
        | Sequence[str]
        | Mapping[str, KerasScoreType]
        | tuple[str, KerasScoreType]
        | str
    ):
        """Generate token sequences.

        Args:
            *args: Input prompt string, batch of prompts, or token tensor.
            **kwargs: Decoding options such as `max_length`.

        Returns:
            Generated output tensor or string.

        """
        ...


@runtime_checkable
class _TFIdentityNamespaceProtocol(Protocol):
    """Protocol for `tensorflow` module exposing `identity`."""

    def identity(
        self,
        input_val: ForwardOutput
        | TfTensorProtocol
        | Sequence[int]
        | Sequence[str]
        | Mapping[str, KerasScoreType]
        | tuple[str, KerasScoreType]
        | str,
        /,
    ) -> ForwardOutput | TfTensorProtocol | str:
        """Materialize tensor evaluation via `tf.identity`.

        Args:
            input_val: Forward or generation output.

        Returns:
            Materialized output.

        """
        ...


@runtime_checkable
class _KerasConfigNamespaceProtocol(Protocol):
    """Protocol for `keras.config` exposing `set_floatx`."""

    def set_floatx(self, dtype: str, /) -> None:
        """Set global Keras default float precision policy.

        Args:
            dtype: Floating-point precision name (e.g., `'float32'`, `'bfloat16'`).

        """
        ...


@runtime_checkable
class _KerasBenchmarkModuleProtocol(Protocol):
    """Protocol for `keras` module exposing `.config`."""

    @property
    def config(self) -> _KerasConfigNamespaceProtocol:
        """Provide `keras.config` namespace."""
        ...


@runtime_checkable
class _TFExperimentalConfigProtocol(Protocol):
    """Protocol for `tf.config.experimental` GPU memory statistics helpers."""

    def reset_memory_stats(self, device: str, /) -> None:
        """Reset peak memory tracking statistics on `device`.

        Args:
            device: TensorFlow device string (e.g., `'GPU:0'`).

        """
        ...

    def get_memory_info(self, device: str, /) -> Mapping[str, float | int]:
        """Return `'current'` and `'peak'` byte allocations on `device`.

        Args:
            device: TensorFlow device string (e.g., `'GPU:0'`).

        Returns:
            Mapping with `'current'` and `'peak'` byte counts.

        """
        ...


@runtime_checkable
class _TFPhysicalDeviceProtocol(Protocol):
    """Structural protocol satisfied by TensorFlow `PhysicalDevice` descriptors or device strings."""


@runtime_checkable
class _TFConfigNamespaceProtocol(Protocol):
    """Protocol for `tf.config` physical device discovery and `.experimental` memory stats."""

    @property
    def experimental(self) -> _TFExperimentalConfigProtocol:
        """Provide `tf.config.experimental` namespace."""
        ...

    def list_physical_devices(
        self,
        device_type: str | None = None,
        /,
    ) -> Sequence[_TFPhysicalDeviceProtocol | str]:
        """List visible physical accelerator devices matching `device_type`.

        Args:
            device_type: Optional device class (`'GPU'`, `'TPU'`, `'CPU'`).

        Returns:
            Sequence of physical device descriptors.

        """
        ...


@runtime_checkable
class _TFRandomNamespaceProtocol(Protocol):
    """Protocol for `tf.random` seeding and uniform integer tensor generation."""

    def set_seed(self, seed: int, /) -> None:
        """Set global TensorFlow random seed.

        Args:
            seed: Deterministic integer seed.

        """
        ...

    def uniform(
        self,
        shape: tuple[int, int],
        *,
        minval: int,
        maxval: int,
        dtype: DTypeProtocol,
    ) -> TensorType:
        """Sample uniform random integer tensor of `shape`.

        Args:
            shape: 2D `(batch_size, seq_len)` shape tuple.
            minval: Inclusive minimum token ID.
            maxval: Exclusive maximum token ID.
            dtype: Target integer dtype.

        Returns:
            Sampled input token `TensorType`.

        """
        ...


@runtime_checkable
class _TFBenchmarkModuleProtocol(Protocol):
    """Protocol for `tensorflow` device placement, RNG, XLA `tf.function`, and memory stats."""

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `tf.int32` dtype."""
        ...

    @property
    def config(self) -> _TFConfigNamespaceProtocol:
        """Provide `tf.config` namespace."""
        ...

    @property
    def random(self) -> _TFRandomNamespaceProtocol:
        """Provide `tf.random` namespace."""
        ...

    def device(self, device_name: str, /) -> AbstractContextManager[None]:
        """Return a device placement context manager.

        Args:
            device_name: TensorFlow device specification string.

        Returns:
            Device placement context manager.

        """
        ...

    def function[FnT](
        self,
        *,
        jit_compile: bool = False,
    ) -> Callable[[FnT], FnT]:
        """Return a decorator compiling a Python callable with optional XLA `jit_compile`.

        Args:
            jit_compile: Whether to enable XLA JIT compilation.

        Returns:
            Decorator preserving `FnT`.

        """
        ...


keras: _KerasBenchmarkModuleProtocol | None = optional_import("keras", _KerasBenchmarkModuleProtocol)
tf: _TFBenchmarkModuleProtocol | None = optional_import("tensorflow", _TFBenchmarkModuleProtocol)


def _load_keras_model(model_name: str, dtype: str) -> KerasCausalLMProtocol:
    """Load and configure a compiled KerasNLP `GemmaCausalLM` model for benchmarking.

    Args:
        model_name: Target KerasNLP preset or model identifier.
        dtype: Floating-point precision policy string (e.g., `'bfloat16'`, `'float32'`).

    Returns:
        Loaded `GemmaCausalLM` model instance.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        ImportError: Internally raised and wrapped if `keras_nlp.models.GemmaCausalLM` is unavailable.
        InferenceError: If the model preset cannot be loaded.

    """
    if keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")
    try:
        keras.config.set_floatx(dtype)
        gemma_causal_lm_cls = import_optional_symbol(
            "keras_nlp.models",
            "GemmaCausalLM",
            _GemmaCausalLMLoaderProtocol,
        )
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


type KerasBenchmarkModel = KerasCausalLMProtocol | _KerasGenerateModelProtocol | Callable[..., ForwardOutput | str]


def _materialize_tf_output(
    out: ForwardOutput
    | TfTensorProtocol
    | Sequence[int]
    | Sequence[str]
    | Mapping[str, KerasScoreType]
    | tuple[str, KerasScoreType]
    | str,
) -> None:
    """Materialize `out` via `tf.identity` when `tf.identity` is available and accepts `out`.

    Args:
        out: Forward or generation output value.

    """
    if isinstance(tf, _TFIdentityNamespaceProtocol) and callable(tf.identity):
        try:
            _ = tf.identity(out)
        except (AttributeError, TypeError):
            pass


def _run_benchmark_pass(
    model: KerasBenchmarkModel,
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
        def forward_pass(inputs: TensorType) -> ForwardOutput | str:
            """Run JIT-compiled forward pass.

            Args:
                inputs: Input token tensor.

            Returns:
                Output logits tensor or string.

            """
            assert isinstance(model, (_KerasCallableModelProtocol, KerasCausalLMProtocol))
            return model(inputs)

        if "GPU" in device_str:
            try:
                tf.config.experimental.reset_memory_stats(device_str.replace("/", ""))
            except ValueError:
                pass

        def _run_keras_step(gen_tokens: int) -> None:
            """Execute one Keras prefill or generate step and materialize output."""
            if mode == "prefill":
                out = forward_pass(dummy_inputs)
                _materialize_tf_output(out)
            elif isinstance(model, _KerasGenerateModelProtocol) and callable(model.generate):
                out = model.generate(dummy_inputs, max_length=32 + gen_tokens)
                _materialize_tf_output(out)

        tokens_per_sec, latency_ms = execute_timed_benchmark_loop(
            lambda: _run_keras_step(max_new_tokens),
            warmup_steps=warmup_steps,
            num_runs=num_runs,
            tokens_per_iter=32 if mode == "prefill" else max_new_tokens,
            batch_size=batch_size,
            warmup_step_fn=lambda: _run_keras_step(2),
            use_perf_counter=True,
        )

        try:
            if "GPU" in device_str:
                memory_info = tf.config.experimental.get_memory_info(device_str.replace("/", ""))
                memory_mb = memory_info.get("peak", memory_info.get("current", 0.0)) / (1024 * 1024)
            else:
                memory_mb = get_current_rss_mb()
        except ValueError:
            memory_mb = get_current_rss_mb()

        return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_keras_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    *,
    model_loader: Callable[[str, str], KerasBenchmarkModel] | None = None,
    options: Mapping[str, JSONValue] | None = None,
) -> BenchmarkResultDict:
    """Benchmark a model using the Keras backend with optional injected `model_loader`.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        model_loader: Optional injected `(model_name, dtype) -> model` loader callable.
        options: Optional benchmark configuration mapping (`num_runs`, `dtype`, `mode`,
            `max_new_tokens`, `warmup_steps`).

    Returns:
        Structured `BenchmarkResultDict` containing throughput, latency, memory, and status.

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.

    """
    opts: Mapping[str, JSONValue] = options if options is not None else {}

    def _run() -> tuple[float, float, float]:
        """Load Keras model and execute benchmark pass.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        dtype = str(opts.get("dtype", "bfloat16"))
        mode = str(opts.get("mode", "prefill"))
        max_new_tokens = int(str(opts.get("max_new_tokens", 128)))
        warmup_steps = int(str(opts.get("warmup_steps", 5)))

        load_fn = model_loader if model_loader is not None else _load_keras_model
        model = load_fn(model_name, dtype)
        num_runs = int(str(opts.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs, warmup_steps, mode, max_new_tokens, hardware)

    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name="keras",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
    )


def benchmark_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the Keras backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional options such as `num_runs`, `dtype`, `mode`, `max_new_tokens`, `warmup_steps`.

    Returns:
        Structured `BenchmarkResultDict` containing throughput, latency, memory, and status.

    """
    return benchmark_keras_model(model_name, hardware=hardware, batch_size=batch_size, options=kwargs)


__all__ = [
    "KerasCausalLMProtocol",
    "TfModuleProtocol",
    "benchmark_model",
]
