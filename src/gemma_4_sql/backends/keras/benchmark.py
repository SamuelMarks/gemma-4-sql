"""Keras-specific benchmarking pipeline."""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import KerasScoreType
from gemma_4_sql.backends.common_benchmark import (
    BenchmarkTelemetry,
    DetailedLoopMetrics,
    execute_detailed_benchmark_loop,
    get_current_rss_mb,
    normalize_benchmark_mode,
    normalize_hardware_family,
    run_benchmark_wrapper,
)
from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend
from gemma_4_sql.backends.keras._runtime import (
    KerasSyncableOutput,
    KerasWithDistributionProtocol,
    enter_keras_distribution_scope,
    initialize_tf_tpu_cluster,
    load_keras_gemma_model,
    load_preset_with_hf_fallback,
    query_keras_tpu_hbm_mb,
    synchronize_keras_outputs,
)
from gemma_4_sql.backends.lazy_loader import (
    ModuleNamespaceProtocol,
    import_optional_symbol,
    optional_import,
)
from gemma_4_sql.exceptions import HardwareUnavailableError
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


@runtime_checkable
class _KerasRandomNamespaceProtocol(Protocol):
    """Protocol for `keras.random` exposing `randint`."""

    def randint(
        self,
        shape: tuple[int, int],
        *,
        minval: int = 0,
        maxval: int = 100,
        dtype: str = "int32",
    ) -> TensorType:
        """Sample uniform random integers of `shape`."""
        ...


@runtime_checkable
class _KerasWithRandomProtocol(Protocol):
    """Protocol for `keras` module exposing `.random`."""

    @property
    def random(self) -> _KerasRandomNamespaceProtocol:
        """Provide `keras.random` namespace."""
        ...


@runtime_checkable
class _KerasOpsOnesProtocol(Protocol):
    """Protocol for `keras.ops` exposing `ones`."""

    def ones(self, shape: tuple[int, int], *, dtype: str = "int32") -> TensorType:
        """Create a tensor of ones with `shape` and `dtype`."""
        ...


@runtime_checkable
class _KerasWithOpsOnesProtocol(Protocol):
    """Protocol for `keras` module exposing `.ops.ones`."""

    @property
    def ops(self) -> _KerasOpsOnesProtocol:
        """Provide `keras.ops` namespace."""
        ...


@runtime_checkable
class _JaxJitProtocol(Protocol):
    """Protocol for `jax` exposing `.jit`."""

    def jit[FnT](self, fun: FnT, /) -> FnT:
        """Compile `fun` with XLA JIT."""
        ...


_LAST_KERAS_METRICS: contextvars.ContextVar[tuple[DetailedLoopMetrics, str, int, float | None] | None] = (
    contextvars.ContextVar("keras_last_benchmark_metrics", default=None)
)


def _load_keras_model(model_name: str, dtype: str) -> KerasCausalLMProtocol:
    """Load and configure a compiled KerasHub or KerasNLP Gemma model for benchmarking.

    Args:
        model_name: Target KerasHub/KerasNLP preset or HuggingFace model identifier.
        dtype: Floating-point precision policy string (e.g., `'bfloat16'`, `'float32'`).

    Returns:
        Loaded `KerasCausalLMProtocol` model instance.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        ImportError: Internally raised and wrapped if neither KerasHub nor KerasNLP is available.
        InferenceError: If the model preset cannot be loaded.

    """
    if keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")
    try:
        keras.config.set_floatx(dtype)
        gemma_causal_lm_cls = load_keras_gemma_model(
            model_name,
            _GemmaCausalLMLoaderProtocol,
            symbol_importer=import_optional_symbol,
        )
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_hub / keras_nlp GemmaCausalLM is not available.")
        loaded_model: KerasCausalLMProtocol = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
        return loaded_model
    except (ImportError, ValueError) as err:
        from gemma_4_sql.exceptions import InferenceError

        msg = f"Failed to load actual model {model_name}"
        raise InferenceError(msg, backend="keras", model_name=model_name, cause=err) from err


def _get_device_str(hardware: str) -> str:
    """Map hardware target identifier to a TensorFlow device placement string.

    Args:
        hardware: Target hardware identifier (`'cpu'`, `'gpu'`, `'tpu'`, `'tpu-v6e'`, etc.).

    Returns:
        TensorFlow device specification string (`'/CPU:0'`, `'/GPU:0'`, or `'/TPU:0'`).

    """
    if tf is None:
        return "/CPU:0"
    hw_family = normalize_hardware_family(hardware)
    if hw_family == "cpu":
        return "/CPU:0"
    if hw_family == "gpu" and tf.config.list_physical_devices("GPU"):
        return "/GPU:0"
    if hw_family == "tpu":
        resolver, _ = initialize_tf_tpu_cluster(tf if isinstance(tf, ModuleNamespaceProtocol) else None)
        if resolver is not None or tf.config.list_physical_devices("TPU"):
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


def _run_non_tf_keras_pass(
    model: KerasBenchmarkModel,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    canon_mode: str,
    max_new_tokens: int,
    hardware: str,
    *,
    active_backend: str,
    strict_hardware: bool = False,
) -> tuple[float, float, float]:
    """Execute a native Keras 3 JAX or PyTorch/XLA benchmark pass without TensorFlow.

    Args:
        model: Instantiated Keras model.
        batch_size: Sequences per batch.
        num_runs: Number of timed iterations.
        warmup_steps: Number of warmup iterations.
        canon_mode: Normalized benchmark mode (`'prefill'`, `'decode'`, `'end-to-end'`).
        max_new_tokens: Tokens generated per sequence in non-prefill modes.
        hardware: Requested hardware label.
        active_backend: `'jax'` or `'torch'`.
        strict_hardware: Whether to raise `HardwareUnavailableError` on CPU fallback.

    Returns:
        3-tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    Raises:
        DependencyMissingError: If `keras` exposes neither `.random` nor `.ops`.

    """
    dist_keras = keras if isinstance(keras, KerasWithDistributionProtocol) else None
    with enter_keras_distribution_scope(
        active_backend,
        hardware,
        keras_mod=dist_keras,
        strict_hardware=strict_hardware,
        backend_label=f"keras_{active_backend}",
    ) as (resolved_dev, dev_count):
        if isinstance(keras, _KerasWithRandomProtocol):
            dummy_inputs: TensorType = keras.random.randint(
                (batch_size, 32),
                minval=1,
                maxval=256000,
                dtype="int32",
            )
        elif isinstance(keras, _KerasWithOpsOnesProtocol):
            dummy_inputs = keras.ops.ones((batch_size, 32), dtype="int32")
        else:
            from gemma_4_sql.exceptions import DependencyMissingError

            raise DependencyMissingError("keras.random or keras.ops is required for Keras benchmarking.")

        def _raw_forward(inp: TensorType) -> ForwardOutput | str:
            """Invoke `model(inp)` directly.

            Args:
                inp: Batch input token IDs.

            Returns:
                Forward pass output logits or string.

            """
            assert isinstance(model, (_KerasCallableModelProtocol, KerasCausalLMProtocol))
            return model(inp)

        forward_fn: Callable[[TensorType], ForwardOutput | str] = _raw_forward
        if active_backend == "jax":
            jax_mod = optional_import("jax", _JaxJitProtocol)
            if jax_mod is not None:
                jitted = jax_mod.jit(_raw_forward)

                def _safe_jitted(inp: TensorType) -> ForwardOutput | str:
                    """Execute `jax.jit` forward pass with fallback for non-JAX stub outputs.

                    Args:
                        inp: Batch input token IDs.

                    Returns:
                        Compiled or eager forward output.

                    """
                    try:
                        return jitted(inp)
                    except TypeError:
                        return _raw_forward(inp)

                forward_fn = _safe_jitted

        last_out: list[KerasSyncableOutput] = []

        def _step(gen_tokens: int) -> None:
            """Execute one Keras 3 prefill or generate step on JAX or PyTorch/XLA."""
            if canon_mode == "prefill":
                out_val = forward_fn(dummy_inputs)
                last_out.clear()
                last_out.append(out_val)
            elif isinstance(model, _KerasGenerateModelProtocol) and callable(model.generate):
                out_gen = model.generate(dummy_inputs, max_length=32 + gen_tokens)
                last_out.clear()
                last_out.append(out_gen)

        def _sync() -> None:
            """Synchronize pending JAX or PyTorch/XLA device work."""
            cur = last_out[-1] if last_out else None
            synchronize_keras_outputs(active_backend, cur, device=resolved_dev)

        is_xla_tpu = resolved_dev.startswith(("tpu", "xla"))
        warmup_tokens = max_new_tokens if is_xla_tpu else 2
        metrics = execute_detailed_benchmark_loop(
            lambda: _step(max_new_tokens),
            warmup_steps=warmup_steps,
            num_runs=num_runs,
            tokens_per_iter=32 if canon_mode == "prefill" else max_new_tokens,
            batch_size=batch_size,
            warmup_step_fn=lambda: _step(warmup_tokens),
            sync_fn=_sync,
        )
        peak_hbm = query_keras_tpu_hbm_mb(active_backend, device=resolved_dev)
        memory_mb = peak_hbm if peak_hbm is not None else get_current_rss_mb()
        _LAST_KERAS_METRICS.set((metrics, resolved_dev, dev_count, peak_hbm))
        return (float(metrics.tokens_per_sec), float(metrics.latency_ms), float(memory_mb))


def _run_benchmark_pass(
    model: KerasBenchmarkModel,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
    hardware: str,
    *,
    active_backend: str = "tensorflow",
    strict_hardware: bool = False,
) -> tuple[float, float, float]:
    """Execute the warmup and timed benchmark pass for a Keras model across JAX, TF, or PyTorch.

    Args:
        model: Instantiated Keras model.
        batch_size: Number of sequences processed per batch.
        num_runs: Number of timed benchmark iterations.
        warmup_steps: Number of untimed warmup iterations before timing begins.
        mode: Execution mode (`'prefill'`, `'decode'`, `'generate'`, or `'end-to-end'`).
        max_new_tokens: Number of tokens generated in non-prefill modes.
        hardware: Target hardware identifier (`'cpu'`, `'gpu'`, `'tpu'`, `'tpu-v6e'`, etc.).
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        strict_hardware: Whether to raise `HardwareUnavailableError` on CPU fallback.

    Returns:
        A 3-tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    Raises:
        DependencyMissingError: If `active_backend == 'tensorflow'` and TensorFlow is missing.
        HardwareUnavailableError: If `strict_hardware=True` and requested TPU/GPU falls back to CPU.

    """
    canon_mode = normalize_benchmark_mode(mode, backend="keras")
    if active_backend in {"jax", "torch"}:
        return _run_non_tf_keras_pass(
            model,
            batch_size,
            num_runs,
            warmup_steps,
            canon_mode,
            max_new_tokens,
            hardware,
            active_backend=active_backend,
            strict_hardware=strict_hardware,
        )

    if tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorFlow dependencies are missing.")

    tf.random.set_seed(42)
    device_str = _get_device_str(hardware)
    hw_family = normalize_hardware_family(hardware)
    if strict_hardware and hw_family in {"tpu", "gpu"} and device_str == "/CPU:0":
        msg = f"Keras TensorFlow backend fell back to '/CPU:0' when '{hardware}' was requested."
        raise HardwareUnavailableError(msg, hardware=hardware, backend="keras_tensorflow")

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

        if "GPU" in device_str or "TPU" in device_str:
            try:
                tf.config.experimental.reset_memory_stats(device_str.replace("/", ""))
            except ValueError:
                pass

        def _run_keras_step(gen_tokens: int) -> None:
            """Execute one Keras prefill or generate step and materialize output."""
            if canon_mode == "prefill":
                out = forward_pass(dummy_inputs)
                _materialize_tf_output(out)
            elif isinstance(model, _KerasGenerateModelProtocol) and callable(model.generate):
                out = model.generate(dummy_inputs, max_length=32 + gen_tokens)
                _materialize_tf_output(out)

        warmup_gen_tokens = max_new_tokens if "TPU" in device_str else 2
        metrics = execute_detailed_benchmark_loop(
            lambda: _run_keras_step(max_new_tokens),
            warmup_steps=warmup_steps,
            num_runs=num_runs,
            tokens_per_iter=32 if canon_mode == "prefill" else max_new_tokens,
            batch_size=batch_size,
            warmup_step_fn=lambda: _run_keras_step(warmup_gen_tokens),
        )

        peak_hbm: float | None = None
        try:
            if "GPU" in device_str or "TPU" in device_str:
                memory_info = tf.config.experimental.get_memory_info(device_str.replace("/", ""))
                memory_mb = memory_info.get("peak", memory_info.get("current", 0.0)) / (1024 * 1024)
                peak_hbm = float(memory_mb)
            else:
                memory_mb = get_current_rss_mb()
        except ValueError:
            memory_mb = get_current_rss_mb()

        dev_count = max(1, len(tf.config.list_physical_devices("TPU"))) if "TPU" in device_str else 1
        _LAST_KERAS_METRICS.set((metrics, device_str, dev_count, peak_hbm))
        return (float(metrics.tokens_per_sec), float(metrics.latency_ms), float(memory_mb))


def benchmark_keras_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    *,
    model_loader: Callable[[str, str], KerasBenchmarkModel] | None = None,
    options: Mapping[str, JSONValue] | None = None,
) -> BenchmarkResultDict:
    """Benchmark a model using Keras 3 (`keras_jax`, `keras_tensorflow`, or `keras_pytorch`).

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'tpu-v6e'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        model_loader: Optional injected `(model_name, dtype) -> model` loader callable.
        options: Optional benchmark configuration mapping (`num_runs`, `dtype`, `mode`,
            `max_new_tokens`, `warmup_steps`, `backend_alias`, `keras_backend`, `strict_hardware`).

    Returns:
        Structured `BenchmarkResultDict` containing throughput, latency, memory, and status.

    Raises:
        DependencyMissingError: If Keras (or TensorFlow on the TF backend) is missing.

    """
    opts: Mapping[str, JSONValue] = options if options is not None else {}
    backend_alias = str(opts.get("backend_alias", "keras"))
    strict_hw = bool(opts.get("strict_hardware", False))
    explicit_kb = str(opts["keras_backend"]) if "keras_backend" in opts and opts["keras_backend"] is not None else None

    if backend_alias == "keras" and explicit_kb is None and tf is not None:
        active_kb = "tensorflow"
    else:
        active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)

    if keras is None or (active_kb == "tensorflow" and tf is None):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.", backend=backend_alias)

    def _run() -> BenchmarkTelemetry:
        """Load Keras model, execute multi-backend benchmark pass, and package telemetry.

        Returns:
            `BenchmarkTelemetry` across timed iterations.

        """
        dtype = str(opts.get("dtype", "bfloat16"))
        mode = str(opts.get("mode", "prefill"))
        max_new_tokens = int(str(opts.get("max_new_tokens", 128)))
        warmup_steps = int(str(opts.get("warmup_steps", 5)))
        num_runs = int(str(opts.get("num_runs", 5)))

        load_fn = model_loader if model_loader is not None else _load_keras_model
        model = load_fn(model_name, dtype)
        ctx_tok = _LAST_KERAS_METRICS.set(None)
        try:
            tps, lat_ms, mem_mb = _run_benchmark_pass(
                model,
                batch_size,
                num_runs,
                warmup_steps,
                mode,
                max_new_tokens,
                hardware,
                active_backend=active_kb,
                strict_hardware=strict_hw,
            )
            recorded = _LAST_KERAS_METRICS.get()
        finally:
            _LAST_KERAS_METRICS.reset(ctx_tok)

        if recorded is not None:
            loop_m, res_dev, dev_cnt, pk_hbm = recorded
            return BenchmarkTelemetry(
                tokens_per_sec=tps,
                latency_ms=lat_ms,
                memory_mb=mem_mb,
                resolved_device=res_dev,
                device_count=dev_cnt,
                compile_latency_ms=loop_m.compile_latency_ms,
                peak_hbm_mb=pk_hbm,
                keras_backend=active_kb,
                latency_samples=loop_m.latency_samples,
            )
        return BenchmarkTelemetry(
            tokens_per_sec=tps,
            latency_ms=lat_ms,
            memory_mb=mem_mb,
            keras_backend=active_kb,
        )

    return run_benchmark_wrapper(
        backend_name=backend_alias,
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
        strict_hardware=strict_hw,
        raise_on_error=bool(opts["raise_on_error"]) if "raise_on_error" in opts else None,
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
