"""Multi-backend Keras 3 runtime helpers for JAX, TensorFlow, and PyTorch TPU execution."""

from __future__ import annotations

import contextlib
import logging
import types
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql._types_models import KerasScoreType
from gemma_4_sql.backends.common_benchmark import normalize_hardware_family
from gemma_4_sql.backends.common_dpo import ArithmeticTensorProtocol
from gemma_4_sql.backends.lazy_loader import (
    ModuleNamespaceProtocol,
    import_optional_symbol,
    optional_import,
)
from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRandintProtocol, TorchRuntimeAdapter
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import ForwardOutput, TfTensorProtocol

logger = logging.getLogger(__name__)

_tf_tpu_init_state: list[bool] = [False]


@runtime_checkable
class KerasPresetLoaderProtocol[ModelT](Protocol):
    """Protocol for KerasHub / KerasNLP model classes exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> ModelT:
        """Load a Keras causal language model from `model_name` or preset URI.

        Args:
            model_name: Preset name, local path, or `hf://` URI.

        Returns:
            Instantiated Keras model of type `ModelT`.

        """
        ...


@runtime_checkable
class KerasScopeContextProtocol(Protocol):
    """Protocol for distribution or device scope context managers."""

    def __enter__(self) -> Self | None:
        """Enter distribution scope."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool | None:
        """Exit distribution scope."""
        ...


@runtime_checkable
class KerasDistributionStrategyProtocol(Protocol):
    """Protocol for `tf.distribute.Strategy` or `keras.distribution.Distribution` exposing `.scope()`."""

    def scope(self) -> KerasScopeContextProtocol:
        """Return context manager activating this distribution strategy.

        Returns:
            Active strategy scope context manager.

        """
        ...


@runtime_checkable
class KerasDistributionNamespaceProtocol(Protocol):
    """Protocol for `keras.distribution` multi-device data/model parallelism."""

    def DataParallel(
        self,
        *,
        devices: Sequence[str] | None = None,
    ) -> KerasDistributionStrategyProtocol:
        """Construct a `DataParallel` distribution across `devices`.

        Args:
            devices: Optional explicit accelerator device names.

        Returns:
            Configured `KerasDistributionStrategyProtocol`.

        """
        ...


@runtime_checkable
class KerasWithDistributionProtocol(Protocol):
    """Protocol for `keras` module exposing `.distribution`."""

    @property
    def distribution(self) -> KerasDistributionNamespaceProtocol:
        """Provide `keras.distribution` namespace."""
        ...


@runtime_checkable
class TPUClusterResolverProtocol(Protocol):
    """Protocol for `tf.distribute.cluster_resolver.TPUClusterResolver`."""

    def master(self) -> str:
        """Return TPU cluster master target URI.

        Returns:
            Master gRPC/local endpoint string.

        """
        ...


@runtime_checkable
class _TFClusterResolverNamespaceProtocol(Protocol):
    """Protocol for `tf.distribute.cluster_resolver` exposing `TPUClusterResolver`."""

    def TPUClusterResolver(self, tpu: str = "local", /) -> TPUClusterResolverProtocol:
        """Construct a `TPUClusterResolver` for `tpu`.

        Args:
            tpu: TPU name or `'local'` on Cloud TPU VMs.

        Returns:
            Initialized `TPUClusterResolverProtocol`.

        """
        ...


@runtime_checkable
class _TFDistributeTPUProtocol(Protocol):
    """Protocol for `tf.distribute` exposing `TPUStrategy` and `cluster_resolver`."""

    @property
    def cluster_resolver(self) -> _TFClusterResolverNamespaceProtocol:
        """Provide `tf.distribute.cluster_resolver`."""
        ...

    def TPUStrategy(self, resolver: TPUClusterResolverProtocol, /) -> KerasDistributionStrategyProtocol:
        """Construct a `TPUStrategy` bound to `resolver`.

        Args:
            resolver: Connected `TPUClusterResolverProtocol`.

        Returns:
            Configured `TPUStrategy`.

        """
        ...


@runtime_checkable
class _TFMirroredDistributeProtocol(Protocol):
    """Protocol for `tf.distribute` exposing `MirroredStrategy`."""

    def MirroredStrategy(self) -> KerasDistributionStrategyProtocol:
        """Construct a single-host `MirroredStrategy`.

        Returns:
            Configured `MirroredStrategy`.

        """
        ...


@runtime_checkable
class _TFTpuExperimentalProtocol(Protocol):
    """Protocol for `tf.tpu.experimental` exposing `initialize_tpu_system`."""

    def initialize_tpu_system(self, resolver: TPUClusterResolverProtocol, /) -> Sequence[str]:
        """Initialize the TPU mesh topology for `resolver`.

        Args:
            resolver: Connected `TPUClusterResolverProtocol`.

        Returns:
            Sequence of initialized TPU logical device descriptors.

        """
        ...


@runtime_checkable
class _TFTpuNamespaceProtocol(Protocol):
    """Protocol for `tf.tpu` exposing `.experimental`."""

    @property
    def experimental(self) -> _TFTpuExperimentalProtocol:
        """Provide `tf.tpu.experimental`."""
        ...


@runtime_checkable
class _TFConfigExperimentalConnectProtocol(Protocol):
    """Protocol for `tf.config.experimental_connect_to_cluster` and device queries."""

    def experimental_connect_to_cluster(self, resolver: TPUClusterResolverProtocol, /) -> None:
        """Connect TensorFlow runtime to the TPU cluster described by `resolver`.

        Args:
            resolver: `TPUClusterResolverProtocol` instance.

        """
        ...

    def list_logical_devices(self, device_type: str = "TPU", /) -> Sequence[str]:
        """Return initialized logical devices of `device_type`.

        Args:
            device_type: Device type filter (`'TPU'`, `'GPU'`, `'CPU'`).

        Returns:
            Sequence of logical device descriptors.

        """
        ...


@runtime_checkable
class _TFConfigPhysicalProtocol(Protocol):
    """Protocol for `tf.config` exposing `list_physical_devices`."""

    def list_physical_devices(self, device_type: str, /) -> Sequence[str]:
        """Return physical devices matching `device_type`.

        Args:
            device_type: Device type filter (`'TPU'`, `'GPU'`, `'CPU'`).

        Returns:
            Sequence of physical device descriptors.

        """
        ...


@runtime_checkable
class _TFConfigMemExperimentalProtocol(Protocol):
    """Protocol for `tf.config.experimental.get_memory_info`."""

    def get_memory_info(self, dev: str, /) -> Mapping[str, int | float]:
        """Return memory telemetry mapping (`'peak'`, `'current'`) for `dev`.

        Args:
            dev: TensorFlow device string (e.g., `'TPU:0'`, `'GPU:0'`).

        Returns:
            Memory usage mapping in bytes.

        """
        ...


@runtime_checkable
class _TFConfigWithMemExperimentalProtocol(Protocol):
    """Protocol for `tf.config` exposing `.experimental.get_memory_info`."""

    @property
    def experimental(self) -> _TFConfigMemExperimentalProtocol:
        """Provide `tf.config.experimental`."""
        ...


@runtime_checkable
class _TFConfigHolderProtocol(Protocol):
    """Single top-level protocol for modules exposing `.config`."""

    @property
    def config(
        self,
    ) -> (
        _TFConfigExperimentalConnectProtocol
        | _TFConfigPhysicalProtocol
        | _TFConfigWithMemExperimentalProtocol
        | ModuleNamespaceProtocol
    ):
        """Provide `tf.config` namespace."""
        ...


@runtime_checkable
class _TFDistributeHolderProtocol(Protocol):
    """Single top-level protocol for modules exposing `.distribute`."""

    @property
    def distribute(
        self,
    ) -> _TFDistributeTPUProtocol | _TFMirroredDistributeProtocol | ModuleNamespaceProtocol:
        """Provide `tf.distribute` namespace."""
        ...


@runtime_checkable
class _TFTpuHolderProtocol(Protocol):
    """Single top-level protocol for modules exposing `.tpu`."""

    @property
    def tpu(self) -> _TFTpuNamespaceProtocol:
        """Provide `tf.tpu` namespace."""
        ...


@runtime_checkable
class _JaxMemDeviceProtocol(Protocol):
    """Protocol for a JAX device optionally exposing `memory_stats()`."""

    def memory_stats(self) -> Mapping[str, int | float] | None:
        """Return HBM memory statistics mapping or `None`."""
        ...


@runtime_checkable
class _JaxRuntimeQueryProtocol(Protocol):
    """Protocol for `jax` device discovery and `block_until_ready` synchronization."""

    def devices(self, backend: str | None = None, /) -> Sequence[_JaxMemDeviceProtocol | str]:
        """Return visible JAX devices for `backend`.

        Args:
            backend: Optional backend filter (`'tpu'`, `'gpu'`, `'cpu'`).

        Returns:
            Sequence of JAX device descriptors.

        """
        ...

    def block_until_ready[OutT](self, x: OutT, /) -> OutT:
        """Block until asynchronous XLA computation producing `x` completes.

        Args:
            x: PyTree or array output.

        Returns:
            Synchronized `x`.

        """
        ...


type KerasSyncableOutput = (
    ForwardOutput
    | TfTensorProtocol
    | ArithmeticTensorProtocol
    | Sequence[int]
    | Sequence[str]
    | Mapping[str, KerasScoreType]
    | tuple[str, KerasScoreType]
    | str
    | float
    | int
    | None
)


def reset_tf_tpu_init_state() -> None:
    """Reset the module-level TensorFlow TPU cluster initialization flag (for testing)."""
    _tf_tpu_init_state[0] = False


def load_keras_gemma_model[LoaderT](
    model_name: str,
    loader_proto: type[LoaderT],
    *,
    explicit_cls: LoaderT | None = None,
    symbol_importer: Callable[[str, str, type[LoaderT]], LoaderT | None] = import_optional_symbol,
) -> LoaderT | None:
    """Resolve a KerasHub (`Gemma4CausalLM` / `GemmaCausalLM` / `CausalLM`) or KerasNLP loader class.

    Args:
        model_name: Target model name (logged for diagnostics).
        loader_proto: `@runtime_checkable` loader protocol class with `.from_preset`.
        explicit_cls: Optional pre-imported or monkeypatched loader class.
        symbol_importer: Symbol lookup callable (defaults to `import_optional_symbol`).

    Returns:
        Resolved preset loader class satisfying `loader_proto`, or `None` if unavailable.

    """
    _ = model_name
    if explicit_cls is not None:
        return explicit_cls
    candidates: tuple[tuple[str, str], ...] = (
        ("keras_hub.models", "Gemma4CausalLM"),
        ("keras_hub.models", "GemmaCausalLM"),
        ("keras_hub.models", "CausalLM"),
        ("keras_nlp.models", "GemmaCausalLM"),
    )
    for mod_path, sym_name in candidates:
        found = symbol_importer(mod_path, sym_name, loader_proto)
        if found is not None:
            return found
    return None


def load_preset_with_hf_fallback[ModelT](
    loader_cls: KerasPresetLoaderProtocol[ModelT],
    model_name: str,
) -> ModelT:
    """Invoke `loader_cls.from_preset(model_name)` with automatic `hf://` retry for HuggingFace IDs.

    Args:
        loader_cls: KerasHub or KerasNLP causal LM class exposing `from_preset`.
        model_name: Preset name or HuggingFace repo ID (e.g., `'google/gemma-4-E4B'`).

    Returns:
        Loaded Keras model instance.

    Raises:
        ValueError: If `model_name` cannot be resolved directly or with an `hf://` prefix.

    """
    try:
        return loader_cls.from_preset(model_name)
    except ValueError:
        if "/" in model_name and not model_name.startswith("hf://"):
            return loader_cls.from_preset(f"hf://{model_name}")
        raise


def initialize_tf_tpu_cluster(
    tf_mod: (
        _TFConfigHolderProtocol | _TFDistributeHolderProtocol | _TFTpuHolderProtocol | ModuleNamespaceProtocol | None
    ),
) -> tuple[TPUClusterResolverProtocol | None, int]:
    """Idempotently connect and initialize a Cloud TPU VM cluster on TensorFlow.

    Args:
        tf_mod: Loaded `tensorflow` module or typed test double.

    Returns:
        2-tuple of `(resolver, logical_tpu_count)` or `(None, 0)` when unavailable.

    """
    if not (
        isinstance(tf_mod, _TFDistributeHolderProtocol)
        and isinstance(tf_mod, _TFConfigHolderProtocol)
        and isinstance(tf_mod, _TFTpuHolderProtocol)
    ):
        return (None, 0)
    dist_ns = tf_mod.distribute
    cfg_ns = tf_mod.config
    if not (isinstance(dist_ns, _TFDistributeTPUProtocol) and isinstance(cfg_ns, _TFConfigExperimentalConnectProtocol)):
        return (None, 0)
    try:
        resolver = dist_ns.cluster_resolver.TPUClusterResolver("local")
        if not _tf_tpu_init_state[0]:
            cfg_ns.experimental_connect_to_cluster(resolver)
            tf_mod.tpu.experimental.initialize_tpu_system(resolver)
            _tf_tpu_init_state[0] = True
        logical = cfg_ns.list_logical_devices("TPU")
        return (resolver, max(1, len(logical)))
    except (ValueError, RuntimeError, OSError) as err:
        logger.debug("TensorFlow TPUClusterResolver('local') unavailable: %s", err)
        return (None, 0)


def _list_tf_physical_devices(
    tf_mod: _TFConfigHolderProtocol | _TFDistributeHolderProtocol | ModuleNamespaceProtocol | None,
    device_type: str,
) -> Sequence[str]:
    """Query `tf_mod.config.list_physical_devices(device_type)` safely when supported.

    Args:
        tf_mod: Optional TensorFlow module.
        device_type: Device type string (`'TPU'`, `'GPU'`).

    Returns:
        Sequence of physical device identifiers or empty tuple.

    """
    if isinstance(tf_mod, _TFConfigHolderProtocol) and isinstance(tf_mod.config, _TFConfigPhysicalProtocol):
        return tf_mod.config.list_physical_devices(device_type)
    return ()


@contextlib.contextmanager
def enter_keras_distribution_scope(
    active_backend: str,
    hardware: str,
    *,
    tf_mod: _TFConfigHolderProtocol | _TFDistributeHolderProtocol | ModuleNamespaceProtocol | None = None,
    keras_mod: KerasWithDistributionProtocol | ModuleNamespaceProtocol | None = None,
    strict_hardware: bool = False,
    backend_label: str = "keras",
) -> Generator[tuple[str, int], None, None]:
    """Enter the appropriate multi-chip TPU/GPU distribution scope for `active_backend`.

    Args:
        active_backend: Active Keras 3 backend (`'jax'`, `'tensorflow'`, or `'torch'`).
        hardware: Requested hardware string (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        tf_mod: Optional `tensorflow` module when `active_backend == 'tensorflow'`.
        keras_mod: Optional `keras` module exposing `.distribution`.
        strict_hardware: If `True`, raise `HardwareUnavailableError` on accelerator fallback.
        backend_label: Backend alias recorded on `HardwareUnavailableError`.

    Yields:
        2-tuple of `(resolved_device, device_count)`.

    Raises:
        HardwareUnavailableError: When `strict_hardware=True` and requested TPU/GPU is unavailable.

    """
    hw_family = normalize_hardware_family(hardware)

    if active_backend == "jax":
        jax_mod = optional_import("jax", _JaxRuntimeQueryProtocol)
        if hw_family in {"tpu", "gpu"} and jax_mod is not None:
            try:
                devs = jax_mod.devices(hw_family)
            except RuntimeError:
                devs = ()
            if len(devs) > 0:
                resolved = f"{hw_family}:0"
                count = len(devs)
                if isinstance(keras_mod, KerasWithDistributionProtocol) and count > 1:
                    dist = keras_mod.distribution.DataParallel(devices=[str(d) for d in devs])
                    with dist.scope():
                        yield (resolved, count)
                    return
                yield (resolved, count)
                return
        if strict_hardware and hw_family in {"tpu", "gpu"}:
            msg = f"Keras JAX backend could not resolve '{hardware}' devices."
            raise HardwareUnavailableError(msg, hardware=hardware, backend=backend_label)
        yield ("cpu", 1)
        return

    if active_backend == "torch":
        adapter = TorchRuntimeAdapter(optional_import("torch", TorchRandintProtocol))
        dev = adapter.get_device(hardware, strict_hardware=strict_hardware)
        count = adapter.get_device_count(hardware) if dev != "cpu" else 1
        yield (dev, count)
        return

    # TensorFlow backend path
    if hw_family == "tpu":
        resolver, count = initialize_tf_tpu_cluster(tf_mod)
        if (
            resolver is not None
            and isinstance(tf_mod, _TFDistributeHolderProtocol)
            and isinstance(tf_mod.distribute, _TFDistributeTPUProtocol)
        ):
            strategy = tf_mod.distribute.TPUStrategy(resolver)
            with strategy.scope():
                yield ("/TPU:0", count)
            return
        phys_tpus = _list_tf_physical_devices(tf_mod, "TPU")
        if len(phys_tpus) > 0:
            yield ("/TPU:0", len(phys_tpus))
            return
        if strict_hardware:
            msg = f"Keras TensorFlow backend could not initialize TPU for '{hardware}'."
            raise HardwareUnavailableError(msg, hardware=hardware, backend=backend_label)

    if hw_family == "gpu":
        gpus = _list_tf_physical_devices(tf_mod, "GPU")
        if len(gpus) > 0:
            if isinstance(tf_mod, _TFDistributeHolderProtocol) and isinstance(
                tf_mod.distribute, _TFMirroredDistributeProtocol
            ):
                with tf_mod.distribute.MirroredStrategy().scope():
                    yield ("/GPU:0", len(gpus))
                return
            yield ("/GPU:0", len(gpus))
            return
        if strict_hardware:
            msg = f"Keras TensorFlow backend found zero GPUs for '{hardware}'."
            raise HardwareUnavailableError(msg, hardware=hardware, backend=backend_label)

    if isinstance(tf_mod, _TFDistributeHolderProtocol) and isinstance(tf_mod.distribute, _TFMirroredDistributeProtocol):
        with tf_mod.distribute.MirroredStrategy().scope():
            yield ("/CPU:0", 1)
        return
    yield ("/CPU:0", 1)


def synchronize_keras_outputs(
    active_backend: str,
    output: KerasSyncableOutput,
    *,
    device: str = "cpu",
) -> None:
    """Synchronize asynchronous accelerator execution for `active_backend`.

    Args:
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        output: Forward or generation output value.
        device: Resolved runtime device string.

    """
    if active_backend == "jax":
        jax_mod = optional_import("jax", _JaxRuntimeQueryProtocol)
        if jax_mod is not None:
            try:
                _ = jax_mod.block_until_ready(output)
            except (TypeError, ValueError, RuntimeError):
                pass
        return
    if active_backend == "torch":
        TorchRuntimeAdapter(optional_import("torch", TorchRandintProtocol)).synchronize(device)


def _extract_jax_device_peak_mb(dev: _JaxMemDeviceProtocol | str) -> float | None:
    """Return peak HBM usage in MiB for a single JAX device, or `None` if unavailable.

    Args:
        dev: JAX device descriptor or string.

    Returns:
        Peak memory in MiB or `None`.

    """
    if not isinstance(dev, _JaxMemDeviceProtocol):
        return None
    stats = dev.memory_stats()
    if stats is None:
        return None
    raw = stats.get("peak_bytes_in_use", stats.get("bytes_in_use"))
    if raw is not None and raw > 0:
        return float(raw) / (1024.0 * 1024.0)
    return None


def query_keras_tpu_hbm_mb(
    active_backend: str,
    *,
    device: str = "cpu",
    tf_mod: _TFConfigHolderProtocol | ModuleNamespaceProtocol | None = None,
) -> float | None:
    """Query peak accelerator HBM/VRAM usage in MiB for `active_backend`, or `None` on CPU.

    Args:
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        device: Resolved device string (`'tpu:0'`, `'xla:0'`, `'/TPU:0'`, `'/GPU:0'`, `'cpu'`).
        tf_mod: Optional `tensorflow` module for TensorFlow memory queries.

    Returns:
        Peak HBM/VRAM in MiB when measurable on an accelerator, otherwise `None`.

    """
    if active_backend == "jax" and device.startswith(("tpu", "gpu")):
        jax_mod = optional_import("jax", _JaxRuntimeQueryProtocol)
        if jax_mod is not None:
            dev_kind = "tpu" if device.startswith("tpu") else "gpu"
            try:
                devs = jax_mod.devices(dev_kind)
            except RuntimeError:
                devs = ()
            peaks = [mb for d in devs if (mb := _extract_jax_device_peak_mb(d)) is not None]
            if peaks:
                return max(peaks)
        return None

    if active_backend == "torch" and device.startswith(("xla", "cuda", "mps")):
        adapter = TorchRuntimeAdapter(optional_import("torch", TorchRandintProtocol))
        val = adapter.max_memory_mb(device, -1.0)
        return val if val > 0.0 else None

    is_tf_accel = active_backend == "tensorflow" and ("TPU" in device or "GPU" in device)
    if (
        is_tf_accel
        and isinstance(tf_mod, _TFConfigHolderProtocol)
        and isinstance(tf_mod.config, _TFConfigWithMemExperimentalProtocol)
    ):
        try:
            info = tf_mod.config.experimental.get_memory_info(device.replace("/", ""))
            raw_tf = info.get("peak", info.get("current", 0.0))
            if raw_tf > 0:
                return float(raw_tf) / (1024.0 * 1024.0)
        except ValueError:
            return None
    return None


__all__ = [
    "KerasDistributionNamespaceProtocol",
    "KerasDistributionStrategyProtocol",
    "KerasPresetLoaderProtocol",
    "KerasScopeContextProtocol",
    "KerasSyncableOutput",
    "KerasWithDistributionProtocol",
    "TPUClusterResolverProtocol",
    "enter_keras_distribution_scope",
    "initialize_tf_tpu_cluster",
    "load_keras_gemma_model",
    "load_preset_with_hf_fallback",
    "query_keras_tpu_hbm_mb",
    "reset_tf_tpu_init_state",
    "synchronize_keras_outputs",
]
