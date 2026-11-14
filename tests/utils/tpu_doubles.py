"""Strictly typed protocol test doubles for TPU runtimes (`torch_xla`, TF `TPUStrategy`, Keras 3, and JAX).

All doubles satisfy `@runtime_checkable` protocols without `MagicMock`, `Any`, `cast`, or `object`.
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from gemma_4_sql.backends.pytorch.runtime_adapter import XLADeviceDescriptorProtocol
from gemma_4_sql.type_hints import JaxShardingProtocol, TorchOptimizerProtocol


def _empty_dist_calls() -> list[dict[str, str | int | None]]:
    """Construct an empty typed call list for `FakeJaxDistributedNamespace`.

    Returns:
        Empty list of keyword-argument dictionaries.

    """
    return []


@dataclass(frozen=True)
class FakeXLADeviceDescriptor:
    """Typed `torch.device('xla:0')` descriptor satisfying `XLADeviceDescriptorProtocol`.

    Attributes:
        name: Full device string (`'xla:0'`).
        type: Device type string (`'xla'`).

    """

    name: str = "xla:0"
    type: str = "xla"

    def __str__(self) -> str:
        """Return string representation (`'xla:0'`).

        Returns:
            Device string.

        """
        return self.name


class FakeTorchXLAModule:
    """Typed test double satisfying `TorchXLAModuleProtocol`."""

    def __init__(self, device_names: Sequence[str] = ("xla:0", "xla:1", "xla:2", "xla:3")) -> None:
        """Initialize simulated PJRT XLA devices and sync counter.

        Args:
            device_names: Visible XLA device names on the host.

        """
        self._devices = tuple(FakeXLADeviceDescriptor(name=n) for n in device_names)
        self.sync_calls: int = 0

    def device(self) -> FakeXLADeviceDescriptor | str:
        """Return primary XLA device descriptor or `'xla:0'` fallback.

        Returns:
            Primary `FakeXLADeviceDescriptor` or `'xla:0'`.

        """
        return self._devices[0] if self._devices else "xla:0"

    def sync(self) -> None:
        """Increment `sync_calls` counter."""
        self.sync_calls += 1

    def devices(self) -> Sequence[FakeXLADeviceDescriptor]:
        """Return all configured XLA device descriptors.

        Returns:
            Tuple of `FakeXLADeviceDescriptor` items.

        """
        return self._devices


class FakeTorchXLACoreModel:
    """Typed test double satisfying `TorchXLACoreModelProtocol` (`torch_xla.core.xla_model`)."""

    def __init__(
        self,
        tpu_devices: Sequence[str] = ("xla:0", "xla:1", "xla:2", "xla:3"),
        peak_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Initialize `xm` double with simulated TPU device list and HBM telemetry.

        Args:
            tpu_devices: Supported TPU device strings.
            peak_bytes: Reported `peak_bytes_used` value.

        """
        self._tpu_devices = tuple(tpu_devices)
        self.peak_bytes = peak_bytes
        self.mark_step_calls = 0
        self.wait_ops_calls = 0
        self.optimizer_steps = 0

    def xla_device(self) -> str:
        """Return primary XLA device string.

        Returns:
            First configured TPU device string or `'xla:0'`.

        """
        return self._tpu_devices[0] if self._tpu_devices else "xla:0"

    def get_xla_supported_devices(self, devkind: str | None = None) -> Sequence[str]:
        """Return supported XLA devices filtered by `devkind`.

        Args:
            devkind: Optional `'TPU'` filter.

        Returns:
            Tuple of XLA device strings.

        """
        if devkind in {None, "TPU"}:
            return self._tpu_devices
        return ()

    def mark_step(self) -> None:
        """Increment `mark_step_calls`."""
        self.mark_step_calls += 1

    def wait_device_ops(self) -> None:
        """Increment `wait_ops_calls`."""
        self.wait_ops_calls += 1

    def get_memory_info(self, device: XLADeviceDescriptorProtocol | str, /) -> Mapping[str, int | float]:
        """Return simulated HBM usage dictionary for `device`.

        Args:
            device: XLA device descriptor or string.

        Returns:
            Mapping containing `peak_bytes_used` and `bytes_used`.

        """
        _ = device
        return {"peak_bytes_used": self.peak_bytes, "bytes_used": self.peak_bytes // 2}

    def optimizer_step(
        self,
        optimizer: TorchOptimizerProtocol,
        *,
        barrier: bool = False,
    ) -> None:
        """Record XLA optimizer step and invoke `optimizer.step()`.

        Args:
            optimizer: PyTorch optimizer instance.
            barrier: Whether XLA barrier was requested.

        """
        _ = barrier
        self.optimizer_steps += 1
        optimizer.step()


@dataclass
class FakeTorchXLARuntime:
    """Typed test double satisfying `TorchXLARuntimeProtocol` (`torch_xla.runtime`).

    Attributes:
        ordinal: Global ordinal (`0..world-1`).
        world: Total PJRT world size.
        local_ord: Host-local ordinal.

    """

    ordinal: int = 0
    world: int = 4
    local_ord: int = 0

    def global_ordinal(self) -> int:
        """Return `self.ordinal`.

        Returns:
            Global ordinal integer.

        """
        return self.ordinal

    def world_size(self) -> int:
        """Return `self.world`.

        Returns:
            Total PJRT world size.

        """
        return self.world

    def local_ordinal(self) -> int:
        """Return `self.local_ord`.

        Returns:
            Host-local ordinal integer.

        """
        return self.local_ord


class FakeScopeContext:
    """Reusable distribution scope context manager satisfying `KerasScopeContextProtocol`."""

    def __init__(self) -> None:
        """Initialize enter/exit counters."""
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> Self:
        """Increment `entered` and return `self`.

        Returns:
            `Self` inside scope.

        """
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool:
        """Increment `exited` without suppressing exceptions.

        Returns:
            `False`.

        """
        _ = (exc_type, exc_val, exc_tb)
        self.exited += 1
        return False


class FakeStrategy:
    """Typed distribution strategy (`TPUStrategy` / `MirroredStrategy` / `DataParallel`) double."""

    def __init__(self) -> None:
        """Initialize strategy with a tracked `FakeScopeContext`."""
        self.scope_ctx = FakeScopeContext()

    def scope(self) -> FakeScopeContext:
        """Return tracked `FakeScopeContext`.

        Returns:
            `self.scope_ctx`.

        """
        return self.scope_ctx


@dataclass(frozen=True)
class FakeTPUClusterResolver:
    """Typed `TPUClusterResolver` double satisfying `TPUClusterResolverProtocol`.

    Attributes:
        tpu_name: Target TPU identifier (`'local'`).

    """

    tpu_name: str = "local"

    def master(self) -> str:
        """Return simulated TPU master endpoint.

        Returns:
            Master string.

        """
        return f"grpc://{self.tpu_name}:8470"


class FakeTFClusterResolverNamespace:
    """Typed `tf.distribute.cluster_resolver` double exposing `TPUClusterResolver`."""

    def __init__(self, *, should_raise: bool = False) -> None:
        """Configure whether `TPUClusterResolver` succeeds or raises `ValueError`.

        Args:
            should_raise: If `True`, `TPUClusterResolver` raises `ValueError`.

        """
        self.should_raise = should_raise

    def TPUClusterResolver(self, tpu: str = "local", /) -> FakeTPUClusterResolver:
        """Construct a `FakeTPUClusterResolver` or raise `ValueError`.

        Args:
            tpu: Target TPU name.

        Returns:
            `FakeTPUClusterResolver` instance.

        Raises:
            ValueError: When `self.should_raise` is `True`.

        """
        if self.should_raise:
            raise ValueError("No TPU found")
        return FakeTPUClusterResolver(tpu_name=tpu)


class FakeTFDistributeNamespace:
    """Typed `tf.distribute` double exposing `TPUStrategy`, `MirroredStrategy`, and `cluster_resolver`."""

    def __init__(self, *, resolver_raises: bool = False) -> None:
        """Initialize strategy tracking and `cluster_resolver`.

        Args:
            resolver_raises: Whether `TPUClusterResolver` raises `ValueError`.

        """
        self.cluster_resolver = FakeTFClusterResolverNamespace(should_raise=resolver_raises)
        self.last_tpu_strategy: FakeStrategy | None = None
        self.last_mirrored_strategy: FakeStrategy | None = None

    def TPUStrategy(self, resolver: FakeTPUClusterResolver, /) -> FakeStrategy:
        """Construct a tracked `FakeStrategy` for `resolver`.

        Args:
            resolver: Connected `FakeTPUClusterResolver`.

        Returns:
            Tracked `FakeStrategy`.

        """
        _ = resolver.master()
        strat = FakeStrategy()
        self.last_tpu_strategy = strat
        return strat

    def MirroredStrategy(self) -> FakeStrategy:
        """Construct a tracked `FakeStrategy` for GPU/CPU mirroring.

        Returns:
            Tracked `FakeStrategy`.

        """
        strat = FakeStrategy()
        self.last_mirrored_strategy = strat
        return strat


class FakeKerasDistributionNamespace:
    """Typed `keras.distribution` double exposing `DataParallel` and `ModelParallel`."""

    def __init__(self) -> None:
        """Initialize recorded device lists and strategies."""
        self.data_parallel_devices: list[tuple[str, ...]] = []
        self.model_parallel_calls: int = 0
        self.last_strategy: FakeStrategy | None = None

    def DataParallel(self, *, devices: Sequence[str] | None = None) -> FakeStrategy:
        """Record `devices` and return a tracked `FakeStrategy`.

        Args:
            devices: Participating accelerator device names.

        Returns:
            Tracked `FakeStrategy`.

        """
        self.data_parallel_devices.append(tuple(devices) if devices is not None else ())
        strat = FakeStrategy()
        self.last_strategy = strat
        return strat

    def ModelParallel(self, *, layout_map: Mapping[str, str] | None = None) -> FakeStrategy:
        """Record `ModelParallel` invocation and return a tracked `FakeStrategy`.

        Args:
            layout_map: Optional parameter sharding layout mapping.

        Returns:
            Tracked `FakeStrategy`.

        """
        _ = layout_map
        self.model_parallel_calls += 1
        strat = FakeStrategy()
        self.last_strategy = strat
        return strat


@dataclass
class FakeKerasGemma4Model:
    """Lightweight KerasHub Gemma 4 causal LM instance double.

    Attributes:
        preset: Resolved preset identifier (`'hf://google/gemma-4-E4B'`, etc.).

    """

    preset: str

    def generate(self, prompt: str, max_length: int = 64) -> str:
        """Return deterministic SQL string referencing `max_length`.

        Args:
            prompt: Input prompt string.
            max_length: Maximum generated sequence length.

        Returns:
            Deterministic SQL completion string.

        """
        return f"{prompt} SELECT {max_length};"


class FakeKerasGemma4CausalLM:
    """Typed `keras_hub.models.Gemma4CausalLM` loader double supporting `hf://` preset checks."""

    def __init__(self, *, require_hf_scheme: bool = False) -> None:
        """Configure preset loading behavior.

        Args:
            require_hf_scheme: If `True`, bare `'org/repo'` IDs raise `ValueError` until prefixed with `'hf://'`.

        """
        self.require_hf_scheme = require_hf_scheme
        self.loaded_presets: list[str] = []

    def from_preset(self, model_name: str, /) -> FakeKerasGemma4Model:
        """Load `FakeKerasGemma4Model` or raise `ValueError` for bare HuggingFace IDs when configured.

        Args:
            model_name: Preset name or `hf://` URI.

        Returns:
            Instantiated `FakeKerasGemma4Model`.

        Raises:
            ValueError: When `require_hf_scheme=True` and `model_name` lacks `'hf://'` or is unknown.

        """
        self.loaded_presets.append(model_name)
        if self.require_hf_scheme and not model_name.startswith("hf://"):
            raise ValueError(f"Preset '{model_name}' requires hf:// URI")
        return FakeKerasGemma4Model(preset=model_name)


@dataclass(frozen=True)
class FakeJaxTPUDevice:
    """Typed JAX TPU device double satisfying `JaxMemoryStatsDeviceProtocol`.

    Attributes:
        dev_id: Device label (`'TPU_0'`, `'TPU_1'`, etc.).
        peak_bytes: Reported `peak_bytes_in_use` or `None` when stats are disabled.
        raise_on_stats: If `True`, `memory_stats()` raises `RuntimeError`.

    """

    dev_id: str = "TPU_0"
    peak_bytes: int | None = 256 * 1024 * 1024
    raise_on_stats: bool = False

    def __str__(self) -> str:
        """Return `self.dev_id`.

        Returns:
            Device label string.

        """
        return self.dev_id

    def memory_stats(self) -> Mapping[str, int | float] | None:
        """Return HBM memory dictionary, `None`, or raise `RuntimeError`.

        Returns:
            Memory stats mapping or `None`.

        Raises:
            RuntimeError: When `self.raise_on_stats` is `True`.

        """
        if self.raise_on_stats:
            raise RuntimeError("stats unavailable")
        if self.peak_bytes is None:
            return None
        return {"peak_bytes_in_use": self.peak_bytes, "bytes_in_use": self.peak_bytes // 2}


class FakeNamedSharding:
    """Typed `jax.sharding.NamedSharding` double satisfying `JaxShardingProtocol`."""

    def __init__(self, mesh: str, spec: tuple[str | None, ...]) -> None:
        """Record `mesh` and `spec` metadata.

        Args:
            mesh: Mesh identifier string.
            spec: PartitionSpec axis tuple.

        """
        self.mesh = mesh
        self.spec = spec
        self._device_set: tuple[str | int | None, ...] = ("TPU_0", "TPU_1")

    @property
    def device_set(self) -> Sequence[str | int | None]:
        """Provide `device_set` required by `JaxShardingProtocol`."""
        return self._device_set

    def is_fully_addressable(self) -> bool:
        """Return `True` for simulated addressable sharding.

        Returns:
            `True`.

        """
        return True


class FakeJaxShardingNamespace:
    """Typed `jax.sharding` namespace double."""

    def Mesh(self, devices: Sequence[FakeJaxTPUDevice | str], axis_names: tuple[str, ...], /) -> str:
        """Format simulated mesh descriptor string.

        Args:
            devices: Mesh devices.
            axis_names: Mesh axis names.

        Returns:
            Formatted mesh string.

        """
        return f"Mesh(n={len(devices)},axes={axis_names})"

    def PartitionSpec(self, *axes: str | None) -> tuple[str | None, ...]:
        """Return `axes` tuple.

        Args:
            *axes: Partition axis names.

        Returns:
            Tuple of axis names.

        """
        return axes

    def NamedSharding(self, mesh: str, spec: tuple[str | None, ...], /) -> JaxShardingProtocol:
        """Construct a `FakeNamedSharding`.

        Args:
            mesh: Mesh descriptor string.
            spec: Partition axis tuple.

        Returns:
            `FakeNamedSharding` instance.

        """
        return FakeNamedSharding(mesh, spec)


@dataclass
class FakeJaxDistributedNamespace:
    """Typed `jax.distributed` namespace double recording `initialize` calls."""

    calls: list[dict[str, str | int | None]] = field(default_factory=_empty_dist_calls)
    should_raise: bool = False

    def initialize(self, **kwargs: str | int | None) -> None:
        """Record `kwargs` or raise `RuntimeError` when `should_raise=True`.

        Args:
            **kwargs: Coordinator initialization keyword arguments.

        Raises:
            RuntimeError: When `self.should_raise` is `True`.

        """
        self.calls.append(dict(kwargs))
        if self.should_raise:
            raise RuntimeError("already initialized")


class FakeMultiHostJaxModule:
    """Typed multi-host / multi-chip `jax` module double."""

    def __init__(
        self,
        *,
        tpu_devices: Sequence[FakeJaxTPUDevice] = (),
        proc_index: int = 0,
        proc_count: int = 1,
        dist_raises: bool = False,
    ) -> None:
        """Configure simulated TPU topology and multi-host process ranks.

        Args:
            tpu_devices: Simulated local TPU devices.
            proc_index: Zero-based host rank.
            proc_count: Total host count.
            dist_raises: Whether `distributed.initialize` raises `RuntimeError`.

        """
        self._tpu_devices = tuple(tpu_devices)
        self._cpu_devices = (FakeJaxTPUDevice("CPU_0", peak_bytes=None),)
        self._proc_index = proc_index
        self._proc_count = proc_count
        self.distributed = FakeJaxDistributedNamespace(should_raise=dist_raises)
        self.sharding = FakeJaxShardingNamespace()
        self.device_put_calls: int = 0

    def devices(self, backend: str | None = None, /) -> Sequence[FakeJaxTPUDevice]:
        """Return simulated devices for `backend`.

        Args:
            backend: Optional `'tpu'`, `'gpu'`, or `'cpu'` backend selector.

        Returns:
            Matching tuple of `FakeJaxTPUDevice` items.

        Raises:
            RuntimeError: When querying `'gpu'` (simulating absent GPU backend).

        """
        if backend == "tpu":
            return self._tpu_devices
        if backend == "gpu":
            raise RuntimeError("no gpu backend")
        return self._cpu_devices if not self._tpu_devices else self._tpu_devices

    def process_index(self) -> int:
        """Return simulated zero-based host process index.

        Returns:
            Host process index.

        """
        return self._proc_index

    def process_count(self) -> int:
        """Return simulated total host process count.

        Returns:
            Total host count.

        """
        return self._proc_count

    def local_device_count(self) -> int:
        """Return local device count.

        Returns:
            Count of local TPU or CPU devices.

        """
        return max(1, len(self._tpu_devices))

    def device_count(self) -> int:
        """Return global device count (`local * proc_count`).

        Returns:
            Global device count.

        """
        return self.local_device_count() * self._proc_count

    def device_put[ArrT](self, x: ArrT, device: JaxShardingProtocol, /) -> ArrT:
        """Record `device_put` sharding placement and return `x`.

        Args:
            x: Input array.
            device: Target `JaxShardingProtocol`.

        Returns:
            Unchanged `x`.

        """
        _ = device
        self.device_put_calls += 1
        return x

    def block_until_ready[OutT](self, x: OutT, /) -> OutT:
        """Return `x` immediately.

        Args:
            x: Output value.

        Returns:
            `x`.

        """
        return x


__all__ = [
    "FakeJaxDistributedNamespace",
    "FakeJaxShardingNamespace",
    "FakeJaxTPUDevice",
    "FakeKerasDistributionNamespace",
    "FakeKerasGemma4CausalLM",
    "FakeKerasGemma4Model",
    "FakeMultiHostJaxModule",
    "FakeNamedSharding",
    "FakeScopeContext",
    "FakeStrategy",
    "FakeTFClusterResolverNamespace",
    "FakeTFDistributeNamespace",
    "FakeTPUClusterResolver",
    "FakeTorchXLACoreModel",
    "FakeTorchXLAModule",
    "FakeTorchXLARuntime",
    "FakeXLADeviceDescriptor",
]
