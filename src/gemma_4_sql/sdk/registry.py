"""Backend registry for gemma-4-sql."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol
from gemma_4_sql.exceptions import ConfigurationError

from .protocols import BackendProtocol


@runtime_checkable
class EntryPointProtocol(Protocol):
    """Protocol representing an `importlib.metadata.EntryPoint` entry."""

    @property
    def name(self) -> str:
        """Provide the entry point name."""
        ...

    def load(self) -> ModuleNamespaceProtocol:
        """Load and return the target backend module."""
        ...


@runtime_checkable
class SelectableEntryPointsProtocol(Protocol):
    """Protocol representing `importlib.metadata.EntryPoints` with `.select(group=...)`."""

    def select(self, *, group: str) -> Iterable[EntryPointProtocol]:
        """Select entry points belonging to `group`.

        Args:
            group: Entry-point group identifier.

        Returns:
            Iterable of matching `EntryPointProtocol` descriptors.

        """
        ...


@runtime_checkable
class AliasFactoryAdapterProtocol(Protocol):
    """Protocol for adapter classes exposing a `.from_alias(name)` factory method."""

    def from_alias(self, alias: str) -> BackendProtocol | ModuleNamespaceProtocol:
        """Construct an adapter instance for the registered backend alias.

        Args:
            alias: Registered backend identifier.

        Returns:
            Constructed adapter instance.

        """
        ...


@runtime_checkable
class AdapterModuleProtocol(Protocol):
    """Protocol representing a backend `.adapter` module exposing `ADAPTER_CLASS`."""

    @property
    def ADAPTER_CLASS(
        self,
    ) -> Callable[..., BackendProtocol | ModuleNamespaceProtocol] | AliasFactoryAdapterProtocol | None:
        """Provide the backend adapter class constructor."""
        ...


type EntryPointsCollection = Mapping[str, Iterable[EntryPointProtocol]] | SelectableEntryPointsProtocol
type EntryPointsProviderCallable = Callable[[], EntryPointsCollection]
type ModuleLoaderCallable = Callable[[str], ModuleNamespaceProtocol]

ENTRY_POINTS: dict[str, EntryPointProtocol] = {}
_BACKEND_CACHE: dict[str, BackendProtocol] = {}
_CACHE_LOCK = threading.RLock()


def _default_entry_points_provider() -> EntryPointsCollection:
    """Return discovered entry points from `importlib.metadata.entry_points()`.

    Returns:
        Entry-points collection supporting `.select(group=...)` or group-keyed mapping access.

    """
    return importlib.metadata.entry_points()


def _default_module_loader(module_name: str) -> ModuleNamespaceProtocol:
    """Import a module namespace via `importlib.import_module`.

    Args:
        module_name: Dotted module path.

    Returns:
        Imported `ModuleNamespaceProtocol` instance.

    """
    return importlib.import_module(module_name)


def clear_backend_cache() -> None:
    """Clear the cached backend instances and discovered entry points."""
    with _CACHE_LOCK:
        ENTRY_POINTS.clear()
        _BACKEND_CACHE.clear()


@contextmanager
def override_backend(name: str, backend_impl: BackendProtocol) -> Generator[BackendProtocol, None, None]:
    """Temporarily register `backend_impl` in `_BACKEND_CACHE[name]` for dependency injection.

    Args:
        name: Backend alias to override (e.g., `'jax'`, `'pytorch'`).
        backend_impl: Typed `BackendProtocol` implementation or test double.

    Yields:
        The active `backend_impl` instance.

    """
    with _CACHE_LOCK:
        had_prev = name in _BACKEND_CACHE
        prev = _BACKEND_CACHE.get(name)
        _BACKEND_CACHE[name] = backend_impl
    try:
        yield backend_impl
    finally:
        with _CACHE_LOCK:
            if had_prev and prev is not None:
                _BACKEND_CACHE[name] = prev
            else:
                _BACKEND_CACHE.pop(name, None)


def _discover_entry_points(
    entry_points_provider: EntryPointsProviderCallable = _default_entry_points_provider,
) -> None:
    """Populate `ENTRY_POINTS` from `entry_points_provider()`.

    Args:
        entry_points_provider: Callable returning installed entry-point descriptors.

    """
    eps_all: EntryPointsCollection = entry_points_provider()
    eps_list: Iterable[EntryPointProtocol] = (
        eps_all.select(group="gemma_4_sql.backends")
        if isinstance(eps_all, SelectableEntryPointsProtocol)
        else eps_all.get("gemma_4_sql.backends", ())
    )
    for ep in eps_list:
        ENTRY_POINTS[ep.name] = ep


def validate_backend_name(
    name: str,
    *,
    entry_points_provider: EntryPointsProviderCallable = _default_entry_points_provider,
) -> None:
    """Validate that `name` is a registered backend without importing heavy ML runtimes.

    Args:
        name: Backend identifier (e.g., `'jax'`, `'pytorch'`, `'keras'`, `'maxtext'`, `'mlx'`).
        entry_points_provider: Optional entry-points discovery provider for dependency injection.

    Raises:
        ConfigurationError: If `name` is not registered in `_BACKEND_CACHE` or `ENTRY_POINTS`.

    """
    with _CACHE_LOCK:
        if name in _BACKEND_CACHE:
            return
        if not ENTRY_POINTS:
            _discover_entry_points(entry_points_provider)
        if name not in ENTRY_POINTS:
            msg = f"Unknown backend: {name}"
            raise ConfigurationError(msg, backend=name)


def get_backend(
    name: str,
    *,
    entry_points_provider: EntryPointsProviderCallable = _default_entry_points_provider,
    module_loader: ModuleLoaderCallable = _default_module_loader,
) -> BackendProtocol:
    """Provide a registered backend instance by name.

    Args:
        name: The backend identifier (e.g., `'jax'`, `'pytorch'`, `'keras'`, `'maxtext'`, `'mlx'`).
        entry_points_provider: Optional entry-points discovery provider for dependency injection.
        module_loader: Optional adapter module loader for dependency injection.

    Returns:
        The instantiated `BackendProtocol` implementation.

    Raises:
        ConfigurationError: If the backend name is unknown or its adapter module/class is missing.
        TypeError: If the resolved adapter instance does not satisfy `BackendProtocol`.

    """
    with _CACHE_LOCK:
        if name in _BACKEND_CACHE:
            return _BACKEND_CACHE[name]

        if not ENTRY_POINTS:
            _discover_entry_points(entry_points_provider)

        if name not in ENTRY_POINTS:
            msg = f"Unknown backend: {name}"
            raise ConfigurationError(msg, backend=name)

        module = ENTRY_POINTS[name].load()

        adapter_module_name = f"{module.__name__}.adapter"
        try:
            adapter_module = module_loader(adapter_module_name)
        except ImportError as e:
            msg = f"Backend {name} is missing the adapter module."
            raise ConfigurationError(msg, backend=name, cause=e) from e

        adapter_class: Callable[..., BackendProtocol | ModuleNamespaceProtocol] | AliasFactoryAdapterProtocol | None = (
            None
        )
        if isinstance(adapter_module, AdapterModuleProtocol):
            adapter_class = adapter_module.ADAPTER_CLASS

        if adapter_class is None:
            adapter_class_name = f"{name.capitalize()}Backend"
            msg = f"Adapter class {adapter_class_name} not found in {adapter_module_name}"
            raise ConfigurationError(msg, backend=name)

        if isinstance(adapter_class, AliasFactoryAdapterProtocol):
            raw_instance = adapter_class.from_alias(name)
        else:
            raw_instance = adapter_class()
        if not isinstance(raw_instance, BackendProtocol):
            msg = f"Adapter instance for {name} does not satisfy BackendProtocol"
            raise TypeError(msg)
        _BACKEND_CACHE[name] = raw_instance
        return raw_instance


__all__ = [
    "AdapterModuleProtocol",
    "AliasFactoryAdapterProtocol",
    "EntryPointProtocol",
    "EntryPointsCollection",
    "EntryPointsProviderCallable",
    "ModuleLoaderCallable",
    "SelectableEntryPointsProtocol",
    "clear_backend_cache",
    "get_backend",
    "override_backend",
    "validate_backend_name",
]
