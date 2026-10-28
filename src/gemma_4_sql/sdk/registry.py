"""Backend registry for gemma-4-sql."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, runtime_checkable

from gemma_4_sql.exceptions import ConfigurationError

from .protocols import BackendProtocol


@runtime_checkable
class EntryPointProtocol(Protocol):
    """Protocol representing an `importlib.metadata.EntryPoint` entry."""

    @property
    def name(self) -> str:
        """Provide the entry point name."""
        ...

    def load(self) -> types.ModuleType:
        """Load and return the target backend module."""
        ...


@runtime_checkable
class SelectableEntryPointsProtocol(Protocol):
    """Protocol representing `importlib.metadata.EntryPoints` with `.select(group=...)`."""

    def select(self, *, group: str) -> Iterable[EntryPointProtocol]:
        """Select entry points belonging to `group`."""
        ...


ENTRY_POINTS: dict[str, EntryPointProtocol] = {}
_BACKEND_CACHE: dict[str, BackendProtocol] = {}
_CACHE_LOCK = threading.Lock()


def clear_backend_cache() -> None:
    """Clear the cached backend instances and discovered entry points."""
    with _CACHE_LOCK:
        ENTRY_POINTS.clear()
        _BACKEND_CACHE.clear()


def _discover_entry_points() -> None:
    """Populate `ENTRY_POINTS` from `importlib.metadata.entry_points()`."""
    eps_all: Mapping[str, Iterable[EntryPointProtocol]] | SelectableEntryPointsProtocol = importlib.metadata.entry_points()
    eps_list: Iterable[EntryPointProtocol] = eps_all.get("gemma_4_sql.backends", ()) if isinstance(eps_all, Mapping) else eps_all.select(group="gemma_4_sql.backends")
    for ep in eps_list:
        ENTRY_POINTS[ep.name] = ep


def get_backend(name: str) -> BackendProtocol:
    """Provide a registered backend instance by name.

    Args:
        name: The backend identifier (e.g., 'jax', 'pytorch', 'keras', 'maxtext', 'mlx').

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
            _discover_entry_points()

        if name not in ENTRY_POINTS:
            msg = f"Unknown backend: {name}"
            raise ConfigurationError(msg, backend=name)

        module = ENTRY_POINTS[name].load()

        adapter_module_name = f"{module.__name__}.adapter"
        try:
            adapter_module = importlib.import_module(adapter_module_name)
        except ImportError as e:
            msg = f"Backend {name} is missing the adapter module."
            raise ConfigurationError(msg, backend=name, cause=e) from e

        adapter_class: Callable[..., BackendProtocol | types.ModuleType] | None = None
        try:
            adapter_class = adapter_module.ADAPTER_CLASS
        except AttributeError:
            adapter_class = None

        if adapter_class is None:
            adapter_class_name = f"{name.capitalize()}Backend"
            mod_dict = vars(adapter_module)
            if adapter_class_name not in mod_dict:
                msg = f"Adapter class {adapter_class_name} not found in {adapter_module_name}"
                raise ConfigurationError(msg, backend=name)
            resolved_cls: Callable[..., BackendProtocol | types.ModuleType] = mod_dict[adapter_class_name]
            adapter_class = resolved_cls

        raw_instance = adapter_class(backend_alias=name) if name.startswith("pytorch_") else adapter_class()
        if not isinstance(raw_instance, BackendProtocol):
            msg = f"Adapter instance for {name} does not satisfy BackendProtocol"
            raise TypeError(msg)
        _BACKEND_CACHE[name] = raw_instance
        return raw_instance


__all__ = [
    "EntryPointProtocol",
    "SelectableEntryPointsProtocol",
    "clear_backend_cache",
    "get_backend",
]
