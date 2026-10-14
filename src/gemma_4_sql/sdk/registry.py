"""Backend registry for gemma-4-sql."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from typing import Any, cast

from gemma_4_sql.exceptions import ConfigurationError

from .protocols import BackendProtocol

ENTRY_POINTS: dict[str, importlib.metadata.EntryPoint] = {}
_BACKEND_CACHE: dict[str, BackendProtocol] = {}
_CACHE_LOCK = threading.Lock()


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
            eps_all = importlib.metadata.entry_points()
            eps_list: Any = cast(dict[str, Any], eps_all).get("gemma_4_sql.backends", []) if isinstance(eps_all, dict) else eps_all.select(group="gemma_4_sql.backends")
            for ep in cast(list[Any], eps_list):
                ENTRY_POINTS[ep.name] = ep

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

        adapter_class_name = f"{name.capitalize().replace('_hf', '').replace('_native', '')}Backend"
        if name == "pytorch" or name.startswith("pytorch_"):
            adapter_class_name = "PyTorchBackend"
        elif name == "mlx":
            adapter_class_name = "MLXBackend"
        elif name == "maxtext":
            adapter_class_name = "MaxTextBackend"

        try:
            adapter_class = getattr(adapter_module, adapter_class_name)
        except AttributeError as e:
            msg = f"Adapter class {adapter_class_name} not found in {adapter_module_name}"
            raise ConfigurationError(msg, backend=name, cause=e) from e

        raw_instance = adapter_class(backend_alias=name) if name.startswith("pytorch_") else adapter_class()
        if not isinstance(raw_instance, BackendProtocol):
            msg = f"Adapter instance for {name} does not satisfy BackendProtocol"
            raise TypeError(msg)
        _BACKEND_CACHE[name] = raw_instance
        return raw_instance


__all__ = ["get_backend"]
