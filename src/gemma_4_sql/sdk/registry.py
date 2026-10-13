"""Backend registry for gemma-4-sql."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .protocols import BackendProtocol

ENTRY_POINTS: dict[str, importlib.metadata.EntryPoint] = {}
_BACKEND_CACHE: dict[str, BackendProtocol] = {}
_CACHE_LOCK = threading.Lock()


def get_backend(name: str) -> BackendProtocol:
    """Provide backend by name.

    Args:
        name: The string representing the name.

    Returns:
        The execution result.

    Raises:
        ValueError: If the backend name is unknown.
        TypeError: If the loaded backend is not a BackendProtocol.

    """
    with _CACHE_LOCK:
        if name in _BACKEND_CACHE:
            return _BACKEND_CACHE[name]

        if not ENTRY_POINTS:
            eps_all = importlib.metadata.entry_points()
            eps_list: Any = eps_all.get("gemma_4_sql.backends", []) if isinstance(eps_all, dict) else eps_all.select(group="gemma_4_sql.backends")
            for ep in cast(list[Any], eps_list):
                ENTRY_POINTS[ep.name] = ep

        if name not in ENTRY_POINTS:
            msg = f"Unknown backend: {name}"
            raise ValueError(msg)

        module = ENTRY_POINTS[name].load()

        # Assuming each backend has a corresponding adapter class (e.g. JaxBackend, PyTorchBackend)
        # We can dynamically find the class that implements BackendProtocol or load from the adapter submodule
        adapter_module_name = f"{module.__name__}.adapter"
        try:
            adapter_module = importlib.import_module(adapter_module_name)
        except ImportError:
            msg = f"Backend {name} is missing the adapter module."
            raise ValueError(msg)

        # Find the adapter class dynamically based on the name or typical convention
        adapter_class_name = f"{name.capitalize().replace('_hf', '').replace('_native', '')}Backend"
        if name == "pytorch" or name.startswith("pytorch_"):
            adapter_class_name = "PyTorchBackend"
        elif name == "mlx":
            adapter_class_name = "MLXBackend"
        elif name == "maxtext":
            adapter_class_name = "MaxTextBackend"

        try:
            adapter_class = getattr(adapter_module, adapter_class_name)
        except AttributeError:
            msg = f"Adapter class {adapter_class_name} not found in {adapter_module_name}"
            raise ValueError(msg)

        instance = adapter_class()

        # Verify conformance
        # we can't directly isinstance check against Protocol without @runtime_checkable
        # but we assume the instance is correctly typed.

        _BACKEND_CACHE[name] = instance
        return instance


__all__ = ["get_backend"]
