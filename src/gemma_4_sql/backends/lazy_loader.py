"""Centralized lazy loader for optional backend dependencies."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import types
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Generic, TypeVar, overload

logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ImportError,)
_LOADER_LOCK = threading.RLock()


@contextmanager
def catch_optional_imports() -> Generator[None, None, None]:
    """Context manager to gracefully catch missing optional backend dependencies.

    Yields:
        None when entering the suppressed optional import block.

    """
    with suppress(*OPTIONAL_IMPORT_ERRORS):
        yield


T_co = TypeVar("T_co", bound=types.ModuleType, covariant=True)
TAttr = TypeVar("TAttr")

_LOADER_REGISTRY: dict[str, list[LazyLoader[types.ModuleType]]] = {}


class LazyLoader(Generic[T_co]):
    """Thread-safe lazy loader for an optional module dependency."""

    def __init__(self, module_name: str) -> None:
        """Initialize the lazy loader for the given module path.

        Args:
            module_name: Dotted module name string to import lazily.

        """
        self.module_name = module_name
        self._module: types.ModuleType | None = None
        self._loaded = False
        with _LOADER_LOCK:
            _LOADER_REGISTRY.setdefault(module_name, []).append(self)

    def reset(self) -> None:
        """Reset the cached module state so the next `get_module()` re-imports."""
        with _LOADER_LOCK:
            self._module = None
            self._loaded = False

    def set_override(self, mock_mod: types.ModuleType | None) -> tuple[bool, types.ModuleType | None]:
        """Set a temporary module override and return the previous `(loaded, module)` state.

        Args:
            mock_mod: Replacement module or `None`.

        Returns:
            Tuple of `(previous_loaded, previous_module)`.

        """
        with _LOADER_LOCK:
            prev = (self._loaded, self._module)
            self._loaded = True
            self._module = mock_mod
            return prev

    def restore_state(self, state: tuple[bool, types.ModuleType | None]) -> None:
        """Restore a previously saved `(loaded, module)` state.

        Args:
            state: Tuple of `(previous_loaded, previous_module)`.

        """
        with _LOADER_LOCK:
            self._loaded, self._module = state

    def get_module(self) -> types.ModuleType | None:
        """Provide the imported module if available, otherwise return None.

        Returns:
            The imported `types.ModuleType` instance or `None` if not installed.

        """
        with _LOADER_LOCK:
            if self.module_name in sys.modules and not sys.modules.get(self.module_name):
                return None
            if not self._loaded:
                if self.module_name == "keras" or self.module_name.startswith(("keras.", "keras_nlp")):
                    os.environ.setdefault("KERAS_BACKEND", "torch")
                try:
                    self._module = importlib.import_module(self.module_name)
                except (ImportError, AttributeError) as err:
                    logger.debug("Optional module %s not available: %s", self.module_name, err)
                    self._module = None
                self._loaded = True
            return self._module

    @property
    def is_available(self) -> bool:
        """Indicate whether the target module can be imported."""
        return self.get_module() is not None


def reset_all_lazy_loaders() -> None:
    """Reset all registered `LazyLoader` instances in a thread-safe manner."""
    with _LOADER_LOCK:
        for loader_list in _LOADER_REGISTRY.values():
            for loader in loader_list:
                loader.reset()


@contextmanager
def override_lazy_module(module_name: str, mock_mod: types.ModuleType | None) -> Generator[None, None, None]:
    """Temporarily override all `LazyLoader` instances for `module_name` within a context block.

    Args:
        module_name: Dotted module name registered in `_LOADER_REGISTRY`.
        mock_mod: Replacement module or `None` to simulate a missing dependency.

    Yields:
        None while the override is active.

    """
    with _LOADER_LOCK:
        loaders = _LOADER_REGISTRY.get(module_name)
        if not loaders:
            LazyLoader(module_name)
            loaders = _LOADER_REGISTRY[module_name]
        saved_states = [(ldr, ldr.set_override(mock_mod)) for ldr in loaders]
    try:
        yield
    finally:
        with _LOADER_LOCK:
            for ldr, prev_state in saved_states:
                ldr.restore_state(prev_state)


def optional_import(module_name: str) -> types.ModuleType | None:
    """Import an optional module using LazyLoader, returning None if unavailable.

    Args:
        module_name: Dotted module name string to import.

    Returns:
        The imported `types.ModuleType` instance or `None` if unavailable.

    """
    with _LOADER_LOCK:
        existing = _LOADER_REGISTRY.get(module_name)
        loader = existing[0] if existing else LazyLoader(module_name)
    mod = loader.get_module()
    if module_name in sys.modules:
        return sys.modules.get(module_name)
    return mod


@overload
def optional_attr(module_name: str, attr_name: str, _default_type: type[TAttr]) -> TAttr | None:  # noqa: D418
    """Import an optional module and retrieve an attribute typed as `TAttr | None`."""


@overload
def optional_attr(module_name: str, attr_name: str, _default_type: None = None) -> types.ModuleType | None:  # noqa: D418
    """Import an optional module and retrieve a dynamic attribute or submodule."""


def optional_attr(module_name: str, attr_name: str, _default_type: type[TAttr] | None = None) -> TAttr | types.ModuleType | None:
    """Import an optional module and retrieve an attribute from it, returning None if unavailable.

    Args:
        module_name: Dotted module name string to import.
        attr_name: Attribute name on the target module.
        _default_type: Optional expected attribute type for static type inference.

    Returns:
        The attribute value from the module or `None` if the module or attribute is unavailable.

    """
    mod = optional_import(module_name)
    if mod is None:
        return None
    try:
        return getattr(mod, attr_name, None)
    except (ImportError, AttributeError, RuntimeError):
        return None


_JAX_LOADER: LazyLoader[types.ModuleType] = LazyLoader("jax")
_JNP_LOADER: LazyLoader[types.ModuleType] = LazyLoader("jax.numpy")
_FLAX_NNX_LOADER: LazyLoader[types.ModuleType] = LazyLoader("flax.nnx")
_TENSORFLOW_LOADER: LazyLoader[types.ModuleType] = LazyLoader("tensorflow")
_KERAS_LOADER: LazyLoader[types.ModuleType] = LazyLoader("keras")
_TORCH_LOADER: LazyLoader[types.ModuleType] = LazyLoader("torch")
_MLX_LOADER: LazyLoader[types.ModuleType] = LazyLoader("mlx")
_DUCKDB_LOADER: LazyLoader[types.ModuleType] = LazyLoader("duckdb")
_TRANSFORMERS_LOADER: LazyLoader[types.ModuleType] = LazyLoader("transformers")
_SAFETENSORS_LOADER: LazyLoader[types.ModuleType] = LazyLoader("safetensors")
_MAXTEXT_GEMMA4_LOADER: LazyLoader[types.ModuleType] = LazyLoader("maxtext.models.gemma4")


def get_jax() -> types.ModuleType | None:
    """Provide the `jax` module if installed.

    Returns:
        The `jax` module or `None`.

    """
    return _JAX_LOADER.get_module()


def get_jnp() -> types.ModuleType | None:
    """Provide the `jax.numpy` module if installed.

    Returns:
        The `jax.numpy` module or `None`.

    """
    return _JNP_LOADER.get_module()


def get_flax_nnx() -> types.ModuleType | None:
    """Provide the `flax.nnx` module if installed.

    Returns:
        The `flax.nnx` module or `None`.

    """
    return _FLAX_NNX_LOADER.get_module()


def get_tensorflow() -> types.ModuleType | None:
    """Provide the `tensorflow` module if installed.

    Returns:
        The `tensorflow` module or `None`.

    """
    return _TENSORFLOW_LOADER.get_module()


def get_keras() -> types.ModuleType | None:
    """Provide the `keras` module if installed.

    Returns:
        The `keras` module or `None`.

    """
    return _KERAS_LOADER.get_module()


def get_torch() -> types.ModuleType | None:
    """Provide the `torch` module if installed.

    Returns:
        The `torch` module or `None`.

    """
    return _TORCH_LOADER.get_module()


def get_mlx() -> types.ModuleType | None:
    """Provide the `mlx` module if installed.

    Returns:
        The `mlx` module or `None`.

    """
    return _MLX_LOADER.get_module()


def get_duckdb() -> types.ModuleType | None:
    """Provide the `duckdb` module if installed.

    Returns:
        The `duckdb` module or `None`.

    """
    return _DUCKDB_LOADER.get_module()


def get_transformers() -> types.ModuleType | None:
    """Provide the `transformers` module if installed.

    Returns:
        The `transformers` module or `None`.

    """
    return _TRANSFORMERS_LOADER.get_module()


def get_safetensors() -> types.ModuleType | None:
    """Provide the `safetensors` module if installed.

    Returns:
        The `safetensors` module or `None`.

    """
    return _SAFETENSORS_LOADER.get_module()


def get_maxtext_gemma4() -> types.ModuleType | None:
    """Provide the `maxtext.models.gemma4` module if installed.

    Returns:
        The `maxtext.models.gemma4` module or `None`.

    """
    return _MAXTEXT_GEMMA4_LOADER.get_module()


__all__ = [
    "OPTIONAL_IMPORT_ERRORS",
    "LazyLoader",
    "catch_optional_imports",
    "get_duckdb",
    "get_flax_nnx",
    "get_jax",
    "get_jnp",
    "get_keras",
    "get_maxtext_gemma4",
    "get_mlx",
    "get_safetensors",
    "get_tensorflow",
    "get_torch",
    "get_transformers",
    "optional_attr",
    "optional_import",
    "override_lazy_module",
    "reset_all_lazy_loaders",
]
