"""Centralized lazy loader for optional backend dependencies."""

from __future__ import annotations

import importlib
import logging
import types
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ImportError,)


@contextmanager
def catch_optional_imports() -> Generator[None, None, None]:
    """Context manager to gracefully catch missing optional backend dependencies.

    Yields:
        None when entering the suppressed optional import block.

    """
    with suppress(*OPTIONAL_IMPORT_ERRORS):
        yield


T = TypeVar("T", bound=types.ModuleType)


class LazyLoader(Generic[T]):
    """Lazily load an optional module dependency upon first request."""

    def __init__(self, module_name: str) -> None:
        """Initialize the lazy loader for the given module path.

        Args:
            module_name: Dotted module name string to import lazily.

        """
        self.module_name = module_name
        self._module: T | None = None
        self._loaded = False

    def get_module(self) -> T | None:
        """Provide the imported module if available, otherwise return None.

        Returns:
            The imported `types.ModuleType` instance or `None` if not installed.

        """
        if not self._loaded:
            try:
                loaded_mod: Any = importlib.import_module(self.module_name)
                self._module = loaded_mod
            except (ImportError, AttributeError) as err:
                logger.debug("Optional module %s not available: %s", self.module_name, err)
                self._module = None
            self._loaded = True
        return self._module

    @property
    def is_available(self) -> bool:
        """Return True if the target module can be imported."""
        return self.get_module() is not None


def optional_import(module_name: str) -> Any:
    """Import an optional module using LazyLoader, returning None if unavailable.

    Args:
        module_name: Dotted module name string to import.

    Returns:
        The imported `types.ModuleType` instance or `None` if unavailable.

    """
    return LazyLoader(module_name).get_module()


def optional_attr(module_name: str, attr_name: str) -> Any:
    """Import an optional module and retrieve an attribute from it, returning None if unavailable.

    Args:
        module_name: Dotted module name string to import.
        attr_name: Attribute name on the target module.

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
