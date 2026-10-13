"""Centralized lazy loader for optional backend dependencies."""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    import types
    from collections.abc import Generator
logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ImportError,)


@contextmanager
def catch_optional_imports() -> Generator[None, None, None]:
    """Context manager to gracefully catch missing optional backend dependencies.

    Yields:
        The yielded output.

    """
    with suppress(*OPTIONAL_IMPORT_ERRORS):
        yield


T = TypeVar("T", bound="types.ModuleType")


class LazyLoader(Generic[T]):
    """Lazily load an optional dependency."""

    def __init__(self, module_name: str) -> None:
        """Initialize the lazy loader.

        Args:
            module_name: The string representing the module name.

        """
        self.module_name = module_name
        self._module: T | None = None
        self._loaded = False

    def get_module(self) -> T | None:
        """Provide the module if available, otherwise return None.

        Returns:
            types.ModuleType: The resulting output from the operation.

        """
        if not self._loaded:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError:
                self._module = None
            self._loaded = True
        return self._module

    @property
    def is_available(self) -> bool:
        """Check if the module is available."""
        return self.get_module() is not None


_JAX_LOADER = LazyLoader("jax")
_JNP_LOADER = LazyLoader("jax.numpy")
_FLAX_NNX_LOADER = LazyLoader("flax.nnx")
_TENSORFLOW_LOADER = LazyLoader("tensorflow")
_KERAS_LOADER = LazyLoader("keras")
_TORCH_LOADER = LazyLoader("torch")
_MLX_LOADER = LazyLoader("mlx")
_DUCKDB_LOADER = LazyLoader("duckdb")
_TRANSFORMERS_LOADER = LazyLoader("transformers")
_SAFETENSORS_LOADER = LazyLoader("safetensors")
_MAXTEXT_GEMMA4_LOADER = LazyLoader("maxtext.models.gemma4")


def get_jax() -> T | None:
    """Provide the jax module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _JAX_LOADER.get_module()


def get_jnp() -> T | None:
    """Provide the jax.numpy module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _JNP_LOADER.get_module()


def get_flax_nnx() -> T | None:
    """Provide the flax.nnx module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _FLAX_NNX_LOADER.get_module()


def get_tensorflow() -> T | None:
    """Provide the tensorflow module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TENSORFLOW_LOADER.get_module()


def get_keras() -> T | None:
    """Provide the keras module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _KERAS_LOADER.get_module()


def get_torch() -> T | None:
    """Provide the torch module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TORCH_LOADER.get_module()


def get_mlx() -> T | None:
    """Provide the mlx module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _MLX_LOADER.get_module()


def get_duckdb() -> T | None:
    """Provide the duckdb module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _DUCKDB_LOADER.get_module()


def get_transformers() -> T | None:
    """Provide the transformers module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TRANSFORMERS_LOADER.get_module()


def get_safetensors() -> T | None:
    """Provide the safetensors module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _SAFETENSORS_LOADER.get_module()


def get_maxtext_gemma4() -> T | None:
    """Provide the maxtext.models.gemma4 module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _MAXTEXT_GEMMA4_LOADER.get_module()
