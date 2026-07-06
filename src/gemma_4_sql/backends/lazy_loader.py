"""Centralized lazy loader for optional backend dependencies."""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types
    from collections.abc import Iterator
logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ImportError,)


@contextmanager
def catch_optional_imports() -> Iterator[None]:
    """Context manager to gracefully catch missing optional backend dependencies.

    Yields:
        The yielded output.
    """
    with suppress(OPTIONAL_IMPORT_ERRORS):
        yield


class LazyLoader:
    """Lazily load an optional dependency."""

    def __init__(self, module_name: str) -> None:
        """Initialize the lazy loader.

        Args:
            module_name: The string representing the module name.
        """
        self.module_name = module_name
        self._module: types.ModuleType | None = None
        self._loaded = False

    def get_module(self) -> types.ModuleType | None:
        """Get the module if available, otherwise return None.

        Returns:
            types.ModuleType: The resulting output from the operation.

        """
        if not self._loaded:  # pragma: no cover
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError:
                self._module = None
            self._loaded = True
        return self._module

    @property
    def is_available(self) -> bool:
        """Check if the module is available."""
        return self.get_module() is not None  # pragma: no cover


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


def get_jax() -> types.ModuleType | None:
    """Get the jax module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _JAX_LOADER.get_module()


def get_jnp() -> types.ModuleType | None:
    """Get the jax.numpy module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _JNP_LOADER.get_module()


def get_flax_nnx() -> types.ModuleType | None:
    """Get the flax.nnx module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _FLAX_NNX_LOADER.get_module()


def get_tensorflow() -> types.ModuleType | None:
    """Get the tensorflow module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TENSORFLOW_LOADER.get_module()


def get_keras() -> types.ModuleType | None:
    """Get the keras module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _KERAS_LOADER.get_module()


def get_torch() -> types.ModuleType | None:
    """Get the torch module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TORCH_LOADER.get_module()


def get_mlx() -> types.ModuleType | None:
    """Get the mlx module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _MLX_LOADER.get_module()


def get_duckdb() -> types.ModuleType | None:
    """Get the duckdb module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _DUCKDB_LOADER.get_module()


def get_transformers() -> types.ModuleType | None:
    """Get the transformers module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _TRANSFORMERS_LOADER.get_module()


def get_safetensors() -> types.ModuleType | None:
    """Get the safetensors module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _SAFETENSORS_LOADER.get_module()


def get_maxtext_gemma4() -> types.ModuleType | None:
    """Get the maxtext.models.gemma4 module.

    Returns:
        types.ModuleType: The resulting output from the operation.

    """
    return _MAXTEXT_GEMMA4_LOADER.get_module()
