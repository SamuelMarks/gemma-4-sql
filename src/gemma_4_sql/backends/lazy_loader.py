"""Centralized lazy loader for optional backend dependencies."""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError)


@contextmanager
def catch_optional_imports() -> Iterator[None]:
    """Context manager to gracefully catch missing optional backend dependencies."""
    with suppress(OPTIONAL_IMPORT_ERRORS):
        yield


class LazyLoader:
    """Lazily load an optional dependency."""

    def __init__(self, module_name: str) -> None:
        """Initialize the lazy loader."""
        self.module_name = module_name
        self._module: object | None = None
        self._loaded = False

    def get_module(self) -> object | None:
        """Get the module if available, otherwise return None."""
        if not self._loaded:
            try:
                self._module = importlib.import_module(self.module_name)
            except (ImportError, ValueError, TypeError, AttributeError, RuntimeError, OSError):
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


def get_jax() -> object | None:
    """Get the jax module."""
    return _JAX_LOADER.get_module()


def get_jnp() -> object | None:
    """Get the jax.numpy module."""
    return _JNP_LOADER.get_module()


def get_flax_nnx() -> object | None:
    """Get the flax.nnx module."""
    return _FLAX_NNX_LOADER.get_module()


def get_tensorflow() -> object | None:
    """Get the tensorflow module."""
    return _TENSORFLOW_LOADER.get_module()


def get_keras() -> object | None:
    """Get the keras module."""
    return _KERAS_LOADER.get_module()


def get_torch() -> object | None:
    """Get the torch module."""
    return _TORCH_LOADER.get_module()


def get_mlx() -> object | None:
    """Get the mlx module."""
    return _MLX_LOADER.get_module()


def get_duckdb() -> object | None:
    """Get the duckdb module."""
    return _DUCKDB_LOADER.get_module()


def get_transformers() -> object | None:
    """Get the transformers module."""
    return _TRANSFORMERS_LOADER.get_module()


def get_safetensors() -> object | None:
    """Get the safetensors module."""
    return _SAFETENSORS_LOADER.get_module()


def get_maxtext_gemma4() -> object | None:
    """Get the maxtext.models.gemma4 module."""
    return _MAXTEXT_GEMMA4_LOADER.get_module()
