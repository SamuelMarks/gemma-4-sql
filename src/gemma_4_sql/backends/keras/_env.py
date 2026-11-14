"""Explicit Keras 3 backend environment configuration and mid-process conflict validation."""

from __future__ import annotations

import os
import sys
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.constants import KERAS_BACKEND_ALIAS_MAP
from gemma_4_sql.exceptions import ConfigurationError


@runtime_checkable
class _KerasBackendQueryNamespaceProtocol(Protocol):
    """Protocol for `keras.backend` exposing `.backend()`."""

    def backend(self) -> str:
        """Return the active Keras 3 backend name (`'jax'`, `'tensorflow'`, or `'torch'`).

        Returns:
            Active Keras backend identifier string.

        """
        ...


@runtime_checkable
class _KerasWithBackendProtocol(Protocol):
    """Protocol for a loaded `keras` module exposing `.backend.backend()`."""

    @property
    def backend(self) -> _KerasBackendQueryNamespaceProtocol:
        """Provide the `keras.backend` namespace."""
        ...


def ensure_keras_backend_configured(default_backend: str = "torch") -> str:
    """Ensure `KERAS_BACKEND` is configured before `keras` initializes its backend.

    Only pre-imports `torch` when the resolved `KERAS_BACKEND` is `'torch'`, avoiding unnecessary
    PyTorch imports inside pure `keras_jax` or `keras_tensorflow` virtual environments.

    Args:
        default_backend: Default Keras backend when `KERAS_BACKEND` is unset.

    Returns:
        Active `KERAS_BACKEND` environment string.

    """
    active = os.environ.setdefault("KERAS_BACKEND", default_backend)
    if active == "torch":
        _ = optional_import("torch")
    return active


def resolve_and_validate_keras_backend(
    backend_alias: str | None = None,
    explicit_backend: str | None = None,
) -> str:
    """Resolve the target `KERAS_BACKEND` from `backend_alias` and verify process consistency.

    Because Keras 3 binds its computation backend once when `keras` is first imported in a Python
    process, requesting an incompatible alias (e.g., `keras_jax` inside a process that already
    initialized `KERAS_BACKEND='torch'`) raises `ConfigurationError` rather than silently running
    on the wrong engine.

    Args:
        backend_alias: Optional entry-point alias (`'keras_jax'`, `'keras_tensorflow'`, `'keras_pytorch'`).
        explicit_backend: Optional explicit backend override (`'jax'`, `'tensorflow'`, `'torch'`).

    Returns:
        Resolved active Keras backend name (`'jax'`, `'tensorflow'`, or `'torch'`).

    Raises:
        ConfigurationError: If `keras` is already imported with a different backend than requested.

    """
    requested: str | None = explicit_backend
    if requested is None and backend_alias is not None:
        requested = KERAS_BACKEND_ALIAS_MAP.get(backend_alias)

    loaded_keras = sys.modules.get("keras")
    if isinstance(loaded_keras, _KerasWithBackendProtocol):
        current = str(loaded_keras.backend.backend())
        if requested is not None and current != requested:
            msg = (
                f"Keras backend mismatch for '{backend_alias or requested}': current Python process "
                f"already initialized KERAS_BACKEND='{current}', cannot switch to '{requested}' "
                f"mid-process. Run each Keras backend in an isolated process or virtualenv."
            )
            raise ConfigurationError(msg, backend=backend_alias or "keras")
        return current

    if requested is not None:
        os.environ["KERAS_BACKEND"] = requested
        return ensure_keras_backend_configured(requested)
    return os.environ.get("KERAS_BACKEND", ACTIVE_KERAS_BACKEND)


ACTIVE_KERAS_BACKEND: str = ensure_keras_backend_configured()

__all__ = [
    "ACTIVE_KERAS_BACKEND",
    "ensure_keras_backend_configured",
    "resolve_and_validate_keras_backend",
]
