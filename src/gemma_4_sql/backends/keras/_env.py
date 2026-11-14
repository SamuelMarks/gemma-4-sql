"""Explicit Keras backend environment configuration prior to importing `keras`."""

from __future__ import annotations

import os

from gemma_4_sql.backends.lazy_loader import optional_import


def ensure_keras_backend_configured(default_backend: str = "torch") -> str:
    """Ensure `KERAS_BACKEND` is configured before `keras` initializes its backend.

    Args:
        default_backend: Default Keras backend when `KERAS_BACKEND` is unset.

    Returns:
        Active `KERAS_BACKEND` environment string.

    """
    _ = optional_import("torch")
    return os.environ.setdefault("KERAS_BACKEND", default_backend)


ACTIVE_KERAS_BACKEND: str = ensure_keras_backend_configured()

__all__ = ["ACTIVE_KERAS_BACKEND", "ensure_keras_backend_configured"]
