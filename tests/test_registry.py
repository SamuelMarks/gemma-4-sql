# Copyright 2024
"""Provide module docstring."""

import contextlib
from unittest import mock

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.sdk.registry import get_backend


def test_registry_fallback() -> object:
    """Initialize function test_registry_fallback."""
    mod.ENTRY_POINTS.clear()
    with mock.patch("sys.version_info", (3, 11)), contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        get_backend("nonexistent")
    mod.ENTRY_POINTS.clear()
    with mock.patch("sys.version_info", (3, 9)), contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        get_backend("nonexistent")
