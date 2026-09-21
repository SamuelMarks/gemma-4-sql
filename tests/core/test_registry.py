"""Tests for the SDK backend registry module."""

from __future__ import annotations

import contextlib
import importlib.metadata
from unittest import mock

import pytest

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.sdk.registry import get_backend


def test_registry_fallback() -> None:
    """Test registry fallback behavior across different mock python versions."""
    mod.ENTRY_POINTS.clear()
    with mock.patch("sys.version_info", (3, 11)), contextlib.suppress(ValueError):
        get_backend("nonexistent")
    mod.ENTRY_POINTS.clear()
    with mock.patch("sys.version_info", (3, 9)), contextlib.suppress(ValueError):
        get_backend("nonexistent")


def test_registry_dict_entry_points() -> None:
    """Test get_backend when importlib.metadata.entry_points returns a dict."""
    mod.ENTRY_POINTS.clear()
    fake_ep = mock.MagicMock(spec=importlib.metadata.EntryPoint)
    fake_ep.name = "mock_backend"
    fake_backend = mock.MagicMock()
    fake_ep.load.return_value = fake_backend

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        backend = get_backend("mock_backend")
        assert backend is fake_backend
        fake_ep.load.assert_called_once()


def test_registry_select_entry_points() -> None:
    """Test get_backend when importlib.metadata.entry_points has a .select() method."""
    mod.ENTRY_POINTS.clear()
    fake_ep = mock.MagicMock(spec=importlib.metadata.EntryPoint)
    fake_ep.name = "select_backend"
    fake_backend = mock.MagicMock()
    fake_ep.load.return_value = fake_backend

    mock_eps = mock.MagicMock()
    mock_eps.select.return_value = [fake_ep]

    with mock.patch("importlib.metadata.entry_points", return_value=mock_eps):
        backend = get_backend("select_backend")
        assert backend is fake_backend
        mock_eps.select.assert_called_once_with(group="gemma_4_sql.backends")


def test_registry_caching() -> None:
    """Test that entry points are cached after the first lookup."""
    mod.ENTRY_POINTS.clear()
    fake_ep = mock.MagicMock(spec=importlib.metadata.EntryPoint)
    fake_ep.name = "cached_backend"
    fake_ep.load.return_value = "mock_loaded"

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}) as mock_ep_func:
        get_backend("cached_backend")
        assert mock_ep_func.call_count == 1
        get_backend("cached_backend")
        assert mock_ep_func.call_count == 1


def test_registry_unknown_backend() -> None:
    """Test that requesting an unregistered backend name raises ValueError."""
    mod.ENTRY_POINTS.clear()
    with pytest.raises(ValueError, match="Unknown backend: definitely_not_real"):
        get_backend("definitely_not_real")
