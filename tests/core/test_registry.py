"""Tests for the SDK backend registry module."""

from __future__ import annotations

from unittest import mock

import pytest

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.sdk.registry import get_backend


def test_registry_fallback() -> None:
    """Test registry fallback behavior."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    with pytest.raises(ValueError, match="Unknown backend: nonexistent"):
        get_backend("nonexistent")


def test_registry_missing_adapter_module(monkeypatch) -> None:
    """Test registry missing adapter module behavior."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    fake_ep = mock.MagicMock()
    fake_ep.name = "mock_backend"

    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "mock_module.adapter":
            raise ImportError("Mock missing adapter")
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        with pytest.raises(ValueError, match="Backend mock_backend is missing the adapter module."):
            get_backend("mock_backend")


def test_registry_missing_adapter_class(monkeypatch) -> None:
    """Test registry missing adapter class behavior."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    fake_ep = mock.MagicMock()
    fake_ep.name = "mock_backend"

    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock(spec=[])

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        with pytest.raises(ValueError, match="Adapter class Mock_backendBackend not found in mock_module.adapter"):
            get_backend("mock_backend")


def test_registry_dict_entry_points(monkeypatch) -> None:
    """Test get_backend when importlib.metadata.entry_points returns a dict."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    fake_ep = mock.MagicMock()
    fake_ep.name = "mock_backend"

    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock()
    fake_adapter.Mock_backendBackend = fake_adapter_class

    # fake importlib.import_module
    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        backend = get_backend("mock_backend")
        assert backend is fake_adapter_class.return_value
        fake_ep.load.assert_called_once()


def test_registry_select_entry_points(monkeypatch) -> None:
    """Test get_backend when importlib.metadata.entry_points has a .select() method."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()
    fake_ep = mock.MagicMock()
    fake_ep.name = "select_backend"
    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock()
    fake_adapter.Select_backendBackend = fake_adapter_class

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    mock_eps = mock.MagicMock()
    mock_eps.select.return_value = [fake_ep]

    with mock.patch("importlib.metadata.entry_points", return_value=mock_eps):
        backend = get_backend("select_backend")
        assert backend is fake_adapter_class.return_value
        mock_eps.select.assert_called_once_with(group="gemma_4_sql.backends")


def test_registry_caching(monkeypatch) -> None:
    """Test that entry points are cached after the first lookup."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()
    fake_ep = mock.MagicMock()
    fake_ep.name = "cached_backend"
    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock()
    fake_adapter.Cached_backendBackend = fake_adapter_class

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}) as mock_ep_func:
        b1 = get_backend("cached_backend")
        assert mock_ep_func.call_count == 1
        b2 = get_backend("cached_backend")
        assert mock_ep_func.call_count == 1
        assert b1 is b2


def test_registry_unknown_backend() -> None:
    """Test that requesting an unregistered backend name raises ValueError."""
    mod.ENTRY_POINTS.clear()
    with pytest.raises(ValueError, match="Unknown backend: definitely_not_real"):
        get_backend("definitely_not_real")
