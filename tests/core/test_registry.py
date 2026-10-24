"""Tests for the SDK backend registry module."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import clear_backend_cache, get_backend


@pytest.fixture(autouse=True)
def _clean_registry_caches() -> object:
    """Clear registry caches before and after each test in test_registry.py."""
    clear_backend_cache()
    yield
    clear_backend_cache()


def test_registry_fallback() -> None:
    """Test registry fallback behavior."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    with pytest.raises(ValueError, match="Unknown backend: nonexistent"):
        get_backend("nonexistent")


def test_registry_missing_adapter_module(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def mock_import_module(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept mock_module.adapter import."""
        if name == "mock_module.adapter":
            raise ImportError("Mock missing adapter")
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        with pytest.raises(ValueError, match=r"Backend mock_backend is missing the adapter module\."):
            get_backend("mock_backend")


def test_registry_missing_adapter_class(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def mock_import_module(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept mock_module.adapter import."""
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        with pytest.raises(ValueError, match=r"Adapter class Mock_backendBackend not found in mock_module\.adapter"):
            get_backend("mock_backend")


def test_registry_dict_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_backend when importlib.metadata.entry_points returns a dict."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    fake_ep = mock.MagicMock()
    fake_ep.name = "mock_backend"

    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock(return_value=mock.MagicMock(spec=BackendProtocol))
    fake_adapter.ADAPTER_CLASS = None
    fake_adapter.Mock_backendBackend = fake_adapter_class

    # fake importlib.import_module
    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept mock_module.adapter import."""
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}):
        backend = get_backend("mock_backend")
        assert backend is fake_adapter_class.return_value
        fake_ep.load.assert_called_once()


def test_registry_select_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_backend when importlib.metadata.entry_points has a .select() method."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()
    fake_ep = mock.MagicMock()
    fake_ep.name = "select_backend"
    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock(return_value=mock.MagicMock(spec=BackendProtocol))
    fake_adapter.ADAPTER_CLASS = fake_adapter_class
    fake_adapter.Select_backendBackend = fake_adapter_class

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept mock_module.adapter import."""
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


def test_registry_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that entry points are cached after the first lookup."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()
    fake_ep = mock.MagicMock()
    fake_ep.name = "cached_backend"
    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    fake_adapter = mock.MagicMock()
    fake_adapter_class = mock.MagicMock(return_value=mock.MagicMock(spec=BackendProtocol))
    fake_adapter.ADAPTER_CLASS = fake_adapter_class
    fake_adapter.Cached_backendBackend = fake_adapter_class

    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept mock_module.adapter import."""
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


def test_registry_invalid_protocol_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_backend raises TypeError when adapter instance does not satisfy BackendProtocol."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()
    fake_ep = mock.MagicMock()
    fake_ep.name = "invalid_proto"
    fake_module = mock.MagicMock()
    fake_module.__name__ = "mock_module"
    fake_ep.load.return_value = fake_module

    class IncompleteAdapter:
        """Adapter stub missing BackendProtocol methods."""

    fake_adapter = mock.MagicMock()
    fake_adapter.ADAPTER_CLASS = IncompleteAdapter
    fake_adapter.Invalid_protoBackend = IncompleteAdapter

    import importlib

    orig_import_module: Any = importlib.import_module

    def mock_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        """Return fake_adapter for mock_module.adapter."""
        if name == "mock_module.adapter":
            return fake_adapter
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with (
        mock.patch("importlib.metadata.entry_points", return_value={"gemma_4_sql.backends": [fake_ep]}),
        pytest.raises(TypeError, match="does not satisfy BackendProtocol"),
    ):
        get_backend("invalid_proto")
