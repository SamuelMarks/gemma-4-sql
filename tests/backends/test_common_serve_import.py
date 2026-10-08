"""Module docstring."""

import builtins
import importlib
from unittest.mock import patch

import pytest


def test_common_serve_import_fallback():
    """Docstring for test_common_serve_import_fallback."""
    # Test fallback inside a clean module load without messing up sys.modules permanently
    import gemma_4_sql.backends.common_serve as mod

    # We can just manually patch mod.uvicorn to None for the function call
    old_uvicorn = mod.uvicorn
    mod.uvicorn = None
    try:
        from gemma_4_sql.exceptions import DependencyMissingError

        with pytest.raises(DependencyMissingError):
            mod.serve_model_wrapper("test", "model", 8080, 1, False, "", lambda: None)
    finally:
        mod.uvicorn = old_uvicorn

    old_fastapi = mod.FastAPI
    mod.FastAPI = None
    try:
        from gemma_4_sql.exceptions import DependencyMissingError

        with pytest.raises(DependencyMissingError):
            mod.serve_model_wrapper("test", "model", 8080, 1, False, "", lambda: None)
    finally:
        mod.FastAPI = old_fastapi


def test_import_exception_logging():
    """Docstring for test_import_exception_logging."""
    # to hit lines 30-35 we do the __import__ mock
    import gemma_4_sql.backends.common_serve as mod

    old_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name in ("fastapi", "uvicorn"):
            raise ImportError(f"Mocked ImportError for {name}")
        return old_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(mod)
        assert mod.uvicorn is None
        assert mod.FastAPI is None

    # RESTORE IT!
    importlib.reload(mod)
    assert mod.uvicorn is not None
