"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.etl as mod


def test_maxtext_etl_import_error():
    """Docstring for test_maxtext_etl_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "datasets" or name.startswith("grain"):
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.datasets is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_etl_get_sampler():
    """Docstring for test_maxtext_etl_get_sampler."""
    with patch.object(mod, "grain", MagicMock()):
        mod._get_sampler(10, distributed=True)
        mod._get_sampler(10, distributed=False)
