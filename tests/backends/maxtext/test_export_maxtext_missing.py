"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.export as mod


def test_maxtext_export_import_error():
    """Docstring for test_maxtext_export_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "orbax" or name == "jax":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_export_success():
    """Docstring for test_maxtext_export_success."""
    with patch.object(mod, "jax", MagicMock()):
        with patch.object(mod, "jnp", MagicMock()):
            with patch.object(mod, "ocp", MagicMock()):
                with patch("pathlib.Path.mkdir"):
                    res = mod.export_model("model", "path", weights={})
                    assert res["status"] == "exported_with_maxtext_orbax"
