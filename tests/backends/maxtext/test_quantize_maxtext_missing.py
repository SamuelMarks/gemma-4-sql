"""Module docstring."""

import builtins
import importlib

import gemma_4_sql.backends.maxtext.quantize as mod


def test_maxtext_quantize_import_error():
    """Docstring for test_maxtext_quantize_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "jax" or name == "maxtext":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_quantize_branches():
    # just run quantize_model and trigger all branches
    """Docstring for test_maxtext_quantize_branches."""
