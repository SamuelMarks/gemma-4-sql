"""Module docstring."""

import importlib
import sys


def test_numpy_missing_fallback():
    """Docstring for test_numpy_missing_fallback."""
    # Save original modules
    orig_numpy = sys.modules.get("numpy")
    orig_th = sys.modules.get("gemma_4_sql.type_hints")

    if "numpy" in sys.modules:
        del sys.modules["numpy"]
    if "gemma_4_sql.type_hints" in sys.modules:
        del sys.modules["gemma_4_sql.type_hints"]

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return orig_import(name, *args, **kwargs)

    from unittest.mock import patch

    with patch("builtins.__import__", side_effect=mock_import):
        import gemma_4_sql.type_hints as th

        assert th.ndarray is object

    # Restore original modules
    if orig_numpy:
        sys.modules["numpy"] = orig_numpy
    if orig_th:
        sys.modules["gemma_4_sql.type_hints"] = orig_th
    else:
        if "gemma_4_sql.type_hints" in sys.modules:
            del sys.modules["gemma_4_sql.type_hints"]

    # Ensure it can be re-imported correctly
    importlib.reload(sys.modules.get("gemma_4_sql.type_hints", __import__("gemma_4_sql.type_hints", fromlist=[""])))
