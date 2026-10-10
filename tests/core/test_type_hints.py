"""Module docstring."""

import importlib
import sys


def test_numpy_missing_fallback(monkeypatch):
    """Docstring for test_numpy_missing_fallback."""
    orig_th = sys.modules.get("gemma_4_sql.type_hints")

    monkeypatch.setitem(sys.modules, "numpy", None)
    if "gemma_4_sql.type_hints" in sys.modules:
        monkeypatch.delitem(sys.modules, "gemma_4_sql.type_hints")

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import gemma_4_sql.type_hints as th

    assert th.ndarray is object

    monkeypatch.undo()

    # Ensure it can be re-imported correctly
    if orig_th:
        sys.modules["gemma_4_sql.type_hints"] = orig_th
        importlib.reload(sys.modules["gemma_4_sql.type_hints"])
    else:
        importlib.reload(__import__("gemma_4_sql.type_hints", fromlist=[""]))
